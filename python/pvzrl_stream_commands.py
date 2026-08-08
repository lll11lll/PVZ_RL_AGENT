"""Strict Streamer Mode V1 commands, queueing, and dispatch cadence.

This module is deliberately independent of Twitch and of the game runtime.
Network adapters remove raw viewer identity before producing
``StreamSourceMessage`` records.  This layer parses the bounded command text,
stores only structured commands plus keyed viewer hashes, and offers at most
one currently legal command at each monotonic dispatch opportunity.

Rows, columns, and seed slots are converted from the viewer-facing one-based
syntax to the policy's zero-based identity exactly once, during parsing.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
import time
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Deque, Iterable, Mapping, Optional, Protocol, Sequence, Tuple

from pvzrl_action_space import (
    ADVENTURE_IDENTITY_MAX_SEED_SLOTS,
    DEFAULT_COLS,
    DEFAULT_ROWS,
)
from pvzrl_fusion import FUSION_RECIPES, FUSION_RESULT_NAME_TO_ID
from pvzrl_registry import PlantRegistry, get_plant_registry, normalize_plant_name

try:
    from pvzrl_streamer_source import StreamCommandSource, StreamSourceMessage
except ImportError:  # pragma: no cover - keeps this isolated layer importable during staged rollout.
    @dataclass(frozen=True)
    class StreamSourceMessage:  # type: ignore[no-redef]
        platform: str
        delivery_id: str
        event_id: str
        viewer_hash: str
        command_text: str
        received_monotonic: float
        published_at: Optional[str] = None

    class StreamCommandSource(Protocol):  # type: ignore[no-redef]
        def start(self) -> None: ...
        def stop(self, timeout_seconds: float = 5.0) -> bool: ...
        def drain_messages(self, max_items: Optional[int] = None) -> Sequence[StreamSourceMessage]: ...
        def clear(self) -> int: ...
        def set_accepting(self, accepting: bool, *, reason: str = "phase_change") -> int: ...
        def get_diagnostics(self) -> Mapping[str, Any]: ...


DEFAULT_MAX_MESSAGE_LENGTH = 200
DEFAULT_QUEUE_CAPACITY = 256
DEFAULT_COMMAND_TTL_SECONDS = 30.0
DEFAULT_OPPORTUNITY_INTERVAL_SECONDS = 2.0
DEFAULT_MAX_POLL_MESSAGES = 16

_ALLOWED_MESSAGE = re.compile(r"^[A-Za-z0-9_! -]+$")
_VIEWER_HASH = re.compile(r"^[0-9a-f]{64}$")
_SAFE_EVENT_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,256}$")
_SAFE_PHASE_NAME = re.compile(r"^[A-Za-z0-9_.:-]{0,64}$")
_SAFE_SOURCE_METADATA_KEYS = frozenset(
    {
        "command_id",
        "event_id",
        "viewer_hash",
        "phase_generation",
        "command_kind",
        "viewer_command",
    }
)
_SAFE_VIEWER_COMMAND_KEYS = frozenset(
    {
        "kind",
        "row",
        "column",
        "plant_type_id",
        "canonical_plant_name",
        "seed_slot",
        "fusion_result_type_id",
        "canonical_fusion_result_name",
    }
)


class ViewerCommandKind(str, Enum):
    PLANT = "plant"
    SLOT = "slot"
    FUSE_RESULT = "fuse_result"
    FUSE_TILE = "fuse_tile"


class ViewerCommandParseError(ValueError):
    """A stable rejection code for untrusted chat input."""

    def __init__(self, reason: str) -> None:
        self.reason = str(reason or "invalid_command")
        super().__init__(self.reason)


@dataclass(frozen=True, slots=True)
class ViewerCommand:
    """One privacy-safe V1 command in zero-based policy coordinates."""

    kind: ViewerCommandKind
    row: int
    column: int
    plant_type_id: Optional[int] = None
    canonical_plant_name: str = ""
    seed_slot: Optional[int] = None
    fusion_result_type_id: Optional[int] = None
    canonical_fusion_result_name: str = ""

    def __post_init__(self) -> None:
        kind = self.kind if isinstance(self.kind, ViewerCommandKind) else ViewerCommandKind(str(self.kind))
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "row", int(self.row))
        object.__setattr__(self, "column", int(self.column))
        if not 0 <= self.row < DEFAULT_ROWS:
            raise ValueError("viewer command row must already be zero-based and in range")
        if not 0 <= self.column < DEFAULT_COLS:
            raise ValueError("viewer command column must already be zero-based and in range")

        if kind is ViewerCommandKind.PLANT:
            if self.plant_type_id is None or int(self.plant_type_id) < 0:
                raise ValueError("plant command requires a canonical plant type")
            if not str(self.canonical_plant_name or ""):
                raise ValueError("plant command requires a canonical plant name")
            object.__setattr__(self, "plant_type_id", int(self.plant_type_id))
        elif kind is ViewerCommandKind.SLOT:
            if self.seed_slot is None or not 0 <= int(self.seed_slot) < ADVENTURE_IDENTITY_MAX_SEED_SLOTS:
                raise ValueError("slot command requires a zero-based identity slot")
            object.__setattr__(self, "seed_slot", int(self.seed_slot))
        elif kind is ViewerCommandKind.FUSE_RESULT:
            if self.fusion_result_type_id is None or int(self.fusion_result_type_id) < 0:
                raise ValueError("fusion-result command requires a known result type")
            if not str(self.canonical_fusion_result_name or ""):
                raise ValueError("fusion-result command requires a canonical result name")
            object.__setattr__(self, "fusion_result_type_id", int(self.fusion_result_type_id))

    def to_safe_dict(self) -> dict[str, Any]:
        """Return structured metadata that contains no chat text or raw identity."""

        return {
            "kind": self.kind.value,
            "row": self.row,
            "column": self.column,
            "plant_type_id": self.plant_type_id,
            "canonical_plant_name": self.canonical_plant_name,
            "seed_slot": self.seed_slot,
            "fusion_result_type_id": self.fusion_result_type_id,
            "canonical_fusion_result_name": self.canonical_fusion_result_name,
        }


@dataclass(frozen=True, slots=True)
class ViewerCommandParseOutcome:
    accepted: bool
    reason: str
    command: Optional[ViewerCommand] = None

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "accepted": bool(self.accepted),
            "reason": self.reason,
            "command": self.command.to_safe_dict() if self.command is not None else None,
        }


_FUSION_RESULT_CANONICAL_NAMES = {
    int(recipe.result_plant_type): str(recipe.result_plant_name) for recipe in FUSION_RECIPES
}


class ViewerCommandParser:
    """Whitelist parser for the four Streamer Mode V1 command concepts."""

    def __init__(
        self,
        *,
        registry: Optional[PlantRegistry] = None,
        max_message_length: int = DEFAULT_MAX_MESSAGE_LENGTH,
    ) -> None:
        if int(max_message_length) <= 0:
            raise ValueError("max_message_length must be positive")
        self.registry = registry or get_plant_registry()
        self.max_message_length = int(max_message_length)

    def parse(self, text: Any) -> ViewerCommand:
        if not isinstance(text, str):
            raise ViewerCommandParseError("not_text")
        if len(text) > self.max_message_length:
            raise ViewerCommandParseError("message_too_long")
        if not text or not text.strip(" "):
            raise ViewerCommandParseError("empty")
        if any(ord(char) > 0x7F or ord(char) < 0x20 or ord(char) == 0x7F for char in text):
            raise ViewerCommandParseError("invalid_character")
        if _ALLOWED_MESSAGE.fullmatch(text) is None:
            raise ViewerCommandParseError("invalid_character")

        tokens = text.strip(" ").split()
        if not tokens or not tokens[0].startswith("!"):
            raise ViewerCommandParseError("not_command")
        command_name = tokens[0].lower()
        if command_name == "!plant":
            return self._parse_plant(tokens)
        if command_name == "!slot":
            return self._parse_slot(tokens)
        if command_name == "!fuse":
            return self._parse_fuse(tokens)
        raise ViewerCommandParseError("unknown_command")

    def try_parse(self, text: Any) -> ViewerCommandParseOutcome:
        try:
            return ViewerCommandParseOutcome(True, "", self.parse(text))
        except ViewerCommandParseError as exc:
            return ViewerCommandParseOutcome(False, exc.reason, None)

    @staticmethod
    def _parse_coordinate(token: str, *, axis: str, maximum: int) -> int:
        if not token.isascii() or not token.isdigit():
            raise ViewerCommandParseError(f"invalid_{axis}")
        one_based = int(token)
        if not 1 <= one_based <= int(maximum):
            raise ViewerCommandParseError(f"{axis}_out_of_range")
        return one_based - 1

    def _target(self, row_token: str, column_token: str) -> Tuple[int, int]:
        return (
            self._parse_coordinate(row_token, axis="row", maximum=DEFAULT_ROWS),
            self._parse_coordinate(column_token, axis="column", maximum=DEFAULT_COLS),
        )

    def _parse_plant(self, tokens: Sequence[str]) -> ViewerCommand:
        if len(tokens) < 4:
            raise ViewerCommandParseError("wrong_argument_count")
        row, column = self._target(tokens[-2], tokens[-1])
        supplied_name = " ".join(tokens[1:-2])
        if not supplied_name:
            raise ViewerCommandParseError("wrong_argument_count")
        plant_type = self.registry.resolve_name(supplied_name)
        if plant_type is None:
            raise ViewerCommandParseError("unknown_plant")
        return ViewerCommand(
            kind=ViewerCommandKind.PLANT,
            row=row,
            column=column,
            plant_type_id=int(plant_type),
            canonical_plant_name=self.registry.canonical_name(int(plant_type)),
        )

    def _parse_slot(self, tokens: Sequence[str]) -> ViewerCommand:
        if len(tokens) != 4:
            raise ViewerCommandParseError("wrong_argument_count")
        seed_slot = self._parse_coordinate(
            tokens[1],
            axis="slot",
            maximum=ADVENTURE_IDENTITY_MAX_SEED_SLOTS,
        )
        row, column = self._target(tokens[2], tokens[3])
        return ViewerCommand(
            kind=ViewerCommandKind.SLOT,
            row=row,
            column=column,
            seed_slot=seed_slot,
        )

    def _parse_fuse(self, tokens: Sequence[str]) -> ViewerCommand:
        if len(tokens) == 3:
            row, column = self._target(tokens[1], tokens[2])
            return ViewerCommand(kind=ViewerCommandKind.FUSE_TILE, row=row, column=column)
        if len(tokens) < 4:
            raise ViewerCommandParseError("wrong_argument_count")
        row, column = self._target(tokens[-2], tokens[-1])
        supplied_name = " ".join(tokens[1:-2])
        normalized_name = normalize_plant_name(supplied_name)
        result_type = FUSION_RESULT_NAME_TO_ID.get(normalized_name)
        if result_type is None:
            raise ViewerCommandParseError("unknown_fusion_result")
        return ViewerCommand(
            kind=ViewerCommandKind.FUSE_RESULT,
            row=row,
            column=column,
            fusion_result_type_id=int(result_type),
            canonical_fusion_result_name=_FUSION_RESULT_CANONICAL_NAMES[int(result_type)],
        )


def parse_viewer_command(
    text: Any,
    *,
    registry: Optional[PlantRegistry] = None,
    max_message_length: int = DEFAULT_MAX_MESSAGE_LENGTH,
) -> ViewerCommand:
    return ViewerCommandParser(
        registry=registry,
        max_message_length=max_message_length,
    ).parse(text)


def try_parse_viewer_command(
    text: Any,
    *,
    registry: Optional[PlantRegistry] = None,
    max_message_length: int = DEFAULT_MAX_MESSAGE_LENGTH,
) -> ViewerCommandParseOutcome:
    return ViewerCommandParser(
        registry=registry,
        max_message_length=max_message_length,
    ).try_parse(text)


@dataclass(frozen=True, slots=True)
class QueuedViewerCommand:
    command_id: str
    event_id: str
    viewer_hash: str
    command: ViewerCommand
    received_monotonic: float
    expires_monotonic: float
    phase_generation: int

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "command_id": self.command_id,
            "event_id": self.event_id,
            "viewer_hash": self.viewer_hash,
            "command": self.command.to_safe_dict(),
            "received_monotonic": self.received_monotonic,
            "expires_monotonic": self.expires_monotonic,
            "phase_generation": self.phase_generation,
        }


@dataclass(frozen=True, slots=True)
class ViewerCommandOutcome:
    status: str
    reason: str = ""
    command_id: str = ""
    event_id: str = ""
    viewer_hash: str = ""
    command: Optional[ViewerCommand] = None
    action_id: Optional[int] = None
    frame_identity: str = ""
    phase_generation: int = 0
    occurred_monotonic: float = 0.0

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "command_id": self.command_id,
            "event_id": self.event_id,
            "viewer_hash": self.viewer_hash,
            "command": self.command.to_safe_dict() if self.command is not None else None,
            "action_id": self.action_id,
            "frame_identity": self.frame_identity,
            "phase_generation": self.phase_generation,
            "occurred_monotonic": self.occurred_monotonic,
        }


@dataclass(frozen=True, slots=True)
class ViewerCommandQueueSnapshot:
    capacity: int
    depth: int
    ttl_seconds: float
    phase_generation: int
    phase_name: str
    counters: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "depth": self.depth,
            "ttl_seconds": self.ttl_seconds,
            "phase_generation": self.phase_generation,
            "phase_name": self.phase_name,
            "counters": dict(self.counters),
        }


class BoundedViewerCommandQueue:
    """A bounded, thread-safe FIFO that never evicts an accepted command."""

    def __init__(
        self,
        *,
        capacity: int = DEFAULT_QUEUE_CAPACITY,
        ttl_seconds: float = DEFAULT_COMMAND_TTL_SECONDS,
        dedupe_capacity: Optional[int] = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if int(capacity) <= 0:
            raise ValueError("capacity must be positive")
        if not math.isfinite(float(ttl_seconds)) or float(ttl_seconds) <= 0.0:
            raise ValueError("ttl_seconds must be finite and positive")
        resolved_dedupe_capacity = (
            max(1024, int(capacity) * 8) if dedupe_capacity is None else int(dedupe_capacity)
        )
        if resolved_dedupe_capacity < int(capacity):
            raise ValueError("dedupe_capacity must be at least queue capacity")
        self.capacity = int(capacity)
        self.ttl_seconds = float(ttl_seconds)
        self.dedupe_capacity = resolved_dedupe_capacity
        self._monotonic = monotonic
        self._queue: Deque[QueuedViewerCommand] = deque()
        self._seen_message_keys: OrderedDict[str, None] = OrderedDict()
        self._phase_generation = 0
        self._phase_name = "startup"
        self._sequence = 0
        self._counters: Counter[str] = Counter()
        self._lock = threading.RLock()

    @property
    def phase_generation(self) -> int:
        with self._lock:
            return int(self._phase_generation)

    def __len__(self) -> int:
        with self._lock:
            return len(self._queue)

    @staticmethod
    def _message_key(message_id: Any) -> str:
        if not isinstance(message_id, str) or _SAFE_EVENT_ID.fullmatch(message_id) is None:
            raise ValueError("message_id must be a bounded safe event identifier")
        return hashlib.sha256(message_id.encode("utf-8", errors="strict")).hexdigest()

    @staticmethod
    def _normalize_viewer_hash(viewer_hash: Any) -> str:
        normalized = str(viewer_hash or "").strip().lower()
        if _VIEWER_HASH.fullmatch(normalized) is None:
            raise ValueError("viewer_hash must be a 64-character HMAC-SHA256 hex digest")
        return normalized

    def _remember_message_key(self, key: str) -> None:
        self._seen_message_keys[key] = None
        self._seen_message_keys.move_to_end(key)
        while len(self._seen_message_keys) > self.dedupe_capacity:
            self._seen_message_keys.popitem(last=False)

    def enqueue(
        self,
        command: ViewerCommand,
        *,
        message_id: str,
        viewer_hash: str,
        received_monotonic: Optional[float] = None,
        phase_generation: Optional[int] = None,
    ) -> ViewerCommandOutcome:
        if not isinstance(command, ViewerCommand):
            raise TypeError("command must be a ViewerCommand")
        message_key = self._message_key(message_id)
        normalized_viewer_hash = self._normalize_viewer_hash(viewer_hash)
        now = float(self._monotonic() if received_monotonic is None else received_monotonic)
        if not math.isfinite(now):
            raise ValueError("received_monotonic must be finite")

        with self._lock:
            generation = self._phase_generation
            if phase_generation is not None and int(phase_generation) != generation:
                self._counters["phase_rejected"] += 1
                return ViewerCommandOutcome(
                    status="phase_rejected",
                    reason="phase_generation_mismatch",
                    event_id=message_id,
                    viewer_hash=normalized_viewer_hash,
                    command=command,
                    phase_generation=generation,
                    occurred_monotonic=now,
                )
            if message_key in self._seen_message_keys:
                self._counters["duplicate"] += 1
                return ViewerCommandOutcome(
                    status="duplicate",
                    reason="duplicate_message_id",
                    event_id=message_id,
                    viewer_hash=normalized_viewer_hash,
                    command=command,
                    phase_generation=generation,
                    occurred_monotonic=now,
                )

            # Remember even capacity-rejected deliveries so reconnect replay
            # cannot repeatedly pressure the newest side of the FIFO.
            self._remember_message_key(message_key)
            if len(self._queue) >= self.capacity:
                self._counters["capacity_rejected"] += 1
                return ViewerCommandOutcome(
                    status="capacity_rejected",
                    reason="queue_full_reject_newest",
                    event_id=message_id,
                    viewer_hash=normalized_viewer_hash,
                    command=command,
                    phase_generation=generation,
                    occurred_monotonic=now,
                )

            self._sequence += 1
            command_id = hashlib.sha256(
                f"{generation}:{self._sequence}:{message_key}".encode("ascii")
            ).hexdigest()[:24]
            queued = QueuedViewerCommand(
                command_id=command_id,
                event_id=message_id,
                viewer_hash=normalized_viewer_hash,
                command=command,
                received_monotonic=now,
                expires_monotonic=now + self.ttl_seconds,
                phase_generation=generation,
            )
            self._queue.append(queued)
            self._counters["accepted"] += 1
            return ViewerCommandOutcome(
                status="accepted",
                command_id=command_id,
                event_id=message_id,
                viewer_hash=normalized_viewer_hash,
                command=command,
                phase_generation=generation,
                occurred_monotonic=now,
            )

    def peek(self) -> Optional[QueuedViewerCommand]:
        with self._lock:
            return self._queue[0] if self._queue else None

    def pop_head(self, command_id: str, *, status: str, reason: str = "", now: Optional[float] = None) -> Optional[Tuple[QueuedViewerCommand, ViewerCommandOutcome]]:
        at = float(self._monotonic() if now is None else now)
        with self._lock:
            if not self._queue or self._queue[0].command_id != str(command_id):
                return None
            queued = self._queue.popleft()
            normalized_status = str(status or "removed")
            self._counters[normalized_status] += 1
            return queued, ViewerCommandOutcome(
                status=normalized_status,
                reason=str(reason or ""),
                command_id=queued.command_id,
                event_id=queued.event_id,
                viewer_hash=queued.viewer_hash,
                command=queued.command,
                phase_generation=queued.phase_generation,
                occurred_monotonic=at,
            )

    def record_status(self, status: str, count: int = 1) -> None:
        parsed_count = int(count)
        if parsed_count <= 0:
            return
        with self._lock:
            self._counters[str(status or "unknown")] += parsed_count

    def begin_phase(self, phase_name: str, *, clear_dedupe: bool = False) -> ViewerCommandQueueSnapshot:
        normalized_name = str(phase_name or "phase").strip()
        if _SAFE_PHASE_NAME.fullmatch(normalized_name) is None:
            raise ValueError("phase_name contains unsafe characters")
        with self._lock:
            cleared = len(self._queue)
            self._queue.clear()
            if clear_dedupe:
                self._seen_message_keys.clear()
            self._phase_generation += 1
            self._phase_name = normalized_name
            self._counters["phase_changes"] += 1
            self._counters["phase_cleared"] += cleared
            return self._snapshot_locked()

    def clear(self, *, increment_generation: bool = True) -> int:
        with self._lock:
            cleared = len(self._queue)
            self._queue.clear()
            self._counters["cleared"] += cleared
            if increment_generation:
                self._phase_generation += 1
                self._counters["phase_changes"] += 1
            return int(cleared)

    def snapshot(self) -> ViewerCommandQueueSnapshot:
        with self._lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> ViewerCommandQueueSnapshot:
        return ViewerCommandQueueSnapshot(
            capacity=self.capacity,
            depth=len(self._queue),
            ttl_seconds=self.ttl_seconds,
            phase_generation=self._phase_generation,
            phase_name=self._phase_name,
            counters=MappingProxyType(dict(sorted(self._counters.items()))),
        )


class ViewerActionResolutionLike(Protocol):
    legal: bool
    classification: str
    reason: str
    action_id: Optional[int]
    frame_identity: str


@dataclass(frozen=True, slots=True)
class ViewerCommandControllerTick:
    opportunity_opened: bool
    next_opportunity_monotonic: float
    outcomes: Tuple[ViewerCommandOutcome, ...]
    selected: Optional[QueuedViewerCommand] = None
    resolution: Optional[Any] = None
    execution_result: Optional[Any] = None

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "opportunity_opened": bool(self.opportunity_opened),
            "next_opportunity_monotonic": self.next_opportunity_monotonic,
            "outcomes": [outcome.to_safe_dict() for outcome in self.outcomes],
            "selected": self.selected.to_safe_dict() if self.selected is not None else None,
        }


class ViewerCommandController:
    """Poll a source and dispatch no more than one legal FIFO command per cadence."""

    def __init__(
        self,
        *,
        source: Optional[StreamCommandSource] = None,
        parser: Optional[ViewerCommandParser] = None,
        queue: Optional[BoundedViewerCommandQueue] = None,
        opportunity_interval_seconds: float = DEFAULT_OPPORTUNITY_INTERVAL_SECONDS,
        max_poll_messages: int = DEFAULT_MAX_POLL_MESSAGES,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not math.isfinite(float(opportunity_interval_seconds)) or float(opportunity_interval_seconds) <= 0.0:
            raise ValueError("opportunity_interval_seconds must be finite and positive")
        if int(max_poll_messages) <= 0:
            raise ValueError("max_poll_messages must be positive")
        self.source = source
        self.parser = parser or ViewerCommandParser()
        self.queue = queue if queue is not None else BoundedViewerCommandQueue(monotonic=monotonic)
        self.opportunity_interval_seconds = float(opportunity_interval_seconds)
        self.max_poll_messages = int(max_poll_messages)
        self._monotonic = monotonic
        self._next_opportunity_monotonic = float(monotonic())
        self._accepting = True
        self._closed = False
        self._source_gate_epoch: Optional[int] = None

    @property
    def next_opportunity_monotonic(self) -> float:
        return float(self._next_opportunity_monotonic)

    def start(self) -> None:
        if self.source is not None:
            self.source.start()

    def close(self, *, timeout_seconds: float = 5.0) -> bool:
        self._closed = True
        if self.source is None:
            return True
        return bool(self.source.stop(timeout_seconds=float(timeout_seconds)))

    def begin_phase(
        self,
        phase_name: str,
        *,
        accepting: bool,
        now: Optional[float] = None,
        clear_dedupe: bool = False,
    ) -> ViewerCommandQueueSnapshot:
        at = float(self._monotonic() if now is None else now)
        snapshot = self.queue.begin_phase(phase_name, clear_dedupe=clear_dedupe)
        self._accepting = bool(accepting)
        self._next_opportunity_monotonic = at
        if self.source is not None:
            self.source.set_accepting(bool(accepting), reason=str(phase_name or "phase_change"))
            try:
                diagnostics = self.source.get_diagnostics()
                epoch = diagnostics.get("stream_source_gate_epoch")
                self._source_gate_epoch = int(epoch) if epoch is not None else None
            except Exception:
                self._source_gate_epoch = None
        return snapshot

    @staticmethod
    def _safe_message_event_id(message: Any) -> str:
        event_id = getattr(message, "event_id", None)
        if event_id is None and isinstance(message, Mapping):
            event_id = message.get("event_id") or message.get("message_id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("missing_event_id")
        return event_id

    @staticmethod
    def _message_value(message: Any, name: str, default: Any = None) -> Any:
        if isinstance(message, Mapping):
            return message.get(name, default)
        return getattr(message, name, default)

    def poll_source(self, *, now: Optional[float] = None) -> Tuple[ViewerCommandOutcome, ...]:
        if self.source is None or self._closed or not self._accepting:
            return ()
        at = float(self._monotonic() if now is None else now)
        available_queue_slots = max(0, int(self.queue.capacity) - len(self.queue))
        if available_queue_slots <= 0:
            self.queue.record_status("source_poll_deferred_queue_full")
            return ()
        poll_limit = min(self.max_poll_messages, available_queue_slots)
        try:
            diagnostics = self.source.get_diagnostics()
            source_epoch_value = diagnostics.get("stream_source_gate_epoch")
            source_epoch = int(source_epoch_value) if source_epoch_value is not None else None
            if (
                source_epoch is not None
                and self._source_gate_epoch is not None
                and source_epoch != self._source_gate_epoch
            ):
                phase_name = self.queue.snapshot().phase_name
                self.queue.begin_phase(phase_name)
                self.queue.record_status("source_epoch_cleared")
            self._source_gate_epoch = source_epoch
            messages: Iterable[Any] = self.source.drain_messages(max_items=poll_limit)
        except Exception as exc:  # source failures are observable but never block the game loop.
            self.queue.record_status("source_error")
            return (
                ViewerCommandOutcome(
                    status="source_error",
                    reason=f"source_error:{type(exc).__name__}",
                    phase_generation=self.queue.phase_generation,
                    occurred_monotonic=at,
                ),
            )

        outcomes = []
        generation = self.queue.phase_generation
        for index, message in enumerate(messages):
            if index >= poll_limit:
                self.queue.record_status("poll_limit_rejected")
                break
            try:
                event_id = self._safe_message_event_id(message)
                if _SAFE_EVENT_ID.fullmatch(event_id) is None:
                    raise ValueError("unsafe_event_id")
                viewer_hash = self.queue._normalize_viewer_hash(  # source boundary is revalidated here.
                    self._message_value(message, "viewer_hash", "")
                )
                command_text = self._message_value(message, "command_text", None)
                received = float(self._message_value(message, "received_monotonic", at))
                if not math.isfinite(received) or received > at:
                    received = at
                parsed = self.parser.try_parse(command_text)
                if not parsed.accepted or parsed.command is None:
                    self.queue.record_status("parse_rejected")
                    outcomes.append(
                        ViewerCommandOutcome(
                            status="parse_rejected",
                            reason=parsed.reason,
                            event_id=event_id,
                            viewer_hash=viewer_hash,
                            phase_generation=generation,
                            occurred_monotonic=at,
                        )
                    )
                    continue
                outcomes.append(
                    self.queue.enqueue(
                        parsed.command,
                        message_id=event_id,
                        viewer_hash=viewer_hash,
                        received_monotonic=received,
                        phase_generation=generation,
                    )
                )
            except (TypeError, ValueError, UnicodeError) as exc:
                self.queue.record_status("source_message_rejected")
                reason = str(exc) if str(exc) in {"missing_event_id"} else type(exc).__name__
                outcomes.append(
                    ViewerCommandOutcome(
                        status="source_message_rejected",
                        reason=reason,
                        phase_generation=generation,
                        occurred_monotonic=at,
                    )
                )
        return tuple(outcomes)

    def tick(
        self,
        resolve_command: Callable[[ViewerCommand], ViewerActionResolutionLike],
        *,
        execute_command: Optional[Callable[[QueuedViewerCommand, ViewerActionResolutionLike], Any]] = None,
        now: Optional[float] = None,
    ) -> ViewerCommandControllerTick:
        at = float(self._monotonic() if now is None else now)
        outcomes = list(self.poll_source(now=at))
        if self._closed or not self._accepting or at < self._next_opportunity_monotonic:
            return ViewerCommandControllerTick(
                opportunity_opened=False,
                next_opportunity_monotonic=self._next_opportunity_monotonic,
                outcomes=tuple(outcomes),
            )

        # Do not catch up missed windows with a burst.  A late opportunity opens
        # one dispatch and the next starts a full interval from this moment.
        self._next_opportunity_monotonic = at + self.opportunity_interval_seconds
        while True:
            queued = self.queue.peek()
            if queued is None:
                return ViewerCommandControllerTick(
                    opportunity_opened=True,
                    next_opportunity_monotonic=self._next_opportunity_monotonic,
                    outcomes=tuple(outcomes),
                )
            if queued.phase_generation != self.queue.phase_generation:
                removed = self.queue.pop_head(
                    queued.command_id,
                    status="phase_stale",
                    reason="phase_generation_mismatch",
                    now=at,
                )
                if removed is not None:
                    outcomes.append(removed[1])
                continue
            if at >= queued.expires_monotonic:
                removed = self.queue.pop_head(
                    queued.command_id,
                    status="expired",
                    reason="command_ttl_expired",
                    now=at,
                )
                if removed is not None:
                    outcomes.append(removed[1])
                continue

            try:
                resolution = resolve_command(queued.command)
            except Exception as exc:
                removed = self.queue.pop_head(
                    queued.command_id,
                    status="unresolvable",
                    reason=f"resolver_error:{type(exc).__name__}",
                    now=at,
                )
                if removed is not None:
                    outcomes.append(removed[1])
                continue

            legal = bool(getattr(resolution, "legal", False))
            action_id = getattr(resolution, "action_id", None)
            classification = str(getattr(resolution, "classification", "unresolvable") or "unresolvable")
            reason = str(getattr(resolution, "reason", "") or "")
            frame_identity = str(getattr(resolution, "frame_identity", "") or "")
            if not legal or action_id is None:
                drop_status = classification if classification in {"currently_illegal", "unresolvable", "stale"} else "unresolvable"
                removed = self.queue.pop_head(
                    queued.command_id,
                    status=drop_status,
                    reason=reason or drop_status,
                    now=at,
                )
                if removed is not None:
                    outcome = removed[1]
                    outcomes.append(
                        ViewerCommandOutcome(
                            status=outcome.status,
                            reason=outcome.reason,
                            command_id=outcome.command_id,
                            event_id=outcome.event_id,
                            viewer_hash=outcome.viewer_hash,
                            command=outcome.command,
                            action_id=int(action_id) if action_id is not None else None,
                            frame_identity=frame_identity,
                            phase_generation=outcome.phase_generation,
                            occurred_monotonic=outcome.occurred_monotonic,
                        )
                    )
                continue

            removed = self.queue.pop_head(
                queued.command_id,
                status="selected" if execute_command is None else "dispatched",
                now=at,
            )
            if removed is None:
                continue
            selected, selection_outcome = removed
            selection_outcome = ViewerCommandOutcome(
                status=selection_outcome.status,
                command_id=selection_outcome.command_id,
                event_id=selection_outcome.event_id,
                viewer_hash=selection_outcome.viewer_hash,
                command=selection_outcome.command,
                action_id=int(action_id),
                frame_identity=frame_identity,
                phase_generation=selection_outcome.phase_generation,
                occurred_monotonic=selection_outcome.occurred_monotonic,
            )
            outcomes.append(selection_outcome)
            if execute_command is None:
                return ViewerCommandControllerTick(
                    opportunity_opened=True,
                    next_opportunity_monotonic=self._next_opportunity_monotonic,
                    outcomes=tuple(outcomes),
                    selected=selected,
                    resolution=resolution,
                )
            try:
                execution_result = execute_command(selected, resolution)
            except Exception as exc:
                self.queue.record_status("execution_error")
                outcomes.append(
                    ViewerCommandOutcome(
                        status="execution_error",
                        reason=f"execution_error:{type(exc).__name__}",
                        command_id=selected.command_id,
                        event_id=selected.event_id,
                        viewer_hash=selected.viewer_hash,
                        command=selected.command,
                        action_id=int(action_id),
                        frame_identity=frame_identity,
                        phase_generation=selected.phase_generation,
                        occurred_monotonic=at,
                    )
                )
                execution_result = None
            return ViewerCommandControllerTick(
                opportunity_opened=True,
                next_opportunity_monotonic=self._next_opportunity_monotonic,
                outcomes=tuple(outcomes),
                selected=selected,
                resolution=resolution,
                execution_result=execution_result,
            )

    def validate_selected_for_execution(
        self,
        selected: QueuedViewerCommand,
        *,
        now: Optional[float] = None,
    ) -> Optional[ViewerCommandOutcome]:
        """Recheck phase/source ownership immediately before game mutation.

        Selection and environment execution are separate operations.  A
        disconnect or phase change in that small window invalidates the popped
        command instead of allowing one last stale Twitch action.
        """

        at = float(self._monotonic() if now is None else now)
        reason = ""
        if self._closed or not self._accepting:
            reason = "controller_not_accepting"
        elif selected.phase_generation != self.queue.phase_generation:
            reason = "phase_generation_mismatch"
        elif self.source is not None:
            try:
                diagnostics = self.source.get_diagnostics()
                epoch_value = diagnostics.get("stream_source_gate_epoch")
                epoch = int(epoch_value) if epoch_value is not None else None
                if diagnostics.get("stream_source_accepting") is False:
                    reason = "source_not_accepting"
                elif (
                    epoch is not None
                    and self._source_gate_epoch is not None
                    and epoch != self._source_gate_epoch
                ):
                    reason = "source_epoch_changed"
                self._source_gate_epoch = epoch
            except Exception as exc:
                reason = f"source_error:{type(exc).__name__}"
        if not reason:
            return None
        phase_name = self.queue.snapshot().phase_name
        self.queue.begin_phase(phase_name)
        self.queue.record_status("execution_gate_rejected")
        return ViewerCommandOutcome(
            status="source_stale",
            reason=reason,
            command_id=selected.command_id,
            event_id=selected.event_id,
            viewer_hash=selected.viewer_hash,
            command=selected.command,
            phase_generation=selected.phase_generation,
            occurred_monotonic=at,
        )

    def diagnostics(self) -> dict[str, Any]:
        snapshot = self.queue.snapshot().to_safe_dict()
        return {
            "streamer_command_accepting": bool(self._accepting),
            "streamer_command_closed": bool(self._closed),
            "streamer_command_next_opportunity_monotonic": self._next_opportunity_monotonic,
            "streamer_command_opportunity_interval_seconds": self.opportunity_interval_seconds,
            "streamer_command_queue": snapshot,
        }


def safe_action_source_metadata(queued: QueuedViewerCommand) -> Mapping[str, Any]:
    """Return the only viewer metadata allowed onto action/status artifacts."""

    return MappingProxyType(
        {
            "command_id": queued.command_id,
            "event_id": queued.event_id,
            "viewer_hash": queued.viewer_hash,
            "phase_generation": queued.phase_generation,
            "command_kind": queued.command.kind.value,
            "viewer_command": queued.command.to_safe_dict(),
        }
    )


def filter_safe_action_source_metadata(metadata: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Drop arbitrary caller fields so raw text/identity cannot enter action logs."""

    if not isinstance(metadata, Mapping):
        return MappingProxyType({})
    safe: dict[str, Any] = {}
    for raw_key, value in metadata.items():
        key = str(raw_key)
        if key not in _SAFE_SOURCE_METADATA_KEYS:
            continue
        if key == "viewer_hash":
            normalized = str(value or "").strip().lower()
            if _VIEWER_HASH.fullmatch(normalized) is not None:
                safe[key] = normalized
        elif key == "event_id":
            event_id = str(value or "")
            if _SAFE_EVENT_ID.fullmatch(event_id) is not None:
                safe[key] = event_id
        elif key in {"command_id", "command_kind"}:
            text = str(value or "")
            if _SAFE_EVENT_ID.fullmatch(text) is not None:
                safe[key] = text
        elif key == "phase_generation":
            try:
                safe[key] = max(0, int(value))
            except (TypeError, ValueError, OverflowError):
                continue
        elif key == "viewer_command" and isinstance(value, Mapping):
            safe[key] = {
                str(command_key): command_value
                for command_key, command_value in value.items()
                if str(command_key) in _SAFE_VIEWER_COMMAND_KEYS
                and isinstance(command_value, (str, int, type(None)))
            }
    return MappingProxyType(safe)


# Short aliases for integration code while retaining viewer-specific names in
# status and tests.
StreamCommandController = ViewerCommandController
BoundedStreamCommandQueue = BoundedViewerCommandQueue


__all__ = [
    "BoundedStreamCommandQueue",
    "BoundedViewerCommandQueue",
    "DEFAULT_COMMAND_TTL_SECONDS",
    "DEFAULT_MAX_MESSAGE_LENGTH",
    "DEFAULT_OPPORTUNITY_INTERVAL_SECONDS",
    "DEFAULT_QUEUE_CAPACITY",
    "QueuedViewerCommand",
    "StreamCommandController",
    "ViewerCommand",
    "ViewerCommandController",
    "ViewerCommandControllerTick",
    "ViewerCommandKind",
    "ViewerCommandOutcome",
    "ViewerCommandParseError",
    "ViewerCommandParseOutcome",
    "ViewerCommandParser",
    "ViewerCommandQueueSnapshot",
    "filter_safe_action_source_metadata",
    "parse_viewer_command",
    "safe_action_source_metadata",
    "try_parse_viewer_command",
]
