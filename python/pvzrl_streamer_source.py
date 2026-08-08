"""Privacy-safe, non-blocking input-source contract for Streamer Mode.

Network adapters own their I/O and publish normalized messages into the bounded
buffer below.  The training/environment thread only drains the buffer; it never
performs network work.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import re
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Protocol, Sequence, Tuple


@dataclass(frozen=True)
class StreamSourceMessage:
    """A source-neutral viewer message with raw identity removed.

    ``delivery_id`` identifies the transport delivery while ``event_id``
    identifies the underlying platform chat event.  ``viewer_hash`` must be a
    keyed one-way hash produced at the network adapter boundary.
    """

    platform: str
    delivery_id: str
    event_id: str
    viewer_hash: str
    command_text: str = field(repr=False)
    received_monotonic: float
    published_at: Optional[str] = None


class StreamCommandSource(Protocol):
    """Lifecycle and polling contract shared by mock and network sources."""

    def start(self) -> None:
        ...

    def stop(self, timeout_seconds: float = 5.0) -> bool:
        ...

    def drain_messages(self, max_items: Optional[int] = None) -> List[StreamSourceMessage]:
        ...

    def clear(self) -> int:
        ...

    def set_accepting(self, accepting: bool, *, reason: str = "phase_change") -> int:
        ...

    def get_diagnostics(self) -> Dict[str, Any]:
        ...


class BoundedStreamMessageBuffer:
    """Thread-safe bounded FIFO with an epoch-based phase gate.

    A producer snapshots the gate before normalizing an event and supplies that
    epoch when publishing.  If TRAIN/EVALUATE changed in the meantime, the
    message is discarded instead of leaking across the phase boundary.
    """

    def __init__(self, capacity: int, *, accepting: bool = True) -> None:
        parsed_capacity = int(capacity)
        if parsed_capacity <= 0:
            raise ValueError("stream message capacity must be positive")
        self.capacity = parsed_capacity
        self._messages: Deque[StreamSourceMessage] = deque()
        self._accepting = bool(accepting)
        self._gate_epoch = 0
        self._gate_reason = "startup"
        self._lock = threading.Lock()
        self._counts: Counter[str] = Counter()

    def gate_snapshot(self) -> Tuple[bool, int]:
        with self._lock:
            return bool(self._accepting), int(self._gate_epoch)

    def publish(self, message: StreamSourceMessage, *, gate_epoch: int) -> bool:
        with self._lock:
            if int(gate_epoch) != self._gate_epoch:
                self._counts["discarded_phase_stale"] += 1
                return False
            if not self._accepting:
                self._counts["discarded_not_accepting"] += 1
                return False
            if len(self._messages) >= self.capacity:
                # Preserve the already accepted FIFO ordering.  Fresh overflow
                # is observable through diagnostics rather than silently
                # evicting an earlier command.
                self._counts["discarded_queue_full"] += 1
                return False
            self._messages.append(message)
            self._counts["published"] += 1
            return True

    def drain(self, max_items: Optional[int] = None) -> List[StreamSourceMessage]:
        if max_items is not None and int(max_items) < 0:
            raise ValueError("max_items must be non-negative or None")
        with self._lock:
            limit = len(self._messages) if max_items is None else min(len(self._messages), int(max_items))
            drained = [self._messages.popleft() for _ in range(limit)]
            self._counts["drained"] += len(drained)
            return drained

    def clear(self) -> int:
        with self._lock:
            cleared = len(self._messages)
            self._messages.clear()
            self._counts["cleared"] += cleared
            return int(cleared)

    def set_accepting(self, accepting: bool, *, reason: str = "phase_change") -> int:
        with self._lock:
            cleared = len(self._messages)
            self._messages.clear()
            self._accepting = bool(accepting)
            self._gate_epoch += 1
            self._gate_reason = str(reason or "phase_change")
            self._counts["phase_changes"] += 1
            self._counts["cleared_on_phase_change"] += cleared
            return int(cleared)

    def diagnostics(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "stream_source_queue_capacity": int(self.capacity),
                "stream_source_queue_depth": int(len(self._messages)),
                "stream_source_accepting": bool(self._accepting),
                "stream_source_gate_epoch": int(self._gate_epoch),
                "stream_source_gate_reason": str(self._gate_reason),
                "stream_source_messages_published": int(self._counts["published"]),
                "stream_source_messages_drained": int(self._counts["drained"]),
                "stream_source_messages_cleared": int(self._counts["cleared"]),
                "stream_source_messages_cleared_on_phase_change": int(
                    self._counts["cleared_on_phase_change"]
                ),
                "stream_source_messages_discarded_queue_full": int(
                    self._counts["discarded_queue_full"]
                ),
                "stream_source_messages_discarded_not_accepting": int(
                    self._counts["discarded_not_accepting"]
                ),
                "stream_source_messages_discarded_phase_stale": int(
                    self._counts["discarded_phase_stale"]
                ),
                "stream_source_phase_changes": int(self._counts["phase_changes"]),
            }


_VIEWER_HASH_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_LOCAL_MESSAGE_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


@dataclass(frozen=True)
class ScriptedStreamSourceRecord:
    """One trusted local/mock input record.

    ``local_viewer_id`` is accepted only so the source can HMAC it at the
    privacy boundary.  It is hidden from repr and never copied into messages or
    diagnostics.  Callers may instead provide an already-derived 64-hex
    ``viewer_hash``.
    """

    command_text: str = field(repr=False)
    viewer_hash: str = ""
    local_viewer_id: str = field(default="", repr=False)
    delivery_id: str = ""
    event_id: str = ""
    received_monotonic: Optional[float] = None
    published_at: Optional[str] = None


class DeterministicStreamCommandSource:
    """In-memory local/mock source using the same bounded source contract.

    Initial records are one-shot inputs emitted by ``start``.  A deterministic
    simulator may also call ``submit`` at explicit wrapper steps.  This class
    deliberately accepts no file path, performs no file I/O, and creates no
    worker thread; the caller owns any trusted script loading or scheduling.
    """

    def __init__(
        self,
        records: Sequence[ScriptedStreamSourceRecord] = (),
        *,
        viewer_hash_secret: Optional[bytes] = None,
        queue_capacity: int = 256,
        accepting: bool = True,
        max_command_chars: int = 512,
        max_command_bytes: int = 2_048,
        max_script_records: int = 100_000,
        dedupe_capacity: int = 10_000,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if len(records) > int(max_script_records):
            raise ValueError("mock stream script exceeds the configured record limit")
        if any(not isinstance(record, ScriptedStreamSourceRecord) for record in records):
            raise TypeError("mock stream records must be ScriptedStreamSourceRecord values")
        if int(dedupe_capacity) <= 0:
            raise ValueError("mock stream dedupe capacity must be positive")
        secret = bytes(viewer_hash_secret or b"")
        if secret and not 16 <= len(secret) <= 4_096:
            raise ValueError("viewer hash secret must contain between 16 and 4096 bytes")
        self._script_records: Deque[ScriptedStreamSourceRecord] = deque(records)
        self._viewer_hash_secret = secret
        self._buffer = BoundedStreamMessageBuffer(queue_capacity, accepting=accepting)
        self._max_command_chars = max(1, int(max_command_chars))
        self._max_command_bytes = max(1, int(max_command_bytes))
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._running = False
        self._sequence = 0
        self._counts: Counter[str] = Counter()
        self._seen_delivery_ids: set[str] = set()
        self._seen_event_ids: set[str] = set()
        self._seen_id_pairs: Deque[Tuple[str, str]] = deque()
        self._dedupe_capacity = int(dedupe_capacity)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(script_records={len(self._script_records)}, "
            f"queue_capacity={self._buffer.capacity})"
        )

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
        self._pump_script_records(self._buffer.capacity)

    def _pump_script_records(self, maximum: int) -> None:
        """Move only one bounded window of scripted input into the source buffer."""

        for _ in range(max(0, int(maximum))):
            with self._lock:
                if not self._running or not self._script_records:
                    return
                record = self._script_records.popleft()
            self.submit(
                record.command_text,
                viewer_hash=record.viewer_hash,
                local_viewer_id=record.local_viewer_id,
                delivery_id=record.delivery_id,
                event_id=record.event_id,
                received_monotonic=record.received_monotonic,
                published_at=record.published_at,
            )

    def stop(self, timeout_seconds: float = 5.0) -> bool:
        del timeout_seconds
        with self._lock:
            self._running = False
        accepting, _ = self._buffer.gate_snapshot()
        self._buffer.set_accepting(accepting, reason="source_stopped")
        return True

    def submit(
        self,
        command_text: str,
        *,
        viewer_hash: str = "",
        local_viewer_id: str = "",
        delivery_id: str = "",
        event_id: str = "",
        received_monotonic: Optional[float] = None,
        published_at: Optional[str] = None,
    ) -> bool:
        """Normalize and enqueue one trusted local record without blocking."""

        _, gate_epoch = self._buffer.gate_snapshot()
        with self._lock:
            if not self._running:
                self._counts["discarded_not_running"] += 1
                return False
            self._sequence += 1
            sequence = self._sequence
        message = self._normalize_record(
            command_text,
            viewer_hash=viewer_hash,
            local_viewer_id=local_viewer_id,
            delivery_id=delivery_id or f"mock-delivery-{sequence}",
            event_id=event_id or f"mock-event-{sequence}",
            received_monotonic=received_monotonic,
            published_at=published_at,
        )
        if message is None:
            return False
        with self._lock:
            if message.delivery_id in self._seen_delivery_ids or message.event_id in self._seen_event_ids:
                self._counts["duplicates"] += 1
                return False
            if len(self._seen_id_pairs) >= self._dedupe_capacity:
                old_delivery_id, old_event_id = self._seen_id_pairs.popleft()
                self._seen_delivery_ids.discard(old_delivery_id)
                self._seen_event_ids.discard(old_event_id)
            self._seen_delivery_ids.add(message.delivery_id)
            self._seen_event_ids.add(message.event_id)
            self._seen_id_pairs.append((message.delivery_id, message.event_id))
        published = self._buffer.publish(message, gate_epoch=gate_epoch)
        with self._lock:
            self._counts["submitted"] += 1
            if published:
                self._counts["enqueued"] += 1
        return published

    def drain_messages(self, max_items: Optional[int] = None) -> List[StreamSourceMessage]:
        drained = self._buffer.drain(max_items=max_items)
        # Refill only after the consumer has taken a bounded batch.  Large mock
        # scripts therefore cannot monopolize startup or overflow the buffer.
        self._pump_script_records(len(drained) or self._buffer.capacity)
        return drained

    def clear(self) -> int:
        return self._buffer.clear()

    def set_accepting(self, accepting: bool, *, reason: str = "phase_change") -> int:
        return self._buffer.set_accepting(accepting, reason=reason)

    def get_diagnostics(self) -> Dict[str, Any]:
        with self._lock:
            diagnostics: Dict[str, Any] = {
                "stream_source_type": "deterministic_local",
                "stream_source_running": bool(self._running),
                "stream_source_script_records_pending": int(len(self._script_records)),
                "stream_source_local_records_submitted": int(self._counts["submitted"]),
                "stream_source_local_records_enqueued": int(self._counts["enqueued"]),
                "stream_source_local_records_duplicate": int(self._counts["duplicates"]),
                "stream_source_local_records_malformed": int(self._counts["malformed"]),
                "stream_source_local_records_discarded_not_running": int(
                    self._counts["discarded_not_running"]
                ),
            }
        diagnostics.update(self._buffer.diagnostics())
        return diagnostics

    def _normalize_record(
        self,
        command_text: str,
        *,
        viewer_hash: str,
        local_viewer_id: str,
        delivery_id: str,
        event_id: str,
        received_monotonic: Optional[float],
        published_at: Optional[str],
    ) -> Optional[StreamSourceMessage]:
        try:
            if not isinstance(command_text, str) or len(command_text) > self._max_command_chars:
                raise ValueError
            encoded_command = command_text.encode("utf-8")
            if len(encoded_command) > self._max_command_bytes:
                raise ValueError
            if not isinstance(delivery_id, str) or not _LOCAL_MESSAGE_ID_RE.fullmatch(delivery_id):
                raise ValueError
            if not isinstance(event_id, str) or not _LOCAL_MESSAGE_ID_RE.fullmatch(event_id):
                raise ValueError
            if not isinstance(viewer_hash, str):
                raise ValueError
            normalized_hash = viewer_hash.lower()
            if normalized_hash:
                if not _VIEWER_HASH_RE.fullmatch(normalized_hash):
                    raise ValueError
            else:
                if not isinstance(local_viewer_id, str) or not local_viewer_id or not self._viewer_hash_secret:
                    raise ValueError
                encoded_viewer_id = local_viewer_id.encode("utf-8")
                if len(encoded_viewer_id) > 1_024:
                    raise ValueError
                normalized_hash = hmac.new(
                    self._viewer_hash_secret,
                    encoded_viewer_id,
                    hashlib.sha256,
                ).hexdigest()
            if isinstance(received_monotonic, bool):
                raise ValueError
            received = self._monotonic() if received_monotonic is None else float(received_monotonic)
            if not math.isfinite(received) or received < 0.0:
                raise ValueError
            safe_published_at = published_at
            if safe_published_at is not None:
                if (
                    not isinstance(safe_published_at, str)
                    or len(safe_published_at) > 64
                    or any(not 33 <= ord(character) <= 126 for character in safe_published_at)
                ):
                    raise ValueError
            return StreamSourceMessage(
                platform="mock",
                delivery_id=delivery_id,
                event_id=event_id,
                viewer_hash=normalized_hash,
                command_text=command_text,
                received_monotonic=received,
                published_at=safe_published_at,
            )
        except (TypeError, ValueError, UnicodeError):
            with self._lock:
                self._counts["malformed"] += 1
            return None
