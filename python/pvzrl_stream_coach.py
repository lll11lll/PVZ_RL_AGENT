"""Structured mock stream-coach helpers.

This module preserves the original lightweight payload/source helpers and adds
an additive crowd-coach stack for parser/voting/rate-limit logic.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Protocol, Sequence, Set, Tuple, Union

from pvzrl_action_space import build_action_space_spec, normalize_action_space_mode
from pvzrl_human_coach import (
    COACH_PENDING_RETRY_REASONS,
    COACH_REJECTION_PENDING_COMMAND,
    parse_coach_command,
    validate_coach_command,
)


# ---------------------------------------------------------------------------
# Existing lightweight interfaces (kept unchanged)
# ---------------------------------------------------------------------------

VALID_COACH_COMMANDS = {"plant", "fuse", "defend", "economy", "wait"}


def _safe_int(value: Any, default: int = -1) -> int:
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class StreamCoachCommand:
    command: str
    seed_index: int = -1
    row: int = -1
    col: int = -1
    vote_count: int = 0
    timestamp: float = 0.0
    active_viewers_estimate: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "command": self.command,
            "seed_index": int(self.seed_index),
            "row": int(self.row),
            "col": int(self.col),
            "vote_count": int(self.vote_count),
            "timestamp": float(self.timestamp),
        }
        if self.active_viewers_estimate is not None:
            payload["active_viewers_estimate"] = int(self.active_viewers_estimate)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class MockStreamChatRecord:
    step: int
    user: str
    message: str
    index: int
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t": int(self.step),
            "user": str(self.user),
            "message": str(self.message),
            "index": int(self.index),
            "timestamp": float(self.timestamp),
        }


@dataclass(frozen=True)
class StreamCoachSourceMessage:
    platform: str
    message_id: str
    user_id: str
    display_name: str
    text: str
    published_at: Optional[float]
    received_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class StreamCoachSource(Protocol):
    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def drain_messages(self, *, step_index: Optional[int] = None) -> List[StreamCoachSourceMessage]:
        ...

    def clear(self) -> int:
        ...

    def get_diagnostics(self) -> Dict[str, Any]:
        ...


def command_from_payload(payload: Any) -> Optional[StreamCoachCommand]:
    if not isinstance(payload, dict):
        return None
    data = payload
    raw_message = payload.get(
        "parser_command",
        payload.get("message", payload.get("raw_text", payload.get("text"))),
    )
    if raw_message:
        parsed_from_text = _stream_command_from_raw_text(
            str(raw_message),
            vote_count=_safe_int(payload.get("vote_count", payload.get("votes", 1)), default=1),
            timestamp=_safe_float(payload.get("timestamp", payload.get("t", time.time())), default=time.time()),
            active_viewers_estimate=_safe_int(
                payload.get("active_viewers_estimate", payload.get("viewer_count")),
                default=-1,
            ),
        )
        if parsed_from_text is not None:
            metadata = dict(parsed_from_text.metadata)
            if payload.get("user") is not None:
                metadata["user"] = str(payload.get("user"))
            if payload.get("t") is not None:
                metadata["script_step"] = _safe_int(payload.get("t"), default=0)
            parsed_from_text.metadata.update(metadata)
        return parsed_from_text
    if isinstance(payload.get("selected_command"), dict):
        data = payload["selected_command"]
    command_raw = str(data.get("command") or data.get("kind") or "").strip().lower()
    if command_raw.startswith("!"):
        command_raw = command_raw[1:]
    if command_raw not in VALID_COACH_COMMANDS:
        return None

    vote_count = _safe_int(
        payload.get("vote_count", payload.get("votes", data.get("vote_count", data.get("votes", 0)))),
        default=0,
    )
    timestamp = _safe_float(
        payload.get("timestamp", data.get("timestamp", time.time())),
        default=time.time(),
    )
    active_viewers = _safe_int(
        payload.get(
            "active_viewers_estimate",
            payload.get("viewer_count", data.get("active_viewers_estimate", data.get("viewer_count"))),
        ),
        default=-1,
    )
    if active_viewers < 0:
        active_viewers_opt: Optional[int] = None
    else:
        active_viewers_opt = int(active_viewers)

    seed_index = _safe_int(
        data.get("seed_index", data.get("seed_slot_index", data.get("seedSlotIndex"))),
        default=-1,
    )
    row = _safe_int(data.get("row"), default=-1)
    col = _safe_int(data.get("col", data.get("column")), default=-1)

    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    return StreamCoachCommand(
        command=command_raw,
        seed_index=seed_index,
        row=row,
        col=col,
        vote_count=max(0, int(vote_count)),
        timestamp=float(timestamp),
        active_viewers_estimate=active_viewers_opt,
        metadata=dict(metadata),
    )


def _stream_command_from_raw_text(
    raw_text: str,
    *,
    vote_count: int = 1,
    timestamp: Optional[float] = None,
    active_viewers_estimate: int = -1,
) -> Optional[StreamCoachCommand]:
    parsed = parse_coach_command(str(raw_text or ""), timestamp=timestamp, source="stream")
    if not bool(parsed.valid_syntax):
        return None
    active_viewers = int(active_viewers_estimate)
    metadata = {
        "raw_text": str(raw_text or ""),
        "coach_command_id": int(getattr(parsed, "coach_command_id", 0) or 0),
    }
    return StreamCoachCommand(
        command=str(parsed.kind or "").strip().lower(),
        seed_index=int(parsed.seed_index if parsed.seed_index is not None else -1),
        row=int(parsed.row if parsed.row is not None else -1),
        col=int(parsed.col if parsed.col is not None else -1),
        vote_count=max(1, int(vote_count or 1)),
        timestamp=float(time.time() if timestamp is None else timestamp),
        active_viewers_estimate=active_viewers if active_viewers >= 0 else None,
        metadata=metadata,
    )


class JsonlCoachCommandSource:
    """Read appended JSONL mock chat/command records.

    The legacy poll_latest() method is preserved for existing tests. The shared
    source interface emits normalized chat-like messages so mock mode uses the
    same parser/aggregator path as future network sources.
    """

    def __init__(self, path: Optional[Path], *, start_at_end: bool = True) -> None:
        self.path = path
        self._offset = path.stat().st_size if start_at_end and path is not None and path.exists() else 0
        self._started = False
        self._sequence = 0
        self._messages_emitted = 0
        self._last_error = ""
        self._last_message_id = ""
        self._last_message_text = ""
        self._last_clear_count = 0

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    def poll_latest(self) -> Optional[StreamCoachCommand]:
        latest: Optional[StreamCoachCommand] = None
        for payload in self._read_new_payloads():
            parsed = command_from_payload(payload)
            if parsed is not None:
                latest = parsed
        return latest

    def drain_messages(self, *, step_index: Optional[int] = None) -> List[StreamCoachSourceMessage]:
        del step_index
        messages: List[StreamCoachSourceMessage] = []
        for payload in self._read_new_payloads():
            messages.extend(self._payload_to_source_messages(payload))
        self._messages_emitted += len(messages)
        if messages:
            self._last_message_id = str(messages[-1].message_id)
            self._last_message_text = str(messages[-1].text)
        return messages

    def clear(self) -> int:
        return self.clear_to_end()

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "stream_source_type": "jsonl",
            "stream_source_path": str(self.path or ""),
            "stream_source_started": bool(self._started),
            "stream_source_messages_emitted": int(self._messages_emitted),
            "stream_source_last_message_id": str(self._last_message_id or ""),
            "stream_source_last_message": str(self._last_message_text or ""),
            "stream_source_last_error": str(self._last_error or ""),
            "stream_source_last_clear_count": int(self._last_clear_count),
        }

    def _read_new_payloads(self) -> List[Dict[str, Any]]:
        if self.path is None:
            return []
        try:
            if not self.path.exists():
                self._offset = 0
                return []
            size = self.path.stat().st_size
            if size < self._offset:
                self._offset = 0
            payloads: List[Dict[str, Any]] = []
            with self.path.open("r", encoding="utf-8") as handle:
                handle.seek(self._offset)
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        self._last_error = "json_decode_error"
                        continue
                    if isinstance(payload, dict):
                        payloads.append(payload)
                self._offset = handle.tell()
            self._last_error = ""
            return payloads
        except OSError as exc:
            self._last_error = str(exc)
            return []

    def clear_to_end(self) -> int:
        if self.path is None:
            return 0
        try:
            if not self.path.exists():
                self._offset = 0
                return 0
            skipped = 0
            with self.path.open("r", encoding="utf-8") as handle:
                handle.seek(self._offset)
                for raw_line in handle:
                    if raw_line.strip():
                        skipped += 1
                self._offset = handle.tell()
            self._last_clear_count = int(skipped)
            return int(skipped)
        except OSError as exc:
            self._last_error = str(exc)
            return 0

    def _payload_to_source_messages(self, payload: Dict[str, Any]) -> List[StreamCoachSourceMessage]:
        raw_message = payload.get(
            "parser_command",
            payload.get("message", payload.get("raw_text", payload.get("text"))),
        )
        base_user = str(
            payload.get("user")
            or payload.get("username")
            or payload.get("display_name")
            or payload.get("author")
            or "mock_viewer"
        ).strip() or "mock_viewer"
        received_at = time.time()
        published_at_raw = payload.get("published_at", payload.get("timestamp"))
        published_at = _safe_float(published_at_raw, default=0.0) if published_at_raw is not None else 0.0
        base_message_id = str(
            payload.get("message_id")
            or payload.get("id")
            or payload.get("event_id")
            or ""
        ).strip()

        if raw_message is not None and str(raw_message).strip():
            self._sequence += 1
            message_id = base_message_id or f"mock-jsonl:{self._sequence}"
            return [
                StreamCoachSourceMessage(
                    platform=str(payload.get("platform") or STREAM_COACH_PLATFORM_MOCK),
                    message_id=message_id,
                    user_id=str(payload.get("user_id") or base_user),
                    display_name=base_user,
                    text=str(raw_message).strip(),
                    published_at=float(published_at) if published_at > 0.0 else None,
                    received_at=float(received_at),
                    metadata={key: value for key, value in payload.items() if key not in {"parser_command", "message", "raw_text", "text"}},
                )
            ]

        stream_command = command_from_payload(payload)
        if stream_command is None:
            return []
        raw_text = _chat_text_from_stream_command(stream_command)
        vote_weight = max(1, int(stream_command.vote_count or 1))
        messages: List[StreamCoachSourceMessage] = []
        for idx in range(vote_weight):
            self._sequence += 1
            suffix = f":vote:{idx}" if vote_weight > 1 else ""
            message_id = f"{base_message_id or f'mock-jsonl:{self._sequence}'}{suffix}"
            synthetic_user = f"{base_user}_{idx}" if vote_weight > 1 else base_user
            metadata = dict(stream_command.metadata)
            metadata.update(
                {
                    "legacy_selected_command": stream_command.to_dict(),
                    "active_viewers_estimate": stream_command.active_viewers_estimate,
                    "vote_index": int(idx),
                    "vote_count": int(vote_weight),
                }
            )
            messages.append(
                StreamCoachSourceMessage(
                    platform=str(payload.get("platform") or STREAM_COACH_PLATFORM_MOCK),
                    message_id=message_id,
                    user_id=str(payload.get("user_id") or synthetic_user),
                    display_name=synthetic_user,
                    text=raw_text,
                    published_at=float(stream_command.timestamp or 0.0) if stream_command.timestamp > 0.0 else None,
                    received_at=float(received_at),
                    metadata=metadata,
                )
            )
        return messages


class MockStreamScriptSource:
    """Deterministically replay JSONL chat records by wrapper step.

    Each line is a JSON object with at least a chat message. The preferred
    schema is: {"t": 5, "user": "mock_viewer_1", "message": "!wait"}.
    `t` is interpreted as the zero-based PvZ wrapper step at which the message
    becomes visible to the normal stream parser/aggregator.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.records: List[MockStreamChatRecord] = []
        self.load_errors: List[str] = []
        self._cursor = 0
        self._emitted_count = 0
        self._started = False
        self._last_message_id = ""
        self._last_message_text = ""
        self._load()

    @property
    def loaded(self) -> bool:
        return bool(self.records) and not self.load_errors

    @property
    def emitted_count(self) -> int:
        return int(self._emitted_count)

    @property
    def total_count(self) -> int:
        return int(len(self.records))

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    def drain_messages(self, *, step_index: Optional[int] = None) -> List[StreamCoachSourceMessage]:
        messages: List[StreamCoachSourceMessage] = []
        received_at = time.time()
        for record in self.poll_due(step_index=step_index):
            message_id = f"mock-script:{self.path.name}:{int(record.index)}"
            source_message = StreamCoachSourceMessage(
                platform=STREAM_COACH_PLATFORM_MOCK,
                message_id=message_id,
                user_id=str(record.user),
                display_name=str(record.user),
                text=str(record.message),
                published_at=float(record.timestamp) if record.timestamp > 0.0 else None,
                received_at=float(received_at),
                metadata={
                    "script_path": str(self.path),
                    "script_step": int(record.step),
                    "script_index": int(record.index),
                },
            )
            messages.append(source_message)
        if messages:
            self._last_message_id = str(messages[-1].message_id)
            self._last_message_text = str(messages[-1].text)
        return messages

    def poll_due(self, *, step_index: Optional[int] = None) -> List[MockStreamChatRecord]:
        if not self.records:
            return []
        if step_index is None:
            if self._cursor >= len(self.records):
                return []
            record = self.records[self._cursor]
            self._cursor += 1
            self._emitted_count += 1
            return [record]
        due_step = max(0, int(step_index))
        due: List[MockStreamChatRecord] = []
        while self._cursor < len(self.records) and int(self.records[self._cursor].step) <= due_step:
            due.append(self.records[self._cursor])
            self._cursor += 1
        self._emitted_count += len(due)
        return due

    def pending_count(self, *, step_index: Optional[int] = None) -> int:
        del step_index
        return max(0, len(self.records) - int(self._cursor))

    def clear_to_end(self) -> int:
        # A script is an explicitly requested replay source, not a stale append
        # queue. Reset cleanup clears aggregator state but leaves future script
        # messages scheduled for this run.
        return 0

    def clear(self) -> int:
        return self.clear_to_end()

    def get_diagnostics(self) -> Dict[str, Any]:
        fields = self.diagnostics_fields()
        fields.update(
            {
                "stream_source_type": "mock_script",
                "stream_source_path": str(self.path),
                "stream_source_started": bool(self._started),
                "stream_source_messages_emitted": int(self.emitted_count),
                "stream_source_last_message_id": str(self._last_message_id or ""),
                "stream_source_last_message": str(self._last_message_text or ""),
                "stream_source_last_error": "; ".join(self.load_errors),
                "stream_source_last_clear_count": 0,
            }
        )
        return fields

    def diagnostics_fields(self, *, step_index: Optional[int] = None) -> Dict[str, Any]:
        return {
            "mock_stream_script": str(self.path),
            "mock_stream_script_loaded": bool(self.loaded),
            "mock_stream_script_total_messages": int(self.total_count),
            "mock_stream_script_emitted_messages": int(self.emitted_count),
            "mock_stream_script_pending_messages": int(self.pending_count(step_index=step_index)),
            "mock_stream_script_exhausted": bool(self._cursor >= len(self.records)) if self.records else False,
            "mock_stream_script_load_errors": list(self.load_errors),
        }

    def _load(self) -> None:
        self.records = []
        self.load_errors = []
        if not self.path.exists():
            self.load_errors.append(f"mock_stream_script_missing:{self.path}")
            return
        try:
            raw_lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            self.load_errors.append(f"mock_stream_script_read_failed:{exc}")
            return
        for line_number, raw_line in enumerate(raw_lines, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                self.load_errors.append(f"line {line_number}: json_error:{exc.msg}")
                continue
            if not isinstance(payload, dict):
                self.load_errors.append(f"line {line_number}: expected_object")
                continue
            message = str(payload.get("message", payload.get("raw_text", payload.get("text", ""))) or "").strip()
            if not message:
                self.load_errors.append(f"line {line_number}: missing_message")
                continue
            step = _safe_int(payload.get("t", payload.get("step", payload.get("at_step", 0))), default=0)
            user = str(payload.get("user", payload.get("username", f"mock_viewer_{line_number}")) or "").strip()
            timestamp = _safe_float(payload.get("timestamp", payload.get("time", 0.0)), default=0.0)
            self.records.append(
                MockStreamChatRecord(
                    step=max(0, int(step)),
                    user=user or f"mock_viewer_{line_number}",
                    message=message,
                    index=int(line_number - 1),
                    timestamp=float(timestamp),
                )
            )
        self.records.sort(key=lambda item: (int(item.step), int(item.index)))


class StreamCoachRateLimiter:
    """Simple rolling 60-second action limiter."""

    def __init__(self, max_actions_per_minute: int) -> None:
        self.max_actions_per_minute = max(0, int(max_actions_per_minute))
        self._accepted_timestamps: Deque[float] = deque()

    def allow(self, now: Optional[float] = None) -> bool:
        if self.max_actions_per_minute <= 0:
            return False
        current = float(time.time() if now is None else now)
        cutoff = current - 60.0
        while self._accepted_timestamps and self._accepted_timestamps[0] < cutoff:
            self._accepted_timestamps.popleft()
        if len(self._accepted_timestamps) >= self.max_actions_per_minute:
            return False
        self._accepted_timestamps.append(current)
        return True


# ---------------------------------------------------------------------------
# Additive crowd-coach stack (new names, no lightweight interface breakage)
# ---------------------------------------------------------------------------

STREAM_COACH_PLATFORM_MOCK = "mock"

COACH_REJECTION_UNKNOWN_COMMAND = "unknown_command"
COACH_REJECTION_MALFORMED_COMMAND = "malformed_command"
COACH_REJECTION_RATE_LIMITED = "user_rate_limited"
COACH_REJECTION_USER_SPAM = "user_spam_window"
COACH_REJECTION_BELOW_VOTE_THRESHOLD = "below_vote_threshold"
COACH_REJECTION_NO_LEGAL_COMMAND = "no_legal_command"
COACH_REJECTION_ACTION_RATE_LIMITED = "action_rate_limited"


@dataclass
class StreamCoachMessage:
    platform: str
    username_hash: str
    raw_text: str
    timestamp: float
    parsed_command: Optional[Dict[str, Any]]
    valid_syntax: bool
    rejected_reason: str = ""


@dataclass
class CrowdCoachVote:
    canonical: str
    command: Dict[str, Any]
    vote_count: int
    unique_voters: int
    first_timestamp: float


@dataclass
class CrowdCoachDecision:
    selected: bool
    fallback_to_ppo: bool
    selected_command: Optional[Dict[str, Any]] = None
    selected_vote_count: int = 0
    selected_policy_action: Optional[int] = None
    selected_bridge_command: Optional[Dict[str, Any]] = None
    selected_action_label: str = ""
    coach_match: bool = False
    override_applied: bool = False
    pending: bool = False
    rejected_reason: str = ""
    top_commands: List[Dict[str, Any]] = field(default_factory=list)


def parse_chat_command(raw_text: str) -> Tuple[Optional[Dict[str, Any]], bool, str]:
    """Parse raw chat command text through the human-coach parser."""

    parsed = parse_coach_command(str(raw_text or ""), source="stream")
    if not bool(parsed.valid_syntax):
        return None, False, str(parsed.rejected_reason or COACH_REJECTION_MALFORMED_COMMAND)
    return _command_dict_from_human(parsed), True, ""


def hash_stream_username(username: str) -> str:
    text = str(username or "").strip().lower() or "anonymous"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class _StreamCoachJsonlLogger:
    def __init__(self, path: Optional[Path]) -> None:
        self.path = Path(path) if path else None

    def log(self, event_type: str, payload: Dict[str, Any]) -> None:
        if self.path is None:
            return
        record = {"timestamp": time.time(), "event_type": str(event_type), **payload}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        except OSError:
            return


class CrowdCoachAggregator:
    """Crowd command parser, vote aggregator, validator, and selector."""

    def __init__(
        self,
        *,
        window_sec: float = 3.0,
        min_votes: int = 2,
        max_actions_per_minute: int = 20,
        user_min_interval_sec: float = 0.35,
        user_window_max_messages: int = 3,
        platform: str = STREAM_COACH_PLATFORM_MOCK,
        log_path: Optional[Path] = None,
    ) -> None:
        self.window_sec = max(0.25, float(window_sec))
        self.min_votes = max(1, int(min_votes))
        self.user_min_interval_sec = max(0.0, float(user_min_interval_sec))
        self.user_window_max_messages = max(1, int(user_window_max_messages))
        self.platform = str(platform or STREAM_COACH_PLATFORM_MOCK)
        self.action_rate_limiter = StreamCoachRateLimiter(max_actions_per_minute=max_actions_per_minute)
        self._messages: Deque[StreamCoachMessage] = deque()
        self._user_times: Dict[str, Deque[float]] = defaultdict(deque)
        self._logger = _StreamCoachJsonlLogger(log_path)

        self.stream_coach_last_command: Optional[Dict[str, Any]] = None
        self.stream_coach_last_action: Optional[Union[int, str]] = None
        self.stream_coach_last_vote_count: int = 0
        self.stream_coach_last_rejected_command: Optional[Dict[str, Any]] = None
        self.stream_coach_last_rejected_reason: str = ""
        self.stream_coach_last_error: str = ""
        self.stream_coach_top_commands: List[Dict[str, Any]] = []
        self.stream_coach_messages_seen: int = 0
        self.stream_coach_commands_parsed: int = 0
        self.stream_coach_commands_accepted: int = 0
        self.stream_coach_commands_rejected: int = 0
        self.stream_coach_decisions_accepted: int = 0
        self.stream_coach_decisions_rejected: int = 0
        self.stream_coach_validated_count: int = 0
        self.stream_coach_applied_count: int = 0
        self.stream_coach_dry_run_count: int = 0
        self.stream_coach_last_user: str = ""
        self.stream_coach_last_message: str = ""
        self.stream_coach_last_parsed_command: Optional[Dict[str, Any]] = None
        self.stream_coach_last_command_status: str = "idle"
        self.stream_coach_last_reject_reason: str = ""
        self.stream_coach_last_validated_command: str = ""
        self.stream_coach_last_applied_command: str = ""
        self.stream_coach_legal_execution_count: int = 0
        self.stream_coach_override_count: int = 0
        self.stream_coach_match_count: int = 0
        self.stream_coach_rejected_count: int = 0
        self.stream_coach_reward_total: float = 0.0
        self.stream_coach_fusion_attempt_count: int = 0
        self.stream_coach_fusion_success_count: int = 0
        self.stream_coach_fusion_failure_count: int = 0
        self.stream_coach_fusion_rejected_count: int = 0
        self.stream_coach_fusion_last_command: Optional[Dict[str, Any]] = None
        self.stream_coach_fusion_last_result: str = ""
        self.stream_coach_fusion_last_rejected_reason: str = ""
        self.stream_coach_fusion_last_execution_mode: str = ""
        self.stream_coach_fusion_last_bridge_method_used: str = ""
        self.stream_coach_fusion_last_bridge_result_reason: str = ""
        self.stream_coach_fusion_last_scope: str = ""
        self.stream_coach_fusion_last_changed_tile_count: int = 0
        self.stream_coach_fusion_last_non_source_tiles_changed: bool = False
        self.stream_coach_fusion_last_global_side_effect: bool = False
        self.stream_coach_selected_bridge_command: Optional[Dict[str, Any]] = None
        self.stream_coach_last_executed_command_id: Optional[int] = None
        self._pending_vote_command: Optional[Dict[str, Any]] = None
        self._pending_vote_count: int = 0
        self._pending_vote_first_timestamp: float = 0.0
        self._pending_vote_reason: str = ""

    def ingest_message(
        self,
        *,
        platform: str,
        username: str,
        raw_text: str,
        timestamp: Optional[float] = None,
        display_name: Optional[str] = None,
    ) -> StreamCoachMessage:
        now = float(time.time() if timestamp is None else timestamp)
        user_hash = hash_stream_username(username)
        self.stream_coach_messages_seen += 1
        self.stream_coach_last_user = str(display_name or username or "")
        self.stream_coach_last_message = str(raw_text or "")
        self._prune(now)
        rate_reason = self._rate_limit_reason(user_hash, now)
        if rate_reason:
            message = StreamCoachMessage(
                platform=str(platform or self.platform),
                username_hash=user_hash,
                raw_text=str(raw_text or ""),
                timestamp=now,
                parsed_command=None,
                valid_syntax=False,
                rejected_reason=rate_reason,
            )
            self.stream_coach_rejected_count += 1
            self.stream_coach_commands_rejected += 1
            self.stream_coach_last_rejected_command = {"raw_text": message.raw_text}
            self.stream_coach_last_rejected_reason = rate_reason
            self.stream_coach_last_reject_reason = rate_reason
            self.stream_coach_last_error = rate_reason
            self.stream_coach_last_command_status = "rejected"
            self._logger.log("stream_message_rejected", asdict(message))
            return message

        parsed, valid, reason = parse_chat_command(raw_text)
        message = StreamCoachMessage(
            platform=str(platform or self.platform),
            username_hash=user_hash,
            raw_text=str(raw_text or ""),
            timestamp=now,
            parsed_command=parsed,
            valid_syntax=bool(valid),
            rejected_reason=str(reason or ""),
        )
        if not message.valid_syntax:
            self.stream_coach_rejected_count += 1
            self.stream_coach_commands_rejected += 1
            self.stream_coach_last_rejected_command = {"raw_text": message.raw_text}
            self.stream_coach_last_rejected_reason = message.rejected_reason
            self.stream_coach_last_reject_reason = message.rejected_reason
            self.stream_coach_last_error = message.rejected_reason
            self.stream_coach_last_command_status = "rejected"
            self._logger.log("stream_message_rejected", asdict(message))
            return message

        self.stream_coach_commands_parsed += 1
        self.stream_coach_commands_accepted += 1
        self.stream_coach_last_parsed_command = dict(parsed) if isinstance(parsed, dict) else None
        self.stream_coach_last_reject_reason = ""
        self.stream_coach_last_error = ""
        self.stream_coach_last_command_status = "accepted"
        self._messages.append(message)
        self._user_times[user_hash].append(now)
        self._logger.log("stream_message_accepted", asdict(message))
        return message

    def top_votes(self, *, now: Optional[float] = None, limit: int = 5) -> List[CrowdCoachVote]:
        current = float(time.time() if now is None else now)
        self._prune(current)
        grouped: Dict[str, Dict[str, Any]] = {}
        for message in self._messages:
            if not message.valid_syntax or not isinstance(message.parsed_command, dict):
                continue
            canonical = str(message.parsed_command.get("canonical") or "")
            if not canonical:
                continue
            row = grouped.get(canonical)
            if row is None:
                grouped[canonical] = {
                    "canonical": canonical,
                    "command": dict(message.parsed_command),
                    "vote_count": 1,
                    "voters": {message.username_hash},
                    "first_timestamp": float(message.timestamp),
                }
            else:
                row["vote_count"] += 1
                row["voters"].add(message.username_hash)
                row["first_timestamp"] = min(float(row["first_timestamp"]), float(message.timestamp))
        votes = [
            CrowdCoachVote(
                canonical=str(row["canonical"]),
                command=dict(row["command"]),
                vote_count=int(row["vote_count"]),
                unique_voters=int(len(row["voters"])),
                first_timestamp=float(row["first_timestamp"]),
            )
            for row in grouped.values()
        ]
        votes.sort(key=lambda item: (-item.vote_count, item.first_timestamp, item.canonical))
        return votes[: max(1, int(limit))]

    def choose_highest_voted_legal_command(
        self,
        *,
        observation: Optional[Dict[str, Any]],
        legal_actions: Optional[Sequence[int]],
        action_space_mode: str,
        ppo_action: Optional[int],
        fusion_enabled: bool = False,
        fusion_bridge_probe: Optional[
            Callable[[Any, Dict[str, Any], Sequence[int], Any], Tuple[Optional[Dict[str, Any]], str, Dict[str, Any]]]
        ] = None,
        now: Optional[float] = None,
    ) -> CrowdCoachDecision:
        current = float(time.time() if now is None else now)
        votes = self.top_votes(now=current, limit=5)
        self.stream_coach_top_commands = [self._vote_to_status_row(vote) for vote in votes]

        candidates: List[Tuple[CrowdCoachVote, bool]] = [(vote, False) for vote in votes]
        pending_candidate = self._pending_vote_candidate()
        if pending_candidate is not None:
            pending_canonical = str(pending_candidate.canonical)
            if not any(str(vote.canonical) == pending_canonical for vote in votes):
                candidates.append((pending_candidate, True))

        if not candidates:
            decision = CrowdCoachDecision(
                selected=False,
                fallback_to_ppo=True,
                rejected_reason=COACH_REJECTION_NO_LEGAL_COMMAND,
                top_commands=list(self.stream_coach_top_commands),
            )
            self.stream_coach_last_command_status = (
                self.stream_coach_last_command_status if self.stream_coach_last_command_status != "idle" else "idle"
            )
            self._logger.log("stream_window_decision", asdict(decision))
            return decision

        selected_vote: Optional[CrowdCoachVote] = None
        selected_action: Optional[int] = None
        selected_bridge_command: Optional[Dict[str, Any]] = None
        selected_action_label = ""
        selected_rejection = COACH_REJECTION_NO_LEGAL_COMMAND
        selected_pending_wait = False
        selected_pending_reason = ""
        for vote, from_pending_cache in candidates:
            if not from_pending_cache and int(vote.vote_count) < self.min_votes:
                selected_rejection = COACH_REJECTION_BELOW_VOTE_THRESHOLD
                continue
            action, rejection, bridge_command, action_label = self._validate_vote_command(
                vote.command,
                observation=observation or {},
                legal_actions=legal_actions,
                action_space_mode=action_space_mode,
                fusion_enabled=fusion_enabled,
                fusion_bridge_probe=fusion_bridge_probe,
            )
            if action is not None:
                selected_vote = vote
                selected_action = int(action)
                selected_bridge_command = dict(bridge_command) if isinstance(bridge_command, dict) else None
                selected_action_label = str(action_label or f"policy_action:{int(action)}")
                selected_pending_wait = False
                selected_pending_reason = ""
                self._clear_pending_vote()
                break
            if self._is_pending_retry_rejection(vote.command, rejection):
                selected_vote = vote
                selected_action = int(self._wait_action_for_state(observation=observation or {}, action_space_mode=action_space_mode))
                selected_bridge_command = None
                selected_pending_wait = True
                selected_pending_reason = str(rejection or "illegal_action")
                selected_action_label = f"pending_wait:{selected_pending_reason}"
                self._remember_pending_vote(vote, reason=selected_pending_reason)
                break
            if rejection:
                selected_rejection = str(rejection)

        if selected_vote is None or selected_action is None:
            self._clear_pending_vote()
            self.stream_coach_rejected_count += 1
            self.stream_coach_decisions_rejected += 1
            self.stream_coach_last_rejected_reason = selected_rejection
            self.stream_coach_last_reject_reason = selected_rejection
            self.stream_coach_last_error = selected_rejection
            self.stream_coach_last_command_status = "rejected"
            self.stream_coach_last_rejected_command = candidates[0][0].command if candidates else None
            decision = CrowdCoachDecision(
                selected=False,
                fallback_to_ppo=True,
                rejected_reason=selected_rejection,
                top_commands=list(self.stream_coach_top_commands),
            )
            self._logger.log("stream_window_decision", asdict(decision))
            return decision

        if not selected_pending_wait and not self.action_rate_limiter.allow(now=current):
            self.stream_coach_rejected_count += 1
            self.stream_coach_decisions_rejected += 1
            self.stream_coach_last_rejected_reason = COACH_REJECTION_ACTION_RATE_LIMITED
            self.stream_coach_last_reject_reason = COACH_REJECTION_ACTION_RATE_LIMITED
            self.stream_coach_last_error = COACH_REJECTION_ACTION_RATE_LIMITED
            self.stream_coach_last_command_status = "rejected"
            self.stream_coach_last_rejected_command = dict(selected_vote.command)
            decision = CrowdCoachDecision(
                selected=False,
                fallback_to_ppo=True,
                rejected_reason=COACH_REJECTION_ACTION_RATE_LIMITED,
                top_commands=list(self.stream_coach_top_commands),
            )
            self._logger.log("stream_window_decision", asdict(decision))
            return decision

        coach_match = ppo_action is not None and int(ppo_action) == int(selected_action)
        override_applied = not coach_match
        if coach_match and not selected_pending_wait:
            self.stream_coach_match_count += 1
        if override_applied and not selected_pending_wait:
            self.stream_coach_override_count += 1
        if not selected_pending_wait:
            self.stream_coach_legal_execution_count += 1
            self.stream_coach_decisions_accepted += 1
            self.stream_coach_validated_count += 1
        self.stream_coach_last_command = dict(selected_vote.command)
        self.stream_coach_last_action = int(selected_action)
        self.stream_coach_last_vote_count = int(selected_vote.vote_count)
        self.stream_coach_last_rejected_reason = (
            f"{COACH_REJECTION_PENDING_COMMAND}:{selected_pending_reason}" if selected_pending_wait else ""
        )
        self.stream_coach_last_reject_reason = str(selected_pending_reason if selected_pending_wait else "")
        self.stream_coach_last_error = self.stream_coach_last_rejected_reason
        self.stream_coach_last_command_status = "pending" if selected_pending_wait else "validated"
        self.stream_coach_last_validated_command = _chat_text_from_payload(selected_vote.command)
        self.stream_coach_selected_bridge_command = (
            dict(selected_bridge_command) if isinstance(selected_bridge_command, dict) else None
        )
        if isinstance(selected_bridge_command, dict):
            self.stream_coach_fusion_last_command = dict(selected_vote.command)
            self.stream_coach_fusion_last_result = "pending"

        decision = CrowdCoachDecision(
            selected=True,
            fallback_to_ppo=False,
            selected_command=dict(selected_vote.command),
            selected_vote_count=int(selected_vote.vote_count),
            selected_policy_action=int(selected_action),
            selected_bridge_command=selected_bridge_command,
            selected_action_label=selected_action_label or f"policy_action:{int(selected_action)}",
            coach_match=bool(coach_match),
            override_applied=bool(override_applied),
            pending=bool(selected_pending_wait),
            rejected_reason=str(selected_pending_reason if selected_pending_wait else ""),
            top_commands=list(self.stream_coach_top_commands),
        )
        self._logger.log("stream_window_decision", asdict(decision))
        return decision

    def add_reward(self, delta: float) -> None:
        self.stream_coach_reward_total += float(delta)

    def record_dry_run_decision(self, decision: CrowdCoachDecision) -> None:
        if not bool(getattr(decision, "selected", False)) or bool(getattr(decision, "pending", False)):
            return
        self.stream_coach_dry_run_count += 1
        self.stream_coach_last_command_status = "dry_run"
        selected_command = getattr(decision, "selected_command", None)
        if isinstance(selected_command, dict):
            self.stream_coach_last_validated_command = _chat_text_from_payload(selected_command)

    def record_applied_decision(self, decision: CrowdCoachDecision) -> None:
        if not bool(getattr(decision, "selected", False)) or bool(getattr(decision, "pending", False)):
            return
        selected_command = getattr(decision, "selected_command", None)
        self.stream_coach_applied_count += 1
        self.stream_coach_last_command_status = "applied"
        if isinstance(selected_command, dict):
            applied_text = _chat_text_from_payload(selected_command)
            self.stream_coach_last_validated_command = applied_text
            self.stream_coach_last_applied_command = applied_text

    def diagnostics_fields(
        self,
        *,
        enabled: bool,
        platform: Optional[str] = None,
        active_viewers_estimate: Optional[int] = None,
    ) -> Dict[str, Any]:
        return {
            "stream_coach_enabled": bool(enabled),
            "stream_coach_platform": str(platform or self.platform),
            "stream_coach_active_viewers_estimate": active_viewers_estimate,
            "stream_coach_messages_seen": int(self.stream_coach_messages_seen),
            "stream_coach_commands_parsed": int(self.stream_coach_commands_parsed),
            "stream_coach_commands_accepted": int(self.stream_coach_commands_accepted),
            "stream_coach_commands_rejected": int(self.stream_coach_commands_rejected),
            "stream_coach_decisions_accepted": int(self.stream_coach_decisions_accepted),
            "stream_coach_decisions_rejected": int(self.stream_coach_decisions_rejected),
            "stream_coach_validated_count": int(self.stream_coach_validated_count),
            "stream_coach_applied_count": int(self.stream_coach_applied_count),
            "stream_coach_dry_run_count": int(self.stream_coach_dry_run_count),
            "stream_coach_last_user": str(self.stream_coach_last_user or ""),
            "stream_coach_last_message": str(self.stream_coach_last_message or ""),
            "stream_coach_last_parsed_command": self.stream_coach_last_parsed_command,
            "stream_coach_last_command_status": str(self.stream_coach_last_command_status or ""),
            "stream_coach_last_reject_reason": str(self.stream_coach_last_reject_reason or ""),
            "stream_coach_last_validated_command": str(self.stream_coach_last_validated_command or ""),
            "stream_coach_last_applied_command": str(self.stream_coach_last_applied_command or ""),
            "last_stream_user": str(self.stream_coach_last_user or ""),
            "last_validated_coach_command": str(self.stream_coach_last_validated_command or ""),
            "stream_coach_active_window_message_count": int(len(self._messages)),
            "pending_stream_commands": int(1 if self._pending_vote_command is not None else 0),
            "stream_coach_last_command": self.stream_coach_last_command,
            "stream_coach_last_action": self.stream_coach_last_action,
            "stream_coach_last_vote_count": int(self.stream_coach_last_vote_count),
            "stream_coach_override_count": int(self.stream_coach_override_count),
            "stream_coach_match_count": int(self.stream_coach_match_count),
            "stream_coach_rejected_count": int(self.stream_coach_rejected_count),
            "stream_coach_legal_execution_count": int(self.stream_coach_legal_execution_count),
            "stream_coach_reward_total": float(self.stream_coach_reward_total),
            "stream_coach_top_commands": list(self.stream_coach_top_commands),
            "stream_coach_last_rejected_command": self.stream_coach_last_rejected_command,
            "stream_coach_last_rejected_reason": str(self.stream_coach_last_rejected_reason or ""),
            "stream_coach_last_error": str(self.stream_coach_last_error or ""),
            "selected_bridge_command": self.stream_coach_selected_bridge_command,
            "last_executed_coach_command_id": self.stream_coach_last_executed_command_id,
            "stream_coach_fusion_attempt_count": int(self.stream_coach_fusion_attempt_count),
            "stream_coach_fusion_success_count": int(self.stream_coach_fusion_success_count),
            "stream_coach_fusion_failure_count": int(self.stream_coach_fusion_failure_count),
            "stream_coach_fusion_rejected_count": int(self.stream_coach_fusion_rejected_count),
            "stream_fusion_last_command": self.stream_coach_fusion_last_command,
            "stream_fusion_last_result": str(self.stream_coach_fusion_last_result or ""),
            "stream_fusion_last_rejected_reason": str(self.stream_coach_fusion_last_rejected_reason or ""),
            "stream_fusion_last_execution_mode": str(self.stream_coach_fusion_last_execution_mode or ""),
            "stream_fusion_last_bridge_method_used": str(self.stream_coach_fusion_last_bridge_method_used or ""),
            "stream_fusion_last_bridge_result_reason": str(self.stream_coach_fusion_last_bridge_result_reason or ""),
            "stream_fusion_last_scope": str(self.stream_coach_fusion_last_scope or ""),
            "stream_fusion_last_changed_tile_count": int(self.stream_coach_fusion_last_changed_tile_count),
            "stream_fusion_last_non_source_tiles_changed": bool(self.stream_coach_fusion_last_non_source_tiles_changed),
            "stream_fusion_last_global_side_effect": bool(self.stream_coach_fusion_last_global_side_effect),
        }

    def _validate_vote_command(
        self,
        command_payload: Dict[str, Any],
        *,
        observation: Dict[str, Any],
        legal_actions: Optional[Sequence[int]],
        action_space_mode: str,
        fusion_enabled: bool,
        fusion_bridge_probe: Optional[
            Callable[[Any, Dict[str, Any], Sequence[int], Any], Tuple[Optional[Dict[str, Any]], str, Dict[str, Any]]]
        ],
    ) -> Tuple[Optional[int], str, Optional[Dict[str, Any]], str]:
        raw_text = _chat_text_from_payload(command_payload)
        parsed = parse_coach_command(raw_text, source="stream")
        if not parsed.valid_syntax:
            return None, str(parsed.rejected_reason or COACH_REJECTION_MALFORMED_COMMAND), None, ""

        mode = normalize_action_space_mode(action_space_mode)
        rows = max(1, _safe_int(observation.get("rowCount"), default=5))
        cols = max(1, _safe_int(observation.get("columnCount"), default=10))
        plant_types = _plant_types_from_observation(observation)
        max_seed_slots = _infer_max_seed_slots(observation, mode, plant_types)
        action_mask = _mask_from_legal_actions(
            legal_actions=legal_actions,
            mode=mode,
            rows=rows,
            cols=cols,
            plant_types=plant_types,
            max_seed_slots=max_seed_slots,
        )
        observation_for_validation = dict(observation)
        if legal_actions is not None:
            observation_for_validation["legalActions"] = list(legal_actions)
        validation = validate_coach_command(
            parsed,
            action_space_mode=mode,
            observation=observation_for_validation,
            action_mask=action_mask,
            plant_types=plant_types,
            max_seed_slots=max_seed_slots,
            rows=rows,
            cols=cols,
            fusion_enabled=bool(fusion_enabled),
            fusion_bridge_probe=fusion_bridge_probe,
        )
        if not validation.legal or validation.policy_action is None:
            return None, str(validation.rejected_reason or "illegal_action"), None, ""
        bridge_command = dict(validation.bridge_command) if isinstance(validation.bridge_command, dict) else None
        if bridge_command is not None:
            bridge_command["coach_command_id"] = int(getattr(parsed, "coach_command_id", 0) or 0)
            bridge_command["coach_command_timestamp"] = float(getattr(parsed, "timestamp", 0.0) or 0.0)
            bridge_command["coach_command_source"] = str(getattr(parsed, "source", "") or "stream")
            bridge_command["executed_from_fresh_coach_command"] = True
        action_label = "fusion_step" if bridge_command else f"policy_action:{int(validation.policy_action)}"
        return int(validation.policy_action), "", bridge_command, action_label

    def _prune(self, now: float) -> None:
        cutoff = float(now) - float(self.window_sec)
        while self._messages and float(self._messages[0].timestamp) < cutoff:
            self._messages.popleft()
        for user_hash, times in list(self._user_times.items()):
            while times and float(times[0]) < cutoff:
                times.popleft()
            if not times:
                self._user_times.pop(user_hash, None)

    def _rate_limit_reason(self, user_hash: str, now: float) -> str:
        entries = self._user_times[user_hash]
        cutoff = float(now) - float(self.window_sec)
        while entries and float(entries[0]) < cutoff:
            entries.popleft()
        if entries and (float(now) - float(entries[-1])) < float(self.user_min_interval_sec):
            return COACH_REJECTION_RATE_LIMITED
        if len(entries) >= int(self.user_window_max_messages):
            return COACH_REJECTION_USER_SPAM
        return ""

    def _vote_to_status_row(self, vote: CrowdCoachVote) -> Dict[str, Any]:
        return {
            "canonical": str(vote.canonical),
            "command": dict(vote.command),
            "votes": int(vote.vote_count),
            "unique_voters": int(vote.unique_voters),
            "first_timestamp": float(vote.first_timestamp),
        }

    def _pending_vote_candidate(self) -> Optional[CrowdCoachVote]:
        if not isinstance(self._pending_vote_command, dict):
            return None
        return CrowdCoachVote(
            canonical=_canonical_command(self._pending_vote_command),
            command=dict(self._pending_vote_command),
            vote_count=max(1, int(self._pending_vote_count)),
            unique_voters=max(1, int(self._pending_vote_count)),
            first_timestamp=float(self._pending_vote_first_timestamp or 0.0),
        )

    def _remember_pending_vote(self, vote: CrowdCoachVote, *, reason: str) -> None:
        self._pending_vote_command = dict(vote.command)
        self._pending_vote_count = max(1, int(vote.vote_count))
        self._pending_vote_first_timestamp = float(vote.first_timestamp)
        self._pending_vote_reason = str(reason or "")

    def _clear_pending_vote(self) -> None:
        self._pending_vote_command = None
        self._pending_vote_count = 0
        self._pending_vote_first_timestamp = 0.0
        self._pending_vote_reason = ""

    def clear_pending_state(self) -> bool:
        stale_detected = bool(self._messages or self._pending_vote_command)
        self._messages.clear()
        self._user_times.clear()
        self.stream_coach_top_commands = []
        self.stream_coach_selected_bridge_command = None
        self.stream_coach_last_command_status = "idle"
        self.stream_coach_last_reject_reason = ""
        self.stream_coach_last_validated_command = ""
        self.stream_coach_last_applied_command = ""
        self.stream_coach_last_command = None
        self.stream_coach_last_action = None
        self.stream_coach_last_vote_count = 0
        self._clear_pending_vote()
        return bool(stale_detected)

    def _is_pending_retry_rejection(self, command_payload: Dict[str, Any], rejection: str) -> bool:
        kind = str(command_payload.get("command") or "").strip().lower()
        if kind not in {"plant", "fuse"}:
            return False
        return str(rejection or "").strip().lower() in COACH_PENDING_RETRY_REASONS

    def _wait_action_for_state(self, *, observation: Dict[str, Any], action_space_mode: str) -> int:
        mode = normalize_action_space_mode(action_space_mode)
        rows = max(1, _safe_int(observation.get("rowCount"), default=5))
        cols = max(1, _safe_int(observation.get("columnCount"), default=10))
        plant_types = _plant_types_from_observation(observation)
        max_seed_slots = _infer_max_seed_slots(observation, mode, plant_types)
        spec = build_action_space_spec(
            mode=mode,
            plant_types=[int(value) for value in plant_types],
            max_seed_slots=max_seed_slots,
            rows=int(rows),
            cols=int(cols),
        )
        return int(spec.wait_action)


class StreamCoachController:
    """Mock/local controller that feeds local JSONL commands into aggregation."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        mode: str = STREAM_COACH_PLATFORM_MOCK,
        platform: str = STREAM_COACH_PLATFORM_MOCK,
        source: Optional[StreamCoachSource] = None,
        command_path: Optional[Path] = None,
        mock_script_path: Optional[Path] = None,
        dry_run: bool = True,
        apply_enabled: bool = False,
        window_sec: float = 3.0,
        min_votes: int = 2,
        max_actions_per_minute: int = 20,
        user_min_interval_sec: float = 0.35,
        user_window_max_messages: int = 3,
        log_path: Optional[Path] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.mode = str(mode or platform or STREAM_COACH_PLATFORM_MOCK).strip().lower()
        self.platform = str(platform or STREAM_COACH_PLATFORM_MOCK)
        self.mock_script_path = Path(mock_script_path) if mock_script_path else None
        self.mock_script_source = MockStreamScriptSource(self.mock_script_path) if self.mock_script_path else None
        self.command_path = Path(command_path) if command_path else None
        self.source: StreamCoachSource
        if source is not None:
            self.source = source
        elif self.mock_script_source is not None:
            self.source = self.mock_script_source
        else:
            self.source = JsonlCoachCommandSource(command_path)
        self.dry_run = bool(dry_run) or not bool(apply_enabled)
        self.apply_enabled = bool(apply_enabled) and not bool(self.dry_run)
        self.aggregator = CrowdCoachAggregator(
            window_sec=window_sec,
            min_votes=min_votes,
            max_actions_per_minute=max_actions_per_minute,
            user_min_interval_sec=user_min_interval_sec,
            user_window_max_messages=user_window_max_messages,
            platform=self.platform,
            log_path=log_path,
        )
        self._last_active_viewers_estimate: Optional[int] = None
        self._last_poll_step_index: Optional[int] = None
        self._last_poll_time: float = time.time()
        self._startup_stale_clear_performed = False
        self._stale_messages_cleared = 0
        self._clear_count = 0
        self._last_clear_reason = ""
        start_fn = getattr(self.source, "start", None)
        if callable(start_fn):
            start_fn()

    def poll_source(
        self,
        *,
        username: str = "mock_source",
        step_index: Optional[int] = None,
    ) -> List[StreamCoachMessage]:
        if not self.enabled:
            return []
        self._last_poll_step_index = None if step_index is None else int(step_index)
        self._last_poll_time = time.time()
        messages: List[StreamCoachMessage] = []
        drain_fn = getattr(self.source, "drain_messages", None)
        if not callable(drain_fn):
            return messages
        for source_message in drain_fn(step_index=step_index):
            if isinstance(source_message.metadata, dict):
                active_viewers = source_message.metadata.get("active_viewers_estimate")
                if active_viewers is not None:
                    parsed_viewers = _safe_int(active_viewers, default=-1)
                    self._last_active_viewers_estimate = parsed_viewers if parsed_viewers >= 0 else None
            source_username = str(source_message.user_id or source_message.display_name or username)
            source_timestamp = (
                float(source_message.published_at)
                if source_message.published_at is not None
                else float(source_message.received_at or time.time())
            )
            message = self.aggregator.ingest_message(
                platform=source_message.platform or self.platform,
                username=source_username,
                display_name=str(source_message.display_name or source_username),
                raw_text=source_message.text,
                timestamp=source_timestamp,
            )
            messages.append(message)
        return messages

    def poll_mock_source(
        self,
        *,
        username: str = "mock_source",
        step_index: Optional[int] = None,
    ) -> List[StreamCoachMessage]:
        return self.poll_source(username=username, step_index=step_index)

    def choose_action(
        self,
        *,
        observation: Optional[Dict[str, Any]],
        legal_actions: Optional[Sequence[int]],
        action_space_mode: str,
        ppo_action: Optional[int],
        fusion_enabled: bool = False,
        fusion_bridge_probe: Optional[
            Callable[[Any, Dict[str, Any], Sequence[int], Any], Tuple[Optional[Dict[str, Any]], str, Dict[str, Any]]]
        ] = None,
        now: Optional[float] = None,
    ) -> CrowdCoachDecision:
        if not self.enabled:
            return CrowdCoachDecision(selected=False, fallback_to_ppo=True, rejected_reason="disabled")
        return self.aggregator.choose_highest_voted_legal_command(
            observation=observation,
            legal_actions=legal_actions,
            action_space_mode=action_space_mode,
            ppo_action=ppo_action,
            fusion_enabled=fusion_enabled,
            fusion_bridge_probe=fusion_bridge_probe,
            now=now,
        )

    def apply_step_outcome(self, decision: CrowdCoachDecision, info: Dict[str, Any]) -> None:
        if not bool(getattr(decision, "selected", False)) or bool(getattr(decision, "pending", False)):
            return
        self.aggregator.record_applied_decision(decision)
        selected_command = getattr(decision, "selected_command", None)
        selected_bridge_command = getattr(decision, "selected_bridge_command", None)
        if isinstance(selected_bridge_command, dict):
            self.aggregator.stream_coach_selected_bridge_command = None
        command_id = None
        if isinstance(selected_bridge_command, dict):
            command_id = _safe_int(selected_bridge_command.get("coach_command_id"), default=0)
        if command_id and command_id > 0:
            self.aggregator.stream_coach_last_executed_command_id = int(command_id)
        if not isinstance(selected_bridge_command, dict):
            return
        self.aggregator.stream_coach_fusion_attempt_count += 1
        self.aggregator.stream_coach_fusion_last_command = dict(selected_command) if isinstance(selected_command, dict) else None
        action_result = info.get("action_result") if isinstance(info, dict) else {}
        if not isinstance(action_result, dict):
            action_result = {}
        placement = action_result.get("placement") if isinstance(action_result.get("placement"), dict) else {}
        fusion_succeeded = bool(
            action_result.get("fusionSucceeded")
            if "fusionSucceeded" in action_result
            else placement.get("fusionSucceeded", placement.get("success", False))
        )
        bridge_reason = str(
            action_result.get("bridgeResultReason")
            or placement.get("bridgeResultReason")
            or action_result.get("illegalReason")
            or placement.get("illegalReason")
            or ""
        )
        self.aggregator.stream_coach_fusion_last_result = "success" if fusion_succeeded else "failed"
        self.aggregator.stream_coach_fusion_last_rejected_reason = "" if fusion_succeeded else bridge_reason
        self.aggregator.stream_coach_fusion_last_execution_mode = str(
            action_result.get("fusionExecutionMode")
            or placement.get("fusionExecutionMode")
            or ""
        )
        self.aggregator.stream_coach_fusion_last_bridge_method_used = str(
            action_result.get("bridgeMethodUsed")
            or placement.get("bridgeMethodUsed")
            or ""
        )
        self.aggregator.stream_coach_fusion_last_bridge_result_reason = bridge_reason
        self.aggregator.stream_coach_fusion_last_scope = str(
            action_result.get("fusionScope")
            or placement.get("fusionScope")
            or ""
        )
        changed_tile_count = _safe_int(action_result.get("changedTileCount"), default=-1)
        if changed_tile_count < 0:
            changed_tile_count = _safe_int(placement.get("changedTileCount"), default=0)
        self.aggregator.stream_coach_fusion_last_changed_tile_count = int(changed_tile_count)
        self.aggregator.stream_coach_fusion_last_non_source_tiles_changed = bool(
            action_result.get("nonSourceTilesChanged")
            or placement.get("nonSourceTilesChanged")
        )
        self.aggregator.stream_coach_fusion_last_global_side_effect = bool(
            action_result.get("globalFusionSideEffect")
            or placement.get("globalFusionSideEffect")
        )
        if fusion_succeeded:
            self.aggregator.stream_coach_fusion_success_count += 1
        else:
            self.aggregator.stream_coach_fusion_failure_count += 1
            self.aggregator.stream_coach_fusion_rejected_count += 1
            self.aggregator.stream_coach_last_error = bridge_reason

    def record_dry_run_decision(self, decision: CrowdCoachDecision) -> None:
        self.aggregator.record_dry_run_decision(decision)

    def diagnostics_fields(self) -> Dict[str, Any]:
        fields = self.aggregator.diagnostics_fields(
            enabled=bool(self.enabled),
            platform=self.platform,
            active_viewers_estimate=self._last_active_viewers_estimate,
        )
        source_pending = 0
        if self.mock_script_source is not None:
            source_pending = self.mock_script_source.pending_count(step_index=self._last_poll_step_index)
        diag_fn = getattr(self.source, "get_diagnostics", None)
        if callable(diag_fn):
            fields.update(diag_fn())
        elif self.mock_script_source is not None:
            fields.update(self.mock_script_source.diagnostics_fields(step_index=self._last_poll_step_index))
        fields["stream_coach_mode"] = str(self.mode or self.platform or STREAM_COACH_PLATFORM_MOCK)
        fields["stream_coach_command_path"] = str(self.command_path or "")
        fields["stream_coach_alive"] = bool(self.enabled and self._source_alive())
        fields["stream_coach_alive_status"] = self._alive_status()
        fields["stream_coach_dry_run"] = bool(self.dry_run)
        fields["stream_coach_apply_enabled"] = bool(self.apply_enabled)
        fields["stream_coach_last_poll_step"] = self._last_poll_step_index
        fields["stream_coach_last_poll_age_seconds"] = max(0.0, time.time() - float(self._last_poll_time))
        fields["stream_coach_startup_stale_cleared"] = bool(self._startup_stale_clear_performed)
        fields["stream_coach_stale_messages_cleared"] = int(self._stale_messages_cleared)
        fields["stream_coach_clear_count"] = int(self._clear_count)
        fields["stream_coach_last_clear_reason"] = str(self._last_clear_reason or "")
        fields["pending_stream_commands"] = int(fields.get("pending_stream_commands", 0) or 0) + int(source_pending)
        fields["stream_messages_seen"] = int(fields.get("stream_coach_messages_seen", 0) or 0)
        fields["stream_commands_parsed"] = int(fields.get("stream_coach_commands_parsed", 0) or 0)
        fields["stream_commands_accepted"] = int(fields.get("stream_coach_commands_accepted", 0) or 0)
        fields["stream_commands_rejected"] = int(fields.get("stream_coach_commands_rejected", 0) or 0)
        fields["mock_stream_messages_seen"] = int(fields.get("stream_coach_messages_seen", 0) or 0)
        fields["mock_stream_commands_parsed"] = int(fields.get("stream_coach_commands_parsed", 0) or 0)
        fields["mock_stream_commands_accepted"] = int(fields.get("stream_coach_commands_accepted", 0) or 0)
        fields["mock_stream_commands_rejected"] = int(fields.get("stream_coach_commands_rejected", 0) or 0)
        fields["last_stream_user"] = str(fields.get("stream_coach_last_user") or "")
        fields["last_stream_message"] = str(fields.get("stream_coach_last_message") or "")
        fields["last_stream_parsed_command"] = fields.get("stream_coach_last_parsed_command")
        fields["last_stream_command_status"] = str(fields.get("stream_coach_last_command_status") or "")
        fields["last_stream_reject_reason"] = str(fields.get("stream_coach_last_reject_reason") or "")
        fields["last_validated_coach_command"] = str(fields.get("stream_coach_last_validated_command") or "")
        fields["last_applied_coach_command"] = str(fields.get("stream_coach_last_applied_command") or "")
        return fields

    def clear_pending_state(self, *, clear_source: bool = True, reason: str = "reset") -> bool:
        stale_detected = self.aggregator.clear_pending_state()
        self._clear_count += 1
        self._last_clear_reason = str(reason or "reset")
        if clear_source and self.source is not None:
            self._startup_stale_clear_performed = True
            clear_fn = getattr(self.source, "clear", None)
            if not callable(clear_fn):
                clear_fn = getattr(self.source, "clear_to_end", None)
            if callable(clear_fn):
                try:
                    cleared = int(clear_fn() or 0)
                    self._stale_messages_cleared += max(0, int(cleared))
                    stale_detected = bool(cleared > 0 or stale_detected)
                except Exception:
                    stale_detected = True
        return bool(stale_detected)

    def _source_alive(self) -> bool:
        if not self.enabled:
            return False
        if self.mock_script_source is not None:
            return not bool(self.mock_script_source.load_errors)
        return True

    def _alive_status(self) -> str:
        if not self.enabled:
            return "off"
        if self.mock_script_source is not None and self.mock_script_source.load_errors:
            return "dead"
        age = max(0.0, time.time() - float(self._last_poll_time))
        if age > 15.0:
            return "stale"
        return "alive"


def _command_dict_from_human(command: Any) -> Dict[str, Any]:
    kind = str(getattr(command, "kind", "") or "").strip().lower()
    payload: Dict[str, Any] = {
        "command": kind,
        "raw_text": str(getattr(command, "raw_text", "") or ""),
        "timestamp": float(getattr(command, "timestamp", 0.0) or 0.0),
        "source": str(getattr(command, "source", "") or ""),
        "coach_command_id": int(getattr(command, "coach_command_id", 0) or 0),
    }
    if kind in {"plant", "fuse"}:
        payload["seed_index"] = int(getattr(command, "seed_index", -1))
        payload["row"] = int(getattr(command, "row", -1))
        payload["col"] = int(getattr(command, "col", -1))
    elif kind == "defend":
        payload["row"] = int(getattr(command, "row", -1))
    payload["canonical"] = _canonical_command(payload)
    return payload


def _canonical_command(payload: Dict[str, Any]) -> str:
    name = str(payload.get("command") or "").strip().lower()
    if name in {"plant", "fuse"}:
        return f"{name}:{_safe_int(payload.get('seed_index'))}:{_safe_int(payload.get('row'))}:{_safe_int(payload.get('col'))}"
    if name == "defend":
        return f"{name}:{_safe_int(payload.get('row'))}"
    return name


def _chat_text_from_payload(payload: Dict[str, Any]) -> str:
    raw_text = str(payload.get("raw_text") or "").strip()
    if raw_text:
        return raw_text
    command = str(payload.get("command") or "").strip().lower()
    if command in {"plant", "fuse"}:
        return f"!{command} {_safe_int(payload.get('seed_index'))} {_safe_int(payload.get('row'))} {_safe_int(payload.get('col'))}"
    if command == "defend":
        return f"!defend {_safe_int(payload.get('row'))}"
    if command == "economy":
        return "!economy"
    return "!wait"


def _chat_text_from_stream_command(command: StreamCoachCommand) -> str:
    if command.command in {"plant", "fuse"}:
        return f"!{command.command} {int(command.seed_index)} {int(command.row)} {int(command.col)}"
    if command.command == "defend":
        return f"!defend {int(command.row)}"
    if command.command == "economy":
        return "!economy"
    return "!wait"


def _plant_types_from_observation(observation: Dict[str, Any]) -> List[int]:
    slots = observation.get("seedSlots")
    if not isinstance(slots, list):
        return []
    values: List[int] = []
    for slot in slots:
        if not isinstance(slot, dict):
            continue
        values.append(_safe_int(slot.get("plantType"), default=-1))
    return [value for value in values if value >= 0]


def _infer_max_seed_slots(observation: Dict[str, Any], mode: str, plant_types: Sequence[int]) -> Optional[int]:
    raw = observation.get("max_seed_slots", observation.get("maxSeedSlots"))
    if raw is not None:
        value = _safe_int(raw, default=-1)
        if value > 0:
            return int(value)
    slots = observation.get("seedSlots")
    if isinstance(slots, list) and slots:
        return len(slots)
    if plant_types:
        return len(plant_types)
    normalized_mode = normalize_action_space_mode(mode)
    if normalized_mode in {"dynamic_14", "adventure_14slot_identity"}:
        return 14
    return None


def _mask_from_legal_actions(
    *,
    legal_actions: Optional[Sequence[int]],
    mode: str,
    rows: int,
    cols: int,
    plant_types: Sequence[int],
    max_seed_slots: Optional[int],
) -> Optional[List[bool]]:
    if legal_actions is None:
        return None
    spec = build_action_space_spec(
        mode=mode,
        plant_types=[int(value) for value in plant_types],
        max_seed_slots=max_seed_slots,
        rows=int(rows),
        cols=int(cols),
    )
    mask = [False] * int(spec.action_count)
    for value in legal_actions:
        action = _safe_int(value, default=-1)
        if 0 <= action < len(mask):
            mask[action] = True
    return mask
