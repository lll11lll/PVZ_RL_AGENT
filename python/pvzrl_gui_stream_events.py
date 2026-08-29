"""Bounded, privacy-safe Streamer V1 event history for GUI consumers.

The Streamer runtime remains the authority for command legality and execution.
This module only tails its canonical ``logs/streamer_events.jsonl`` artifact and
projects records into a small, display-oriented schema.  It deliberately has
no Tk dependency and starts no threads; GUI code owns the polling cadence.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import deque
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Deque, Mapping, Optional, Tuple, Union

from pvzrl_action_space import (
    ADVENTURE_IDENTITY_ACTION_COUNT,
    ADVENTURE_IDENTITY_MAX_SEED_SLOTS,
    DEFAULT_COLS,
    DEFAULT_ROWS,
    adventure_identity_action_to_slot_cell,
)
from pvzrl_file_tail import IncrementalLineTailReader


STREAMER_EVENT_LOG_RELATIVE_PATH = Path("logs") / "streamer_events.jsonl"
DEFAULT_EVENT_HISTORY_ROWS = 100
DEFAULT_EVENT_READ_BYTES = 256 * 1024
DEFAULT_EVENT_PENDING_BYTES = 2 * DEFAULT_EVENT_READ_BYTES

_VIEWER_HASH_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_SAFE_CODE_RE = re.compile(r"^[A-Za-z0-9_.:+/\-]{1,128}$")
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9 .+'()\-]{1,96}$")
_KNOWN_EVENTS = frozenset({"viewer_command", "executed_decision", "bc_demo_result"})
_KNOWN_COMMAND_KINDS = frozenset({"plant", "slot", "fuse_result", "fuse_tile"})
_KNOWN_ACTION_SOURCES = frozenset({"MODEL", "TWITCH", "VIEWER", "SCRIPTED", "SYSTEM"})
_EXECUTION_VERIFIED_STATUS = "executed_verified"

_UNSAFE_EXACT_KEYS = frozenset(
    {
        "raw_text",
        "raw_chat",
        "raw_identity",
        "command_text",
        "chat_text",
        "message",
        "message_text",
        "text",
        "username",
        "display_name",
        "user_id",
        "user_login",
        "user_name",
        "viewer_id",
        "twitch_user_id",
        "chatter_user_id",
        "chatter_user_login",
        "chatter_user_name",
    }
)
_UNSAFE_KEY_FRAGMENTS = ("token", "secret", "password", "authorization", "cookie")


def streamer_event_log_path(run_directory: Union[str, Path]) -> Path:
    """Return the one canonical Streamer event-log path for a run directory."""

    return Path(run_directory) / STREAMER_EVENT_LOG_RELATIVE_PATH


def _optional_path(value: Any) -> Optional[Path]:
    text = str(value or "").strip()
    if not text or text.lower() in {"-", "n/a", "unknown"}:
        return None
    try:
        return Path(text).resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return None


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _marked_streamer_experiment(path: Path) -> Optional[Path]:
    """Find a nearby canonical Streamer experiment marker, if one exists."""

    candidates = (path, *tuple(path.parents)[:10])
    for candidate in candidates:
        try:
            if (
                (candidate / "streamer_state.json").is_file()
                or streamer_event_log_path(candidate).is_file()
            ):
                return candidate
        except OSError:
            continue
    return None


def _structural_streamer_experiment(path: Path) -> Optional[Path]:
    """Recover the experiment root from maintained cycle/evaluation layouts."""

    parts = path.parts
    for index, part in enumerate(parts):
        if index > 0 and str(part).casefold() in {"cycles", "evaluations"}:
            return Path(*parts[:index])
    return None


def resolve_streamer_experiment_directory(
    *,
    explicit_experiment: Any = None,
    active_run: Any = None,
    configured_run: Any = None,
    launched_run: Any = None,
    prefer_configured: bool = False,
) -> Optional[Path]:
    """Resolve the experiment root that owns the canonical event log.

    Inner Streamer train/evaluation entrypoints publish their cycle artifact
    directory as ``active_run`` while the event log remains owned by the outer
    experiment.  This resolver accepts a future explicit status field, the
    GUI-owned launch directory, canonical filesystem markers, and the
    maintained ``cycles``/``evaluations`` layouts.  A stale status projection
    may explicitly prefer the currently configured run.
    """

    explicit = _optional_path(explicit_experiment)
    active = _optional_path(active_run)
    configured = _optional_path(configured_run)
    launched = _optional_path(launched_run)

    if prefer_configured and configured is not None:
        return configured
    if explicit is not None:
        return explicit
    if launched is not None and (active is None or _path_is_within(active, launched)):
        return launched
    if active is not None:
        marked = _marked_streamer_experiment(active)
        if marked is not None:
            return marked
        structural = _structural_streamer_experiment(active)
        if structural is not None:
            return structural
        if configured is not None and _path_is_within(active, configured):
            return configured
        return active
    return launched or configured


def _safe_int(value: Any, *, minimum: int, maximum: int) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        parsed = int(value)
    else:
        return None
    if not minimum <= parsed <= maximum:
        return None
    return parsed


def _safe_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _safe_identifier(value: Any) -> str:
    text = str(value or "") if isinstance(value, str) else ""
    return text if _SAFE_IDENTIFIER_RE.fullmatch(text) is not None else ""


def _safe_code(value: Any) -> str:
    text = str(value or "") if isinstance(value, str) else ""
    return text if _SAFE_CODE_RE.fullmatch(text) is not None else ""


def _safe_name(value: Any) -> str:
    text = str(value or "").strip() if isinstance(value, str) else ""
    return text if _SAFE_NAME_RE.fullmatch(text) is not None else ""


def _short_viewer_label(value: Any) -> str:
    if not isinstance(value, str) or _VIEWER_HASH_RE.fullmatch(value) is None:
        return ""
    return f"hash:{value[:8].lower()}"


def _viewer_coordinate(value: Any, *, maximum: int) -> Optional[int]:
    """Convert a canonical zero-based coordinate exactly once for display."""

    canonical = _safe_int(value, minimum=0, maximum=maximum - 1)
    return canonical + 1 if canonical is not None else None


def _structured_command(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw_command: Optional[Mapping[str, Any]] = None
    for key in ("command", "parsed_fields", "structured_command"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            raw_command = candidate
            break

    if raw_command is not None:
        kind = str(raw_command.get("kind") or "").strip().lower()
        row = _viewer_coordinate(raw_command.get("row"), maximum=DEFAULT_ROWS)
        column = _viewer_coordinate(raw_command.get("column"), maximum=DEFAULT_COLS)
        if kind in _KNOWN_COMMAND_KINDS and row is not None and column is not None:
            normalized: dict[str, Any] = {
                "kind": kind,
                "row": row,
                "column": column,
                "coordinate_base": 1,
            }
            if kind == "slot":
                slot = _viewer_coordinate(
                    raw_command.get("seed_slot"),
                    maximum=ADVENTURE_IDENTITY_MAX_SEED_SLOTS,
                )
                if slot is None:
                    return {}
                normalized["slot"] = slot
            elif kind == "plant":
                plant_type = _safe_int(
                    raw_command.get("plant_type_id"), minimum=0, maximum=2_147_483_647
                )
                plant_name = _safe_name(raw_command.get("canonical_plant_name"))
                if plant_type is not None:
                    normalized["plant_type_id"] = plant_type
                if plant_name:
                    normalized["plant_name"] = plant_name
            elif kind == "fuse_result":
                result_type = _safe_int(
                    raw_command.get("fusion_result_type_id"),
                    minimum=0,
                    maximum=2_147_483_647,
                )
                result_name = _safe_name(raw_command.get("canonical_fusion_result_name"))
                if result_type is not None:
                    normalized["fusion_result_type_id"] = result_type
                if result_name:
                    normalized["fusion_result_name"] = result_name
            return normalized

    # BC-result records contain only the canonical action ID.  Decoding that
    # ID uses the maintained 14 x (6 x 10) policy contract rather than a GUI
    # copy of the geometry.
    action_id = _event_action_id(payload)
    if action_id is None:
        return {}
    decoded = adventure_identity_action_to_slot_cell(action_id)
    kind = int(decoded.get("kind", -1))
    if kind == 0:
        return {"kind": "wait", "coordinate_base": 1}
    if kind != 1:
        return {}
    return {
        "kind": "action",
        "slot": int(decoded["slot_index"]) + 1,
        "row": int(decoded["row"]) + 1,
        "column": int(decoded["column"]) + 1,
        "coordinate_base": 1,
    }


def format_streamer_command(command: Mapping[str, Any]) -> str:
    """Render one already-normalized command without reconstructing raw chat."""

    if not isinstance(command, Mapping):
        return ""
    kind = str(command.get("kind") or "")
    if kind == "wait":
        return "Wait"
    row = _safe_int(command.get("row"), minimum=1, maximum=DEFAULT_ROWS)
    column = _safe_int(command.get("column"), minimum=1, maximum=DEFAULT_COLS)
    target = f"R{row} C{column}" if row is not None and column is not None else ""
    if kind in {"slot", "action"}:
        slot = _safe_int(
            command.get("slot"), minimum=1, maximum=ADVENTURE_IDENTITY_MAX_SEED_SLOTS
        )
        prefix = f"Slot {slot}" if slot is not None else "Slot action"
    elif kind == "plant":
        prefix = _safe_name(command.get("plant_name")) or "Plant"
    elif kind == "fuse_result":
        result = _safe_name(command.get("fusion_result_name"))
        prefix = f"Fuse to {result}" if result else "Fusion"
    elif kind == "fuse_tile":
        prefix = "Fuse tile"
    else:
        return ""
    return f"{prefix} at {target}" if target else prefix


def _event_action_id(payload: Mapping[str, Any]) -> Optional[int]:
    for key in ("viewer_action_id", "canonical_action_id", "action_id", "viewer_action"):
        value = _safe_int(
            payload.get(key), minimum=0, maximum=ADVENTURE_IDENTITY_ACTION_COUNT - 1
        )
        if value is not None:
            return value
    return None


def _format_wall_time(value: float) -> Optional[str]:
    if not 0.0 <= value <= 32_503_680_000.0:  # through year 3000
        return None
    try:
        return datetime.fromtimestamp(value).astimezone().strftime("%H:%M:%S")
    except (OSError, OverflowError, ValueError):
        return None


def _event_time(
    payload: Mapping[str, Any], observed_at_unix: Optional[float]
) -> Tuple[str, str]:
    for key in ("executed_at_unix", "timestamp_unix", "time_unix", "created_at_unix"):
        value = _safe_float(payload.get(key))
        rendered = _format_wall_time(value) if value is not None else None
        if rendered is not None:
            return rendered, "wall"

    for key in ("published_at", "timestamp", "occurred_at"):
        value = payload.get(key)
        if not isinstance(value, str) or not value or len(value) > 64:
            continue
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            continue
        return parsed.astimezone().strftime("%H:%M:%S"), "wall"

    monotonic = _safe_float(payload.get("occurred_monotonic"))
    if monotonic is not None and monotonic >= 0.0:
        return f"T+{monotonic:.3f}s", "monotonic"

    observed = _safe_float(observed_at_unix)
    rendered = _format_wall_time(observed) if observed is not None else None
    if rendered is not None:
        return rendered, "observed"
    return "", "missing"


def _unsafe_key_count(value: Any, *, node_limit: int = 4_096) -> int:
    """Count privacy-risk keys iteratively without trusting nesting depth."""

    stack = [value]
    visited = 0
    unsafe = 0
    while stack and visited < node_limit:
        current = stack.pop()
        visited += 1
        if isinstance(current, Mapping):
            for raw_key, item in current.items():
                key = str(raw_key).strip().lower().replace("-", "_")
                if key in _UNSAFE_EXACT_KEYS or any(
                    fragment in key for fragment in _UNSAFE_KEY_FRAGMENTS
                ):
                    unsafe += 1
                if isinstance(item, (Mapping, list, tuple)):
                    stack.append(item)
        elif isinstance(current, (list, tuple)):
            stack.extend(item for item in current if isinstance(item, (Mapping, list, tuple)))
    return unsafe


def normalize_streamer_event(
    payload: Mapping[str, Any],
    *,
    observed_at_unix: Optional[float] = None,
) -> dict[str, Any]:
    """Project one canonical log event into the GUI's strict safe schema.

    Full viewer hashes, raw records, raw chat, identities, and credential-like
    fields are never copied.  ``executed`` is deliberately fail-closed: only
    the runtime's explicit ``executed_verified`` status proves execution.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("Streamer event must be a mapping")

    raw_event = str(payload.get("event") or "").strip().lower()
    event_type = raw_event if raw_event in _KNOWN_EVENTS else "unknown"
    event_time, time_source = _event_time(payload, observed_at_unix)
    command = _structured_command(payload)

    nested_execution = payload.get("execution_result")
    nested_execution = nested_execution if isinstance(nested_execution, Mapping) else {}
    execution_status = _safe_code(payload.get("execution_status")) or _safe_code(
        nested_execution.get("status")
    )
    status = _safe_code(payload.get("status")) or execution_status or event_type

    raw_legality = _safe_code(payload.get("canonical_legality_result")) or _safe_code(
        payload.get("legality")
    )
    legality = raw_legality.upper()
    raw_legal = payload.get("legal")
    legal: Optional[bool] = raw_legal if isinstance(raw_legal, bool) else None
    if legal is None and legality == "LEGAL":
        legal = True
    elif legal is None and legality in {
        "ILLEGAL",
        "CURRENTLY_ILLEGAL",
        "TEMPORARILY_BLOCKED",
        "PERMANENTLY_INVALID",
        "UNRESOLVABLE",
        "STALE",
    }:
        legal = False
    if not legality and legal is not None:
        legality = "LEGAL" if legal else "ILLEGAL"

    source = str(payload.get("action_source") or "").strip().upper()
    if source not in _KNOWN_ACTION_SOURCES:
        if event_type in {"viewer_command", "bc_demo_result"}:
            source = "VIEWER"
        else:
            source = "UNKNOWN"

    if execution_status:
        execution_verified: Optional[bool] = execution_status == _EXECUTION_VERIFIED_STATUS
        executed: Optional[bool] = bool(execution_verified)
    else:
        execution_verified = None
        executed = None

    bridge_success_value = payload.get("bridge_success")
    if not isinstance(bridge_success_value, bool):
        bridge_success_value = nested_execution.get("bridge_success")
    bridge_success = bridge_success_value if isinstance(bridge_success_value, bool) else None

    result = (
        _safe_code(payload.get("bridge_reason"))
        or _safe_code(payload.get("reason"))
        or _safe_code(payload.get("bc_demo_reject_reason"))
        or execution_status
        or status
    )
    bc_recorded = payload.get("bc_demo_recorded")
    if not isinstance(bc_recorded, bool):
        bc_recorded = None

    return {
        "event_type": event_type,
        "time": event_time,
        "time_source": time_source,
        "viewer": _short_viewer_label(payload.get("viewer_hash")),
        "command": command,
        "command_label": format_streamer_command(command),
        "command_id": _safe_identifier(payload.get("command_id")),
        "action_id": _event_action_id(payload),
        "legality": legality or "UNKNOWN",
        "legal": legal,
        "status": status,
        "executed": executed,
        "execution_verified": execution_verified,
        "execution_status": execution_status,
        "result": result,
        "bridge_success": bridge_success,
        "action_source": source,
        "bc_demo_recorded": bc_recorded,
        "cycle_id": _safe_int(payload.get("cycle_id"), minimum=0, maximum=2_147_483_647),
        "episode_id": _safe_int(
            payload.get("episode_id"), minimum=0, maximum=2_147_483_647
        ),
    }


def _is_viewer_history_event(
    payload: Mapping[str, Any], row: Mapping[str, Any]
) -> bool:
    """Retain Viewer command, Viewer execution, and BC outcome records only."""

    event_type = str(row.get("event_type") or "")
    if event_type in {"viewer_command", "bc_demo_result"}:
        return True
    if event_type != "executed_decision":
        return False
    if str(row.get("action_source") or "").upper() in {"TWITCH", "VIEWER"}:
        return True
    if _VIEWER_HASH_RE.fullmatch(str(payload.get("viewer_hash") or "")) is not None:
        return True
    return _safe_int(
        payload.get("viewer_action_id"),
        minimum=0,
        maximum=ADVENTURE_IDENTITY_ACTION_COUNT - 1,
    ) is not None


class StreamerEventReader:
    """Incrementally retain a bounded safe projection of Streamer events."""

    def __init__(
        self,
        path: Union[str, Path],
        *,
        max_rows: int = DEFAULT_EVENT_HISTORY_ROWS,
        start_at_end: bool = False,
        max_read_bytes: int = DEFAULT_EVENT_READ_BYTES,
        max_pending_bytes: int = DEFAULT_EVENT_PENDING_BYTES,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if int(max_rows) <= 0:
            raise ValueError("max_rows must be positive")
        self.path = Path(path)
        self.max_rows = int(max_rows)
        self.start_at_end = bool(start_at_end)
        self.max_read_bytes = max(1, int(max_read_bytes))
        self.max_pending_bytes = max(self.max_read_bytes, int(max_pending_bytes))
        self._clock = clock
        self._rows: Deque[dict[str, Any]] = deque(maxlen=self.max_rows)
        self._unsafe_fields_dropped = 0
        self._non_viewer_events_filtered = 0
        self._bootstrap_rows_pending = 0
        self._tail = self._new_tail()
        self._bootstrap_recent_rows()

    @classmethod
    def for_run(
        cls, run_directory: Union[str, Path], **kwargs: Any
    ) -> "StreamerEventReader":
        return cls(streamer_event_log_path(run_directory), **kwargs)

    def _new_tail(self) -> IncrementalLineTailReader:
        # Existing logs are bootstrapped from their recent bounded tail below;
        # incremental reads then begin at the captured end offset.  For a path
        # that does not exist yet, start-at-end still begins at offset zero.
        return IncrementalLineTailReader(
            self.path,
            start_at_end=True,
            max_read_bytes=self.max_read_bytes,
            max_pending_bytes=self.max_pending_bytes,
        )

    def _append_payload(self, payload: Mapping[str, Any]) -> bool:
        try:
            row = normalize_streamer_event(payload, observed_at_unix=self._clock())
        except (TypeError, ValueError, OverflowError, RecursionError):
            self._tail.note_malformed_record("invalid_streamer_event_shape")
            return False
        if not _is_viewer_history_event(payload, row):
            self._non_viewer_events_filtered += 1
            return False
        self._unsafe_fields_dropped += _unsafe_key_count(payload)
        self._rows.append(row)
        return True

    def _consume_line(self, line: str) -> bool:
        if not line.strip():
            return False
        try:
            payload = json.loads(line)
        except (json.JSONDecodeError, RecursionError, ValueError):
            self._tail.note_malformed_record("malformed_streamer_event_json")
            return False
        if not isinstance(payload, Mapping):
            self._tail.note_malformed_record("streamer_event_not_object")
            return False
        return self._append_payload(payload)

    def _bootstrap_recent_rows(self) -> None:
        """Load recent complete records while leaving incremental reads at EOF."""

        if self.start_at_end:
            return
        end_offset = int(self._tail.read_offset)
        if end_offset <= 0:
            return
        read_limit = max(self.max_read_bytes, self.max_pending_bytes)
        start_offset = max(0, end_offset - read_limit)
        try:
            with self.path.open("rb") as handle:
                handle.seek(start_offset)
                data = handle.read(end_offset - start_offset)
        except OSError as exc:
            self._tail.last_error = str(exc)
            return
        if start_offset > 0:
            newline = data.find(b"\n")
            if newline < 0:
                return
            data = data[newline + 1 :]
        if data and not data.endswith(b"\n"):
            newline = data.rfind(b"\n")
            data = data[: newline + 1] if newline >= 0 else b""
        for raw_line in data.splitlines():
            try:
                line = raw_line.decode("utf-8", errors="strict")
            except UnicodeError:
                self._tail.note_malformed_record("malformed_streamer_event_utf8")
                continue
            if self._consume_line(line):
                self._bootstrap_rows_pending += 1

    def set_path(self, path: Union[str, Path]) -> None:
        resolved = Path(path)
        if resolved == self.path:
            return
        self.path = resolved
        self.reset()

    def set_run_directory(self, run_directory: Union[str, Path]) -> None:
        self.set_path(streamer_event_log_path(run_directory))

    def reset(self) -> None:
        self._rows.clear()
        self._unsafe_fields_dropped = 0
        self._non_viewer_events_filtered = 0
        self._bootstrap_rows_pending = 0
        self._tail = self._new_tail()
        self._bootstrap_recent_rows()

    @property
    def rows(self) -> Tuple[dict[str, Any], ...]:
        # Callers cannot mutate the retained nested command dictionaries.
        return tuple(deepcopy(row) for row in self._rows)

    def read(self) -> Tuple[Tuple[dict[str, Any], ...], dict[str, Any]]:
        """Read one bounded chunk and return retained rows plus diagnostics."""

        rows_added = int(self._bootstrap_rows_pending)
        self._bootstrap_rows_pending = 0
        for line in self._tail.read_lines():
            if self._consume_line(line):
                rows_added += 1
        return self.rows, self.diagnostics(rows_added=rows_added)

    poll = read

    def diagnostics(self, *, rows_added: int = 0) -> dict[str, Any]:
        try:
            exists = self.path.is_file()
        except OSError:
            exists = False
        tail = self._tail.diagnostics()
        return {
            "path": self.path,
            "exists": exists,
            "rows_retained": len(self._rows),
            "rows_added": max(0, int(rows_added)),
            "row_capacity": self.max_rows,
            "unsafe_fields_dropped": self._unsafe_fields_dropped,
            "non_viewer_events_filtered": self._non_viewer_events_filtered,
            "pending_partial_bytes": int(tail.get("pending_bytes", 0) or 0),
            "rotation_count": int(tail.get("rotation_count", 0) or 0),
            "truncation_count": int(tail.get("truncation_count", 0) or 0),
            "decode_error_count": int(tail.get("decode_error_count", 0) or 0),
            "malformed_record_count": int(tail.get("malformed_record_count", 0) or 0),
            "oversized_record_count": int(tail.get("oversized_record_count", 0) or 0),
            "last_error": str(tail.get("last_error") or ""),
        }


__all__ = [
    "DEFAULT_EVENT_HISTORY_ROWS",
    "STREAMER_EVENT_LOG_RELATIVE_PATH",
    "StreamerEventReader",
    "format_streamer_command",
    "normalize_streamer_event",
    "resolve_streamer_experiment_directory",
    "streamer_event_log_path",
]
