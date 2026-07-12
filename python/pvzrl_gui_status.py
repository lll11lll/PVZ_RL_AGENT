"""Live-status reading and health classification for the Tk dashboard."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple


LIVE_MAX_AGE_SECONDS = 5.0
STALE_MAX_AGE_SECONDS = 30.0
StatusInfo = Dict[str, Any]
StatusPayload = Dict[str, Any]
StatusSignature = Tuple[int, int, int, int, int]
HealthClassifier = Callable[[float, Optional[StatusPayload]], str]
MISSING = object()
_NON_RENDERED_VOLATILE_KEYS = frozenset({"updated_at", "stream_coach_last_poll_age_seconds"})


def _lookup_path(payload: Any, path: str) -> Any:
    current = payload
    for part in path.split("."):
        if not isinstance(current, dict):
            return MISSING
        if part in current:
            current = current[part]
            continue
        lower_lookup = {str(key).lower(): key for key in current.keys()}
        key = lower_lookup.get(part.lower())
        if key is None:
            return MISSING
        current = current[key]
    return current


def first_status_value(payload: Any, paths: Tuple[str, ...], default: Any = None) -> Any:
    for path in paths:
        value = _lookup_path(payload, path)
        if value is not MISSING and value is not None and value != "":
            return value
    return default


class NormalizedStatusIndex:
    """Cache case-insensitive dictionary keys once for compatibility lookups."""

    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self._lower_keys: Dict[int, Dict[str, Any]] = {}
        self._objects: set[int] = set()
        self._visit(payload)

    def _visit(self, value: Any) -> None:
        if isinstance(value, dict):
            object_id = id(value)
            if object_id in self._objects:
                return
            self._objects.add(object_id)
            self._lower_keys[object_id] = {str(key).lower(): key for key in value.keys()}
            for item in value.values():
                self._visit(item)
        elif isinstance(value, list):
            for item in value:
                self._visit(item)

    def contains(self, value: Any) -> bool:
        return isinstance(value, dict) and id(value) in self._objects

    def lookup(self, payload: Any, path: str) -> Any:
        current = payload
        for part in path.split("."):
            if not isinstance(current, dict):
                return MISSING
            if part in current:
                current = current[part]
                continue
            key = self._lower_keys.get(id(current), {}).get(part.lower())
            if key is None:
                return MISSING
            current = current[key]
        return current

    def first(self, payload: Any, paths: Iterable[str], default: Any = None) -> Any:
        for path in paths:
            value = self.lookup(payload, path)
            if value is not MISSING and value is not None and value != "":
                return value
        return default


def classify_live_health(age: float, payload: Optional[StatusPayload] = None) -> str:
    """Classify file freshness before compatibility blocked-state aliases."""

    normalized_payload = payload if isinstance(payload, dict) else {}
    if age > STALE_MAX_AGE_SECONDS:
        return "DEAD"
    if age >= LIVE_MAX_AGE_SECONDS:
        return "STALE"
    blocked_reason = str(
        first_status_value(
            normalized_payload,
            (
                "blocked_reason",
                "adventure.blocked_reason",
                "post_win_blocked_reason",
                "adventure.post_win_blocked_reason",
            ),
            default="",
        )
        or ""
    )
    if blocked_reason.startswith("post_win_") or blocked_reason in {
        "trophy_visible_but_click_failed",
        "reward_or_unlock_click_failed_after_win",
    }:
        return "BLOCKED_POST_WIN"
    if "seed_selection" in blocked_reason:
        return "BLOCKED_SEED_SELECTION"
    if "gameplay_ready" in blocked_reason or "gameplay" in blocked_reason:
        return "BLOCKED_GAMEPLAY_READY"
    return "LIVE"


class LiveStatusReader:
    """Read one atomic JSON status file while caching unchanged parse results."""

    def __init__(
        self,
        path: Path,
        *,
        health_classifier: HealthClassifier = classify_live_health,
    ) -> None:
        self.path = Path(path)
        self.health_classifier = health_classifier
        self.signature: Optional[StatusSignature] = None
        self.cached_payload: Optional[StatusPayload] = None
        self.cached_state = "MISSING"
        self.cached_parse_error = ""

    def set_path(self, path: Path) -> None:
        resolved = Path(path)
        if resolved == self.path:
            return
        self.path = resolved
        self.reset()

    def reset(self) -> None:
        self.signature = None
        self.cached_payload = None
        self.cached_state = "MISSING"
        self.cached_parse_error = ""

    def read(self) -> Tuple[Optional[StatusPayload], StatusInfo]:
        path = self.path
        info: StatusInfo = {
            "path": path,
            "exists": False,
            "size": None,
            "mtime": None,
            "age": None,
            "health": "MISSING",
            "parse_error": "",
        }
        try:
            stat = path.stat()
        except FileNotFoundError:
            self.reset()
            return None, info
        except OSError as exc:
            info["health"] = "MALFORMED"
            info["parse_error"] = f"OSError while stat() reading live status: {exc}"
            return None, info

        age = max(0.0, time.time() - stat.st_mtime)
        info.update({"exists": True, "size": int(stat.st_size), "mtime": stat.st_mtime, "age": age})
        signature: StatusSignature = (
            int(getattr(stat, "st_dev", 0) or 0),
            int(getattr(stat, "st_ino", 0) or 0),
            int(stat.st_size),
            int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))),
            int(getattr(stat, "st_ctime_ns", int(stat.st_ctime * 1_000_000_000))),
        )
        if stat.st_size == 0:
            self.signature = signature
            self.cached_payload = None
            self.cached_state = "EMPTY"
            self.cached_parse_error = ""
            info["health"] = "EMPTY"
            return None, info

        if signature == self.signature:
            info["unchanged"] = True
            info["parse_error"] = self.cached_parse_error
            if self.cached_payload is not None:
                info["health"] = self.health_classifier(age, self.cached_payload)
                return self.cached_payload, info
            info["health"] = self.cached_state
            return None, info

        try:
            content = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            self.reset()
            info.update({"exists": False, "size": None, "mtime": None, "age": None, "health": "MISSING"})
            return None, info
        except OSError as exc:
            info["health"] = "MALFORMED"
            info["parse_error"] = f"OSError while reading live status: {exc}"
            return None, info

        if not content.strip():
            self.signature = signature
            self.cached_payload = None
            self.cached_state = "EMPTY"
            self.cached_parse_error = ""
            info["health"] = "EMPTY"
            return None, info

        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            info["health"] = "MALFORMED"
            info["parse_error"] = f"{exc.msg} at line {exc.lineno}, column {exc.colno}"
            self._cache_failure(signature, str(info["parse_error"]))
            return None, info

        if not isinstance(payload, dict):
            info["health"] = "MALFORMED"
            info["parse_error"] = f"Expected JSON object, got {type(payload).__name__}"
            self._cache_failure(signature, str(info["parse_error"]))
            return None, info

        info["health"] = self.health_classifier(age, payload)
        self.signature = signature
        self.cached_payload = payload
        self.cached_state = str(info["health"])
        self.cached_parse_error = ""
        return payload, info

    def _cache_failure(self, signature: StatusSignature, error: str) -> None:
        self.signature = signature
        self.cached_payload = None
        self.cached_state = "MALFORMED"
        self.cached_parse_error = str(error or "")


@dataclass(frozen=True, eq=False)
class DiagnosticsRenderKey:
    payload: Optional[StatusPayload]
    normalized_payload: Any
    health: str
    using_last_good: bool

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, DiagnosticsRenderKey)
            and self.health == other.health
            and self.using_last_good == other.using_last_good
            and (
                self.payload is other.payload
                or self.normalized_payload == other.normalized_payload
            )
        )


def _normalized_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        rows = []
        for key, item in value.items():
            normalized_key = str(key).lower()
            if normalized_key in _NON_RENDERED_VOLATILE_KEYS:
                rows.append((normalized_key, "<volatile>"))
                continue
            rows.append((normalized_key, _normalized_json_value(item)))
        return tuple(sorted(rows, key=lambda row: row[0]))
    if isinstance(value, list):
        return tuple(_normalized_json_value(item) for item in value)
    return value


def normalized_status_view(payload: Optional[StatusPayload]) -> Any:
    """Canonicalize JSON content used by the GUI, excluding writer-only churn."""

    if payload is None:
        return None
    return _normalized_json_value(payload)


def diagnostics_render_key(
    payload: Optional[StatusPayload],
    health: str,
    using_last_good: bool,
    *,
    previous: Optional[DiagnosticsRenderKey] = None,
) -> DiagnosticsRenderKey:
    """Return a cheap key that changes when the cached view must rerender."""

    normalized_health = str(health)
    normalized_last_good = bool(using_last_good)
    if (
        previous is not None
        and previous.payload is payload
        and previous.health == normalized_health
        and previous.using_last_good == normalized_last_good
    ):
        return previous
    return DiagnosticsRenderKey(
        payload,
        normalized_status_view(payload),
        normalized_health,
        normalized_last_good,
    )
