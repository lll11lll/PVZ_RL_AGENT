"""Bounded, privacy-safe Streamer V1 event logging."""

from __future__ import annotations

import json
import hashlib
import os
import threading
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np


_FORBIDDEN_EXACT_KEYS = frozenset(
    {
        "raw_text",
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
        "twitch_user_id",
        "chatter_user_id",
        "chatter_user_login",
        "chatter_user_name",
    }
)
_FORBIDDEN_KEY_FRAGMENTS = ("token", "secret", "password", "authorization", "cookie")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_safe_dict"):
        return _json_safe(value.to_safe_dict())
    return str(value)


def _privacy_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in _FORBIDDEN_EXACT_KEYS or any(
                fragment in normalized for fragment in _FORBIDDEN_KEY_FRAGMENTS
            ):
                continue
            cleaned[str(key)] = _privacy_safe(item)
        return cleaned
    if isinstance(value, (list, tuple)):
        return [_privacy_safe(item) for item in value]
    return value


class BufferedStreamerEventLogger:
    """Append compact structured events without observations or raw chat text."""

    def __init__(
        self,
        path: Path,
        *,
        flush_records: int = 256,
        max_bytes: int = 64 * 1024 * 1024,
        backup_count: int = 3,
    ) -> None:
        self.path = path
        self.flush_records = max(1, int(flush_records))
        self.max_bytes = max(1024, int(max_bytes))
        self.backup_count = max(1, int(backup_count))
        self._pending: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def append(self, event: Mapping[str, Any], *, force: bool = False) -> None:
        row = _privacy_safe(_json_safe(dict(event)))
        if not isinstance(row, dict):
            raise TypeError("Streamer log row must be an object")
        with self._lock:
            self._pending.append(row)
            if force or len(self._pending) >= self.flush_records:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def close(self) -> None:
        self.flush()

    def _flush_locked(self) -> None:
        if not self._pending:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pending_bytes = sum(
            len(json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")) + 1
            for row in self._pending
        )
        if self.path.is_file() and self.path.stat().st_size + pending_bytes > self.max_bytes:
            self._rotate_locked()
        with self.path.open("a", encoding="utf-8") as handle:
            for row in self._pending:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._pending.clear()

    def _rotate_locked(self) -> None:
        oldest = self.path.with_name(f"{self.path.name}.{self.backup_count}")
        oldest.unlink(missing_ok=True)
        for index in range(self.backup_count - 1, 0, -1):
            source = self.path.with_name(f"{self.path.name}.{index}")
            if source.is_file():
                os.replace(source, self.path.with_name(f"{self.path.name}.{index + 1}"))
        if self.path.is_file():
            os.replace(self.path, self.path.with_name(f"{self.path.name}.1"))


def compact_observation_revision(observation: Optional[Mapping[str, Any]]) -> str:
    if not isinstance(observation, Mapping):
        return ""
    for key in ("observationRevision", "stateRevision", "frameCount", "frame"):
        if key in observation and observation.get(key) is not None:
            return str(observation.get(key))[:128]
    return ""


def observation_vector_digest(observation: Any) -> str:
    """Hash one model-facing observation without persisting its contents."""

    array = np.asarray(observation, dtype=np.float32)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


__all__ = [
    "BufferedStreamerEventLogger",
    "compact_observation_revision",
    "observation_vector_digest",
]
