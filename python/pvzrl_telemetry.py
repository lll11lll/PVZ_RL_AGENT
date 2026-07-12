"""Low-overhead, compatibility-preserving telemetry writers.

Live status is a latest-state artifact, not an event log.  Ordinary updates may
therefore be coalesced, while state transitions and terminal/failure evidence
must be written immediately.  Event streams such as episode JSONL remain
append-only and are never passed through this writer.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Hashable, Mapping, Optional, Tuple


DEFAULT_LIVE_STATUS_INTERVAL_SECONDS = 0.5

_SIGNIFICANT_KEYS = (
    "mode",
    "run_mode",
    "status",
    "health",
    "state",
    "blocked_reason",
    "done_reason",
    "terminal_reason",
    "last_result",
    "latest_terminal_result",
    "current_level",
    "frontier_level",
    "current_attempt",
    "episode",
    "episode_index",
    "current_episode",
    "current_wave",
    "wave",
    "max_wave",
    "screenState",
    "terminalHint",
    "post_win_blocked_reason",
    "post_win_decision",
    "post_win_transition_allowed",
    "post_win_active",
    "post_win_last_state",
    "timeout_classification",
    "soft_timeout_reached",
    "rewardScreenVisible",
    "unlockScreenVisible",
    "seedSelectionActive",
)

_SIGNIFICANT_NESTED_PATHS = (
    ("gameplay", "wave"),
    ("gameplay", "screen_state"),
    ("gameplay", "gameplay_ready"),
    ("agent", "episode"),
    ("adventure", "state"),
    ("adventure", "screenState"),
    ("adventure", "current_level"),
    ("adventure", "current_attempt"),
)

_FORCED_STATUS_VALUES = frozenset(
    {
        "blocked",
        "complete",
        "completed",
        "dead",
        "error",
        "failed",
        "failure",
        "finished",
        "stopped",
        "timeout",
    }
)


def _hashable(value: Any) -> Hashable:
    """Return a stable, cheap token for the small values selected below."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(item) for item in value)
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _hashable(item)) for key, item in value.items()))
    return str(value)


def live_status_significant_state(*sources: Optional[Mapping[str, Any]]) -> Tuple[Hashable, ...]:
    """Build the state-change token shared by fixed and Adventure writers.

    Step counters, rewards, sun, and action diagnostics are intentionally not
    included: they are refreshed on the interval.  Lifecycle, screen, episode,
    level, wave, terminal, and blocked state changes bypass the interval.
    """

    token = []
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        token.extend((key, _hashable(source.get(key))) for key in _SIGNIFICANT_KEYS if key in source)
        for path in _SIGNIFICANT_NESTED_PATHS:
            value: Any = source
            for part in path:
                if not isinstance(value, Mapping) or part not in value:
                    value = None
                    break
                value = value[part]
            if value is not None:
                token.append((".".join(path), _hashable(value)))
    return tuple(token)


def live_status_requires_immediate_write(payload: Mapping[str, Any]) -> bool:
    """Return whether a payload contains terminal, timeout, or freeze evidence."""

    status = str(payload.get("status") or "").strip().lower()
    if status in _FORCED_STATUS_VALUES:
        return True
    evidence = " ".join(
        str(payload.get(key) or "").strip().lower()
        for key in ("state", "done_reason", "terminal_reason", "blocked_reason")
    )
    return any(marker in evidence for marker in ("blocked", "complete", "error", "fail", "freeze", "timeout"))


@dataclass(frozen=True)
class LiveStatusWriteStats:
    attempts: int
    payload_builds: int
    writes: int
    skipped: int


class LiveStatusWriter:
    """Atomically publish latest-state JSON with interval/change coalescing."""

    def __init__(
        self,
        path: Optional[Path],
        *,
        min_interval_seconds: float = DEFAULT_LIVE_STATUS_INTERVAL_SECONDS,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.path = path
        self.min_interval_seconds = max(0.0, float(min_interval_seconds))
        self._monotonic = monotonic
        self.last_payload: Dict[str, Any] = {}
        self._last_significant_state: Optional[Hashable] = None
        self._last_write_at: Optional[float] = None
        self._write_index = 0
        self._last_warning_at = 0.0
        self._attempt_count = 0
        self._payload_build_count = 0
        self._write_count = 0
        self._skipped_count = 0

    @property
    def stats(self) -> LiveStatusWriteStats:
        return LiveStatusWriteStats(
            attempts=self._attempt_count,
            payload_builds=self._payload_build_count,
            writes=self._write_count,
            skipped=self._skipped_count,
        )

    def _is_due(self, now: float, significant_state: Hashable, *, force: bool) -> bool:
        return bool(
            force
            or self._last_write_at is None
            or significant_state != self._last_significant_state
            or now - self._last_write_at >= self.min_interval_seconds
        )

    def write(
        self,
        payload: Dict[str, Any],
        *,
        force: bool = False,
        significant_state: Optional[Hashable] = None,
    ) -> bool:
        """Publish ``payload`` when due and return whether disk was updated."""

        self._attempt_count += 1
        self._payload_build_count += 1
        now = self._monotonic()
        token = significant_state if significant_state is not None else live_status_significant_state(payload)
        force = bool(force or live_status_requires_immediate_write(payload))
        if not self._is_due(now, token, force=force):
            self._skipped_count += 1
            return False
        return self._publish(payload, now=now, significant_state=token)

    def write_lazy(
        self,
        payload_builder: Callable[[], Dict[str, Any]],
        *,
        significant_state: Hashable,
        force: bool = False,
    ) -> bool:
        """Build and publish a payload only when its cheap state token is due."""

        self._attempt_count += 1
        now = self._monotonic()
        if not self._is_due(now, significant_state, force=force):
            self._skipped_count += 1
            return False
        payload = payload_builder()
        self._payload_build_count += 1
        return self._publish(payload, now=now, significant_state=significant_state)

    def _publish(self, payload: Dict[str, Any], *, now: float, significant_state: Hashable) -> bool:
        self.last_payload = payload
        if self.path is None:
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_index += 1
        tmp_path = self.path.with_name(f"{self.path.name}.{os.getpid()}.{id(self)}.{self._write_index}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(20):
            try:
                os.replace(tmp_path, self.path)
                # A transition is published only after the atomic replace
                # succeeds.  A Windows file lock must not make an immediate
                # retry look redundant and suppress the only evidence.
                self._last_write_at = now
                self._last_significant_state = significant_state
                self._write_count += 1
                return True
            except PermissionError:
                time.sleep(0.025 + attempt * 0.005)
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        warning_now = self._monotonic()
        if warning_now - self._last_warning_at > 5.0:
            self._last_warning_at = warning_now
            print(f"[telemetry] warning: live status file is locked; skipped one write to {self.path}", flush=True)
        return False
