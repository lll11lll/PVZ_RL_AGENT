"""Explicit mutable runtime owners with compatibility projections.

These records replace clusters of parallel scalar fields.  They deliberately
contain no bridge, filesystem, policy, or UI behavior.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(slots=True)
class EpisodeRuntimeState:
    """Core wrapper episode accounting, excluding specialized diagnostics."""

    index: int = -1
    step_count: int = 0
    global_step_count: int = 0
    reward_total: float = 0.0
    plants_placed: int = 0
    illegal_actions: int = 0
    sun_spent: int = 0
    legal_action_total: int = 0
    start_kills: int = 0
    start_mowers: int = 0
    reset_success: bool = True
    reset_seconds: float = 0.0

    def begin(
        self,
        *,
        start_kills: int,
        start_mowers: int,
        reset_success: bool,
        reset_seconds: float,
    ) -> None:
        self.index += 1
        self.step_count = 0
        self.reward_total = 0.0
        self.plants_placed = 0
        self.illegal_actions = 0
        self.sun_spent = 0
        self.legal_action_total = 0
        self.start_kills = int(start_kills)
        self.start_mowers = int(start_mowers)
        self.reset_success = bool(reset_success)
        self.reset_seconds = float(reset_seconds)

    def record_step(self, reward: float) -> None:
        self.step_count += 1
        self.global_step_count += 1
        self.reward_total += float(reward)


@dataclass(slots=True)
class WatchdogRuntimeState:
    """Buffered per-episode action timing and freeze classifications."""

    action_durations: List[float]
    freeze_count: int
    freezes_by_type: Counter[str]
    freezes_by_plant: Counter[str]
    freezes_by_fusion_pair: Counter[str]
    freezes_by_grid: Counter[str]
    freezes_by_screen_state: Counter[str]
    freezes_by_level: Counter[str]

    @classmethod
    def empty(cls) -> "WatchdogRuntimeState":
        return cls(
            action_durations=[],
            freeze_count=0,
            freezes_by_type=Counter(),
            freezes_by_plant=Counter(),
            freezes_by_fusion_pair=Counter(),
            freezes_by_grid=Counter(),
            freezes_by_screen_state=Counter(),
            freezes_by_level=Counter(),
        )

    def reset_episode(self) -> None:
        self.action_durations = []
        self.freeze_count = 0
        self.freezes_by_type = Counter()
        self.freezes_by_plant = Counter()
        self.freezes_by_fusion_pair = Counter()
        self.freezes_by_grid = Counter()
        self.freezes_by_screen_state = Counter()
        self.freezes_by_level = Counter()

    def record(
        self,
        duration: float,
        *,
        timed_out: bool,
        action_type: str,
        plant: str,
        fusion_pair: str,
        row: Any,
        column: Any,
        screen_state: Any,
        level: Any,
    ) -> None:
        self.action_durations.append(float(duration))
        if not timed_out:
            return
        self.freeze_count += 1
        self.freezes_by_type[action_type] += 1
        self.freezes_by_plant[plant or "unknown"] += 1
        self.freezes_by_fusion_pair[fusion_pair or "not_fusion"] += 1
        self.freezes_by_grid[f"{row},{column}"] += 1
        self.freezes_by_screen_state[str(screen_state or "unknown")] += 1
        self.freezes_by_level[str(level or "unknown")] += 1

    def compatibility_summary(self) -> Dict[str, Any]:
        durations = sorted(self.action_durations)
        if durations:
            p95_index = min(
                len(durations) - 1,
                max(0, int(math.ceil(0.95 * len(durations))) - 1),
            )
            mean_duration = float(sum(durations) / len(durations))
            max_duration = float(durations[-1])
            p95_duration = float(durations[p95_index])
        else:
            mean_duration = max_duration = p95_duration = 0.0
        return {
            "mean_action_duration_seconds": mean_duration,
            "max_action_duration_seconds": max_duration,
            "p95_action_duration_seconds": p95_duration,
            "action_freeze_count": int(self.freeze_count),
            "freezes_by_action_type": dict(self.freezes_by_type),
            "freezes_by_plant": dict(self.freezes_by_plant),
            "freezes_by_fusion_pair": dict(self.freezes_by_fusion_pair),
            "freezes_by_grid_coordinate": dict(self.freezes_by_grid),
            "freezes_by_screen_state": dict(self.freezes_by_screen_state),
            "freezes_by_level": dict(self.freezes_by_level),
        }


@dataclass(slots=True)
class ResetRuntimeState:
    generation_id: int
    reason: str
    phase: str = "idle"
    requires_seed_selection: bool = False
    saw_seed_selection: bool = False
    clicked_lets_rock: bool = False
    started_from_loss: bool = False
    started_from_win: bool = False
    fixed_post_win_replay: bool = False
    fixed_terminal_reset: bool = False
    unsafe_gameplay_ready_before_seed_count: int = 0

    def set_phase(self, phase: str) -> None:
        self.phase = str(phase)

    def compatibility_fields(self) -> Dict[str, Any]:
        return {
            "resetGenerationId": int(self.generation_id),
            "resetPhase": str(self.phase),
            "resetPhaseFinal": str(self.phase),
            "requireSeedSelectionThisReset": bool(self.requires_seed_selection),
            "sawSeedSelectionThisReset": bool(self.saw_seed_selection),
            "clickedLetsRockThisReset": bool(self.clicked_lets_rock),
            "resetStartedFromLoss": bool(self.started_from_loss),
            "resetStartedFromWin": bool(self.started_from_win),
            "fixedTrainPostWinReplayReset": bool(self.fixed_post_win_replay),
            "fixedTrainTerminalReset": bool(self.fixed_terminal_reset),
            "unsafeGameplayReadyBeforeSeedCount": int(
                self.unsafe_gameplay_ready_before_seed_count
            ),
        }


__all__ = ["EpisodeRuntimeState", "ResetRuntimeState", "WatchdogRuntimeState"]
