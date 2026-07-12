"""Explicit mutable runtime owners with compatibility projections.

These records replace clusters of parallel scalar fields.  They deliberately
contain no bridge, filesystem, policy, or UI behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


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


__all__ = ["ResetRuntimeState"]
