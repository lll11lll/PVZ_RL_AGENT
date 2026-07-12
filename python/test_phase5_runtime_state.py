"""Contracts for explicit base reset runtime ownership."""

from __future__ import annotations

from typing import Any

import pytest

from pvzrl_env import PvZEnvConfig, PvZGymEnv
from pvzrl_runtime_state import ResetRuntimeState


def _fresh_board() -> dict[str, Any]:
    return {
        "frameCount": 20,
        "rowCount": 5,
        "columnCount": 10,
        "boardFound": True,
        "gameplayReady": True,
        "actualGameplayReady": True,
        "seedSelectionActive": False,
        "seedSelectionPanelActive": False,
        "screenState": "gameplay",
        "nextStep": "play",
        "terminalHint": "running",
        "done": False,
        "over": False,
        "wave": 0,
        "killCount": 0,
        "plantCount": 0,
        "visiblePlantObjectCount": 0,
        "zombieCount": 0,
        "bulletCount": 0,
        "sun": 500,
        "logicalMowerCount": 5,
        "visibleMowerObjectCount": 5,
        "duplicateMowerRowCount": 0,
        "seedSlotCount": 1,
        "seedSlots": [{"slotIndex": 0, "plantType": 0}],
        "activeGameplayCardBankCount": 1,
        "legalActionCount": 2,
        "legalActions": [0, 1],
    }


def test_reset_runtime_state_projects_exact_compatibility_fields() -> None:
    state = ResetRuntimeState(
        generation_id=7,
        reason="win",
        phase="seed_selection",
        requires_seed_selection=True,
        saw_seed_selection=True,
        clicked_lets_rock=True,
        started_from_win=True,
        fixed_post_win_replay=True,
        fixed_terminal_reset=True,
        unsafe_gameplay_ready_before_seed_count=2,
    )
    assert state.compatibility_fields() == {
        "resetGenerationId": 7,
        "resetPhase": "seed_selection",
        "resetPhaseFinal": "seed_selection",
        "requireSeedSelectionThisReset": True,
        "sawSeedSelectionThisReset": True,
        "clickedLetsRockThisReset": True,
        "resetStartedFromLoss": False,
        "resetStartedFromWin": True,
        "fixedTrainPostWinReplayReset": True,
        "fixedTrainTerminalReset": True,
        "unsafeGameplayReadyBeforeSeedCount": 2,
    }
    state.set_phase("done")
    assert state.phase == "done"


def test_manual_reset_machine_uses_one_state_owner_and_preserves_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = PvZGymEnv(
        PvZEnvConfig(
            row_count=5,
            column_count=10,
            reset_wait_timeout=1.0,
            reset_poll_seconds=0.01,
        )
    )
    observation = _fresh_board()
    env._reset_generation_id = 9
    monkeypatch.setattr(
        env,
        "observe",
        lambda **_kwargs: observation,
    )
    monkeypatch.setattr(env, "reset_cleanup", lambda **_kwargs: {"ok": True})
    monkeypatch.setattr(
        env,
        "_wait_for_cleanup_valid",
        lambda **_kwargs: (observation, True, "cleanup_ok"),
    )
    monkeypatch.setattr(
        env,
        "_wait_for_post_reset_playable",
        lambda **_kwargs: (observation, True, "playable_ok"),
    )
    monkeypatch.setattr(env, "legal_actions", lambda *_args, **_kwargs: [0, 1])
    reset_result: dict[str, Any] = {
        "ok": True,
        "methodUsed": "state_machine",
        "stages": [],
        "resetGenerationId": 9,
    }
    try:
        result = env._reset_state_machine(
            reset_result,
            allow_active_gameplay_reset=True,
            reset_reason="manual",
        )
        assert result is observation
        assert reset_result["resetSuccess"] is True
        assert reset_result["resetGenerationId"] == 9
        assert reset_result["resetPhase"] == "done"
        assert reset_result["resetPhaseFinal"] == "done"
        assert reset_result["requireSeedSelectionThisReset"] is False
        assert reset_result["sawSeedSelectionThisReset"] is False
        assert reset_result["clickedLetsRockThisReset"] is False
        assert reset_result["postResetLegalActionCount"] == 2
        assert reset_result["cleanupValidation"] == "cleanup_ok"
        assert reset_result["postResetPlayableValidation"] == "playable_ok"
        assert env._reset_requires_seed_flow is False
        assert env._saw_seed_selection_this_reset is False
        assert env._clicked_lets_rock_this_reset is False
    finally:
        env.client.close()
