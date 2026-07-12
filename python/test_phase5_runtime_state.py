"""Parity contracts for the explicit Phase 5 environment runtime owners."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from pvzrl_env import PvZEnvConfig, PvZGymEnv
from pvzrl_runtime_state import EpisodeRuntimeState, ResetRuntimeState, WatchdogRuntimeState
from test_refactor_support import make_wrapper, observation_for_wrapper


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


def test_episode_runtime_begin_resets_episode_fields_but_preserves_global_steps() -> None:
    state = EpisodeRuntimeState(global_step_count=37)
    state.begin(
        start_kills=4,
        start_mowers=5,
        reset_success=False,
        reset_seconds=1.25,
    )
    state.record_step(2.5)
    state.plants_placed = 2
    state.illegal_actions = 1
    state.sun_spent = 125
    state.legal_action_total = 14

    state.begin(
        start_kills=9,
        start_mowers=4,
        reset_success=True,
        reset_seconds=0.5,
    )

    assert state.index == 1
    assert state.global_step_count == 38
    assert state.step_count == 0
    assert state.reward_total == 0.0
    assert state.plants_placed == 0
    assert state.illegal_actions == 0
    assert state.sun_spent == 0
    assert state.legal_action_total == 0
    assert state.start_kills == 9
    assert state.start_mowers == 4
    assert state.reset_success is True
    assert state.reset_seconds == 0.5


def test_watchdog_runtime_summary_preserves_existing_fields_and_nearest_rank_p95() -> None:
    state = WatchdogRuntimeState.empty()
    state.record(
        0.1,
        timed_out=False,
        action_type="wait",
        plant="",
        fusion_pair="",
        row=-1,
        column=-1,
        screen_state="gameplay",
        level=3,
    )
    state.record(
        0.2,
        timed_out=False,
        action_type="plant",
        plant="Peashooter",
        fusion_pair="",
        row=1,
        column=2,
        screen_state="gameplay",
        level=3,
    )
    state.record(
        1.0,
        timed_out=True,
        action_type="fusion",
        plant="SunFlower",
        fusion_pair="SunFlower+Peashooter",
        row=2,
        column=3,
        screen_state="gameplay",
        level=3,
    )

    assert state.compatibility_summary() == {
        "mean_action_duration_seconds": pytest.approx(1.3 / 3.0),
        "max_action_duration_seconds": 1.0,
        "p95_action_duration_seconds": 1.0,
        "action_freeze_count": 1,
        "freezes_by_action_type": {"fusion": 1},
        "freezes_by_plant": {"SunFlower": 1},
        "freezes_by_fusion_pair": {"SunFlower+Peashooter": 1},
        "freezes_by_grid_coordinate": {"2,3": 1},
        "freezes_by_screen_state": {"gameplay": 1},
        "freezes_by_level": {"3": 1},
    }


def test_reset_and_adventure_initialization_share_core_accounting_without_drift() -> None:
    wrapper = make_wrapper(identity=False)
    reset_observation = observation_for_wrapper(wrapper)
    reset_observation.update(frameCount=10, killCount=3, logicalMowerCount=4)
    reset_payload = {
        "ok": True,
        "resetSuccess": False,
        "reset_ms": 1250,
        "resetRewardUiCleanupCount": 2,
        "resetRewardUiCleanupBlockedCount": 1,
        "resetAfterFalseRewardSignalCount": 3,
        "blockedCleanupDuringGameplayCount": 4,
        "suspiciousCleanupRewardUiCount": 5,
    }
    try:
        wrapper.episode_state.global_step_count = 22
        wrapper.watchdog_state.record(
            0.4,
            timed_out=True,
            action_type="wait",
            plant="",
            fusion_pair="",
            row=-1,
            column=-1,
            screen_state="gameplay",
            level=1,
        )
        wrapper._initialize_episode_accounting(
            reset_observation,
            reset_payload,
            source="test_reset",
            include_reset_safety_fields=True,
        )
        assert wrapper.episode_state == EpisodeRuntimeState(
            index=0,
            step_count=0,
            global_step_count=22,
            reward_total=0.0,
            plants_placed=0,
            illegal_actions=0,
            sun_spent=0,
            legal_action_total=0,
            start_kills=3,
            start_mowers=4,
            reset_success=False,
            reset_seconds=1.25,
        )
        assert wrapper._episode_reset_reward_ui_cleanup_count == 2
        assert wrapper._episode_reset_reward_ui_cleanup_blocked_count == 1
        assert wrapper._episode_reset_after_false_reward_signal_count == 3
        assert wrapper._episode_blocked_cleanup_during_gameplay_count == 4
        assert wrapper._episode_suspicious_cleanup_reward_ui_count == 5
        assert wrapper._action_diagnostic_summary()["action_freeze_count"] == 0

        wrapper.episode_state.record_step(7.0)
        wrapper.episode_state.plants_placed = 9
        adventure_observation = copy.deepcopy(reset_observation)
        adventure_observation.update(frameCount=11, killCount=8, logicalMowerCount=3)
        wrapper._initialize_episode_accounting(
            adventure_observation,
            {
                "ok": True,
                "timeToPlayableSeconds": 0.75,
                "resetRewardUiCleanupBlockedCount": 99,
            },
            source="test_adventure",
            include_reset_safety_fields=False,
        )
        assert wrapper.episode_state.index == 1
        assert wrapper.episode_state.global_step_count == 23
        assert wrapper.episode_state.step_count == 0
        assert wrapper.episode_state.reward_total == 0.0
        assert wrapper.episode_state.plants_placed == 0
        assert wrapper.episode_state.start_kills == 8
        assert wrapper.episode_state.start_mowers == 3
        assert wrapper.episode_state.reset_success is True
        assert wrapper.episode_state.reset_seconds == 0.75
        assert wrapper._episode_reset_reward_ui_cleanup_blocked_count == 99
        assert wrapper._episode_blocked_cleanup_during_gameplay_count == 0
        assert wrapper._episode_suspicious_cleanup_reward_ui_count == 0
    finally:
        wrapper.base.client.close()


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
    monkeypatch.setattr(env, "observe", lambda **_kwargs: observation)
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
