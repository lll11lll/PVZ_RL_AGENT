"""Focused contracts for the shadow-only Generalist progression reducer."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

import pvzrl_adventure_generalist as generalist_module
from pvzrl_adventure_generalist import AdventureGeneralistTrainingEnv
from pvzrl_generalist_progression import (
    GeneralistEpisodeOutcome,
    GeneralistProgressionConfig,
    GeneralistProgressionState,
    begin_generalist_attempt,
    fresh_generalist_progression,
    reduce_generalist_episode,
)
from test_adventure_generalist_14slot_identity import fake_generalist_env


def _config(*, required: int = 1, attempts: int = 3, levels: int = 4):
    return GeneralistProgressionConfig.normalized(
        adventure_start_level=4,
        max_adventure_levels=levels,
        max_attempts_per_level=attempts,
        frontier_win_streak_required=required,
    )


def _finish(
    state: GeneralistProgressionState,
    config: GeneralistProgressionConfig,
    result: str,
    *,
    source: str = "frontier",
    replay_succeeded: bool | None = None,
    replay_reason: str = "",
):
    return reduce_generalist_episode(
        state,
        GeneralistEpisodeOutcome(
            result=result,
            episode_level=state.current_level,
            sample_source=source,
            same_level_replay_succeeded=replay_succeeded,
            same_level_replay_reason=replay_reason,
        ),
        config,
    )


def test_n1_win_promotes_and_retains_last_episode_mastery_latches() -> None:
    config = _config(required=1)
    started = begin_generalist_attempt(fresh_generalist_progression(config))

    transition = _finish(started, config, "win")

    assert transition.episode_attempt == 1
    assert transition.post_win_decision == "advance_next_level"
    assert transition.post_win_transition_allowed is True
    assert transition.collect_post_win_transition is True
    assert transition.same_level_replay_requested is False
    assert transition.state.current_level == 5
    assert transition.state.current_attempt == 0
    assert transition.state.frontier_win_streak == 0
    assert transition.state.cleared_levels == (4,)
    assert transition.state.frontier_mastered_levels == (4,)
    assert transition.state.last_episode.mastery_ready is True
    assert transition.state.last_episode.promoted is True
    assert transition.state.last_episode.reset_reason == "promoted"

    next_attempt = begin_generalist_attempt(transition.state)
    assert next_attempt.last_episode is transition.state.last_episode
    assert next_attempt.current_attempt == 1


def test_n3_requires_same_level_replays_then_promotes() -> None:
    config = _config(required=3)
    state = fresh_generalist_progression(config)

    first = _finish(
        begin_generalist_attempt(state),
        config,
        "win",
        replay_succeeded=True,
    )
    assert first.post_win_decision == "replay_same_level"
    assert first.same_level_replay_requested is True
    assert first.state.current_level == 4
    assert first.state.frontier_win_streak == 1
    assert first.state.last_episode.mastery_ready is False

    second = _finish(
        begin_generalist_attempt(first.state),
        config,
        "post_win_pending",
        source="frontier_mastery_replay",
        replay_succeeded=True,
    )
    assert second.same_level_replay_requested is True
    assert second.state.frontier_win_streak == 2

    third = _finish(
        begin_generalist_attempt(second.state),
        config,
        "win",
        source="frontier_mastery_replay",
    )
    assert third.post_win_decision == "advance_next_level"
    assert third.same_level_replay_requested is False
    assert third.collect_post_win_transition is True
    assert third.state.current_level == 5
    assert third.state.frontier_win_streak == 0
    assert third.state.frontier_mastered_levels == (4,)
    assert third.state.last_episode.mastery_ready is True
    assert third.state.last_episode.promoted is True


@pytest.mark.parametrize(
    ("result", "expected_reason"),
    [
        ("loss", "loss"),
        ("timeout", "timeout"),
        ("env_corruption", "env_corruption"),
    ],
)
def test_frontier_failures_reset_pending_mastery_streak(
    result: str,
    expected_reason: str,
) -> None:
    config = _config(required=3)
    state = GeneralistProgressionState(
        current_level=4,
        current_attempt=1,
        frontier_win_streak=2,
    )

    transition = _finish(
        state,
        config,
        result,
        source="frontier_mastery_replay",
    )

    assert transition.state.frontier_win_streak == 0
    assert transition.state.last_episode.reset_reason == expected_reason


def test_non_frontier_episode_does_not_change_frontier_mastery_or_level() -> None:
    config = _config(required=3)
    state = GeneralistProgressionState(
        current_level=4,
        current_attempt=2,
        frontier_win_streak=1,
        cleared_levels=(2,),
    )

    maintenance_win = _finish(state, config, "win", source="maintenance")
    assert maintenance_win.post_win_decision == "hold_frontier"
    assert maintenance_win.state.current_level == 4
    assert maintenance_win.state.current_attempt == 0
    assert maintenance_win.state.frontier_win_streak == 1
    assert maintenance_win.state.cleared_levels == (2, 4)
    assert maintenance_win.state.last_episode.mastery_ready is False
    assert maintenance_win.state.last_episode.promoted is False
    assert maintenance_win.state.last_episode.reset_reason == ""

    maintenance_loss = _finish(
        GeneralistProgressionState(
            current_level=4,
            current_attempt=1,
            frontier_win_streak=1,
        ),
        config,
        "loss",
        source="maintenance",
    )
    assert maintenance_loss.state.frontier_win_streak == 1
    assert maintenance_loss.state.last_episode.reset_reason == ""


def test_max_level_clamp_still_reports_promotion() -> None:
    config = _config(required=1, levels=1)
    state = begin_generalist_attempt(fresh_generalist_progression(config))

    transition = _finish(state, config, "win")

    assert transition.state.current_level == config.maximum_level == 4
    assert transition.state.frontier_mastered_levels == (4,)
    assert transition.state.last_episode.mastery_ready is True
    assert transition.state.last_episode.promoted is True
    assert transition.state.last_episode.reset_reason == "promoted"


def test_attempt_rollover_is_limited_to_loss_and_timeout() -> None:
    config = _config(required=3, attempts=2)
    at_limit = GeneralistProgressionState(current_level=4, current_attempt=2)

    assert _finish(at_limit, config, "loss").state.current_attempt == 0
    assert _finish(at_limit, config, "timeout").state.current_attempt == 0
    assert _finish(at_limit, config, "env_corruption").state.current_attempt == 2
    assert _finish(at_limit, config, "unknown").state.current_attempt == 2

    below_limit = GeneralistProgressionState(current_level=4, current_attempt=1)
    assert _finish(below_limit, config, "loss").state.current_attempt == 1


def test_failed_same_level_replay_requests_recovery_without_losing_streak() -> None:
    config = _config(required=3)
    started = begin_generalist_attempt(fresh_generalist_progression(config))

    transition = _finish(
        started,
        config,
        "win",
        replay_succeeded=False,
        replay_reason="win_replay_reset_failed",
    )

    assert transition.same_level_replay_requested is True
    assert transition.replay_recovery_required is True
    assert transition.post_win_blocked_reason == "win_replay_reset_failed"
    assert transition.state.current_level == 4
    assert transition.state.frontier_win_streak == 1
    assert transition.state.frontier_replay_supported is False
    assert transition.state.frontier_replay_blocked_reason == (
        "win_replay_reset_failed"
    )
    assert transition.state.frontier_replay_recovery_required is True
    assert transition.state.last_episode.reset_reason == (
        "same_level_replay_recovery_required"
    )


def test_ppo_resume_starts_fresh_progress_and_records_are_immutable() -> None:
    config = _config(required=3)
    scratch = fresh_generalist_progression(config, ppo_resume=False)
    resumed = fresh_generalist_progression(config, ppo_resume=True)

    assert resumed == scratch
    assert resumed.current_level == 4
    assert resumed.current_attempt == 0
    assert resumed.frontier_win_streak == 0
    assert resumed.cleared_levels == ()
    assert resumed.frontier_mastered_levels == ()
    with pytest.raises(FrozenInstanceError):
        resumed.current_level = 6  # type: ignore[misc]


def test_state_normalizes_mutable_level_inputs_without_aliasing() -> None:
    cleared = [2, 4]
    mastered = [2]
    state = GeneralistProgressionState(
        current_level=4,
        cleared_levels=cleared,  # type: ignore[arg-type]
        frontier_mastered_levels=mastered,  # type: ignore[arg-type]
    )
    cleared.append(9)
    mastered.append(9)
    assert state.cleared_levels == (2, 4)
    assert state.frontier_mastered_levels == (2,)


def test_successful_replay_clears_recovery_but_retains_diagnostic_reason() -> None:
    config = _config(required=3)
    state = GeneralistProgressionState(
        current_level=4,
        current_attempt=1,
        frontier_win_streak=1,
        frontier_replay_supported=False,
        frontier_replay_blocked_reason="earlier_failure",
        frontier_replay_recovery_required=True,
    )
    transition = _finish(
        state,
        config,
        "win",
        source="frontier_mastery_replay",
        replay_succeeded=True,
    )
    assert transition.state.frontier_replay_supported is True
    assert transition.state.frontier_replay_recovery_required is False
    assert transition.state.frontier_replay_blocked_reason == "earlier_failure"


def test_failed_replay_uses_stable_default_reason() -> None:
    transition = _finish(
        begin_generalist_attempt(fresh_generalist_progression(_config(required=3))),
        _config(required=3),
        "win",
        replay_succeeded=False,
    )
    assert transition.post_win_blocked_reason == (
        "frontier_win_streak_requires_same_level_replay_support"
    )


def test_wrapper_compatibility_fields_project_from_one_canonical_record() -> None:
    env = AdventureGeneralistTrainingEnv.__new__(AdventureGeneralistTrainingEnv)
    env.adventure_start_level = 4
    env.max_adventure_levels = 3
    env.max_attempts_per_level = 2
    env.frontier_win_streak_required = 3
    env.current_level = 5
    env.current_attempt = 2
    env.frontier_win_streak = 1
    env.cleared_levels = [2, 4]
    env.frontier_mastered_levels = [2]
    env.frontier_replay_supported = False
    env.frontier_replay_blocked_reason = "earlier_failure"
    env.frontier_mastery_ready = True
    env.frontier_promoted_this_episode = True
    env.frontier_mastery_reset_reason = "promoted"

    state = env._progression_state_value()
    assert state.current_level == 5
    assert state.current_attempt == 2
    assert state.frontier_win_streak == 1
    assert state.cleared_levels == (2, 4)
    assert state.frontier_mastered_levels == (2,)
    assert state.frontier_replay_supported is False
    assert state.frontier_replay_blocked_reason == "earlier_failure"
    assert state.last_episode.mastery_ready is True
    assert state.last_episode.promoted is True
    assert state.last_episode.reset_reason == "promoted"

    projected_cleared = env.cleared_levels
    projected_mastered = env.frontier_mastered_levels
    projected_cleared.append(9)
    projected_mastered.append(9)
    assert env.cleared_levels == [2, 4]
    assert env.frontier_mastered_levels == [2]
    env._assert_progression_projection()


def test_replay_hook_observes_legacy_pre_effect_progression_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    env = fake_generalist_env(tmp_path)
    env.current_level = 4
    env.current_attempt = 2
    env.frontier_win_streak_required = 3
    env.frontier_win_streak = 1
    env.frontier_mastery_reset_reason = "prior_timeout"
    env.current_sample_source = "frontier_mastery_replay"
    env._safe_adventure_state = lambda: {"currentAdventureLevel": 4}
    seen = {}

    def replay_hook(*_args, **_kwargs):
        seen.update(
            current_level=env.current_level,
            current_attempt=env.current_attempt,
            frontier_win_streak=env.frontier_win_streak,
            cleared_levels=env.cleared_levels,
            frontier_mastered_levels=env.frontier_mastered_levels,
            reset_reason=env.frontier_mastery_reset_reason,
            context_streak=env.context.get("frontier_win_streak"),
        )
        return True, ""

    monkeypatch.setattr(generalist_module, "replay_current_level_after_validation_win", replay_hook)
    monkeypatch.setattr(generalist_module, "build_live_status", lambda *_args, **_kwargs: {})
    env._finish_episode({"episode_summary": {"done_reason": "win"}})

    assert seen == {
        "current_level": 4,
        "current_attempt": 0,
        "frontier_win_streak": 2,
        "cleared_levels": [4],
        "frontier_mastered_levels": [],
        "reset_reason": "prior_timeout",
        "context_streak": 2,
    }
    assert env.current_level == 4
    assert env.current_attempt == 0
    assert env.frontier_win_streak == 2
    assert env.frontier_mastery_reset_reason == ""


def test_promotion_collect_hook_observes_threshold_streak_before_final_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    env = fake_generalist_env(tmp_path)
    env.current_level = 4
    env.current_attempt = 2
    env.frontier_win_streak_required = 3
    env.frontier_win_streak = 2
    env.cleared_levels = [2]
    env.frontier_mastered_levels = [2]
    env.frontier_mastery_reset_reason = "prior_loss"
    env.current_sample_source = "frontier_mastery_replay"
    seen = {}

    def collect_hook(*_args, **_kwargs):
        seen.update(
            current_level=env.current_level,
            current_attempt=env.current_attempt,
            frontier_win_streak=env.frontier_win_streak,
            cleared_levels=env.cleared_levels,
            frontier_mastered_levels=env.frontier_mastered_levels,
            reset_reason=env.frontier_mastery_reset_reason,
            context_streak=env.context.get("frontier_win_streak"),
            context_ready=env.context.get("frontier_mastery_ready"),
        )
        return (
            {"screenState": "seed_selection"},
            True,
            {},
            [],
            [],
            "",
            {"post_win_transition_completed": True},
        )

    monkeypatch.setattr(generalist_module, "collect_post_win_unlocks", collect_hook)
    monkeypatch.setattr(generalist_module, "build_live_status", lambda *_args, **_kwargs: {})
    env._finish_episode({"episode_summary": {"done_reason": "win"}})

    assert seen == {
        "current_level": 4,
        "current_attempt": 0,
        "frontier_win_streak": 3,
        "cleared_levels": [2, 4],
        "frontier_mastered_levels": [2],
        "reset_reason": "prior_loss",
        "context_streak": 3,
        "context_ready": True,
    }
    assert env.current_level == 5
    assert env.current_attempt == 0
    assert env.frontier_win_streak == 0
    assert env.frontier_mastered_levels == [2, 4]
    assert env.frontier_mastery_ready is True
    assert env.frontier_promoted_this_episode is True


def test_replay_hook_exception_leaves_legacy_pre_effect_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    env = fake_generalist_env(tmp_path)
    env.current_level = 4
    env.current_attempt = 2
    env.frontier_win_streak_required = 3
    env.frontier_win_streak = 1
    env.current_sample_source = "frontier"

    def replay_hook(*_args, **_kwargs):
        raise RuntimeError("simulated_replay_failure")

    monkeypatch.setattr(generalist_module, "replay_current_level_after_validation_win", replay_hook)
    with pytest.raises(RuntimeError, match="simulated_replay_failure"):
        env._finish_episode({"episode_summary": {"done_reason": "win"}})

    assert env.current_level == 4
    assert env.current_attempt == 0
    assert env.frontier_win_streak == 2
    assert env.cleared_levels == [4]
    assert env.frontier_mastered_levels == []
    assert env.frontier_mastery_ready is False
    assert env.frontier_promoted_this_episode is False


def test_reset_clears_replay_recovery_latch_before_starting_attempt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    env = fake_generalist_env(tmp_path)
    env.current_attempt = 0
    env.strict_startup_validation = True
    env._startup_validation_completed = True
    env._set_progression_fields(frontier_replay_recovery_required=True)
    env.context["frontier_replay_recovery_required"] = True
    env._base_fusion_policy = "none"
    env.config.fusion_policy = "none"
    env.base = SimpleNamespace(config=SimpleNamespace(fusion_policy="none"))
    env._sample_curriculum_mode = lambda: "frontier"
    env._sample_source = lambda: "frontier"
    validation_phases = []
    env.validate_startup_state = lambda *, phase, raise_on_failure: validation_phases.append(phase) or {"ok": True}
    env.start_episode_from_observation = lambda observation, reset_info: (observation, reset_info)
    monkeypatch.setattr(generalist_module, "build_live_status", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        generalist_module,
        "prepare_adventure_gameplay",
        lambda *_args, **_kwargs: ({"frame": 1}, {"reset": True}, ""),
    )

    observation, reset_info = env.reset()

    assert validation_phases == ["same_level_replay_recovery"]
    assert env._progression_state_value().frontier_replay_recovery_required is False
    assert env.context["frontier_replay_recovery_required"] is False
    assert env.current_attempt == 1
    assert observation == {"frame": 1}
    assert reset_info == {"reset": True}


def test_empty_seed_screen_does_not_expand_loadout_from_unlock_fallback(tmp_path) -> None:
    env = fake_generalist_env(tmp_path)
    expected_loadout = list(env.current_loadout)
    env.curriculum.record_unlocked(["CherryBomb"], episode_index=1)
    env.curriculum.episode_index = 2
    env.episode_index = 2
    env.confirmed_unlock_event_seeds = ["CherryBomb"]
    env._apply_loadout = lambda _loadout: None

    selected, blocked = env._on_seed_selection_screen(
        {
            "screenState": "seed_selection",
            "isSeedSelectionScreen": True,
            "visibleSeedCardNames": [],
            "availableSeedNames": [],
            "selectedSeedNames": [],
            "unlockedSeedNames": ["SunFlower", "Peashooter", "CherryBomb"],
            "seedSlotCapacity": 4,
        },
        expected_loadout,
    )

    assert blocked == ""
    assert selected == expected_loadout
    assert env.context["raw_selectable_seeds"] == []
    assert env.context["selected_loadout"] == expected_loadout
