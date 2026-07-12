"""Focused contracts for the shadow-only Generalist progression reducer."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from pvzrl_generalist_progression import (
    GeneralistEpisodeOutcome,
    GeneralistProgressionConfig,
    GeneralistProgressionState,
    begin_generalist_attempt,
    fresh_generalist_progression,
    reduce_generalist_episode,
)


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
