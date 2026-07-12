"""Pure Adventure Generalist progression transitions.

This module is intentionally shadow-only during the Phase 5 migration.  It
models the progression mutations currently performed by
``AdventureGeneralistTrainingEnv`` without importing that wrapper or carrying
out bridge requests, post-win clicks, telemetry writes, or PPO operations.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple


FRONTIER_SAMPLE_SOURCES = frozenset({"frontier", "frontier_mastery_replay"})
REPLAY_RECOVERY_RESET_REASON = "same_level_replay_recovery_required"
DEFAULT_REPLAY_BLOCKED_REASON = (
    "frontier_win_streak_requires_same_level_replay_support"
)


@dataclass(frozen=True, slots=True)
class GeneralistProgressionConfig:
    """Normalized values that control one Generalist progression run."""

    adventure_start_level: int
    max_adventure_levels: int
    max_attempts_per_level: int
    frontier_win_streak_required: int

    @classmethod
    def normalized(
        cls,
        *,
        adventure_start_level: int,
        max_adventure_levels: int,
        max_attempts_per_level: int,
        frontier_win_streak_required: int,
    ) -> "GeneralistProgressionConfig":
        return cls(
            adventure_start_level=max(1, int(adventure_start_level)),
            max_adventure_levels=max(1, int(max_adventure_levels)),
            max_attempts_per_level=max(1, int(max_attempts_per_level)),
            frontier_win_streak_required=max(
                1, int(frontier_win_streak_required)
            ),
        )

    @property
    def maximum_level(self) -> int:
        return self.adventure_start_level + self.max_adventure_levels - 1


@dataclass(frozen=True, slots=True)
class GeneralistMasteryLatches:
    """Diagnostics retained from the most recently completed episode."""

    mastery_ready: bool = False
    promoted: bool = False
    reset_reason: str = ""
    sample_source: str = "frontier"


@dataclass(frozen=True, slots=True)
class GeneralistProgressionState:
    """Immutable wrapper-owned progression state between episodes."""

    current_level: int
    current_attempt: int = 0
    frontier_win_streak: int = 0
    cleared_levels: Tuple[int, ...] = ()
    frontier_mastered_levels: Tuple[int, ...] = ()
    frontier_replay_supported: bool = True
    frontier_replay_blocked_reason: str = ""
    frontier_replay_recovery_required: bool = False
    last_episode: GeneralistMasteryLatches = GeneralistMasteryLatches()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cleared_levels",
            tuple(int(level) for level in self.cleared_levels),
        )
        object.__setattr__(
            self,
            "frontier_mastered_levels",
            tuple(int(level) for level in self.frontier_mastered_levels),
        )


@dataclass(frozen=True, slots=True)
class GeneralistEpisodeOutcome:
    """Authoritative facts available when a Generalist episode finishes.

    ``same_level_replay_succeeded`` is relevant only when a frontier win has
    not yet met an N>1 mastery threshold.  ``None`` means the pure reducer is
    requesting that external replay operation but has not received its result.
    """

    result: str
    episode_level: int
    sample_source: str
    same_level_replay_succeeded: Optional[bool] = None
    same_level_replay_reason: str = ""


@dataclass(frozen=True, slots=True)
class GeneralistProgressionTransition:
    """New state plus effect requests that the pure reducer cannot execute."""

    state: GeneralistProgressionState
    episode_level: int
    episode_attempt: int
    is_win_like: bool
    is_frontier_mastery_attempt: bool
    post_win_decision: str
    post_win_transition_allowed: bool
    collect_post_win_transition: bool
    same_level_replay_requested: bool
    replay_recovery_required: bool
    post_win_blocked_reason: str


def fresh_generalist_progression(
    config: GeneralistProgressionConfig,
    *,
    ppo_resume: bool = False,
) -> GeneralistProgressionState:
    """Return fresh wrapper progress, including for a PPO checkpoint resume.

    PPO archives contain policy/training state, not this wrapper's Adventure
    frontier.  The flag is accepted to make that compatibility boundary
    explicit: both scratch and checkpoint-warm-start runs begin with identical
    progression state.
    """

    del ppo_resume
    return GeneralistProgressionState(current_level=config.adventure_start_level)


def begin_generalist_attempt(
    state: GeneralistProgressionState,
) -> GeneralistProgressionState:
    """Start one episode while retaining the prior episode's latch values."""

    return replace(state, current_attempt=int(state.current_attempt) + 1)


def _append_unique(values: Tuple[int, ...], value: int) -> Tuple[int, ...]:
    return values if value in values else (*values, value)


def _failure_reset_reason(result: str) -> str:
    if result == "loss":
        return "loss"
    if result == "timeout":
        return "timeout"
    if result == "env_corruption":
        return "env_corruption"
    return result or "failure"


def reduce_generalist_episode(
    state: GeneralistProgressionState,
    outcome: GeneralistEpisodeOutcome,
    config: GeneralistProgressionConfig,
) -> GeneralistProgressionTransition:
    """Reduce one terminal episode using the current Generalist semantics."""

    result = str(outcome.result or "unknown")
    episode_level = int(outcome.episode_level)
    sample_source = str(outcome.sample_source or "frontier")
    episode_attempt = int(state.current_attempt)
    frontier_level_before = int(state.current_level)
    is_win_like = result in {"win", "post_win_pending"}
    is_frontier_mastery_attempt = (
        sample_source in FRONTIER_SAMPLE_SOURCES
        and episode_level == frontier_level_before
    )

    current_level = frontier_level_before
    current_attempt = episode_attempt
    frontier_win_streak = int(state.frontier_win_streak)
    cleared_levels = state.cleared_levels
    mastered_levels = state.frontier_mastered_levels
    replay_supported = bool(state.frontier_replay_supported)
    replay_blocked_reason = str(state.frontier_replay_blocked_reason or "")
    replay_recovery_required = bool(
        state.frontier_replay_recovery_required
    )
    mastery_ready = False
    promoted = False
    reset_reason = ""
    post_win_decision = ""
    transition_allowed = False
    collect_post_win_transition = False
    same_level_replay_requested = False
    post_win_blocked_reason = ""

    if is_win_like:
        cleared_levels = _append_unique(cleared_levels, episode_level)
        current_attempt = 0
        if is_frontier_mastery_attempt:
            frontier_win_streak += 1
            mastery_ready = (
                frontier_win_streak >= config.frontier_win_streak_required
            )
            post_win_decision = (
                "advance_next_level" if mastery_ready else "replay_same_level"
            )
            transition_allowed = mastery_ready
            if mastery_ready:
                collect_post_win_transition = True
                promoted = True
                mastered_levels = _append_unique(mastered_levels, episode_level)
                current_level = min(config.maximum_level, episode_level + 1)
                frontier_win_streak = 0
                reset_reason = "promoted"
            else:
                current_level = episode_level
                if config.frontier_win_streak_required > 1:
                    same_level_replay_requested = True
                    if outcome.same_level_replay_succeeded is False:
                        replay_supported = False
                        replay_blocked_reason = str(
                            outcome.same_level_replay_reason
                            or DEFAULT_REPLAY_BLOCKED_REASON
                        )
                        replay_recovery_required = True
                        post_win_blocked_reason = replay_blocked_reason
                        reset_reason = REPLAY_RECOVERY_RESET_REASON
                    elif outcome.same_level_replay_succeeded is True:
                        # The live wrapper marks support restored and clears the
                        # recovery request, while retaining its stored previous
                        # blocked-reason attribute until another owner clears it.
                        replay_supported = True
                        replay_recovery_required = False
        else:
            post_win_decision = "hold_frontier"
            current_level = frontier_level_before
    else:
        if is_frontier_mastery_attempt and frontier_win_streak > 0:
            reset_reason = _failure_reset_reason(result)
            frontier_win_streak = 0
        if (
            result in {"loss", "timeout"}
            and current_attempt >= config.max_attempts_per_level
        ):
            current_attempt = 0

    latches = GeneralistMasteryLatches(
        mastery_ready=mastery_ready,
        promoted=promoted,
        reset_reason=reset_reason,
        sample_source=sample_source,
    )
    next_state = GeneralistProgressionState(
        current_level=current_level,
        current_attempt=current_attempt,
        frontier_win_streak=frontier_win_streak,
        cleared_levels=cleared_levels,
        frontier_mastered_levels=mastered_levels,
        frontier_replay_supported=replay_supported,
        frontier_replay_blocked_reason=replay_blocked_reason,
        frontier_replay_recovery_required=replay_recovery_required,
        last_episode=latches,
    )
    return GeneralistProgressionTransition(
        state=next_state,
        episode_level=episode_level,
        episode_attempt=episode_attempt,
        is_win_like=is_win_like,
        is_frontier_mastery_attempt=is_frontier_mastery_attempt,
        post_win_decision=post_win_decision,
        post_win_transition_allowed=transition_allowed,
        collect_post_win_transition=collect_post_win_transition,
        same_level_replay_requested=same_level_replay_requested,
        replay_recovery_required=replay_recovery_required,
        post_win_blocked_reason=post_win_blocked_reason,
    )
