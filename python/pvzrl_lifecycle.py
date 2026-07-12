"""Pure lifecycle classification and immutable Phase 5 runtime state records.

The bridge observation remains authoritative.  This module is initially a
shadow representation: it classifies observations and recommends a category
of next transition, but it performs no clicks, resets, bridge requests, file
writes, or progression mutations.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional, Tuple


LIFECYCLE_ACTIVE_GAMEPLAY = "active_gameplay"
LIFECYCLE_POST_WIN_PENDING = "post_win_pending"
LIFECYCLE_LOSS_PENDING = "loss_pending"
LIFECYCLE_RESETTING = "resetting"
LIFECYCLE_READY = "ready"
LIFECYCLE_UNKNOWN = "unknown"

PHASE_STARTUP = "startup"
PHASE_LOADING = "loading"
PHASE_SEED_SELECTION = "seed_selection"
PHASE_ACTIVE_GAMEPLAY = "active_gameplay"
PHASE_READY = "ready"
PHASE_WIN_PENDING_CONFIRMATION = "win_pending_confirmation"
PHASE_WIN = "win"
PHASE_REWARD_UNLOCK = "reward_unlock"
PHASE_SAME_LEVEL_REPLAY = "same_level_replay"
PHASE_ADVENTURE_PROGRESSION = "adventure_progression"
PHASE_LOSS = "loss"
PHASE_TIMEOUT = "timeout"
PHASE_CORRUPTION = "corruption"
PHASE_UNKNOWN = "unknown"


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(default if value is None else value)
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _screen_state(observation: Mapping[str, Any]) -> str:
    return str(observation.get("screenState") or observation.get("screen_state") or "")


def seed_selection_visible(observation: Mapping[str, Any]) -> bool:
    return bool(
        observation.get("seedSelectionScreenVisible")
        or observation.get("isSeedSelectionScreen")
        or observation.get("seedSelectionActive")
        or observation.get("seedSelectionPanelActive")
        or observation.get("onSeedSelectionScreen")
        or _screen_state(observation) == "seed_selection"
    )


def adventure_seed_selection_visible(observation: Mapping[str, Any]) -> bool:
    return bool(
        observation.get("isSeedSelectionScreen")
        or observation.get("seedSelectionActive")
        or observation.get("seedSelectionPanelActive")
        or _screen_state(observation) == "seed_selection"
    )


def adventure_gameplay_ready_visible(observation: Mapping[str, Any]) -> bool:
    return bool(
        (
            observation.get("isGameplayReady")
            or observation.get("gameplayReady")
            or _screen_state(observation) == "gameplay"
        )
        and not adventure_seed_selection_visible(observation)
    )


def adventure_startup_visible(observation: Mapping[str, Any]) -> bool:
    return bool(
        observation.get("startupPopupVisible")
        or observation.get("startupOkButtonVisible")
        or _screen_state(observation) == "startup_popup"
    )


def adventure_loss_visible(observation: Mapping[str, Any]) -> bool:
    return bool(
        observation.get("isGameOverScreen")
        or observation.get("restartButtonActive")
        or observation.get("restartDetectionReason")
        or _screen_state(observation) in {"game_over", "game_over_restart_screen"}
    )


def adventure_post_win_continue_visible(observation: Mapping[str, Any]) -> bool:
    return bool(
        observation.get("rewardObjectVisible")
        or observation.get("levelCompleteScreenVisible")
        or observation.get("postWinContinueVisible")
        or str(observation.get("nextStep") or "")
        in {"cleanup_reward_ui", "click_trophy", "click_reward_continue"}
        or _screen_state(observation)
        in {"level_complete_trophy", "reward_screen", "reward_unlock"}
    )


def adventure_menu_visible(observation: Mapping[str, Any]) -> bool:
    screen = _screen_state(observation)
    return bool(
        observation.get("isAdventureButtonVisible")
        and screen in {"main_menu", "loading_or_menu", "adventure_menu"}
        and not adventure_post_win_continue_visible(observation)
        and not adventure_startup_visible(observation)
        and not adventure_seed_selection_visible(observation)
        and not adventure_gameplay_ready_visible(observation)
    )


def _live_board_progress(observation: Mapping[str, Any]) -> bool:
    if not bool(observation.get("boardFound")):
        return False
    if bool(observation.get("done")) or bool(observation.get("over")):
        return False
    if str(observation.get("terminalHint") or "") != "running":
        return False
    if bool(observation.get("seedSelectionActive")):
        return False
    return any(
        _safe_int(observation.get(key)) > 0
        for key in (
            "wave",
            "plantCount",
            "visiblePlantObjectCount",
            "zombieCount",
            "bulletCount",
            "killCount",
        )
    )


def _active_gameplay_progress(observation: Mapping[str, Any]) -> bool:
    if bool(observation.get("gameplayReady")) or bool(observation.get("actualGameplayReady")):
        return True
    return any(
        _safe_int(observation.get(key)) > 0
        for key in (
            "wave",
            "killCount",
            "plantCount",
            "visiblePlantObjectCount",
            "zombieCount",
            "logicalZombieCount",
            "sceneZombieObjectCount",
            "bulletCount",
            "logicalBulletCount",
            "sceneBulletObjectCount",
            "seedSlotCount",
            "activeGameplayCardBankCount",
        )
    )


def _confirmed_post_win(observation: Mapping[str, Any]) -> bool:
    screen = _screen_state(observation)
    terminal_hint = str(observation.get("terminalHint") or "")
    explicit = bool(
        observation.get("trophyVisible")
        or observation.get("levelCompleteTrophyVisible")
        or observation.get("postWinClickRequired")
        or observation.get("rewardObjectVisible")
        or observation.get("rewardScreenVisible")
        or observation.get("unlockScreenVisible")
        or observation.get("newPlantUnlockedVisible")
    )
    derived = bool(
        observation.get("isRewardScreen")
        or observation.get("isNewPlantUnlockedScreen")
        or observation.get("levelCompleteScreenVisible")
        or screen in {"level_complete_trophy", "reward_unlock", "reward_screen"}
    )
    if not (explicit or derived) or _live_board_progress(observation):
        return False
    return bool(explicit or terminal_hint != "running")


def _confirmed_loss(observation: Mapping[str, Any]) -> bool:
    if _confirmed_post_win(observation):
        return False
    screen = _screen_state(observation)
    return bool(
        observation.get("gameOverRestartScreenVisible")
        or observation.get("loseMenuVisible")
        or observation.get("lossMenuActive")
        or observation.get("gameOverTextVisible")
        or observation.get("onGameOverScreen")
        or observation.get("onLossScreen")
        or (
            (
                observation.get("restartButtonVisible")
                or observation.get("restartButtonActive")
                or observation.get("onRestartScreen")
            )
            and bool(observation.get("gameOverTextVisible"))
        )
        or observation.get("nextStep") == "click_restart"
        or (
            screen in {"game_over", "game_over_restart_screen"}
            and bool(
                observation.get("gameOverTextVisible")
                or observation.get("onGameOverScreen")
                or observation.get("onLossScreen")
                or observation.get("lossMenuActive")
                or observation.get("nextStep") == "click_restart"
            )
        )
    )


def _confirmed_active_gameplay(observation: Mapping[str, Any]) -> bool:
    return bool(
        observation.get("gameplayReady")
        and not observation.get("done")
        and not observation.get("over")
        and not seed_selection_visible(observation)
        and not _confirmed_post_win(observation)
        and not _confirmed_loss(observation)
    )


def _active_bank_count(observation: Mapping[str, Any]) -> int:
    direct = _safe_int(observation.get("activeGameplayCardBankCount"))
    if direct > 0:
        return direct
    cards = observation.get("activeGameplayCardBankCards", [])
    if isinstance(cards, list) and cards:
        return len(cards)
    entries = observation.get("activeGameplayCardBankPlantTypeCounts", [])
    if not isinstance(entries, list):
        return 0
    return sum(
        _safe_int(entry.get("count"))
        for entry in entries
        if isinstance(entry, Mapping)
    )


def _fresh_playable_board(
    observation: Mapping[str, Any],
    *,
    fallback_rows: int,
) -> bool:
    if not bool(observation.get("gameplayReady")):
        return False
    if not bool(observation.get("actualGameplayReady", observation.get("gameplayReady"))):
        return False
    if bool(observation.get("done")) or bool(observation.get("over")):
        return False
    screen = _screen_state(observation)
    if screen and screen != "gameplay":
        return False
    if seed_selection_visible(observation):
        return False
    if any(
        bool(observation.get(key))
        for key in (
            "blockingRewardUiActive",
            "trophyVisible",
            "levelCompleteTrophyVisible",
            "postWinClickRequired",
            "rewardScreenVisible",
            "unlockScreenVisible",
            "newPlantUnlockedVisible",
            "onGameOverScreen",
            "lossMenuActive",
            "onRestartScreen",
        )
    ):
        return False
    if any(
        _safe_int(observation.get(key)) != 0
        for key in (
            "wave",
            "killCount",
            "plantCount",
            "visiblePlantObjectCount",
            "zombieCount",
            "bulletCount",
        )
    ):
        return False
    rows = max(1, _safe_int(observation.get("rowCount"), fallback_rows))
    if _safe_int(observation.get("logicalMowerCount"), rows) != rows:
        return False
    if _safe_int(observation.get("duplicateMowerRowCount")) != 0:
        return False
    slot_count = _safe_int(observation.get("seedSlotCount"), -1)
    if slot_count <= 0:
        slots = observation.get("seedSlots", [])
        slot_count = len(slots) if isinstance(slots, list) else 0
    if slot_count <= 0 or _active_bank_count(observation) <= 0:
        return False
    legal_count = _safe_int(observation.get("legalActionCount"))
    if legal_count <= 0:
        legal = observation.get("legalActions", [])
        legal_count = len(legal) if isinstance(legal, list) else 0
    if legal_count <= 0:
        return False
    next_step = observation.get("nextStep") or observation.get("next_step")
    return bool(
        next_step in (None, "", "play")
        or (isinstance(next_step, str) and next_step.lower() == "play")
    )


def _dirty_active_board(
    observation: Mapping[str, Any],
    *,
    fallback_rows: int,
    start_sun: Optional[int],
) -> bool:
    if not (_confirmed_active_gameplay(observation) or _live_board_progress(observation)):
        return False
    if _fresh_playable_board(observation, fallback_rows=fallback_rows):
        return False
    wave = _safe_int(observation.get("wave"))
    rows = max(1, _safe_int(observation.get("rowCount"), fallback_rows))
    logical_mowers = _safe_int(observation.get("logicalMowerCount"), rows)
    visible_mowers = _safe_int(observation.get("visibleMowerObjectCount"), rows)
    resolved_start_sun = 500 if start_sun is None else int(start_sun)
    sun = _safe_int(observation.get("sun"), resolved_start_sun)
    sun_drift = wave > 0 and sun != resolved_start_sun
    return any(
        (
            wave > 0,
            _safe_int(observation.get("killCount")) > 0,
            _safe_int(observation.get("plantCount")) > 0,
            _safe_int(observation.get("visiblePlantObjectCount")) > 0,
            _safe_int(observation.get("zombieCount")) > 0,
            _safe_int(observation.get("bulletCount")) > 0,
            logical_mowers < rows,
            visible_mowers < rows,
            sun_drift,
        )
    )


def _stale_cleanup_ui(observation: Mapping[str, Any]) -> bool:
    cleanup = bool(
        observation.get("nextStep") == "cleanup_reward_ui"
        or observation.get("blockingRewardUiActive")
        or _screen_state(observation)
        in {"reward_unlock", "reward_screen", "level_complete_trophy"}
    )
    return bool(
        cleanup
        and not observation.get("done")
        and not observation.get("over")
        and not seed_selection_visible(observation)
        and not _active_gameplay_progress(observation)
        and observation.get("boardFound")
    )


def legacy_lifecycle_state(
    observation: Mapping[str, Any],
    *,
    fallback_rows: int = 5,
    start_sun: Optional[int] = 500,
) -> str:
    """Pure byte-for-value projection of the pre-Phase-5 base classifier."""

    if _live_board_progress(observation):
        return LIFECYCLE_ACTIVE_GAMEPLAY
    if _confirmed_post_win(observation):
        return LIFECYCLE_POST_WIN_PENDING
    if _confirmed_loss(observation):
        return LIFECYCLE_LOSS_PENDING
    if _confirmed_active_gameplay(observation):
        if _dirty_active_board(
            observation,
            fallback_rows=fallback_rows,
            start_sun=start_sun,
        ):
            return LIFECYCLE_ACTIVE_GAMEPLAY
        return LIFECYCLE_READY
    if seed_selection_visible(observation) or _stale_cleanup_ui(observation):
        return LIFECYCLE_RESETTING
    return LIFECYCLE_UNKNOWN


def legacy_done_reason(observation: Mapping[str, Any]) -> str:
    terminal_hint = str(observation.get("terminalHint") or "")
    screen = _screen_state(observation)
    post_win = bool(
        observation.get("trophyVisible")
        or observation.get("levelCompleteTrophyVisible")
        or observation.get("postWinClickRequired")
        or observation.get("rewardObjectVisible")
        or observation.get("rewardScreenVisible")
        or observation.get("unlockScreenVisible")
        or observation.get("newPlantUnlockedVisible")
        or observation.get("isRewardScreen")
        or observation.get("isNewPlantUnlockedScreen")
        or observation.get("levelCompleteScreenVisible")
        or (
            screen in {"level_complete_trophy", "reward_unlock", "reward_screen"}
            and terminal_hint != "running"
        )
    )
    if post_win and not _live_board_progress(observation):
        return "win"
    restart_screen = bool(
        not observation.get("onPauseMenu")
        and not observation.get("pauseMenuActive")
        and (
            observation.get("onGameOverScreen")
            or observation.get("lossMenuActive")
            or (
                observation.get("onRestartScreen")
                and observation.get("gameOverTextVisible")
            )
        )
    )
    return "loss" if restart_screen else "none"


@dataclass(frozen=True, slots=True)
class LifecycleContext:
    timeout: bool = False
    corruption: bool = False
    transition_target: str = ""
    expected_level: int = 0
    observed_level: int = 0
    identity_reliable: bool = False
    level_mismatch_authoritative: bool = False


@dataclass(frozen=True, slots=True)
class LifecycleClassification:
    phase: str
    legacy_state: str
    done_reason: str
    next_directive: str
    screen_state: str
    seed_selection_visible: bool
    gameplay_ready: bool
    adventure_seed_selection_visible: bool
    adventure_gameplay_ready: bool
    adventure_startup_visible: bool
    adventure_loss_visible: bool
    adventure_menu_visible: bool
    live_board_progress: bool
    post_win_evidence: bool
    loss_evidence: bool
    transitional: bool
    expected_level: int = 0
    observed_level: int = 0
    identity_reliable: bool = False
    level_mismatch_authoritative: bool = False
    reasons: Tuple[str, ...] = ()


def classify_lifecycle(
    observation: Mapping[str, Any],
    *,
    context: LifecycleContext = LifecycleContext(),
    fallback_rows: int = 5,
    start_sun: Optional[int] = 500,
) -> LifecycleClassification:
    """Return a side-effect-free shadow classification for one observation."""

    legacy_state = legacy_lifecycle_state(
        observation,
        fallback_rows=fallback_rows,
        start_sun=start_sun,
    )
    done_reason = legacy_done_reason(observation)
    screen = _screen_state(observation)
    seed_visible = seed_selection_visible(observation)
    adventure_seed_visible = adventure_seed_selection_visible(observation)
    adventure_gameplay_ready = adventure_gameplay_ready_visible(observation)
    adventure_startup = adventure_startup_visible(observation)
    adventure_loss = adventure_loss_visible(observation)
    adventure_menu = adventure_menu_visible(observation)
    progress = _live_board_progress(observation)
    post_win = _confirmed_post_win(observation)
    loss = _confirmed_loss(observation)
    derived_possible_win = bool(
        _safe_int(observation.get("maxWave")) > 0
        and _safe_int(observation.get("wave"))
        >= _safe_int(observation.get("maxWave"))
        and _safe_int(observation.get("zombieCount")) == 0
        and not bool(observation.get("moreZombiesComing", False))
    )
    possible_win = bool(
        (
            str(observation.get("terminalHint") or "") == "possible_win"
            or derived_possible_win
        )
        and not post_win
        and not loss
    )
    transition_target = str(context.transition_target or "")
    reasons: list[str] = []

    if context.corruption:
        phase, directive = PHASE_CORRUPTION, "reset_corruption"
        reasons.append("environment_corruption")
    elif context.timeout:
        phase, directive = PHASE_TIMEOUT, "reset_timeout"
        reasons.append("action_or_episode_timeout")
    elif transition_target == PHASE_SAME_LEVEL_REPLAY:
        phase, directive = PHASE_SAME_LEVEL_REPLAY, "continue_same_level_replay"
        reasons.append("same_level_replay_requested")
    elif transition_target == PHASE_ADVENTURE_PROGRESSION:
        phase, directive = PHASE_ADVENTURE_PROGRESSION, "continue_adventure_progression"
        reasons.append("adventure_progression_requested")
    elif adventure_startup:
        phase, directive = PHASE_STARTUP, "dismiss_startup_popup"
        reasons.append("startup_popup_visible")
    elif seed_visible:
        phase, directive = PHASE_SEED_SELECTION, "select_seeds_or_start"
        reasons.append("seed_selection_visible")
    elif post_win:
        if screen in {"reward_unlock", "reward_screen"} or bool(
            observation.get("rewardScreenVisible")
            or observation.get("unlockScreenVisible")
            or observation.get("newPlantUnlockedVisible")
        ):
            phase, directive = PHASE_REWARD_UNLOCK, "collect_or_continue_reward"
        else:
            phase, directive = PHASE_WIN, "continue_post_win"
        reasons.append("confirmed_post_win_ui")
    elif adventure_menu or screen in {
        "loading",
        "loading_or_menu",
        "transition",
        "main_menu",
        "adventure_menu",
    }:
        phase, directive = PHASE_LOADING, "wait_or_enter_adventure"
        reasons.append(screen or "loading")
    elif loss or adventure_loss:
        phase, directive = PHASE_LOSS, "click_restart"
        reasons.append("confirmed_loss_ui")
    elif legacy_state == LIFECYCLE_RESETTING:
        phase, directive = PHASE_LOADING, "continue_reset_transition"
        reasons.append("stale_cleanup_or_reset_transition")
    elif possible_win:
        phase, directive = PHASE_WIN_PENDING_CONFIRMATION, "wait_terminal_confirmation"
        reasons.append("possible_win_without_confirmed_ui")
    elif legacy_state == LIFECYCLE_ACTIVE_GAMEPLAY:
        phase, directive = PHASE_ACTIVE_GAMEPLAY, "play"
        reasons.append("active_board_progress")
    elif legacy_state == LIFECYCLE_READY:
        phase, directive = PHASE_READY, "play"
        reasons.append("fresh_playable_board")
    else:
        phase, directive = PHASE_UNKNOWN, "observe"
        reasons.append("no_authoritative_lifecycle_signal")

    if (
        context.level_mismatch_authoritative
        and context.expected_level > 0
        and context.observed_level > 0
        and context.expected_level != context.observed_level
    ):
        directive = "block_level_identity_mismatch"
        reasons.append("authoritative_level_identity_mismatch")

    return LifecycleClassification(
        phase=phase,
        legacy_state=legacy_state,
        done_reason=done_reason,
        next_directive=directive,
        screen_state=screen,
        seed_selection_visible=seed_visible,
        gameplay_ready=bool(observation.get("gameplayReady") and not seed_visible),
        adventure_seed_selection_visible=adventure_seed_visible,
        adventure_gameplay_ready=adventure_gameplay_ready,
        adventure_startup_visible=adventure_startup,
        adventure_loss_visible=adventure_loss,
        adventure_menu_visible=adventure_menu,
        live_board_progress=progress,
        post_win_evidence=post_win,
        loss_evidence=loss,
        transitional=phase
        not in {PHASE_ACTIVE_GAMEPLAY, PHASE_READY, PHASE_UNKNOWN},
        expected_level=max(0, int(context.expected_level)),
        observed_level=max(0, int(context.observed_level)),
        identity_reliable=bool(context.identity_reliable),
        level_mismatch_authoritative=bool(
            context.level_mismatch_authoritative
        ),
        reasons=tuple(reasons),
    )


__all__ = [
    "LifecycleClassification",
    "LifecycleContext",
    "adventure_gameplay_ready_visible",
    "adventure_loss_visible",
    "adventure_menu_visible",
    "adventure_post_win_continue_visible",
    "adventure_seed_selection_visible",
    "adventure_startup_visible",
    "classify_lifecycle",
    "legacy_done_reason",
    "legacy_lifecycle_state",
    "seed_selection_visible",
]
