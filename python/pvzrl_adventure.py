"""Adventure Generalist evaluation/progression runner for PvZRL.

This module intentionally runs the loaded Generalist policy in inference mode
only. It does not call PPO learning APIs and it keeps seed/action-space
compatibility checks outside the live game loop so failures are clear and early.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from pvzrl_env import (
    classify_done_reason,
    decode_action,
    is_restart_screen_observation,
    normalize_plant_name,
    plant_type_name,
    registry_entries,
)
from pvzrl_rewards import REWARD_EPISODE_TOTAL_FIELDS, REWARD_POLICY_VERSION
from pvzrl_fusion import FUSION_POLICY_NONE, fusion_live_fields
from pvzrl_human_coach import human_coach_live_status_from_hook
from pvzrl_lifecycle import (
    adventure_gameplay_ready_visible,
    adventure_seed_selection_visible,
    adventure_startup_visible,
    screen_state as lifecycle_screen_state,
)
from pvzrl_sb3 import PvZMaskedPPOEnv, PvZSB3Config
from pvzrl_seed_inventory import inventory_from_runtime_sources
from pvzrl_telemetry import LiveStatusWriter, live_status_significant_state


BASE_UNLOCKED_SEEDS = ("SunFlower", "Peashooter")
KNOWN_ADVENTURE_LEVEL_UNLOCKS = {
    1: ["CherryBomb"],
}

POST_WIN_RECOVERY_TIMEOUT_SECONDS = 35.0
POST_WIN_RECOVERY_POLL_SECONDS = 0.25
DEFAULT_ADVENTURE_SOFT_MAX_STEPS = 2000
DEFAULT_ADVENTURE_HARD_MAX_STEPS = 3500
ADVENTURE_GENERALIST_EVAL_RUN_MODE = "adventure_generalist_14slot_eval"
ADVENTURE_GENERALIST_ACTION_SPACE_MODE = "adventure_14slot_identity_full_v2"
LEVEL_IDENTITY_POST_WIN_STATES = {"level_complete_trophy", "reward_unlock", "reward_screen"}
LEVEL_IDENTITY_TRANSITIONAL_STATES = {
    *LEVEL_IDENTITY_POST_WIN_STATES,
    "seed_selection",
    "loading",
    "loading_or_menu",
    "transition",
}
LEVEL_IDENTITY_CHALLENGE_MODE_MARKERS = (
    "challenge",
    "mini_game",
    "minigame",
    "survival",
    "puzzle",
    "vase",
    "iz",
)


@dataclass
class AdventureAttemptLog:
    attempt: int
    result: str = "unknown"
    done_reason: str = "none"
    terminal_reason: str = ""
    episode_reward: float = 0.0
    episode_length: int = 0
    final_wave: int = 0
    max_wave: int = 0
    zombies_killed: int = 0
    plants_placed: int = 0
    mowers_lost: int = 0
    illegal_actions: int = 0
    bridge_errors: int = 0
    reset_failures: int = 0
    selected_seeds: List[str] = field(default_factory=list)
    available_seeds: List[str] = field(default_factory=list)
    unlocked_seeds: List[str] = field(default_factory=list)
    progression_index: int = 0
    adventure_world: int = 0
    adventure_stage: int = 0
    adventure_level_label: str = ""
    blocked_reason: str = ""
    win_detected: bool = False
    trophy_visible: bool = False
    trophy_clicked: bool = False
    reward_screen_seen: bool = False
    unlock_screen_seen: bool = False
    post_win_transition_completed: bool = False
    post_win_blocked_reason: str = ""
    trophy_click_count: int = 0
    reward_continue_click_count: int = 0
    screen_state_at_terminal: str = ""
    terminal_trophy_visible: bool = False
    terminal_level_complete_trophy_visible: bool = False
    terminal_post_win_click_required: bool = False
    terminal_reward_screen_visible: bool = False
    terminal_unlock_screen_visible: bool = False
    terminal_new_plant_unlocked_visible: bool = False
    terminal_reward_object_visible: bool = False
    wins_after: int = 0
    losses_after: int = 0
    consecutive_wins_after: int = 0
    advanced_after: bool = False
    required_consecutive_wins_remaining: int = 0
    soft_max_steps: int = DEFAULT_ADVENTURE_SOFT_MAX_STEPS
    hard_max_steps: int = DEFAULT_ADVENTURE_HARD_MAX_STEPS
    final_wave_extension_enabled: bool = True
    soft_timeout_reached: bool = False
    soft_timeout_extended: bool = False
    soft_timeout_step: int = 0
    steps_after_soft_timeout: int = 0
    timeout_classification: str = "none"
    tactical_mask_enabled: bool = False
    wallnut_tactical_mask_enabled: bool = False
    cherrybomb_tactical_mask_enabled: bool = False
    episode_summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdventureLevelLog:
    level: int
    progression_index: int = 0
    adventure_world: int = 0
    adventure_stage: int = 0
    adventure_level_label: str = ""
    advance_on_wins: int = 1
    attempts: int = 0
    wins: int = 0
    losses: int = 0
    consecutive_wins: int = 0
    advanced: bool = False
    blocked_reason: str = ""
    win_detected: bool = False
    trophy_click_count: int = 0
    reward_continue_click_count: int = 0
    post_win_transition_completed: bool = False
    post_win_blocked_reason: str = ""
    selected_seeds: List[str] = field(default_factory=list)
    unlocked_before_level: List[str] = field(default_factory=list)
    unlock_screen_seen: bool = False
    unlock_screen_snapshot: Dict[str, Any] = field(default_factory=dict)
    unlocked_after_level: List[str] = field(default_factory=list)
    available_seed_names: List[str] = field(default_factory=list)
    available_seed_names_after_level: List[str] = field(default_factory=list)
    unknown_unlock_objects: List[Dict[str, Any]] = field(default_factory=list)
    unknown_visible_seed_names: List[str] = field(default_factory=list)
    attempt_logs: List[Dict[str, Any]] = field(default_factory=list)
    avg_reward: float = 0.0
    avg_wave: float = 0.0
    avg_kills: float = 0.0
    avg_plants: float = 0.0
    avg_mowers_lost: float = 0.0
    row_defense_response_rate: float = 0.0
    undefended_threat_ratio_by_row: Dict[str, float] = field(default_factory=dict)


def launch_gui(live_status_path: Path) -> Optional[subprocess.Popen[Any]]:
    gui_path = Path(__file__).with_name("pvzrl_gui.py")
    if not gui_path.exists():
        print(f"[adventure] GUI requested but not found: {gui_path}")
        return None
    return subprocess.Popen(
        [sys.executable, str(gui_path), "--live-status-path", str(live_status_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _seed_alias_map() -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for entry in registry_entries():
        canonical = str(entry.get("canonical_name") or "").strip()
        if not canonical:
            continue
        names = [canonical, *(str(alias) for alias in entry.get("aliases", []) or [])]
        for name in names:
            if name:
                aliases[normalize_plant_name(name)] = canonical
    return aliases


SEED_NAME_ALIASES = _seed_alias_map()


def canonical_seed_name(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"none", "unknown", "-1"}:
        return ""
    if text.lstrip("-").isdigit():
        try:
            plant_type = int(text)
        except ValueError:
            plant_type = -1
        return plant_type_name(plant_type) if plant_type >= 0 else ""
    return SEED_NAME_ALIASES.get(normalize_plant_name(text), text)


def _ordered_unique_seed_names(values: List[Any]) -> List[str]:
    output: List[str] = []
    seen = set()
    for value in values:
        name = canonical_seed_name(value)
        if not name or name in seen:
            continue
        output.append(name)
        seen.add(name)
    return output


def _canonical_seed_sequence(values: List[Any]) -> List[str]:
    """Canonicalize a requested deck without discarding intentional duplicates."""

    return [name for name in (canonical_seed_name(value) for value in values) if name]


EVALUATION_LOADOUT_PRIORITY = (
    # The core remains useful in every world.  Pool support is deliberately
    # activated only when the interactive CardUI state proves these cards can
    # be selected; no world-number heuristic invents a seed-card capability.
    "SunFlower",
    "Peashooter",
    "LilyPad",
    "Tanglekelp",
)


@dataclass
class RuntimeEvaluationLoadoutSelector:
    """Choose a stable, runtime-authorized evaluation loadout at seed screens.

    The full-Adventure model keeps fourteen permanent *model* slots, while a
    live seed chooser may expose fewer selectable cards.  This selector never
    expands from registry or remembered unlock evidence alone: a new card can
    enter evaluation only after the live chooser reports it.  It consequently
    makes Pool support available without forcing aquatic cards into land
    levels or trusting stale profile text.
    """

    configured_seed_list: List[str]
    max_seed_slots: int
    context: Dict[str, Any]
    rotation_cursor: int = 0

    def __post_init__(self) -> None:
        self.configured_seed_list = _canonical_seed_sequence(self.configured_seed_list)
        self.max_seed_slots = max(1, int(self.max_seed_slots))

    def _selectable_seed_names(self, state: Dict[str, Any]) -> List[str]:
        values: List[Any] = []
        for key in (
            "selectableSeedNames",
            "confirmedSelectableSeedCardNames",
            "visibleSeedCardNames",
            "availableSeedNames",
        ):
            values.extend(_state_list(state, key))
        for slot in _state_list(state, "seedSlots"):
            if not isinstance(slot, dict):
                continue
            if any(
                bool(slot.get(key))
                for key in (
                    "availableSeedCard",
                    "isAvailableSeedCard",
                    "visibleSeedCard",
                    "isVisibleSeedCard",
                    "seedChoice",
                    "isSeedChoice",
                    "selectableSeedCard",
                    "isSelectableSeedCard",
                )
            ):
                values.append(
                    slot.get("plantTypeName")
                    or slot.get("displayName")
                    or slot.get("seedName")
                    or slot.get("plantType")
                )
        supported = {canonical_seed_name(entry.get("canonical_name")) for entry in registry_entries()}
        return [
            name
            for name in _ordered_unique_seed_names(values)
            if name in supported
        ]

    def _capacity(self, state: Dict[str, Any], current: List[str]) -> int:
        candidates: List[int] = []
        for key in (
            "seedBankCapacity",
            "seedSlotCapacity",
            "seedSlotCount",
            "seedCardSlotCount",
            "seedSelectionSlotCount",
            "selectableSeedCapacity",
            "currentSeedBankCapacity",
            "activeSeedSlotCapacity",
        ):
            try:
                value = int(state.get(key, 0) or 0)
            except (TypeError, ValueError):
                value = 0
            if value > 0:
                candidates.append(value)
        # Without direct capacity evidence, preserve the current count.  The
        # selectable-card list can authorize a rotation but not a bank resize.
        fallback = len(current) or len(self.configured_seed_list) or 1
        return min(self.max_seed_slots, max(1, max(candidates, default=fallback)))

    def __call__(self, state: Dict[str, Any], current_seed_list: List[str]) -> Tuple[List[str], str]:
        current = _canonical_seed_sequence(current_seed_list or self.configured_seed_list)
        selectable = self._selectable_seed_names(state)
        capacity = self._capacity(state, current)
        if not selectable:
            self.context.update(
                {
                    "evaluation_seed_selection_mode": "runtime_rotation",
                    "evaluation_seed_selection_evidence": "none_preserved_current_loadout",
                    "selectable_seeds": [],
                    "selected_seeds": list(current[:capacity]),
                    "selected_loadout": list(current[:capacity]),
                }
            )
            return list(current[:capacity]), ""

        selectable_set = set(selectable)
        selected: List[str] = []
        for seed in EVALUATION_LOADOUT_PRIORITY:
            if seed in selectable_set and seed not in selected and len(selected) < capacity:
                selected.append(seed)
        for seed in current:
            if (
                seed in selectable_set
                and selected.count(seed) < current.count(seed)
                and len(selected) < capacity
            ):
                selected.append(seed)
        remaining = [seed for seed in selectable if seed not in selected]
        if remaining and len(selected) < capacity:
            offset = self.rotation_cursor % len(remaining)
            rotated = remaining[offset:] + remaining[:offset]
            selected.extend(rotated[: max(0, capacity - len(selected))])
            self.rotation_cursor += 1
        if not selected:
            return [], "evaluation_seed_selection_no_supported_runtime_cards"

        self.context.update(
            {
                "evaluation_seed_selection_mode": "runtime_rotation",
                "evaluation_seed_selection_evidence": "interactive_cardui",
                "evaluation_seed_selection_capacity": int(capacity),
                "evaluation_seed_selection_rotation_cursor": int(self.rotation_cursor),
                "selectable_seeds": list(selectable),
                "selected_seeds": list(selected),
                "selected_loadout": list(selected),
                "selected_loadout_count": len(selected),
                "active_seed_slot_count": len(selected),
                "inactive_seed_slot_count": max(0, self.max_seed_slots - len(selected)),
                "inactive_model_slots": max(0, self.max_seed_slots - len(selected)),
                "loadout_reason": "evaluation_runtime_cardui_rotation",
            }
        )
        return selected, ""


def _state_list(state: Dict[str, Any], key: str) -> List[Any]:
    values = state.get(key, [])
    return values if isinstance(values, list) else []


def _snapshot_list(snapshot: Dict[str, Any], key: str) -> List[Any]:
    values = snapshot.get(key, [])
    return values if isinstance(values, list) else []


def _plant_types_to_names(values: List[Any]) -> List[str]:
    names: List[str] = []
    for value in values:
        try:
            plant_type = int(value)
        except (TypeError, ValueError):
            continue
        if plant_type >= 0:
            names.append(plant_type_name(plant_type))
    return _ordered_unique_seed_names(names)


def _ordered_int_values(values: List[Any]) -> List[int]:
    output: List[int] = []
    seen = set()
    for value in values:
        try:
            plant_type = int(value)
        except (TypeError, ValueError):
            continue
        if plant_type < 0 or plant_type in seen:
            continue
        output.append(plant_type)
        seen.add(plant_type)
    return output


def _card_names(cards: List[Any]) -> List[str]:
    names: List[Any] = []
    for card in cards:
        if not isinstance(card, dict):
            continue
        names.append(card.get("plantTypeName") or card.get("displayName") or card.get("gameObjectName"))
        if card.get("plantType") is not None:
            names.append(card.get("plantType"))
    return _ordered_unique_seed_names(names)


def _unknown_objects_from_state(state: Dict[str, Any], snapshot: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    raw_objects: List[Any] = []
    for key in ("unknownUnlockObjects", "unknownVisibleSeedCards"):
        raw_objects.extend(_state_list(state, key))
        raw_objects.extend(_snapshot_list(snapshot, key))
    output: List[Dict[str, Any]] = []
    seen = set()
    for item in raw_objects:
        if not isinstance(item, dict):
            continue
        compact = {
            "object_name": item.get("name") or item.get("gameObjectName") or item.get("displayName") or "",
            "path": item.get("hierarchyPath") or "",
            "plantType": item.get("plantType", item.get("newPlantUnlockedPlantType", -1)),
            "text": item.get("text") or item.get("textBg") or "",
            "className": item.get("className") or "",
        }
        key = tuple(str(compact.get(field, "")) for field in ("object_name", "path", "plantType", "text", "className"))
        if key in seen:
            continue
        seen.add(key)
        output.append(compact)
    return output


def _snapshot_unlock_state(state: Dict[str, Any], source: str = "") -> Dict[str, Any]:
    nested = state.get("unlockSnapshot", {})
    snapshot = nested if isinstance(nested, dict) else {}
    new_type = state.get("newPlantUnlockedPlantType", snapshot.get("newPlantUnlockedPlantType", -1))
    try:
        new_type_int = int(new_type)
    except (TypeError, ValueError):
        new_type_int = -1
    new_name = canonical_seed_name(
        state.get("newPlantUnlockedName") or snapshot.get("newPlantUnlockedName") or (plant_type_name(new_type_int) if new_type_int >= 0 else "")
    )
    visible_seed_names = _ordered_unique_seed_names(
        _state_list(state, "confirmedSelectableSeedCardNames")
        + _snapshot_list(snapshot, "confirmedSelectableSeedCardNames")
        + _state_list(state, "visibleSeedCardNames")
        + _snapshot_list(snapshot, "visibleSeedCardNames")
        + _plant_types_to_names(_state_list(state, "confirmedSelectableSeedPlantTypes"))
        + _plant_types_to_names(_snapshot_list(snapshot, "confirmedSelectableSeedPlantTypes"))
        + _plant_types_to_names(_state_list(state, "visibleSeedPlantTypes"))
        + _plant_types_to_names(_snapshot_list(snapshot, "visibleSeedPlantTypes"))
    )
    unknown_objects = _unknown_objects_from_state(state, snapshot)
    confirmed_seed_types = _ordered_int_values(
        _state_list(state, "confirmedSelectableSeedPlantTypes")
        + _snapshot_list(snapshot, "confirmedSelectableSeedPlantTypes")
        + _state_list(state, "visibleSeedPlantTypes")
        + _snapshot_list(snapshot, "visibleSeedPlantTypes")
    )
    diagnostic_names = _ordered_unique_seed_names(
        _state_list(state, "diagnosticPlantNames")
        + _snapshot_list(snapshot, "diagnosticPlantNames")
        + ([new_name] if new_name else [])
    )
    diagnostic_types = _state_list(state, "diagnosticPlantTypes") + _snapshot_list(snapshot, "diagnosticPlantTypes")
    if new_type_int >= 0:
        diagnostic_types.append(new_type_int)
    return {
        "source": source,
        "screenState": state.get("screenState", ""),
        "rewardScreenVisible": bool(state.get("rewardScreenVisible") or snapshot.get("rewardScreenVisible") or state.get("isRewardScreen")),
        "unlockScreenVisible": bool(state.get("unlockScreenVisible") or snapshot.get("unlockScreenVisible") or state.get("isNewPlantUnlockedScreen")),
        "newPlantUnlockedVisible": bool(state.get("newPlantUnlockedVisible") or snapshot.get("newPlantUnlockedVisible") or bool(new_name)),
        "newPlantUnlockedName": new_name,
        "newPlantUnlockedPlantType": new_type_int,
        "visibleRewardTexts": _state_list(state, "visibleRewardTexts") + _snapshot_list(snapshot, "visibleRewardTexts"),
        "visibleSeedCardNames": visible_seed_names,
        "visibleSeedPlantTypes": confirmed_seed_types,
        "confirmedSelectableSeedCardNames": visible_seed_names,
        "confirmedSelectableSeedPlantTypes": confirmed_seed_types,
        "diagnosticPlantNames": diagnostic_names,
        "diagnosticPlantTypes": diagnostic_types,
        "unknownUnlockObjects": unknown_objects,
        "unknownVisibleSeedCards": _unknown_objects_from_state({"unknownVisibleSeedCards": _state_list(state, "unknownVisibleSeedCards")}, snapshot),
    }


def update_unlocked_from_state(
    unlocked: Counter[str],
    state: Dict[str, Any],
    *,
    source: str = "",
    level: Optional[int] = None,
    use_known_level_fallback: bool = False,
) -> List[str]:
    before = set(unlocked.keys())
    snapshot = _snapshot_unlock_state(state, source=source)
    values: List[Any] = []
    for key in ("unlockedSeedNames", "availableSeedNames", "selectedSeedNames", "visibleSeedCardNames"):
        values.extend(_state_list(state, key))
    values.extend(snapshot.get("visibleSeedCardNames", []))
    for name in _ordered_unique_seed_names(values):
        unlocked[name] += 1
    if use_known_level_fallback and level in KNOWN_ADVENTURE_LEVEL_UNLOCKS:
        for name in KNOWN_ADVENTURE_LEVEL_UNLOCKS[int(level)]:
            unlocked[canonical_seed_name(name)] += 1
    return sorted(set(unlocked.keys()) - before)


def _is_reward_or_unlock_state(state: Dict[str, Any]) -> bool:
    return bool(
        state.get("isRewardScreen")
        or state.get("blockingRewardUiActive")
        or state.get("isNewPlantUnlockedScreen")
        or state.get("rewardScreenVisible")
        or state.get("unlockScreenVisible")
        or state.get("newPlantUnlockedVisible")
    )


def _is_trophy_state(state: Dict[str, Any]) -> bool:
    return bool(
        state.get("trophyVisible")
        or state.get("levelCompleteTrophyVisible")
        or state.get("postWinClickRequired")
    )


def _is_post_win_continue_state(state: Dict[str, Any]) -> bool:
    screen_state = str(state.get("screenState") or "")
    next_step = str(state.get("nextStep") or "")
    return bool(
        state.get("rewardObjectVisible")
        or state.get("levelCompleteScreenVisible")
        or state.get("postWinContinueVisible")
        or next_step in {"cleanup_reward_ui", "click_trophy", "click_reward_continue"}
        or screen_state in {"level_complete_trophy", "reward_screen", "reward_unlock"}
    )


def _is_loss_restart_state(state: Dict[str, Any]) -> bool:
    screen_state = str(state.get("screenState") or "")
    return bool(
        state.get("isGameOverScreen")
        or state.get("restartButtonActive")
        or state.get("restartDetectionReason")
        or screen_state in {"game_over", "game_over_restart_screen"}
    )


def _is_stable_post_win_state(state: Dict[str, Any]) -> bool:
    return bool(
        state.get("isSeedSelectionScreen")
        or state.get("isAdventureButtonVisible")
        or state.get("isGameplayReady")
        or state.get("screenState") in {"main_menu", "loading_or_menu", "transition"}
    ) and not _is_trophy_state(state)


def _post_win_last_state(state: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "screenState",
        "nextStep",
        "gameplayReady",
        "isGameplayReady",
        "seedSelectionActive",
        "isSeedSelectionScreen",
        "isAdventureButtonVisible",
        "startupPopupVisible",
        "startupOkButtonVisible",
        "trophyVisible",
        "levelCompleteTrophyVisible",
        "postWinClickRequired",
        "rewardObjectVisible",
        "rewardScreenVisible",
        "unlockScreenVisible",
        "newPlantUnlockedVisible",
        "blockingRewardUiActive",
        "currentAdventureLevel",
    )
    return {key: state.get(key) for key in keys if key in state}


def _reward_unlock_visible(state: Dict[str, Any]) -> bool:
    return bool(
        state.get("rewardScreenVisible")
        or state.get("unlockScreenVisible")
        or state.get("newPlantUnlockedVisible")
        or state.get("isRewardScreen")
        or state.get("isNewPlantUnlockedScreen")
        or state.get("blockingRewardUiActive")
    )


def _post_win_timeout_reason(state: Dict[str, Any], transition: Dict[str, Any]) -> str:
    if _is_trophy_state(state) or transition.get("trophy_visible"):
        return "post_win_timeout_trophy_still_visible"
    if _reward_unlock_visible(state) or transition.get("reward_screen_seen") or transition.get("unlock_screen_seen"):
        return "post_win_timeout_reward_unlock"
    if state.get("seedSelectionActive") and not state.get("isSeedSelectionScreen"):
        return "post_win_seed_selection_not_reached"
    return "post_win_timeout_unknown_screen"


def _post_win_log_line(
    *,
    elapsed: float,
    state: Dict[str, Any],
    click_target: str,
    click_ok: Optional[bool],
    context: Dict[str, Any],
) -> str:
    click_text = "-" if click_ok is None else str(bool(click_ok))
    return (
        "[adventure] post-win recovery "
        f"elapsed={elapsed:.2f}s "
        f"screenState={state.get('screenState', '')} "
        f"nextStep={state.get('nextStep', '')} "
        f"gameplayReady={bool(state.get('gameplayReady') or state.get('isGameplayReady'))} "
        f"seedSelectionActive={bool(state.get('seedSelectionActive') or state.get('isSeedSelectionScreen'))} "
        f"trophyVisible={bool(state.get('trophyVisible') or state.get('levelCompleteTrophyVisible'))} "
        f"postWinClickRequired={bool(state.get('postWinClickRequired'))} "
        f"rewardUnlockVisible={_reward_unlock_visible(state)} "
        f"clickTarget={click_target or '-'} "
        f"clickOk={click_text} "
        f"current_level={context.get('current_level', '')} "
        f"wins_current={context.get('wins_this_level', context.get('wins', 0))} "
        f"total_wins={context.get('total_wins', 0)}"
    )


def _record_post_win_iteration(
    writer: LiveStatusWriter,
    env: PvZMaskedPPOEnv,
    context: Dict[str, Any],
    state: Dict[str, Any],
    *,
    started: float,
    click_target: str = "",
    click_result: Optional[Dict[str, Any]] = None,
) -> None:
    elapsed = max(0.0, time.monotonic() - started)
    click_ok = None if click_result is None else bool(click_result.get("ok", False))
    context["post_win_active"] = True
    context["post_win_elapsed"] = round(elapsed, 3)
    context["post_win_last_state"] = _post_win_last_state(state)
    context["post_win_last_click_target"] = click_target
    context["post_win_last_click_ok"] = click_ok
    if click_result is not None:
        context["last_ui_action"] = click_result
    print(
        _post_win_log_line(
            elapsed=elapsed,
            state=state,
            click_target=click_target,
            click_ok=click_ok,
            context=context,
        ),
        flush=True,
    )
    writer.write(build_live_status(env, context, adventure_state=state))


def _unlock_snapshot_seen(snapshot: Dict[str, Any]) -> bool:
    return bool(
        snapshot.get("rewardScreenVisible")
        or snapshot.get("unlockScreenVisible")
        or snapshot.get("newPlantUnlockedVisible")
        or snapshot.get("newPlantUnlockedName")
        or snapshot.get("visibleRewardTexts")
        or snapshot.get("visibleSeedCardNames")
        or snapshot.get("unknownUnlockObjects")
    )


def _unique_list(values: List[Any]) -> List[Any]:
    output: List[Any] = []
    seen = set()
    for value in values:
        try:
            key = json.dumps(value, sort_keys=True)
        except TypeError:
            key = str(value)
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output


def _combine_unlock_snapshots(snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
    combined: Dict[str, Any] = {
        "rewardScreenVisible": False,
        "unlockScreenVisible": False,
        "newPlantUnlockedVisible": False,
        "newPlantUnlockedName": "",
        "newPlantUnlockedPlantType": -1,
        "visibleRewardTexts": [],
        "visibleSeedCardNames": [],
        "visibleSeedPlantTypes": [],
        "unknownUnlockObjects": [],
        "unknownVisibleSeedCards": [],
        "snapshots": snapshots,
    }
    for snapshot in snapshots:
        combined["rewardScreenVisible"] = bool(combined["rewardScreenVisible"] or snapshot.get("rewardScreenVisible"))
        combined["unlockScreenVisible"] = bool(combined["unlockScreenVisible"] or snapshot.get("unlockScreenVisible"))
        combined["newPlantUnlockedVisible"] = bool(combined["newPlantUnlockedVisible"] or snapshot.get("newPlantUnlockedVisible"))
        if not combined["newPlantUnlockedName"] and snapshot.get("newPlantUnlockedName"):
            combined["newPlantUnlockedName"] = snapshot.get("newPlantUnlockedName")
        if int(combined["newPlantUnlockedPlantType"] or -1) < 0:
            try:
                plant_type = int(snapshot.get("newPlantUnlockedPlantType", -1) or -1)
            except (TypeError, ValueError):
                plant_type = -1
            if plant_type >= 0:
                combined["newPlantUnlockedPlantType"] = plant_type
        for key in ("visibleRewardTexts", "visibleSeedCardNames", "visibleSeedPlantTypes", "unknownUnlockObjects", "unknownVisibleSeedCards"):
            values = snapshot.get(key, [])
            if isinstance(values, list):
                combined[key].extend(values)
    for key in ("visibleRewardTexts", "visibleSeedCardNames", "visibleSeedPlantTypes", "unknownUnlockObjects", "unknownVisibleSeedCards"):
        combined[key] = _unique_list(combined[key])
    return combined


def _post_win_level_handoff_status(
    state: Dict[str, Any],
    completed_level: int,
) -> Tuple[str, Dict[str, Any]]:
    """Classify a gameplay frame at the sequential Adventure handoff.

    A terminal detector can finish before the old board disappears.  During
    that narrow window the bridge truthfully reports gameplayReady for the
    completed level.  That is transition evidence, not proof that the next
    Adventure level is ready and not an identity contradiction by itself.
    """

    current_level = max(1, int(completed_level))
    expected_level = current_level + 1
    diagnostics = adventure_level_identity_diagnostics(state, expected_level)
    if diagnostics.get("level_identity_reliable"):
        return "confirmed", diagnostics

    observed_levels = [
        _positive_int_or_none(diagnostics.get("bridge_detected_level")),
        _positive_int_or_none(diagnostics.get("profile_adventure_level")),
    ]
    unexpected_levels = sorted(
        {
            value
            for value in observed_levels
            if value is not None and value not in {current_level, expected_level}
        }
    )
    if diagnostics.get("challenge_mode_context") or unexpected_levels:
        diagnostics = dict(diagnostics)
        diagnostics["unexpected_levels"] = unexpected_levels
        return "blocked", diagnostics
    return "transitional", diagnostics


def collect_post_win_unlocks(
    env: PvZMaskedPPOEnv,
    writer: LiveStatusWriter,
    context: Dict[str, Any],
    unlocked: Counter[str],
    level: int,
) -> Tuple[Dict[str, Any], bool, Dict[str, Any], List[str], List[Dict[str, Any]], str, Dict[str, Any]]:
    snapshots: List[Dict[str, Any]] = []
    transition = {
        "win_detected": True,
        "trophy_visible": False,
        "trophy_clicked": False,
        "reward_screen_seen": False,
        "unlock_screen_seen": False,
        "post_win_transition_completed": False,
        "post_win_blocked_reason": "",
        "trophy_click_count": 0,
        "reward_continue_click_count": 0,
    }
    context.update(transition)
    context["post_win_active"] = True
    context["post_win_elapsed"] = 0.0
    context["post_win_click_attempts"] = 0
    context["post_win_last_state"] = {}
    context["post_win_last_click_target"] = ""
    context["post_win_last_click_ok"] = None
    context["state"] = "POST_WIN_RECOVERY"
    started = time.monotonic()
    deadline = started + POST_WIN_RECOVERY_TIMEOUT_SECONDS
    state: Dict[str, Any] = {}
    blocked_reason = ""
    available_after: List[str] = []
    seed_names = list(context.get("selected_seeds", []) or env.config.seed_list)
    expected_next_level = max(1, int(level)) + 1
    last_level_identity: Dict[str, Any] = {}
    transition["expected_next_adventure_level"] = expected_next_level
    transition["level_handoff_status"] = "waiting"
    transition["level_identity"] = {}

    while time.monotonic() < deadline:
        state = env.base.adventure_screen_state()
        transition["trophy_visible"] = bool(transition["trophy_visible"] or _is_trophy_state(state))
        transition["reward_screen_seen"] = bool(
            transition["reward_screen_seen"] or state.get("isRewardScreen") or state.get("rewardScreenVisible")
        )
        transition["unlock_screen_seen"] = bool(
            transition["unlock_screen_seen"]
            or state.get("unlockScreenVisible")
            or state.get("newPlantUnlockedVisible")
            or state.get("isNewPlantUnlockedScreen")
        )
        snapshot = _snapshot_unlock_state(state, source="post_win_recovery")
        if _unlock_snapshot_seen(snapshot):
            snapshots.append(snapshot)
        update_unlocked_from_state(unlocked, state, source="post_win_recovery", level=level)
        context.update(transition)

        click_target = ""
        click: Optional[Dict[str, Any]] = None

        if state.get("isGameplayReady") or state.get("gameplayReady"):
            handoff_status, level_identity = _post_win_level_handoff_status(state, level)
            last_level_identity = dict(level_identity)
            transition["level_handoff_status"] = handoff_status
            transition["level_identity"] = dict(level_identity)
            context["post_win_level_identity"] = dict(level_identity)
            if handoff_status == "confirmed":
                context["state"] = "POST_WIN_GAMEPLAY_READY"
                transition["post_win_transition_completed"] = True
                context.update(transition)
                _record_post_win_iteration(writer, env, context, state, started=started)
                break
            if handoff_status == "blocked":
                blocked_reason = (
                    "post_win_level_identity_unreliable:"
                    f"expected={expected_next_level}:"
                    f"reason={level_identity.get('level_identity_reason', 'unreliable')}:"
                    f"mismatches={','.join(str(value) for value in level_identity.get('level_identity_mismatches', []))}"
                ).rstrip(":")
                context["state"] = "POST_WIN_LEVEL_HANDOFF_BLOCKED"
                context.update(transition)
                _record_post_win_iteration(writer, env, context, state, started=started)
                break
            context["state"] = "POST_WIN_WAIT_LEVEL_HANDOFF"
            context.update(transition)
            _record_post_win_iteration(writer, env, context, state, started=started)
            time.sleep(max(POST_WIN_RECOVERY_POLL_SECONDS, env.config.poll_seconds))
            continue

        if state.get("isSeedSelectionScreen") or state.get("seedSelectionActive"):
            context["state"] = "POST_WIN_SEED_SELECTION"
            click_target = "auto_select_seeds"
            click = env.base.auto_select_seeds(seed_list=seed_names, start_level=True)
            context["last_seed_selection"] = click
            context["last_reset_reason"] = "post_win_auto_select_seeds"
            context["post_win_click_attempts"] = int(context.get("post_win_click_attempts", 0) or 0) + 1
            _record_post_win_iteration(writer, env, context, state, started=started, click_target=click_target, click_result=click)
            if not click.get("ok", False):
                blocked_reason = "post_win_bridge_state_inconsistent"
                break
            try:
                observation = env.base.wait_for_gameplay_ready(
                    timeout=max(1.0, min(env.config.gameplay_ready_timeout, deadline - time.monotonic())),
                    poll_seconds=env.config.poll_seconds,
                    quiet=True,
                    fail_on_terminal=False,
                )
                if observation.get("gameplayReady"):
                    handoff_status, level_identity = _post_win_level_handoff_status(
                        observation,
                        level,
                    )
                    last_level_identity = dict(level_identity)
                    transition["level_handoff_status"] = handoff_status
                    transition["level_identity"] = dict(level_identity)
                    context["post_win_level_identity"] = dict(level_identity)
                    if handoff_status == "confirmed":
                        transition["post_win_transition_completed"] = True
                        context.update(transition)
                        state = env.base.adventure_screen_state()
                        _record_post_win_iteration(writer, env, context, state, started=started)
                        break
                    if handoff_status == "blocked":
                        blocked_reason = (
                            "post_win_level_identity_unreliable:"
                            f"expected={expected_next_level}:"
                            f"reason={level_identity.get('level_identity_reason', 'unreliable')}:"
                            f"mismatches={','.join(str(value) for value in level_identity.get('level_identity_mismatches', []))}"
                        ).rstrip(":")
                        break
                    context["state"] = "POST_WIN_WAIT_LEVEL_HANDOFF"
                    context.update(transition)
                    time.sleep(max(POST_WIN_RECOVERY_POLL_SECONDS, env.config.poll_seconds))
                    continue
            except Exception as exc:
                context["last_error"] = str(exc)
                blocked_reason = "post_win_bridge_state_inconsistent"
                break
            blocked_reason = "post_win_seed_selection_not_reached"
            break

        if state.get("startupPopupVisible") or state.get("startupOkButtonVisible"):
            context["state"] = "POST_WIN_DISMISS_STARTUP_POPUP"
            click_target = "startup_ok"
            click = env.base.click_startup_ok_once()
            context["post_win_click_attempts"] = int(context.get("post_win_click_attempts", 0) or 0) + 1
            _record_post_win_iteration(writer, env, context, state, started=started, click_target=click_target, click_result=click)
            time.sleep(max(0.4, env.config.poll_seconds))
            continue

        if _is_trophy_state(state):
            context["state"] = "POST_WIN_CLICK_TROPHY"
            click_target = "trophy"
            click = env.base.click_trophy_once()
            context["post_win_click_attempts"] = int(context.get("post_win_click_attempts", 0) or 0) + 1
            transition["trophy_clicked"] = bool(transition["trophy_clicked"] or click.get("ok", False))
            transition["trophy_click_count"] = int(transition["trophy_click_count"]) + 1
            context.update(transition)
            _record_post_win_iteration(writer, env, context, state, started=started, click_target=click_target, click_result=click)
            time.sleep(max(0.4, env.config.poll_seconds))
            continue

        if _is_reward_or_unlock_state(state) or _is_post_win_continue_state(state):
            context["state"] = "POST_WIN_CLICK_REWARD_CONTINUE"
            click_target = "reward_continue"
            click = env.base.click_reward_continue_once()
            context["post_win_click_attempts"] = int(context.get("post_win_click_attempts", 0) or 0) + 1
            transition["reward_continue_click_count"] = int(transition["reward_continue_click_count"]) + 1
            context.update(transition)
            _record_post_win_iteration(writer, env, context, state, started=started, click_target=click_target, click_result=click)
            time.sleep(max(0.4, env.config.poll_seconds))
            continue

        if state.get("isAdventureButtonVisible"):
            context["state"] = "POST_WIN_PRESS_ADVENTURE"
            click_target = "adventure"
            click = env.base.press_adventure_once()
            context["post_win_click_attempts"] = int(context.get("post_win_click_attempts", 0) or 0) + 1
            _record_post_win_iteration(writer, env, context, state, started=started, click_target=click_target, click_result=click)
            time.sleep(max(0.5, env.config.poll_seconds))
            continue

        if _is_stable_post_win_state(state):
            context["state"] = "POST_WIN_STABLE_DELAY"
            available_after = _ordered_unique_seed_names(_state_list(state, "availableSeedNames"))
            _record_post_win_iteration(writer, env, context, state, started=started)
            time.sleep(max(POST_WIN_RECOVERY_POLL_SECONDS, env.config.poll_seconds))
            continue

        context["state"] = "POST_WIN_WAIT_TRANSITION"
        _record_post_win_iteration(writer, env, context, state, started=started)
        time.sleep(max(POST_WIN_RECOVERY_POLL_SECONDS, env.config.poll_seconds))

    if not transition["post_win_transition_completed"] and not blocked_reason:
        if last_level_identity:
            blocked_reason = (
                "post_win_level_handoff_timeout:"
                f"expected={expected_next_level}:"
                f"reason={last_level_identity.get('level_identity_reason', 'unreliable')}:"
                f"mismatches={','.join(str(value) for value in last_level_identity.get('level_identity_mismatches', []))}"
            ).rstrip(":")
        else:
            blocked_reason = _post_win_timeout_reason(state, transition)

    fallback_added = update_unlocked_from_state(
        unlocked,
        state,
        source="known_level_unlock_fallback",
        level=level,
        use_known_level_fallback=True,
    )
    combined = _combine_unlock_snapshots(snapshots)
    known_unlocks = [canonical_seed_name(name) for name in KNOWN_ADVENTURE_LEVEL_UNLOCKS.get(int(level), [])]
    if fallback_added and known_unlocks:
        combined["fallbackKnownUnlocks"] = known_unlocks
        if not combined.get("newPlantUnlockedName"):
            combined["newPlantUnlockedName"] = known_unlocks[0]
            try:
                combined["newPlantUnlockedPlantType"] = int(
                    next(
                        entry.get("plant_type_id")
                        for entry in registry_entries()
                        if canonical_seed_name(entry.get("canonical_name")) == known_unlocks[0]
                    )
                )
            except Exception:
                combined["newPlantUnlockedPlantType"] = -1
        combined["newPlantUnlockedVisible"] = bool(combined.get("newPlantUnlockedName"))

    if combined.get("newPlantUnlockedName"):
        context["latest_unlock"] = combined["newPlantUnlockedName"]
    context["unlocked_seeds"] = sorted(unlocked.keys())
    context["post_win_active"] = False
    context["post_win_elapsed"] = round(max(0.0, time.monotonic() - started), 3)
    transition["post_win_blocked_reason"] = blocked_reason
    if blocked_reason:
        context["blocked_reason"] = blocked_reason
    context.update(transition)
    unknown_objects = _unique_list(list(combined.get("unknownUnlockObjects", [])) + list(combined.get("unknownVisibleSeedCards", [])))
    return (
        state,
        bool(transition["unlock_screen_seen"]),
        combined,
        available_after,
        unknown_objects,
        blocked_reason,
        transition,
    )


def required_consecutive_wins_remaining(consecutive_wins: int, advance_on_wins: int) -> int:
    return max(0, int(advance_on_wins) - int(consecutive_wins))


def adventure_level_metadata(level: int, progression_index: int = 0) -> Dict[str, Any]:
    numeric_level = max(1, int(level or 1))
    world = (numeric_level - 1) // 10 + 1
    stage = (numeric_level - 1) % 10 + 1
    return {
        "progression_index": int(progression_index),
        "adventure_world": int(world),
        "adventure_stage": int(stage),
        "adventure_level_label": f"{world}-{stage}",
    }


def apply_level_metadata(target: Any, level: int, progression_index: int) -> None:
    for key, value in adventure_level_metadata(level, progression_index).items():
        setattr(target, key, value)


def normalize_adventure_timeout_config(
    soft_max_steps: Any,
    hard_max_steps: Any,
    final_wave_extension: Any,
) -> Tuple[int, int, bool]:
    try:
        soft = int(soft_max_steps)
    except (TypeError, ValueError):
        soft = DEFAULT_ADVENTURE_SOFT_MAX_STEPS
    try:
        hard = int(hard_max_steps)
    except (TypeError, ValueError):
        hard = DEFAULT_ADVENTURE_HARD_MAX_STEPS
    soft = max(1, soft)
    hard = max(soft, max(1, hard))
    return soft, hard, bool(final_wave_extension)


def adventure_stop_reason(reason: str) -> str:
    value = str(reason or "").strip()
    if not value:
        return "failed_level_after_max_attempts"
    lowered = value.lower()
    if lowered in {"timeout_hard_cap", "hard_cap_timeout"}:
        return "timeout_hard_cap"
    if lowered in {"timeout_soft_cap_no_extension", "soft_cap_timeout_no_extension"}:
        return "timeout_soft_cap_no_extension"
    if lowered == "timeout":
        return "timeout_hard_cap"
    if lowered.startswith("post_win_") or "post_win_transition" in lowered:
        return "post_win_transition_failed"
    if "bridge_error" in lowered:
        return "bridge_error"
    if "env_corruption" in lowered or "board_state_refreshed" in lowered:
        return "env_corruption"
    if "reset" in lowered or "seed_selection" in lowered or "startup_popup" in lowered:
        return "reset_error"
    if "loss_retry_failed" in lowered:
        return "reset_error"
    if "unhandled_screen" in lowered or "reward_screen_unhandled" in lowered or "game_over_unhandled" in lowered:
        return "unhandled_screen"
    if value == "max_attempts_reached":
        return "failed_level_after_max_attempts"
    if value == "loss":
        return "failed_level_after_max_attempts"
    return "unhandled_screen"


def _timeout_context_from_log(log: AdventureAttemptLog) -> Dict[str, Any]:
    return {
        "soft_max_steps": int(log.soft_max_steps),
        "hard_max_steps": int(log.hard_max_steps),
        "final_wave_extension_enabled": bool(log.final_wave_extension_enabled),
        "soft_timeout_reached": bool(log.soft_timeout_reached),
        "soft_timeout_extended": bool(log.soft_timeout_extended),
        "soft_timeout_step": int(log.soft_timeout_step),
        "steps_after_soft_timeout": int(log.steps_after_soft_timeout),
        "timeout_classification": str(log.timeout_classification or "none"),
    }


def update_timeout_context(context: Dict[str, Any], log: AdventureAttemptLog) -> None:
    context.update(_timeout_context_from_log(log))


def update_attempt_progress(
    attempt: AdventureAttemptLog,
    level: AdventureLevelLog,
    context: Dict[str, Any],
    advance_on_wins: int,
) -> None:
    remaining = required_consecutive_wins_remaining(level.consecutive_wins, advance_on_wins)
    attempt.wins_after = int(level.wins)
    attempt.losses_after = int(level.losses)
    attempt.consecutive_wins_after = int(level.consecutive_wins)
    attempt.advanced_after = bool(level.advanced)
    attempt.required_consecutive_wins_remaining = int(remaining)
    context["wins_this_level"] = int(level.wins)
    context["wins"] = int(level.wins)
    context["losses"] = int(level.losses)
    context["consecutive_wins"] = int(level.consecutive_wins)
    context["advanced"] = bool(level.advanced)
    context["required_consecutive_wins_remaining"] = int(remaining)


def _positive_int_or_none(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _first_positive_int(state: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[int]:
    for key in keys:
        value = _positive_int_or_none(state.get(key))
        if value is not None:
            return value
    return None


def adventure_screen_state_name(state: Dict[str, Any]) -> str:
    return lifecycle_screen_state(state)


def adventure_seed_selection_detected(state: Dict[str, Any]) -> bool:
    return adventure_seed_selection_visible(state)


def adventure_gameplay_ready_detected(state: Dict[str, Any]) -> bool:
    return adventure_gameplay_ready_visible(state)


def adventure_bridge_detected_level(state: Dict[str, Any]) -> Optional[int]:
    return _first_positive_int(state, ("currentAdventureLevel", "bridgeDetectedLevel", "detectedAdventureLevel"))


def adventure_profile_adventure_level(state: Dict[str, Any]) -> Optional[int]:
    return _first_positive_int(
        state,
        (
            "profileAdventureLevel",
            "profile_adventure_level",
            "profileCurrentAdventureLevel",
            "profile_current_adventure_level",
        ),
    )


def adventure_ui_world_level_text(state: Dict[str, Any]) -> str:
    for key in (
        "uiWorldLevelText",
        "ui_world_level_text",
        "uiLevelText",
        "ui_level_text",
        "levelText",
        "level_text",
    ):
        value = str(state.get(key) or "").strip()
        if value:
            return value
    world = _positive_int_or_none(state.get("currentWorldOrStage"))
    stage = _positive_int_or_none(state.get("currentDayLevel"))
    if world is not None and stage is not None:
        return f"{world}-{stage}"
    return ""


def adventure_current_mode_text(state: Dict[str, Any]) -> str:
    return str(
        state.get("currentMode")
        or state.get("current_mode")
        or state.get("gameBoardType")
        or state.get("game_board_type")
        or ""
    )


def adventure_challenge_mode_context(state: Dict[str, Any]) -> bool:
    mode = adventure_current_mode_text(state).replace("-", "_").replace(" ", "_").lower()
    screen_state = adventure_screen_state_name(state).replace("-", "_").lower()
    return any(marker in mode or marker in screen_state for marker in LEVEL_IDENTITY_CHALLENGE_MODE_MARKERS)


def adventure_level_identity_diagnostics(
    state: Dict[str, Any],
    expected_level: Optional[int],
    *,
    stable_screen_states: Tuple[str, ...] = ("seed_selection", "gameplay"),
    transitional_screen_states: Tuple[str, ...] = (),
) -> Dict[str, Any]:
    expected = _positive_int_or_none(expected_level)
    detected = adventure_bridge_detected_level(state)
    profile_level = adventure_profile_adventure_level(state)
    screen_state = adventure_screen_state_name(state)
    seed_selection = adventure_seed_selection_detected(state)
    gameplay_ready = adventure_gameplay_ready_detected(state)
    stable = screen_state in stable_screen_states or (seed_selection and "seed_selection" in stable_screen_states) or (
        gameplay_ready and "gameplay" in stable_screen_states
    )
    transitional = screen_state in transitional_screen_states
    challenge_context = adventure_challenge_mode_context(state)
    mismatches: List[str] = []
    if expected is not None and detected is not None and detected != expected:
        mismatches.append(f"bridge_detected_level:{detected}!={expected}")
    if expected is not None and profile_level is not None and profile_level != expected:
        mismatches.append(f"profile_adventure_level:{profile_level}!={expected}")
    source_available = detected is not None or profile_level is not None
    reliable = bool(stable and source_available and not transitional and not challenge_context and not mismatches)
    if challenge_context:
        reason = "challenge_mode_context"
    elif transitional:
        reason = "transitional_screen_state"
    elif mismatches:
        reason = "level_source_mismatch"
    elif stable and not source_available:
        reason = "level_source_unavailable"
    elif stable:
        reason = "ok"
    else:
        reason = "navigation_or_unstable_screen"
    return {
        "wrapper_expected_level": expected,
        "bridge_detected_level": detected,
        "profile_adventure_level": profile_level,
        "profile_adventure_level_source": str(
            state.get("profileAdventureLevelSource") or state.get("profile_adventure_level_source") or ""
        ),
        "ui_world_level_text": adventure_ui_world_level_text(state),
        "screenState": screen_state,
        "currentMode": adventure_current_mode_text(state),
        "seedSelectionDetected": bool(seed_selection),
        "gameplayReadyDetected": bool(gameplay_ready),
        "level_identity_reliable": bool(reliable),
        "level_identity_reason": reason,
        "level_identity_mismatches": mismatches,
        "challenge_mode_context": bool(challenge_context),
        "transitional_screen_state": bool(transitional),
    }


def _rollback_seed_selection_transaction(env: PvZMaskedPPOEnv) -> None:
    hook = getattr(env, "_rollback_pending_seed_selection", None)
    if callable(hook):
        hook()


def _commit_seed_selection_transaction(
    env: PvZMaskedPPOEnv,
    selection: Dict[str, Any],
    seed_list: List[str],
) -> bool:
    hook = getattr(env, "_commit_pending_seed_selection", None)
    if not callable(hook):
        return True
    return bool(hook(selection, list(seed_list)))


def _interactive_seed_selection_after_transition(
    selection: Dict[str, Any],
    state: Dict[str, Any],
) -> bool:
    """Return true only for a proven safe post-transition chooser retry."""

    transition_status = str(selection.get("transitionStatus") or "")
    if bool(selection.get("startInvoked")):
        # Once Let's Rock owns the transition, only the bounded multi-frame
        # stability check inside ``auto_select_seeds`` may authorize another
        # chooser transaction.  A raw/stale seed flag is insufficient.
        return transition_status == "seed_selection_still_interactive"
    if transition_status == "seed_selection_still_interactive":
        return True
    if not isinstance(state, dict):
        return False
    seed_visible = bool(
        state.get("isSeedSelectionScreen")
        or state.get("seedSelectionActive")
        or state.get("screenState") == "seed_selection"
    )
    if not seed_visible or bool(state.get("isGameplayReady") or state.get("gameplayReady")):
        return False
    strict_fields_present = "seedSelectionPanelActive" in state or "startButtonActive" in state
    if strict_fields_present:
        return bool(state.get("seedSelectionPanelActive") and state.get("startButtonActive"))
    # Compatibility for an older bridge response before a start click.  This
    # path can never authorize a retry after start invocation.
    return True


def replay_current_level_after_validation_win(
    env: PvZMaskedPPOEnv,
    writer: LiveStatusWriter,
    context: Dict[str, Any],
    timeout: float = 8.0,
    expected_level: Optional[int] = None,
    seed_selection_callback: Optional[Callable[[Dict[str, Any], List[str]], Tuple[List[str], str]]] = None,
) -> Tuple[bool, str]:
    run_mode = str(context.get("run_mode") or context.get("mode") or "")
    log_prefix = "[adventure-generalist]" if "generalist" in run_mode else "[adventure]"
    context["state"] = "REPLAY_CURRENT_LEVEL"
    context["last_reset_reason"] = "same_level_validation_replay"
    context["reset_phase"] = "request_same_level_replay"
    context["expected_transition_target"] = "same_level_replay"
    context["seed_selection_expected"] = True
    poll_seconds = float(getattr(env.config, "poll_seconds", 0.2) or 0.2)
    seed_names = list(context.get("selected_seeds", []) or getattr(env.config, "seed_list", []))
    expected_level_int: Optional[int] = None
    if expected_level is not None:
        try:
            expected_level_int = int(expected_level)
        except (TypeError, ValueError):
            expected_level_int = None

    def observed_positive_level(state: Dict[str, Any]) -> Optional[int]:
        raw_level = state.get("currentAdventureLevel")
        if raw_level in (None, ""):
            return None
        try:
            level = int(raw_level)
        except (TypeError, ValueError):
            return None
        return level if level > 0 else None

    def level_mismatch(state: Dict[str, Any]) -> bool:
        if expected_level_int is None:
            return False
        actual_level = observed_positive_level(state)
        if actual_level is None:
            return False
        return int(actual_level) != int(expected_level_int)

    def blocked_for_level_mismatch(state: Dict[str, Any]) -> Tuple[bool, str]:
        if not level_mismatch(state):
            return False, ""
        actual_level = observed_positive_level(state)
        diagnostics = adventure_level_identity_diagnostics(
            state,
            expected_level_int,
            stable_screen_states=("gameplay",),
            transitional_screen_states=tuple(LEVEL_IDENTITY_TRANSITIONAL_STATES),
        )
        context["frontier_replay_level_after_win"] = actual_level
        context["frontier_replay_last_state"] = _post_win_last_state(state)
        context["frontier_replay_level_identity"] = diagnostics
        context["level_identity"] = diagnostics
        context["wrapper_expected_level"] = diagnostics.get("wrapper_expected_level")
        context["bridge_detected_level"] = diagnostics.get("bridge_detected_level")
        context["profile_adventure_level"] = diagnostics.get("profile_adventure_level")
        context["ui_world_level_text"] = diagnostics.get("ui_world_level_text", "")
        context["level_identity_reliable"] = bool(diagnostics.get("level_identity_reliable"))
        if not replay_level_check_authoritative(state):
            context["frontier_replay_ignored_level_mismatch"] = (
                f"{actual_level}!={expected_level_int} screenState={screen_state_name(state) or 'unknown'}"
            )
            print(
                f"{log_prefix} same_level_replay level_mismatch_ignored "
                f"screenState={screen_state_name(state) or 'unknown'} "
                f"expected_level={expected_level_int} "
                f"detected_level={actual_level} "
                f"profile_adventure_level={diagnostics.get('profile_adventure_level') or 'unknown'} "
                f"ui_world_level_text={diagnostics.get('ui_world_level_text') or 'unknown'} "
                "level_identity_reliable=false "
                f"reason={diagnostics.get('level_identity_reason') or 'unknown'}"
            )
            return False, ""
        return True, f"same_level_replay_advanced_to_unexpected_level:{actual_level}!={expected_level_int}"

    screen_state_name = adventure_screen_state_name

    def raw_level_text(state: Dict[str, Any]) -> str:
        raw_level = state.get("currentAdventureLevel")
        return "unknown" if raw_level in (None, "") else str(raw_level)

    def seed_selection_visible(state: Dict[str, Any]) -> bool:
        return bool(
            state.get("isSeedSelectionScreen")
            or state.get("seedSelectionActive")
            or screen_state_name(state) == "seed_selection"
        )

    def gameplay_ready_visible(state: Dict[str, Any]) -> bool:
        return bool(
            (state.get("isGameplayReady") or state.get("gameplayReady") or screen_state_name(state) == "gameplay")
            and not seed_selection_visible(state)
        )

    startup_popup_visible = adventure_startup_visible

    def actual_adventure_menu_visible(state: Dict[str, Any]) -> bool:
        screen_state = screen_state_name(state)
        return bool(
            state.get("isAdventureButtonVisible")
            and screen_state in {"main_menu", "loading_or_menu", "adventure_menu"}
            and not _is_post_win_continue_state(state)
            and not startup_popup_visible(state)
            and not seed_selection_visible(state)
            and not gameplay_ready_visible(state)
        )

    last_logged_state_key: Optional[Tuple[str, str, bool, bool, bool, bool]] = None

    def log_replay_state(state: Dict[str, Any], *, reason: str) -> None:
        nonlocal last_logged_state_key
        key = (
            screen_state_name(state),
            raw_level_text(state),
            bool(seed_selection_visible(state)),
            bool(gameplay_ready_visible(state)),
            bool(state.get("isAdventureButtonVisible")),
            bool(startup_popup_visible(state)),
        )
        if key == last_logged_state_key:
            return
        last_logged_state_key = key
        print(
            f"{log_prefix} same_level_replay state "
            f"reason={reason} "
            f"screenState={key[0] or 'unknown'} "
            f"expected_level={expected_level_int if expected_level_int is not None else 'unknown'} "
            f"detected_level={key[1]} "
            f"seedSelectionDetected={'true' if key[2] else 'false'} "
            f"gameplayReadyDetected={'true' if key[3] else 'false'} "
            f"adventureButtonVisible={'true' if key[4] else 'false'} "
            f"profile_adventure_level={adventure_profile_adventure_level(state) or 'unknown'} "
            f"ui_world_level_text={adventure_ui_world_level_text(state) or 'unknown'} "
            f"level_identity_reliable={'true' if adventure_level_identity_diagnostics(state, expected_level_int, stable_screen_states=('gameplay',), transitional_screen_states=tuple(LEVEL_IDENTITY_TRANSITIONAL_STATES)).get('level_identity_reliable') else 'false'}"
        )

    def replay_state_is_recoverable(state: Dict[str, Any]) -> bool:
        return bool(
            state.get("startupPopupVisible")
            or state.get("startupOkButtonVisible")
            or state.get("isAdventureButtonVisible")
            or state.get("isSeedSelectionScreen")
            or state.get("seedSelectionActive")
            or state.get("isGameplayReady")
            or state.get("gameplayReady")
        )

    def replay_level_check_authoritative(state: Dict[str, Any]) -> bool:
        return bool(gameplay_ready_visible(state) and screen_state_name(state) not in LEVEL_IDENTITY_TRANSITIONAL_STATES)

    try:
        before_state = env.base.adventure_screen_state()
        context["frontier_replay_last_state"] = _post_win_last_state(before_state)
        print(
            f"{log_prefix} reset "
            "reason=same_level_replay "
            f"current_screen={before_state.get('screenState')} "
            "expected_transition_target=same_level_replay "
            "seed_selection_expected=True "
            f"expected_level={expected_level_int if expected_level_int is not None else 'unknown'}"
        )
        if _is_post_win_continue_state(before_state):
            print(
                f"{log_prefix} same_level_replay post_win_ui_detected "
                f"screenState={screen_state_name(before_state) or 'unknown'} "
                "action=auto_reset_hook"
            )
        writer.write(build_live_status(env, context, adventure_state=before_state))
        mismatch, reason = blocked_for_level_mismatch(before_state)
        if mismatch:
            return False, reason
        replay = env.base.auto_reset(
            start_sun=getattr(env.config, "start_sun", None),
            allow_active_gameplay_reset=False,
            reset_reason="same_level_validation_replay",
            require_seed_selection_path=True,
        )
    except Exception as exc:
        context["last_error"] = str(exc)
        return False, "win_replay_reset_failed"
    context["last_ui_action"] = replay
    context["frontier_replay_auto_reset"] = replay
    context["frontier_replay_auto_reset_ok"] = bool(replay.get("ok", True))
    context["frontier_replay_auto_reset_method"] = str(replay.get("methodUsed") or "")
    print(
        f"{log_prefix} same_level_replay auto_reset "
        f"ok={bool(replay.get('ok', True))} "
        f"method={replay.get('methodUsed')} "
        f"message={replay.get('message', '')}"
    )
    if not replay.get("ok", True):
        try:
            state = env.base.adventure_screen_state()
        except Exception:
            state = replay.get("observation", {}) if isinstance(replay.get("observation"), dict) else {}
        context["frontier_replay_last_state"] = _post_win_last_state(state)
        mismatch, reason = blocked_for_level_mismatch(state)
        if mismatch:
            return False, reason
        if replay_state_is_recoverable(state):
            context["frontier_replay_auto_reset_warning"] = str(replay.get("message") or "auto_reset_failed_recoverable_state")
            print(
                f"{log_prefix} same_level_replay auto_reset_failed "
                f"screenState={screen_state_name(state) or 'unknown'} "
                f"recoverable=true "
                f"message={replay.get('message', '')}"
            )
        return False, str(replay.get("message") or "win_replay_reset_failed")
    replay_observation = replay.get("observation", {}) if isinstance(replay, dict) else {}
    if isinstance(replay_observation, dict):
        context["frontier_replay_reset_observation"] = _post_win_last_state(replay_observation)
        print(
            f"{log_prefix} same_level_replay reset_observation "
            f"screenState={screen_state_name(replay_observation) or 'unknown'} "
            f"expected_level={expected_level_int if expected_level_int is not None else 'unknown'} "
            f"detected_level={raw_level_text(replay_observation)} "
            f"seedSelectionDetected={'true' if seed_selection_visible(replay_observation) else 'false'} "
            f"gameplayReadyDetected={'true' if gameplay_ready_visible(replay_observation) else 'false'}"
        )
    if isinstance(replay_observation, dict) and gameplay_ready_visible(replay_observation):
        state = env.base.adventure_screen_state()
        context["frontier_replay_last_state"] = _post_win_last_state(state)
        log_replay_state(state, reason="gameplay_ready_after_auto_reset")
        mismatch, reason = blocked_for_level_mismatch(state)
        if mismatch:
            return False, reason
        print(
            f"{log_prefix} same_level_replay gameplayReady detected "
            f"screenState={screen_state_name(state) or 'unknown'} "
            f"expected_level={expected_level_int if expected_level_int is not None else 'unknown'} "
            f"detected_level={raw_level_text(state)}"
        )
        return True, ""

    deadline = time.monotonic() + max(0.5, timeout)
    adventure_skip_logged = False
    seed_selection_attempts = 0
    while time.monotonic() < deadline:
        try:
            state = env.base.adventure_screen_state()
        except Exception as exc:
            context["last_error"] = str(exc)
            return False, "win_replay_reset_failed"
        writer.write(build_live_status(env, context, adventure_state=state))
        context["frontier_replay_last_state"] = _post_win_last_state(state)
        log_replay_state(state, reason="poll")
        mismatch, reason = blocked_for_level_mismatch(state)
        if mismatch:
            return False, reason
        if _is_post_win_continue_state(state):
            context["state"] = "REPLAY_CURRENT_LEVEL_WAIT_RESET"
            context["reset_phase"] = "wait_reset_hook"
            if state.get("isAdventureButtonVisible") and not actual_adventure_menu_visible(state) and not adventure_skip_logged:
                adventure_skip_logged = True
                context["frontier_replay_press_adventure_skipped"] = True
                context["frontier_replay_press_adventure_skip_reason"] = "reset_hook_succeeded_not_menu"
                print(
                    f"{log_prefix} same_level_replay press_adventure skipped "
                    "reason=reset_hook_succeeded_not_menu "
                    f"screenState={screen_state_name(state) or 'unknown'} "
                    f"expected_level={expected_level_int if expected_level_int is not None else 'unknown'} "
                    f"detected_level={raw_level_text(state)}"
                )
            time.sleep(max(0.05, poll_seconds))
            continue
        if startup_popup_visible(state):
            context["state"] = "REPLAY_CURRENT_LEVEL_STARTUP_POPUP"
            context["reset_phase"] = "dismiss_startup_popup"
            try:
                click = env.base.click_startup_ok_once()
            except Exception as exc:
                context["last_error"] = str(exc)
                return False, "win_replay_startup_popup_failed"
            context["last_ui_action"] = click
            if not click.get("ok", False):
                return False, "win_replay_startup_popup_failed"
            time.sleep(max(0.05, poll_seconds))
            continue
        if actual_adventure_menu_visible(state):
            context["state"] = "REPLAY_CURRENT_LEVEL_ADVENTURE_MENU"
            context["reset_phase"] = "press_adventure"
            try:
                click = env.base.press_adventure_once()
            except Exception as exc:
                context["last_error"] = str(exc)
                context["frontier_replay_press_adventure_warning"] = f"exception:{exc}"
                print(
                    f"{log_prefix} same_level_replay press_adventure warning "
                    f"ok=false method= exception={exc}"
                )
                time.sleep(max(0.1, poll_seconds))
                continue
            context["last_ui_action"] = click
            print(
                f"{log_prefix} same_level_replay press_adventure "
                "used=true "
                f"ok={click.get('ok')} method={click.get('methodUsed')}"
            )
            if not click.get("ok", False):
                context["frontier_replay_press_adventure_warning"] = "win_replay_press_adventure_failed"
                print(
                    f"{log_prefix} same_level_replay press_adventure warning "
                    "reason=win_replay_press_adventure_failed "
                    f"screenState={screen_state_name(state) or 'unknown'}"
                )
                time.sleep(max(0.1, poll_seconds))
                continue
            time.sleep(max(0.1, poll_seconds))
            continue
        if state.get("isAdventureButtonVisible") and not adventure_skip_logged:
            adventure_skip_logged = True
            context["frontier_replay_press_adventure_skipped"] = True
            context["frontier_replay_press_adventure_skip_reason"] = "not_main_menu"
            print(
                f"{log_prefix} same_level_replay press_adventure skipped "
                "reason=not_main_menu "
                f"screenState={screen_state_name(state) or 'unknown'} "
                f"expected_level={expected_level_int if expected_level_int is not None else 'unknown'} "
                f"detected_level={raw_level_text(state)}"
            )
        if seed_selection_visible(state):
            context["state"] = "REPLAY_CURRENT_LEVEL_SEED_SELECTION"
            context["reset_phase"] = "seed_selection"
            context["frontier_replay_seed_selection_detected"] = True
            print(
                f"{log_prefix} same_level_replay seed_selection detected "
                f"screenState={screen_state_name(state) or 'unknown'} "
                f"expected_level={expected_level_int if expected_level_int is not None else 'unknown'} "
                f"detected_level={raw_level_text(state)}"
            )
            if level_mismatch(state) and not replay_level_check_authoritative(state):
                context["frontier_replay_waiting_for_reliable_level_identity"] = True
                time.sleep(max(0.05, poll_seconds))
                continue
            seed_selection_attempts += 1
            context["frontier_replay_seed_selection_attempts"] = seed_selection_attempts
            if seed_selection_callback is not None:
                try:
                    callback_seed_list, callback_blocked_reason = seed_selection_callback(state, list(seed_names))
                except Exception as exc:
                    _rollback_seed_selection_transaction(env)
                    callback_seed_list, callback_blocked_reason = [], f"seed_selection_callback_failed:{exc}"
                if callback_blocked_reason:
                    _rollback_seed_selection_transaction(env)
                    return False, str(callback_blocked_reason)
                if callback_seed_list:
                    seed_names = [str(seed).strip() for seed in callback_seed_list if str(seed).strip()]
            try:
                selection = env.base.auto_select_seeds(seed_list=seed_names, start_level=True)
            except Exception as exc:
                _rollback_seed_selection_transaction(env)
                context["last_error"] = str(exc)
                context["frontier_replay_last_seed_selection_message"] = f"exception:{exc}"
                print(
                    f"{log_prefix} same_level_replay seed_selection warning "
                    f"attempt={seed_selection_attempts} ok=false exception={exc}"
                )
                time.sleep(max(0.1, poll_seconds))
                continue
            context["last_seed_selection"] = selection
            context["frontier_replay_last_seed_selection_ok"] = bool(selection.get("ok", False))
            context["frontier_replay_last_seed_selection_message"] = str(selection.get("message", ""))
            context["frontier_replay_last_seed_selection_actions"] = list(selection.get("actions", []) or [])[-8:]
            context["frontier_replay_last_seed_selection_start_log"] = dict(selection.get("startLog", {}) or {})
            if selection.get("ok", False):
                try:
                    committed = _commit_seed_selection_transaction(env, selection, seed_names)
                except Exception as exc:
                    committed = False
                    selection = dict(selection)
                    selection["message"] = f"seed_selection_commit_failed:{exc}"
                if not committed:
                    _rollback_seed_selection_transaction(env)
                    selection = dict(selection)
                    selection["ok"] = False
                    selection["message"] = str(
                        selection.get("message") or "seed_selection_commit_failed"
                    )
                    context["frontier_replay_last_seed_selection_ok"] = False
                    return False, str(selection["message"])
            if not selection.get("ok", False):
                _rollback_seed_selection_transaction(env)
                selection_state = selection.get("after") or selection.get("afterStart") or selection.get("afterSelectionBeforeStart") or state
                if isinstance(selection_state, dict):
                    context["frontier_replay_last_state"] = _post_win_last_state(selection_state)
                    mismatch, reason = blocked_for_level_mismatch(selection_state)
                    if mismatch:
                        return False, reason
                print(
                    f"{log_prefix} same_level_replay seed_selection warning "
                    f"attempt={seed_selection_attempts} ok=false "
                    f"message={selection.get('message', '')} "
                    f"actions={list(selection.get('actions', []) or [])[-4:]}"
                )
                writer.write(build_live_status(env, context, adventure_state=selection_state if isinstance(selection_state, dict) else state))
                time.sleep(max(0.2, poll_seconds, float(getattr(env.config, "seed_click_delay", 0.35) or 0.35)))
                continue
            try:
                observation = env.base.wait_for_gameplay_ready(
                    timeout=max(1.0, min(timeout, deadline - time.monotonic() + 1.0)),
                    poll_seconds=poll_seconds,
                    quiet=True,
                    fail_on_terminal=False,
                )
            except Exception as exc:
                context["last_error"] = str(exc)
                print(
                    f"{log_prefix} same_level_replay gameplay_ready warning "
                    f"attempt={seed_selection_attempts} exception={exc}"
                )
                time.sleep(max(0.1, poll_seconds))
                continue
            state = env.base.adventure_screen_state()
            context["frontier_replay_last_state"] = _post_win_last_state(state)
            log_replay_state(state, reason="after_seed_selection")
            mismatch, reason = blocked_for_level_mismatch(state)
            if mismatch:
                return False, reason
            if gameplay_ready_visible(observation) or gameplay_ready_visible(state):
                context["state"] = "REPLAY_CURRENT_LEVEL_READY"
                context["reset_phase"] = "done"
                context["frontier_replay_gameplay_ready_detected"] = True
                print(
                    f"{log_prefix} same_level_replay gameplayReady detected "
                    f"screenState={screen_state_name(state) or screen_state_name(observation) or 'unknown'} "
                    f"expected_level={expected_level_int if expected_level_int is not None else 'unknown'} "
                    f"detected_level={raw_level_text(state)}"
                )
                return True, ""
            print(
                f"{log_prefix} same_level_replay gameplay_ready warning "
                f"attempt={seed_selection_attempts} ready=false "
                f"screenState={screen_state_name(state) or screen_state_name(observation) or 'unknown'}"
            )
            time.sleep(max(0.1, poll_seconds))
            continue
        if gameplay_ready_visible(state):
            context["state"] = "REPLAY_CURRENT_LEVEL_READY"
            context["reset_phase"] = "done"
            context["frontier_replay_gameplay_ready_detected"] = True
            print(
                f"{log_prefix} same_level_replay gameplayReady detected "
                f"screenState={screen_state_name(state) or 'unknown'} "
                f"expected_level={expected_level_int if expected_level_int is not None else 'unknown'} "
                f"detected_level={raw_level_text(state)}"
            )
            return True, ""
        time.sleep(max(0.05, poll_seconds))
    context["frontier_replay_timeout"] = True
    timeout_reason = "win_replay_seed_selection_or_gameplay_timeout"
    print(
        f"{log_prefix} same_level_replay timeout "
        f"reason={timeout_reason} "
        f"expected_level={expected_level_int if expected_level_int is not None else 'unknown'}"
    )
    return False, timeout_reason


def dismiss_startup_popup_if_needed(
    env: PvZMaskedPPOEnv,
    writer: LiveStatusWriter,
    context: Dict[str, Any],
    state: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Dict[str, Any], str]:
    state = state or env.base.adventure_screen_state()
    if not (state.get("startupPopupVisible") or state.get("startupOkButtonVisible")):
        return False, state, ""
    context["state"] = "DISMISS_STARTUP_POPUP"
    writer.write(build_live_status(env, context, adventure_state=state))
    click = env.base.click_startup_ok_once()
    context["last_ui_action"] = click
    if not click.get("ok", False):
        return True, state, "startup_popup_dismiss_failed"
    dismissed = env.base.wait_for_startup_popup_dismissed(timeout=5.0, poll_seconds=env.config.poll_seconds)
    writer.write(build_live_status(env, context, adventure_state=dismissed))
    if dismissed.get("startupPopupVisible") or dismissed.get("startupOkButtonVisible"):
        return True, dismissed, "startup_popup_still_visible_after_ok"
    return True, dismissed, ""


def prepare_adventure_gameplay(
    env: PvZMaskedPPOEnv,
    writer: LiveStatusWriter,
    context: Dict[str, Any],
    seed_list: List[str],
    timeout: float,
    seed_selection_callback: Optional[Callable[[Dict[str, Any], List[str]], Tuple[List[str], str]]] = None,
    expected_level: Optional[int] = None,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any], str]:
    deadline = time.monotonic() + max(1.0, timeout)
    last_state: Dict[str, Any] = {}
    active_seed_list = [str(seed).strip() for seed in seed_list if str(seed).strip()]
    seed_selection_attempts = 0
    adventure_click_attempts = 0

    def level_identity_block(state: Dict[str, Any]) -> str:
        if expected_level is None:
            return ""
        diagnostics = adventure_level_identity_diagnostics(state, int(expected_level))
        if diagnostics.get("level_identity_reliable"):
            return ""
        reason = str(diagnostics.get("level_identity_reason") or "unreliable")
        mismatches = ",".join(str(value) for value in diagnostics.get("level_identity_mismatches", []))
        return f"adventure_level_identity_unreliable:{reason}:{mismatches}".rstrip(":")

    while time.monotonic() < deadline:
        state = env.base.adventure_screen_state()
        last_state = state
        writer.write(build_live_status(env, context, adventure_state=state))

        dismissed, state, popup_error = dismiss_startup_popup_if_needed(env, writer, context, state)
        if popup_error:
            return None, {"reset": {"ok": False, "methodUsed": "click_startup_ok_once"}}, popup_error
        if dismissed:
            time.sleep(max(0.25, env.config.poll_seconds))
            continue

        # A terminal board can remain structurally present for a few frames.
        # Loss/restart evidence must outrank that stale gameplay-ready signal.
        if _is_loss_restart_state(state):
            context["state"] = "LOSS_RECOVERY_CLICK_RESTART"
            click = env.base.click_try_again_once()
            context["last_ui_action"] = click
            if not click.get("ok", False):
                return None, {"reset": {"ok": False, "methodUsed": "click_try_again_once"}}, "game_over_unhandled"
            time.sleep(max(0.5, env.config.poll_seconds))
            continue

        if state.get("isGameplayReady"):
            identity_block = level_identity_block(state)
            if identity_block:
                return None, {"reset": {"ok": False, "methodUsed": "level_identity"}}, identity_block
            observation = env.base.wait_for_gameplay_ready(
                timeout=max(1.0, min(timeout, deadline - time.monotonic() + 1.0)),
                poll_seconds=env.config.poll_seconds,
                quiet=True,
                fail_on_terminal=False,
            )
            context["last_reset_reason"] = "adventure_existing_gameplay"
            return observation, {"reset": {"ok": True, "methodUsed": "adventure_existing_gameplay"}}, ""

        if state.get("isAdventureButtonVisible"):
            adventure_click_attempts += 1
            context["adventure_click_attempts"] = adventure_click_attempts
            click = env.base.press_adventure_once()
            context["last_ui_action"] = click
            if not click.get("ok", False) and adventure_click_attempts >= 3:
                return None, {"reset": {"ok": False, "methodUsed": "press_adventure_once"}}, (
                    "adventure_navigation_click_failed: "
                    + str(click.get("message") or "Adventure button click failed")
                )
            if click.get("ok", False) and adventure_click_attempts >= 6:
                return None, {"reset": {"ok": False, "methodUsed": "press_adventure_once"}}, (
                    "adventure_navigation_no_transition: "
                    + str(click.get("methodUsed") or "Adventure click reported success")
                )
            time.sleep(max(0.5, env.config.poll_seconds))
            continue

        if state.get("isSeedSelectionScreen"):
            identity_block = level_identity_block(state)
            if identity_block:
                return None, {"reset": {"ok": False, "methodUsed": "level_identity"}}, identity_block
            seed_selection_attempts += 1
            context["seed_selection_attempts"] = seed_selection_attempts
            context["state"] = "SEED_SELECTION"
            if seed_selection_callback is not None:
                try:
                    callback_seed_list, callback_blocked_reason = seed_selection_callback(state, list(active_seed_list))
                except Exception as exc:
                    _rollback_seed_selection_transaction(env)
                    callback_seed_list, callback_blocked_reason = [], f"seed_selection_callback_failed:{exc}"
                if callback_blocked_reason:
                    _rollback_seed_selection_transaction(env)
                    return None, {"reset": {"ok": False, "methodUsed": "seed_selection_callback"}}, callback_blocked_reason
                if callback_seed_list:
                    active_seed_list = [str(seed).strip() for seed in callback_seed_list if str(seed).strip()]
            try:
                selection = env.base.auto_select_seeds(seed_list=active_seed_list, start_level=True)
            except Exception as exc:
                _rollback_seed_selection_transaction(env)
                return None, {"reset": {"ok": False, "methodUsed": "auto_select_seeds"}}, (
                    f"seed_selection_failed: {exc}"
                )
            context["last_seed_selection"] = selection
            context["last_reset_reason"] = "auto_select_seeds"
            transition_status = str(selection.get("transitionStatus") or "")
            context["seed_selection_transition_status"] = transition_status
            context["seed_selection_start_invoked"] = bool(selection.get("startInvoked"))
            if selection.get("startInvoked"):
                context["state"] = (
                    "GAMEPLAY_READY"
                    if transition_status == "gameplay_confirmed"
                    else "WAITING_FOR_GAMEPLAY"
                )
            if selection.get("ok", False):
                try:
                    committed = _commit_seed_selection_transaction(env, selection, active_seed_list)
                except Exception as exc:
                    committed = False
                    selection = dict(selection)
                    selection["message"] = f"seed_selection_commit_failed:{exc}"
                if not committed:
                    _rollback_seed_selection_transaction(env)
                    return None, {"reset": {"ok": False, "methodUsed": "seed_selection_commit"}}, (
                        "seed_selection_failed: "
                        + str(selection.get("message") or "seed_selection_commit_failed")
                    )
            if not selection.get("ok", False):
                _rollback_seed_selection_transaction(env)
                selection_state = (
                    selection.get("after")
                    or selection.get("afterStart")
                    or selection.get("afterSelectionBeforeStart")
                    or state
                )
                still_on_seed_screen = isinstance(selection_state, dict) and bool(
                    selection_state.get("isSeedSelectionScreen")
                    or selection_state.get("seedSelectionActive")
                    or selection_state.get("screenState") == "seed_selection"
                )
                gameplay_ready = isinstance(selection_state, dict) and bool(
                    selection_state.get("isGameplayReady") or selection_state.get("gameplayReady")
                )
                safe_retry = _interactive_seed_selection_after_transition(selection, selection_state)
                if seed_selection_attempts < 3 and still_on_seed_screen and not gameplay_ready and safe_retry:
                    retry_reason = str(selection.get("message", "unknown"))
                    context["seed_selection_retry_reason"] = retry_reason
                    print(
                        "[adventure] seed selection retry "
                        f"attempt={seed_selection_attempts} "
                        f"transition_status={transition_status or 'pre_start_failure'} "
                        f"reason={retry_reason}"
                    )
                    time.sleep(max(0.1, env.config.poll_seconds))
                    continue
                return None, {"reset": {"ok": False, "methodUsed": "auto_select_seeds"}}, (
                    "seed_selection_failed: " + str(selection.get("message", "unknown"))
                )
            final_observation = selection.get("finalObservation")
            if isinstance(final_observation, dict) and bool(
                final_observation.get("gameplayReady")
                and not final_observation.get("seedSelectionActive")
            ):
                observation = final_observation
            else:
                context["state"] = "WAITING_FOR_GAMEPLAY"
                observation = env.base.wait_for_gameplay_ready(
                    timeout=max(1.0, min(timeout, deadline - time.monotonic() + 1.0)),
                    poll_seconds=env.config.poll_seconds,
                    quiet=True,
                    fail_on_terminal=False,
                )
            context["state"] = "GAMEPLAY_READY"
            context["last_reset_reason"] = "auto_select_seeds"
            return (
                observation,
                {
                    "reset": {
                        "ok": True,
                        "methodUsed": "auto_select_seeds",
                        "autoSelectSeeds": selection,
                        "seedList": list(active_seed_list),
                    }
                },
                "",
            )

        if state.get("isRewardScreen") or state.get("blockingRewardUiActive"):
            click = env.base.click_reward_continue_once()
            context["last_ui_action"] = click
            if not click.get("ok", False):
                return None, {"reset": {"ok": False, "methodUsed": "click_reward_continue_once"}}, "reward_screen_unhandled"
            time.sleep(max(0.5, env.config.poll_seconds))
            continue

        if state.get("boardFound") and not state.get("isGameplayReady") and not state.get("isSeedSelectionScreen"):
            try:
                observation = env.base.observe(force_restart_probe=True)
            except Exception:
                observation = {}
            if observation and is_restart_screen_observation(observation):
                click = env.base.click_try_again_once()
                context["last_ui_action"] = click
                if not click.get("ok", False):
                    return None, {"reset": {"ok": False, "methodUsed": "click_try_again_once"}}, "game_over_unhandled"
                time.sleep(max(0.5, env.config.poll_seconds))
                continue

        time.sleep(max(0.05, env.config.poll_seconds))

    screen_state = str(last_state.get("screenState") or "unknown")
    return None, {"reset": {"ok": False, "methodUsed": "adventure_prepare_timeout"}}, f"unhandled_screen:{screen_state}"


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _episode_summary_from_info(info: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("episode_summary", "episode_summary_candidate"):
        summary = info.get(key) if isinstance(info, dict) else {}
        if isinstance(summary, dict) and summary:
            return dict(summary)
    return {}


def _observation_from_info(env: PvZMaskedPPOEnv, info: Dict[str, Any]) -> Dict[str, Any]:
    raw = info.get("raw_observation", {}) if isinstance(info, dict) else {}
    if isinstance(raw, dict) and raw:
        return raw
    observation = getattr(env, "_last_observation", None)
    return observation if isinstance(observation, dict) else {}


def _adventure_state_for_terminal(
    env: PvZMaskedPPOEnv,
    info: Dict[str, Any],
    raw: Dict[str, Any],
) -> Dict[str, Any]:
    adventure_state = info.get("adventure_state", {}) if isinstance(info, dict) else {}
    if isinstance(adventure_state, dict) and adventure_state:
        return adventure_state
    try:
        state = env.base.adventure_screen_state()
        return state if isinstance(state, dict) else {}
    except Exception:
        return {"screenState": raw.get("screenState", "")}


def _timeout_summary(
    log: AdventureAttemptLog,
    info: Dict[str, Any],
    raw: Dict[str, Any],
    terminal_reason: str,
) -> Dict[str, Any]:
    summary = _episode_summary_from_info(info)
    if not summary:
        summary = {
            "run_mode": ADVENTURE_GENERALIST_EVAL_RUN_MODE,
            "result": "timeout",
            "reward_total": float(log.episode_reward),
            "episode_reward": float(log.episode_reward),
            "episode_length": int(log.episode_length),
            "final_wave": _safe_int(raw.get("wave"), 0),
            "max_wave": _safe_int(raw.get("maxWave"), 0),
            "zombies_killed": 0,
            "plants_placed": 0,
            "sun_remaining": _safe_int(raw.get("sun"), 0),
            "mowers_lost": 0,
            "bridge_errors": int(log.bridge_errors),
            "illegal_actions": int(log.illegal_actions),
            "reset_success": True,
        }
    summary.update(
        {
            "result": "timeout",
            "done_reason": "timeout",
            "terminal_reason": terminal_reason,
            "episode_length": int(log.episode_length),
            "episode_reward": float(log.episode_reward),
            "reward_total": float(summary.get("reward_total", log.episode_reward) or log.episode_reward),
            "timeout": True,
            "win": False,
            "loss": False,
        }
    )
    return summary


def _with_timeout_diagnostics(summary: Dict[str, Any], log: AdventureAttemptLog) -> Dict[str, Any]:
    summary.update(_timeout_context_from_log(log))
    summary["timeout_classification"] = str(log.timeout_classification or "none")
    return summary


def _soft_extension_decision(
    observation: Dict[str, Any],
    adventure_state: Dict[str, Any],
    info: Dict[str, Any],
    *,
    enabled: bool,
) -> Tuple[bool, str]:
    screen_state = str(adventure_state.get("screenState") or observation.get("screenState") or "")
    gameplay = bool(
        screen_state == "gameplay"
        or adventure_state.get("isGameplayReady")
        or observation.get("gameplayReady")
        or observation.get("actualGameplayReady")
    )
    wave = _safe_int(observation.get("wave"), 0)
    max_wave = _safe_int(observation.get("maxWave"), 0)
    final_wave_reached = bool(max_wave > 0 and wave >= max_wave)
    terminal_hint = str(observation.get("terminalHint") or "")
    confirmed_loss = bool(
        adventure_state.get("isGameOverScreen")
        or is_restart_screen_observation(observation)
        or screen_state in {"game_over", "game_over_restart_screen"}
        or terminal_hint == "game_over_or_loss"
    )
    confirmed_win = bool(
        adventure_state.get("isLevelComplete")
        or adventure_state.get("trophyVisible")
        or adventure_state.get("levelCompleteTrophyVisible")
        or adventure_state.get("rewardScreenVisible")
        or adventure_state.get("unlockScreenVisible")
        or screen_state in {"level_complete_trophy", "reward_unlock", "reward_screen"}
    )
    summary = _episode_summary_from_info(info)
    env_corruption = bool(
        str(info.get("done_reason") or summary.get("done_reason") or "") == "env_corruption"
        or _safe_int(info.get("env_corruption_count", summary.get("env_corruption_count", 0)), 0) > 0
    )
    if not enabled:
        return False, "extension_disabled"
    if not gameplay:
        return False, f"screen_not_gameplay:{screen_state or 'unknown'}"
    if confirmed_loss:
        return False, "confirmed_loss"
    if confirmed_win:
        return False, "confirmed_win"
    if env_corruption:
        return False, "env_corruption"
    if not final_wave_reached:
        return False, f"not_final_wave:{wave}/{max_wave}"
    return True, "final_wave_gameplay"


def _finalize_policy_attempt(
    env: PvZMaskedPPOEnv,
    log: AdventureAttemptLog,
    info: Dict[str, Any],
    *,
    forced_done_reason: str = "",
    forced_terminal_reason: str = "",
    forced_timeout_classification: str = "",
) -> AdventureAttemptLog:
    summary = _episode_summary_from_info(info)
    raw = _observation_from_info(env, info)
    done_reason = str(
        forced_done_reason
        or summary.get("done_reason")
        or info.get("done_reason")
        or classify_done_reason(raw)
    )
    terminal_reason = str(forced_terminal_reason or summary.get("terminal_reason") or info.get("terminal_reason") or "")
    adventure_state = _adventure_state_for_terminal(env, info, raw)
    log.screen_state_at_terminal = str(adventure_state.get("screenState") or raw.get("screenState") or "")
    log.terminal_trophy_visible = bool(adventure_state.get("trophyVisible"))
    log.terminal_level_complete_trophy_visible = bool(adventure_state.get("levelCompleteTrophyVisible"))
    log.terminal_post_win_click_required = bool(adventure_state.get("postWinClickRequired"))
    log.terminal_reward_screen_visible = bool(adventure_state.get("rewardScreenVisible"))
    log.terminal_unlock_screen_visible = bool(adventure_state.get("unlockScreenVisible"))
    log.terminal_new_plant_unlocked_visible = bool(adventure_state.get("newPlantUnlockedVisible"))
    log.terminal_reward_object_visible = bool(adventure_state.get("rewardObjectVisible"))
    terminal_post_win_visible = bool(
        log.terminal_trophy_visible
        or log.terminal_level_complete_trophy_visible
        or log.terminal_post_win_click_required
        or log.terminal_reward_screen_visible
        or log.terminal_unlock_screen_visible
        or log.terminal_new_plant_unlocked_visible
        or log.terminal_reward_object_visible
    )
    terminal_screen = str(adventure_state.get("screenState") or raw.get("screenState") or "").strip().lower()
    terminal_loss_visible = bool(
        terminal_screen in {"game_over", "game_over_restart", "loss", "lose_menu"}
        or adventure_state.get("gameOverVisible")
        or adventure_state.get("restartButtonVisible")
        or raw.get("gameOverVisible")
        or raw.get("restartButtonVisible")
    )
    # The bridge can leave a stale win/done hint in the episode summary while
    # the authoritative UI has already settled on the loss/restart screen. Do
    # not enter the 35-second post-win unlock loop in that state.
    if terminal_loss_visible and not terminal_post_win_visible:
        done_reason = "loss"
        terminal_reason = "game_over_restart_screen"
    if terminal_post_win_visible and done_reason not in ("post_win_pending", "win"):
        done_reason = "post_win_pending"
        if not terminal_reason or terminal_reason in {"game_over_restart_screen", "timeout", "timeout_hard_cap"}:
            terminal_reason = "level_complete_trophy" if log.terminal_trophy_visible else "reward_unlock"

    if done_reason == "timeout":
        if forced_timeout_classification:
            log.timeout_classification = forced_timeout_classification
        elif terminal_reason in {"timeout_hard_cap", "hard_cap_timeout"} or log.episode_length >= log.hard_max_steps:
            log.timeout_classification = "hard_cap_timeout"
            terminal_reason = "timeout_hard_cap"
        elif log.soft_timeout_reached and not log.soft_timeout_extended:
            log.timeout_classification = "soft_cap_timeout_no_extension"
            terminal_reason = "timeout_soft_cap_no_extension"
        else:
            log.timeout_classification = "hard_cap_timeout"
            terminal_reason = "timeout_hard_cap"
    elif log.soft_timeout_extended and done_reason in {"win", "post_win_pending"}:
        log.timeout_classification = "soft_extended_then_win"
    elif log.soft_timeout_extended and done_reason == "loss":
        log.timeout_classification = "soft_extended_then_loss"
    elif log.soft_timeout_extended:
        log.timeout_classification = "none"
    elif not log.soft_timeout_reached:
        log.timeout_classification = "none"

    if done_reason == "post_win_pending":
        log.result = "post_win_pending"
    elif done_reason == "env_corruption":
        log.result = "env_corruption"
        log.blocked_reason = terminal_reason or "env_corruption"
    else:
        log.result = "win" if done_reason == "win" else "loss" if done_reason == "loss" else "timeout"
    if log.result == "timeout":
        log.blocked_reason = (
            "timeout_hard_cap"
            if log.timeout_classification == "hard_cap_timeout"
            else "timeout_soft_cap_no_extension"
            if log.timeout_classification == "soft_cap_timeout_no_extension"
            else terminal_reason
            or "timeout"
        )
    log.done_reason = done_reason
    log.terminal_reason = terminal_reason
    log.final_wave = int(summary.get("final_wave", raw.get("wave", 0)) or 0)
    log.max_wave = int(summary.get("max_wave", raw.get("maxWave", 0)) or 0)
    log.zombies_killed = int(summary.get("zombies_killed", 0) or 0)
    log.plants_placed = int(summary.get("plants_placed", 0) or 0)
    log.mowers_lost = int(summary.get("mowers_lost", 0) or 0)
    log.illegal_actions = int(summary.get("illegal_actions", 0) or 0)
    log.bridge_errors = int(summary.get("bridge_errors", log.bridge_errors) or 0)
    log.reset_failures = 0 if bool(summary.get("reset_success", True)) else 1
    log.steps_after_soft_timeout = max(0, int(log.episode_length) - int(log.soft_timeout_step or log.episode_length))
    if not summary:
        summary = _timeout_summary(log, info, raw, terminal_reason)
    summary.update(
        {
            "result": log.result if log.result != "post_win_pending" else "post_win_pending",
            "done_reason": log.done_reason,
            "terminal_reason": log.terminal_reason,
            "episode_reward": float(log.episode_reward),
            "episode_length": int(log.episode_length),
            "final_wave": int(log.final_wave),
            "max_wave": int(log.max_wave),
            "zombies_killed": int(log.zombies_killed),
            "plants_placed": int(log.plants_placed),
            "mowers_lost": int(log.mowers_lost),
            "bridge_errors": int(log.bridge_errors),
            "illegal_actions": int(log.illegal_actions),
            "win": log.result == "win",
            "loss": log.result == "loss",
            "timeout": log.result == "timeout",
        }
    )
    log.episode_summary = _with_timeout_diagnostics(dict(summary), log)
    return log


def run_policy_attempt(
    env: PvZMaskedPPOEnv,
    model: Any,
    writer: LiveStatusWriter,
    context: Dict[str, Any],
    attempt_index: int,
    selected_seeds: List[str],
    deterministic: bool,
    tracker_level: int,
    progression_index: int,
    soft_max_steps: int,
    hard_max_steps: int,
    final_wave_extension: bool,
    seed_selection_callback: Optional[Callable[[Dict[str, Any], List[str]], Tuple[List[str], str]]] = None,
) -> AdventureAttemptLog:
    log = AdventureAttemptLog(attempt=attempt_index, selected_seeds=list(selected_seeds))
    apply_level_metadata(log, tracker_level, progression_index)
    log.soft_max_steps = int(soft_max_steps)
    log.hard_max_steps = int(hard_max_steps)
    log.final_wave_extension_enabled = bool(final_wave_extension)
    log.tactical_mask_enabled = bool(
        getattr(env.config, "tactical_masks", False)
        or getattr(env.config, "wallnut_tactical_mask", False)
        or getattr(env.config, "cherrybomb_tactical_mask", False)
    )
    log.wallnut_tactical_mask_enabled = bool(
        getattr(env.config, "tactical_masks", False)
        or getattr(env.config, "wallnut_tactical_mask", False)
    )
    log.cherrybomb_tactical_mask_enabled = bool(
        getattr(env.config, "tactical_masks", False)
        or getattr(env.config, "cherrybomb_tactical_mask", False)
    )
    update_timeout_context(context, log)
    observation, reset_info, blocked_reason = prepare_adventure_gameplay(
        env,
        writer,
        context,
        selected_seeds,
        timeout=env.config.gameplay_ready_timeout,
        seed_selection_callback=seed_selection_callback,
        expected_level=tracker_level,
    )
    if observation is None:
        log.result = "blocked"
        log.reset_failures = 1
        log.blocked_reason = blocked_reason
        update_timeout_context(context, log)
        return log

    log.selected_seeds = list(context.get("selected_seeds", selected_seeds) or selected_seeds)

    obs, _ = env.start_episode_from_observation(observation, reset_info)
    context["current_attempt"] = attempt_index
    start_state = env.base.adventure_screen_state()
    log.available_seeds = list(start_state.get("availableSeedNames", []) or [])
    log.unlocked_seeds = list(start_state.get("unlockedSeedNames", []) or [])
    last_info: Dict[str, Any] = {}
    while True:
        try:
            masks = env.action_masks()
            action, _ = model.predict(obs, deterministic=deterministic, action_masks=masks)
            action_id = int(action)
            obs, reward, terminated, truncated, info = env.step(action_id)
            last_info = info
        except Exception as exc:
            log.result = "bridge_error"
            log.done_reason = "bridge_error"
            log.bridge_errors += 1
            log.blocked_reason = str(exc)
            context["last_error"] = str(exc)
            update_timeout_context(context, log)
            writer.write(build_live_status(env, context, last_info=last_info))
            return log

        log.episode_reward += float(reward)
        log.episode_length += 1
        context["last_action_id"] = action_id
        context["last_bridge_action"] = info.get("bridge_action", action_id) if isinstance(info, dict) else action_id
        context["last_reward"] = float(reward)
        if log.soft_timeout_reached:
            log.steps_after_soft_timeout = max(0, log.episode_length - log.soft_timeout_step)
        update_timeout_context(context, log)
        raw_observation = info.get("raw_observation") if isinstance(info, dict) else None
        if hasattr(writer, "write_lazy"):
            writer.write_lazy(
                lambda: build_live_status(env, context, last_info=info),
                significant_state=live_status_significant_state(context, raw_observation, info),
                force=bool(terminated or truncated),
            )
        else:  # Compatibility for lightweight test/report writers.
            writer.write(build_live_status(env, context, last_info=info))

        if terminated or truncated:
            _finalize_policy_attempt(env, log, info)
            update_timeout_context(context, log)
            return log

        if soft_max_steps > 0 and not log.soft_timeout_reached and log.episode_length >= soft_max_steps:
            raw = _observation_from_info(env, info)
            adventure_state = _adventure_state_for_terminal(env, info, raw)
            should_extend, extension_reason = _soft_extension_decision(
                raw,
                adventure_state,
                info,
                enabled=bool(final_wave_extension),
            )
            log.soft_timeout_reached = True
            log.soft_timeout_step = int(log.episode_length)
            log.soft_timeout_extended = bool(should_extend)
            log.steps_after_soft_timeout = 0
            log.timeout_classification = "timeout_soft_extended" if should_extend else "soft_cap_timeout_no_extension"
            context["soft_timeout_extension_reason"] = extension_reason
            update_timeout_context(context, log)
            print(
                "[adventure] soft cap reached "
                f"step={log.episode_length} "
                f"wave={raw.get('wave')}/{raw.get('maxWave')} "
                f"screenState={adventure_state.get('screenState') or raw.get('screenState')} "
                f"extension={should_extend} hard_max_steps={hard_max_steps} "
                f"reason={extension_reason}"
            )
            writer.write(build_live_status(env, context, adventure_state=adventure_state, last_info=info))
            if should_extend:
                continue
            synthetic_info = dict(info)
            synthetic_summary = _timeout_summary(log, synthetic_info, raw, "timeout_soft_cap_no_extension")
            synthetic_info["episode_summary"] = _with_timeout_diagnostics(synthetic_summary, log)
            synthetic_info["done_reason"] = "timeout"
            synthetic_info["terminal_reason"] = "timeout_soft_cap_no_extension"
            synthetic_info["raw_observation"] = raw
            synthetic_info["adventure_state"] = adventure_state
            _finalize_policy_attempt(
                env,
                log,
                synthetic_info,
                forced_done_reason="timeout",
                forced_terminal_reason="timeout_soft_cap_no_extension",
                forced_timeout_classification="soft_cap_timeout_no_extension",
            )
            update_timeout_context(context, log)
            return log


def build_rows_payload(observation: Dict[str, Any], diagnostics: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    peashooters = diagnostics.get("peashooters_by_row") or diagnostics.get("current_peashooters_by_row") or {}
    threat_steps = diagnostics.get("threat_steps_by_row") or {}
    undefended = diagnostics.get("undefended_threat_steps_by_row") or {}
    lanes = observation.get("lanes", []) if isinstance(observation, dict) else []
    lane_by_row = {str(int(lane.get("row", -1))): lane for lane in lanes if isinstance(lane, dict)}
    row_count = int(observation.get("rowCount", 5) or 5) if isinstance(observation, dict) else 5
    for row in range(row_count):
        key = str(row)
        lane = lane_by_row.get(key, {})
        rows[key] = {
            "peashooters": int(peashooters.get(key, 0) or peashooters.get(row, 0) or 0),
            "threatened": int(lane.get("zombieCount", 0) or 0) > 0,
            "undefended_threat": int(undefended.get(key, 0) or undefended.get(row, 0) or 0) > 0,
            "threat_steps": int(threat_steps.get(key, 0) or threat_steps.get(row, 0) or 0),
        }
    return rows


def build_agent_payload(
    env: PvZMaskedPPOEnv,
    context: Dict[str, Any],
    last_info: Optional[Dict[str, Any]] = None,
    *,
    legal_action_count: Optional[int] = None,
) -> Dict[str, Any]:
    observation = env._last_observation or {}
    action_id = int(context.get("last_action_id", 0) or 0)
    if hasattr(env, "decode_policy_action"):
        decoded = env.decode_policy_action(action_id, observation)
    else:
        decoded = decode_action(action_id, observation, env.config.plant_types)
    plant_type = int(decoded.get("plant_type", -1))
    if legal_action_count is None:
        try:
            legal_count = int(env.action_masks().sum())
        except Exception:
            legal = observation.get("legalActions", []) if isinstance(observation, dict) else []
            legal_count = len(legal) if isinstance(legal, list) else 0
    else:
        legal_count = int(legal_action_count)
    action_type = "wait" if int(decoded.get("kind", 0)) == 0 else "plant" if int(decoded.get("kind", 0)) == 1 else "invalid"
    return {
        "last_action": action_id,
        "bridge_action": context.get("last_bridge_action", action_id),
        "decoded_action": {
            "type": action_type,
            "plant": plant_type_name(plant_type) if plant_type >= 0 else None,
            "seed_slot": int(decoded.get("slot_index", -1)),
            "row": int(decoded.get("row", -1)),
            "col": int(decoded.get("column", -1)),
        },
        "legal_action_count": legal_count,
        "action_space_mode": (
            getattr(env, "action_spec", None).mode
            if getattr(env, "action_spec", None) is not None
            else ADVENTURE_GENERALIST_ACTION_SPACE_MODE
        ),
        "action_decoder_version": getattr(env, "action_spec", None).action_decoder_version if getattr(env, "action_spec", None) is not None else "",
        "observation_version": getattr(env, "action_spec", None).observation_version if getattr(env, "action_spec", None) is not None else "",
    }


def build_live_status(
    env: PvZMaskedPPOEnv,
    context: Dict[str, Any],
    adventure_state: Optional[Dict[str, Any]] = None,
    last_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    observation = env._last_observation or {}
    adventure_state = adventure_state or {}
    last_info = last_info or {}
    unlock_snapshot = _snapshot_unlock_state(adventure_state, source="live_status")
    available_seeds = _ordered_unique_seed_names(
        list(adventure_state.get("availableSeedNames", []) or [])
        + list(adventure_state.get("visibleSeedCardNames", []) or [])
    )
    reward_totals = getattr(env, "_episode_reward_totals", {})
    reward_diagnostics = (
        dict(last_info.get("reward_diagnostics", {}))
        if isinstance(last_info, dict)
        and isinstance(last_info.get("reward_diagnostics"), dict)
        else {}
    )
    lane_diagnostics = last_info.get("lane_diagnostics", {}) if isinstance(last_info, dict) else {}
    fusion_diagnostics = dict(
        last_info.get("fusion_diagnostics", {})
        if isinstance(last_info, dict) and isinstance(last_info.get("fusion_diagnostics"), dict)
        else getattr(getattr(env, "base", None), "_last_fusion_diagnostics", {}) or {}
    )
    mask_diagnostics = last_info.get("mask_diagnostics", {}) if isinstance(last_info, dict) else {}
    if not isinstance(mask_diagnostics, dict):
        mask_diagnostics = {}
    if not mask_diagnostics and observation:
        try:
            mask_diagnostics = dict(env.base.mask_diagnostics(observation))
        except Exception:
            mask_diagnostics = {}
    for key, value in mask_diagnostics.items():
        if str(key).startswith("fusion_"):
            fusion_diagnostics[str(key)] = value
    fusion_fields = fusion_live_fields(
        fusion_diagnostics,
        str(getattr(getattr(env, "config", None), "fusion_policy", FUSION_POLICY_NONE)),
    )
    fusion_fields["fusion_action_mask_enabled"] = bool(
        getattr(getattr(env, "config", None), "fusion_action_mask_enabled", False)
    )
    summary = context.get("eval_summary", {})
    if hasattr(env, "_coach_live_status"):
        try:
            coach_fields = dict(env._coach_live_status())  # type: ignore[attr-defined]
        except Exception:
            coach_fields = human_coach_live_status_from_hook(
                getattr(env, "human_coach_hook", None),
                enabled=bool(getattr(getattr(env, "config", None), "human_coach_enabled", False)),
                platform=str(getattr(getattr(env, "config", None), "human_coach_platform", "mock") or "mock"),
            )
    else:
        coach_fields = human_coach_live_status_from_hook(
            getattr(env, "human_coach_hook", None),
            enabled=bool(getattr(getattr(env, "config", None), "human_coach_enabled", False)),
            platform=str(getattr(getattr(env, "config", None), "human_coach_platform", "mock") or "mock"),
        )
    try:
        legal_action_count = int(env.action_masks().sum())
    except Exception:
        legal_action_count = 0
    mask_block_reason_counts = {}
    if isinstance(lane_diagnostics, dict):
        raw_counts = lane_diagnostics.get("python_mask_block_reason_counts", {})
        if isinstance(raw_counts, dict):
            mask_block_reason_counts = {str(key): int(value or 0) for key, value in raw_counts.items()}
    max_seed_slots = int(getattr(getattr(env, "action_spec", None), "max_seed_slots", len(env.config.plant_types)) or len(env.config.plant_types))
    seed_inventory = inventory_from_runtime_sources(
        observation=observation,
        adventure_state=adventure_state,
        context=context,
        max_seed_slots=max_seed_slots,
        legal_action_count=legal_action_count,
        mask_block_reason_counts=mask_block_reason_counts,
    )
    compatibility = {
        "stage": context.get("current_stage", ""),
        "model_family": context.get("current_model_family", context.get("model_family", "")),
        "model_path": context.get("current_model_path", ""),
        "action_space_mode": getattr(
            getattr(env, "action_spec", None),
            "mode",
            ADVENTURE_GENERALIST_ACTION_SPACE_MODE,
        ),
        "action_count": int(getattr(env, "action_count", 0) or 0),
        "action_decoder_version": getattr(getattr(env, "action_spec", None), "action_decoder_version", ""),
        "observation_version": getattr(getattr(env, "action_spec", None), "observation_version", ""),
        "metadata_path": context.get("metadata_path", ""),
        "metadata_inferred": bool(context.get("metadata_inferred", False)),
        "compatible": not bool(context.get("blocked_reason", "")),
        "blocked_reason": context.get("blocked_reason", ""),
        "legal_action_count": legal_action_count,
        "mask_block_reason_counts": mask_block_reason_counts,
    }
    model_compatibility = context.get("model_compatibility", {})
    if not isinstance(model_compatibility, dict) or not model_compatibility:
        model_compatibility = {
            "compatible": True,
            "blocked_reason": None,
            "model_family": compatibility["model_family"],
            "model_seed_list": list(context.get("selected_seeds", [])),
            "env_seed_list": list(env.config.seed_list),
            "model_action_count": compatibility["action_count"],
            "env_action_count": compatibility["action_count"],
            "action_decoder_version": compatibility["action_decoder_version"],
            "observation_version": compatibility["observation_version"],
            "metadata_path": compatibility["metadata_path"],
            "metadata_inferred": compatibility["metadata_inferred"],
            "warnings": [],
        }
    compatibility.update(
        {
            "compatible": bool(model_compatibility.get("compatible", compatibility["compatible"])),
            "blocked_reason": model_compatibility.get("blocked_reason") or "",
            "model_seed_list": model_compatibility.get("model_seed_list", []),
            "env_seed_list": model_compatibility.get("env_seed_list", []),
            "model_action_count": model_compatibility.get("model_action_count"),
            "env_action_count": model_compatibility.get("env_action_count", compatibility["action_count"]),
            "model_compatibility_blocked_reason": model_compatibility.get("blocked_reason"),
        }
    )
    run_mode = str(context.get("run_mode") or context.get("mode") or ADVENTURE_GENERALIST_EVAL_RUN_MODE)
    selected_loadout = list(context.get("selected_seeds", getattr(env.config, "seed_list", [])))
    selected_loadout_count = int(context.get("selected_loadout_count", len(selected_loadout)) or len(selected_loadout))
    selected_loadout_count = max(0, min(max_seed_slots, selected_loadout_count))
    configured_seed_list = list(
        context.get(
            "configured_seed_list",
            context.get("initial_loadout", getattr(env.config, "seed_list", [])),
        )
    )
    observed_seed_bank_capacity = int(
        context.get("observed_seed_bank_capacity")
        or context.get("active_seed_slot_capacity")
        or context.get("current_seed_bank_capacity")
        or selected_loadout_count
        or len(getattr(env.config, "seed_list", []) or [])
        or 0
    )
    observed_seed_bank_capacity = max(0, min(max_seed_slots, observed_seed_bank_capacity))
    bridge_reported_capacity = context.get("bridge_reported_capacity", None)
    inferred_capacity_from_unlocks = int(
        context.get("inferred_capacity_from_unlocks")
        or observed_seed_bank_capacity
        or 0
    )
    inferred_capacity_from_unlocks = max(0, min(max_seed_slots, inferred_capacity_from_unlocks))
    effective_seed_capacity = int(
        context.get("effective_seed_capacity")
        or observed_seed_bank_capacity
        or selected_loadout_count
        or 0
    )
    effective_seed_capacity = max(0, min(max_seed_slots, effective_seed_capacity))
    max_effective_seed_capacity_seen = int(
        context.get("max_effective_seed_capacity_seen")
        or effective_seed_capacity
        or 0
    )
    max_effective_seed_capacity_seen = max(0, min(max_seed_slots, max_effective_seed_capacity_seen))
    inferred_capacity_source = str(context.get("inferred_capacity_source", "") or "")
    capacity_inference_reason = str(context.get("capacity_inference_reason", "") or "")
    available_priority_seeds = list(context.get("available_priority_seeds", []) or [])
    rejected_priority_seeds = list(context.get("rejected_priority_seeds", []) or [])
    active_seed_slot_count = int(selected_loadout_count)
    inactive_seed_slot_count = max(0, max_seed_slots - active_seed_slot_count)
    inactive_model_slots = max(0, max_seed_slots - selected_loadout_count)
    eligible_seeds = list(context.get("eligible_seeds", []))
    selectable_seeds = list(context.get("selectable_seeds", available_seeds))
    cleared_levels = list(context.get("cleared_levels", []))
    startup_validation = context.get("startup_validation", {})
    if not isinstance(startup_validation, dict):
        startup_validation = {}
    level_identity = context.get("level_identity", startup_validation.get("level_identity", {}))
    if not isinstance(level_identity, dict):
        level_identity = {}
    payload = {
        "mode": run_mode,
        "run_mode": run_mode,
        "reward_policy_version": REWARD_POLICY_VERSION,
        "status": context.get("status", "running"),
        "updated_at": time.time(),
        "stop_reason": context.get("stop_reason", ""),
        "blocked_reason": context.get("blocked_reason", ""),
        "model_family": context.get("current_model_family", context.get("model_family", "")),
        "action_count": compatibility["action_count"],
        "max_seed_slots": max_seed_slots,
        "observed_capacity": observed_seed_bank_capacity,
        "observed_seed_bank_capacity": observed_seed_bank_capacity,
        "active_seed_slot_capacity": observed_seed_bank_capacity,
        "current_seed_bank_capacity": observed_seed_bank_capacity,
        "bridge_reported_capacity": bridge_reported_capacity,
        "inferred_capacity_from_unlocks": inferred_capacity_from_unlocks,
        "effective_seed_capacity": effective_seed_capacity,
        "max_effective_seed_capacity_seen": max_effective_seed_capacity_seen,
        "inferred_capacity_source": inferred_capacity_source,
        "capacity_inference_reason": capacity_inference_reason,
        "available_priority_seeds": available_priority_seeds,
        "rejected_priority_seeds": rejected_priority_seeds,
        "active_seed_slot_count": active_seed_slot_count,
        "inactive_seed_slot_count": inactive_seed_slot_count,
        "selected_loadout_count": selected_loadout_count,
        "inactive_model_slots": inactive_model_slots,
        "configured_seed_list": configured_seed_list,
        "seed_order_source": context.get("seed_order_source", ""),
        "seed_order_preserved": bool(context.get("seed_order_preserved", True)),
        "seed_order_blocked_reason": context.get("seed_order_blocked_reason", ""),
        "randomize_seed_order": bool(context.get("randomize_seed_order", False)),
        "episode_sample_source": context.get("episode_sample_source", ""),
        "requested_episode_sample_source": context.get("requested_episode_sample_source", ""),
        "level_replay_supported": bool(context.get("level_replay_supported", False)),
        "level_replay_blocked_reason": context.get("level_replay_blocked_reason", ""),
        "frontier_win_streak": int(context.get("frontier_win_streak", 0) or 0),
        "frontier_win_streak_required": int(context.get("frontier_win_streak_required", 1) or 1),
        "wins_on_current_level": int(
            context.get("wins_on_current_level", context.get("frontier_win_streak", 0)) or 0
        ),
        "wins_before_advance": int(
            context.get("wins_before_advance", context.get("frontier_win_streak_required", 1)) or 1
        ),
        "frontier_mastery_ready": bool(context.get("frontier_mastery_ready", False)),
        "frontier_promoted_this_episode": bool(context.get("frontier_promoted_this_episode", False)),
        "frontier_mastery_reset_reason": context.get("frontier_mastery_reset_reason", ""),
        "mastery_sample_source": context.get("mastery_sample_source", context.get("episode_sample_source", "")),
        "frontier_replay_supported": bool(
            context.get("frontier_replay_supported", context.get("level_replay_supported", False))
        ),
        "frontier_replay_blocked_reason": context.get("frontier_replay_blocked_reason", ""),
        "frontier_replay_seed_selection_attempts": int(context.get("frontier_replay_seed_selection_attempts", 0) or 0),
        "frontier_replay_last_seed_selection_ok": context.get("frontier_replay_last_seed_selection_ok", None),
        "frontier_replay_last_seed_selection_message": context.get("frontier_replay_last_seed_selection_message", ""),
        "frontier_replay_last_seed_selection_actions": context.get("frontier_replay_last_seed_selection_actions", []),
        "frontier_replay_last_seed_selection_start_log": context.get("frontier_replay_last_seed_selection_start_log", {}),
        "frontier_mastered_levels": list(context.get("frontier_mastered_levels", [])),
        "post_win_decision": context.get("post_win_decision", ""),
        "post_win_transition_allowed": bool(context.get("post_win_transition_allowed", False)),
        "expected_transition_target": context.get("expected_transition_target", ""),
        "seed_selection_expected": bool(context.get("seed_selection_expected", False)),
        "reset_phase": context.get("reset_phase", context.get("last_reset_phase", "")),
        "startup_validation": startup_validation,
        "startup_validation_ok": startup_validation.get("ok", context.get("startup_validation_ok", None)),
        "startup_validation_reason": startup_validation.get("reason", context.get("startup_validation_reason", "")),
        "level_identity": level_identity,
        "wrapper_expected_level": level_identity.get("wrapper_expected_level", context.get("wrapper_expected_level", None)),
        "bridge_detected_level": level_identity.get("bridge_detected_level", context.get("bridge_detected_level", None)),
        "profile_adventure_level": level_identity.get("profile_adventure_level", context.get("profile_adventure_level", None)),
        "profile_adventure_level_source": level_identity.get(
            "profile_adventure_level_source",
            context.get("profile_adventure_level_source", ""),
        ),
        "ui_world_level_text": level_identity.get("ui_world_level_text", context.get("ui_world_level_text", "")),
        "screenState": level_identity.get("screenState", adventure_state.get("screenState", observation.get("screenState", ""))),
        "seedSelectionDetected": level_identity.get(
            "seedSelectionDetected",
            bool(adventure_state.get("isSeedSelectionScreen") or adventure_state.get("seedSelectionActive")),
        ),
        "gameplayReadyDetected": level_identity.get(
            "gameplayReadyDetected",
            bool(adventure_state.get("isGameplayReady") or adventure_state.get("gameplayReady")),
        ),
        "level_identity_reliable": level_identity.get(
            "level_identity_reliable",
            context.get("level_identity_reliable", None),
        ),
        "adventure_phase": context.get("state", adventure_state.get("screenState", "unknown")),
        "latest_terminal_result": context.get("latest_terminal_result", context.get("last_result", "")),
        "frontier_level": context.get("frontier_level", context.get("current_level", "")),
        "cleared_levels": cleared_levels,
        "eligible_seeds": eligible_seeds,
        "selectable_seeds": selectable_seeds,
        "selected_loadout": selected_loadout,
        "loadout_provenance": list(context.get("loadout_provenance", [])),
        "per_seed_diagnostics": dict(context.get("per_seed_diagnostics", {})),
        "curriculum_state_path": context.get("curriculum_state_path", ""),
        "curriculum_restore_status": context.get("curriculum_restore_status", ""),
        "curriculum_restore_blocked_reason": context.get("curriculum_restore_blocked_reason", ""),
        "loadout_reason": context.get("loadout_reason", ""),
        "excluded_new_plants": context.get("excluded_new_plants", []),
        "current_selected_seed_loadout": selected_loadout,
        "slot_identities": selected_loadout,
        "newly_unlocked": context.get("newly_unlocked", []),
        "new_slot_unlock_event": context.get("new_slot_unlock_event", ""),
        "current_level": int(context.get("current_level", adventure_state.get("currentAdventureLevel", 0)) or 0),
        "progression_index": int(context.get("progression_index", 0) or 0),
        "adventure_world": int(context.get("adventure_world", adventure_state.get("currentWorldOrStage", 0)) or 0),
        "adventure_stage": int(context.get("adventure_stage", adventure_state.get("currentDayLevel", 0)) or 0),
        "adventure_level_label": context.get("adventure_level_label", ""),
        "max_levels": int(context.get("max_adventure_levels", 0) or 0),
        "current_attempt": int(context.get("current_attempt", 0) or 0),
        "max_attempts": int(context.get("max_attempts_per_level", 0) or 0),
        "wins_current": int(context.get("wins_this_level", context.get("wins", 0)) or 0),
        "total_wins": int(context.get("total_wins", 0) or 0),
        "losses": int(context.get("total_losses", context.get("losses", 0)) or 0),
        "terminal_reason": context.get("terminal_reason", ""),
        "last_reset_reason": context.get("last_reset_reason", ""),
        "soft_max_steps": int(context.get("soft_max_steps", 0) or 0),
        "hard_max_steps": int(context.get("hard_max_steps", 0) or 0),
        "final_wave_extension_enabled": bool(context.get("final_wave_extension_enabled", False)),
        "soft_timeout_reached": bool(context.get("soft_timeout_reached", False)),
        "soft_timeout_extended": bool(context.get("soft_timeout_extended", False)),
        "soft_timeout_step": int(context.get("soft_timeout_step", 0) or 0),
        "steps_after_soft_timeout": int(context.get("steps_after_soft_timeout", 0) or 0),
        "timeout_classification": context.get("timeout_classification", "none"),
        "soft_timeout_extension_reason": context.get("soft_timeout_extension_reason", ""),
        "tactical_mask_enabled": bool(
            getattr(env.config, "tactical_masks", False)
            or getattr(env.config, "wallnut_tactical_mask", False)
            or getattr(env.config, "cherrybomb_tactical_mask", False)
        ),
        "wallnut_tactical_mask_enabled": bool(
            getattr(env.config, "tactical_masks", False)
            or getattr(env.config, "wallnut_tactical_mask", False)
        ),
        "cherrybomb_tactical_mask_enabled": bool(
            getattr(env.config, "tactical_masks", False)
            or getattr(env.config, "cherrybomb_tactical_mask", False)
        ),
        "total_legal_action_count": int(
            mask_diagnostics.get("total_legal_action_count", legal_action_count) or 0
        ),
        "wait_legal": bool(mask_diagnostics.get("wait_legal", legal_action_count > 0)),
        "active_seed_slot_mask_diagnostics": list(
            mask_diagnostics.get("active_seed_slots", []) or []
        ),
        "mask_diagnostics": dict(mask_diagnostics),
        "seed_list": selected_loadout,
        "model_path": context.get("current_model_path", ""),
        "active_run": context.get("active_run", ""),
        "post_win_active": bool(context.get("post_win_active", False)),
        "post_win_elapsed": float(context.get("post_win_elapsed", 0.0) or 0.0),
        "post_win_click_attempts": int(context.get("post_win_click_attempts", 0) or 0),
        "post_win_last_state": context.get("post_win_last_state", {}),
        "post_win_last_click_target": context.get("post_win_last_click_target", ""),
        "post_win_last_click_ok": context.get("post_win_last_click_ok", None),
        "post_win_blocked_reason": context.get("post_win_blocked_reason", ""),
        "post_win_transition": context.get("post_win_transition", {}),
        "adventure": {
            "state": context.get("state", adventure_state.get("screenState", "unknown")),
            "level": int(context.get("current_level", adventure_state.get("currentAdventureLevel", 0)) or 0),
            "current_level": int(context.get("current_level", adventure_state.get("currentAdventureLevel", 0)) or 0),
            "progression_index": int(context.get("progression_index", 0) or 0),
            "world": int(context.get("adventure_world", adventure_state.get("currentWorldOrStage", 0)) or 0),
            "stage": int(context.get("adventure_stage", adventure_state.get("currentDayLevel", 0)) or 0),
            "level_label": context.get("adventure_level_label", ""),
            "adventure_world": int(context.get("adventure_world", adventure_state.get("currentWorldOrStage", 0)) or 0),
            "adventure_stage": int(context.get("adventure_stage", adventure_state.get("currentDayLevel", 0)) or 0),
            "adventure_level_label": context.get("adventure_level_label", ""),
            "max_adventure_levels": int(context.get("max_adventure_levels", 0) or 0),
            "attempt": int(context.get("current_attempt", 0) or 0),
            "current_attempt": int(context.get("current_attempt", 0) or 0),
            "max_attempts_per_level": int(context.get("max_attempts_per_level", 0) or 0),
            "consecutive_wins": int(context.get("consecutive_wins", 0) or 0),
            "wins_this_level": int(context.get("wins_this_level", 0) or 0),
            "wins": int(context.get("wins", context.get("wins_this_level", 0)) or 0),
            "total_wins": int(context.get("total_wins", 0) or 0),
            "losses": int(context.get("losses", 0) or 0),
            "total_losses": int(context.get("total_losses", context.get("losses", 0)) or 0),
            "advanced": bool(context.get("advanced", False)),
            "advance_on_wins": int(context.get("advance_on_wins", 1) or 1),
            "required_consecutive_wins_remaining": int(
                context.get(
                    "required_consecutive_wins_remaining",
                    required_consecutive_wins_remaining(
                        int(context.get("consecutive_wins", 0) or 0),
                        int(context.get("advance_on_wins", 1) or 1),
                    ),
                )
                or 0
            ),
            "last_result": context.get("last_result", ""),
            "blocked_reason": context.get("blocked_reason", ""),
            "win_detected": bool(context.get("win_detected", False)),
            "trophy_visible": bool(context.get("trophy_visible", False) or adventure_state.get("trophyVisible")),
            "trophy_clicked": bool(context.get("trophy_clicked", False)),
            "reward_screen_seen": bool(context.get("reward_screen_seen", False)),
            "unlock_screen_seen": bool(context.get("unlock_screen_seen", False)),
            "post_win_transition_completed": bool(context.get("post_win_transition_completed", False)),
            "post_win_blocked_reason": context.get("post_win_blocked_reason", ""),
            "post_win_active": bool(context.get("post_win_active", False)),
            "post_win_elapsed": float(context.get("post_win_elapsed", 0.0) or 0.0),
            "post_win_click_attempts": int(context.get("post_win_click_attempts", 0) or 0),
            "post_win_last_state": context.get("post_win_last_state", {}),
            "post_win_last_click_target": context.get("post_win_last_click_target", ""),
            "post_win_last_click_ok": context.get("post_win_last_click_ok", None),
            "trophy_click_count": int(context.get("trophy_click_count", 0) or 0),
            "reward_continue_click_count": int(context.get("reward_continue_click_count", 0) or 0),
            "terminal_reason": context.get("terminal_reason", ""),
            "last_reset_reason": context.get("last_reset_reason", ""),
            "soft_max_steps": int(context.get("soft_max_steps", 0) or 0),
            "hard_max_steps": int(context.get("hard_max_steps", 0) or 0),
            "final_wave_extension_enabled": bool(context.get("final_wave_extension_enabled", False)),
            "soft_timeout_reached": bool(context.get("soft_timeout_reached", False)),
            "soft_timeout_extended": bool(context.get("soft_timeout_extended", False)),
            "soft_timeout_step": int(context.get("soft_timeout_step", 0) or 0),
            "steps_after_soft_timeout": int(context.get("steps_after_soft_timeout", 0) or 0),
            "timeout_classification": context.get("timeout_classification", "none"),
            "soft_timeout_extension_reason": context.get("soft_timeout_extension_reason", ""),
            "unlocked_seeds": sorted(context.get("unlocked_seeds", [])),
            "configured_seed_list": configured_seed_list,
            "selected_seeds": selected_loadout,
            "selected_loadout": selected_loadout,
            "selected_loadout_count": selected_loadout_count,
            "seed_order_source": context.get("seed_order_source", ""),
            "seed_order_preserved": bool(context.get("seed_order_preserved", True)),
            "seed_order_blocked_reason": context.get("seed_order_blocked_reason", ""),
            "randomize_seed_order": bool(context.get("randomize_seed_order", False)),
            "eligible_seeds": eligible_seeds,
            "selectable_seeds": selectable_seeds,
            "loadout_reason": context.get("loadout_reason", ""),
            "excluded_new_plants": context.get("excluded_new_plants", []),
            "active_seed_slot_count": active_seed_slot_count,
            "inactive_seed_slot_count": inactive_seed_slot_count,
            "max_seed_slots": max_seed_slots,
            "observed_capacity": observed_seed_bank_capacity,
            "observed_seed_bank_capacity": observed_seed_bank_capacity,
            "active_seed_slot_capacity": observed_seed_bank_capacity,
            "current_seed_bank_capacity": observed_seed_bank_capacity,
            "bridge_reported_capacity": bridge_reported_capacity,
            "inferred_capacity_from_unlocks": inferred_capacity_from_unlocks,
            "effective_seed_capacity": effective_seed_capacity,
            "max_effective_seed_capacity_seen": max_effective_seed_capacity_seen,
            "inferred_capacity_source": inferred_capacity_source,
            "capacity_inference_reason": capacity_inference_reason,
            "available_priority_seeds": available_priority_seeds,
            "rejected_priority_seeds": rejected_priority_seeds,
            "inactive_model_slots": inactive_model_slots,
            "available_seeds": available_seeds,
            "latest_unlock": context.get("latest_unlock", ""),
            "newly_unlocked": context.get("newly_unlocked", []),
            "current_stage": context.get("current_stage", ""),
            "current_model_family": context.get("current_model_family", ""),
            "current_model_path": context.get("current_model_path", ""),
            "frontier_level": context.get("frontier_level", context.get("current_level", "")),
            "cleared_levels": cleared_levels,
            "episode_sample_source": context.get("episode_sample_source", ""),
            "requested_episode_sample_source": context.get("requested_episode_sample_source", ""),
            "level_replay_supported": bool(context.get("level_replay_supported", False)),
            "level_replay_blocked_reason": context.get("level_replay_blocked_reason", ""),
            "frontier_win_streak": int(context.get("frontier_win_streak", 0) or 0),
            "frontier_win_streak_required": int(context.get("frontier_win_streak_required", 1) or 1),
            "wins_on_current_level": int(
                context.get("wins_on_current_level", context.get("frontier_win_streak", 0)) or 0
            ),
            "wins_before_advance": int(
                context.get("wins_before_advance", context.get("frontier_win_streak_required", 1)) or 1
            ),
            "frontier_mastery_ready": bool(context.get("frontier_mastery_ready", False)),
            "frontier_promoted_this_episode": bool(context.get("frontier_promoted_this_episode", False)),
            "frontier_mastery_reset_reason": context.get("frontier_mastery_reset_reason", ""),
            "mastery_sample_source": context.get("mastery_sample_source", context.get("episode_sample_source", "")),
            "frontier_replay_supported": bool(
                context.get("frontier_replay_supported", context.get("level_replay_supported", False))
            ),
            "frontier_replay_blocked_reason": context.get("frontier_replay_blocked_reason", ""),
            "frontier_replay_seed_selection_attempts": int(context.get("frontier_replay_seed_selection_attempts", 0) or 0),
            "frontier_replay_last_seed_selection_ok": context.get("frontier_replay_last_seed_selection_ok", None),
            "frontier_replay_last_seed_selection_message": context.get("frontier_replay_last_seed_selection_message", ""),
            "frontier_replay_last_seed_selection_actions": context.get("frontier_replay_last_seed_selection_actions", []),
            "frontier_replay_last_seed_selection_start_log": context.get("frontier_replay_last_seed_selection_start_log", {}),
            "frontier_mastered_levels": list(context.get("frontier_mastered_levels", [])),
            "post_win_decision": context.get("post_win_decision", ""),
            "post_win_transition_allowed": bool(context.get("post_win_transition_allowed", False)),
            "expected_transition_target": context.get("expected_transition_target", ""),
            "seed_selection_expected": bool(context.get("seed_selection_expected", False)),
            "reset_phase": context.get("reset_phase", context.get("last_reset_phase", "")),
            "startup_validation": startup_validation,
            "startup_validation_ok": startup_validation.get("ok", context.get("startup_validation_ok", None)),
            "startup_validation_reason": startup_validation.get("reason", context.get("startup_validation_reason", "")),
            "level_identity": level_identity,
            "wrapper_expected_level": level_identity.get("wrapper_expected_level", context.get("wrapper_expected_level", None)),
            "bridge_detected_level": level_identity.get("bridge_detected_level", context.get("bridge_detected_level", None)),
            "profile_adventure_level": level_identity.get("profile_adventure_level", context.get("profile_adventure_level", None)),
            "profile_adventure_level_source": level_identity.get(
                "profile_adventure_level_source",
                context.get("profile_adventure_level_source", ""),
            ),
            "ui_world_level_text": level_identity.get("ui_world_level_text", context.get("ui_world_level_text", "")),
            "screenState": level_identity.get("screenState", adventure_state.get("screenState", observation.get("screenState", ""))),
            "seedSelectionDetected": level_identity.get("seedSelectionDetected", False),
            "gameplayReadyDetected": level_identity.get("gameplayReadyDetected", False),
            "level_identity_reliable": level_identity.get(
                "level_identity_reliable",
                context.get("level_identity_reliable", None),
            ),
            "missing_required_unlocked": seed_inventory.get("missing_required_unlocked", []),
            "missing_required_available": seed_inventory.get("missing_required_available", []),
            "post_win_transition": context.get("post_win_transition", {}),
        },
        "gameplay": {
            "sun": int(observation.get("sun", 0) or 0),
            "wave": int(observation.get("wave", 0) or 0),
            "max_wave": int(observation.get("maxWave", 0) or 0),
            "plants": int(observation.get("plantCount", 0) or 0),
            "zombies": int(observation.get("zombieCount", 0) or 0),
            "mowers_lost": max(0, int(observation.get("rowCount", 5) or 5) - int(observation.get("logicalMowerCount", 5) or 5)),
            "gameplay_ready": bool(observation.get("gameplayReady")),
            "screen_state": observation.get("screenState") or adventure_state.get("screenState"),
        },
        "screen": {
            "startup_popup_visible": bool(adventure_state.get("startupPopupVisible") or observation.get("startupPopupVisible")),
            "startup_ok_button_visible": bool(adventure_state.get("startupOkButtonVisible") or observation.get("startupOkButtonVisible")),
            "main_menu_blocked_by_popup": bool(adventure_state.get("mainMenuBlockedByPopup") or observation.get("mainMenuBlockedByPopup")),
            "reward_screen_visible": bool(unlock_snapshot.get("rewardScreenVisible")),
            "unlock_screen_visible": bool(unlock_snapshot.get("unlockScreenVisible")),
            "new_plant_unlocked_visible": bool(unlock_snapshot.get("newPlantUnlockedVisible")),
            "trophy_visible": bool(adventure_state.get("trophyVisible")),
            "level_complete_trophy_visible": bool(adventure_state.get("levelCompleteTrophyVisible")),
            "post_win_click_required": bool(adventure_state.get("postWinClickRequired")),
        },
        "unlock": {
            "unlock_screen_visible": bool(unlock_snapshot.get("unlockScreenVisible")),
            "reward_screen_visible": bool(unlock_snapshot.get("rewardScreenVisible")),
            "new_plant_unlocked_visible": bool(unlock_snapshot.get("newPlantUnlockedVisible")),
            "new_plant_unlocked_name": unlock_snapshot.get("newPlantUnlockedName", ""),
            "new_plant_unlocked_plant_type": unlock_snapshot.get("newPlantUnlockedPlantType", -1),
            "latest_unlock": context.get("latest_unlock", "") or unlock_snapshot.get("newPlantUnlockedName", ""),
            "unlocked_seeds": sorted(context.get("unlocked_seeds", [])),
            "eligible_seeds": eligible_seeds,
            "newly_unlocked": context.get("newly_unlocked", []),
            "available_seeds": available_seeds,
            "selectable_seeds": selectable_seeds,
            "loadout_reason": context.get("loadout_reason", ""),
            "excluded_new_plants": context.get("excluded_new_plants", []),
            "unknown_visible_seed_cards": unlock_snapshot.get("unknownVisibleSeedCards", []),
            "unknown_unlock_objects": unlock_snapshot.get("unknownUnlockObjects", []),
        },
        "seed_inventory": seed_inventory,
        "model_compatibility": model_compatibility,
        "compatibility": compatibility,
        "coach": dict(coach_fields),
        "human_coach": dict(coach_fields),
        "stream_coach": dict(coach_fields),
        "fusion": dict(fusion_fields),
        "agent": build_agent_payload(
            env,
            context,
            last_info,
            legal_action_count=legal_action_count,
        ),
        "rows": build_rows_payload(observation, lane_diagnostics),
        "reward": {
            "reward_policy_version": REWARD_POLICY_VERSION,
            "episode_reward": float(getattr(env, "_episode_reward", 0.0) or 0.0),
            "terminal_reward_total": float(reward_totals.get("win_loss_reward_total", 0.0) or 0.0),
            "threat_reward_total": float(reward_totals.get("threat_delta_reward_total", 0.0) or 0.0),
            "mower_penalty_total": float(reward_totals.get("mower_loss_penalty_total", 0.0) or 0.0),
            "illegal_action_penalty_total": float(reward_totals.get("illegal_penalty_total", 0.0) or 0.0),
            "reward_component_total": sum(
                float(reward_totals.get(field, 0.0) or 0.0)
                for field in REWARD_EPISODE_TOTAL_FIELDS
            ),
            "reward_unattributed_adjustment_total": float(
                getattr(env, "_episode_reward_unattributed_adjustment_total", 0.0) or 0.0
            ),
            **reward_diagnostics,
            **{field: float(reward_totals.get(field, 0.0) or 0.0) for field in REWARD_EPISODE_TOTAL_FIELDS},
        },
        "eval": summary,
        **fusion_fields,
        **coach_fields,
    }
    reward_payload = payload["reward"]
    reward_payload["reward_components_match"] = abs(
        float(reward_payload.get("episode_reward", 0.0) or 0.0)
        - (
            float(reward_payload.get("reward_component_total", 0.0) or 0.0)
            + float(reward_payload.get("reward_unattributed_adjustment_total", 0.0) or 0.0)
        )
    ) <= 1e-6
    # Streamer V1 status is additive and only present when the source-neutral
    # controller is attached.  Generalist/evaluation status remains on the
    # existing schema when no Twitch controller is active.
    if getattr(env, "streamer_v1_controller", None) is not None:
        try:
            payload.update(env._streamer_v1_live_status())
        except Exception:
            payload.setdefault("streamer_controller_error", "status_unavailable")
    return payload


def aggregate_level_metrics(level: AdventureLevelLog) -> None:
    attempts = [log for log in level.attempt_logs if log.get("result") in {"win", "loss", "timeout"}]
    count = max(1, len(attempts))
    level.avg_reward = sum(float(log.get("episode_reward", 0.0) or 0.0) for log in attempts) / count
    level.avg_wave = sum(int(log.get("final_wave", 0) or 0) for log in attempts) / count
    level.avg_kills = sum(int(log.get("zombies_killed", 0) or 0) for log in attempts) / count
    level.avg_plants = sum(int(log.get("plants_placed", 0) or 0) for log in attempts) / count
    level.avg_mowers_lost = sum(int(log.get("mowers_lost", 0) or 0) for log in attempts) / count
    response_rates = []
    undefended_totals: Dict[str, float] = {}
    for log in attempts:
        summary = log.get("episode_summary", {}) if isinstance(log.get("episode_summary"), dict) else {}
        if "row_defense_response_rate" in summary:
            response_rates.append(float(summary.get("row_defense_response_rate") or 0.0))
        ratios = summary.get("undefended_threat_ratio_by_row", {})
        if isinstance(ratios, dict):
            for key, value in ratios.items():
                undefended_totals[str(key)] = undefended_totals.get(str(key), 0.0) + float(value or 0.0)
    level.row_defense_response_rate = sum(response_rates) / len(response_rates) if response_rates else 0.0
    level.undefended_threat_ratio_by_row = {key: value / count for key, value in sorted(undefended_totals.items())}


def summarize_progress(
    levels: List[AdventureLevelLog],
    *,
    terminal_episodes_only: bool = False,
) -> Dict[str, Any]:
    attempts = [
        attempt
        for level in levels
        for attempt in level.attempt_logs
        if not terminal_episodes_only
        or attempt.get("result") in {"win", "loss", "timeout"}
    ]
    count = max(1, len(attempts))
    timeout_classifications: Counter[str] = Counter(
        str(attempt.get("timeout_classification") or "none") for attempt in attempts
    )
    return {
        "episodes_completed": len(attempts),
        "win_rate": sum(1 for attempt in attempts if attempt.get("result") == "win") / count,
        "losses": sum(1 for attempt in attempts if attempt.get("result") == "loss"),
        "timeouts": sum(1 for attempt in attempts if attempt.get("result") == "timeout"),
        "timeout_classifications": dict(sorted(timeout_classifications.items())),
        "soft_timeout_reached_count": sum(1 for attempt in attempts if attempt.get("soft_timeout_reached")),
        "soft_timeout_extended_count": sum(1 for attempt in attempts if attempt.get("soft_timeout_extended")),
        "tactical_mask_enabled": any(bool(attempt.get("tactical_mask_enabled")) for attempt in attempts),
        "avg_reward": sum(float(attempt.get("episode_reward", 0.0) or 0.0) for attempt in attempts) / count,
        "avg_wave": sum(int(attempt.get("final_wave", 0) or 0) for attempt in attempts) / count,
        "avg_kills": sum(int(attempt.get("zombies_killed", 0) or 0) for attempt in attempts) / count,
        "avg_plants": sum(int(attempt.get("plants_placed", 0) or 0) for attempt in attempts) / count,
        "avg_mowers_lost": sum(int(attempt.get("mowers_lost", 0) or 0) for attempt in attempts) / count,
        "reset_failures": sum(int(attempt.get("reset_failures", 0) or 0) for attempt in attempts),
        "bridge_errors": sum(int(attempt.get("bridge_errors", 0) or 0) for attempt in attempts),
        "illegal_actions": sum(int(attempt.get("illegal_actions", 0) or 0) for attempt in attempts),
    }


def run_adventure_eval(
    *,
    config: Dict[str, Any],
    env_config: PvZSB3Config,
    model: Any,
    model_path: Path,
    deterministic: bool,
    advance_on_wins: int,
    max_adventure_levels: int,
    max_attempts_per_level: int,
    adventure_start_level: int,
    live_status_path: Optional[Path],
    gui: bool = False,
    adventure_soft_max_steps: Optional[int] = None,
    adventure_hard_max_steps: Optional[int] = None,
    adventure_final_wave_extension: Optional[bool] = None,
    evaluation_episode_limit: Optional[int] = None,
    strict_level_identity: bool = False,
) -> Dict[str, Any]:
    advance_on_wins = max(1, int(advance_on_wins))
    max_adventure_levels = max(1, int(max_adventure_levels))
    max_attempts_per_level = max(1, int(max_attempts_per_level))
    soft_max_steps, hard_max_steps, final_wave_extension = normalize_adventure_timeout_config(
        adventure_soft_max_steps if adventure_soft_max_steps is not None else config.get("adventure_soft_max_steps", DEFAULT_ADVENTURE_SOFT_MAX_STEPS),
        adventure_hard_max_steps if adventure_hard_max_steps is not None else config.get("adventure_hard_max_steps", DEFAULT_ADVENTURE_HARD_MAX_STEPS),
        adventure_final_wave_extension
        if adventure_final_wave_extension is not None
        else config.get("adventure_final_wave_extension", True),
    )
    config["adventure_soft_max_steps"] = int(soft_max_steps)
    config["adventure_hard_max_steps"] = int(hard_max_steps)
    config["adventure_final_wave_extension"] = bool(final_wave_extension)
    config["max_steps"] = int(hard_max_steps)
    env_config.max_steps = int(hard_max_steps)
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = LiveStatusWriter(live_status_path)
    gui_process = launch_gui(live_status_path) if gui and live_status_path is not None else None
    env = PvZMaskedPPOEnv(env_config)
    unlocked: Counter[str] = Counter({canonical_seed_name(seed): 1 for seed in BASE_UNLOCKED_SEEDS})
    levels: List[AdventureLevelLog] = []
    stop_reason = ""
    episode_limit = (
        max(1, int(evaluation_episode_limit))
        if evaluation_episode_limit is not None
        else None
    )
    episodes_completed = 0
    episode_limit_reached = False
    level_identity_initial: Dict[str, Any] = {}
    level_identity_start: Dict[str, Any] = {}
    level_identity_start_navigation_deferred = False
    level_identity_end: Dict[str, Any] = {}
    def resolve_stop_reason(reason: str) -> str:
        return adventure_stop_reason(reason)
    context: Dict[str, Any] = {
        "status": "running",
        "state": "STARTING",
        "advance_on_wins": advance_on_wins,
        "max_adventure_levels": max_adventure_levels,
        "max_attempts_per_level": max_attempts_per_level,
        "soft_max_steps": int(soft_max_steps),
        "hard_max_steps": int(hard_max_steps),
        "final_wave_extension_enabled": bool(final_wave_extension),
        "soft_timeout_reached": False,
        "soft_timeout_extended": False,
        "soft_timeout_step": 0,
        "steps_after_soft_timeout": 0,
        "timeout_classification": "none",
        "soft_timeout_extension_reason": "",
        "active_run": str(run_dir),
        "selected_seeds": list(config.get("seed_list", [])),
        "unlocked_seeds": sorted(unlocked.keys()),
        "latest_unlock": "",
        "wins": 0,
        "total_wins": 0,
        "losses": 0,
        "total_losses": 0,
        "advanced": False,
        "required_consecutive_wins_remaining": advance_on_wins,
        "terminal_reason": "",
        "last_reset_reason": "",
        "win_detected": False,
        "trophy_visible": False,
        "trophy_clicked": False,
        "reward_screen_seen": False,
        "unlock_screen_seen": False,
        "post_win_transition_completed": False,
        "post_win_blocked_reason": "",
        "post_win_active": False,
        "post_win_elapsed": 0.0,
        "post_win_click_attempts": 0,
        "post_win_last_state": {},
        "post_win_last_click_target": "",
        "post_win_last_click_ok": None,
        "trophy_click_count": 0,
        "reward_continue_click_count": 0,
        "eval_summary": {},
        "model_compatibility": config.get("model_compatibility", {}),
        "run_mode": ADVENTURE_GENERALIST_EVAL_RUN_MODE,
        "current_stage": "adventure_generalist_14slot_identity_full_v2",
        "current_model_family": str(config.get("model_family", "")),
        "current_model_path": str(model_path),
        "metadata_path": str(config.get("metadata_path", "")),
    }
    runtime_loadout_selector: Optional[RuntimeEvaluationLoadoutSelector] = None
    if bool(config.get("evaluation_runtime_seed_rotation", True)):
        runtime_loadout_selector = RuntimeEvaluationLoadoutSelector(
            configured_seed_list=list(config.get("seed_list", [])),
            max_seed_slots=int(getattr(env.config, "max_seed_slots", 14) or 14),
            context=context,
        )
        context["evaluation_seed_selection_mode"] = "runtime_rotation"
    else:
        context["evaluation_seed_selection_mode"] = "configured_static"
    print(
        "[adventure] timeout "
        f"soft_max_steps={soft_max_steps} "
        f"hard_max_steps={hard_max_steps} "
        f"final_wave_extension={final_wave_extension}"
    )
    print(
        "[adventure] tactical_masks="
        f"{bool(config.get('tactical_masks') or config.get('wallnut_tactical_mask') or config.get('cherrybomb_tactical_mask'))} "
        f"wallnut={bool(config.get('tactical_masks') or config.get('wallnut_tactical_mask'))} "
        f"cherrybomb={bool(config.get('tactical_masks') or config.get('cherrybomb_tactical_mask'))}"
    )
    try:
        env.base.configure()
        initial_state = env.base.adventure_screen_state()
        level_identity_start = adventure_level_identity_diagnostics(
            initial_state,
            int(adventure_start_level),
        )
        level_identity_initial = dict(level_identity_start)
        context["level_identity_initial"] = dict(level_identity_initial)
        context["level_identity_start"] = dict(level_identity_start)
        if strict_level_identity and not level_identity_start.get("level_identity_reliable"):
            start_reason = str(level_identity_start.get("level_identity_reason") or "")
            start_mismatches = list(level_identity_start.get("level_identity_mismatches", []) or [])
            can_navigate_to_identity = bool(
                start_reason == "navigation_or_unstable_screen"
                and not start_mismatches
                and not level_identity_start.get("challenge_mode_context")
            )
            if not can_navigate_to_identity:
                raise RuntimeError(
                    "blocked_reason=streamer_evaluation_start_level_unreliable: "
                    f"expected={adventure_start_level} "
                    f"reason={start_reason} "
                    f"mismatches={start_mismatches}"
                )

            level_identity_start_navigation_deferred = True
            context["level_identity_start_navigation_deferred"] = True
            context["state"] = "STARTUP_NAVIGATION_FOR_LEVEL_IDENTITY"
            prepared_observation, prepared_reset, prepared_reason = prepare_adventure_gameplay(
                env,
                writer,
                context,
                list(config.get("seed_list", [])),
                timeout=env.config.gameplay_ready_timeout,
                seed_selection_callback=runtime_loadout_selector,
                expected_level=int(adventure_start_level),
            )
            context["last_ui_action"] = prepared_reset.get("reset", prepared_reset)
            if prepared_observation is None:
                raise RuntimeError(
                    "blocked_reason=streamer_evaluation_start_navigation_failed: "
                    f"expected={adventure_start_level} reason={prepared_reason}"
                )

            prepared_state = env.base.adventure_screen_state()
            prepared_candidates = [prepared_observation, prepared_state]
            prepared_diagnostics = [
                adventure_level_identity_diagnostics(candidate, int(adventure_start_level))
                for candidate in prepared_candidates
                if isinstance(candidate, dict)
            ]
            level_identity_start = next(
                (
                    diagnostics
                    for diagnostics in prepared_diagnostics
                    if diagnostics.get("level_identity_reliable")
                ),
                prepared_diagnostics[-1] if prepared_diagnostics else {},
            )
            context["level_identity_start"] = dict(level_identity_start)
            if not level_identity_start.get("level_identity_reliable"):
                raise RuntimeError(
                    "blocked_reason=streamer_evaluation_start_level_unreliable_after_navigation: "
                    f"expected={adventure_start_level} "
                    f"reason={level_identity_start.get('level_identity_reason', '')} "
                    f"mismatches={level_identity_start.get('level_identity_mismatches', [])}"
                )
        for level_offset in range(max(0, max_adventure_levels)):
            tracker_level = int(adventure_start_level) + level_offset
            progression_index = level_offset + 1
            level = AdventureLevelLog(level=tracker_level, advance_on_wins=advance_on_wins)
            apply_level_metadata(level, tracker_level, progression_index)
            level.unlocked_before_level = sorted(unlocked.keys())
            context["current_level"] = tracker_level
            context.update(adventure_level_metadata(tracker_level, progression_index))
            context["wins_this_level"] = 0
            context["wins"] = 0
            context["losses"] = 0
            context["consecutive_wins"] = 0
            context["advanced"] = False
            context["required_consecutive_wins_remaining"] = advance_on_wins
            context["blocked_reason"] = ""
            context["win_detected"] = False
            context["trophy_visible"] = False
            context["trophy_clicked"] = False
            context["reward_screen_seen"] = False
            context["unlock_screen_seen"] = False
            context["post_win_transition_completed"] = False
            context["post_win_blocked_reason"] = ""
            context["post_win_active"] = False
            context["post_win_elapsed"] = 0.0
            context["post_win_click_attempts"] = 0
            context["post_win_last_state"] = {}
            context["post_win_last_click_target"] = ""
            context["post_win_last_click_ok"] = None
            context["trophy_click_count"] = 0
            context["reward_continue_click_count"] = 0
            context["terminal_reason"] = ""
            context["last_reset_reason"] = ""
            context["soft_timeout_reached"] = False
            context["soft_timeout_extended"] = False
            context["soft_timeout_step"] = 0
            context["steps_after_soft_timeout"] = 0
            context["timeout_classification"] = "none"
            context["soft_timeout_extension_reason"] = ""
            last_blocked_reason = ""

            for attempt_index in range(1, max(1, max_attempts_per_level) + 1):
                state = env.base.adventure_screen_state()
                update_unlocked_from_state(unlocked, state, source="level_start", level=tracker_level)
                available = _ordered_unique_seed_names(list(state.get("availableSeedNames", []) or []))
                selected_seeds = list(context.get("selected_seeds", []) or config.get("seed_list", []))
                level.selected_seeds = selected_seeds
                level.available_seed_names = available
                level.unknown_visible_seed_names = list(state.get("unknownVisibleSeedNames", []) or [])
                context["selected_seeds"] = selected_seeds
                context["unlocked_seeds"] = sorted(unlocked.keys())
                context["state"] = "GAMEPLAY"
                context["current_attempt"] = attempt_index

                attempt = run_policy_attempt(
                    env,
                    model,
                    writer,
                    context,
                    attempt_index=attempt_index,
                    selected_seeds=selected_seeds,
                    deterministic=deterministic,
                    tracker_level=tracker_level,
                    progression_index=progression_index,
                    soft_max_steps=soft_max_steps,
                    hard_max_steps=hard_max_steps,
                    final_wave_extension=final_wave_extension,
                    seed_selection_callback=runtime_loadout_selector,
                )
                selected_seeds = list(attempt.selected_seeds)
                level.selected_seeds = selected_seeds
                episode_counted = attempt.result in {"win", "loss", "timeout"}
                if episode_counted:
                    episodes_completed += 1
                episode_limit_reached = bool(
                    episode_limit is not None and episodes_completed >= episode_limit
                )
                context["evaluation_episode_limit"] = episode_limit
                context["evaluation_episodes_completed"] = episodes_completed
                level.attempts += 1
                context["last_result"] = attempt.result
                context["terminal_reason"] = attempt.terminal_reason or attempt.done_reason
                print(f"[adventure] attempt ended result={attempt.result}")
                print(f"[adventure] done_reason={attempt.done_reason} terminal_reason={attempt.terminal_reason}")
                print(
                    "[adventure] timeout "
                    f"classification={attempt.timeout_classification} "
                    f"soft_reached={attempt.soft_timeout_reached} "
                    f"soft_extended={attempt.soft_timeout_extended} "
                    f"steps_after_soft={attempt.steps_after_soft_timeout}"
                )
                print(
                    "[adventure] trophyVisible="
                    f"{attempt.terminal_trophy_visible} postWinClickRequired={attempt.terminal_post_win_click_required}"
                )

                post_win_handled = False
                post_win_transition: Dict[str, Any] = {}
                post_win_blocked_reason = ""
                final_state: Dict[str, Any] = {}
                unlock_screen_seen = False
                unlock_snapshot: Dict[str, Any] = {}
                available_after: List[str] = []
                unknown_unlock_objects: List[Dict[str, Any]] = []

                if attempt.result == "post_win_pending":
                    print("[adventure] entering post-win handler=True")
                    context["state"] = "POST_WIN_PENDING"
                    (
                        final_state,
                        unlock_screen_seen,
                        unlock_snapshot,
                        available_after,
                        unknown_unlock_objects,
                        post_win_blocked_reason,
                        post_win_transition,
                    ) = collect_post_win_unlocks(env, writer, context, unlocked, tracker_level)
                    attempt.trophy_visible = bool(post_win_transition.get("trophy_visible", False))
                    attempt.trophy_clicked = bool(post_win_transition.get("trophy_clicked", False))
                    attempt.reward_screen_seen = bool(post_win_transition.get("reward_screen_seen", False))
                    attempt.unlock_screen_seen = bool(post_win_transition.get("unlock_screen_seen", unlock_screen_seen))
                    attempt.post_win_transition_completed = bool(post_win_transition.get("post_win_transition_completed", False))
                    attempt.post_win_blocked_reason = str(
                        post_win_transition.get("post_win_blocked_reason", post_win_blocked_reason) or ""
                    )
                    attempt.trophy_click_count = int(post_win_transition.get("trophy_click_count", 0) or 0)
                    attempt.reward_continue_click_count = int(post_win_transition.get("reward_continue_click_count", 0) or 0)
                    if post_win_blocked_reason:
                        attempt.blocked_reason = post_win_blocked_reason
                        context["blocked_reason"] = post_win_blocked_reason
                        last_blocked_reason = post_win_blocked_reason
                        print("[adventure] post-win transition completed=False")
                        update_attempt_progress(attempt, level, context, advance_on_wins)
                        writer.write(build_live_status(env, context, adventure_state=final_state))
                        level.attempt_logs.append(asdict(attempt))
                        break
                    print("[adventure] post-win transition completed=True")
                    post_win_handled = True
                    attempt.result = "win"
                    attempt.done_reason = "win"
                    attempt.terminal_reason = attempt.terminal_reason or "level_complete_trophy"
                    if attempt.timeout_classification == "timeout_soft_extended":
                        attempt.timeout_classification = "soft_extended_then_win"
                    attempt.episode_summary.update(
                        {
                            "result": "win",
                            "done_reason": "win",
                            "win": True,
                            "loss": False,
                            "timeout": False,
                            "timeout_classification": attempt.timeout_classification,
                        }
                    )
                    context["last_result"] = attempt.result
                    if not episode_counted:
                        episodes_completed += 1
                        episode_counted = True
                        episode_limit_reached = bool(
                            episode_limit is not None and episodes_completed >= episode_limit
                        )
                        context["evaluation_episodes_completed"] = episodes_completed

                if attempt.result == "win":
                    attempt.win_detected = True
                    level.win_detected = True
                    context["win_detected"] = True
                    level.wins += 1
                    context["total_wins"] = int(context.get("total_wins", 0) or 0) + 1
                    level.consecutive_wins += 1
                    update_attempt_progress(attempt, level, context, advance_on_wins)
                    writer.write(build_live_status(env, context))
                    if level.consecutive_wins >= advance_on_wins:
                        context["state"] = "LEVEL_WIN_DETECTED"
                        if not post_win_handled:
                            print("[adventure] entering post-win handler=True")
                            (
                                final_state,
                                unlock_screen_seen,
                                unlock_snapshot,
                                available_after,
                                unknown_unlock_objects,
                                post_win_blocked_reason,
                                post_win_transition,
                            ) = collect_post_win_unlocks(env, writer, context, unlocked, tracker_level)
                            print(
                                "[adventure] post-win transition completed="
                                f"{bool(post_win_transition.get('post_win_transition_completed', False))}"
                            )
                        attempt.win_detected = True
                        attempt.trophy_visible = bool(post_win_transition.get("trophy_visible", False))
                        attempt.trophy_clicked = bool(post_win_transition.get("trophy_clicked", False))
                        attempt.reward_screen_seen = bool(post_win_transition.get("reward_screen_seen", False))
                        attempt.unlock_screen_seen = bool(post_win_transition.get("unlock_screen_seen", unlock_screen_seen))
                        attempt.post_win_transition_completed = bool(post_win_transition.get("post_win_transition_completed", False))
                        attempt.post_win_blocked_reason = str(post_win_transition.get("post_win_blocked_reason", post_win_blocked_reason) or "")
                        attempt.trophy_click_count = int(post_win_transition.get("trophy_click_count", 0) or 0)
                        attempt.reward_continue_click_count = int(post_win_transition.get("reward_continue_click_count", 0) or 0)
                        level.win_detected = True
                        level.trophy_click_count += attempt.trophy_click_count
                        level.reward_continue_click_count += attempt.reward_continue_click_count
                        level.post_win_transition_completed = attempt.post_win_transition_completed
                        level.post_win_blocked_reason = attempt.post_win_blocked_reason
                        level.unlock_screen_seen = unlock_screen_seen
                        level.unlock_screen_snapshot = unlock_snapshot
                        level.available_seed_names_after_level = available_after
                        level.unknown_unlock_objects = unknown_unlock_objects
                        level.unknown_visible_seed_names = _ordered_unique_seed_names(
                            list(final_state.get("unknownVisibleSeedNames", []) or [])
                            + _card_names(list(final_state.get("unknownVisibleSeedCards", []) or []))
                        )
                        level.unlocked_after_level = sorted(unlocked.keys())
                        if post_win_blocked_reason:
                            level.consecutive_wins = 0
                            last_blocked_reason = post_win_blocked_reason
                            attempt.blocked_reason = post_win_blocked_reason
                            context["blocked_reason"] = last_blocked_reason
                            update_attempt_progress(attempt, level, context, advance_on_wins)
                            writer.write(build_live_status(env, context, adventure_state=final_state))
                            level.attempt_logs.append(asdict(attempt))
                            break
                        level.advanced = True
                        update_attempt_progress(attempt, level, context, advance_on_wins)
                        writer.write(build_live_status(env, context, adventure_state=final_state))
                        print(f"[adventure] level={tracker_level} advanced=True")
                        print(f"[adventure] next level={tracker_level + 1}")
                        level.attempt_logs.append(asdict(attempt))
                        break
                    if episode_limit_reached:
                        stop_reason = "evaluation_episode_limit_reached"
                        replay_ok, replay_reason = replay_current_level_after_validation_win(
                            env,
                            writer,
                            context,
                            seed_selection_callback=runtime_loadout_selector,
                            expected_level=tracker_level,
                        )
                        if not replay_ok:
                            last_blocked_reason = replay_reason or "win_replay_reset_failed"
                            attempt.blocked_reason = last_blocked_reason
                            context["blocked_reason"] = last_blocked_reason
                        level.attempt_logs.append(asdict(attempt))
                        break
                    if attempt_index >= max_attempts_per_level:
                        level.attempt_logs.append(asdict(attempt))
                        break
                    replay_ok, replay_reason = replay_current_level_after_validation_win(
                        env,
                        writer,
                        context,
                        seed_selection_callback=runtime_loadout_selector,
                        expected_level=tracker_level,
                    )
                    if not replay_ok:
                        level.consecutive_wins = 0
                        last_blocked_reason = replay_reason or "win_replay_reset_failed"
                        attempt.blocked_reason = last_blocked_reason
                        context["blocked_reason"] = last_blocked_reason
                        update_attempt_progress(attempt, level, context, advance_on_wins)
                        writer.write(build_live_status(env, context))
                        level.attempt_logs.append(asdict(attempt))
                        break
                    update_attempt_progress(attempt, level, context, advance_on_wins)
                    level.attempt_logs.append(asdict(attempt))
                    continue

                level.consecutive_wins = 0
                if attempt.result == "loss":
                    level.losses += 1
                    context["total_losses"] = int(context.get("total_losses", 0) or 0) + 1
                    update_attempt_progress(attempt, level, context, advance_on_wins)
                    writer.write(build_live_status(env, context))
                    loss_boundary_reached = bool(
                        episode_limit_reached or attempt_index >= max_attempts_per_level
                    )
                    if episode_limit_reached:
                        stop_reason = "evaluation_episode_limit_reached"
                    context["state"] = "LOSS_RECOVERY"
                    recovery_observation, recovery_reset, recovery_reason = prepare_adventure_gameplay(
                        env,
                        writer,
                        context,
                        selected_seeds,
                        timeout=env.config.gameplay_ready_timeout,
                        seed_selection_callback=runtime_loadout_selector,
                        expected_level=tracker_level if strict_level_identity else None,
                    )
                    context["last_ui_action"] = recovery_reset.get("reset", recovery_reset)
                    if recovery_observation is None:
                        last_blocked_reason = recovery_reason or "loss_retry_failed"
                        attempt.blocked_reason = last_blocked_reason
                        context["blocked_reason"] = last_blocked_reason
                        update_attempt_progress(attempt, level, context, advance_on_wins)
                        writer.write(build_live_status(env, context))
                        level.attempt_logs.append(asdict(attempt))
                        break
                    context["state"] = "GAMEPLAY_READY_AFTER_LOSS"
                    context["last_reset_reason"] = "loss_recovery"
                    level.attempt_logs.append(asdict(attempt))
                    if loss_boundary_reached:
                        break
                    continue

                if attempt.result == "timeout":
                    update_attempt_progress(attempt, level, context, advance_on_wins)
                    writer.write(build_live_status(env, context))
                    level.attempt_logs.append(asdict(attempt))
                    timeout_boundary_reached = bool(
                        episode_limit_reached or attempt_index >= max_attempts_per_level
                    )
                    if episode_limit_reached:
                        stop_reason = "evaluation_episode_limit_reached"
                    try:
                        _, reset_info = env.base.reset(reset_reason="timeout")
                        context["last_ui_action"] = (
                            reset_info.get("reset", reset_info)
                            if isinstance(reset_info, dict)
                            else reset_info
                        )
                    except Exception as exc:
                        last_blocked_reason = "timeout_reset_failed"
                        attempt.blocked_reason = last_blocked_reason
                        context["blocked_reason"] = last_blocked_reason
                        context["last_error"] = str(exc)
                        if strict_level_identity:
                            raise RuntimeError(
                                "blocked_reason=streamer_evaluation_timeout_reset_failed"
                            ) from exc
                        break
                    time.sleep(max(0.5, env.config.poll_seconds))
                    if timeout_boundary_reached:
                        break
                    continue

                if attempt.result == "env_corruption":
                    update_attempt_progress(attempt, level, context, advance_on_wins)
                    writer.write(build_live_status(env, context))
                    if attempt_index >= max_attempts_per_level:
                        level.attempt_logs.append(asdict(attempt))
                        break
                    if episode_limit_reached:
                        stop_reason = "evaluation_episode_limit_reached"
                        level.attempt_logs.append(asdict(attempt))
                        break
                    context["state"] = "RESET_ENV_CORRUPTION"
                    try:
                        _, reset_info = env.base.reset(reset_reason="env_corruption")
                        context["last_ui_action"] = reset_info.get("reset", reset_info) if isinstance(reset_info, dict) else reset_info
                    except Exception as exc:
                        last_blocked_reason = "env_corruption_reset_failed"
                        attempt.blocked_reason = last_blocked_reason
                        context["blocked_reason"] = last_blocked_reason
                        context["last_error"] = str(exc)
                        update_attempt_progress(attempt, level, context, advance_on_wins)
                        writer.write(build_live_status(env, context))
                        level.attempt_logs.append(asdict(attempt))
                        break
                    time.sleep(max(0.5, env.config.poll_seconds))
                    level.attempt_logs.append(asdict(attempt))
                    continue

                if attempt.reset_failures:
                    last_blocked_reason = attempt.blocked_reason or "reset_failure"
                    context["blocked_reason"] = last_blocked_reason
                    update_attempt_progress(attempt, level, context, advance_on_wins)
                    writer.write(build_live_status(env, context))
                    level.attempt_logs.append(asdict(attempt))
                    break
                if attempt.bridge_errors:
                    last_blocked_reason = attempt.blocked_reason or "bridge_error"
                    context["blocked_reason"] = last_blocked_reason
                    update_attempt_progress(attempt, level, context, advance_on_wins)
                    writer.write(build_live_status(env, context))
                    level.attempt_logs.append(asdict(attempt))
                    break
                last_blocked_reason = attempt.blocked_reason or attempt.result
                context["blocked_reason"] = last_blocked_reason
                update_attempt_progress(attempt, level, context, advance_on_wins)
                writer.write(build_live_status(env, context))
                level.attempt_logs.append(asdict(attempt))
                break

            aggregate_level_metrics(level)
            if not level.unlocked_after_level:
                level.unlocked_after_level = sorted(unlocked.keys())
            if not level.available_seed_names_after_level:
                try:
                    end_state = env.base.adventure_screen_state()
                    level.available_seed_names_after_level = _ordered_unique_seed_names(list(end_state.get("availableSeedNames", []) or []))
                    level.unknown_visible_seed_names = _ordered_unique_seed_names(
                        list(end_state.get("unknownVisibleSeedNames", []) or []) + level.unknown_visible_seed_names
                    )
                except Exception:
                    pass
            if not level.advanced:
                level.blocked_reason = level.blocked_reason or last_blocked_reason or "max_attempts_reached"
                context["blocked_reason"] = level.blocked_reason
                stop_reason = resolve_stop_reason(level.blocked_reason)
            context["unlocked_seeds"] = sorted(unlocked.keys())
            levels.append(level)
            context["eval_summary"] = summarize_progress(
                levels,
                terminal_episodes_only=evaluation_episode_limit is not None,
            )
            if env is not None:
                writer.write(build_live_status(env, context))
            if episode_limit_reached:
                stop_reason = "evaluation_episode_limit_reached"
                break
            if not level.advanced:
                break

        next_adventure_level = int(adventure_start_level)
        if levels:
            final_level = levels[-1]
            next_adventure_level = int(final_level.level) + (1 if final_level.advanced else 0)
        if strict_level_identity:
            handoff_observation, _, handoff_blocked_reason = prepare_adventure_gameplay(
                env,
                writer,
                context,
                list(context.get("selected_seeds", []) or config.get("seed_list", [])),
                timeout=env.config.gameplay_ready_timeout,
                seed_selection_callback=runtime_loadout_selector,
                expected_level=next_adventure_level,
            )
            if handoff_observation is None:
                raise RuntimeError(
                    "blocked_reason=streamer_evaluation_handoff_failed: "
                    f"expected={next_adventure_level} reason={handoff_blocked_reason}"
                )
        end_state = env.base.adventure_screen_state()
        level_identity_end = adventure_level_identity_diagnostics(
            end_state,
            next_adventure_level,
        )
        context["level_identity_end"] = dict(level_identity_end)
        if strict_level_identity and not level_identity_end.get("level_identity_reliable"):
            raise RuntimeError(
                "blocked_reason=streamer_evaluation_end_level_unreliable: "
                f"expected={next_adventure_level} "
                f"reason={level_identity_end.get('level_identity_reason', '')} "
                f"mismatches={level_identity_end.get('level_identity_mismatches', [])}"
            )

    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                print(f"[adventure] warning: env.close failed: {exc}")
        if gui_process is not None and gui_process.poll() is not None:
            gui_process = None

    if not stop_reason:
        stop_reason = "completed_requested_levels"
    context["status"] = "complete"
    context["stop_reason"] = stop_reason
    print(f"[adventure] stopping reason={stop_reason}")
    payload = {
        "status": "complete",
        "mode": ADVENTURE_GENERALIST_EVAL_RUN_MODE,
        "model_path": str(model_path),
        "stop_reason": stop_reason,
        "levels_requested": int(max_adventure_levels),
        "levels_completed": len(levels),
        "advance_on_wins": int(advance_on_wins),
        "max_adventure_levels": int(max_adventure_levels),
        "max_attempts_per_level": int(max_attempts_per_level),
        "evaluation_episode_limit": episode_limit,
        "evaluation_episodes_completed": int(episodes_completed),
        "adventure_start_level": int(adventure_start_level),
        "next_adventure_level": int(
            levels[-1].level + (1 if levels[-1].advanced else 0)
            if levels
            else adventure_start_level
        ),
        "level_identity_initial": level_identity_initial,
        "level_identity_start": level_identity_start,
        "level_identity_start_navigation_deferred": bool(
            level_identity_start_navigation_deferred
        ),
        "level_identity_end": level_identity_end,
        "adventure_start_level_label": adventure_level_metadata(adventure_start_level, 1)["adventure_level_label"],
        "soft_max_steps": int(soft_max_steps),
        "hard_max_steps": int(hard_max_steps),
        "final_wave_extension_enabled": bool(final_wave_extension),
        "tactical_mask_enabled": bool(
            config.get("tactical_masks") or config.get("wallnut_tactical_mask") or config.get("cherrybomb_tactical_mask")
        ),
        "wallnut_tactical_mask_enabled": bool(config.get("tactical_masks") or config.get("wallnut_tactical_mask")),
        "cherrybomb_tactical_mask_enabled": bool(config.get("tactical_masks") or config.get("cherrybomb_tactical_mask")),
        "deterministic": bool(deterministic),
        "model_compatibility": context.get("model_compatibility", {}),
        "levels": [asdict(level) for level in levels],
        "unlocked_seeds_final": sorted(unlocked.keys()),
        "bridge_errors": sum(int(attempt.get("bridge_errors", 0) or 0) for level in levels for attempt in level.attempt_logs),
        "reset_failures": sum(int(attempt.get("reset_failures", 0) or 0) for level in levels for attempt in level.attempt_logs),
        "summary": summarize_progress(
            levels,
            terminal_episodes_only=evaluation_episode_limit is not None,
        ),
    }
    output_path = run_dir / "adventure_progression_results.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved Adventure progression JSON to {output_path}")
    if live_status_path is not None:
        writer.write(
            {
                **writer.last_payload,
                "eval": payload["summary"],
                "status": "complete",
                "stop_reason": stop_reason,
                "soft_max_steps": int(soft_max_steps),
                "hard_max_steps": int(hard_max_steps),
                "final_wave_extension_enabled": bool(final_wave_extension),
            }
        )
    return payload
