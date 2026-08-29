"""Pure safety-event composition; lifecycle classification stays caller-owned."""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set as AbstractSet
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Optional

from pvzrl_observation_facts import StepFacts, build_step_facts
from pvzrl_registry import get_plant_registry


_EVENT_FIELDS = (
    ("step", "frameCount"), ("wave", "wave"), ("maxWave", "maxWave"),
    ("zombieCount", "zombieCount"), ("plantCount", "plantCount"),
    ("gameplayReady", "gameplayReady"), ("screenState", "screenState"),
    ("nextStep", "nextStep"), ("done", "done"), ("over", "over"),
    ("terminalHint", "terminalHint"),
)


@dataclass(frozen=True, slots=True)
class EnvironmentSafetyResult:
    """Newly owned diagnostics plus immutable mower-loss state."""

    diagnostics: Dict[str, Any]
    next_lost_mower_rows: FrozenSet[int]
    next_missing_mower_rows: FrozenSet[int]
    mower_baseline_ready: bool


def _coerce(value: Any, default: Any) -> Any:
    try:
        return type(default)(default if value is None else value)
    except (TypeError, ValueError):
        return default


def _row_count(observation: Mapping[str, Any], fallback: int) -> int:
    return max(0, _coerce(observation.get("rowCount") or fallback, 0))


def _cooldowns_by_slot(
    observation: Mapping[str, Any], fallback_plant_types: Sequence[int]
) -> Dict[int, Dict[str, Any]]:
    snapshots: Dict[int, Dict[str, Any]] = {}
    slots = observation.get("seedSlots", []) or []
    source = slots if isinstance(slots, list) and slots else observation.get("cardCooldowns", []) or []
    if not isinstance(source, list):
        return snapshots
    for fallback_index, item in enumerate(source):
        if not isinstance(item, Mapping):
            continue
        slot = _coerce(item.get("slotIndex"), fallback_index)
        configured = fallback_plant_types[slot] if 0 <= slot < len(fallback_plant_types) else -1
        snapshots[slot] = {
            "slot": slot,
            "plantType": _coerce(item.get("plantType"), configured),
            "plantTypeName": str(item.get("plantTypeName") or ""),
            "currentCooldown": _coerce(item.get("currentCooldown"), 0.0),
            "rawCooldown": _coerce(item.get("rawCooldown"), 0.0),
            "fullCooldown": _coerce(item.get("fullCooldown"), 0.0),
            "ready": bool(item.get("ready")),
            "cardInstanceId": _coerce(item.get("cardInstanceId"), 0),
        }
    return snapshots


def _cooldowns_from_seed_slot_facts(facts: StepFacts) -> Dict[int, Dict[str, Any]]:
    """Project the legacy cooldown snapshot from the already-built slot facts.

    Iterating the positional tuple and overwriting by explicit slot index keeps
    the bridge contract's duplicate-slot, last-entry-wins behavior.  Invalid
    non-mapping entries remain excluded exactly as in ``_cooldowns_by_slot``.
    """

    snapshots: Dict[int, Dict[str, Any]] = {}
    for fact in facts.seed_slots:
        if not fact.valid:
            continue
        slot = int(fact.slot_index)
        snapshots[slot] = {
            "slot": slot,
            "plantType": int(fact.cooldown_plant_type),
            "plantTypeName": str(fact.cooldown_plant_type_name),
            "currentCooldown": float(fact.current_cooldown),
            "rawCooldown": float(fact.raw_cooldown),
            "fullCooldown": float(fact.full_cooldown),
            "ready": bool(fact.ready),
            "cardInstanceId": int(fact.card_instance_id),
        }
    return snapshots


def _cooldown_snapshot(
    observation: Mapping[str, Any],
    facts: StepFacts,
    fallback_plant_types: Sequence[int],
) -> Dict[int, Dict[str, Any]]:
    raw_slots = observation.get("seedSlots", [])
    if isinstance(raw_slots, list) and raw_slots:
        return _cooldowns_from_seed_slot_facts(facts)
    # StepFacts deliberately synthesizes configured seed slots when the bridge
    # omits them; retain the distinct legacy cardCooldowns fallback here.
    return _cooldowns_by_slot(observation, fallback_plant_types)


def append_safety_event(
    events: list[Dict[str, Any]],
    event: str,
    observation: Mapping[str, Any],
    **fields: Any,
) -> None:
    events.append({
        "event": event,
        **{target: observation.get(source) for target, source in _EVENT_FIELDS},
        **fields,
    })


def compose_environment_safety_diagnostics(
    previous: Optional[Mapping[str, Any]],
    current: Mapping[str, Any],
    *,
    requested_action: int,
    fallback_plant_types: Sequence[int] = (),
    fallback_row_count: int = 5,
    lost_mower_rows: AbstractSet[int] = frozenset(),
    missing_mower_rows: AbstractSet[int] = frozenset(),
    mower_baseline_ready: bool = False,
    live_board_progress: bool,
    post_win_signal_present: bool,
    cleanup_signal_active: bool,
    suspicious_cleanup_signal_during_gameplay: bool,
    previous_confirmed_postgame: bool,
    current_confirmed_postgame: bool,
    previous_facts: Optional[StepFacts] = None,
    current_facts: Optional[StepFacts] = None,
) -> EnvironmentSafetyResult:
    """Compose safety diagnostics and immutable per-attempt mower history."""

    fallback_types = tuple(int(value) for value in fallback_plant_types)
    current_snapshot = current_facts or build_step_facts(current, fallback_types)
    previous_snapshot = previous_facts or (
        build_step_facts(previous, fallback_types) if previous is not None else None
    )
    next_lost_rows = {int(row) for row in lost_mower_rows}
    next_missing_rows = {int(row) for row in missing_mower_rows}
    next_mower_baseline_ready = bool(mower_baseline_ready)
    events: list[Dict[str, Any]] = []

    if live_board_progress and post_win_signal_present:
        append_safety_event(events, "false_reward_unlock_during_gameplay", current, last_action=requested_action)
        append_safety_event(events, "post_win_veto_live_board", current, last_action=requested_action)
    if live_board_progress and (
        current.get("nextStep") == "cleanup_reward_ui" or cleanup_signal_active
    ):
        append_safety_event(events, "false_cleanup_reward_ui_during_gameplay", current, last_action=requested_action)
    if suspicious_cleanup_signal_during_gameplay:
        append_safety_event(
            events,
            "suspicious_cleanup_reward_ui_during_gameplay",
            current,
            last_action=requested_action,
        )

    active_runtime_context = bool(
        previous
        and not previous_confirmed_postgame
        and not current_confirmed_postgame
        and not current_snapshot.lifecycle.seed_selection_active
        and not current_snapshot.lifecycle.over
    )
    respawn_rows: list[int] = []
    if previous and previous_snapshot is not None and active_runtime_context:
        previous_rows = previous_snapshot.mower.active_rows
        current_rows = current_snapshot.mower.active_rows
        if current_rows is not None:
            rows = max(_row_count(previous, fallback_row_count), _row_count(current, fallback_row_count))
            expected_rows = set(range(rows))
            current_active_rows = {
                int(row) for row in current_rows if 0 <= int(row) < rows
            }
            full_current_baseline = bool(
                rows > 0
                and expected_rows.issubset(current_active_rows)
                and int(current_snapshot.mower.logical_count) >= rows
                and int(current_snapshot.mower.visible_count) >= rows
            )

            if not next_mower_baseline_ready:
                # A structurally playable board can precede mower materialization.
                # UNKNOWN -> PRESENT establishes the baseline; it is not a respawn.
                if full_current_baseline:
                    next_mower_baseline_ready = True
                    next_missing_rows.clear()
            else:
                absent_rows = expected_rows - current_active_rows
                confirmed_absent_rows = absent_rows.intersection(next_missing_rows)
                next_lost_rows.update(confirmed_absent_rows)
                # Two consecutive known observations are required before an
                # absence is treated as consumed. A one-frame scan omission is
                # forgotten as soon as the row is visible again.
                next_missing_rows = absent_rows - next_lost_rows
                respawn_rows = sorted(current_active_rows.intersection(next_lost_rows))

            if respawn_rows:
                previous_active_rows = previous_rows or frozenset()
                append_safety_event(
                    events,
                    "mower_respawn_detected",
                    current,
                    rows=respawn_rows,
                    mowers_before=[row in previous_active_rows for row in range(rows)],
                    mowers_after=[row in current_active_rows for row in range(rows)],
                    last_action=requested_action,
                )

        previous_plant_count = _coerce(previous.get("plantCount"), 0)
        current_plant_count = _coerce(current.get("plantCount"), 0)
        previous_visible_plants = _coerce(previous.get("visiblePlantObjectCount"), 0)
        current_visible_plants = _coerce(current.get("visiblePlantObjectCount"), 0)
        previous_wave = int(previous_snapshot.lifecycle.wave)
        current_wave = int(current_snapshot.lifecycle.wave)
        previous_time = float(previous_snapshot.lifecycle.time)
        current_time = float(current_snapshot.lifecycle.time)
        previous_mower_count = int(previous_snapshot.mower.count)
        current_mower_count = int(current_snapshot.mower.count)
        board_refreshed = (
            previous_plant_count >= 8
            and current_plant_count <= max(1, int(previous_plant_count * 0.25))
            and current_visible_plants <= max(1, int(previous_visible_plants * 0.25))
        ) or (previous_wave > 0 and current_wave < previous_wave) or (
            previous_time > 5.0 and current_time + 1.0 < previous_time
        ) or bool(respawn_rows)
        if board_refreshed:
            append_safety_event(
                events,
                "board_refresh_detected",
                current,
                plant_count_before=previous_plant_count,
                plant_count_after=current_plant_count,
                visible_plant_count_before=previous_visible_plants,
                visible_plant_count_after=current_visible_plants,
                wave_before=previous_wave,
                wave_after=current_wave,
                time_before=previous_time,
                time_after=current_time,
                mower_count_before=previous_mower_count,
                mower_count_after=current_mower_count,
                last_action=requested_action,
            )

        previous_cooldowns = _cooldown_snapshot(
            previous,
            previous_snapshot,
            fallback_types,
        )
        current_cooldowns = _cooldown_snapshot(
            current,
            current_snapshot,
            fallback_types,
        )
        cooldown_reset_candidates: list[Dict[str, Any]] = []
        for slot, before in previous_cooldowns.items():
            after = current_cooldowns.get(slot)
            if not after:
                continue
            before_cd = float(before.get("currentCooldown") or 0.0)
            after_cd = float(after.get("currentCooldown") or 0.0)
            full_cd = max(float(before.get("fullCooldown") or 0.0), float(after.get("fullCooldown") or 0.0))
            drop_amount = max(0.0, before_cd - after_cd)
            elapsed_game_time = max(0.0, current_time - previous_time)
            suspicious_drop = (
                before_cd > max(1.0, full_cd * 0.35)
                and after_cd <= 0.05
                and elapsed_game_time < max(0.0, drop_amount - 0.75)
            )
            if suspicious_drop:
                plant = after.get("plantTypeName") or get_plant_registry().canonical_name(
                    int(after.get("plantType", -1))
                )
                cooldown_reset_candidates.append(
                    {
                        "slot": slot,
                        "plant": plant,
                        "cooldown_before": before_cd,
                        "cooldown_after": after_cd,
                        "full_cooldown": full_cd,
                        "drop_amount": drop_amount,
                        "elapsed_game_time": elapsed_game_time,
                        "last_action": requested_action,
                    }
                )
            before_id = int(before.get("cardInstanceId") or 0)
            after_id = int(after.get("cardInstanceId") or 0)
            if before_id and after_id and before_id != after_id:
                plant = after.get("plantTypeName") or get_plant_registry().canonical_name(
                    int(after.get("plantType", -1))
                )
                append_safety_event(
                    events,
                    "seed_slot_object_id_changed_during_gameplay",
                    current,
                    slot=slot,
                    plant=plant,
                    card_id_before=before_id,
                    card_id_after=after_id,
                    last_action=requested_action,
                )
        if len(cooldown_reset_candidates) >= 2:
            for candidate in cooldown_reset_candidates:
                append_safety_event(events, "cooldown_reset_detected", current, **candidate)
        elif cooldown_reset_candidates:
            append_safety_event(
                events,
                "cooldown_drop_observed",
                current,
                **cooldown_reset_candidates[0],
                reason="single_slot_drop_not_global_reset",
            )

    corruption_names = {
        "mower_respawn_detected",
        "cooldown_reset_detected",
        "seed_slot_object_id_changed_during_gameplay",
        "board_refresh_detected",
    }
    corruption_count = sum(event.get("event") in corruption_names for event in events)
    diagnostics: Dict[str, Any] = {
        "environment_corruption_detected": bool(corruption_count),
        "environment_corruption_penalty": 10.0 if corruption_count else 0.0,
        "env_corruption_count": corruption_count,
    }
    for key, name in (
        ("mower_respawn_detected_count", "mower_respawn_detected"),
        ("cooldown_reset_detected_count", "cooldown_reset_detected"),
        ("board_refresh_detected_count", "board_refresh_detected"),
        ("false_reward_unlock_during_gameplay_count", "false_reward_unlock_during_gameplay"),
        ("false_cleanup_reward_ui_during_gameplay_count", "false_cleanup_reward_ui_during_gameplay"),
        ("post_win_veto_live_board_count", "post_win_veto_live_board"),
        ("blocked_cleanup_during_gameplay_count", "suspicious_cleanup_reward_ui_during_gameplay"),
        ("suspicious_cleanup_reward_ui_count", "suspicious_cleanup_reward_ui_during_gameplay"),
    ):
        diagnostics[key] = sum(event.get("event") == name for event in events)
    diagnostics.update(
        reset_reward_ui_cleanup_count=0,
        reset_reward_ui_cleanup_blocked_count=0,
        safety_events=events,
    )
    return EnvironmentSafetyResult(
        diagnostics,
        frozenset(next_lost_rows),
        frozenset(next_missing_rows),
        bool(next_mower_baseline_ready),
    )


__all__ = [
    "EnvironmentSafetyResult",
    "append_safety_event",
    "compose_environment_safety_diagnostics",
]
