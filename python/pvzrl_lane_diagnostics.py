"""Pure composition of the public lane-diagnostics payload."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from pvzrl_observation_facts import StepFacts
from pvzrl_rewards import (
    RewardCompositionState,
    _actionable_threat_rows,
    _lane_danger_by_row,
    _lane_zombie_counts,
    _legal_peashooter_actions_by_row,
    _mower_risk_rows,
    _nearby_zombie_context,
    _nearest_zombie_x_by_row,
    _plant_counts_by_row,
    _row_count,
    _tough_zombies_by_row,
    _valuable_plant_columns,
    _wallnut_blocker_count,
    reward_action_from_result,
)


@dataclass(frozen=True, slots=True)
class LaneDiagnosticsInput:
    """All values captured by the caller after reward composition."""

    previous_facts: StepFacts
    current_facts: StepFacts
    post_reward_state: RewardCompositionState
    current_legal_actions: Tuple[int, ...] = ()
    previous_legal_actions: Tuple[int, ...] = ()
    action_result: Mapping[str, Any] = field(default_factory=dict)
    mask_diagnostics: Mapping[str, Any] = field(default_factory=dict)
    cherry_delayed_diagnostics: Mapping[str, Any] = field(default_factory=dict)
    fallback_rows: int = 5
    fallback_columns: int = 9
    close_threat_threshold: float = 0.6


def _row_dict(values: Mapping[int, Any], rows: int) -> Dict[str, int]:
    return {str(row): int(values.get(row, 0)) for row in range(max(0, rows))}


def _undefended_rows(
    facts: StepFacts,
    rows: int,
    threshold: float,
    *,
    lanes: Mapping[int, int],
    shooters: Mapping[int, int],
    danger: Mapping[int, float],
) -> list[int]:
    return [row for row, count in lanes.items() if count > 0 and shooters.get(row, 0) == 0 and danger.get(row, 0.0) >= threshold]


def compose_lane_diagnostics(data: LaneDiagnosticsInput) -> Dict[str, Any]:
    """Compose diagnostics without masks, legal-action queries, mutation, or I/O."""

    prior, current = data.previous_facts, data.current_facts
    rows = _row_count(current, data.fallback_rows)
    cols = max(0, int(current.columns or data.fallback_columns))
    prior_rows = _row_count(prior, data.fallback_rows)
    prior_cols = max(0, int(prior.columns or data.fallback_columns))
    action = reward_action_from_result(data.action_result)

    current_lanes = _lane_zombie_counts(current, rows)
    prior_lanes = _lane_zombie_counts(prior, prior_rows)
    danger = _lane_danger_by_row(current, rows)
    prior_danger = _lane_danger_by_row(prior, prior_rows)
    plants = _plant_counts_by_row(current, rows)
    shooters = _plant_counts_by_row(current, rows, plant_type=0)
    prior_shooters = _plant_counts_by_row(prior, prior_rows, plant_type=0)
    sunflowers = _plant_counts_by_row(current, rows, plant_type=1)
    prior_sunflowers = _plant_counts_by_row(prior, prior_rows, plant_type=1)
    threat_rows = [row for row, count in current_lanes.items() if count > 0]
    prior_threat_rows = [row for row, count in prior_lanes.items() if count > 0]
    undefended = _undefended_rows(
        current,
        rows,
        data.close_threat_threshold,
        lanes=current_lanes,
        shooters=shooters,
        danger=danger,
    )
    prior_undefended = _undefended_rows(
        prior,
        prior_rows,
        data.close_threat_threshold,
        lanes=prior_lanes,
        shooters=prior_shooters,
        danger=prior_danger,
    )
    legal_peas = _legal_peashooter_actions_by_row(current, rows, cols, data.current_legal_actions)
    prior_legal_peas = _legal_peashooter_actions_by_row(prior, prior_rows, prior_cols, data.previous_legal_actions)

    mower_losses: Optional[Dict[int, int]] = None
    if prior.mower.active_rows is not None and current.mower.active_rows is not None:
        mower_losses = {row: 0 for row in range(max(prior_rows, rows))}
        for row in prior.mower.active_rows - current.mower.active_rows:
            if row in mower_losses:
                mower_losses[row] = 1

    response_row = int(action.row)
    response_applied = bool(
        not action.illegal and action.plant_placed and action.plant_type == 0
        and response_row >= 0 and prior_lanes.get(response_row, 0) > 0
        and prior_shooters.get(response_row, 0) == 0
    )
    if not response_applied:
        response_row = -1
    defense_opportunities = [row for row in prior_undefended if prior_legal_peas.get(row, 0) > 0]
    if response_row >= 0 and response_row not in defense_opportunities:
        defense_opportunities.append(response_row)
    defense_responses = {response_row: 1} if response_row >= 0 else {}

    ready_types = set(prior.ready_seed_types)
    actionable = _actionable_threat_rows(prior, prior_rows)
    current_actionable = _actionable_threat_rows(current, rows)
    tough = _tough_zombies_by_row(current, rows)
    prior_tough = _tough_zombies_by_row(prior, prior_rows)
    mower_risk = _mower_risk_rows(prior, prior_rows)
    current_mower_risk = _mower_risk_rows(current, rows)
    plant_placed = bool(action.plant_placed and not action.illegal)
    plant_type, action_row, action_col = int(action.plant_type), int(action.row), int(action.column)
    action_kind = str(action.kind)
    covered_rows = sum(1 for count in shooters.values() if count > 0)
    coverage_rate = float(covered_rows) / float(rows) if rows > 0 else 0.0
    prior_zero_defense = [row for row in prior_threat_rows if prior_shooters.get(row, 0) == 0]
    prior_defense_available = [row for row in prior_zero_defense if prior_legal_peas.get(row, 0) > 0]
    zero_defense = [row for row in threat_rows if shooters.get(row, 0) == 0]

    first_pea = first_defense = mower_save_row = -1
    overdefended = least_defended = full_coverage = False
    sunflower_overbuild = sunflower_greed = False
    wallnut_blocks = wallnut_low_value = wallnut_threat = False
    wallnut_between = wallnut_frontline = wallnut_emergency = False
    cherry_threat = cherry_low_value = cherry_cluster = cherry_emergency = False
    tough_response = False
    if plant_placed and 0 <= action_row < rows:
        if plant_type == 0:
            prior_count = prior_shooters.get(action_row, 0)
            if prior_count == 0 and shooters.get(action_row, 0) > 0:
                first_pea = action_row
                if action_row in prior_threat_rows:
                    first_defense = action_row
            if prior_threat_rows:
                counts = {row: prior_shooters.get(row, 0) for row in prior_threat_rows}
                least_defended = action_row in counts and prior_count == min(counts.values())
            overdefended = prior_count >= 2 and any(row != action_row for row in prior_zero_defense)
            full_coverage = rows > 0 and sum(1 for count in prior_shooters.values() if count > 0) < rows and covered_rows >= rows
        elif plant_type == 1:
            sunflower_overbuild = sum(sunflowers.values()) >= 5 and any(prior_shooters.get(row, 0) == 0 for row in range(rows)) and bool(prior_threat_rows)
            sunflower_greed = sum(prior_sunflowers.values()) >= 6 and bool(prior_zero_defense) and 0 in ready_types
        elif plant_type == 3:
            nearest = _nearest_zombie_x_by_row(prior, prior_rows).get(action_row)
            valuable_behind = any(col < action_col for col in _valuable_plant_columns(prior, action_row))
            wallnut_blocks = bool(action_row in prior_threat_rows and valuable_behind and (nearest is None or float(nearest) - float(action_col) <= 4.0 or prior_shooters.get(action_row, 0) == 0))
            wallnut_threat = action_row in prior_threat_rows
            wallnut_between = bool(action_row in prior_threat_rows and nearest is not None and float(action_col) < float(nearest))
            wallnut_frontline = bool(wallnut_blocks and valuable_behind)
            wallnut_emergency = action_row in mower_risk
            wallnut_low_value = bool(not wallnut_blocks and (action_row not in prior_threat_rows or (nearest is not None and float(nearest) < float(action_col) - 0.5)))
        elif plant_type == 2:
            nearby = _nearby_zombie_context(prior, action_row, action_col, radius=2.75)
            cherry_threat = action_row in prior_threat_rows
            cherry_low_value = not cherry_threat and nearby.get("zombies", 0) <= 0
            cherry_cluster = int(nearby.get("zombies", 0) or 0) >= 2
            cherry_emergency = action_row in mower_risk
        if action_row in mower_risk and (plant_type in {0, 2, 3} or action_kind == "fusion"):
            mower_save_row = action_row
        if prior_tough.get(action_row, {}).get("tough", 0) > 0 and (plant_type in {0, 2, 3} or action_kind == "fusion"):
            tough_response = True

    waited_rows = list(prior_defense_available) if action_kind == "wait" else []
    planted_elsewhere = [row for row in prior_defense_available if row != action_row] if plant_placed else []
    sunflower_undefended = list(prior_zero_defense) if plant_placed and plant_type == 1 else []
    state = data.post_reward_state
    age_sum = {row: int(state.undefended_threat_age_by_row[row]) for row in zero_defense if row < len(state.undefended_threat_age_by_row)}
    age_count = {row: 1 for row in age_sum}
    age_max = {row: int(state.max_undefended_threat_age_by_row[row]) for row in range(rows) if row < len(state.max_undefended_threat_age_by_row)}
    high_danger = [row for row, value in danger.items() if value >= 0.65 and shooters.get(row, 0) == 0 and _wallnut_blocker_count(current, row) <= 0]
    mower_exposure = [row for row in current_mower_risk if shooters.get(row, 0) == 0 and _wallnut_blocker_count(current, row) <= 0]

    mask = data.mask_diagnostics
    mask_counts = Counter(mask.get("python_mask_block_reason_counts", {}) or {})
    action_audit = data.action_result.get("actionAudit") if isinstance(data.action_result.get("actionAudit"), Mapping) else {}
    pre_audit = data.action_result.get("preStepMaskAudit") if isinstance(data.action_result.get("preStepMaskAudit"), Mapping) else {}
    if pre_audit:
        mask_counts[str(pre_audit.get("pythonFilterReason") or "blocked")] += 1
    illegal_reason = str(data.action_result.get("illegalReason") or "")
    cooldown_exposed = illegal_reason == "cooldown" and bool(action_audit.get("pythonMaskValueBefore") or action_audit.get("bridgeLegalActionsValueBefore"))
    cherry = data.cherry_delayed_diagnostics

    return {
        "action_kind": action.kind,
        "action_plant_type": plant_type,
        "action_row": action_row,
        "action_column": action_col,
        "plant_placed": bool(action.plant_placed),
        "lane_response_reward_applied": response_applied,
        "previous_total_danger": sum(prior_danger.values()),
        "current_total_danger": sum(danger.values()),
        "danger_delta": sum(danger.values()) - sum(prior_danger.values()),
        "mowers_lost_this_step": max(0, prior.mower.count - current.mower.count),
        "mower_losses_by_row": _row_dict(mower_losses, rows) if mower_losses is not None else {},
        "plants_by_row": _row_dict(plants, rows),
        "peashooters_by_row": _row_dict(shooters, rows),
        "sunflowers_by_row": _row_dict(sunflowers, rows),
        "threat_rows": threat_rows,
        "undefended_threat_rows": undefended,
        "threat_steps_by_row": _row_dict({row: 1 for row in threat_rows}, rows),
        "undefended_threat_steps_by_row": _row_dict({row: 1 for row in undefended}, rows),
        "undefended_threat_age_sum_by_row": _row_dict(age_sum, rows),
        "undefended_threat_age_count_by_row": _row_dict(age_count, rows),
        "undefended_threat_age_max_by_row": _row_dict(age_max, rows),
        "wait_under_threat": action_kind == "wait" and bool(prior_threat_rows),
        "wait_while_actionable_threat": action_kind == "wait" and bool(actionable),
        "wait_while_actionable_threat_by_row": _row_dict({row: 1 for row in actionable}, rows),
        "wait_while_peashooter_affordable_ready": action_kind == "wait" and 0 in ready_types and bool(actionable),
        "wait_while_wallnut_affordable_ready": action_kind == "wait" and 3 in ready_types and bool(actionable),
        "wait_while_cherrybomb_affordable_ready": action_kind == "wait" and 2 in ready_types and bool(actionable),
        "close_zombie_undefended_count": len(undefended),
        "close_zombie_with_no_defense_count": len(current_actionable),
        "close_zombie_undefended_rows": undefended,
        "illegal_reason": illegal_reason,
        "legal_peashooter_actions_by_row": _row_dict(legal_peas, rows),
        "pre_action_legal_peashooter_actions_by_row": _row_dict(prior_legal_peas, rows),
        "peashooter_available_but_waited_by_row": _row_dict({row: 1 for row in waited_rows}, rows),
        "peashooter_available_but_planted_elsewhere_by_row": _row_dict({row: 1 for row in planted_elsewhere}, rows),
        "sunflower_while_undefended_threat_by_row": _row_dict({row: 1 for row in sunflower_undefended}, rows),
        "row_defense_opportunities_by_row": _row_dict({row: 1 for row in defense_opportunities}, rows),
        "row_defense_responses_by_row": _row_dict(defense_responses, rows),
        "threatened_rows_with_zero_defender_steps_by_row": _row_dict({row: 1 for row in zero_defense}, rows),
        "plant_in_threatened_row": plant_placed and action_row in prior_threat_rows,
        "plant_in_unthreatened_row": plant_placed and action_row >= 0 and action_row not in prior_threat_rows,
        "first_peashooter_row": first_pea,
        "first_defense_row": first_defense,
        "overdefended_while_undefended": overdefended,
        "least_defended_threatened_row_plant": least_defended,
        "rows_with_peashooter_count": covered_rows,
        "peashooter_coverage_rate": coverage_rate,
        "all_rows_peashooter_covered": full_coverage,
        "sunflower_count_when_first_full_coverage": sum(sunflowers.values()) if full_coverage else -1,
        "sunflower_overbuild_before_defense": sunflower_overbuild,
        "sunflower_greed_while_defense_missing": sunflower_greed,
        "active_threat_rows_without_peashooter_count": len(zero_defense),
        "wallnut_placement": plant_placed and plant_type == 3,
        "wallnut_threatened_lane": wallnut_threat,
        "wallnut_between_zombie_and_house": wallnut_between,
        "wallnut_frontline": wallnut_frontline,
        "wallnut_emergency_block": wallnut_emergency,
        "wallnut_blocks_active_threat": wallnut_blocks,
        "wallnut_low_value_placement": wallnut_low_value,
        "wallnut_placements_by_row": _row_dict({action_row: 1} if plant_placed and plant_type == 3 else {}, rows),
        "wallnut_placements_by_col": {str(action_col): 1} if plant_placed and plant_type == 3 and action_col >= 0 else {},
        "cherrybomb_used": plant_placed and plant_type == 2,
        "cherrybomb_used_under_threat": cherry_threat,
        "cherrybomb_used_low_value": cherry_low_value,
        "cherrybomb_cluster_use": cherry_cluster,
        "cherrybomb_emergency_use": cherry_emergency,
        "cherrybomb_delayed_kills": int(cherry.get("kills", 0) or 0),
        "cherrybomb_delayed_zero_kill": int(cherry.get("zero_kill", 0) or 0),
        "cherrybomb_buckethead_kill_credit": int(cherry.get("buckethead", 0) or 0),
        "cherrybomb_conehead_kill_credit": int(cherry.get("conehead", 0) or 0),
        "mower_risk_steps_by_row": _row_dict({row: 1 for row in current_mower_risk}, rows),
        "high_danger_unanswered_steps": len(high_danger),
        "mower_exposure_steps": len(mower_exposure),
        "max_row_danger": max(danger.values()) if danger else 0.0,
        "avg_row_danger": sum(danger.values()) / max(1, len(danger)) if danger else 0.0,
        "mower_saves_estimated_by_row": _row_dict({mower_save_row: 1} if mower_save_row >= 0 else {}, rows),
        "buckethead_count_by_row": _row_dict({row: values.get("buckethead", 0) for row, values in tough.items()}, rows),
        "conehead_count_by_row": _row_dict({row: values.get("conehead", 0) for row, values in tough.items()}, rows),
        "tough_zombie_count_by_row": _row_dict({row: values.get("tough", 0) for row, values in tough.items()}, rows),
        "tough_zombie_response": tough_response,
        "legal_actions_by_seed_slot": mask.get("legal_actions_by_seed_slot", {}),
        "bridge_legal_actions_by_seed_slot": mask.get("bridge_legal_actions_by_seed_slot", {}),
        "python_mask_block_reason_counts": dict(sorted(mask_counts.items())),
        "tactical_mask_enabled": bool(mask.get("tactical_mask_enabled")),
        "wallnut_tactical_mask_enabled": bool(mask.get("wallnut_tactical_mask_enabled")),
        "cherrybomb_tactical_mask_enabled": bool(mask.get("cherrybomb_tactical_mask_enabled")),
        "wallnut_actions_masked": int(mask.get("wallnut_actions_masked") or 0),
        "cherrybomb_actions_masked": int(mask.get("cherrybomb_actions_masked") or 0),
        "wallnut_actions_available": int(mask.get("wallnut_actions_available") or 0),
        "cherrybomb_actions_available": int(mask.get("cherrybomb_actions_available") or 0),
        "mask_all_but_wait_count": int(mask.get("mask_all_but_wait_count") or 0),
        "pre_step_mask_blocked_action": bool(data.action_result.get("preStepMaskBlockedAction")),
        "cooldown_illegal_exposed_by_mask": cooldown_exposed,
        "mask_bridge_disagreement": bool(action_audit and action_audit.get("pythonMaskValueBefore") != action_audit.get("bridgeLegalActionsValueBefore")),
    }


__all__ = ["LaneDiagnosticsInput", "compose_lane_diagnostics"]
