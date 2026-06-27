"""Conservative fusion diagnostics and scripted assist helpers.

Fusion is intentionally modeled outside the PPO action space.  The helpers in
this module only score and validate bridge-reported candidates; the bridge
remains the final source of legality for any actual fusion execution.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple


FUSION_POLICY_NONE = "none"
FUSION_POLICY_OBSERVE = "observe"
FUSION_POLICY_SCRIPTED = "scripted"
FUSION_POLICIES = {FUSION_POLICY_NONE, FUSION_POLICY_OBSERVE, FUSION_POLICY_SCRIPTED}
FUSION_POLICY_ALIASES = {
    "assist": FUSION_POLICY_SCRIPTED,
}

FUSION_REJECTION_REASONS = (
    "fusion_policy_none",
    "fusion_not_available",
    "fusion_not_unlocked",
    "source_not_found",
    "target_not_available",
    "incomplete_fusion_mapping",
    "not_gameplay",
    "gameplay_not_ready",
    "unsafe_state",
    "bridge_rejected",
    "low_strategic_value",
    "invalid_row_col",
    "terminal_or_transition_state",
    "seed_selection_active",
    "reward_or_unlock_screen_active",
    "action_space_mismatch",
    "exception",
)

PLANT_NAMES = {
    0: "Peashooter",
    1: "SunFlower",
    2: "CherryBomb",
    3: "WallNut",
    1030: "Repeater",
    1032: "GatlingPea",
    1033: "TwinSunFlower",
}

# A tiny first-pass allowlist.  Unknown mappings are observed but never executed.
FUSION_RULES: Dict[Tuple[int, int], Dict[str, Any]] = {
    (0, 0): {
        "predicted_result_name": "Repeater",
        "predicted_result_type": 1030,
        "reason": "Peashooter-on-Peashooter should improve lane DPS.",
        "role": "dps",
        "scripted_enabled": True,
    },
    (1, 1): {
        "predicted_result_name": "TwinSunFlower",
        "predicted_result_type": 1033,
        "reason": "SunFlower-on-SunFlower is known economy fusion but is not a defensive assist.",
        "role": "economy",
        "scripted_enabled": False,
    },
}

CONEHEAD_TYPES = {2, 12}
BUCKETHEAD_TYPES = {4, 13}


def normalize_fusion_policy(value: Any) -> str:
    policy = str(value or FUSION_POLICY_NONE).strip().lower()
    policy = FUSION_POLICY_ALIASES.get(policy, policy)
    if policy not in FUSION_POLICIES:
        raise ValueError(f"unknown fusion policy: {value!r}")
    return policy


def plant_name(plant_type: Any, fallback: str = "") -> str:
    try:
        plant_id = int(plant_type)
    except (TypeError, ValueError):
        return fallback or "unknown"
    return fallback or PLANT_NAMES.get(plant_id, str(plant_id))


def default_fusion_diagnostics(policy: str = FUSION_POLICY_NONE) -> Dict[str, Any]:
    policy = normalize_fusion_policy(policy)
    return {
        "fusion_policy": policy,
        "fusion_available": False,
        "fusion_candidate_count": 0,
        "fusion_candidates": [],
        "fusion_top_candidate": None,
        "fusion_last_attempt": None,
        "fusion_last_result": None,
        "fusion_last_execution_mode": "",
        "fusion_last_bridge_method_used": "",
        "fusion_last_bridge_result_reason": "",
        "fusion_last_duplicate_stack_detected": False,
        "last_fusion_scope": "",
        "last_fusion_changed_tile_count": 0,
        "last_fusion_non_source_tiles_changed": False,
        "last_fusion_global_side_effect": False,
        "last_executed_coach_command_id": None,
        "fusion_last_rejected_reason": "",
        "fusion_attempted_count": 0,
        "fusion_success_count": 0,
        "fusion_failed_count": 0,
        "fusion_rejected_count": 0,
        "fusion_rejected_reasons": {},
        "fusion_by_result_type": {},
        "fusion_by_source_type": {},
        "fusion_by_row": {},
        "fusion_under_threat_count": 0,
        "fusion_near_buckethead_count": 0,
        "fusion_near_conehead_count": 0,
        "fusion_estimated_mower_save_count": 0,
        "fusion_kills_after_use_total": 0,
        "fusion_avg_kills_after_use": 0.0,
        "fusion_bridge_error_count": 0,
        "fusion_unsafe_state_block_count": 0,
    }


def build_fusion_diagnostics(
    policy: str,
    observation: Dict[str, Any],
    *,
    bridge_probe: Optional[Dict[str, Any]] = None,
    bridge_error: Optional[str] = None,
) -> Dict[str, Any]:
    diag = default_fusion_diagnostics(policy)
    if diag["fusion_policy"] == FUSION_POLICY_NONE:
        return diag

    if bridge_error:
        diag["fusion_bridge_error_count"] = 1

    candidates = scan_fusion_candidates(observation, bridge_probe=bridge_probe)
    diag["fusion_candidates"] = candidates
    diag["fusion_candidate_count"] = len(candidates)
    diag["fusion_available"] = any(bool(candidate.get("fusion_legal")) for candidate in candidates)
    if candidates:
        top = max(candidates, key=lambda item: float(item.get("strategic_score") or 0.0))
        diag["fusion_top_candidate"] = compact_candidate(top)
    return diag


def scan_fusion_candidates(
    observation: Dict[str, Any],
    *,
    bridge_probe: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    plants = [plant for plant in observation.get("plants", []) or [] if isinstance(plant, dict)]
    slots = [slot for slot in observation.get("seedSlots", []) or [] if isinstance(slot, dict)]
    rows = _safe_int(observation.get("rowCount"), default=5)
    cols = _safe_int(observation.get("columnCount"), default=10)
    sun = _safe_int(observation.get("sun"), default=0)
    probe_candidates = _probe_candidate_index(bridge_probe)
    candidates: List[Dict[str, Any]] = []

    for source in plants:
        source_row = _safe_int(source.get("row"), default=-1)
        source_col = _safe_int(source.get("column"), default=-1)
        source_type = _safe_int(source.get("type"), default=-1)
        if not (0 <= source_row < rows and 0 <= source_col < cols):
            continue
        for slot in slots:
            ingredient_type = _safe_int(slot.get("plantType"), default=-1)
            slot_index = _safe_int(slot.get("slotIndex"), default=-1)
            if slot_index < 0 or ingredient_type < 0:
                continue
            rule = FUSION_RULES.get((source_type, ingredient_type))
            blocked_reason = "" if rule else "incomplete_fusion_mapping"
            fusion_legal = bool(rule)
            probe = _match_probe_candidate(
                probe_candidates,
                source,
                source_row,
                source_col,
                source_type,
                slot_index,
                ingredient_type,
            )
            if probe is not None:
                probe_legal = bool(
                    probe.get("fusionLegal")
                    if "fusionLegal" in probe
                    else probe.get("fusion_legal")
                )
                probe_reason = str(
                    probe.get("fusionBlockedReason")
                    or probe.get("fusion_blocked_reason")
                    or probe.get("blockedReason")
                    or ""
                )
                fusion_legal = bool(rule) and probe_legal
                if probe_reason:
                    blocked_reason = probe_reason
                elif not probe_legal:
                    blocked_reason = "bridge_rejected"
                elif not rule:
                    blocked_reason = "incomplete_fusion_mapping"
                else:
                    blocked_reason = ""
            elif not _slot_available(slot, sun):
                fusion_legal = False
                blocked_reason = "target_not_available"
            elif rule:
                fusion_legal = False
                blocked_reason = "bridge_rejected"

            lane = _lane_context(observation, source_row, source_col)
            health_ratio = _source_health_ratio(source)
            strategic_score, reason = _strategic_score(source_type, ingredient_type, rule, lane, health_ratio)
            if rule and not bool(rule.get("scripted_enabled")) and not blocked_reason:
                fusion_legal = False
                blocked_reason = "low_strategic_value"

            candidates.append(
                {
                    "source_plant_name": str(source.get("typeName") or plant_name(source_type)),
                    "source_plant_type": source_type,
                    "source_instance_id": _safe_int(source.get("instanceId"), source.get("instanceID"), default=0),
                    "source_row": source_row,
                    "source_col": source_col,
                    "target_or_ingredient_name": str(slot.get("plantTypeName") or plant_name(ingredient_type)),
                    "target_or_ingredient_type": ingredient_type,
                    "ingredient_seed_slot_index": slot_index,
                    "ingredient_card_instance_id": _safe_int(slot.get("cardInstanceId"), default=0),
                    "predicted_result_name": str((rule or {}).get("predicted_result_name") or "unknown"),
                    "predicted_result_type": _safe_int((rule or {}).get("predicted_result_type"), default=-1),
                    "fusion_legal": bool(fusion_legal and not blocked_reason),
                    "fusion_blocked_reason": blocked_reason,
                    "lane_danger_score": lane["lane_danger_score"],
                    "nearby_zombie_count": lane["nearby_zombie_count"],
                    "nearby_conehead_count": lane["nearby_conehead_count"],
                    "nearby_buckethead_count": lane["nearby_buckethead_count"],
                    "nearest_zombie_distance": lane["nearest_zombie_distance"],
                    "source_health_ratio": health_ratio,
                    "strategic_score": strategic_score,
                    "reason": reason,
                }
            )

    return sorted(candidates, key=lambda item: float(item.get("strategic_score") or 0.0), reverse=True)


def choose_scripted_fusion_candidate(diagnostics: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    policy = normalize_fusion_policy(diagnostics.get("fusion_policy"))
    if policy == FUSION_POLICY_NONE:
        return None, "fusion_policy_none"
    candidates = diagnostics.get("fusion_candidates")
    if not isinstance(candidates, list) or not candidates:
        return None, "fusion_not_available"

    best_rejection = "low_strategic_value"
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        blocked = str(candidate.get("fusion_blocked_reason") or "")
        if blocked:
            best_rejection = blocked
            continue
        if not bool(candidate.get("fusion_legal")):
            best_rejection = "bridge_rejected"
            continue
        if not _is_scripted_allowlisted(candidate):
            best_rejection = "incomplete_fusion_mapping"
            continue
        score = float(candidate.get("strategic_score") or 0.0)
        lane_danger = float(candidate.get("lane_danger_score") or 0.0)
        pressure = (
            int(candidate.get("nearby_zombie_count") or 0) >= 2
            or int(candidate.get("nearby_conehead_count") or 0) > 0
            or int(candidate.get("nearby_buckethead_count") or 0) > 0
        )
        if score >= 1.25 and (lane_danger >= 0.25 or pressure):
            return candidate, ""
        best_rejection = "low_strategic_value"
    return None, best_rejection


def validate_scripted_fusion_candidate(
    candidate: Optional[Dict[str, Any]],
    observation: Dict[str, Any],
    *,
    action_count: int,
    expected_action_count: int,
    reset_active: bool = False,
    action_already_executed: bool = False,
) -> str:
    if candidate is None:
        return "fusion_not_available"
    if action_count != expected_action_count:
        return "action_space_mismatch"
    if action_already_executed or reset_active:
        return "unsafe_state"
    if bool(observation.get("done")) or bool(observation.get("over")):
        return "terminal_or_transition_state"
    if bool(observation.get("seedSelectionActive")) or str(observation.get("screenState") or "") == "seed_selection":
        return "seed_selection_active"
    if bool(observation.get("blockingRewardUiActive")) or str(observation.get("screenState") or "") in {
        "reward_unlock",
        "reward_screen",
        "level_complete_trophy",
    }:
        return "reward_or_unlock_screen_active"
    if not bool(observation.get("boardFound", True)):
        return "not_gameplay"
    if not bool(observation.get("gameplayReady")):
        return "gameplay_not_ready"
    row = _safe_int(candidate.get("source_row"), default=-1)
    col = _safe_int(candidate.get("source_col"), default=-1)
    rows = _safe_int(observation.get("rowCount"), default=5)
    cols = _safe_int(observation.get("columnCount"), default=10)
    if not (0 <= row < rows and 0 <= col < cols):
        return "invalid_row_col"
    if not _source_exists(observation, candidate):
        return "source_not_found"
    if not _slot_available_for_candidate(observation, candidate):
        return "target_not_available"
    blocked = str(candidate.get("fusion_blocked_reason") or "")
    if blocked:
        return blocked if blocked in FUSION_REJECTION_REASONS else "bridge_rejected"
    if not bool(candidate.get("fusion_legal")):
        return "bridge_rejected"
    if not _is_scripted_allowlisted(candidate):
        return "incomplete_fusion_mapping"
    score = float(candidate.get("strategic_score") or 0.0)
    if score < 1.25:
        return "low_strategic_value"
    return ""


def compact_candidate(candidate: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(candidate, dict):
        return None
    keys = (
        "source_plant_name",
        "source_plant_type",
        "source_row",
        "source_col",
        "target_or_ingredient_name",
        "target_or_ingredient_type",
        "predicted_result_name",
        "predicted_result_type",
        "fusion_legal",
        "fusion_blocked_reason",
        "lane_danger_score",
        "nearby_zombie_count",
        "nearby_conehead_count",
        "nearby_buckethead_count",
        "nearest_zombie_distance",
        "strategic_score",
        "reason",
    )
    return {key: candidate.get(key) for key in keys if key in candidate}


def apply_fusion_attempt_result(
    diagnostics: Dict[str, Any],
    candidate: Optional[Dict[str, Any]],
    result: Optional[Dict[str, Any]],
    *,
    rejected_reason: str = "",
    bridge_error: Optional[str] = None,
) -> Dict[str, Any]:
    diag = dict(diagnostics)
    reasons = Counter(diag.get("fusion_rejected_reasons") or {})
    if bridge_error:
        diag["fusion_bridge_error_count"] = int(diag.get("fusion_bridge_error_count") or 0) + 1
        rejected_reason = rejected_reason or "exception"
    if rejected_reason:
        reasons[rejected_reason] += 1
        diag["fusion_rejected_count"] = int(diag.get("fusion_rejected_count") or 0) + 1
        diag["fusion_last_rejected_reason"] = rejected_reason
        if rejected_reason in {"unsafe_state", "not_gameplay", "gameplay_not_ready", "terminal_or_transition_state"}:
            diag["fusion_unsafe_state_block_count"] = int(diag.get("fusion_unsafe_state_block_count") or 0) + 1
    if result is not None:
        diag["fusion_attempted_count"] = int(diag.get("fusion_attempted_count") or 0) + 1
        success = bool(
            result.get("fusionSucceeded")
            if "fusionSucceeded" in result
            else result.get("fusion_success")
        )
        if success:
            diag["fusion_success_count"] = int(diag.get("fusion_success_count") or 0) + 1
            _bump_dict(diag, "fusion_by_result_type", str((candidate or {}).get("predicted_result_type", -1)))
            _bump_dict(diag, "fusion_by_source_type", str((candidate or {}).get("source_plant_type", -1)))
            _bump_dict(diag, "fusion_by_row", str((candidate or {}).get("source_row", -1)))
            if float((candidate or {}).get("lane_danger_score") or 0.0) > 0.0:
                diag["fusion_under_threat_count"] = int(diag.get("fusion_under_threat_count") or 0) + 1
            if int((candidate or {}).get("nearby_buckethead_count") or 0) > 0:
                diag["fusion_near_buckethead_count"] = int(diag.get("fusion_near_buckethead_count") or 0) + 1
            if int((candidate or {}).get("nearby_conehead_count") or 0) > 0:
                diag["fusion_near_conehead_count"] = int(diag.get("fusion_near_conehead_count") or 0) + 1
            if float((candidate or {}).get("lane_danger_score") or 0.0) >= 0.6:
                diag["fusion_estimated_mower_save_count"] = int(diag.get("fusion_estimated_mower_save_count") or 0) + 1
        else:
            diag["fusion_failed_count"] = int(diag.get("fusion_failed_count") or 0) + 1
            reason = str(result.get("fusionRejectedReason") or result.get("illegalReason") or "bridge_rejected")
            reasons[reason] += 1
            diag["fusion_last_rejected_reason"] = reason
        diag["fusion_last_attempt"] = compact_candidate(candidate)
        diag["fusion_last_result"] = {
            "success": success,
            "illegalReason": result.get("illegalReason"),
            "fusionRejectedReason": result.get("fusionRejectedReason"),
            "fusionExecutionMode": str(result.get("fusionExecutionMode") or ""),
            "bridgeMethodUsed": str(result.get("bridgeMethodUsed") or ""),
            "bridgeResultReason": str(result.get("bridgeResultReason") or ""),
            "duplicateStackDetected": bool(result.get("duplicateStackDetected")),
            "fusionScope": str(result.get("fusionScope") or result.get("fusion_scope") or ""),
            "changedTileCount": _safe_int(result.get("changedTileCount"), result.get("changed_tile_count"), default=0),
            "nonSourceTilesChanged": bool(
                result.get("nonSourceTilesChanged")
                if "nonSourceTilesChanged" in result
                else result.get("non_source_tiles_changed")
            ),
            "globalFusionSideEffect": bool(
                result.get("globalFusionSideEffect")
                if "globalFusionSideEffect" in result
                else result.get("global_fusion_side_effect")
            ),
            "plantCountOnTileBefore": _safe_int(result.get("plantCountOnTileBefore"), default=0),
            "plantCountOnTileAfter": _safe_int(result.get("plantCountOnTileAfter"), default=0),
            "sourceTileOccupiedBefore": bool(result.get("sourceTileOccupiedBefore")),
            "sourcePlantBefore": result.get("sourcePlantBefore"),
            "resultingPlantAfter": result.get("resultingPlantAfter"),
        }
        diag["fusion_last_execution_mode"] = str(result.get("fusionExecutionMode") or "")
        diag["fusion_last_bridge_method_used"] = str(result.get("bridgeMethodUsed") or "")
        diag["fusion_last_bridge_result_reason"] = str(result.get("bridgeResultReason") or "")
        diag["fusion_last_duplicate_stack_detected"] = bool(result.get("duplicateStackDetected"))
        diag["last_fusion_scope"] = str(result.get("fusionScope") or result.get("fusion_scope") or "")
        diag["last_fusion_changed_tile_count"] = _safe_int(result.get("changedTileCount"), result.get("changed_tile_count"), default=0)
        diag["last_fusion_non_source_tiles_changed"] = bool(
            result.get("nonSourceTilesChanged")
            if "nonSourceTilesChanged" in result
            else result.get("non_source_tiles_changed")
        )
        diag["last_fusion_global_side_effect"] = bool(
            result.get("globalFusionSideEffect")
            if "globalFusionSideEffect" in result
            else result.get("global_fusion_side_effect")
        )
        diag["last_executed_coach_command_id"] = result.get("last_executed_coach_command_id", result.get("executed_coach_command_id"))
    diag["fusion_rejected_reasons"] = dict(sorted(reasons.items()))
    return diag


def merge_episode_fusion_stats(target: Any, diagnostics: Dict[str, Any]) -> None:
    if not isinstance(diagnostics, dict):
        return
    for name in (
        "fusion_candidate_count",
        "fusion_attempted_count",
        "fusion_success_count",
        "fusion_failed_count",
        "fusion_rejected_count",
        "fusion_under_threat_count",
        "fusion_near_buckethead_count",
        "fusion_near_conehead_count",
        "fusion_estimated_mower_save_count",
        "fusion_kills_after_use_total",
        "fusion_bridge_error_count",
        "fusion_unsafe_state_block_count",
    ):
        value = int(diagnostics.get(name) or 0)
        if name == "fusion_candidate_count":
            total_name = "fusion_candidate_count_total"
            setattr(target, total_name, int(getattr(target, total_name, 0)) + value)
        else:
            setattr(target, name, int(getattr(target, name, 0)) + value)
    for name in ("fusion_rejected_reasons", "fusion_by_result_type", "fusion_by_source_type", "fusion_by_row"):
        current = getattr(target, name, {})
        if not isinstance(current, dict):
            current = {}
        for key, value in (diagnostics.get(name) or {}).items():
            current[str(key)] = int(current.get(str(key), 0)) + int(value)
        setattr(target, name, dict(sorted(current.items())))
    setattr(target, "fusion_policy", str(diagnostics.get("fusion_policy") or getattr(target, "fusion_policy", "none")))


def fusion_live_fields(diagnostics: Optional[Dict[str, Any]], policy: str = FUSION_POLICY_NONE) -> Dict[str, Any]:
    diag = diagnostics if isinstance(diagnostics, dict) else default_fusion_diagnostics(policy)
    return {
        "fusion_policy": diag.get("fusion_policy", policy),
        "fusion_available": bool(diag.get("fusion_available")),
        "fusion_candidate_count": int(diag.get("fusion_candidate_count") or 0),
        "fusion_top_candidate": diag.get("fusion_top_candidate"),
        "fusion_last_attempt": diag.get("fusion_last_attempt"),
        "fusion_last_result": diag.get("fusion_last_result"),
        "fusion_last_execution_mode": str(diag.get("fusion_last_execution_mode") or ""),
        "fusion_last_bridge_method_used": str(diag.get("fusion_last_bridge_method_used") or ""),
        "fusion_last_bridge_result_reason": str(diag.get("fusion_last_bridge_result_reason") or ""),
        "fusion_last_duplicate_stack_detected": bool(diag.get("fusion_last_duplicate_stack_detected")),
        "last_fusion_scope": str(diag.get("last_fusion_scope") or ""),
        "last_fusion_changed_tile_count": int(diag.get("last_fusion_changed_tile_count") or 0),
        "last_fusion_non_source_tiles_changed": bool(diag.get("last_fusion_non_source_tiles_changed")),
        "last_fusion_global_side_effect": bool(diag.get("last_fusion_global_side_effect")),
        "last_executed_coach_command_id": diag.get("last_executed_coach_command_id"),
        "fusion_last_rejected_reason": str(diag.get("fusion_last_rejected_reason") or ""),
        "fusion_attempted_count": int(diag.get("fusion_attempted_count") or 0),
        "fusion_success_count": int(diag.get("fusion_success_count") or 0),
        "fusion_failed_count": int(diag.get("fusion_failed_count") or 0),
        "fusion_rejected_count": int(diag.get("fusion_rejected_count") or 0),
        "fusion_rejected_reasons": dict(diag.get("fusion_rejected_reasons") or {}),
    }


def _probe_candidate_index(bridge_probe: Optional[Dict[str, Any]]) -> Dict[Tuple[int, int, int, int, int], Dict[str, Any]]:
    if not isinstance(bridge_probe, dict):
        return {}
    raw_candidates = bridge_probe.get("fusionCandidates") or bridge_probe.get("fusion_candidates") or []
    if not isinstance(raw_candidates, list):
        return {}
    index: Dict[Tuple[int, int, int, int, int], Dict[str, Any]] = {}
    for candidate in raw_candidates:
        if not isinstance(candidate, dict):
            continue
        row = _safe_int(candidate.get("sourceRow"), candidate.get("source_row"), default=-1)
        col = _safe_int(candidate.get("sourceCol"), candidate.get("sourceColumn"), candidate.get("source_col"), default=-1)
        source_type = _safe_int(candidate.get("sourcePlantType"), candidate.get("source_plant_type"), default=-1)
        slot_index = _safe_int(candidate.get("ingredientSeedSlotIndex"), candidate.get("seedSlotIndex"), candidate.get("ingredient_seed_slot_index"), default=-1)
        ingredient_type = _safe_int(candidate.get("ingredientPlantType"), candidate.get("targetPlantType"), candidate.get("ingredient_plant_type"), default=-1)
        index[(row, col, source_type, slot_index, ingredient_type)] = candidate
    return index


def _match_probe_candidate(
    index: Dict[Tuple[int, int, int, int, int], Dict[str, Any]],
    source: Dict[str, Any],
    row: int,
    col: int,
    source_type: int,
    slot_index: int,
    ingredient_type: int,
) -> Optional[Dict[str, Any]]:
    candidate = index.get((row, col, source_type, slot_index, ingredient_type))
    if candidate is not None:
        return candidate
    source_instance = _safe_int(source.get("instanceId"), source.get("instanceID"), default=0)
    for probe in index.values():
        if source_instance <= 0:
            continue
        probe_instance = _safe_int(probe.get("sourceInstanceId"), probe.get("source_instance_id"), default=0)
        if probe_instance == source_instance and _safe_int(probe.get("ingredientSeedSlotIndex"), probe.get("seedSlotIndex"), default=-2) == slot_index:
            return probe
    return None


def _slot_available(slot: Dict[str, Any], sun: int) -> bool:
    cost = _safe_int(slot.get("seedCost"), default=0)
    return bool(slot.get("usable", True)) and not bool(slot.get("disabled")) and bool(slot.get("ready")) and sun >= cost


def _slot_available_for_candidate(observation: Dict[str, Any], candidate: Dict[str, Any]) -> bool:
    slot_index = _safe_int(candidate.get("ingredient_seed_slot_index"), default=-1)
    ingredient_type = _safe_int(candidate.get("target_or_ingredient_type"), default=-1)
    sun = _safe_int(observation.get("sun"), default=0)
    for slot in observation.get("seedSlots", []) or []:
        if not isinstance(slot, dict):
            continue
        if _safe_int(slot.get("slotIndex"), default=-2) != slot_index:
            continue
        if _safe_int(slot.get("plantType"), default=-3) != ingredient_type:
            return False
        return _slot_available(slot, sun)
    return False


def _source_exists(observation: Dict[str, Any], candidate: Dict[str, Any]) -> bool:
    row = _safe_int(candidate.get("source_row"), default=-1)
    col = _safe_int(candidate.get("source_col"), default=-1)
    source_type = _safe_int(candidate.get("source_plant_type"), default=-1)
    source_instance = _safe_int(candidate.get("source_instance_id"), default=0)
    for plant in observation.get("plants", []) or []:
        if not isinstance(plant, dict):
            continue
        if _safe_int(plant.get("row"), default=-2) != row or _safe_int(plant.get("column"), default=-2) != col:
            continue
        if _safe_int(plant.get("type"), default=-3) != source_type:
            continue
        plant_instance = _safe_int(plant.get("instanceId"), plant.get("instanceID"), default=0)
        if source_instance > 0 and plant_instance > 0 and plant_instance != source_instance:
            continue
        return True
    return False


def _is_scripted_allowlisted(candidate: Dict[str, Any]) -> bool:
    key = (
        _safe_int(candidate.get("source_plant_type"), default=-1),
        _safe_int(candidate.get("target_or_ingredient_type"), default=-1),
    )
    return bool(FUSION_RULES.get(key, {}).get("scripted_enabled"))


def _lane_context(observation: Dict[str, Any], source_row: int, source_col: int) -> Dict[str, Any]:
    lane = {}
    for item in observation.get("lanes", []) or []:
        if isinstance(item, dict) and _safe_int(item.get("row"), default=-1) == source_row:
            lane = item
            break
    lane_danger = _safe_float(lane.get("toughZombiePressureScore"), lane.get("danger"), default=-1.0)
    nearest_x = _safe_float(lane.get("nearestZombieX"), default=99.0)
    if lane_danger < 0.0:
        lane_danger = max(0.0, 1.0 - nearest_x / 10.0) if nearest_x < 99.0 else 0.0
    zombies = [
        zombie for zombie in observation.get("zombies", []) or []
        if isinstance(zombie, dict)
        and _safe_int(zombie.get("row"), default=-1) == source_row
        and bool(zombie.get("alive", True))
    ]
    nearby = []
    for zombie in zombies:
        zx = _safe_float(zombie.get("x"), default=nearest_x)
        if zx >= float(source_col) - 0.5 and zx - float(source_col) <= 5.0:
            nearby.append(zombie)
    if not nearby and zombies:
        nearby = zombies
    coneheads = sum(1 for zombie in nearby if is_conehead_zombie(zombie))
    bucketheads = sum(1 for zombie in nearby if is_buckethead_zombie(zombie))
    return {
        "lane_danger_score": round(float(lane_danger), 4),
        "nearby_zombie_count": len(nearby),
        "nearby_conehead_count": coneheads,
        "nearby_buckethead_count": bucketheads,
        "nearest_zombie_distance": round(max(0.0, nearest_x - float(source_col)), 4) if nearest_x < 99.0 else None,
    }


def _strategic_score(
    source_type: int,
    ingredient_type: int,
    rule: Optional[Dict[str, Any]],
    lane: Dict[str, Any],
    health_ratio: float,
) -> Tuple[float, str]:
    del source_type, ingredient_type
    score = float(lane.get("lane_danger_score") or 0.0)
    score += min(1.0, 0.25 * int(lane.get("nearby_zombie_count") or 0))
    score += 0.55 * int(lane.get("nearby_conehead_count") or 0)
    score += 0.8 * int(lane.get("nearby_buckethead_count") or 0)
    distance = lane.get("nearest_zombie_distance")
    if isinstance(distance, (int, float)) and float(distance) <= 4.0:
        score += 0.35
    if health_ratio < 0.5:
        score += 0.15
    if not rule:
        return round(score, 4), "mapping unknown; observe only"
    role = str(rule.get("role") or "")
    if role == "dps":
        score += 0.35
    elif role == "economy":
        score -= 0.5
    return round(max(0.0, score), 4), str(rule.get("reason") or "known fusion mapping")


def _source_health_ratio(source: Dict[str, Any]) -> float:
    max_health = max(1.0, _safe_float(source.get("maxHealth"), default=1.0))
    return round(max(0.0, min(1.0, _safe_float(source.get("health"), default=max_health) / max_health)), 4)


def is_conehead_zombie(zombie: Dict[str, Any]) -> bool:
    zombie_type = _safe_int(zombie.get("type"), default=-1)
    name = str(zombie.get("typeName") or "").lower()
    return zombie_type in CONEHEAD_TYPES or "cone" in name or "roadblock" in name or "路障" in name


def is_buckethead_zombie(zombie: Dict[str, Any]) -> bool:
    zombie_type = _safe_int(zombie.get("type"), default=-1)
    name = str(zombie.get("typeName") or "").lower()
    return zombie_type in BUCKETHEAD_TYPES or "bucket" in name or "铁桶" in name


def is_tough_zombie(zombie: Dict[str, Any]) -> bool:
    return (
        is_conehead_zombie(zombie)
        or is_buckethead_zombie(zombie)
        or _safe_int(zombie.get("health"), default=0) >= 600
        or _safe_int(zombie.get("maxHealth"), default=0) >= 600
    )


def count_tough_zombies_by_row(observation: Dict[str, Any]) -> Dict[int, Dict[str, int]]:
    rows = _safe_int(observation.get("rowCount"), default=5)
    counts = {
        row: {"buckethead": 0, "conehead": 0, "tough": 0}
        for row in range(max(0, rows))
    }
    for lane in observation.get("lanes", []) or []:
        if not isinstance(lane, dict):
            continue
        row = _safe_int(lane.get("row"), default=-1)
        if row not in counts:
            continue
        counts[row]["buckethead"] = _safe_int(lane.get("bucketheadCount"), lane.get("buckethead_count"), default=counts[row]["buckethead"])
        counts[row]["conehead"] = _safe_int(lane.get("coneheadCount"), lane.get("conehead_count"), default=counts[row]["conehead"])
        counts[row]["tough"] = _safe_int(lane.get("toughZombieCount"), lane.get("tough_zombie_count"), default=counts[row]["tough"])
    for zombie in observation.get("zombies", []) or []:
        if not isinstance(zombie, dict) or not bool(zombie.get("alive", True)):
            continue
        row = _safe_int(zombie.get("row"), default=-1)
        if row not in counts:
            continue
        if is_buckethead_zombie(zombie):
            counts[row]["buckethead"] += 1
        if is_conehead_zombie(zombie):
            counts[row]["conehead"] += 1
        if is_tough_zombie(zombie):
            counts[row]["tough"] += 1
    return counts


def _bump_dict(payload: Dict[str, Any], field: str, key: str) -> None:
    values = dict(payload.get(field) or {})
    values[str(key)] = int(values.get(str(key), 0)) + 1
    payload[field] = dict(sorted(values.items()))


def _safe_int(*values: Any, default: int = 0) -> int:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return int(default)


def _safe_float(*values: Any, default: float = 0.0) -> float:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return float(default)
