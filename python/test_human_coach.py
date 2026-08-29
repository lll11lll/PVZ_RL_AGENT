"""Bridge-free tests for local human coach command handling."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from pvzrl_action_space import (
    ACTION_SPACE_ADVENTURE_14_IDENTITY,
    ADVENTURE_IDENTITY_ACTION_COUNT,
    ADVENTURE_IDENTITY_MAX_SEED_SLOTS,
    CELLS_PER_SLOT,
    DEFAULT_COLS,
    DEFAULT_ROWS,
)
from pvzrl_human_coach import (
    COACH_REWARD_FUSION_SUCCESS_COMPONENT,
    COACH_REWARD_LEGAL_EXECUTION_COMPONENT,
    COACH_REWARD_MATCH_COMPONENT,
    COACH_REWARD_OVERRIDE_PENALTY_COMPONENT,
    COACH_REWARD_TACTICAL_USEFULNESS_COMPONENT,
    FileCoachCommandSource,
    HumanCoachOverrideHook,
    QueueCoachCommandSource,
    command_to_policy_action,
    parse_coach_command,
    validate_coach_command,
)


FIVE_LANE_LIVE_ROWS = DEFAULT_ROWS - 1


def seed_slot(index: int, plant_type: int, name: str) -> Dict[str, Any]:
    return {
        "slotIndex": index,
        "plantType": plant_type,
        "plantTypeName": name,
        "ready": True,
        "usable": True,
        "seedCost": 50,
    }


def observation(
    legal_actions: List[int],
    include_fusion_board: bool = False,
    sun: int = 500,
    *,
    live_rows: int = FIVE_LANE_LIVE_ROWS,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        # Keep explicit five-lane coverage: model actions remain padded to six
        # rows, while live legality masks the absent sixth lane.
        "rowCount": int(live_rows),
        "columnCount": DEFAULT_COLS,
        "gameplayReady": True,
        "boardFound": True,
        "canReadBoard": True,
        "sun": int(sun),
        "seedSlots": [
            seed_slot(0, 1, "SunFlower"),
            seed_slot(1, 0, "Peashooter"),
            seed_slot(2, 3, "WallNut"),
            seed_slot(3, 2, "CherryBomb"),
        ],
        "legalActions": legal_actions,
    }
    if include_fusion_board:
        payload["plants"] = [
            {
                "instanceId": 111,
                "row": 1,
                "column": 1,
                "type": 0,
                "typeName": "Peashooter",
                "alive": True,
            }
        ]
        payload["lanes"] = [
            {
                "row": 1,
                "danger": 0.72,
                "toughZombiePressureScore": 0.72,
                "nearestZombieX": 3.1,
            }
        ]
        payload["zombies"] = [
            {
                "row": 1,
                "x": 3.0,
                "type": 4,
                "typeName": "BucketheadZombie",
                "alive": True,
            }
        ]
    return payload


def observation_duplicate_sunflower_fusion(legal_actions: List[int], sun: int = 500) -> Dict[str, Any]:
    payload = observation(legal_actions, include_fusion_board=True, sun=sun)
    payload["seedSlots"] = [
        seed_slot(0, 1, "SunFlower"),
        seed_slot(1, 1, "SunFlower"),
        seed_slot(2, 0, "Peashooter"),
        seed_slot(3, 0, "Peashooter"),
    ]
    return payload


class FakeActionSpec:
    mode = ACTION_SPACE_ADVENTURE_14_IDENTITY
    max_seed_slots = ADVENTURE_IDENTITY_MAX_SEED_SLOTS
    rows = DEFAULT_ROWS
    cols = DEFAULT_COLS


class FakeConfig:
    def __init__(self, *, fusion_policy: str = "none", human_coach_fusion_enabled: bool = False) -> None:
        self.plant_types = [1, 0, 3, 2]
        self.fusion_policy = str(fusion_policy)
        self.human_coach_enabled = True
        self.human_coach_platform = "mock"
        self.human_coach_fusion_enabled = bool(human_coach_fusion_enabled)


class FakeBridgeClient:
    def __init__(self, *, probe_payload: Optional[Dict[str, Any]] = None, raise_probe: bool = False) -> None:
        self.probe_payload = probe_payload if probe_payload is not None else {}
        self.raise_probe = bool(raise_probe)
        self.requests: List[Dict[str, Any]] = []

    def request(self, command: str, **payload: Any) -> Dict[str, Any]:
        self.requests.append({"command": str(command), "payload": dict(payload)})
        if str(command) != "fusion_probe":
            return {}
        if self.raise_probe:
            raise RuntimeError("probe_failed")
        return dict(self.probe_payload)


class FakeBase:
    def __init__(self, client: Optional[FakeBridgeClient]) -> None:
        self.client = client


class FakeEnv:
    def __init__(
        self,
        obs: Dict[str, Any],
        action_mask: List[bool],
        *,
        fusion_policy: str = "none",
        human_coach_fusion_enabled: bool = False,
        bridge_client: Optional[FakeBridgeClient] = None,
        fusion_step_success: bool = True,
        fusion_execution_mode: str = "dedicated_fusion",
        fusion_bridge_method_used: str = "Mouse.TryToSetPlantByCard",
        fusion_bridge_result_reason: str = "success",
        fusion_duplicate_stack_detected: bool = False,
        fusion_source_tile_occupied_before: bool = True,
        fusion_plant_count_on_tile_before: int = 1,
        fusion_plant_count_on_tile_after: int = 1,
        fusion_predicted_result_resolution_source: str = "checkmix_object.resultPlantType",
        fusion_mix_lookup_found: bool = True,
        fusion_mix_lookup_key: str = "0+1",
        fusion_pre_source_name: str = "Peashooter",
        fusion_ingredient_name: str = "SunFlower",
        fusion_post_result_name: str = "DoubleShooer",
        fusion_no_effect_reason: str = "",
        fusion_changed_tile_count: int = 1,
        fusion_changed_tiles: Optional[List[Dict[str, Any]]] = None,
        fusion_non_source_tiles_changed: bool = False,
        fusion_global_fusion_side_effect: bool = False,
        fusion_scope: str = "tile_scoped",
        tactical_signal: bool = False,
    ) -> None:
        self.action_spec = FakeActionSpec()
        self.config = FakeConfig(
            fusion_policy=fusion_policy,
            human_coach_fusion_enabled=human_coach_fusion_enabled,
        )
        self.rows = int(obs.get("rowCount", DEFAULT_ROWS) or DEFAULT_ROWS)
        self.cols = DEFAULT_COLS
        self.base = FakeBase(bridge_client)
        self._last_observation = obs
        self._action_mask = action_mask
        self.executed_actions: List[int] = []
        self.executed_bridge_commands: List[Optional[Dict[str, Any]]] = []
        self.coach_contexts: List[Optional[Dict[str, Any]]] = []
        self.fusion_step_success = bool(fusion_step_success)
        self.fusion_execution_mode = str(fusion_execution_mode)
        self.fusion_bridge_method_used = str(fusion_bridge_method_used)
        self.fusion_bridge_result_reason = str(fusion_bridge_result_reason)
        self.fusion_duplicate_stack_detected = bool(fusion_duplicate_stack_detected)
        self.fusion_source_tile_occupied_before = bool(fusion_source_tile_occupied_before)
        self.fusion_plant_count_on_tile_before = int(fusion_plant_count_on_tile_before)
        self.fusion_plant_count_on_tile_after = int(fusion_plant_count_on_tile_after)
        self.fusion_predicted_result_resolution_source = str(fusion_predicted_result_resolution_source)
        self.fusion_mix_lookup_found = bool(fusion_mix_lookup_found)
        self.fusion_mix_lookup_key = str(fusion_mix_lookup_key)
        self.fusion_pre_source_name = str(fusion_pre_source_name)
        self.fusion_ingredient_name = str(fusion_ingredient_name)
        self.fusion_post_result_name = str(fusion_post_result_name)
        self.fusion_no_effect_reason = str(fusion_no_effect_reason)
        self.fusion_changed_tile_count = int(fusion_changed_tile_count)
        self.fusion_changed_tiles = (
            [dict(item) for item in fusion_changed_tiles]
            if isinstance(fusion_changed_tiles, list)
            else None
        )
        self.fusion_non_source_tiles_changed = bool(fusion_non_source_tiles_changed)
        self.fusion_global_fusion_side_effect = bool(fusion_global_fusion_side_effect)
        self.fusion_scope = str(fusion_scope)
        self.tactical_signal = bool(tactical_signal)

    def action_masks(self) -> List[bool]:
        return list(self._action_mask)

    def step(
        self,
        action: int,
        *,
        coach_bridge_command: Optional[Dict[str, Any]] = None,
        coach_context: Optional[Dict[str, Any]] = None,
    ):
        self.executed_actions.append(int(action))
        self.executed_bridge_commands.append(dict(coach_bridge_command) if isinstance(coach_bridge_command, dict) else None)
        self.coach_contexts.append(dict(coach_context) if isinstance(coach_context, dict) else None)

        action_result: Dict[str, Any] = {
            "requestedAction": int(action),
            "executedAction": int(action),
            "illegalAction": False,
            "illegalReason": None,
            "plantPlaced": False,
            "costPaid": False,
            "cooldownStarted": False,
            "observation": dict(self._last_observation),
        }
        lane_diag: Dict[str, Any] = {
            "lane_response_reward_applied": False,
            "mower_saves_estimated_by_row": {},
        }
        if isinstance(coach_bridge_command, dict) and str(coach_bridge_command.get("command") or "") == "fusion_step":
            fusion_success = bool(self.fusion_step_success)
            fusion_illegal_reason = None
            if not fusion_success:
                if not self.fusion_source_tile_occupied_before:
                    fusion_illegal_reason = "source_tile_not_occupied"
                elif (
                    self.fusion_global_fusion_side_effect
                    or self.fusion_non_source_tiles_changed
                    or self.fusion_changed_tile_count > 1
                ):
                    fusion_illegal_reason = "global_fusion_side_effect"
                elif self.fusion_duplicate_stack_detected:
                    fusion_illegal_reason = "duplicate_stack_detected"
                else:
                    fusion_illegal_reason = "bridge_rejected"
            bridge_reason = (
                str(self.fusion_bridge_result_reason)
                if fusion_success and self.fusion_bridge_result_reason
                else str(fusion_illegal_reason or self.fusion_bridge_result_reason or "bridge_rejected")
            )
            source_row = int(coach_bridge_command.get("source_row", -1))
            source_col = int(coach_bridge_command.get("source_col", -1))
            source_type = int(coach_bridge_command.get("source_plant_type", -1))
            predicted_type = int(coach_bridge_command.get("predicted_result_type", -1))
            coach_command_id = int(coach_bridge_command.get("coach_command_id", 0) or 0)
            source_before = {
                "instanceId": int(coach_bridge_command.get("source_instance_id", 0) or 0),
                "plantType": source_type,
                "plantTypeName": self.fusion_pre_source_name if self.fusion_pre_source_name else ("SunFlower" if source_type == 1 else str(source_type)),
                "row": source_row,
                "column": source_col,
                "source": "logical",
            }
            resulting_after = {
                "instanceId": int(coach_bridge_command.get("source_instance_id", 0) or 0),
                "plantType": predicted_type if predicted_type >= 0 else source_type,
                "plantTypeName": self.fusion_post_result_name if self.fusion_post_result_name else ("TwinFlower" if predicted_type == 1033 else str(predicted_type if predicted_type >= 0 else source_type)),
                "row": source_row,
                "column": source_col,
                "source": "logical",
            }
            changed_tiles = (
                [dict(item) for item in self.fusion_changed_tiles]
                if isinstance(self.fusion_changed_tiles, list)
                else [
                    {
                        "row": source_row,
                        "column": source_col,
                        "beforePlantCount": self.fusion_plant_count_on_tile_before,
                        "afterPlantCount": self.fusion_plant_count_on_tile_after,
                        "beforePlants": [dict(source_before)],
                        "afterPlants": [dict(resulting_after)],
                    }
                ]
            )
            changed_tile_count = int(self.fusion_changed_tile_count)
            non_source_tiles_changed = bool(self.fusion_non_source_tiles_changed)
            global_fusion_side_effect = bool(self.fusion_global_fusion_side_effect)
            fusion_scope = str(self.fusion_scope)
            ingredient_type = int(coach_bridge_command.get("ingredient_plant_type", -1))
            placement = {
                "success": fusion_success,
                "plantPlaced": True,
                "costPaid": fusion_success,
                "cooldownStarted": fusion_success,
                "illegalReason": fusion_illegal_reason,
                "fusionExecutionMode": self.fusion_execution_mode,
                "sourceTileOccupiedBefore": self.fusion_source_tile_occupied_before,
                "plantCountOnTileBefore": self.fusion_plant_count_on_tile_before,
                "plantCountOnTileAfter": self.fusion_plant_count_on_tile_after,
                "sourcePlantBefore": source_before,
                "resultingPlantAfter": resulting_after,
                "duplicateStackDetected": self.fusion_duplicate_stack_detected,
                "bridgeMethodUsed": self.fusion_bridge_method_used,
                "bridgeResultReason": bridge_reason,
                "predictedResultResolutionSource": self.fusion_predicted_result_resolution_source,
                "mixLookupFound": self.fusion_mix_lookup_found,
                "mixLookupKey": self.fusion_mix_lookup_key,
                "preSourceType": source_type,
                "preSourceName": source_before.get("plantTypeName"),
                "ingredientType": ingredient_type,
                "ingredientName": self.fusion_ingredient_name,
                "postResultType": resulting_after.get("plantType"),
                "postResultName": resulting_after.get("plantTypeName"),
                "noEffectReason": self.fusion_no_effect_reason,
                "requestedSourceRow": source_row,
                "requestedSourceCol": source_col,
                "requestedSourceInstanceId": int(coach_bridge_command.get("source_instance_id", 0) or 0),
                "changedTileCount": changed_tile_count,
                "changedTiles": changed_tiles,
                "nonSourceTilesChanged": non_source_tiles_changed,
                "globalFusionSideEffect": global_fusion_side_effect,
                "fusionScope": fusion_scope,
                "requested_source_row": source_row,
                "requested_source_col": source_col,
                "requested_source_instance_id": int(coach_bridge_command.get("source_instance_id", 0) or 0),
                "changed_tile_count": changed_tile_count,
                "changed_tiles": changed_tiles,
                "non_source_tiles_changed": non_source_tiles_changed,
                "global_fusion_side_effect": global_fusion_side_effect,
                "fusion_scope": fusion_scope,
                "bridge_method_used": self.fusion_bridge_method_used,
                "bridge_result_reason": bridge_reason,
                "executed_from_fresh_coach_command": bool(coach_bridge_command.get("executed_from_fresh_coach_command")),
                "coach_command_age_seconds": 0.0,
                "startup_command_blocked": False,
                "coach_command_queue_cleared_on_reset": True,
                "executed_coach_command_id": coach_command_id,
                "last_executed_coach_command_id": coach_command_id,
            }
            action_result.update(
                {
                    "fusionAttempted": True,
                    "fusionSucceeded": fusion_success,
                    "fusionOverrideApplied": fusion_success,
                    "plantPlaced": True,
                    "illegalAction": not fusion_success,
                    "illegalReason": fusion_illegal_reason,
                    "fusionRejectedReason": fusion_illegal_reason,
                    "fusionExecutionMode": self.fusion_execution_mode,
                    "sourceTileOccupiedBefore": self.fusion_source_tile_occupied_before,
                    "plantCountOnTileBefore": self.fusion_plant_count_on_tile_before,
                    "plantCountOnTileAfter": self.fusion_plant_count_on_tile_after,
                    "sourcePlantBefore": source_before,
                    "resultingPlantAfter": resulting_after,
                    "duplicateStackDetected": self.fusion_duplicate_stack_detected,
                    "bridgeMethodUsed": self.fusion_bridge_method_used,
                    "bridgeResultReason": bridge_reason,
                    "predictedResultResolutionSource": self.fusion_predicted_result_resolution_source,
                    "mixLookupFound": self.fusion_mix_lookup_found,
                    "mixLookupKey": self.fusion_mix_lookup_key,
                    "preSourceType": source_type,
                    "preSourceName": source_before.get("plantTypeName"),
                    "ingredientType": ingredient_type,
                    "ingredientName": self.fusion_ingredient_name,
                    "postResultType": resulting_after.get("plantType"),
                    "postResultName": resulting_after.get("plantTypeName"),
                    "noEffectReason": self.fusion_no_effect_reason,
                    "requestedSourceRow": source_row,
                    "requestedSourceCol": source_col,
                    "requestedSourceInstanceId": int(coach_bridge_command.get("source_instance_id", 0) or 0),
                    "changedTileCount": changed_tile_count,
                    "changedTiles": changed_tiles,
                    "nonSourceTilesChanged": non_source_tiles_changed,
                    "globalFusionSideEffect": global_fusion_side_effect,
                    "fusionScope": fusion_scope,
                    "requested_source_row": source_row,
                    "requested_source_col": source_col,
                    "requested_source_instance_id": int(coach_bridge_command.get("source_instance_id", 0) or 0),
                    "changed_tile_count": changed_tile_count,
                    "changed_tiles": changed_tiles,
                    "non_source_tiles_changed": non_source_tiles_changed,
                    "global_fusion_side_effect": global_fusion_side_effect,
                    "fusion_scope": fusion_scope,
                    "bridge_method_used": self.fusion_bridge_method_used,
                    "bridge_result_reason": bridge_reason,
                    "executed_from_fresh_coach_command": bool(coach_bridge_command.get("executed_from_fresh_coach_command")),
                    "coach_command_age_seconds": 0.0,
                    "startup_command_blocked": False,
                    "coach_command_queue_cleared_on_reset": True,
                    "executed_coach_command_id": coach_command_id,
                    "last_executed_coach_command_id": coach_command_id,
                    "placement": placement,
                    "decoded": {
                        "kind": "fusion",
                        "row": source_row,
                        "column": source_col,
                        "plant_type": predicted_type,
                    },
                }
            )
            lane_diag["lane_response_reward_applied"] = bool(self.tactical_signal)
            if self.tactical_signal:
                lane_diag["mower_saves_estimated_by_row"] = {"1": 1}
        else:
            if int(action) != 0:
                action_result["plantPlaced"] = True
                action_result["decoded"] = {"kind": "plant", "row": 2, "column": 4, "plant_type": 1}
            else:
                action_result["decoded"] = {"kind": "wait", "row": -1, "column": -1, "plant_type": -1}

        info = {
            "raw_observation": dict(self._last_observation),
            "action_result": action_result,
            "reward_breakdown": {"reward_total": 1.0},
            "lane_diagnostics": lane_diag,
        }
        return "obs", 1.0, False, False, info


def mask_with(*actions: int) -> List[bool]:
    mask = [False] * ADVENTURE_IDENTITY_ACTION_COUNT
    for action in actions:
        mask[int(action)] = True
    return mask


def assert_case(results: List[Dict[str, Any]], name: str, condition: bool, detail: Any = None) -> None:
    results.append({"case": name, "passed": bool(condition), "detail": detail})


def fusion_probe_payload() -> Dict[str, Any]:
    return {
        "fusionCandidates": [
            {
                "sourceRow": 1,
                "sourceCol": 1,
                "sourcePlantType": 0,
                "sourceInstanceId": 111,
                "ingredientSeedSlotIndex": 0,
                "ingredientPlantType": 1,
                "fusionLegal": True,
                "predictedResultType": 1030,
                "predictedResultName": "DoubleShooer",
                "predictedResultResolutionSource": "checkmix_object.resultPlantType",
                "mixLookupFound": True,
                "mixLookupKey": "0+1",
                "strategicScore": 1.9,
            }
        ]
    }


def fusion_probe_payload_slot_index(slot_index: int) -> Dict[str, Any]:
    return {
        "fusionCandidates": [
            {
                "sourceRow": 1,
                "sourceCol": 1,
                "sourcePlantType": 0,
                "sourceInstanceId": 111,
                "ingredientSeedSlotIndex": int(slot_index),
                "ingredientPlantType": 1,
                "fusionLegal": True,
                "predictedResultType": 1030,
                "predictedResultName": "DoubleShooer",
                "predictedResultResolutionSource": "checkmix_object.resultPlantType",
                "mixLookupFound": True,
                "mixLookupKey": "0+1",
                "strategicScore": 1.9,
            }
        ]
    }


def fusion_probe_payload_duplicate_slot_fallback() -> Dict[str, Any]:
    return {
        "fusionCandidates": [
            {
                "sourceRow": 1,
                "sourceCol": 1,
                "sourcePlantType": 0,
                "sourceInstanceId": 111,
                "ingredientSeedSlotIndex": 0,
                "ingredientPlantType": 1,
                "fusionLegal": False,
                "fusionBlockedReason": "bridge_rejected",
                "predictedResultType": 1030,
                "predictedResultName": "DoubleShooer",
                "predictedResultResolutionSource": "checkmix_object.resultPlantType",
                "mixLookupFound": True,
                "mixLookupKey": "0+1",
                "strategicScore": 1.1,
            },
            {
                "sourceRow": 1,
                "sourceCol": 1,
                "sourcePlantType": 0,
                "sourceInstanceId": 111,
                "ingredientSeedSlotIndex": 1,
                "ingredientPlantType": 1,
                "fusionLegal": True,
                "predictedResultType": 1030,
                "predictedResultName": "DoubleShooer",
                "predictedResultResolutionSource": "checkmix_object.resultPlantType",
                "mixLookupFound": True,
                "mixLookupKey": "0+1",
                "strategicScore": 1.9,
            },
        ]
    }


def fusion_probe_payload_duplicate_slot_all_illegal() -> Dict[str, Any]:
    return {
        "fusionCandidates": [
            {
                "sourceRow": 1,
                "sourceCol": 1,
                "sourcePlantType": 0,
                "sourceInstanceId": 111,
                "ingredientSeedSlotIndex": 0,
                "ingredientPlantType": 1,
                "fusionLegal": False,
                "fusionBlockedReason": "bridge_rejected",
                "predictedResultType": 1030,
                "predictedResultName": "DoubleShooer",
                "predictedResultResolutionSource": "checkmix_object.resultPlantType",
                "mixLookupFound": True,
                "mixLookupKey": "0+1",
                "strategicScore": 1.0,
            },
            {
                "sourceRow": 1,
                "sourceCol": 1,
                "sourcePlantType": 0,
                "sourceInstanceId": 111,
                "ingredientSeedSlotIndex": 1,
                "ingredientPlantType": 1,
                "fusionLegal": False,
                "fusionBlockedReason": "bridge_rejected",
                "predictedResultType": 1030,
                "predictedResultName": "DoubleShooer",
                "predictedResultResolutionSource": "checkmix_object.resultPlantType",
                "mixLookupFound": True,
                "mixLookupKey": "0+1",
                "strategicScore": 1.0,
            },
        ]
    }


def main() -> int:
    results: List[Dict[str, Any]] = []

    plant = parse_coach_command("!plant 0 2 4", timestamp=1.0)
    assert_case(
        results,
        "parse !plant 0 2 4",
        plant.valid_syntax and plant.kind == "plant" and plant.seed_index == 0 and plant.row == 2 and plant.col == 4,
        plant.to_dict(),
    )

    fuse = parse_coach_command("!fuse 0 1 1", timestamp=1.0)
    assert_case(
        results,
        "parse !fuse 0 1 1",
        fuse.valid_syntax and fuse.kind == "fuse" and fuse.seed_index == 0 and fuse.row == 1 and fuse.col == 1,
        fuse.to_dict(),
    )
    plain_plant = parse_coach_command("plant 0 2 4", timestamp=1.0)
    paren_plant = parse_coach_command("plant(0,2,4)", timestamp=1.0)
    plain_wait = parse_coach_command("wait", timestamp=1.0)
    plain_defend = parse_coach_command("defend 3", timestamp=1.0)
    plain_economy = parse_coach_command("economy", timestamp=1.0)
    assert_case(
        results,
        "parse plant 0 2 4",
        plain_plant.valid_syntax and plain_plant.kind == "plant" and plain_plant.seed_index == 0 and plain_plant.row == 2 and plain_plant.col == 4,
        plain_plant.to_dict(),
    )
    assert_case(
        results,
        "parse plant(0,2,4)",
        paren_plant.valid_syntax and paren_plant.kind == "plant" and paren_plant.seed_index == 0 and paren_plant.row == 2 and paren_plant.col == 4,
        paren_plant.to_dict(),
    )
    assert_case(results, "parse wait", plain_wait.valid_syntax and plain_wait.kind == "wait", plain_wait.to_dict())
    assert_case(results, "parse defend 3", plain_defend.valid_syntax and plain_defend.kind == "defend" and plain_defend.row == 3, plain_defend.to_dict())
    assert_case(results, "parse economy", plain_economy.valid_syntax and plain_economy.kind == "economy", plain_economy.to_dict())

    wait = parse_coach_command("!wait", timestamp=1.0)
    defend = parse_coach_command("!defend 3", timestamp=1.0)
    economy = parse_coach_command("!economy", timestamp=1.0)
    assert_case(results, "parse !wait", wait.valid_syntax and wait.kind == "wait", wait.to_dict())
    assert_case(results, "parse !defend 3", defend.valid_syntax and defend.kind == "defend" and defend.row == 3, defend.to_dict())
    assert_case(results, "parse !economy", economy.valid_syntax and economy.kind == "economy", economy.to_dict())

    malformed = parse_coach_command("!plant 0 2")
    assert_case(
        results,
        "reject malformed command",
        not malformed.valid_syntax and malformed.rejected_reason == "plant_expects_seed_row_col",
        malformed.to_dict(),
    )

    action, reason = command_to_policy_action(
        plant,
        action_space_mode=ACTION_SPACE_ADVENTURE_14_IDENTITY,
        rows=FIVE_LANE_LIVE_ROWS,
        cols=DEFAULT_COLS,
        max_seed_slots=ADVENTURE_IDENTITY_MAX_SEED_SLOTS,
        plant_types=[1, 0, 3, 2],
    )
    assert_case(
        results,
        "convert plant to Full-Adventure action",
        action == 25 and reason == "",
        {"action": action, "reason": reason},
    )
    action_000, _ = command_to_policy_action(
        parse_coach_command("plant 0 0 0"),
        action_space_mode=ACTION_SPACE_ADVENTURE_14_IDENTITY,
        rows=FIVE_LANE_LIVE_ROWS,
        cols=DEFAULT_COLS,
        max_seed_slots=ADVENTURE_IDENTITY_MAX_SEED_SLOTS,
        plant_types=[1, 0, 3, 2],
    )
    action_100, _ = command_to_policy_action(
        parse_coach_command("plant 1 0 0"),
        action_space_mode=ACTION_SPACE_ADVENTURE_14_IDENTITY,
        rows=FIVE_LANE_LIVE_ROWS,
        cols=DEFAULT_COLS,
        max_seed_slots=ADVENTURE_IDENTITY_MAX_SEED_SLOTS,
        plant_types=[1, 0, 3, 2],
    )
    action_last, _ = command_to_policy_action(
        parse_coach_command(
            f"plant {ADVENTURE_IDENTITY_MAX_SEED_SLOTS - 1} "
            f"{DEFAULT_ROWS - 1} {DEFAULT_COLS - 1}"
        ),
        action_space_mode=ACTION_SPACE_ADVENTURE_14_IDENTITY,
        rows=FIVE_LANE_LIVE_ROWS,
        cols=DEFAULT_COLS,
        max_seed_slots=ADVENTURE_IDENTITY_MAX_SEED_SLOTS,
        plant_types=[1, 0, 3, 2],
    )
    assert_case(results, "plant 0 0 0 -> action 1", action_000 == 1, action_000)
    assert_case(
        results,
        f"plant 1 0 0 -> action {1 + CELLS_PER_SLOT}",
        action_100 == 1 + CELLS_PER_SLOT,
        action_100,
    )
    assert_case(
        results,
        f"last slot/cell -> action {ADVENTURE_IDENTITY_ACTION_COUNT - 1}",
        action_last == ADVENTURE_IDENTITY_ACTION_COUNT - 1,
        action_last,
    )

    obs = observation([0, 25])
    validation = validate_coach_command(
        plant,
        action_space_mode=ACTION_SPACE_ADVENTURE_14_IDENTITY,
        observation=obs,
        action_mask=mask_with(0, 25),
        plant_types=[1, 0, 3, 2],
        max_seed_slots=14,
    )
    assert_case(results, "validate legal plant action", validation.legal and validation.policy_action == 25, validation.to_dict())

    illegal = validate_coach_command(
        plant,
        action_space_mode=ACTION_SPACE_ADVENTURE_14_IDENTITY,
        observation=observation([0]),
        action_mask=mask_with(0),
        plant_types=[1, 0, 3, 2],
        max_seed_slots=14,
    )
    assert_case(
        results,
        "never emit illegal action",
        not illegal.legal and illegal.policy_action == 25 and illegal.rejected_reason == "illegal_action",
        illegal.to_dict(),
    )

    out_of_bounds = validate_coach_command(
        parse_coach_command("!plant 99 2 4"),
        action_space_mode=ACTION_SPACE_ADVENTURE_14_IDENTITY,
        observation=obs,
        action_mask=mask_with(0, 25),
        plant_types=[1, 0, 3, 2],
        max_seed_slots=14,
    )
    assert_case(
        results,
        "reject seed outside bounds",
        not out_of_bounds.legal and out_of_bounds.rejected_reason == "seed_index_out_of_bounds",
        out_of_bounds.to_dict(),
    )
    padded_sixth_lane_action = 1 + (DEFAULT_ROWS - 1) * DEFAULT_COLS + 4
    masked_sixth_lane = validate_coach_command(
        parse_coach_command(f"plant 0 {DEFAULT_ROWS - 1} 4"),
        action_space_mode=ACTION_SPACE_ADVENTURE_14_IDENTITY,
        observation=obs,
        action_mask=mask_with(0, 25),
        plant_types=[1, 0, 3, 2],
        max_seed_slots=14,
    )
    assert_case(
        results,
        "five-lane board masks the padded sixth-lane action",
        not masked_sixth_lane.legal
        and masked_sixth_lane.policy_action == padded_sixth_lane_action
        and masked_sixth_lane.rejected_reason == "illegal_action",
        masked_sixth_lane.to_dict(),
    )
    legal_sixth_lane = validate_coach_command(
        parse_coach_command(f"plant 0 {DEFAULT_ROWS - 1} 4"),
        action_space_mode=ACTION_SPACE_ADVENTURE_14_IDENTITY,
        observation=observation(
            [0, padded_sixth_lane_action],
            live_rows=DEFAULT_ROWS,
        ),
        action_mask=mask_with(0, padded_sixth_lane_action),
        plant_types=[1, 0, 3, 2],
        max_seed_slots=ADVENTURE_IDENTITY_MAX_SEED_SLOTS,
    )
    assert_case(
        results,
        "six-lane board accepts the same sixth-lane action",
        legal_sixth_lane.legal
        and legal_sixth_lane.policy_action == padded_sixth_lane_action,
        legal_sixth_lane.to_dict(),
    )
    out_of_bounds_col = validate_coach_command(
        parse_coach_command("plant 0 4 10"),
        action_space_mode=ACTION_SPACE_ADVENTURE_14_IDENTITY,
        observation=obs,
        action_mask=mask_with(0, 25),
        plant_types=[1, 0, 3, 2],
        max_seed_slots=14,
    )
    assert_case(
        results,
        "reject col outside bounds",
        not out_of_bounds_col.legal and out_of_bounds_col.rejected_reason == "col_out_of_bounds",
        out_of_bounds_col.to_dict(),
    )

    rejected_fuse = validate_coach_command(
        fuse,
        action_space_mode=ACTION_SPACE_ADVENTURE_14_IDENTITY,
        observation=observation([0, 12], include_fusion_board=True),
        action_mask=mask_with(0, 12),
        plant_types=[1, 0, 3, 2],
        max_seed_slots=14,
        fusion_enabled=False,
    )
    assert_case(
        results,
        "reject fuse when fusion disabled",
        not rejected_fuse.legal and rejected_fuse.rejected_reason == "fusion_disabled",
        rejected_fuse.to_dict(),
    )

    probe_env = FakeEnv(
        observation([0], include_fusion_board=True),
        mask_with(0),
        fusion_policy="none",
        human_coach_fusion_enabled=True,
        bridge_client=FakeBridgeClient(probe_payload=fusion_probe_payload()),
    )
    probe_hook = HumanCoachOverrideHook(
        enabled=True,
        source=QueueCoachCommandSource(["!fuse 0 1 1"]),
        fusion_enabled=True,
    )
    probe_decision = probe_hook.select_action(probe_env, 0)
    assert_case(
        results,
        "fusion bridge capability probe builds legal !fuse command",
        bool(
            not probe_decision.rejected
            and isinstance(probe_decision.selected_bridge_command, dict)
            and probe_decision.selected_bridge_command.get("command") == "fusion_step"
            and probe_decision.selected_action == 0
        ),
        probe_decision.to_dict(),
    )
    assert_case(
        results,
        "single fuse command creates exactly one fresh fusion_step bridge command",
        bool(
            isinstance(probe_decision.selected_bridge_command, dict)
            and int(probe_decision.selected_bridge_command.get("coach_command_id", 0) or 0) > 0
            and bool(probe_decision.selected_bridge_command.get("executed_from_fresh_coach_command"))
            and sum(
                1
                for request in probe_env.base.client.requests
                if isinstance(request, dict) and request.get("command") == "fusion_probe"
            )
            == 1
            and not any(
                isinstance(request, dict) and request.get("command") == "fusion_step"
                for request in probe_env.base.client.requests
            )
        ),
        {
            "bridge_command": probe_decision.selected_bridge_command,
            "requests": probe_env.base.client.requests,
        },
    )

    bare_probe_env = FakeEnv(
        observation([0], include_fusion_board=True),
        mask_with(0),
        fusion_policy="none",
        human_coach_fusion_enabled=True,
        bridge_client=FakeBridgeClient(probe_payload=fusion_probe_payload()),
    )
    bare_probe_env.client = bare_probe_env.base.client
    del bare_probe_env.base
    bare_probe_hook = HumanCoachOverrideHook(
        enabled=True,
        source=QueueCoachCommandSource(["!fuse 0 1 1"]),
        fusion_enabled=True,
    )
    bare_probe_decision = bare_probe_hook.select_action(bare_probe_env, 0)
    assert_case(
        results,
        "fusion bridge probe supports a bare PvZGymEnv client boundary",
        bool(
            not bare_probe_decision.rejected
            and isinstance(bare_probe_decision.selected_bridge_command, dict)
            and bare_probe_decision.selected_bridge_command.get("command") == "fusion_step"
        ),
        bare_probe_decision.to_dict(),
    )

    offset_obs = observation([0], include_fusion_board=True)
    for index, slot in enumerate(offset_obs.get("seedSlots", [])):
        if isinstance(slot, dict):
            slot["slotIndex"] = int(index + 1)
    offset_probe_env = FakeEnv(
        offset_obs,
        mask_with(0),
        fusion_policy="none",
        human_coach_fusion_enabled=True,
        bridge_client=FakeBridgeClient(probe_payload=fusion_probe_payload_slot_index(1)),
    )
    offset_probe_hook = HumanCoachOverrideHook(
        enabled=True,
        source=QueueCoachCommandSource(["!fuse 0 1 1"]),
        fusion_enabled=True,
    )
    offset_probe_decision = offset_probe_hook.select_action(offset_probe_env, 0)
    offset_validation_diag = (
        offset_probe_decision.validation.diagnostics
        if offset_probe_decision.validation is not None and isinstance(offset_probe_decision.validation.diagnostics, dict)
        else {}
    )
    assert_case(
        results,
        "fusion probe maps command slot index to runtime slot index",
        bool(
            not offset_probe_decision.rejected
            and isinstance(offset_probe_decision.selected_bridge_command, dict)
            and int(offset_probe_decision.selected_bridge_command.get("ingredient_seed_slot_index", -1)) == 1
            and int(offset_validation_diag.get("resolved_seed_slot_index", -1)) == 1
            and int(offset_validation_diag.get("matched_seed_slot_index", -1)) == 1
        ),
        offset_probe_decision.to_dict(),
    )

    duplicate_env = FakeEnv(
        observation_duplicate_sunflower_fusion([0]),
        mask_with(0),
        fusion_policy="none",
        human_coach_fusion_enabled=True,
        bridge_client=FakeBridgeClient(probe_payload=fusion_probe_payload_duplicate_slot_fallback()),
    )
    duplicate_hook = HumanCoachOverrideHook(
        enabled=True,
        source=QueueCoachCommandSource(["!fuse 0 1 1"]),
        fusion_enabled=True,
    )
    duplicate_decision = duplicate_hook.select_action(duplicate_env, 0)
    duplicate_diag = (
        duplicate_decision.validation.diagnostics
        if duplicate_decision.validation is not None and isinstance(duplicate_decision.validation.diagnostics, dict)
        else {}
    )
    assert_case(
        results,
        "duplicate-slot fallback selects legal equivalent slot",
        bool(
            not duplicate_decision.rejected
            and isinstance(duplicate_decision.selected_bridge_command, dict)
            and int(duplicate_decision.selected_bridge_command.get("ingredient_seed_slot_index", -1)) == 1
            and bool(duplicate_diag.get("duplicate_slot_fallback_attempted"))
            and bool(duplicate_diag.get("duplicate_slot_fallback_applied"))
            and int(duplicate_diag.get("duplicate_slot_fallback_from_seed_slot_index", -1)) == 0
            and int(duplicate_diag.get("duplicate_slot_fallback_to_seed_slot_index", -1)) == 1
        ),
        duplicate_decision.to_dict(),
    )

    duplicate_fail_env = FakeEnv(
        observation_duplicate_sunflower_fusion([0]),
        mask_with(0),
        fusion_policy="none",
        human_coach_fusion_enabled=True,
        bridge_client=FakeBridgeClient(probe_payload=fusion_probe_payload_duplicate_slot_all_illegal()),
    )
    duplicate_fail_hook = HumanCoachOverrideHook(
        enabled=True,
        source=QueueCoachCommandSource(["!fuse 0 1 1"]),
        fusion_enabled=True,
    )
    duplicate_fail_decision = duplicate_fail_hook.select_action(duplicate_fail_env, 0)
    duplicate_fail_diag = (
        duplicate_fail_decision.validation.diagnostics
        if duplicate_fail_decision.validation is not None and isinstance(duplicate_fail_decision.validation.diagnostics, dict)
        else {}
    )
    assert_case(
        results,
        "explicit illegal slot fails cleanly when no legal duplicate exists",
        bool(
            duplicate_fail_decision.rejected
            and duplicate_fail_decision.rejected_reason == "bridge_rejected"
            and bool(duplicate_fail_diag.get("duplicate_slot_fallback_attempted"))
            and (not bool(duplicate_fail_diag.get("duplicate_slot_fallback_applied")))
            and isinstance(duplicate_fail_diag.get("candidate_slots_checked"), list)
            and len(duplicate_fail_diag.get("candidate_slots_checked")) >= 2
            and duplicate_fail_diag.get("source_found") is True
        ),
        duplicate_fail_decision.to_dict(),
    )

    assert_case(
        results,
        "bridge_rejected includes rich fusion probe diagnostics",
        bool(
            isinstance(duplicate_fail_diag.get("candidate_slots_checked"), list)
            and isinstance(duplicate_fail_diag.get("selected_slot_probe"), dict)
            and "bridge_rejection_reason" in duplicate_fail_diag
            and "requested_seed_slot_index" in duplicate_fail_diag
            and "requested_row" in duplicate_fail_diag
            and "requested_col" in duplicate_fail_diag
        ),
        duplicate_fail_diag,
    )
    duplicate_fail_status = duplicate_fail_hook.live_status_fields()
    assert_case(
        results,
        "live status exposes last fusion probe diagnostics",
        bool(
            isinstance(duplicate_fail_status.get("human_coach_last_fusion_probe_diagnostics"), dict)
            and isinstance(duplicate_fail_status.get("stream_coach_last_fusion_probe_diagnostics"), dict)
            and duplicate_fail_status.get("human_coach_last_fusion_probe_diagnostics", {}).get("fusion_probe_reason")
            == "bridge_rejected"
        ),
        duplicate_fail_status,
    )

    no_bridge_env = FakeEnv(
        observation([0], include_fusion_board=True),
        mask_with(0),
        fusion_policy="none",
        human_coach_fusion_enabled=True,
        bridge_client=None,
    )
    no_bridge_hook = HumanCoachOverrideHook(
        enabled=True,
        source=QueueCoachCommandSource(["!fuse 0 1 1"]),
        fusion_enabled=True,
    )
    no_bridge_decision = no_bridge_hook.select_action(no_bridge_env, 0)
    assert_case(
        results,
        "reject fuse when bridge probe is unavailable",
        bool(no_bridge_decision.rejected and no_bridge_decision.rejected_reason == "fusion_bridge_unavailable"),
        no_bridge_decision.to_dict(),
    )

    env = FakeEnv(obs, mask_with(0, 25))
    hook = HumanCoachOverrideHook(enabled=True, source=QueueCoachCommandSource(["!plant 0 2 4"]))
    _obs, _reward, _terminated, _truncated, info = hook.step_env(env, 0)
    assert_case(
        results,
        "override env.step with legal coach action",
        env.executed_actions == [25] and info["human_coach"]["override_applied"] is True,
        info.get("human_coach"),
    )
    assert_case(
        results,
        "14-slot non-fusion plant path remains unchanged",
        bool(
            info.get("human_coach", {}).get("command", {}).get("kind") == "plant"
            and info.get("human_coach", {}).get("selected_action") == 25
            and info.get("human_coach", {}).get("selected_bridge_command") is None
        ),
        info.get("human_coach"),
    )

    fallback_env = FakeEnv(observation([0]), mask_with(0))
    fallback_hook = HumanCoachOverrideHook(enabled=True, source=QueueCoachCommandSource(["!plant 99 2 4"]))
    _obs, _reward, _terminated, _truncated, fallback_info = fallback_hook.step_env(fallback_env, 0)
    assert_case(
        results,
        "fallback to PPO for permanently invalid coach command",
        bool(
            fallback_env.executed_actions == [0]
            and fallback_info["human_coach"]["rejected"] is True
            and fallback_info["human_coach"]["event"] == "coach_rejected"
        ),
        fallback_info.get("human_coach"),
    )

    pending_env = FakeEnv(observation([0], sun=0), mask_with(0))
    pending_hook = HumanCoachOverrideHook(enabled=True, source=QueueCoachCommandSource(["!plant 0 2 4"]))
    _obs, _reward, _terminated, _truncated, pending_info_1 = pending_hook.step_env(pending_env, 25)
    pending_env._last_observation = observation([0, 25], sun=500)
    pending_env._action_mask = mask_with(0, 25)
    _obs, _reward, _terminated, _truncated, pending_info_2 = pending_hook.step_env(pending_env, 0)
    assert_case(
        results,
        "queue transiently illegal command and execute when legal",
        bool(
            pending_env.executed_actions == [0, 25]
            and pending_info_1.get("human_coach", {}).get("event") == "coach_pending"
            and pending_info_1.get("human_coach", {}).get("rejected") is False
            and pending_info_2.get("human_coach", {}).get("event") == "coach_override"
            and pending_info_2.get("human_coach", {}).get("rejected") is False
        ),
        {
            "first_step": pending_info_1.get("human_coach"),
            "second_step": pending_info_2.get("human_coach"),
            "executed_actions": pending_env.executed_actions,
        },
    )

    pending_fuse_env = FakeEnv(
        observation_duplicate_sunflower_fusion([0], sun=0),
        mask_with(0),
        fusion_policy="none",
        human_coach_fusion_enabled=True,
        bridge_client=FakeBridgeClient(probe_payload=fusion_probe_payload_duplicate_slot_fallback()),
    )
    pending_fuse_hook = HumanCoachOverrideHook(
        enabled=True,
        source=QueueCoachCommandSource(["!fuse 0 1 1"]),
        fusion_enabled=True,
    )
    _obs, _reward, _terminated, _truncated, pending_fuse_info = pending_fuse_hook.step_env(pending_fuse_env, 0)
    pending_fuse_diag = pending_fuse_info.get("human_coach", {}).get("validation", {}).get("diagnostics", {})
    assert_case(
        results,
        "fuse command enters pending when sun is insufficient",
        bool(
            pending_fuse_info.get("human_coach", {}).get("event") == "coach_pending"
            and pending_fuse_info.get("human_coach", {}).get("rejected_reason") == "insufficient_sun"
            and pending_fuse_diag.get("pending") is True
            and pending_fuse_diag.get("pending_command_kind") == "fuse"
        ),
        pending_fuse_info.get("human_coach"),
    )
    clear_hook = HumanCoachOverrideHook(
        enabled=True,
        source=QueueCoachCommandSource(["!fuse 0 1 1"]),
        fusion_enabled=True,
    )
    stale_cleared = clear_hook.clear_pending_state(clear_source=True, reason="eval_start")
    clear_status = clear_hook.live_status_fields()
    clear_env = FakeEnv(
        observation([0], include_fusion_board=True),
        mask_with(0),
        fusion_policy="none",
        human_coach_fusion_enabled=True,
        bridge_client=FakeBridgeClient(probe_payload=fusion_probe_payload()),
    )
    _obs, _reward, _terminated, _truncated, clear_info = clear_hook.step_env(clear_env, 0)
    assert_case(
        results,
        "reset clears pending coach and fusion queue state",
        bool(
            stale_cleared
            and clear_status.get("coach_command_queue_cleared_on_reset") is True
            and clear_status.get("pending_coach_command") is None
            and clear_status.get("selected_bridge_command") is None
            and clear_status.get("startup_command_blocked") is True
            and clear_env.executed_bridge_commands == [None]
            and clear_info.get("human_coach", {}).get("event") == "no_command"
        ),
        {"status": clear_status, "info": clear_info.get("human_coach")},
    )

    clean_clear_hook = HumanCoachOverrideHook(enabled=True, source=QueueCoachCommandSource([]), fusion_enabled=True)
    clean_stale = clean_clear_hook.clear_pending_state(clear_source=True, reason="eval_start")
    clean_status = clean_clear_hook.live_status_fields()
    assert_case(
        results,
        "clean eval startup live status has no pending selected command",
        bool(
            clean_stale is False
            and clean_status.get("coach_command_queue_cleared_on_reset") is True
            and clean_status.get("pending_coach_command") is None
            and clean_status.get("selected_bridge_command") is None
            and clean_status.get("startup_command_blocked") is False
        ),
        clean_status,
    )

    no_command_env = FakeEnv(obs, mask_with(0, 25))
    no_command_hook = HumanCoachOverrideHook(enabled=True, source=QueueCoachCommandSource([]))
    _obs, _reward, _terminated, _truncated, no_command_info = no_command_hook.step_env(no_command_env, 0)
    assert_case(
        results,
        "fallback to PPO when no coach command is available",
        no_command_env.executed_actions == [0] and no_command_info["human_coach"]["event"] == "no_command",
        no_command_info.get("human_coach"),
    )

    fusion_env = FakeEnv(
        observation([0], include_fusion_board=True),
        mask_with(0),
        fusion_policy="none",
        human_coach_fusion_enabled=True,
        bridge_client=FakeBridgeClient(probe_payload=fusion_probe_payload()),
        fusion_step_success=True,
        tactical_signal=True,
    )
    fusion_hook = HumanCoachOverrideHook(
        enabled=True,
        source=QueueCoachCommandSource(["!fuse 0 1 1"]),
        reward_enabled=True,
        fusion_enabled=True,
    )
    _obs, fusion_reward, _terminated, _truncated, fusion_info = fusion_hook.step_env(fusion_env, 0)
    fusion_breakdown = fusion_info.get("reward_breakdown", {})
    fusion_status = fusion_hook.live_status_fields()
    assert_case(
        results,
        "!fuse execution route passes bridge command to env.step",
        bool(
            fusion_env.executed_actions == [0]
            and fusion_env.executed_bridge_commands
            and isinstance(fusion_env.executed_bridge_commands[0], dict)
            and fusion_env.executed_bridge_commands[0].get("command") == "fusion_step"
        ),
        {
            "executed_actions": fusion_env.executed_actions,
            "executed_bridge_commands": fusion_env.executed_bridge_commands,
            "decision": fusion_info.get("human_coach"),
        },
    )
    fusion_action_result = fusion_info.get("action_result", {})
    fusion_placement = fusion_action_result.get("placement", {}) if isinstance(fusion_action_result, dict) else {}
    assert_case(
        results,
        "successful fuse reports dedicated fusion diagnostics",
        bool(
            isinstance(fusion_action_result, dict)
            and str(fusion_action_result.get("fusionExecutionMode")) == "dedicated_fusion"
            and str(fusion_action_result.get("bridgeMethodUsed")).strip() != ""
            and "createplant.setplant" not in str(fusion_action_result.get("bridgeMethodUsed", "")).lower()
            and bool(fusion_action_result.get("sourceTileOccupiedBefore"))
            and int(fusion_action_result.get("plantCountOnTileBefore", 0)) >= 1
            and int(fusion_action_result.get("plantCountOnTileAfter", 0)) == 1
                and bool(isinstance(fusion_action_result.get("sourcePlantBefore"), dict))
                and bool(isinstance(fusion_action_result.get("resultingPlantAfter"), dict))
                and int(fusion_action_result.get("changedTileCount", 0)) == 1
                and not bool(fusion_action_result.get("nonSourceTilesChanged"))
                and not bool(fusion_action_result.get("globalFusionSideEffect"))
                and str(fusion_action_result.get("fusionScope") or "") == "tile_scoped"
                and isinstance(fusion_action_result.get("changedTiles"), list)
                and len(fusion_action_result.get("changedTiles") or []) == 1
                and isinstance(fusion_placement, dict)
                and str(fusion_placement.get("fusionExecutionMode")) == "dedicated_fusion"
                and str(fusion_placement.get("bridgeMethodUsed", "")).strip() != ""
            ),
        fusion_action_result,
    )
    selected_bridge_command = fusion_env.executed_bridge_commands[0] if fusion_env.executed_bridge_commands else {}
    assert_case(
        results,
        "Peashooter+SunFlower fusion probe resolves predicted result mapping",
        bool(
            isinstance(selected_bridge_command, dict)
            and int(selected_bridge_command.get("source_plant_type", -1)) == 0
            and int(selected_bridge_command.get("ingredient_plant_type", -1)) == 1
            and int(selected_bridge_command.get("predicted_result_type", -1)) > 0
            and str(selected_bridge_command.get("predicted_result_name") or "") != ""
        ),
        selected_bridge_command,
    )
    assert_case(
        results,
        "fusion result exposes mix lookup diagnostics",
        bool(
            isinstance(fusion_action_result, dict)
            and str(fusion_action_result.get("predictedResultResolutionSource") or "") != ""
            and bool(fusion_action_result.get("mixLookupFound"))
            and str(fusion_action_result.get("mixLookupKey") or "") != ""
            and int(fusion_action_result.get("preSourceType", -1)) >= 0
            and str(fusion_action_result.get("preSourceName") or "") != ""
            and int(fusion_action_result.get("ingredientType", -1)) >= 0
            and str(fusion_action_result.get("ingredientName") or "") != ""
            and int(fusion_action_result.get("postResultType", -1)) >= 0
            and str(fusion_action_result.get("postResultName") or "") != ""
        ),
        fusion_action_result,
    )
    assert_case(
        results,
        "coach fusion receives no duplicate or tactical shaping",
        bool(
            float(fusion_breakdown.get(COACH_REWARD_MATCH_COMPONENT, 0.0)) == 0.0
            and float(fusion_breakdown.get(COACH_REWARD_LEGAL_EXECUTION_COMPONENT, 0.0)) == 0.0
            and float(fusion_breakdown.get(COACH_REWARD_FUSION_SUCCESS_COMPONENT, 0.0)) == 0.0
            and float(fusion_breakdown.get(COACH_REWARD_TACTICAL_USEFULNESS_COMPONENT, 0.0)) == 0.0
            and float(fusion_reward) == 1.0
        ),
        {"reward": fusion_reward, "reward_breakdown": fusion_breakdown},
    )
    assert_case(
        results,
        "fusion/tactical diagnostics stay tracked without reward shaping",
        bool(
            int(fusion_status.get("human_coach_fusion_attempt_count", 0)) >= 1
            and int(fusion_status.get("human_coach_fusion_success_count", 0)) >= 1
            and int(fusion_status.get("human_coach_tactical_useful_count", 0)) >= 1
            and float(fusion_status.get("human_coach_fusion_success_reward_total", 0.0)) == 0.0
        ),
        fusion_status,
    )
    duplicate_id_command = parse_coach_command("!fuse 0 1 1")
    duplicate_id_env = FakeEnv(
        observation([0], include_fusion_board=True),
        mask_with(0),
        fusion_policy="none",
        human_coach_fusion_enabled=True,
        bridge_client=FakeBridgeClient(probe_payload=fusion_probe_payload()),
        fusion_step_success=True,
    )
    duplicate_id_hook = HumanCoachOverrideHook(
        enabled=True,
        source=QueueCoachCommandSource([duplicate_id_command, duplicate_id_command]),
        fusion_enabled=True,
    )
    _obs, _reward, _terminated, _truncated, duplicate_id_info_1 = duplicate_id_hook.step_env(duplicate_id_env, 0)
    _obs, _reward, _terminated, _truncated, duplicate_id_info_2 = duplicate_id_hook.step_env(duplicate_id_env, 0)
    assert_case(
        results,
        "same coach command id cannot execute twice",
        bool(
            len(duplicate_id_env.executed_bridge_commands) == 2
            and isinstance(duplicate_id_env.executed_bridge_commands[0], dict)
            and duplicate_id_env.executed_bridge_commands[1] is None
            and duplicate_id_info_2.get("human_coach", {}).get("rejected") is True
            and duplicate_id_info_2.get("human_coach", {}).get("rejected_reason") == "coach_command_already_executed"
        ),
        {
            "first": duplicate_id_info_1.get("human_coach"),
            "second": duplicate_id_info_2.get("human_coach"),
            "executed_bridge_commands": duplicate_id_env.executed_bridge_commands,
        },
    )
    assert_case(
        results,
        "live status exposes dedicated fusion method diagnostics",
        bool(
            str(fusion_status.get("fusion_last_execution_mode") or "") == "dedicated_fusion"
            and str(fusion_status.get("fusion_last_bridge_method_used") or "") != ""
            and str(fusion_status.get("fusion_last_bridge_method_used") or "").lower().find("createplant.setplant") < 0
            and str(fusion_status.get("fusion_last_bridge_result_reason") or "") != ""
            and fusion_status.get("fusion_last_duplicate_stack_detected") is False
            and int(fusion_status.get("fusion_last_plant_count_on_tile_before", -1)) >= 1
            and int(fusion_status.get("fusion_last_plant_count_on_tile_after", -1)) == 1
            and isinstance(fusion_status.get("fusion_last_source_plant_before"), dict)
            and isinstance(fusion_status.get("fusion_last_resulting_plant_after"), dict)
            and str(fusion_status.get("fusion_last_predicted_result_resolution_source") or "") != ""
            and bool(fusion_status.get("fusion_last_mix_lookup_found"))
            and str(fusion_status.get("fusion_last_mix_lookup_key") or "") != ""
            and int(fusion_status.get("fusion_last_pre_source_type", -1)) >= 0
            and str(fusion_status.get("fusion_last_pre_source_name") or "") != ""
            and int(fusion_status.get("fusion_last_ingredient_type", -1)) >= 0
            and str(fusion_status.get("fusion_last_ingredient_name") or "") != ""
            and int(fusion_status.get("fusion_last_post_result_type", -1)) >= 0
            and str(fusion_status.get("fusion_last_post_result_name") or "") != ""
            and str(fusion_status.get("last_fusion_scope") or "") == "tile_scoped"
            and int(fusion_status.get("last_fusion_changed_tile_count", -1)) == 1
            and fusion_status.get("last_fusion_non_source_tiles_changed") is False
            and fusion_status.get("last_fusion_global_side_effect") is False
            and isinstance(fusion_status.get("last_executed_coach_command_id"), int)
        ),
        fusion_status,
    )

    fuse_requires_occupied_env = FakeEnv(
        observation([0], include_fusion_board=True),
        mask_with(0),
        fusion_policy="none",
        human_coach_fusion_enabled=True,
        bridge_client=FakeBridgeClient(probe_payload=fusion_probe_payload()),
        fusion_step_success=False,
        fusion_source_tile_occupied_before=False,
        fusion_bridge_result_reason="source_tile_not_occupied",
    )
    fuse_requires_occupied_hook = HumanCoachOverrideHook(
        enabled=True,
        source=QueueCoachCommandSource(["!fuse 0 1 1"]),
        fusion_enabled=True,
    )
    _obs, _reward, _terminated, _truncated, fuse_requires_occupied_info = fuse_requires_occupied_hook.step_env(
        fuse_requires_occupied_env,
        0,
    )
    occupied_action_result = fuse_requires_occupied_info.get("action_result", {})
    assert_case(
        results,
        "fuse requires an occupied source tile",
        bool(
            isinstance(occupied_action_result, dict)
            and occupied_action_result.get("fusionSucceeded") is False
            and occupied_action_result.get("sourceTileOccupiedBefore") is False
            and str(occupied_action_result.get("illegalReason") or "") == "source_tile_not_occupied"
            and str(occupied_action_result.get("bridgeResultReason") or "") == "source_tile_not_occupied"
        ),
        occupied_action_result,
    )

    duplicate_stack_env = FakeEnv(
        observation([0], include_fusion_board=True),
        mask_with(0),
        fusion_policy="none",
        human_coach_fusion_enabled=True,
        bridge_client=FakeBridgeClient(probe_payload=fusion_probe_payload()),
        fusion_step_success=False,
        fusion_duplicate_stack_detected=True,
        fusion_plant_count_on_tile_before=1,
        fusion_plant_count_on_tile_after=2,
        fusion_bridge_result_reason="duplicate_stack_detected",
    )
    duplicate_stack_hook = HumanCoachOverrideHook(
        enabled=True,
        source=QueueCoachCommandSource(["!fuse 0 1 1"]),
        fusion_enabled=True,
    )
    _obs, _reward, _terminated, _truncated, duplicate_stack_info = duplicate_stack_hook.step_env(duplicate_stack_env, 0)
    duplicate_action_result = duplicate_stack_info.get("action_result", {})
    duplicate_stack_status = duplicate_stack_hook.live_status_fields()
    assert_case(
        results,
        "duplicate fusion stack is detected as failure",
        bool(
            isinstance(duplicate_action_result, dict)
            and duplicate_action_result.get("fusionSucceeded") is False
            and duplicate_action_result.get("duplicateStackDetected") is True
            and int(duplicate_action_result.get("plantCountOnTileBefore", 0)) == 1
            and int(duplicate_action_result.get("plantCountOnTileAfter", 0)) == 2
            and str(duplicate_action_result.get("illegalReason") or "") == "duplicate_stack_detected"
        ),
        duplicate_action_result,
    )
    assert_case(
        results,
        "fusion failure reason is surfaced in coach live status",
        bool(
            duplicate_stack_status.get("fusion_last_duplicate_stack_detected") is True
            and int(duplicate_stack_status.get("fusion_last_plant_count_on_tile_after", -1)) == 2
            and str(duplicate_stack_status.get("fusion_last_bridge_result_reason") or "") == "duplicate_stack_detected"
            and str(duplicate_stack_status.get("human_coach_last_error") or "") == "duplicate_stack_detected"
        ),
        duplicate_stack_status,
    )

    global_side_effect_env = FakeEnv(
        observation([0], include_fusion_board=True),
        mask_with(0),
        fusion_policy="none",
        human_coach_fusion_enabled=True,
        bridge_client=FakeBridgeClient(probe_payload=fusion_probe_payload()),
        fusion_step_success=True,
        fusion_bridge_result_reason="success",
        fusion_changed_tile_count=2,
        fusion_non_source_tiles_changed=True,
        fusion_global_fusion_side_effect=True,
        fusion_scope="global_side_effect_detected",
        fusion_changed_tiles=[
            {
                "row": 1,
                "column": 1,
                "beforePlantCount": 1,
                "afterPlantCount": 1,
                "beforePlants": [{"instanceId": 111, "plantType": 0, "plantTypeName": "Peashooter", "row": 1, "column": 1}],
                "afterPlants": [{"instanceId": 111, "plantType": 1000, "plantTypeName": "PeaSunFlower", "row": 1, "column": 1}],
            },
            {
                "row": 2,
                "column": 2,
                "beforePlantCount": 1,
                "afterPlantCount": 1,
                "beforePlants": [{"instanceId": 222, "plantType": 0, "plantTypeName": "Peashooter", "row": 2, "column": 2}],
                "afterPlants": [{"instanceId": 222, "plantType": 1000, "plantTypeName": "PeaSunFlower", "row": 2, "column": 2}],
            },
        ],
    )
    global_side_effect_hook = HumanCoachOverrideHook(
        enabled=True,
        source=QueueCoachCommandSource(["!fuse 0 1 1"]),
        fusion_enabled=True,
    )
    _obs, _reward, _terminated, _truncated, global_side_effect_info = global_side_effect_hook.step_env(global_side_effect_env, 0)
    global_side_effect_result = global_side_effect_info.get("action_result", {})
    global_side_effect_status = global_side_effect_hook.live_status_fields()
    assert_case(
        results,
        "bridge-free postcondition validator rejects global fusion side effects",
        bool(
            isinstance(global_side_effect_result, dict)
            and global_side_effect_result.get("fusionSucceeded") is False
            and str(global_side_effect_result.get("illegalReason") or "") == "global_fusion_side_effect"
            and str(global_side_effect_result.get("bridgeResultReason") or "") == "fusion_mutated_non_source_tiles"
            and int(global_side_effect_result.get("changedTileCount", 0)) > 1
            and bool(global_side_effect_result.get("nonSourceTilesChanged"))
            and bool(global_side_effect_result.get("globalFusionSideEffect"))
            and str(global_side_effect_result.get("fusionScope") or "") == "global_side_effect_detected"
            and isinstance(global_side_effect_result.get("changedTiles"), list)
            and len(global_side_effect_result.get("changedTiles") or []) > 1
        ),
        global_side_effect_result,
    )
    assert_case(
        results,
        "global fusion side effect failure reason is surfaced in live status",
        bool(
            str(global_side_effect_status.get("fusion_last_bridge_result_reason") or "") == "fusion_mutated_non_source_tiles"
            and str(global_side_effect_status.get("human_coach_last_error") or "") == "global_fusion_side_effect"
        ),
        global_side_effect_status,
    )

    override_env = FakeEnv(obs, mask_with(0, 25))
    override_hook = HumanCoachOverrideHook(
        enabled=True,
        source=QueueCoachCommandSource(["!plant 0 2 4"]),
        reward_enabled=True,
    )
    _obs, override_reward, _terminated, _truncated, override_info = override_hook.step_env(override_env, 0)
    override_breakdown = override_info.get("reward_breakdown", {})
    assert_case(
        results,
        "override penalty is applied when coach action differs from PPO action",
        bool(float(override_breakdown.get(COACH_REWARD_OVERRIDE_PENALTY_COMPONENT, 0.0)) < 0.0 and float(override_reward) > 0.0),
        {"reward": override_reward, "reward_breakdown": override_breakdown},
    )

    with tempfile.TemporaryDirectory(prefix="pvzrl_human_coach_test_") as temp_dir:
        log_path = Path(temp_dir) / "coach.jsonl"
        log_env = FakeEnv(obs, mask_with(0, 25))
        log_hook = HumanCoachOverrideHook(
            enabled=True,
            source=QueueCoachCommandSource(["!plant 0 2 4"]),
            log_path=log_path,
            reward_enabled=True,
        )
        _obs, reward, _terminated, _truncated, _info = log_hook.step_env(log_env, 25)
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        logged = json.loads(lines[0]) if lines else {}
        live_status = log_hook.live_status_fields()
        assert_case(
            results,
            "jsonl logging and live status diagnostics",
            bool(
                lines
                and logged.get("event") == "coach_match"
                and live_status.get("stream_coach_match_count") == 1
                and live_status.get("stream_coach_reward_total", 0.0) > 0.0
                and reward > 1.0
            ),
            {"logged": logged, "live_status": live_status, "reward": reward},
        )

    with tempfile.TemporaryDirectory(prefix="pvzrl_human_coach_queue_test_") as temp_dir:
        queue_path = Path(temp_dir) / "coach_commands.jsonl"
        queue_path.write_text('{"command":"!plant 0 2 4"', encoding="utf-8")
        file_source = FileCoachCommandSource(queue_path)
        first_poll = file_source.poll()
        with queue_path.open("a", encoding="utf-8") as handle:
            handle.write("}\n")
            handle.write("{not json}\n")
            handle.write(json.dumps({"source": "gui", "command": "plant 0 2 4"}) + "\n")
        second_poll = file_source.poll()
        third_poll = file_source.poll()
        with queue_path.open("a", encoding="utf-8") as handle:
            handle.write("!plant 0 2 4\n")
        queue_env = FakeEnv(obs, mask_with(0, 25))
        queue_hook = HumanCoachOverrideHook(enabled=True, source=file_source)
        _obs, _reward, _terminated, _truncated, queue_info = queue_hook.step_env(queue_env, 0)
        assert_case(
            results,
            "file coach queue starts at EOF and consumes only fresh GUI commands",
            bool(
                first_poll is None
                and second_poll == "plant 0 2 4"
                and third_poll is None
                and queue_env.executed_actions == [25]
                and queue_info.get("human_coach", {}).get("event") == "coach_override"
            ),
            {
                "first_poll": first_poll,
                "second_poll": second_poll,
                "third_poll": third_poll,
                "executed_actions": queue_env.executed_actions,
                "human_coach": queue_info.get("human_coach"),
            },
        )

    payload = {"ok": all(result["passed"] for result in results), "results": results}
    print(json.dumps(payload, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
