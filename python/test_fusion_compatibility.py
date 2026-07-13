"""Bridge-free tests for the centralized fusion compatibility system.

Covers the single-source-of-truth table and helpers in pvzrl_fusion, the model
action mask plant/fuse distinction in pvzrl_env, and the shared coach command
validation in pvzrl_human_coach (also used by the stream coach).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pvzrl_fusion as F
from pvzrl_action_space import ACTION_SPACE_ADVENTURE_14_IDENTITY
from pvzrl_env import PvZEnvConfig, PvZGymEnv, decode_action
from pvzrl_human_coach import parse_coach_command, validate_coach_command


PEASHOOTER, SUNFLOWER, WALLNUT, CHERRYBOMB = 0, 1, 2, 3  # registry ids: pea=0 sun=1 cherry=2 wall=3
CELLS = 50


def assert_case(results: List[Dict[str, Any]], name: str, condition: bool, detail: Any = None) -> None:
    results.append({"case": name, "passed": bool(condition), "detail": detail})


def board_observation(plant_at=None, sun: int = 500) -> Dict[str, Any]:
    """Standard 5x10 board; seed slots positionally = SunFlower, Peashooter, WallNut, CherryBomb."""
    obs: Dict[str, Any] = {
        "rowCount": 5,
        "columnCount": 10,
        "gameplayReady": True,
        "boardFound": True,
        "canReadBoard": True,
        "sun": int(sun),
        "seedSlots": [
            {"slotIndex": 0, "plantType": 1, "plantTypeName": "SunFlower", "ready": True, "usable": True, "seedCost": 50},
            {"slotIndex": 1, "plantType": 0, "plantTypeName": "Peashooter", "ready": True, "usable": True, "seedCost": 100},
            {"slotIndex": 2, "plantType": 3, "plantTypeName": "WallNut", "ready": True, "usable": True, "seedCost": 50},
            {"slotIndex": 3, "plantType": 2, "plantTypeName": "CherryBomb", "ready": True, "usable": True, "seedCost": 150},
        ],
        "plants": [],
    }
    if plant_at is not None:
        row, col, plant_type = plant_at
        obs["plants"] = [{"row": row, "column": col, "type": plant_type, "alive": True}]
    return obs


def encode(slot_index: int, row: int, col: int) -> int:
    """Generalist identity placement encoding (1 + slot*cells + row*cols + col)."""
    return 1 + slot_index * CELLS + row * 10 + col


def run_compatibility_table_cases(results: List[Dict[str, Any]]) -> None:
    # The 3 required pairs, both directions.
    assert_case(results, "SunFlower + Peashooter is legal (both directions)",
                F.are_fusion_compatible(1, 0) and F.are_fusion_compatible(0, 1))
    assert_case(results, "SunFlower + CherryBomb is illegal (both directions)",
                (not F.are_fusion_compatible(1, 2)) and (not F.are_fusion_compatible(2, 1)))
    assert_case(results, "Peashooter + CherryBomb is legal (both directions)",
                F.are_fusion_compatible(0, 2) and F.are_fusion_compatible(2, 0))
    # The two self-upgrades explicitly listed in FUSION_RULES are compatible.
    assert_case(results, "known self-fusions are compatible",
                F.are_fusion_compatible(1, 1) and F.are_fusion_compatible(0, 0))
    assert_case(results, "unknown plant ids are not compatible",
                not F.are_fusion_compatible(999, 0))
    # Name-based + alias-based resolution agrees with id-based resolution.
    assert_case(results, "names and aliases resolve to the same compatibility",
                F.are_fusion_compatible("SunFlower", "Peashooter")
                and F.are_fusion_compatible("sun", "pea")
                and not F.are_fusion_compatible("SunFlower", "CherryBomb"))
    # Symmetry of the published table.
    table = F.fusion_compatibility_table()
    symmetric = all(
        other in table.get(name, []) and name in table.get(other, [])
        for name, partners in table.items()
        for other in partners
    )
    assert_case(results, "fusion_compatibility_table is symmetric", symmetric, table)


def run_normalize_cases(results: List[Dict[str, Any]]) -> None:
    assert_case(results, "normalize int id", F.normalize_plant_name_or_id(2) == 2)
    assert_case(results, "normalize numeric string", F.normalize_plant_name_or_id("1") == 1)
    assert_case(results, "normalize canonical name", F.normalize_plant_name_or_id("CherryBomb") == 2)
    assert_case(results, "normalize alias", F.normalize_plant_name_or_id("pea") == 0)
    assert_case(results, "normalize seed-slot dict", F.normalize_plant_name_or_id({"plantType": 1}) == 1)
    assert_case(results, "normalize plant dict", F.normalize_plant_name_or_id({"type": 0}) == 0)
    assert_case(results, "normalize unknown name -> None", F.normalize_plant_name_or_id("Zomboni") is None)
    assert_case(results, "normalize None -> None", F.normalize_plant_name_or_id(None) is None)
    assert_case(results, "normalize bool is rejected", F.normalize_plant_name_or_id(True) is None)


def run_legality_helper_cases(results: List[Dict[str, Any]]) -> None:
    # Case 1: SunFlower tile + Peashooter seed -> legal fusion.
    assert_case(results, "Case1 SunFlower+Peashooter legal",
                F.is_legal_fusion_action(board_observation((2, 4, SUNFLOWER)), 2, 4, 1)
                and F.get_fusion_illegal_reason(board_observation((2, 4, SUNFLOWER)), 2, 4, 1) == "")
    # Case 2: SunFlower tile + CherryBomb seed -> incompatible_pair.
    assert_case(results, "Case2 SunFlower+CherryBomb -> incompatible_pair",
                F.get_fusion_illegal_reason(board_observation((2, 4, SUNFLOWER)), 2, 4, 3) == "incompatible_pair"
                and not F.is_legal_fusion_action(board_observation((2, 4, SUNFLOWER)), 2, 4, 3))
    # Case 3: Peashooter tile + CherryBomb seed -> legal fusion.
    assert_case(results, "Case3 Peashooter+CherryBomb legal",
                F.is_legal_fusion_action(board_observation((2, 4, PEASHOOTER)), 2, 4, 3))
    assert_case(results, "Case4 SunFlower+SunFlower legal",
                F.is_legal_fusion_action(board_observation((2, 4, SUNFLOWER)), 2, 4, 0))
    assert_case(results, "Case5 Peashooter+Peashooter legal",
                F.is_legal_fusion_action(board_observation((2, 4, PEASHOOTER)), 2, 4, 1))
    # Empty tile + Peashooter -> empty_tile (fusion illegal).
    assert_case(results, "Case6 empty tile -> empty_tile",
                F.get_fusion_illegal_reason(board_observation(None), 2, 4, 1) == "empty_tile")
    # fusion disabled / sun / cooldown reasons.
    assert_case(results, "fusion disabled reason",
                F.get_fusion_illegal_reason(board_observation((2, 4, SUNFLOWER)), 2, 4, 1, fusion_enabled=False)
                == "fusion_disabled")
    low_sun = board_observation((2, 4, PEASHOOTER), sun=0)
    assert_case(results, "insufficient sun reason for compatible pair",
                F.get_fusion_illegal_reason(low_sun, 2, 4, 3) == "insufficient_sun")


def run_mask_cases(results: List[Dict[str, Any]]) -> None:
    env_on = PvZGymEnv(PvZEnvConfig(plant_types=[1, 0, 3, 2], fusion_action_mask_enabled=True))
    env_off = PvZGymEnv(PvZEnvConfig(plant_types=[1, 0, 3, 2], fusion_action_mask_enabled=False))
    obs = board_observation((2, 4, SUNFLOWER))  # SunFlower occupies (2,4)
    fuse_compatible = encode(1, 2, 4)   # Peashooter seed -> SunFlower tile
    fuse_incompatible = encode(3, 2, 4)  # CherryBomb seed -> SunFlower tile
    plant_empty = encode(1, 0, 0)        # Peashooter on empty tile
    # Bridge legal actions intentionally EXCLUDE occupied-tile actions (fusion is
    # not in the bridge's normal placement set); they include the empty-tile plant.
    bridge = [0, plant_empty]

    def filter_result(env: PvZGymEnv, action: int, bridge_actions: List[int]) -> tuple[bool, str]:
        decision = env.action_decision(
            action,
            obs,
            source="fusion_compatibility_test",
            bridge_actions=bridge_actions,
        )
        return bool(decision.legal), str(decision.rejection_reason or "")

    decoded = decode_action(fuse_compatible, obs, [1, 0, 3, 2])
    on_compatible = filter_result(env_on, fuse_compatible, bridge)
    on_incompatible = filter_result(env_on, fuse_incompatible, bridge)
    on_plant_empty = filter_result(env_on, plant_empty, bridge)
    off_compatible = filter_result(env_off, fuse_compatible, bridge)
    obs["legalActions"] = list(bridge)
    complete_mask = env_on.action_mask(obs)

    assert_case(results, "mask exposes legal compatible fusion action",
                on_compatible == (True, ""), {"decoded": decoded, "result": on_compatible})
    assert_case(results, "mask blocks incompatible fusion action",
                on_incompatible == (False, "incompatible_pair"), on_incompatible)
    assert_case(results, "mask keeps normal plant on empty tile legal",
                on_plant_empty == (True, ""), on_plant_empty)
    assert_case(results, "occupied tile stays illegal when fusion mask disabled",
                off_compatible == (False, "occupied_cell"), off_compatible)
    assert_case(results, "complete action mask includes compatible fusion",
                bool(complete_mask[fuse_compatible]), complete_mask[fuse_compatible])
    assert_case(results, "complete action mask includes SunFlower self-fusion",
                bool(complete_mask[encode(0, 2, 4)]), complete_mask[encode(0, 2, 4)])
    assert_case(results, "complete action mask excludes incompatible fusion",
                not bool(complete_mask[fuse_incompatible]), complete_mask[fuse_incompatible])
    sunflower_self_fusion = filter_result(env_on, encode(0, 2, 4), [0, encode(0, 2, 4)])
    assert_case(results, "mask exposes SunFlower self-fusion",
                sunflower_self_fusion == (True, ""), sunflower_self_fusion)

    diag = env_on._fusion_mask_diagnostics(obs)
    assert_case(results, "mask diagnostics report available + incompatible counts",
                diag["fusion_actions_available_count"] == 2
                and diag["fusion_actions_masked_incompatible_count"] == 2
                and diag["fusion_candidate_tiles"] == [(2, 4)]
                and isinstance(diag["fusion_compatibility_table"], dict),
                diag)
    diag_off = env_off._fusion_mask_diagnostics(obs)
    assert_case(results, "mask diagnostics report disabled suppression when off",
                diag_off["fusion_actions_masked_disabled_count"] == 2
                and diag_off["fusion_actions_available_count"] == 0,
                diag_off)


def validate_fuse(cmd: str, obs: Dict[str, Any]):
    parsed = parse_coach_command(cmd)
    return validate_coach_command(
        parsed,
        action_space_mode=ACTION_SPACE_ADVENTURE_14_IDENTITY,
        observation=obs,
        plant_types=[1, 0, 3, 2],
        max_seed_slots=14,
        fusion_enabled=True,
        fusion_bridge_probe=None,
    )


def run_coach_cases(results: List[Dict[str, Any]]) -> None:
    sunflower_tile = board_observation((2, 4, SUNFLOWER))
    peashooter_tile = board_observation((2, 4, PEASHOOTER))

    # fuse 1 2 4 = Peashooter seed onto SunFlower -> compatible; passes compat gate
    # (only stopped later because this bridge-free test provides no fusion probe).
    compatible = validate_fuse("fuse 1 2 4", sunflower_tile)
    assert_case(results, "coach accepts compatible pair past the compatibility gate",
                compatible.rejected_reason != "incompatible_pair"
                and compatible.rejected_reason != "empty_tile",
                compatible.to_dict())

    sunflower_self = validate_fuse("fuse 0 2 4", sunflower_tile)
    assert_case(results, "coach accepts SunFlower+SunFlower past compatibility gate",
                sunflower_self.rejected_reason != "incompatible_pair",
                sunflower_self.to_dict())

    peashooter_self = validate_fuse("fuse 1 2 4", peashooter_tile)
    assert_case(results, "coach accepts Peashooter+Peashooter past compatibility gate",
                peashooter_self.rejected_reason != "incompatible_pair",
                peashooter_self.to_dict())

    # fuse 3 2 4 = CherryBomb seed onto SunFlower -> incompatible.
    incompatible = validate_fuse("fuse 3 2 4", sunflower_tile)
    pair = incompatible.diagnostics.get("fusion_incompatible_pair") if isinstance(incompatible.diagnostics, dict) else None
    assert_case(results, "coach rejects incompatible pair with incompatible_pair reason",
                incompatible.rejected_reason == "incompatible_pair"
                and isinstance(pair, dict)
                and pair.get("existing") == "SunFlower"
                and pair.get("selected") == "CherryBomb"
                and pair.get("row") == 2 and pair.get("col") == 4,
                incompatible.to_dict())

    # Peashooter tile + CherryBomb seed -> compatible (Case 3 via coach).
    case3 = validate_fuse("fuse 3 2 4", peashooter_tile)
    assert_case(results, "coach accepts Peashooter+CherryBomb past compatibility gate",
                case3.rejected_reason != "incompatible_pair", case3.to_dict())

    # Empty tile -> empty_tile.
    empty = validate_fuse("fuse 1 0 0", board_observation(None))
    assert_case(results, "coach rejects fuse on empty tile with empty_tile reason",
                empty.rejected_reason == "empty_tile", empty.to_dict())

    # Fusion disabled is still reported before compatibility.
    parsed = parse_coach_command("fuse 3 2 4")
    disabled = validate_coach_command(
        parsed,
        action_space_mode=ACTION_SPACE_ADVENTURE_14_IDENTITY,
        observation=sunflower_tile,
        plant_types=[1, 0, 3, 2],
        max_seed_slots=14,
        fusion_enabled=False,
        fusion_bridge_probe=None,
    )
    assert_case(results, "coach reports fusion_disabled before compatibility",
                disabled.rejected_reason == "fusion_disabled", disabled.to_dict())


def main() -> int:
    results: List[Dict[str, Any]] = []
    run_compatibility_table_cases(results)
    run_normalize_cases(results)
    run_legality_helper_cases(results)
    run_mask_cases(results)
    run_coach_cases(results)
    payload = {"ok": all(item["passed"] for item in results), "results": results}
    print(json.dumps(payload, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
