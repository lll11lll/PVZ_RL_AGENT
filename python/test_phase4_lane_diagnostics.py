"""Differential contracts for the pure lane-diagnostics compositor."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from typing import Any

import pytest

from pvzrl_env import PvZEnvConfig, PvZGymEnv
from pvzrl_lane_diagnostics import LaneDiagnosticsInput, compose_lane_diagnostics
from pvzrl_observation_facts import build_step_facts
from pvzrl_rewards import RewardCompositionState
from test_refactor_support import make_wrapper


ROWS, COLUMNS = 5, 10
PLANT_TYPES = [1, 0, 2, 3]


def _encode(slot: int, row: int, column: int) -> int:
    return 1 + slot * ROWS * COLUMNS + row * COLUMNS + column


def _observation() -> dict[str, Any]:
    legal = [0] + [_encode(1, row, 8) for row in range(ROWS)]
    return {
        "rowCount": ROWS,
        "columnCount": COLUMNS,
        "actionCount": 1 + len(PLANT_TYPES) * ROWS * COLUMNS,
        "boardFound": True,
        "canReadBoard": True,
        "gameplayReady": True,
        "sun": 500,
        "seedSlots": [
            {"slotIndex": 0, "plantType": 1, "seedCost": 50, "ready": True, "usable": True},
            {"slotIndex": 1, "plantType": 0, "seedCost": 100, "ready": True, "usable": True},
            {"slotIndex": 2, "plantType": 2, "seedCost": 150, "ready": True, "usable": True},
            {"slotIndex": 3, "plantType": 3, "seedCost": 50, "ready": True, "usable": True},
        ],
        "plants": [
            {"row": 1, "column": 1, "type": 1, "typeName": "SunFlower", "health": 300},
            {"row": 3, "column": 2, "type": 0, "typeName": "Peashooter", "health": 300},
        ],
        "zombies": [
            {"row": 1, "x": 5.5, "type": 4, "typeName": "Buckethead", "health": 900, "maxHealth": 900},
            {"row": 1, "x": 6.0, "type": 2, "typeName": "Conehead", "health": 500, "maxHealth": 500},
            {"row": 3, "x": 1.8, "type": 0, "typeName": "Zombie", "health": 200, "maxHealth": 200},
        ],
        "lanes": [
            {"row": 0, "zombieCount": 0, "danger": 0.0},
            {"row": 1, "zombieCount": 2, "danger": 0.82, "nearestZombieX": 5.5},
            {"row": 2, "zombieCount": 0, "danger": 0.0},
            {"row": 3, "zombieCount": 1, "danger": 0.7, "nearestZombieX": 1.8},
            {"row": 4, "zombieCount": 0, "danger": 0.0},
        ],
        "visibleMowers": [
            {"row": row, "activeInHierarchy": True, "inBoardBounds": True, "inMowerArray": True}
            for row in range(ROWS)
        ],
        "logicalMowerCount": ROWS,
        "visibleMowerObjectCount": ROWS,
        "legalActions": legal,
        "legalActionCount": len(legal),
    }


def _placement(plant_type: int, row: int = 1, column: int = 4) -> dict[str, Any]:
    return {
        "decoded": {"kind": "plant", "plantType": plant_type, "row": row, "column": column},
        "placement": {"success": True, "plantType": plant_type, "row": row, "column": column},
    }


CASES = {
    "wait": {"decoded": {"kind": "wait"}},
    "peashooter": _placement(0, column=2),
    "wallnut": _placement(3),
    "cherrybomb": _placement(2),
    "fusion": {
        "decoded": {"kind": "fusion", "resultPlantType": 1030, "row": 1, "column": 1},
        "fusionSucceeded": True,
    },
    "illegal_cooldown": {
        **_placement(0, column=2),
        "illegalAction": True,
        "illegalReason": "cooldown",
        "preStepMaskBlockedAction": True,
        "preStepMaskAudit": {"pythonFilterReason": "cooldown"},
        "actionAudit": {"pythonMaskValueBefore": True, "bridgeLegalActionsValueBefore": False},
    },
}

# Captured from the legacy PvZGymEnv.lane_diagnostics implementation before
# its Phase 4 replacement with the pure compositor. These hashes keep the full
# 94-field payload contract independent of the compatibility wrapper.
CANONICAL_PAYLOAD_SHA256 = {
    "wait": "36ed465fe8066dc8e9010efcc0100b5ca17a7dd0bf08793a34e6ad587e2ef098",
    "peashooter": "154aa630c7d6a23703c0cadb26418a0bad224f45df0afa41d0321355923f9091",
    "wallnut": "447c1554af9f04e9ab1bf38616fff3866e3dd5c0b4124124e79ff5d074a7c22a",
    "cherrybomb": "63fe26386b289e084857b976965dfd45adb228c40a8036cc9ffae7675d101efb",
    "fusion": "48be976f338367ba6dc352929b854942ba04d1750e95ca40c36476802ef67e15",
    "illegal_cooldown": "e7fdc3bfd2384dd277f2d07dd165121234813cbb242250263489dece10c9f315",
}


def _canonical_payload_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assert_exact_types(actual: Any, expected: Any) -> None:
    assert type(actual) is type(expected)
    if isinstance(expected, dict):
        assert [(type(key), key) for key in actual] == [(type(key), key) for key in expected]
        for key, value in expected.items():
            _assert_exact_types(actual[key], value)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for actual_value, expected_value in zip(actual, expected):
            _assert_exact_types(actual_value, expected_value)
    else:
        assert actual == expected


def _compare(case_name: str) -> dict[str, Any]:
    env = PvZGymEnv(PvZEnvConfig(plant_types=list(PLANT_TYPES)))
    previous = _observation()
    current = copy.deepcopy(previous)
    current["lanes"][1]["danger"] = 0.76
    current["visibleMowers"] = [item for item in current["visibleMowers"] if item["row"] != 1]
    current["logicalMowerCount"] = ROWS - 1
    current["visibleMowerObjectCount"] = ROWS - 1
    action_result = copy.deepcopy(CASES[case_name])
    if case_name in {"peashooter", "wallnut", "cherrybomb"}:
        current["plants"].append(
            {
                "row": 1,
                "column": int(action_result["decoded"]["column"]),
                "type": int(action_result["decoded"]["plantType"]),
            }
        )

    state = replace(
        env._reward_state,
        undefended_threat_age_by_row=(0, 4, 0, 0, 0),
        max_undefended_threat_age_by_row=(0, 7, 0, 2, 0),
        undefended_threat_age_sum_by_row=(0, 9, 0, 2, 0),
        undefended_threat_age_count_by_row=(0, 4, 0, 2, 0),
    )
    env._reward_state = state
    current_legal = tuple(int(value) for value in current["legalActions"])
    previous_legal = tuple(int(value) for value in previous["legalActions"])
    mask = {
        "legal_actions_by_seed_slot": {"0": 0, "1": 5, "2": 2, "3": 1},
        "bridge_legal_actions_by_seed_slot": {"0": 0, "1": 4, "2": 2, "3": 1},
        "python_mask_block_reason_counts": {"occupied": 3, "cooldown": 1},
        "tactical_mask_enabled": True,
        "wallnut_tactical_mask_enabled": True,
        "cherrybomb_tactical_mask_enabled": False,
        "wallnut_actions_masked": 2,
        "cherrybomb_actions_masked": 0,
        "wallnut_actions_available": 1,
        "cherrybomb_actions_available": 2,
        "mask_all_but_wait_count": 0,
    }
    cherry = {"kills": 2, "zero_kill": 0, "buckethead": 1, "conehead": 1}
    prior_facts = build_step_facts(previous, PLANT_TYPES)
    current_facts = build_step_facts(current, PLANT_TYPES)

    # Explicit snapshots are mandatory: neither implementation may fall back
    # to an action-mask/legal-action query in this differential test.
    env.legal_actions = lambda *_args, **_kwargs: pytest.fail("unexpected legal_actions call")
    env.action_mask = lambda *_args, **_kwargs: pytest.fail("unexpected action_mask call")
    env.mask_diagnostics = lambda *_args, **_kwargs: pytest.fail("unexpected mask_diagnostics call")
    expected = env.lane_diagnostics(
        previous,
        current,
        action_result,
        list(current_legal),
        cherry_delayed_diagnostics=cherry,
        previous_facts=prior_facts,
        current_facts=current_facts,
        previous_legal_actions=list(previous_legal),
        mask_diagnostics_snapshot=mask,
    )
    inputs = LaneDiagnosticsInput(
        previous_facts=prior_facts,
        current_facts=current_facts,
        post_reward_state=state,
        current_legal_actions=current_legal,
        previous_legal_actions=previous_legal,
        action_result=action_result,
        mask_diagnostics=mask,
        cherry_delayed_diagnostics=cherry,
        fallback_rows=env.config.row_count,
        fallback_columns=env.config.column_count,
        close_threat_threshold=env.config.reward.close_threat_threshold,
    )
    before = copy.deepcopy((action_result, mask, cherry))
    actual = compose_lane_diagnostics(inputs)
    assert compose_lane_diagnostics(inputs) == actual
    assert (action_result, mask, cherry) == before
    _assert_exact_types(actual, expected)
    return actual


@pytest.mark.parametrize("case_name", CASES)
def test_pure_lane_diagnostics_matches_legacy(case_name: str) -> None:
    diagnostics = _compare(case_name)
    assert len(diagnostics) == 94
    assert _canonical_payload_sha256(diagnostics) == CANONICAL_PAYLOAD_SHA256[case_name]
    if case_name == "wait":
        assert diagnostics["wait_under_threat"] is True
    elif case_name == "peashooter":
        assert diagnostics["lane_response_reward_applied"] is True
        assert diagnostics["first_defense_row"] == 1
    elif case_name == "wallnut":
        assert diagnostics["wallnut_blocks_active_threat"] is True
        assert diagnostics["wallnut_emergency_block"] is True
    elif case_name == "cherrybomb":
        assert diagnostics["cherrybomb_cluster_use"] is True
        assert diagnostics["cherrybomb_buckethead_kill_credit"] == 1
    elif case_name == "fusion":
        assert diagnostics["tough_zombie_response"] is True
    else:
        assert diagnostics["plant_placed"] is True
        assert diagnostics["lane_response_reward_applied"] is False
        assert diagnostics["cooldown_illegal_exposed_by_mask"] is True
        assert diagnostics["mask_bridge_disagreement"] is True


def test_lane_diagnostics_input_is_frozen() -> None:
    observation = _observation()
    facts = build_step_facts(observation, PLANT_TYPES)
    inputs = LaneDiagnosticsInput(
        previous_facts=facts,
        current_facts=facts,
        post_reward_state=RewardCompositionState.initial(5),
    )
    with pytest.raises(FrozenInstanceError):
        inputs.fallback_rows = 7  # type: ignore[misc]


def test_episode_lane_aggregation_does_not_evaluate_present_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = make_wrapper()
    observation = _observation()
    info = {
        "lane_diagnostics": {
            "plants_by_row": {"1": 2},
            "peashooters_by_row": {"3": 1},
            "sunflowers_by_row": {"1": 1},
        }
    }
    monkeypatch.setattr(
        wrapper,
        "_plant_counts_by_row",
        lambda *_args, **_kwargs: pytest.fail("eager row-count fallback evaluated"),
    )
    try:
        wrapper._record_lane_diagnostics(0, observation, info)
        assert wrapper._episode_final_plants_by_row == {1: 2}
        assert wrapper._episode_final_peashooters_by_row == {3: 1}
        assert wrapper._episode_final_sunflowers_by_row == {1: 1}
    finally:
        wrapper.close()
