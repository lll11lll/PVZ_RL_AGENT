"""Adventure Generalist compatibility locks for the maintained repository path."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import numpy as np

from pvzrl_action_space import (
    ACTION_SPACE_ADVENTURE_14_IDENTITY,
    ADVENTURE_IDENTITY_ACTION_DECODER_VERSION,
    ADVENTURE_IDENTITY_OBSERVATION_VERSION,
    build_action_space_spec,
    decode_policy_action,
    structural_adventure_identity_mask,
)
from pvzrl_adventure import build_live_status
from pvzrl_env import REWARD_COMPONENT_FIELDS
from pvzrl_fusion import (
    FUSION_ILLEGAL_BRIDGE_UNAVAILABLE,
    FUSION_ILLEGAL_COOLDOWN,
    FUSION_ILLEGAL_DISABLED,
    FUSION_ILLEGAL_EMPTY_TILE,
    FUSION_ILLEGAL_INCOMPATIBLE,
    FUSION_ILLEGAL_INSUFFICIENT_SUN,
    FUSION_ILLEGAL_INVALID_COL,
    FUSION_ILLEGAL_INVALID_ROW,
    FUSION_ILLEGAL_INVALID_SEED_SLOT,
    FUSION_RULES,
    build_fusion_diagnostics,
    default_fusion_diagnostics,
    get_fusion_illegal_reason,
)
from test_refactor_support import (
    ROOT,
    array_sha256,
    json_sha256,
    load_observation_fixture,
    make_wrapper,
    mask_sha256,
    observation_for_wrapper,
)
from train_ppo import EPISODE_METRIC_FIELDS, clean_episode_row


IDENTITY_VECTOR_SHA = "7c1f3e65fce9e6902f1a7ed0ff356684ce178ae48a43d2de9f968c008c54cfcf"
IDENTITY_MASK_SHA = "764737ef8796041925e1cdf007ffa9b5cc6608684066e4621fb0680530c62032"
ALLOWED_FIXTURE_ACTIONS = [0, 1, 2, 10, 24, 121, 144, 170]
REWARD_BREAKDOWN_SHA = "7e7df41261d99d14ec18746012e9177795f0eed7f1aa6e48b2cd33dd8f6bfa8f"
FUSION_CANDIDATES_SHA = "8b3f99050f1e56262bb9af0e59b3826811234c14f1af95477d78d03356913853"
REWARD_FIELDS_SHA = "5c92950f37796911b476b8a45c9e6bb63396062d85f3bf76e6607e08c92a8376"
EPISODE_FIELDS_SHA = "416cc3b88d36cb83124ddac6f657ab7fc810003e05351e1642d1aa083b298f41"
FUSION_FIELDS_SHA = "3fe73b60ad282971dd505f727b02808bd3ef48534498a1efd3a10da0eba95b2f"
LIVE_TOP_LEVEL_KEYS_SHA = "0094f3a33cdd98457ec003af5ac84967e71e9971238a6f38b8ab46d52bba56ed"
OBSERVATION_DTO_SHA = "72e380bceb9d78cd80310a275c6dd7a6cf208f600d48ac114aef7ccb90b957ae"


def test_generalist_action_space_count_versions_and_decoder() -> None:
    identity = build_action_space_spec(
        mode=ACTION_SPACE_ADVENTURE_14_IDENTITY,
        plant_types=[1, 1, 0, 0],
        max_seed_slots=14,
    )
    assert (identity.action_count, identity.wait_action, identity.placement_action_min, identity.placement_action_max) == (841, 0, 1, 840)
    assert (identity.action_decoder_version, identity.observation_version) == (
        ADVENTURE_IDENTITY_ACTION_DECODER_VERSION,
        ADVENTURE_IDENTITY_OBSERVATION_VERSION,
    )

    for action in range(identity.action_count):
        decoded = decode_policy_action(
            action,
            mode=identity.mode,
            plant_types=[1, 1, 0, 0],
            max_seed_slots=identity.max_seed_slots,
        )
        if action == identity.wait_action:
            assert decoded["kind"] == 0
            continue
        assert decoded["kind"] == 1
        assert 0 <= decoded["slot_index"] < identity.max_seed_slots
        assert 0 <= decoded["row"] < 6
        assert 0 <= decoded["column"] < 10


def test_structural_masks_keep_wait_and_inactive_slots_stable() -> None:
    identity = structural_adventure_identity_mask(4)
    assert len(identity) == 841
    assert sum(bool(value) for value in identity) == 241
    assert identity[0] and all(identity[index] for index in range(1, 241))
    assert not any(identity[index] for index in range(241, 841))


def test_generalist_observation_vector_is_exact() -> None:
    identity = make_wrapper()
    try:
        identity_observation = observation_for_wrapper(identity)
        identity_vector = identity._encode_observation(identity_observation)
        assert identity_vector.shape == (4364,)
        assert identity_vector.dtype == np.float32
        assert array_sha256(identity_vector) == IDENTITY_VECTOR_SHA
    finally:
        identity.close()


def test_complete_fixture_masks_lock_sun_cooldown_occupancy_and_fusion() -> None:
    identity = make_wrapper()
    try:
        identity._last_observation = observation_for_wrapper(identity)
        identity_mask = identity.action_masks()
        assert identity_mask.shape == (841,)
        assert np.flatnonzero(identity_mask).tolist() == ALLOWED_FIXTURE_ACTIONS
        assert mask_sha256(identity_mask) == IDENTITY_MASK_SHA
        assert identity_mask[0]
        assert not identity_mask[61]  # duplicate SunFlower slot is cooling down
        assert not identity_mask[204]  # duplicate Peashooter slot is unaffordable
        assert identity_mask[144]  # occupied Peashooter tile is a legal self-fusion
        assert not identity_mask[840]  # inactive slot 13 remains masked
    finally:
        identity.close()


def test_fusion_recipes_reason_order_and_candidate_snapshot() -> None:
    expected_results = {
        (0, 0): 1030,
        (1030, 0): 1090,
        (1090, 0): 1032,
        (1, 1): 1033,
    }
    assert {pair: int(rule["predicted_result_type"]) for pair, rule in FUSION_RULES.items()} == expected_results

    observation = load_observation_fixture()
    common = {"fusion_enabled": True, "fusion_bridge_available": True}
    assert get_fusion_illegal_reason(observation, -1, 99, 99, fusion_enabled=False) == FUSION_ILLEGAL_DISABLED
    assert get_fusion_illegal_reason(observation, -1, 99, 99, fusion_bridge_available=False) == FUSION_ILLEGAL_BRIDGE_UNAVAILABLE
    assert get_fusion_illegal_reason(observation, -1, 0, 0, **common) == FUSION_ILLEGAL_INVALID_ROW
    assert get_fusion_illegal_reason(observation, 0, 10, 0, **common) == FUSION_ILLEGAL_INVALID_COL
    assert get_fusion_illegal_reason(observation, 0, 0, 99, **common) == FUSION_ILLEGAL_INVALID_SEED_SLOT
    assert get_fusion_illegal_reason(observation, 3, 3, 0, **common) == FUSION_ILLEGAL_EMPTY_TILE
    assert get_fusion_illegal_reason(observation, 4, 9, 0, **common) == FUSION_ILLEGAL_INCOMPATIBLE
    assert get_fusion_illegal_reason(observation, 0, 0, 1, **common) == FUSION_ILLEGAL_COOLDOWN
    assert get_fusion_illegal_reason(observation, 2, 3, 3, **common) == FUSION_ILLEGAL_INSUFFICIENT_SUN
    assert get_fusion_illegal_reason(observation, 0, 0, 0, **common) == ""

    diagnostics = build_fusion_diagnostics("observe", observation)
    assert diagnostics["fusion_candidate_count"] == 12
    assert json_sha256(diagnostics["fusion_candidates"]) == FUSION_CANDIDATES_SHA


def test_reward_components_and_total_are_numerically_locked() -> None:
    wrapper = make_wrapper()
    try:
        previous = observation_for_wrapper(wrapper)
        current = copy.deepcopy(previous)
        current["killCount"] = 8
        current["wave"] = 4
        current["time"] = 13.0
        breakdown = wrapper.base.compute_reward_breakdown(
            previous,
            current,
            {"ok": True, "plantPlaced": False, "kind": "wait"},
        )
        assert set(breakdown) == set(REWARD_COMPONENT_FIELDS) | {"reward_total"}
        assert json_sha256(breakdown) == REWARD_BREAKDOWN_SHA
        assert abs(float(breakdown["reward_total"]) - 2.25) <= 1e-12
        assert abs(
            float(breakdown["reward_total"])
            - sum(float(breakdown[field]) for field in REWARD_COMPONENT_FIELDS)
        ) <= 1e-12
    finally:
        wrapper.close()


def test_episode_reward_fusion_and_live_status_schema_snapshots() -> None:
    assert json_sha256(list(REWARD_COMPONENT_FIELDS)) == REWARD_FIELDS_SHA
    assert json_sha256(list(EPISODE_METRIC_FIELDS)) == EPISODE_FIELDS_SHA
    assert json_sha256(sorted(default_fusion_diagnostics("observe"))) == FUSION_FIELDS_SHA
    assert list(clean_episode_row({}, fallback_episode=0)) == list(EPISODE_METRIC_FIELDS)

    wrapper = make_wrapper()
    try:
        wrapper._last_observation = observation_for_wrapper(wrapper)
        payload = build_live_status(
            wrapper,
            {"mode": "contract", "phase": "fixture", "episode": 1, "step": 2},
            {
                "availableSeedNames": ["SunFlower", "Peashooter"],
                "unlockedSeedNames": ["SunFlower", "Peashooter"],
            },
            {},
        )
        assert json_sha256(sorted(payload)) == LIVE_TOP_LEVEL_KEYS_SHA
        assert "schema_version" not in payload  # baseline; Phase 4 may migrate this deliberately
        for section in ("gameplay", "agent", "reward", "coach", "compatibility", "seed_inventory", "fusion", "adventure", "rows"):
            assert isinstance(payload.get(section), dict)
        assert payload["compatibility"]["action_count"] == 841
        assert payload["agent"]["observation_version"] == ADVENTURE_IDENTITY_OBSERVATION_VERSION
    finally:
        wrapper.close()


def test_bridge_observation_dto_property_names_and_types_are_locked() -> None:
    bridge_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "src" / "PvZRLBridge").glob("*.cs"))
    )
    match = re.search(r"internal sealed class ObservationDto\s*\{(.*?)\n\}", bridge_text, re.DOTALL)
    assert match is not None
    properties = sorted(
        [property_type, property_name]
        for property_type, property_name in re.findall(
            r"public\s+([A-Za-z0-9_?<>,.\[\]]+)\s+(\w+)\s*\{\s*get;\s*(?:set;\s*)?\}",
            match.group(1),
            re.DOTALL,
        )
    )
    assert len(properties) == 122
    assert json_sha256(properties) == OBSERVATION_DTO_SHA
    assert ["List<PlantDto>", "Plants"] in properties
    assert ["List<SeedSlotDto>", "SeedSlots"] in properties
    assert ["List<int>", "LegalActions"] in properties
    assert ["long", "SunSpawnCompensationApplyCount"] in properties


def test_protected_generalist_model_metadata_contract_fixture() -> None:
    contracts_path = ROOT / "python" / "fixtures" / "refactor_contracts" / "model_contracts.json"
    contracts = json.loads(contracts_path.read_text(encoding="utf-8"))
    assert contracts["schema_version"] == 3
    protected = contracts["protected_generalist"]
    assert (
        protected["action_count"],
        protected["max_seed_slots"],
        protected["decoder_wait_action"],
        protected["placement_action_range"],
        protected["rows"],
        protected["cols"],
    ) == (841, 14, 0, [1, 840], 6, 10)
    assert protected["action_decoder_version"] == ADVENTURE_IDENTITY_ACTION_DECODER_VERSION
    assert protected["observation_version"] == ADVENTURE_IDENTITY_OBSERVATION_VERSION
    assert protected["identity_seed_slots"] is True
    assert protected["observation_shape"] == [4364]
    assert protected["num_timesteps"] == 0
    assert protected["seed_list"] == ["SunFlower", "SunFlower", "Peashooter", "Peashooter"]
    assert protected["plant_types"] == [1, 1, 0, 0]
