from __future__ import annotations

import copy
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import pvzrl_actions
import pvzrl_fusion
import pvzrl_observation_facts
import pvzrl_env

from pvzrl_actions import (
    ActionValidationConfig,
    build_action_validation_context,
)
from pvzrl_fusion import (
    build_fusion_diagnostics,
    count_tough_zombies_by_row,
    fusion_intent_from_candidate,
    get_fusion_illegal_reason,
    plant_type_at_cell,
    scan_fusion_candidates,
    validate_fusion_intent,
)
from pvzrl_observation_facts import StepFactsCache, build_step_facts, observation_identity
from pvzrl_env import PvZEnvConfig, PvZGymEnv


FIXTURE = Path(__file__).parent / "fixtures" / "refactor_contracts" / "synthetic_observation.json"


def _observation() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _legacy_occupancy(observation: dict) -> tuple[tuple[int, int, int], ...]:
    occupied: dict[tuple[int, int], int] = {}
    for key in ("plants", "visiblePlants"):
        values = observation.get(key, [])
        for plant in values if isinstance(values, list) else []:
            if not isinstance(plant, dict):
                continue
            if key == "visiblePlants" and (
                not bool(plant.get("activeInHierarchy", True))
                or not bool(plant.get("inBoardBounds", True))
            ):
                continue
            try:
                row = int(plant.get("row", -1))
                column = int(plant.get("column", -1))
                plant_type = int(plant.get("type", plant.get("plantType", -1)))
            except (TypeError, ValueError):
                continue
            occupied.setdefault((row, column), plant_type)
    return tuple((row, column, plant_type) for (row, column), plant_type in occupied.items())


def test_fixture_builds_all_reusable_indexes_without_contract_changes() -> None:
    observation = _observation()
    facts = build_step_facts(observation, (1, 1, 0, 0))

    assert (facts.rows, facts.columns, facts.action_count, facts.sun) == (5, 10, 701, 175)
    assert facts.identity.revision == "1234"
    assert len(facts.identity.content_digest) == 64
    assert facts.occupancy == ((0, 0, 1), (2, 3, 0), (4, 9, 1030))
    assert facts.occupied_cells == frozenset({(0, 0), (2, 3), (4, 9)})
    assert [plant.plant_type for plant in facts.plants_by_lane[4]] == [1030]
    assert [plant.cell for plant in facts.plants_by_type[0]] == [(2, 3)]
    assert [zombie.x for zombie in facts.zombies_by_lane[4]] == [4.0]
    assert facts.nearest_zombie_by_lane[1].x == 7.5
    assert facts.lane_by_row[4].danger == pytest.approx(0.6)
    assert facts.total_lane_danger == pytest.approx(0.85)
    assert facts.seed_slot(2).plant_type == 0
    assert facts.seed_slot(1).ready is False
    assert facts.seed_slot(1).current_cooldown > 0.05
    assert facts.seed_slot(3).affordable is False
    assert facts.ready_seed_types == frozenset({0, 1})
    assert facts.mower.count == 5 and facts.mower.active_rows is None
    assert facts.lifecycle.gameplay_ready and facts.lifecycle.screen_state == "gameplay"
    assert not facts.safety.plant_count_mismatch
    assert not facts.safety.zombie_count_mismatch


def test_primary_counts_and_visible_fallback_occupancy_are_separate() -> None:
    observation = {
        "rowCount": 3,
        "columnCount": 4,
        "plants": [
            {"row": 0, "column": 0, "type": 1},
            {"row": 0, "column": 0, "type": 2},
            {"row": 2, "column": 3, "type": 3},
        ],
        "visiblePlants": [
            {"row": 0, "column": 0, "type": 4},
            {"row": 1, "column": 1, "type": 5},
            {"row": 1, "column": 2, "type": 6, "activeInHierarchy": False},
            {"row": 1, "column": 3, "type": 7, "inBoardBounds": False},
        ],
    }
    facts = build_step_facts(observation)

    assert len(facts.plants) == 3
    assert len(facts.visible_plants) == 2
    assert [plant.plant_type for plant in facts.plants_by_lane[0]] == [1, 2]
    assert 1 not in facts.plants_by_lane
    assert [plant.plant_type for plant in facts.primary_plant_stacks_by_cell[(0, 0)]] == [1, 2]
    assert facts.occupant_by_cell[(0, 0)].plant_type == 1
    assert facts.last_primary_plant_by_cell[(0, 0)].plant_type == 2
    assert facts.occupant_by_cell[(1, 1)].plant_type == 5
    assert (1, 2) not in facts.occupied_cells and (1, 3) not in facts.occupied_cells
    assert facts.safety.duplicate_primary_cells == ((0, 0),)
    assert facts.occupancy == _legacy_occupancy(observation)

    col_alias = {"plants": [{"row": 2, "col": 3, "type": 8}]}
    alias_facts = build_step_facts(col_alias)
    assert alias_facts.occupancy == ((2, -1, 8),)
    assert plant_type_at_cell(col_alias, 2, 3, facts=alias_facts) == 8


def test_seed_defaults_preserve_distinct_action_scan_and_fusion_semantics() -> None:
    fallback = (1, 0)
    facts = build_step_facts({"sun": 0}, fallback)
    first = facts.seed_slot(0)
    assert first is not None and first.synthetic
    assert not first.usable and not first.ready
    assert first.fusion_usable and first.fusion_ready
    assert not first.legacy_ready  # Candidate scans historically require an explicit ready=true.

    missing_flags = build_step_facts(
        {"sun": 100, "seedSlots": [{"slotIndex": 4, "plantType": 1, "seedCost": 50}]}
    ).seed_slot(4)
    assert missing_flags is not None
    assert not missing_flags.usable and not missing_flags.ready
    assert missing_flags.fusion_usable and missing_flags.fusion_ready
    assert not missing_flags.legacy_ready


def test_snapshot_is_deeply_immutable_and_detached_from_mutable_input() -> None:
    observation = _observation()
    facts = build_step_facts(observation)
    original_sun = facts.sun
    original_health = facts.plants[0].health
    original_slot_cost = facts.seed_slots[0].seed_cost

    observation["sun"] = 999
    observation["plants"][0]["health"] = -1
    observation["seedSlots"][0]["seedCost"] = 999
    assert (facts.sun, facts.plants[0].health, facts.seed_slots[0].seed_cost) == (
        original_sun,
        original_health,
        original_slot_cost,
    )
    with pytest.raises(FrozenInstanceError):
        facts.plants[0].health = 10
    with pytest.raises(TypeError):
        facts.occupant_by_cell[(0, 0)] = facts.plants[0]
    with pytest.raises(TypeError):
        facts.plants_by_lane[0] = ()


def test_content_cache_reuses_equal_payload_and_invalidates_same_dict_mutation() -> None:
    observation = _observation()
    cache = StepFactsCache()
    first = cache.get(observation, (1, 1, 0, 0))
    second = cache.get(observation, (1, 1, 0, 0))
    equal_observation = copy.deepcopy(observation)
    equal_copy = cache.get(equal_observation, (1, 1, 0, 0))
    assert first is second is equal_copy
    assert (cache.hits, cache.misses) == (2, 1)
    # A verified content-equal mapping becomes the trusted owner; nested
    # consumers must not hash it again.
    assert cache.get_known(equal_observation, (1, 1, 0, 0)) is first
    assert (cache.hits, cache.misses) == (3, 1)

    observation["seedSlots"][0]["currentCooldown"] = 4.0
    mutated = cache.get(observation, (1, 1, 0, 0))
    assert mutated is not first
    assert mutated.identity.content_digest != first.identity.content_digest
    assert (cache.hits, cache.misses) == (3, 2)

    changed_fallback = cache.get(observation, (1, 0))
    assert changed_fallback is not mutated
    assert changed_fallback.identity == mutated.identity
    cache.clear()
    assert cache.get(observation, (1, 0)) is not changed_fallback


def test_trusted_cache_reuse_requires_the_retained_mapping_object() -> None:
    cache = StepFactsCache()
    original = {"frameCount": 1, "sun": 10}
    original_facts = cache.get(original)
    replacement = {"frameCount": 2, "sun": 999}

    replacement_facts = cache.get_known(replacement)

    assert replacement_facts is not original_facts
    assert replacement_facts.sun == 999
    # Retaining and comparing the mapping object prevents recycled integer ids
    # from ever authorizing trusted reuse.
    assert cache._object is replacement


def test_identity_changes_for_nested_content_even_when_frame_is_unchanged() -> None:
    observation = _observation()
    before = observation_identity(observation)
    observation["plants"][1]["health"] -= 1
    after = observation_identity(observation)
    assert before.revision == after.revision == "1234"
    assert before.content_digest != after.content_digest


def test_action_context_reuses_snapshot_and_matches_legacy_views() -> None:
    observation = _observation()
    fallback = (1, 1, 0, 0)
    facts = build_step_facts(observation, fallback)
    config = ActionValidationConfig(
        action_space_mode="fixed",
        plant_types=fallback,
        max_seed_slots=14,
        rows=5,
        cols=10,
    )
    context = build_action_validation_context(
        observation,
        config=config,
        bridge_legal_actions=observation["legalActions"],
        facts=facts,
    )
    assert context.facts is facts
    assert context.seed_slots is facts.seed_slots
    assert context.occupancy == _legacy_occupancy(observation)


def test_fusion_helpers_accept_one_snapshot_with_byte_equivalent_results() -> None:
    observation = _observation()
    facts = build_step_facts(observation)
    kwargs = {"fusion_bridge_available": True, "check_seed_resources": True}
    for row, column, slot in ((0, 0, 0), (0, 0, 1), (2, 3, 3), (4, 9, 0), (3, 3, 0)):
        assert get_fusion_illegal_reason(observation, row, column, slot, **kwargs) == get_fusion_illegal_reason(
            observation,
            row,
            column,
            slot,
            facts=facts,
            **kwargs,
        )
    assert scan_fusion_candidates(observation) == scan_fusion_candidates(observation, facts=facts)
    assert build_fusion_diagnostics("observe", observation) == build_fusion_diagnostics(
        "observe", observation, facts=facts
    )
    assert count_tough_zombies_by_row(observation) == count_tough_zombies_by_row(
        observation, facts=facts
    )


def test_fusion_validation_with_facts_never_rebuilds_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    observation = _observation()
    facts = build_step_facts(observation)
    candidate = scan_fusion_candidates(observation, facts=facts)[0]
    intent = fusion_intent_from_candidate(candidate, source="phase4_contract")

    def fail_rebuild(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("validate_fusion_intent rebuilt supplied facts")

    monkeypatch.setattr(pvzrl_fusion, "build_step_facts", fail_rebuild)
    validate_fusion_intent(intent, observation, facts=facts)


def test_localized_tough_zombie_names_match_legacy_and_fact_paths() -> None:
    observation = {
        "rowCount": 2,
        "columnCount": 10,
        "zombies": [
            {"row": 0, "type": -1, "typeName": "路障僵尸", "alive": True},
            {"row": 1, "type": -1, "typeName": "铁桶僵尸", "alive": True},
        ],
    }
    facts = build_step_facts(observation)
    assert count_tough_zombies_by_row(observation) == count_tough_zombies_by_row(
        observation,
        facts=facts,
    ) == {
        0: {"tough": 1, "buckethead": 0, "conehead": 1},
        1: {"tough": 1, "buckethead": 1, "conehead": 0},
    }


def test_tactical_mask_hashes_observation_once_at_owner_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = {
        "frameCount": 11,
        "rowCount": 5,
        "columnCount": 10,
        "actionCount": 201,
        "gameplayReady": True,
        "boardFound": True,
        "sun": 500,
        "legalActions": list(range(201)),
        "seedSlots": [
            {
                "slotIndex": index,
                "plantType": plant_type,
                "ready": True,
                "usable": True,
                "seedCost": 50,
            }
            for index, plant_type in enumerate((1, 0, 2, 3))
        ],
        "plants": [{"row": 2, "column": 1, "type": 0}],
        "zombies": [{"row": 2, "x": 5.0, "alive": True}],
        "lanes": [{"row": 2, "zombieCount": 1, "danger": 0.8}],
        "mowerCount": 5,
    }
    env = PvZGymEnv(
        PvZEnvConfig(
            plant_types=[1, 0, 2, 3],
            tactical_masks=True,
            fusion_action_mask_enabled=False,
        )
    )
    calls = 0
    original = pvzrl_observation_facts.observation_identity

    def counted(observation_payload: object) -> object:
        nonlocal calls
        calls += 1
        return original(observation_payload)  # type: ignore[arg-type]

    monkeypatch.setattr(pvzrl_observation_facts, "observation_identity", counted)
    monkeypatch.setattr(
        pvzrl_env,
        "seed_slots_from_observation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("tactical mask rescanned raw seed slots")
        ),
    )
    try:
        mask = env.action_mask(observation)
        assert len(mask) == 201
        assert calls == 1
        monkeypatch.setattr(
            pvzrl_actions.ActionDecisionCache,
            "decision_for",
            lambda *_args, **_kwargs: pytest.fail(
                "mask diagnostics cloned cached action decisions"
            ),
        )
        diagnostics = env.mask_diagnostics(observation, mask)
        assert diagnostics["python_legal_action_count"] == sum(mask)
        assert calls == 2
    finally:
        env.close()


def test_malformed_entries_and_explicit_slot_index_fallback_are_stable() -> None:
    observation = {
        "rowCount": 2,
        "columnCount": 2,
        "plantCount": 2,
        "zombieCount": 2,
        "plants": [None, {"row": 0, "column": 0, "type": 1}],
        "visiblePlants": ["bad", {"row": 1, "column": 1, "plantType": 3}],
        "zombies": [False, {"row": 9, "type": 2, "x": 1}],
        "seedSlots": [None, {"slotIndex": 7, "plantType": 0}],
        "visibleMowers": [
            {"row": 0},
            {"row": 1, "inMowerArray": False},
            {"row": 3},
        ],
    }
    facts = build_step_facts(observation)
    assert facts.safety.malformed_plant_count == 1
    assert facts.safety.malformed_visible_plant_count == 1
    assert facts.safety.malformed_zombie_count == 1
    assert facts.safety.malformed_seed_slot_count == 1
    assert facts.safety.plant_count_mismatch and facts.safety.zombie_count_mismatch
    assert facts.safety.invalid_zombie_count == 1
    assert facts.seed_slot(7).plant_type == 0
    assert facts.seed_slot(0) is None
    assert facts.seed_slot(0, positional_fallback=False) is None
    assert facts.mower.active_rows == frozenset({0, 3})


def test_fusion_legacy_defaults_survive_typed_plant_and_zombie_facts() -> None:
    observation = {
        "rowCount": 5,
        "columnCount": 10,
        "sun": 100,
        "plants": [{"row": 2, "column": 4, "type": 1, "typeName": "SunFlower"}],
        "seedSlots": [
            {
                "slotIndex": 0,
                "plantType": 0,
                "plantTypeName": "Peashooter",
                "ready": True,
                "usable": True,
                "seedCost": 100,
            }
        ],
        "zombies": [{"row": 2, "type": 2, "alive": True}],
        "lanes": [{"row": 2, "zombieCount": 1, "nearestZombieX": 7.0}],
    }
    facts = build_step_facts(observation)
    candidate = scan_fusion_candidates(observation, facts=facts)[0]
    # Legacy fusion scoring treated missing plant health as full health and a
    # missing zombie x-position as the lane's nearestZombieX.
    assert candidate["source_health_ratio"] == 1.0
    assert candidate["nearby_zombie_count"] == 1
    assert candidate["nearest_zombie_distance"] == 3.0
