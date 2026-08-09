"""Regression coverage for runtime unlock discovery and rotating loadouts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from pvzrl_adventure import update_unlocked_from_state
from pvzrl_adventure_generalist import (
    ADVENTURE_GENERALIST_INITIAL_LOADOUT,
    AdventureGeneralistTrainingEnv,
    AdventureSeedCurriculum,
    _filter_supported_seed_names,
    _selectable_from_seed_screen_state,
)
from pvzrl_action_space import (
    ACTION_SPACE_ADVENTURE_14_IDENTITY,
    build_action_space_spec,
)
from pvzrl_env import plant_type_name, resolve_seed_list
from pvzrl_model_metadata import validate_model_metadata


ROOT = Path(__file__).resolve().parents[1]


def _full_bank() -> list[str]:
    return [
        "SunFlower",
        "SunFlower",
        "Peashooter",
        "Peashooter",
        "WallNut",
        "CherryBomb",
        "PotatoMine",
    ]


def _curriculum(discovered: list[str]) -> AdventureSeedCurriculum:
    curriculum = AdventureSeedCurriculum(
        initial_loadout=ADVENTURE_GENERALIST_INITIAL_LOADOUT,
        max_seed_slots=7,
        seed_order_source="explicit_config",
    )
    curriculum.record_unlocked(discovered, episode_index=0)
    curriculum.episode_index = 1
    return curriculum


def test_runtime_cards_after_chomper_enter_canonical_unlocked_and_selectable_catalog() -> None:
    unlocked = Counter({
        "SunFlower": 1,
        "Peashooter": 1,
        "WallNut": 1,
        "CherryBomb": 1,
        "PotatoMine": 1,
        "Chomper": 1,
    })
    state = {
        "visibleSeedCardNames": ["Chomper", "SmallPuff", "FumeShroom", "HypnoShroom"],
        "visibleSeedPlantTypes": [5, 6, 7, 8],
        "availableSeedNames": ["Chomper", "SmallPuff", "FumeShroom", "HypnoShroom"],
    }

    newly_unlocked = update_unlocked_from_state(unlocked, state, source="seed_selection")
    selectable = _filter_supported_seed_names(_selectable_from_seed_screen_state(state))

    assert {"SmallPuff", "FumeShroom", "HypnoShroom"}.issubset(newly_unlocked)
    assert {"SmallPuff", "FumeShroom", "HypnoShroom"}.issubset(unlocked)
    assert {"Chomper", "SmallPuff", "FumeShroom", "HypnoShroom"}.issubset(selectable)
    assert resolve_seed_list(["Chomper", "SmallPuff", "FumeShroom", "HypnoShroom"]) == [5, 6, 7, 8]
    assert [plant_type_name(plant_type) for plant_type in [5, 6, 7, 8]] == [
        "Chomper",
        "SmallPuff",
        "FumeShroom",
        "HypnoShroom",
    ]


def test_full_bank_new_unlock_rotates_instead_of_capacity_full() -> None:
    curriculum = _curriculum(["WallNut", "CherryBomb", "PotatoMine", "Chomper"])
    decision = curriculum.choose_loadout(
        curriculum.unlocked_seeds(),
        observed_capacity=7,
        previous_loadout=_full_bank(),
        validation_seeds=curriculum.unlocked_seeds(),
    )

    assert len(decision.selected_loadout) == 7
    assert "Chomper" in decision.selected_loadout
    assert decision.loadout_reason.startswith("rotation_")
    assert all(row["reason"] != "capacity_full" for row in decision.excluded_new_plants)


def test_duplicate_legacy_slots_are_replaceable_and_oldest_pending_unlock_is_guaranteed() -> None:
    curriculum = _curriculum(
        ["WallNut", "CherryBomb", "PotatoMine", "Chomper", "SmallPuff", "FumeShroom"]
    )
    decision = curriculum.choose_loadout(
        curriculum.unlocked_seeds(),
        observed_capacity=7,
        previous_loadout=_full_bank(),
        validation_seeds=curriculum.unlocked_seeds(),
    )

    starter_duplicate_count = decision.selected_loadout.count("SunFlower") + decision.selected_loadout.count("Peashooter")
    assert starter_duplicate_count < 4
    assert "Chomper" in decision.selected_loadout
    assert "FumeShroom" not in decision.selected_loadout
    assert decision.loadout_reason == "rotation_guaranteed_unlock"
    assert decision.guaranteed_seeds == ["WallNut", "CherryBomb", "PotatoMine", "Chomper", "SmallPuff"]


def test_unlocked_pool_rotates_remaining_cards_through_full_capacity() -> None:
    discovered = [
        "WallNut",
        "CherryBomb",
        "PotatoMine",
        "Chomper",
        "SmallPuff",
        "FumeShroom",
        "HypnoShroom",
        "ScaredyShroom",
        "IceShroom",
    ]
    curriculum = _curriculum(discovered)
    previous = _full_bank()
    loadouts: list[tuple[str, ...]] = []
    for _ in range(len(discovered) + 2):
        decision = curriculum.choose_loadout(
            curriculum.unlocked_seeds(),
            observed_capacity=7,
            previous_loadout=previous,
            validation_seeds=curriculum.unlocked_seeds(),
        )
        loadouts.append(tuple(decision.selected_loadout))
        curriculum.commit_loadout(decision)
        previous = decision.selected_loadout

    assert len(set(loadouts)) > 1
    assert all("SunFlower" in loadout and "Peashooter" in loadout for loadout in loadouts)
    observed_non_core = set().union(*(set(loadout) for loadout in loadouts))
    assert set(discovered).issubset(observed_non_core)


def test_slot_identity_latch_rejects_mid_episode_mutation() -> None:
    env = object.__new__(AdventureGeneralistTrainingEnv)
    env.config = SimpleNamespace(plant_types=[1, 5, 0, 7])
    env._episode_slot_identity = (1, 5, 0, 7)

    env._assert_episode_slot_identity()
    env._assert_episode_slot_identity()
    env.config.plant_types = [1, 6, 0, 7]
    with pytest.raises(RuntimeError, match="generalist_slot_identity_changed_mid_episode"):
        env._assert_episode_slot_identity()


def test_dynamic_slot_identities_keep_the_frozen_generalist_contract() -> None:
    spec = build_action_space_spec(
        mode=ACTION_SPACE_ADVENTURE_14_IDENTITY,
        plant_types=resolve_seed_list(["SunFlower", "Chomper", "SmallPuff", "FumeShroom"]),
        max_seed_slots=14,
    )

    assert spec.action_count == 701
    assert spec.observation_version == "adventure_14slot_identity_v1"
    assert spec.action_decoder_version == "seedslot14x50_plus_wait_v1"
    assert spec.max_seed_slots == 14
    assert spec.identity_seed_slots is True


def test_existing_generalist_checkpoint_metadata_and_load_remain_compatible() -> None:
    candidates = sorted(
        (
            path
            for path in (ROOT / "runs").glob(
                "ppo_adventure_generalist_14slot_identity_v1_*/checkpoints/ppo_pvz_370000_steps.zip"
            )
            if path.is_file()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        pytest.skip("protected local 370000-step checkpoint is not present")

    model_path = candidates[0]
    config_path = model_path.parents[2] / "resolved_config.json"
    if not config_path.is_file():
        config_path = ROOT / "configs" / "ppo_adventure_generalist_14slot_identity_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    report = validate_model_metadata(
        model_path,
        config,
        model_action_count=701,
        model_observation_shape=(4297,),
    )
    assert report.ok, report.to_dict()

    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(str(model_path), device="cpu")
    assert model.action_space.n == 701
    assert tuple(model.observation_space.shape) == (4297,)
