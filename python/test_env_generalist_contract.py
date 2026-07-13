"""Environment-level locks for the sole Adventure Generalist policy contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pvzrl_action_space import (
    ACTION_SPACE_ADVENTURE_14_IDENTITY,
    ADVENTURE_IDENTITY_ACTION_COUNT,
    ADVENTURE_IDENTITY_ACTION_DECODER_VERSION,
)
from pvzrl_env import (
    RUN_MODE_ADVENTURE_GENERALIST_14SLOT_EVAL,
    RUN_MODE_ADVENTURE_GENERALIST_14SLOT_TRAIN,
    PvZEnvConfig,
    PvZGymEnv,
)


FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "refactor_contracts"
    / "synthetic_observation.json"
)


def _observation() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_environment_defaults_are_generalist_only() -> None:
    config = PvZEnvConfig()
    assert config.run_mode == RUN_MODE_ADVENTURE_GENERALIST_14SLOT_TRAIN
    assert config.plant_types == [1, 1, 0, 0]
    assert config.seed_list == ["SunFlower", "SunFlower", "Peashooter", "Peashooter"]
    assert config.auto_select_seeds is True
    assert (config.row_count, config.column_count) == (5, 10)
    assert not hasattr(config, "target_level")
    assert not hasattr(config, "adventure_eval_mode")


@pytest.mark.parametrize("removed_mode", ["fixed_train", "fixed_eval", "adventure_eval", ""])
def test_environment_rejects_removed_run_modes(removed_mode: str) -> None:
    with pytest.raises(ValueError, match="Unsupported run_mode"):
        PvZGymEnv(PvZEnvConfig(run_mode=removed_mode))


@pytest.mark.parametrize(
    "config",
    [
        PvZEnvConfig(row_count=4),
        PvZEnvConfig(column_count=9),
        PvZEnvConfig(plant_types=list(range(15))),
        PvZEnvConfig(seed_list=[str(index) for index in range(15)]),
    ],
)
def test_environment_rejects_non_generalist_geometry_or_capacity(config: PvZEnvConfig) -> None:
    with pytest.raises(ValueError, match="Adventure Generalist"):
        PvZGymEnv(config)


@pytest.mark.parametrize(
    "run_mode",
    [
        RUN_MODE_ADVENTURE_GENERALIST_14SLOT_TRAIN,
        RUN_MODE_ADVENTURE_GENERALIST_14SLOT_EVAL,
    ],
)
def test_action_cache_is_pinned_to_generalist_identity_despite_bridge_slot_hint(
    run_mode: str,
) -> None:
    env = PvZGymEnv(PvZEnvConfig(run_mode=run_mode))
    try:
        observation = _observation()
        # Four live cards make the bridge's active-slot action hint 201.  That
        # must never resize the model-facing Generalist policy contract.
        observation["actionCount"] = 201
        validation = env._action_validation_config(observation)
        mask = env.action_mask(observation)
        assert validation.action_space_mode == ACTION_SPACE_ADVENTURE_14_IDENTITY
        assert validation.max_seed_slots == 14
        assert validation.spec.action_count == ADVENTURE_IDENTITY_ACTION_COUNT == 701
        assert validation.spec.action_decoder_version == ADVENTURE_IDENTITY_ACTION_DECODER_VERSION
        assert validation.spec.wait_action == 0
        assert (validation.spec.placement_action_min, validation.spec.placement_action_max) == (1, 700)
        assert len(mask) == 701
        assert mask[0]
    finally:
        env.close()
