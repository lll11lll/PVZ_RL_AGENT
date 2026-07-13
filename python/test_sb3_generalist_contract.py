"""Adventure Generalist-only SB3 adapter contracts."""

from __future__ import annotations

import pytest

from pvzrl_action_space import (
    ACTION_SPACE_ADVENTURE_14_IDENTITY,
    ADVENTURE_IDENTITY_ACTION_COUNT,
    ADVENTURE_IDENTITY_ACTION_DECODER_VERSION,
    ADVENTURE_IDENTITY_OBSERVATION_VERSION,
)
from pvzrl_env import (
    RUN_MODE_ADVENTURE_GENERALIST_14SLOT_EVAL,
    RUN_MODE_ADVENTURE_GENERALIST_14SLOT_TRAIN,
)
from pvzrl_sb3 import PvZSB3Config


def test_default_sb3_config_is_exact_generalist_contract() -> None:
    config = PvZSB3Config()
    metadata = config.get_env_metadata()

    assert config.run_mode == RUN_MODE_ADVENTURE_GENERALIST_14SLOT_TRAIN
    assert config.seed_list == ["SunFlower", "SunFlower", "Peashooter", "Peashooter"]
    assert config.plant_types == [1, 1, 0, 0]
    assert config.auto_select_seeds is True
    assert config.enable_board_plant_identity is True
    assert metadata["action_space_mode"] == ACTION_SPACE_ADVENTURE_14_IDENTITY
    assert metadata["env_action_count"] == ADVENTURE_IDENTITY_ACTION_COUNT == 701
    assert metadata["max_seed_slots"] == 14
    assert metadata["dynamic_seed_slots"] is True
    assert metadata["identity_seed_slots"] is True
    assert metadata["observation_version"] == ADVENTURE_IDENTITY_OBSERVATION_VERSION
    assert metadata["action_decoder_version"] == ADVENTURE_IDENTITY_ACTION_DECODER_VERSION
    assert metadata["decoder_wait_action"] == 0
    assert metadata["placement_action_range"] == [1, 700]
    assert not hasattr(config, "target_level")
    assert not hasattr(config, "adventure_eval_mode")
    assert not hasattr(config, "dynamic_seed_slots")


def test_generalist_evaluation_run_mode_is_accepted() -> None:
    config = PvZSB3Config(run_mode=RUN_MODE_ADVENTURE_GENERALIST_14SLOT_EVAL)
    assert config.run_mode == RUN_MODE_ADVENTURE_GENERALIST_14SLOT_EVAL


@pytest.mark.parametrize(
    "obsolete_mode",
    ["fixed_train", "fixed_eval", "level3_specialist", "adventure_eval", ""],
)
def test_obsolete_run_modes_are_rejected(obsolete_mode: str) -> None:
    with pytest.raises(ValueError, match="Unsupported run_mode"):
        PvZSB3Config(run_mode=obsolete_mode)


@pytest.mark.parametrize("obsolete_action_mode", ["fixed", "dynamic_14"])
def test_obsolete_action_modes_are_rejected(obsolete_action_mode: str) -> None:
    with pytest.raises(ValueError, match="Unsupported action_space_mode"):
        PvZSB3Config(action_space_mode=obsolete_action_mode)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("max_seed_slots", 4, "exactly 14 seed slots"),
        ("observation_version", "fixed_slot_v1", "observation_version mismatch"),
        ("action_decoder_version", "max_seed_slots_14_v1", "action_decoder_version mismatch"),
        ("column_count", 9, "requires a 5x10 board"),
    ],
)
def test_non_generalist_contract_values_are_rejected(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        PvZSB3Config(**{field: value})
