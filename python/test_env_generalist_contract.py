"""Environment-level locks for the sole Adventure Generalist policy contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pvzrl_action_space import (
    ACTION_SPACE_ADVENTURE_14_IDENTITY,
    ADVENTURE_IDENTITY_ACTION_COUNT,
    ADVENTURE_IDENTITY_ACTION_DECODER_VERSION,
    CELLS_PER_SLOT,
    DEFAULT_COLS,
    DEFAULT_ROWS,
)
from pvzrl_env import (
    RUN_MODE_ADVENTURE_GENERALIST_14SLOT_EVAL,
    RUN_MODE_ADVENTURE_GENERALIST_14SLOT_TRAIN,
    PvZEnvConfig,
    PvZGymEnv,
    decode_action,
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
    assert (config.row_count, config.column_count) == (DEFAULT_ROWS, DEFAULT_COLS)
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
        assert validation.spec.action_count == ADVENTURE_IDENTITY_ACTION_COUNT == 841
        assert validation.spec.action_decoder_version == ADVENTURE_IDENTITY_ACTION_DECODER_VERSION
        assert validation.spec.wait_action == 0
        assert (validation.spec.placement_action_min, validation.spec.placement_action_max) == (1, 840)
        assert len(mask) == 841
        assert mask[0]
    finally:
        env.close()


def test_five_lane_board_uses_padded_six_lane_action_blocks() -> None:
    env = PvZGymEnv(PvZEnvConfig())
    try:
        observation = _observation()
        observation["actionCount"] = ADVENTURE_IDENTITY_ACTION_COUNT
        # Slot 2, row 0, column 1 is action 122 in the new permanent 60-cell
        # block.  Action 51 is instead slot 0's padded sixth row and must not
        # be silently reinterpreted on a five-lane board.
        ready_slot_cell = 1 + 2 * CELLS_PER_SLOT + 1
        padded_sixth_row = 1 + 5 * DEFAULT_COLS
        observation["legalActions"] = [0, ready_slot_cell]
        observation["legalActionCount"] = 2
        mask = env.action_mask(observation)
        assert mask[ready_slot_cell]
        assert not mask[padded_sixth_row]
        decoded = decode_action(ready_slot_cell, observation, [1, 1, 0, 0])
        assert (decoded["slot_index"], decoded["row"], decoded["column"]) == (2, 0, 1)
    finally:
        env.close()
