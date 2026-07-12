from __future__ import annotations

import json
from pathlib import Path

import pytest

from pvzrl_model_metadata import (
    BLOCKED_BOARD_GEOMETRY,
    BLOCKED_IDENTITY_SEED_SLOTS,
    BLOCKED_METADATA_VERSION,
    BLOCKED_OBSERVATION_SHAPE,
    BLOCKED_PLACEMENT_ACTION_RANGE,
    MODEL_METADATA_FILENAME,
    env_metadata_from_config,
    model_metadata_from_config,
    observation_shape_from_config,
    validate_model_metadata,
)


FIXED_CONFIG = {
    "model_family": "fixed-contract",
    "seed_list": ["SunFlower", "Peashooter", "WallNut", "CherryBomb"],
    "plant_types": [1, 0, 3, 2],
    "action_space_mode": "fixed",
    "max_seed_slots": 4,
    "row_count": 5,
    "column_count": 10,
}

IDENTITY_CONFIG = {
    "model_family": "ppo_adventure_generalist_14slot_identity_v1",
    "seed_list": ["SunFlower", "SunFlower", "Peashooter", "Peashooter"],
    "plant_types": [1, 1, 0, 0],
    "action_space_mode": "adventure_14slot_identity",
    "max_seed_slots": 14,
    "row_count": 5,
    "column_count": 10,
}


def _model_with_metadata(tmp_path: Path, config: dict) -> tuple[Path, Path]:
    run_dir = tmp_path / str(config["model_family"])
    run_dir.mkdir()
    model_path = run_dir / "model.zip"
    model_path.write_bytes(b"fixture")
    metadata_path = run_dir / MODEL_METADATA_FILENAME
    metadata_path.write_text(json.dumps(model_metadata_from_config(config), indent=2), encoding="utf-8")
    return model_path, metadata_path


@pytest.mark.parametrize(
    ("config", "action_count", "shape"),
    [
        (FIXED_CONFIG, 201, [357]),
        (IDENTITY_CONFIG, 701, [4297]),
    ],
)
def test_exact_observation_shapes_and_complete_metadata_pass(
    tmp_path: Path,
    config: dict,
    action_count: int,
    shape: list[int],
) -> None:
    model_path, _ = _model_with_metadata(tmp_path, config)
    assert observation_shape_from_config(config) == shape
    env_metadata = env_metadata_from_config(config)
    assert env_metadata["observation_shape"] == shape
    assert model_metadata_from_config(config)["observation_shape"] == shape
    result = validate_model_metadata(
        model_path,
        config,
        model_action_count=action_count,
        model_observation_shape=tuple(shape),
    )
    assert result.ok, result.to_dict()
    assert result.actual["loaded_observation_shape"] == shape


@pytest.mark.parametrize(
    ("field", "value", "blocked_reason"),
    [
        ("metadata_version", 999, BLOCKED_METADATA_VERSION),
        ("identity_seed_slots", False, BLOCKED_IDENTITY_SEED_SLOTS),
        ("decoder_wait_action", 700, "action_decoder_mismatch"),
        ("placement_action_range", [0, 699], BLOCKED_PLACEMENT_ACTION_RANGE),
        ("rows", 6, BLOCKED_BOARD_GEOMETRY),
        ("cols", 9, BLOCKED_BOARD_GEOMETRY),
        ("cells_per_seed_slot", 49, BLOCKED_BOARD_GEOMETRY),
    ],
)
def test_identity_metadata_drift_is_rejected(
    tmp_path: Path,
    field: str,
    value: object,
    blocked_reason: str,
) -> None:
    model_path, metadata_path = _model_with_metadata(tmp_path, IDENTITY_CONFIG)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[field] = value
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    result = validate_model_metadata(
        model_path,
        IDENTITY_CONFIG,
        model_action_count=701,
        model_observation_shape=(4297,),
    )
    assert not result.ok
    assert result.blocked_reason == blocked_reason


def test_loaded_observation_shape_mismatch_is_rejected(tmp_path: Path) -> None:
    model_path, _ = _model_with_metadata(tmp_path, IDENTITY_CONFIG)
    result = validate_model_metadata(
        model_path,
        IDENTITY_CONFIG,
        model_action_count=701,
        model_observation_shape=(357,),
    )
    assert not result.ok
    assert result.blocked_reason == BLOCKED_OBSERVATION_SHAPE
    assert "model=[357]" in result.details
    assert "environment=[4297]" in result.details


def test_declared_observation_shape_mismatch_is_rejected_without_loading_model(tmp_path: Path) -> None:
    model_path, metadata_path = _model_with_metadata(tmp_path, IDENTITY_CONFIG)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["observation_shape"] = [357]
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    result = validate_model_metadata(model_path, IDENTITY_CONFIG, model_action_count=701)
    assert not result.ok
    assert result.blocked_reason == BLOCKED_OBSERVATION_SHAPE
    assert "metadata.observation_shape" in result.details


@pytest.mark.parametrize("malformed_shape", ["4297", [], ["bad"]])
def test_explicit_malformed_observation_shape_is_not_treated_as_legacy_absence(
    tmp_path: Path,
    malformed_shape: object,
) -> None:
    model_path, metadata_path = _model_with_metadata(tmp_path, IDENTITY_CONFIG)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["observation_shape"] = malformed_shape
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    result = validate_model_metadata(model_path, IDENTITY_CONFIG, model_action_count=701)
    assert not result.ok
    assert result.blocked_reason == BLOCKED_OBSERVATION_SHAPE
    assert "malformed" in result.details


def test_fixed_legacy_false_identity_default_remains_compatible(tmp_path: Path) -> None:
    model_path, metadata_path = _model_with_metadata(tmp_path, FIXED_CONFIG)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("identity_seed_slots", None)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    result = validate_model_metadata(
        model_path,
        FIXED_CONFIG,
        model_action_count=201,
        model_observation_shape=(357,),
    )
    assert result.ok, result.to_dict()
