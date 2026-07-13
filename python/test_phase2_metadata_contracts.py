from __future__ import annotations

import json
from pathlib import Path

import pytest

from pvzrl_model_metadata import (
    BLOCKED_ACTION_SPACE_MODE,
    BLOCKED_BOARD_GEOMETRY,
    BLOCKED_DYNAMIC_SEED_SLOTS,
    BLOCKED_IDENTITY_SEED_SLOTS,
    BLOCKED_METADATA_VERSION,
    BLOCKED_MISSING_METADATA,
    BLOCKED_MODEL_FAMILY,
    BLOCKED_OBSERVATION_SHAPE,
    BLOCKED_PLACEMENT_ACTION_RANGE,
    MODEL_METADATA_FILENAME,
    env_metadata_from_config,
    model_metadata_from_config,
    observation_shape_from_config,
    validate_model_metadata,
)


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


def test_exact_generalist_contract_and_complete_metadata_pass(tmp_path: Path) -> None:
    model_path, _ = _model_with_metadata(tmp_path, IDENTITY_CONFIG)
    shape = [4297]
    assert observation_shape_from_config(IDENTITY_CONFIG) == shape
    env_metadata = env_metadata_from_config(IDENTITY_CONFIG)
    assert env_metadata["observation_shape"] == shape
    metadata = model_metadata_from_config(IDENTITY_CONFIG)
    assert metadata["observation_shape"] == shape
    assert metadata["action_space_mode"] == "adventure_14slot_identity"
    assert metadata["action_count"] == 701
    assert metadata["decoder_wait_action"] == 0
    assert metadata["placement_action_range"] == [1, 700]
    assert metadata["action_decoder_version"] == "seedslot14x50_plus_wait_v1"
    assert metadata["observation_version"] == "adventure_14slot_identity_v1"
    assert metadata["dynamic_seed_slots"] is True
    assert metadata["identity_seed_slots"] is True
    result = validate_model_metadata(
        model_path,
        IDENTITY_CONFIG,
        model_action_count=701,
        model_observation_shape=tuple(shape),
    )
    assert result.ok, result.to_dict()
    assert result.actual["loaded_observation_shape"] == shape


@pytest.mark.parametrize(
    ("field", "value", "blocked_reason"),
    [
        ("metadata_version", 999, BLOCKED_METADATA_VERSION),
        ("action_space_mode", "fixed", BLOCKED_ACTION_SPACE_MODE),
        ("dynamic_seed_slots", False, BLOCKED_DYNAMIC_SEED_SLOTS),
        ("identity_seed_slots", False, BLOCKED_IDENTITY_SEED_SLOTS),
        ("model_family", "obsolete-specialist", BLOCKED_MODEL_FAMILY),
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
        model_observation_shape=(4296,),
    )
    assert not result.ok
    assert result.blocked_reason == BLOCKED_OBSERVATION_SHAPE
    assert "model=[4296]" in result.details
    assert "environment=[4297]" in result.details


def test_declared_observation_shape_mismatch_is_rejected_without_loading_model(tmp_path: Path) -> None:
    model_path, metadata_path = _model_with_metadata(tmp_path, IDENTITY_CONFIG)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["observation_shape"] = [4296]
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


def test_legacy_resolved_config_is_not_inferred_as_checkpoint_metadata(tmp_path: Path) -> None:
    run_dir = tmp_path / "legacy-only"
    run_dir.mkdir()
    model_path = run_dir / "model.zip"
    model_path.write_bytes(b"fixture")
    (run_dir / "resolved_config.json").write_text(
        json.dumps(IDENTITY_CONFIG, indent=2),
        encoding="utf-8",
    )
    result = validate_model_metadata(
        model_path,
        IDENTITY_CONFIG,
        model_action_count=701,
        model_observation_shape=(4297,),
    )
    assert not result.ok
    assert result.blocked_reason == BLOCKED_MISSING_METADATA
    assert result.metadata_inferred is False
