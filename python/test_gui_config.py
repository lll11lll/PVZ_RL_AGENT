from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from pvzrl_gui_config import (
    FULL_ADVENTURE_CONTRACT,
    STREAMER_V1_ARG_FLAGS,
    TwitchEnvironmentNames,
    build_streamer_v1_argv_additions,
    inspect_model_compatibility,
    inspect_twitch_credentials,
    validate_evaluation_form,
    validate_streamer_v1_form,
    validate_train_form,
)
from pvzrl_model_metadata import MODEL_METADATA_FILENAME, model_metadata_from_config
from train_ppo import build_arg_parser


def _checkpoint(tmp_path: Path, *, changes: dict[str, object] | None = None) -> Path:
    run_dir = tmp_path / "source"
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / "model.zip"
    model_path.write_bytes(b"metadata-only fixture")
    metadata = model_metadata_from_config(FULL_ADVENTURE_CONTRACT.expected_config())
    metadata["run_mode"] = "adventure_generalist_14slot_train"
    metadata.update(changes or {})
    (run_dir / MODEL_METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return model_path


def _twitch_environment() -> dict[str, str]:
    return {
        "PVZRL_TWITCH_CLIENT_ID": "client-secret-value",
        "PVZRL_TWITCH_USER_ACCESS_TOKEN": "token-secret-value",
        "PVZRL_TWITCH_BROADCASTER_USER_ID": "broadcaster-secret-value",
        "PVZRL_TWITCH_EVENTSUB_USER_ID": "optional-user-secret-value",
        "PVZRL_TWITCH_VIEWER_HASH_SECRET": "hash-secret-value",
    }


def _streamer_values(tmp_path: Path, baseline: Path) -> dict[str, object]:
    return {
        "streamer_platform": "twitch",
        "streamer_baseline_checkpoint": str(baseline),
        "run_dir": str(tmp_path / "experiment"),
        "live_status_path": str(tmp_path / "experiment" / "live_status.json"),
        "adventure_start_level": "1",
        "max_adventure_levels": "50",
        "max_attempts_per_level": "10",
        "n_steps": "500",
        "batch_size": "50",
        "streamer_intervention_interval_seconds": "2.0",
        "streamer_command_ttl_seconds": "10.0",
        "streamer_command_queue_capacity": "256",
        "streamer_message_max_chars": "256",
        "streamer_policy_steps_per_cycle": "25000",
        "streamer_checkpoint_policy_steps": "5000",
        "streamer_evaluation_episodes": "50",
        "streamer_max_cycles": "0",
        "streamer_endurance_hours": "0",
        "streamer_bc_enabled": True,
        "streamer_bc_coefficient": "0.01",
        "streamer_demonstration_capacity": "4096",
        "streamer_demonstration_persist_every": "512",
        "streamer_bc_batch_size": "32",
        "streamer_bc_update_frequency": "1",
        "streamer_bc_min_demonstrations": "8",
    }


def test_contract_descriptor_is_immutable_and_exact_full_adventure() -> None:
    contract = FULL_ADVENTURE_CONTRACT
    assert (contract.rows, contract.cols) == (6, 10)
    assert contract.cells_per_seed_slot == 60
    assert contract.max_seed_slots == 14
    assert contract.action_count == 841
    assert contract.wait_action == 0
    assert contract.placement_action_range == (1, 840)
    assert contract.observation_shape == (4364,)
    assert contract.action_decoder_version == "seedslot14x60_padded6x10_plus_wait_v2"
    assert contract.initial_loadout == (
        "SunFlower",
        "SunFlower",
        "Peashooter",
        "Peashooter",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        contract.rows = 5  # type: ignore[misc]


def test_metadata_only_inspection_accepts_every_exact_semantic_field(tmp_path: Path) -> None:
    model_path = _checkpoint(tmp_path)
    inspection = inspect_model_compatibility(model_path)
    assert inspection.compatible, inspection.issues
    assert inspection.metadata_path == model_path.parent / MODEL_METADATA_FILENAME
    assert inspection.declared_metadata["rows"] == 6
    assert inspection.declared_metadata["cols"] == 10
    assert inspection.declared_metadata["cells_per_seed_slot"] == 60
    assert inspection.declared_metadata["observation_shape"] == (4364,)
    with pytest.raises(TypeError):
        inspection.declared_metadata["rows"] = 5  # type: ignore[index]


@pytest.mark.parametrize(
    ("changes", "expected_code"),
    [
        ({"rows": 5}, "board_geometry_mismatch"),
        ({"cols": 9}, "board_geometry_mismatch"),
        ({"cells_per_seed_slot": 50}, "board_geometry_mismatch"),
        ({"action_count": 701}, "action_count_mismatch"),
        ({"observation_shape": [4297]}, "observation_shape_mismatch"),
        ({"model_family": "old-family"}, "model_family_mismatch"),
        ({"seed_list": ["SunFlower", "Peashooter"]}, "seed_list_mismatch"),
        ({"dynamic_seed_slots": "true"}, "dynamic_seed_slots_mismatch"),
    ],
)
def test_inspection_rejects_contract_drift_including_geometry(
    tmp_path: Path, changes: dict[str, object], expected_code: str
) -> None:
    inspection = inspect_model_compatibility(_checkpoint(tmp_path, changes=changes))
    assert not inspection.compatible
    assert expected_code in {issue.code for issue in inspection.issues}


def test_inspection_requires_model_and_canonical_metadata(tmp_path: Path) -> None:
    missing = inspect_model_compatibility(tmp_path / "missing.zip")
    assert not missing.ok
    assert {issue.code for issue in missing.issues} == {
        "model_path_missing",
        "missing_model_metadata",
    }
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"fixture")
    no_metadata = inspect_model_compatibility(model_path)
    assert no_metadata.blocked_reason == "missing_model_metadata"


def test_credential_readiness_exposes_only_names_and_boolean_states() -> None:
    environment = _twitch_environment()
    readiness = inspect_twitch_credentials(TwitchEnvironmentNames(), environ=environment)
    assert readiness.ready
    assert readiness.missing_names == ()
    assert all(state.present for state in readiness.variables)
    rendered = repr(readiness)
    for secret_value in environment.values():
        assert secret_value not in rendered


def test_credential_readiness_reports_missing_and_invalid_names_without_values() -> None:
    names = TwitchEnvironmentNames(access_token="bad-name", user_id="")
    readiness = inspect_twitch_credentials(names, environ={})
    assert not readiness.ready
    codes = {issue.code for issue in readiness.issues}
    assert "invalid_streamer_environment_variable_name" in codes
    assert "streamer_twitch_environment_missing" in codes
    assert "bad-name" not in readiness.missing_names
    assert DEFAULT_OPTIONAL_USER_ID_NOT_REQUIRED(readiness)


def DEFAULT_OPTIONAL_USER_ID_NOT_REQUIRED(readiness: object) -> bool:
    # Kept as a named assertion helper so a future required-user-id change is
    # explicit in this credential-boundary test.
    return any(
        state.role == "user_id" and not state.required and not state.present
        for state in readiness.variables  # type: ignore[attr-defined]
    )


def test_train_and_evaluation_forms_validate_paths_numbers_and_metadata(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    train = validate_train_form(
        {
            "run_dir": "train-output",
            "resume_model_path": str(checkpoint),
            "total_timesteps": "512",
            "checkpoint_freq": "128",
            "n_steps": "128",
            "batch_size": "64",
            "adventure_start_level": "1",
            "max_adventure_levels": "50",
            "adventure_soft_max_steps": "2000",
            "adventure_hard_max_steps": "3500",
        },
        base_dir=tmp_path,
    )
    assert train.ok, train.issues
    assert train.value is not None
    assert train.value.run_dir == (tmp_path / "train-output").resolve()
    assert train.value.live_status_path == train.value.run_dir / "live_status.json"

    unsafe_resume = validate_train_form(
        {
            "run_dir": str(checkpoint.parent),
            "resume_model_path": str(checkpoint),
        },
        base_dir=tmp_path,
    )
    assert not unsafe_resume.ok
    assert "resume_run_dir_must_be_new" in {
        issue.code for issue in unsafe_resume.errors
    }

    evaluation = validate_evaluation_form(
        {"model_path": str(checkpoint), "run_dir": "eval-output"},
        base_dir=tmp_path,
    )
    assert evaluation.ok, evaluation.issues
    assert evaluation.value is not None
    assert evaluation.value.model_path == checkpoint.resolve()

    unsafe_eval = validate_evaluation_form(
        {"model_path": str(checkpoint), "run_dir": str(checkpoint.parent)},
        base_dir=tmp_path,
    )
    assert not unsafe_eval.ok
    assert "evaluation_source_inside_run_dir" in {
        issue.code for issue in unsafe_eval.errors
    }

    invalid = validate_train_form(
        {
            "run_dir": "bad-output",
            "total_timesteps": "0",
            "n_steps": "nan",
            "adventure_start_level": "6",
            "max_adventure_levels": "5",
            "adventure_soft_max_steps": "3000",
            "adventure_hard_max_steps": "2000",
        },
        base_dir=tmp_path,
    )
    assert not invalid.ok
    assert invalid.value is None
    assert {
        "value_below_minimum",
        "invalid_integer",
        "adventure_step_limits_invalid",
    }.issubset({issue.code for issue in invalid.errors})

    # Backend semantics: max_adventure_levels is a number of sequential
    # levels to attempt, not an absolute final-level identifier.
    later_level_smoke = validate_train_form(
        {
            "run_dir": "later-level-smoke",
            "adventure_start_level": "11",
            "max_adventure_levels": "1",
        },
        base_dir=tmp_path,
    )
    assert later_level_smoke.ok, later_level_smoke.issues


def test_streamer_form_and_argv_match_real_parser_without_secret_values(tmp_path: Path) -> None:
    baseline = _checkpoint(tmp_path)
    values = _streamer_values(tmp_path, baseline)
    values["adventure_start_level"] = "11"
    values["max_adventure_levels"] = "1"
    result = validate_streamer_v1_form(
        values,
        base_dir=tmp_path,
        environ=_twitch_environment(),
    )
    assert result.ok, result.issues
    assert result.value is not None
    argv = build_streamer_v1_argv_additions(result.value)
    assert "--streamer-v1" in argv
    assert "--streamer-demonstration-persist-every" not in argv
    assert all(value not in argv for value in _twitch_environment().values())
    assert set(argument for argument in argv if argument.startswith("--")) <= STREAMER_V1_ARG_FLAGS

    parsed = build_arg_parser().parse_args(["--adventure-generalist-train", *argv])
    assert parsed.streamer_v1_enabled is True
    assert parsed.n_steps == 500
    assert parsed.batch_size == 50
    assert parsed.streamer_policy_steps_per_cycle == 25000
    assert parsed.streamer_bc_enabled is True
    assert parsed.adventure_start_level == 11
    assert parsed.max_adventure_levels == 1


def test_streamer_form_rejects_editable_legacy_board_contract(tmp_path: Path) -> None:
    baseline = _checkpoint(tmp_path)
    values = _streamer_values(tmp_path, baseline)
    values.update(
        {
            "initial_loadout": "SunFlower,Peashooter",
            "max_seed_slots": "5",
        }
    )

    result = validate_streamer_v1_form(
        values,
        base_dir=tmp_path,
        environ=_twitch_environment(),
    )

    assert not result.ok
    assert {"initial_loadout_mismatch", "value_below_minimum"} <= {
        issue.code for issue in result.errors
    }


def test_streamer_alignment_and_resource_bounds_match_backend(tmp_path: Path) -> None:
    baseline = _checkpoint(tmp_path)
    values = _streamer_values(tmp_path, baseline)
    values.update(
        {
            "n_steps": "512",
            "batch_size": "50",
            "streamer_policy_steps_per_cycle": "25000",
            "streamer_checkpoint_policy_steps": "26000",
            "streamer_demonstration_capacity": "16",
            "streamer_demonstration_persist_every": "17",
            "streamer_bc_batch_size": "18",
            "streamer_bc_min_demonstrations": "19",
        }
    )
    result = validate_streamer_v1_form(
        values, base_dir=tmp_path, environ=_twitch_environment()
    )
    assert not result.ok
    codes = {issue.code for issue in result.issues}
    assert "streamer_minibatch_alignment" in codes
    assert "streamer_cycle_rollout_alignment" in codes
    assert "invalid_streamer_checkpoint_interval" in codes
    assert "invalid_streamer_demonstration_bounds" in codes


def test_mock_script_is_required_and_local_identity_needs_hash_secret(tmp_path: Path) -> None:
    baseline = _checkpoint(tmp_path)
    values = _streamer_values(tmp_path, baseline)
    values["streamer_platform"] = "mock"
    missing = validate_streamer_v1_form(values, base_dir=tmp_path, environ={})
    assert "streamer_mock_script_missing" in {issue.code for issue in missing.issues}

    script = tmp_path / "mock.jsonl"
    script.write_text('{"command":"!wait","local_viewer_id":"local-user"}\n', encoding="utf-8")
    values["streamer_mock_script"] = str(script)
    no_secret = validate_streamer_v1_form(values, base_dir=tmp_path, environ={})
    assert "invalid_streamer_mock_script" in {issue.code for issue in no_secret.issues}
    assert "local-user" not in repr(no_secret)

    autonomous_script = tmp_path / "autonomous.jsonl"
    autonomous_script.write_text("# empty autonomous mock is valid\n", encoding="utf-8")
    values["streamer_mock_script"] = str(autonomous_script)
    valid = validate_streamer_v1_form(values, base_dir=tmp_path, environ={})
    assert valid.ok, valid.issues
    assert valid.value is not None
    assert "--streamer-mock-script" in build_streamer_v1_argv_additions(valid.value)


def test_streamer_baseline_must_remain_outside_experiment(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    baseline = _checkpoint(experiment)
    values = _streamer_values(tmp_path, baseline)
    values["run_dir"] = str(experiment)
    result = validate_streamer_v1_form(
        values, base_dir=tmp_path, environ=_twitch_environment()
    )
    assert not result.ok
    assert "streamer_baseline_inside_experiment" in {
        issue.code for issue in result.issues
    }


def test_output_and_live_status_paths_fail_closed_without_blocking_empty_folder(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    empty = tmp_path / "empty-output"
    empty.mkdir()
    safe = validate_train_form({"run_dir": str(empty)}, base_dir=tmp_path)
    assert safe.ok, safe.issues

    occupied = tmp_path / "occupied-output"
    occupied.mkdir()
    (occupied / "model.zip").write_bytes(b"existing-user-artifact")
    unsafe_output = validate_train_form({"run_dir": str(occupied)}, base_dir=tmp_path)
    assert not unsafe_output.ok
    assert "run_dir_not_empty" in {issue.code for issue in unsafe_output.errors}

    status_over_model = validate_evaluation_form(
        {
            "model_path": str(checkpoint),
            "run_dir": str(tmp_path / "eval-output"),
            "live_status_path": str(checkpoint),
        },
        base_dir=tmp_path,
    )
    assert not status_over_model.ok
    assert "live_status_path_collision" in {
        issue.code for issue in status_over_model.errors
    }


def test_config_and_board_timeout_preflight(tmp_path: Path) -> None:
    missing_config = validate_train_form(
        {
            "run_dir": str(tmp_path / "missing-config-output"),
            "config_path": str(tmp_path / "missing.json"),
        },
        base_dir=tmp_path,
    )
    assert "config_path_missing" in {issue.code for issue in missing_config.errors}

    invalid_config = tmp_path / "invalid.json"
    invalid_config.write_text("[]\n", encoding="utf-8")
    invalid_root = validate_train_form(
        {
            "run_dir": str(tmp_path / "invalid-config-output"),
            "config_path": str(invalid_config),
        },
        base_dir=tmp_path,
    )
    assert "invalid_config_root" in {issue.code for issue in invalid_root.errors}

    invalid_timeout = validate_train_form(
        {"run_dir": str(tmp_path / "bad-timeout"), "board_timeout": "0"},
        base_dir=tmp_path,
    )
    assert "value_below_minimum" in {issue.code for issue in invalid_timeout.errors}

    valid_config = tmp_path / "valid.json"
    valid_config.write_text("{}\n", encoding="utf-8")
    status_over_config = validate_train_form(
        {
            "run_dir": str(tmp_path / "status-config-output"),
            "config_path": str(valid_config),
            "live_status_path": str(valid_config),
        },
        base_dir=tmp_path,
    )
    assert "live_status_path_collision" in {
        issue.code for issue in status_over_config.errors
    }


def test_streamer_nonempty_directory_requires_canonical_resume_evidence(
    tmp_path: Path,
) -> None:
    baseline = _checkpoint(tmp_path)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "notes.txt").write_text("not a Streamer experiment", encoding="utf-8")
    foreign_values = _streamer_values(tmp_path, baseline)
    foreign_values["run_dir"] = str(foreign)
    foreign_values["live_status_path"] = str(foreign / "live_status.json")
    rejected = validate_streamer_v1_form(
        foreign_values, base_dir=tmp_path, environ=_twitch_environment()
    )
    assert "run_dir_not_empty" in {issue.code for issue in rejected.errors}

    experiment = tmp_path / "recognized"
    experiment.mkdir()
    (experiment / "streamer_state.json").write_text("{}\n", encoding="utf-8")
    resume_values = _streamer_values(tmp_path, baseline)
    resume_values["run_dir"] = str(experiment)
    resume_values["live_status_path"] = str(experiment / "live_status.json")
    accepted = validate_streamer_v1_form(
        resume_values, base_dir=tmp_path, environ=_twitch_environment()
    )
    assert accepted.ok, accepted.issues
