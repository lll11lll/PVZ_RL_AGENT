"""Focused contracts for the Generalist-only typed run configuration."""

from __future__ import annotations

import json
from argparse import Namespace
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from pvzrl_config import ConfigSource, IgnoredLegacyConfigWarning, resolve_config_value
from train_ppo import build_arg_parser, build_config, build_resolved_config


ROOT = Path(__file__).resolve().parents[1]
GENERALIST_CONFIG = ROOT / "configs" / "ppo_adventure_generalist_14slot_identity_v1.json"
GENERALIST_TRAIN = "adventure_generalist_14slot_train"
GENERALIST_EVAL = "adventure_generalist_14slot_eval"
GENERALIST_LOADOUT = ["SunFlower", "SunFlower", "Peashooter", "Peashooter"]


@pytest.mark.parametrize(
    ("cli_values", "json_values", "mode_values", "global_values", "expected", "source"),
    [
        ({"setting": 0}, {"setting": 2}, {"setting": 3}, {"setting": 4}, 0, ConfigSource.CLI),
        ({"setting": False}, {"setting": True}, {"setting": True}, {"setting": True}, False, ConfigSource.CLI),
        ({"setting": ""}, {"setting": "json"}, {"setting": "mode"}, {"setting": "global"}, "", ConfigSource.CLI),
        ({"setting": None}, {"setting": 2}, {"setting": 3}, {"setting": 4}, 2, ConfigSource.JSON),
        ({}, {"setting": None}, {"setting": 3}, {"setting": 4}, None, ConfigSource.JSON),
        ({}, {}, {"setting": 3}, {"setting": 4}, 3, ConfigSource.MODE_DEFAULT),
        ({}, {}, {}, {"setting": 4}, 4, ConfigSource.GLOBAL_DEFAULT),
    ],
)
def test_precedence_matrix(
    cli_values: dict[str, object],
    json_values: dict[str, object],
    mode_values: dict[str, object],
    global_values: dict[str, object],
    expected: object,
    source: ConfigSource,
) -> None:
    resolved = resolve_config_value(
        "setting",
        cli_namespace=Namespace(**cli_values),
        json_values=json_values,
        mode_defaults=mode_values,
        global_defaults=global_values,
    )
    assert resolved.value == expected
    assert resolved.source is source


def test_precedence_requires_a_value_or_default() -> None:
    with pytest.raises(KeyError, match="No configured value"):
        resolve_config_value("setting", cli_namespace=Namespace(), json_values={})


def test_generalist_parser_marks_precedence_sensitive_options_unsupplied() -> None:
    args = build_arg_parser().parse_args([])
    assert args.advance_on_wins is None
    assert args.max_adventure_levels is None
    assert args.max_attempts_per_level is None
    assert args.adventure_start_level is None


def test_json_values_win_when_generalist_cli_options_are_unsupplied() -> None:
    args = build_arg_parser().parse_args([])
    raw_config = {
        "advance_on_wins": 2,
        "max_adventure_levels": 8,
        "max_attempts_per_level": 9,
        "adventure_start_level": 4,
    }
    config = build_config(args, raw_config)
    assert {key: config[key] for key in raw_config} == raw_config


def test_explicit_generalist_cli_values_win_over_json() -> None:
    args = build_arg_parser().parse_args(
        [
            "--advance-on-wins",
            "3",
            "--max-adventure-levels",
            "7",
            "--max-attempts-per-level",
            "6",
            "--adventure-start-level",
            "5",
        ]
    )
    config = build_config(
        args,
        {
            "advance_on_wins": 2,
            "max_adventure_levels": 8,
            "max_attempts_per_level": 9,
            "adventure_start_level": 4,
        },
    )
    assert config["advance_on_wins"] == 3
    assert config["max_adventure_levels"] == 7
    assert config["max_attempts_per_level"] == 6
    assert config["adventure_start_level"] == 5


def test_global_generalist_defaults_are_stable() -> None:
    config = build_config(build_arg_parser().parse_args([]), {})
    assert config["run_mode"] == GENERALIST_TRAIN
    assert config["advance_on_wins"] == 1
    assert config["max_adventure_levels"] == 5
    assert config["max_attempts_per_level"] == 10
    assert config["adventure_start_level"] == 1
    assert config["seed_list"] == GENERALIST_LOADOUT
    assert config["plant_types"] == [1, 1, 0, 0]


def test_ignored_generalist_json_field_emits_actionable_warning() -> None:
    args = build_arg_parser().parse_args([])
    with pytest.warns(IgnoredLegacyConfigWarning, match="enable_fusion_diagnostics"):
        config = build_config(args, {"enable_fusion_diagnostics": True})
    assert "enable_fusion_diagnostics" not in config


@pytest.mark.parametrize(
    "raw_config",
    [
        {"proximity_penalty": 0.125},
        {"reward": {"proximity_penalty": 0.125}},
    ],
)
def test_unused_reward_field_warns_but_preserves_resolved_shape(raw_config: dict) -> None:
    args = build_arg_parser().parse_args([])
    with pytest.warns(IgnoredLegacyConfigWarning, match="proximity_penalty"):
        config = build_config(args, raw_config)
    assert config["reward"]["proximity_penalty"] == 0.125


def test_quick_wait_mode_defaults_follow_normal_precedence() -> None:
    quick_args = build_arg_parser().parse_args(["--quick-wait"])
    assert build_config(quick_args, {})["board_timeout"] == 60.0
    assert build_config(quick_args, {"board_timeout": 75.0})["board_timeout"] == 75.0
    cli_args = build_arg_parser().parse_args(["--quick-wait", "--board-timeout", "45"])
    assert build_config(cli_args, {"board_timeout": 75.0})["board_timeout"] == 45.0


@pytest.mark.parametrize(
    ("cli", "json_mode", "expected_mode"),
    [
        (["--adventure-generalist-eval"], GENERALIST_TRAIN, GENERALIST_EVAL),
        (["--adventure-generalist-train"], GENERALIST_EVAL, GENERALIST_TRAIN),
        ([], GENERALIST_EVAL, GENERALIST_EVAL),
        ([], GENERALIST_TRAIN, GENERALIST_TRAIN),
    ],
)
def test_generalist_cli_shortcuts_win_over_json_run_mode(
    cli: list[str],
    json_mode: str,
    expected_mode: str,
) -> None:
    config = build_config(build_arg_parser().parse_args(cli), {"run_mode": json_mode})
    assert config["run_mode"] == expected_mode


def test_conflicting_generalist_mode_forms_are_rejected() -> None:
    args = build_arg_parser().parse_args(
        ["--run-mode", GENERALIST_TRAIN, "--adventure-generalist-eval"]
    )
    with pytest.raises(SystemExit, match="--run-mode conflicts"):
        build_config(args, {})

    args = build_arg_parser().parse_args(
        ["--adventure-generalist-train", "--adventure-generalist-eval"]
    )
    with pytest.raises(SystemExit, match="mutually exclusive"):
        build_config(args, {})


@pytest.mark.parametrize(
    "obsolete_mode",
    ["fixed_train", "fixed_eval", "level3_specialist", "adventure_eval"],
)
def test_obsolete_run_modes_are_rejected(obsolete_mode: str) -> None:
    args = build_arg_parser().parse_args([])
    with pytest.raises(SystemExit, match="unsupported_run_mode"):
        build_config(args, {"run_mode": obsolete_mode})


def test_parser_exposes_only_maintained_mode_controls() -> None:
    parser = build_arg_parser()
    options = parser._option_string_actions
    assert "--adventure-generalist-train" in options
    assert "--adventure-generalist-eval" in options
    for obsolete in (
        "--train",
        "--eval",
        "--level3-train",
        "--level3-eval",
        "--target-level",
        "--adventure",
        "--adventure-eval",
        "--allow-missing-model-metadata",
        "--model-schedule",
        "--router-dry-run",
        "--dry-run-level",
        "--dry-run-unlocked-seeds",
        "--dry-run-available-seeds",
        "--experimental-dynamic-seed-slots",
        "--action-space-mode",
        "--plant-types",
        "--auto-select-seeds",
        "--max-steps",
    ):
        assert obsolete not in options


@pytest.mark.parametrize("obsolete_action_mode", ["fixed", "dynamic_14"])
def test_obsolete_action_space_modes_are_rejected(obsolete_action_mode: str) -> None:
    args = build_arg_parser().parse_args([])
    with pytest.raises(SystemExit, match="unsupported_action_space_mode"):
        build_config(args, {"action_space_mode": obsolete_action_mode})


def test_json_plant_types_preserve_duplicate_slot_order_and_validate_names() -> None:
    args = build_arg_parser().parse_args([])
    config = build_config(
        args,
        {"seed_list": GENERALIST_LOADOUT, "plant_types": [1, 1, 0, 0]},
    )
    assert config["plant_types"] == [1, 1, 0, 0]

    with pytest.raises(SystemExit, match="seed_plant_type_mismatch"):
        build_config(
            args,
            {"seed_list": GENERALIST_LOADOUT, "plant_types": [0, 1, 0, 0]},
        )


@pytest.mark.parametrize(
    ("cli", "raw_config", "expected"),
    [
        (["--stream-coach-platform", "twitch"], {"stream_coach_mode": "youtube"}, "twitch"),
        (["--stream-coach-mode", "youtube"], {"stream_coach_platform": "twitch"}, "youtube"),
        ([], {"stream_coach_mode": "youtube", "stream_coach_platform": "twitch"}, "youtube"),
    ],
)
def test_stream_coach_mode_aliases_resolve_as_one_semantic_field(
    cli: list[str],
    raw_config: dict,
    expected: str,
) -> None:
    config = build_config(build_arg_parser().parse_args(cli), raw_config)
    assert config["stream_coach_mode"] == expected
    assert config["stream_coach_platform"] == expected


def test_conflicting_stream_coach_cli_aliases_are_rejected() -> None:
    args = build_arg_parser().parse_args(
        ["--stream-coach-mode", "youtube", "--stream-coach-platform", "twitch"]
    )
    with pytest.raises(SystemExit, match="conflicting CLI aliases"):
        build_config(args, {})


def test_typed_sections_round_trip_without_flat_contract_drift() -> None:
    args = build_arg_parser().parse_args([])
    resolved = build_resolved_config(args, {})
    flat = resolved.to_flat_dict()

    assert flat == build_config(args, {})
    assert resolved.optimization.total_timesteps == flat["total_timesteps"]
    assert resolved.environment.board_timeout == flat["board_timeout"]
    assert resolved.seed_actions.seed_list == tuple(flat["seed_list"])
    assert resolved.adventure.advance_on_wins == flat["advance_on_wins"]
    assert resolved.model_contract.action_count == 701
    assert resolved.model_contract.action_decoder_version == "seedslot14x50_plus_wait_v1"
    assert resolved.model_contract.observation_version == "adventure_14slot_identity_v1"
    assert dict(resolved.reward) == flat["reward"]
    assert resolved.value_sources["learning_rate"] is ConfigSource.GLOBAL_DEFAULT

    flat["seed_list"].append("WallNut")
    flat["reward"][next(iter(flat["reward"]))] = 999.0
    assert resolved.seed_actions.seed_list == tuple(GENERALIST_LOADOUT)
    assert resolved.to_flat_dict()["seed_list"] == GENERALIST_LOADOUT
    assert 999.0 not in resolved.reward.values()

    with pytest.raises(TypeError):
        resolved._flat["seed_list"] = ()  # type: ignore[index]
    with pytest.raises(TypeError):
        resolved.value_sources["learning_rate"] = ConfigSource.CLI  # type: ignore[index]
    assert isinstance(resolved._flat["seed_list"], tuple)

    with pytest.raises(FrozenInstanceError):
        resolved.adventure.advance_on_wins = 99  # type: ignore[misc]


def test_removed_configuration_fields_are_absent() -> None:
    resolved = build_resolved_config(build_arg_parser().parse_args([]), {})
    flat = resolved.to_flat_dict()
    for obsolete in (
        "legacy_max_steps",
        "experimental_dynamic_seed_slots",
        "target_level",
        "adventure_eval_mode",
        "allow_missing_model_metadata",
        "incompatible_with_4slot_specialist",
    ):
        assert obsolete not in flat
    assert not hasattr(resolved.environment, "legacy_max_steps")
    assert not hasattr(resolved.seed_actions, "experimental_dynamic_seed_slots")
    assert not hasattr(resolved.adventure, "target_level")
    assert not hasattr(resolved.adventure, "adventure_eval_mode")
    assert not hasattr(resolved.artifacts, "allow_missing_model_metadata")
    assert not hasattr(resolved.model_contract, "incompatible_with_4slot_specialist")


def test_typed_config_exposes_immutable_value_provenance() -> None:
    args = build_arg_parser().parse_args(["--learning-rate", "0.002"])
    resolved = build_resolved_config(args, {"learning_rate": 0.001, "port": 40000})
    assert resolved.optimization.learning_rate == 0.002
    assert resolved.value_sources["learning_rate"] is ConfigSource.CLI
    assert resolved.value_sources["port"] is ConfigSource.JSON
    assert resolved.value_sources["gamma"] is ConfigSource.GLOBAL_DEFAULT


def test_tracked_generalist_config_preserves_protected_contract() -> None:
    raw_config = json.loads(GENERALIST_CONFIG.read_text(encoding="utf-8"))
    args = build_arg_parser().parse_args(["--adventure-generalist-train"])
    config = build_config(args, raw_config)
    assert config["run_mode"] == GENERALIST_TRAIN
    assert config["model_family"] == "ppo_adventure_generalist_14slot_identity_v1"
    assert config["seed_list"] == GENERALIST_LOADOUT
    assert config["plant_types"] == [1, 1, 0, 0]
    assert config["max_seed_slots"] == 14
    assert config["action_count"] == 701
    assert config["decoder_wait_action"] == 0
    assert config["placement_action_range"] == [1, 700]
    assert config["action_decoder_version"] == "seedslot14x50_plus_wait_v1"
    assert config["observation_version"] == "adventure_14slot_identity_v1"
    assert config["identity_seed_slots"] is True
    assert "incompatible_with_4slot_specialist" not in config


def _streamer_args(*extra: str) -> Namespace:
    return build_arg_parser().parse_args(
        [
            "--streamer-v1",
            "--streamer-platform",
            "mock",
            "--streamer-baseline-checkpoint",
            "baseline.zip",
            *extra,
        ]
    )


def test_streamer_intervention_interval_has_a_practical_minimum() -> None:
    with pytest.raises(SystemExit, match="invalid_streamer_intervention_interval"):
        build_config(
            _streamer_args("--streamer-intervention-interval-seconds", "0.099"),
            {},
        )
    assert build_config(
        _streamer_args("--streamer-intervention-interval-seconds", "0.1"),
        {},
    )["streamer_intervention_interval_seconds"] == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("option", "field_name"),
    [
        ("--streamer-twitch-client-id-env", "streamer_twitch_client_id_env"),
        ("--streamer-twitch-access-token-env", "streamer_twitch_access_token_env"),
        ("--streamer-twitch-broadcaster-id-env", "streamer_twitch_broadcaster_id_env"),
        ("--streamer-twitch-user-id-env", "streamer_twitch_user_id_env"),
        ("--streamer-viewer-hash-secret-env", "streamer_viewer_hash_secret_env"),
    ],
)
def test_streamer_env_fields_reject_literal_credential_shaped_values_without_echoing_them(
    option: str,
    field_name: str,
) -> None:
    literal = "oauth:literal-secret-must-not-escape"
    with pytest.raises(SystemExit) as captured:
        build_config(_streamer_args(option, literal), {})
    message = str(captured.value)
    assert "invalid_streamer_environment_variable_name" in message
    assert field_name in message
    assert literal not in message


@pytest.mark.parametrize(
    ("option", "field_name"),
    [
        ("--n-steps", "n_steps"),
        ("--batch-size", "batch_size"),
    ],
)
def test_streamer_rejects_nonpositive_rollout_dimensions(
    option: str,
    field_name: str,
) -> None:
    with pytest.raises(SystemExit) as captured:
        build_config(_streamer_args(option, "0"), {})
    assert field_name in str(captured.value)
