"""Focused contracts for typed run configuration and precedence."""

from __future__ import annotations

import json
from argparse import Namespace
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from pvzrl_config import ConfigSource, IgnoredLegacyConfigWarning, resolve_config_value
from train_ppo import build_arg_parser, build_config, build_resolved_config, execution_route_for_config


ROOT = Path(__file__).resolve().parents[1]
MODEL_SCHEDULE = ROOT / "configs" / "model_schedule.json"
GENERALIST_CONFIG = ROOT / "configs" / "ppo_adventure_generalist_14slot_identity_v1.json"


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


def test_adventure_parser_marks_precedence_sensitive_options_unsupplied() -> None:
    args = build_arg_parser().parse_args([])
    assert args.advance_on_wins is None
    assert args.max_adventure_levels is None
    assert args.max_attempts_per_level is None
    assert args.adventure_start_level is None


def test_json_values_win_when_adventure_cli_options_are_unsupplied() -> None:
    args = build_arg_parser().parse_args([])
    raw_config = {
        "advance_on_wins": 2,
        "max_adventure_levels": 8,
        "max_attempts_per_level": 9,
        "adventure_start_level": 4,
    }
    config = build_config(args, raw_config)
    assert {key: config[key] for key in raw_config} == raw_config


def test_explicit_adventure_cli_values_win_over_json() -> None:
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


def test_global_adventure_defaults_remain_backward_compatible() -> None:
    config = build_config(build_arg_parser().parse_args([]), {})
    assert config["advance_on_wins"] == 1
    assert config["max_adventure_levels"] == 5
    assert config["max_attempts_per_level"] == 10
    assert config["adventure_start_level"] == 1


def test_ignored_legacy_json_field_emits_actionable_warning() -> None:
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
def test_unused_legacy_reward_field_warns_but_preserves_resolved_shape(raw_config: dict) -> None:
    args = build_arg_parser().parse_args([])
    with pytest.warns(IgnoredLegacyConfigWarning, match="proximity_penalty"):
        config = build_config(args, raw_config)
    assert config["reward"]["proximity_penalty"] == 0.125


def test_mode_defaults_are_applied_after_cli_and_json() -> None:
    quick_args = build_arg_parser().parse_args(["--quick-wait"])
    assert build_config(quick_args, {})["board_timeout"] == 60.0
    assert build_config(quick_args, {"board_timeout": 75.0})["board_timeout"] == 75.0
    cli_args = build_arg_parser().parse_args(["--quick-wait", "--board-timeout", "45"])
    assert build_config(cli_args, {"board_timeout": 75.0})["board_timeout"] == 45.0

    level3_args = build_arg_parser().parse_args(["--level3-train"])
    level3 = build_config(level3_args, {})
    assert level3["target_level"] == 3
    assert level3["seed_list"] == ["SunFlower", "Peashooter", "WallNut", "CherryBomb"]


@pytest.mark.parametrize(
    ("cli", "json_mode", "expected_mode"),
    [
        (["--adventure-generalist-eval"], "adventure_generalist_14slot_train", "adventure_generalist_14slot_eval"),
        (["--level3-train"], "adventure_generalist_14slot_train", "level3_specialist"),
        (["--train"], "adventure_eval", "fixed_train"),
        (["--eval"], "adventure_generalist_14slot_train", "fixed_eval"),
        ([], "adventure_generalist_14slot_train", "adventure_generalist_14slot_train"),
    ],
)
def test_explicit_cli_mode_shortcuts_win_over_json_run_mode(
    cli: list[str],
    json_mode: str,
    expected_mode: str,
) -> None:
    config = build_config(build_arg_parser().parse_args(cli), {"run_mode": json_mode})
    assert config["run_mode"] == expected_mode


def test_conflicting_explicit_run_mode_forms_are_rejected() -> None:
    args = build_arg_parser().parse_args(
        ["--run-mode", "fixed_train", "--adventure-generalist-eval"]
    )
    with pytest.raises(SystemExit, match="--run-mode conflicts"):
        build_config(args, {})


@pytest.mark.parametrize(
    ("cli", "expected"),
    [
        (["--run-mode", "adventure_generalist_14slot_train", "--train"], "adventure_generalist_14slot_train"),
        (["--run-mode", "adventure_eval", "--eval"], "adventure_eval"),
    ],
)
def test_generic_operation_flags_do_not_conflict_with_explicit_run_mode(
    cli: list[str],
    expected: str,
) -> None:
    assert build_config(build_arg_parser().parse_args(cli), {})["run_mode"] == expected


def test_explicit_fixed_eval_rejects_generic_train() -> None:
    args = build_arg_parser().parse_args(["--run-mode", "fixed_eval", "--train"])
    with pytest.raises(SystemExit, match="fixed_eval cannot be combined"):
        build_config(args, {})


def test_explicit_fixed_train_rejects_generic_eval() -> None:
    args = build_arg_parser().parse_args(["--run-mode", "fixed_train", "--eval"])
    with pytest.raises(SystemExit, match="fixed_train cannot be combined"):
        build_config(args, {})


@pytest.mark.parametrize(
    ("cli", "raw_config", "configured_explicit", "expected_route"),
    [
        (["--adventure-eval", "--eval"], {}, False, "adventure_eval"),
        (["--adventure-generalist-eval", "--eval"], {}, False, "adventure_eval"),
        (["--run-mode", "fixed_eval"], {}, True, "fixed_eval"),
        ([], {"run_mode": "fixed_train"}, True, "train"),
        ([], {}, False, ""),
    ],
)
def test_main_execution_route_uses_resolved_mode_not_raw_generic_flags(
    cli: list[str],
    raw_config: dict,
    configured_explicit: bool,
    expected_route: str,
) -> None:
    args = build_arg_parser().parse_args(cli)
    config = build_config(args, raw_config)
    assert (
        execution_route_for_config(
            config,
            args,
            configured_mode_explicit=configured_explicit,
        )
        == expected_route
    )


def test_json_plant_types_preserve_duplicate_slot_order_and_validate_names() -> None:
    args = build_arg_parser().parse_args([])
    config = build_config(
        args,
        {
            "seed_list": ["SunFlower", "SunFlower", "Peashooter", "Peashooter"],
            "plant_types": [1, 1, 0, 0],
        },
    )
    assert config["plant_types"] == [1, 1, 0, 0]

    with pytest.raises(SystemExit, match="seed_plant_type_mismatch"):
        build_config(
            args,
            {
                "seed_list": ["SunFlower", "Peashooter"],
                "plant_types": [0, 1],
            },
        )


def test_explicit_cli_plant_types_win_over_json_before_alignment_check() -> None:
    args = build_arg_parser().parse_args(["--plant-types", "1,0"])
    config = build_config(
        args,
        {"seed_list": ["SunFlower", "Peashooter"], "plant_types": [0, 1]},
    )
    assert config["plant_types"] == [1, 0]


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
    assert resolved.model_contract.action_count == flat["action_count"]
    assert dict(resolved.reward) == flat["reward"]
    assert resolved.value_sources["learning_rate"] is ConfigSource.GLOBAL_DEFAULT

    flat["seed_list"].append("WallNut")
    flat["reward"][next(iter(flat["reward"]))] = 999.0
    assert resolved.seed_actions.seed_list == ("SunFlower", "Peashooter")
    assert resolved.to_flat_dict()["seed_list"] == ["SunFlower", "Peashooter"]
    assert 999.0 not in resolved.reward.values()

    with pytest.raises(TypeError):
        resolved._flat["seed_list"] = ()  # type: ignore[index]
    with pytest.raises(TypeError):
        resolved.value_sources["learning_rate"] = ConfigSource.CLI  # type: ignore[index]
    assert isinstance(resolved._flat["seed_list"], tuple)

    with pytest.raises(FrozenInstanceError):
        resolved.adventure.advance_on_wins = 99  # type: ignore[misc]


def test_typed_config_exposes_immutable_value_provenance() -> None:
    args = build_arg_parser().parse_args(["--learning-rate", "0.002"])
    resolved = build_resolved_config(args, {"learning_rate": 0.001, "port": 40000})
    assert resolved.optimization.learning_rate == 0.002
    assert resolved.value_sources["learning_rate"] is ConfigSource.CLI
    assert resolved.value_sources["port"] is ConfigSource.JSON
    assert resolved.value_sources["gamma"] is ConfigSource.GLOBAL_DEFAULT


def test_tracked_model_schedule_paths_are_runnable_from_repository_root() -> None:
    schedule = json.loads(MODEL_SCHEDULE.read_text(encoding="utf-8"))
    assert schedule["example_status"] == "validated_local_example"
    assert schedule["path_base"] == "repository_root"
    assert schedule["requires_local_model_artifacts"] is True
    stages = schedule["stages"]
    assert [stage["id"] for stage in stages] == ["early", "post_unlock_utility"]
    missing_local_artifacts = []
    for stage in stages:
        configured_path = Path(stage["model_path"])
        assert not configured_path.is_absolute()
        resolved_path = ROOT / configured_path
        assert resolved_path.suffix == ".zip"
        if not resolved_path.is_file():
            missing_local_artifacts.append(str(configured_path))
    if missing_local_artifacts:
        pytest.skip(
            "validated example requires optional gitignored models: "
            + ", ".join(missing_local_artifacts)
        )


def test_tracked_generalist_config_preserves_model_metadata_values() -> None:
    raw_config = json.loads(GENERALIST_CONFIG.read_text(encoding="utf-8"))
    args = build_arg_parser().parse_args(["--adventure-generalist-train"])
    with pytest.warns(IgnoredLegacyConfigWarning, match="enable_fusion_diagnostics"):
        config = build_config(args, raw_config)
    assert config["run_mode"] == "adventure_generalist_14slot_train"
    assert config["model_family"] == "ppo_adventure_generalist_14slot_identity_v1"
    assert config["seed_list"] == ["SunFlower", "SunFlower", "Peashooter", "Peashooter"]
    assert config["plant_types"] == [1, 1, 0, 0]
    assert config["max_seed_slots"] == 14
    assert config["checkpoint_warm_start_reason"] == "new_incompatible_architecture"
