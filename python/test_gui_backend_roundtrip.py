from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Any, Mapping

import pvzrl_gui_process
from pvzrl_gui import PROJECT_ROOT, PvZDashboard
from pvzrl_gui_config import (
    FULL_ADVENTURE_CONTRACT,
    StreamerV1LaunchValues,
    TwitchEnvironmentNames,
)
from test_gui_commands import Var, _dashboard
from train_ppo import (
    build_arg_parser,
    build_config,
    env_metadata_for_config,
    execution_route_for_config,
    load_json,
)


def _resolved_gui_command(
    command: list[str],
    raw_overrides: Mapping[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Round-trip one GUI argv through the authoritative CLI/config path."""

    args = build_arg_parser().parse_args(command[2:])
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    raw_config = load_json(config_path)
    raw_config.update(raw_overrides or {})
    config = build_config(args, raw_config)
    return args, config


def _assert_full_adventure_contract(config: dict[str, Any]) -> None:
    contract = FULL_ADVENTURE_CONTRACT
    metadata = env_metadata_for_config(config)

    assert metadata["model_family"] == contract.model_family
    assert metadata["action_space_mode"] == contract.action_space_mode
    assert metadata["env_action_count"] == contract.action_count
    assert metadata["action_decoder_version"] == contract.action_decoder_version
    assert metadata["decoder_wait_action"] == contract.wait_action
    assert tuple(metadata["placement_action_range"]) == contract.placement_action_range
    assert metadata["observation_version"] == contract.observation_version
    assert tuple(metadata["observation_shape"]) == contract.observation_shape
    assert metadata["max_seed_slots"] == contract.max_seed_slots
    assert metadata["dynamic_seed_slots"] is True
    assert metadata["identity_seed_slots"] is True
    assert (metadata["rows"], metadata["cols"]) == (contract.rows, contract.cols)
    assert metadata["cells_per_seed_slot"] == contract.cells_per_seed_slot


def _assert_no_legacy_coach_flags(command: list[str]) -> None:
    flags = {argument for argument in command if argument.startswith("--")}
    assert "--human-coach-enabled" not in flags
    assert "--stream-coach-enabled" not in flags
    assert "--stream-coach-mode" not in flags
    assert "--stream-coach-platform" not in flags


def test_train_and_evaluation_gui_argv_round_trip_through_canonical_resolver() -> None:
    dashboard = _dashboard()

    train_command = dashboard._build_adventure_generalist_command()
    train_args, train_config = _resolved_gui_command(train_command)

    assert train_args.adventure_generalist_train is True
    assert train_args.adventure_generalist_eval is False
    assert execution_route_for_config(train_config) == "train"
    assert train_config["run_dir"] == "runs/generalist_snapshot"
    assert train_config["n_steps"] == 512
    assert train_config["batch_size"] == 64
    _assert_full_adventure_contract(train_config)
    _assert_no_legacy_coach_flags(train_command)

    # The fixture deliberately gives evaluation values that differ from the
    # training values.  The round-trip proves the Evaluation page owns its
    # form state rather than silently reusing Training inputs.
    eval_command = dashboard._build_adventure_generalist_eval_command()
    eval_args, eval_config = _resolved_gui_command(eval_command)

    assert eval_args.adventure_generalist_train is False
    assert eval_args.adventure_generalist_eval is True
    assert execution_route_for_config(eval_config) == "adventure_generalist_eval"
    assert eval_config["run_dir"] == "runs/generalist_eval_snapshot"
    assert eval_config["adventure_start_level"] == 3
    assert eval_config["max_adventure_levels"] == 9
    assert eval_config["max_attempts_per_level"] == 7
    assert eval_config["game_speed"] == 3.5
    assert eval_config["step_seconds"] == 0.08
    assert eval_config["adventure_soft_max_steps"] == 2100
    assert eval_config["adventure_hard_max_steps"] == 3600
    assert eval_config["adventure_final_wave_extension"] is False
    # The selected config enables both masks. The Evaluation form's explicit
    # false choices must still win through argparse and ConfigResolver.
    assert eval_config["wallnut_tactical_mask"] is False
    assert eval_config["fusion_action_mask_enabled"] is False
    assert eval_config["streamer_v1_enabled"] is False
    assert eval_config["human_coach_enabled"] is False
    assert eval_config["stream_coach_enabled"] is False
    _assert_full_adventure_contract(eval_config)
    _assert_no_legacy_coach_flags(eval_command)


def test_gui_explicit_false_booleans_override_loaded_json_true_values() -> None:
    dashboard = _dashboard()
    for variable_name in (
        "generalist_quick_wait_var",
        "generalist_wait_gameplay_ready_var",
        "generalist_unlock_curriculum_var",
        "generalist_randomize_seed_order_var",
        "generalist_replay_cleared_var",
        "generalist_tactical_masks_var",
        "generalist_wallnut_mask_var",
        "generalist_cherrybomb_mask_var",
        "generalist_fusion_action_mask_train_var",
        "human_coach_enabled_var",
        "human_coach_reward_var",
        "stream_coach_enabled_var",
        "stream_coach_reward_var",
        "coach_allow_fusion_planning_var",
        "fusion_bridge_enabled_var",
    ):
        getattr(dashboard, variable_name).set(False)

    expected_false = {
        "quick_wait",
        "wait_gameplay_ready",
        "unlock_aware_seed_curriculum",
        "randomize_seed_order",
        "adventure_replay_cleared_levels",
        "tactical_masks",
        "wallnut_tactical_mask",
        "cherrybomb_tactical_mask",
        "fusion_action_mask_enabled",
        "streamer_v1_enabled",
        "human_coach_enabled",
        "human_coach_reward",
        "stream_coach_enabled",
        "stream_coach_reward",
        "coach_allow_fusion_planning",
        "fusion_bridge_enabled",
    }
    command = dashboard._build_adventure_generalist_command()
    args, config = _resolved_gui_command(
        command,
        {key: True for key in expected_false},
    )

    for key in expected_false:
        assert config[key] is False, key
    assert args.quick_wait is False
    assert args.wait_gameplay_ready is False
    assert args.streamer_v1_enabled is False


def test_streamer_gui_argv_round_trip_uses_generalist_backend_without_secrets(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    dashboard = _dashboard()
    dashboard.streamer_config_path_var = Var("configs/streamer_full_v2.example.json")

    secret_values = {
        "ROUNDTRIP_CLIENT_ENV": "client-secret-value",
        "ROUNDTRIP_TOKEN_ENV": "access-token-secret-value",
        "ROUNDTRIP_BROADCASTER_ENV": "broadcaster-secret-value",
        "ROUNDTRIP_USER_ENV": "eventsub-user-secret-value",
        "ROUNDTRIP_HASH_ENV": "viewer-hash-secret-value",
    }
    for name, value in secret_values.items():
        monkeypatch.setenv(name, value)

    baseline = tmp_path / "baseline" / "model.zip"
    baseline.parent.mkdir(parents=True)
    baseline.write_bytes(b"synthetic-baseline")
    mock_script = tmp_path / "viewer_commands.jsonl"
    mock_script.write_text("# autonomous mock source\n", encoding="utf-8")
    run_dir = tmp_path / "streamer_experiment"
    launch_values = StreamerV1LaunchValues(
        platform="mock",
        baseline_checkpoint=baseline,
        run_dir=run_dir,
        live_status_path=run_dir / "live_status.json",
        adventure_start_level=4,
        max_adventure_levels=50,
        max_attempts_per_level=10,
        n_steps=500,
        batch_size=50,
        intervention_interval_seconds=2.0,
        command_ttl_seconds=10.0,
        command_queue_capacity=256,
        message_max_chars=256,
        policy_steps_per_cycle=25_000,
        checkpoint_policy_steps=5_000,
        evaluation_episodes=50,
        max_cycles=3,
        endurance_hours=0.0,
        bc_enabled=True,
        bc_coefficient=0.01,
        demonstration_capacity=4_096,
        demonstration_persist_every=512,
        bc_batch_size=32,
        bc_update_frequency=1,
        bc_min_demonstrations=8,
        twitch_environment_names=TwitchEnvironmentNames(
            client_id="ROUNDTRIP_CLIENT_ENV",
            access_token="ROUNDTRIP_TOKEN_ENV",
            broadcaster_id="ROUNDTRIP_BROADCASTER_ENV",
            user_id="ROUNDTRIP_USER_ENV",
            viewer_hash_secret="ROUNDTRIP_HASH_ENV",
        ),
        mock_script=mock_script,
    )

    command = dashboard._build_streamer_v1_command(launch_values)
    args, config = _resolved_gui_command(command)

    assert args.adventure_generalist_train is True
    assert args.streamer_v1_enabled is True
    assert execution_route_for_config(config) == "train"
    assert config["streamer_v1_enabled"] is True
    assert config["streamer_platform"] == "mock"
    assert Path(config["streamer_baseline_checkpoint"]) == baseline
    assert Path(config["run_dir"]) == run_dir
    assert config["adventure_start_level"] == 4
    assert config["n_steps"] == 500
    assert config["batch_size"] == 50
    assert config["streamer_policy_steps_per_cycle"] == 25_000
    assert config["streamer_checkpoint_policy_steps"] == 5_000
    assert config["streamer_demonstration_persist_every"] == 512
    _assert_full_adventure_contract(config)
    _assert_no_legacy_coach_flags(command)

    assert command.count("--streamer-v1") == 1
    assert command.count("--run-dir") == 1
    assert command.count("--live-status-path") == 1
    rendered_command = " ".join(command)
    rendered_config = json.dumps(config, default=str, sort_keys=True)
    for secret in secret_values.values():
        assert secret not in rendered_command
        assert secret not in rendered_config


def test_streamer_gui_false_booleans_override_loaded_json_true_values(
    tmp_path: Path,
) -> None:
    dashboard = _dashboard()
    dashboard.streamer_config_path_var = Var("configs/streamer_full_v2.example.json")
    dashboard.streamer_quick_wait_var = Var(False)
    dashboard.streamer_wait_gameplay_ready_var = Var(False)
    baseline = tmp_path / "baseline" / "model.zip"
    baseline.parent.mkdir(parents=True)
    baseline.write_bytes(b"synthetic-baseline")
    run_dir = tmp_path / "streamer-experiment"
    values = StreamerV1LaunchValues(
        platform="twitch",
        baseline_checkpoint=baseline,
        run_dir=run_dir,
        live_status_path=run_dir / "live_status.json",
        adventure_start_level=1,
        max_adventure_levels=50,
        max_attempts_per_level=10,
        n_steps=500,
        batch_size=50,
        intervention_interval_seconds=2.0,
        command_ttl_seconds=10.0,
        command_queue_capacity=256,
        message_max_chars=256,
        policy_steps_per_cycle=25_000,
        checkpoint_policy_steps=5_000,
        evaluation_episodes=50,
        max_cycles=0,
        endurance_hours=0.0,
        bc_enabled=False,
        bc_coefficient=0.01,
        demonstration_capacity=4_096,
        demonstration_persist_every=512,
        bc_batch_size=32,
        bc_update_frequency=1,
        bc_min_demonstrations=8,
        twitch_environment_names=TwitchEnvironmentNames(),
        mock_script=None,
    )

    command = dashboard._build_streamer_v1_command(values)
    args, config = _resolved_gui_command(
        command,
        {
            "quick_wait": True,
            "wait_gameplay_ready": True,
            "streamer_bc_enabled": True,
            "human_coach_enabled": True,
            "stream_coach_enabled": True,
        },
    )

    assert args.quick_wait is False
    assert args.wait_gameplay_ready is False
    assert args.streamer_bc_enabled is False
    assert config["quick_wait"] is False
    assert config["wait_gameplay_ready"] is False
    assert config["streamer_bc_enabled"] is False
    assert config["human_coach_enabled"] is False
    assert config["stream_coach_enabled"] is False


class _FakeVar:
    def __init__(self, value: str = "") -> None:
        self.value = value

    def get(self) -> str:
        return self.value

    def set(self, value: Any) -> None:
        self.value = str(value)


class _FakeButton:
    def __init__(self) -> None:
        self.state = ""

    def configure(self, *, state: str) -> None:
        self.state = state


class _RestartProcess:
    stdout = None

    def poll(self) -> None:
        return None

    def wait(self) -> int:
        return 0


def _process_dashboard(tmp_path: Path) -> PvZDashboard:
    dashboard = PvZDashboard.__new__(PvZDashboard)
    dashboard.project_root = tmp_path
    dashboard.repo_root = tmp_path
    dashboard.log_queue = queue.Queue(maxsize=20)
    dashboard._log_queue_put_lock = threading.Lock()
    dashboard._log_queue_drop_lock = threading.Lock()
    dashboard._log_queue_dropped_items = 0
    dashboard._reader_thread = None
    dashboard._stopper_thread = None
    dashboard._stopping_process = None
    dashboard._closing = False
    dashboard.active_process = None
    dashboard.active_process_name = ""
    dashboard.active_process_started_at = None
    dashboard.active_process_started_wall_time = None
    dashboard.active_run_path = ""
    dashboard.active_run_var = _FakeVar()
    dashboard.live_status_path = tmp_path / "live_status.json"
    dashboard.live_status_path_var = _FakeVar(str(dashboard.live_status_path))
    dashboard._live_status_reader = None
    dashboard.last_good_status = None
    dashboard.last_good_read_time = None
    dashboard.last_live_parse_error = ""
    dashboard.last_live_health = ""
    dashboard.live_writer_warning_emitted = False
    dashboard.process_lifecycle_state = "OFFLINE"
    dashboard.process_lifecycle_detail = ""
    dashboard.process_status_var = _FakeVar()
    dashboard.process_lifecycle_var = _FakeVar()
    dashboard.launch_buttons = [_FakeButton()]
    dashboard.stop_buttons = [_FakeButton()]
    dashboard._test_messages = []
    dashboard._append_log = dashboard._test_messages.append
    return dashboard


def test_nonzero_process_exit_enters_error_and_allows_safe_restart(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    dashboard = _process_dashboard(tmp_path)
    failed_process = object()
    dashboard.active_process = failed_process
    dashboard.active_process_name = "first run"
    dashboard.active_process_started_at = 1.0
    dashboard.active_process_started_wall_time = 1.0

    dashboard._handle_process_exit("first run", failed_process, 7)

    assert dashboard.active_process is None
    assert dashboard.process_lifecycle_state == "ERROR"
    assert dashboard.process_status_var.get() == "ERROR: first run exited 7"
    assert dashboard.launch_buttons[0].state == "normal"
    assert dashboard.stop_buttons[0].state == "disabled"

    restarted = _RestartProcess()
    popen_calls: list[tuple[list[str], dict[str, Any]]] = []

    def fake_popen(command: list[str], **kwargs: Any) -> _RestartProcess:
        popen_calls.append((list(command), dict(kwargs)))
        return restarted

    monkeypatch.setattr(pvzrl_gui_process.subprocess, "Popen", fake_popen)
    dashboard.launch_process("restart", ["python", "train.py"])
    assert dashboard._reader_thread is not None
    dashboard._reader_thread.join(timeout=1.0)

    assert len(popen_calls) == 1
    assert dashboard.active_process is restarted
    assert dashboard.active_process_name == "restart"
    assert dashboard.process_lifecycle_state == "STARTING"
    assert dashboard.process_status_var.get() == "STARTING: restart"
    assert dashboard.launch_buttons[0].state == "disabled"
    assert dashboard.stop_buttons[0].state == "normal"
