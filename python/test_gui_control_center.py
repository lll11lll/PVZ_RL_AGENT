from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import tkinter as tk

from pvzrl_gui import PvZDashboard
from pvzrl_gui_config import FULL_ADVENTURE_CONTRACT, validate_streamer_v1_form


def _write_full_model(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"synthetic-full-adventure-model")
    (path.parent / "model_metadata.json").write_text(
        json.dumps(FULL_ADVENTURE_CONTRACT.expected_metadata()),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def control_center(tmp_path: Path):
    try:
        root = tk.Tk()
    except tk.TclError as exc:  # pragma: no cover - display-less CI fallback.
        pytest.skip(f"Tk display unavailable: {exc}")
    root.withdraw()
    dashboard = PvZDashboard(root, tmp_path / "live_status.json")
    try:
        yield dashboard, root
    finally:
        if not dashboard._destroyed:
            dashboard._on_close()
        try:
            root.update_idletasks()
        except tk.TclError:
            pass


def test_control_center_constructs_all_pages_and_preserves_form_state(control_center) -> None:
    dashboard, root = control_center
    expected_pages = {
        "Dashboard",
        "Training",
        "Evaluation",
        "Streamer",
        "Runs & Models",
        "Diagnostics",
        "Local Coach",
        "Settings",
    }
    assert set(dashboard.page_frames) == expected_pages
    assert len(dashboard.fusion_tile_buttons) == 60

    dashboard.generalist_total_timesteps_var.set("12345")
    dashboard._show_page("Evaluation")
    dashboard._show_page("Training")
    root.update_idletasks()
    assert dashboard.generalist_total_timesteps_var.get() == "12345"


def test_config_loading_updates_only_the_target_form(control_center) -> None:
    dashboard, _root = control_center
    dashboard.generalist_run_dir_var.set("runs/prepared-training")
    dashboard.eval_run_dir_var.set("runs/prepared-evaluation")
    dashboard.streamer_run_dir_var.set("runs/prepared-streamer")

    dashboard._apply_config_to_forms(
        {
            "run_dir": "runs/loaded-evaluation",
            "adventure_start_level": 7,
            "max_adventure_levels": 42,
            "max_attempts_per_level": 6,
        },
        target="evaluation",
    )

    assert dashboard.eval_run_dir_var.get() == "runs/loaded-evaluation"
    assert dashboard.eval_start_level_var.get() == "7"
    assert dashboard.generalist_run_dir_var.get() == "runs/prepared-training"
    assert dashboard.streamer_run_dir_var.get() == "runs/prepared-streamer"

    dashboard._apply_config_to_forms(
        {
            "run_dir": "runs/loaded-streamer",
            "adventure_start_level": 9,
            "max_adventure_levels": 50,
            "max_attempts_per_level": 4,
            "quick_wait": False,
            "wait_gameplay_ready": True,
        },
        target="streamer",
    )

    assert dashboard.streamer_run_dir_var.get() == "runs/loaded-streamer"
    assert dashboard.streamer_start_level_var.get() == "9"
    assert dashboard.streamer_max_attempts_var.get() == "4"
    assert dashboard.streamer_quick_wait_var.get() is False
    assert dashboard.generalist_run_dir_var.get() == "runs/prepared-training"
    assert dashboard.eval_run_dir_var.get() == "runs/loaded-evaluation"


def test_settings_config_target_saves_selected_form_and_adopts_path(
    control_center,
    tmp_path: Path,
    monkeypatch,
) -> None:
    dashboard, _root = control_center
    dashboard._show_page("Settings")
    dashboard.settings_config_target_var.set("Evaluation")
    dashboard.eval_run_dir_var.set("runs/eval-only")
    dashboard.generalist_run_dir_var.set("runs/train-kept")
    destination = tmp_path / "saved-evaluation.json"
    monkeypatch.setattr(
        "pvzrl_gui.filedialog.asksaveasfilename",
        lambda **_kwargs: str(destination),
    )

    dashboard.save_active_form_config()

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["run_mode"] == "adventure_generalist_14slot_eval"
    assert payload["run_dir"] == "runs/eval-only"
    assert payload["initial_loadout"] == [
        "SunFlower",
        "SunFlower",
        "Peashooter",
        "Peashooter",
    ]
    assert payload["reward_policy_version"] == "generalized_threat_v2"
    assert payload["reward"] == {
        "fusion_success_reward": 0.15,
        "illegal_action_penalty": 0.1,
        "kill_reward": 0.25,
        "loss_penalty": 15.0,
        "mower_loss_penalty": 2.0,
        "threat_delta_clip": 1.0,
        "threat_delta_coef": 0.75,
        "wave_reward": 2.0,
        "win_reward": 15.0,
    }
    assert dashboard.eval_config_path_var.get() == str(destination)
    assert dashboard.generalist_run_dir_var.get() == "runs/train-kept"


def test_reset_defaults_replaces_custom_config_source_for_selected_workflow(
    control_center,
) -> None:
    dashboard, _root = control_center
    canonical = str(Path("configs") / "ppo_adventure_generalist_full_v2.json")

    dashboard._show_page("Settings")
    dashboard.settings_config_target_var.set("Evaluation")
    dashboard.eval_config_path_var.set("configs/custom-evaluation.json")
    dashboard.reset_gui_defaults()
    assert dashboard.eval_config_path_var.get() == canonical

    dashboard.settings_config_target_var.set("Training")
    dashboard.generalist_config_path_var.set("configs/custom-training.json")
    dashboard.reset_gui_defaults()
    assert dashboard.generalist_config_path_var.get() == canonical


def test_full_adventure_action_inspector_reaches_row_six_slot_fourteen(control_center) -> None:
    dashboard, _root = control_center
    dashboard.action_inspector_id_var.set("840")
    dashboard.inspect_action_id()
    rendered = dashboard.action_inspector_result_var.get()
    assert "slot 14" in rendered
    assert "R6 C10" in rendered
    assert "row=5 col=9" in rendered


def test_status_projection_uses_canonical_streamer_schema_and_geometry(
    control_center, tmp_path: Path
) -> None:
    dashboard, _root = control_center
    payload = {
        "status": "running",
        "run_mode": "adventure_generalist_14slot_train",
        "active_run": str(tmp_path / "streamer"),
        "streamer_v1_enabled": True,
        "streamer_mode": "STREAM_TRAIN",
        "streamer_platform": "mock",
        "streamer_cycle": 4,
        "current_model_ppo_steps": 125000,
        "baseline_model_ppo_steps": 100000,
        "next_evaluation_countdown": 5000,
        "next_adventure_level": 8,
        "viewer_command_queue_depth": 3,
        "viewer_commands_accepted_count": 12,
        "viewer_commands_rejected_count": 5,
        "viewer_commands_invalid_count": 2,
        "viewer_intervention_count": 7,
        "last_action_source": "TWITCH",
        "last_viewer_action": {
            "command_type": "slot",
            "requested_slot": 13,
            "requested_row": 5,
            "requested_col": 9,
            "execution_status": "executed_verified",
        },
        "ppo_updates_enabled": True,
        "bc_updates_enabled": True,
        "evaluation_chat_control": False,
        "bc_demonstration_count": 9,
        "bc_demo_rejected_count": 1,
        "bc_update_count": 2,
        "bc_loss": 0.125,
        "board_rows": 6,
        "board_cols": 10,
        # Canonical row diagnostics are a mapping, never board geometry.
        "rows": {"0": {"danger": 0.1}, "5": {"danger": 0.9}},
        "gameplay": {"sun": 175, "plants": 10, "zombies": 6},
        "reward_policy_version": "generalized_threat_v2",
        "reward": {
            "reward_policy_version": "generalized_threat_v2",
            "episode_reward": 6.25,
            "kill_reward_total": 1.0,
            "wave_reward_total": 2.0,
            "terminal_reward_total": 0.0,
            "threat_reward_total": 0.35,
            "mower_penalty_total": -2.0,
            "illegal_action_penalty_total": -0.1,
            "fusion_reward_total": 0.15,
            "threat_before": 0.42,
            "threat_after": 0.31,
            "threat_raw_delta": 0.8,
            "threat_clipped_delta": 0.11,
            "reward_components_match": True,
            "reward_unattributed_adjustment_total": 0.0,
        },
    }

    dashboard._render(payload, health="LIVE", using_last_good=False)

    assert dashboard.stream_phase_var.get() == "STREAM_TRAIN"
    assert dashboard.stream_cycle_var.get() == "4"
    assert dashboard.stream_accepted_var.get() == "12"
    assert dashboard.stream_rejected_var.get() == "5"
    assert dashboard.stream_total_commands_var.get() == "17"
    assert dashboard.stream_invalid_var.get() == "2"
    assert dashboard.stream_latest_command_var.get() == "Slot 14 at R6 C10"
    assert dashboard.stream_last_result_var.get() == "executed_verified"
    assert dashboard.stream_action_source_var.get() == "TWITCH"
    assert dashboard.stream_board_geometry_var.get() == "6×10 live · 6×10 policy"
    assert dashboard.stream_ppo_enabled_var.get() == "ENABLED"
    assert dashboard.stream_bc_updates_var.get() == "ENABLED"
    assert dashboard.stream_eval_chat_control_var.get() == "DISABLED"
    reward_panel = dashboard.last_panel_content["Reward Breakdown"]
    assert "generalized_threat_v2" in reward_panel
    assert "threat_before" in reward_panel
    assert "components_match" in reward_panel

    dashboard._render_no_status("MISSING")
    assert dashboard.stream_phase_var.get() == "OFFLINE"
    assert dashboard.header_health_var.get() == "MISSING"


def test_streamer_event_history_switches_runs_without_leaking_rows(
    control_center, tmp_path: Path
) -> None:
    dashboard, _root = control_center
    viewer_hash = "ab" * 32

    def write_event(run_dir: Path, command_id: str, action_id: int, row: int) -> None:
        path = run_dir / "logs" / "streamer_events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "event": "executed_decision",
                    "command_id": command_id,
                    "viewer_hash": viewer_hash,
                    "action_source": "TWITCH",
                    "execution_status": "executed_verified",
                    "bridge_reason": "success",
                    "viewer_action_id": action_id,
                    "parsed_fields": {
                        "kind": "slot",
                        "seed_slot": 13,
                        "row": row,
                        "column": 9,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )

    run_a = tmp_path / "run-a"
    run_b = tmp_path / "run-b"
    write_event(run_a, "command-a", 840, 5)
    write_event(run_b, "command-b", 830, 4)

    dashboard.refresh_streamer_event_history(run_a)
    first = [dashboard.stream_event_tree.item(item, "values") for item in dashboard.stream_event_tree.get_children("")]
    assert len(first) == 1
    assert any("hash:abababab" in value for value in first[0])
    assert any("R6 C10" in value for value in first[0])
    assert any("slot 14" in value for value in first[0])
    assert dashboard.stream_latest_command_var.get() == "Slot 14 at R6 C10"
    assert dashboard.stream_last_result_var.get() == "success"

    dashboard.refresh_streamer_event_history(run_b)
    second = [dashboard.stream_event_tree.item(item, "values") for item in dashboard.stream_event_tree.get_children("")]
    assert len(second) == 1
    assert any("R5 C10" in value for value in second[0])
    assert all("R6 C10" not in value for row_values in second for value in row_values)

    empty_run = tmp_path / "empty-run"
    empty_run.mkdir()
    dashboard.refresh_streamer_event_history(empty_run)
    assert dashboard.stream_event_tree.get_children("") == ()
    assert dashboard.stream_latest_command_var.get() == "-"
    assert dashboard.stream_last_result_var.get() == "-"

    dashboard.stream_latest_command_var.set("stale command")
    dashboard.stream_last_result_var.set("stale result")
    dashboard.refresh_streamer_event_history("")
    assert dashboard.stream_event_tree.get_children("") == ()
    assert dashboard.stream_latest_command_var.get() == "-"
    assert dashboard.stream_last_result_var.get() == "-"


def test_live_streamer_status_resolves_nested_cycle_to_experiment_history(
    control_center, tmp_path: Path
) -> None:
    dashboard, _root = control_center
    experiment = tmp_path / "streamer-experiment"
    cycle_run = experiment / "cycles" / "cycle_000002" / "train"
    cycle_run.mkdir(parents=True)
    (experiment / "streamer_state.json").write_text("{}\n", encoding="utf-8")
    event_path = experiment / "logs" / "streamer_events.jsonl"
    event_path.parent.mkdir(parents=True)
    event_path.write_text(
        json.dumps(
            {
                "event": "executed_decision",
                "command_id": "nested-cycle-command",
                "viewer_hash": "ef" * 32,
                "action_source": "TWITCH",
                "execution_status": "executed_verified",
                "bridge_reason": "success",
                "viewer_action_id": 840,
                "parsed_fields": {
                    "kind": "slot",
                    "seed_slot": 13,
                    "row": 5,
                    "column": 9,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dashboard.streamer_run_dir_var.set(str(tmp_path / "unrelated-configured-run"))
    dashboard.live_status_path.write_text(
        json.dumps(
            {
                "status": "running",
                "health": "LIVE",
                "updated_at": time.time(),
                "streamer_v1_enabled": True,
                "streamer_mode": "STREAM_TRAIN",
                "active_run": str(cycle_run),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dashboard.refresh_diagnostics_now()

    assert dashboard._streamer_event_path == experiment.resolve()
    values = [
        dashboard.stream_event_tree.item(item, "values")
        for item in dashboard.stream_event_tree.get_children("")
    ]
    assert len(values) == 1
    assert any("R6 C10" in value for value in values[0])


def test_artifact_refresh_is_async_explicit_and_compatibility_gated(
    control_center, tmp_path: Path
) -> None:
    dashboard, root = control_center
    runs = tmp_path / "runs"
    model = _write_full_model(runs / "valid" / "model.zip")
    (model.parent / "summary.json").write_text(
        json.dumps({"status": "complete", "model_steps": 321}),
        encoding="utf-8",
    )
    dashboard.runs_root_var.set(str(runs))
    dashboard.generalist_eval_model_path_var.set("")
    dashboard.generalist_resume_model_path_var.set("")
    dashboard.streamer_baseline_checkpoint_var.set("")

    dashboard.refresh_artifact_index()
    deadline = time.monotonic() + 5.0
    while "Scanning" in dashboard.artifact_index_status_var.get() and time.monotonic() < deadline:
        root.update()
        time.sleep(0.01)

    assert "1 models" in dashboard.artifact_index_status_var.get()
    assert dashboard.generalist_eval_model_path_var.get() == ""
    assert dashboard.generalist_resume_model_path_var.get() == ""
    assert dashboard.streamer_baseline_checkpoint_var.get() == ""

    item = dashboard.artifact_tree.get_children("")[0]
    dashboard.artifact_tree.selection_set(item)
    dashboard.use_selected_artifact_for_evaluation()
    assert Path(dashboard.generalist_eval_model_path_var.get()) == model.resolve()


def test_model_refresh_opens_explicit_artifact_selector_without_selecting(control_center) -> None:
    dashboard, _root = control_center
    dashboard.generalist_eval_model_path_var.set("eval-kept.zip")
    dashboard.generalist_resume_model_path_var.set("resume-kept.zip")
    dashboard.streamer_baseline_checkpoint_var.set("baseline-kept.zip")
    refreshes: list[bool] = []
    dashboard.refresh_artifact_index = lambda: refreshes.append(True)

    dashboard.refresh_generalist_models()

    assert dashboard.current_page == "Runs & Models"
    assert refreshes == [True]
    assert dashboard.generalist_eval_model_path_var.get() == "eval-kept.zip"
    assert dashboard.generalist_resume_model_path_var.get() == "resume-kept.zip"
    assert dashboard.streamer_baseline_checkpoint_var.get() == "baseline-kept.zip"


def test_start_actions_use_validators_and_streamer_wraps_existing_backend(
    control_center, tmp_path: Path
) -> None:
    dashboard, _root = control_center
    launched: list[tuple[str, list[str]]] = []
    dashboard.launch_process = lambda name, command: launched.append((name, list(command)))
    dashboard.live_status_path_var.set(str(tmp_path / "live_status.json"))

    dashboard.generalist_run_dir_var.set(str(tmp_path / "fresh-train"))
    dashboard.start_adventure_generalist_train()
    assert launched[-1][0] == "Start Adventure Generalist Train"

    model = _write_full_model(tmp_path / "baseline" / "model.zip")
    dashboard.generalist_eval_model_path_var.set(str(model))
    dashboard.eval_run_dir_var.set(str(tmp_path / "eval"))
    dashboard.start_adventure_generalist_eval()
    assert launched[-1][0] == "Start Adventure Generalist Eval"

    mock_script = tmp_path / "mock.jsonl"
    mock_script.write_text(
        json.dumps({"command": "!slot 14 6 10", "viewer_hash": "cd" * 32}) + "\n",
        encoding="utf-8",
    )
    dashboard.streamer_platform_var.set("mock")
    dashboard.streamer_baseline_checkpoint_var.set(str(model))
    dashboard.streamer_run_dir_var.set(str(tmp_path / "streamer"))
    dashboard.streamer_mock_script_var.set(str(mock_script))
    dashboard.start_streamer_v1()

    name, command = launched[-1]
    assert name == "Start Streamer V1"
    assert "--adventure-generalist-train" in command
    assert "--streamer-v1" in command
    assert "--streamer-platform" in command
    assert "--stream-coach-enabled" not in command
    assert "--human-coach-enabled" not in command

    validation = validate_streamer_v1_form(
        dashboard._streamer_form_mapping(), base_dir=dashboard.project_root
    )
    assert validation.ok
