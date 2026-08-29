from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

from pvzrl_gui import PROJECT_ROOT, PvZDashboard


class Var:
    def __init__(self, value: Any) -> None:
        self.value = value

    def get(self) -> Any:
        return self.value

    def set(self, value: Any) -> None:
        self.value = value


def _dashboard() -> PvZDashboard:
    dashboard = PvZDashboard.__new__(PvZDashboard)
    dashboard.project_root = PROJECT_ROOT
    dashboard.repo_root = PROJECT_ROOT
    dashboard.live_status_path = Path("runs/live_status.json")
    values = {
        "generalist_config_path_var": "configs/ppo_adventure_generalist_full_v2.json",
        "eval_config_path_var": "configs/ppo_adventure_generalist_full_v2.json",
        "live_status_path_var": "runs/live_status.json",
        "train_lab_mode_var": "Normal",
        "eval_lab_mode_var": "Normal",
        "human_coach_enabled_var": False,
        "human_coach_reward_var": False,
        "human_coach_bonus_var": "",
        "human_coach_match_bonus_var": "",
        "human_coach_override_penalty_var": "",
        "human_coach_fusion_reward_var": "",
        "human_coach_tactical_reward_var": "",
        "human_coach_log_path_var": "runs/human_coach.jsonl",
        "human_coach_command_path_var": "runs/coach_commands.jsonl",
        "assisted_execution_mode_var": "override",
        "intervention_log_path_var": "logs/interventions/dashboard.jsonl",
        "stream_coach_enabled_var": False,
        "stream_coach_platform_var": "mock",
        "stream_coach_window_sec_var": "4",
        "stream_coach_min_votes_var": "3",
        "stream_coach_max_actions_per_minute_var": "18",
        "stream_coach_reward_var": False,
        "stream_coach_dry_run_var": True,
        "stream_coach_log_path_var": "runs/stream_coach.jsonl",
        "stream_coach_mock_script_var": "scripts/mock_stream_commands.jsonl",
        "coach_allow_fusion_planning_var": False,
        "fusion_bridge_enabled_var": False,
        "generalist_total_timesteps_var": "4444",
        "generalist_checkpoint_freq_var": "555",
        "generalist_n_steps_var": "512",
        "generalist_batch_size_var": "64",
        "generalist_initial_loadout_var": "SunFlower,SunFlower,Peashooter,Peashooter",
        "generalist_max_seed_slots_var": "14",
        "generalist_start_level_var": "1",
        "generalist_max_levels_var": "10",
        "generalist_max_attempts_var": "12",
        "generalist_game_speed_var": "4.0",
        "generalist_step_seconds_var": "0.05",
        "generalist_board_timeout_var": "63",
        "generalist_soft_max_steps_var": "2200",
        "generalist_hard_max_steps_var": "3700",
        "generalist_final_wave_extension_var": True,
        "generalist_quick_wait_var": True,
        "generalist_wait_gameplay_ready_var": True,
        "generalist_unlock_curriculum_var": True,
        "generalist_curriculum_var": "conservative",
        "generalist_randomize_seed_order_var": False,
        "generalist_unlock_delay_var": "1",
        "generalist_new_plant_prob_var": "0.2",
        "generalist_replay_cleared_var": True,
        "generalist_frontier_prob_var": "0.6",
        "generalist_recent_prob_var": "0.3",
        "generalist_maintenance_prob_var": "0.1",
        "generalist_frontier_win_streak_required_var": "2",
        "generalist_tactical_masks_var": True,
        "generalist_wallnut_mask_var": True,
        "generalist_cherrybomb_mask_var": True,
        "generalist_fusion_action_mask_train_var": True,
        "generalist_fusion_action_mask_eval_var": False,
        "generalist_resume_model_path_var": "",
        "generalist_run_dir_var": "runs/generalist_snapshot",
        "generalist_eval_model_path_var": "runs/generalist/model.zip",
        "eval_initial_loadout_var": "SunFlower,SunFlower,Peashooter,Peashooter",
        "eval_max_seed_slots_var": "14",
        "eval_start_level_var": "3",
        "eval_max_levels_var": "9",
        "eval_max_attempts_var": "7",
        "eval_game_speed_var": "3.5",
        "eval_step_seconds_var": "0.08",
        "eval_board_timeout_var": "77",
        "eval_soft_max_steps_var": "2100",
        "eval_hard_max_steps_var": "3600",
        "eval_final_wave_extension_var": False,
        "eval_quick_wait_var": False,
        "eval_wait_gameplay_ready_var": True,
        "eval_tactical_masks_var": True,
        "eval_wallnut_mask_var": False,
        "eval_cherrybomb_mask_var": True,
        "eval_run_dir_var": "runs/generalist_eval_snapshot",
    }
    for name, value in values.items():
        setattr(dashboard, name, Var(value))
    return dashboard


def _normalized(command: list[str]) -> list[str]:
    root = str(PROJECT_ROOT).replace("\\", "/")
    normalized = []
    for argument in command:
        text = str(argument).replace("\\", "/")
        if text == str(sys.executable).replace("\\", "/"):
            normalized.append("<python>")
        else:
            normalized.append(text.replace(root, "<root>"))
    return normalized


def _snapshot_hash(command: list[str]) -> str:
    encoded = json.dumps(_normalized(command), ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _commands() -> dict[str, Callable[[], list[str]]]:
    dashboard = _dashboard()

    def generalist_resume() -> list[str]:
        dashboard.generalist_resume_model_path_var.set("runs/generalist/checkpoint.zip")
        return dashboard._build_adventure_generalist_command()

    def coach_fusion() -> list[str]:
        dashboard.train_lab_mode_var.set("Fusion")
        dashboard.human_coach_reward_var.set(True)
        dashboard.human_coach_bonus_var.set("0.11")
        dashboard.human_coach_match_bonus_var.set("0.22")
        dashboard.human_coach_override_penalty_var.set("-0.33")
        dashboard.human_coach_fusion_reward_var.set("0.44")
        dashboard.human_coach_tactical_reward_var.set("0.55")
        dashboard.stream_coach_enabled_var.set(True)
        dashboard.stream_coach_reward_var.set(True)
        dashboard.stream_coach_dry_run_var.set(False)
        return dashboard._build_adventure_generalist_command()

    return {
        "generalist_fresh": dashboard._build_adventure_generalist_command,
        "generalist_resume": generalist_resume,
        "generalist_eval": dashboard._build_adventure_generalist_eval_command,
        "coach_stream_fusion": coach_fusion,
    }


EXPECTED_COMMAND_HASHES = {
    "generalist_fresh": "0e7ea5f580238319c55598ec45840ae3281b877c54493e959867e8d1f5cddfc8",
    "generalist_resume": "5302f2536322eba701fe6bda427f03ad53e7b15ae065172f0bbab2323157d8fe",
    "generalist_eval": "c89c30dfd194d8c1e4c75b83d9e0a01f3c8cadf7c88fea2b6337480efb6233b1",
    "coach_stream_fusion": "4ac57a4020acc99cb63a766506212241d913ec42f43d79fd42e09973d1c89de4",
}


@pytest.mark.parametrize("name", tuple(EXPECTED_COMMAND_HASHES))
def test_gui_command_snapshot(name: str) -> None:
    command = _commands()[name]()
    actual = _snapshot_hash(command)
    assert actual == EXPECTED_COMMAND_HASHES[name], (
        f"{name} argv snapshot changed: {actual}\n" + json.dumps(_normalized(command), indent=2)
    )


def test_gui_command_surface_is_generalist_only() -> None:
    dashboard = _dashboard()
    for obsolete_builder in (
        "_build_train_command",
        "_build_eval_command",
        "_build_adventure_command",
        "_build_level3_command",
    ):
        assert not hasattr(dashboard, obsolete_builder)

    for command in _commands().values():
        argv = command()
        assert "--action-space-mode" not in argv
        assert not any(flag in argv for flag in ("--train", "--eval", "--adventure-eval", "--level3-train", "--level3-eval"))

    coach_command = _commands()["coach_stream_fusion"]()
    assert "--coach-fusion-success-reward" not in coach_command
    assert "--coach-tactical-usefulness-reward" not in coach_command


def test_evaluation_command_uses_independent_eval_form_state() -> None:
    dashboard = _dashboard()
    command = dashboard._build_adventure_generalist_eval_command()

    def value(flag: str) -> str:
        return command[command.index(flag) + 1]

    assert value("--run-dir") == "runs/generalist_eval_snapshot"
    assert value("--adventure-start-level") == "3"
    assert value("--max-adventure-levels") == "9"
    assert value("--max-attempts-per-level") == "7"
    assert value("--game-speed") == "3.5"
    assert value("--step-seconds") == "0.08"
    assert value("--board-timeout") == "77"
    assert value("--adventure-soft-max-steps") == "2100"
    assert value("--adventure-hard-max-steps") == "3600"
    assert "--no-adventure-final-wave-extension" in command
    assert "--wallnut-tactical-mask" not in command
    assert "--no-wallnut-tactical-mask" in command
    assert "--cherrybomb-tactical-mask" in command
    assert "--human-coach-enabled" not in command
    assert "--no-human-coach" in command
    assert "--stream-coach-enabled" not in command
    assert "--no-stream-coach" in command
