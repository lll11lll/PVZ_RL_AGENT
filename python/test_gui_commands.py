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
        "seed_list_var": "SunFlower,Peashooter,WallNut,CherryBomb",
        "total_timesteps_var": "1111",
        "max_steps_var": "222",
        "step_seconds_var": "0.07",
        "game_speed_var": "3.5",
        "start_sun_var": "450",
        "board_timeout_var": "61",
        "gameplay_ready_timeout_var": "31",
        "checkpoint_freq_var": "333",
        "quick_wait_var": True,
        "wait_gameplay_ready_var": True,
        "auto_select_seeds_var": True,
        "debug_perf_var": True,
        "fusion_policy_var": "observe",
        "run_dir_var": "",
        "run_name_var": "snapshot run",
        "model_path_var": "runs/fixed/model.zip",
        "episodes_var": "7",
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
        "adventure_eval_var": True,
        "adventure_model_path_var": "runs/adventure/model.zip",
        "adventure_episodes_var": "8",
        "adventure_game_speed_var": "3.75",
        "adventure_step_seconds_var": "0.06",
        "adventure_soft_max_steps_var": "2100",
        "adventure_hard_max_steps_var": "3600",
        "adventure_final_wave_extension_var": True,
        "adventure_board_timeout_var": "62",
        "adventure_quick_wait_var": True,
        "adventure_wait_gameplay_ready_var": True,
        "adventure_auto_select_seeds_var": True,
        "adventure_advance_on_wins_var": True,
        "adventure_advance_wins_var": "2",
        "adventure_max_levels_var": "6",
        "adventure_max_attempts_var": "11",
        "adventure_seed_list_var": "SunFlower,Peashooter,WallNut,CherryBomb",
        "adventure_plant_types_var": "1,0,3,2",
        "adventure_tactical_masks_var": True,
        "adventure_wallnut_mask_var": True,
        "adventure_cherrybomb_mask_var": True,
        "adventure_fusion_policy_var": "none",
        "generalist_total_timesteps_var": "4444",
        "generalist_checkpoint_freq_var": "555",
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
        "generalist_eval_episodes_var": "9",
        "level3_mode_var": "train",
        "level3_target_level_var": "3",
        "level3_model_path_var": "runs/level3/model.zip",
        "level3_total_timesteps_var": "6666",
        "level3_episodes_var": "25",
        "level3_seed_list_var": "SunFlower,Peashooter,WallNut,CherryBomb",
        "level3_plant_types_var": "1,0,3,2",
        "level3_game_speed_var": "4.0",
        "level3_step_seconds_var": "0.05",
        "level3_max_steps_var": "1200",
        "level3_board_timeout_var": "64",
        "level3_tactical_masks_var": True,
        "level3_wallnut_mask_var": True,
        "level3_cherrybomb_mask_var": True,
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

    def fixed_resume() -> list[str]:
        return dashboard._build_train_command(resume=True)

    def generalist_resume() -> list[str]:
        dashboard.generalist_resume_model_path_var.set("runs/generalist/checkpoint.zip")
        return dashboard._build_adventure_generalist_command()

    def level3_eval() -> list[str]:
        dashboard.level3_mode_var.set("eval")
        return dashboard._build_level3_command()

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
        "fixed_train": lambda: dashboard._build_train_command(),
        "fixed_resume": fixed_resume,
        "fixed_eval": lambda: dashboard._build_eval_command(),
        "adventure_eval": dashboard._build_adventure_command,
        "generalist_fresh": dashboard._build_adventure_generalist_command,
        "generalist_resume": generalist_resume,
        "generalist_eval": dashboard._build_adventure_generalist_eval_command,
        "level3_train": dashboard._build_level3_command,
        "level3_eval": level3_eval,
        "coach_stream_fusion": coach_fusion,
    }


EXPECTED_COMMAND_HASHES = {
    "fixed_train": "fc55d0c6c61d8b6ad5e35ec23f75a5a81f504e35abb3f0bb2edfee511a358750",
    "fixed_resume": "fdf3db5dccdde74bab930e1f891318e2c4e6833c09323bc0a77b561569d9d13c",
    "fixed_eval": "67aa165a373b2288bf80ecd6b4d8552f1b8b41bcfa43c0d345bc6771ec78f758",
    "adventure_eval": "2843d85674e79cc50a9c7ccaf8a36b6f7f46d27beb427d96d102c7756710cd34",
    "generalist_fresh": "30ae9133395de867dafbe7f7745e81b794cbd3e47dbd13457f7372c481567968",
    "generalist_resume": "c176b2e252d4c07eeb498c27ea78b9d2cfd8f1800fd595335b79a1bf1a2bb61d",
    "generalist_eval": "17241c31276ae2ca6a563c8fbd2e30f064c7dadb971af8fe5885038b3d18e9e3",
    "level3_train": "4433ac8c6e9e159f68352abebe28f7e63b5091af7fd4b3282c29f03bfb525225",
    "level3_eval": "a24a9d3bd50ee39095f3a711dd77adecfff7b95bc3a8d12bcf154ef84fe12446",
    "coach_stream_fusion": "1e0d75402e2a092010d8e16503032b460766f70cc1bc3361fef1ed9cba749735",
}


@pytest.mark.parametrize("name", tuple(EXPECTED_COMMAND_HASHES))
def test_gui_command_snapshot(name: str) -> None:
    command = _commands()[name]()
    actual = _snapshot_hash(command)
    assert actual == EXPECTED_COMMAND_HASHES[name], (
        f"{name} argv snapshot changed: {actual}\n" + json.dumps(_normalized(command), indent=2)
    )
