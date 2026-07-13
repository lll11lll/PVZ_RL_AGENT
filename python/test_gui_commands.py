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
    "generalist_fresh": "e78855ee8babcee83b869f4e86b9997924a5460b0ed8ecc74e64108a15448651",
    "generalist_resume": "20ad5007105a721401acff4db55d7989aec2247bbc90db1412ecaf9db9ecf2de",
    "generalist_eval": "3b026cb774ee1d297994e82ad73b7dd901d322d3f579851e3c066447c7e6afb4",
    "coach_stream_fusion": "bf8496e1408cfdfc68c162b2a108814ab6020f99f8046d8164be45f2d7039d7f",
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


def test_model_discovery_rejects_non_generalist_metadata(tmp_path: Path) -> None:
    dashboard = _dashboard()
    dashboard.repo_root = tmp_path

    fixed_dir = tmp_path / "runs" / "obsolete_fixed" / "checkpoints"
    fixed_dir.mkdir(parents=True)
    (fixed_dir / "ppo_pvz_999_steps.zip").write_bytes(b"fixed")
    (fixed_dir.parent / "model_metadata.json").write_text(
        json.dumps(
            {
                "metadata_version": 1,
                "model_family": "ppo_fixed_specialist",
                "action_count": 201,
                "action_space_mode": "fixed",
                "action_decoder_version": "fixed_slot_4x50_plus_wait_v1",
                "observation_version": "fixed_slot_v1",
                "max_seed_slots": 4,
            }
        ),
        encoding="utf-8",
    )

    generalist_dir = tmp_path / "runs" / "generalist" / "checkpoints"
    generalist_dir.mkdir(parents=True)
    expected = generalist_dir / "ppo_pvz_370000_steps.zip"
    expected.write_bytes(b"generalist")
    (generalist_dir.parent / "model_metadata.json").write_text(
        json.dumps(
            {
                "metadata_version": 1,
                "model_family": "ppo_adventure_generalist_14slot_identity_v1",
                "action_count": 701,
                "action_space_mode": "adventure_14slot_identity",
                "action_decoder_version": "seedslot14x50_plus_wait_v1",
                "observation_version": "adventure_14slot_identity_v1",
                "max_seed_slots": 14,
            }
        ),
        encoding="utf-8",
    )

    assert dashboard._find_newest_usable_model_zip() == expected
