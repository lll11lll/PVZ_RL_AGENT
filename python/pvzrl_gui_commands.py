"""Command construction for the Tk dashboard."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, List, Optional

from pvzrl_action_space import ACTION_SPACE_ADVENTURE_14_IDENTITY


DEFAULT_INTERVENTION_LOG_PATH = Path("logs") / "interventions" / "dashboard_interventions.jsonl"
STREAM_COACH_PLATFORMS = ("mock", "twitch", "youtube")
ADVENTURE_GENERALIST_ACTION_SPACE_MODE = ACTION_SPACE_ADVENTURE_14_IDENTITY


class _FallbackValue:
    def __init__(self, value: Any) -> None:
        self._value = value

    def get(self) -> Any:
        return self._value


class GuiCommandMixin:
    def _script_command(self, script_name: str, *args: str) -> List[str]:
        return [sys.executable, str(Path("python") / script_name), *args]

    def _add_optional_value(self, command: List[str], flag: str, value: str) -> None:
        cleaned = value.strip()
        if cleaned:
            command.extend([flag, cleaned])

    def _add_enabled_flag(self, command: List[str], flag: str, enabled: bool) -> None:
        if enabled:
            command.append(flag)

    def _safe_run_name(self, value: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value.strip())
        return "_".join(part for part in cleaned.split("_") if part) or "run"

    def _append_train_run_target(self, command: List[str]) -> None:
        run_dir = self.run_dir_var.get().strip()
        run_name = self.run_name_var.get().strip()
        if run_dir:
            command.extend(["--run-dir", run_dir])
        elif run_name:
            command.extend(["--run-dir", str(Path("runs") / self._safe_run_name(run_name))])

    def _append_eval_run_dir(self, command: List[str]) -> None:
        run_dir = self.run_dir_var.get().strip()
        if run_dir:
            command.extend(["--run-dir", run_dir])

    def _append_common_train_eval_flags(self, command: List[str]) -> None:
        self._add_enabled_flag(command, "--quick-wait", self.quick_wait_var.get())
        self._add_enabled_flag(command, "--wait-gameplay-ready", self.wait_gameplay_ready_var.get())
        self._add_enabled_flag(command, "--auto-select-seeds", self.auto_select_seeds_var.get())
        self._add_enabled_flag(command, "--debug-perf", self.debug_perf_var.get())
        command.extend(["--fusion-policy", self.fusion_policy_var.get().strip() or "none"])
        lab_var_name = "eval_lab_mode_var" if any("eval" in part for part in command) else "train_lab_mode_var"
        lab_mode = getattr(self, lab_var_name, _FallbackValue("Normal")).get()
        self._append_coach_flags(command, lab_mode=lab_mode)

    def _append_coach_flags(self, command: List[str], *, lab_mode: str = "") -> None:
        selected_lab_mode = str(lab_mode or "Normal").strip().lower()
        human_enabled = bool(self.human_coach_enabled_var.get()) or selected_lab_mode in {"assisted", "fusion"}
        stream_enabled = bool(self.stream_coach_enabled_var.get())
        self._add_enabled_flag(command, "--human-coach-enabled", human_enabled)
        if human_enabled:
            command_mode = getattr(self, "assisted_execution_mode_var", _FallbackValue("override")).get()
            intervention_path = getattr(
                self,
                "intervention_log_path_var",
                _FallbackValue(str(DEFAULT_INTERVENTION_LOG_PATH)),
            ).get()
            command.extend(["--human-coach-command-mode", command_mode or "override"])
            self._add_optional_value(command, "--intervention-log-path", intervention_path)
            self._add_enabled_flag(command, "--human-coach-reward", self.human_coach_reward_var.get())
            self._add_optional_value(command, "--coach-legal-execution-reward", self.human_coach_bonus_var.get())
            self._add_optional_value(command, "--coach-match-reward", self.human_coach_match_bonus_var.get())
            self._add_optional_value(command, "--coach-override-penalty", self.human_coach_override_penalty_var.get())
            self._add_optional_value(command, "--coach-fusion-success-reward", self.human_coach_fusion_reward_var.get())
            self._add_optional_value(command, "--coach-tactical-usefulness-reward", self.human_coach_tactical_reward_var.get())
            self._add_optional_value(command, "--human-coach-log-path", self.human_coach_log_path_var.get())
            self._add_optional_value(command, "--human-coach-command-path", self.human_coach_command_path_var.get())

        self._add_enabled_flag(command, "--stream-coach-enabled", stream_enabled)
        if stream_enabled:
            command.extend(["--stream-coach-mode", self.stream_coach_platform_var.get().strip() or STREAM_COACH_PLATFORMS[0]])
            command.extend(["--stream-coach-platform", self.stream_coach_platform_var.get().strip() or STREAM_COACH_PLATFORMS[0]])
            self._add_optional_value(command, "--stream-coach-window-sec", self.stream_coach_window_sec_var.get())
            self._add_optional_value(command, "--stream-coach-min-votes", self.stream_coach_min_votes_var.get())
            self._add_optional_value(
                command,
                "--stream-coach-max-actions-per-minute",
                self.stream_coach_max_actions_per_minute_var.get(),
            )
            self._add_enabled_flag(command, "--stream-coach-reward", self.stream_coach_reward_var.get())
            dry_run_var = getattr(self, "stream_coach_dry_run_var", None)
            dry_run_enabled = True if dry_run_var is None else bool(dry_run_var.get())
            if dry_run_enabled:
                command.append("--stream-coach-dry-run")
            else:
                command.append("--stream-coach-apply")
            self._add_optional_value(command, "--stream-coach-command-path", self.human_coach_command_path_var.get())
            self._add_optional_value(command, "--stream-coach-log-path", self.stream_coach_log_path_var.get())
            mock_script_var = getattr(self, "stream_coach_mock_script_var", None)
            self._add_optional_value(
                command,
                "--stream-coach-mock-script",
                "" if mock_script_var is None else mock_script_var.get(),
            )

        fusion_enabled = selected_lab_mode == "fusion"
        self._add_enabled_flag(command, "--coach-allow-fusion-planning", self.coach_allow_fusion_planning_var.get() or fusion_enabled)
        self._add_enabled_flag(command, "--fusion-bridge-enabled", self.fusion_bridge_enabled_var.get() or fusion_enabled)

    def _append_live_status_arg(self, command: List[str]) -> None:
        command.extend(["--live-status-path", self._path_for_command(self.live_status_path)])

    def _build_train_command(self, resume: bool = False) -> List[str]:
        command = self._script_command("train_ppo.py", "--train")
        self._add_optional_value(command, "--seed-list", self.seed_list_var.get())
        self._add_optional_value(command, "--total-timesteps", self.total_timesteps_var.get())
        self._add_optional_value(command, "--max-steps", self.max_steps_var.get())
        self._add_optional_value(command, "--step-seconds", self.step_seconds_var.get())
        self._add_optional_value(command, "--game-speed", self.game_speed_var.get())
        self._add_optional_value(command, "--start-sun", self.start_sun_var.get())
        self._add_optional_value(command, "--board-timeout", self.board_timeout_var.get())
        self._add_optional_value(command, "--gameplay-ready-timeout", self.gameplay_ready_timeout_var.get())
        self._add_optional_value(command, "--checkpoint-freq", self.checkpoint_freq_var.get())
        self._append_common_train_eval_flags(command)
        if resume:
            model_path = self.model_path_var.get().strip()
            if model_path:
                command.extend(["--model-path", str(self._resolve_text_path(model_path))])
        self._append_train_run_target(command)
        self._append_live_status_arg(command)
        return command

    def _build_eval_command(self, episodes_override: Optional[str] = None) -> List[str]:
        command = self._script_command("train_ppo.py", "--eval")
        model_path = self.model_path_var.get().strip()
        if model_path:
            command.extend(["--model-path", str(self._resolve_text_path(model_path))])
        self._add_optional_value(command, "--episodes", episodes_override or self.episodes_var.get())
        self._add_optional_value(command, "--max-steps", self.max_steps_var.get())
        self._add_optional_value(command, "--step-seconds", self.step_seconds_var.get())
        self._add_optional_value(command, "--game-speed", self.game_speed_var.get())
        self._append_common_train_eval_flags(command)
        self._add_optional_value(command, "--seed-list", self.seed_list_var.get())
        self._append_eval_run_dir(command)
        self._append_live_status_arg(command)
        return command

    def _build_adventure_command(self) -> List[str]:
        command = self._script_command("train_ppo.py")
        self._add_enabled_flag(command, "--adventure-eval", self.adventure_eval_var.get())
        model_path = self.adventure_model_path_var.get().strip()
        if model_path:
            command.extend(["--model-path", str(self._resolve_text_path(model_path))])
        self._add_optional_value(command, "--episodes", self.adventure_episodes_var.get())
        self._add_optional_value(command, "--game-speed", self.adventure_game_speed_var.get())
        self._add_optional_value(command, "--step-seconds", self.adventure_step_seconds_var.get())
        self._add_optional_value(command, "--adventure-soft-max-steps", self.adventure_soft_max_steps_var.get())
        self._add_optional_value(command, "--adventure-hard-max-steps", self.adventure_hard_max_steps_var.get())
        command.append(
            "--adventure-final-wave-extension"
            if self.adventure_final_wave_extension_var.get()
            else "--no-adventure-final-wave-extension"
        )
        self._add_optional_value(command, "--board-timeout", self.adventure_board_timeout_var.get())
        self._add_enabled_flag(command, "--quick-wait", self.adventure_quick_wait_var.get())
        self._add_enabled_flag(command, "--wait-gameplay-ready", self.adventure_wait_gameplay_ready_var.get())
        self._add_enabled_flag(command, "--auto-select-seeds", self.adventure_auto_select_seeds_var.get())
        if self.adventure_advance_on_wins_var.get():
            self._add_optional_value(command, "--advance-on-wins", self.adventure_advance_wins_var.get())
        self._add_optional_value(command, "--max-adventure-levels", self.adventure_max_levels_var.get())
        self._add_optional_value(command, "--max-attempts-per-level", self.adventure_max_attempts_var.get())
        self._add_optional_value(command, "--seed-list", self.adventure_seed_list_var.get())
        self._add_optional_value(command, "--plant-types", self.adventure_plant_types_var.get())
        self._add_enabled_flag(command, "--tactical-masks", self.adventure_tactical_masks_var.get())
        self._add_enabled_flag(command, "--wallnut-tactical-mask", self.adventure_wallnut_mask_var.get())
        self._add_enabled_flag(command, "--cherrybomb-tactical-mask", self.adventure_cherrybomb_mask_var.get())
        command.extend(["--fusion-policy", self.adventure_fusion_policy_var.get().strip() or "none"])
        self._append_coach_flags(
            command,
            lab_mode=getattr(self, "eval_lab_mode_var", _FallbackValue("Normal")).get(),
        )
        self._append_live_status_arg(command)
        return command

    def _build_adventure_generalist_command(self) -> List[str]:
        command = self._script_command("train_ppo.py", "--adventure-generalist-train")
        self._add_optional_value(command, "--total-timesteps", self.generalist_total_timesteps_var.get())
        self._add_optional_value(command, "--checkpoint-freq", self.generalist_checkpoint_freq_var.get())
        self._add_optional_value(command, "--initial-loadout", self.generalist_initial_loadout_var.get())
        self._add_optional_value(command, "--seed-list", self.generalist_initial_loadout_var.get())
        command.extend(["--action-space-mode", ADVENTURE_GENERALIST_ACTION_SPACE_MODE])
        self._add_optional_value(command, "--max-seed-slots", self.generalist_max_seed_slots_var.get())
        self._add_optional_value(command, "--adventure-start-level", self.generalist_start_level_var.get())
        self._add_optional_value(command, "--max-adventure-levels", self.generalist_max_levels_var.get())
        self._add_optional_value(command, "--max-attempts-per-level", self.generalist_max_attempts_var.get())
        self._add_optional_value(command, "--game-speed", self.generalist_game_speed_var.get())
        self._add_optional_value(command, "--step-seconds", self.generalist_step_seconds_var.get())
        self._add_optional_value(command, "--board-timeout", self.generalist_board_timeout_var.get())
        self._add_optional_value(command, "--adventure-soft-max-steps", self.generalist_soft_max_steps_var.get())
        self._add_optional_value(command, "--adventure-hard-max-steps", self.generalist_hard_max_steps_var.get())
        command.append(
            "--adventure-final-wave-extension"
            if self.generalist_final_wave_extension_var.get()
            else "--no-adventure-final-wave-extension"
        )
        self._add_enabled_flag(command, "--quick-wait", self.generalist_quick_wait_var.get())
        self._add_enabled_flag(command, "--wait-gameplay-ready", self.generalist_wait_gameplay_ready_var.get())
        self._add_enabled_flag(command, "--unlock-aware-seed-curriculum", self.generalist_unlock_curriculum_var.get())
        self._add_optional_value(command, "--seed-curriculum", self.generalist_curriculum_var.get())
        self._add_enabled_flag(command, "--randomize-seed-order", self.generalist_randomize_seed_order_var.get())
        self._add_optional_value(command, "--unlock-introduction-delay", self.generalist_unlock_delay_var.get())
        self._add_optional_value(command, "--new-plant-min-inclusion-prob", self.generalist_new_plant_prob_var.get())
        self._add_enabled_flag(command, "--adventure-replay-cleared-levels", self.generalist_replay_cleared_var.get())
        self._add_optional_value(command, "--adventure-frontier-sample-prob", self.generalist_frontier_prob_var.get())
        self._add_optional_value(command, "--adventure-recent-cleared-sample-prob", self.generalist_recent_prob_var.get())
        self._add_optional_value(command, "--adventure-maintenance-sample-prob", self.generalist_maintenance_prob_var.get())
        self._add_optional_value(
            command,
            "--adventure-frontier-win-streak-required",
            self.generalist_frontier_win_streak_required_var.get(),
        )
        self._add_enabled_flag(command, "--tactical-masks", self.generalist_tactical_masks_var.get())
        self._add_enabled_flag(command, "--wallnut-tactical-mask", self.generalist_wallnut_mask_var.get())
        self._add_enabled_flag(command, "--cherrybomb-tactical-mask", self.generalist_cherrybomb_mask_var.get())
        self._add_enabled_flag(command, "--fusion-action-mask-enabled", self.generalist_fusion_action_mask_train_var.get())
        resume_model_path = self.generalist_resume_model_path_var.get().strip()
        if resume_model_path:
            command.extend(["--resume-model-path", str(self._resolve_text_path(resume_model_path))])
        run_dir = self.generalist_run_dir_var.get().strip()
        if run_dir:
            command.extend(["--run-dir", run_dir])
        self._append_coach_flags(
            command,
            lab_mode=getattr(self, "train_lab_mode_var", _FallbackValue("Normal")).get(),
        )
        self._append_live_status_arg(command)
        return command

    def _build_adventure_generalist_eval_command(self) -> List[str]:
        command = self._script_command("train_ppo.py", "--adventure-generalist-eval")
        model_path = self.generalist_eval_model_path_var.get().strip()
        if model_path:
            command.extend(["--model-path", str(self._resolve_text_path(model_path))])
        self._add_optional_value(command, "--episodes", self.generalist_eval_episodes_var.get())
        self._add_optional_value(command, "--initial-loadout", self.generalist_initial_loadout_var.get())
        self._add_optional_value(command, "--seed-list", self.generalist_initial_loadout_var.get())
        command.extend(["--action-space-mode", ADVENTURE_GENERALIST_ACTION_SPACE_MODE])
        self._add_optional_value(command, "--max-seed-slots", self.generalist_max_seed_slots_var.get())
        self._add_optional_value(command, "--adventure-start-level", self.generalist_start_level_var.get())
        self._add_optional_value(command, "--max-adventure-levels", self.generalist_max_levels_var.get())
        self._add_optional_value(command, "--max-attempts-per-level", self.generalist_max_attempts_var.get())
        self._add_optional_value(command, "--game-speed", self.generalist_game_speed_var.get())
        self._add_optional_value(command, "--step-seconds", self.generalist_step_seconds_var.get())
        self._add_optional_value(command, "--board-timeout", self.generalist_board_timeout_var.get())
        self._add_optional_value(command, "--adventure-soft-max-steps", self.generalist_soft_max_steps_var.get())
        self._add_optional_value(command, "--adventure-hard-max-steps", self.generalist_hard_max_steps_var.get())
        command.append(
            "--adventure-final-wave-extension"
            if self.generalist_final_wave_extension_var.get()
            else "--no-adventure-final-wave-extension"
        )
        self._add_enabled_flag(command, "--quick-wait", self.generalist_quick_wait_var.get())
        self._add_enabled_flag(command, "--wait-gameplay-ready", self.generalist_wait_gameplay_ready_var.get())
        self._add_enabled_flag(command, "--tactical-masks", self.generalist_tactical_masks_var.get())
        self._add_enabled_flag(command, "--wallnut-tactical-mask", self.generalist_wallnut_mask_var.get())
        self._add_enabled_flag(command, "--cherrybomb-tactical-mask", self.generalist_cherrybomb_mask_var.get())
        self._add_enabled_flag(command, "--fusion-action-mask-enabled", self.generalist_fusion_action_mask_eval_var.get())
        run_dir = self.generalist_run_dir_var.get().strip()
        if run_dir:
            command.extend(["--run-dir", run_dir])
        self._append_coach_flags(
            command,
            lab_mode=getattr(self, "eval_lab_mode_var", _FallbackValue("Normal")).get(),
        )
        self._append_live_status_arg(command)
        return command

    def _build_level3_command(self) -> List[str]:
        mode = self.level3_mode_var.get().strip().lower()
        command = self._script_command("train_ppo.py", "--level3-eval" if mode == "eval" else "--level3-train")
        self._add_optional_value(command, "--target-level", self.level3_target_level_var.get())
        if mode == "eval":
            model_path = self.level3_model_path_var.get().strip()
            if model_path:
                command.extend(["--model-path", str(self._resolve_text_path(model_path))])
            self._add_optional_value(command, "--episodes", self.level3_episodes_var.get())
        else:
            self._add_optional_value(command, "--total-timesteps", self.level3_total_timesteps_var.get())
            model_path = self.level3_model_path_var.get().strip()
            if model_path:
                command.extend(["--model-path", str(self._resolve_text_path(model_path))])
        self._add_optional_value(command, "--seed-list", self.level3_seed_list_var.get())
        self._add_optional_value(command, "--plant-types", self.level3_plant_types_var.get())
        self._add_optional_value(command, "--game-speed", self.level3_game_speed_var.get())
        self._add_optional_value(command, "--step-seconds", self.level3_step_seconds_var.get())
        self._add_optional_value(command, "--max-steps", self.level3_max_steps_var.get())
        self._add_optional_value(command, "--board-timeout", self.level3_board_timeout_var.get())
        command.extend(["--quick-wait", "--wait-gameplay-ready", "--auto-select-seeds"])
        self._add_enabled_flag(command, "--tactical-masks", self.level3_tactical_masks_var.get())
        self._add_enabled_flag(command, "--wallnut-tactical-mask", self.level3_wallnut_mask_var.get())
        self._add_enabled_flag(command, "--cherrybomb-tactical-mask", self.level3_cherrybomb_mask_var.get())
        command.extend(["--fusion-policy", "none"])
        self._append_live_status_arg(command)
        return command

    def _command_text(self, command: List[str]) -> str:
        return subprocess.list2cmdline(command)
