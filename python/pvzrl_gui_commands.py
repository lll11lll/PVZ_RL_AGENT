"""Command construction for the Tk dashboard."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, List

from pvzrl_gui_config import (
    StreamerV1LaunchValues,
    build_streamer_v1_argv_additions,
)


DEFAULT_INTERVENTION_LOG_PATH = Path("logs") / "interventions" / "dashboard_interventions.jsonl"
STREAM_COACH_PLATFORMS = ("mock", "twitch", "youtube")


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

    def _add_boolean_option(
        self,
        command: List[str],
        positive_flag: str,
        negative_flag: str,
        enabled: bool,
    ) -> None:
        """Emit an explicit parser-backed boolean so JSON cannot override the form."""

        command.append(positive_flag if enabled else negative_flag)

    def _append_coach_flags(self, command: List[str], *, lab_mode: str = "") -> None:
        selected_lab_mode = str(lab_mode or "Normal").strip().lower()
        human_enabled = bool(self.human_coach_enabled_var.get()) or selected_lab_mode in {"assisted", "fusion"}
        stream_enabled = bool(self.stream_coach_enabled_var.get())
        self._add_boolean_option(
            command, "--human-coach-enabled", "--no-human-coach", human_enabled
        )
        if human_enabled:
            command_mode = getattr(self, "assisted_execution_mode_var", _FallbackValue("override")).get()
            intervention_path = getattr(
                self,
                "intervention_log_path_var",
                _FallbackValue(str(DEFAULT_INTERVENTION_LOG_PATH)),
            ).get()
            command.extend(["--human-coach-command-mode", command_mode or "override"])
            self._add_optional_value(command, "--intervention-log-path", intervention_path)
            self._add_optional_value(command, "--coach-legal-execution-reward", self.human_coach_bonus_var.get())
            self._add_optional_value(command, "--coach-match-reward", self.human_coach_match_bonus_var.get())
            self._add_optional_value(command, "--coach-override-penalty", self.human_coach_override_penalty_var.get())
            self._add_optional_value(command, "--human-coach-log-path", self.human_coach_log_path_var.get())
            self._add_optional_value(command, "--human-coach-command-path", self.human_coach_command_path_var.get())
        self._add_boolean_option(
            command,
            "--human-coach-reward",
            "--no-human-coach-reward",
            bool(self.human_coach_reward_var.get()),
        )

        self._add_boolean_option(
            command, "--stream-coach-enabled", "--no-stream-coach", stream_enabled
        )
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
        self._add_boolean_option(
            command,
            "--stream-coach-reward",
            "--no-stream-coach-reward",
            bool(self.stream_coach_reward_var.get()),
        )

        fusion_enabled = selected_lab_mode == "fusion"
        self._add_boolean_option(
            command,
            "--coach-allow-fusion-planning",
            "--no-coach-allow-fusion-planning",
            bool(self.coach_allow_fusion_planning_var.get() or fusion_enabled),
        )
        self._add_boolean_option(
            command,
            "--fusion-bridge-enabled",
            "--no-fusion-bridge",
            bool(self.fusion_bridge_enabled_var.get() or fusion_enabled),
        )

    def _append_live_status_arg(self, command: List[str]) -> None:
        variable = getattr(self, "live_status_path_var", None)
        raw_path = str(variable.get() if variable is not None else "").strip()
        path = self._resolve_text_path(raw_path) if raw_path else self.live_status_path
        command.extend(["--live-status-path", self._path_for_command(path)])

    def _append_config_arg(self, command: List[str], variable_name: str) -> None:
        variable = getattr(self, variable_name, None)
        raw_path = str(variable.get() if variable is not None else "").strip()
        if raw_path:
            command.extend(["--config", self._path_for_command(self._resolve_text_path(raw_path))])

    def _build_adventure_generalist_command(self) -> List[str]:
        command = self._script_command("train_ppo.py", "--adventure-generalist-train")
        command.append("--no-streamer-v1")
        self._append_config_arg(command, "generalist_config_path_var")
        self._add_optional_value(command, "--total-timesteps", self.generalist_total_timesteps_var.get())
        self._add_optional_value(command, "--checkpoint-freq", self.generalist_checkpoint_freq_var.get())
        self._add_optional_value(
            command,
            "--n-steps",
            getattr(self, "generalist_n_steps_var", _FallbackValue("512")).get(),
        )
        self._add_optional_value(
            command,
            "--batch-size",
            getattr(self, "generalist_batch_size_var", _FallbackValue("64")).get(),
        )
        self._add_optional_value(command, "--initial-loadout", self.generalist_initial_loadout_var.get())
        self._add_optional_value(command, "--seed-list", self.generalist_initial_loadout_var.get())
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
        self._add_boolean_option(
            command, "--quick-wait", "--no-quick-wait", bool(self.generalist_quick_wait_var.get())
        )
        self._add_boolean_option(
            command,
            "--wait-gameplay-ready",
            "--no-wait-gameplay-ready",
            bool(self.generalist_wait_gameplay_ready_var.get()),
        )
        self._add_boolean_option(
            command,
            "--unlock-aware-seed-curriculum",
            "--no-unlock-aware-seed-curriculum",
            bool(self.generalist_unlock_curriculum_var.get()),
        )
        self._add_optional_value(command, "--seed-curriculum", self.generalist_curriculum_var.get())
        self._add_boolean_option(
            command,
            "--randomize-seed-order",
            "--no-randomize-seed-order",
            bool(self.generalist_randomize_seed_order_var.get()),
        )
        self._add_optional_value(command, "--unlock-introduction-delay", self.generalist_unlock_delay_var.get())
        self._add_optional_value(command, "--new-plant-min-inclusion-prob", self.generalist_new_plant_prob_var.get())
        self._add_boolean_option(
            command,
            "--adventure-replay-cleared-levels",
            "--no-adventure-replay-cleared-levels",
            bool(self.generalist_replay_cleared_var.get()),
        )
        self._add_optional_value(command, "--adventure-frontier-sample-prob", self.generalist_frontier_prob_var.get())
        self._add_optional_value(command, "--adventure-recent-cleared-sample-prob", self.generalist_recent_prob_var.get())
        self._add_optional_value(command, "--adventure-maintenance-sample-prob", self.generalist_maintenance_prob_var.get())
        self._add_optional_value(
            command,
            "--adventure-frontier-win-streak-required",
            self.generalist_frontier_win_streak_required_var.get(),
        )
        self._add_boolean_option(
            command, "--tactical-masks", "--no-tactical-masks", bool(self.generalist_tactical_masks_var.get())
        )
        self._add_boolean_option(
            command,
            "--wallnut-tactical-mask",
            "--no-wallnut-tactical-mask",
            bool(self.generalist_wallnut_mask_var.get()),
        )
        self._add_boolean_option(
            command,
            "--cherrybomb-tactical-mask",
            "--no-cherrybomb-tactical-mask",
            bool(self.generalist_cherrybomb_mask_var.get()),
        )
        self._add_boolean_option(
            command,
            "--fusion-action-mask-enabled",
            "--no-fusion-action-mask",
            bool(self.generalist_fusion_action_mask_train_var.get()),
        )
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
        command.extend(("--no-streamer-v1", "--no-human-coach", "--no-stream-coach"))
        self._append_config_arg(command, "eval_config_path_var")
        model_path = self.generalist_eval_model_path_var.get().strip()
        if model_path:
            command.extend(["--model-path", str(self._resolve_text_path(model_path))])
        initial_loadout = getattr(
            self, "eval_initial_loadout_var", self.generalist_initial_loadout_var
        ).get()
        self._add_optional_value(command, "--initial-loadout", initial_loadout)
        self._add_optional_value(command, "--seed-list", initial_loadout)
        self._add_optional_value(
            command,
            "--max-seed-slots",
            getattr(self, "eval_max_seed_slots_var", self.generalist_max_seed_slots_var).get(),
        )
        self._add_optional_value(command, "--adventure-start-level", getattr(self, "eval_start_level_var", self.generalist_start_level_var).get())
        self._add_optional_value(command, "--max-adventure-levels", getattr(self, "eval_max_levels_var", self.generalist_max_levels_var).get())
        self._add_optional_value(command, "--max-attempts-per-level", getattr(self, "eval_max_attempts_var", self.generalist_max_attempts_var).get())
        self._add_optional_value(command, "--game-speed", getattr(self, "eval_game_speed_var", self.generalist_game_speed_var).get())
        self._add_optional_value(command, "--step-seconds", getattr(self, "eval_step_seconds_var", self.generalist_step_seconds_var).get())
        self._add_optional_value(command, "--board-timeout", getattr(self, "eval_board_timeout_var", self.generalist_board_timeout_var).get())
        self._add_optional_value(command, "--adventure-soft-max-steps", getattr(self, "eval_soft_max_steps_var", self.generalist_soft_max_steps_var).get())
        self._add_optional_value(command, "--adventure-hard-max-steps", getattr(self, "eval_hard_max_steps_var", self.generalist_hard_max_steps_var).get())
        final_wave_extension = getattr(
            self,
            "eval_final_wave_extension_var",
            self.generalist_final_wave_extension_var,
        ).get()
        command.append(
            "--adventure-final-wave-extension"
            if final_wave_extension
            else "--no-adventure-final-wave-extension"
        )
        self._add_boolean_option(
            command,
            "--quick-wait",
            "--no-quick-wait",
            bool(getattr(self, "eval_quick_wait_var", self.generalist_quick_wait_var).get()),
        )
        self._add_boolean_option(
            command,
            "--wait-gameplay-ready",
            "--no-wait-gameplay-ready",
            bool(getattr(self, "eval_wait_gameplay_ready_var", self.generalist_wait_gameplay_ready_var).get()),
        )
        self._add_boolean_option(
            command,
            "--tactical-masks",
            "--no-tactical-masks",
            bool(getattr(self, "eval_tactical_masks_var", self.generalist_tactical_masks_var).get()),
        )
        self._add_boolean_option(
            command,
            "--wallnut-tactical-mask",
            "--no-wallnut-tactical-mask",
            bool(getattr(self, "eval_wallnut_mask_var", self.generalist_wallnut_mask_var).get()),
        )
        self._add_boolean_option(
            command,
            "--cherrybomb-tactical-mask",
            "--no-cherrybomb-tactical-mask",
            bool(getattr(self, "eval_cherrybomb_mask_var", self.generalist_cherrybomb_mask_var).get()),
        )
        self._add_boolean_option(
            command,
            "--fusion-action-mask-enabled",
            "--no-fusion-action-mask",
            bool(self.generalist_fusion_action_mask_eval_var.get()),
        )
        run_dir = getattr(self, "eval_run_dir_var", self.generalist_run_dir_var).get().strip()
        if run_dir:
            command.extend(["--run-dir", run_dir])
        self._append_live_status_arg(command)
        return command

    def _build_streamer_v1_command(
        self,
        launch_values: StreamerV1LaunchValues,
    ) -> List[str]:
        """Wrap the existing Streamer V1 backend; no runtime logic lives here."""

        command = self._script_command("train_ppo.py", "--adventure-generalist-train")
        self._append_config_arg(command, "streamer_config_path_var")
        command.extend(build_streamer_v1_argv_additions(launch_values))
        command.extend(("--no-human-coach", "--no-stream-coach"))
        self._add_boolean_option(
            command,
            "--quick-wait",
            "--no-quick-wait",
            bool(getattr(self, "streamer_quick_wait_var", _FallbackValue(True)).get()),
        )
        self._add_boolean_option(
            command,
            "--wait-gameplay-ready",
            "--no-wait-gameplay-ready",
            bool(getattr(self, "streamer_wait_gameplay_ready_var", _FallbackValue(True)).get()),
        )
        return command

    def _command_text(self, command: List[str]) -> str:
        return subprocess.list2cmdline(command)
