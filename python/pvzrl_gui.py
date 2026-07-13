"""Compact Tkinter dashboard for PvZRL commands, runs, and live diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk

from pvzrl_assisted_coach import (
    LAWN_COLS,
    LAWN_ROWS,
    SEED_PACKET_SLOTS,
    AssistedCoachCommand,
    AssistedCommandQueue,
    AssistedCommandStatus,
    AssistedCommandType,
    AssistedCommandValidator,
    AssistedExecutionMode,
    InterventionJSONLLogger,
    queue_rows,
)
from pvzrl_gui_commands import (
    ADVENTURE_GENERALIST_ACTION_SPACE_MODE,
    DEFAULT_INTERVENTION_LOG_PATH,
    STREAM_COACH_PLATFORMS,
    GuiCommandMixin,
)
from pvzrl_gui_coach import CoachCommandSink, CoachQueueCommand
from pvzrl_gui_process import (
    LOG_BACKLOG_POLL_MS,
    LOG_DRAIN_BUDGET_SECONDS,
    LOG_DRAIN_MAX_ITEMS,
    LOG_POLL_MS,
    LOG_QUEUE_MAX_ITEMS,
    STOP_GRACE_SECONDS,
    STOP_KILL_WAIT_SECONDS,
    ProcessLogMixin,
)
from pvzrl_gui_status import (
    LIVE_MAX_AGE_SECONDS,
    STALE_MAX_AGE_SECONDS,
    DiagnosticsRenderKey,
    LiveStatusReader,
)
from pvzrl_gui_view import (
    POLL_MS,
    GuiStatusViewMixin,
)
from pvzrl_registry import get_plant_registry


LOG_HISTORY_MAX_LINES = 5000
LOG_HISTORY_MAX_CHARS = 1_000_000
LOG_VIEW_REFRESH_MIN_SECONDS = 0.1
CLOSE_POLL_MS = 50
PROCESS_THREAD_JOIN_SECONDS = 0.2
CLOSE_LOG_DRAIN_SECONDS = 0.3
ROLLING_WIN_WINDOW = 20
MODEL_ZIP_NAMES = {"model.zip", "final_model.zip"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIVE_STATUS_PATH = Path("runs") / "live_status.json"
DEFAULT_COACH_COMMAND_QUEUE_PATH = Path("runs") / "coach_commands.jsonl"
DEFAULT_HUMAN_COACH_LOG_PATH = Path("runs") / "human_coach.jsonl"
DEFAULT_STREAM_COACH_LOG_PATH = Path("runs") / "stream_coach.jsonl"
_PLANT_REGISTRY = get_plant_registry()
_FOUR_SLOT_CURRENT = _PLANT_REGISTRY.require_gui_preset("four_slot_current")
_FOUR_SLOT_DUPLICATE = _PLANT_REGISTRY.require_gui_preset("four_slot_duplicate")
ADVENTURE_DEFAULT_PLANT_TYPES = _FOUR_SLOT_CURRENT.plant_type_csv
STRUCTURED_COACH_COMMANDS = tuple(command.value for command in AssistedCommandType)
ASSISTED_EXECUTION_MODES = tuple(mode.value for mode in AssistedExecutionMode)
LAB_MODES = ("Normal", "Assisted", "Fusion", "Curriculum")
LEVEL3_SEED_LIST = _FOUR_SLOT_CURRENT.seed_csv
LEVEL3_PLANT_TYPES = _FOUR_SLOT_CURRENT.plant_type_csv
ADVENTURE_GENERALIST_MODEL_FAMILY = "ppo_adventure_generalist_14slot_identity_v1"
ADVENTURE_GENERALIST_INITIAL_LOADOUT = _FOUR_SLOT_DUPLICATE.seed_csv


class _Tooltip:
    """Small dependency-free hover tooltip for compact dashboard controls."""

    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self.popup: Optional[tk.Toplevel] = None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _show(self, _event: Any = None) -> None:
        if self.popup is not None or not self.text:
            return
        popup = tk.Toplevel(self.widget)
        popup.wm_overrideredirect(True)
        popup.wm_geometry(f"+{self.widget.winfo_rootx() + 18}+{self.widget.winfo_rooty() + 22}")
        label = ttk.Label(popup, text=self.text, justify="left", padding=(7, 4), relief="solid")
        label.pack()
        self.popup = popup

    def _hide(self, _event: Any = None) -> None:
        if self.popup is not None:
            self.popup.destroy()
            self.popup = None


class PvZDashboard(GuiCommandMixin, ProcessLogMixin, GuiStatusViewMixin):
    def __init__(self, root: tk.Tk, live_status_path: Path):
        self.root = root
        self.project_root = PROJECT_ROOT
        self.repo_root = self.project_root
        self.live_status_path = self._resolve_path(live_status_path)
        self.root.title("PvZRL Dashboard")
        self.root.geometry("1180x780")
        self.root.minsize(980, 650)

        default_model = self._find_newest_model_zip()
        default_adventure_model = self._find_newest_usable_model_zip() or default_model
        self.total_timesteps_var = tk.StringVar(value="250000")
        self.max_steps_var = tk.StringVar(value="1000")
        self.step_seconds_var = tk.StringVar(value="0.05")
        self.game_speed_var = tk.StringVar(value="4.0")
        self.seed_list_var = tk.StringVar(value=_FOUR_SLOT_CURRENT.seed_csv)
        self.model_path_var = tk.StringVar(value=str(default_model) if default_model else "")
        self.episodes_var = tk.StringVar(value="10")
        self.run_dir_var = tk.StringVar(value="")
        self.run_name_var = tk.StringVar(value="")
        self.start_sun_var = tk.StringVar(value="500")
        self.board_timeout_var = tk.StringVar(value="60")
        self.gameplay_ready_timeout_var = tk.StringVar(value="30")
        self.checkpoint_freq_var = tk.StringVar(value="5000")
        self.fusion_policy_var = tk.StringVar(value="none")
        self.quick_wait_var = tk.BooleanVar(value=True)
        self.wait_gameplay_ready_var = tk.BooleanVar(value=True)
        self.auto_select_seeds_var = tk.BooleanVar(value=True)
        self.debug_perf_var = tk.BooleanVar(value=False)
        self.fast_only_var = tk.BooleanVar(value=False)
        self.adventure_model_path_var = tk.StringVar(value=str(default_adventure_model) if default_adventure_model else "")
        self.adventure_seed_list_var = tk.StringVar(value=_FOUR_SLOT_CURRENT.seed_csv)
        self.adventure_plant_types_var = tk.StringVar(value=ADVENTURE_DEFAULT_PLANT_TYPES)
        self.adventure_episodes_var = tk.StringVar(value="5")
        self.adventure_max_levels_var = tk.StringVar(value="5")
        self.adventure_max_attempts_var = tk.StringVar(value="10")
        self.adventure_advance_wins_var = tk.StringVar(value="1")
        self.adventure_game_speed_var = tk.StringVar(value="4.0")
        self.adventure_step_seconds_var = tk.StringVar(value="0.05")
        self.adventure_soft_max_steps_var = tk.StringVar(value="2000")
        self.adventure_hard_max_steps_var = tk.StringVar(value="3500")
        self.adventure_board_timeout_var = tk.StringVar(value="60")
        self.adventure_live_status_var = tk.StringVar(value=self._path_for_command(self.live_status_path))
        self.adventure_eval_var = tk.BooleanVar(value=True)
        self.adventure_advance_on_wins_var = tk.BooleanVar(value=True)
        self.adventure_auto_select_seeds_var = tk.BooleanVar(value=True)
        self.adventure_wait_gameplay_ready_var = tk.BooleanVar(value=True)
        self.adventure_quick_wait_var = tk.BooleanVar(value=True)
        self.adventure_final_wave_extension_var = tk.BooleanVar(value=True)
        self.adventure_tactical_masks_var = tk.BooleanVar(value=True)
        self.adventure_wallnut_mask_var = tk.BooleanVar(value=True)
        self.adventure_cherrybomb_mask_var = tk.BooleanVar(value=True)
        self.adventure_fusion_policy_var = tk.StringVar(value="none")
        self.generalist_total_timesteps_var = tk.StringVar(value="250000")
        self.generalist_checkpoint_freq_var = tk.StringVar(value="5000")
        self.generalist_initial_loadout_var = tk.StringVar(value=ADVENTURE_GENERALIST_INITIAL_LOADOUT)
        self.generalist_max_seed_slots_var = tk.StringVar(value="14")
        self.generalist_start_level_var = tk.StringVar(value="1")
        self.generalist_max_levels_var = tk.StringVar(value="10")
        self.generalist_max_attempts_var = tk.StringVar(value="10")
        self.generalist_game_speed_var = tk.StringVar(value="4.0")
        self.generalist_step_seconds_var = tk.StringVar(value="0.05")
        self.generalist_board_timeout_var = tk.StringVar(value="60")
        self.generalist_soft_max_steps_var = tk.StringVar(value="2000")
        self.generalist_hard_max_steps_var = tk.StringVar(value="3500")
        self.generalist_frontier_prob_var = tk.StringVar(value="0.60")
        self.generalist_recent_prob_var = tk.StringVar(value="0.30")
        self.generalist_maintenance_prob_var = tk.StringVar(value="0.10")
        self.generalist_frontier_win_streak_required_var = tk.StringVar(value="1")
        self.generalist_unlock_delay_var = tk.StringVar(value="0")
        self.generalist_new_plant_prob_var = tk.StringVar(value="0.15")
        self.generalist_run_dir_var = tk.StringVar(value="")
        self.generalist_resume_model_path_var = tk.StringVar(value="")
        self.generalist_eval_model_path_var = tk.StringVar(value=str(default_adventure_model) if default_adventure_model else "")
        self.generalist_eval_episodes_var = tk.StringVar(value="5")
        self.generalist_unlock_curriculum_var = tk.BooleanVar(value=True)
        self.generalist_replay_cleared_var = tk.BooleanVar(value=True)
        self.generalist_final_wave_extension_var = tk.BooleanVar(value=True)
        self.generalist_wait_gameplay_ready_var = tk.BooleanVar(value=True)
        self.generalist_quick_wait_var = tk.BooleanVar(value=True)
        self.generalist_tactical_masks_var = tk.BooleanVar(value=True)
        self.generalist_wallnut_mask_var = tk.BooleanVar(value=True)
        self.generalist_cherrybomb_mask_var = tk.BooleanVar(value=True)
        # Model self-fusion action mask: default ON for training so the agent
        # learns to fuse; optional (default OFF) for eval. Unchecking either
        # restores legacy behavior (occupied tiles illegal, no model fusion).
        self.generalist_fusion_action_mask_train_var = tk.BooleanVar(value=True)
        self.generalist_fusion_action_mask_eval_var = tk.BooleanVar(value=False)
        self.generalist_curriculum_var = tk.StringVar(value="conservative")
        self.generalist_randomize_seed_order_var = tk.BooleanVar(value=False)
        self.level3_mode_var = tk.StringVar(value="train")
        self.level3_target_level_var = tk.StringVar(value="3")
        self.level3_model_path_var = tk.StringVar(value=str(default_model) if default_model else "")
        self.level3_total_timesteps_var = tk.StringVar(value="750000")
        self.level3_episodes_var = tk.StringVar(value="25")
        self.level3_max_steps_var = tk.StringVar(value="1200")
        self.level3_step_seconds_var = tk.StringVar(value="0.05")
        self.level3_game_speed_var = tk.StringVar(value="4.0")
        self.level3_board_timeout_var = tk.StringVar(value="90")
        self.level3_seed_list_var = tk.StringVar(value=LEVEL3_SEED_LIST)
        self.level3_plant_types_var = tk.StringVar(value=LEVEL3_PLANT_TYPES)
        self.level3_tactical_masks_var = tk.BooleanVar(value=True)
        self.level3_wallnut_mask_var = tk.BooleanVar(value=True)
        self.level3_cherrybomb_mask_var = tk.BooleanVar(value=True)
        self.human_coach_enabled_var = tk.BooleanVar(value=False)
        self.human_coach_reward_var = tk.BooleanVar(value=False)
        self.human_coach_bonus_var = tk.StringVar(value="")
        self.human_coach_match_bonus_var = tk.StringVar(value="")
        self.human_coach_override_penalty_var = tk.StringVar(value="")
        # Reward sent to the model when a coach command succeeds. Blank = use the
        # train_ppo defaults (--coach-fusion-success-reward / -tactical-usefulness).
        self.human_coach_fusion_reward_var = tk.StringVar(value="")
        self.human_coach_tactical_reward_var = tk.StringVar(value="")
        self.human_coach_log_path_var = tk.StringVar(value=str(DEFAULT_HUMAN_COACH_LOG_PATH))
        self.human_coach_command_path_var = tk.StringVar(value=str(DEFAULT_COACH_COMMAND_QUEUE_PATH))
        self.human_coach_command_input_var = tk.StringVar(value="")
        self.structured_command_type_var = tk.StringVar(value=AssistedCommandType.PLANT.value)
        self.structured_row_var = tk.StringVar(value="2")
        self.structured_col_var = tk.StringVar(value="4")
        self.structured_seed_slot_var = tk.StringVar(value="0")
        self.structured_custom_text_var = tk.StringVar(value="")
        self.structured_preview_var = tk.StringVar(value="PLANT 2 4 0")
        self.assisted_command_source_var = tk.StringVar(value="dashboard")
        self.assisted_command_user_var = tk.StringVar(value="local")
        self.assisted_execution_mode_var = tk.StringVar(value=AssistedExecutionMode.OVERRIDE.value)
        self.train_lab_mode_var = tk.StringVar(value="Normal")
        self.eval_lab_mode_var = tk.StringVar(value="Normal")
        self.intervention_log_path_var = tk.StringVar(value=str(DEFAULT_INTERVENTION_LOG_PATH))
        self.assisted_queue_summary_var = tk.StringVar(value="pending=0 approved=0 rejected=0 executed=0")
        self.command_enablement_var = tk.StringVar(value="Coach commands off | Viewer commands off | mode=override")
        self.fusion_command_feedback_var = tk.StringVar(value="Fusion legality is checked against the live bridge at execution.")
        self.stream_coach_enabled_var = tk.BooleanVar(value=False)
        self.stream_coach_platform_var = tk.StringVar(value=STREAM_COACH_PLATFORMS[0])
        self.stream_coach_window_sec_var = tk.StringVar(value="3")
        self.stream_coach_min_votes_var = tk.StringVar(value="2")
        self.stream_coach_max_actions_per_minute_var = tk.StringVar(value="20")
        self.stream_coach_reward_var = tk.BooleanVar(value=False)
        self.stream_coach_dry_run_var = tk.BooleanVar(value=True)
        self.stream_coach_log_path_var = tk.StringVar(value=str(DEFAULT_STREAM_COACH_LOG_PATH))
        self.stream_coach_mock_script_var = tk.StringVar(value="")
        self.coach_allow_fusion_planning_var = tk.BooleanVar(value=False)
        self.fusion_bridge_enabled_var = tk.BooleanVar(value=False)
        self.human_coach_enabled_status_var = tk.StringVar(value="n/a")
        self.human_coach_last_command_var = tk.StringVar(value="n/a")
        self.human_coach_last_action_var = tk.StringVar(value="n/a")
        self.human_coach_last_error_var = tk.StringVar(value="n/a")
        self.human_coach_override_count_var = tk.StringVar(value="n/a")
        self.human_coach_match_count_var = tk.StringVar(value="n/a")
        self.human_coach_reward_total_var = tk.StringVar(value="n/a")
        self.stream_coach_enabled_status_var = tk.StringVar(value="n/a")
        self.stream_coach_platform_status_var = tk.StringVar(value="n/a")
        self.stream_coach_dry_run_status_var = tk.StringVar(value="n/a")
        self.stream_coach_alive_status_var = tk.StringVar(value="n/a")
        self.stream_coach_last_message_var = tk.StringVar(value="n/a")
        self.stream_coach_last_parsed_command_var = tk.StringVar(value="n/a")
        self.stream_coach_last_applied_command_var = tk.StringVar(value="n/a")
        self.stream_coach_accept_reject_var = tk.StringVar(value="n/a")
        self.stream_coach_last_reject_reason_var = tk.StringVar(value="n/a")
        self.stream_coach_pending_count_var = tk.StringVar(value="n/a")
        self.stream_coach_top_command_var = tk.StringVar(value="n/a")
        self.stream_coach_last_selected_command_var = tk.StringVar(value="n/a")
        self.stream_coach_last_action_var = tk.StringVar(value="n/a")
        self.stream_coach_rejected_count_var = tk.StringVar(value="n/a")
        self.stream_coach_last_vote_count_var = tk.StringVar(value="n/a")
        self.stream_coach_override_count_var = tk.StringVar(value="n/a")
        self.stream_coach_match_count_var = tk.StringVar(value="n/a")
        self.stream_coach_reward_total_var = tk.StringVar(value="n/a")
        self.fusion_bridge_available_var = tk.StringVar(value="n/a")
        self.fusion_bridge_enabled_status_var = tk.StringVar(value="n/a")
        self.fusion_last_command_var = tk.StringVar(value="n/a")
        self.fusion_last_result_var = tk.StringVar(value="n/a")
        self.fusion_attempt_count_var = tk.StringVar(value="n/a")
        self.fusion_success_count_var = tk.StringVar(value="n/a")
        self.fusion_failure_count_var = tk.StringVar(value="n/a")
        self.fusion_rejected_count_var = tk.StringVar(value="n/a")
        self.coach_queue_status_var = tk.StringVar(value="Queue idle")
        self.process_status_var = tk.StringVar(value="Idle")
        self.live_status_var = tk.StringVar(
            value=f"Live: {self.live_status_path} | exists=no | size=- | age=- | health=MISSING"
        )
        self.diagnostics_status_var = tk.StringVar(value="MISSING - no live status has been read")
        self.schema_keys_var = tk.StringVar(value="Schema keys: -")
        self.active_run_var = tk.StringVar(value="Active run: unknown")
        self.train_advanced_expanded_var = tk.BooleanVar(value=False)
        self.log_filter_var = tk.StringVar(value="")
        self.log_severity_var = tk.StringVar(value="All")
        self.log_pause_autoscroll_var = tk.BooleanVar(value=False)
        self.fusion_selected_seed_var = tk.StringVar(value="0")
        self.fusion_selected_tile_var = tk.StringVar(value="Selected tile: none")
        self.fusion_command_preview_var = tk.StringVar(value="Select a seed slot, then click lawn tiles.")
        self.diagnostic_visibility_vars: Dict[str, tk.BooleanVar] = {
            title: tk.BooleanVar(value=title in {"Gameplay", "Agent", "Reward Breakdown", "Human Interventions", "Viewer Queue Stats"})
            for title in (
                "Adventure",
                "Gameplay",
                "Agent",
                "Rows",
                "Eval",
                "Fusion",
                "Seed Inventory / Compatibility",
                "Reward Breakdown",
                "Action Distribution",
                "Plant Usage",
                "Fusion Usage",
                "Human Interventions",
                "Failed Episodes",
                "Viewer Queue Stats",
            )
        }

        self.panels: Dict[str, tk.Text] = {}
        self.diagnostic_panel_frames: Dict[str, ttk.LabelFrame] = {}
        self.last_panel_content: Dict[str, str] = {}
        self.last_adventure_status_content = ""
        self.last_generalist_status_content = ""
        self.launch_buttons: List[ttk.Button] = []
        self.stop_buttons: List[ttk.Button] = []
        self.log_text: Optional[scrolledtext.ScrolledText] = None
        self.train_preview: Optional[scrolledtext.ScrolledText] = None
        self.eval_preview: Optional[scrolledtext.ScrolledText] = None
        self.adventure_preview: Optional[scrolledtext.ScrolledText] = None
        self.generalist_preview: Optional[scrolledtext.ScrolledText] = None
        self.level3_preview: Optional[scrolledtext.ScrolledText] = None
        self.adventure_status_text: Optional[tk.Text] = None
        self.generalist_status_text: Optional[tk.Text] = None
        self.train_advanced_frame: Optional[ttk.LabelFrame] = None
        self.log_history: List[str] = []
        self.log_history_chars = 0
        self.log_dropped_lines = 0
        self.fusion_grid: Dict[Tuple[int, int], str] = {}
        self.fusion_tile_buttons: Dict[Tuple[int, int], ttk.Button] = {}
        self.fusion_selected_tile: Optional[Tuple[int, int]] = None
        self.structured_row_widget: Optional[ttk.Spinbox] = None
        self.structured_col_widget: Optional[ttk.Spinbox] = None
        self.structured_seed_widget: Optional[ttk.Spinbox] = None
        self.structured_custom_widget: Optional[ttk.Entry] = None
        self.assisted_queue_tree: Optional[ttk.Treeview] = None
        self._structured_raw_copy_value = ""
        self.assisted_command_queue = AssistedCommandQueue()
        self.intervention_logger = InterventionJSONLLogger(self._resolve_path(DEFAULT_INTERVENTION_LOG_PATH))
        self.log_queue: queue.Queue[Any] = queue.Queue(maxsize=LOG_QUEUE_MAX_ITEMS)
        self._log_queue_put_lock = threading.Lock()
        self._log_queue_drop_lock = threading.Lock()
        self._log_queue_dropped_items = 0
        self.active_process: Optional[subprocess.Popen[str]] = None
        self.active_process_name = ""
        self.active_process_started_at: Optional[float] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._stopper_thread: Optional[threading.Thread] = None
        self._stopping_process: Optional[subprocess.Popen[str]] = None
        self._poll_after_id: Optional[str] = None
        self._log_after_id: Optional[str] = None
        self._log_view_after_id: Optional[str] = None
        self._close_after_id: Optional[str] = None
        self._last_log_view_refresh_at = 0.0
        self._log_view_dirty = False
        self._log_notice_present = False
        self._closing = False
        self._destroyed = False
        self._close_deadline = 0.0
        self.active_run_path = ""
        self.live_writer_warning_emitted = False
        self.last_good_status: Optional[Dict[str, Any]] = None
        self.last_good_read_time: Optional[float] = None
        self.last_live_parse_error = ""
        self.last_live_health = ""
        self.last_live_warning_key = ""
        self._live_status_reader = LiveStatusReader(self.live_status_path)
        self._diagnostics_render_key: Optional[DiagnosticsRenderKey] = None

        self._build()
        self._append_log(f"Project root: {self.project_root}\n")
        self._append_log(f"Live status path: {self.live_status_path}\n")
        self._append_log(f"[gui] polling diagnostics every {POLL_MS} ms\n")
        self._bind_command_preview_updates()
        self._update_command_previews()
        if default_model is not None:
            self._append_log(f"Selected default model path: {default_model}\n")
        if default_adventure_model is not None and default_adventure_model != default_model:
            self._append_log(f"Selected Adventure metadata-backed model path: {default_adventure_model}\n")
        self._poll()
        self._drain_log_queue()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _resolve_path(self, path: Path) -> Path:
        expanded = path.expanduser()
        if expanded.is_absolute():
            return expanded.resolve()
        return (self.project_root / expanded).resolve()

    def _resolve_text_path(self, raw_path: str) -> Path:
        return self._resolve_path(Path(raw_path.strip()))

    def _path_for_command(self, path: Path) -> str:
        resolved = self._resolve_path(path)
        try:
            return str(resolved.relative_to(self.project_root))
        except ValueError:
            return str(resolved)

    def _path_for_display(self, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return "n/a"
        try:
            path = self._resolve_text_path(text)
            return str(path.relative_to(self.project_root))
        except (OSError, ValueError):
            return text

    def _find_newest_model_zip(self) -> Optional[Path]:
        runs_dir = self.repo_root / "runs"
        if not runs_dir.exists():
            return None
        zips: List[Path] = []
        for path in runs_dir.rglob("*.zip"):
            try:
                if not path.is_file() or path.stat().st_size <= 0:
                    continue
            except OSError:
                continue
            zips.append(path)
        preferred = [
            path
            for path in zips
            if path.name.lower() in MODEL_ZIP_NAMES or path.parent.name.lower() == "checkpoints"
        ]
        candidates = preferred or zips
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)

    def _find_newest_usable_model_zip(self) -> Optional[Path]:
        runs_dir = self.repo_root / "runs"
        if not runs_dir.exists():
            return None
        zips: List[Path] = []
        for path in runs_dir.rglob("*.zip"):
            try:
                if not path.is_file() or path.stat().st_size <= 0 or not self._has_canonical_model_metadata(path):
                    continue
            except OSError:
                continue
            zips.append(path)
        preferred = [
            path
            for path in zips
            if path.name.lower() in MODEL_ZIP_NAMES or path.parent.name.lower() == "checkpoints"
        ]
        candidates = preferred or zips
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)

    def _has_canonical_model_metadata(self, model_path: Path) -> bool:
        directories = [model_path.parent]
        if model_path.parent.name.lower() == "checkpoints":
            directories.append(model_path.parent.parent)
        for directory in directories:
            metadata_path = directory / "model_metadata.json"
            if not metadata_path.exists():
                continue
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict) and payload.get("metadata_version") is not None:
                return True
        return False

    def _build(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=4)
        self.root.rowconfigure(2, weight=1)

        notebook = ttk.Notebook(self.root)
        notebook.grid(row=0, column=0, sticky="nsew", padx=8, pady=(8, 4))

        train_tab = ttk.Frame(notebook)
        eval_tab = ttk.Frame(notebook)
        coach_tab = ttk.Frame(notebook)
        diagnostics_tab = ttk.Frame(notebook)
        runs_tab = ttk.Frame(notebook)
        notebook.add(train_tab, text="Train")
        notebook.add(eval_tab, text="Eval")
        notebook.add(coach_tab, text="Coach")
        notebook.add(diagnostics_tab, text="Diagnostics")
        notebook.add(runs_tab, text="Runs/Models")

        self._build_train_tab(train_tab)
        self._build_eval_tab(eval_tab)
        self._build_coach_tab(coach_tab)
        self._build_diagnostics_tab(diagnostics_tab)
        self._build_runs_models_tab(runs_tab)
        self._build_status_bar()
        self._build_log_panel()

    def _build_runs_models_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        notebook = ttk.Notebook(parent)
        notebook.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        runs = ttk.Frame(notebook)
        fusion = ttk.Frame(notebook)
        notebook.add(runs, text="Runs and Models")
        notebook.add(fusion, text="Fusion Planner")
        self._build_runs_tab(runs)
        self._build_fusion_tab(fusion)

    def _build_status_bar(self) -> None:
        frame = ttk.Frame(self.root)
        frame.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 4))
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=4)
        ttk.Label(frame, text="Process:").grid(row=0, column=0, sticky="w")
        ttk.Label(frame, textvariable=self.process_status_var, anchor="w").grid(row=0, column=1, sticky="ew", padx=(4, 12))
        ttk.Label(frame, textvariable=self.live_status_var, anchor="w").grid(row=0, column=2, sticky="ew", padx=(0, 8))
        self._add_stop_button(frame, "Stop Active Process").grid(row=0, column=3, sticky="e")

    def _build_log_panel(self) -> None:
        frame = ttk.LabelFrame(self.root, text="Subprocess logs")
        frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 8))
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)
        toolbar = ttk.Frame(frame)
        toolbar.grid(row=0, column=0, sticky="ew", padx=4, pady=(2, 4))
        toolbar.columnconfigure(1, weight=1)
        ttk.Label(toolbar, text="Search").grid(row=0, column=0, sticky="w")
        ttk.Entry(toolbar, textvariable=self.log_filter_var, width=28).grid(
            row=0, column=1, sticky="ew", padx=(5, 8)
        )
        ttk.Label(toolbar, text="Severity").grid(row=0, column=2, sticky="w")
        ttk.Combobox(
            toolbar,
            textvariable=self.log_severity_var,
            values=("All", "ERROR", "Warning", "Live status", "Process"),
            state="readonly",
            width=12,
        ).grid(row=0, column=3, sticky="w", padx=(5, 8))
        ttk.Checkbutton(
            toolbar,
            text="Pause autoscroll",
            variable=self.log_pause_autoscroll_var,
        ).grid(row=0, column=4, sticky="w", padx=(0, 8))
        ttk.Button(toolbar, text="Copy selected", command=self.copy_selected_logs).grid(
            row=0, column=5, sticky="w", padx=(0, 5)
        )
        ttk.Button(toolbar, text="Clear", command=self.clear_logs).grid(row=0, column=6, sticky="w")
        self.log_text = scrolledtext.ScrolledText(frame, height=5, wrap="word", font=("Consolas", 9))
        self.log_text.grid(row=1, column=0, sticky="nsew")
        self.log_text.configure(state="disabled")
        self.log_filter_var.trace_add("write", lambda *_args: self._refresh_log_view())
        self.log_severity_var.trace_add("write", lambda *_args: self._refresh_log_view())

    def _build_train_tab(self, parent: ttk.Frame) -> None:
        content = self._make_scrollable_container(parent)
        content.columnconfigure(0, weight=1)

        status = ttk.LabelFrame(content, text="Adventure Generalist Training")
        status.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 3))
        status.columnconfigure(1, weight=1)
        ttk.Label(status, text="Mode", style="Heading.TLabel").grid(row=0, column=0, sticky="w", padx=6, pady=3)
        ttk.Label(
            status,
            text=f"{ADVENTURE_GENERALIST_MODEL_FAMILY} · 14-slot identity-aware policy",
        ).grid(row=0, column=1, sticky="w", padx=6, pady=3)
        ttk.Label(status, text="Process").grid(row=0, column=2, sticky="e", padx=(6, 2), pady=3)
        ttk.Label(status, textvariable=self.process_status_var).grid(row=0, column=3, sticky="w", padx=(2, 6), pady=3)
        ttk.Label(status, text="Lab mode").grid(row=1, column=0, sticky="w", padx=6, pady=3)
        ttk.Combobox(
            status, textvariable=self.train_lab_mode_var, values=LAB_MODES, state="readonly", width=13
        ).grid(row=1, column=1, sticky="w", padx=6, pady=3)
        ttk.Checkbutton(status, text="Streamer mode", variable=self.stream_coach_enabled_var).grid(
            row=1, column=2, sticky="e", padx=6, pady=3
        )
        ttk.Label(status, textvariable=self.assisted_execution_mode_var).grid(
            row=1, column=3, sticky="w", padx=6, pady=3
        )
        ttk.Label(status, textvariable=self.command_enablement_var).grid(
            row=2, column=0, columnspan=4, sticky="w", padx=6, pady=(0, 3)
        )

        core = ttk.LabelFrame(content, text="Core Training Settings")
        core.grid(row=1, column=0, sticky="ew", padx=8, pady=3)
        for column in (1, 3):
            core.columnconfigure(column, weight=1)
        self._add_labeled_entry(
            core, 0, 0, "Timesteps", self.generalist_total_timesteps_var,
            tooltip="Total PPO environment steps for this Adventure Generalist run.",
        )
        self._add_labeled_entry(
            core, 0, 2, "Initial loadout", self.generalist_initial_loadout_var, width=42,
            tooltip="Fixed bootstrap loadout used before Adventure unlocks expand the seed inventory.",
        )
        self._add_labeled_entry(
            core, 1, 0, "Start level", self.generalist_start_level_var,
            tooltip="First Adventure level eligible for the curriculum frontier.",
        )
        self._add_labeled_entry(
            core, 1, 2, "Max levels", self.generalist_max_levels_var,
            tooltip="Highest Adventure level included in this training run.",
        )
        self._add_labeled_entry(
            core, 2, 0, "Max attempts", self.generalist_max_attempts_var,
            tooltip="Maximum failed attempts allowed per level before the run blocks.",
        )
        self._add_labeled_entry(
            core, 2, 2, "Game speed", self.generalist_game_speed_var,
            tooltip="PvZ simulation speed multiplier used by the bridge.",
        )

        run = ttk.LabelFrame(content, text="Model / Run Paths")
        run.grid(row=2, column=0, sticky="ew", padx=8, pady=3)
        run.columnconfigure(1, weight=1)
        self._add_labeled_entry(
            run, 0, 0, "Run directory", self.generalist_run_dir_var, width=68,
            tooltip="Optional output directory. Blank lets train_ppo.py create the run folder.",
        )
        ttk.Button(run, text="Browse", command=self.browse_run_folder).grid(row=0, column=2, padx=(0, 6), pady=2)
        self._add_labeled_entry(
            run, 1, 0, "Resume model .zip", self.generalist_resume_model_path_var, width=68,
            tooltip="Existing compatible 14-slot model. Leave blank for a fresh initialization.",
        )
        ttk.Button(run, text="Browse", command=self.browse_generalist_resume_model).grid(row=1, column=2, padx=(0, 4), pady=2)
        ttk.Button(run, text="Refresh", command=self.refresh_generalist_resume_model).grid(row=1, column=3, padx=(0, 6), pady=2)

        actions = ttk.Frame(content)
        actions.grid(row=3, column=0, sticky="ew", padx=8, pady=3)
        self._add_launch_button(actions, "Start Training", self.start_adventure_generalist_train).grid(
            row=0, column=0, sticky="w", padx=(0, 5)
        )
        self._add_launch_button(actions, "Resume Training", self.resume_training).grid(
            row=0, column=1, sticky="w", padx=(0, 5)
        )
        self._add_stop_button(actions, "Stop Process").grid(row=0, column=2, sticky="w", padx=(0, 5))
        ttk.Button(actions, text="Advanced Settings ▸", command=self._toggle_train_advanced).grid(
            row=0, column=3, sticky="w"
        )

        advanced = ttk.LabelFrame(content, text="Advanced Settings")
        advanced.grid(row=4, column=0, sticky="ew", padx=8, pady=3)
        for column in (1, 3):
            advanced.columnconfigure(column, weight=1)
        self.train_advanced_frame = advanced
        self._add_labeled_entry(
            advanced, 0, 0, "Checkpoint frequency", self.generalist_checkpoint_freq_var,
            tooltip="Environment-step interval between saved PPO checkpoints.",
        )
        self._add_labeled_entry(
            advanced, 0, 2, "Step seconds", self.generalist_step_seconds_var,
            tooltip="Wall-clock delay between bridge actions.",
        )
        self._add_labeled_entry(
            advanced, 1, 0, "Board timeout", self.generalist_board_timeout_var,
            tooltip="Seconds allowed for the board to reach a usable state.",
        )
        self._add_labeled_entry(
            advanced, 1, 2, "Soft / hard max steps", self.generalist_soft_max_steps_var,
            tooltip="Soft episode step budget; the hard cap is configured beside it.",
        )
        ttk.Entry(advanced, textvariable=self.generalist_hard_max_steps_var, width=12).grid(
            row=1, column=4, sticky="w", padx=6, pady=2
        )
        ttk.Label(advanced, text="Curriculum ⓘ").grid(row=2, column=0, sticky="w", padx=6, pady=2)
        curriculum = ttk.Combobox(
            advanced,
            textvariable=self.generalist_curriculum_var,
            values=("conservative", "varied"),
            state="readonly",
            width=15,
        )
        curriculum.grid(row=2, column=1, sticky="w", padx=6, pady=2)
        _Tooltip(curriculum, "Controls how aggressively cleared levels and the current frontier are sampled.")
        self._add_labeled_entry(advanced, 2, 2, "Unlock delay", self.generalist_unlock_delay_var)
        self._add_labeled_entry(advanced, 3, 0, "Frontier probability", self.generalist_frontier_prob_var)
        self._add_labeled_entry(advanced, 3, 2, "Recent / maintenance", self.generalist_recent_prob_var)
        ttk.Entry(advanced, textvariable=self.generalist_maintenance_prob_var, width=12).grid(
            row=3, column=4, sticky="w", padx=6, pady=2
        )
        self._add_labeled_entry(advanced, 4, 0, "New plant min probability", self.generalist_new_plant_prob_var)
        self._add_labeled_entry(advanced, 4, 2, "Promotion win streak", self.generalist_frontier_win_streak_required_var)
        option_specs = (
            ("Unlock-aware curriculum", self.generalist_unlock_curriculum_var),
            ("Replay cleared levels", self.generalist_replay_cleared_var),
            ("Randomize seed order", self.generalist_randomize_seed_order_var),
            ("Wait for gameplay", self.generalist_wait_gameplay_ready_var),
            ("Quick wait", self.generalist_quick_wait_var),
            ("Final-wave extension", self.generalist_final_wave_extension_var),
            ("Tactical masks", self.generalist_tactical_masks_var),
            ("WallNut mask", self.generalist_wallnut_mask_var),
            ("CherryBomb mask", self.generalist_cherrybomb_mask_var),
            ("Fusion mask (model self-fuse)", self.generalist_fusion_action_mask_train_var),
        )
        for index, (label, variable) in enumerate(option_specs):
            ttk.Checkbutton(advanced, text=label, variable=variable).grid(
                row=5 + index // 3, column=index % 3, sticky="w", padx=6, pady=2
            )
        advanced.grid_remove()

        preview = ttk.LabelFrame(content, text="Command Preview")
        preview.grid(row=5, column=0, sticky="ew", padx=8, pady=(3, 6))
        preview.rowconfigure(0, weight=1)
        preview.columnconfigure(0, weight=1)
        self.train_preview = self._preview_box(preview, height=6)

    def _build_eval_tab(self, parent: ttk.Frame) -> None:
        content = self._make_scrollable_container(parent)
        content.columnconfigure(0, weight=1)

        header = ttk.LabelFrame(content, text="Adventure Generalist Evaluation")
        header.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 3))
        header.columnconfigure(1, weight=1)
        ttk.Label(
            header,
            text="Compatibility",
        ).grid(row=0, column=0, sticky="w", padx=6, pady=3)
        ttk.Label(
            header,
            text=f"Requires family={ADVENTURE_GENERALIST_MODEL_FAMILY}, action_count=701, max_seed_slots=14",
        ).grid(row=0, column=1, sticky="w", padx=6, pady=3)
        ttk.Label(header, text="Lab mode").grid(row=1, column=0, sticky="w", padx=6, pady=3)
        ttk.Combobox(
            header, textvariable=self.eval_lab_mode_var, values=LAB_MODES, state="readonly", width=13
        ).grid(row=1, column=1, sticky="w", padx=6, pady=3)
        ttk.Checkbutton(header, text="Streamer mode", variable=self.stream_coach_enabled_var).grid(
            row=1, column=2, sticky="w", padx=6, pady=3
        )
        ttk.Label(header, textvariable=self.command_enablement_var).grid(
            row=2, column=0, columnspan=3, sticky="w", padx=6, pady=(0, 3)
        )

        model = ttk.LabelFrame(content, text="Model / Run")
        model.grid(row=1, column=0, sticky="ew", padx=8, pady=3)
        model.columnconfigure(1, weight=1)
        self._add_labeled_entry(
            model, 0, 0, "Model .zip", self.generalist_eval_model_path_var, width=68,
            tooltip="Metadata-backed Adventure Generalist checkpoint selected for evaluation.",
        )
        ttk.Button(model, text="Browse model", command=self.browse_generalist_eval_model).grid(
            row=0, column=2, padx=(0, 4), pady=2
        )
        ttk.Button(model, text="Refresh", command=self.refresh_generalist_eval_model).grid(
            row=0, column=3, padx=(0, 6), pady=2
        )
        self._add_labeled_entry(
            model, 1, 0, "Run directory", self.generalist_run_dir_var, width=68,
            tooltip="Optional output directory for evaluation metrics and live status.",
        )
        ttk.Button(model, text="Browse run", command=self.browse_run_folder).grid(
            row=1, column=2, padx=(0, 6), pady=2
        )

        settings = ttk.LabelFrame(content, text="Evaluation Settings")
        settings.grid(row=2, column=0, sticky="ew", padx=8, pady=3)
        for column in (1, 3):
            settings.columnconfigure(column, weight=1)
        self._add_labeled_entry(settings, 0, 0, "Episodes", self.generalist_eval_episodes_var)
        self._add_labeled_entry(settings, 0, 2, "Initial loadout", self.generalist_initial_loadout_var, width=42)
        self._add_labeled_entry(settings, 1, 0, "Start level", self.generalist_start_level_var)
        self._add_labeled_entry(settings, 1, 2, "Max levels", self.generalist_max_levels_var)
        self._add_labeled_entry(settings, 2, 0, "Max attempts", self.generalist_max_attempts_var)
        self._add_labeled_entry(settings, 2, 2, "Game speed", self.generalist_game_speed_var)
        self._add_labeled_entry(settings, 3, 0, "Step seconds", self.generalist_step_seconds_var)
        self._add_labeled_entry(settings, 3, 2, "Board timeout", self.generalist_board_timeout_var)
        ttk.Checkbutton(settings, text="Wait for gameplay", variable=self.generalist_wait_gameplay_ready_var).grid(
            row=4, column=0, sticky="w", padx=6, pady=3
        )
        ttk.Checkbutton(settings, text="Quick wait", variable=self.generalist_quick_wait_var).grid(
            row=4, column=1, sticky="w", padx=6, pady=3
        )
        ttk.Checkbutton(settings, text="Tactical masks", variable=self.generalist_tactical_masks_var).grid(
            row=4, column=2, sticky="w", padx=6, pady=3
        )
        ttk.Checkbutton(settings, text="Fusion mask (model self-fuse)", variable=self.generalist_fusion_action_mask_eval_var).grid(
            row=5, column=0, columnspan=2, sticky="w", padx=6, pady=3
        )

        actions = ttk.Frame(content)
        actions.grid(row=3, column=0, sticky="ew", padx=8, pady=3)
        self._add_launch_button(actions, "Start Evaluation", self.start_adventure_generalist_eval).grid(
            row=0, column=0, sticky="w", padx=(0, 5)
        )
        self._add_stop_button(actions, "Stop Process").grid(row=0, column=1, sticky="w", padx=(0, 5))

        utilities = ttk.LabelFrame(content, text="Model / Run Utilities")
        utilities.grid(row=4, column=0, sticky="ew", padx=8, pady=3)
        utility_specs = (
            ("Browse model", self.browse_generalist_eval_model),
            ("Browse run folder", self.browse_run_folder),
            ("Open run folder", self.open_run_folder),
            ("Refresh models", self.refresh_generalist_models),
            ("Analyze selected run", self.analyze_selected_run),
            ("Show charts", self.show_charts),
        )
        for index, (label, command) in enumerate(utility_specs):
            ttk.Button(utilities, text=label, command=command).grid(
                row=index // 3, column=index % 3, sticky="ew", padx=4, pady=3
            )
            utilities.columnconfigure(index % 3, weight=1)

        preview = ttk.LabelFrame(content, text="Command Preview")
        preview.grid(row=5, column=0, sticky="ew", padx=8, pady=(3, 6))
        preview.rowconfigure(0, weight=1)
        preview.columnconfigure(0, weight=1)
        self.eval_preview = self._preview_box(preview, height=6)

    def _build_diagnostics_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(3, weight=1)

        toolbar = ttk.Frame(parent)
        toolbar.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 3))
        toolbar.columnconfigure(2, weight=1)
        ttk.Button(toolbar, text="Refresh Now", command=self.refresh_diagnostics_now).grid(
            row=0, column=0, sticky="w", padx=(0, 5)
        )
        ttk.Button(toolbar, text="Show Raw JSON", command=self.show_raw_live_json).grid(
            row=0, column=1, sticky="w", padx=(0, 10)
        )
        ttk.Label(toolbar, textvariable=self.diagnostics_status_var, anchor="w").grid(
            row=0, column=2, sticky="ew", padx=(0, 10)
        )
        ttk.Label(toolbar, textvariable=self.active_run_var, anchor="e").grid(row=0, column=3, sticky="e")

        visibility = ttk.LabelFrame(parent, text="Visible Panels")
        visibility.grid(row=1, column=0, sticky="ew", padx=8, pady=3)
        for index, title in enumerate(self.diagnostic_visibility_vars):
            ttk.Checkbutton(
                visibility,
                text=title,
                variable=self.diagnostic_visibility_vars[title],
                command=self._toggle_diagnostic_panels,
            ).grid(row=index // 4, column=index % 4, sticky="w", padx=6, pady=2)
            visibility.columnconfigure(index % 4, weight=1)

        tests = ttk.Frame(parent)
        tests.grid(row=2, column=0, sticky="ew", padx=8, pady=3)
        ttk.Label(tests, text="Targeted tests:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        for index, (label, command) in enumerate(
            (
                ("Auto reset", self.auto_reset_test),
                ("Seed selection", self.seed_selection_test),
                ("Cooldown", self.cooldown_test),
                ("Bridge / perf", self.bridge_perf_diagnostic),
            ),
            start=1,
        ):
            self._add_launch_button(tests, label, command).grid(row=0, column=index, sticky="w", padx=(0, 5))

        grid = ttk.Frame(parent)
        grid.grid(row=3, column=0, sticky="nsew", padx=8, pady=(0, 6))
        self.diagnostic_panel_container = grid
        for column in range(3):
            grid.columnconfigure(column, weight=1)
        for row in range(2):
            grid.rowconfigure(row, weight=1)

        for title in self.diagnostic_visibility_vars:
            frame = ttk.LabelFrame(grid, text=title)
            frame.rowconfigure(0, weight=1)
            frame.columnconfigure(0, weight=1)
            text = tk.Text(frame, height=5, wrap="word", font=("Consolas", 9))
            text.grid(row=0, column=0, sticky="nsew")
            text.configure(state="disabled")
            self.panels[title] = text
            self.diagnostic_panel_frames[title] = frame
        self._toggle_diagnostic_panels()

    def _build_coach_tab(self, parent: ttk.Frame) -> None:
        content = self._make_scrollable_container(parent)
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=1)

        structured = ttk.LabelFrame(content, text="Structured Coach Command")
        structured.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=(6, 3))
        structured.columnconfigure(1, weight=1)
        structured.columnconfigure(7, weight=1)
        ttk.Label(structured, text="Command").grid(row=0, column=0, sticky="w", padx=6, pady=3)
        command_combo = ttk.Combobox(
            structured,
            textvariable=self.structured_command_type_var,
            values=STRUCTURED_COACH_COMMANDS,
            state="readonly",
            width=11,
        )
        command_combo.grid(row=0, column=1, sticky="w", padx=6, pady=3)
        _Tooltip(command_combo, "Select the coach command type to serialize.")

        ttk.Label(structured, text="Row").grid(row=0, column=2, sticky="w", padx=(12, 3), pady=3)
        self.structured_row_widget = ttk.Spinbox(
            structured,
            textvariable=self.structured_row_var,
            from_=0,
            to=LAWN_ROWS - 1,
            width=5,
        )
        self.structured_row_widget.grid(row=0, column=3, sticky="w", padx=3, pady=3)
        _Tooltip(self.structured_row_widget, f"Lawn row, from 0 through {LAWN_ROWS - 1}.")

        ttk.Label(structured, text="Column").grid(row=0, column=4, sticky="w", padx=(12, 3), pady=3)
        self.structured_col_widget = ttk.Spinbox(
            structured,
            textvariable=self.structured_col_var,
            from_=0,
            to=LAWN_COLS - 1,
            width=5,
        )
        self.structured_col_widget.grid(row=0, column=5, sticky="w", padx=3, pady=3)
        _Tooltip(self.structured_col_widget, f"Lawn column, from 0 through {LAWN_COLS - 1}.")

        ttk.Label(structured, text="Seed packet").grid(row=0, column=6, sticky="w", padx=(12, 3), pady=3)
        self.structured_seed_widget = ttk.Spinbox(
            structured,
            textvariable=self.structured_seed_slot_var,
            from_=0,
            to=SEED_PACKET_SLOTS - 1,
            width=5,
        )
        self.structured_seed_widget.grid(row=0, column=7, sticky="w", padx=3, pady=3)
        _Tooltip(
            self.structured_seed_widget,
            f"Adventure Generalist seed slot, from 0 through {SEED_PACKET_SLOTS - 1}.",
        )

        ttk.Label(structured, text="Plant / target").grid(row=1, column=0, sticky="w", padx=6, pady=3)
        self.structured_custom_widget = ttk.Entry(
            structured,
            textvariable=self.structured_custom_text_var,
            state="normal",
        )
        self.structured_custom_widget.grid(row=1, column=1, columnspan=7, sticky="ew", padx=6, pady=3)
        _Tooltip(self.structured_custom_widget, "Optional plant/fusion target. PLANT/FUSE use the numeric seed slot above.")

        ttk.Label(structured, text="Preview").grid(row=2, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(
            structured,
            textvariable=self.structured_preview_var,
            state="readonly",
            font=("Consolas", 9),
        ).grid(row=2, column=1, columnspan=5, sticky="ew", padx=6, pady=3)
        ttk.Button(
            structured,
            text="Send Structured Command",
            command=self.send_structured_coach_command,
        ).grid(row=2, column=6, sticky="ew", padx=4, pady=3)
        ttk.Button(
            structured,
            text="Copy to Raw Input",
            command=self.copy_structured_command_to_raw,
        ).grid(row=2, column=7, sticky="ew", padx=(4, 6), pady=3)

        quick = ttk.Frame(structured)
        quick.grid(row=3, column=0, columnspan=8, sticky="ew", padx=6, pady=(2, 3))
        ttk.Label(quick, text="Source").grid(row=0, column=0, sticky="w", padx=(0, 4))
        ttk.Entry(quick, textvariable=self.assisted_command_source_var, width=12).grid(row=0, column=1, padx=(0, 6))
        ttk.Label(quick, text="User").grid(row=0, column=2, sticky="w", padx=(0, 4))
        ttk.Entry(quick, textvariable=self.assisted_command_user_var, width=12).grid(row=0, column=3, padx=(0, 8))
        ttk.Label(quick, text="Execution mode").grid(row=0, column=4, sticky="w", padx=(0, 4))
        ttk.Combobox(
            quick,
            textvariable=self.assisted_execution_mode_var,
            values=ASSISTED_EXECUTION_MODES,
            state="readonly",
            width=18,
        ).grid(row=0, column=5, padx=(0, 8))
        ttk.Label(quick, text="Intervention log").grid(row=0, column=6, sticky="w", padx=(0, 4))
        ttk.Entry(quick, textvariable=self.intervention_log_path_var, width=36).grid(row=0, column=7, sticky="ew")
        quick.columnconfigure(7, weight=1)
        ttk.Label(quick, text="Quick:").grid(row=1, column=0, sticky="w", padx=(0, 4))
        for index, (label, command) in enumerate(
            (("Save Sun", "SAVE_SUN"), ("Pause Agent", "PAUSE_AGENT"), ("Resume Agent", "RESUME_AGENT"), ("Force Eval", "FORCE_EVAL")),
            start=1,
        ):
            ttk.Button(
                quick,
                text=label,
                command=lambda value=command: self._set_structured_command_type(value),
            ).grid(row=1, column=index, sticky="w", padx=(0, 4))
        ttk.Label(
            structured,
            text="Format: COMMAND ROW COL TARGET. Example: PLANT 2 4 0",
        ).grid(row=4, column=0, columnspan=4, sticky="w", padx=6, pady=(2, 4))
        ttk.Label(structured, textvariable=self.coach_queue_status_var, anchor="e").grid(
            row=4, column=4, columnspan=4, sticky="ew", padx=6, pady=(2, 4)
        )

        queue_frame = ttk.LabelFrame(content, text="Assisted Command Queue")
        queue_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=8, pady=3)
        queue_frame.columnconfigure(0, weight=1)
        columns = ("time", "source", "command", "row", "col", "target", "status")
        self.assisted_queue_tree = ttk.Treeview(queue_frame, columns=columns, show="headings", height=6)
        for name, width in (("time", 75), ("source", 150), ("command", 120), ("row", 45), ("col", 45), ("target", 90), ("status", 80)):
            self.assisted_queue_tree.heading(name, text=name.title())
            self.assisted_queue_tree.column(name, width=width, stretch=name in {"source", "command", "target"})
        self.assisted_queue_tree.grid(row=0, column=0, columnspan=5, sticky="ew", padx=6, pady=4)
        for index, (label, callback) in enumerate(
            (("Approve", self.approve_assisted_command), ("Reject", self.reject_assisted_command),
             ("Modify", self.modify_assisted_command), ("Execute", self.execute_assisted_command))
        ):
            ttk.Button(queue_frame, text=label, command=callback).grid(row=1, column=index, padx=(6 if index == 0 else 2, 2), pady=(0, 5), sticky="w")
        ttk.Label(queue_frame, textvariable=self.assisted_queue_summary_var, anchor="e").grid(
            row=1, column=4, sticky="e", padx=6, pady=(0, 5)
        )

        manual = ttk.LabelFrame(content, text="Raw Manual Command")
        manual.grid(row=2, column=0, sticky="nsew", padx=(8, 4), pady=3)
        manual.columnconfigure(1, weight=1)
        ttk.Checkbutton(manual, text="Enable human coach", variable=self.human_coach_enabled_var).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=6, pady=3
        )
        self._add_labeled_entry(manual, 1, 0, "Command queue", self.human_coach_command_path_var, width=42)
        self._add_labeled_value(manual, 2, "Queue status", self.coach_queue_status_var, width=42)
        command_row = ttk.Frame(manual)
        command_row.grid(row=3, column=0, columnspan=2, sticky="ew", padx=6, pady=3)
        command_row.columnconfigure(0, weight=1)
        ttk.Entry(command_row, textvariable=self.human_coach_command_input_var).grid(
            row=0, column=0, sticky="ew", padx=(0, 5)
        )
        ttk.Button(command_row, text="Send command", command=self.send_human_coach_command).grid(row=0, column=1)
        ttk.Label(
            manual,
            text="Raw input keeps the existing parser syntax and queue behavior.",
        ).grid(row=4, column=0, columnspan=2, sticky="w", padx=6, pady=(2, 6))

        streamer = ttk.LabelFrame(content, text="Streamer Coach")
        streamer.grid(row=2, column=1, sticky="nsew", padx=(4, 8), pady=3)
        streamer.columnconfigure(1, weight=1)
        ttk.Checkbutton(streamer, text="Enable stream coach", variable=self.stream_coach_enabled_var).grid(
            row=0, column=0, sticky="w", padx=6, pady=3
        )
        ttk.Checkbutton(streamer, text="Dry run", variable=self.stream_coach_dry_run_var).grid(
            row=0, column=1, sticky="w", padx=6, pady=3
        )
        ttk.Label(streamer, text="Platform").grid(row=1, column=0, sticky="w", padx=6, pady=2)
        ttk.Combobox(
            streamer,
            textvariable=self.stream_coach_platform_var,
            values=STREAM_COACH_PLATFORMS,
            state="readonly",
            width=14,
        ).grid(row=1, column=1, sticky="w", padx=6, pady=2)
        self._add_labeled_entry(streamer, 2, 0, "Mock script", self.stream_coach_mock_script_var, width=42)
        self._add_labeled_entry(streamer, 3, 0, "Stream log", self.stream_coach_log_path_var, width=42)
        self._add_labeled_entry(streamer, 4, 0, "Voting window", self.stream_coach_window_sec_var)
        self._add_labeled_entry(streamer, 5, 0, "Minimum votes", self.stream_coach_min_votes_var)
        self._add_labeled_entry(streamer, 6, 0, "Max actions / min", self.stream_coach_max_actions_per_minute_var)

        rewards = ttk.LabelFrame(content, text="Coach Rewards")
        rewards.grid(row=3, column=0, sticky="nsew", padx=(8, 4), pady=3)
        rewards.columnconfigure(1, weight=1)
        ttk.Checkbutton(rewards, text="Human coach reward", variable=self.human_coach_reward_var).grid(
            row=0, column=0, sticky="w", padx=6, pady=3
        )
        ttk.Checkbutton(rewards, text="Stream coach reward", variable=self.stream_coach_reward_var).grid(
            row=0, column=1, sticky="w", padx=6, pady=3
        )
        self._add_labeled_entry(rewards, 1, 0, "Legal execution reward", self.human_coach_bonus_var)
        self._add_labeled_entry(rewards, 2, 0, "Match reward", self.human_coach_match_bonus_var)
        self._add_labeled_entry(rewards, 3, 0, "Override penalty", self.human_coach_override_penalty_var)
        self._add_labeled_entry(rewards, 4, 0, "Fusion success reward", self.human_coach_fusion_reward_var)
        self._add_labeled_entry(rewards, 5, 0, "Tactical usefulness reward", self.human_coach_tactical_reward_var)
        self._add_labeled_value(rewards, 6, "Human reward total", self.human_coach_reward_total_var, width=32)
        self._add_labeled_value(rewards, 7, "Stream reward total", self.stream_coach_reward_total_var, width=32)

        fusion = ttk.LabelFrame(content, text="Fusion Bridge Controls")
        fusion.grid(row=3, column=1, sticky="nsew", padx=(4, 8), pady=3)
        fusion.columnconfigure(1, weight=1)
        ttk.Checkbutton(fusion, text="Enable fusion planning", variable=self.coach_allow_fusion_planning_var).grid(
            row=0, column=0, sticky="w", padx=6, pady=3
        )
        ttk.Checkbutton(fusion, text="Enable fusion bridge", variable=self.fusion_bridge_enabled_var).grid(
            row=0, column=1, sticky="w", padx=6, pady=3
        )
        self._add_labeled_value(fusion, 1, "Bridge available", self.fusion_bridge_available_var, width=36)
        self._add_labeled_value(fusion, 2, "Bridge enabled live", self.fusion_bridge_enabled_status_var, width=36)
        self._add_labeled_value(fusion, 3, "Last fusion command", self.fusion_last_command_var, width=36)
        self._add_labeled_value(fusion, 4, "Last fusion result", self.fusion_last_result_var, width=36)
        self._add_labeled_value(fusion, 5, "Attempts / successes", self.fusion_attempt_count_var, width=36)
        self._add_labeled_value(fusion, 6, "Failures / rejected", self.fusion_failure_count_var, width=36)
        ttk.Label(fusion, textvariable=self.fusion_command_feedback_var, wraplength=460).grid(
            row=7, column=0, columnspan=2, sticky="w", padx=6, pady=(2, 5)
        )

        live = ttk.LabelFrame(content, text="Live Coach Diagnostics")
        live.grid(row=4, column=0, columnspan=2, sticky="ew", padx=8, pady=(3, 6))
        for column in range(3):
            live.columnconfigure(column, weight=1)

        human_live = ttk.LabelFrame(live, text="Manual")
        human_live.grid(row=0, column=0, sticky="nsew", padx=(4, 2), pady=4)
        human_live.columnconfigure(1, weight=1)
        for row, (label, variable) in enumerate(
            (
                ("Enabled", self.human_coach_enabled_status_var),
                ("Last command", self.human_coach_last_command_var),
                ("Last action", self.human_coach_last_action_var),
                ("Last error", self.human_coach_last_error_var),
                ("Overrides", self.human_coach_override_count_var),
                ("Matches", self.human_coach_match_count_var),
                ("Reward total", self.human_coach_reward_total_var),
            )
        ):
            self._add_labeled_value(human_live, row, label, variable, width=28)

        stream_live = ttk.LabelFrame(live, text="Streamer")
        stream_live.grid(row=0, column=1, sticky="nsew", padx=2, pady=4)
        stream_live.columnconfigure(1, weight=1)
        for row, (label, variable) in enumerate(
            (
                ("Enabled", self.stream_coach_enabled_status_var),
                ("Mode", self.stream_coach_platform_status_var),
                ("Dry-run / apply", self.stream_coach_dry_run_status_var),
                ("Alive", self.stream_coach_alive_status_var),
                ("Last message", self.stream_coach_last_message_var),
                ("Parsed", self.stream_coach_last_parsed_command_var),
                ("Applied", self.stream_coach_last_applied_command_var),
                ("Accepted / rejected", self.stream_coach_accept_reject_var),
                ("Reject reason", self.stream_coach_last_reject_reason_var),
                ("Pending", self.stream_coach_pending_count_var),
                ("Top command", self.stream_coach_top_command_var),
                ("Selected command", self.stream_coach_last_selected_command_var),
                ("Selected action", self.stream_coach_last_action_var),
                ("Rejected count", self.stream_coach_rejected_count_var),
                ("Vote count", self.stream_coach_last_vote_count_var),
                ("Overrides", self.stream_coach_override_count_var),
                ("Matches", self.stream_coach_match_count_var),
                ("Reward total", self.stream_coach_reward_total_var),
            )
        ):
            self._add_labeled_value(stream_live, row, label, variable, width=28)

        fusion_live = ttk.LabelFrame(live, text="Fusion")
        fusion_live.grid(row=0, column=2, sticky="nsew", padx=(2, 4), pady=4)
        fusion_live.columnconfigure(1, weight=1)
        for row, (label, variable) in enumerate(
            (
                ("Available", self.fusion_bridge_available_var),
                ("Enabled", self.fusion_bridge_enabled_status_var),
                ("Last command", self.fusion_last_command_var),
                ("Last result", self.fusion_last_result_var),
                ("Attempts", self.fusion_attempt_count_var),
                ("Successes", self.fusion_success_count_var),
                ("Failures", self.fusion_failure_count_var),
                ("Rejected", self.fusion_rejected_count_var),
            )
        ):
            self._add_labeled_value(fusion_live, row, label, variable, width=28)

        for variable in (
            self.structured_command_type_var,
            self.structured_row_var,
            self.structured_col_var,
            self.structured_seed_slot_var,
            self.structured_custom_text_var,
        ):
            variable.trace_add("write", self._update_structured_command_preview)
        self._update_structured_command_preview()

    def _set_structured_command_type(self, command: str) -> None:
        self.structured_command_type_var.set(command)

    # Serialization and validation stay outside Tk widget construction so
    # future stream/chat adapters can share the same typed command schema.
    def _structured_command_object(self) -> AssistedCoachCommand:
        command_type = AssistedCommandType(self.structured_command_type_var.get().strip().upper())
        needs_position = command_type in AssistedCommandValidator.POSITION_COMMANDS
        needs_seed = command_type in {AssistedCommandType.PLANT, AssistedCommandType.FUSE}

        def optional_int(value: str, required: bool) -> Optional[int]:
            text = value.strip()
            if not required:
                return None
            return int(text)

        if needs_seed:
            target = self.structured_seed_slot_var.get().strip()
        elif command_type == AssistedCommandType.BOOST:
            target = self.structured_custom_text_var.get().strip()
        else:
            target = ""
        return AssistedCoachCommand(
            command_type=command_type,
            source=self.assisted_command_source_var.get().strip() or "dashboard",
            user=self.assisted_command_user_var.get().strip() or "local",
            row=optional_int(self.structured_row_var.get(), needs_position),
            col=optional_int(self.structured_col_var.get(), needs_position),
            target=target,
        )

    def _build_structured_coach_command(self) -> str:
        try:
            return self._structured_command_object().display_text()
        except (ValueError, TypeError):
            return ""

    def _update_structured_command_preview(self, *_args: Any) -> None:
        try:
            command_type = AssistedCommandType(self.structured_command_type_var.get().strip().upper())
        except ValueError:
            command_type = AssistedCommandType.PLANT
        requires_position = command_type in AssistedCommandValidator.POSITION_COMMANDS
        requires_seed = command_type in {AssistedCommandType.PLANT, AssistedCommandType.FUSE}
        uses_target = command_type == AssistedCommandType.BOOST
        for widget, enabled in (
            (self.structured_row_widget, requires_position),
            (self.structured_col_widget, requires_position),
            (self.structured_seed_widget, requires_seed),
            (self.structured_custom_widget, uses_target),
        ):
            if widget is not None:
                widget.configure(state="normal" if enabled else "disabled")
        try:
            command = self._structured_command_object()
            self.structured_preview_var.set(command.display_text())
            if command.command_type == AssistedCommandType.FUSE:
                self.fusion_command_feedback_var.set("Fusion will be revalidated against bridge availability and board legality at execution.")
        except (ValueError, TypeError):
            self.structured_preview_var.set("Invalid numeric field")

    def _structured_coach_validation_error(self) -> str:
        try:
            result = AssistedCommandValidator.validate(self._structured_command_object())
        except (ValueError, TypeError) as exc:
            return f"Invalid command field: {exc}"
        return "" if result.valid else result.reason

    def _structured_parser_command(self) -> str:
        try:
            result = AssistedCommandValidator.validate(self._structured_command_object())
        except (ValueError, TypeError):
            return ""
        return result.backend_command

    def send_structured_coach_command(self) -> None:
        """Validate and submit a command to the moderation queue."""
        error = self._structured_coach_validation_error()
        if error:
            self.coach_queue_status_var.set(f"Queue error: {error}")
            self._append_log(f"ERROR: assisted command rejected: {error}\n")
            return
        command = self._structured_command_object()
        self.assisted_command_queue.submit(command)
        self.coach_queue_status_var.set(f"Queued {command.command_id}: {command.display_text()}")
        self._append_log(f"[coach] queued {command.command_id} from {command.source}/{command.user}: {command.display_text()}\n")
        self._log_dashboard_intervention(command, AssistedCommandStatus.PENDING.value)
        self._refresh_assisted_queue()

    def copy_structured_command_to_raw(self) -> None:
        command = self._build_structured_coach_command()
        self.human_coach_command_input_var.set(command)
        self._structured_raw_copy_value = command

    def _selected_assisted_command(self) -> Optional[AssistedCoachCommand]:
        if self.assisted_queue_tree is None:
            return None
        selection = self.assisted_queue_tree.selection()
        if not selection:
            self.coach_queue_status_var.set("Queue error: select a command first")
            return None
        return self.assisted_command_queue.get(str(selection[0]))

    def _refresh_assisted_queue(self) -> None:
        if self.assisted_queue_tree is not None:
            selected = self.assisted_queue_tree.selection()
            for item_id in self.assisted_queue_tree.get_children():
                self.assisted_queue_tree.delete(item_id)
            for row in queue_rows(self.assisted_command_queue.all()):
                self.assisted_queue_tree.insert("", "end", iid=row[0], values=row[1:])
            if selected and self.assisted_queue_tree.exists(selected[0]):
                self.assisted_queue_tree.selection_set(selected[0])
        counts = self.assisted_command_queue.counts()
        self.assisted_queue_summary_var.set(" ".join(f"{name}={counts[name]}" for name in ("pending", "approved", "rejected", "executed")))

    def approve_assisted_command(self) -> None:
        command = self._selected_assisted_command()
        if command is None:
            return
        self.assisted_command_queue.set_status(command.command_id, AssistedCommandStatus.APPROVED, "approved locally")
        self.coach_queue_status_var.set(f"Approved {command.command_id}")
        self._log_dashboard_intervention(command, AssistedCommandStatus.APPROVED.value)
        self._refresh_assisted_queue()

    def reject_assisted_command(self) -> None:
        command = self._selected_assisted_command()
        if command is None:
            return
        self.assisted_command_queue.set_status(command.command_id, AssistedCommandStatus.REJECTED, "rejected locally")
        self.coach_queue_status_var.set(f"Rejected {command.command_id}")
        self._log_dashboard_intervention(command, AssistedCommandStatus.REJECTED.value)
        self._refresh_assisted_queue()

    def modify_assisted_command(self) -> None:
        command = self._selected_assisted_command()
        if command is None:
            return
        error = self._structured_coach_validation_error()
        if error:
            self.coach_queue_status_var.set(f"Modify error: {error}")
            self._append_log(f"ERROR: assisted command modify failed: {error}\n")
            return
        self.assisted_command_queue.modify(command.command_id, self._structured_command_object())
        self.coach_queue_status_var.set(f"Modified {command.command_id}; approval reset")
        self._refresh_assisted_queue()

    def execute_assisted_command(self) -> None:
        command = self._selected_assisted_command()
        if command is None:
            return
        if command.status != AssistedCommandStatus.APPROVED:
            self.coach_queue_status_var.set("Execute error: command must be approved first")
            self._append_log("ERROR: assisted command must be approved before execution\n")
            return
        validation = AssistedCommandValidator.validate(command)
        if not validation.backend_supported:
            reason = f"{command.command_type.value} has no safe game-loop adapter yet"
            self.assisted_command_queue.set_status(command.command_id, AssistedCommandStatus.REJECTED, reason)
            self.coach_queue_status_var.set(f"Execute rejected: {reason}")
            self._append_log(f"ERROR: {reason}\n")
            self._log_dashboard_intervention(command, AssistedCommandStatus.REJECTED.value, note=reason)
            self._refresh_assisted_queue()
            return
        queued = self._queue_coach_command(
            command.display_text(),
            source=f"assisted:{command.source}:{command.user}",
            parser_command=validation.backend_command,
        )
        if not queued:
            return
        self.assisted_command_queue.set_status(command.command_id, AssistedCommandStatus.EXECUTED, "queued for runtime validation")
        self._log_dashboard_intervention(command, AssistedCommandStatus.EXECUTED.value)
        if command.command_type == AssistedCommandType.FUSE:
            self.fusion_command_feedback_var.set("Fusion queued; final legality/result will appear in Live Coach Diagnostics.")
        self._refresh_assisted_queue()

    def _log_dashboard_intervention(
        self,
        command: AssistedCoachCommand,
        status: str,
        *,
        note: str = "",
    ) -> None:
        try:
            path = self._resolve_text_path(self.intervention_log_path_var.get() or str(DEFAULT_INTERVENTION_LOG_PATH))
            self.intervention_logger = InterventionJSONLLogger(path)
            live = self.last_good_status if isinstance(self.last_good_status, dict) else {}
            self.intervention_logger.log(
                run_id=str(live.get("run_id") or live.get("run_name") or self.active_process_name or "dashboard"),
                episode_id=int(live.get("episode") or live.get("episode_id") or 0),
                step=int(live.get("step") or live.get("global_step") or 0),
                mode=str(live.get("mode") or "dashboard"),
                model_action=live.get("last_action"),
                human_command=command.to_dict(),
                command_source=f"{command.source}/{command.user}",
                status=status,
                board_state_summary=live.get("board_state_summary") if isinstance(live.get("board_state_summary"), dict) else {},
                metadata={"execution_mode": self.assisted_execution_mode_var.get(), "note": note},
            )
        except (OSError, TypeError, ValueError) as exc:
            self._append_log(f"ERROR: intervention log write failed: {exc}\n")

    def _build_fusion_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=4)
        parent.columnconfigure(1, weight=2)
        parent.rowconfigure(0, weight=1)

        lawn = ttk.LabelFrame(parent, text="Fusion Lawn · click a tile to assign the selected seed slot")
        lawn.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)
        for column in range(LAWN_COLS):
            lawn.columnconfigure(column, weight=1)
        for row in range(LAWN_ROWS):
            lawn.rowconfigure(row, weight=1)
            for column in range(LAWN_COLS):
                button = ttk.Button(
                    lawn,
                    text=f"r{row} c{column}\n—",
                    command=lambda r=row, c=column: self._assign_fusion_tile(r, c),
                )
                button.grid(row=row, column=column, sticky="nsew", padx=2, pady=2)
                self.fusion_tile_buttons[(row, column)] = button

        controls = ttk.LabelFrame(parent, text="Planner Controls")
        controls.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=8)
        controls.columnconfigure(1, weight=1)
        ttk.Label(controls, text="Selected seed packet").grid(row=0, column=0, sticky="w", padx=6, pady=3)
        ttk.Combobox(
            controls,
            textvariable=self.fusion_selected_seed_var,
            values=tuple(str(index) for index in range(SEED_PACKET_SLOTS)),
            state="normal",
            width=20,
        ).grid(row=0, column=1, sticky="ew", padx=6, pady=3)
        ttk.Label(controls, textvariable=self.fusion_selected_tile_var).grid(
            row=1, column=0, columnspan=2, sticky="w", padx=6, pady=3
        )
        ttk.Button(controls, text="Clear tile", command=self._clear_fusion_tile).grid(
            row=2, column=0, sticky="ew", padx=6, pady=3
        )
        ttk.Button(controls, text="Clear board", command=self._clear_fusion_board).grid(
            row=2, column=1, sticky="ew", padx=6, pady=3
        )
        ttk.Button(controls, text="Preview fusion command", command=self._refresh_fusion_preview).grid(
            row=3, column=0, columnspan=2, sticky="ew", padx=6, pady=3
        )
        ttk.Button(controls, text="Queue fusion command", command=self.queue_fusion_commands).grid(
            row=4, column=0, columnspan=2, sticky="ew", padx=6, pady=3
        )
        ttk.Checkbutton(
            controls,
            text="Enable fusion bridge",
            variable=self.fusion_bridge_enabled_var,
        ).grid(row=5, column=0, columnspan=2, sticky="w", padx=6, pady=3)

        preview = ttk.LabelFrame(controls, text="Command Preview")
        preview.grid(row=6, column=0, columnspan=2, sticky="nsew", padx=6, pady=6)
        preview.columnconfigure(0, weight=1)
        ttk.Label(
            preview,
            textvariable=self.fusion_command_preview_var,
            justify="left",
            wraplength=330,
            font=("Consolas", 9),
        ).grid(row=0, column=0, sticky="ew", padx=6, pady=6)

        diagnostics = ttk.LabelFrame(controls, text="Fusion Diagnostics")
        diagnostics.grid(row=7, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 6))
        diagnostics.columnconfigure(1, weight=1)
        self._add_labeled_value(diagnostics, 0, "Bridge available", self.fusion_bridge_available_var, width=28)
        self._add_labeled_value(diagnostics, 1, "Last command", self.fusion_last_command_var, width=28)
        self._add_labeled_value(diagnostics, 2, "Last result", self.fusion_last_result_var, width=28)
        self._add_labeled_value(diagnostics, 3, "Attempts", self.fusion_attempt_count_var, width=28)
        self._add_labeled_value(diagnostics, 4, "Successes", self.fusion_success_count_var, width=28)
        self._add_labeled_value(diagnostics, 5, "Failures", self.fusion_failure_count_var, width=28)
        self._add_labeled_value(diagnostics, 6, "Rejected", self.fusion_rejected_count_var, width=28)

    def _build_runs_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        paths = ttk.LabelFrame(parent, text="Runs and Models")
        paths.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 3))
        paths.columnconfigure(1, weight=1)
        self._add_labeled_entry(paths, 0, 0, "Model .zip", self.model_path_var, width=56)
        self._add_labeled_entry(paths, 1, 0, "Run dir", self.run_dir_var, width=56)

        actions = ttk.Frame(parent)
        actions.grid(row=1, column=0, sticky="ew", padx=8, pady=3)
        ttk.Button(actions, text="Browse model.zip", command=self.browse_model).grid(row=0, column=0, sticky="w", padx=(0, 5))
        ttk.Button(actions, text="Browse run folder", command=self.browse_run_folder).grid(row=0, column=1, sticky="w", padx=(0, 5))
        ttk.Button(actions, text="Open run folder", command=self.open_run_folder).grid(row=0, column=2, sticky="w", padx=(0, 5))
        ttk.Button(actions, text="Refresh Models", command=self.refresh_models).grid(row=0, column=3, sticky="w")

        charts = ttk.LabelFrame(parent, text="Analyze / Charts")
        charts.grid(row=2, column=0, sticky="ew", padx=8, pady=3)
        ttk.Button(charts, text="Analyze Selected Run", command=self.analyze_selected_run).grid(row=0, column=0, sticky="w", padx=6, pady=6)
        ttk.Button(charts, text="Show Charts", command=self.show_charts).grid(row=0, column=1, sticky="w", padx=6, pady=6)

    def _toggle_train_advanced(self) -> None:
        frame = self.train_advanced_frame
        if frame is None:
            return
        expanded = not self.train_advanced_expanded_var.get()
        self.train_advanced_expanded_var.set(expanded)
        if expanded:
            frame.grid()
        else:
            frame.grid_remove()

    def _toggle_diagnostic_panels(self) -> None:
        visible_titles = [
            title for title, variable in self.diagnostic_visibility_vars.items() if variable.get()
        ]
        for frame in self.diagnostic_panel_frames.values():
            frame.grid_remove()
        for index, title in enumerate(visible_titles):
            frame = self.diagnostic_panel_frames.get(title)
            if frame is None:
                continue
            frame.grid(
                row=index // 3,
                column=index % 3,
                sticky="nsew",
                padx=4,
                pady=4,
            )

    def _assign_fusion_tile(self, row: int, column: int) -> None:
        seed = self.fusion_selected_seed_var.get().strip()
        if not seed:
            self._append_log("ERROR: Select a seed packet before assigning a Fusion tile.\n")
            return
        self.fusion_selected_tile = (row, column)
        self.fusion_grid[(row, column)] = seed
        self.fusion_selected_tile_var.set(f"Selected tile: row {row}, column {column}")
        button = self.fusion_tile_buttons.get((row, column))
        if button is not None:
            button.configure(text=f"r{row} c{column}\nslot {seed}")
        self._refresh_fusion_preview()

    def _clear_fusion_tile(self) -> None:
        tile = self.fusion_selected_tile
        if tile is None:
            return
        self.fusion_grid.pop(tile, None)
        button = self.fusion_tile_buttons.get(tile)
        if button is not None:
            button.configure(text=f"r{tile[0]} c{tile[1]}\n—")
        self._refresh_fusion_preview()

    def _clear_fusion_board(self) -> None:
        self.fusion_grid.clear()
        for (row, column), button in self.fusion_tile_buttons.items():
            button.configure(text=f"r{row} c{column}\n—")
        self.fusion_selected_tile = None
        self.fusion_selected_tile_var.set("Selected tile: none")
        self._refresh_fusion_preview()

    def _build_fusion_command_from_grid(self) -> str:
        """Keep planner serialization isolated from the Fusion tab layout."""
        commands = [
            f"fuse {seed} {row} {column}"
            for (row, column), seed in sorted(self.fusion_grid.items())
        ]
        return "\n".join(commands)

    def _refresh_fusion_preview(self) -> None:
        command = self._build_fusion_command_from_grid()
        self.fusion_command_preview_var.set(command or "Select a seed slot, then click lawn tiles.")

    def queue_fusion_commands(self) -> None:
        command_text = self._build_fusion_command_from_grid()
        if not command_text:
            self._append_log("ERROR: Fusion board is empty; nothing was queued.\n")
            return
        try:
            queue_path = self._coach_command_queue_path()
            count = CoachCommandSink(queue_path).append(
                CoachQueueCommand(command=command, source="gui_fusion")
                for command in command_text.splitlines()
            )
        except (OSError, ValueError) as exc:
            self.coach_queue_status_var.set(f"Queue error: fusion write failed ({exc})")
            self._append_log(f"ERROR: Failed to queue Fusion commands: {exc}\n")
            return
        self.coach_queue_status_var.set(f"Queued {count} Fusion command(s) at {time.strftime('%H:%M:%S')}")
        self._append_log(f"Queued {count} Fusion command(s) to {queue_path}.\n")

    def _add_labeled_entry(
        self,
        parent: ttk.Frame,
        row: int,
        column: int,
        label: str,
        variable: tk.StringVar,
        width: int = 16,
        columnspan: int = 1,
        tooltip: str = "",
    ) -> ttk.Entry:
        label_widget = ttk.Label(parent, text=f"{label} ⓘ" if tooltip else label)
        label_widget.grid(row=row, column=column, sticky="w", padx=6, pady=2)
        if tooltip:
            _Tooltip(label_widget, tooltip)
        entry = ttk.Entry(parent, textvariable=variable, width=width)
        entry.grid(row=row, column=column + 1, columnspan=columnspan, sticky="ew", padx=6, pady=2)
        return entry

    def _add_labeled_value(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        width: int = 64,
    ) -> ttk.Label:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=2)
        value = ttk.Label(parent, textvariable=variable, anchor="w", width=width)
        value.grid(row=row, column=1, sticky="ew", padx=6, pady=2)
        return value

    def _add_launch_button(self, parent: ttk.Frame, text: str, command: Any) -> ttk.Button:
        button = ttk.Button(parent, text=text, command=command)
        self.launch_buttons.append(button)
        return button

    def _add_stop_button(self, parent: ttk.Frame, text: str) -> ttk.Button:
        button = ttk.Button(parent, text=text, command=self.stop_active_process, state="disabled")
        self.stop_buttons.append(button)
        return button

    def _preview_box(self, parent: ttk.Frame, height: int) -> scrolledtext.ScrolledText:
        box = scrolledtext.ScrolledText(parent, height=height, wrap="word", font=("Consolas", 9))
        box.grid(row=0, column=0, sticky="nsew")
        box.configure(state="disabled")
        return box

    def _make_scrollable_container(self, parent: ttk.Frame) -> ttk.Frame:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        container = ttk.Frame(parent)
        container.grid(row=0, column=0, sticky="nsew")
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

        canvas = tk.Canvas(container, highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        content = ttk.Frame(canvas)
        window_id = canvas.create_window((0, 0), window=content, anchor="nw")

        def _on_content_configure(_event: Any) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event: Any) -> None:
            canvas.itemconfigure(window_id, width=event.width)

        def _on_mousewheel(event: Any) -> str:
            delta = int(getattr(event, "delta", 0) or 0)
            if delta:
                canvas.yview_scroll(int(-delta / 120), "units")
            elif int(getattr(event, "num", 0) or 0) == 4:
                canvas.yview_scroll(-1, "units")
            elif int(getattr(event, "num", 0) or 0) == 5:
                canvas.yview_scroll(1, "units")
            return "break"

        def _bind_wheel(_event: Any) -> None:
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
            canvas.bind_all("<Button-4>", _on_mousewheel)
            canvas.bind_all("<Button-5>", _on_mousewheel)

        def _unbind_wheel(_event: Any) -> None:
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        content.bind("<Configure>", _on_content_configure)
        canvas.bind("<Configure>", _on_canvas_configure)
        canvas.bind("<Enter>", _bind_wheel)
        canvas.bind("<Leave>", _unbind_wheel)
        content.bind("<Enter>", _bind_wheel)
        content.bind("<Leave>", _unbind_wheel)
        return content

    def _bind_command_preview_updates(self) -> None:
        variables: List[Any] = [
            self.total_timesteps_var,
            self.max_steps_var,
            self.step_seconds_var,
            self.game_speed_var,
            self.seed_list_var,
            self.model_path_var,
            self.episodes_var,
            self.run_dir_var,
            self.run_name_var,
            self.start_sun_var,
            self.board_timeout_var,
            self.gameplay_ready_timeout_var,
            self.checkpoint_freq_var,
            self.fusion_policy_var,
            self.quick_wait_var,
            self.wait_gameplay_ready_var,
            self.auto_select_seeds_var,
            self.debug_perf_var,
            self.fast_only_var,
            self.adventure_model_path_var,
            self.adventure_seed_list_var,
            self.adventure_plant_types_var,
            self.adventure_episodes_var,
            self.adventure_max_levels_var,
            self.adventure_max_attempts_var,
            self.adventure_advance_wins_var,
            self.adventure_game_speed_var,
            self.adventure_step_seconds_var,
            self.adventure_soft_max_steps_var,
            self.adventure_hard_max_steps_var,
            self.adventure_board_timeout_var,
            self.adventure_eval_var,
            self.adventure_advance_on_wins_var,
            self.adventure_auto_select_seeds_var,
            self.adventure_wait_gameplay_ready_var,
            self.adventure_quick_wait_var,
            self.adventure_final_wave_extension_var,
            self.adventure_tactical_masks_var,
            self.adventure_wallnut_mask_var,
            self.adventure_cherrybomb_mask_var,
            self.adventure_fusion_policy_var,
            self.generalist_total_timesteps_var,
            self.generalist_checkpoint_freq_var,
            self.generalist_initial_loadout_var,
            self.generalist_max_seed_slots_var,
            self.generalist_start_level_var,
            self.generalist_max_levels_var,
            self.generalist_max_attempts_var,
            self.generalist_game_speed_var,
            self.generalist_step_seconds_var,
            self.generalist_board_timeout_var,
            self.generalist_soft_max_steps_var,
            self.generalist_hard_max_steps_var,
            self.generalist_frontier_prob_var,
            self.generalist_recent_prob_var,
            self.generalist_maintenance_prob_var,
            self.generalist_frontier_win_streak_required_var,
            self.generalist_unlock_delay_var,
            self.generalist_new_plant_prob_var,
            self.generalist_run_dir_var,
            self.generalist_resume_model_path_var,
            self.generalist_eval_model_path_var,
            self.generalist_eval_episodes_var,
            self.generalist_unlock_curriculum_var,
            self.generalist_replay_cleared_var,
            self.generalist_final_wave_extension_var,
            self.generalist_wait_gameplay_ready_var,
            self.generalist_quick_wait_var,
            self.generalist_tactical_masks_var,
            self.generalist_wallnut_mask_var,
            self.generalist_cherrybomb_mask_var,
            self.generalist_fusion_action_mask_train_var,
            self.generalist_fusion_action_mask_eval_var,
            self.generalist_curriculum_var,
            self.level3_mode_var,
            self.level3_target_level_var,
            self.level3_model_path_var,
            self.level3_total_timesteps_var,
            self.level3_episodes_var,
            self.level3_max_steps_var,
            self.level3_step_seconds_var,
            self.level3_game_speed_var,
            self.level3_board_timeout_var,
            self.level3_seed_list_var,
            self.level3_plant_types_var,
            self.level3_tactical_masks_var,
            self.level3_wallnut_mask_var,
            self.level3_cherrybomb_mask_var,
            self.human_coach_enabled_var,
            self.human_coach_reward_var,
            self.human_coach_bonus_var,
            self.human_coach_match_bonus_var,
            self.human_coach_override_penalty_var,
            self.human_coach_fusion_reward_var,
            self.human_coach_tactical_reward_var,
            self.human_coach_log_path_var,
            self.human_coach_command_path_var,
            self.stream_coach_enabled_var,
            self.stream_coach_platform_var,
            self.stream_coach_window_sec_var,
            self.stream_coach_min_votes_var,
            self.stream_coach_max_actions_per_minute_var,
            self.stream_coach_reward_var,
            self.stream_coach_dry_run_var,
            self.stream_coach_log_path_var,
            self.stream_coach_mock_script_var,
            self.coach_allow_fusion_planning_var,
            self.fusion_bridge_enabled_var,
            self.assisted_execution_mode_var,
            self.train_lab_mode_var,
            self.eval_lab_mode_var,
            self.intervention_log_path_var,
        ]
        for variable in variables:
            variable.trace_add("write", lambda *_args: self._update_command_previews())
        for variable in (
            self.human_coach_enabled_var,
            self.stream_coach_enabled_var,
            self.train_lab_mode_var,
            self.eval_lab_mode_var,
            self.assisted_execution_mode_var,
        ):
            variable.trace_add("write", self._update_command_enablement)
        self._update_command_enablement()

    def _update_command_enablement(self, *_args: Any) -> None:
        train_lab = self.train_lab_mode_var.get().strip().lower()
        eval_lab = self.eval_lab_mode_var.get().strip().lower()
        coach_enabled = bool(self.human_coach_enabled_var.get()) or train_lab in {"assisted", "fusion"} or eval_lab in {"assisted", "fusion"}
        viewer_enabled = bool(self.stream_coach_enabled_var.get())
        self.command_enablement_var.set(
            f"Coach commands {'enabled' if coach_enabled else 'off'} | "
            f"Viewer commands {'enabled' if viewer_enabled else 'off'} | "
            f"mode={self.assisted_execution_mode_var.get()}"
        )

    def _update_command_previews(self) -> None:
        if self.train_preview is not None:
            resume_model_path = self.generalist_resume_model_path_var.get().strip()
            lines = [
                "Adventure Generalist Train:",
                self._command_text(self._build_adventure_generalist_command()),
                "",
                f"Model family: {ADVENTURE_GENERALIST_MODEL_FAMILY}",
                "Resume mode: " + (
                    self._path_for_display(resume_model_path)
                    if resume_model_path
                    else "fresh initialization (resume model is blank)"
                ),
            ]
            self._set_text_widget(self.train_preview, "\n".join(lines))
        if self.eval_preview is not None:
            lines = [
                "Adventure Generalist Eval:",
            ]
            if self.generalist_eval_model_path_var.get().strip():
                lines.append(self._command_text(self._build_adventure_generalist_eval_command()))
            else:
                lines.append("model_path is required for Adventure Generalist evaluation")
            lines.extend(
                [
                    "",
                    f"Compatibility: family={ADVENTURE_GENERALIST_MODEL_FAMILY}, action_count=701, max_seed_slots=14",
                ]
            )
            self._set_text_widget(self.eval_preview, "\n".join(lines))
        if self.adventure_preview is not None:
            lines = [
                "Start Adventure Eval:",
                self._command_text(self._build_adventure_command()),
            ]
            if not self.adventure_eval_var.get():
                lines.extend(["", "Note: --adventure-eval is required for this launch button."])
            self._set_text_widget(self.adventure_preview, "\n".join(lines))
        if self.generalist_preview is not None:
            resume_model_path = self.generalist_resume_model_path_var.get().strip()
            lines = [
                "Adventure Generalist Train:",
                self._command_text(self._build_adventure_generalist_command()),
                "",
                "Adventure Generalist Eval:",
            ]
            if self.generalist_eval_model_path_var.get().strip():
                lines.append(self._command_text(self._build_adventure_generalist_eval_command()))
            else:
                lines.append("model_path is required for Adventure Generalist eval")
            if resume_model_path:
                lines.extend(["", f"Resume training model zip: {self._path_for_display(resume_model_path)}"])
            else:
                lines.extend(["", "Resume training model zip: (blank -> fresh model initialization)"])
            lines.extend(
                [
                    "",
                    f"Model family: {ADVENTURE_GENERALIST_MODEL_FAMILY}",
                    "Resume supported via --resume-model-path; eval uses the separate Eval model .zip field.",
                ]
            )
            self._set_text_widget(self.generalist_preview, "\n".join(lines))
        if self.level3_preview is not None:
            lines = [
                "Start Level 3 Specialist:",
                self._command_text(self._build_level3_command()),
            ]
            self._set_text_widget(self.level3_preview, "\n".join(lines))

    def _set_text_widget(self, widget: tk.Text, content: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", content)
        widget.configure(state="disabled")

    def _coach_command_queue_path(self) -> Path:
        raw_path = self.human_coach_command_path_var.get().strip() or str(DEFAULT_COACH_COMMAND_QUEUE_PATH)
        return self._resolve_text_path(raw_path)

    def _queue_coach_command(
        self,
        raw_command: str,
        source: str = "gui",
        *,
        parser_command: str = "",
    ) -> bool:
        """Append one command to the shared JSONL coach queue."""
        command = str(raw_command or "").strip()
        if not command:
            self.coach_queue_status_var.set("Queue error: command input is empty")
            self._append_log("ERROR: Coach command input is empty.\n")
            return False
        try:
            queue_path = self._coach_command_queue_path()
        except (OSError, ValueError) as exc:
            self.coach_queue_status_var.set(f"Queue error: invalid queue path ({exc})")
            self._append_log(f"ERROR: Invalid coach command queue path: {exc}\n")
            return False

        try:
            CoachCommandSink(queue_path).append(
                [
                    CoachQueueCommand(
                        command=command,
                        source=str(source or "gui"),
                        parser_command=str(parser_command or ""),
                    )
                ]
            )
        except (OSError, ValueError) as exc:
            self.coach_queue_status_var.set(f"Queue error: write failed ({exc})")
            self._append_log(f"ERROR: Failed to append coach command to queue: {exc}\n")
            return False

        self.coach_queue_status_var.set(
            f"Queued command at {time.strftime('%H:%M:%S')} (pending environment parse)"
        )
        self._append_log(f"Queued coach command to {queue_path}: {command}\n")
        return True

    def send_human_coach_command(self) -> None:
        raw_command = self.human_coach_command_input_var.get().strip()
        parser_command = ""
        if raw_command and raw_command == getattr(self, "_structured_raw_copy_value", ""):
            parser_command = self._structured_parser_command()
        if self._queue_coach_command(raw_command, source="gui", parser_command=parser_command):
            self.human_coach_command_input_var.set("")
            self._structured_raw_copy_value = ""

    def start_training(self) -> None:
        if self.fast_only_var.get():
            self._append_log("Note: --fast-only omitted from training; train_ppo.py does not support it.\n")
        self.launch_process("Start Training", self._build_train_command(resume=False))

    def resume_training(self) -> None:
        raw_model_path = self.generalist_resume_model_path_var.get().strip()
        if not raw_model_path:
            self._append_log("ERROR: Adventure Generalist resume requires a resume model .zip.\n")
            return
        model_path = self._resolve_text_path(raw_model_path)
        if not model_path.exists():
            self._append_log(f"ERROR: Resume model does not exist: {model_path}\n")
            return
        self.start_adventure_generalist_train()

    def run_eval(self) -> None:
        self._launch_eval("Run Eval", self._build_eval_command())

    def run_25_episode_eval(self) -> None:
        self._launch_eval("Run 25-Episode Eval", self._build_eval_command(episodes_override="25"))

    def start_level3_specialist(self) -> None:
        if self.level3_mode_var.get().strip().lower() == "eval":
            raw_model_path = self.level3_model_path_var.get().strip()
            if not raw_model_path:
                self._append_log("ERROR: Level 3 eval requires model_path.\n")
                return
            model_path = self._resolve_text_path(raw_model_path)
            if not model_path.exists():
                self._append_log(f"ERROR: Level 3 model does not exist: {model_path}\n")
                return
        self.launch_process("Start Level 3", self._build_level3_command())

    def _launch_eval(self, name: str, command: List[str]) -> None:
        raw_model_path = self.model_path_var.get().strip()
        if not raw_model_path:
            self._append_log("ERROR: Eval requires model_path.\n")
            return
        model_path = self._resolve_text_path(raw_model_path)
        if not model_path.exists():
            self._append_log(f"ERROR: Eval model does not exist: {model_path}\n")
            return
        if self.fast_only_var.get():
            self._append_log("Note: --fast-only omitted from eval; train_ppo.py does not support it.\n")
        self.launch_process(name, command)

    def start_adventure_eval(self) -> None:
        if not self.adventure_eval_var.get():
            self._append_log("ERROR: Adventure launch requires --adventure-eval.\n")
            return
        raw_model_path = self.adventure_model_path_var.get().strip()
        if not raw_model_path:
            self._append_log("ERROR: Adventure eval requires model_path.\n")
            return
        model_path = self._resolve_text_path(raw_model_path)
        if not model_path.exists():
            self._append_log(f"ERROR: Adventure model does not exist: {model_path}\n")
            return
        if not self.adventure_advance_on_wins_var.get():
            self._append_log("Note: --advance-on-wins omitted; train_ppo.py defaults to 1.\n")
        self.launch_process("Start Adventure Eval", self._build_adventure_command())

    def start_adventure_generalist_train(self) -> None:
        if self.generalist_initial_loadout_var.get().strip() != ADVENTURE_GENERALIST_INITIAL_LOADOUT:
            self._append_log(
                "ERROR: Adventure Generalist training requires initial loadout "
                f"{ADVENTURE_GENERALIST_INITIAL_LOADOUT}.\n"
            )
            return
        if self.generalist_max_seed_slots_var.get().strip() != "14":
            self._append_log("ERROR: Adventure Generalist training requires max seed slots = 14.\n")
            return
        raw_resume_path = self.generalist_resume_model_path_var.get().strip()
        if raw_resume_path:
            resume_path = self._resolve_text_path(raw_resume_path)
            if not resume_path.exists():
                self._append_log(f"ERROR: Adventure Generalist resume model does not exist: {resume_path}\n")
                return
            self._append_log(f"Launching Adventure Generalist training (resume): {resume_path}\n")
        else:
            self._append_log("Launching Adventure Generalist training (fresh model initialization)...\n")
        self.launch_process("Start Adventure Generalist Train", self._build_adventure_generalist_command())

    def start_adventure_generalist_eval(self) -> None:
        if self.generalist_initial_loadout_var.get().strip() != ADVENTURE_GENERALIST_INITIAL_LOADOUT:
            self._append_log(
                "ERROR: Adventure Generalist v1 eval requires initial loadout "
                f"{ADVENTURE_GENERALIST_INITIAL_LOADOUT}.\n"
            )
            return
        if self.generalist_max_seed_slots_var.get().strip() != "14":
            self._append_log("ERROR: Adventure Generalist v1 eval requires max seed slots = 14.\n")
            return
        raw_model_path = self.generalist_eval_model_path_var.get().strip()
        if not raw_model_path:
            self._append_log("ERROR: Adventure Generalist eval requires model_path.\n")
            return
        model_path = self._resolve_text_path(raw_model_path)
        if not model_path.exists():
            self._append_log(f"ERROR: Adventure Generalist eval model does not exist: {model_path}\n")
            return
        self.launch_process("Start Adventure Generalist Eval", self._build_adventure_generalist_eval_command())

    def auto_reset_test(self) -> None:
        command = self._script_command("pvzrl_env.py", "--terminal-auto-reset-test")
        self._add_optional_value(command, "--episodes", self.episodes_var.get())
        self._add_optional_value(command, "--max-steps", self.max_steps_var.get())
        command.extend(["--policy", "teacher"])
        self._add_optional_value(command, "--step-seconds", self.step_seconds_var.get())
        self._add_optional_value(command, "--game-speed", self.game_speed_var.get())
        self._append_env_options(command)
        self._add_optional_value(command, "--seed-list", self.seed_list_var.get())
        self.launch_process("Auto reset test", command)

    def seed_selection_test(self) -> None:
        command = self._script_command("pvzrl_env.py", "--auto-select-seeds-test")
        self._add_optional_value(command, "--seed-list", self.seed_list_var.get())
        self._add_optional_value(command, "--episodes", self.episodes_var.get())
        self._append_env_options(command)
        self.launch_process("Seed selection test", command)

    def cooldown_test(self) -> None:
        command = self._script_command("pvzrl_env.py", "--cooldown-test")
        self._add_optional_value(command, "--step-seconds", self.step_seconds_var.get())
        self._add_optional_value(command, "--game-speed", self.game_speed_var.get())
        self._append_env_options(command)
        self._add_optional_value(command, "--seed-list", self.seed_list_var.get())
        self.launch_process("Cooldown test", command)

    def bridge_perf_diagnostic(self) -> None:
        command = self._script_command(
            "pvzrl_env.py",
            "--validate-reliability",
            "--smoke-runs",
            "3",
            "--max-steps",
            "100",
        )
        self._append_env_options(command, force_debug_perf=True)
        self.launch_process("Bridge/perf diagnostic", command)

    def _append_env_options(self, command: List[str], force_debug_perf: bool = False) -> None:
        self._add_enabled_flag(command, "--quick-wait", self.quick_wait_var.get())
        self._add_enabled_flag(command, "--wait-gameplay-ready", self.wait_gameplay_ready_var.get())
        self._add_enabled_flag(command, "--auto-select-seeds", self.auto_select_seeds_var.get())
        self._add_enabled_flag(command, "--debug-perf", force_debug_perf or self.debug_perf_var.get())
        self._add_enabled_flag(command, "--fast-only", self.fast_only_var.get())
        command.extend(["--fusion-policy", self.fusion_policy_var.get().strip() or "none"])
        self._add_optional_value(command, "--start-sun", self.start_sun_var.get())
        self._add_optional_value(command, "--board-timeout", self.board_timeout_var.get())
        self._add_optional_value(command, "--gameplay-ready-timeout", self.gameplay_ready_timeout_var.get())

    def browse_model(self) -> None:
        initial_dir = self.repo_root / "runs"
        filename = filedialog.askopenfilename(
            title="Select model.zip",
            initialdir=str(initial_dir if initial_dir.exists() else self.repo_root),
            filetypes=[("Zip models", "*.zip"), ("All files", "*.*")],
        )
        if filename:
            self.model_path_var.set(filename)
            self._append_log(f"Selected model path: {filename}\n")

    def browse_adventure_model(self) -> None:
        initial_dir = self.repo_root / "runs"
        filename = filedialog.askopenfilename(
            title="Select Adventure model.zip",
            initialdir=str(initial_dir if initial_dir.exists() else self.repo_root),
            filetypes=[("Zip models", "*.zip"), ("All files", "*.*")],
        )
        if filename:
            self.adventure_model_path_var.set(filename)
            self._append_log(f"Selected Adventure model path: {filename}\n")

    def browse_generalist_eval_model(self) -> None:
        initial_dir = self.repo_root / "runs"
        filename = filedialog.askopenfilename(
            title="Select Adventure Generalist eval model.zip",
            initialdir=str(initial_dir if initial_dir.exists() else self.repo_root),
            filetypes=[("Zip models", "*.zip"), ("All files", "*.*")],
        )
        if filename:
            self.generalist_eval_model_path_var.set(filename)
            self._append_log(f"Selected Adventure Generalist eval model path: {filename}\n")

    def browse_generalist_resume_model(self) -> None:
        initial_dir = self.repo_root / "runs"
        filename = filedialog.askopenfilename(
            title="Select Adventure Generalist resume model.zip",
            initialdir=str(initial_dir if initial_dir.exists() else self.repo_root),
            filetypes=[("Zip models", "*.zip"), ("All files", "*.*")],
        )
        if filename:
            self.generalist_resume_model_path_var.set(filename)
            self._append_log(f"Selected Adventure Generalist resume model path: {filename}\n")

    def browse_run_folder(self) -> None:
        initial_dir = self.repo_root / "runs"
        dirname = filedialog.askdirectory(
            title="Select run folder",
            initialdir=str(initial_dir if initial_dir.exists() else self.repo_root),
        )
        if dirname:
            self.run_dir_var.set(dirname)
            self.generalist_run_dir_var.set(dirname)
            self._append_log(f"Selected run directory: {dirname}\n")

    def refresh_generalist_models(self) -> None:
        model = self._find_newest_usable_model_zip()
        if model is None:
            self._append_log("Refresh Models: no metadata-backed Adventure Generalist model found.\n")
            return
        self.generalist_eval_model_path_var.set(str(model))
        self.generalist_resume_model_path_var.set(str(model))
        self._append_log(f"Refresh Models: selected Adventure Generalist model: {model}\n")

    def refresh_models(self) -> None:
        model = self._find_newest_model_zip()
        if model is None:
            self._append_log("Refresh Models: no .zip models found under runs.\n")
            return
        self.model_path_var.set(str(model))
        self._append_log(f"Refresh Models: selected newest model path: {model}\n")

    def refresh_adventure_model(self) -> None:
        model = self._find_newest_usable_model_zip()
        if model is None:
            self._append_log("Refresh Adventure model: no metadata-backed .zip models found under runs.\n")
            return
        self.adventure_model_path_var.set(str(model))
        self._append_log(f"Refresh Adventure model: selected newest metadata-backed model path: {model}\n")

    def refresh_generalist_eval_model(self) -> None:
        model = self._find_newest_usable_model_zip()
        if model is None:
            self._append_log("Refresh Adventure Generalist eval model: no metadata-backed .zip models found under runs.\n")
            return
        self.generalist_eval_model_path_var.set(str(model))
        self._append_log(f"Refresh Adventure Generalist eval model: selected newest metadata-backed model path: {model}\n")

    def refresh_generalist_resume_model(self) -> None:
        model = self._find_newest_usable_model_zip()
        if model is None:
            self._append_log("Refresh Adventure Generalist resume model: no metadata-backed .zip models found under runs.\n")
            return
        self.generalist_resume_model_path_var.set(str(model))
        self._append_log(f"Refresh Adventure Generalist resume model: selected newest metadata-backed model path: {model}\n")

    def open_run_folder(self) -> None:
        path = self._selected_run_dir()
        if path is None:
            path = self.repo_root / "runs"
        if not path.exists():
            self._append_log(f"ERROR: Folder does not exist: {path}\n")
            return
        try:
            if os.name == "nt":
                subprocess.Popen(["explorer", str(path)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
            self._append_log(f"Opened folder: {path}\n")
        except OSError as exc:
            self._append_log(f"ERROR: Failed to open folder {path}: {exc}\n")

    def _selected_run_dir(self) -> Optional[Path]:
        generalist_run_var = getattr(self, "generalist_run_dir_var", None)
        generalist_eval_var = getattr(self, "generalist_eval_model_path_var", None)
        generalist_resume_var = getattr(self, "generalist_resume_model_path_var", None)
        run_dir = (
            generalist_run_var.get().strip() if generalist_run_var is not None else ""
        ) or self.run_dir_var.get().strip()
        model_path = (
            (generalist_eval_var.get().strip() if generalist_eval_var is not None else "")
            or (generalist_resume_var.get().strip() if generalist_resume_var is not None else "")
            or self.model_path_var.get().strip()
        )
        if run_dir:
            return self._resolve_text_path(run_dir)
        if model_path:
            return self._resolve_text_path(model_path).parent
        runs_dir = self.repo_root / "runs"
        return runs_dir if runs_dir.exists() else None

    def open_adventure_run_folder(self) -> None:
        path = self._selected_adventure_run_dir()
        if path is None:
            path = self.repo_root / "runs"
        if not path.exists():
            self._append_log(f"ERROR: Adventure folder does not exist: {path}\n")
            return
        try:
            if os.name == "nt":
                subprocess.Popen(["explorer", str(path)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
            self._append_log(f"Opened Adventure folder: {path}\n")
        except OSError as exc:
            self._append_log(f"ERROR: Failed to open Adventure folder {path}: {exc}\n")

    def _selected_adventure_run_dir(self) -> Optional[Path]:
        if self.active_run_path:
            return self._resolve_text_path(self.active_run_path)
        model_path = self.adventure_model_path_var.get().strip()
        if model_path:
            return self._resolve_text_path(model_path).parent
        runs_dir = self.repo_root / "runs"
        return runs_dir if runs_dir.exists() else None

    def analyze_selected_run(self) -> None:
        run_dir = self._selected_run_dir()
        if run_dir is None or not run_dir.exists():
            self._append_log(f"ERROR: Cannot analyze missing run directory: {run_dir}\n")
            return
        self._append_log(f"\nAnalyzing selected run: {run_dir}\n")
        config_path = run_dir / "config.json"
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                summary = {
                    key: config.get(key)
                    for key in ("model_family", "seed_list", "total_timesteps", "max_steps", "game_speed", "run_dir")
                    if key in config
                }
                self._append_log(f"config.json: {json.dumps(summary, sort_keys=True)}\n")
            except Exception as exc:
                self._append_log(f"ERROR: Could not read config.json: {exc}\n")
        else:
            self._append_log(f"Missing config.json: {config_path}\n")

        command_path = run_dir / "command_used.txt"
        if command_path.exists():
            try:
                command = command_path.read_text(encoding="utf-8").strip()
                self._append_log(f"command_used.txt: {command}\n")
            except Exception as exc:
                self._append_log(f"ERROR: Could not read command_used.txt: {exc}\n")
        else:
            self._append_log(f"Missing command_used.txt: {command_path}\n")

        self._log_csv_summary(run_dir / "episode_metrics.csv")
        self._log_csv_summary(run_dir / "monitor.csv")

    def _log_csv_summary(self, csv_path: Path) -> None:
        if not csv_path.exists():
            self._append_log(f"Missing CSV: {csv_path}\n")
            return
        try:
            import pandas as pd
        except ImportError:
            self._append_log("pandas is not installed; CSV summary unavailable.\n")
            return
        try:
            data = pd.read_csv(csv_path, comment="#" if csv_path.name == "monitor.csv" else None)
            self._append_log(f"{csv_path.name}: rows={len(data)} columns={list(data.columns)}\n")
        except pd.errors.EmptyDataError:
            self._append_log(f"Empty CSV: {csv_path}\n")
        except Exception as exc:
            self._append_log(f"ERROR: Could not summarize {csv_path}: {exc}\n")

    def show_charts(self) -> None:
        run_dir = self._selected_run_dir()
        if run_dir is None or not run_dir.exists():
            self._append_log(f"ERROR: Cannot chart missing run directory: {run_dir}\n")
            return
        self._append_log(f"\nGenerating charts for selected run: {run_dir}\n")
        self._generate_charts(run_dir)

    def _generate_charts(self, run_dir: Path) -> None:
        try:
            import pandas as pd
        except ImportError:
            self._append_log("pandas is not installed; charts were not generated.\n")
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            self._append_log("matplotlib is not installed; charts were not generated.\n")
            return

        metrics = self._read_chart_csv(pd, run_dir / "episode_metrics.csv")
        monitor = self._read_chart_csv(pd, run_dir / "monitor.csv", monitor=True)
        if metrics is None and monitor is None:
            self._append_log("No CSV data available for charting.\n")
            return

        charts_dir = run_dir / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)
        generated: List[Path] = []
        skipped: List[str] = []

        generated.extend(self._plot_first_available(plt, pd, charts_dir, "reward_curve.png", "Episode Reward", "reward", metrics, monitor, ["episode_reward", "reward", "r"]))
        win_chart = self._plot_rolling_win_rate(plt, pd, charts_dir, metrics)
        if win_chart:
            generated.append(win_chart)
        else:
            skipped.append("rolling win rate: no win/success column")
        generated.extend(self._plot_first_available(plt, pd, charts_dir, "wave_curve.png", "Wave Reached", "wave", metrics, monitor, ["final_wave", "wave", "avg_wave", "max_wave_reached"]))
        generated.extend(self._plot_first_available(plt, pd, charts_dir, "kills_curve.png", "Kills Per Episode", "kills", metrics, monitor, ["zombies_killed", "kills", "avg_kills"]))
        generated.extend(self._plot_first_available(plt, pd, charts_dir, "episode_length.png", "Episode Length", "steps", metrics, monitor, ["episode_length", "length", "steps", "l"]))
        generated.extend(self._plot_first_available(plt, pd, charts_dir, "illegal_actions.png", "Illegal Actions", "illegal actions", metrics, monitor, ["illegal_actions", "illegal"]))
        generated.extend(self._plot_first_available(plt, pd, charts_dir, "mower_losses.png", "Mower Losses", "mowers lost", metrics, monitor, ["mowers_lost", "mower_losses"]))

        for message in skipped:
            self._append_log(f"Skipped chart: {message}\n")
        if not generated:
            self._append_log("No chartable columns found; no charts generated.\n")
            return
        self._append_log("Generated chart PNGs:\n")
        for path in generated:
            self._append_log(f"  {path}\n")

    def _read_chart_csv(self, pd: Any, path: Path, monitor: bool = False) -> Any:
        if not path.exists():
            self._append_log(f"Missing CSV for charts: {path}\n")
            return None
        try:
            data = pd.read_csv(path, comment="#" if monitor else None)
        except pd.errors.EmptyDataError:
            self._append_log(f"Empty CSV for charts: {path}\n")
            return None
        except Exception as exc:
            self._append_log(f"ERROR: Could not read chart CSV {path}: {exc}\n")
            return None
        if data.empty:
            self._append_log(f"CSV has no rows for charts: {path}\n")
            return None
        return data

    def _plot_first_available(
        self,
        plt: Any,
        pd: Any,
        charts_dir: Path,
        filename: str,
        title: str,
        ylabel: str,
        metrics: Any,
        monitor: Any,
        columns: List[str],
    ) -> List[Path]:
        for data in (metrics, monitor):
            if data is None:
                continue
            column = self._find_column(data, columns)
            if column is None:
                continue
            series = pd.to_numeric(data[column], errors="coerce")
            if series.dropna().empty:
                continue
            path = charts_dir / filename
            self._plot_series(plt, data, series, title, ylabel, path)
            return [path]
        self._append_log(f"Skipped chart: {title}; missing columns {columns}\n")
        return []

    def _plot_rolling_win_rate(self, plt: Any, pd: Any, charts_dir: Path, metrics: Any) -> Optional[Path]:
        if metrics is None:
            return None
        column = self._find_column(metrics, ["win", "won", "success", "is_win", "episode_win"])
        inferred = False
        if column is not None:
            raw = metrics[column]
            wins = pd.to_numeric(raw.map(self._win_value_to_float), errors="coerce")
        else:
            column = self._find_column(metrics, ["win_loss_reward_total"])
            if column is None:
                return None
            inferred = True
            wins = (pd.to_numeric(metrics[column], errors="coerce") > 0).astype(float)
        if wins.dropna().empty:
            return None
        if inferred:
            self._append_log("Rolling win rate inferred from win_loss_reward_total because no win column was found.\n")
        rolling = wins.rolling(window=ROLLING_WIN_WINDOW, min_periods=1).mean()
        path = charts_dir / "rolling_win_rate.png"
        self._plot_series(plt, metrics, rolling, f"Rolling Win Rate ({ROLLING_WIN_WINDOW})", "win rate", path, ylim=(0.0, 1.0))
        return path

    def _win_value_to_float(self, value: Any) -> Optional[float]:
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        text = str(value).strip().lower()
        if text in {"1", "1.0", "true", "yes", "y", "win", "won", "success"}:
            return 1.0
        if text in {"0", "0.0", "false", "no", "n", "loss", "lost", "fail", "failed", ""}:
            return 0.0
        try:
            return 1.0 if float(text) > 0.0 else 0.0
        except ValueError:
            return None

    def _find_column(self, data: Any, candidates: List[str]) -> Optional[str]:
        columns = {str(column).lower(): column for column in data.columns}
        for candidate in candidates:
            column = columns.get(candidate.lower())
            if column is not None:
                return str(column)
        return None

    def _plot_series(
        self,
        plt: Any,
        data: Any,
        series: Any,
        title: str,
        ylabel: str,
        path: Path,
        ylim: Optional[Tuple[float, float]] = None,
    ) -> None:
        episode_column = self._find_column(data, ["episode"])
        x_values = data[episode_column] if episode_column is not None else range(1, len(series) + 1)
        fig, ax = plt.subplots(figsize=(7.0, 3.4))
        ax.plot(x_values, series)
        ax.set_title(title)
        ax.set_xlabel("episode")
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def clear_logs(self) -> None:
        self._cancel_after("_log_view_after_id")
        history = getattr(self, "log_history", None)
        if history is not None:
            history.clear()
        self.log_history_chars = 0
        self.log_dropped_lines = 0
        self._log_view_dirty = False
        self._log_notice_present = False
        if self.log_text is None:
            return
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _append_log(self, text: str) -> None:
        if not hasattr(self, "log_history"):
            self.log_history = []
        text = str(text)
        lines = text.splitlines(keepends=True) if text else []
        if text and not lines:
            lines = [text]
        if not hasattr(self, "log_history_chars"):
            self.log_history_chars = sum(len(line) for line in self.log_history)
        if not hasattr(self, "log_dropped_lines"):
            self.log_dropped_lines = 0
        previous_history_length = len(self.log_history)
        self.log_history.extend(lines)
        self.log_history_chars += sum(len(line) for line in lines)

        drop_count = max(0, len(self.log_history) - LOG_HISTORY_MAX_LINES)
        dropped_chars = sum(len(line) for line in self.log_history[:drop_count])
        retained_chars = self.log_history_chars - dropped_chars
        while drop_count < len(self.log_history) and retained_chars > LOG_HISTORY_MAX_CHARS:
            retained_chars -= len(self.log_history[drop_count])
            drop_count += 1
        requires_full_log_refresh = bool(
            drop_count
            and any(not entry.endswith("\n") for entry in self.log_history[:drop_count])
        )
        if drop_count:
            del self.log_history[:drop_count]
            self.log_history_chars = max(0, retained_chars)
            self.log_dropped_lines += drop_count
        incoming_drop_count = max(0, drop_count - previous_history_length)
        retained_text = "".join(lines[incoming_drop_count:])
        if self.log_text is None:
            return
        filter_var = getattr(self, "log_filter_var", None)
        severity_var = getattr(self, "log_severity_var", None)
        pause_var = getattr(self, "log_pause_autoscroll_var", None)
        filtering = (filter_var is not None and filter_var.get().strip()) or (
            severity_var is not None and severity_var.get() != "All"
        )
        if filtering:
            self._request_log_view_refresh()
            return
        if requires_full_log_refresh:
            self._refresh_log_view()
            return
        self._append_unfiltered_log_view(retained_text, drop_count)

    def _append_unfiltered_log_view(self, text: str, drop_count: int) -> None:
        if self.log_text is None:
            return
        pause_var = getattr(self, "log_pause_autoscroll_var", None)
        self.log_text.configure(state="normal")
        if drop_count:
            if bool(getattr(self, "_log_notice_present", False)):
                self.log_text.delete("1.0", "2.0")
            self.log_text.delete("1.0", f"{drop_count + 1}.0")
            notice = self._log_drop_notice()
            if notice:
                self.log_text.insert("1.0", notice)
            self._log_notice_present = bool(notice)
        if text:
            self.log_text.insert("end", text)
        if pause_var is None or not pause_var.get():
            self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _request_log_view_refresh(self) -> None:
        self._log_view_dirty = True
        elapsed = time.monotonic() - float(getattr(self, "_last_log_view_refresh_at", 0.0) or 0.0)
        if elapsed >= LOG_VIEW_REFRESH_MIN_SECONDS:
            self._refresh_log_view()
            return
        if getattr(self, "_log_view_after_id", None) is not None:
            return
        delay_ms = max(1, int((LOG_VIEW_REFRESH_MIN_SECONDS - elapsed) * 1000))
        self._schedule_after("_log_view_after_id", delay_ms, self._flush_log_view_refresh)

    def _flush_log_view_refresh(self) -> None:
        self._log_view_after_id = None
        if self._destroyed or not bool(getattr(self, "_log_view_dirty", False)):
            return
        self._refresh_log_view()

    def _refresh_log_view(self) -> None:
        self._cancel_after("_log_view_after_id")
        if self.log_text is None:
            return
        filter_var = getattr(self, "log_filter_var", None)
        severity_var = getattr(self, "log_severity_var", None)
        pause_var = getattr(self, "log_pause_autoscroll_var", None)
        search = "" if filter_var is None else filter_var.get().strip().lower()
        severity = "All" if severity_var is None else severity_var.get()
        lines = "".join(self.log_history).splitlines(keepends=True)

        def matches(line: str) -> bool:
            lowered = line.lower()
            if search and search not in lowered:
                return False
            if severity == "ERROR":
                return "error" in lowered
            if severity == "Warning":
                return "warning" in lowered or "warn" in lowered
            if severity == "Live status":
                return "live status" in lowered
            if severity == "Process":
                return any(token in lowered for token in ("process", "starting subprocess", "exited", "terminate"))
            return True

        filtered = self._log_drop_notice() + "".join(line for line in lines if matches(line))
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.insert("1.0", filtered)
        if pause_var is None or not pause_var.get():
            self.log_text.see("end")
        self.log_text.configure(state="disabled")
        self._last_log_view_refresh_at = time.monotonic()
        self._log_view_dirty = False
        self._log_notice_present = bool(self._log_drop_notice())

    def _log_drop_notice(self) -> str:
        dropped = int(getattr(self, "log_dropped_lines", 0) or 0)
        if dropped <= 0:
            return ""
        return (
            f"[gui] log retention dropped {dropped} older line(s); "
            f"showing the newest {len(self.log_history)} line(s).\n"
        )

    def copy_selected_logs(self) -> None:
        if self.log_text is None:
            return
        try:
            selected = self.log_text.get("sel.first", "sel.last")
        except tk.TclError:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(selected)

    def _on_close(self) -> None:
        if self._closing or self._destroyed:
            return
        self._closing = True
        self._close_deadline = time.monotonic() + STOP_GRACE_SECONDS + STOP_KILL_WAIT_SECONDS + 1.0
        self._cancel_scheduled_callbacks()
        process = self.active_process
        if process is not None and process.poll() is None:
            self._begin_process_stop(self.active_process_name or "process", process)
        self._poll_close_cleanup()

    def _poll_close_cleanup(self) -> None:
        self._close_after_id = None
        if self._destroyed:
            return
        self._consume_log_queue(max(LOG_DRAIN_MAX_ITEMS, 1000), max(LOG_DRAIN_BUDGET_SECONDS, 0.02))

        process = self.active_process
        exit_code = process.poll() if process is not None else 0
        if process is not None and exit_code is not None:
            self._handle_process_exit(self.active_process_name or "process", process, int(exit_code))
            process = None

        if process is not None and time.monotonic() >= self._close_deadline:
            try:
                process.kill()
            except OSError as exc:
                self._append_log(f"ERROR: Final close-time kill failed: {exc}\n")
            if process.poll() is None:
                self._append_log(
                    "ERROR: Process did not report exit by the hard GUI close deadline; "
                    "closing the dashboard after final kill attempt.\n"
                )
            process = None

        if process is not None and process.poll() is None:
            self._schedule_after("_close_after_id", CLOSE_POLL_MS, self._poll_close_cleanup)
            return

        for thread in (self._stopper_thread, self._reader_thread):
            if thread is not None and thread is not threading.current_thread() and thread.is_alive():
                thread.join(timeout=PROCESS_THREAD_JOIN_SECONDS)
        self._drain_remaining_logs_for_close()
        self._finish_close()

    def _drain_remaining_logs_for_close(self) -> None:
        deadline = time.monotonic() + CLOSE_LOG_DRAIN_SECONDS
        while not self.log_queue.empty() and time.monotonic() < deadline:
            self._consume_log_queue(max(LOG_DRAIN_MAX_ITEMS, 1000), max(LOG_DRAIN_BUDGET_SECONDS, 0.025))
        self._consume_log_queue(max(LOG_DRAIN_MAX_ITEMS, 1000), max(LOG_DRAIN_BUDGET_SECONDS, 0.025))

    def _finish_close(self) -> None:
        if self._destroyed:
            return
        self._cancel_scheduled_callbacks()
        self._destroyed = True
        self.active_process = None
        self._stopping_process = None
        self._reader_thread = None
        self._stopper_thread = None
        try:
            self.root.destroy()
        except tk.TclError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="PvZRL interactive dashboard")
    parser.add_argument("--live-status-path", type=Path, default=DEFAULT_LIVE_STATUS_PATH)
    args = parser.parse_args()
    root = tk.Tk()
    PvZDashboard(root, args.live_status_path)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
