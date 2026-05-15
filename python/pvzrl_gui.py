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
from typing import Any, Dict, Iterable, List, Optional, Tuple

import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk


POLL_MS = 1000
LOG_POLL_MS = 100
STOP_GRACE_SECONDS = 5.0
ROLLING_WIN_WINDOW = 20
MODEL_ZIP_NAMES = {"model.zip", "final_model.zip"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIVE_STATUS_PATH = Path("runs") / "live_status.json"
LIVE_MAX_AGE_SECONDS = 5.0
STALE_MAX_AGE_SECONDS = 30.0
MISSING = object()
ADVENTURE_DEFAULT_PLANT_TYPES = "1,0,3,2"
FUSION_POLICY_CHOICES = ("none", "observe", "scripted", "assist")
LEVEL3_SEED_LIST = "SunFlower,Peashooter,WallNut,CherryBomb"
LEVEL3_PLANT_TYPES = "1,0,3,2"
ADVENTURE_GENERALIST_MODEL_FAMILY = "ppo_adventure_generalist_14slot_identity_v1"
ADVENTURE_GENERALIST_INITIAL_LOADOUT = "SunFlower,SunFlower,Peashooter,Peashooter"
ADVENTURE_GENERALIST_ACTION_SPACE_MODE = "adventure_14slot_identity"
PROFILES = {
    "4-slot current": "SunFlower,Peashooter,WallNut,CherryBomb",
    "2-slot stable": "SunFlower,Peashooter",
    "4-slot duplicate": "SunFlower,SunFlower,Peashooter,Peashooter",
}


def fmt_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value) if value else "-"
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    if value is None or value == "":
        return "-"
    return str(value)


def lines_from_pairs(pairs: Iterable[Tuple[str, Any]]) -> str:
    rows = [(label, fmt_value(value)) for label, value in pairs]
    width = max([len(label) for label, _ in rows] or [1])
    return "\n".join(f"{label.ljust(width)}  {value}" for label, value in rows)


def row_panel_lines(rows: Dict[str, Any]) -> str:
    output: List[str] = []
    for row, payload in sorted(rows.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else 99):
        if not isinstance(payload, dict):
            continue
        output.append(
            f"row {row}: pea={payload.get('peashooters', 0)} "
            f"threat={payload.get('threatened', False)} "
            f"undef={payload.get('undefended_threat', False)} "
            f"steps={payload.get('threat_steps', 0)}"
        )
    return "\n".join(output) if output else "-"


class PvZDashboard:
    def __init__(self, root: tk.Tk, live_status_path: Path):
        self.root = root
        self.project_root = PROJECT_ROOT
        self.repo_root = self.project_root
        self.live_status_path = self._resolve_path(live_status_path)
        self.root.title("PvZRL Dashboard")
        self.root.geometry("850x550")
        self.root.minsize(760, 500)

        default_model = self._find_newest_model_zip()
        default_adventure_model = self._find_newest_usable_model_zip() or default_model
        self.total_timesteps_var = tk.StringVar(value="250000")
        self.max_steps_var = tk.StringVar(value="1000")
        self.step_seconds_var = tk.StringVar(value="0.05")
        self.game_speed_var = tk.StringVar(value="4.0")
        self.seed_list_var = tk.StringVar(value=PROFILES["4-slot current"])
        self.model_path_var = tk.StringVar(value=str(default_model) if default_model else "")
        self.episodes_var = tk.StringVar(value="10")
        self.run_dir_var = tk.StringVar(value="")
        self.run_name_var = tk.StringVar(value="")
        self.start_sun_var = tk.StringVar(value="500")
        self.board_timeout_var = tk.StringVar(value="60")
        self.gameplay_ready_timeout_var = tk.StringVar(value="30")
        self.checkpoint_freq_var = tk.StringVar(value="5000")
        self.profile_var = tk.StringVar(value="4-slot current")
        self.fusion_policy_var = tk.StringVar(value="none")
        self.quick_wait_var = tk.BooleanVar(value=True)
        self.wait_gameplay_ready_var = tk.BooleanVar(value=True)
        self.auto_select_seeds_var = tk.BooleanVar(value=True)
        self.debug_perf_var = tk.BooleanVar(value=False)
        self.fast_only_var = tk.BooleanVar(value=False)
        self.adventure_model_path_var = tk.StringVar(value=str(default_adventure_model) if default_adventure_model else "")
        self.adventure_seed_list_var = tk.StringVar(value=PROFILES["4-slot current"])
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
        self.generalist_curriculum_var = tk.StringVar(value="conservative")
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
        self.process_status_var = tk.StringVar(value="Idle")
        self.live_status_var = tk.StringVar(
            value=f"Live: {self.live_status_path} | exists=no | size=- | age=- | health=MISSING"
        )
        self.diagnostics_status_var = tk.StringVar(value="MISSING - no live status has been read")
        self.schema_keys_var = tk.StringVar(value="Schema keys: -")
        self.active_run_var = tk.StringVar(value="Active run: unknown")

        self.panels: Dict[str, tk.Text] = {}
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
        self.log_queue: queue.Queue[Any] = queue.Queue()
        self.active_process: Optional[subprocess.Popen[str]] = None
        self.active_process_name = ""
        self.active_process_started_at: Optional[float] = None
        self.active_run_path = ""
        self.live_writer_warning_emitted = False
        self.last_good_status: Optional[Dict[str, Any]] = None
        self.last_good_read_time: Optional[float] = None
        self.last_live_parse_error = ""
        self.last_live_health = ""
        self.last_live_warning_key = ""

        self._build()
        self._append_log(f"Project root: {self.project_root}\n")
        self._append_log(f"Live status path: {self.live_status_path}\n")
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
        level3_tab = ttk.Frame(notebook)
        adventure_tab = ttk.Frame(notebook)
        generalist_tab = ttk.Frame(notebook)
        diagnostics_tab = ttk.Frame(notebook)
        runs_tab = ttk.Frame(notebook)
        logs_tab = ttk.Frame(notebook)
        notebook.add(train_tab, text="Train")
        notebook.add(eval_tab, text="Eval")
        notebook.add(level3_tab, text="Level 3")
        notebook.add(adventure_tab, text="Adventure")
        notebook.add(generalist_tab, text="Adventure Generalist")
        notebook.add(diagnostics_tab, text="Diagnostics")
        notebook.add(runs_tab, text="Runs/Models")
        notebook.add(logs_tab, text="Logs")

        self._build_train_tab(train_tab)
        self._build_eval_tab(eval_tab)
        self._build_level3_tab(level3_tab)
        self._build_adventure_tab(adventure_tab)
        self._build_adventure_generalist_tab(generalist_tab)
        self._build_diagnostics_tab(diagnostics_tab)
        self._build_runs_tab(runs_tab)
        self._build_logs_tab(logs_tab)
        self._build_status_bar()
        self._build_log_panel()

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
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        self.log_text = scrolledtext.ScrolledText(frame, height=5, wrap="word", font=("Consolas", 9))
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.log_text.configure(state="disabled")

    def _build_train_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(5, weight=1)

        profile = ttk.LabelFrame(parent, text="Profile and Seeds")
        profile.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 3))
        profile.columnconfigure(1, weight=1)
        ttk.Label(profile, text="Profile").grid(row=0, column=0, sticky="w", padx=6, pady=3)
        combo = ttk.Combobox(profile, textvariable=self.profile_var, values=list(PROFILES), state="readonly", width=22)
        combo.grid(row=0, column=1, sticky="w", padx=6, pady=3)
        combo.bind("<<ComboboxSelected>>", self._on_profile_selected)
        self._add_labeled_entry(profile, 1, 0, "Seed list", self.seed_list_var, width=50)

        settings = ttk.LabelFrame(parent, text="Training Settings")
        settings.grid(row=1, column=0, sticky="ew", padx=8, pady=3)
        for column in (1, 3):
            settings.columnconfigure(column, weight=1)
        self._add_labeled_entry(settings, 0, 0, "Timesteps", self.total_timesteps_var)
        self._add_labeled_entry(settings, 0, 2, "Max steps", self.max_steps_var)
        self._add_labeled_entry(settings, 1, 0, "Step seconds", self.step_seconds_var)
        self._add_labeled_entry(settings, 1, 2, "Game speed", self.game_speed_var)
        self._add_labeled_entry(settings, 2, 0, "Start sun", self.start_sun_var)
        self._add_labeled_entry(settings, 2, 2, "Checkpoint freq", self.checkpoint_freq_var)
        self._add_labeled_entry(settings, 3, 0, "Board timeout", self.board_timeout_var)
        self._add_labeled_entry(settings, 3, 2, "Ready timeout", self.gameplay_ready_timeout_var)

        run = ttk.LabelFrame(parent, text="Run / Resume")
        run.grid(row=2, column=0, sticky="ew", padx=8, pady=3)
        run.columnconfigure(1, weight=1)
        self._add_labeled_entry(run, 0, 0, "Run dir", self.run_dir_var, width=50)
        self._add_labeled_entry(run, 1, 0, "Run suffix", self.run_name_var, width=50)
        self._add_labeled_entry(run, 2, 0, "Resume model", self.model_path_var, width=50)

        options = ttk.LabelFrame(parent, text="Options")
        options.grid(row=3, column=0, sticky="ew", padx=8, pady=3)
        self._add_options(options)

        actions = ttk.Frame(parent)
        actions.grid(row=4, column=0, sticky="ew", padx=8, pady=3)
        self._add_launch_button(actions, "Start Training", self.start_training).grid(row=0, column=0, sticky="w", padx=(0, 5))
        self._add_launch_button(actions, "Resume Training", self.resume_training).grid(row=0, column=1, sticky="w", padx=(0, 5))
        ttk.Button(actions, text="Refresh Models", command=self.refresh_models).grid(row=0, column=2, sticky="w", padx=(0, 5))
        self._add_stop_button(actions, "Stop Active Process").grid(row=0, column=3, sticky="w")

        preview = ttk.LabelFrame(parent, text="Command Preview")
        preview.grid(row=5, column=0, sticky="nsew", padx=8, pady=(3, 6))
        preview.rowconfigure(0, weight=1)
        preview.columnconfigure(0, weight=1)
        self.train_preview = self._preview_box(preview, height=4)

    def _build_eval_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(5, weight=1)

        model = ttk.LabelFrame(parent, text="Model")
        model.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 3))
        model.columnconfigure(1, weight=1)
        self._add_labeled_entry(model, 0, 0, "Model .zip", self.model_path_var, width=56)
        self._add_labeled_entry(model, 1, 0, "Run dir", self.run_dir_var, width=56)

        settings = ttk.LabelFrame(parent, text="Evaluation Settings")
        settings.grid(row=1, column=0, sticky="ew", padx=8, pady=3)
        for column in (1, 3):
            settings.columnconfigure(column, weight=1)
        self._add_labeled_entry(settings, 0, 0, "Episodes", self.episodes_var)
        self._add_labeled_entry(settings, 0, 2, "Max steps", self.max_steps_var)
        self._add_labeled_entry(settings, 1, 0, "Step seconds", self.step_seconds_var)
        self._add_labeled_entry(settings, 1, 2, "Game speed", self.game_speed_var)
        self._add_labeled_entry(settings, 2, 0, "Seed list", self.seed_list_var, width=48, columnspan=3)

        options = ttk.LabelFrame(parent, text="Options")
        options.grid(row=2, column=0, sticky="ew", padx=8, pady=3)
        self._add_options(options)

        actions = ttk.Frame(parent)
        actions.grid(row=3, column=0, sticky="ew", padx=8, pady=3)
        self._add_launch_button(actions, "Run Eval", self.run_eval).grid(row=0, column=0, sticky="w", padx=(0, 5))
        self._add_launch_button(actions, "Run 25-Episode Eval", self.run_25_episode_eval).grid(row=0, column=1, sticky="w", padx=(0, 5))
        ttk.Button(actions, text="Refresh Models", command=self.refresh_models).grid(row=0, column=2, sticky="w", padx=(0, 5))
        self._add_stop_button(actions, "Stop Active Process").grid(row=0, column=3, sticky="w")

        preview = ttk.LabelFrame(parent, text="Command Preview")
        preview.grid(row=4, column=0, sticky="nsew", padx=8, pady=(3, 6))
        preview.rowconfigure(0, weight=1)
        preview.columnconfigure(0, weight=1)
        self.eval_preview = self._preview_box(preview, height=4)

    def _build_level3_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(4, weight=1)

        mode = ttk.LabelFrame(parent, text="Level 3 Specialist")
        mode.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 3))
        mode.columnconfigure(1, weight=1)
        ttk.Radiobutton(mode, text="Train", variable=self.level3_mode_var, value="train").grid(row=0, column=0, sticky="w", padx=6, pady=3)
        ttk.Radiobutton(mode, text="Eval", variable=self.level3_mode_var, value="eval").grid(row=0, column=1, sticky="w", padx=6, pady=3)
        self._add_labeled_entry(mode, 1, 0, "Target level", self.level3_target_level_var)
        self._add_labeled_entry(mode, 1, 2, "Model .zip", self.level3_model_path_var, width=46, columnspan=3)

        settings = ttk.LabelFrame(parent, text="Run Settings")
        settings.grid(row=1, column=0, sticky="ew", padx=8, pady=3)
        for column in (1, 3):
            settings.columnconfigure(column, weight=1)
        self._add_labeled_entry(settings, 0, 0, "Timesteps", self.level3_total_timesteps_var)
        self._add_labeled_entry(settings, 0, 2, "Episodes", self.level3_episodes_var)
        self._add_labeled_entry(settings, 1, 0, "Max steps", self.level3_max_steps_var)
        self._add_labeled_entry(settings, 1, 2, "Game speed", self.level3_game_speed_var)
        self._add_labeled_entry(settings, 2, 0, "Step seconds", self.level3_step_seconds_var)
        self._add_labeled_entry(settings, 2, 2, "Board timeout", self.level3_board_timeout_var)
        self._add_labeled_entry(settings, 3, 0, "Seed list", self.level3_seed_list_var, width=46, columnspan=3)
        self._add_labeled_entry(settings, 4, 0, "Plant types", self.level3_plant_types_var, width=46, columnspan=3)

        masks = ttk.LabelFrame(parent, text="Tactical Masks")
        masks.grid(row=2, column=0, sticky="ew", padx=8, pady=3)
        ttk.Checkbutton(masks, text="tactical masks", variable=self.level3_tactical_masks_var).grid(row=0, column=0, sticky="w", padx=6, pady=3)
        ttk.Checkbutton(masks, text="WallNut mask", variable=self.level3_wallnut_mask_var).grid(row=0, column=1, sticky="w", padx=6, pady=3)
        ttk.Checkbutton(masks, text="CherryBomb mask", variable=self.level3_cherrybomb_mask_var).grid(row=0, column=2, sticky="w", padx=6, pady=3)

        actions = ttk.Frame(parent)
        actions.grid(row=3, column=0, sticky="ew", padx=8, pady=3)
        self._add_launch_button(actions, "Start Level 3", self.start_level3_specialist).grid(row=0, column=0, sticky="w", padx=(0, 5))
        ttk.Button(actions, text="Refresh Models", command=self.refresh_models).grid(row=0, column=1, sticky="w", padx=(0, 5))
        self._add_stop_button(actions, "Stop Run").grid(row=0, column=2, sticky="w", padx=(0, 5))

        preview = ttk.LabelFrame(parent, text="Command Preview")
        preview.grid(row=4, column=0, sticky="nsew", padx=8, pady=(3, 6))
        preview.rowconfigure(0, weight=1)
        preview.columnconfigure(0, weight=1)
        self.level3_preview = self._preview_box(preview, height=5)

    def _build_diagnostics_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)
        actions = ttk.LabelFrame(parent, text="Diagnostic Commands")
        actions.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 3))
        for column in range(4):
            actions.columnconfigure(column, weight=1)
        self._add_launch_button(actions, "Auto reset test", self.auto_reset_test).grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        self._add_launch_button(actions, "Seed selection test", self.seed_selection_test).grid(row=0, column=1, sticky="ew", padx=4, pady=4)
        self._add_launch_button(actions, "Cooldown test", self.cooldown_test).grid(row=0, column=2, sticky="ew", padx=4, pady=4)
        self._add_launch_button(actions, "Bridge/perf diagnostic", self.bridge_perf_diagnostic).grid(row=0, column=3, sticky="ew", padx=4, pady=4)
        ttk.Button(actions, text="Refresh Diagnostics Now", command=self.refresh_diagnostics_now).grid(
            row=1, column=0, columnspan=2, sticky="ew", padx=4, pady=(0, 4)
        )
        ttk.Button(actions, text="Show Raw Live JSON", command=self.show_raw_live_json).grid(
            row=1, column=2, columnspan=2, sticky="ew", padx=4, pady=(0, 4)
        )

        status = ttk.Frame(parent)
        status.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 4))
        status.columnconfigure(0, weight=3)
        status.columnconfigure(1, weight=2)
        status.columnconfigure(2, weight=2)
        ttk.Label(status, textvariable=self.diagnostics_status_var, anchor="w").grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ttk.Label(status, textvariable=self.schema_keys_var, anchor="w").grid(row=0, column=1, sticky="ew", padx=(0, 8))
        ttk.Label(status, textvariable=self.active_run_var, anchor="w").grid(row=0, column=2, sticky="ew")

        grid = ttk.Frame(parent)
        grid.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 6))
        for column in range(3):
            grid.columnconfigure(column, weight=1)
        for row in range(3):
            grid.rowconfigure(row, weight=1)

        panel_specs = [
            ("Adventure", 0, 0),
            ("Gameplay", 0, 1),
            ("Agent", 0, 2),
            ("Rows", 1, 0),
            ("Reward", 1, 1),
            ("Eval", 1, 2),
            ("Fusion", 2, 0),
            ("Seed Inventory / Compatibility", 2, 1),
        ]
        for title, row, column in panel_specs:
            frame = ttk.LabelFrame(grid, text=title)
            frame.grid(row=row, column=column, columnspan=2 if title == "Seed Inventory / Compatibility" else 1, sticky="nsew", padx=4, pady=4)
            frame.rowconfigure(0, weight=1)
            frame.columnconfigure(0, weight=1)
            text = tk.Text(frame, height=5, wrap="word", font=("Consolas", 9))
            text.grid(row=0, column=0, sticky="nsew")
            text.configure(state="disabled")
            self.panels[title] = text

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

    def _build_logs_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        actions = ttk.LabelFrame(parent, text="Logs")
        actions.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 3))
        ttk.Button(actions, text="Clear logs", command=self.clear_logs).grid(row=0, column=0, sticky="w", padx=6, pady=6)

    def _add_options(self, parent: ttk.Frame) -> None:
        ttk.Checkbutton(parent, text="quick-wait", variable=self.quick_wait_var).grid(row=0, column=0, sticky="w", padx=6, pady=3)
        ttk.Checkbutton(parent, text="wait gameplay", variable=self.wait_gameplay_ready_var).grid(row=0, column=1, sticky="w", padx=6, pady=3)
        ttk.Checkbutton(parent, text="auto seeds", variable=self.auto_select_seeds_var).grid(row=0, column=2, sticky="w", padx=6, pady=3)
        ttk.Checkbutton(parent, text="debug perf", variable=self.debug_perf_var).grid(row=0, column=3, sticky="w", padx=6, pady=3)
        ttk.Checkbutton(parent, text="fast-only diag", variable=self.fast_only_var).grid(row=0, column=4, sticky="w", padx=6, pady=3)
        ttk.Label(parent, text="fusion").grid(row=1, column=0, sticky="w", padx=6, pady=3)
        ttk.Combobox(
            parent,
            textvariable=self.fusion_policy_var,
            values=FUSION_POLICY_CHOICES,
            state="readonly",
            width=10,
        ).grid(row=1, column=1, sticky="w", padx=6, pady=3)

    def _add_labeled_entry(
        self,
        parent: ttk.Frame,
        row: int,
        column: int,
        label: str,
        variable: tk.StringVar,
        width: int = 16,
        columnspan: int = 1,
    ) -> ttk.Entry:
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w", padx=6, pady=2)
        entry = ttk.Entry(parent, textvariable=variable, width=width)
        entry.grid(row=row, column=column + 1, columnspan=columnspan, sticky="ew", padx=6, pady=2)
        return entry

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
        ]
        for variable in variables:
            variable.trace_add("write", lambda *_args: self._update_command_previews())

    def _on_profile_selected(self, _event: Any = None) -> None:
        seed_list = PROFILES.get(self.profile_var.get())
        if seed_list:
            self.seed_list_var.set(seed_list)
            self._append_log(f"Profile selected: {self.profile_var.get()} seed_list={seed_list}\n")

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
        self._append_live_status_arg(command)
        return command

    def _build_adventure_generalist_command(self) -> List[str]:
        command = self._script_command("train_ppo.py", "--adventure-generalist-train")
        self._add_optional_value(command, "--total-timesteps", self.generalist_total_timesteps_var.get())
        self._add_optional_value(command, "--checkpoint-freq", self.generalist_checkpoint_freq_var.get())
        self._add_optional_value(command, "--initial-loadout", self.generalist_initial_loadout_var.get())
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
        run_dir = self.generalist_run_dir_var.get().strip()
        if run_dir:
            command.extend(["--run-dir", run_dir])
        self._append_live_status_arg(command)
        return command

    def _build_adventure_generalist_eval_command(self) -> List[str]:
        command = self._script_command("train_ppo.py", "--adventure-generalist-eval")
        model_path = self.generalist_eval_model_path_var.get().strip()
        if model_path:
            command.extend(["--model-path", str(self._resolve_text_path(model_path))])
        self._add_optional_value(command, "--episodes", self.generalist_eval_episodes_var.get())
        self._add_optional_value(command, "--initial-loadout", self.generalist_initial_loadout_var.get())
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
        run_dir = self.generalist_run_dir_var.get().strip()
        if run_dir:
            command.extend(["--run-dir", run_dir])
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

    def _update_command_previews(self) -> None:
        if self.train_preview is not None:
            lines = [
                "Start Training:",
                self._command_text(self._build_train_command(resume=False)),
                "",
                "Resume Training:",
            ]
            if self.model_path_var.get().strip():
                lines.append(self._command_text(self._build_train_command(resume=True)))
            else:
                lines.append("model_path is required for resume")
            if self.fast_only_var.get():
                lines.extend(["", "Note: --fast-only is supported by pvzrl_env.py diagnostics only; omitted from train_ppo.py commands."])
            self._set_text_widget(self.train_preview, "\n".join(lines))
        if self.eval_preview is not None:
            lines = [
                "Run Eval:",
                self._command_text(self._build_eval_command()),
                "",
                "Run 25-Episode Eval:",
                self._command_text(self._build_eval_command(episodes_override="25")),
            ]
            if self.fast_only_var.get():
                lines.extend(["", "Note: --fast-only is supported by pvzrl_env.py diagnostics only; omitted from train_ppo.py commands."])
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
            lines.extend(
                [
                    "",
                    f"Model family: {ADVENTURE_GENERALIST_MODEL_FAMILY}",
                    "Fresh 14-slot identity architecture; not a 4-slot checkpoint continuation.",
                ]
            )
            self._set_text_widget(self.generalist_preview, "\n".join(lines))
        if self.level3_preview is not None:
            lines = [
                "Start Level 3 Specialist:",
                self._command_text(self._build_level3_command()),
            ]
            self._set_text_widget(self.level3_preview, "\n".join(lines))

    def _build_adventure_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        parent.rowconfigure(3, weight=1)

        model = ttk.LabelFrame(parent, text="Model")
        model.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 3))
        model.columnconfigure(1, weight=1)
        self._add_labeled_entry(model, 0, 0, "Model .zip", self.adventure_model_path_var, width=54)
        ttk.Button(model, text="Browse", command=self.browse_adventure_model).grid(row=0, column=2, sticky="w", padx=(0, 5), pady=2)
        ttk.Button(model, text="Refresh", command=self.refresh_adventure_model).grid(row=0, column=3, sticky="w", padx=(0, 6), pady=2)
        live_entry = self._add_labeled_entry(model, 1, 0, "Live status", self.adventure_live_status_var, width=54)
        live_entry.configure(state="readonly")

        main = ttk.Frame(parent)
        main.grid(row=1, column=0, sticky="nsew", padx=8, pady=3)
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)

        settings = ttk.LabelFrame(main, text="Adventure Settings")
        settings.grid(row=0, column=0, sticky="nsew", padx=(0, 4), pady=0)
        for column in (1, 3):
            settings.columnconfigure(column, weight=1)
        self._add_labeled_entry(settings, 0, 0, "Seed list", self.adventure_seed_list_var, width=42, columnspan=3)
        self._add_labeled_entry(settings, 1, 0, "Plant types", self.adventure_plant_types_var, width=42, columnspan=3)
        self._add_labeled_entry(settings, 2, 0, "Max levels", self.adventure_max_levels_var)
        self._add_labeled_entry(settings, 2, 2, "Max attempts", self.adventure_max_attempts_var)
        self._add_labeled_entry(settings, 3, 0, "Episodes", self.adventure_episodes_var)
        self._add_labeled_entry(settings, 3, 2, "Wins to advance", self.adventure_advance_wins_var)
        self._add_labeled_entry(settings, 4, 0, "Game speed", self.adventure_game_speed_var)
        self._add_labeled_entry(settings, 4, 2, "Step seconds", self.adventure_step_seconds_var)
        self._add_labeled_entry(settings, 5, 0, "Soft Max Steps", self.adventure_soft_max_steps_var)
        self._add_labeled_entry(settings, 5, 2, "Hard Max Steps", self.adventure_hard_max_steps_var)
        self._add_labeled_entry(settings, 6, 0, "Board timeout", self.adventure_board_timeout_var)

        ttk.Checkbutton(settings, text="adventure eval", variable=self.adventure_eval_var).grid(row=7, column=0, sticky="w", padx=6, pady=3)
        ttk.Checkbutton(settings, text="advance on wins", variable=self.adventure_advance_on_wins_var).grid(row=7, column=1, sticky="w", padx=6, pady=3)
        ttk.Checkbutton(settings, text="auto seeds", variable=self.adventure_auto_select_seeds_var).grid(row=7, column=2, sticky="w", padx=6, pady=3)
        ttk.Checkbutton(settings, text="wait gameplay", variable=self.adventure_wait_gameplay_ready_var).grid(row=7, column=3, sticky="w", padx=6, pady=3)
        ttk.Checkbutton(settings, text="quick wait", variable=self.adventure_quick_wait_var).grid(row=8, column=0, sticky="w", padx=6, pady=3)
        ttk.Checkbutton(settings, text="final-wave extension", variable=self.adventure_final_wave_extension_var).grid(row=8, column=1, sticky="w", padx=6, pady=3)
        ttk.Checkbutton(settings, text="tactical masks", variable=self.adventure_tactical_masks_var).grid(row=9, column=0, sticky="w", padx=6, pady=3)
        ttk.Checkbutton(settings, text="WallNut mask", variable=self.adventure_wallnut_mask_var).grid(row=9, column=1, sticky="w", padx=6, pady=3)
        ttk.Checkbutton(settings, text="CherryBomb mask", variable=self.adventure_cherrybomb_mask_var).grid(row=9, column=2, sticky="w", padx=6, pady=3)
        ttk.Label(settings, text="fusion").grid(row=8, column=2, sticky="w", padx=6, pady=3)
        ttk.Combobox(
            settings,
            textvariable=self.adventure_fusion_policy_var,
            values=FUSION_POLICY_CHOICES,
            state="readonly",
            width=10,
        ).grid(row=8, column=3, sticky="w", padx=6, pady=3)

        status = ttk.LabelFrame(main, text="Adventure Live Summary")
        status.grid(row=0, column=1, sticky="nsew", padx=(4, 0), pady=0)
        status.rowconfigure(0, weight=1)
        status.columnconfigure(0, weight=1)
        self.adventure_status_text = scrolledtext.ScrolledText(status, height=12, wrap="word", font=("Consolas", 9))
        self.adventure_status_text.grid(row=0, column=0, sticky="nsew")
        self.adventure_status_text.configure(state="disabled")

        actions = ttk.Frame(parent)
        actions.grid(row=2, column=0, sticky="ew", padx=8, pady=3)
        self._add_launch_button(actions, "Start Adventure Eval", self.start_adventure_eval).grid(row=0, column=0, sticky="w", padx=(0, 5))
        self._add_stop_button(actions, "Stop Run").grid(row=0, column=1, sticky="w", padx=(0, 5))
        ttk.Button(actions, text="Open Run Folder", command=self.open_adventure_run_folder).grid(row=0, column=2, sticky="w", padx=(0, 5))
        ttk.Button(actions, text="Refresh Diagnostics Now", command=self.refresh_diagnostics_now).grid(row=0, column=3, sticky="w", padx=(0, 5))

        preview = ttk.LabelFrame(parent, text="Command Preview")
        preview.grid(row=3, column=0, sticky="nsew", padx=8, pady=(3, 6))
        preview.rowconfigure(0, weight=1)
        preview.columnconfigure(0, weight=1)
        self.adventure_preview = self._preview_box(preview, height=3)

    def _build_adventure_generalist_tab(self, parent: ttk.Frame) -> None:
        scroll_content = self._make_scrollable_container(parent)
        scroll_content.columnconfigure(0, weight=1)

        schema = ttk.LabelFrame(scroll_content, text="Adventure Generalist 14-Slot Identity")
        schema.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 3))
        schema.columnconfigure(1, weight=1)
        ttk.Label(schema, text=f"family: {ADVENTURE_GENERALIST_MODEL_FAMILY}").grid(
            row=0, column=0, columnspan=4, sticky="w", padx=6, pady=2
        )
        ttk.Label(schema, text="schema: action_count=701, wait=0, slots=14, observation=adventure_14slot_identity_v1").grid(
            row=1, column=0, columnspan=4, sticky="w", padx=6, pady=2
        )
        self._add_labeled_entry(schema, 2, 0, "Initial loadout", self.generalist_initial_loadout_var, width=52, columnspan=3)
        max_slots_entry = self._add_labeled_entry(schema, 3, 0, "Max seed slots", self.generalist_max_seed_slots_var)
        max_slots_entry.configure(state="readonly")

        launch = ttk.LabelFrame(scroll_content, text="Launch Controls")
        launch.grid(row=1, column=0, sticky="ew", padx=8, pady=3)
        launch.columnconfigure(0, weight=1)
        launch.columnconfigure(1, weight=1)
        launch.columnconfigure(2, weight=1)
        ttk.Label(launch, text="Starts a fresh 14-slot Adventure Generalist training run.").grid(
            row=0, column=0, columnspan=3, sticky="w", padx=6, pady=(2, 2)
        )
        self._add_launch_button(launch, "Start Adventure Generalist Train", self.start_adventure_generalist_train).grid(
            row=1, column=0, sticky="w", padx=6, pady=(0, 4)
        )
        self._add_stop_button(launch, "Stop Run").grid(row=1, column=1, sticky="w", padx=6, pady=(0, 4))

        settings = ttk.LabelFrame(scroll_content, text="Basic Training Settings")
        settings.grid(row=2, column=0, sticky="ew", padx=8, pady=3)
        for column in (1, 3):
            settings.columnconfigure(column, weight=1)
        self._add_labeled_entry(settings, 0, 0, "Timesteps", self.generalist_total_timesteps_var)
        self._add_labeled_entry(settings, 0, 2, "Checkpoint freq", self.generalist_checkpoint_freq_var)
        self._add_labeled_entry(settings, 1, 0, "Start level", self.generalist_start_level_var)
        self._add_labeled_entry(settings, 1, 2, "Max levels", self.generalist_max_levels_var)
        self._add_labeled_entry(settings, 2, 0, "Max attempts", self.generalist_max_attempts_var)
        self._add_labeled_entry(settings, 2, 2, "Game speed", self.generalist_game_speed_var)
        self._add_labeled_entry(settings, 3, 0, "Step seconds", self.generalist_step_seconds_var)
        self._add_labeled_entry(settings, 3, 2, "Board timeout", self.generalist_board_timeout_var)
        self._add_labeled_entry(settings, 4, 0, "Soft Max Steps", self.generalist_soft_max_steps_var)
        self._add_labeled_entry(settings, 4, 2, "Hard Max Steps", self.generalist_hard_max_steps_var)
        self._add_labeled_entry(settings, 5, 0, "Run dir", self.generalist_run_dir_var, width=48, columnspan=3)
        self._add_labeled_entry(settings, 6, 0, "Eval model .zip", self.generalist_eval_model_path_var, width=44, columnspan=2)
        ttk.Button(settings, text="Browse", command=self.browse_generalist_eval_model).grid(row=6, column=3, sticky="w", padx=(0, 5), pady=2)
        ttk.Button(settings, text="Refresh", command=self.refresh_generalist_eval_model).grid(row=6, column=4, sticky="w", padx=(0, 6), pady=2)
        self._add_labeled_entry(settings, 7, 0, "Eval episodes", self.generalist_eval_episodes_var)
        self._add_launch_button(settings, "Start Adventure Generalist Eval", self.start_adventure_generalist_eval).grid(
            row=7, column=2, sticky="w", padx=6, pady=2
        )

        curriculum = ttk.LabelFrame(scroll_content, text="Advanced Curriculum / Unlock Behavior")
        curriculum.grid(row=3, column=0, sticky="ew", padx=8, pady=3)
        for column in (1, 3):
            curriculum.columnconfigure(column, weight=1)
        ttk.Checkbutton(curriculum, text="unlock-aware", variable=self.generalist_unlock_curriculum_var).grid(row=0, column=0, sticky="w", padx=6, pady=3)
        ttk.Checkbutton(curriculum, text="replay cleared", variable=self.generalist_replay_cleared_var).grid(row=0, column=1, sticky="w", padx=6, pady=3)
        ttk.Label(curriculum, text="curriculum").grid(row=0, column=2, sticky="w", padx=6, pady=3)
        ttk.Combobox(
            curriculum,
            textvariable=self.generalist_curriculum_var,
            values=("conservative", "varied"),
            state="readonly",
            width=14,
        ).grid(row=0, column=3, sticky="w", padx=6, pady=3)
        self._add_labeled_entry(curriculum, 1, 0, "Unlock delay", self.generalist_unlock_delay_var)
        self._add_labeled_entry(curriculum, 1, 2, "New plant min prob", self.generalist_new_plant_prob_var)
        self._add_labeled_entry(curriculum, 2, 0, "Frontier prob", self.generalist_frontier_prob_var)
        self._add_labeled_entry(curriculum, 2, 2, "Recent prob", self.generalist_recent_prob_var)
        self._add_labeled_entry(curriculum, 3, 0, "Maintenance prob", self.generalist_maintenance_prob_var)
        self._add_labeled_entry(
            curriculum,
            3,
            2,
            "Wins in a row before level promotion",
            self.generalist_frontier_win_streak_required_var,
        )

        masks = ttk.LabelFrame(scroll_content, text="Masks / Timeouts")
        masks.grid(row=4, column=0, sticky="ew", padx=8, pady=3)
        ttk.Checkbutton(masks, text="quick wait", variable=self.generalist_quick_wait_var).grid(row=0, column=0, sticky="w", padx=6, pady=3)
        ttk.Checkbutton(masks, text="wait gameplay", variable=self.generalist_wait_gameplay_ready_var).grid(row=0, column=1, sticky="w", padx=6, pady=3)
        ttk.Checkbutton(masks, text="final-wave extension", variable=self.generalist_final_wave_extension_var).grid(row=0, column=2, sticky="w", padx=6, pady=3)
        ttk.Checkbutton(masks, text="tactical masks", variable=self.generalist_tactical_masks_var).grid(row=1, column=0, sticky="w", padx=6, pady=3)
        ttk.Checkbutton(masks, text="WallNut mask", variable=self.generalist_wallnut_mask_var).grid(row=1, column=1, sticky="w", padx=6, pady=3)
        ttk.Checkbutton(masks, text="CherryBomb mask", variable=self.generalist_cherrybomb_mask_var).grid(row=1, column=2, sticky="w", padx=6, pady=3)

        preview = ttk.LabelFrame(scroll_content, text="Command Preview")
        preview.grid(row=5, column=0, sticky="ew", padx=8, pady=(3, 3))
        preview.rowconfigure(0, weight=1)
        preview.columnconfigure(0, weight=1)
        self.generalist_preview = self._preview_box(preview, height=3)

        status = ttk.LabelFrame(scroll_content, text="Adventure Generalist Live Status")
        status.grid(row=6, column=0, sticky="ew", padx=8, pady=(3, 6))
        status.rowconfigure(0, weight=1)
        status.columnconfigure(0, weight=1)
        self.generalist_status_text = scrolledtext.ScrolledText(status, height=6, wrap="word", font=("Consolas", 9))
        self.generalist_status_text.grid(row=0, column=0, sticky="nsew")
        self.generalist_status_text.configure(state="disabled")

    def _set_text_widget(self, widget: tk.Text, content: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", content)
        widget.configure(state="disabled")

    def start_training(self) -> None:
        if self.fast_only_var.get():
            self._append_log("Note: --fast-only omitted from training; train_ppo.py does not support it.\n")
        self.launch_process("Start Training", self._build_train_command(resume=False))

    def resume_training(self) -> None:
        raw_model_path = self.model_path_var.get().strip()
        if not raw_model_path:
            self._append_log("ERROR: Resume requires model_path.\n")
            return
        model_path = self._resolve_text_path(raw_model_path)
        if not model_path.exists():
            self._append_log(f"ERROR: Resume model does not exist: {model_path}\n")
            return
        if self.fast_only_var.get():
            self._append_log("Note: --fast-only omitted from resume; train_ppo.py does not support it.\n")
        self.launch_process("Resume Training", self._build_train_command(resume=True))

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
        self._append_log("Launching Adventure Generalist training...\n")
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

    def launch_process(self, name: str, command: List[str]) -> None:
        if self.active_process is not None:
            self._append_log(f"ERROR: Cannot launch {name}; {self.active_process_name} is still running.\n")
            return
        self._append_log(f"\nStarting subprocess: {name}\n")
        model_path = self._command_arg(command, "--model-path")
        run_dir = self._command_arg(command, "--run-dir")
        live_status_arg = self._command_arg(command, "--live-status-path")
        if model_path:
            self._append_log(f"Selected model path: {model_path}\n")
        if live_status_arg:
            try:
                resolved_live_status = self._resolve_text_path(live_status_arg)
            except (OSError, ValueError):
                resolved_live_status = Path(live_status_arg)
            self._append_log(f"[gui] live status path: {resolved_live_status}\n")
        if run_dir:
            self._append_log(f"Selected run directory: {run_dir}\n")
            self.active_run_path = str(self._resolve_text_path(run_dir))
            self.active_run_var.set(f"Active run: {self.active_run_path}")
        elif ("--adventure-eval" in command or "--adventure-generalist-eval" in command) and model_path:
            self.active_run_path = str(self._resolve_text_path(model_path).parent)
            self.active_run_var.set(f"Active run: {self.active_run_path}")
            self._append_log(f"Inferred Adventure run directory: {self.active_run_path}\n")
        else:
            self.active_run_path = ""
            self.active_run_var.set("Active run: unknown")
        self._append_log(f"Launching with cwd: {self.project_root}\n")
        self._append_log(f"$ {subprocess.list2cmdline(command)}\n")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            process = subprocess.Popen(
                command,
                cwd=str(self.project_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            self._append_log(f"ERROR: Failed to launch {name}: {exc}\n")
            self.process_status_var.set("Idle")
            return

        self.active_process = process
        self.active_process_name = name
        self.active_process_started_at = time.monotonic()
        self.live_writer_warning_emitted = False
        self.process_status_var.set(f"Running: {name}")
        self._set_running(True)
        thread = threading.Thread(target=self._read_process_output, args=(name, process), name=f"{name} log reader", daemon=True)
        thread.start()

    def _command_arg(self, command: List[str], flag: str) -> str:
        try:
            index = command.index(flag)
        except ValueError:
            return ""
        next_index = index + 1
        if next_index >= len(command):
            return ""
        return command[next_index]

    def _read_process_output(self, name: str, process: subprocess.Popen[str]) -> None:
        try:
            if process.stdout is not None:
                for line in process.stdout:
                    self.log_queue.put(line)
        finally:
            exit_code = process.wait()
            self.log_queue.put(("process_exit", name, process, exit_code))

    def stop_active_process(self) -> None:
        process = self.active_process
        if process is None:
            self._append_log("No active process to stop.\n")
            return
        name = self.active_process_name or "process"
        for button in self.stop_buttons:
            button.configure(state="disabled")
        try:
            process.terminate()
            self._append_log(f"[{name}] terminate() sent.\n")
        except OSError as exc:
            self._append_log(f"ERROR: Failed to terminate {name}: {exc}\n")
            if process.poll() is None:
                for button in self.stop_buttons:
                    button.configure(state="normal")
            return
        thread = threading.Thread(target=self._wait_then_kill, args=(name, process), name=f"{name} stopper", daemon=True)
        thread.start()

    def _wait_then_kill(self, name: str, process: subprocess.Popen[str]) -> None:
        try:
            process.wait(timeout=STOP_GRACE_SECONDS)
            self.log_queue.put(f"[{name}] process exited after terminate().\n")
        except subprocess.TimeoutExpired:
            self.log_queue.put(f"[{name}] terminate() timed out; kill() used.\n")
            try:
                process.kill()
                process.wait()
            except OSError as exc:
                self.log_queue.put(f"ERROR: Failed to kill {name}: {exc}\n")

    def _drain_log_queue(self) -> None:
        try:
            while True:
                item = self.log_queue.get_nowait()
                if isinstance(item, tuple) and item and item[0] == "process_exit":
                    _, name, process, exit_code = item
                    self._handle_process_exit(str(name), process, int(exit_code))
                else:
                    self._append_log(str(item))
        except queue.Empty:
            pass
        self.root.after(LOG_POLL_MS, self._drain_log_queue)

    def _handle_process_exit(self, name: str, process: subprocess.Popen[str], exit_code: int) -> None:
        self._append_log(f"[{name}] exited with code {exit_code}\n")
        if self.active_process is process:
            self.active_process = None
            self.active_process_name = ""
            self.active_process_started_at = None
            self.live_writer_warning_emitted = False
            self.process_status_var.set("Idle")
            self._set_running(False)

    def _set_running(self, running: bool) -> None:
        state = "disabled" if running else "normal"
        for button in self.launch_buttons:
            button.configure(state=state)
        for button in self.stop_buttons:
            button.configure(state="normal" if running else "disabled")

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

    def browse_run_folder(self) -> None:
        initial_dir = self.repo_root / "runs"
        dirname = filedialog.askdirectory(
            title="Select run folder",
            initialdir=str(initial_dir if initial_dir.exists() else self.repo_root),
        )
        if dirname:
            self.run_dir_var.set(dirname)
            self._append_log(f"Selected run directory: {dirname}\n")

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
        run_dir = self.run_dir_var.get().strip()
        model_path = self.model_path_var.get().strip()
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
        if self.log_text is None:
            return
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _append_log(self, text: str) -> None:
        if self.log_text is None:
            return
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _set_panel(self, title: str, content: str) -> None:
        text = self.panels.get(title)
        if text is None:
            return
        if self.last_panel_content.get(title) == content:
            return
        self.last_panel_content[title] = content
        text.configure(state="normal")
        text.delete("1.0", "end")
        text.insert("1.0", content)
        text.configure(state="disabled")

    def _set_adventure_status(self, content: str) -> None:
        if self.adventure_status_text is None:
            return
        if self.last_adventure_status_content == content:
            return
        self.last_adventure_status_content = content
        self.adventure_status_text.configure(state="normal")
        self.adventure_status_text.delete("1.0", "end")
        self.adventure_status_text.insert("1.0", content)
        self.adventure_status_text.configure(state="disabled")

    def _set_generalist_status(self, content: str) -> None:
        if self.generalist_status_text is None:
            return
        if self.last_generalist_status_content == content:
            return
        self.last_generalist_status_content = content
        self.generalist_status_text.configure(state="normal")
        self.generalist_status_text.delete("1.0", "end")
        self.generalist_status_text.insert("1.0", content)
        self.generalist_status_text.configure(state="disabled")

    def _poll(self) -> None:
        self._refresh_diagnostics(auto=True)
        self.root.after(POLL_MS, self._poll)

    def refresh_diagnostics_now(self) -> None:
        self._refresh_diagnostics(auto=False)

    def _refresh_diagnostics(self, auto: bool) -> None:
        del auto
        try:
            payload, info = self._read_live_status_file()
            if payload is not None:
                self.last_good_status = payload
                self.last_good_read_time = time.time()
                self.last_live_parse_error = ""
            elif info.get("parse_error"):
                self.last_live_parse_error = str(info["parse_error"])

            using_last_good = payload is None and self.last_good_status is not None
            display_payload = payload if payload is not None else self.last_good_status
            self._set_live_status(info)
            self._set_diagnostics_status(info, using_last_good=using_last_good)
            self._render_diagnostics_payload(display_payload, str(info["health"]), using_last_good)
            self._log_live_health_change(info)
            self._maybe_warn_live_writer(info)
        except Exception as exc:
            self.last_live_parse_error = f"GUI diagnostics error: {exc}"
            info = {
                "path": self.live_status_path,
                "exists": False,
                "size": None,
                "mtime": None,
                "age": None,
                "health": "MALFORMED",
                "parse_error": self.last_live_parse_error,
            }
            try:
                info["exists"] = self.live_status_path.exists()
            except OSError:
                pass
            self._set_live_status(info)
            self._set_diagnostics_status(info, using_last_good=self.last_good_status is not None)
            self._render_no_status("MALFORMED")
            self._log_live_warning(
                f"gui_diagnostics_error:{type(exc).__name__}",
                f"Live diagnostics GUI error: {exc}\n",
            )

    def _read_live_status_file(self) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        path = self.live_status_path
        info: Dict[str, Any] = {
            "path": path,
            "exists": False,
            "size": None,
            "mtime": None,
            "age": None,
            "health": "MISSING",
            "parse_error": "",
        }
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None, info
        except OSError as exc:
            info["health"] = "MALFORMED"
            info["parse_error"] = f"OSError while stat() reading live status: {exc}"
            return None, info

        age = max(0.0, time.time() - stat.st_mtime)
        info.update({"exists": True, "size": int(stat.st_size), "mtime": stat.st_mtime, "age": age})
        if stat.st_size == 0:
            info["health"] = "EMPTY"
            return None, info

        try:
            with path.open("r", encoding="utf-8") as handle:
                content = handle.read()
        except FileNotFoundError:
            info.update({"exists": False, "size": None, "mtime": None, "age": None, "health": "MISSING"})
            return None, info
        except OSError as exc:
            info["health"] = "MALFORMED"
            info["parse_error"] = f"OSError while reading live status: {exc}"
            return None, info

        if not content.strip():
            info["health"] = "EMPTY"
            return None, info

        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            info["health"] = "MALFORMED"
            info["parse_error"] = f"{exc.msg} at line {exc.lineno}, column {exc.colno}"
            return None, info

        if not isinstance(payload, dict):
            info["health"] = "MALFORMED"
            info["parse_error"] = f"Expected JSON object, got {type(payload).__name__}"
            return None, info

        info["health"] = self._live_health(age, payload)
        return payload, info

    def _live_health(self, age: float, payload: Optional[Dict[str, Any]] = None) -> str:
        payload = payload if isinstance(payload, dict) else {}
        blocked_reason = str(
            self._first_value(
                payload,
                ["blocked_reason", "adventure.blocked_reason", "post_win_blocked_reason", "adventure.post_win_blocked_reason"],
                default="",
            )
            or ""
        )
        if blocked_reason.startswith("post_win_") or blocked_reason in {
            "trophy_visible_but_click_failed",
            "reward_or_unlock_click_failed_after_win",
        }:
            return "BLOCKED_POST_WIN"
        if "seed_selection" in blocked_reason:
            return "BLOCKED_SEED_SELECTION"
        if "gameplay_ready" in blocked_reason or "gameplay" in blocked_reason:
            return "BLOCKED_GAMEPLAY_READY"
        if age < LIVE_MAX_AGE_SECONDS:
            return "LIVE"
        if age <= STALE_MAX_AGE_SECONDS:
            return "STALE"
        return "DEAD"

    def _set_live_status(self, info: Dict[str, Any]) -> None:
        path = info.get("path", self.live_status_path)
        exists_text = "yes" if info.get("exists") else "no"
        size_text = self._fmt_size(info.get("size"))
        age_text = self._fmt_age(info.get("age"))
        health = info.get("health", "MISSING")
        self.live_status_var.set(f"Live: {path} | exists={exists_text} | size={size_text} | age={age_text} | health={health}")

    def _set_diagnostics_status(self, info: Dict[str, Any], using_last_good: bool) -> None:
        health = str(info.get("health", "MISSING"))
        if health == "LIVE":
            prefix = "LIVE"
        elif health.startswith("BLOCKED_"):
            prefix = health
        elif using_last_good or (health in {"STALE", "DEAD"} and self.last_good_status is not None):
            prefix = f"{health} - showing last good values"
        else:
            prefix = f"{health} - no live values available"
        parse_error = self.last_live_parse_error or str(info.get("parse_error") or "")
        parse_text = parse_error if parse_error else "-"
        if len(parse_text) > 180:
            parse_text = parse_text[:177] + "..."
        self.diagnostics_status_var.set(
            f"{prefix} | modified={self._fmt_time(info.get('mtime'))} | "
            f"last_success={self._fmt_time(self.last_good_read_time)} | parse_error={parse_text}"
        )

    def _log_live_warning(self, key: str, message: str) -> None:
        if key == self.last_live_warning_key:
            return
        self.last_live_warning_key = key
        self._append_log(message)

    def _log_live_health_change(self, info: Dict[str, Any]) -> None:
        health = str(info.get("health", "MISSING"))
        if health == self.last_live_health:
            return
        previous = self.last_live_health
        self.last_live_health = health
        age_text = self._fmt_age(info.get("age"))
        size_text = self._fmt_size(info.get("size"))
        path = info.get("path", self.live_status_path)
        if previous:
            self._append_log(f"Live status health changed: {previous} -> {health}: {path} age={age_text} size={size_text}\n")
        else:
            self._append_log(f"Live status health: {health}: {path} age={age_text} size={size_text}\n")

    def _maybe_warn_live_writer(self, info: Dict[str, Any]) -> None:
        if not self._active_process_is_running():
            return
        if str(info.get("health")) == "LIVE" or str(info.get("health", "")).startswith("BLOCKED_"):
            self.live_writer_warning_emitted = False
            return
        started_at = self.active_process_started_at
        if started_at is None or time.monotonic() - started_at < 10.0:
            return
        health = str(info.get("health", "MISSING"))
        age = info.get("age")
        stale_long_enough = isinstance(age, (int, float)) and float(age) >= 10.0
        not_updating = health in {"MISSING", "EMPTY", "MALFORMED", "DEAD"} or (health == "STALE" and stale_long_enough)
        if not not_updating or self.live_writer_warning_emitted:
            return
        self.live_writer_warning_emitted = True
        self._append_log(
            "Warning: active process is running, but live_status.json is not updating. "
            "Check writer path or whether the training/eval code emits live diagnostics.\n"
        )

    def _active_process_is_running(self) -> bool:
        return self.active_process is not None and self.active_process.poll() is None

    def _fmt_age(self, age: Any) -> str:
        return "-" if not isinstance(age, (int, float)) else f"{float(age):.1f}s"

    def _fmt_size(self, size: Any) -> str:
        return "-" if not isinstance(size, int) else str(size)

    def _fmt_time(self, timestamp: Any) -> str:
        if not isinstance(timestamp, (int, float)):
            return "-"
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(timestamp)))

    def show_raw_live_json(self) -> None:
        self._refresh_diagnostics(auto=False)
        if self.last_good_status is None:
            content = "No live status available."
        else:
            note = ""
            if self.last_live_health == "MALFORMED":
                note = "Current live_status.json is malformed; showing last successfully parsed JSON.\n\n"
            elif self.last_live_health in {"MISSING", "EMPTY"}:
                note = f"Current live_status.json is {self.last_live_health}; showing last successfully parsed JSON.\n\n"
            content = note + json.dumps(self.last_good_status, indent=2, sort_keys=True)
        popup = tk.Toplevel(self.root)
        popup.title("Raw Live JSON")
        popup.geometry("720x520")
        popup.minsize(520, 320)
        popup.rowconfigure(0, weight=1)
        popup.columnconfigure(0, weight=1)
        box = scrolledtext.ScrolledText(popup, wrap="none", font=("Consolas", 9))
        box.grid(row=0, column=0, sticky="nsew")
        box.insert("1.0", content)
        box.configure(state="disabled")

    def _render_no_status(self, health: str) -> None:
        self.schema_keys_var.set("Schema keys: -")
        self.active_run_var.set(f"Active run: {self.active_run_path or 'unknown'}")
        message = f"No live status available ({health})."
        self._set_adventure_status(self._adventure_status_content({}, health=health, using_last_good=False))
        self._set_generalist_status(self._generalist_status_content({}, health=health, using_last_good=False))
        for title in list(self.panels):
            if title == "Rows":
                self._set_panel(title, "No row diagnostics in live_status.json")
            else:
                self._set_panel(title, message)

    def _render_diagnostics_payload(self, payload: Optional[Dict[str, Any]], health: str, using_last_good: bool) -> None:
        try:
            if payload is None:
                self._render_no_status(health)
            else:
                self._render(payload, health=health, using_last_good=using_last_good)
        except Exception as exc:
            self._render_no_status(f"{health}; render error")
            self._log_live_warning(
                f"diagnostics_render_error:{type(exc).__name__}",
                f"Live diagnostics render error: {exc}\n",
            )

    def _render(self, payload: Dict[str, Any], health: str = "LIVE", using_last_good: bool = False) -> None:
        if not isinstance(payload, dict):
            payload = {}
        self.schema_keys_var.set("Schema keys: " + self._top_level_keys(payload))
        self.active_run_var.set("Active run: " + self._active_run_text(payload))
        screen = self._as_dict(self._lookup_path(payload, "screen"))
        unlock = self._as_dict(self._lookup_path(payload, "unlock"))
        seed_inventory = self._as_dict(self._lookup_path(payload, "seed_inventory"))
        compatibility = self._as_dict(self._lookup_path(payload, "compatibility"))
        model_compatibility = self._as_dict(self._lookup_path(payload, "model_compatibility"))
        if model_compatibility:
            compatibility = {**compatibility, **model_compatibility}
        self._set_adventure_status(self._adventure_status_content(payload, health=health, using_last_good=using_last_good))
        self._set_generalist_status(self._generalist_status_content(payload, health=health, using_last_good=using_last_good))

        self._set_panel(
            "Adventure",
            lines_from_pairs(
                [
                    ("state", self._first_value(payload, ["adventure.state", "state", "screenState", "phase"])),
                    ("level", self._first_value(payload, ["adventure.level", "level", "adventure_level"])),
                    ("attempt", self._first_value(payload, ["adventure.attempt", "attempt", "level_attempt"])),
                    ("wins", self._first_value(payload, ["adventure.wins", "adventure.wins_this_level", "wins", "eval.wins"])),
                    ("losses", self._first_value(payload, ["adventure.losses", "losses", "eval.losses"])),
                    ("consecutive", self._first_value(payload, ["adventure.consecutive", "adventure.consecutive_wins", "consecutive_wins", "consecutive"])),
                    ("advance_on", self._first_value(payload, ["adventure.advance_on", "adventure.advance_on_wins", "advance_on_wins"])),
                    ("remaining", self._first_value(payload, ["adventure.required_consecutive_wins_remaining", "required_consecutive_wins_remaining"])),
                    ("advanced", self._first_value(payload, ["adventure.advanced", "advanced"])),
                    ("last_result", self._first_value(payload, ["adventure.last_result", "last_result"])),
                    ("blocked", self._first_value(payload, ["adventure.blocked_reason", "blocked_reason"])),
                    ("stage", self._first_value(payload, ["adventure.current_stage", "current_stage", "compatibility.stage"])),
                    ("family", self._first_value(payload, ["adventure.current_model_family", "current_model_family", "compatibility.model_family"])),
                    ("win_detected", self._first_value(payload, ["adventure.win_detected", "win_detected"])),
                    ("trophy_visible", self._first_value(payload, ["adventure.trophy_visible", "screen.trophy_visible", "trophy_visible"])),
                    ("trophy_clicked", self._first_value(payload, ["adventure.trophy_clicked", "trophy_clicked"])),
                    ("post_win", self._first_value(payload, ["adventure.post_win_transition_completed", "post_win_transition_completed"])),
                    ("post_win_block", self._first_value(payload, ["adventure.post_win_blocked_reason", "post_win_blocked_reason"])),
                    ("post_win_active", self._first_value(payload, ["adventure.post_win_active", "post_win_active"])),
                    ("post_win_elapsed", self._first_value(payload, ["adventure.post_win_elapsed", "post_win_elapsed"])),
                    ("post_win_clicks", self._first_value(payload, ["adventure.post_win_click_attempts", "post_win_click_attempts"])),
                    ("post_win_target", self._first_value(payload, ["adventure.post_win_last_click_target", "post_win_last_click_target"])),
                    ("post_win_click_ok", self._first_value(payload, ["adventure.post_win_last_click_ok", "post_win_last_click_ok"])),
                    ("post_win_state", self._first_value(payload, ["adventure.post_win_last_state.screenState", "post_win_last_state.screenState"])),
                    ("latest_unlock", self._first_value(payload, ["adventure.latest_unlock", "unlock.latest_unlock", "latest_unlock"])),
                    ("unlocked", self._first_value(payload, ["adventure.unlocked_seeds", "seed_inventory.unlocked_seeds", "unlocked_seeds"])),
                    ("selected", self._first_value(payload, ["adventure.selected_seeds", "seed_inventory.selected_seeds", "selected_seeds"])),
                    ("available", self._first_value(payload, ["adventure.available_seeds", "seed_inventory.available_seeds", "available_seeds"])),
                    ("missing_unlock", self._first_value(payload, ["adventure.missing_required_unlocked", "seed_inventory.missing_required_unlocked"])),
                    ("missing_avail", self._first_value(payload, ["adventure.missing_required_available", "seed_inventory.missing_required_available"])),
                    ("conservative", self._first_value(payload, ["adventure.conservative_seeds", "seed_inventory.conservative_seeds"])),
                    ("allow_new", self._first_value(payload, ["adventure.allow_new_plants", "seed_inventory.allow_new_plants"])),
                ]
            ),
        )
        self._set_panel(
            "Gameplay",
            lines_from_pairs(
                [
                    ("sun", self._first_value(payload, ["gameplay.sun", "current_sun", "sun", "game.sun"])),
                    ("wave", self._wave_text(payload)),
                    ("plants", self._first_value(payload, ["gameplay.plants", "current_plants", "plants", "plant_count", "plantCount"])),
                    ("zombies", self._first_value(payload, ["gameplay.zombies", "current_zombies", "zombies", "zombie_count", "zombieCount"])),
                    ("mowers_lost", self._first_value(payload, ["gameplay.mowers_lost", "mowers_lost"])),
                    ("ready", self._first_value(payload, ["gameplay.ready", "gameplay.gameplay_ready", "gameplayReady", "gameplay_ready"])),
                    ("screen", self._first_value(payload, ["gameplay.screen", "gameplay.screen_state", "screen", "screenState", "terminalHint"])),
                    ("startup_popup", screen.get("startup_popup_visible")),
                    ("ok_button", screen.get("startup_ok_button_visible")),
                    ("menu_blocked", screen.get("main_menu_blocked_by_popup")),
                    ("reward_screen", screen.get("reward_screen_visible")),
                    ("unlock_screen", screen.get("unlock_screen_visible")),
                    ("trophy", screen.get("trophy_visible")),
                    ("post_win_click", screen.get("post_win_click_required")),
                    ("new_plant_ui", screen.get("new_plant_unlocked_visible")),
                    ("new_plant", unlock.get("new_plant_unlocked_name")),
                    ("new_plant_type", unlock.get("new_plant_unlocked_plant_type")),
                    ("unknown_cards", self._safe_len(unlock.get("unknown_visible_seed_cards"))),
                    ("unknown_unlocks", self._safe_len(unlock.get("unknown_unlock_objects"))),
                ]
            ),
        )
        self._set_panel(
            "Agent",
            lines_from_pairs(
                [
                    ("timestep", self._first_value(payload, ["train.total_timesteps", "current_timestep", "total_timesteps"])),
                    ("episode", self._first_value(payload, ["train.current_episode", "current_episode", "summary.episode", "eval.episode"])),
                    ("episode_step", self._first_value(payload, ["train.current_step", "agent.episode_step", "current_step", "summary.episode_length", "eval.episode_length", "episode_length"])),
                    ("last_action", self._first_value(payload, ["agent.last_action", "last_action", "action"])),
                    ("bridge_action", self._first_value(payload, ["agent.bridge_action", "bridge_action"])),
                    ("type", self._first_value(payload, ["agent.type", "agent.decoded_action.type", "action_type", "last_action_type"])),
                    ("plant", self._first_value(payload, ["agent.plant", "agent.decoded_action.plant", "plant", "plant_type"])),
                    ("seed_slot", self._first_value(payload, ["agent.seed_slot", "agent.decoded_action.seed_slot", "seed_slot"])),
                    ("row", self._first_value(payload, ["agent.row", "agent.decoded_action.row", "row"])),
                    ("col", self._first_value(payload, ["agent.col", "agent.decoded_action.col", "col"])),
                    ("legal_count", self._first_value(payload, ["agent.legal_count", "agent.legal_action_count", "legal_action_count", "legal_count"])),
                    ("tactical_mask", self._first_value(payload, ["tactical_mask_enabled", "summary.tactical_mask_enabled", "eval.tactical_mask_enabled"])),
                    ("wallnut_masked", self._first_value(payload, ["summary.wallnut_actions_masked", "eval.wallnut_actions_masked"])),
                    ("cherry_masked", self._first_value(payload, ["summary.cherrybomb_actions_masked", "eval.cherrybomb_actions_masked"])),
                    ("decoder", self._first_value(payload, ["agent.action_decoder_version", "action_decoder_version", "compatibility.action_decoder_version"])),
                    ("obs", self._first_value(payload, ["agent.observation_version", "observation_version", "compatibility.observation_version"])),
                ]
            ),
        )
        self._set_panel("Rows", self._row_panel_lines(payload))
        self._set_panel(
            "Reward",
            lines_from_pairs(
                [
                    ("episode", self._first_value(payload, ["reward.episode", "reward.episode_reward", "current_reward", "episode_reward", "reward_total"])),
                    ("kill", self._first_value(payload, ["reward.kill", "reward.kill_reward_total", "kill_reward_total"])),
                    ("wave", self._first_value(payload, ["reward.wave", "reward.wave_reward_total", "wave_reward_total"])),
                    ("win_loss", self._first_value(payload, ["reward.win_loss", "reward.win_loss_reward_total", "win_loss_reward_total"])),
                    ("plant_health", self._first_value(payload, ["reward.plant_health", "reward.plant_health_loss_penalty_total", "plant_health_loss_penalty_total"])),
                    ("undef_threat", self._first_value(payload, ["reward.undef_threat", "reward.undefended_threat_penalty_total", "undefended_threat_penalty_total"])),
                    ("first_def", self._first_value(payload, ["reward.first_def", "reward.first_defense_reward_total", "reward.first_defense_undefended_threatened_row_reward_total", "first_defense_reward_total", "first_def"])),
                    ("sun_undef", self._first_value(payload, ["reward.sunflower_while_undefended_threat_penalty_total", "sunflower_while_undefended_threat_penalty_total"])),
                    ("elsewhere", self._first_value(payload, ["reward.plant_elsewhere_while_undefended_threat_penalty_total", "plant_elsewhere_while_undefended_threat_penalty_total"])),
                    ("reduced", self._first_value(payload, ["reward.reduce_undefended_threat_reward_total", "reduce_undefended_threat_reward_total"])),
                ]
            ),
        )
        self._set_panel(
            "Eval",
            lines_from_pairs(
                [
                    ("timesteps", self._first_value(payload, ["train.total_timesteps", "eval.total_timesteps", "total_timesteps", "current_timestep"])),
                    ("target_steps", self._first_value(payload, ["train.target_timesteps", "config.total_timesteps", "target_timesteps"])),
                    ("episodes", self._first_value(payload, ["eval.episodes", "eval.episodes_completed", "summary.episodes", "episodes", "episode_count"])),
                    ("win_rate", self._first_value(payload, ["eval.win_rate", "win_rate"])),
                    ("avg_reward", self._first_value(payload, ["eval.avg_reward", "avg_reward"])),
                    ("avg_wave", self._first_value(payload, ["eval.avg_wave", "avg_wave"])),
                    ("avg_kills", self._first_value(payload, ["eval.avg_kills", "avg_kills"])),
                    ("avg_plants", self._first_value(payload, ["eval.avg_plants", "avg_plants"])),
                    ("avg_mowers", self._first_value(payload, ["eval.avg_mowers", "eval.avg_mowers_lost", "avg_mowers", "avg_mowers_lost"])),
                    ("reset_fail", self._first_value(payload, ["eval.reset_failures", "reset_failures"])),
                    ("bridge_err", self._first_value(payload, ["eval.bridge_errors", "bridge_errors"])),
                    ("illegal", self._first_value(payload, ["eval.illegal_actions", "illegal_actions"])),
                ]
            ),
        )
        fusion = self._as_dict(self._lookup_path(payload, "fusion"))
        self._set_panel(
            "Fusion",
            lines_from_pairs(
                [
                    ("policy", self._first_value(payload, ["fusion.fusion_policy", "fusion_policy"])),
                    ("available", self._first_value(payload, ["fusion.fusion_available", "fusion_available"])),
                    ("candidates", self._first_value(payload, ["fusion.fusion_candidate_count", "fusion_candidate_count"])),
                    ("top", self._first_value(payload, ["fusion.fusion_top_candidate", "fusion_top_candidate"])),
                    ("attempts", self._first_value(payload, ["fusion.fusion_attempted_count", "fusion_attempted_count"])),
                    ("successes", self._first_value(payload, ["fusion.fusion_success_count", "fusion_success_count"])),
                    ("failures", self._first_value(payload, ["fusion.fusion_failed_count", "fusion_failed_count"])),
                    ("rejections", self._first_value(payload, ["fusion.fusion_rejected_count", "fusion_rejected_count"])),
                    ("last_reject", self._first_value(payload, ["fusion.fusion_last_rejected_reason", "fusion_last_rejected_reason"])),
                    ("reasons", fusion.get("fusion_rejected_reasons", payload.get("fusion_rejected_reasons"))),
                ]
            ),
        )
        self._set_panel(
            "Seed Inventory / Compatibility",
            lines_from_pairs(
                [
                    ("stage", compatibility.get("stage")),
                    ("family", compatibility.get("model_family")),
                    ("model", compatibility.get("model_path")),
                    ("mode", compatibility.get("action_space_mode")),
                    ("model_actions", compatibility.get("model_action_count")),
                    ("env_actions", compatibility.get("env_action_count", compatibility.get("action_count"))),
                    ("decoder", compatibility.get("action_decoder_version")),
                    ("observation", compatibility.get("observation_version")),
                    ("metadata", compatibility.get("metadata_path")),
                    ("compatible", compatibility.get("compatible")),
                    ("blocked", compatibility.get("blocked_reason")),
                    ("model_seeds", compatibility.get("model_seed_list")),
                    ("env_seeds", compatibility.get("env_seed_list")),
                    ("selected", self._first_value(payload, ["selected_loadout", "adventure.selected_loadout", "seed_inventory.selected_seeds"])),
                    ("active_slots", self._first_value(payload, ["active_seed_slot_count", "adventure.active_seed_slot_count"])),
                    ("eligible", self._first_value(payload, ["eligible_seeds", "adventure.eligible_seeds", "unlock.eligible_seeds"])),
                    ("unlocked", seed_inventory.get("unlocked_seeds")),
                    ("available", seed_inventory.get("available_seeds")),
                    ("missing_unlock", seed_inventory.get("missing_required_unlocked")),
                    ("missing_avail", seed_inventory.get("missing_required_available")),
                    ("selected_ratio", seed_inventory.get("selected_slot_ratio")),
                    ("available_ratio", seed_inventory.get("available_slot_ratio")),
                    ("required_avail", seed_inventory.get("required_available_ratio")),
                    ("legal_count", seed_inventory.get("legal_action_count")),
                    ("mask_blocks", seed_inventory.get("mask_block_reason_counts")),
                ]
            ),
        )

    def _adventure_status_content(self, payload: Dict[str, Any], health: str, using_last_good: bool) -> str:
        if not isinstance(payload, dict):
            payload = {}
        compatibility = self._as_dict(self._lookup_path(payload, "compatibility"))
        model_compatibility = self._as_dict(self._lookup_path(payload, "model_compatibility"))
        if model_compatibility:
            compatibility = {**compatibility, **model_compatibility}
        fusion = self._as_dict(self._lookup_path(payload, "fusion"))
        health_text = f"{health} (showing last good)" if using_last_good else health
        active_run = self._active_run_text(payload)
        model_path = self._adventure_first(
            payload,
            ["model_path", "adventure.current_model_path", "compatibility.model_path", "model_compatibility.model_path"],
        )
        return lines_from_pairs(
            [
                ("health", health_text),
                ("mode", self._adventure_first(payload, ["mode", "run_mode", "config.run_mode"])),
                ("run_mode", self._adventure_first(payload, ["run_mode", "config.run_mode", "mode"])),
                ("target_level", self._adventure_first(payload, ["target_level", "config.target_level"])),
                ("compatible", compatibility.get("compatible", "n/a")),
                ("blocked_reason", self._adventure_compat_blocked_reason(payload, compatibility)),
                ("stop_reason", self._adventure_first(payload, ["stop_reason", "adventure.stop_reason"])),
                ("model_path", self._path_for_display(model_path)),
                ("active_run", self._path_for_display(active_run)),
                ("model_family", self._adventure_first(payload, ["model_family", "adventure.current_model_family", "compatibility.model_family"])),
                ("action_count", self._adventure_first(payload, ["action_count", "compatibility.action_count", "compatibility.env_action_count"])),
                ("max_seed_slots", self._adventure_first(payload, ["max_seed_slots", "adventure.max_seed_slots", "seed_inventory.max_seed_slots"])),
                ("active_slots", self._adventure_first(payload, ["active_seed_slot_count", "adventure.active_seed_slot_count"])),
                ("inactive_slots", self._adventure_first(payload, ["inactive_seed_slot_count", "adventure.inactive_seed_slot_count"])),
                ("env_seeds", self._adventure_first(payload, ["compatibility.env_seed_list", "model_compatibility.env_seed_list", "seed_list", "adventure.selected_seeds", "seed_inventory.selected_seeds"])),
                ("model_seeds", self._adventure_first(payload, ["compatibility.model_seed_list", "model_compatibility.model_seed_list"])),
                ("current_level", self._adventure_first(payload, ["adventure.level", "adventure.current_level", "current_level", "adventure_level"])),
                ("level_label", self._adventure_first(payload, ["adventure.adventure_level_label", "adventure.level_label", "adventure_level_label"])),
                ("max_levels", self._adventure_first(payload, ["adventure.max_adventure_levels", "max_adventure_levels", "levels_requested"])),
                ("current_attempt", self._adventure_first(payload, ["adventure.attempt", "adventure.current_attempt", "current_attempt", "attempt", "level_attempt"])),
                ("max_attempts", self._adventure_first(payload, ["adventure.max_attempts_per_level", "max_attempts_per_level"])),
                ("wins_current", self._adventure_first(payload, ["adventure.wins_this_level", "adventure.wins", "wins_this_level", "wins"])),
                ("total_wins", self._adventure_total_wins(payload)),
                ("losses", self._adventure_first(payload, ["adventure.losses", "losses", "eval.losses", "summary.losses"])),
                ("timeouts", self._adventure_first(payload, ["adventure.timeouts", "timeouts", "eval.timeouts", "summary.timeouts"])),
                ("frontier_level", self._adventure_first(payload, ["frontier_level", "adventure.frontier_level"])),
                ("cleared_levels", self._adventure_first(payload, ["cleared_levels", "adventure.cleared_levels"])),
                ("sample_source", self._adventure_first(payload, ["episode_sample_source", "adventure.episode_sample_source"])),
                ("requested_source", self._adventure_first(payload, ["requested_episode_sample_source", "adventure.requested_episode_sample_source"])),
                ("replay_supported", self._adventure_first(payload, ["level_replay_supported", "adventure.level_replay_supported"])),
                ("replay_blocked", self._adventure_first(payload, ["level_replay_blocked_reason", "adventure.level_replay_blocked_reason"])),
                ("terminal_reason", self._adventure_first(payload, ["terminal_reason", "episode_summary.terminal_reason", "eval.terminal_reason", "adventure.terminal_reason"])),
                ("soft_max_steps", self._adventure_first(payload, ["adventure.soft_max_steps", "soft_max_steps"])),
                ("hard_max_steps", self._adventure_first(payload, ["adventure.hard_max_steps", "hard_max_steps"])),
                ("final_wave_ext", self._adventure_first(payload, ["adventure.final_wave_extension_enabled", "final_wave_extension_enabled"])),
                ("soft_timeout", self._adventure_first(payload, ["adventure.soft_timeout_reached", "soft_timeout_reached"])),
                ("soft_extended", self._adventure_first(payload, ["adventure.soft_timeout_extended", "soft_timeout_extended"])),
                ("soft_step", self._adventure_first(payload, ["adventure.soft_timeout_step", "soft_timeout_step"])),
                ("steps_after_soft", self._adventure_first(payload, ["adventure.steps_after_soft_timeout", "steps_after_soft_timeout"])),
                ("timeout_class", self._adventure_first(payload, ["adventure.timeout_classification", "timeout_classification"])),
                ("last_reset_reason", self._adventure_first(payload, ["last_reset_reason", "reset.reason", "reset.reset_reason", "reset.methodUsed", "last_reset.reason"])),
                ("seed_list", self._adventure_first(payload, ["selected_loadout", "current_selected_seed_loadout", "seed_list", "adventure.selected_loadout", "adventure.selected_seeds", "seed_inventory.selected_seeds", "compatibility.env_seed_list"])),
                ("eligible", self._adventure_first(payload, ["eligible_seeds", "adventure.eligible_seeds", "unlock.eligible_seeds"])),
                ("unlocked", self._adventure_first(payload, ["unlock.unlocked_seeds", "adventure.unlocked_seeds", "seed_inventory.unlocked_seeds"])),
                ("new_unlocks", self._adventure_first(payload, ["newly_unlocked", "adventure.newly_unlocked", "unlock.newly_unlocked"])),
                ("plant_types", self._adventure_first(payload, ["plant_types", "compatibility.env_plant_types"])),
                ("recent_win_rate", self._adventure_first(payload, ["recent_win_rate", "summary.win_rate", "eval.win_rate", "win_rate"])),
                ("sunflowers", self._adventure_first(payload, ["plants_by_type.sunflower", "summary.sunflowers_planted", "eval.sunflowers_planted"])),
                ("peashooters", self._adventure_first(payload, ["plants_by_type.peashooter", "summary.peashooters_planted", "eval.peashooters_planted"])),
                ("wallnuts", self._adventure_first(payload, ["plants_by_type.wallnut", "summary.wallnuts_planted", "eval.wallnuts_planted"])),
                ("cherrybombs", self._adventure_first(payload, ["plants_by_type.cherrybomb", "summary.cherrybombs_planted", "eval.cherrybombs_planted"])),
                ("cherry_kills", self._adventure_first(payload, ["summary.cherrybomb_kills_total", "eval.cherrybomb_kills_total"])),
                ("cherry_zero", self._adventure_first(payload, ["summary.cherrybomb_zero_kill_uses", "eval.cherrybomb_zero_kill_uses"])),
                ("wallnut_useful", self._adventure_first(payload, ["summary.wallnut_threatened_lane_placements", "eval.wallnut_threatened_lane_placements"])),
                ("wallnut_useless", self._adventure_first(payload, ["summary.wallnut_useless_placements", "eval.wallnut_useless_placements"])),
                ("tactical_mask", self._adventure_first(payload, ["tactical_mask_enabled", "summary.tactical_mask_enabled", "eval.tactical_mask_enabled"])),
                ("wallnut_mask", self._adventure_first(payload, ["wallnut_tactical_mask_enabled", "summary.wallnut_tactical_mask_enabled", "eval.wallnut_tactical_mask_enabled"])),
                ("cherry_mask", self._adventure_first(payload, ["cherrybomb_tactical_mask_enabled", "summary.cherrybomb_tactical_mask_enabled", "eval.cherrybomb_tactical_mask_enabled"])),
                ("wallnut_masked", self._adventure_first(payload, ["summary.wallnut_actions_masked", "eval.wallnut_actions_masked"])),
                ("cherry_masked", self._adventure_first(payload, ["summary.cherrybomb_actions_masked", "eval.cherrybomb_actions_masked"])),
                ("post_win_active", self._adventure_first(payload, ["post_win_active", "adventure.post_win_active"])),
                ("post_win_elapsed", self._adventure_first(payload, ["post_win_elapsed", "adventure.post_win_elapsed"])),
                ("post_win_clicks", self._adventure_first(payload, ["post_win_click_attempts", "adventure.post_win_click_attempts"])),
                ("post_win_target", self._adventure_first(payload, ["post_win_last_click_target", "adventure.post_win_last_click_target"])),
                ("post_win_click_ok", self._adventure_first(payload, ["post_win_last_click_ok", "adventure.post_win_last_click_ok"])),
                ("post_win_state", self._adventure_first(payload, ["post_win_last_state.screenState", "adventure.post_win_last_state.screenState"])),
                ("fusion_policy", self._adventure_first(payload, ["fusion.fusion_policy", "fusion_policy"])),
                ("fusion_candidates", self._adventure_first(payload, ["fusion.fusion_candidate_count", "fusion_candidate_count", "fusion.fusion_candidate_count_total", "fusion_candidate_count_total", "fusion.fusion_candidates"])),
                ("fusion_attempts", self._adventure_first(payload, ["fusion.fusion_attempted_count", "fusion_attempted_count"])),
                ("fusion_successes", self._adventure_first(payload, ["fusion.fusion_success_count", "fusion_success_count"])),
                ("fusion_rejects", self._adventure_first(payload, ["fusion.fusion_rejected_count", "fusion_rejected_count"])),
                ("fusion_errors", self._adventure_first(payload, ["fusion.fusion_bridge_error_count", "fusion_bridge_error_count", "fusion.fusion_unsafe_state_block_count", "fusion_unsafe_state_block_count"])),
                ("wave", self._adventure_or_na(self._wave_text(payload))),
                ("zombies", self._adventure_first(payload, ["gameplay.zombies", "current_zombies", "zombies", "zombie_count", "zombieCount"])),
                ("plants", self._adventure_first(payload, ["gameplay.plants", "current_plants", "plants", "plant_count", "plantCount"])),
                ("sun", self._adventure_first(payload, ["gameplay.sun", "current_sun", "sun", "game.sun"])),
                ("reward", self._adventure_first(payload, ["reward.episode_reward", "reward.episode", "current_reward", "episode_reward", "reward_total", "last_reward"])),
                ("timestep", self._adventure_first(payload, ["train.total_timesteps", "current_timestep", "total_timesteps"])),
                ("episode_step", self._adventure_first(payload, ["episode_step", "episode.step", "agent.episode_step", "train.current_step", "current_step", "episode_length", "summary.episode_length", "step"])),
                ("fusion_reasons", fusion.get("fusion_rejected_reasons", payload.get("fusion_rejected_reasons", "n/a"))),
            ]
        )

    def _generalist_status_content(self, payload: Dict[str, Any], health: str, using_last_good: bool) -> str:
        if not isinstance(payload, dict):
            payload = {}
        health_text = f"{health} (showing last good)" if using_last_good else health
        streak_current = self._first_value(payload, ["frontier_win_streak", "adventure.frontier_win_streak"], default="0")
        streak_required = self._first_value(
            payload,
            ["frontier_win_streak_required", "adventure.frontier_win_streak_required"],
            default="1",
        )
        wins_current = self._first_value(
            payload,
            ["wins_on_current_level", "adventure.wins_on_current_level", "frontier_win_streak", "adventure.frontier_win_streak"],
            default=streak_current,
        )
        wins_required = self._first_value(
            payload,
            ["wins_before_advance", "adventure.wins_before_advance", "frontier_win_streak_required", "adventure.frontier_win_streak_required"],
            default=streak_required,
        )
        replay_supported = self._first_value(
            payload,
            ["frontier_replay_supported", "adventure.frontier_replay_supported"],
            default="n/a",
        )
        replay_blocked_reason = self._first_value(
            payload,
            ["frontier_replay_blocked_reason", "adventure.frontier_replay_blocked_reason"],
            default="",
        )
        replay_status = str(replay_supported)
        if replay_blocked_reason not in ("", None, "n/a"):
            replay_status = f"{replay_status} ({replay_blocked_reason})"
        return lines_from_pairs(
            [
                ("health", health_text),
                ("mode", self._first_value(payload, ["run_mode", "mode"], default="n/a")),
                ("status", self._first_value(payload, ["status", "adventure.status"], default="n/a")),
                ("phase", self._first_value(payload, ["adventure_phase", "adventure.state", "state"], default="n/a")),
                ("frontier_level", self._first_value(payload, ["frontier_level", "adventure.frontier_level"], default="n/a")),
                ("current_level", self._first_value(payload, ["current_level", "adventure.current_level", "adventure.level"], default="n/a")),
                ("current_attempt", self._first_value(payload, ["current_attempt", "adventure.current_attempt", "adventure.attempt"], default="n/a")),
                ("frontier_streak", f"{streak_current} / {streak_required}"),
                ("wins_before_advance", f"{wins_current} / {wins_required}"),
                (
                    "promotion_ready",
                    self._first_value(payload, ["frontier_mastery_ready", "adventure.frontier_mastery_ready"], default="n/a"),
                ),
                (
                    "decision",
                    self._first_value(payload, ["post_win_decision", "adventure.post_win_decision"], default="n/a"),
                ),
                (
                    "transition_allowed",
                    self._first_value(payload, ["post_win_transition_allowed", "adventure.post_win_transition_allowed"], default="n/a"),
                ),
                (
                    "latest_result",
                    self._first_value(payload, ["latest_terminal_result", "last_result", "adventure.last_result"], default="n/a"),
                ),
                ("reset_phase", self._first_value(payload, ["reset_phase", "adventure.reset_phase"], default="n/a")),
                (
                    "expected_transition",
                    self._first_value(payload, ["expected_transition_target", "adventure.expected_transition_target"], default="n/a"),
                ),
                (
                    "seed_selection_expected",
                    self._first_value(payload, ["seed_selection_expected", "adventure.seed_selection_expected"], default="n/a"),
                ),
                ("replay_support", replay_status),
                ("latest_unlock", self._first_value(payload, ["latest_unlock", "unlock.latest_unlock", "adventure.latest_unlock"], default="n/a")),
                ("unlocked_seeds", self._first_value(payload, ["unlocked_seeds", "unlock.unlocked_seeds", "adventure.unlocked_seeds"], default="n/a")),
                ("eligible_seeds", self._first_value(payload, ["eligible_seeds", "unlock.eligible_seeds", "adventure.eligible_seeds"], default="n/a")),
                ("selectable_seeds", self._first_value(payload, ["selectable_seeds", "unlock.selectable_seeds", "adventure.selectable_seeds"], default="n/a")),
                ("observed_capacity", self._first_value(payload, ["observed_seed_bank_capacity", "adventure.observed_seed_bank_capacity"], default="n/a")),
                ("selected_loadout", self._first_value(payload, ["selected_loadout", "adventure.selected_loadout"], default="n/a")),
                ("selected_count", self._first_value(payload, ["selected_loadout_count", "adventure.selected_loadout_count"], default="n/a")),
                ("inactive_slots", self._first_value(payload, ["inactive_model_slots", "adventure.inactive_model_slots"], default="n/a")),
                ("loadout_reason", self._first_value(payload, ["loadout_reason", "adventure.loadout_reason"], default="n/a")),
                ("last_reset_reason", self._first_value(payload, ["last_reset_reason", "adventure.last_reset_reason"], default="n/a")),
                ("post_win_block", self._first_value(payload, ["post_win_blocked_reason", "adventure.post_win_blocked_reason"], default="n/a")),
            ]
        )

    def _adventure_first(self, payload: Dict[str, Any], paths: List[str]) -> Any:
        return self._first_value(payload, paths, default="n/a")

    def _adventure_or_na(self, value: Any) -> Any:
        return "n/a" if value is None or value == "" else value

    def _adventure_compat_blocked_reason(self, payload: Dict[str, Any], compatibility: Dict[str, Any]) -> Any:
        value = compatibility.get("blocked_reason") or compatibility.get("model_compatibility_blocked_reason")
        if value:
            return value
        return self._adventure_first(
            payload,
            [
                "model_compatibility.blocked_reason",
                "compatibility.blocked_reason",
                "compatibility.model_compatibility_blocked_reason",
                "blocked_reason",
                "adventure.blocked_reason",
            ],
        )

    def _adventure_total_wins(self, payload: Dict[str, Any]) -> Any:
        value = self._first_value(
            payload,
            ["adventure.total_wins", "adventure.wins_total", "total_wins", "wins_total", "summary.wins", "eval.wins_total"],
            default=MISSING,
        )
        if value is not MISSING:
            return value
        episodes = self._first_value(payload, ["summary.episodes_completed", "eval.episodes_completed", "eval.episodes"], default=MISSING)
        win_rate = self._first_value(payload, ["summary.win_rate", "eval.win_rate", "win_rate"], default=MISSING)
        if episodes is not MISSING and win_rate is not MISSING:
            try:
                return int(round(float(episodes) * float(win_rate)))
            except (TypeError, ValueError):
                pass
        return "n/a"

    def _top_level_keys(self, payload: Dict[str, Any]) -> str:
        keys = sorted(str(key) for key in payload.keys())
        return ", ".join(keys) if keys else "-"

    def _active_run_text(self, payload: Dict[str, Any]) -> str:
        run_path = self._first_value(
            payload,
            [
                "active_run",
                "active_run_dir",
                "active_run_path",
                "run_dir",
                "run_path",
                "run.path",
                "train.run_dir",
                "eval.run_dir",
            ],
        )
        if run_path not in (None, ""):
            return str(self._resolve_text_path(str(run_path)))
        return self.active_run_path or "unknown"

    def _as_dict(self, value: Any) -> Dict[str, Any]:
        return value if isinstance(value, dict) else {}

    def _lookup_path(self, payload: Any, path: str) -> Any:
        current = payload
        for part in path.split("."):
            if not isinstance(current, dict):
                return MISSING
            if part in current:
                current = current[part]
                continue
            lower_lookup = {str(key).lower(): key for key in current.keys()}
            key = lower_lookup.get(part.lower())
            if key is None:
                return MISSING
            current = current[key]
        return current

    def _first_value(self, payload: Any, paths: List[str], default: Any = None) -> Any:
        for path in paths:
            value = self._lookup_path(payload, path)
            if value is not MISSING and value is not None and value != "":
                return value
        return default

    def _wave_text(self, payload: Dict[str, Any]) -> Any:
        wave = self._first_value(payload, ["gameplay.wave", "current_wave", "wave", "wave_text"])
        max_wave = self._first_value(payload, ["gameplay.max_wave", "gameplay.maxWave", "max_wave", "maxWave"])
        if wave is None:
            return None
        if max_wave is not None and "/" not in str(wave):
            return f"{wave}/{max_wave}"
        return wave

    def _safe_len(self, value: Any) -> int:
        if isinstance(value, (list, tuple, dict, set)):
            return len(value)
        return 0

    def _row_panel_lines(self, payload: Dict[str, Any]) -> str:
        rows = self._normalize_row_source(self._first_value(payload, ["rows", "row_diagnostics", "lane_diagnostics"], default=MISSING))
        if not rows:
            rows = self._rows_from_fallback_fields(payload)
        if not rows:
            return "No row diagnostics in live_status.json"
        output: List[str] = []
        for row, values in sorted(rows.items(), key=lambda item: self._row_sort_key(item[0])):
            output.append(
                f"row {row}: pea={fmt_value(values.get('peashooters'))} "
                f"threat={fmt_value(values.get('threatened'))} "
                f"undef={fmt_value(values.get('undefended_threat'))} "
                f"steps={fmt_value(values.get('threat_steps'))}"
            )
        return "\n".join(output) if output else "No row diagnostics in live_status.json"

    def _normalize_row_source(self, source: Any) -> Dict[str, Dict[str, Any]]:
        rows: Dict[str, Dict[str, Any]] = {}
        if source is MISSING or source is None:
            return rows
        if isinstance(source, list):
            for index, item in enumerate(source):
                if isinstance(item, dict):
                    row_id = self._first_value(item, ["row", "row_index", "lane", "lane_index"], default=index)
                    rows[str(row_id)] = self._row_record(item)
            return rows
        if isinstance(source, dict):
            if any(isinstance(value, dict) for value in source.values()):
                for row_id, item in source.items():
                    if isinstance(item, dict):
                        actual_row = self._first_value(item, ["row", "row_index", "lane", "lane_index"], default=row_id)
                        rows[str(actual_row)] = self._row_record(item)
                return rows
            return self._rows_from_fallback_fields(source)
        return rows

    def _row_record(self, data: Dict[str, Any]) -> Dict[str, Any]:
        zombies = self._first_value(data, ["zombies", "zombie_count"])
        threatened = self._first_value(data, ["threatened", "threat", "has_threat", "under_threat"])
        if threatened is None and isinstance(zombies, (int, float)):
            threatened = zombies > 0
        return {
            "peashooters": self._first_value(data, ["peashooters", "pea", "peashooter_count", "current_peashooters", "plants", "plant_count"], default=0),
            "threatened": threatened if threatened is not None else False,
            "undefended_threat": self._first_value(data, ["undefended_threat", "undefended", "undefended_threatened", "undefended_threat_steps"], default=False),
            "threat_steps": self._first_value(data, ["threat_steps", "steps", "threat_step_count"], default=0),
        }

    def _rows_from_fallback_fields(self, payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        peashooters = self._first_value(
            payload,
            [
                "peashooters_by_row",
                "current_peashooters_by_row",
                "lane_diagnostics.peashooters_by_row",
                "lane_diagnostics.current_peashooters_by_row",
            ],
            default={},
        )
        threat_steps = self._first_value(payload, ["threat_steps_by_row", "lane_diagnostics.threat_steps_by_row"], default={})
        undefended = self._first_value(
            payload,
            [
                "undefended_by_row",
                "undefended_threat_steps_by_row",
                "lane_diagnostics.undefended_by_row",
                "lane_diagnostics.undefended_threat_steps_by_row",
            ],
            default={},
        )
        sources = [item for item in (peashooters, threat_steps, undefended) if isinstance(item, dict)]
        row_ids = sorted({str(key) for source in sources for key in source.keys()}, key=self._row_sort_key)
        rows: Dict[str, Dict[str, Any]] = {}
        for row_id in row_ids:
            rows[row_id] = {
                "peashooters": self._row_dict_value(peashooters, row_id, default=0),
                "threatened": self._row_dict_value(threat_steps, row_id, default=0) not in (0, False, None, ""),
                "undefended_threat": self._row_dict_value(undefended, row_id, default=0) not in (0, False, None, ""),
                "threat_steps": self._row_dict_value(threat_steps, row_id, default=0),
            }
        return rows

    def _row_dict_value(self, mapping: Any, row_id: str, default: Any = None) -> Any:
        if not isinstance(mapping, dict):
            return default
        if row_id in mapping:
            return mapping[row_id]
        try:
            numeric = int(row_id)
        except ValueError:
            return default
        return mapping.get(numeric, default)

    def _row_sort_key(self, row_id: Any) -> Tuple[int, str]:
        text = str(row_id)
        try:
            return int(text), text
        except ValueError:
            return 999, text

    def _on_close(self) -> None:
        if self.active_process is not None and self.active_process.poll() is None:
            try:
                self.active_process.terminate()
            except OSError:
                pass
        self.root.destroy()


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
