"""Application shell and first-class workflow pages for the PvZRL Tk GUI.

This module owns presentation only.  The dashboard still launches the canonical
``train_ppo.py`` entrypoint and consumes canonical status/artifacts; it does not
own Adventure, Streamer, Twitch, PPO, BC, action, or bridge behavior.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import tkinter as tk
from tkinter import ttk

from pvzrl_action_space import (
    ADVENTURE_IDENTITY_ACTION_COUNT,
    ADVENTURE_IDENTITY_ACTION_DECODER_VERSION,
    ADVENTURE_IDENTITY_MAX_SEED_SLOTS,
    ADVENTURE_IDENTITY_OBSERVATION_VERSION,
    DEFAULT_COLS,
    DEFAULT_ROWS,
)
from pvzrl_gui_config import TwitchEnvironmentNames, inspect_twitch_credentials


PAGE_DASHBOARD = "Dashboard"
PAGE_TRAINING = "Training"
PAGE_EVALUATION = "Evaluation"
PAGE_STREAMER = "Streamer"
PAGE_RUNS = "Runs & Models"
PAGE_DIAGNOSTICS = "Diagnostics"
PAGE_COACH = "Local Coach"
PAGE_SETTINGS = "Settings"

PAGE_ORDER = (
    PAGE_DASHBOARD,
    PAGE_TRAINING,
    PAGE_EVALUATION,
    PAGE_STREAMER,
    PAGE_RUNS,
    PAGE_DIAGNOSTICS,
    PAGE_COACH,
    PAGE_SETTINGS,
)

class GuiApplicationShellMixin:
    """Build and update the persistent navigation shell."""

    def _configure_gui_styles(self) -> None:
        style = ttk.Style(self.root)
        available = set(style.theme_names())
        if "vista" in available:
            try:
                style.theme_use("vista")
            except tk.TclError:
                pass
        style.configure("AppTitle.TLabel", font=("Segoe UI Semibold", 16))
        style.configure("PageTitle.TLabel", font=("Segoe UI Semibold", 14))
        style.configure("SectionTitle.TLabel", font=("Segoe UI Semibold", 10))
        style.configure("Metric.TLabel", font=("Segoe UI Semibold", 12))
        style.configure("Muted.TLabel", foreground="#59636e")
        style.configure("Contract.TLabel", foreground="#35586c")
        style.configure("Nav.TButton", anchor="w", padding=(12, 8))
        style.configure("Primary.TButton", padding=(12, 7))
        style.configure("Danger.TButton", padding=(10, 6))
        style.configure("State.TLabel", font=("Segoe UI Semibold", 9), padding=(8, 3))
        style.configure("Card.TFrame", relief="solid", borderwidth=1)

    def _build_application_shell(self) -> None:
        self._configure_gui_styles()
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=5)
        self.root.rowconfigure(3, weight=1)

        self._build_global_header()

        body = ttk.Frame(self.root)
        body.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 6))
        body.rowconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)

        navigation = ttk.Frame(body, width=154)
        navigation.grid(row=0, column=0, sticky="nsw", padx=(0, 10))
        navigation.grid_propagate(False)
        navigation.columnconfigure(0, weight=1)

        host = ttk.Frame(body)
        host.grid(row=0, column=1, sticky="nsew")
        host.rowconfigure(0, weight=1)
        host.columnconfigure(0, weight=1)

        self.page_frames = {}
        self.navigation_buttons = {}
        for row, page_name in enumerate(PAGE_ORDER):
            button = ttk.Button(
                navigation,
                text=page_name,
                style="Nav.TButton",
                command=lambda value=page_name: self._show_page(value),
            )
            button.grid(row=row, column=0, sticky="ew", pady=(0, 3))
            self.navigation_buttons[page_name] = button
            frame = ttk.Frame(host)
            frame.grid(row=0, column=0, sticky="nsew")
            self.page_frames[page_name] = frame

        self._build_dashboard_page(self.page_frames[PAGE_DASHBOARD])
        self._build_train_tab(self.page_frames[PAGE_TRAINING])
        self._build_eval_tab(self.page_frames[PAGE_EVALUATION])
        self._build_streamer_page(self.page_frames[PAGE_STREAMER])
        self._build_runs_models_page(self.page_frames[PAGE_RUNS])
        self._build_diagnostics_tab(self.page_frames[PAGE_DIAGNOSTICS])
        self._build_local_coach_page(self.page_frames[PAGE_COACH])
        self._build_settings_page(self.page_frames[PAGE_SETTINGS])

        self._build_status_bar()
        self._build_log_panel()
        self._show_page(PAGE_DASHBOARD)

    def _build_global_header(self) -> None:
        header = ttk.Frame(self.root, padding=(12, 9, 12, 7))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)
        ttk.Label(header, text="PvZRL Control Center", style="AppTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            header,
            text=(
                f"Full Adventure · {DEFAULT_ROWS}×{DEFAULT_COLS} · "
                f"{ADVENTURE_IDENTITY_MAX_SEED_SLOTS} slots · "
                f"{ADVENTURE_IDENTITY_ACTION_COUNT} actions · 4,364 observations"
            ),
            style="Contract.TLabel",
        ).grid(row=0, column=1, sticky="w", padx=(16, 8))
        ttk.Label(header, textvariable=self.process_lifecycle_var, style="State.TLabel").grid(
            row=0, column=2, sticky="e", padx=(8, 5)
        )
        ttk.Label(header, textvariable=self.header_health_var, style="State.TLabel").grid(
            row=0, column=3, sticky="e"
        )

    def _show_page(self, page_name: str) -> None:
        selected = page_name if page_name in self.page_frames else PAGE_DASHBOARD
        self.page_frames[selected].tkraise()
        self.current_page = selected
        self.current_page_var.set(selected)
        for name, button in self.navigation_buttons.items():
            button.state(["disabled"] if name == selected else ["!disabled"])
        if selected == PAGE_RUNS:
            self.refresh_artifact_index()
        elif selected == PAGE_STREAMER:
            self.refresh_streamer_credentials()

    def _build_dashboard_page(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(3, weight=1)
        ttk.Label(parent, text="Dashboard", style="PageTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(2, 8)
        )

        actions = ttk.Frame(parent)
        actions.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        for column in range(3):
            actions.columnconfigure(column, weight=1)
        for column, (label, page, detail) in enumerate(
            (
                ("Train", PAGE_TRAINING, "Create or resume a Full-Adventure Generalist run"),
                ("Evaluate", PAGE_EVALUATION, "Run autonomous Adventure evaluation"),
                ("Stream", PAGE_STREAMER, "Launch and monitor Streamer V1 / Twitch"),
            )
        ):
            card = ttk.Frame(actions, style="Card.TFrame", padding=10)
            card.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 5, 0))
            card.columnconfigure(0, weight=1)
            ttk.Button(
                card,
                text=label,
                style="Primary.TButton",
                command=lambda value=page: self._show_page(value),
            ).grid(row=0, column=0, sticky="ew")
            ttk.Label(card, text=detail, style="Muted.TLabel", wraplength=260).grid(
                row=1, column=0, sticky="w", pady=(7, 0)
            )

        overview = ttk.LabelFrame(parent, text="Current system")
        overview.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        for column in range(4):
            overview.columnconfigure(column, weight=1)
        metrics = (
            ("State", self.dashboard_state_var),
            ("Mode", self.dashboard_mode_var),
            ("Adventure level", self.dashboard_level_var),
            ("PPO timesteps", self.dashboard_timesteps_var),
            ("Model", self.dashboard_model_var),
            ("Run directory", self.dashboard_run_var),
            ("Game / bridge", self.dashboard_bridge_var),
            ("Twitch", self.dashboard_twitch_var),
        )
        for index, (label, variable) in enumerate(metrics):
            card = ttk.Frame(overview, padding=(9, 7))
            card.grid(row=index // 4, column=index % 4, sticky="nsew", padx=3, pady=3)
            ttk.Label(card, text=label, style="Muted.TLabel").grid(row=0, column=0, sticky="w")
            ttk.Label(card, textvariable=variable, style="Metric.TLabel", wraplength=220).grid(
                row=1, column=0, sticky="w", pady=(2, 0)
            )

        alerts = ttk.LabelFrame(parent, text="Operator attention")
        alerts.grid(row=3, column=0, sticky="nsew")
        alerts.columnconfigure(0, weight=1)
        alerts.rowconfigure(1, weight=1)
        ttk.Label(alerts, textvariable=self.dashboard_health_var, style="SectionTitle.TLabel").grid(
            row=0, column=0, sticky="w", padx=8, pady=(7, 2)
        )
        ttk.Label(
            alerts,
            textvariable=self.dashboard_warning_var,
            justify="left",
            anchor="nw",
            wraplength=900,
        ).grid(row=1, column=0, sticky="nsew", padx=8, pady=(2, 8))

    def _build_streamer_page(self, parent: ttk.Frame) -> None:
        content = self._make_scrollable_container(parent)
        content.columnconfigure(0, weight=1)
        ttk.Label(content, text="Streamer V1 / Twitch", style="PageTitle.TLabel").grid(
            row=0, column=0, sticky="w", padx=8, pady=(3, 2)
        )
        ttk.Label(
            content,
            text=(
                "First-class launcher for the existing Streamer V1 orchestrator. "
                "Viewer actions, PPO, BC, evaluation, checkpoints, and Twitch remain backend-owned."
            ),
            style="Muted.TLabel",
            wraplength=980,
        ).grid(row=1, column=0, sticky="w", padx=8, pady=(0, 7))

        state = ttk.LabelFrame(content, text="Stream controls")
        state.grid(row=2, column=0, sticky="ew", padx=8, pady=3)
        state.columnconfigure(5, weight=1)
        ttk.Label(state, text="Lifecycle", style="SectionTitle.TLabel").grid(
            row=0, column=0, sticky="w", padx=7, pady=6
        )
        ttk.Label(state, textvariable=self.stream_lifecycle_var, style="State.TLabel").grid(
            row=0, column=1, sticky="w", padx=(0, 12), pady=6
        )
        self._add_launch_button(state, "Start Streamer", self.start_streamer_v1).grid(
            row=0, column=2, sticky="w", padx=(0, 5), pady=5
        )
        self._add_stop_button(state, "Stop Streamer").grid(
            row=0, column=3, sticky="w", padx=(0, 8), pady=5
        )
        ttk.Label(state, textvariable=self.stream_validation_var, anchor="w").grid(
            row=0, column=5, sticky="ew", padx=7, pady=6
        )

        setup = ttk.LabelFrame(content, text="Stream configuration")
        setup.grid(row=3, column=0, sticky="ew", padx=8, pady=3)
        for column in (1, 3):
            setup.columnconfigure(column, weight=1)
        self._add_labeled_entry(setup, 0, 0, "Config JSON", self.streamer_config_path_var, width=48)
        ttk.Button(setup, text="Load", command=self.load_streamer_config).grid(
            row=0, column=4, sticky="w", padx=(0, 5), pady=2
        )
        self._add_labeled_entry(setup, 1, 0, "Baseline checkpoint", self.streamer_baseline_checkpoint_var, width=48)
        ttk.Button(setup, text="Browse", command=self.browse_streamer_baseline).grid(
            row=1, column=4, sticky="w", padx=(0, 5), pady=2
        )
        self._add_labeled_entry(setup, 2, 0, "Experiment directory", self.streamer_run_dir_var, width=48)
        ttk.Button(setup, text="Browse", command=self.browse_streamer_run_folder).grid(
            row=2, column=4, sticky="w", padx=(0, 5), pady=2
        )
        ttk.Label(setup, text="Platform").grid(row=3, column=0, sticky="w", padx=6, pady=2)
        ttk.Combobox(
            setup,
            textvariable=self.streamer_platform_var,
            values=("twitch", "mock"),
            state="readonly",
            width=14,
        ).grid(row=3, column=1, sticky="w", padx=6, pady=2)
        self._add_labeled_entry(setup, 3, 2, "Adventure start", self.streamer_start_level_var)
        self._add_labeled_entry(setup, 4, 0, "Levels to attempt", self.streamer_max_levels_var)
        self._add_labeled_entry(setup, 4, 2, "Attempts / level", self.streamer_max_attempts_var)
        self._add_labeled_entry(setup, 5, 0, "Mock JSONL", self.streamer_mock_script_var, width=48)
        ttk.Button(setup, text="Browse", command=self.browse_streamer_mock_script).grid(
            row=5, column=4, sticky="w", padx=(0, 5), pady=2
        )
        ttk.Checkbutton(
            setup,
            text="Quick wait",
            variable=self.streamer_quick_wait_var,
        ).grid(row=6, column=0, sticky="w", padx=6, pady=3)
        ttk.Checkbutton(
            setup,
            text="Wait for gameplay-ready",
            variable=self.streamer_wait_gameplay_ready_var,
        ).grid(row=6, column=2, columnspan=2, sticky="w", padx=6, pady=3)

        credentials = ttk.LabelFrame(content, text="Twitch credential readiness · names only")
        credentials.grid(row=4, column=0, sticky="ew", padx=8, pady=3)
        credentials.columnconfigure(1, weight=1)
        credentials.columnconfigure(3, weight=1)
        credential_specs = (
            ("Client ID env", self.streamer_client_id_env_var, self.streamer_client_id_ready_var),
            ("Access token env", self.streamer_access_token_env_var, self.streamer_access_token_ready_var),
            ("Broadcaster ID env", self.streamer_broadcaster_id_env_var, self.streamer_broadcaster_id_ready_var),
            ("EventSub user env", self.streamer_user_id_env_var, self.streamer_user_id_ready_var),
            ("Viewer hash secret env", self.streamer_hash_secret_env_var, self.streamer_hash_secret_ready_var),
        )
        for index, (label, name_var, ready_var) in enumerate(credential_specs):
            row = index // 2
            base = 0 if index % 2 == 0 else 2
            ttk.Label(credentials, text=label).grid(row=row, column=base, sticky="w", padx=6, pady=2)
            field = ttk.Frame(credentials)
            field.grid(row=row, column=base + 1, sticky="ew", padx=6, pady=2)
            field.columnconfigure(0, weight=1)
            ttk.Entry(field, textvariable=name_var).grid(row=0, column=0, sticky="ew")
            ttk.Label(field, textvariable=ready_var, width=12, anchor="e").grid(row=0, column=1, padx=(6, 0))
        ttk.Button(credentials, text="Refresh readiness", command=self.refresh_streamer_credentials).grid(
            row=3, column=0, sticky="w", padx=6, pady=5
        )
        ttk.Label(credentials, textvariable=self.streamer_credentials_summary_var, style="Muted.TLabel").grid(
            row=3, column=1, columnspan=3, sticky="w", padx=6, pady=5
        )

        tuning = ttk.LabelFrame(content, text="Cycle and learning settings")
        tuning.grid(row=5, column=0, sticky="ew", padx=8, pady=3)
        for column in (1, 3, 5):
            tuning.columnconfigure(column, weight=1)
        specs = (
            ("PPO rollout", self.streamer_n_steps_var),
            ("PPO batch", self.streamer_batch_size_var),
            ("Policy steps / cycle", self.streamer_policy_steps_var),
            ("CURRENT save steps", self.streamer_checkpoint_steps_var),
            ("Evaluation episodes", self.streamer_eval_episodes_var),
            ("Max cycles (0=∞)", self.streamer_max_cycles_var),
            ("Endurance hours", self.streamer_endurance_hours_var),
            ("Viewer cadence sec", self.streamer_interval_var),
            ("Command TTL sec", self.streamer_ttl_var),
            ("Queue capacity", self.streamer_queue_capacity_var),
            ("Message max chars", self.streamer_message_max_var),
            ("BC coefficient", self.streamer_bc_coefficient_var),
            ("Demo capacity", self.streamer_demo_capacity_var),
            ("Demo persist (config)", self.streamer_demo_persist_var),
            ("BC batch", self.streamer_bc_batch_var),
            ("BC frequency", self.streamer_bc_frequency_var),
            ("BC minimum demos", self.streamer_bc_min_var),
        )
        for index, (label, variable) in enumerate(specs):
            row = index // 3
            base = (index % 3) * 2
            entry = self._add_labeled_entry(tuning, row, base, label, variable, width=14)
            if variable is self.streamer_demo_persist_var:
                entry.configure(state="readonly")
        ttk.Checkbutton(tuning, text="Masked behavior cloning enabled", variable=self.streamer_bc_enabled_var).grid(
            row=6, column=0, columnspan=2, sticky="w", padx=6, pady=4
        )
        ttk.Label(tuning, text="PPO updates: enabled in STREAM_TRAIN · Evaluation chat control: disabled").grid(
            row=6, column=2, columnspan=4, sticky="w", padx=6, pady=4
        )

        monitor = ttk.LabelFrame(content, text="Live Stream monitor")
        monitor.grid(row=6, column=0, sticky="ew", padx=8, pady=3)
        monitor.columnconfigure(0, weight=1)
        notebook = ttk.Notebook(monitor)
        notebook.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        runtime = ttk.Frame(notebook, padding=6)
        viewer = ttk.Frame(notebook, padding=6)
        learning = ttk.Frame(notebook, padding=6)
        roles = ttk.Frame(notebook, padding=6)
        game = ttk.Frame(notebook, padding=6)
        notebook.add(runtime, text="Runtime")
        notebook.add(viewer, text="Viewer activity")
        notebook.add(learning, text="PPO / BC")
        notebook.add(roles, text="Models / evaluations")
        notebook.add(game, text="Game state")
        self._build_stream_metric_grid(
            runtime,
            (
                ("Health", self.stream_health_var),
                ("Phase", self.stream_phase_var),
                ("Cycle", self.stream_cycle_var),
                ("Uptime", self.stream_uptime_var),
                ("Active run", self.stream_active_run_var),
                ("Next Adventure", self.stream_next_level_var),
                ("Current model steps", self.stream_model_steps_var),
                ("Baseline steps", self.stream_baseline_steps_var),
                ("Evaluation countdown", self.stream_eval_countdown_var),
                ("Twitch connection", self.stream_connection_var),
            ),
        )
        self._build_stream_metric_grid(
            viewer,
            (
                ("Queue depth", self.stream_queue_depth_var),
                ("Total commands", self.stream_total_commands_var),
                ("Accepted", self.stream_accepted_var),
                ("Rejected", self.stream_rejected_var),
                ("Invalid", self.stream_invalid_var),
                ("Expired", self.stream_expired_var),
                ("Verified interventions", self.stream_interventions_var),
                ("Viewer attempts", self.stream_attempts_var),
                ("Distinct hashed viewers", self.stream_viewers_var),
                ("Pending reason", self.stream_pending_reason_var),
                ("Latest source", self.stream_action_source_var),
                ("Latest command", self.stream_latest_command_var),
                ("Latest result", self.stream_last_result_var),
            ),
        )
        history = ttk.LabelFrame(viewer, text="Privacy-safe recent command outcomes")
        history.grid(row=4, column=0, columnspan=5, sticky="ew", pady=(7, 0))
        history.columnconfigure(0, weight=1)
        columns = ("time", "viewer", "command", "target", "legal", "executed", "result")
        self.stream_event_tree = ttk.Treeview(history, columns=columns, show="headings", height=7)
        widths = {"time": 120, "viewer": 90, "command": 140, "target": 150, "legal": 80, "executed": 80, "result": 210}
        for name in columns:
            self.stream_event_tree.heading(name, text=name.title())
            self.stream_event_tree.column(name, width=widths[name], stretch=name in {"command", "target", "result"})
        self.stream_event_tree.grid(row=0, column=0, sticky="ew")

        self._build_stream_metric_grid(
            learning,
            (
                ("PPO updates", self.stream_ppo_enabled_var),
                ("BC updates", self.stream_bc_updates_var),
                ("Evaluation chat control", self.stream_eval_chat_control_var),
                ("Policy timesteps", self.stream_policy_timesteps_var),
                ("Environment actions", self.stream_environment_actions_var),
                ("Demonstrations", self.stream_demo_count_var),
                ("Rejected demos", self.stream_demo_rejected_var),
                ("BC updates", self.stream_bc_update_count_var),
                ("BC loss", self.stream_bc_loss_var),
            ),
        )
        self._build_stream_metric_grid(
            roles,
            (
                ("BASELINE", self.stream_baseline_checkpoint_status_var),
                ("CURRENT", self.stream_current_checkpoint_var),
                ("BEST", self.stream_best_checkpoint_var),
                ("Baseline evaluation", self.stream_baseline_eval_var),
                ("Current evaluation", self.stream_current_eval_var),
                ("Best evaluation", self.stream_best_eval_var),
                ("Comparison", self.stream_evaluation_comparison_var),
            ),
        )
        self._build_stream_metric_grid(
            game,
            (
                ("Adventure level", self.stream_game_level_var),
                ("Wave", self.stream_wave_var),
                ("Sun", self.stream_sun_var),
                ("Plants", self.stream_plants_var),
                ("Zombies", self.stream_zombies_var),
                ("Live board", self.stream_board_geometry_var),
                ("Selected seed bank", self.stream_seed_bank_var),
                ("Unlocked plants", self.stream_unlocked_var),
            ),
        )

        preview = ttk.LabelFrame(content, text="Backend command preview")
        preview.grid(row=7, column=0, sticky="ew", padx=8, pady=(3, 8))
        preview.columnconfigure(0, weight=1)
        self.streamer_preview = self._preview_box(preview, height=6)

    def _build_stream_metric_grid(
        self,
        parent: ttk.Frame,
        metrics: Iterable[Tuple[str, tk.StringVar]],
    ) -> None:
        for column in range(5):
            parent.columnconfigure(column, weight=1)
        for index, (label, variable) in enumerate(metrics):
            card = ttk.Frame(parent, style="Card.TFrame", padding=7)
            card.grid(row=index // 5, column=index % 5, sticky="nsew", padx=2, pady=2)
            ttk.Label(card, text=label, style="Muted.TLabel").grid(row=0, column=0, sticky="w")
            ttk.Label(card, textvariable=variable, wraplength=190).grid(row=1, column=0, sticky="w", pady=(2, 0))

    def _build_runs_models_page(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)
        ttk.Label(parent, text="Runs & Models", style="PageTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(2, 8)
        )
        toolbar = ttk.Frame(parent)
        toolbar.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        toolbar.columnconfigure(5, weight=1)
        ttk.Button(toolbar, text="Refresh index", command=self.refresh_artifact_index).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(toolbar, text="Use for evaluation", command=self.use_selected_artifact_for_evaluation).grid(row=0, column=1, padx=(0, 5))
        ttk.Button(toolbar, text="Use for resume", command=self.use_selected_artifact_for_resume).grid(row=0, column=2, padx=(0, 5))
        ttk.Button(toolbar, text="Use as Streamer baseline", command=self.use_selected_artifact_for_streamer).grid(row=0, column=3, padx=(0, 5))
        ttk.Button(toolbar, text="Open run folder", command=self.open_selected_artifact_folder).grid(row=0, column=4, padx=(0, 5))
        ttk.Label(toolbar, textvariable=self.artifact_index_status_var, anchor="e").grid(row=0, column=5, sticky="e")

        split = ttk.Panedwindow(parent, orient="vertical")
        split.grid(row=2, column=0, sticky="nsew")
        browser = ttk.Frame(split)
        details = ttk.LabelFrame(split, text="Selected artifact")
        split.add(browser, weight=3)
        split.add(details, weight=1)
        browser.columnconfigure(0, weight=1)
        browser.rowconfigure(0, weight=1)
        columns = ("role", "compatible", "timesteps", "level", "modified", "path")
        self.artifact_tree = ttk.Treeview(browser, columns=columns, show="headings", height=14)
        widths = {"role": 105, "compatible": 90, "timesteps": 100, "level": 65, "modified": 145, "path": 520}
        for name in columns:
            self.artifact_tree.heading(name, text=name.title())
            self.artifact_tree.column(name, width=widths[name], stretch=name == "path")
        self.artifact_tree.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(browser, orient="vertical", command=self.artifact_tree.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.artifact_tree.configure(yscrollcommand=scroll.set)
        self.artifact_tree.bind("<<TreeviewSelect>>", lambda _event: self.show_selected_artifact_details())
        details.columnconfigure(0, weight=1)
        ttk.Label(
            details,
            textvariable=self.artifact_details_var,
            anchor="nw",
            justify="left",
            wraplength=1020,
        ).grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

    def _build_local_coach_page(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        ttk.Label(parent, text="Local Coach & Advanced Tools", style="PageTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(2, 8)
        )
        notebook = ttk.Notebook(parent)
        notebook.grid(row=1, column=0, sticky="nsew")
        coach = ttk.Frame(notebook)
        board = ttk.Frame(notebook)
        notebook.add(coach, text="Local assisted coach")
        notebook.add(board, text=f"{DEFAULT_ROWS}×{DEFAULT_COLS} board / fusion inspector")
        self._build_coach_tab(coach)
        self._build_fusion_tab(board)

    def _build_settings_page(self, parent: ttk.Frame) -> None:
        content = self._make_scrollable_container(parent)
        content.columnconfigure(0, weight=1)
        ttk.Label(content, text="Settings", style="PageTitle.TLabel").grid(
            row=0, column=0, sticky="w", padx=8, pady=(3, 8)
        )
        paths = ttk.LabelFrame(content, text="Canonical paths")
        paths.grid(row=1, column=0, sticky="ew", padx=8, pady=3)
        paths.columnconfigure(1, weight=1)
        self._add_labeled_entry(paths, 0, 0, "Training config", self.generalist_config_path_var, width=72)
        self._add_labeled_entry(paths, 1, 0, "Evaluation config", self.eval_config_path_var, width=72)
        self._add_labeled_entry(paths, 2, 0, "Streamer config", self.streamer_config_path_var, width=72)
        self._add_labeled_entry(paths, 3, 0, "Runs root", self.runs_root_var, width=72)
        self._add_labeled_entry(paths, 4, 0, "Live status", self.live_status_path_var, width=72)
        ttk.Button(paths, text="Apply live-status path", command=self.apply_live_status_path).grid(
            row=4, column=2, sticky="w", padx=(0, 6), pady=2
        )

        contract = ttk.LabelFrame(content, text="Immutable Full-Adventure model contract")
        contract.grid(row=2, column=0, sticky="ew", padx=8, pady=3)
        contract.columnconfigure(1, weight=1)
        values = (
            ("Policy board", f"{DEFAULT_ROWS} rows × {DEFAULT_COLS} columns (five-lane boards pad/mask row 6)"),
            ("Identity slots", str(ADVENTURE_IDENTITY_MAX_SEED_SLOTS)),
            ("Action surface", f"0 wait; 1..840 placement/fusion; count={ADVENTURE_IDENTITY_ACTION_COUNT}"),
            ("Decoder", ADVENTURE_IDENTITY_ACTION_DECODER_VERSION),
            ("Observation", f"{ADVENTURE_IDENTITY_OBSERVATION_VERSION}; shape=(4364,)"),
        )
        for row, (label, value) in enumerate(values):
            ttk.Label(contract, text=label).grid(row=row, column=0, sticky="nw", padx=6, pady=3)
            ttk.Label(contract, text=value, wraplength=820).grid(row=row, column=1, sticky="w", padx=6, pady=3)

        config_tools = ttk.LabelFrame(content, text="GUI configuration")
        config_tools.grid(row=3, column=0, sticky="ew", padx=8, pady=3)
        ttk.Label(config_tools, text="Form target").grid(row=0, column=0, padx=6, pady=6)
        ttk.Combobox(
            config_tools,
            textvariable=self.settings_config_target_var,
            values=("Training", "Evaluation", "Streamer"),
            state="readonly",
            width=14,
        ).grid(row=0, column=1, padx=6, pady=6)
        ttk.Button(config_tools, text="Load JSON into active form", command=self.load_active_form_config).grid(
            row=0, column=2, padx=6, pady=6
        )
        ttk.Button(config_tools, text="Save active form JSON", command=self.save_active_form_config).grid(
            row=0, column=3, padx=6, pady=6
        )
        ttk.Button(config_tools, text="Reset canonical defaults", command=self.reset_gui_defaults).grid(
            row=0, column=4, padx=6, pady=6
        )
        ttk.Label(
            config_tools,
            text=(
                "Saved configuration contains form settings and environment-variable names only. "
                "Credential values are never read into or written by the GUI."
            ),
            style="Muted.TLabel",
            wraplength=760,
        ).grid(row=1, column=0, columnspan=5, sticky="w", padx=6, pady=(0, 6))

        inspector = ttk.LabelFrame(content, text="Full-Adventure action inspector")
        inspector.grid(row=4, column=0, sticky="ew", padx=8, pady=3)
        inspector.columnconfigure(2, weight=1)
        ttk.Label(inspector, text="Canonical action ID (0..840)").grid(
            row=0, column=0, sticky="w", padx=6, pady=6
        )
        ttk.Entry(inspector, textvariable=self.action_inspector_id_var, width=12).grid(
            row=0, column=1, sticky="w", padx=6, pady=6
        )
        ttk.Button(inspector, text="Decode", command=self.inspect_action_id).grid(
            row=0, column=2, sticky="w", padx=6, pady=6
        )
        ttk.Label(
            inspector,
            textvariable=self.action_inspector_result_var,
            wraplength=880,
        ).grid(row=1, column=0, columnspan=3, sticky="w", padx=6, pady=(0, 7))

    def refresh_streamer_credentials(self) -> Dict[str, bool]:
        names = TwitchEnvironmentNames(
            client_id=self.streamer_client_id_env_var.get().strip(),
            access_token=self.streamer_access_token_env_var.get().strip(),
            broadcaster_id=self.streamer_broadcaster_id_env_var.get().strip(),
            user_id=self.streamer_user_id_env_var.get().strip(),
            viewer_hash_secret=self.streamer_hash_secret_env_var.get().strip(),
        )
        report = inspect_twitch_credentials(
            names,
            required=self.streamer_platform_var.get() == "twitch",
        )
        variable_by_role = {
            "client_id": self.streamer_client_id_ready_var,
            "access_token": self.streamer_access_token_ready_var,
            "broadcaster_id": self.streamer_broadcaster_id_ready_var,
            "user_id": self.streamer_user_id_ready_var,
            "viewer_hash_secret": self.streamer_hash_secret_ready_var,
        }
        readiness: Dict[str, bool] = {}
        for state in report.variables:
            readiness[state.env_name] = state.present
            if not state.valid_name:
                label = "INVALID"
            elif state.present:
                label = "CONFIGURED"
            elif state.required:
                label = "MISSING"
            else:
                label = "OPTIONAL"
            variable_by_role[state.role].set(label)
        if self.streamer_platform_var.get() != "twitch":
            summary = "Mock source selected; Twitch credential values are not required."
        elif report.issues:
            summary = "; ".join(issue.message for issue in report.issues)
        else:
            summary = "Required Twitch credential variables are configured (values hidden)."
        self.streamer_credentials_summary_var.set(summary)
        return readiness

    def _render_application_status(
        self,
        payload: Dict[str, Any],
        *,
        health: str,
        using_last_good: bool,
    ) -> None:
        """Project canonical status into stable high-level page fields."""

        value = self._first_value

        def gate_text(raw: Any) -> str:
            if isinstance(raw, str):
                enabled_value = raw.strip().lower() in {"1", "true", "yes", "enabled", "on"}
            else:
                enabled_value = bool(raw)
            return "ENABLED" if enabled_value else "DISABLED"
        status = value(payload, ["status", "state"], default="offline")
        mode = value(payload, ["streamer_mode", "mode", "run_mode"], default="-")
        blocked = value(payload, ["blocked_reason", "adventure.blocked_reason"], default="")
        error = value(payload, ["error", "last_error", "twitch_last_error_code"], default="")
        terminal = value(payload, ["terminal_reason", "done_reason"], default="")
        active_run = value(payload, ["active_run", "run_dir"], default=self.active_run_path or "-")
        model_path = value(
            payload,
            ["current_checkpoint", "model_path", "current_model_path", "compatibility.model_path"],
            default="-",
        )
        level = value(
            payload,
            ["adventure.current_level", "current_level", "level", "adventure_level", "next_adventure_level"],
            default="-",
        )
        steps = value(
            payload,
            ["current_model_ppo_steps", "ppo_policy_timesteps", "total_timesteps", "current_timestep"],
            default="-",
        )
        bridge = value(
            payload,
            ["bridge.connected", "bridge_connected", "gameplay.bridge_connected"],
            default="unknown",
        )
        twitch = value(
            payload,
            ["twitch_connection_state", "stream_source_connection_state", "streamer_platform"],
            default="offline",
        )
        stale = " · last good" if using_last_good else ""
        self._set_live_variable(self.header_health_var, f"{health}{stale}")
        self._set_live_variable(self.dashboard_state_var, status)
        self._set_live_variable(self.dashboard_mode_var, mode)
        self._set_live_variable(self.dashboard_level_var, level)
        self._set_live_variable(self.dashboard_timesteps_var, steps)
        self._set_live_variable(self.dashboard_model_var, self._path_for_display(model_path))
        self._set_live_variable(self.dashboard_run_var, self._path_for_display(active_run))
        self._set_live_variable(self.dashboard_bridge_var, bridge)
        self._set_live_variable(self.dashboard_twitch_var, twitch)
        self._set_live_variable(self.dashboard_health_var, f"{health} · {status}")
        warning = blocked or error or terminal or "No current warning or blocked reason."
        self._set_live_variable(self.dashboard_warning_var, warning)

        enabled = bool(value(payload, ["streamer_v1_enabled"], default=False))
        if enabled and (health == "LIVE" or str(health).startswith("BLOCKED_")):
            stream_lifecycle = "LIVE"
        elif self.active_process is not None:
            stream_lifecycle = self.process_lifecycle_var.get()
        else:
            stream_lifecycle = "OFFLINE"
        self._set_live_variable(self.stream_lifecycle_var, stream_lifecycle)
        self._set_live_variable(self.stream_health_var, health)
        self._set_live_variable(self.stream_phase_var, value(payload, ["streamer_mode", "mode"], default="OFFLINE"))
        self._set_live_variable(self.stream_cycle_var, value(payload, ["streamer_cycle", "current_cycle"], default="0"))
        self._set_live_variable(self.stream_active_run_var, self._path_for_display(active_run))
        self._set_live_variable(self.stream_next_level_var, value(payload, ["next_adventure_level", "current_level"], default="-"))
        self._set_live_variable(self.stream_model_steps_var, value(payload, ["current_model_ppo_steps", "model_steps"], default="0"))
        self._set_live_variable(self.stream_baseline_steps_var, value(payload, ["baseline_model_ppo_steps"], default="0"))
        self._set_live_variable(self.stream_eval_countdown_var, value(payload, ["next_evaluation_countdown", "next_evaluation_policy_steps"], default="0"))
        self._set_live_variable(self.stream_connection_var, twitch)
        started = value(payload, ["started_at", "streamer_started_at"], default=None)
        if isinstance(started, (int, float)):
            self._set_live_variable(self.stream_uptime_var, self._duration_text(max(0.0, time.time() - float(started))))
        elif self.active_process_started_at is not None:
            self._set_live_variable(self.stream_uptime_var, self._duration_text(max(0.0, time.monotonic() - self.active_process_started_at)))
        else:
            self._set_live_variable(self.stream_uptime_var, "-")

        queue = value(payload, ["streamer_command_queue"], default={})
        queue = queue if isinstance(queue, dict) else {}
        counters = queue.get("counters") if isinstance(queue.get("counters"), dict) else {}
        self._set_live_variable(self.stream_queue_depth_var, value(payload, ["viewer_command_queue_depth", "streamer_command_queue.depth"], default=0))
        accepted_commands = value(
            payload,
            ["viewer_commands_accepted_count"],
            default=counters.get("accepted", counters.get("enqueued", 0)),
        )
        rejected_commands = value(
            payload,
            [
                "viewer_commands_rejected_count",
                "streamer_permanently_rejected_count",
                "permanently_rejected",
            ],
            default=counters.get("permanently_rejected", 0),
        )
        self._set_live_variable(self.stream_accepted_var, accepted_commands)
        self._set_live_variable(self.stream_rejected_var, rejected_commands)
        total_commands = value(
            payload,
            ["viewer_command_total", "viewer_commands_received", "twitch_notifications_received"],
            default=counters.get("received", counters.get("offered")),
        )
        if total_commands is None:
            try:
                total_commands = int(accepted_commands) + int(rejected_commands)
            except (TypeError, ValueError, OverflowError):
                total_commands = 0
        self._set_live_variable(self.stream_total_commands_var, total_commands)
        self._set_live_variable(self.stream_invalid_var, value(payload, ["viewer_commands_invalid_count", "viewer_command_invalid_count", "invalid_viewer_command_count"], default=counters.get("permanently_rejected", counters.get("invalid", counters.get("parse_rejected", 0)))))
        self._set_live_variable(self.stream_expired_var, value(payload, ["streamer_expired_count", "expired"], default=counters.get("expired", 0)))
        self._set_live_variable(self.stream_interventions_var, value(payload, ["viewer_intervention_count"], default=0))
        self._set_live_variable(self.stream_attempts_var, value(payload, ["viewer_step_attempts"], default=0))
        self._set_live_variable(self.stream_viewers_var, value(payload, ["distinct_hashed_viewer_count"], default=0))
        self._set_live_variable(self.stream_pending_reason_var, value(payload, ["pending_viewer_block_reason"], default="-"))
        self._set_live_variable(self.stream_action_source_var, value(payload, ["last_action_source"], default="MODEL"))
        last_action = value(payload, ["last_viewer_action"], default={})
        last_action = last_action if isinstance(last_action, dict) else {}
        command_type = str(last_action.get("command_type") or "").lower()
        row = last_action.get("requested_row")
        column = last_action.get("requested_col")
        slot = last_action.get("requested_slot")
        try:
            viewer_row = int(row) + 1 if row is not None else None
            viewer_col = int(column) + 1 if column is not None else None
            viewer_slot = int(slot) + 1 if slot is not None else None
        except (TypeError, ValueError, OverflowError):
            viewer_row = viewer_col = viewer_slot = None
        if command_type == "slot" and None not in (viewer_slot, viewer_row, viewer_col):
            latest_command = f"Slot {viewer_slot} at R{viewer_row} C{viewer_col}"
        elif command_type == "plant" and None not in (viewer_row, viewer_col):
            plant = last_action.get("requested_plant") or last_action.get("plant_name") or "Plant"
            latest_command = f"{plant} at R{viewer_row} C{viewer_col}"
        elif command_type in {"fuse", "fuse_tile"} and None not in (viewer_row, viewer_col):
            latest_command = f"Fuse tile at R{viewer_row} C{viewer_col}"
        else:
            latest_command = "-"
        self._set_live_variable(self.stream_latest_command_var, latest_command)
        self._set_live_variable(
            self.stream_last_result_var,
            last_action.get("execution_status") or last_action.get("result") or "-",
        )

        self._set_live_variable(self.stream_ppo_enabled_var, gate_text(value(payload, ["ppo_updates_enabled"], default=bool(enabled and str(mode) == "STREAM_TRAIN"))))
        self._set_live_variable(self.stream_bc_updates_var, gate_text(value(payload, ["bc_updates_enabled"], default=False)))
        self._set_live_variable(self.stream_eval_chat_control_var, gate_text(value(payload, ["evaluation_chat_control"], default=False)))
        self._set_live_variable(self.stream_policy_timesteps_var, value(payload, ["ppo_policy_timesteps", "cycle_policy_steps_completed"], default=0))
        self._set_live_variable(self.stream_environment_actions_var, value(payload, ["total_environment_actions"], default=0))
        self._set_live_variable(self.stream_demo_count_var, value(payload, ["bc_demonstration_count"], default=0))
        self._set_live_variable(self.stream_demo_rejected_var, value(payload, ["bc_demo_rejected_count"], default=0))
        self._set_live_variable(self.stream_bc_update_count_var, value(payload, ["bc_update_count"], default=0))
        self._set_live_variable(self.stream_bc_loss_var, value(payload, ["bc_loss"], default=0))

        self._set_live_variable(self.stream_baseline_checkpoint_status_var, self._path_for_display(value(payload, ["baseline_checkpoint"], default="-")))
        self._set_live_variable(self.stream_current_checkpoint_var, self._path_for_display(value(payload, ["current_checkpoint"], default="-")))
        self._set_live_variable(self.stream_best_checkpoint_var, self._path_for_display(value(payload, ["best_checkpoint"], default="-")))
        self._set_live_variable(self.stream_baseline_eval_var, value(payload, ["baseline_evaluation"], default={}))
        self._set_live_variable(self.stream_current_eval_var, value(payload, ["current_evaluation"], default={}))
        self._set_live_variable(self.stream_best_eval_var, value(payload, ["best_evaluation"], default={}))
        self._set_live_variable(self.stream_evaluation_comparison_var, value(payload, ["evaluation_comparison"], default={}))

        self._set_live_variable(self.stream_game_level_var, level)
        self._set_live_variable(self.stream_wave_var, self._wave_text(payload))
        self._set_live_variable(self.stream_sun_var, value(payload, ["gameplay.sun", "sun", "current_sun"], default="-"))
        self._set_live_variable(self.stream_plants_var, value(payload, ["gameplay.plants", "plants", "plant_count"], default="-"))
        self._set_live_variable(self.stream_zombies_var, value(payload, ["gameplay.zombies", "zombies", "zombie_count"], default="-"))
        live_rows = value(payload, ["board_rows", "gameplay.board_rows", "gameplay.row_count", "row_count"], default=DEFAULT_ROWS)
        live_cols = value(payload, ["board_cols", "gameplay.cols", "cols"], default=DEFAULT_COLS)
        self._set_live_variable(self.stream_board_geometry_var, f"{live_rows}×{live_cols} live · {DEFAULT_ROWS}×{DEFAULT_COLS} policy")
        self._set_live_variable(self.stream_seed_bank_var, value(payload, ["selected_loadout", "current_selected_seed_loadout", "seed_inventory.selected_seeds"], default="-"))
        self._set_live_variable(self.stream_unlocked_var, value(payload, ["unlocked_seeds", "seed_inventory.unlocked_seeds"], default="-"))

    @staticmethod
    def _duration_text(seconds: float) -> str:
        total = max(0, int(seconds))
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
