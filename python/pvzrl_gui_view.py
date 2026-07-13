"""Normalized diagnostics view formatting for the Tk dashboard."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import tkinter as tk
from tkinter import scrolledtext

from pvzrl_gui_status import (
    MISSING,
    DiagnosticsRenderKey,
    LiveStatusReader,
    NormalizedStatusIndex,
    classify_live_health,
    diagnostics_render_key,
)


POLL_MS = 1000


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


class _FallbackStringVar:
    def __init__(self, value: str = "n/a") -> None:
        self._value = str(value)

    def get(self) -> str:
        return self._value

    def set(self, value: Any) -> None:
        self._value = fmt_value(value)


class GuiStatusViewMixin:
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
        self._poll_after_id = None
        if self._closing or self._destroyed:
            return
        self._refresh_diagnostics(auto=True)
        self._schedule_after("_poll_after_id", POLL_MS, self._poll)

    def refresh_diagnostics_now(self) -> None:
        self._refresh_diagnostics(auto=False)

    def _refresh_diagnostics(self, auto: bool) -> None:
        del auto
        try:
            payload, info = self._read_live_status_file()
            if payload is not None:
                self.last_good_status = payload
                if not bool(info.get("unchanged")):
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

    def _status_reader(self) -> LiveStatusReader:
        reader = getattr(self, "_live_status_reader", None)
        if reader is None:
            reader = LiveStatusReader(self.live_status_path)
            self._live_status_reader = reader
        elif reader.path != self.live_status_path:
            reader.set_path(self.live_status_path)
        return reader

    def _read_live_status_file(self) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        return self._status_reader().read()

    def _live_health(self, age: float, payload: Optional[Dict[str, Any]] = None) -> str:
        return classify_live_health(age, payload)

    def _set_live_status(self, info: Dict[str, Any]) -> None:
        path = info.get("path", self.live_status_path)
        exists_text = "yes" if info.get("exists") else "no"
        size_text = self._fmt_size(info.get("size"))
        age_text = self._fmt_age(info.get("age"))
        health = info.get("health", "MISSING")
        self._set_live_variable(
            self.live_status_var,
            f"Live: {path} | exists={exists_text} | size={size_text} | age={age_text} | health={health}",
        )

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
        self._set_live_variable(
            self.diagnostics_status_var,
            f"{prefix} | modified={self._fmt_time(info.get('mtime'))} | "
            f"last_success={self._fmt_time(self.last_good_read_time)} | parse_error={parse_text}",
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
        self._set_coach_live_fields({})
        for title in list(self.panels):
            if title == "Rows":
                self._set_panel(title, "No row diagnostics in live_status.json")
            else:
                self._set_panel(title, message)

    def _render_diagnostics_payload(self, payload: Optional[Dict[str, Any]], health: str, using_last_good: bool) -> None:
        previous_key = getattr(self, "_diagnostics_render_key", None)
        render_key = diagnostics_render_key(
            payload,
            health,
            using_last_good,
            previous=previous_key,
        )
        if render_key == previous_key:
            return
        try:
            if payload is None:
                self._render_no_status(health)
            else:
                self._render(payload, health=health, using_last_good=using_last_good)
            self._diagnostics_render_key = render_key
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
        self._set_coach_live_fields(payload)

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
                    ("fusion_mask", self._first_value(payload, ["fusion_action_mask_enabled", "fusion.fusion_action_mask_enabled", "summary.fusion_action_mask_enabled", "eval.fusion_action_mask_enabled"])),
                    ("wallnut_masked", self._first_value(payload, ["summary.wallnut_actions_masked", "eval.wallnut_actions_masked"])),
                    ("cherry_masked", self._first_value(payload, ["summary.cherrybomb_actions_masked", "eval.cherrybomb_actions_masked"])),
                    ("decoder", self._first_value(payload, ["agent.action_decoder_version", "action_decoder_version", "compatibility.action_decoder_version"])),
                    ("obs", self._first_value(payload, ["agent.observation_version", "observation_version", "compatibility.observation_version"])),
                ]
            ),
        )
        self._set_panel("Rows", self._row_panel_lines(payload))
        self._set_panel(
            "Reward Breakdown",
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
                    ("action_mask", self._first_value(payload, ["fusion.fusion_action_mask_enabled", "fusion_action_mask_enabled"])),
                    ("available", self._first_value(payload, ["fusion.fusion_available", "fusion_available"])),
                    ("candidates", self._first_value(payload, ["fusion.fusion_candidate_count", "fusion_candidate_count"])),
                    ("mask_legal", self._first_value(payload, ["fusion.fusion_actions_available_count", "fusion_actions_available_count", "mask_diagnostics.fusion_actions_available_count"])),
                    ("mask_incompat", self._first_value(payload, ["fusion.fusion_actions_masked_incompatible_count", "fusion_actions_masked_incompatible_count", "mask_diagnostics.fusion_actions_masked_incompatible_count"])),
                    ("top", self._first_value(payload, ["fusion.fusion_top_candidate", "fusion_top_candidate"])),
                    ("attempts", self._first_value(payload, ["fusion.fusion_attempted_count", "fusion_attempted_count"])),
                    ("successes", self._first_value(payload, ["fusion.fusion_success_count", "fusion_success_count"])),
                    ("failures", self._first_value(payload, ["fusion.fusion_failed_count", "fusion_failed_count"])),
                    ("rejections", self._first_value(payload, ["fusion.fusion_rejected_count", "fusion_rejected_count"])),
                    ("last_reject", self._first_value(payload, ["fusion.fusion_last_rejected_reason", "fusion_last_rejected_reason"])),
                    ("last_incompat", self._first_value(payload, ["fusion.fusion_last_incompatible_pair", "fusion_last_incompatible_pair"])),
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
        self._set_panel(
            "Action Distribution",
            lines_from_pairs(
                [
                    ("actions", self._first_value(payload, ["action_distribution", "summary.action_distribution", "eval.action_distribution"])),
                    ("legal", self._first_value(payload, ["legal_action_count", "agent.legal_action_count"])),
                    ("illegal", self._first_value(payload, ["illegal_actions", "eval.illegal_actions"])),
                    ("last", self._first_value(payload, ["last_action", "agent.last_action"])),
                ]
            ),
        )
        self._set_panel(
            "Plant Usage",
            lines_from_pairs(
                [
                    ("by_type", self._first_value(payload, ["plants_by_type", "summary.plants_by_type", "eval.plants_by_type"])),
                    ("sunflowers", self._first_value(payload, ["summary.sunflowers_planted", "eval.sunflowers_planted"])),
                    ("peashooters", self._first_value(payload, ["summary.peashooters_planted", "eval.peashooters_planted"])),
                    ("wallnuts", self._first_value(payload, ["summary.wallnuts_planted", "eval.wallnuts_planted"])),
                    ("cherrybombs", self._first_value(payload, ["summary.cherrybombs_planted", "eval.cherrybombs_planted"])),
                ]
            ),
        )
        self._set_panel("Fusion Usage", self.last_panel_content.get("Fusion", "No fusion diagnostics available."))
        self._set_panel(
            "Human Interventions",
            lines_from_pairs(
                [
                    ("command_mode", self._first_value(payload, ["human_coach_command_mode", "human_coach.human_coach_command_mode"], default=self.assisted_execution_mode_var.get())),
                    ("last_command", self._first_value(payload, ["human_coach_last_command", "human_coach.command"])),
                    ("last_action", self._first_value(payload, ["human_coach_last_action", "human_coach.selected_action"])),
                    ("overrides", self._first_value(payload, ["human_coach_override_count", "stream_coach_override_count"])),
                    ("matches", self._first_value(payload, ["human_coach_match_count", "stream_coach_match_count"])),
                    ("rejected", self._first_value(payload, ["human_coach_rejected_count", "stream_coach_rejected_count"])),
                    ("local_queue", self.assisted_command_queue.counts()),
                    ("log", self.intervention_log_path_var.get()),
                ]
            ),
        )
        self._set_panel(
            "Failed Episodes",
            lines_from_pairs(
                [
                    ("losses", self._first_value(payload, ["adventure.losses", "eval.losses", "losses"])),
                    ("reset_failures", self._first_value(payload, ["eval.reset_failures", "reset_failures"])),
                    ("bridge_errors", self._first_value(payload, ["eval.bridge_errors", "bridge_errors"])),
                    ("blocked_reason", self._first_value(payload, ["blocked_reason", "adventure.blocked_reason"])),
                    ("last_result", self._first_value(payload, ["last_result", "adventure.last_result"])),
                ]
            ),
        )
        self._set_panel(
            "Viewer Queue Stats",
            lines_from_pairs(
                [
                    ("local", self.assisted_command_queue.counts()),
                    ("pending_stream", self._first_value(payload, ["pending_stream_commands", "stream_coach.pending_count"])),
                    ("messages_seen", self._first_value(payload, ["mock_stream_messages_seen", "stream_coach_messages_seen"])),
                    ("accepted", self._first_value(payload, ["mock_stream_commands_accepted", "stream_coach_commands_accepted"])),
                    ("rejected", self._first_value(payload, ["mock_stream_commands_rejected", "stream_coach_commands_rejected"])),
                    ("top", self._first_value(payload, ["stream_coach_top_commands"])),
                ]
            ),
        )

    def _live_value_text(self, payload: Dict[str, Any], paths: List[str], default: str = "n/a") -> str:
        value = self._first_value(payload, paths, default=MISSING)
        if value is MISSING or value in (None, ""):
            return default
        return fmt_value(value)

    def _set_live_variable(self, variable: Any, value: Any) -> bool:
        target = str(value)
        try:
            current = str(variable.get())
        except (AttributeError, tk.TclError):
            current = ""
        if current == target:
            return False
        variable.set(value)
        return True

    def _set_coach_live_fields(self, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            payload = {}
        for var_name in (
            "human_coach_enabled_status_var",
            "human_coach_last_command_var",
            "human_coach_last_action_var",
            "human_coach_last_error_var",
            "human_coach_override_count_var",
            "human_coach_match_count_var",
            "human_coach_reward_total_var",
            "stream_coach_enabled_status_var",
            "stream_coach_platform_status_var",
            "stream_coach_dry_run_status_var",
            "stream_coach_alive_status_var",
            "stream_coach_last_message_var",
            "stream_coach_last_parsed_command_var",
            "stream_coach_last_applied_command_var",
            "stream_coach_accept_reject_var",
            "stream_coach_last_reject_reason_var",
            "stream_coach_pending_count_var",
            "stream_coach_top_command_var",
            "stream_coach_last_selected_command_var",
            "stream_coach_last_action_var",
            "stream_coach_rejected_count_var",
            "stream_coach_last_vote_count_var",
            "stream_coach_override_count_var",
            "stream_coach_match_count_var",
            "stream_coach_reward_total_var",
            "fusion_bridge_available_var",
            "fusion_bridge_enabled_status_var",
            "fusion_last_command_var",
            "fusion_last_result_var",
            "fusion_attempt_count_var",
            "fusion_success_count_var",
            "fusion_failure_count_var",
            "fusion_rejected_count_var",
        ):
            if not hasattr(self, var_name):
                setattr(self, var_name, _FallbackStringVar())
        self._set_live_variable(self.human_coach_enabled_status_var,
            self._live_value_text(payload, ["human_coach_enabled", "human_coach.human_coach_enabled"])
        )
        self._set_live_variable(self.human_coach_last_command_var,
            self._live_value_text(
                payload,
                [
                    "human_coach_last_command",
                    "human_coach.command",
                    "human_coach.last_command",
                ],
            )
        )
        self._set_live_variable(self.human_coach_last_action_var,
            self._live_value_text(
                payload,
                [
                    "human_coach_last_action",
                    "human_coach.last_action",
                    "human_coach.validation.policy_action",
                ],
            )
        )
        self._set_live_variable(self.human_coach_last_error_var,
            self._live_value_text(
                payload,
                [
                    "human_coach_last_error",
                    "human_coach.last_error",
                    "human_coach_last_rejected_reason",
                    "human_coach.last_rejected_reason",
                    "human_coach.rejected_reason",
                    "fusion_last_rejected_reason",
                    "fusion_last_bridge_result_reason",
                    "stream_fusion_last_bridge_result_reason",
                ],
            )
        )
        self._set_live_variable(self.human_coach_override_count_var,
            self._live_value_text(payload, ["human_coach_override_count", "human_coach.override_count"])
        )
        self._set_live_variable(self.human_coach_match_count_var,
            self._live_value_text(payload, ["human_coach_match_count", "human_coach.match_count"])
        )
        self._set_live_variable(self.human_coach_reward_total_var,
            self._live_value_text(payload, ["human_coach_reward_total", "human_coach.reward_total"])
        )

        self._set_live_variable(self.stream_coach_enabled_status_var,
            self._live_value_text(payload, ["coach.stream_coach_enabled", "stream_coach.stream_coach_enabled", "stream_coach_enabled"])
        )
        self._set_live_variable(self.stream_coach_platform_status_var,
            self._live_value_text(payload, ["coach.stream_coach_mode", "stream_coach.stream_coach_mode", "stream_coach_mode", "coach.stream_coach_platform", "stream_coach_platform", "stream_coach.stream_coach_platform"])
        )
        dry_run = self._first_value(
            payload,
            ["coach.stream_coach_dry_run", "stream_coach.stream_coach_dry_run", "stream_coach_dry_run"],
            default="n/a",
        )
        apply_enabled = self._first_value(
            payload,
            ["coach.stream_coach_apply_enabled", "stream_coach.stream_coach_apply_enabled", "stream_coach_apply_enabled"],
            default="n/a",
        )
        dry_run_status_var = getattr(self, "stream_coach_dry_run_status_var", None)
        if dry_run_status_var is not None:
            self._set_live_variable(dry_run_status_var, f"dry-run={fmt_value(dry_run)} / apply={fmt_value(apply_enabled)}")
        self._set_live_variable(self.stream_coach_alive_status_var,
            self._live_value_text(payload, ["coach.stream_coach_alive_status", "stream_coach.stream_coach_alive_status", "stream_coach_alive_status", "coach.stream_coach_alive", "stream_coach.stream_coach_alive", "stream_coach_alive"])
        )
        self._set_live_variable(self.stream_coach_last_message_var,
            self._live_value_text(payload, ["coach.last_stream_message", "stream_coach.last_stream_message", "last_stream_message", "coach.stream_coach_last_message", "stream_coach_last_message", "stream_coach.stream_coach_last_message"])
        )
        self._set_live_variable(self.stream_coach_last_parsed_command_var,
            self._live_value_text(payload, ["coach.last_stream_parsed_command", "stream_coach.last_stream_parsed_command", "last_stream_parsed_command", "coach.stream_coach_last_parsed_command", "stream_coach_last_parsed_command", "stream_coach.stream_coach_last_parsed_command"])
        )
        self._set_live_variable(self.stream_coach_last_applied_command_var,
            self._live_value_text(payload, ["coach.last_applied_coach_command", "stream_coach.last_applied_coach_command", "last_applied_coach_command", "coach.stream_coach_last_applied_command", "stream_coach_last_applied_command", "stream_coach.stream_coach_last_applied_command"])
        )
        accepted = self._first_value(
            payload,
            ["coach.stream_commands_accepted", "stream_coach.stream_commands_accepted", "stream_commands_accepted", "coach.mock_stream_commands_accepted", "mock_stream_commands_accepted", "coach.stream_coach_commands_accepted", "stream_coach_commands_accepted", "stream_coach.stream_coach_commands_accepted"],
            default="n/a",
        )
        rejected = self._first_value(
            payload,
            ["coach.stream_commands_rejected", "stream_coach.stream_commands_rejected", "stream_commands_rejected", "coach.mock_stream_commands_rejected", "mock_stream_commands_rejected", "coach.stream_coach_commands_rejected", "stream_coach_commands_rejected", "stream_coach.stream_coach_commands_rejected"],
            default="n/a",
        )
        status = self._first_value(
            payload,
            ["coach.last_stream_command_status", "stream_coach.last_stream_command_status", "last_stream_command_status", "coach.stream_coach_last_command_status", "stream_coach_last_command_status", "stream_coach.stream_coach_last_command_status"],
            default="n/a",
        )
        self._set_live_variable(self.stream_coach_accept_reject_var, f"{accepted} / {rejected} ({status})")
        self._set_live_variable(self.stream_coach_last_reject_reason_var,
            self._live_value_text(payload, ["coach.last_stream_reject_reason", "stream_coach.last_stream_reject_reason", "last_stream_reject_reason", "coach.stream_coach_last_reject_reason", "stream_coach_last_reject_reason", "stream_coach.stream_coach_last_reject_reason"])
        )
        self._set_live_variable(self.stream_coach_pending_count_var,
            self._live_value_text(payload, ["coach.pending_stream_commands", "stream_coach.pending_stream_commands", "pending_stream_commands", "stream_coach_pending_message_count"])
        )
        self._set_live_variable(self.stream_coach_top_command_var,
            self._live_value_text(payload, ["coach.stream_coach_top_commands", "stream_coach.stream_coach_top_commands", "stream_coach_top_commands", "stream_coach.top_commands"])
        )
        self._set_live_variable(self.stream_coach_last_selected_command_var,
            self._live_value_text(payload, ["coach.last_validated_coach_command", "stream_coach.last_validated_coach_command", "last_validated_coach_command", "coach.stream_coach_last_command", "stream_coach_last_command", "stream_coach.stream_coach_last_command"])
        )
        self._set_live_variable(self.stream_coach_last_action_var,
            self._live_value_text(payload, ["coach.stream_coach_last_action", "stream_coach.stream_coach_last_action", "stream_coach_last_action"])
        )
        self._set_live_variable(self.stream_coach_rejected_count_var,
            self._live_value_text(payload, ["coach.stream_coach_rejected_count", "stream_coach.stream_coach_rejected_count", "stream_coach_rejected_count"])
        )
        self._set_live_variable(self.stream_coach_last_vote_count_var,
            self._live_value_text(payload, ["coach.stream_coach_last_vote_count", "stream_coach.stream_coach_last_vote_count", "stream_coach_last_vote_count"])
        )
        self._set_live_variable(self.stream_coach_override_count_var,
            self._live_value_text(payload, ["coach.stream_coach_override_count", "stream_coach.stream_coach_override_count", "stream_coach_override_count"])
        )
        self._set_live_variable(self.stream_coach_match_count_var,
            self._live_value_text(payload, ["coach.stream_coach_match_count", "stream_coach.stream_coach_match_count", "stream_coach_match_count"])
        )
        self._set_live_variable(self.stream_coach_reward_total_var,
            self._live_value_text(payload, ["coach.stream_coach_reward_total", "stream_coach.stream_coach_reward_total", "stream_coach_reward_total"])
        )

        self._set_live_variable(self.fusion_bridge_enabled_status_var,
            self._live_value_text(payload, ["fusion_bridge_enabled", "fusion.fusion_bridge_enabled"])
        )
        self._set_live_variable(self.fusion_bridge_available_var,
            self._live_value_text(payload, ["fusion_bridge_available", "fusion.fusion_bridge_available", "fusion.fusion_available"])
        )
        self._set_live_variable(self.fusion_last_command_var,
            self._live_value_text(payload, ["fusion_last_command", "fusion.fusion_last_command", "stream_fusion_last_command", "stream_coach.stream_fusion_last_command"])
        )
        self._set_live_variable(self.fusion_last_result_var,
            self._live_value_text(payload, ["fusion_last_result", "fusion.fusion_last_result", "stream_fusion_last_result", "stream_coach.stream_fusion_last_result"])
        )
        self._set_live_variable(self.fusion_attempt_count_var,
            self._live_value_text(payload, ["fusion_attempted_count", "fusion.fusion_attempted_count", "stream_coach_fusion_attempt_count", "stream_coach.stream_coach_fusion_attempt_count"])
        )
        self._set_live_variable(self.fusion_success_count_var,
            self._live_value_text(payload, ["fusion_success_count", "fusion.fusion_success_count", "stream_coach_fusion_success_count", "stream_coach.stream_coach_fusion_success_count"])
        )
        self._set_live_variable(self.fusion_failure_count_var,
            self._live_value_text(payload, ["fusion_failed_count", "fusion.fusion_failed_count", "stream_coach_fusion_failure_count", "stream_coach.stream_coach_fusion_failure_count"])
        )
        self._set_live_variable(self.fusion_rejected_count_var,
            self._live_value_text(payload, ["fusion_rejected_count", "fusion.fusion_rejected_count", "stream_coach_fusion_rejected_count", "stream_coach.stream_coach_fusion_rejected_count"])
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
                ("fusion_mask", self._adventure_first(payload, ["fusion_action_mask_enabled", "fusion.fusion_action_mask_enabled", "summary.fusion_action_mask_enabled", "eval.fusion_action_mask_enabled"])),
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
                    "startup_validation",
                    self._first_value(payload, ["startup_validation_reason", "adventure.startup_validation_reason"], default="n/a"),
                ),
                (
                    "level_identity",
                    self._first_value(payload, ["level_identity_reliable", "adventure.level_identity_reliable"], default="n/a"),
                ),
                (
                    "wrapper_expected_level",
                    self._first_value(payload, ["wrapper_expected_level", "adventure.wrapper_expected_level"], default="n/a"),
                ),
                (
                    "bridge_detected_level",
                    self._first_value(payload, ["bridge_detected_level", "adventure.bridge_detected_level"], default="n/a"),
                ),
                (
                    "profile_adventure_level",
                    self._first_value(payload, ["profile_adventure_level", "adventure.profile_adventure_level"], default="n/a"),
                ),
                (
                    "screen_state",
                    self._first_value(payload, ["screenState", "adventure.screenState", "screen.screen_state"], default="n/a"),
                ),
                (
                    "expected_transition",
                    self._first_value(payload, ["expected_transition_target", "adventure.expected_transition_target"], default="n/a"),
                ),
                (
                    "seed_selection_expected",
                    self._first_value(payload, ["seed_selection_expected", "adventure.seed_selection_expected"], default="n/a"),
                ),
                ("replay_support", replay_status),
                ("replay_seed_attempts", self._first_value(payload, ["frontier_replay_seed_selection_attempts", "adventure.frontier_replay_seed_selection_attempts"], default="n/a")),
                ("replay_seed_ok", self._first_value(payload, ["frontier_replay_last_seed_selection_ok", "adventure.frontier_replay_last_seed_selection_ok"], default="n/a")),
                ("replay_seed_msg", self._first_value(payload, ["frontier_replay_last_seed_selection_message", "adventure.frontier_replay_last_seed_selection_message"], default="n/a")),
                ("latest_unlock", self._first_value(payload, ["latest_unlock", "unlock.latest_unlock", "adventure.latest_unlock"], default="n/a")),
                ("unlocked_seeds", self._first_value(payload, ["unlocked_seeds", "unlock.unlocked_seeds", "adventure.unlocked_seeds"], default="n/a")),
                ("eligible_seeds", self._first_value(payload, ["eligible_seeds", "unlock.eligible_seeds", "adventure.eligible_seeds"], default="n/a")),
                ("selectable_seeds", self._first_value(payload, ["selectable_seeds", "unlock.selectable_seeds", "adventure.selectable_seeds"], default="n/a")),
                ("observed_capacity", self._first_value(payload, ["observed_seed_bank_capacity", "adventure.observed_seed_bank_capacity"], default="n/a")),
                ("effective_capacity", self._first_value(payload, ["effective_seed_capacity", "adventure.effective_seed_capacity"], default="n/a")),
                ("bridge_capacity", self._first_value(payload, ["bridge_reported_capacity", "adventure.bridge_reported_capacity"], default="n/a")),
                ("inferred_capacity", self._first_value(payload, ["inferred_capacity_from_unlocks", "adventure.inferred_capacity_from_unlocks"], default="n/a")),
                ("capacity_source", self._first_value(payload, ["inferred_capacity_source", "adventure.inferred_capacity_source"], default="n/a")),
                ("capacity_reason", self._first_value(payload, ["capacity_inference_reason", "adventure.capacity_inference_reason"], default="n/a")),
                ("rejected_priority", self._first_value(payload, ["rejected_priority_seeds", "adventure.rejected_priority_seeds"], default="n/a")),
                ("configured_seed_list", self._first_value(payload, ["configured_seed_list", "adventure.configured_seed_list"], default="n/a")),
                ("selected_loadout", self._first_value(payload, ["selected_loadout", "adventure.selected_loadout"], default="n/a")),
                ("selected_count", self._first_value(payload, ["selected_loadout_count", "adventure.selected_loadout_count"], default="n/a")),
                ("inactive_slots", self._first_value(payload, ["inactive_model_slots", "adventure.inactive_model_slots"], default="n/a")),
                ("seed_order_source", self._first_value(payload, ["seed_order_source", "adventure.seed_order_source"], default="n/a")),
                ("seed_order_preserved", self._first_value(payload, ["seed_order_preserved", "adventure.seed_order_preserved"], default="n/a")),
                ("seed_order_block", self._first_value(payload, ["seed_order_blocked_reason", "adventure.seed_order_blocked_reason"], default="n/a")),
                ("seed_order_warning", self._first_value(payload, ["seed_order_warning", "adventure.seed_order_warning"], default="n/a")),
                ("loadout_reason", self._first_value(payload, ["loadout_reason", "adventure.loadout_reason"], default="n/a")),
                ("last_reset_reason", self._first_value(payload, ["last_reset_reason", "adventure.last_reset_reason"], default="n/a")),
                ("post_win_block", self._first_value(payload, ["post_win_blocked_reason", "adventure.post_win_blocked_reason"], default="n/a")),
                ("stream_enabled", self._first_value(payload, ["stream_coach_enabled", "human_coach_enabled"], default="n/a")),
                ("stream_mode", self._first_value(payload, ["stream_coach_mode", "stream_coach_platform", "human_coach_platform"], default="n/a")),
                ("stream_alive", self._first_value(payload, ["stream_coach_alive_status", "stream_coach_alive"], default="n/a")),
                ("stream_script", self._first_value(payload, ["mock_stream_script", "stream_coach.mock_stream_script"], default="n/a")),
                ("stream_seen", self._first_value(payload, ["mock_stream_messages_seen", "stream_coach_messages_seen"], default="n/a")),
                ("stream_parsed", self._first_value(payload, ["mock_stream_commands_parsed", "stream_coach_commands_parsed"], default="n/a")),
                ("stream_accepted", self._first_value(payload, ["mock_stream_commands_accepted", "stream_coach_commands_accepted"], default="n/a")),
                ("stream_rejected_msgs", self._first_value(payload, ["mock_stream_commands_rejected", "stream_coach_commands_rejected"], default="n/a")),
                ("stream_pending", self._first_value(payload, ["pending_stream_commands"], default="n/a")),
                ("stream_last_msg", self._first_value(payload, ["last_stream_message", "stream_coach_last_message"], default="n/a")),
                ("stream_last_parsed", self._first_value(payload, ["last_stream_parsed_command", "stream_coach_last_parsed_command"], default="n/a")),
                ("stream_last_status", self._first_value(payload, ["last_stream_command_status", "stream_coach_last_command_status"], default="n/a")),
                ("stream_last_reason", self._first_value(payload, ["last_stream_reject_reason", "stream_coach_last_reject_reason"], default="n/a")),
                ("stream_last_cmd", self._first_value(payload, ["stream_coach_last_command", "human_coach_last_command"], default="n/a")),
                ("stream_last_applied", self._first_value(payload, ["last_applied_coach_command", "stream_coach_last_applied_command"], default="n/a")),
                ("stream_last_action", self._first_value(payload, ["stream_coach_last_action", "human_coach_last_action"], default="n/a")),
                ("stream_top", self._first_value(payload, ["stream_coach_top_commands"], default="n/a")),
                ("stream_last_reject", self._first_value(payload, ["stream_coach_last_rejected_command", "human_coach_last_rejected_command"], default="n/a")),
                ("stream_votes", self._first_value(payload, ["stream_coach_last_vote_count", "human_coach_last_vote_count"], default="n/a")),
                ("stream_overrides", self._first_value(payload, ["stream_coach_override_count", "human_coach_override_count"], default="n/a")),
                ("stream_matches", self._first_value(payload, ["stream_coach_match_count", "human_coach_match_count"], default="n/a")),
                ("stream_rejected", self._first_value(payload, ["stream_coach_rejected_count", "human_coach_rejected_count"], default="n/a")),
                ("stream_reward", self._first_value(payload, ["stream_coach_reward_total", "human_coach_reward_total"], default="n/a")),
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

    def _status_index_for(self, payload: Any) -> NormalizedStatusIndex:
        index = getattr(self, "_normalized_status_index", None)
        if index is not None and index.contains(payload):
            return index
        index = NormalizedStatusIndex(payload)
        self._normalized_status_index = index
        return index

    def _lookup_path(self, payload: Any, path: str) -> Any:
        return self._status_index_for(payload).lookup(payload, path)

    def _first_value(self, payload: Any, paths: List[str], default: Any = None) -> Any:
        return self._status_index_for(payload).first(payload, paths, default)

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
