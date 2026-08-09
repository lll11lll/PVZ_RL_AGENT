"""Focused tests for Adventure Generalist 14-slot identity unlock adoption."""

from __future__ import annotations

import json
import inspect
import io
import tempfile
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List

import pvzrl_adventure as adventure_module
import pvzrl_adventure_generalist as generalist_module
from pvzrl_action_space import (
    ACTION_SPACE_ADVENTURE_14_IDENTITY,
    ADVENTURE_IDENTITY_ACTION_COUNT,
    adventure_identity_action_to_slot_cell,
    build_action_space_spec,
    structural_adventure_identity_mask,
)
from pvzrl_adventure_generalist import (
    ADVENTURE_GENERALIST_INITIAL_LOADOUT,
    AdventureGeneralistTrainingEnv,
    AdventureSeedCurriculum,
    BLOCKED_INITIAL_LOADOUT_UNAVAILABLE,
    SEED_ORDER_SOURCE_EXPLICIT,
)
from pvzrl_env import PvZGymEnv, decode_action
from pvzrl_gui import (
    ADVENTURE_GENERALIST_INITIAL_LOADOUT as GUI_INITIAL_LOADOUT,
    PvZDashboard,
)
from pvzrl_model_metadata import write_model_metadata
from pvzrl_seed_inventory import (
    ADVENTURE_IDENTITY_FEATURES_PER_SLOT,
    ADVENTURE_IDENTITY_ONE_HOT_WIDTH,
    ADVENTURE_IDENTITY_SCALAR_FEATURES_PER_SLOT,
    adventure_identity_feature_count,
    adventure_identity_features,
)
from train_ppo import build_config, validate_adventure_generalist_model_compatibility


class Var:
    def __init__(self, value: str | bool) -> None:
        self.value = value

    def get(self) -> str | bool:
        return self.value

    def set(self, value: str | bool) -> None:
        self.value = value


class DefaultArgs(Namespace):
    def __getattr__(self, _name: str) -> Any:
        return None


def slot(plant_type: int, name: str, index: int) -> Dict[str, Any]:
    return {
        "slotIndex": index,
        "plantType": plant_type,
        "plantTypeName": name,
        "displayName": name,
        "seedCost": 50 if name == "SunFlower" else 100,
        "currentCooldown": 0,
        "rawCooldown": 0,
        "fullCooldown": 7.5,
        "ready": True,
        "usable": True,
    }


def observation(seed_slots: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "sun": 500,
        "wave": 1,
        "maxWave": 10,
        "plantCount": 0,
        "zombieCount": 0,
        "gameplayReady": True,
        "rowCount": 5,
        "columnCount": 10,
        "seedSlots": seed_slots,
        "legalActions": [0, 1],
        "unlockedSeedNames": ["SunFlower", "Peashooter", "WallNut", "CherryBomb"],
    }


def fake_dashboard() -> PvZDashboard:
    dashboard = PvZDashboard.__new__(PvZDashboard)
    dashboard.live_status_path = Path("runs/live_status.json")
    dashboard.project_root = Path.cwd()
    dashboard.repo_root = Path.cwd()
    dashboard.active_run_path = ""
    dashboard.generalist_total_timesteps_var = Var("1234")
    dashboard.generalist_checkpoint_freq_var = Var("250")
    dashboard.generalist_initial_loadout_var = Var(GUI_INITIAL_LOADOUT)
    dashboard.generalist_max_seed_slots_var = Var("14")
    dashboard.generalist_start_level_var = Var("1")
    dashboard.generalist_max_levels_var = Var("3")
    dashboard.generalist_max_attempts_var = Var("2")
    dashboard.generalist_game_speed_var = Var("4.0")
    dashboard.generalist_step_seconds_var = Var("0.05")
    dashboard.generalist_board_timeout_var = Var("60")
    dashboard.generalist_soft_max_steps_var = Var("2000")
    dashboard.generalist_hard_max_steps_var = Var("3500")
    dashboard.generalist_frontier_prob_var = Var("0.60")
    dashboard.generalist_recent_prob_var = Var("0.30")
    dashboard.generalist_maintenance_prob_var = Var("0.10")
    dashboard.generalist_frontier_win_streak_required_var = Var("1")
    dashboard.generalist_unlock_delay_var = Var("0")
    dashboard.generalist_new_plant_prob_var = Var("0.15")
    dashboard.generalist_run_dir_var = Var("")
    dashboard.generalist_resume_model_path_var = Var("")
    dashboard.generalist_eval_model_path_var = Var("")
    dashboard.generalist_unlock_curriculum_var = Var(True)
    dashboard.generalist_replay_cleared_var = Var(True)
    dashboard.generalist_final_wave_extension_var = Var(True)
    dashboard.generalist_wait_gameplay_ready_var = Var(True)
    dashboard.generalist_quick_wait_var = Var(True)
    dashboard.generalist_tactical_masks_var = Var(True)
    dashboard.generalist_wallnut_mask_var = Var(True)
    dashboard.generalist_cherrybomb_mask_var = Var(True)
    dashboard.generalist_fusion_action_mask_train_var = Var(True)
    dashboard.generalist_fusion_action_mask_eval_var = Var(False)
    dashboard.generalist_curriculum_var = Var("conservative")
    dashboard.generalist_randomize_seed_order_var = Var(False)
    dashboard.human_coach_enabled_var = Var(False)
    dashboard.human_coach_reward_var = Var(False)
    dashboard.human_coach_bonus_var = Var("")
    dashboard.human_coach_match_bonus_var = Var("")
    dashboard.human_coach_override_penalty_var = Var("")
    dashboard.human_coach_fusion_reward_var = Var("")
    dashboard.human_coach_tactical_reward_var = Var("")
    dashboard.human_coach_log_path_var = Var("runs/human_coach.jsonl")
    dashboard.human_coach_command_path_var = Var("runs/coach_commands.jsonl")
    dashboard.human_coach_command_input_var = Var("")
    dashboard.stream_coach_enabled_var = Var(False)
    dashboard.stream_coach_platform_var = Var("mock")
    dashboard.stream_coach_window_sec_var = Var("3")
    dashboard.stream_coach_min_votes_var = Var("2")
    dashboard.stream_coach_max_actions_per_minute_var = Var("20")
    dashboard.stream_coach_reward_var = Var(False)
    dashboard.stream_coach_log_path_var = Var("runs/stream_coach.jsonl")
    dashboard.coach_allow_fusion_planning_var = Var(False)
    dashboard.fusion_bridge_enabled_var = Var(False)
    dashboard.human_coach_enabled_status_var = Var("n/a")
    dashboard.human_coach_last_command_var = Var("n/a")
    dashboard.human_coach_last_action_var = Var("n/a")
    dashboard.human_coach_last_error_var = Var("n/a")
    dashboard.human_coach_override_count_var = Var("n/a")
    dashboard.human_coach_match_count_var = Var("n/a")
    dashboard.human_coach_reward_total_var = Var("n/a")
    dashboard.stream_coach_enabled_status_var = Var("n/a")
    dashboard.stream_coach_platform_status_var = Var("n/a")
    dashboard.stream_coach_top_command_var = Var("n/a")
    dashboard.stream_coach_last_selected_command_var = Var("n/a")
    dashboard.stream_coach_last_action_var = Var("n/a")
    dashboard.stream_coach_rejected_count_var = Var("n/a")
    dashboard.stream_coach_last_vote_count_var = Var("n/a")
    dashboard.stream_coach_override_count_var = Var("n/a")
    dashboard.stream_coach_match_count_var = Var("n/a")
    dashboard.stream_coach_reward_total_var = Var("n/a")
    dashboard.fusion_bridge_enabled_status_var = Var("n/a")
    dashboard.fusion_bridge_available_var = Var("n/a")
    dashboard.fusion_last_command_var = Var("n/a")
    dashboard.fusion_last_result_var = Var("n/a")
    dashboard.fusion_success_count_var = Var("n/a")
    dashboard.fusion_rejected_count_var = Var("n/a")
    dashboard.coach_queue_status_var = Var("Queue idle")
    dashboard.last_generalist_status_content = ""
    dashboard.generalist_status_text = None
    dashboard._append_log = lambda _text: None
    dashboard.launch_process = lambda _name, _command: None
    return dashboard


def fake_generalist_env(tmp_dir: Path) -> AdventureGeneralistTrainingEnv:
    env = AdventureGeneralistTrainingEnv.__new__(AdventureGeneralistTrainingEnv)
    env.run_dir = tmp_dir
    env.progress_jsonl_path = tmp_dir / "adventure_training_progress.jsonl"
    env.plant_unlocks_path = tmp_dir / "plant_unlocks.json"
    env.seed_slot_unlocks_path = tmp_dir / "seed_slot_unlocks.json"
    env.max_seed_slots = 14
    env.current_level = 1
    env.current_attempt = 1
    env.episode_index = 0
    env.current_sample_source = "frontier"
    env.configured_seed_list = list(ADVENTURE_GENERALIST_INITIAL_LOADOUT)
    env.current_loadout = list(ADVENTURE_GENERALIST_INITIAL_LOADOUT)
    env.current_loadout_reason = "initial"
    env.seed_order_source = "default_canonical"
    env.randomize_seed_order = False
    env.current_seed_order_source = "default_canonical"
    env.current_seed_order_preserved = True
    env.current_seed_order_blocked_reason = ""
    env.current_selectable_seeds = ["SunFlower", "Peashooter"]
    env.current_excluded_new_plants = []
    env.observed_seed_bank_capacity = 4
    env.bridge_reported_capacity = None
    env.inferred_capacity_from_unlocks = 4
    env.effective_seed_capacity = 4
    env.max_effective_seed_capacity_seen = 4
    env.inferred_capacity_source = "initial_starter_loadout"
    env.capacity_inference_reason = "initial starter loadout"
    env.available_priority_seeds = []
    env.rejected_priority_seeds = []
    env.confirmed_unlock_event_seeds = []
    env.infer_capacity_from_unlocks = True
    env.allow_weak_unlocked_capacity_fallback = False
    env.cleared_levels = []
    env.adventure_start_level = 1
    env.max_adventure_levels = 5
    env.max_attempts_per_level = 3
    env.frontier_win_streak_required = 1
    env.frontier_win_streak = 0
    env.frontier_mastered_levels = []
    env.frontier_replay_supported = True
    env.frontier_replay_blocked_reason = ""
    env.frontier_mastery_reset_reason = ""
    env.frontier_mastery_ready = False
    env.frontier_promoted_this_episode = False
    env._hard_blocked_reason = ""
    env.sample_probs = {"frontier": 0.6, "recent_cleared": 0.3, "maintenance": 0.1}
    env.replay_cleared_levels = False
    env.config = type(
        "Cfg",
        (),
        {
            "gameplay_ready_timeout": 8.0,
            "poll_seconds": 0.001,
            "start_sun": None,
            "seed_list": list(ADVENTURE_GENERALIST_INITIAL_LOADOUT),
            "plant_types": [1, 1, 0, 0],
            "tactical_masks": False,
            "wallnut_tactical_mask": False,
            "cherrybomb_tactical_mask": False,
        },
    )()
    env._last_observation = {}
    env._episode_reward_totals = {}
    env._episode_reward = 0.0
    env.action_count = 701
    env.rows = 5
    env.cols = 10
    env.action_spec = build_action_space_spec(
        mode=ACTION_SPACE_ADVENTURE_14_IDENTITY,
        plant_types=[1, 1, 0, 0],
        max_seed_slots=14,
    )
    env.curriculum = AdventureSeedCurriculum(
        initial_loadout=ADVENTURE_GENERALIST_INITIAL_LOADOUT,
        unlock_aware=True,
        unlock_introduction_delay=0,
        new_plant_min_inclusion_prob=1.0,
    )
    env.context = {
        "mode": "adventure_generalist_14slot_train",
        "run_mode": "adventure_generalist_14slot_train",
        "selected_seeds": list(env.current_loadout),
        "configured_seed_list": list(env.configured_seed_list),
        "selected_loadout": list(env.current_loadout),
        "selected_loadout_count": len(env.current_loadout),
        "seed_order_source": env.current_seed_order_source,
        "seed_order_preserved": env.current_seed_order_preserved,
        "seed_order_blocked_reason": env.current_seed_order_blocked_reason,
        "randomize_seed_order": env.randomize_seed_order,
        "active_seed_slot_count": len(env.current_loadout),
        "inactive_seed_slot_count": 10,
        "max_seed_slots": 14,
        "observed_capacity": 4,
        "observed_seed_bank_capacity": 4,
        "active_seed_slot_capacity": 4,
        "current_seed_bank_capacity": 4,
        "bridge_reported_capacity": None,
        "inferred_capacity_from_unlocks": 4,
        "effective_seed_capacity": 4,
        "max_effective_seed_capacity_seen": 4,
        "inferred_capacity_source": "initial_starter_loadout",
        "capacity_inference_reason": "initial starter loadout",
        "available_priority_seeds": [],
        "rejected_priority_seeds": [],
        "confirmed_unlock_event_seeds": [],
        "inactive_model_slots": 10,
        "eligible_seeds": env.curriculum.eligible_seeds(),
        "selectable_seeds": list(env.current_selectable_seeds),
        "loadout_reason": env.current_loadout_reason,
        "excluded_new_plants": [],
        "frontier_win_streak": env.frontier_win_streak,
        "frontier_win_streak_required": env.frontier_win_streak_required,
        "frontier_mastery_ready": False,
        "frontier_promoted_this_episode": False,
        "frontier_mastery_reset_reason": "",
        "mastery_sample_source": env.current_sample_source,
        "frontier_replay_supported": True,
        "frontier_replay_blocked_reason": "",
        "frontier_mastered_levels": [],
    }

    class _Writer:
        def write(self, _payload: Dict[str, Any]) -> None:
            return None

    env.writer = _Writer()
    env._append_progress = lambda progress: setattr(env, "_last_progress", progress)
    env._safe_adventure_state = lambda: (
        env.base.adventure_screen_state()
        if getattr(env, "base", None) is not None and hasattr(env.base, "adventure_screen_state")
        else {
            "screenState": "reward_screen",
            "isSeedSelectionScreen": True,
            "availableSeedNames": ["SunFlower", "Peashooter", "WallNut"],
        }
    )
    return env


def assert_case(results: List[Dict[str, Any]], name: str, condition: bool, detail: Any = None) -> None:
    results.append({"case": name, "passed": bool(condition), "detail": detail})


def reason_for(excluded: List[Dict[str, Any]], seed: str) -> str:
    for row in excluded:
        if str(row.get("seed")) == seed:
            return str(row.get("reason", ""))
    return ""


def build_generalist_config_for_test(*, resume_model_path: str = "") -> Dict[str, Any]:
    args = DefaultArgs(
        run_mode=None,
        run_dir=None,
        model=None,
        resume_model_path=Path(resume_model_path) if resume_model_path else None,
        adventure_generalist_train=True,
        adventure_generalist_eval=False,
    )
    return build_config(args, {})


def assert_resume_metadata_compatibility_checks(results: List[Dict[str, Any]]) -> None:
    resume_config = build_generalist_config_for_test(resume_model_path="runs/old_generalist/model.zip")
    with tempfile.TemporaryDirectory(prefix="pvzrl_generalist_resume_") as temp_dir:
        root = Path(temp_dir)
        run_dir = root / "source_run"
        run_dir.mkdir(parents=True, exist_ok=True)
        model_path = run_dir / "model.zip"
        model_path.write_bytes(b"fake")
        write_model_metadata(run_dir, resume_config, model_path=model_path, config_path=run_dir / "resolved_config.json")

        try:
            summary = validate_adventure_generalist_model_compatibility(
                model_path,
                resume_config,
                "test compatible resume metadata",
                model_action_count=701,
            )
            ok = summary.get("compatible") is True
        except SystemExit as exc:
            ok = False
            summary = {"error": str(exc)}
        assert_case(results, "Adventure Generalist compatible resume metadata passes", ok, summary)

        metadata_path = run_dir / "model_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["action_count"] = 201
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        try:
            validate_adventure_generalist_model_compatibility(
                model_path,
                resume_config,
                "test wrong action count",
                model_action_count=201,
            )
            error_text = ""
        except SystemExit as exc:
            error_text = str(exc)
        assert_case(
            results,
            "Adventure Generalist resume rejects wrong action count with incompatible_resume_model",
            "blocked_reason=incompatible_resume_model" in error_text
            and "expected_action_count=701" in error_text
            and "actual_action_count=201" in error_text,
            error_text,
        )

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["action_count"] = 701
        metadata["model_family"] = "ppo_sunflower_peashooter"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        try:
            validate_adventure_generalist_model_compatibility(
                model_path,
                resume_config,
                "test wrong model family",
                model_action_count=701,
            )
            error_text = ""
        except SystemExit as exc:
            error_text = str(exc)
        assert_case(
            results,
            "Adventure Generalist resume rejects wrong model family",
            "blocked_reason=incompatible_resume_model" in error_text
            and "expected_model_family=ppo_adventure_generalist_14slot_identity_v1" in error_text
            and "actual_model_family=ppo_sunflower_peashooter" in error_text,
            error_text,
        )

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["model_family"] = "ppo_adventure_generalist_14slot_identity_v1"
        metadata["observation_version"] = "fixed_slot_v1"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        try:
            validate_adventure_generalist_model_compatibility(
                model_path,
                resume_config,
                "test wrong observation schema",
                model_action_count=701,
            )
            error_text = ""
        except SystemExit as exc:
            error_text = str(exc)
        assert_case(
            results,
            "Adventure Generalist resume rejects wrong observation schema",
            "blocked_reason=incompatible_resume_model" in error_text
            and "expected_observation_schema=adventure_14slot_identity_v1" in error_text
            and "actual_observation_schema=fixed_slot_v1" in error_text,
            error_text,
        )

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["observation_version"] = "adventure_14slot_identity_v1"
        metadata["seed_list"] = ["SunFlower", "Peashooter", "SunFlower", "Peashooter"]
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        try:
            validate_adventure_generalist_model_compatibility(
                model_path,
                resume_config,
                "test changed seed slot order",
                model_action_count=701,
            )
            error_text = ""
        except SystemExit as exc:
            error_text = str(exc)
        assert_case(
            results,
            "Adventure Generalist resume rejects changed seed slot order",
            "blocked_reason=incompatible_resume_model" in error_text
            and "mismatch_fields=seed_list" in error_text
            and "expected_seed_list=SunFlower,SunFlower,Peashooter,Peashooter" in error_text
            and "actual_seed_list=SunFlower,Peashooter,SunFlower,Peashooter" in error_text,
            error_text,
        )


def assert_same_level_replay_recovers_from_menu(results: List[Dict[str, Any]]) -> None:
    original_live_status = adventure_module.build_live_status

    class _Base:
        def __init__(self) -> None:
            self.states = [
                {
                    "screenState": "reward_unlock",
                    "currentAdventureLevel": 1,
                    "isSeedSelectionScreen": False,
                    "isGameplayReady": False,
                },
                {
                    "screenState": "loading_or_menu",
                    "currentAdventureLevel": -1,
                    "isAdventureButtonVisible": True,
                    "isSeedSelectionScreen": False,
                    "isGameplayReady": False,
                },
                {
                    "screenState": "seed_selection",
                    "currentAdventureLevel": 1,
                    "isSeedSelectionScreen": True,
                    "seedSelectionActive": True,
                    "isGameplayReady": False,
                },
                {
                    "screenState": "gameplay",
                    "currentAdventureLevel": 1,
                    "isSeedSelectionScreen": False,
                    "seedSelectionActive": False,
                    "isGameplayReady": True,
                    "gameplayReady": True,
                },
            ]
            self.index = 0
            self.press_adventure_calls = 0
            self.seed_selection_calls = 0

        def adventure_screen_state(self) -> Dict[str, Any]:
            state = self.states[min(self.index, len(self.states) - 1)]
            self.index += 1
            return dict(state)

        def auto_reset(self, **_kwargs: Any) -> Dict[str, Any]:
            return {"ok": True, "methodUsed": "UIMgr.EnterGame", "observation": {"gameplayReady": False}}

        def press_adventure_once(self) -> Dict[str, Any]:
            self.press_adventure_calls += 1
            return {"ok": True, "methodUsed": "press_adventure_once"}

        def auto_select_seeds(self, **_kwargs: Any) -> Dict[str, Any]:
            self.seed_selection_calls += 1
            return {"ok": True, "startInvoked": True}

        def wait_for_gameplay_ready(self, **_kwargs: Any) -> Dict[str, Any]:
            return {"gameplayReady": True, "seedSelectionActive": False}

    class _Writer:
        def write(self, _payload: Dict[str, Any]) -> None:
            return None

    base = _Base()
    env = type(
        "Env",
        (),
        {"base": base, "config": type("Cfg", (), {"poll_seconds": 0.001, "start_sun": None})()},
    )()
    context = {"run_mode": "adventure_generalist_14slot_train", "selected_seeds": ["SunFlower", "Peashooter"]}
    try:
        adventure_module.build_live_status = lambda *_args, **_kwargs: {}
        ok, reason = adventure_module.replay_current_level_after_validation_win(
            env,
            _Writer(),
            context,
            timeout=1.0,
            expected_level=1,
        )
    finally:
        adventure_module.build_live_status = original_live_status
    assert_case(
        results,
        "same-level replay ignores menu level -1 and drives Adventure back to seed selection",
        ok and reason == "" and base.press_adventure_calls == 1 and base.seed_selection_calls == 1,
        {
            "ok": ok,
            "reason": reason,
            "press_adventure_calls": base.press_adventure_calls,
            "seed_selection_calls": base.seed_selection_calls,
            "context": context,
        },
    )


def assert_same_level_replay_skips_stale_adventure_button_after_reset(results: List[Dict[str, Any]]) -> None:
    original_live_status = adventure_module.build_live_status

    class _Base:
        def __init__(self) -> None:
            self.states = [
                {
                    "screenState": "level_complete_trophy",
                    "currentAdventureLevel": 2,
                    "isAdventureButtonVisible": True,
                    "levelCompleteTrophyVisible": True,
                    "isSeedSelectionScreen": False,
                    "isGameplayReady": False,
                },
                {
                    "screenState": "level_complete_trophy",
                    "currentAdventureLevel": 2,
                    "isAdventureButtonVisible": True,
                    "levelCompleteTrophyVisible": True,
                    "isSeedSelectionScreen": False,
                    "isGameplayReady": False,
                },
                {
                    "screenState": "seed_selection",
                    "currentAdventureLevel": 2,
                    "isAdventureButtonVisible": False,
                    "isSeedSelectionScreen": True,
                    "seedSelectionActive": True,
                    "isGameplayReady": False,
                },
                {
                    "screenState": "gameplay",
                    "currentAdventureLevel": 2,
                    "isAdventureButtonVisible": False,
                    "isSeedSelectionScreen": False,
                    "seedSelectionActive": False,
                    "isGameplayReady": True,
                    "gameplayReady": True,
                },
            ]
            self.index = 0
            self.press_adventure_calls = 0
            self.seed_selection_calls = 0

        def adventure_screen_state(self) -> Dict[str, Any]:
            state = self.states[min(self.index, len(self.states) - 1)]
            self.index += 1
            return dict(state)

        def auto_reset(self, **_kwargs: Any) -> Dict[str, Any]:
            return {
                "ok": True,
                "methodUsed": "LoseMenuBtn.TryAgain.OnMouseUp",
                "observation": {
                    "screenState": "level_complete_trophy",
                    "currentAdventureLevel": 2,
                    "gameplayReady": False,
                    "seedSelectionActive": False,
                },
                "message": "Invoked in-game reset hook; Python should wait for board/gameplayReady.",
            }

        def press_adventure_once(self) -> Dict[str, Any]:
            self.press_adventure_calls += 1
            return {"ok": False, "methodUsed": ""}

        def auto_select_seeds(self, **_kwargs: Any) -> Dict[str, Any]:
            self.seed_selection_calls += 1
            return {"ok": True, "startInvoked": True}

        def wait_for_gameplay_ready(self, **_kwargs: Any) -> Dict[str, Any]:
            return {"screenState": "gameplay", "currentAdventureLevel": 2, "gameplayReady": True, "seedSelectionActive": False}

    class _Writer:
        def write(self, _payload: Dict[str, Any]) -> None:
            return None

    base = _Base()
    env = type(
        "Env",
        (),
        {"base": base, "config": type("Cfg", (), {"poll_seconds": 0.001, "start_sun": None})()},
    )()
    context = {"run_mode": "adventure_generalist_14slot_train", "selected_seeds": ["SunFlower", "Peashooter"]}
    try:
        adventure_module.build_live_status = lambda *_args, **_kwargs: {}
        ok, reason = adventure_module.replay_current_level_after_validation_win(
            env,
            _Writer(),
            context,
            timeout=1.0,
            expected_level=2,
        )
    finally:
        adventure_module.build_live_status = original_live_status
    assert_case(
        results,
        "same-level replay accepts TryAgain reset hook and skips stale Adventure click",
        ok and reason == "" and base.press_adventure_calls == 0 and base.seed_selection_calls == 1,
        {
            "ok": ok,
            "reason": reason,
            "press_adventure_calls": base.press_adventure_calls,
            "seed_selection_calls": base.seed_selection_calls,
            "context": context,
        },
    )


def assert_same_level_replay_ignores_stale_post_win_level_mismatch(results: List[Dict[str, Any]]) -> None:
    original_live_status = adventure_module.build_live_status

    class _Base:
        def __init__(self) -> None:
            self.states = [
                {
                    "screenState": "level_complete_trophy",
                    "currentAdventureLevel": 1,
                    "levelCompleteTrophyVisible": True,
                    "isSeedSelectionScreen": False,
                    "isGameplayReady": False,
                },
                {
                    "screenState": "level_complete_trophy",
                    "currentAdventureLevel": 1,
                    "levelCompleteTrophyVisible": True,
                    "isSeedSelectionScreen": False,
                    "isGameplayReady": False,
                },
                {
                    "screenState": "seed_selection",
                    "currentAdventureLevel": 5,
                    "isSeedSelectionScreen": True,
                    "seedSelectionActive": True,
                    "isGameplayReady": False,
                },
                {
                    "screenState": "gameplay",
                    "currentAdventureLevel": 5,
                    "isSeedSelectionScreen": False,
                    "seedSelectionActive": False,
                    "isGameplayReady": True,
                    "gameplayReady": True,
                },
            ]
            self.index = 0
            self.seed_selection_calls = 0

        def adventure_screen_state(self) -> Dict[str, Any]:
            state = self.states[min(self.index, len(self.states) - 1)]
            self.index += 1
            return dict(state)

        def auto_reset(self, **_kwargs: Any) -> Dict[str, Any]:
            return {
                "ok": True,
                "methodUsed": "LoseMenuBtn.TryAgain.OnMouseUp",
                "observation": {
                    "screenState": "level_complete_trophy",
                    "currentAdventureLevel": 1,
                    "gameplayReady": False,
                    "seedSelectionActive": False,
                },
                "message": "Invoked in-game reset hook; Python should wait for board/gameplayReady.",
            }

        def auto_select_seeds(self, **_kwargs: Any) -> Dict[str, Any]:
            self.seed_selection_calls += 1
            return {"ok": True, "startInvoked": True}

        def wait_for_gameplay_ready(self, **_kwargs: Any) -> Dict[str, Any]:
            return {"screenState": "gameplay", "currentAdventureLevel": 5, "gameplayReady": True, "seedSelectionActive": False}

    class _Writer:
        def write(self, _payload: Dict[str, Any]) -> None:
            return None

    base = _Base()
    env = type(
        "Env",
        (),
        {"base": base, "config": type("Cfg", (), {"poll_seconds": 0.001, "start_sun": None})()},
    )()
    context = {"run_mode": "adventure_generalist_14slot_train", "selected_seeds": ["SunFlower", "Peashooter"]}
    try:
        adventure_module.build_live_status = lambda *_args, **_kwargs: {}
        ok, reason = adventure_module.replay_current_level_after_validation_win(
            env,
            _Writer(),
            context,
            timeout=1.0,
            expected_level=5,
        )
    finally:
        adventure_module.build_live_status = original_live_status
    assert_case(
        results,
        "same-level replay ignores stale post-win level mismatch until seed selection",
        ok
        and reason == ""
        and base.seed_selection_calls == 1
        and "1!=5" in str(context.get("frontier_replay_ignored_level_mismatch", "")),
        {
            "ok": ok,
            "reason": reason,
            "seed_selection_calls": base.seed_selection_calls,
            "context": context,
        },
    )


def assert_same_level_replay_retries_transient_seed_selection_failure(results: List[Dict[str, Any]]) -> None:
    original_live_status = adventure_module.build_live_status

    class _Base:
        def __init__(self) -> None:
            self.states = [
                {
                    "screenState": "level_complete_trophy",
                    "currentAdventureLevel": 1,
                    "levelCompleteTrophyVisible": True,
                    "isSeedSelectionScreen": False,
                    "isGameplayReady": False,
                },
                {
                    "screenState": "seed_selection",
                    "currentAdventureLevel": 1,
                    "isSeedSelectionScreen": True,
                    "seedSelectionActive": True,
                    "isGameplayReady": False,
                },
                {
                    "screenState": "seed_selection",
                    "currentAdventureLevel": 1,
                    "isSeedSelectionScreen": True,
                    "seedSelectionActive": True,
                    "isGameplayReady": False,
                },
                {
                    "screenState": "gameplay",
                    "currentAdventureLevel": 1,
                    "isSeedSelectionScreen": False,
                    "seedSelectionActive": False,
                    "isGameplayReady": True,
                    "gameplayReady": True,
                },
            ]
            self.index = 0
            self.seed_selection_calls = 0

        def adventure_screen_state(self) -> Dict[str, Any]:
            state = self.states[min(self.index, len(self.states) - 1)]
            self.index += 1
            return dict(state)

        def auto_reset(self, **_kwargs: Any) -> Dict[str, Any]:
            return {
                "ok": True,
                "methodUsed": "UIMgr.EnterGame",
                "observation": {
                    "screenState": "seed_selection",
                    "currentAdventureLevel": 1,
                    "isSeedSelectionScreen": True,
                    "seedSelectionActive": True,
                },
            }

        def auto_select_seeds(self, **_kwargs: Any) -> Dict[str, Any]:
            self.seed_selection_calls += 1
            if self.seed_selection_calls == 1:
                return {
                    "ok": False,
                    "message": "Seed selection UI is active but not stable/safe for automation.",
                    "actions": ["seed_selection_not_stable_for_clicks"],
                    "after": {
                        "screenState": "seed_selection",
                        "currentAdventureLevel": 1,
                        "isSeedSelectionScreen": True,
                        "seedSelectionActive": True,
                    },
                    "startLog": {"startClicked": False},
                }
            return {"ok": True, "startInvoked": True, "message": "ready", "actions": [], "startLog": {"startClicked": True}}

        def wait_for_gameplay_ready(self, **_kwargs: Any) -> Dict[str, Any]:
            return {"screenState": "gameplay", "currentAdventureLevel": 1, "gameplayReady": True, "seedSelectionActive": False}

    class _Writer:
        def write(self, _payload: Dict[str, Any]) -> None:
            return None

    base = _Base()
    env = type(
        "Env",
        (),
        {
            "base": base,
            "config": type(
                "Cfg",
                (),
                {
                    "poll_seconds": 0.001,
                    "start_sun": None,
                    "seed_click_delay": 0.001,
                },
            )(),
        },
    )()
    context = {
        "run_mode": "adventure_generalist_14slot_train",
        "selected_seeds": ["SunFlower", "SunFlower", "Peashooter", "Peashooter"],
    }
    try:
        adventure_module.build_live_status = lambda *_args, **_kwargs: {}
        ok, reason = adventure_module.replay_current_level_after_validation_win(
            env,
            _Writer(),
            context,
            timeout=1.0,
            expected_level=1,
        )
    finally:
        adventure_module.build_live_status = original_live_status
    assert_case(
        results,
        "same-level replay retries transient seed selection automation failure",
        ok
        and reason == ""
        and base.seed_selection_calls == 2
        and int(context.get("frontier_replay_seed_selection_attempts", 0) or 0) == 2
        and context.get("frontier_replay_gameplay_ready_detected") is True,
        {
            "ok": ok,
            "reason": reason,
            "seed_selection_calls": base.seed_selection_calls,
            "context": context,
        },
    )


def assert_startup_validation_rejects_challenge_state(results: List[Dict[str, Any]]) -> None:
    class _Base:
        def adventure_screen_state(self) -> Dict[str, Any]:
            return {
                "screenState": "seed_selection",
                "currentMode": "Challenge",
                "currentAdventureLevel": 1,
                "isSeedSelectionScreen": True,
                "seedSelectionActive": True,
            }

    with tempfile.TemporaryDirectory(prefix="pvzrl_startup_challenge_") as tmp:
        env = fake_generalist_env(Path(tmp))
        env.strict_startup_validation = True
        env._startup_validation_completed = False
        env.base = _Base()
        result = env.validate_startup_state(phase="test", raise_on_failure=False, timeout=0.05)
    assert_case(
        results,
        "startup validation reports unsupported Challenge-like state",
        result.get("ok") is False
        and result.get("reason") == "unsupported_startup_state_challenge_mode"
        and result.get("blocked_reason") == "adventure_generalist_startup_validation_failed",
        result,
    )


def assert_startup_validation_accepts_expected_seed_selection(results: List[Dict[str, Any]]) -> None:
    class _Base:
        def adventure_screen_state(self) -> Dict[str, Any]:
            return {
                "screenState": "seed_selection",
                "currentMode": "Adventure",
                "currentAdventureLevel": 1,
                "profileAdventureLevel": 1,
                "profileAdventureLevelSource": "test",
                "isSeedSelectionScreen": True,
                "seedSelectionActive": True,
                "currentWorldOrStage": 1,
                "currentDayLevel": 1,
            }

    with tempfile.TemporaryDirectory(prefix="pvzrl_startup_seed_") as tmp:
        env = fake_generalist_env(Path(tmp))
        env.strict_startup_validation = True
        env._startup_validation_completed = False
        env.base = _Base()
        result = env.validate_startup_state(phase="test", raise_on_failure=False, timeout=0.05)
    identity = result.get("level_identity", {})
    assert_case(
        results,
        "startup validation accepts seed selection for expected level",
        result.get("ok") is True
        and result.get("reason") == "clean_seed_selection"
        and identity.get("wrapper_expected_level") == 1
        and identity.get("bridge_detected_level") == 1
        and identity.get("profile_adventure_level") == 1
        and identity.get("level_identity_reliable") is True,
        result,
    )


def assert_same_level_replay_reward_unlock_mismatch_waits_for_stable_gameplay(results: List[Dict[str, Any]]) -> None:
    original_live_status = adventure_module.build_live_status

    class _Base:
        def __init__(self) -> None:
            self.states = [
                {
                    "screenState": "reward_unlock",
                    "currentAdventureLevel": 1,
                    "rewardScreenVisible": True,
                    "isSeedSelectionScreen": False,
                    "isGameplayReady": False,
                },
                {
                    "screenState": "reward_unlock",
                    "currentAdventureLevel": 1,
                    "rewardScreenVisible": True,
                    "isSeedSelectionScreen": False,
                    "isGameplayReady": False,
                },
                {
                    "screenState": "seed_selection",
                    "currentAdventureLevel": 2,
                    "isSeedSelectionScreen": True,
                    "seedSelectionActive": True,
                    "isGameplayReady": False,
                },
                {
                    "screenState": "gameplay",
                    "currentAdventureLevel": 2,
                    "isSeedSelectionScreen": False,
                    "seedSelectionActive": False,
                    "isGameplayReady": True,
                    "gameplayReady": True,
                },
            ]
            self.index = 0
            self.seed_selection_calls = 0

        def adventure_screen_state(self) -> Dict[str, Any]:
            state = self.states[min(self.index, len(self.states) - 1)]
            self.index += 1
            return dict(state)

        def auto_reset(self, **_kwargs: Any) -> Dict[str, Any]:
            return {
                "ok": True,
                "methodUsed": "test_auto_reset",
                "observation": {"screenState": "reward_unlock", "currentAdventureLevel": 1},
            }

        def auto_select_seeds(self, **_kwargs: Any) -> Dict[str, Any]:
            self.seed_selection_calls += 1
            return {"ok": True, "startInvoked": True}

        def wait_for_gameplay_ready(self, **_kwargs: Any) -> Dict[str, Any]:
            return {"screenState": "gameplay", "currentAdventureLevel": 2, "gameplayReady": True, "seedSelectionActive": False}

    class _Writer:
        def write(self, _payload: Dict[str, Any]) -> None:
            return None

    base = _Base()
    env = type(
        "Env",
        (),
        {"base": base, "config": type("Cfg", (), {"poll_seconds": 0.001, "start_sun": None})()},
    )()
    context = {"run_mode": "adventure_generalist_14slot_train", "selected_seeds": ["SunFlower", "Peashooter"]}
    try:
        adventure_module.build_live_status = lambda *_args, **_kwargs: {}
        ok, reason = adventure_module.replay_current_level_after_validation_win(
            env,
            _Writer(),
            context,
            timeout=1.0,
            expected_level=2,
        )
    finally:
        adventure_module.build_live_status = original_live_status
    assert_case(
        results,
        "post-win reward_unlock level mismatch is diagnostic until stable gameplay",
        ok
        and reason == ""
        and context.get("level_identity_reliable") is False
        and "1!=2" in str(context.get("frontier_replay_ignored_level_mismatch", "")),
        {"ok": ok, "reason": reason, "context": context},
    )


def assert_same_level_replay_seed_selection_mismatch_waits_without_autostart(results: List[Dict[str, Any]]) -> None:
    original_live_status = adventure_module.build_live_status

    class _Base:
        def __init__(self) -> None:
            self.index = 0
            self.seed_selection_calls = 0

        def adventure_screen_state(self) -> Dict[str, Any]:
            self.index += 1
            if self.index == 1:
                return {
                    "screenState": "reward_unlock",
                    "currentAdventureLevel": 2,
                    "rewardScreenVisible": True,
                    "isSeedSelectionScreen": False,
                    "isGameplayReady": False,
                }
            return {
                "screenState": "seed_selection",
                "currentAdventureLevel": 1,
                "isSeedSelectionScreen": True,
                "seedSelectionActive": True,
                "isGameplayReady": False,
            }

        def auto_reset(self, **_kwargs: Any) -> Dict[str, Any]:
            return {
                "ok": True,
                "methodUsed": "test_auto_reset",
                "observation": {
                    "screenState": "seed_selection",
                    "currentAdventureLevel": 1,
                    "isSeedSelectionScreen": True,
                    "seedSelectionActive": True,
                },
            }

        def auto_select_seeds(self, **_kwargs: Any) -> Dict[str, Any]:
            self.seed_selection_calls += 1
            return {"ok": True}

    class _Writer:
        def write(self, _payload: Dict[str, Any]) -> None:
            return None

    base = _Base()
    env = type(
        "Env",
        (),
        {
            "base": base,
            "config": type("Cfg", (), {"poll_seconds": 0.001, "start_sun": None, "seed_click_delay": 0.001})(),
        },
    )()
    context = {"run_mode": "adventure_generalist_14slot_train", "selected_seeds": ["SunFlower", "Peashooter"]}
    try:
        adventure_module.build_live_status = lambda *_args, **_kwargs: {}
        ok, reason = adventure_module.replay_current_level_after_validation_win(
            env,
            _Writer(),
            context,
            timeout=0.05,
            expected_level=2,
        )
    finally:
        adventure_module.build_live_status = original_live_status
    assert_case(
        results,
        "post-win seed_selection level mismatch waits without auto-starting wrong level",
        ok is False
        and reason == "win_replay_seed_selection_or_gameplay_timeout"
        and base.seed_selection_calls == 0
        and context.get("frontier_replay_waiting_for_reliable_level_identity") is True,
        {"ok": ok, "reason": reason, "seed_selection_calls": base.seed_selection_calls, "context": context},
    )


def assert_same_level_replay_failure_requests_recovery_without_hard_block(results: List[Dict[str, Any]]) -> None:
    original_replay = generalist_module.replay_current_level_after_validation_win
    with tempfile.TemporaryDirectory(prefix="pvzrl_replay_recovery_") as tmp:
        env = fake_generalist_env(Path(tmp))
        env.frontier_win_streak_required = 2
        env.context["frontier_win_streak_required"] = 2
        env.current_level = 2
        env.current_sample_source = "frontier"
        env.current_attempt = 1
        env._safe_adventure_state = lambda: {
            "screenState": "reward_unlock",
            "currentAdventureLevel": 1,
            "rewardScreenVisible": True,
        }
        try:
            generalist_module.replay_current_level_after_validation_win = (
                lambda *_args, **_kwargs: (False, "same_level_replay_advanced_to_unexpected_level:1!=2")
            )
            env._finish_episode({"done_reason": "win", "episode_summary": {"done_reason": "win"}})
        finally:
            generalist_module.replay_current_level_after_validation_win = original_replay
        hard_blocked = str(getattr(env, "_hard_blocked_reason", "") or "")
        recovery_required = bool(env.context.get("frontier_replay_recovery_required"))
        replay_reason = str(env.context.get("frontier_replay_blocked_reason") or "")
    assert_case(
        results,
        "same-level replay mismatch requests recovery without hard-blocking SB3",
        hard_blocked == ""
        and recovery_required is True
        and "same_level_replay_advanced_to_unexpected_level:1!=2" in replay_reason,
        {"hard_blocked": hard_blocked, "recovery_required": recovery_required, "replay_reason": replay_reason},
    )


def main() -> int:
    results: List[Dict[str, Any]] = []
    assert_startup_validation_rejects_challenge_state(results)
    assert_startup_validation_accepts_expected_seed_selection(results)
    assert_same_level_replay_reward_unlock_mismatch_waits_for_stable_gameplay(results)
    assert_same_level_replay_seed_selection_mismatch_waits_without_autostart(results)
    assert_same_level_replay_failure_requests_recovery_without_hard_block(results)
    assert_same_level_replay_recovers_from_menu(results)
    assert_same_level_replay_skips_stale_adventure_button_after_reset(results)
    assert_same_level_replay_ignores_stale_post_win_level_mismatch(results)
    assert_same_level_replay_retries_transient_seed_selection_failure(results)

    spec = build_action_space_spec(mode=ACTION_SPACE_ADVENTURE_14_IDENTITY, plant_types=[1, 1, 0, 0])
    assert_case(results, "action count is exactly 701", spec.action_count == ADVENTURE_IDENTITY_ACTION_COUNT)
    assert_case(results, "wait action decodes to wait", adventure_identity_action_to_slot_cell(0)["kind"] == 0)
    decoded = adventure_identity_action_to_slot_cell(1 + 13 * 50 + 49)
    assert_case(
        results,
        "placement action decodes slot/row/col",
        decoded == {"kind": 1, "slot_index": 13, "row": 4, "column": 9},
        decoded,
    )

    mask = structural_adventure_identity_mask(4)
    assert_case(results, "wait remains valid", mask[0] is True)
    assert_case(results, "slot 3 placement valid", mask[1 + 3 * 50] is True)
    assert_case(results, "inactive slot 4 placement masked", mask[1 + 4 * 50] is False)
    assert_case(results, "last inactive slot masked", mask[700] is False)

    four_slots = [
        slot(1, "SunFlower", 0),
        slot(1, "SunFlower", 1),
        slot(0, "Peashooter", 2),
        slot(0, "Peashooter", 3),
    ]
    five_slots = [*four_slots, slot(3, "WallNut", 4)]
    fourteen_slots = [*five_slots, slot(2, "CherryBomb", 5), *[slot(0, "Peashooter", i) for i in range(6, 14)]]
    expected_len = adventure_identity_feature_count(14)
    features_4 = adventure_identity_features(observation(four_slots), 14)
    features_5 = adventure_identity_features(observation(five_slots), 14)
    features_14 = adventure_identity_features(observation(fourteen_slots), 14)
    assert_case(results, "observation identity shape stable at 4 slots", len(features_4) == expected_len)
    assert_case(results, "observation identity shape stable at 5 slots", len(features_5) == expected_len)
    assert_case(results, "observation identity shape stable at 14 slots", len(features_14) == expected_len)

    slot0_onehot = features_4[
        ADVENTURE_IDENTITY_SCALAR_FEATURES_PER_SLOT:
        ADVENTURE_IDENTITY_SCALAR_FEATURES_PER_SLOT + ADVENTURE_IDENTITY_ONE_HOT_WIDTH
    ]
    slot2_start = 2 * ADVENTURE_IDENTITY_FEATURES_PER_SLOT
    slot2_onehot = features_4[
        slot2_start + ADVENTURE_IDENTITY_SCALAR_FEATURES_PER_SLOT:
        slot2_start + ADVENTURE_IDENTITY_SCALAR_FEATURES_PER_SLOT + ADVENTURE_IDENTITY_ONE_HOT_WIDTH
    ]
    assert_case(results, "identity encoding changes with plant type", slot0_onehot != slot2_onehot)
    assert_case(
        results,
        "duplicate SunFlower slots share identity one-hot",
        slot0_onehot[1] == 1.0 and features_4[ADVENTURE_IDENTITY_FEATURES_PER_SLOT + ADVENTURE_IDENTITY_SCALAR_FEATURES_PER_SLOT + 1] == 1.0,
    )

    curriculum_block = AdventureSeedCurriculum(initial_loadout=ADVENTURE_GENERALIST_INITIAL_LOADOUT)
    try:
        curriculum_block.choose_loadout(["SunFlower"], observed_capacity=4)
        blocked = False
    except RuntimeError as exc:
        blocked = BLOCKED_INITIAL_LOADOUT_UNAVAILABLE in str(exc)
    assert_case(results, "required starter seed set blocks when unavailable", blocked)

    curriculum = AdventureSeedCurriculum(
        initial_loadout=ADVENTURE_GENERALIST_INITIAL_LOADOUT,
        unlock_aware=True,
        unlock_introduction_delay=0,
        new_plant_min_inclusion_prob=1.0,
    )
    curriculum.episode_index = 0
    curriculum.record_unlocked(["WallNut"], episode_index=0)
    curriculum.episode_index = 1
    decision_missing = curriculum.choose_loadout(["SunFlower", "Peashooter"], observed_capacity=4)
    assert_case(results, "unlock persistence keeps WallNut eligible without selectable snapshot", "WallNut" in decision_missing.eligible_seeds)
    assert_case(
        results,
        "missing selectable snapshot excludes WallNut by reason",
        reason_for(decision_missing.excluded_new_plants, "WallNut") == "not_selectable",
        decision_missing.excluded_new_plants,
    )
    curriculum.episode_index = 2
    decision_future = curriculum.choose_loadout(["SunFlower", "Peashooter", "WallNut"], observed_capacity=4)
    assert_case(
        results,
        "future selectable snapshot waits for open capacity before adding WallNut",
        decision_future.selected_loadout == ["SunFlower", "Peashooter", "WallNut", "SunFlower"]
        and decision_future.guaranteed_seeds == ["WallNut"],
        {"loadout": decision_future.selected_loadout, "reason": decision_future.loadout_reason},
    )

    decision_cap4 = curriculum.choose_loadout(["SunFlower", "Peashooter", "WallNut"], observed_capacity=4)
    decision_cap5 = curriculum.choose_loadout(["SunFlower", "Peashooter", "WallNut"], observed_capacity=5)
    decision_cap2 = curriculum.choose_loadout(["SunFlower", "Peashooter"], observed_capacity=2)
    explicit_curriculum = AdventureSeedCurriculum(
        initial_loadout=ADVENTURE_GENERALIST_INITIAL_LOADOUT,
        seed_curriculum="varied",
        seed_order_source=SEED_ORDER_SOURCE_EXPLICIT,
        randomize_seed_order=False,
    )
    explicit_decision = explicit_curriculum.choose_loadout(["SunFlower", "Peashooter"], observed_capacity=4)
    assert_case(
        results,
        "explicit seed list preserves duplicate order",
        explicit_decision.selected_loadout == ADVENTURE_GENERALIST_INITIAL_LOADOUT
        and explicit_decision.seed_order_source == "explicit_config"
        and explicit_decision.seed_order_preserved is True,
        {
            "loadout": explicit_decision.selected_loadout,
            "source": explicit_decision.seed_order_source,
            "preserved": explicit_decision.seed_order_preserved,
            "reason": explicit_decision.loadout_reason,
        },
    )
    assert_case(
        results,
        "explicit seed list suppresses varied starter mix without randomize flag",
        explicit_decision.loadout_reason != "varied_starter_mix",
        explicit_decision.loadout_reason,
    )
    explicit_unlock_curriculum = AdventureSeedCurriculum(
        initial_loadout=ADVENTURE_GENERALIST_INITIAL_LOADOUT,
        seed_order_source=SEED_ORDER_SOURCE_EXPLICIT,
        randomize_seed_order=False,
        new_plant_min_inclusion_prob=0.0,
    )
    explicit_unlock_curriculum.record_unlocked(["WallNut", "CherryBomb"], episode_index=0)
    explicit_unlock_curriculum.episode_index = 1
    explicit_unlock_decision = explicit_unlock_curriculum.choose_loadout(
        ["SunFlower", "Peashooter", "WallNut", "CherryBomb"],
        observed_capacity=6,
    )
    assert_case(
        results,
        "explicit starter duplicates are preserved while unlocked plants append",
        explicit_unlock_decision.selected_loadout == [
            "SunFlower",
            "Peashooter",
            "WallNut",
            "CherryBomb",
            "SunFlower",
            "SunFlower",
        ]
        and explicit_unlock_decision.loadout_reason == "rotation_guaranteed_unlock"
        and explicit_unlock_decision.seed_order_preserved is False
        and explicit_unlock_decision.blocked_reason == "",
        {
            "loadout": explicit_unlock_decision.selected_loadout,
            "reason": explicit_unlock_decision.loadout_reason,
            "source": explicit_unlock_decision.seed_order_source,
            "preserved": explicit_unlock_decision.seed_order_preserved,
            "blocked": explicit_unlock_decision.blocked_reason,
        },
    )
    explicit_locked_capacity4 = explicit_unlock_curriculum.choose_loadout(
        ["SunFlower", "Peashooter", "WallNut", "CherryBomb"],
        observed_capacity=4,
    )
    assert_case(
        results,
        "explicit starter duplicates are not replaced when capacity is still four",
        explicit_locked_capacity4.selected_loadout == ["SunFlower", "Peashooter", "WallNut", "CherryBomb"]
        and explicit_locked_capacity4.guaranteed_seeds == ["WallNut", "CherryBomb"],
        {
            "loadout": explicit_locked_capacity4.selected_loadout,
            "reason": explicit_locked_capacity4.loadout_reason,
            "excluded": explicit_locked_capacity4.excluded_new_plants,
        },
    )
    varied_without_randomize = AdventureSeedCurriculum(
        initial_loadout=ADVENTURE_GENERALIST_INITIAL_LOADOUT,
        seed_curriculum="varied",
        randomize_seed_order=False,
    ).choose_loadout(["SunFlower", "Peashooter"], observed_capacity=4)
    assert_case(
        results,
        "varied starter mix is disabled without explicit randomize flag",
        varied_without_randomize.selected_loadout == ADVENTURE_GENERALIST_INITIAL_LOADOUT
        and varied_without_randomize.loadout_reason != "varied_starter_mix",
        {
            "loadout": varied_without_randomize.selected_loadout,
            "reason": varied_without_randomize.loadout_reason,
        },
    )
    varied_with_randomize = AdventureSeedCurriculum(
        initial_loadout=ADVENTURE_GENERALIST_INITIAL_LOADOUT,
        seed_curriculum="varied",
        randomize_seed_order=True,
    ).choose_loadout(["SunFlower", "Peashooter"], observed_capacity=4)
    assert_case(
        results,
        "varied starter mix only activates with explicit randomize flag",
        varied_with_randomize.selected_loadout == ADVENTURE_GENERALIST_INITIAL_LOADOUT
        and varied_with_randomize.loadout_reason == "explicit_config"
        and varied_with_randomize.seed_order_source == "default_canonical",
        {
            "loadout": varied_with_randomize.selected_loadout,
            "reason": varied_with_randomize.loadout_reason,
            "source": varied_with_randomize.seed_order_source,
        },
    )
    replay_seed_selection = {
        "screenState": "seed_selection",
        "isSeedSelectionScreen": True,
        "currentAdventureLevel": 4,
        "availableSeedNames": ["SunFlower", "Peashooter", "WallNut", "CherryBomb"],
        "seedSlotCapacity": 6,
    }
    replay_gameplay = {"screenState": "gameplay", "gameplayReady": True, "currentAdventureLevel": 4}

    class _ReplayBase:
        def __init__(self) -> None:
            self.reset_called = False
            self.selection_done = False
            self.selected_seed_list: List[str] = []

        def adventure_screen_state(self) -> Dict[str, Any]:
            if self.selection_done:
                return dict(replay_gameplay)
            if self.reset_called:
                return dict(replay_seed_selection)
            return {"screenState": "reward_screen", "rewardObjectVisible": True, "currentAdventureLevel": 4}

        def auto_reset(self, **_kwargs: Any) -> Dict[str, Any]:
            self.reset_called = True
            return {"ok": True, "methodUsed": "test_auto_reset", "observation": dict(replay_seed_selection)}

        def auto_select_seeds(self, seed_list: List[str], start_level: bool = True) -> Dict[str, Any]:
            self.selected_seed_list = list(seed_list)
            self.selection_done = True
            return {"ok": True, "message": "ok", "actions": [], "startLevel": bool(start_level)}

        def wait_for_gameplay_ready(self, **_kwargs: Any) -> Dict[str, Any]:
            return dict(replay_gameplay)

    replay_env = type("ReplayEnv", (), {})()
    replay_env.base = _ReplayBase()
    replay_env.config = type("ReplayConfig", (), {"poll_seconds": 0.01, "start_sun": None})()
    replay_context = {
        "mode": "adventure_generalist_14slot_train",
        "selected_seeds": list(ADVENTURE_GENERALIST_INITIAL_LOADOUT),
    }
    replay_callback_inputs: List[List[str]] = []

    def _replay_seed_callback(_state: Dict[str, Any], current_seed_list: List[str]) -> tuple[List[str], str]:
        replay_callback_inputs.append(list(current_seed_list))
        return ADVENTURE_GENERALIST_INITIAL_LOADOUT + ["WallNut", "CherryBomb"], ""

    original_replay_live_status = adventure_module.build_live_status
    try:
        adventure_module.build_live_status = lambda *_args, **_kwargs: {}
        replay_ok, replay_reason = adventure_module.replay_current_level_after_validation_win(
            replay_env,
            type("ReplayWriter", (), {"write": lambda self, _payload: None})(),
            replay_context,
            timeout=1.0,
            expected_level=4,
            seed_selection_callback=_replay_seed_callback,
        )
    finally:
        adventure_module.build_live_status = original_replay_live_status
    assert_case(
        results,
        "same-level replay refreshes seed list through callback before auto-select",
        replay_ok
        and replay_reason == ""
        and replay_env.base.selected_seed_list == ADVENTURE_GENERALIST_INITIAL_LOADOUT + ["WallNut", "CherryBomb"]
        and replay_callback_inputs == [ADVENTURE_GENERALIST_INITIAL_LOADOUT],
        {
            "ok": replay_ok,
            "reason": replay_reason,
            "selected": replay_env.base.selected_seed_list,
            "callback_inputs": replay_callback_inputs,
        },
    )
    with tempfile.TemporaryDirectory(prefix="pvzrl_seed_order_runtime_") as tmp:
        runtime_env = fake_generalist_env(Path(tmp))
        runtime_env.curriculum = AdventureSeedCurriculum(
            initial_loadout=ADVENTURE_GENERALIST_INITIAL_LOADOUT,
            seed_curriculum="varied",
            seed_order_source=SEED_ORDER_SOURCE_EXPLICIT,
            randomize_seed_order=False,
        )
        runtime_env.seed_order_source = SEED_ORDER_SOURCE_EXPLICIT
        runtime_env._apply_loadout = lambda _loadout: None
        selected_runtime, blocked_runtime = runtime_env._on_seed_selection_screen(
            {
                "visibleSeedCardNames": ["SunFlower", "Peashooter"],
                "availableSeedNames": ["SunFlower", "Peashooter"],
                "seedSlotCapacity": 4,
            },
            list(ADVENTURE_GENERALIST_INITIAL_LOADOUT),
        )
    assert_case(
        results,
        "runtime selected_loadout equals configured seed_list",
        selected_runtime == ADVENTURE_GENERALIST_INITIAL_LOADOUT
        and blocked_runtime == ""
        and runtime_env.context.get("configured_seed_list") == ADVENTURE_GENERALIST_INITIAL_LOADOUT
        and runtime_env.context.get("selected_loadout") == ADVENTURE_GENERALIST_INITIAL_LOADOUT
        and runtime_env.context.get("pending_seed_selection") is True
        and runtime_env.context.get("proposed_selected_loadout") == ADVENTURE_GENERALIST_INITIAL_LOADOUT
        and runtime_env.context.get("proposed_seed_validation_source") == "selectable",
        {
            "selected": selected_runtime,
            "blocked": blocked_runtime,
            "context": runtime_env.context,
        },
    )
    with tempfile.TemporaryDirectory(prefix="pvzrl_seed_order_runtime_unlock_") as tmp:
        runtime_unlock_env = fake_generalist_env(Path(tmp))
        runtime_unlock_env.curriculum = AdventureSeedCurriculum(
            initial_loadout=ADVENTURE_GENERALIST_INITIAL_LOADOUT,
            seed_order_source=SEED_ORDER_SOURCE_EXPLICIT,
            randomize_seed_order=False,
        )
        runtime_unlock_env.seed_order_source = SEED_ORDER_SOURCE_EXPLICIT
        runtime_unlock_env.curriculum.record_unlocked(["WallNut", "CherryBomb"], episode_index=0)
        runtime_unlock_env.curriculum.episode_index = 1
        runtime_unlock_env.episode_index = 1
        runtime_unlock_env._apply_loadout = lambda _loadout: None
        selected_unlock_runtime, blocked_unlock_runtime = runtime_unlock_env._on_seed_selection_screen(
            {
                "visibleSeedCardNames": ["SunFlower", "Peashooter", "WallNut", "CherryBomb"],
                "availableSeedNames": ["SunFlower", "Peashooter", "WallNut", "CherryBomb"],
                "seedSlotCapacity": 6,
            },
            list(ADVENTURE_GENERALIST_INITIAL_LOADOUT),
        )
    assert_case(
        results,
        "runtime appends unlocked plants after duplicated starter loadout when capacity opens",
        selected_unlock_runtime == ["SunFlower", "Peashooter", "WallNut", "CherryBomb", "SunFlower", "SunFlower"]
        and blocked_unlock_runtime == ""
        and runtime_unlock_env.context.get("selected_loadout") == ADVENTURE_GENERALIST_INITIAL_LOADOUT
        and runtime_unlock_env.context.get("proposed_selected_loadout") == selected_unlock_runtime
        and runtime_unlock_env.context.get("proposed_loadout_reason") == "rotation_guaranteed_unlock",
        {
            "selected": selected_unlock_runtime,
            "blocked": blocked_unlock_runtime,
            "context": runtime_unlock_env.context,
        },
    )
    with tempfile.TemporaryDirectory(prefix="pvzrl_seed_order_runtime_inferred_capacity_") as tmp:
        inferred_runtime_env = fake_generalist_env(Path(tmp))
        inferred_runtime_env.curriculum = AdventureSeedCurriculum(
            initial_loadout=ADVENTURE_GENERALIST_INITIAL_LOADOUT,
            seed_order_source=SEED_ORDER_SOURCE_EXPLICIT,
            randomize_seed_order=False,
        )
        inferred_runtime_env.seed_order_source = SEED_ORDER_SOURCE_EXPLICIT
        inferred_runtime_env.curriculum.record_unlocked(["WallNut", "CherryBomb"], episode_index=0)
        inferred_runtime_env.curriculum.episode_index = 1
        inferred_runtime_env.episode_index = 1
        inferred_runtime_env._apply_loadout = lambda _loadout: None
        selected_inferred_runtime, blocked_inferred_runtime = inferred_runtime_env._on_seed_selection_screen(
            {
                "visibleSeedCardNames": ["SunFlower", "Peashooter", "WallNut", "CherryBomb"],
                "availableSeedNames": ["SunFlower", "Peashooter", "WallNut", "CherryBomb"],
                "selectedSeedNames": ["SunFlower", "SunFlower", "Peashooter", "Peashooter"],
                "seedSlotCapacity": 4,
            },
            list(ADVENTURE_GENERALIST_INITIAL_LOADOUT),
        )
    assert_case(
        results,
        "runtime infers six-slot loadout when bridge capacity is stuck at four",
        selected_inferred_runtime == ["SunFlower", "Peashooter", "WallNut", "CherryBomb", "SunFlower", "SunFlower"]
        and blocked_inferred_runtime == ""
        and inferred_runtime_env.context.get("observed_seed_bank_capacity") == 4
        and inferred_runtime_env.context.get("effective_seed_capacity") == 6
        and inferred_runtime_env.context.get("max_effective_seed_capacity_seen") == 6,
        {
            "selected": selected_inferred_runtime,
            "blocked": blocked_inferred_runtime,
            "context": inferred_runtime_env.context,
        },
    )
    with tempfile.TemporaryDirectory(prefix="pvzrl_seed_order_runtime_cherry_unlock_") as tmp:
        cherry_runtime_env = fake_generalist_env(Path(tmp))
        cherry_runtime_env.curriculum = AdventureSeedCurriculum(
            initial_loadout=ADVENTURE_GENERALIST_INITIAL_LOADOUT,
            seed_order_source=SEED_ORDER_SOURCE_EXPLICIT,
            randomize_seed_order=False,
        )
        cherry_runtime_env.seed_order_source = SEED_ORDER_SOURCE_EXPLICIT
        cherry_runtime_env.curriculum.record_unlocked(["CherryBomb"], episode_index=1)
        cherry_runtime_env.curriculum.episode_index = 2
        cherry_runtime_env.episode_index = 2
        cherry_runtime_env.confirmed_unlock_event_seeds = ["CherryBomb"]
        cherry_runtime_env._apply_loadout = lambda _loadout: None
        selected_cherry_runtime, blocked_cherry_runtime = cherry_runtime_env._on_seed_selection_screen(
            {
                "visibleSeedCardNames": ["SunFlower", "Peashooter"],
                "availableSeedNames": ["SunFlower", "Peashooter"],
                "selectedSeedNames": ["SunFlower", "SunFlower", "Peashooter", "Peashooter"],
                "seedSlotCapacity": 4,
            },
            list(ADVENTURE_GENERALIST_INITIAL_LOADOUT),
        )
    assert_case(
        results,
        "runtime selects CherryBomb from confirmed unlock event when bridge stays at four visible slots",
        selected_cherry_runtime == ["SunFlower", "Peashooter", "CherryBomb", "SunFlower", "SunFlower"]
        and blocked_cherry_runtime == ""
        and cherry_runtime_env.context.get("effective_seed_capacity") == 5
        and cherry_runtime_env.context.get("selected_loadout") == ADVENTURE_GENERALIST_INITIAL_LOADOUT
        and cherry_runtime_env.context.get("proposed_selected_loadout") == selected_cherry_runtime
        and cherry_runtime_env.context.get("proposed_rejected_priority_seeds", []) == [],
        {
            "selected": selected_cherry_runtime,
            "blocked": blocked_cherry_runtime,
            "context": cherry_runtime_env.context,
        },
    )
    with tempfile.TemporaryDirectory(prefix="pvzrl_seed_order_runtime_empty_selectable_") as tmp:
        empty_selectable_env = fake_generalist_env(Path(tmp))
        empty_selectable_env.curriculum = AdventureSeedCurriculum(
            initial_loadout=ADVENTURE_GENERALIST_INITIAL_LOADOUT,
            seed_order_source=SEED_ORDER_SOURCE_EXPLICIT,
            randomize_seed_order=False,
        )
        empty_selectable_env.seed_order_source = SEED_ORDER_SOURCE_EXPLICIT
        empty_selectable_env.curriculum.record_unlocked(["CherryBomb"], episode_index=1)
        empty_selectable_env.curriculum.episode_index = 2
        empty_selectable_env.episode_index = 2
        empty_selectable_env.confirmed_unlock_event_seeds = ["CherryBomb"]
        empty_selectable_env._apply_loadout = lambda _loadout: None
        empty_selectable_log = io.StringIO()
        with redirect_stdout(empty_selectable_log):
            selected_empty_runtime, blocked_empty_runtime = empty_selectable_env._on_seed_selection_screen(
                {
                    "screenState": "seed_selection",
                    "isSeedSelectionScreen": True,
                    "visibleSeedCardNames": [],
                    "availableSeedNames": [],
                    "selectedSeedNames": [],
                    "unlockedSeedNames": ["SunFlower", "Peashooter", "CherryBomb"],
                    "seedSlotCapacity": 4,
                },
                list(ADVENTURE_GENERALIST_INITIAL_LOADOUT),
            )
        empty_selectable_output = empty_selectable_log.getvalue()
    assert_case(
        results,
        "empty selectable defers loadout expansion until the current UI exposes cards",
        selected_empty_runtime == ADVENTURE_GENERALIST_INITIAL_LOADOUT
        and blocked_empty_runtime == ""
        and empty_selectable_env.context.get("raw_selectable_seeds") == []
        and empty_selectable_env.context.get("seed_validation_source")
        == "selectable_empty_using_unlocked_or_eligible_fallback"
        and empty_selectable_env.context.get("selected_loadout")
        == ADVENTURE_GENERALIST_INITIAL_LOADOUT
        and empty_selectable_env.context.get("seed_order_blocked_reason") == ""
        and "selectable_empty_using_unlocked_or_eligible_fallback" in empty_selectable_output
        and "selectable=[]" in empty_selectable_output
        and "eligible=['SunFlower', 'Peashooter', 'CherryBomb']" in empty_selectable_output
        and "unlocked=['SunFlower', 'Peashooter', 'CherryBomb']" in empty_selectable_output
        and "blocked_reason=None" in empty_selectable_output,
        {
            "selected": selected_empty_runtime,
            "blocked": blocked_empty_runtime,
            "context": empty_selectable_env.context,
            "log": empty_selectable_output,
        },
    )
    assert_case(
        results,
        "capacity=2 starter loadout preserves SunFlower and Peashooter",
        decision_cap2.selected_loadout == ["SunFlower", "Peashooter"],
        decision_cap2.selected_loadout,
    )
    assert_case(results, "capacity=4 keeps loadout length <=4", len(decision_cap4.selected_loadout) <= 4, decision_cap4.selected_loadout)
    assert_case(
        results,
        "capacity=5 can fill fifth slot",
        len(decision_cap5.selected_loadout) == 5 and "WallNut" in decision_cap5.selected_loadout,
        decision_cap5.selected_loadout,
    )
    inferred_capacity_unique_names = generalist_module._infer_seed_bank_capacity_from_state(
        {
            "visibleSeedCardNames": ["SunFlower", "Peashooter"],
            "availableSeedNames": ["SunFlower", "Peashooter"],
            "selectedSeedNames": [],
            "unlockedSeedNames": ["SunFlower", "Peashooter"],
        },
        context={},
        previous_capacity=4,
        selected_loadout=list(ADVENTURE_GENERALIST_INITIAL_LOADOUT),
    )
    assert_case(
        results,
        "seed-screen unique names do not shrink duplicate-capable starter capacity",
        int(inferred_capacity_unique_names) == 4,
        {"inferred_capacity": inferred_capacity_unique_names},
    )
    inferred_capacity_numeric = generalist_module._infer_seed_bank_capacity_from_state(
        {"seedSlotCapacity": 6},
        context={},
        previous_capacity=4,
        selected_loadout=list(ADVENTURE_GENERALIST_INITIAL_LOADOUT),
    )
    assert_case(
        results,
        "explicit seed-slot capacity can raise inferred capacity",
        int(inferred_capacity_numeric) == 6,
        {"inferred_capacity": inferred_capacity_numeric},
    )
    starter_capacity = generalist_module.resolve_adventure_generalist_seed_capacity(
        {
            "visibleSeedCardNames": ["SunFlower", "Peashooter"],
            "availableSeedNames": ["SunFlower", "Peashooter"],
            "selectedSeedNames": ["SunFlower", "SunFlower", "Peashooter", "Peashooter"],
            "seedSlotCapacity": 4,
        },
        context={},
        previous_observed_capacity=4,
        previous_effective_capacity=4,
        selected_loadout=list(ADVENTURE_GENERALIST_INITIAL_LOADOUT),
        eligible_seeds=["SunFlower", "Peashooter"],
        unlocked_seeds=["SunFlower", "Peashooter"],
    )
    assert_case(
        results,
        "capacity inference starter only remains four",
        starter_capacity.inferred_capacity_from_unlocks == 4
        and starter_capacity.effective_seed_capacity == 4,
        starter_capacity.__dict__,
    )
    cherry_alone_capacity = generalist_module.resolve_adventure_generalist_seed_capacity(
        {
            "visibleSeedCardNames": ["SunFlower", "Peashooter", "CherryBomb"],
            "availableSeedNames": ["SunFlower", "Peashooter", "CherryBomb"],
            "selectedSeedNames": ["SunFlower", "SunFlower", "Peashooter", "Peashooter"],
            "seedSlotCapacity": 4,
        },
        context={},
        previous_observed_capacity=4,
        previous_effective_capacity=4,
        selected_loadout=list(ADVENTURE_GENERALIST_INITIAL_LOADOUT),
        eligible_seeds=["SunFlower", "Peashooter", "CherryBomb"],
        unlocked_seeds=["SunFlower", "Peashooter", "CherryBomb"],
    )
    assert_case(
        results,
        "capacity inference expands to five when CherryBomb alone is selectable",
        cherry_alone_capacity.observed_capacity == 4
        and cherry_alone_capacity.inferred_capacity_from_unlocks >= 5
        and cherry_alone_capacity.effective_seed_capacity == 5
        and cherry_alone_capacity.available_priority_seeds == ["CherryBomb"],
        cherry_alone_capacity.__dict__,
    )
    cherry_medium_capacity = generalist_module.resolve_adventure_generalist_seed_capacity(
        {
            "visibleSeedCardNames": ["SunFlower", "Peashooter"],
            "availableSeedNames": ["SunFlower", "Peashooter"],
            "selectedSeedNames": ["SunFlower", "SunFlower", "Peashooter", "Peashooter"],
            "seedSlotCapacity": 4,
        },
        context={},
        previous_observed_capacity=4,
        previous_effective_capacity=4,
        selected_loadout=list(ADVENTURE_GENERALIST_INITIAL_LOADOUT),
        eligible_seeds=["SunFlower", "Peashooter", "CherryBomb"],
        unlocked_seeds=["SunFlower", "Peashooter", "CherryBomb"],
        medium_confirmed_seeds=["CherryBomb"],
    )
    assert_case(
        results,
        "capacity inference uses confirmed CherryBomb unlock event even when visible list is starter only",
        cherry_medium_capacity.inferred_capacity_from_unlocks == 5
        and cherry_medium_capacity.effective_seed_capacity == 5
        and cherry_medium_capacity.inferred_capacity_source == "unlock_event_priority_seed",
        cherry_medium_capacity.__dict__,
    )
    wallnut_cherry_capacity = generalist_module.resolve_adventure_generalist_seed_capacity(
        {
            "visibleSeedCardNames": ["SunFlower", "Peashooter", "WallNut", "CherryBomb"],
            "availableSeedNames": ["SunFlower", "Peashooter", "WallNut", "CherryBomb"],
            "selectedSeedNames": ["SunFlower", "SunFlower", "Peashooter", "Peashooter"],
            "seedSlotCapacity": 4,
        },
        context={},
        previous_observed_capacity=4,
        previous_effective_capacity=4,
        selected_loadout=list(ADVENTURE_GENERALIST_INITIAL_LOADOUT),
        eligible_seeds=["SunFlower", "Peashooter", "WallNut", "CherryBomb"],
        unlocked_seeds=["SunFlower", "Peashooter", "WallNut", "CherryBomb"],
    )
    assert_case(
        results,
        "capacity inference expands to six when WallNut and CherryBomb are selectable",
        wallnut_cherry_capacity.observed_capacity == 4
        and wallnut_cherry_capacity.inferred_capacity_from_unlocks >= 6
        and wallnut_cherry_capacity.effective_seed_capacity == 6
        and wallnut_cherry_capacity.inferred_capacity_source == "selectable_priority_seeds",
        wallnut_cherry_capacity.__dict__,
    )
    assert_case(
        results,
        "CherryBomb naming variants canonicalize for selection",
        [
            generalist_module.canonicalize_seed_name("CherryBomb"),
            generalist_module.canonicalize_seed_name("Cherry Bomb"),
            generalist_module.canonicalize_seed_name("cherry_bomb"),
        ]
        == ["CherryBomb", "CherryBomb", "CherryBomb"],
        {
            "CherryBomb": generalist_module.canonicalize_seed_name("CherryBomb"),
            "Cherry Bomb": generalist_module.canonicalize_seed_name("Cherry Bomb"),
            "cherry_bomb": generalist_module.canonicalize_seed_name("cherry_bomb"),
        },
    )
    bridge_capacity = generalist_module.resolve_adventure_generalist_seed_capacity(
        {"seedSlotCapacity": 6},
        context={},
        previous_observed_capacity=4,
        previous_effective_capacity=4,
        selected_loadout=list(ADVENTURE_GENERALIST_INITIAL_LOADOUT),
        eligible_seeds=["SunFlower", "Peashooter"],
        unlocked_seeds=["SunFlower", "Peashooter"],
    )
    assert_case(
        results,
        "bridge reported six preserves effective capacity",
        bridge_capacity.bridge_reported_capacity == 6
        and bridge_capacity.observed_capacity == 6
        and bridge_capacity.effective_seed_capacity >= 6,
        bridge_capacity.__dict__,
    )
    overreported_unlocked = [
        "SunFlower",
        "Peashooter",
        "WallNut",
        "CherryBomb",
        "PotatoMine",
        "Chomper",
        "Chomper",
        "SmallPuff",
        "FumeShroom",
        "HypnoShroom",
        "FumeShroom",
        "Gravebuster",
    ]
    false_unlocked_capacity = generalist_module.resolve_adventure_generalist_seed_capacity(
        {
            "visibleSeedCardNames": ["SunFlower", "Peashooter"],
            "availableSeedNames": ["SunFlower", "Peashooter"],
            "selectedSeedNames": ["SunFlower", "SunFlower", "Peashooter", "Peashooter"],
            "unlockedSeedNames": overreported_unlocked,
            "seedSlotCapacity": 4,
        },
        context={},
        previous_observed_capacity=4,
        previous_effective_capacity=4,
        selected_loadout=list(ADVENTURE_GENERALIST_INITIAL_LOADOUT),
        eligible_seeds=["SunFlower", "Peashooter"],
        unlocked_seeds=overreported_unlocked,
    )
    assert_case(
        results,
        "all unlocked overreport with selectable starter only does not jump to fourteen",
        false_unlocked_capacity.effective_seed_capacity == 4
        and false_unlocked_capacity.inferred_capacity_from_unlocks == 4,
        false_unlocked_capacity.__dict__,
    )
    selected_only_false_unlocked = generalist_module.resolve_adventure_generalist_seed_capacity(
        {
            "selectedSeedNames": ["SunFlower", "SunFlower", "Peashooter", "Peashooter"],
            "unlockedSeedNames": overreported_unlocked,
            "seedSlotCapacity": 4,
        },
        context={},
        previous_observed_capacity=4,
        previous_effective_capacity=4,
        selected_loadout=list(ADVENTURE_GENERALIST_INITIAL_LOADOUT),
        eligible_seeds=["SunFlower", "Peashooter"],
        unlocked_seeds=overreported_unlocked,
    )
    assert_case(
        results,
        "all unlocked overreport with selected starter only remains four",
        selected_only_false_unlocked.effective_seed_capacity == 4
        and selected_only_false_unlocked.inferred_capacity_from_unlocks == 4,
        selected_only_false_unlocked.__dict__,
    )
    eligible_overreport_false_unlocked = generalist_module.resolve_adventure_generalist_seed_capacity(
        {
            "visibleSeedCardNames": ["SunFlower", "Peashooter"],
            "availableSeedNames": ["SunFlower", "Peashooter"],
            "selectedSeedNames": ["SunFlower", "SunFlower", "Peashooter", "Peashooter"],
            "unlockedSeedNames": overreported_unlocked,
            "seedSlotCapacity": 4,
        },
        context={},
        previous_observed_capacity=4,
        previous_effective_capacity=4,
        selected_loadout=list(ADVENTURE_GENERALIST_INITIAL_LOADOUT),
        eligible_seeds=overreported_unlocked,
        unlocked_seeds=overreported_unlocked,
    )
    assert_case(
        results,
        "all unlocked overreport with eligible overreport still does not expand without confirmed event",
        eligible_overreport_false_unlocked.effective_seed_capacity == 4
        and eligible_overreport_false_unlocked.inferred_capacity_from_unlocks == 4,
        eligible_overreport_false_unlocked.__dict__,
    )
    filtered_priority = generalist_module.normalize_and_filter_priority_seeds(
        ["CherryBomb", "WallNut", "PotatoMine"],
        ["SunFlower", "Peashooter", "Wall-Nut", "Cherry Bomb"],
    )
    assert_case(
        results,
        "capacity priority seeds are filtered through known registry or bridge names",
        filtered_priority == ["CherryBomb", "WallNut"],
        filtered_priority,
    )
    filtered_runtime_candidates = generalist_module._filter_loadout_candidate_seeds(
        ["SunFlower", "Peashooter", "CherryBomb", "IcePea", "GraveBuster"],
        eligible_seeds=["SunFlower", "Peashooter", "CherryBomb", "IcePea", "GraveBuster"],
        unlocked_seeds=["SunFlower", "Peashooter", "CherryBomb", "IcePea", "GraveBuster"],
    )
    assert_case(
        results,
        "runtime loadout candidates retain newly registered plants and ignore unknown names",
        filtered_runtime_candidates == ["SunFlower", "Peashooter", "CherryBomb"],
        filtered_runtime_candidates,
    )
    monotonic_later_capacity = generalist_module.resolve_adventure_generalist_seed_capacity(
        {
            "visibleSeedCardNames": ["SunFlower", "Peashooter"],
            "availableSeedNames": ["SunFlower", "Peashooter"],
            "seedSlotCapacity": 4,
        },
        context={},
        previous_observed_capacity=4,
        previous_effective_capacity=wallnut_cherry_capacity.effective_seed_capacity,
        selected_loadout=list(ADVENTURE_GENERALIST_INITIAL_LOADOUT),
        eligible_seeds=["SunFlower", "Peashooter"],
        unlocked_seeds=overreported_unlocked,
    )
    assert_case(
        results,
        "capacity inference session maximum prevents drop back to four",
        monotonic_later_capacity.inferred_capacity_from_unlocks == 4
        and monotonic_later_capacity.effective_seed_capacity == 6
        and monotonic_later_capacity.max_effective_seed_capacity_seen == 6,
        monotonic_later_capacity.__dict__,
    )

    delay_curriculum = AdventureSeedCurriculum(
        initial_loadout=ADVENTURE_GENERALIST_INITIAL_LOADOUT,
        unlock_aware=True,
        unlock_introduction_delay=2,
        new_plant_min_inclusion_prob=1.0,
    )
    delay_curriculum.record_unlocked(["WallNut"], episode_index=0)
    delay_curriculum.episode_index = 1
    delay_decision = delay_curriculum.choose_loadout(["SunFlower", "Peashooter", "WallNut"], observed_capacity=5)
    assert_case(
        results,
        "conservative delay gate excludes new seed until delay passes",
        reason_for(delay_decision.excluded_new_plants, "WallNut") == "" and "WallNut" in delay_decision.selected_loadout,
        delay_decision.excluded_new_plants,
    )

    conservative_curriculum = AdventureSeedCurriculum(
        initial_loadout=ADVENTURE_GENERALIST_INITIAL_LOADOUT,
        unlock_aware=True,
        unlock_introduction_delay=0,
        new_plant_min_inclusion_prob=1.0,
    )
    conservative_curriculum.record_unlocked(["WallNut"], episode_index=0)
    conservative_curriculum.episode_index = 1
    conservative_decision = conservative_curriculum.choose_loadout(["SunFlower", "Peashooter", "WallNut"], observed_capacity=5)
    assert_case(
        results,
        "conservative extra capacity fills appended slot before replacement",
        conservative_decision.selected_loadout == ["SunFlower", "Peashooter", "WallNut", "SunFlower", "SunFlower"],
        conservative_decision.selected_loadout,
    )

    with tempfile.TemporaryDirectory() as tmp:
        env = fake_generalist_env(Path(tmp))
        env.observed_seed_bank_capacity = 5
        env.current_loadout = list(ADVENTURE_GENERALIST_INITIAL_LOADOUT)
        env._write_unlock_files()
        slot_payload = json.loads((env.seed_slot_unlocks_path).read_text(encoding="utf-8"))
        assert_case(
            results,
            "seed_slot_unlocks writes observed capacity separately from selected count",
            int(slot_payload.get("observed_seed_bank_capacity", -1)) == 5
            and int(slot_payload.get("selected_loadout_count", -1)) == 4
            and int(slot_payload.get("active_seed_slot_count", -1)) == 4,
            slot_payload,
        )

    wallnut_obs = observation([slot(1, "SunFlower", 0), *[slot(0, "Peashooter", i) for i in range(1, 5)], slot(3, "WallNut", 5)])
    decoded_wallnut = decode_action(1 + 5 * 50, wallnut_obs, [1, 0, 0, 0])
    cherry_obs = observation([slot(1, "SunFlower", 0), *[slot(0, "Peashooter", i) for i in range(1, 7)], slot(2, "CherryBomb", 7)])
    decoded_cherry = decode_action(1 + 7 * 50, cherry_obs, [1, 0, 0, 0])
    assert_case(results, "WallNut tactical identity follows slot content", decoded_wallnut.get("plant_type") == 3, decoded_wallnut)
    assert_case(results, "CherryBomb tactical identity follows slot content", decoded_cherry.get("plant_type") == 2, decoded_cherry)

    original_collect = generalist_module.collect_post_win_unlocks
    original_replay = generalist_module.replay_current_level_after_validation_win
    original_live_status = generalist_module.build_live_status
    try:
        generalist_module.build_live_status = lambda *_args, **_kwargs: {}

        env_fail = fake_generalist_env(Path(tempfile.mkdtemp()))

        def _raise_collect(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("simulated_unlock_failure")

        generalist_module.collect_post_win_unlocks = _raise_collect
        env_fail._finish_episode({"episode_summary": {"done_reason": "win", "episode_reward": 1.0, "episode_length": 10}})
        assert_case(
            results,
            "post-win failure keeps explicit blocked reason",
            str(env_fail.context.get("post_win_blocked_reason", "")).startswith("post_win_unlock_handling_failed:"),
            env_fail.context.get("post_win_blocked_reason"),
        )
        assert_case(
            results,
            "post-win failure keeps state snapshot",
            isinstance(env_fail.context.get("post_win_last_state"), dict) and bool(env_fail.context.get("post_win_last_state")),
            env_fail.context.get("post_win_last_state"),
        )

        env_ok = fake_generalist_env(Path(tempfile.mkdtemp()))

        def _ok_collect(*_args: Any, **_kwargs: Any) -> Any:
            return (
                {
                    "screenState": "seed_selection",
                    "isSeedSelectionScreen": True,
                    "availableSeedNames": ["SunFlower", "Peashooter", "WallNut"],
                },
                True,
                {
                    "visibleSeedCardNames": ["SunFlower", "Peashooter", "WallNut"],
                    "newPlantUnlockedName": "WallNut",
                },
                ["SunFlower", "Peashooter", "WallNut"],
                [],
                "",
                {"post_win_transition_completed": True, "post_win_blocked_reason": ""},
            )

        generalist_module.collect_post_win_unlocks = _ok_collect
        generalist_module.replay_current_level_after_validation_win = lambda *_args, **_kwargs: (True, "")
        env_ok._finish_episode({"episode_summary": {"done_reason": "win", "episode_reward": 1.0, "episode_length": 10}})
        assert_case(
            results,
            "post-win success records newly unlocked and advances",
            "WallNut" in env_ok.curriculum.unlocked_seeds()
            and "WallNut" in list(env_ok.context.get("newly_unlocked", []))
            and int(env_ok.context.get("current_level", 0) or 0) >= 2,
            {
                "newly_unlocked": env_ok.context.get("newly_unlocked"),
                "current_level": env_ok.context.get("current_level"),
                "last_result": env_ok.context.get("last_result"),
            },
        )
        assert_case(
            results,
            "post-win success does not force env_corruption",
            str(env_ok.context.get("last_result", "")) != "env_corruption",
            env_ok.context.get("last_result"),
        )

        env_n1 = fake_generalist_env(Path(tempfile.mkdtemp()))
        env_n1.current_level = 4
        env_n1.frontier_win_streak_required = 1
        env_n1.current_sample_source = "frontier"
        env_n1._finish_episode({"episode_summary": {"done_reason": "win", "episode_reward": 1.0, "episode_length": 10}})
        assert_case(
            results,
            "N=1 preserves immediate promotion behavior",
            int(env_n1.context.get("current_level", 0) or 0) == 5 and bool(env_n1.context.get("frontier_promoted_this_episode", False)),
            {
                "current_level": env_n1.context.get("current_level"),
                "frontier_promoted_this_episode": env_n1.context.get("frontier_promoted_this_episode"),
            },
        )

        replay_calls: List[str] = []

        def _replay_same_level(*_args: Any, **_kwargs: Any) -> Any:
            replay_calls.append("ok")
            return True, ""

        collect_calls: List[int] = []

        def _ok_collect_tracking(*args: Any, **kwargs: Any) -> Any:
            if len(args) >= 5:
                collect_calls.append(int(args[4]))
            return _ok_collect(*args, **kwargs)

        generalist_module.collect_post_win_unlocks = _ok_collect_tracking
        generalist_module.replay_current_level_after_validation_win = _replay_same_level
        env_n3 = fake_generalist_env(Path(tempfile.mkdtemp()))
        env_n3.current_level = 4
        env_n3.frontier_win_streak_required = 3
        env_n3.current_sample_source = "frontier"
        env_n3._safe_adventure_state = lambda: {"currentAdventureLevel": 4}
        env_n3._finish_episode({"episode_summary": {"done_reason": "win", "episode_reward": 1.0, "episode_length": 10}})
        assert_case(
            results,
            "N=3 one win does not promote frontier level",
            int(env_n3.context.get("current_level", 0) or 0) == 4
            and int(env_n3.context.get("frontier_win_streak", 0) or 0) == 1
            and not bool(env_n3.context.get("frontier_promoted_this_episode", False)),
            {
                "current_level": env_n3.context.get("current_level"),
                "frontier_win_streak": env_n3.context.get("frontier_win_streak"),
                "frontier_promoted_this_episode": env_n3.context.get("frontier_promoted_this_episode"),
            },
        )
        assert_case(
            results,
            "N=3 one win gates trophy/post-win transition before threshold",
            collect_calls == []
            and str(env_n3.context.get("post_win_decision", "")) == "replay_same_level"
            and bool(env_n3.context.get("post_win_transition_allowed", True)) is False
            and bool(env_n3.context.get("post_win_transition", {}).get("trophy_clicked", True)) is False,
            {
                "collect_calls": collect_calls,
                "post_win_decision": env_n3.context.get("post_win_decision"),
                "post_win_transition_allowed": env_n3.context.get("post_win_transition_allowed"),
                "post_win_transition": env_n3.context.get("post_win_transition"),
            },
        )
        assert_case(
            results,
            "mastery replay source is selected while streak is pending",
            env_n3._sample_source() == "frontier_mastery_replay",
            env_n3.context.get("requested_episode_sample_source"),
        )

        env_n3.current_attempt = 1
        env_n3.current_sample_source = "frontier_mastery_replay"
        env_n3._finish_episode({"episode_summary": {"done_reason": "win", "episode_reward": 1.0, "episode_length": 10}})
        env_n3.current_attempt = 1
        env_n3.current_sample_source = "frontier_mastery_replay"
        env_n3._finish_episode({"episode_summary": {"done_reason": "win", "episode_reward": 1.0, "episode_length": 10}})
        assert_case(
            results,
            "N=3 three consecutive wins promotes frontier level",
            int(env_n3.context.get("current_level", 0) or 0) == 5
            and int(env_n3.context.get("frontier_win_streak", 0) or 0) == 0
            and bool(env_n3.context.get("frontier_promoted_this_episode", False))
            and 4 in list(env_n3.context.get("frontier_mastered_levels", [])),
            {
                "current_level": env_n3.context.get("current_level"),
                "frontier_win_streak": env_n3.context.get("frontier_win_streak"),
                "frontier_mastered_levels": env_n3.context.get("frontier_mastered_levels"),
                "frontier_promoted_this_episode": env_n3.context.get("frontier_promoted_this_episode"),
            },
        )
        assert_case(
            results,
            "N=3 post-win transition is invoked only at threshold",
            collect_calls == [4],
            collect_calls,
        )
        last_progress = getattr(env_n3, "_last_progress", None)
        assert_case(
            results,
            "progress rows include frontier mastery diagnostics fields",
            bool(last_progress)
            and hasattr(last_progress, "frontier_win_streak")
            and hasattr(last_progress, "frontier_win_streak_required")
            and hasattr(last_progress, "frontier_mastery_ready")
            and hasattr(last_progress, "frontier_replay_supported"),
            vars(last_progress) if last_progress else {},
        )

        env_loss = fake_generalist_env(Path(tempfile.mkdtemp()))
        env_loss.current_level = 4
        env_loss.frontier_win_streak_required = 3
        env_loss.frontier_win_streak = 2
        env_loss.current_sample_source = "frontier_mastery_replay"
        env_loss._finish_episode({"episode_summary": {"done_reason": "loss", "episode_reward": -1.0, "episode_length": 10}})
        assert_case(
            results,
            "loss resets frontier mastery streak to zero",
            int(env_loss.context.get("frontier_win_streak", -1)) == 0
            and str(env_loss.context.get("frontier_mastery_reset_reason", "")) == "loss",
            {
                "frontier_win_streak": env_loss.context.get("frontier_win_streak"),
                "frontier_mastery_reset_reason": env_loss.context.get("frontier_mastery_reset_reason"),
            },
        )

        env_timeout = fake_generalist_env(Path(tempfile.mkdtemp()))
        env_timeout.current_level = 4
        env_timeout.frontier_win_streak_required = 3
        env_timeout.frontier_win_streak = 2
        env_timeout.current_sample_source = "frontier_mastery_replay"
        env_timeout._finish_episode({"episode_summary": {"done_reason": "timeout", "episode_reward": -1.0, "episode_length": 10}})
        assert_case(
            results,
            "timeout resets frontier mastery streak to zero",
            int(env_timeout.context.get("frontier_win_streak", -1)) == 0
            and str(env_timeout.context.get("frontier_mastery_reset_reason", "")) == "timeout",
            {
                "frontier_win_streak": env_timeout.context.get("frontier_win_streak"),
                "frontier_mastery_reset_reason": env_timeout.context.get("frontier_mastery_reset_reason"),
            },
        )

        env_maintenance = fake_generalist_env(Path(tempfile.mkdtemp()))
        env_maintenance.current_level = 4
        env_maintenance.frontier_win_streak_required = 3
        env_maintenance.frontier_win_streak = 1
        env_maintenance.current_sample_source = "maintenance"
        env_maintenance._finish_episode({"episode_summary": {"done_reason": "win", "episode_reward": 1.0, "episode_length": 10}})
        assert_case(
            results,
            "non-frontier sample wins do not increment mastery streak",
            int(env_maintenance.context.get("frontier_win_streak", -1) or -1) == 1
            and int(env_maintenance.context.get("current_level", 0) or 0) == 4,
            {
                "sample_source": env_maintenance.current_sample_source,
                "frontier_win_streak": env_maintenance.context.get("frontier_win_streak"),
                "current_level": env_maintenance.context.get("current_level"),
            },
        )

        env_unlock_before_promotion = fake_generalist_env(Path(tempfile.mkdtemp()))
        env_unlock_before_promotion.current_level = 4
        env_unlock_before_promotion.frontier_win_streak_required = 3
        env_unlock_before_promotion.current_sample_source = "frontier"
        env_unlock_before_promotion._safe_adventure_state = lambda: {"currentAdventureLevel": 4}
        collect_calls.clear()
        env_unlock_before_promotion._finish_episode({"episode_summary": {"done_reason": "win", "episode_reward": 1.0, "episode_length": 10}})
        assert_case(
            results,
            "unlocks are not collected before frontier mastery promotion",
            "WallNut" not in list(env_unlock_before_promotion.context.get("newly_unlocked", []))
            and int(env_unlock_before_promotion.context.get("current_level", 0) or 0) == 4
            and collect_calls == [],
            {
                "newly_unlocked": env_unlock_before_promotion.context.get("newly_unlocked"),
                "current_level": env_unlock_before_promotion.context.get("current_level"),
                "collect_calls": collect_calls,
            },
        )

        def _replay_blocked(*_args: Any, **_kwargs: Any) -> Any:
            return False, "win_replay_reset_failed"

        generalist_module.replay_current_level_after_validation_win = _replay_blocked
        env_replay_blocked = fake_generalist_env(Path(tempfile.mkdtemp()))
        env_replay_blocked.current_level = 4
        env_replay_blocked.frontier_win_streak_required = 3
        env_replay_blocked.current_sample_source = "frontier"
        env_replay_blocked._finish_episode({"episode_summary": {"done_reason": "win", "episode_reward": 1.0, "episode_length": 10}})
        assert_case(
            results,
            "replay unavailable with N>1 requests recovery with precise reason",
            str(env_replay_blocked.context.get("blocked_reason", "")) == "win_replay_reset_failed"
            and str(env_replay_blocked.context.get("frontier_replay_blocked_reason", "")) == "win_replay_reset_failed"
            and str(env_replay_blocked._hard_blocked_reason) == ""
            and env_replay_blocked.context.get("frontier_replay_recovery_required") is True
            and int(env_replay_blocked.context.get("current_level", 0) or 0) == 4,
            {
                "blocked_reason": env_replay_blocked.context.get("blocked_reason"),
                "frontier_replay_blocked_reason": env_replay_blocked.context.get("frontier_replay_blocked_reason"),
                "frontier_replay_recovery_required": env_replay_blocked.context.get("frontier_replay_recovery_required"),
                "current_level": env_replay_blocked.context.get("current_level"),
            },
        )
    finally:
        generalist_module.collect_post_win_unlocks = original_collect
        generalist_module.replay_current_level_after_validation_win = original_replay
        generalist_module.build_live_status = original_live_status

    dash = fake_dashboard()
    generalist_tab_source = inspect.getsource(PvZDashboard._build_train_tab)
    assert_case(results, "GUI has reusable scrollable container helper", hasattr(PvZDashboard, "_make_scrollable_container"))
    assert_case(
        results,
        "GUI live generalist training tab uses scrollable container",
        "_make_scrollable_container(parent)" in generalist_tab_source,
        generalist_tab_source,
    )
    status_idx = generalist_tab_source.find("Adventure Generalist Training")
    basic_idx = generalist_tab_source.find("Core Training Settings")
    paths_idx = generalist_tab_source.find("Model / Run Paths")
    advanced_idx = generalist_tab_source.find("Advanced Settings")
    preview_idx = generalist_tab_source.find("Command Preview")
    train_button_idx = generalist_tab_source.find("Start Training")
    resume_button_idx = generalist_tab_source.find("Resume Training")
    assert_case(
        results,
        "GUI live section hierarchy is ordered for usability",
        0 <= status_idx < basic_idx < paths_idx < train_button_idx < advanced_idx < preview_idx,
        {
            "status_idx": status_idx,
            "basic_idx": basic_idx,
            "paths_idx": paths_idx,
            "advanced_idx": advanced_idx,
            "preview_idx": preview_idx,
        },
    )
    assert_case(
        results,
        "GUI live training actions expose start and resume before advanced settings",
        0 <= train_button_idx < resume_button_idx < advanced_idx,
        {
            "train_button_idx": train_button_idx,
            "resume_button_idx": resume_button_idx,
            "advanced_idx": advanced_idx,
        },
    )

    resume_cli_config = build_generalist_config_for_test(resume_model_path="runs/old_generalist/model.zip")
    resume_model_path_text = str(resume_cli_config.get("resume_model_path", "")).replace("\\", "/")
    model_path_text = str(resume_cli_config.get("model_path", "")).replace("\\", "/")
    assert_case(
        results,
        "CLI resume-model-path maps to training continuation config",
        bool(resume_cli_config.get("resume_training"))
        and bool(resume_cli_config.get("checkpoint_warm_start"))
        and resume_model_path_text.endswith("runs/old_generalist/model.zip")
        and model_path_text.endswith("runs/old_generalist/model.zip"),
        {
            "resume_training": resume_cli_config.get("resume_training"),
            "checkpoint_warm_start": resume_cli_config.get("checkpoint_warm_start"),
            "resume_model_path": resume_cli_config.get("resume_model_path"),
            "model_path": resume_cli_config.get("model_path"),
        },
    )
    fresh_cli_config = build_generalist_config_for_test()
    assert_case(
        results,
        "CLI blank resume path preserves fresh Adventure Generalist training mode",
        not bool(fresh_cli_config.get("resume_training"))
        and not bool(fresh_cli_config.get("checkpoint_warm_start"))
        and bool(fresh_cli_config.get("scratch_initialization"))
        and str(fresh_cli_config.get("model_path", "")) == "",
        {
            "resume_training": fresh_cli_config.get("resume_training"),
            "checkpoint_warm_start": fresh_cli_config.get("checkpoint_warm_start"),
            "scratch_initialization": fresh_cli_config.get("scratch_initialization"),
            "model_path": fresh_cli_config.get("model_path"),
        },
    )
    raw_config_seed_list = build_config(
        DefaultArgs(
            run_mode=None,
            run_dir=None,
            model=None,
            resume_model_path=None,
            adventure_generalist_train=True,
            adventure_generalist_eval=False,
        ),
        {
            "seed_list": ["SunFlower", "SunFlower", "Peashooter", "Peashooter"],
            "initial_loadout": ["SunFlower", "SunFlower", "Peashooter", "Peashooter"],
        },
    )
    assert_case(
        results,
        "Adventure Generalist raw config seed_list is authoritative",
        raw_config_seed_list.get("seed_list") == ADVENTURE_GENERALIST_INITIAL_LOADOUT
        and raw_config_seed_list.get("configured_seed_list") == ADVENTURE_GENERALIST_INITIAL_LOADOUT
        and raw_config_seed_list.get("seed_order_source") == "explicit_config",
        {
            "seed_list": raw_config_seed_list.get("seed_list"),
            "configured_seed_list": raw_config_seed_list.get("configured_seed_list"),
            "seed_order_source": raw_config_seed_list.get("seed_order_source"),
        },
    )
    assert_resume_metadata_compatibility_checks(results)

    train_command = dash._build_adventure_generalist_command()
    assert_case(results, "GUI train command uses Adventure Generalist train flag", "--adventure-generalist-train" in train_command, train_command)
    assert_case(
        results,
        "GUI train command relies on the sole Generalist action contract",
        "--action-space-mode" not in train_command,
        train_command,
    )
    assert_case(results, "GUI train command includes initial loadout", GUI_INITIAL_LOADOUT in train_command, train_command)
    seed_list_index = train_command.index("--seed-list") if "--seed-list" in train_command else -1
    assert_case(
        results,
        "GUI train command treats Generalist seed list field as authoritative",
        seed_list_index >= 0
        and (seed_list_index + 1) < len(train_command)
        and train_command[seed_list_index + 1] == GUI_INITIAL_LOADOUT,
        train_command,
    )
    assert_case(
        results,
        "GUI train command omits seed order randomization by default",
        "--randomize-seed-order" not in train_command,
        train_command,
    )
    assert_case(
        results,
        "GUI train command blank resume field omits resume flag",
        "--resume-model-path" not in train_command and "--model-path" not in train_command and "--model" not in train_command,
        train_command,
    )
    dash.generalist_resume_model_path_var = Var("runs/fake_generalist_model.zip")
    resumed_train_command = dash._build_adventure_generalist_command()
    resume_flag_index = resumed_train_command.index("--resume-model-path") if "--resume-model-path" in resumed_train_command else -1
    expected_resume_path = str((Path.cwd() / "runs" / "fake_generalist_model.zip").resolve())
    actual_resume_path = (
        resumed_train_command[resume_flag_index + 1]
        if resume_flag_index >= 0 and (resume_flag_index + 1) < len(resumed_train_command)
        else ""
    )
    assert_case(
        results,
        "GUI train command populated resume field includes --resume-model-path",
        resume_flag_index >= 0 and actual_resume_path == expected_resume_path,
        {"command": resumed_train_command, "expected_resume_path": expected_resume_path, "actual_resume_path": actual_resume_path},
    )
    assert_case(
        results,
        "GUI train command emits frontier mastery streak flag",
        "--adventure-frontier-win-streak-required" in resumed_train_command and "1" in resumed_train_command,
        resumed_train_command,
    )
    dash_randomize = fake_dashboard()
    dash_randomize.generalist_randomize_seed_order_var = Var(True)
    randomized_train_command = dash_randomize._build_adventure_generalist_command()
    assert_case(
        results,
        "GUI train command only enables randomization behind explicit checkbox",
        "--randomize-seed-order" in randomized_train_command,
        randomized_train_command,
    )
    dash_with_coach = fake_dashboard()
    dash_with_coach.human_coach_enabled_var = Var(True)
    dash_with_coach.human_coach_reward_var = Var(True)
    dash_with_coach.human_coach_log_path_var = Var("runs/human_coach.jsonl")
    dash_with_coach.human_coach_command_path_var = Var("runs/coach_commands.jsonl")
    dash_with_coach.stream_coach_enabled_var = Var(True)
    dash_with_coach.stream_coach_platform_var = Var("mock")
    dash_with_coach.stream_coach_reward_var = Var(True)
    dash_with_coach.stream_coach_log_path_var = Var("runs/stream_coach.jsonl")
    dash_with_coach.coach_allow_fusion_planning_var = Var(True)
    dash_with_coach.fusion_bridge_enabled_var = Var(True)
    dash_with_coach.human_coach_fusion_reward_var = Var("0.07")
    coach_train_command = dash_with_coach._build_adventure_generalist_command()
    assert_case(
        results,
        "Adventure Generalist launch includes coach flags when enabled",
        bool(
            "--human-coach-enabled" in coach_train_command
            and "--human-coach-reward" in coach_train_command
            and "--human-coach-log-path" in coach_train_command
            and "--human-coach-command-path" in coach_train_command
            and "--stream-coach-enabled" in coach_train_command
            and "--stream-coach-platform" in coach_train_command
            and "--stream-coach-reward" in coach_train_command
            and "--stream-coach-log-path" in coach_train_command
            and "--coach-allow-fusion-planning" in coach_train_command
            and "--fusion-bridge-enabled" in coach_train_command
            and "--live-status-path" in coach_train_command
        ),
        coach_train_command,
    )
    assert_case(
        results,
        "Coach fusion success reward entry flows the adjusted value into the command",
        "--coach-fusion-success-reward" in coach_train_command
        and coach_train_command[coach_train_command.index("--coach-fusion-success-reward") + 1] == "0.07",
        coach_train_command,
    )
    assert_case(
        results,
        "Coach reward flags omitted when value is blank",
        "--coach-fusion-success-reward" not in fake_dashboard()._build_adventure_generalist_command()
        and "--coach-tactical-usefulness-reward" not in coach_train_command,
        coach_train_command,
    )
    dash_without_coach = fake_dashboard()
    coachless_train_command = dash_without_coach._build_adventure_generalist_command()
    assert_case(
        results,
        "Adventure Generalist launch omits coach flags when disabled",
        bool(
            "--human-coach-enabled" not in coachless_train_command
            and "--stream-coach-enabled" not in coachless_train_command
            and "--coach-allow-fusion-planning" not in coachless_train_command
            and "--fusion-bridge-enabled" not in coachless_train_command
        ),
        coachless_train_command,
    )
    coach_eval_command = dash_with_coach._build_adventure_generalist_eval_command()
    assert_case(
        results,
        "Adventure Generalist eval launch includes coach flags when enabled",
        "--human-coach-enabled" in coach_eval_command and "--stream-coach-enabled" in coach_eval_command,
        coach_eval_command,
    )

    fusion_mask_default = fake_dashboard()
    default_train_command = fusion_mask_default._build_adventure_generalist_command()
    default_eval_command = fusion_mask_default._build_adventure_generalist_eval_command()
    assert_case(
        results,
        "Adventure Generalist train defaults fusion action mask ON",
        "--fusion-action-mask-enabled" in default_train_command,
        default_train_command,
    )
    assert_case(
        results,
        "Adventure Generalist eval defaults fusion action mask OFF",
        "--fusion-action-mask-enabled" not in default_eval_command,
        default_eval_command,
    )
    legacy_train_dash = fake_dashboard()
    legacy_train_dash.generalist_fusion_action_mask_train_var = Var(False)
    legacy_train_command = legacy_train_dash._build_adventure_generalist_command()
    assert_case(
        results,
        "Unchecking train fusion mask restores legacy command (flag omitted)",
        "--fusion-action-mask-enabled" not in legacy_train_command,
        legacy_train_command,
    )
    fusion_eval_dash = fake_dashboard()
    fusion_eval_dash.generalist_fusion_action_mask_eval_var = Var(True)
    fusion_eval_command = fusion_eval_dash._build_adventure_generalist_eval_command()
    assert_case(
        results,
        "Enabling eval fusion mask adds the flag to the eval command",
        "--fusion-action-mask-enabled" in fusion_eval_command,
        fusion_eval_command,
    )

    train_source = Path("python/train_ppo.py").read_text(encoding="utf-8")
    assert_case(
        results,
        "CLI parser includes adventure frontier mastery streak argument",
        "--adventure-frontier-win-streak-required" in train_source,
        "--adventure-frontier-win-streak-required",
    )
    assert_case(
        results,
        "CLI parser includes fusion action mask argument and live status surfaces it",
        "--fusion-action-mask-enabled" in train_source
        and "\"fusion_action_mask_enabled\"" in train_source,
        "--fusion-action-mask-enabled",
    )
    assert_case(
        results,
        "CLI parser includes Adventure Generalist --resume-model-path",
        "--resume-model-path" in train_source,
        "--resume-model-path",
    )
    assert_case(
        results,
        "CLI parser includes explicit seed order randomization flag",
        "--randomize-seed-order" in train_source,
        "--randomize-seed-order",
    )

    dash.generalist_eval_model_path_var = Var("runs/fake_generalist_model.zip")
    eval_command = dash._build_adventure_generalist_eval_command()
    assert_case(results, "GUI eval command uses Adventure Generalist eval flag", "--adventure-generalist-eval" in eval_command, eval_command)

    with tempfile.TemporaryDirectory(prefix="pvzrl_gui_coach_queue_") as temp_dir:
        queue_dash = fake_dashboard()
        queue_path = Path(temp_dir) / "coach_commands.jsonl"
        queue_logs: List[str] = []
        queue_dash._append_log = lambda text: queue_logs.append(str(text))
        queue_dash.human_coach_command_path_var = Var(str(queue_path))
        queue_dash.human_coach_command_input_var = Var("plant 0 2 4")
        queue_dash.send_human_coach_command()
        queue_lines = queue_path.read_text(encoding="utf-8").strip().splitlines()
        queue_payload = json.loads(queue_lines[-1]) if queue_lines else {}
        assert_case(
            results,
            "GUI command queue writes valid JSONL",
            bool(queue_lines and str(queue_payload.get("command", "")) == "plant 0 2 4"),
            {"lines": queue_lines, "payload": queue_payload, "logs": queue_logs},
        )

    dash._set_coach_live_fields({})
    assert_case(
        results,
        "GUI diagnostics does not crash when coach fields are missing",
        bool(
            dash.human_coach_enabled_status_var.get() == "n/a"
            and dash.stream_coach_enabled_status_var.get() == "n/a"
            and dash.fusion_bridge_available_var.get() == "n/a"
        ),
        {
            "human_enabled": dash.human_coach_enabled_status_var.get(),
            "stream_enabled": dash.stream_coach_enabled_status_var.get(),
            "fusion_bridge_available": dash.fusion_bridge_available_var.get(),
        },
    )
    with tempfile.TemporaryDirectory(prefix="pvzrl_gui_live_status_") as temp_dir:
        live_dash = fake_dashboard()
        live_path = Path(temp_dir) / "live_status.json"
        payload = {
            "mode": "adventure_generalist_14slot_train",
            "action_count": 701,
            "action_decoder_version": "seedslot14x50_plus_wait_v1",
            "current_level": 2,
            "wave": 3,
            "sun": 450,
            "human_coach_enabled": True,
            "human_coach_last_command": {"kind": "plant", "seed_index": 0, "row": 2, "col": 4},
            "human_coach_last_action": 25,
            "stream_coach_enabled": True,
            "stream_coach_last_command": {"command": "plant", "seed_index": 0, "row": 2, "col": 4},
            "stream_coach_last_action": 25,
            "stream_coach_last_vote_count": 3,
            "fusion_bridge_available": True,
            "fusion_last_result": "success",
            "human_coach_override_count": 1,
            "human_coach_match_count": 2,
            "human_coach_reward_total": 0.03,
        }
        live_path.write_text(json.dumps(payload), encoding="utf-8")
        live_dash.live_status_path = live_path
        parsed_payload, parsed_info = live_dash._read_live_status_file()
        live_dash._set_coach_live_fields(parsed_payload or {})
        assert_case(
            results,
            "GUI diagnostics updates from live_status.json",
            bool(
                isinstance(parsed_payload, dict)
                and parsed_info.get("health") in {"LIVE", "STALE", "DEAD", "BLOCKED_POST_WIN", "BLOCKED_SEED_SELECTION", "BLOCKED_GAMEPLAY_READY"}
                and live_dash.human_coach_last_action_var.get() == "25"
                and live_dash.stream_coach_last_vote_count_var.get() == "3"
            ),
            {
                "health": parsed_info.get("health"),
                "human_last_action": live_dash.human_coach_last_action_var.get(),
                "stream_vote_count": live_dash.stream_coach_last_vote_count_var.get(),
            },
        )

    dash_missing_model = fake_dashboard()
    log_lines: List[str] = []
    dash_missing_model._append_log = lambda text: log_lines.append(str(text))
    dash_missing_model.launch_process = lambda _name, _command: log_lines.append("launched")
    dash_missing_model.start_adventure_generalist_eval()
    assert_case(
        results,
        "GUI eval launch requires model path",
        any("requires model_path" in line for line in log_lines),
        log_lines,
    )

    dash_invalid_loadout = fake_dashboard()
    loadout_logs: List[str] = []
    loadout_launches: List[str] = []
    dash_invalid_loadout.generalist_initial_loadout_var = Var("SunFlower,Peashooter")
    dash_invalid_loadout._append_log = lambda text: loadout_logs.append(str(text))
    dash_invalid_loadout.launch_process = lambda name, _command: loadout_launches.append(str(name))
    dash_invalid_loadout.start_adventure_generalist_train()
    assert_case(
        results,
        "GUI train validation blocks invalid initial loadout",
        any("requires initial loadout" in line for line in loadout_logs) and not loadout_launches,
        {"logs": loadout_logs, "launches": loadout_launches},
    )

    dash_invalid_slots = fake_dashboard()
    slot_logs: List[str] = []
    slot_launches: List[str] = []
    dash_invalid_slots.generalist_max_seed_slots_var = Var("13")
    dash_invalid_slots._append_log = lambda text: slot_logs.append(str(text))
    dash_invalid_slots.launch_process = lambda name, _command: slot_launches.append(str(name))
    dash_invalid_slots.start_adventure_generalist_train()
    assert_case(
        results,
        "GUI train validation blocks invalid max seed slots",
        any("requires max seed slots = 14" in line for line in slot_logs) and not slot_launches,
        {"logs": slot_logs, "launches": slot_launches},
    )

    dash_valid_launch = fake_dashboard()
    launch_logs: List[str] = []
    launch_calls: List[str] = []
    dash_valid_launch._append_log = lambda text: launch_logs.append(str(text))
    dash_valid_launch.launch_process = lambda name, _command: launch_calls.append(str(name))
    dash_valid_launch.start_adventure_generalist_train()
    assert_case(
        results,
        "GUI train launch logs clear startup message",
        any("Launching Adventure Generalist training" in line for line in launch_logs) and "Start Adventure Generalist Train" in launch_calls,
        {"logs": launch_logs, "launches": launch_calls},
    )

    dash_missing_resume = fake_dashboard()
    missing_resume_logs: List[str] = []
    missing_resume_launches: List[str] = []
    dash_missing_resume.generalist_resume_model_path_var = Var("runs/does_not_exist_resume_model.zip")
    dash_missing_resume._append_log = lambda text: missing_resume_logs.append(str(text))
    dash_missing_resume.launch_process = lambda name, _command: missing_resume_launches.append(str(name))
    dash_missing_resume.start_adventure_generalist_train()
    assert_case(
        results,
        "GUI train resume path must exist",
        any("resume model does not exist" in line for line in missing_resume_logs) and not missing_resume_launches,
        {"logs": missing_resume_logs, "launches": missing_resume_launches},
    )

    status_payload = {
        "run_mode": "adventure_generalist_14slot_train",
        "status": "running",
        "adventure_phase": "REPLAY_CURRENT_LEVEL",
        "frontier_level": 4,
        "current_level": 4,
        "current_attempt": 2,
        "frontier_win_streak": 2,
        "frontier_win_streak_required": 3,
        "wins_on_current_level": 2,
        "wins_before_advance": 3,
        "frontier_mastery_ready": False,
        "post_win_decision": "replay_same_level",
        "post_win_transition_allowed": False,
        "latest_terminal_result": "win",
        "reset_phase": "waiting_seed_selection",
        "expected_transition_target": "same_level_replay",
        "seed_selection_expected": True,
        "frontier_replay_supported": True,
        "latest_unlock": "WallNut",
        "unlocked_seeds": ["SunFlower", "Peashooter", "WallNut"],
        "eligible_seeds": ["SunFlower", "Peashooter", "WallNut"],
        "selectable_seeds": ["SunFlower", "Peashooter", "WallNut"],
        "observed_seed_bank_capacity": 5,
        "bridge_reported_capacity": 4,
        "inferred_capacity_from_unlocks": 5,
        "effective_seed_capacity": 5,
        "max_effective_seed_capacity_seen": 5,
        "inferred_capacity_source": "selectable_priority_seeds",
        "capacity_inference_reason": "WallNut available",
        "rejected_priority_seeds": [],
        "selected_loadout": ["SunFlower", "SunFlower", "Peashooter", "Peashooter", "WallNut"],
        "selected_loadout_count": 5,
        "inactive_model_slots": 9,
        "loadout_reason": "conservative_fill_open_slots_with_new",
        "post_win_blocked_reason": "",
    }
    generalist_status = dash._generalist_status_content(status_payload, health="LIVE", using_last_good=False)
    assert_case(
        results,
        "GUI generalist status renders selectable seeds capacity and frontier streak",
        "selectable_seeds" in generalist_status
        and "SunFlower, Peashooter, WallNut" in generalist_status
        and "observed_capacity" in generalist_status
        and "effective_capacity" in generalist_status
        and "capacity_source" in generalist_status
        and "selectable_priority_seeds" in generalist_status
        and "rejected_priority" in generalist_status
        and "5" in generalist_status
        and "frontier_streak" in generalist_status
        and "2 / 3" in generalist_status
        and "wins_before_advance" in generalist_status
        and "decision" in generalist_status
        and "replay_same_level" in generalist_status
        and "reset_phase" in generalist_status
        and "waiting_seed_selection" in generalist_status,
        generalist_status,
    )

    reset_source = inspect.getsource(PvZGymEnv._reset_state_machine)
    assert_case(
        results,
        "reset state machine fails loudly instead of looping forever on impossible seed-selection wait",
        "seedSelectionImpossibleState" in reset_source
        and "gameplay_ready_observed_repeatedly_before_seed_selection" in reset_source,
        "seedSelectionImpossibleState" in reset_source,
    )

    payload = {"ok": all(result["passed"] for result in results), "results": results}
    print(json.dumps(payload, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
