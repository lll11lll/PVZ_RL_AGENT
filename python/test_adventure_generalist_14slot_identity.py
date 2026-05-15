"""Focused tests for Adventure Generalist 14-slot identity unlock adoption."""

from __future__ import annotations

import json
import inspect
import tempfile
from pathlib import Path
from typing import Any, Dict, List

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
    BLOCKED_FRONTIER_REPLAY_REQUIRED,
    BLOCKED_INITIAL_LOADOUT_UNAVAILABLE,
)
from pvzrl_env import PvZGymEnv, decode_action
from pvzrl_gui import (
    ADVENTURE_GENERALIST_ACTION_SPACE_MODE,
    ADVENTURE_GENERALIST_INITIAL_LOADOUT as GUI_INITIAL_LOADOUT,
    PvZDashboard,
)
from pvzrl_seed_inventory import (
    ADVENTURE_IDENTITY_FEATURES_PER_SLOT,
    ADVENTURE_IDENTITY_ONE_HOT_WIDTH,
    ADVENTURE_IDENTITY_SCALAR_FEATURES_PER_SLOT,
    adventure_identity_feature_count,
    adventure_identity_features,
)


class Var:
    def __init__(self, value: str | bool) -> None:
        self.value = value

    def get(self) -> str | bool:
        return self.value

    def set(self, value: str | bool) -> None:
        self.value = value


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
    dashboard.generalist_eval_model_path_var = Var("")
    dashboard.generalist_eval_episodes_var = Var("5")
    dashboard.generalist_unlock_curriculum_var = Var(True)
    dashboard.generalist_replay_cleared_var = Var(True)
    dashboard.generalist_final_wave_extension_var = Var(True)
    dashboard.generalist_wait_gameplay_ready_var = Var(True)
    dashboard.generalist_quick_wait_var = Var(True)
    dashboard.generalist_tactical_masks_var = Var(True)
    dashboard.generalist_wallnut_mask_var = Var(True)
    dashboard.generalist_cherrybomb_mask_var = Var(True)
    dashboard.generalist_curriculum_var = Var("conservative")
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
    env.current_loadout = list(ADVENTURE_GENERALIST_INITIAL_LOADOUT)
    env.current_loadout_reason = "initial"
    env.current_selectable_seeds = ["SunFlower", "Peashooter"]
    env.current_excluded_new_plants = []
    env.observed_seed_bank_capacity = 4
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
    env.config = type("Cfg", (), {"gameplay_ready_timeout": 8.0})()
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
        "selected_loadout_count": len(env.current_loadout),
        "active_seed_slot_count": len(env.current_loadout),
        "inactive_seed_slot_count": 10,
        "max_seed_slots": 14,
        "observed_seed_bank_capacity": 4,
        "active_seed_slot_capacity": 4,
        "current_seed_bank_capacity": 4,
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
    env._safe_adventure_state = lambda: {
        "screenState": "reward_screen",
        "isSeedSelectionScreen": True,
        "availableSeedNames": ["SunFlower", "Peashooter", "WallNut"],
    }
    return env


def assert_case(results: List[Dict[str, Any]], name: str, condition: bool, detail: Any = None) -> None:
    results.append({"case": name, "passed": bool(condition), "detail": detail})


def reason_for(excluded: List[Dict[str, Any]], seed: str) -> str:
    for row in excluded:
        if str(row.get("seed")) == seed:
            return str(row.get("reason", ""))
    return ""


def main() -> int:
    results: List[Dict[str, Any]] = []

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
        "future selectable snapshot can adopt WallNut in same run",
        "WallNut" in decision_future.selected_loadout,
        {"loadout": decision_future.selected_loadout, "reason": decision_future.loadout_reason},
    )

    decision_cap4 = curriculum.choose_loadout(["SunFlower", "Peashooter", "WallNut"], observed_capacity=4)
    decision_cap5 = curriculum.choose_loadout(["SunFlower", "Peashooter", "WallNut"], observed_capacity=5)
    assert_case(results, "capacity=4 keeps loadout length <=4", len(decision_cap4.selected_loadout) <= 4, decision_cap4.selected_loadout)
    assert_case(
        results,
        "capacity=5 can fill fifth slot",
        len(decision_cap5.selected_loadout) == 5 and "WallNut" in decision_cap5.selected_loadout,
        decision_cap5.selected_loadout,
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
        reason_for(delay_decision.excluded_new_plants, "WallNut") == "unlock_delay" and "WallNut" not in delay_decision.selected_loadout,
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
        conservative_decision.selected_loadout[:4] == ADVENTURE_GENERALIST_INITIAL_LOADOUT and conservative_decision.selected_loadout[4] == "WallNut",
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
            "replay unavailable with N>1 fails with explicit blocked reason",
            str(env_replay_blocked.context.get("blocked_reason", "")) == BLOCKED_FRONTIER_REPLAY_REQUIRED
            and str(env_replay_blocked.context.get("frontier_replay_blocked_reason", "")) == BLOCKED_FRONTIER_REPLAY_REQUIRED
            and str(env_replay_blocked._hard_blocked_reason) == BLOCKED_FRONTIER_REPLAY_REQUIRED
            and int(env_replay_blocked.context.get("current_level", 0) or 0) == 4,
            {
                "blocked_reason": env_replay_blocked.context.get("blocked_reason"),
                "frontier_replay_blocked_reason": env_replay_blocked.context.get("frontier_replay_blocked_reason"),
                "current_level": env_replay_blocked.context.get("current_level"),
            },
        )
    finally:
        generalist_module.collect_post_win_unlocks = original_collect
        generalist_module.replay_current_level_after_validation_win = original_replay
        generalist_module.build_live_status = original_live_status

    dash = fake_dashboard()
    generalist_tab_source = inspect.getsource(PvZDashboard._build_adventure_generalist_tab)
    assert_case(results, "GUI has reusable scrollable container helper", hasattr(PvZDashboard, "_make_scrollable_container"))
    assert_case(
        results,
        "GUI generalist tab uses scrollable container",
        "_make_scrollable_container(parent)" in generalist_tab_source,
        generalist_tab_source,
    )
    launch_idx = generalist_tab_source.find("Launch Controls")
    basic_idx = generalist_tab_source.find("Basic Training Settings")
    advanced_idx = generalist_tab_source.find("Advanced Curriculum / Unlock Behavior")
    masks_idx = generalist_tab_source.find("Masks / Timeouts")
    preview_idx = generalist_tab_source.find("Command Preview")
    train_button_idx = generalist_tab_source.find("Start Adventure Generalist Train")
    assert_case(
        results,
        "GUI section hierarchy is ordered for usability",
        0 <= launch_idx < basic_idx < advanced_idx < masks_idx < preview_idx,
        {
            "launch_idx": launch_idx,
            "basic_idx": basic_idx,
            "advanced_idx": advanced_idx,
            "masks_idx": masks_idx,
            "preview_idx": preview_idx,
        },
    )
    assert_case(
        results,
        "GUI Start button is in top launch section before basic settings",
        0 <= train_button_idx < basic_idx,
        {"train_button_idx": train_button_idx, "basic_idx": basic_idx},
    )

    train_command = dash._build_adventure_generalist_command()
    assert_case(results, "GUI train command uses Adventure Generalist train flag", "--adventure-generalist-train" in train_command, train_command)
    assert_case(results, "GUI train command includes 14-slot action-space mode", ADVENTURE_GENERALIST_ACTION_SPACE_MODE in train_command, train_command)
    assert_case(results, "GUI train command includes initial loadout", GUI_INITIAL_LOADOUT in train_command, train_command)
    assert_case(results, "GUI train command does not warm-start from model", "--model-path" not in train_command and "--model" not in train_command, train_command)
    assert_case(
        results,
        "GUI train command emits frontier mastery streak flag",
        "--adventure-frontier-win-streak-required" in train_command and "1" in train_command,
        train_command,
    )

    train_source = Path("python/train_ppo.py").read_text(encoding="utf-8")
    assert_case(
        results,
        "CLI parser includes adventure frontier mastery streak argument",
        "--adventure-frontier-win-streak-required" in train_source,
        "--adventure-frontier-win-streak-required",
    )

    dash.generalist_eval_model_path_var = Var("runs/fake_generalist_model.zip")
    eval_command = dash._build_adventure_generalist_eval_command()
    assert_case(results, "GUI eval command uses Adventure Generalist eval flag", "--adventure-generalist-eval" in eval_command, eval_command)

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
