from __future__ import annotations

import json
from pathlib import Path

import pytest

from pvzrl_adventure_generalist import (
    ADVENTURE_GENERALIST_INITIAL_LOADOUT,
    AdventureGeneralistTrainingEnv,
    AdventureSeedCurriculum,
    CURRICULUM_STATE_SCHEMA_VERSION,
)
from pvzrl_sb3 import PvZSB3Config


def _curriculum() -> AdventureSeedCurriculum:
    return AdventureSeedCurriculum(
        initial_loadout=ADVENTURE_GENERALIST_INITIAL_LOADOUT,
        max_seed_slots=4,
        seed_order_source="explicit_config",
        new_unlock_guarantee_episodes=4,
    )


def _real_env(run_dir: Path) -> AdventureGeneralistTrainingEnv:
    return AdventureGeneralistTrainingEnv(
        PvZSB3Config(
            seed_list=list(ADVENTURE_GENERALIST_INITIAL_LOADOUT),
            plant_types=[1, 1, 0, 0],
        ),
        run_dir=run_dir,
        live_status_path=None,
        initial_loadout=list(ADVENTURE_GENERALIST_INITIAL_LOADOUT),
        max_adventure_levels=5,
        max_attempts_per_level=3,
        adventure_start_level=1,
        unlock_aware_seed_curriculum=True,
        seed_curriculum="conservative",
        unlock_introduction_delay=0,
        new_plant_min_inclusion_prob=0.15,
        seed_order_source="explicit_config",
        randomize_seed_order=False,
        infer_capacity_from_unlocks=True,
        allow_weak_unlocked_capacity_fallback=False,
        replay_cleared_levels=False,
        frontier_sample_prob=0.6,
        recent_cleared_sample_prob=0.3,
        maintenance_sample_prob=0.1,
        frontier_win_streak_required=1,
    )


def test_curriculum_state_round_trip_preserves_guarantees_counters_and_provenance(
    tmp_path: Path,
) -> None:
    curriculum = _curriculum()
    curriculum.record_confirmed_selectable(["WallNut", "CherryBomb"])
    curriculum.record_unlocked(["WallNut"], episode_index=3)
    curriculum.episode_index = 3
    curriculum.record_eligible(["WallNut"], episode_index=3)
    curriculum.record_eligible(["WallNut"], episode_index=3)

    decision = curriculum.choose_loadout(
        selectable_seeds=["SunFlower", "Peashooter", "WallNut"],
        observed_capacity=4,
        previous_loadout=ADVENTURE_GENERALIST_INITIAL_LOADOUT,
        validation_seeds=["SunFlower", "Peashooter", "WallNut"],
    )
    curriculum.commit_loadout(decision, episode_index=3)
    curriculum.record_action_usage("WallNut", source="model")
    curriculum.record_action_usage("WallNut", source="viewer")
    curriculum.record_bc_demonstration("WallNut")

    env = object.__new__(AdventureGeneralistTrainingEnv)
    env.curriculum = curriculum
    env.curriculum_state_path = tmp_path / "curriculum_state.json"
    env.episode_index = curriculum.episode_index
    env.current_loadout = list(decision.selected_loadout)
    env.current_loadout_provenance = list(decision.loadout_provenance)
    env.persist_curriculum_state()

    state_path = tmp_path / "curriculum_state.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == CURRICULUM_STATE_SCHEMA_VERSION
    assert payload["rotation_cursor"] == curriculum.rotation_cursor
    assert payload["guarantees"]["WallNut"]["remaining"] == 3
    assert payload["episodes_eligible"]["WallNut"] == 1
    assert payload["model_actions_executed"]["WallNut"] == 1
    assert payload["viewer_actions_executed"]["WallNut"] == 1
    assert payload["bc_demonstrations_recorded"]["WallNut"] == 1
    assert payload["last_included_episode"]["WallNut"] == 3
    assert payload["loadout_provenance"] == decision.loadout_provenance
    next_rng_value = curriculum.rng.random()
    assert isinstance(payload["rng_state"], list)

    restored = _curriculum()
    restored.restore_state(payload)
    diagnostics = restored.per_seed_diagnostics(selected_loadout=decision.selected_loadout)
    wallnut = diagnostics["WallNut"]
    assert restored.episode_index == 3
    assert restored.unlock_episode["WallNut"] == 3
    assert restored.guarantee_remaining("WallNut") == 3
    assert restored.rotation_cursor == curriculum.rotation_cursor
    assert wallnut["episodes_eligible"] == 1
    assert wallnut["episodes_included"] == 1
    assert wallnut["model_actions_executed"] == 1
    assert wallnut["viewer_actions_executed"] == 1
    assert wallnut["bc_demonstrations_recorded"] == 1
    assert wallnut["last_included_episode"] == 3
    assert wallnut["currently_guaranteed"] is True
    assert wallnut["currently_selected"] is True
    assert restored.rng.random() == next_rng_value


def test_stale_curriculum_state_cannot_select_a_seed_absent_from_live_evidence() -> None:
    curriculum = _curriculum()
    curriculum.record_unlocked(["WallNut"], episode_index=1)
    curriculum.episode_index = 1
    payload = {
        "schema_version": CURRICULUM_STATE_SCHEMA_VERSION,
        "unlock_order": ["SunFlower", "Peashooter", "WallNut", "NotARealPlant"],
        "unlock_episode": {
            "SunFlower": 0,
            "Peashooter": 0,
            "WallNut": 1,
            "NotARealPlant": 0,
        },
        "episodes_included": {"WallNut": 99, "NotARealPlant": 99},
        "guarantees": {
            "WallNut": {
                "seed": "WallNut",
                "required_inclusions": 4,
                "completed_inclusions": 0,
            },
            "NotARealPlant": {
                "seed": "NotARealPlant",
                "required_inclusions": 4,
                "completed_inclusions": 0,
            },
        },
    }

    curriculum.restore_state(payload)
    decision = curriculum.choose_loadout(
        selectable_seeds=["SunFlower", "Peashooter"],
        observed_capacity=4,
        previous_loadout=ADVENTURE_GENERALIST_INITIAL_LOADOUT,
        validation_seeds=["SunFlower", "Peashooter"],
    )
    assert "WallNut" not in decision.selected_loadout
    assert "NotARealPlant" not in curriculum.unlock_episode
    assert "NotARealPlant" not in curriculum.per_seed_diagnostics()


def test_real_env_restart_preserves_last_committed_loadout_and_state_file(tmp_path: Path) -> None:
    first = _real_env(tmp_path)
    first.curriculum.record_unlocked(["WallNut"], episode_index=1)
    first.curriculum.episode_index = 1
    decision = first.curriculum.choose_loadout(
        selectable_seeds=["SunFlower", "Peashooter", "WallNut"],
        observed_capacity=4,
        previous_loadout=ADVENTURE_GENERALIST_INITIAL_LOADOUT,
        validation_seeds=["SunFlower", "Peashooter", "WallNut"],
    )
    first.curriculum.commit_loadout(decision, episode_index=1)
    first.current_loadout = list(decision.selected_loadout)
    first.current_loadout_provenance = list(decision.loadout_provenance)
    first.persist_curriculum_state()
    before = json.loads((tmp_path / "curriculum_state.json").read_text(encoding="utf-8"))

    second = _real_env(tmp_path)
    after = json.loads((tmp_path / "curriculum_state.json").read_text(encoding="utf-8"))
    assert second.curriculum_restore_status == "restored"
    assert second.episode_index == 1
    assert second.curriculum.last_committed_loadout == decision.selected_loadout
    assert second.curriculum.guarantee_remaining("WallNut") == 3
    assert after["current_loadout"] == before["current_loadout"]
    assert after["loadout_provenance"] == before["loadout_provenance"]


def test_rotated_slot_and_verified_streamer_events_update_only_the_right_seed() -> None:
    curriculum = _curriculum()
    curriculum.record_unlocked(["WallNut", "CherryBomb"], episode_index=1)
    env = object.__new__(AdventureGeneralistTrainingEnv)
    env.curriculum = curriculum
    env.current_loadout = ["SunFlower", "Peashooter", "WallNut", "CherryBomb"]
    env.current_loadout_provenance = curriculum.loadout_provenance(env.current_loadout)
    env.episode_index = 1
    env.context = {}
    env._streamer_v1_last_action = {}

    env._record_seed_usage_from_step(
        {
            "action_source": "MODEL",
            "action_result": {
                "actionDecision": {"intent": {"seed_slot": 2}},
            },
        }
    )
    assert curriculum.model_actions_executed["WallNut"] == 1

    env._streamer_v1_last_action = {"requested_slot": 3}
    env._record_seed_usage_from_step(
        {
            "action_source": "TWITCH",
            "streamer_transition": {
                "execution_succeeded": True,
                "execution_status": "executed_verified",
            },
            "action_result": {
                "actionDecision": {"intent": {"seed_slot": 3}},
            },
        }
    )
    assert curriculum.viewer_actions_executed["CherryBomb"] == 1

    env._streamer_v1_last_action = {"requested_slot": 2}
    env._record_seed_usage_from_step(
        {
            "action_source": "TWITCH",
            "streamer_transition": {
                "execution_succeeded": False,
                "execution_status": "rejected",
            },
            "action_result": {
                "actionDecision": {"intent": {"seed_slot": 2}},
            },
        }
    )
    assert curriculum.viewer_actions_executed["WallNut"] == 0


def test_bc_diagnostics_count_only_a_recorded_verified_viewer_demonstration() -> None:
    curriculum = _curriculum()
    curriculum.record_unlocked(["WallNut"], episode_index=1)
    env = object.__new__(AdventureGeneralistTrainingEnv)
    env.curriculum = curriculum
    env.current_loadout = ["SunFlower", "Peashooter", "WallNut", "SunFlower"]
    env.current_loadout_provenance = curriculum.loadout_provenance(env.current_loadout)
    env.context = {}
    env._streamer_v1_last_action = {
        "requested_slot": 2,
        "viewer_action_id": 42,
        "viewer_observation_revision": "frame-7",
    }

    env.record_streamer_bc_result(
        action_id=42,
        observation_revision="frame-7",
        recorded=True,
    )
    assert curriculum.bc_demonstrations_recorded["WallNut"] == 1

    env.record_streamer_bc_result(
        action_id=43,
        observation_revision="frame-8",
        recorded=False,
        reject_reason="mask_changed",
    )
    assert curriculum.bc_demonstrations_recorded["WallNut"] == 1


def test_curriculum_state_schema_mismatch_fails_closed() -> None:
    with pytest.raises(ValueError, match="schema_version_mismatch"):
        _curriculum().restore_state({"schema_version": CURRICULUM_STATE_SCHEMA_VERSION + 1})
