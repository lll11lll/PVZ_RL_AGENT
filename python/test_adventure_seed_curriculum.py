"""Focused CORE/guarantee/rotation and transactional seed-selection tests."""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest

from pvzrl_adventure_generalist import (
    ADVENTURE_GENERALIST_INITIAL_LOADOUT,
    AdventureGeneralistTrainingEnv,
    AdventureSeedCurriculum,
)
from pvzrl_env import resolve_seed_list
from train_ppo import build_arg_parser, build_config


def _curriculum(*, capacity: int = 4, guarantee_episodes: int = 4) -> AdventureSeedCurriculum:
    return AdventureSeedCurriculum(
        initial_loadout=ADVENTURE_GENERALIST_INITIAL_LOADOUT,
        max_seed_slots=capacity,
        new_unlock_guarantee_episodes=guarantee_episodes,
        seed_order_source="explicit_config",
    )


def _decision(curriculum: AdventureSeedCurriculum, *, capacity: int | None = None):
    selectable = curriculum.unlocked_seeds()
    return curriculum.choose_loadout(
        selectable_seeds=selectable,
        observed_capacity=capacity or curriculum.max_seed_slots,
        previous_loadout=ADVENTURE_GENERALIST_INITIAL_LOADOUT,
        validation_seeds=selectable,
    )


def test_core_is_unique_after_unlock_but_initial_duplicate_slots_are_preserved() -> None:
    curriculum = _curriculum(capacity=4)
    initial = curriculum.choose_loadout(
        selectable_seeds=["SunFlower", "Peashooter"],
        observed_capacity=4,
        previous_loadout=ADVENTURE_GENERALIST_INITIAL_LOADOUT,
        validation_seeds=["SunFlower", "Peashooter"],
    )
    assert initial.selected_loadout == ADVENTURE_GENERALIST_INITIAL_LOADOUT

    curriculum.record_unlocked(["WallNut"], episode_index=1)
    curriculum.episode_index = 1
    after_unlock = _decision(curriculum)
    assert after_unlock.selected_loadout[:3] == ["SunFlower", "Peashooter", "WallNut"]
    assert after_unlock.selected_loadout.count("SunFlower") + after_unlock.selected_loadout.count("Peashooter") == 3
    assert [row["source"] for row in after_unlock.loadout_provenance[:3]] == [
        "core",
        "core",
        "guaranteed_unlock",
    ]


def test_new_unlock_guarantee_is_transactional_and_completes_after_four_commits() -> None:
    curriculum = _curriculum()
    curriculum.record_unlocked(["WallNut"], episode_index=1)
    curriculum.episode_index = 1

    before = (
        curriculum.guarantee_remaining("WallNut"),
        dict(curriculum.episodes_included),
        curriculum.rotation_cursor,
    )
    proposal = _decision(curriculum)
    repeat = _decision(curriculum)
    assert proposal.selected_loadout == repeat.selected_loadout
    assert proposal.proposed_rotation_cursor == repeat.proposed_rotation_cursor
    assert (
        curriculum.guarantee_remaining("WallNut"),
        dict(curriculum.episodes_included),
        curriculum.rotation_cursor,
    ) == before

    with pytest.raises(RuntimeError, match="curriculum_commit_loadout_mismatch"):
        curriculum.commit_loadout(proposal, selected_loadout=["SunFlower"])
    assert curriculum.guarantee_remaining("WallNut") == 4
    assert curriculum.episodes_included == before[1]

    for expected_remaining in (3, 2, 1, 0):
        proposal = _decision(curriculum)
        assert proposal.guaranteed_seeds == ["WallNut"]
        curriculum.commit_loadout(proposal)
        assert curriculum.guarantee_remaining("WallNut") == expected_remaining
        assert curriculum.episodes_included["WallNut"] == 4 - expected_remaining

    completed = _decision(curriculum)
    assert completed.guaranteed_seeds == []


def test_multiple_pending_guarantees_use_oldest_first_and_defer_on_capacity() -> None:
    curriculum = _curriculum(capacity=3)
    curriculum.record_unlocked(["WallNut", "CherryBomb"], episode_index=1)
    curriculum.episode_index = 1

    proposal = _decision(curriculum, capacity=3)
    assert proposal.selected_loadout == ["SunFlower", "Peashooter", "WallNut"]
    assert proposal.guaranteed_seeds == ["WallNut"]
    assert {
        row["seed"]: row["reason"]
        for row in proposal.excluded_new_plants
        if row["seed"] == "CherryBomb"
    }["CherryBomb"] == "guarantee_capacity_deferred"
    curriculum.commit_loadout(proposal)
    assert curriculum.guarantee_remaining("WallNut") == 3
    assert curriculum.guarantee_remaining("CherryBomb") == 4


def test_rotation_weight_and_order_prioritize_underexposed_seeds_deterministically() -> None:
    first = _curriculum(guarantee_episodes=0)
    first.record_unlocked(["WallNut", "CherryBomb", "PotatoMine"], episode_index=1)
    first.episode_index = 1
    first.episodes_included.update({"WallNut": 5, "CherryBomb": 0, "PotatoMine": 1})

    second = _curriculum(guarantee_episodes=0)
    second.record_unlocked(["WallNut", "CherryBomb", "PotatoMine"], episode_index=1)
    second.episode_index = 1
    second.episodes_included.update({"WallNut": 5, "CherryBomb": 0, "PotatoMine": 1})

    assert first.rotation_weight("CherryBomb") == 1.0
    assert first.rotation_weight("WallNut") == pytest.approx(1.0 / 6.0)
    first_proposal = _decision(first)
    second_proposal = _decision(second)
    assert first_proposal.selected_loadout == second_proposal.selected_loadout
    assert first_proposal.selected_loadout[:4] == [
        "SunFlower",
        "Peashooter",
        "CherryBomb",
        "PotatoMine",
    ]


def test_unselectable_pending_guarantee_is_not_falsely_committed() -> None:
    curriculum = _curriculum()
    curriculum.record_unlocked(["WallNut"], episode_index=1)
    curriculum.episode_index = 1
    proposal = curriculum.choose_loadout(
        selectable_seeds=["SunFlower", "Peashooter"],
        observed_capacity=4,
        previous_loadout=ADVENTURE_GENERALIST_INITIAL_LOADOUT,
        validation_seeds=["SunFlower", "Peashooter"],
    )
    assert "WallNut" not in proposal.selected_loadout
    assert proposal.guaranteed_seeds == []
    assert curriculum.guarantee_remaining("WallNut") == 4


def test_core_and_guarantee_configuration_honor_cli_over_json() -> None:
    args = build_arg_parser().parse_args(
        [
            "--core-seed-names",
            "WallNut,Chomper",
            "--new-unlock-guarantee-episodes",
            "0",
        ]
    )
    config = build_config(
        args,
        {
            "core_seed_names": ["SunFlower", "Peashooter"],
            "new_unlock_guarantee_episodes": 4,
        },
    )
    assert config["core_seed_names"] == ["WallNut", "Chomper"]
    assert config["new_unlock_guarantee_episodes"] == 0


def _transaction_env(curriculum: AdventureSeedCurriculum, decision, probe):
    env = object.__new__(AdventureGeneralistTrainingEnv)
    initial_types = resolve_seed_list(ADVENTURE_GENERALIST_INITIAL_LOADOUT)
    env.curriculum = curriculum
    env.context = {}
    env.configured_seed_list = list(ADVENTURE_GENERALIST_INITIAL_LOADOUT)
    env.current_loadout = list(ADVENTURE_GENERALIST_INITIAL_LOADOUT)
    env.current_loadout_reason = "initial"
    env.current_seed_order_source = "explicit_config"
    env.current_seed_order_preserved = True
    env.current_seed_order_blocked_reason = ""
    env.current_selectable_seeds = ["SunFlower", "Peashooter"]
    env.current_excluded_new_plants = []
    env.rejected_priority_seeds = []
    env.seed_order_source = "explicit_config"
    env.randomize_seed_order = False
    env.effective_seed_capacity = 4
    env.observed_seed_bank_capacity = 4
    env.max_seed_slots = 14
    env.episode_index = 1
    env._episode_slot_identity = tuple(initial_types)
    env.config = SimpleNamespace(seed_list=list(ADVENTURE_GENERALIST_INITIAL_LOADOUT), plant_types=list(initial_types))
    env.base = SimpleNamespace(
        config=SimpleNamespace(seed_list=list(ADVENTURE_GENERALIST_INITIAL_LOADOUT), plant_types=list(initial_types)),
        seed_probe=lambda: probe,
    )
    env._capacity_context_fields = lambda: {}
    env._pending_seed_selection = {
        "decision": decision,
        "previous_loadout": list(ADVENTURE_GENERALIST_INITIAL_LOADOUT),
        "proposed_loadout": list(decision.selected_loadout),
        "episode_index": 1,
        "raw_selectable": list(curriculum.unlocked_seeds()),
        "selection_candidates": list(curriculum.unlocked_seeds()),
        "validation_source": "selectable",
        "available_for_seed_validation": list(curriculum.unlocked_seeds()),
        "available_priority_seeds": [],
    }
    return env


def test_transactional_commit_updates_config_and_curriculum_only_after_canonical_success() -> None:
    curriculum = _curriculum()
    curriculum.record_unlocked(["WallNut"], episode_index=1)
    curriculum.episode_index = 1
    decision = _decision(curriculum)
    expected_types = resolve_seed_list(decision.selected_loadout)
    probe = {
        "activeGameplayCardBankCards": [{"plantType": value} for value in expected_types],
        "seedSelectionActive": False,
    }
    env = _transaction_env(curriculum, decision, probe)
    selection = {
        "ok": True,
        "verification": {"success": True, "selectedSeedTypes": list(expected_types)},
    }

    assert env._commit_pending_seed_selection(selection, decision.selected_loadout) is True
    assert env.current_loadout == decision.selected_loadout
    assert env.config.plant_types == expected_types
    assert env.base.config.plant_types == expected_types
    assert env._pending_seed_selection is None
    assert curriculum.guarantee_remaining("WallNut") == 3
    assert curriculum.episodes_included["WallNut"] == 1


def test_transactional_commit_uses_ordered_runtime_slots_not_raw_card_scan_order() -> None:
    curriculum = _curriculum()
    curriculum.record_unlocked(["WallNut"], episode_index=1)
    curriculum.episode_index = 1
    decision = _decision(curriculum)
    expected_types = resolve_seed_list(decision.selected_loadout)
    probe = {
        # Unity's raw CardUI scan is not a slot-order contract.
        "activeGameplayCardBankCards": [
            {"plantType": value} for value in reversed(expected_types)
        ],
        # The bridge's slot DTO is explicitly ordered by slotIndex.
        "activeGameplaySeedSlots": [
            {"slotIndex": index, "plantType": value}
            for index, value in enumerate(expected_types)
        ],
        "seedSelectionActive": False,
    }
    env = _transaction_env(curriculum, decision, probe)
    selection = {
        "ok": True,
        "verification": {
            "success": True,
            "selectedSeedTypes": list(reversed(expected_types)),
        },
    }

    assert env._commit_pending_seed_selection(selection, decision.selected_loadout) is True
    assert env.current_loadout == decision.selected_loadout
    assert env.config.plant_types == expected_types


def test_transactional_failure_preserves_state_until_explicit_rollback() -> None:
    curriculum = _curriculum()
    curriculum.record_unlocked(["WallNut"], episode_index=1)
    curriculum.episode_index = 1
    decision = _decision(curriculum)
    expected_types = resolve_seed_list(decision.selected_loadout)
    env = _transaction_env(
        curriculum,
        decision,
        {
            "activeGameplayCardBankCards": [{"plantType": expected_types[0]}],
            "seedSelectionActive": False,
        },
    )
    selection = {
        "ok": True,
        "verification": {"success": True, "selectedSeedTypes": list(expected_types)},
    }

    assert env._commit_pending_seed_selection(selection, decision.selected_loadout) is False
    assert env.current_loadout == ADVENTURE_GENERALIST_INITIAL_LOADOUT
    assert env.config.plant_types == resolve_seed_list(ADVENTURE_GENERALIST_INITIAL_LOADOUT)
    assert curriculum.guarantee_remaining("WallNut") == 4
    assert curriculum.episodes_included["WallNut"] == 0
    assert curriculum.rotation_cursor == 0
    assert env._pending_seed_selection is not None

    env._rollback_pending_seed_selection()
    assert env._pending_seed_selection is None
