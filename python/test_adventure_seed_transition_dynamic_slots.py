from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

import pvzrl_adventure as adventure
from pvzrl_action_space import (
    ACTION_SPACE_ADVENTURE_14_IDENTITY,
    ADVENTURE_IDENTITY_ACTION_COUNT,
    CELLS_PER_SLOT,
    DEFAULT_COLS,
    build_action_space_spec,
)
from pvzrl_env import (
    PvZEnvConfig,
    PvZGymEnv,
    gameplay_slot_identity_verification,
    resolve_seed_list,
)
from pvzrl_observation_layout import build_observation_layout
from pvzrl_registry import get_plant_registry
from test_adventure_seed_curriculum import _curriculum, _decision, _transaction_env
from test_refactor_support import load_observation_fixture


RUNTIME_NAMES = [
    "SunFlower",
    "SunFlower",
    "Peashooter",
    "WallNut",
    "CherryBomb",
    "PotatoMine",
    "Chomper",
    "SmallPuff",
]
RUNTIME_TYPES = [1, 1, 0, 3, 2, 4, 5, 6]


class _Writer:
    def write(self, _payload):
        return None


def _runtime_slots() -> list[dict]:
    registry = get_plant_registry()
    costs = {
        0: 100,
        1: 50,
        2: 150,
        3: 50,
        4: 25,
        5: 150,
        6: 0,
    }
    return [
        {
            "slotIndex": index,
            "plantType": plant_type,
            "plantTypeName": registry.canonical_name(plant_type),
            "seedCost": costs[plant_type],
            "ready": True,
            "usable": True,
            "disabled": False,
            "isAvailable": True,
            "rawCooldown": 7.5,
            "currentCooldown": 0.0,
            "fullCooldown": 7.5,
            "source": "runtime_card_ui",
        }
        for index, plant_type in enumerate(RUNTIME_TYPES)
    ]


def _gameplay_observation() -> dict:
    observation = copy.deepcopy(load_observation_fixture())
    observation.update(
        {
            "boardFound": True,
            "canReadBoard": True,
            "gameplayReady": True,
            "actualGameplayReady": True,
            "isGameplayReady": True,
            "screenState": "gameplay",
            "seedSelectionActive": False,
            "isSeedSelectionScreen": False,
            "rowCount": 5,
            "columnCount": 10,
            "sun": 999,
            "plants": [],
            "seedSlots": _runtime_slots(),
            "seedSlotCount": len(RUNTIME_TYPES),
            "slotPlantTypes": list(RUNTIME_TYPES),
            "actionCount": ADVENTURE_IDENTITY_ACTION_COUNT,
            "legalActions": [
                0,
                *(
                    1 + slot * CELLS_PER_SLOT + row * DEFAULT_COLS + column
                    for slot in range(len(RUNTIME_TYPES))
                    for row in range(5)
                    for column in range(DEFAULT_COLS)
                ),
            ],
        }
    )
    return observation


def _successful_selection(*, stale_after: bool = False) -> dict:
    final_observation = _gameplay_observation()
    after = (
        {
            "screenState": "seed_selection",
            "isSeedSelectionScreen": True,
            "seedSelectionActive": True,
            "seedSelectionPanelActive": True,
            "startButtonActive": True,
            "activeGameplayCardBankCards": [],
        }
        if stale_after
        else final_observation
    )
    return {
        "ok": True,
        "startInvoked": True,
        "transitionStatus": "gameplay_confirmed",
        "verification": {
            "success": True,
            "selectedSeedTypes": list(reversed(RUNTIME_TYPES)),
        },
        "after": after,
        "afterStart": after,
        "finalObservation": final_observation,
        "gameplaySlotVerification": gameplay_slot_identity_verification(
            final_observation,
            RUNTIME_TYPES,
        ),
    }


def _prepare_env(selection_results: list[dict]):
    class Base:
        def __init__(self) -> None:
            self.selection_calls = 0
            self.wait_calls = 0

        def adventure_screen_state(self):
            return {
                "screenState": "seed_selection",
                "isSeedSelectionScreen": True,
                "seedSelectionActive": True,
            }

        def auto_select_seeds(self, *, seed_list, start_level):
            assert seed_list == RUNTIME_NAMES
            assert start_level is True
            result = selection_results[self.selection_calls]
            self.selection_calls += 1
            return result

        def wait_for_gameplay_ready(self, **_kwargs):
            self.wait_calls += 1
            return _gameplay_observation()

    base = Base()
    return (
        SimpleNamespace(
            base=base,
            config=SimpleNamespace(poll_seconds=0.0, gameplay_ready_timeout=1.0),
        ),
        base,
    )


@pytest.mark.parametrize("stale_after", [False, True])
def test_successful_start_uses_canonical_gameplay_frame_without_duplicate_selection(
    monkeypatch,
    stale_after: bool,
) -> None:
    monkeypatch.setattr(adventure, "build_live_status", lambda *_args, **_kwargs: {})
    env, base = _prepare_env([_successful_selection(stale_after=stale_after)])
    context: dict = {}

    observation, reset_info, reason = adventure.prepare_adventure_gameplay(
        env,
        _Writer(),
        context,
        RUNTIME_NAMES,
        timeout=1.0,
    )

    assert reason == ""
    assert observation["slotPlantTypes"] == RUNTIME_TYPES
    assert reset_info["reset"]["methodUsed"] == "auto_select_seeds"
    assert base.selection_calls == 1
    assert base.wait_calls == 0
    assert context["seed_selection_transition_status"] == "gameplay_confirmed"
    assert context["state"] == "GAMEPLAY_READY"


def test_genuine_interactive_seed_screen_after_transition_timeout_retries_once(monkeypatch) -> None:
    monkeypatch.setattr(adventure, "build_live_status", lambda *_args, **_kwargs: {})
    interactive = {
        "screenState": "seed_selection",
        "isSeedSelectionScreen": True,
        "seedSelectionActive": True,
        "seedSelectionPanelActive": True,
        "startButtonActive": True,
    }
    failed = {
        "ok": False,
        "startInvoked": True,
        "transitionStatus": "seed_selection_still_interactive",
        "after": interactive,
        "message": "canonical gameplay transition timed out",
    }
    env, base = _prepare_env([failed, _successful_selection()])

    observation, _reset_info, reason = adventure.prepare_adventure_gameplay(
        env,
        _Writer(),
        {},
        RUNTIME_NAMES,
        timeout=1.0,
    )

    assert reason == ""
    assert observation["gameplayReady"] is True
    assert base.selection_calls == 2


def test_one_stale_seed_flag_after_start_does_not_authorize_retry(monkeypatch) -> None:
    monkeypatch.setattr(adventure, "build_live_status", lambda *_args, **_kwargs: {})
    failed = {
        "ok": False,
        "startInvoked": True,
        "transitionStatus": "gameplay_transition_timeout",
        "after": {
            "screenState": "seed_selection",
            "isSeedSelectionScreen": True,
            "seedSelectionActive": True,
        },
        "message": "single stale chooser frame",
    }
    env, base = _prepare_env([failed])

    observation, _reset_info, reason = adventure.prepare_adventure_gameplay(
        env,
        _Writer(),
        {},
        RUNTIME_NAMES,
        timeout=1.0,
    )

    assert observation is None
    assert "single stale chooser frame" in reason
    assert base.selection_calls == 1


def test_gameplay_slot_verification_rejects_wrong_order_and_accepts_empty_ui_bank() -> None:
    observation = _gameplay_observation()
    observation["activeGameplayCardBankCards"] = []
    good = gameplay_slot_identity_verification(observation, RUNTIME_TYPES)
    wrong = gameplay_slot_identity_verification(observation, list(reversed(RUNTIME_TYPES)))

    assert good["success"] is True
    assert good["activeSeedTypes"] == RUNTIME_TYPES
    assert wrong["success"] is False
    assert wrong["reason"] == "ordered_runtime_seed_slot_identity_mismatch"


def _seed_selection_probe() -> dict:
    cards = [
        {"plantType": plant_type, "plantTypeName": name}
        for plant_type, name in zip(RUNTIME_TYPES, RUNTIME_NAMES)
    ]
    return {
        "boardFound": True,
        "gameplayReady": False,
        "seedSelectionActive": True,
        "seedSelectionPanelActive": True,
        "startButtonActive": True,
        "blockingRewardUiActive": False,
        "selectedSeedBankCards": cards,
        "selectedBankVisibleCount": len(cards),
        "selectedBankPlantTypeCounts": [],
        "availableSeedCards": [],
        "stalePreselectedCards": [],
        "activeGameplayCardBankCards": [],
    }


def test_auto_select_accepts_canonical_gameplay_slots_despite_later_stale_ui_probe() -> None:
    env = object.__new__(PvZGymEnv)
    env.config = SimpleNamespace(
        seed_list=list(RUNTIME_NAMES),
        plant_types=list(RUNTIME_TYPES),
        lets_rock_delay=0.0,
        post_start_delay=0.0,
        seed_click_delay=0.0,
        reset_wait_timeout=1.0,
        reset_poll_seconds=0.0,
        debug_performance=False,
    )
    selected = _seed_selection_probe()
    stale_after_start = {
        **selected,
        "selectedSeedBankCards": [],
        "selectedBankVisibleCount": 0,
        "activeGameplayCardBankCards": [],
    }
    env.seed_probe = lambda: stale_after_start
    env._wait_for_stable_seed_selection = lambda **_kwargs: (True, selected)
    env.press_lets_rock_once = lambda: {
        "ok": True,
        "startClicked": True,
        "methodUsed": "test",
        "actions": ["lets_rock"],
    }
    env.wait_for_gameplay_ready = lambda **_kwargs: _gameplay_observation()

    result = env.auto_select_seeds(seed_list=RUNTIME_NAMES, start_level=True)

    assert result["ok"] is True
    assert result["startInvoked"] is True
    assert result["transitionStatus"] == "gameplay_confirmed"
    assert result["gameplaySlotVerification"]["success"] is True
    assert result["postTransitionProbe"]["seedSelectionActive"] is True
    assert result["postTransitionProbe"]["activeGameplayCardBankCards"] == []


def test_auto_select_requires_stable_interactive_screen_before_retry_status() -> None:
    env = object.__new__(PvZGymEnv)
    env.config = SimpleNamespace(
        seed_list=list(RUNTIME_NAMES),
        plant_types=list(RUNTIME_TYPES),
        lets_rock_delay=0.0,
        post_start_delay=0.0,
        seed_click_delay=0.0,
        reset_wait_timeout=1.0,
        reset_poll_seconds=0.0,
        debug_performance=False,
    )
    selected = _seed_selection_probe()
    env.seed_probe = lambda: selected
    stable_calls = 0

    def stable_probe(**_kwargs):
        nonlocal stable_calls
        stable_calls += 1
        return (True, selected) if stable_calls <= 2 else (False, selected)

    env._wait_for_stable_seed_selection = stable_probe
    env.press_lets_rock_once = lambda: {"ok": True, "startClicked": True, "actions": []}

    def timeout(**_kwargs):
        raise TimeoutError("transition timed out")

    env.wait_for_gameplay_ready = timeout

    result = env.auto_select_seeds(seed_list=RUNTIME_NAMES, start_level=True)

    assert result["ok"] is False
    assert result["startInvoked"] is True
    assert result["transitionStatus"] == "gameplay_transition_timeout"
    assert stable_calls == 3


def test_curriculum_commit_uses_final_gameplay_slots_not_stale_card_ui_probe() -> None:
    curriculum = _curriculum(capacity=8)
    curriculum.record_unlocked(RUNTIME_NAMES[3:], episode_index=1)
    curriculum.episode_index = 1
    decision = _decision(curriculum, capacity=8)
    expected_types = resolve_seed_list(decision.selected_loadout)
    stale_probe = {
        "seedSelectionActive": True,
        "seedSelectionPanelActive": True,
        "startButtonActive": True,
        "activeGameplayCardBankCards": [],
    }
    env = _transaction_env(curriculum, decision, stale_probe)
    env.effective_seed_capacity = 8
    final_observation = _gameplay_observation()
    final_observation["seedSlots"] = [
        {**slot, "plantType": expected_types[index]}
        for index, slot in enumerate(final_observation["seedSlots"][: len(expected_types)])
    ]
    selection = {
        "ok": True,
        "verification": {
            "success": True,
            "selectedSeedTypes": list(reversed(expected_types)),
        },
        "finalObservation": final_observation,
        "gameplaySlotVerification": gameplay_slot_identity_verification(
            final_observation,
            expected_types,
        ),
    }

    before_remaining = {
        name: curriculum.guarantee_remaining(name)
        for name in decision.guaranteed_seeds
    }
    assert env._commit_pending_seed_selection(selection, decision.selected_loadout) is True
    after_remaining = {
        name: curriculum.guarantee_remaining(name)
        for name in decision.guaranteed_seeds
    }
    assert all(after_remaining[name] == before_remaining[name] - 1 for name in before_remaining)
    assert env._pending_seed_selection is None
    assert env._commit_pending_seed_selection(selection, decision.selected_loadout) is False
    assert {
        name: curriculum.guarantee_remaining(name)
        for name in decision.guaranteed_seeds
    } == after_remaining


def test_eight_runtime_slots_are_not_capped_by_four_slot_startup_metadata() -> None:
    observation = _gameplay_observation()
    env = PvZGymEnv(
        PvZEnvConfig(
            seed_list=["SunFlower", "SunFlower", "Peashooter", "Peashooter"],
            plant_types=[1, 1, 0, 0],
        )
    )
    try:
        mask = env.action_mask(observation)
        diagnostics = env.mask_diagnostics(observation, mask)
        assert len(mask) == ADVENTURE_IDENTITY_ACTION_COUNT == 841
        assert diagnostics["active_seed_slot_count"] == 8
        assert diagnostics["total_legal_action_count"] == 401
        assert diagnostics["wait_legal"] is True
        for slot_index in range(4, 8):
            first_action = 1 + slot_index * CELLS_PER_SLOT
            assert all(mask[first_action : first_action + 50])
            decision = env.action_decision(first_action, observation)
            assert decision.legal is True
            assert decision.intent.seed_slot == slot_index
            assert decision.selected_plant_type == RUNTIME_TYPES[slot_index]
            slot_diag = diagnostics["active_seed_slots"][slot_index]
            assert slot_diag["legal_cells"] == 50
            assert slot_diag["total_legal_actions"] == 50

        smallpuff = diagnostics["active_seed_slots"][7]
        assert smallpuff["plant_identity"] == "SmallPuff"
        assert smallpuff["seed_cost"] == 0
        assert smallpuff["affordable"] is True
        assert smallpuff["usable"] is True
    finally:
        env.close()


def test_tactical_masks_cannot_disable_every_new_runtime_slot() -> None:
    observation = _gameplay_observation()
    env = PvZGymEnv(
        PvZEnvConfig(
            seed_list=["SunFlower", "SunFlower", "Peashooter", "Peashooter"],
            plant_types=[1, 1, 0, 0],
            tactical_masks=True,
        )
    )
    try:
        mask = env.action_mask(observation)
        diagnostics = env.mask_diagnostics(observation, mask)
        for slot_index in (5, 6, 7):
            assert diagnostics["active_seed_slots"][slot_index]["legal_cells"] == 50
        assert diagnostics["total_legal_action_count"] > 1
    finally:
        env.close()


def test_protected_action_and_observation_widths_remain_constant() -> None:
    spec = build_action_space_spec(
        mode=ACTION_SPACE_ADVENTURE_14_IDENTITY,
        plant_types=RUNTIME_TYPES,
        max_seed_slots=14,
        rows=5,
        cols=10,
    )
    assert spec.action_count == 841
    assert build_observation_layout(spec).total_features == 4364
