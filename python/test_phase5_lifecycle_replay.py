"""Recorded trace and differential contracts for shared lifecycle authority."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import pytest

from pvzrl_adventure import (
    adventure_gameplay_ready_detected,
    adventure_seed_selection_detected,
)
from pvzrl_env import PvZEnvConfig, PvZGymEnv, classify_done_reason
from pvzrl_lifecycle import (
    LifecycleContext,
    adventure_gameplay_ready_visible,
    adventure_seed_selection_visible,
    classify_lifecycle,
    legacy_done_reason,
    legacy_lifecycle_state,
)


FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "refactor_contracts"
    / "lifecycle_replay_phase5.json"
)


def _fixture() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _observation(defaults: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    observation = copy.deepcopy(defaults)
    observation.update(copy.deepcopy(overlay))
    return observation


@pytest.fixture(scope="module")
def legacy_env() -> PvZGymEnv:
    env = PvZGymEnv(PvZEnvConfig(row_count=5, start_sun=500))
    yield env
    # No bridge was opened; avoid close()'s best-effort restore request.
    env.client.close()


@pytest.mark.parametrize("trace", _fixture()["traces"], ids=lambda trace: trace["name"])
def test_lifecycle_authority_replays_captured_behavior(
    trace: dict[str, Any],
    legacy_env: PvZGymEnv,
) -> None:
    payload = _fixture()
    observation = _observation(payload["defaults"], trace["observation"])
    context = LifecycleContext(**trace.get("context", {}))
    expected = trace["expected"]
    shadow = classify_lifecycle(
        observation,
        context=context,
        fallback_rows=legacy_env.config.row_count,
        start_sun=legacy_env.config.start_sun,
    )
    projection = {
        key: getattr(shadow, key)
        for key in expected
    }
    assert projection == expected
    assert shadow.legacy_state == legacy_env._classify_lifecycle_state(observation)
    assert shadow.done_reason == classify_done_reason(observation)
    assert shadow.adventure_seed_selection_visible is adventure_seed_selection_detected(
        observation
    )
    assert shadow.adventure_gameplay_ready is adventure_gameplay_ready_detected(
        observation
    )


def test_pure_legacy_projection_matches_current_classifier_on_random_valid_signals(
    legacy_env: PvZGymEnv,
) -> None:
    rng = random.Random(20260712)
    screens = (
        "",
        "gameplay",
        "seed_selection",
        "loading_or_menu",
        "level_complete_trophy",
        "reward_unlock",
        "reward_screen",
        "game_over_restart_screen",
    )
    hints = ("", "running", "possible_win", "reward_unlock", "game_over_or_loss")
    for frame in range(1000):
        observation: dict[str, Any] = {
            "frameCount": frame,
            "rowCount": 5,
            "boardFound": rng.choice((False, True)),
            "gameplayReady": rng.choice((False, True)),
            "actualGameplayReady": rng.choice((False, True)),
            "seedSelectionActive": rng.choice((False, True)),
            "seedSelectionPanelActive": rng.choice((False, True)),
            "screenState": rng.choice(screens),
            "terminalHint": rng.choice(hints),
            "done": rng.choice((False, True)),
            "over": rng.choice((False, True)),
            "wave": rng.randrange(0, 3),
            "killCount": rng.randrange(0, 2),
            "plantCount": rng.randrange(0, 3),
            "visiblePlantObjectCount": rng.randrange(0, 3),
            "zombieCount": rng.randrange(0, 3),
            "bulletCount": rng.randrange(0, 2),
            "sun": rng.choice((450, 500)),
            "logicalMowerCount": rng.randrange(3, 6),
            "visibleMowerObjectCount": rng.randrange(3, 6),
            "duplicateMowerRowCount": rng.randrange(0, 2),
            "seedSlots": (
                [{"slotIndex": 0, "plantType": 0}]
                if rng.choice((False, True))
                else []
            ),
            "activeGameplayCardBankCount": rng.randrange(0, 2),
            "legalActions": ([0] if rng.choice((False, True)) else []),
            "nextStep": rng.choice(
                ("", "play", "cleanup_reward_ui", "click_restart")
            ),
            "trophyVisible": rng.choice((False, True)),
            "rewardScreenVisible": rng.choice((False, True)),
            "unlockScreenVisible": rng.choice((False, True)),
            "onGameOverScreen": rng.choice((False, True)),
            "onRestartScreen": rng.choice((False, True)),
            "gameOverTextVisible": rng.choice((False, True)),
            "lossMenuActive": rng.choice((False, True)),
            "blockingRewardUiActive": rng.choice((False, True)),
        }
        assert legacy_lifecycle_state(
            observation,
            fallback_rows=5,
            start_sun=500,
        ) == legacy_env._classify_lifecycle_state(observation)
        assert legacy_done_reason(observation) == classify_done_reason(observation)


def test_base_and_adventure_alias_vocabularies_remain_explicit() -> None:
    observation = {
        "gameplayReady": True,
        "onSeedSelectionScreen": True,
        "screenState": "transition",
    }
    shadow = classify_lifecycle(observation)
    assert shadow.seed_selection_visible is True
    assert shadow.gameplay_ready is False
    assert shadow.adventure_seed_selection_visible is False
    assert shadow.adventure_gameplay_ready is True
    assert adventure_seed_selection_visible(observation) is False
    assert adventure_gameplay_ready_visible(observation) is True


def test_adventure_only_startup_loss_and_menu_aliases_are_preserved() -> None:
    startup = classify_lifecycle({"screenState": "startup_popup"})
    loss = classify_lifecycle({"isGameOverScreen": True})
    menu = classify_lifecycle(
        {"screenState": "adventure_menu", "isAdventureButtonVisible": True}
    )
    assert (startup.phase, startup.adventure_startup_visible) == ("startup", True)
    assert (loss.phase, loss.adventure_loss_visible) == ("loss", True)
    assert (menu.phase, menu.adventure_menu_visible) == ("loading", True)

    post_win_menu = classify_lifecycle(
        {
            "screenState": "main_menu",
            "isAdventureButtonVisible": True,
            "rewardObjectVisible": True,
        }
    )
    assert post_win_menu.adventure_menu_visible is False
    assert post_win_menu.phase == "win"
