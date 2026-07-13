"""Wrapper/base observation ownership regressions for Phase 5."""

from __future__ import annotations

import copy
from typing import Any, Dict

import pytest

from pvzrl_env import RUN_MODE_ADVENTURE_GENERALIST_14SLOT_EVAL
from pvzrl_observation_facts import observation_identity
from test_refactor_support import make_wrapper, observation_for_wrapper


def _step_info(observation: Dict[str, Any], *, done_reason: str = "") -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "action_result": {
            "action": 0,
            "requestedAction": 0,
            "executedAction": 0,
            "illegalAction": False,
            "plantPlaced": False,
            "costPaid": False,
            "cooldownStarted": False,
        },
        "reward_breakdown": {"reward_total": 0.0},
        "legal_actions": list(observation.get("legalActions", [0])),
        "bridge_legal_actions": list(observation.get("legalActions", [0])),
    }
    if done_reason:
        info.update(
            done_reason=done_reason,
            terminal_reason=done_reason,
            needs_reset=True,
        )
    return info


def test_equal_content_observation_copies_are_synchronized() -> None:
    wrapper = make_wrapper()
    observation = observation_for_wrapper(wrapper)
    wrapper._adopt_observation(observation, source="test")
    wrapper.base.previous_observation = copy.deepcopy(observation)
    try:
        mask = wrapper.action_masks()
        assert bool(mask[0]) is True
        assert wrapper._last_observation is not wrapper.base.previous_observation
        assert observation_identity(wrapper._last_observation).token == observation_identity(
            wrapper.base.previous_observation
        ).token
    finally:
        wrapper.base.client.close()


@pytest.mark.parametrize("boundary", ["action_masks", "step"])
def test_divergent_observations_fail_before_policy_or_bridge_work(
    boundary: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = make_wrapper()
    observation = observation_for_wrapper(wrapper)
    wrapper._adopt_observation(observation, source="test")
    divergent = copy.deepcopy(observation)
    divergent["sun"] = int(divergent.get("sun", 0)) + 25
    wrapper.base.previous_observation = divergent
    called = False

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError("bridge work started before synchronization assertion")

    monkeypatch.setattr(wrapper.base, "action_mask", forbidden)
    monkeypatch.setattr(wrapper.base, "step", forbidden)
    try:
        with pytest.raises(RuntimeError, match=f"boundary={boundary}"):
            wrapper.action_masks() if boundary == "action_masks" else wrapper.step(0)
        assert called is False
    finally:
        wrapper.base.client.close()


def test_reset_and_adventure_start_use_one_observation_adoption_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = make_wrapper()
    wrapper.config.wait_for_board = False
    wrapper.config.wait_gameplay_ready = False
    reset_observation = observation_for_wrapper(wrapper)
    reset_observation["frameCount"] = 100
    monkeypatch.setattr(wrapper.base, "configure", lambda: {"ok": True})
    monkeypatch.setattr(
        wrapper.base,
        "reset",
        lambda **_kwargs: (
            reset_observation,
            {"reset": {"ok": True, "resetSuccess": True, "methodUsed": "test_reset"}},
        ),
    )
    try:
        wrapper.reset()
        assert wrapper._last_observation is reset_observation
        assert wrapper.base.previous_observation is reset_observation
        assert wrapper._last_observation_identity == observation_identity(reset_observation)

        wrapper._begin_observation_transition(reason="test_adventure_effect")
        adventure_observation = copy.deepcopy(reset_observation)
        adventure_observation["frameCount"] = 101
        wrapper.start_episode_from_observation(
            adventure_observation,
            {"reset": {"ok": True, "methodUsed": "test_adventure"}},
        )
        assert wrapper.transition_pending is False
        assert wrapper._last_observation is adventure_observation
        assert wrapper.base.previous_observation is adventure_observation
        assert wrapper._last_observation_identity == observation_identity(adventure_observation)
    finally:
        wrapper.base.client.close()


def test_adventure_terminal_transition_blocks_policy_until_fresh_adoption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = make_wrapper()
    wrapper.config.run_mode = RUN_MODE_ADVENTURE_GENERALIST_14SLOT_EVAL
    wrapper.base.config.run_mode = RUN_MODE_ADVENTURE_GENERALIST_14SLOT_EVAL
    observation = observation_for_wrapper(wrapper)
    wrapper._adopt_observation(observation, source="test")
    terminal_observation = copy.deepcopy(observation)
    terminal_observation["frameCount"] = int(observation.get("frameCount", 0)) + 1
    terminal_observation.update(
        done=True,
        over=True,
        gameplayReady=False,
        screenState="game_over",
        terminalHint="loss",
    )
    monkeypatch.setattr(
        wrapper.base,
        "step",
        lambda *_args, **_kwargs: (
            terminal_observation,
            0.0,
            True,
            False,
            _step_info(terminal_observation, done_reason="loss"),
        ),
    )
    try:
        _encoded, _reward, terminated, _truncated, _info = wrapper.step(0)
        assert terminated is True
        assert wrapper._last_observation is terminal_observation
        assert wrapper.base.previous_observation is terminal_observation
        assert wrapper.transition_pending is True

        with pytest.raises(RuntimeError, match="transition_pending.*boundary=action_masks"):
            wrapper.action_masks()
        with pytest.raises(RuntimeError, match="transition_pending.*boundary=step"):
            wrapper.step(0)

        next_observation = copy.deepcopy(observation)
        next_observation["frameCount"] = int(terminal_observation["frameCount"]) + 1
        wrapper.start_episode_from_observation(
            next_observation,
            {"reset": {"ok": True, "methodUsed": "post_transition"}},
        )
        assert wrapper.transition_pending is False
        assert wrapper._last_observation is next_observation
        assert wrapper.base.previous_observation is next_observation
    finally:
        wrapper.base.client.close()
