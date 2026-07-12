"""Confirmed Phase 5 reset-boundary defect regressions."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from pvzrl_env import RUN_MODE_LEVEL3_SPECIALIST, PvZEnvConfig, PvZGymEnv
from test_refactor_support import load_observation_fixture, make_wrapper


def test_level3_reset_reaches_state_machine_without_obsolete_preflight_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = PvZGymEnv(
        PvZEnvConfig(
            run_mode=RUN_MODE_LEVEL3_SPECIALIST,
            target_level=3,
        )
    )
    playable = {
        "frameCount": 1,
        "rowCount": 5,
        "columnCount": 10,
        "boardFound": True,
        "gameplayReady": True,
        "actualGameplayReady": True,
        "screenState": "gameplay",
        "terminalHint": "running",
    }
    monkeypatch.setattr(env, "configure", lambda: {"ok": True})
    monkeypatch.setattr(
        env,
        "level3_specialist_start_state",
        lambda: pytest.fail("reset duplicated the trainer-owned Level-3 preflight"),
    )
    monkeypatch.setattr(
        env,
        "_reset_state_machine",
        lambda *_args, **_kwargs: dict(playable),
    )
    monkeypatch.setattr(env, "begin_new_attempt", lambda *_args, **_kwargs: None)
    try:
        observation, info = env.reset()
        assert observation == playable
        payload = info["reset"]
        assert payload["resetReason"] == "level3_start"
        assert "level3SpecialistStartState" not in payload
        assert "targetLevel" not in payload
    finally:
        env.client.close()


def test_action_freeze_keeps_episode_label_but_hands_base_a_valid_reset_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = make_wrapper(identity=False)
    wrapper.config.wait_for_board = False
    wrapper.config.wait_gameplay_ready = False
    observation = load_observation_fixture()
    observation["actionCount"] = wrapper.action_count
    wrapper._last_observation = observation
    wrapper.base.previous_observation = observation
    timeout_info = {
        "action_result": {
            "action": 0,
            "requestedAction": 0,
            "executedAction": 0,
            "bridgeTimeout": True,
            "illegalAction": False,
            "plantPlaced": False,
        },
        "reward_breakdown": {
            "reward_total": -10.0,
            "env_corruption_penalty": -10.0,
        },
        "terminal_reason": "action_timeout",
        "done_reason": "action_freeze",
        "needs_reset": True,
        "environment_corruption_detected": True,
        "env_corruption_count": 1,
        "legal_actions": [0],
        "bridge_legal_actions": [0],
    }
    monkeypatch.setattr(
        wrapper.base,
        "step",
        lambda *_args, **_kwargs: (
            observation,
            -10.0,
            True,
            False,
            copy.deepcopy(timeout_info),
        ),
    )
    try:
        _encoded, _reward, terminated, _truncated, info = wrapper.step(0)
        assert terminated is True
        assert info["done_reason"] == "action_freeze"
        assert info["episode_summary"]["result"] == "action_freeze"
        assert wrapper._next_reset_reason == "env_corruption"
        captured: dict[str, Any] = {}
        monkeypatch.setattr(wrapper.base, "configure", lambda: {"ok": True})

        def capture_reset(
            reset_reason: str = "",
            allow_active_gameplay_reset: bool = False,
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            captured.update(
                reset_reason=reset_reason,
                allow_active_gameplay_reset=allow_active_gameplay_reset,
            )
            return observation, {
                "reset": {
                    "ok": True,
                    "resetSuccess": True,
                    "methodUsed": "captured",
                }
            }

        monkeypatch.setattr(wrapper.base, "reset", capture_reset)
        wrapper.reset()
        assert captured == {
            "reset_reason": "env_corruption",
            "allow_active_gameplay_reset": True,
        }
    finally:
        wrapper.base.client.close()
