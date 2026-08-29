"""Focused regressions for Adventure gameplay startup safety and speed."""

from __future__ import annotations

import copy
from typing import Any, Dict, Iterable

from pvzrl_env import (
    RUN_MODE_ADVENTURE_GENERALIST_14SLOT_EVAL,
    PvZEnvConfig,
    PvZGymEnv,
)
from test_refactor_support import load_observation_fixture, make_wrapper


def _with_mowers(observation: Dict[str, Any], mower_rows: Iterable[int]) -> Dict[str, Any]:
    updated = copy.deepcopy(observation)
    rows = sorted({int(row) for row in mower_rows})
    updated["logicalMowerCount"] = len(rows)
    updated["visibleMowerObjectCount"] = len(rows)
    updated["visibleMowers"] = [
        {
            "row": row,
            "activeInHierarchy": True,
            "inBoardBounds": True,
            "inMowerArray": True,
        }
        for row in rows
    ]
    return updated


def _startup_frame(*, frame: int, plant_count: int, mower_rows: Iterable[int]) -> Dict[str, Any]:
    observation = load_observation_fixture()
    observation.update(
        {
            "frameCount": frame,
            "boardFound": True,
            "canReadBoard": True,
            "createPlantFound": True,
            "gameplayReady": True,
            "actualGameplayReady": True,
            "seedSelectionActive": False,
            "screenState": "gameplay",
            "nextStep": "play",
            "terminalHint": "running",
            "done": False,
            "over": False,
            "rowCount": 5,
            "columnCount": 10,
            "wave": 0,
            "maxWave": 10,
            "time": 0.0,
            "plantCount": plant_count,
            # Match the live failure: primary plant data arrives before the
            # optional visible-object diagnostic array is populated.
            "visiblePlantObjectCount": 0,
            "zombieCount": 0,
            "zombies": [],
            "lanes": [{"row": row, "zombieCount": 0} for row in range(5)],
            "sun": 500 if plant_count == 0 else 300,
            "actionCount": 701,
            "legalActions": list(range(201)),
            "legalActionCount": 201,
        }
    )
    observation["plants"] = [] if plant_count == 0 else [
        {
            "row": 3,
            "column": 7,
            "type": 0,
            "typeName": "Peashooter",
            "instanceId": 9001,
            "health": 300,
            "maxHealth": 300,
        }
    ]
    observation["visiblePlants"] = []
    for slot in observation.get("seedSlots", []):
        slot["ready"] = True
        slot["disabled"] = False
        slot["isAvailable"] = True
        slot["usable"] = True
        slot["currentCooldown"] = 0.0
    return _with_mowers(observation, mower_rows)


def _event_names(diagnostics: Dict[str, Any]) -> list[str]:
    return [str(event.get("event")) for event in diagnostics.get("safety_events", [])]


def test_initial_mower_materialization_after_valid_plant_keeps_episode_running() -> None:
    """Reproduce public_alpha_04 frame 5194 -> 5202 exactly enough to step."""

    frame_a = _startup_frame(frame=5194, plant_count=0, mower_rows=())
    frame_b = _startup_frame(frame=5202, plant_count=1, mower_rows=range(5))
    frame_b["seedSlots"][3]["ready"] = False
    frame_b["seedSlots"][3]["currentCooldown"] = 7.5

    class FakeClient:
        def __init__(self) -> None:
            self.requests: list[tuple[str, Dict[str, Any]]] = []

        def request(self, command: str, **payload: Any) -> Dict[str, Any]:
            self.requests.append((command, dict(payload)))
            if command == "step":
                return {
                    "action": 188,
                    "illegalAction": False,
                    "plantPlaced": True,
                    "costPaid": True,
                    "cooldownStarted": True,
                    "observation": copy.deepcopy(frame_b),
                }
            if command == "restore_game_speed":
                return {"ok": True}
            return copy.deepcopy(frame_b)

        def close(self) -> None:
            return None

    wrapper = make_wrapper(fusion_enabled=False)
    base = wrapper.base
    base.config.step_seconds = 0.0
    base.config.seed_screen_check_interval = 10_000
    base._steps_since_seed_screen_check = 0
    base.begin_new_attempt(frame_a, reason="test_initial_mower_materialization")
    base.client = FakeClient()  # type: ignore[assignment]
    try:
        _observation, _reward, terminated, truncated, info = base.step(188)
        events = _event_names(info)
        assert terminated is False
        assert truncated is False
        assert info["env_corruption_count"] == 0
        assert info.get("done_reason") not in {"env_corruption", "action_freeze"}
        assert "mower_respawn_detected" not in events
        assert "board_refresh_detected" not in events
        assert info["action_result"]["plantPlaced"] is True
    finally:
        wrapper.close()


def test_mower_respawn_requires_present_confirmed_absent_then_present() -> None:
    env = PvZGymEnv(PvZEnvConfig(run_mode=RUN_MODE_ADVENTURE_GENERALIST_14SLOT_EVAL))
    full = _startup_frame(frame=6000, plant_count=8, mower_rows=range(5))
    absent_once = _startup_frame(frame=6008, plant_count=8, mower_rows=range(1, 5))
    absent_confirmed = _startup_frame(frame=6016, plant_count=8, mower_rows=range(1, 5))
    respawned = _startup_frame(frame=6024, plant_count=8, mower_rows=range(5))
    env.begin_new_attempt(full, reason="test_confirmed_mower_history")

    first = env._environment_safety_diagnostics(
        full,
        absent_once,
        action_result={},
        requested_action=0,
    )
    second = env._environment_safety_diagnostics(
        absent_once,
        absent_confirmed,
        action_result={},
        requested_action=0,
    )
    third = env._environment_safety_diagnostics(
        absent_confirmed,
        respawned,
        action_result={},
        requested_action=0,
    )

    assert first["environment_corruption_detected"] is False
    assert second["environment_corruption_detected"] is False
    assert third["environment_corruption_detected"] is True
    assert "mower_respawn_detected" in _event_names(third)
    assert "board_refresh_detected" in _event_names(third)


def test_one_incomplete_mower_frame_does_not_create_consumed_history() -> None:
    env = PvZGymEnv(PvZEnvConfig(run_mode=RUN_MODE_ADVENTURE_GENERALIST_14SLOT_EVAL))
    full = _startup_frame(frame=7000, plant_count=8, mower_rows=range(5))
    noisy = _startup_frame(frame=7008, plant_count=8, mower_rows=range(1, 5))
    recovered = _startup_frame(frame=7016, plant_count=8, mower_rows=range(5))
    env.begin_new_attempt(full, reason="test_noisy_mower_frame")

    env._environment_safety_diagnostics(full, noisy, action_result={}, requested_action=0)
    diagnostics = env._environment_safety_diagnostics(
        noisy,
        recovered,
        action_result={},
        requested_action=0,
    )

    assert diagnostics["environment_corruption_detected"] is False
    assert "mower_respawn_detected" not in _event_names(diagnostics)
    assert "board_refresh_detected" not in _event_names(diagnostics)


def test_gameplay_speed_is_reapplied_and_verified_at_ready_boundary(capsys: Any) -> None:
    seed_screen = _startup_frame(frame=7999, plant_count=0, mower_rows=range(5))
    seed_screen.update(
        gameplayReady=False,
        isGameplayReady=False,
        screenState="seed_selection",
        gameSpeed=1.0,
        requestedGameSpeed=4.0,
        gameSpeedMode="game_speed",
        unityTimeScale=1.0,
        effectiveGameSpeed=1.0,
    )
    initial = _startup_frame(frame=8000, plant_count=0, mower_rows=range(5))
    initial.update(
        gameSpeed=4.0,
        requestedGameSpeed=4.0,
        gameSpeedMode="game_speed",
        unityTimeScale=1.0,
        effectiveGameSpeed=1.0,
    )
    corrected = copy.deepcopy(initial)
    corrected.update(
        frameCount=8001,
        gameSpeed=4.0,
        requestedGameSpeed=4.0,
        unityTimeScale=4.0,
        effectiveGameSpeed=4.0,
    )

    class SpeedClient:
        def __init__(self) -> None:
            self.requests: list[str] = []

        def request(self, command: str, **_payload: Any) -> Dict[str, Any]:
            self.requests.append(command)
            if command == "configure":
                return {"gameSpeed": 4.0, "gameSpeedMode": "game_speed"}
            if command == "observe":
                return copy.deepcopy(corrected)
            if command == "restore_game_speed":
                return {"ok": True}
            raise AssertionError(f"unexpected bridge command: {command}")

        def close(self) -> None:
            return None

    env = PvZGymEnv(
        PvZEnvConfig(
            run_mode=RUN_MODE_ADVENTURE_GENERALIST_14SLOT_EVAL,
            game_speed=4.0,
            game_speed_mode="game_speed",
        )
    )
    client = SpeedClient()
    env.client = client  # type: ignore[assignment]
    try:
        assert env.ensure_gameplay_speed(seed_screen) is seed_screen
        assert client.requests == []
        verified = env.ensure_gameplay_speed(initial, timeout=0.1, poll_seconds=0.0)
        assert verified["requestedGameSpeed"] == 4.0
        assert verified["effectiveGameSpeed"] == 4.0
        assert verified["unityTimeScale"] == 4.0
        assert client.requests == ["configure", "observe"]
        output = capsys.readouterr().out
        assert "[game-speed] gameplay entered desired=4.0 effective_before=1.0" in output
        assert "applied=4.0" in output
        assert "effective_after=4.0" in output
        assert "time_scale_after=4.0" in output
        assert "screenState=gameplay" in output
        assert "gameplayReady=True" in output
    finally:
        env.close()


def test_gameplay_speed_check_is_idempotent_when_already_effective() -> None:
    observation = _startup_frame(frame=9000, plant_count=0, mower_rows=range(5))
    observation.update(
        gameSpeed=4.0,
        requestedGameSpeed=4.0,
        gameSpeedMode="game_speed",
        unityTimeScale=4.0,
        effectiveGameSpeed=4.0,
    )

    class NoMutationClient:
        def __init__(self) -> None:
            self.requests: list[str] = []

        def request(self, command: str, **_payload: Any) -> Dict[str, Any]:
            self.requests.append(command)
            if command == "restore_game_speed":
                return {"ok": True}
            raise AssertionError(f"speed was already correct; unexpected command: {command}")

        def close(self) -> None:
            return None

    env = PvZGymEnv(
        PvZEnvConfig(
            run_mode=RUN_MODE_ADVENTURE_GENERALIST_14SLOT_EVAL,
            game_speed=4.0,
            game_speed_mode="game_speed",
        )
    )
    client = NoMutationClient()
    env.client = client  # type: ignore[assignment]
    try:
        assert env.ensure_gameplay_speed(observation) is observation
        assert client.requests == []
    finally:
        # Avoid making the test's cleanup part of the idempotence assertion.
        env.client.close()
