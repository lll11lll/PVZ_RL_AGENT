"""Parity contracts for the shared leaf-level deadline poller."""

from __future__ import annotations

from typing import Any, Callable

import pytest

import pvzrl_env as env_module
from pvzrl_env import PvZEnvConfig, PvZGymEnv


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(float(seconds))
        self.now += float(seconds)


def _scripted_probe(*values: Any) -> Callable[[], dict[str, Any]]:
    remaining = list(values)

    def probe() -> dict[str, Any]:
        value = remaining.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    return probe


@pytest.fixture
def env() -> PvZGymEnv:
    instance = PvZGymEnv(PvZEnvConfig(reset_poll_seconds=0.2))
    yield instance
    instance.client.close()


def test_startup_popup_wait_preserves_predicate_cadence_and_return_identity(
    env: PvZGymEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    popup = {"screenState": "startup_popup", "startupPopupVisible": True}
    dismissed = {"screenState": "loading_or_menu", "startupPopupVisible": False}
    probe = _scripted_probe(popup, dismissed)
    monkeypatch.setattr(env_module, "time", clock)
    monkeypatch.setattr(env, "adventure_screen_state", probe)

    result = env.wait_for_startup_popup_dismissed(timeout=1.0, poll_seconds=0.01)

    assert result is dismissed
    assert clock.sleeps == [0.05]


def test_startup_popup_timeout_returns_last_state_without_raising(
    env: PvZGymEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    first = {"frameCount": 1, "startupOkButtonVisible": True}
    second = {"frameCount": 2, "startupOkButtonVisible": True}
    probe = _scripted_probe(first, second)
    monkeypatch.setattr(env_module, "time", clock)
    monkeypatch.setattr(env, "adventure_screen_state", probe)

    result = env.wait_for_startup_popup_dismissed(timeout=0.0, poll_seconds=0.05)

    assert result is second
    assert clock.sleeps == [0.05, 0.05]


def test_wait_for_board_retries_same_exceptions_and_does_not_adopt_observation(
    env: PvZGymEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    missing = {"frameCount": 1, "boardFound": False}
    found = {"frameCount": 2, "boardFound": True}
    marker = {"frameCount": -1}
    env.previous_observation = marker
    monkeypatch.setattr(env_module, "time", clock)
    monkeypatch.setattr(
        env,
        "observe",
        _scripted_probe(ConnectionError("bridge starting"), missing, found),
    )

    result = env.wait_for_board(timeout=1.0, poll_seconds=0.1, quiet=True)

    assert result is found
    assert env.previous_observation is marker
    assert clock.sleeps == [0.1, 0.1]


def test_wait_for_board_timeout_keeps_exact_last_observation_text(
    env: PvZGymEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    first = {"frameCount": 1, "boardFound": False}
    second = {"frameCount": 2, "boardFound": False}
    monkeypatch.setattr(env_module, "time", clock)
    monkeypatch.setattr(env, "observe", _scripted_probe(first, second))

    with pytest.raises(TimeoutError) as raised:
        env.wait_for_board(timeout=0.2, poll_seconds=0.1, quiet=True)

    assert str(raised.value) == (
        "Timed out waiting for boardFound=true. Manual path: click the green OK on the "
        "startup popup, click Adventure, select plants, then enter/start the board. "
        "Last observation: {'frameCount': 2, 'boardFound': False}"
    )
    assert clock.sleeps == [0.1, 0.1]


def test_gameplay_ready_wait_preserves_gate_retries_and_observation_adoption(
    env: PvZGymEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    still_selecting = {
        "gameplayReady": True,
        "seedSelectionActive": True,
        "boardFound": True,
    }
    ready = {
        "gameplayReady": True,
        "seedSelectionActive": False,
        "boardFound": True,
    }
    monkeypatch.setattr(env_module, "time", clock)
    monkeypatch.setattr(
        env,
        "observe",
        _scripted_probe(OSError("transient"), still_selecting, ready),
    )

    result = env.wait_for_gameplay_ready(
        timeout=1.0,
        poll_seconds=0.1,
        quiet=True,
    )

    assert result is ready
    assert env.previous_observation is ready
    assert clock.sleeps == [0.1, 0.1]


def test_gameplay_ready_terminal_abort_keeps_exact_error_and_no_sleep(
    env: PvZGymEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    terminal = {
        "boardFound": True,
        "done": True,
        "gameplayReady": False,
        "terminalHint": "game_over_or_loss",
        "wave": 7,
        "maxWave": 10,
    }
    marker = {"frameCount": -1}
    env.previous_observation = marker
    monkeypatch.setattr(env_module, "time", clock)
    monkeypatch.setattr(env, "observe", lambda: terminal)

    with pytest.raises(RuntimeError) as raised:
        env.wait_for_gameplay_ready(
            timeout=1.0,
            poll_seconds=0.1,
            quiet=True,
            fail_on_terminal=True,
        )

    assert str(raised.value) == (
        "Board is present, but it is already in a terminal/end-screen state "
        "(terminalHint=game_over_or_loss, wave=7/10). Start or reset a playable "
        "episode before waiting for gameplayReady."
    )
    assert env.previous_observation is marker
    assert clock.sleeps == []


def test_gameplay_ready_timeout_keeps_exact_last_observation_text(
    env: PvZGymEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    first = {"frameCount": 1, "gameplayReady": False}
    second = {"frameCount": 2, "gameplayReady": False}
    monkeypatch.setattr(env_module, "time", clock)
    monkeypatch.setattr(env, "observe", _scripted_probe(first, second))

    with pytest.raises(TimeoutError) as raised:
        env.wait_for_gameplay_ready(
            timeout=0.2,
            poll_seconds=0.1,
            quiet=True,
            fail_on_terminal=False,
        )

    assert str(raised.value) == (
        "Timed out waiting for gameplayReady=true. Last observation: "
        "{'frameCount': 2, 'gameplayReady': False}"
    )
    assert clock.sleeps == [0.1, 0.1]
