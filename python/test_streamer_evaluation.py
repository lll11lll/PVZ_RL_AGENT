from __future__ import annotations

from collections import Counter
from types import SimpleNamespace
from pathlib import Path

import pvzrl_adventure as adventure
from pvzrl_adventure_generalist import ADVENTURE_GENERALIST_MODEL_FAMILY


class _Writer:
    def write(self, _payload):
        return None


class _Base:
    def __init__(self, level: int) -> None:
        self.level = level
        self.wait_calls = 0

    def adventure_screen_state(self):
        return {
            "screenState": "gameplay",
            "isGameplayReady": True,
            "gameplayReady": True,
            "currentAdventureLevel": self.level,
            "profileAdventureLevel": self.level,
        }

    def wait_for_gameplay_ready(self, **_kwargs):
        self.wait_calls += 1
        return {"gameplayReady": True, "currentAdventureLevel": self.level}


def _env(level: int):
    return SimpleNamespace(
        base=_Base(level),
        config=SimpleNamespace(poll_seconds=0.0),
    )


def test_existing_gameplay_is_rejected_before_action_when_level_is_mislabeled(monkeypatch):
    monkeypatch.setattr(adventure, "build_live_status", lambda *_args, **_kwargs: {})
    env = _env(3)
    observation, _reset_info, reason = adventure.prepare_adventure_gameplay(
        env,
        _Writer(),
        {},
        ["SunFlower"],
        timeout=1.0,
        expected_level=2,
    )
    assert observation is None
    assert "adventure_level_identity_unreliable" in reason
    assert env.base.wait_calls == 0


def test_existing_gameplay_is_accepted_when_bridge_profile_and_expected_level_match(monkeypatch):
    monkeypatch.setattr(adventure, "build_live_status", lambda *_args, **_kwargs: {})
    env = _env(3)
    observation, reset_info, reason = adventure.prepare_adventure_gameplay(
        env,
        _Writer(),
        {},
        ["SunFlower"],
        timeout=1.0,
        expected_level=3,
    )
    assert observation is not None
    assert reset_info["reset"]["methodUsed"] == "adventure_existing_gameplay"
    assert reason == ""
    assert env.base.wait_calls == 1


def test_prepare_gameplay_prioritizes_loss_over_stale_gameplay_ready(monkeypatch):
    monkeypatch.setattr(adventure, "build_live_status", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(adventure.time, "sleep", lambda _seconds: None)

    class Base:
        def __init__(self) -> None:
            self.restarted = False
            self.restart_calls = 0

        def adventure_screen_state(self):
            if not self.restarted:
                return {
                    "screenState": "gameplay",
                    "isGameplayReady": True,
                    "isGameOverScreen": True,
                    "restartButtonActive": True,
                    "currentAdventureLevel": 13,
                    "profileAdventureLevel": 13,
                }
            return {
                "screenState": "seed_selection",
                "isGameplayReady": False,
                "isSeedSelectionScreen": True,
                "seedSelectionActive": True,
                "currentAdventureLevel": 13,
                "profileAdventureLevel": 13,
            }

        def click_try_again_once(self):
            self.restart_calls += 1
            self.restarted = True
            return {"ok": True, "methodUsed": "LoseMenuBtn.TryAgain.OnMouseUp"}

        def auto_select_seeds(self, **_kwargs):
            return {
                "ok": True,
                "startInvoked": True,
                "transitionStatus": "gameplay_confirmed",
                "finalObservation": {
                    "screenState": "gameplay",
                    "gameplayReady": True,
                    "isGameplayReady": True,
                    "seedSelectionActive": False,
                    "currentAdventureLevel": 13,
                    "profileAdventureLevel": 13,
                },
            }

    base = Base()
    env = SimpleNamespace(
        base=base,
        config=SimpleNamespace(poll_seconds=0.0, gameplay_ready_timeout=1.0),
    )

    observation, reset_info, reason = adventure.prepare_adventure_gameplay(
        env,
        _Writer(),
        {},
        ["SunFlower"],
        timeout=1.0,
        expected_level=13,
    )

    assert reason == ""
    assert observation is not None and observation["gameplayReady"] is True
    assert reset_info["reset"]["methodUsed"] == "auto_select_seeds"
    assert base.restart_calls == 1


def test_strict_evaluation_navigates_before_requiring_start_level_identity(
    monkeypatch,
    tmp_path: Path,
) -> None:
    events = []

    class Base:
        def __init__(self) -> None:
            self.ready = False

        def configure(self):
            events.append("configure")

        def adventure_screen_state(self):
            if not self.ready:
                return {
                    "screenState": "startup_popup",
                    "startupPopupVisible": True,
                    "startupOkButtonVisible": True,
                    "currentAdventureLevel": -1,
                    "profileAdventureLevel": -1,
                }
            return {
                "screenState": "gameplay",
                "isGameplayReady": True,
                "gameplayReady": True,
                "currentAdventureLevel": 13,
                "profileAdventureLevel": 13,
                "availableSeedNames": ["SunFlower", "Peashooter"],
            }

    class Env:
        def __init__(self, _config):
            self.base = Base()
            self.config = SimpleNamespace(poll_seconds=0.0, gameplay_ready_timeout=1.0)

        def close(self):
            events.append("close")

    class Writer:
        last_payload = {}

        def __init__(self, _path):
            pass

        def write(self, payload, **_kwargs):
            self.last_payload = dict(payload)

    monkeypatch.setattr(adventure, "PvZMaskedPPOEnv", Env)
    monkeypatch.setattr(adventure, "LiveStatusWriter", Writer)
    monkeypatch.setattr(adventure, "build_live_status", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(adventure.time, "sleep", lambda _seconds: None)

    def prepare(env, *_args, **_kwargs):
        events.append("prepare")
        env.base.ready = True
        return (
            {
                "screenState": "gameplay",
                "gameplayReady": True,
                "currentAdventureLevel": 13,
                "profileAdventureLevel": 13,
            },
            {"reset": {"ok": True, "methodUsed": "auto_select_seeds"}},
            "",
        )

    monkeypatch.setattr(adventure, "prepare_adventure_gameplay", prepare)

    def run_attempt(*_args, **_kwargs):
        events.append("attempt")
        return adventure.AdventureAttemptLog(
            attempt=1,
            result="loss",
            done_reason="loss",
            terminal_reason="game_over_restart_screen",
        )

    monkeypatch.setattr(adventure, "run_policy_attempt", run_attempt)
    payload = adventure.run_adventure_eval(
        config={
            "run_dir": str(tmp_path / "evaluation"),
            "seed_list": ["SunFlower", "Peashooter"],
            "model_family": ADVENTURE_GENERALIST_MODEL_FAMILY,
        },
        env_config=SimpleNamespace(max_steps=1),
        model=object(),
        model_path=tmp_path / "model.zip",
        deterministic=True,
        advance_on_wins=1,
        max_adventure_levels=1,
        max_attempts_per_level=1,
        adventure_start_level=13,
        live_status_path=None,
        evaluation_episode_limit=1,
        strict_level_identity=True,
    )

    assert events.index("prepare") < events.index("attempt")
    assert payload["level_identity_initial"]["level_identity_reason"] == "navigation_or_unstable_screen"
    assert payload["level_identity_start"]["level_identity_reliable"] is True
    assert payload["level_identity_start"]["bridge_detected_level"] == 13
    assert payload["level_identity_start_navigation_deferred"] is True


def test_post_win_handoff_waits_through_stale_completed_level_gameplay(monkeypatch):
    monkeypatch.setattr(adventure, "build_live_status", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(adventure.time, "sleep", lambda _seconds: None)

    class Base:
        def __init__(self) -> None:
            self.calls = 0

        def adventure_screen_state(self):
            self.calls += 1
            level = 13 if self.calls <= 2 else 14
            return {
                "screenState": "gameplay",
                "isGameplayReady": True,
                "gameplayReady": True,
                "currentAdventureLevel": level,
                "profileAdventureLevel": level,
            }

    base = Base()
    env = SimpleNamespace(
        base=base,
        config=SimpleNamespace(seed_list=["SunFlower"], poll_seconds=0.0),
    )
    context = {"selected_seeds": ["SunFlower"]}

    state, _unlock_seen, _snapshot, _available, _unknown, blocked, transition = (
        adventure.collect_post_win_unlocks(
            env,
            _Writer(),
            context,
            Counter(),
            13,
        )
    )

    assert blocked == ""
    assert base.calls == 3
    assert state["currentAdventureLevel"] == 14
    assert transition["post_win_transition_completed"] is True
    assert transition["expected_next_adventure_level"] == 14
    assert transition["level_handoff_status"] == "confirmed"
    assert transition["level_identity"]["bridge_detected_level"] == 14


def test_post_win_handoff_rejects_an_unexpected_third_level(monkeypatch):
    monkeypatch.setattr(adventure, "build_live_status", lambda *_args, **_kwargs: {})

    class Base:
        def adventure_screen_state(self):
            return {
                "screenState": "gameplay",
                "isGameplayReady": True,
                "gameplayReady": True,
                "currentAdventureLevel": 15,
                "profileAdventureLevel": 15,
            }

    env = SimpleNamespace(
        base=Base(),
        config=SimpleNamespace(seed_list=["SunFlower"], poll_seconds=0.0),
    )

    _state, _unlock_seen, _snapshot, _available, _unknown, blocked, transition = (
        adventure.collect_post_win_unlocks(
            env,
            _Writer(),
            {"selected_seeds": ["SunFlower"]},
            Counter(),
            13,
        )
    )

    assert "post_win_level_identity_unreliable" in blocked
    assert "expected=14" in blocked
    assert transition["post_win_transition_completed"] is False
    assert transition["level_handoff_status"] == "blocked"


def test_seed_selection_retries_transient_seed_screen_failure(monkeypatch):
    monkeypatch.setattr(adventure, "build_live_status", lambda *_args, **_kwargs: {})

    class Base:
        def __init__(self) -> None:
            self.selection_calls = 0

        def adventure_screen_state(self):
            return {
                "screenState": "seed_selection",
                "isSeedSelectionScreen": True,
                "seedSelectionActive": True,
                "currentAdventureLevel": 1,
                "profileAdventureLevel": 1,
            }

        def auto_select_seeds(self, *, seed_list, start_level):
            assert seed_list == ["SunFlower"]
            assert start_level is True
            self.selection_calls += 1
            if self.selection_calls == 1:
                return {
                    "ok": False,
                    "message": "seed cards are still settling",
                    "after": self.adventure_screen_state(),
                }
            return {"ok": True, "startInvoked": True}

        def wait_for_gameplay_ready(self, **_kwargs):
            return {"gameplayReady": True, "isGameplayReady": True}

    base = Base()
    env = SimpleNamespace(
        base=base,
        config=SimpleNamespace(poll_seconds=0.0, gameplay_ready_timeout=1.0),
    )

    observation, reset_info, reason = adventure.prepare_adventure_gameplay(
        env,
        _Writer(),
        {},
        ["SunFlower"],
        timeout=1.0,
    )

    assert reason == ""
    assert observation["gameplayReady"] is True
    assert reset_info["reset"]["methodUsed"] == "auto_select_seeds"
    assert base.selection_calls == 2


def test_evaluation_summary_counts_only_completed_terminal_episodes() -> None:
    level = adventure.AdventureLevelLog(level=1, advance_on_wins=1)
    level.attempt_logs = [
        {"result": "env_corruption", "episode_reward": 999.0},
        {"result": "blocked", "episode_reward": 999.0},
        {"result": "win", "episode_reward": 10.0},
        {"result": "loss", "episode_reward": -2.0},
        {"result": "timeout", "episode_reward": 1.0},
    ]
    summary = adventure.summarize_progress([level], terminal_episodes_only=True)
    assert summary["episodes_completed"] == 3
    assert summary["win_rate"] == 1 / 3
    assert summary["avg_reward"] == 3.0

    ordinary_summary = adventure.summarize_progress([level])
    assert ordinary_summary["episodes_completed"] == 5
    assert ordinary_summary["avg_reward"] == 401.4


def test_final_capped_timeout_resets_before_streamer_handoff(monkeypatch, tmp_path: Path) -> None:
    events = []

    class Base:
        def configure(self):
            events.append("configure")

        def adventure_screen_state(self):
            return {
                "screenState": "gameplay",
                "isGameplayReady": True,
                "gameplayReady": True,
                "currentAdventureLevel": 1,
                "profileAdventureLevel": 1,
                "availableSeedNames": ["SunFlower", "Peashooter"],
            }

        def reset(self, *, reset_reason: str):
            events.append(f"reset:{reset_reason}")
            return {}, {"reset": {"ok": True, "methodUsed": reset_reason}}

    class Env:
        def __init__(self, _config):
            self.base = Base()
            self.config = SimpleNamespace(poll_seconds=0.0, gameplay_ready_timeout=1.0)

        def close(self):
            events.append("close")

    class Writer:
        last_payload = {}

        def __init__(self, _path):
            pass

        def write(self, payload, **_kwargs):
            self.last_payload = dict(payload)

    monkeypatch.setattr(adventure, "PvZMaskedPPOEnv", Env)
    monkeypatch.setattr(adventure, "LiveStatusWriter", Writer)
    monkeypatch.setattr(adventure, "build_live_status", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(adventure.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        adventure,
        "run_policy_attempt",
        lambda *_args, **_kwargs: adventure.AdventureAttemptLog(
            attempt=1,
            result="timeout",
            done_reason="timeout",
            terminal_reason="timeout_hard_cap",
            timeout_classification="hard_cap_timeout",
        ),
    )

    def prepare_handoff(*_args, **_kwargs):
        assert "reset:timeout" in events
        events.append("handoff")
        return {}, {"reset": {"ok": True}}, ""

    monkeypatch.setattr(adventure, "prepare_adventure_gameplay", prepare_handoff)
    config = {
        "run_dir": str(tmp_path / "evaluation"),
        "seed_list": ["SunFlower", "Peashooter"],
        "model_family": ADVENTURE_GENERALIST_MODEL_FAMILY,
    }
    env_config = SimpleNamespace(max_steps=1)
    payload = adventure.run_adventure_eval(
        config=config,
        env_config=env_config,
        model=object(),
        model_path=tmp_path / "model.zip",
        deterministic=True,
        advance_on_wins=1,
        max_adventure_levels=1,
        max_attempts_per_level=1,
        adventure_start_level=1,
        live_status_path=None,
        adventure_soft_max_steps=1,
        adventure_hard_max_steps=1,
        adventure_final_wave_extension=False,
        evaluation_episode_limit=1,
        strict_level_identity=True,
    )

    assert events.index("reset:timeout") < events.index("handoff")
    assert payload["evaluation_episodes_completed"] == 1
    assert payload["stop_reason"] == "evaluation_episode_limit_reached"


def test_final_capped_loss_recovers_before_streamer_handoff(monkeypatch, tmp_path: Path) -> None:
    events = []

    class Base:
        def configure(self):
            events.append("configure")

        def adventure_screen_state(self):
            return {
                "screenState": "gameplay",
                "isGameplayReady": True,
                "gameplayReady": True,
                "currentAdventureLevel": 1,
                "profileAdventureLevel": 1,
                "availableSeedNames": ["SunFlower", "Peashooter"],
            }

    class Env:
        def __init__(self, _config):
            self.base = Base()
            self.config = SimpleNamespace(poll_seconds=0.0, gameplay_ready_timeout=1.0)

        def close(self):
            events.append("close")

    class Writer:
        last_payload = {}

        def __init__(self, _path):
            pass

        def write(self, payload, **_kwargs):
            self.last_payload = dict(payload)

    monkeypatch.setattr(adventure, "PvZMaskedPPOEnv", Env)
    monkeypatch.setattr(adventure, "LiveStatusWriter", Writer)
    monkeypatch.setattr(adventure, "build_live_status", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(adventure.time, "sleep", lambda _seconds: None)

    def run_attempt(*_args, **_kwargs):
        events.append("attempt:loss")
        return adventure.AdventureAttemptLog(
            attempt=1,
            result="loss",
            done_reason="loss",
            terminal_reason="game_over_restart_screen",
        )

    monkeypatch.setattr(adventure, "run_policy_attempt", run_attempt)

    def prepare(*_args, **_kwargs):
        events.append("prepare")
        return (
            {"gameplayReady": True, "currentAdventureLevel": 1},
            {"reset": {"ok": True, "methodUsed": "auto_select_seeds"}},
            "",
        )

    monkeypatch.setattr(adventure, "prepare_adventure_gameplay", prepare)
    payload = adventure.run_adventure_eval(
        config={
            "run_dir": str(tmp_path / "evaluation"),
            "seed_list": ["SunFlower", "Peashooter"],
            "model_family": ADVENTURE_GENERALIST_MODEL_FAMILY,
        },
        env_config=SimpleNamespace(max_steps=1),
        model=object(),
        model_path=tmp_path / "model.zip",
        deterministic=True,
        advance_on_wins=1,
        max_adventure_levels=1,
        max_attempts_per_level=1,
        adventure_start_level=1,
        live_status_path=None,
        evaluation_episode_limit=1,
        strict_level_identity=True,
    )

    assert events.count("prepare") == 2
    assert events.index("attempt:loss") < events.index("prepare")
    assert payload["evaluation_episodes_completed"] == 1
    assert payload["stop_reason"] == "evaluation_episode_limit_reached"
