from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pvzrl_adventure as adventure


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
        "model_family": "ppo_adventure_generalist_14slot_identity_v1",
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
