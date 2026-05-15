"""Bridge-free regression checks for Adventure Eval timeout semantics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import pvzrl_adventure as adventure


@dataclass
class FakeConfig:
    poll_seconds: float = 0.0
    gameplay_ready_timeout: float = 1.0
    plant_types: List[int] = None  # type: ignore[assignment]
    seed_list: List[str] = None  # type: ignore[assignment]
    tactical_masks: bool = True
    wallnut_tactical_mask: bool = True
    cherrybomb_tactical_mask: bool = True

    def __post_init__(self) -> None:
        if self.plant_types is None:
            self.plant_types = [1, 0, 3, 2]
        if self.seed_list is None:
            self.seed_list = ["SunFlower", "Peashooter", "WallNut", "CherryBomb"]


class FakeMask:
    def sum(self) -> int:
        return 1


class FakeModel:
    def predict(self, _obs: Any, deterministic: bool, action_masks: Any) -> Tuple[int, None]:
        return 0, None


class FakeBase:
    def __init__(self, env: "FakeEnv") -> None:
        self.env = env

    def adventure_screen_state(self) -> Dict[str, Any]:
        observation = self.env._last_observation or {}
        screen_state = str(observation.get("screenState") or "gameplay")
        return {
            "screenState": screen_state,
            "isGameplayReady": screen_state == "gameplay",
            "isGameOverScreen": screen_state == "game_over_restart_screen",
            "isLevelComplete": screen_state in {"level_complete_trophy", "reward_unlock", "reward_screen"},
            "trophyVisible": screen_state == "level_complete_trophy",
            "levelCompleteTrophyVisible": screen_state == "level_complete_trophy",
            "rewardScreenVisible": screen_state == "reward_unlock",
            "unlockScreenVisible": False,
            "currentAdventureLevel": 9,
            "currentWorldOrStage": 1,
            "currentDayLevel": 9,
        }


class FakeEnv:
    def __init__(self, scripted_steps: Iterable[Dict[str, Any]]) -> None:
        self.config = FakeConfig()
        self.base = FakeBase(self)
        self._last_observation: Dict[str, Any] = {}
        self._scripted_steps = list(scripted_steps)
        self._index = 0

    def start_episode_from_observation(self, observation: Dict[str, Any], _reset_info: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        self._last_observation = dict(observation)
        return "obs", {}

    def action_masks(self) -> FakeMask:
        return FakeMask()

    def step(self, _action: int) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._index >= len(self._scripted_steps):
            raise AssertionError("FakeEnv exhausted scripted steps")
        step = self._scripted_steps[self._index]
        self._index += 1
        observation = dict(step["observation"])
        self._last_observation = observation
        terminated = bool(step.get("terminated", False))
        truncated = bool(step.get("truncated", False))
        done_reason = str(step.get("done_reason") or ("timeout" if truncated else "none"))
        terminal_reason = str(step.get("terminal_reason") or ("timeout" if truncated else ""))
        summary = {
            "run_mode": "adventure_eval",
            "target_level": 0,
            "episode": 0,
            "result": done_reason,
            "reward_total": float(self._index),
            "done_reason": done_reason,
            "episode_reward": float(self._index),
            "episode_length": self._index,
            "terminal_reason": terminal_reason,
            "final_wave": int(observation.get("wave", 0)),
            "max_wave": int(observation.get("maxWave", 0)),
            "zombies_killed": int(observation.get("killCount", 0)),
            "plants_placed": int(observation.get("plantCount", 0)),
            "mowers_lost": 0,
            "reset_success": True,
            "bridge_errors": 0,
            "illegal_actions": 0,
            "tactical_mask_enabled": True,
            "wallnut_tactical_mask_enabled": True,
            "cherrybomb_tactical_mask_enabled": True,
            "win": done_reason == "win",
            "loss": done_reason == "loss",
            "timeout": done_reason == "timeout",
        }
        info: Dict[str, Any] = {
            "raw_observation": observation,
            "episode_summary_candidate": dict(summary),
            "done_reason": done_reason,
            "terminal_reason": terminal_reason,
        }
        if terminated or truncated:
            info["episode_summary"] = dict(summary)
        return "obs", 1.0, terminated, truncated, info


class FakeWriter:
    def __init__(self) -> None:
        self.last_payload: Dict[str, Any] = {}

    def write(self, payload: Dict[str, Any]) -> None:
        self.last_payload = dict(payload)


def observation(step: int, wave: int, max_wave: int, screen_state: str = "gameplay") -> Dict[str, Any]:
    return {
        "screenState": screen_state,
        "gameplayReady": screen_state == "gameplay",
        "actualGameplayReady": screen_state == "gameplay",
        "terminalHint": "running",
        "done": False,
        "over": False,
        "wave": wave,
        "maxWave": max_wave,
        "killCount": step,
        "plantCount": step,
        "zombieCount": 1,
    }


def run_attempt(scripted_steps: List[Dict[str, Any]], *, soft: int = 3, hard: int = 5) -> adventure.AdventureAttemptLog:
    original_prepare = adventure.prepare_adventure_gameplay
    original_live = adventure.build_live_status
    try:
        adventure.prepare_adventure_gameplay = lambda *_args, **_kwargs: (observation(0, 0, 3), {"reset": {"ok": True}}, "")
        adventure.build_live_status = lambda *_args, **_kwargs: {}
        return adventure.run_policy_attempt(
            FakeEnv(scripted_steps),
            FakeModel(),
            FakeWriter(),  # type: ignore[arg-type]
            {
                "soft_max_steps": soft,
                "hard_max_steps": hard,
                "final_wave_extension_enabled": True,
            },
            attempt_index=1,
            selected_seeds=["SunFlower", "Peashooter", "WallNut", "CherryBomb"],
            deterministic=True,
            tracker_level=9,
            progression_index=7,
            soft_max_steps=soft,
            hard_max_steps=hard,
            final_wave_extension=True,
        )
    finally:
        adventure.prepare_adventure_gameplay = original_prepare
        adventure.build_live_status = original_live


def main() -> int:
    cases: List[Dict[str, Any]] = []

    soft_extended_win = run_attempt(
        [
            {"observation": observation(1, 1, 3)},
            {"observation": observation(2, 3, 3)},
            {"observation": observation(3, 3, 3)},
            {"observation": observation(4, 3, 3, "level_complete_trophy"), "terminated": True, "done_reason": "win", "terminal_reason": "win"},
        ]
    )
    cases.append(
        {
            "case": "soft final-wave extension then win",
            "passed": soft_extended_win.result == "win"
            and soft_extended_win.soft_timeout_reached
            and soft_extended_win.soft_timeout_extended
            and soft_extended_win.timeout_classification == "soft_extended_then_win",
            "log": {
                "result": soft_extended_win.result,
                "soft_timeout_reached": soft_extended_win.soft_timeout_reached,
                "soft_timeout_extended": soft_extended_win.soft_timeout_extended,
                "timeout_classification": soft_extended_win.timeout_classification,
                "steps_after_soft_timeout": soft_extended_win.steps_after_soft_timeout,
            },
        }
    )

    soft_no_extension = run_attempt(
        [
            {"observation": observation(1, 1, 3)},
            {"observation": observation(2, 2, 3)},
            {"observation": observation(3, 2, 3)},
            {"observation": observation(4, 3, 3)},
        ]
    )
    cases.append(
        {
            "case": "soft cap no extension before final wave",
            "passed": soft_no_extension.result == "timeout"
            and soft_no_extension.timeout_classification == "soft_cap_timeout_no_extension"
            and soft_no_extension.terminal_reason == "timeout_soft_cap_no_extension",
            "log": {
                "result": soft_no_extension.result,
                "terminal_reason": soft_no_extension.terminal_reason,
                "timeout_classification": soft_no_extension.timeout_classification,
            },
        }
    )

    hard_cap = run_attempt(
        [
            {"observation": observation(1, 1, 3)},
            {"observation": observation(2, 3, 3)},
            {"observation": observation(3, 3, 3)},
            {"observation": observation(4, 3, 3)},
            {"observation": observation(5, 3, 3), "truncated": True, "done_reason": "timeout", "terminal_reason": "timeout"},
        ]
    )
    cases.append(
        {
            "case": "soft extension then hard cap",
            "passed": hard_cap.result == "timeout"
            and hard_cap.soft_timeout_extended
            and hard_cap.timeout_classification == "hard_cap_timeout"
            and hard_cap.terminal_reason == "timeout_hard_cap"
            and adventure.adventure_stop_reason(hard_cap.blocked_reason) == "timeout_hard_cap",
            "log": {
                "result": hard_cap.result,
                "terminal_reason": hard_cap.terminal_reason,
                "blocked_reason": hard_cap.blocked_reason,
                "timeout_classification": hard_cap.timeout_classification,
            },
        }
    )

    label = adventure.adventure_level_metadata(9, 7)
    cases.append(
        {
            "case": "level label helper",
            "passed": label["adventure_level_label"] == "1-9"
            and label["adventure_world"] == 1
            and label["adventure_stage"] == 9
            and label["progression_index"] == 7,
            "log": label,
        }
    )
    cases.append(
        {
            "case": "timeout stop reasons are not unhandled_screen",
            "passed": adventure.adventure_stop_reason("timeout_hard_cap") == "timeout_hard_cap"
            and adventure.adventure_stop_reason("timeout_soft_cap_no_extension") == "timeout_soft_cap_no_extension"
            and adventure.adventure_stop_reason("timeout") == "timeout_hard_cap",
            "log": {
                "timeout_hard_cap": adventure.adventure_stop_reason("timeout_hard_cap"),
                "timeout_soft_cap_no_extension": adventure.adventure_stop_reason("timeout_soft_cap_no_extension"),
                "timeout": adventure.adventure_stop_reason("timeout"),
            },
        }
    )

    payload = {"ok": all(case["passed"] for case in cases), "results": cases}
    print(json.dumps(payload, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
