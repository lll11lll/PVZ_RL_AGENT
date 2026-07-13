"""Regression checks for Adventure attempt-boundary corruption tracking.

These checks avoid the bridge and exercise the Python-side safety detector
directly. They verify that Adventure can start a new board with all mowers
present even if the previous attempt lost mowers, while preserving live-game
respawn detection once the new attempt is armed.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List

from pvzrl_env import RUN_MODE_ADVENTURE_GENERALIST_14SLOT_EVAL, PvZEnvConfig, PvZGymEnv


def observation(
    mower_rows: Iterable[int],
    *,
    frame: int,
    wave: int,
    plants: int,
    zombies: int = 0,
    row_count: int = 5,
) -> Dict[str, Any]:
    rows = set(int(row) for row in mower_rows)
    return {
        "frameCount": frame,
        "boardFound": True,
        "gameplayReady": True,
        "actualGameplayReady": True,
        "seedSelectionActive": False,
        "screenState": "gameplay",
        "nextStep": "play",
        "terminalHint": "running",
        "done": False,
        "over": False,
        "rowCount": row_count,
        "wave": wave,
        "maxWave": 10,
        "plantCount": plants,
        "visiblePlantObjectCount": plants,
        "zombieCount": zombies,
        "bulletCount": 0,
        "killCount": 0,
        "logicalMowerCount": len(rows),
        "visibleMowerObjectCount": len(rows),
        "visibleMowers": [
            {
                "row": row,
                "activeInHierarchy": True,
                "inBoardBounds": True,
                "inMowerArray": True,
            }
            for row in sorted(rows)
        ],
    }


def event_names(diagnostics: Dict[str, Any]) -> List[str]:
    return [str(event.get("event")) for event in diagnostics.get("safety_events", [])]


def main() -> int:
    env = PvZGymEnv(PvZEnvConfig(run_mode=RUN_MODE_ADVENTURE_GENERALIST_14SLOT_EVAL))
    results: List[Dict[str, Any]] = []

    fresh_board = observation(range(5), frame=100, wave=0, plants=0)
    first_action_after_fresh_board = observation(range(5), frame=120, wave=0, plants=1)
    env._episode_lost_mower_rows.update({1, 3, 4})
    env.begin_new_attempt(fresh_board, reason="test_adventure_seed_to_gameplay")
    diagnostics = env._environment_safety_diagnostics(
        fresh_board,
        first_action_after_fresh_board,
        action_result={"plantPlaced": True},
        requested_action=166,
    )
    results.append(
        {
            "case": "fresh adventure board clears stale mower-loss rows",
            "passed": not diagnostics["environment_corruption_detected"]
            and "mower_respawn_detected" not in event_names(diagnostics),
            "events": event_names(diagnostics),
        }
    )

    live_before_loss = observation(range(5), frame=200, wave=3, plants=8, zombies=3)
    live_after_loss = observation([0, 2, 4], frame=220, wave=3, plants=8, zombies=2)
    live_after_respawn = observation(range(5), frame=240, wave=3, plants=9, zombies=2)
    env.begin_new_attempt(live_before_loss, reason="test_live_board")
    env._environment_safety_diagnostics(
        live_before_loss,
        live_after_loss,
        action_result={},
        requested_action=0,
    )
    live_diagnostics = env._environment_safety_diagnostics(
        live_after_loss,
        live_after_respawn,
        action_result={},
        requested_action=0,
    )
    results.append(
        {
            "case": "live mower respawn is still detected",
            "passed": live_diagnostics["environment_corruption_detected"]
            and "mower_respawn_detected" in event_names(live_diagnostics),
            "events": event_names(live_diagnostics),
        }
    )

    payload = {"ok": all(result["passed"] for result in results), "results": results}
    print(json.dumps(payload, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
