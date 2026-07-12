from __future__ import annotations

import copy
from dataclasses import replace

from pvzrl_rewards import PendingCherryEvent
from test_refactor_support import make_wrapper, observation_for_wrapper


def _lane_diagnostics_for_cherry_event(event: dict, *, kill_delta: int) -> dict:
    wrapper = make_wrapper(identity=True)
    try:
        previous = observation_for_wrapper(wrapper)
        current = copy.deepcopy(previous)
        current["killCount"] = int(previous.get("killCount", 0)) + int(kill_delta)
        wrapper.base._reward_state = replace(
            wrapper.base._reward_state,
            pending_cherry_events=(
                PendingCherryEvent(
                    row=int(event.get("row", -1)),
                    column=int(event.get("column", -1)),
                    age=int(event.get("age", 0)),
                    kills=int(event.get("kills", 0)),
                    nearby_tough=int(event.get("nearby_tough", 0)),
                    nearby_buckethead=int(event.get("nearby_buckethead", 0)),
                    nearby_conehead=int(event.get("nearby_conehead", 0)),
                    mower_risk=bool(event.get("mower_risk")),
                    credited=bool(event.get("credited")),
                ),
            ),
        )
        reward_events: dict = {}
        wrapper.base.compute_reward_breakdown(
            previous,
            current,
            {"kind": "wait", "plantPlaced": False},
            event_diagnostics=reward_events,
        )
        return wrapper.base.lane_diagnostics(
            previous,
            current,
            {"kind": "wait", "plantPlaced": False},
            [],
            cherry_delayed_diagnostics=reward_events.get("cherry_delayed"),
        )
    finally:
        wrapper.close()


def test_delayed_cherry_kill_and_heavy_credit_reach_lane_diagnostics() -> None:
    diagnostics = _lane_diagnostics_for_cherry_event(
        {
            "age": 0,
            "kills": 0,
            "credited": False,
            "nearby_tough": 1,
            "nearby_buckethead": 1,
            "nearby_conehead": 1,
            "mower_risk": False,
        },
        kill_delta=2,
    )
    assert diagnostics["cherrybomb_delayed_kills"] == 2
    assert diagnostics["cherrybomb_delayed_zero_kill"] == 0
    assert diagnostics["cherrybomb_buckethead_kill_credit"] == 1
    assert diagnostics["cherrybomb_conehead_kill_credit"] == 1


def test_delayed_cherry_zero_kill_reaches_lane_diagnostics() -> None:
    diagnostics = _lane_diagnostics_for_cherry_event(
        {
            "age": 80,
            "kills": 0,
            "credited": False,
            "nearby_tough": 0,
            "nearby_buckethead": 0,
            "nearby_conehead": 0,
            "mower_risk": False,
        },
        kill_delta=0,
    )
    assert diagnostics["cherrybomb_delayed_kills"] == 0
    assert diagnostics["cherrybomb_delayed_zero_kill"] == 1
    assert diagnostics["cherrybomb_buckethead_kill_credit"] == 0
    assert diagnostics["cherrybomb_conehead_kill_credit"] == 0
