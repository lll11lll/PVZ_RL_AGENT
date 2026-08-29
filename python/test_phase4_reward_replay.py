"""Captured Phase 4 reward replay and hidden-state compatibility contracts.

The JSON fixture was recorded against commit ``a147e93`` before the Phase 4
runtime extraction.  It intentionally locks both public component values and
the private state transitions that influence rewards on later steps.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import pvzrl_env
import pvzrl_rewards

from pvzrl_env import (
    REWARD_COMPONENT_FIELDS,
    PvZEnvConfig,
    PvZGymEnv,
    RewardConfig,
)
from pvzrl_rewards import PendingCherryEvent
from pvzrl_observation_facts import build_step_facts


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "refactor_contracts"
REPLAY_FIXTURE = FIXTURE_DIR / "reward_replay_phase4.json"


def _load_replay() -> dict[str, Any]:
    return json.loads(REPLAY_FIXTURE.read_text(encoding="utf-8"))


def _load_base_observation(replay: dict[str, Any]) -> dict[str, Any]:
    path = FIXTURE_DIR / str(replay["base_observation_fixture"])
    return json.loads(path.read_text(encoding="utf-8"))


def _refresh_observation_counts(observation: dict[str, Any]) -> None:
    plants = observation.get("plants") or []
    zombies = observation.get("zombies") or []
    observation["plantCount"] = len(plants)
    observation["visiblePlantObjectCount"] = len(plants)
    observation["totalPlantHealth"] = sum(int(plant.get("health", 0) or 0) for plant in plants)
    observation["zombieCount"] = len(zombies)


def _apply_observation_spec(observation: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(observation)
    if "zombies_replace" in spec:
        updated["zombies"] = copy.deepcopy(spec["zombies_replace"])
    if spec.get("plants_add"):
        updated.setdefault("plants", []).extend(copy.deepcopy(spec["plants_add"]))
    for raw_row, lane_update in (spec.get("lane_updates") or {}).items():
        row = int(raw_row)
        lane = next(item for item in updated.get("lanes", []) if int(item.get("row", -1)) == row)
        lane.update(copy.deepcopy(lane_update))
    if spec.get("legal_actions_add"):
        legal = set(int(value) for value in updated.get("legalActions", []) or [])
        legal.update(int(value) for value in spec["legal_actions_add"])
        updated["legalActions"] = sorted(legal)
        updated["legalActionCount"] = len(legal)

    _refresh_observation_counts(updated)
    for key, value in (spec.get("scalar_set") or {}).items():
        updated[str(key)] = copy.deepcopy(value)
    for key, delta in (spec.get("scalar_deltas") or {}).items():
        updated[str(key)] = updated.get(str(key), 0) + delta
    return updated


def _make_env(case: dict[str, Any]) -> PvZGymEnv:
    reward = RewardConfig()
    for field_name, value in (case.get("reward_overrides") or {}).items():
        setattr(reward, str(field_name), float(value))
    env = PvZGymEnv(
        PvZEnvConfig(
            plant_types=[1, 1, 0, 0],
            fusion_policy="observe",
            fusion_action_mask_enabled=True,
            reward=reward,
        )
    )
    state = case.get("state_before") or {}
    updates: dict[str, Any] = {
        "pending_cherry_events": tuple(
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
            )
            for event in state.get("pending_cherry_events") or []
        )
    }
    for field_name in (
        "undefended_threat_age_by_row",
        "max_undefended_threat_age_by_row",
        "undefended_threat_age_sum_by_row",
        "undefended_threat_age_count_by_row",
    ):
        if field_name in state:
            updates[field_name] = tuple(int(value) for value in state[field_name])
    if "all_rows_peashooter_coverage_rewarded" in state:
        updates["all_rows_peashooter_coverage_rewarded"] = bool(
            state["all_rows_peashooter_coverage_rewarded"]
        )
    if "all_active_threatened_rows_coverage_rewarded" in state:
        updates["all_active_threatened_rows_coverage_rewarded"] = bool(
            state["all_active_threatened_rows_coverage_rewarded"]
        )
    env._reward_state = replace(env._reward_state, **updates)
    return env


def _reward_state_projection(env: PvZGymEnv) -> dict[str, Any]:
    state = env._reward_state
    return {
        "undefended_threat_age_by_row": list(state.undefended_threat_age_by_row),
        "max_undefended_threat_age_by_row": list(state.max_undefended_threat_age_by_row),
        "undefended_threat_age_sum_by_row": list(state.undefended_threat_age_sum_by_row),
        "undefended_threat_age_count_by_row": list(state.undefended_threat_age_count_by_row),
        "pending_cherry_events": [
            {
                "row": event.row,
                "column": event.column,
                "age": event.age,
                "kills": event.kills,
                "nearby_tough": event.nearby_tough,
                "nearby_buckethead": event.nearby_buckethead,
                "nearby_conehead": event.nearby_conehead,
                "mower_risk": event.mower_risk,
                "credited": event.credited,
            }
            for event in state.pending_cherry_events
        ],
        "all_rows_peashooter_coverage_rewarded": bool(state.all_rows_peashooter_coverage_rewarded),
        "all_active_threatened_rows_coverage_rewarded": bool(
            state.all_active_threatened_rows_coverage_rewarded
        ),
    }


def _assert_float_mapping(
    actual: dict[str, Any],
    expected: dict[str, Any],
    *,
    tolerance: float,
) -> None:
    assert set(actual) == set(expected)
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, bool) or isinstance(expected_value, str):
            assert actual_value == expected_value, key
        else:
            assert float(actual_value) == pytest.approx(float(expected_value), abs=tolerance, rel=0.0), key


def test_captured_environment_reward_replay() -> None:
    replay = _load_replay()
    tolerance = float(replay["tolerance"])
    assert replay["captured_from_commit"] == "a147e93"
    # The fixture remains an archival snapshot of the removed tactical policy.
    # Its old public keys stay present, while V2 adds threat_delta_reward.
    assert set(replay["component_fields"]).issubset(set(REWARD_COMPONENT_FIELDS))

    base = _load_base_observation(replay)
    active_v2_fields = {
        "kill_reward",
        "wave_reward",
        "win_loss_reward",
        "illegal_penalty",
        "mower_loss_penalty",
        "threat_delta_reward",
        "fusion_reward",
    }
    for case in replay["environment_cases"]:
        env = _make_env(case)
        previous = _apply_observation_spec(base, case.get("previous") or {})
        current = _apply_observation_spec(previous, case.get("current") or {})
        events: dict[str, Any] = {}
        breakdown = env.compute_reward_breakdown(
            previous,
            current,
            copy.deepcopy(case["action_result"]),
            event_diagnostics=events,
        )

        assert float(breakdown["reward_total"]) == pytest.approx(
            sum(float(breakdown[field]) for field in REWARD_COMPONENT_FIELDS),
            abs=tolerance,
            rel=0.0,
        )
        assert all(
            float(breakdown[field]) == pytest.approx(0.0, abs=tolerance, rel=0.0)
            for field in REWARD_COMPONENT_FIELDS
            if field not in active_v2_fields
        )
        assert events["reward_policy_version"] == "generalized_threat_v2"


def _fusion_observation(*, threatened: bool) -> dict[str, Any]:
    return {
        "rowCount": 5,
        "columnCount": 10,
        "gameplayReady": True,
        "boardFound": True,
        "sun": 500,
        "seedSlots": [
            {"slotIndex": 0, "plantType": 1, "plantTypeName": "SunFlower", "ready": True, "usable": True},
            {"slotIndex": 1, "plantType": 0, "plantTypeName": "Peashooter", "ready": True, "usable": True},
        ],
        "plants": [{"row": 2, "column": 4, "type": 1, "typeName": "SunFlower"}],
        "zombies": ([{"row": 2, "alive": True}] if threatened else []),
        "lanes": ([{"row": 2, "zombieCount": 1, "danger": 0.8}] if threatened else []),
    }


def _fusion_action_result(step: dict[str, Any]) -> dict[str, Any]:
    success = bool(step["success"])
    reason = str(step.get("reason") or "")
    legal = bool(step["legal"])
    column = int(step["column"])
    candidate = {
        "source_plant_type": 1,
        "source_plant_name": "SunFlower",
        "source_row": 2,
        "source_col": column,
        "target_or_ingredient_type": 0,
        "target_or_ingredient_name": "Peashooter",
        "ingredient_seed_slot_index": 1,
        "fusion_legal": legal,
        "fusion_blocked_reason": "" if legal else "incompatible_pair",
    }
    result = {
        "fusionAttempted": True,
        "fusionSucceeded": success,
        "fusion_success": success,
        "fusionRejectedReason": reason,
        "illegalReason": reason or None,
        "changedTileCount": 1 if success else 0,
        "bridgeMethodUsed": "test_bridge",
        "fusionExecutionSource": "model_action_mask",
        "fusionCandidate": candidate,
        "decoded": {
            "kind": "fusion",
            "sourcePlantType": 1,
            "ingredientPlantType": 0,
            "row": 2,
            "column": column,
        },
    }
    if step.get("event_id"):
        result["fusionEventId"] = str(step["event_id"])
    return result


def test_captured_fusion_reward_sequences() -> None:
    replay = _load_replay()
    tolerance = float(replay["tolerance"])
    expected_by_case = {
        "success_context": [0.15],
        "incompatible_repeat": [0.0, 0.0],
        "cap_then_penalty": [0.15, 0.15, 0.0],
        "duplicate_event": [0.15, 0.0],
    }
    for case in replay["fusion_sequences"]:
        env = PvZGymEnv(
            PvZEnvConfig(
                fusion_policy="observe",
                reward=RewardConfig(max_fusion_reward_per_episode=float(case["max_positive_reward"])),
            )
        )
        deltas = [
            env._compose_step_reward(
                _fusion_observation(threatened=bool(step["threatened"])),
                _fusion_observation(threatened=bool(step["threatened"])),
                _fusion_action_result(step),
                previous_legal_actions=[],
            ).breakdown.component("fusion_reward")
            for step in case["steps"]
        ]
        assert deltas == pytest.approx(
            expected_by_case[case["name"]], abs=tolerance, rel=0.0
        )
        live = env._fusion_reward_live_fields()
        assert live["fusion_reward_total"] == pytest.approx(sum(deltas), abs=tolerance)
        assert live["fusion_success_reward_total"] == pytest.approx(sum(deltas), abs=tolerance)
        for field in (
            "fusion_attempt_reward_total",
            "fusion_new_recipe_reward_total",
            "fusion_recursive_reward_total",
            "fusion_tier_reward_total",
            "fusion_repeat_decay_total",
            "fusion_threatened_row_bonus_total",
            "fusion_active_wave_bonus_total",
            "fusion_defensive_value_bonus_total",
            "fusion_incompatible_penalty_total",
            "fusion_empty_tile_penalty_total",
            "fusion_failed_penalty_total",
            "fusion_bridge_error_penalty_total",
            "fusion_spam_penalty_total",
        ):
            assert live[field] == 0.0


def test_step_compositor_reuses_supplied_facts_for_fusion_usefulness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = _fusion_observation(threatened=True)
    facts = build_step_facts(observation)
    result = _fusion_action_result(
        {
            "success": True,
            "legal": True,
            "reason": "",
            "column": 4,
            "event_id": "facts-reuse-fusion",
        }
    )
    env = PvZGymEnv(PvZEnvConfig(fusion_policy="observe"))

    def fail_rebuild(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("step reward rebuilt supplied observation facts")

    monkeypatch.setattr(pvzrl_rewards, "build_step_facts", fail_rebuild)
    composition = env._compose_step_reward(
        observation,
        observation,
        result,
        previous_facts=facts,
        current_facts=facts,
        previous_legal_actions=[],
    )
    assert composition.breakdown.component("fusion_reward") > 0.0


class _RewardStepClient:
    def __init__(
        self,
        observation: dict[str, Any],
        action_result: dict[str, Any] | None = None,
    ) -> None:
        self.observation = copy.deepcopy(observation)
        self.action_result = copy.deepcopy(action_result or {})

    def request(self, command: str, **payload: Any) -> dict[str, Any]:
        if command == "step":
            result = {
                "action": int(payload.get("action", 0)),
                "illegalAction": False,
                "illegalReason": None,
                "plantPlaced": False,
                "costPaid": False,
                "cooldownStarted": False,
                "observation": copy.deepcopy(self.observation),
            }
            result.update(copy.deepcopy(self.action_result))
            result["observation"] = copy.deepcopy(self.observation)
            return result
        if command == "restore_game_speed":
            return {"ok": True}
        return copy.deepcopy(self.observation)

    def close(self) -> None:
        return None


def _terminal_step_env(
    previous: dict[str, Any],
    current: dict[str, Any],
    *,
    action_result: dict[str, Any] | None = None,
    fusion_policy: str = "none",
) -> PvZGymEnv:
    env = PvZGymEnv(
        PvZEnvConfig(plant_types=[1, 1, 0, 0], fusion_policy=fusion_policy)
    )
    env.previous_observation = copy.deepcopy(previous)
    env.config.step_seconds = 0.0
    env.config.seed_screen_check_interval = 10_000
    env._steps_since_seed_screen_check = 0
    env.client = _RewardStepClient(current, action_result)  # type: ignore[assignment]
    return env


def test_forced_seed_probe_keeps_reward_previous_facts_and_legal_actions_paired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = _load_replay()
    previous = _load_base_observation(replay)
    for lane in previous["lanes"]:
        lane["danger"] = 0.0
    previous["lanes"][1]["danger"] = 0.1
    previous["terminalHint"] = "running"
    previous["done"] = False
    previous["legalActions"] = sorted(
        {int(value) for value in previous.get("legalActions", [])} | {133}
    )
    previous["legalActionCount"] = len(previous["legalActions"])

    probe = copy.deepcopy(previous)
    probe["frameCount"] = int(previous.get("frameCount", 0)) + 1
    probe["lanes"][1]["danger"] = 0.8
    probe["legalActions"] = [0]
    probe["legalActionCount"] = 1
    current = copy.deepcopy(probe)
    current["frameCount"] += 1

    env = _terminal_step_env(previous, current)
    env.config.seed_screen_check_interval = 1
    env._steps_since_seed_screen_check = 1
    env._step_facts_cache.get(
        env.previous_observation,
        env.config.plant_types,
    )

    def forced_observe(*, force_seed_probe: bool = False) -> dict[str, Any]:
        assert force_seed_probe is True
        return copy.deepcopy(probe)

    monkeypatch.setattr(env, "observe", forced_observe)
    monkeypatch.setattr(
        pvzrl_env,
        "build_step_facts",
        lambda *_args, **_kwargs: pytest.fail(
            "forced probe rebuilt the cached reward-previous frame"
        ),
    )
    try:
        _next, _reward, _done, _truncated, info = env.step(0)
        breakdown = info["reward_breakdown"]
        lanes = info["lane_diagnostics"]
        assert breakdown["danger_delta_reward"] == 0.0
        assert breakdown["threat_delta_reward"] == 0.0
        assert lanes["previous_total_danger"] == pytest.approx(0.1, abs=1e-12)
        assert lanes["current_total_danger"] == pytest.approx(0.8, abs=1e-12)
        assert sum(lanes["pre_action_legal_peashooter_actions_by_row"].values()) > 0
        assert sum(lanes["legal_peashooter_actions_by_row"].values()) == 0
    finally:
        env.close()


def test_transient_possible_win_is_nonterminal_and_not_rewarded() -> None:
    replay = _load_replay()
    previous = _load_base_observation(replay)
    previous["terminalHint"] = "running"
    previous["done"] = False
    current = copy.deepcopy(previous)
    current.update({"done": True, "over": False, "terminalHint": "possible_win"})
    env = _terminal_step_env(previous, current)
    try:
        _observation, reward, done, _truncated, info = env.step(0)
        breakdown = info["reward_breakdown"]
        assert done is False
        assert info["done_reason"] == "none"
        assert breakdown["win_loss_reward"] == 0.0
        assert reward == pytest.approx(
            sum(float(breakdown[field]) for field in REWARD_COMPONENT_FIELDS),
            abs=1e-9,
            rel=0.0,
        )
    finally:
        env.close()


def test_board_refresh_win_is_rewarded_exactly_once() -> None:
    replay = _load_replay()
    previous = _load_base_observation(replay)
    previous["terminalHint"] = "running"
    previous["done"] = False
    current = copy.deepcopy(previous)
    current.update({"done": True, "over": False, "terminalHint": "possible_win"})
    env = _terminal_step_env(previous, current)
    env._environment_safety_diagnostics = lambda *_args, **_kwargs: {  # type: ignore[method-assign]
        "environment_corruption_detected": True,
        "environment_corruption_penalty": 10.0,
        "env_corruption_count": 1,
        "safety_events": [{"event": "board_refresh_detected"}],
    }
    try:
        _observation, reward, done, _truncated, info = env.step(0)
        breakdown = info["reward_breakdown"]
        assert done is True
        assert info["done_reason"] == "win"
        assert breakdown["win_loss_reward"] == env.config.reward.win_reward
        assert reward == pytest.approx(
            sum(float(breakdown[field]) for field in REWARD_COMPONENT_FIELDS),
            abs=1e-9,
            rel=0.0,
        )
        assert reward < env.config.reward.win_reward * 2.0
    finally:
        env.close()


@pytest.mark.parametrize("terminal_hint", ["reward_unlock", "", "running"])
def test_confirmed_reward_screen_win_is_rewarded_for_all_bridge_hints(
    terminal_hint: str,
) -> None:
    replay = _load_replay()
    previous = _load_base_observation(replay)
    previous.update({"terminalHint": "running", "done": False})
    current = copy.deepcopy(previous)
    current.update(
        {
            "done": True,
            "over": False,
            "terminalHint": terminal_hint,
            "screenState": "reward_unlock",
            "rewardScreenVisible": True,
        }
    )
    env = _terminal_step_env(previous, current)
    env._environment_safety_diagnostics = lambda *_args, **_kwargs: {  # type: ignore[method-assign]
        "environment_corruption_detected": False,
        "environment_corruption_penalty": 0.0,
        "env_corruption_count": 0,
        "safety_events": [],
    }
    try:
        _observation, reward, done, _truncated, info = env.step(0)
        breakdown = info["reward_breakdown"]
        assert done is True
        assert info["done_reason"] == "win"
        assert breakdown["win_loss_reward"] == env.config.reward.win_reward
        assert reward == pytest.approx(
            sum(float(breakdown[field]) for field in REWARD_COMPONENT_FIELDS),
            abs=1e-9,
            rel=0.0,
        )
    finally:
        env.close()


@pytest.mark.parametrize("terminal_hint", ["game_over_or_loss", "", "running"])
def test_confirmed_restart_screen_loss_is_penalized_for_all_bridge_hints(
    terminal_hint: str,
) -> None:
    replay = _load_replay()
    terminal = _load_base_observation(replay)
    terminal.update(
        {
            "done": True,
            "over": True,
            "terminalHint": terminal_hint,
            "screenState": "game_over_restart_screen",
            "onGameOverScreen": True,
            "gameOverTextVisible": True,
            "gameplayReady": False,
        }
    )
    env = _terminal_step_env(terminal, terminal)
    try:
        _observation, reward, done, _truncated, info = env.step(0)
        breakdown = info["reward_breakdown"]
        assert done is True
        assert breakdown["win_loss_reward"] == -env.config.reward.loss_penalty
        assert reward == pytest.approx(
            sum(float(breakdown[field]) for field in REWARD_COMPONENT_FIELDS),
            abs=1e-9,
            rel=0.0,
        )
    finally:
        env.close()


def test_successful_fusion_on_terminal_response_is_composed_exactly_once() -> None:
    replay = _load_replay()
    previous = _load_base_observation(replay)
    previous.update({"terminalHint": "running", "done": False, "over": False})
    terminal = copy.deepcopy(previous)
    terminal.update(
        {
            "done": True,
            "over": True,
            "terminalHint": "game_over_or_loss",
            "screenState": "game_over_restart_screen",
            "onGameOverScreen": True,
            "gameOverTextVisible": True,
            "gameplayReady": False,
        }
    )
    fusion_result = _fusion_action_result(
        {
            "success": True,
            "legal": True,
            "reason": "",
            "column": 4,
            "event_id": "terminal-fusion-event",
        }
    )
    env = _terminal_step_env(
        previous,
        terminal,
        action_result=fusion_result,
        fusion_policy="observe",
    )
    try:
        _observation, reward, done, _truncated, info = env.step(0)
        breakdown = info["reward_breakdown"]
        result = info["action_result"]
        assert done is True
        assert breakdown["fusion_reward"] > 0.0
        assert breakdown["win_loss_reward"] == -env.config.reward.loss_penalty
        assert result["fusionRewardApplied"] is True
        assert result["fusionRewardDelta"] == pytest.approx(
            breakdown["fusion_reward"], abs=1e-9, rel=0.0
        )
        assert result["structuredActionResult"]["executed"] is True
        assert result["actionDecision"]["legal"] is True
        assert "terminal-fusion-event" in env._reward_state.fusion.accounted_event_ids
        assert reward == pytest.approx(
            sum(float(breakdown[field]) for field in REWARD_COMPONENT_FIELDS),
            abs=1e-9,
            rel=0.0,
        )
    finally:
        env.close()
