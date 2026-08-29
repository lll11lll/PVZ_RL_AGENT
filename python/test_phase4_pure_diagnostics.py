from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, Dict, Iterable, List

import pytest
import pvzrl_diagnostics

from pvzrl_diagnostics import compose_environment_safety_diagnostics
from pvzrl_observation_facts import build_step_facts


def _observation(
    *,
    frame: int,
    mower_rows: Iterable[int],
    plant_count: int = 4,
    visible_plant_count: int | None = None,
    wave: int = 3,
    game_time: float = 10.0,
    seed_slots: List[Dict[str, Any]] | None = None,
    row_count: int = 5,
) -> Dict[str, Any]:
    active_rows = sorted(int(row) for row in mower_rows)
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
        "columnCount": 9,
        "wave": wave,
        "maxWave": 10,
        "time": game_time,
        "plantCount": plant_count,
        "visiblePlantObjectCount": (
            plant_count if visible_plant_count is None else visible_plant_count
        ),
        "zombieCount": 2,
        "logicalMowerCount": len(active_rows),
        "visibleMowerObjectCount": len(active_rows),
        "visibleMowers": [
            {
                "row": row,
                "activeInHierarchy": True,
                "inBoardBounds": True,
                "inMowerArray": True,
            }
            for row in active_rows
        ],
        "seedSlots": seed_slots or [],
    }


def _event(observation: Dict[str, Any], event: str, **fields: Any) -> Dict[str, Any]:
    return {
        "event": event,
        "step": observation.get("frameCount"),
        "wave": observation.get("wave"),
        "maxWave": observation.get("maxWave"),
        "zombieCount": observation.get("zombieCount"),
        "plantCount": observation.get("plantCount"),
        "gameplayReady": observation.get("gameplayReady"),
        "screenState": observation.get("screenState"),
        "nextStep": observation.get("nextStep"),
        "done": observation.get("done"),
        "over": observation.get("over"),
        "terminalHint": observation.get("terminalHint"),
        **fields,
    }


def _diagnostics(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    corrupting = {
        "mower_respawn_detected",
        "cooldown_reset_detected",
        "seed_slot_object_id_changed_during_gameplay",
        "board_refresh_detected",
    }
    corruption_count = sum(event["event"] in corrupting for event in events)
    return {
        "environment_corruption_detected": corruption_count > 0,
        "environment_corruption_penalty": 10.0 if corruption_count else 0.0,
        "env_corruption_count": corruption_count,
        "mower_respawn_detected_count": sum(
            event["event"] == "mower_respawn_detected" for event in events
        ),
        "cooldown_reset_detected_count": sum(
            event["event"] == "cooldown_reset_detected" for event in events
        ),
        "board_refresh_detected_count": sum(
            event["event"] == "board_refresh_detected" for event in events
        ),
        "false_reward_unlock_during_gameplay_count": sum(
            event["event"] == "false_reward_unlock_during_gameplay" for event in events
        ),
        "false_cleanup_reward_ui_during_gameplay_count": sum(
            event["event"] == "false_cleanup_reward_ui_during_gameplay" for event in events
        ),
        "post_win_veto_live_board_count": sum(
            event["event"] == "post_win_veto_live_board" for event in events
        ),
        "blocked_cleanup_during_gameplay_count": sum(
            event["event"] == "suspicious_cleanup_reward_ui_during_gameplay"
            for event in events
        ),
        "suspicious_cleanup_reward_ui_count": sum(
            event["event"] == "suspicious_cleanup_reward_ui_during_gameplay"
            for event in events
        ),
        "reset_reward_ui_cleanup_count": 0,
        "reset_reward_ui_cleanup_blocked_count": 0,
        "safety_events": events,
    }


def _compose(
    previous: Dict[str, Any],
    current: Dict[str, Any],
    *,
    requested_action: int,
    lost_mower_rows: frozenset[int] = frozenset(),
    missing_mower_rows: frozenset[int] = frozenset(),
    mower_baseline_ready: bool = False,
    with_facts: bool = False,
):
    plant_types = (1, 0)
    return compose_environment_safety_diagnostics(
        previous,
        current,
        requested_action=requested_action,
        fallback_plant_types=plant_types,
        fallback_row_count=5,
        lost_mower_rows=lost_mower_rows,
        missing_mower_rows=missing_mower_rows,
        mower_baseline_ready=mower_baseline_ready,
        live_board_progress=False,
        post_win_signal_present=False,
        cleanup_signal_active=False,
        suspicious_cleanup_signal_during_gameplay=False,
        previous_confirmed_postgame=False,
        current_confirmed_postgame=False,
        previous_facts=build_step_facts(previous, plant_types) if with_facts else None,
        current_facts=build_step_facts(current, plant_types) if with_facts else None,
    )


def test_mower_respawn_preserves_exact_event_and_advances_immutable_state() -> None:
    before_loss = _observation(frame=9, mower_rows=(0, 1, 2, 3), row_count=4)
    absent_once = _observation(frame=10, mower_rows=(0, 2, 3), row_count=4)
    missing_result = _compose(
        before_loss,
        absent_once,
        requested_action=0,
        mower_baseline_ready=True,
    )
    assert missing_result.diagnostics == _diagnostics([])
    assert missing_result.next_lost_mower_rows == frozenset()
    assert missing_result.next_missing_mower_rows == frozenset({1})

    previous = _observation(frame=11, mower_rows=(0, 2, 3), row_count=4)
    loss_result = _compose(
        absent_once,
        previous,
        requested_action=0,
        lost_mower_rows=missing_result.next_lost_mower_rows,
        missing_mower_rows=missing_result.next_missing_mower_rows,
        mower_baseline_ready=missing_result.mower_baseline_ready,
    )
    assert loss_result.diagnostics == _diagnostics([])
    assert loss_result.next_lost_mower_rows == frozenset({1})
    assert loss_result.next_missing_mower_rows == frozenset()

    current = _observation(frame=12, mower_rows=(0, 1, 2, 3), row_count=4)
    result = _compose(
        previous,
        current,
        requested_action=166,
        lost_mower_rows=loss_result.next_lost_mower_rows,
        missing_mower_rows=loss_result.next_missing_mower_rows,
        mower_baseline_ready=loss_result.mower_baseline_ready,
    )
    mower_event = _event(
        current,
        "mower_respawn_detected",
        rows=[1],
        mowers_before=[True, False, True, True],
        mowers_after=[True, True, True, True],
        last_action=166,
    )
    board_event = _event(
        current,
        "board_refresh_detected",
        plant_count_before=4,
        plant_count_after=4,
        visible_plant_count_before=4,
        visible_plant_count_after=4,
        wave_before=3,
        wave_after=3,
        time_before=10.0,
        time_after=10.0,
        mower_count_before=3,
        mower_count_after=4,
        last_action=166,
    )

    assert result.diagnostics == _diagnostics([mower_event, board_event])
    assert result.next_lost_mower_rows == frozenset({1})
    with pytest.raises(FrozenInstanceError):
        result.next_lost_mower_rows = frozenset()  # type: ignore[misc]


def test_two_slot_cooldown_reset_uses_prebuilt_facts_and_exact_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before_slots = [
        {
            "slotIndex": 0,
            "plantType": 1,
            "plantTypeName": "SunFlower",
            "currentCooldown": 6.0,
            "fullCooldown": 10.0,
            "cardInstanceId": 101,
        },
        {
            "slotIndex": 1,
            "plantType": 0,
            "plantTypeName": "Peashooter",
            "currentCooldown": 7.0,
            "fullCooldown": 10.0,
            "cardInstanceId": 102,
        },
    ]
    after_slots = [
        {**slot, "currentCooldown": 0.0, "ready": True}
        for slot in before_slots
    ]
    previous = _observation(
        frame=20,
        mower_rows=range(5),
        game_time=10.0,
        seed_slots=before_slots,
    )
    current = _observation(
        frame=21,
        mower_rows=range(5),
        game_time=10.1,
        seed_slots=after_slots,
    )
    monkeypatch.setattr(
        pvzrl_diagnostics,
        "_cooldowns_by_slot",
        lambda *_args, **_kwargs: pytest.fail("raw seed slots were rescanned"),
    )
    result = _compose(previous, current, requested_action=17, with_facts=True)
    events = [
        _event(
            current,
            "cooldown_reset_detected",
            slot=0,
            plant="SunFlower",
            cooldown_before=6.0,
            cooldown_after=0.0,
            full_cooldown=10.0,
            drop_amount=6.0,
            elapsed_game_time=0.09999999999999964,
            last_action=17,
        ),
        _event(
            current,
            "cooldown_reset_detected",
            slot=1,
            plant="Peashooter",
            cooldown_before=7.0,
            cooldown_after=0.0,
            full_cooldown=10.0,
            drop_amount=7.0,
            elapsed_game_time=0.09999999999999964,
            last_action=17,
        ),
    ]

    assert result.diagnostics == _diagnostics(events)
    assert result.next_lost_mower_rows == frozenset()


def test_seed_slot_fact_projection_preserves_legacy_last_entry_wins() -> None:
    observation = _observation(
        frame=22,
        mower_rows=range(5),
        seed_slots=[
            {
                "slotIndex": 0,
                "plantType": 9,
                "currentCooldown": 8.0,
                "cardInstanceId": 100,
            },
            None,
            {
                "slotIndex": 0,
                "typeName": "AliasMustNotLeakIntoSafetyEvents",
                "currentCooldown": 5.0,
                "rawCooldown": 0.5,
                "fullCooldown": 7.5,
                "ready": True,
                "cardInstanceId": 200,
            },
        ],  # type: ignore[list-item]
    )
    facts = build_step_facts(observation, (1, 0))
    expected = pvzrl_diagnostics._cooldowns_by_slot(observation, (1, 0))
    actual = pvzrl_diagnostics._cooldowns_from_seed_slot_facts(facts)
    assert actual == expected
    assert actual[0]["plantType"] == 1
    assert actual[0]["plantTypeName"] == ""
    assert actual[0]["cardInstanceId"] == 200


def test_board_refresh_event_payload_matches_legacy_shape() -> None:
    previous = _observation(
        frame=30,
        mower_rows=(0, 1, 2, 3),
        plant_count=12,
        visible_plant_count=12,
        wave=4,
        game_time=20.0,
    )
    current = _observation(
        frame=31,
        mower_rows=(0, 1, 2, 3),
        plant_count=1,
        visible_plant_count=1,
        wave=1,
        game_time=2.0,
    )
    result = _compose(previous, current, requested_action=0)
    expected_event = _event(
        current,
        "board_refresh_detected",
        plant_count_before=12,
        plant_count_after=1,
        visible_plant_count_before=12,
        visible_plant_count_after=1,
        wave_before=4,
        wave_after=1,
        time_before=20.0,
        time_after=2.0,
        mower_count_before=4,
        mower_count_after=4,
        last_action=0,
    )

    assert result.diagnostics == _diagnostics([expected_event])
    assert result.next_lost_mower_rows == frozenset()
