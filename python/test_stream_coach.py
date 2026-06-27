"""Bridge-free tests for stream crowd-coach parsing and aggregation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from pvzrl_action_space import ACTION_SPACE_ADVENTURE_14_IDENTITY
from pvzrl_stream_coach import (
    COACH_REJECTION_RATE_LIMITED,
    CrowdCoachAggregator,
    JsonlCoachCommandSource,
    MockStreamScriptSource,
    StreamCoachController,
    StreamCoachRateLimiter,
    command_from_payload,
    parse_chat_command,
)


def seed_slot(slot_index: int, plant_type: int, name: str) -> Dict[str, Any]:
    return {
        "slotIndex": int(slot_index),
        "plantType": int(plant_type),
        "plantTypeName": str(name),
        "ready": True,
        "usable": True,
        "seedCost": 50,
    }


def observation(legal_actions: List[int], sun: int = 500) -> Dict[str, Any]:
    return {
        "rowCount": 5,
        "columnCount": 10,
        "boardFound": True,
        "gameplayReady": True,
        "sun": int(sun),
        "seedSlots": [
            seed_slot(0, 1, "SunFlower"),
            seed_slot(1, 0, "Peashooter"),
            seed_slot(2, 3, "WallNut"),
            seed_slot(3, 2, "CherryBomb"),
        ],
        "legalActions": list(legal_actions),
    }


def assert_case(results: List[Dict[str, Any]], name: str, condition: bool, detail: Any = None) -> None:
    results.append({"case": name, "passed": bool(condition), "detail": detail})


def main() -> int:
    results: List[Dict[str, Any]] = []

    lightweight_direct = command_from_payload(
        {
            "command": "plant",
            "seed_index": 0,
            "row": 2,
            "col": 4,
            "vote_count": 3,
            "timestamp": 123.0,
        }
    )
    assert_case(
        results,
        "lightweight command_from_payload parses direct command",
        bool(
            lightweight_direct is not None
            and lightweight_direct.command == "plant"
            and lightweight_direct.seed_index == 0
            and lightweight_direct.row == 2
            and lightweight_direct.col == 4
            and lightweight_direct.vote_count == 3
        ),
        lightweight_direct.to_dict() if lightweight_direct is not None else None,
    )

    lightweight_nested = command_from_payload(
        {
            "selected_command": {
                "command": "wait",
            },
            "vote_count": 2,
            "timestamp": 124.0,
        }
    )
    assert_case(
        results,
        "lightweight command_from_payload parses selected_command payload",
        bool(
            lightweight_nested is not None
            and lightweight_nested.command == "wait"
            and lightweight_nested.vote_count == 2
        ),
        lightweight_nested.to_dict() if lightweight_nested is not None else None,
    )

    limiter = StreamCoachRateLimiter(max_actions_per_minute=2)
    allow_1 = limiter.allow(now=1.0)
    allow_2 = limiter.allow(now=2.0)
    allow_3 = limiter.allow(now=3.0)
    allow_4 = limiter.allow(now=62.0)
    assert_case(
        results,
        "lightweight StreamCoachRateLimiter rolling window",
        bool(allow_1 and allow_2 and (not allow_3) and allow_4),
        {"allow_1": allow_1, "allow_2": allow_2, "allow_3": allow_3, "allow_4": allow_4},
    )

    with tempfile.TemporaryDirectory(prefix="pvzrl_stream_coach_test_") as temp_dir:
        source_path = f"{temp_dir}/commands.jsonl"
        with open(source_path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"selected_command": {"command": "wait"}, "timestamp": 100.0}) + "\n")
            handle.write(
                json.dumps(
                    {
                        "selected_command": {"command": "plant", "seed_index": 0, "row": 2, "col": 4},
                        "vote_count": 3,
                        "timestamp": 101.0,
                        "active_viewers_estimate": 12,
                    }
                )
                + "\n"
            )
        source = JsonlCoachCommandSource(path=Path(source_path))
        first_poll = source.poll_latest()
        with open(source_path, "a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "selected_command": {"command": "plant", "seed_index": 1, "row": 3, "col": 5},
                        "vote_count": 4,
                        "timestamp": 102.0,
                        "active_viewers_estimate": 15,
                    }
                )
                + "\n"
            )
        latest = source.poll_latest()
        replay_source = JsonlCoachCommandSource(path=Path(source_path), start_at_end=False)
        replay_latest = replay_source.poll_latest()
        assert_case(
            results,
            "lightweight JsonlCoachCommandSource starts at EOF by default and reads fresh commands",
            bool(
                first_poll is None
                and latest is not None
                and latest.command == "plant"
                and latest.seed_index == 1
                and latest.vote_count == 4
                and latest.active_viewers_estimate == 15
                and replay_latest is not None
                and replay_latest.command == "plant"
            ),
            {
                "first_poll": first_poll.to_dict() if first_poll is not None else None,
                "latest": latest.to_dict() if latest is not None else None,
                "replay_latest": replay_latest.to_dict() if replay_latest is not None else None,
            },
        )

    with tempfile.TemporaryDirectory(prefix="pvzrl_stream_source_test_") as temp_dir:
        source_path = Path(temp_dir) / "messages.jsonl"
        source = JsonlCoachCommandSource(path=source_path)
        source.start()
        source_path.write_text(
            "\n".join(
                [
                    json.dumps({"user": "viewer_a", "message_id": "m1", "message": "!wait", "timestamp": 10.0}),
                    json.dumps(
                        {
                            "user": "viewer_b",
                            "message_id": "m2",
                            "selected_command": {"command": "plant", "seed_index": 0, "row": 2, "col": 4},
                            "vote_count": 2,
                            "timestamp": 11.0,
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        drained = source.drain_messages()
        source_diag = source.get_diagnostics()
        assert_case(
            results,
            "jsonl stream source normalizes raw chat and legacy vote records",
            bool(
                len(drained) == 3
                and drained[0].message_id == "m1"
                and drained[0].display_name == "viewer_a"
                and drained[0].text == "!wait"
                and drained[1].text == "!plant 0 2 4"
                and drained[2].user_id == "viewer_b_1"
                and source_diag.get("stream_source_messages_emitted") == 3
            ),
            {"messages": [message.__dict__ for message in drained], "diagnostics": source_diag},
        )

    parsed_plant, plant_ok, plant_reason = parse_chat_command("!plant 0 2 4")
    assert_case(
        results,
        "parse !plant 0 2 4",
        bool(
            plant_ok
            and not plant_reason
            and isinstance(parsed_plant, dict)
            and parsed_plant.get("command") == "plant"
            and parsed_plant.get("seed_index") == 0
            and parsed_plant.get("row") == 2
            and parsed_plant.get("col") == 4
        ),
        {"parsed": parsed_plant, "reason": plant_reason},
    )
    parsed_plain_plant, plain_plant_ok, plain_plant_reason = parse_chat_command("plant 0 2 4")
    assert_case(
        results,
        "parse plant 0 2 4",
        bool(
            plain_plant_ok
            and not plain_plant_reason
            and isinstance(parsed_plain_plant, dict)
            and parsed_plain_plant.get("command") == "plant"
            and parsed_plain_plant.get("seed_index") == 0
            and parsed_plain_plant.get("row") == 2
            and parsed_plain_plant.get("col") == 4
        ),
        {"parsed": parsed_plain_plant, "reason": plain_plant_reason},
    )

    parsed_fuse, fuse_ok, fuse_reason = parse_chat_command("!fuse 0 1 1")
    assert_case(
        results,
        "parse !fuse 0 1 1",
        bool(
            fuse_ok
            and not fuse_reason
            and isinstance(parsed_fuse, dict)
            and parsed_fuse.get("command") == "fuse"
            and parsed_fuse.get("seed_index") == 0
            and parsed_fuse.get("row") == 1
            and parsed_fuse.get("col") == 1
        ),
        {"parsed": parsed_fuse, "reason": fuse_reason},
    )

    malformed, malformed_ok, malformed_reason = parse_chat_command("!plant 0 2")
    assert_case(
        results,
        "reject malformed commands",
        bool((not malformed_ok) and malformed is None and bool(malformed_reason)),
        {"parsed": malformed, "reason": malformed_reason},
    )
    unsupported, unsupported_ok, unsupported_reason = parse_chat_command("!prefer Peashooter")
    assert_case(
        results,
        "reject unsupported mock chat commands with reason",
        bool((not unsupported_ok) and unsupported is None and unsupported_reason == "unknown_command"),
        {"parsed": unsupported, "reason": unsupported_reason},
    )

    agg_votes = CrowdCoachAggregator(window_sec=3.0, min_votes=1, log_path=None)
    agg_votes.ingest_message(platform="mock", username="alice", raw_text="!plant 0 2 4", timestamp=10.0)
    agg_votes.ingest_message(platform="mock", username="bob", raw_text="!plant 0 2 4", timestamp=10.2)
    top = agg_votes.top_votes(now=10.5, limit=3)
    assert_case(
        results,
        "aggregate duplicate commands",
        bool(top and top[0].canonical == "plant:0:2:4" and top[0].vote_count == 2),
        [vote.__dict__ for vote in top],
    )

    agg_spam = CrowdCoachAggregator(window_sec=3.0, min_votes=1, user_min_interval_sec=1.0, log_path=None)
    first = agg_spam.ingest_message(platform="mock", username="same_user", raw_text="!wait", timestamp=20.0)
    second = agg_spam.ingest_message(platform="mock", username="same_user", raw_text="!wait", timestamp=20.25)
    assert_case(
        results,
        "reject spam from same user",
        bool(first.valid_syntax and (not second.valid_syntax) and second.rejected_reason == COACH_REJECTION_RATE_LIMITED),
        {"first": first.__dict__, "second": second.__dict__},
    )

    legal_obs = observation([0, 25, 26])
    agg_select = CrowdCoachAggregator(window_sec=3.0, min_votes=2, log_path=None)
    agg_select.ingest_message(platform="mock", username="u1", raw_text="!plant 0 2 4", timestamp=30.0)
    agg_select.ingest_message(platform="mock", username="u2", raw_text="!plant 0 2 4", timestamp=30.1)
    agg_select.ingest_message(platform="mock", username="u3", raw_text="!wait", timestamp=30.2)
    decision = agg_select.choose_highest_voted_legal_command(
        observation=legal_obs,
        legal_actions=legal_obs["legalActions"],
        action_space_mode=ACTION_SPACE_ADVENTURE_14_IDENTITY,
        ppo_action=0,
        now=31.0,
    )
    assert_case(
        results,
        "choose highest-voted legal command",
        bool(decision.selected and decision.selected_policy_action == 25 and decision.selected_vote_count == 2),
        decision.__dict__,
    )

    illegal_obs = observation([0], sun=0)
    agg_illegal = CrowdCoachAggregator(window_sec=3.0, min_votes=1, log_path=None)
    agg_illegal.ingest_message(platform="mock", username="u1", raw_text="!plant 0 2 4", timestamp=40.0)
    illegal_decision = agg_illegal.choose_highest_voted_legal_command(
        observation=illegal_obs,
        legal_actions=illegal_obs["legalActions"],
        action_space_mode=ACTION_SPACE_ADVENTURE_14_IDENTITY,
        ppo_action=0,
        now=40.5,
    )
    assert_case(
        results,
        "queue transiently blocked command as pending wait",
        bool(
            illegal_decision.selected
            and bool(getattr(illegal_decision, "pending", False))
            and illegal_decision.selected_policy_action == 0
        ),
        illegal_decision.__dict__,
    )

    pending_followup = agg_illegal.choose_highest_voted_legal_command(
        observation=observation([0, 25], sun=500),
        legal_actions=[0, 25],
        action_space_mode=ACTION_SPACE_ADVENTURE_14_IDENTITY,
        ppo_action=0,
        now=44.5,
    )
    assert_case(
        results,
        "pending command executes after becoming legal even outside vote window",
        bool(
            pending_followup.selected
            and (not bool(getattr(pending_followup, "pending", False)))
            and pending_followup.selected_policy_action == 25
        ),
        pending_followup.__dict__,
    )

    threshold_obs = observation([0, 25])
    agg_threshold = CrowdCoachAggregator(window_sec=3.0, min_votes=2, log_path=None)
    agg_threshold.ingest_message(platform="mock", username="u1", raw_text="!plant 0 2 4", timestamp=50.0)
    threshold_decision = agg_threshold.choose_highest_voted_legal_command(
        observation=threshold_obs,
        legal_actions=threshold_obs["legalActions"],
        action_space_mode=ACTION_SPACE_ADVENTURE_14_IDENTITY,
        ppo_action=25,
        now=50.4,
    )
    assert_case(
        results,
        "fallback to PPO when no command passes threshold",
        bool((not threshold_decision.selected) and threshold_decision.fallback_to_ppo),
        threshold_decision.__dict__,
    )

    with tempfile.TemporaryDirectory(prefix="pvzrl_stream_controller_test_") as temp_dir:
        source_path = f"{temp_dir}/source.jsonl"
        with open(source_path, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "selected_command": {"command": "plant", "seed_index": 0, "row": 2, "col": 4},
                        "vote_count": 3,
                        "timestamp": 200.0,
                        "active_viewers_estimate": 18,
                    }
                )
                + "\n"
            )
        controller = StreamCoachController(
            enabled=True,
            command_path=Path(source_path),
            window_sec=3.0,
            min_votes=2,
            max_actions_per_minute=20,
        )
        ingested_initial = controller.poll_mock_source(username="local")
        with open(source_path, "a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "selected_command": {"command": "plant", "seed_index": 0, "row": 2, "col": 4},
                        "vote_count": 3,
                        "timestamp": 201.0,
                        "active_viewers_estimate": 18,
                    }
                )
                + "\n"
            )
        ingested = controller.poll_mock_source(username="local")
        control_decision = controller.choose_action(
            observation=observation([0, 25]),
            legal_actions=[0, 25],
            action_space_mode=ACTION_SPACE_ADVENTURE_14_IDENTITY,
            ppo_action=0,
            now=202.0,
        )
        diagnostics = controller.diagnostics_fields()
        assert_case(
            results,
            "controller starts at EOF and selects only fresh crowd command",
            bool(
                len(ingested_initial) == 0
                and len(ingested) == 3
                and control_decision.selected
                and control_decision.selected_policy_action == 25
                and diagnostics.get("stream_coach_active_viewers_estimate") == 18
            ),
            {
                "initial_ingested_count": len(ingested_initial),
                "ingested_count": len(ingested),
                "decision": control_decision.__dict__,
                "diagnostics": diagnostics,
            },
        )

    with tempfile.TemporaryDirectory(prefix="pvzrl_mock_stream_script_test_") as temp_dir:
        script_path = Path(temp_dir) / "mock_stream.jsonl"
        script_records = [
            {"t": 3, "user": "u3", "message": "!economy"},
            {"t": 1, "user": "u1", "message": "!wait"},
            {"t": 2, "user": "u2", "message": "!defend 2"},
        ]
        script_path.write_text("\n".join(json.dumps(row) for row in script_records) + "\n", encoding="utf-8")
        script_source = MockStreamScriptSource(script_path)
        before_due = script_source.poll_due(step_index=0)
        first_due = script_source.poll_due(step_index=1)
        second_due = script_source.poll_due(step_index=3)
        script_source_for_drain = MockStreamScriptSource(script_path)
        script_source_for_drain.start()
        drained_script_messages = script_source_for_drain.drain_messages(step_index=2)
        assert_case(
            results,
            "mock script emits messages in deterministic step order",
            bool(
                before_due == []
                and [record.message for record in first_due] == ["!wait"]
                and [record.message for record in second_due] == ["!defend 2", "!economy"]
                and [message.text for message in drained_script_messages] == ["!wait", "!defend 2"]
                and drained_script_messages[0].message_id.endswith(":1")
                and script_source.pending_count(step_index=3) == 0
            ),
            {
                "before_due": [record.to_dict() for record in before_due],
                "first_due": [record.to_dict() for record in first_due],
                "second_due": [record.to_dict() for record in second_due],
                "drained_script_messages": [message.__dict__ for message in drained_script_messages],
                "diagnostics": script_source.diagnostics_fields(step_index=3),
            },
        )

    with tempfile.TemporaryDirectory(prefix="pvzrl_mock_stream_controller_script_test_") as temp_dir:
        script_path = Path(temp_dir) / "mock_stream.jsonl"
        script_path.write_text(
            "\n".join(
                [
                    json.dumps({"t": 0, "user": "viewer_a", "message": "!wait"}),
                    json.dumps({"t": 1, "user": "viewer_b", "message": "!prefer Peashooter"}),
                    json.dumps({"t": 2, "user": "viewer_c", "message": "!plant 0 2 4"}),
                    json.dumps({"t": 2, "user": "viewer_d", "message": "!plant 0 2 4"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        controller = StreamCoachController(
            enabled=True,
            mode="mock",
            platform="mock",
            mock_script_path=script_path,
            window_sec=10.0,
            min_votes=1,
            max_actions_per_minute=20,
        )
        emitted_0 = controller.poll_mock_source(step_index=0)
        decision_0 = controller.choose_action(
            observation=observation([0, 25]),
            legal_actions=[0, 25],
            action_space_mode=ACTION_SPACE_ADVENTURE_14_IDENTITY,
            ppo_action=25,
            now=1.0,
        )
        emitted_1 = controller.poll_mock_source(step_index=1)
        emitted_2 = controller.poll_mock_source(step_index=2)
        decision_2 = controller.choose_action(
            observation=observation([0, 25]),
            legal_actions=[0, 25],
            action_space_mode=ACTION_SPACE_ADVENTURE_14_IDENTITY,
            ppo_action=0,
            now=2.5,
        )
        diagnostics = controller.diagnostics_fields()
        live_status_path = Path(temp_dir) / "live_status.json"
        live_payload = {"coach": diagnostics, "stream_coach": diagnostics, **diagnostics}
        live_status_path.write_text(json.dumps(live_payload, indent=2), encoding="utf-8")
        loaded_live = json.loads(live_status_path.read_text(encoding="utf-8"))
        assert_case(
            results,
            "scripted mock stream flows through controller diagnostics and live status",
            bool(
                len(emitted_0) == 1
                and decision_0.selected
                and decision_0.selected_policy_action == 0
                and len(emitted_1) == 1
                and emitted_1[0].rejected_reason == "unknown_command"
                and len(emitted_2) == 2
                and decision_2.selected
                and diagnostics.get("stream_coach_mode") == "mock"
                and diagnostics.get("stream_coach_alive") is True
                and diagnostics.get("mock_stream_messages_seen") == 4
                and diagnostics.get("mock_stream_commands_parsed") == 3
                and diagnostics.get("mock_stream_commands_rejected") == 1
                and diagnostics.get("last_stream_message") == "!plant 0 2 4"
                and loaded_live.get("coach", {}).get("stream_coach_mode") == "mock"
                and loaded_live.get("stream_coach", {}).get("mock_stream_commands_rejected") == 1
            ),
            {
                "emitted_0": [message.__dict__ for message in emitted_0],
                "emitted_1": [message.__dict__ for message in emitted_1],
                "emitted_2": [message.__dict__ for message in emitted_2],
                "decision_0": decision_0.__dict__,
                "decision_2": decision_2.__dict__,
                "diagnostics": diagnostics,
                "live": loaded_live,
            },
        )

    dry_controller = StreamCoachController(
        enabled=True,
        mode="mock",
        platform="mock",
        dry_run=True,
        apply_enabled=False,
        window_sec=10.0,
        min_votes=1,
        max_actions_per_minute=20,
    )
    dry_controller.aggregator.ingest_message(platform="mock", username="dry_user", raw_text="!wait", timestamp=300.0)
    dry_decision = dry_controller.choose_action(
        observation=observation([0, 25]),
        legal_actions=[0, 25],
        action_space_mode=ACTION_SPACE_ADVENTURE_14_IDENTITY,
        ppo_action=25,
        now=300.5,
    )
    dry_controller.record_dry_run_decision(dry_decision)
    dry_diag = dry_controller.diagnostics_fields()
    assert_case(
        results,
        "dry-run validates command without marking it applied",
        bool(
            dry_decision.selected
            and dry_diag.get("stream_coach_dry_run") is True
            and dry_diag.get("stream_coach_apply_enabled") is False
            and dry_diag.get("stream_coach_last_command_status") == "dry_run"
            and dry_diag.get("last_validated_coach_command") == "!wait"
            and dry_diag.get("last_applied_coach_command") == ""
            and dry_diag.get("stream_coach_dry_run_count") == 1
        ),
        {"decision": dry_decision.__dict__, "diagnostics": dry_diag},
    )

    apply_controller = StreamCoachController(
        enabled=True,
        mode="mock",
        platform="mock",
        dry_run=False,
        apply_enabled=True,
        window_sec=10.0,
        min_votes=1,
        max_actions_per_minute=20,
    )
    apply_controller.aggregator.ingest_message(platform="mock", username="apply_user", raw_text="!wait", timestamp=310.0)
    apply_decision = apply_controller.choose_action(
        observation=observation([0, 25]),
        legal_actions=[0, 25],
        action_space_mode=ACTION_SPACE_ADVENTURE_14_IDENTITY,
        ppo_action=25,
        now=310.5,
    )
    apply_controller.apply_step_outcome(apply_decision, {})
    apply_diag = apply_controller.diagnostics_fields()
    assert_case(
        results,
        "apply mode marks safe command as applied",
        bool(
            apply_decision.selected
            and apply_diag.get("stream_coach_dry_run") is False
            and apply_diag.get("stream_coach_apply_enabled") is True
            and apply_diag.get("stream_coach_last_command_status") == "applied"
            and apply_diag.get("last_applied_coach_command") == "!wait"
            and apply_diag.get("stream_coach_applied_count") == 1
        ),
        {"decision": apply_decision.__dict__, "diagnostics": apply_diag},
    )

    with tempfile.TemporaryDirectory(prefix="pvzrl_stream_stale_test_") as temp_dir:
        source_path = Path(temp_dir) / "stale.jsonl"
        source_path.write_text(json.dumps({"user": "old", "message": "!wait"}) + "\n", encoding="utf-8")
        stale_source = JsonlCoachCommandSource(path=source_path, start_at_end=False)
        stale_controller = StreamCoachController(
            enabled=True,
            mode="mock",
            platform="mock",
            source=stale_source,
            dry_run=True,
            apply_enabled=False,
        )
        stale_detected = stale_controller.clear_pending_state(clear_source=True, reason="startup")
        emitted_after_clear = stale_controller.poll_source(step_index=0)
        stale_diag = stale_controller.diagnostics_fields()
        assert_case(
            results,
            "startup stale command clearing prevents replay",
            bool(
                stale_detected
                and emitted_after_clear == []
                and stale_diag.get("stream_coach_startup_stale_cleared") is True
                and stale_diag.get("stream_coach_stale_messages_cleared") == 1
                and stale_diag.get("stream_coach_last_clear_reason") == "startup"
            ),
            {
                "emitted_after_clear": [message.__dict__ for message in emitted_after_clear],
                "diagnostics": stale_diag,
            },
        )

    payload = {"ok": all(row["passed"] for row in results), "results": results}
    print(json.dumps(payload, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
