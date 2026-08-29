from __future__ import annotations

import json
import os
from pathlib import Path

from pvzrl_gui_stream_events import (
    StreamerEventReader,
    normalize_streamer_event,
    resolve_streamer_experiment_directory,
    streamer_event_log_path,
)


VIEWER_HASH = "abcdef12" * 8


def _append(path: Path, payload: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _viewer_event(command_id: str, *, row: int = 0) -> dict[str, object]:
    return {
        "event": "viewer_command",
        "occurred_monotonic": 12.5,
        "viewer_hash": VIEWER_HASH,
        "command_id": command_id,
        "status": "accepted",
        "legality": "LEGAL",
        "command": {
            "kind": "slot",
            "seed_slot": 0,
            "row": row,
            "column": 0,
        },
    }


def test_canonical_log_path_is_owned_by_the_run_directory(tmp_path: Path) -> None:
    assert streamer_event_log_path(tmp_path) == tmp_path / "logs" / "streamer_events.jsonl"
    assert StreamerEventReader.for_run(tmp_path).path == streamer_event_log_path(tmp_path)


def test_reader_tolerates_missing_partial_malformed_and_rotated_logs(tmp_path: Path) -> None:
    path = tmp_path / "streamer_events.jsonl"
    reader = StreamerEventReader(path, max_rows=8)

    rows, diagnostics = reader.read()
    assert rows == ()
    assert diagnostics["exists"] is False

    partial = json.dumps(_viewer_event("partial"))
    path.write_text(partial[:-1], encoding="utf-8")
    rows, diagnostics = reader.read()
    assert rows == ()
    assert diagnostics["pending_partial_bytes"] > 0

    with path.open("a", encoding="utf-8") as handle:
        handle.write(partial[-1:] + "\n")
        handle.write("{not-json}\n")
    _append(path, _viewer_event("complete"))
    rows, diagnostics = reader.read()
    assert [row["command_id"] for row in rows] == ["partial", "complete"]
    assert diagnostics["malformed_record_count"] == 1

    rotated = path.with_name(path.name + ".1")
    os.replace(path, rotated)
    _append(path, _viewer_event("after-rotation"))
    rows, diagnostics = reader.read()
    assert rows[-1]["command_id"] == "after-rotation"
    assert diagnostics["rotation_count"] + diagnostics["truncation_count"] >= 1


def test_normalizer_exposes_only_a_short_hash_and_allowlisted_command_fields() -> None:
    payload = _viewer_event("safe-command")
    payload.update(
        {
            "username": "must-not-persist",
            "raw_text": "!slot 1 1 1",
            "access_token": "credential-must-not-persist",
            "nested": {
                "chatter_user_login": "nested-identity",
                "cookie": "nested-cookie",
            },
        }
    )
    command = dict(payload["command"])  # type: ignore[arg-type]
    command.update(
        {
            "message": "raw-chat",
            "display_name": "nested-display-name",
            "secret_value": "nested-secret",
        }
    )
    payload["command"] = command

    path_rendered = json.dumps(normalize_streamer_event(payload), sort_keys=True)
    assert normalize_streamer_event(payload)["viewer"] == "hash:abcdef12"
    assert VIEWER_HASH not in path_rendered
    for unsafe_value in (
        "must-not-persist",
        "!slot 1 1 1",
        "credential-must-not-persist",
        "nested-identity",
        "nested-cookie",
        "raw-chat",
        "nested-display-name",
        "nested-secret",
    ):
        assert unsafe_value not in path_rendered
    assert normalize_streamer_event({**payload, "viewer_hash": "not-a-hash"})["viewer"] == ""


def test_row_six_and_slot_fourteen_are_converted_once_for_viewers() -> None:
    row = normalize_streamer_event(
        {
            "event": "executed_decision",
            "viewer_hash": VIEWER_HASH,
            "action_source": "TWITCH",
            "execution_status": "executed_verified",
            "bridge_success": True,
            "parsed_fields": {
                "kind": "slot",
                "seed_slot": 13,
                "row": 5,
                "column": 9,
            },
            "viewer_action_id": 840,
        }
    )
    assert row["command"] == {
        "kind": "slot",
        "row": 6,
        "column": 10,
        "coordinate_base": 1,
        "slot": 14,
    }
    assert row["command_label"] == "Slot 14 at R6 C10"

    decoded = normalize_streamer_event(
        {
            "event": "bc_demo_result",
            "action_id": 840,
            "bc_demo_recorded": True,
        }
    )
    assert decoded["command"] == {
        "kind": "action",
        "slot": 14,
        "row": 6,
        "column": 10,
        "coordinate_base": 1,
    }


def test_reader_retains_only_the_configured_number_of_recent_rows(tmp_path: Path) -> None:
    path = tmp_path / "streamer_events.jsonl"
    for index in range(7):
        _append(path, _viewer_event(f"command-{index}"))

    reader = StreamerEventReader(path, max_rows=3)
    rows, diagnostics = reader.read()
    assert [row["command_id"] for row in rows] == ["command-4", "command-5", "command-6"]
    assert diagnostics["rows_retained"] == 3
    assert diagnostics["row_capacity"] == 3


def test_existing_large_log_bootstraps_recent_rows_then_tails_incrementally(
    tmp_path: Path,
) -> None:
    path = tmp_path / "streamer_events.jsonl"
    for index in range(200):
        _append(path, _viewer_event(f"command-{index}"))
    assert path.stat().st_size > 2_048

    reader = StreamerEventReader(
        path,
        max_rows=4,
        max_read_bytes=512,
        max_pending_bytes=2_048,
    )
    rows, diagnostics = reader.read()
    assert [row["command_id"] for row in rows] == [
        "command-196",
        "command-197",
        "command-198",
        "command-199",
    ]
    assert diagnostics["rows_added"] >= 4

    _append(path, _viewer_event("command-200"))
    rows, diagnostics = reader.read()
    assert [row["command_id"] for row in rows] == [
        "command-197",
        "command-198",
        "command-199",
        "command-200",
    ]
    assert diagnostics["rows_added"] == 1


def test_history_filters_model_steps_but_keeps_viewer_and_bc_outcomes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "streamer_events.jsonl"
    _append(path, _viewer_event("accepted-viewer"))
    for action_id in range(150):
        _append(
            path,
            {
                "event": "executed_decision",
                "action_source": "MODEL",
                "model_action": action_id % 841,
                "executed_action": action_id % 841,
            },
        )
    _append(
        path,
        {
            "event": "bc_demo_result",
            "viewer_action_id": 840,
            "bc_demo_recorded": True,
        },
    )
    _append(
        path,
        {
            "event": "executed_decision",
            "action_source": "TWITCH",
            "viewer_hash": VIEWER_HASH,
            "command_id": "verified-viewer",
            "viewer_action_id": 840,
            "execution_status": "executed_verified",
            "parsed_fields": {
                "kind": "slot",
                "seed_slot": 13,
                "row": 5,
                "column": 9,
            },
        },
    )

    rows, diagnostics = StreamerEventReader(path, max_rows=5).read()
    assert [row["event_type"] for row in rows] == [
        "viewer_command",
        "bc_demo_result",
        "executed_decision",
    ]
    assert rows[-1]["command_label"] == "Slot 14 at R6 C10"
    assert rows[-1]["executed"] is True
    assert diagnostics["non_viewer_events_filtered"] == 150


def test_experiment_directory_resolution_handles_inner_runs_and_stale_status(
    tmp_path: Path,
) -> None:
    experiment = tmp_path / "actual-experiment"
    cycle_run = experiment / "cycles" / "cycle_000004" / "train"
    cycle_run.mkdir(parents=True)
    (experiment / "streamer_state.json").write_text("{}\n", encoding="utf-8")
    configured = tmp_path / "configured-experiment"
    configured.mkdir()

    assert resolve_streamer_experiment_directory(
        active_run=cycle_run,
        configured_run=configured,
    ) == experiment.resolve()
    assert resolve_streamer_experiment_directory(
        active_run=cycle_run,
        configured_run=configured,
        prefer_configured=True,
    ) == configured.resolve()

    structural_experiment = tmp_path / "without-markers"
    evaluation_run = structural_experiment / "evaluations" / "baseline"
    evaluation_run.mkdir(parents=True)
    assert resolve_streamer_experiment_directory(
        active_run=evaluation_run,
    ) == structural_experiment.resolve()

    explicit = tmp_path / "explicit-experiment"
    assert resolve_streamer_experiment_directory(
        explicit_experiment=explicit,
        active_run=cycle_run,
    ) == explicit.resolve()


def test_execution_is_verified_only_by_the_explicit_runtime_status() -> None:
    base = {
        "event": "executed_decision",
        "viewer_hash": VIEWER_HASH,
        "action_source": "TWITCH",
        "viewer_action_id": 1,
        "executed_action": 1,
        "bridge_success": True,
        "legal": True,
        "parsed_fields": {"kind": "slot", "seed_slot": 0, "row": 0, "column": 0},
    }
    dispatched = normalize_streamer_event(
        {
            **base,
            "execution_status": "dispatched",
            "bridge_reason": "post_observation_no_mutation",
        }
    )
    verified = normalize_streamer_event(
        {
            **base,
            "execution_status": "executed_verified",
            "bridge_reason": "success",
        }
    )

    assert dispatched["bridge_success"] is True
    assert dispatched["executed"] is False
    assert dispatched["execution_verified"] is False
    assert dispatched["result"] == "post_observation_no_mutation"
    assert verified["executed"] is True
    assert verified["execution_verified"] is True
    assert verified["result"] == "success"
