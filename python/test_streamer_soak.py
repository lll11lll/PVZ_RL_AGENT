from __future__ import annotations

import json

import pytest

from pvzrl_streamer_soak import (
    evaluate_streamer_soak_invariants,
    main,
    run_streamer_soak,
)


def test_message_count_soak_exercises_bounds_gates_expiry_and_demos() -> None:
    report = run_streamer_soak(
        message_count=640,
        queue_capacity=8,
        demo_capacity=5,
        seed=17,
    )

    assert report["ok"] is True
    assert report["invariants"] == {"passed": True, "failures": []}
    assert report["messages"]["attempted"] == 640
    assert report["messages"]["received"] == (
        640 + report["messages"]["phase_probe_attempted"]
    )
    assert report["messages"]["parsed"] > 0
    assert report["messages"]["selected"] > 5
    assert report["messages"]["rejected"] > 0
    assert report["messages"]["expired"] > 0
    assert report["queue"]["maximum_depth"] <= 8
    assert report["queue"]["source_maximum_depth"] <= 8
    assert report["queue"]["final_depth"] == 0
    assert report["demonstrations"]["size"] == 5
    assert report["demonstrations"]["total_evicted"] > 0
    assert report["phases"]["transitions"] > 2
    assert report["phases"]["reconnect_cycles"] > 0
    assert report["phases"]["source_gate_epoch"] > 0
    assert report["safety"] == {
        "stale_selections": 0,
        "dispatches_while_gated": 0,
        "phase_leaks": 0,
        "gate_epoch_regressions": 0,
    }
    assert report["errors"] == {"count": 0, "types": {}}
    assert report["memory"]["peak_bytes"] >= report["memory"]["current_bytes"]
    assert report["throughput_messages_per_second"] > 0.0


def test_message_count_soak_has_deterministic_functional_counters() -> None:
    first = run_streamer_soak(
        message_count=257,
        queue_capacity=7,
        demo_capacity=4,
        seed=123,
    )
    second = run_streamer_soak(
        message_count=257,
        queue_capacity=7,
        demo_capacity=4,
        seed=123,
    )

    assert first["messages"] == second["messages"]
    assert first["queue"] == second["queue"]
    assert first["phases"] == second["phases"]
    assert first["demonstrations"] == second["demonstrations"]
    assert first["safety"] == second["safety"]


def test_duration_mode_runs_at_least_one_batch() -> None:
    report = run_streamer_soak(
        duration_hours=1e-8,
        queue_capacity=4,
        demo_capacity=2,
        seed=5,
    )
    assert report["ok"] is True
    assert report["mode"] == "duration"
    assert report["messages"]["attempted"] >= 1


def test_invariant_evaluator_detects_bound_stale_phase_and_error_failures() -> None:
    report = run_streamer_soak(
        message_count=32,
        queue_capacity=4,
        demo_capacity=2,
        seed=9,
    )
    report["queue"]["maximum_depth"] = 5
    report["safety"]["stale_selections"] = 1
    report["safety"]["phase_leaks"] = 1
    report["errors"]["count"] = 1

    assert evaluate_streamer_soak_invariants(report) == (
        "queue_capacity_exceeded",
        "stale_command_selected",
        "phase_command_leak",
        "runtime_errors_observed",
    )


def test_cli_writes_report_and_returns_success(tmp_path, capsys) -> None:
    target = tmp_path / "reports" / "soak.json"
    exit_code = main(
        [
            "--message-count",
            "96",
            "--queue-capacity",
            "6",
            "--demo-capacity",
            "3",
            "--seed",
            "42",
            "--report-path",
            str(target),
        ]
    )

    assert exit_code == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["messages"]["attempted"] == 96
    assert "!slot" not in target.read_text(encoding="utf-8")
    assert "synthetic-viewer" not in target.read_text(encoding="utf-8")
    assert json.loads(capsys.readouterr().out)["ok"] is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"duration_hours": 1.0, "message_count": 1},
        {"message_count": 0},
        {"duration_hours": float("nan")},
        {"message_count": 1, "queue_capacity": 0},
        {"message_count": 1, "demo_capacity": 0},
    ],
)
def test_soak_rejects_invalid_configuration(kwargs) -> None:
    with pytest.raises(ValueError):
        run_streamer_soak(**kwargs)
