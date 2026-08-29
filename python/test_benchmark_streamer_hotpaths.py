from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark_streamer_hotpaths import main, run_benchmarks


EXPECTED_RESULTS = {
    "baseline_policy_loop_idle_check",
    "streamer_controller_idle_tick",
    "synthetic_generalist_env_step_without_streamer",
    "synthetic_generalist_env_step_with_quiet_streamer",
    "viewer_command_parse",
    "command_source_poll_parse_enqueue",
    "fifo_legal_command_selection",
    "current_mask_action_resolution",
    "structured_event_log_buffered_append",
    "structured_event_log_batch_fsync",
    "masked_bc_combined_optimizer_update",
}


def test_streamer_hotpath_report_exercises_all_synthetic_production_paths() -> None:
    report = run_benchmarks(
        samples=1,
        rounds=1,
        inner_iterations=2,
        poll_batch=2,
        log_batch=2,
        bc_updates=1,
    )

    assert report["schema_version"] == 1
    assert report["benchmark"] == "pvzrl_streamer_v1_hotpaths"
    assert report["synthetic_bridge_free"] is True
    assert report["contracts"]["observation_shape"] == [4364]
    assert report["contracts"]["action_count"] == 841
    assert report["contracts"]["identity_slots"] == 14
    assert report["contracts"]["mask_true_count"] == 1
    assert report["contracts"]["resolved_action_id"] == 134
    assert report["contracts"]["viewer_coordinates_converted_once"] == {
        "viewer_row": 2,
        "viewer_column": 4,
        "viewer_slot": 3,
        "internal_row": 1,
        "internal_column": 3,
        "internal_slot": 2,
    }
    assert set(report["results"]) == EXPECTED_RESULTS

    for name, measurement in report["results"].items():
        assert measurement.get("skipped") is not True, name
        assert measurement["operation_count"] > 0, name
        assert measurement["median_us_per_operation"] >= 0.0, name
        assert measurement["p95_us_per_operation"] >= 0.0, name

    bc = report["contracts"]["bc"]
    assert bc["observation_shape"] == [4364]
    assert bc["action_count"] == 841
    assert bc["bc_updates_observed"] == 1
    assert bc["last_bc_loss"] > 0.0
    assert bc["ppo_actor_pressure_neutralized"] is True
    assert report["comparisons"]["idle_loop_interpretation"].startswith("Python dispatch")

    step_contract = report["contracts"]["synthetic_environment_step"]
    assert step_contract["wrapper"] == "PvZMaskedPPOEnv.step"
    assert step_contract["bridge_operation_replaced"] is True
    assert step_contract["baseline_steps_observed"] == 4
    assert step_contract["streamer_steps_observed"] == 4
    assert step_contract["expected_steps_per_arm"] == 4
    assert step_contract["streamer_viewer_interventions"] == 0
    assert step_contract["streamer_queue_depth"] == 0
    assert step_contract["event_logging_in_step_pair"] is False
    step_comparison = report["comparisons"]["synthetic_environment_step"]
    assert step_comparison["paired_sample_count"] == 1
    assert step_comparison["cases_interleaved"] is True
    assert "not live bridge or Unity latency" in step_comparison["interpretation"]


def test_streamer_hotpath_cli_writes_json_atomically_and_can_skip_bc(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path = tmp_path / "nested" / "streamer-hotpaths.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text("partial-old-report", encoding="utf-8")

    assert main(
        [
            "--samples",
            "1",
            "--rounds",
            "1",
            "--inner-iterations",
            "2",
            "--poll-batch",
            "2",
            "--log-batch",
            "2",
            "--bc-updates",
            "0",
            "--json-out",
            str(report_path),
        ]
    ) == 0

    captured = capsys.readouterr()
    stdout_payload = json.loads(captured.out)
    file_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert file_payload == stdout_payload
    assert file_payload["results"]["masked_bc_combined_optimizer_update"] == {
        "skipped": True,
        "reason": "bc_updates_zero",
    }
    assert not list(report_path.parent.glob(f".{report_path.name}.*.tmp"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"samples": 0},
        {"rounds": 21},
        {"inner_iterations": 2049},
        {"poll_batch": 257},
        {"log_batch": 0},
        {"bc_updates": 11},
        {"samples": 100, "rounds": 20, "inner_iterations": 51},
        {"samples": 41, "rounds": 1, "inner_iterations": 25},
    ],
)
def test_streamer_hotpath_resource_bounds_fail_closed(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        run_benchmarks(**kwargs)
