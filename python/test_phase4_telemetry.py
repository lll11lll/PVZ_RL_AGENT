from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any, Dict

import pvzrl_telemetry
from pvzrl_adventure import build_agent_payload, build_live_status
from pvzrl_sb3 import PvZMaskedPPOEnv
from pvzrl_telemetry import LiveStatusWriter, live_status_significant_state
from test_refactor_support import make_wrapper, observation_for_wrapper
from train_ppo import (
    EPISODE_METRIC_FIELDS,
    EpisodeMetricWriter,
    build_runtime_live_status_payload,
    clean_episode_row,
    write_progress_csv_rows,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_lazy_status_writer_reduces_ordinary_builds_and_writes(tmp_path: Path) -> None:
    clock = FakeClock()
    path = tmp_path / "live_status.json"
    writer = LiveStatusWriter(path, min_interval_seconds=5.0, monotonic=clock)
    builds = 0

    def build() -> Dict[str, Any]:
        nonlocal builds
        builds += 1
        return {"status": "running", "state": "TRAINING_STEP", "current_step": builds}

    token = live_status_significant_state({"status": "running", "state": "TRAINING_STEP"})
    for _ in range(20):
        writer.write_lazy(build, significant_state=token)
        clock.advance(1.0)

    assert writer.stats.attempts == 20
    assert writer.stats.writes == 4
    assert writer.stats.skipped == 16
    assert writer.stats.payload_builds == builds == 4


def test_significant_transitions_and_forced_final_updates_bypass_interval(tmp_path: Path) -> None:
    clock = FakeClock()
    path = tmp_path / "live_status.json"
    writer = LiveStatusWriter(path, min_interval_seconds=60.0, monotonic=clock)
    states = [
        {"status": "starting", "state": "STARTUP_VALIDATION"},
        {"status": "running", "state": "SEED_SELECTION", "screenState": "seed_selection"},
        {"status": "running", "state": "POST_WIN_PENDING", "screenState": "reward_unlock"},
    ]
    for payload in states:
        writer.write_lazy(
            lambda payload=payload: dict(payload),
            significant_state=live_status_significant_state(payload),
        )
    final_payload = {"status": "complete", "state": "EPISODE_COMPLETE", "done_reason": "win"}
    writer.write_lazy(
        lambda: dict(final_payload),
        significant_state=live_status_significant_state(final_payload),
        force=True,
    )

    assert writer.stats.writes == 4
    assert json.loads(path.read_text(encoding="utf-8")) == final_payload


def test_timeout_and_freeze_payloads_are_never_coalesced(tmp_path: Path) -> None:
    clock = FakeClock()
    path = tmp_path / "live_status.json"
    writer = LiveStatusWriter(path, min_interval_seconds=60.0, monotonic=clock)
    token = ("same",)
    writer.write({"status": "running", "state": "TRAINING_STEP"}, significant_state=token)
    writer.write(
        {"status": "running", "state": "TRAINING_STEP", "terminal_reason": "action_timeout"},
        significant_state=token,
    )
    writer.write(
        {"status": "running", "state": "TRAINING_STEP", "done_reason": "action_freeze"},
        significant_state=token,
    )
    assert writer.stats.writes == 3


def test_failed_atomic_publish_does_not_suppress_immediate_retry(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    clock = FakeClock()
    path = tmp_path / "live_status.json"
    writer = LiveStatusWriter(path, min_interval_seconds=60.0, monotonic=clock)
    payload = {"status": "running", "state": "SEED_SELECTION", "episode": 4}
    token = live_status_significant_state(payload)
    real_replace = pvzrl_telemetry.os.replace

    def locked(*_args: object, **_kwargs: object) -> None:
        raise PermissionError("locked")

    monkeypatch.setattr(pvzrl_telemetry.os, "replace", locked)
    monkeypatch.setattr(pvzrl_telemetry.time, "sleep", lambda _seconds: None)
    assert writer.write(payload, significant_state=token) is False
    assert not path.exists()

    monkeypatch.setattr(pvzrl_telemetry.os, "replace", real_replace)
    assert writer.write(payload, significant_state=token) is True
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert writer.stats.writes == 1
    assert writer.stats.skipped == 0


def test_episode_callback_keys_are_significant_transitions() -> None:
    assert live_status_significant_state({"episode": 1}) != live_status_significant_state(
        {"episode": 2}
    )
    assert live_status_significant_state(
        {"episode_index": 1}
    ) != live_status_significant_state({"episode_index": 2})


def test_runtime_live_status_builder_keeps_compatibility_keys_and_types() -> None:
    payload = build_runtime_live_status_payload(
        config={
            "run_mode": "fixed_train",
            "target_level": 3,
            "run_dir": "runs/example",
            "seed_list": ["SunFlower", "Peashooter"],
            "plant_types": [1, 0],
            "action_space_mode": "fixed",
            "max_seed_slots": 2,
            "total_timesteps": 100,
        },
        status="running",
        mode="fixed_train",
        summary={"episode": 2, "episode_length": 7, "episode_reward": 1.25, "total_timesteps": 42},
        observation={
            "sun": 150,
            "wave": 3,
            "maxWave": 10,
            "plantCount": 4,
            "zombieCount": 5,
            "gameplayReady": True,
            "screenState": "gameplay",
            "legalActions": [0, 1, 2],
            "lanes": [{"row": 0, "zombieCount": 2, "danger": 0.5}],
        },
    )
    expected_keys = {
        "mode", "run_mode", "status", "health", "updated_at", "target_level", "blocked_reason",
        "active_run", "model_path", "current_timestep", "total_timesteps", "target_timesteps",
        "seed_list", "plant_types", "action_count", "tactical_mask_enabled", "fusion_action_mask_enabled",
        "current_episode", "current_step", "current_wave", "max_wave", "current_reward", "recent_win_rate",
        "recent_avg_wave", "recent_avg_kills", "current_sun", "current_plants", "current_zombies", "sun",
        "wave", "maxWave", "plantCount", "zombieCount", "gameplayReady", "screenState", "terminalHint",
        "legal_action_count", "gameplay", "agent", "reward", "train", "rows", "plants_by_type",
        "zombies_by_row", "row_danger", "coach", "stream_coach", "human_coach", "summary", "eval",
    }
    assert expected_keys <= set(payload)
    assert isinstance(payload["updated_at"], float)
    assert isinstance(payload["gameplay"], dict)
    assert isinstance(payload["agent"], dict)
    assert isinstance(payload["reward"], dict)
    assert isinstance(payload["rows"], dict)
    assert payload["legal_action_count"] == 3
    assert payload["screenState"] == "gameplay"


def test_adventure_live_status_computes_action_mask_once() -> None:
    env = make_wrapper(identity=False)
    env._last_observation = observation_for_wrapper(env)
    calls = 0

    class Mask:
        @staticmethod
        def sum() -> int:
            return 3

    def counted_action_masks() -> Mask:
        nonlocal calls
        calls += 1
        return Mask()

    env.action_masks = counted_action_masks  # type: ignore[method-assign]
    context = {"mode": "adventure_eval", "last_action_id": 0}

    standalone_agent = build_agent_payload(env, context, {})
    assert calls == 1
    calls = 0

    payload = build_live_status(env, context, {}, {})

    assert calls == 1
    assert payload["agent"]["legal_action_count"] == 3
    assert payload["agent"] == standalone_agent


def test_episode_csv_and_json_row_schema_remain_exact(tmp_path: Path) -> None:
    row = clean_episode_row({"episode": 4, "done_reason": "win", "episode_reward": 2.5}, 0)
    csv_path = tmp_path / "episode_metrics.csv"
    written, fieldnames = write_progress_csv_rows(csv_path, [row], list(EPISODE_METRIC_FIELDS), False)
    assert written is True
    assert fieldnames == list(EPISODE_METRIC_FIELDS)
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(EPISODE_METRIC_FIELDS)
        loaded = next(reader)
    assert list(row) == list(EPISODE_METRIC_FIELDS)
    assert loaded["episode"] == "4"
    assert loaded["done_reason"] == "win"

    metrics = EpisodeMetricWriter(tmp_path / "paired.csv", tmp_path / "paired.jsonl")
    rows = metrics.append_summaries([{"episode": 4, "done_reason": "win", "episode_reward": 2.5}])
    jsonl_row = json.loads((tmp_path / "paired.jsonl").read_text(encoding="utf-8"))
    assert list(rows[0]) == list(EPISODE_METRIC_FIELDS)
    assert list(jsonl_row) == list(EPISODE_METRIC_FIELDS)
    assert rows[0] == jsonl_row

    batch = metrics.append_summaries(
        [
            {"done_reason": "timeout", "episode_reward": 0.0},
            {"done_reason": "loss", "episode_reward": -1.0},
        ]
    )
    # Preserve the legacy callback fallback for simultaneous vector summaries:
    # both see the same pre-append completed-row count.
    assert [row["episode"] for row in batch] == [1, 1]


def _watchdog_env(tmp_path: Path, *, verbose: bool) -> PvZMaskedPPOEnv:
    env = object.__new__(PvZMaskedPPOEnv)
    env.config = type(
        "Config",
        (),
        {
            "enable_action_watchdog": True,
            "action_diagnostics_path": str(tmp_path / "actions.jsonl"),
            "save_freeze_debug_bundle": True,
            "freeze_debug_dir": str(tmp_path / "freeze"),
            "debug_performance": verbose,
        },
    )()
    env._episode_index = 1
    env._step_count = 2
    env._reset_action_diagnostics()
    env.decode_policy_action = lambda action, observation=None: {  # type: ignore[method-assign]
        "kind": "wait",
        "seed_slot": -1,
        "row": -1,
        "col": -1,
    }
    return env


def _record_watchdog(
    env: PvZMaskedPPOEnv,
    *,
    timed_out: bool = False,
    invalid: bool = False,
    exception_text: str = "",
    corruption: bool = False,
    safety_event: bool = False,
    changed: bool = False,
) -> Dict[str, Any]:
    observation = {
        "frameCount": 1,
        "screenState": "gameplay",
        "gameplayReady": True,
        "sun": 100,
        "plants": [],
        "seedSlots": [
            {
                "slotIndex": 0,
                "plantType": 1,
                "ready": False,
                "currentCooldown": 5.0,
            }
        ],
    }
    post_observation = copy.deepcopy(observation)
    if changed:
        post_observation["plants"] = [
            {"row": 2, "column": 4, "type": 1, "instanceId": 99}
        ]
        post_observation["plantCount"] = 1
        post_observation["seedSlots"][0].update(
            {"ready": True, "currentCooldown": 0.0}
        )
    return env._record_action_diagnostic(
        policy_action=0,
        bridge_action=0,
        pre_observation=observation,
        post_observation=post_observation,
        info={
            "action_result": {"bridgeTimeout": timed_out, "illegalAction": invalid},
            "environment_corruption_detected": corruption,
            "done_reason": "env_corruption" if corruption else "",
            "terminal_reason": "board_state_refreshed_during_gameplay" if corruption else "",
            "safety_events": (
                [{"event": "board_refresh_detected"}]
                if corruption
                else [{"event": "fixed_level_possible_win_confirmation"}]
                if safety_event
                else []
            ),
        },
        started_at=100.0,
        duration=0.1,
        timed_out=timed_out,
        exception_text=exception_text or ("timeout" if timed_out else ""),
    )


def test_watchdog_buffers_normal_timing_but_persists_anomalies_and_verbose_mode(tmp_path: Path) -> None:
    normal = _watchdog_env(tmp_path / "normal", verbose=False)
    normal_record = _record_watchdog(normal)
    assert normal_record["detail_persisted"] is False
    assert normal_record["pre_action_state_hash"] == ""
    assert not Path(normal.config.action_diagnostics_path).exists()
    assert normal._action_diagnostic_summary()["mean_action_duration_seconds"] == 0.1

    changed = _watchdog_env(tmp_path / "changed", verbose=False)
    changed_record = _record_watchdog(changed, changed=True)
    assert changed_record["detail_persisted"] is False
    assert changed_record["pre_action_state"]["plants"] == []
    assert changed_record["pre_action_state"]["seed_slots"] == []
    assert changed_record["board_changed"] is True
    assert changed_record["cooldowns_changed"] is True

    timeout = _watchdog_env(tmp_path / "timeout", verbose=False)
    timeout_record = _record_watchdog(timeout, timed_out=True)
    assert timeout_record["detail_persisted"] is True
    assert timeout_record["pre_action_state_hash"]
    assert Path(timeout.config.action_diagnostics_path).exists()
    assert Path(timeout_record["debug_bundle_path"]).exists()

    verbose = _watchdog_env(tmp_path / "verbose", verbose=True)
    verbose_record = _record_watchdog(verbose)
    assert verbose_record["detail_persisted"] is True
    assert Path(verbose.config.action_diagnostics_path).exists()

    invalid = _watchdog_env(tmp_path / "invalid", verbose=False)
    invalid_record = _record_watchdog(invalid, invalid=True)
    assert invalid_record["anomaly"] is True
    assert invalid_record["detail_persisted"] is True
    assert Path(invalid.config.action_diagnostics_path).exists()

    errored = _watchdog_env(tmp_path / "error", verbose=False)
    error_record = _record_watchdog(errored, exception_text="bridge error")
    assert error_record["anomaly"] is True
    assert error_record["detail_persisted"] is True
    assert Path(errored.config.action_diagnostics_path).exists()

    corrupted = _watchdog_env(tmp_path / "corruption", verbose=False)
    corruption_record = _record_watchdog(corrupted, corruption=True)
    assert corruption_record["anomaly"] is True
    assert corruption_record["classification"] == "environment_corruption"
    assert corruption_record["detail_persisted"] is True
    assert corruption_record["safety_events"] == [{"event": "board_refresh_detected"}]
    assert Path(corrupted.config.action_diagnostics_path).exists()

    safety = _watchdog_env(tmp_path / "safety", verbose=False)
    safety_record = _record_watchdog(safety, safety_event=True)
    assert safety_record["anomaly"] is True
    assert safety_record["classification"] == "safety_event"
    assert safety_record["environment_corruption_detected"] is False
    assert safety_record["detail_persisted"] is True
    assert Path(safety.config.action_diagnostics_path).exists()
