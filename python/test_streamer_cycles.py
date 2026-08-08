from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

import pytest

from pvzrl_streamer import (
    StreamerCheckpointManager,
    atomic_write_json,
    compare_evaluations,
    run_streamer_cycles,
)


def _model(path: Path, body: bytes = b"model") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    (path.parent / "model_metadata.json").write_text(
        json.dumps({"metadata_version": 4, "model_family": "test-generalist"}),
        encoding="utf-8",
    )
    return path


def test_evaluation_comparison_uses_win_rate_then_reward_and_retains_ties() -> None:
    incumbent = {"summary": {"win_rate": 0.5, "avg_reward": 10.0}}
    assert compare_evaluations({"summary": {"win_rate": 0.6, "avg_reward": -99.0}}, incumbent) == 1
    assert compare_evaluations({"summary": {"win_rate": 0.5, "avg_reward": 11.0}}, incumbent) == 1
    assert compare_evaluations({"summary": {"win_rate": 0.5, "avg_reward": 10.0}}, incumbent) == 0
    assert compare_evaluations({"summary": {"win_rate": 0.4, "avg_reward": 999.0}}, incumbent) == -1


def test_checkpoint_roles_are_atomic_and_worse_current_cannot_replace_best(tmp_path: Path) -> None:
    baseline = _model(tmp_path / "source" / "model.zip", b"baseline")
    manager = StreamerCheckpointManager(tmp_path / "experiment", baseline)
    current_source = _model(tmp_path / "cycle" / "model.zip", b"current-one")

    manager.save_current(current_source, model_steps=525_000, training_cycle=1, training_metrics={})
    assert manager.current_model_path.read_bytes() == b"current-one"
    promoted = manager.promote_best_if_improved(
        manager.current_model_path,
        evaluation={"summary": {"win_rate": 0.7, "avg_reward": 4.0}},
        model_steps=525_000,
        training_cycle=1,
    )
    assert promoted is True
    best_bytes = manager.best_model_path.read_bytes()

    worse_source = _model(tmp_path / "cycle-two" / "model.zip", b"current-two")
    manager.save_current(worse_source, model_steps=550_000, training_cycle=2, training_metrics={})
    promoted = manager.promote_best_if_improved(
        manager.current_model_path,
        evaluation={"summary": {"win_rate": 0.6, "avg_reward": 999.0}},
        model_steps=550_000,
        training_cycle=2,
    )
    assert promoted is False
    assert manager.best_model_path.read_bytes() == best_bytes
    assert manager.current_model_path.read_bytes() == b"current-two"
    assert not list((tmp_path / "experiment").rglob("*.tmp"))


def test_baseline_hash_guard_detects_mutation(tmp_path: Path) -> None:
    baseline = _model(tmp_path / "source" / "model.zip", b"baseline")
    manager = StreamerCheckpointManager(tmp_path / "experiment", baseline)
    baseline.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="streamer_baseline_changed"):
        manager.verify_baseline_immutable()


def test_existing_experiment_rejects_a_different_baseline(tmp_path: Path) -> None:
    first = _model(tmp_path / "source-one" / "model.zip", b"baseline-one")
    second = _model(tmp_path / "source-two" / "model.zip", b"baseline-two")
    experiment = tmp_path / "experiment"
    StreamerCheckpointManager(experiment, first)
    with pytest.raises(RuntimeError, match="streamer_baseline_role_changed"):
        StreamerCheckpointManager(experiment, second)


class _Gate:
    def __init__(self) -> None:
        self.events: List[str] = []

    def enter_train(self, cycle: int) -> None:
        self.events.append(f"train:{cycle}")

    def enter_evaluate(self, cycle: int) -> None:
        self.events.append(f"eval:{cycle}")

    def shutdown(self) -> None:
        self.events.append("shutdown")


def test_cycle_order_baseline_current_best_and_policy_step_accounting(tmp_path: Path) -> None:
    baseline = _model(tmp_path / "baseline" / "model.zip", b"baseline")
    gate = _Gate()
    trained_from: List[Path] = []
    eval_calls: List[tuple[Path, int, int]] = []

    def train_cycle(start: Path, cycle: int, run_dir: Path, steps: int, level: int) -> Dict[str, Any]:
        trained_from.append(start)
        output = _model(run_dir / "model.zip", f"cycle-{cycle}".encode())
        return {
            "model_path": str(output),
            "model_steps": 500_000 + cycle * steps,
            "ppo_policy_timesteps": steps,
            "total_environment_actions": steps + cycle,
            "viewer_interventions": cycle,
            "next_adventure_level": level,
        }

    def evaluate(model: Path, _run_dir: Path, episodes: int, level: int) -> Dict[str, Any]:
        eval_calls.append((model, episodes, level))
        score = 0.5 if model == baseline else 0.5 + 0.1 * len(trained_from)
        return {
            "adventure_start_level": level,
            "next_adventure_level": level,
            "summary": {"episodes_completed": episodes, "win_rate": score, "avg_reward": score * 10},
        }

    result = run_streamer_cycles(
        config={
            "run_dir": str(tmp_path / "experiment"),
            "streamer_baseline_checkpoint": str(baseline),
            "streamer_evaluation_episodes": 50,
            "streamer_policy_steps_per_cycle": 25_000,
            "streamer_max_cycles": 2,
            "streamer_endurance_hours": 0.0,
        },
        train_cycle=train_cycle,
        evaluate_checkpoint=evaluate,
        phase_gate=gate,
    )

    assert gate.events == ["eval:0", "train:1", "eval:1", "train:2", "eval:2", "shutdown"]
    assert eval_calls[0] == (baseline, 50, 1)
    assert trained_from[0] == baseline
    assert trained_from[1].name == "model.zip"
    assert result["completed_cycle"] == 2
    assert result["ppo_policy_timesteps"] == 50_000
    assert result["status"] == "complete"
    assert (tmp_path / "experiment" / "checkpoints" / "current" / "model.zip").read_bytes() == b"cycle-2"
    assert (tmp_path / "experiment" / "checkpoints" / "best" / "model.zip").read_bytes() == b"cycle-2"
    cycle_rows = (tmp_path / "experiment" / "streamer_cycles.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(cycle_rows) == 2
    assert json.loads(cycle_rows[-1])["best_promoted"] is True

    # A changed protocol cannot safely re-run the immutable baseline after the
    # live Adventure profile has moved.
    eval_calls.clear()
    with pytest.raises(RuntimeError, match="streamer_baseline_evaluation_protocol_changed"):
        run_streamer_cycles(
            config={
                "run_dir": str(tmp_path / "experiment"),
                "streamer_baseline_checkpoint": str(baseline),
                "streamer_evaluation_episodes": 40,
                "streamer_policy_steps_per_cycle": 25_000,
                "streamer_max_cycles": 2,
                "streamer_endurance_hours": 0.0,
            },
            train_cycle=train_cycle,
            evaluate_checkpoint=evaluate,
            phase_gate=gate,
        )
    assert eval_calls == []


def test_mid_cycle_current_checkpoint_resumes_only_remaining_policy_steps(tmp_path: Path) -> None:
    baseline = _model(tmp_path / "baseline" / "model.zip", b"baseline")
    experiment = tmp_path / "experiment"
    eval_calls: List[tuple[Path, int, int]] = []

    def evaluate(model: Path, _run_dir: Path, episodes: int, level: int) -> Dict[str, Any]:
        eval_calls.append((model, episodes, level))
        return {
            "adventure_start_level": level,
            "next_adventure_level": level,
            "summary": {
                "episodes_completed": episodes,
                "win_rate": 0.5 if model == baseline else 0.6,
                "avg_reward": 5.0,
                "model_steps": 500_000,
            }
        }

    def interrupted_train(
        _start: Path, cycle: int, run_dir: Path, steps: int, level: int
    ) -> Dict[str, Any]:
        assert cycle == 1
        assert steps == 25_000
        recovery = _model(run_dir / "recovery.zip", b"recovery-10k")
        manager = StreamerCheckpointManager(experiment, baseline)
        manager.save_current(
            recovery,
            model_steps=510_000,
            training_cycle=1,
            training_metrics={
                "status": "in_progress",
                "cycle_start_model_steps": 500_000,
                "cycle_target_policy_steps": 25_000,
                "ppo_policy_timesteps": 10_000,
                "total_environment_actions": 10_500,
                "viewer_interventions": 500,
                "next_adventure_level": level,
            },
        )
        raise RuntimeError("simulated_crash")

    base_config = {
        "run_dir": str(experiment),
        "streamer_baseline_checkpoint": str(baseline),
        "streamer_evaluation_episodes": 50,
        "streamer_policy_steps_per_cycle": 25_000,
        "streamer_max_cycles": 1,
        "streamer_endurance_hours": 0.0,
    }
    with pytest.raises(RuntimeError, match="simulated_crash"):
        run_streamer_cycles(
            config=base_config,
            train_cycle=interrupted_train,
            evaluate_checkpoint=evaluate,
        )

    resume_calls: List[tuple[Path, int]] = []

    def resumed_train(
        start: Path, cycle: int, run_dir: Path, steps: int, level: int
    ) -> Dict[str, Any]:
        resume_calls.append((start, steps))
        assert cycle == 1
        output = _model(run_dir / "model.zip", b"finished-25k")
        return {
            "model_path": str(output),
            "model_steps": 525_000,
            "ppo_policy_timesteps": 15_000,
            "total_environment_actions": 15_100,
            "viewer_interventions": 100,
            "next_adventure_level": level,
        }

    result = run_streamer_cycles(
        config=base_config,
        train_cycle=resumed_train,
        evaluate_checkpoint=evaluate,
    )
    assert resume_calls == [(experiment / "checkpoints" / "current" / "model.zip", 15_000)]
    assert result["completed_cycle"] == 1
    assert result["ppo_policy_timesteps"] == 25_000
    training = result["last_cycle"]["training"]
    assert training["ppo_policy_timesteps"] == 25_000
    assert training["total_environment_actions"] == 25_600
    assert training["viewer_interventions"] == 600


def test_resume_fails_closed_when_completed_state_has_no_current(tmp_path: Path) -> None:
    baseline = _model(tmp_path / "baseline" / "model.zip", b"baseline")
    experiment = tmp_path / "experiment"
    experiment.mkdir(parents=True)
    (experiment / "streamer_state.json").write_text(
        json.dumps({"completed_cycle": 1, "ppo_policy_timesteps": 25_000}),
        encoding="utf-8",
    )

    def evaluate(_model: Path, _run_dir: Path, episodes: int, level: int) -> Dict[str, Any]:
        return {
            "adventure_start_level": level,
            "next_adventure_level": level,
            "summary": {"episodes_completed": episodes, "win_rate": 0.5, "avg_reward": 1.0},
        }

    with pytest.raises(RuntimeError, match="streamer_current_missing_for_completed_state"):
        run_streamer_cycles(
            config={
                "run_dir": str(experiment),
                "streamer_baseline_checkpoint": str(baseline),
                "streamer_evaluation_episodes": 50,
                "streamer_policy_steps_per_cycle": 25_000,
                "streamer_max_cycles": 1,
                "streamer_endurance_hours": 0.0,
            },
            train_cycle=lambda *_args: {},
            evaluate_checkpoint=evaluate,
        )


def test_corrupt_best_record_fails_closed_instead_of_overwriting(tmp_path: Path) -> None:
    baseline = _model(tmp_path / "baseline" / "model.zip", b"baseline")
    manager = StreamerCheckpointManager(tmp_path / "experiment", baseline)
    manager.best_model_path.parent.mkdir(parents=True, exist_ok=True)
    manager.best_model_path.write_bytes(b"existing-best")
    manager.best_record_path.write_text("{broken", encoding="utf-8")
    with pytest.raises(RuntimeError, match="streamer_best_record_invalid"):
        manager.promote_best_if_improved(
            baseline,
            evaluation={"summary": {"win_rate": 1.0, "avg_reward": 1.0}},
            model_steps=500_000,
            training_cycle=0,
        )


def test_cycle_log_appends_after_a_truncated_final_row(tmp_path: Path) -> None:
    from pvzrl_streamer import _append_cycle_record_once

    path = tmp_path / "cycles.jsonl"
    path.write_text('{"cycle":', encoding="utf-8")
    _append_cycle_record_once(path, {"cycle": 2, "status": "complete"})
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == '{"cycle":'
    assert json.loads(lines[1]) == {"cycle": 2, "status": "complete"}


def test_atomic_json_failure_does_not_leave_temporary_file(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    with pytest.raises(TypeError):
        atomic_write_json(target, {"not_json": object()})
    assert not target.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_existing_cycle_evaluation_protocol_mismatch_fails_closed(tmp_path: Path) -> None:
    baseline = _model(tmp_path / "baseline" / "model.zip", b"baseline")
    experiment = tmp_path / "experiment"
    config = {
        "run_dir": str(experiment),
        "streamer_baseline_checkpoint": str(baseline),
        "streamer_evaluation_episodes": 2,
        "streamer_policy_steps_per_cycle": 10,
        "streamer_max_cycles": 1,
        "streamer_endurance_hours": 0.0,
    }

    def train_cycle(
        _start: Path, cycle: int, run_dir: Path, steps: int, level: int
    ) -> Dict[str, Any]:
        output = _model(run_dir / "model.zip", f"cycle-{cycle}".encode())
        return {
            "model_path": str(output),
            "model_steps": 500_000 + steps,
            "ppo_policy_timesteps": steps,
            "next_adventure_level": level,
        }

    evaluation_calls = 0

    def interrupted_evaluate(
        model: Path, _run_dir: Path, episodes: int, level: int
    ) -> Dict[str, Any]:
        nonlocal evaluation_calls
        evaluation_calls += 1
        if model != baseline:
            raise RuntimeError("simulated_evaluation_crash")
        return {
            "adventure_start_level": level,
            "next_adventure_level": level,
            "summary": {
                "episodes_completed": episodes,
                "win_rate": 0.5,
                "avg_reward": 1.0,
                "model_steps": 500_000,
            },
        }

    with pytest.raises(RuntimeError, match="simulated_evaluation_crash"):
        run_streamer_cycles(
            config=config,
            train_cycle=train_cycle,
            evaluate_checkpoint=interrupted_evaluate,
        )
    assert evaluation_calls == 2

    def must_not_evaluate(*_args: Any) -> Dict[str, Any]:
        raise AssertionError("interrupted evaluation must never be rerun implicitly")

    with pytest.raises(RuntimeError, match="streamer_cycle_evaluation_interrupted"):
        run_streamer_cycles(
            config=config,
            train_cycle=train_cycle,
            evaluate_checkpoint=must_not_evaluate,
        )

    evaluation_path = (
        experiment / "cycles" / "cycle_000001" / "evaluation" / "evaluation.json"
    )
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_path.write_text(
        json.dumps(
            {
                "streamer_cycle": 1,
                "current_sha256": "wrong-checkpoint",
                "evaluation_protocol": {
                    "episodes": 2,
                    "deterministic": True,
                    "promotion_order": ["win_rate", "avg_reward"],
                    "tie_behavior": "retain_incumbent",
                    "adventure_start_level": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="streamer_cycle_evaluation_protocol_changed"):
        run_streamer_cycles(
            config=config,
            train_cycle=train_cycle,
            evaluate_checkpoint=must_not_evaluate,
        )


def test_adventure_level_handoff_is_sequential_and_cross_level_scores_are_unknown(
    tmp_path: Path,
) -> None:
    baseline = _model(tmp_path / "baseline" / "model.zip", b"baseline")
    train_starts: List[int] = []
    eval_starts: List[int] = []

    def train_cycle(
        _start: Path,
        cycle: int,
        run_dir: Path,
        steps: int,
        level: int,
    ) -> Dict[str, Any]:
        train_starts.append(level)
        output = _model(run_dir / "model.zip", f"cycle-{cycle}".encode())
        return {
            "model_path": str(output),
            "model_steps": 500_000 + cycle * steps,
            "ppo_policy_timesteps": steps,
            "next_adventure_level": level + 1,
        }

    def evaluate(model: Path, _run_dir: Path, episodes: int, level: int) -> Dict[str, Any]:
        eval_starts.append(level)
        return {
            "adventure_start_level": level,
            "next_adventure_level": level + 1,
            "summary": {
                "episodes_completed": episodes,
                "win_rate": 0.5 if model == baseline else 1.0,
                "avg_reward": 1.0 if model == baseline else 100.0,
            },
        }

    result = run_streamer_cycles(
        config={
            "run_dir": str(tmp_path / "experiment"),
            "streamer_baseline_checkpoint": str(baseline),
            "streamer_evaluation_episodes": 50,
            "streamer_policy_steps_per_cycle": 25_000,
            "streamer_max_cycles": 2,
            "streamer_endurance_hours": 0.0,
            "adventure_start_level": 1,
        },
        train_cycle=train_cycle,
        evaluate_checkpoint=evaluate,
    )

    assert eval_starts == [1, 3, 5]
    assert train_starts == [2, 4]
    assert result["next_adventure_level"] == 6
    assert result["last_cycle"]["comparison_to_baseline"] == "UNKNOWN"
    assert result["last_cycle"]["best_promoted"] is False
    best = json.loads(
        (tmp_path / "experiment" / "checkpoints" / "best" / "streamer_checkpoint.json").read_text(
            encoding="utf-8"
        )
    )
    assert best["training_cycle"] == 0


def test_corrupt_baseline_record_fails_closed(tmp_path: Path) -> None:
    baseline = _model(tmp_path / "baseline" / "model.zip", b"baseline")
    experiment = tmp_path / "experiment"
    manager = StreamerCheckpointManager(experiment, baseline)
    manager.baseline_record_path.write_text("{broken", encoding="utf-8")
    with pytest.raises(RuntimeError, match="streamer_baseline_record_invalid"):
        StreamerCheckpointManager(experiment, baseline)


def test_interrupted_current_commit_preserves_prior_and_repairs_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pvzrl_streamer

    baseline = _model(tmp_path / "baseline" / "model.zip", b"baseline")
    manager = StreamerCheckpointManager(tmp_path / "experiment", baseline)
    first = _model(tmp_path / "first" / "model.zip", b"first")
    second = _model(tmp_path / "second" / "model.zip", b"second")
    manager.save_current(first, model_steps=510_000, training_cycle=1, training_metrics={})
    prior = manager.current_record()
    assert prior is not None

    real_atomic_write_json = pvzrl_streamer.atomic_write_json

    def fail_current_record(path: Path, payload: Dict[str, Any]) -> None:
        if path == manager.current_record_path:
            raise OSError("injected record failure")
        real_atomic_write_json(path, payload)

    monkeypatch.setattr(pvzrl_streamer, "atomic_write_json", fail_current_record)
    with pytest.raises(OSError, match="injected record failure"):
        manager.save_current(second, model_steps=520_000, training_cycle=1, training_metrics={})
    monkeypatch.setattr(pvzrl_streamer, "atomic_write_json", real_atomic_write_json)

    recovered = manager.current_record()
    assert recovered is not None
    assert recovered["sha256"] == prior["sha256"]
    assert manager.current_model_path.read_bytes() == b"first"


def test_current_and_best_version_retention_is_bounded_to_two(tmp_path: Path) -> None:
    baseline = _model(tmp_path / "baseline" / "model.zip", b"baseline")
    manager = StreamerCheckpointManager(tmp_path / "experiment", baseline)
    for cycle in range(1, 5):
        source = _model(tmp_path / f"source-{cycle}" / "model.zip", f"model-{cycle}".encode())
        manager.save_current(source, model_steps=500_000 + cycle, training_cycle=cycle, training_metrics={})
        manager.promote_best_if_improved(
            manager.current_model_path,
            evaluation={
                "adventure_start_level": 1,
                "summary": {"win_rate": cycle / 10.0, "avg_reward": float(cycle)},
            },
            model_steps=500_000 + cycle,
            training_cycle=cycle,
        )
    assert len(list(manager.current_versions_dir.glob("v-*"))) <= 2
    assert len(list(manager.best_versions_dir.glob("v-*"))) <= 2


def test_interrupted_best_metadata_is_completed_on_retry(tmp_path: Path) -> None:
    baseline = _model(tmp_path / "baseline" / "model.zip", b"baseline")
    manager = StreamerCheckpointManager(tmp_path / "experiment", baseline)
    digest = __import__("hashlib").sha256(b"baseline").hexdigest()
    orphan_dir = manager.best_versions_dir / f"v-{digest[:16]}"
    orphan_dir.mkdir(parents=True)
    (orphan_dir / "model.zip").write_bytes(b"baseline")
    assert manager.promote_best_if_improved(
        baseline,
        evaluation={"adventure_start_level": 1, "summary": {"win_rate": 0.5, "avg_reward": 1.0}},
        model_steps=500_000,
        training_cycle=0,
    )
    assert (orphan_dir / "model_metadata.json").is_file()


def test_baseline_checkpoint_must_be_outside_experiment_directory(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    baseline = _model(experiment / "source" / "model.zip", b"baseline")
    with pytest.raises(RuntimeError, match="streamer_baseline_inside_experiment"):
        StreamerCheckpointManager(experiment, baseline)


def test_corrupt_baseline_evaluation_marker_fails_closed(tmp_path: Path) -> None:
    baseline = _model(tmp_path / "baseline" / "model.zip", b"baseline")
    experiment = tmp_path / "experiment"
    StreamerCheckpointManager(experiment, baseline)
    marker = experiment / "evaluations" / "baseline" / "evaluation_state.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("{broken", encoding="utf-8")

    with pytest.raises(RuntimeError, match="streamer_baseline_evaluation_marker_invalid"):
        run_streamer_cycles(
            config={
                "run_dir": str(experiment),
                "streamer_baseline_checkpoint": str(baseline),
                "streamer_evaluation_episodes": 2,
                "streamer_policy_steps_per_cycle": 10,
                "streamer_max_cycles": 1,
            },
            train_cycle=lambda *_args: {},
            evaluate_checkpoint=lambda *_args: {},
        )


def test_fresh_baseline_evaluation_must_match_requested_start_level(tmp_path: Path) -> None:
    baseline = _model(tmp_path / "baseline" / "model.zip", b"baseline")

    def wrong_level(
        _model_path: Path,
        _run_dir: Path,
        episodes: int,
        level: int,
    ) -> Dict[str, Any]:
        return {
            "adventure_start_level": level + 1,
            "next_adventure_level": level + 1,
            "summary": {"episodes_completed": episodes, "win_rate": 0.5},
        }

    with pytest.raises(RuntimeError, match="streamer_baseline_evaluation_level_mismatch"):
        run_streamer_cycles(
            config={
                "run_dir": str(tmp_path / "experiment"),
                "streamer_baseline_checkpoint": str(baseline),
                "streamer_evaluation_episodes": 2,
                "streamer_policy_steps_per_cycle": 10,
                "streamer_max_cycles": 1,
                "adventure_start_level": 7,
            },
            train_cycle=lambda *_args: {},
            evaluate_checkpoint=wrong_level,
        )


def test_target_step_periodic_current_cannot_skip_training_handoff(tmp_path: Path) -> None:
    baseline = _model(tmp_path / "baseline" / "model.zip", b"baseline")
    experiment = tmp_path / "experiment"
    config = {
        "run_dir": str(experiment),
        "streamer_baseline_checkpoint": str(baseline),
        "streamer_evaluation_episodes": 2,
        "streamer_policy_steps_per_cycle": 10,
        "streamer_max_cycles": 1,
    }
    evaluation_calls = 0

    def evaluate(
        _model_path: Path,
        _run_dir: Path,
        episodes: int,
        level: int,
    ) -> Dict[str, Any]:
        nonlocal evaluation_calls
        evaluation_calls += 1
        return {
            "adventure_start_level": level,
            "next_adventure_level": level,
            "summary": {"episodes_completed": episodes, "win_rate": 0.5},
        }

    def interrupted_train(
        _start: Path,
        cycle: int,
        run_dir: Path,
        steps: int,
        level: int,
    ) -> Dict[str, Any]:
        assert steps == 10
        periodic = _model(run_dir / "periodic.zip", b"periodic-at-target")
        manager = StreamerCheckpointManager(experiment, baseline)
        manager.save_current(
            periodic,
            model_steps=500_010,
            training_cycle=cycle,
            training_metrics={
                "status": "in_progress",
                "ppo_policy_timesteps": 10,
                "total_environment_actions": 12,
                "viewer_interventions": 2,
                "next_adventure_level": level,
            },
        )
        raise RuntimeError("simulated_crash_after_periodic_current")

    with pytest.raises(RuntimeError, match="simulated_crash_after_periodic_current"):
        run_streamer_cycles(
            config=config,
            train_cycle=interrupted_train,
            evaluate_checkpoint=evaluate,
        )

    def must_not_train(*_args: Any) -> Dict[str, Any]:
        raise AssertionError("target-step periodic CURRENT must not resume or evaluate")

    with pytest.raises(RuntimeError, match="streamer_recovery_phase_handoff_unproven"):
        run_streamer_cycles(
            config=config,
            train_cycle=must_not_train,
            evaluate_checkpoint=evaluate,
        )
    assert evaluation_calls == 1


def test_missing_best_for_completed_experiment_fails_closed(tmp_path: Path) -> None:
    baseline = _model(tmp_path / "baseline" / "model.zip", b"baseline")
    experiment = tmp_path / "experiment"
    config = {
        "run_dir": str(experiment),
        "streamer_baseline_checkpoint": str(baseline),
        "streamer_evaluation_episodes": 2,
        "streamer_policy_steps_per_cycle": 10,
        "streamer_max_cycles": 1,
    }

    def train(
        _start: Path,
        cycle: int,
        run_dir: Path,
        steps: int,
        level: int,
    ) -> Dict[str, Any]:
        model = _model(run_dir / "model.zip", b"cycle-model")
        return {
            "model_path": str(model),
            "model_steps": 500_000 + steps,
            "ppo_policy_timesteps": steps,
            "next_adventure_level": level,
        }

    def evaluate(
        model: Path,
        _run_dir: Path,
        episodes: int,
        level: int,
    ) -> Dict[str, Any]:
        return {
            "adventure_start_level": level,
            "next_adventure_level": level,
            "summary": {
                "episodes_completed": episodes,
                "win_rate": 0.5 if model == baseline else 0.6,
                "model_steps": 500_000 if model == baseline else 500_010,
            },
        }

    run_streamer_cycles(config=config, train_cycle=train, evaluate_checkpoint=evaluate)
    shutil.rmtree(experiment / "checkpoints" / "best")

    with pytest.raises(RuntimeError, match="streamer_best_missing_for_existing_state"):
        run_streamer_cycles(
            config=config,
            train_cycle=lambda *_args: {},
            evaluate_checkpoint=lambda *_args: {},
        )


def test_stale_same_cycle_current_cannot_replace_completed_state_evidence(tmp_path: Path) -> None:
    baseline = _model(tmp_path / "baseline" / "model.zip", b"baseline")
    experiment = tmp_path / "experiment"
    config = {
        "run_dir": str(experiment),
        "streamer_baseline_checkpoint": str(baseline),
        "streamer_evaluation_episodes": 2,
        "streamer_policy_steps_per_cycle": 10,
        "streamer_max_cycles": 1,
    }

    def train(
        _start: Path,
        cycle: int,
        run_dir: Path,
        steps: int,
        level: int,
    ) -> Dict[str, Any]:
        model = _model(run_dir / "model.zip", b"completed-model")
        return {
            "model_path": str(model),
            "model_steps": 500_000 + steps,
            "ppo_policy_timesteps": steps,
            "next_adventure_level": level,
        }

    def evaluate(
        _model_path: Path,
        _run_dir: Path,
        episodes: int,
        level: int,
    ) -> Dict[str, Any]:
        return {
            "adventure_start_level": level,
            "next_adventure_level": level,
            "summary": {
                "episodes_completed": episodes,
                "win_rate": 0.5,
                "model_steps": 500_010,
            },
        }

    run_streamer_cycles(config=config, train_cycle=train, evaluate_checkpoint=evaluate)
    stale = _model(tmp_path / "stale" / "model.zip", b"stale-same-cycle")
    manager = StreamerCheckpointManager(experiment, baseline)
    manager.save_current(
        stale,
        model_steps=500_005,
        training_cycle=1,
        training_metrics={
            "status": "trained_complete",
            "ppo_policy_timesteps": 10,
            "next_adventure_level": 1,
        },
    )

    with pytest.raises(RuntimeError, match="streamer_current_state_hash_mismatch"):
        run_streamer_cycles(
            config=config,
            train_cycle=lambda *_args: {},
            evaluate_checkpoint=lambda *_args: {},
        )
