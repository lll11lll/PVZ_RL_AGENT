"""Streamer V1 train/evaluate orchestration and protected checkpoint roles.

This module is deliberately game-agnostic.  It coordinates the maintained
Adventure Generalist trainer and evaluator through injected callables; it does
not implement another environment, action path, or evaluation loop.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Protocol


STREAM_TRAIN = "STREAM_TRAIN"
EVALUATE = "EVALUATE"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def atomic_copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with source.open("rb") as input_handle, temporary.open("wb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _summary(metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    value = metrics.get("summary")
    return value if isinstance(value, Mapping) else metrics


def evaluation_score(metrics: Mapping[str, Any]) -> tuple[float, float]:
    """Return the immutable V1 promotion tuple: win rate, then mean return."""

    summary = _summary(metrics)
    return (
        float(summary.get("win_rate", 0.0) or 0.0),
        float(summary.get("avg_reward", summary.get("mean_reward", 0.0)) or 0.0),
    )


def compare_evaluations(candidate: Mapping[str, Any], incumbent: Mapping[str, Any]) -> int:
    """Compare evaluations deterministically; an exact tie retains incumbent."""

    candidate_score = evaluation_score(candidate)
    incumbent_score = evaluation_score(incumbent)
    return 1 if candidate_score > incumbent_score else (-1 if candidate_score < incumbent_score else 0)


def _metadata_candidate(model_path: Path) -> Optional[Path]:
    candidates = [model_path.parent / "model_metadata.json"]
    if model_path.parent.name.lower() == "checkpoints":
        candidates.append(model_path.parent.parent / "model_metadata.json")
    return next((candidate for candidate in candidates if candidate.is_file()), None)


@dataclass(frozen=True)
class CheckpointRecord:
    role: str
    model_path: str
    sha256: str
    model_steps: int
    training_cycle: int
    saved_at: str
    evaluation: Dict[str, Any]


class StreamerCheckpointManager:
    """Maintain immutable BASELINE plus atomic CURRENT and protected BEST."""

    def __init__(self, experiment_dir: Path, baseline_checkpoint: Path) -> None:
        self.experiment_dir = experiment_dir
        self.baseline_checkpoint = baseline_checkpoint
        if not baseline_checkpoint.is_file():
            raise FileNotFoundError(f"Streamer baseline checkpoint not found: {baseline_checkpoint}")
        try:
            baseline_checkpoint.resolve().relative_to(experiment_dir.resolve())
        except ValueError:
            pass
        else:
            raise RuntimeError(
                "blocked_reason=streamer_baseline_inside_experiment: "
                "BASELINE must be outside the Streamer experiment directory"
            )
        self.baseline_sha256 = sha256_file(baseline_checkpoint)
        self.current_model_path = experiment_dir / "checkpoints" / "current" / "model.zip"
        self.best_model_path = experiment_dir / "checkpoints" / "best" / "model.zip"
        self.current_versions_dir = experiment_dir / "checkpoints" / "current" / "versions"
        self.best_versions_dir = experiment_dir / "checkpoints" / "best" / "versions"
        self.baseline_record_path = experiment_dir / "checkpoints" / "baseline.json"
        self.current_record_path = self.current_model_path.parent / "streamer_checkpoint.json"
        self.best_record_path = self.best_model_path.parent / "streamer_checkpoint.json"
        baseline_metadata = _metadata_candidate(baseline_checkpoint)
        baseline_record = {
            "role": "BASELINE",
            "model_path": str(baseline_checkpoint.resolve()),
            "sha256": self.baseline_sha256,
            "metadata_path": str(baseline_metadata.resolve()) if baseline_metadata is not None else "",
            "metadata_sha256": sha256_file(baseline_metadata) if baseline_metadata is not None else "",
            "captured_at": utc_now_iso(),
        }
        prior_baseline = _read_json_object(self.baseline_record_path)
        if self.baseline_record_path.is_file() and prior_baseline is None:
            raise RuntimeError("blocked_reason=streamer_baseline_record_invalid")
        if prior_baseline is not None:
            if str(prior_baseline.get("role") or "") != "BASELINE":
                raise RuntimeError("blocked_reason=streamer_baseline_record_invalid")
            prior_path = str(prior_baseline.get("model_path") or "")
            prior_sha256 = str(prior_baseline.get("sha256") or "")
            prior_metadata_path = str(prior_baseline.get("metadata_path") or "")
            prior_metadata_sha256 = str(prior_baseline.get("metadata_sha256") or "")
            if (
                prior_path != baseline_record["model_path"]
                or prior_sha256 != self.baseline_sha256
                or prior_metadata_path != baseline_record["metadata_path"]
                or prior_metadata_sha256 != baseline_record["metadata_sha256"]
            ):
                raise RuntimeError(
                    "blocked_reason=streamer_baseline_role_changed: an existing Streamer experiment "
                    "cannot be resumed with a different BASELINE checkpoint"
                )
        else:
            atomic_write_json(self.baseline_record_path, baseline_record)

    def verify_baseline_immutable(self) -> None:
        baseline_record = _read_json_object(self.baseline_record_path)
        if baseline_record is None or str(baseline_record.get("role") or "") != "BASELINE":
            raise RuntimeError("blocked_reason=streamer_baseline_record_invalid")
        actual = sha256_file(self.baseline_checkpoint)
        if actual != self.baseline_sha256 or str(baseline_record.get("sha256") or "") != actual:
            raise RuntimeError(
                "blocked_reason=streamer_baseline_changed: immutable baseline hash mismatch "
                f"expected={self.baseline_sha256} actual={actual}"
            )
        metadata_path_text = str(baseline_record.get("metadata_path") or "")
        metadata_hash = str(baseline_record.get("metadata_sha256") or "")
        if metadata_path_text:
            metadata_path = Path(metadata_path_text)
            if not metadata_path.is_file() or not metadata_hash or sha256_file(metadata_path) != metadata_hash:
                raise RuntimeError("blocked_reason=streamer_baseline_metadata_changed")

    def _copy_compatibility_metadata(self, source_model: Path, role_dir: Path) -> None:
        source_metadata = _metadata_candidate(source_model)
        if source_metadata is None:
            raise FileNotFoundError(f"Canonical model_metadata.json not found for {source_model}")
        try:
            payload = json.loads(source_metadata.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not copy checkpoint metadata from {source_metadata}: {exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"Invalid checkpoint metadata object: {source_metadata}")
        payload["model_path"] = str(role_dir / "model.zip")
        atomic_write_json(role_dir / "model_metadata.json", payload)

    def _materialize_version(
        self,
        source_model: Path,
        versions_dir: Path,
        *,
        role: str,
    ) -> tuple[Path, str, str]:
        source_sha256 = sha256_file(source_model)
        # A short hash keeps role paths below legacy Windows MAX_PATH while the
        # full digest remains authoritative in the record and is revalidated.
        version_dir = versions_dir / f"v-{source_sha256[:16]}"
        version_model = version_dir / "model.zip"
        if not version_model.is_file():
            atomic_copy_file(source_model, version_model)
        elif sha256_file(version_model) != source_sha256:
            raise RuntimeError(f"blocked_reason=streamer_{role.lower()}_version_hash_mismatch")

        # Metadata is independently recoverable.  If a process stopped after the
        # model copy but before metadata, a retry completes the immutable version.
        version_metadata = version_dir / "model_metadata.json"
        if not version_metadata.is_file():
            self._copy_compatibility_metadata(source_model, version_dir)
        if not version_metadata.is_file():
            raise RuntimeError(f"blocked_reason=streamer_{role.lower()}_version_metadata_missing")
        return version_model, source_sha256, sha256_file(version_metadata)

    @staticmethod
    def _prune_versions(
        versions_dir: Path,
        *,
        keep: tuple[Path, ...],
        maximum: int = 2,
    ) -> None:
        """Retain only the active and immediately prior immutable generations."""

        if not versions_dir.is_dir():
            return
        keep_resolved = {path.resolve() for path in keep if path}
        candidates = [
            path
            for path in versions_dir.iterdir()
            if path.is_dir() and path.name.startswith("v-")
        ]
        candidates.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
        retained = set(keep_resolved)
        for path in candidates:
            if len(retained) >= max(1, int(maximum)):
                break
            retained.add(path.resolve())
        for path in candidates:
            if path.resolve() not in retained:
                try:
                    shutil.rmtree(path)
                except OSError:
                    # Retention cleanup is best-effort after the atomic role
                    # pointer is committed; it must not invalidate that save.
                    continue

    def _role_record(
        self,
        *,
        role: str,
        record_path: Path,
        alias_model: Path,
        versions_dir: Path,
    ) -> Optional[Dict[str, Any]]:
        if not record_path.is_file() and not alias_model.is_file():
            return None
        if not record_path.is_file():
            raise RuntimeError(f"blocked_reason=streamer_{role.lower()}_record_missing")
        try:
            payload = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"blocked_reason=streamer_{role.lower()}_record_invalid") from exc
        if not isinstance(payload, dict) or str(payload.get("role") or "") != role:
            raise RuntimeError(f"blocked_reason=streamer_{role.lower()}_record_invalid")
        authoritative_model = Path(str(payload.get("model_path") or ""))
        expected_hash = str(payload.get("sha256") or "")
        if (
            not authoritative_model.is_file()
            or not expected_hash
            or sha256_file(authoritative_model) != expected_hash
        ):
            raise RuntimeError(f"blocked_reason=streamer_{role.lower()}_hash_mismatch")
        if int(payload.get("checkpoint_format", 1) or 1) >= 2:
            try:
                authoritative_model.resolve().relative_to(versions_dir.resolve())
            except ValueError as exc:
                raise RuntimeError(
                    f"blocked_reason=streamer_{role.lower()}_record_path_invalid"
                ) from exc
        metadata_path = authoritative_model.parent / "model_metadata.json"
        expected_metadata_hash = str(payload.get("metadata_sha256") or "")
        if not metadata_path.is_file() or (
            expected_metadata_hash and sha256_file(metadata_path) != expected_metadata_hash
        ):
            raise RuntimeError(f"blocked_reason=streamer_{role.lower()}_metadata_mismatch")
        # The conventional model.zip is a repairable compatibility alias.  The
        # immutable version selected by the atomic record is authoritative.
        alias_replaced = not alias_model.is_file() or sha256_file(alias_model) != expected_hash
        if alias_replaced:
            atomic_copy_file(authoritative_model, alias_model)
        alias_metadata = alias_model.parent / "model_metadata.json"
        if alias_replaced or not alias_metadata.is_file():
            self._copy_compatibility_metadata(authoritative_model, alias_model.parent)
        return payload

    def save_current(
        self,
        source_model: Path,
        *,
        model_steps: int,
        training_cycle: int,
        training_metrics: Mapping[str, Any],
    ) -> CheckpointRecord:
        prior = self.current_record()
        self.verify_baseline_immutable()
        version_model, source_sha256, metadata_sha256 = self._materialize_version(
            source_model,
            self.current_versions_dir,
            role="CURRENT",
        )
        atomic_copy_file(version_model, self.current_model_path)
        self._copy_compatibility_metadata(version_model, self.current_model_path.parent)
        record = CheckpointRecord(
            role="CURRENT",
            model_path=str(version_model),
            sha256=source_sha256,
            model_steps=int(model_steps),
            training_cycle=int(training_cycle),
            saved_at=utc_now_iso(),
            evaluation={},
        )
        atomic_write_json(
            self.current_record_path,
            {
                **asdict(record),
                "checkpoint_format": 2,
                "metadata_sha256": metadata_sha256,
                "compatibility_alias": str(self.current_model_path),
                "training_metrics": dict(training_metrics),
                "source_model": str(source_model),
            },
        )
        prior_path = Path(str(prior.get("model_path") or "")) if prior else version_model
        self._prune_versions(
            self.current_versions_dir,
            keep=(version_model.parent, prior_path.parent),
        )
        return record

    def current_record(self) -> Optional[Dict[str, Any]]:
        return self._role_record(
            role="CURRENT",
            record_path=self.current_record_path,
            alias_model=self.current_model_path,
            versions_dir=self.current_versions_dir,
        )

    def best_record(self) -> Optional[Dict[str, Any]]:
        return self._role_record(
            role="BEST",
            record_path=self.best_record_path,
            alias_model=self.best_model_path,
            versions_dir=self.best_versions_dir,
        )

    def promote_best_if_improved(
        self,
        source_model: Path,
        *,
        evaluation: Mapping[str, Any],
        model_steps: int,
        training_cycle: int,
    ) -> bool:
        prior = self.best_record()
        prior_evaluation = prior.get("evaluation", {}) if isinstance(prior, Mapping) else {}
        if prior is not None:
            candidate_level = int(evaluation.get("adventure_start_level", 0) or 0)
            prior_level = int(prior_evaluation.get("adventure_start_level", 0) or 0)
            if candidate_level <= 0 or candidate_level != prior_level:
                return False
        if prior is not None and compare_evaluations(evaluation, prior_evaluation) <= 0:
            return False
        self.verify_baseline_immutable()
        version_model, source_sha256, metadata_sha256 = self._materialize_version(
            source_model,
            self.best_versions_dir,
            role="BEST",
        )

        # Refresh the conventional alias first.  The record remains the atomic
        # logical pointer, so an interrupted alias update cannot replace BEST.
        atomic_copy_file(version_model, self.best_model_path)
        self._copy_compatibility_metadata(version_model, self.best_model_path.parent)
        record = CheckpointRecord(
            role="BEST",
            model_path=str(version_model),
            sha256=source_sha256,
            model_steps=int(model_steps),
            training_cycle=int(training_cycle),
            saved_at=utc_now_iso(),
            evaluation=dict(evaluation),
        )
        atomic_write_json(
            self.best_record_path,
            {
                **asdict(record),
                "checkpoint_format": 2,
                "metadata_sha256": metadata_sha256,
                "compatibility_alias": str(self.best_model_path),
                "source_current_checkpoint": str(source_model),
                "prior_best_metrics": dict(prior_evaluation),
            },
        )
        prior_path = Path(str(prior.get("model_path") or "")) if prior else version_model
        self._prune_versions(
            self.best_versions_dir,
            keep=(version_model.parent, prior_path.parent),
        )
        return True


class PhaseGate(Protocol):
    def enter_train(self, cycle: int) -> None: ...
    def enter_evaluate(self, cycle: int) -> None: ...
    def shutdown(self) -> None: ...


class _PhaseGateGuard:
    """Make phase shutdown idempotent across setup, loop, and error paths."""

    def __init__(self, target: PhaseGate) -> None:
        self.target = target
        self._shutdown = False

    def enter_train(self, cycle: int) -> None:
        self.target.enter_train(cycle)

    def enter_evaluate(self, cycle: int) -> None:
        self.target.enter_evaluate(cycle)

    def shutdown(self) -> None:
        if not self._shutdown:
            self._shutdown = True
            self.target.shutdown()


TrainCycle = Callable[[Path, int, Path, int, int], Mapping[str, Any]]
EvaluateCheckpoint = Callable[[Path, Path, int, int], Mapping[str, Any]]
StatusSink = Callable[[Mapping[str, Any]], None]


def _read_json_object(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _append_cycle_record_once(path: Path, record: Mapping[str, Any]) -> None:
    """Append one durable cycle row and tolerate a truncated final record."""

    cycle = int(record.get("cycle", 0) or 0)
    seen = False
    if path.is_file():
        try:
            with path.open("r", encoding="utf-8") as existing_handle:
                for line in existing_handle:
                    try:
                        existing = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(existing, dict) and int(existing.get("cycle", 0) or 0) == cycle:
                        seen = True
                        break
        except OSError:
            pass
    if seen:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_separator = False
    if path.is_file() and path.stat().st_size > 0:
        with path.open("rb") as existing_handle:
            existing_handle.seek(-1, os.SEEK_END)
            needs_separator = existing_handle.read(1) != b"\n"
    with path.open("a", encoding="utf-8") as handle:
        if needs_separator:
            handle.write("\n")
        handle.write(json.dumps(dict(record), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _merge_cycle_training_metrics(
    prior: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    target_policy_steps: int,
) -> Dict[str, Any]:
    merged = dict(current)
    delta_fields = (
        "total_environment_actions",
        "viewer_interventions",
        "policy_transitions_collected",
        "bc_demonstrations_added",
        "bc_update_count",
        "streamer_checkpoint_write_count",
    )
    for field in delta_fields:
        merged[field] = int(prior.get(field, 0) or 0) + int(current.get(field, 0) or 0)
    invocation_steps = int(current.get("ppo_policy_timesteps", 0) or 0)
    merged["ppo_policy_timesteps_this_invocation"] = invocation_steps
    merged["ppo_policy_timesteps"] = (
        int(prior.get("ppo_policy_timesteps", 0) or 0) + invocation_steps
    )
    merged["status"] = "trained_complete"
    return merged


def _required_positive_level(metrics: Mapping[str, Any], field: str) -> int:
    try:
        level = int(metrics.get(field, 0) or 0)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"blocked_reason=streamer_{field}_invalid") from exc
    if level <= 0:
        raise RuntimeError(f"blocked_reason=streamer_{field}_missing")
    return level


def _compatible_evaluation_comparison(
    candidate: Mapping[str, Any],
    incumbent: Mapping[str, Any],
) -> Optional[int]:
    """Compare only evaluations that began at the same proven Adventure level."""

    try:
        candidate_level = int(candidate.get("adventure_start_level", 0) or 0)
        incumbent_level = int(incumbent.get("adventure_start_level", 0) or 0)
    except (TypeError, ValueError):
        return None
    if candidate_level <= 0 or candidate_level != incumbent_level:
        return None
    return compare_evaluations(candidate, incumbent)


def run_streamer_cycles(
    *,
    config: Mapping[str, Any],
    train_cycle: TrainCycle,
    evaluate_checkpoint: EvaluateCheckpoint,
    phase_gate: Optional[PhaseGate] = None,
    status_sink: Optional[StatusSink] = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> Dict[str, Any]:
    guarded_gate = _PhaseGateGuard(phase_gate) if phase_gate is not None else None
    try:
        return _run_streamer_cycles_impl(
            config=config,
            train_cycle=train_cycle,
            evaluate_checkpoint=evaluate_checkpoint,
            phase_gate=guarded_gate,
            status_sink=status_sink,
            monotonic=monotonic,
        )
    finally:
        if guarded_gate is not None:
            guarded_gate.shutdown()


def _run_streamer_cycles_impl(
    *,
    config: Mapping[str, Any],
    train_cycle: TrainCycle,
    evaluate_checkpoint: EvaluateCheckpoint,
    phase_gate: Optional[PhaseGate] = None,
    status_sink: Optional[StatusSink] = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> Dict[str, Any]:
    """Run baseline evaluation and repeated Generalist train/eval cycles."""

    experiment_dir = Path(str(config["run_dir"]))
    experiment_dir.mkdir(parents=True, exist_ok=True)
    baseline = Path(str(config["streamer_baseline_checkpoint"]))
    manager = StreamerCheckpointManager(experiment_dir, baseline)
    episodes = int(config.get("streamer_evaluation_episodes", 50) or 50)
    policy_steps = int(config.get("streamer_policy_steps_per_cycle", 25000) or 25000)
    max_cycles = int(config.get("streamer_max_cycles", 0) or 0)
    endurance_hours = float(config.get("streamer_endurance_hours", 0.0) or 0.0)
    deadline = monotonic() + endurance_hours * 3600.0 if endurance_hours > 0.0 else None
    state_path = experiment_dir / "streamer_state.json"
    cycle_log_path = experiment_dir / "streamer_cycles.jsonl"
    baseline_eval_path = experiment_dir / "evaluations" / "baseline" / "evaluation.json"
    baseline_eval_marker_path = baseline_eval_path.parent / "evaluation_state.json"
    baseline_start_level = max(1, int(config.get("adventure_start_level", 1) or 1))
    base_evaluation_protocol = {
        "episodes": episodes,
        "deterministic": True,
        "promotion_order": ["win_rate", "avg_reward"],
        "tie_behavior": "retain_incumbent",
    }
    baseline_evaluation_protocol = {
        **base_evaluation_protocol,
        "adventure_start_level": baseline_start_level,
    }
    baseline_eval = _read_json_object(baseline_eval_path)
    baseline_marker = _read_json_object(baseline_eval_marker_path)
    if baseline_eval_path.is_file() and baseline_eval is None:
        raise RuntimeError("blocked_reason=streamer_baseline_evaluation_invalid")
    if baseline_eval_marker_path.is_file() and baseline_marker is None:
        raise RuntimeError("blocked_reason=streamer_baseline_evaluation_marker_invalid")
    if baseline_eval is not None and (
        str(baseline_eval.get("baseline_sha256") or "") != manager.baseline_sha256
        or baseline_eval.get("evaluation_protocol") != baseline_evaluation_protocol
    ):
        raise RuntimeError("blocked_reason=streamer_baseline_evaluation_protocol_changed")
    if baseline_eval is not None:
        if baseline_marker is None:
            raise RuntimeError("blocked_reason=streamer_baseline_evaluation_marker_missing")
        if (
            str(baseline_marker.get("baseline_sha256") or "") != manager.baseline_sha256
            or baseline_marker.get("evaluation_protocol") != baseline_evaluation_protocol
            or str(baseline_marker.get("status") or "") not in {"in_progress", "complete"}
        ):
            raise RuntimeError("blocked_reason=streamer_baseline_evaluation_marker_mismatch")
        if str(baseline_marker.get("status") or "") == "in_progress":
            atomic_write_json(
                baseline_eval_marker_path,
                {
                    "status": "complete",
                    "baseline_sha256": manager.baseline_sha256,
                    "evaluation_protocol": baseline_evaluation_protocol,
                    "completed_at": utc_now_iso(),
                    "recovered_from_complete_evaluation": True,
                },
            )
    if baseline_eval is None:
        if baseline_marker is not None:
            raise RuntimeError("blocked_reason=streamer_baseline_evaluation_interrupted")
        atomic_write_json(
            baseline_eval_marker_path,
            {
                "status": "in_progress",
                "baseline_sha256": manager.baseline_sha256,
                "evaluation_protocol": baseline_evaluation_protocol,
                "started_at": utc_now_iso(),
            },
        )
        if phase_gate is not None:
            phase_gate.enter_evaluate(0)
        if status_sink is not None:
            status_sink(
                {
                    "status": "running",
                    "mode": EVALUATE,
                    "streamer_phase": EVALUATE,
                    "evaluation_role": "BASELINE",
                    "completed_cycle": 0,
                    "viewer_command_queue_depth": 0,
                    "evaluation_chat_control": False,
                }
            )
        baseline_eval_dir = baseline_eval_path.parent
        baseline_eval = dict(
            evaluate_checkpoint(
                baseline,
                baseline_eval_dir,
                episodes,
                baseline_start_level,
            )
        )
        baseline_eval.setdefault("adventure_start_level", baseline_start_level)
        if int(baseline_eval.get("adventure_start_level", 0) or 0) != baseline_start_level:
            raise RuntimeError("blocked_reason=streamer_baseline_evaluation_level_mismatch")
        _required_positive_level(baseline_eval, "next_adventure_level")
        baseline_eval["baseline_sha256"] = manager.baseline_sha256
        baseline_eval["evaluation_protocol"] = baseline_evaluation_protocol
        atomic_write_json(baseline_eval_path, baseline_eval)
        atomic_write_json(
            baseline_eval_marker_path,
            {
                "status": "complete",
                "baseline_sha256": manager.baseline_sha256,
                "evaluation_protocol": baseline_evaluation_protocol,
                "completed_at": utc_now_iso(),
            },
        )
    if int(baseline_eval.get("adventure_start_level", 0) or 0) != baseline_start_level:
        raise RuntimeError("blocked_reason=streamer_baseline_evaluation_level_mismatch")
    _required_positive_level(baseline_eval, "next_adventure_level")
    existing_state_value = _read_json_object(state_path)
    if state_path.is_file() and existing_state_value is None:
        raise RuntimeError("blocked_reason=streamer_state_invalid")
    existing_state = existing_state_value or {}
    if isinstance(existing_state.get("last_cycle"), Mapping):
        _append_cycle_record_once(cycle_log_path, existing_state["last_cycle"])
    completed_cycle = max(0, int(existing_state.get("completed_cycle", 0) or 0))
    in_progress_cycle = max(0, int(existing_state.get("in_progress_cycle", 0) or 0))
    if in_progress_cycle and in_progress_cycle != completed_cycle + 1:
        raise RuntimeError("blocked_reason=streamer_state_cycle_mismatch")
    current_record = manager.current_record()
    current_record_cycle = int(current_record.get("training_cycle", 0) or 0) if current_record else 0
    best_record = manager.best_record()
    established_experiment = bool(
        state_path.is_file()
        or current_record is not None
        or cycle_log_path.is_file()
    )
    allowed_current_cycles = {completed_cycle}
    if in_progress_cycle:
        allowed_current_cycles.add(in_progress_cycle)
    if current_record is not None and current_record_cycle not in allowed_current_cycles:
        raise RuntimeError("blocked_reason=streamer_current_cycle_mismatch")
    if completed_cycle > 0 and current_record is None:
        raise RuntimeError("blocked_reason=streamer_current_missing_for_completed_state")
    if not existing_state and current_record is not None:
        raise RuntimeError("blocked_reason=streamer_state_missing_for_current_checkpoint")
    if completed_cycle > 0 and current_record is not None and current_record_cycle == completed_cycle:
        last_cycle = existing_state.get("last_cycle")
        if (
            not isinstance(last_cycle, Mapping)
            or int(last_cycle.get("cycle", 0) or 0) != completed_cycle
        ):
            raise RuntimeError("blocked_reason=streamer_completed_cycle_evidence_missing")
        last_evaluation = last_cycle.get("evaluation")
        if not isinstance(last_evaluation, Mapping):
            raise RuntimeError("blocked_reason=streamer_completed_cycle_evaluation_missing")
        expected_sha256 = str(last_evaluation.get("current_sha256") or "")
        actual_sha256 = str(current_record.get("sha256") or "")
        if not expected_sha256 or expected_sha256 != actual_sha256:
            raise RuntimeError("blocked_reason=streamer_current_state_hash_mismatch")
        try:
            expected_model_steps = int(existing_state.get("current_model_steps", 0) or 0)
            actual_model_steps = int(current_record.get("model_steps", 0) or 0)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("blocked_reason=streamer_current_state_steps_invalid") from exc
        if expected_model_steps <= 0 or expected_model_steps != actual_model_steps:
            raise RuntimeError("blocked_reason=streamer_current_state_steps_mismatch")
    if best_record is None:
        if established_experiment:
            raise RuntimeError("blocked_reason=streamer_best_missing_for_existing_state")
        if not manager.promote_best_if_improved(
            baseline,
            evaluation=baseline_eval,
            model_steps=int(_summary(baseline_eval).get("model_steps", 0) or 0),
            training_cycle=0,
        ):
            raise RuntimeError("blocked_reason=streamer_baseline_best_bootstrap_failed")

    start_model = manager.current_model_path if current_record is not None else baseline
    previous_eval: Mapping[str, Any] = existing_state.get("current_evaluation", baseline_eval)
    cumulative_policy_steps = int(existing_state.get("ppo_policy_timesteps", 0) or 0)
    next_adventure_level = _required_positive_level(
        existing_state if existing_state else baseline_eval,
        "next_adventure_level",
    )
    result: Dict[str, Any] = dict(existing_state)

    try:
        while max_cycles == 0 or completed_cycle < max_cycles:
            if deadline is not None and monotonic() >= deadline:
                break
            cycle = in_progress_cycle or (completed_cycle + 1)
            prior_training: Mapping[str, Any] = {}
            train_start_level = int(next_adventure_level)
            if current_record is not None and current_record_cycle == cycle:
                candidate = current_record.get("training_metrics")
                prior_training = candidate if isinstance(candidate, Mapping) else {}
                start_model = manager.current_model_path
                train_start_level = _required_positive_level(
                    prior_training,
                    "next_adventure_level",
                )
            completed_in_cycle = int(prior_training.get("ppo_policy_timesteps", 0) or 0)
            if completed_in_cycle < 0 or completed_in_cycle > policy_steps:
                raise RuntimeError("blocked_reason=streamer_recovery_policy_steps_invalid")
            remaining_policy_steps = policy_steps - completed_in_cycle
            if not in_progress_cycle:
                in_progress_cycle = cycle
                cycle_start_state = {
                    **dict(result),
                    "status": "running",
                    "mode": STREAM_TRAIN,
                    "completed_cycle": completed_cycle,
                    "in_progress_cycle": cycle,
                    "ppo_policy_timesteps": cumulative_policy_steps,
                    "current_evaluation": dict(previous_eval),
                    "baseline_evaluation": baseline_eval,
                    "baseline_checkpoint": str(baseline),
                    "baseline_sha256": manager.baseline_sha256,
                    "cycle_target_policy_steps": policy_steps,
                    "cycle_policy_steps_completed": completed_in_cycle,
                    "cycle_start_adventure_level": train_start_level,
                    "next_adventure_level": train_start_level,
                    "cycle_started_at": utc_now_iso(),
                }
                atomic_write_json(state_path, cycle_start_state)
                result = cycle_start_state
            if phase_gate is not None:
                phase_gate.enter_train(cycle)
            best_before_training = manager.best_record() or {}
            if status_sink is not None:
                status_sink(
                    {
                        "status": "running",
                        "mode": STREAM_TRAIN,
                        "streamer_phase": STREAM_TRAIN,
                        "completed_cycle": completed_cycle,
                        "current_cycle": cycle,
                        "current_checkpoint": str(start_model),
                        "baseline_checkpoint": str(baseline),
                        "baseline_sha256": manager.baseline_sha256,
                        "baseline_evaluation": baseline_eval,
                        "best_checkpoint": str(manager.best_model_path),
                        "best_evaluation": best_before_training.get("evaluation", {}),
                        "best_model_steps": int(best_before_training.get("model_steps", 0) or 0),
                        "evaluation_chat_control": False,
                        "next_evaluation_policy_steps": remaining_policy_steps,
                        "next_adventure_level": train_start_level,
                        "bc_updates_enabled": bool(
                            config.get("streamer_bc_enabled", True)
                        ),
                        "ppo_updates_enabled": True,
                    }
                )
            cycle_dir = experiment_dir / "cycles" / f"cycle_{cycle:06d}"
            if remaining_policy_steps > 0:
                invocation_training = dict(
                    train_cycle(
                        start_model,
                        cycle,
                        cycle_dir / "train",
                        remaining_policy_steps,
                        train_start_level,
                    )
                )
                source_model = Path(str(invocation_training.get("model_path") or ""))
                if not source_model.is_file():
                    raise RuntimeError(
                        f"Streamer training cycle did not produce a model: {source_model}"
                    )
                training = _merge_cycle_training_metrics(
                    prior_training,
                    invocation_training,
                    target_policy_steps=policy_steps,
                )
                evaluation_start_level = _required_positive_level(
                    training,
                    "next_adventure_level",
                )
                model_steps = int(
                    training.get("model_steps", cumulative_policy_steps + policy_steps) or 0
                )
                manager.save_current(
                    source_model,
                    model_steps=model_steps,
                    training_cycle=cycle,
                    training_metrics=training,
                )
                current_record = manager.current_record()
                current_record_cycle = cycle
                try:
                    source_model.resolve().relative_to((cycle_dir / "train").resolve())
                except ValueError:
                    pass
                else:
                    # CURRENT's immutable version is now authoritative; retaining
                    # another full model in every cycle would grow disk forever.
                    source_model.unlink(missing_ok=True)
            else:
                if current_record is None or current_record_cycle != cycle:
                    raise RuntimeError("blocked_reason=streamer_recovery_current_missing")
                if str(prior_training.get("status") or "") != "trained_complete":
                    raise RuntimeError(
                        "blocked_reason=streamer_recovery_phase_handoff_unproven"
                    )
                training = dict(prior_training)
                training["ppo_policy_timesteps"] = policy_steps
                model_steps = int(current_record.get("model_steps", 0) or 0)
                evaluation_start_level = _required_positive_level(
                    training,
                    "next_adventure_level",
                )
            if int(training.get("ppo_policy_timesteps", 0) or 0) != policy_steps:
                raise RuntimeError(
                    "blocked_reason=streamer_policy_step_target_mismatch: "
                    f"expected={policy_steps} actual={training.get('ppo_policy_timesteps', 0)}"
                )

            if phase_gate is not None:
                phase_gate.enter_evaluate(cycle)
            if status_sink is not None:
                status_sink(
                    {
                        "status": "running",
                        "mode": EVALUATE,
                        "streamer_phase": EVALUATE,
                        "completed_cycle": completed_cycle,
                        "current_cycle": cycle,
                        "current_checkpoint": str(manager.current_model_path),
                        "current_model_steps": model_steps,
                        "baseline_evaluation": baseline_eval,
                        "best_evaluation": best_before_training.get("evaluation", {}),
                        "best_model_steps": int(best_before_training.get("model_steps", 0) or 0),
                        "viewer_command_queue_depth": 0,
                        "evaluation_chat_control": False,
                        "bc_updates_enabled": False,
                        "ppo_updates_enabled": False,
                        "next_adventure_level": evaluation_start_level,
                    }
                )
            evaluation_path = cycle_dir / "evaluation" / "evaluation.json"
            evaluation_value = _read_json_object(evaluation_path)
            if evaluation_path.is_file() and evaluation_value is None:
                raise RuntimeError("blocked_reason=streamer_cycle_evaluation_invalid")
            evaluation = evaluation_value or {}
            current_sha256 = str((current_record or {}).get("sha256") or "")
            evaluation_protocol = {
                **base_evaluation_protocol,
                "adventure_start_level": evaluation_start_level,
            }
            evaluation_marker_path = evaluation_path.parent / "evaluation_state.json"
            evaluation_marker = _read_json_object(evaluation_marker_path)
            if evaluation_marker_path.is_file() and evaluation_marker is None:
                raise RuntimeError("blocked_reason=streamer_cycle_evaluation_marker_invalid")
            marker_identity = {
                "streamer_cycle": cycle,
                "current_sha256": current_sha256,
                "evaluation_protocol": evaluation_protocol,
            }
            if evaluation_path.is_file():
                if (
                    int(evaluation.get("streamer_cycle", 0) or 0) != cycle
                    or str(evaluation.get("current_sha256") or "") != current_sha256
                    or evaluation.get("evaluation_protocol") != evaluation_protocol
                ):
                    raise RuntimeError(
                        "blocked_reason=streamer_cycle_evaluation_protocol_changed"
                    )
                if evaluation_marker is None:
                    raise RuntimeError("blocked_reason=streamer_cycle_evaluation_marker_missing")
                if (
                    any(evaluation_marker.get(key) != value for key, value in marker_identity.items())
                    or str(evaluation_marker.get("status") or "") not in {"in_progress", "complete"}
                ):
                    raise RuntimeError("blocked_reason=streamer_cycle_evaluation_marker_mismatch")
                if str(evaluation_marker.get("status") or "") == "in_progress":
                    atomic_write_json(
                        evaluation_marker_path,
                        {
                            **marker_identity,
                            "status": "complete",
                            "completed_at": utc_now_iso(),
                            "recovered_from_complete_evaluation": True,
                        },
                    )
            else:
                if evaluation_marker is not None:
                    raise RuntimeError(
                        "blocked_reason=streamer_cycle_evaluation_interrupted: "
                        f"marker={evaluation_marker_path} "
                        f"status={evaluation_marker.get('status', '')!r} "
                        f"started_at={evaluation_marker.get('started_at', '')!r} "
                        f"error_type={evaluation_marker.get('error_type', '')!r} "
                        f"error={evaluation_marker.get('error', '')!r}; "
                        f"missing_result={evaluation_path}"
                    )
                evaluation_started_at = utc_now_iso()
                atomic_write_json(
                    evaluation_marker_path,
                    {
                        **marker_identity,
                        "status": "in_progress",
                        "started_at": evaluation_started_at,
                    },
                )
                try:
                    evaluation = dict(
                        evaluate_checkpoint(
                            manager.current_model_path,
                            cycle_dir / "evaluation",
                            episodes,
                            evaluation_start_level,
                        )
                    )
                    evaluation.setdefault("adventure_start_level", evaluation_start_level)
                    if int(evaluation.get("adventure_start_level", 0) or 0) != evaluation_start_level:
                        raise RuntimeError("blocked_reason=streamer_cycle_evaluation_level_mismatch")
                    _required_positive_level(evaluation, "next_adventure_level")
                    evaluation["streamer_cycle"] = cycle
                    evaluation["current_sha256"] = current_sha256
                    evaluation["evaluation_protocol"] = evaluation_protocol
                    atomic_write_json(evaluation_path, evaluation)
                    atomic_write_json(
                        evaluation_marker_path,
                        {
                            **marker_identity,
                            "status": "complete",
                            "completed_at": utc_now_iso(),
                        },
                    )
                except BaseException as exc:
                    # Keep the transaction fence in place, but preserve the
                    # first failure so the next invocation can identify why
                    # the result file was never committed.
                    try:
                        atomic_write_json(
                            evaluation_marker_path,
                            {
                                **marker_identity,
                                "status": "in_progress",
                                "started_at": evaluation_started_at,
                                "interrupted_at": utc_now_iso(),
                                "error_type": type(exc).__name__,
                                "error": str(exc)[:1000],
                            },
                        )
                    except Exception:
                        pass
                    raise
            next_adventure_level = _required_positive_level(
                evaluation,
                "next_adventure_level",
            )
            existing_best = manager.best_record() or {}
            already_promoted = bool(
                int(existing_best.get("training_cycle", -1) or -1) == cycle
                and str(existing_best.get("sha256") or "") == current_sha256
            )
            comparison_to_existing_best = _compatible_evaluation_comparison(
                evaluation,
                existing_best.get("evaluation", {}),
            )
            promoted = already_promoted
            if not promoted and comparison_to_existing_best is not None and comparison_to_existing_best > 0:
                promoted = manager.promote_best_if_improved(
                    manager.current_model_path,
                    evaluation=evaluation,
                    model_steps=model_steps,
                    training_cycle=cycle,
                )
            best_record = manager.best_record() or {}
            comparison_to_baseline = _compatible_evaluation_comparison(
                evaluation,
                baseline_eval,
            )
            comparison_to_previous = _compatible_evaluation_comparison(
                evaluation,
                previous_eval,
            )
            cycle_record: Dict[str, Any] = {
                "cycle": cycle,
                "phase": EVALUATE,
                "started_from": str(start_model),
                "current_checkpoint": str(manager.current_model_path),
                "best_checkpoint": str(manager.best_model_path),
                "training": training,
                "evaluation": evaluation,
                "comparison_to_baseline": (
                    comparison_to_baseline if comparison_to_baseline is not None else "UNKNOWN"
                ),
                "comparison_to_previous_current": (
                    comparison_to_previous if comparison_to_previous is not None else "UNKNOWN"
                ),
                "comparison_to_best_before_promotion": (
                    comparison_to_existing_best
                    if comparison_to_existing_best is not None
                    else "UNKNOWN"
                ),
                "comparison_protocol_compatible": {
                    "baseline": comparison_to_baseline is not None,
                    "previous_current": comparison_to_previous is not None,
                    "best": comparison_to_existing_best is not None,
                },
                "best_promoted": promoted,
                "baseline_evaluation": baseline_eval,
                "best_evaluation": best_record.get("evaluation", {}),
                "ppo_policy_timesteps": cumulative_policy_steps + policy_steps,
                "train_start_adventure_level": train_start_level,
                "evaluation_start_adventure_level": evaluation_start_level,
                "next_adventure_level": next_adventure_level,
                "completed_at": utc_now_iso(),
            }
            completed_cycle = cycle
            in_progress_cycle = 0
            cumulative_policy_steps += policy_steps
            previous_eval = evaluation
            start_model = manager.current_model_path
            result = {
                "status": "running",
                "mode": STREAM_TRAIN,
                "completed_cycle": completed_cycle,
                "in_progress_cycle": 0,
                "ppo_policy_timesteps": cumulative_policy_steps,
                "current_model_steps": model_steps,
                "current_checkpoint": str(manager.current_model_path),
                "best_checkpoint": str(manager.best_model_path),
                "baseline_checkpoint": str(baseline),
                "baseline_sha256": manager.baseline_sha256,
                "baseline_evaluation": baseline_eval,
                "current_evaluation": evaluation,
                "best_evaluation": best_record.get("evaluation", {}),
                "best_model_steps": int(best_record.get("model_steps", 0) or 0),
                "next_adventure_level": next_adventure_level,
                "last_cycle": cycle_record,
            }
            atomic_write_json(state_path, result)
            _append_cycle_record_once(cycle_log_path, cycle_record)
            if status_sink is not None:
                status_sink(result)
            manager.verify_baseline_immutable()
    finally:
        if phase_gate is not None:
            phase_gate.shutdown()

    result = dict(result)
    result["status"] = "complete"
    result["stop_reason"] = (
        "endurance_deadline_reached"
        if deadline is not None and monotonic() >= deadline
        else "max_cycles_reached"
        if max_cycles > 0 and completed_cycle >= max_cycles
        else "stopped"
    )
    atomic_write_json(state_path, result)
    if status_sink is not None:
        status_sink(result)
    return result
