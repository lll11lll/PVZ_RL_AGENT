"""Bridge-free deterministic endurance simulator for Streamer Mode V1.

This executable deliberately uses the production deterministic source, strict
parser, bounded viewer-command queue/controller, and demonstration buffer.  It
does not open a network connection or instantiate the game environment.  A
wall-clock duration run is suitable for observing long-lived Python resource
behavior; ``--message-count`` provides the same deterministic fault pattern in
a fast CI-friendly form.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import time
import tracemalloc
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence
import uuid

import numpy as np

from pvzrl_action_space import (
    ADVENTURE_IDENTITY_ACTION_COUNT,
    CELLS_PER_SLOT,
    build_action_space_spec,
)
from pvzrl_demonstrations import DemonstrationBuffer
from pvzrl_observation_layout import build_observation_layout
from pvzrl_stream_commands import (
    BoundedViewerCommandQueue,
    ViewerCommand,
    ViewerCommandController,
    ViewerCommandOutcome,
)
from pvzrl_streamer_source import DeterministicStreamCommandSource


SOAK_REPORT_VERSION = 1
DEFAULT_QUEUE_CAPACITY = 256
DEFAULT_DEMO_CAPACITY = 4_096
OBSERVATION_SHAPE = build_observation_layout(build_action_space_spec()).shape


ACTION_COUNT = ADVENTURE_IDENTITY_ACTION_COUNT
_VIEWER_HASH_SECRET = b"pvzrl-streamer-soak-local-secret-v1"


class _VirtualClock:
    def __init__(self, value: float = 1_000.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return float(self.value)

    def advance(self, seconds: float) -> float:
        self.value += float(seconds)
        return float(self.value)


@dataclass(frozen=True, slots=True)
class _SyntheticResolution:
    legal: bool
    classification: str
    reason: str
    action_id: Optional[int]
    frame_identity: str


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _message_text(sequence: int, rng: random.Random) -> tuple[str, str]:
    """Return a deterministic command and its intended simulation class."""

    row = 1 + rng.randrange(6)
    column = 1 + rng.randrange(10)
    if sequence % 19 == 0:
        return "!unknown command", "parse_rejected"
    if sequence % 17 == 0:
        return f"!slot 1 {row} {column}", "expired"
    if sequence % 13 == 0:
        return f"!slot 14 {row} {column}", "unresolvable"
    if sequence % 11 == 0:
        return f"!slot 13 {row} {column}", "currently_illegal"
    if sequence % 5 == 0:
        return f"!plant Peashooter {row} {column}", "resolved"
    if sequence % 7 == 0:
        return f"!fuse {row} {column}", "resolved"
    slot = 1 + rng.randrange(12)
    return f"!slot {slot} {row} {column}", "resolved"


def _resolve_command(command: ViewerCommand, *, frame: int) -> _SyntheticResolution:
    if command.seed_slot == 13:
        return _SyntheticResolution(
            legal=False,
            classification="unresolvable",
            reason="synthetic_slot_unavailable",
            action_id=None,
            frame_identity=f"soak:{frame}",
        )
    if command.seed_slot == 12:
        return _SyntheticResolution(
            legal=False,
            classification="currently_illegal",
            reason="synthetic_cooldown",
            action_id=None,
            frame_identity=f"soak:{frame}",
        )
    slot = int(command.seed_slot or 0)
    action_id = 1 + slot * CELLS_PER_SLOT + int(command.row) * 10 + int(command.column)
    return _SyntheticResolution(
        legal=True,
        classification="resolved",
        reason="",
        action_id=action_id,
        frame_identity=f"soak:{frame}",
    )


def _record_outcomes(outcomes: Iterable[ViewerCommandOutcome], counters: Counter[str]) -> None:
    for outcome in outcomes:
        counters[str(outcome.status or "unknown")] += 1


def evaluate_streamer_soak_invariants(report: Mapping[str, Any]) -> tuple[str, ...]:
    """Return stable invariant failure codes for a completed soak report."""

    failures = []
    queue = dict(report.get("queue") or {})
    demonstrations = dict(report.get("demonstrations") or {})
    safety = dict(report.get("safety") or {})
    messages = dict(report.get("messages") or {})
    errors = dict(report.get("errors") or {})

    queue_capacity = int(queue.get("capacity", 0))
    if queue_capacity <= 0 or int(queue.get("maximum_depth", -1)) > queue_capacity:
        failures.append("queue_capacity_exceeded")
    if int(queue.get("final_depth", -1)) != 0:
        failures.append("queue_not_empty_after_shutdown_gate")
    source_capacity = int(queue.get("source_capacity", 0))
    if source_capacity <= 0 or int(queue.get("source_maximum_depth", -1)) > source_capacity:
        failures.append("source_capacity_exceeded")

    demo_capacity = int(demonstrations.get("capacity", 0))
    demo_size = int(demonstrations.get("size", -1))
    demo_added = int(demonstrations.get("total_added", -1))
    demo_evicted = int(demonstrations.get("total_evicted", -1))
    if demo_capacity <= 0 or not 0 <= demo_size <= demo_capacity:
        failures.append("demonstration_capacity_exceeded")
    if demo_added - demo_evicted != demo_size:
        failures.append("demonstration_accounting_mismatch")

    if int(safety.get("stale_selections", -1)) != 0:
        failures.append("stale_command_selected")
    if int(safety.get("dispatches_while_gated", -1)) != 0:
        failures.append("command_dispatched_while_gated")
    if int(safety.get("phase_leaks", -1)) != 0:
        failures.append("phase_command_leak")
    if int(safety.get("gate_epoch_regressions", -1)) != 0:
        failures.append("source_gate_epoch_regressed")
    if int(errors.get("count", -1)) != 0:
        failures.append("runtime_errors_observed")
    if int(messages.get("selected", -1)) > int(messages.get("opportunities", -1)):
        failures.append("more_than_one_selection_per_opportunity")
    if int(messages.get("received", -1)) < int(messages.get("source_enqueued", -1)):
        failures.append("source_receive_accounting_invalid")
    return tuple(failures)


def run_streamer_soak(
    *,
    duration_hours: Optional[float] = None,
    message_count: Optional[int] = None,
    queue_capacity: int = DEFAULT_QUEUE_CAPACITY,
    demo_capacity: int = DEFAULT_DEMO_CAPACITY,
    seed: int = 1_337,
) -> Dict[str, Any]:
    """Run the bridge-free Streamer simulation and return a JSON-safe report."""

    if (duration_hours is None) == (message_count is None):
        raise ValueError("exactly one of duration_hours or message_count must be supplied")
    if duration_hours is not None and (
        not math.isfinite(float(duration_hours)) or float(duration_hours) <= 0.0
    ):
        raise ValueError("duration_hours must be finite and positive")
    if message_count is not None and int(message_count) <= 0:
        raise ValueError("message_count must be positive")
    if int(queue_capacity) <= 0:
        raise ValueError("queue_capacity must be positive")
    if int(demo_capacity) <= 0:
        raise ValueError("demo_capacity must be positive")

    clock = _VirtualClock()
    rng = random.Random(int(seed))
    source = DeterministicStreamCommandSource(
        (),
        viewer_hash_secret=_VIEWER_HASH_SECRET,
        queue_capacity=int(queue_capacity),
        accepting=False,
        max_command_chars=256,
        monotonic=clock,
    )
    queue = BoundedViewerCommandQueue(
        capacity=int(queue_capacity),
        ttl_seconds=3.0,
        dedupe_capacity=max(1_024, int(queue_capacity) * 8),
        monotonic=clock,
    )
    controller = ViewerCommandController(
        source=source,
        queue=queue,
        opportunity_interval_seconds=1.0,
        max_poll_messages=int(queue_capacity),
        monotonic=clock,
    )
    demonstrations = DemonstrationBuffer(
        int(demo_capacity),
        observation_shape=OBSERVATION_SHAPE,
        action_count=ACTION_COUNT,
    )
    observation = np.zeros(OBSERVATION_SHAPE, dtype=np.float32)
    action_mask = np.ones((ACTION_COUNT,), dtype=bool)

    outcomes: Counter[str] = Counter()
    safety: Counter[str] = Counter()
    runtime_errors: Counter[str] = Counter()
    maximum_queue_depth = 0
    maximum_source_depth = 0
    phase_transitions = 0
    reconnect_cycles = 0
    gate_epoch_regressions = 0
    maximum_gate_epoch = 0
    previous_gate_epoch = 0
    attempted = 0
    phase_probe_attempted = 0
    frame = 0
    batch_size = max(4, min(32, int(queue_capacity) + 2))
    phase_period = max(32, int(queue_capacity) * 4)
    reconnect_period = max(47, int(queue_capacity) * 6)

    already_tracing = tracemalloc.is_tracing()
    if not already_tracing:
        tracemalloc.start()
    memory_start_current, _ = tracemalloc.get_traced_memory()
    wall_started = time.perf_counter()
    deadline = (
        wall_started + float(duration_hours) * 60.0 * 60.0
        if duration_hours is not None
        else None
    )

    def observe_bounds() -> None:
        nonlocal maximum_queue_depth, maximum_source_depth
        nonlocal previous_gate_epoch, maximum_gate_epoch, gate_epoch_regressions
        maximum_queue_depth = max(maximum_queue_depth, len(queue))
        diagnostics = source.get_diagnostics()
        source_depth = int(diagnostics.get("stream_source_queue_depth", 0))
        maximum_source_depth = max(maximum_source_depth, source_depth)
        gate_epoch = int(diagnostics.get("stream_source_gate_epoch", 0))
        if gate_epoch < previous_gate_epoch:
            gate_epoch_regressions += 1
        previous_gate_epoch = gate_epoch
        maximum_gate_epoch = max(maximum_gate_epoch, gate_epoch)

    try:
        controller.begin_phase("stream_train", accepting=True, now=clock())
        phase_transitions += 1
        controller.start()
        observe_bounds()

        while True:
            if message_count is not None and attempted >= int(message_count):
                break
            if deadline is not None and attempted > 0 and time.perf_counter() >= deadline:
                break

            remaining = (
                batch_size
                if message_count is None
                else min(batch_size, int(message_count) - attempted)
            )
            for _ in range(remaining):
                sequence = attempted + 1
                command_text, command_class = _message_text(sequence, rng)
                received = clock() - 4.0 if command_class == "expired" else clock()
                source.submit(
                    command_text,
                    local_viewer_id=f"synthetic-viewer-{sequence % 97}",
                    delivery_id=f"soak-delivery-{sequence}",
                    event_id=f"soak-event-{sequence}",
                    received_monotonic=received,
                )
                attempted += 1
            observe_bounds()

            frame += 1
            clock.advance(1.0)
            accepting_at_dispatch = bool(
                source.get_diagnostics().get("stream_source_accepting", False)
            )

            def execute(selected: Any, resolution: _SyntheticResolution) -> str:
                if not accepting_at_dispatch:
                    safety["dispatches_while_gated"] += 1
                if selected.phase_generation != queue.phase_generation:
                    safety["stale_selections"] += 1
                demonstrations.add_if_eligible(
                    observation,
                    action_mask,
                    {
                        "viewer_controlled": True,
                        "demo_eligible": True,
                        "execution_succeeded": True,
                        "executed_action": int(resolution.action_id),
                        "behavior_source": "synthetic_streamer_soak",
                        "execution_status": "success",
                        "demonstration": {
                            "phase_generation": int(selected.phase_generation),
                            "frame": int(frame),
                        },
                    },
                )
                return "synthetic_success"

            tick = controller.tick(
                lambda command: _resolve_command(command, frame=frame),
                execute_command=execute,
                now=clock(),
            )
            _record_outcomes(tick.outcomes, outcomes)
            observe_bounds()

            if attempted % reconnect_period < batch_size:
                queued_generation = queue.phase_generation
                source.set_accepting(False, reason="synthetic_disconnect")
                reconnect_cycles += 1
                observe_bounds()
                controller.poll_source(now=clock())
                if len(queue) != 0:
                    safety["phase_leaks"] += 1
                if queue.phase_generation <= queued_generation:
                    safety["phase_leaks"] += 1
                source.set_accepting(True, reason="synthetic_reconnect")
                controller.poll_source(now=clock())
                observe_bounds()

            if attempted % phase_period < batch_size:
                controller.begin_phase("evaluate", accepting=False, now=clock())
                phase_transitions += 1
                observe_bounds()
                probe_sequence = attempted + 1_000_000_000
                phase_probe_attempted += 1
                if source.submit(
                    "!slot 1 1 1",
                    local_viewer_id="synthetic-phase-probe",
                    delivery_id=f"soak-delivery-{probe_sequence}",
                    event_id=f"soak-event-{probe_sequence}",
                    received_monotonic=clock(),
                ):
                    safety["phase_leaks"] += 1
                gated_tick = controller.tick(
                    lambda command: _resolve_command(command, frame=frame),
                    now=clock(),
                )
                _record_outcomes(gated_tick.outcomes, outcomes)
                if gated_tick.opportunity_opened or gated_tick.selected is not None:
                    safety["dispatches_while_gated"] += 1
                controller.begin_phase("stream_train", accepting=True, now=clock())
                phase_transitions += 1
                observe_bounds()
    except Exception as exc:  # Produce an inspectable failed report, without sensitive text.
        runtime_errors[type(exc).__name__] += 1
    finally:
        try:
            controller.begin_phase("shutdown", accepting=False, now=clock())
            phase_transitions += 1
        except Exception as exc:
            runtime_errors[type(exc).__name__] += 1
        try:
            if not controller.close(timeout_seconds=1.0):
                runtime_errors["source_shutdown_timeout"] += 1
        except Exception as exc:
            runtime_errors[type(exc).__name__] += 1
        observe_bounds()

    elapsed = max(0.0, time.perf_counter() - wall_started)
    memory_current, memory_peak = tracemalloc.get_traced_memory()
    if not already_tracing:
        tracemalloc.stop()

    source_diagnostics = source.get_diagnostics()
    queue_snapshot = queue.snapshot()
    parsed = (
        outcomes["accepted"]
        + outcomes["capacity_rejected"]
        + outcomes["phase_rejected"]
        + outcomes["duplicate"]
    )
    selected = outcomes["selected"] + outcomes["dispatched"]
    rejected_statuses = (
        "parse_rejected",
        "capacity_rejected",
        "phase_rejected",
        "source_message_rejected",
        "currently_illegal",
        "unresolvable",
        "stale",
        "phase_stale",
        "source_stale",
        "source_error",
        "execution_error",
    )
    rejected = sum(outcomes[status] for status in rejected_statuses)
    source_received = int(
        source_diagnostics.get("stream_source_local_records_submitted", 0)
    )
    source_enqueued = int(
        source_diagnostics.get("stream_source_local_records_enqueued", 0)
    )
    source_rejected = max(0, source_received - source_enqueued)
    report: Dict[str, Any] = {
        "format_version": SOAK_REPORT_VERSION,
        "ok": False,
        "mode": "duration" if duration_hours is not None else "message_count",
        "seed": int(seed),
        "requested": {
            "duration_hours": float(duration_hours) if duration_hours is not None else None,
            "message_count": int(message_count) if message_count is not None else None,
        },
        "elapsed_seconds": elapsed,
        "throughput_messages_per_second": source_received / elapsed if elapsed > 0.0 else 0.0,
        "messages": {
            "attempted": int(attempted),
            "phase_probe_attempted": int(phase_probe_attempted),
            "received": source_received,
            "source_enqueued": source_enqueued,
            "source_rejected": source_rejected,
            "parsed": int(parsed),
            "parse_rejected": int(outcomes["parse_rejected"]),
            "selected": int(selected),
            "rejected": int(rejected + source_rejected),
            "expired": int(outcomes["expired"]),
            "currently_illegal": int(outcomes["currently_illegal"]),
            "unresolvable": int(outcomes["unresolvable"]),
            "opportunities": int(frame),
            "outcomes": dict(sorted(outcomes.items())),
        },
        "queue": {
            "capacity": int(queue_capacity),
            "maximum_depth": int(maximum_queue_depth),
            "final_depth": int(queue_snapshot.depth),
            "source_capacity": int(queue_capacity),
            "source_maximum_depth": int(maximum_source_depth),
            "source_final_depth": int(
                source_diagnostics.get("stream_source_queue_depth", 0)
            ),
            "counters": dict(queue_snapshot.counters),
        },
        "phases": {
            "transitions": int(phase_transitions),
            "controller_generation": int(queue_snapshot.phase_generation),
            "reconnect_cycles": int(reconnect_cycles),
            "source_gate_epoch": int(
                source_diagnostics.get("stream_source_gate_epoch", 0)
            ),
            "maximum_source_gate_epoch": int(maximum_gate_epoch),
        },
        "demonstrations": {
            "capacity": int(demo_capacity),
            "size": int(len(demonstrations)),
            "total_added": int(demonstrations.total_added),
            "total_evicted": int(demonstrations.total_evicted),
        },
        "memory": {
            "current_bytes": max(0, int(memory_current - memory_start_current)),
            "peak_bytes": max(0, int(memory_peak - memory_start_current)),
            "tracemalloc_already_active": bool(already_tracing),
        },
        "safety": {
            "stale_selections": int(safety["stale_selections"]),
            "dispatches_while_gated": int(safety["dispatches_while_gated"]),
            "phase_leaks": int(safety["phase_leaks"]),
            "gate_epoch_regressions": int(gate_epoch_regressions),
        },
        "errors": {
            "count": int(sum(runtime_errors.values())),
            "types": dict(sorted(runtime_errors.items())),
        },
        "source_diagnostics": source_diagnostics,
    }
    failures = evaluate_streamer_soak_invariants(report)
    report["invariants"] = {
        "passed": len(failures) == 0,
        "failures": list(failures),
    }
    report["ok"] = len(failures) == 0
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a deterministic bridge-free Streamer Mode V1 endurance simulation."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--duration-hours",
        type=float,
        help="Run against a real wall-clock deadline (for example, 6).",
    )
    mode.add_argument(
        "--message-count",
        type=int,
        help="Run a fixed number of synthetic messages for a fast check.",
    )
    parser.add_argument("--queue-capacity", type=int, default=DEFAULT_QUEUE_CAPACITY)
    parser.add_argument("--demo-capacity", type=int, default=DEFAULT_DEMO_CAPACITY)
    parser.add_argument("--seed", type=int, default=1_337)
    parser.add_argument("--report-path", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        report = run_streamer_soak(
            duration_hours=args.duration_hours,
            message_count=args.message_count,
            queue_capacity=args.queue_capacity,
            demo_capacity=args.demo_capacity,
            seed=args.seed,
        )
    except ValueError as exc:
        build_argument_parser().error(str(exc))
    _atomic_json_write(Path(args.report_path), report)
    print(
        json.dumps(
            {
                "ok": report["ok"],
                "report_path": str(Path(args.report_path)),
                "elapsed_seconds": report["elapsed_seconds"],
                "received": report["messages"]["received"],
                "selected": report["messages"]["selected"],
                "invariant_failures": report["invariants"]["failures"],
            },
            sort_keys=True,
        )
    )
    return 0 if bool(report["ok"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_DEMO_CAPACITY",
    "DEFAULT_QUEUE_CAPACITY",
    "SOAK_REPORT_VERSION",
    "build_argument_parser",
    "evaluate_streamer_soak_invariants",
    "main",
    "run_streamer_soak",
]
