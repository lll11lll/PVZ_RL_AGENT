"""Bounded, bridge-free microbenchmarks for Streamer Mode V1 hot paths.

The benchmark deliberately exercises production parser, queue/controller,
action-resolution, event-logging, and masked-BC code.  It uses synthetic
fixtures and never connects to Twitch, the localhost bridge, Unity, or the Tk
dashboard.  Results therefore describe Python overhead only, not live game
latency or rollout throughput.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from pvzrl_action_space import (
    ADVENTURE_IDENTITY_ACTION_COUNT,
    ADVENTURE_IDENTITY_MAX_SEED_SLOTS,
    CELLS_PER_SLOT,
    adventure_identity_action_to_slot_cell,
    build_action_space_spec,
)
from pvzrl_actions import ACTION_KIND_PLACEMENT, ActionDecision, ActionIntent
from pvzrl_observation_layout import build_observation_layout
from pvzrl_stream_actions import ViewerActionResolution, resolve_viewer_action
from pvzrl_stream_commands import (
    BoundedViewerCommandQueue,
    ViewerCommand,
    ViewerCommandController,
    ViewerCommandParser,
)
from pvzrl_streamer_logging import BufferedStreamerEventLogger
from pvzrl_streamer_ppo import StreamerMaskablePPO
from pvzrl_streamer_source import DeterministicStreamCommandSource
from test_refactor_support import dense_observation, make_wrapper


OBSERVATION_SIZE = build_observation_layout(build_action_space_spec()).total_features
VIEWER_HASH = "a" * 64
BENCHMARK_ROW = 1
BENCHMARK_COLUMN = 3
BENCHMARK_SLOT = 2


class _FakeClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return float(self.value)

    def advance(self, seconds: float) -> float:
        self.value += float(seconds)
        return float(self.value)


def _nearest_rank(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, math.ceil(float(percentile) * len(ordered)) - 1))
    return ordered[index]


def _measurement(
    batch_operation: Callable[[], Any],
    *,
    operations_per_sample: int,
    samples: int,
    rounds: int,
    unit: str,
    warmups: int = 1,
) -> dict[str, Any]:
    """Time a pre-batched operation and normalize results to its logical unit."""

    for _ in range(max(0, int(warmups))):
        batch_operation()
    elapsed_ns: list[int] = []
    round_medians_ns: list[float] = []
    was_enabled = gc.isenabled()
    try:
        gc.disable()
        for _ in range(int(rounds)):
            current: list[int] = []
            for _sample in range(int(samples)):
                started = time.perf_counter_ns()
                batch_operation()
                duration = time.perf_counter_ns() - started
                current.append(duration)
                elapsed_ns.append(duration)
            round_medians_ns.append(statistics.median(current))
    finally:
        if was_enabled:
            gc.enable()

    divisor = max(1, int(operations_per_sample))
    per_operation_us = [value / divisor / 1_000.0 for value in elapsed_ns]
    return {
        "unit": str(unit),
        "operations_per_sample": divisor,
        "samples_per_round": int(samples),
        "rounds": int(rounds),
        "sample_count": len(elapsed_ns),
        "operation_count": len(elapsed_ns) * divisor,
        "median_us_per_operation": round(statistics.median(per_operation_us), 6),
        "p95_us_per_operation": round(_nearest_rank(per_operation_us, 0.95), 6),
        "min_us_per_operation": round(min(per_operation_us), 6),
        "max_us_per_operation": round(max(per_operation_us), 6),
        "round_medians_ms_per_batch": [
            round(value / 1_000_000.0, 6) for value in round_medians_ns
        ],
    }


def _prepared_measurement(
    prepare: Callable[[], None],
    timed_operation: Callable[[], Any],
    finalize: Callable[[Any], None],
    *,
    operations_per_sample: int,
    samples: int,
    rounds: int,
    unit: str,
    warmups: int = 1,
) -> dict[str, Any]:
    """Time an operation while keeping fixture setup and validation outside it."""

    def one() -> int:
        prepare()
        started = time.perf_counter_ns()
        result = timed_operation()
        duration = time.perf_counter_ns() - started
        finalize(result)
        return duration

    for _ in range(max(0, int(warmups))):
        one()
    elapsed_ns: list[int] = []
    round_medians_ns: list[float] = []
    was_enabled = gc.isenabled()
    try:
        gc.disable()
        for _ in range(int(rounds)):
            current: list[int] = []
            for _sample in range(int(samples)):
                duration = one()
                current.append(duration)
                elapsed_ns.append(duration)
            round_medians_ns.append(statistics.median(current))
    finally:
        if was_enabled:
            gc.enable()

    divisor = max(1, int(operations_per_sample))
    per_operation_us = [value / divisor / 1_000.0 for value in elapsed_ns]
    return {
        "unit": str(unit),
        "operations_per_sample": divisor,
        "samples_per_round": int(samples),
        "rounds": int(rounds),
        "sample_count": len(elapsed_ns),
        "operation_count": len(elapsed_ns) * divisor,
        "median_us_per_operation": round(statistics.median(per_operation_us), 6),
        "p95_us_per_operation": round(_nearest_rank(per_operation_us, 0.95), 6),
        "min_us_per_operation": round(min(per_operation_us), 6),
        "max_us_per_operation": round(max(per_operation_us), 6),
        "round_medians_ms_per_batch": [
            round(value / 1_000_000.0, 6) for value in round_medians_ns
        ],
    }


def _paired_measurement(
    baseline_operation: Callable[[], Any],
    streamer_operation: Callable[[], Any],
    *,
    operations_per_sample: int,
    samples: int,
    rounds: int,
    warmups: int = 1,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Interleave two equivalent batches and retain paired overhead samples."""

    for _ in range(max(0, int(warmups))):
        baseline_operation()
        streamer_operation()
    baseline_elapsed_ns: list[int] = []
    streamer_elapsed_ns: list[int] = []
    baseline_round_medians_ns: list[float] = []
    streamer_round_medians_ns: list[float] = []
    was_enabled = gc.isenabled()
    try:
        gc.disable()
        for round_index in range(int(rounds)):
            baseline_current: list[int] = []
            streamer_current: list[int] = []
            for sample_index in range(int(samples)):
                # Alternate order so cache, scheduler, and power-state drift do
                # not always favor the same arm.
                if (round_index + sample_index) % 2 == 0:
                    started = time.perf_counter_ns()
                    baseline_operation()
                    baseline_duration = time.perf_counter_ns() - started
                    started = time.perf_counter_ns()
                    streamer_operation()
                    streamer_duration = time.perf_counter_ns() - started
                else:
                    started = time.perf_counter_ns()
                    streamer_operation()
                    streamer_duration = time.perf_counter_ns() - started
                    started = time.perf_counter_ns()
                    baseline_operation()
                    baseline_duration = time.perf_counter_ns() - started
                baseline_current.append(baseline_duration)
                streamer_current.append(streamer_duration)
                baseline_elapsed_ns.append(baseline_duration)
                streamer_elapsed_ns.append(streamer_duration)
            baseline_round_medians_ns.append(statistics.median(baseline_current))
            streamer_round_medians_ns.append(statistics.median(streamer_current))
    finally:
        if was_enabled:
            gc.enable()

    divisor = max(1, int(operations_per_sample))

    def summarize(values: Sequence[int], round_medians: Sequence[float]) -> dict[str, Any]:
        per_operation_us = [value / divisor / 1_000.0 for value in values]
        return {
            "unit": "synthetic_environment_step",
            "operations_per_sample": divisor,
            "samples_per_round": int(samples),
            "rounds": int(rounds),
            "sample_count": len(values),
            "operation_count": len(values) * divisor,
            "median_us_per_operation": round(statistics.median(per_operation_us), 6),
            "p95_us_per_operation": round(_nearest_rank(per_operation_us, 0.95), 6),
            "min_us_per_operation": round(min(per_operation_us), 6),
            "max_us_per_operation": round(max(per_operation_us), 6),
            "round_medians_ms_per_batch": [
                round(value / 1_000_000.0, 6) for value in round_medians
            ],
        }

    baseline_per_step_us = [value / divisor / 1_000.0 for value in baseline_elapsed_ns]
    streamer_per_step_us = [value / divisor / 1_000.0 for value in streamer_elapsed_ns]
    paired_deltas = [
        streamer_value - baseline_value
        for baseline_value, streamer_value in zip(baseline_per_step_us, streamer_per_step_us)
    ]
    paired_ratios = [
        streamer_value / baseline_value
        for baseline_value, streamer_value in zip(baseline_per_step_us, streamer_per_step_us)
        if baseline_value > 0.0
    ]
    comparison = {
        "paired_sample_count": len(paired_deltas),
        "median_overhead_us_per_step": round(statistics.median(paired_deltas), 6),
        "p95_overhead_us_per_step": round(_nearest_rank(paired_deltas, 0.95), 6),
        "median_streamer_to_baseline_ratio": (
            round(statistics.median(paired_ratios), 6) if paired_ratios else None
        ),
        "cases_interleaved": True,
        "interpretation": (
            "production Python wrapper with a deterministic synthetic base step; "
            "not live bridge or Unity latency"
        ),
    }
    return (
        summarize(baseline_elapsed_ns, baseline_round_medians_ns),
        summarize(streamer_elapsed_ns, streamer_round_medians_ns),
        comparison,
    )


def _policy_action(slot: int, row: int, column: int) -> int:
    return 1 + int(slot) * CELLS_PER_SLOT + int(row) * 10 + int(column)


def _action_fixture() -> tuple[ViewerCommand, np.ndarray, Mapping[int, ActionDecision]]:
    """Build one immutable current-frame decision snapshot for the real identity."""

    command = ViewerCommandParser().parse("!slot 3 2 4")
    action_mask = np.zeros(ADVENTURE_IDENTITY_ACTION_COUNT, dtype=bool)
    decisions: dict[int, ActionDecision] = {}
    for slot in range(ADVENTURE_IDENTITY_MAX_SEED_SLOTS):
        action_id = _policy_action(slot, BENCHMARK_ROW, BENCHMARK_COLUMN)
        decoded = adventure_identity_action_to_slot_cell(action_id)
        legal = slot == BENCHMARK_SLOT
        intent = ActionIntent(
            source="mask",
            policy_action=action_id,
            bridge_action=action_id,
            action_kind=ACTION_KIND_PLACEMENT,
            seed_slot=slot,
            row=BENCHMARK_ROW,
            column=BENCHMARK_COLUMN,
            decoded_action=decoded,
        )
        decisions[action_id] = ActionDecision(
            intent=intent,
            legal=legal,
            rejection_reason="" if legal else "slot_not_usable",
            frame_identity="benchmark-frame:1",
            config_fingerprint="benchmark-generalist-14slot",
            resolved_action_kind=ACTION_KIND_PLACEMENT,
            selected_plant_type=0 if slot == BENCHMARK_SLOT else slot,
            existing_plant_type=-1,
            bridge_authoritative=True,
        )
        action_mask[action_id] = legal
    return command, action_mask, decisions


def _install_synthetic_generalist_step(wrapper: Any) -> None:
    """Replace only bridge mutation while retaining ``PvZMaskedPPOEnv.step``."""

    initial = dense_observation(slot_count=ADVENTURE_IDENTITY_MAX_SEED_SLOTS)
    initial["frameCount"] = 1
    wrapper.config.max_steps = 0
    wrapper.base.config.max_steps = 0
    wrapper._initialize_episode_accounting(
        copy.deepcopy(initial),
        {},
        source="streamer_hotpath_benchmark",
        include_reset_safety_fields=False,
    )
    frame = 1

    def synthetic_base_step(
        action: int,
        *,
        coach_bridge_command: Optional[Mapping[str, Any]] = None,
        coach_context: Optional[Mapping[str, Any]] = None,
        action_intent: Optional[ActionIntent] = None,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        del coach_bridge_command, coach_context, action_intent
        nonlocal frame
        frame += 1
        previous = wrapper._last_observation
        if not isinstance(previous, dict):
            raise AssertionError("synthetic wrapper lost its current observation")
        observation = dict(previous)
        observation["frameCount"] = frame
        legal_actions = observation.get("legalActions", [0])
        info = {
            "action_result": {
                "action": int(action),
                "requestedAction": int(action),
                "executedAction": int(action),
                "illegalAction": False,
                "plantPlaced": False,
                "costPaid": False,
                "cooldownStarted": False,
                "decoded": {"kind": "wait"},
            },
            "reward_breakdown": {"reward_total": 0.0},
            "legal_actions": legal_actions,
            "bridge_legal_actions": legal_actions,
        }
        return observation, 0.0, False, False, info

    wrapper.base.step = synthetic_base_step


def _synthetic_generalist_step_pair(
    *,
    samples: int,
    rounds: int,
    inner_iterations: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Compare the same canonical wrapper step with only an idle Streamer gate added."""

    baseline_wrapper = make_wrapper(fusion_enabled=True)
    streamer_wrapper = make_wrapper(fusion_enabled=True)
    step_iterations = min(int(inner_iterations), 25)
    controller = ViewerCommandController(opportunity_interval_seconds=2.0)
    try:
        _install_synthetic_generalist_step(baseline_wrapper)
        _install_synthetic_generalist_step(streamer_wrapper)
        # Attach only the quiet controller branch.  Event-log I/O is excluded
        # here and measured separately by the two structured logging cases.
        controller.begin_phase("STREAM_TRAIN", accepting=True)
        controller.start()
        streamer_wrapper.streamer_v1_controller = controller
        streamer_wrapper.streamer_v1_platform = "synthetic"

        def baseline_batch() -> None:
            for _ in range(step_iterations):
                baseline_wrapper.step(0)

        def streamer_batch() -> None:
            for _ in range(step_iterations):
                streamer_wrapper.step(0)

        baseline_result, streamer_result, comparison = _paired_measurement(
            baseline_batch,
            streamer_batch,
            operations_per_sample=step_iterations,
            samples=samples,
            rounds=rounds,
        )
        expected_steps = (1 + int(samples) * int(rounds)) * step_iterations
        validation = {
            "wrapper": "PvZMaskedPPOEnv.step",
            "bridge_operation_replaced": True,
            "synthetic_base_step": "deterministic_wait_action",
            "step_iterations_per_sample": step_iterations,
            "baseline_steps_observed": int(baseline_wrapper.episode_state.step_count),
            "streamer_steps_observed": int(streamer_wrapper.episode_state.step_count),
            "expected_steps_per_arm": expected_steps,
            "streamer_viewer_interventions": int(
                streamer_wrapper._streamer_v1_intervention_count
            ),
            "streamer_queue_depth": int(controller.queue.snapshot().depth),
            "streamer_source": "none",
            "event_logging_in_step_pair": False,
        }
        if (
            validation["baseline_steps_observed"] != expected_steps
            or validation["streamer_steps_observed"] != expected_steps
            or validation["streamer_viewer_interventions"] != 0
            or validation["streamer_queue_depth"] != 0
        ):
            raise AssertionError("synthetic environment-step pair violated quiet-controller invariants")
        return baseline_result, streamer_result, comparison, validation
    finally:
        baseline_wrapper.close()
        streamer_wrapper.close()


class _SyntheticMaskedEnv(gym.Env[np.ndarray, int]):
    """Minimal fixed-state environment with the maintained model geometry."""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(ADVENTURE_IDENTITY_ACTION_COUNT)
        self.observation = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
        self.observation[:4] = np.asarray([0.25, -0.5, 0.75, 0.125], dtype=np.float32)
        self.mask = np.zeros(ADVENTURE_IDENTITY_ACTION_COUNT, dtype=bool)
        self.mask[:16] = True

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        super().reset(seed=seed)
        return self.observation.copy(), {}

    def action_masks(self) -> np.ndarray:
        return self.mask.copy()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not bool(self.mask[int(action)]):
            raise AssertionError("synthetic policy selected a masked action")
        return self.observation.copy(), 0.0, False, False, {}


def _masked_bc_benchmark(*, updates: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Measure production combined optimizer updates with PPO actor pressure neutralized."""

    vec_env = DummyVecEnv([_SyntheticMaskedEnv])
    model = StreamerMaskablePPO(
        "MlpPolicy",
        vec_env,
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        gamma=0.9,
        gae_lambda=0.8,
        learning_rate=1e-3,
        ent_coef=0.0,
        vf_coef=0.0,
        max_grad_norm=10.0,
        normalize_advantage=True,
        seed=17,
        policy_kwargs={"net_arch": {"pi": [16], "vf": [16]}},
        demonstration_capacity=32,
        bc_enabled=True,
        bc_coefficient=0.01,
        bc_batch_size=8,
        bc_update_frequency=1,
        bc_min_demonstrations=1,
        bc_seed=23,
        verbose=0,
    )
    legal_mask = np.zeros(ADVENTURE_IDENTITY_ACTION_COUNT, dtype=bool)
    legal_mask[:16] = True
    try:
        for index in range(16):
            observation = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
            observation[:4] = np.asarray(
                [0.25, -0.5, 0.75, float(index) / 32.0],
                dtype=np.float32,
            )
            model.demonstration_buffer.add(
                observation,
                legal_mask,
                1,
                metadata={"episode_id": f"synthetic-bc-{index}"},
            )
        _total, callback = model._setup_learn(
            total_timesteps=2,
            callback=None,
            reset_num_timesteps=True,
            tb_log_name="streamer_hotpath_benchmark",
            progress_bar=False,
        )
        if not model.collect_rollouts(
            vec_env,
            callback,
            model.rollout_buffer,
            n_rollout_steps=2,
            use_masking=True,
        ):
            raise RuntimeError("synthetic BC rollout setup stopped unexpectedly")
        model.rollout_buffer.advantages.fill(0.0)
        model.rollout_buffer.returns[:] = model.rollout_buffer.values
        before_updates = int(model.bc_update_count)
        measurement = _measurement(
            model.train,
            operations_per_sample=1,
            samples=int(updates),
            rounds=1,
            unit="combined_optimizer_update",
            warmups=0,
        )
        added_updates = int(model.bc_update_count) - before_updates
        validation = {
            "observation_shape": list(model.observation_space.shape),
            "action_count": int(model.action_space.n),
            "legal_actions_per_demo": int(legal_mask.sum()),
            "demonstration_count": len(model.demonstration_buffer),
            "bc_updates_observed": added_updates,
            "last_bc_loss": float(model.last_bc_loss),
            "last_bc_policy_agreement": float(model.last_bc_policy_agreement),
            "ppo_actor_pressure_neutralized": True,
        }
        if added_updates != int(updates) or not math.isfinite(float(model.last_bc_loss)):
            raise AssertionError("production masked-BC update did not satisfy the benchmark contract")
        return measurement, validation
    finally:
        vec_env.close()


def _validate_options(
    *,
    samples: int,
    rounds: int,
    inner_iterations: int,
    poll_batch: int,
    log_batch: int,
    bc_updates: int,
) -> None:
    bounds = {
        "samples": (int(samples), 1, 100),
        "rounds": (int(rounds), 1, 20),
        "inner_iterations": (int(inner_iterations), 1, 2_048),
        "poll_batch": (int(poll_batch), 1, 256),
        "log_batch": (int(log_batch), 1, 1_024),
        "bc_updates": (int(bc_updates), 0, 10),
    }
    for name, (value, minimum, maximum) in bounds.items():
        if not minimum <= value <= maximum:
            raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    if int(samples) * int(rounds) * int(inner_iterations) > 100_000:
        raise ValueError("requested in-memory benchmark exceeds the 100,000-operation safety bound")
    if int(samples) * int(rounds) * int(poll_batch) > 100_000:
        raise ValueError("requested polling benchmark exceeds the 100,000-message safety bound")
    if int(samples) * int(rounds) * int(log_batch) > 200_000:
        raise ValueError("requested logging benchmark exceeds the 200,000-event safety bound")
    if int(samples) * int(rounds) * min(int(inner_iterations), 25) > 1_000:
        raise ValueError(
            "requested synthetic environment-step pair exceeds the 1,000-step-per-arm safety bound"
        )


def run_benchmarks(
    *,
    samples: int = 7,
    rounds: int = 3,
    inner_iterations: int = 200,
    poll_batch: int = 32,
    log_batch: int = 256,
    bc_updates: int = 3,
) -> dict[str, Any]:
    """Run all bounded Streamer hot-path cases and return a JSON-safe report."""

    _validate_options(
        samples=samples,
        rounds=rounds,
        inner_iterations=inner_iterations,
        poll_batch=poll_batch,
        log_batch=log_batch,
        bc_updates=bc_updates,
    )
    command, action_mask, decisions = _action_fixture()

    def resolve_current_frame(parsed: ViewerCommand) -> ViewerActionResolution:
        return resolve_viewer_action(
            parsed,
            action_mask=action_mask,
            action_decision=decisions.get,
            source="twitch",
        )

    resolved_once = resolve_current_frame(command)
    if not resolved_once.legal or resolved_once.action_id != _policy_action(
        BENCHMARK_SLOT,
        BENCHMARK_ROW,
        BENCHMARK_COLUMN,
    ):
        raise AssertionError("synthetic action fixture does not resolve through the canonical identity")

    results: dict[str, Any] = {}

    idle_controller = ViewerCommandController(
        opportunity_interval_seconds=3_600.0,
    )
    idle_controller.tick(resolve_current_frame)
    idle_deadline = idle_controller.next_opportunity_monotonic

    def baseline_idle_batch() -> int:
        count = 0
        for _ in range(int(inner_iterations)):
            count += int(time.monotonic() < idle_deadline)
        return count

    def controller_idle_batch() -> int:
        count = 0
        for _ in range(int(inner_iterations)):
            tick = idle_controller.tick(resolve_current_frame)
            count += int(not tick.opportunity_opened)
        return count

    results["baseline_policy_loop_idle_check"] = _measurement(
        baseline_idle_batch,
        operations_per_sample=inner_iterations,
        samples=samples,
        rounds=rounds,
        unit="idle_check",
    )
    results["streamer_controller_idle_tick"] = _measurement(
        controller_idle_batch,
        operations_per_sample=inner_iterations,
        samples=samples,
        rounds=rounds,
        unit="controller_tick",
    )

    (
        synthetic_step_baseline,
        synthetic_step_streamer,
        synthetic_step_comparison,
        synthetic_step_validation,
    ) = _synthetic_generalist_step_pair(
        samples=samples,
        rounds=rounds,
        inner_iterations=inner_iterations,
    )
    results["synthetic_generalist_env_step_without_streamer"] = synthetic_step_baseline
    results["synthetic_generalist_env_step_with_quiet_streamer"] = synthetic_step_streamer

    parser = ViewerCommandParser()

    def parse_batch() -> None:
        for _ in range(int(inner_iterations)):
            parsed = parser.try_parse("!plant sunflower 2 4")
            if not parsed.accepted:
                raise AssertionError("benchmark command unexpectedly failed parsing")

    results["viewer_command_parse"] = _measurement(
        parse_batch,
        operations_per_sample=inner_iterations,
        samples=samples,
        rounds=rounds,
        unit="command",
    )

    poll_clock = _FakeClock()
    poll_source = DeterministicStreamCommandSource(
        queue_capacity=poll_batch,
        dedupe_capacity=max(1_024, poll_batch * 8),
        monotonic=poll_clock,
    )
    poll_queue = BoundedViewerCommandQueue(
        capacity=poll_batch,
        ttl_seconds=30.0,
        dedupe_capacity=max(1_024, poll_batch * 8),
        monotonic=poll_clock,
    )
    poll_controller = ViewerCommandController(
        source=poll_source,
        queue=poll_queue,
        max_poll_messages=poll_batch,
        monotonic=poll_clock,
    )
    poll_source.start()
    poll_controller.begin_phase("stream_train", accepting=True, now=poll_clock())
    poll_sequence = 0

    def prepare_poll() -> None:
        nonlocal poll_sequence
        for _ in range(int(poll_batch)):
            poll_sequence += 1
            accepted = poll_source.submit(
                "!slot 3 2 4",
                viewer_hash=VIEWER_HASH,
                delivery_id=f"benchmark-delivery-{poll_sequence}",
                event_id=f"benchmark-event-{poll_sequence}",
                received_monotonic=poll_clock(),
            )
            if not accepted:
                raise AssertionError("bounded synthetic source rejected benchmark input")

    def finalize_poll(outcomes: Any) -> None:
        if len(outcomes) != int(poll_batch) or any(
            outcome.status != "accepted" for outcome in outcomes
        ):
            raise AssertionError("source polling did not parse and enqueue the full batch")
        poll_queue.clear(increment_generation=False)

    results["command_source_poll_parse_enqueue"] = _prepared_measurement(
        prepare_poll,
        lambda: poll_controller.poll_source(now=poll_clock()),
        finalize_poll,
        operations_per_sample=poll_batch,
        samples=samples,
        rounds=rounds,
        unit="message",
    )
    poll_controller.close(timeout_seconds=0.1)

    selection_clock = _FakeClock()
    selection_queue = BoundedViewerCommandQueue(
        capacity=inner_iterations,
        ttl_seconds=1_000_000.0,
        dedupe_capacity=max(1_024, inner_iterations * 4),
        monotonic=selection_clock,
    )
    selection_controller = ViewerCommandController(
        queue=selection_queue,
        opportunity_interval_seconds=2.0,
        monotonic=selection_clock,
    )
    selection_controller.begin_phase("stream_train", accepting=True, now=selection_clock())
    selection_sequence = 0

    def prepare_selection() -> None:
        nonlocal selection_sequence
        for _ in range(int(inner_iterations)):
            selection_sequence += 1
            outcome = selection_queue.enqueue(
                command,
                message_id=f"selection-event-{selection_sequence}",
                viewer_hash=VIEWER_HASH,
                received_monotonic=selection_clock(),
            )
            if outcome.status != "accepted":
                raise AssertionError("selection fixture could not fill its bounded FIFO")

    def select_batch() -> int:
        selected = 0
        for _ in range(int(inner_iterations)):
            selection_clock.advance(2.0)
            tick = selection_controller.tick(
                resolve_current_frame,
                now=selection_clock(),
            )
            selected += int(tick.selected is not None)
        return selected

    def finalize_selection(selected: Any) -> None:
        if int(selected) != int(inner_iterations) or len(selection_queue) != 0:
            raise AssertionError("FIFO selection did not dispatch one legal command per opportunity")

    results["fifo_legal_command_selection"] = _prepared_measurement(
        prepare_selection,
        select_batch,
        finalize_selection,
        operations_per_sample=inner_iterations,
        samples=samples,
        rounds=rounds,
        unit="selected_command",
    )

    def resolve_batch() -> None:
        for _ in range(int(inner_iterations)):
            resolution = resolve_current_frame(command)
            if not resolution.legal:
                raise AssertionError("current-frame mask validation unexpectedly rejected the fixture")

    results["current_mask_action_resolution"] = _measurement(
        resolve_batch,
        operations_per_sample=inner_iterations,
        samples=samples,
        rounds=rounds,
        unit="resolution",
    )

    event_counter = 0
    with tempfile.TemporaryDirectory(prefix="pvzrl_streamer_hotpaths_") as temp_dir:
        temp_root = Path(temp_dir)
        buffered_logger = BufferedStreamerEventLogger(
            temp_root / "buffered_events.jsonl",
            flush_records=inner_iterations + 1,
            max_bytes=1 * 1024 * 1024,
            backup_count=1,
        )

        def buffered_log_batch() -> None:
            nonlocal event_counter
            for _ in range(int(inner_iterations)):
                event_counter += 1
                buffered_logger.append(
                    {
                        "event": "viewer_command_selected",
                        "command_id": f"buffered-{event_counter}",
                        "viewer_hash": VIEWER_HASH,
                        "action_id": resolved_once.action_id,
                        "command": command.to_safe_dict(),
                    }
                )

        results["structured_event_log_buffered_append"] = _prepared_measurement(
            lambda: None,
            buffered_log_batch,
            lambda _result: buffered_logger.flush(),
            operations_per_sample=inner_iterations,
            samples=samples,
            rounds=rounds,
            unit="event",
        )
        buffered_logger.close()

        durable_logger = BufferedStreamerEventLogger(
            temp_root / "durable_events.jsonl",
            flush_records=log_batch,
            max_bytes=4 * 1024 * 1024,
            backup_count=1,
        )

        def durable_log_batch() -> None:
            nonlocal event_counter
            for _ in range(int(log_batch)):
                event_counter += 1
                durable_logger.append(
                    {
                        "event": "viewer_command_outcome",
                        "command_id": f"durable-{event_counter}",
                        "viewer_hash": VIEWER_HASH,
                        "status": "executed",
                        "action_id": resolved_once.action_id,
                    }
                )

        results["structured_event_log_batch_fsync"] = _measurement(
            durable_log_batch,
            operations_per_sample=log_batch,
            samples=samples,
            rounds=rounds,
            unit="event",
        )
        durable_logger.close()

    bc_validation: dict[str, Any]
    if bc_updates > 0:
        bc_measurement, bc_validation = _masked_bc_benchmark(updates=bc_updates)
        results["masked_bc_combined_optimizer_update"] = bc_measurement
    else:
        bc_validation = {"skipped": True, "reason": "bc_updates_zero"}
        results["masked_bc_combined_optimizer_update"] = {
            "skipped": True,
            "reason": "bc_updates_zero",
        }

    baseline_us = float(results["baseline_policy_loop_idle_check"]["median_us_per_operation"])
    controller_us = float(results["streamer_controller_idle_tick"]["median_us_per_operation"])
    idle_ratio = controller_us / baseline_us if baseline_us > 0.0 else None
    return {
        "schema_version": 1,
        "benchmark": "pvzrl_streamer_v1_hotpaths",
        "synthetic_bridge_free": True,
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "pid": os.getpid(),
        },
        "methodology": {
            "clock": "perf_counter_ns",
            "percentile": "nearest_rank_p95",
            "gc_disabled_during_timed_samples": True,
            "setup_and_validation_outside_timed_poll_selection_samples": True,
            "samples": int(samples),
            "rounds": int(rounds),
            "inner_iterations": int(inner_iterations),
            "poll_batch": int(poll_batch),
            "log_batch": int(log_batch),
            "bc_updates": int(bc_updates),
            "synthetic_environment_step_iterations_per_sample": int(
                synthetic_step_validation["step_iterations_per_sample"]
            ),
            "limitations": [
                "synthetic fixtures only; no credentialed Twitch connection",
                "no localhost bridge, Unity, game, profile, or live Adventure timing",
                "no full PPO rollout throughput or Tk rendering measurement",
                "event fsync timing depends on operating-system cache and storage state",
                "BC timing uses a tiny synthetic network on the maintained observation/action geometry",
                "the paired synthetic environment-step case excludes event-log I/O, which is measured separately",
            ],
        },
        "contracts": {
            "observation_shape": [OBSERVATION_SIZE],
            "action_count": ADVENTURE_IDENTITY_ACTION_COUNT,
            "identity_slots": ADVENTURE_IDENTITY_MAX_SEED_SLOTS,
            "mask_true_count": int(action_mask.sum()),
            "resolved_action_id": int(resolved_once.action_id),
            "viewer_coordinates_converted_once": {
                "viewer_row": 2,
                "viewer_column": 4,
                "viewer_slot": 3,
                "internal_row": command.row,
                "internal_column": command.column,
                "internal_slot": command.seed_slot,
            },
            "bc": bc_validation,
            "synthetic_environment_step": synthetic_step_validation,
        },
        "comparisons": {
            "idle_controller_overhead_us": round(controller_us - baseline_us, 6),
            "idle_controller_to_baseline_ratio": (
                round(idle_ratio, 6) if idle_ratio is not None else None
            ),
            "idle_loop_interpretation": "Python dispatch overhead only; not environment-step latency",
            "synthetic_environment_step": synthetic_step_comparison,
        },
        "results": results,
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace a JSON report without leaving a partial destination."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run bounded, bridge-free Streamer Mode V1 hot-path benchmarks."
    )
    parser.add_argument("--samples", type=int, default=7, help="Timed samples per round (1-100).")
    parser.add_argument("--rounds", type=int, default=3, help="Independent rounds (1-20).")
    parser.add_argument(
        "--inner-iterations",
        type=int,
        default=200,
        help="In-memory operations per timed sample (1-2048).",
    )
    parser.add_argument(
        "--poll-batch",
        type=int,
        default=32,
        help="Source messages per prepared polling sample (1-256).",
    )
    parser.add_argument(
        "--log-batch",
        type=int,
        default=256,
        help="Structured events per durable fsync batch (1-1024).",
    )
    parser.add_argument(
        "--bc-updates",
        type=int,
        default=3,
        help="Production combined masked-BC optimizer updates (0-10; 0 skips).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional atomically replaced JSON report path (runs/ is recommended).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        payload = run_benchmarks(
            samples=args.samples,
            rounds=args.rounds,
            inner_iterations=args.inner_iterations,
            poll_batch=args.poll_batch,
            log_batch=args.log_batch,
            bc_updates=args.bc_updates,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    print(rendered)
    if args.json_out is not None:
        _atomic_write_json(args.json_out, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
