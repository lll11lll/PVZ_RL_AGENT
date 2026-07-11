"""Repeatable bridge-free microbenchmarks for the PvZRL refactor.

The fixture is deliberately synthetic and dense.  Results describe Python
scan/allocation/I/O pressure only; they are not rollout SPS or live bridge
latency measurements.
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
from typing import Any, Callable, Dict, List

import numpy as np

from pvzrl_adventure import LiveStatusWriter, build_live_status
from pvzrl_fusion import build_fusion_diagnostics
from pvzrl_gui import PvZDashboard
from test_refactor_support import (
    array_sha256,
    cooldown_variant,
    dense_observation,
    json_sha256,
    low_sun_variant,
    make_wrapper,
    mask_sha256,
    observation_for_wrapper,
)


def percentile_nearest_rank(values: List[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def measure(
    operation: Callable[[], Any],
    *,
    samples: int,
    rounds: int,
    warmups: int = 3,
) -> Dict[str, Any]:
    for _ in range(max(0, warmups)):
        operation()
    all_ms: List[float] = []
    round_medians: List[float] = []
    was_enabled = gc.isenabled()
    try:
        gc.disable()
        for _ in range(rounds):
            current: List[float] = []
            for _sample in range(samples):
                started = time.perf_counter_ns()
                operation()
                elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
                current.append(elapsed_ms)
                all_ms.append(elapsed_ms)
            round_medians.append(statistics.median(current))
    finally:
        if was_enabled:
            gc.enable()
    return {
        "samples_per_round": int(samples),
        "rounds": int(rounds),
        "sample_count": len(all_ms),
        "median_ms": round(statistics.median(all_ms), 6),
        "p95_ms": round(percentile_nearest_rank(all_ms, 0.95), 6),
        "min_ms": round(min(all_ms), 6),
        "max_ms": round(max(all_ms), 6),
        "round_medians_ms": [round(value, 6) for value in round_medians],
    }


def run_benchmarks(*, samples: int, rounds: int) -> Dict[str, Any]:
    fixed = make_wrapper(identity=False, fusion_enabled=True)
    identity = make_wrapper(identity=True, fusion_enabled=True)
    identity_no_fusion = make_wrapper(identity=True, fusion_enabled=False)
    fixed_dense = dense_observation(slot_count=4)
    identity_dense = dense_observation(slot_count=14)
    identity_low_sun = low_sun_variant(identity_dense)
    identity_cooldown = cooldown_variant(identity_dense)
    fixed._last_observation = fixed_dense
    identity._last_observation = identity_dense
    identity_no_fusion._last_observation = identity_dense

    sparse_fixed = observation_for_wrapper(fixed)
    sparse_identity = observation_for_wrapper(identity)
    reward_previous = copy.deepcopy(identity_dense)
    reward_current = copy.deepcopy(identity_dense)
    reward_current["killCount"] = int(reward_previous.get("killCount", 0)) + 1
    reward_current["wave"] = int(reward_previous.get("wave", 0)) + 1
    reward_action = {"ok": True, "plantPlaced": False, "kind": "wait"}

    results: Dict[str, Any] = {}
    try:
        results["action_mask_fixed_dense"] = measure(fixed.action_masks, samples=samples, rounds=rounds)
        results["action_mask_identity_dense"] = measure(identity.action_masks, samples=samples, rounds=rounds)

        def identity_low_sun_mask() -> np.ndarray:
            identity._last_observation = identity_low_sun
            try:
                return identity.action_masks()
            finally:
                identity._last_observation = identity_dense

        def identity_cooldown_mask() -> np.ndarray:
            identity._last_observation = identity_cooldown
            try:
                return identity.action_masks()
            finally:
                identity._last_observation = identity_dense

        results["action_mask_identity_low_sun"] = measure(identity_low_sun_mask, samples=samples, rounds=rounds)
        results["action_mask_identity_cooldown"] = measure(identity_cooldown_mask, samples=samples, rounds=rounds)
        results["action_mask_identity_no_fusion"] = measure(
            identity_no_fusion.action_masks,
            samples=samples,
            rounds=rounds,
        )
        results["observation_encode_fixed_sparse"] = measure(
            lambda: fixed._encode_observation(sparse_fixed),
            samples=samples,
            rounds=rounds,
        )
        results["observation_encode_identity_sparse"] = measure(
            lambda: identity._encode_observation(sparse_identity),
            samples=samples,
            rounds=rounds,
        )
        results["reward_breakdown_dense"] = measure(
            lambda: identity.base.compute_reward_breakdown(reward_previous, reward_current, reward_action),
            samples=samples,
            rounds=rounds,
        )
        results["fusion_candidate_scan_dense"] = measure(
            lambda: build_fusion_diagnostics("observe", identity_dense),
            samples=samples,
            rounds=rounds,
        )
        live_context = {"mode": "benchmark", "phase": "fixture", "episode": 1, "step": 2}
        live_state = {
            "availableSeedNames": list(identity_dense.get("availableSeedNames", [])),
            "unlockedSeedNames": list(identity_dense.get("unlockedSeedNames", [])),
        }
        results["live_status_build_dense"] = measure(
            lambda: build_live_status(identity, live_context, live_state, {}),
            samples=samples,
            rounds=rounds,
        )
        live_payload = build_live_status(identity, live_context, live_state, {})
        results["live_status_json_serialize"] = measure(
            lambda: json.dumps(live_payload, indent=2),
            samples=samples,
            rounds=rounds,
        )
        with tempfile.TemporaryDirectory(prefix="pvzrl_refactor_benchmark_") as temp_dir:
            temp_path = Path(temp_dir)
            live_path = temp_path / "live_status.json"
            writer = LiveStatusWriter(live_path)
            results["live_status_atomic_write"] = measure(
                lambda: writer.write(live_payload),
                samples=samples,
                rounds=rounds,
                warmups=1,
            )
            live_path.write_text(json.dumps(live_payload, indent=2) + "\n", encoding="utf-8")
            dashboard = PvZDashboard.__new__(PvZDashboard)
            dashboard.live_status_path = live_path
            results["gui_unchanged_status_read_parse"] = measure(
                dashboard._read_live_status_file,
                samples=samples,
                rounds=rounds,
            )

            def status_signature() -> tuple[int, int]:
                stat = live_path.stat()
                return stat.st_mtime_ns, stat.st_size

            results["gui_status_signature_only"] = measure(
                status_signature,
                samples=samples,
                rounds=rounds,
            )

        dense_mask = identity.action_masks()
        dense_vector = identity._encode_observation(identity_dense)
        return {
            "schema_version": 1,
            "environment": {
                "platform": platform.platform(),
                "python": sys.version.split()[0],
                "implementation": platform.python_implementation(),
                "pid": os.getpid(),
            },
            "methodology": {
                "clock": "perf_counter_ns",
                "p95": "nearest_rank",
                "gc_disabled_during_samples": True,
                "samples_per_round": int(samples),
                "rounds": int(rounds),
                "fixture": "synthetic_dense_5x10_25plants_30zombies",
                "limitations": [
                    "no live bridge or Unity timing",
                    "no PPO inference or rollout SPS",
                    "no full Tk widget rendering",
                    "filesystem results depend on Windows cache and device state",
                ],
            },
            "contracts": {
                "dense_observation_sha256": json_sha256(identity_dense),
                "identity_mask_sha256": mask_sha256(dense_mask),
                "identity_vector_sha256": array_sha256(dense_vector),
                "identity_mask_true_count": int(dense_mask.sum()),
                "identity_vector_size": int(dense_vector.size),
                "live_payload_bytes": len(json.dumps(live_payload).encode("utf-8")),
            },
            "results": results,
        }
    finally:
        fixed.close()
        identity.close()
        identity_no_fusion.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bridge-free PvZRL refactor hot-path benchmarks.")
    parser.add_argument("--samples", type=int, default=30, help="Samples per operation in each round.")
    parser.add_argument("--rounds", type=int, default=3, help="Independent rounds per operation.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON result path (runs/ is recommended).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples <= 0 or args.rounds <= 0:
        raise SystemExit("--samples and --rounds must be positive")
    payload = run_benchmarks(samples=int(args.samples), rounds=int(args.rounds))
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
