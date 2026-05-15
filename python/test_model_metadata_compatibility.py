"""Lightweight compatibility checks for PvZRL model metadata.

This test intentionally avoids loading PPO weights or starting the game. It
creates temporary run folders with model_metadata.json and validates the safety
rules that prevent same-action-count seed-slot drift.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from pvzrl_model_metadata import (
    format_compatibility_failure,
    model_compatibility_live_status,
    validate_model_metadata,
    write_model_metadata,
)


CONSERVATIVE_SEEDS = ["SunFlower", "SunFlower", "Peashooter", "Peashooter"]
CONSERVATIVE_PLANT_TYPES = [1, 1, 0, 0]
EXPANDED_SEEDS = ["SunFlower", "Peashooter", "WallNut", "CherryBomb"]
EXPANDED_PLANT_TYPES = [1, 0, 3, 2]


def model_family(seed_list: List[str]) -> str:
    unique_parts: List[str] = []
    for seed in seed_list:
        part = "".join(ch.lower() if ch.isalnum() else "_" for ch in seed).strip("_")
        if part not in unique_parts:
            unique_parts.append(part)
    return f"ppo_{len(seed_list)}slot_{'_'.join(unique_parts)}"


def config(seed_list: List[str], plant_types: List[int]) -> Dict[str, Any]:
    return {
        "model_family": model_family(seed_list),
        "seed_list": list(seed_list),
        "plant_types": list(plant_types),
        "action_space_mode": "fixed",
        "max_seed_slots": len(seed_list),
        "row_count": 5,
        "column_count": 10,
        "total_timesteps": 123,
    }


def fake_model(run_root: Path, name: str, metadata_config: Optional[Dict[str, Any]]) -> Path:
    run_dir = run_root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / "model.zip"
    model_path.write_bytes(b"fake ppo zip placeholder")
    if metadata_config is not None:
        write_model_metadata(run_dir, metadata_config, model_path=model_path, config_path=run_dir / "resolved_config.json")
    return model_path


def assert_case(name: str, actual_ok: bool, actual_reason: Optional[str], expected_ok: bool, expected_reason: Optional[str]) -> Dict[str, Any]:
    passed = actual_ok == expected_ok and (expected_reason is None or actual_reason == expected_reason)
    return {
        "case": name,
        "passed": passed,
        "compatible": actual_ok,
        "blocked_reason": actual_reason,
        "expected_compatible": expected_ok,
        "expected_blocked_reason": expected_reason,
    }


def main() -> int:
    conservative_config = config(CONSERVATIVE_SEEDS, CONSERVATIVE_PLANT_TYPES)
    expanded_config = config(EXPANDED_SEEDS, EXPANDED_PLANT_TYPES)
    results: List[Dict[str, Any]] = []
    examples: Dict[str, Any] = {}

    with tempfile.TemporaryDirectory(prefix="pvzrl_metadata_test_") as temp_dir:
        root = Path(temp_dir)
        conservative_model = fake_model(root, "conservative", conservative_config)
        expanded_model = fake_model(root, "expanded", expanded_config)
        missing_model = fake_model(root, "missing_metadata", None)

        checks = [
            (
                "A conservative model + conservative seed list passes",
                validate_model_metadata(conservative_model, conservative_config, model_action_count=201),
                True,
                None,
            ),
            (
                "B conservative model + expanded seed list fails",
                validate_model_metadata(conservative_model, expanded_config, model_action_count=201),
                False,
                "seed_list_mismatch",
            ),
            (
                "C expanded model + expanded seed list passes",
                validate_model_metadata(expanded_model, expanded_config, model_action_count=201),
                True,
                None,
            ),
            (
                "D expanded model + conservative seed list fails",
                validate_model_metadata(expanded_model, conservative_config, model_action_count=201),
                False,
                "seed_list_mismatch",
            ),
            (
                "E missing metadata fails cleanly",
                validate_model_metadata(missing_model, conservative_config, model_action_count=201),
                False,
                "missing_model_metadata",
            ),
            (
                "F action_count mismatch fails cleanly",
                validate_model_metadata(expanded_model, expanded_config, model_action_count=202),
                False,
                "action_count_mismatch",
            ),
        ]

        for name, report, expected_ok, expected_reason in checks:
            results.append(assert_case(name, report.ok, report.blocked_reason, expected_ok, expected_reason))

        pass_report = checks[0][1]
        failure_report = checks[1][1]
        live_status = model_compatibility_live_status(pass_report)
        results.append(
            assert_case(
                "G live_status model_compatibility includes core fields",
                bool(
                    live_status.get("compatible") is True
                    and live_status.get("model_seed_list") == CONSERVATIVE_SEEDS
                    and live_status.get("env_seed_list") == CONSERVATIVE_SEEDS
                    and live_status.get("model_action_count") == 201
                    and live_status.get("env_action_count") == 201
                ),
                None,
                True,
                None,
            )
        )
        examples = {
            "compatibility_pass": model_compatibility_live_status(pass_report),
            "compatibility_failure_text": format_compatibility_failure(failure_report),
        }

    payload = {"ok": all(result["passed"] for result in results), "results": results, "examples": examples}
    print(json.dumps(payload, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
