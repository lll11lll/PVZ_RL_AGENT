"""Bridge-free compatibility checks for Adventure Generalist metadata."""

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


MODEL_FAMILY = "ppo_adventure_generalist_14slot_identity_v1"
INITIAL_SEEDS = ["SunFlower", "SunFlower", "Peashooter", "Peashooter"]
INITIAL_PLANT_TYPES = [1, 1, 0, 0]


def config(
    seed_list: List[str] = INITIAL_SEEDS,
    plant_types: List[int] = INITIAL_PLANT_TYPES,
    *,
    model_family: str = MODEL_FAMILY,
) -> Dict[str, Any]:
    return {
        "model_family": model_family,
        "seed_list": list(seed_list),
        "plant_types": list(plant_types),
        "action_space_mode": "adventure_14slot_identity",
        "max_seed_slots": 14,
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
        write_model_metadata(
            run_dir,
            metadata_config,
            model_path=model_path,
            config_path=run_dir / "resolved_config.json",
        )
    return model_path


def assert_case(
    name: str,
    actual_ok: bool,
    actual_reason: Optional[str],
    expected_ok: bool,
    expected_reason: Optional[str],
) -> Dict[str, Any]:
    passed = actual_ok == expected_ok and (
        expected_reason is None or actual_reason == expected_reason
    )
    return {
        "case": name,
        "passed": passed,
        "compatible": actual_ok,
        "blocked_reason": actual_reason,
        "expected_compatible": expected_ok,
        "expected_blocked_reason": expected_reason,
    }


def main() -> int:
    initial_config = config()
    # Seed inventory may grow/change across Adventure progression without
    # changing identity-slot action meanings.
    progressed_config = config(
        ["SunFlower", "Peashooter", "WallNut", "CherryBomb"],
        [1, 0, 3, 2],
    )
    wrong_family_config = config(model_family="obsolete-specialist")
    results: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="pvzrl_metadata_test_") as temp_dir:
        root = Path(temp_dir)
        generalist_model = fake_model(root, "generalist", initial_config)
        missing_model = fake_model(root, "missing_metadata", None)

        checks = [
            (
                "A Generalist model and canonical contract pass",
                validate_model_metadata(
                    generalist_model,
                    initial_config,
                    model_action_count=701,
                    model_observation_shape=(4297,),
                ),
                True,
                None,
            ),
            (
                "B progression loadout changes preserve identity compatibility",
                validate_model_metadata(
                    generalist_model,
                    progressed_config,
                    model_action_count=701,
                    model_observation_shape=(4297,),
                ),
                True,
                None,
            ),
            (
                "C obsolete model family fails",
                validate_model_metadata(
                    generalist_model,
                    wrong_family_config,
                    model_action_count=701,
                    model_observation_shape=(4297,),
                ),
                False,
                "model_family_mismatch",
            ),
            (
                "D missing canonical metadata fails",
                validate_model_metadata(
                    missing_model,
                    initial_config,
                    model_action_count=701,
                    model_observation_shape=(4297,),
                ),
                False,
                "missing_model_metadata",
            ),
            (
                "E loaded action count mismatch fails",
                validate_model_metadata(
                    generalist_model,
                    initial_config,
                    model_action_count=700,
                    model_observation_shape=(4297,),
                ),
                False,
                "action_count_mismatch",
            ),
        ]

        for name, report, expected_ok, expected_reason in checks:
            results.append(
                assert_case(
                    name,
                    report.ok,
                    report.blocked_reason,
                    expected_ok,
                    expected_reason,
                )
            )

        pass_report = checks[0][1]
        family_failure = checks[2][1]
        live_status = model_compatibility_live_status(pass_report)
        results.append(
            assert_case(
                "F live status exposes the exact Generalist contract",
                bool(
                    live_status.get("compatible") is True
                    and live_status.get("model_action_count") == 701
                    and live_status.get("env_action_count") == 701
                    and live_status.get("action_space_mode") == "adventure_14slot_identity"
                    and live_status.get("action_decoder_version")
                    == "seedslot14x50_plus_wait_v1"
                    and live_status.get("observation_version")
                    == "adventure_14slot_identity_v1"
                    and live_status.get("model_observation_shape") == [4297]
                    and live_status.get("env_observation_shape") == [4297]
                ),
                None,
                True,
                None,
            )
        )
        examples = {
            "compatibility_pass": live_status,
            "compatibility_failure_text": format_compatibility_failure(family_failure),
        }

    payload = {
        "ok": all(result["passed"] for result in results),
        "results": results,
        "examples": examples,
    }
    print(json.dumps(payload, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
