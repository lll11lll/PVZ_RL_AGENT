"""Bridge-free compatibility checks for Adventure Generalist metadata."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from pvzrl_action_space import (
    ACTION_SPACE_ADVENTURE_14_IDENTITY,
    ADVENTURE_IDENTITY_ACTION_COUNT,
    ADVENTURE_IDENTITY_ACTION_DECODER_VERSION,
    ADVENTURE_IDENTITY_MAX_SEED_SLOTS,
    ADVENTURE_IDENTITY_OBSERVATION_VERSION,
    DEFAULT_COLS,
    DEFAULT_ROWS,
)
from pvzrl_adventure_generalist import ADVENTURE_GENERALIST_MODEL_FAMILY
from pvzrl_model_metadata import (
    format_compatibility_failure,
    model_compatibility_live_status,
    validate_model_metadata,
    write_model_metadata,
)
from pvzrl_observation_layout import observation_shape_for_config
from pvzrl_rewards import REWARD_POLICY_VERSION


MODEL_FAMILY = ADVENTURE_GENERALIST_MODEL_FAMILY
LEGACY_MODEL_FAMILY = "ppo_adventure_generalist_14slot_identity_v1"
INITIAL_SEEDS = ["SunFlower", "SunFlower", "Peashooter", "Peashooter"]
INITIAL_PLANT_TYPES = [1, 1, 0, 0]
OBSERVATION_SHAPE = observation_shape_for_config(
    {
        "action_space_mode": ACTION_SPACE_ADVENTURE_14_IDENTITY,
        "max_seed_slots": ADVENTURE_IDENTITY_MAX_SEED_SLOTS,
        "row_count": DEFAULT_ROWS,
        "column_count": DEFAULT_COLS,
    }
)


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
        "action_space_mode": ACTION_SPACE_ADVENTURE_14_IDENTITY,
        "max_seed_slots": ADVENTURE_IDENTITY_MAX_SEED_SLOTS,
        "row_count": DEFAULT_ROWS,
        "column_count": DEFAULT_COLS,
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


def test_reward_policy_mismatch_warns_but_does_not_block() -> None:
    with tempfile.TemporaryDirectory(prefix="pvzrl_reward_metadata_test_") as temp_dir:
        root = Path(temp_dir)
        model_path = fake_model(root, "legacy_reward", config())
        metadata_path = model_path.parent / "model_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["reward_policy_version"] = "legacy_strategy_v1"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

        report = validate_model_metadata(
            model_path,
            {**config(), "reward_policy_version": REWARD_POLICY_VERSION},
            model_action_count=ADVENTURE_IDENTITY_ACTION_COUNT,
            model_observation_shape=OBSERVATION_SHAPE,
        )

    assert report.ok is True
    assert report.blocked_reason is None
    assert any("reward_policy_version_mismatch" in item for item in report.warnings)
    live = model_compatibility_live_status(report)
    assert live["model_reward_policy_version"] == "legacy_strategy_v1"
    assert live["env_reward_policy_version"] == REWARD_POLICY_VERSION
    assert live["reward_policy_version_mismatch"] is True


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
        legacy_model = fake_model(
            root,
            "legacy_v1_family",
            config(model_family=LEGACY_MODEL_FAMILY),
        )
        missing_model = fake_model(root, "missing_metadata", None)

        checks = [
            (
                "A Generalist model and canonical contract pass",
                validate_model_metadata(
                    generalist_model,
                    initial_config,
                    model_action_count=ADVENTURE_IDENTITY_ACTION_COUNT,
                    model_observation_shape=OBSERVATION_SHAPE,
                ),
                True,
                None,
            ),
            (
                "B progression loadout changes preserve identity compatibility",
                validate_model_metadata(
                    generalist_model,
                    progressed_config,
                    model_action_count=ADVENTURE_IDENTITY_ACTION_COUNT,
                    model_observation_shape=OBSERVATION_SHAPE,
                ),
                True,
                None,
            ),
            (
                "C obsolete model family fails",
                validate_model_metadata(
                    generalist_model,
                    wrong_family_config,
                    model_action_count=ADVENTURE_IDENTITY_ACTION_COUNT,
                    model_observation_shape=OBSERVATION_SHAPE,
                ),
                False,
                "model_family_mismatch",
            ),
            (
                "C2 historical v1 checkpoint family fails",
                validate_model_metadata(
                    legacy_model,
                    initial_config,
                    model_action_count=ADVENTURE_IDENTITY_ACTION_COUNT,
                    model_observation_shape=OBSERVATION_SHAPE,
                ),
                False,
                "model_family_mismatch",
            ),
            (
                "D missing canonical metadata fails",
                validate_model_metadata(
                    missing_model,
                    initial_config,
                    model_action_count=ADVENTURE_IDENTITY_ACTION_COUNT,
                    model_observation_shape=OBSERVATION_SHAPE,
                ),
                False,
                "missing_model_metadata",
            ),
            (
                "E loaded action count mismatch fails",
                validate_model_metadata(
                    generalist_model,
                    initial_config,
                    model_action_count=ADVENTURE_IDENTITY_ACTION_COUNT - 1,
                    model_observation_shape=OBSERVATION_SHAPE,
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
                    and live_status.get("model_action_count") == ADVENTURE_IDENTITY_ACTION_COUNT
                    and live_status.get("env_action_count") == ADVENTURE_IDENTITY_ACTION_COUNT
                    and live_status.get("action_space_mode") == ACTION_SPACE_ADVENTURE_14_IDENTITY
                    and live_status.get("action_decoder_version")
                    == ADVENTURE_IDENTITY_ACTION_DECODER_VERSION
                    and live_status.get("observation_version")
                    == ADVENTURE_IDENTITY_OBSERVATION_VERSION
                    and live_status.get("model_observation_shape") == list(OBSERVATION_SHAPE)
                    and live_status.get("env_observation_shape") == list(OBSERVATION_SHAPE)
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
