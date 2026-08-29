"""Model/environment compatibility metadata for PvZRL policies."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pvzrl_action_space import (
    ACTION_SPACE_ADVENTURE_14_IDENTITY,
    ADVENTURE_IDENTITY_MAX_SEED_SLOTS,
    action_count_for_config,
    build_action_space_spec,
    normalize_action_space_mode,
    spec_from_config,
)
from pvzrl_observation_layout import observation_shape_for_config
from pvzrl_rewards import REWARD_POLICY_VERSION


MODEL_METADATA_FILENAME = "model_metadata.json"
MODEL_METADATA_VERSION = 1

BLOCKED_MISSING_METADATA = "missing_model_metadata"
BLOCKED_ACTION_COUNT = "action_count_mismatch"
BLOCKED_ACTION_DECODER = "action_decoder_mismatch"
BLOCKED_OBSERVATION = "observation_version_mismatch"
BLOCKED_MAX_SEED_SLOTS = "max_seed_slots_mismatch"
BLOCKED_DYNAMIC_SEED_SLOTS = "dynamic_seed_slots_mismatch"
BLOCKED_ACTION_SPACE_MODE = "action_space_mode_mismatch"
BLOCKED_METADATA_VERSION = "metadata_version_mismatch"
BLOCKED_IDENTITY_SEED_SLOTS = "identity_seed_slots_mismatch"
BLOCKED_OBSERVATION_SHAPE = "observation_shape_mismatch"
BLOCKED_PLACEMENT_ACTION_RANGE = "placement_action_range_mismatch"
BLOCKED_BOARD_GEOMETRY = "board_geometry_mismatch"
BLOCKED_MODEL_FAMILY = "model_family_mismatch"


@dataclass
class CompatibilityCheck:
    """Structured compatibility report.

    ``compatible/model_metadata/env_metadata`` are the canonical report.  The
    ``ok/expected/actual`` projections remain in-process aliases for current
    callers and live-status construction.
    """

    ok: bool
    blocked_reason: Optional[str] = None
    details: str = ""
    metadata_path: str = ""
    metadata_inferred: bool = False
    expected: Dict[str, Any] = field(default_factory=dict)
    actual: Dict[str, Any] = field(default_factory=dict)
    model_metadata: Dict[str, Any] = field(default_factory=dict)
    env_metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def compatible(self) -> bool:
        return self.ok

    def to_dict(self) -> Dict[str, Any]:
        model_metadata = self.model_metadata or self.actual
        env_metadata = self.env_metadata or self.expected
        return {
            "compatible": self.ok,
            "blocked_reason": self.blocked_reason,
            "model_metadata": model_metadata,
            "env_metadata": env_metadata,
            "warnings": list(self.warnings),
            "metadata_path": self.metadata_path,
            "metadata_inferred": self.metadata_inferred,
            "details": self.details,
            # Current in-process projections used by training/status callers.
            "ok": self.ok,
            "expected": self.expected or env_metadata,
            "actual": self.actual or model_metadata,
        }


def normalized_seed_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(seed).strip() for seed in value if str(seed).strip()]
    if isinstance(value, str):
        seeds: List[str] = []
        for part in value.split(","):
            token = part.strip()
            if token:
                seeds.append(token)
        return seeds
    return []


def normalized_plant_types(value: Any) -> List[int]:
    if not isinstance(value, list):
        return []
    plant_types: List[int] = []
    for item in value:
        try:
            plant_types.append(int(item))
        except (TypeError, ValueError):
            return []
    return plant_types


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalized_int_list(value: Any) -> List[int]:
    if not isinstance(value, (list, tuple)):
        return []
    result: List[int] = []
    for item in value:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            return []
    return result


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def model_metadata_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    config = apply_model_metadata_defaults(config)
    spec = spec_from_config(config)
    seed_list = normalized_seed_list(config.get("seed_list", []))
    plant_types = normalized_plant_types(config.get("plant_types", []))
    metadata: Dict[str, Any] = {
        "metadata_version": MODEL_METADATA_VERSION,
        "model_family": str(config.get("model_family") or ""),
        "seed_list": seed_list,
        "plant_types": plant_types,
        "reward_policy_version": str(
            config.get("reward_policy_version") or REWARD_POLICY_VERSION
        ),
        **spec.to_metadata(),
        "observation_shape": observation_shape_from_config(config),
    }
    if config.get("created_at"):
        metadata["created_at"] = str(config.get("created_at"))
    if config.get("config_path"):
        metadata["config_path"] = str(config.get("config_path"))
    if config.get("total_timesteps") is not None:
        metadata["total_timesteps"] = _optional_int(config.get("total_timesteps"))
    for key in (
        "run_mode",
        "checkpoint_warm_start",
        "warm_start_used",
        "checkpoint_warm_start_reason",
        "resume_training",
        "resume_model_path",
        "resume_source_model_family",
        "scratch_initialization",
        "initial_loadout",
        "configured_seed_list",
        "seed_order_source",
        "seed_order_preserved",
        "randomize_seed_order",
        "max_seed_slots",
        "unlock_aware_seed_curriculum",
        "seed_curriculum",
        "unlock_introduction_delay",
        "new_plant_min_inclusion_prob",
        "active_seed_slots_at_start",
        "adventure_frontier_sample_prob",
        "adventure_recent_cleared_sample_prob",
        "adventure_maintenance_sample_prob",
        "adventure_frontier_win_streak_required",
        "adventure_replay_cleared_levels",
        "tactical_masks",
        "wallnut_tactical_mask",
        "cherrybomb_tactical_mask",
    ):
        if key in config:
            metadata[key] = config.get(key)
    return metadata


def apply_model_metadata_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    updated = dict(config)
    mode = normalize_action_space_mode(
        updated.get("action_space_mode", ACTION_SPACE_ADVENTURE_14_IDENTITY)
    )
    plant_types = normalized_plant_types(updated.get("plant_types", []))
    max_seed_slots = updated.get("max_seed_slots", ADVENTURE_IDENTITY_MAX_SEED_SLOTS)
    if max_seed_slots is None:
        max_seed_slots = ADVENTURE_IDENTITY_MAX_SEED_SLOTS
    spec = build_action_space_spec(
        mode=mode,
        plant_types=plant_types,
        max_seed_slots=int(max_seed_slots),
        rows=int(updated.get("row_count", updated.get("rows", 6)) or 6),
        cols=int(updated.get("column_count", updated.get("cols", 10)) or 10),
    )
    updated.update(spec.to_metadata())
    updated["action_space_mode"] = mode
    updated["dynamic_seed_slots"] = spec.dynamic_seed_slots
    updated["identity_seed_slots"] = spec.identity_seed_slots
    return updated


def env_metadata_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    updated = apply_model_metadata_defaults(config)
    spec = spec_from_config(updated)
    seed_list = normalized_seed_list(updated.get("seed_list", []))
    plant_types = normalized_plant_types(updated.get("plant_types", []))
    return {
        "resolved_seed_list": seed_list,
        "resolved_plant_types": plant_types,
        "env_action_count": int(spec.action_count),
        "action_space_mode": spec.mode,
        "max_seed_slots": int(spec.max_seed_slots),
        "dynamic_seed_slots": bool(spec.dynamic_seed_slots),
        "identity_seed_slots": bool(spec.identity_seed_slots),
        "observation_version": spec.observation_version,
        "action_decoder_version": spec.action_decoder_version,
        "decoder_wait_action": int(spec.wait_action),
        "placement_action_range": [int(spec.placement_action_min), int(spec.placement_action_max)],
        "rows": int(spec.rows),
        "cols": int(spec.cols),
        "cells_per_seed_slot": int(spec.rows) * int(spec.cols),
        "observation_shape": observation_shape_from_config(updated),
        "model_family": str(updated.get("model_family") or ""),
        "reward_policy_version": str(
            updated.get("reward_policy_version") or REWARD_POLICY_VERSION
        ),
    }


def observation_shape_from_config(config: Dict[str, Any]) -> List[int]:
    updated = apply_model_metadata_defaults(config)
    return [int(value) for value in observation_shape_for_config(updated)]


def expected_model_metadata_from_env(env_metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "metadata_version": MODEL_METADATA_VERSION,
        "model_family": str(env_metadata.get("model_family") or ""),
        "seed_list": normalized_seed_list(env_metadata.get("resolved_seed_list", [])),
        "plant_types": normalized_plant_types(env_metadata.get("resolved_plant_types", [])),
        "action_space_mode": str(
            env_metadata.get("action_space_mode") or ACTION_SPACE_ADVENTURE_14_IDENTITY
        ),
        "action_count": _optional_int(env_metadata.get("env_action_count")),
        "max_seed_slots": _optional_int(env_metadata.get("max_seed_slots")),
        "dynamic_seed_slots": bool(env_metadata.get("dynamic_seed_slots", False)),
        "identity_seed_slots": bool(env_metadata.get("identity_seed_slots", False)),
        "observation_version": str(env_metadata.get("observation_version") or ""),
        "action_decoder_version": str(env_metadata.get("action_decoder_version") or ""),
        "decoder_wait_action": _optional_int(env_metadata.get("decoder_wait_action")),
        "placement_action_range": list(env_metadata.get("placement_action_range", []) or []),
        "rows": _optional_int(env_metadata.get("rows")),
        "cols": _optional_int(env_metadata.get("cols")),
        "cells_per_seed_slot": _optional_int(env_metadata.get("cells_per_seed_slot")),
        "observation_shape": _normalized_int_list(env_metadata.get("observation_shape")),
        "reward_policy_version": str(
            env_metadata.get("reward_policy_version") or REWARD_POLICY_VERSION
        ),
    }


def write_model_metadata(
    run_dir: Path,
    config: Dict[str, Any],
    model_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
) -> Path:
    metadata_config = dict(config)
    metadata_config["created_at"] = metadata_config.get("created_at") or _utc_now()
    if config_path is not None:
        metadata_config["config_path"] = str(config_path)
    metadata = model_metadata_from_config(metadata_config)
    if model_path is not None:
        metadata["model_path"] = str(model_path)
    path = run_dir / MODEL_METADATA_FILENAME
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def _model_metadata_dirs(model_path: Path) -> List[Path]:
    dirs: List[Path] = [model_path.parent]
    if model_path.parent.name.lower() == "checkpoints":
        dirs.append(model_path.parent.parent)
    return dirs


def model_metadata_candidates(model_path: Path) -> List[Path]:
    dirs = _model_metadata_dirs(model_path)
    candidates: List[Path] = []
    for directory in dirs:
        metadata = directory / MODEL_METADATA_FILENAME
        if metadata not in candidates:
            candidates.append(metadata)
    return candidates


def _load_json(path: Path) -> Tuple[Optional[Dict[str, Any]], str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, f"invalid_json:{exc}"
    except OSError as exc:
        return None, f"read_failed:{exc}"
    if not isinstance(value, dict):
        return None, "invalid_json:not_object"
    return value, ""


def load_model_metadata(
    model_path: Path,
) -> Tuple[Optional[Path], Optional[Dict[str, Any]], bool, str]:
    for candidate in model_metadata_candidates(model_path):
        if not candidate.exists():
            continue
        raw, load_error = _load_json(candidate)
        if raw is None:
            return candidate, None, False, load_error or BLOCKED_MISSING_METADATA
        return candidate, raw, False, ""
    return None, None, False, BLOCKED_MISSING_METADATA


def _compatibility_failure(
    blocked_reason: str,
    details: str,
    metadata_path: Optional[Path],
    inferred: bool,
    expected: Dict[str, Any],
    actual: Dict[str, Any],
    env_metadata: Dict[str, Any],
    warnings: Optional[List[str]] = None,
) -> CompatibilityCheck:
    return CompatibilityCheck(
        ok=False,
        blocked_reason=blocked_reason,
        details=details,
        metadata_path=str(metadata_path) if metadata_path is not None else "",
        metadata_inferred=inferred,
        expected=expected,
        actual=actual,
        model_metadata=actual,
        env_metadata=env_metadata,
        warnings=list(warnings or []),
    )


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def validate_model_metadata(
    model_path: Path,
    expected_config: Optional[Dict[str, Any]] = None,
    *,
    model_action_count: Optional[int] = None,
    model_observation_shape: Optional[Any] = None,
    env_metadata: Optional[Dict[str, Any]] = None,
) -> CompatibilityCheck:
    if env_metadata is None:
        if expected_config is None:
            raise ValueError("expected_config or env_metadata is required")
        env_metadata = env_metadata_from_config(expected_config)
    expected = expected_model_metadata_from_env(env_metadata)
    metadata_path, actual, inferred, load_blocked_reason = load_model_metadata(model_path)
    warnings: List[str] = []
    if actual is None:
        return _compatibility_failure(
            load_blocked_reason or BLOCKED_MISSING_METADATA,
            f"No canonical {MODEL_METADATA_FILENAME} found for {model_path}.",
            metadata_path,
            inferred,
            expected,
            {},
            env_metadata,
            warnings,
        )

    actual = dict(actual)
    if model_observation_shape is not None:
        actual["loaded_observation_shape"] = _normalized_int_list(model_observation_shape)

    model_reward_policy = str(
        actual.get("reward_policy_version") or "legacy_or_unknown"
    )
    env_reward_policy = str(
        env_metadata.get("reward_policy_version") or REWARD_POLICY_VERSION
    )
    if model_reward_policy != env_reward_policy:
        warnings.append(
            "reward_policy_version_mismatch:"
            f"model={model_reward_policy},environment={env_reward_policy};"
            " continuing because reward policy does not change model dimensions"
        )

    metadata_version = _optional_int(actual.get("metadata_version"))
    if metadata_version != MODEL_METADATA_VERSION:
        return _compatibility_failure(
            BLOCKED_METADATA_VERSION,
            f"metadata_version: model={actual.get('metadata_version')!r}, supported={MODEL_METADATA_VERSION}",
            metadata_path,
            inferred,
            expected,
            actual,
            env_metadata,
            warnings,
        )

    env_action_count = _optional_int(env_metadata.get("env_action_count"))
    if env_action_count is None:
        return _compatibility_failure(
            BLOCKED_ACTION_COUNT,
            "environment metadata does not include env_action_count",
            metadata_path,
            inferred,
            expected,
            actual,
            env_metadata,
            warnings,
        )

    if model_action_count is not None and int(model_action_count) != int(env_action_count):
        return _compatibility_failure(
            BLOCKED_ACTION_COUNT,
            f"model action_space.n={model_action_count}, environment env_action_count={env_action_count}",
            metadata_path,
            inferred,
            expected,
            actual,
            env_metadata,
            warnings,
        )

    actual_action_count = _optional_int(actual.get("action_count"))
    if actual_action_count is None or actual_action_count != env_action_count:
        return _compatibility_failure(
            BLOCKED_ACTION_COUNT,
            f"model action_count={actual.get('action_count')!r}, environment env_action_count={env_action_count}",
            metadata_path,
            inferred,
            expected,
            actual,
            env_metadata,
            warnings,
        )

    env_mode = normalize_action_space_mode(
        env_metadata.get("action_space_mode", ACTION_SPACE_ADVENTURE_14_IDENTITY)
    )
    actual_mode_value = actual.get("action_space_mode")
    try:
        if actual_mode_value in (None, ""):
            raise ValueError("missing action_space_mode")
        actual_mode = normalize_action_space_mode(actual_mode_value)
    except ValueError:
        return _compatibility_failure(
            BLOCKED_ACTION_SPACE_MODE,
            f"unsupported model action_space_mode={actual.get('action_space_mode')!r}",
            metadata_path,
            inferred,
            expected,
            actual,
            env_metadata,
            warnings,
        )

    actual_decoder = str(actual.get("action_decoder_version", ""))
    env_decoder = str(env_metadata.get("action_decoder_version", ""))
    if actual_decoder != env_decoder:
        return _compatibility_failure(
            BLOCKED_ACTION_DECODER,
            f"action_decoder_version: model={actual_decoder!r}, environment={env_decoder!r}",
            metadata_path,
            inferred,
            expected,
            actual,
            env_metadata,
            warnings,
        )

    actual_observation = str(actual.get("observation_version", ""))
    env_observation = str(env_metadata.get("observation_version", ""))
    if actual_observation != env_observation:
        return _compatibility_failure(
            BLOCKED_OBSERVATION,
            f"observation_version: model={actual_observation!r}, environment={env_observation!r}",
            metadata_path,
            inferred,
            expected,
            actual,
            env_metadata,
            warnings,
        )

    expected_observation_shape = _normalized_int_list(env_metadata.get("observation_shape"))
    declared_observation_shape = _normalized_int_list(actual.get("observation_shape"))
    if "observation_shape" in actual and not declared_observation_shape:
        return _compatibility_failure(
            BLOCKED_OBSERVATION_SHAPE,
            f"metadata.observation_shape is malformed: model={actual.get('observation_shape')!r}",
            metadata_path,
            inferred,
            expected,
            actual,
            env_metadata,
            warnings,
        )
    if declared_observation_shape and declared_observation_shape != expected_observation_shape:
        return _compatibility_failure(
            BLOCKED_OBSERVATION_SHAPE,
            f"metadata.observation_shape: model={declared_observation_shape}, environment={expected_observation_shape}",
            metadata_path,
            inferred,
            expected,
            actual,
            env_metadata,
            warnings,
        )
    loaded_observation_shape = _normalized_int_list(model_observation_shape)
    if model_observation_shape is not None and (
        not expected_observation_shape or loaded_observation_shape != expected_observation_shape
    ):
        return _compatibility_failure(
            BLOCKED_OBSERVATION_SHAPE,
            f"observation_space.shape: model={loaded_observation_shape}, environment={expected_observation_shape}",
            metadata_path,
            inferred,
            expected,
            actual,
            env_metadata,
            warnings,
        )

    actual_max_seed_slots = _optional_int(actual.get("max_seed_slots"))
    env_max_seed_slots = _optional_int(env_metadata.get("max_seed_slots"))
    if actual_max_seed_slots is None or actual_max_seed_slots != env_max_seed_slots:
        return _compatibility_failure(
            BLOCKED_MAX_SEED_SLOTS,
            f"max_seed_slots: model={actual.get('max_seed_slots')!r}, environment={env_max_seed_slots}",
            metadata_path,
            inferred,
            expected,
            actual,
            env_metadata,
            warnings,
        )

    actual_dynamic_seed_slots = _bool_value(actual.get("dynamic_seed_slots", False))
    env_dynamic_seed_slots = _bool_value(env_metadata.get("dynamic_seed_slots", False))
    if actual_dynamic_seed_slots != env_dynamic_seed_slots:
        return _compatibility_failure(
            BLOCKED_DYNAMIC_SEED_SLOTS,
            f"dynamic_seed_slots: model={actual_dynamic_seed_slots}, environment={env_dynamic_seed_slots}",
            metadata_path,
            inferred,
            expected,
            actual,
            env_metadata,
            warnings,
        )

    actual_identity_seed_slots = _bool_value(actual.get("identity_seed_slots", False))
    env_identity_seed_slots = _bool_value(env_metadata.get("identity_seed_slots", False))
    if actual_identity_seed_slots != env_identity_seed_slots:
        return _compatibility_failure(
            BLOCKED_IDENTITY_SEED_SLOTS,
            f"identity_seed_slots: model={actual_identity_seed_slots}, environment={env_identity_seed_slots}",
            metadata_path,
            inferred,
            expected,
            actual,
            env_metadata,
            warnings,
        )

    if actual_mode != env_mode:
        return _compatibility_failure(
            BLOCKED_ACTION_SPACE_MODE,
            f"action_space_mode: model={actual_mode!r}, environment={env_mode!r}",
            metadata_path,
            inferred,
            expected,
            actual,
            env_metadata,
            warnings,
        )

    actual_wait = _optional_int(actual.get("decoder_wait_action"))
    env_wait = _optional_int(env_metadata.get("decoder_wait_action"))
    if actual_wait is None or env_wait is None or actual_wait != env_wait:
        return _compatibility_failure(
            BLOCKED_ACTION_DECODER,
            f"decoder_wait_action: model={actual_wait}, environment={env_wait}",
            metadata_path,
            inferred,
            expected,
            actual,
            env_metadata,
            warnings,
        )

    actual_placement_range = _normalized_int_list(actual.get("placement_action_range"))
    env_placement_range = _normalized_int_list(env_metadata.get("placement_action_range"))
    if actual_placement_range != env_placement_range or len(actual_placement_range) != 2:
        return _compatibility_failure(
            BLOCKED_PLACEMENT_ACTION_RANGE,
            f"placement_action_range: model={actual_placement_range}, environment={env_placement_range}",
            metadata_path,
            inferred,
            expected,
            actual,
            env_metadata,
            warnings,
        )

    for field_name in ("rows", "cols", "cells_per_seed_slot"):
        actual_value = _optional_int(actual.get(field_name))
        env_value = _optional_int(env_metadata.get(field_name))
        if actual_value is None or env_value is None or actual_value != env_value:
            return _compatibility_failure(
                BLOCKED_BOARD_GEOMETRY,
                f"{field_name}: model={actual.get(field_name)!r}, environment={env_value}",
                metadata_path,
                inferred,
                expected,
                actual,
                env_metadata,
                warnings,
            )

    model_family = str(actual.get("model_family") or "")
    env_family = str(env_metadata.get("model_family") or "")
    if model_family and env_family and model_family != env_family:
        return _compatibility_failure(
            BLOCKED_MODEL_FAMILY,
            f"model_family: model={model_family!r}, environment={env_family!r}",
            metadata_path,
            inferred,
            expected,
            actual,
            env_metadata,
            warnings,
        )

    return CompatibilityCheck(
        ok=True,
        blocked_reason=None,
        metadata_path=str(metadata_path) if metadata_path is not None else "",
        metadata_inferred=inferred,
        expected=expected,
        actual=actual,
        model_metadata=actual,
        env_metadata=env_metadata,
        warnings=warnings,
    )


def _join_seed_list(values: Any) -> str:
    seeds = normalized_seed_list(values)
    return ",".join(seeds) if seeds else "-"


def _join_plant_types(values: Any) -> str:
    plant_types = normalized_plant_types(values)
    return ",".join(str(value) for value in plant_types) if plant_types else "-"


def format_compatibility_failure(result: CompatibilityCheck) -> str:
    model_metadata = result.model_metadata or result.actual or {}
    env_metadata = result.env_metadata or result.expected or {}
    lines = [
        "ERROR: Model/environment compatibility check failed.",
        f"blocked_reason: {result.blocked_reason or BLOCKED_MISSING_METADATA}",
        "",
        "model seed_list:",
        f"  {_join_seed_list(model_metadata.get('seed_list', []))}",
        "",
        "environment seed_list:",
        f"  {_join_seed_list(env_metadata.get('resolved_seed_list', env_metadata.get('seed_list', [])))}",
        "",
        "model plant_types:",
        f"  {_join_plant_types(model_metadata.get('plant_types', []))}",
        "",
        "environment plant_types:",
        f"  {_join_plant_types(env_metadata.get('resolved_plant_types', env_metadata.get('plant_types', [])))}",
        "",
        "model action_count:",
        f"  {model_metadata.get('action_count', '-')}",
        "",
        "environment action_count:",
        f"  {env_metadata.get('env_action_count', env_metadata.get('action_count', '-'))}",
    ]
    if result.details:
        lines.extend(["", f"details: {result.details}"])
    if result.metadata_path:
        lines.extend(["", f"metadata_path: {result.metadata_path}"])
    lines.extend(
        [
            "",
            "This model cannot run under a different Generalist action or observation contract.",
            "Use the current Adventure Generalist checkpoint family and configuration.",
        ]
    )
    return "\n".join(lines)


def model_compatibility_live_status(result: CompatibilityCheck) -> Dict[str, Any]:
    model_metadata = result.model_metadata or result.actual or {}
    env_metadata = result.env_metadata or result.expected or {}
    return {
        "compatible": bool(result.ok),
        "blocked_reason": result.blocked_reason,
        "model_family": str(model_metadata.get("model_family") or env_metadata.get("model_family") or ""),
        "model_seed_list": normalized_seed_list(model_metadata.get("seed_list", [])),
        "env_seed_list": normalized_seed_list(env_metadata.get("resolved_seed_list", env_metadata.get("seed_list", []))),
        "model_plant_types": normalized_plant_types(model_metadata.get("plant_types", [])),
        "env_plant_types": normalized_plant_types(env_metadata.get("resolved_plant_types", env_metadata.get("plant_types", []))),
        "model_action_count": _optional_int(model_metadata.get("action_count")),
        "env_action_count": _optional_int(env_metadata.get("env_action_count", env_metadata.get("action_count"))),
        "action_space_mode": str(env_metadata.get("action_space_mode") or model_metadata.get("action_space_mode") or ""),
        "action_decoder_version": str(env_metadata.get("action_decoder_version") or model_metadata.get("action_decoder_version") or ""),
        "observation_version": str(env_metadata.get("observation_version") or model_metadata.get("observation_version") or ""),
        "max_seed_slots": _optional_int(env_metadata.get("max_seed_slots", model_metadata.get("max_seed_slots"))),
        "dynamic_seed_slots": _bool_value(env_metadata.get("dynamic_seed_slots", model_metadata.get("dynamic_seed_slots", False))),
        "identity_seed_slots": _bool_value(env_metadata.get("identity_seed_slots", model_metadata.get("identity_seed_slots", False))),
        "model_observation_shape": _normalized_int_list(
            model_metadata.get("loaded_observation_shape", model_metadata.get("observation_shape", []))
        ),
        "env_observation_shape": _normalized_int_list(env_metadata.get("observation_shape", [])),
        "model_reward_policy_version": str(
            model_metadata.get("reward_policy_version") or "legacy_or_unknown"
        ),
        "env_reward_policy_version": str(
            env_metadata.get("reward_policy_version") or REWARD_POLICY_VERSION
        ),
        "reward_policy_version_mismatch": str(
            model_metadata.get("reward_policy_version") or "legacy_or_unknown"
        )
        != str(env_metadata.get("reward_policy_version") or REWARD_POLICY_VERSION),
        "metadata_path": result.metadata_path,
        "metadata_inferred": bool(result.metadata_inferred),
        "warnings": list(result.warnings),
    }


def expected_action_count(expected_config: Dict[str, Any]) -> int:
    return action_count_for_config(apply_model_metadata_defaults(expected_config))
