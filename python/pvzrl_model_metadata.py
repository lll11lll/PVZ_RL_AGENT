"""Model/environment compatibility metadata for PvZRL policies."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pvzrl_action_space import (
    ACTION_SPACE_ADVENTURE_14_IDENTITY,
    ACTION_SPACE_DYNAMIC_14,
    ACTION_SPACE_FIXED,
    action_count_for_config,
    build_action_space_spec,
    normalize_action_space_mode,
    spec_from_config,
)


MODEL_METADATA_FILENAME = "model_metadata.json"
MODEL_METADATA_VERSION = 1
LEGACY_CONFIG_FILENAMES = ("resolved_config.json", "config.json")

BLOCKED_MISSING_METADATA = "missing_model_metadata"
BLOCKED_ACTION_COUNT = "action_count_mismatch"
BLOCKED_SEED_LIST = "seed_list_mismatch"
BLOCKED_PLANT_TYPE = "plant_type_mismatch"
BLOCKED_ACTION_DECODER = "action_decoder_mismatch"
BLOCKED_OBSERVATION = "observation_version_mismatch"
BLOCKED_MAX_SEED_SLOTS = "max_seed_slots_mismatch"
BLOCKED_DYNAMIC_SEED_SLOTS = "dynamic_seed_slots_mismatch"
BLOCKED_ACTION_SPACE_MODE = "action_space_mode_mismatch"


@dataclass
class CompatibilityCheck:
    """Structured compatibility report.

    The legacy ``ok/expected/actual`` fields are kept for existing callers while
    ``compatible/model_metadata/env_metadata`` provide the canonical report.
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
            # Backward-compatible aliases.
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
        **spec.to_metadata(),
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
        "incompatible_with_4slot_specialist",
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
    mode = normalize_action_space_mode(updated.get("action_space_mode", ACTION_SPACE_FIXED))
    plant_types = normalized_plant_types(updated.get("plant_types", []))
    max_seed_slots = updated.get("max_seed_slots")
    if max_seed_slots is None:
        max_seed_slots = 14 if mode in {ACTION_SPACE_DYNAMIC_14, ACTION_SPACE_ADVENTURE_14_IDENTITY} else len(plant_types)
    spec = build_action_space_spec(
        mode=mode,
        plant_types=plant_types,
        max_seed_slots=int(max_seed_slots),
        rows=int(updated.get("row_count", updated.get("rows", 5)) or 5),
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
        "model_family": str(updated.get("model_family") or ""),
    }


def expected_model_metadata_from_env(env_metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "metadata_version": MODEL_METADATA_VERSION,
        "model_family": str(env_metadata.get("model_family") or ""),
        "seed_list": normalized_seed_list(env_metadata.get("resolved_seed_list", [])),
        "plant_types": normalized_plant_types(env_metadata.get("resolved_plant_types", [])),
        "action_space_mode": str(env_metadata.get("action_space_mode") or ACTION_SPACE_FIXED),
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


def model_metadata_candidates(model_path: Path, *, include_legacy: bool = True) -> List[Path]:
    dirs = _model_metadata_dirs(model_path)
    candidates: List[Path] = []
    for directory in dirs:
        metadata = directory / MODEL_METADATA_FILENAME
        if metadata not in candidates:
            candidates.append(metadata)
    if include_legacy:
        for directory in dirs:
            for filename in LEGACY_CONFIG_FILENAMES:
                candidate = directory / filename
                if candidate not in candidates:
                    candidates.append(candidate)
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


def infer_fixed_metadata_from_legacy_config(config: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    seed_list = normalized_seed_list(config.get("seed_list", []))
    plant_types = normalized_plant_types(config.get("plant_types", []))
    raw_action_count = config.get("action_count")
    if not seed_list or not plant_types or raw_action_count is None:
        return None, BLOCKED_MISSING_METADATA
    action_count = _optional_int(raw_action_count)
    if action_count is None:
        return None, BLOCKED_MISSING_METADATA
    expected_action_count = 1 + len(plant_types) * 5 * 10
    if action_count != expected_action_count:
        return None, BLOCKED_ACTION_DECODER
    inferred_config = dict(config)
    inferred_config["seed_list"] = seed_list
    inferred_config["plant_types"] = plant_types
    inferred_config["action_space_mode"] = ACTION_SPACE_FIXED
    inferred_config["max_seed_slots"] = len(plant_types)
    inferred_config["action_count"] = action_count
    metadata = model_metadata_from_config(apply_model_metadata_defaults(inferred_config))
    metadata["metadata_source"] = "inferred_legacy_config"
    return metadata, ""


def load_model_metadata(
    model_path: Path,
    *,
    allow_missing_model_metadata: bool = False,
) -> Tuple[Optional[Path], Optional[Dict[str, Any]], bool, str]:
    for candidate in model_metadata_candidates(model_path, include_legacy=False):
        if not candidate.exists():
            continue
        raw, load_error = _load_json(candidate)
        if raw is None:
            return candidate, None, False, load_error or BLOCKED_MISSING_METADATA
        return candidate, raw, False, ""

    if not allow_missing_model_metadata:
        return None, None, False, BLOCKED_MISSING_METADATA

    saw_legacy = False
    legacy_blocked_reason = BLOCKED_MISSING_METADATA
    for candidate in model_metadata_candidates(model_path, include_legacy=True):
        if candidate.name == MODEL_METADATA_FILENAME or not candidate.exists():
            continue
        raw, load_error = _load_json(candidate)
        if raw is None:
            return candidate, None, False, load_error or BLOCKED_MISSING_METADATA
        saw_legacy = True
        inferred, blocked_reason = infer_fixed_metadata_from_legacy_config(raw)
        if inferred is not None:
            return candidate, inferred, True, ""
        legacy_blocked_reason = blocked_reason or legacy_blocked_reason
    return None, None, False, legacy_blocked_reason if saw_legacy else BLOCKED_MISSING_METADATA


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


def _values_match(left: Any, right: Any) -> bool:
    return left == right


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
    env_metadata: Optional[Dict[str, Any]] = None,
    allow_missing_model_metadata: bool = False,
) -> CompatibilityCheck:
    if env_metadata is None:
        if expected_config is None:
            raise ValueError("expected_config or env_metadata is required")
        env_metadata = env_metadata_from_config(expected_config)
    expected = expected_model_metadata_from_env(env_metadata)
    metadata_path, actual, inferred, load_blocked_reason = load_model_metadata(
        model_path,
        allow_missing_model_metadata=allow_missing_model_metadata,
    )
    warnings: List[str] = []
    if inferred:
        warnings.append("model_metadata_inferred_from_legacy_config")
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

    actual_mode = normalize_action_space_mode(actual.get("action_space_mode", ACTION_SPACE_FIXED))
    env_mode = normalize_action_space_mode(env_metadata.get("action_space_mode", ACTION_SPACE_FIXED))

    model_seed_list = normalized_seed_list(actual.get("seed_list", []))
    env_seed_list = normalized_seed_list(env_metadata.get("resolved_seed_list", []))
    if env_mode != ACTION_SPACE_ADVENTURE_14_IDENTITY and not _values_match(model_seed_list, env_seed_list):
        return _compatibility_failure(
            BLOCKED_SEED_LIST,
            "model seed_list does not match resolved environment seed_list",
            metadata_path,
            inferred,
            expected,
            actual,
            env_metadata,
            warnings,
        )

    model_plant_types = normalized_plant_types(actual.get("plant_types", []))
    env_plant_types = normalized_plant_types(env_metadata.get("resolved_plant_types", []))
    if env_mode != ACTION_SPACE_ADVENTURE_14_IDENTITY and not _values_match(model_plant_types, env_plant_types):
        return _compatibility_failure(
            BLOCKED_PLANT_TYPE,
            "model plant_types do not match resolved environment plant_types",
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
    if actual_wait is not None and env_wait is not None and actual_wait != env_wait:
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

    model_family = str(actual.get("model_family") or "")
    env_family = str(env_metadata.get("model_family") or "")
    if model_family and env_family and model_family != env_family:
        warnings.append(f"model_family_mismatch:model={model_family},environment={env_family}")

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
            "This model cannot be evaluated under a different seed-slot layout because action meanings would change.",
            "Use the correct --seed-list or train a new model family.",
        ]
    )
    return "\n".join(lines)


def compatibility_or_raise(
    model_path: Path,
    expected_config: Dict[str, Any],
    *,
    model_action_count: Optional[int] = None,
    env_metadata: Optional[Dict[str, Any]] = None,
    allow_missing_model_metadata: bool = False,
) -> CompatibilityCheck:
    result = validate_model_metadata(
        model_path,
        expected_config,
        model_action_count=model_action_count,
        env_metadata=env_metadata,
        allow_missing_model_metadata=allow_missing_model_metadata,
    )
    if result.ok:
        return result
    raise SystemExit(format_compatibility_failure(result))


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
        "metadata_path": result.metadata_path,
        "metadata_inferred": bool(result.metadata_inferred),
        "warnings": list(result.warnings),
    }


def expected_action_count(expected_config: Dict[str, Any]) -> int:
    return action_count_for_config(apply_model_metadata_defaults(expected_config))
