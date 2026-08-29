"""Pure GUI-side validation for maintained Full-Adventure launches.

This module is deliberately smaller than the runtime configuration resolver.
It validates user-entered form values before the GUI starts ``train_ppo.py``;
the backend remains authoritative for CLI/JSON precedence and execution.

No function in this module imports Tk, loads a PPO model, contacts Twitch, or
returns credential values.  Model compatibility here means that the canonical
``model_metadata.json`` exactly declares the maintained Full-Adventure v2
contract.  ``MaskablePPO.load`` remains the runtime proof that a checkpoint is
readable.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Generic, Mapping, Optional, Sequence, Tuple, TypeVar

from pvzrl_action_space import (
    ACTION_SPACE_ADVENTURE_14_IDENTITY,
    ADVENTURE_IDENTITY_ACTION_COUNT,
    ADVENTURE_IDENTITY_ACTION_DECODER_VERSION,
    ADVENTURE_IDENTITY_MAX_SEED_SLOTS,
    ADVENTURE_IDENTITY_OBSERVATION_VERSION,
    ADVENTURE_IDENTITY_WAIT_ACTION,
    CELLS_PER_SLOT,
    DEFAULT_COLS,
    DEFAULT_ROWS,
)
from pvzrl_model_metadata import (
    MODEL_METADATA_FILENAME,
    MODEL_METADATA_VERSION,
    load_model_metadata,
    model_metadata_from_config,
    validate_model_metadata,
)
from pvzrl_rewards import REWARD_POLICY_VERSION


FULL_ADVENTURE_MODEL_FAMILY = "ppo_adventure_generalist_14slot_identity_full_v2"
FULL_ADVENTURE_TRAIN_RUN_MODE = "adventure_generalist_14slot_train"
FULL_ADVENTURE_EVAL_RUN_MODE = "adventure_generalist_14slot_eval"
FULL_ADVENTURE_INITIAL_LOADOUT = (
    "SunFlower",
    "SunFlower",
    "Peashooter",
    "Peashooter",
)
FULL_ADVENTURE_INITIAL_PLANT_TYPES = (1, 1, 0, 0)
FULL_ADVENTURE_MAX_LEVELS = 50
FULL_ADVENTURE_OBSERVATION_SHAPE = (4364,)

STREAMER_ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
MOCK_SCRIPT_MAX_BYTES = 16 * 1024 * 1024
MOCK_SCRIPT_MAX_LINES = 100_000
MOCK_SCRIPT_MAX_LINE_CHARS = 4_096
GUI_CONFIG_MAX_BYTES = 2 * 1024 * 1024

_RESERVED_RUN_OUTPUT_NAMES = frozenset(
    {
        "config.json",
        "resolved_config.json",
        MODEL_METADATA_FILENAME.lower(),
        "model.zip",
        "final_model.zip",
        "evaluation.json",
        "streamer_state.json",
        "streamer_cycles.jsonl",
    }
)

DEFAULT_TWITCH_CLIENT_ID_ENV = "PVZRL_TWITCH_CLIENT_ID"
DEFAULT_TWITCH_ACCESS_TOKEN_ENV = "PVZRL_TWITCH_USER_ACCESS_TOKEN"
DEFAULT_TWITCH_BROADCASTER_ID_ENV = "PVZRL_TWITCH_BROADCASTER_USER_ID"
DEFAULT_TWITCH_USER_ID_ENV = "PVZRL_TWITCH_EVENTSUB_USER_ID"
DEFAULT_VIEWER_HASH_SECRET_ENV = "PVZRL_TWITCH_VIEWER_HASH_SECRET"


@dataclass(frozen=True, slots=True)
class FullAdventureContract:
    """Immutable checkpoint-semantic description shown by the GUI."""

    metadata_version: int
    model_family: str
    action_space_mode: str
    action_decoder_version: str
    observation_version: str
    rows: int
    cols: int
    max_seed_slots: int
    cells_per_seed_slot: int
    action_count: int
    wait_action: int
    placement_action_range: Tuple[int, int]
    observation_shape: Tuple[int, ...]
    dynamic_seed_slots: bool
    identity_seed_slots: bool
    initial_loadout: Tuple[str, ...]
    initial_plant_types: Tuple[int, ...]

    def expected_config(self) -> dict[str, Any]:
        """Return a fresh minimal config accepted by metadata helpers."""

        return {
            "model_family": self.model_family,
            "reward_policy_version": REWARD_POLICY_VERSION,
            "seed_list": list(self.initial_loadout),
            "plant_types": list(self.initial_plant_types),
            "action_space_mode": self.action_space_mode,
            "max_seed_slots": self.max_seed_slots,
            "row_count": self.rows,
            "column_count": self.cols,
        }

    def expected_metadata(self) -> dict[str, Any]:
        """Return the canonical semantic metadata (without run artifacts)."""

        metadata = model_metadata_from_config(self.expected_config())
        return {
            "metadata_version": metadata["metadata_version"],
            "model_family": metadata["model_family"],
            "seed_list": list(metadata["seed_list"]),
            "plant_types": list(metadata["plant_types"]),
            "action_space_mode": metadata["action_space_mode"],
            "action_count": metadata["action_count"],
            "max_seed_slots": metadata["max_seed_slots"],
            "dynamic_seed_slots": metadata["dynamic_seed_slots"],
            "identity_seed_slots": metadata["identity_seed_slots"],
            "observation_version": metadata["observation_version"],
            "action_decoder_version": metadata["action_decoder_version"],
            "decoder_wait_action": metadata["decoder_wait_action"],
            "placement_action_range": list(metadata["placement_action_range"]),
            "rows": metadata["rows"],
            "cols": metadata["cols"],
            "cells_per_seed_slot": metadata["cells_per_seed_slot"],
            "observation_shape": list(metadata["observation_shape"]),
        }


FULL_ADVENTURE_CONTRACT = FullAdventureContract(
    metadata_version=MODEL_METADATA_VERSION,
    model_family=FULL_ADVENTURE_MODEL_FAMILY,
    action_space_mode=ACTION_SPACE_ADVENTURE_14_IDENTITY,
    action_decoder_version=ADVENTURE_IDENTITY_ACTION_DECODER_VERSION,
    observation_version=ADVENTURE_IDENTITY_OBSERVATION_VERSION,
    rows=DEFAULT_ROWS,
    cols=DEFAULT_COLS,
    max_seed_slots=ADVENTURE_IDENTITY_MAX_SEED_SLOTS,
    cells_per_seed_slot=CELLS_PER_SLOT,
    action_count=ADVENTURE_IDENTITY_ACTION_COUNT,
    wait_action=ADVENTURE_IDENTITY_WAIT_ACTION,
    placement_action_range=(1, ADVENTURE_IDENTITY_ACTION_COUNT - 1),
    observation_shape=FULL_ADVENTURE_OBSERVATION_SHAPE,
    dynamic_seed_slots=True,
    identity_seed_slots=True,
    initial_loadout=FULL_ADVENTURE_INITIAL_LOADOUT,
    initial_plant_types=FULL_ADVENTURE_INITIAL_PLANT_TYPES,
)


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One actionable, field-addressable form or compatibility issue."""

    code: str
    field: str
    message: str
    severity: str = "error"


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ValidationResult(Generic[T]):
    """Parsed immutable value plus every issue found in one validation pass."""

    value: Optional[T]
    issues: Tuple[ValidationIssue, ...] = ()

    @property
    def ok(self) -> bool:
        return self.value is not None and not any(
            issue.severity == "error" for issue in self.issues
        )

    @property
    def errors(self) -> Tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> Tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")


def _freeze_metadata_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_freeze_metadata_value(item) for item in value)
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _freeze_metadata_value(item) for key, item in value.items()}
        )
    return value


@dataclass(frozen=True, slots=True)
class ModelCompatibilityInspection:
    """Metadata-only compatibility result; never a model-load claim."""

    model_path: Path
    metadata_path: Optional[Path]
    compatible: bool
    issues: Tuple[ValidationIssue, ...]
    declared_metadata: Mapping[str, Any]

    @property
    def ok(self) -> bool:
        return self.compatible

    @property
    def blocked_reason(self) -> Optional[str]:
        return self.issues[0].code if self.issues else None


_METADATA_ISSUE_CODES = {
    "metadata_version": "metadata_version_mismatch",
    "model_family": "model_family_mismatch",
    "seed_list": "seed_list_mismatch",
    "plant_types": "plant_types_mismatch",
    "action_space_mode": "action_space_mode_mismatch",
    "action_count": "action_count_mismatch",
    "max_seed_slots": "max_seed_slots_mismatch",
    "dynamic_seed_slots": "dynamic_seed_slots_mismatch",
    "identity_seed_slots": "identity_seed_slots_mismatch",
    "observation_version": "observation_version_mismatch",
    "action_decoder_version": "action_decoder_mismatch",
    "decoder_wait_action": "action_decoder_mismatch",
    "placement_action_range": "placement_action_range_mismatch",
    "rows": "board_geometry_mismatch",
    "cols": "board_geometry_mismatch",
    "cells_per_seed_slot": "board_geometry_mismatch",
    "observation_shape": "observation_shape_mismatch",
    "run_mode": "run_mode_mismatch",
}


def _bounded_repr(value: Any, limit: int = 120) -> str:
    rendered = repr(value)
    return rendered if len(rendered) <= limit else rendered[: limit - 3] + "..."


def _metadata_values_equal(field: str, actual: Any, expected: Any) -> bool:
    if field in {
        "metadata_version",
        "action_count",
        "max_seed_slots",
        "decoder_wait_action",
        "rows",
        "cols",
        "cells_per_seed_slot",
    }:
        return isinstance(actual, int) and not isinstance(actual, bool) and actual == expected
    if field in {"dynamic_seed_slots", "identity_seed_slots"}:
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return False
        if field in {"plant_types", "placement_action_range", "observation_shape"}:
            return all(
                isinstance(item, int) and not isinstance(item, bool)
                for item in actual
            ) and actual == expected
        return all(isinstance(item, str) for item in actual) and actual == expected
    return isinstance(actual, str) and actual == expected


def inspect_model_compatibility(
    model_path: os.PathLike[str] | str,
    *,
    contract: FullAdventureContract = FULL_ADVENTURE_CONTRACT,
    base_dir: Optional[os.PathLike[str] | str] = None,
) -> ModelCompatibilityInspection:
    """Inspect exact canonical metadata without importing or loading SB3.

    The returned mapping contains only known contract fields.  Arbitrary
    metadata extensions are intentionally not reflected into GUI state.
    """

    raw_text = str(model_path).strip()
    root = Path(base_dir) if base_dir is not None else Path.cwd()
    path = Path(raw_text) if raw_text else root / "<missing-model>"
    if not path.is_absolute():
        path = root / path
    path = path.resolve(strict=False)
    issues: list[ValidationIssue] = []
    if not raw_text:
        issues.append(
            ValidationIssue("model_path_required", "model_path", "Select a model .zip file.")
        )
    elif path.suffix.lower() != ".zip":
        issues.append(
            ValidationIssue("model_path_not_zip", "model_path", "The model path must end in .zip.")
        )
    if raw_text and not path.is_file():
        issues.append(
            ValidationIssue("model_path_missing", "model_path", f"Model file not found: {path}")
        )

    metadata_path, raw_metadata, _inferred, load_error = load_model_metadata(path)
    if raw_metadata is None:
        code = "missing_model_metadata" if load_error == "missing_model_metadata" else "invalid_model_metadata"
        message = (
            f"No canonical {MODEL_METADATA_FILENAME} was found for {path}."
            if code == "missing_model_metadata"
            else f"Could not read canonical {MODEL_METADATA_FILENAME}: {load_error}"
        )
        issues.append(ValidationIssue(code, "model_metadata", message))
        return ModelCompatibilityInspection(
            model_path=path,
            metadata_path=metadata_path,
            compatible=False,
            issues=tuple(issues),
            declared_metadata=MappingProxyType({}),
        )

    expected = contract.expected_metadata()
    declared = {field: raw_metadata.get(field) for field in expected}
    for field, expected_value in expected.items():
        actual_value = raw_metadata.get(field)
        if _metadata_values_equal(field, actual_value, expected_value):
            continue
        issues.append(
            ValidationIssue(
                _METADATA_ISSUE_CODES[field],
                field,
                f"{field} must be {_bounded_repr(expected_value)}; metadata declares "
                f"{_bounded_repr(actual_value)}.",
            )
        )

    declared_run_mode = raw_metadata.get("run_mode")
    if declared_run_mode not in (None, "") and declared_run_mode not in {
        FULL_ADVENTURE_TRAIN_RUN_MODE,
        FULL_ADVENTURE_EVAL_RUN_MODE,
    }:
        declared["run_mode"] = declared_run_mode
        issues.append(
            ValidationIssue(
                _METADATA_ISSUE_CODES["run_mode"],
                "run_mode",
                "run_mode must be a maintained Adventure Generalist train/eval mode when declared.",
            )
        )

    # Retain the backend's compatibility decision as a defense against this
    # GUI checklist drifting behind newly tightened canonical validation.
    canonical = validate_model_metadata(path, contract.expected_config())
    if not canonical.ok and not any(
        issue.code == str(canonical.blocked_reason or "") for issue in issues
    ):
        issues.append(
            ValidationIssue(
                str(canonical.blocked_reason or "model_metadata_incompatible"),
                "model_metadata",
                canonical.details or "Canonical model metadata validation failed.",
            )
        )

    frozen_declared = MappingProxyType(
        {key: _freeze_metadata_value(value) for key, value in declared.items()}
    )
    return ModelCompatibilityInspection(
        model_path=path,
        metadata_path=metadata_path,
        compatible=not issues,
        issues=tuple(issues),
        declared_metadata=frozen_declared,
    )


@dataclass(frozen=True, slots=True)
class TwitchEnvironmentNames:
    client_id: str = DEFAULT_TWITCH_CLIENT_ID_ENV
    access_token: str = DEFAULT_TWITCH_ACCESS_TOKEN_ENV
    broadcaster_id: str = DEFAULT_TWITCH_BROADCASTER_ID_ENV
    user_id: str = DEFAULT_TWITCH_USER_ID_ENV
    viewer_hash_secret: str = DEFAULT_VIEWER_HASH_SECRET_ENV

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "TwitchEnvironmentNames":
        def text(canonical: str, short: str, default: str) -> str:
            value = values.get(canonical, values.get(short, default))
            return str(value or "").strip()

        return cls(
            client_id=text("streamer_twitch_client_id_env", "client_id", DEFAULT_TWITCH_CLIENT_ID_ENV),
            access_token=text(
                "streamer_twitch_access_token_env",
                "access_token",
                DEFAULT_TWITCH_ACCESS_TOKEN_ENV,
            ),
            broadcaster_id=text(
                "streamer_twitch_broadcaster_id_env",
                "broadcaster_id",
                DEFAULT_TWITCH_BROADCASTER_ID_ENV,
            ),
            user_id=text("streamer_twitch_user_id_env", "user_id", DEFAULT_TWITCH_USER_ID_ENV),
            viewer_hash_secret=text(
                "streamer_viewer_hash_secret_env",
                "viewer_hash_secret",
                DEFAULT_VIEWER_HASH_SECRET_ENV,
            ),
        )


@dataclass(frozen=True, slots=True)
class CredentialVariableState:
    role: str
    env_name: str
    required: bool
    valid_name: bool
    present: bool


@dataclass(frozen=True, slots=True)
class CredentialReadiness:
    variables: Tuple[CredentialVariableState, ...]
    issues: Tuple[ValidationIssue, ...]

    @property
    def ready(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def missing_names(self) -> Tuple[str, ...]:
        return tuple(
            state.env_name
            for state in self.variables
            if state.required and state.valid_name and not state.present
        )


def inspect_twitch_credentials(
    env_names: TwitchEnvironmentNames | Mapping[str, Any],
    *,
    environ: Optional[Mapping[str, str]] = None,
    required: bool = True,
) -> CredentialReadiness:
    """Return configured/missing states without ever copying secret values."""

    names = (
        env_names
        if isinstance(env_names, TwitchEnvironmentNames)
        else TwitchEnvironmentNames.from_mapping(env_names)
    )
    source = os.environ if environ is None else environ
    definitions = (
        ("client_id", names.client_id, True),
        ("access_token", names.access_token, True),
        ("broadcaster_id", names.broadcaster_id, True),
        ("user_id", names.user_id, False),
        ("viewer_hash_secret", names.viewer_hash_secret, True),
    )
    states: list[CredentialVariableState] = []
    issues: list[ValidationIssue] = []
    for role, env_name, role_required in definitions:
        is_required = bool(required and role_required)
        valid_name = bool(env_name and STREAMER_ENV_NAME_PATTERN.fullmatch(env_name))
        present = bool(valid_name and source.get(env_name))
        states.append(
            CredentialVariableState(
                role=role,
                env_name=env_name,
                required=is_required,
                valid_name=valid_name,
                present=present,
            )
        )
        if env_name and not valid_name:
            issues.append(
                ValidationIssue(
                    "invalid_streamer_environment_variable_name",
                    f"streamer_twitch_{role}_env",
                    f"{env_name!r} is not a valid environment-variable name.",
                )
            )
        elif is_required and not env_name:
            issues.append(
                ValidationIssue(
                    "streamer_environment_variable_name_required",
                    f"streamer_twitch_{role}_env",
                    f"Configure the environment-variable name for {role}.",
                )
            )
        elif is_required and not present:
            issues.append(
                ValidationIssue(
                    "streamer_twitch_environment_missing",
                    f"streamer_twitch_{role}_env",
                    f"Set environment variable {env_name} before starting Twitch mode.",
                )
            )
    return CredentialReadiness(tuple(states), tuple(issues))


@dataclass(frozen=True, slots=True)
class TrainLaunchValues:
    run_dir: Path
    live_status_path: Path
    resume_model_path: Optional[Path]
    total_timesteps: int
    checkpoint_freq: int
    n_steps: int
    batch_size: int
    adventure_start_level: int
    max_adventure_levels: int
    max_attempts_per_level: int
    adventure_soft_max_steps: int
    adventure_hard_max_steps: int
    board_timeout: float
    game_speed: float
    step_seconds: float


@dataclass(frozen=True, slots=True)
class EvaluationLaunchValues:
    model_path: Path
    run_dir: Path
    live_status_path: Path
    adventure_start_level: int
    max_adventure_levels: int
    max_attempts_per_level: int
    adventure_soft_max_steps: int
    adventure_hard_max_steps: int
    board_timeout: float
    game_speed: float
    step_seconds: float


@dataclass(frozen=True, slots=True)
class StreamerV1LaunchValues:
    platform: str
    baseline_checkpoint: Path
    run_dir: Path
    live_status_path: Path
    adventure_start_level: int
    max_adventure_levels: int
    max_attempts_per_level: int
    n_steps: int
    batch_size: int
    intervention_interval_seconds: float
    command_ttl_seconds: float
    command_queue_capacity: int
    message_max_chars: int
    policy_steps_per_cycle: int
    checkpoint_policy_steps: int
    evaluation_episodes: int
    max_cycles: int
    endurance_hours: float
    bc_enabled: bool
    bc_coefficient: float
    demonstration_capacity: int
    demonstration_persist_every: int
    bc_batch_size: int
    bc_update_frequency: int
    bc_min_demonstrations: int
    twitch_environment_names: TwitchEnvironmentNames
    mock_script: Optional[Path]


_MISSING = object()


def _mapping_value(values: Mapping[str, Any], key: str, default: Any = _MISSING) -> Any:
    value = values.get(key, _MISSING)
    if value is _MISSING or value is None or (isinstance(value, str) and not value.strip()):
        return default
    return value


def _parse_int(
    values: Mapping[str, Any],
    key: str,
    issues: list[ValidationIssue],
    *,
    default: Any = _MISSING,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> Optional[int]:
    raw = _mapping_value(values, key, default)
    if raw is _MISSING:
        issues.append(ValidationIssue("value_required", key, f"{key} is required."))
        return None
    try:
        if isinstance(raw, bool):
            raise ValueError
        parsed = int(str(raw).strip()) if isinstance(raw, str) else int(raw)
        if isinstance(raw, float) and not raw.is_integer():
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        issues.append(ValidationIssue("invalid_integer", key, f"{key} must be a whole number."))
        return None
    if minimum is not None and parsed < minimum:
        issues.append(ValidationIssue("value_below_minimum", key, f"{key} must be at least {minimum}."))
    if maximum is not None and parsed > maximum:
        issues.append(ValidationIssue("value_above_maximum", key, f"{key} cannot exceed {maximum}."))
    return parsed


def _parse_float(
    values: Mapping[str, Any],
    key: str,
    issues: list[ValidationIssue],
    *,
    default: Any = _MISSING,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    minimum_inclusive: bool = True,
) -> Optional[float]:
    raw = _mapping_value(values, key, default)
    if raw is _MISSING:
        issues.append(ValidationIssue("value_required", key, f"{key} is required."))
        return None
    try:
        if isinstance(raw, bool):
            raise ValueError
        parsed = float(str(raw).strip()) if isinstance(raw, str) else float(raw)
    except (TypeError, ValueError, OverflowError):
        issues.append(ValidationIssue("invalid_number", key, f"{key} must be numeric."))
        return None
    if not math.isfinite(parsed):
        issues.append(ValidationIssue("non_finite_number", key, f"{key} must be finite."))
        return parsed
    if minimum is not None and (
        parsed < minimum or (not minimum_inclusive and parsed == minimum)
    ):
        qualifier = "greater than" if not minimum_inclusive else "at least"
        issues.append(ValidationIssue("value_below_minimum", key, f"{key} must be {qualifier} {minimum}."))
    if maximum is not None and parsed > maximum:
        issues.append(ValidationIssue("value_above_maximum", key, f"{key} cannot exceed {maximum}."))
    return parsed


def _parse_bool(values: Mapping[str, Any], key: str, *, default: bool) -> bool:
    raw = _mapping_value(values, key, default)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _path_from_value(
    values: Mapping[str, Any],
    key: str,
    issues: list[ValidationIssue],
    *,
    base_dir: Path,
    required: bool,
) -> Optional[Path]:
    raw = _mapping_value(values, key, _MISSING)
    if raw is _MISSING:
        if required:
            issues.append(ValidationIssue("path_required", key, f"{key} is required."))
        return None
    text = str(raw).strip()
    if not text:
        if required:
            issues.append(ValidationIssue("path_required", key, f"{key} is required."))
        return None
    path = Path(text)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve(strict=False)


def _parse_common_adventure_values(
    values: Mapping[str, Any], issues: list[ValidationIssue]
) -> tuple[
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[float],
    Optional[float],
    Optional[float],
]:
    start_level = _parse_int(
        values,
        "adventure_start_level",
        issues,
        default=1,
        minimum=1,
        maximum=FULL_ADVENTURE_MAX_LEVELS,
    )
    max_levels = _parse_int(
        values,
        "max_adventure_levels",
        issues,
        default=FULL_ADVENTURE_MAX_LEVELS,
        minimum=1,
        maximum=FULL_ADVENTURE_MAX_LEVELS,
    )
    max_attempts = _parse_int(values, "max_attempts_per_level", issues, default=10, minimum=1)
    soft_steps = _parse_int(values, "adventure_soft_max_steps", issues, default=2000, minimum=1)
    hard_steps = _parse_int(values, "adventure_hard_max_steps", issues, default=3500, minimum=1)
    game_speed = _parse_float(values, "game_speed", issues, default=4.0, minimum=0.0, minimum_inclusive=False)
    step_seconds = _parse_float(values, "step_seconds", issues, default=0.05, minimum=0.0)
    board_timeout = _parse_float(
        values,
        "board_timeout",
        issues,
        default=60.0,
        minimum=0.0,
        minimum_inclusive=False,
    )
    if soft_steps is not None and hard_steps is not None and hard_steps < soft_steps:
        issues.append(
            ValidationIssue(
                "adventure_step_limits_invalid",
                "adventure_hard_max_steps",
                "adventure_hard_max_steps must be at least adventure_soft_max_steps.",
            )
        )
    return (
        start_level,
        max_levels,
        max_attempts,
        soft_steps,
        hard_steps,
        game_speed,
        step_seconds,
        board_timeout,
    )


def _status_path(
    values: Mapping[str, Any], run_dir: Optional[Path], issues: list[ValidationIssue], base_dir: Path
) -> Optional[Path]:
    status = _path_from_value(
        values, "live_status_path", issues, base_dir=base_dir, required=False
    )
    return status if status is not None else (run_dir / "live_status.json" if run_dir else None)


def _preflight_config_path(
    values: Mapping[str, Any], issues: list[ValidationIssue], base_dir: Path
) -> Optional[Path]:
    """Validate an explicitly selected backend JSON configuration, when present."""

    path = _path_from_value(
        values, "config_path", issues, base_dir=base_dir, required=False
    )
    if path is None:
        return None
    if not path.is_file():
        issues.append(
            ValidationIssue(
                "config_path_missing",
                "config_path",
                f"Configuration file not found: {path}",
            )
        )
        return path
    try:
        if path.stat().st_size > GUI_CONFIG_MAX_BYTES:
            issues.append(
                ValidationIssue(
                    "config_file_too_large",
                    "config_path",
                    f"Configuration file exceeds {GUI_CONFIG_MAX_BYTES // (1024 * 1024)} MiB.",
                )
            )
            return path
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        issues.append(
            ValidationIssue(
                "invalid_config_json",
                "config_path",
                f"Configuration JSON could not be read: {exc}",
            )
        )
        return path
    if not isinstance(payload, dict):
        issues.append(
            ValidationIssue(
                "invalid_config_root",
                "config_path",
                "Configuration root must be a JSON object.",
            )
        )
    return path


def _is_path_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(directory.resolve(strict=False))
    except ValueError:
        return False
    return True


def _directory_has_entries(path: Path) -> bool:
    try:
        next(path.iterdir())
    except StopIteration:
        return False
    return True


def _valid_json_object(path: Path) -> bool:
    try:
        return isinstance(json.loads(path.read_text(encoding="utf-8")), dict)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False


def _is_recognized_streamer_experiment(path: Path) -> bool:
    """Return whether a nonempty directory has canonical Streamer resume evidence."""

    return _valid_json_object(path / "streamer_state.json") or _valid_json_object(
        path / "checkpoints" / "baseline.json"
    )


def _validate_output_directory(
    run_dir: Optional[Path],
    issues: list[ValidationIssue],
    *,
    allow_streamer_resume: bool = False,
) -> None:
    if run_dir is None or not run_dir.exists() or not run_dir.is_dir():
        return
    try:
        nonempty = _directory_has_entries(run_dir)
    except OSError as exc:
        issues.append(
            ValidationIssue(
                "run_dir_unreadable", "run_dir", f"Run directory could not be inspected: {exc}"
            )
        )
        return
    if not nonempty:
        return
    if allow_streamer_resume and _is_recognized_streamer_experiment(run_dir):
        return
    message = (
        "Select an empty experiment directory, or a recognizable existing Streamer "
        "experiment with canonical state/checkpoint evidence."
        if allow_streamer_resume
        else "Select a new or empty output directory so existing run artifacts are not overwritten."
    )
    issues.append(ValidationIssue("run_dir_not_empty", "run_dir", message))


def _validate_config_output_separation(
    config_path: Optional[Path], run_dir: Optional[Path], issues: list[ValidationIssue]
) -> None:
    if config_path is not None and run_dir is not None and _is_path_within(
        config_path, run_dir
    ):
        issues.append(
            ValidationIssue(
                "config_inside_output_dir",
                "config_path",
                "Keep the input configuration outside the output directory.",
            )
        )


def _validate_live_status_destination(
    status_path: Optional[Path],
    run_dir: Optional[Path],
    issues: list[ValidationIssue],
    *,
    protected_paths: Sequence[Optional[Path]] = (),
) -> None:
    """Reject destinations that an atomic status replace could destructively overwrite."""

    if status_path is None:
        return
    resolved = status_path.resolve(strict=False)
    if status_path.exists() and status_path.is_dir():
        issues.append(
            ValidationIssue(
                "live_status_path_is_directory",
                "live_status_path",
                "Live-status path must name a JSON file, not a directory.",
            )
        )
    for protected in protected_paths:
        if protected is not None and resolved == protected.resolve(strict=False):
            issues.append(
                ValidationIssue(
                    "live_status_path_collision",
                    "live_status_path",
                    f"Live-status output would overwrite protected input/artifact: {protected}",
                )
            )
            return
    if status_path.suffix.lower() == ".zip":
        issues.append(
            ValidationIssue(
                "live_status_path_collision",
                "live_status_path",
                "Live-status output cannot target a model/checkpoint .zip file.",
            )
        )
        return
    if run_dir is not None and resolved == run_dir.resolve(strict=False):
        issues.append(
            ValidationIssue(
                "live_status_path_collision",
                "live_status_path",
                "Live-status output cannot be the run directory itself.",
            )
        )
        return
    if (
        run_dir is not None
        and status_path.parent.resolve(strict=False) == run_dir.resolve(strict=False)
        and status_path.name.lower() in _RESERVED_RUN_OUTPUT_NAMES
    ):
        issues.append(
            ValidationIssue(
                "live_status_path_collision",
                "live_status_path",
                f"{status_path.name} is reserved for another run artifact.",
            )
        )


def _has_errors(issues: Sequence[ValidationIssue]) -> bool:
    return any(issue.severity == "error" for issue in issues)


def _validate_full_adventure_form_contract(
    values: Mapping[str, Any], issues: list[ValidationIssue]
) -> None:
    """Reject editable form values that would contradict policy semantics."""

    raw_loadout = _mapping_value(
        values,
        "initial_loadout",
        ",".join(FULL_ADVENTURE_CONTRACT.initial_loadout),
    )
    if isinstance(raw_loadout, str):
        loadout = tuple(item.strip() for item in raw_loadout.split(",") if item.strip())
    elif isinstance(raw_loadout, (list, tuple)):
        loadout = tuple(str(item).strip() for item in raw_loadout)
    else:
        loadout = ()
    if loadout != FULL_ADVENTURE_CONTRACT.initial_loadout:
        issues.append(
            ValidationIssue(
                "initial_loadout_mismatch",
                "initial_loadout",
                "Initial loadout must preserve the Full-Adventure starter slot order: "
                + ",".join(FULL_ADVENTURE_CONTRACT.initial_loadout)
                + ".",
            )
        )

    max_slots = _parse_int(
        values,
        "max_seed_slots",
        issues,
        default=FULL_ADVENTURE_CONTRACT.max_seed_slots,
        minimum=FULL_ADVENTURE_CONTRACT.max_seed_slots,
        maximum=FULL_ADVENTURE_CONTRACT.max_seed_slots,
    )
    if max_slots is not None and max_slots != FULL_ADVENTURE_CONTRACT.max_seed_slots:
        issues.append(
            ValidationIssue(
                "max_seed_slots_mismatch",
                "max_seed_slots",
                f"Full-Adventure requires exactly {FULL_ADVENTURE_CONTRACT.max_seed_slots} identity slots.",
            )
        )


def validate_train_form(
    values: Mapping[str, Any],
    *,
    base_dir: Optional[os.PathLike[str] | str] = None,
) -> ValidationResult[TrainLaunchValues]:
    """Parse and preflight a fresh/resume Generalist training form."""

    root = Path(base_dir) if base_dir is not None else Path.cwd()
    issues: list[ValidationIssue] = []
    _validate_full_adventure_form_contract(values, issues)
    config_path = _preflight_config_path(values, issues, root)
    run_dir = _path_from_value(values, "run_dir", issues, base_dir=root, required=True)
    if run_dir is not None and run_dir.exists() and not run_dir.is_dir():
        issues.append(ValidationIssue("run_dir_not_directory", "run_dir", f"Run path is a file: {run_dir}"))
    _validate_output_directory(run_dir, issues)
    _validate_config_output_separation(config_path, run_dir, issues)
    status_path = _status_path(values, run_dir, issues, root)
    resume_path = _path_from_value(
        values, "resume_model_path", issues, base_dir=root, required=False
    )
    resume_metadata_path: Optional[Path] = None
    if resume_path is not None:
        inspection = inspect_model_compatibility(resume_path, base_dir=root)
        issues.extend(inspection.issues)
        resume_metadata_path = inspection.metadata_path
        if run_dir is not None:
            source_run_dir = (
                resume_path.parent.parent
                if resume_path.parent.name.lower() == "checkpoints"
                else resume_path.parent
            )
            if source_run_dir.resolve(strict=False) == run_dir.resolve(strict=False):
                issues.append(
                    ValidationIssue(
                        "resume_run_dir_must_be_new",
                        "run_dir",
                        "Choose a new destination run directory; resume never writes into the source run.",
                    )
                )
            try:
                resume_path.relative_to(run_dir)
            except ValueError:
                pass
            else:
                issues.append(
                    ValidationIssue(
                        "resume_source_inside_run_dir",
                        "run_dir",
                        "Use a new run directory so resume artifacts remain separate from the source checkpoint.",
                    )
                )
    _validate_live_status_destination(
        status_path,
        run_dir,
        issues,
        protected_paths=(resume_path, resume_metadata_path, config_path),
    )
    total = _parse_int(values, "total_timesteps", issues, default=25000, minimum=1)
    checkpoint = _parse_int(values, "checkpoint_freq", issues, default=5000, minimum=1)
    n_steps = _parse_int(values, "n_steps", issues, default=512, minimum=1)
    batch_size = _parse_int(values, "batch_size", issues, default=64, minimum=1)
    common = _parse_common_adventure_values(values, issues)
    parsed = (
        TrainLaunchValues(
            run_dir=run_dir,
            live_status_path=status_path,
            resume_model_path=resume_path,
            total_timesteps=total,
            checkpoint_freq=checkpoint,
            n_steps=n_steps,
            batch_size=batch_size,
            adventure_start_level=common[0],
            max_adventure_levels=common[1],
            max_attempts_per_level=common[2],
            adventure_soft_max_steps=common[3],
            adventure_hard_max_steps=common[4],
            board_timeout=common[7],
            game_speed=common[5],
            step_seconds=common[6],
        )
        if None not in (run_dir, status_path, total, checkpoint, n_steps, batch_size, *common)
        else None
    )
    return ValidationResult(None if _has_errors(issues) else parsed, tuple(issues))


def validate_evaluation_form(
    values: Mapping[str, Any],
    *,
    base_dir: Optional[os.PathLike[str] | str] = None,
) -> ValidationResult[EvaluationLaunchValues]:
    """Parse and preflight an inference-only Generalist evaluation form."""

    root = Path(base_dir) if base_dir is not None else Path.cwd()
    issues: list[ValidationIssue] = []
    _validate_full_adventure_form_contract(values, issues)
    config_path = _preflight_config_path(values, issues, root)
    model_path = _path_from_value(values, "model_path", issues, base_dir=root, required=True)
    run_dir = _path_from_value(values, "run_dir", issues, base_dir=root, required=True)
    if run_dir is not None and run_dir.exists() and not run_dir.is_dir():
        issues.append(ValidationIssue("run_dir_not_directory", "run_dir", f"Run path is a file: {run_dir}"))
    _validate_output_directory(run_dir, issues)
    _validate_config_output_separation(config_path, run_dir, issues)
    status_path = _status_path(values, run_dir, issues, root)
    model_metadata_path: Optional[Path] = None
    if model_path is not None:
        inspection = inspect_model_compatibility(model_path, base_dir=root)
        issues.extend(inspection.issues)
        model_metadata_path = inspection.metadata_path
        if run_dir is not None:
            try:
                model_path.relative_to(run_dir)
            except ValueError:
                pass
            else:
                issues.append(
                    ValidationIssue(
                        "evaluation_source_inside_run_dir",
                        "run_dir",
                        "Use a separate evaluation output directory so the source checkpoint remains immutable.",
                    )
                )
    _validate_live_status_destination(
        status_path,
        run_dir,
        issues,
        protected_paths=(model_path, model_metadata_path, config_path),
    )
    common = _parse_common_adventure_values(values, issues)
    parsed = (
        EvaluationLaunchValues(
            model_path=model_path,
            run_dir=run_dir,
            live_status_path=status_path,
            adventure_start_level=common[0],
            max_adventure_levels=common[1],
            max_attempts_per_level=common[2],
            adventure_soft_max_steps=common[3],
            adventure_hard_max_steps=common[4],
            board_timeout=common[7],
            game_speed=common[5],
            step_seconds=common[6],
        )
        if None not in (model_path, run_dir, status_path, *common)
        else None
    )
    return ValidationResult(None if _has_errors(issues) else parsed, tuple(issues))


def _validate_mock_script(
    path: Optional[Path],
    *,
    secret_env_name: str,
    environ: Mapping[str, str],
) -> Tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []
    if path is None:
        return (ValidationIssue("streamer_mock_script_missing", "streamer_mock_script", "Select a mock JSONL script."),)
    if not path.is_file():
        return (ValidationIssue("streamer_mock_script_missing", "streamer_mock_script", f"Mock script not found: {path}"),)
    try:
        if path.stat().st_size > MOCK_SCRIPT_MAX_BYTES:
            return (
                ValidationIssue(
                    "invalid_streamer_mock_script",
                    "streamer_mock_script",
                    f"Mock script exceeds {MOCK_SCRIPT_MAX_BYTES} bytes.",
                ),
            )
        with path.open("r", encoding="utf-8") as handle:
            for line_number in range(1, MOCK_SCRIPT_MAX_LINES + 2):
                raw_line = handle.readline(MOCK_SCRIPT_MAX_LINE_CHARS + 1)
                if raw_line == "":
                    break
                if line_number > MOCK_SCRIPT_MAX_LINES:
                    raise ValueError("script_record_limit_exceeded")
                if len(raw_line) > MOCK_SCRIPT_MAX_LINE_CHARS or (
                    len(raw_line) == MOCK_SCRIPT_MAX_LINE_CHARS and not raw_line.endswith("\n")
                ):
                    raise ValueError(f"line_{line_number}_too_long")
                stripped = raw_line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                payload = json.loads(stripped)
                if not isinstance(payload, dict):
                    raise ValueError(f"line_{line_number}_must_be_object")
                allowed = {
                    "command",
                    "viewer_hash",
                    "local_viewer_id",
                    "delivery_id",
                    "event_id",
                    "published_at",
                }
                unexpected = sorted(str(key) for key in payload if str(key) not in allowed)
                if unexpected:
                    raise ValueError(
                        f"line_{line_number}_unexpected_fields={','.join(unexpected)}"
                    )
                if not isinstance(payload.get("command"), str):
                    raise ValueError(f"line_{line_number}_command_required")
                if payload.get("local_viewer_id") and (
                    not secret_env_name or not environ.get(secret_env_name)
                ):
                    raise ValueError(
                        f"line_{line_number}_local_viewer_id_requires="
                        f"{secret_env_name or '<viewer_hash_secret_env>'}"
                    )
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        issues.append(
            ValidationIssue(
                "invalid_streamer_mock_script",
                "streamer_mock_script",
                f"Mock script is invalid: {exc}",
            )
        )
    return tuple(issues)


def validate_streamer_v1_form(
    values: Mapping[str, Any],
    *,
    base_dir: Optional[os.PathLike[str] | str] = None,
    environ: Optional[Mapping[str, str]] = None,
    validate_mock_script: bool = True,
) -> ValidationResult[StreamerV1LaunchValues]:
    """Parse Streamer V1 form values using the backend's resource gates."""

    root = Path(base_dir) if base_dir is not None else Path.cwd()
    env_source = os.environ if environ is None else environ
    issues: list[ValidationIssue] = []
    _validate_full_adventure_form_contract(values, issues)
    config_path = _preflight_config_path(values, issues, root)
    platform = str(_mapping_value(values, "streamer_platform", "twitch") or "").strip().lower()
    if platform not in {"twitch", "mock"}:
        issues.append(
            ValidationIssue("invalid_streamer_platform", "streamer_platform", "Platform must be twitch or mock.")
        )
    baseline = _path_from_value(
        values, "streamer_baseline_checkpoint", issues, base_dir=root, required=True
    )
    run_dir = _path_from_value(values, "run_dir", issues, base_dir=root, required=True)
    if run_dir is not None and run_dir.exists() and not run_dir.is_dir():
        issues.append(ValidationIssue("run_dir_not_directory", "run_dir", f"Run path is a file: {run_dir}"))
    _validate_output_directory(run_dir, issues, allow_streamer_resume=True)
    _validate_config_output_separation(config_path, run_dir, issues)
    status_path = _status_path(values, run_dir, issues, root)
    baseline_metadata_path: Optional[Path] = None
    if baseline is not None:
        inspection = inspect_model_compatibility(baseline, base_dir=root)
        issues.extend(inspection.issues)
        baseline_metadata_path = inspection.metadata_path
        if run_dir is not None:
            try:
                baseline.relative_to(run_dir)
            except ValueError:
                pass
            else:
                issues.append(
                    ValidationIssue(
                        "streamer_baseline_inside_experiment",
                        "streamer_baseline_checkpoint",
                        "BASELINE must be outside the Streamer experiment directory.",
                    )
                )
    _validate_live_status_destination(
        status_path,
        run_dir,
        issues,
        protected_paths=(baseline, baseline_metadata_path, config_path),
    )

    start_level = _parse_int(
        values,
        "adventure_start_level",
        issues,
        default=1,
        minimum=1,
        maximum=FULL_ADVENTURE_MAX_LEVELS,
    )
    max_levels = _parse_int(
        values,
        "max_adventure_levels",
        issues,
        default=FULL_ADVENTURE_MAX_LEVELS,
        minimum=1,
        maximum=FULL_ADVENTURE_MAX_LEVELS,
    )
    max_attempts = _parse_int(values, "max_attempts_per_level", issues, default=10, minimum=1)

    n_steps = _parse_int(values, "n_steps", issues, default=500, minimum=1, maximum=8192)
    batch_size = _parse_int(values, "batch_size", issues, default=50, minimum=1, maximum=4096)
    interval = _parse_float(
        values,
        "streamer_intervention_interval_seconds",
        issues,
        default=2.0,
        minimum=0.1,
        maximum=3600.0,
    )
    ttl = _parse_float(
        values,
        "streamer_command_ttl_seconds",
        issues,
        default=10.0,
        minimum=0.0,
        maximum=3600.0,
        minimum_inclusive=False,
    )
    queue_capacity = _parse_int(
        values, "streamer_command_queue_capacity", issues, default=256, minimum=1, maximum=4096
    )
    message_chars = _parse_int(
        values, "streamer_message_max_chars", issues, default=256, minimum=1, maximum=1024
    )
    cycle_steps = _parse_int(
        values,
        "streamer_policy_steps_per_cycle",
        issues,
        default=25000,
        minimum=1,
        maximum=10_000_000,
    )
    checkpoint_steps = _parse_int(
        values,
        "streamer_checkpoint_policy_steps",
        issues,
        default=5000,
        minimum=1,
        maximum=10_000_000,
    )
    evaluation_episodes = _parse_int(
        values,
        "streamer_evaluation_episodes",
        issues,
        default=50,
        minimum=1,
        maximum=10_000,
    )
    max_cycles = _parse_int(
        values, "streamer_max_cycles", issues, default=0, minimum=0, maximum=1_000_000
    )
    endurance = _parse_float(
        values, "streamer_endurance_hours", issues, default=0.0, minimum=0.0, maximum=168.0
    )
    bc_enabled = _parse_bool(values, "streamer_bc_enabled", default=True)
    bc_coefficient = _parse_float(
        values, "streamer_bc_coefficient", issues, default=0.01, minimum=0.0, maximum=1.0
    )
    demo_capacity = _parse_int(
        values,
        "streamer_demonstration_capacity",
        issues,
        default=4096,
        minimum=1,
        maximum=16_384,
    )
    demo_persist = _parse_int(
        values,
        "streamer_demonstration_persist_every",
        issues,
        default=512,
        minimum=1,
        maximum=16_384,
    )
    bc_batch = _parse_int(
        values, "streamer_bc_batch_size", issues, default=32, minimum=1, maximum=4096
    )
    bc_frequency = _parse_int(
        values,
        "streamer_bc_update_frequency",
        issues,
        default=1,
        minimum=1,
        maximum=1_000_000,
    )
    bc_minimum = _parse_int(
        values,
        "streamer_bc_min_demonstrations",
        issues,
        default=8,
        minimum=1,
        maximum=16_384,
    )

    if n_steps and batch_size and n_steps % batch_size != 0:
        issues.append(
            ValidationIssue(
                "streamer_minibatch_alignment",
                "batch_size",
                "n_steps must be divisible by batch_size.",
            )
        )
    if cycle_steps and n_steps and cycle_steps % n_steps != 0:
        issues.append(
            ValidationIssue(
                "streamer_cycle_rollout_alignment",
                "streamer_policy_steps_per_cycle",
                "streamer_policy_steps_per_cycle must be divisible by n_steps.",
            )
        )
    if checkpoint_steps and cycle_steps and checkpoint_steps > cycle_steps:
        issues.append(
            ValidationIssue(
                "invalid_streamer_checkpoint_interval",
                "streamer_checkpoint_policy_steps",
                "Checkpoint policy steps cannot exceed cycle policy steps.",
            )
        )
    for value, field, message in (
        (demo_persist, "streamer_demonstration_persist_every", "Persistence interval cannot exceed demonstration capacity."),
        (bc_batch, "streamer_bc_batch_size", "BC batch size cannot exceed demonstration capacity."),
        (bc_minimum, "streamer_bc_min_demonstrations", "BC minimum cannot exceed demonstration capacity."),
    ):
        if value is not None and demo_capacity is not None and value > demo_capacity:
            issues.append(ValidationIssue("invalid_streamer_demonstration_bounds", field, message))

    names = TwitchEnvironmentNames.from_mapping(values)
    credential_readiness = inspect_twitch_credentials(
        names, environ=env_source, required=platform == "twitch"
    )
    issues.extend(credential_readiness.issues)
    mock_script = _path_from_value(
        values, "streamer_mock_script", issues, base_dir=root, required=False
    )
    if platform == "mock" and validate_mock_script:
        issues.extend(
            _validate_mock_script(
                mock_script,
                secret_env_name=names.viewer_hash_secret,
                environ=env_source,
            )
        )

    candidate_values = (
        baseline,
        run_dir,
        status_path,
        start_level,
        max_levels,
        max_attempts,
        n_steps,
        batch_size,
        interval,
        ttl,
        queue_capacity,
        message_chars,
        cycle_steps,
        checkpoint_steps,
        evaluation_episodes,
        max_cycles,
        endurance,
        bc_coefficient,
        demo_capacity,
        demo_persist,
        bc_batch,
        bc_frequency,
        bc_minimum,
    )
    parsed = (
        StreamerV1LaunchValues(
            platform=platform,
            baseline_checkpoint=baseline,
            run_dir=run_dir,
            live_status_path=status_path,
            adventure_start_level=start_level,
            max_adventure_levels=max_levels,
            max_attempts_per_level=max_attempts,
            n_steps=n_steps,
            batch_size=batch_size,
            intervention_interval_seconds=interval,
            command_ttl_seconds=ttl,
            command_queue_capacity=queue_capacity,
            message_max_chars=message_chars,
            policy_steps_per_cycle=cycle_steps,
            checkpoint_policy_steps=checkpoint_steps,
            evaluation_episodes=evaluation_episodes,
            max_cycles=max_cycles,
            endurance_hours=endurance,
            bc_enabled=bc_enabled,
            bc_coefficient=bc_coefficient,
            demonstration_capacity=demo_capacity,
            demonstration_persist_every=demo_persist,
            bc_batch_size=bc_batch,
            bc_update_frequency=bc_frequency,
            bc_min_demonstrations=bc_minimum,
            twitch_environment_names=names,
            mock_script=mock_script,
        )
        if None not in candidate_values
        else None
    )
    return ValidationResult(None if _has_errors(issues) else parsed, tuple(issues))


# Every entry is defined by ``train_ppo.build_arg_parser``.  Keeping the set
# public lets tests/integrators reject accidental GUI-only CLI inventions.
STREAMER_V1_ARG_FLAGS = frozenset(
    {
        "--streamer-v1",
        "--streamer-platform",
        "--streamer-baseline-checkpoint",
        "--streamer-intervention-interval-seconds",
        "--streamer-command-ttl-seconds",
        "--streamer-command-queue-capacity",
        "--streamer-message-max-chars",
        "--streamer-policy-steps-per-cycle",
        "--streamer-checkpoint-policy-steps",
        "--streamer-evaluation-episodes",
        "--streamer-max-cycles",
        "--streamer-endurance-hours",
        "--streamer-bc-enabled",
        "--no-streamer-bc",
        "--streamer-bc-coefficient",
        "--streamer-demonstration-capacity",
        "--streamer-bc-batch-size",
        "--streamer-bc-update-frequency",
        "--streamer-bc-min-demonstrations",
        "--streamer-twitch-client-id-env",
        "--streamer-twitch-access-token-env",
        "--streamer-twitch-broadcaster-id-env",
        "--streamer-twitch-user-id-env",
        "--streamer-viewer-hash-secret-env",
        "--streamer-mock-script",
        "--n-steps",
        "--batch-size",
        "--adventure-start-level",
        "--max-adventure-levels",
        "--max-attempts-per-level",
        "--run-dir",
        "--live-status-path",
    }
)


def _cli_number(value: int | float) -> str:
    if isinstance(value, float):
        return format(value, ".15g")
    return str(value)


def build_streamer_v1_argv_additions(values: StreamerV1LaunchValues) -> Tuple[str, ...]:
    """Build parser-backed CLI additions for ``train_ppo.py``.

    ``streamer_demonstration_persist_every`` is validated because it is a
    backend resource constraint, but is intentionally absent here: the
    current parser exposes it only through JSON configuration.
    """

    names = values.twitch_environment_names
    argv: list[str] = [
        "--streamer-v1",
        "--streamer-platform",
        values.platform,
        "--streamer-baseline-checkpoint",
        str(values.baseline_checkpoint),
        "--run-dir",
        str(values.run_dir),
        "--live-status-path",
        str(values.live_status_path),
        "--adventure-start-level",
        str(values.adventure_start_level),
        "--max-adventure-levels",
        str(values.max_adventure_levels),
        "--max-attempts-per-level",
        str(values.max_attempts_per_level),
        "--n-steps",
        str(values.n_steps),
        "--batch-size",
        str(values.batch_size),
        "--streamer-intervention-interval-seconds",
        _cli_number(values.intervention_interval_seconds),
        "--streamer-command-ttl-seconds",
        _cli_number(values.command_ttl_seconds),
        "--streamer-command-queue-capacity",
        str(values.command_queue_capacity),
        "--streamer-message-max-chars",
        str(values.message_max_chars),
        "--streamer-policy-steps-per-cycle",
        str(values.policy_steps_per_cycle),
        "--streamer-checkpoint-policy-steps",
        str(values.checkpoint_policy_steps),
        "--streamer-evaluation-episodes",
        str(values.evaluation_episodes),
        "--streamer-max-cycles",
        str(values.max_cycles),
        "--streamer-endurance-hours",
        _cli_number(values.endurance_hours),
        "--streamer-bc-enabled" if values.bc_enabled else "--no-streamer-bc",
        "--streamer-bc-coefficient",
        _cli_number(values.bc_coefficient),
        "--streamer-demonstration-capacity",
        str(values.demonstration_capacity),
        "--streamer-bc-batch-size",
        str(values.bc_batch_size),
        "--streamer-bc-update-frequency",
        str(values.bc_update_frequency),
        "--streamer-bc-min-demonstrations",
        str(values.bc_min_demonstrations),
        "--streamer-twitch-client-id-env",
        names.client_id,
        "--streamer-twitch-access-token-env",
        names.access_token,
        "--streamer-twitch-broadcaster-id-env",
        names.broadcaster_id,
        "--streamer-twitch-user-id-env",
        names.user_id,
        "--streamer-viewer-hash-secret-env",
        names.viewer_hash_secret,
    ]
    if values.platform == "mock" and values.mock_script is not None:
        argv.extend(("--streamer-mock-script", str(values.mock_script)))
    return tuple(argv)
