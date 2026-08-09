"""Adventure Generalist MaskablePPO training and evaluation entrypoint."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple

import numpy as np

from pvzrl_adventure import (
    DEFAULT_ADVENTURE_HARD_MAX_STEPS,
    DEFAULT_ADVENTURE_SOFT_MAX_STEPS,
    run_adventure_eval,
)
from pvzrl_adventure_generalist import (
    ADVENTURE_GENERALIST_INITIAL_LOADOUT,
    DEFAULT_CORE_SEED_NAMES,
    DEFAULT_NEW_UNLOCK_GUARANTEE_EPISODES,
    ADVENTURE_GENERALIST_MODEL_FAMILY,
    ADVENTURE_GENERALIST_RUN_MODE_EVAL,
    ADVENTURE_GENERALIST_RUN_MODE_TRAIN,
    AdventureGeneralistTrainingEnv,
    SEED_CAPACITY_MAX,
    SEED_ORDER_SOURCE_DEFAULT,
    SEED_ORDER_SOURCE_EXPLICIT,
    SEED_ORDER_SOURCE_RANDOMIZED,
    parse_initial_loadout,
)
from pvzrl_action_space import (
    ACTION_SPACE_ADVENTURE_14_IDENTITY,
    ADVENTURE_IDENTITY_ACTION_DECODER_VERSION,
    ADVENTURE_IDENTITY_ACTION_COUNT,
    ADVENTURE_IDENTITY_OBSERVATION_VERSION,
    action_count_for_config as action_space_count_for_config,
    build_action_space_spec,
    normalize_action_space_mode,
)
from pvzrl_env import parse_seed_list, resolve_seed_list
from pvzrl_rewards import (
    FUSION_REWARD_COMPONENT_NAMES,
    REWARD_COMPONENT_FIELDS,
    REWARD_EPISODE_TOTAL_FIELDS,
    RewardConfig,
)
from pvzrl_fusion import FUSION_POLICY_NONE, fusion_live_fields, normalize_fusion_policy
from pvzrl_human_coach import human_coach_live_status_defaults
from pvzrl_config import (
    CONFIG_UNSET,
    ConfigSource,
    ConfigResolver,
    ResolvedRunConfig,
    warn_ignored_legacy_fields,
)
from pvzrl_model_metadata import (
    CompatibilityCheck,
    apply_model_metadata_defaults,
    env_metadata_from_config,
    format_compatibility_failure,
    model_compatibility_live_status,
    validate_model_metadata,
    write_model_metadata,
)
from pvzrl_sb3 import PvZMaskedPPOEnv, PvZSB3Config
from pvzrl_telemetry import LiveStatusWriter, live_status_significant_state
from pvzrl_streamer import EVALUATE, STREAM_TRAIN, run_streamer_cycles


DEFAULT_CONFIG_PATH = Path("configs/ppo_adventure_generalist_14slot_identity_v1.json")
STREAMER_ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
LANE_DIAGNOSTIC_DICT_FIELDS = [
    "plants_by_row",
    "peashooters_by_row",
    "sunflowers_by_row",
    "threat_steps_by_row",
    "undefended_threat_steps_by_row",
    "undefended_threat_age_max_by_row",
    "mower_losses_by_row",
    "illegal_reason_counts",
    "legal_peashooter_actions_by_row",
    "peashooter_available_but_waited_by_row",
    "peashooter_available_but_planted_elsewhere_by_row",
    "sunflower_while_undefended_threat_by_row",
    "plant_actions_by_row",
    "peashooter_actions_by_row",
    "sunflower_actions_by_row",
    "plant_placements_by_row",
    "peashooter_placements_by_row",
    "sunflower_placements_by_row",
    "row_defense_opportunities_by_row",
    "row_defense_responses_by_row",
    "threatened_rows_with_zero_defender_steps_by_row",
    "first_defense_step_by_row",
    "first_peashooter_by_row_step",
    "legal_actions_by_seed_slot",
    "bridge_legal_actions_by_seed_slot",
    "python_mask_block_reason_counts",
    "wallnut_placements_by_row",
    "wallnut_placements_by_col",
    "mower_risk_steps_by_row",
    "mower_saves_estimated_by_row",
    "buckethead_count_by_row",
    "conehead_count_by_row",
    "tough_zombie_count_by_row",
    "fusion_rejected_reasons",
    "fusion_by_result_type",
    "fusion_by_source_type",
    "fusion_by_row",
]
LANE_DIAGNOSTIC_FLOAT_DICT_FIELDS = [
    "undefended_threat_ratio_by_row",
    "undefended_threat_age_avg_by_row",
    "row_defense_response_rate_by_row",
    "peashooters_per_threat_step_by_row",
]
LANE_DIAGNOSTIC_NUMERIC_FIELDS = [
    "wait_under_threat_count",
    "close_zombie_undefended_count",
    "wait_actions",
    "plant_actions",
    "wait_action_percent",
    "plant_action_percent",
    "row_defense_response_rate",
    "plants_in_threatened_row_ratio",
    "plants_in_unthreatened_row_ratio",
    "overdefended_while_undefended_count",
    "least_defended_threatened_row_plant_count",
    "rows_with_peashooter_count",
    "all_rows_peashooter_covered_step",
    "sunflower_count_when_first_full_coverage",
    "sunflower_overbuild_before_defense_count",
    "sunflower_overbuild_count",
    "peashooter_coverage_rate_by_step",
    "pre_step_mask_blocked_count",
    "cooldown_illegal_exposed_by_mask_count",
    "mask_bridge_disagreement_count",
    "env_corruption_count",
    "mower_respawn_detected_count",
    "cooldown_reset_detected_count",
    "board_refresh_detected_count",
    "false_reward_unlock_during_gameplay_count",
    "false_cleanup_reward_ui_during_gameplay_count",
    "post_win_veto_live_board_count",
    "blocked_cleanup_during_gameplay_count",
    "suspicious_cleanup_reward_ui_count",
    "reset_reward_ui_cleanup_count",
    "reset_reward_ui_cleanup_blocked_count",
    "reset_after_false_reward_signal_count",
    "timeout_reset_requested_count",
    "wait_while_actionable_threat_count",
    "wait_while_peashooter_affordable_ready_count",
    "wait_while_wallnut_affordable_ready_count",
    "wait_while_cherrybomb_affordable_ready_count",
    "active_threat_rows_without_peashooter_count",
    "sunflower_greed_while_defense_missing_count",
    "wallnut_blocks_active_threat_count",
    "wallnut_low_value_placement_count",
    "wallnut_threatened_lane_placements",
    "wallnut_between_zombie_and_house_count",
    "wallnut_frontline_count",
    "wallnut_emergency_blocks",
    "wallnut_useless_placements",
    "cherrybomb_used_count",
    "cherrybomb_kills_total",
    "cherrybomb_zero_kill_count",
    "cherrybomb_zero_kill_uses",
    "cherrybomb_cluster_uses",
    "cherrybomb_emergency_uses",
    "cherrybomb_heavy_zombie_kills",
    "cherrybomb_buckethead_kills",
    "cherrybomb_conehead_kills",
    "cherrybomb_used_under_threat_count",
    "cherrybomb_used_low_value_count",
    "close_zombie_with_no_defense_count",
    "undefended_threat_steps",
    "high_danger_unanswered_steps",
    "mower_exposure_steps",
    "overdefense_count",
    "mower_losses",
    "legal_action_count_mean",
    "max_row_danger",
    "avg_row_danger",
    "tactical_mask_enabled",
    "wallnut_actions_masked",
    "cherrybomb_actions_masked",
    "wallnut_actions_available",
    "cherrybomb_actions_available",
    "mask_all_but_wait_count",
    "tough_zombie_response_count",
    "fusion_candidate_count_total",
    "fusion_attempted_count",
    "fusion_success_count",
    "fusion_failed_count",
    "fusion_rejected_count",
    "fusion_under_threat_count",
    "fusion_near_buckethead_count",
    "fusion_near_conehead_count",
    "fusion_estimated_mower_save_count",
    "fusion_kills_after_use_total",
    "fusion_bridge_error_count",
    "fusion_unsafe_state_block_count",
]
PROGRESS_CSV_DIAGNOSTIC_INT_FIELDS = [
    "recursive_fusion_count",
    "highest_fusion_tier",
    "action_freeze_count",
]
PROGRESS_CSV_ACTION_DURATION_FIELDS = [
    "mean_action_duration_seconds",
    "p95_action_duration_seconds",
    "max_action_duration_seconds",
]
PROGRESS_CSV_DIAGNOSTIC_FIELDS = PROGRESS_CSV_DIAGNOSTIC_INT_FIELDS + PROGRESS_CSV_ACTION_DURATION_FIELDS
FUSION_REWARD_FLOAT_FIELDS = [
    f"{name}_total" for name in FUSION_REWARD_COMPONENT_NAMES
] + [
    "fusion_last_reward_delta",
    "fusion_last_usefulness_bonus",
]
EPISODE_STRING_FIELDS = [
    "fusion_policy",
    "fusion_last_reward_reason",
    "fusion_last_source",
]
EPISODE_METRIC_FIELDS = [
    "run_mode",
    "episode",
    "result",
    "reward_total",
    "episode_reward",
    "episode_length",
    "terminal_reason",
    "done_reason",
    "win",
    "loss",
    "timeout",
    "final_wave",
    "max_wave",
    "zombies_killed",
    "plants_placed",
    "sunflowers_planted",
    "peashooters_planted",
    "wallnuts_planted",
    "cherrybombs_planted",
    "sun_spent",
    "sun_remaining",
    "mowers_lost",
    "reset_success",
    "reset_seconds",
    "bridge_errors",
    "illegal_actions",
    "avg_legal_actions",
] + EPISODE_STRING_FIELDS + LANE_DIAGNOSTIC_DICT_FIELDS + LANE_DIAGNOSTIC_FLOAT_DICT_FIELDS + LANE_DIAGNOSTIC_NUMERIC_FIELDS + [
    "wallnut_damage_absorbed_total",
    "cherrybomb_avg_kills_per_use",
    "fusion_candidate_count_avg",
    "fusion_avg_kills_after_use",
] + PROGRESS_CSV_DIAGNOSTIC_FIELDS + FUSION_REWARD_FLOAT_FIELDS + ["fusion_reward_capped"] + list(REWARD_EPISODE_TOTAL_FIELDS)


def require_maskable_ppo() -> Any:
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:
        raise SystemExit(
            "MaskablePPO requires sb3-contrib, which is not installed in this environment.\n"
            "Install PPO dependencies with:\n\n"
            "  python -m pip install -r requirements-ppo.txt\n"
        ) from exc
    return MaskablePPO


def require_sb3_callbacks() -> Any:
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    return BaseCallback, CallbackList, CheckpointCallback, DummyVecEnv, Monitor


def load_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def pick_reward(args: argparse.Namespace, raw_config: Dict[str, Any], key: str, fallback: float) -> float:
    value = getattr(args, key, None)
    if value is not None:
        return float(value)
    reward_config = raw_config.get("reward", {})
    if isinstance(reward_config, dict) and key in reward_config:
        return float(reward_config[key])
    return float(raw_config.get(key, fallback))


def build_reward_config(
    args: argparse.Namespace,
    raw_config: Dict[str, Any],
) -> Dict[str, float]:
    defaults = asdict(RewardConfig())
    reward = {
        key: pick_reward(args, raw_config, key, float(default_value))
        for key, default_value in defaults.items()
    }
    return reward


def reward_config_from_mapping(values: Any) -> RewardConfig:
    if isinstance(values, RewardConfig):
        return values
    if not isinstance(values, dict):
        values = {}
    defaults = asdict(RewardConfig())
    payload = {
        key: float(values.get(key, default_value))
        for key, default_value in defaults.items()
    }
    return RewardConfig(**payload)


class TeeStream:
    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.streams[0], name)


class RotatingTextStream:
    """Small bounded file-like stream for long-running console capture."""

    def __init__(self, path: Path, *, max_bytes: int, backup_count: int) -> None:
        self.path = path
        self.max_bytes = max(1024, int(max_bytes))
        self.backup_count = max(1, int(backup_count))
        self._lock = threading.Lock()
        if path.is_file() and path.stat().st_size >= self.max_bytes:
            self._rotate_files()
        self._handle: TextIO = path.open("a", encoding="utf-8")
        self._size = int(path.stat().st_size) if path.is_file() else 0

    def _rotate_files(self) -> None:
        oldest = self.path.with_name(f"{self.path.name}.{self.backup_count}")
        oldest.unlink(missing_ok=True)
        for index in range(self.backup_count - 1, 0, -1):
            source = self.path.with_name(f"{self.path.name}.{index}")
            if source.is_file():
                os.replace(source, self.path.with_name(f"{self.path.name}.{index + 1}"))
        if self.path.is_file():
            os.replace(self.path, self.path.with_name(f"{self.path.name}.1"))

    def write(self, data: str) -> int:
        encoded_size = len(str(data).encode("utf-8"))
        with self._lock:
            if self._size > 0 and self._size + encoded_size > self.max_bytes:
                self._handle.close()
                self._rotate_files()
                self._handle = self.path.open("a", encoding="utf-8")
                self._size = 0
            written = self._handle.write(data)
            self._size += encoded_size
            return written

    def flush(self) -> None:
        with self._lock:
            self._handle.flush()

    def close(self) -> None:
        with self._lock:
            self._handle.close()


class TeeOutput:
    def __init__(
        self,
        path: Path,
        *,
        rotate_max_bytes: Optional[int] = None,
        rotate_backup_count: int = 3,
    ):
        self.path = path
        self.rotate_max_bytes = rotate_max_bytes
        self.rotate_backup_count = int(rotate_backup_count)
        self.handle: Optional[TextIO] = None
        self.old_stdout: Optional[TextIO] = None
        self.old_stderr: Optional[TextIO] = None

    def __enter__(self) -> "TeeOutput":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = (
            RotatingTextStream(
                self.path,
                max_bytes=int(self.rotate_max_bytes),
                backup_count=self.rotate_backup_count,
            )
            if self.rotate_max_bytes is not None
            else self.path.open("a", encoding="utf-8")
        )
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        sys.stdout = TeeStream(sys.stdout, self.handle)  # type: ignore[assignment]
        sys.stderr = TeeStream(sys.stderr, self.handle)  # type: ignore[assignment]
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.old_stdout is not None:
            sys.stdout = self.old_stdout
        if self.old_stderr is not None:
            sys.stderr = self.old_stderr
        if self.handle is not None:
            self.handle.close()


def command_used() -> str:
    return "python " + subprocess.list2cmdline(sys.argv)


def action_count_for_config(config: Dict[str, Any]) -> int:
    return action_space_count_for_config(apply_model_metadata_defaults(config))


def normalized_seed_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(seed).strip() for seed in value if str(seed).strip()]
    if isinstance(value, str):
        return parse_seed_list(value)
    return []


def seed_slot_signature(config: Dict[str, Any]) -> Dict[str, Any]:
    seed_list = normalized_seed_list(config.get("seed_list", []))
    plant_types = [int(plant_type) for plant_type in config.get("plant_types", [])]
    action_count = action_count_for_config(config)
    return {
        "seed_list": seed_list,
        "plant_types": plant_types,
        "action_count": action_count,
        "action_space_mode": str(config.get("action_space_mode", ACTION_SPACE_ADVENTURE_14_IDENTITY)),
        "action_decoder_version": str(config.get("action_decoder_version", "")),
        "observation_version": str(config.get("observation_version", "")),
    }


def _model_action_count(model: Any) -> int:
    value = getattr(getattr(model, "action_space", None), "n", None)
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _model_observation_shape(model: Any) -> Optional[List[int]]:
    shape = getattr(getattr(model, "observation_space", None), "shape", None)
    if not isinstance(shape, (list, tuple)):
        return None
    try:
        return [int(value) for value in shape]
    except (TypeError, ValueError):
        return None


def compatibility_summary_from_report(result: CompatibilityCheck) -> Dict[str, Any]:
    expected = result.expected
    actual = result.actual
    env_metadata = result.env_metadata
    live_status = model_compatibility_live_status(result)
    return {
        "compatible": bool(result.ok),
        "blocked_reason": result.blocked_reason,
        "metadata_path": result.metadata_path,
        "metadata_inferred": result.metadata_inferred,
        "model_family": actual.get("model_family", expected.get("model_family", "")),
        "model_seed_list": normalized_seed_list(actual.get("seed_list", [])),
        "model_plant_types": [int(value) for value in actual.get("plant_types", [])],
        "expected_seed_list": normalized_seed_list(expected.get("seed_list", [])),
        "expected_plant_types": [int(value) for value in expected.get("plant_types", [])],
        "env_seed_list": normalized_seed_list(env_metadata.get("resolved_seed_list", [])),
        "env_plant_types": [int(value) for value in env_metadata.get("resolved_plant_types", [])],
        "action_count": int(env_metadata.get("env_action_count", expected.get("action_count", 0)) or 0),
        "model_action_count": actual.get("action_count"),
        "env_action_count": env_metadata.get("env_action_count"),
        "action_space_mode": env_metadata.get("action_space_mode", expected.get("action_space_mode")),
        "action_decoder_version": env_metadata.get("action_decoder_version", expected.get("action_decoder_version")),
        "observation_version": env_metadata.get("observation_version", expected.get("observation_version")),
        "model_compatibility": live_status,
        "warnings": list(result.warnings),
    }


def env_metadata_for_config(config: Dict[str, Any], env: Optional[Any] = None) -> Dict[str, Any]:
    if env is not None and hasattr(env, "get_env_metadata"):
        metadata = env.get_env_metadata()
    elif env is not None and hasattr(getattr(env, "env", None), "get_env_metadata"):
        metadata = env.env.get_env_metadata()
    else:
        metadata = env_metadata_from_config(config)
    metadata["model_family"] = str(config.get("model_family") or metadata.get("model_family") or "")
    return metadata


def loaded_model_compatibility_report(
    model: Any,
    model_path: Path,
    config: Dict[str, Any],
    *,
    env_metadata: Optional[Dict[str, Any]] = None,
) -> CompatibilityCheck:
    return validate_model_metadata(
        model_path,
        config,
        model_action_count=_model_action_count(model),
        model_observation_shape=_model_observation_shape(model),
        env_metadata=env_metadata or env_metadata_for_config(config),
    )


def raise_if_incompatible(result: CompatibilityCheck) -> None:
    if not result.ok:
        raise SystemExit(format_compatibility_failure(result))


def print_compatibility_report(prefix: str, result: CompatibilityCheck) -> None:
    print(f"{prefix} model_compatibility=" + json.dumps(result.to_dict(), separators=(",", ":"), sort_keys=True))


def _metadata_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _metadata_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def validate_adventure_generalist_model_compatibility(
    model_path: Path,
    config: Dict[str, Any],
    context: str,
    *,
    model: Optional[Any] = None,
    model_action_count: Optional[int] = None,
    model_observation_shape: Optional[Any] = None,
    env_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    env_metadata = env_metadata or env_metadata_for_config(config)
    loaded_action_count = _model_action_count(model) if model is not None else model_action_count
    loaded_observation_shape = (
        _model_observation_shape(model) if model is not None else model_observation_shape
    )
    result = validate_model_metadata(
        model_path,
        config,
        model_action_count=loaded_action_count,
        model_observation_shape=loaded_observation_shape,
        env_metadata=env_metadata,
    )
    print_compatibility_report(f"[compat:{context}]", result)

    model_metadata = result.model_metadata or result.actual or {}
    model_family = str(model_metadata.get("model_family") or "")
    raw_mode = model_metadata.get("action_space_mode", "")
    try:
        model_mode = normalize_action_space_mode(raw_mode)
    except ValueError:
        model_mode = str(raw_mode or "")
    metadata_action_count = _metadata_int(model_metadata.get("action_count"), default=-1)
    effective_action_count = _metadata_int(loaded_action_count, default=metadata_action_count)
    model_max_seed_slots = _metadata_int(model_metadata.get("max_seed_slots"), default=-1)
    model_observation_version = str(model_metadata.get("observation_version") or "")
    model_decoder_version = str(model_metadata.get("action_decoder_version") or "")
    model_identity_seed_slots = _metadata_bool(model_metadata.get("identity_seed_slots", False))
    model_wait_action = _metadata_int(model_metadata.get("decoder_wait_action"), default=-1)
    model_placement_range = model_metadata.get("placement_action_range")
    model_seed_list = normalized_seed_list(model_metadata.get("seed_list", []))
    runtime_seed_list = normalized_seed_list(env_metadata.get("resolved_seed_list", config.get("seed_list", [])))
    run_mode = str(model_metadata.get("run_mode") or "")
    expected_schema = "14_slots_x_50_cells_plus_wait"
    actual_schema = (
        f"mode={model_mode};wait={model_wait_action};range={model_placement_range};decoder={model_decoder_version}"
    )

    def _raise_incompatible_resume(mismatch_fields: List[str]) -> None:
        raise SystemExit(
            "blocked_reason=incompatible_resume_model\n"
            f"model_path={model_path}\n"
            f"expected_model_family={ADVENTURE_GENERALIST_MODEL_FAMILY}\n"
            f"actual_model_family={model_family or 'missing'}\n"
            f"expected_action_count={ADVENTURE_IDENTITY_ACTION_COUNT}\n"
            f"actual_action_count={effective_action_count}\n"
            f"expected_action_decoder_version={ADVENTURE_IDENTITY_ACTION_DECODER_VERSION}\n"
            f"actual_action_decoder_version={model_decoder_version or 'missing'}\n"
            f"expected_action_schema={expected_schema}\n"
            f"actual_action_schema={actual_schema}\n"
            f"expected_observation_schema={ADVENTURE_IDENTITY_OBSERVATION_VERSION}\n"
            f"actual_observation_schema={model_observation_version or 'missing'}\n"
            f"expected_max_seed_slots={SEED_CAPACITY_MAX}\n"
            f"actual_max_seed_slots={model_max_seed_slots}\n"
            f"expected_identity_seed_slots=True\n"
            f"actual_identity_seed_slots={model_identity_seed_slots}\n"
            f"expected_action_space_mode={ACTION_SPACE_ADVENTURE_14_IDENTITY}\n"
            f"actual_action_space_mode={model_mode}\n"
            f"expected_run_mode={ADVENTURE_GENERALIST_RUN_MODE_TRAIN}|{ADVENTURE_GENERALIST_RUN_MODE_EVAL}\n"
            f"actual_run_mode={run_mode or 'missing'}\n"
            f"expected_seed_list={','.join(runtime_seed_list) or 'missing'}\n"
            f"actual_seed_list={','.join(model_seed_list) or 'missing'}\n"
            f"mismatch_fields={','.join(mismatch_fields)}"
        )

    if not result.ok:
        blocked_to_field = {
            "action_count_mismatch": "action_count",
            "action_decoder_mismatch": "action_decoder_version",
            "observation_version_mismatch": "observation_version",
            "max_seed_slots_mismatch": "max_seed_slots",
            "action_space_mode_mismatch": "action_space_mode",
            "seed_list_mismatch": "seed_list",
            "model_family_mismatch": "model_family",
        }
        mapped = blocked_to_field.get(str(result.blocked_reason or ""))
        if mapped:
            _raise_incompatible_resume([mapped])
        raise_if_incompatible(result)

    strict_errors: List[str] = []
    if model_family != ADVENTURE_GENERALIST_MODEL_FAMILY:
        strict_errors.append("model_family")
    if model_seed_list != runtime_seed_list:
        strict_errors.append("seed_list")
    if metadata_action_count != ADVENTURE_IDENTITY_ACTION_COUNT:
        strict_errors.append("action_count")
    if model_mode != ACTION_SPACE_ADVENTURE_14_IDENTITY:
        strict_errors.append("action_space_mode")
    if model_decoder_version != ADVENTURE_IDENTITY_ACTION_DECODER_VERSION:
        strict_errors.append("action_decoder_version")
    if model_observation_version != ADVENTURE_IDENTITY_OBSERVATION_VERSION:
        strict_errors.append("observation_version")
    if model_max_seed_slots != SEED_CAPACITY_MAX:
        strict_errors.append("max_seed_slots")
    if not model_identity_seed_slots:
        strict_errors.append("identity_seed_slots")
    if run_mode and run_mode not in {ADVENTURE_GENERALIST_RUN_MODE_TRAIN, ADVENTURE_GENERALIST_RUN_MODE_EVAL}:
        strict_errors.append("run_mode")

    if strict_errors:
        _raise_incompatible_resume(strict_errors)

    summary = compatibility_summary_from_report(result)
    print(
        "[adventure-generalist] model_compatibility_check=ok "
        f"context={context} "
        f"model_path={model_path} "
        f"metadata_path={result.metadata_path or ''} "
        f"action_count={summary.get('model_action_count') or summary.get('action_count')} "
        f"observation_version={summary.get('observation_version')} "
        f"action_decoder_version={summary.get('action_decoder_version')} "
        f"max_seed_slots={model_max_seed_slots}"
    )
    return summary


def resolve_effective_run_mode(args: argparse.Namespace, raw_config: Dict[str, Any]) -> str:
    """Resolve one of the two maintained Adventure Generalist run modes."""

    shortcuts: List[str] = []
    if bool(getattr(args, "adventure_generalist_train", False)):
        shortcuts.append(ADVENTURE_GENERALIST_RUN_MODE_TRAIN)
    if bool(getattr(args, "adventure_generalist_eval", False)):
        shortcuts.append(ADVENTURE_GENERALIST_RUN_MODE_EVAL)
    if len(shortcuts) > 1:
        raise SystemExit(
            "blocked_reason=invalid_cli: --adventure-generalist-train and "
            "--adventure-generalist-eval are mutually exclusive."
        )

    explicit_run_mode = str(getattr(args, "run_mode", "") or "").strip().lower()
    shortcut_mode = shortcuts[0] if shortcuts else ""
    if explicit_run_mode and shortcut_mode and explicit_run_mode != shortcut_mode:
        raise SystemExit(
            "blocked_reason=invalid_cli: --run-mode conflicts with explicit mode shortcut: "
            f"{explicit_run_mode}!={shortcut_mode}"
        )

    configured_run_mode = str(raw_config.get("run_mode", "") or "").strip().lower()
    selected = explicit_run_mode or shortcut_mode or configured_run_mode or ADVENTURE_GENERALIST_RUN_MODE_TRAIN
    maintained_modes = {ADVENTURE_GENERALIST_RUN_MODE_TRAIN, ADVENTURE_GENERALIST_RUN_MODE_EVAL}
    if selected not in maintained_modes:
        raise SystemExit(
            "blocked_reason=unsupported_run_mode: "
            f"{selected}; Adventure Generalist is the sole maintained training/evaluation path."
        )
    return selected


def _build_config_mapping(
    args: argparse.Namespace,
    raw_config: Dict[str, Any],
    *,
    resolution_sources: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    warn_ignored_legacy_fields(raw_config)
    resolver = ConfigResolver(args, raw_config)
    value = resolver.value
    enabled = resolver.enabled
    quick_wait = enabled("quick_wait")
    board_timeout = value(
        "board_timeout",
        180.0,
        mode_default=60.0 if quick_wait else CONFIG_UNSET,
    )
    gameplay_ready_timeout = value(
        "gameplay_ready_timeout",
        60.0,
        mode_default=30.0 if quick_wait else CONFIG_UNSET,
    )
    poll_seconds = value("poll_seconds", 0.2)
    run_mode = resolve_effective_run_mode(args, raw_config)
    explicit_mode_cli = bool(
        getattr(args, "run_mode", None)
        or getattr(args, "adventure_generalist_train", False)
        or getattr(args, "adventure_generalist_eval", False)
    )
    resolver.sources["run_mode"] = (
        ConfigSource.CLI
        if explicit_mode_cli
        else ConfigSource.JSON
        if "run_mode" in raw_config
        else ConfigSource.GLOBAL_DEFAULT
    )
    adventure_generalist_train_requested = run_mode == ADVENTURE_GENERALIST_RUN_MODE_TRAIN
    adventure_generalist_eval_requested = run_mode == ADVENTURE_GENERALIST_RUN_MODE_EVAL
    adventure_generalist_requested = adventure_generalist_train_requested or adventure_generalist_eval_requested
    raw_initial_loadout = value("initial_loadout", ",".join(ADVENTURE_GENERALIST_INITIAL_LOADOUT))
    initial_loadout = parse_initial_loadout(raw_initial_loadout)
    raw_core_seed_names = value("core_seed_names", list(DEFAULT_CORE_SEED_NAMES))
    core_seed_names = parse_initial_loadout(raw_core_seed_names)
    if not core_seed_names:
        core_seed_names = list(DEFAULT_CORE_SEED_NAMES)
    new_unlock_guarantee_episodes = max(
        0,
        int(
            value(
                "new_unlock_guarantee_episodes",
                DEFAULT_NEW_UNLOCK_GUARANTEE_EPISODES,
            )
            or 0
        ),
    )
    default_seed_list = ",".join(initial_loadout)
    cli_seed_list = getattr(args, "seed_list", None)
    config_seed_list_present = "seed_list" in raw_config and raw_config.get("seed_list") not in (None, "")
    seed_order_source = SEED_ORDER_SOURCE_DEFAULT
    if adventure_generalist_requested and cli_seed_list is None and not config_seed_list_present:
        raw_seed_list = default_seed_list
        resolver.sources["seed_list"] = ConfigSource.MODE_DEFAULT
    else:
        raw_seed_list = value(
            "seed_list",
            default_seed_list,
            mode_default=default_seed_list,
        )
        if adventure_generalist_requested and (cli_seed_list is not None or config_seed_list_present):
            seed_order_source = SEED_ORDER_SOURCE_EXPLICIT
    if isinstance(raw_seed_list, list):
        seed_list = [str(seed).strip() for seed in raw_seed_list if str(seed).strip()]
    else:
        seed_list = parse_seed_list(str(raw_seed_list))
    derived_plant_types = resolve_seed_list(seed_list)
    raw_plant_types = value(
        "plant_types",
        derived_plant_types,
        mode_default=derived_plant_types,
    )
    if isinstance(raw_plant_types, str):
        plant_types = [int(part.strip()) for part in raw_plant_types.split(",") if part.strip()]
    elif isinstance(raw_plant_types, (list, tuple)):
        plant_types = [int(part) for part in raw_plant_types]
    else:
        raise SystemExit(
            "blocked_reason=invalid_plant_types: expected comma-separated text or a JSON list of integers."
        )
    if plant_types != derived_plant_types:
        raise SystemExit(
            "blocked_reason=seed_plant_type_mismatch: seed_list and plant_types must preserve identical slot order; "
            f"seed_list_resolves_to={derived_plant_types} configured_plant_types={plant_types}"
        )
    raw_action_space_mode = value("action_space_mode", ACTION_SPACE_ADVENTURE_14_IDENTITY)
    try:
        requested_action_space_mode = normalize_action_space_mode(raw_action_space_mode)
    except ValueError as exc:
        raise SystemExit(
            "blocked_reason=unsupported_action_space_mode: "
            f"{raw_action_space_mode}; Adventure Generalist requires "
            f"{ACTION_SPACE_ADVENTURE_14_IDENTITY}."
        ) from exc
    plant_types = list(derived_plant_types)
    if adventure_generalist_train_requested and args.run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = str(Path("runs") / f"{ADVENTURE_GENERALIST_MODEL_FAMILY}_{timestamp}")
    elif (
        adventure_generalist_eval_requested
        and args.run_dir is None
        and getattr(args, "model", None) is not None
    ):
        run_dir = str(Path(args.model).resolve().parent)
    else:
        run_dir = str(value("run_dir", f"runs/{ADVENTURE_GENERALIST_MODEL_FAMILY}"))

    game_speed = float(value("game_speed", 4.0))
    game_speed_mode = str(value("game_speed_mode", "game_speed"))
    if args.valid_speed_mode:
        game_speed_mode = "safe"
    fusion_policy = normalize_fusion_policy(
        value("fusion_policy", FUSION_POLICY_NONE)
    )
    if adventure_generalist_requested:
        if list(seed_list) != list(initial_loadout):
            raise SystemExit(
                "blocked_reason=invalid_adventure_generalist_seed_list: "
                f"expected {','.join(initial_loadout)} got {','.join(seed_list)}"
            )
        if requested_action_space_mode != ACTION_SPACE_ADVENTURE_14_IDENTITY:
            raise SystemExit(
                "blocked_reason=invalid_adventure_generalist_action_space: "
                "Adventure Generalist requires adventure_14slot_identity."
            )

    raw_adventure_soft = getattr(args, "adventure_soft_max_steps", None)
    if raw_adventure_soft is None:
        raw_adventure_soft = raw_config.get("adventure_soft_max_steps", DEFAULT_ADVENTURE_SOFT_MAX_STEPS)
    adventure_soft_max_steps = max(1, int(raw_adventure_soft))
    adventure_hard_max_steps = max(
        adventure_soft_max_steps,
        int(value("adventure_hard_max_steps", DEFAULT_ADVENTURE_HARD_MAX_STEPS)),
    )
    raw_final_wave_extension = getattr(args, "adventure_final_wave_extension", None)
    if raw_final_wave_extension is None:
        raw_final_wave_extension = raw_config.get("adventure_final_wave_extension", True)
    adventure_final_wave_extension = bool(raw_final_wave_extension)
    raw_generalist_strict_startup_validation = getattr(args, "adventure_generalist_strict_startup_validation", None)
    if raw_generalist_strict_startup_validation is None:
        raw_generalist_strict_startup_validation = raw_config.get("adventure_generalist_strict_startup_validation", True)
    adventure_generalist_strict_startup_validation = bool(raw_generalist_strict_startup_validation)
    env_max_steps = adventure_hard_max_steps
    requested_eval_model_path = str(args.model or raw_config.get("model_path", "") or "").strip()
    requested_resume_model_path = str(getattr(args, "resume_model_path", None) or raw_config.get("resume_model_path", "") or "").strip()
    if requested_resume_model_path and requested_eval_model_path and requested_resume_model_path != requested_eval_model_path:
        raise SystemExit(
            "blocked_reason=invalid_cli: --resume-model-path conflicts with --model-path. "
            "Provide one path or make them match."
        )
    requested_model_path = requested_resume_model_path or requested_eval_model_path
    if requested_resume_model_path and adventure_generalist_eval_requested:
        raise SystemExit("blocked_reason=invalid_cli: --resume-model-path is training-only.")
    requested_training_continuation = bool(requested_model_path) and adventure_generalist_train_requested
    randomize_seed_order = enabled(
        "randomize_seed_order",
        json_aliases=("seed_order_randomization",),
    )
    if adventure_generalist_requested and randomize_seed_order:
        seed_order_source = SEED_ORDER_SOURCE_RANDOMIZED
    coach_allow_fusion_planning = enabled("coach_allow_fusion_planning")
    fusion_bridge_enabled = enabled("fusion_bridge_enabled")
    human_coach_enabled = enabled("human_coach_enabled")
    streamer_v1_enabled = enabled("streamer_v1_enabled")
    human_coach_command_mode = str(
        getattr(args, "human_coach_command_mode", None)
        or raw_config.get("human_coach_command_mode", "override")
        or "override"
    ).strip().lower()
    stream_coach_enabled = enabled("stream_coach_enabled")
    try:
        stream_coach_mode = str(
            resolver.aliased_value(
                "stream_coach_mode",
                "mock",
                cli_keys=("stream_coach_mode", "stream_coach_platform"),
                json_keys=("stream_coach_mode", "stream_coach_platform"),
                skip_blank=True,
            )
        ).strip().lower()
    except ValueError as exc:
        raise SystemExit(f"blocked_reason=invalid_cli: {exc}") from exc
    resolver.sources["stream_coach_platform"] = resolver.sources["stream_coach_mode"]
    default_coach_command_path = "runs/coach_commands.jsonl"
    raw_human_command_path = (
        getattr(args, "human_coach_command_path", None)
        or raw_config.get("human_coach_command_path", "")
        or ""
    )
    human_coach_command_path = str(raw_human_command_path or "").strip()
    if human_coach_enabled and not human_coach_command_path:
        human_coach_command_path = default_coach_command_path
    raw_stream_command_path = (
        getattr(args, "stream_coach_command_path", None)
        or raw_config.get("stream_coach_command_path", "")
        or ""
    )
    stream_coach_command_path = str(raw_stream_command_path or "").strip()
    if stream_coach_enabled and not stream_coach_command_path:
        stream_coach_command_path = human_coach_command_path or default_coach_command_path
    human_coach_fusion_enabled = bool(
        enabled("human_coach_fusion_enabled")
        or coach_allow_fusion_planning
        or fusion_bridge_enabled
    )
    stream_coach_fusion_enabled = bool(
        enabled("stream_coach_fusion_enabled")
        or coach_allow_fusion_planning
        or fusion_bridge_enabled
    )
    stream_coach_apply_enabled = enabled(
        "stream_coach_apply_enabled",
        cli_key="stream_coach_apply",
        json_aliases=("stream_coach_apply",),
    )
    stream_coach_dry_run = bool(raw_config.get("stream_coach_dry_run", True))
    if getattr(args, "stream_coach_apply", False):
        stream_coach_dry_run = False
    elif getattr(args, "stream_coach_dry_run", False):
        stream_coach_dry_run = True
        stream_coach_apply_enabled = False
    elif "stream_coach_apply_enabled" in raw_config or "stream_coach_apply" in raw_config:
        stream_coach_dry_run = not bool(stream_coach_apply_enabled)
    if stream_coach_dry_run:
        stream_coach_apply_enabled = False
    elif not stream_coach_apply_enabled:
        stream_coach_apply_enabled = True

    config = {
        "policy": raw_config.get("policy", "MlpPolicy"),
        "model_family": ADVENTURE_GENERALIST_MODEL_FAMILY,
        "total_timesteps": int(value("total_timesteps", 25000)),
        "learning_rate": float(value("learning_rate", 3e-4)),
        "n_steps": int(
            value("n_steps", 512, mode_default=500 if streamer_v1_enabled else CONFIG_UNSET)
        ),
        "batch_size": int(
            value("batch_size", 64, mode_default=50 if streamer_v1_enabled else CONFIG_UNSET)
        ),
        "gamma": float(value("gamma", 0.99)),
        "gae_lambda": float(value("gae_lambda", 0.95)),
        "ent_coef": float(value("ent_coef", 0.01)),
        "clip_range": float(value("clip_range", 0.2)),
        "verbose": int(value("verbose", 1)),
        "max_steps": int(env_max_steps),
        "adventure_soft_max_steps": int(adventure_soft_max_steps),
        "adventure_hard_max_steps": int(adventure_hard_max_steps),
        "adventure_final_wave_extension": bool(adventure_final_wave_extension),
        "step_seconds": float(value("step_seconds", 0.05)),
        "game_speed": game_speed,
        "game_speed_mode": game_speed_mode,
        "start_sun": int(value("start_sun", 500)),
        "seed": int(value("seed", 12345)),
        "board_timeout": float(board_timeout),
        "gameplay_ready_timeout": float(gameplay_ready_timeout),
        "poll_seconds": float(poll_seconds),
        "wait_gameplay_ready": enabled("wait_gameplay_ready", True),
        "skip_board_wait": enabled("skip_board_wait"),
        "quick_wait": quick_wait,
        "plant_types": plant_types,
        "action_space_mode": requested_action_space_mode,
        "max_seed_slots": int(
            value(
                "max_seed_slots",
                SEED_CAPACITY_MAX,
                mode_default=SEED_CAPACITY_MAX,
            )
        ),
        "auto_select_seeds": True,
        "seed_list": seed_list,
        "initial_loadout": list(initial_loadout),
        "configured_seed_list": list(seed_list),
        "seed_order_source": seed_order_source,
        "seed_order_preserved": True,
        "randomize_seed_order": bool(randomize_seed_order),
        "seed_click_delay": float(value("seed_click_delay", 0.35)),
        "lets_rock_delay": float(value("lets_rock_delay", 0.5)),
        "post_start_delay": float(value("post_start_delay", 1.0)),
        "seed_screen_check_interval": int(value("seed_screen_check_interval", 100)),
        "debug_performance": enabled("debug_performance"),
        "debug_observation": enabled("debug_observation"),
        "debug_sun": bool(
            enabled("debug_sun")
            or enabled("debug_performance")
        ),
        "debug_sun_sample_interval": int(value("debug_sun_sample_interval", 25)),
        "fusion_policy": fusion_policy,
        "fusion_action_mask_enabled": enabled(
            "fusion_action_mask_enabled",
            mode_default=streamer_v1_enabled,
        ),
        "enable_board_plant_identity": bool(
            value(
                "enable_board_plant_identity",
                False,
                mode_default=True if streamer_v1_enabled else CONFIG_UNSET,
            )
        ),
        "enable_fusion_chain_rewards": bool(value("enable_fusion_chain_rewards", False)),
        "enable_recipe_discovery_reward": bool(value("enable_recipe_discovery_reward", False)),
        "enable_repeat_recipe_decay": bool(value("enable_repeat_recipe_decay", False)),
        "enable_fusion_curriculum": bool(value("enable_fusion_curriculum", False)),
        "enable_later_plant_curriculum": bool(value("enable_later_plant_curriculum", False)),
        "enable_coach_fusion_sampling": bool(value("enable_coach_fusion_sampling", False)),
        "fusion_curriculum_prob": float(value("fusion_curriculum_prob", 0.20)),
        "later_plant_curriculum_prob": float(value("later_plant_curriculum_prob", 0.10)),
        "coach_fusion_prob": float(value("coach_fusion_prob", 0.10)),
        "run_mode": run_mode,
        "tactical_masks": enabled("tactical_masks"),
        "wallnut_tactical_mask": enabled("wallnut_tactical_mask"),
        "cherrybomb_tactical_mask": enabled("cherrybomb_tactical_mask"),
        "checkpoint_warm_start": bool(requested_training_continuation),
        "warm_start_used": False,
        "checkpoint_warm_start_reason": (
            "compatible_adventure_generalist_continuation_requested"
            if adventure_generalist_train_requested and requested_training_continuation
            else str(value("checkpoint_warm_start_reason", "") or "")
        ),
        "resume_training": bool(requested_training_continuation),
        "resume_model_path": requested_model_path if requested_training_continuation else "",
        "resume_source_model_family": "",
        "scratch_initialization": bool(adventure_generalist_train_requested and not requested_training_continuation),
        "active_seed_slots_at_start": len(initial_loadout),
        "unlock_aware_seed_curriculum": enabled("unlock_aware_seed_curriculum", True),
        "seed_curriculum": str(value("seed_curriculum", "conservative")),
        "unlock_introduction_delay": int(value("unlock_introduction_delay", 0)),
        "new_plant_min_inclusion_prob": float(value("new_plant_min_inclusion_prob", 0.15)),
        "core_seed_names": list(core_seed_names),
        "new_unlock_guarantee_episodes": int(new_unlock_guarantee_episodes),
        "infer_capacity_from_unlocks": bool(
            value(
                "infer_capacity_from_unlocks",
                False,
                mode_default=True,
            )
        ),
        "allow_weak_unlocked_capacity_fallback": enabled("allow_weak_unlocked_capacity_fallback"),
        "adventure_replay_cleared_levels": enabled("adventure_replay_cleared_levels"),
        "adventure_frontier_sample_prob": float(value("adventure_frontier_sample_prob", 0.60)),
        "adventure_recent_cleared_sample_prob": float(value("adventure_recent_cleared_sample_prob", 0.30)),
        "adventure_maintenance_sample_prob": float(value("adventure_maintenance_sample_prob", 0.10)),
        "adventure_frontier_win_streak_required": max(
            1,
            int(value("adventure_frontier_win_streak_required", 1) or 1),
        ),
        "adventure_generalist_strict_startup_validation": bool(adventure_generalist_strict_startup_validation),
        "adventure_start_level": int(value("adventure_start_level", 1)),
        "max_adventure_levels": int(value("max_adventure_levels", 5)),
        "max_attempts_per_level": int(value("max_attempts_per_level", 10)),
        "advance_on_wins": int(value("advance_on_wins", 1)),
        "human_coach_enabled": bool(human_coach_enabled),
        "human_coach_command_path": str(human_coach_command_path),
        "human_coach_log_path": str(value("human_coach_log_path", "runs/human_coach.jsonl") or ""),
        "human_coach_reward": enabled("human_coach_reward"),
        "human_coach_fusion_enabled": bool(human_coach_fusion_enabled),
        "human_coach_platform": str(raw_config.get("human_coach_platform", "mock") or "mock"),
        "human_coach_command_mode": human_coach_command_mode,
        "intervention_log_path": str(
            value("intervention_log_path", "logs/interventions/interventions.jsonl")
        ),
        "stream_coach_enabled": bool(stream_coach_enabled),
        "stream_coach_mode": stream_coach_mode,
        "stream_coach_platform": stream_coach_mode,
        "stream_coach_window_sec": float(
            value("stream_coach_window_sec", 3.0)
        ),
        "stream_coach_min_votes": int(
            value("stream_coach_min_votes", 2)
        ),
        "stream_coach_max_actions_per_minute": int(
            value("stream_coach_max_actions_per_minute", 20)
        ),
        "stream_coach_command_path": str(stream_coach_command_path),
        "stream_coach_mock_script": str(value("stream_coach_mock_script", "") or ""),
        "stream_coach_dry_run": bool(stream_coach_dry_run),
        "stream_coach_apply_enabled": bool(stream_coach_apply_enabled),
        "stream_coach_reward": enabled("stream_coach_reward"),
        "stream_coach_log_path": str(value("stream_coach_log_path", "runs/stream_coach.jsonl") or ""),
        "stream_coach_fusion_enabled": bool(stream_coach_fusion_enabled),
        "coach_allow_fusion_planning": bool(coach_allow_fusion_planning),
        "fusion_bridge_enabled": bool(fusion_bridge_enabled),
        "streamer_v1_enabled": bool(streamer_v1_enabled),
        "streamer_platform": str(value("streamer_platform", "twitch") or "twitch").strip().lower(),
        "streamer_baseline_checkpoint": str(value("streamer_baseline_checkpoint", "") or "").strip(),
        "streamer_intervention_interval_seconds": float(
            value("streamer_intervention_interval_seconds", 2.0)
        ),
        "streamer_command_ttl_seconds": float(value("streamer_command_ttl_seconds", 10.0)),
        "streamer_command_queue_capacity": int(value("streamer_command_queue_capacity", 256)),
        "streamer_message_max_chars": int(value("streamer_message_max_chars", 256)),
        "streamer_policy_steps_per_cycle": int(value("streamer_policy_steps_per_cycle", 25000)),
        "streamer_checkpoint_policy_steps": int(
            value("streamer_checkpoint_policy_steps", 5000)
        ),
        "streamer_evaluation_episodes": int(value("streamer_evaluation_episodes", 50)),
        "streamer_max_cycles": int(value("streamer_max_cycles", 0)),
        "streamer_endurance_hours": float(value("streamer_endurance_hours", 0.0)),
        "streamer_bc_enabled": bool(value("streamer_bc_enabled", True)),
        "streamer_bc_coefficient": float(value("streamer_bc_coefficient", 0.01)),
        "streamer_demonstration_capacity": int(value("streamer_demonstration_capacity", 4096)),
        "streamer_demonstration_persist_every": int(
            value("streamer_demonstration_persist_every", 512)
        ),
        "streamer_bc_batch_size": int(value("streamer_bc_batch_size", 32)),
        "streamer_bc_update_frequency": int(value("streamer_bc_update_frequency", 1)),
        "streamer_bc_min_demonstrations": int(value("streamer_bc_min_demonstrations", 8)),
        "streamer_twitch_client_id_env": str(
            value("streamer_twitch_client_id_env", "PVZRL_TWITCH_CLIENT_ID") or ""
        ).strip(),
        "streamer_twitch_access_token_env": str(
            value("streamer_twitch_access_token_env", "PVZRL_TWITCH_USER_ACCESS_TOKEN") or ""
        ).strip(),
        "streamer_twitch_broadcaster_id_env": str(
            value("streamer_twitch_broadcaster_id_env", "PVZRL_TWITCH_BROADCASTER_USER_ID") or ""
        ).strip(),
        "streamer_twitch_user_id_env": str(
            value("streamer_twitch_user_id_env", "PVZRL_TWITCH_EVENTSUB_USER_ID") or ""
        ).strip(),
        "streamer_viewer_hash_secret_env": str(
            value("streamer_viewer_hash_secret_env", "PVZRL_TWITCH_VIEWER_HASH_SECRET") or ""
        ).strip(),
        "streamer_mock_script": str(value("streamer_mock_script", "") or "").strip(),
        "reward": build_reward_config(args, raw_config),
        "model_path": requested_model_path,
        "run_dir": run_dir,
        "checkpoint_freq": int(value("checkpoint_freq", 5000)),
        "host": str(value("host", "127.0.0.1")),
        "port": int(value("port", 32323)),
        "timeout": float(value("timeout", 10.0)),
        "enable_action_watchdog": bool(value("enable_action_watchdog", True)),
        "action_timeout_seconds": float(value("action_timeout_seconds", 10.0)),
        "save_freeze_debug_bundle": bool(value("save_freeze_debug_bundle", True)),
        "action_diagnostics_path": str(
            value("action_diagnostics_path", str(Path(run_dir) / "action_diagnostics.jsonl"))
        ),
        "freeze_debug_dir": str(
            value("freeze_debug_dir", str(Path(run_dir) / "freeze_debug"))
        ),
        "game_exe": str(value("game_exe", "") or ""),
    }
    config = apply_model_metadata_defaults(config)
    if config["streamer_v1_enabled"]:
        if run_mode != ADVENTURE_GENERALIST_RUN_MODE_TRAIN:
            raise SystemExit(
                "blocked_reason=invalid_streamer_run_mode: Streamer V1 is an overlay on Adventure Generalist training."
            )
        if config["streamer_platform"] not in {"twitch", "mock"}:
            raise SystemExit(
                "blocked_reason=invalid_streamer_platform: expected twitch or mock."
            )
        if not config["streamer_baseline_checkpoint"]:
            raise SystemExit(
                "blocked_reason=streamer_baseline_checkpoint_required: configure a repository-relative Generalist .zip path."
            )
        streamer_env_fields = {
            "streamer_twitch_client_id_env": config["streamer_twitch_client_id_env"],
            "streamer_twitch_access_token_env": config["streamer_twitch_access_token_env"],
            "streamer_twitch_broadcaster_id_env": config[
                "streamer_twitch_broadcaster_id_env"
            ],
            "streamer_twitch_user_id_env": config["streamer_twitch_user_id_env"],
            "streamer_viewer_hash_secret_env": config["streamer_viewer_hash_secret_env"],
        }
        invalid_env_fields = [
            name
            for name, env_name in streamer_env_fields.items()
            if env_name and STREAMER_ENV_NAME_PATTERN.fullmatch(str(env_name)) is None
        ]
        if config["streamer_platform"] == "twitch":
            invalid_env_fields.extend(
                name
                for name in (
                    "streamer_twitch_client_id_env",
                    "streamer_twitch_access_token_env",
                    "streamer_twitch_broadcaster_id_env",
                    "streamer_viewer_hash_secret_env",
                )
                if not streamer_env_fields[name]
            )
        if invalid_env_fields:
            raise SystemExit(
                "blocked_reason=invalid_streamer_environment_variable_name: "
                + ",".join(sorted(set(invalid_env_fields)))
            )
        positive_streamer_fields = {
            "n_steps": config["n_steps"],
            "batch_size": config["batch_size"],
            "streamer_intervention_interval_seconds": config["streamer_intervention_interval_seconds"],
            "streamer_command_ttl_seconds": config["streamer_command_ttl_seconds"],
            "streamer_command_queue_capacity": config["streamer_command_queue_capacity"],
            "streamer_message_max_chars": config["streamer_message_max_chars"],
            "streamer_policy_steps_per_cycle": config["streamer_policy_steps_per_cycle"],
            "streamer_checkpoint_policy_steps": config["streamer_checkpoint_policy_steps"],
            "streamer_evaluation_episodes": config["streamer_evaluation_episodes"],
            "streamer_demonstration_capacity": config["streamer_demonstration_capacity"],
            "streamer_demonstration_persist_every": config[
                "streamer_demonstration_persist_every"
            ],
            "streamer_bc_batch_size": config["streamer_bc_batch_size"],
            "streamer_bc_update_frequency": config["streamer_bc_update_frequency"],
            "streamer_bc_min_demonstrations": config["streamer_bc_min_demonstrations"],
        }
        invalid = [name for name, configured in positive_streamer_fields.items() if float(configured) <= 0]
        if invalid:
            raise SystemExit(
                "blocked_reason=invalid_streamer_config: positive values required for " + ",".join(invalid)
            )
        finite_fields = {
            "streamer_intervention_interval_seconds": config["streamer_intervention_interval_seconds"],
            "streamer_command_ttl_seconds": config["streamer_command_ttl_seconds"],
            "streamer_endurance_hours": config["streamer_endurance_hours"],
            "streamer_bc_coefficient": config["streamer_bc_coefficient"],
        }
        if any(not math.isfinite(float(value)) for value in finite_fields.values()):
            raise SystemExit("blocked_reason=invalid_streamer_config: finite numeric values required.")
        if float(config["streamer_intervention_interval_seconds"]) < 0.1:
            raise SystemExit(
                "blocked_reason=invalid_streamer_intervention_interval: "
                "expected a finite value in [0.1,3600]."
            )
        upper_bounds = {
            "n_steps": 8_192,
            "batch_size": 4_096,
            "streamer_intervention_interval_seconds": 3600.0,
            "streamer_command_ttl_seconds": 3600.0,
            "streamer_command_queue_capacity": 4096,
            "streamer_message_max_chars": 1024,
            "streamer_policy_steps_per_cycle": 10_000_000,
            "streamer_checkpoint_policy_steps": 10_000_000,
            "streamer_evaluation_episodes": 10_000,
            "streamer_demonstration_capacity": 16_384,
            "streamer_demonstration_persist_every": 16_384,
            "streamer_bc_batch_size": 4096,
            "streamer_bc_update_frequency": 1_000_000,
            "streamer_bc_min_demonstrations": 16_384,
            "streamer_max_cycles": 1_000_000,
            "streamer_endurance_hours": 168.0,
        }
        excessive = [
            name
            for name, maximum in upper_bounds.items()
            if float(config[name]) > float(maximum)
        ]
        if excessive:
            raise SystemExit(
                "blocked_reason=invalid_streamer_config: resource ceiling exceeded for "
                + ",".join(excessive)
            )
        if not 0.0 <= config["streamer_bc_coefficient"] <= 1.0:
            raise SystemExit(
                "blocked_reason=invalid_streamer_bc_coefficient: expected a finite value in [0,1]."
            )
        if config["streamer_max_cycles"] < 0 or config["streamer_endurance_hours"] < 0.0:
            raise SystemExit("blocked_reason=invalid_streamer_duration: max cycles and endurance hours cannot be negative.")
        if config["streamer_policy_steps_per_cycle"] % config["n_steps"] != 0:
            raise SystemExit(
                "blocked_reason=streamer_cycle_rollout_alignment: "
                "streamer_policy_steps_per_cycle must be divisible by n_steps for exact policy-step cycles."
            )
        if config["n_steps"] % config["batch_size"] != 0:
            raise SystemExit(
                "blocked_reason=streamer_minibatch_alignment: n_steps must be divisible by batch_size."
            )
        if config["streamer_checkpoint_policy_steps"] > config["streamer_policy_steps_per_cycle"]:
            raise SystemExit(
                "blocked_reason=invalid_streamer_config: checkpoint interval cannot exceed cycle policy steps."
            )
        if config["streamer_demonstration_persist_every"] > config["streamer_demonstration_capacity"]:
            raise SystemExit(
                "blocked_reason=invalid_streamer_config: demonstration persist interval cannot exceed capacity."
            )
        if config["streamer_bc_batch_size"] > config["streamer_demonstration_capacity"]:
            raise SystemExit(
                "blocked_reason=invalid_streamer_config: BC batch size cannot exceed demonstration capacity."
            )
        if config["streamer_bc_min_demonstrations"] > config["streamer_demonstration_capacity"]:
            raise SystemExit(
                "blocked_reason=invalid_streamer_config: BC minimum cannot exceed demonstration capacity."
            )
        if stream_coach_enabled or human_coach_enabled:
            raise SystemExit(
                "blocked_reason=conflicting_coach_mode: Streamer V1 owns intervention collection; disable legacy coach overrides."
            )
    if adventure_generalist_requested and action_count_for_config(config) != ADVENTURE_IDENTITY_ACTION_COUNT:
        raise SystemExit(
            "blocked_reason=invalid_adventure_generalist_action_count: "
            f"expected {ADVENTURE_IDENTITY_ACTION_COUNT} got {action_count_for_config(config)}"
        )
    if resolution_sources is not None:
        resolution_sources.update(resolver.sources)
    return config


def build_resolved_config(args: argparse.Namespace, raw_config: Dict[str, Any]) -> ResolvedRunConfig:
    """Resolve CLI/JSON/default inputs into immutable typed sections."""

    resolution_sources: Dict[str, Any] = {}
    flat = _build_config_mapping(args, raw_config, resolution_sources=resolution_sources)
    return ResolvedRunConfig.from_flat(flat, value_sources=resolution_sources)


def build_config(args: argparse.Namespace, raw_config: Dict[str, Any]) -> Dict[str, Any]:
    """Backward-compatible flat adapter used by existing runtime consumers."""

    return build_resolved_config(args, raw_config).to_flat_dict()


def make_env_config(config: Dict[str, Any]) -> PvZSB3Config:
    return PvZSB3Config(
        host=config["host"],
        port=config["port"],
        timeout=config["timeout"],
        enable_action_watchdog=bool(config.get("enable_action_watchdog", True)),
        action_timeout_seconds=float(config.get("action_timeout_seconds", config["timeout"])),
        save_freeze_debug_bundle=bool(config.get("save_freeze_debug_bundle", True)),
        action_diagnostics_path=str(config.get("action_diagnostics_path", "") or ""),
        freeze_debug_dir=str(config.get("freeze_debug_dir", "") or ""),
        step_seconds=config["step_seconds"],
        plant_types=list(config["plant_types"]),
        action_space_mode=str(config.get("action_space_mode", ACTION_SPACE_ADVENTURE_14_IDENTITY)),
        max_seed_slots=int(config.get("max_seed_slots", len(config["plant_types"]))),
        observation_version=str(config.get("observation_version", "")),
        action_decoder_version=str(config.get("action_decoder_version", "")),
        row_count=5,
        column_count=10,
        game_speed=config["game_speed"],
        game_speed_mode=config.get("game_speed_mode", "game_speed"),
        seed=config["seed"],
        start_sun=config["start_sun"],
        max_steps=config["max_steps"],
        wait_for_board=not config["skip_board_wait"],
        wait_gameplay_ready=config["wait_gameplay_ready"],
        board_timeout=config["board_timeout"],
        gameplay_ready_timeout=config["gameplay_ready_timeout"],
        poll_seconds=config["poll_seconds"],
        auto_select_seeds=config["auto_select_seeds"],
        seed_list=list(config["seed_list"]),
        seed_click_delay=config["seed_click_delay"],
        lets_rock_delay=config["lets_rock_delay"],
        post_start_delay=config["post_start_delay"],
        seed_screen_check_interval=config["seed_screen_check_interval"],
        debug_performance=config["debug_performance"],
        debug_observation=config["debug_observation"],
        debug_sun=config.get("debug_sun", False),
        debug_sun_sample_interval=config.get("debug_sun_sample_interval", 25),
        fusion_policy=str(config.get("fusion_policy", FUSION_POLICY_NONE)),
        fusion_action_mask_enabled=bool(config.get("fusion_action_mask_enabled", False)),
        enable_board_plant_identity=bool(config.get("enable_board_plant_identity", False)),
        enable_fusion_chain_rewards=bool(config.get("enable_fusion_chain_rewards", False)),
        enable_recipe_discovery_reward=bool(config.get("enable_recipe_discovery_reward", False)),
        enable_repeat_recipe_decay=bool(config.get("enable_repeat_recipe_decay", False)),
        enable_fusion_curriculum=bool(config.get("enable_fusion_curriculum", False)),
        enable_later_plant_curriculum=bool(config.get("enable_later_plant_curriculum", False)),
        enable_coach_fusion_sampling=bool(config.get("enable_coach_fusion_sampling", False)),
        fusion_curriculum_prob=float(config.get("fusion_curriculum_prob", 0.20)),
        later_plant_curriculum_prob=float(config.get("later_plant_curriculum_prob", 0.10)),
        coach_fusion_prob=float(config.get("coach_fusion_prob", 0.10)),
        run_mode=str(config.get("run_mode", ADVENTURE_GENERALIST_RUN_MODE_TRAIN)),
        tactical_masks=bool(config.get("tactical_masks", False)),
        wallnut_tactical_mask=bool(config.get("wallnut_tactical_mask", False)),
        cherrybomb_tactical_mask=bool(config.get("cherrybomb_tactical_mask", False)),
        game_exe=str(config.get("game_exe") or "") or None,
        human_coach_enabled=bool(config.get("human_coach_enabled", False)),
        human_coach_log_path=str(config.get("human_coach_log_path", "") or ""),
        human_coach_command_path=str(config.get("human_coach_command_path", "") or ""),
        human_coach_reward=bool(config.get("human_coach_reward", False)),
        human_coach_fusion_enabled=bool(config.get("human_coach_fusion_enabled", False)),
        human_coach_platform=str(config.get("human_coach_platform", "mock") or "mock"),
        human_coach_command_mode=str(config.get("human_coach_command_mode", "override") or "override"),
        intervention_log_path=str(config.get("intervention_log_path", "logs/interventions/interventions.jsonl")),
        stream_coach_enabled=bool(config.get("stream_coach_enabled", False)),
        stream_coach_mode=str(config.get("stream_coach_mode", config.get("stream_coach_platform", "mock")) or "mock"),
        stream_coach_platform=str(config.get("stream_coach_platform", "mock") or "mock"),
        stream_coach_window_sec=float(config.get("stream_coach_window_sec", 3.0) or 3.0),
        stream_coach_min_votes=int(config.get("stream_coach_min_votes", 2) or 2),
        stream_coach_max_actions_per_minute=int(config.get("stream_coach_max_actions_per_minute", 20) or 20),
        stream_coach_reward=bool(config.get("stream_coach_reward", False)),
        stream_coach_log_path=str(config.get("stream_coach_log_path", "") or ""),
        stream_coach_command_path=str(config.get("stream_coach_command_path", "") or ""),
        stream_coach_mock_script=str(config.get("stream_coach_mock_script", "") or ""),
        stream_coach_dry_run=bool(config.get("stream_coach_dry_run", True)),
        stream_coach_apply_enabled=bool(config.get("stream_coach_apply_enabled", False)),
        stream_coach_fusion_enabled=bool(config.get("stream_coach_fusion_enabled", False)),
        reward=reward_config_from_mapping(config.get("reward", {})),
    )


def make_monitored_env(config: Dict[str, Any], monitor_path: Path, live_status_path: Optional[Path] = None) -> Any:
    _, _, _, _, Monitor = require_sb3_callbacks()
    if str(config.get("run_mode", "")) == ADVENTURE_GENERALIST_RUN_MODE_TRAIN:
        env = AdventureGeneralistTrainingEnv(
            make_env_config(config),
            run_dir=Path(config["run_dir"]),
            live_status_path=live_status_path,
            initial_loadout=parse_initial_loadout(config.get("initial_loadout", ADVENTURE_GENERALIST_INITIAL_LOADOUT)),
            max_adventure_levels=int(config.get("max_adventure_levels", 5) or 5),
            max_attempts_per_level=int(config.get("max_attempts_per_level", 10) or 10),
            adventure_start_level=int(config.get("adventure_start_level", 1) or 1),
            unlock_aware_seed_curriculum=bool(config.get("unlock_aware_seed_curriculum", True)),
            seed_curriculum=str(config.get("seed_curriculum", "conservative")),
            unlock_introduction_delay=int(config.get("unlock_introduction_delay", 0) or 0),
            new_plant_min_inclusion_prob=float(config.get("new_plant_min_inclusion_prob", 0.15) or 0.0),
            core_seed_names=list(config.get("core_seed_names", DEFAULT_CORE_SEED_NAMES) or DEFAULT_CORE_SEED_NAMES),
            new_unlock_guarantee_episodes=int(
                config.get("new_unlock_guarantee_episodes", DEFAULT_NEW_UNLOCK_GUARANTEE_EPISODES)
                or 0
            ),
            seed_order_source=str(config.get("seed_order_source", "default_canonical")),
            randomize_seed_order=bool(config.get("randomize_seed_order", False)),
            infer_capacity_from_unlocks=bool(config.get("infer_capacity_from_unlocks", True)),
            allow_weak_unlocked_capacity_fallback=bool(config.get("allow_weak_unlocked_capacity_fallback", False)),
            replay_cleared_levels=bool(config.get("adventure_replay_cleared_levels", False)),
            frontier_sample_prob=float(config.get("adventure_frontier_sample_prob", 0.60) or 0.0),
            recent_cleared_sample_prob=float(config.get("adventure_recent_cleared_sample_prob", 0.30) or 0.0),
            maintenance_sample_prob=float(config.get("adventure_maintenance_sample_prob", 0.10) or 0.0),
            frontier_win_streak_required=int(config.get("adventure_frontier_win_streak_required", 1) or 1),
            strict_startup_validation=bool(config.get("adventure_generalist_strict_startup_validation", True)),
        )
    else:
        env = PvZMaskedPPOEnv(make_env_config(config))
    return Monitor(
        env,
        filename=str(monitor_path),
        info_keywords=(
            "done_reason",
            "terminal_reason",
            "final_wave",
            "zombies_killed",
            "plants_placed",
            "sun_spent",
            "sun_remaining",
            "mowers_lost",
            "reset_success",
            "reset_seconds",
            "bridge_errors",
            "illegal_actions",
            "avg_legal_actions",
            *EPISODE_STRING_FIELDS,
            *LANE_DIAGNOSTIC_DICT_FIELDS,
            *LANE_DIAGNOSTIC_FLOAT_DICT_FIELDS,
            *LANE_DIAGNOSTIC_NUMERIC_FIELDS,
            "wallnut_damage_absorbed_total",
            "cherrybomb_avg_kills_per_use",
            "fusion_candidate_count_avg",
            "fusion_avg_kills_after_use",
            *FUSION_REWARD_FLOAT_FIELDS,
            "fusion_reward_capped",
            *REWARD_EPISODE_TOTAL_FIELDS,
        ),
    )


def make_streamer_v1_controller(config: Dict[str, Any]) -> Any:
    """Build the source-neutral bounded controller without starting network I/O."""

    from pvzrl_stream_commands import (
        BoundedViewerCommandQueue,
        ViewerCommandController,
        ViewerCommandParser,
    )

    platform = str(config.get("streamer_platform", "twitch") or "twitch").lower()
    capacity = int(config.get("streamer_command_queue_capacity", 256) or 256)
    max_chars = int(config.get("streamer_message_max_chars", 256) or 256)
    if platform == "twitch":
        from pvzrl_twitch import TwitchCredentials, TwitchEventSubSource

        env_names = {
            "client_id": str(config.get("streamer_twitch_client_id_env") or ""),
            "access_token": str(config.get("streamer_twitch_access_token_env") or ""),
            "broadcaster_user_id": str(config.get("streamer_twitch_broadcaster_id_env") or ""),
            "eventsub_user_id": str(config.get("streamer_twitch_user_id_env") or ""),
            "viewer_hash_secret": str(config.get("streamer_viewer_hash_secret_env") or ""),
        }
        required = ("client_id", "access_token", "broadcaster_user_id", "viewer_hash_secret")
        missing = [env_names[key] or f"<{key}_env>" for key in required if not env_names[key] or not os.environ.get(env_names[key])]
        if missing:
            raise SystemExit(
                "blocked_reason=streamer_twitch_environment_missing: set " + ",".join(missing)
            )
        credentials = TwitchCredentials(
            client_id=os.environ[env_names["client_id"]],
            access_token=os.environ[env_names["access_token"]],
            broadcaster_user_id=os.environ[env_names["broadcaster_user_id"]],
            eventsub_user_id=(
                os.environ.get(env_names["eventsub_user_id"], "")
                if env_names["eventsub_user_id"]
                else ""
            ),
            viewer_hash_secret=os.environ[env_names["viewer_hash_secret"]].encode("utf-8"),
        )
        source = TwitchEventSubSource(
            credentials,
            queue_capacity=capacity,
            accepting=False,
            max_command_chars=max_chars,
        )
    elif platform == "mock":
        from pvzrl_streamer_source import (
            DeterministicStreamCommandSource,
            ScriptedStreamSourceRecord,
        )

        mock_path = Path(str(config.get("streamer_mock_script") or ""))
        if not mock_path.is_file():
            raise SystemExit(f"blocked_reason=streamer_mock_script_missing: {mock_path}")
        max_mock_bytes = 16 * 1024 * 1024
        if mock_path.stat().st_size > max_mock_bytes:
            raise SystemExit(
                "blocked_reason=invalid_streamer_mock_script: "
                f"path={mock_path} reason=script_byte_limit_exceeded"
            )
        records = []
        secret_env_name = str(config.get("streamer_viewer_hash_secret_env") or "")
        secret_text = os.environ.get(secret_env_name, "") if secret_env_name else ""
        try:
            with mock_path.open("r", encoding="utf-8") as handle:
                line_number = 0
                while True:
                    raw_line = handle.readline(4_097)
                    if raw_line == "":
                        break
                    line_number += 1
                    if line_number > 100_000:
                        raise ValueError("script_record_limit_exceeded")
                    if len(raw_line) > 4_096 or (
                        len(raw_line) == 4_096 and not raw_line.endswith("\n")
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
                    command = payload.get("command")
                    if not isinstance(command, str):
                        raise ValueError(f"line_{line_number}_command_required")
                    local_viewer_id = payload.get("local_viewer_id", "")
                    viewer_hash = payload.get("viewer_hash", "")
                    if local_viewer_id and not secret_text:
                        raise ValueError(
                            f"line_{line_number}_local_viewer_id_requires={secret_env_name or '<viewer_hash_secret_env>'}"
                        )
                    records.append(
                        ScriptedStreamSourceRecord(
                            command_text=command,
                            viewer_hash=str(viewer_hash or ""),
                            local_viewer_id=str(local_viewer_id or ""),
                            delivery_id=str(payload.get("delivery_id") or ""),
                            event_id=str(payload.get("event_id") or ""),
                            published_at=(
                                str(payload.get("published_at"))
                                if payload.get("published_at") is not None
                                else None
                            ),
                        )
                    )
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise SystemExit(
                "blocked_reason=invalid_streamer_mock_script: "
                f"path={mock_path} reason={exc}"
            ) from exc
        source = DeterministicStreamCommandSource(
            records,
            viewer_hash_secret=secret_text.encode("utf-8") if secret_text else None,
            queue_capacity=capacity,
            accepting=False,
            max_command_chars=max_chars,
        )
    else:
        raise SystemExit(f"blocked_reason=invalid_streamer_platform: {platform}")

    return ViewerCommandController(
        source=source,
        parser=ViewerCommandParser(max_message_length=max_chars),
        queue=BoundedViewerCommandQueue(
            capacity=capacity,
            ttl_seconds=float(config.get("streamer_command_ttl_seconds", 10.0) or 10.0),
        ),
        opportunity_interval_seconds=float(
            config.get("streamer_intervention_interval_seconds", 2.0) or 2.0
        ),
    )


def write_action_map(config: Dict[str, Any], path: Path) -> None:
    rows = 5
    cols = 10
    plant_types = list(config["plant_types"])
    seed_list = list(config.get("seed_list", []))
    spec = build_action_space_spec(
        mode=str(config.get("action_space_mode", ACTION_SPACE_ADVENTURE_14_IDENTITY)),
        plant_types=[int(value) for value in plant_types],
        max_seed_slots=int(config.get("max_seed_slots", len(plant_types))),
        rows=rows,
        cols=cols,
    )
    action_count = action_count_for_config(config)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            f"mode={spec.mode} seed_slots={len(plant_types)} max_seed_slots={spec.max_seed_slots} "
            f"rows={rows} cols={cols} action_count={action_count} "
            f"decoder={spec.action_decoder_version} observation={spec.observation_version}\n"
        )
        handle.write("0 wait\n")
        for slot_index in range(spec.max_seed_slots):
            plant_type = plant_types[slot_index] if slot_index < len(plant_types) else -1
            name = seed_list[slot_index] if slot_index < len(seed_list) else "inactive"
            active = slot_index < len(seed_list)
            for row in range(rows):
                for col in range(cols):
                    action = 1 + slot_index * rows * cols + row * cols + col
                    handle.write(
                        f"{action} slot={slot_index} plant={name} plant_type={plant_type} "
                        f"row={row} col={col} active={active}\n"
                    )


def clean_episode_row(summary: Dict[str, Any], fallback_episode: int) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    for field in EPISODE_METRIC_FIELDS:
        row[field] = summary.get(field)
    row["run_mode"] = str(row.get("run_mode") or "")
    row["episode"] = int(row["episode"] if row["episode"] is not None else fallback_episode)
    row["result"] = str(row.get("result") or row.get("done_reason") or "none")
    row["reward_total"] = float(row.get("reward_total") or row.get("episode_reward") or 0.0)
    row["episode_reward"] = float(row["episode_reward"] or 0.0)
    row["episode_length"] = int(row["episode_length"] or 0)
    row["terminal_reason"] = str(row["terminal_reason"] or "")
    row["done_reason"] = str(row["done_reason"] or "none")
    row["win"] = bool(row["win"])
    row["loss"] = bool(row["loss"])
    row["timeout"] = bool(row["timeout"])
    for field in EPISODE_STRING_FIELDS:
        row[field] = str(row.get(field) or "")
    for field in (
        "final_wave",
        "max_wave",
        "zombies_killed",
        "plants_placed",
        "sunflowers_planted",
        "peashooters_planted",
        "wallnuts_planted",
        "cherrybombs_planted",
        "sun_spent",
        "sun_remaining",
        "mowers_lost",
        "mower_losses",
        "bridge_errors",
        "illegal_actions",
    ):
        row[field] = int(row[field] or 0)
    row["reset_success"] = bool(row["reset_success"]) if row["reset_success"] is not None else True
    row["reset_seconds"] = float(row["reset_seconds"] or 0.0)
    row["avg_legal_actions"] = float(row["avg_legal_actions"] or 0.0)
    for field in LANE_DIAGNOSTIC_DICT_FIELDS:
        row[field] = normalize_count_dict(row.get(field))
    for field in LANE_DIAGNOSTIC_FLOAT_DICT_FIELDS:
        row[field] = normalize_float_dict(row.get(field))
    for field in (
        "wait_under_threat_count",
        "close_zombie_undefended_count",
        "wait_actions",
        "plant_actions",
        "overdefended_while_undefended_count",
        "least_defended_threatened_row_plant_count",
        "rows_with_peashooter_count",
        "all_rows_peashooter_covered_step",
        "sunflower_count_when_first_full_coverage",
        "sunflower_overbuild_before_defense_count",
        "sunflower_overbuild_count",
        "pre_step_mask_blocked_count",
        "cooldown_illegal_exposed_by_mask_count",
        "mask_bridge_disagreement_count",
        "timeout_reset_requested_count",
        "wait_while_actionable_threat_count",
        "wait_while_peashooter_affordable_ready_count",
        "wait_while_wallnut_affordable_ready_count",
        "wait_while_cherrybomb_affordable_ready_count",
        "active_threat_rows_without_peashooter_count",
        "sunflower_greed_while_defense_missing_count",
        "wallnut_blocks_active_threat_count",
        "wallnut_low_value_placement_count",
        "wallnut_threatened_lane_placements",
        "wallnut_between_zombie_and_house_count",
        "wallnut_frontline_count",
        "wallnut_emergency_blocks",
        "wallnut_useless_placements",
        "cherrybomb_used_count",
        "cherrybomb_kills_total",
        "cherrybomb_zero_kill_count",
        "cherrybomb_zero_kill_uses",
        "cherrybomb_cluster_uses",
        "cherrybomb_emergency_uses",
        "cherrybomb_heavy_zombie_kills",
        "cherrybomb_buckethead_kills",
        "cherrybomb_conehead_kills",
        "cherrybomb_used_under_threat_count",
        "cherrybomb_used_low_value_count",
        "close_zombie_with_no_defense_count",
        "undefended_threat_steps",
        "high_danger_unanswered_steps",
        "mower_exposure_steps",
        "overdefense_count",
        "wallnut_actions_masked",
        "cherrybomb_actions_masked",
        "wallnut_actions_available",
        "cherrybomb_actions_available",
        "mask_all_but_wait_count",
        "tough_zombie_response_count",
        "fusion_candidate_count_total",
        "fusion_attempted_count",
        "fusion_success_count",
        "fusion_failed_count",
        "fusion_rejected_count",
        "fusion_under_threat_count",
        "fusion_near_buckethead_count",
        "fusion_near_conehead_count",
        "fusion_estimated_mower_save_count",
        "fusion_kills_after_use_total",
        "fusion_bridge_error_count",
        "fusion_unsafe_state_block_count",
        *PROGRESS_CSV_DIAGNOSTIC_INT_FIELDS,
    ):
        row[field] = int(row.get(field) or 0)
    for field in (
        "wait_action_percent",
        "plant_action_percent",
        "row_defense_response_rate",
        "legal_action_count_mean",
        "plants_in_threatened_row_ratio",
        "plants_in_unthreatened_row_ratio",
        "peashooter_coverage_rate_by_step",
        "wallnut_damage_absorbed_total",
        "cherrybomb_avg_kills_per_use",
        "fusion_candidate_count_avg",
        "fusion_avg_kills_after_use",
        "max_row_danger",
        "avg_row_danger",
        *PROGRESS_CSV_ACTION_DURATION_FIELDS,
        *FUSION_REWARD_FLOAT_FIELDS,
        *REWARD_EPISODE_TOTAL_FIELDS,
    ):
        row[field] = float(row.get(field) or 0.0)
    row["tactical_mask_enabled"] = bool(row.get("tactical_mask_enabled"))
    row["fusion_reward_capped"] = bool(row.get("fusion_reward_capped"))
    return row


def normalize_count_dict(value: Any) -> Dict[str, int]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    if not isinstance(value, dict):
        return {}
    normalized: Dict[str, int] = {}
    for key, raw_count in value.items():
        try:
            normalized[str(key)] = int(raw_count)
        except (TypeError, ValueError):
            continue
    return dict(sorted(normalized.items(), key=lambda item: (0, int(item[0])) if item[0].isdigit() else (1, item[0])))


def normalize_float_dict(value: Any) -> Dict[str, float]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    if not isinstance(value, dict):
        return {}
    normalized: Dict[str, float] = {}
    for key, raw_value in value.items():
        try:
            normalized[str(key)] = float(raw_value)
        except (TypeError, ValueError):
            continue
    return dict(sorted(normalized.items(), key=lambda item: (0, int(item[0])) if item[0].isdigit() else (1, item[0])))


def csv_safe_row(row: Dict[str, Any]) -> Dict[str, Any]:
    safe = dict(row)
    for field in LANE_DIAGNOSTIC_DICT_FIELDS + LANE_DIAGNOSTIC_FLOAT_DICT_FIELDS:
        safe[field] = json.dumps(safe.get(field) or {}, separators=(",", ":"), sort_keys=True)
    return safe


def extend_csv_fieldnames(fieldnames: List[str], extra_fields: List[str]) -> List[str]:
    extended = list(fieldnames)
    seen = set(extended)
    for field in extra_fields:
        if field not in seen:
            extended.append(field)
            seen.add(field)
    return extended


def filter_row_to_fieldnames(row: Dict[str, Any], fieldnames: List[str]) -> Dict[str, Any]:
    safe = csv_safe_row(row)
    return {key: safe.get(key, "") for key in fieldnames}


def ensure_progress_csv_fieldnames(csv_path: Path, fieldnames: List[str]) -> List[str]:
    schema_fieldnames = extend_csv_fieldnames(fieldnames, PROGRESS_CSV_DIAGNOSTIC_FIELDS)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return schema_fieldnames

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        loaded_fieldnames = list(reader.fieldnames or [])
        if not loaded_fieldnames:
            return schema_fieldnames
        migrated_fieldnames = extend_csv_fieldnames(
            loaded_fieldnames,
            [field for field in schema_fieldnames if field not in loaded_fieldnames],
        )
        if migrated_fieldnames == loaded_fieldnames:
            return migrated_fieldnames
        rows = list(reader)

    temp_path = csv_path.with_name(f"{csv_path.name}.tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=migrated_fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in migrated_fieldnames})
    temp_path.replace(csv_path)
    return migrated_fieldnames


def write_progress_csv_rows(
    csv_path: Path,
    rows: List[Dict[str, Any]],
    fieldnames: List[str],
    header_written: bool,
) -> tuple[bool, List[str]]:
    writer_fieldnames = ensure_progress_csv_fieldnames(csv_path, fieldnames)
    header_written = header_written or (csv_path.exists() and csv_path.stat().st_size > 0)
    with csv_path.open("a", newline="", encoding="utf-8") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=writer_fieldnames, extrasaction="ignore")
        if not header_written:
            writer.writeheader()
            header_written = True
        for row in rows:
            safe_row = filter_row_to_fieldnames(row, list(writer.fieldnames or []))
            writer.writerow(safe_row)
    return header_written, writer_fieldnames


class EpisodeMetricWriter:
    """Own normalization and the paired CSV/JSONL episode event streams."""

    def __init__(self, csv_path: Path, jsonl_path: Path) -> None:
        self.csv_path = csv_path
        self.jsonl_path = jsonl_path
        self.rows: List[Dict[str, Any]] = []
        self.csv_fieldnames = ensure_progress_csv_fieldnames(csv_path, EPISODE_METRIC_FIELDS)
        self._header_written = csv_path.exists() and csv_path.stat().st_size > 0

    def append_summaries(self, summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Preserve the callback's historical fallback: all summaries observed in
        # one vectorized callback use the same pre-append episode index.
        rows = [clean_episode_row(summary, len(self.rows)) for summary in summaries]
        if not rows:
            return []
        self._header_written, self.csv_fieldnames = write_progress_csv_rows(
            self.csv_path,
            rows,
            self.csv_fieldnames,
            self._header_written,
        )
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.jsonl_path.open("a", encoding="utf-8") as jsonl_handle:
            jsonl_handle.write("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))
        self.rows.extend(rows)
        return rows


def _dict_field(record: Any, field_name: str, normalizer: Any) -> Dict[Any, Any]:
    if isinstance(record, dict):
        return normalizer(record.get(field_name))
    value = getattr(record, field_name, {})
    return value if isinstance(value, dict) else {}


def sum_count_dicts_from_rows(rows: List[Any], field_name: str) -> Dict[str, int]:
    totals: Counter[str] = Counter()
    for row in rows:
        values = _dict_field(row, field_name, normalize_count_dict)
        for key, value in values.items():
            try:
                totals[str(key)] += int(value)
            except (TypeError, ValueError):
                continue
    return dict(sorted(totals.items(), key=lambda item: (0, int(item[0])) if item[0].isdigit() else (1, item[0])))


def max_count_dicts_from_rows(rows: List[Any], field_name: str) -> Dict[str, int]:
    totals: Counter[str] = Counter()
    for row in rows:
        values = _dict_field(row, field_name, normalize_count_dict)
        for key, value in values.items():
            try:
                totals[str(key)] = max(int(totals.get(str(key), 0)), int(value))
            except (TypeError, ValueError):
                continue
    return dict(sorted(totals.items(), key=lambda item: (0, int(item[0])) if item[0].isdigit() else (1, item[0])))


def ratio_dict(numerator: Dict[str, int], denominator: Dict[str, int]) -> Dict[str, float]:
    rows = sorted(
        set(str(key) for key in numerator.keys()) | set(str(key) for key in denominator.keys()),
        key=lambda item: (0, int(item)) if item.isdigit() else (1, item),
    )
    result: Dict[str, float] = {}
    for row in rows:
        denom = int(denominator.get(row, 0) or 0)
        result[row] = float(numerator.get(row, 0)) / float(denom) if denom > 0 else 0.0
    return result


def total_count(values: Dict[str, int]) -> int:
    return sum(int(value) for value in values.values())


def weighted_average_from_rows(rows: List[Dict[str, Any]], field_name: str, weight_field: str) -> float:
    weighted_total = 0.0
    weight_total = 0.0
    for row in rows:
        try:
            value = float(row.get(field_name, 0.0) or 0.0)
            weight = float(row.get(weight_field, 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if weight <= 0.0:
            continue
        weighted_total += value * weight
        weight_total += weight
    return weighted_total / weight_total if weight_total > 0.0 else 0.0


def weighted_average_dict_from_rows(rows: List[Any], field_name: str, weight_field: str) -> Dict[str, float]:
    totals: Counter[str] = Counter()
    weights: Counter[str] = Counter()
    for row in rows:
        values = _dict_field(row, field_name, normalize_float_dict)
        row_weights = _dict_field(row, weight_field, normalize_count_dict)
        for key, raw_value in values.items():
            try:
                value = float(raw_value)
                weight = int(row_weights.get(str(key), 0) or 0)
            except (TypeError, ValueError):
                continue
            if weight <= 0:
                continue
            totals[str(key)] += value * weight
            weights[str(key)] += weight
    keys = sorted(set(totals.keys()) | set(weights.keys()), key=lambda item: (0, int(item)) if item.isdigit() else (1, item))
    return {
        key: float(totals.get(key, 0.0)) / float(weights.get(key, 0))
        if weights.get(key, 0) > 0
        else 0.0
        for key in keys
    }


def average_positive_from_rows(rows: List[Any], field_name: str) -> float:
    values: List[float] = []
    for row in rows:
        try:
            raw_value = row.get(field_name, 0.0) if isinstance(row, dict) else getattr(row, field_name, 0.0)
            value = float(raw_value or 0.0)
        except (TypeError, ValueError):
            continue
        if value > 0.0:
            values.append(value)
    return sum(values) / len(values) if values else 0.0


def average_positive_step_dict_from_rows(rows: List[Any], field_name: str) -> Dict[str, float]:
    totals: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    for row in rows:
        values = _dict_field(row, field_name, normalize_count_dict)
        for key, raw_value in values.items():
            try:
                value = int(raw_value)
            except (TypeError, ValueError):
                continue
            if int(value) > 0:
                totals[str(key)] += value
                counts[str(key)] += 1
    keys = sorted(set(totals.keys()) | set(counts.keys()), key=lambda item: (0, int(item)) if item.isdigit() else (1, item))
    return {
        key: float(totals.get(key, 0)) / float(counts.get(key, 0))
        if counts.get(key, 0) > 0
        else 0.0
        for key in keys
    }


class PerformanceAccumulator:
    def __init__(self) -> None:
        self.samples = 0
        self.totals: Dict[str, float] = {}
        self.restart_modes: Counter[str] = Counter()

    def add(self, perf: Dict[str, Any]) -> None:
        self.samples += 1
        for key, value in perf.items():
            if isinstance(value, (int, float)):
                self.totals[key] = self.totals.get(key, 0.0) + float(value)
        mode = str(perf.get("restart_detection_mode") or "unknown")
        self.restart_modes[mode] += 1

    def summary(self, fps_avg: float) -> Dict[str, Any]:
        averages = {key: value / max(1, self.samples) for key, value in sorted(self.totals.items())}
        return {
            "samples": self.samples,
            "fps_avg": fps_avg,
            "averages": averages,
            "restart_detection_modes": dict(self.restart_modes),
        }


def summarize_episode_rows(
    rows: List[Dict[str, Any]],
    total_timesteps: int,
    run_dir: Path,
    model_path: Path,
    fps_avg: float,
) -> Dict[str, Any]:
    count = max(1, len(rows))
    threat_steps = sum_count_dicts_from_rows(rows, "threat_steps_by_row")
    undefended_steps = sum_count_dicts_from_rows(rows, "undefended_threat_steps_by_row")
    threatened_zero_defender_steps = sum_count_dicts_from_rows(rows, "threatened_rows_with_zero_defender_steps_by_row")
    row_defense_opportunities = sum_count_dicts_from_rows(rows, "row_defense_opportunities_by_row")
    row_defense_responses = sum_count_dicts_from_rows(rows, "row_defense_responses_by_row")
    peashooter_placements = sum_count_dicts_from_rows(rows, "peashooter_placements_by_row")
    full_coverage_sunflower_counts = []
    for row in rows:
        try:
            covered_step = int(row.get("all_rows_peashooter_covered_step", 0) or 0)
            sunflower_count = int(row.get("sunflower_count_when_first_full_coverage", -1))
        except (TypeError, ValueError):
            continue
        if covered_step > 0 and sunflower_count >= 0:
            full_coverage_sunflower_counts.append(sunflower_count)
    reward_component_totals = {
        field: sum(float(row.get(field, 0.0) or 0.0) for row in rows)
        for field in REWARD_EPISODE_TOTAL_FIELDS
    }
    summary = {
        "total_timesteps": total_timesteps,
        "episodes": len(rows),
        "avg_reward": sum(float(row.get("episode_reward", 0.0)) for row in rows) / count,
        "avg_length": sum(int(row.get("episode_length", 0)) for row in rows) / count,
        "win_rate": sum(1 for row in rows if row.get("win")) / count,
        "avg_wave": sum(int(row.get("final_wave", 0)) for row in rows) / count,
        "avg_kills": sum(int(row.get("zombies_killed", 0)) for row in rows) / count,
        "avg_plants": sum(int(row.get("plants_placed", 0)) for row in rows) / count,
        "sunflowers_planted": sum(int(row.get("sunflowers_planted", 0) or 0) for row in rows),
        "peashooters_planted": sum(int(row.get("peashooters_planted", 0) or 0) for row in rows),
        "wallnuts_planted": sum(int(row.get("wallnuts_planted", 0) or 0) for row in rows),
        "cherrybombs_planted": sum(int(row.get("cherrybombs_planted", 0) or 0) for row in rows),
        "avg_mowers_lost": sum(int(row.get("mowers_lost", 0)) for row in rows) / count,
        "reset_failures": sum(1 for row in rows if not row.get("reset_success", True)),
        "bridge_errors": sum(int(row.get("bridge_errors", 0)) for row in rows),
        "fps_avg": fps_avg,
        "plants_by_row": sum_count_dicts_from_rows(rows, "plants_by_row"),
        "peashooters_by_row": sum_count_dicts_from_rows(rows, "peashooters_by_row"),
        "sunflowers_by_row": sum_count_dicts_from_rows(rows, "sunflowers_by_row"),
        "threat_steps_by_row": threat_steps,
        "undefended_threat_steps_by_row": undefended_steps,
        "undefended_threat_ratio_by_row": ratio_dict(undefended_steps, threat_steps),
        "undefended_threat_age_avg_by_row": weighted_average_dict_from_rows(
            rows,
            "undefended_threat_age_avg_by_row",
            "threatened_rows_with_zero_defender_steps_by_row",
        ),
        "undefended_threat_age_max_by_row": max_count_dicts_from_rows(rows, "undefended_threat_age_max_by_row"),
        "threatened_rows_with_zero_defender_steps_by_row": threatened_zero_defender_steps,
        "peashooters_per_threat_step_by_row": ratio_dict(peashooter_placements, threat_steps),
        "first_defense_step_by_row": average_positive_step_dict_from_rows(rows, "first_defense_step_by_row"),
        "plants_in_threatened_row_ratio": weighted_average_from_rows(rows, "plants_in_threatened_row_ratio", "plants_placed"),
        "plants_in_unthreatened_row_ratio": weighted_average_from_rows(rows, "plants_in_unthreatened_row_ratio", "plants_placed"),
        "overdefended_while_undefended_count": sum(int(row.get("overdefended_while_undefended_count", 0) or 0) for row in rows),
        "least_defended_threatened_row_plant_count": sum(int(row.get("least_defended_threatened_row_plant_count", 0) or 0) for row in rows),
        "rows_with_peashooter_count": sum(int(row.get("rows_with_peashooter_count", 0) or 0) for row in rows) / count,
        "all_rows_peashooter_covered_step": average_positive_from_rows(rows, "all_rows_peashooter_covered_step"),
        "first_peashooter_by_row_step": average_positive_step_dict_from_rows(rows, "first_peashooter_by_row_step"),
        "sunflower_count_when_first_full_coverage": (
            sum(full_coverage_sunflower_counts) / len(full_coverage_sunflower_counts)
            if full_coverage_sunflower_counts
            else -1.0
        ),
        "sunflower_overbuild_before_defense_count": sum(int(row.get("sunflower_overbuild_before_defense_count", 0) or 0) for row in rows),
        "sunflower_overbuild_count": sum(int(row.get("sunflower_overbuild_count", 0) or 0) for row in rows),
        "peashooter_coverage_rate_by_step": weighted_average_from_rows(rows, "peashooter_coverage_rate_by_step", "episode_length"),
        "mower_losses_by_row": sum_count_dicts_from_rows(rows, "mower_losses_by_row"),
        "wait_under_threat_count": sum(int(row.get("wait_under_threat_count", 0) or 0) for row in rows),
        "close_zombie_undefended_count": sum(int(row.get("close_zombie_undefended_count", 0) or 0) for row in rows),
        "illegal_reason_counts": sum_count_dicts_from_rows(rows, "illegal_reason_counts"),
        "legal_peashooter_actions_by_row": sum_count_dicts_from_rows(rows, "legal_peashooter_actions_by_row"),
        "peashooter_available_but_waited_by_row": sum_count_dicts_from_rows(rows, "peashooter_available_but_waited_by_row"),
        "peashooter_available_but_planted_elsewhere_by_row": sum_count_dicts_from_rows(
            rows,
            "peashooter_available_but_planted_elsewhere_by_row",
        ),
        "sunflower_while_undefended_threat_by_row": sum_count_dicts_from_rows(rows, "sunflower_while_undefended_threat_by_row"),
        "legal_actions_by_seed_slot": sum_count_dicts_from_rows(rows, "legal_actions_by_seed_slot"),
        "bridge_legal_actions_by_seed_slot": sum_count_dicts_from_rows(rows, "bridge_legal_actions_by_seed_slot"),
        "python_mask_block_reason_counts": sum_count_dicts_from_rows(rows, "python_mask_block_reason_counts"),
        "pre_step_mask_blocked_count": sum(int(row.get("pre_step_mask_blocked_count", 0) or 0) for row in rows),
        "cooldown_illegal_exposed_by_mask_count": sum(int(row.get("cooldown_illegal_exposed_by_mask_count", 0) or 0) for row in rows),
        "mask_bridge_disagreement_count": sum(int(row.get("mask_bridge_disagreement_count", 0) or 0) for row in rows),
        "wait_while_actionable_threat_count": sum(int(row.get("wait_while_actionable_threat_count", 0) or 0) for row in rows),
        "active_threat_rows_without_peashooter_count": sum(int(row.get("active_threat_rows_without_peashooter_count", 0) or 0) for row in rows),
        "sunflower_greed_while_defense_missing_count": sum(int(row.get("sunflower_greed_while_defense_missing_count", 0) or 0) for row in rows),
        "wallnut_placements_by_row": sum_count_dicts_from_rows(rows, "wallnut_placements_by_row"),
        "wallnut_placements_by_col": sum_count_dicts_from_rows(rows, "wallnut_placements_by_col"),
        "wallnut_blocks_active_threat_count": sum(int(row.get("wallnut_blocks_active_threat_count", 0) or 0) for row in rows),
        "wallnut_low_value_placement_count": sum(int(row.get("wallnut_low_value_placement_count", 0) or 0) for row in rows),
        "wallnut_threatened_lane_placements": sum(int(row.get("wallnut_threatened_lane_placements", 0) or 0) for row in rows),
        "wallnut_between_zombie_and_house_count": sum(int(row.get("wallnut_between_zombie_and_house_count", 0) or 0) for row in rows),
        "wallnut_frontline_count": sum(int(row.get("wallnut_frontline_count", 0) or 0) for row in rows),
        "wallnut_emergency_blocks": sum(int(row.get("wallnut_emergency_blocks", 0) or 0) for row in rows),
        "wallnut_useless_placements": sum(int(row.get("wallnut_useless_placements", 0) or 0) for row in rows),
        "cherrybomb_used_count": sum(int(row.get("cherrybomb_used_count", 0) or 0) for row in rows),
        "cherrybomb_kills_total": sum(int(row.get("cherrybomb_kills_total", 0) or 0) for row in rows),
        "cherrybomb_avg_kills_per_use": (
            sum(int(row.get("cherrybomb_kills_total", 0) or 0) for row in rows)
            / max(1, sum(int(row.get("cherrybomb_used_count", 0) or 0) for row in rows))
        ),
        "cherrybomb_zero_kill_count": sum(int(row.get("cherrybomb_zero_kill_count", 0) or 0) for row in rows),
        "cherrybomb_zero_kill_uses": sum(int(row.get("cherrybomb_zero_kill_uses", 0) or 0) for row in rows),
        "cherrybomb_cluster_uses": sum(int(row.get("cherrybomb_cluster_uses", 0) or 0) for row in rows),
        "cherrybomb_emergency_uses": sum(int(row.get("cherrybomb_emergency_uses", 0) or 0) for row in rows),
        "cherrybomb_heavy_zombie_kills": sum(int(row.get("cherrybomb_heavy_zombie_kills", 0) or 0) for row in rows),
        "mower_risk_steps_by_row": sum_count_dicts_from_rows(rows, "mower_risk_steps_by_row"),
        "mower_saves_estimated_by_row": sum_count_dicts_from_rows(rows, "mower_saves_estimated_by_row"),
        "buckethead_count_by_row": sum_count_dicts_from_rows(rows, "buckethead_count_by_row"),
        "conehead_count_by_row": sum_count_dicts_from_rows(rows, "conehead_count_by_row"),
        "tough_zombie_count_by_row": sum_count_dicts_from_rows(rows, "tough_zombie_count_by_row"),
        "tough_zombie_response_count": sum(int(row.get("tough_zombie_response_count", 0) or 0) for row in rows),
        "undefended_threat_steps": sum(int(row.get("undefended_threat_steps", 0) or 0) for row in rows),
        "high_danger_unanswered_steps": sum(int(row.get("high_danger_unanswered_steps", 0) or 0) for row in rows),
        "mower_exposure_steps": sum(int(row.get("mower_exposure_steps", 0) or 0) for row in rows),
        "overdefense_count": sum(int(row.get("overdefense_count", 0) or 0) for row in rows),
        "mower_losses": sum(int(row.get("mower_losses", 0) or 0) for row in rows),
        "legal_action_count_mean": weighted_average_from_rows(rows, "legal_action_count_mean", "episode_length"),
        "max_row_danger": max((float(row.get("max_row_danger", 0.0) or 0.0) for row in rows), default=0.0),
        "avg_row_danger": weighted_average_from_rows(rows, "avg_row_danger", "episode_length"),
        "tactical_mask_enabled": any(bool(row.get("tactical_mask_enabled")) for row in rows),
        "wallnut_actions_masked": sum(int(row.get("wallnut_actions_masked", 0) or 0) for row in rows),
        "cherrybomb_actions_masked": sum(int(row.get("cherrybomb_actions_masked", 0) or 0) for row in rows),
        "wallnut_actions_available": sum(int(row.get("wallnut_actions_available", 0) or 0) for row in rows),
        "cherrybomb_actions_available": sum(int(row.get("cherrybomb_actions_available", 0) or 0) for row in rows),
        "mask_all_but_wait_count": sum(int(row.get("mask_all_but_wait_count", 0) or 0) for row in rows),
        "fusion_policy": str(rows[-1].get("fusion_policy") or FUSION_POLICY_NONE) if rows else FUSION_POLICY_NONE,
        "fusion_candidate_count_total": sum(int(row.get("fusion_candidate_count_total", 0) or 0) for row in rows),
        "fusion_candidate_count_avg": weighted_average_from_rows(rows, "fusion_candidate_count_avg", "episode_length"),
        "fusion_attempted_count": sum(int(row.get("fusion_attempted_count", 0) or 0) for row in rows),
        "fusion_success_count": sum(int(row.get("fusion_success_count", 0) or 0) for row in rows),
        "fusion_failed_count": sum(int(row.get("fusion_failed_count", 0) or 0) for row in rows),
        "fusion_rejected_count": sum(int(row.get("fusion_rejected_count", 0) or 0) for row in rows),
        "fusion_rejected_reasons": sum_count_dicts_from_rows(rows, "fusion_rejected_reasons"),
        "fusion_by_result_type": sum_count_dicts_from_rows(rows, "fusion_by_result_type"),
        "fusion_by_source_type": sum_count_dicts_from_rows(rows, "fusion_by_source_type"),
        "fusion_by_row": sum_count_dicts_from_rows(rows, "fusion_by_row"),
        "fusion_under_threat_count": sum(int(row.get("fusion_under_threat_count", 0) or 0) for row in rows),
        "fusion_near_buckethead_count": sum(int(row.get("fusion_near_buckethead_count", 0) or 0) for row in rows),
        "fusion_near_conehead_count": sum(int(row.get("fusion_near_conehead_count", 0) or 0) for row in rows),
        "fusion_estimated_mower_save_count": sum(int(row.get("fusion_estimated_mower_save_count", 0) or 0) for row in rows),
        "fusion_kills_after_use_total": sum(int(row.get("fusion_kills_after_use_total", 0) or 0) for row in rows),
        "fusion_avg_kills_after_use": weighted_average_from_rows(rows, "fusion_avg_kills_after_use", "fusion_success_count"),
        "fusion_bridge_error_count": sum(int(row.get("fusion_bridge_error_count", 0) or 0) for row in rows),
        "fusion_unsafe_state_block_count": sum(int(row.get("fusion_unsafe_state_block_count", 0) or 0) for row in rows),
        "row_defense_opportunities_by_row": row_defense_opportunities,
        "row_defense_responses_by_row": row_defense_responses,
        "row_defense_response_rate_by_row": ratio_dict(row_defense_responses, row_defense_opportunities),
        "row_defense_response_rate": (
            total_count(row_defense_responses) / total_count(row_defense_opportunities)
            if total_count(row_defense_opportunities) > 0
            else 0.0
        ),
        "reward_component_totals": reward_component_totals,
        "reward_component_avgs": {field: value / count for field, value in reward_component_totals.items()},
        "model_path": str(model_path),
        "run_dir": str(run_dir),
    }
    summary.update({
        field: sum(float(row.get(field, 0.0) or 0.0) for row in rows)
        for field in FUSION_REWARD_FLOAT_FIELDS[:-2]
    })
    summary["fusion_reward_total"] = float(reward_component_totals.get("fusion_reward_total", 0.0))
    summary["fusion_reward_capped"] = any(bool(row.get("fusion_reward_capped")) for row in rows)
    if rows:
        summary["fusion_last_reward_delta"] = float(rows[-1].get("fusion_last_reward_delta", 0.0) or 0.0)
        summary["fusion_last_usefulness_bonus"] = float(
            rows[-1].get("fusion_last_usefulness_bonus", 0.0) or 0.0
        )
        summary["fusion_last_reward_reason"] = str(rows[-1].get("fusion_last_reward_reason") or "")
        summary["fusion_last_source"] = str(rows[-1].get("fusion_last_source") or "")
    return summary


def print_validation_summary(summary: Dict[str, Any]) -> None:
    print("PPO Validation Summary")
    print("----------------------")
    for key in (
        "total_timesteps",
        "episodes",
        "avg_reward",
        "avg_length",
        "win_rate",
        "avg_wave",
        "avg_kills",
        "avg_plants",
        "avg_mowers_lost",
        "reset_failures",
        "bridge_errors",
        "peashooters_by_row",
        "threat_steps_by_row",
        "undefended_threat_ratio_by_row",
        "undefended_threat_age_avg_by_row",
        "undefended_threat_age_max_by_row",
        "threatened_rows_with_zero_defender_steps_by_row",
        "peashooters_per_threat_step_by_row",
        "first_defense_step_by_row",
        "plants_in_threatened_row_ratio",
        "plants_in_unthreatened_row_ratio",
        "overdefended_while_undefended_count",
        "least_defended_threatened_row_plant_count",
        "rows_with_peashooter_count",
        "all_rows_peashooter_covered_step",
        "first_peashooter_by_row_step",
        "sunflower_count_when_first_full_coverage",
        "sunflower_overbuild_before_defense_count",
        "peashooter_coverage_rate_by_step",
        "mower_losses_by_row",
        "row_defense_response_rate",
        "row_defense_response_rate_by_row",
        "illegal_reason_counts",
        "legal_actions_by_seed_slot",
        "peashooter_available_but_waited_by_row",
        "peashooter_available_but_planted_elsewhere_by_row",
        "sunflower_while_undefended_threat_by_row",
        "bridge_legal_actions_by_seed_slot",
        "python_mask_block_reason_counts",
        "pre_step_mask_blocked_count",
        "cooldown_illegal_exposed_by_mask_count",
        "mask_bridge_disagreement_count",
        "reward_component_avgs",
        "fps_avg",
        "model_path",
        "run_dir",
    ):
        value = summary.get(key)
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        elif isinstance(value, dict):
            print(f"{key}: {json.dumps(value, separators=(',', ':'), sort_keys=True)}")
        else:
            print(f"{key}: {value}")


def resolved_live_status_path(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (Path.cwd() / expanded).resolve()


def unwrap_pvz_env(env: Optional[Any]) -> Optional[PvZMaskedPPOEnv]:
    current = env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, PvZMaskedPPOEnv):
            return current
        nested = getattr(current, "env", None)
        if nested is current:
            break
        current = nested
    return None


def live_status_rows_from_observation(observation: Dict[str, Any], summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    def row_value(mapping: Any, row_id: str) -> Any:
        if not isinstance(mapping, dict):
            return None
        if row_id in mapping:
            return mapping[row_id]
        try:
            numeric = int(row_id)
        except (TypeError, ValueError):
            return None
        return mapping.get(numeric)

    rows: Dict[str, Dict[str, Any]] = {}
    lanes = observation.get("lanes", []) if isinstance(observation, dict) else []
    if isinstance(lanes, list):
        for index, lane in enumerate(lanes):
            if not isinstance(lane, dict):
                continue
            row = lane.get("row", index)
            row_id = str(row)
            zombie_count = lane.get("zombieCount", lane.get("zombies", 0))
            try:
                threatened = int(zombie_count or 0) > 0
            except (TypeError, ValueError):
                threatened = bool(zombie_count)
            rows[row_id] = {
                "row": row,
                "zombies": zombie_count,
                "danger": lane.get("danger"),
                "peashooters": lane.get("peashooterCount", lane.get("peashooters")),
                "threatened": threatened,
                "undefended_threat": lane.get("undefendedThreat", lane.get("undefended_threat", False)),
                "threat_steps": 0,
            }
    peashooters = summary.get("peashooters_by_row")
    threat_steps = summary.get("threat_steps_by_row")
    undefended = summary.get("undefended_threat_steps_by_row")
    for source in (peashooters, threat_steps, undefended):
        if not isinstance(source, dict):
            continue
        for row_id in source.keys():
            rows.setdefault(str(row_id), {"row": row_id})
    for row_id, payload in rows.items():
        peashooter_count = row_value(peashooters, str(row_id))
        row_threat_steps = row_value(threat_steps, str(row_id))
        row_undefended = row_value(undefended, str(row_id))
        if peashooter_count is not None:
            payload["peashooters"] = peashooter_count
        if row_threat_steps is not None:
            payload["threat_steps"] = row_threat_steps
            payload["threatened"] = bool(row_threat_steps)
        if row_undefended is not None:
            payload["undefended_threat"] = bool(row_undefended)
    return rows


def coach_live_status_fields_from_summary(config: Dict[str, Any], summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    summary = summary or {}
    stream_summary = summary.get("stream_coach") if isinstance(summary, dict) else None
    coach_summary = summary.get("coach") if isinstance(summary, dict) else None
    human_summary = summary.get("human_coach") if isinstance(summary, dict) else None
    selected_summary = None
    for candidate in (coach_summary, stream_summary, human_summary):
        if isinstance(candidate, dict) and candidate:
            selected_summary = candidate
            break
    fields = (
        dict(selected_summary)
        if isinstance(selected_summary, dict)
        else human_coach_live_status_defaults(
            enabled=bool(config.get("human_coach_enabled", False)),
            platform=str(config.get("human_coach_platform", "mock") or "mock"),
        )
    )
    stream_enabled = bool(config.get("stream_coach_enabled", False))
    dry_run = bool(config.get("stream_coach_dry_run", True))
    apply_enabled = bool(config.get("stream_coach_apply_enabled", False)) and not dry_run
    fields.setdefault(
        "stream_coach_mode",
        str(config.get("stream_coach_mode", config.get("stream_coach_platform", "mock")) or "mock") if stream_enabled else "off",
    )
    fields.setdefault("stream_coach_enabled", stream_enabled)
    fields.setdefault("stream_coach_platform", str(config.get("stream_coach_platform", "mock") or "mock"))
    fields.setdefault("stream_coach_alive", bool(stream_enabled))
    fields.setdefault("stream_coach_alive_status", "alive" if stream_enabled else "off")
    fields.setdefault("stream_coach_dry_run", bool(dry_run))
    fields.setdefault("stream_coach_apply_enabled", bool(apply_enabled))
    fields.setdefault("stream_coach_command_path", str(config.get("stream_coach_command_path", "") or ""))
    fields.setdefault("mock_stream_script", str(config.get("stream_coach_mock_script", "") or ""))
    fields.setdefault("stream_coach_messages_seen", 0)
    fields.setdefault("stream_coach_commands_parsed", 0)
    fields.setdefault("stream_coach_commands_accepted", 0)
    fields.setdefault("stream_coach_commands_rejected", 0)
    fields.setdefault("stream_messages_seen", int(fields.get("stream_coach_messages_seen", 0) or 0))
    fields.setdefault("stream_commands_parsed", int(fields.get("stream_coach_commands_parsed", 0) or 0))
    fields.setdefault("stream_commands_accepted", int(fields.get("stream_coach_commands_accepted", 0) or 0))
    fields.setdefault("stream_commands_rejected", int(fields.get("stream_coach_commands_rejected", 0) or 0))
    fields.setdefault("stream_coach_validated_count", 0)
    fields.setdefault("stream_coach_applied_count", 0)
    fields.setdefault("stream_coach_dry_run_count", 0)
    fields.setdefault("stream_coach_last_user", "")
    fields.setdefault("last_stream_user", str(fields.get("stream_coach_last_user") or ""))
    fields.setdefault("stream_coach_last_message", "")
    fields.setdefault("last_stream_message", str(fields.get("stream_coach_last_message") or ""))
    fields.setdefault("stream_coach_last_parsed_command", None)
    fields.setdefault("last_stream_parsed_command", fields.get("stream_coach_last_parsed_command"))
    fields.setdefault("stream_coach_last_command_status", "idle" if stream_enabled else "off")
    fields.setdefault("last_stream_command_status", str(fields.get("stream_coach_last_command_status") or ""))
    fields.setdefault("stream_coach_last_reject_reason", "")
    fields.setdefault("last_stream_reject_reason", str(fields.get("stream_coach_last_reject_reason") or ""))
    fields.setdefault("stream_coach_last_validated_command", "")
    fields.setdefault("last_validated_coach_command", str(fields.get("stream_coach_last_validated_command") or ""))
    fields.setdefault("stream_coach_last_applied_command", "")
    fields.setdefault("last_applied_coach_command", str(fields.get("stream_coach_last_applied_command") or ""))
    fields.setdefault("pending_stream_commands", 0)
    fields.setdefault("stream_coach_startup_stale_cleared", False)
    fields.setdefault("stream_coach_stale_messages_cleared", 0)
    fields.setdefault("stream_coach_clear_count", 0)
    fields.setdefault("stream_coach_last_clear_reason", "")
    return fields


def streamer_live_status_fields(
    config: Dict[str, Any],
    summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return stable overlay-ready fields without copying observations."""

    if not bool(config.get("streamer_v1_enabled", False)):
        return {}
    values = summary if isinstance(summary, dict) else {}
    runtime = values.get("streamer_runtime")
    runtime_fields = dict(runtime) if isinstance(runtime, dict) else {}
    phase = str(config.get("streamer_phase") or runtime_fields.get("streamer_mode") or STREAM_TRAIN)
    current_steps = int(
        values.get(
            "model_steps",
            values.get(
                "total_timesteps",
                runtime_fields.get(
                    "current_model_ppo_steps",
                    config.get("streamer_current_model_steps", 0),
                ),
            ),
        )
        or 0
    )
    baseline_steps = int(config.get("streamer_baseline_model_steps", 0) or 0)
    target = int(config.get("streamer_policy_steps_per_cycle", 0) or 0)
    cycle_completed = int(
        runtime_fields.get(
            "cycle_policy_steps_completed",
            values.get("ppo_policy_timesteps", 0),
        )
        or 0
    )
    fields = {
        "streamer_v1_enabled": True,
        "streamer_mode": phase,
        "streamer_cycle": int(config.get("streamer_cycle", 0) or 0),
        "streamer_platform": str(config.get("streamer_platform", "twitch") or "twitch"),
        "current_model_ppo_steps": current_steps,
        "baseline_model_ppo_steps": baseline_steps,
        "cycle_policy_steps_completed": cycle_completed,
        "next_evaluation_countdown": max(0, target - cycle_completed),
        "evaluation_chat_control": False,
        "bc_updates_enabled": bool(
            phase == STREAM_TRAIN and config.get("streamer_bc_enabled", True)
        ),
        "bc_demonstration_count": int(
            values.get(
                "bc_demonstration_count",
                runtime_fields.get("bc_demonstration_count", 0),
            )
            or 0
        ),
        "bc_loss": float(values.get("bc_loss", runtime_fields.get("bc_loss", 0.0)) or 0.0),
        "bc_update_count": int(
            values.get("bc_update_count", runtime_fields.get("bc_update_count", 0)) or 0
        ),
        "baseline_evaluation": config.get("streamer_baseline_evaluation", {}),
        "current_evaluation": config.get("streamer_current_evaluation", {}),
        "best_evaluation": config.get("streamer_best_evaluation", {}),
        "best_model_steps": int(config.get("streamer_best_model_steps", 0) or 0),
    }
    fields.update(runtime_fields)
    return fields


def write_eval_live_status(
    live_status_path: Optional[Path],
    *,
    config: Dict[str, Any],
    model_path: Path,
    report: CompatibilityCheck,
    status: str,
    mode: str = ADVENTURE_GENERALIST_RUN_MODE_EVAL,
    summary: Optional[Dict[str, Any]] = None,
    status_writer: Optional[LiveStatusWriter] = None,
) -> None:
    if live_status_path is None:
        return
    live_status_path = resolved_live_status_path(live_status_path)
    if live_status_path is None:
        return
    compatibility = model_compatibility_live_status(report)
    fusion_fields = fusion_live_fields(
        (summary or {}).get("fusion") if isinstance(summary, dict) else None,
        str(config.get("fusion_policy", FUSION_POLICY_NONE)),
    )
    fusion_fields["fusion_action_mask_enabled"] = bool(config.get("fusion_action_mask_enabled", False))
    if isinstance(summary, dict):
        for key in (
            "fusion_policy",
            "fusion_candidate_count_total",
            "fusion_candidate_count_avg",
            "fusion_attempted_count",
            "fusion_success_count",
            "fusion_failed_count",
            "fusion_rejected_count",
            "fusion_rejected_reasons",
            "fusion_reward_total",
            *FUSION_REWARD_FLOAT_FIELDS,
            "fusion_reward_capped",
            "fusion_last_reward_reason",
            "fusion_last_source",
        ):
            if key in summary:
                live_key = "fusion_candidate_count" if key == "fusion_candidate_count_total" else key
                fusion_fields[live_key] = summary[key]
    coach_fields = coach_live_status_fields_from_summary(config, summary)
    streamer_fields = streamer_live_status_fields(config, summary)
    payload = {
        "mode": mode,
        "run_mode": str(config.get("run_mode", mode)),
        "status": status,
        "health": "LIVE" if status == "running" else ("DEAD" if status in {"complete", "blocked"} else status.upper()),
        "updated_at": time.time(),
        "model_path": str(model_path),
        "active_run": str(config.get("run_dir", "")),
        "model_family": compatibility.get("model_family") or config.get("model_family", ""),
        "seed_list": list(config.get("seed_list", [])),
        "plant_types": list(config.get("plant_types", [])),
        "action_count": action_count_for_config(config),
        "tactical_mask_enabled": bool(config.get("tactical_masks") or config.get("wallnut_tactical_mask") or config.get("cherrybomb_tactical_mask")),
        "fusion_action_mask_enabled": bool(config.get("fusion_action_mask_enabled", False)),
        "recent_win_rate": (summary or {}).get("win_rate") if isinstance(summary, dict) else None,
        "recent_avg_wave": (summary or {}).get("avg_wave") if isinstance(summary, dict) else None,
        "recent_avg_kills": (summary or {}).get("avg_kills") if isinstance(summary, dict) else None,
        "model_compatibility": compatibility,
        # Backward-compatible dashboard field.
        "compatibility": compatibility,
        "eval": summary or {},
        "fusion": dict(fusion_fields),
        "coach": dict(coach_fields),
        "stream_coach": dict(coach_fields),
        "human_coach": dict(coach_fields),
        **fusion_fields,
        **coach_fields,
        **streamer_fields,
    }
    writer = status_writer or LiveStatusWriter(live_status_path, min_interval_seconds=0.0)
    writer.write(payload, force=status != "running" or status_writer is None)


def build_runtime_live_status_payload(
    *,
    config: Dict[str, Any],
    status: str,
    mode: str,
    model_path: Optional[Path] = None,
    summary: Optional[Dict[str, Any]] = None,
    observation: Optional[Dict[str, Any]] = None,
    blocked_reason: str = "",
) -> Dict[str, Any]:
    summary = summary or {}
    observation = observation or {}
    row_danger = {}
    for lane in observation.get("lanes", []) or []:
        if not isinstance(lane, dict):
            continue
        row = lane.get("row")
        if row is None:
            continue
        row_danger[str(row)] = lane.get("danger")
    current_episode = summary.get("episode")
    current_step = summary.get("episode_length", summary.get("current_step"))
    current_timestep = summary.get("total_timesteps", summary.get("current_timestep"))
    current_wave = observation.get("wave", summary.get("final_wave"))
    max_wave = observation.get("maxWave", summary.get("max_wave"))
    current_sun = observation.get("sun", summary.get("sun_remaining"))
    current_plants = observation.get("plantCount", observation.get("visiblePlantObjectCount", summary.get("plants_placed")))
    current_zombies = observation.get("zombieCount", observation.get("logicalZombieCount"))
    legal_actions = observation.get("legalActions", [])
    legal_action_count = observation.get("legalActionCount")
    if legal_action_count is None and isinstance(legal_actions, list):
        legal_action_count = len(legal_actions)
    rows = live_status_rows_from_observation(observation, summary)
    gameplay = {
        "sun": current_sun,
        "wave": current_wave,
        "max_wave": max_wave,
        "plants": current_plants,
        "zombies": current_zombies,
        "mowers_lost": summary.get("mowers_lost"),
        "ready": observation.get("gameplayReady"),
        "gameplay_ready": observation.get("gameplayReady"),
        "screen": observation.get("screenState") or observation.get("terminalHint"),
        "screen_state": observation.get("screenState"),
    }
    agent = {
        "episode": current_episode,
        "episode_step": current_step,
        "total_timesteps": current_timestep,
        "legal_action_count": legal_action_count,
        "action_decoder_version": str(config.get("action_decoder_version", "")),
        "observation_version": str(config.get("observation_version", "")),
    }
    reward = {
        "episode": summary.get("episode_reward"),
        "episode_reward": summary.get("episode_reward"),
        "reward_total": summary.get("reward_total", summary.get("episode_reward")),
    }
    for key in REWARD_EPISODE_TOTAL_FIELDS:
        if key in summary:
            reward[key] = summary[key]
    coach_fields = coach_live_status_fields_from_summary(config, summary)
    streamer_fields = streamer_live_status_fields(config, summary)
    payload = {
        "mode": mode,
        "run_mode": str(config.get("run_mode", mode)),
        "status": status,
        "health": "LIVE" if status == "running" else ("DEAD" if status in {"complete", "blocked"} else status.upper()),
        "updated_at": time.time(),
        "blocked_reason": blocked_reason,
        "active_run": str(config.get("run_dir", "")),
        "model_path": str(model_path or config.get("model_path", "")),
        "current_timestep": current_timestep,
        "total_timesteps": current_timestep,
        "target_timesteps": int(config.get("total_timesteps", 0) or 0),
        "seed_list": list(config.get("seed_list", [])),
        "plant_types": list(config.get("plant_types", [])),
        "action_count": action_count_for_config(config),
        "tactical_mask_enabled": bool(config.get("tactical_masks") or config.get("wallnut_tactical_mask") or config.get("cherrybomb_tactical_mask")),
        "fusion_action_mask_enabled": bool(config.get("fusion_action_mask_enabled", False)),
        "current_episode": current_episode,
        "current_step": current_step,
        "current_wave": current_wave,
        "max_wave": max_wave,
        "current_reward": summary.get("episode_reward"),
        "recent_win_rate": summary.get("win_rate"),
        "recent_avg_wave": summary.get("avg_wave"),
        "recent_avg_kills": summary.get("avg_kills"),
        "current_sun": current_sun,
        "current_plants": current_plants,
        "current_zombies": current_zombies,
        "sun": current_sun,
        "wave": current_wave,
        "maxWave": max_wave,
        "plantCount": current_plants,
        "zombieCount": current_zombies,
        "gameplayReady": observation.get("gameplayReady"),
        "screenState": observation.get("screenState"),
        "terminalHint": observation.get("terminalHint"),
        "legal_action_count": legal_action_count,
        "gameplay": gameplay,
        "agent": agent,
        "reward": reward,
        "train": {
            "current_episode": current_episode,
            "current_step": current_step,
            "total_timesteps": current_timestep,
            "target_timesteps": int(config.get("total_timesteps", 0) or 0),
            "run_dir": str(config.get("run_dir", "")),
        },
        "rows": rows,
        "plants_by_type": {
            "sunflower": summary.get("sunflowers_planted"),
            "peashooter": summary.get("peashooters_planted"),
            "wallnut": summary.get("wallnuts_planted"),
            "cherrybomb": summary.get("cherrybombs_planted"),
        },
        "zombies_by_row": {
            str(lane.get("row")): lane.get("zombieCount")
            for lane in observation.get("lanes", []) or []
            if isinstance(lane, dict) and lane.get("row") is not None
        },
        "row_danger": row_danger,
        "coach": dict(coach_fields),
        "stream_coach": dict(coach_fields),
        "human_coach": dict(coach_fields),
        "summary": summary,
        "eval": summary,
        **coach_fields,
        **streamer_fields,
    }
    return payload


def write_runtime_live_status(
    live_status_path: Optional[Path],
    *,
    config: Dict[str, Any],
    status: str,
    mode: str,
    model_path: Optional[Path] = None,
    summary: Optional[Dict[str, Any]] = None,
    observation: Optional[Dict[str, Any]] = None,
    blocked_reason: str = "",
    status_writer: Optional[LiveStatusWriter] = None,
    force: bool = False,
) -> None:
    if live_status_path is None:
        return
    resolved_path = resolved_live_status_path(live_status_path)
    if resolved_path is None:
        return
    summary_values = summary or {}
    observation_values = observation or {}
    writer = status_writer or LiveStatusWriter(resolved_path, min_interval_seconds=0.0)
    immediate = bool(force or status != "running" or status_writer is None)
    writer.write_lazy(
        lambda: build_runtime_live_status_payload(
            config=config,
            status=status,
            mode=mode,
            model_path=model_path,
            summary=summary_values,
            observation=observation_values,
            blocked_reason=blocked_reason,
        ),
        significant_state=live_status_significant_state(
            {"status": status, "blocked_reason": blocked_reason, "mode": mode},
            summary_values,
            observation_values,
        ),
        force=immediate,
    )


def finish_streamer_episode_boundary(
    model: Any,
    vec_env: Any,
    *,
    maximum_actions: int,
    action_masks_fn: Optional[Any] = None,
) -> int:
    """Finish a partial episode without advancing PPO/BC training counters."""

    episode_starts = np.asarray(
        getattr(model, "_last_episode_starts", [False]), dtype=bool
    ).reshape(-1)
    if episode_starts.size and bool(episode_starts[0]):
        return 0
    observation = getattr(model, "_last_obs", None)
    if observation is None:
        raise RuntimeError("blocked_reason=streamer_phase_handoff_observation_missing")
    if action_masks_fn is None:
        from sb3_contrib.common.maskable.utils import get_action_masks

        action_masks_fn = get_action_masks
    for action_count in range(1, max(1, int(maximum_actions)) + 1):
        masks = action_masks_fn(vec_env)
        actions, _ = model.predict(
            observation,
            deterministic=True,
            action_masks=masks,
        )
        observation, _, dones, _ = vec_env.step(actions)
        done_values = np.asarray(dones, dtype=bool).reshape(-1)
        if done_values.size and bool(done_values[0]):
            model._last_obs = observation
            model._last_episode_starts = done_values
            return action_count
    raise RuntimeError("blocked_reason=streamer_phase_handoff_action_limit")


def train(config: Dict[str, Any], live_status_path: Optional[Path] = None) -> Dict[str, Any]:
    streamer_v1_enabled = bool(config.get("streamer_v1_enabled", False))
    demonstration_buffer = None
    streamer_controller = None
    streamer_model_kwargs: Dict[str, Any] = {}
    if streamer_v1_enabled:
        from pvzrl_demonstrations import DemonstrationBuffer
        from pvzrl_streamer_ppo import StreamerMaskablePPO

        MaskablePPO = StreamerMaskablePPO
        experiment_dir = Path(str(config.get("streamer_experiment_dir") or config["run_dir"]))
        demonstration_path = experiment_dir / "demonstrations" / "viewer_demonstrations.npz"
        demonstration_capacity = int(config.get("streamer_demonstration_capacity", 4096) or 4096)
        if demonstration_path.is_file():
            demonstration_buffer = DemonstrationBuffer.load(
                demonstration_path,
                capacity=demonstration_capacity,
                expected_observation_shape=(4297,),
                expected_action_count=ADVENTURE_IDENTITY_ACTION_COUNT,
            )
        else:
            demonstration_buffer = DemonstrationBuffer(
                demonstration_capacity,
                observation_shape=(4297,),
                action_count=ADVENTURE_IDENTITY_ACTION_COUNT,
                persist_path=demonstration_path,
            )
        streamer_model_kwargs = {
            "demonstration_buffer": demonstration_buffer,
            "demonstration_capacity": demonstration_capacity,
            "demonstration_persist_path": demonstration_path,
            "demonstration_persist_every": int(
                config.get("streamer_demonstration_persist_every", 512) or 512
            ),
            "bc_enabled": bool(config.get("streamer_bc_enabled", True)),
            "bc_coefficient": float(config.get("streamer_bc_coefficient", 0.01) or 0.0),
            "bc_batch_size": int(config.get("streamer_bc_batch_size", 32) or 32),
            "bc_update_frequency": int(config.get("streamer_bc_update_frequency", 1) or 1),
            "bc_min_demonstrations": int(config.get("streamer_bc_min_demonstrations", 8) or 8),
        }
        if not bool(config.get("streamer_preserve_bc_rng_state", False)):
            streamer_model_kwargs["bc_seed"] = int(config.get("seed", 0) or 0)
        streamer_controller = make_streamer_v1_controller(config)
    else:
        MaskablePPO = require_maskable_ppo()
    BaseCallback, CallbackList, CheckpointCallback, DummyVecEnv, _ = require_sb3_callbacks()

    live_status_path = resolved_live_status_path(live_status_path)
    runtime_status_writer = LiveStatusWriter(live_status_path)
    if live_status_path is not None:
        print(f"[train] live_status_path={live_status_path}")
    run_dir = Path(config["run_dir"])
    initial_model_path = Path(str(config.get("model_path") or ""))
    requested_continuation = bool(str(config.get("model_path") or "").strip())
    generalist_training = str(config.get("run_mode", "")) == ADVENTURE_GENERALIST_RUN_MODE_TRAIN
    if not generalist_training:
        raise SystemExit(
            "blocked_reason=unsupported_training_mode: Adventure Generalist is the sole maintained trainer."
        )
    config["resume_training"] = bool(requested_continuation)
    config["resume_model_path"] = str(initial_model_path) if requested_continuation else ""
    config["resume_source_model_family"] = str(config.get("resume_source_model_family") or "")

    def _safe_resolve(path: Path) -> Path:
        try:
            return path.resolve()
        except OSError:
            return path

    if requested_continuation:
        source_run_dir = initial_model_path.parent.parent if initial_model_path.parent.name.lower() == "checkpoints" else initial_model_path.parent
        resolved_source_run_dir = _safe_resolve(source_run_dir)
        resolved_run_dir = _safe_resolve(run_dir)
        if resolved_source_run_dir == resolved_run_dir:
            raise SystemExit(
                "blocked_reason=resume_run_dir_must_be_new: "
                f"resume_source_run_dir={resolved_source_run_dir} run_dir={resolved_run_dir}"
            )
        resolved_resume_model = _safe_resolve(initial_model_path)
        resolved_model_target = _safe_resolve(run_dir / "model.zip")
        resolved_final_model_target = _safe_resolve(run_dir / "final_model.zip")
        resolved_checkpoint_dir = _safe_resolve(run_dir / "checkpoints")
        if (
            resolved_resume_model in {resolved_model_target, resolved_final_model_target}
            or _safe_resolve(initial_model_path.parent) == resolved_checkpoint_dir
        ):
            raise SystemExit(
                "blocked_reason=resume_model_overwrite_risk: "
                f"resume_model_path={resolved_resume_model} run_dir={resolved_run_dir}"
            )

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    config.update(apply_model_metadata_defaults(config))
    config["action_count"] = action_count_for_config(config)
    config["seed_slot_signature"] = seed_slot_signature(config)
    config_path = run_dir / "resolved_config.json"

    def _write_run_config_files() -> None:
        (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    _write_run_config_files()
    write_model_metadata(run_dir, config, config_path=config_path)
    (run_dir / "command_used.txt").write_text(command_used() + "\n", encoding="utf-8")
    write_action_map(config, run_dir / "action_map.txt")
    if str(config.get("run_mode", "")) == ADVENTURE_GENERALIST_RUN_MODE_TRAIN:
        initial_loadout = list(config.get("initial_loadout", []))
        selected_count = len(initial_loadout)
        observed_capacity = max(1, min(14, int(config.get("active_seed_slots_at_start", selected_count) or selected_count)))
        frontier_win_streak_required = max(1, int(config.get("adventure_frontier_win_streak_required", 1) or 1))
        (run_dir / "adventure_training_progress.json").write_text(
            json.dumps(
                {
                    "status": "starting",
                    "run_mode": ADVENTURE_GENERALIST_RUN_MODE_TRAIN,
                    "model_family": ADVENTURE_GENERALIST_MODEL_FAMILY,
                    "max_seed_slots": 14,
                    "observed_seed_bank_capacity": observed_capacity,
                    "active_seed_slot_capacity": observed_capacity,
                    "current_seed_bank_capacity": observed_capacity,
                    "selected_loadout_count": selected_count,
                    "active_seed_slot_count": selected_count,
                    "inactive_seed_slot_count": max(0, 14 - selected_count),
                    "inactive_model_slots": max(0, 14 - selected_count),
                    "configured_seed_list": list(config.get("configured_seed_list", initial_loadout)),
                    "selected_loadout": list(initial_loadout),
                    "seed_order_source": str(config.get("seed_order_source", SEED_ORDER_SOURCE_DEFAULT)),
                    "seed_order_preserved": True,
                    "seed_order_blocked_reason": "",
                    "cleared_levels": [],
                    "frontier_level": int(config.get("adventure_start_level", 1) or 1),
                    "frontier_win_streak": 0,
                    "frontier_win_streak_required": frontier_win_streak_required,
                    "frontier_mastery_ready": False,
                    "frontier_promoted_this_episode": False,
                    "frontier_mastery_reset_reason": "",
                    "mastery_sample_source": "frontier",
                    "frontier_replay_supported": True,
                    "frontier_replay_blocked_reason": "",
                    "frontier_mastered_levels": [],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (run_dir / "adventure_training_progress.jsonl").touch()
        (run_dir / "plant_unlocks.json").write_text(
            json.dumps(
                {
                    "status": "starting",
                    "unlocked_seeds": list(config.get("initial_loadout", [])),
                    "eligible_seeds": list(config.get("initial_loadout", [])),
                    "configured_seed_list": list(config.get("configured_seed_list", initial_loadout)),
                    "selected_loadout": list(initial_loadout),
                    "seed_order_source": str(config.get("seed_order_source", SEED_ORDER_SOURCE_DEFAULT)),
                    "seed_order_preserved": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (run_dir / "seed_slot_unlocks.json").write_text(
            json.dumps(
                {
                    "status": "starting",
                    "max_seed_slots": 14,
                    "observed_seed_bank_capacity": observed_capacity,
                    "active_seed_slot_capacity": observed_capacity,
                    "current_seed_bank_capacity": observed_capacity,
                    "selected_loadout_count": selected_count,
                    "active_seed_slot_count": selected_count,
                    "inactive_seed_slot_count": max(0, 14 - selected_count),
                    "inactive_model_slots": max(0, 14 - selected_count),
                    "configured_seed_list": list(config.get("configured_seed_list", initial_loadout)),
                    "selected_loadout": list(initial_loadout),
                    "seed_order_source": str(config.get("seed_order_source", SEED_ORDER_SOURCE_DEFAULT)),
                    "seed_order_preserved": True,
                    "seed_order_blocked_reason": "",
                    "frontier_win_streak": 0,
                    "frontier_win_streak_required": frontier_win_streak_required,
                    "frontier_mastery_ready": False,
                    "frontier_promoted_this_episode": False,
                    "frontier_mastery_reset_reason": "",
                    "frontier_replay_supported": True,
                    "frontier_replay_blocked_reason": "",
                    "frontier_mastered_levels": [],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print(
        "Resolved PPO action space: "
        f"mode={config['action_space_mode']} seed_slots={len(config['plant_types'])}, "
        f"max_seed_slots={config['max_seed_slots']} rows=5, cols=10, "
        f"action_count={config['action_count']} decoder={config['action_decoder_version']}"
    )
    print(
        "Resolved seed slots: "
        f"seed_list={config['seed_slot_signature']['seed_list']} "
        f"plant_types={config['seed_slot_signature']['plant_types']} "
        f"model_family={config.get('model_family', '')}"
    )
    if str(config.get("run_mode", "")) == ADVENTURE_GENERALIST_RUN_MODE_TRAIN:
        print(f"[adventure-generalist] run_mode={config['run_mode']}")
        print(f"[adventure-generalist] model_family={config['model_family']}")
        print(f"[adventure-generalist] observation_version={config['observation_version']}")
        print(f"[adventure-generalist] action_decoder_version={config['action_decoder_version']}")
        print(
            "[adventure-generalist] "
            f"max_seed_slots={config['max_seed_slots']} "
            f"active_seed_slots={len(config.get('initial_loadout', []))} "
            f"action_count={config['action_count']}"
        )
        print(f"[adventure-generalist] initial_loadout={list(config.get('initial_loadout', []))}")
        print(
            "[adventure-generalist] seed_order "
            f"configured_seed_list={list(config.get('configured_seed_list', config.get('seed_list', [])))} "
            f"seed_order_source={config.get('seed_order_source', SEED_ORDER_SOURCE_DEFAULT)} "
            f"randomize_seed_order={bool(config.get('randomize_seed_order', False))}"
        )
        print(f"[adventure-generalist] unlock_aware_seed_curriculum={bool(config.get('unlock_aware_seed_curriculum'))}")
        print(f"[adventure-generalist] infer_capacity_from_unlocks={bool(config.get('infer_capacity_from_unlocks', True))}")
        print(f"[adventure-generalist] allow_weak_unlocked_capacity_fallback={bool(config.get('allow_weak_unlocked_capacity_fallback', False))}")
        print(f"[adventure-generalist] frontier_win_streak_required={int(config.get('adventure_frontier_win_streak_required', 1) or 1)}")
        print(
            "[adventure-generalist] "
            f"strict_startup_validation={bool(config.get('adventure_generalist_strict_startup_validation', True))}"
        )
        resume_training = bool(config.get("resume_training", False))
        resume_model_path = str(config.get("resume_model_path") or "").strip()
        print(f"[adventure-generalist] resume_training={'true' if resume_training else 'false'}")
        if resume_model_path:
            print(f"[adventure-generalist] resume_model_path={resume_model_path}")
        print(f"[adventure-generalist] additional_timesteps={int(config.get('total_timesteps', 0) or 0)}")
        if resume_training:
            print(f"[adventure-generalist] checkpoint_warm_start=True model_path={resume_model_path}")
        else:
            print("[adventure-generalist] checkpoint_warm_start=False reason=scratch_initialization")

    class ExperimentCallback(BaseCallback):
        def __init__(self, csv_path: Path, jsonl_path: Path):
            super().__init__()
            self.metrics = EpisodeMetricWriter(csv_path, jsonl_path)
            self.rows = self.metrics.rows
            self.performance = PerformanceAccumulator()

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            for info in infos:
                perf = info.get("performance")
                if isinstance(perf, dict):
                    self.performance.add(perf)
            self.metrics.append_summaries(
                [
                    info["episode_summary"]
                    for info in infos
                    if isinstance(info, dict) and isinstance(info.get("episode_summary"), dict)
                ]
            )
            if streamer_v1_enabled and isinstance(pvz_env, AdventureGeneralistTrainingEnv):
                current_steps = int(getattr(self.model, "num_timesteps", 0) or 0)
                cycle_steps = int(
                    config.get("streamer_cycle_policy_steps_completed_before", 0) or 0
                ) + max(0, current_steps - starting_model_steps)
                pvz_env.update_streamer_v1_training_status(
                    current_model_ppo_steps=current_steps,
                    cycle_policy_steps_completed=cycle_steps,
                    next_evaluation_countdown=max(
                        0,
                        int(config.get("streamer_policy_steps_per_cycle", 0) or 0)
                        - cycle_steps,
                    ),
                    bc_demonstration_count=(
                        len(demonstration_buffer) if demonstration_buffer is not None else 0
                    ),
                    bc_loss=float(getattr(self.model, "last_bc_loss", 0.0) or 0.0),
                    bc_update_count=int(getattr(self.model, "bc_update_count", 0) or 0),
                    total_environment_actions=int(
                        getattr(self.model, "total_environment_actions", 0) or 0
                    ),
                    viewer_intervention_count=int(
                        getattr(self.model, "viewer_interventions", 0) or 0
                    ),
                )
            return True

    vec_env = DummyVecEnv([lambda: make_monitored_env(config, run_dir / "monitor.csv", live_status_path=live_status_path)])
    runtime_env = vec_env.envs[0] if getattr(vec_env, "envs", None) else None
    runtime_env_metadata = env_metadata_for_config(config, runtime_env)
    continuing = requested_continuation and initial_model_path.exists()
    if continuing:
        print(f"[adventure-generalist] model_path_load_requested={initial_model_path}")
        print("[adventure-generalist] loading prior PPO model for continued training")
        compatibility = validate_adventure_generalist_model_compatibility(
            initial_model_path,
            config,
            "Adventure Generalist training continuation metadata",
            env_metadata=runtime_env_metadata,
        )
        config["resume_source_model_family"] = str(compatibility.get("model_family") or ADVENTURE_GENERALIST_MODEL_FAMILY)
        print(f"Continuing PPO model from {initial_model_path}")
        model = MaskablePPO.load(
            str(initial_model_path),
            env=vec_env,
            tensorboard_log=str(run_dir / "tensorboard"),
            **(
                {
                    "n_steps": int(config["n_steps"]),
                    "batch_size": int(config["batch_size"]),
                }
                if streamer_v1_enabled
                else {}
            ),
            **streamer_model_kwargs,
        )
        model.verbose = config["verbose"]
        validate_adventure_generalist_model_compatibility(
            initial_model_path,
            config,
            "Adventure Generalist training continuation loaded model",
            model=model,
            env_metadata=runtime_env_metadata,
        )
        config["warm_start_used"] = True
        _write_run_config_files()
        write_model_metadata(run_dir, config, config_path=config_path)
        print(f"[adventure-generalist] model_path_loaded={initial_model_path}")
    elif requested_continuation:
        raise SystemExit(f"blocked_reason=resume_model_path_missing: {initial_model_path}")
    else:
        model = MaskablePPO(
            config["policy"],
            vec_env,
            learning_rate=config["learning_rate"],
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            gamma=config["gamma"],
            gae_lambda=config["gae_lambda"],
            ent_coef=config["ent_coef"],
            clip_range=config["clip_range"],
            verbose=config["verbose"],
            seed=config["seed"],
            tensorboard_log=str(run_dir / "tensorboard"),
            **streamer_model_kwargs,
        )

    pvz_env = unwrap_pvz_env(runtime_env)
    if streamer_v1_enabled and not isinstance(pvz_env, AdventureGeneralistTrainingEnv):
        vec_env.close()
        raise SystemExit(
            "blocked_reason=streamer_generalist_env_missing: expected AdventureGeneralistTrainingEnv."
        )

    starting_model_steps = int(getattr(model, "num_timesteps", 0) or 0)
    starting_environment_actions = int(getattr(model, "total_environment_actions", 0) or 0)
    starting_viewer_interventions = int(getattr(model, "viewer_interventions", 0) or 0)
    starting_policy_transitions = int(getattr(model, "policy_transitions_collected", 0) or 0)
    starting_bc_updates = int(getattr(model, "bc_update_count", 0) or 0)
    starting_demonstrations_added = int(
        getattr(demonstration_buffer, "total_added", 0) or 0
    )
    streamer_checkpoint_state = {
        "write_count": 0,
        "next_policy_step": starting_model_steps
        + int(config.get("streamer_checkpoint_policy_steps", 5000) or 5000),
    }
    if streamer_v1_enabled:
        from pvzrl_streamer import StreamerCheckpointManager

        prior = config.get("streamer_cycle_prior_training_metrics")
        prior_metrics = prior if isinstance(prior, dict) else {}

        def persist_streamer_current(checkpoint_model: Any) -> None:
            current_steps = int(getattr(checkpoint_model, "num_timesteps", 0) or 0)
            if current_steps < int(streamer_checkpoint_state["next_policy_step"]):
                return

            def accumulated(name: str, current: int) -> int:
                return int(prior_metrics.get(name, 0) or 0) + max(0, int(current))

            temporary = run_dir / "checkpoints" / (
                f".streamer-current-{os.getpid()}-{time.time_ns()}.zip"
            )
            cycle_before = int(
                config.get("streamer_cycle_policy_steps_completed_before", 0) or 0
            )
            cycle_now = cycle_before + max(0, current_steps - starting_model_steps)
            try:
                checkpoint_model.save(str(temporary))
                manager = StreamerCheckpointManager(
                    Path(str(config.get("streamer_experiment_dir") or config["run_dir"])),
                    Path(str(config["streamer_baseline_checkpoint"])),
                )
                manager.save_current(
                    temporary,
                    model_steps=current_steps,
                    training_cycle=int(config.get("streamer_cycle", 0) or 0),
                    training_metrics={
                        "status": "in_progress",
                        "cycle": int(config.get("streamer_cycle", 0) or 0),
                        "cycle_start_model_steps": int(
                            config.get("streamer_cycle_start_model_steps", starting_model_steps)
                            or starting_model_steps
                        ),
                        "cycle_target_policy_steps": int(
                            config.get("streamer_policy_steps_per_cycle", 0) or 0
                        ),
                        "next_adventure_level": int(
                            getattr(pvz_env, "current_level", config.get("adventure_start_level", 1))
                            or 1
                        ),
                        "ppo_policy_timesteps": cycle_now,
                        "total_environment_actions": accumulated(
                            "total_environment_actions",
                            int(getattr(checkpoint_model, "total_environment_actions", 0) or 0)
                            - starting_environment_actions,
                        ),
                        "viewer_interventions": accumulated(
                            "viewer_interventions",
                            int(getattr(checkpoint_model, "viewer_interventions", 0) or 0)
                            - starting_viewer_interventions,
                        ),
                        "policy_transitions_collected": accumulated(
                            "policy_transitions_collected",
                            int(getattr(checkpoint_model, "policy_transitions_collected", 0) or 0)
                            - starting_policy_transitions,
                        ),
                        "bc_demonstrations_added": accumulated(
                            "bc_demonstrations_added",
                            int(getattr(demonstration_buffer, "total_added", 0) or 0)
                            - starting_demonstrations_added,
                        ),
                        "bc_update_count": accumulated(
                            "bc_update_count",
                            int(getattr(checkpoint_model, "bc_update_count", 0) or 0)
                            - starting_bc_updates,
                        ),
                        "streamer_checkpoint_write_count": accumulated(
                            "streamer_checkpoint_write_count",
                            int(streamer_checkpoint_state["write_count"]) + 1,
                        ),
                    },
                )
            finally:
                temporary.unlink(missing_ok=True)
            streamer_checkpoint_state["write_count"] = int(
                streamer_checkpoint_state["write_count"]
            ) + 1
            frequency = int(config.get("streamer_checkpoint_policy_steps", 5000) or 5000)
            while int(streamer_checkpoint_state["next_policy_step"]) <= current_steps:
                streamer_checkpoint_state["next_policy_step"] = int(
                    streamer_checkpoint_state["next_policy_step"]
                ) + frequency

        model.set_streamer_checkpoint_hook(persist_streamer_current)
    experiment_callback = ExperimentCallback(run_dir / "episode_metrics.csv", run_dir / "episode_metrics.jsonl")
    callback_items: List[Any] = [experiment_callback]
    if not streamer_v1_enabled:
        callback_items.insert(
            0,
            CheckpointCallback(
                save_freq=max(1, config["checkpoint_freq"]),
                save_path=str(run_dir / "checkpoints"),
                name_prefix="ppo_pvz",
            ),
        )
    callbacks = CallbackList(callback_items)
    write_runtime_live_status(
        live_status_path,
        config=config,
        status="running",
        mode=ADVENTURE_GENERALIST_RUN_MODE_TRAIN,
        model_path=Path(str(config.get("model_path") or run_dir / "model.zip")),
        summary={
            "episode": None,
            "episode_length": None,
            "episode_reward": None,
            "total_timesteps": int(getattr(model, "num_timesteps", 0) or 0),
        },
        status_writer=runtime_status_writer,
        force=True,
    )
    if bool(config.get("adventure_generalist_strict_startup_validation", True)):
        if not isinstance(pvz_env, AdventureGeneralistTrainingEnv):
            vec_env.close()
            raise SystemExit(
                "blocked_reason=adventure_generalist_startup_validation_failed: "
                "could not locate AdventureGeneralistTrainingEnv before PPO learn()."
            )
        validation = pvz_env.validate_startup_state(phase="pre_learn", raise_on_failure=False)
        if not validation.get("ok", False):
            error = str(validation.get("actionable_error") or validation.get("reason") or "startup validation failed")
            vec_env.close()
            raise SystemExit(f"blocked_reason=adventure_generalist_startup_validation_failed: {error}")
    print(
        "[adventure-generalist] "
        f"sb3_learn_reset_num_timesteps={'false' if continuing else 'true'}"
    )
    started = time.perf_counter()
    streamer_runtime_status: Dict[str, Any] = {}
    streamer_phase_handoff_actions = 0
    try:
        if streamer_v1_enabled:
            assert isinstance(pvz_env, AdventureGeneralistTrainingEnv)
            assert streamer_controller is not None
            experiment_dir = Path(
                str(config.get("streamer_experiment_dir") or config["run_dir"])
            )
            pvz_env.attach_streamer_v1_controller(
                streamer_controller,
                event_log_path=experiment_dir / "logs" / "streamer_events.jsonl",
                training_cycle=int(config.get("streamer_cycle", 0) or 0),
                platform=str(config.get("streamer_platform", "twitch") or "twitch"),
                training_status={
                    "current_model_ppo_steps": int(getattr(model, "num_timesteps", 0) or 0),
                    "baseline_model_ppo_steps": int(
                        config.get("streamer_baseline_model_steps", 0) or 0
                    ),
                    "cycle_policy_steps_completed": int(
                        config.get("streamer_cycle_policy_steps_completed_before", 0) or 0
                    ),
                    "next_evaluation_countdown": int(
                        config.get("streamer_policy_steps_per_cycle", 0) or 0
                    )
                    - int(config.get("streamer_cycle_policy_steps_completed_before", 0) or 0),
                    "bc_demonstration_count": (
                        len(demonstration_buffer) if demonstration_buffer is not None else 0
                    ),
                    "bc_loss": float(getattr(model, "last_bc_loss", 0.0) or 0.0),
                    "bc_update_count": int(getattr(model, "bc_update_count", 0) or 0),
                },
            )
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=callbacks,
            reset_num_timesteps=not continuing,
        )
        if streamer_v1_enabled:
            # Exact PPO accounting stops at the requested rollout boundary.  If
            # that boundary is mid-episode, finish the episode autonomously with
            # Twitch gated and without PPO/BC updates so evaluation starts clean.
            assert streamer_controller is not None
            streamer_controller.begin_phase("TRAIN_DRAIN", accepting=False)
            streamer_phase_handoff_actions = finish_streamer_episode_boundary(
                model,
                vec_env,
                maximum_actions=max(
                    1, int(config.get("adventure_hard_max_steps", 20_000) or 20_000)
                ),
            )
            if streamer_phase_handoff_actions:
                model.total_environment_actions = int(
                    getattr(model, "total_environment_actions", 0) or 0
                ) + streamer_phase_handoff_actions
    finally:
        if streamer_v1_enabled and isinstance(pvz_env, AdventureGeneralistTrainingEnv):
            streamer_runtime_status = pvz_env._streamer_v1_live_status()
            streamer_runtime_status["next_adventure_level"] = int(pvz_env.current_level)
            streamer_runtime_status["phase_handoff_actions"] = int(
                streamer_phase_handoff_actions
            )
        vec_env.close()
    elapsed = max(1e-6, time.perf_counter() - started)
    timesteps = int(getattr(model, "num_timesteps", config["total_timesteps"]))
    fps_avg = max(0, timesteps - starting_model_steps) / elapsed
    model.save(str(run_dir / "model"))
    model_path = run_dir / "model.zip"
    final_model_path = run_dir / "final_model.zip"
    if model_path.exists() and not streamer_v1_enabled:
        shutil.copyfile(model_path, final_model_path)
    write_model_metadata(run_dir, config, model_path=model_path, config_path=config_path)
    summary = summarize_episode_rows(
        getattr(experiment_callback, "rows", []),
        timesteps,
        run_dir,
        model_path,
        fps_avg,
    )
    summary["model_steps"] = timesteps
    summary["ppo_policy_timesteps"] = max(0, timesteps - starting_model_steps)
    summary["model_path"] = str(model_path)
    if streamer_v1_enabled:
        lifetime_environment_actions = int(getattr(model, "total_environment_actions", 0) or 0)
        lifetime_viewer_interventions = int(getattr(model, "viewer_interventions", 0) or 0)
        lifetime_policy_transitions = int(getattr(model, "policy_transitions_collected", 0) or 0)
        lifetime_bc_updates = int(getattr(model, "bc_update_count", 0) or 0)
        lifetime_demonstrations_added = int(
            getattr(demonstration_buffer, "total_added", 0) or 0
        )
        summary.update(
            {
                "total_environment_actions": max(
                    0, lifetime_environment_actions - starting_environment_actions
                ),
                "viewer_interventions": max(
                    0, lifetime_viewer_interventions - starting_viewer_interventions
                ),
                "policy_transitions_collected": max(
                    0, lifetime_policy_transitions - starting_policy_transitions
                ),
                "bc_demonstration_count": int(len(demonstration_buffer)) if demonstration_buffer is not None else 0,
                "bc_demonstrations_added": max(
                    0, lifetime_demonstrations_added - starting_demonstrations_added
                ),
                "bc_update_count": max(0, lifetime_bc_updates - starting_bc_updates),
                "bc_loss": float(getattr(model, "last_bc_loss", 0.0) or 0.0),
                "bc_policy_agreement": float(getattr(model, "last_bc_policy_agreement", 0.0) or 0.0),
                "ppo_policy_loss": float(getattr(model, "last_policy_loss", 0.0) or 0.0),
                "ppo_value_loss": float(getattr(model, "last_value_loss", 0.0) or 0.0),
                "ppo_entropy_loss": float(getattr(model, "last_entropy_loss", 0.0) or 0.0),
                "streamer_phase_handoff_actions": int(streamer_phase_handoff_actions),
                "next_adventure_level": int(
                    streamer_runtime_status.get(
                        "next_adventure_level",
                        config.get("adventure_start_level", 1),
                    )
                    or 1
                ),
                "streamer_lifetime_counters": {
                    "total_environment_actions": lifetime_environment_actions,
                    "viewer_interventions": lifetime_viewer_interventions,
                    "policy_transitions_collected": lifetime_policy_transitions,
                    "bc_demonstrations_added": lifetime_demonstrations_added,
                    "bc_update_count": lifetime_bc_updates,
                },
                "streamer_runtime": streamer_runtime_status,
            }
        )
        summary["streamer_checkpoint_write_count"] = int(
            streamer_checkpoint_state["write_count"]
        )
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if config.get("debug_performance"):
        perf_summary = experiment_callback.performance.summary(fps_avg)
        (run_dir / "performance_summary.json").write_text(json.dumps(perf_summary, indent=2), encoding="utf-8")
    write_runtime_live_status(
        live_status_path,
        config=config,
        status="complete",
        mode=ADVENTURE_GENERALIST_RUN_MODE_TRAIN,
        model_path=model_path,
        summary=summary,
        status_writer=runtime_status_writer,
        force=True,
    )
    print_validation_summary(summary)
    print(f"Saved final model to {model_path}")
    if final_model_path.is_file():
        print(f"Saved compatibility model copy to {final_model_path}")
    print(f"Saved monitor log to {run_dir / 'monitor.csv'}")
    print(f"Saved episode metrics to {run_dir / 'episode_metrics.csv'}")
    return summary


def adventure_evaluate(config: Dict[str, Any], model_path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    MaskablePPO = require_maskable_ppo()
    model = MaskablePPO.load(str(model_path))
    env_metadata = env_metadata_for_config(config)
    report = loaded_model_compatibility_report(model, model_path, config, env_metadata=env_metadata)
    print_compatibility_report("[compat:Adventure Generalist eval]", report)
    if not report.ok:
        write_eval_live_status(
            args.live_status_path,
            config=config,
            model_path=model_path,
            report=report,
            status="blocked",
            mode=ADVENTURE_GENERALIST_RUN_MODE_EVAL,
        )
    raise_if_incompatible(report)
    compatibility = validate_adventure_generalist_model_compatibility(
        model_path,
        config,
        "Adventure Generalist evaluation",
        model=model,
        env_metadata=env_metadata,
    )
    config["metadata_path"] = report.metadata_path
    config["metadata_inferred"] = bool(report.metadata_inferred)
    config["model_compatibility"] = model_compatibility_live_status(report)
    print(f"[adventure-generalist-eval] policy=ppo model_path={model_path}")
    print(
        "[adventure-generalist-eval] inference_only=True "
        f"deterministic={args.deterministic} action_count={compatibility['action_count']}"
    )
    print(
        "[adventure-generalist-eval] compatible_seed_slots=True "
        f"seed_list={compatibility['expected_seed_list']} "
        f"plant_types={compatibility['expected_plant_types']} "
        f"decoder={compatibility['action_decoder_version']} "
        f"observation={compatibility['observation_version']} "
        f"metadata={compatibility['metadata_path']}"
    )
    payload = run_adventure_eval(
        config=config,
        env_config=make_env_config(config),
        model=model,
        model_path=model_path,
        deterministic=bool(args.deterministic),
        advance_on_wins=max(1, int(config["advance_on_wins"])),
        max_adventure_levels=max(1, int(config["max_adventure_levels"])),
        max_attempts_per_level=max(1, int(config["max_attempts_per_level"])),
        adventure_start_level=max(1, int(config["adventure_start_level"])),
        live_status_path=args.live_status_path,
        gui=bool(args.gui),
        adventure_soft_max_steps=int(config.get("adventure_soft_max_steps", DEFAULT_ADVENTURE_SOFT_MAX_STEPS)),
        adventure_hard_max_steps=int(config.get("adventure_hard_max_steps", DEFAULT_ADVENTURE_HARD_MAX_STEPS)),
        adventure_final_wave_extension=bool(config.get("adventure_final_wave_extension", True)),
        evaluation_episode_limit=(
            int(config["streamer_evaluation_episode_limit"])
            if config.get("streamer_evaluation_episode_limit") is not None
            else None
        ),
        strict_level_identity=bool(config.get("streamer_v1_enabled", False)),
    )
    summary = payload.get("summary") if isinstance(payload, dict) else None
    if isinstance(summary, dict):
        summary["model_steps"] = int(getattr(model, "num_timesteps", 0) or 0)
    return payload


def run_streamer_v1(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Run Streamer V1 as an overlay on the maintained train/eval entrypoints."""

    baseline = Path(str(config.get("streamer_baseline_checkpoint") or ""))
    if not baseline.is_absolute():
        baseline = Path.cwd() / baseline
    baseline = baseline.resolve()
    if not baseline.is_file():
        raise SystemExit(f"blocked_reason=streamer_baseline_checkpoint_missing: {baseline}")
    config["streamer_baseline_checkpoint"] = str(baseline)

    MaskablePPO = require_maskable_ppo()
    baseline_model = MaskablePPO.load(str(baseline))
    baseline_report = loaded_model_compatibility_report(
        baseline_model,
        baseline,
        config,
        env_metadata=env_metadata_for_config(config),
    )
    print_compatibility_report("[compat:Streamer V1 baseline]", baseline_report)
    raise_if_incompatible(baseline_report)
    config["streamer_baseline_model_steps"] = int(
        getattr(baseline_model, "num_timesteps", 0) or 0
    )
    del baseline_model

    status_writer = LiveStatusWriter(resolved_live_status_path(args.live_status_path))

    def compact_evaluation(value: Any) -> Dict[str, Any]:
        payload = value if isinstance(value, dict) else {}
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else payload
        return {
            key: summary[key]
            for key in (
                "episodes_completed",
                "win_rate",
                "avg_reward",
                "avg_wave",
                "avg_kills",
                "avg_plants",
                "avg_mowers_lost",
                "illegal_actions",
                "model_steps",
            )
            if key in summary
        }

    def read_object(path: Path) -> Dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    experiment_dir = Path(str(config["run_dir"]))

    def experiment_status_context() -> Dict[str, Any]:
        baseline_eval = read_object(
            experiment_dir / "evaluations" / "baseline" / "evaluation.json"
        )
        state = read_object(experiment_dir / "streamer_state.json")
        best = read_object(
            experiment_dir / "checkpoints" / "best" / "streamer_checkpoint.json"
        )
        return {
            "streamer_baseline_evaluation": compact_evaluation(baseline_eval),
            "streamer_current_evaluation": compact_evaluation(
                state.get("current_evaluation", {})
            ),
            "streamer_best_evaluation": compact_evaluation(best.get("evaluation", {})),
            "streamer_best_model_steps": int(best.get("model_steps", 0) or 0),
            "streamer_next_adventure_level": int(
                state.get("next_adventure_level", baseline_eval.get("next_adventure_level", 0))
                or 0
            ),
        }

    def publish_status(payload: Any) -> None:
        state = dict(payload) if isinstance(payload, dict) else {}
        last_cycle = state.get("last_cycle") if isinstance(state.get("last_cycle"), dict) else {}
        training = last_cycle.get("training") if isinstance(last_cycle.get("training"), dict) else {}
        runtime = training.get("streamer_runtime") if isinstance(training.get("streamer_runtime"), dict) else {}
        live_payload = {
            "streamer_v1_enabled": True,
            "mode": str(state.get("mode") or state.get("streamer_phase") or STREAM_TRAIN),
            "streamer_mode": str(
                state.get("streamer_phase") or state.get("mode") or STREAM_TRAIN
            ),
            "run_mode": ADVENTURE_GENERALIST_RUN_MODE_TRAIN,
            "status": str(state.get("status") or "running"),
            "health": "LIVE" if state.get("status") == "running" else "DEAD",
            "updated_at": time.time(),
            "active_run": str(config.get("run_dir", "")),
            "streamer_platform": str(config.get("streamer_platform", "twitch") or "twitch"),
            "streamer_cycle": int(
                state.get("current_cycle", state.get("completed_cycle", 0)) or 0
            ),
            "current_model_ppo_steps": int(
                state.get("current_model_steps", training.get("model_steps", 0)) or 0
            ),
            "baseline_model_ppo_steps": int(
                config.get("streamer_baseline_model_steps", 0) or 0
            ),
            "ppo_policy_timesteps": int(state.get("ppo_policy_timesteps", 0) or 0),
            "next_evaluation_countdown": int(
                state.get("next_evaluation_policy_steps", 0) or 0
            ),
            "next_adventure_level": int(state.get("next_adventure_level", 0) or 0),
            "viewer_command_queue_depth": int(
                state.get("viewer_command_queue_depth", runtime.get("viewer_command_queue_depth", 0))
                or 0
            ),
            "last_viewer_action": runtime.get("last_viewer_action", {}),
            "last_action_source": runtime.get("last_action_source", "MODEL"),
            "viewer_intervention_count": int(
                training.get("viewer_interventions", runtime.get("viewer_intervention_count", 0))
                or 0
            ),
            "evaluation_chat_control": False,
            "bc_updates_enabled": bool(state.get("bc_updates_enabled", False)),
            "ppo_updates_enabled": bool(
                state.get("ppo_updates_enabled", state.get("mode") == STREAM_TRAIN)
            ),
            "bc_demonstration_count": int(training.get("bc_demonstration_count", 0) or 0),
            "bc_loss": float(training.get("bc_loss", 0.0) or 0.0),
            "bc_update_count": int(training.get("bc_update_count", 0) or 0),
            "baseline_evaluation": compact_evaluation(state.get("baseline_evaluation", {})),
            "current_evaluation": compact_evaluation(state.get("current_evaluation", {})),
            "best_evaluation": compact_evaluation(state.get("best_evaluation", {})),
            "best_model_steps": int(state.get("best_model_steps", 0) or 0),
            "current_checkpoint": str(state.get("current_checkpoint") or ""),
            "best_checkpoint": str(state.get("best_checkpoint") or ""),
            "baseline_checkpoint": str(state.get("baseline_checkpoint") or baseline),
            "action_count": int(
                config.get("action_count", ADVENTURE_IDENTITY_ACTION_COUNT)
            ),
            "observation_version": str(config.get("observation_version", "")),
        }
        source_state = runtime.get("twitch_connection_state")
        if source_state is not None:
            live_payload["twitch_connection_state"] = source_state
        status_writer.write(
            live_payload,
            force=live_payload["status"] in {"complete", "blocked", "error"},
        )

    def train_cycle(
        start_model: Path,
        cycle: int,
        cycle_run_dir: Path,
        policy_steps: int,
        adventure_start_level: int,
    ) -> Dict[str, Any]:
        cycle_config = deepcopy(config)
        source_record = read_object(start_model.parent / "streamer_checkpoint.json")
        source_cycle = int(source_record.get("training_cycle", 0) or 0)
        prior_training = (
            source_record.get("training_metrics", {})
            if source_cycle == int(cycle)
            and isinstance(source_record.get("training_metrics"), dict)
            else {}
        )
        source_model_steps = int(source_record.get("model_steps", 0) or 0)
        cycle_start_model_steps = int(
            prior_training.get("cycle_start_model_steps", source_model_steps) or source_model_steps
        )
        cycle_config.update(
            {
                "run_mode": ADVENTURE_GENERALIST_RUN_MODE_TRAIN,
                "run_dir": str(cycle_run_dir),
                "model_path": str(start_model),
                "resume_model_path": str(start_model),
                "resume_training": True,
                "total_timesteps": int(policy_steps),
                "adventure_start_level": int(adventure_start_level),
                "streamer_cycle": int(cycle),
                "streamer_phase": STREAM_TRAIN,
                "streamer_experiment_dir": str(Path(str(config["run_dir"]))),
                "streamer_preserve_bc_rng_state": bool(source_record),
                "streamer_cycle_policy_steps_completed_before": int(
                    prior_training.get("ppo_policy_timesteps", 0) or 0
                ),
                "streamer_cycle_prior_training_metrics": dict(prior_training),
                "streamer_cycle_start_model_steps": cycle_start_model_steps,
                **experiment_status_context(),
            }
        )
        summary = train(cycle_config, args.live_status_path)
        return {
            **summary,
            "cycle": int(cycle),
            "ppo_policy_timesteps": int(summary.get("ppo_policy_timesteps", policy_steps) or 0),
        }

    def evaluate_checkpoint(
        model_path: Path,
        evaluation_dir: Path,
        episodes: int,
        adventure_start_level: int,
    ) -> Dict[str, Any]:
        evaluation_config = deepcopy(config)
        checkpoint_record = read_object(model_path.parent / "streamer_checkpoint.json")
        evaluation_config.update(
            {
                "run_mode": ADVENTURE_GENERALIST_RUN_MODE_EVAL,
                "run_dir": str(evaluation_dir),
                "model_path": str(model_path),
                "streamer_phase": EVALUATE,
                "streamer_evaluation_episode_limit": int(episodes),
                "adventure_start_level": int(adventure_start_level),
                "streamer_bc_enabled": False,
                "human_coach_enabled": False,
                "stream_coach_enabled": False,
                "stream_coach_apply_enabled": False,
                "streamer_current_model_steps": int(
                    checkpoint_record.get("model_steps", 0) or 0
                ),
                **experiment_status_context(),
                "max_adventure_levels": max(int(episodes), int(config.get("max_adventure_levels", 1) or 1)),
                "max_attempts_per_level": max(int(episodes), int(config.get("max_attempts_per_level", 1) or 1)),
            }
        )
        evaluation_args = argparse.Namespace(**vars(args))
        evaluation_args.gui = False
        evaluation_args.deterministic = True
        payload = adventure_evaluate(evaluation_config, model_path, evaluation_args)
        summary = payload.get("summary") if isinstance(payload, dict) else None
        completed = int(summary.get("episodes_completed", 0) or 0) if isinstance(summary, dict) else 0
        top_level_completed = int(payload.get("evaluation_episodes_completed", 0) or 0)
        stop_reason = str(payload.get("stop_reason") or "")
        if (
            completed != int(episodes)
            or top_level_completed != int(episodes)
            or stop_reason != "evaluation_episode_limit_reached"
        ):
            raise RuntimeError(
                "blocked_reason=streamer_evaluation_incomplete: "
                f"requested={episodes} summary_completed={completed} "
                f"top_level_completed={top_level_completed} stop_reason={stop_reason}"
            )
        return payload

    return run_streamer_cycles(
        config=config,
        train_cycle=train_cycle,
        evaluate_checkpoint=evaluate_checkpoint,
        status_sink=publish_status,
    )


def check_deps() -> int:
    missing: List[str] = []
    for module in ("gymnasium", "numpy", "stable_baselines3", "sb3_contrib", "websockets"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        print("PPO readiness: NO")
        print("Missing Python modules:", ", ".join(missing))
        print("Install with: python -m pip install -r requirements-ppo.txt")
        return 1
    print("PPO readiness: YES")
    print("MaskablePPO and Streamer dependencies are installed.")
    return 0


def loaded_model_contract(model_path: Path) -> Tuple[int, Optional[List[int]]]:
    MaskablePPO = require_maskable_ppo()
    model = MaskablePPO.load(str(model_path))
    return _model_action_count(model), _model_observation_shape(model)


def metadata_dry_run(config: Dict[str, Any], model_path: Path) -> int:
    run_mode = str(config.get("run_mode", ""))
    if not model_path.exists():
        if run_mode == ADVENTURE_GENERALIST_RUN_MODE_TRAIN:
            if str(config.get("model_path") or "").strip():
                payload = {"ok": False, "blocked_reason": "model_path_missing", "model_path": str(model_path)}
                print(json.dumps(payload, indent=2))
                return 1
            payload = {
                "ok": True,
                "run_mode": ADVENTURE_GENERALIST_RUN_MODE_TRAIN,
                "model_family": ADVENTURE_GENERALIST_MODEL_FAMILY,
                "scratch_initialization": True,
                "checkpoint_warm_start": False,
                "checkpoint_warm_start_reason": "scratch_initialization",
                "action_count": action_count_for_config(config),
                "max_seed_slots": int(config.get("max_seed_slots", 14) or 14),
                "active_seed_slots": len(config.get("initial_loadout", [])),
                "inactive_seed_slots": max(0, int(config.get("max_seed_slots", 14) or 14) - len(config.get("initial_loadout", []))),
                "observation_version": str(config.get("observation_version", "")),
                "action_decoder_version": str(config.get("action_decoder_version", "")),
                "action_space_mode": str(config.get("action_space_mode", "")),
                "initial_loadout": list(config.get("initial_loadout", [])),
                "model_path": str(model_path),
                "note": "No checkpoint is required for Adventure Generalist 14-slot identity scratch training.",
            }
            print(json.dumps(payload, indent=2))
            return 0
        payload = {"ok": False, "blocked_reason": "model_path_missing", "model_path": str(model_path)}
        print(json.dumps(payload, indent=2))
        return 1
    model_action_count, model_observation_shape = loaded_model_contract(model_path)
    summary = validate_adventure_generalist_model_compatibility(
        model_path,
        config,
        "Adventure Generalist metadata dry run",
        model_action_count=model_action_count,
        model_observation_shape=model_observation_shape,
        env_metadata=env_metadata_for_config(config),
    )
    payload = {
        "ok": True,
        "run_mode": run_mode,
        "model_path": str(model_path),
        "model_action_count": model_action_count,
        "model_observation_shape": model_observation_shape,
        "model_compatibility": summary.get("model_compatibility", {}),
    }
    print(json.dumps(payload, indent=2))
    return 0


def execution_route_for_config(
    config: Dict[str, Any],
) -> str:
    """Return the runtime branch selected by the resolved run mode."""

    run_mode = str(config.get("run_mode") or "")
    if run_mode == ADVENTURE_GENERALIST_RUN_MODE_TRAIN:
        return "train"
    if run_mode == ADVENTURE_GENERALIST_RUN_MODE_EVAL:
        return "adventure_generalist_eval"
    return ""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or evaluate the PvZRL Adventure Generalist MaskablePPO policy.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--adventure-generalist-train",
        action="store_true",
        help="Train the 14-slot identity Adventure Generalist (fresh when no resume model is provided).",
    )
    parser.add_argument("--adventure-generalist-eval", action="store_true", help="Evaluate a 14-slot identity Adventure Generalist in Adventure progression.")
    parser.add_argument(
        "--run-mode",
        choices=(
            ADVENTURE_GENERALIST_RUN_MODE_TRAIN,
            ADVENTURE_GENERALIST_RUN_MODE_EVAL,
        ),
    )
    parser.add_argument("--check-deps", action="store_true")
    parser.add_argument("--metadata-dry-run", action="store_true", help="Validate model metadata/action compatibility without starting the game.")
    parser.add_argument("--model", "--model-path", dest="model", type=Path)
    parser.add_argument(
        "--resume-model-path",
        type=Path,
        default=None,
        help="Training-only checkpoint/model .zip to continue PPO learning from (adds timesteps; does not overwrite source).",
    )
    parser.add_argument("--run-dir")
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--gae-lambda", type=float)
    parser.add_argument("--ent-coef", type=float)
    parser.add_argument("--clip-range", type=float)
    parser.add_argument("--verbose", type=int)
    parser.add_argument("--step-seconds", type=float)
    parser.add_argument("--game-speed", type=float)
    parser.add_argument("--game-speed-mode", choices=("game_speed", "time_scale", "safe"))
    parser.add_argument("--valid-speed-mode", action="store_true")
    parser.add_argument("--start-sun", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--board-timeout", type=float)
    parser.add_argument("--gameplay-ready-timeout", type=float)
    parser.add_argument("--poll-seconds", type=float)
    parser.add_argument("--wait-gameplay-ready", action="store_true")
    parser.add_argument("--skip-board-wait", action="store_true")
    parser.add_argument("--quick-wait", action="store_true")
    parser.add_argument("--seed-list", default=None)
    parser.add_argument("--initial-loadout", default=None)
    parser.add_argument("--max-seed-slots", type=int, default=None)
    parser.add_argument("--seed-click-delay", type=float)
    parser.add_argument("--lets-rock-delay", type=float)
    parser.add_argument("--post-start-delay", type=float)
    parser.add_argument("--seed-screen-check-interval", type=int)
    parser.add_argument("--advance-on-wins", type=int, default=None)
    parser.add_argument("--max-adventure-levels", type=int, default=None)
    parser.add_argument("--max-attempts-per-level", type=int, default=None)
    parser.add_argument("--adventure-start-level", type=int, default=None)
    parser.add_argument("--adventure-soft-max-steps", type=int, default=None)
    parser.add_argument("--adventure-hard-max-steps", type=int, default=None)
    parser.add_argument("--adventure-final-wave-extension", dest="adventure_final_wave_extension", action="store_true", default=None)
    parser.add_argument("--no-adventure-final-wave-extension", dest="adventure_final_wave_extension", action="store_false")
    parser.add_argument(
        "--adventure-generalist-strict-startup-validation",
        dest="adventure_generalist_strict_startup_validation",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-adventure-generalist-strict-startup-validation",
        dest="adventure_generalist_strict_startup_validation",
        action="store_false",
    )
    parser.add_argument("--unlock-aware-seed-curriculum", action="store_true")
    parser.add_argument("--seed-curriculum", choices=("conservative", "varied"), default=None)
    parser.add_argument("--randomize-seed-order", action="store_true")
    parser.add_argument("--unlock-introduction-delay", type=int, default=None)
    parser.add_argument("--new-plant-min-inclusion-prob", type=float, default=None)
    parser.add_argument("--core-seed-names", default=None)
    parser.add_argument("--new-unlock-guarantee-episodes", type=int, default=None)
    parser.add_argument("--infer-capacity-from-unlocks", dest="infer_capacity_from_unlocks", action="store_true", default=None)
    parser.add_argument("--no-infer-capacity-from-unlocks", dest="infer_capacity_from_unlocks", action="store_false")
    parser.add_argument("--allow-weak-unlocked-capacity-fallback", action="store_true")
    parser.add_argument("--adventure-replay-cleared-levels", action="store_true")
    parser.add_argument("--adventure-frontier-sample-prob", type=float, default=None)
    parser.add_argument("--adventure-recent-cleared-sample-prob", type=float, default=None)
    parser.add_argument("--adventure-maintenance-sample-prob", type=float, default=None)
    parser.add_argument("--adventure-frontier-win-streak-required", type=int, default=None)
    parser.add_argument("--live-status-path", type=Path, default=Path("runs/live_status.json"))
    parser.add_argument(
        "--streamer-v1",
        dest="streamer_v1_enabled",
        action="store_true",
        help="Run the FIFO Twitch/mock intervention, PPO+BC, and autonomous evaluation cycle overlay.",
    )
    parser.add_argument("--streamer-platform", choices=("twitch", "mock"), default=None)
    parser.add_argument("--streamer-baseline-checkpoint", type=Path, default=None)
    parser.add_argument("--streamer-intervention-interval-seconds", type=float, default=None)
    parser.add_argument("--streamer-command-ttl-seconds", type=float, default=None)
    parser.add_argument("--streamer-command-queue-capacity", type=int, default=None)
    parser.add_argument("--streamer-message-max-chars", type=int, default=None)
    parser.add_argument("--streamer-policy-steps-per-cycle", type=int, default=None)
    parser.add_argument("--streamer-checkpoint-policy-steps", type=int, default=None)
    parser.add_argument("--streamer-evaluation-episodes", type=int, default=None)
    parser.add_argument("--streamer-max-cycles", type=int, default=None)
    parser.add_argument("--streamer-endurance-hours", type=float, default=None)
    parser.add_argument("--streamer-bc-enabled", dest="streamer_bc_enabled", action="store_true", default=None)
    parser.add_argument("--no-streamer-bc", dest="streamer_bc_enabled", action="store_false")
    parser.add_argument("--streamer-bc-coefficient", type=float, default=None)
    parser.add_argument("--streamer-demonstration-capacity", type=int, default=None)
    parser.add_argument("--streamer-bc-batch-size", type=int, default=None)
    parser.add_argument("--streamer-bc-update-frequency", type=int, default=None)
    parser.add_argument("--streamer-bc-min-demonstrations", type=int, default=None)
    parser.add_argument("--streamer-twitch-client-id-env", default=None)
    parser.add_argument("--streamer-twitch-access-token-env", default=None)
    parser.add_argument("--streamer-twitch-broadcaster-id-env", default=None)
    parser.add_argument("--streamer-twitch-user-id-env", default=None)
    parser.add_argument("--streamer-viewer-hash-secret-env", default=None)
    parser.add_argument("--streamer-mock-script", type=Path, default=None)
    parser.add_argument("--human-coach-enabled", action="store_true", help="Enable local/mock human coach action overrides.")
    parser.add_argument(
        "--human-coach-command-mode",
        choices=("override", "assist", "coach_only", "viewer_suggestion"),
        default=None,
        help="How approved human commands interact with the model action.",
    )
    parser.add_argument(
        "--intervention-log-path",
        type=Path,
        default=None,
        help="JSONL path for unified assisted intervention records.",
    )
    parser.add_argument("--human-coach-command-path", type=Path, default=None, help="Plain text or JSONL file of local coach commands.")
    parser.add_argument("--human-coach-log-path", type=Path, default=None, help="JSONL log path for human coach decisions.")
    parser.add_argument("--human-coach-reward", action="store_true", help="Apply small optional coach reward shaping.")
    parser.add_argument("--human-coach-fusion-enabled", action="store_true", help="Enable bridge-probed !fuse coach overrides.")
    parser.add_argument("--stream-coach-enabled", action="store_true", help="Enable mock/local stream crowd coach overrides.")
    parser.add_argument(
        "--stream-coach-mode",
        choices=("twitch", "youtube", "mock"),
        default=None,
        help="Stream coach mode/source family. Alias-friendly companion to --stream-coach-platform.",
    )
    parser.add_argument(
        "--stream-coach-platform",
        choices=("twitch", "youtube", "mock"),
        default=None,
        help="Stream coach platform: twitch|youtube|mock.",
    )
    parser.add_argument("--stream-coach-window-sec", type=float, default=None, help="Crowd vote window length in seconds.")
    parser.add_argument("--stream-coach-min-votes", type=int, default=None, help="Minimum votes needed to select a crowd command.")
    parser.add_argument("--stream-coach-max-actions-per-minute", type=int, default=None, help="Upper bound for accepted crowd actions per minute.")
    parser.add_argument("--stream-coach-command-path", type=Path, default=None, help="Mock/local stream command JSONL source path.")
    parser.add_argument("--stream-coach-mock-script", type=Path, default=None, help="Deterministic mock stream chat JSONL script.")
    parser.add_argument("--stream-coach-dry-run", action="store_true", help="Parse/validate stream commands without applying them.")
    parser.add_argument("--stream-coach-apply", action="store_true", help="Allow validated safe stream commands to affect the active coach path.")
    parser.add_argument("--stream-coach-reward", action="store_true", help="Apply optional stream crowd-coach reward shaping.")
    parser.add_argument("--stream-coach-log-path", type=Path, default=None, help="JSONL log path for stream crowd-coach events.")
    parser.add_argument("--coach-allow-fusion-planning", action="store_true", help="Allow coach !fuse planning via fusion probe when available.")
    parser.add_argument("--fusion-bridge-enabled", action="store_true", help="Enable fusion bridge probe routing for coach commands.")
    parser.add_argument("--fusion-policy", choices=("none", "observe", "scripted", "assist"), default=None)
    parser.add_argument(
        "--fusion-action-mask-enabled",
        action="store_true",
        help="Expose occupied compatible tiles as legal fuse actions in the model action mask "
        "(and route those placements to the fusion bridge). Off by default.",
    )
    parser.add_argument("--enable-board-plant-identity", dest="enable_board_plant_identity", action="store_true", default=None)
    parser.add_argument("--no-board-plant-identity", dest="enable_board_plant_identity", action="store_false")
    parser.add_argument("--enable-fusion-chain-rewards", dest="enable_fusion_chain_rewards", action="store_true", default=None)
    parser.add_argument("--no-fusion-chain-rewards", dest="enable_fusion_chain_rewards", action="store_false")
    parser.add_argument("--enable-recipe-discovery-reward", dest="enable_recipe_discovery_reward", action="store_true", default=None)
    parser.add_argument("--no-recipe-discovery-reward", dest="enable_recipe_discovery_reward", action="store_false")
    parser.add_argument("--enable-repeat-recipe-decay", dest="enable_repeat_recipe_decay", action="store_true", default=None)
    parser.add_argument("--no-repeat-recipe-decay", dest="enable_repeat_recipe_decay", action="store_false")
    parser.add_argument("--enable-fusion-curriculum", dest="enable_fusion_curriculum", action="store_true", default=None)
    parser.add_argument("--no-fusion-curriculum", dest="enable_fusion_curriculum", action="store_false")
    parser.add_argument("--enable-later-plant-curriculum", dest="enable_later_plant_curriculum", action="store_true", default=None)
    parser.add_argument("--no-later-plant-curriculum", dest="enable_later_plant_curriculum", action="store_false")
    parser.add_argument("--enable-coach-fusion-sampling", dest="enable_coach_fusion_sampling", action="store_true", default=None)
    parser.add_argument("--no-coach-fusion-sampling", dest="enable_coach_fusion_sampling", action="store_false")
    parser.add_argument("--fusion-curriculum-prob", type=float)
    parser.add_argument("--later-plant-curriculum-prob", type=float)
    parser.add_argument("--coach-fusion-prob", type=float)
    parser.add_argument("--tactical-masks", action="store_true")
    parser.add_argument("--wallnut-tactical-mask", action="store_true")
    parser.add_argument("--cherrybomb-tactical-mask", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--deterministic", dest="deterministic", action="store_true", default=True)
    parser.add_argument("--stochastic", dest="deterministic", action="store_false")
    parser.add_argument("--debug-performance", "--debug-perf", dest="debug_performance", action="store_true")
    parser.add_argument("--debug-observation", action="store_true")
    parser.add_argument("--debug-sun", action="store_true")
    parser.add_argument(
        "--debug-sun-sample-interval",
        type=int,
        default=None,
        help="Sample interval in steps for debug sun economy logging.",
    )
    parser.add_argument("--mower-loss-penalty", type=float, default=None)
    parser.add_argument("--danger-delta-scale", type=float, default=None)
    parser.add_argument("--lane-response-reward", type=float, default=None)
    parser.add_argument("--undefended-close-threat-penalty", type=float, default=None)
    parser.add_argument("--close-threat-threshold", type=float, default=None)
    parser.add_argument("--threat-balanced-row-reward", type=float, default=None)
    parser.add_argument("--threat-balanced-zero-defender-bonus", type=float, default=None)
    parser.add_argument("--overdefended-row-penalty", type=float, default=None)
    parser.add_argument("--role-positioning-reward", type=float, default=None)
    parser.add_argument("--first-peashooter-in-row-reward", type=float, default=None)
    parser.add_argument("--first-defense-undefended-threatened-row-reward", type=float, default=None)
    parser.add_argument("--all-rows-peashooter-coverage-reward", type=float, default=None)
    parser.add_argument("--sunflower-overbuild-before-defense-penalty", type=float, default=None)
    parser.add_argument("--defense-before-extra-economy-reward", type=float, default=None)
    parser.add_argument("--sunflower-while-undefended-threat-penalty", type=float, default=None)
    parser.add_argument("--plant-elsewhere-while-undefended-threat-penalty", type=float, default=None)
    parser.add_argument("--undefended-threat-grace-steps", type=float, default=None)
    parser.add_argument("--late-undefended-threat-penalty", type=float, default=None)
    parser.add_argument("--reduce-undefended-threat-reward", type=float, default=None)
    parser.add_argument("--wait-while-actionable-threat-penalty", type=float, default=None)
    parser.add_argument("--first-peashooter-threatened-row-reward", type=float, default=None)
    parser.add_argument("--all-active-threatened-rows-have-peashooter-reward", type=float, default=None)
    parser.add_argument("--sunflower-greed-while-defense-missing-penalty", type=float, default=None)
    parser.add_argument("--wallnut-blocks-active-threat-reward", type=float, default=None)
    parser.add_argument("--wallnut-low-value-placement-penalty", type=float, default=None)
    parser.add_argument("--cherrybomb-tactical-kill-reward", type=float, default=None)
    parser.add_argument("--cherrybomb-tough-bonus-reward", type=float, default=None)
    parser.add_argument("--cherrybomb-mower-save-bonus-reward", type=float, default=None)
    parser.add_argument("--cherrybomb-wasted-penalty", type=float, default=None)
    parser.add_argument("--mower-risk-reduction-reward", type=float, default=None)
    parser.add_argument("--tough-zombie-response-reward", type=float, default=None)
    parser.add_argument("--coach-match-reward", type=float, default=None)
    parser.add_argument("--coach-legal-execution-reward", type=float, default=None)
    parser.add_argument("--coach-override-penalty", type=float, default=None)
    parser.add_argument("--coach-fusion-success-reward", type=float, default=None)
    parser.add_argument("--coach-tactical-usefulness-reward", type=float, default=None)
    parser.add_argument("--fusion-attempt-reward", type=float, default=None)
    parser.add_argument("--fusion-success-reward", type=float, default=None)
    parser.add_argument("--fusion-new-recipe-reward", type=float, default=None)
    parser.add_argument("--fusion-recursive-reward", type=float, default=None)
    parser.add_argument("--fusion-tier2-reward", type=float, default=None)
    parser.add_argument("--fusion-tier3-reward", type=float, default=None)
    parser.add_argument("--fusion-repeat-reward-multiplier", type=float, default=None)
    parser.add_argument("--fusion-threatened-row-bonus", type=float, default=None)
    parser.add_argument("--fusion-active-wave-bonus", type=float, default=None)
    parser.add_argument("--fusion-defensive-value-bonus", type=float, default=None)
    parser.add_argument("--fusion-incompatible-penalty", type=float, default=None)
    parser.add_argument("--fusion-empty-tile-penalty", type=float, default=None)
    parser.add_argument("--fusion-failed-penalty", type=float, default=None)
    parser.add_argument("--fusion-bridge-error-penalty", type=float, default=None)
    parser.add_argument("--fusion-spam-penalty", type=float, default=None)
    parser.add_argument("--max-fusion-reward-per-episode", type=float, default=None)
    parser.add_argument("--checkpoint-freq", type=int)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--enable-action-watchdog", dest="enable_action_watchdog", action="store_true", default=None)
    parser.add_argument("--no-action-watchdog", dest="enable_action_watchdog", action="store_false")
    parser.add_argument("--action-timeout-seconds", type=float)
    parser.add_argument("--save-freeze-debug-bundle", dest="save_freeze_debug_bundle", action="store_true", default=None)
    parser.add_argument("--no-save-freeze-debug-bundle", dest="save_freeze_debug_bundle", action="store_false")
    parser.add_argument("--action-diagnostics-path")
    parser.add_argument("--freeze-debug-dir")
    parser.add_argument("--game-exe", dest="game_exe")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    if args.adventure_generalist_train and args.adventure_generalist_eval:
        raise SystemExit(
            "blocked_reason=invalid_cli: --adventure-generalist-train and --adventure-generalist-eval are mutually exclusive."
        )

    if args.check_deps:
        return check_deps()

    raw_config = load_json(args.config)
    config = build_config(args, raw_config)
    if args.metadata_dry_run:
        model_path = args.model or Path(config["model_path"] or Path(config["run_dir"]) / "model.zip")
        return metadata_dry_run(config, model_path)
    if bool(config.get("streamer_v1_enabled", False)):
        run_dir = Path(config["run_dir"])
        with TeeOutput(
            run_dir / "streamer.log",
            rotate_max_bytes=64 * 1024 * 1024,
            rotate_backup_count=3,
        ):
            print(f"Streamer experiment directory: {run_dir}")
            run_streamer_v1(config, args)
        return 0
    execution_route = execution_route_for_config(config)
    if execution_route == "train":
        run_dir = Path(config["run_dir"])
        with TeeOutput(run_dir / "train.log"):
            print(f"Run directory: {run_dir}")
            train(config, args.live_status_path)
        return 0
    if execution_route == "adventure_generalist_eval":
        configured_model = str(config.get("model_path") or "").strip()
        model_path = args.model or (Path(configured_model) if configured_model else Path(config["run_dir"]) / "model.zip")
        if not model_path.exists() and (Path(config["run_dir"]) / "final_model.zip").exists():
            model_path = Path(config["run_dir"]) / "final_model.zip"
        if not model_path.exists():
            raise SystemExit(f"Model not found for Adventure Generalist evaluation: {model_path}")
        with TeeOutput(Path(config["run_dir"]) / "eval.log"):
            adventure_evaluate(config, model_path, args)
        return 0
    raise SystemExit(f"blocked_reason=unsupported_run_mode: {config.get('run_mode')!r}")


if __name__ == "__main__":
    raise SystemExit(main())
