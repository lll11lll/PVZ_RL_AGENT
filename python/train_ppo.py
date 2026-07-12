"""Minimal MaskablePPO train/eval entrypoint for the limited PvZRL setup."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple

from pvzrl_adventure import (
    DEFAULT_ADVENTURE_HARD_MAX_STEPS,
    DEFAULT_ADVENTURE_SOFT_MAX_STEPS,
    run_adventure_eval,
)
from pvzrl_adventure_generalist import (
    ADVENTURE_GENERALIST_INITIAL_LOADOUT,
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
    ACTION_SPACE_DYNAMIC_14,
    ACTION_SPACE_FIXED,
    ADVENTURE_IDENTITY_ACTION_DECODER_VERSION,
    ADVENTURE_IDENTITY_ACTION_COUNT,
    ADVENTURE_IDENTITY_OBSERVATION_VERSION,
    DYNAMIC_WAIT_ACTION,
    action_count_for_config as action_space_count_for_config,
    build_action_space_spec,
    normalize_action_space_mode,
)
from pvzrl_env import REWARD_COMPONENT_FIELDS, REWARD_EPISODE_TOTAL_FIELDS, RewardConfig, parse_seed_list, resolve_seed_list
from pvzrl_env import (
    LEVEL3_SPECIALIST_PLANT_TYPES,
    LEVEL3_SPECIALIST_SEED_LIST,
    LEVEL3_SPECIALIST_TARGET_LEVEL,
    RUN_MODE_ADVENTURE_EVAL,
    RUN_MODE_LEVEL3_SPECIALIST,
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
from pvzrl_model_router import ModelRouter, stage_config
from pvzrl_sb3 import PvZMaskedPPOEnv, PvZSB3Config


DEFAULT_CONFIG_PATH = Path("configs/ppo_sunflower_peashooter.json")
LEVEL3_REWARD_DEFAULTS = {
    "early_sunflower_reward": 0.18,
    "safe_sunflower_position_reward": 0.08,
    "sunflower_overbuild_penalty": 0.12,
    "economy_collapse_penalty": 0.12,
    "first_defense_in_threatened_row_reward": 0.35,
    "threatened_lane_coverage_reward": 0.18,
    "row_balance_reward": 0.12,
    "useful_peashooter_position_reward": 0.08,
    "overdefense_penalty": 0.08,
    "wallnut_threatened_lane_reward": 0.16,
    "wallnut_between_zombie_and_house_reward": 0.18,
    "wallnut_frontline_reward": 0.12,
    "wallnut_emergency_block_reward": 0.28,
    "wallnut_useless_penalty": 0.12,
    "cherrybomb_kill_reward": 0.16,
    "cherrybomb_heavy_zombie_bonus": 0.28,
    "cherrybomb_cluster_bonus": 0.18,
    "cherrybomb_emergency_reward": 0.22,
    "cherrybomb_zero_kill_penalty": 0.25,
    "cherrybomb_low_value_penalty": 0.16,
    "row_danger_delta_reward": 0.01,
    "high_danger_unanswered_penalty": 0.025,
    "mower_exposure_penalty": 0.035,
    "minimum_viable_defense_reward": 0.06,
}
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
    "fusion_attempt_reward_total",
    "fusion_success_reward_total",
    "fusion_new_recipe_reward_total",
    "fusion_recursive_reward_total",
    "fusion_tier_reward_total",
    "fusion_repeat_decay_total",
    "fusion_threatened_row_bonus_total",
    "fusion_active_wave_bonus_total",
    "fusion_defensive_value_bonus_total",
    "fusion_incompatible_penalty_total",
    "fusion_empty_tile_penalty_total",
    "fusion_failed_penalty_total",
    "fusion_bridge_error_penalty_total",
    "fusion_spam_penalty_total",
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
    "target_level",
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


@dataclass
class EvalLog:
    policy: str
    episode: int
    run_mode: str = ""
    target_level: int = 0
    result: str = "none"
    reward_total: float = 0.0
    episode_reward: float = 0.0
    episode_length: int = 0
    terminal_reason: str = ""
    done_reason: str = "none"
    final_wave: int = 0
    max_wave: int = 0
    zombies_killed: int = 0
    plants_placed: int = 0
    sunflowers_planted: int = 0
    peashooters_planted: int = 0
    wallnuts_planted: int = 0
    cherrybombs_planted: int = 0
    sun_spent: int = 0
    sun_remaining: int = 0
    mowers_lost: int = 0
    reset_success: bool = True
    reset_seconds: float = 0.0
    avg_legal_actions: float = 0.0
    illegal_actions: int = 0
    bridge_errors: int = 0
    reset_failures: int = 0
    plants_by_row: Dict[str, int] = field(default_factory=dict)
    peashooters_by_row: Dict[str, int] = field(default_factory=dict)
    sunflowers_by_row: Dict[str, int] = field(default_factory=dict)
    threat_steps_by_row: Dict[str, int] = field(default_factory=dict)
    undefended_threat_steps_by_row: Dict[str, int] = field(default_factory=dict)
    undefended_threat_age_avg_by_row: Dict[str, float] = field(default_factory=dict)
    undefended_threat_age_max_by_row: Dict[str, int] = field(default_factory=dict)
    mower_losses_by_row: Dict[str, int] = field(default_factory=dict)
    wait_under_threat_count: int = 0
    close_zombie_undefended_count: int = 0
    illegal_reason_counts: Dict[str, int] = field(default_factory=dict)
    legal_peashooter_actions_by_row: Dict[str, int] = field(default_factory=dict)
    peashooter_available_but_waited_by_row: Dict[str, int] = field(default_factory=dict)
    peashooter_available_but_planted_elsewhere_by_row: Dict[str, int] = field(default_factory=dict)
    sunflower_while_undefended_threat_by_row: Dict[str, int] = field(default_factory=dict)
    wait_actions: int = 0
    plant_actions: int = 0
    wait_action_percent: float = 0.0
    plant_action_percent: float = 0.0
    plant_actions_by_row: Dict[str, int] = field(default_factory=dict)
    peashooter_actions_by_row: Dict[str, int] = field(default_factory=dict)
    sunflower_actions_by_row: Dict[str, int] = field(default_factory=dict)
    plant_placements_by_row: Dict[str, int] = field(default_factory=dict)
    peashooter_placements_by_row: Dict[str, int] = field(default_factory=dict)
    sunflower_placements_by_row: Dict[str, int] = field(default_factory=dict)
    row_defense_opportunities_by_row: Dict[str, int] = field(default_factory=dict)
    row_defense_responses_by_row: Dict[str, int] = field(default_factory=dict)
    undefended_threat_ratio_by_row: Dict[str, float] = field(default_factory=dict)
    row_defense_response_rate_by_row: Dict[str, float] = field(default_factory=dict)
    row_defense_response_rate: float = 0.0
    threatened_rows_with_zero_defender_steps_by_row: Dict[str, int] = field(default_factory=dict)
    peashooters_per_threat_step_by_row: Dict[str, float] = field(default_factory=dict)
    first_defense_step_by_row: Dict[str, int] = field(default_factory=dict)
    plants_in_threatened_row_count: int = 0
    plants_in_unthreatened_row_count: int = 0
    plants_in_threatened_row_ratio: float = 0.0
    plants_in_unthreatened_row_ratio: float = 0.0
    overdefended_while_undefended_count: int = 0
    least_defended_threatened_row_plant_count: int = 0
    rows_with_peashooter_count: int = 0
    all_rows_peashooter_covered_step: int = 0
    first_peashooter_by_row_step: Dict[str, int] = field(default_factory=dict)
    sunflower_count_when_first_full_coverage: int = -1
    sunflower_overbuild_before_defense_count: int = 0
    sunflower_overbuild_count: int = 0
    peashooter_coverage_rate_by_step: float = 0.0
    peashooter_coverage_rate_sum: float = 0.0
    legal_actions_by_seed_slot: Dict[str, int] = field(default_factory=dict)
    bridge_legal_actions_by_seed_slot: Dict[str, int] = field(default_factory=dict)
    python_mask_block_reason_counts: Dict[str, int] = field(default_factory=dict)
    pre_step_mask_blocked_count: int = 0
    cooldown_illegal_exposed_by_mask_count: int = 0
    mask_bridge_disagreement_count: int = 0
    kill_reward_total: float = 0.0
    wave_reward_total: float = 0.0
    win_loss_reward_total: float = 0.0
    illegal_penalty_total: float = 0.0
    mower_loss_penalty_total: float = 0.0
    danger_delta_reward_total: float = 0.0
    undefended_threat_penalty_total: float = 0.0
    lane_response_reward_total: float = 0.0
    plant_health_loss_penalty_total: float = 0.0
    threat_balanced_row_reward_total: float = 0.0
    overdefended_row_penalty_total: float = 0.0
    role_positioning_reward_total: float = 0.0
    first_peashooter_in_row_reward_total: float = 0.0
    first_defense_undefended_threatened_row_reward_total: float = 0.0
    all_rows_peashooter_coverage_reward_total: float = 0.0
    sunflower_overbuild_before_defense_penalty_total: float = 0.0
    defense_before_extra_economy_reward_total: float = 0.0
    sunflower_while_undefended_threat_penalty_total: float = 0.0
    plant_elsewhere_while_undefended_threat_penalty_total: float = 0.0
    late_undefended_threat_penalty_total: float = 0.0
    reduce_undefended_threat_reward_total: float = 0.0
    wait_while_actionable_threat_penalty_total: float = 0.0
    first_peashooter_threatened_row_reward_total: float = 0.0
    all_active_threatened_rows_have_peashooter_reward_total: float = 0.0
    sunflower_greed_while_defense_missing_penalty_total: float = 0.0
    early_sunflower_reward_total: float = 0.0
    safe_sunflower_position_reward_total: float = 0.0
    sunflower_overbuild_penalty_total: float = 0.0
    economy_collapse_penalty_total: float = 0.0
    first_defense_in_threatened_row_reward_total: float = 0.0
    threatened_lane_coverage_reward_total: float = 0.0
    row_balance_reward_total: float = 0.0
    useful_peashooter_position_reward_total: float = 0.0
    overdefense_penalty_total: float = 0.0
    wallnut_blocks_active_threat_reward_total: float = 0.0
    wallnut_low_value_placement_penalty_total: float = 0.0
    wallnut_threatened_lane_reward_total: float = 0.0
    wallnut_between_zombie_and_house_reward_total: float = 0.0
    wallnut_frontline_reward_total: float = 0.0
    wallnut_emergency_block_reward_total: float = 0.0
    wallnut_useless_penalty_total: float = 0.0
    cherrybomb_tactical_kill_reward_total: float = 0.0
    cherrybomb_wasted_penalty_total: float = 0.0
    cherrybomb_kill_reward_total: float = 0.0
    cherrybomb_heavy_zombie_bonus_total: float = 0.0
    cherrybomb_cluster_bonus_total: float = 0.0
    cherrybomb_emergency_reward_total: float = 0.0
    cherrybomb_zero_kill_penalty_total: float = 0.0
    cherrybomb_low_value_penalty_total: float = 0.0
    mower_risk_reduction_reward_total: float = 0.0
    tough_zombie_response_reward_total: float = 0.0
    row_danger_delta_reward_total: float = 0.0
    high_danger_unanswered_penalty_total: float = 0.0
    mower_exposure_penalty_total: float = 0.0
    minimum_viable_defense_reward_total: float = 0.0
    wait_while_actionable_threat_count: int = 0
    wait_while_peashooter_affordable_ready_count: int = 0
    wait_while_wallnut_affordable_ready_count: int = 0
    wait_while_cherrybomb_affordable_ready_count: int = 0
    active_threat_rows_without_peashooter_count: int = 0
    sunflower_greed_while_defense_missing_count: int = 0
    wallnut_placements_by_row: Dict[str, int] = field(default_factory=dict)
    wallnut_placements_by_col: Dict[str, int] = field(default_factory=dict)
    wallnut_blocks_active_threat_count: int = 0
    wallnut_low_value_placement_count: int = 0
    wallnut_threatened_lane_placements: int = 0
    wallnut_between_zombie_and_house_count: int = 0
    wallnut_frontline_count: int = 0
    wallnut_emergency_blocks: int = 0
    wallnut_useless_placements: int = 0
    wallnut_damage_absorbed_total: float = 0.0
    cherrybomb_used_count: int = 0
    cherrybomb_kills_total: int = 0
    cherrybomb_avg_kills_per_use: float = 0.0
    cherrybomb_zero_kill_count: int = 0
    cherrybomb_zero_kill_uses: int = 0
    cherrybomb_cluster_uses: int = 0
    cherrybomb_emergency_uses: int = 0
    cherrybomb_heavy_zombie_kills: int = 0
    cherrybomb_buckethead_kills: int = 0
    cherrybomb_conehead_kills: int = 0
    cherrybomb_used_under_threat_count: int = 0
    cherrybomb_used_low_value_count: int = 0
    mower_risk_steps_by_row: Dict[str, int] = field(default_factory=dict)
    mower_saves_estimated_by_row: Dict[str, int] = field(default_factory=dict)
    close_zombie_with_no_defense_count: int = 0
    buckethead_count_by_row: Dict[str, int] = field(default_factory=dict)
    conehead_count_by_row: Dict[str, int] = field(default_factory=dict)
    tough_zombie_count_by_row: Dict[str, int] = field(default_factory=dict)
    tough_zombie_response_count: int = 0
    undefended_threat_steps: int = 0
    high_danger_unanswered_steps: int = 0
    mower_exposure_steps: int = 0
    overdefense_count: int = 0
    mower_losses: int = 0
    legal_action_count_mean: float = 0.0
    max_row_danger: float = 0.0
    avg_row_danger: float = 0.0
    tactical_mask_enabled: bool = False
    wallnut_actions_masked: int = 0
    cherrybomb_actions_masked: int = 0
    wallnut_actions_available: int = 0
    cherrybomb_actions_available: int = 0
    mask_all_but_wait_count: int = 0
    fusion_policy: str = FUSION_POLICY_NONE
    fusion_candidate_count_total: int = 0
    fusion_candidate_count_avg: float = 0.0
    fusion_attempted_count: int = 0
    fusion_success_count: int = 0
    fusion_failed_count: int = 0
    fusion_rejected_count: int = 0
    fusion_rejected_reasons: Dict[str, int] = field(default_factory=dict)
    fusion_by_result_type: Dict[str, int] = field(default_factory=dict)
    fusion_by_source_type: Dict[str, int] = field(default_factory=dict)
    fusion_by_row: Dict[str, int] = field(default_factory=dict)
    fusion_under_threat_count: int = 0
    fusion_near_buckethead_count: int = 0
    fusion_near_conehead_count: int = 0
    fusion_estimated_mower_save_count: int = 0
    fusion_kills_after_use_total: int = 0
    fusion_avg_kills_after_use: float = 0.0
    fusion_bridge_error_count: int = 0
    fusion_unsafe_state_block_count: int = 0
    fusion_reward_total: float = 0.0
    fusion_attempt_reward_total: float = 0.0
    fusion_success_reward_total: float = 0.0
    fusion_new_recipe_reward_total: float = 0.0
    fusion_recursive_reward_total: float = 0.0
    fusion_tier_reward_total: float = 0.0
    fusion_repeat_decay_total: float = 0.0
    fusion_threatened_row_bonus_total: float = 0.0
    fusion_active_wave_bonus_total: float = 0.0
    fusion_defensive_value_bonus_total: float = 0.0
    fusion_incompatible_penalty_total: float = 0.0
    fusion_empty_tile_penalty_total: float = 0.0
    fusion_failed_penalty_total: float = 0.0
    fusion_bridge_error_penalty_total: float = 0.0
    fusion_spam_penalty_total: float = 0.0
    fusion_reward_capped: bool = False
    fusion_last_reward_delta: float = 0.0
    fusion_last_reward_reason: str = ""
    fusion_last_usefulness_bonus: float = 0.0
    fusion_last_source: str = ""
    plant_action_counts: Dict[str, int] = field(default_factory=dict)
    successful_placements_by_plant: Dict[str, int] = field(default_factory=dict)
    invalid_actions_by_plant: Dict[str, int] = field(default_factory=dict)
    fusion_attempts_by_pair: Dict[str, int] = field(default_factory=dict)
    fusion_successes_by_pair: Dict[str, int] = field(default_factory=dict)
    fusion_depth_counts: Dict[str, int] = field(default_factory=dict)
    highest_fusion_tier: int = 0
    recursive_fusion_count: int = 0
    action_freeze_count: int = 0
    mean_action_duration_seconds: float = 0.0
    max_action_duration_seconds: float = 0.0
    p95_action_duration_seconds: float = 0.0

    @property
    def won(self) -> bool:
        return self.done_reason == "win"

    @property
    def timed_out(self) -> bool:
        return self.done_reason == "timeout"

    @property
    def actual_terminal(self) -> bool:
        return self.done_reason in ("win", "loss")


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
    *,
    run_mode: Optional[str] = None,
) -> Dict[str, float]:
    defaults = asdict(RewardConfig())
    reward = {
        key: pick_reward(args, raw_config, key, float(default_value))
        for key, default_value in defaults.items()
    }
    # Backward-compatible coach reward aliases kept for existing GUI presets.
    if getattr(args, "coach_legal_execution_reward", None) is None and getattr(args, "human_coach_bonus", None) is not None:
        reward["coach_legal_execution_reward"] = float(getattr(args, "human_coach_bonus"))
    if getattr(args, "coach_match_reward", None) is None and getattr(args, "human_coach_match_bonus", None) is not None:
        reward["coach_match_reward"] = float(getattr(args, "human_coach_match_bonus"))
    if getattr(args, "coach_override_penalty", None) is None and getattr(args, "human_coach_override_penalty", None) is not None:
        reward["coach_override_penalty"] = float(getattr(args, "human_coach_override_penalty"))
    effective_run_mode = run_mode or resolve_effective_run_mode(args, raw_config)
    if effective_run_mode == RUN_MODE_LEVEL3_SPECIALIST:
        raw_reward = raw_config.get("reward", {})
        raw_reward = raw_reward if isinstance(raw_reward, dict) else {}
        for key, value in LEVEL3_REWARD_DEFAULTS.items():
            if getattr(args, key, None) is None and key not in raw_reward and key not in raw_config:
                reward[key] = float(value)
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


class TeeOutput:
    def __init__(self, path: Path):
        self.path = path
        self.handle: Optional[TextIO] = None
        self.old_stdout: Optional[TextIO] = None
        self.old_stderr: Optional[TextIO] = None

    def __enter__(self) -> "TeeOutput":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a", encoding="utf-8")
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


def normalize_run_part(value: str) -> str:
    normalized = "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")
    return "_".join(part for part in normalized.split("_") if part) or "seed"


def model_family_for_seed_list(seed_list: List[str]) -> str:
    unique_parts: List[str] = []
    for seed in seed_list:
        part = normalize_run_part(seed)
        if part not in unique_parts:
            unique_parts.append(part)
    return f"ppo_{len(seed_list)}slot_{'_'.join(unique_parts)}"


def default_train_run_dir(seed_list: List[str]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{model_family_for_seed_list(seed_list)}_{timestamp}"
    return Path("runs") / name


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
        "action_space_mode": str(config.get("action_space_mode", ACTION_SPACE_FIXED)),
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
        allow_missing_model_metadata=bool(config.get("allow_missing_model_metadata", False)),
    )


def raise_if_incompatible(result: CompatibilityCheck) -> None:
    if not result.ok:
        raise SystemExit(format_compatibility_failure(result))


def print_compatibility_report(prefix: str, result: CompatibilityCheck) -> None:
    print(f"{prefix} model_compatibility=" + json.dumps(result.to_dict(), separators=(",", ":"), sort_keys=True))


def validate_model_seed_slots(
    model_path: Path,
    config: Dict[str, Any],
    context: str,
    *,
    env_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result = validate_model_metadata(
        model_path,
        config,
        env_metadata=env_metadata or env_metadata_for_config(config),
        allow_missing_model_metadata=bool(config.get("allow_missing_model_metadata", False)),
    )
    print_compatibility_report(f"[compat:{context}]", result)
    raise_if_incompatible(result)
    return compatibility_summary_from_report(result)


def validate_loaded_model_compatibility(
    model: Any,
    model_path: Path,
    config: Dict[str, Any],
    context: str,
    *,
    env_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result = loaded_model_compatibility_report(
        model,
        model_path,
        config,
        env_metadata=env_metadata or env_metadata_for_config(config),
    )
    print_compatibility_report(f"[compat:{context}]", result)
    raise_if_incompatible(result)
    return compatibility_summary_from_report(result)


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
        allow_missing_model_metadata=bool(config.get("allow_missing_model_metadata", False)),
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
    """Resolve mutually exclusive run modes with all explicit CLI forms first."""

    specialized_cli_modes: List[str] = []
    if getattr(args, "level3_train", False) or getattr(args, "level3_eval", False):
        specialized_cli_modes.append(RUN_MODE_LEVEL3_SPECIALIST)
    if getattr(args, "adventure_generalist_train", False):
        specialized_cli_modes.append(ADVENTURE_GENERALIST_RUN_MODE_TRAIN)
    if getattr(args, "adventure_generalist_eval", False):
        specialized_cli_modes.append(ADVENTURE_GENERALIST_RUN_MODE_EVAL)
    if getattr(args, "adventure_eval", False) or getattr(args, "adventure", False):
        specialized_cli_modes.append(RUN_MODE_ADVENTURE_EVAL)
    distinct_specialized_modes = list(dict.fromkeys(specialized_cli_modes))
    if len(distinct_specialized_modes) > 1:
        raise SystemExit(
            "blocked_reason=invalid_cli: conflicting explicit run-mode shortcuts: "
            + ",".join(distinct_specialized_modes)
        )

    explicit_run_mode = str(getattr(args, "run_mode", "") or "").strip().lower()
    shortcut_mode = distinct_specialized_modes[0] if distinct_specialized_modes else ""
    if not shortcut_mode and not explicit_run_mode:
        if bool(getattr(args, "train", False)):
            shortcut_mode = "fixed_train"
        elif bool(getattr(args, "eval", False)):
            shortcut_mode = "fixed_eval"

    if explicit_run_mode and shortcut_mode and explicit_run_mode != shortcut_mode:
        raise SystemExit(
            "blocked_reason=invalid_cli: --run-mode conflicts with explicit mode shortcut: "
            f"{explicit_run_mode}!={shortcut_mode}"
        )
    if explicit_run_mode == "fixed_eval" and bool(getattr(args, "train", False)):
        raise SystemExit("blocked_reason=invalid_cli: fixed_eval cannot be combined with --train.")
    if explicit_run_mode == "fixed_train" and bool(getattr(args, "eval", False)):
        raise SystemExit("blocked_reason=invalid_cli: fixed_train cannot be combined with --eval.")
    if explicit_run_mode:
        return explicit_run_mode
    if shortcut_mode:
        return shortcut_mode

    configured_run_mode = str(raw_config.get("run_mode", "") or "").strip().lower()
    if configured_run_mode in {
        "fixed_train",
        "fixed_eval",
        RUN_MODE_ADVENTURE_EVAL,
        RUN_MODE_LEVEL3_SPECIALIST,
        ADVENTURE_GENERALIST_RUN_MODE_TRAIN,
        ADVENTURE_GENERALIST_RUN_MODE_EVAL,
    }:
        return configured_run_mode
    if configured_run_mode:
        raise SystemExit(f"blocked_reason=invalid_config_run_mode:{configured_run_mode}")
    return "fixed_train"


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
        or getattr(args, "train", False)
        or getattr(args, "eval", False)
        or getattr(args, "level3_train", False)
        or getattr(args, "level3_eval", False)
        or getattr(args, "adventure", False)
        or getattr(args, "adventure_eval", False)
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
    level3_requested = run_mode == RUN_MODE_LEVEL3_SPECIALIST
    adventure_generalist_train_requested = run_mode == ADVENTURE_GENERALIST_RUN_MODE_TRAIN
    adventure_generalist_eval_requested = run_mode == ADVENTURE_GENERALIST_RUN_MODE_EVAL
    adventure_generalist_requested = adventure_generalist_train_requested or adventure_generalist_eval_requested
    raw_initial_loadout = value("initial_loadout", ",".join(ADVENTURE_GENERALIST_INITIAL_LOADOUT))
    initial_loadout = parse_initial_loadout(raw_initial_loadout)
    if adventure_generalist_requested:
        default_seed_list = ",".join(initial_loadout)
    elif level3_requested:
        default_seed_list = ",".join(LEVEL3_SPECIALIST_SEED_LIST)
    else:
        default_seed_list = "SunFlower,Peashooter"
    cli_seed_list = getattr(args, "seed_list", None)
    config_seed_list_present = "seed_list" in raw_config and raw_config.get("seed_list") not in (None, "")
    seed_order_source = SEED_ORDER_SOURCE_DEFAULT
    if adventure_generalist_requested and cli_seed_list is None and not config_seed_list_present:
        raw_seed_list = default_seed_list
        resolver.sources["seed_list"] = ConfigSource.MODE_DEFAULT
    else:
        raw_seed_list = value(
            "seed_list",
            "SunFlower,Peashooter",
            mode_default=(
                default_seed_list
                if adventure_generalist_requested or level3_requested
                else CONFIG_UNSET
            ),
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
        mode_default=list(LEVEL3_SPECIALIST_PLANT_TYPES) if level3_requested else CONFIG_UNSET,
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
    requested_action_space_mode = normalize_action_space_mode(
        value("action_space_mode", ACTION_SPACE_FIXED)
    )
    experimental_dynamic = enabled("experimental_dynamic_seed_slots")
    if adventure_generalist_requested:
        requested_action_space_mode = ACTION_SPACE_ADVENTURE_14_IDENTITY
        plant_types = list(derived_plant_types)
        experimental_dynamic = False
    elif experimental_dynamic:
        requested_action_space_mode = ACTION_SPACE_DYNAMIC_14
    elif requested_action_space_mode == ACTION_SPACE_DYNAMIC_14:
        raise SystemExit(
            "blocked_reason=experimental_dynamic_seed_slots_required: "
            "dynamic_14 requires --experimental-dynamic-seed-slots."
        )
    if adventure_generalist_train_requested and args.run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = str(Path("runs") / f"{ADVENTURE_GENERALIST_MODEL_FAMILY}_{timestamp}")
    elif args.train and args.run_dir is None:
        run_dir = str(default_train_run_dir(seed_list))
    elif (
        run_mode in {"fixed_eval", RUN_MODE_ADVENTURE_EVAL, ADVENTURE_GENERALIST_RUN_MODE_EVAL}
        and args.run_dir is None
        and getattr(args, "model", None) is not None
    ):
        run_dir = str(Path(args.model).resolve().parent)
    else:
        run_dir = str(value("run_dir", "runs/ppo_sunflower_peashooter"))

    game_speed = float(value("game_speed", 4.0))
    game_speed_mode = str(value("game_speed_mode", "game_speed"))
    if args.valid_speed_mode:
        game_speed_mode = "safe"
    fusion_policy = normalize_fusion_policy(
        value("fusion_policy", FUSION_POLICY_NONE)
    )
    adventure_requested = run_mode == RUN_MODE_ADVENTURE_EVAL

    target_level = int(
        value(
            "target_level",
            0,
            mode_default=LEVEL3_SPECIALIST_TARGET_LEVEL if run_mode == RUN_MODE_LEVEL3_SPECIALIST else CONFIG_UNSET,
        )
    )
    if run_mode == RUN_MODE_LEVEL3_SPECIALIST:
        if target_level != LEVEL3_SPECIALIST_TARGET_LEVEL:
            raise SystemExit("blocked_reason=invalid_level3_target_level: Level 3 specialist requires --target-level 3.")
        if list(seed_list) != list(LEVEL3_SPECIALIST_SEED_LIST):
            raise SystemExit(
                "blocked_reason=invalid_level3_seed_list: "
                f"expected {','.join(LEVEL3_SPECIALIST_SEED_LIST)} got {','.join(seed_list)}"
            )
        if [int(value) for value in plant_types] != list(LEVEL3_SPECIALIST_PLANT_TYPES):
            raise SystemExit(
                "blocked_reason=invalid_level3_plant_types: "
                f"expected {','.join(str(value) for value in LEVEL3_SPECIALIST_PLANT_TYPES)} got "
                f"{','.join(str(value) for value in plant_types)}"
            )
        if requested_action_space_mode != ACTION_SPACE_FIXED or experimental_dynamic:
            raise SystemExit("blocked_reason=invalid_level3_action_space: Level 3 specialist requires fixed 4-slot action space.")
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

    legacy_max_steps = int(value("max_steps", 1000))
    raw_adventure_soft = getattr(args, "adventure_soft_max_steps", None)
    if raw_adventure_soft is None:
        if run_mode in {RUN_MODE_ADVENTURE_EVAL, ADVENTURE_GENERALIST_RUN_MODE_TRAIN, ADVENTURE_GENERALIST_RUN_MODE_EVAL} and getattr(args, "max_steps", None) is not None:
            raw_adventure_soft = getattr(args, "max_steps")
        else:
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
    adventure_run_mode = run_mode in {
        RUN_MODE_ADVENTURE_EVAL,
        ADVENTURE_GENERALIST_RUN_MODE_TRAIN,
        ADVENTURE_GENERALIST_RUN_MODE_EVAL,
    }
    env_max_steps = adventure_hard_max_steps if adventure_run_mode else legacy_max_steps
    requested_eval_model_path = str(args.model or raw_config.get("model_path", "") or "").strip()
    requested_resume_model_path = str(getattr(args, "resume_model_path", None) or raw_config.get("resume_model_path", "") or "").strip()
    if requested_resume_model_path and requested_eval_model_path and requested_resume_model_path != requested_eval_model_path:
        raise SystemExit(
            "blocked_reason=invalid_cli: --resume-model-path conflicts with --model-path. "
            "Provide one path or make them match."
        )
    requested_model_path = requested_resume_model_path or requested_eval_model_path
    if requested_resume_model_path and run_mode in {"fixed_eval", RUN_MODE_ADVENTURE_EVAL, ADVENTURE_GENERALIST_RUN_MODE_EVAL}:
        raise SystemExit("blocked_reason=invalid_cli: --resume-model-path is training-only.")
    requested_training_continuation = bool(requested_model_path) and run_mode not in {
        "fixed_eval",
        RUN_MODE_ADVENTURE_EVAL,
        ADVENTURE_GENERALIST_RUN_MODE_EVAL,
    }
    randomize_seed_order = enabled(
        "randomize_seed_order",
        json_aliases=("seed_order_randomization",),
    )
    if adventure_generalist_requested and randomize_seed_order:
        seed_order_source = SEED_ORDER_SOURCE_RANDOMIZED
    coach_allow_fusion_planning = enabled("coach_allow_fusion_planning")
    fusion_bridge_enabled = enabled("fusion_bridge_enabled")
    human_coach_enabled = enabled("human_coach_enabled")
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
        "model_family": (
            ADVENTURE_GENERALIST_MODEL_FAMILY
            if adventure_generalist_requested
            else str(raw_config.get("model_family", model_family_for_seed_list(seed_list)))
        ),
        "total_timesteps": int(value("total_timesteps", 25000)),
        "learning_rate": float(value("learning_rate", 3e-4)),
        "n_steps": int(value("n_steps", 512)),
        "batch_size": int(value("batch_size", 64)),
        "gamma": float(value("gamma", 0.99)),
        "gae_lambda": float(value("gae_lambda", 0.95)),
        "ent_coef": float(value("ent_coef", 0.01)),
        "clip_range": float(value("clip_range", 0.2)),
        "verbose": int(value("verbose", 1)),
        "max_steps": int(env_max_steps),
        "legacy_max_steps": int(legacy_max_steps),
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
        "experimental_dynamic_seed_slots": bool(experimental_dynamic),
        "max_seed_slots": int(
            value(
                "max_seed_slots",
                len(plant_types),
                mode_default=14 if adventure_generalist_requested else CONFIG_UNSET,
            )
        ),
        "auto_select_seeds": bool(enabled("auto_select_seeds") or adventure_generalist_requested),
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
        "fusion_action_mask_enabled": enabled("fusion_action_mask_enabled"),
        "enable_board_plant_identity": bool(
            value("enable_board_plant_identity", False)
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
        "target_level": int(target_level),
        "tactical_masks": enabled("tactical_masks"),
        "wallnut_tactical_mask": enabled("wallnut_tactical_mask"),
        "cherrybomb_tactical_mask": enabled("cherrybomb_tactical_mask"),
        "adventure_eval_mode": bool(adventure_run_mode),
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
        "incompatible_with_4slot_specialist": bool(adventure_generalist_requested),
        "active_seed_slots_at_start": len(initial_loadout) if adventure_generalist_requested else len(seed_list),
        "unlock_aware_seed_curriculum": bool(
            enabled("unlock_aware_seed_curriculum") or adventure_generalist_requested
        ),
        "seed_curriculum": str(value("seed_curriculum", "conservative")),
        "unlock_introduction_delay": int(value("unlock_introduction_delay", 0)),
        "new_plant_min_inclusion_prob": float(value("new_plant_min_inclusion_prob", 0.15)),
        "infer_capacity_from_unlocks": bool(
            value(
                "infer_capacity_from_unlocks",
                False,
                mode_default=True if adventure_generalist_requested else CONFIG_UNSET,
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
        "allow_missing_model_metadata": enabled("allow_missing_model_metadata"),
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
        "reward": build_reward_config(args, raw_config, run_mode=run_mode),
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
    if run_mode == RUN_MODE_LEVEL3_SPECIALIST and action_count_for_config(config) != 201:
        raise SystemExit(
            f"blocked_reason=invalid_level3_action_count: expected 201 got {action_count_for_config(config)}"
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
        action_space_mode=str(config.get("action_space_mode", ACTION_SPACE_FIXED)),
        max_seed_slots=int(config.get("max_seed_slots", len(config["plant_types"]))),
        observation_version=str(config.get("observation_version", "")),
        action_decoder_version=str(config.get("action_decoder_version", "")),
        dynamic_seed_slots=bool(config.get("dynamic_seed_slots", False)),
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
        run_mode=str(config.get("run_mode", "fixed_train")),
        target_level=int(config.get("target_level", 0) or 0),
        tactical_masks=bool(config.get("tactical_masks", False)),
        wallnut_tactical_mask=bool(config.get("wallnut_tactical_mask", False)),
        cherrybomb_tactical_mask=bool(config.get("cherrybomb_tactical_mask", False)),
        adventure_eval_mode=bool(config.get("adventure_eval_mode", False)),
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


def write_action_map(config: Dict[str, Any], path: Path) -> None:
    rows = 5
    cols = 10
    plant_types = list(config["plant_types"])
    seed_list = list(config.get("seed_list", []))
    spec = build_action_space_spec(
        mode=str(config.get("action_space_mode", ACTION_SPACE_FIXED)),
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
        if spec.mode == ACTION_SPACE_DYNAMIC_14:
            for slot_index in range(spec.max_seed_slots):
                plant_type = plant_types[slot_index] if slot_index < len(plant_types) else -1
                name = seed_list[slot_index] if slot_index < len(seed_list) else str(plant_type)
                for row in range(rows):
                    for col in range(cols):
                        action = slot_index * rows * cols + row * cols + col
                        handle.write(f"{action} slot={slot_index} plant={name} plant_type={plant_type} row={row} col={col}\n")
            handle.write(f"{DYNAMIC_WAIT_ACTION} wait\n")
            return
        if spec.mode == ACTION_SPACE_ADVENTURE_14_IDENTITY:
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
            return
        handle.write("0 wait\n")
        for slot_index, plant_type in enumerate(plant_types):
            name = seed_list[slot_index] if slot_index < len(seed_list) else str(plant_type)
            for row in range(rows):
                for col in range(cols):
                    action = 1 + slot_index * rows * cols + row * cols + col
                    handle.write(f"{action} slot={slot_index} plant={name} plant_type={plant_type} row={row} col={col}\n")


def clean_episode_row(summary: Dict[str, Any], fallback_episode: int) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    for field in EPISODE_METRIC_FIELDS:
        row[field] = summary.get(field)
    row["run_mode"] = str(row.get("run_mode") or "")
    row["target_level"] = int(row.get("target_level") or 0)
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


def sum_count_dicts_from_rows(rows: List[Dict[str, Any]], field_name: str) -> Dict[str, int]:
    totals: Counter[str] = Counter()
    for row in rows:
        values = normalize_count_dict(row.get(field_name))
        for key, value in values.items():
            totals[str(key)] += int(value)
    return dict(sorted(totals.items(), key=lambda item: (0, int(item[0])) if item[0].isdigit() else (1, item[0])))


def max_count_dicts_from_rows(rows: List[Dict[str, Any]], field_name: str) -> Dict[str, int]:
    totals: Counter[str] = Counter()
    for row in rows:
        values = normalize_count_dict(row.get(field_name))
        for key, value in values.items():
            totals[str(key)] = max(int(totals.get(str(key), 0)), int(value))
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


def weighted_average_dict_from_rows(rows: List[Dict[str, Any]], field_name: str, weight_field: str) -> Dict[str, float]:
    totals: Counter[str] = Counter()
    weights: Counter[str] = Counter()
    for row in rows:
        values = normalize_float_dict(row.get(field_name))
        row_weights = normalize_count_dict(row.get(weight_field))
        for key, value in values.items():
            weight = int(row_weights.get(str(key), 0) or 0)
            if weight <= 0:
                continue
            totals[str(key)] += float(value) * weight
            weights[str(key)] += weight
    keys = sorted(set(totals.keys()) | set(weights.keys()), key=lambda item: (0, int(item)) if item.isdigit() else (1, item))
    return {
        key: float(totals.get(key, 0.0)) / float(weights.get(key, 0))
        if weights.get(key, 0) > 0
        else 0.0
        for key in keys
    }


def average_positive_from_rows(rows: List[Dict[str, Any]], field_name: str) -> float:
    values: List[float] = []
    for row in rows:
        try:
            value = float(row.get(field_name, 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if value > 0.0:
            values.append(value)
    return sum(values) / len(values) if values else 0.0


def average_positive_step_dict_from_rows(rows: List[Dict[str, Any]], field_name: str) -> Dict[str, float]:
    totals: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    for row in rows:
        values = normalize_count_dict(row.get(field_name))
        for key, value in values.items():
            if int(value) > 0:
                totals[str(key)] += int(value)
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


def write_eval_placeholder(config: Dict[str, Any], run_dir: Path, model_path: Path) -> None:
    eval_flag = "--level3-eval --target-level 3 " if str(config.get("run_mode", "")) == RUN_MODE_LEVEL3_SPECIALIST else "--eval "
    eval_command = (
        f"python .\\python\\train_ppo.py {eval_flag}"
        f"--model-path {model_path} "
        "--episodes 10 "
        f"--game-speed {config['game_speed']} "
        "--quick-wait --wait-gameplay-ready --auto-select-seeds "
        f"--seed-list {','.join(config['seed_list'])} "
        f"--fusion-policy {config.get('fusion_policy', FUSION_POLICY_NONE)}"
    )
    payload = {"status": "not_run", "eval_command": eval_command}
    (run_dir / "eval_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.{time.monotonic_ns()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    for attempt in range(20):
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError:
            time.sleep(0.025 + attempt * 0.005)
    try:
        tmp_path.unlink(missing_ok=True)
    except OSError:
        pass
    raise PermissionError(f"Could not replace live status file after retries: {path}")


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


def write_eval_live_status(
    live_status_path: Optional[Path],
    *,
    config: Dict[str, Any],
    model_path: Path,
    report: CompatibilityCheck,
    status: str,
    mode: str = "fixed_eval",
    summary: Optional[Dict[str, Any]] = None,
) -> None:
    if live_status_path is None:
        return
    live_status_path = resolved_live_status_path(live_status_path)
    if live_status_path is None:
        return
    if str(config.get("run_mode", "")) == RUN_MODE_LEVEL3_SPECIALIST and mode == "fixed_eval":
        mode = RUN_MODE_LEVEL3_SPECIALIST
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
    payload = {
        "mode": mode,
        "run_mode": str(config.get("run_mode", mode)),
        "target_level": int(config.get("target_level", 0) or 0),
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
    }
    write_json_atomic(live_status_path, payload)


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
) -> None:
    if live_status_path is None:
        return
    summary = summary or {}
    observation = observation or {}
    live_status_path = resolved_live_status_path(live_status_path)
    if live_status_path is None:
        return
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
    payload = {
        "mode": mode,
        "run_mode": str(config.get("run_mode", mode)),
        "status": status,
        "health": "LIVE" if status == "running" else ("DEAD" if status in {"complete", "blocked"} else status.upper()),
        "updated_at": time.time(),
        "target_level": int(config.get("target_level", 0) or 0),
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
    }
    write_json_atomic(live_status_path, payload)


def verify_level3_specialist_start_state(config: Dict[str, Any], live_status_path: Optional[Path], model_path: Optional[Path] = None) -> None:
    if str(config.get("run_mode", "")) != RUN_MODE_LEVEL3_SPECIALIST:
        return
    env = PvZMaskedPPOEnv(make_env_config(config))
    try:
        state = env.base.level3_specialist_start_state()
    finally:
        close_env_safely(env, timeout_seconds=5.0)
    if bool(state.get("ok")):
        write_runtime_live_status(
            live_status_path,
            config=config,
            status="running",
            mode=RUN_MODE_LEVEL3_SPECIALIST,
            model_path=model_path,
            summary={"level3_start_state": state},
            observation=state.get("observation") if isinstance(state.get("observation"), dict) else {},
        )
        return
    write_runtime_live_status(
        live_status_path,
        config=config,
        status="blocked",
        mode=RUN_MODE_LEVEL3_SPECIALIST,
        model_path=model_path,
        summary={"level3_start_state": state},
        observation=state.get("observation") if isinstance(state.get("observation"), dict) else {},
        blocked_reason="not_at_level3_specialist_start_state",
    )
    raise SystemExit("blocked_reason=not_at_level3_specialist_start_state")


def train(config: Dict[str, Any], live_status_path: Optional[Path] = None) -> None:
    MaskablePPO = require_maskable_ppo()
    BaseCallback, CallbackList, CheckpointCallback, DummyVecEnv, _ = require_sb3_callbacks()

    live_status_path = resolved_live_status_path(live_status_path)
    if live_status_path is not None:
        print(f"[train] live_status_path={live_status_path}")
    run_dir = Path(config["run_dir"])
    initial_model_path = Path(str(config.get("model_path") or ""))
    requested_continuation = bool(str(config.get("model_path") or "").strip())
    generalist_training = str(config.get("run_mode", "")) == ADVENTURE_GENERALIST_RUN_MODE_TRAIN
    config["resume_training"] = bool(requested_continuation)
    config["resume_model_path"] = str(initial_model_path) if requested_continuation else ""
    config["resume_source_model_family"] = str(config.get("resume_source_model_family") or "")

    def _safe_resolve(path: Path) -> Path:
        try:
            return path.resolve()
        except OSError:
            return path

    if requested_continuation and generalist_training:
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
            self.csv_path = csv_path
            self.jsonl_path = jsonl_path
            self.rows: List[Dict[str, Any]] = []
            self.performance = PerformanceAccumulator()
            self.csv_fieldnames = ensure_progress_csv_fieldnames(csv_path, EPISODE_METRIC_FIELDS)
            self._header_written = csv_path.exists() and csv_path.stat().st_size > 0

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            pvz_env = unwrap_pvz_env(runtime_env)
            for info in infos:
                perf = info.get("performance")
                if isinstance(perf, dict):
                    self.performance.add(perf)
                raw_observation = info.get("raw_observation") if isinstance(info, dict) else {}
                if isinstance(raw_observation, dict) and str(config.get("run_mode", "")) != ADVENTURE_GENERALIST_RUN_MODE_TRAIN:
                    write_runtime_live_status(
                        live_status_path,
                        config=config,
                        status="running",
                        mode=str(config.get("run_mode", "fixed_train")),
                        model_path=Path(str(config.get("model_path") or run_dir / "model.zip")),
                        observation=raw_observation,
                        summary={
                            "episode": getattr(pvz_env, "_episode_index", None),
                            "episode_length": getattr(pvz_env, "_step_count", None),
                            "episode_reward": getattr(pvz_env, "_episode_reward", None),
                            "total_timesteps": int(getattr(self, "num_timesteps", 0) or 0),
                            "callback_steps": int(getattr(self, "n_calls", 0) or 0),
                        },
                    )
            rows = []
            for info in infos:
                summary = info.get("episode_summary")
                if summary:
                    rows.append(clean_episode_row(summary, len(self.rows)))
            if rows:
                self._header_written, self.csv_fieldnames = write_progress_csv_rows(
                    self.csv_path,
                    rows,
                    self.csv_fieldnames,
                    self._header_written,
                )
                with self.jsonl_path.open("a", encoding="utf-8") as jsonl_handle:
                    for row in rows:
                        jsonl_handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                self.rows.extend(rows)
                if rows and str(config.get("run_mode", "")) != ADVENTURE_GENERALIST_RUN_MODE_TRAIN:
                    write_runtime_live_status(
                        live_status_path,
                        config=config,
                        status="running",
                        mode=str(config.get("run_mode", "fixed_train")),
                        model_path=Path(str(config.get("model_path") or run_dir / "model.zip")),
                        summary=rows[-1],
                    )
            return True

    vec_env = DummyVecEnv([lambda: make_monitored_env(config, run_dir / "monitor.csv", live_status_path=live_status_path)])
    runtime_env = vec_env.envs[0] if getattr(vec_env, "envs", None) else None
    runtime_env_metadata = env_metadata_for_config(config, runtime_env)
    continuing = requested_continuation and initial_model_path.exists()
    if continuing:
        if generalist_training:
            print(f"[adventure-generalist] model_path_load_requested={initial_model_path}")
            print("[adventure-generalist] loading prior PPO model for continued training")
            compatibility = validate_adventure_generalist_model_compatibility(
                initial_model_path,
                config,
                "Adventure Generalist training continuation metadata",
                env_metadata=runtime_env_metadata,
            )
            config["resume_source_model_family"] = str(compatibility.get("model_family") or ADVENTURE_GENERALIST_MODEL_FAMILY)
        else:
            validate_model_seed_slots(
                initial_model_path,
                config,
                "Training continuation",
                env_metadata=runtime_env_metadata,
            )
        print(f"Continuing PPO model from {initial_model_path}")
        model = MaskablePPO.load(
            str(initial_model_path),
            env=vec_env,
            tensorboard_log=str(run_dir / "tensorboard"),
        )
        model.verbose = config["verbose"]
        if generalist_training:
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
        else:
            validate_loaded_model_compatibility(
                model,
                initial_model_path,
                config,
                "Training continuation",
                env_metadata=runtime_env_metadata,
            )
            config["warm_start_used"] = True
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
        )

    experiment_callback = ExperimentCallback(run_dir / "episode_metrics.csv", run_dir / "episode_metrics.jsonl")
    callbacks = CallbackList(
        [
            CheckpointCallback(
                save_freq=max(1, config["checkpoint_freq"]),
                save_path=str(run_dir / "checkpoints"),
                name_prefix="ppo_pvz",
            ),
            experiment_callback,
        ]
    )
    write_runtime_live_status(
        live_status_path,
        config=config,
        status="running",
        mode=str(config.get("run_mode", "fixed_train")),
        model_path=Path(str(config.get("model_path") or run_dir / "model.zip")),
        summary={
            "episode": None,
            "episode_length": None,
            "episode_reward": None,
            "total_timesteps": int(getattr(model, "num_timesteps", 0) or 0),
        },
    )
    if generalist_training and bool(config.get("adventure_generalist_strict_startup_validation", True)):
        pvz_env = unwrap_pvz_env(runtime_env)
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
    if generalist_training:
        print(
            "[adventure-generalist] "
            f"sb3_learn_reset_num_timesteps={'false' if continuing else 'true'}"
        )
    started = time.perf_counter()
    try:
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=callbacks,
            reset_num_timesteps=not continuing,
        )
    finally:
        vec_env.close()
    elapsed = max(1e-6, time.perf_counter() - started)
    timesteps = int(getattr(model, "num_timesteps", config["total_timesteps"]))
    fps_avg = timesteps / elapsed
    model.save(str(run_dir / "model"))
    model_path = run_dir / "model.zip"
    final_model_path = run_dir / "final_model.zip"
    if model_path.exists():
        shutil.copyfile(model_path, final_model_path)
    write_model_metadata(run_dir, config, model_path=model_path, config_path=config_path)
    summary = summarize_episode_rows(
        getattr(experiment_callback, "rows", []),
        timesteps,
        run_dir,
        model_path,
        fps_avg,
    )
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if config.get("debug_performance"):
        perf_summary = experiment_callback.performance.summary(fps_avg)
        (run_dir / "performance_summary.json").write_text(json.dumps(perf_summary, indent=2), encoding="utf-8")
    write_eval_placeholder(config, run_dir, model_path)
    write_runtime_live_status(
        live_status_path,
        config=config,
        status="complete",
        mode=str(config.get("run_mode", "fixed_train")),
        model_path=model_path,
        summary=summary,
    )
    print_validation_summary(summary)
    print(f"Saved final model to {model_path}")
    print(f"Saved compatibility model copy to {final_model_path}")
    print(f"Saved monitor log to {run_dir / 'monitor.csv'}")
    print(f"Saved episode metrics to {run_dir / 'episode_metrics.csv'}")


def evaluate(config: Dict[str, Any], model_path: Path, episodes: int, live_status_path: Optional[Path] = None) -> None:
    MaskablePPO = require_maskable_ppo()
    model = MaskablePPO.load(str(model_path))
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval] policy=ppo model_path={model_path}")
    env_metadata = env_metadata_for_config(config)
    report = loaded_model_compatibility_report(model, model_path, config, env_metadata=env_metadata)
    print_compatibility_report("[compat:Fixed-level eval]", report)
    write_eval_live_status(
        live_status_path,
        config=config,
        model_path=model_path,
        report=report,
        status="blocked" if not report.ok else "running",
    )
    raise_if_incompatible(report)
    compatibility = compatibility_summary_from_report(report)
    print(
        "[eval] compatible_seed_slots=True "
        f"seed_list={compatibility['expected_seed_list']} "
        f"plant_types={compatibility['expected_plant_types']} "
        f"decoder={compatibility['action_decoder_version']} "
        f"observation={compatibility['observation_version']} "
        f"metadata={compatibility['metadata_path']}"
    )

    results: Dict[str, List[EvalLog]] = {"ppo": []}
    ppo_steps = 0
    env = PvZMaskedPPOEnv(make_env_config(config))
    policy_started = time.perf_counter()
    try:
        for episode in range(episodes):
            log = run_eval_episode(env, episode, model)
            results["ppo"].append(log)
            ppo_steps += log.episode_length
            partial_summary = validation_summary_from_logs(
                results["ppo"],
                config["total_timesteps"],
                run_dir,
                model_path,
                fps_avg=ppo_steps / max(1e-6, time.perf_counter() - policy_started),
            )
            write_eval_live_status(
                live_status_path,
                config=config,
                model_path=model_path,
                report=report,
                status="running",
                summary=partial_summary,
            )
    finally:
        close_env_safely(env, timeout_seconds=5.0)

    print_eval_table(results)
    print_eval_diagnostics(results)
    write_eval_outputs(run_dir, results, config, model_path)
    ppo_logs = results.get("ppo", [])
    ppo_elapsed = max(1e-6, time.perf_counter() - policy_started)
    fps_avg = ppo_steps / ppo_elapsed
    summary = validation_summary_from_logs(ppo_logs, config["total_timesteps"], run_dir, model_path, fps_avg=fps_avg)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_eval_live_status(
        live_status_path,
        config=config,
        model_path=model_path,
        report=report,
        status="complete",
        summary=summary,
    )
    print_validation_summary(summary)
    print(f"[eval] completed episodes={len(ppo_logs)} exiting")


def adventure_evaluate(config: Dict[str, Any], model_path: Path, args: argparse.Namespace) -> None:
    MaskablePPO = require_maskable_ppo()
    if args.model_schedule is not None:
        router = ModelRouter.from_file(args.model_schedule)

        def load_stage(stage: Any) -> Dict[str, Any]:
            selected_config = stage_config(config, stage)
            if not stage.model_path.exists():
                raise SystemExit(f"blocked_reason=model_path_missing: {stage.model_path}")
            stage_model = MaskablePPO.load(str(stage.model_path))
            stage_env_metadata = env_metadata_for_config(selected_config)
            report = loaded_model_compatibility_report(
                stage_model,
                stage.model_path,
                selected_config,
                env_metadata=stage_env_metadata,
            )
            print_compatibility_report(f"[compat:Adventure router stage {stage.stage_id}]", report)
            if not report.ok:
                write_eval_live_status(
                    args.live_status_path,
                    config=selected_config,
                    model_path=stage.model_path,
                    report=report,
                    status="blocked",
                    mode="adventure_eval",
                )
            raise_if_incompatible(report)
            compatibility = compatibility_summary_from_report(report)
            selected_config["metadata_path"] = report.metadata_path
            selected_config["metadata_inferred"] = bool(report.metadata_inferred)
            selected_config["model_compatibility"] = model_compatibility_live_status(report)
            return {
                "model": stage_model,
                "model_path": stage.model_path,
                "config": selected_config,
                "env_config": make_env_config(selected_config),
                "compatibility": compatibility,
                "model_compatibility": selected_config["model_compatibility"],
            }

        print(f"[adventure] router_schedule={args.model_schedule}")
        run_adventure_eval(
            config=config,
            env_config=make_env_config(config),
            model=None,
            model_path=model_path,
            deterministic=bool(args.deterministic),
            advance_on_wins=max(1, int(config["advance_on_wins"])),
            max_adventure_levels=max(1, int(config["max_adventure_levels"])),
            max_attempts_per_level=max(1, int(config["max_attempts_per_level"])),
            adventure_start_level=max(1, int(config["adventure_start_level"])),
            conservative_seeds=bool(args.conservative_seeds),
            allow_new_plants=bool(args.allow_new_plants),
            live_status_path=args.live_status_path,
            gui=bool(args.gui),
            model_router=router,
            router_stage_loader=load_stage,
            adventure_soft_max_steps=int(config.get("adventure_soft_max_steps", DEFAULT_ADVENTURE_SOFT_MAX_STEPS)),
            adventure_hard_max_steps=int(config.get("adventure_hard_max_steps", DEFAULT_ADVENTURE_HARD_MAX_STEPS)),
            adventure_final_wave_extension=bool(config.get("adventure_final_wave_extension", True)),
        )
        return

    model = MaskablePPO.load(str(model_path))
    env_metadata = env_metadata_for_config(config)
    report = loaded_model_compatibility_report(model, model_path, config, env_metadata=env_metadata)
    print_compatibility_report("[compat:Adventure eval]", report)
    if not report.ok:
        write_eval_live_status(
            args.live_status_path,
            config=config,
            model_path=model_path,
            report=report,
            status="blocked",
            mode="adventure_eval",
        )
    raise_if_incompatible(report)
    compatibility = compatibility_summary_from_report(report)
    config["metadata_path"] = report.metadata_path
    config["metadata_inferred"] = bool(report.metadata_inferred)
    config["model_compatibility"] = model_compatibility_live_status(report)
    print(f"[adventure] policy=ppo model_path={model_path}")
    print(
        "[adventure] inference_only=True "
        f"deterministic={args.deterministic} action_count={compatibility['action_count']} "
        f"conservative_seeds={args.conservative_seeds} allow_new_plants={args.allow_new_plants}"
    )
    print(
        "[adventure] compatible_seed_slots=True "
        f"seed_list={compatibility['expected_seed_list']} "
        f"plant_types={compatibility['expected_plant_types']} "
        f"decoder={compatibility['action_decoder_version']} "
        f"observation={compatibility['observation_version']} "
        f"metadata={compatibility['metadata_path']}"
    )
    run_adventure_eval(
        config=config,
        env_config=make_env_config(config),
        model=model,
        model_path=model_path,
        deterministic=bool(args.deterministic),
        advance_on_wins=max(1, int(config["advance_on_wins"])),
        max_adventure_levels=max(1, int(config["max_adventure_levels"])),
        max_attempts_per_level=max(1, int(config["max_attempts_per_level"])),
        adventure_start_level=max(1, int(config["adventure_start_level"])),
        conservative_seeds=bool(args.conservative_seeds),
        allow_new_plants=bool(args.allow_new_plants),
        live_status_path=args.live_status_path,
        gui=bool(args.gui),
        adventure_soft_max_steps=int(config.get("adventure_soft_max_steps", DEFAULT_ADVENTURE_SOFT_MAX_STEPS)),
        adventure_hard_max_steps=int(config.get("adventure_hard_max_steps", DEFAULT_ADVENTURE_HARD_MAX_STEPS)),
        adventure_final_wave_extension=bool(config.get("adventure_final_wave_extension", True)),
    )


def close_env_safely(env: PvZMaskedPPOEnv, timeout_seconds: float = 5.0) -> bool:
    error: List[BaseException] = []

    def close_target() -> None:
        try:
            env.close()
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=close_target, name="PvZEvalEnvClose", daemon=True)
    thread.start()
    thread.join(max(0.1, timeout_seconds))
    if thread.is_alive():
        try:
            env.base.client.close()
        except Exception:
            pass
        print(f"[eval] warning: env.close() exceeded {timeout_seconds:.1f}s; continuing shutdown")
        return False
    if error:
        print(f"[eval] warning: env.close() failed: {error[0]}")
        return False
    return True


def apply_eval_diagnostics(log: EvalLog, summary: Dict[str, Any]) -> None:
    for field in LANE_DIAGNOSTIC_DICT_FIELDS:
        setattr(log, field, normalize_count_dict(summary.get(field)))
    for field in LANE_DIAGNOSTIC_FLOAT_DICT_FIELDS:
        setattr(log, field, normalize_float_dict(summary.get(field)))
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
    ):
        setattr(log, field, int(summary.get(field) or 0))
    log.wait_action_percent = float(summary.get("wait_action_percent") or 0.0)
    log.plant_action_percent = float(summary.get("plant_action_percent") or 0.0)
    log.row_defense_response_rate = float(summary.get("row_defense_response_rate") or 0.0)
    log.plants_in_threatened_row_ratio = float(summary.get("plants_in_threatened_row_ratio") or 0.0)
    log.plants_in_unthreatened_row_ratio = float(summary.get("plants_in_unthreatened_row_ratio") or 0.0)
    log.peashooter_coverage_rate_by_step = float(summary.get("peashooter_coverage_rate_by_step") or 0.0)
    for field in EPISODE_STRING_FIELDS:
        setattr(log, field, str(summary.get(field) or ""))
    for field in (
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
    ):
        setattr(log, field, int(summary.get(field) or 0))
    for field in (
        "wallnut_damage_absorbed_total",
        "cherrybomb_avg_kills_per_use",
        "fusion_candidate_count_avg",
        "fusion_avg_kills_after_use",
        "legal_action_count_mean",
        "max_row_danger",
        "avg_row_danger",
        *FUSION_REWARD_FLOAT_FIELDS,
    ):
        setattr(log, field, float(summary.get(field) or 0.0))
    log.tactical_mask_enabled = bool(summary.get("tactical_mask_enabled"))
    log.fusion_reward_capped = bool(summary.get("fusion_reward_capped"))
    log.fusion_last_reward_reason = str(summary.get("fusion_last_reward_reason") or "")
    log.fusion_last_source = str(summary.get("fusion_last_source") or "")
    for field in (
        "plant_action_counts",
        "successful_placements_by_plant",
        "invalid_actions_by_plant",
        "fusion_attempts_by_pair",
        "fusion_successes_by_pair",
        "fusion_depth_counts",
    ):
        setattr(log, field, normalize_count_dict(summary.get(field)))
    for field in REWARD_EPISODE_TOTAL_FIELDS:
        setattr(log, field, float(summary.get(field) or 0.0))


def run_eval_episode(env: PvZMaskedPPOEnv, episode: int, model: Any) -> EvalLog:
    log = EvalLog(policy="ppo", episode=episode)
    try:
        obs, reset_info = env.reset()
        reset_payload = reset_info.get("reset", {}) if isinstance(reset_info, dict) else {}
        log.reset_success = bool(reset_payload.get("resetSuccess", reset_payload.get("ok", True)))
        log.reset_seconds = float(
            reset_payload.get("timeToPlayableSeconds")
            or (float(reset_payload.get("reset_ms", 0.0) or 0.0) / 1000.0)
            or 0.0
        )
    except Exception as exc:
        log.reset_failures = 1
        log.bridge_errors = 1
        log.done_reason = "reset_error"
        log.reset_success = False
        print(f"ppo episode {episode} reset failed: {exc}")
        return log

    start_kills = int(env._last_observation.get("killCount", 0) if env._last_observation else 0)
    start_mowers = int(env._last_observation.get("logicalMowerCount", env.rows) if env._last_observation else env.rows)
    legal_total = 0
    while True:
        try:
            action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
            action_id = int(action)
            obs, reward, terminated, truncated, info = env.step(action_id)
        except Exception as exc:
            log.bridge_errors += 1
            log.done_reason = "bridge_error"
            print(f"ppo episode {episode} bridge failed: {exc}")
            break

        log.episode_reward += float(reward)
        log.episode_length += 1
        legal_actions = info.get("legal_actions", [])
        if isinstance(legal_actions, list):
            legal_total += len(legal_actions)
        log.final_wave = int(info.get("final_wave", log.final_wave))
        log.zombies_killed = int(info.get("zombies_killed", log.zombies_killed))
        log.plants_placed = int(info.get("plants_placed", log.plants_placed))
        log.illegal_actions = int(info.get("illegal_actions", log.illegal_actions))
        action_result = info.get("action_result", {})
        placement = action_result.get("placement") or {}
        if action_result.get("costPaid") or placement.get("costPaid"):
            sun_before = int(action_result.get("sunBefore") or placement.get("sunBefore") or 0)
            sun_after = int(action_result.get("sunAfter") or placement.get("sunAfter") or 0)
            spent = max(0, sun_before - sun_after)
            if spent == 0:
                spent = int(action_result.get("plantCost") or placement.get("plantCost") or 0)
            log.sun_spent += max(0, spent)
        raw = info.get("raw_observation", {})
        log.max_wave = int(raw.get("maxWave", log.max_wave))
        log.sun_remaining = int(raw.get("sun", log.sun_remaining))
        final_mowers = int(raw.get("logicalMowerCount", start_mowers))
        log.mowers_lost = max(0, start_mowers - final_mowers)

        if terminated or truncated:
            summary = info.get("episode_summary", {})
            log.run_mode = str(summary.get("run_mode", log.run_mode))
            log.target_level = int(summary.get("target_level", log.target_level))
            log.result = str(summary.get("result", log.result))
            log.reward_total = float(summary.get("reward_total", log.reward_total or log.episode_reward))
            log.done_reason = str(summary.get("done_reason", info.get("done_reason", "timeout" if truncated else "none")))
            log.terminal_reason = str(summary.get("terminal_reason", info.get("terminal_reason", "")))
            log.final_wave = int(summary.get("final_wave", log.final_wave))
            log.max_wave = int(summary.get("max_wave", log.max_wave))
            log.zombies_killed = int(summary.get("zombies_killed", log.zombies_killed))
            log.plants_placed = int(summary.get("plants_placed", log.plants_placed))
            log.sunflowers_planted = int(summary.get("sunflowers_planted", log.sunflowers_planted))
            log.peashooters_planted = int(summary.get("peashooters_planted", log.peashooters_planted))
            log.wallnuts_planted = int(summary.get("wallnuts_planted", log.wallnuts_planted))
            log.cherrybombs_planted = int(summary.get("cherrybombs_planted", log.cherrybombs_planted))
            log.sun_spent = int(summary.get("sun_spent", log.sun_spent))
            log.sun_remaining = int(summary.get("sun_remaining", log.sun_remaining))
            log.mowers_lost = int(summary.get("mowers_lost", log.mowers_lost))
            log.mower_losses = int(summary.get("mower_losses", log.mower_losses))
            log.reset_success = bool(summary.get("reset_success", log.reset_success))
            log.reset_seconds = float(summary.get("reset_seconds", log.reset_seconds))
            log.bridge_errors = int(summary.get("bridge_errors", log.bridge_errors))
            log.illegal_actions = int(summary.get("illegal_actions", log.illegal_actions))
            log.avg_legal_actions = float(summary.get("avg_legal_actions", legal_total / max(1, log.episode_length)))
            apply_eval_diagnostics(log, summary)
            break

    if log.avg_legal_actions == 0.0:
        log.avg_legal_actions = legal_total / max(1, log.episode_length)
    if not log.reset_success:
        log.reset_failures = 1
    return log


def print_eval_table(results: Dict[str, List[EvalLog]]) -> None:
    headers = ["policy", "episodes", "win_rate", "avg_reward", "avg_len", "avg_wave", "avg_kills", "plants", "mowers", "sun", "reset_fail", "bridge_err"]
    rows: List[List[str]] = []
    for policy, logs in results.items():
        summary = summarize_eval_logs(logs)
        rows.append(
            [
                policy,
                str(summary["episodes"]),
                f"{summary['win_rate']:.2f}",
                f"{summary['avg_reward']:.2f}",
                f"{summary['avg_episode_length']:.1f}",
                f"{summary['avg_wave']:.1f}",
                f"{summary['avg_kills']:.1f}",
                f"{summary['avg_plants']:.1f}",
                f"{summary['avg_mowers_lost']:.1f}",
                f"{summary['avg_sun_remaining']:.1f}",
                str(summary["reset_failures"]),
                str(summary["bridge_errors"]),
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    print("PPO Evaluation")
    print("--------------")
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def summarize_eval_logs(logs: List[EvalLog]) -> Dict[str, Any]:
    count = max(1, len(logs))
    total_steps = sum(log.episode_length for log in logs)
    wait_actions = sum(log.wait_actions for log in logs)
    plant_actions = sum(log.plant_actions for log in logs)
    threat_steps = sum_eval_count_dicts(logs, "threat_steps_by_row")
    undefended_steps = sum_eval_count_dicts(logs, "undefended_threat_steps_by_row")
    threatened_zero_defender_steps = sum_eval_count_dicts(logs, "threatened_rows_with_zero_defender_steps_by_row")
    row_defense_opportunities = sum_eval_count_dicts(logs, "row_defense_opportunities_by_row")
    row_defense_responses = sum_eval_count_dicts(logs, "row_defense_responses_by_row")
    peashooter_placements = sum_eval_count_dicts(logs, "peashooter_placements_by_row")
    plant_weight_total = sum(max(0, log.plants_placed) for log in logs)
    full_coverage_sunflower_counts = [
        log.sunflower_count_when_first_full_coverage
        for log in logs
        if log.all_rows_peashooter_covered_step > 0
        and log.sunflower_count_when_first_full_coverage >= 0
    ]
    reward_component_totals = {
        field: sum(float(getattr(log, field, 0.0)) for log in logs)
        for field in REWARD_EPISODE_TOTAL_FIELDS
    }
    cherrybomb_used_count = sum(log.cherrybomb_used_count for log in logs)
    fusion_success_count = sum(log.fusion_success_count for log in logs)
    fusion_candidate_count_total = sum(log.fusion_candidate_count_total for log in logs)
    return {
        "episodes": len(logs),
        "win_rate": sum(1 for log in logs if log.won) / count,
        "avg_reward": sum(log.episode_reward for log in logs) / count,
        "avg_episode_length": sum(log.episode_length for log in logs) / count,
        "avg_wave": sum(log.final_wave for log in logs) / count,
        "avg_kills": sum(log.zombies_killed for log in logs) / count,
        "avg_plants": sum(log.plants_placed for log in logs) / count,
        "sunflowers_planted": sum(log.sunflowers_planted for log in logs),
        "peashooters_planted": sum(log.peashooters_planted for log in logs),
        "wallnuts_planted": sum(log.wallnuts_planted for log in logs),
        "cherrybombs_planted": sum(log.cherrybombs_planted for log in logs),
        "avg_mowers_lost": sum(log.mowers_lost for log in logs) / count,
        "avg_sun_remaining": sum(log.sun_remaining for log in logs) / count,
        "reset_failures": sum(log.reset_failures for log in logs),
        "bridge_errors": sum(log.bridge_errors for log in logs),
        "wait_actions": wait_actions,
        "plant_actions": plant_actions,
        "wait_action_percent": 100.0 * wait_actions / max(1, total_steps),
        "plant_action_percent": 100.0 * plant_actions / max(1, total_steps),
        "plants_by_row": sum_eval_count_dicts(logs, "plants_by_row"),
        "peashooters_by_row": sum_eval_count_dicts(logs, "peashooters_by_row"),
        "sunflowers_by_row": sum_eval_count_dicts(logs, "sunflowers_by_row"),
        "threat_steps_by_row": threat_steps,
        "undefended_threat_steps_by_row": undefended_steps,
        "undefended_threat_ratio_by_row": ratio_dict(undefended_steps, threat_steps),
        "undefended_threat_age_avg_by_row": weighted_average_dict_from_logs(
            logs,
            "undefended_threat_age_avg_by_row",
            "threatened_rows_with_zero_defender_steps_by_row",
        ),
        "undefended_threat_age_max_by_row": max_eval_count_dicts(logs, "undefended_threat_age_max_by_row"),
        "threatened_rows_with_zero_defender_steps_by_row": threatened_zero_defender_steps,
        "peashooters_per_threat_step_by_row": ratio_dict(peashooter_placements, threat_steps),
        "first_defense_step_by_row": average_positive_step_dict_from_logs(logs, "first_defense_step_by_row"),
        "plants_in_threatened_row_ratio": (
            sum(log.plants_in_threatened_row_ratio * max(0, log.plants_placed) for log in logs) / plant_weight_total
            if plant_weight_total > 0
            else 0.0
        ),
        "plants_in_unthreatened_row_ratio": (
            sum(log.plants_in_unthreatened_row_ratio * max(0, log.plants_placed) for log in logs) / plant_weight_total
            if plant_weight_total > 0
            else 0.0
        ),
        "overdefended_while_undefended_count": sum(log.overdefended_while_undefended_count for log in logs),
        "least_defended_threatened_row_plant_count": sum(log.least_defended_threatened_row_plant_count for log in logs),
        "rows_with_peashooter_count": sum(log.rows_with_peashooter_count for log in logs) / count,
        "all_rows_peashooter_covered_step": average_positive_values(
            [float(log.all_rows_peashooter_covered_step) for log in logs]
        ),
        "first_peashooter_by_row_step": average_positive_step_dict_from_logs(logs, "first_peashooter_by_row_step"),
        "sunflower_count_when_first_full_coverage": (
            sum(full_coverage_sunflower_counts) / len(full_coverage_sunflower_counts)
            if full_coverage_sunflower_counts
            else -1.0
        ),
        "sunflower_overbuild_before_defense_count": sum(log.sunflower_overbuild_before_defense_count for log in logs),
        "sunflower_overbuild_count": sum(log.sunflower_overbuild_count for log in logs),
        "peashooter_coverage_rate_by_step": (
            sum(log.peashooter_coverage_rate_by_step * log.episode_length for log in logs) / max(1, total_steps)
        ),
        "mower_losses_by_row": sum_eval_count_dicts(logs, "mower_losses_by_row"),
        "wait_under_threat_count": sum(log.wait_under_threat_count for log in logs),
        "close_zombie_undefended_count": sum(log.close_zombie_undefended_count for log in logs),
        "illegal_reason_counts": sum_eval_count_dicts(logs, "illegal_reason_counts"),
        "legal_peashooter_actions_by_row": sum_eval_count_dicts(logs, "legal_peashooter_actions_by_row"),
        "peashooter_available_but_waited_by_row": sum_eval_count_dicts(logs, "peashooter_available_but_waited_by_row"),
        "peashooter_available_but_planted_elsewhere_by_row": sum_eval_count_dicts(
            logs,
            "peashooter_available_but_planted_elsewhere_by_row",
        ),
        "sunflower_while_undefended_threat_by_row": sum_eval_count_dicts(logs, "sunflower_while_undefended_threat_by_row"),
        "legal_actions_by_seed_slot": sum_eval_count_dicts(logs, "legal_actions_by_seed_slot"),
        "bridge_legal_actions_by_seed_slot": sum_eval_count_dicts(logs, "bridge_legal_actions_by_seed_slot"),
        "python_mask_block_reason_counts": sum_eval_count_dicts(logs, "python_mask_block_reason_counts"),
        "pre_step_mask_blocked_count": sum(log.pre_step_mask_blocked_count for log in logs),
        "cooldown_illegal_exposed_by_mask_count": sum(log.cooldown_illegal_exposed_by_mask_count for log in logs),
        "mask_bridge_disagreement_count": sum(log.mask_bridge_disagreement_count for log in logs),
        "wait_while_actionable_threat_count": sum(log.wait_while_actionable_threat_count for log in logs),
        "wait_while_peashooter_affordable_ready_count": sum(log.wait_while_peashooter_affordable_ready_count for log in logs),
        "wait_while_wallnut_affordable_ready_count": sum(log.wait_while_wallnut_affordable_ready_count for log in logs),
        "wait_while_cherrybomb_affordable_ready_count": sum(log.wait_while_cherrybomb_affordable_ready_count for log in logs),
        "active_threat_rows_without_peashooter_count": sum(log.active_threat_rows_without_peashooter_count for log in logs),
        "sunflower_greed_while_defense_missing_count": sum(log.sunflower_greed_while_defense_missing_count for log in logs),
        "wallnut_placements_by_row": sum_eval_count_dicts(logs, "wallnut_placements_by_row"),
        "wallnut_placements_by_col": sum_eval_count_dicts(logs, "wallnut_placements_by_col"),
        "wallnut_blocks_active_threat_count": sum(log.wallnut_blocks_active_threat_count for log in logs),
        "wallnut_low_value_placement_count": sum(log.wallnut_low_value_placement_count for log in logs),
        "wallnut_threatened_lane_placements": sum(log.wallnut_threatened_lane_placements for log in logs),
        "wallnut_between_zombie_and_house_count": sum(log.wallnut_between_zombie_and_house_count for log in logs),
        "wallnut_frontline_count": sum(log.wallnut_frontline_count for log in logs),
        "wallnut_emergency_blocks": sum(log.wallnut_emergency_blocks for log in logs),
        "wallnut_useless_placements": sum(log.wallnut_useless_placements for log in logs),
        "wallnut_damage_absorbed_total": sum(float(log.wallnut_damage_absorbed_total) for log in logs),
        "cherrybomb_used_count": cherrybomb_used_count,
        "cherrybomb_kills_total": sum(log.cherrybomb_kills_total for log in logs),
        "cherrybomb_avg_kills_per_use": (
            sum(log.cherrybomb_kills_total for log in logs) / max(1, cherrybomb_used_count)
        ),
        "cherrybomb_zero_kill_count": sum(log.cherrybomb_zero_kill_count for log in logs),
        "cherrybomb_zero_kill_uses": sum(log.cherrybomb_zero_kill_uses for log in logs),
        "cherrybomb_cluster_uses": sum(log.cherrybomb_cluster_uses for log in logs),
        "cherrybomb_emergency_uses": sum(log.cherrybomb_emergency_uses for log in logs),
        "cherrybomb_buckethead_kills": sum(log.cherrybomb_buckethead_kills for log in logs),
        "cherrybomb_conehead_kills": sum(log.cherrybomb_conehead_kills for log in logs),
        "cherrybomb_heavy_zombie_kills": sum(log.cherrybomb_heavy_zombie_kills for log in logs),
        "cherrybomb_used_under_threat_count": sum(log.cherrybomb_used_under_threat_count for log in logs),
        "cherrybomb_used_low_value_count": sum(log.cherrybomb_used_low_value_count for log in logs),
        "mower_risk_steps_by_row": sum_eval_count_dicts(logs, "mower_risk_steps_by_row"),
        "mower_saves_estimated_by_row": sum_eval_count_dicts(logs, "mower_saves_estimated_by_row"),
        "close_zombie_with_no_defense_count": sum(log.close_zombie_with_no_defense_count for log in logs),
        "undefended_threat_steps": sum(log.undefended_threat_steps for log in logs),
        "high_danger_unanswered_steps": sum(log.high_danger_unanswered_steps for log in logs),
        "mower_exposure_steps": sum(log.mower_exposure_steps for log in logs),
        "overdefense_count": sum(log.overdefense_count for log in logs),
        "mower_losses": sum(log.mower_losses for log in logs),
        "legal_action_count_mean": sum(log.legal_action_count_mean or log.avg_legal_actions for log in logs) / count,
        "max_row_danger": max((log.max_row_danger for log in logs), default=0.0),
        "avg_row_danger": sum(log.avg_row_danger for log in logs) / count,
        "tactical_mask_enabled": any(log.tactical_mask_enabled for log in logs),
        "wallnut_actions_masked": sum(log.wallnut_actions_masked for log in logs),
        "cherrybomb_actions_masked": sum(log.cherrybomb_actions_masked for log in logs),
        "wallnut_actions_available": sum(log.wallnut_actions_available for log in logs),
        "cherrybomb_actions_available": sum(log.cherrybomb_actions_available for log in logs),
        "mask_all_but_wait_count": sum(log.mask_all_but_wait_count for log in logs),
        "buckethead_count_by_row": sum_eval_count_dicts(logs, "buckethead_count_by_row"),
        "conehead_count_by_row": sum_eval_count_dicts(logs, "conehead_count_by_row"),
        "tough_zombie_count_by_row": sum_eval_count_dicts(logs, "tough_zombie_count_by_row"),
        "tough_zombie_response_count": sum(log.tough_zombie_response_count for log in logs),
        "fusion_policy": logs[0].fusion_policy if logs else FUSION_POLICY_NONE,
        "fusion_candidate_count_total": fusion_candidate_count_total,
        "fusion_candidate_count_avg": fusion_candidate_count_total / max(1, total_steps),
        "fusion_attempted_count": sum(log.fusion_attempted_count for log in logs),
        "fusion_success_count": fusion_success_count,
        "fusion_failed_count": sum(log.fusion_failed_count for log in logs),
        "fusion_rejected_count": sum(log.fusion_rejected_count for log in logs),
        "fusion_rejected_reasons": sum_eval_count_dicts(logs, "fusion_rejected_reasons"),
        "fusion_by_result_type": sum_eval_count_dicts(logs, "fusion_by_result_type"),
        "fusion_by_source_type": sum_eval_count_dicts(logs, "fusion_by_source_type"),
        "fusion_by_row": sum_eval_count_dicts(logs, "fusion_by_row"),
        "plant_action_counts": sum_eval_count_dicts(logs, "plant_action_counts"),
        "successful_placements_by_plant": sum_eval_count_dicts(logs, "successful_placements_by_plant"),
        "invalid_actions_by_plant": sum_eval_count_dicts(logs, "invalid_actions_by_plant"),
        "fusion_attempts_by_pair": sum_eval_count_dicts(logs, "fusion_attempts_by_pair"),
        "fusion_successes_by_pair": sum_eval_count_dicts(logs, "fusion_successes_by_pair"),
        "fusion_depth_counts": sum_eval_count_dicts(logs, "fusion_depth_counts"),
        "recursive_fusion_count": sum(log.recursive_fusion_count for log in logs),
        "highest_fusion_tier": max((log.highest_fusion_tier for log in logs), default=0),
        "action_freeze_count": sum(log.action_freeze_count for log in logs),
        "mean_action_duration_seconds": (
            sum(log.mean_action_duration_seconds for log in logs) / max(1, len(logs))
        ),
        "max_action_duration_seconds": max((log.max_action_duration_seconds for log in logs), default=0.0),
        "p95_action_duration_seconds": max((log.p95_action_duration_seconds for log in logs), default=0.0),
        "fusion_under_threat_count": sum(log.fusion_under_threat_count for log in logs),
        "fusion_near_buckethead_count": sum(log.fusion_near_buckethead_count for log in logs),
        "fusion_near_conehead_count": sum(log.fusion_near_conehead_count for log in logs),
        "fusion_estimated_mower_save_count": sum(log.fusion_estimated_mower_save_count for log in logs),
        "fusion_kills_after_use_total": sum(log.fusion_kills_after_use_total for log in logs),
        "fusion_avg_kills_after_use": (
            sum(log.fusion_kills_after_use_total for log in logs) / max(1, fusion_success_count)
        ),
        "fusion_bridge_error_count": sum(log.fusion_bridge_error_count for log in logs),
        "fusion_unsafe_state_block_count": sum(log.fusion_unsafe_state_block_count for log in logs),
        "plant_actions_by_row": sum_eval_count_dicts(logs, "plant_actions_by_row"),
        "peashooter_actions_by_row": sum_eval_count_dicts(logs, "peashooter_actions_by_row"),
        "sunflower_actions_by_row": sum_eval_count_dicts(logs, "sunflower_actions_by_row"),
        "plant_placements_by_row": sum_eval_count_dicts(logs, "plant_placements_by_row"),
        "peashooter_placements_by_row": sum_eval_count_dicts(logs, "peashooter_placements_by_row"),
        "sunflower_placements_by_row": sum_eval_count_dicts(logs, "sunflower_placements_by_row"),
        "row_defense_opportunities_by_row": row_defense_opportunities,
        "row_defense_responses_by_row": row_defense_responses,
        "row_defense_response_rate_by_row": ratio_dict(row_defense_responses, row_defense_opportunities),
        "row_defense_response_rate": (
            total_count(row_defense_responses) / total_count(row_defense_opportunities)
            if total_count(row_defense_opportunities) > 0
            else 0.0
        ),
        **{
            field: sum(float(getattr(log, field, 0.0)) for log in logs)
            for field in FUSION_REWARD_FLOAT_FIELDS[:-2]
        },
        "fusion_reward_total": float(reward_component_totals.get("fusion_reward_total", 0.0)),
        "fusion_reward_capped": any(bool(log.fusion_reward_capped) for log in logs),
        "fusion_last_reward_delta": float(logs[-1].fusion_last_reward_delta) if logs else 0.0,
        "fusion_last_usefulness_bonus": float(logs[-1].fusion_last_usefulness_bonus) if logs else 0.0,
        "fusion_last_reward_reason": str(logs[-1].fusion_last_reward_reason) if logs else "",
        "fusion_last_source": str(logs[-1].fusion_last_source) if logs else "",
        "reward_component_totals": reward_component_totals,
        "reward_component_avgs": {field: value / count for field, value in reward_component_totals.items()},
    }


def sum_eval_count_dicts(logs: List[EvalLog], field_name: str) -> Dict[str, int]:
    totals: Counter[str] = Counter()
    for log in logs:
        values = getattr(log, field_name, {})
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            try:
                totals[str(key)] += int(value)
            except (TypeError, ValueError):
                continue
    return dict(sorted(totals.items(), key=lambda item: (0, int(item[0])) if item[0].isdigit() else (1, item[0])))


def max_eval_count_dicts(logs: List[EvalLog], field_name: str) -> Dict[str, int]:
    totals: Counter[str] = Counter()
    for log in logs:
        values = getattr(log, field_name, {})
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            try:
                totals[str(key)] = max(int(totals.get(str(key), 0)), int(value))
            except (TypeError, ValueError):
                continue
    return dict(sorted(totals.items(), key=lambda item: (0, int(item[0])) if item[0].isdigit() else (1, item[0])))


def weighted_average_dict_from_logs(logs: List[EvalLog], field_name: str, weight_field: str) -> Dict[str, float]:
    totals: Counter[str] = Counter()
    weights: Counter[str] = Counter()
    for log in logs:
        values = getattr(log, field_name, {})
        row_weights = getattr(log, weight_field, {})
        if not isinstance(values, dict) or not isinstance(row_weights, dict):
            continue
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


def average_positive_values(values: List[float]) -> float:
    positives = [float(value) for value in values if float(value) > 0.0]
    return sum(positives) / len(positives) if positives else 0.0


def average_positive_step_dict_from_logs(logs: List[EvalLog], field_name: str) -> Dict[str, float]:
    totals: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    for log in logs:
        values = getattr(log, field_name, {})
        if not isinstance(values, dict):
            continue
        for key, raw_value in values.items():
            try:
                value = int(raw_value)
            except (TypeError, ValueError):
                continue
            if value > 0:
                totals[str(key)] += value
                counts[str(key)] += 1
    keys = sorted(set(totals.keys()) | set(counts.keys()), key=lambda item: (0, int(item)) if item.isdigit() else (1, item))
    return {
        key: float(totals.get(key, 0)) / float(counts.get(key, 0))
        if counts.get(key, 0) > 0
        else 0.0
        for key in keys
    }


def print_eval_diagnostics(results: Dict[str, List[EvalLog]]) -> None:
    for policy, logs in results.items():
        summary = summarize_eval_logs(logs)
        payload = {
            "wait_action_percent": summary["wait_action_percent"],
            "plant_action_percent": summary["plant_action_percent"],
            "plants_by_row": summary["plants_by_row"],
            "peashooters_by_row": summary["peashooters_by_row"],
            "sunflowers_by_row": summary["sunflowers_by_row"],
            "plant_actions_by_row": summary["plant_actions_by_row"],
            "peashooter_actions_by_row": summary["peashooter_actions_by_row"],
            "sunflower_actions_by_row": summary["sunflower_actions_by_row"],
            "peashooter_placements_by_row": summary["peashooter_placements_by_row"],
            "sunflower_placements_by_row": summary["sunflower_placements_by_row"],
            "threat_steps_by_row": summary["threat_steps_by_row"],
            "undefended_threat_steps_by_row": summary["undefended_threat_steps_by_row"],
            "undefended_threat_ratio_by_row": summary["undefended_threat_ratio_by_row"],
            "undefended_threat_age_avg_by_row": summary["undefended_threat_age_avg_by_row"],
            "undefended_threat_age_max_by_row": summary["undefended_threat_age_max_by_row"],
            "threatened_rows_with_zero_defender_steps_by_row": summary["threatened_rows_with_zero_defender_steps_by_row"],
            "peashooters_per_threat_step_by_row": summary["peashooters_per_threat_step_by_row"],
            "first_defense_step_by_row": summary["first_defense_step_by_row"],
            "plants_in_threatened_row_ratio": summary["plants_in_threatened_row_ratio"],
            "plants_in_unthreatened_row_ratio": summary["plants_in_unthreatened_row_ratio"],
            "overdefended_while_undefended_count": summary["overdefended_while_undefended_count"],
            "least_defended_threatened_row_plant_count": summary["least_defended_threatened_row_plant_count"],
            "rows_with_peashooter_count": summary["rows_with_peashooter_count"],
            "all_rows_peashooter_covered_step": summary["all_rows_peashooter_covered_step"],
            "first_peashooter_by_row_step": summary["first_peashooter_by_row_step"],
            "sunflower_count_when_first_full_coverage": summary["sunflower_count_when_first_full_coverage"],
            "sunflower_overbuild_before_defense_count": summary["sunflower_overbuild_before_defense_count"],
            "peashooter_coverage_rate_by_step": summary["peashooter_coverage_rate_by_step"],
            "mower_losses_by_row": summary["mower_losses_by_row"],
            "row_defense_opportunities_by_row": summary["row_defense_opportunities_by_row"],
            "row_defense_responses_by_row": summary["row_defense_responses_by_row"],
            "row_defense_response_rate_by_row": summary["row_defense_response_rate_by_row"],
            "row_defense_response_rate": summary["row_defense_response_rate"],
            "wait_under_threat_count": summary["wait_under_threat_count"],
            "close_zombie_undefended_count": summary["close_zombie_undefended_count"],
            "illegal_reason_counts": summary["illegal_reason_counts"],
            "legal_peashooter_actions_by_row": summary["legal_peashooter_actions_by_row"],
            "peashooter_available_but_waited_by_row": summary["peashooter_available_but_waited_by_row"],
            "peashooter_available_but_planted_elsewhere_by_row": summary["peashooter_available_but_planted_elsewhere_by_row"],
            "sunflower_while_undefended_threat_by_row": summary["sunflower_while_undefended_threat_by_row"],
            "legal_actions_by_seed_slot": summary["legal_actions_by_seed_slot"],
            "bridge_legal_actions_by_seed_slot": summary["bridge_legal_actions_by_seed_slot"],
            "python_mask_block_reason_counts": summary["python_mask_block_reason_counts"],
            "pre_step_mask_blocked_count": summary["pre_step_mask_blocked_count"],
            "cooldown_illegal_exposed_by_mask_count": summary["cooldown_illegal_exposed_by_mask_count"],
            "mask_bridge_disagreement_count": summary["mask_bridge_disagreement_count"],
            "wait_while_actionable_threat_count": summary["wait_while_actionable_threat_count"],
            "active_threat_rows_without_peashooter_count": summary["active_threat_rows_without_peashooter_count"],
            "sunflower_greed_while_defense_missing_count": summary["sunflower_greed_while_defense_missing_count"],
            "wallnut_placements_by_row": summary["wallnut_placements_by_row"],
            "wallnut_blocks_active_threat_count": summary["wallnut_blocks_active_threat_count"],
            "wallnut_low_value_placement_count": summary["wallnut_low_value_placement_count"],
            "cherrybomb_used_count": summary["cherrybomb_used_count"],
            "cherrybomb_kills_total": summary["cherrybomb_kills_total"],
            "cherrybomb_avg_kills_per_use": summary["cherrybomb_avg_kills_per_use"],
            "cherrybomb_zero_kill_count": summary["cherrybomb_zero_kill_count"],
            "mower_risk_steps_by_row": summary["mower_risk_steps_by_row"],
            "mower_saves_estimated_by_row": summary["mower_saves_estimated_by_row"],
            "buckethead_count_by_row": summary["buckethead_count_by_row"],
            "conehead_count_by_row": summary["conehead_count_by_row"],
            "tough_zombie_count_by_row": summary["tough_zombie_count_by_row"],
            "tough_zombie_response_count": summary["tough_zombie_response_count"],
            "fusion_policy": summary["fusion_policy"],
            "fusion_candidate_count_total": summary["fusion_candidate_count_total"],
            "fusion_candidate_count_avg": summary["fusion_candidate_count_avg"],
            "fusion_attempted_count": summary["fusion_attempted_count"],
            "fusion_success_count": summary["fusion_success_count"],
            "fusion_failed_count": summary["fusion_failed_count"],
            "fusion_rejected_count": summary["fusion_rejected_count"],
            "fusion_rejected_reasons": summary["fusion_rejected_reasons"],
            "fusion_by_result_type": summary["fusion_by_result_type"],
            "fusion_by_source_type": summary["fusion_by_source_type"],
            "fusion_by_row": summary["fusion_by_row"],
            "fusion_under_threat_count": summary["fusion_under_threat_count"],
            "fusion_near_buckethead_count": summary["fusion_near_buckethead_count"],
            "fusion_near_conehead_count": summary["fusion_near_conehead_count"],
            "fusion_estimated_mower_save_count": summary["fusion_estimated_mower_save_count"],
            "fusion_bridge_error_count": summary["fusion_bridge_error_count"],
            "fusion_unsafe_state_block_count": summary["fusion_unsafe_state_block_count"],
            "plant_action_counts": summary.get("plant_action_counts", {}),
            "successful_placements_by_plant": summary.get("successful_placements_by_plant", {}),
            "invalid_actions_by_plant": summary.get("invalid_actions_by_plant", {}),
            "fusion_attempts_by_pair": summary.get("fusion_attempts_by_pair", {}),
            "fusion_successes_by_pair": summary.get("fusion_successes_by_pair", {}),
            "fusion_depth_counts": summary.get("fusion_depth_counts", {}),
            "recursive_fusion_count": summary.get("recursive_fusion_count", 0),
            "highest_fusion_tier": summary.get("highest_fusion_tier", 0),
            "action_freeze_count": summary.get("action_freeze_count", 0),
            "action_duration_seconds": {
                "mean": summary.get("mean_action_duration_seconds", 0.0),
                "p95": summary.get("p95_action_duration_seconds", 0.0),
                "max": summary.get("max_action_duration_seconds", 0.0),
            },
            "reward_component_avgs": summary["reward_component_avgs"],
        }
        print(f"[eval-diagnostics] policy={policy} " + json.dumps(payload, separators=(",", ":"), sort_keys=True))


def validation_summary_from_logs(
    logs: List[EvalLog],
    total_timesteps: int,
    run_dir: Path,
    model_path: Path,
    fps_avg: float,
) -> Dict[str, Any]:
    summary = summarize_eval_logs(logs)
    return {
        "total_timesteps": total_timesteps,
        "episodes": summary["episodes"],
        "avg_reward": summary["avg_reward"],
        "avg_length": summary["avg_episode_length"],
        "win_rate": summary["win_rate"],
        "avg_wave": summary["avg_wave"],
        "avg_kills": summary["avg_kills"],
        "avg_plants": summary["avg_plants"],
        "avg_mowers_lost": summary["avg_mowers_lost"],
        "reset_failures": summary["reset_failures"],
        "bridge_errors": summary["bridge_errors"],
        "wait_action_percent": summary["wait_action_percent"],
        "plant_action_percent": summary["plant_action_percent"],
        "plants_by_row": summary["plants_by_row"],
        "peashooters_by_row": summary["peashooters_by_row"],
        "sunflowers_by_row": summary["sunflowers_by_row"],
        "peashooter_actions_by_row": summary["peashooter_actions_by_row"],
        "peashooter_placements_by_row": summary["peashooter_placements_by_row"],
        "threat_steps_by_row": summary["threat_steps_by_row"],
        "undefended_threat_steps_by_row": summary["undefended_threat_steps_by_row"],
        "undefended_threat_ratio_by_row": summary["undefended_threat_ratio_by_row"],
        "undefended_threat_age_avg_by_row": summary["undefended_threat_age_avg_by_row"],
        "undefended_threat_age_max_by_row": summary["undefended_threat_age_max_by_row"],
        "threatened_rows_with_zero_defender_steps_by_row": summary["threatened_rows_with_zero_defender_steps_by_row"],
        "peashooters_per_threat_step_by_row": summary["peashooters_per_threat_step_by_row"],
        "first_defense_step_by_row": summary["first_defense_step_by_row"],
        "plants_in_threatened_row_ratio": summary["plants_in_threatened_row_ratio"],
        "plants_in_unthreatened_row_ratio": summary["plants_in_unthreatened_row_ratio"],
        "overdefended_while_undefended_count": summary["overdefended_while_undefended_count"],
        "least_defended_threatened_row_plant_count": summary["least_defended_threatened_row_plant_count"],
        "rows_with_peashooter_count": summary["rows_with_peashooter_count"],
        "all_rows_peashooter_covered_step": summary["all_rows_peashooter_covered_step"],
        "first_peashooter_by_row_step": summary["first_peashooter_by_row_step"],
        "sunflower_count_when_first_full_coverage": summary["sunflower_count_when_first_full_coverage"],
        "sunflower_overbuild_before_defense_count": summary["sunflower_overbuild_before_defense_count"],
        "peashooter_coverage_rate_by_step": summary["peashooter_coverage_rate_by_step"],
        "mower_losses_by_row": summary["mower_losses_by_row"],
        "wait_under_threat_count": summary["wait_under_threat_count"],
        "close_zombie_undefended_count": summary["close_zombie_undefended_count"],
        "illegal_reason_counts": summary["illegal_reason_counts"],
        "legal_peashooter_actions_by_row": summary["legal_peashooter_actions_by_row"],
        "peashooter_available_but_waited_by_row": summary["peashooter_available_but_waited_by_row"],
        "peashooter_available_but_planted_elsewhere_by_row": summary["peashooter_available_but_planted_elsewhere_by_row"],
        "sunflower_while_undefended_threat_by_row": summary["sunflower_while_undefended_threat_by_row"],
        "legal_actions_by_seed_slot": summary["legal_actions_by_seed_slot"],
        "bridge_legal_actions_by_seed_slot": summary["bridge_legal_actions_by_seed_slot"],
        "python_mask_block_reason_counts": summary["python_mask_block_reason_counts"],
        "pre_step_mask_blocked_count": summary["pre_step_mask_blocked_count"],
        "cooldown_illegal_exposed_by_mask_count": summary["cooldown_illegal_exposed_by_mask_count"],
        "mask_bridge_disagreement_count": summary["mask_bridge_disagreement_count"],
        "wait_while_actionable_threat_count": summary["wait_while_actionable_threat_count"],
        "active_threat_rows_without_peashooter_count": summary["active_threat_rows_without_peashooter_count"],
        "sunflower_greed_while_defense_missing_count": summary["sunflower_greed_while_defense_missing_count"],
        "wallnut_placements_by_row": summary["wallnut_placements_by_row"],
        "wallnut_blocks_active_threat_count": summary["wallnut_blocks_active_threat_count"],
        "wallnut_low_value_placement_count": summary["wallnut_low_value_placement_count"],
        "cherrybomb_used_count": summary["cherrybomb_used_count"],
        "cherrybomb_kills_total": summary["cherrybomb_kills_total"],
        "cherrybomb_avg_kills_per_use": summary["cherrybomb_avg_kills_per_use"],
        "cherrybomb_zero_kill_count": summary["cherrybomb_zero_kill_count"],
        "cherrybomb_buckethead_kills": summary["cherrybomb_buckethead_kills"],
        "cherrybomb_conehead_kills": summary["cherrybomb_conehead_kills"],
        "mower_risk_steps_by_row": summary["mower_risk_steps_by_row"],
        "mower_saves_estimated_by_row": summary["mower_saves_estimated_by_row"],
        "buckethead_count_by_row": summary["buckethead_count_by_row"],
        "conehead_count_by_row": summary["conehead_count_by_row"],
        "tough_zombie_count_by_row": summary["tough_zombie_count_by_row"],
        "tough_zombie_response_count": summary["tough_zombie_response_count"],
        "fusion_policy": summary["fusion_policy"],
        "fusion_candidate_count_total": summary["fusion_candidate_count_total"],
        "fusion_candidate_count_avg": summary["fusion_candidate_count_avg"],
        "fusion_attempted_count": summary["fusion_attempted_count"],
        "fusion_success_count": summary["fusion_success_count"],
        "fusion_failed_count": summary["fusion_failed_count"],
        "fusion_rejected_count": summary["fusion_rejected_count"],
        "fusion_rejected_reasons": summary["fusion_rejected_reasons"],
        "fusion_by_result_type": summary["fusion_by_result_type"],
        "fusion_by_source_type": summary["fusion_by_source_type"],
        "fusion_by_row": summary["fusion_by_row"],
        "fusion_under_threat_count": summary["fusion_under_threat_count"],
        "fusion_near_buckethead_count": summary["fusion_near_buckethead_count"],
        "fusion_near_conehead_count": summary["fusion_near_conehead_count"],
        "fusion_estimated_mower_save_count": summary["fusion_estimated_mower_save_count"],
        "fusion_kills_after_use_total": summary["fusion_kills_after_use_total"],
        "fusion_avg_kills_after_use": summary["fusion_avg_kills_after_use"],
        "fusion_bridge_error_count": summary["fusion_bridge_error_count"],
        "fusion_unsafe_state_block_count": summary["fusion_unsafe_state_block_count"],
        "fusion_reward_total": summary["fusion_reward_total"],
        **{field: summary[field] for field in FUSION_REWARD_FLOAT_FIELDS},
        "fusion_reward_capped": summary["fusion_reward_capped"],
        "fusion_last_reward_reason": summary["fusion_last_reward_reason"],
        "fusion_last_source": summary["fusion_last_source"],
        "row_defense_opportunities_by_row": summary["row_defense_opportunities_by_row"],
        "row_defense_responses_by_row": summary["row_defense_responses_by_row"],
        "row_defense_response_rate_by_row": summary["row_defense_response_rate_by_row"],
        "row_defense_response_rate": summary["row_defense_response_rate"],
        "reward_component_totals": summary["reward_component_totals"],
        "reward_component_avgs": summary["reward_component_avgs"],
        "fps_avg": fps_avg,
        "model_path": str(model_path),
        "run_dir": str(run_dir),
    }


def write_eval_outputs(run_dir: Path, results: Dict[str, List[EvalLog]], config: Dict[str, Any], model_path: Path) -> None:
    payload = {
        "status": "complete",
        "model_path": str(model_path),
        "action_count": action_count_for_config(config),
        "model_family": config.get("model_family", ""),
        "seed_slot_signature": seed_slot_signature(config),
        "policies": {policy: summarize_eval_logs(logs) for policy, logs in results.items()},
        "episodes": {policy: [asdict(log) for log in logs] for policy, logs in results.items()},
    }
    (run_dir / "eval_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (run_dir / "eval_results.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(asdict(EvalLog(policy="ppo", episode=0)).keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for logs in results.values():
            for log in logs:
                writer.writerow(csv_safe_row(asdict(log)))
    print(f"Saved evaluation JSON to {run_dir / 'eval_results.json'}")
    print(f"Saved evaluation CSV to {run_dir / 'eval_results.csv'}")


def check_deps() -> int:
    missing: List[str] = []
    for module in ("gymnasium", "numpy", "stable_baselines3", "sb3_contrib"):
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
    print("MaskablePPO dependencies are installed.")
    return 0


def loaded_model_contract(model_path: Path) -> Tuple[int, Optional[List[int]]]:
    MaskablePPO = require_maskable_ppo()
    model = MaskablePPO.load(str(model_path))
    return _model_action_count(model), _model_observation_shape(model)


def loaded_model_action_count(model_path: Path) -> int:
    """Backward-compatible helper retained for external callers."""

    action_count, _observation_shape = loaded_model_contract(model_path)
    return action_count


def metadata_dry_run(config: Dict[str, Any], model_path: Path) -> int:
    if not model_path.exists():
        if str(config.get("run_mode", "")) == ADVENTURE_GENERALIST_RUN_MODE_TRAIN:
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
    if str(config.get("run_mode", "")) == ADVENTURE_GENERALIST_RUN_MODE_TRAIN:
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
            "model_path": str(model_path),
            "model_action_count": model_action_count,
            "model_compatibility": summary.get("model_compatibility", {}),
        }
        print(json.dumps(payload, indent=2))
        return 0
    result = validate_model_metadata(
        model_path,
        config,
        model_action_count=model_action_count,
        model_observation_shape=model_observation_shape,
        env_metadata=env_metadata_for_config(config),
        allow_missing_model_metadata=bool(config.get("allow_missing_model_metadata", False)),
    )
    payload = result.to_dict()
    payload["model_path"] = str(model_path)
    payload["model_action_count"] = model_action_count
    payload["model_compatibility"] = model_compatibility_live_status(result)
    print(json.dumps(payload, indent=2))
    return 0 if result.ok else 1


def router_dry_run(config: Dict[str, Any], args: argparse.Namespace) -> int:
    schedule_path = args.model_schedule or Path("configs/model_schedule.json")
    if not schedule_path.exists():
        payload = {"ok": False, "blocked_reason": "model_schedule_missing", "model_schedule": str(schedule_path)}
        print(json.dumps(payload, indent=2))
        return 1
    router = ModelRouter.from_file(schedule_path)
    level = args.dry_run_level or config.get("adventure_start_level") or "1-1"
    unlocked = parse_seed_list(args.dry_run_unlocked_seeds or "SunFlower,Peashooter")
    available = parse_seed_list(args.dry_run_available_seeds or args.dry_run_unlocked_seeds or "SunFlower,Peashooter")
    decision = router.select_stage(level=level, unlocked_seeds=unlocked, available_seeds=available)
    payload = {
        "ok": bool(decision.ok),
        "model_schedule": str(schedule_path),
        "dry_run_level": level,
        "unlocked_seeds": unlocked,
        "available_seeds": available,
        "decision": decision.to_dict(),
    }
    if not decision.ok or decision.stage is None:
        payload["blocked_reason"] = decision.blocked_reason
        print(json.dumps(payload, indent=2))
        return 1
    if not decision.stage.model_path.exists():
        payload["ok"] = False
        payload["blocked_reason"] = "model_path_missing"
        print(json.dumps(payload, indent=2))
        return 1
    selected_config = stage_config(config, decision.stage)
    model_action_count, model_observation_shape = loaded_model_contract(decision.stage.model_path)
    compatibility = validate_model_metadata(
        decision.stage.model_path,
        selected_config,
        model_action_count=model_action_count,
        model_observation_shape=model_observation_shape,
        env_metadata=env_metadata_for_config(selected_config),
        allow_missing_model_metadata=bool(config.get("allow_missing_model_metadata", False)),
    )
    payload["compatibility"] = compatibility.to_dict()
    payload["model_compatibility"] = model_compatibility_live_status(compatibility)
    payload["model_action_count"] = model_action_count
    if not compatibility.ok:
        payload["ok"] = False
        payload["blocked_reason"] = compatibility.blocked_reason
        print(json.dumps(payload, indent=2))
        return 1
    payload["selected_stage"] = decision.stage.stage_id
    print(json.dumps(payload, indent=2))
    return 0


def execution_route_for_config(
    config: Dict[str, Any],
    args: argparse.Namespace,
    *,
    configured_mode_explicit: bool = False,
) -> str:
    """Return the runtime branch selected by the resolved run mode."""

    run_mode = str(config.get("run_mode") or "")
    if run_mode == ADVENTURE_GENERALIST_RUN_MODE_TRAIN:
        return "train"
    if run_mode in {RUN_MODE_ADVENTURE_EVAL, ADVENTURE_GENERALIST_RUN_MODE_EVAL}:
        return "adventure_eval"
    if run_mode == RUN_MODE_LEVEL3_SPECIALIST:
        if bool(getattr(args, "level3_eval", False)) or (
            bool(getattr(args, "eval", False)) and not bool(getattr(args, "train", False))
        ):
            return "fixed_eval"
        if bool(getattr(args, "level3_train", False)) or bool(getattr(args, "train", False)):
            return "train"
        return ""
    if run_mode == "fixed_eval":
        return "fixed_eval" if bool(getattr(args, "eval", False)) or configured_mode_explicit else ""
    if run_mode == "fixed_train":
        return "train" if bool(getattr(args, "train", False)) or configured_mode_explicit else ""
    return ""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/evaluate MaskablePPO for the limited PvZRL environment.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--level3-train", action="store_true", help="Train the fixed 4-slot Level 3 specialist mode.")
    parser.add_argument("--level3-eval", action="store_true", help="Evaluate the fixed 4-slot Level 3 specialist mode.")
    parser.add_argument("--target-level", type=int, default=None)
    parser.add_argument("--adventure", action="store_true", help="Alias for --adventure-eval.")
    parser.add_argument("--adventure-eval", action="store_true", help="Run inference-only Adventure progression with a loaded PPO model.")
    parser.add_argument(
        "--adventure-generalist-train",
        action="store_true",
        help="Train the 14-slot identity Adventure Generalist (fresh when no resume model is provided).",
    )
    parser.add_argument("--adventure-generalist-eval", action="store_true", help="Evaluate a 14-slot identity Adventure Generalist in Adventure progression.")
    parser.add_argument(
        "--run-mode",
        choices=(
            "fixed_train",
            "fixed_eval",
            RUN_MODE_ADVENTURE_EVAL,
            RUN_MODE_LEVEL3_SPECIALIST,
            ADVENTURE_GENERALIST_RUN_MODE_TRAIN,
            ADVENTURE_GENERALIST_RUN_MODE_EVAL,
        ),
    )
    parser.add_argument("--check-deps", action="store_true")
    parser.add_argument("--metadata-dry-run", action="store_true", help="Validate model metadata/action compatibility without starting the game.")
    parser.add_argument(
        "--allow-missing-model-metadata",
        action="store_true",
        help="Allow legacy config-based metadata inference for old runs. Off by default to prevent silent seed-slot drift.",
    )
    parser.add_argument("--model-schedule", type=Path, default=None, help="Optional staged Adventure model schedule.")
    parser.add_argument("--router-dry-run", action="store_true", help="Validate a model schedule decision without starting the game.")
    parser.add_argument("--dry-run-level", default=None)
    parser.add_argument("--dry-run-unlocked-seeds", default=None)
    parser.add_argument("--dry-run-available-seeds", default=None)
    parser.add_argument("--experimental-dynamic-seed-slots", action="store_true")
    parser.add_argument("--action-space-mode", default=None)
    parser.add_argument("--model", "--model-path", dest="model", type=Path)
    parser.add_argument(
        "--resume-model-path",
        type=Path,
        default=None,
        help="Training-only checkpoint/model .zip to continue PPO learning from (adds timesteps; does not overwrite source).",
    )
    parser.add_argument("--episodes", type=int, default=5)
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
    parser.add_argument("--max-steps", type=int)
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
    parser.add_argument("--plant-types")
    parser.add_argument("--auto-select-seeds", action="store_true")
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
    parser.add_argument("--conservative-seeds", dest="conservative_seeds", action="store_true", default=True)
    parser.add_argument("--no-conservative-seeds", dest="conservative_seeds", action="store_false")
    parser.add_argument("--allow-new-plants", action="store_true", default=False)
    parser.add_argument("--unlock-aware-seed-curriculum", action="store_true")
    parser.add_argument("--seed-curriculum", choices=("conservative", "varied"), default=None)
    parser.add_argument("--randomize-seed-order", action="store_true")
    parser.add_argument("--unlock-introduction-delay", type=int, default=None)
    parser.add_argument("--new-plant-min-inclusion-prob", type=float, default=None)
    parser.add_argument("--infer-capacity-from-unlocks", dest="infer_capacity_from_unlocks", action="store_true", default=None)
    parser.add_argument("--no-infer-capacity-from-unlocks", dest="infer_capacity_from_unlocks", action="store_false")
    parser.add_argument("--allow-weak-unlocked-capacity-fallback", action="store_true")
    parser.add_argument("--adventure-replay-cleared-levels", action="store_true")
    parser.add_argument("--adventure-frontier-sample-prob", type=float, default=None)
    parser.add_argument("--adventure-recent-cleared-sample-prob", type=float, default=None)
    parser.add_argument("--adventure-maintenance-sample-prob", type=float, default=None)
    parser.add_argument("--adventure-frontier-win-streak-required", type=int, default=None)
    parser.add_argument("--live-status-path", type=Path, default=Path("runs/live_status.json"))
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
    parser.add_argument("--human-coach-bonus", type=float, default=None, help="Legacy alias for --coach-legal-execution-reward.")
    parser.add_argument("--human-coach-match-bonus", type=float, default=None, help="Legacy alias for --coach-match-reward.")
    parser.add_argument("--human-coach-override-penalty", type=float, default=None, help="Legacy alias for --coach-override-penalty.")
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

    if args.level3_train and args.level3_eval:
        raise SystemExit("blocked_reason=invalid_cli: --level3-train and --level3-eval are mutually exclusive.")
    if args.adventure_generalist_train and args.adventure_generalist_eval:
        raise SystemExit(
            "blocked_reason=invalid_cli: --adventure-generalist-train and --adventure-generalist-eval are mutually exclusive."
        )
    if (args.level3_train or args.level3_eval) and (
        args.adventure_eval or args.adventure or args.adventure_generalist_train or args.adventure_generalist_eval
    ):
        raise SystemExit("blocked_reason=invalid_cli: Level 3 specialist does not use Adventure Eval progression.")
    if (args.adventure_eval or args.adventure) and args.adventure_generalist_train:
        raise SystemExit("blocked_reason=invalid_cli: Adventure Generalist training is separate from Adventure Eval.")
    if args.level3_train:
        args.train = True
    if args.level3_eval:
        args.eval = True
    if args.adventure_generalist_train:
        args.train = True

    if args.check_deps:
        return check_deps()

    raw_config = load_json(args.config)
    config = build_config(args, raw_config)
    if args.metadata_dry_run:
        model_path = args.model or Path(config["model_path"] or Path(config["run_dir"]) / "model.zip")
        return metadata_dry_run(config, model_path)
    if args.router_dry_run:
        return router_dry_run(config, args)
    configured_mode_explicit = bool(getattr(args, "run_mode", None) or "run_mode" in raw_config)
    execution_route = execution_route_for_config(
        config,
        args,
        configured_mode_explicit=configured_mode_explicit,
    )
    if execution_route == "adventure_eval" and args.train:
        raise SystemExit("Adventure eval is inference-only and cannot be combined with --train.")
    if config.get("run_mode") == ADVENTURE_GENERALIST_RUN_MODE_TRAIN and args.eval:
        raise SystemExit(
            "Adventure Generalist training cannot be combined with generic --eval; "
            "run Adventure Generalist eval separately."
        )
    if config.get("run_mode") == RUN_MODE_LEVEL3_SPECIALIST and execution_route:
        candidate_model = args.model or Path(config["model_path"] or Path(config["run_dir"]) / "model.zip")
        verify_level3_specialist_start_state(config, args.live_status_path, candidate_model)
    if execution_route == "train":
        run_dir = Path(config["run_dir"])
        with TeeOutput(run_dir / "train.log"):
            print(f"Run directory: {run_dir}")
            train(config, args.live_status_path)
            if args.eval:
                model_path = args.model or run_dir / "model.zip"
                if not model_path.exists() and (run_dir / "final_model.zip").exists():
                    model_path = run_dir / "final_model.zip"
                if not model_path.exists():
                    raise SystemExit(f"Model not found: {model_path}")
                evaluate(config, model_path, args.episodes, args.live_status_path)
            return 0
    elif execution_route == "fixed_eval":
        model_path = args.model or Path(config["run_dir"]) / "model.zip"
        if not model_path.exists() and (Path(config["run_dir"]) / "final_model.zip").exists():
            model_path = Path(config["run_dir"]) / "final_model.zip"
        if not model_path.exists():
            raise SystemExit(f"Model not found: {model_path}")
        with TeeOutput(Path(config["run_dir"]) / "train.log"):
            evaluate(config, model_path, args.episodes, args.live_status_path)
        return 0
    elif execution_route == "adventure_eval":
        model_path = args.model or Path(config["run_dir"]) / "model.zip"
        if args.model_schedule is not None:
            model_path = args.model or Path("")
        elif not model_path.exists() and (Path(config["run_dir"]) / "final_model.zip").exists():
            model_path = Path(config["run_dir"]) / "final_model.zip"
        if args.model_schedule is None and not model_path.exists():
            raise SystemExit(f"Model not found for Adventure eval: {model_path}")
        with TeeOutput(Path(config["run_dir"]) / "train.log"):
            adventure_evaluate(config, model_path, args)
        return 0
    if not execution_route:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
