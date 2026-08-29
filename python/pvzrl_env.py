"""Adventure Generalist environment and diagnostic tools for the PvZRL bridge.

The MelonLoader mod owns direct Unity/IL2CPP access. This Python module owns
Generalist reset/step/observe/reward/done behavior and computes rewards from
observation deltas first. It intentionally does not start PPO.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import socket
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from pvzrl_actions import (
    ActionDecision,
    ActionDecisionCache,
    ActionIntent,
    ActionResult,
    ActionValidationConfig,
    build_action_decision_cache,
    build_action_intent,
    build_action_validation_context,
    compatible_pairs_for_observation,
    validate_action_intent,
)
from pvzrl_action_space import (
    ACTION_SPACE_ADVENTURE_14_IDENTITY,
    ADVENTURE_IDENTITY_ACTION_COUNT,
    ADVENTURE_IDENTITY_MAX_SEED_SLOTS,
    CELLS_PER_SLOT,
    DEFAULT_COLS,
    DEFAULT_ROWS,
    is_supported_live_board_geometry,
)
from pvzrl_fusion import (
    FUSION_POLICY_NONE,
    FUSION_POLICY_SCRIPTED,
    FUSION_SOURCE_MODEL,
    FUSION_SOURCE_SCRIPTED,
    FusionDecision,
    FusionIntent,
    account_fusion_execution_once,
    are_fusion_compatible,
    build_fusion_diagnostics,
    choose_scripted_fusion_candidate,
    compact_candidate,
    count_tough_zombies_by_row,
    default_fusion_diagnostics,
    enforce_fusion_scope_contract,
    fusion_compatibility_table,
    fusion_live_fields,
    fusion_execution_from_result,
    fusion_intent_from_candidate,
    get_fusion_illegal_reason,
    merge_episode_fusion_stats,
    normalize_fusion_policy,
    plant_name as fusion_plant_name,
    validate_fusion_intent,
    validate_scripted_fusion_candidate,
)
from pvzrl_registry import (
    DEFAULT_PLANT_REGISTRY_PATH,
    get_plant_registry,
    normalize_plant_name,
)
from pvzrl_observation_facts import StepFacts, StepFactsCache, build_step_facts
from pvzrl_diagnostics import (
    append_safety_event,
    compose_environment_safety_diagnostics,
)
from pvzrl_lane_diagnostics import LaneDiagnosticsInput, compose_lane_diagnostics
from pvzrl_lifecycle import (
    LIFECYCLE_ACTIVE_GAMEPLAY,
    LIFECYCLE_LOSS_PENDING,
    LIFECYCLE_POST_WIN_PENDING,
    LIFECYCLE_READY,
    LIFECYCLE_RESETTING,
    LIFECYCLE_UNKNOWN,
    active_gameplay_progress,
    cleanup_signal_active,
    confirmed_active_gameplay,
    confirmed_loss,
    confirmed_post_win,
    dirty_active_board,
    fresh_playable_board,
    legacy_done_reason,
    legacy_lifecycle_state,
    live_board_progress,
    seed_selection_visible,
    stale_cleanup_ui,
)
from pvzrl_runtime_state import ResetRuntimeState
from pvzrl_rewards import (
    REWARD_COMPONENT_FIELDS,
    REWARD_EPISODE_TOTAL_FIELDS,
    REWARD_POLICY_VERSION,
    RewardComposition,
    RewardCompositionState,
    RewardConfig,
    compose_environment_reward,
    compose_step_reward,
    fusion_reward_live_fields,
    _active_threat_rows as active_threat_rows_from_facts,
    _lane_danger_by_row as lane_danger_by_row_from_facts,
    _mower_risk_rows as mower_risk_rows_from_facts,
    _nearby_zombie_context as nearby_zombie_context_from_facts,
    _nearest_zombie_x_by_row as nearest_zombie_x_by_row_from_facts,
    _plant_counts_by_row as plant_counts_by_row_from_facts,
    _valuable_plant_columns as valuable_plant_columns_from_facts,
    _wallnut_blocker_count as wallnut_blocker_count_from_facts,
)


GENERALIST_INITIAL_PLANT_TYPES = [1, 1, 0, 0]
GENERALIST_INITIAL_SEED_LIST = ["SunFlower", "SunFlower", "Peashooter", "Peashooter"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLANT_REGISTRY_PATH = DEFAULT_PLANT_REGISTRY_PATH
LAWN_STRINGS_PATH = PROJECT_ROOT / "Game Files" / "Mods" / "PvZ_Fusion_Translator" / "Dumps" / "LawnStrings.json"
RUN_MODE_ADVENTURE_GENERALIST_14SLOT_TRAIN = "adventure_generalist_14slot_train"
RUN_MODE_ADVENTURE_GENERALIST_14SLOT_EVAL = "adventure_generalist_14slot_eval"
ADVENTURE_GENERALIST_RUN_MODES = {
    RUN_MODE_ADVENTURE_GENERALIST_14SLOT_TRAIN,
    RUN_MODE_ADVENTURE_GENERALIST_14SLOT_EVAL,
}
LIFECYCLE_RESET_CLEANUP_ALLOWED = {
    LIFECYCLE_POST_WIN_PENDING,
    LIFECYCLE_LOSS_PENDING,
    LIFECYCLE_RESETTING,
}


@dataclass(frozen=True, slots=True)
class _DeadlinePollResult:
    matched: bool
    last_value: Dict[str, Any]


def _poll_dict_until(
    *,
    deadline: float,
    probe: Callable[[], Dict[str, Any]],
    accept: Callable[[Dict[str, Any]], bool],
    poll_seconds: float,
    retry_exceptions: Tuple[type, ...] = (),
    after_reject: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> _DeadlinePollResult:
    """Poll a dictionary probe without owning caller-specific transition policy."""

    last_value: Dict[str, Any] = {}
    while time.monotonic() < deadline:
        try:
            value = probe()
        except retry_exceptions:
            time.sleep(poll_seconds)
            continue
        last_value = value
        if accept(value):
            return _DeadlinePollResult(True, value)
        if after_reject is not None:
            after_reject(value)
        time.sleep(poll_seconds)
    return _DeadlinePollResult(False, last_value)


@dataclass
class PvZEnvConfig:
    host: str = "127.0.0.1"
    port: int = 32323
    timeout: float = 10.0
    enable_action_watchdog: bool = True
    action_timeout_seconds: float = 10.0
    save_freeze_debug_bundle: bool = True
    step_seconds: float = 0.25
    plant_types: List[int] = field(default_factory=lambda: list(GENERALIST_INITIAL_PLANT_TYPES))
    row_count: int = DEFAULT_ROWS
    column_count: int = DEFAULT_COLS
    game_speed: float = 1.0
    game_speed_mode: str = "game_speed"
    seed: int = 12345
    start_sun: Optional[int] = 500
    reset_mode: str = "soft"
    reset_wait_timeout: float = 30.0
    reset_poll_seconds: float = 0.2
    auto_select_seeds: bool = True
    seed_list: List[str] = field(default_factory=lambda: list(GENERALIST_INITIAL_SEED_LIST))
    seed_click_delay: float = 0.35
    lets_rock_delay: float = 0.5
    post_start_delay: float = 1.0
    seed_screen_check_interval: int = 100
    debug_performance: bool = False
    debug_observation: bool = False
    debug_sun: bool = False
    debug_sun_sample_interval: int = 25
    fusion_policy: str = FUSION_POLICY_NONE
    # When True, occupied tiles holding a fusion-compatible plant become legal
    # *fuse* actions in the model action mask (and the model's placement action
    # on such a tile is routed to the fusion bridge).  Default False preserves
    # the historical behavior where every occupied tile is illegal.
    fusion_action_mask_enabled: bool = False
    enable_fusion_chain_rewards: bool = False
    enable_recipe_discovery_reward: bool = False
    enable_repeat_recipe_decay: bool = False
    run_mode: str = RUN_MODE_ADVENTURE_GENERALIST_14SLOT_TRAIN
    tactical_masks: bool = False
    wallnut_tactical_mask: bool = False
    cherrybomb_tactical_mask: bool = False
    game_exe: Optional[str] = None
    reward: RewardConfig = field(default_factory=RewardConfig)


class BridgeTimeoutError(TimeoutError):
    """Raised when the Unity bridge does not answer a request in time."""

    def __init__(self, command: str, timeout: float, message: str = ""):
        self.command = command
        self.timeout = float(timeout)
        detail = message or f"PvZRL bridge request timed out waiting for command={command!r} after {timeout:.1f}s."
        super().__init__(detail)


@dataclass
class EpisodeLog:
    policy: str
    episode_index: int
    reward_policy_version: str = REWARD_POLICY_VERSION
    episode_reward: float = 0.0
    terminal_reason: str = ""
    done_reason: str = "none"
    episode_length: int = 0
    illegal_actions: int = 0
    plants_placed: int = 0
    zombies_killed: int = 0
    final_wave: int = 0
    max_wave: int = 0
    sun_spent: int = 0
    sun_remaining: int = 0
    mowers_lost: int = 0
    reset_success: bool = True
    reset_seconds: float = 0.0
    avg_legal_actions: float = 0.0
    reset_failures: int = 0
    bridge_errors: int = 0
    average_reward_per_step: float = 0.0
    reward_component_total: float = 0.0
    reward_unattributed_adjustment_total: float = 0.0
    reward_components_match: bool = True
    threat_raw_before: float = 0.0
    threat_raw_after: float = 0.0
    threat_before: float = 0.0
    threat_after: float = 0.0
    threat_raw_delta: float = 0.0
    threat_normalized_delta: float = 0.0
    threat_clipped_delta: float = 0.0
    won: bool = False
    lost: bool = False
    timed_out: bool = False
    actual_terminal: bool = False
    error: str = ""
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
    threat_delta_reward_total: float = 0.0
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
    coach_match_reward_total: float = 0.0
    coach_legal_execution_reward_total: float = 0.0
    coach_override_penalty_total: float = 0.0
    coach_fusion_success_reward_total: float = 0.0
    coach_tactical_usefulness_reward_total: float = 0.0
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
    wallnut_threatened_lane_count: int = 0
    wallnut_between_zombie_and_house_count: int = 0
    wallnut_frontline_count: int = 0
    wallnut_emergency_block_count: int = 0
    wallnut_damage_absorbed_total: float = 0.0
    cherrybomb_used_count: int = 0
    cherrybomb_kills_total: int = 0
    cherrybomb_avg_kills_per_use: float = 0.0
    cherrybomb_zero_kill_count: int = 0
    cherrybomb_cluster_use_count: int = 0
    cherrybomb_emergency_use_count: int = 0
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
    high_danger_unanswered_steps: int = 0
    mower_exposure_steps: int = 0
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
    fusion_attempts_by_pair: Dict[str, int] = field(default_factory=dict)
    fusion_successes_by_pair: Dict[str, int] = field(default_factory=dict)
    fusion_failures_by_pair: Dict[str, int] = field(default_factory=dict)
    fusion_result_counts: Dict[str, int] = field(default_factory=dict)
    fusion_depth_counts: Dict[str, int] = field(default_factory=dict)
    recursive_fusion_count: int = 0
    high_tier_fusion_count: int = 0
    highest_fusion_tier: int = 0
    fusion_under_threat_count: int = 0
    fusion_near_buckethead_count: int = 0
    fusion_near_conehead_count: int = 0
    fusion_estimated_mower_save_count: int = 0
    fusion_kills_after_use_total: int = 0
    fusion_avg_kills_after_use: float = 0.0
    fusion_bridge_error_count: int = 0
    fusion_unsafe_state_block_count: int = 0
    # Fusion reward accounting (episode totals). fusion_reward_total also arrives
    # via the reward breakdown; the component totals below are merged from the
    # cumulative fusion diagnostics at episode end.
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


class PvZBridgeClient:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 32323,
        timeout: float = 10.0,
        debug_performance: bool = False,
        action_timeout: Optional[float] = None,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.debug_performance = debug_performance
        self.action_timeout = max(0.1, float(action_timeout if action_timeout is not None else timeout))
        self._sock: Optional[socket.socket] = None
        self._reader = None
        self._writer = None

    def connect(self) -> None:
        if self._sock is not None:
            return
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        sock.settimeout(self.timeout)
        self._sock = sock
        self._reader = sock.makefile("r", encoding="utf-8", newline="\n")
        self._writer = sock.makefile("w", encoding="utf-8", newline="\n")

    def close(self) -> None:
        for handle in (self._reader, self._writer):
            try:
                if handle is not None:
                    handle.close()
            except OSError:
                pass
        try:
            if self._sock is not None:
                self._sock.close()
        except OSError:
            pass
        self._sock = None
        self._reader = None
        self._writer = None

    def request(self, command: str, **payload: Any) -> Dict[str, Any]:
        started = time.perf_counter()
        self.connect()
        request_timeout = self.action_timeout if command in {"step", "fusion_step"} else self.timeout
        if self._sock is not None:
            self._sock.settimeout(request_timeout)
        assert self._reader is not None and self._writer is not None
        message = {"command": command, **payload}
        try:
            self._writer.write(json.dumps(message, separators=(",", ":")) + "\n")
            self._writer.flush()
            line = self._reader.readline()
        except (socket.timeout, TimeoutError) as exc:
            self.close()
            raise BridgeTimeoutError(
                command,
                request_timeout,
                f"PvZRL bridge request timed out waiting for command={command!r} after {request_timeout:.1f}s.",
            ) from exc
        if not line:
            raise ConnectionError("PvZRL bridge disconnected.")
        response = json.loads(line)
        if not response.get("ok", False) and str(response.get("error") or "").strip().lower() == "timeout":
            self.close()
            raise BridgeTimeoutError(
                command,
                request_timeout,
                str(response.get("details") or "Unity main thread did not process the action before the bridge deadline."),
            )
        if not response.get("ok", False):
            raise RuntimeError(f"bridge error: {response.get('error')} {response.get('details')}")
        data = response.get("data", {})
        if self.debug_performance and isinstance(data, dict):
            data["bridge_roundtrip_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        return data


def load_plant_registry(path: Path = PLANT_REGISTRY_PATH) -> Dict[str, Any]:
    """Return the historical mutable payload backed by one cached parse."""

    return get_plant_registry(path).to_legacy_payload()


def registry_entries(path: Path = PLANT_REGISTRY_PATH) -> List[Dict[str, Any]]:
    """Return compatibility dicts while the immutable registry becomes canonical."""

    return [definition.to_legacy_dict() for definition in get_plant_registry(path).plants]


def registry_entry_by_type(plant_type: int) -> Optional[Dict[str, Any]]:
    definition = get_plant_registry().get_by_id(plant_type)
    return definition.to_legacy_dict() if definition is not None else None


def resolve_seed_list(seed_list: List[str]) -> List[int]:
    registry = get_plant_registry()

    resolved: List[int] = []
    unknown: List[str] = []
    for raw in seed_list:
        for token in expand_seed_token(raw):
            if token.lstrip("-").isdigit():
                resolved.append(int(token))
                continue
            plant_type = registry.resolve_name(token)
            if plant_type is None:
                unknown.append(token)
                continue
            resolved.append(plant_type)

    if unknown:
        raise ValueError(f"Unknown seed names in --seed-list: {unknown}. Add aliases to {PLANT_REGISTRY_PATH}.")
    if not resolved:
        raise ValueError("--seed-list did not resolve to any plant type IDs")
    return resolved


def expand_seed_token(raw: str) -> List[str]:
    token = raw.strip()
    if not token:
        return []
    if ":" not in token:
        return [token]
    name, count_text = token.rsplit(":", 1)
    name = name.strip()
    count_text = count_text.strip()
    if not name:
        raise ValueError(f"Invalid --seed-list entry {raw!r}: missing seed name before ':'.")
    if not count_text.isdigit():
        raise ValueError(f"Invalid --seed-list entry {raw!r}: duplicate count must be a positive integer.")
    count = int(count_text)
    if count <= 0:
        raise ValueError(f"Invalid --seed-list entry {raw!r}: duplicate count must be greater than zero.")
    return [name] * count


def parse_seed_list(raw: str) -> List[str]:
    seeds: List[str] = []
    for part in raw.split(","):
        seeds.extend(expand_seed_token(part))
    return seeds


def strip_rich_text(value: str) -> str:
    value = re.sub(r"<[^>]+>", "", value)
    return value.replace("\r\n", "\n").strip()


def load_lawn_almanac(path: Path = LAWN_STRINGS_PATH) -> Dict[int, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    result: Dict[int, Dict[str, Any]] = {}
    for plant in data.get("plants", []):
        try:
            seed_type = int(plant.get("seedType"))
        except Exception:
            continue
        cost_text = str(plant.get("cost", ""))
        red_numbers = re.findall(r"<color=red>([\d.]+)", cost_text)
        cost = int(float(red_numbers[0])) if len(red_numbers) >= 1 else None
        cooldown = float(red_numbers[1]) if len(red_numbers) >= 2 else None
        result[seed_type] = {
            "plant_type": seed_type,
            "display_name": str(plant.get("name", "")),
            "description": strip_rich_text(str(plant.get("info") or plant.get("introduce") or "")),
            "cost": cost,
            "cooldown": cooldown,
            "source": str(path),
        }
    return result


def enrich_almanac_probe(runtime: Dict[str, Any]) -> Dict[str, Any]:
    registry_by_type = {
        int(entry.get("plant_type_id", -999)): entry for entry in registry_entries()
    }
    almanac_by_type = load_lawn_almanac()
    enriched_plants: List[Dict[str, Any]] = []
    for plant in runtime.get("plants", []):
        plant_type = int(plant.get("plantType", -999))
        registry = registry_by_type.get(plant_type, {})
        almanac = almanac_by_type.get(plant_type, {})
        runtime_cost = plant.get("cost")
        runtime_full_cd = plant.get("fullCooldown")
        enriched = dict(plant)
        enriched.update(
            {
                "canonicalName": registry.get("canonical_name") or plant.get("plantTypeName"),
                "aliases": registry.get("aliases", []),
                "role": registry.get("role"),
                "enabledForTraining": bool(registry.get("enabled_for_training", False)),
                "registryCost": registry.get("cost"),
                "registryCooldown": registry.get("cooldown"),
                "almanacDisplayName": almanac.get("display_name"),
                "almanacCost": almanac.get("cost"),
                "almanacCooldown": almanac.get("cooldown"),
                "description": registry.get("description") or almanac.get("description") or plant.get("description"),
                "metadataSources": {
                    "runtime": plant.get("metadataSource"),
                    "registry": str(PLANT_REGISTRY_PATH) if registry else None,
                    "almanacDump": almanac.get("source"),
                },
                "costMatchesRuntime": (
                    registry.get("cost") is not None
                    and runtime_cost is not None
                    and int(registry.get("cost")) == int(runtime_cost)
                ),
                "cooldownMatchesRuntime": (
                    registry.get("cooldown") is not None
                    and runtime_full_cd is not None
                    and abs(float(registry.get("cooldown")) - float(runtime_full_cd or 0.0)) < 0.01
                ),
            }
        )
        enriched_plants.append(enriched)
    runtime["plants"] = enriched_plants
    runtime["registryPath"] = str(PLANT_REGISTRY_PATH)
    runtime["almanacDumpPath"] = str(LAWN_STRINGS_PATH)
    return runtime


def enrich_seed_probe(probe: Dict[str, Any]) -> Dict[str, Any]:
    registry_by_type = {
        int(entry.get("plant_type_id", -999)): entry for entry in registry_entries()
    }

    def enrich_cards(raw_cards: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        cards = []
        unknown = []
        for card in raw_cards:
            plant_type = int(card.get("plantType", -999))
            registry = registry_by_type.get(plant_type)
            enriched = dict(card)
            if registry:
                enriched["matchedRegistryEntry"] = registry.get("canonical_name")
                enriched["role"] = registry.get("role")
                enriched["enabledForTraining"] = bool(registry.get("enabled_for_training", False))
            else:
                enriched["matchedRegistryEntry"] = None
                unknown.append(card)
            cards.append(enriched)
        return cards, unknown

    cards, unknown = enrich_cards(list(probe.get("cards", [])))
    probe["cards"] = cards
    for key in (
        "availableSeedCards",
        "selectedSeedBankCards",
        "activeGameplayCardBankCards",
        "stalePreselectedCards",
        "runtimeCardWrappers",
    ):
        enriched, _ = enrich_cards(list(probe.get(key, [])))
        probe[key] = enriched
    probe["unknownOrUnmatchedCards"] = unknown
    probe["registryPath"] = str(PLANT_REGISTRY_PATH)
    return probe


def count_values(values: List[int]) -> Counter:
    return Counter(int(value) for value in values)


def count_cards(cards: List[Dict[str, Any]]) -> Counter:
    return Counter(int(card.get("plantType", -999)) for card in cards)


def count_entries(entries: List[Dict[str, Any]]) -> Counter:
    counts: Counter = Counter()
    for entry in entries:
        plant_type = int(entry.get("plantType", -999))
        counts[plant_type] += int(entry.get("count", 0))
    return counts


def type_counts_from_probe(probe: Dict[str, Any], card_key: str, count_key: str) -> Counter:
    cards = list(probe.get(card_key, []))
    if cards:
        return count_cards(cards)
    entries = list(probe.get(count_key, []))
    if entries:
        return count_entries(entries)
    return Counter()


def counts_cover(actual: Counter, expected: Counter) -> bool:
    return all(actual.get(plant_type, 0) >= count for plant_type, count in expected.items())


def format_counts(counts: Counter) -> str:
    if not counts:
        return "{}"
    registry = get_plant_registry()
    parts = []
    for plant_type in sorted(counts):
        name = registry.canonical_name(int(plant_type))
        parts.append(f"{name}({plant_type})={counts[plant_type]}")
    return "{" + ", ".join(parts) + "}"


def plant_type_name(plant_type: int) -> str:
    return get_plant_registry().canonical_name(int(plant_type))


def counts_to_entries(counts: Counter) -> List[Dict[str, Any]]:
    return [
        {
            "plantType": int(plant_type),
            "plantTypeName": plant_type_name(int(plant_type)),
            "count": int(count),
        }
        for plant_type, count in sorted(counts.items())
    ]


def selected_bank_state(probe: Dict[str, Any]) -> Tuple[Counter, int]:
    counts = type_counts_from_probe(probe, "selectedSeedBankCards", "selectedBankPlantTypeCounts")
    return counts, int(probe.get("selectedBankVisibleCount", sum(counts.values())))


def active_gameplay_bank_state(probe: Dict[str, Any]) -> Tuple[Counter, int]:
    counts = type_counts_from_probe(
        probe,
        "activeGameplayCardBankCards",
        "activeGameplayCardBankPlantTypeCounts",
    )
    return counts, sum(counts.values())


def ordered_runtime_seed_slot_types(payload: Dict[str, Any]) -> List[int]:
    """Return the exact runtime slot sequence from a canonical bridge payload.

    Gameplay observations expose ``seedSlots`` while seed probes expose
    ``activeGameplaySeedSlots``.  Both are slot-indexed runtime identities;
    raw CardUI scan order and configured startup metadata are deliberately not
    accepted as substitutes.
    """

    if not isinstance(payload, dict):
        return []
    raw_slots: Any = payload.get("seedSlots")
    if not isinstance(raw_slots, list) or not raw_slots:
        raw_slots = payload.get("activeGameplaySeedSlots")
    if not isinstance(raw_slots, list) or not raw_slots:
        return []

    indexed: List[Tuple[int, int]] = []
    for fallback_index, raw_slot in enumerate(raw_slots):
        if not isinstance(raw_slot, dict):
            return []
        try:
            slot_index = int(raw_slot.get("slotIndex", fallback_index))
            plant_type = int(raw_slot.get("plantType", -999))
        except (TypeError, ValueError, OverflowError):
            return []
        if slot_index < 0 or plant_type < 0:
            return []
        indexed.append((slot_index, plant_type))

    indexed.sort(key=lambda item: item[0])
    if [slot_index for slot_index, _ in indexed] != list(range(len(indexed))):
        return []
    return [plant_type for _, plant_type in indexed]


def gameplay_slot_identity_verification(
    observation: Dict[str, Any],
    requested_sequence: List[int],
) -> Dict[str, Any]:
    """Verify the ordered gameplay slots on one canonical gameplay frame."""

    expected = [int(value) for value in requested_sequence]
    actual = ordered_runtime_seed_slot_types(observation)
    gameplay_ready = bool(
        isinstance(observation, dict)
        and (observation.get("gameplayReady") or observation.get("isGameplayReady"))
        and not (observation.get("seedSelectionActive") or observation.get("isSeedSelectionScreen"))
    )
    exact_identity = bool(actual and actual == expected)
    if not gameplay_ready:
        reason = "canonical_gameplay_not_ready"
    elif not actual:
        reason = "ordered_runtime_seed_slots_missing"
    elif actual != expected:
        reason = "ordered_runtime_seed_slot_identity_mismatch"
    else:
        reason = "ok"
    return {
        "success": bool(gameplay_ready and exact_identity),
        "source": "finalObservation.seedSlots",
        "gameplayReady": bool(gameplay_ready),
        "requestedSeedTypes": expected,
        "activeSeedTypes": actual,
        "activeSeedSlotCount": len(actual),
        "reason": reason,
    }


def seed_slots_from_observation(observation: Dict[str, Any], fallback_plant_types: List[int]) -> List[Dict[str, Any]]:
    slots = list(observation.get("seedSlots", []) or [])
    if slots:
        return slots
    return [
        {
            "slotIndex": index,
            "plantType": int(plant_type),
            "plantTypeName": plant_type_name(int(plant_type)),
            "seedCost": 0,
            "ready": False,
            "usable": False,
            "warning": "seedSlots missing from observation; using configured slot placeholder",
        }
        for index, plant_type in enumerate(fallback_plant_types)
    ]


def is_restart_screen_observation(observation: Dict[str, Any]) -> bool:
    if observation.get("onPauseMenu") or observation.get("pauseMenuActive"):
        return False
    return bool(
        observation.get("onGameOverScreen")
        or observation.get("lossMenuActive")
        or (
            observation.get("onRestartScreen")
            and observation.get("gameOverTextVisible")
        )
    )


def missing_values(requested: List[int], selected: List[int]) -> List[int]:
    selected_counts = count_values(selected)
    missing: List[int] = []
    for plant_type in requested:
        count = selected_counts.get(plant_type, 0)
        if count > 0:
            selected_counts[plant_type] = count - 1
        else:
            missing.append(int(plant_type))
    return missing


class PvZGymEnv:
    """Small Gymnasium-like wrapper around the synchronous bridge."""

    def __init__(self, config: Optional[PvZEnvConfig] = None):
        self.config = config or PvZEnvConfig()
        self.config.fusion_policy = normalize_fusion_policy(getattr(self.config, "fusion_policy", FUSION_POLICY_NONE))
        requested_run_mode = str(getattr(self.config, "run_mode", "") or "").strip().lower()
        if requested_run_mode not in ADVENTURE_GENERALIST_RUN_MODES:
            raise ValueError(
                f"Unsupported run_mode: {self.config.run_mode!r}; expected Adventure Generalist "
                f"train/eval mode ({sorted(ADVENTURE_GENERALIST_RUN_MODES)!r})"
            )
        if int(self.config.row_count) != DEFAULT_ROWS or int(self.config.column_count) != DEFAULT_COLS:
            raise ValueError(
                "Adventure Generalist requires a fixed 6x10 model board: "
                f"row_count={self.config.row_count}, column_count={self.config.column_count}"
            )
        if len(self.config.plant_types) > ADVENTURE_IDENTITY_MAX_SEED_SLOTS:
            raise ValueError(
                "Adventure Generalist supports at most 14 configured seed identities: "
                f"plant_types={len(self.config.plant_types)}"
            )
        if len(self.config.seed_list) > ADVENTURE_IDENTITY_MAX_SEED_SLOTS:
            raise ValueError(
                "Adventure Generalist supports at most 14 configured seed names: "
                f"seed_list={len(self.config.seed_list)}"
            )
        self.config.run_mode = requested_run_mode
        self.client = PvZBridgeClient(
            self.config.host,
            self.config.port,
            self.config.timeout,
            debug_performance=self.config.debug_performance,
            action_timeout=(
                self.config.action_timeout_seconds
                if self.config.enable_action_watchdog
                else self.config.timeout
            ),
        )
        self.previous_observation: Optional[Dict[str, Any]] = None
        self._step_facts_cache = StepFactsCache()
        self._action_decision_cache: Optional[ActionDecisionCache] = None
        self._action_decision_cache_hits = 0
        self._action_decision_cache_misses = 0
        self._action_decision_cache_last_hit = False
        self.process: Optional[subprocess.Popen[Any]] = None
        self._steps_since_seed_screen_check = self.config.seed_screen_check_interval
        self._last_episode_ended_by_timeout = False
        self._last_episode_ended_by_win = False
        self._reset_requires_seed_flow = False
        self._saw_seed_selection_this_reset = False
        self._clicked_lets_rock_this_reset = False
        self._reset_reason = ""
        self._reset_generation_id = 0
        self._accepted_board_requires_seed_gate = False
        self._coach_command_queue_cleared_on_reset = True
        self._startup_command_blocked = False
        self._pending_fusion_command: Optional[Dict[str, Any]] = None
        self._selected_bridge_command: Optional[Dict[str, Any]] = None
        self._executed_coach_command_ids: set[int] = set()
        self._last_executed_coach_command_id: Optional[int] = None
        self._coach_fusion_fresh_after_timestamp = time.time()
        self._reset_reward_episode_state()

    def _run_mode(self) -> str:
        run_mode = str(getattr(self.config, "run_mode", "") or "").strip().lower()
        if run_mode in ADVENTURE_GENERALIST_RUN_MODES:
            return run_mode
        raise ValueError(
            f"Unsupported run_mode: {self.config.run_mode!r}; expected Adventure Generalist "
            f"train/eval mode ({sorted(ADVENTURE_GENERALIST_RUN_MODES)!r})"
        )

    def _reset_reward_episode_state(self) -> None:
        self._possible_win_pending_steps = 0
        self._loss_pending_wait_steps = 0
        self._episode_lost_mower_rows: set[int] = set()
        self._episode_missing_mower_rows: set[int] = set()
        self._episode_mower_baseline_ready = False
        self._last_fusion_diagnostics: Dict[str, Any] = default_fusion_diagnostics(self.config.fusion_policy)
        rows = max(0, int(self.config.row_count))
        previous_state = getattr(self, "_reward_state", None)
        recipes_seen_run = (
            previous_state.fusion.recipes_seen_run
            if isinstance(previous_state, RewardCompositionState)
            else ()
        )
        self._reward_state = RewardCompositionState.initial(
            rows,
            recipes_seen_run=recipes_seen_run,
        )
        self._fusion_pipeline_epoch = int(getattr(self, "_fusion_pipeline_epoch", 0)) + 1
        self._fusion_pipeline_event_counter = 0
        self._fusion_accounted_event_ids: set[str] = set()

    def clear_coach_runtime_state(
        self,
        *,
        queue_cleared: bool = True,
        startup_command_blocked: bool = False,
        reason: str = "reset",
    ) -> None:
        del reason
        self._pending_fusion_command = None
        self._selected_bridge_command = None
        self._executed_coach_command_ids.clear()
        self._last_fusion_diagnostics = default_fusion_diagnostics(self.config.fusion_policy)
        self._coach_command_queue_cleared_on_reset = bool(queue_cleared)
        self._startup_command_blocked = bool(startup_command_blocked)
        self._coach_fusion_fresh_after_timestamp = time.time()

    def begin_new_attempt(self, observation: Optional[Dict[str, Any]] = None, reason: str = "") -> None:
        """Reset per-attempt reward and corruption state at a verified board boundary."""
        previous_lost_mower_rows = sorted(getattr(self, "_episode_lost_mower_rows", set()))
        previous_observation = self.previous_observation if isinstance(self.previous_observation, dict) else {}
        self._reset_reward_episode_state()
        self.clear_coach_runtime_state(queue_cleared=True, startup_command_blocked=False, reason=reason or "new_attempt")
        if isinstance(observation, dict):
            self.previous_observation = observation
            self._steps_since_seed_screen_check = 0
        baseline = observation if isinstance(observation, dict) else self.previous_observation
        if not isinstance(baseline, dict):
            baseline = {}
        baseline_rows = max(0, int(self.config.row_count))
        baseline_active_rows = frozenset()
        if baseline:
            baseline_facts = build_step_facts(baseline, self.config.plant_types)
            baseline_rows = max(0, int(baseline_facts.rows or self.config.row_count))
            baseline_active_rows = baseline_facts.mower.active_rows or frozenset()
            self._episode_mower_baseline_ready = bool(
                baseline_rows > 0
                and set(range(baseline_rows)).issubset(baseline_active_rows)
                and int(baseline_facts.mower.logical_count) >= baseline_rows
                and int(baseline_facts.mower.visible_count) >= baseline_rows
            )
        print(
            "[corruption-debug] reset corruption trackers "
            f"reason={reason or 'new_attempt'} "
            f"previous_lost_mower_rows={previous_lost_mower_rows} "
            f"previous_wave={previous_observation.get('wave')} "
            f"previous_plants={previous_observation.get('plantCount')} "
            f"baseline_wave={baseline.get('wave')} "
            f"baseline_plants={baseline.get('plantCount')} "
            f"baseline_zombies={baseline.get('zombieCount')} "
            f"baseline_mowers={baseline.get('logicalMowerCount')} "
            f"screenState={baseline.get('screenState')} "
            f"gameplayReady={baseline.get('gameplayReady')} "
            f"seedSelectionActive={baseline.get('seedSelectionActive')}"
        )
        if baseline:
            print(
                "[corruption-debug] initializing board baseline after new gameplay board "
                f"reason={reason or 'new_attempt'} "
                f"frame={baseline.get('frameCount')} "
                f"wave={baseline.get('wave')}/{baseline.get('maxWave')} "
                f"plantCount={baseline.get('plantCount')} "
                f"zombieCount={baseline.get('zombieCount')} "
                f"mowerCount={baseline.get('logicalMowerCount')} "
                f"nextStep={baseline.get('nextStep')}"
            )
        if self._episode_mower_baseline_ready:
            print(
                "[corruption-baseline] "
                f"mowers_initialized={[row in baseline_active_rows for row in range(baseline_rows)]} "
                "classification=stable_gameplay_entry corruption=False"
            )

    def start_game(self) -> None:
        if not self.config.game_exe:
            return
        if self.process is None or self.process.poll() is not None:
            game_exe = Path(self.config.game_exe).expanduser().resolve()
            self.process = subprocess.Popen(
                [str(game_exe)],
                cwd=str(game_exe.parent),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(5.0)

    def hard_reset(self) -> Dict[str, Any]:
        if not self.config.game_exe:
            return self.client.request("hard_reset")
        game_exe = Path(self.config.game_exe).expanduser().resolve()
        self.client.close()
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
        try:
            subprocess.run(
                ["taskkill", "/IM", game_exe.name, "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10.0,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            pass
        self.client.close()
        if not self._wait_for_bridge_disconnected(timeout=5.0):
            raise RuntimeError(
                "Python-owned hard reset could not verify that the old bridge/game process stopped. "
                "Refusing to continue with a timeout reset against the old active gameplay board."
            )
        self.process = None
        self.start_game()
        bridge = self._wait_for_bridge_available(timeout=max(10.0, min(60.0, self.config.reset_wait_timeout)))
        return {
            "ok": True,
            "message": "Game process restarted by Python wrapper.",
            "gameExe": str(game_exe),
            "bridgeDisconnectedBeforeStart": True,
            "bridge": bridge,
        }

    def _wait_for_bridge_disconnected(self, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + max(0.1, timeout)
        while time.monotonic() < deadline:
            try:
                self.client.request("ping")
            except Exception:
                self.client.close()
                return True
            self.client.close()
            time.sleep(max(0.1, self.config.reset_poll_seconds))
        return False

    def _wait_for_bridge_available(self, timeout: float = 30.0) -> Dict[str, Any]:
        deadline = time.monotonic() + max(1.0, timeout)
        last_error = ""
        while time.monotonic() < deadline:
            try:
                return self.client.request("ping")
            except Exception as exc:
                last_error = str(exc)
                self.client.close()
                time.sleep(max(0.25, self.config.reset_poll_seconds))
        raise TimeoutError(f"Timed out waiting for bridge after game restart. last_error={last_error}")

    def configure(self) -> Dict[str, Any]:
        return self.client.request(
            "configure",
            plant_types=self.config.plant_types,
            row_count=self.config.row_count,
            column_count=self.config.column_count,
            game_speed=self.config.game_speed,
            game_speed_mode=self.config.game_speed_mode,
            seed=self.config.seed,
            seed_screen_check_interval=self.config.seed_screen_check_interval,
            debug_performance=self.config.debug_performance,
            debug_observation=self.config.debug_observation,
            debug_sun=self.config.debug_sun,
            debug_sun_sample_interval=self.config.debug_sun_sample_interval,
        )

    def proof(self, place_test: bool = False, row: int = 0, column: int = 0, plant_type: Optional[int] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"place_test": place_test, "row": row, "column": column}
        if plant_type is not None:
            payload["plant_type"] = plant_type
        return self.client.request("proof", **payload)

    def reset(
        self,
        reset_reason: str = "",
        allow_active_gameplay_reset: bool = False,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        started = time.perf_counter()
        self.configure()
        reset_reason = str(reset_reason or "").strip()
        initial_manual_reset = False
        if not reset_reason:
            previous_observation = self.previous_observation if isinstance(self.previous_observation, dict) else {}
            inferred_reset_reason = ""
            if previous_observation:
                inferred_done_reason = classify_done_reason(previous_observation)
                previous_terminal_hint = str(previous_observation.get("terminalHint") or "")
                if inferred_done_reason in {"win", "loss", "post_win_pending"}:
                    inferred_reset_reason = inferred_done_reason
                elif (
                    bool(previous_observation.get("done"))
                    or bool(previous_observation.get("over"))
                    or previous_terminal_hint == "possible_win"
                    or self._is_confirmed_possible_win(previous_observation)
                    or self._post_win_signal_present(previous_observation)
                ):
                    if (
                        previous_terminal_hint == "possible_win"
                        or self._is_confirmed_possible_win(previous_observation)
                        or self._post_win_signal_present(previous_observation)
                    ):
                        inferred_reset_reason = "win"
                    elif (
                        previous_terminal_hint == "game_over_or_loss"
                        or is_restart_screen_observation(previous_observation)
                    ):
                        inferred_reset_reason = "loss"
                if inferred_reset_reason:
                    reset_reason = inferred_reset_reason
                    print(f"[reset] inferred reset_reason={reset_reason} from previous terminal observation")
            if not reset_reason and self.previous_observation is None:
                reset_reason = "manual"
                initial_manual_reset = True
                allow_active_gameplay_reset = True
            elif not reset_reason and self._has_live_board_progress(self.previous_observation):
                raise RuntimeError("reset called without valid reset_reason during live gameplay")
            elif not reset_reason:
                reset_reason = "manual"
        allowed_reset_reasons = {
            "loss",
            "win",
            "post_win_pending",
            "timeout",
            "env_corruption",
            "manual",
        }
        if reset_reason not in allowed_reset_reasons:
            raise RuntimeError(f"reset called without valid reset_reason={reset_reason}")
        previous_observation = self.previous_observation if isinstance(self.previous_observation, dict) else {}
        timeout_near_win_context = False
        if reset_reason == "timeout" and previous_observation:
            timeout_near_win_context = bool(
                self._post_win_signal_present(previous_observation)
                or self._is_confirmed_possible_win(previous_observation)
                or str(previous_observation.get("terminalHint") or "") == "possible_win"
                or self._classify_lifecycle_state(previous_observation) == LIFECYCLE_POST_WIN_PENDING
            )
            if timeout_near_win_context:
                reset_reason = "post_win_pending"
        if reset_reason in {"win", "post_win_pending"}:
            allow_active_gameplay_reset = False
        timeout_requires_seed_flow = bool(
            reset_reason == "timeout"
            and previous_observation
            and self._timeout_reset_requires_full_seed_flow(previous_observation)
        )
        if timeout_requires_seed_flow:
            allow_active_gameplay_reset = False
        self._last_episode_ended_by_timeout = bool(reset_reason == "timeout")
        self._last_episode_ended_by_win = bool(reset_reason in {"win", "post_win_pending"})
        self._reset_requires_seed_flow = bool(
            timeout_requires_seed_flow
            or reset_reason in {"win", "post_win_pending"}
        )
        self.clear_coach_runtime_state(queue_cleared=True, startup_command_blocked=False, reason=reset_reason)
        self._saw_seed_selection_this_reset = False
        self._clicked_lets_rock_this_reset = False
        self._reset_reason = reset_reason
        self._reset_generation_id += 1
        self._accepted_board_requires_seed_gate = bool(self._reset_requires_seed_flow)
        reset_result: Dict[str, Any] = {
            "ok": True,
            "methodUsed": "state_machine",
            "stages": [],
            "resetReason": reset_reason,
            "allowActiveGameplayReset": bool(allow_active_gameplay_reset),
            "initialManualReset": bool(initial_manual_reset),
            "timeoutNearWinPromotedToPostWinPending": bool(timeout_near_win_context),
            "timeoutRequiresSeedFlow": bool(timeout_requires_seed_flow),
            "resetRequiresSeedFlow": bool(self._reset_requires_seed_flow),
            "resetGenerationId": int(self._reset_generation_id),
            "coach_command_queue_cleared_on_reset": True,
            "startup_command_blocked": False,
        }
        if timeout_requires_seed_flow:
            timeout_context = {
                "screenState": previous_observation.get("screenState"),
                "nextStep": previous_observation.get("nextStep"),
                "wave": previous_observation.get("wave"),
                "maxWave": previous_observation.get("maxWave"),
                "zombieCount": previous_observation.get("zombieCount"),
                "plantCount": previous_observation.get("plantCount"),
                "gameplayReady": previous_observation.get("gameplayReady"),
                "seedSelectionActive": previous_observation.get("seedSelectionActive"),
                "terminalHint": previous_observation.get("terminalHint"),
            }
            reset_result["timeoutContext"] = timeout_context
            print("[reset] timeout episode truncated; next reset requires full seed flow")
            if self.config.game_exe:
                print("[reset] timeout reset using verified hard reset before seed flow")
                try:
                    hard_reset = self.hard_reset()
                    reset_result["hardReset"] = hard_reset
                    reset_result["methodUsed"] = "hard_reset_timeout_seed_flow"
                    self.configure()
                except Exception as exc:
                    reset_result["hardResetError"] = str(exc)
                    reset_result["hardResetFallback"] = "wait_for_terminal_or_seed_flow"
                    print(
                        "[reset] verified hard reset unavailable; "
                        "waiting for terminal/seed flow without unsafe soft reset: "
                        f"{exc}"
                    )
            else:
                reset_result["hardResetRequired"] = True
                reset_result["hardResetFallback"] = "wait_for_terminal_or_seed_flow"
                print(
                    "[reset] no Python-owned hard reset configured; "
                    "waiting for terminal/seed flow without unsafe soft reset"
                )
        if self.previous_observation and self._has_live_board_progress(self.previous_observation):
            if self._post_win_signal_present(self.previous_observation):
                reset_result["resetAfterFalseRewardSignalCount"] = 1
                reset_result["resetAfterFalseRewardSignal"] = {
                    "screenState": self.previous_observation.get("screenState"),
                    "nextStep": self.previous_observation.get("nextStep"),
                    "wave": self.previous_observation.get("wave"),
                    "maxWave": self.previous_observation.get("maxWave"),
                    "zombieCount": self.previous_observation.get("zombieCount"),
                    "plantCount": self.previous_observation.get("plantCount"),
                    "terminalHint": self.previous_observation.get("terminalHint"),
                }
                print(
                    self._format_safety_context(
                        "[safety] reset invoked after false reward/unlock signal during gameplay",
                        self.previous_observation,
                    )
                )
                if reset_reason in {"win", "post_win_pending"}:
                    raise RuntimeError("reset requested after false reward/unlock signal during active gameplay")
        try:
            observation = self._reset_state_machine(
                reset_result,
                allow_active_gameplay_reset=allow_active_gameplay_reset,
                reset_reason=reset_reason,
            )
        except Exception as exc:
            if self.config.game_exe:
                reset_result = self.hard_reset()
                reset_result["fallbackReason"] = str(exc)
                observation = self._reset_state_machine(
                    reset_result,
                    allow_active_gameplay_reset=allow_active_gameplay_reset,
                    reset_reason=reset_reason,
                )
            else:
                try:
                    self.restore_game_speed()
                except Exception:
                    pass
                raise
        self.begin_new_attempt(observation, reason=f"reset:{reset_reason}")
        if self.config.debug_performance:
            reset_result["reset_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        return observation, {"reset": reset_result}

    def _reset_state_machine(
        self,
        reset_result: Dict[str, Any],
        allow_active_gameplay_reset: bool = False,
        reset_reason: str = "",
    ) -> Dict[str, Any]:
        started = time.monotonic()
        deadline = started + self.config.reset_wait_timeout
        stages: List[Dict[str, Any]] = reset_result.setdefault("stages", [])
        restart_attempts = 0
        last_restart_attempt_at = 0.0
        cleanup_attempts = 0
        playable_reset_attempts = 0
        seed_selection_failures = 0
        last_observation: Dict[str, Any] = {}
        requires_seed_selection = bool(
            reset_reason in {"win", "post_win_pending"}
            or reset_result.get("timeoutRequiresSeedFlow")
        )
        reset_state = ResetRuntimeState(
            generation_id=int(self._reset_generation_id),
            reason=str(reset_reason or ""),
            requires_seed_selection=requires_seed_selection,
            started_from_win=requires_seed_selection,
        )
        win_reset_invariant_armed = bool(reset_state.requires_seed_selection)

        def stage(name: str, **fields: Any) -> None:
            stages.append({"stage": name, "elapsed": round(time.monotonic() - started, 3), **fields})

        def set_phase(phase: str) -> None:
            reset_state.set_phase(phase)
            reset_result["resetPhase"] = reset_state.phase

        def sync_invariant_fields() -> None:
            self._reset_requires_seed_flow = bool(reset_state.requires_seed_selection)
            self._saw_seed_selection_this_reset = bool(reset_state.saw_seed_selection)
            self._clicked_lets_rock_this_reset = bool(reset_state.clicked_lets_rock)
            self._reset_reason = str(reset_reason or "")
            reset_result.update(reset_state.compatibility_fields())

        def log_reset_state(observation: Dict[str, Any], note: str = "") -> None:
            line = self._format_reset_state_line(
                observation,
                phase=reset_state.phase,
                require_seed_selection_this_reset=reset_state.requires_seed_selection,
                saw_seed_selection_this_reset=reset_state.saw_seed_selection,
            )
            if note:
                line += f" note={note}"
            print(line)

        def drive_startup_or_menu_once(observation: Dict[str, Any]) -> bool:
            try:
                state = self.adventure_screen_state()
            except Exception as exc:
                stage("menu_probe_failed", error=str(exc), boardFound=observation.get("boardFound"))
                return False
            screen_state = str(state.get("screenState") or "")
            reset_result["lastAdventureScreenState"] = screen_state
            if state.get("startupPopupVisible") or state.get("startupOkButtonVisible"):
                set_phase("startup_popup")
                click = self.click_startup_ok_once()
                stage(
                    "startup_popup_dismissed",
                    ok=click.get("ok"),
                    methodUsed=click.get("methodUsed"),
                    screenState=screen_state,
                )
                print(f"[reset] startup popup dismissed ok={click.get('ok')} method={click.get('methodUsed')}")
                return True
            if state.get("isAdventureButtonVisible"):
                set_phase("main_menu")
                click = self.press_adventure_once()
                stage(
                    "adventure_clicked",
                    ok=click.get("ok"),
                    methodUsed=click.get("methodUsed"),
                    screenState=screen_state,
                )
                print(f"[reset] adventure clicked ok={click.get('ok')} method={click.get('methodUsed')}")
                return True
            return False

        def enforce_loss_reset_invariant(observation: Dict[str, Any], note: str) -> None:
            if not reset_state.started_from_loss:
                return
            self._enforce_loss_reset_seed_selection_invariant(
                observation,
                require_seed_selection_this_reset=reset_state.requires_seed_selection,
                saw_seed_selection_this_reset=reset_state.saw_seed_selection,
                reset_phase=reset_state.phase,
                reset_result=reset_result,
                stage_callback=stage,
                state_logger=log_reset_state,
                note=note,
            )
            reset_state.unsafe_gameplay_ready_before_seed_count = int(
                reset_result.get("unsafeGameplayReadyBeforeSeedCount", 0) or 0
            )

        set_phase("idle")
        sync_invariant_fields()
        if reset_state.requires_seed_selection:
            set_phase("terminal_detected")
            sync_invariant_fields()
            stage(
                "win_reset_invariant_armed",
                resetReason=reset_reason,
                requireSeedSelectionThisReset=True,
                sawSeedSelectionThisReset=False,
            )

        while time.monotonic() < deadline:
            observation = self.observe(force_seed_probe=True, force_restart_probe=True)
            last_observation = observation
            lifecycle_state = self._classify_lifecycle_state(observation)
            next_step = observation.get("nextStep") or observation.get("next_step")
            terminal_hint = str(observation.get("terminalHint") or "")
            possible_win_reset_context = bool(
                terminal_hint == "possible_win"
                or self._is_confirmed_possible_win(observation)
                or self._post_win_signal_present(observation)
            )
            if (
                reset_reason == "timeout"
                and possible_win_reset_context
                and not reset_state.requires_seed_selection
            ):
                reset_state.requires_seed_selection = True
                reset_state.saw_seed_selection = False
                reset_state.started_from_win = True
                set_phase("terminal_detected")
                sync_invariant_fields()
                if not win_reset_invariant_armed:
                    win_reset_invariant_armed = True
                    stage(
                        "win_reset_invariant_armed",
                        resetReason=reset_reason,
                        terminalHint=terminal_hint,
                        gameplayReady=observation.get("gameplayReady"),
                        requireSeedSelectionThisReset=True,
                        sawSeedSelectionThisReset=False,
                    )
            sync_invariant_fields()
            log_reset_state(observation, note="loop")
            if not stages:
                stage(
                    "initial_state",
                    lifecycleState=lifecycle_state,
                    boardFound=observation.get("boardFound"),
                    onGameOverScreen=observation.get("onGameOverScreen"),
                    onLossScreen=observation.get("onLossScreen"),
                    onRestartScreen=observation.get("onRestartScreen"),
                    restartButtonActive=observation.get("restartButtonActive"),
                    restartDetectionReason=observation.get("restartDetectionReason"),
                    onSeedSelectionScreen=observation.get("onSeedSelectionScreen") or observation.get("seedSelectionActive"),
                    gameplayReady=observation.get("gameplayReady"),
                    done=observation.get("done"),
                    nextStep=next_step,
                )

            if drive_startup_or_menu_once(observation):
                sync_invariant_fields()
                time.sleep(max(0.25, self.config.reset_poll_seconds))
                continue

            enforce_loss_reset_invariant(observation, note="loop_head")

            if lifecycle_state == LIFECYCLE_LOSS_PENDING:
                if not reset_state.started_from_loss:
                    reset_state.requires_seed_selection = True
                    reset_state.saw_seed_selection = False
                    reset_state.started_from_loss = True
                    set_phase("terminal_detected")
                    sync_invariant_fields()
                    stage(
                        "loss_reset_invariant_armed",
                        resetReason=reset_reason,
                        lifecycleState=lifecycle_state,
                        requireSeedSelectionThisReset=True,
                        sawSeedSelectionThisReset=False,
                    )
                else:
                    set_phase("terminal_detected")
                    sync_invariant_fields()
                now = time.monotonic()
                if restart_attempts > 0 and now - last_restart_attempt_at < 1.0:
                    time.sleep(self.config.reset_poll_seconds)
                    continue
                if restart_attempts >= 3:
                    raise RuntimeError(f"failed to leave restart screen after {restart_attempts} restart attempts: {observation}")
                if is_restart_screen_observation(observation) and "restartDetectedAtSeconds" not in reset_result:
                    detected_at = round(now - started, 3)
                    reset_result["restartDetectedAtSeconds"] = detected_at
                    stage(
                        "game_over_detected",
                        lifecycleState=lifecycle_state,
                        onGameOverScreen=observation.get("onGameOverScreen"),
                        onRestartScreen=observation.get("onRestartScreen"),
                        restartButtonActive=observation.get("restartButtonActive"),
                        restartDetectionReason=observation.get("restartDetectionReason"),
                    )
                    print(f"[reset] restart_detected_at={detected_at:.2f}s")
                set_phase("click_restart")
                sync_invariant_fields()
                restart = self.auto_reset(
                    start_sun=self.config.start_sun,
                    allow_active_gameplay_reset=allow_active_gameplay_reset,
                    reset_reason=reset_reason,
                    require_loss_seed_selection_path=bool(
                        reset_state.requires_seed_selection
                    ),
                )
                restart_attempts += 1
                last_restart_attempt_at = now
                reset_result["restart"] = restart
                reset_result["methodUsed"] = restart.get("methodUsed", "auto_reset")
                reset_result["terminalDetected"] = restart.get("terminalDetected")
                reset_result["invokedUiRestart"] = restart.get("invokedUiRestart")
                reset_result["restartClickedAtSeconds"] = round(time.monotonic() - started, 3)
                reset_result["restartClickMethod"] = restart.get("restartClickMethod")
                reset_result["lossTerminalDetected"] = bool(restart.get("lossTerminalDetected"))
                reset_result["lossSeedSelectionRequired"] = bool(restart.get("lossSeedSelectionRequired"))
                stage(
                    "restart_clicked",
                    attempt=restart_attempts,
                    restartOk=restart.get("ok"),
                    terminalDetected=restart.get("terminalDetected"),
                    lossTerminalDetected=restart.get("lossTerminalDetected"),
                    lossSeedSelectionRequired=restart.get("lossSeedSelectionRequired"),
                    methodUsed=restart.get("methodUsed"),
                    invokedUiRestart=restart.get("invokedUiRestart"),
                    restartClicked=restart.get("restartClicked"),
                    restartClickMethod=restart.get("restartClickMethod"),
                    restartClickTargetName=restart.get("restartClickTargetName"),
                    restartClickTargetPath=restart.get("restartClickTargetPath"),
                    restartClickError=restart.get("restartClickError"),
                    actions=restart.get("actions", []),
                )
                print(
                    "[reset] restart_clicked_at="
                    f"{reset_result['restartClickedAtSeconds']:.2f}s method={restart.get('methodUsed')}"
                )
                set_phase("waiting_seed_selection")
                sync_invariant_fields()
                time.sleep(max(0.25, self.config.reset_poll_seconds))
                continue

            if next_step == "cleanup_reward_ui" or lifecycle_state in {LIFECYCLE_POST_WIN_PENDING, LIFECYCLE_RESETTING}:
                if lifecycle_state == LIFECYCLE_RESETTING and self._seed_selection_visible(observation):
                    pass
                elif lifecycle_state in {
                    LIFECYCLE_ACTIVE_GAMEPLAY,
                    LIFECYCLE_READY,
                    LIFECYCLE_POST_WIN_PENDING,
                    LIFECYCLE_RESETTING,
                    LIFECYCLE_UNKNOWN,
                }:
                    if cleanup_attempts >= 3 and lifecycle_state == LIFECYCLE_POST_WIN_PENDING:
                        if reset_state.requires_seed_selection:
                            stage(
                                "post_win_soft_reset_blocked",
                                lifecycleState=lifecycle_state,
                                resetReason=reset_reason,
                                terminalHint=observation.get("terminalHint"),
                                message="seed-selection-required reset blocked soft_reset fallback during post-win cleanup",
                            )
                            raise RuntimeError(
                                "Post-win reset requires UI-driven seed-selection path; soft_reset fallback is disabled."
                            )
                        if observation.get("boardFound"):
                            soft_observation, soft_info = self.soft_reset(
                                start_sun=self.config.start_sun,
                                run_init=False,
                                manual_clear=True,
                                allow_active_gameplay_reset=allow_active_gameplay_reset,
                                reset_reason=reset_reason,
                            )
                            cleanup_attempts = 0
                            stage(
                                "soft_reset_after_reward_ui",
                                lifecycleState=lifecycle_state,
                                gameplayReady=soft_observation.get("gameplayReady"),
                                nextStep=soft_observation.get("nextStep"),
                                reset=soft_info.get("reset", soft_info),
                            )
                            time.sleep(max(0.25, self.config.reset_poll_seconds))
                            continue
                        raise RuntimeError(f"failed to clear reward/trophy UI during reset: {observation}")
                    cleanup = self._cleanup_reward_ui_once(
                        observation,
                        reset_result=reset_result,
                        allow_active_gameplay_reset=allow_active_gameplay_reset,
                        reset_reason=reset_reason,
                    )
                    cleanup_attempts += 1
                    stage(
                        "cleanup_reward_ui",
                        attempt=cleanup_attempts,
                        lifecycleState=lifecycle_state,
                        cleaned=cleanup.get("cleaned"),
                        blocked=cleanup.get("blocked"),
                        blockedReason=cleanup.get("blockedReason"),
                        actions=cleanup.get("actions", []),
                    )
                    if cleanup.get("blocked") and lifecycle_state in {LIFECYCLE_ACTIVE_GAMEPLAY, LIFECYCLE_READY}:
                        if self._is_dirty_active_gameplay_board(observation):
                            raise RuntimeError(f"blocked reset cleanup during active gameplay: {observation}")
                        break
                    time.sleep(max(0.25, self.config.reset_poll_seconds))
                    continue

            if self._reward_or_trophy_ui_active(observation):
                if cleanup_attempts >= 3:
                    if reset_state.requires_seed_selection:
                        stage(
                            "post_win_soft_reset_blocked",
                            lifecycleState=lifecycle_state,
                            resetReason=reset_reason,
                            terminalHint=observation.get("terminalHint"),
                            message="seed-selection-required reset blocked soft_reset fallback during reward cleanup",
                        )
                        raise RuntimeError(
                            "Post-win reset requires UI-driven seed-selection path; soft_reset fallback is disabled."
                        )
                    if observation.get("boardFound"):
                        soft_observation, soft_info = self.soft_reset(
                            start_sun=self.config.start_sun,
                            run_init=False,
                            manual_clear=True,
                            allow_active_gameplay_reset=allow_active_gameplay_reset,
                            reset_reason=reset_reason,
                        )
                        cleanup_attempts = 0
                        stage(
                            "soft_reset_after_reward_ui",
                            gameplayReady=soft_observation.get("gameplayReady"),
                            nextStep=soft_observation.get("nextStep"),
                            reset=soft_info.get("reset", soft_info),
                        )
                        time.sleep(max(0.25, self.config.reset_poll_seconds))
                        continue
                    raise RuntimeError(f"failed to clear reward/trophy UI during reset: {observation}")
                cleanup = self._cleanup_reward_ui_once(
                    observation,
                    reset_result=reset_result,
                    allow_active_gameplay_reset=allow_active_gameplay_reset,
                    reset_reason=reset_reason,
                )
                cleanup_attempts += 1
                stage(
                    "cleanup_reward_ui",
                    attempt=cleanup_attempts,
                    lifecycleState=lifecycle_state,
                    cleaned=cleanup.get("cleaned"),
                    blocked=cleanup.get("blocked"),
                    blockedReason=cleanup.get("blockedReason"),
                    actions=cleanup.get("actions", []),
                )
                time.sleep(max(0.25, self.config.reset_poll_seconds))
                continue

            if lifecycle_state == LIFECYCLE_RESETTING and self._seed_selection_visible(observation):
                reset_state.saw_seed_selection = True
                set_phase("seed_selection")
                sync_invariant_fields()
                log_reset_state(observation, note="seed_selection_detected")
                reset_result.setdefault("seedScreenAtSeconds", round(time.monotonic() - started, 3))
                print(f"[reset] seed_screen_at={reset_result['seedScreenAtSeconds']:.2f}s")
                if reset_reason == "timeout":
                    print("[reset] seed screen observed after timeout reset")
                stage(
                    "seed_screen_detected",
                    lifecycleState=lifecycle_state,
                    selectedBankVisibleCount=observation.get("selectedBankVisibleCount"),
                    requireSeedSelectionThisReset=reset_state.requires_seed_selection,
                    sawSeedSelectionThisReset=reset_state.saw_seed_selection,
                )
                if not self.config.auto_select_seeds:
                    raise RuntimeError("reset reached seed selection but auto_select_seeds is disabled")
                stable_seed_screen, stable_seed_probe = self._wait_for_stable_seed_selection(
                    timeout=max(1.0, self.config.seed_click_delay * 3.0),
                    required_consecutive=2,
                )
                if not stable_seed_screen:
                    stage(
                        "seed_screen_waiting_for_stable_ui",
                        selectedBankVisibleCount=stable_seed_probe.get("selectedBankVisibleCount"),
                        boardStartMove=stable_seed_probe.get("boardStartMove"),
                        gameplayReady=stable_seed_probe.get("gameplayReady"),
                    )
                    time.sleep(max(0.25, self.config.reset_poll_seconds))
                    continue
                set_phase("selecting_seeds")
                sync_invariant_fields()
                stage("auto_select_started", seedList=list(self.config.seed_list))
                selection = self.auto_select_seeds(seed_list=self.config.seed_list, start_level=True)
                reset_result["autoSelectSeeds"] = selection
                reset_result["letsRockAtSeconds"] = round(time.monotonic() - started, 3)
                reset_state.clicked_lets_rock = bool(
                    selection.get("startInvoked")
                    or (isinstance(selection.get("startLog"), dict) and selection.get("startLog", {}).get("startInvoked"))
                    or (isinstance(selection.get("startLog"), dict) and selection.get("startLog", {}).get("startClicked"))
                )
                set_phase("clicking_lets_rock")
                sync_invariant_fields()
                stage(
                    "lets_rock_clicked",
                    ok=selection.get("ok"),
                    startInvoked=selection.get("startInvoked"),
                    clickedLetsRockThisReset=bool(reset_state.clicked_lets_rock),
                    startLog=selection.get("startLog", {}),
                )
                print(f"[reset] lets_rock_at={reset_result['letsRockAtSeconds']:.2f}s")
                if not selection.get("ok", False):
                    seed_selection_failures += 1
                    stage(
                        "auto_select_failed_waiting_for_seed_ui_retry",
                        attempt=seed_selection_failures,
                        message=selection.get("message"),
                    )
                    if seed_selection_failures <= 2:
                        retry_stable, retry_probe = self._wait_for_stable_seed_selection(
                            timeout=max(2.0, self.config.post_start_delay + self.config.seed_click_delay),
                            required_consecutive=2,
                        )
                        stage(
                            "auto_select_retry_seed_screen_probe",
                            attempt=seed_selection_failures,
                            stable=retry_stable,
                            seedSelectionActive=retry_probe.get("seedSelectionActive"),
                            boardStartMove=retry_probe.get("boardStartMove"),
                            gameplayReady=retry_probe.get("gameplayReady"),
                        )
                        if retry_stable:
                            time.sleep(max(0.25, self.config.reset_poll_seconds))
                            continue
                    raise RuntimeError(f"failed to auto-select seeds during reset: {selection}")
                set_phase("waiting_gameplay_ready")
                sync_invariant_fields()
                observation = self.wait_for_gameplay_ready(
                    timeout=max(1.0, deadline - time.monotonic()),
                    poll_seconds=self.config.reset_poll_seconds,
                    quiet=True,
                    fail_on_terminal=False,
                )
                enforce_loss_reset_invariant(observation, note="post_lets_rock_wait")
                set_phase("done")
                sync_invariant_fields()
                stage(
                    "gameplay_ready",
                    gameplayReady=observation.get("gameplayReady"),
                    seedSlotCount=observation.get("seedSlotCount"),
                    legalActions=len(observation.get("legalActions", [])),
                )
                reset_result["gameplayReadyAtSeconds"] = round(time.monotonic() - started, 3)
                print(f"[reset] gameplay_ready_at={reset_result['gameplayReadyAtSeconds']:.2f}s")
                if reset_state.requires_seed_selection and reset_state.saw_seed_selection:
                    seed_slot_count = self._safe_int(observation.get("seedSlotCount"), default=0)
                    print(
                        "[reset] accepted board after seed flow: "
                        f"wave={observation.get('wave', 0)} "
                        f"seed_slots={seed_slot_count} "
                        f"legalActionCount={observation.get('legalActionCount', len(observation.get('legalActions', [])))}"
                    )
                break

            if lifecycle_state in {LIFECYCLE_ACTIVE_GAMEPLAY, LIFECYCLE_READY}:
                enforce_loss_reset_invariant(observation, note="active_or_ready_branch")
                if reset_state.requires_seed_selection and not reset_state.saw_seed_selection:
                    set_phase("waiting_seed_selection")
                    sync_invariant_fields()
                    reset_state.unsafe_gameplay_ready_before_seed_count += 1
                    reset_result["unsafeGameplayReadyBeforeSeedCount"] = int(
                        reset_state.unsafe_gameplay_ready_before_seed_count
                    )
                    print(
                        "[reset] rejecting playable board: "
                        f"{reset_reason} reset requires seed selection but "
                        f"sawSeed={bool(reset_state.saw_seed_selection)} "
                        f"seedSelectionActive={observation.get('seedSelectionActive')} "
                        f"screenState={observation.get('screenState')} "
                        f"gameplayReady={observation.get('gameplayReady')}"
                    )
                    stage(
                        "seed_selection_required_blocked_gameplay_ready",
                        lifecycleState=lifecycle_state,
                        resetReason=reset_reason,
                        terminalHint=observation.get("terminalHint"),
                        gameplayReady=observation.get("gameplayReady"),
                        seedSelectionActive=observation.get("seedSelectionActive"),
                        requireSeedSelectionThisReset=True,
                        sawSeedSelectionThisReset=False,
                        unsafeGameplayReadyBeforeSeedCount=int(
                            reset_state.unsafe_gameplay_ready_before_seed_count
                        ),
                    )
                    if reset_state.unsafe_gameplay_ready_before_seed_count >= 3:
                        reset_result["seedSelectionImpossibleState"] = True
                        reset_result["seedSelectionImpossibleReason"] = (
                            "gameplay_ready_observed_repeatedly_before_seed_selection"
                        )
                        raise RuntimeError(
                            "seed-selection-required reset reached gameplay before seed selection repeatedly. "
                            f"reason={reset_reason} phase={reset_state.phase} "
                            f"screenState={observation.get('screenState')} "
                            f"nextStep={next_step} gameplayReady={observation.get('gameplayReady')} "
                            f"seedSelectionActive={observation.get('seedSelectionActive')} "
                            f"terminalHint={terminal_hint} "
                            f"requireSeed={bool(reset_state.requires_seed_selection)} "
                            f"sawSeed={bool(reset_state.saw_seed_selection)}"
                        )
                    time.sleep(max(0.25, self.config.reset_poll_seconds))
                    continue
                if reset_state.requires_seed_selection and not reset_state.clicked_lets_rock:
                    set_phase("waiting_gameplay_ready")
                    sync_invariant_fields()
                    stage(
                        "seed_selection_required_blocked_gameplay_ready_without_lets_rock",
                        lifecycleState=lifecycle_state,
                        resetReason=reset_reason,
                        gameplayReady=observation.get("gameplayReady"),
                        seedSelectionActive=observation.get("seedSelectionActive"),
                        requireSeedSelectionThisReset=True,
                        sawSeedSelectionThisReset=bool(reset_state.saw_seed_selection),
                        clickedLetsRockThisReset=False,
                    )
                    print(
                        "[reset] rejecting gameplay board: seed-selection-required reset has not clicked Let's Rock "
                        f"sawSeed={bool(reset_state.saw_seed_selection)} "
                        f"screenState={observation.get('screenState')} "
                        f"gameplayReady={observation.get('gameplayReady')}"
                    )
                    time.sleep(max(0.25, self.config.reset_poll_seconds))
                    continue
                if self._is_fresh_playable_reset_board(observation):
                    selected_seed_names_raw = observation.get("selectedSeedNames", [])
                    if isinstance(selected_seed_names_raw, list):
                        selected_seed_names = selected_seed_names_raw
                    elif selected_seed_names_raw:
                        selected_seed_names = [str(selected_seed_names_raw)]
                    else:
                        selected_seed_names = []
                    slots = seed_slots_from_observation(observation, self.config.plant_types)
                    slot_entries: List[Tuple[int, int]] = []
                    for index, slot in enumerate(slots):
                        slot_index = self._safe_int(slot.get("slotIndex"), default=index)
                        plant_type = self._safe_int(slot.get("plantType"), default=-1)
                        slot_entries.append((slot_index, plant_type))
                    slot_entries.sort(key=lambda item: item[0])
                    slot_plant_types = [plant_type for _, plant_type in slot_entries]
                    seed_slot_count = self._safe_int(observation.get("seedSlotCount"), default=len(slots))
                    legal_action_count = self._safe_int(observation.get("legalActionCount"), default=0)
                    if legal_action_count <= 0:
                        legal_actions = observation.get("legalActions", [])
                        if isinstance(legal_actions, list):
                            legal_action_count = len(legal_actions)
                    reset_result["reset_accept_reason"] = "fresh_playable_board"
                    reset_result["postResetSlotPlantTypes"] = slot_plant_types
                    reset_result["postResetSelectedSeedNames"] = list(selected_seed_names) if selected_seed_names else []
                    if selected_seed_names:
                        warning = (
                            "selectedSeedNames is diagnostic only unless slot-ordered; "
                            "validate seedSlots/slotPlantTypes for compatibility."
                        )
                        reset_warnings = reset_result.setdefault("resetWarnings", [])
                        if warning not in reset_warnings:
                            reset_warnings.append(warning)
                    stage(
                        "fresh_playable_board",
                        lifecycleState=lifecycle_state,
                        seedSlotCount=seed_slot_count,
                        selectedSeedNames=selected_seed_names,
                        slotPlantTypes=slot_plant_types,
                        legalActionCount=legal_action_count,
                    )
                    print(
                        "[reset] accepted fresh playable board: "
                        f"wave={observation.get('wave', 0)} "
                        f"plants={observation.get('plantCount', 0)} "
                        f"zombies={observation.get('zombieCount', 0)} "
                        f"bullets={observation.get('bulletCount', 0)} "
                        f"mowers={observation.get('logicalMowerCount', 0)} "
                        f"seed_slots={seed_slot_count} "
                        f"selectedSeedNames={selected_seed_names} "
                        f"slotPlantTypes={slot_plant_types} "
                        f"legalActionCount={legal_action_count}"
                    )
                if self._is_dirty_active_gameplay_board(observation):
                    if not allow_active_gameplay_reset:
                        reset_result["activeGameplayResetBlocked"] = True
                        stage(
                            "blocked_active_gameplay_reset",
                            lifecycleState=lifecycle_state,
                            resetReason=reset_reason,
                            plantCount=observation.get("plantCount"),
                            visiblePlantObjectCount=observation.get("visiblePlantObjectCount"),
                            zombieCount=observation.get("zombieCount"),
                            bulletCount=observation.get("bulletCount"),
                            wave=observation.get("wave"),
                            maxWave=observation.get("maxWave"),
                        )
                        raise RuntimeError(f"reset requested during active gameplay without explicit boundary: {observation}")
                    if playable_reset_attempts >= 3:
                        raise RuntimeError(f"failed to clear already-played board during reset: {observation}")
                    soft_observation, soft_info = self.soft_reset(
                        start_sun=self.config.start_sun,
                        run_init=False,
                        manual_clear=True,
                        allow_active_gameplay_reset=allow_active_gameplay_reset,
                        reset_reason=reset_reason,
                    )
                    playable_reset_attempts += 1
                    stage(
                        "soft_reset_played_board",
                        attempt=playable_reset_attempts,
                        lifecycleState=lifecycle_state,
                        plantCount=observation.get("plantCount"),
                        visiblePlantObjectCount=observation.get("visiblePlantObjectCount"),
                        zombieCount=observation.get("zombieCount"),
                        bulletCount=observation.get("bulletCount"),
                        gameplayReady=soft_observation.get("gameplayReady"),
                        nextStep=soft_observation.get("nextStep"),
                        reset=soft_info.get("reset", soft_info),
                    )
                    time.sleep(max(0.25, self.config.reset_poll_seconds))
                    continue
                set_phase("done")
                sync_invariant_fields()
                stage(
                    "gameplay_ready",
                    gameplayReady=True,
                    seedSlotCount=observation.get("seedSlotCount"),
                    legalActions=len(observation.get("legalActions", [])),
                )
                reset_result.setdefault("gameplayReadyAtSeconds", round(time.monotonic() - started, 3))
                break

            time.sleep(self.config.reset_poll_seconds)
        else:
            reset_result["lastLifecycleState"] = self._classify_lifecycle_state(last_observation) if last_observation else LIFECYCLE_UNKNOWN
            raise TimeoutError(
                "reset state machine timed out. "
                f"lastLifecycleState={reset_result['lastLifecycleState']} Last observation: {last_observation}"
            )

        if reset_state.requires_seed_selection and (
            not reset_state.saw_seed_selection
            or not reset_state.clicked_lets_rock
        ):
            sync_invariant_fields()
            stage(
                "seed_flow_reset_invariant_unmet",
                requireSeedSelectionThisReset=True,
                sawSeedSelectionThisReset=bool(reset_state.saw_seed_selection),
                clickedLetsRockThisReset=bool(reset_state.clicked_lets_rock),
                message="seed-selection-required reset did not observe seed selection and Let's Rock before completion",
            )
            raise RuntimeError("Reset invariant failed: seed selection and Let's Rock were not observed before reset completion.")

        cleanup_allow_active_gameplay_reset = bool(allow_active_gameplay_reset)
        cleanup = self.reset_cleanup(
            reset_card_cooldowns=True,
            allow_active_gameplay_reset=cleanup_allow_active_gameplay_reset,
            reset_reason=reset_reason,
        )
        reset_result["cleanup"] = cleanup
        observation, cleanup_ok, cleanup_message = self._wait_for_cleanup_valid(
            require_mowers=True,
            allow_active_gameplay_reset=cleanup_allow_active_gameplay_reset,
            reset_reason=reset_reason,
        )
        reset_result["cleanupValidation"] = cleanup_message
        reset_result["cleanupSuccess"] = cleanup_ok
        stage("cleanup_complete", cleanupSuccess=cleanup_ok, message=cleanup_message)
        if not cleanup_ok:
            raise RuntimeError(f"reset cleanup validation failed: {cleanup_message}")
        observation, playable_ok, playable_message = self._wait_for_post_reset_playable(
            allow_active_gameplay_reset=cleanup_allow_active_gameplay_reset,
            reset_reason=reset_reason,
            require_seed_selection_this_reset=reset_state.requires_seed_selection,
            saw_seed_selection_this_reset=reset_state.saw_seed_selection,
            clicked_lets_rock_this_reset=reset_state.clicked_lets_rock,
            require_mowers=True,
        )
        reset_result["postResetPlayableValidation"] = playable_message
        reset_result["postResetPlayableSuccess"] = playable_ok
        stage("post_reset_playable", playableSuccess=playable_ok, message=playable_message)
        if not playable_ok:
            raise RuntimeError(f"reset post-playable validation failed: {playable_message}")
        enforce_loss_reset_invariant(observation, note="post_reset_playable_validation")

        legal = self.legal_actions(observation)
        reset_result["timeToPlayableSeconds"] = round(time.monotonic() - started, 3)
        if "restartDetectedAtSeconds" in reset_result:
            print(f"[reset] total_reset_seconds={reset_result['timeToPlayableSeconds']:.2f}")
        reset_result["postResetSeedSelectionActive"] = bool(observation.get("seedSelectionActive"))
        invariant_ok = (not reset_state.requires_seed_selection) or (
            reset_state.saw_seed_selection and reset_state.clicked_lets_rock
        )
        reset_result["resetSuccess"] = (
            bool(observation.get("gameplayReady"))
            and not bool(observation.get("seedSelectionActive"))
            and not bool(observation.get("done"))
            and invariant_ok
        )
        reset_result["postResetLegalActionCount"] = len(legal)
        reset_result["postResetWaitOnly"] = legal == [0]
        reset_result["postResetWaitOnlyExpected"] = self._wait_only_expected(observation)
        set_phase("done")
        sync_invariant_fields()
        if legal == [0] and not reset_result["postResetWaitOnlyExpected"]:
            raise RuntimeError(f"reset produced unexpected wait-only legal_actions: {reset_result}")
        return observation

    def auto_reset(
        self,
        start_sun: Optional[int] = None,
        allow_active_gameplay_reset: bool = False,
        reset_reason: str = "",
        require_loss_seed_selection_path: bool = False,
        require_seed_selection_path: bool = False,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"mode": "auto", "run_init": False, "manual_clear": True}
        if allow_active_gameplay_reset:
            payload["allow_active_gameplay_reset"] = True
            payload["reset_reason"] = str(reset_reason or "")
        if require_loss_seed_selection_path:
            payload["require_loss_seed_selection_path"] = True
            payload["requireLossSeedSelectionPath"] = True
        if require_seed_selection_path:
            payload["require_seed_selection_path"] = True
            payload["requireSeedSelectionPath"] = True
        if start_sun is None:
            start_sun = self.config.start_sun
        if start_sun is not None:
            payload["start_sun"] = start_sun
        return self.client.request("auto_reset", **payload)

    def reset_cleanup(
        self,
        destroy_stale: bool = True,
        reset_card_cooldowns: bool = True,
        reset_counters: bool = False,
        allow_active_gameplay_reset: bool = False,
        reset_reason: str = "",
    ) -> Dict[str, Any]:
        return self.client.request(
            "reset_cleanup",
            destroy_stale=destroy_stale,
            refresh_boxes=True,
            reset_card_cooldowns=reset_card_cooldowns,
            reset_counters=reset_counters,
            allow_active_gameplay_reset=allow_active_gameplay_reset,
            reset_reason=str(reset_reason or ""),
        )

    def _cleanup_reward_ui_once(
        self,
        observation: Optional[Dict[str, Any]] = None,
        *,
        reset_result: Optional[Dict[str, Any]] = None,
        allow_active_gameplay_reset: bool = False,
        reset_reason: str = "",
    ) -> Dict[str, Any]:
        observation = observation or self.observe(force_seed_probe=True, force_restart_probe=True)
        lifecycle_state = self._classify_lifecycle_state(observation)
        result: Dict[str, Any] = {
            "ok": False,
            "cleaned": False,
            "blocked": False,
            "lifecycleState": lifecycle_state,
            "screenState": observation.get("screenState"),
            "nextStep": observation.get("nextStep"),
            "actions": [],
        }
        if reset_result is not None:
            reset_result.setdefault("cleanupLifecycleStates", []).append(lifecycle_state)
        if lifecycle_state in {LIFECYCLE_ACTIVE_GAMEPLAY, LIFECYCLE_READY}:
            result["blocked"] = True
            result["blockedReason"] = "active_gameplay"
            if reset_result is not None:
                reset_result["resetRewardUiCleanupBlockedCount"] = (
                    int(reset_result.get("resetRewardUiCleanupBlockedCount", 0) or 0) + 1
                )
                reset_result["blockedCleanupDuringGameplayCount"] = (
                    int(reset_result.get("blockedCleanupDuringGameplayCount", 0) or 0) + 1
                )
            print(self._format_safety_context("[safety] blocked cleanup_reward_ui during active gameplay", observation))
            return result
        if lifecycle_state not in LIFECYCLE_RESET_CLEANUP_ALLOWED:
            result["blocked"] = True
            result["blockedReason"] = "unknown_lifecycle"
            if reset_result is not None:
                reset_result["resetRewardUiCleanupBlockedCount"] = (
                    int(reset_result.get("resetRewardUiCleanupBlockedCount", 0) or 0) + 1
                )
                reset_result["suspiciousCleanupRewardUiCount"] = (
                    int(reset_result.get("suspiciousCleanupRewardUiCount", 0) or 0) + 1
                )
            print(self._format_safety_context("[safety] suspicious cleanup_reward_ui signal ignored", observation))
            return result
        cleanup = self.reset_cleanup(
            reset_card_cooldowns=True,
            allow_active_gameplay_reset=allow_active_gameplay_reset,
            reset_reason=reset_reason,
        )
        result.update(cleanup)
        result["ok"] = bool(cleanup.get("ok", True))
        result["cleaned"] = True
        result["actions"] = cleanup.get("actions", [])
        if reset_result is not None:
            reset_result["resetRewardUiCleanupCount"] = (
                int(reset_result.get("resetRewardUiCleanupCount", 0) or 0) + 1
            )
        return result

    def _reward_or_trophy_ui_active(self, observation: Dict[str, Any]) -> bool:
        """Compatibility wrapper for older reset paths."""
        state = self._classify_lifecycle_state(observation)
        return bool(
            state == LIFECYCLE_POST_WIN_PENDING
            or self._stale_cleanup_ui_after_reset(observation)
        )

    def _cleanup_signal_active(self, observation: Dict[str, Any]) -> bool:
        return cleanup_signal_active(observation)

    def _stale_cleanup_ui_after_reset(self, observation: Dict[str, Any]) -> bool:
        return stale_cleanup_ui(observation)

    def _confirmed_post_win_ui(self, observation: Dict[str, Any]) -> bool:
        return confirmed_post_win(observation)

    def _confirmed_loss_ui(self, observation: Dict[str, Any]) -> bool:
        return confirmed_loss(observation)

    def _seed_selection_visible(self, observation: Dict[str, Any]) -> bool:
        return seed_selection_visible(observation)

    def _confirmed_active_gameplay(self, observation: Dict[str, Any]) -> bool:
        return confirmed_active_gameplay(observation)

    def _has_live_board_progress(self, observation: Dict[str, Any]) -> bool:
        return live_board_progress(observation)

    def _timeout_reset_requires_full_seed_flow(self, observation: Dict[str, Any]) -> bool:
        if not isinstance(observation, dict):
            return False
        if self._seed_selection_visible(observation):
            return False
        if self._confirmed_post_win_ui(observation) or self._confirmed_loss_ui(observation):
            return False
        if self._has_live_board_progress(observation):
            return True
        screen_state = str(observation.get("screenState") or observation.get("screen_state") or "")
        return bool(
            observation.get("boardFound")
            and (observation.get("gameplayReady") or observation.get("actualGameplayReady"))
            and screen_state in {"", "gameplay"}
            and str(observation.get("terminalHint") or "") == "running"
            and not bool(observation.get("done"))
            and not bool(observation.get("over"))
        )

    def _post_win_signal_present(self, observation: Dict[str, Any]) -> bool:
        screen_state = str(observation.get("screenState") or observation.get("screen_state") or "")
        return bool(
            observation.get("trophyVisible")
            or observation.get("levelCompleteTrophyVisible")
            or observation.get("postWinClickRequired")
            or observation.get("rewardObjectVisible")
            or observation.get("rewardScreenVisible")
            or observation.get("unlockScreenVisible")
            or observation.get("newPlantUnlockedVisible")
            or observation.get("isRewardScreen")
            or observation.get("isNewPlantUnlockedScreen")
            or observation.get("levelCompleteScreenVisible")
            or screen_state in {"level_complete_trophy", "reward_unlock", "reward_screen"}
        )

    def _is_fresh_playable_reset_board(self, observation: Dict[str, Any]) -> bool:
        return fresh_playable_board(
            observation,
            fallback_rows=self.config.row_count,
        )

    def _is_dirty_active_gameplay_board(self, observation: Dict[str, Any]) -> bool:
        return dirty_active_board(
            observation,
            fallback_rows=self.config.row_count,
            start_sun=self.config.start_sun,
        )

    def _classify_lifecycle_state(self, observation: Dict[str, Any]) -> str:
        return legacy_lifecycle_state(
            observation,
            fallback_rows=self.config.row_count,
            start_sun=self.config.start_sun,
        )

    def _is_confirmed_post_game_ui(self, observation: Dict[str, Any]) -> bool:
        return self._confirmed_post_win_ui(observation) or self._confirmed_loss_ui(observation)

    def _is_confirmed_possible_win(self, observation: Dict[str, Any]) -> bool:
        try:
            wave = int(observation.get("wave", 0) or 0)
            max_wave = int(observation.get("maxWave", 0) or 0)
            zombie_count = int(observation.get("zombieCount", 0) or 0)
        except (TypeError, ValueError):
            return False
        return bool(
            max_wave > 0
            and wave >= max_wave
            and zombie_count == 0
            and not bool(observation.get("moreZombiesComing", False))
        )

    def _has_active_gameplay_progress(self, observation: Dict[str, Any]) -> bool:
        return active_gameplay_progress(observation)

    def _suspicious_cleanup_signal_during_gameplay(self, observation: Dict[str, Any]) -> bool:
        if not self._cleanup_signal_active(observation):
            return False
        if self._has_live_board_progress(observation):
            return True
        lifecycle_state = self._classify_lifecycle_state(observation)
        if lifecycle_state in LIFECYCLE_RESET_CLEANUP_ALLOWED:
            return False
        return bool(
            lifecycle_state in {LIFECYCLE_ACTIVE_GAMEPLAY, LIFECYCLE_READY}
            or (
                observation.get("boardFound")
                and not observation.get("over")
                and not self._seed_selection_visible(observation)
                and str(observation.get("terminalHint") or "") == "running"
                and self._has_active_gameplay_progress(observation)
            )
        )

    def _format_safety_context(self, prefix: str, observation: Dict[str, Any]) -> str:
        return (
            f"{prefix}: "
            f"step={observation.get('frameCount', '')} "
            f"wave={observation.get('wave', '')}/{observation.get('maxWave', '')} "
            f"zombies={observation.get('zombieCount', '')} "
            f"plants={observation.get('plantCount', '')} "
            f"gameplayReady={observation.get('gameplayReady')} "
            f"screenState={observation.get('screenState')} "
            f"nextStep={observation.get('nextStep')} "
            f"done={observation.get('done')} "
            f"over={observation.get('over')} "
            f"terminalHint={observation.get('terminalHint')}"
        )

    def _format_reset_state_line(
        self,
        observation: Dict[str, Any],
        *,
        phase: str,
        require_seed_selection_this_reset: bool,
        saw_seed_selection_this_reset: bool,
    ) -> str:
        return (
            f"[reset-state] phase={phase} "
            f"screenState={observation.get('screenState')} "
            f"nextStep={observation.get('nextStep') or observation.get('next_step')} "
            f"gameplayReady={observation.get('gameplayReady')} "
            f"seedSelectionActive={observation.get('seedSelectionActive')} "
            f"terminalHint={observation.get('terminalHint')} "
            f"done={observation.get('done')} "
            f"over={observation.get('over')} "
            f"requireSeed={bool(require_seed_selection_this_reset)} "
            f"sawSeed={bool(saw_seed_selection_this_reset)}"
        )

    def _loss_reset_gameplay_ready_without_seed_selection(
        self,
        observation: Dict[str, Any],
        *,
        require_seed_selection_this_reset: bool,
        saw_seed_selection_this_reset: bool,
    ) -> bool:
        return bool(
            require_seed_selection_this_reset
            and not saw_seed_selection_this_reset
            and observation.get("gameplayReady")
            and not observation.get("seedSelectionActive")
        )

    def _enforce_loss_reset_seed_selection_invariant(
        self,
        observation: Dict[str, Any],
        *,
        require_seed_selection_this_reset: bool,
        saw_seed_selection_this_reset: bool,
        reset_phase: str,
        reset_result: Optional[Dict[str, Any]] = None,
        stage_callback: Optional[Any] = None,
        state_logger: Optional[Any] = None,
        note: str = "",
    ) -> None:
        if not self._loss_reset_gameplay_ready_without_seed_selection(
            observation,
            require_seed_selection_this_reset=require_seed_selection_this_reset,
            saw_seed_selection_this_reset=saw_seed_selection_this_reset,
        ):
            return

        if reset_result is not None:
            reset_result["unsafeGameplayReadyBeforeSeedCount"] = int(
                reset_result.get("unsafeGameplayReadyBeforeSeedCount", 0) or 0
            ) + 1

        if callable(stage_callback):
            stage_callback(
                "unsafe_gameplay_ready_before_seed_selection",
                phase=reset_phase,
                screenState=observation.get("screenState"),
                nextStep=observation.get("nextStep") or observation.get("next_step"),
                gameplayReady=observation.get("gameplayReady"),
                seedSelectionActive=observation.get("seedSelectionActive"),
                requireSeedSelectionThisReset=bool(require_seed_selection_this_reset),
                sawSeedSelectionThisReset=bool(saw_seed_selection_this_reset),
                note=note,
            )
        if callable(state_logger):
            state_logger(observation, note=f"unsafe_gameplay_ready_before_seed_selection:{note}")
        else:
            print(
                self._format_reset_state_line(
                    observation,
                    phase=reset_phase,
                    require_seed_selection_this_reset=require_seed_selection_this_reset,
                    saw_seed_selection_this_reset=saw_seed_selection_this_reset,
                )
            )
        raise RuntimeError(
            "Unsafe loss reset: gameplayReady became true before seed selection was observed in this reset sequence."
        )

    def _request_reconnect_once(self, command: str, **payload: Any) -> Dict[str, Any]:
        try:
            return self.client.request(command, **payload)
        except (ConnectionError, OSError, socket.error):
            self.client.close()
            time.sleep(max(0.05, self.config.reset_poll_seconds))
            return self.client.request(command, **payload)

    def seed_probe(self) -> Dict[str, Any]:
        started = time.perf_counter()
        data = enrich_seed_probe(self._request_reconnect_once("seed_probe"))
        if self.config.debug_performance:
            data["seed_probe_ms"] = data.get("seed_probe_ms", round((time.perf_counter() - started) * 1000.0, 3))
        return data

    def _strict_seed_selection_active(self, probe: Dict[str, Any]) -> bool:
        return (
            bool(probe.get("seedSelectionActive"))
            and bool(probe.get("seedSelectionPanelActive"))
            and bool(probe.get("startButtonActive"))
            and not bool(probe.get("gameplayReady"))
            and not bool(probe.get("blockingRewardUiActive"))
        )

    def _wait_for_stable_seed_selection(
        self,
        timeout: float = 2.0,
        required_consecutive: int = 2,
    ) -> Tuple[bool, Dict[str, Any]]:
        deadline = time.monotonic() + max(0.0, timeout)
        poll_seconds = max(0.05, min(self.config.reset_poll_seconds, 0.25))
        last_probe: Dict[str, Any] = {}
        last_signature: Optional[Tuple[Any, ...]] = None
        consecutive = 0
        while time.monotonic() <= deadline:
            probe = self.seed_probe()
            last_probe = probe
            signature = (
                probe.get("selectedBankVisibleCount"),
                probe.get("availableSeedPacketCount"),
                tuple(probe.get("selectedBankPlantTypes", []) or []),
                bool(probe.get("seedSelectionPanelActive")),
                bool(probe.get("startButtonActive")),
            )
            if self._strict_seed_selection_active(probe):
                if signature == last_signature:
                    consecutive += 1
                else:
                    consecutive = 1
                    last_signature = signature
                if consecutive >= max(1, required_consecutive):
                    return True, probe
            else:
                consecutive = 0
                last_signature = signature
            time.sleep(poll_seconds)
        return False, last_probe

    def ui_probe(self, include_all: bool = False, max_entries: int = 350) -> Dict[str, Any]:
        started = time.perf_counter()
        data = self.client.request("ui_probe", include_all=include_all, max_entries=max_entries)
        if self.config.debug_performance:
            data["ui_scan_ms"] = data.get("ui_scan_ms", round((time.perf_counter() - started) * 1000.0, 3))
        return data

    def almanac_probe(self, include_all: bool = False) -> Dict[str, Any]:
        runtime = self.client.request("almanac_probe", plant_types=self.config.plant_types, include_all=include_all)
        return enrich_almanac_probe(runtime)

    def auto_select_seeds(
        self,
        seed_list: Optional[List[str]] = None,
        start_level: bool = True,
    ) -> Dict[str, Any]:
        names = seed_list or self.config.seed_list
        requested_sequence = resolve_seed_list(names)
        requested_counts = count_values(requested_sequence)
        expected_total = len(requested_sequence)
        actions: List[str] = []
        selection_attempts: List[Dict[str, Any]] = []
        selection_steps: List[Dict[str, Any]] = []
        start_log: Dict[str, Any] = {
            "startRequested": bool(start_level),
            "startClicked": False,
            "methodUsed": "",
            "letsRockDelay": self.config.lets_rock_delay,
            "postStartDelay": self.config.post_start_delay,
        }

        before = self.seed_probe()
        active_counts, _ = active_gameplay_bank_state(before)
        if before.get("gameplayReady") and counts_cover(active_counts, requested_counts):
            selected_types = [int(card.get("plantType", -999)) for card in before.get("activeGameplayCardBankCards", [])]
            verification = {
                "success": True,
                "source": "activeGameplayCardBankCards",
                "requestedSeedTypes": requested_sequence,
                "selectedSeedTypes": selected_types,
                "missingSeedTypes": [],
                "requestedPlantTypeCounts": counts_to_entries(requested_counts),
                "selectedPlantTypeCounts": counts_to_entries(active_counts),
            }
            return {
                "ok": True,
                "alreadyGameplayReady": True,
                "requestedSeedTypes": requested_sequence,
                "selectedSeedTypes": selected_types,
                "missingSeedTypes": [],
                "verification": verification,
                "gameplayReadyBeforeStart": True,
                "startRequested": bool(start_level),
                "startInvoked": False,
                "actions": actions,
                "selectionAttempts": selection_attempts,
                "selectionSteps": selection_steps,
                "startLog": start_log,
                "before": before,
                "afterSelectionBeforeStart": before,
                "afterStart": before,
                "after": before,
                "finalObservation": {},
                "message": "Requested seeds were already present in the active gameplay card bank.",
            }

        selected_start, selected_start_count = selected_bank_state(before)
        selected_start_types = [int(card.get("plantType", -999)) for card in before.get("selectedSeedBankCards", [])]
        if not before.get("seedSelectionActive"):
            return self._seed_selection_response(
                ok=False,
                message="Seed selection UI is not active and requested seeds are not already in the active gameplay bank.",
                requested_sequence=requested_sequence,
                requested_counts=requested_counts,
                selected_counts=selected_start,
                selected_types=[],
                missing=requested_sequence,
                before=before,
                after_selection=before,
                after_start=before,
                actions=actions,
                selection_attempts=selection_attempts,
                selection_steps=selection_steps,
                start_log=start_log,
                start_invoked=False,
                gameplay_ready_before_start=bool(before.get("gameplayReady")),
                final_observation={},
            )
        stable, stable_probe = self._wait_for_stable_seed_selection(
            # The chooser animates in after Adventure navigation.  One second
            # is too short on a normal live launch even though the same screen
            # becomes safely interactive a moment later.
            timeout=max(3.0, self.config.seed_click_delay * 3.0),
            required_consecutive=2,
        )
        if not stable:
            return self._seed_selection_response(
                ok=False,
                message="Seed selection UI is active but not stable/safe for automation.",
                requested_sequence=requested_sequence,
                requested_counts=requested_counts,
                selected_counts=selected_bank_state(before)[0],
                selected_types=[int(card.get("plantType", -999)) for card in before.get("selectedSeedBankCards", [])],
                missing=requested_sequence,
                before=before,
                after_selection=stable_probe or before,
                after_start=stable_probe or before,
                actions=actions + ["seed_selection_not_stable_for_clicks"],
                selection_attempts=selection_attempts,
                selection_steps=selection_steps,
                start_log=start_log,
                start_invoked=False,
                gameplay_ready_before_start=bool(before.get("gameplayReady")),
                final_observation={},
            )
        before = stable_probe
        selected_start, selected_start_count = selected_bank_state(before)
        selected_start_types = [int(card.get("plantType", -999)) for card in before.get("selectedSeedBankCards", [])]
        unexpected_selected = [
            int(plant_type)
            for plant_type, count in selected_start.items()
            if count > requested_counts.get(plant_type, 0)
        ]
        if selected_start_count > expected_total or unexpected_selected:
            return self._seed_selection_response(
                ok=False,
                message="Selected seed bank contains cards outside the requested duplicate-aware loadout.",
                requested_sequence=requested_sequence,
                requested_counts=requested_counts,
                selected_counts=selected_start,
                selected_types=selected_start_types,
                missing=requested_sequence,
                before=before,
                after_selection=before,
                after_start=before,
                actions=actions,
                selection_attempts=selection_attempts,
                selection_steps=selection_steps,
                start_log=start_log,
                start_invoked=False,
                gameplay_ready_before_start=bool(before.get("gameplayReady")),
                final_observation={},
            )

        after_selection = before
        selection_ok = True
        selected_so_far: Counter = Counter(selected_start)
        requested_to_click = missing_values(requested_sequence, selected_start_types)
        preselected_cards = list(before.get("stalePreselectedCards", []) or [])
        preselected_counts = count_cards(preselected_cards)
        preselected_types = [int(card.get("plantType", -999)) for card in preselected_cards if isinstance(card, dict)]
        use_preselected_loadout = (
            not selected_bank_state(before)[1]
            and bool(before.get("seedSelectionActive"))
            and bool(before.get("startButtonActive"))
            and counts_cover(preselected_counts, requested_counts)
        )
        initial_settle_delay = max(0.0, self.config.seed_click_delay)
        if initial_settle_delay:
            actions.append(f"initial_seed_selection_settle_delay:{initial_settle_delay}")
            time.sleep(initial_settle_delay)
        if use_preselected_loadout:
            actions.append("stale_preselected_loadout_matches_requested")
            after_selection = before
        else:
            if selected_start_count:
                actions.append(f"selected_bank_already_contains:{counts_to_entries(selected_start)}")
            for step_index, plant_type in enumerate(requested_to_click):
                step_attempts: List[Dict[str, Any]] = []
                step_ok = False
                duplicate_index = selected_so_far.get(plant_type, 0)
                for attempt_number in range(1, 4):
                    stable_before_click, probe_before = self._wait_for_stable_seed_selection(
                        timeout=max(0.75, self.config.seed_click_delay * 2.0),
                        required_consecutive=1,
                    )
                    if not stable_before_click:
                        attempt_log = {
                            "stepIndex": step_index,
                            "attemptNumber": attempt_number,
                            "attemptIndex": len(selection_attempts),
                            "plantType": int(plant_type),
                            "plantTypeName": plant_type_name(int(plant_type)),
                            "methodUsed": "strict_seed_screen_gate",
                            "candidateInstanceId": 0,
                            "candidateHierarchyPath": None,
                            "selectedBankCountBefore": selected_bank_state(probe_before)[1],
                            "selectedBankCountAfter": selected_bank_state(probe_before)[1],
                            "selectedBankTypeCountBefore": selected_bank_state(probe_before)[0].get(plant_type, 0),
                            "selectedBankTypeCountAfter": selected_bank_state(probe_before)[0].get(plant_type, 0),
                            "selectedBankCountIncreased": False,
                            "selectedBankTypeCountIncreased": False,
                            "selectedBankMultisetAfter": counts_to_entries(selected_bank_state(probe_before)[0]),
                            "selectedBankMultisetAfterText": format_counts(selected_bank_state(probe_before)[0]),
                            "delayUsed": self.config.seed_click_delay,
                            "clickInvoked": False,
                            "duplicateSelectionIndex": duplicate_index,
                            "duplicateCostIncreaseAccessible": False,
                            "duplicateCostIncreaseDetected": False,
                            "visibleCostsBefore": [],
                            "visibleCostsAfter": [],
                            "success": False,
                            "error": "Seed selection UI was not stable/safe immediately before seed click.",
                        }
                        selection_attempts.append(attempt_log)
                        step_attempts.append(attempt_log)
                        after_selection = probe_before
                        break
                    before_counts, before_count = selected_bank_state(probe_before)
                    before_type_count = before_counts.get(plant_type, 0)
                    click = self.select_seed_card_once(
                        plant_type=plant_type,
                        attempt_index=len(selection_attempts),
                        duplicate_selection_index=duplicate_index,
                    )
                    actions.extend(str(action) for action in click.get("actions", []))
                    time.sleep(max(0.0, self.config.seed_click_delay))
                    probe_after = self.seed_probe()
                    after_counts, after_count = selected_bank_state(probe_after)
                    after_type_count = after_counts.get(plant_type, 0)
                    bridge_attempt = dict(click.get("attempt", {}))
                    click_invoked = bool(click.get("clickInvoked") or bridge_attempt.get("clickInvoked"))
                    count_increased = after_count == before_count + 1
                    type_count_increased = after_type_count == before_type_count + 1
                    attempt_ok = click_invoked and count_increased and type_count_increased
                    attempt_log = {
                        "stepIndex": step_index,
                        "attemptNumber": attempt_number,
                        "attemptIndex": len(selection_attempts),
                        "plantType": int(plant_type),
                        "plantTypeName": plant_type_name(int(plant_type)),
                        "methodUsed": bridge_attempt.get("methodUsed", "CardUI.OnMouseDown()"),
                        "candidateInstanceId": bridge_attempt.get("candidateInstanceId", 0),
                        "candidateHierarchyPath": bridge_attempt.get("candidateHierarchyPath"),
                        "selectedBankCountBefore": before_count,
                        "selectedBankCountAfter": after_count,
                        "selectedBankTypeCountBefore": before_type_count,
                        "selectedBankTypeCountAfter": after_type_count,
                        "selectedBankCountIncreased": count_increased,
                        "selectedBankTypeCountIncreased": type_count_increased,
                        "selectedBankMultisetAfter": counts_to_entries(after_counts),
                        "selectedBankMultisetAfterText": format_counts(after_counts),
                        "delayUsed": self.config.seed_click_delay,
                        "clickInvoked": click_invoked,
                        "duplicateSelectionIndex": duplicate_index,
                        "duplicateCostIncreaseAccessible": bool(bridge_attempt.get("duplicateCostIncreaseAccessible")),
                        "duplicateCostIncreaseDetected": bool(bridge_attempt.get("duplicateCostIncreaseDetected")),
                        "visibleCostsBefore": bridge_attempt.get("visibleCostsBefore", []),
                        "visibleCostsAfter": bridge_attempt.get("visibleCostsAfter", []),
                        "success": attempt_ok,
                        "error": "" if attempt_ok else bridge_attempt.get("error") or "Click did not increase the visible selected bank by exactly one after delay.",
                    }
                    selection_attempts.append(attempt_log)
                    step_attempts.append(attempt_log)
                    after_selection = probe_after
                    if attempt_ok:
                        step_ok = True
                        selected_so_far[plant_type] += 1
                        break
                    if attempt_number < 3:
                        time.sleep(max(0.0, self.config.seed_click_delay))

                selection_steps.append(
                    {
                        "stepIndex": step_index,
                        "plantType": int(plant_type),
                        "plantTypeName": plant_type_name(int(plant_type)),
                        "duplicateSelectionIndex": duplicate_index,
                        "attempts": step_attempts,
                        "success": step_ok,
                        "selectedBankCountAfter": step_attempts[-1]["selectedBankCountAfter"] if step_attempts else None,
                        "selectedBankMultisetAfter": step_attempts[-1]["selectedBankMultisetAfter"] if step_attempts else [],
                        "selectedBankMultisetAfterText": step_attempts[-1]["selectedBankMultisetAfterText"] if step_attempts else "{}",
                    }
                )
                if not step_ok:
                    selection_ok = False
                    break

        if use_preselected_loadout:
            selected_counts = preselected_counts
            selected_count = sum(preselected_counts.values())
            selected_types = preselected_types
        else:
            selected_counts, selected_count = selected_bank_state(after_selection)
            selected_types = [int(card.get("plantType", -999)) for card in after_selection.get("selectedSeedBankCards", [])]
        missing = missing_values(requested_sequence, selected_types)
        verified = selected_count == expected_total and selected_counts == requested_counts and not missing
        gameplay_ready_before_start = bool(after_selection.get("gameplayReady"))
        pre_start_ok = (
            selection_ok
            and verified
            and bool(after_selection.get("seedSelectionActive"))
            and bool(after_selection.get("startButtonActive"))
        )
        start_invoked = False
        after_start = after_selection
        immediate_after_start: Dict[str, Any] = {}
        post_transition_probe: Dict[str, Any] = {}
        final_observation: Dict[str, Any] = {}
        final_gameplay_ready = False
        final_seed_selection_active = bool(after_selection.get("seedSelectionActive"))
        final_active_counts: Counter = Counter()
        gameplay_slot_verification: Dict[str, Any] = {
            "success": False,
            "source": "finalObservation.seedSlots",
            "gameplayReady": False,
            "requestedSeedTypes": list(requested_sequence),
            "activeSeedTypes": [],
            "activeSeedSlotCount": 0,
            "reason": "start_not_invoked",
        }
        transition_status = "start_not_invoked"

        if start_level and pre_start_ok:
            start_log["preStartSelectedBankCount"] = selected_count
            start_log["preStartSelectedBankMultiset"] = counts_to_entries(selected_counts)
            start_log["preStartSelectedBankMultisetText"] = format_counts(selected_counts)
            time.sleep(max(0.0, self.config.lets_rock_delay))
            stable_before_start, stable_start_probe = self._wait_for_stable_seed_selection(
                timeout=max(0.75, self.config.seed_click_delay * 2.0),
                required_consecutive=1,
            )
            if not stable_before_start:
                start_log["error"] = "Start not pressed because seed selection UI was no longer stable/safe."
                after_selection = stable_start_probe or after_selection
                pre_start_ok = False
            else:
                after_selection = stable_start_probe
                selected_counts, selected_count = selected_bank_state(after_selection)
                selected_types = [
                    int(card.get("plantType", -999)) for card in after_selection.get("selectedSeedBankCards", [])
                ]
                missing = missing_values(requested_sequence, selected_types)
                verified = selected_count == expected_total and selected_counts == requested_counts and not missing
                pre_start_ok = (
                    verified
                    and bool(after_selection.get("seedSelectionActive"))
                    and bool(after_selection.get("startButtonActive"))
                )

        if start_level and pre_start_ok:
            start_response = self.press_lets_rock_once()
            start_invoked = bool(start_response.get("startClicked") or start_response.get("ok"))
            start_log.update(
                {
                    "startClicked": start_invoked,
                    "methodUsed": start_response.get("methodUsed", ""),
                    "startHierarchyPath": start_response.get("startHierarchyPath"),
                    "actions": start_response.get("actions", []),
                }
            )
            actions.extend(str(action) for action in start_response.get("actions", []))
            time.sleep(max(0.0, self.config.post_start_delay))
            immediate_after_start = self.seed_probe()
            after_start = immediate_after_start
            transition_status = "waiting_for_gameplay"
            try:
                final_observation = self.wait_for_gameplay_ready(
                    timeout=self.config.reset_wait_timeout,
                    poll_seconds=self.config.reset_poll_seconds,
                    quiet=True,
                    fail_on_terminal=False,
                )
            except Exception as exc:
                start_log["gameplayReadyError"] = str(exc)
                final_observation = {}
            gameplay_slot_verification = gameplay_slot_identity_verification(
                final_observation,
                requested_sequence,
            )
            final_gameplay_ready = bool(gameplay_slot_verification.get("success"))
            if final_gameplay_ready:
                # The accepted observation is one coherent bridge frame: raw
                # gameplay readiness, seed-screen inactivity, an active CardUI
                # bank, and ordered runtime seed slots all agreed.  A separate
                # UI probe may briefly retain chooser objects after Let's Rock;
                # keep it as diagnostics but never let it veto this stronger
                # canonical frame.
                transition_status = "gameplay_confirmed"
                post_transition_probe = self.seed_probe()
            else:
                # A start click that did not reach canonical gameplay may only
                # be retried after the chooser is proven interactable and
                # stable across multiple post-timeout frames.  One stale flag
                # is not sufficient.
                stable_seed_screen, stable_probe = self._wait_for_stable_seed_selection(
                    timeout=max(0.75, self.config.seed_click_delay * 2.0),
                    required_consecutive=2,
                )
                post_transition_probe = stable_probe or self.seed_probe()
                transition_status = (
                    "seed_selection_still_interactive"
                    if stable_seed_screen
                    else "gameplay_transition_timeout"
                )
            after_start = post_transition_probe or immediate_after_start
            final_active_counts, _ = active_gameplay_bank_state(after_start)
            final_seed_selection_active = bool(after_start.get("seedSelectionActive"))
            if not final_gameplay_ready and transition_status == "seed_selection_still_interactive":
                actions.append("bridge_auto_select_fallback_skipped_after_start_click")
                start_log["fallbackStart"] = {
                    "ok": False,
                    "startInvoked": False,
                    "message": "Skipped bridge auto-select/start fallback because seed UI was still active after Start; forcing InitBoard here can corrupt UI state.",
                    "actions": [],
                }
            start_log["gameplayReady"] = final_gameplay_ready
            start_log["seedSelectionActiveAfterStart"] = final_seed_selection_active
            start_log["transitionStatus"] = transition_status
            start_log["gameplaySlotVerification"] = dict(gameplay_slot_verification)
            start_log["activeGameplayBankMultiset"] = counts_to_entries(final_active_counts)
            start_log["activeGameplayBankMultisetText"] = format_counts(final_active_counts)
        elif start_level:
            start_log["error"] = (
                "Start not pressed because pre-start UI verification failed "
                f"(selectionOk={selection_ok}, verified={verified}, seedSelectionActive={after_selection.get('seedSelectionActive')}, "
                f"startButtonActive={after_selection.get('startButtonActive')})."
            )

        ok = (
            selection_ok
            and verified
            and (
                not start_level
                or (
                    start_invoked
                    and final_gameplay_ready
                )
            )
        )
        message = "Requested seeds selected with paced UI verification."
        if start_level:
            if ok:
                message = "Requested seeds selected and gameplayReady verified."
            else:
                start_failure = start_log.get("error") or start_log.get("gameplayReadyError") or "none"
                message = (
                    "Paced seed selection/start failed; do not start training. "
                    f"selection_ok={selection_ok} verified={verified} "
                    f"start_invoked={start_invoked} gameplay_ready={final_gameplay_ready} "
                    f"seed_selection_active={final_seed_selection_active} "
                    f"active_bank={format_counts(final_active_counts)} "
                    f"transition_status={transition_status} "
                    f"slot_identity={gameplay_slot_verification.get('reason', 'unknown')} "
                    f"requested={format_counts(requested_counts)} "
                    f"start_error={start_failure}"
                )

        return self._seed_selection_response(
            ok=ok,
            message=message,
            requested_sequence=requested_sequence,
            requested_counts=requested_counts,
            selected_counts=selected_counts,
            selected_types=selected_types,
            missing=missing,
            before=before,
            after_selection=after_selection,
            after_start=after_start,
            actions=actions,
            selection_attempts=selection_attempts,
            selection_steps=selection_steps,
            start_log=start_log,
            start_invoked=start_invoked,
            gameplay_ready_before_start=gameplay_ready_before_start,
            final_observation=final_observation,
            transition_status=transition_status,
            gameplay_slot_verification=gameplay_slot_verification,
            immediate_after_start=immediate_after_start,
            post_transition_probe=post_transition_probe,
        )

    def select_seed_card_once(
        self,
        plant_type: int,
        attempt_index: int = 0,
        duplicate_selection_index: int = 0,
    ) -> Dict[str, Any]:
        return self._request_reconnect_once(
            "select_seed_card_once",
            plant_type=int(plant_type),
            attempt_index=int(attempt_index),
            duplicate_selection_index=int(duplicate_selection_index),
        )

    def press_lets_rock_once(self) -> Dict[str, Any]:
        return self._request_reconnect_once("press_lets_rock_once")

    def _seed_selection_response(
        self,
        ok: bool,
        message: str,
        requested_sequence: List[int],
        requested_counts: Counter,
        selected_counts: Counter,
        selected_types: List[int],
        missing: List[int],
        before: Dict[str, Any],
        after_selection: Dict[str, Any],
        after_start: Dict[str, Any],
        actions: List[str],
        selection_attempts: List[Dict[str, Any]],
        selection_steps: List[Dict[str, Any]],
        start_log: Dict[str, Any],
        start_invoked: bool,
        gameplay_ready_before_start: bool,
        final_observation: Dict[str, Any],
        transition_status: str = "",
        gameplay_slot_verification: Optional[Dict[str, Any]] = None,
        immediate_after_start: Optional[Dict[str, Any]] = None,
        post_transition_probe: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        verification = {
            "success": not missing and selected_counts == requested_counts,
            "source": "selectedSeedBankCards",
            "requestedSeedTypes": requested_sequence,
            "selectedSeedTypes": selected_types,
            "missingSeedTypes": missing,
            "requestedPlantTypeCounts": counts_to_entries(requested_counts),
            "selectedPlantTypeCounts": counts_to_entries(selected_counts),
        }
        return {
            "ok": bool(ok),
            "requestedSeedTypes": requested_sequence,
            "selectedSeedTypes": selected_types,
            "missingSeedTypes": missing,
            "verification": verification,
            "gameplayReadyBeforeStart": gameplay_ready_before_start,
            "startRequested": bool(start_log.get("startRequested")),
            "startInvoked": bool(start_invoked),
            "actions": actions,
            "selectionAttempts": selection_attempts,
            "selectionSteps": selection_steps,
            "startLog": start_log,
            "before": before,
            "afterSelectionBeforeStart": after_selection,
            "afterStart": after_start,
            "after": after_start,
            "immediateAfterStart": dict(immediate_after_start or {}),
            "postTransitionProbe": dict(post_transition_probe or {}),
            "finalObservation": final_observation,
            "transitionStatus": str(transition_status or ""),
            "gameplaySlotVerification": dict(gameplay_slot_verification or {}),
            "message": message,
        }

    def soft_reset(
        self,
        start_sun: Optional[int] = None,
        run_init: bool = False,
        manual_clear: bool = True,
        allow_active_gameplay_reset: bool = False,
        reset_reason: str = "",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        payload: Dict[str, Any] = {"run_init": run_init, "manual_clear": manual_clear}
        if allow_active_gameplay_reset:
            payload["allow_active_gameplay_reset"] = True
            payload["reset_reason"] = str(reset_reason or "")
        if start_sun is None:
            start_sun = self.config.start_sun
        if start_sun is not None:
            payload["start_sun"] = start_sun
        reset_result = self.client.request("soft_reset", **payload)
        reset_result["cleanup"] = self.reset_cleanup(
            reset_card_cooldowns=True,
            allow_active_gameplay_reset=allow_active_gameplay_reset,
            reset_reason=reset_reason,
        )
        time.sleep(max(0.05, self.config.reset_poll_seconds))
        observation = self.observe()
        self.begin_new_attempt(observation, reason=f"soft_reset:{reset_reason or 'manual'}")
        return observation, {"reset": reset_result}

    def observe(
        self,
        debug_observation: Optional[bool] = None,
        force_seed_probe: bool = False,
        force_restart_probe: bool = False,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        include_debug = self.config.debug_observation if debug_observation is None else bool(debug_observation)
        observation = self.client.request(
            "observe",
            debug_observation=include_debug,
            force_seed_probe=force_seed_probe,
            force_restart_probe=force_restart_probe,
        )
        if self.config.debug_performance:
            observation["observe_ms"] = observation.get("observe_ms", round((time.perf_counter() - started) * 1000.0, 3))
        return observation

    def screen_state_fast(self) -> Dict[str, Any]:
        started = time.perf_counter()
        state = self.client.request("screen_state_fast")
        if self.config.debug_performance:
            state["screen_state_fast_ms"] = state.get("screen_state_fast_ms", round((time.perf_counter() - started) * 1000.0, 3))
        return state

    def adventure_screen_state(self) -> Dict[str, Any]:
        started = time.perf_counter()
        state = self.client.request("adventure_screen_state")
        if self.config.debug_performance:
            state["adventure_screen_state_ms"] = state.get(
                "adventure_screen_state_ms",
                round((time.perf_counter() - started) * 1000.0, 3),
            )
        return state

    def press_adventure_once(self) -> Dict[str, Any]:
        return self._request_reconnect_once("press_adventure_once")

    def click_startup_ok_once(self) -> Dict[str, Any]:
        return self._request_reconnect_once("click_startup_ok_once")

    def dismiss_startup_popup_once(self) -> Dict[str, Any]:
        return self.click_startup_ok_once()

    def wait_for_startup_popup_dismissed(self, timeout: float = 5.0, poll_seconds: Optional[float] = None) -> Dict[str, Any]:
        poll = self.config.reset_poll_seconds if poll_seconds is None else float(poll_seconds)
        deadline = time.monotonic() + max(0.1, timeout)
        result = _poll_dict_until(
            deadline=deadline,
            probe=self.adventure_screen_state,
            accept=lambda state: not (
                state.get("startupPopupVisible")
                or state.get("startupOkButtonVisible")
            ),
            poll_seconds=max(0.05, poll),
        )
        return result.last_value

    def click_reward_continue_once(self) -> Dict[str, Any]:
        return self._request_reconnect_once("click_reward_continue_once")

    def click_trophy_once(self) -> Dict[str, Any]:
        return self._request_reconnect_once("click_trophy_once")

    def click_level_complete_reward_once(self) -> Dict[str, Any]:
        return self.click_trophy_once()

    def click_try_again_once(self) -> Dict[str, Any]:
        return self._request_reconnect_once("click_try_again_once")

    def wait_for_board(self, timeout: float = 180.0, poll_seconds: float = 0.2, quiet: bool = False) -> Dict[str, Any]:
        started = time.monotonic()
        deadline = time.monotonic() + timeout
        if not quiet:
            print(
                "Waiting for board. In the game: click the green OK on the "
                "startup popup, click Adventure, select plants, then enter/start the board."
            )
        result = _poll_dict_until(
            deadline=deadline,
            probe=self.observe,
            accept=lambda observation: bool(observation.get("boardFound")),
            poll_seconds=poll_seconds,
            retry_exceptions=(ConnectionError, OSError, RuntimeError),
        )
        if result.matched:
            if not quiet:
                elapsed = time.monotonic() - started
                print(f"Board detected after {elapsed:.2f} seconds.")
            return result.last_value
        raise TimeoutError(
            "Timed out waiting for boardFound=true. Manual path: click the green OK on the "
            "startup popup, click Adventure, select plants, then enter/start the board. "
            f"Last observation: {result.last_value}"
        )

    def wait_for_gameplay_ready(
        self,
        timeout: float = 60.0,
        poll_seconds: float = 0.2,
        quiet: bool = False,
        fail_on_terminal: bool = True,
    ) -> Dict[str, Any]:
        started = time.monotonic()
        deadline = time.monotonic() + timeout
        if not quiet:
            print("Waiting for gameplayReady=true with selected seed packets available.")

        def reject_terminal(observation: Dict[str, Any]) -> None:
            if fail_on_terminal and observation.get("boardFound") and observation.get("done"):
                hint = observation.get("terminalHint", "unknown")
                wave = observation.get("wave", "?")
                max_wave = observation.get("maxWave", "?")
                raise RuntimeError(
                    "Board is present, but it is already in a terminal/end-screen state "
                    f"(terminalHint={hint}, wave={wave}/{max_wave}). Start or reset a playable episode before waiting for gameplayReady."
                )

        result = _poll_dict_until(
            deadline=deadline,
            probe=self.observe,
            accept=lambda observation: bool(
                observation.get("gameplayReady")
                and not observation.get("seedSelectionActive")
            ),
            poll_seconds=poll_seconds,
            retry_exceptions=(ConnectionError, OSError, RuntimeError),
            after_reject=reject_terminal,
        )
        if result.matched:
            if not quiet:
                elapsed = time.monotonic() - started
                print(f"gameplayReady detected after {elapsed:.2f} seconds.")
            self.previous_observation = result.last_value
            return result.last_value
        raise TimeoutError(
            f"Timed out waiting for gameplayReady=true. Last observation: {result.last_value}"
        )

    def ensure_seeds_then_gameplay_ready(
        self,
        seed_list: Optional[List[str]] = None,
        timeout: float = 60.0,
        poll_seconds: float = 0.2,
        quiet: bool = False,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        names = seed_list or self.config.seed_list
        requested_sequence = resolve_seed_list(names)
        requested_counts = count_values(requested_sequence)
        deadline = time.monotonic() + timeout
        last_state: Dict[str, Any] = {}
        reward_cleanup_attempts = 0
        while time.monotonic() < deadline:
            probe = self.seed_probe()
            observation = self.observe() if probe.get("boardFound") else {}
            active_counts, _ = active_gameplay_bank_state(probe)
            last_state = {"probe": probe, "observation": observation}

            if not probe.get("boardFound"):
                time.sleep(poll_seconds)
                continue
            if probe.get("blockingRewardUiActive"):
                cleanup = self._cleanup_reward_ui_once(
                    observation,
                    allow_active_gameplay_reset=False,
                    reset_reason="ensure_seeds",
                )
                if cleanup.get("cleaned") and reward_cleanup_attempts < 3:
                    reward_cleanup_attempts += 1
                    time.sleep(max(0.25, poll_seconds))
                    continue
                raise RuntimeError(
                    "Cannot ensure seeds while reward/trophy UI remains visible or cleanup is blocked. "
                    f"blockingRewardUiActive={probe.get('blockingRewardUiActive')}, cleanup={cleanup}"
                )
            if observation.get("done"):
                raise RuntimeError(
                    "Cannot ensure seeds from terminal state; reset must clear it first. "
                    f"done={observation.get('done')}, terminalHint={observation.get('terminalHint')}"
                )
            if probe.get("seedSelectionActive"):
                selection = self.auto_select_seeds(seed_list=names, start_level=True)
                if not selection.get("ok", False):
                    raise RuntimeError(f"auto_select_seeds failed: {selection}")
                observation = self.wait_for_gameplay_ready(timeout=timeout, poll_seconds=poll_seconds, quiet=quiet, fail_on_terminal=False)
                final_probe = self.seed_probe()
                final_counts, _ = active_gameplay_bank_state(final_probe)
                if final_probe.get("seedSelectionActive") or not counts_cover(final_counts, requested_counts):
                    raise RuntimeError(
                        "auto_select_seeds returned but gameplay bank was not verified: "
                        f"seedSelectionActive={final_probe.get('seedSelectionActive')}, activeBank={format_counts(final_counts)}"
                    )
                return observation, selection

            if (
                observation.get("gameplayReady")
                and not observation.get("seedSelectionActive")
                and probe.get("gameplayReady")
                and counts_cover(active_counts, requested_counts)
            ):
                observation = self.wait_for_gameplay_ready(timeout=timeout, poll_seconds=poll_seconds, quiet=quiet, fail_on_terminal=False)
                return observation, {
                    "ok": True,
                    "alreadyGameplayReady": True,
                    "requestedSeedTypes": requested_sequence,
                    "activeGameplayCardBankTypeCounts": dict(active_counts),
                    "message": "Requested seeds were present in the active gameplay card bank and seed selection was inactive.",
                }

            time.sleep(poll_seconds)

        raise TimeoutError(f"Timed out ensuring seeds/gameplayReady. Last state: {last_state}")

    def _wait_for_cleanup_valid(
        self,
        require_mowers: bool = False,
        allow_active_gameplay_reset: bool = False,
        reset_reason: str = "",
    ) -> Tuple[Dict[str, Any], bool, str]:
        deadline = time.monotonic() + self.config.reset_wait_timeout
        last_observation: Dict[str, Any] = {}
        last_message = "no cleanup observation"
        cleanup_retry_count = 0
        next_cleanup_retry = time.monotonic()
        while time.monotonic() < deadline:
            time.sleep(max(0.05, self.config.reset_poll_seconds))
            observation = self.observe(debug_observation=True, force_seed_probe=True)
            last_observation = observation
            ok, message = verify_reset_cleanup_state(self, observation, require_mowers=require_mowers)
            last_message = message
            if ok:
                return observation, True, message
            if (
                require_mowers
                and "mower" in message.lower()
                and cleanup_retry_count < 3
                and time.monotonic() >= next_cleanup_retry
            ):
                retry = self.reset_cleanup(
                    reset_card_cooldowns=True,
                    allow_active_gameplay_reset=allow_active_gameplay_reset,
                    reset_reason=reset_reason,
                )
                cleanup_retry_count += 1
                retry_actions = retry.get("actions", [])
                last_message = f"{message}; cleanup_retry_{cleanup_retry_count}_actions={retry_actions}"
                next_cleanup_retry = time.monotonic() + max(0.5, self.config.reset_poll_seconds * 2.0)
        return last_observation, False, last_message

    def _wait_for_post_reset_playable(
        self,
        allow_active_gameplay_reset: bool = False,
        reset_reason: str = "",
        require_seed_selection_this_reset: bool = False,
        saw_seed_selection_this_reset: bool = False,
        clicked_lets_rock_this_reset: bool = False,
        require_mowers: bool = True,
    ) -> Tuple[Dict[str, Any], bool, str]:
        deadline = time.monotonic() + self.config.reset_wait_timeout
        last_observation: Dict[str, Any] = {}
        last_message = "no post-reset observation"
        seed_retry_count = 0
        cleanup_retry_count = 0
        reward_ui_cleanup_count = 0
        reward_ui_soft_reset_count = 0
        residue_soft_reset_count = 0
        wait_only_soft_reset_count = 0
        expected_counts = count_values(resolve_seed_list(self.config.seed_list))
        while time.monotonic() < deadline:
            time.sleep(max(0.05, self.config.reset_poll_seconds))
            observation = self.observe(debug_observation=True, force_seed_probe=True, force_restart_probe=True)
            last_observation = observation

            if is_restart_screen_observation(observation):
                last_message = "restart screen still active after reset"
                continue
            if observation.get("seedSelectionActive"):
                last_message = "seed selection still active after reset"
                if self.config.auto_select_seeds and seed_retry_count < 2:
                    seed_retry_count += 1
                    try:
                        self.auto_select_seeds(seed_list=self.config.seed_list, start_level=True)
                    except (ConnectionError, OSError):
                        self.client.close()
                        self.auto_select_seeds(seed_list=self.config.seed_list, start_level=True)
                    except Exception as exc:
                        last_message = f"post-reset seed retry failed: {exc}"
                continue
            if self._reward_or_trophy_ui_active(observation):
                last_message = (
                    "reward/trophy UI still active after reset "
                    f"nextStep={observation.get('nextStep')}"
                )
                if reward_ui_cleanup_count < 3:
                    reward_ui_cleanup_count += 1
                    self.reset_cleanup(
                        reset_card_cooldowns=True,
                        allow_active_gameplay_reset=allow_active_gameplay_reset,
                        reset_reason=reset_reason,
                    )
                    continue
                if observation.get("boardFound") and reward_ui_soft_reset_count < 2:
                    if require_seed_selection_this_reset:
                        last_message = (
                            "reward/trophy UI remained active but soft reset fallback is disabled for seed-selection-required resets"
                        )
                        return observation, False, last_message
                    reward_ui_soft_reset_count += 1
                    reward_ui_cleanup_count = 0
                    self.soft_reset(
                        start_sun=self.config.start_sun,
                        run_init=False,
                        manual_clear=True,
                        allow_active_gameplay_reset=allow_active_gameplay_reset,
                        reset_reason=reset_reason,
                    )
                    continue
                continue
            if not observation.get("gameplayReady"):
                last_message = f"gameplayReady=false nextStep={observation.get('nextStep')}"
                continue
            if self._loss_reset_gameplay_ready_without_seed_selection(
                observation,
                require_seed_selection_this_reset=require_seed_selection_this_reset,
                saw_seed_selection_this_reset=saw_seed_selection_this_reset,
            ):
                last_message = (
                    "unsafe seed-selection-required reset: gameplayReady observed before seed selection in this reset sequence"
                )
                print(
                    "[reset] rejecting playable board: "
                    f"{reset_reason} reset requires seed selection but "
                    f"sawSeed={bool(saw_seed_selection_this_reset)} "
                    f"seedSelectionActive={observation.get('seedSelectionActive')} "
                    f"screenState={observation.get('screenState')} "
                    f"gameplayReady={observation.get('gameplayReady')}"
                )
                return observation, False, last_message
            if require_seed_selection_this_reset and not clicked_lets_rock_this_reset:
                last_message = (
                    "unsafe seed-selection-required reset: gameplayReady observed before Let's Rock was clicked"
                )
                print(
                    "[reset] rejecting gameplay board: seed-selection-required reset has not clicked Let's Rock "
                    f"sawSeed={bool(saw_seed_selection_this_reset)} "
                    f"screenState={observation.get('screenState')} "
                    f"gameplayReady={observation.get('gameplayReady')}"
                )
                return observation, False, last_message
            if not observation.get("boardFound") or not observation.get("canReadBoard", True):
                last_message = "board is not readable"
                continue
            if self._is_dirty_active_gameplay_board(observation):
                last_message = (
                    "board still contains previous episode residue: "
                    f"plants={observation.get('plantCount')} "
                    f"visiblePlants={observation.get('visiblePlantObjectCount')} "
                    f"zombies={observation.get('zombieCount')} "
                    f"bullets={observation.get('bulletCount')}"
                )
                if residue_soft_reset_count < 3:
                    if require_seed_selection_this_reset:
                        last_message = (
                            "dirty board detected but soft reset fallback is disabled for seed-selection-required resets"
                        )
                        return observation, False, last_message
                    residue_soft_reset_count += 1
                    self.soft_reset(
                        start_sun=self.config.start_sun,
                        run_init=False,
                        manual_clear=True,
                        allow_active_gameplay_reset=allow_active_gameplay_reset,
                        reset_reason=reset_reason,
                    )
                    continue
            cleanup_ok, cleanup_message = verify_reset_cleanup_state(self, observation, require_mowers=require_mowers)
            if not cleanup_ok:
                last_message = cleanup_message
                if "mower" in cleanup_message.lower() and cleanup_retry_count < 3:
                    cleanup_retry_count += 1
                    self.reset_cleanup(
                        reset_card_cooldowns=True,
                        allow_active_gameplay_reset=allow_active_gameplay_reset,
                        reset_reason=reset_reason,
                    )
                continue
            slots = seed_slots_from_observation(observation, self.config.plant_types)
            slot_counts = count_values([int(slot.get("plantType", -999)) for slot in slots])
            if len(slots) < len(self.config.plant_types) or not counts_cover(slot_counts, expected_counts):
                last_message = (
                    f"seed slots not readable or wrong multiset: slots={len(slots)}, "
                    f"expected={format_counts(expected_counts)}, observed={format_counts(slot_counts)}"
                )
                continue
            mask = self.action_mask(observation)
            if not mask or not bool(mask[0]):
                last_message = "action mask missing legal wait action"
                continue
            legal = [action for action, allowed in enumerate(mask) if allowed]
            if legal == [0] and not self._wait_only_expected(observation):
                last_message = f"unexpected wait-only legal_actions after reset: {cleanup_message}"
                if wait_only_soft_reset_count < 2:
                    if require_seed_selection_this_reset:
                        last_message = (
                            "unexpected wait-only mask after reset but soft reset fallback is disabled for seed-selection-required resets"
                        )
                        return observation, False, last_message
                    wait_only_soft_reset_count += 1
                    self.soft_reset(
                        start_sun=self.config.start_sun,
                        run_init=False,
                        manual_clear=True,
                        allow_active_gameplay_reset=allow_active_gameplay_reset,
                        reset_reason=reset_reason,
                    )
                    continue
                continue
            mask_diag = self.mask_diagnostics(observation, mask)
            return observation, True, (
                f"gameplayReady=true boardReadable=true seedSlots={len(slots)} "
                f"legalActions={sum(1 for allowed in mask if allowed)} "
                f"legalActionsBySeedSlot={mask_diag.get('legal_actions_by_seed_slot')}; {cleanup_message}"
            )
        return last_observation, False, last_message

    def _wait_only_expected(self, observation: Dict[str, Any]) -> bool:
        if not observation.get("gameplayReady"):
            return True
        if int(observation.get("sun", 0)) <= 0:
            return True
        slots = observation.get("seedSlots", [])
        if slots and not any(bool(slot.get("ready")) and bool(slot.get("usable", True)) for slot in slots):
            return True
        if slots:
            positive_costs = [int(slot.get("seedCost", 0)) for slot in slots if int(slot.get("seedCost", 0)) > 0]
            if positive_costs:
                return int(observation.get("sun", 0)) < min(positive_costs)
        cooldowns = observation.get("cardCooldowns", [])
        if cooldowns and not any(bool(card.get("ready")) for card in cooldowns):
            return True
        plant_costs = observation.get("plantCosts", [])
        if plant_costs:
            positive_costs = [int(cost.get("cost", 0)) for cost in plant_costs if int(cost.get("cost", 0)) > 0]
            if positive_costs:
                return int(observation.get("sun", 0)) < min(positive_costs)
        return False

    def bridge_legal_actions(self, observation: Optional[Dict[str, Any]] = None) -> List[int]:
        started = time.perf_counter()
        if observation is not None and is_restart_screen_observation(observation):
            return [0]
        if observation is not None and observation.get("seedSelectionActive"):
            return [0]
        if observation is not None and "legalActions" in observation:
            return [int(action) for action in observation.get("legalActions", [])]
        data = self.client.request("legal_actions")
        if self.config.debug_performance:
            data["legal_actions_ms"] = data.get("legal_actions_ms", round((time.perf_counter() - started) * 1000.0, 3))
        if is_restart_screen_observation(data):
            return [0]
        if data.get("seedSelectionActive"):
            return [0]
        return [int(action) for action in data.get("legalActions", [])]

    def legal_actions(
        self,
        observation: Optional[Dict[str, Any]] = None,
        *,
        facts: Optional[StepFacts] = None,
    ) -> List[int]:
        obs = observation or self.previous_observation or self.observe()
        return [
            action
            for action, allowed in enumerate(self.action_mask(obs, facts=facts))
            if allowed
        ]

    def teacher_action(self, observation: Optional[Dict[str, Any]] = None) -> int:
        if observation is not None and is_restart_screen_observation(observation):
            return 0
        if observation is not None and observation.get("seedSelectionActive"):
            return 0
        try:
            data = self.client.request("teacher_action")
            if is_restart_screen_observation(data):
                return 0
            if data.get("seedSelectionActive"):
                return 0
            return int(data.get("action", 0))
        except Exception:
            return self.rule_based_teacher_action(observation)

    def _adventure_terminal_override(self) -> Tuple[Optional[str], Dict[str, Any]]:
        try:
            state = self.adventure_screen_state()
        except Exception as exc:
            return None, {"error": str(exc)}
        trophy_visible = bool(
            state.get("trophyVisible")
            or state.get("levelCompleteTrophyVisible")
            or state.get("postWinClickRequired")
        )
        reward_visible = bool(
            state.get("rewardScreenVisible")
            or state.get("unlockScreenVisible")
            or state.get("newPlantUnlockedVisible")
        )
        if trophy_visible:
            return "level_complete_trophy", state
        if reward_visible:
            return "reward_unlock", state
        return None, state

    def _fusion_diagnostics_for_step(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        policy = normalize_fusion_policy(getattr(self.config, "fusion_policy", FUSION_POLICY_NONE))
        if policy == FUSION_POLICY_NONE:
            self._last_fusion_diagnostics = default_fusion_diagnostics(policy)
            return self._last_fusion_diagnostics
        bridge_probe: Optional[Dict[str, Any]] = None
        bridge_error: Optional[str] = None
        try:
            bridge_probe = self.client.request("fusion_probe", return_observation=False)
        except Exception as exc:
            bridge_error = str(exc)
        facts = self._step_facts_cache.get_known(observation, self.config.plant_types)
        diagnostics = build_fusion_diagnostics(
            policy,
            observation,
            bridge_probe=bridge_probe,
            bridge_error=bridge_error,
            facts=facts,
        )
        self._last_fusion_diagnostics = diagnostics
        return diagnostics

    def _next_fusion_pipeline_event_id(self, source: str) -> str:
        counter = int(getattr(self, "_fusion_pipeline_event_counter", 0)) + 1
        self._fusion_pipeline_event_counter = counter
        epoch = int(getattr(self, "_fusion_pipeline_epoch", 0))
        return f"fusion:{epoch}:{str(source or 'manual')}:{counter}"

    def _execute_fusion_intent(
        self,
        pre_observation: Dict[str, Any],
        diagnostics: Dict[str, Any],
        intent: FusionIntent,
        *,
        precondition_rejection: str = "",
        require_known_recipe: bool = False,
        coach_bridge_command: Optional[Dict[str, Any]] = None,
        rejection_attempted: bool = False,
        return_rejection_result: bool = False,
        rejection_reason_aliases: Optional[Dict[str, str]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """Validate, execute, scope-check, decorate, and account one fusion.

        Model, scripted, human, stream, GUI/manual, and debug requests all enter
        here after source-specific parsing.  Unity's ``fusion_step`` response is
        still the final authority; Python only supplies a pure preflight guard.
        """

        event_id = self._next_fusion_pipeline_event_id(intent.source)
        facts = self._step_facts_cache.get_known(pre_observation, self.config.plant_types)
        decision = validate_fusion_intent(
            intent,
            pre_observation,
            fusion_enabled=True,
            fusion_bridge_available=bool(hasattr(getattr(self, "client", None), "request")),
            plant_types=self.config.plant_types,
            check_seed_resources=True,
            precondition_rejection=precondition_rejection,
            require_known_recipe=require_known_recipe,
            facts=facts,
        )
        if not decision.legal and isinstance(rejection_reason_aliases, dict):
            aliased_reason = str(rejection_reason_aliases.get(decision.rejection_reason) or decision.rejection_reason)
            if aliased_reason != decision.rejection_reason:
                decision = FusionDecision(
                    intent=decision.intent,
                    legal=False,
                    rejection_reason=aliased_reason,
                    recipe=decision.recipe,
                    compatibility_kind=decision.compatibility_kind,
                    bridge_command=decision.bridge_command,
                )
        candidate = intent.candidate_dict()
        result: Optional[Dict[str, Any]] = None
        attempted = False
        bridge_completed = False

        if not decision.legal:
            if isinstance(coach_bridge_command, dict):
                result = self._blocked_coach_fusion_result(
                    reason=decision.rejection_reason,
                    requested_action=int(intent.requested_action),
                    executed_action=int(intent.executed_action),
                    candidate=candidate,
                    coach_bridge_command=coach_bridge_command,
                    attempted=bool(rejection_attempted),
                )
            elif return_rejection_result:
                result = self._failed_fusion_action_result(
                    candidate=candidate,
                    reason=decision.rejection_reason,
                    requested_action=int(intent.requested_action),
                    executed_action=int(intent.executed_action),
                    source=self._legacy_fusion_execution_source(intent),
                    attempted=bool(rejection_attempted),
                )
            attempted = bool(rejection_attempted)
        else:
            bridge_payload = dict(decision.bridge_command)
            if isinstance(coach_bridge_command, dict):
                command_id = self._coach_command_id_from_bridge_command(coach_bridge_command)
                bridge_payload.update(
                    {
                        "coach_command_id": int(command_id or 0),
                        "coach_command_timestamp": float(coach_bridge_command.get("coach_command_timestamp") or 0.0),
                        "executed_from_fresh_coach_command": True,
                        "coach_command_age_seconds": self._coach_command_age_seconds(coach_bridge_command),
                        "coach_command_queue_cleared_on_reset": bool(self._coach_command_queue_cleared_on_reset),
                        "startup_command_blocked": False,
                    }
                )
            attempted = True
            try:
                raw_result = self.client.request("fusion_step", **bridge_payload, return_observation=False)
                if isinstance(raw_result, dict):
                    result = raw_result
                    bridge_completed = True
                else:
                    raise RuntimeError("fusion_step returned a non-object response")
            except Exception as exc:
                reason = "bridge_error"
                if isinstance(coach_bridge_command, dict):
                    result = self._blocked_coach_fusion_result(
                        reason=reason,
                        requested_action=int(intent.requested_action),
                        executed_action=int(intent.executed_action),
                        candidate=candidate,
                        coach_bridge_command=coach_bridge_command,
                        attempted=True,
                        bridge_error=str(exc),
                    )
                else:
                    result = self._failed_fusion_action_result(
                        candidate=candidate,
                        reason=reason,
                        requested_action=int(intent.requested_action),
                        executed_action=int(intent.executed_action),
                        source=self._legacy_fusion_execution_source(intent),
                        bridge_error=str(exc),
                        attempted=True,
                    )

        if isinstance(result, dict):
            result.setdefault("requestedAction", int(intent.requested_action))
            result.setdefault("executedAction", int(intent.executed_action))
            result.setdefault("fusionAttempted", bool(attempted))
            result["fusionOverrideApplied"] = bool(result.get("fusionSucceeded"))
            result["fusionExecutionSource"] = self._legacy_fusion_execution_source(intent)
            result["fusionIntentSource"] = str(intent.source)
            result["fusionEventId"] = event_id
            result.setdefault("requestedSourceRow", int(intent.source_row))
            result.setdefault("requestedSourceCol", int(intent.source_col))
            result.setdefault("requestedSourceInstanceId", int(intent.source_instance_id))
            result["fusionCandidate"] = compact_candidate(candidate)
            result["fusionIntent"] = {
                "source": str(intent.source),
                "row": int(intent.source_row),
                "column": int(intent.source_col),
                "seed_slot": int(intent.ingredient_seed_slot_index),
                "source_plant_type": int(intent.source_plant_type),
                "ingredient_plant_type": int(intent.ingredient_plant_type),
                "predicted_result_type": int(intent.predicted_result_type),
                "compatibility_kind": str(intent.compatibility_kind),
            }
            result.setdefault(
                "decoded",
                {
                    "kind": "fusion",
                    "sourcePlantType": int(intent.source_plant_type),
                    "ingredientPlantType": int(intent.ingredient_plant_type),
                    "resultPlantType": int(intent.predicted_result_type),
                    "row": int(intent.source_row),
                    "column": int(intent.source_col),
                },
            )
            self._enforce_fusion_scope_contract(result)
            if isinstance(coach_bridge_command, dict):
                command_id = self._coach_command_id_from_bridge_command(coach_bridge_command)
                result["coachFusionOverrideApplied"] = bool(result.get("fusionSucceeded"))
                result.setdefault("executed_from_fresh_coach_command", bool(bridge_completed))
                result.setdefault("coach_command_age_seconds", self._coach_command_age_seconds(coach_bridge_command))
                result.setdefault("startup_command_blocked", False)
                result.setdefault("coach_command_queue_cleared_on_reset", bool(self._coach_command_queue_cleared_on_reset))
                result.setdefault("coach_command_source", str(coach_bridge_command.get("coach_command_source") or ""))
                result.setdefault("executed_coach_command_id", command_id)
                if bridge_completed and command_id is not None:
                    self._executed_coach_command_ids.add(int(command_id))
                    self._last_executed_coach_command_id = int(command_id)
                result["last_executed_coach_command_id"] = self._last_executed_coach_command_id

        execution = fusion_execution_from_result(
            decision,
            result,
            event_id=event_id,
            attempted=bool(attempted),
            rejection_reason=decision.rejection_reason,
        )
        accounted = getattr(self, "_fusion_accounted_event_ids", None)
        if not isinstance(accounted, set):
            accounted = set()
            self._fusion_accounted_event_ids = accounted
        diagnostics, applied = account_fusion_execution_once(diagnostics, execution, accounted)
        if isinstance(result, dict):
            result["fusionAccountingApplied"] = bool(applied)
        self._last_fusion_diagnostics = diagnostics
        return result, diagnostics

    @staticmethod
    def _legacy_fusion_execution_source(intent: FusionIntent) -> str:
        explicit = str(intent.metadata.get("legacy_execution_source") or "")
        if explicit:
            return explicit
        if intent.source == FUSION_SOURCE_MODEL:
            return "model_action_mask"
        return str(intent.source)

    def _maybe_execute_scripted_fusion(
        self,
        pre_observation: Dict[str, Any],
        diagnostics: Dict[str, Any],
        requested_action: int,
        executed_action: int,
        action_count: int,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        policy = normalize_fusion_policy(getattr(self.config, "fusion_policy", FUSION_POLICY_NONE))
        if policy != FUSION_POLICY_SCRIPTED:
            return None, diagnostics
        candidate, select_rejection = choose_scripted_fusion_candidate(diagnostics)
        validation_rejection = validate_scripted_fusion_candidate(
            candidate,
            pre_observation,
            action_count=int(action_count),
            expected_action_count=ADVENTURE_IDENTITY_ACTION_COUNT,
            reset_active=False,
            action_already_executed=False,
        )
        rejection = validation_rejection or select_rejection
        # A no-candidate scan is observational, not an attempted fusion.  Once a
        # concrete candidate exists, every source uses the same pipeline.
        if candidate is None:
            self._last_fusion_diagnostics = diagnostics
            return None, diagnostics
        intent = fusion_intent_from_candidate(
            candidate,
            source=FUSION_SOURCE_SCRIPTED,
            requested_action=requested_action,
            executed_action=executed_action,
            metadata={"legacy_execution_source": "scripted"},
        )
        return self._execute_fusion_intent(
            pre_observation,
            diagnostics,
            intent,
            precondition_rejection=rejection,
            require_known_recipe=True,
            rejection_attempted=False,
            return_rejection_result=False,
        )

    def _maybe_execute_model_fusion(
        self,
        pre_observation: Dict[str, Any],
        diagnostics: Dict[str, Any],
        requested_action: int,
        executed_action: int,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """Route a model placement action onto an occupied compatible tile to fusion.

        Only fires when the model-fusion action mask is enabled.  The mask has
        already verified compatibility, occupancy, and seed readiness, so this
        simply rebuilds the candidate from the chosen action and asks the bridge
        to perform the fusion (mirroring the scripted fusion path).
        """

        if not self._fusion_action_mask_enabled():
            return None, diagnostics
        if int(executed_action) <= 0:
            return None, diagnostics
        facts = self._step_facts_cache.get_known(
            pre_observation,
            self.config.plant_types,
        )
        rows = int(facts.rows)
        cols = int(facts.columns)
        if rows <= 0 or cols <= 0:
            return None, diagnostics
        cells = rows * cols
        encoded = int(executed_action) - 1
        slot_index = encoded // cells
        cell = encoded % cells
        row, col = cell // cols, cell % cols
        slot_fact = (
            facts.seed_slots[slot_index]
            if 0 <= slot_index < len(facts.seed_slots)
            and facts.seed_slots[slot_index].valid
            else None
        )
        seed_type = int(slot_fact.plant_type) if slot_fact else -1
        occupant = facts.occupant_by_cell.get((row, col))
        if occupant is None:
            return None, diagnostics  # empty tile -> normal placement, let bridge step handle it
        existing_type = int(occupant.plant_type)
        primary_stack = facts.primary_plant_stacks_by_cell.get((row, col), ())
        source_plant = primary_stack[0] if primary_stack else None
        candidate = {
            "source_plant_type": int(existing_type),
            "source_plant_name": str(
                (source_plant.type_name if source_plant is not None else "")
                or fusion_plant_name(existing_type)
            ),
            "source_instance_id": int(
                source_plant.instance_id if source_plant is not None else 0
            ),
            "source_row": row,
            "source_col": col,
            "target_or_ingredient_type": int(seed_type),
            "target_or_ingredient_name": str(
                (slot_fact.plant_type_name if slot_fact is not None else "")
                or fusion_plant_name(seed_type)
            ),
            "ingredient_seed_slot_index": int(
                slot_fact.slot_index if slot_fact is not None else slot_index
            ),
            "ingredient_card_instance_id": int(
                slot_fact.card_instance_id if slot_fact is not None else 0
            ),
            "predicted_result_type": -1,
            "predicted_result_name": "",
            "fusion_legal": True,
            "fusion_blocked_reason": "",
            "fusion_runtime_authorized": True,
        }
        intent = fusion_intent_from_candidate(
            candidate,
            source=FUSION_SOURCE_MODEL,
            requested_action=requested_action,
            executed_action=executed_action,
            metadata={"legacy_execution_source": "model_action_mask"},
        )
        # Always return an occupied-tile decision (including a clean rejection)
        # so a stale/mismatched mask can never fall through to normal placement.
        return self._execute_fusion_intent(
            pre_observation,
            diagnostics,
            intent,
            rejection_attempted=False,
            return_rejection_result=True,
        )

    # ------------------------------------------------------------------
    # Fusion reward policy (pure compositor adapters)
    # ------------------------------------------------------------------

    def _fusion_reward_live_fields(self) -> Dict[str, Any]:
        return fusion_reward_live_fields(self._reward_state.fusion)

    def _coach_command_id_from_bridge_command(self, coach_bridge_command: Dict[str, Any]) -> Optional[int]:
        try:
            command_id = int(coach_bridge_command.get("coach_command_id") or 0)
        except (TypeError, ValueError):
            command_id = 0
        return int(command_id) if command_id > 0 else None

    def _coach_command_age_seconds(self, coach_bridge_command: Dict[str, Any]) -> Optional[float]:
        try:
            timestamp = float(coach_bridge_command.get("coach_command_timestamp") or 0.0)
        except (TypeError, ValueError):
            timestamp = 0.0
        if timestamp <= 0.0:
            return None
        return max(0.0, time.time() - timestamp)

    def _failed_fusion_action_result(
        self,
        *,
        candidate: Dict[str, Any],
        reason: str,
        requested_action: int,
        executed_action: int,
        source: str,
        bridge_error: str = "",
        attempted: bool = True,
    ) -> Dict[str, Any]:
        """Build a reward-visible result for a fusion that reached an attempt and failed."""
        return {
            "action": int(executed_action),
            "requestedAction": int(requested_action),
            "executedAction": int(executed_action),
            "fusionAttempted": bool(attempted),
            "fusionSucceeded": False,
            "fusion_success": False,
            "fusionOverrideApplied": False,
            "illegalAction": True,
            "illegalReason": str(reason),
            "fusionRejectedReason": str(reason),
            "bridgeResultReason": str(reason),
            "bridgeError": str(bridge_error or ""),
            "changedTileCount": 0,
            "changed_tile_count": 0,
            "bridgeMethodUsed": "error" if bridge_error else "none",
            "fusionExecutionSource": str(source),
            "fusionCandidate": compact_candidate(candidate),
            "decoded": {
                "kind": "fusion",
                "sourcePlantType": int(candidate.get("source_plant_type", -1)),
                "ingredientPlantType": int(candidate.get("target_or_ingredient_type", -1)),
                "resultPlantType": int(candidate.get("predicted_result_type", -1)),
                "row": int(candidate.get("source_row", -1)),
                "column": int(candidate.get("source_col", -1)),
            },
        }

    def _coach_fusion_freshness_rejection(
        self,
        pre_observation: Dict[str, Any],
        coach_bridge_command: Dict[str, Any],
    ) -> str:
        if not bool(pre_observation.get("gameplayReady")):
            return "gameplay_not_ready"
        command_id = self._coach_command_id_from_bridge_command(coach_bridge_command)
        if command_id is None:
            return "startup_stale_command_blocked"
        if command_id in self._executed_coach_command_ids:
            return "coach_command_already_executed"
        if not bool(coach_bridge_command.get("executed_from_fresh_coach_command")):
            return "startup_stale_command_blocked"
        try:
            command_timestamp = float(coach_bridge_command.get("coach_command_timestamp") or 0.0)
        except (TypeError, ValueError):
            command_timestamp = 0.0
        if command_timestamp > 0.0 and command_timestamp + 1e-6 < float(self._coach_fusion_fresh_after_timestamp):
            return "startup_stale_command_blocked"
        return ""

    def _blocked_coach_fusion_result(
        self,
        *,
        reason: str,
        requested_action: int,
        executed_action: int,
        candidate: Dict[str, Any],
        coach_bridge_command: Dict[str, Any],
        attempted: bool = False,
        bridge_error: str = "",
    ) -> Dict[str, Any]:
        startup_blocked = str(reason) == "startup_stale_command_blocked"
        if startup_blocked:
            self._startup_command_blocked = True
        command_id = self._coach_command_id_from_bridge_command(coach_bridge_command)
        age_seconds = self._coach_command_age_seconds(coach_bridge_command)
        scope = "startup_stale_command_blocked" if startup_blocked else "unknown"
        return {
            "action": int(executed_action),
            "requestedAction": int(requested_action),
            "executedAction": int(executed_action),
            "fusionAttempted": bool(attempted),
            "fusionSucceeded": False,
            "fusion_success": False,
            "fusionOverrideApplied": False,
            "coachFusionOverrideApplied": False,
            "illegalAction": True,
            "illegalReason": str(reason),
            "fusionRejectedReason": str(reason),
            "bridgeResultReason": str(reason),
            "bridge_result_reason": str(reason),
            "bridgeError": str(bridge_error or ""),
            "coach_command_source": str(coach_bridge_command.get("coach_command_source") or ""),
            "fusionExecutionSource": str(coach_bridge_command.get("coach_command_source") or "human_coach"),
            "fusionScope": scope,
            "fusion_scope": scope,
            "requestedSourceRow": int(candidate.get("source_row", -1)),
            "requestedSourceCol": int(candidate.get("source_col", -1)),
            "requestedSourceInstanceId": int(candidate.get("source_instance_id", 0) or 0),
            "requested_source_row": int(candidate.get("source_row", -1)),
            "requested_source_col": int(candidate.get("source_col", -1)),
            "requested_source_instance_id": int(candidate.get("source_instance_id", 0) or 0),
            "sourceTileOccupiedBefore": False,
            "source_tile_occupied_before": False,
            "sourcePlantBefore": None,
            "source_plant_before": None,
            "resultingPlantAfter": None,
            "resulting_plant_after": None,
            "changedTileCount": 0,
            "changed_tile_count": 0,
            "changedTiles": [],
            "changed_tiles": [],
            "nonSourceTilesChanged": False,
            "non_source_tiles_changed": False,
            "globalFusionSideEffect": False,
            "global_fusion_side_effect": False,
            "duplicateStackDetected": False,
            "duplicate_stack_detected": False,
            "bridgeMethodUsed": "error" if bridge_error else "none",
            "bridge_method_used": "error" if bridge_error else "none",
            "executed_from_fresh_coach_command": False,
            "coach_command_age_seconds": age_seconds,
            "startup_command_blocked": bool(startup_blocked),
            "coach_command_queue_cleared_on_reset": bool(self._coach_command_queue_cleared_on_reset),
            "executed_coach_command_id": command_id,
            "last_executed_coach_command_id": self._last_executed_coach_command_id,
            "fusionCandidate": compact_candidate(candidate),
        }

    def _enforce_fusion_scope_contract(self, result: Dict[str, Any]) -> None:
        enforce_fusion_scope_contract(result)

    def _maybe_execute_coach_fusion_command(
        self,
        pre_observation: Dict[str, Any],
        diagnostics: Dict[str, Any],
        requested_action: int,
        executed_action: int,
        coach_bridge_command: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        if not isinstance(coach_bridge_command, dict):
            return None, diagnostics
        if str(coach_bridge_command.get("command") or "").strip().lower() != "fusion_step":
            return None, diagnostics
        candidate = self._coach_fusion_candidate_from_bridge_command(coach_bridge_command)
        freshness_rejection = self._coach_fusion_freshness_rejection(pre_observation, coach_bridge_command)
        rejection = freshness_rejection or self._coach_fusion_rejection_reason(
            pre_observation,
            candidate,
        )
        raw_source = str(coach_bridge_command.get("coach_command_source") or "human_coach")
        intent = fusion_intent_from_candidate(
            candidate,
            source=raw_source,
            requested_action=requested_action,
            executed_action=executed_action,
            metadata={"legacy_execution_source": raw_source},
        )
        return self._execute_fusion_intent(
            pre_observation,
            diagnostics,
            intent,
            precondition_rejection=rejection,
            coach_bridge_command=coach_bridge_command,
            # Stale/startup commands never reached execution.  Other coach
            # preflight failures historically represent an attempted command.
            rejection_attempted=bool(rejection and not freshness_rejection),
            return_rejection_result=True,
            rejection_reason_aliases={
                "invalid_row": "invalid_row_col",
                "invalid_col": "invalid_row_col",
                "invalid_seed_slot": "target_not_available",
                "seed_unavailable": "target_not_available",
                "cooldown_not_ready": "target_not_available",
                "insufficient_sun": "target_not_available",
            },
        )

    def _coach_fusion_candidate_from_bridge_command(self, coach_bridge_command: Dict[str, Any]) -> Dict[str, Any]:
        candidate = coach_bridge_command.get("candidate")
        if isinstance(candidate, dict):
            payload = dict(candidate)
        else:
            payload = {}
        payload.setdefault("source_instance_id", int(coach_bridge_command.get("source_instance_id") or 0))
        payload.setdefault("source_row", int(coach_bridge_command.get("source_row", -1)))
        payload.setdefault("source_col", int(coach_bridge_command.get("source_col", -1)))
        payload.setdefault("source_plant_type", int(coach_bridge_command.get("source_plant_type", -1)))
        payload.setdefault("ingredient_seed_slot_index", int(coach_bridge_command.get("ingredient_seed_slot_index", -1)))
        payload.setdefault("target_or_ingredient_type", int(coach_bridge_command.get("ingredient_plant_type", -1)))
        payload.setdefault("predicted_result_type", int(coach_bridge_command.get("predicted_result_type", -1)))
        payload.setdefault("predicted_result_name", str(coach_bridge_command.get("predicted_result_name") or ""))
        payload.setdefault("fusion_legal", True)
        payload.setdefault("fusion_blocked_reason", "")
        payload.setdefault("fusion_runtime_authorized", True)
        return payload

    def _coach_fusion_rejection_reason(
        self,
        observation: Dict[str, Any],
        candidate: Dict[str, Any],
    ) -> str:
        if bool(observation.get("done")) or bool(observation.get("over")):
            return "terminal_or_transition_state"
        if bool(observation.get("seedSelectionActive")):
            return "seed_selection_active"
        if not bool(observation.get("boardFound", True)):
            return "not_gameplay"
        if not bool(observation.get("gameplayReady")):
            return "gameplay_not_ready"
        row = int(candidate.get("source_row", -1))
        col = int(candidate.get("source_col", -1))
        rows = int(observation.get("rowCount") or self.config.row_count or 5)
        cols = int(observation.get("columnCount") or self.config.column_count or 10)
        if not (0 <= row < rows and 0 <= col < cols):
            return "invalid_row_col"
        # Source identity, slot identity, compatibility, CardUI cost/cooldown,
        # and unlock/readiness are handled once by validate_fusion_intent in the
        # shared execution adapter.  This forwarding hook owns only coach-only
        # lifecycle precedence retained for compatibility.
        return ""

    def step(
        self,
        action: int,
        *,
        coach_bridge_command: Optional[Dict[str, Any]] = None,
        coach_context: Optional[Dict[str, Any]] = None,
        action_intent: Optional[ActionIntent] = None,
        action_source: str = "direct",
        action_source_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        step_started = time.perf_counter()

        def attach_structured_action_result(
            observation: Dict[str, Any],
            action_result: Dict[str, Any],
            *,
            decision: Optional[ActionDecision] = None,
            executed: bool,
        ) -> ActionDecision:
            resolved_decision = decision
            if resolved_decision is None:
                context_payload = dict(coach_context or {})
                source = str(context_payload.get("source") or action_source or "direct")
                metadata = dict(action_source_metadata or {})
                if context_payload:
                    metadata.update(context_payload)
                resolved_decision = self.action_decision(
                    int(action),
                    observation,
                    source=source,
                    source_metadata=metadata,
                    bridge_command=coach_bridge_command,
                    intent=action_intent,
                )
            structured = ActionResult.from_execution(
                resolved_decision,
                dict(action_result),
                executed=bool(executed),
            )
            action_result["structuredActionResult"] = structured.to_dict(
                include_execution_result=False
            )
            action_result["actionIntent"] = resolved_decision.intent.to_dict()
            action_result["actionDecision"] = resolved_decision.to_dict()
            return resolved_decision

        def restart_terminal_return(
            observation: Dict[str, Any],
            action_result: Dict[str, Any],
            reward_action_result: Optional[Dict[str, Any]] = None,
            terminal_reason: str = "game_over_restart_screen",
            done_reason: Optional[str] = None,
            adventure_state: Optional[Dict[str, Any]] = None,
            *,
            executed: bool = False,
            decision: Optional[ActionDecision] = None,
            previous_facts: Optional[StepFacts] = None,
            current_facts: Optional[StepFacts] = None,
            previous_legal_actions: Optional[List[int]] = None,
            fusion_diagnostics: Optional[Dict[str, Any]] = None,
            pre_step_mask_blocked_action: bool = False,
        ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
            attach_structured_action_result(
                observation,
                action_result,
                decision=decision,
                executed=executed,
            )
            previous_observation = self.previous_observation
            prior_snapshot = previous_facts or (
                self._step_facts_cache.get(
                    previous_observation,
                    self.config.plant_types,
                )
                if isinstance(previous_observation, dict)
                else None
            )
            next_snapshot = current_facts or (
                prior_snapshot
                if previous_observation is observation and prior_snapshot is not None
                else self._step_facts_cache.get(observation, self.config.plant_types)
            )
            prior_legal = (
                list(previous_legal_actions)
                if previous_legal_actions is not None
                else self.legal_actions(
                    previous_observation,
                    facts=prior_snapshot,
                )
                if isinstance(previous_observation, dict)
                else []
            )
            reward_result = reward_action_result or action_result
            classified_reason = classify_done_reason(observation)
            terminal_reward_override: Optional[float] = None
            if done_reason == "post_win_pending":
                # This restart UI belongs to a win that was already accounted
                # for by the Adventure progression owner.
                terminal_reward_override = 0.0
            elif done_reason == "win" or (not done_reason and classified_reason == "win"):
                terminal_reward_override = float(self.config.reward.win_reward)
            elif done_reason == "loss" or (not done_reason and classified_reason == "loss"):
                terminal_reward_override = -float(self.config.reward.loss_penalty)
            reward_composition = self._compose_step_reward(
                previous_observation,
                observation,
                reward_result,
                terminal_reward_override=terminal_reward_override,
                previous_facts=prior_snapshot,
                current_facts=next_snapshot,
                previous_legal_actions=prior_legal,
            )
            if reward_result is not action_result:
                action_result.update(reward_composition.annotations_dict())
            reward_events = reward_composition.diagnostics_dict()
            reward_breakdown = reward_composition.breakdown.to_dict()
            reward = float(reward_breakdown["reward_total"])
            resolved_fusion_diagnostics = (
                fusion_diagnostics
                if isinstance(fusion_diagnostics, dict)
                else action_result.get("fusionDiagnostics")
                if isinstance(action_result.get("fusionDiagnostics"), dict)
                else None
            )
            if resolved_fusion_diagnostics is not None:
                resolved_fusion_diagnostics.update(self._fusion_reward_live_fields())
            legal_started = time.perf_counter()
            legal = self.legal_actions(observation, facts=next_snapshot)
            legal_ms = round((time.perf_counter() - legal_started) * 1000.0, 3)
            info: Dict[str, Any] = {
                "action_result": action_result,
                "reward_breakdown": reward_breakdown,
                "reward_diagnostics": reward_events,
                "terminal_hint": observation.get("terminalHint"),
                "terminal_reason": terminal_reason,
                "needs_reset": True,
                "legal_actions": legal,
                "bridge_legal_actions": self.bridge_legal_actions(observation),
                "pre_step_mask_blocked_action": bool(pre_step_mask_blocked_action),
            }
            if resolved_fusion_diagnostics is not None:
                info["fusion_diagnostics"] = resolved_fusion_diagnostics
                info.update(
                    fusion_live_fields(
                        resolved_fusion_diagnostics,
                        self.config.fusion_policy,
                    )
                )
            if done_reason:
                info["done_reason"] = done_reason
            if adventure_state is not None:
                info["adventure_state"] = adventure_state
            mask_diag = self.mask_diagnostics(observation, facts=next_snapshot)
            info["mask_diagnostics"] = mask_diag
            info["lane_diagnostics"] = self.lane_diagnostics(
                previous_observation,
                observation,
                action_result,
                legal,
                cherry_delayed_diagnostics=reward_events.get("cherry_delayed"),
                previous_facts=prior_snapshot,
                current_facts=next_snapshot,
                previous_legal_actions=prior_legal,
                mask_diagnostics_snapshot=mask_diag,
            )
            if self.config.debug_performance:
                info["performance"] = {
                    "step_ms": round((time.perf_counter() - step_started) * 1000.0, 3),
                    "python_step_ms": round((time.perf_counter() - step_started) * 1000.0, 3),
                    "legal_actions_ms": legal_ms,
                    "observe_ms": observation.get("observe_ms"),
                    "bridge_observe_ms": observation.get("bridge_observe_ms"),
                    "screen_check_ms": observation.get("screen_check_ms"),
                    "seed_probe_ms": observation.get("seed_probe_ms"),
                    "ui_scan_ms": observation.get("ui_scan_ms"),
                    "bridge_roundtrip_ms": observation.get("bridge_roundtrip_ms"),
                    "bridge_step_ms": action_result.get("step_ms") if isinstance(action_result, dict) else None,
                    "restart_detection_mode": observation.get("restartDetectionMode"),
                }
            self.previous_observation = observation
            self._steps_since_seed_screen_check = self.config.seed_screen_check_interval
            return observation, reward, True, False, info

        interval = max(0, int(self.config.seed_screen_check_interval))
        reward_previous_observation = self.previous_observation
        reward_previous_facts: Optional[StepFacts] = None
        pre_observation = self.previous_observation or self.observe()
        if is_restart_screen_observation(pre_observation):
            override_reason, adventure_state = self._adventure_terminal_override()
            if override_reason:
                action_result = {
                    "action": int(action),
                    "illegalAction": False,
                    "illegalReason": None,
                    "plantPlaced": False,
                    "costPaid": False,
                    "cooldownStarted": False,
                    "observation": pre_observation,
                }
                return restart_terminal_return(
                    pre_observation,
                    action_result,
                    terminal_reason=override_reason,
                    done_reason="post_win_pending",
                    adventure_state=adventure_state,
                )
            action_result = {
                "action": int(action),
                "illegalAction": True,
                "illegalReason": "game_over_restart_screen",
                "plantPlaced": False,
                "costPaid": False,
                "cooldownStarted": False,
                "observation": pre_observation,
            }
            return restart_terminal_return(pre_observation, action_result)

        needs_seed_screen_check = (
            self.previous_observation is None
            or bool(self.previous_observation.get("seedSelectionActive"))
            or not bool(self.previous_observation.get("gameplayReady"))
            or (interval > 0 and self._steps_since_seed_screen_check >= interval)
        )
        if needs_seed_screen_check:
            if isinstance(reward_previous_observation, dict):
                # Capture the prior owner before the forced probe advances the
                # execution frame.  On normal steps this is a zero-hash known-
                # owner reuse; on externally injected state it builds once.
                reward_previous_facts = self._step_facts_cache.get_known(
                    reward_previous_observation,
                    self.config.plant_types,
                )
            pre_observation = self.observe(force_seed_probe=True)
            self._steps_since_seed_screen_check = 0
            if is_restart_screen_observation(pre_observation):
                override_reason, adventure_state = self._adventure_terminal_override()
                if override_reason:
                    action_result = {
                        "action": int(action),
                        "illegalAction": False,
                        "illegalReason": None,
                        "plantPlaced": False,
                        "costPaid": False,
                        "cooldownStarted": False,
                        "observation": pre_observation,
                    }
                    return restart_terminal_return(
                        pre_observation,
                        action_result,
                        terminal_reason=override_reason,
                        done_reason="post_win_pending",
                        adventure_state=adventure_state,
                        previous_facts=reward_previous_facts,
                    )
                action_result = {
                    "action": int(action),
                    "illegalAction": True,
                    "illegalReason": "game_over_restart_screen",
                    "plantPlaced": False,
                    "costPaid": False,
                    "cooldownStarted": False,
                    "observation": pre_observation,
                }
                return restart_terminal_return(
                    pre_observation,
                    action_result,
                    previous_facts=reward_previous_facts,
                )
        requested_action = int(action)
        coach_bridge_payload = dict(coach_bridge_command) if isinstance(coach_bridge_command, dict) else None
        coach_context_payload = dict(coach_context) if isinstance(coach_context, dict) else None
        executed_action = requested_action
        pre_step_facts = self._step_facts_cache.get(
            pre_observation,
            self.config.plant_types,
        )
        decision_source = str((coach_context_payload or {}).get("source") or action_source or "direct")
        canonical_viewer_execution = decision_source.strip().lower() in {
            "twitch",
            "viewer",
        }
        # Streamer policy restrictions (including tactical masks) constrain
        # PPO sampling, but they are not physical game legality.  Viewer
        # execution must validate against the canonical action path for this
        # exact pre-dispatch observation.
        pre_step_mask = self.action_mask(
            pre_observation,
            facts=pre_step_facts,
            include_tactical_masks=not canonical_viewer_execution,
        )
        pre_step_legal_actions = [
            index for index, allowed in enumerate(pre_step_mask) if bool(allowed)
        ]
        decision_metadata = dict(action_source_metadata or {})
        if coach_context_payload:
            decision_metadata.update(coach_context_payload)
        pre_step_decision = self.action_decision(
            requested_action,
            pre_observation,
            source=decision_source,
            source_metadata=decision_metadata,
            bridge_command=coach_bridge_payload,
            intent=action_intent,
            facts=pre_step_facts,
            include_tactical_masks=not canonical_viewer_execution,
        )
        pre_step_mask_blocked = bool(requested_action != 0 and not pre_step_decision.legal)
        pre_step_audit: Dict[str, Any] = {}
        if pre_step_mask_blocked:
            pre_step_audit = self.action_legality_audit(
                requested_action,
                pre_observation,
                pre_step_mask,
                facts=pre_step_facts,
            )
            executed_action = 0

        fusion_diagnostics = self._fusion_diagnostics_for_step(pre_observation)
        coach_fusion_action_result, fusion_diagnostics = self._maybe_execute_coach_fusion_command(
            pre_observation,
            fusion_diagnostics,
            requested_action,
            executed_action,
            coach_bridge_payload,
        )
        fusion_action_result: Optional[Dict[str, Any]] = coach_fusion_action_result
        if fusion_action_result is None:
            fusion_action_result, fusion_diagnostics = self._maybe_execute_model_fusion(
                pre_observation,
                fusion_diagnostics,
                requested_action,
                executed_action,
            )
        if fusion_action_result is None:
            fusion_action_result, fusion_diagnostics = self._maybe_execute_scripted_fusion(
                pre_observation,
                fusion_diagnostics,
                requested_action,
                executed_action,
                len(pre_step_mask),
            )
        if isinstance(reward_previous_observation, dict):
            if reward_previous_observation is pre_observation:
                reward_previous_facts = pre_step_facts
                reward_previous_legal_actions = pre_step_legal_actions
            else:
                # A forced seed probe may refresh execution state without
                # advancing the reward owner.  Keep the stored observation,
                # its facts, and its legal actions as one coherent snapshot.
                if reward_previous_facts is None:
                    reward_previous_facts = build_step_facts(
                        reward_previous_observation,
                        self.config.plant_types,
                    )
                reward_previous_legal_actions = self.legal_actions(
                    reward_previous_observation,
                    facts=reward_previous_facts,
                )
        else:
            reward_previous_facts = None
            reward_previous_legal_actions = []
        try:
            action_result = fusion_action_result or self.client.request("step", action=executed_action, return_observation=False)
        except BridgeTimeoutError as exc:
            observation = pre_observation if isinstance(pre_observation, dict) else {}
            legal = (
                list(pre_step_legal_actions)
                if observation
                else [0]
            )
            bridge_event = {
                "event": "bridge_timeout",
                "command": exc.command,
                "timeout": exc.timeout,
                "requestedAction": requested_action,
                "executedAction": executed_action,
                "screenState": observation.get("screenState"),
                "nextStep": observation.get("nextStep"),
                "gameplayReady": observation.get("gameplayReady"),
                "seedSelectionActive": observation.get("seedSelectionActive"),
                "terminalHint": observation.get("terminalHint"),
            }
            print(
                "[bridge] timeout "
                f"command={exc.command} timeout={exc.timeout:.1f}s "
                f"screenState={observation.get('screenState')} "
                f"nextStep={observation.get('nextStep')} "
                f"gameplayReady={observation.get('gameplayReady')} "
                f"seedSelectionActive={observation.get('seedSelectionActive')} "
                f"terminalHint={observation.get('terminalHint')}"
            )
            action_result = {
                "action": int(executed_action),
                "requestedAction": requested_action,
                "executedAction": executed_action,
                "illegalAction": False,
                "illegalReason": None,
                "plantPlaced": False,
                "costPaid": False,
                "cooldownStarted": False,
                "bridgeTimeout": True,
                "observation": observation,
            }
            attach_structured_action_result(
                observation,
                action_result,
                decision=pre_step_decision,
                executed=True,
            )
            reward_composition = self._compose_step_reward(
                reward_previous_observation,
                observation,
                action_result,
                exceptional_adjustments={"env_corruption_penalty": -10.0},
                previous_facts=reward_previous_facts,
                current_facts=pre_step_facts,
                previous_legal_actions=reward_previous_legal_actions,
            )
            reward_events = reward_composition.diagnostics_dict()
            reward_breakdown = reward_composition.breakdown.to_dict()
            reward = float(reward_breakdown["reward_total"])
            info = {
                "action_result": action_result,
                "reward_breakdown": reward_breakdown,
                "reward_diagnostics": reward_events,
                "terminal_hint": observation.get("terminalHint"),
                "terminal_reason": "bridge_timeout",
                "done_reason": "env_corruption",
                "needs_reset": True,
                "bridge_error": str(exc),
                "bridge_errors": 1,
                "legal_actions": legal,
                "bridge_legal_actions": legal,
                "pre_step_mask_blocked_action": pre_step_mask_blocked,
                "environment_corruption_detected": True,
                "env_corruption_count": 1,
                "safety_events": [bridge_event],
                "fusion_diagnostics": fusion_diagnostics,
            }
            mask_diag = (
                self.mask_diagnostics(observation, facts=pre_step_facts)
                if observation
                else {}
            )
            info["mask_diagnostics"] = mask_diag
            info["lane_diagnostics"] = self.lane_diagnostics(
                reward_previous_observation,
                observation,
                action_result,
                legal,
                cherry_delayed_diagnostics=reward_events.get("cherry_delayed"),
                previous_facts=reward_previous_facts,
                current_facts=pre_step_facts,
                previous_legal_actions=reward_previous_legal_actions,
                mask_diagnostics_snapshot=mask_diag,
            ) if observation else {}
            self.previous_observation = observation
            self._steps_since_seed_screen_check = self.config.seed_screen_check_interval
            return observation, reward, True, False, info
        if isinstance(action_result, dict):
            action_result["requestedAction"] = requested_action
            action_result["executedAction"] = executed_action
            action_result["preStepMaskBlockedAction"] = pre_step_mask_blocked
            action_result["fusionPolicy"] = fusion_diagnostics.get("fusion_policy")
            action_result["fusionDiagnostics"] = fusion_diagnostics
            if coach_context_payload is not None:
                human_coach_payload = dict(coach_context_payload)
                if coach_bridge_payload is not None:
                    human_coach_payload["bridgeCommand"] = dict(coach_bridge_payload)
                action_result["humanCoach"] = human_coach_payload
            if pre_step_mask_blocked:
                action_result["preStepMaskAudit"] = pre_step_audit
            attach_structured_action_result(
                pre_observation,
                action_result,
                decision=pre_step_decision,
                executed=True,
            )
        time.sleep(self.config.step_seconds)
        observation = action_result.get("observation") if isinstance(action_result, dict) else None
        if not isinstance(observation, dict):
            observation = self.observe()
        current_step_facts = self._step_facts_cache.get(
            observation,
            self.config.plant_types,
        )
        if isinstance(action_result, dict):
            audit_action = requested_action if pre_step_mask_blocked else executed_action
            if action_result.get("illegalAction") or pre_step_mask_blocked:
                action_result["actionAudit"] = self.action_legality_audit(
                    audit_action,
                    pre_observation,
                    pre_step_mask,
                    after=observation,
                    action_result=action_result,
                    facts=pre_step_facts,
                )
        self._steps_since_seed_screen_check += 1
        if is_restart_screen_observation(observation):
            override_reason, adventure_state = self._adventure_terminal_override()
            if override_reason:
                adjusted_result = dict(action_result) if isinstance(action_result, dict) else {"action": int(action)}
                adjusted_result["illegalAction"] = False
                adjusted_result["illegalReason"] = None
                return restart_terminal_return(
                    observation,
                    adjusted_result,
                    terminal_reason=override_reason,
                    done_reason="post_win_pending",
                    adventure_state=adventure_state,
                    executed=True,
                    decision=pre_step_decision,
                    previous_facts=reward_previous_facts,
                    current_facts=current_step_facts,
                    previous_legal_actions=reward_previous_legal_actions,
                    fusion_diagnostics=fusion_diagnostics,
                    pre_step_mask_blocked_action=pre_step_mask_blocked,
                )
            return restart_terminal_return(
                observation,
                action_result,
                executed=True,
                decision=pre_step_decision,
                previous_facts=reward_previous_facts,
                current_facts=current_step_facts,
                previous_legal_actions=reward_previous_legal_actions,
                fusion_diagnostics=fusion_diagnostics,
                pre_step_mask_blocked_action=pre_step_mask_blocked,
            )
        if observation.get("seedSelectionActive") or not observation.get("gameplayReady"):
            self._steps_since_seed_screen_check = interval
        safety_diagnostics = self._environment_safety_diagnostics(
            pre_observation,
            observation,
            action_result,
            requested_action,
            previous_facts=pre_step_facts,
            current_facts=current_step_facts,
        )
        for event in safety_diagnostics.get("safety_events", []) or []:
            event_name = event.get("event")
            if event_name == "suspicious_cleanup_reward_ui_during_gameplay":
                print(self._format_safety_context("[safety] blocked cleanup_reward_ui during active gameplay", observation))
            elif event_name in (
                "mower_respawn_detected",
                "cooldown_reset_detected",
                "seed_slot_object_id_changed_during_gameplay",
                "board_refresh_detected",
            ):
                print(f"[corruption] {event_name}: {event}")
        reward_events: Dict[str, Any] = {}
        terminal_reward_override: Optional[float] = None
        exceptional_reward_adjustments: Dict[str, float] = {}
        done = bool(observation.get("done", False))
        terminal_hint = str(observation.get("terminalHint") or "")
        done_reason_override: Optional[str] = None
        terminal_reason_override: Optional[str] = None
        if (
            terminal_hint == "possible_win"
            and not self._is_confirmed_post_game_ui(observation)
            and not bool(observation.get("over"))
        ):
            self._possible_win_pending_steps += 1
            done = False
            terminal_reward_override = 0.0
            append_safety_event(
                safety_diagnostics.setdefault("safety_events", []),
                "suspicious_screen_state_transition",
                observation,
                reason="possible_win_pending_confirmation",
                pending_steps=self._possible_win_pending_steps,
                last_action=requested_action,
                run_mode=self._run_mode(),
            )
            print(self._format_safety_context("[safety] delayed possible_win until post-game UI confirmation", observation))
            done_reason_override = "none"
            terminal_reason_override = ""
        else:
            self._possible_win_pending_steps = 0
        if (
            done
            and done_reason_override is None
            and not self._is_confirmed_post_game_ui(observation)
            and not is_restart_screen_observation(observation)
            and (terminal_hint == "game_over_or_loss" or bool(observation.get("over")))
        ):
            self._loss_pending_wait_steps += 1
            done = False
            done_reason_override = "none"
            terminal_reason_override = ""
            append_safety_event(
                safety_diagnostics.setdefault("safety_events", []),
                "loss_waiting_for_restart_screen",
                observation,
                pending_steps=self._loss_pending_wait_steps,
                last_action=requested_action,
                run_mode=self._run_mode(),
            )
            if self._loss_pending_wait_steps in {1, 10, 25} or self._loss_pending_wait_steps % 50 == 0:
                print(self._format_safety_context("[terminal] waiting for restart screen before reset", observation))
        elif done_reason_override not in {"none", None} or is_restart_screen_observation(observation):
            self._loss_pending_wait_steps = 0
        if bool(safety_diagnostics.get("environment_corruption_detected")):
            safety_events = list(safety_diagnostics.get("safety_events", []) or [])
            board_refresh_detected = any(
                str(event.get("event") or "") == "board_refresh_detected"
                for event in safety_events
            )
            possible_win_transition_context = bool(
                terminal_hint == "possible_win"
                or self._possible_win_pending_steps > 0
                or self._is_confirmed_possible_win(observation)
                or (pre_observation is not None and self._is_confirmed_possible_win(pre_observation))
            )
            if board_refresh_detected and possible_win_transition_context and not is_restart_screen_observation(observation):
                done = True
                done_reason_override = "win"
                terminal_reason_override = "win_transition"
                append_safety_event(
                    safety_events,
                    "win_transition_board_refresh_detected",
                    observation,
                    last_action=requested_action,
                )
                safety_diagnostics["safety_events"] = safety_events
                safety_diagnostics["environment_corruption_detected"] = False
                safety_diagnostics["environment_corruption_penalty"] = 0.0
                safety_diagnostics["env_corruption_count"] = 0
                print(
                    self._format_safety_context(
                        "[terminal] win transition detected from board refresh near possible_win",
                        observation,
                    )
                )
            else:
                penalty = float(safety_diagnostics.get("environment_corruption_penalty") or 10.0)
                exceptional_reward_adjustments["env_corruption_penalty"] = -penalty
                done = True
                done_reason_override = "env_corruption"
                terminal_reason_override = "board_state_refreshed_during_gameplay"
        reward_done_reason = (
            str(done_reason_override)
            if done_reason_override not in {None, "none", ""}
            else classify_done_reason(observation)
        )
        if done and reward_done_reason == "win":
            terminal_reward_override = float(self.config.reward.win_reward)
        elif done and reward_done_reason == "loss":
            terminal_reward_override = -float(self.config.reward.loss_penalty)
        reward_composition = self._compose_step_reward(
            reward_previous_observation,
            observation,
            action_result,
            terminal_reward_override=terminal_reward_override,
            exceptional_adjustments=exceptional_reward_adjustments,
            previous_facts=reward_previous_facts,
            current_facts=current_step_facts,
            previous_legal_actions=reward_previous_legal_actions,
        )
        reward_events.update(reward_composition.diagnostics_dict())
        reward_breakdown = reward_composition.breakdown.to_dict()
        reward = float(reward_breakdown["reward_total"])
        if isinstance(fusion_diagnostics, dict):
            fusion_diagnostics.update(self._fusion_reward_live_fields())
        truncated = False
        legal_started = time.perf_counter()
        legal = self.legal_actions(observation, facts=current_step_facts)
        legal_ms = round((time.perf_counter() - legal_started) * 1000.0, 3)
        info = {
            "action_result": action_result,
            "reward_breakdown": reward_breakdown,
            "reward_diagnostics": reward_events,
            "terminal_hint": observation.get("terminalHint"),
            "legal_actions": legal,
            "bridge_legal_actions": self.bridge_legal_actions(observation),
            "pre_step_mask_blocked_action": pre_step_mask_blocked,
            "fusion_diagnostics": fusion_diagnostics,
            **fusion_live_fields(fusion_diagnostics, self.config.fusion_policy),
            "coach_command_queue_cleared_on_reset": bool(self._coach_command_queue_cleared_on_reset),
            "pending_coach_command": None,
            "selected_bridge_command": None,
            "last_executed_coach_command_id": self._last_executed_coach_command_id,
            "startup_command_blocked": bool(self._startup_command_blocked),
            **safety_diagnostics,
        }
        if done_reason_override is not None:
            info["done_reason"] = done_reason_override
        if terminal_reason_override is not None:
            info["terminal_reason"] = terminal_reason_override
        inferred_done_reason = str(info.get("done_reason") or "") or classify_done_reason(observation)
        if inferred_done_reason == "win":
            info["done_reason"] = "win"
            info["terminal_reason"] = "win"
        mask_diag = self.mask_diagnostics(observation, facts=current_step_facts)
        info["mask_diagnostics"] = mask_diag
        info["lane_diagnostics"] = self.lane_diagnostics(
            reward_previous_observation,
            observation,
            action_result,
            legal,
            cherry_delayed_diagnostics=reward_events.get("cherry_delayed"),
            previous_facts=reward_previous_facts,
            current_facts=current_step_facts,
            previous_legal_actions=reward_previous_legal_actions,
            mask_diagnostics_snapshot=mask_diag,
        )
        if self.config.debug_performance:
            info["performance"] = {
                "step_ms": round((time.perf_counter() - step_started) * 1000.0, 3),
                "python_step_ms": round((time.perf_counter() - step_started) * 1000.0, 3),
                "legal_actions_ms": legal_ms,
                "observe_ms": observation.get("observe_ms"),
                "bridge_observe_ms": observation.get("bridge_observe_ms"),
                "screen_check_ms": observation.get("screen_check_ms"),
                "seed_probe_ms": observation.get("seed_probe_ms"),
                "ui_scan_ms": observation.get("ui_scan_ms"),
                "bridge_roundtrip_ms": observation.get("bridge_roundtrip_ms"),
                "bridge_step_ms": action_result.get("step_ms") if isinstance(action_result, dict) else None,
                "restart_detection_mode": observation.get("restartDetectionMode"),
            }
        self.previous_observation = observation
        return observation, reward, done, truncated, info

    def compute_reward(
        self,
        previous: Optional[Dict[str, Any]],
        current: Dict[str, Any],
        action_result: Optional[Dict[str, Any]] = None,
    ) -> float:
        return self.compute_reward_breakdown(previous, current, action_result)["reward_total"]

    def compute_reward_breakdown(
        self,
        previous: Optional[Dict[str, Any]],
        current: Dict[str, Any],
        action_result: Optional[Dict[str, Any]] = None,
        *,
        event_diagnostics: Optional[Dict[str, Any]] = None,
        previous_facts: Optional[StepFacts] = None,
        current_facts: Optional[StepFacts] = None,
        previous_legal_actions: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """Compatibility adapter over the pure per-observation compositor."""

        state = self._reward_state
        legal_actions = (
            list(previous_legal_actions)
            if previous_legal_actions is not None
            else self.legal_actions(previous) if isinstance(previous, dict) else []
        )
        composition = compose_environment_reward(
            previous,
            current,
            action_result,
            config=self.config.reward,
            state=state,
            plant_types=self.config.plant_types,
            fallback_rows=self.config.row_count,
            fallback_columns=self.config.column_count,
            previous_legal_actions=legal_actions,
            previous_facts=previous_facts,
            current_facts=current_facts,
        )
        self._reward_state = composition.state
        if event_diagnostics is not None:
            event_diagnostics.update(composition.diagnostics_dict())
        return composition.breakdown.to_dict()

    def _compose_step_reward(
        self,
        previous: Optional[Dict[str, Any]],
        current: Dict[str, Any],
        action_result: Optional[Dict[str, Any]],
        *,
        terminal_reward_override: Optional[float] = None,
        exceptional_adjustments: Optional[Dict[str, float]] = None,
        previous_facts: Optional[StepFacts] = None,
        current_facts: Optional[StepFacts] = None,
        previous_legal_actions: Optional[List[int]] = None,
    ) -> RewardComposition:
        state = self._reward_state
        legal_actions = (
            list(previous_legal_actions)
            if previous_legal_actions is not None
            else self.legal_actions(previous) if isinstance(previous, dict) else []
        )
        composition = compose_step_reward(
            previous,
            current,
            action_result,
            config=self.config.reward,
            state=state,
            plant_types=self.config.plant_types,
            fallback_rows=self.config.row_count,
            fallback_columns=self.config.column_count,
            previous_legal_actions=legal_actions,
            previous_facts=previous_facts,
            current_facts=current_facts,
            enable_recipe_discovery_reward=bool(
                self.config.enable_recipe_discovery_reward
            ),
            enable_fusion_chain_rewards=bool(self.config.enable_fusion_chain_rewards),
            enable_repeat_recipe_decay=bool(self.config.enable_repeat_recipe_decay),
            terminal_reward_override=terminal_reward_override,
            exceptional_adjustments=exceptional_adjustments,
        )
        self._reward_state = composition.state
        if isinstance(action_result, dict):
            action_result.update(composition.annotations_dict())
        return composition

    def lane_diagnostics(
        self,
        previous: Optional[Dict[str, Any]],
        current: Dict[str, Any],
        action_result: Optional[Dict[str, Any]],
        legal_actions: Optional[List[int]] = None,
        *,
        cherry_delayed_diagnostics: Optional[Dict[str, Any]] = None,
        previous_facts: Optional[StepFacts] = None,
        current_facts: Optional[StepFacts] = None,
        previous_legal_actions: Optional[List[int]] = None,
        mask_diagnostics_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        previous_observation = previous or current
        prior_snapshot = previous_facts or build_step_facts(
            previous_observation,
            self.config.plant_types,
        )
        next_snapshot = current_facts or (
            prior_snapshot
            if previous_observation is current
            else build_step_facts(current, self.config.plant_types)
        )
        prior_legal = (
            list(previous_legal_actions)
            if previous_legal_actions is not None
            else self.legal_actions(previous_observation, facts=prior_snapshot)
        )
        mask_diagnostics = (
            mask_diagnostics_snapshot
            if mask_diagnostics_snapshot is not None
            else self.mask_diagnostics(
                current,
                self.action_mask(current, facts=next_snapshot),
                facts=next_snapshot,
            )
        )
        return compose_lane_diagnostics(
            LaneDiagnosticsInput(
                previous_facts=prior_snapshot,
                current_facts=next_snapshot,
                post_reward_state=self._reward_state,
                current_legal_actions=tuple(legal_actions or ()),
                previous_legal_actions=tuple(prior_legal),
                action_result=action_result or {},
                mask_diagnostics=mask_diagnostics,
                cherry_delayed_diagnostics=cherry_delayed_diagnostics or {},
                fallback_rows=self.config.row_count,
                fallback_columns=self.config.column_count,
                close_threat_threshold=self.config.reward.close_threat_threshold,
            )
        )
    def _action_validation_config(
        self,
        observation: Dict[str, Any],
        *,
        facts: Optional[StepFacts] = None,
        include_tactical_masks: bool = True,
    ) -> ActionValidationConfig:
        snapshot = facts or self._step_facts_cache.get(
            observation,
            self.config.plant_types,
        )
        rows = int(snapshot.rows or self.config.row_count)
        cols = int(snapshot.columns or self.config.column_count)
        if not is_supported_live_board_geometry(rows, cols):
            raise ValueError(
                "Adventure Generalist supports only 5x10 or 6x10 live boards: "
                f"row_count={rows}, column_count={cols}"
            )
        slots = snapshot.seed_slots
        if len(slots) > ADVENTURE_IDENTITY_MAX_SEED_SLOTS:
            raise ValueError(
                "Adventure Generalist observation exceeds 14 seed slots: "
                f"seed_slots={len(slots)}"
            )
        if len(self.config.plant_types) > ADVENTURE_IDENTITY_MAX_SEED_SLOTS:
            raise ValueError(
                "Adventure Generalist configuration exceeds 14 plant identities: "
                f"plant_types={len(self.config.plant_types)}"
            )
        return ActionValidationConfig(
            action_space_mode=ACTION_SPACE_ADVENTURE_14_IDENTITY,
            plant_types=tuple(int(value) for value in self.config.plant_types),
            max_seed_slots=ADVENTURE_IDENTITY_MAX_SEED_SLOTS,
            # Fixed model geometry; ActionValidationContext retains the live
            # dimensions for bounds checks and masking.
            rows=self.config.row_count,
            cols=self.config.column_count,
            fusion_action_mask_enabled=self._fusion_action_mask_enabled(),
            tactical_masks=bool(
                include_tactical_masks and getattr(self.config, "tactical_masks", False)
            ),
            wallnut_tactical_mask=bool(
                include_tactical_masks and getattr(self.config, "wallnut_tactical_mask", False)
            ),
            cherrybomb_tactical_mask=bool(
                include_tactical_masks
                and getattr(self.config, "cherrybomb_tactical_mask", False)
            ),
            fusion_compatible_pairs=compatible_pairs_for_observation(
                observation,
                self.config.plant_types,
                are_fusion_compatible,
                facts=snapshot,
            ),
            compatibility_version="relevant_observation_pairs_v1",
        )

    def _tactical_action_rejections(
        self,
        observation: Dict[str, Any],
        action_count: int,
        *,
        facts: Optional[StepFacts] = None,
        include_tactical_masks: bool = True,
    ) -> Dict[int, str]:
        if not include_tactical_masks or not self._tactical_masks_enabled():
            return {}
        tactical_enabled = bool(getattr(self.config, "tactical_masks", False))
        wallnut_enabled = tactical_enabled or bool(getattr(self.config, "wallnut_tactical_mask", False))
        cherry_enabled = tactical_enabled or bool(getattr(self.config, "cherrybomb_tactical_mask", False))
        snapshot = self._facts_snapshot(observation, facts)
        rows = int(observation.get("rowCount") or 0)
        columns = int(observation.get("columnCount") or 0)
        if rows <= 0 or columns <= 0:
            return {}
        cells = CELLS_PER_SLOT
        rejections: Dict[int, str] = {}
        for action_id in range(1, max(0, int(action_count))):
            encoded = action_id - 1
            slot_index = encoded // cells
            cell = encoded % cells
            plant_type = (
                int(snapshot.seed_slots[slot_index].plant_type)
                if 0 <= slot_index < len(snapshot.seed_slots)
                else -1
            )
            row, column = divmod(cell, DEFAULT_COLS)
            if row >= rows or column >= columns:
                continue
            if plant_type == 3 and wallnut_enabled:
                allowed, reason = self._wallnut_tactical_action_allowed(
                    observation,
                    row,
                    column,
                    facts=snapshot,
                )
                if not allowed:
                    rejections[action_id] = f"tactical_wallnut_{reason}"
            elif plant_type == 2 and cherry_enabled:
                allowed, reason = self._cherrybomb_tactical_action_allowed(
                    observation,
                    row,
                    column,
                    facts=snapshot,
                )
                if not allowed:
                    rejections[action_id] = f"tactical_cherrybomb_{reason}"
        return rejections

    def _action_cache_for(
        self,
        observation: Dict[str, Any],
        *,
        bridge_actions: Optional[List[int]] = None,
        facts: Optional[StepFacts] = None,
        include_tactical_masks: bool = True,
    ) -> ActionDecisionCache:
        resolved_bridge_actions = [
            int(action)
            for action in (
                bridge_actions if bridge_actions is not None else self.bridge_legal_actions(observation)
            )
        ]
        snapshot = facts or self._step_facts_cache.get(
            observation,
            self.config.plant_types,
        )
        validation_config = self._action_validation_config(
            observation,
            facts=snapshot,
            include_tactical_masks=include_tactical_masks,
        )
        action_count = validation_config.spec.action_count
        restart_screen = is_restart_screen_observation(observation)
        context = build_action_validation_context(
            observation,
            config=validation_config,
            bridge_legal_actions=resolved_bridge_actions,
            restart_screen=restart_screen,
            facts=snapshot,
        )
        cached = self._action_decision_cache
        if cached is not None and cached.key == context.cache_key:
            self._action_decision_cache_hits += 1
            self._action_decision_cache_last_hit = True
            return cached
        tactical_rejections = self._tactical_action_rejections(
            observation,
            action_count,
            facts=snapshot,
            include_tactical_masks=include_tactical_masks,
        )
        if tactical_rejections:
            tactical = tuple(sorted((int(action), str(reason)) for action, reason in tactical_rejections.items()))
            context = replace(
                context,
                tactical_rejections=tactical,
                tactical_rejection_by_action=dict(tactical),
            )
        cache = build_action_decision_cache(
            observation,
            config=validation_config,
            bridge_legal_actions=resolved_bridge_actions,
            restart_screen=restart_screen,
            tactical_rejections=tactical_rejections,
            source="mask",
            context=context,
        )
        self._action_decision_cache = cache
        self._action_decision_cache_misses += 1
        self._action_decision_cache_last_hit = False
        return cache

    def action_cache_diagnostics(self) -> Dict[str, Any]:
        cache = self._action_decision_cache
        return {
            "hits": int(self._action_decision_cache_hits),
            "misses": int(self._action_decision_cache_misses),
            "last_hit": bool(self._action_decision_cache_last_hit),
            "key": cache.key.to_dict() if cache is not None else None,
            "action_count": len(cache.decisions) if cache is not None else 0,
            "legal_action_count": (
                sum(1 for allowed in cache.mask if allowed) if cache is not None else 0
            ),
        }

    def action_decision(
        self,
        action: int,
        observation: Optional[Dict[str, Any]] = None,
        *,
        source: str = "execution_safeguard",
        source_metadata: Optional[Dict[str, Any]] = None,
        bridge_command: Optional[Dict[str, Any]] = None,
        bridge_actions: Optional[List[int]] = None,
        intent: Optional[ActionIntent] = None,
        facts: Optional[StepFacts] = None,
        include_tactical_masks: bool = True,
    ) -> ActionDecision:
        obs = observation or self.previous_observation or self.observe()
        cache = self._action_cache_for(
            obs,
            bridge_actions=bridge_actions,
            facts=facts,
            include_tactical_masks=include_tactical_masks,
        )
        action_id = int(action)
        resolved_intent = intent
        if resolved_intent is None:
            resolved_intent = build_action_intent(
                action_id,
                source=source,
                mode=cache.context.config.action_space_mode,
                observation=obs,
                plant_types=cache.context.config.plant_types,
                max_seed_slots=cache.context.config.max_seed_slots,
                rows=cache.context.config.rows,
                cols=cache.context.config.cols,
                bridge_command=bridge_command,
                source_metadata=source_metadata,
            )
        if resolved_intent.bridge_action != action_id:
            return ActionDecision(
                intent=resolved_intent,
                legal=False,
                rejection_reason="action_identity_mismatch",
                frame_identity=cache.context.frame_identity.token,
                config_fingerprint=cache.context.config_fingerprint,
                resolved_action_kind=resolved_intent.action_kind,
                required_sun=0,
                bridge_authoritative=True,
                cache_reused=False,
            )
        cached_decision = cache.decision_for(
            action_id,
            intent=resolved_intent,
            cache_reused=True,
        )
        if cached_decision is not None:
            return cached_decision
        return validate_action_intent(resolved_intent, cache.context)

    def action_mask(
        self,
        observation: Optional[Dict[str, Any]] = None,
        *,
        facts: Optional[StepFacts] = None,
        include_tactical_masks: bool = True,
    ) -> List[int]:
        obs = observation or self.previous_observation or self.observe()
        cache = self._action_cache_for(
            obs,
            facts=facts,
            include_tactical_masks=include_tactical_masks,
        )
        return [1 if allowed else 0 for allowed in cache.mask]

    def mask_diagnostics(
        self,
        observation: Optional[Dict[str, Any]] = None,
        mask: Optional[List[int]] = None,
        *,
        facts: Optional[StepFacts] = None,
    ) -> Dict[str, Any]:
        obs = observation or self.previous_observation or self.observe()
        bridge_actions = self.bridge_legal_actions(obs)
        cache = self._action_cache_for(
            obs,
            bridge_actions=bridge_actions,
            facts=facts,
        )
        snapshot = cache.context.facts
        current_mask = mask if mask is not None else [1 if allowed else 0 for allowed in cache.mask]
        blocked_counts: Counter[str] = Counter()
        for action in bridge_actions:
            action_id = int(action)
            if action_id <= 0:
                continue
            decision = (
                cache.decisions[action_id]
                if 0 <= action_id < len(cache.decisions)
                else None
            )
            if decision is not None and not decision.legal:
                blocked_counts[decision.rejection_reason or "filtered"] += 1
        tactical_diag: Dict[str, Any] = {}
        if self._tactical_masks_enabled():
            tactical_enabled = bool(getattr(self.config, "tactical_masks", False))
            wallnut_enabled = tactical_enabled or bool(getattr(self.config, "wallnut_tactical_mask", False))
            cherry_enabled = tactical_enabled or bool(getattr(self.config, "cherrybomb_tactical_mask", False))
            tactical_counts: Counter[str] = Counter()
            wallnut_masked = 0
            cherrybomb_masked = 0
            wallnut_available = 0
            cherrybomb_available = 0
            for action in bridge_actions:
                action_id = int(action)
                if action_id <= 0:
                    continue
                decision = (
                    cache.decisions[action_id]
                    if 0 <= action_id < len(cache.decisions)
                    else None
                )
                if decision is None:
                    continue
                reason = str(decision.rejection_reason or "")
                if reason.startswith("tactical_wallnut_"):
                    wallnut_masked += 1
                    tactical_counts[reason] += 1
                elif reason.startswith("tactical_cherrybomb_"):
                    cherrybomb_masked += 1
                    tactical_counts[reason] += 1
                elif decision.legal and decision.selected_plant_type == 3:
                    wallnut_available += 1
                elif decision.legal and decision.selected_plant_type == 2:
                    cherrybomb_available += 1
            tactical_diag = {
                "tactical_mask_enabled": bool(wallnut_enabled or cherry_enabled),
                "wallnut_tactical_mask_enabled": bool(wallnut_enabled),
                "cherrybomb_tactical_mask_enabled": bool(cherry_enabled),
                "wallnut_actions_masked": int(wallnut_masked),
                "cherrybomb_actions_masked": int(cherrybomb_masked),
                "wallnut_actions_available": int(wallnut_available),
                "cherrybomb_actions_available": int(cherrybomb_available),
                "mask_all_but_wait_count": 1 if sum(1 for value in current_mask if value) <= 1 else 0,
                "tactical_mask_block_reason_counts": dict(sorted(tactical_counts.items())),
            }
        active_slot_diagnostics = self._active_seed_slot_mask_diagnostics(
            obs,
            cache,
            facts=snapshot,
        )
        total_legal_action_count = sum(1 for allowed in current_mask if allowed)
        return {
            "legal_actions_by_seed_slot": self._legal_actions_by_seed_slot(
                obs,
                current_mask,
                facts=snapshot,
            ),
            "bridge_legal_actions_by_seed_slot": self._legal_actions_by_seed_slot(
                obs,
                bridge_actions,
                facts=snapshot,
            ),
            "python_mask_block_reason_counts": dict(sorted(blocked_counts.items())),
            "python_legal_action_count": total_legal_action_count,
            "bridge_legal_action_count": len(bridge_actions),
            "total_legal_action_count": total_legal_action_count,
            "wait_legal": bool(current_mask[0]) if current_mask else False,
            "active_seed_slot_count": len(snapshot.seed_slots),
            "active_seed_slots": active_slot_diagnostics,
            "slot_readiness_by_seed_slot": self._slot_readiness_by_seed_slot(
                obs,
                facts=snapshot,
            ),
            "action_cache": self.action_cache_diagnostics(),
            **self._fusion_mask_diagnostics(obs, facts=snapshot, cache=cache),
            **tactical_diag,
        }

    def _tactical_masks_enabled(self) -> bool:
        return bool(
            getattr(self.config, "tactical_masks", False)
            or getattr(self.config, "wallnut_tactical_mask", False)
            or getattr(self.config, "cherrybomb_tactical_mask", False)
        )

    def _wallnut_tactical_action_allowed(
        self,
        observation: Dict[str, Any],
        row: int,
        column: int,
        *,
        facts: Optional[StepFacts] = None,
    ) -> Tuple[bool, str]:
        if row < 0 or column < 0:
            return False, "invalid_cell"
        snapshot = self._facts_snapshot(observation, facts)
        rows = max(0, int(snapshot.rows or self.config.row_count))
        threat_rows = set(active_threat_rows_from_facts(snapshot, rows))
        if row not in threat_rows:
            return False, "no_row_threat"
        nearest_x = nearest_zombie_x_by_row_from_facts(snapshot, rows).get(row)
        if nearest_x is not None and float(nearest_x) < float(column) - 0.5:
            return False, "behind_zombie"
        if wallnut_blocker_count_from_facts(snapshot, row) > 0:
            return False, "duplicate_blocker"
        mower_risk = row in mower_risk_rows_from_facts(snapshot, rows)
        if column <= 2 and not mower_risk:
            return False, "too_far_back"
        valuable_behind = any(
            col < column
            for col in valuable_plant_columns_from_facts(snapshot, row)
        )
        shooters = plant_counts_by_row_from_facts(snapshot, rows, 0).get(row, 0)
        if mower_risk:
            return True, "mower_risk"
        if valuable_behind and (nearest_x is None or float(nearest_x) - float(column) <= 5.0):
            return True, "frontline"
        if shooters > 0 and nearest_x is not None and float(nearest_x) - float(column) <= 4.0:
            return True, "shooter_screen"
        return False, "non_blocking"

    def _cherrybomb_tactical_action_allowed(
        self,
        observation: Dict[str, Any],
        row: int,
        column: int,
        *,
        facts: Optional[StepFacts] = None,
    ) -> Tuple[bool, str]:
        if row < 0 or column < 0:
            return False, "invalid_cell"
        snapshot = self._facts_snapshot(observation, facts)
        rows = max(0, int(snapshot.rows or self.config.row_count))
        nearby = nearby_zombie_context_from_facts(
            snapshot,
            row,
            column,
            radius=2.75,
        )
        if int(nearby.get("zombies", 0)) >= 2:
            return True, "cluster"
        if int(nearby.get("tough", 0)) > 0 or int(nearby.get("buckethead", 0)) > 0 or int(nearby.get("conehead", 0)) > 0:
            return True, "heavy"
        if row in mower_risk_rows_from_facts(snapshot, rows):
            return True, "mower_risk"
        danger = lane_danger_by_row_from_facts(snapshot, rows).get(row, 0.0)
        if danger >= 0.65 and int(nearby.get("zombies", 0)) > 0:
            return True, "high_danger"
        return False, "low_value"
    def action_legality_audit(
        self,
        action: int,
        before: Dict[str, Any],
        mask: Optional[List[int]] = None,
        after: Optional[Dict[str, Any]] = None,
        action_result: Optional[Dict[str, Any]] = None,
        *,
        facts: Optional[StepFacts] = None,
    ) -> Dict[str, Any]:
        action_id = int(action)
        bridge_actions = self.bridge_legal_actions(before)
        cache = self._action_cache_for(
            before,
            bridge_actions=bridge_actions,
            facts=facts,
        )
        current_mask = mask if mask is not None else [1 if allowed else 0 for allowed in cache.mask]
        decision = cache.decision_for(action_id)
        if decision is None:
            decision = self.action_decision(
                action_id,
                before,
                source="diagnostic",
                bridge_actions=bridge_actions,
                facts=facts,
            )
        decoded = dict(decision.intent.decoded_action)
        slot_index = int(decoded.get("slot_index", -1))
        slot_before = self._seed_slot_snapshot(before, slot_index)
        slot_after = self._seed_slot_snapshot(after or {}, slot_index)
        allowed = bool(decision.legal)
        filter_reason = str(decision.rejection_reason or "")
        plant_type = int(decoded.get("plant_type", -1))
        return {
            "action": action_id,
            "decoded": decoded,
            "slotIndex": slot_index,
            "plantType": plant_type,
            "plantTypeName": plant_type_name(plant_type) if plant_type >= 0 else "none",
            "row": int(decoded.get("row", -1)),
            "column": int(decoded.get("column", -1)),
            "pythonMaskValueBefore": bool(0 <= action_id < len(current_mask) and current_mask[action_id]),
            "pythonFilterValueBefore": bool(allowed),
            "pythonFilterReason": filter_reason,
            "bridgeLegalActionsValueBefore": action_id in set(bridge_actions),
            "bridgeLegalActionCountBefore": len(bridge_actions),
            "slotBefore": slot_before,
            "slotAfter": slot_after,
            "sunBefore": int(before.get("sun", 0) or 0),
            "sunAfter": int((after or {}).get("sun", before.get("sun", 0)) or 0),
            "gameplayReadyBefore": bool(before.get("gameplayReady")),
            "boardReadableBefore": bool(before.get("boardFound")) and bool(before.get("canReadBoard", True)),
            "seedSelectionActiveBefore": bool(before.get("seedSelectionActive")),
            "illegalReason": str(action_result.get("illegalReason") or "") if isinstance(action_result, dict) else "",
            "legalActionsBySeedSlotBefore": self._legal_actions_by_seed_slot(before, current_mask),
            "bridgeLegalActionsBySeedSlotBefore": self._legal_actions_by_seed_slot(before, bridge_actions),
            "actionIntent": decision.intent.to_dict(),
            "actionDecision": decision.to_dict(),
            "actionCacheKey": cache.key.to_dict(),
        }

    def _fusion_action_mask_enabled(self) -> bool:
        return bool(getattr(self.config, "fusion_action_mask_enabled", False))

    def _fusion_mask_diagnostics(
        self,
        observation: Dict[str, Any],
        *,
        facts: Optional[StepFacts] = None,
        cache: Optional[ActionDecisionCache] = None,
    ) -> Dict[str, Any]:
        """Classify occupied-tile/seed pairings into fusion mask diagnostics counts."""

        enabled = self._fusion_action_mask_enabled()
        out: Dict[str, Any] = {
            "fusion_action_mask_enabled": bool(enabled),
            "fusion_actions_available_count": 0,
            "fusion_candidate_tiles": [],
            "fusion_actions_masked_empty_tile": 0,
            "fusion_actions_masked_incompatible_count": 0,
            "fusion_actions_masked_disabled_count": 0,
            "fusion_actions_masked_cooldown_count": 0,
            "fusion_actions_masked_sun_count": 0,
            "fusion_compatibility_table": fusion_compatibility_table(),
        }
        rows = int(observation.get("rowCount") or 0)
        cols = int(observation.get("columnCount") or 0)
        if rows <= 0 or cols <= 0:
            return out
        snapshot = self._facts_snapshot(observation, facts)
        decision_cache = cache or self._action_cache_for(
            observation,
            facts=snapshot,
        )
        slots = snapshot.seed_slots
        occupied_cells = {
            cell: plant.plant_type
            for cell, plant in snapshot.last_primary_plant_by_cell.items()
            if 0 <= cell[0] < rows and 0 <= cell[1] < cols
        }
        available_tiles: set = set()
        for (prow, pcol), _existing in occupied_cells.items():
            for slot_index, slot in enumerate(slots):
                seed_slot_index = int(slot.slot_index if slot.slot_index >= 0 else slot_index)
                action_id = (
                    1
                    + seed_slot_index * CELLS_PER_SLOT
                    + prow * DEFAULT_COLS
                    + pcol
                )
                decision = (
                    decision_cache.decisions[action_id]
                    if 0 <= action_id < len(decision_cache.decisions)
                    else None
                )
                # Always classify as if fusion were enabled so the disabled case
                # can report how many compatible fusions are being suppressed.
                reason = get_fusion_illegal_reason(
                    observation,
                    prow,
                    pcol,
                    seed_slot_index,
                    fusion_enabled=True,
                    plant_types=self.config.plant_types,
                    facts=snapshot,
                )
                runtime_authorized = action_id in decision_cache.context.bridge_legal_actions
                if enabled and decision is not None and decision.legal and decision.resolved_action_kind == "fusion":
                    out["fusion_actions_available_count"] += 1
                    available_tiles.add((prow, pcol))
                elif reason == "incompatible_pair" and not runtime_authorized:
                    out["fusion_actions_masked_incompatible_count"] += 1
                elif reason == "cooldown_not_ready":
                    out["fusion_actions_masked_cooldown_count"] += 1
                elif reason in ("insufficient_sun",):
                    out["fusion_actions_masked_sun_count"] += 1
                elif reason == "":
                    if enabled:
                        if decision is not None and decision.legal:
                            out["fusion_actions_available_count"] += 1
                            available_tiles.add((prow, pcol))
                    else:
                        out["fusion_actions_masked_disabled_count"] += 1
                elif not enabled and runtime_authorized:
                    out["fusion_actions_masked_disabled_count"] += 1
        out["fusion_candidate_tiles"] = sorted(available_tiles)
        return out
    def _legal_actions_by_seed_slot(
        self,
        observation: Dict[str, Any],
        actions_or_mask: Any,
        *,
        facts: Optional[StepFacts] = None,
    ) -> Dict[str, int]:
        snapshot = self._facts_snapshot(observation, facts)
        counts = {str(index): 0 for index in range(len(snapshot.seed_slots))}
        if (
            isinstance(actions_or_mask, list)
            and len(actions_or_mask) == ADVENTURE_IDENTITY_ACTION_COUNT
            and all(isinstance(item, bool) or int(item) in (0, 1) for item in actions_or_mask)
        ):
            actions = [index for index, allowed in enumerate(actions_or_mask) if bool(allowed)]
        elif isinstance(actions_or_mask, list):
            actions = [int(action) for action in actions_or_mask]
        else:
            actions = []
        rows = int(observation.get("rowCount") or 0)
        columns = int(observation.get("columnCount") or 0)
        if rows <= 0 or columns <= 0:
            return counts
        for action in actions:
            if int(action) <= 0:
                continue
            slot_index = (int(action) - 1) // CELLS_PER_SLOT
            if str(slot_index) in counts:
                counts[str(slot_index)] += 1
        return counts

    def _active_seed_slot_mask_diagnostics(
        self,
        observation: Dict[str, Any],
        cache: ActionDecisionCache,
        *,
        facts: Optional[StepFacts] = None,
    ) -> List[Dict[str, Any]]:
        """Explain every active runtime slot against the canonical mask."""

        snapshot = self._facts_snapshot(observation, facts)
        rows = int(cache.context.rows)
        columns = int(cache.context.cols)
        cells = CELLS_PER_SLOT
        sun = int(cache.context.sun)
        diagnostics: List[Dict[str, Any]] = []
        for fallback_index, slot in enumerate(snapshot.seed_slots):
            slot_index = int(slot.slot_index if slot.slot_index >= 0 else fallback_index)
            reasons: Counter[str] = Counter()
            legal_cells = 0
            bridge_legal_cells = 0
            if cells > 0:
                first_action = 1 + slot_index * cells
                for action_id in range(first_action, first_action + cells):
                    if action_id in cache.context.bridge_legal_actions:
                        bridge_legal_cells += 1
                    if not 0 <= action_id < len(cache.decisions):
                        reasons["action_out_of_range"] += 1
                        continue
                    decision = cache.decisions[action_id]
                    if decision.legal:
                        legal_cells += 1
                    else:
                        reasons[str(decision.rejection_reason or "filtered")] += 1
            slot_snapshot = self._seed_slot_snapshot(
                observation,
                slot_index,
                facts=snapshot,
            )
            tactical_masked = sum(
                count for reason, count in reasons.items() if reason.startswith("tactical_")
            )
            other_masked = sum(
                count
                for reason, count in reasons.items()
                if reason
                not in {
                    "cooldown_not_ready",
                    "insufficient_sun",
                    "bridge_legal_actions_missing",
                    "occupied_cell",
                }
                and not reason.startswith("tactical_")
            )
            dominant_reason = ""
            if legal_cells == 0 and reasons:
                dominant_reason = reasons.most_common(1)[0][0]
            diagnostics.append(
                {
                    "slot_index": slot_index,
                    "plant_type": int(slot.plant_type),
                    "plant_identity": slot.plant_type_name
                    or (plant_type_name(slot.plant_type) if slot.plant_type >= 0 else "unknown"),
                    "ready": bool(slot.ready),
                    "usable": bool(slot.usable and not slot.disabled),
                    "disabled": bool(slot.disabled),
                    "seed_cost": int(slot.seed_cost),
                    "cost_source": str(slot_snapshot.get("source") or ""),
                    "sun": sun,
                    "affordable": bool(sun >= int(slot.seed_cost)),
                    "legal_cells": int(legal_cells),
                    "bridge_legal_cells": int(bridge_legal_cells),
                    "masked_by_cooldown": int(reasons.get("cooldown_not_ready", 0)),
                    "masked_by_sun": int(reasons.get("insufficient_sun", 0)),
                    "masked_by_tactical_masks": int(tactical_masked),
                    "masked_by_bridge_legality": int(reasons.get("bridge_legal_actions_missing", 0)),
                    "masked_by_occupancy": int(reasons.get("occupied_cell", 0)),
                    "masked_by_other": int(other_masked),
                    "total_legal_actions": int(legal_cells),
                    "mask_reason_counts": dict(sorted(reasons.items())),
                    "reason": dominant_reason,
                }
            )
        return diagnostics

    def _slot_readiness_by_seed_slot(
        self,
        observation: Dict[str, Any],
        *,
        facts: Optional[StepFacts] = None,
    ) -> Dict[str, Dict[str, Any]]:
        snapshot = self._facts_snapshot(observation, facts)
        result: Dict[str, Dict[str, Any]] = {}
        for index, slot in enumerate(snapshot.seed_slots):
            slot_index = int(slot.slot_index if slot.slot_index >= 0 else index)
            result[str(slot_index)] = self._seed_slot_snapshot(
                observation,
                slot_index,
                facts=snapshot,
            )
        return result
    def rule_based_teacher_action(self, observation: Optional[Dict[str, Any]] = None) -> int:
        obs = observation or self.previous_observation or self.observe()
        legal = set(self.legal_actions(obs))
        if not legal:
            return 0

        rows = int(obs.get("rowCount") or self.config.row_count)
        cols = int(obs.get("columnCount") or self.config.column_count)
        lanes = obs.get("lanes", [])
        plants = obs.get("plants", [])
        slots = seed_slots_from_observation(obs, self.config.plant_types)

        def ready_slots(plant_type: int) -> List[Dict[str, Any]]:
            sun = int(obs.get("sun", 0))
            return sorted(
                (
                    slot for slot in slots
                    if int(slot.get("plantType", -999)) == int(plant_type)
                    and bool(slot.get("usable", slot.get("ready", False)))
                    and bool(slot.get("ready", False))
                    and sun >= int(slot.get("seedCost", 0))
                ),
                key=lambda slot: (int(slot.get("seedCost", 0)), int(slot.get("slotIndex", 0))),
            )

        sunflower_type = 1
        sunflower_slots = ready_slots(sunflower_type)
        if sunflower_slots and len(plants) < max(2, rows):
            action = self._encode_action(int(sunflower_slots[0].get("slotIndex", 0)), min(rows - 1, len(plants) % rows), 0, rows, cols)
            if action in legal:
                return action

        threatened = sorted(
            (lane for lane in lanes if lane.get("zombieCount", 0) > 0),
            key=lambda lane: lane.get("nearestZombieX") if lane.get("nearestZombieX") is not None else 9999,
        )
        peashooter_type = 0
        wallnut_type = 3
        for lane in threatened:
            row = int(lane.get("row", 0))
            nearest_x = lane.get("nearestZombieX")
            plant_type = wallnut_type if nearest_x is not None and nearest_x <= 2.0 else peashooter_type
            plant_slots = ready_slots(plant_type)
            if not plant_slots:
                continue
            for col in range(cols):
                action = self._encode_action(int(plant_slots[0].get("slotIndex", 0)), row, col, rows, cols)
                if action in legal:
                    return action

        return 0 if 0 in legal else min(legal)

    def restore_game_speed(self) -> Dict[str, Any]:
        return self.client.request("restore_game_speed")

    def ensure_gameplay_speed(
        self,
        observation: Dict[str, Any],
        *,
        timeout: float = 1.0,
        poll_seconds: float = 0.05,
    ) -> Dict[str, Any]:
        """Verify configured speed once at a canonical gameplay boundary.

        The bridge continuously corrects speed drift. This boundary check makes
        a seed-to-gameplay scene reset observable and re-sends configuration
        only when the returned gameplay frame proves a mismatch.
        """

        if not isinstance(observation, dict) or not bool(observation.get("gameplayReady")):
            return observation
        if not any(
            key in observation
            for key in ("requestedGameSpeed", "effectiveGameSpeed", "gameSpeed")
        ):
            # Bridge-free fixtures predate speed diagnostics. Do not invent a
            # live validation result when the fields are genuinely unavailable.
            return observation

        desired = max(0.01, float(self.config.game_speed))
        mode = str(self.config.game_speed_mode or "game_speed").strip().lower()

        def speed_value(payload: Dict[str, Any], key: str) -> Optional[float]:
            try:
                value = payload.get(key)
                return None if value is None else float(value)
            except (TypeError, ValueError):
                return None

        def matches(payload: Dict[str, Any]) -> bool:
            requested = speed_value(payload, "requestedGameSpeed")
            effective = speed_value(payload, "effectiveGameSpeed")
            unity_time_scale = speed_value(payload, "unityTimeScale")
            return bool(
                requested is not None
                and effective is not None
                and unity_time_scale is not None
                and abs(requested - desired) <= 0.001
                and abs(effective - desired) <= 0.001
                and abs(unity_time_scale - desired) <= 0.001
            )

        current = observation
        reapplied = False
        effective_before = speed_value(current, "effectiveGameSpeed")
        time_scale_before = speed_value(current, "unityTimeScale")
        print(
            "[game-speed] gameplay entered "
            f"desired={desired:.1f} "
            f"effective_before={effective_before if effective_before is not None else 'unknown'} "
            f"time_scale_before={time_scale_before if time_scale_before is not None else 'unknown'} "
            f"screenState={current.get('screenState')} "
            f"gameplayReady={bool(current.get('gameplayReady'))}"
        )
        if mode != "safe" and not matches(current):
            self.configure()
            reapplied = True
            deadline = time.monotonic() + max(0.0, float(timeout))
            while True:
                current = self.observe()
                if matches(current) or time.monotonic() >= deadline:
                    break
                time.sleep(max(0.0, float(poll_seconds)))

        effective = speed_value(current, "effectiveGameSpeed")
        requested = speed_value(current, "requestedGameSpeed")
        unity_time_scale = speed_value(current, "unityTimeScale")
        applied = f"{desired:.1f}" if reapplied else "not_needed"
        print(
            "[game-speed] "
            f"applied={applied} "
            f"desired={desired:.1f} "
            f"requested={requested if requested is not None else 'unknown'} "
            f"effective_after={effective if effective is not None else 'unknown'} "
            f"time_scale_after={unity_time_scale if unity_time_scale is not None else 'unknown'} "
            f"mode={mode} screenState={current.get('screenState')} "
            f"gameplayReady={bool(current.get('gameplayReady'))} reapplied={reapplied}"
        )
        if mode != "safe" and not matches(current):
            raise RuntimeError(
                "gameplay_speed_not_applied: "
                f"desired={desired:.3f} requested={requested} effective={effective} "
                f"unityTimeScale={unity_time_scale} mode={mode} "
                f"screenState={current.get('screenState')}"
            )
        return current

    def close(self) -> None:
        try:
            self.restore_game_speed()
        except Exception:
            pass
        self.client.close()

    def _encode_action(self, seed_slot_index: int, row: int, column: int, rows: int, cols: int) -> int:
        if seed_slot_index < 0:
            return 0
        del rows, cols
        return 1 + seed_slot_index * CELLS_PER_SLOT + row * DEFAULT_COLS + column

    def _environment_safety_diagnostics(
        self,
        previous: Optional[Dict[str, Any]],
        current: Dict[str, Any],
        action_result: Optional[Dict[str, Any]],
        requested_action: int,
        *,
        previous_facts: Optional[StepFacts] = None,
        current_facts: Optional[StepFacts] = None,
    ) -> Dict[str, Any]:
        del action_result
        baseline_was_ready = bool(self._episode_mower_baseline_ready)
        result = compose_environment_safety_diagnostics(
            previous,
            current,
            requested_action=int(requested_action),
            fallback_plant_types=self.config.plant_types,
            fallback_row_count=self.config.row_count,
            lost_mower_rows=self._episode_lost_mower_rows,
            missing_mower_rows=self._episode_missing_mower_rows,
            mower_baseline_ready=self._episode_mower_baseline_ready,
            live_board_progress=self._has_live_board_progress(current),
            post_win_signal_present=self._post_win_signal_present(current),
            cleanup_signal_active=bool(
                current.get("nextStep") == "cleanup_reward_ui"
                or self._cleanup_signal_active(current)
            ),
            suspicious_cleanup_signal_during_gameplay=(
                self._suspicious_cleanup_signal_during_gameplay(current)
            ),
            previous_confirmed_postgame=bool(
                previous and self._is_confirmed_post_game_ui(previous)
            ),
            current_confirmed_postgame=self._is_confirmed_post_game_ui(current),
            previous_facts=previous_facts,
            current_facts=current_facts,
        )
        self._episode_lost_mower_rows = set(result.next_lost_mower_rows)
        self._episode_missing_mower_rows = set(result.next_missing_mower_rows)
        self._episode_mower_baseline_ready = bool(result.mower_baseline_ready)
        if not baseline_was_ready and self._episode_mower_baseline_ready:
            current_snapshot = current_facts or build_step_facts(current, self.config.plant_types)
            active_rows = current_snapshot.mower.active_rows or frozenset()
            rows = max(0, int(current_snapshot.rows or self.config.row_count))
            print(
                "[corruption-baseline] "
                f"mowers_initialized={[row in active_rows for row in range(rows)]} "
                "classification=initial_materialization corruption=False"
            )
        return result.diagnostics
    def _facts_snapshot(
        self,
        observation: Dict[str, Any],
        facts: Optional[StepFacts] = None,
    ) -> StepFacts:
        return facts or self._step_facts_cache.get(observation, self.config.plant_types)

    def _seed_slot_snapshot(
        self,
        observation: Dict[str, Any],
        slot_index: int,
        *,
        facts: Optional[StepFacts] = None,
    ) -> Dict[str, Any]:
        if slot_index < 0:
            return {}
        slot = self._facts_snapshot(observation, facts).seed_slot(slot_index)
        if slot is None:
            return {"slotIndex": int(slot_index), "present": False}
        return {
            "slotIndex": int(slot_index),
            "present": True,
            "plantType": slot.plant_type,
            "plantTypeName": slot.plant_type_name or (
                plant_type_name(slot.plant_type) if slot.plant_type >= 0 else "unknown"
            ),
            "seedCost": slot.seed_cost,
            "usable": slot.usable,
            "ready": slot.ready,
            "disabled": slot.disabled,
            "isAvailable": slot.is_available,
            "rawCooldown": slot.raw_cooldown,
            "currentCooldown": slot.current_cooldown,
            "fullCooldown": slot.full_cooldown,
            "cardInstanceId": slot.card_instance_id,
            "source": slot.source,
        }

    @staticmethod
    def _safe_int(*values: Any, default: int = 0) -> int:
        for value in values:
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return int(default)

def validate_observation(observation: Dict[str, Any]) -> None:
    required = [
        "boardFound",
        "sun",
        "wave",
        "maxWave",
        "rowCount",
        "columnCount",
        "plants",
        "zombies",
        "lanes",
        "legalActions",
    ]
    missing = [key for key in required if key not in observation]
    if missing:
        raise AssertionError(f"observation is missing keys: {missing}")
    if not observation.get("boardFound"):
        raise AssertionError("observation says boardFound=false")
    if int(observation.get("rowCount", 0)) <= 0 or int(observation.get("columnCount", 0)) <= 0:
        raise AssertionError("observation has invalid board dimensions")


def parse_plant_types(raw: Optional[str]) -> List[int]:
    if raw is None or not raw.strip():
        return list(GENERALIST_INITIAL_PLANT_TYPES)
    values: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError("--plant-types must include at least one plant type id")
    return values


def decode_action(action: int, observation: Dict[str, Any], plant_types: List[int]) -> Dict[str, int]:
    if action <= 0:
        return {"kind": 0, "slot_index": -1, "plant_type": -1, "row": -1, "column": -1}
    live_rows = int(observation.get("rowCount", 0))
    live_cols = int(observation.get("columnCount", 0))
    if live_rows <= 0 or live_cols <= 0:
        return {"kind": -1, "slot_index": -1, "plant_type": -1, "row": -1, "column": -1}
    encoded = action - 1
    slot_index = encoded // CELLS_PER_SLOT
    cell = encoded % CELLS_PER_SLOT
    slots = seed_slots_from_observation(observation, plant_types)
    plant_type = int(slots[slot_index].get("plantType", -1)) if 0 <= slot_index < len(slots) else -1
    return {"kind": 1, "slot_index": slot_index, "plant_type": plant_type, "row": cell // DEFAULT_COLS, "column": cell % DEFAULT_COLS}


def legal_actions_for_plant_type(env: PvZGymEnv, observation: Dict[str, Any], plant_type: int) -> List[int]:
    return [
        action for action in env.legal_actions(observation)
        if decode_action(action, observation, env.config.plant_types).get("plant_type") == plant_type
    ]


def card_cooldown_for(observation: Dict[str, Any], plant_type: int) -> Dict[str, Any]:
    for cooldown in observation.get("cardCooldowns", []):
        if int(cooldown.get("plantType", -999)) == plant_type:
            return cooldown
    return {}


def verify_legal_actions_exclude_occupied(env: PvZGymEnv, observation: Dict[str, Any]) -> Tuple[bool, str]:
    legal = set(env.legal_actions(observation))
    occupied = {(int(p.get("row", -1)), int(p.get("column", -1))) for p in observation.get("plants", [])}
    visible_occupied = {
        (int(p.get("row", -1)), int(p.get("column", -1)))
        for p in observation.get("visiblePlants", [])
        if bool(p.get("activeInHierarchy", True)) and bool(p.get("inBoardBounds", True))
    }
    occupied |= visible_occupied
    bad_actions = []
    for action in legal:
        decoded = decode_action(action, observation, env.config.plant_types)
        if decoded["kind"] == 1 and (decoded["row"], decoded["column"]) in occupied:
            bad_actions.append(action)
    if bad_actions:
        return False, f"legal actions include occupied cells: {bad_actions[:10]}"
    return True, f"occupied_cells={len(occupied)}, visible_occupied={len(visible_occupied)}, legal_actions={len(legal)}"


def verify_reset_cleanup_state(
    env: PvZGymEnv,
    observation: Dict[str, Any],
    require_mowers: bool = False,
) -> Tuple[bool, str]:
    logical = int(observation.get("plantCount", len(observation.get("plants", []))))
    visible = int(observation.get("visiblePlantObjectCount", logical))
    stale = int(observation.get("staleVisiblePlantObjectCount", 0))
    logical_mowers = int(observation.get("logicalMowerCount", 0))
    visible_mowers = int(observation.get("visibleMowerObjectCount", logical_mowers))
    stale_mowers = int(observation.get("staleVisibleMowerObjectCount", 0))
    duplicate_mower_rows = int(observation.get("duplicateMowerRowCount", 0))
    row_count = int(observation.get("rowCount", 0))
    if visible != logical:
        return False, f"visible plant count does not match logical count: logical={logical}, visible={visible}, stale={stale}"
    if stale != 0:
        return False, f"stale visible plant objects remain: logical={logical}, visible={visible}, stale={stale}"
    if visible_mowers != logical_mowers:
        return False, (
            "visible mower count does not match logical count: "
            f"logical={logical_mowers}, visible={visible_mowers}, stale={stale_mowers}, duplicate_rows={duplicate_mower_rows}"
        )
    if stale_mowers != 0:
        return False, (
            "stale visible mower objects remain: "
            f"logical={logical_mowers}, visible={visible_mowers}, stale={stale_mowers}, duplicate_rows={duplicate_mower_rows}"
        )
    if duplicate_mower_rows != 0:
        return False, (
            "duplicate visible mower rows remain: "
            f"rows={observation.get('duplicateMowerRows', [])}, logical={logical_mowers}, visible={visible_mowers}"
        )
    if require_mowers and row_count > 0 and (logical_mowers != row_count or visible_mowers != row_count):
        return False, (
            "mowers are not ready one-per-row yet: "
            f"rows={row_count}, logical={logical_mowers}, visible={visible_mowers}, stale={stale_mowers}"
        )
    if visible_mowers > 0 and row_count > 0 and visible_mowers != row_count:
        return False, f"expected one visible mower per row: rows={row_count}, visible_mowers={visible_mowers}"
    occupied_ok, occupied_message = verify_legal_actions_exclude_occupied(env, observation)
    if not occupied_ok:
        return False, occupied_message
    return True, (
        f"plants logical={logical}, visible={visible}, stale={stale}; "
        f"mowers logical={logical_mowers}, visible={visible_mowers}, stale={stale_mowers}, "
        f"duplicate_rows={duplicate_mower_rows}; {occupied_message}"
    )


def verify_teacher_action_legal(env: PvZGymEnv, observation: Dict[str, Any]) -> Tuple[bool, str]:
    legal = set(env.legal_actions(observation))
    action = env.teacher_action(observation)
    return action in legal or action == 0, f"teacher_action={action}, legal={action in legal}"


def classify_done_reason(observation: Dict[str, Any]) -> str:
    """Map bridge terminal hints onto the small runner reason set."""
    return legacy_done_reason(observation)


def smoke_test(
    env: PvZGymEnv,
    wait_for_board: bool = True,
    board_timeout: float = 180.0,
    setup_reset: bool = False,
) -> bool:
    checks: List[Tuple[str, bool, str]] = []

    def record(name: str, ok: bool, message: str = "") -> None:
        checks.append((name, ok, message))

    try:
        env.configure()
        env.client.connect()
        record("connect_to_bridge", True)
    except Exception as exc:
        record("connect_to_bridge", False, str(exc))
        print_smoke_results(checks)
        return False

    try:
        observation = env.wait_for_board(timeout=board_timeout) if wait_for_board else env.observe()
        validate_observation(observation)
        if setup_reset:
            observation, _ = env.soft_reset(start_sun=env.config.start_sun, run_init=False)
            validate_observation(observation)
        record("observe_structured_state", True)
    except Exception as exc:
        record("observe_structured_state", False, str(exc))
        print_smoke_results(checks)
        return False

    try:
        legal = env.legal_actions(observation)
        record("legal_actions_nonempty", len(legal) > 0 and all(isinstance(action, int) for action in legal), f"count={len(legal)}")
        ok, message = verify_legal_actions_exclude_occupied(env, observation)
        record("legal_actions_exclude_occupied", ok, message)
        ok, message = verify_teacher_action_legal(env, observation)
        record("teacher_action_legal", ok, message)
    except Exception as exc:
        record("legal_actions_nonempty", False, str(exc))
        print_smoke_results(checks)
        return False

    try:
        before_time = float(observation.get("time", 0.0))
        before_frame = int(observation.get("frameCount", 0))
        wait_obs, _, _, _, _ = env.step(0)
        after_time = float(wait_obs.get("time", 0.0))
        after_frame = int(wait_obs.get("frameCount", 0))
        advanced = after_frame > before_frame or after_time > before_time
        record("step_wait_advances", advanced, f"time {before_time:.3f}->{after_time:.3f}, frame {before_frame}->{after_frame}")
        observation = wait_obs
    except Exception as exc:
        record("step_wait_advances", False, str(exc))
        print_smoke_results(checks)
        return False

    try:
        legal = [action for action in env.legal_actions(observation) if action > 0]
        if not legal:
            record("step_valid_plant_places", False, "no legal plant action is currently available")
        else:
            plant_count_before = int(observation.get("plantCount", len(observation.get("plants", []))))
            plant_obs, _, _, _, info = env.step(legal[0])
            placement = info.get("action_result", {}).get("placement") or {}
            plant_count_after = int(plant_obs.get("plantCount", len(plant_obs.get("plants", []))))
            ok = bool(placement.get("success")) or plant_count_after > plant_count_before
            record("step_valid_plant_places", ok, f"action={legal[0]}, plant_count {plant_count_before}->{plant_count_after}")
            observation = plant_obs
    except Exception as exc:
        record("step_valid_plant_places", False, str(exc))
        print_smoke_results(checks)
        return False

    try:
        rows = int(observation.get("rowCount") or env.config.row_count)
        cols = int(observation.get("columnCount") or env.config.column_count)
        invalid_action = 1 + len(env.config.plant_types) * rows * cols + 999
        _, _, _, _, info = env.step(invalid_action)
        action_result = info.get("action_result", {})
        blocked = bool(action_result.get("preStepMaskBlockedAction"))
        illegal = bool(action_result.get("illegalAction"))
        record("illegal_action_rejected_safely", illegal or blocked, f"action={invalid_action}, illegal={illegal}, pre_step_blocked={blocked}")
    except Exception as exc:
        record("illegal_action_rejected_safely", False, str(exc))
        print_smoke_results(checks)
        return False

    try:
        reset_obs, reset_info = env.soft_reset(run_init=False)
        validate_observation(reset_obs)
        reset_ok = bool(reset_info.get("reset", {}).get("ok", True))
        clean_state = int(reset_obs.get("plantCount", 0)) == 0 and int(reset_obs.get("zombieCount", 0)) == 0
        record("soft_reset_valid_state", reset_ok and clean_state, f"plantCount={reset_obs.get('plantCount')}, zombieCount={reset_obs.get('zombieCount')}, reset={reset_info.get('reset', {})}")
    except Exception as exc:
        record("soft_reset_valid_state", False, str(exc))
        print_smoke_results(checks)
        return False

    try:
        validate_observation(env.observe())
        record("post_smoke_observe_no_crash", True)
    except Exception as exc:
        record("post_smoke_observe_no_crash", False, str(exc))

    print_smoke_results(checks)
    return all(ok for _, ok, _ in checks)


def adventure_state_smoke(
    env: PvZGymEnv,
    duration_seconds: float = 60.0,
    poll_seconds: float = 0.2,
    auto_select_seeds: bool = False,
    seed_list: Optional[List[str]] = None,
) -> bool:
    """Read and lightly exercise Adventure screen detection without loading PPO."""
    started = time.monotonic()
    deadline = started + max(1.0, duration_seconds)
    observed: Counter[str] = Counter()
    last_state = ""
    errors: List[str] = []
    seed_names = seed_list or env.config.seed_list
    print("Adventure State Smoke")
    print("---------------------")
    while time.monotonic() < deadline:
        try:
            state = env.adventure_screen_state()
            screen_state = str(state.get("screenState") or "unknown")
            observed[screen_state] += 1
            if screen_state != last_state:
                print(
                    "[adventure-smoke] "
                    f"t={time.monotonic() - started:.1f}s state={screen_state} "
                    f"level={state.get('currentAdventureLevel')} "
                    f"startup_popup={state.get('startupPopupVisible')} "
                    f"startup_ok={state.get('startupOkButtonVisible')} "
                    f"blocked={state.get('mainMenuBlockedByPopup')} "
                    f"adventure={state.get('isAdventureButtonVisible')} "
                    f"seed={state.get('isSeedSelectionScreen')} "
                    f"gameplay={state.get('isGameplayReady')} "
                    f"reward={state.get('isRewardScreen')} "
                    f"trophy={state.get('trophyVisible')} "
                    f"post_win_click={state.get('postWinClickRequired')} "
                    f"game_over={state.get('isGameOverScreen')}"
                )
                last_state = screen_state

            if state.get("startupPopupVisible") or state.get("startupOkButtonVisible"):
                click = env.click_startup_ok_once()
                print(f"[adventure-smoke] click_startup_ok ok={click.get('ok')} method={click.get('methodUsed')}")
                dismissed = env.wait_for_startup_popup_dismissed(timeout=5.0, poll_seconds=poll_seconds)
                print(
                    "[adventure-smoke] startup_popup_after="
                    f"{dismissed.get('startupPopupVisible')} startup_ok_after={dismissed.get('startupOkButtonVisible')}"
                )
                time.sleep(max(0.5, poll_seconds))
                continue

            if state.get("isAdventureButtonVisible"):
                click = env.press_adventure_once()
                print(f"[adventure-smoke] press_adventure ok={click.get('ok')} method={click.get('methodUsed')}")
                time.sleep(max(0.5, poll_seconds))
                continue

            if state.get("trophyVisible") or state.get("levelCompleteTrophyVisible") or state.get("postWinClickRequired"):
                click = env.click_trophy_once()
                print(f"[adventure-smoke] click_trophy ok={click.get('ok')} method={click.get('methodUsed')}")
                time.sleep(max(0.5, poll_seconds))
                continue

            if state.get("isRewardScreen") or state.get("blockingRewardUiActive"):
                click = env.click_reward_continue_once()
                print(f"[adventure-smoke] click_reward_continue ok={click.get('ok')} method={click.get('methodUsed')}")
                time.sleep(max(0.5, poll_seconds))
                continue

            if state.get("isGameOverScreen"):
                click = env.click_try_again_once()
                print(f"[adventure-smoke] click_try_again ok={click.get('ok')} method={click.get('methodUsed')}")
                time.sleep(max(0.5, poll_seconds))
                continue

            if auto_select_seeds and state.get("isSeedSelectionScreen"):
                selection = env.auto_select_seeds(seed_list=seed_names, start_level=True)
                print(f"[adventure-smoke] auto_select_seeds ok={selection.get('ok')} message={selection.get('message')}")
                time.sleep(max(0.5, poll_seconds))
                continue
        except Exception as exc:
            errors.append(str(exc))
            print(f"[adventure-smoke] error={exc}")
            time.sleep(max(0.5, poll_seconds))
            continue

        time.sleep(max(0.05, poll_seconds))

    print("[adventure-smoke] observed=" + json.dumps(dict(observed), sort_keys=True))
    if errors:
        print("[adventure-smoke] errors=" + json.dumps(errors[-5:], indent=2))
    return bool(observed) and len(errors) == 0


def sun_cost_test(env: PvZGymEnv) -> bool:
    checks: List[Tuple[str, bool, str]] = []

    def record(name: str, ok: bool, message: str = "") -> None:
        checks.append((name, ok, message))

    try:
        env.configure()
        low_obs, _ = env.soft_reset(start_sun=0, run_init=False)
        validate_observation(low_obs)
        low_legal = env.legal_actions(low_obs)
        low_plant_actions = [action for action in low_legal if action > 0]
        record(
            "low_sun_masks_plant_actions",
            low_legal == [0] or len(low_plant_actions) == 0,
            f"sun={low_obs.get('sun')}, legal_count={len(low_legal)}",
        )
    except Exception as exc:
        record("low_sun_masks_plant_actions", False, str(exc))
        print_smoke_results(checks)
        return False

    try:
        enough_start_sun = max(500, int(env.config.start_sun or 0))
        enough_obs, _ = env.soft_reset(start_sun=enough_start_sun, run_init=False)
        validate_observation(enough_obs)
        enough_legal = [action for action in env.legal_actions(enough_obs) if action > 0]
        record(
            "enough_sun_includes_plant_actions",
            len(enough_legal) > 0,
            f"sun={enough_obs.get('sun')}, plant_actions={len(enough_legal)}",
        )
    except Exception as exc:
        record("enough_sun_includes_plant_actions", False, str(exc))
        print_smoke_results(checks)
        return False

    try:
        action = enough_legal[0]
        before_sun = int(enough_obs.get("sun", 0))
        plant_obs, _, _, _, info = env.step(action)
        action_result = info.get("action_result", {})
        placement = action_result.get("placement") or {}
        plant_cost = int(action_result.get("plantCost") or placement.get("plantCost") or 0)
        sun_before = int(action_result.get("sunBefore") or placement.get("sunBefore") or before_sun)
        sun_after = int(action_result.get("sunAfter") or placement.get("sunAfter") or plant_obs.get("sun", 0))
        cost_paid = bool(action_result.get("costPaid") or placement.get("costPaid"))
        plant_placed = bool(action_result.get("plantPlaced") or placement.get("plantPlaced") or placement.get("success"))
        record(
            "successful_plant_decreases_sun",
            plant_placed and cost_paid and sun_before - sun_after == plant_cost and plant_cost > 0,
            f"action={action}, cost={plant_cost}, sun {sun_before}->{sun_after}, source={placement.get('costSource')}",
        )
    except Exception as exc:
        record("successful_plant_decreases_sun", False, str(exc))
        print_smoke_results(checks)
        return False

    try:
        low_obs, _ = env.soft_reset(start_sun=0, run_init=False)
        _, _, _, _, info = env.step(action)
        action_result = info.get("action_result", {})
        action_audit = action_result.get("actionAudit") if isinstance(action_result.get("actionAudit"), dict) else {}
        record(
            "unaffordable_plant_blocked",
            (
                bool(action_result.get("illegalAction")) and action_result.get("illegalReason") == "insufficient_sun"
            )
            or (
                bool(action_result.get("preStepMaskBlockedAction"))
                and action_audit.get("pythonFilterReason") == "insufficient_sun"
            ),
            (
                f"illegal={action_result.get('illegalAction')}, reason={action_result.get('illegalReason')}, "
                f"pre_step_blocked={action_result.get('preStepMaskBlockedAction')}, filter={action_audit.get('pythonFilterReason')}"
            ),
        )
    except Exception as exc:
        record("unaffordable_plant_rejected", False, str(exc))
        print_smoke_results(checks)
        return False

    try:
        reset_obs, _ = env.soft_reset(start_sun=env.config.start_sun, run_init=False)
        expected_sun = int(env.config.start_sun or 0)
        record(
            "soft_reset_restores_start_sun",
            int(reset_obs.get("sun", -1)) == expected_sun,
            f"sun={reset_obs.get('sun')}, expected={expected_sun}",
        )
    except Exception as exc:
        record("soft_reset_restores_start_sun", False, str(exc))

    print_smoke_results(checks)
    return all(ok for _, ok, _ in checks)


def cooldown_test(env: PvZGymEnv) -> bool:
    checks: List[Tuple[str, bool, str]] = []

    def record(name: str, ok: bool, message: str = "") -> None:
        checks.append((name, ok, message))

    def wait_until_ready(plant_type: int, max_wait_steps: int = 600) -> Tuple[Dict[str, Any], int]:
        observation = env.observe()
        for step_index in range(max_wait_steps + 1):
            cooldown = card_cooldown_for(observation, plant_type)
            if cooldown.get("ready") and legal_actions_for_plant_type(env, observation, plant_type):
                return observation, step_index
            observation, _, _, _, _ = env.step(0)
        return observation, max_wait_steps

    def exercise_plant_cooldown(plant_type: int, name: str) -> None:
        observation, _ = env.soft_reset(start_sun=max(500, int(env.config.start_sun or 0)), run_init=False)
        validate_observation(observation)
        before_actions = legal_actions_for_plant_type(env, observation, plant_type)
        before_count = len(before_actions)
        if not before_actions:
            record(f"{name}_initially_legal", False, f"legal_count={before_count}")
            return

        action = before_actions[0]
        sun_before = int(observation.get("sun", 0))
        placed_obs, _, _, _, info = env.step(action)
        action_result = info.get("action_result", {})
        placement = action_result.get("placement") or {}
        plant_placed = bool(action_result.get("plantPlaced") or placement.get("plantPlaced") or placement.get("success"))
        record(
            f"{name}_placement_starts_cooldown",
            plant_placed and bool(action_result.get("cooldownStarted") or placement.get("cooldownStarted")),
            f"action={action}, cooldown={card_cooldown_for(placed_obs, plant_type)}",
        )

        during_actions = legal_actions_for_plant_type(env, placed_obs, plant_type)
        record(
            f"{name}_actions_masked_during_cooldown",
            len(during_actions) < before_count and 0 in env.legal_actions(placed_obs),
            f"before={before_count}, during={len(during_actions)}, wait_legal={0 in env.legal_actions(placed_obs)}",
        )

        rejected_action = next((candidate for candidate in before_actions if candidate != action), action)
        reject_obs, _, _, _, reject_info = env.step(rejected_action)
        reject_result = reject_info.get("action_result", {})
        reject_placement = reject_result.get("placement") or {}
        sun_after_reject = int(reject_obs.get("sun", 0))
        expected_sun = int(reject_result.get("sunBefore") or reject_placement.get("sunBefore") or sun_after_reject)
        reject_sun_after = int(reject_result.get("sunAfter") or reject_placement.get("sunAfter") or sun_after_reject)
        reject_audit = reject_result.get("actionAudit") if isinstance(reject_result.get("actionAudit"), dict) else {}
        pre_step_blocked = bool(reject_result.get("preStepMaskBlockedAction"))
        record(
            f"{name}_cooldown_blocked_without_spending_sun",
            (
                (
                    bool(reject_result.get("illegalAction"))
                    and reject_result.get("illegalReason") == "cooldown"
                )
                or (
                    pre_step_blocked
                    and reject_audit.get("pythonFilterReason") == "cooldown_not_ready"
                )
            )
            and reject_sun_after == expected_sun,
            (
                f"illegal={reject_result.get('illegalAction')}, reason={reject_result.get('illegalReason')}, "
                f"pre_step_blocked={pre_step_blocked}, filter={reject_audit.get('pythonFilterReason')}, "
                f"reject_sun={expected_sun}->{reject_sun_after}"
            ),
        )

        ready_obs, wait_steps = wait_until_ready(plant_type)
        ready_actions = legal_actions_for_plant_type(env, ready_obs, plant_type)
        record(
            f"{name}_actions_return_after_cooldown",
            len(ready_actions) > 0 and card_cooldown_for(ready_obs, plant_type).get("ready"),
            f"wait_steps={wait_steps}, ready_count={len(ready_actions)}, cooldown={card_cooldown_for(ready_obs, plant_type)}",
        )

    try:
        env.configure()
        env.wait_for_gameplay_ready(timeout=30.0, poll_seconds=0.25)
        record("gameplay_ready", True)
    except Exception as exc:
        record("gameplay_ready", False, str(exc))
        print_smoke_results(checks)
        return False

    try:
        exercise_plant_cooldown(1, "sunflower")
        exercise_plant_cooldown(0, "peashooter")
    except Exception as exc:
        record("cooldown_test_runtime", False, str(exc))

    print_smoke_results(checks)
    return all(ok for _, ok, _ in checks)


def _select_live_fusion_probe_candidates(
    raw_probe: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], int]:
    """Return the first legal and Peashooter/SunFlower live candidates.

    An empty board or a board with only unrelated legal candidates is setup
    state, not a failed semantic check. The live harness uses the second return
    value to decide whether it must place a Peashooter before testing fusion.
    """

    candidates = raw_probe.get("fusionCandidates") or raw_probe.get("fusion_candidates") or []
    if not isinstance(candidates, list):
        candidates = []
    first_legal: Optional[Dict[str, Any]] = None
    pea_sun_legal: Optional[Dict[str, Any]] = None
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        is_legal = bool(candidate.get("fusionLegal") or candidate.get("fusion_legal"))
        if is_legal and first_legal is None:
            first_legal = candidate
        try:
            source_type = int(candidate.get("sourcePlantType", candidate.get("source_plant_type", -1)))
            ingredient_type = int(
                candidate.get(
                    "ingredientPlantType",
                    candidate.get("ingredient_plant_type", candidate.get("targetPlantType", -1)),
                )
            )
        except (TypeError, ValueError):
            continue
        if is_legal and source_type == 0 and ingredient_type == 1:
            pea_sun_legal = candidate
            break
    return first_legal, pea_sun_legal, len(candidates)


def _live_test_value_int(payload: Dict[str, Any], *keys: str, default: int = -1) -> int:
    for key in keys:
        try:
            if key in payload:
                return int(payload.get(key))
        except (TypeError, ValueError):
            continue
    return int(default)


def _live_test_slot_for_plant_type(obs_payload: Dict[str, Any], plant_type: int) -> Optional[Dict[str, Any]]:
    for slot_item in obs_payload.get("seedSlots", []) or []:
        if not isinstance(slot_item, dict):
            continue
        if int(slot_item.get("plantType", -1)) != int(plant_type):
            continue
        if not bool(slot_item.get("usable", True)):
            continue
        if bool(slot_item.get("disabled", False)):
            continue
        return slot_item
    return None


def _live_test_plant_at_cell(obs_payload: Dict[str, Any], row: int, col: int) -> Optional[Dict[str, Any]]:
    for plant_item in obs_payload.get("plants", []) or []:
        if not isinstance(plant_item, dict):
            continue
        if int(plant_item.get("row", -1)) == int(row) and int(plant_item.get("column", -1)) == int(col):
            return plant_item
    return None


def _live_test_first_empty_cells(env: PvZGymEnv, obs_payload: Dict[str, Any], count: int) -> List[Tuple[int, int]]:
    rows = int(obs_payload.get("rowCount") or env.config.row_count or 5)
    cols = int(obs_payload.get("columnCount") or env.config.column_count or 10)
    occupied = set()
    for plant in obs_payload.get("plants", []) or []:
        if not isinstance(plant, dict):
            continue
        occupied.add((int(plant.get("row", -1)), int(plant.get("column", -1))))
    selected: List[Tuple[int, int]] = []
    for row in range(rows):
        for col in range(cols):
            if (row, col) in occupied:
                continue
            selected.append((row, col))
            if len(selected) >= int(count):
                return selected
    return selected


def _live_test_placement_action(env: PvZGymEnv, slot_index: int, row: int, col: int, obs_payload: Dict[str, Any]) -> int:
    rows = int(obs_payload.get("rowCount") or env.config.row_count or 5)
    cols = int(obs_payload.get("columnCount") or env.config.column_count or 10)
    return int(1 + int(slot_index) * rows * cols + int(row) * cols + int(col))


def _live_test_wait_until_slot_ready(
    env: PvZGymEnv,
    slot_index: int,
    min_sun: int,
    max_wait_steps: int = 600,
) -> Tuple[Dict[str, Any], int, str]:
    current_observation = env.observe()
    for step_index in range(int(max_wait_steps) + 1):
        selected_slot = None
        for slot in current_observation.get("seedSlots", []) or []:
            if not isinstance(slot, dict):
                continue
            if int(slot.get("slotIndex", -1)) == int(slot_index):
                selected_slot = slot
                break
        ready = bool(selected_slot and selected_slot.get("ready", False))
        usable = bool(selected_slot and selected_slot.get("usable", True) and not selected_slot.get("disabled", False))
        sun_ok = int(current_observation.get("sun", 0) or 0) >= int(min_sun)
        if ready and usable and sun_ok:
            return current_observation, step_index, ""
        current_observation, _, _, _, _ = env.step(0)
    return current_observation, int(max_wait_steps), "timeout"


def fusion_semantics_test(env: PvZGymEnv) -> bool:
    checks: List[Tuple[str, bool, str]] = []

    def record(name: str, ok: bool, message: str = "") -> None:
        checks.append((name, ok, message))

    value_int = _live_test_value_int

    def value_bool(payload: Dict[str, Any], *keys: str) -> bool:
        for key in keys:
            if key in payload:
                return bool(payload.get(key))
        return False

    slot_for_plant_type = _live_test_slot_for_plant_type
    plant_at_cell = _live_test_plant_at_cell

    def first_empty_cells(obs_payload: Dict[str, Any], count: int) -> List[Tuple[int, int]]:
        return _live_test_first_empty_cells(env, obs_payload, count)

    def placement_action_for(slot_index: int, row: int, col: int, obs_payload: Dict[str, Any]) -> int:
        return _live_test_placement_action(env, slot_index, row, col, obs_payload)

    def wait_until_slot_ready(slot_index: int, min_sun: int, max_wait_steps: int = 600) -> Tuple[Dict[str, Any], int, str]:
        return _live_test_wait_until_slot_ready(env, slot_index, min_sun, max_wait_steps)

    try:
        env.configure()
        observation = env.wait_for_gameplay_ready(timeout=30.0, poll_seconds=0.25, quiet=True, fail_on_terminal=False)
        record("gameplay_ready", bool(observation.get("gameplayReady")), f"screenState={observation.get('screenState')}")
    except Exception as exc:
        record("gameplay_ready", False, str(exc))
        print_smoke_results(checks)
        return False

    probe: Dict[str, Any] = {}
    legal_candidate: Optional[Dict[str, Any]] = None
    peashooter_sunflower_candidate: Optional[Dict[str, Any]] = None
    try:
        probe = env.client.request("fusion_probe")
        legal_candidate, peashooter_sunflower_candidate, candidate_count = _select_live_fusion_probe_candidates(probe)
        print(
            "[fusion-test] initial_probe "
            f"candidate_count={candidate_count} "
            f"peashooter_sunflower_available={peashooter_sunflower_candidate is not None}"
        )
    except Exception as exc:
        record("fusion_probe_runtime", False, str(exc))
        print_smoke_results(checks)
        return False

    if peashooter_sunflower_candidate is None:
        try:
            rows = int(observation.get("rowCount") or env.config.row_count or 5)
            cols = int(observation.get("columnCount") or env.config.column_count or 10)
            occupied = set()
            for plant in observation.get("plants", []) or []:
                if isinstance(plant, dict):
                    occupied.add((int(plant.get("row", -1)), int(plant.get("column", -1))))
            empty_cell = None
            for row in range(rows):
                for col in range(cols):
                    if (row, col) not in occupied:
                        empty_cell = (row, col)
                        break
                if empty_cell is not None:
                    break
            peashooter_slot_index = -1
            sunflower_slot_ready = False
            for slot in observation.get("seedSlots", []) or []:
                if not isinstance(slot, dict):
                    continue
                plant_type = int(slot.get("plantType", -1))
                ready = bool(slot.get("ready", True))
                usable = bool(slot.get("usable", True)) and not bool(slot.get("disabled", False))
                cost = int(slot.get("seedCost", 0) or 0)
                affordable = int(observation.get("sun", 0) or 0) >= cost
                if plant_type == 0 and ready and usable and affordable:
                    peashooter_slot_index = int(slot.get("slotIndex", -1))
                if plant_type == 1 and ready and usable and affordable:
                    sunflower_slot_ready = True
            setup_ok = empty_cell is not None and peashooter_slot_index >= 0 and sunflower_slot_ready
            if not setup_ok:
                record(
                    "fusion_probe_setup_for_peashooter_sunflower",
                    False,
                    f"empty_cell={empty_cell}, peashooter_slot={peashooter_slot_index}, sunflower_ready={sunflower_slot_ready}",
                )
            else:
                setup_action = 1 + peashooter_slot_index * rows * cols + int(empty_cell[0]) * cols + int(empty_cell[1])
                _obs, _reward, _terminated, _truncated, setup_info = env.step(int(setup_action))
                setup_result = setup_info.get("action_result", {}) if isinstance(setup_info, dict) else {}
                setup_legal = bool(setup_result.get("plantPlaced")) and not bool(setup_result.get("illegalAction"))
                record(
                    "fusion_probe_setup_for_peashooter_sunflower",
                    setup_legal,
                    (
                        f"action={setup_action}, placed={setup_result.get('plantPlaced')}, "
                        f"illegal={setup_result.get('illegalAction')}, reason={setup_result.get('illegalReason')}"
                    ),
                )
                probe = env.client.request("fusion_probe")
                legal_candidate, peashooter_sunflower_candidate, candidate_count = _select_live_fusion_probe_candidates(probe)
                record(
                    "fusion_probe_after_setup_has_legal_candidate",
                    legal_candidate is not None,
                    f"candidate_count={candidate_count}",
                )
                record(
                    "fusion_probe_after_setup_has_peashooter_sunflower_candidate",
                    peashooter_sunflower_candidate is not None,
                    (
                        f"candidate_count={candidate_count}, "
                        f"selected_source={value_int(peashooter_sunflower_candidate or {}, 'sourcePlantType', 'source_plant_type', default=-1)}, "
                        f"selected_ingredient={value_int(peashooter_sunflower_candidate or {}, 'ingredientPlantType', 'ingredient_plant_type', 'targetPlantType', default=-1)}"
                    ),
                )
        except Exception as exc:
            record("fusion_probe_setup_runtime", False, str(exc))

    if legal_candidate is None:
        print_smoke_results(checks)
        return False

    selected_candidate = peashooter_sunflower_candidate or legal_candidate
    source_row = value_int(selected_candidate, "sourceRow", "source_row")
    source_col = value_int(selected_candidate, "sourceCol", "sourceColumn", "source_col")
    source_type = value_int(selected_candidate, "sourcePlantType", "source_plant_type")
    source_instance = value_int(selected_candidate, "sourceInstanceId", "source_instance_id", default=0)
    slot_index = value_int(selected_candidate, "ingredientSeedSlotIndex", "ingredient_seed_slot_index")
    ingredient_type = value_int(selected_candidate, "ingredientPlantType", "ingredient_plant_type", "targetPlantType", default=-1)
    predicted_type = value_int(selected_candidate, "predictedResultType", "predicted_result_type", default=-1)
    predicted_name = str(
        selected_candidate.get("predictedResultName")
        or selected_candidate.get("predicted_result_name")
        or ""
    )
    predicted_resolution_source = str(
        selected_candidate.get("predictedResultResolutionSource")
        or selected_candidate.get("predicted_result_resolution_source")
        or ""
    )
    mix_lookup_found = value_bool(selected_candidate, "mixLookupFound", "mix_lookup_found")
    mix_lookup_key = str(
        selected_candidate.get("mixLookupKey")
        or selected_candidate.get("mix_lookup_key")
        or ""
    )
    record(
        "peashooter_sunflower_candidate_available_without_mutating_probe",
        (
            peashooter_sunflower_candidate is not None
            and predicted_resolution_source != ""
            and "checkmix" not in predicted_resolution_source.lower()
        ),
        (
            f"source={source_type}, ingredient={ingredient_type}, predicted_type={predicted_type}, "
            f"predicted_name={predicted_name}, resolution={predicted_resolution_source}, "
            f"mix_lookup_found={mix_lookup_found}, mix_lookup_key={mix_lookup_key}"
        ),
    )

    fusion_result: Dict[str, Any] = {}
    try:
        fusion_result = env.client.request(
            "fusion_step",
            source_instance_id=source_instance,
            source_row=source_row,
            source_col=source_col,
            source_plant_type=source_type,
            ingredient_seed_slot_index=slot_index,
            ingredient_plant_type=ingredient_type,
            predicted_result_type=predicted_type,
            predicted_result_name=predicted_name,
            return_observation=False,
        )
        bridge_method = str(fusion_result.get("bridgeMethodUsed") or "")
        mode = str(fusion_result.get("fusionExecutionMode") or "")
        duplicate_detected = bool(fusion_result.get("duplicateStackDetected"))
        plant_count_after = int(fusion_result.get("plantCountOnTileAfter", 0) or 0)
        record(
            "fusion_uses_dedicated_execution_path",
            mode == "dedicated_fusion" and "createplant.setplant" not in bridge_method.lower(),
            f"mode={mode}, method={bridge_method}",
        )
        record(
            "fuse_success_reports_diagnostics",
            (
                bool(fusion_result.get("fusionSucceeded"))
                and bool(fusion_result.get("sourceTileOccupiedBefore"))
                and int(fusion_result.get("plantCountOnTileBefore", 0) or 0) >= 1
                and plant_count_after == 1
                and isinstance(fusion_result.get("sourcePlantBefore"), dict)
                and isinstance(fusion_result.get("resultingPlantAfter"), dict)
                and not duplicate_detected
                and str(fusion_result.get("predictedResultResolutionSource") or "") != ""
                and int(fusion_result.get("preSourceType", -1)) >= 0
                and str(fusion_result.get("preSourceName") or "") != ""
                and int(fusion_result.get("ingredientType", -1)) >= 0
                and str(fusion_result.get("ingredientName") or "") != ""
                and int(fusion_result.get("postResultType", -1)) >= 0
                and str(fusion_result.get("postResultName") or "") != ""
            ),
            (
                f"succeeded={fusion_result.get('fusionSucceeded')}, "
                f"before={fusion_result.get('plantCountOnTileBefore')}, after={plant_count_after}, "
                f"duplicate={duplicate_detected}, reason={fusion_result.get('bridgeResultReason')}, "
                f"prediction_source={fusion_result.get('predictedResultResolutionSource')}, "
                f"mix_lookup={fusion_result.get('mixLookupFound')}/{fusion_result.get('mixLookupKey')}, "
                f"post={fusion_result.get('postResultType')}:{fusion_result.get('postResultName')}"
            ),
        )
    except Exception as exc:
        record("fusion_step_runtime", False, str(exc))
        print_smoke_results(checks)
        return False

    try:
        rows = int(observation.get("rowCount") or env.config.row_count or 5)
        cols = int(observation.get("columnCount") or env.config.column_count or 10)
        occupied = set()
        for plant in observation.get("plants", []) or []:
            if isinstance(plant, dict):
                occupied.add((int(plant.get("row", -1)), int(plant.get("column", -1))))
        empty_cell = None
        for row in range(rows):
            for col in range(cols):
                if (row, col) not in occupied:
                    empty_cell = (row, col)
                    break
            if empty_cell is not None:
                break
        if empty_cell is None:
            record("fusion_rejects_empty_source_tile", False, "no empty cell found to probe")
        else:
            empty_result = env.client.request(
                "fusion_step",
                source_instance_id=0,
                source_row=int(empty_cell[0]),
                source_col=int(empty_cell[1]),
                source_plant_type=source_type,
                ingredient_seed_slot_index=slot_index,
                ingredient_plant_type=ingredient_type,
                predicted_result_type=predicted_type,
                predicted_result_name=predicted_name,
                return_observation=False,
            )
            empty_reason = str(empty_result.get("illegalReason") or empty_result.get("fusionRejectedReason") or "")
            record(
                "fusion_rejects_empty_source_tile",
                (not bool(empty_result.get("fusionSucceeded"))) and empty_reason in {"source_not_found", "source_tile_not_occupied"},
                f"reason={empty_reason}",
            )
    except Exception as exc:
        record("fusion_empty_source_runtime", False, str(exc))

    try:
        rows = int(observation.get("rowCount") or env.config.row_count or 5)
        cols = int(observation.get("columnCount") or env.config.column_count or 10)
        placement_action = 1 + slot_index * rows * cols + source_row * cols + source_col
        _obs, _reward, _terminated, _truncated, info = env.step(int(placement_action))
        action_result = info.get("action_result", {}) if isinstance(info, dict) else {}
        action_audit = action_result.get("actionAudit") if isinstance(action_result, dict) else {}
        if not isinstance(action_audit, dict):
            action_audit = {}
        occupied_rejected = (
            bool(action_result.get("illegalAction")) and str(action_result.get("illegalReason") or "") == "occupied_cell"
        ) or (
            bool(action_result.get("preStepMaskBlockedAction"))
            and str(action_audit.get("pythonFilterReason") or "") == "occupied_cell"
        )
        record(
            "normal_placement_rejects_occupied_tile",
            occupied_rejected,
            (
                f"illegal={action_result.get('illegalAction')}, reason={action_result.get('illegalReason')}, "
                f"pre_step_blocked={action_result.get('preStepMaskBlockedAction')}, "
                f"filter={action_audit.get('pythonFilterReason')}"
            ),
        )
    except Exception as exc:
        record("placement_occupied_runtime", False, str(exc))

    try:
        second_fuse = env.client.request(
            "fusion_step",
            source_instance_id=source_instance,
            source_row=source_row,
            source_col=source_col,
            source_plant_type=source_type,
            ingredient_seed_slot_index=slot_index,
            ingredient_plant_type=ingredient_type,
            predicted_result_type=predicted_type,
            predicted_result_name=predicted_name,
            return_observation=False,
        )
        duplicate_detected = bool(second_fuse.get("duplicateStackDetected"))
        duplicate_handled = (not bool(second_fuse.get("fusionSucceeded"))) and str(second_fuse.get("illegalReason") or "") == "duplicate_stack_detected"
        record(
            "duplicate_stack_detection_is_failure_when_triggered",
            (not duplicate_detected) or duplicate_handled,
            (
                f"duplicate={duplicate_detected}, fusionSucceeded={second_fuse.get('fusionSucceeded')}, "
                f"illegalReason={second_fuse.get('illegalReason')}, plantCountAfter={second_fuse.get('plantCountOnTileAfter')}"
            ),
        )
    except Exception as exc:
        record("duplicate_detection_runtime", False, str(exc))

    try:
        scoped_observation = env.observe()
        peashooter_slot = slot_for_plant_type(scoped_observation, 0)
        sunflower_slot = slot_for_plant_type(scoped_observation, 1)
        if peashooter_slot is None or sunflower_slot is None:
            record(
                "tile_scoped_fusion_setup",
                False,
                f"missing_slots peashooter={peashooter_slot is not None}, sunflower={sunflower_slot is not None}",
            )
        else:
            peashooter_slot_index = int(peashooter_slot.get("slotIndex", -1))
            sunflower_slot_index = int(sunflower_slot.get("slotIndex", -1))
            peashooter_cost = int(peashooter_slot.get("seedCost", 0) or 0)
            sunflower_cost = int(sunflower_slot.get("seedCost", 0) or 0)
            if peashooter_slot_index < 0 or sunflower_slot_index < 0:
                record(
                    "tile_scoped_fusion_setup",
                    False,
                    f"invalid_slot_indices peashooter={peashooter_slot_index}, sunflower={sunflower_slot_index}",
                )
            else:
                scoped_observation, wait_steps_1, wait_reason_1 = wait_until_slot_ready(
                    peashooter_slot_index,
                    min_sun=peashooter_cost,
                )
                empty_cells = first_empty_cells(scoped_observation, 2)
                if wait_reason_1 or len(empty_cells) < 2:
                    record(
                        "tile_scoped_fusion_setup",
                        False,
                        f"wait_reason={wait_reason_1}, wait_steps={wait_steps_1}, empty_cells={empty_cells}",
                    )
                else:
                    source_cell = empty_cells[0]
                    control_cell = empty_cells[1]
                    first_action = placement_action_for(peashooter_slot_index, source_cell[0], source_cell[1], scoped_observation)
                    scoped_observation, _reward, _terminated, _truncated, first_place_info = env.step(first_action)
                    first_action_result = first_place_info.get("action_result", {}) if isinstance(first_place_info, dict) else {}
                    first_placed = bool(first_action_result.get("plantPlaced")) and not bool(first_action_result.get("illegalAction"))
                    if not first_placed:
                        record(
                            "tile_scoped_fusion_setup",
                            False,
                            f"first_place_failed action={first_action}, illegal={first_action_result.get('illegalAction')}, reason={first_action_result.get('illegalReason')}",
                        )
                    else:
                        scoped_observation, wait_steps_2, wait_reason_2 = wait_until_slot_ready(
                            peashooter_slot_index,
                            min_sun=peashooter_cost + sunflower_cost,
                        )
                        second_action = placement_action_for(peashooter_slot_index, control_cell[0], control_cell[1], scoped_observation)
                        scoped_observation, _reward, _terminated, _truncated, second_place_info = env.step(second_action)
                        second_action_result = second_place_info.get("action_result", {}) if isinstance(second_place_info, dict) else {}
                        second_placed = bool(second_action_result.get("plantPlaced")) and not bool(second_action_result.get("illegalAction"))
                        if wait_reason_2 or not second_placed:
                            record(
                                "tile_scoped_fusion_setup",
                                False,
                                (
                                    f"wait_reason={wait_reason_2}, wait_steps={wait_steps_2}, "
                                    f"second_place_failed action={second_action}, illegal={second_action_result.get('illegalAction')}, "
                                    f"reason={second_action_result.get('illegalReason')}"
                                ),
                            )
                        else:
                            source_before = plant_at_cell(scoped_observation, source_cell[0], source_cell[1])
                            control_before = plant_at_cell(scoped_observation, control_cell[0], control_cell[1])
                            if source_before is None or control_before is None:
                                record(
                                    "tile_scoped_fusion_setup",
                                    False,
                                    f"missing_source_or_control source={source_before}, control={control_before}",
                                )
                            else:
                                control_before_type = int(control_before.get("type", control_before.get("plantType", -1)))
                                control_before_name = str(control_before.get("typeName") or control_before.get("plantTypeName") or "")
                                control_before_id = int(control_before.get("instanceId", control_before.get("instanceID", 0)) or 0)
                                scoped_observation, wait_steps_3, wait_reason_3 = wait_until_slot_ready(
                                    sunflower_slot_index,
                                    min_sun=sunflower_cost,
                                )
                                if wait_reason_3:
                                    record(
                                        "tile_scoped_fusion_setup",
                                        False,
                                        f"sunflower_wait_timeout steps={wait_steps_3}",
                                    )
                                else:
                                    source_before = plant_at_cell(scoped_observation, source_cell[0], source_cell[1]) or source_before
                                    source_before_type = int(source_before.get("type", source_before.get("plantType", -1)))
                                    source_before_id = int(source_before.get("instanceId", source_before.get("instanceID", 0)) or 0)
                                    scoped_fusion_result = env.client.request(
                                        "fusion_step",
                                        source_instance_id=source_before_id,
                                        source_row=int(source_cell[0]),
                                        source_col=int(source_cell[1]),
                                        source_plant_type=source_before_type,
                                        ingredient_seed_slot_index=sunflower_slot_index,
                                        ingredient_plant_type=1,
                                        predicted_result_type=predicted_type,
                                        predicted_result_name=predicted_name,
                                        return_observation=True,
                                    )
                                    changed_tile_count = int(
                                        scoped_fusion_result.get("changedTileCount")
                                        or scoped_fusion_result.get("changed_tile_count")
                                        or 0
                                    )
                                    changed_tiles = scoped_fusion_result.get("changedTiles")
                                    if not isinstance(changed_tiles, list):
                                        changed_tiles = scoped_fusion_result.get("changed_tiles")
                                    if not isinstance(changed_tiles, list):
                                        changed_tiles = []
                                    non_source_tiles_changed = bool(
                                        scoped_fusion_result.get("nonSourceTilesChanged")
                                        if "nonSourceTilesChanged" in scoped_fusion_result
                                        else scoped_fusion_result.get("non_source_tiles_changed")
                                    )
                                    global_side_effect = bool(
                                        scoped_fusion_result.get("globalFusionSideEffect")
                                        if "globalFusionSideEffect" in scoped_fusion_result
                                        else scoped_fusion_result.get("global_fusion_side_effect")
                                    )
                                    fusion_scope_value = str(
                                        scoped_fusion_result.get("fusionScope")
                                        or scoped_fusion_result.get("fusion_scope")
                                        or ""
                                    )
                                    post_observation = scoped_fusion_result.get("observation")
                                    if not isinstance(post_observation, dict):
                                        post_observation = env.observe()
                                    control_after = plant_at_cell(post_observation, control_cell[0], control_cell[1])
                                    control_after_type = int(control_after.get("type", control_after.get("plantType", -1))) if isinstance(control_after, dict) else -1
                                    control_after_name = str(control_after.get("typeName") or control_after.get("plantTypeName") or "") if isinstance(control_after, dict) else ""
                                    control_after_id = int(control_after.get("instanceId", control_after.get("instanceID", 0)) or 0) if isinstance(control_after, dict) else 0
                                    source_tile_changed_only = (
                                        changed_tile_count == 1
                                        and len(changed_tiles) == 1
                                        and int(value_int(changed_tiles[0], "row", "source_row", default=-1)) == int(source_cell[0])
                                        and int(value_int(changed_tiles[0], "column", "col", "source_col", default=-1)) == int(source_cell[1])
                                        and not non_source_tiles_changed
                                        and not global_side_effect
                                    )
                                    control_unchanged = (
                                        isinstance(control_after, dict)
                                        and control_after_type == control_before_type
                                        and control_after_name == control_before_name
                                        and (control_before_id <= 0 or control_after_id == control_before_id)
                                    )
                                    record(
                                        "tile_scoped_fusion_only_changes_requested_tile",
                                        bool(
                                            scoped_fusion_result.get("fusionSucceeded")
                                            and source_tile_changed_only
                                            and fusion_scope_value == "tile_scoped"
                                        ),
                                        (
                                            f"succeeded={scoped_fusion_result.get('fusionSucceeded')}, changed_count={changed_tile_count}, "
                                            f"changed_tiles={changed_tiles}, non_source={non_source_tiles_changed}, "
                                            f"global_side_effect={global_side_effect}, fusion_scope={fusion_scope_value}, "
                                            f"bridge_reason={scoped_fusion_result.get('bridgeResultReason')}"
                                        ),
                                    )
                                    record(
                                        "tile_scoped_fusion_preserves_non_target_peashooter",
                                        control_unchanged,
                                        (
                                            f"before={control_before_type}:{control_before_name}:{control_before_id}, "
                                            f"after={control_after_type}:{control_after_name}:{control_after_id}, "
                                            f"source_cell={source_cell}, control_cell={control_cell}"
                                        ),
                                    )
    except Exception as exc:
        record("tile_scoped_fusion_runtime", False, str(exc))

    print_smoke_results(checks)
    return all(ok for _, ok, _ in checks)


def coach_fusion_scope_test(env: PvZGymEnv) -> bool:
    """Live bridge entrypoint focused on coach-originated tile-scoped fusion."""
    from pvzrl_human_coach import HumanCoachOverrideHook, QueueCoachCommandSource

    checks: List[Tuple[str, bool, str]] = []

    def record(name: str, ok: bool, message: str = "") -> None:
        checks.append((name, ok, message))

    value_int = _live_test_value_int
    slot_for_plant_type = _live_test_slot_for_plant_type
    plant_at_cell = _live_test_plant_at_cell

    def first_empty_cells(obs_payload: Dict[str, Any], count: int) -> List[Tuple[int, int]]:
        return _live_test_first_empty_cells(env, obs_payload, count)

    def placement_action_for(slot_index: int, row: int, col: int, obs_payload: Dict[str, Any]) -> int:
        return _live_test_placement_action(env, slot_index, row, col, obs_payload)

    def wait_until_slot_ready(slot_index: int, min_sun: int, max_wait_steps: int = 600) -> Tuple[Dict[str, Any], int, str]:
        return _live_test_wait_until_slot_ready(env, slot_index, min_sun, max_wait_steps)

    try:
        env.configure()
        observation = env.wait_for_gameplay_ready(timeout=30.0, poll_seconds=0.25, quiet=True, fail_on_terminal=False)
        record("gameplay_ready", bool(observation.get("gameplayReady")), f"screenState={observation.get('screenState')}")
    except Exception as exc:
        record("gameplay_ready", False, str(exc))
        print_smoke_results(checks)
        return False

    try:
        if hasattr(env, "clear_coach_runtime_state"):
            env.clear_coach_runtime_state(
                queue_cleared=True,
                startup_command_blocked=False,
                reason="coach_fusion_scope_test_start",
            )
        startup_source = QueueCoachCommandSource(["!fuse 0 0 0"])
        startup_hook = HumanCoachOverrideHook(
            enabled=True,
            source=startup_source,
            fusion_enabled=True,
            platform="mock",
        )
        startup_stale_detected = startup_hook.clear_pending_state(
            clear_source=True,
            reason="coach_fusion_scope_test_start",
        )
        startup_status = startup_hook.live_status_fields()
        record(
            "startup_queue_clear_reports_stale_only_when_stale_command_discarded",
            (
                bool(startup_stale_detected)
                and startup_status.get("coach_command_queue_cleared_on_reset") is True
                and startup_status.get("pending_coach_command") is None
                and startup_status.get("selected_bridge_command") is None
                and startup_status.get("startup_command_blocked") is True
            ),
            json.dumps(
                {
                    "stale_detected": startup_stale_detected,
                    "coach_command_queue_cleared_on_reset": startup_status.get("coach_command_queue_cleared_on_reset"),
                    "pending_coach_command": startup_status.get("pending_coach_command"),
                    "selected_bridge_command": startup_status.get("selected_bridge_command"),
                    "startup_command_blocked": startup_status.get("startup_command_blocked"),
                },
                sort_keys=True,
            ),
        )

        clean_hook = HumanCoachOverrideHook(
            enabled=True,
            source=QueueCoachCommandSource(),
            fusion_enabled=True,
            platform="mock",
        )
        clean_stale_detected = clean_hook.clear_pending_state(
            clear_source=True,
            reason="coach_fusion_scope_test_clean_start",
        )
        clean_status = clean_hook.live_status_fields()
        record(
            "clean_startup_live_status_has_no_pending_or_selected_command",
            (
                not bool(clean_stale_detected)
                and clean_status.get("coach_command_queue_cleared_on_reset") is True
                and clean_status.get("pending_coach_command") is None
                and clean_status.get("selected_bridge_command") is None
                and clean_status.get("last_executed_coach_command_id") is None
                and clean_status.get("startup_command_blocked") is False
            ),
            json.dumps(
                {
                    "stale_detected": clean_stale_detected,
                    "coach_command_queue_cleared_on_reset": clean_status.get("coach_command_queue_cleared_on_reset"),
                    "pending_coach_command": clean_status.get("pending_coach_command"),
                    "selected_bridge_command": clean_status.get("selected_bridge_command"),
                    "last_executed_coach_command_id": clean_status.get("last_executed_coach_command_id"),
                    "startup_command_blocked": clean_status.get("startup_command_blocked"),
                },
                sort_keys=True,
            ),
        )
        setattr(env, "_last_observation", observation)
        no_cmd_observation, _reward, _terminated, _truncated, no_cmd_info = clean_hook.step_env(env, 0)
        no_cmd_result = no_cmd_info.get("action_result", {}) if isinstance(no_cmd_info, dict) else {}
        record(
            "startup_no_automatic_fusion_before_fresh_command",
            (
                not bool(no_cmd_result.get("fusionAttempted"))
                and not bool(no_cmd_result.get("fusionSucceeded"))
                and clean_hook.live_status_fields().get("selected_bridge_command") is None
                and clean_hook.live_status_fields().get("pending_coach_command") is None
            ),
            (
                f"fusionAttempted={no_cmd_result.get('fusionAttempted')}, "
                f"fusionSucceeded={no_cmd_result.get('fusionSucceeded')}, "
                f"selected={clean_hook.live_status_fields().get('selected_bridge_command')}"
            ),
        )
        observation = no_cmd_observation if isinstance(no_cmd_observation, dict) else env.observe()
    except Exception as exc:
        record("startup_queue_clear_runtime", False, str(exc))

    try:
        scoped_observation = env.observe()
        peashooter_slot = slot_for_plant_type(scoped_observation, 0)
        sunflower_slot = slot_for_plant_type(scoped_observation, 1)
        if peashooter_slot is None or sunflower_slot is None:
            record(
                "coach_tile_scoped_fusion_setup",
                False,
                f"missing_slots peashooter={peashooter_slot is not None}, sunflower={sunflower_slot is not None}",
            )
        else:
            peashooter_slot_index = int(peashooter_slot.get("slotIndex", -1))
            sunflower_slot_index = int(sunflower_slot.get("slotIndex", -1))
            peashooter_cost = int(peashooter_slot.get("seedCost", 0) or 0)
            sunflower_cost = int(sunflower_slot.get("seedCost", 0) or 0)
            if peashooter_slot_index < 0 or sunflower_slot_index < 0:
                record(
                    "coach_tile_scoped_fusion_setup",
                    False,
                    f"invalid_slot_indices peashooter={peashooter_slot_index}, sunflower={sunflower_slot_index}",
                )
            else:
                scoped_observation, wait_steps_1, wait_reason_1 = wait_until_slot_ready(
                    peashooter_slot_index,
                    min_sun=peashooter_cost,
                )
                empty_cells = first_empty_cells(scoped_observation, 2)
                if wait_reason_1 or len(empty_cells) < 2:
                    record(
                        "coach_tile_scoped_fusion_setup",
                        False,
                        f"wait_reason={wait_reason_1}, wait_steps={wait_steps_1}, empty_cells={empty_cells}",
                    )
                else:
                    source_cell = empty_cells[0]
                    control_cell = empty_cells[1]
                    first_action = placement_action_for(peashooter_slot_index, source_cell[0], source_cell[1], scoped_observation)
                    scoped_observation, _reward, _terminated, _truncated, first_place_info = env.step(first_action)
                    first_action_result = first_place_info.get("action_result", {}) if isinstance(first_place_info, dict) else {}
                    first_placed = bool(first_action_result.get("plantPlaced")) and not bool(first_action_result.get("illegalAction"))
                    if not first_placed:
                        record(
                            "coach_tile_scoped_fusion_setup",
                            False,
                            f"first_place_failed action={first_action}, illegal={first_action_result.get('illegalAction')}, reason={first_action_result.get('illegalReason')}",
                        )
                    else:
                        scoped_observation, wait_steps_2, wait_reason_2 = wait_until_slot_ready(
                            peashooter_slot_index,
                            min_sun=peashooter_cost + sunflower_cost,
                        )
                        second_action = placement_action_for(peashooter_slot_index, control_cell[0], control_cell[1], scoped_observation)
                        scoped_observation, _reward, _terminated, _truncated, second_place_info = env.step(second_action)
                        second_action_result = second_place_info.get("action_result", {}) if isinstance(second_place_info, dict) else {}
                        second_placed = bool(second_action_result.get("plantPlaced")) and not bool(second_action_result.get("illegalAction"))
                        if wait_reason_2 or not second_placed:
                            record(
                                "coach_tile_scoped_fusion_setup",
                                False,
                                (
                                    f"wait_reason={wait_reason_2}, wait_steps={wait_steps_2}, "
                                    f"second_place_failed action={second_action}, illegal={second_action_result.get('illegalAction')}, "
                                    f"reason={second_action_result.get('illegalReason')}"
                                ),
                            )
                        else:
                            source_before = plant_at_cell(scoped_observation, source_cell[0], source_cell[1])
                            control_before = plant_at_cell(scoped_observation, control_cell[0], control_cell[1])
                            if source_before is None or control_before is None:
                                record(
                                    "coach_tile_scoped_fusion_setup",
                                    False,
                                    f"missing_source_or_control source={source_before}, control={control_before}",
                                )
                            else:
                                control_before_type = int(control_before.get("type", control_before.get("plantType", -1)))
                                control_before_name = str(control_before.get("typeName") or control_before.get("plantTypeName") or "")
                                control_before_id = int(control_before.get("instanceId", control_before.get("instanceID", 0)) or 0)
                                scoped_observation, wait_steps_3, wait_reason_3 = wait_until_slot_ready(
                                    sunflower_slot_index,
                                    min_sun=sunflower_cost,
                                )
                                if wait_reason_3:
                                    record(
                                        "coach_tile_scoped_fusion_setup",
                                        False,
                                        f"sunflower_wait_timeout steps={wait_steps_3}",
                                    )
                                else:
                                    if hasattr(env, "clear_coach_runtime_state"):
                                        env.clear_coach_runtime_state(
                                            queue_cleared=True,
                                            startup_command_blocked=False,
                                            reason="coach_fusion_scope_test_before_fresh_command",
                                        )
                                    scoped_observation = env.observe()
                                    source_before = plant_at_cell(scoped_observation, source_cell[0], source_cell[1]) or source_before
                                    control_before = plant_at_cell(scoped_observation, control_cell[0], control_cell[1]) or control_before
                                    source_before_id = int(source_before.get("instanceId", source_before.get("instanceID", 0)) or 0)
                                    source_before_type = int(source_before.get("type", source_before.get("plantType", -1)))
                                    command_source = QueueCoachCommandSource()
                                    coach_hook = HumanCoachOverrideHook(
                                        enabled=True,
                                        source=command_source,
                                        fusion_enabled=True,
                                        platform="mock",
                                    )
                                    fresh_clear_stale = coach_hook.clear_pending_state(
                                        clear_source=True,
                                        reason="coach_fusion_scope_test_fresh_boundary",
                                    )
                                    command_source.submit(f"!fuse {sunflower_slot_index} {source_cell[0]} {source_cell[1]}")
                                    setattr(env, "_last_observation", scoped_observation)
                                    post_observation, _reward, _terminated, _truncated, info = coach_hook.step_env(env, 0)
                                    action_result = info.get("action_result", {}) if isinstance(info, dict) else {}
                                    changed_tile_count = int(
                                        action_result.get("changedTileCount")
                                        or action_result.get("changed_tile_count")
                                        or 0
                                    )
                                    changed_tiles = action_result.get("changedTiles")
                                    if not isinstance(changed_tiles, list):
                                        changed_tiles = action_result.get("changed_tiles")
                                    if not isinstance(changed_tiles, list):
                                        changed_tiles = []
                                    non_source_tiles_changed = bool(
                                        action_result.get("nonSourceTilesChanged")
                                        if "nonSourceTilesChanged" in action_result
                                        else action_result.get("non_source_tiles_changed")
                                    )
                                    global_side_effect = bool(
                                        action_result.get("globalFusionSideEffect")
                                        if "globalFusionSideEffect" in action_result
                                        else action_result.get("global_fusion_side_effect")
                                    )
                                    fusion_scope_value = str(
                                        action_result.get("fusionScope")
                                        or action_result.get("fusion_scope")
                                        or ""
                                    )
                                    post_payload = post_observation if isinstance(post_observation, dict) else env.observe()
                                    control_after = plant_at_cell(post_payload, control_cell[0], control_cell[1])
                                    control_after_type = int(control_after.get("type", control_after.get("plantType", -1))) if isinstance(control_after, dict) else -1
                                    control_after_name = str(control_after.get("typeName") or control_after.get("plantTypeName") or "") if isinstance(control_after, dict) else ""
                                    control_after_id = int(control_after.get("instanceId", control_after.get("instanceID", 0)) or 0) if isinstance(control_after, dict) else 0
                                    source_tile_changed_only = (
                                        changed_tile_count == 1
                                        and len(changed_tiles) == 1
                                        and int(value_int(changed_tiles[0], "row", "source_row", default=-1)) == int(source_cell[0])
                                        and int(value_int(changed_tiles[0], "column", "col", "source_col", default=-1)) == int(source_cell[1])
                                        and not non_source_tiles_changed
                                        and not global_side_effect
                                    )
                                    control_unchanged = (
                                        isinstance(control_after, dict)
                                        and control_after_type == control_before_type
                                        and control_after_name == control_before_name
                                        and (control_before_id <= 0 or control_after_id == control_before_id)
                                    )
                                    hook_status = coach_hook.live_status_fields()
                                    record(
                                        "coach_fuse_command_uses_fresh_single_bridge_command",
                                        (
                                            not bool(fresh_clear_stale)
                                            and isinstance(info.get("human_coach"), dict)
                                            and isinstance(info.get("human_coach", {}).get("selected_bridge_command"), dict)
                                            and info.get("human_coach", {}).get("selected_bridge_command", {}).get("command") == "fusion_step"
                                            and hook_status.get("selected_bridge_command") is None
                                            and hook_status.get("pending_coach_command") is None
                                            and hook_status.get("last_executed_coach_command_id") is not None
                                        ),
                                        (
                                            f"fresh_clear_stale={fresh_clear_stale}, "
                                            f"selected={info.get('human_coach', {}).get('selected_bridge_command')}, "
                                            f"status_selected={hook_status.get('selected_bridge_command')}, "
                                            f"last_id={hook_status.get('last_executed_coach_command_id')}"
                                        ),
                                    )
                                    record(
                                        "coach_tile_scoped_fusion_only_changes_requested_tile",
                                        bool(
                                            action_result.get("fusionSucceeded")
                                            and source_tile_changed_only
                                            and fusion_scope_value == "tile_scoped"
                                        ),
                                        (
                                            f"succeeded={action_result.get('fusionSucceeded')}, changed_count={changed_tile_count}, "
                                            f"changed_tiles={changed_tiles}, non_source={non_source_tiles_changed}, "
                                            f"global_side_effect={global_side_effect}, fusion_scope={fusion_scope_value}, "
                                            f"bridge_reason={action_result.get('bridgeResultReason')}, "
                                            f"source_before={source_before_type}:{source_before_id}"
                                        ),
                                    )
                                    record(
                                        "coach_tile_scoped_fusion_preserves_second_source_plant",
                                        control_unchanged,
                                        (
                                            f"before={control_before_type}:{control_before_name}:{control_before_id}, "
                                            f"after={control_after_type}:{control_after_name}:{control_after_id}, "
                                            f"source_cell={source_cell}, control_cell={control_cell}"
                                        ),
                                    )
    except Exception as exc:
        record("coach_tile_scoped_fusion_runtime", False, str(exc))

    print_smoke_results(checks)
    return all(ok for _, ok, _ in checks)


def reset_cleanup_test(env: PvZGymEnv, episodes: int) -> bool:
    checks: List[Tuple[str, bool, str]] = []

    def record(name: str, ok: bool, message: str = "") -> None:
        checks.append((name, ok, message))

    def plant_actions(observation: Dict[str, Any]) -> List[int]:
        return [action for action in env.legal_actions(observation) if action > 0]

    try:
        env.configure()
    except Exception as exc:
        record("configure", False, str(exc))
        print_smoke_results(checks)
        return False

    for episode in range(episodes):
        try:
            observation, reset_info = env.reset()
            validate_observation(observation)
            cleanup_ok, cleanup_message = verify_reset_cleanup_state(env, observation, require_mowers=True)
            reset_payload = reset_info.get("reset", {})
            record(
                f"episode_{episode}_initial_cleanup",
                cleanup_ok and int(observation.get("plantCount", 0)) == 0,
                (
                    f"{cleanup_message}, method={reset_payload.get('methodUsed')}, "
                    f"cleanupSuccess={reset_payload.get('cleanupSuccess')}"
                ),
            )
            if not cleanup_ok:
                continue
        except Exception as exc:
            record(f"episode_{episode}_initial_reset", False, str(exc))
            continue

        placed_actions: List[int] = []
        try:
            for placement_index in range(2):
                legal = plant_actions(observation)
                if not legal:
                    break
                action = legal[0]
                observation, _, _, _, info = env.step(action)
                action_result = info.get("action_result", {})
                placement = action_result.get("placement") or {}
                if action_result.get("plantPlaced") or placement.get("success"):
                    placed_actions.append(action)
            logical_before = int(observation.get("plantCount", 0))
            visible_before = int(observation.get("visiblePlantObjectCount", logical_before))
            record(
                f"episode_{episode}_placed_multiple_before_reset",
                len(placed_actions) >= 2 and logical_before >= 2 and visible_before >= 2,
                f"placed={placed_actions}, logical={logical_before}, visible={visible_before}",
            )
        except Exception as exc:
            record(f"episode_{episode}_place_before_reset", False, str(exc))
            continue

        try:
            observation, reset_info = env.reset()
            validate_observation(observation)
            cleanup_ok, cleanup_message = verify_reset_cleanup_state(env, observation, require_mowers=True)
            logical_after = int(observation.get("plantCount", 0))
            visible_after = int(observation.get("visiblePlantObjectCount", logical_after))
            stale_after = int(observation.get("staleVisiblePlantObjectCount", 0))
            record(
                f"episode_{episode}_old_plants_removed_after_reset",
                cleanup_ok and logical_after == 0 and visible_after == 0 and stale_after == 0,
                f"logical={logical_after}, visible={visible_after}, stale={stale_after}, {cleanup_message}",
            )
        except Exception as exc:
            record(f"episode_{episode}_reset_after_plants", False, str(exc))
            continue

        try:
            legal = plant_actions(observation)
            if not legal:
                record(f"episode_{episode}_new_plant_after_reset", False, "no legal plant actions after reset")
                continue
            action = legal[0]
            observation, _, _, _, info = env.step(action)
            action_result = info.get("action_result", {})
            placement = action_result.get("placement") or {}
            logical_new = int(observation.get("plantCount", 0))
            visible_new = int(observation.get("visiblePlantObjectCount", logical_new))
            stale_new = int(observation.get("staleVisiblePlantObjectCount", 0))
            occupied_ok, occupied_message = verify_legal_actions_exclude_occupied(env, observation)
            record(
                f"episode_{episode}_single_new_plant_no_overlap",
                bool(action_result.get("plantPlaced") or placement.get("success"))
                and logical_new == 1
                and visible_new == 1
                and stale_new == 0
                and occupied_ok,
                f"action={action}, logical={logical_new}, visible={visible_new}, stale={stale_new}, {occupied_message}",
            )
        except Exception as exc:
            record(f"episode_{episode}_new_plant_after_reset", False, str(exc))

    print("Reset Cleanup Test")
    print("------------------")
    for name, ok, message in checks:
        status = "PASS" if ok else "FAIL"
        suffix = f" - {message}" if message else ""
        print(f"{status:4} {name}{suffix}")
    return all(ok for _, ok, _ in checks)


def auto_select_seeds_test(env: PvZGymEnv, seed_list: List[str], episodes: int) -> bool:
    checks: List[Tuple[str, bool, str]] = []
    expected_sequence = resolve_seed_list(seed_list)
    expected_counts = count_values(expected_sequence)

    def record(name: str, ok: bool, message: str = "") -> None:
        checks.append((name, ok, message))

    for episode in range(episodes):
        try:
            env.configure()
            probe_before = env.seed_probe()
            selection = env.auto_select_seeds(seed_list=seed_list, start_level=True)
            verification = selection.get("verification", {})
            selected_counts = count_values([int(value) for value in verification.get("selectedSeedTypes", [])])
            missing = [int(value) for value in verification.get("missingSeedTypes", [])]
            record(
                f"episode_{episode}_auto_select_verified",
                bool(selection.get("ok")) and counts_cover(selected_counts, expected_counts) and not missing,
                f"expected={format_counts(expected_counts)}, selected={format_counts(selected_counts)}, missing={missing}, seedSelectionActive={probe_before.get('seedSelectionActive')}",
            )
            observation = env.wait_for_gameplay_ready(
                timeout=env.config.reset_wait_timeout,
                poll_seconds=env.config.reset_poll_seconds,
                quiet=True,
                fail_on_terminal=False,
            )
            probe_after = env.seed_probe()
            active_counts = type_counts_from_probe(
                probe_after,
                "activeGameplayCardBankCards",
                "activeGameplayCardBankPlantTypeCounts",
            )
            record(
                f"episode_{episode}_gameplay_ready_after_seed_select",
                bool(observation.get("gameplayReady")) and counts_cover(active_counts, expected_counts),
                f"gameplayReady={observation.get('gameplayReady')}, legalActions={len(env.legal_actions(observation))}, activeBank={format_counts(active_counts)}",
            )
        except Exception as exc:
            record(f"episode_{episode}_auto_select_runtime", False, str(exc))
            break

    print("Auto Select Seeds Test")
    print("----------------------")
    for name, ok, message in checks:
        status = "PASS" if ok else "FAIL"
        suffix = f" - {message}" if message else ""
        print(f"{status:4} {name}{suffix}")
    return all(ok for _, ok, _ in checks)


def fresh_seed_select_test(env: PvZGymEnv, seed_list: List[str]) -> bool:
    checks: List[Tuple[str, bool, str]] = []
    requested_sequence = resolve_seed_list(seed_list)
    expected_counts = count_values(requested_sequence)
    expected_total = len(requested_sequence)

    def record(name: str, ok: bool, message: str = "") -> None:
        checks.append((name, ok, message))

    def print_results() -> bool:
        print("Fresh Seed Select Test")
        print("----------------------")
        for name, ok, message in checks:
            status = "PASS" if ok else "FAIL"
            suffix = f" - {message}" if message else ""
            print(f"{status:4} {name}{suffix}")
        return all(ok for _, ok, _ in checks)

    def chooser_ready(probe: Dict[str, Any]) -> Tuple[bool, Counter, Counter, int]:
        available = type_counts_from_probe(probe, "availableSeedCards", "availableCardPlantTypeCounts")
        selected = type_counts_from_probe(probe, "selectedSeedBankCards", "selectedBankPlantTypeCounts")
        selected_count = int(probe.get("selectedBankVisibleCount", sum(selected.values())))
        available_ok = all(available.get(plant_type, 0) > 0 for plant_type in expected_counts)
        ready = (
            bool(probe.get("seedSelectionActive"))
            and bool(probe.get("seedSelectionPanelActive"))
            and bool(probe.get("startButtonActive"))
            and selected_count == 0
            and available_ok
        )
        return ready, available, selected, selected_count

    try:
        env.configure()
        before = env.seed_probe()
        available_start = type_counts_from_probe(before, "availableSeedCards", "availableCardPlantTypeCounts")
        selected_start = type_counts_from_probe(before, "selectedSeedBankCards", "selectedBankPlantTypeCounts")
        selected_start_count = int(before.get("selectedBankVisibleCount", sum(selected_start.values())))
        start_ok = True
        start_ok &= bool(before.get("seedSelectionActive"))
        record(
            "fresh_seed_selection_active_start",
            bool(before.get("seedSelectionActive")),
            f"seedSelectionActive={before.get('seedSelectionActive')}, seedSelectionPanelActive={before.get('seedSelectionPanelActive')}, chooseYourPlantsTextActive={before.get('chooseYourPlantsTextActive')}",
        )
        lets_rock_visible = bool(before.get("startButtonActive"))
        start_ok &= lets_rock_visible
        record(
            "fresh_lets_rock_visible_start",
            lets_rock_visible,
            f"startButtonActive={before.get('startButtonActive')}",
        )
        start_ok &= selected_start_count == 0
        record(
            "fresh_selected_bank_empty_start",
            selected_start_count == 0,
            f"selectedBankVisibleCount={selected_start_count}, selectedBank={format_counts(selected_start)}",
        )
        available_ok = all(available_start.get(plant_type, 0) > 0 for plant_type in expected_counts)
        start_ok &= available_ok
        record(
            "fresh_available_cards_visible_start",
            available_ok,
            f"required={format_counts(Counter({plant_type: 1 for plant_type in expected_counts}))}, available={format_counts(available_start)}",
        )
        if not start_ok:
            record(
                "fresh_state_required",
                False,
                "Non-fresh state: this command requires the visible Choose Your Plants screen with an empty selected bank.",
            )
            return print_results()

        stable_probe = before
        stable_available = available_start
        stable_selected = selected_start
        stable_selected_count = selected_start_count
        consecutive_ready = 0
        stable_ok = False
        deadline = time.monotonic() + max(2.0, env.config.reset_poll_seconds * 10.0)
        while time.monotonic() < deadline:
            probe = env.seed_probe()
            ready, available, selected, selected_count = chooser_ready(probe)
            if ready:
                consecutive_ready += 1
                stable_probe = probe
                stable_available = available
                stable_selected = selected
                stable_selected_count = selected_count
                if consecutive_ready >= 2:
                    stable_ok = True
                    break
            else:
                consecutive_ready = 0
            time.sleep(max(0.05, env.config.reset_poll_seconds))

        record(
            "fresh_chooser_stable_before_select",
            stable_ok,
            (
                f"seedSelectionActive={stable_probe.get('seedSelectionActive')}, "
                f"seedSelectionPanelActive={stable_probe.get('seedSelectionPanelActive')}, "
                f"startButtonActive={stable_probe.get('startButtonActive')}, "
                f"selectedBankVisibleCount={stable_selected_count}, "
                f"selectedBank={format_counts(stable_selected)}, "
                f"available={format_counts(stable_available)}"
            ),
        )
        if not stable_ok:
            return print_results()

        selection = env.auto_select_seeds(seed_list=seed_list, start_level=True)
        after_selection = selection.get("afterSelectionBeforeStart", {})
        selected_after = type_counts_from_probe(after_selection, "selectedSeedBankCards", "selectedBankPlantTypeCounts")
        selected_after_count = int(after_selection.get("selectedBankVisibleCount", sum(selected_after.values())))

        record(
            "fresh_selected_bank_count_increased",
            selected_after_count == expected_total,
            f"start=0, afterSelectionBeforeStart={selected_after_count}, expected={expected_total}",
        )
        record(
            "fresh_selected_bank_contains_requested",
            selected_after == expected_counts,
            f"expected={format_counts(expected_counts)}, selected={format_counts(selected_after)}",
        )
        record(
            "fresh_pre_start_still_in_seed_selection",
            bool(after_selection.get("seedSelectionActive")) and not bool(after_selection.get("gameplayReady")),
            f"seedSelectionActive={after_selection.get('seedSelectionActive')}, gameplayReady={after_selection.get('gameplayReady')}",
        )
        attempts = list(selection.get("selectionAttempts", []))
        steps = list(selection.get("selectionSteps", []))
        attempt_ok = len(steps) == expected_total and all(
            step.get("success")
            and any(
                attempt.get("success")
                and attempt.get("methodUsed") == "CardUI.OnMouseDown()"
                and attempt.get("selectedBankCountIncreased")
                and attempt.get("selectedBankTypeCountIncreased")
                for attempt in step.get("attempts", [])
            )
            for step in steps
        )
        record(
            "fresh_cardui_onmousedown_selected_cards",
            attempt_ok,
            f"steps={[(s.get('plantType'), s.get('plantTypeName'), s.get('success'), [(a.get('attemptNumber'), a.get('selectedBankCountBefore'), a.get('selectedBankCountAfter'), a.get('success')) for a in s.get('attempts', [])]) for s in steps]}",
        )
        duplicate_requested = any(count > 1 for count in expected_counts.values())
        if duplicate_requested:
            duplicate_attempts = [attempt for attempt in attempts if int(attempt.get("duplicateSelectionIndex", 0)) > 0]
            duplicate_cost_detected = any(bool(attempt.get("duplicateCostIncreaseDetected")) for attempt in duplicate_attempts)
            record(
                "fresh_duplicate_cost_increase_detected",
                duplicate_cost_detected,
                f"duplicateAttempts={[(a.get('plantType'), a.get('visibleCostsBefore'), a.get('visibleCostsAfter'), a.get('duplicateCostIncreaseDetected')) for a in duplicate_attempts]}",
            )

        record(
            "fresh_auto_select_response_ok",
            bool(selection.get("ok")),
            f"ok={selection.get('ok')}, missing={selection.get('missingSeedTypes')}, startInvoked={selection.get('startInvoked')}, startLog={selection.get('startLog')}",
        )
        observation = env.wait_for_gameplay_ready(
            timeout=env.config.reset_wait_timeout,
            poll_seconds=env.config.reset_poll_seconds,
            quiet=True,
            fail_on_terminal=False,
        )
        final_probe = env.seed_probe()
        active_counts = type_counts_from_probe(
            final_probe,
            "activeGameplayCardBankCards",
            "activeGameplayCardBankPlantTypeCounts",
        )
        final_ok = (
            bool(observation.get("gameplayReady") or final_probe.get("gameplayReady"))
            and not bool(final_probe.get("seedSelectionActive"))
            and counts_cover(active_counts, expected_counts)
        )
        deadline = time.monotonic() + max(8.0, env.config.reset_poll_seconds * 40.0)
        while not final_ok and time.monotonic() < deadline:
            time.sleep(max(0.05, env.config.reset_poll_seconds))
            observation = env.observe()
            final_probe = env.seed_probe()
            active_counts = type_counts_from_probe(
                final_probe,
                "activeGameplayCardBankCards",
                "activeGameplayCardBankPlantTypeCounts",
            )
            final_ok = (
                bool(observation.get("gameplayReady") or final_probe.get("gameplayReady"))
                and not bool(final_probe.get("seedSelectionActive"))
                and counts_cover(active_counts, expected_counts)
            )
        record(
            "fresh_gameplay_ready_after_verified_start",
            final_ok,
            f"gameplayReady={observation.get('gameplayReady')}, seedSelectionActive={final_probe.get('seedSelectionActive')}, activeBank={format_counts(active_counts)}, legalActions={len(env.legal_actions(observation))}",
        )
    except Exception as exc:
        record("fresh_seed_select_runtime", False, str(exc))

    return print_results()


def seed_screen_gating_test(env: PvZGymEnv) -> bool:
    checks: List[Tuple[str, bool, str]] = []

    def record(name: str, ok: bool, message: str = "") -> None:
        checks.append((name, ok, message))

    print("Seed Screen Gating Test")
    print("-----------------------")
    try:
        probe = env.seed_probe()
        observation = env.observe()
        legal = env.legal_actions(observation)
        raw_legal = env.client.request("legal_actions")
        fresh_active = bool(probe.get("seedSelectionActive")) and bool(observation.get("seedSelectionActive"))
        record(
            "seed_screen_active",
            fresh_active,
            f"probe.seedSelectionActive={probe.get('seedSelectionActive')}, obs.seedSelectionActive={observation.get('seedSelectionActive')}",
        )
        record(
            "seed_screen_gameplay_not_ready",
            not bool(observation.get("gameplayReady")) and not bool(observation.get("actualGameplayReady")),
            f"gameplayReady={observation.get('gameplayReady')}, actualGameplayReady={observation.get('actualGameplayReady')}",
        )
        record(
            "seed_screen_legal_wait_only",
            legal == [0] and raw_legal.get("legalActions") == [0],
            f"wrapperLegal={legal}, bridgeLegal={raw_legal.get('legalActions')}, reason={raw_legal.get('reason')}",
        )
        _, _, _, _, info = env.step(1)
        action_result = info.get("action_result", {})
        action_audit = action_result.get("actionAudit") if isinstance(action_result.get("actionAudit"), dict) else {}
        record(
            "seed_screen_plant_action_rejected",
            (
                bool(action_result.get("illegalAction")) and action_result.get("illegalReason") == "seed_selection_active"
            )
            or (
                bool(action_result.get("preStepMaskBlockedAction"))
                and action_audit.get("pythonFilterReason") in {"seed_selection_active", "gameplay_not_ready"}
            ),
            (
                f"illegal={action_result.get('illegalAction')}, reason={action_result.get('illegalReason')}, "
                f"pre_step_blocked={action_result.get('preStepMaskBlockedAction')}, filter={action_audit.get('pythonFilterReason')}"
            ),
        )
        if fresh_active:
            selection = env.auto_select_seeds(seed_list=env.config.seed_list, start_level=True)
            after_obs = env.wait_for_gameplay_ready(
                timeout=env.config.reset_wait_timeout,
                poll_seconds=env.config.reset_poll_seconds,
                quiet=True,
                fail_on_terminal=False,
            )
            after_probe = env.seed_probe()
            after_legal = env.legal_actions(after_obs)
            active_counts, _ = active_gameplay_bank_state(after_probe)
            requested_counts = count_values(resolve_seed_list(env.config.seed_list))
            record(
                "seed_screen_auto_select_ok",
                bool(selection.get("ok")),
                f"ok={selection.get('ok')}, startInvoked={selection.get('startInvoked')}",
            )
            record(
                "post_start_gameplay_ready",
                bool(after_obs.get("gameplayReady")) and not bool(after_obs.get("seedSelectionActive")),
                f"gameplayReady={after_obs.get('gameplayReady')}, seedSelectionActive={after_obs.get('seedSelectionActive')}",
            )
            record(
                "post_start_legal_actions_expanded",
                len(after_legal) > 1 and counts_cover(active_counts, requested_counts),
                f"legalActions={len(after_legal)}, activeBank={format_counts(active_counts)}",
            )
    except Exception as exc:
        record("seed_screen_gating_runtime", False, str(exc))

    for name, ok, message in checks:
        status = "PASS" if ok else "FAIL"
        suffix = f" - {message}" if message else ""
        print(f"{status:4} {name}{suffix}")
    return all(ok for _, ok, _ in checks)


def restart_screen_detection_test(env: PvZGymEnv, fast_only: bool = False) -> bool:
    observation = env.screen_state_fast() if fast_only else env.observe(debug_observation=True, force_restart_probe=True)
    terminal_reason = "game_over_restart_screen" if is_restart_screen_observation(observation) else ""
    fields = {
        "boardFound": observation.get("boardFound"),
        "canReadBoard": observation.get("canReadBoard"),
        "gameplayReady": observation.get("gameplayReady"),
        "actualGameplayReady": observation.get("actualGameplayReady"),
        "done": observation.get("done"),
        "over": observation.get("over"),
        "onGameOverScreen": observation.get("onGameOverScreen"),
        "lossMenuActive": observation.get("lossMenuActive"),
        "gameOverTextVisible": observation.get("gameOverTextVisible"),
        "onRestartScreen": observation.get("onRestartScreen"),
        "restartButtonActive": observation.get("restartButtonActive"),
        "restartButtonName": observation.get("restartButtonName"),
        "restartButtonPath": observation.get("restartButtonPath"),
        "restartDetectionReason": observation.get("restartDetectionReason"),
        "restartDetectionMode": observation.get("restartDetectionMode"),
        "onPauseMenu": observation.get("onPauseMenu"),
        "pauseMenuActive": observation.get("pauseMenuActive"),
        "pauseRestartButtonActive": observation.get("pauseRestartButtonActive"),
        "terminal_reason": terminal_reason,
        "legalActionReason": observation.get("legalActionReason"),
        "legalActions": observation.get("legalActions"),
        "nextStep": observation.get("nextStep"),
        "bridge_observe_ms": observation.get("bridge_observe_ms"),
        "screen_check_ms": observation.get("screen_check_ms"),
        "ui_scan_ms": observation.get("ui_scan_ms"),
    }
    print("Restart Screen Detection Test")
    print("-----------------------------")
    print(json.dumps(fields, indent=2))
    ok = (
        bool(observation.get("onGameOverScreen"))
        and (bool(observation.get("lossMenuActive")) or bool(observation.get("gameOverTextVisible")))
        and not bool(observation.get("onPauseMenu") or observation.get("pauseMenuActive"))
        and terminal_reason == "game_over_restart_screen"
        and (fast_only or observation.get("legalActions") == [0])
        and (fast_only or observation.get("nextStep") == "click_restart")
    )
    print(("PASS" if ok else "FAIL") + " restart_screen_detected")
    return ok


def pause_menu_restart_test(env: PvZGymEnv) -> bool:
    observation = env.screen_state_fast()
    terminal_reason = "game_over_restart_screen" if is_restart_screen_observation(observation) else ""
    fields = {
        "boardFound": observation.get("boardFound"),
        "gameplayReady": observation.get("gameplayReady"),
        "over": observation.get("over"),
        "onPauseMenu": observation.get("onPauseMenu"),
        "pauseMenuActive": observation.get("pauseMenuActive"),
        "pauseRestartButtonActive": observation.get("pauseRestartButtonActive"),
        "onGameOverScreen": observation.get("onGameOverScreen"),
        "lossMenuActive": observation.get("lossMenuActive"),
        "gameOverTextVisible": observation.get("gameOverTextVisible"),
        "onRestartScreen": observation.get("onRestartScreen"),
        "restartButtonActive": observation.get("restartButtonActive"),
        "restartDetectionMode": observation.get("restartDetectionMode"),
        "terminal_reason": terminal_reason,
        "screen_check_ms": observation.get("screen_check_ms"),
    }
    print("Pause Menu Restart Test")
    print("-----------------------")
    print(json.dumps(fields, indent=2))
    ok = (
        bool(observation.get("onPauseMenu") or observation.get("pauseMenuActive"))
        and not bool(observation.get("onGameOverScreen"))
        and not bool(observation.get("onRestartScreen"))
        and not bool(observation.get("restartButtonActive"))
        and terminal_reason != "game_over_restart_screen"
    )
    print(("PASS" if ok else "FAIL") + " pause_menu_restart_ignored")
    return ok


def print_smoke_results(checks: List[Tuple[str, bool, str]]) -> None:
    print("Smoke Test")
    print("----------")
    for name, ok, message in checks:
        status = "PASS" if ok else "FAIL"
        suffix = f" - {message}" if message else ""
        print(f"{status:4} {name}{suffix}")


def select_action(env: PvZGymEnv, policy: str, observation: Dict[str, Any], rng: random.Random) -> int:
    legal = env.legal_actions(observation)
    if policy == "wait":
        return 0
    if policy == "random":
        return rng.choice(legal or [0])
    if policy == "teacher":
        return env.teacher_action(observation)
    raise ValueError(f"unknown policy: {policy}")


LANE_EPISODE_DICT_FIELDS = (
    "plants_by_row",
    "peashooters_by_row",
    "sunflowers_by_row",
    "threat_steps_by_row",
    "undefended_threat_steps_by_row",
    "undefended_threat_age_avg_by_row",
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
    "peashooters_per_threat_step_by_row",
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
)


def _add_row_counts(target: Dict[str, int], values: Any) -> None:
    if not isinstance(values, dict):
        return
    for key, value in values.items():
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        target[str(key)] = int(target.get(str(key), 0)) + count


def _update_row_max(target: Dict[str, int], values: Any) -> None:
    if not isinstance(values, dict):
        return
    for key, value in values.items():
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        target[str(key)] = max(int(target.get(str(key), 0)), count)


def _bump_row_counter(target: Dict[str, int], row: int) -> None:
    if row < 0:
        return
    key = str(row)
    target[key] = int(target.get(key, 0)) + 1


def _ratio_by_row(numerator: Dict[str, int], denominator: Dict[str, int]) -> Dict[str, float]:
    rows = sorted(
        set(str(key) for key in numerator.keys()) | set(str(key) for key in denominator.keys()),
        key=lambda item: (0, int(item)) if item.isdigit() else (1, item),
    )
    return {
        row: (float(numerator.get(row, 0)) / float(denominator.get(row, 0)))
        if int(denominator.get(row, 0) or 0) > 0
        else 0.0
        for row in rows
    }


def accumulate_reward_episode_totals(log: EpisodeLog, info: Dict[str, Any]) -> None:
    breakdown = info.get("reward_breakdown") if isinstance(info, dict) else {}
    if not isinstance(breakdown, dict):
        return
    component_sum = 0.0
    for component in REWARD_COMPONENT_FIELDS:
        field_name = f"{component}_total"
        try:
            value = float(breakdown.get(component) or 0.0)
        except (TypeError, ValueError):
            continue
        setattr(log, field_name, float(getattr(log, field_name, 0.0)) + value)
        component_sum += value
    log.reward_component_total += component_sum
    try:
        step_total = float(breakdown.get("reward_total") or 0.0)
    except (TypeError, ValueError):
        step_total = component_sum
    log.reward_unattributed_adjustment_total += step_total - component_sum
    diagnostics = info.get("reward_diagnostics")
    if isinstance(diagnostics, dict):
        log.reward_policy_version = str(
            diagnostics.get("reward_policy_version") or REWARD_POLICY_VERSION
        )
        for field_name in (
            "threat_raw_before",
            "threat_raw_after",
            "threat_before",
            "threat_after",
            "threat_raw_delta",
            "threat_normalized_delta",
            "threat_clipped_delta",
        ):
            try:
                setattr(log, field_name, float(diagnostics.get(field_name) or 0.0))
            except (TypeError, ValueError):
                pass


def accumulate_lane_episode_diagnostics(log: EpisodeLog, action: int, info: Dict[str, Any]) -> None:
    diag = info.get("lane_diagnostics") if isinstance(info, dict) else {}
    if not isinstance(diag, dict):
        diag = {}

    log.plants_by_row = dict(diag.get("plants_by_row") or log.plants_by_row)
    log.peashooters_by_row = dict(diag.get("peashooters_by_row") or log.peashooters_by_row)
    log.sunflowers_by_row = dict(diag.get("sunflowers_by_row") or log.sunflowers_by_row)
    _add_row_counts(log.threat_steps_by_row, diag.get("threat_steps_by_row"))
    _add_row_counts(log.undefended_threat_steps_by_row, diag.get("undefended_threat_steps_by_row"))
    age_sum_by_row = getattr(log, "_undefended_threat_age_sum_by_row", {})
    age_count_by_row = getattr(log, "_undefended_threat_age_count_by_row", {})
    _add_row_counts(age_sum_by_row, diag.get("undefended_threat_age_sum_by_row"))
    _add_row_counts(age_count_by_row, diag.get("undefended_threat_age_count_by_row"))
    setattr(log, "_undefended_threat_age_sum_by_row", age_sum_by_row)
    setattr(log, "_undefended_threat_age_count_by_row", age_count_by_row)
    _update_row_max(log.undefended_threat_age_max_by_row, diag.get("undefended_threat_age_max_by_row"))
    _add_row_counts(log.mower_losses_by_row, diag.get("mower_losses_by_row"))
    _add_row_counts(log.legal_peashooter_actions_by_row, diag.get("legal_peashooter_actions_by_row"))
    _add_row_counts(log.peashooter_available_but_waited_by_row, diag.get("peashooter_available_but_waited_by_row"))
    _add_row_counts(
        log.peashooter_available_but_planted_elsewhere_by_row,
        diag.get("peashooter_available_but_planted_elsewhere_by_row"),
    )
    _add_row_counts(log.sunflower_while_undefended_threat_by_row, diag.get("sunflower_while_undefended_threat_by_row"))
    _add_row_counts(log.row_defense_opportunities_by_row, diag.get("row_defense_opportunities_by_row"))
    _add_row_counts(log.row_defense_responses_by_row, diag.get("row_defense_responses_by_row"))
    _add_row_counts(
        log.threatened_rows_with_zero_defender_steps_by_row,
        diag.get("threatened_rows_with_zero_defender_steps_by_row"),
    )
    _add_row_counts(log.legal_actions_by_seed_slot, diag.get("legal_actions_by_seed_slot"))
    _add_row_counts(log.bridge_legal_actions_by_seed_slot, diag.get("bridge_legal_actions_by_seed_slot"))
    _add_row_counts(log.python_mask_block_reason_counts, diag.get("python_mask_block_reason_counts"))
    _add_row_counts(log.wallnut_placements_by_row, diag.get("wallnut_placements_by_row"))
    _add_row_counts(log.wallnut_placements_by_col, diag.get("wallnut_placements_by_col"))
    _add_row_counts(log.mower_risk_steps_by_row, diag.get("mower_risk_steps_by_row"))
    _add_row_counts(log.mower_saves_estimated_by_row, diag.get("mower_saves_estimated_by_row"))
    _add_row_counts(log.buckethead_count_by_row, diag.get("buckethead_count_by_row"))
    _add_row_counts(log.conehead_count_by_row, diag.get("conehead_count_by_row"))
    _add_row_counts(log.tough_zombie_count_by_row, diag.get("tough_zombie_count_by_row"))
    if bool(diag.get("pre_step_mask_blocked_action")):
        log.pre_step_mask_blocked_count += 1
    if bool(diag.get("cooldown_illegal_exposed_by_mask")):
        log.cooldown_illegal_exposed_by_mask_count += 1
    if bool(diag.get("mask_bridge_disagreement")):
        log.mask_bridge_disagreement_count += 1

    if bool(diag.get("plant_in_threatened_row")):
        log.plants_in_threatened_row_count += 1
    if bool(diag.get("plant_in_unthreatened_row")):
        log.plants_in_unthreatened_row_count += 1
    if bool(diag.get("overdefended_while_undefended")):
        log.overdefended_while_undefended_count += 1
    if bool(diag.get("least_defended_threatened_row_plant")):
        log.least_defended_threatened_row_plant_count += 1
    if bool(diag.get("sunflower_overbuild_before_defense")):
        log.sunflower_overbuild_before_defense_count += 1
    if bool(diag.get("wait_while_actionable_threat")):
        log.wait_while_actionable_threat_count += 1
    if bool(diag.get("wait_while_peashooter_affordable_ready")):
        log.wait_while_peashooter_affordable_ready_count += 1
    if bool(diag.get("wait_while_wallnut_affordable_ready")):
        log.wait_while_wallnut_affordable_ready_count += 1
    if bool(diag.get("wait_while_cherrybomb_affordable_ready")):
        log.wait_while_cherrybomb_affordable_ready_count += 1
    if bool(diag.get("sunflower_greed_while_defense_missing")):
        log.sunflower_greed_while_defense_missing_count += 1
    if bool(diag.get("wallnut_blocks_active_threat")):
        log.wallnut_blocks_active_threat_count += 1
    if bool(diag.get("wallnut_low_value_placement")):
        log.wallnut_low_value_placement_count += 1
    if bool(diag.get("cherrybomb_used")):
        log.cherrybomb_used_count += 1
    if bool(diag.get("cherrybomb_used_under_threat")):
        log.cherrybomb_used_under_threat_count += 1
    if bool(diag.get("cherrybomb_used_low_value")):
        log.cherrybomb_used_low_value_count += 1
    if bool(diag.get("tough_zombie_response")):
        log.tough_zombie_response_count += 1
    try:
        log.active_threat_rows_without_peashooter_count += int(diag.get("active_threat_rows_without_peashooter_count") or 0)
        log.close_zombie_with_no_defense_count += int(diag.get("close_zombie_with_no_defense_count") or 0)
        log.cherrybomb_kills_total += int(diag.get("cherrybomb_delayed_kills") or 0)
        log.cherrybomb_zero_kill_count += int(diag.get("cherrybomb_delayed_zero_kill") or 0)
        log.cherrybomb_buckethead_kills += int(diag.get("cherrybomb_buckethead_kill_credit") or 0)
        log.cherrybomb_conehead_kills += int(diag.get("cherrybomb_conehead_kill_credit") or 0)
    except (TypeError, ValueError):
        pass
    fusion_diag = info.get("fusion_diagnostics") if isinstance(info, dict) else {}
    if isinstance(fusion_diag, dict):
        merge_episode_fusion_stats(log, fusion_diag)

    try:
        log.rows_with_peashooter_count = int(diag.get("rows_with_peashooter_count") or log.rows_with_peashooter_count)
    except (TypeError, ValueError):
        pass
    try:
        log.peashooter_coverage_rate_sum += float(diag.get("peashooter_coverage_rate") or 0.0)
    except (TypeError, ValueError):
        pass

    step_number = log.episode_length + 1
    first_peashooter_row = PvZGymEnv._safe_int(diag.get("first_peashooter_row"), default=-1)
    if first_peashooter_row >= 0:
        key = str(first_peashooter_row)
        if int(log.first_peashooter_by_row_step.get(key, 0) or 0) <= 0:
            log.first_peashooter_by_row_step[key] = step_number

    first_defense_row = PvZGymEnv._safe_int(diag.get("first_defense_row"), default=-1)
    if first_defense_row >= 0:
        key = str(first_defense_row)
        if int(log.first_defense_step_by_row.get(key, 0) or 0) <= 0:
            log.first_defense_step_by_row[key] = step_number

    if bool(diag.get("all_rows_peashooter_covered")) and log.all_rows_peashooter_covered_step <= 0:
        log.all_rows_peashooter_covered_step = step_number
        try:
            log.sunflower_count_when_first_full_coverage = int(
                diag.get("sunflower_count_when_first_full_coverage")
            )
        except (TypeError, ValueError):
            log.sunflower_count_when_first_full_coverage = -1

    if bool(diag.get("wait_under_threat")):
        log.wait_under_threat_count += 1
    try:
        log.close_zombie_undefended_count += int(diag.get("close_zombie_undefended_count") or 0)
    except (TypeError, ValueError):
        pass

    illegal_reason = str(diag.get("illegal_reason") or "")
    if illegal_reason:
        log.illegal_reason_counts[illegal_reason] = int(log.illegal_reason_counts.get(illegal_reason, 0)) + 1

    action_result = info.get("action_result", {}) if isinstance(info, dict) else {}
    placement = action_result.get("placement") if isinstance(action_result, dict) and isinstance(action_result.get("placement"), dict) else {}
    executed_action = PvZGymEnv._safe_int(action_result.get("executedAction"), default=int(action)) if isinstance(action_result, dict) else int(action)
    row = PvZGymEnv._safe_int(diag.get("action_row"), placement.get("row"), default=-1)
    plant_type = PvZGymEnv._safe_int(diag.get("action_plant_type"), placement.get("plantType"), default=-1)
    plant_placed = bool(diag.get("plant_placed") or placement.get("success") or placement.get("plantPlaced"))
    action_kind = str(diag.get("action_kind") or "")

    if executed_action == 0 and action_kind != "fusion":
        log.wait_actions += 1
    else:
        log.plant_actions += 1
        _bump_row_counter(log.plant_actions_by_row, row)
        if plant_type == 0:
            _bump_row_counter(log.peashooter_actions_by_row, row)
        elif plant_type == 1:
            _bump_row_counter(log.sunflower_actions_by_row, row)

    if plant_placed:
        _bump_row_counter(log.plant_placements_by_row, row)
        if plant_type == 0:
            _bump_row_counter(log.peashooter_placements_by_row, row)
        elif plant_type == 1:
            _bump_row_counter(log.sunflower_placements_by_row, row)


def finalize_lane_episode_diagnostics(log: EpisodeLog) -> None:
    total_actions = max(1, log.wait_actions + log.plant_actions)
    log.wait_action_percent = 100.0 * log.wait_actions / total_actions
    log.plant_action_percent = 100.0 * log.plant_actions / total_actions
    log.undefended_threat_ratio_by_row = _ratio_by_row(log.undefended_threat_steps_by_row, log.threat_steps_by_row)
    log.undefended_threat_age_avg_by_row = _ratio_by_row(
        getattr(log, "_undefended_threat_age_sum_by_row", {}),
        getattr(log, "_undefended_threat_age_count_by_row", {}),
    )
    log.row_defense_response_rate_by_row = _ratio_by_row(
        log.row_defense_responses_by_row,
        log.row_defense_opportunities_by_row,
    )
    total_opportunities = sum(int(value) for value in log.row_defense_opportunities_by_row.values())
    total_responses = sum(int(value) for value in log.row_defense_responses_by_row.values())
    log.row_defense_response_rate = total_responses / total_opportunities if total_opportunities > 0 else 0.0
    placement_context_total = log.plants_in_threatened_row_count + log.plants_in_unthreatened_row_count
    log.plants_in_threatened_row_ratio = (
        log.plants_in_threatened_row_count / placement_context_total
        if placement_context_total > 0
        else 0.0
    )
    log.plants_in_unthreatened_row_ratio = (
        log.plants_in_unthreatened_row_count / placement_context_total
        if placement_context_total > 0
        else 0.0
    )
    log.peashooters_per_threat_step_by_row = _ratio_by_row(
        log.peashooter_placements_by_row,
        log.threat_steps_by_row,
    )
    log.peashooter_coverage_rate_by_step = (
        log.peashooter_coverage_rate_sum / max(1, log.episode_length)
    )
    log.cherrybomb_avg_kills_per_use = (
        float(log.cherrybomb_kills_total) / float(log.cherrybomb_used_count)
        if log.cherrybomb_used_count > 0
        else 0.0
    )
    log.fusion_candidate_count_avg = (
        float(log.fusion_candidate_count_total) / float(max(1, log.episode_length))
    )
    log.fusion_avg_kills_after_use = (
        float(log.fusion_kills_after_use_total) / float(log.fusion_success_count)
        if log.fusion_success_count > 0
        else 0.0
    )
    row_keys = sorted(
        set(log.plants_by_row.keys())
        | set(log.peashooters_by_row.keys())
        | set(log.threat_steps_by_row.keys())
        | set(log.threatened_rows_with_zero_defender_steps_by_row.keys())
        | set(log.undefended_threat_age_avg_by_row.keys())
        | set(log.undefended_threat_age_max_by_row.keys()),
        key=lambda item: (0, int(item)) if str(item).isdigit() else (1, str(item)),
    )
    for row in row_keys:
        log.first_peashooter_by_row_step.setdefault(str(row), 0)
        log.first_defense_step_by_row.setdefault(str(row), 0)
    for field_name in LANE_EPISODE_DICT_FIELDS:
        values = getattr(log, field_name)
        if isinstance(values, dict):
            setattr(
                log,
                field_name,
                dict(sorted(values.items(), key=lambda item: (0, int(item[0])) if str(item[0]).isdigit() else (1, str(item[0])))),
            )


def run_episode(
    env: PvZGymEnv,
    policy: str,
    episode_index: int,
    max_steps: int,
    rng: random.Random,
    reset_each_episode: bool = True,
) -> EpisodeLog:
    log = EpisodeLog(policy=policy, episode_index=episode_index, done_reason="none")
    try:
        if reset_each_episode:
            observation, reset_info = env.reset()
            reset_payload = reset_info.get("reset", {})
            log.reset_success = bool(reset_payload.get("resetSuccess", reset_payload.get("ok", True)))
            log.reset_seconds = float(
                reset_payload.get("timeToPlayableSeconds")
                or (float(reset_payload.get("reset_ms", 0.0) or 0.0) / 1000.0)
                or 0.0
            )
            if not reset_payload.get("ok", True):
                log.reset_failures += 1
                log.reset_success = False
        else:
            observation = env.observe()
            env.previous_observation = observation
    except Exception as exc:
        log.reset_failures += 1
        log.bridge_errors += 1
        log.done_reason = "reset_error"
        log.reset_success = False
        log.error = str(exc)
        return log

    try:
        validate_observation(observation)
    except Exception as exc:
        log.reset_failures += 1
        log.done_reason = "reset_error"
        log.error = str(exc)
        return log

    start_kills = int(observation.get("killCount", 0))
    start_mowers = int(observation.get("logicalMowerCount", observation.get("rowCount", env.config.row_count)))
    legal_action_total = 0
    log.final_wave = int(observation.get("wave", 0))
    log.max_wave = int(observation.get("maxWave", 0))
    log.sun_remaining = int(observation.get("sun", 0))

    while max_steps == 0 or log.episode_length < max_steps:
        try:
            action = select_action(env, policy, observation, rng)
            observation, reward, done, _, info = env.step(
                action,
                action_source="random" if policy == "random" else "scripted",
                action_source_metadata={"baseline_policy": str(policy)},
            )
        except Exception as exc:
            log.bridge_errors += 1
            log.done_reason = "bridge_error"
            log.error = str(exc)
            break

        action_result = info.get("action_result", {})
        placement = action_result.get("placement") or {}
        legal_actions = info.get("legal_actions") or observation.get("legalActions", [])
        if isinstance(legal_actions, list):
            legal_action_total += len(legal_actions)
        accumulate_lane_episode_diagnostics(log, action, info)
        accumulate_reward_episode_totals(log, info)

        log.episode_reward += reward
        log.episode_length += 1
        log.final_wave = int(observation.get("wave", log.final_wave))
        log.max_wave = int(observation.get("maxWave", log.max_wave))
        log.sun_remaining = int(observation.get("sun", log.sun_remaining))
        final_mowers = int(observation.get("logicalMowerCount", start_mowers))
        log.mowers_lost = max(0, start_mowers - final_mowers)

        if action_result.get("illegalAction"):
            log.illegal_actions += 1
        if placement.get("success"):
            log.plants_placed += 1
        if action_result.get("costPaid") or placement.get("costPaid"):
            sun_before = int(action_result.get("sunBefore") or placement.get("sunBefore") or 0)
            sun_after = int(action_result.get("sunAfter") or placement.get("sunAfter") or observation.get("sun", 0))
            spent = max(0, sun_before - sun_after)
            if spent == 0:
                spent = int(action_result.get("plantCost") or placement.get("plantCost") or 0)
            log.sun_spent += max(0, spent)

        done_reason = str(info.get("done_reason") or "") or classify_done_reason(observation)
        log.terminal_reason = str(info.get("terminal_reason") or "")
        if done_reason in ("win", "loss", "post_win_pending"):
            log.done_reason = done_reason
            break
    else:
        log.done_reason = "timeout"

    final_kills = int(observation.get("killCount", start_kills))
    log.zombies_killed = max(0, final_kills - start_kills)
    log.average_reward_per_step = log.episode_reward / max(1, log.episode_length)
    log.final_wave = int(observation.get("wave", log.final_wave))
    log.max_wave = int(observation.get("maxWave", log.max_wave))
    log.avg_legal_actions = legal_action_total / max(1, log.episode_length)
    log.won = log.done_reason == "win"
    log.lost = log.done_reason == "loss"
    log.timed_out = log.done_reason == "timeout"
    log.actual_terminal = log.done_reason in ("win", "loss")
    log.reward_components_match = abs(
        log.episode_reward
        - (log.reward_component_total + log.reward_unattributed_adjustment_total)
    ) <= 1e-6
    finalize_lane_episode_diagnostics(log)
    return log


def evaluate_baselines(env: PvZGymEnv, episodes: int, max_steps: int) -> Dict[str, List[EpisodeLog]]:
    rng = random.Random(env.config.seed)
    results: Dict[str, List[EpisodeLog]] = {}
    for policy in ("wait", "random", "teacher"):
        policy_logs: List[EpisodeLog] = []
        for episode in range(episodes):
            policy_logs.append(run_episode(env, policy, episode, max_steps, rng))
        results[policy] = policy_logs
    return results


def sun_spawn_rate_test(env: PvZGymEnv, duration_seconds: float, no_planting: bool = True) -> Dict[str, Any]:
    started = time.perf_counter()
    observation = env.observe()
    start_game_time = float(observation.get("time", 0.0))
    start_sun = int(observation.get("sun", 0))
    last_sun = start_sun
    sun_gained = 0
    steps = 0

    sun_spawned = 0
    active_peak = None
    prev_active = observation.get("activeFallingSunCount")
    if isinstance(prev_active, int):
        active_peak = prev_active

    sun_collected_start = observation.get("sunCollectedCount") if isinstance(observation.get("sunCollectedCount"), int) else None
    sun_collected = None
    terminal_reason = ""
    done_reason = ""
    terminated_early = False

    while time.perf_counter() - started < duration_seconds:
        action = 0 if no_planting else env.teacher_action(observation)
        observation, _, done, truncated, info = env.step(action)
        steps += 1
        current_sun = int(observation.get("sun", last_sun))
        if current_sun > last_sun:
            sun_gained += current_sun - last_sun
        last_sun = current_sun

        active = observation.get("activeFallingSunCount")
        if isinstance(active, int):
            if isinstance(prev_active, int) and active > prev_active:
                sun_spawned += active - prev_active
            active_peak = active if active_peak is None else max(active_peak, active)
            prev_active = active

        collected_total = observation.get("sunCollectedCount")
        if isinstance(collected_total, int):
            base = sun_collected_start or 0
            sun_collected = max(0, collected_total - base)

        if done or truncated:
            terminated_early = True
            done_reason = str(info.get("done_reason") or ("timeout" if truncated else ""))
            terminal_reason = str(info.get("terminal_reason") or "")
            break

    real_seconds = max(1e-6, time.perf_counter() - started)
    game_seconds = max(0.0, float(observation.get("time", 0.0)) - start_game_time)
    sun_per_real = sun_gained / real_seconds
    sun_per_game = sun_gained / max(1e-6, game_seconds)

    sun_value_assumed = 25
    if sun_collected is None:
        sun_collected = int(round(sun_gained / max(1, sun_value_assumed)))

    report = {
        "real_seconds": real_seconds,
        "game_seconds": game_seconds,
        "sun_objects_spawned": sun_spawned if active_peak is not None else None,
        "sun_objects_collected": sun_collected,
        "sun_gained": sun_gained,
        "sun_per_real_second": sun_per_real,
        "sun_per_game_second": sun_per_game,
        "activeFallingSunPeak": active_peak,
        "sun_value_assumed": sun_value_assumed,
        "requestedGameSpeed": observation.get("requestedGameSpeed"),
        "gameSpeedMode": observation.get("gameSpeedMode"),
        "currentUnityTimeScale": observation.get("unityTimeScale"),
        "fixedDeltaTime": observation.get("fixedDeltaTime"),
        "effectiveGameSpeed": observation.get("effectiveGameSpeed"),
        "skySunSpawnInterval": observation.get("skySunSpawnInterval"),
        "sunSpawnCountPerMinute": observation.get("sunSpawnCountPerMinute"),
        "newZombieWaveCountDown": observation.get("newZombieWaveCountDown"),
        "nextZombieWaveCountDown": observation.get("nextZombieWaveCountDown"),
        "hugeWaveCountDown": observation.get("hugeWaveCountDown"),
        "steps": steps,
        "terminated_early": terminated_early,
        "done_reason": done_reason,
        "terminal_reason": terminal_reason,
    }
    print(json.dumps(report, indent=2))
    return report


def ppo_loop_sun_drift_test(
    env: PvZGymEnv,
    steps: int,
    no_planting: bool = True,
    log_interval: int = 500,
    wait_gameplay_ready: bool = True,
) -> Dict[str, Any]:
    """Run the PPO observe/action-mask/step/reset cadence without learning."""
    started = time.perf_counter()
    observation, reset_info = _ppo_loop_reset(env, wait_gameplay_ready)
    episode = 0
    reset_events = 1
    samples: List[Dict[str, Any]] = []
    window_spawn = 0
    window_gain = 0
    last_sun = _optional_int(observation.get("sun"))
    last_active = _active_sun_count_from_observation(observation)
    last_game_time = _optional_float(observation.get("time"))

    for step_index in range(1, max(0, steps) + 1):
        legal = env.legal_actions(observation)
        if no_planting:
            action = 0
        else:
            candidate = env.teacher_action(observation)
            action = int(candidate) if int(candidate) in set(legal) else 0

        observation, _, done, truncated, info = env.step(action)
        current_sun = _optional_int(observation.get("sun"))
        current_active = _active_sun_count_from_observation(observation)
        current_game_time = _optional_float(observation.get("time"))

        sun_gain = 0
        if current_sun is not None and last_sun is not None and current_sun > last_sun:
            sun_gain = current_sun - last_sun

        spawn_delta = 0
        if current_active is not None and last_active is not None and current_active > last_active:
            spawn_delta = current_active - last_active

        game_delta = 0.0
        if current_game_time is not None and last_game_time is not None and current_game_time >= last_game_time:
            game_delta = current_game_time - last_game_time

        window_spawn += spawn_delta
        window_gain += sun_gain
        sample = {
            "step": step_index,
            "episode": episode,
            "realTime": time.perf_counter() - started,
            "gameDelta": game_delta,
            "sunSpawnDelta": spawn_delta,
            "sunGainDelta": sun_gain,
            "sun": current_sun,
            "activeSunObjectCount": current_active,
            "activeBoardCount": _optional_int(observation.get("activeBoardCount")),
            "activeSkySunSpawnerCount": _optional_int(observation.get("activeSkySunSpawnerCount")),
            "unityTimeScale": _optional_float(observation.get("unityTimeScale")),
            "requestedGameSpeed": _optional_float(observation.get("requestedGameSpeed")),
            "effectiveGameSpeed": _optional_float(observation.get("effectiveGameSpeed")),
            "validSpeedModeApplyCount": _optional_int(observation.get("validSpeedModeApplyCount")),
            "speedApplyCount": _optional_int(observation.get("speedApplyCount")),
            "sunSpawnCompensationApplyCount": _optional_int(observation.get("sunSpawnCompensationApplyCount")),
            "skySunSpawnInterval": _optional_float(observation.get("skySunSpawnInterval")),
            "skySunSpawnTimer": _optional_float(observation.get("skySunSpawnTimer")),
            "boardInstanceId": _optional_int(observation.get("boardInstanceId")),
            "resetCount": _optional_int(observation.get("resetCount")),
            "letsRockClickCount": _optional_int(observation.get("letsRockClickCount")),
            "bridgeUpdateLoopCount": _optional_int(observation.get("bridgeUpdateLoopCount")),
            "activeCoroutineCount": _optional_int(observation.get("activeCoroutineCount")),
            "done": bool(done),
            "truncated": bool(truncated),
            "doneReason": info.get("done_reason"),
            "terminalReason": info.get("terminal_reason"),
        }
        samples.append(sample)

        last_sun = current_sun
        last_active = current_active
        last_game_time = current_game_time

        if log_interval > 0 and step_index % log_interval == 0:
            payload = {
                "step": step_index,
                "episode": episode,
                "currentTimeScale": sample["unityTimeScale"],
                "requestedGameSpeed": sample["requestedGameSpeed"],
                "validSpeedModeApplyCount": sample["validSpeedModeApplyCount"],
                "speedApplyCount": sample["speedApplyCount"],
                "sunSpawnCompensationApplyCount": sample["sunSpawnCompensationApplyCount"],
                "activeBoardCount": sample["activeBoardCount"],
                "activeSkySunSpawnerCount": sample["activeSkySunSpawnerCount"],
                "activeSunObjectCount": sample["activeSunObjectCount"],
                "sunSpawnCountWindow": window_spawn,
                "sunGainedWindow": window_gain,
                "skySunSpawnInterval": sample["skySunSpawnInterval"],
                "skySunSpawnTimer": sample["skySunSpawnTimer"],
                "boardInstanceId": sample["boardInstanceId"],
                "resetCount": sample["resetCount"],
                "letsRockClickCount": sample["letsRockClickCount"],
                "bridgeUpdateLoopCount": sample["bridgeUpdateLoopCount"],
                "activeCoroutineCount": sample["activeCoroutineCount"],
            }
            print("[ppo-loop-sun-drift] " + json.dumps(payload, separators=(",", ":")))
            window_spawn = 0
            window_gain = 0

        if done or truncated:
            episode += 1
            observation, reset_info = _ppo_loop_reset(env, wait_gameplay_ready)
            reset_events += 1
            last_sun = _optional_int(observation.get("sun"))
            last_active = _active_sun_count_from_observation(observation)
            last_game_time = _optional_float(observation.get("time"))

    first_window = samples[: min(500, len(samples))]
    last_window = samples[-min(500, len(samples)):] if samples else []
    report = {
        "steps_requested": steps,
        "steps_completed": len(samples),
        "episodes_started": reset_events,
        "last_reset": _reset_info_summary(reset_info),
        "first_500": _sun_drift_window_summary(first_window),
        "last_500": _sun_drift_window_summary(last_window),
        "activeBoardCount": _trend_summary(samples, "activeBoardCount"),
        "activeSkySunSpawnerCount": _trend_summary(samples, "activeSkySunSpawnerCount"),
        "activeSunObjectCount": _trend_summary(samples, "activeSunObjectCount"),
        "validSpeedModeApplyCount": _counter_summary(samples, "validSpeedModeApplyCount"),
        "speedApplyCount": _counter_summary(samples, "speedApplyCount"),
        "sunSpawnCompensationApplyCount": _counter_summary(samples, "sunSpawnCompensationApplyCount"),
        "skySunSpawnInterval": _trend_summary(samples, "skySunSpawnInterval"),
        "skySunSpawnTimer": _trend_summary(samples, "skySunSpawnTimer"),
        "unityTimeScale": _trend_summary(samples, "unityTimeScale"),
        "boardInstanceIds": sorted({sample["boardInstanceId"] for sample in samples if sample.get("boardInstanceId") is not None}),
        "resetCount": _counter_summary(samples, "resetCount"),
        "letsRockClickCount": _counter_summary(samples, "letsRockClickCount"),
        "bridgeUpdateLoopCount": _counter_summary(samples, "bridgeUpdateLoopCount"),
        "acceptance": {
            "activeBoardCountRemainedOne": _last_value(samples, "activeBoardCount") == 1 and _max_value(samples, "activeBoardCount") == 1,
            "activeSkySunSpawnerCountRemainedOne": _last_value(samples, "activeSkySunSpawnerCount") == 1 and _max_value(samples, "activeSkySunSpawnerCount") == 1,
            "validSpeedModeNotAppliedEveryStep": not _counter_increased_every_step(samples, "validSpeedModeApplyCount"),
            "sunSpawnCompensationNotAppliedEveryStep": not _counter_increased_every_step(samples, "sunSpawnCompensationApplyCount"),
            "unityTimeScaleStable": not _changed_over_time(samples, "unityTimeScale"),
            "skySunSpawnIntervalStable": not _changed_over_time(samples, "skySunSpawnInterval"),
        },
    }
    print(json.dumps(report, indent=2))
    return report


def reset_stress_test(env: PvZGymEnv, cycles: int) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    reset_failures = 0
    bridge_errors = 0
    env.configure()
    for cycle in range(max(0, cycles)):
        started = time.perf_counter()
        try:
            ping = env.client.request("ping")
            observation, reset_info = env.reset()
            validate_observation(observation)
            cleanup_ok, cleanup_message = verify_reset_cleanup_state(env, observation, require_mowers=True)
            mask = env.action_mask(observation)
            mask_diag = env.mask_diagnostics(observation, mask)
            slots = seed_slots_from_observation(observation, env.config.plant_types)
            reset_payload = reset_info.get("reset", {}) if isinstance(reset_info, dict) else {}
            ok = (
                bool(reset_payload.get("resetSuccess", reset_payload.get("ok", True)))
                and bool(observation.get("gameplayReady"))
                and bool(observation.get("boardFound"))
                and bool(observation.get("canReadBoard", True))
                and cleanup_ok
                and len(slots) >= len(env.config.plant_types)
                and bool(mask)
                and bool(mask[0])
            )
            result = {
                "cycle": cycle,
                "ok": ok,
                "seconds": round(time.perf_counter() - started, 3),
                "pingBoardFound": bool(ping.get("boardFound")),
                "gameplayReady": bool(observation.get("gameplayReady")),
                "boardFound": bool(observation.get("boardFound")),
                "canReadBoard": bool(observation.get("canReadBoard", True)),
                "seedSlotCount": len(slots),
                "logicalMowerCount": int(observation.get("logicalMowerCount", 0) or 0),
                "visibleMowerObjectCount": int(observation.get("visibleMowerObjectCount", 0) or 0),
                "duplicateMowerRowCount": int(observation.get("duplicateMowerRowCount", 0) or 0),
                "legalActionCount": sum(1 for allowed in mask if allowed),
                "legalActionsBySeedSlot": mask_diag.get("legal_actions_by_seed_slot", {}),
                "bridgeLegalActionsBySeedSlot": mask_diag.get("bridge_legal_actions_by_seed_slot", {}),
                "cleanupOk": cleanup_ok,
                "cleanupMessage": cleanup_message,
                "reset": _reset_info_summary(reset_info),
            }
            if not ok:
                reset_failures += 1
            print("[reset-stress] " + json.dumps(result, separators=(",", ":"), sort_keys=True))
            results.append(result)
        except Exception as exc:
            reset_failures += 1
            bridge_errors += 1
            result = {
                "cycle": cycle,
                "ok": False,
                "seconds": round(time.perf_counter() - started, 3),
                "error": str(exc),
            }
            print("[reset-stress] " + json.dumps(result, separators=(",", ":"), sort_keys=True))
            results.append(result)
            env.client.close()
    summary = {
        "cycles_requested": cycles,
        "cycles_completed": len(results),
        "reset_failures": reset_failures,
        "bridge_errors": bridge_errors,
        "passed": reset_failures == 0 and bridge_errors == 0 and len(results) == max(0, cycles),
        "results": results,
    }
    print(json.dumps(summary, indent=2))
    return summary


def duplicate_slot_validation_report(env: PvZGymEnv, observation: Dict[str, Any]) -> Dict[str, Any]:
    mask = env.action_mask(observation)
    mask_diag = env.mask_diagnostics(observation, mask)
    slots = seed_slots_from_observation(observation, env.config.plant_types)
    slots_by_type: Dict[str, List[int]] = {}
    for slot in slots:
        plant_type = str(slot.get("plantType", -1))
        slots_by_type.setdefault(plant_type, []).append(int(slot.get("slotIndex", len(slots_by_type))))
    return {
        "slotsByPlantType": slots_by_type,
        "slotReadinessBySeedSlot": mask_diag.get("slot_readiness_by_seed_slot", {}),
        "legalActionsBySeedSlot": mask_diag.get("legal_actions_by_seed_slot", {}),
        "bridgeLegalActionsBySeedSlot": mask_diag.get("bridge_legal_actions_by_seed_slot", {}),
        "note": "Duplicate-card legality is reported by slot index; same plant types in different slots are not merged.",
    }


def cooldown_mask_test(
    env: PvZGymEnv,
    steps: int,
    output_path: Optional[Path] = None,
    log_interval: int = 100,
) -> Dict[str, Any]:
    rng = random.Random(env.config.seed)
    env.configure()
    observation, reset_info = _ppo_loop_reset(env, wait_gameplay_ready=True)
    if output_path is None:
        output_path = PROJECT_ROOT / "runs" / "diagnostics" / f"cooldown_mask_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    illegal_reason_counts: Counter[str] = Counter()
    mask_block_reason_counts: Counter[str] = Counter()
    cooldown_illegal_exposed_by_mask = 0
    pre_step_mask_blocked = 0
    bridge_errors = 0
    reset_failures = 0
    events = 0
    steps_completed = 0
    duplicate_report_start = duplicate_slot_validation_report(env, observation)

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "start", "duplicateSlotReport": duplicate_report_start}, separators=(",", ":")) + "\n")
        for step_index in range(1, max(0, steps) + 1):
            try:
                mask = env.action_mask(observation)
                legal_actions = [action for action, allowed in enumerate(mask) if bool(allowed)]
                plant_actions = [action for action in legal_actions if action > 0]
                action = rng.choice(plant_actions) if plant_actions else 0
                pre_audit = env.action_legality_audit(action, observation, mask)
                observation, _, done, truncated, info = env.step(action)
                steps_completed += 1
            except Exception as exc:
                bridge_errors += 1
                events += 1
                handle.write(json.dumps({"event": "bridge_error", "step": step_index, "error": str(exc)}, separators=(",", ":")) + "\n")
                env.client.close()
                try:
                    observation, reset_info = _ppo_loop_reset(env, wait_gameplay_ready=True)
                except Exception as reset_exc:
                    reset_failures += 1
                    handle.write(json.dumps({"event": "reset_error", "step": step_index, "error": str(reset_exc)}, separators=(",", ":")) + "\n")
                    break
                continue

            action_result = info.get("action_result", {}) if isinstance(info, dict) else {}
            audit = action_result.get("actionAudit") if isinstance(action_result, dict) and isinstance(action_result.get("actionAudit"), dict) else pre_audit
            if bool(action_result.get("preStepMaskBlockedAction")):
                pre_step_mask_blocked += 1
                reason = str(audit.get("pythonFilterReason") or "blocked")
                mask_block_reason_counts[reason] += 1
                events += 1
                handle.write(json.dumps({"event": "pre_step_mask_blocked_action", "step": step_index, "audit": audit}, separators=(",", ":")) + "\n")

            if bool(action_result.get("illegalAction")):
                reason = str(action_result.get("illegalReason") or "unknown")
                illegal_reason_counts[reason] += 1
                exposed = reason == "cooldown" and bool(
                    audit.get("pythonMaskValueBefore") or audit.get("bridgeLegalActionsValueBefore")
                )
                if exposed:
                    cooldown_illegal_exposed_by_mask += 1
                events += 1
                handle.write(
                    json.dumps(
                        {
                            "event": "illegal_action",
                            "step": step_index,
                            "illegalReason": reason,
                            "cooldownIllegalExposedByMask": exposed,
                            "audit": audit,
                            "actionResult": action_result,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )

            if log_interval > 0 and step_index % log_interval == 0:
                payload = {
                    "step": step_index,
                    "legalActions": len(legal_actions),
                    "plantActions": len(plant_actions),
                    "preStepMaskBlocked": pre_step_mask_blocked,
                    "cooldownIllegalExposedByMask": cooldown_illegal_exposed_by_mask,
                    "illegalReasonCounts": dict(illegal_reason_counts),
                }
                print("[cooldown-mask] " + json.dumps(payload, separators=(",", ":"), sort_keys=True))

            if done or truncated or is_restart_screen_observation(observation):
                try:
                    observation, reset_info = _ppo_loop_reset(env, wait_gameplay_ready=True)
                except Exception as exc:
                    reset_failures += 1
                    events += 1
                    handle.write(json.dumps({"event": "reset_error", "step": step_index, "error": str(exc)}, separators=(",", ":")) + "\n")
                    break

    summary = {
        "steps_requested": steps,
        "steps_completed": steps_completed,
        "cooldown_illegal_exposed_by_mask": cooldown_illegal_exposed_by_mask,
        "pre_step_mask_blocked_action_count": pre_step_mask_blocked,
        "illegal_reason_counts": dict(sorted(illegal_reason_counts.items())),
        "python_mask_block_reason_counts": dict(sorted(mask_block_reason_counts.items())),
        "bridge_errors": bridge_errors,
        "reset_failures": reset_failures,
        "diagnostic_log": str(output_path),
        "duplicate_slot_report_start": duplicate_report_start,
        "events_logged": events,
        "passed": (
            cooldown_illegal_exposed_by_mask == 0
            and bridge_errors == 0
            and reset_failures == 0
            and steps_completed == max(0, steps)
        ),
    }
    print(json.dumps(summary, indent=2))
    return summary


def _ppo_loop_reset(env: PvZGymEnv, wait_gameplay_ready: bool) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    observation, reset_info = env.reset()
    if not wait_gameplay_ready:
        return observation, reset_info

    reset_payload = reset_info.setdefault("reset", {}) if isinstance(reset_info, dict) else {}
    require_seed_selection = bool(reset_payload.get("requireSeedSelectionThisReset"))
    saw_seed_selection = bool(reset_payload.get("sawSeedSelectionThisReset"))
    invariant_ok = (not require_seed_selection) or saw_seed_selection

    try:
        if env.config.auto_select_seeds:
            observation, selection = env.ensure_seeds_then_gameplay_ready(
                seed_list=env.config.seed_list,
                timeout=env.config.reset_wait_timeout,
                poll_seconds=env.config.reset_poll_seconds,
                quiet=True,
            )
            reset_payload["postResetEnsureSeeds"] = selection
            reset_payload["effectiveResetSuccess"] = bool(
                observation.get("gameplayReady")
                and not observation.get("seedSelectionActive")
                and not observation.get("done")
                and invariant_ok
            )
        else:
            observation = env.wait_for_gameplay_ready(
                timeout=env.config.reset_wait_timeout,
                poll_seconds=env.config.reset_poll_seconds,
                quiet=True,
                fail_on_terminal=False,
            )
            reset_info.setdefault("reset", {})["effectiveResetSuccess"] = bool(
                observation.get("gameplayReady")
                and not observation.get("seedSelectionActive")
                and not observation.get("done")
                and invariant_ok
            )
    except Exception as exc:
        reset_info.setdefault("reset", {})["postResetGameplayReadyError"] = str(exc)
    return observation, reset_info


def _active_sun_count_from_observation(observation: Dict[str, Any]) -> Optional[int]:
    for key in ("activeSunObjectCount", "activeFallingSunCount"):
        value = _optional_int(observation.get(key))
        if value is not None:
            return value
    return None


def _reset_info_summary(reset_info: Dict[str, Any]) -> Dict[str, Any]:
    reset = reset_info.get("reset", {}) if isinstance(reset_info, dict) else {}
    stages = []
    stage_entries = reset.get("stages", []) if isinstance(reset.get("stages"), list) else []
    for stage in stage_entries:
        if not isinstance(stage, dict):
            continue
        stages.append(
            {
                "stage": stage.get("stage"),
                "elapsed": stage.get("elapsed"),
                "gameplayReady": stage.get("gameplayReady"),
                "onSeedSelectionScreen": stage.get("onSeedSelectionScreen"),
                "done": stage.get("done"),
                "attempt": stage.get("attempt"),
                "methodUsed": stage.get("methodUsed"),
            }
        )
    cleanup = reset.get("cleanup", {}) if isinstance(reset.get("cleanup"), dict) else {}
    return {
        "methodUsed": reset.get("methodUsed"),
        "resetSuccess": reset.get("resetSuccess"),
        "effectiveResetSuccess": reset.get("effectiveResetSuccess", reset.get("resetSuccess")),
        "timeToPlayableSeconds": reset.get("timeToPlayableSeconds"),
        "requireSeedSelectionThisReset": reset.get("requireSeedSelectionThisReset"),
        "sawSeedSelectionThisReset": reset.get("sawSeedSelectionThisReset"),
        "resetStartedFromLoss": reset.get("resetStartedFromLoss"),
        "resetPhaseFinal": reset.get("resetPhaseFinal"),
        "unsafeGameplayReadyBeforeSeedCount": reset.get("unsafeGameplayReadyBeforeSeedCount"),
        "postResetSeedSelectionActive": reset.get("postResetSeedSelectionActive"),
        "postResetLegalActionCount": reset.get("postResetLegalActionCount"),
        "postResetEnsureSeeds": reset.get("postResetEnsureSeeds", {}).get("message")
        if isinstance(reset.get("postResetEnsureSeeds"), dict)
        else None,
        "cleanupActions": cleanup.get("actions", []) if isinstance(cleanup, dict) else [],
        "stages": stages,
        "postResetGameplayReadyError": reset.get("postResetGameplayReadyError"),
    }


def _optional_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _numeric_values(samples: List[Dict[str, Any]], key: str) -> List[float]:
    values: List[float] = []
    for sample in samples:
        value = _optional_float(sample.get(key))
        if value is not None:
            values.append(value)
    return values


def _first_value(samples: List[Dict[str, Any]], key: str) -> Optional[float]:
    values = _numeric_values(samples, key)
    return values[0] if values else None


def _last_value(samples: List[Dict[str, Any]], key: str) -> Optional[float]:
    values = _numeric_values(samples, key)
    return values[-1] if values else None


def _max_value(samples: List[Dict[str, Any]], key: str) -> Optional[float]:
    values = _numeric_values(samples, key)
    return max(values) if values else None


def _changed_over_time(samples: List[Dict[str, Any]], key: str, tolerance: float = 1e-4) -> bool:
    values = _numeric_values(samples, key)
    if len(values) < 2:
        return False
    return max(values) - min(values) > tolerance


def _trend_summary(samples: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    values = _numeric_values(samples, key)
    if not values:
        return {"first": None, "last": None, "min": None, "max": None, "changed": False, "increased": False}
    return {
        "first": values[0],
        "last": values[-1],
        "min": min(values),
        "max": max(values),
        "changed": max(values) - min(values) > 1e-4,
        "increased": values[-1] > values[0],
    }


def _counter_summary(samples: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    values = _numeric_values(samples, key)
    if not values:
        return {"first": None, "last": None, "delta": None, "deltaPerStep": None, "increasedEveryStep": False}
    delta = values[-1] - values[0]
    return {
        "first": values[0],
        "last": values[-1],
        "delta": delta,
        "deltaPerStep": delta / max(1, len(samples) - 1),
        "increasedEveryStep": _counter_increased_every_step(samples, key),
    }


def _counter_increased_every_step(samples: List[Dict[str, Any]], key: str) -> bool:
    values = _numeric_values(samples, key)
    if len(values) < 2:
        return False
    return (values[-1] - values[0]) >= max(1, len(values) - 1) * 0.9


def _sun_drift_window_summary(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not samples:
        return {
            "steps": 0,
            "sunSpawnCount": 0,
            "sunGained": 0,
            "sunSpawnPerStep": 0.0,
            "sunGainedPerStep": 0.0,
            "sunSpawnPerGameSecond": None,
            "sunGainedPerGameSecond": None,
        }
    steps = len(samples)
    sun_spawn = sum(int(sample.get("sunSpawnDelta") or 0) for sample in samples)
    sun_gained = sum(int(sample.get("sunGainDelta") or 0) for sample in samples)
    game_seconds = sum(float(sample.get("gameDelta") or 0.0) for sample in samples)
    return {
        "steps": steps,
        "sunSpawnCount": sun_spawn,
        "sunGained": sun_gained,
        "sunSpawnPerStep": sun_spawn / max(1, steps),
        "sunGainedPerStep": sun_gained / max(1, steps),
        "gameSeconds": game_seconds,
        "sunSpawnPerGameSecond": sun_spawn / game_seconds if game_seconds > 1e-6 else None,
        "sunGainedPerGameSecond": sun_gained / game_seconds if game_seconds > 1e-6 else None,
    }


def print_baseline_table(results: Dict[str, List[EpisodeLog]]) -> None:
    headers = [
        "policy",
        "episodes",
        "win_rate",
        "avg_reward",
        "avg_episode_length",
        "avg_wave",
        "avg_kills",
        "avg_plants",
        "avg_mowers_lost",
        "avg_sun_remaining",
        "reset_failures",
        "bridge_errors",
    ]
    rows: List[List[str]] = []
    for policy, logs in results.items():
        summary = summarize_policy(logs)
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

    print("Baseline Evaluation")
    print("-------------------")
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def summarize_policy(logs: List[EpisodeLog]) -> Dict[str, float]:
    count = max(1, len(logs))
    avg_kills = sum(log.zombies_killed for log in logs) / count
    avg_wave = sum(log.final_wave for log in logs) / count
    avg_plants = sum(log.plants_placed for log in logs) / count
    return {
        "episodes": len(logs),
        "win_rate": sum(1 for log in logs if log.won) / count,
        "avg_reward": sum(log.episode_reward for log in logs) / count,
        "avg_episode_length": sum(log.episode_length for log in logs) / count,
        "avg_zombies_killed": avg_kills,
        "avg_final_wave": avg_wave,
        "avg_plants_placed": avg_plants,
        "avg_kills": avg_kills,
        "avg_wave": avg_wave,
        "avg_plants": avg_plants,
        "avg_mowers_lost": sum(log.mowers_lost for log in logs) / count,
        "avg_sun_remaining": sum(log.sun_remaining for log in logs) / count,
        "actual_terminals": sum(1 for log in logs if log.actual_terminal),
        "timeouts": sum(1 for log in logs if log.timed_out),
        "illegal_actions": sum(log.illegal_actions for log in logs),
        "reset_failures": sum(log.reset_failures for log in logs),
        "bridge_errors": sum(log.bridge_errors for log in logs),
    }


def print_terminal_test(log: EpisodeLog, max_steps: int) -> None:
    print("Terminal Long Run")
    print("-----------------")
    print(f"policy: {log.policy}")
    print(f"done_reason: {log.done_reason}")
    print(f"actual_terminal: {str(log.actual_terminal).lower()}")
    print(f"timed_out: {str(log.timed_out).lower()}")
    print(f"max_steps: {'none' if max_steps == 0 else max_steps}")
    print(f"elapsed_steps: {log.episode_length}")
    print(f"final_wave: {log.final_wave}")
    print(f"max_wave: {log.max_wave}")
    print(f"zombies_killed: {log.zombies_killed}")
    print(f"plants_placed: {log.plants_placed}")
    print(f"illegal_actions: {log.illegal_actions}")
    print(f"episode_reward: {log.episode_reward:.2f}")
    print(f"average_reward_per_step: {log.average_reward_per_step:.4f}")
    print(f"reset_failures: {log.reset_failures}")
    print(f"bridge_errors: {log.bridge_errors}")
    if log.error:
        print(f"error: {log.error}")


def dump_action_map(env: PvZGymEnv, observation: Optional[Dict[str, Any]] = None) -> None:
    obs = observation or env.observe()
    rows = int(obs.get("rowCount") or env.config.row_count)
    cols = int(obs.get("columnCount") or env.config.column_count)
    slots = seed_slots_from_observation(obs, env.config.plant_types)
    print("Action Map")
    print("----------")
    print("0 wait")
    for slot in slots:
        slot_index = int(slot.get("slotIndex", 0))
        plant_type = int(slot.get("plantType", -1))
        name = str(slot.get("plantTypeName") or plant_type_name(plant_type))
        cost = int(slot.get("seedCost", 0))
        ready = bool(slot.get("ready", False))
        usable = bool(slot.get("usable", False))
        warning = str(slot.get("warning") or "")
        for row in range(rows):
            for col in range(cols):
                action = 1 + slot_index * rows * cols + row * cols + col
                suffix = f" warning={warning}" if warning else ""
                print(f"{action} slot={slot_index} {name} cost={cost} row={row} col={col} ready={ready} usable={usable}{suffix}")


def terminal_auto_reset_test(env: PvZGymEnv, episodes: int, max_steps: int, policy: str) -> bool:
    rng = random.Random(env.config.seed)
    logs: List[EpisodeLog] = []
    reset_results: List[Dict[str, Any]] = []
    loss_invariant_failures = 0

    def has_stage_sequence(stages: List[str], required_sequence: List[str]) -> bool:
        idx = 0
        for stage_name in stages:
            if idx < len(required_sequence) and stage_name == required_sequence[idx]:
                idx += 1
        return idx == len(required_sequence)

    print("Terminal Auto Reset Test")
    print("------------------------")
    for episode in range(episodes):
        # The caller prepares the initial board and this test performs the post-terminal
        # reset itself, so a pre-episode reset would be an unsafe active-gameplay reset.
        log = run_episode(env, policy, episode, max_steps, rng, reset_each_episode=False)
        logs.append(log)
        print(
            f"episode={episode} done_reason={log.done_reason} actual_terminal={str(log.actual_terminal).lower()} "
            f"timed_out={str(log.timed_out).lower()} steps={log.episode_length} wave={log.final_wave}/{log.max_wave} "
            f"kills={log.zombies_killed} plants={log.plants_placed} reward={log.episode_reward:.2f}"
        )
        if log.error:
            print(f"episode={episode} error={log.error}")

        started = time.monotonic()
        try:
            reset_obs, reset_info = env.reset()
            reset_payload = reset_info.get("reset", {})
            reset_payload["observedResetSeconds"] = round(time.monotonic() - started, 3)
            reset_payload["postResetDone"] = bool(reset_obs.get("done"))
            reset_payload["postResetGameplayReady"] = bool(reset_obs.get("gameplayReady"))
            reset_results.append(reset_payload)
            print(
                f"reset episode={episode} terminal_detected={reset_payload.get('terminalDetected')} "
                f"method={reset_payload.get('methodUsed')} success={reset_payload.get('resetSuccess')} "
                f"time_to_playable={reset_payload.get('timeToPlayableSeconds')}s "
                f"legal_actions={reset_payload.get('postResetLegalActionCount')}"
            )
            stages = [str(stage.get("stage")) for stage in reset_payload.get("stages", [])]
            if stages:
                print(f"reset episode={episode} stages={' -> '.join(stages)}")
            is_loss_episode = bool(log.done_reason == "loss" or log.terminal_reason == "game_over_restart_screen")
            if is_loss_episode:
                require_seed = bool(reset_payload.get("requireSeedSelectionThisReset"))
                saw_seed = bool(reset_payload.get("sawSeedSelectionThisReset"))
                unsafe_count = int(reset_payload.get("unsafeGameplayReadyBeforeSeedCount", 0) or 0)
                stage_order_ok = has_stage_sequence(
                    stages,
                    ["restart_clicked", "seed_screen_detected", "lets_rock_clicked", "gameplay_ready"],
                )
                reset_payload["lossStageOrderValid"] = stage_order_ok
                print(
                    f"reset episode={episode} loss_invariant requireSeed={require_seed} "
                    f"sawSeed={saw_seed} stageOrderOk={stage_order_ok} unsafeCount={unsafe_count}"
                )
                if not (require_seed and saw_seed and stage_order_ok and unsafe_count == 0):
                    loss_invariant_failures += 1
                    reset_payload["resetSuccess"] = False
                    print(
                        f"reset episode={episode} loss_invariant_failure requireSeed={require_seed} "
                        f"sawSeed={saw_seed} stageOrderOk={stage_order_ok} unsafeCount={unsafe_count}"
                    )
        except Exception as exc:
            reset_results.append({"ok": False, "error": str(exc)})
            print(f"reset episode={episode} failed: {exc}")
            break

    bridge_errors = sum(log.bridge_errors for log in logs)
    reset_failures = sum(log.reset_failures for log in logs)
    terminal_count = sum(1 for log in logs if log.actual_terminal)
    reset_successes = sum(1 for result in reset_results if result.get("resetSuccess"))
    print(
        f"summary episodes={len(logs)} terminals={terminal_count} reset_successes={reset_successes} "
        f"bridge_errors={bridge_errors} reset_failures={reset_failures} "
        f"loss_invariant_failures={loss_invariant_failures}"
    )
    return (
        len(logs) == episodes
        and len(reset_results) == episodes
        and reset_successes == episodes
        and bridge_errors == 0
        and reset_failures == 0
        and loss_invariant_failures == 0
    )


def loss_reset_invariant_negative_test(env: PvZGymEnv) -> bool:
    print("Loss Reset Invariant Negative Test")
    print("----------------------------------")
    unsafe_observation = {
        "boardFound": True,
        "canReadBoard": True,
        "screenState": "gameplay",
        "nextStep": "play",
        "gameplayReady": True,
        "seedSelectionActive": False,
        "terminalHint": "running",
        "done": False,
        "over": False,
    }
    reset_result: Dict[str, Any] = {
        "resetSuccess": False,
        "requireSeedSelectionThisReset": True,
        "sawSeedSelectionThisReset": False,
        "unsafeGameplayReadyBeforeSeedCount": 0,
    }
    previous_before = env.previous_observation
    try:
        env._enforce_loss_reset_seed_selection_invariant(
            unsafe_observation,
            require_seed_selection_this_reset=True,
            saw_seed_selection_this_reset=False,
            reset_phase="waiting_seed_selection",
            reset_result=reset_result,
            note="negative_test",
        )
    except RuntimeError as exc:
        unsafe_count = int(reset_result.get("unsafeGameplayReadyBeforeSeedCount", 0) or 0)
        reset_success = bool(reset_result.get("resetSuccess", False))
        episode_not_started = env.previous_observation is previous_before
        print(f"PASS loss_reset_invariant_raises: {exc}")
        print(
            f"details unsafe_count={unsafe_count} reset_success={reset_success} "
            f"episode_not_started={episode_not_started}"
        )
        return unsafe_count >= 1 and not reset_success and episode_not_started

    print("FAIL loss_reset_invariant_raises: no RuntimeError was raised.")
    return False


def validate_reliability(env: PvZGymEnv, smoke_runs: int, board_timeout: float, max_steps: int) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "smoke_runs": smoke_runs,
        "smoke_passes": 0,
        "smoke_failures": 0,
        "contract_checks": {},
        "baselines": {},
        "ready_for_ppo": False,
        "remaining_blockers": [],
    }

    env.wait_for_board(timeout=board_timeout)
    for _ in range(smoke_runs):
        passed = smoke_test(env, wait_for_board=False, board_timeout=board_timeout)
        if passed:
            report["smoke_passes"] += 1
        else:
            report["smoke_failures"] += 1

    observation, reset_info = env.reset()
    validate_observation(observation)
    legal = env.legal_actions(observation)
    teacher = env.teacher_action(observation)
    occupied_ok, occupied_message = verify_legal_actions_exclude_occupied(env, observation)
    teacher_ok, teacher_message = verify_teacher_action_legal(env, observation)

    before_frame = int(observation.get("frameCount", 0))
    before_time = float(observation.get("time", 0.0))
    wait_obs, _, _, _, _ = env.step(0)
    wait_advances = int(wait_obs.get("frameCount", 0)) > before_frame or float(wait_obs.get("time", 0.0)) > before_time

    valid_actions = [action for action in env.legal_actions(wait_obs) if action > 0]
    valid_changes_state = False
    if valid_actions:
        before_cells = {(p.get("row"), p.get("column"), p.get("type")) for p in wait_obs.get("plants", [])}
        plant_count_before = int(wait_obs.get("plantCount", 0))
        plant_obs, _, _, _, info = env.step(valid_actions[0])
        after_cells = {(p.get("row"), p.get("column"), p.get("type")) for p in plant_obs.get("plants", [])}
        plant_count_after = int(plant_obs.get("plantCount", 0))
        valid_changes_state = bool((after_cells - before_cells) or plant_count_after > plant_count_before or (info.get("action_result", {}).get("placement") or {}).get("success"))

    invalid_action = 1 + len(env.config.plant_types) * int(observation.get("rowCount", env.config.row_count)) * int(observation.get("columnCount", env.config.column_count)) + 999
    _, _, _, _, invalid_info = env.step(invalid_action)
    invalid_action_result = invalid_info.get("action_result", {})
    invalid_safe = bool(invalid_action_result.get("illegalAction")) or bool(invalid_action_result.get("preStepMaskBlockedAction"))

    reset_obs, reset_info = env.soft_reset()
    reset_valid = bool(reset_info.get("reset", {}).get("ok", True))
    validate_observation(reset_obs)

    report["contract_checks"] = {
        "legal_actions_nonempty": len(legal) > 0,
        "legal_actions_exclude_occupied": occupied_ok,
        "legal_actions_exclude_occupied_detail": occupied_message,
        "teacher_action_legal": teacher_ok,
        "teacher_action_detail": teacher_message,
        "teacher_action": teacher,
        "step_wait_advances": wait_advances,
        "step_valid_action_changes_state": valid_changes_state,
        "invalid_action_safe": invalid_safe,
        "reset_valid_observation": reset_valid,
    }

    for episodes in (5, 20):
        results = evaluate_baselines(env, episodes, max_steps)
        report["baselines"][str(episodes)] = {
            policy: {
                **summarize_policy(logs),
                "done_observed": any(log.actual_terminal for log in logs),
            }
            for policy, logs in results.items()
        }

    blockers = []
    if report["smoke_failures"]:
        blockers.append("smoke test did not pass repeatedly")
    for name, value in report["contract_checks"].items():
        if name.endswith("_detail") or name == "teacher_action":
            continue
        if not value:
            blockers.append(name)
    baseline_20 = report["baselines"].get("20", {})
    if baseline_20:
        teacher_reward = baseline_20.get("teacher", {}).get("avg_reward", 0.0)
        random_reward = baseline_20.get("random", {}).get("avg_reward", 0.0)
        wait_reward = baseline_20.get("wait", {}).get("avg_reward", 0.0)
        teacher_kills = baseline_20.get("teacher", {}).get("avg_zombies_killed", 0.0)
        random_kills = baseline_20.get("random", {}).get("avg_zombies_killed", 0.0)
        wait_kills = baseline_20.get("wait", {}).get("avg_zombies_killed", 0.0)
        teacher_wave = baseline_20.get("teacher", {}).get("avg_final_wave", 0.0)
        random_wave = baseline_20.get("random", {}).get("avg_final_wave", 0.0)
        wait_wave = baseline_20.get("wait", {}).get("avg_final_wave", 0.0)
        any_reward = any(summary.get("avg_reward", 0.0) != 0.0 for summary in baseline_20.values())
        any_kills = any(summary.get("avg_zombies_killed", 0.0) > 0.0 for summary in baseline_20.values())
        any_wave = any(summary.get("avg_final_wave", 0.0) > 0.0 for summary in baseline_20.values())
        any_done = any(summary.get("done_observed", False) for summary in baseline_20.values())
        if not (any_reward or any_kills or any_wave):
            blockers.append("baseline did not exercise active zombie/wave gameplay")
        if not any_done:
            blockers.append("done/win/loss was not observed during validation")
        if not (
            teacher_reward > max(random_reward, wait_reward)
            or teacher_kills > max(random_kills, wait_kills)
            or teacher_wave > max(random_wave, wait_wave)
        ):
            blockers.append("teacher does not yet beat random/wait baseline")
        for policy, summary in baseline_20.items():
            if summary.get("reset_failures", 0) or summary.get("bridge_errors", 0):
                blockers.append(f"{policy} baseline had reset failures or bridge errors")

    report["remaining_blockers"] = blockers
    report["ready_for_ppo"] = not blockers
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="PvZRL bridge client")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=32323)
    parser.add_argument("--game-exe", default=None, help="Optional game executable path for Python-owned hard resets.")
    parser.add_argument("--proof", action="store_true")
    parser.add_argument("--place-test", action="store_true")
    parser.add_argument("--observe", action="store_true")
    parser.add_argument("--legal-actions", action="store_true")
    parser.add_argument("--teacher-action", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--adventure-state-smoke", action="store_true")
    parser.add_argument("--sun-cost-test", action="store_true")
    parser.add_argument("--cooldown-test", action="store_true")
    parser.add_argument("--fusion-semantics-test", action="store_true")
    parser.add_argument("--coach-fusion-scope-test", action="store_true")
    parser.add_argument("--reset-cleanup-test", action="store_true")
    parser.add_argument("--reset-state-machine-test", action="store_true")
    parser.add_argument("--reset-stress-test", action="store_true", help="Run repeated reset cycles with auto-select and mask/mower validation.")
    parser.add_argument("--restart-screen-detection-test", action="store_true")
    parser.add_argument("--fast-only", action="store_true", help="Use only fast cached screen-state detection for restart-screen tests.")
    parser.add_argument("--pause-menu-restart-test", action="store_true")
    parser.add_argument("--seed-probe", action="store_true")
    parser.add_argument("--ui-probe", action="store_true")
    parser.add_argument("--include-all-ui", action="store_true", help="Include all active objects in --ui-probe, not only UI-like objects.")
    parser.add_argument("--ui-probe-max", type=int, default=350, help="Maximum objects to print from --ui-probe.")
    parser.add_argument("--almanac-probe", action="store_true")
    parser.add_argument("--include-all-plants", action="store_true", help="Include every runtime PlantType enum entry in --almanac-probe.")
    parser.add_argument("--auto-select-seeds", action="store_true", help="Select configured seed packets when seed selection is active.")
    parser.add_argument("--auto-select-seeds-test", action="store_true")
    parser.add_argument("--fresh-seed-select-test", action="store_true")
    parser.add_argument("--seed-screen-gating-test", action="store_true")
    parser.add_argument("--dump-action-map", action="store_true")
    parser.add_argument(
        "--seed-list",
        default="SunFlower,SunFlower,Peashooter,Peashooter",
        help="Comma-separated seed names or PlantType ids for Generalist automation.",
    )
    parser.add_argument("--seed-click-delay", type=float, default=0.35, help="Seconds to wait after each seed packet click before verifying the selected bank.")
    parser.add_argument("--lets-rock-delay", type=float, default=0.5, help="Seconds to wait after all seed packets are verified before pressing Start/Let's Rock.")
    parser.add_argument("--post-start-delay", type=float, default=1.0, help="Seconds to wait after pressing Start/Let's Rock before gameplayReady polling.")
    parser.add_argument("--seed-screen-check-interval", type=int, default=100, help="Normal-gameplay steps between full seed-screen safety probes. Use 0 to disable periodic checks.")
    parser.add_argument("--debug-performance", "--debug-perf", dest="debug_performance", action="store_true", help="Include timing diagnostics such as observe_ms, seed_probe_ms, ui_scan_ms, and bridge_roundtrip_ms.")
    parser.add_argument("--debug-observation", action="store_true", help="Include large debug arrays such as visiblePlants/visibleMowers in normal observations.")
    parser.add_argument("--debug-sun", action="store_true", help="Enable expensive sun-spawn diagnostics and object counts.")
    parser.add_argument("--debug-sun-sample-interval", type=int, default=25)
    parser.add_argument("--eval-baselines", action="store_true")
    parser.add_argument("--validate-reliability", action="store_true")
    parser.add_argument("--terminal-test-long-run", action="store_true")
    parser.add_argument("--terminal-auto-reset-test", action="store_true")
    parser.add_argument("--sun-spawn-rate-test", action="store_true")
    parser.add_argument("--ppo-loop-sun-drift-test", action="store_true")
    parser.add_argument("--cooldown-mask-test", action="store_true", help="Step with the current action mask and trace cooldown-mask leaks.")
    parser.add_argument(
        "--loss-reset-invariant-negative-test",
        action="store_true",
        help="Simulate gameplayReady-before-seed-selection during loss reset and assert fail-fast RuntimeError.",
    )
    parser.add_argument("--diagnostic-output", type=Path, default=None, help="Optional JSONL output path for diagnostic tests.")
    parser.add_argument("--duration-seconds", type=float, default=60.0)
    parser.add_argument("--steps", type=int, default=5000, help="Step count for PPO-cadence diagnostics.")
    parser.add_argument("--sun-drift-log-interval", type=int, default=500)
    parser.add_argument("--no-planting", action="store_true")
    parser.add_argument("--restore-game-speed", action="store_true")
    parser.add_argument(
        "--reset-before-terminal-test",
        action="store_true",
        help="Soft-reset before terminal long-run. Off by default so naturally started levels can complete normally.",
    )
    parser.add_argument("--smoke-runs", type=int, default=3)
    parser.add_argument("--wait-for-board", action="store_true", help="Wait until Board exists before observe/action checks.")
    parser.add_argument("--skip-board-wait", action="store_true", help="Do not auto-wait for board before smoke/baseline commands.")
    parser.add_argument("--wait-gameplay-ready", action="store_true", help="Wait until Board and configured seed cards are ready for legal plant actions.")
    parser.add_argument("--board-timeout", type=float, default=None)
    parser.add_argument("--gameplay-ready-timeout", type=float, default=None)
    parser.add_argument("--poll-seconds", type=float, default=None)
    parser.add_argument(
        "--quick-wait",
        action="store_true",
        help="Fast wait defaults: board-timeout 60, gameplay-ready-timeout 30, poll-seconds 0.2.",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--no-timeout", action="store_true", help="Equivalent to --max-steps 0; only win/loss or bridge errors end an episode.")
    parser.add_argument("--policy", choices=("wait", "random", "teacher"), default="teacher")
    parser.add_argument("--fusion-policy", choices=("none", "observe", "scripted", "assist"), default="none")
    parser.add_argument("--step-seconds", type=float, default=None)
    parser.add_argument("--game-speed", type=float, default=1.0)
    parser.add_argument("--game-speed-mode", choices=("game_speed", "time_scale", "safe"), default="game_speed")
    parser.add_argument("--valid-speed-mode", action="store_true")
    parser.add_argument("--start-sun", type=int, default=None)
    parser.add_argument(
        "--plant-types",
        default=None,
        help="Comma-separated PlantType ids. Default: 1,1,0,0 (Generalist initial loadout).",
    )
    parser.add_argument("--action", type=int)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON for evaluation logs.")
    args = parser.parse_args()

    if args.quick_wait:
        if args.board_timeout is None:
            args.board_timeout = 60.0
        if args.gameplay_ready_timeout is None:
            args.gameplay_ready_timeout = 30.0
        if args.poll_seconds is None:
            args.poll_seconds = 0.2

    if args.board_timeout is None:
        args.board_timeout = 180.0
    if args.gameplay_ready_timeout is None:
        args.gameplay_ready_timeout = 60.0
    if args.poll_seconds is None:
        args.poll_seconds = 0.2

    default_config = PvZEnvConfig()
    start_sun_was_set = args.start_sun is not None
    start_sun = args.start_sun if args.start_sun is not None else default_config.start_sun
    parsed_seed_list = parse_seed_list(args.seed_list)
    configured_plant_types = parse_plant_types(args.plant_types) if args.plant_types else resolve_seed_list(parsed_seed_list)
    game_speed_mode = args.game_speed_mode
    if args.valid_speed_mode:
        game_speed_mode = "safe"
        if args.game_speed != 1.0:
            print("valid-speed-mode enabled; requested --game-speed will be treated as unsafe and not applied.")
    effective_step_seconds = (
        float(args.step_seconds)
        if args.step_seconds is not None
        else 0.05 if (args.ppo_loop_sun_drift_test or args.cooldown_mask_test) else 0.25
    )
    config = PvZEnvConfig(
        host=args.host,
        port=args.port,
        step_seconds=effective_step_seconds,
        plant_types=configured_plant_types,
        game_speed=args.game_speed,
        game_speed_mode=game_speed_mode,
        start_sun=start_sun,
        reset_wait_timeout=args.gameplay_ready_timeout,
        reset_poll_seconds=args.poll_seconds,
        auto_select_seeds=args.auto_select_seeds,
        seed_list=parsed_seed_list,
        seed_click_delay=max(0.0, args.seed_click_delay),
        lets_rock_delay=max(0.0, args.lets_rock_delay),
        post_start_delay=max(0.0, args.post_start_delay),
        seed_screen_check_interval=max(0, args.seed_screen_check_interval),
        debug_performance=bool(args.debug_performance),
        debug_observation=bool(args.debug_observation),
        debug_sun=bool(args.debug_sun),
        debug_sun_sample_interval=max(0, args.debug_sun_sample_interval),
        fusion_policy=args.fusion_policy,
        game_exe=str(args.game_exe) if args.game_exe else None,
    )
    max_steps = 0 if args.no_timeout else max(0, args.max_steps)
    env = PvZGymEnv(config)
    try:
        configured = False

        def ensure_configured() -> None:
            nonlocal configured
            if not configured:
                env.configure()
                configured = True

        def prepare_active_board() -> None:
            if not args.skip_board_wait:
                env.wait_for_board(timeout=args.board_timeout, poll_seconds=args.poll_seconds)
            ensure_configured()
            if args.auto_select_seeds:
                current = env.observe()
                seed_state = env.seed_probe() if current.get("boardFound") else {}
                if current.get("boardFound") and (current.get("done") or seed_state.get("blockingRewardUiActive")):
                    env.reset()
                else:
                    env.ensure_seeds_then_gameplay_ready(
                        seed_list=parse_seed_list(args.seed_list),
                        timeout=args.gameplay_ready_timeout,
                        poll_seconds=args.poll_seconds,
                    )
            elif args.wait_gameplay_ready:
                env.wait_for_gameplay_ready(timeout=args.gameplay_ready_timeout, poll_seconds=args.poll_seconds)

        exit_code = 0
        if args.restore_game_speed:
            print(json.dumps(env.restore_game_speed(), indent=2))
            return 0
        if args.wait_for_board:
            wait_started = time.monotonic()
            observation = env.wait_for_board(timeout=args.board_timeout, poll_seconds=args.poll_seconds)
            ensure_configured()
            if args.auto_select_seeds:
                seed_state = env.seed_probe() if observation.get("boardFound") else {}
                if observation.get("done") or seed_state.get("blockingRewardUiActive"):
                    observation, reset_info = env.reset()
                    selection = reset_info.get("reset", {}).get("autoSelectSeeds", {"ok": True, "terminalReset": True})
                else:
                    observation, selection = env.ensure_seeds_then_gameplay_ready(
                        seed_list=parse_seed_list(args.seed_list),
                        timeout=args.gameplay_ready_timeout,
                        poll_seconds=args.poll_seconds,
                    )
                print(json.dumps({"autoSelectSeeds": selection}, indent=2))
            elif args.wait_gameplay_ready:
                observation = env.wait_for_gameplay_ready(timeout=args.gameplay_ready_timeout, poll_seconds=args.poll_seconds)
            print(f"Total wait time: {time.monotonic() - wait_started:.2f} seconds.")
            print(json.dumps(observation, indent=2))
        if args.proof:
            ensure_configured()
            print(json.dumps(env.proof(place_test=args.place_test), indent=2))
        if args.observe:
            ensure_configured()
            if args.wait_gameplay_ready:
                env.wait_for_gameplay_ready(timeout=args.gameplay_ready_timeout, poll_seconds=args.poll_seconds)
            print(json.dumps(env.observe(), indent=2))
        if args.legal_actions:
            ensure_configured()
            if args.wait_gameplay_ready:
                env.wait_for_gameplay_ready(timeout=args.gameplay_ready_timeout, poll_seconds=args.poll_seconds)
            if start_sun_was_set:
                env.soft_reset(start_sun=start_sun, run_init=False)
            print(json.dumps(env.client.request("legal_actions"), indent=2))
        if args.teacher_action:
            ensure_configured()
            if args.wait_gameplay_ready:
                env.wait_for_gameplay_ready(timeout=args.gameplay_ready_timeout, poll_seconds=args.poll_seconds)
            print(json.dumps(env.client.request("teacher_action"), indent=2))
        if args.seed_probe:
            ensure_configured()
            print(json.dumps(env.seed_probe(), indent=2))
        if args.ui_probe:
            ensure_configured()
            print(json.dumps(env.ui_probe(include_all=args.include_all_ui, max_entries=args.ui_probe_max), indent=2))
        if args.almanac_probe:
            ensure_configured()
            print(json.dumps(env.almanac_probe(include_all=args.include_all_plants), indent=2))
        standalone_auto_select = (
            args.auto_select_seeds
            and not args.wait_for_board
            and not args.proof
            and not args.observe
            and not args.legal_actions
            and not args.teacher_action
            and not args.seed_probe
            and not args.ui_probe
            and not args.almanac_probe
            and not args.smoke_test
            and not args.adventure_state_smoke
            and not args.sun_cost_test
            and not args.cooldown_test
            and not args.fusion_semantics_test
            and not args.coach_fusion_scope_test
            and not args.reset_cleanup_test
            and not args.reset_state_machine_test
            and not args.reset_stress_test
            and not args.restart_screen_detection_test
            and not args.pause_menu_restart_test
            and not args.auto_select_seeds_test
            and not args.fresh_seed_select_test
            and not args.seed_screen_gating_test
            and not args.dump_action_map
            and not args.eval_baselines
            and not args.validate_reliability
            and not args.terminal_test_long_run
            and not args.terminal_auto_reset_test
            and not args.sun_spawn_rate_test
            and not args.ppo_loop_sun_drift_test
            and not args.cooldown_mask_test
            and not args.loss_reset_invariant_negative_test
            and args.action is None
        )
        if standalone_auto_select:
            ensure_configured()
            print(json.dumps(env.auto_select_seeds(seed_list=parse_seed_list(args.seed_list), start_level=True), indent=2))
        if args.action is not None:
            ensure_configured()
            if args.wait_gameplay_ready:
                env.wait_for_gameplay_ready(timeout=args.gameplay_ready_timeout, poll_seconds=args.poll_seconds)
            obs, reward, done, truncated, info = env.step(
                args.action,
                action_source="debug",
                action_source_metadata={"entrypoint": "--action"},
            )
            print(json.dumps({"observation": obs, "reward": reward, "done": done, "truncated": truncated, "info": info}, indent=2))
        if args.smoke_test:
            prepare_active_board()
            exit_code = 0 if smoke_test(
                env,
                wait_for_board=False,
                board_timeout=args.board_timeout,
                setup_reset=start_sun_was_set,
            ) else 1
        if args.adventure_state_smoke:
            ensure_configured()
            exit_code = 0 if adventure_state_smoke(
                env,
                duration_seconds=args.duration_seconds,
                poll_seconds=args.poll_seconds,
                auto_select_seeds=args.auto_select_seeds,
                seed_list=parse_seed_list(args.seed_list),
            ) else 1
        if args.sun_cost_test:
            prepare_active_board()
            exit_code = 0 if sun_cost_test(env) else 1
        if args.cooldown_test:
            prepare_active_board()
            exit_code = 0 if cooldown_test(env) else 1
        if args.fusion_semantics_test:
            prepare_active_board()
            exit_code = 0 if fusion_semantics_test(env) else 1
        if args.coach_fusion_scope_test:
            prepare_active_board()
            exit_code = 0 if coach_fusion_scope_test(env) else 1
        if args.reset_cleanup_test:
            prepare_active_board()
            exit_code = 0 if reset_cleanup_test(env, args.episodes) else 1
        if args.reset_state_machine_test:
            ensure_configured()
            observation, info = env.reset()
            payload = {"observation": observation, "info": info}
            print(json.dumps(payload, indent=2))
            reset_info = info.get("reset", {})
            exit_code = 0 if reset_info.get("resetSuccess") else 1
        if args.reset_stress_test:
            if not args.skip_board_wait:
                env.wait_for_board(timeout=args.board_timeout, poll_seconds=args.poll_seconds)
            ensure_configured()
            report = reset_stress_test(env, args.episodes)
            exit_code = 0 if report.get("passed") else 1
        if args.restart_screen_detection_test:
            ensure_configured()
            exit_code = 0 if restart_screen_detection_test(env, fast_only=args.fast_only) else 1
        if args.pause_menu_restart_test:
            ensure_configured()
            exit_code = 0 if pause_menu_restart_test(env) else 1
        if args.auto_select_seeds_test:
            if not args.skip_board_wait:
                env.wait_for_board(timeout=args.board_timeout, poll_seconds=args.poll_seconds)
            ensure_configured()
            exit_code = 0 if auto_select_seeds_test(env, parse_seed_list(args.seed_list), args.episodes) else 1
        if args.fresh_seed_select_test:
            if not args.skip_board_wait:
                env.wait_for_board(timeout=args.board_timeout, poll_seconds=args.poll_seconds)
            ensure_configured()
            exit_code = 0 if fresh_seed_select_test(env, parse_seed_list(args.seed_list)) else 1
        if args.seed_screen_gating_test:
            if not args.skip_board_wait:
                env.wait_for_board(timeout=args.board_timeout, poll_seconds=args.poll_seconds)
            ensure_configured()
            exit_code = 0 if seed_screen_gating_test(env) else 1
        if args.dump_action_map:
            prepare_active_board()
            observation = env.observe()
            dump_action_map(env, observation)
        if args.eval_baselines:
            prepare_active_board()
            results = evaluate_baselines(env, args.episodes, max_steps)
            print_baseline_table(results)
            if args.json:
                print(json.dumps({policy: [asdict(log) for log in logs] for policy, logs in results.items()}, indent=2))
        if args.sun_spawn_rate_test:
            prepare_active_board()
            sun_spawn_rate_test(env, args.duration_seconds, args.no_planting)
        if args.ppo_loop_sun_drift_test:
            if not args.skip_board_wait:
                env.wait_for_board(timeout=args.board_timeout, poll_seconds=args.poll_seconds)
            ensure_configured()
            ppo_loop_sun_drift_test(
                env,
                steps=max(0, args.steps),
                no_planting=args.no_planting,
                log_interval=max(0, args.sun_drift_log_interval),
                wait_gameplay_ready=args.wait_gameplay_ready,
            )
        if args.cooldown_mask_test:
            if not args.skip_board_wait:
                env.wait_for_board(timeout=args.board_timeout, poll_seconds=args.poll_seconds)
            ensure_configured()
            report = cooldown_mask_test(
                env,
                steps=max(0, args.steps),
                output_path=args.diagnostic_output,
                log_interval=max(0, args.sun_drift_log_interval),
            )
            exit_code = 0 if report.get("passed") else 1
        if args.loss_reset_invariant_negative_test:
            exit_code = 0 if loss_reset_invariant_negative_test(env) else 1
        if args.validate_reliability:
            report = validate_reliability(env, args.smoke_runs, args.board_timeout, max_steps)
            print(json.dumps(report, indent=2))
            exit_code = 0 if report.get("ready_for_ppo") else 1
        if args.terminal_test_long_run:
            prepare_active_board()
            log = run_episode(
                env,
                args.policy,
                0,
                max_steps,
                random.Random(env.config.seed),
                reset_each_episode=args.reset_before_terminal_test,
            )
            print_terminal_test(log, max_steps)
            if args.json:
                print(json.dumps(asdict(log), indent=2))
            exit_code = 0 if log.actual_terminal else 1
        if args.terminal_auto_reset_test:
            prepare_active_board()
            exit_code = 0 if terminal_auto_reset_test(env, args.episodes, max_steps, args.policy) else 1
        return exit_code
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
