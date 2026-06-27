"""Gymnasium adapter for MaskablePPO training.

This module turns the existing bridge-backed PvZGymEnv into a small fixed-shape
Gymnasium environment. It does not change bridge behavior, rewards, plant types,
or reset semantics.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pvzrl_action_space import (
    ACTION_SPACE_ADVENTURE_14_IDENTITY,
    ACTION_SPACE_FIXED,
    DYNAMIC_WAIT_ACTION,
    build_action_space_spec,
    decode_policy_action,
    legacy_action_to_policy_action,
    normalize_action_space_mode,
    policy_action_to_legacy_action,
)
from pvzrl_env import (
    DEFAULT_PLANT_TYPES,
    REWARD_COMPONENT_FIELDS,
    REWARD_EPISODE_TOTAL_FIELDS,
    RUN_MODE_ADVENTURE_GENERALIST_14SLOT_EVAL,
    RUN_MODE_ADVENTURE_GENERALIST_14SLOT_TRAIN,
    RUN_MODE_ADVENTURE_EVAL,
    RUN_MODE_FIXED_TRAIN,
    RUN_MODE_LEVEL3_SPECIALIST,
    RUN_MODES,
    PvZEnvConfig,
    PvZGymEnv,
    RewardConfig,
    classify_done_reason,
    resolve_seed_list,
)
from pvzrl_fusion import FUSION_POLICY_NONE, merge_episode_fusion_stats, normalize_fusion_policy
from pvzrl_human_coach import (
    COACH_REWARD_LEGAL_EXECUTION_COMPONENT,
    COACH_REWARD_MATCH_COMPONENT,
    COACH_REWARD_OVERRIDE_PENALTY_COMPONENT,
    FileCoachCommandSource,
    HumanCoachOverrideHook,
    build_env_fusion_probe,
    human_coach_live_status_defaults,
    human_coach_live_status_from_hook,
)
from pvzrl_seed_inventory import (
    adventure_identity_feature_count,
    adventure_identity_features,
    seed_inventory_v2_feature_count,
    seed_inventory_v2_features,
)
from pvzrl_stream_coach import StreamCoachController
from pvzrl_assisted_coach import InterventionJSONLLogger


@dataclass
class PvZSB3Config:
    host: str = "127.0.0.1"
    port: int = 32323
    timeout: float = 10.0
    step_seconds: float = 0.05
    plant_types: List[int] = None  # type: ignore[assignment]
    action_space_mode: str = ACTION_SPACE_FIXED
    max_seed_slots: Optional[int] = None
    observation_version: str = ""
    action_decoder_version: str = ""
    dynamic_seed_slots: bool = False
    row_count: int = 5
    column_count: int = 10
    game_speed: float = 4.0
    game_speed_mode: str = "game_speed"
    seed: int = 12345
    start_sun: int = 500
    max_steps: int = 1000
    wait_for_board: bool = True
    wait_gameplay_ready: bool = True
    board_timeout: float = 60.0
    gameplay_ready_timeout: float = 30.0
    poll_seconds: float = 0.2
    auto_select_seeds: bool = False
    seed_list: List[str] = field(default_factory=lambda: ["SunFlower", "Peashooter"])
    seed_click_delay: float = 0.35
    lets_rock_delay: float = 0.5
    post_start_delay: float = 1.0
    seed_screen_check_interval: int = 100
    debug_performance: bool = False
    debug_observation: bool = False
    debug_sun: bool = False
    debug_sun_sample_interval: int = 25
    fusion_policy: str = FUSION_POLICY_NONE
    fusion_action_mask_enabled: bool = False
    run_mode: str = RUN_MODE_FIXED_TRAIN
    target_level: int = 0
    tactical_masks: bool = False
    wallnut_tactical_mask: bool = False
    cherrybomb_tactical_mask: bool = False
    adventure_eval_mode: bool = False
    game_exe: Optional[str] = None
    human_coach_enabled: bool = False
    human_coach_log_path: str = ""
    human_coach_command_path: str = ""
    human_coach_reward: bool = False
    human_coach_fusion_enabled: bool = False
    human_coach_platform: str = "mock"
    human_coach_command_mode: str = "override"
    intervention_log_path: str = "logs/interventions/interventions.jsonl"
    stream_coach_enabled: bool = False
    stream_coach_mode: str = "mock"
    stream_coach_platform: str = "mock"
    stream_coach_window_sec: float = 3.0
    stream_coach_min_votes: int = 2
    stream_coach_max_actions_per_minute: int = 20
    stream_coach_reward: bool = False
    stream_coach_log_path: str = ""
    stream_coach_command_path: str = ""
    stream_coach_mock_script: str = ""
    stream_coach_dry_run: bool = True
    stream_coach_apply_enabled: bool = False
    stream_coach_fusion_enabled: bool = False
    reward: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self) -> None:
        requested_run_mode = str(self.run_mode or "").strip().lower()
        if requested_run_mode not in RUN_MODES:
            requested_run_mode = RUN_MODE_ADVENTURE_EVAL if bool(self.adventure_eval_mode) else RUN_MODE_FIXED_TRAIN
        self.run_mode = requested_run_mode
        self.adventure_eval_mode = requested_run_mode in {
            RUN_MODE_ADVENTURE_EVAL,
            RUN_MODE_ADVENTURE_GENERALIST_14SLOT_TRAIN,
            RUN_MODE_ADVENTURE_GENERALIST_14SLOT_EVAL,
        }
        self.fusion_policy = normalize_fusion_policy(self.fusion_policy)
        if self.plant_types is None:
            try:
                self.plant_types = resolve_seed_list(list(self.seed_list))
            except Exception:
                self.plant_types = list(DEFAULT_PLANT_TYPES)
        self.action_space_mode = normalize_action_space_mode(self.action_space_mode)
        spec = build_action_space_spec(
            mode=self.action_space_mode,
            plant_types=list(self.plant_types),
            max_seed_slots=self.max_seed_slots,
            rows=self.row_count,
            cols=self.column_count,
        )
        self.max_seed_slots = int(spec.max_seed_slots)
        self.observation_version = self.observation_version or spec.observation_version
        self.action_decoder_version = self.action_decoder_version or spec.action_decoder_version
        self.dynamic_seed_slots = spec.dynamic_seed_slots

    def get_env_metadata(self) -> Dict[str, Any]:
        spec = build_action_space_spec(
            mode=self.action_space_mode,
            plant_types=list(self.plant_types),
            max_seed_slots=self.max_seed_slots,
            rows=self.row_count,
            cols=self.column_count,
        )
        return {
            "resolved_seed_list": list(self.seed_list),
            "resolved_plant_types": [int(value) for value in self.plant_types],
            "env_action_count": int(spec.action_count),
            "action_space_mode": spec.mode,
            "max_seed_slots": int(spec.max_seed_slots),
            "dynamic_seed_slots": bool(spec.dynamic_seed_slots),
            "identity_seed_slots": bool(spec.identity_seed_slots),
            "observation_version": str(self.observation_version or spec.observation_version),
            "action_decoder_version": str(self.action_decoder_version or spec.action_decoder_version),
            "decoder_wait_action": int(spec.wait_action),
            "placement_action_range": [int(spec.placement_action_min), int(spec.placement_action_max)],
            "rows": int(spec.rows),
            "cols": int(spec.cols),
            "cells_per_seed_slot": int(spec.rows) * int(spec.cols),
        }


def _stream_raw_text(command: Optional[Dict[str, Any]]) -> str:
    if not isinstance(command, dict):
        return ""
    name = str(command.get("command") or "").strip().lower()
    if name in {"plant", "fuse"}:
        return f"!{name} {int(command.get('seed_index', -1))} {int(command.get('row', -1))} {int(command.get('col', -1))}"
    if name == "defend":
        return f"!defend {int(command.get('row', -1))}"
    if name == "economy":
        return "!economy"
    if name == "wait":
        return "!wait"
    return ""


class PvZMaskedPPOEnv(gym.Env[np.ndarray, int]):
    """Fixed-shape Gymnasium wrapper around the live PvZ bridge environment."""

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[PvZSB3Config] = None):
        super().__init__()
        self.config = config or PvZSB3Config()
        env_config = PvZEnvConfig(
            host=self.config.host,
            port=self.config.port,
            timeout=self.config.timeout,
            step_seconds=self.config.step_seconds,
            plant_types=list(self.config.plant_types),
            row_count=self.config.row_count,
            column_count=self.config.column_count,
            game_speed=self.config.game_speed,
            game_speed_mode=self.config.game_speed_mode,
            seed=self.config.seed,
            start_sun=self.config.start_sun,
            reset_wait_timeout=self.config.gameplay_ready_timeout,
            reset_poll_seconds=self.config.poll_seconds,
            auto_select_seeds=self.config.auto_select_seeds,
            seed_list=list(self.config.seed_list),
            seed_click_delay=self.config.seed_click_delay,
            lets_rock_delay=self.config.lets_rock_delay,
            post_start_delay=self.config.post_start_delay,
            seed_screen_check_interval=self.config.seed_screen_check_interval,
            debug_performance=self.config.debug_performance,
            debug_observation=self.config.debug_observation,
            debug_sun=self.config.debug_sun,
            debug_sun_sample_interval=self.config.debug_sun_sample_interval,
            fusion_policy=self.config.fusion_policy,
            fusion_action_mask_enabled=self.config.fusion_action_mask_enabled,
            run_mode=self.config.run_mode,
            target_level=self.config.target_level,
            tactical_masks=self.config.tactical_masks,
            wallnut_tactical_mask=self.config.wallnut_tactical_mask,
            cherrybomb_tactical_mask=self.config.cherrybomb_tactical_mask,
            adventure_eval_mode=self.config.adventure_eval_mode,
            game_exe=self.config.game_exe,
            reward=self.config.reward,
        )
        self.base = PvZGymEnv(env_config)
        self.rows = self.config.row_count
        self.cols = self.config.column_count
        self.cells = self.rows * self.cols
        self.action_spec = build_action_space_spec(
            mode=self.config.action_space_mode,
            plant_types=list(self.config.plant_types),
            max_seed_slots=self.config.max_seed_slots,
            rows=self.rows,
            cols=self.cols,
        )
        self.action_count = self.action_spec.action_count
        self.global_features = 12
        self.card_slot_count = self.action_spec.max_seed_slots if self.action_spec.dynamic_seed_slots else len(self.config.plant_types)
        self.card_features = 5 * self.card_slot_count
        self.cell_features = 6 * self.cells
        self.lane_features = 5 * self.rows
        if self.action_spec.mode == ACTION_SPACE_ADVENTURE_14_IDENTITY:
            self.seed_inventory_features = adventure_identity_feature_count(self.action_spec.max_seed_slots)
        elif self.action_spec.dynamic_seed_slots:
            self.seed_inventory_features = seed_inventory_v2_feature_count(self.action_spec.max_seed_slots)
        else:
            self.seed_inventory_features = 0
        self.observation_size = (
            self.global_features
            + self.card_features
            + self.cell_features
            + self.lane_features
            + self.seed_inventory_features
        )
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(self.observation_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.action_count)
        self._last_observation: Optional[Dict[str, Any]] = None
        self._step_count = 0
        self._episode_reward = 0.0
        self._episode_plants = 0
        self._episode_illegal = 0
        self._episode_index = -1
        self._episode_sun_spent = 0
        self._episode_legal_action_total = 0
        self._start_kills = 0
        self._start_mowers = 0
        self._last_reset_success = True
        self._last_reset_seconds = 0.0
        self._last_action_mask_ms = 0.0
        self._perf_samples = 0
        self._perf_totals: Dict[str, float] = {}
        self._perf_report_interval = 100
        self._global_step_count = 0
        self._sun_diag_report_interval = 500
        self._sun_diag_spawn_window = 0
        self._sun_diag_gained_window = 0
        self._sun_diag_last_sun: Optional[int] = None
        self._sun_diag_last_active_count: Optional[int] = None
        self._next_reset_reason = ""
        self._allow_active_gameplay_reset_next = False
        self._last_episode_ended_by_timeout = False
        self._last_episode_ended_by_win = False
        self._reset_requires_seed_flow_next = False
        self.human_coach_hook: Optional[HumanCoachOverrideHook] = None
        self.stream_coach_controller: Optional[StreamCoachController] = None
        self.intervention_logger = InterventionJSONLLogger(
            Path(self.config.intervention_log_path or "logs/interventions/interventions.jsonl")
        )
        self._stream_coach_source_path = str(self.config.stream_coach_command_path or "")
        self._fusion_probe = build_env_fusion_probe(self)
        if self.config.human_coach_enabled:
            source = FileCoachCommandSource(self.config.human_coach_command_path) if self.config.human_coach_command_path else None
            self.human_coach_hook = HumanCoachOverrideHook(
                enabled=True,
                source=source,
                log_path=self.config.human_coach_log_path or None,
                reward_enabled=bool(self.config.human_coach_reward),
                fusion_enabled=bool(self.config.human_coach_fusion_enabled),
                platform=self.config.human_coach_platform or "mock",
                command_mode=self.config.human_coach_command_mode or "override",
                match_reward=float(getattr(self.config.reward, "coach_match_reward", 0.02)),
                legal_execution_reward=float(getattr(self.config.reward, "coach_legal_execution_reward", 0.01)),
                override_penalty=float(getattr(self.config.reward, "coach_override_penalty", -0.01)),
                fusion_success_reward=float(getattr(self.config.reward, "coach_fusion_success_reward", 0.03)),
                tactical_usefulness_reward=float(getattr(self.config.reward, "coach_tactical_usefulness_reward", 0.01)),
            )
        if self.config.stream_coach_enabled:
            stream_command_path = None
            if self.config.stream_coach_command_path:
                stream_command_path = Path(self.config.stream_coach_command_path)
            mock_script_path = None
            if self.config.stream_coach_mock_script:
                mock_script_path = Path(self.config.stream_coach_mock_script)
            self.stream_coach_controller = StreamCoachController(
                enabled=True,
                mode=self.config.stream_coach_mode or self.config.stream_coach_platform or "mock",
                platform=self.config.stream_coach_platform or "mock",
                command_path=stream_command_path,
                mock_script_path=mock_script_path,
                dry_run=bool(self.config.stream_coach_dry_run),
                apply_enabled=bool(self.config.stream_coach_apply_enabled),
                window_sec=float(self.config.stream_coach_window_sec),
                min_votes=int(self.config.stream_coach_min_votes),
                max_actions_per_minute=int(self.config.stream_coach_max_actions_per_minute),
                log_path=Path(self.config.stream_coach_log_path) if self.config.stream_coach_log_path else None,
            )
            self._stream_coach_source_path = str(stream_command_path or "")
        print(f"[coach] human coach enabled={bool(self.config.human_coach_enabled)}")
        print(
            "[coach] command queue path="
            f"{str(self.config.human_coach_command_path or self.config.stream_coach_command_path or '')}"
        )
        print(f"[coach] action_count={int(self.action_count)}")
        print(f"[coach] decoder={self.action_spec.action_decoder_version}")
        print(f"[coach] fusion bridge available={bool(self._fusion_probe is not None)}")
        print(f"[coach] stream coach enabled={bool(self.config.stream_coach_enabled)}")
        if self.config.stream_coach_enabled:
            print(f"[coach] stream coach mode={self.config.stream_coach_mode or self.config.stream_coach_platform or 'mock'}")
            print(f"[coach] stream coach dry_run={bool(self.config.stream_coach_dry_run)} apply_enabled={bool(self.config.stream_coach_apply_enabled)}")
            print(f"[coach] stream command queue path={self._stream_coach_source_path}")
            if self.config.stream_coach_mock_script:
                print(f"[coach] mock stream script={self.config.stream_coach_mock_script}")
        self._reset_lane_episode_diagnostics({})

    def get_env_metadata(self) -> Dict[str, Any]:
        metadata = self.config.get_env_metadata()
        metadata["env_action_count"] = int(self.action_count)
        metadata["action_space_mode"] = self.action_spec.mode
        metadata["max_seed_slots"] = int(self.action_spec.max_seed_slots)
        metadata["dynamic_seed_slots"] = bool(self.action_spec.dynamic_seed_slots)
        metadata["identity_seed_slots"] = bool(self.action_spec.identity_seed_slots)
        metadata["observation_version"] = self.action_spec.observation_version
        metadata["action_decoder_version"] = self.action_spec.action_decoder_version
        return metadata

    def _clear_coach_command_state_on_reset(self, *, reason: str = "reset") -> bool:
        stale_detected = False
        if self.human_coach_hook is not None and hasattr(self.human_coach_hook, "clear_pending_state"):
            try:
                stale_detected = bool(
                    self.human_coach_hook.clear_pending_state(
                        clear_source=True,
                        reason=reason,
                        preserve_startup_blocked=str(reason or "").endswith("_ready"),
                    )
                    or stale_detected
                )
            except Exception:
                stale_detected = True
        if self.stream_coach_controller is not None and hasattr(self.stream_coach_controller, "clear_pending_state"):
            try:
                stale_detected = bool(
                    self.stream_coach_controller.clear_pending_state(clear_source=True, reason=reason)
                    or stale_detected
                )
            except Exception:
                stale_detected = True
        if hasattr(self.base, "clear_coach_runtime_state"):
            self.base.clear_coach_runtime_state(
                queue_cleared=True,
                startup_command_blocked=bool(stale_detected),
                reason=reason,
            )
        return bool(stale_detected)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        option_reset_reason = str((options or {}).get("reset_reason") or "")
        reset_reason = option_reset_reason or self._next_reset_reason
        reset_requires_seed_flow = bool(self._reset_requires_seed_flow_next)
        allow_active_gameplay_reset = bool(
            (options or {}).get("allow_active_gameplay_reset")
            or self._allow_active_gameplay_reset_next
        )
        self._next_reset_reason = ""
        self._allow_active_gameplay_reset_next = False
        self._reset_requires_seed_flow_next = False
        startup_stale_detected = self._clear_coach_command_state_on_reset(reason=f"{reset_reason or 'reset'}_start")
        if self.config.wait_for_board:
            self.base.wait_for_board(
                timeout=self.config.board_timeout,
                poll_seconds=self.config.poll_seconds,
                quiet=True,
            )
        self.base.configure()

        observation, reset_info = self.base.reset(
            reset_reason=reset_reason,
            allow_active_gameplay_reset=allow_active_gameplay_reset,
        )
        if self.config.wait_gameplay_ready:
            try:
                observation = self.base.wait_for_gameplay_ready(
                    timeout=self.config.gameplay_ready_timeout,
                    poll_seconds=self.config.poll_seconds,
                    quiet=True,
                )
            except RuntimeError:
                if allow_active_gameplay_reset and reset_reason == "env_corruption":
                    observation, reset_info = self.base.soft_reset(
                        start_sun=self.config.start_sun,
                        run_init=False,
                        allow_active_gameplay_reset=allow_active_gameplay_reset,
                        reset_reason=reset_reason,
                    )
                else:
                    raise
        stale_detected = bool(
            self._clear_coach_command_state_on_reset(reason=f"{reset_reason or 'reset'}_ready")
            or startup_stale_detected
        )
        if bool(startup_stale_detected) and hasattr(self.base, "clear_coach_runtime_state"):
            self.base.clear_coach_runtime_state(
                queue_cleared=True,
                startup_command_blocked=True,
                reason=f"{reset_reason or 'reset'}_ready",
            )

        reset_payload = reset_info.get("reset", {}) if isinstance(reset_info, dict) else {}
        reset_payload["coach_command_queue_cleared_on_reset"] = True
        reset_payload["startup_command_blocked"] = bool(stale_detected)
        if reset_requires_seed_flow:
            reset_payload["wrapperRequiredSeedFlow"] = True
        self._last_episode_ended_by_timeout = False
        self._last_episode_ended_by_win = False
        self._last_observation = observation
        self._episode_index += 1
        self._step_count = 0
        self._episode_reward = 0.0
        self._episode_plants = 0
        self._episode_illegal = 0
        self._episode_sun_spent = 0
        self._episode_legal_action_total = 0
        self._start_kills = int(observation.get("killCount", 0))
        self._start_mowers = int(observation.get("logicalMowerCount", observation.get("rowCount", self.rows)))
        self._last_reset_success = bool(reset_payload.get("resetSuccess", reset_payload.get("ok", True)))
        self._last_reset_seconds = float(
            reset_payload.get("timeToPlayableSeconds")
            or (float(reset_payload.get("reset_ms", 0.0) or 0.0) / 1000.0)
            or 0.0
        )
        self._reset_lane_episode_diagnostics(observation)
        self._episode_reset_reward_ui_cleanup_count = self._safe_int_value(
            reset_payload.get("resetRewardUiCleanupCount"),
            default=0,
        )
        self._episode_reset_reward_ui_cleanup_blocked_count = self._safe_int_value(
            reset_payload.get("resetRewardUiCleanupBlockedCount"),
            default=0,
        )
        self._episode_reset_after_false_reward_signal_count = self._safe_int_value(
            reset_payload.get("resetAfterFalseRewardSignalCount"),
            default=0,
        )
        self._episode_blocked_cleanup_during_gameplay_count += self._safe_int_value(
            reset_payload.get("blockedCleanupDuringGameplayCount"),
            default=0,
        )
        self._episode_suspicious_cleanup_reward_ui_count += self._safe_int_value(
            reset_payload.get("suspiciousCleanupRewardUiCount"),
            default=0,
        )
        self._reset_sun_diagnostic_baseline(observation)
        return self._encode_observation(observation), {"raw_observation": observation, **reset_info}

    def start_episode_from_observation(
        self,
        observation: Dict[str, Any],
        reset_info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Begin wrapper episode accounting from an already active Adventure board."""
        reset_info = reset_info or {"reset": {"ok": True, "methodUsed": "adventure_existing_board"}}
        reset_payload = reset_info.get("reset", {}) if isinstance(reset_info, dict) else {}
        method_used = str(reset_payload.get("methodUsed") or "adventure_existing_board")
        self.base.begin_new_attempt(observation, reason=f"adventure:{method_used}")
        stale_detected = self._clear_coach_command_state_on_reset(reason=f"adventure_{method_used}_start")
        reset_payload["coach_command_queue_cleared_on_reset"] = True
        reset_payload["startup_command_blocked"] = bool(stale_detected)
        legal_actions_for_log = observation.get("legalActions", [])
        legal_action_count_for_log = observation.get("legalActionCount")
        if legal_action_count_for_log is None:
            legal_action_count_for_log = len(legal_actions_for_log) if isinstance(legal_actions_for_log, list) else 0
        print(
            "[adventure] new attempt started "
            f"method={method_used} "
            f"wave={observation.get('wave')}/{observation.get('maxWave')} "
            f"plants={observation.get('plantCount')} "
            f"zombies={observation.get('zombieCount')} "
            f"mowers={observation.get('logicalMowerCount')} "
            f"seed_slots={observation.get('seedSlotCount')} "
            f"legalActionCount={legal_action_count_for_log}"
        )
        self._last_observation = observation
        self._episode_index += 1
        self._step_count = 0
        self._episode_reward = 0.0
        self._episode_plants = 0
        self._episode_illegal = 0
        self._episode_sun_spent = 0
        self._episode_legal_action_total = 0
        self._start_kills = int(observation.get("killCount", 0))
        self._start_mowers = int(observation.get("logicalMowerCount", observation.get("rowCount", self.rows)))
        self._last_reset_success = bool(reset_payload.get("resetSuccess", reset_payload.get("ok", True)))
        self._last_reset_seconds = float(
            reset_payload.get("timeToPlayableSeconds")
            or (float(reset_payload.get("reset_ms", 0.0) or 0.0) / 1000.0)
            or 0.0
        )
        self._reset_lane_episode_diagnostics(observation)
        self._episode_reset_reward_ui_cleanup_count = self._safe_int_value(
            reset_payload.get("resetRewardUiCleanupCount"),
            default=0,
        )
        self._episode_reset_reward_ui_cleanup_blocked_count = self._safe_int_value(
            reset_payload.get("resetRewardUiCleanupBlockedCount"),
            default=0,
        )
        self._episode_reset_after_false_reward_signal_count = self._safe_int_value(
            reset_payload.get("resetAfterFalseRewardSignalCount"),
            default=0,
        )
        self._reset_sun_diagnostic_baseline(observation)
        return self._encode_observation(observation), {"raw_observation": observation, **reset_info}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        ppo_action = int(action)
        policy_action = int(action)
        coach_decision = None
        stream_decision = None
        coach_context: Dict[str, Any] = {}
        selected_bridge_command: Optional[Dict[str, Any]] = None

        if self.human_coach_hook is not None:
            coach_decision = self.human_coach_hook.select_action(self, policy_action)
            policy_action = int(coach_decision.selected_action)
            selected_bridge_command = (
                dict(coach_decision.selected_bridge_command)
                if isinstance(coach_decision.selected_bridge_command, dict)
                else None
            )
            coach_context = {
                "enabled": bool(coach_decision.enabled),
                "source": "human",
                "rewardEnabled": bool(self.config.human_coach_reward),
                "event": str(coach_decision.event),
                "overrideApplied": bool(coach_decision.override_applied),
                "coachMatch": bool(coach_decision.coach_match),
                "selectedAction": int(coach_decision.selected_action),
                "ppoAction": int(coach_decision.ppo_action),
                "command": coach_decision.command.to_dict() if coach_decision.command is not None else None,
                "validation": coach_decision.validation.to_dict() if coach_decision.validation is not None else None,
            }
            command_payload = coach_decision.command.to_dict() if coach_decision.command is not None else None
            print(
                "[coach] source=human "
                f"raw={command_payload.get('raw_text') if isinstance(command_payload, dict) else ''!r} "
                f"parsed={command_payload} "
                f"action={policy_action} "
                f"legal={bool(not coach_decision.rejected)} "
                f"outcome={coach_decision.event} "
                f"reason={coach_decision.rejected_reason!r}"
            )
        elif self.stream_coach_controller is not None:
            observation_for_stream = self._last_observation if isinstance(self._last_observation, dict) else {}
            try:
                action_mask = self.action_masks()
            except Exception:
                action_mask = None
            legal_policy_actions = self._legal_actions_from_mask(action_mask)
            self.stream_coach_controller.poll_source(
                username="mock_source",
                step_index=self._global_step_count,
            )
            stream_fusion_enabled = bool(
                self.config.stream_coach_fusion_enabled
                or self.config.human_coach_fusion_enabled
                or self.fusion_policy != FUSION_POLICY_NONE
            )
            stream_decision = self.stream_coach_controller.choose_action(
                observation=observation_for_stream,
                legal_actions=legal_policy_actions,
                action_space_mode=self.action_spec.mode,
                ppo_action=ppo_action,
                fusion_enabled=stream_fusion_enabled,
                fusion_bridge_probe=self._fusion_probe,
            )
            stream_apply_enabled = bool(self.config.stream_coach_apply_enabled) and not bool(self.config.stream_coach_dry_run)
            if stream_decision.selected and stream_decision.selected_policy_action is not None:
                candidate_policy_action = int(stream_decision.selected_policy_action)
                if stream_apply_enabled:
                    policy_action = int(candidate_policy_action)
                    selected_bridge_command = (
                        dict(stream_decision.selected_bridge_command)
                        if isinstance(stream_decision.selected_bridge_command, dict)
                        else None
                    )
                elif self.stream_coach_controller is not None:
                    self.stream_coach_controller.record_dry_run_decision(stream_decision)
                stream_event = (
                    "coach_pending"
                    if bool(getattr(stream_decision, "pending", False))
                    else (
                        "coach_dry_run"
                        if not stream_apply_enabled
                        else ("coach_match" if bool(stream_decision.coach_match) else "coach_override")
                    )
                )
                if stream_apply_enabled:
                    coach_context = {
                        "enabled": True,
                        "source": "stream",
                        "rewardEnabled": bool(self.config.stream_coach_reward),
                        "event": stream_event,
                        "overrideApplied": bool(stream_decision.override_applied),
                        "coachMatch": bool(stream_decision.coach_match),
                        "selectedAction": int(policy_action),
                        "candidateAction": int(candidate_policy_action),
                        "ppoAction": int(ppo_action),
                        "dryRun": False,
                        "applyEnabled": True,
                        "command": dict(stream_decision.selected_command) if isinstance(stream_decision.selected_command, dict) else None,
                        "voteCount": int(stream_decision.selected_vote_count or 0),
                        "validation": {
                            "legal": True,
                            "policy_action": int(policy_action),
                        },
                    }
                print(
                    "[coach] source=stream "
                    f"raw={_stream_raw_text(stream_decision.selected_command)!r} "
                    f"parsed={stream_decision.selected_command} "
                    f"action={candidate_policy_action} "
                    f"dry_run={bool(not stream_apply_enabled)} "
                    "legal=True "
                    f"outcome={stream_event} "
                    f"reason={str(stream_decision.rejected_reason or '')!r}"
                )
            else:
                rejected_reason = str(stream_decision.rejected_reason or "")
                if rejected_reason not in {"", "no_legal_command", "below_vote_threshold"}:
                    print(
                        "[coach] source=stream raw='' parsed=None "
                        f"action={ppo_action} legal=False outcome=fallback_to_ppo "
                        f"reason={rejected_reason!r}"
                    )
        bridge_action = self._policy_action_to_bridge_action(policy_action)
        observation, reward, done, _, info = self.base.step(
            bridge_action,
            coach_bridge_command=selected_bridge_command,
            coach_context=coach_context if coach_context else None,
        )
        if coach_decision is not None and self.human_coach_hook is not None:
            coach_delta = float(self.human_coach_hook.apply_step_outcome(coach_decision, info if isinstance(info, dict) else {}))
            if coach_delta != 0.0:
                reward = float(reward) + coach_delta
        elif (
            stream_decision is not None
            and stream_decision.selected
            and bool(self.config.stream_coach_apply_enabled)
            and not bool(self.config.stream_coach_dry_run)
        ):
            stream_delta = self._apply_stream_coach_reward(stream_decision, info if isinstance(info, dict) else {})
            if self.stream_coach_controller is not None:
                self.stream_coach_controller.apply_step_outcome(stream_decision, info if isinstance(info, dict) else {})
            if stream_delta != 0.0:
                reward = float(reward) + stream_delta
                if self.stream_coach_controller is not None:
                    self.stream_coach_controller.aggregator.add_reward(stream_delta)
        if self.action_spec.dynamic_seed_slots:
            info["policy_action"] = policy_action
            info["bridge_action"] = bridge_action
            action_result = info.get("action_result", {})
            if isinstance(action_result, dict):
                action_result["policyAction"] = policy_action
                action_result["bridgeAction"] = bridge_action
                action_result["policyActionDecoderVersion"] = self.action_spec.action_decoder_version
        if coach_decision is not None:
            coach_payload = coach_decision.to_dict()
            coach_status = self._coach_live_status()
            info["human_coach"] = coach_payload
            info["stream_coach"] = coach_status
            action_result = info.get("action_result", {})
            if isinstance(action_result, dict):
                action_result["humanCoach"] = coach_payload
                action_result["humanCoachStatus"] = coach_status
        elif stream_decision is not None:
            coach_status = self._coach_live_status()
            info["stream_coach"] = coach_status
            stream_payload = {
                "enabled": bool(self.config.stream_coach_enabled),
                "source": "stream",
                "event": (
                    "coach_pending"
                    if bool(getattr(stream_decision, "pending", False))
                    else (
                        "coach_dry_run"
                        if bool(self.config.stream_coach_dry_run) or not bool(self.config.stream_coach_apply_enabled)
                        else (
                            "coach_match"
                            if bool(stream_decision.coach_match)
                            else ("coach_override" if stream_decision.selected else "fallback_to_ppo")
                        )
                    )
                ),
                "dry_run": bool(self.config.stream_coach_dry_run) or not bool(self.config.stream_coach_apply_enabled),
                "apply_enabled": bool(self.config.stream_coach_apply_enabled) and not bool(self.config.stream_coach_dry_run),
                "ppo_action": int(ppo_action),
                "selected_action": int(policy_action),
                "candidate_action": int(stream_decision.selected_policy_action) if stream_decision.selected_policy_action is not None else int(ppo_action),
                "override_applied": bool(stream_decision.override_applied)
                and bool(self.config.stream_coach_apply_enabled)
                and not bool(self.config.stream_coach_dry_run),
                "coach_match": bool(stream_decision.coach_match),
                "rejected": bool(not stream_decision.selected),
                "rejected_reason": str(stream_decision.rejected_reason or ""),
                "command": dict(stream_decision.selected_command) if isinstance(stream_decision.selected_command, dict) else None,
                "vote_count": int(stream_decision.selected_vote_count or 0),
            }
            info["human_coach"] = stream_payload
            action_result = info.get("action_result", {})
            if isinstance(action_result, dict):
                action_result["streamCoach"] = stream_payload
                action_result["streamCoachStatus"] = coach_status
        if "stream_coach" not in info:
            info["stream_coach"] = self._coach_live_status()
        if "human_coach" not in info:
            info["human_coach"] = {
                "enabled": bool(self.config.human_coach_enabled),
                "source": "none",
                "event": "no_command",
                "ppo_action": int(ppo_action),
                "selected_action": int(policy_action),
                "override_applied": False,
                "coach_match": False,
                "rejected": False,
                "rejected_reason": "",
                "command": None,
            }
        intervention_command: Any = None
        intervention_source = ""
        intervention_status = ""
        if coach_decision is not None and coach_decision.command is not None:
            intervention_command = coach_decision.command.to_dict()
            intervention_source = str(coach_decision.command.source or "human")
            intervention_status = (
                "rejected" if coach_decision.rejected else
                ("approved" if coach_decision.event == "coach_suggestion" else
                 ("pending" if coach_decision.event == "coach_pending" else "executed"))
            )
        elif stream_decision is not None and isinstance(stream_decision.selected_command, dict):
            intervention_command = dict(stream_decision.selected_command)
            intervention_source = "stream"
            intervention_status = "executed" if stream_decision.selected else "rejected"
        if intervention_command is not None:
            board = observation if isinstance(observation, dict) else {}
            self.intervention_logger.log(
                run_id=str(getattr(self.config, "run_id", "") or self.config.run_mode),
                episode_id=int(self._episode_index),
                step=int(self._global_step_count),
                mode="eval" if "eval" in str(self.config.run_mode).lower() else "train",
                model_action=int(ppo_action),
                human_command=intervention_command,
                command_source=intervention_source,
                status=intervention_status,
                board_state_summary={
                    "sun": board.get("sun"),
                    "wave": board.get("wave"),
                    "plant_count": len(board.get("plants", [])) if isinstance(board.get("plants"), list) else None,
                    "zombie_count": len(board.get("zombies", [])) if isinstance(board.get("zombies"), list) else None,
                },
                reward_after=float(reward),
                metadata={"selected_action": int(policy_action), "command_mode": self.config.human_coach_command_mode},
            )
        self._last_observation = observation
        self._step_count += 1
        self._global_step_count += 1
        self._episode_reward += float(reward)
        self._record_sun_diagnostics(observation)
        self._record_lane_diagnostics(bridge_action, observation, info)
        self._record_reward_breakdown(info)
        self._record_environment_safety(info)

        action_result = info.get("action_result", {})
        placement = action_result.get("placement") or {}
        if action_result.get("illegalAction"):
            self._episode_illegal += 1
        if placement.get("success"):
            self._episode_plants += 1
        if action_result.get("costPaid") or placement.get("costPaid"):
            sun_before = int(action_result.get("sunBefore") or placement.get("sunBefore") or 0)
            sun_after = int(action_result.get("sunAfter") or placement.get("sunAfter") or observation.get("sun", 0))
            spent = max(0, sun_before - sun_after)
            if spent == 0:
                spent = int(action_result.get("plantCost") or placement.get("plantCost") or 0)
            self._episode_sun_spent += max(0, spent)
        legal_actions = info.get("legal_actions") or observation.get("legalActions", [])
        if isinstance(legal_actions, list):
            self._episode_legal_action_total += len(legal_actions)

        terminal_reason = str(info.get("terminal_reason") or "")
        done_reason = str(info.get("done_reason") or "") or classify_done_reason(observation)
        if terminal_reason in ("level_complete_trophy", "reward_unlock"):
            if self.base._confirmed_post_win_ui(observation):
                done_reason = "post_win_pending"
            elif self.base._has_live_board_progress(observation):
                done_reason = "none"
        elif terminal_reason == "game_over_restart_screen":
            done_reason = "loss"
            print(f"[terminal] reason={terminal_reason} step={self._step_count}")
        if done_reason == "win" and not terminal_reason:
            terminal_reason = "win"
            info["terminal_reason"] = "win"
            info["done_reason"] = "win"
        if done_reason == "win":
            print(
                "[terminal] reason=win "
                f"step={self._step_count} "
                f"wave={observation.get('wave')}/{observation.get('maxWave')} "
                f"run_mode={self.config.run_mode}"
            )
        terminated = (
            terminal_reason == "game_over_restart_screen"
            or done_reason in ("win", "loss", "post_win_pending")
            or bool(done)
        )
        truncated = False
        if not terminated and self.config.max_steps > 0 and self._step_count >= self.config.max_steps:
            truncated = True
            done_reason = "timeout"
            terminal_reason = "timeout"
            timeout_event = {
                "event": "timeout_reset_requested",
                "timeout_type": "max_steps",
                "episode_step": self._step_count,
                "max_steps": self.config.max_steps,
                "allowed": True,
                "wave": observation.get("wave"),
                "maxWave": observation.get("maxWave"),
                "zombieCount": observation.get("zombieCount"),
                "plantCount": observation.get("plantCount"),
                "gameplayReady": observation.get("gameplayReady"),
                "screenState": observation.get("screenState"),
                "nextStep": observation.get("nextStep"),
                "terminalHint": observation.get("terminalHint"),
                "run_mode": self.config.run_mode,
            }
            info.setdefault("safety_events", []).append(timeout_event)
            info["timeout_reset_requested_count"] = 1
            info["done_reason"] = "timeout"
            info["terminal_reason"] = "timeout"
            info["truncated"] = True
            self._episode_timeout_reset_requested_count += 1
            self._last_episode_ended_by_timeout = True
            self._reset_requires_seed_flow_next = True
            print(
                "[reset] timeout_reset_requested "
                f"episode_step={self._step_count} max_steps={self.config.max_steps} "
                f"wave={observation.get('wave')}/{observation.get('maxWave')} "
                f"run_mode={self.config.run_mode}"
            )
            print("[reset] timeout episode truncated; next reset requires full seed flow")

        final_kills = int(observation.get("killCount", self._start_kills))
        final_mowers = int(observation.get("logicalMowerCount", self._start_mowers))
        avg_legal_actions = self._episode_legal_action_total / max(1, self._step_count)
        placement_context_total = (
            self._episode_plants_in_threatened_row_count
            + self._episode_plants_in_unthreatened_row_count
        )
        episode_summary = {
            "run_mode": self.config.run_mode,
            "target_level": int(self.config.target_level or 0),
            "episode": self._episode_index,
            "result": done_reason,
            "reward_total": self._episode_reward,
            "done_reason": done_reason,
            "episode_reward": self._episode_reward,
            "episode_length": self._step_count,
            "terminal_reason": terminal_reason,
            "final_wave": int(observation.get("wave", 0)),
            "max_wave": int(observation.get("maxWave", 0)),
            "zombies_killed": max(0, final_kills - self._start_kills),
            "plants_placed": self._episode_plants,
            "sunflowers_planted": sum(self._episode_sunflower_placements_by_row.values()),
            "peashooters_planted": sum(self._episode_peashooter_placements_by_row.values()),
            "wallnuts_planted": sum(self._episode_wallnut_placements_by_row.values()),
            "cherrybombs_planted": self._episode_cherrybomb_used_count,
            "sun_spent": self._episode_sun_spent,
            "sun_remaining": int(observation.get("sun", 0)),
            "mowers_lost": max(0, self._start_mowers - final_mowers),
            "mower_losses": max(0, self._start_mowers - final_mowers),
            "reset_success": self._last_reset_success,
            "reset_seconds": self._last_reset_seconds,
            "bridge_errors": 0,
            "illegal_actions": self._episode_illegal,
            "avg_legal_actions": avg_legal_actions,
            "legal_action_count_mean": avg_legal_actions,
            "plants_by_row": self._row_dict(self._episode_final_plants_by_row),
            "peashooters_by_row": self._row_dict(self._episode_final_peashooters_by_row),
            "sunflowers_by_row": self._row_dict(self._episode_final_sunflowers_by_row),
            "threat_steps_by_row": self._row_dict(self._episode_threat_steps_by_row),
            "undefended_threat_steps_by_row": self._row_dict(self._episode_undefended_threat_steps_by_row),
            "undefended_threat_age_avg_by_row": self._ratio_row_dict(
                self._episode_undefended_threat_age_sum_by_row,
                self._episode_undefended_threat_age_count_by_row,
            ),
            "undefended_threat_age_max_by_row": self._row_dict(self._episode_undefended_threat_age_max_by_row),
            "mower_losses_by_row": self._row_dict(self._episode_mower_losses_by_row),
            "wait_under_threat_count": self._episode_wait_under_threat_count,
            "close_zombie_undefended_count": self._episode_close_zombie_undefended_count,
            "illegal_reason_counts": self._plain_counter_dict(self._episode_illegal_reason_counts),
            "legal_peashooter_actions_by_row": self._row_dict(self._episode_legal_peashooter_actions_by_row),
            "peashooter_available_but_waited_by_row": self._row_dict(
                self._episode_peashooter_available_but_waited_by_row,
            ),
            "peashooter_available_but_planted_elsewhere_by_row": self._row_dict(
                self._episode_peashooter_available_but_planted_elsewhere_by_row,
            ),
            "sunflower_while_undefended_threat_by_row": self._row_dict(
                self._episode_sunflower_while_undefended_threat_by_row,
            ),
            "wait_actions": self._episode_wait_actions,
            "plant_actions": self._episode_plant_actions,
            "wait_action_percent": 100.0 * self._episode_wait_actions / max(1, self._step_count),
            "plant_action_percent": 100.0 * self._episode_plant_actions / max(1, self._step_count),
            "plant_actions_by_row": self._row_dict(self._episode_plant_actions_by_row),
            "peashooter_actions_by_row": self._row_dict(self._episode_peashooter_actions_by_row),
            "sunflower_actions_by_row": self._row_dict(self._episode_sunflower_actions_by_row),
            "plant_placements_by_row": self._row_dict(self._episode_plant_placements_by_row),
            "peashooter_placements_by_row": self._row_dict(self._episode_peashooter_placements_by_row),
            "sunflower_placements_by_row": self._row_dict(self._episode_sunflower_placements_by_row),
            "row_defense_opportunities_by_row": self._row_dict(self._episode_row_defense_opportunities_by_row),
            "row_defense_responses_by_row": self._row_dict(self._episode_row_defense_responses_by_row),
            "undefended_threat_ratio_by_row": self._ratio_row_dict(
                self._episode_undefended_threat_steps_by_row,
                self._episode_threat_steps_by_row,
            ),
            "row_defense_response_rate_by_row": self._ratio_row_dict(
                self._episode_row_defense_responses_by_row,
                self._episode_row_defense_opportunities_by_row,
            ),
            "row_defense_response_rate": self._ratio_total(
                self._episode_row_defense_responses_by_row,
                self._episode_row_defense_opportunities_by_row,
            ),
            "threatened_rows_with_zero_defender_steps_by_row": self._row_dict(
                self._episode_threatened_zero_defender_steps_by_row,
            ),
            "peashooters_per_threat_step_by_row": self._ratio_row_dict(
                self._episode_peashooter_placements_by_row,
                self._episode_threat_steps_by_row,
            ),
            "first_defense_step_by_row": self._row_dict(self._episode_first_defense_step_by_row),
            "plants_in_threatened_row_ratio": (
                self._episode_plants_in_threatened_row_count / placement_context_total
                if placement_context_total > 0
                else 0.0
            ),
            "plants_in_unthreatened_row_ratio": (
                self._episode_plants_in_unthreatened_row_count / placement_context_total
                if placement_context_total > 0
                else 0.0
            ),
            "overdefended_while_undefended_count": self._episode_overdefended_while_undefended_count,
            "overdefense_count": self._episode_overdefended_while_undefended_count,
            "least_defended_threatened_row_plant_count": self._episode_least_defended_threatened_row_plant_count,
            "rows_with_peashooter_count": self._episode_rows_with_peashooter_count,
            "all_rows_peashooter_covered_step": self._episode_all_rows_peashooter_covered_step,
            "first_peashooter_by_row_step": self._row_dict(self._episode_first_peashooter_by_row_step),
            "sunflower_count_when_first_full_coverage": self._episode_sunflower_count_when_first_full_coverage,
            "sunflower_overbuild_before_defense_count": self._episode_sunflower_overbuild_before_defense_count,
            "sunflower_overbuild_count": self._episode_sunflower_overbuild_before_defense_count,
            "peashooter_coverage_rate_by_step": (
                self._episode_peashooter_coverage_rate_sum / max(1, self._step_count)
            ),
            "legal_actions_by_seed_slot": self._row_dict(
                self._episode_legal_actions_by_seed_slot,
                rows=self.card_slot_count,
            ),
            "bridge_legal_actions_by_seed_slot": self._row_dict(
                self._episode_bridge_legal_actions_by_seed_slot,
                rows=self.card_slot_count,
            ),
            "python_mask_block_reason_counts": self._plain_counter_dict(self._episode_python_mask_block_reason_counts),
            "pre_step_mask_blocked_count": self._episode_pre_step_mask_blocked_count,
            "cooldown_illegal_exposed_by_mask_count": self._episode_cooldown_illegal_exposed_by_mask_count,
            "mask_bridge_disagreement_count": self._episode_mask_bridge_disagreement_count,
            "wait_while_actionable_threat_count": self._episode_wait_while_actionable_threat_count,
            "wait_while_peashooter_affordable_ready_count": self._episode_wait_while_peashooter_affordable_ready_count,
            "wait_while_wallnut_affordable_ready_count": self._episode_wait_while_wallnut_affordable_ready_count,
            "wait_while_cherrybomb_affordable_ready_count": self._episode_wait_while_cherrybomb_affordable_ready_count,
            "active_threat_rows_without_peashooter_count": self._episode_active_threat_rows_without_peashooter_count,
            "sunflower_greed_while_defense_missing_count": self._episode_sunflower_greed_while_defense_missing_count,
            "wallnut_placements_by_row": self._row_dict(self._episode_wallnut_placements_by_row),
            "wallnut_placements_by_col": self._row_dict(self._episode_wallnut_placements_by_col, rows=self.cols),
            "wallnut_blocks_active_threat_count": self._episode_wallnut_blocks_active_threat_count,
            "wallnut_low_value_placement_count": self._episode_wallnut_low_value_placement_count,
            "wallnut_threatened_lane_placements": self._episode_wallnut_threatened_lane_count,
            "wallnut_between_zombie_and_house_count": self._episode_wallnut_between_zombie_and_house_count,
            "wallnut_frontline_count": self._episode_wallnut_frontline_count,
            "wallnut_emergency_blocks": self._episode_wallnut_emergency_block_count,
            "wallnut_useless_placements": self._episode_wallnut_low_value_placement_count,
            "wallnut_damage_absorbed_total": self._episode_wallnut_damage_absorbed_total,
            "cherrybomb_used_count": self._episode_cherrybomb_used_count,
            "cherrybomb_kills_total": self._episode_cherrybomb_kills_total,
            "cherrybomb_avg_kills_per_use": (
                self._episode_cherrybomb_kills_total / max(1, self._episode_cherrybomb_used_count)
            ),
            "cherrybomb_zero_kill_count": self._episode_cherrybomb_zero_kill_count,
            "cherrybomb_zero_kill_uses": self._episode_cherrybomb_zero_kill_count,
            "cherrybomb_cluster_uses": self._episode_cherrybomb_cluster_use_count,
            "cherrybomb_emergency_uses": self._episode_cherrybomb_emergency_use_count,
            "cherrybomb_buckethead_kills": self._episode_cherrybomb_buckethead_kills,
            "cherrybomb_conehead_kills": self._episode_cherrybomb_conehead_kills,
            "cherrybomb_heavy_zombie_kills": (
                self._episode_cherrybomb_buckethead_kills + self._episode_cherrybomb_conehead_kills
            ),
            "cherrybomb_used_under_threat_count": self._episode_cherrybomb_used_under_threat_count,
            "cherrybomb_used_low_value_count": self._episode_cherrybomb_used_low_value_count,
            "mower_risk_steps_by_row": self._row_dict(self._episode_mower_risk_steps_by_row),
            "mower_saves_estimated_by_row": self._row_dict(self._episode_mower_saves_estimated_by_row),
            "close_zombie_with_no_defense_count": self._episode_close_zombie_with_no_defense_count,
            "undefended_threat_steps": sum(self._episode_undefended_threat_steps_by_row.values()),
            "high_danger_unanswered_steps": self._episode_high_danger_unanswered_steps,
            "mower_exposure_steps": self._episode_mower_exposure_steps,
            "max_row_danger": self._episode_max_row_danger,
            "avg_row_danger": self._episode_avg_row_danger_sum / max(1, self._step_count),
            "tactical_mask_enabled": self._episode_tactical_mask_enabled,
            "wallnut_actions_masked": self._episode_wallnut_actions_masked,
            "cherrybomb_actions_masked": self._episode_cherrybomb_actions_masked,
            "wallnut_actions_available": self._episode_wallnut_actions_available,
            "cherrybomb_actions_available": self._episode_cherrybomb_actions_available,
            "mask_all_but_wait_count": self._episode_mask_all_but_wait_count,
            "buckethead_count_by_row": self._row_dict(self._episode_buckethead_count_by_row),
            "conehead_count_by_row": self._row_dict(self._episode_conehead_count_by_row),
            "tough_zombie_count_by_row": self._row_dict(self._episode_tough_zombie_count_by_row),
            "tough_zombie_response_count": self._episode_tough_zombie_response_count,
            "fusion_policy": self.fusion_policy,
            "fusion_candidate_count_total": self.fusion_candidate_count_total,
            "fusion_candidate_count_avg": self.fusion_candidate_count_total / max(1, self._step_count),
            "fusion_attempted_count": self.fusion_attempted_count,
            "fusion_success_count": self.fusion_success_count,
            "fusion_failed_count": self.fusion_failed_count,
            "fusion_rejected_count": self.fusion_rejected_count,
            "fusion_rejected_reasons": self._plain_counter_dict(Counter(self.fusion_rejected_reasons)),
            "fusion_by_result_type": self._plain_counter_dict(Counter(self.fusion_by_result_type)),
            "fusion_by_source_type": self._plain_counter_dict(Counter(self.fusion_by_source_type)),
            "fusion_by_row": self._plain_counter_dict(Counter(self.fusion_by_row)),
            "fusion_under_threat_count": self.fusion_under_threat_count,
            "fusion_near_buckethead_count": self.fusion_near_buckethead_count,
            "fusion_near_conehead_count": self.fusion_near_conehead_count,
            "fusion_estimated_mower_save_count": self.fusion_estimated_mower_save_count,
            "fusion_kills_after_use_total": self.fusion_kills_after_use_total,
            "fusion_avg_kills_after_use": (
                self.fusion_kills_after_use_total / max(1, self.fusion_success_count)
            ),
            "fusion_bridge_error_count": self.fusion_bridge_error_count,
            "fusion_unsafe_state_block_count": self.fusion_unsafe_state_block_count,
            "human_coach": self._coach_live_status(),
            "env_corruption_count": self._episode_env_corruption_count,
            "mower_respawn_detected_count": self._episode_mower_respawn_detected_count,
            "cooldown_reset_detected_count": self._episode_cooldown_reset_detected_count,
            "board_refresh_detected_count": self._episode_board_refresh_detected_count,
            "false_reward_unlock_during_gameplay_count": self._episode_false_reward_unlock_during_gameplay_count,
            "false_cleanup_reward_ui_during_gameplay_count": self._episode_false_cleanup_reward_ui_during_gameplay_count,
            "post_win_veto_live_board_count": self._episode_post_win_veto_live_board_count,
            "blocked_cleanup_during_gameplay_count": self._episode_blocked_cleanup_during_gameplay_count,
            "suspicious_cleanup_reward_ui_count": self._episode_suspicious_cleanup_reward_ui_count,
            "reset_reward_ui_cleanup_count": self._episode_reset_reward_ui_cleanup_count,
            "reset_reward_ui_cleanup_blocked_count": self._episode_reset_reward_ui_cleanup_blocked_count,
            "reset_after_false_reward_signal_count": self._episode_reset_after_false_reward_signal_count,
            "timeout_reset_requested_count": self._episode_timeout_reset_requested_count,
            **{
                field_name: float(self._episode_reward_totals.get(field_name, 0.0))
                for field_name in REWARD_EPISODE_TOTAL_FIELDS
            },
            "win": done_reason == "win",
            "loss": done_reason == "loss",
            "timeout": done_reason == "timeout",
        }
        info.update(
            {
                "raw_observation": observation,
                "episode_summary_candidate": dict(episode_summary),
                "done_reason": done_reason,
                "terminal_reason": terminal_reason,
                "final_wave": episode_summary["final_wave"],
                "zombies_killed": episode_summary["zombies_killed"],
                "plants_placed": self._episode_plants,
                "illegal_actions": self._episode_illegal,
            }
        )
        if self.config.debug_performance:
            perf = dict(info.get("performance", {}))
            perf["action_mask_ms"] = self._last_action_mask_ms
            perf["legal_mask_ms"] = self._last_action_mask_ms
            info["performance"] = perf
            self._record_performance(perf)
        if terminated or truncated:
            next_reset_reason = done_reason
            allow_active_gameplay_reset_next = False
            terminal_hint = str(observation.get("terminalHint") or "")
            timeout_near_terminal_win = bool(
                done_reason == "timeout"
                and (
                    terminal_hint == "possible_win"
                    or self.base._post_win_signal_present(observation)
                    or bool(observation.get("done"))
                    or bool(observation.get("over"))
                )
            )
            if done_reason == "env_corruption":
                allow_active_gameplay_reset_next = terminal_reason != "bridge_timeout"
            elif done_reason in ("win", "post_win_pending") and self.config.run_mode in {RUN_MODE_FIXED_TRAIN, RUN_MODE_LEVEL3_SPECIALIST}:
                allow_active_gameplay_reset_next = False
                self._last_episode_ended_by_win = True
                self._reset_requires_seed_flow_next = True
            elif done_reason == "timeout":
                allow_active_gameplay_reset_next = False
                if timeout_near_terminal_win:
                    next_reset_reason = "post_win_pending"
            self._next_reset_reason = next_reset_reason
            self._allow_active_gameplay_reset_next = allow_active_gameplay_reset_next
            info.update(episode_summary)
            info["episode_summary"] = episode_summary

        return self._encode_observation(observation), float(reward), terminated, truncated, info

    def _legal_actions_from_mask(self, action_mask: Optional[np.ndarray]) -> Optional[List[int]]:
        if action_mask is None:
            return None
        actions: List[int] = []
        try:
            for index, allowed in enumerate(action_mask):
                if bool(allowed):
                    actions.append(int(index))
        except TypeError:
            return None
        return actions

    def _apply_stream_coach_reward(self, decision: Any, info: Dict[str, Any]) -> float:
        if (
            not bool(self.config.stream_coach_reward)
            or not bool(getattr(decision, "selected", False))
            or bool(getattr(decision, "pending", False))
        ):
            return 0.0
        components = {
            COACH_REWARD_MATCH_COMPONENT: 0.0,
            COACH_REWARD_LEGAL_EXECUTION_COMPONENT: float(getattr(self.config.reward, "coach_legal_execution_reward", 0.01)),
            COACH_REWARD_OVERRIDE_PENALTY_COMPONENT: 0.0,
        }
        if bool(getattr(decision, "coach_match", False)):
            components[COACH_REWARD_MATCH_COMPONENT] = float(getattr(self.config.reward, "coach_match_reward", 0.02))
        if bool(getattr(decision, "override_applied", False)):
            components[COACH_REWARD_OVERRIDE_PENALTY_COMPONENT] = float(
                getattr(self.config.reward, "coach_override_penalty", -0.01)
            )
        delta = sum(float(value) for value in components.values())
        if delta == 0.0:
            return 0.0
        breakdown = info.get("reward_breakdown")
        if not isinstance(breakdown, dict):
            breakdown = {}
        for key, value in components.items():
            breakdown[key] = float(breakdown.get(key) or 0.0) + float(value)
        breakdown["reward_total"] = float(breakdown.get("reward_total") or 0.0) + float(delta)
        info["reward_breakdown"] = breakdown
        return float(delta)

    def _coach_live_status(self) -> Dict[str, Any]:
        human_fields = human_coach_live_status_from_hook(
            self.human_coach_hook,
            enabled=bool(self.config.human_coach_enabled),
            platform=self.config.human_coach_platform or "mock",
        )
        payload = dict(human_fields)
        if self.stream_coach_controller is not None:
            stream_fields = self.stream_coach_controller.diagnostics_fields()
            payload.update(stream_fields)
        elif "stream_coach_enabled" not in payload:
            payload.update(
                human_coach_live_status_defaults(
                    enabled=False,
                    platform=str(self.config.stream_coach_platform or "mock"),
                )
            )
        if self.stream_coach_controller is None and not bool(self.config.stream_coach_enabled):
            payload["stream_coach_enabled"] = False
            payload["stream_coach_mode"] = "off"
            payload["stream_coach_alive"] = False
            payload["stream_coach_alive_status"] = "off"
        else:
            payload.setdefault("stream_coach_mode", str(self.config.stream_coach_mode or self.config.stream_coach_platform or "mock"))
            payload.setdefault("stream_coach_alive", bool(self.config.stream_coach_enabled))
            payload.setdefault("stream_coach_alive_status", "alive" if bool(self.config.stream_coach_enabled) else "off")
        payload.setdefault("stream_coach_command_path", str(self.config.stream_coach_command_path or ""))
        payload.setdefault("mock_stream_script", str(self.config.stream_coach_mock_script or ""))
        payload.setdefault("stream_coach_dry_run", bool(self.config.stream_coach_dry_run))
        payload.setdefault("stream_coach_apply_enabled", bool(self.config.stream_coach_apply_enabled) and not bool(self.config.stream_coach_dry_run))
        payload.setdefault("stream_coach_messages_seen", 0)
        payload.setdefault("stream_coach_commands_parsed", 0)
        payload.setdefault("stream_coach_commands_accepted", 0)
        payload.setdefault("stream_coach_commands_rejected", 0)
        payload.setdefault("stream_messages_seen", int(payload.get("stream_coach_messages_seen", 0) or 0))
        payload.setdefault("stream_commands_parsed", int(payload.get("stream_coach_commands_parsed", 0) or 0))
        payload.setdefault("stream_commands_accepted", int(payload.get("stream_coach_commands_accepted", 0) or 0))
        payload.setdefault("stream_commands_rejected", int(payload.get("stream_coach_commands_rejected", 0) or 0))
        payload.setdefault("stream_coach_validated_count", 0)
        payload.setdefault("stream_coach_applied_count", 0)
        payload.setdefault("stream_coach_dry_run_count", 0)
        payload.setdefault("last_stream_user", str(payload.get("stream_coach_last_user") or ""))
        payload.setdefault("stream_coach_last_message", "")
        payload.setdefault("stream_coach_last_parsed_command", None)
        payload.setdefault("stream_coach_last_command_status", "off" if not bool(self.config.stream_coach_enabled) else "idle")
        payload.setdefault("stream_coach_last_reject_reason", "")
        payload.setdefault("stream_coach_last_validated_command", "")
        payload.setdefault("stream_coach_last_applied_command", "")
        payload.setdefault("last_stream_message", str(payload.get("stream_coach_last_message") or ""))
        payload.setdefault("last_stream_parsed_command", payload.get("stream_coach_last_parsed_command"))
        payload.setdefault("last_stream_command_status", str(payload.get("stream_coach_last_command_status") or ""))
        payload.setdefault("last_stream_reject_reason", str(payload.get("stream_coach_last_reject_reason") or ""))
        payload.setdefault("last_validated_coach_command", str(payload.get("stream_coach_last_validated_command") or ""))
        payload.setdefault("last_applied_coach_command", str(payload.get("stream_coach_last_applied_command") or ""))
        payload.setdefault("pending_stream_commands", 0)
        payload.setdefault("stream_coach_startup_stale_cleared", False)
        payload.setdefault("stream_coach_stale_messages_cleared", 0)
        payload.setdefault("stream_coach_clear_count", 0)
        payload.setdefault("stream_coach_last_clear_reason", "")
        payload.setdefault(
            "fusion_bridge_enabled",
            bool(self.config.human_coach_fusion_enabled or self.config.stream_coach_fusion_enabled),
        )
        if payload.get("fusion_bridge_available") is None:
            payload["fusion_bridge_available"] = bool(self._fusion_probe is not None)
        if "fusion_last_result" not in payload:
            payload["fusion_last_result"] = ""
        if "fusion_last_command" not in payload:
            payload["fusion_last_command"] = None
        payload.setdefault("fusion_last_execution_mode", "")
        payload.setdefault("fusion_last_bridge_method_used", "")
        payload.setdefault("fusion_last_bridge_result_reason", "")
        payload.setdefault("fusion_last_duplicate_stack_detected", False)
        payload.setdefault("fusion_last_source_tile_occupied_before", False)
        payload.setdefault("fusion_last_plant_count_on_tile_before", 0)
        payload.setdefault("fusion_last_plant_count_on_tile_after", 0)
        payload.setdefault("fusion_last_source_plant_before", None)
        payload.setdefault("fusion_last_resulting_plant_after", None)
        payload.setdefault("fusion_last_predicted_result_resolution_source", "")
        payload.setdefault("fusion_last_mix_lookup_found", False)
        payload.setdefault("fusion_last_mix_lookup_key", "")
        payload.setdefault("fusion_last_pre_source_type", -1)
        payload.setdefault("fusion_last_pre_source_name", "")
        payload.setdefault("fusion_last_ingredient_type", -1)
        payload.setdefault("fusion_last_ingredient_name", "")
        payload.setdefault("fusion_last_post_result_type", -1)
        payload.setdefault("fusion_last_post_result_name", "")
        payload.setdefault("fusion_last_no_effect_reason", "")
        payload.setdefault("last_fusion_scope", "")
        payload.setdefault("last_fusion_changed_tile_count", 0)
        payload.setdefault("last_fusion_non_source_tiles_changed", False)
        payload.setdefault("last_fusion_global_side_effect", False)
        payload.setdefault("pending_coach_command", None)
        payload.setdefault("selected_bridge_command", None)
        payload.setdefault("last_executed_coach_command_id", None)
        payload.setdefault("coach_command_queue_cleared_on_reset", True)
        payload.setdefault("startup_command_blocked", False)
        payload.setdefault("stream_fusion_last_execution_mode", "")
        payload.setdefault("stream_fusion_last_bridge_method_used", "")
        payload.setdefault("stream_fusion_last_bridge_result_reason", "")
        payload.setdefault("stream_fusion_last_duplicate_stack_detected", False)
        payload.setdefault("stream_fusion_last_source_tile_occupied_before", False)
        payload.setdefault("stream_fusion_last_plant_count_on_tile_before", 0)
        payload.setdefault("stream_fusion_last_plant_count_on_tile_after", 0)
        payload.setdefault("stream_fusion_last_source_plant_before", None)
        payload.setdefault("stream_fusion_last_resulting_plant_after", None)
        payload.setdefault("stream_fusion_last_predicted_result_resolution_source", "")
        payload.setdefault("stream_fusion_last_mix_lookup_found", False)
        payload.setdefault("stream_fusion_last_mix_lookup_key", "")
        payload.setdefault("stream_fusion_last_pre_source_type", -1)
        payload.setdefault("stream_fusion_last_pre_source_name", "")
        payload.setdefault("stream_fusion_last_ingredient_type", -1)
        payload.setdefault("stream_fusion_last_ingredient_name", "")
        payload.setdefault("stream_fusion_last_post_result_type", -1)
        payload.setdefault("stream_fusion_last_post_result_name", "")
        payload.setdefault("stream_fusion_last_no_effect_reason", "")
        return payload

    def action_masks(self) -> np.ndarray:
        started = time.perf_counter()
        observation = self._last_observation or self.base.observe()
        mask = np.zeros(self.action_count, dtype=bool)
        raw_mask = self.base.action_mask(observation)
        if self.action_spec.dynamic_seed_slots:
            for legacy_action, allowed in enumerate(raw_mask):
                if not bool(allowed):
                    continue
                policy_action = legacy_action_to_policy_action(
                    legacy_action,
                    mode=self.action_spec.mode,
                )
                if 0 <= policy_action < self.action_count:
                    mask[policy_action] = True
            if self.action_spec.mode == ACTION_SPACE_ADVENTURE_14_IDENTITY:
                mask[0] = True
            else:
                mask[DYNAMIC_WAIT_ACTION] = True
        else:
            for action, allowed in enumerate(raw_mask[: self.action_count]):
                if bool(allowed):
                    mask[action] = True
            mask[0] = True
        self._last_action_mask_ms = round((time.perf_counter() - started) * 1000.0, 3)
        return mask

    def _policy_action_to_bridge_action(self, action: int) -> int:
        if not self.action_spec.dynamic_seed_slots:
            return int(action)
        return policy_action_to_legacy_action(
            int(action),
            mode=self.action_spec.mode,
            rows=self.rows,
            cols=self.cols,
        )

    def decode_policy_action(self, action: int, observation: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
        return decode_policy_action(
            int(action),
            mode=self.action_spec.mode,
            observation=observation if observation is not None else self._last_observation,
            plant_types=list(self.config.plant_types),
            max_seed_slots=self.action_spec.max_seed_slots,
            rows=self.rows,
            cols=self.cols,
        )

    def _record_performance(self, perf: Dict[str, Any]) -> None:
        self._perf_samples += 1
        for key in (
            "python_step_ms",
            "bridge_step_ms",
            "bridge_observe_ms",
            "observe_ms",
            "screen_check_ms",
            "ui_scan_ms",
            "seed_probe_ms",
            "legal_actions_ms",
            "action_mask_ms",
            "legal_mask_ms",
        ):
            value = perf.get(key)
            if isinstance(value, (int, float)):
                self._perf_totals[key] = self._perf_totals.get(key, 0.0) + float(value)
        if self._perf_samples % self._perf_report_interval == 0:
            parts = []
            for key in (
                "python_step_ms",
                "bridge_observe_ms",
                "screen_check_ms",
                "ui_scan_ms",
                "legal_actions_ms",
                "action_mask_ms",
                "legal_mask_ms",
            ):
                if key in self._perf_totals:
                    parts.append(f"{key}={self._perf_totals[key] / self._perf_samples:.2f}")
            mode = perf.get("restart_detection_mode") or "unknown"
            print(f"[perf] steps={self._perf_samples} restartDetectionMode={mode} " + " ".join(parts))

    def _reset_sun_diagnostic_baseline(self, observation: Dict[str, Any]) -> None:
        self._sun_diag_last_sun = self._safe_int(observation.get("sun"))
        self._sun_diag_last_active_count = self._active_sun_count(observation)

    def _record_sun_diagnostics(self, observation: Dict[str, Any]) -> None:
        current_sun = self._safe_int(observation.get("sun"))
        if current_sun is not None:
            if self._sun_diag_last_sun is not None and current_sun > self._sun_diag_last_sun:
                self._sun_diag_gained_window += current_sun - self._sun_diag_last_sun
            self._sun_diag_last_sun = current_sun

        active_count = self._active_sun_count(observation)
        if active_count is not None:
            if self._sun_diag_last_active_count is not None and active_count > self._sun_diag_last_active_count:
                self._sun_diag_spawn_window += active_count - self._sun_diag_last_active_count
            self._sun_diag_last_active_count = active_count

        if not (self.config.debug_performance or self.config.debug_sun):
            return
        if self._global_step_count <= 0 or self._global_step_count % self._sun_diag_report_interval != 0:
            return

        payload = {
            "step": self._global_step_count,
            "episode": self._episode_index,
            "currentTimeScale": observation.get("unityTimeScale"),
            "requestedGameSpeed": observation.get("requestedGameSpeed"),
            "gameSpeedMode": observation.get("gameSpeedMode"),
            "effectiveGameSpeed": observation.get("effectiveGameSpeed"),
            "validSpeedModeApplyCount": observation.get("validSpeedModeApplyCount"),
            "speedApplyCount": observation.get("speedApplyCount"),
            "sunSpawnCompensationApplyCount": observation.get("sunSpawnCompensationApplyCount"),
            "activeBoardCount": observation.get("activeBoardCount"),
            "activeSkySunSpawnerCount": observation.get("activeSkySunSpawnerCount"),
            "activeSunObjectCount": observation.get("activeSunObjectCount"),
            "sunSpawnCountWindow": self._sun_diag_spawn_window,
            "sunGainedWindow": self._sun_diag_gained_window,
            "skySunSpawnInterval": observation.get("skySunSpawnInterval"),
            "skySunSpawnTimer": observation.get("skySunSpawnTimer"),
            "boardInstanceId": observation.get("boardInstanceId"),
            "resetCount": observation.get("resetCount"),
            "letsRockClickCount": observation.get("letsRockClickCount"),
            "bridgeUpdateLoopCount": observation.get("bridgeUpdateLoopCount"),
            "activeCoroutineCount": observation.get("activeCoroutineCount"),
        }
        print("[sun-drift] " + json.dumps(payload, separators=(",", ":")))
        self._sun_diag_spawn_window = 0
        self._sun_diag_gained_window = 0

    @staticmethod
    def _active_sun_count(observation: Dict[str, Any]) -> Optional[int]:
        for key in ("activeSunObjectCount", "activeFallingSunCount"):
            value = PvZMaskedPPOEnv._safe_int(observation.get(key))
            if value is not None:
                return value
        return None

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        if isinstance(value, bool) or value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _reset_lane_episode_diagnostics(self, observation: Dict[str, Any]) -> None:
        self._episode_final_plants_by_row: Counter[int] = self._plant_counts_by_row(observation)
        self._episode_final_peashooters_by_row: Counter[int] = self._plant_counts_by_row(observation, plant_type=0)
        self._episode_final_sunflowers_by_row: Counter[int] = self._plant_counts_by_row(observation, plant_type=1)
        self._episode_threat_steps_by_row: Counter[int] = Counter()
        self._episode_undefended_threat_steps_by_row: Counter[int] = Counter()
        self._episode_undefended_threat_age_sum_by_row: Counter[int] = Counter()
        self._episode_undefended_threat_age_count_by_row: Counter[int] = Counter()
        self._episode_undefended_threat_age_max_by_row: Counter[int] = Counter()
        self._episode_mower_losses_by_row: Counter[int] = Counter()
        self._episode_wait_under_threat_count = 0
        self._episode_close_zombie_undefended_count = 0
        self._episode_illegal_reason_counts: Counter[str] = Counter()
        self._episode_legal_peashooter_actions_by_row: Counter[int] = Counter()
        self._episode_peashooter_available_but_waited_by_row: Counter[int] = Counter()
        self._episode_peashooter_available_but_planted_elsewhere_by_row: Counter[int] = Counter()
        self._episode_sunflower_while_undefended_threat_by_row: Counter[int] = Counter()
        self._episode_wait_actions = 0
        self._episode_plant_actions = 0
        self._episode_plant_actions_by_row: Counter[int] = Counter()
        self._episode_peashooter_actions_by_row: Counter[int] = Counter()
        self._episode_sunflower_actions_by_row: Counter[int] = Counter()
        self._episode_plant_placements_by_row: Counter[int] = Counter()
        self._episode_peashooter_placements_by_row: Counter[int] = Counter()
        self._episode_sunflower_placements_by_row: Counter[int] = Counter()
        self._episode_row_defense_opportunities_by_row: Counter[int] = Counter()
        self._episode_row_defense_responses_by_row: Counter[int] = Counter()
        self._episode_threatened_zero_defender_steps_by_row: Counter[int] = Counter()
        self._episode_first_defense_step_by_row: Counter[int] = Counter()
        self._episode_first_peashooter_by_row_step: Counter[int] = Counter()
        self._episode_plants_in_threatened_row_count = 0
        self._episode_plants_in_unthreatened_row_count = 0
        self._episode_overdefended_while_undefended_count = 0
        self._episode_least_defended_threatened_row_plant_count = 0
        self._episode_rows_with_peashooter_count = sum(
            1 for count in self._episode_final_peashooters_by_row.values()
            if count > 0
        )
        self._episode_all_rows_peashooter_covered_step = 0
        self._episode_sunflower_count_when_first_full_coverage = -1
        self._episode_sunflower_overbuild_before_defense_count = 0
        self._episode_peashooter_coverage_rate_sum = 0.0
        self._episode_legal_actions_by_seed_slot: Counter[int] = Counter()
        self._episode_bridge_legal_actions_by_seed_slot: Counter[int] = Counter()
        self._episode_python_mask_block_reason_counts: Counter[str] = Counter()
        self._episode_pre_step_mask_blocked_count = 0
        self._episode_cooldown_illegal_exposed_by_mask_count = 0
        self._episode_mask_bridge_disagreement_count = 0
        self._episode_wait_while_actionable_threat_count = 0
        self._episode_wait_while_peashooter_affordable_ready_count = 0
        self._episode_wait_while_wallnut_affordable_ready_count = 0
        self._episode_wait_while_cherrybomb_affordable_ready_count = 0
        self._episode_active_threat_rows_without_peashooter_count = 0
        self._episode_sunflower_greed_while_defense_missing_count = 0
        self._episode_wallnut_placements_by_row: Counter[int] = Counter()
        self._episode_wallnut_placements_by_col: Counter[int] = Counter()
        self._episode_wallnut_blocks_active_threat_count = 0
        self._episode_wallnut_low_value_placement_count = 0
        self._episode_wallnut_threatened_lane_count = 0
        self._episode_wallnut_between_zombie_and_house_count = 0
        self._episode_wallnut_frontline_count = 0
        self._episode_wallnut_emergency_block_count = 0
        self._episode_wallnut_damage_absorbed_total = 0.0
        self._episode_cherrybomb_used_count = 0
        self._episode_cherrybomb_kills_total = 0
        self._episode_cherrybomb_zero_kill_count = 0
        self._episode_cherrybomb_cluster_use_count = 0
        self._episode_cherrybomb_emergency_use_count = 0
        self._episode_cherrybomb_buckethead_kills = 0
        self._episode_cherrybomb_conehead_kills = 0
        self._episode_cherrybomb_used_under_threat_count = 0
        self._episode_cherrybomb_used_low_value_count = 0
        self._episode_mower_risk_steps_by_row: Counter[int] = Counter()
        self._episode_mower_saves_estimated_by_row: Counter[int] = Counter()
        self._episode_close_zombie_with_no_defense_count = 0
        self._episode_buckethead_count_by_row: Counter[int] = Counter()
        self._episode_conehead_count_by_row: Counter[int] = Counter()
        self._episode_tough_zombie_count_by_row: Counter[int] = Counter()
        self._episode_tough_zombie_response_count = 0
        self._episode_high_danger_unanswered_steps = 0
        self._episode_mower_exposure_steps = 0
        self._episode_max_row_danger = 0.0
        self._episode_avg_row_danger_sum = 0.0
        self._episode_tactical_mask_enabled = False
        self._episode_wallnut_actions_masked = 0
        self._episode_cherrybomb_actions_masked = 0
        self._episode_wallnut_actions_available = 0
        self._episode_cherrybomb_actions_available = 0
        self._episode_mask_all_but_wait_count = 0
        self.fusion_policy = self.config.fusion_policy
        self.fusion_candidate_count_total = 0
        self.fusion_attempted_count = 0
        self.fusion_success_count = 0
        self.fusion_failed_count = 0
        self.fusion_rejected_count = 0
        self.fusion_rejected_reasons: Dict[str, int] = {}
        self.fusion_by_result_type: Dict[str, int] = {}
        self.fusion_by_source_type: Dict[str, int] = {}
        self.fusion_by_row: Dict[str, int] = {}
        self.fusion_under_threat_count = 0
        self.fusion_near_buckethead_count = 0
        self.fusion_near_conehead_count = 0
        self.fusion_estimated_mower_save_count = 0
        self.fusion_kills_after_use_total = 0
        self.fusion_bridge_error_count = 0
        self.fusion_unsafe_state_block_count = 0
        self._episode_env_corruption_count = 0
        self._episode_mower_respawn_detected_count = 0
        self._episode_cooldown_reset_detected_count = 0
        self._episode_board_refresh_detected_count = 0
        self._episode_blocked_cleanup_during_gameplay_count = 0
        self._episode_suspicious_cleanup_reward_ui_count = 0
        self._episode_reset_reward_ui_cleanup_count = 0
        self._episode_reset_reward_ui_cleanup_blocked_count = 0
        self._episode_false_reward_unlock_during_gameplay_count = 0
        self._episode_false_cleanup_reward_ui_during_gameplay_count = 0
        self._episode_post_win_veto_live_board_count = 0
        self._episode_reset_after_false_reward_signal_count = 0
        self._episode_timeout_reset_requested_count = 0
        self._episode_reward_totals: Dict[str, float] = {field: 0.0 for field in REWARD_EPISODE_TOTAL_FIELDS}

    def _record_lane_diagnostics(self, action: int, observation: Dict[str, Any], info: Dict[str, Any]) -> None:
        diag = info.get("lane_diagnostics") if isinstance(info, dict) else {}
        if not isinstance(diag, dict):
            diag = {}

        self._replace_row_counts(self._episode_final_plants_by_row, diag.get("plants_by_row"), self._plant_counts_by_row(observation))
        self._replace_row_counts(self._episode_final_peashooters_by_row, diag.get("peashooters_by_row"), self._plant_counts_by_row(observation, plant_type=0))
        self._replace_row_counts(self._episode_final_sunflowers_by_row, diag.get("sunflowers_by_row"), self._plant_counts_by_row(observation, plant_type=1))
        self._add_row_counts(self._episode_threat_steps_by_row, diag.get("threat_steps_by_row"))
        self._add_row_counts(self._episode_undefended_threat_steps_by_row, diag.get("undefended_threat_steps_by_row"))
        self._add_row_counts(self._episode_undefended_threat_age_sum_by_row, diag.get("undefended_threat_age_sum_by_row"))
        self._add_row_counts(self._episode_undefended_threat_age_count_by_row, diag.get("undefended_threat_age_count_by_row"))
        self._update_row_max(self._episode_undefended_threat_age_max_by_row, diag.get("undefended_threat_age_max_by_row"))
        self._add_row_counts(self._episode_mower_losses_by_row, diag.get("mower_losses_by_row"))
        self._add_row_counts(self._episode_legal_peashooter_actions_by_row, diag.get("legal_peashooter_actions_by_row"))
        self._add_row_counts(
            self._episode_peashooter_available_but_waited_by_row,
            diag.get("peashooter_available_but_waited_by_row"),
        )
        self._add_row_counts(
            self._episode_peashooter_available_but_planted_elsewhere_by_row,
            diag.get("peashooter_available_but_planted_elsewhere_by_row"),
        )
        self._add_row_counts(
            self._episode_sunflower_while_undefended_threat_by_row,
            diag.get("sunflower_while_undefended_threat_by_row"),
        )
        self._add_row_counts(self._episode_row_defense_opportunities_by_row, diag.get("row_defense_opportunities_by_row"))
        self._add_row_counts(self._episode_row_defense_responses_by_row, diag.get("row_defense_responses_by_row"))
        self._add_row_counts(
            self._episode_threatened_zero_defender_steps_by_row,
            diag.get("threatened_rows_with_zero_defender_steps_by_row"),
        )
        self._add_row_counts(self._episode_legal_actions_by_seed_slot, diag.get("legal_actions_by_seed_slot"))
        self._add_row_counts(self._episode_bridge_legal_actions_by_seed_slot, diag.get("bridge_legal_actions_by_seed_slot"))
        self._add_reason_counts(self._episode_python_mask_block_reason_counts, diag.get("python_mask_block_reason_counts"))
        self._add_row_counts(self._episode_wallnut_placements_by_row, diag.get("wallnut_placements_by_row"))
        self._add_row_counts(self._episode_wallnut_placements_by_col, diag.get("wallnut_placements_by_col"))
        self._add_row_counts(self._episode_mower_risk_steps_by_row, diag.get("mower_risk_steps_by_row"))
        self._add_row_counts(self._episode_mower_saves_estimated_by_row, diag.get("mower_saves_estimated_by_row"))
        self._add_row_counts(self._episode_buckethead_count_by_row, diag.get("buckethead_count_by_row"))
        self._add_row_counts(self._episode_conehead_count_by_row, diag.get("conehead_count_by_row"))
        self._add_row_counts(self._episode_tough_zombie_count_by_row, diag.get("tough_zombie_count_by_row"))
        if bool(diag.get("pre_step_mask_blocked_action")):
            self._episode_pre_step_mask_blocked_count += 1
        if bool(diag.get("cooldown_illegal_exposed_by_mask")):
            self._episode_cooldown_illegal_exposed_by_mask_count += 1
        if bool(diag.get("mask_bridge_disagreement")):
            self._episode_mask_bridge_disagreement_count += 1

        if bool(diag.get("wait_under_threat")):
            self._episode_wait_under_threat_count += 1
        close_count = self._safe_int_value(diag.get("close_zombie_undefended_count"), default=0)
        self._episode_close_zombie_undefended_count += max(0, close_count)
        if bool(diag.get("plant_in_threatened_row")):
            self._episode_plants_in_threatened_row_count += 1
        if bool(diag.get("plant_in_unthreatened_row")):
            self._episode_plants_in_unthreatened_row_count += 1
        if bool(diag.get("overdefended_while_undefended")):
            self._episode_overdefended_while_undefended_count += 1
        if bool(diag.get("least_defended_threatened_row_plant")):
            self._episode_least_defended_threatened_row_plant_count += 1
        if bool(diag.get("sunflower_overbuild_before_defense")):
            self._episode_sunflower_overbuild_before_defense_count += 1
        if bool(diag.get("wait_while_actionable_threat")):
            self._episode_wait_while_actionable_threat_count += 1
        if bool(diag.get("wait_while_peashooter_affordable_ready")):
            self._episode_wait_while_peashooter_affordable_ready_count += 1
        if bool(diag.get("wait_while_wallnut_affordable_ready")):
            self._episode_wait_while_wallnut_affordable_ready_count += 1
        if bool(diag.get("wait_while_cherrybomb_affordable_ready")):
            self._episode_wait_while_cherrybomb_affordable_ready_count += 1
        self._episode_active_threat_rows_without_peashooter_count += self._safe_int_value(
            diag.get("active_threat_rows_without_peashooter_count"),
            default=0,
        )
        if bool(diag.get("sunflower_greed_while_defense_missing")):
            self._episode_sunflower_greed_while_defense_missing_count += 1
        if bool(diag.get("wallnut_blocks_active_threat")):
            self._episode_wallnut_blocks_active_threat_count += 1
        if bool(diag.get("wallnut_low_value_placement")):
            self._episode_wallnut_low_value_placement_count += 1
        if bool(diag.get("wallnut_threatened_lane")):
            self._episode_wallnut_threatened_lane_count += 1
        if bool(diag.get("wallnut_between_zombie_and_house")):
            self._episode_wallnut_between_zombie_and_house_count += 1
        if bool(diag.get("wallnut_frontline")):
            self._episode_wallnut_frontline_count += 1
        if bool(diag.get("wallnut_emergency_block")):
            self._episode_wallnut_emergency_block_count += 1
        if bool(diag.get("cherrybomb_used")):
            self._episode_cherrybomb_used_count += 1
        self._episode_cherrybomb_kills_total += self._safe_int_value(diag.get("cherrybomb_delayed_kills"), default=0)
        self._episode_cherrybomb_zero_kill_count += self._safe_int_value(diag.get("cherrybomb_delayed_zero_kill"), default=0)
        if bool(diag.get("cherrybomb_cluster_use")):
            self._episode_cherrybomb_cluster_use_count += 1
        if bool(diag.get("cherrybomb_emergency_use")):
            self._episode_cherrybomb_emergency_use_count += 1
        self._episode_cherrybomb_buckethead_kills += self._safe_int_value(diag.get("cherrybomb_buckethead_kill_credit"), default=0)
        self._episode_cherrybomb_conehead_kills += self._safe_int_value(diag.get("cherrybomb_conehead_kill_credit"), default=0)
        if bool(diag.get("cherrybomb_used_under_threat")):
            self._episode_cherrybomb_used_under_threat_count += 1
        if bool(diag.get("cherrybomb_used_low_value")):
            self._episode_cherrybomb_used_low_value_count += 1
        self._episode_close_zombie_with_no_defense_count += self._safe_int_value(
            diag.get("close_zombie_with_no_defense_count"),
            default=0,
        )
        if bool(diag.get("tough_zombie_response")):
            self._episode_tough_zombie_response_count += 1
        self._episode_high_danger_unanswered_steps += self._safe_int_value(
            diag.get("high_danger_unanswered_steps"),
            default=0,
        )
        self._episode_mower_exposure_steps += self._safe_int_value(
            diag.get("mower_exposure_steps"),
            default=0,
        )
        try:
            self._episode_max_row_danger = max(self._episode_max_row_danger, float(diag.get("max_row_danger") or 0.0))
            self._episode_avg_row_danger_sum += float(diag.get("avg_row_danger") or 0.0)
        except (TypeError, ValueError):
            pass
        if bool(diag.get("tactical_mask_enabled")):
            self._episode_tactical_mask_enabled = True
        self._episode_wallnut_actions_masked += self._safe_int_value(diag.get("wallnut_actions_masked"), default=0)
        self._episode_cherrybomb_actions_masked += self._safe_int_value(diag.get("cherrybomb_actions_masked"), default=0)
        self._episode_wallnut_actions_available += self._safe_int_value(diag.get("wallnut_actions_available"), default=0)
        self._episode_cherrybomb_actions_available += self._safe_int_value(diag.get("cherrybomb_actions_available"), default=0)
        self._episode_mask_all_but_wait_count += self._safe_int_value(diag.get("mask_all_but_wait_count"), default=0)
        fusion_diag = info.get("fusion_diagnostics") if isinstance(info, dict) else {}
        if isinstance(fusion_diag, dict):
            merge_episode_fusion_stats(self, fusion_diag)
        self._episode_rows_with_peashooter_count = self._safe_int_value(
            diag.get("rows_with_peashooter_count"),
            default=self._episode_rows_with_peashooter_count,
        )
        try:
            self._episode_peashooter_coverage_rate_sum += float(diag.get("peashooter_coverage_rate") or 0.0)
        except (TypeError, ValueError):
            pass
        first_peashooter_row = self._safe_int_value(diag.get("first_peashooter_row"), default=-1)
        if first_peashooter_row >= 0 and self._episode_first_peashooter_by_row_step.get(first_peashooter_row, 0) <= 0:
            self._episode_first_peashooter_by_row_step[first_peashooter_row] = self._step_count
        first_defense_row = self._safe_int_value(diag.get("first_defense_row"), default=-1)
        if first_defense_row >= 0 and self._episode_first_defense_step_by_row.get(first_defense_row, 0) <= 0:
            self._episode_first_defense_step_by_row[first_defense_row] = self._step_count
        if bool(diag.get("all_rows_peashooter_covered")) and self._episode_all_rows_peashooter_covered_step <= 0:
            self._episode_all_rows_peashooter_covered_step = self._step_count
            self._episode_sunflower_count_when_first_full_coverage = self._safe_int_value(
                diag.get("sunflower_count_when_first_full_coverage"),
                default=-1,
            )

        action_result = info.get("action_result", {}) if isinstance(info, dict) else {}
        executed_action = self._safe_int_value(action_result.get("executedAction"), default=int(action)) if isinstance(action_result, dict) else int(action)
        placement = action_result.get("placement") if isinstance(action_result, dict) and isinstance(action_result.get("placement"), dict) else {}
        illegal_reason = str(diag.get("illegal_reason") or (action_result.get("illegalReason") if isinstance(action_result, dict) else "") or "")
        if illegal_reason:
            self._episode_illegal_reason_counts[illegal_reason] += 1

        row = self._safe_int_value(diag.get("action_row"), default=self._safe_int_value(placement.get("row"), default=-1))
        plant_type = self._safe_int_value(diag.get("action_plant_type"), default=self._safe_int_value(placement.get("plantType"), default=-1))
        plant_placed = bool(diag.get("plant_placed") or placement.get("success") or placement.get("plantPlaced"))
        action_kind = str(diag.get("action_kind") or "")

        if executed_action == 0 and action_kind != "fusion":
            self._episode_wait_actions += 1
        else:
            self._episode_plant_actions += 1
            self._bump_row(self._episode_plant_actions_by_row, row)
            if plant_type == 0:
                self._bump_row(self._episode_peashooter_actions_by_row, row)
            elif plant_type == 1:
                self._bump_row(self._episode_sunflower_actions_by_row, row)

        if plant_placed:
            self._bump_row(self._episode_plant_placements_by_row, row)
            if plant_type == 0:
                self._bump_row(self._episode_peashooter_placements_by_row, row)
            elif plant_type == 1:
                self._bump_row(self._episode_sunflower_placements_by_row, row)

    def _record_reward_breakdown(self, info: Dict[str, Any]) -> None:
        breakdown = info.get("reward_breakdown") if isinstance(info, dict) else {}
        if not isinstance(breakdown, dict):
            return
        for component in REWARD_COMPONENT_FIELDS:
            field_name = f"{component}_total"
            try:
                value = float(breakdown.get(component) or 0.0)
            except (TypeError, ValueError):
                continue
            self._episode_reward_totals[field_name] = self._episode_reward_totals.get(field_name, 0.0) + value

    def _record_environment_safety(self, info: Dict[str, Any]) -> None:
        if not isinstance(info, dict):
            return
        self._episode_env_corruption_count += self._safe_int_value(info.get("env_corruption_count"), default=0)
        self._episode_mower_respawn_detected_count += self._safe_int_value(
            info.get("mower_respawn_detected_count"),
            default=0,
        )
        self._episode_cooldown_reset_detected_count += self._safe_int_value(
            info.get("cooldown_reset_detected_count"),
            default=0,
        )
        self._episode_board_refresh_detected_count += self._safe_int_value(
            info.get("board_refresh_detected_count"),
            default=0,
        )
        self._episode_false_reward_unlock_during_gameplay_count += self._safe_int_value(
            info.get("false_reward_unlock_during_gameplay_count"),
            default=0,
        )
        self._episode_false_cleanup_reward_ui_during_gameplay_count += self._safe_int_value(
            info.get("false_cleanup_reward_ui_during_gameplay_count"),
            default=0,
        )
        self._episode_post_win_veto_live_board_count += self._safe_int_value(
            info.get("post_win_veto_live_board_count"),
            default=0,
        )
        self._episode_blocked_cleanup_during_gameplay_count += self._safe_int_value(
            info.get("blocked_cleanup_during_gameplay_count"),
            default=0,
        )
        self._episode_suspicious_cleanup_reward_ui_count += self._safe_int_value(
            info.get("suspicious_cleanup_reward_ui_count"),
            default=0,
        )
        self._episode_reset_reward_ui_cleanup_count += self._safe_int_value(
            info.get("reset_reward_ui_cleanup_count"),
            default=0,
        )
        self._episode_reset_reward_ui_cleanup_blocked_count += self._safe_int_value(
            info.get("reset_reward_ui_cleanup_blocked_count"),
            default=0,
        )

    def _plant_counts_by_row(self, observation: Dict[str, Any], plant_type: Optional[int] = None) -> Counter[int]:
        counts: Counter[int] = Counter()
        if not isinstance(observation, dict):
            return counts
        for plant in observation.get("plants", []) or []:
            if not isinstance(plant, dict):
                continue
            row = self._safe_int_value(plant.get("row"), default=-1)
            observed_type = self._safe_int_value(plant.get("type"), default=-999)
            if row < 0:
                continue
            if plant_type is not None and observed_type != int(plant_type):
                continue
            counts[row] += 1
        return counts

    def _replace_row_counts(self, target: Counter[int], values: Any, fallback: Counter[int]) -> None:
        target.clear()
        if isinstance(values, dict):
            self._add_row_counts(target, values)
        else:
            target.update(fallback)

    def _add_row_counts(self, target: Counter[int], values: Any) -> None:
        if not isinstance(values, dict):
            return
        for key, value in values.items():
            try:
                row = int(key)
                count = int(value)
            except (TypeError, ValueError):
                continue
            target[row] += count

    def _update_row_max(self, target: Counter[int], values: Any) -> None:
        if not isinstance(values, dict):
            return
        for key, value in values.items():
            try:
                row = int(key)
                count = int(value)
            except (TypeError, ValueError):
                continue
            target[row] = max(int(target.get(row, 0)), count)

    def _add_reason_counts(self, target: Counter[str], values: Any) -> None:
        if not isinstance(values, dict):
            return
        for key, value in values.items():
            try:
                count = int(value)
            except (TypeError, ValueError):
                continue
            target[str(key)] += count

    def _bump_row(self, target: Counter[int], row: int) -> None:
        if row >= 0:
            target[row] += 1

    def _row_dict(self, values: Counter[int], rows: Optional[int] = None) -> Dict[str, int]:
        row_count = self.rows if rows is None else rows
        result = {str(row): int(values.get(row, 0)) for row in range(max(0, row_count))}
        for row, count in values.items():
            if 0 <= int(row) < row_count:
                continue
            result[str(row)] = int(count)
        return result

    def _ratio_row_dict(self, numerator: Counter[int], denominator: Counter[int]) -> Dict[str, float]:
        result: Dict[str, float] = {}
        rows = sorted(set(range(max(0, self.rows))) | set(int(row) for row in numerator.keys()) | set(int(row) for row in denominator.keys()))
        for row in rows:
            denom = int(denominator.get(row, 0))
            result[str(row)] = float(numerator.get(row, 0)) / float(denom) if denom > 0 else 0.0
        return result

    @staticmethod
    def _ratio_total(numerator: Counter[int], denominator: Counter[int]) -> float:
        denom = sum(int(value) for value in denominator.values())
        return float(sum(int(value) for value in numerator.values())) / float(denom) if denom > 0 else 0.0

    @staticmethod
    def _plain_counter_dict(values: Counter[Any]) -> Dict[str, int]:
        return {str(key): int(value) for key, value in sorted(values.items(), key=lambda item: str(item[0]))}

    @staticmethod
    def _safe_int_value(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    def close(self) -> None:
        self.base.close()
        super().close()

    def _encode_observation(self, observation: Dict[str, Any]) -> np.ndarray:
        values: List[float] = []
        max_wave = max(1, int(observation.get("maxWave", 0) or 1))
        cells = max(1, self.cells)
        legal_count = len(observation.get("legalActions", []))
        values.extend(
            [
                self._clip(float(observation.get("sun", 0)) / 1000.0),
                self._clip(float(observation.get("wave", 0)) / max_wave),
                self._clip(max_wave / 20.0),
                self._clip(float(observation.get("killCount", 0)) / 100.0),
                self._clip(float(observation.get("time", 0.0)) / 300.0),
                1.0 if observation.get("moreZombiesComing") else 0.0,
                1.0 if observation.get("gameplayReady") else 0.0,
                1.0 if observation.get("done") else 0.0,
                1.0 if observation.get("over") else 0.0,
                self._clip(legal_count / max(1, self.action_count)),
                self._clip(float(observation.get("plantCount", 0)) / cells),
                self._clip(float(observation.get("zombieCount", 0)) / 50.0),
            ]
        )

        raw_slots = list(observation.get("seedSlots", []) or [])
        sun = float(observation.get("sun", 0))
        for slot_index in range(self.card_slot_count):
            plant_type = (
                int(self.config.plant_types[slot_index])
                if slot_index < len(self.config.plant_types)
                else -1
            )
            card = raw_slots[slot_index] if slot_index < len(raw_slots) else {}
            full_cd = max(1e-6, float(card.get("fullCooldown", 0.0) or 0.0))
            cost = float(card.get("seedCost", 0.0) or self._cost_for_type(plant_type))
            values.extend(
                [
                    1.0 if card.get("ready") else 0.0,
                    self._clip(float(card.get("currentCooldown", 0.0) or 0.0) / full_cd),
                    self._clip(float(card.get("rawCooldown", 0.0) or 0.0) / full_cd),
                    self._clip(cost / 500.0),
                    1.0 if sun >= cost else 0.0,
                ]
            )

        plant_grid = {(int(p.get("row", -1)), int(p.get("column", -1))): p for p in observation.get("plants", [])}
        for row in range(self.rows):
            for col in range(self.cols):
                plant = plant_grid.get((row, col))
                if not plant:
                    values.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    continue
                max_health = max(1.0, float(plant.get("maxHealth", 1) or 1))
                plant_type = int(plant.get("type", -1))
                values.extend(
                    [
                        1.0,
                        1.0 if plant_type == 1 else 0.0,
                        1.0 if plant_type == 0 else 0.0,
                        self._clip(float(plant.get("health", 0)) / max_health),
                        self._clip(float(plant.get("attackCooldown", 0.0) or 0.0) / 10.0),
                        self._clip(float(plant.get("produceCooldown", 0.0) or 0.0) / 30.0),
                    ]
                )

        lanes = {int(lane.get("row", -1)): lane for lane in observation.get("lanes", [])}
        for row in range(self.rows):
            lane = lanes.get(row, {})
            nearest_x = lane.get("nearestZombieX")
            if nearest_x is None:
                nearest_x_value = 1.0
                danger = 0.0
            else:
                nearest_x_float = float(nearest_x)
                nearest_x_value = self._clip(nearest_x_float / 12.0)
                danger = self._clip(max(0.0, 1.0 - nearest_x_float / 10.0))
            values.extend(
                [
                    self._clip(float(lane.get("zombieCount", 0)) / 10.0),
                    nearest_x_value,
                    self._clip(float(lane.get("nearestZombieHealth", 0) or 0) / 1000.0),
                    self._clip(float(lane.get("nearestZombieType", 0) or 0) / 100.0),
                    danger,
                ]
            )

        if self.action_spec.mode == ACTION_SPACE_ADVENTURE_14_IDENTITY:
            values.extend(adventure_identity_features(observation, self.action_spec.max_seed_slots))
        elif self.action_spec.dynamic_seed_slots:
            values.extend(seed_inventory_v2_features(observation, self.action_spec.max_seed_slots))

        return np.asarray(values, dtype=np.float32)

    @staticmethod
    def _clip(value: float) -> float:
        return float(np.clip(value, -10.0, 10.0))

    @staticmethod
    def _cost_for_type(plant_type: int) -> int:
        if int(plant_type) == 1:
            return 50
        if int(plant_type) == 0:
            return 100
        return 0
