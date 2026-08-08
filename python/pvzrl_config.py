"""Typed resolved configuration and precedence helpers for PvZRL runs.

The runtime consumes one flat dictionary. ``ResolvedRunConfig`` adds immutable,
coherent sections without changing that public contract:
``to_flat_dict()`` returns the same keys and JSON-compatible value shapes.
"""

from __future__ import annotations

import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, Generic, Mapping, Optional, Tuple, TypeVar


class _Unset:
    def __repr__(self) -> str:
        return "CONFIG_UNSET"


CONFIG_UNSET = _Unset()


class IgnoredLegacyConfigWarning(UserWarning):
    """A recognized legacy JSON key no longer controls runtime behavior."""


IGNORED_LEGACY_FIELDS: Mapping[str, str] = MappingProxyType(
    {
        "enable_fusion_diagnostics": (
            "fusion diagnostics are emitted by the supported fusion pipeline; "
            "use fusion_policy and the fusion reward/diagnostic settings instead"
        ),
        "proximity_penalty": (
            "the former zombie-proximity reward term has no runtime consumer; "
            "the value is retained in resolved reward output only for compatibility"
        ),
    }
)
_NESTED_REWARD_LEGACY_FIELDS = frozenset({"proximity_penalty"})


def warn_ignored_legacy_fields(json_values: Mapping[str, Any]) -> Tuple[str, ...]:
    """Warn deterministically for recognized legacy no-op JSON settings."""

    ignored_locations = [key for key in IGNORED_LEGACY_FIELDS if key in json_values]
    raw_reward = json_values.get("reward", {})
    if isinstance(raw_reward, Mapping):
        ignored_locations.extend(
            f"reward.{key}"
            for key in _NESTED_REWARD_LEGACY_FIELDS
            if key in raw_reward
        )
    ignored = tuple(ignored_locations)
    for location in ignored:
        key = location.rsplit(".", 1)[-1]
        warnings.warn(
            f"Ignored legacy configuration key {location!r}: {IGNORED_LEGACY_FIELDS[key]}.",
            IgnoredLegacyConfigWarning,
            stacklevel=2,
        )
    return ignored


class ConfigSource(str, Enum):
    """The precedence tier that supplied a resolved value."""

    CLI = "cli"
    JSON = "json"
    MODE_DEFAULT = "mode_default"
    GLOBAL_DEFAULT = "global_default"


T = TypeVar("T")


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class ResolvedValue(Generic[T]):
    value: T
    source: ConfigSource


@dataclass
class ConfigResolver:
    """Resolve CLI/JSON/default values and retain their authoritative source."""

    cli_namespace: Any
    json_values: Mapping[str, Any]
    sources: Dict[str, ConfigSource] = field(default_factory=dict)

    def value(
        self,
        key: str,
        global_default: Any,
        *,
        mode_default: Any = CONFIG_UNSET,
    ) -> Any:
        mode_defaults = None if mode_default is CONFIG_UNSET else {key: mode_default}
        resolved = resolve_config_value(
            key,
            cli_namespace=self.cli_namespace,
            json_values=self.json_values,
            mode_defaults=mode_defaults,
            global_defaults={key: global_default},
        )
        self.sources[key] = resolved.source
        return resolved.value

    def enabled(
        self,
        key: str,
        global_default: bool = False,
        *,
        cli_key: Optional[str] = None,
        json_aliases: Tuple[str, ...] = (),
        mode_default: Any = CONFIG_UNSET,
    ) -> bool:
        """Resolve an enable-only argparse switch without treating False as explicit.

        Store-true options use ``False`` as argparse's unsupplied sentinel. A
        true CLI flag wins; otherwise JSON (including false), mode, and global
        defaults are considered in order. Paired true/false options should use
        :meth:`value` with an argparse default of ``None`` instead.
        """

        cli_name = cli_key or key
        if getattr(self.cli_namespace, cli_name, False) is True:
            self.sources[key] = ConfigSource.CLI
            return True
        json_keys = tuple(json_key for json_key in (key, *json_aliases) if json_key in self.json_values)
        if json_keys:
            self.sources[key] = ConfigSource.JSON
            # Legacy enable aliases historically combined with logical OR.
            return any(bool(self.json_values[json_key]) for json_key in json_keys)
        if mode_default is not CONFIG_UNSET:
            self.sources[key] = ConfigSource.MODE_DEFAULT
            return bool(mode_default)
        self.sources[key] = ConfigSource.GLOBAL_DEFAULT
        return bool(global_default)

    def aliased_value(
        self,
        key: str,
        global_default: Any,
        *,
        cli_keys: Tuple[str, ...],
        json_keys: Tuple[str, ...],
        skip_blank: bool = False,
    ) -> Any:
        """Resolve one semantic value exposed through multiple option names."""

        cli_candidates = []
        for cli_key in cli_keys:
            candidate = getattr(self.cli_namespace, cli_key, CONFIG_UNSET)
            if candidate is CONFIG_UNSET or candidate is None or (skip_blank and candidate == ""):
                continue
            cli_candidates.append((cli_key, candidate))
        distinct_cli_values = {str(candidate) for _name, candidate in cli_candidates}
        if len(distinct_cli_values) > 1:
            raise ValueError(
                f"conflicting CLI aliases for {key}: "
                + ", ".join(f"{name}={candidate!r}" for name, candidate in cli_candidates)
            )
        if cli_candidates:
            self.sources[key] = ConfigSource.CLI
            return cli_candidates[0][1]

        for json_key in json_keys:
            if json_key not in self.json_values:
                continue
            candidate = self.json_values[json_key]
            if skip_blank and candidate in (None, ""):
                continue
            self.sources[key] = ConfigSource.JSON
            return candidate
        self.sources[key] = ConfigSource.GLOBAL_DEFAULT
        return global_default


def resolve_config_value(
    key: str,
    *,
    cli_namespace: Any,
    json_values: Mapping[str, Any],
    mode_defaults: Optional[Mapping[str, Any]] = None,
    global_defaults: Optional[Mapping[str, Any]] = None,
) -> ResolvedValue[Any]:
    """Resolve ``key`` using CLI > JSON > mode > global precedence.

    ``None`` is the argparse marker for an unsupplied option.  Other falsey
    values, including ``False``, ``0``, and ``""``, are explicit and win.
    JSON ``null`` remains explicit to preserve the pre-existing JSON contract.
    """

    cli_value = getattr(cli_namespace, key, CONFIG_UNSET)
    if cli_value is not CONFIG_UNSET and cli_value is not None:
        return ResolvedValue(cli_value, ConfigSource.CLI)
    if key in json_values:
        return ResolvedValue(json_values[key], ConfigSource.JSON)
    if mode_defaults is not None and key in mode_defaults:
        return ResolvedValue(mode_defaults[key], ConfigSource.MODE_DEFAULT)
    if global_defaults is not None and key in global_defaults:
        return ResolvedValue(global_defaults[key], ConfigSource.GLOBAL_DEFAULT)
    raise KeyError(f"No configured value or default for {key!r}")


def _tuple_str(value: Any) -> Tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value)


def _tuple_int(value: Any) -> Tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(int(item) for item in value)


@dataclass(frozen=True)
class OptimizationConfig:
    policy: str
    total_timesteps: int
    learning_rate: float
    n_steps: int
    batch_size: int
    gamma: float
    gae_lambda: float
    ent_coef: float
    clip_range: float
    verbose: int


@dataclass(frozen=True)
class EnvironmentConfig:
    max_steps: int
    step_seconds: float
    game_speed: float
    game_speed_mode: str
    start_sun: int
    seed: int
    board_timeout: float
    gameplay_ready_timeout: float
    poll_seconds: float
    wait_gameplay_ready: bool
    skip_board_wait: bool
    quick_wait: bool


@dataclass(frozen=True)
class SeedActionConfig:
    plant_types: Tuple[int, ...]
    action_space_mode: str
    max_seed_slots: int
    auto_select_seeds: bool
    seed_list: Tuple[str, ...]
    initial_loadout: Tuple[str, ...]
    configured_seed_list: Tuple[str, ...]
    seed_order_source: str
    seed_order_preserved: bool
    randomize_seed_order: bool
    seed_click_delay: float
    lets_rock_delay: float
    post_start_delay: float
    seed_screen_check_interval: int


@dataclass(frozen=True)
class AdventureConfig:
    run_mode: str
    adventure_soft_max_steps: int
    adventure_hard_max_steps: int
    adventure_final_wave_extension: bool
    checkpoint_warm_start: bool
    warm_start_used: bool
    checkpoint_warm_start_reason: str
    resume_training: bool
    resume_model_path: str
    resume_source_model_family: str
    scratch_initialization: bool
    active_seed_slots_at_start: int
    unlock_aware_seed_curriculum: bool
    seed_curriculum: str
    unlock_introduction_delay: int
    new_plant_min_inclusion_prob: float
    infer_capacity_from_unlocks: bool
    allow_weak_unlocked_capacity_fallback: bool
    adventure_replay_cleared_levels: bool
    adventure_frontier_sample_prob: float
    adventure_recent_cleared_sample_prob: float
    adventure_maintenance_sample_prob: float
    adventure_frontier_win_streak_required: int
    adventure_generalist_strict_startup_validation: bool
    adventure_start_level: int
    max_adventure_levels: int
    max_attempts_per_level: int
    advance_on_wins: int


@dataclass(frozen=True)
class FusionConfig:
    fusion_policy: str
    fusion_action_mask_enabled: bool
    enable_board_plant_identity: bool
    enable_fusion_chain_rewards: bool
    enable_recipe_discovery_reward: bool
    enable_repeat_recipe_decay: bool
    enable_fusion_curriculum: bool
    enable_later_plant_curriculum: bool
    enable_coach_fusion_sampling: bool
    fusion_curriculum_prob: float
    later_plant_curriculum_prob: float
    coach_fusion_prob: float


@dataclass(frozen=True)
class CoachConfig:
    human_coach_enabled: bool
    human_coach_command_path: str
    human_coach_log_path: str
    human_coach_reward: bool
    human_coach_fusion_enabled: bool
    human_coach_platform: str
    human_coach_command_mode: str
    intervention_log_path: str
    stream_coach_enabled: bool
    stream_coach_mode: str
    stream_coach_platform: str
    stream_coach_window_sec: float
    stream_coach_min_votes: int
    stream_coach_max_actions_per_minute: int
    stream_coach_command_path: str
    stream_coach_mock_script: str
    stream_coach_dry_run: bool
    stream_coach_apply_enabled: bool
    stream_coach_reward: bool
    stream_coach_log_path: str
    stream_coach_fusion_enabled: bool
    coach_allow_fusion_planning: bool
    fusion_bridge_enabled: bool


@dataclass(frozen=True)
class StreamerV1Config:
    """Typed Streamer overlay configuration for the maintained Generalist path."""

    enabled: bool
    platform: str
    baseline_checkpoint: str
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
    twitch_client_id_env: str
    twitch_access_token_env: str
    twitch_broadcaster_id_env: str
    twitch_user_id_env: str
    viewer_hash_secret_env: str
    mock_script: str


@dataclass(frozen=True)
class DiagnosticsConfig:
    debug_performance: bool
    debug_observation: bool
    debug_sun: bool
    debug_sun_sample_interval: int
    tactical_masks: bool
    wallnut_tactical_mask: bool
    cherrybomb_tactical_mask: bool
    enable_action_watchdog: bool
    action_timeout_seconds: float
    save_freeze_debug_bundle: bool
    action_diagnostics_path: str
    freeze_debug_dir: str


@dataclass(frozen=True)
class ArtifactConfig:
    model_family: str
    model_path: str
    run_dir: str
    checkpoint_freq: int


@dataclass(frozen=True)
class BridgeConfig:
    host: str
    port: int
    timeout: float
    game_exe: str


@dataclass(frozen=True)
class ModelContractConfig:
    action_count: int
    dynamic_seed_slots: bool
    identity_seed_slots: bool
    observation_version: str
    action_decoder_version: str
    decoder_wait_action: int
    placement_action_range: Tuple[int, ...]
    rows: int
    cols: int
    cells_per_seed_slot: int


@dataclass(frozen=True)
class ResolvedRunConfig:
    """Immutable typed view over the resolved Generalist configuration."""

    optimization: OptimizationConfig
    environment: EnvironmentConfig
    seed_actions: SeedActionConfig
    adventure: AdventureConfig
    fusion: FusionConfig
    coach: CoachConfig
    streamer_v1: StreamerV1Config
    diagnostics: DiagnosticsConfig
    artifacts: ArtifactConfig
    bridge: BridgeConfig
    model_contract: ModelContractConfig
    reward: Mapping[str, float]
    value_sources: Mapping[str, ConfigSource]
    _flat: Mapping[str, Any] = field(repr=False, compare=False)

    @classmethod
    def from_flat(
        cls,
        values: Mapping[str, Any],
        *,
        value_sources: Optional[Mapping[str, ConfigSource]] = None,
    ) -> "ResolvedRunConfig":
        flat = deepcopy(dict(values))
        reward_values = flat.get("reward", {})
        reward = {
            str(key): float(value)
            for key, value in (reward_values.items() if isinstance(reward_values, Mapping) else ())
        }
        return cls(
            optimization=OptimizationConfig(
                policy=str(flat["policy"]),
                total_timesteps=int(flat["total_timesteps"]),
                learning_rate=float(flat["learning_rate"]),
                n_steps=int(flat["n_steps"]),
                batch_size=int(flat["batch_size"]),
                gamma=float(flat["gamma"]),
                gae_lambda=float(flat["gae_lambda"]),
                ent_coef=float(flat["ent_coef"]),
                clip_range=float(flat["clip_range"]),
                verbose=int(flat["verbose"]),
            ),
            environment=EnvironmentConfig(
                max_steps=int(flat["max_steps"]),
                step_seconds=float(flat["step_seconds"]),
                game_speed=float(flat["game_speed"]),
                game_speed_mode=str(flat["game_speed_mode"]),
                start_sun=int(flat["start_sun"]),
                seed=int(flat["seed"]),
                board_timeout=float(flat["board_timeout"]),
                gameplay_ready_timeout=float(flat["gameplay_ready_timeout"]),
                poll_seconds=float(flat["poll_seconds"]),
                wait_gameplay_ready=bool(flat["wait_gameplay_ready"]),
                skip_board_wait=bool(flat["skip_board_wait"]),
                quick_wait=bool(flat["quick_wait"]),
            ),
            seed_actions=SeedActionConfig(
                plant_types=_tuple_int(flat["plant_types"]),
                action_space_mode=str(flat["action_space_mode"]),
                max_seed_slots=int(flat["max_seed_slots"]),
                auto_select_seeds=bool(flat["auto_select_seeds"]),
                seed_list=_tuple_str(flat["seed_list"]),
                initial_loadout=_tuple_str(flat["initial_loadout"]),
                configured_seed_list=_tuple_str(flat["configured_seed_list"]),
                seed_order_source=str(flat["seed_order_source"]),
                seed_order_preserved=bool(flat["seed_order_preserved"]),
                randomize_seed_order=bool(flat["randomize_seed_order"]),
                seed_click_delay=float(flat["seed_click_delay"]),
                lets_rock_delay=float(flat["lets_rock_delay"]),
                post_start_delay=float(flat["post_start_delay"]),
                seed_screen_check_interval=int(flat["seed_screen_check_interval"]),
            ),
            adventure=AdventureConfig(
                run_mode=str(flat["run_mode"]),
                adventure_soft_max_steps=int(flat["adventure_soft_max_steps"]),
                adventure_hard_max_steps=int(flat["adventure_hard_max_steps"]),
                adventure_final_wave_extension=bool(flat["adventure_final_wave_extension"]),
                checkpoint_warm_start=bool(flat["checkpoint_warm_start"]),
                warm_start_used=bool(flat["warm_start_used"]),
                checkpoint_warm_start_reason=str(flat["checkpoint_warm_start_reason"]),
                resume_training=bool(flat["resume_training"]),
                resume_model_path=str(flat["resume_model_path"]),
                resume_source_model_family=str(flat["resume_source_model_family"]),
                scratch_initialization=bool(flat["scratch_initialization"]),
                active_seed_slots_at_start=int(flat["active_seed_slots_at_start"]),
                unlock_aware_seed_curriculum=bool(flat["unlock_aware_seed_curriculum"]),
                seed_curriculum=str(flat["seed_curriculum"]),
                unlock_introduction_delay=int(flat["unlock_introduction_delay"]),
                new_plant_min_inclusion_prob=float(flat["new_plant_min_inclusion_prob"]),
                infer_capacity_from_unlocks=bool(flat["infer_capacity_from_unlocks"]),
                allow_weak_unlocked_capacity_fallback=bool(flat["allow_weak_unlocked_capacity_fallback"]),
                adventure_replay_cleared_levels=bool(flat["adventure_replay_cleared_levels"]),
                adventure_frontier_sample_prob=float(flat["adventure_frontier_sample_prob"]),
                adventure_recent_cleared_sample_prob=float(flat["adventure_recent_cleared_sample_prob"]),
                adventure_maintenance_sample_prob=float(flat["adventure_maintenance_sample_prob"]),
                adventure_frontier_win_streak_required=int(flat["adventure_frontier_win_streak_required"]),
                adventure_generalist_strict_startup_validation=bool(flat["adventure_generalist_strict_startup_validation"]),
                adventure_start_level=int(flat["adventure_start_level"]),
                max_adventure_levels=int(flat["max_adventure_levels"]),
                max_attempts_per_level=int(flat["max_attempts_per_level"]),
                advance_on_wins=int(flat["advance_on_wins"]),
            ),
            fusion=FusionConfig(
                fusion_policy=str(flat["fusion_policy"]),
                fusion_action_mask_enabled=bool(flat["fusion_action_mask_enabled"]),
                enable_board_plant_identity=bool(flat["enable_board_plant_identity"]),
                enable_fusion_chain_rewards=bool(flat["enable_fusion_chain_rewards"]),
                enable_recipe_discovery_reward=bool(flat["enable_recipe_discovery_reward"]),
                enable_repeat_recipe_decay=bool(flat["enable_repeat_recipe_decay"]),
                enable_fusion_curriculum=bool(flat["enable_fusion_curriculum"]),
                enable_later_plant_curriculum=bool(flat["enable_later_plant_curriculum"]),
                enable_coach_fusion_sampling=bool(flat["enable_coach_fusion_sampling"]),
                fusion_curriculum_prob=float(flat["fusion_curriculum_prob"]),
                later_plant_curriculum_prob=float(flat["later_plant_curriculum_prob"]),
                coach_fusion_prob=float(flat["coach_fusion_prob"]),
            ),
            coach=CoachConfig(
                human_coach_enabled=bool(flat["human_coach_enabled"]),
                human_coach_command_path=str(flat["human_coach_command_path"]),
                human_coach_log_path=str(flat["human_coach_log_path"]),
                human_coach_reward=bool(flat["human_coach_reward"]),
                human_coach_fusion_enabled=bool(flat["human_coach_fusion_enabled"]),
                human_coach_platform=str(flat["human_coach_platform"]),
                human_coach_command_mode=str(flat["human_coach_command_mode"]),
                intervention_log_path=str(flat["intervention_log_path"]),
                stream_coach_enabled=bool(flat["stream_coach_enabled"]),
                stream_coach_mode=str(flat["stream_coach_mode"]),
                stream_coach_platform=str(flat["stream_coach_platform"]),
                stream_coach_window_sec=float(flat["stream_coach_window_sec"]),
                stream_coach_min_votes=int(flat["stream_coach_min_votes"]),
                stream_coach_max_actions_per_minute=int(flat["stream_coach_max_actions_per_minute"]),
                stream_coach_command_path=str(flat["stream_coach_command_path"]),
                stream_coach_mock_script=str(flat["stream_coach_mock_script"]),
                stream_coach_dry_run=bool(flat["stream_coach_dry_run"]),
                stream_coach_apply_enabled=bool(flat["stream_coach_apply_enabled"]),
                stream_coach_reward=bool(flat["stream_coach_reward"]),
                stream_coach_log_path=str(flat["stream_coach_log_path"]),
                stream_coach_fusion_enabled=bool(flat["stream_coach_fusion_enabled"]),
                coach_allow_fusion_planning=bool(flat["coach_allow_fusion_planning"]),
                fusion_bridge_enabled=bool(flat["fusion_bridge_enabled"]),
            ),
            streamer_v1=StreamerV1Config(
                enabled=bool(flat["streamer_v1_enabled"]),
                platform=str(flat["streamer_platform"]),
                baseline_checkpoint=str(flat["streamer_baseline_checkpoint"]),
                intervention_interval_seconds=float(flat["streamer_intervention_interval_seconds"]),
                command_ttl_seconds=float(flat["streamer_command_ttl_seconds"]),
                command_queue_capacity=int(flat["streamer_command_queue_capacity"]),
                message_max_chars=int(flat["streamer_message_max_chars"]),
                policy_steps_per_cycle=int(flat["streamer_policy_steps_per_cycle"]),
                checkpoint_policy_steps=int(flat["streamer_checkpoint_policy_steps"]),
                evaluation_episodes=int(flat["streamer_evaluation_episodes"]),
                max_cycles=int(flat["streamer_max_cycles"]),
                endurance_hours=float(flat["streamer_endurance_hours"]),
                bc_enabled=bool(flat["streamer_bc_enabled"]),
                bc_coefficient=float(flat["streamer_bc_coefficient"]),
                demonstration_capacity=int(flat["streamer_demonstration_capacity"]),
                demonstration_persist_every=int(flat["streamer_demonstration_persist_every"]),
                bc_batch_size=int(flat["streamer_bc_batch_size"]),
                bc_update_frequency=int(flat["streamer_bc_update_frequency"]),
                bc_min_demonstrations=int(flat["streamer_bc_min_demonstrations"]),
                twitch_client_id_env=str(flat["streamer_twitch_client_id_env"]),
                twitch_access_token_env=str(flat["streamer_twitch_access_token_env"]),
                twitch_broadcaster_id_env=str(flat["streamer_twitch_broadcaster_id_env"]),
                twitch_user_id_env=str(flat["streamer_twitch_user_id_env"]),
                viewer_hash_secret_env=str(flat["streamer_viewer_hash_secret_env"]),
                mock_script=str(flat["streamer_mock_script"]),
            ),
            diagnostics=DiagnosticsConfig(
                debug_performance=bool(flat["debug_performance"]),
                debug_observation=bool(flat["debug_observation"]),
                debug_sun=bool(flat["debug_sun"]),
                debug_sun_sample_interval=int(flat["debug_sun_sample_interval"]),
                tactical_masks=bool(flat["tactical_masks"]),
                wallnut_tactical_mask=bool(flat["wallnut_tactical_mask"]),
                cherrybomb_tactical_mask=bool(flat["cherrybomb_tactical_mask"]),
                enable_action_watchdog=bool(flat["enable_action_watchdog"]),
                action_timeout_seconds=float(flat["action_timeout_seconds"]),
                save_freeze_debug_bundle=bool(flat["save_freeze_debug_bundle"]),
                action_diagnostics_path=str(flat["action_diagnostics_path"]),
                freeze_debug_dir=str(flat["freeze_debug_dir"]),
            ),
            artifacts=ArtifactConfig(
                model_family=str(flat["model_family"]),
                model_path=str(flat["model_path"]),
                run_dir=str(flat["run_dir"]),
                checkpoint_freq=int(flat["checkpoint_freq"]),
            ),
            bridge=BridgeConfig(
                host=str(flat["host"]),
                port=int(flat["port"]),
                timeout=float(flat["timeout"]),
                game_exe=str(flat["game_exe"]),
            ),
            model_contract=ModelContractConfig(
                action_count=int(flat["action_count"]),
                dynamic_seed_slots=bool(flat["dynamic_seed_slots"]),
                identity_seed_slots=bool(flat["identity_seed_slots"]),
                observation_version=str(flat["observation_version"]),
                action_decoder_version=str(flat["action_decoder_version"]),
                decoder_wait_action=int(flat["decoder_wait_action"]),
                placement_action_range=_tuple_int(flat["placement_action_range"]),
                rows=int(flat["rows"]),
                cols=int(flat["cols"]),
                cells_per_seed_slot=int(flat["cells_per_seed_slot"]),
            ),
            reward=MappingProxyType(reward),
            value_sources=MappingProxyType(dict(value_sources or {})),
            _flat=_deep_freeze(flat),
        )

    def to_flat_dict(self) -> Dict[str, Any]:
        """Return a defensive copy of the backward-compatible flat mapping."""

        return _deep_thaw(self._flat)
