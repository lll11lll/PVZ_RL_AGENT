"""Adventure Generalist 14-slot identity training support."""

from __future__ import annotations

import json
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pvzrl_adventure import (
    BASE_UNLOCKED_SEEDS,
    LEVEL_IDENTITY_POST_WIN_STATES,
    LiveStatusWriter,
    adventure_bridge_detected_level,
    adventure_challenge_mode_context,
    adventure_gameplay_ready_detected,
    adventure_level_identity_diagnostics,
    adventure_profile_adventure_level,
    adventure_screen_state_name,
    adventure_seed_selection_detected,
    adventure_ui_world_level_text,
    build_live_status,
    canonical_seed_name as _adventure_canonical_seed_name,
    collect_post_win_unlocks,
    prepare_adventure_gameplay,
    replay_current_level_after_validation_win,
)
from pvzrl_env import (
    active_gameplay_bank_state,
    normalize_plant_name,
    parse_seed_list,
    plant_type_name,
    registry_entries,
    resolve_seed_list,
)
from pvzrl_fusion import FUSION_POLICY_SCRIPTED
from pvzrl_generalist_progression import (
    GeneralistEpisodeOutcome,
    GeneralistProgressionConfig,
    GeneralistProgressionState,
    GeneralistProgressionTransition,
    begin_generalist_attempt,
    fresh_generalist_progression,
    reduce_generalist_episode,
)
from pvzrl_sb3 import PvZMaskedPPOEnv, PvZSB3Config
from pvzrl_telemetry import live_status_significant_state


ADVENTURE_GENERALIST_MODEL_FAMILY = "ppo_adventure_generalist_14slot_identity_v1"
ADVENTURE_GENERALIST_RUN_MODE_TRAIN = "adventure_generalist_14slot_train"
ADVENTURE_GENERALIST_RUN_MODE_EVAL = "adventure_generalist_14slot_eval"
ADVENTURE_GENERALIST_INITIAL_LOADOUT = ["SunFlower", "SunFlower", "Peashooter", "Peashooter"]
BLOCKED_INITIAL_LOADOUT_UNAVAILABLE = "required_initial_adventure_generalist_loadout_unavailable"
BLOCKED_FRONTIER_REPLAY_REQUIRED = "frontier_win_streak_requires_same_level_replay_support"
BLOCKED_STARTUP_VALIDATION_FAILED = "adventure_generalist_startup_validation_failed"
SEED_ORDER_SOURCE_EXPLICIT = "explicit_config"
SEED_ORDER_SOURCE_DEFAULT = "default_canonical"
SEED_ORDER_SOURCE_RANDOMIZED = "randomized"
CURRICULUM_STATE_SCHEMA_VERSION = 1

SEED_PRIORITY = [
    "SunFlower",
    "Peashooter",
    "WallNut",
    "CherryBomb",
    "PotatoMine",
    "Chomper",
    "SmallPuff",
    "FumeShroom",
    "HypnoShroom",
    "ScaredyShroom",
    "IceShroom",
    "DoomShroom",
    "LilyPad",
    "Squash",
    "ThreePeater",
    "Tanglekelp",
    "Jalapeno",
    "Caltrop",
    "TorchWood",
    "SeaShroom",
    "Plantern",
    "Cactus",
    "Blover",
    "StarFruit",
    "Pumpkin",
    "Magnetshroom",
    "Cabbagepult",
    "Pot",
    "Cornpult",
    "Garlic",
    "Umbrellaleaf",
    "Marigold",
    "Melonpult",
    "Shulkflower",
    "ElectricOnion",
    "PineFurnace",
    "SpruceShooter",
    "IceLotus",
    "WaterAloes",
    "Bamboo",
    "Imitater",
]

# Capacity probes are evidence about the live bank, not a level/night policy.
# Keep the bounded fallback derived from the canonical catalog so adding a
# newly supported runtime card does not require another gameplay branch.
SEED_CAPACITY_INFERENCE_PRIORITY = [
    seed for seed in SEED_PRIORITY if seed not in BASE_UNLOCKED_SEEDS
][:4]

SEED_CAPACITY_MAX = 14
DEFAULT_CORE_SEED_NAMES = ("SunFlower", "Peashooter")
DEFAULT_NEW_UNLOCK_GUARANTEE_EPISODES = 4


@dataclass
class SeedCapacityInference:
    bridge_reported_capacity: Optional[int]
    observed_capacity: int
    inferred_capacity_from_unlocks: int
    effective_seed_capacity: int
    max_effective_seed_capacity_seen: int
    inferred_capacity_source: str
    capacity_inference_reason: str
    available_priority_seeds: List[str] = field(default_factory=list)
    rejected_priority_seeds: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class AdventureGeneralistProgress:
    episode: int
    level: int
    attempt: int
    sample_source: str
    result: str
    selected_loadout: List[str]
    active_seed_slot_count: int
    unlocked_seeds: List[str]
    configured_seed_list: List[str] = field(default_factory=list)
    selected_loadout_count: int = 0
    observed_seed_bank_capacity: int = 0
    bridge_reported_capacity: Optional[int] = None
    inferred_capacity_from_unlocks: int = 0
    effective_seed_capacity: int = 0
    max_effective_seed_capacity_seen: int = 0
    inferred_capacity_source: str = ""
    capacity_inference_reason: str = ""
    available_priority_seeds: List[str] = field(default_factory=list)
    rejected_priority_seeds: List[Dict[str, str]] = field(default_factory=list)
    inactive_model_slots: int = 0
    selectable_seeds: List[str] = field(default_factory=list)
    eligible_seeds: List[str] = field(default_factory=list)
    loadout_reason: str = ""
    seed_order_source: str = SEED_ORDER_SOURCE_DEFAULT
    seed_order_preserved: bool = True
    seed_order_blocked_reason: str = ""
    excluded_new_plants: List[Dict[str, str]] = field(default_factory=list)
    newly_unlocked: List[str] = field(default_factory=list)
    post_win_transition: Dict[str, Any] = field(default_factory=dict)
    post_win_blocked_reason: str = ""
    post_win_last_state: Dict[str, Any] = field(default_factory=dict)
    unknown_unlock_objects: List[Dict[str, Any]] = field(default_factory=list)
    terminal_reason: str = ""
    timeout_classification: str = "none"
    reward: float = 0.0
    length: int = 0
    plant_availability: List[Dict[str, Any]] = field(default_factory=list)
    plant_action_counts: Dict[str, int] = field(default_factory=dict)
    successful_placements_by_plant: Dict[str, int] = field(default_factory=dict)
    invalid_actions_by_plant: Dict[str, int] = field(default_factory=dict)
    action_mask_validity_by_seed_slot: Dict[str, int] = field(default_factory=dict)
    fusion_attempts_by_pair: Dict[str, int] = field(default_factory=dict)
    fusion_successes_by_pair: Dict[str, int] = field(default_factory=dict)
    fusion_depth_counts: Dict[str, int] = field(default_factory=dict)
    highest_fusion_tier: int = 0
    recursive_fusion_count: int = 0
    action_freeze_count: int = 0
    reward_component_totals: Dict[str, float] = field(default_factory=dict)
    frontier_win_streak: int = 0
    frontier_win_streak_required: int = 1
    frontier_mastery_ready: bool = False
    frontier_promoted_this_episode: bool = False
    frontier_mastery_reset_reason: str = ""
    mastery_sample_source: str = ""
    frontier_replay_supported: bool = True
    frontier_replay_blocked_reason: str = ""
    frontier_mastered_levels: List[int] = field(default_factory=list)
    loadout_provenance: List[Dict[str, Any]] = field(default_factory=list)
    per_seed_diagnostics: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class LoadoutDecision:
    selected_loadout: List[str]
    loadout_reason: str
    eligible_seeds: List[str]
    selectable_seeds: List[str]
    excluded_new_plants: List[Dict[str, str]]
    observed_seed_bank_capacity: int
    configured_seed_list: List[str] = field(default_factory=list)
    seed_order_source: str = SEED_ORDER_SOURCE_DEFAULT
    seed_order_preserved: bool = True
    blocked_reason: str = ""
    validation_source: str = "selectable"
    validation_seeds: List[str] = field(default_factory=list)
    guaranteed_seeds: List[str] = field(default_factory=list)
    proposed_rotation_cursor: int = 0
    loadout_provenance: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class UnlockGuaranteeState:
    seed: str
    required_inclusions: int
    completed_inclusions: int = 0

    @property
    def remaining(self) -> int:
        return max(0, int(self.required_inclusions) - int(self.completed_inclusions))

    @property
    def pending(self) -> bool:
        return self.remaining > 0


class AdventureSeedCurriculum:
    def __init__(
        self,
        *,
        initial_loadout: Iterable[str],
        max_seed_slots: int = SEED_CAPACITY_MAX,
        unlock_aware: bool = True,
        seed_curriculum: str = "conservative",
        unlock_introduction_delay: int = 0,
        new_plant_min_inclusion_prob: float = 0.15,
        seed_order_source: str = SEED_ORDER_SOURCE_DEFAULT,
        randomize_seed_order: bool = False,
        core_seed_names: Optional[Iterable[str]] = None,
        new_unlock_guarantee_episodes: int = DEFAULT_NEW_UNLOCK_GUARANTEE_EPISODES,
    ) -> None:
        self.initial_loadout = [name for name in _canonical_seed_sequence(initial_loadout)]
        self.max_seed_slots = _clamp_capacity(max_seed_slots, maximum=SEED_CAPACITY_MAX)
        self.unlock_aware = bool(unlock_aware)
        self.seed_curriculum = str(seed_curriculum or "conservative").strip().lower()
        self.unlock_introduction_delay = max(0, int(unlock_introduction_delay))
        self.new_plant_min_inclusion_prob = max(0.0, min(1.0, float(new_plant_min_inclusion_prob)))
        self.seed_order_source = _normalize_seed_order_source(seed_order_source)
        self.randomize_seed_order = bool(randomize_seed_order)
        configured_core = core_seed_names if core_seed_names is not None else DEFAULT_CORE_SEED_NAMES
        self.core_seed_names = _ordered_unique_seed_names(list(configured_core))
        self.new_unlock_guarantee_episodes = max(0, int(new_unlock_guarantee_episodes))
        self.unlock_episode: Dict[str, int] = {}
        self.unlock_order: List[str] = []
        self.rotation_cursor = 0
        self.rng = random.Random()
        self.episode_index = 0
        self.episodes_included: Dict[str, int] = {}
        self.episodes_eligible: Dict[str, int] = {}
        self.model_actions_executed: Dict[str, int] = {}
        self.viewer_actions_executed: Dict[str, int] = {}
        self.bc_demonstrations_recorded: Dict[str, int] = {}
        self.last_included_episode: Dict[str, int] = {}
        self.last_eligible_episode: Dict[str, int] = {}
        self.confirmed_selectable_history: List[str] = []
        self.guarantees: Dict[str, UnlockGuaranteeState] = {}
        self.last_committed_loadout: List[str] = []
        self.last_committed_loadout_provenance: List[Dict[str, Any]] = []
        self.restored_current_loadout: List[str] = []
        self.restored_loadout_provenance: List[Dict[str, Any]] = []
        for seed in BASE_UNLOCKED_SEEDS:
            name = canonicalize_seed_name(seed)
            self.unlock_episode[name] = 0
            self.episodes_included.setdefault(name, 0)
            self.episodes_eligible.setdefault(name, 0)
            self.model_actions_executed.setdefault(name, 0)
            self.viewer_actions_executed.setdefault(name, 0)
            self.bc_demonstrations_recorded.setdefault(name, 0)
            if name not in self.unlock_order:
                self.unlock_order.append(name)
        for seed in self.initial_loadout:
            name = canonicalize_seed_name(seed)
            self.unlock_episode[name] = 0
            self.episodes_included.setdefault(name, 0)
            self.episodes_eligible.setdefault(name, 0)
            self.model_actions_executed.setdefault(name, 0)
            self.viewer_actions_executed.setdefault(name, 0)
            self.bc_demonstrations_recorded.setdefault(name, 0)
            if name not in self.unlock_order:
                self.unlock_order.append(name)

    def _ensure_seed_diagnostic(self, seed: Any) -> str:
        name = canonicalize_seed_name(seed)
        if not name:
            return ""
        self.episodes_included.setdefault(name, 0)
        self.episodes_eligible.setdefault(name, 0)
        self.model_actions_executed.setdefault(name, 0)
        self.viewer_actions_executed.setdefault(name, 0)
        self.bc_demonstrations_recorded.setdefault(name, 0)
        return name

    def record_unlocked(self, seeds: Iterable[str], episode_index: Optional[int] = None) -> List[str]:
        episode = self.episode_index if episode_index is None else int(episode_index)
        newly_unlocked: List[str] = []
        for seed in seeds:
            name = canonicalize_seed_name(seed)
            if not name or name in self.unlock_episode:
                continue
            self.unlock_episode[name] = episode
            self.unlock_order.append(name)
            if self.new_unlock_guarantee_episodes > 0:
                self.guarantees[name] = UnlockGuaranteeState(
                    seed=name,
                    required_inclusions=self.new_unlock_guarantee_episodes,
                )
            self._ensure_seed_diagnostic(name)
            if name not in self.confirmed_selectable_history:
                self.confirmed_selectable_history.append(name)
            newly_unlocked.append(name)
        return _ordered_by_priority(newly_unlocked)

    def record_confirmed_selectable(self, seeds: Iterable[str]) -> List[str]:
        confirmed: List[str] = []
        for seed in _canonical_seed_sequence(seeds):
            name = self._ensure_seed_diagnostic(seed)
            if not name:
                continue
            if name not in self.confirmed_selectable_history:
                self.confirmed_selectable_history.append(name)
                confirmed.append(name)
        return _ordered_by_priority(confirmed)

    def record_eligible(self, seeds: Iterable[str], episode_index: Optional[int] = None) -> None:
        episode = self.episode_index if episode_index is None else int(episode_index)
        for seed in _canonical_seed_sequence(seeds):
            name = canonicalize_seed_name(seed)
            if not name or name not in self.unlock_episode:
                continue
            self._ensure_seed_diagnostic(name)
            if self.last_eligible_episode.get(name) == episode:
                continue
            self.episodes_eligible[name] = int(self.episodes_eligible.get(name, 0)) + 1
            self.last_eligible_episode[name] = episode

    def record_action_usage(self, seed: Any, *, source: str) -> None:
        name = canonicalize_seed_name(seed)
        if not name or name not in self.unlock_episode:
            return
        self._ensure_seed_diagnostic(name)
        normalized_source = str(source or "model").strip().lower()
        if normalized_source == "viewer":
            self.viewer_actions_executed[name] = int(self.viewer_actions_executed.get(name, 0)) + 1
        else:
            self.model_actions_executed[name] = int(self.model_actions_executed.get(name, 0)) + 1

    def record_bc_demonstration(self, seed: Any) -> None:
        name = canonicalize_seed_name(seed)
        if not name or name not in self.unlock_episode:
            return
        self._ensure_seed_diagnostic(name)
        self.bc_demonstrations_recorded[name] = int(self.bc_demonstrations_recorded.get(name, 0)) + 1

    def loadout_provenance(self, loadout: Iterable[str]) -> List[Dict[str, Any]]:
        active_guarantees = {state.seed for state in self.active_guarantees()}
        core_seen: set[str] = set()
        provenance: List[Dict[str, Any]] = []
        for slot_index, seed in enumerate(_canonical_seed_sequence(loadout)):
            source = "rotation"
            if seed in self.core_seed_names and seed not in core_seen:
                source = "core"
                core_seen.add(seed)
            elif seed in active_guarantees:
                source = "guaranteed_unlock"
            provenance.append({"slot_index": int(slot_index), "seed": seed, "source": source})
        return provenance

    def per_seed_diagnostics(
        self,
        *,
        selected_loadout: Optional[Iterable[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        names = set(self.unlock_episode)
        names.update(self.episodes_included)
        names.update(self.episodes_eligible)
        names.update(self.model_actions_executed)
        names.update(self.viewer_actions_executed)
        names.update(self.bc_demonstrations_recorded)
        names.update(self.core_seed_names)
        selected_names = set(_canonical_seed_sequence(selected_loadout or []))
        result: Dict[str, Dict[str, Any]] = {}
        for name in self._unlock_ordered(names):
            state = self.guarantees.get(name)
            result[name] = {
                "episodes_eligible": int(self.episodes_eligible.get(name, 0)),
                "episodes_included": int(self.episodes_included.get(name, 0)),
                "model_actions_executed": int(self.model_actions_executed.get(name, 0)),
                "viewer_actions_executed": int(self.viewer_actions_executed.get(name, 0)),
                "bc_demonstrations_recorded": int(self.bc_demonstrations_recorded.get(name, 0)),
                "last_included_episode": self.last_included_episode.get(name),
                "currently_core": bool(name in self.core_seed_names),
                "currently_guaranteed": bool(state is not None and state.pending),
                "guarantee_remaining": int(state.remaining if state is not None else 0),
                "current_rotation_weight": float(self.rotation_weight(name)),
                "currently_selected": bool(name in selected_names),
                "unlock_episode": int(self.unlock_episode.get(name, 0)),
                "confirmed_selectable": bool(name in self.confirmed_selectable_history),
            }
        return result

    def restore_state(self, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            raise ValueError("curriculum_state_payload_not_object")
        if int(payload.get("schema_version", 0) or 0) != CURRICULUM_STATE_SCHEMA_VERSION:
            raise ValueError("curriculum_state_schema_version_mismatch")
        known_registry_names = {
            canonicalize_seed_name(entry.get("canonical_name", ""))
            for entry in registry_entries()
            if isinstance(entry, dict) and entry.get("canonical_name")
        }
        known_names = known_registry_names | set(self.initial_loadout) | set(BASE_UNLOCKED_SEEDS)

        def valid_name(value: Any) -> str:
            name = canonicalize_seed_name(value)
            return name if name in known_names else ""

        def restore_counter(key: str) -> Dict[str, int]:
            raw = payload.get(key, {})
            if not isinstance(raw, dict):
                return {}
            restored: Dict[str, int] = {}
            for raw_name, raw_value in raw.items():
                name = valid_name(raw_name)
                if not name:
                    continue
                try:
                    restored[name] = max(0, int(raw_value or 0))
                except (TypeError, ValueError):
                    raise ValueError(f"curriculum_state_invalid_counter:{key}:{raw_name}")
            return restored

        persisted_order = payload.get("unlock_order", [])
        if not isinstance(persisted_order, list):
            raise ValueError("curriculum_state_unlock_order_invalid")
        persisted_unlock_episode = restore_counter("unlock_episode")
        for raw_name in persisted_order:
            name = valid_name(raw_name)
            if name:
                persisted_unlock_episode.setdefault(name, 0)
        self.unlock_episode = {}
        self.unlock_order = []
        for seed in list(BASE_UNLOCKED_SEEDS) + list(self.initial_loadout) + list(persisted_order):
            name = valid_name(seed)
            if name and name not in self.unlock_episode:
                self.unlock_episode[name] = int(persisted_unlock_episode.get(name, 0))
                self.unlock_order.append(name)
        for name, episode in persisted_unlock_episode.items():
            if name not in self.unlock_episode:
                self.unlock_episode[name] = int(episode)
                self.unlock_order.append(name)

        self.episode_index = max(0, int(payload.get("episode_index", 0) or 0))
        self.episodes_included = restore_counter("episodes_included")
        self.episodes_eligible = restore_counter("episodes_eligible")
        self.model_actions_executed = restore_counter("model_actions_executed")
        self.viewer_actions_executed = restore_counter("viewer_actions_executed")
        self.bc_demonstrations_recorded = restore_counter("bc_demonstrations_recorded")
        self.last_included_episode = restore_counter("last_included_episode")
        self.last_eligible_episode = restore_counter("last_eligible_episode")
        self.confirmed_selectable_history = [
            name for name in _canonical_seed_sequence(payload.get("confirmed_selectable_history", []))
            if valid_name(name)
        ]
        self.rotation_cursor = max(0, int(payload.get("rotation_cursor", 0) or 0))
        raw_rng_state = payload.get("rng_state")
        if raw_rng_state is not None:
            if not isinstance(raw_rng_state, list) or len(raw_rng_state) != 3:
                raise ValueError("curriculum_state_rng_state_invalid")
            try:
                rng_state = (
                    int(raw_rng_state[0]),
                    tuple(int(value) for value in raw_rng_state[1]),
                    raw_rng_state[2],
                )
                self.rng.setstate(rng_state)
            except (TypeError, ValueError, IndexError) as exc:
                raise ValueError("curriculum_state_rng_state_invalid") from exc
        self.last_committed_loadout = [
            name for name in _canonical_seed_sequence(payload.get("last_committed_loadout", []))
            if valid_name(name)
        ]
        self.restored_current_loadout = [
            name for name in _canonical_seed_sequence(payload.get("current_loadout", []))
            if valid_name(name)
        ]
        raw_provenance = payload.get("loadout_provenance", [])
        self.restored_loadout_provenance = []
        if isinstance(raw_provenance, list):
            for row in raw_provenance:
                if not isinstance(row, dict):
                    continue
                name = valid_name(row.get("seed", ""))
                if not name:
                    continue
                try:
                    slot_index = int(row.get("slot_index", len(self.restored_loadout_provenance)))
                except (TypeError, ValueError):
                    continue
                self.restored_loadout_provenance.append(
                    {
                        "slot_index": slot_index,
                        "seed": name,
                        "source": str(row.get("source", "rotation") or "rotation"),
                    }
                )
        restored_guarantees = payload.get("guarantees", {})
        if not isinstance(restored_guarantees, dict):
            raise ValueError("curriculum_state_guarantees_invalid")
        self.guarantees = {}
        for raw_name, raw_state in restored_guarantees.items():
            name = valid_name(raw_name)
            if not name:
                continue
            if not isinstance(raw_state, dict):
                raise ValueError(f"curriculum_state_guarantee_invalid:{raw_name}")
            required = max(0, int(raw_state.get("required_inclusions", 0) or 0))
            completed = max(0, min(required, int(raw_state.get("completed_inclusions", 0) or 0)))
            self.guarantees[name] = UnlockGuaranteeState(name, required, completed)
        self.last_committed_loadout_provenance = list(self.restored_loadout_provenance)
        if not self.last_committed_loadout_provenance and self.last_committed_loadout:
            self.last_committed_loadout_provenance = self.loadout_provenance(
                self.last_committed_loadout
            )
        for name in self.unlock_episode:
            self._ensure_seed_diagnostic(name)
        for name in self.initial_loadout:
            if name not in self.confirmed_selectable_history:
                self.confirmed_selectable_history.append(name)

    def unlocked_seeds(self) -> List[str]:
        return _ordered_by_priority(self.unlock_episode.keys())

    def eligible_seeds(self) -> List[str]:
        eligible = set(self.initial_loadout)
        if self.unlock_aware:
            for seed, episode in self.unlock_episode.items():
                if self.episode_index - int(episode) >= self.unlock_introduction_delay:
                    eligible.add(seed)
        return _ordered_by_priority(eligible)

    def guarantee_remaining(self, seed: str) -> int:
        name = canonicalize_seed_name(seed)
        state = self.guarantees.get(name)
        return state.remaining if state is not None else 0

    def active_guarantees(self) -> List[UnlockGuaranteeState]:
        return [
            self.guarantees[name]
            for name in self._unlock_ordered(self.guarantees.keys())
            if self.guarantees[name].pending
        ]

    def rotation_weight(self, seed: str) -> float:
        name = canonicalize_seed_name(seed)
        return 1.0 / (1.0 + float(self.episodes_included.get(name, 0)))

    def choose_loadout(
        self,
        selectable_seeds: Iterable[str],
        observed_capacity: int,
        previous_loadout: Optional[Iterable[str]] = None,
        validation_seeds: Optional[Iterable[str]] = None,
        validation_source: str = "selectable",
    ) -> LoadoutDecision:
        capacity = _clamp_capacity(observed_capacity, maximum=self.max_seed_slots)
        selectable_list_raw = _canonical_seed_sequence(selectable_seeds)
        selectable_list = _ordered_by_priority(selectable_list_raw)
        selectable_set = set(selectable_list)
        if not selectable_set:
            fallback = _canonical_seed_sequence(previous_loadout or self.initial_loadout)
            selectable_list = _ordered_by_priority(fallback)
            selectable_set = set(selectable_list)

        validation_source_text = str(validation_source or "selectable").strip() or "selectable"
        validation_list = (
            _ordered_by_priority(_canonical_seed_sequence(validation_seeds))
            if validation_seeds is not None
            else list(selectable_list)
        )
        validation_set = set(validation_list)
        if not validation_set:
            validation_list = list(selectable_list)
            validation_set = set(validation_list)
        if not validation_set:
            fallback = _canonical_seed_sequence(previous_loadout or self.initial_loadout)
            validation_list = _ordered_by_priority(fallback)
            validation_set = set(validation_list)

        required_initial_unique = _ordered_by_priority(set(self.initial_loadout))
        missing_initial = [seed for seed in required_initial_unique if seed not in validation_set]
        if missing_initial:
            raise RuntimeError(
                f"blocked_reason={BLOCKED_INITIAL_LOADOUT_UNAVAILABLE}: missing={','.join(missing_initial)}"
            )

        eligible_all = self.eligible_seeds()
        eligible_selectable = [seed for seed in eligible_all if seed in selectable_set]
        initial_unique = set(self.initial_loadout)
        known_new = [
            seed
            for seed in self._unlock_ordered(self.unlock_episode.keys())
            if seed not in initial_unique
        ]

        exclusion_reasons: Dict[str, str] = {}
        for seed in known_new:
            episode = int(self.unlock_episode.get(seed, 0))
            if self.unlock_aware and (self.episode_index - episode) < self.unlock_introduction_delay:
                exclusion_reasons[seed] = "unlock_delay"
            elif seed not in selectable_set:
                exclusion_reasons[seed] = "not_selectable"

        new_candidates = [
            seed
            for seed in known_new
            if seed in selectable_set
            and (not self.unlock_aware or (self.episode_index - int(self.unlock_episode.get(seed, 0))) >= self.unlock_introduction_delay)
        ]

        previous = _canonical_seed_sequence(previous_loadout or [])
        explicit_order_locked = self.seed_order_source == SEED_ORDER_SOURCE_EXPLICIT and not self.randomize_seed_order
        active_guarantees = [
            state.seed
            for state in self.active_guarantees()
            if state.seed in selectable_set
        ]
        has_progressed = bool(
            self.last_committed_loadout
            or any(
                int(self.episodes_included.get(seed, 0)) > 0
                for seed in self.unlock_episode
                if seed not in initial_unique
            )
        )
        preserve_initial = (
            not active_guarantees
            and not new_candidates
            and not has_progressed
            and capacity >= len(self.initial_loadout)
        )
        guaranteed_selected: List[str] = []
        proposed_rotation_cursor = int(self.rotation_cursor)
        loadout_provenance: List[Dict[str, str]] = []
        if preserve_initial:
            if capacity < len(self.initial_loadout):
                raise RuntimeError(
                    "blocked_reason=configured_seed_list_exceeds_observed_capacity: "
                    f"requested={len(self.initial_loadout)} observed={capacity}"
                )
            selected = [seed for seed in self.initial_loadout if seed in selectable_set]
            local_reasons = {}
            reason = "explicit_config"
            loadout_provenance = self.loadout_provenance(selected)
        else:
            selected, local_reasons, reason, guaranteed_selected, proposed_rotation_cursor, loadout_provenance = self._build_component_loadout(
                capacity=capacity,
                selectable_set=selectable_set,
                eligible_selectable=eligible_selectable,
                previous_loadout=previous,
            )

        exclusion_reasons.update(local_reasons)
        selected_set = set(selected)
        excluded_new_plants: List[Dict[str, str]] = []
        for seed in known_new:
            if seed in selected_set:
                continue
            excluded_new_plants.append({"seed": seed, "reason": exclusion_reasons.get(seed, "capacity_full")})
        excluded_new_plants.sort(key=lambda row: (_priority_index(row.get("seed", "")), str(row.get("reason", ""))))
        selected_loadout = selected[:capacity]
        intentional_unlock_reorder = bool(
            (guaranteed_selected or new_candidates)
            and reason.startswith("rotation_")
        )
        blocked_reason = (
            ""
            if intentional_unlock_reorder
            else _seed_order_blocked_reason(self.initial_loadout, selected_loadout, capacity)
        )

        return LoadoutDecision(
            selected_loadout=selected_loadout,
            loadout_reason=reason,
            eligible_seeds=eligible_all,
            selectable_seeds=selectable_list,
            excluded_new_plants=excluded_new_plants,
            observed_seed_bank_capacity=capacity,
            configured_seed_list=list(self.initial_loadout),
            seed_order_source=SEED_ORDER_SOURCE_RANDOMIZED if reason.startswith("varied_") else self.seed_order_source,
            seed_order_preserved=_seed_order_preserved(self.initial_loadout, selected_loadout),
            blocked_reason=blocked_reason,
            validation_source=validation_source_text,
            validation_seeds=list(validation_list),
            guaranteed_seeds=list(guaranteed_selected),
            proposed_rotation_cursor=int(proposed_rotation_cursor),
            loadout_provenance=list(loadout_provenance),
        )

    def _unlock_ordered(self, seeds: Iterable[str]) -> List[str]:
        requested = set(_canonical_seed_sequence(seeds))
        ordered: List[str] = []
        seen: set[str] = set()
        for seed in self.unlock_order:
            if seed in requested and seed not in seen:
                ordered.append(seed)
                seen.add(seed)
        for seed in _ordered_by_priority(requested):
            if seed not in seen:
                ordered.append(seed)
                seen.add(seed)
        return ordered

    def _build_component_loadout(
        self,
        *,
        capacity: int,
        selectable_set: set[str],
        eligible_selectable: List[str],
        previous_loadout: List[str],
    ) -> Tuple[List[str], Dict[str, str], str, List[str], int, List[Dict[str, str]]]:
        """Build CORE -> NEW_UNLOCK_GUARANTEE -> ROTATION without side effects."""

        del previous_loadout  # The committed counters, not stale slot order, drive rotation.
        exclusion_reasons: Dict[str, str] = {}
        selected: List[str] = []
        sources: List[str] = []
        guaranteed_selected: List[str] = []
        core = [seed for seed in self.core_seed_names if seed in selectable_set]

        for seed in core:
            if len(selected) >= capacity:
                break
            selected.append(seed)
            sources.append("core")

        active_states = self.active_guarantees()
        for state in active_states:
            seed = state.seed
            if seed not in selectable_set:
                exclusion_reasons[seed] = "not_selectable"
                continue
            if seed in selected:
                guaranteed_selected.append(seed)
                continue
            if len(selected) >= capacity:
                exclusion_reasons[seed] = "guarantee_capacity_deferred"
                continue
            selected.append(seed)
            sources.append("guaranteed_unlock")
            guaranteed_selected.append(seed)

        rotation_candidates = [
            seed
            for seed in self._unlock_ordered(eligible_selectable)
            if seed in selectable_set and seed not in set(selected)
        ]
        rotation_order = self._exposure_balanced_order(rotation_candidates)
        inserted_rotation: List[str] = []
        for seed in rotation_order:
            if len(selected) >= capacity:
                break
            selected.append(seed)
            sources.append("rotation")
            inserted_rotation.append(seed)

        # Duplicate starter identities remain supported as explicit rotation
        # fillers after every unique authorized candidate has had a chance.
        duplicate_fill = [seed for seed in self.initial_loadout if seed in selectable_set]
        if not duplicate_fill:
            duplicate_fill = list(core)
        fill_index = 0
        while len(selected) < capacity and duplicate_fill:
            selected.append(duplicate_fill[fill_index % len(duplicate_fill)])
            sources.append("rotation")
            fill_index += 1

        for state in active_states:
            if state.seed not in guaranteed_selected and state.seed in selectable_set:
                exclusion_reasons.setdefault(state.seed, "guarantee_capacity_deferred")

        if guaranteed_selected:
            reason = "rotation_guaranteed_unlock"
        elif inserted_rotation:
            reason = "rotation_exposure_balanced"
        elif active_states:
            reason = "rotation_guarantee_deferred"
        else:
            reason = "rotation_stable"

        proposed_cursor = int(self.rotation_cursor)
        if rotation_candidates:
            proposed_cursor = (proposed_cursor + 1) % len(rotation_candidates)
        provenance = [
            {"slot_index": int(index), "seed": seed, "source": sources[index]}
            for index, seed in enumerate(selected)
        ]
        return selected[:capacity], exclusion_reasons, reason, guaranteed_selected, proposed_cursor, provenance

    def _exposure_balanced_order(self, candidates: Iterable[str]) -> List[str]:
        unique = self._unlock_ordered(candidates)
        if not unique:
            return []
        unlock_index = {seed: index for index, seed in enumerate(self.unlock_order)}
        priority_index = {seed: _priority_index(seed) for seed in unique}
        ordered = sorted(
            unique,
            key=lambda seed: (
                int(self.episodes_included.get(seed, 0)),
                unlock_index.get(seed, len(unlock_index) + priority_index.get(seed, 0)),
                priority_index.get(seed, 0),
                seed,
            ),
        )
        # Rotate only within equal-exposure groups so the cursor cannot make a
        # well-exposed seed outrank a seed that has never been shown.
        output: List[str] = []
        cursor = int(self.rotation_cursor)
        index = 0
        while index < len(ordered):
            count = int(self.episodes_included.get(ordered[index], 0))
            end = index + 1
            while end < len(ordered) and int(self.episodes_included.get(ordered[end], 0)) == count:
                end += 1
            group = ordered[index:end]
            if len(group) > 1:
                offset = cursor % len(group)
                group = group[offset:] + group[:offset]
            output.extend(group)
            index = end
        return output

    def commit_loadout(
        self,
        decision: LoadoutDecision,
        *,
        selected_loadout: Optional[Iterable[str]] = None,
        episode_index: Optional[int] = None,
    ) -> None:
        """Commit exposure and guarantee progress after canonical UI success."""

        committed = _canonical_seed_sequence(
            decision.selected_loadout if selected_loadout is None else selected_loadout
        )
        expected = _canonical_seed_sequence(decision.selected_loadout)
        if committed != expected:
            raise RuntimeError(
                "curriculum_commit_loadout_mismatch: "
                f"expected={expected} committed={committed}"
            )
        if episode_index is not None:
            self.episode_index = max(self.episode_index, int(episode_index))
        for seed in _ordered_unique_seed_names(committed):
            self.episodes_included[seed] = int(self.episodes_included.get(seed, 0)) + 1
            self.last_included_episode[seed] = int(self.episode_index)
        committed_set = set(committed)
        for seed in decision.guaranteed_seeds:
            state = self.guarantees.get(seed)
            if state is None or seed not in committed_set:
                continue
            state.completed_inclusions = min(
                int(state.required_inclusions),
                int(state.completed_inclusions) + 1,
            )
        self.rotation_cursor = int(decision.proposed_rotation_cursor)
        self.last_committed_loadout = list(committed)
        self.last_committed_loadout_provenance = list(
            decision.loadout_provenance or self.loadout_provenance(committed)
        )
        self.restored_current_loadout = []
        self.restored_loadout_provenance = []

    def _build_rotating_loadout(
        self,
        *,
        capacity: int,
        selectable_set: set[str],
        eligible_selectable: List[str],
        new_candidates: List[str],
        previous_loadout: List[str],
    ) -> Tuple[List[str], Dict[str, str], str]:
        """Keep a small core while making unlocks and rotation observable.

        The configured duplicate slots are retained for checkpoint-compatible
        startup, but they are deliberately the first replacement targets once
        the live bank is full.  Only this seed-selection boundary can change
        the list; the action decoder continues to address the resulting slot
        identities for the whole episode.
        """

        exclusion_reasons: Dict[str, str] = {}
        selected = [seed for seed in previous_loadout if seed in selectable_set][:capacity]
        if not selected:
            selected = [seed for seed in self.initial_loadout if seed in selectable_set][:capacity]
        if not selected:
            selected = [seed for seed in eligible_selectable if seed in selectable_set][:capacity]

        core: List[str] = []
        for seed in self.initial_loadout:
            if seed in selectable_set and seed not in core:
                core.append(seed)

        # Keep one copy of every configured core seed whenever the current
        # screen authorizes it.  If a stale loadout omitted one, replace a
        # non-core slot before considering a core duplicate.
        for core_seed in core:
            if core_seed in selected:
                continue
            if len(selected) < capacity:
                selected.append(core_seed)
                continue
            replacement = self._replacement_index(
                selected,
                core=set(core),
                protected=set(),
            )
            if replacement >= 0:
                selected[replacement] = core_seed

        selected_set = set(selected)
        eligible_pool = [
            seed
            for seed in self._unlock_ordered(eligible_selectable)
            if seed in selectable_set and seed not in set(core)
        ]
        if not eligible_pool:
            eligible_pool = [
                seed
                for seed in _ordered_by_priority(eligible_selectable)
                if seed in selectable_set and seed not in set(core)
            ]

        newest = new_candidates[-1] if new_candidates else ""
        changed = False
        inserted: set[str] = set()
        if newest and newest not in selected_set:
            if len(selected) < capacity:
                selected.append(newest)
            else:
                replacement = self._replacement_index(
                    selected,
                    core=set(core),
                    protected=set(),
                )
                if replacement >= 0:
                    selected[replacement] = newest
                else:
                    exclusion_reasons[newest] = "rotation_pending"
            if newest in selected:
                selected_set.add(newest)
                inserted.add(newest)
                changed = True

        # Fill open capacity with unlocked runtime identities in discovery
        # order, then rotate one remaining identity on each full-bank visit.
        rotation_pool = [seed for seed in eligible_pool if seed != newest]
        if rotation_pool:
            offset = self.rotation_cursor % len(rotation_pool)
            rotation_pool = rotation_pool[offset:] + rotation_pool[:offset]
        for seed in rotation_pool:
            if seed in selected_set:
                continue
            if len(selected) < capacity:
                selected.append(seed)
                selected_set.add(seed)
                inserted.add(seed)
                changed = True
                continue
            replacement = self._replacement_index(
                selected,
                core=set(core),
                protected=set(inserted) | ({newest} if newest in selected_set else set()),
            )
            if replacement < 0:
                break
            selected[replacement] = seed
            selected_set = set(selected)
            inserted.add(seed)
            changed = True
            break

        if rotation_pool:
            self.rotation_cursor = (self.rotation_cursor + 1) % len(rotation_pool)

        # Keep the checkpoint's duplicate starter shape until there is an
        # actual unlocked identity to place, and use core duplicates only as
        # harmless fillers after every authorized unique card is represented.
        fill_order = [seed for seed in self.initial_loadout if seed in selectable_set]
        if not fill_order:
            fill_order = [seed for seed in eligible_selectable if seed in selectable_set]
        fill_index = 0
        while len(selected) < capacity and fill_order:
            selected.append(fill_order[fill_index % len(fill_order)])
            fill_index += 1

        for seed in new_candidates:
            if seed not in selected_set:
                exclusion_reasons.setdefault(seed, "rotation_pending")

        if newest in inserted:
            reason = "rotation_newest_unlock"
        elif inserted:
            reason = "rotation_unlocked_pool"
        elif changed:
            reason = "rotation_replacement"
        else:
            reason = "rotation_stable"
        return selected[:capacity], exclusion_reasons, reason

    def _replacement_index(
        self,
        selected: List[str],
        *,
        core: set[str],
        protected: set[str],
    ) -> int:
        for index in range(len(selected) - 1, -1, -1):
            seed = selected[index]
            if seed in core and seed not in protected and selected.count(seed) > 1:
                return index
        for index in range(len(selected) - 1, -1, -1):
            seed = selected[index]
            if seed not in protected and selected.count(seed) > 1:
                return index
        if selected:
            for offset in range(len(selected)):
                index = (self.rotation_cursor + offset) % len(selected)
                if selected[index] not in core and selected[index] not in protected:
                    return index
            for index in range(len(selected) - 1, -1, -1):
                if selected[index] not in protected:
                    return index
        return -1

    def _build_conservative_loadout(
        self,
        *,
        capacity: int,
        selectable_set: set[str],
        eligible_selectable: List[str],
        new_candidates: List[str],
        previous_loadout: List[str],
    ) -> Tuple[List[str], Dict[str, str], str]:
        exclusion_reasons: Dict[str, str] = {}

        base = self._starter_loadout_for_capacity(capacity, selectable_set)
        if not base and previous_loadout:
            base = [seed for seed in previous_loadout if seed in selectable_set]
        if not base:
            base = [seed for seed in eligible_selectable if seed in selectable_set]
        selected = list(base[:capacity])
        reason = "initial_unlock_wait"
        added_new = set()

        if len(selected) < capacity:
            reason = "conservative_fill_open_slots"
            for seed in new_candidates:
                if len(selected) >= capacity:
                    exclusion_reasons.setdefault(seed, "capacity_full")
                    continue
                if seed in selected:
                    continue
                selected.append(seed)
                added_new.add(seed)
            fill_order = [seed for seed in self.initial_loadout if seed in selectable_set]
            if not fill_order:
                fill_order = [seed for seed in eligible_selectable if seed in selectable_set]
            while len(selected) < capacity and fill_order:
                selected.append(fill_order[(len(selected) - len(base)) % len(fill_order)])
            if added_new:
                reason = "conservative_fill_open_slots_with_new"
            elif new_candidates:
                reason = "conservative_fill_open_slots_starter"
            else:
                reason = "conservative_starter_fill"
        elif new_candidates:
            for seed in new_candidates:
                exclusion_reasons.setdefault(seed, "capacity_full")
            reason = "initial_unlock_wait_capacity_full"

        if new_candidates and not added_new:
            for seed in new_candidates:
                exclusion_reasons.setdefault(seed, "capacity_full")
        for seed in new_candidates:
            if seed not in added_new and len(selected) >= capacity:
                exclusion_reasons.setdefault(seed, "capacity_full")
        return selected[:capacity], exclusion_reasons, reason

    def _build_explicit_append_loadout(
        self,
        *,
        capacity: int,
        selectable_set: set[str],
        new_candidates: List[str],
    ) -> Tuple[List[str], Dict[str, str], str]:
        exclusion_reasons: Dict[str, str] = {}
        selected = [seed for seed in self.initial_loadout if seed in selectable_set][:capacity]
        added_new = set()
        for seed in new_candidates:
            if len(selected) >= capacity:
                exclusion_reasons.setdefault(seed, "capacity_full")
                continue
            if seed in selected:
                continue
            selected.append(seed)
            added_new.add(seed)

        if added_new:
            reason = "explicit_config_append_new_slots"
        elif new_candidates:
            reason = "explicit_config_capacity_full"
        else:
            reason = "explicit_config"
        return selected[:capacity], exclusion_reasons, reason

    def _starter_loadout_for_capacity(self, capacity: int, selectable_set: set[str]) -> List[str]:
        base = [seed for seed in self.initial_loadout if seed in selectable_set]
        if capacity >= len(self.initial_loadout):
            return base

        unique_first: List[str] = []
        seen = set()
        for seed in self.initial_loadout:
            if seed not in selectable_set or seed in seen:
                continue
            unique_first.append(seed)
            seen.add(seed)
            if len(unique_first) >= capacity:
                return unique_first

        balanced = list(unique_first)
        for seed in self.initial_loadout:
            if len(balanced) >= capacity:
                break
            if seed in selectable_set:
                balanced.append(seed)
        return balanced

    def _build_varied_loadout(
        self,
        *,
        capacity: int,
        selectable_set: set[str],
        eligible_selectable: List[str],
        new_candidates: List[str],
        initial_unique: set[str],
    ) -> Tuple[List[str], Dict[str, str], str]:
        exclusion_reasons: Dict[str, str] = {}
        selected: List[str] = []
        added_new = set()

        for core_seed in ("SunFlower", "Peashooter"):
            if core_seed in selectable_set and core_seed in eligible_selectable and len(selected) < capacity and core_seed not in selected:
                selected.append(core_seed)

        for seed in eligible_selectable:
            if len(selected) >= capacity:
                exclusion_reasons.setdefault(seed, "capacity_full")
                continue
            if seed in selected:
                continue
            is_new_seed = seed not in initial_unique
            if is_new_seed and self.rng.random() > max(self.new_plant_min_inclusion_prob, 0.5):
                exclusion_reasons.setdefault(seed, "probability_gate")
                continue
            selected.append(seed)
            if is_new_seed:
                added_new.add(seed)

        fill_order = [seed for seed in eligible_selectable if seed in selectable_set]
        if not fill_order:
            fill_order = [seed for seed in self.initial_loadout if seed in selectable_set]
        while len(selected) < capacity and fill_order:
            selected.append(fill_order[len(selected) % len(fill_order)])

        for seed in new_candidates:
            if seed not in added_new and len(selected) >= capacity:
                exclusion_reasons.setdefault(seed, "capacity_full")
            elif seed not in added_new:
                exclusion_reasons.setdefault(seed, "probability_gate")
        reason = "varied_unlock" if added_new else "varied_starter_mix"
        return selected[:capacity], exclusion_reasons, reason


class AdventureGeneralistTrainingEnv(PvZMaskedPPOEnv):
    """MaskablePPO environment that advances Adventure progression between episodes."""

    def _progression_state_value(self) -> GeneralistProgressionState:
        state = self.__dict__.get("_progression_state")
        if isinstance(state, GeneralistProgressionState):
            return state
        start_level = max(1, int(self.__dict__.get("adventure_start_level", 1)))
        state = GeneralistProgressionState(current_level=start_level)
        self.__dict__["_progression_state"] = state
        return state

    def _progression_config_value(self) -> GeneralistProgressionConfig:
        return GeneralistProgressionConfig.normalized(
            adventure_start_level=self.__dict__.get("adventure_start_level", 1),
            max_adventure_levels=self.__dict__.get("max_adventure_levels", 1),
            max_attempts_per_level=self.__dict__.get("max_attempts_per_level", 1),
            frontier_win_streak_required=self.__dict__.get(
                "frontier_win_streak_required", 1
            ),
        )

    def _set_progression_fields(self, **changes: Any) -> None:
        self.__dict__["_progression_state"] = replace(
            self._progression_state_value(),
            **changes,
        )

    def _set_progression_latches(self, **changes: Any) -> None:
        state = self._progression_state_value()
        self.__dict__["_progression_state"] = replace(
            state,
            last_episode=replace(state.last_episode, **changes),
        )

    def _apply_progression_transition(
        self,
        transition: GeneralistProgressionTransition,
    ) -> None:
        self._apply_progression_state(transition.state)

    def _apply_progression_state(self, state: GeneralistProgressionState) -> None:
        self.__dict__["_progression_state"] = state
        self._assert_progression_projection()

    @staticmethod
    def _pre_effect_progression_state(
        state: GeneralistProgressionState,
        transition: GeneralistProgressionTransition,
    ) -> GeneralistProgressionState:
        """Project the exact mutable state legacy hooks observed after a win."""

        if not transition.is_win_like:
            return state
        provisional_streak = int(state.frontier_win_streak)
        if transition.is_frontier_mastery_attempt:
            provisional_streak += 1
        return replace(
            state,
            current_attempt=0,
            frontier_win_streak=provisional_streak,
            cleared_levels=transition.state.cleared_levels,
        )

    def _assert_progression_projection(self) -> None:
        state = self._progression_state_value()
        assert self.current_level == state.current_level
        assert self.current_attempt == state.current_attempt
        assert self.frontier_win_streak == state.frontier_win_streak
        assert tuple(self.cleared_levels) == state.cleared_levels
        assert tuple(self.frontier_mastered_levels) == state.frontier_mastered_levels
        assert self.frontier_replay_supported == state.frontier_replay_supported
        assert self.frontier_replay_blocked_reason == state.frontier_replay_blocked_reason
        assert self.frontier_mastery_ready == state.last_episode.mastery_ready
        assert self.frontier_promoted_this_episode == state.last_episode.promoted
        assert self.frontier_mastery_reset_reason == state.last_episode.reset_reason

    @property
    def current_level(self) -> int:
        return int(self._progression_state_value().current_level)

    @current_level.setter
    def current_level(self, value: int) -> None:
        self._set_progression_fields(current_level=int(value))

    @property
    def current_attempt(self) -> int:
        return int(self._progression_state_value().current_attempt)

    @current_attempt.setter
    def current_attempt(self, value: int) -> None:
        self._set_progression_fields(current_attempt=int(value))

    @property
    def frontier_win_streak(self) -> int:
        return int(self._progression_state_value().frontier_win_streak)

    @frontier_win_streak.setter
    def frontier_win_streak(self, value: int) -> None:
        self._set_progression_fields(frontier_win_streak=int(value))

    @property
    def cleared_levels(self) -> List[int]:
        return list(self._progression_state_value().cleared_levels)

    @cleared_levels.setter
    def cleared_levels(self, value: Iterable[int]) -> None:
        self._set_progression_fields(cleared_levels=tuple(int(level) for level in value))

    @property
    def frontier_mastered_levels(self) -> List[int]:
        return list(self._progression_state_value().frontier_mastered_levels)

    @frontier_mastered_levels.setter
    def frontier_mastered_levels(self, value: Iterable[int]) -> None:
        self._set_progression_fields(
            frontier_mastered_levels=tuple(int(level) for level in value)
        )

    @property
    def frontier_replay_supported(self) -> bool:
        return bool(self._progression_state_value().frontier_replay_supported)

    @frontier_replay_supported.setter
    def frontier_replay_supported(self, value: bool) -> None:
        self._set_progression_fields(frontier_replay_supported=bool(value))

    @property
    def frontier_replay_blocked_reason(self) -> str:
        return str(self._progression_state_value().frontier_replay_blocked_reason)

    @frontier_replay_blocked_reason.setter
    def frontier_replay_blocked_reason(self, value: str) -> None:
        self._set_progression_fields(frontier_replay_blocked_reason=str(value or ""))

    @property
    def frontier_mastery_ready(self) -> bool:
        return bool(self._progression_state_value().last_episode.mastery_ready)

    @frontier_mastery_ready.setter
    def frontier_mastery_ready(self, value: bool) -> None:
        self._set_progression_latches(mastery_ready=bool(value))

    @property
    def frontier_promoted_this_episode(self) -> bool:
        return bool(self._progression_state_value().last_episode.promoted)

    @frontier_promoted_this_episode.setter
    def frontier_promoted_this_episode(self, value: bool) -> None:
        self._set_progression_latches(promoted=bool(value))

    @property
    def frontier_mastery_reset_reason(self) -> str:
        return str(self._progression_state_value().last_episode.reset_reason)

    @frontier_mastery_reset_reason.setter
    def frontier_mastery_reset_reason(self, value: str) -> None:
        self._set_progression_latches(reset_reason=str(value or ""))

    def __init__(
        self,
        config: PvZSB3Config,
        *,
        run_dir: Path,
        live_status_path: Optional[Path],
        initial_loadout: List[str],
        max_adventure_levels: int,
        max_attempts_per_level: int,
        adventure_start_level: int,
        unlock_aware_seed_curriculum: bool,
        seed_curriculum: str,
        unlock_introduction_delay: int,
        new_plant_min_inclusion_prob: float,
        seed_order_source: str,
        randomize_seed_order: bool,
        infer_capacity_from_unlocks: bool,
        allow_weak_unlocked_capacity_fallback: bool,
        replay_cleared_levels: bool,
        frontier_sample_prob: float,
        recent_cleared_sample_prob: float,
        maintenance_sample_prob: float,
        frontier_win_streak_required: int,
        strict_startup_validation: bool = True,
        core_seed_names: Optional[Iterable[str]] = None,
        new_unlock_guarantee_episodes: int = DEFAULT_NEW_UNLOCK_GUARANTEE_EPISODES,
        curriculum_state_path: Optional[Path | str] = None,
    ) -> None:
        super().__init__(config)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.progress_jsonl_path = self.run_dir / "adventure_training_progress.jsonl"
        self.plant_unlocks_path = self.run_dir / "plant_unlocks.json"
        self.seed_slot_unlocks_path = self.run_dir / "seed_slot_unlocks.json"
        self.curriculum_state_path = Path(curriculum_state_path) if curriculum_state_path else self.run_dir / "curriculum_state.json"
        self.writer = LiveStatusWriter(live_status_path)
        self.curriculum = AdventureSeedCurriculum(
            initial_loadout=initial_loadout,
            max_seed_slots=SEED_CAPACITY_MAX,
            unlock_aware=unlock_aware_seed_curriculum,
            seed_curriculum=seed_curriculum,
            unlock_introduction_delay=unlock_introduction_delay,
            new_plant_min_inclusion_prob=new_plant_min_inclusion_prob,
            seed_order_source=seed_order_source,
            randomize_seed_order=randomize_seed_order,
            core_seed_names=core_seed_names,
            new_unlock_guarantee_episodes=new_unlock_guarantee_episodes,
        )
        self.curriculum_restore_status = "new"
        self.curriculum_restore_blocked_reason = ""
        if self.curriculum_state_path.is_file():
            try:
                persisted_curriculum = json.loads(self.curriculum_state_path.read_text(encoding="utf-8"))
                self.curriculum.restore_state(persisted_curriculum)
                self.curriculum_restore_status = "restored"
            except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
                self.curriculum_restore_blocked_reason = f"curriculum_state_restore_failed:{exc}"
                raise RuntimeError(f"blocked_reason={self.curriculum_restore_blocked_reason}") from exc
        self.max_seed_slots = SEED_CAPACITY_MAX
        self.max_adventure_levels = max(1, int(max_adventure_levels))
        self.max_attempts_per_level = max(1, int(max_attempts_per_level))
        self.adventure_start_level = max(1, int(adventure_start_level))
        self.frontier_win_streak_required = max(1, int(frontier_win_streak_required))
        self.__dict__["_progression_state"] = fresh_generalist_progression(
            self._progression_config_value(),
            ppo_resume=False,
        )
        self._hard_blocked_reason = ""
        self.strict_startup_validation = bool(strict_startup_validation)
        self._startup_validation_completed = False
        self.replay_cleared_levels = bool(replay_cleared_levels)
        self.sample_probs = {
            "frontier": float(frontier_sample_prob),
            "recent_cleared": float(recent_cleared_sample_prob),
            "maintenance": float(maintenance_sample_prob),
        }
        self.episode_index = int(self.curriculum.episode_index)
        self.current_sample_source = "frontier"
        self.configured_seed_list = list(self.curriculum.initial_loadout)
        self.seed_order_source = _normalize_seed_order_source(seed_order_source)
        self.randomize_seed_order = bool(randomize_seed_order)
        self.infer_capacity_from_unlocks = bool(infer_capacity_from_unlocks)
        self.allow_weak_unlocked_capacity_fallback = bool(allow_weak_unlocked_capacity_fallback)
        self.observed_seed_bank_capacity = _clamp_capacity(len(initial_loadout), maximum=self.max_seed_slots)
        self.bridge_reported_capacity: Optional[int] = None
        self.inferred_capacity_from_unlocks = self.observed_seed_bank_capacity
        self.effective_seed_capacity = self.observed_seed_bank_capacity
        self.max_effective_seed_capacity_seen = self.effective_seed_capacity
        self.inferred_capacity_source = "initial_starter_loadout"
        self.capacity_inference_reason = "initial starter loadout"
        self.available_priority_seeds: List[str] = []
        self.rejected_priority_seeds: List[Dict[str, str]] = []
        self.confirmed_unlock_event_seeds: List[str] = []
        self.current_loadout = list(initial_loadout[: self.observed_seed_bank_capacity])
        self.current_loadout_provenance = self.curriculum.loadout_provenance(self.current_loadout)
        self.current_loadout_reason = "initial"
        self.current_seed_order_source = self.seed_order_source
        self.current_seed_order_preserved = _seed_order_preserved(self.configured_seed_list, self.current_loadout)
        self.current_seed_order_blocked_reason = _seed_order_blocked_reason(
            self.configured_seed_list,
            self.current_loadout,
            self.observed_seed_bank_capacity,
        )
        self.current_selectable_seeds = _ordered_by_priority(set(self.current_loadout))
        self.current_excluded_new_plants: List[Dict[str, str]] = []
        self._pending_seed_selection: Optional[Dict[str, Any]] = None
        self._base_fusion_policy = str(self.config.fusion_policy)
        self.current_curriculum_mode = "frontier"
        self._apply_loadout(self.current_loadout)
        self._episode_slot_identity = tuple(int(value) for value in self.config.plant_types)
        self.context: Dict[str, Any] = {
            "mode": ADVENTURE_GENERALIST_RUN_MODE_TRAIN,
            "run_mode": ADVENTURE_GENERALIST_RUN_MODE_TRAIN,
            "status": "starting",
            "state": "STARTING",
            "active_run": str(self.run_dir),
            "current_stage": "adventure_generalist_14slot_identity_v1",
            "current_model_family": ADVENTURE_GENERALIST_MODEL_FAMILY,
            "current_model_path": "",
            "configured_seed_list": list(self.configured_seed_list),
            "selected_seeds": list(self.current_loadout),
            "selected_loadout": list(self.current_loadout),
            "selected_loadout_count": len(self.current_loadout),
            "episode_slot_identity": list(self._episode_slot_identity),
            "seed_order_source": self.current_seed_order_source,
            "seed_order_preserved": bool(self.current_seed_order_preserved),
            "seed_order_blocked_reason": self.current_seed_order_blocked_reason,
            "randomize_seed_order": bool(self.randomize_seed_order),
            "unlocked_seeds": self.curriculum.unlocked_seeds(),
            "eligible_seeds": self.curriculum.eligible_seeds(),
            "selectable_seeds": list(self.current_selectable_seeds),
            "active_seed_slot_count": len(self.current_loadout),
            "inactive_seed_slot_count": max(0, self.max_seed_slots - len(self.current_loadout)),
            "max_seed_slots": self.max_seed_slots,
            **self._capacity_context_fields(),
            "inactive_model_slots": max(0, self.max_seed_slots - len(self.current_loadout)),
            "loadout_reason": self.current_loadout_reason,
            "loadout_provenance": list(self.current_loadout_provenance),
            "per_seed_diagnostics": self.curriculum.per_seed_diagnostics(
                selected_loadout=self.current_loadout
            ),
            "curriculum_state_path": str(self.curriculum_state_path),
            "curriculum_restore_status": self.curriculum_restore_status,
            "curriculum_restore_blocked_reason": self.curriculum_restore_blocked_reason,
            "excluded_new_plants": list(self.current_excluded_new_plants),
            "current_level": self.current_level,
            "current_attempt": 0,
            "max_adventure_levels": self.max_adventure_levels,
            "max_attempts_per_level": self.max_attempts_per_level,
            "frontier_level": self.current_level,
            "cleared_levels": [],
            "episode_sample_source": self.current_sample_source,
            "requested_episode_sample_source": self.current_sample_source,
            "level_replay_supported": False,
            "level_replay_blocked_reason": "cleared_level_replay_requires_save_state_router",
            "frontier_win_streak": self.frontier_win_streak,
            "frontier_win_streak_required": self.frontier_win_streak_required,
            "wins_on_current_level": self.frontier_win_streak,
            "wins_before_advance": self.frontier_win_streak_required,
            "frontier_mastery_ready": False,
            "frontier_promoted_this_episode": False,
            "frontier_mastery_reset_reason": "",
            "mastery_sample_source": self.current_sample_source,
            "frontier_replay_supported": self.frontier_replay_supported,
            "frontier_replay_blocked_reason": self.frontier_replay_blocked_reason,
            "frontier_mastered_levels": list(self.frontier_mastered_levels),
            "post_win_decision": "",
            "post_win_transition_allowed": False,
            "adventure_generalist_strict_startup_validation": bool(self.strict_startup_validation),
            "startup_validation_ok": None,
            "startup_validation_reason": "",
            "level_identity_reliable": None,
            "unlock_aware_seed_curriculum": bool(unlock_aware_seed_curriculum),
            "seed_curriculum": seed_curriculum,
            "core_seed_names": list(self.curriculum.core_seed_names),
            "new_unlock_guarantee_episodes": int(self.curriculum.new_unlock_guarantee_episodes),
            "post_win_transition": {},
            "post_win_blocked_reason": "",
            "infer_capacity_from_unlocks": bool(self.infer_capacity_from_unlocks),
            "allow_weak_unlocked_capacity_fallback": bool(self.allow_weak_unlocked_capacity_fallback),
            "confirmed_unlock_event_seeds": list(self.confirmed_unlock_event_seeds),
        }
        if self.frontier_win_streak_required > 1 and not callable(replay_current_level_after_validation_win):
            self._hard_blocked_reason = BLOCKED_FRONTIER_REPLAY_REQUIRED
            self.frontier_replay_supported = False
            self.frontier_replay_blocked_reason = BLOCKED_FRONTIER_REPLAY_REQUIRED
            self.context.update(
                {
                    "status": "blocked",
                    "state": "BLOCKED_FRONTIER_REPLAY",
                    "blocked_reason": BLOCKED_FRONTIER_REPLAY_REQUIRED,
                    "frontier_replay_supported": False,
                    "frontier_replay_blocked_reason": BLOCKED_FRONTIER_REPLAY_REQUIRED,
                }
            )
        self._persist_curriculum_state()

    def persist_curriculum_state(self) -> None:
        """Atomically persist curriculum and per-seed usage outside PPO tensors."""

        curriculum = self.curriculum
        rng_state = curriculum.rng.getstate()
        persisted_loadout = list(curriculum.last_committed_loadout)
        persisted_provenance = list(curriculum.last_committed_loadout_provenance)
        if not persisted_loadout:
            persisted_loadout = list(
                getattr(curriculum, "restored_current_loadout", [])
                or self.current_loadout
            )
            persisted_provenance = list(
                getattr(curriculum, "restored_loadout_provenance", [])
                or getattr(self, "current_loadout_provenance", [])
            )
        payload = {
            "schema_version": CURRICULUM_STATE_SCHEMA_VERSION,
            "model_family": ADVENTURE_GENERALIST_MODEL_FAMILY,
            "updated_at": time.time(),
            "episode_index": max(
                int(self.episode_index),
                int(curriculum.episode_index),
            ),
            "unlock_order": list(curriculum.unlock_order),
            "unlock_episode": dict(sorted(curriculum.unlock_episode.items())),
            "confirmed_selectable_history": list(curriculum.confirmed_selectable_history),
            "guarantees": {
                name: {
                    "seed": state.seed,
                    "required_inclusions": int(state.required_inclusions),
                    "completed_inclusions": int(state.completed_inclusions),
                    "remaining": int(state.remaining),
                    "pending": bool(state.pending),
                }
                for name, state in sorted(curriculum.guarantees.items())
            },
            "episodes_eligible": dict(sorted(curriculum.episodes_eligible.items())),
            "episodes_included": dict(sorted(curriculum.episodes_included.items())),
            "model_actions_executed": dict(sorted(curriculum.model_actions_executed.items())),
            "viewer_actions_executed": dict(sorted(curriculum.viewer_actions_executed.items())),
            "bc_demonstrations_recorded": dict(sorted(curriculum.bc_demonstrations_recorded.items())),
            "last_included_episode": dict(sorted(curriculum.last_included_episode.items())),
            "last_eligible_episode": dict(sorted(curriculum.last_eligible_episode.items())),
            "rotation_cursor": int(curriculum.rotation_cursor),
            "rng_state": [
                rng_state[0],
                list(rng_state[1]),
                rng_state[2],
            ],
            "last_committed_loadout": list(curriculum.last_committed_loadout),
            "current_loadout": persisted_loadout,
            "loadout_provenance": persisted_provenance,
            "per_seed_diagnostics": curriculum.per_seed_diagnostics(
                selected_loadout=self.current_loadout
            ),
        }
        path_value = getattr(self, "curriculum_state_path", None)
        if path_value is None:
            run_dir = getattr(self, "run_dir", None)
            if run_dir is None:
                # Keep bridge-free legacy test doubles and recovery probes
                # usable; real environments always set this path in __init__.
                return
            path_value = Path(run_dir) / "curriculum_state.json"
            self.curriculum_state_path = path_value
        path = Path(path_value)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(str(temporary), str(path))
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def _persist_curriculum_state(self) -> None:
        self.persist_curriculum_state()

    def _raise_if_hard_blocked(self) -> None:
        blocked_reason = str(getattr(self, "_hard_blocked_reason", "") or "")
        if not blocked_reason:
            return
        self.context.update(
            {
                "status": "blocked",
                "state": "BLOCKED",
                "blocked_reason": blocked_reason,
                "frontier_replay_supported": False,
                "frontier_replay_blocked_reason": blocked_reason,
            }
        )
        self.writer.write(build_live_status(self, self.context, adventure_state=self._safe_adventure_state()))
        raise RuntimeError(f"blocked_reason={blocked_reason}")

    def _capacity_context_fields(self) -> Dict[str, Any]:
        observed = _clamp_capacity(
            int(getattr(self, "observed_seed_bank_capacity", len(getattr(self, "current_loadout", []))) or 1),
            maximum=self.max_seed_slots,
        )
        inferred = _clamp_capacity(
            int(getattr(self, "inferred_capacity_from_unlocks", observed) or observed),
            maximum=self.max_seed_slots,
        )
        effective = _clamp_capacity(
            int(getattr(self, "effective_seed_capacity", observed) or observed),
            maximum=self.max_seed_slots,
        )
        max_seen = _clamp_capacity(
            int(getattr(self, "max_effective_seed_capacity_seen", effective) or effective),
            maximum=self.max_seed_slots,
        )
        return {
            "observed_capacity": observed,
            "observed_seed_bank_capacity": observed,
            "active_seed_slot_capacity": observed,
            "current_seed_bank_capacity": observed,
            "bridge_reported_capacity": getattr(self, "bridge_reported_capacity", None),
            "inferred_capacity_from_unlocks": inferred,
            "effective_seed_capacity": effective,
            "max_effective_seed_capacity_seen": max_seen,
            "inferred_capacity_source": str(getattr(self, "inferred_capacity_source", "") or ""),
            "capacity_inference_reason": str(getattr(self, "capacity_inference_reason", "") or ""),
            "available_priority_seeds": list(getattr(self, "available_priority_seeds", []) or []),
            "rejected_priority_seeds": list(getattr(self, "rejected_priority_seeds", []) or []),
            "confirmed_unlock_event_seeds": list(getattr(self, "confirmed_unlock_event_seeds", []) or []),
        }

    def _apply_capacity_inference(self, inference: SeedCapacityInference) -> None:
        self.bridge_reported_capacity = inference.bridge_reported_capacity
        self.observed_seed_bank_capacity = _clamp_capacity(inference.observed_capacity, maximum=self.max_seed_slots)
        self.inferred_capacity_from_unlocks = _clamp_capacity(
            inference.inferred_capacity_from_unlocks,
            maximum=self.max_seed_slots,
        )
        self.max_effective_seed_capacity_seen = _clamp_capacity(
            inference.max_effective_seed_capacity_seen,
            maximum=self.max_seed_slots,
        )
        self.effective_seed_capacity = _clamp_capacity(
            inference.effective_seed_capacity,
            maximum=self.max_seed_slots,
        )
        self.inferred_capacity_source = str(inference.inferred_capacity_source or "")
        self.capacity_inference_reason = str(inference.capacity_inference_reason or "")
        self.available_priority_seeds = list(inference.available_priority_seeds)
        self.rejected_priority_seeds = list(inference.rejected_priority_seeds)

    def _record_confirmed_unlock_event_seeds(self, seeds: Iterable[str]) -> List[str]:
        candidates = _canonical_seed_sequence(seeds)
        if not candidates:
            return []
        priority_order = normalize_and_filter_priority_seeds(
            SEED_CAPACITY_INFERENCE_PRIORITY,
            _known_seed_names_from_registry() + candidates,
        )
        priority_keys = {normalize_plant_name(seed) for seed in priority_order}
        existing_keys = {normalize_plant_name(seed) for seed in getattr(self, "confirmed_unlock_event_seeds", [])}
        newly_confirmed: List[str] = []
        for seed in candidates:
            key = normalize_plant_name(seed)
            if key not in priority_keys or key in existing_keys:
                continue
            self.confirmed_unlock_event_seeds.append(seed)
            newly_confirmed.append(seed)
            existing_keys.add(key)
        self.confirmed_unlock_event_seeds = _ordered_by_priority(self.confirmed_unlock_event_seeds)
        return _ordered_by_priority(newly_confirmed)

    def validate_startup_state(
        self,
        *,
        phase: str = "pre_learn",
        timeout: Optional[float] = None,
        raise_on_failure: bool = True,
    ) -> Dict[str, Any]:
        expected_level = int(self.current_level)
        if not self.strict_startup_validation:
            result = {
                "ok": True,
                "phase": phase,
                "strict": False,
                "reason": "strict_startup_validation_disabled",
                "level_identity": adventure_level_identity_diagnostics({}, expected_level),
            }
            self._record_startup_validation(result, {})
            self._startup_validation_completed = True
            return result

        startup_timeout = max(1.0, float(timeout or getattr(self.config, "gameplay_ready_timeout", 8.0) or 8.0))
        initial_state = self._safe_adventure_state()
        initial = self._evaluate_startup_state(initial_state, expected_level, phase=phase, recovery_attempted=False)
        self._record_startup_validation(initial, initial_state)
        if initial.get("ok"):
            self._startup_validation_completed = True
            return initial

        recovery = self._recover_startup_state(
            expected_level=expected_level,
            phase=phase,
            timeout=startup_timeout,
            initial_reason=str(initial.get("reason") or ""),
        )
        if recovery.get("ok"):
            self._startup_validation_completed = True
            self.context["frontier_replay_recovery_required"] = False
            return recovery

        reason = str(recovery.get("reason") or initial.get("reason") or "startup_level_identity_unreliable")
        error = self._startup_actionable_error(reason, recovery.get("level_identity", initial.get("level_identity", {})))
        failed = {
            **recovery,
            "ok": False,
            "phase": phase,
            "strict": True,
            "reason": reason,
            "blocked_reason": BLOCKED_STARTUP_VALIDATION_FAILED,
            "actionable_error": error,
            "initial": initial,
        }
        self._record_startup_validation(failed, recovery.get("state_snapshot", initial_state))
        if raise_on_failure:
            raise RuntimeError(f"blocked_reason={BLOCKED_STARTUP_VALIDATION_FAILED}: {error}")
        return failed

    def _evaluate_startup_state(
        self,
        state: Dict[str, Any],
        expected_level: int,
        *,
        phase: str,
        recovery_attempted: bool,
        action: str = "",
    ) -> Dict[str, Any]:
        diagnostics = adventure_level_identity_diagnostics(
            state,
            expected_level,
            stable_screen_states=("seed_selection", "gameplay"),
            transitional_screen_states=tuple(LEVEL_IDENTITY_POST_WIN_STATES),
        )
        screen_state = adventure_screen_state_name(state)
        seed_selection = adventure_seed_selection_detected(state)
        gameplay_ready = adventure_gameplay_ready_detected(state)
        post_win = bool(
            screen_state in LEVEL_IDENTITY_POST_WIN_STATES
            or state.get("trophyVisible")
            or state.get("levelCompleteTrophyVisible")
            or state.get("rewardScreenVisible")
            or state.get("unlockScreenVisible")
            or state.get("newPlantUnlockedVisible")
            or state.get("blockingRewardUiActive")
        )
        navigation = bool(
            state.get("startupPopupVisible")
            or state.get("startupOkButtonVisible")
            or (
                state.get("isAdventureButtonVisible")
                and not seed_selection
                and not gameplay_ready
                and not post_win
            )
            or screen_state in {"main_menu", "adventure_menu", "loading_or_menu", "startup_popup"}
        )
        mismatches = list(diagnostics.get("level_identity_mismatches", []) or [])
        if adventure_challenge_mode_context(state):
            ok = False
            reason = "unsupported_startup_state_challenge_mode"
            recoverable = False
        elif post_win:
            ok = False
            reason = "unsupported_startup_state_stale_post_win"
            recoverable = True
        elif seed_selection or gameplay_ready:
            ok = bool(diagnostics.get("level_identity_reliable"))
            reason = "clean_seed_selection" if seed_selection and ok else "clean_gameplay" if gameplay_ready and ok else "startup_level_identity_unreliable"
            if mismatches:
                reason = "startup_level_source_mismatch"
            recoverable = not mismatches
        elif navigation:
            ok = False
            reason = "startup_navigation_requires_level_identity_check"
            recoverable = True
        else:
            ok = False
            reason = f"unsupported_startup_state:{screen_state or 'unknown'}"
            recoverable = False
        return {
            "ok": bool(ok),
            "phase": phase,
            "strict": True,
            "reason": reason,
            "recoverable": bool(recoverable),
            "recovery_attempted": bool(recovery_attempted),
            "action": action,
            "screenState": screen_state,
            "seedSelectionDetected": bool(seed_selection),
            "gameplayReadyDetected": bool(gameplay_ready),
            "level_identity": diagnostics,
            "state_snapshot": _snapshot_startup_state(state),
        }

    def _recover_startup_state(
        self,
        *,
        expected_level: int,
        phase: str,
        timeout: float,
        initial_reason: str,
    ) -> Dict[str, Any]:
        deadline = time.monotonic() + max(1.0, timeout)
        poll_seconds = max(0.05, float(getattr(self.config, "poll_seconds", 0.2) or 0.2))
        last_result: Dict[str, Any] = {
            "ok": False,
            "phase": phase,
            "strict": True,
            "reason": initial_reason or "startup_recovery_not_started",
            "recoverable": False,
            "recovery_attempted": True,
        }
        while time.monotonic() < deadline:
            state = self._safe_adventure_state()
            result = self._evaluate_startup_state(
                state,
                expected_level,
                phase=phase,
                recovery_attempted=True,
            )
            last_result = result
            self._record_startup_validation(result, state)
            if result.get("ok"):
                return result
            if not result.get("recoverable", False):
                return result

            action = ""
            try:
                if state.get("startupPopupVisible") or state.get("startupOkButtonVisible"):
                    action = "click_startup_ok_once"
                    self.context["last_ui_action"] = self.base.click_startup_ok_once()
                elif state.get("trophyVisible") or state.get("levelCompleteTrophyVisible") or state.get("postWinClickRequired"):
                    action = "click_trophy_once"
                    self.context["last_ui_action"] = self.base.click_trophy_once()
                elif (
                    state.get("rewardScreenVisible")
                    or state.get("unlockScreenVisible")
                    or state.get("newPlantUnlockedVisible")
                    or state.get("isRewardScreen")
                    or state.get("blockingRewardUiActive")
                ):
                    action = "click_reward_continue_once"
                    self.context["last_ui_action"] = self.base.click_reward_continue_once()
                elif state.get("isAdventureButtonVisible") and not adventure_seed_selection_detected(state):
                    action = "press_adventure_once"
                    self.context["last_ui_action"] = self.base.press_adventure_once()
                elif adventure_seed_selection_detected(state):
                    diagnostics = result.get("level_identity", {})
                    if diagnostics.get("level_identity_mismatches"):
                        return result
                    action = "wait_seed_selection_level_identity"
                    time.sleep(poll_seconds)
                elif adventure_gameplay_ready_detected(state):
                    action = "wait_for_gameplay_ready"
                    self.base.wait_for_gameplay_ready(
                        timeout=max(1.0, min(timeout, deadline - time.monotonic() + 1.0)),
                        poll_seconds=poll_seconds,
                        quiet=True,
                        fail_on_terminal=False,
                    )
                else:
                    action = "wait"
                    time.sleep(poll_seconds)
                    continue
            except Exception as exc:
                result = {
                    **result,
                    "ok": False,
                    "reason": f"startup_recovery_action_failed:{action}:{exc}",
                    "action": action,
                    "recoverable": False,
                }
                self._record_startup_validation(result, state)
                return result

            if action:
                self.context["startup_validation_recovery_action"] = action
            time.sleep(max(0.1, poll_seconds))

        return {
            **last_result,
            "ok": False,
            "reason": f"startup_recovery_timeout:{last_result.get('reason', 'unknown')}",
            "recovery_attempted": True,
        }

    def _record_startup_validation(self, result: Dict[str, Any], state: Dict[str, Any]) -> None:
        diagnostics = result.get("level_identity", {})
        if not isinstance(diagnostics, dict):
            diagnostics = {}
        compact = dict(result)
        compact["level_identity"] = diagnostics
        compact["state_snapshot"] = _snapshot_startup_state(state)
        self.context.update(
            {
                "status": "running" if result.get("ok") else "validating",
                "state": "STARTUP_VALIDATION_OK" if result.get("ok") else "STARTUP_VALIDATION",
                "startup_validation": compact,
                "startup_validation_ok": result.get("ok"),
                "startup_validation_reason": result.get("reason", ""),
                "level_identity": diagnostics,
                "wrapper_expected_level": diagnostics.get("wrapper_expected_level"),
                "bridge_detected_level": diagnostics.get("bridge_detected_level"),
                "profile_adventure_level": diagnostics.get("profile_adventure_level"),
                "profile_adventure_level_source": diagnostics.get("profile_adventure_level_source", ""),
                "ui_world_level_text": diagnostics.get("ui_world_level_text", ""),
                "level_identity_reliable": diagnostics.get("level_identity_reliable"),
                "screenState": diagnostics.get("screenState"),
                "seedSelectionDetected": diagnostics.get("seedSelectionDetected"),
                "gameplayReadyDetected": diagnostics.get("gameplayReadyDetected"),
            }
        )
        if result.get("blocked_reason"):
            self.context.update(
                {
                    "status": "blocked",
                    "state": "STARTUP_VALIDATION_FAILED",
                    "blocked_reason": result.get("blocked_reason"),
                }
            )
        print(
            "[adventure-generalist] startup_validation "
            f"phase={result.get('phase', '')} "
            f"ok={'true' if result.get('ok') else 'false'} "
            f"reason={result.get('reason', '')} "
            f"wrapper_expected_level={diagnostics.get('wrapper_expected_level') or 'unknown'} "
            f"bridge_detected_level={diagnostics.get('bridge_detected_level') or 'unknown'} "
            f"profile_adventure_level={diagnostics.get('profile_adventure_level') or 'unknown'} "
            f"ui_world_level_text={diagnostics.get('ui_world_level_text') or 'unknown'} "
            f"screenState={diagnostics.get('screenState') or 'unknown'} "
            f"seedSelectionDetected={'true' if diagnostics.get('seedSelectionDetected') else 'false'} "
            f"gameplayReadyDetected={'true' if diagnostics.get('gameplayReadyDetected') else 'false'} "
            f"level_identity_reliable={'true' if diagnostics.get('level_identity_reliable') else 'false'}"
        )
        try:
            self.writer.write(build_live_status(self, self.context, adventure_state=state))
        except Exception as exc:
            self.context["startup_validation_live_status_error"] = str(exc)

    def _startup_actionable_error(self, reason: str, diagnostics: Any) -> str:
        diag = diagnostics if isinstance(diagnostics, dict) else {}
        expected = diag.get("wrapper_expected_level", self.current_level)
        detected = diag.get("bridge_detected_level", "unknown")
        profile = diag.get("profile_adventure_level", "unknown")
        screen_state = diag.get("screenState", "unknown")
        mode = diag.get("currentMode", "unknown")
        return (
            f"{reason}; expected_level={expected}; bridge_detected_level={detected}; "
            f"profile_adventure_level={profile}; screenState={screen_state}; currentMode={mode}. "
            "Return the game to the main menu, select the intended profile, enter Adventure through the normal Adventure button, "
            "and make sure the seed-selection or gameplay screen is for the expected Adventure level before starting training."
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):  # type: ignore[override]
        self._raise_if_hard_blocked()
        needs_startup_validation = bool(
            self.strict_startup_validation
            and (
                not getattr(self, "_startup_validation_completed", False)
                or self._progression_state_value().frontier_replay_recovery_required
            )
        )
        if needs_startup_validation:
            validation_phase = (
                "same_level_replay_recovery"
                if self._progression_state_value().frontier_replay_recovery_required
                else "first_reset"
            )
            self.validate_startup_state(phase=validation_phase, raise_on_failure=True)
            self._set_progression_fields(frontier_replay_recovery_required=False)
            self.context["frontier_replay_recovery_required"] = False
        self.curriculum.episode_index = self.episode_index
        adventure_state = self._safe_adventure_state()
        self.current_curriculum_mode = self._sample_curriculum_mode()
        fusion_assisted = self.current_curriculum_mode in {"fusion_chain", "coach_fusion"}
        self.config.fusion_policy = FUSION_POLICY_SCRIPTED if fusion_assisted else self._base_fusion_policy
        self.base.config.fusion_policy = self.config.fusion_policy
        self.current_sample_source = self._sample_source()
        self.__dict__["_progression_state"] = begin_generalist_attempt(
            self._progression_state_value()
        )
        self._assert_progression_projection()
        self.context.update(
            {
                "status": "running",
                "state": "PREPARE_GAMEPLAY",
                "blocked_reason": "",
                "current_level": self.current_level,
                "frontier_level": self.current_level,
                "current_attempt": self.current_attempt,
                "configured_seed_list": list(self.configured_seed_list),
                "selected_seeds": list(self.current_loadout),
                "selected_loadout": list(self.current_loadout),
                "selected_loadout_count": len(self.current_loadout),
                "loadout_provenance": list(getattr(self, "current_loadout_provenance", [])),
                "per_seed_diagnostics": self.curriculum.per_seed_diagnostics(
                    selected_loadout=self.current_loadout
                ),
                "seed_order_source": self.current_seed_order_source,
                "seed_order_preserved": bool(self.current_seed_order_preserved),
                "seed_order_blocked_reason": self.current_seed_order_blocked_reason,
                "randomize_seed_order": bool(self.randomize_seed_order),
                "active_seed_slot_count": len(self.current_loadout),
                "inactive_seed_slot_count": max(0, self.max_seed_slots - len(self.current_loadout)),
                **self._capacity_context_fields(),
                "inactive_model_slots": max(0, self.max_seed_slots - len(self.current_loadout)),
                "unlocked_seeds": self.curriculum.unlocked_seeds(),
                "eligible_seeds": self.curriculum.eligible_seeds(),
                "selectable_seeds": list(self.current_selectable_seeds),
                "episode_sample_source": self.current_sample_source,
                "curriculum_mode": self.current_curriculum_mode,
                "mastery_sample_source": self.current_sample_source,
                "loadout_reason": self.current_loadout_reason,
                "excluded_new_plants": list(self.current_excluded_new_plants),
                "cleared_levels": list(self.cleared_levels),
                "frontier_win_streak": self.frontier_win_streak,
                "frontier_win_streak_required": self.frontier_win_streak_required,
                "wins_on_current_level": self.frontier_win_streak,
                "wins_before_advance": self.frontier_win_streak_required,
                "frontier_mastery_ready": False,
                "frontier_promoted_this_episode": False,
                "frontier_mastery_reset_reason": self.frontier_mastery_reset_reason,
                "frontier_replay_supported": self.frontier_replay_supported,
                "frontier_replay_blocked_reason": self.frontier_replay_blocked_reason,
                "frontier_mastered_levels": list(self.frontier_mastered_levels),
                "post_win_decision": "",
                "post_win_transition_allowed": False,
            }
        )
        self.writer.write(build_live_status(self, self.context, adventure_state=adventure_state))
        observation, reset_info, blocked_reason = prepare_adventure_gameplay(
            self,
            self.writer,
            self.context,
            list(self.current_loadout),
            timeout=self.config.gameplay_ready_timeout,
            seed_selection_callback=self._on_seed_selection_screen,
        )
        if observation is None:
            self.context["blocked_reason"] = blocked_reason or "adventure_generalist_prepare_failed"
            self.writer.write(build_live_status(self, self.context, adventure_state=adventure_state))
            raise RuntimeError(f"blocked_reason={self.context['blocked_reason']}")
        self._episode_slot_identity = tuple(int(value) for value in self.config.plant_types)
        self.context["episode_slot_identity"] = list(self._episode_slot_identity)
        return self.start_episode_from_observation(observation, reset_info)

    def step(self, action: int):  # type: ignore[override]
        self._assert_episode_slot_identity()
        encoded, reward, terminated, truncated, info = super().step(action)
        self._record_seed_usage_from_step(info)
        self.context.update(
            {
                "status": "running",
                "state": "TRAINING_STEP",
                "last_reward": float(reward),
                "episode_index": int(self.episode_index),
                "latest_terminal_result": "",
                "frontier_win_streak": int(self.frontier_win_streak),
                "frontier_win_streak_required": int(self.frontier_win_streak_required),
                "wins_on_current_level": int(self.frontier_win_streak),
                "wins_before_advance": int(self.frontier_win_streak_required),
                "frontier_mastery_ready": bool(self.frontier_mastery_ready),
                "frontier_promoted_this_episode": bool(self.frontier_promoted_this_episode),
            }
        )
        raw_observation = info.get("raw_observation") if isinstance(info, dict) else None
        if hasattr(self.writer, "write_lazy"):
            self.writer.write_lazy(
                lambda: build_live_status(self, self.context, last_info=info),
                significant_state=live_status_significant_state(self.context, raw_observation, info),
                force=bool(terminated or truncated),
            )
        else:  # Compatibility for lightweight test/report writers.
            self.writer.write(build_live_status(self, self.context, last_info=info))
        if terminated or truncated:
            self._finish_episode(info)
        return encoded, reward, terminated, truncated, info

    def _seed_name_for_action_info(
        self,
        info: Dict[str, Any],
        *,
        slot_override: Optional[int] = None,
    ) -> str:
        action_result = info.get("action_result") if isinstance(info, dict) else None
        action_result = action_result if isinstance(action_result, dict) else {}
        decision = action_result.get("actionDecision")
        decision = decision if isinstance(decision, dict) else {}
        intent = decision.get("intent")
        intent = intent if isinstance(intent, dict) else {}
        slot_value = slot_override
        if slot_value is None:
            slot_value = intent.get("seed_slot")
        try:
            slot_index = int(slot_value) if slot_value is not None else -1
        except (TypeError, ValueError):
            slot_index = -1
        if 0 <= slot_index < len(self.current_loadout):
            return canonicalize_seed_name(self.current_loadout[slot_index])
        selected_type = decision.get("selected_plant_type", -1)
        if selected_type in (None, -1):
            decoded = action_result.get("decoded")
            if isinstance(decoded, dict):
                selected_type = decoded.get("plantType", decoded.get("ingredientPlantType", -1))
        try:
            name = canonicalize_seed_name(plant_type_name(int(selected_type)))
        except (TypeError, ValueError):
            return ""
        return name if name in self.curriculum.unlock_episode else ""

    def _record_seed_usage_from_step(self, info: Dict[str, Any]) -> None:
        if not isinstance(info, dict):
            return
        source = str(info.get("action_source") or "MODEL").strip().upper()
        if source == "TWITCH":
            transition = info.get("streamer_transition")
            diagnostics = info.get("streamer_viewer_diagnostics")
            verified = bool(
                isinstance(transition, dict)
                and transition.get("execution_succeeded")
                and str(transition.get("execution_status") or "") == "executed_verified"
            )
            if not verified and isinstance(diagnostics, dict):
                verified = str(diagnostics.get("execution_status") or "") == "executed_verified"
            last_action = getattr(self, "_streamer_v1_last_action", {}) or {}
            slot_override = last_action.get("requested_slot") if isinstance(last_action, dict) else None
            if verified:
                seed = self._seed_name_for_action_info(info, slot_override=slot_override)
                self.curriculum.record_action_usage(seed, source="viewer")
                self.context["per_seed_diagnostics"] = self.curriculum.per_seed_diagnostics(
                    selected_loadout=self.current_loadout
                )
                try:
                    self.persist_curriculum_state()
                except Exception as exc:
                    self.context["curriculum_persistence_error"] = str(exc)
            return
        seed = self._seed_name_for_action_info(info)
        self.curriculum.record_action_usage(seed, source="model")
        self.context["per_seed_diagnostics"] = self.curriculum.per_seed_diagnostics(
            selected_loadout=self.current_loadout
        )

    def record_streamer_bc_result(
        self,
        *,
        action_id: int,
        observation_revision: str,
        recorded: bool,
        reject_reason: str = "",
    ) -> None:
        super().record_streamer_bc_result(
            action_id=action_id,
            observation_revision=observation_revision,
            recorded=recorded,
            reject_reason=reject_reason,
        )
        if not recorded:
            return
        last_action = getattr(self, "_streamer_v1_last_action", {}) or {}
        slot_override = last_action.get("requested_slot") if isinstance(last_action, dict) else None
        seed = self._seed_name_for_action_info(
            {"action_result": {"actionDecision": {"intent": {"seed_slot": slot_override}}}},
            slot_override=slot_override,
        )
        self.curriculum.record_bc_demonstration(seed)
        self.context["per_seed_diagnostics"] = self.curriculum.per_seed_diagnostics(
            selected_loadout=self.current_loadout
        )
        try:
            self._persist_curriculum_state()
        except Exception as exc:
            self.context["curriculum_persistence_error"] = str(exc)

    def _assert_episode_slot_identity(self) -> None:
        """Reject action-slot remapping after gameplay has started."""

        current_slot_identity = tuple(int(value) for value in self.config.plant_types)
        expected_slot_identity = tuple(getattr(self, "_episode_slot_identity", current_slot_identity))
        if current_slot_identity != expected_slot_identity:
            raise RuntimeError(
                "blocked_reason=generalist_slot_identity_changed_mid_episode: "
                f"expected={list(expected_slot_identity)} current={list(current_slot_identity)}"
            )

    def _finish_episode(self, info: Dict[str, Any]) -> None:
        summary = info.get("episode_summary", {}) if isinstance(info, dict) else {}
        result = str(summary.get("done_reason") or info.get("done_reason") or "unknown")
        progression_before = self._progression_state_value()
        episode_level = int(progression_before.current_level)
        episode_attempt = int(progression_before.current_attempt)
        newly_unlocked: List[str] = []
        post_win_blocked_reason = ""
        post_win_transition: Dict[str, Any] = {}
        post_win_last_state: Dict[str, Any] = {}
        unknown_unlock_objects: List[Dict[str, Any]] = []
        frontier_promoted_this_episode = False
        frontier_mastery_ready = False
        frontier_mastery_reset_reason = ""
        frontier_replay_blocked_reason = str(
            progression_before.frontier_replay_blocked_reason or ""
        )
        mastery_sample_source = str(self.current_sample_source or "frontier")

        def collect_allowed_post_win_transition() -> None:
            nonlocal newly_unlocked
            nonlocal post_win_blocked_reason
            nonlocal post_win_transition
            nonlocal post_win_last_state
            nonlocal unknown_unlock_objects
            nonlocal result
            unlocked_counter = Counter({seed: 1 for seed in self.curriculum.unlocked_seeds()})
            try:
                (
                    post_win_state,
                    _unlock_seen,
                    unlock_snapshot,
                    available_after,
                    unknown_objects,
                    blocked_reason,
                    transition,
                ) = collect_post_win_unlocks(
                    self,
                    self.writer,
                    self.context,
                    unlocked_counter,
                    self.current_level,
                )
                post_win_transition = transition if isinstance(transition, dict) else {}
                post_win_blocked_reason = str(blocked_reason or "")
                post_win_last_state = _snapshot_post_win_state(post_win_state)
                unknown_unlock_objects = unknown_objects if isinstance(unknown_objects, list) else []
                confirmed_event_seeds = list(unlock_snapshot.get("visibleSeedCardNames", []) or [])
                if unlock_snapshot.get("newPlantUnlockedName"):
                    confirmed_event_seeds.append(str(unlock_snapshot.get("newPlantUnlockedName")))
                confirmed_event_seeds.extend(available_after if isinstance(available_after, list) else [])
                confirmed_event_seeds.extend(_state_seed_names(post_win_state, ("visibleSeedCardNames", "availableSeedNames")))
                confirmed_event_seeds.extend(list(unlock_snapshot.get("fallbackKnownUnlocks", []) or []))
                self._record_confirmed_unlock_event_seeds(confirmed_event_seeds)
                seeds = list(confirmed_event_seeds)
                seeds.extend(available_after if isinstance(available_after, list) else [])
                seeds.extend(_available_from_state(post_win_state))
                seeds.extend(list(unlocked_counter.keys()))
                newly_unlocked = self.curriculum.record_unlocked(seeds, self.episode_index)
                if newly_unlocked:
                    self.context["latest_unlock"] = newly_unlocked[-1]
                if post_win_blocked_reason and result == "win":
                    result = "post_win_pending"
            except Exception as exc:
                post_win_blocked_reason = f"post_win_unlock_handling_failed:{exc}"
                post_win_transition = {
                    "post_win_transition_completed": False,
                    "post_win_blocked_reason": post_win_blocked_reason,
                    "exception": str(exc),
                }
                fallback_state = self._safe_adventure_state()
                post_win_last_state = _snapshot_post_win_state(fallback_state)
                unknown_unlock_objects = []
                fallback_seeds = _available_from_state(fallback_state)
                newly_unlocked = self.curriculum.record_unlocked(fallback_seeds, self.episode_index)
                if result == "win":
                    result = "post_win_pending"

        progression_config = self._progression_config_value()
        replay_succeeded: Optional[bool] = None
        replay_reason = ""
        preview_transition = reduce_generalist_episode(
            progression_before,
            GeneralistEpisodeOutcome(
                result=result,
                episode_level=episode_level,
                sample_source=mastery_sample_source,
            ),
            progression_config,
        )

        if preview_transition.is_win_like:
            pre_effect_state = self._pre_effect_progression_state(
                progression_before,
                preview_transition,
            )
            self._apply_progression_state(pre_effect_state)
            if preview_transition.is_frontier_mastery_attempt:
                preview_streak = int(pre_effect_state.frontier_win_streak)
                threshold_met = bool(
                    preview_transition.state.last_episode.mastery_ready
                )
                post_win_decision = preview_transition.post_win_decision
                self.context.update(
                    {
                        "wins_on_current_level": preview_streak,
                        "wins_before_advance": int(self.frontier_win_streak_required),
                        "frontier_win_streak": preview_streak,
                        "frontier_win_streak_required": int(self.frontier_win_streak_required),
                        "frontier_mastery_ready": threshold_met,
                        "post_win_decision": post_win_decision,
                        "post_win_transition_allowed": bool(
                            preview_transition.post_win_transition_allowed
                        ),
                    }
                )
                print(
                    "[adventure-generalist] "
                    f"win detected level={episode_level} "
                    f"wins_on_level={preview_streak}/{int(self.frontier_win_streak_required)}"
                )
                print(
                    "[adventure-generalist] "
                    f"decision={post_win_decision} threshold_met={threshold_met}"
                )
                print(
                    "[adventure-generalist] "
                    "post_win_transition_allowed="
                    f"{'true' if preview_transition.post_win_transition_allowed else 'false'}"
                )
                if preview_transition.collect_post_win_transition:
                    collect_allowed_post_win_transition()
                else:
                    post_win_transition = {
                        "win_detected": True,
                        "post_win_decision": "replay_same_level",
                        "post_win_transition_allowed": False,
                        "post_win_transition_completed": False,
                        "post_win_blocked_reason": "",
                        "trophy_clicked": False,
                        "trophy_click_count": 0,
                        "reward_continue_click_count": 0,
                    }
                    if preview_transition.same_level_replay_requested:
                        replay_ok, replay_reason = replay_current_level_after_validation_win(
                            self,
                            self.writer,
                            self.context,
                            timeout=self.config.gameplay_ready_timeout,
                            expected_level=int(episode_level),
                            seed_selection_callback=self._on_seed_selection_screen,
                        )
                        replay_succeeded = bool(replay_ok)
                        replay_state = self._safe_adventure_state()
                        replay_level = int(
                            replay_state.get("currentAdventureLevel", episode_level)
                            or episode_level
                        )
                        self.context["frontier_replay_level_after_win"] = replay_level
                        self.context["frontier_replay_last_state"] = _snapshot_post_win_state(
                            replay_state
                        )
                        if (
                            replay_succeeded
                            and replay_level != int(episode_level)
                            and _replay_level_check_authoritative(replay_state)
                        ):
                            replay_succeeded = False
                            replay_reason = (
                                "same_level_replay_advanced_to_unexpected_level:"
                                f"{replay_level}!={int(episode_level)}"
                            )
                        if not replay_succeeded:
                            frontier_replay_blocked_reason = str(
                                replay_reason or BLOCKED_FRONTIER_REPLAY_REQUIRED
                            )
                            post_win_blocked_reason = frontier_replay_blocked_reason
                            print(
                                "[adventure-generalist] "
                                "same_level_replay_recovery_required "
                                f"reason={frontier_replay_blocked_reason}"
                            )
                            self.context.update(
                                {
                                    "frontier_replay_supported": False,
                                    "frontier_replay_blocked_reason": frontier_replay_blocked_reason,
                                    "frontier_replay_recovery_required": True,
                                    "level_identity_reliable": False,
                                    "post_win_blocked_reason": frontier_replay_blocked_reason,
                                }
                            )
                        else:
                            frontier_replay_blocked_reason = ""
                            self.context["frontier_replay_recovery_required"] = False
                            print(
                                "[adventure-generalist] "
                                f"same_level_replay_ready level={int(episode_level)} "
                                f"wins_on_level={preview_streak}/{int(self.frontier_win_streak_required)}"
                            )
            else:
                self.context.update(
                    {
                        "post_win_decision": "hold_frontier",
                        "post_win_transition_allowed": False,
                        "wins_on_current_level": int(progression_before.frontier_win_streak),
                        "wins_before_advance": int(self.frontier_win_streak_required),
                    }
                )

        progression_transition = reduce_generalist_episode(
            progression_before,
            GeneralistEpisodeOutcome(
                result=result,
                episode_level=episode_level,
                sample_source=mastery_sample_source,
                same_level_replay_succeeded=replay_succeeded,
                same_level_replay_reason=replay_reason,
            ),
            progression_config,
        )
        self._apply_progression_transition(progression_transition)
        frontier_promoted_this_episode = bool(
            progression_transition.state.last_episode.promoted
        )
        frontier_mastery_ready = bool(
            progression_transition.state.last_episode.mastery_ready
        )
        frontier_mastery_reset_reason = str(
            progression_transition.state.last_episode.reset_reason or ""
        )
        frontier_replay_blocked_reason = str(
            progression_transition.state.frontier_replay_blocked_reason or ""
        )

        selected_loadout_count = len(self.current_loadout)
        unlocked_now = set(self.curriculum.unlocked_seeds())
        selectable_now = set(self.current_selectable_seeds)
        action_counts = dict(summary.get("plant_action_counts") or {})
        placement_counts = dict(summary.get("successful_placements_by_plant") or {})
        invalid_by_plant = dict(summary.get("invalid_actions_by_plant") or {})
        mask_by_slot = dict(summary.get("legal_actions_by_seed_slot") or {})
        plant_availability: List[Dict[str, Any]] = []
        for seed in SEED_PRIORITY:
            slot_indices = [index for index, selected in enumerate(self.current_loadout) if selected == seed]
            mask_valid_count = sum(int(mask_by_slot.get(str(index), mask_by_slot.get(index, 0)) or 0) for index in slot_indices)
            selected_count = int(action_counts.get(seed, 0) or 0)
            placed_count = int(placement_counts.get(seed, 0) or 0)
            if seed not in unlocked_now:
                reason = "not_unlocked_or_not_reported_by_bridge"
            elif seed not in selectable_now:
                reason = "unlocked_but_not_selectable"
            elif not slot_indices:
                reason = "selectable_but_not_in_loadout"
            elif mask_valid_count <= 0:
                reason = "selected_but_never_mask_valid"
            elif selected_count <= 0:
                reason = "valid_but_policy_ignored"
            elif placed_count <= 0:
                reason = "selected_but_never_successfully_placed"
            else:
                reason = "used_successfully"
            plant_availability.append(
                {
                    "plant": seed,
                    "unlocked": seed in unlocked_now,
                    "selectable": seed in selectable_now,
                    "included_in_selected_loadout": bool(slot_indices),
                    "represented_in_observation": bool(slot_indices),
                    "model_action_slots": slot_indices,
                    "mask_valid_action_count": mask_valid_count,
                    "policy_selected_count": selected_count,
                    "successful_placement_count": placed_count,
                    "invalid_action_count": int(invalid_by_plant.get(seed, 0) or 0),
                    "reason": reason,
                }
            )
        progress = AdventureGeneralistProgress(
            episode=self.episode_index,
            level=episode_level,
            attempt=episode_attempt,
            sample_source=self.current_sample_source,
            result=result,
            selected_loadout=list(self.current_loadout),
            loadout_provenance=list(getattr(self, "current_loadout_provenance", [])),
            configured_seed_list=list(self.configured_seed_list),
            selected_loadout_count=selected_loadout_count,
            active_seed_slot_count=selected_loadout_count,
            observed_seed_bank_capacity=self.observed_seed_bank_capacity,
            bridge_reported_capacity=self.bridge_reported_capacity,
            inferred_capacity_from_unlocks=self.inferred_capacity_from_unlocks,
            effective_seed_capacity=self.effective_seed_capacity,
            max_effective_seed_capacity_seen=self.max_effective_seed_capacity_seen,
            inferred_capacity_source=self.inferred_capacity_source,
            capacity_inference_reason=self.capacity_inference_reason,
            available_priority_seeds=list(self.available_priority_seeds),
            rejected_priority_seeds=list(self.rejected_priority_seeds),
            inactive_model_slots=max(0, self.max_seed_slots - selected_loadout_count),
            selectable_seeds=list(self.current_selectable_seeds),
            eligible_seeds=self.curriculum.eligible_seeds(),
            loadout_reason=self.current_loadout_reason,
            seed_order_source=self.current_seed_order_source,
            seed_order_preserved=bool(self.current_seed_order_preserved),
            seed_order_blocked_reason=self.current_seed_order_blocked_reason,
            excluded_new_plants=list(self.current_excluded_new_plants),
            unlocked_seeds=self.curriculum.unlocked_seeds(),
            newly_unlocked=newly_unlocked,
            post_win_transition=post_win_transition,
            post_win_blocked_reason=post_win_blocked_reason,
            post_win_last_state=post_win_last_state,
            unknown_unlock_objects=unknown_unlock_objects,
            terminal_reason=str(summary.get("terminal_reason") or info.get("terminal_reason") or ""),
            timeout_classification=str(summary.get("timeout_classification") or "none"),
            reward=float(summary.get("episode_reward") or 0.0),
            length=int(summary.get("episode_length") or 0),
            plant_availability=plant_availability,
            plant_action_counts=action_counts,
            successful_placements_by_plant=placement_counts,
            invalid_actions_by_plant=invalid_by_plant,
            action_mask_validity_by_seed_slot={str(key): int(value or 0) for key, value in mask_by_slot.items()},
            fusion_attempts_by_pair=dict(summary.get("fusion_attempts_by_pair") or {}),
            fusion_successes_by_pair=dict(summary.get("fusion_successes_by_pair") or {}),
            fusion_depth_counts=dict(summary.get("fusion_depth_counts") or {}),
            highest_fusion_tier=int(summary.get("highest_fusion_tier") or 0),
            recursive_fusion_count=int(summary.get("recursive_fusion_count") or 0),
            action_freeze_count=int(summary.get("action_freeze_count") or 0),
            reward_component_totals={
                str(key): float(value or 0.0)
                for key, value in summary.items()
                if str(key).endswith("_reward_total") or str(key).endswith("_penalty_total")
            },
            frontier_win_streak=int(self.frontier_win_streak),
            frontier_win_streak_required=int(self.frontier_win_streak_required),
            frontier_mastery_ready=bool(frontier_mastery_ready),
            frontier_promoted_this_episode=bool(frontier_promoted_this_episode),
            frontier_mastery_reset_reason=str(frontier_mastery_reset_reason or ""),
            mastery_sample_source=str(mastery_sample_source or ""),
            frontier_replay_supported=bool(self.frontier_replay_supported),
            frontier_replay_blocked_reason=str(self.frontier_replay_blocked_reason or frontier_replay_blocked_reason or ""),
            frontier_mastered_levels=list(self.frontier_mastered_levels),
            per_seed_diagnostics=self.curriculum.per_seed_diagnostics(
                selected_loadout=self.current_loadout
            ),
        )
        self._append_progress(progress)
        self.episode_index += 1
        self.curriculum.episode_index = int(self.episode_index)
        self.context.update(
            {
                "state": "EPISODE_COMPLETE",
                "last_result": result,
                "latest_terminal_result": result,
                "blocked_reason": str(self._hard_blocked_reason or post_win_blocked_reason or ""),
                "current_level": self.current_level,
                "frontier_level": self.current_level,
                "current_attempt": self.current_attempt,
                "cleared_levels": list(self.cleared_levels),
                "unlocked_seeds": self.curriculum.unlocked_seeds(),
                "eligible_seeds": self.curriculum.eligible_seeds(),
                "selectable_seeds": list(self.current_selectable_seeds),
                "newly_unlocked": newly_unlocked,
                "configured_seed_list": list(self.configured_seed_list),
                "selected_seeds": list(self.current_loadout),
                "selected_loadout": list(self.current_loadout),
                "selected_loadout_count": selected_loadout_count,
                "loadout_provenance": list(getattr(self, "current_loadout_provenance", [])),
                "per_seed_diagnostics": self.curriculum.per_seed_diagnostics(
                    selected_loadout=self.current_loadout
                ),
                "seed_order_source": self.current_seed_order_source,
                "seed_order_preserved": bool(self.current_seed_order_preserved),
                "seed_order_blocked_reason": self.current_seed_order_blocked_reason,
                "randomize_seed_order": bool(self.randomize_seed_order),
                "active_seed_slot_count": selected_loadout_count,
                "inactive_seed_slot_count": max(0, self.max_seed_slots - selected_loadout_count),
                "inactive_model_slots": max(0, self.max_seed_slots - selected_loadout_count),
                **self._capacity_context_fields(),
                "loadout_reason": self.current_loadout_reason,
                "excluded_new_plants": list(self.current_excluded_new_plants),
                "post_win_blocked_reason": post_win_blocked_reason,
                "post_win_transition": post_win_transition,
                "post_win_last_state": post_win_last_state,
                "unknown_unlock_objects": unknown_unlock_objects,
                "frontier_win_streak": int(self.frontier_win_streak),
                "frontier_win_streak_required": int(self.frontier_win_streak_required),
                "wins_on_current_level": int(self.frontier_win_streak),
                "wins_before_advance": int(self.frontier_win_streak_required),
                "frontier_mastery_ready": bool(frontier_mastery_ready),
                "frontier_promoted_this_episode": bool(frontier_promoted_this_episode),
                "frontier_mastery_reset_reason": str(frontier_mastery_reset_reason or ""),
                "mastery_sample_source": str(mastery_sample_source or self.current_sample_source or ""),
                "frontier_replay_supported": bool(self.frontier_replay_supported),
                "frontier_replay_blocked_reason": str(self.frontier_replay_blocked_reason or frontier_replay_blocked_reason or ""),
                "frontier_mastered_levels": list(self.frontier_mastered_levels),
                "post_win_decision": self.context.get("post_win_decision", ""),
                "post_win_transition_allowed": bool(self.context.get("post_win_transition_allowed", False)),
            }
        )
        self._persist_curriculum_state()
        self._write_unlock_files()
        self.writer.write(build_live_status(self, self.context, last_info=info))

    def _rollback_pending_seed_selection(self) -> None:
        self._pending_seed_selection = None
        self.context.update(
            {
                "pending_seed_selection": False,
                "proposed_selected_loadout": [],
                "proposed_loadout_reason": "",
                "proposed_loadout_provenance": [],
                "proposed_guaranteed_seeds": [],
                "proposed_rejected_priority_seeds": [],
            }
        )

    def _verify_pending_seed_selection(
        self,
        selection: Dict[str, Any],
        proposed_loadout: Iterable[str],
    ) -> bool:
        if not isinstance(selection, dict) or not bool(selection.get("ok", False)):
            self.context["seed_selection_commit_blocked_reason"] = "selection_not_ok"
            return False
        verification = selection.get("verification")
        if isinstance(verification, dict) and verification.get("success") is False:
            self.context["seed_selection_commit_blocked_reason"] = "selection_verification_failed"
            return False
        try:
            expected_types = [int(value) for value in resolve_seed_list(list(proposed_loadout))]
        except (TypeError, ValueError):
            self.context["seed_selection_commit_blocked_reason"] = "proposed_seed_list_unresolvable"
            return False

        selected_raw: Any = (
            verification.get("selectedSeedTypes")
            if isinstance(verification, dict) and "selectedSeedTypes" in verification
            else selection.get("selectedSeedTypes")
        )
        try:
            selected_types = [int(value) for value in list(selected_raw or [])]
        except (TypeError, ValueError):
            self.context["seed_selection_commit_blocked_reason"] = "selection_types_unresolvable"
            return False
        if selected_types != expected_types or Counter(selected_types) != Counter(expected_types):
            self.context["seed_selection_commit_blocked_reason"] = (
                "selection_card_multiset_or_slot_identity_mismatch"
            )
            return False

        base = getattr(self, "base", None)
        seed_probe = getattr(base, "seed_probe", None)
        if not callable(seed_probe):
            return True
        try:
            probe = seed_probe()
        except Exception as exc:
            self.context["seed_selection_commit_blocked_reason"] = f"post_selection_probe_failed:{exc}"
            return False
        if not isinstance(probe, dict):
            self.context["seed_selection_commit_blocked_reason"] = "post_selection_probe_invalid"
            return False
        active_counts, _ = active_gameplay_bank_state(probe)
        if active_counts != Counter(expected_types):
            self.context["seed_selection_commit_blocked_reason"] = (
                "post_selection_active_gameplay_bank_mismatch"
            )
            return False
        active_cards = list(probe.get("activeGameplayCardBankCards", []) or [])
        if active_cards:
            try:
                active_types = [int(card.get("plantType", -999)) for card in active_cards]
            except (AttributeError, TypeError, ValueError):
                self.context["seed_selection_commit_blocked_reason"] = "post_selection_cards_invalid"
                return False
            if active_types != expected_types:
                self.context["seed_selection_commit_blocked_reason"] = (
                    "post_selection_slot_identity_mismatch"
                )
                return False
        if bool(probe.get("seedSelectionActive")):
            self.context["seed_selection_commit_blocked_reason"] = "post_selection_seed_screen_still_active"
            return False
        return True

    def _commit_pending_seed_selection(
        self,
        selection: Dict[str, Any],
        seed_list: Iterable[str],
    ) -> bool:
        pending = self._pending_seed_selection
        if not isinstance(pending, dict):
            self.context["seed_selection_commit_blocked_reason"] = "no_pending_seed_selection"
            return False
        proposed = _canonical_seed_sequence(pending.get("proposed_loadout", []))
        committed = _canonical_seed_sequence(seed_list)
        if committed != proposed:
            self.context["seed_selection_commit_blocked_reason"] = "callback_seed_list_changed"
            return False
        if not self._verify_pending_seed_selection(selection, proposed):
            return False

        decision = pending.get("decision")
        if not isinstance(decision, LoadoutDecision):
            self.context["seed_selection_commit_blocked_reason"] = "pending_decision_invalid"
            return False

        curriculum = self.curriculum
        curriculum_snapshot = {
            "episode_index": int(curriculum.episode_index),
            "episodes_included": dict(curriculum.episodes_included),
            "last_included_episode": dict(curriculum.last_included_episode),
            "completed_inclusions": {
                name: int(state.completed_inclusions)
                for name, state in curriculum.guarantees.items()
            },
            "rotation_cursor": int(curriculum.rotation_cursor),
            "last_committed_loadout": list(curriculum.last_committed_loadout),
            "last_committed_loadout_provenance": list(
                curriculum.last_committed_loadout_provenance
            ),
            "restored_current_loadout": list(curriculum.restored_current_loadout),
            "restored_loadout_provenance": list(curriculum.restored_loadout_provenance),
        }
        config_seed_list = list(self.config.seed_list)
        config_plant_types = list(self.config.plant_types)
        base_seed_list = list(self.base.config.seed_list)
        base_plant_types = list(self.base.config.plant_types)
        previous_loadout = list(self.current_loadout)
        previous_provenance = list(getattr(self, "current_loadout_provenance", []))
        try:
            self._apply_loadout(proposed)
            curriculum.commit_loadout(
                decision,
                selected_loadout=proposed,
                episode_index=self.episode_index,
            )
            self.current_loadout = list(proposed)
            self.current_loadout_provenance = list(
                decision.loadout_provenance or curriculum.loadout_provenance(proposed)
            )
            self._persist_curriculum_state()
        except Exception as exc:
            self.config.seed_list = config_seed_list
            self.config.plant_types = config_plant_types
            self.base.config.seed_list = base_seed_list
            self.base.config.plant_types = base_plant_types
            curriculum.episode_index = curriculum_snapshot["episode_index"]
            curriculum.episodes_included = curriculum_snapshot["episodes_included"]
            curriculum.last_included_episode = curriculum_snapshot["last_included_episode"]
            curriculum.rotation_cursor = curriculum_snapshot["rotation_cursor"]
            curriculum.last_committed_loadout = curriculum_snapshot["last_committed_loadout"]
            curriculum.last_committed_loadout_provenance = curriculum_snapshot[
                "last_committed_loadout_provenance"
            ]
            curriculum.restored_current_loadout = curriculum_snapshot["restored_current_loadout"]
            curriculum.restored_loadout_provenance = curriculum_snapshot[
                "restored_loadout_provenance"
            ]
            for name, completed in curriculum_snapshot["completed_inclusions"].items():
                if name in curriculum.guarantees:
                    curriculum.guarantees[name].completed_inclusions = completed
            self.current_loadout = previous_loadout
            self.current_loadout_provenance = previous_provenance
            self.context["seed_selection_commit_blocked_reason"] = f"commit_failed:{exc}"
            return False

        self.current_loadout = list(proposed)
        self.current_loadout_provenance = list(
            decision.loadout_provenance or curriculum.loadout_provenance(proposed)
        )
        self.current_loadout_reason = str(decision.loadout_reason or "")
        self.current_seed_order_source = str(decision.seed_order_source or self.seed_order_source)
        self.current_seed_order_preserved = bool(decision.seed_order_preserved)
        self.current_seed_order_blocked_reason = str(decision.blocked_reason or "")
        self.current_selectable_seeds = list(decision.selectable_seeds)
        self.current_excluded_new_plants = list(decision.excluded_new_plants)
        self.rejected_priority_seeds = _rejected_priority_seed_diagnostics(
            list(pending.get("available_priority_seeds", [])),
            selected_loadout=self.current_loadout,
            selectable_seeds=list(pending.get("selection_candidates", decision.selectable_seeds)),
            effective_capacity=int(getattr(self, "effective_seed_capacity", len(proposed))),
            excluded_new_plants=self.current_excluded_new_plants,
        )
        self._episode_slot_identity = tuple(int(value) for value in self.config.plant_types)
        self._pending_seed_selection = None
        self.context.update(
            {
                "pending_seed_selection": False,
                "proposed_selected_loadout": [],
                "proposed_loadout_reason": "",
                "proposed_loadout_provenance": [],
                "proposed_guaranteed_seeds": [],
                "proposed_rejected_priority_seeds": [],
                "episode_slot_identity": list(self._episode_slot_identity),
                "configured_seed_list": list(self.configured_seed_list),
                "selected_seeds": list(self.current_loadout),
                "selected_loadout": list(self.current_loadout),
                "selected_loadout_count": len(self.current_loadout),
                "seed_order_source": self.current_seed_order_source,
                "seed_order_preserved": bool(self.current_seed_order_preserved),
                "seed_order_blocked_reason": self.current_seed_order_blocked_reason,
                "randomize_seed_order": bool(self.randomize_seed_order),
                "active_seed_slot_count": len(self.current_loadout),
                "inactive_seed_slot_count": max(0, self.max_seed_slots - len(self.current_loadout)),
                "inactive_model_slots": max(0, self.max_seed_slots - len(self.current_loadout)),
                **self._capacity_context_fields(),
                "raw_selectable_seeds": list(pending.get("raw_selectable", [])),
                "selectable_seeds": list(decision.selectable_seeds),
                "seed_validation_source": str(pending.get("validation_source", "selectable")),
                "seed_validation_available": list(pending.get("available_for_seed_validation", [])),
                "available_for_seed_validation": list(pending.get("available_for_seed_validation", [])),
                "eligible_seeds": list(decision.eligible_seeds),
                "unlocked_seeds": self.curriculum.unlocked_seeds(),
                "loadout_reason": self.current_loadout_reason,
                "excluded_new_plants": list(self.current_excluded_new_plants),
                "guaranteed_seeds": list(decision.guaranteed_seeds),
                "loadout_provenance": list(self.current_loadout_provenance),
                "per_seed_diagnostics": curriculum.per_seed_diagnostics(
                    selected_loadout=self.current_loadout
                ),
                "curriculum_rotation_cursor": int(self.curriculum.rotation_cursor),
            }
        )
        return True

    def _on_seed_selection_screen(self, state: Dict[str, Any], current_seed_list: List[str]) -> Tuple[List[str], str]:
        if getattr(self, "_pending_seed_selection", None) is not None:
            self._rollback_pending_seed_selection()
        selectable = _filter_supported_seed_names(_selectable_from_seed_screen_state(state))
        self._record_confirmed_unlock_event_seeds(selectable)
        capacity_inference = resolve_adventure_generalist_seed_capacity(
            state,
            context=self.context,
            previous_observed_capacity=self.observed_seed_bank_capacity,
            previous_effective_capacity=self.max_effective_seed_capacity_seen,
            selected_loadout=current_seed_list or self.current_loadout,
            eligible_seeds=self.curriculum.eligible_seeds(),
            unlocked_seeds=self.curriculum.unlocked_seeds(),
            medium_confirmed_seeds=getattr(self, "confirmed_unlock_event_seeds", []),
            infer_capacity_from_unlocks=self.infer_capacity_from_unlocks,
            allow_weak_unlocked_capacity_fallback=self.allow_weak_unlocked_capacity_fallback,
            max_seed_slots=self.max_seed_slots,
        )
        self._apply_capacity_inference(capacity_inference)
        observed_capacity = capacity_inference.observed_capacity
        effective_capacity = capacity_inference.effective_seed_capacity
        candidate_evidence = _filter_supported_seed_names(
            list(selectable) + list(capacity_inference.available_priority_seeds)
        )
        self.curriculum.record_confirmed_selectable(candidate_evidence)
        self.curriculum.record_unlocked(candidate_evidence, self.episode_index)
        self.curriculum.record_eligible(candidate_evidence, self.episode_index)
        eligible_seeds = self.curriculum.eligible_seeds()
        unlocked_seeds = self.curriculum.unlocked_seeds()
        available_for_seed_validation, validation_source = _available_for_seed_validation(
            selectable,
            eligible_seeds,
            unlocked_seeds,
        )
        if not selectable and (eligible_seeds or unlocked_seeds):
            print(
                "[adventure-generalist] selectable_empty_using_unlocked_or_eligible_fallback "
                f"validation_source={validation_source} "
                f"available_for_seed_validation={list(available_for_seed_validation)}"
            )
        # An empty selectable list is an incomplete/transitioning seed-screen
        # observation, not proof that every unlocked registry entry can be
        # clicked on this screen.  Keep the fallback for validating the
        # configured starter loadout, but never use it to expand the active
        # loadout.  Capacity/unlock evidence may extend a populated screen
        # whose current cards are visible; it must not turn an empty UI probe
        # into a request for a card that the next strict selector cannot see.
        current_seed_screen_candidates = list(selectable)
        if selectable:
            current_seed_screen_candidates.extend(capacity_inference.available_priority_seeds)
        selection_candidates = _filter_loadout_candidate_seeds(
            current_seed_screen_candidates,
            eligible_seeds=eligible_seeds,
            unlocked_seeds=unlocked_seeds,
        )
        previous_loadout = list(current_seed_list or self.current_loadout)
        selection_capacity = effective_capacity
        if not selectable:
            # Capacity inference can remain useful for diagnostics and later
            # probes, but an empty current UI cannot authorize adding even a
            # duplicate starter card to the next strict selection request.
            selection_capacity = min(effective_capacity, len(previous_loadout))
        print(
            "[adventure-generalist] capacity_input "
            f"selectable={list(selectable)} "
            f"eligible={eligible_seeds} "
            f"unlocked={unlocked_seeds} "
            f"validation_source={validation_source} "
            f"available_for_seed_validation={list(available_for_seed_validation)} "
            f"confirmed_unlock_events={list(getattr(self, 'confirmed_unlock_event_seeds', []))}"
        )
        print(
            "[adventure-generalist] seed_capacity "
            f"observed={observed_capacity} "
            f"bridge_reported={capacity_inference.bridge_reported_capacity} "
            f"inferred={capacity_inference.inferred_capacity_from_unlocks} "
            f"effective={effective_capacity} "
            f"selection_capacity={selection_capacity} "
            f"max_seen={capacity_inference.max_effective_seed_capacity_seen} "
            f"source={capacity_inference.inferred_capacity_source} "
            f"reason={capacity_inference.capacity_inference_reason}"
        )
        try:
            decision = self.curriculum.choose_loadout(
                selectable_seeds=selection_candidates,
                observed_capacity=selection_capacity,
                previous_loadout=previous_loadout,
                validation_seeds=available_for_seed_validation,
                validation_source=validation_source,
            )
        except RuntimeError as exc:
            blocked_reason = str(exc).replace("blocked_reason=", "", 1) if str(exc).startswith("blocked_reason=") else str(exc)
            fallback = _canonical_seed_sequence(current_seed_list or self.current_loadout)[:effective_capacity]
            self.current_loadout = list(fallback)
            self.current_selectable_seeds = _ordered_by_priority(selection_candidates)
            self.current_excluded_new_plants = []
            self.rejected_priority_seeds = _rejected_priority_seed_diagnostics(
                capacity_inference.available_priority_seeds,
                selected_loadout=self.current_loadout,
                selectable_seeds=selection_candidates,
                effective_capacity=effective_capacity,
                excluded_new_plants=[],
            )
            self.current_loadout_reason = "seed_selection_blocked"
            self.current_seed_order_source = self.seed_order_source
            self.current_seed_order_preserved = _seed_order_preserved(self.configured_seed_list, self.current_loadout)
            self.current_seed_order_blocked_reason = blocked_reason or _seed_order_blocked_reason(
                self.configured_seed_list,
                self.current_loadout,
                effective_capacity,
            )
            print(
                "[adventure-generalist] seed_selection "
                f"level={self.current_level} "
                f"observed_capacity={observed_capacity} "
                f"effective_capacity={effective_capacity} "
                f"configured_seed_list={list(self.configured_seed_list)} "
                f"selected_loadout={list(self.current_loadout)} "
                f"selectable={list(selectable)} "
                f"eligible={eligible_seeds} "
                f"unlocked={unlocked_seeds} "
                f"validation_source={validation_source} "
                f"seed_order_source={self.current_seed_order_source} "
                f"seed_order_preserved={bool(self.current_seed_order_preserved)} "
                f"blocked_reason={self.current_seed_order_blocked_reason}"
            )
            self.context.update(
                {
                    "raw_selectable_seeds": list(selectable),
                    "selectable_seeds": list(self.current_selectable_seeds),
                    "seed_validation_source": validation_source,
                    "seed_validation_available": list(available_for_seed_validation),
                    "available_for_seed_validation": list(available_for_seed_validation),
                    **self._capacity_context_fields(),
                    "configured_seed_list": list(self.configured_seed_list),
                    "selected_seeds": list(self.current_loadout),
                    "selected_loadout": list(self.current_loadout),
                    "selected_loadout_count": len(self.current_loadout),
                    "seed_order_source": self.current_seed_order_source,
                    "seed_order_preserved": bool(self.current_seed_order_preserved),
                    "seed_order_blocked_reason": self.current_seed_order_blocked_reason,
                    "randomize_seed_order": bool(self.randomize_seed_order),
                    "active_seed_slot_count": len(self.current_loadout),
                    "inactive_seed_slot_count": max(0, self.max_seed_slots - len(self.current_loadout)),
                    "inactive_model_slots": max(0, self.max_seed_slots - len(self.current_loadout)),
                    "loadout_reason": self.current_loadout_reason,
                    "excluded_new_plants": list(self.current_excluded_new_plants),
                    "eligible_seeds": eligible_seeds,
                    "unlocked_seeds": unlocked_seeds,
                }
            )
            return list(self.current_loadout), blocked_reason

        proposed_loadout = list(decision.selected_loadout[:effective_capacity])
        self._pending_seed_selection = {
            "decision": decision,
            "previous_loadout": list(self.current_loadout),
            "proposed_loadout": list(proposed_loadout),
            "episode_index": int(self.episode_index),
            "raw_selectable": list(selectable),
            "selection_candidates": list(selection_candidates),
            "validation_source": validation_source,
            "available_for_seed_validation": list(available_for_seed_validation),
            "available_priority_seeds": list(capacity_inference.available_priority_seeds),
        }
        self.context.update(
            {
                "pending_seed_selection": True,
                "proposed_selected_loadout": list(proposed_loadout),
                "proposed_loadout_reason": str(decision.loadout_reason or ""),
                "proposed_loadout_provenance": list(decision.loadout_provenance),
                "proposed_guaranteed_seeds": list(decision.guaranteed_seeds),
                "proposed_rotation_cursor": int(decision.proposed_rotation_cursor),
                "proposed_seed_validation_source": validation_source,
                "proposed_seed_validation_available": list(available_for_seed_validation),
                "raw_selectable_seeds": list(selectable),
                "seed_validation_source": validation_source,
                "seed_validation_available": list(available_for_seed_validation),
                "available_for_seed_validation": list(available_for_seed_validation),
                "proposed_selectable_seeds": list(decision.selectable_seeds),
                "proposed_eligible_seeds": list(decision.eligible_seeds),
                "proposed_unlocked_seeds": list(unlocked_seeds),
                **self._capacity_context_fields(),
            }
        )
        proposed_rejected_priority_seeds = _rejected_priority_seed_diagnostics(
            capacity_inference.available_priority_seeds,
            selected_loadout=proposed_loadout,
            selectable_seeds=selection_candidates,
            effective_capacity=effective_capacity,
            excluded_new_plants=decision.excluded_new_plants,
        )
        self.context["proposed_rejected_priority_seeds"] = list(proposed_rejected_priority_seeds)
        for row in proposed_rejected_priority_seeds:
            print(
                "[adventure-generalist] priority_seed_rejected "
                f"seed={row.get('seed', '')} reason={row.get('reason', '')}"
            )
        print(
            "[adventure-generalist] seed_selection "
            f"level={self.current_level} "
            f"observed_capacity={observed_capacity} "
            f"effective_capacity={effective_capacity} "
            f"configured_seed_list={list(decision.configured_seed_list or self.configured_seed_list)} "
            f"proposed_loadout={list(proposed_loadout)} "
            f"selectable={list(selectable)} "
            f"eligible={eligible_seeds} "
            f"unlocked={unlocked_seeds} "
            f"validation_source={validation_source} "
            f"seed_order_source={self.current_seed_order_source} "
            f"seed_order_preserved={bool(self.current_seed_order_preserved)} "
            f"selection_candidates={list(decision.selectable_seeds)} "
            f"loadout_reason={decision.loadout_reason} "
            f"blocked_reason={decision.blocked_reason or 'None'}"
        )
        return list(proposed_loadout), ""

    def _apply_loadout(self, loadout: List[str]) -> None:
        capacity = _clamp_capacity(getattr(self, "effective_seed_capacity", self.observed_seed_bank_capacity), maximum=self.max_seed_slots)
        effective_loadout = _canonical_seed_sequence(loadout)[:capacity]
        plant_types = resolve_seed_list(effective_loadout)
        self.config.seed_list = list(effective_loadout)
        self.config.plant_types = list(plant_types)
        self.base.config.seed_list = list(effective_loadout)
        self.base.config.plant_types = list(plant_types)

    def _sample_source(self) -> str:
        if self.frontier_win_streak_required > 1 and self.frontier_win_streak > 0:
            self.context["requested_episode_sample_source"] = "frontier_mastery_replay"
            self.context["mastery_sample_source"] = "frontier_mastery_replay"
            return "frontier_mastery_replay"
        if not self.replay_cleared_levels or not self.cleared_levels:
            self.context["requested_episode_sample_source"] = "frontier"
            self.context["mastery_sample_source"] = "frontier"
            return "frontier"
        total = sum(max(0.0, value) for value in self.sample_probs.values())
        if total <= 0:
            self.context["requested_episode_sample_source"] = "frontier"
            self.context["mastery_sample_source"] = "frontier"
            return "frontier"
        draw = random.random() * total
        running = 0.0
        requested = "frontier"
        for name, prob in self.sample_probs.items():
            running += max(0.0, prob)
            if draw <= running:
                requested = name
                break
        self.context["requested_episode_sample_source"] = requested
        self.context["mastery_sample_source"] = requested
        if requested != "frontier":
            self.context["level_replay_supported"] = False
            self.context["level_replay_blocked_reason"] = "cleared_level_replay_requires_save_state_router"
            return "frontier"
        return requested

    def _sample_curriculum_mode(self) -> str:
        weighted: List[Tuple[str, float]] = []
        if bool(getattr(self.config, "enable_fusion_curriculum", False)):
            weighted.append(("fusion_chain", max(0.0, float(getattr(self.config, "fusion_curriculum_prob", 0.20)))))
        if bool(getattr(self.config, "enable_later_plant_curriculum", False)):
            weighted.append(("later_plant", max(0.0, float(getattr(self.config, "later_plant_curriculum_prob", 0.10)))))
        if bool(getattr(self.config, "enable_coach_fusion_sampling", False)):
            weighted.append(("coach_fusion", max(0.0, float(getattr(self.config, "coach_fusion_prob", 0.10)))))
        draw = random.random()
        running = 0.0
        for name, probability in weighted:
            running += probability
            if draw < min(1.0, running):
                return name
        return "frontier"

    def _safe_adventure_state(self) -> Dict[str, Any]:
        try:
            state = self.base.adventure_screen_state()
            if isinstance(state, dict):
                return state
        except Exception as exc:
            self.context["last_adventure_state_error"] = str(exc)
        return {}

    def _append_progress(self, progress: AdventureGeneralistProgress) -> None:
        row = asdict(progress)
        with self.progress_jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        progress_path = self.run_dir / "adventure_training_progress.json"
        payload = {
            "status": "blocked" if str(self._hard_blocked_reason or "") else "running",
            "model_family": ADVENTURE_GENERALIST_MODEL_FAMILY,
            "run_mode": ADVENTURE_GENERALIST_RUN_MODE_TRAIN,
            "max_seed_slots": self.max_seed_slots,
            "observed_seed_bank_capacity": int(progress.observed_seed_bank_capacity),
            "active_seed_slot_capacity": int(progress.observed_seed_bank_capacity),
            "current_seed_bank_capacity": int(progress.observed_seed_bank_capacity),
            "observed_capacity": int(progress.observed_seed_bank_capacity),
            "bridge_reported_capacity": progress.bridge_reported_capacity,
            "inferred_capacity_from_unlocks": int(progress.inferred_capacity_from_unlocks),
            "effective_seed_capacity": int(progress.effective_seed_capacity),
            "max_effective_seed_capacity_seen": int(progress.max_effective_seed_capacity_seen),
            "inferred_capacity_source": progress.inferred_capacity_source,
            "capacity_inference_reason": progress.capacity_inference_reason,
            "available_priority_seeds": list(progress.available_priority_seeds),
            "rejected_priority_seeds": list(progress.rejected_priority_seeds),
            "selected_loadout_count": int(progress.selected_loadout_count),
            "active_seed_slot_count": int(progress.selected_loadout_count),
            "inactive_seed_slot_count": max(0, self.max_seed_slots - int(progress.selected_loadout_count)),
            "inactive_model_slots": max(0, self.max_seed_slots - int(progress.selected_loadout_count)),
            "configured_seed_list": list(progress.configured_seed_list),
            "selected_loadout": list(progress.selected_loadout),
            "loadout_reason": progress.loadout_reason,
            "seed_order_source": progress.seed_order_source,
            "seed_order_preserved": bool(progress.seed_order_preserved),
            "seed_order_blocked_reason": progress.seed_order_blocked_reason,
            "selectable_seeds": list(progress.selectable_seeds),
            "eligible_seeds": list(progress.eligible_seeds),
            "excluded_new_plants": list(progress.excluded_new_plants),
            "plant_availability": list(progress.plant_availability),
            "plant_action_counts": dict(progress.plant_action_counts),
            "successful_placements_by_plant": dict(progress.successful_placements_by_plant),
            "invalid_actions_by_plant": dict(progress.invalid_actions_by_plant),
            "action_mask_validity_by_seed_slot": dict(progress.action_mask_validity_by_seed_slot),
            "fusion_attempts_by_pair": dict(progress.fusion_attempts_by_pair),
            "fusion_successes_by_pair": dict(progress.fusion_successes_by_pair),
            "fusion_depth_counts": dict(progress.fusion_depth_counts),
            "highest_fusion_tier": int(progress.highest_fusion_tier),
            "recursive_fusion_count": int(progress.recursive_fusion_count),
            "action_freeze_count": int(progress.action_freeze_count),
            "reward_component_totals": dict(progress.reward_component_totals),
            "post_win_transition": dict(progress.post_win_transition),
            "post_win_blocked_reason": progress.post_win_blocked_reason,
            "post_win_last_state": dict(progress.post_win_last_state),
            "frontier_win_streak": int(progress.frontier_win_streak),
            "frontier_win_streak_required": int(progress.frontier_win_streak_required),
            "frontier_mastery_ready": bool(progress.frontier_mastery_ready),
            "frontier_promoted_this_episode": bool(progress.frontier_promoted_this_episode),
            "frontier_mastery_reset_reason": str(progress.frontier_mastery_reset_reason or ""),
            "mastery_sample_source": str(progress.mastery_sample_source or ""),
            "frontier_replay_supported": bool(progress.frontier_replay_supported),
            "frontier_replay_blocked_reason": str(progress.frontier_replay_blocked_reason or ""),
            "current_level": self.current_level,
            "frontier_level": self.current_level,
            "cleared_levels": list(self.cleared_levels),
            "frontier_mastered_levels": list(progress.frontier_mastered_levels),
            "latest": row,
        }
        progress_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_unlock_files(self) -> None:
        selected_count = len(self.current_loadout)
        curriculum_state_path = getattr(
            self,
            "curriculum_state_path",
            Path(getattr(self, "run_dir", Path("."))) / "curriculum_state.json",
        )
        unlock_payload = {
            "updated_at": time.time(),
            "unlocked_seeds": self.curriculum.unlocked_seeds(),
            "eligible_seeds": self.curriculum.eligible_seeds(),
            "selectable_seeds": list(self.current_selectable_seeds),
            "unlock_episode": dict(sorted(self.curriculum.unlock_episode.items())),
            "loadout_reason": self.current_loadout_reason,
            "configured_seed_list": list(self.configured_seed_list),
            "selected_loadout": list(self.current_loadout),
            "loadout_provenance": list(getattr(self, "current_loadout_provenance", [])),
            "per_seed_diagnostics": self.curriculum.per_seed_diagnostics(
                selected_loadout=self.current_loadout
            ),
            "curriculum_state_path": str(curriculum_state_path),
            "curriculum_state_schema_version": CURRICULUM_STATE_SCHEMA_VERSION,
            "seed_order_source": self.current_seed_order_source,
            "seed_order_preserved": bool(self.current_seed_order_preserved),
            "seed_order_blocked_reason": self.current_seed_order_blocked_reason,
            "excluded_new_plants": list(self.current_excluded_new_plants),
            **self._capacity_context_fields(),
            "frontier_win_streak": int(self.frontier_win_streak),
            "frontier_win_streak_required": int(self.frontier_win_streak_required),
            "frontier_mastered_levels": list(self.frontier_mastered_levels),
        }
        self.plant_unlocks_path.write_text(json.dumps(unlock_payload, indent=2), encoding="utf-8")
        slot_payload = {
            "updated_at": time.time(),
            "max_seed_slots": self.max_seed_slots,
            **self._capacity_context_fields(),
            "selected_loadout_count": selected_count,
            "active_seed_slot_count": selected_count,
            "inactive_seed_slot_count": max(0, self.max_seed_slots - selected_count),
            "inactive_model_slots": max(0, self.max_seed_slots - selected_count),
            "configured_seed_list": list(self.configured_seed_list),
            "selected_loadout": list(self.current_loadout),
            "loadout_provenance": list(getattr(self, "current_loadout_provenance", [])),
            "per_seed_diagnostics": self.curriculum.per_seed_diagnostics(
                selected_loadout=self.current_loadout
            ),
            "curriculum_state_path": str(curriculum_state_path),
            "curriculum_state_schema_version": CURRICULUM_STATE_SCHEMA_VERSION,
            "loadout_reason": self.current_loadout_reason,
            "seed_order_source": self.current_seed_order_source,
            "seed_order_preserved": bool(self.current_seed_order_preserved),
            "seed_order_blocked_reason": self.current_seed_order_blocked_reason,
            "selectable_seeds": list(self.current_selectable_seeds),
            "eligible_seeds": self.curriculum.eligible_seeds(),
            "excluded_new_plants": list(self.current_excluded_new_plants),
            "frontier_win_streak": int(self.frontier_win_streak),
            "frontier_win_streak_required": int(self.frontier_win_streak_required),
            "frontier_mastery_ready": bool(self.frontier_mastery_ready),
            "frontier_promoted_this_episode": bool(self.frontier_promoted_this_episode),
            "frontier_mastery_reset_reason": str(self.frontier_mastery_reset_reason or ""),
            "frontier_replay_supported": bool(self.frontier_replay_supported),
            "frontier_replay_blocked_reason": str(self.frontier_replay_blocked_reason or ""),
            "frontier_mastered_levels": list(self.frontier_mastered_levels),
        }
        self.seed_slot_unlocks_path.write_text(json.dumps(slot_payload, indent=2), encoding="utf-8")


def _available_from_state(state: Dict[str, Any]) -> List[str]:
    values: List[str] = []
    for key in ("availableSeedNames", "visibleSeedCardNames", "unlockedSeedNames", "selectedSeedNames"):
        raw = state.get(key, [])
        if isinstance(raw, list):
            values.extend(str(item) for item in raw if str(item).strip())
    return _ordered_by_priority(_canonical_seed_sequence(values))


def _selectable_from_seed_screen_state(state: Dict[str, Any]) -> List[str]:
    values: List[str] = []
    for key in ("selectableSeedNames", "visibleSeedCardNames", "availableSeedNames", "selectedSeedNames"):
        raw = state.get(key, [])
        if isinstance(raw, list):
            values.extend(str(item) for item in raw if str(item).strip())
    values.extend(_seed_slot_choice_seed_names(state))
    return _ordered_by_priority(_canonical_seed_sequence(values))


def _available_for_seed_validation(
    selectable: Iterable[str],
    eligible: Iterable[str],
    unlocked: Iterable[str],
) -> Tuple[List[str], str]:
    selectable_list = _ordered_by_priority(_canonical_seed_sequence(selectable))
    if selectable_list:
        return selectable_list, "selectable"
    eligible_list = _ordered_by_priority(_canonical_seed_sequence(eligible))
    if eligible_list:
        return eligible_list, "selectable_empty_using_unlocked_or_eligible_fallback"
    unlocked_list = _ordered_by_priority(_canonical_seed_sequence(unlocked))
    if unlocked_list:
        return unlocked_list, "selectable_empty_using_unlocked_or_eligible_fallback"
    return [], "empty"


def _supported_registry_seed_names() -> List[str]:
    names: List[str] = []
    for entry in registry_entries():
        canonical = canonicalize_seed_name(entry.get("canonical_name", ""))
        if not canonical:
            continue
        try:
            plant_type = int(entry.get("plant_type_id", -1))
        except (TypeError, ValueError):
            plant_type = -1
        if plant_type >= 0:
            names.append(canonical)
    return _ordered_by_priority(names)


def _filter_supported_seed_names(seeds: Iterable[str]) -> List[str]:
    supported_keys = {normalize_plant_name(seed) for seed in _supported_registry_seed_names()}
    output: List[str] = []
    seen = set()
    for seed in _ordered_by_priority(_canonical_seed_sequence(seeds)):
        key = normalize_plant_name(seed)
        if supported_keys and key not in supported_keys:
            continue
        if key in seen:
            continue
        output.append(seed)
        seen.add(key)
    return output


def _filter_loadout_candidate_seeds(
    candidates: Iterable[str],
    *,
    eligible_seeds: Iterable[str],
    unlocked_seeds: Iterable[str],
) -> List[str]:
    allowed_keys = {
        normalize_plant_name(seed)
        for seed in _canonical_seed_sequence(list(eligible_seeds) + list(unlocked_seeds))
    }
    supported_keys = {normalize_plant_name(seed) for seed in _supported_registry_seed_names()}
    output: List[str] = []
    seen = set()
    for seed in _ordered_by_priority(_canonical_seed_sequence(candidates)):
        key = normalize_plant_name(seed)
        if supported_keys and key not in supported_keys:
            continue
        if allowed_keys and key not in allowed_keys:
            continue
        if key in seen:
            continue
        output.append(seed)
        seen.add(key)
    return output


def resolve_adventure_generalist_seed_capacity(
    state: Dict[str, Any],
    *,
    context: Dict[str, Any],
    previous_observed_capacity: int,
    previous_effective_capacity: int,
    selected_loadout: List[str],
    eligible_seeds: Iterable[str],
    unlocked_seeds: Iterable[str],
    medium_confirmed_seeds: Iterable[str] = (),
    infer_capacity_from_unlocks: bool = True,
    allow_weak_unlocked_capacity_fallback: bool = False,
    max_seed_slots: int = SEED_CAPACITY_MAX,
) -> SeedCapacityInference:
    maximum = _clamp_capacity(max_seed_slots, maximum=SEED_CAPACITY_MAX)
    observed_capacity = _infer_seed_bank_capacity_from_state(
        state,
        context=context,
        previous_capacity=previous_observed_capacity,
        selected_loadout=selected_loadout,
        max_seed_slots=maximum,
    )
    bridge_reported_capacity = _bridge_reported_seed_capacity_from_state(
        state,
        selected_loadout=selected_loadout,
        max_seed_slots=maximum,
    )
    inferred_capacity, source, reason, available_priority = _infer_capacity_from_available_priority(
        state,
        eligible_seeds=eligible_seeds,
        unlocked_seeds=unlocked_seeds,
        medium_confirmed_seeds=medium_confirmed_seeds,
        selected_loadout=selected_loadout,
        infer_capacity_from_unlocks=infer_capacity_from_unlocks,
        allow_weak_unlocked_capacity_fallback=allow_weak_unlocked_capacity_fallback,
        max_seed_slots=maximum,
    )
    raw_effective = _clamp_capacity(max(observed_capacity, inferred_capacity), maximum=maximum)
    previous_effective = _clamp_capacity(previous_effective_capacity or raw_effective, maximum=maximum)
    max_seen = _clamp_capacity(max(previous_effective, raw_effective), maximum=maximum)
    return SeedCapacityInference(
        bridge_reported_capacity=bridge_reported_capacity,
        observed_capacity=observed_capacity,
        inferred_capacity_from_unlocks=inferred_capacity,
        effective_seed_capacity=max_seen,
        max_effective_seed_capacity_seen=max_seen,
        inferred_capacity_source=source,
        capacity_inference_reason=reason,
        available_priority_seeds=available_priority,
        rejected_priority_seeds=[],
    )


def normalize_and_filter_priority_seeds(priority_order: Iterable[str], known_seed_names: Iterable[str]) -> List[str]:
    known_keys = {normalize_plant_name(seed) for seed in _canonical_seed_sequence(known_seed_names)}
    output: List[str] = []
    seen = set()
    for seed in priority_order:
        name = canonicalize_seed_name(seed)
        if not name:
            continue
        key = normalize_plant_name(name)
        if key in known_keys and key not in seen:
            output.append(name)
            seen.add(key)
    return output


def _infer_capacity_from_available_priority(
    state: Dict[str, Any],
    *,
    eligible_seeds: Iterable[str],
    unlocked_seeds: Iterable[str],
    medium_confirmed_seeds: Iterable[str],
    selected_loadout: List[str],
    infer_capacity_from_unlocks: bool,
    allow_weak_unlocked_capacity_fallback: bool,
    max_seed_slots: int,
) -> Tuple[int, str, str, List[str]]:
    base_capacity = _clamp_capacity(4, maximum=max_seed_slots)
    if not infer_capacity_from_unlocks:
        return base_capacity, "disabled", "capacity inference disabled", []

    visible_available = _ordered_by_priority(
        _state_seed_names(state, ("visibleSeedCardNames", "availableSeedNames"))
    )
    selected_names = _ordered_by_priority(_state_seed_names(state, ("selectedSeedNames",)))
    seed_slot_choice_names = _seed_slot_choice_seed_names(state)
    medium_confirmed = _ordered_by_priority(medium_confirmed_seeds)
    registry_names = _known_seed_names_from_registry()

    priority_order = normalize_and_filter_priority_seeds(
        SEED_CAPACITY_INFERENCE_PRIORITY,
        registry_names,
    )
    selected_loadout_len = len([seed for seed in selected_loadout if str(seed).strip()])
    medium_priority = _available_priority_seeds(medium_confirmed, priority_order)

    if visible_available:
        priority = _available_priority_seeds(visible_available, priority_order)
        if priority:
            return _priority_capacity_result(
                priority,
                base_capacity=base_capacity,
                max_seed_slots=max_seed_slots,
                source="selectable_priority_seeds",
            )
        if medium_priority:
            return _priority_capacity_result(
                medium_priority,
                base_capacity=base_capacity,
                max_seed_slots=max_seed_slots,
                source="unlock_event_priority_seed",
            )
        return base_capacity, "selectable_starter_only", "selectable seed names only show starter loadout", []

    if seed_slot_choice_names:
        priority = _available_priority_seeds(seed_slot_choice_names, priority_order)
        if priority:
            return _priority_capacity_result(
                priority,
                base_capacity=base_capacity,
                max_seed_slots=max_seed_slots,
                source="seed_bank_inventory_priority_seeds",
            )
        if medium_priority:
            return _priority_capacity_result(
                medium_priority,
                base_capacity=base_capacity,
                max_seed_slots=max_seed_slots,
                source="unlock_event_priority_seed",
            )
        return base_capacity, "seed_bank_inventory_starter_only", "seed bank inventory only shows starter loadout", []

    if selected_names:
        priority = _available_priority_seeds(selected_names, priority_order)
        if not priority and medium_priority:
            return _priority_capacity_result(
                medium_priority,
                base_capacity=base_capacity,
                max_seed_slots=max_seed_slots,
                source="unlock_event_priority_seed",
            )
        capped_capacity = max(base_capacity, selected_loadout_len)
        capacity = min(capped_capacity, base_capacity + len(priority))
        reason = _priority_capacity_reason(priority) if priority else "selected seed names only show starter loadout"
        return (
            _clamp_capacity(capacity, maximum=max_seed_slots),
            "selected_seed_names_current_loadout",
            reason,
            priority,
        )

    if medium_priority:
        return _priority_capacity_result(
            medium_priority,
            base_capacity=base_capacity,
            max_seed_slots=max_seed_slots,
            source="unlock_event_priority_seed",
        )

    if allow_weak_unlocked_capacity_fallback:
        weak_priority_order = normalize_and_filter_priority_seeds(
            SEED_CAPACITY_INFERENCE_PRIORITY,
            registry_names,
        )
        unlocked_priority = _available_priority_seeds(unlocked_seeds, weak_priority_order)
        if unlocked_priority:
            return _priority_capacity_result(
                unlocked_priority,
                base_capacity=base_capacity,
                max_seed_slots=max_seed_slots,
                source="weak_unlocked_seed_fallback",
            )

    return base_capacity, "starter_only", "no confirmed priority seeds available", []


def _priority_capacity_result(
    priority: List[str],
    *,
    base_capacity: int,
    max_seed_slots: int,
    source: str,
) -> Tuple[int, str, str, List[str]]:
    capacity = _clamp_capacity(base_capacity + len(priority), maximum=max_seed_slots)
    return capacity, source, _priority_capacity_reason(priority), list(priority)


def _priority_capacity_reason(priority: List[str]) -> str:
    if "WallNut" in priority and "CherryBomb" in priority:
        return "WallNut+CherryBomb available"
    if priority:
        return "+".join(priority) + " available"
    return "no confirmed priority seeds available"


def _rejected_priority_seed_diagnostics(
    priority_seeds: Iterable[str],
    *,
    selected_loadout: Iterable[str],
    selectable_seeds: Iterable[str],
    effective_capacity: int,
    excluded_new_plants: Iterable[Dict[str, str]],
) -> List[Dict[str, str]]:
    selected_keys = {normalize_plant_name(seed) for seed in _canonical_seed_sequence(selected_loadout)}
    selectable_keys = {normalize_plant_name(seed) for seed in _canonical_seed_sequence(selectable_seeds)}
    excluded_reason_by_key = {
        normalize_plant_name(canonicalize_seed_name(row.get("seed", ""))): str(row.get("reason", "") or "")
        for row in excluded_new_plants
        if isinstance(row, dict)
    }
    selected_len = len(_canonical_seed_sequence(selected_loadout))
    rejected: List[Dict[str, str]] = []
    for seed in _ordered_by_priority(priority_seeds):
        key = normalize_plant_name(seed)
        if key in selected_keys:
            continue
        if key not in selectable_keys:
            reason = "name_not_in_selectable"
        elif selected_len >= int(effective_capacity):
            reason = excluded_reason_by_key.get(key) or "capacity_limit"
        else:
            reason = excluded_reason_by_key.get(key) or "not_selected"
        rejected.append({"seed": seed, "reason": reason})
    return rejected


def _available_priority_seeds(available: Iterable[str], priority_order: Iterable[str]) -> List[str]:
    available_keys = {normalize_plant_name(seed) for seed in _canonical_seed_sequence(available)}
    return [seed for seed in priority_order if normalize_plant_name(seed) in available_keys]


def _known_seed_names_from_registry() -> List[str]:
    names: List[str] = []
    for entry in registry_entries():
        canonical = str(entry.get("canonical_name", "")).strip()
        if canonical:
            names.append(canonical)
        aliases = entry.get("aliases", [])
        if isinstance(aliases, list):
            names.extend(str(alias).strip() for alias in aliases if str(alias).strip())
    return names


def _state_seed_names(state: Dict[str, Any], keys: Iterable[str]) -> List[str]:
    values: List[str] = []
    for key in keys:
        raw = state.get(key, [])
        if isinstance(raw, list):
            values.extend(str(item).strip() for item in raw if str(item).strip())
    return _ordered_by_priority(_canonical_seed_sequence(values))


def _seed_slot_choice_seed_names(state: Dict[str, Any]) -> List[str]:
    slots = state.get("seedSlots", [])
    if not isinstance(slots, list):
        return []
    values: List[str] = []
    for slot in slots:
        if not isinstance(slot, dict) or not _slot_represents_available_seed_choice(slot):
            continue
        name = str(slot.get("plantTypeName") or slot.get("displayName") or slot.get("seedName") or "").strip()
        if not name:
            plant_type = _safe_int(slot.get("plantType"), default=-1)
            name = plant_type_name(plant_type) if plant_type >= 0 else ""
        if name:
            values.append(name)
    return _ordered_by_priority(_canonical_seed_sequence(values))


def _slot_represents_available_seed_choice(slot: Dict[str, Any]) -> bool:
    for key in (
        "availableSeedCard",
        "isAvailableSeedCard",
        "visibleSeedCard",
        "isVisibleSeedCard",
        "seedChoice",
        "isSeedChoice",
        "selectableSeedCard",
        "isSelectableSeedCard",
    ):
        if slot.get(key):
            return True
    return False


def _bridge_reported_seed_capacity_from_state(
    state: Dict[str, Any],
    *,
    selected_loadout: List[str],
    max_seed_slots: int,
) -> Optional[int]:
    candidates: List[int] = []
    for key in (
        "seedBankCapacity",
        "seedSlotCapacity",
        "seedSlotCount",
        "seedCardSlotCount",
        "seedSelectionSlotCount",
        "selectableSeedCapacity",
        "currentSeedBankCapacity",
        "activeSeedSlotCapacity",
    ):
        value = _safe_int(state.get(key), default=0)
        if value > 0:
            candidates.append(value)
    slot_capacity = _seed_slots_capacity_candidate(state, selected_loadout=selected_loadout)
    if slot_capacity > 0:
        candidates.append(slot_capacity)
    if not candidates:
        return None
    return _clamp_capacity(max(candidates), maximum=max_seed_slots)


def _infer_seed_bank_capacity_from_state(
    state: Dict[str, Any],
    *,
    context: Dict[str, Any],
    previous_capacity: int,
    selected_loadout: List[str],
    max_seed_slots: int = SEED_CAPACITY_MAX,
) -> int:
    explicit_override = _first_int_from_values(
        [
            context.get("observed_seed_bank_capacity_override"),
            context.get("seed_bank_capacity_override"),
            context.get("active_seed_slot_capacity_override"),
        ]
    )
    if explicit_override is not None:
        return _clamp_capacity(explicit_override, maximum=max_seed_slots)

    minimum_expected_capacity = _clamp_capacity(
        _first_int_from_values(
            [
                len(selected_loadout),
                previous_capacity,
                context.get("observed_seed_bank_capacity"),
                context.get("active_seed_slot_capacity"),
                context.get("current_seed_bank_capacity"),
            ]
        )
        or 1,
        maximum=max_seed_slots,
    )

    bridge_reported = _bridge_reported_seed_capacity_from_state(
        state,
        selected_loadout=selected_loadout,
        max_seed_slots=max_seed_slots,
    )
    if bridge_reported is not None:
        probed_capacity = _clamp_capacity(bridge_reported, maximum=max_seed_slots)
        return max(minimum_expected_capacity, probed_capacity)

    return minimum_expected_capacity


def _seed_slots_capacity_candidate(state: Dict[str, Any], *, selected_loadout: List[str]) -> int:
    raw = state.get("seedSlots", [])
    if not isinstance(raw, list) or not raw:
        return 0
    selected_count = len([seed for seed in selected_loadout if str(seed).strip()])
    if len(raw) <= selected_count:
        return len(raw)
    if any(isinstance(slot, dict) and _slot_represents_available_seed_choice(slot) for slot in raw):
        return len(raw)
    return 0


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _first_int_from_values(values: List[Any]) -> Optional[int]:
    for value in values:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def _clamp_capacity(value: int, *, maximum: int) -> int:
    return max(1, min(int(maximum), int(value)))


def _normalize_seed_order_source(value: Any) -> str:
    text = str(value or "").strip()
    if text in {SEED_ORDER_SOURCE_EXPLICIT, SEED_ORDER_SOURCE_DEFAULT, SEED_ORDER_SOURCE_RANDOMIZED}:
        return text
    return SEED_ORDER_SOURCE_DEFAULT


def canonicalize_seed_name(value: Any, known_seed_names: Optional[Iterable[str]] = None) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    canonical = _adventure_canonical_seed_name(text)
    if canonical:
        return canonical
    key = normalize_plant_name(text)
    for known in known_seed_names or []:
        known_name = _adventure_canonical_seed_name(known)
        if known_name and normalize_plant_name(known_name) == key:
            return known_name
    return text


def _ordered_unique_seed_names(values: Iterable[Any]) -> List[str]:
    output: List[str] = []
    seen: set[str] = set()
    for value in values:
        name = canonicalize_seed_name(value)
        if name and name not in seen:
            output.append(name)
            seen.add(name)
    return output


def _canonical_seed_sequence(values: Iterable[Any], known_seed_names: Optional[Iterable[str]] = None) -> List[str]:
    output: List[str] = []
    for value in values:
        name = canonicalize_seed_name(value, known_seed_names)
        if name:
            output.append(name)
    return output


def _seed_order_preserved(configured_seed_list: Iterable[str], selected_loadout: Iterable[str]) -> bool:
    configured = [str(seed).strip() for seed in configured_seed_list if str(seed).strip()]
    selected = [str(seed).strip() for seed in selected_loadout if str(seed).strip()]
    if not configured:
        return True
    if len(selected) >= len(configured):
        return selected[: len(configured)] == configured
    return selected == configured[: len(selected)]


def _seed_order_blocked_reason(
    configured_seed_list: Iterable[str],
    selected_loadout: Iterable[str],
    observed_capacity: int,
) -> str:
    configured = [str(seed).strip() for seed in configured_seed_list if str(seed).strip()]
    selected = [str(seed).strip() for seed in selected_loadout if str(seed).strip()]
    if not configured or _seed_order_preserved(configured, selected):
        return ""
    if int(observed_capacity) < len(configured):
        return f"configured_seed_list_exceeds_observed_capacity:requested={len(configured)} observed={int(observed_capacity)}"
    return "selected_loadout_order_differs_from_configured_seed_list"


def _ordered_by_priority(values: Iterable[str]) -> List[str]:
    seen = {name for name in _canonical_seed_sequence(values)}
    output: List[str] = []
    for seed in SEED_PRIORITY:
        if seed in seen:
            output.append(seed)
            seen.remove(seed)
    output.extend(sorted(seen))
    return output


def _priority_index(seed: str) -> int:
    seed_name = canonicalize_seed_name(seed)
    try:
        return SEED_PRIORITY.index(seed_name)
    except ValueError:
        return len(SEED_PRIORITY) + 1


def _snapshot_post_win_state(state: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "screenState",
        "nextStep",
        "gameplayReady",
        "isGameplayReady",
        "isSeedSelectionScreen",
        "seedSelectionActive",
        "isAdventureButtonVisible",
        "trophyVisible",
        "levelCompleteTrophyVisible",
        "postWinClickRequired",
        "rewardScreenVisible",
        "unlockScreenVisible",
        "newPlantUnlockedVisible",
        "newPlantUnlockedName",
        "currentAdventureLevel",
    )
    return {key: state.get(key) for key in keys if key in state}


def _snapshot_startup_state(state: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "screenState",
        "currentMode",
        "gameBoardType",
        "isMainMenu",
        "isAdventureButtonVisible",
        "startupPopupVisible",
        "startupOkButtonVisible",
        "mainMenuBlockedByPopup",
        "isSeedSelectionScreen",
        "seedSelectionActive",
        "seedSelectionPanelActive",
        "isGameplayReady",
        "gameplayReady",
        "isLevelComplete",
        "isRewardScreen",
        "trophyVisible",
        "levelCompleteTrophyVisible",
        "postWinClickRequired",
        "rewardScreenVisible",
        "unlockScreenVisible",
        "newPlantUnlockedVisible",
        "blockingRewardUiActive",
        "currentAdventureLevel",
        "profileAdventureLevel",
        "profileAdventureLevelSource",
        "currentWorldOrStage",
        "currentDayLevel",
        "uiWorldLevelText",
        "uiLevelText",
    )
    snapshot = {key: state.get(key) for key in keys if key in state}
    snapshot.setdefault("screenState", adventure_screen_state_name(state))
    snapshot.setdefault("currentMode", state.get("currentMode") or state.get("gameBoardType") or "")
    snapshot.setdefault("currentAdventureLevel", adventure_bridge_detected_level(state))
    snapshot.setdefault("profileAdventureLevel", adventure_profile_adventure_level(state))
    snapshot.setdefault("uiWorldLevelText", adventure_ui_world_level_text(state))
    snapshot.setdefault("seedSelectionDetected", adventure_seed_selection_detected(state))
    snapshot.setdefault("gameplayReadyDetected", adventure_gameplay_ready_detected(state))
    return snapshot


def _replay_level_check_authoritative(state: Dict[str, Any]) -> bool:
    screen_state = str(state.get("screenState") or state.get("screen_state") or "")
    seed_selection = bool(state.get("isSeedSelectionScreen") or state.get("seedSelectionActive") or screen_state == "seed_selection")
    gameplay_ready = bool(
        (state.get("isGameplayReady") or state.get("gameplayReady") or screen_state == "gameplay")
        and not seed_selection
    )
    return bool(gameplay_ready and screen_state == "gameplay")


def parse_initial_loadout(value: Any) -> List[str]:
    if isinstance(value, list):
        return _canonical_seed_sequence(value)
    text = str(value or "").strip()
    return _canonical_seed_sequence(parse_seed_list(text)) if text else list(ADVENTURE_GENERALIST_INITIAL_LOADOUT)
