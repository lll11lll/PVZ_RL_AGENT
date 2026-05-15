"""Adventure Generalist 14-slot identity training support."""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pvzrl_adventure import (
    BASE_UNLOCKED_SEEDS,
    LiveStatusWriter,
    build_live_status,
    collect_post_win_unlocks,
    prepare_adventure_gameplay,
    replay_current_level_after_validation_win,
)
from pvzrl_env import parse_seed_list, resolve_seed_list
from pvzrl_sb3 import PvZMaskedPPOEnv, PvZSB3Config


ADVENTURE_GENERALIST_MODEL_FAMILY = "ppo_adventure_generalist_14slot_identity_v1"
ADVENTURE_GENERALIST_RUN_MODE_TRAIN = "adventure_generalist_14slot_train"
ADVENTURE_GENERALIST_RUN_MODE_EVAL = "adventure_generalist_14slot_eval"
ADVENTURE_GENERALIST_INITIAL_LOADOUT = ["SunFlower", "SunFlower", "Peashooter", "Peashooter"]
BLOCKED_INITIAL_LOADOUT_UNAVAILABLE = "required_initial_adventure_generalist_loadout_unavailable"
BLOCKED_FRONTIER_REPLAY_REQUIRED = "frontier_win_streak_requires_same_level_replay_support"

SEED_PRIORITY = [
    "SunFlower",
    "Peashooter",
    "WallNut",
    "CherryBomb",
    "PotatoMine",
    "SnowPea",
    "Repeater",
]

SEED_CAPACITY_MAX = 14


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
    selected_loadout_count: int = 0
    observed_seed_bank_capacity: int = 0
    inactive_model_slots: int = 0
    selectable_seeds: List[str] = field(default_factory=list)
    eligible_seeds: List[str] = field(default_factory=list)
    loadout_reason: str = ""
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
    frontier_win_streak: int = 0
    frontier_win_streak_required: int = 1
    frontier_mastery_ready: bool = False
    frontier_promoted_this_episode: bool = False
    frontier_mastery_reset_reason: str = ""
    mastery_sample_source: str = ""
    frontier_replay_supported: bool = True
    frontier_replay_blocked_reason: str = ""
    frontier_mastered_levels: List[int] = field(default_factory=list)


@dataclass
class LoadoutDecision:
    selected_loadout: List[str]
    loadout_reason: str
    eligible_seeds: List[str]
    selectable_seeds: List[str]
    excluded_new_plants: List[Dict[str, str]]
    observed_seed_bank_capacity: int


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
    ) -> None:
        self.initial_loadout = [str(seed).strip() for seed in initial_loadout if str(seed).strip()]
        self.max_seed_slots = _clamp_capacity(max_seed_slots, maximum=SEED_CAPACITY_MAX)
        self.unlock_aware = bool(unlock_aware)
        self.seed_curriculum = str(seed_curriculum or "conservative").strip().lower()
        self.unlock_introduction_delay = max(0, int(unlock_introduction_delay))
        self.new_plant_min_inclusion_prob = max(0.0, min(1.0, float(new_plant_min_inclusion_prob)))
        self.unlock_episode: Dict[str, int] = {}
        self.episode_index = 0
        for seed in BASE_UNLOCKED_SEEDS:
            self.unlock_episode[str(seed)] = 0
        for seed in self.initial_loadout:
            self.unlock_episode[str(seed)] = 0

    def record_unlocked(self, seeds: Iterable[str], episode_index: Optional[int] = None) -> List[str]:
        episode = self.episode_index if episode_index is None else int(episode_index)
        newly_unlocked: List[str] = []
        for seed in seeds:
            name = str(seed or "").strip()
            if not name or name in self.unlock_episode:
                continue
            self.unlock_episode[name] = episode
            newly_unlocked.append(name)
        return _ordered_by_priority(newly_unlocked)

    def unlocked_seeds(self) -> List[str]:
        return _ordered_by_priority(self.unlock_episode.keys())

    def eligible_seeds(self) -> List[str]:
        eligible = set(self.initial_loadout)
        if self.unlock_aware:
            for seed, episode in self.unlock_episode.items():
                if self.episode_index - int(episode) >= self.unlock_introduction_delay:
                    eligible.add(seed)
        return _ordered_by_priority(eligible)

    def choose_loadout(
        self,
        selectable_seeds: Iterable[str],
        observed_capacity: int,
        previous_loadout: Optional[Iterable[str]] = None,
    ) -> LoadoutDecision:
        capacity = _clamp_capacity(observed_capacity, maximum=self.max_seed_slots)
        selectable_list_raw = [str(seed).strip() for seed in selectable_seeds if str(seed).strip()]
        selectable_list = _ordered_by_priority(selectable_list_raw)
        selectable_set = set(selectable_list)
        if not selectable_set:
            fallback = [str(seed).strip() for seed in (previous_loadout or self.initial_loadout) if str(seed).strip()]
            selectable_list = _ordered_by_priority(fallback)
            selectable_set = set(selectable_list)

        required_initial_unique = _ordered_by_priority(set(self.initial_loadout))
        missing_initial = [seed for seed in required_initial_unique if seed not in selectable_set]
        if missing_initial:
            raise RuntimeError(
                f"blocked_reason={BLOCKED_INITIAL_LOADOUT_UNAVAILABLE}: missing={','.join(missing_initial)}"
            )

        eligible_all = self.eligible_seeds()
        eligible_selectable = [seed for seed in eligible_all if seed in selectable_set]
        initial_unique = set(self.initial_loadout)
        known_new = [seed for seed in _ordered_by_priority(self.unlock_episode.keys()) if seed not in initial_unique]

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

        previous = [str(seed).strip() for seed in (previous_loadout or []) if str(seed).strip()]
        if self.seed_curriculum == "varied":
            selected, local_reasons, reason = self._build_varied_loadout(
                capacity=capacity,
                selectable_set=selectable_set,
                eligible_selectable=eligible_selectable,
                new_candidates=new_candidates,
                initial_unique=initial_unique,
            )
        else:
            selected, local_reasons, reason = self._build_conservative_loadout(
                capacity=capacity,
                selectable_set=selectable_set,
                eligible_selectable=eligible_selectable,
                new_candidates=new_candidates,
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

        return LoadoutDecision(
            selected_loadout=selected[:capacity],
            loadout_reason=reason,
            eligible_seeds=eligible_all,
            selectable_seeds=selectable_list,
            excluded_new_plants=excluded_new_plants,
            observed_seed_bank_capacity=capacity,
        )

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
        base = [seed for seed in self.initial_loadout if seed in selectable_set]
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
                if random.random() <= self.new_plant_min_inclusion_prob:
                    selected.append(seed)
                    added_new.add(seed)
                else:
                    exclusion_reasons.setdefault(seed, "probability_gate")
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
            replaced = False
            for seed in new_candidates:
                if random.random() > self.new_plant_min_inclusion_prob:
                    exclusion_reasons.setdefault(seed, "probability_gate")
                    continue
                replace_index = max(0, len(selected) - 1)
                if len(selected) > 2:
                    replace_index = max(2, replace_index)
                selected[replace_index] = seed
                added_new.add(seed)
                replaced = True
                reason = "conservative_replace_tail_slot"
                break
            if not replaced:
                reason = "initial_unlock_wait"

        if new_candidates and not added_new:
            for seed in new_candidates:
                exclusion_reasons.setdefault(seed, "capacity_full")
        for seed in new_candidates:
            if seed not in added_new and len(selected) >= capacity:
                exclusion_reasons.setdefault(seed, "capacity_full")
        return selected[:capacity], exclusion_reasons, reason

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
            if is_new_seed and random.random() > max(self.new_plant_min_inclusion_prob, 0.5):
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
        replay_cleared_levels: bool,
        frontier_sample_prob: float,
        recent_cleared_sample_prob: float,
        maintenance_sample_prob: float,
        frontier_win_streak_required: int,
    ) -> None:
        super().__init__(config)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.progress_jsonl_path = self.run_dir / "adventure_training_progress.jsonl"
        self.plant_unlocks_path = self.run_dir / "plant_unlocks.json"
        self.seed_slot_unlocks_path = self.run_dir / "seed_slot_unlocks.json"
        self.writer = LiveStatusWriter(live_status_path)
        self.curriculum = AdventureSeedCurriculum(
            initial_loadout=initial_loadout,
            max_seed_slots=SEED_CAPACITY_MAX,
            unlock_aware=unlock_aware_seed_curriculum,
            seed_curriculum=seed_curriculum,
            unlock_introduction_delay=unlock_introduction_delay,
            new_plant_min_inclusion_prob=new_plant_min_inclusion_prob,
        )
        self.max_seed_slots = SEED_CAPACITY_MAX
        self.max_adventure_levels = max(1, int(max_adventure_levels))
        self.max_attempts_per_level = max(1, int(max_attempts_per_level))
        self.adventure_start_level = max(1, int(adventure_start_level))
        self.frontier_win_streak_required = max(1, int(frontier_win_streak_required))
        self.frontier_win_streak = 0
        self.frontier_mastered_levels: List[int] = []
        self.frontier_replay_supported = True
        self.frontier_replay_blocked_reason = ""
        self.frontier_mastery_reset_reason = ""
        self.frontier_promoted_this_episode = False
        self.frontier_mastery_ready = False
        self._hard_blocked_reason = ""
        self.replay_cleared_levels = bool(replay_cleared_levels)
        self.sample_probs = {
            "frontier": float(frontier_sample_prob),
            "recent_cleared": float(recent_cleared_sample_prob),
            "maintenance": float(maintenance_sample_prob),
        }
        self.current_level = self.adventure_start_level
        self.current_attempt = 0
        self.episode_index = 0
        self.cleared_levels: List[int] = []
        self.current_sample_source = "frontier"
        self.observed_seed_bank_capacity = _clamp_capacity(len(initial_loadout), maximum=self.max_seed_slots)
        self.current_loadout = list(initial_loadout[: self.observed_seed_bank_capacity])
        self.current_loadout_reason = "initial"
        self.current_selectable_seeds = _ordered_by_priority(set(self.current_loadout))
        self.current_excluded_new_plants: List[Dict[str, str]] = []
        self._apply_loadout(self.current_loadout)
        self.context: Dict[str, Any] = {
            "mode": ADVENTURE_GENERALIST_RUN_MODE_TRAIN,
            "run_mode": ADVENTURE_GENERALIST_RUN_MODE_TRAIN,
            "status": "starting",
            "state": "STARTING",
            "active_run": str(self.run_dir),
            "current_stage": "adventure_generalist_14slot_identity_v1",
            "current_model_family": ADVENTURE_GENERALIST_MODEL_FAMILY,
            "current_model_path": "",
            "selected_seeds": list(self.current_loadout),
            "selected_loadout_count": len(self.current_loadout),
            "unlocked_seeds": self.curriculum.unlocked_seeds(),
            "eligible_seeds": self.curriculum.eligible_seeds(),
            "selectable_seeds": list(self.current_selectable_seeds),
            "active_seed_slot_count": len(self.current_loadout),
            "inactive_seed_slot_count": max(0, self.max_seed_slots - len(self.current_loadout)),
            "max_seed_slots": self.max_seed_slots,
            "observed_seed_bank_capacity": self.observed_seed_bank_capacity,
            "active_seed_slot_capacity": self.observed_seed_bank_capacity,
            "current_seed_bank_capacity": self.observed_seed_bank_capacity,
            "inactive_model_slots": max(0, self.max_seed_slots - len(self.current_loadout)),
            "loadout_reason": self.current_loadout_reason,
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
            "unlock_aware_seed_curriculum": bool(unlock_aware_seed_curriculum),
            "seed_curriculum": seed_curriculum,
            "post_win_transition": {},
            "post_win_blocked_reason": "",
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

    def _set_hard_blocked(self, reason: str) -> None:
        blocked_reason = str(reason or BLOCKED_FRONTIER_REPLAY_REQUIRED)
        self._hard_blocked_reason = blocked_reason
        self.frontier_replay_supported = False
        self.frontier_replay_blocked_reason = blocked_reason
        self.context.update(
            {
                "status": "blocked",
                "state": "BLOCKED_FRONTIER_REPLAY",
                "blocked_reason": blocked_reason,
                "frontier_replay_supported": False,
                "frontier_replay_blocked_reason": blocked_reason,
            }
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):  # type: ignore[override]
        self._raise_if_hard_blocked()
        self.curriculum.episode_index = self.episode_index
        adventure_state = self._safe_adventure_state()
        self.current_sample_source = self._sample_source()
        self.current_attempt += 1
        self.context.update(
            {
                "status": "running",
                "state": "PREPARE_GAMEPLAY",
                "blocked_reason": "",
                "current_level": self.current_level,
                "frontier_level": self.current_level,
                "current_attempt": self.current_attempt,
                "selected_seeds": list(self.current_loadout),
                "selected_loadout_count": len(self.current_loadout),
                "active_seed_slot_count": len(self.current_loadout),
                "inactive_seed_slot_count": max(0, self.max_seed_slots - len(self.current_loadout)),
                "observed_seed_bank_capacity": self.observed_seed_bank_capacity,
                "active_seed_slot_capacity": self.observed_seed_bank_capacity,
                "current_seed_bank_capacity": self.observed_seed_bank_capacity,
                "inactive_model_slots": max(0, self.max_seed_slots - len(self.current_loadout)),
                "unlocked_seeds": self.curriculum.unlocked_seeds(),
                "eligible_seeds": self.curriculum.eligible_seeds(),
                "selectable_seeds": list(self.current_selectable_seeds),
                "episode_sample_source": self.current_sample_source,
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
        return self.start_episode_from_observation(observation, reset_info)

    def step(self, action: int):  # type: ignore[override]
        encoded, reward, terminated, truncated, info = super().step(action)
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
        self.writer.write(build_live_status(self, self.context, last_info=info))
        if terminated or truncated:
            self._finish_episode(info)
        return encoded, reward, terminated, truncated, info

    def _finish_episode(self, info: Dict[str, Any]) -> None:
        summary = info.get("episode_summary", {}) if isinstance(info, dict) else {}
        result = str(summary.get("done_reason") or info.get("done_reason") or "unknown")
        episode_level = self.current_level
        episode_attempt = self.current_attempt
        newly_unlocked: List[str] = []
        post_win_blocked_reason = ""
        post_win_transition: Dict[str, Any] = {}
        post_win_last_state: Dict[str, Any] = {}
        unknown_unlock_objects: List[Dict[str, Any]] = []
        frontier_level_before = int(self.current_level)
        frontier_promoted_this_episode = False
        frontier_mastery_ready = False
        frontier_mastery_reset_reason = ""
        frontier_replay_blocked_reason = str(self.frontier_replay_blocked_reason or "")
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
                seeds = list(unlock_snapshot.get("visibleSeedCardNames", []) or [])
                if unlock_snapshot.get("newPlantUnlockedName"):
                    seeds.append(str(unlock_snapshot.get("newPlantUnlockedName")))
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

        is_win_like = result in {"win", "post_win_pending"}
        is_frontier_mastery_attempt = (
            mastery_sample_source in {"frontier", "frontier_mastery_replay"}
            and int(episode_level) == int(frontier_level_before)
        )

        if is_win_like and episode_level not in self.cleared_levels:
            self.cleared_levels.append(episode_level)

        if is_win_like:
            self.current_attempt = 0
            if is_frontier_mastery_attempt:
                self.frontier_win_streak = int(self.frontier_win_streak) + 1
                frontier_mastery_ready = self.frontier_win_streak >= self.frontier_win_streak_required
                threshold_met = bool(frontier_mastery_ready)
                post_win_decision = "advance_next_level" if threshold_met else "replay_same_level"
                self.context.update(
                    {
                        "wins_on_current_level": int(self.frontier_win_streak),
                        "wins_before_advance": int(self.frontier_win_streak_required),
                        "frontier_win_streak": int(self.frontier_win_streak),
                        "frontier_win_streak_required": int(self.frontier_win_streak_required),
                        "frontier_mastery_ready": bool(frontier_mastery_ready),
                        "post_win_decision": post_win_decision,
                        "post_win_transition_allowed": bool(threshold_met),
                    }
                )
                print(
                    "[adventure-generalist] "
                    f"win detected level={episode_level} "
                    f"wins_on_level={int(self.frontier_win_streak)}/{int(self.frontier_win_streak_required)}"
                )
                print(
                    "[adventure-generalist] "
                    f"decision={post_win_decision} threshold_met={threshold_met}"
                )
                print(
                    "[adventure-generalist] "
                    f"post_win_transition_allowed={'true' if threshold_met else 'false'}"
                )
                if frontier_mastery_ready:
                    collect_allowed_post_win_transition()
                    frontier_promoted_this_episode = True
                    if episode_level not in self.frontier_mastered_levels:
                        self.frontier_mastered_levels.append(int(episode_level))
                    self.current_level = min(
                        self.adventure_start_level + self.max_adventure_levels - 1,
                        episode_level + 1,
                    )
                    self.frontier_win_streak = 0
                    frontier_mastery_reset_reason = "promoted"
                else:
                    self.current_level = int(episode_level)
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
                    if self.frontier_win_streak_required > 1:
                        replay_ok, replay_reason = replay_current_level_after_validation_win(
                            self,
                            self.writer,
                            self.context,
                            timeout=self.config.gameplay_ready_timeout,
                            expected_level=int(episode_level),
                        )
                        replay_state = self._safe_adventure_state()
                        replay_level = int(replay_state.get("currentAdventureLevel", episode_level) or episode_level)
                        self.context["frontier_replay_level_after_win"] = replay_level
                        self.context["frontier_replay_last_state"] = _snapshot_post_win_state(replay_state)
                        if replay_ok and replay_level != int(episode_level):
                            replay_ok = False
                            replay_reason = BLOCKED_FRONTIER_REPLAY_REQUIRED
                        if not replay_ok:
                            frontier_replay_blocked_reason = BLOCKED_FRONTIER_REPLAY_REQUIRED
                            self._set_hard_blocked(BLOCKED_FRONTIER_REPLAY_REQUIRED)
                        else:
                            self.frontier_replay_supported = True
                            frontier_replay_blocked_reason = ""
            else:
                self.context.update(
                    {
                        "post_win_decision": "hold_frontier",
                        "post_win_transition_allowed": False,
                        "wins_on_current_level": int(self.frontier_win_streak),
                        "wins_before_advance": int(self.frontier_win_streak_required),
                    }
                )
                self.current_level = int(frontier_level_before)
        else:
            if is_frontier_mastery_attempt and self.frontier_win_streak > 0:
                if result == "loss":
                    frontier_mastery_reset_reason = "loss"
                elif result == "timeout":
                    frontier_mastery_reset_reason = "timeout"
                elif result == "env_corruption":
                    frontier_mastery_reset_reason = "env_corruption"
                else:
                    frontier_mastery_reset_reason = str(result or "failure")
                self.frontier_win_streak = 0
            if result in {"loss", "timeout"} and self.current_attempt >= self.max_attempts_per_level:
                self.current_attempt = 0

        self.frontier_promoted_this_episode = bool(frontier_promoted_this_episode)
        self.frontier_mastery_ready = bool(frontier_mastery_ready)
        self.frontier_mastery_reset_reason = str(frontier_mastery_reset_reason or "")

        selected_loadout_count = len(self.current_loadout)
        progress = AdventureGeneralistProgress(
            episode=self.episode_index,
            level=episode_level,
            attempt=episode_attempt,
            sample_source=self.current_sample_source,
            result=result,
            selected_loadout=list(self.current_loadout),
            selected_loadout_count=selected_loadout_count,
            active_seed_slot_count=selected_loadout_count,
            observed_seed_bank_capacity=self.observed_seed_bank_capacity,
            inactive_model_slots=max(0, self.max_seed_slots - selected_loadout_count),
            selectable_seeds=list(self.current_selectable_seeds),
            eligible_seeds=self.curriculum.eligible_seeds(),
            loadout_reason=self.current_loadout_reason,
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
            frontier_win_streak=int(self.frontier_win_streak),
            frontier_win_streak_required=int(self.frontier_win_streak_required),
            frontier_mastery_ready=bool(frontier_mastery_ready),
            frontier_promoted_this_episode=bool(frontier_promoted_this_episode),
            frontier_mastery_reset_reason=str(frontier_mastery_reset_reason or ""),
            mastery_sample_source=str(mastery_sample_source or ""),
            frontier_replay_supported=bool(self.frontier_replay_supported),
            frontier_replay_blocked_reason=str(self.frontier_replay_blocked_reason or frontier_replay_blocked_reason or ""),
            frontier_mastered_levels=list(self.frontier_mastered_levels),
        )
        self._append_progress(progress)
        self.episode_index += 1
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
                "selected_seeds": list(self.current_loadout),
                "selected_loadout_count": selected_loadout_count,
                "active_seed_slot_count": selected_loadout_count,
                "inactive_seed_slot_count": max(0, self.max_seed_slots - selected_loadout_count),
                "inactive_model_slots": max(0, self.max_seed_slots - selected_loadout_count),
                "observed_seed_bank_capacity": self.observed_seed_bank_capacity,
                "active_seed_slot_capacity": self.observed_seed_bank_capacity,
                "current_seed_bank_capacity": self.observed_seed_bank_capacity,
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
        self._write_unlock_files()
        self.writer.write(build_live_status(self, self.context, last_info=info))

    def _on_seed_selection_screen(self, state: Dict[str, Any], current_seed_list: List[str]) -> Tuple[List[str], str]:
        selectable = _selectable_from_seed_screen_state(state)
        capacity = _infer_seed_bank_capacity_from_state(
            state,
            context=self.context,
            previous_capacity=self.observed_seed_bank_capacity,
            selected_loadout=current_seed_list or self.current_loadout,
        )
        self.observed_seed_bank_capacity = capacity
        self.curriculum.record_unlocked(selectable, self.episode_index)
        try:
            decision = self.curriculum.choose_loadout(
                selectable_seeds=selectable,
                observed_capacity=capacity,
                previous_loadout=current_seed_list or self.current_loadout,
            )
        except RuntimeError as exc:
            blocked_reason = str(exc).replace("blocked_reason=", "", 1) if str(exc).startswith("blocked_reason=") else str(exc)
            fallback = [str(seed).strip() for seed in (current_seed_list or self.current_loadout) if str(seed).strip()][:capacity]
            self.current_loadout = list(fallback)
            self.current_selectable_seeds = _ordered_by_priority(selectable)
            self.current_excluded_new_plants = []
            self.current_loadout_reason = "seed_selection_blocked"
            self.context.update(
                {
                    "selectable_seeds": list(self.current_selectable_seeds),
                    "observed_seed_bank_capacity": capacity,
                    "active_seed_slot_capacity": capacity,
                    "current_seed_bank_capacity": capacity,
                    "selected_seeds": list(self.current_loadout),
                    "selected_loadout_count": len(self.current_loadout),
                    "active_seed_slot_count": len(self.current_loadout),
                    "inactive_seed_slot_count": max(0, self.max_seed_slots - len(self.current_loadout)),
                    "inactive_model_slots": max(0, self.max_seed_slots - len(self.current_loadout)),
                    "loadout_reason": self.current_loadout_reason,
                    "excluded_new_plants": list(self.current_excluded_new_plants),
                    "eligible_seeds": self.curriculum.eligible_seeds(),
                    "unlocked_seeds": self.curriculum.unlocked_seeds(),
                }
            )
            return list(self.current_loadout), blocked_reason

        self.current_loadout = list(decision.selected_loadout[:capacity])
        self.current_loadout_reason = str(decision.loadout_reason or "")
        self.current_selectable_seeds = list(decision.selectable_seeds)
        self.current_excluded_new_plants = list(decision.excluded_new_plants)
        self._apply_loadout(self.current_loadout)
        self.context.update(
            {
                "selected_seeds": list(self.current_loadout),
                "selected_loadout_count": len(self.current_loadout),
                "active_seed_slot_count": len(self.current_loadout),
                "inactive_seed_slot_count": max(0, self.max_seed_slots - len(self.current_loadout)),
                "inactive_model_slots": max(0, self.max_seed_slots - len(self.current_loadout)),
                "observed_seed_bank_capacity": decision.observed_seed_bank_capacity,
                "active_seed_slot_capacity": decision.observed_seed_bank_capacity,
                "current_seed_bank_capacity": decision.observed_seed_bank_capacity,
                "selectable_seeds": list(decision.selectable_seeds),
                "eligible_seeds": list(decision.eligible_seeds),
                "unlocked_seeds": self.curriculum.unlocked_seeds(),
                "loadout_reason": self.current_loadout_reason,
                "excluded_new_plants": list(self.current_excluded_new_plants),
            }
        )
        return list(self.current_loadout), ""

    def _apply_loadout(self, loadout: List[str]) -> None:
        capacity = _clamp_capacity(self.observed_seed_bank_capacity, maximum=self.max_seed_slots)
        effective_loadout = [str(seed).strip() for seed in loadout if str(seed).strip()][:capacity]
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
            "selected_loadout_count": int(progress.selected_loadout_count),
            "active_seed_slot_count": int(progress.selected_loadout_count),
            "inactive_seed_slot_count": max(0, self.max_seed_slots - int(progress.selected_loadout_count)),
            "inactive_model_slots": max(0, self.max_seed_slots - int(progress.selected_loadout_count)),
            "selected_loadout": list(progress.selected_loadout),
            "loadout_reason": progress.loadout_reason,
            "selectable_seeds": list(progress.selectable_seeds),
            "eligible_seeds": list(progress.eligible_seeds),
            "excluded_new_plants": list(progress.excluded_new_plants),
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
        unlock_payload = {
            "updated_at": time.time(),
            "unlocked_seeds": self.curriculum.unlocked_seeds(),
            "eligible_seeds": self.curriculum.eligible_seeds(),
            "selectable_seeds": list(self.current_selectable_seeds),
            "unlock_episode": dict(sorted(self.curriculum.unlock_episode.items())),
            "loadout_reason": self.current_loadout_reason,
            "excluded_new_plants": list(self.current_excluded_new_plants),
            "frontier_win_streak": int(self.frontier_win_streak),
            "frontier_win_streak_required": int(self.frontier_win_streak_required),
            "frontier_mastered_levels": list(self.frontier_mastered_levels),
        }
        self.plant_unlocks_path.write_text(json.dumps(unlock_payload, indent=2), encoding="utf-8")
        slot_payload = {
            "updated_at": time.time(),
            "max_seed_slots": self.max_seed_slots,
            "observed_seed_bank_capacity": self.observed_seed_bank_capacity,
            "active_seed_slot_capacity": self.observed_seed_bank_capacity,
            "current_seed_bank_capacity": self.observed_seed_bank_capacity,
            "selected_loadout_count": selected_count,
            "active_seed_slot_count": selected_count,
            "inactive_seed_slot_count": max(0, self.max_seed_slots - selected_count),
            "inactive_model_slots": max(0, self.max_seed_slots - selected_count),
            "selected_loadout": list(self.current_loadout),
            "loadout_reason": self.current_loadout_reason,
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
    return _ordered_by_priority(values)


def _selectable_from_seed_screen_state(state: Dict[str, Any]) -> List[str]:
    values: List[str] = []
    for key in ("visibleSeedCardNames", "availableSeedNames", "selectedSeedNames", "unlockedSeedNames"):
        raw = state.get(key, [])
        if isinstance(raw, list):
            values.extend(str(item) for item in raw if str(item).strip())
    return _ordered_by_priority(values)


def _infer_seed_bank_capacity_from_state(
    state: Dict[str, Any],
    *,
    context: Dict[str, Any],
    previous_capacity: int,
    selected_loadout: List[str],
) -> int:
    explicit_override = _first_int_from_values(
        [
            context.get("observed_seed_bank_capacity_override"),
            context.get("seed_bank_capacity_override"),
            context.get("active_seed_slot_capacity_override"),
        ]
    )
    if explicit_override is not None:
        return _clamp_capacity(explicit_override, maximum=SEED_CAPACITY_MAX)

    probe_candidates: List[int] = []
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
            probe_candidates.append(value)
    for key in ("visibleSeedCardNames", "availableSeedNames", "seedSelectionCards", "seedCards"):
        raw = state.get(key, [])
        if isinstance(raw, list) and raw:
            probe_candidates.append(len(raw))
    if probe_candidates:
        return _clamp_capacity(max(probe_candidates), maximum=SEED_CAPACITY_MAX)

    gameplay_candidates: List[int] = []
    for key in ("seedSlots",):
        raw = state.get(key, [])
        if isinstance(raw, list) and raw:
            gameplay_candidates.append(len(raw))
    if gameplay_candidates:
        return _clamp_capacity(max(gameplay_candidates), maximum=SEED_CAPACITY_MAX)

    fallback = _first_int_from_values(
        [
            previous_capacity,
            context.get("observed_seed_bank_capacity"),
            context.get("active_seed_slot_capacity"),
            context.get("current_seed_bank_capacity"),
            len(selected_loadout),
        ]
    )
    return _clamp_capacity(fallback or 1, maximum=SEED_CAPACITY_MAX)


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


def _ordered_by_priority(values: Iterable[str]) -> List[str]:
    seen = {str(value).strip() for value in values if str(value).strip()}
    output: List[str] = []
    for seed in SEED_PRIORITY:
        if seed in seen:
            output.append(seed)
            seen.remove(seed)
    output.extend(sorted(seen))
    return output


def _priority_index(seed: str) -> int:
    seed_name = str(seed or "").strip()
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


def parse_initial_loadout(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(seed).strip() for seed in value if str(seed).strip()]
    text = str(value or "").strip()
    return parse_seed_list(text) if text else list(ADVENTURE_GENERALIST_INITIAL_LOADOUT)
