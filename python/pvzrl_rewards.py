"""Pure per-step reward composition for PvZRL.

The environment owns bridge and episode side effects.  This module owns the
reward schema, reward configuration, immutable reward state, and deterministic
composition.  Every calculation accepts explicit observations/state and
returns the next state; it never mutates bridge observations or caller-owned
action results.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from pvzrl_fusion import (
    BUCKETHEAD_TYPES,
    CONEHEAD_TYPES,
    FUSION_ILLEGAL_INCOMPATIBLE,
    fusion_tier,
    normalize_fusion_source,
    plant_name as fusion_plant_name,
)
from pvzrl_observation_facts import (
    StepFacts,
    ZombieFact,
    build_step_facts,
    safe_float as _safe_float,
    safe_int as _safe_int,
)


REWARD_COMPONENT_FIELDS = (
    "kill_reward",
    "wave_reward",
    "win_loss_reward",
    "illegal_penalty",
    "mower_loss_penalty",
    "danger_delta_reward",
    "undefended_threat_penalty",
    "lane_response_reward",
    "plant_health_loss_penalty",
    "threat_balanced_row_reward",
    "overdefended_row_penalty",
    "role_positioning_reward",
    "first_peashooter_in_row_reward",
    "first_defense_undefended_threatened_row_reward",
    "all_rows_peashooter_coverage_reward",
    "sunflower_overbuild_before_defense_penalty",
    "defense_before_extra_economy_reward",
    "sunflower_while_undefended_threat_penalty",
    "plant_elsewhere_while_undefended_threat_penalty",
    "late_undefended_threat_penalty",
    "reduce_undefended_threat_reward",
    "wait_while_actionable_threat_penalty",
    "first_peashooter_threatened_row_reward",
    "all_active_threatened_rows_have_peashooter_reward",
    "sunflower_greed_while_defense_missing_penalty",
    "early_sunflower_reward",
    "safe_sunflower_position_reward",
    "sunflower_overbuild_penalty",
    "economy_collapse_penalty",
    "first_defense_in_threatened_row_reward",
    "threatened_lane_coverage_reward",
    "row_balance_reward",
    "useful_peashooter_position_reward",
    "overdefense_penalty",
    "wallnut_blocks_active_threat_reward",
    "wallnut_low_value_placement_penalty",
    "wallnut_threatened_lane_reward",
    "wallnut_between_zombie_and_house_reward",
    "wallnut_frontline_reward",
    "wallnut_emergency_block_reward",
    "wallnut_useless_penalty",
    "cherrybomb_tactical_kill_reward",
    "cherrybomb_wasted_penalty",
    "cherrybomb_kill_reward",
    "cherrybomb_heavy_zombie_bonus",
    "cherrybomb_cluster_bonus",
    "cherrybomb_emergency_reward",
    "cherrybomb_zero_kill_penalty",
    "cherrybomb_low_value_penalty",
    "mower_risk_reduction_reward",
    "tough_zombie_response_reward",
    "row_danger_delta_reward",
    "high_danger_unanswered_penalty",
    "mower_exposure_penalty",
    "minimum_viable_defense_reward",
    "coach_match_reward",
    "coach_legal_execution_reward",
    "coach_override_penalty",
    "coach_fusion_success_reward",
    "coach_tactical_usefulness_reward",
    "fusion_reward",
)
REWARD_EPISODE_TOTAL_FIELDS = tuple(f"{field}_total" for field in REWARD_COMPONENT_FIELDS)

FUSION_REWARD_COMPONENT_NAMES = (
    "fusion_attempt_reward",
    "fusion_success_reward",
    "fusion_new_recipe_reward",
    "fusion_recursive_reward",
    "fusion_tier_reward",
    "fusion_repeat_decay",
    "fusion_threatened_row_bonus",
    "fusion_active_wave_bonus",
    "fusion_defensive_value_bonus",
    "fusion_incompatible_penalty",
    "fusion_empty_tile_penalty",
    "fusion_failed_penalty",
    "fusion_bridge_error_penalty",
    "fusion_spam_penalty",
)


@dataclass
class RewardConfig:
    kill_reward: float = 1.0
    wave_reward: float = 2.0
    plant_health_loss_penalty: float = 0.002
    illegal_action_penalty: float = 0.15
    mower_loss_penalty: float = 1.25
    danger_delta_scale: float = 0.01
    lane_response_reward: float = 0.45
    undefended_close_threat_penalty: float = 0.02
    close_threat_threshold: float = 0.6
    threat_balanced_row_reward: float = 0.5
    threat_balanced_zero_defender_bonus: float = 0.25
    overdefended_row_penalty: float = 0.2
    role_positioning_reward: float = 0.25
    first_peashooter_in_row_reward: float = 0.75
    first_defense_undefended_threatened_row_reward: float = 1.25
    all_rows_peashooter_coverage_reward: float = 3.0
    sunflower_overbuild_before_defense_penalty: float = 0.2
    defense_before_extra_economy_reward: float = 0.5
    sunflower_while_undefended_threat_penalty: float = 0.45
    plant_elsewhere_while_undefended_threat_penalty: float = 0.25
    undefended_threat_grace_steps: float = 40.0
    late_undefended_threat_penalty: float = 0.03
    reduce_undefended_threat_reward: float = 1.0
    wait_while_actionable_threat_penalty: float = 0.05
    first_peashooter_threatened_row_reward: float = 0.35
    all_active_threatened_rows_have_peashooter_reward: float = 0.15
    sunflower_greed_while_defense_missing_penalty: float = 0.15
    early_sunflower_reward: float = 0.0
    safe_sunflower_position_reward: float = 0.0
    sunflower_overbuild_penalty: float = 0.0
    economy_collapse_penalty: float = 0.0
    first_defense_in_threatened_row_reward: float = 0.0
    threatened_lane_coverage_reward: float = 0.0
    row_balance_reward: float = 0.0
    useful_peashooter_position_reward: float = 0.0
    overdefense_penalty: float = 0.0
    wallnut_blocks_active_threat_reward: float = 0.25
    wallnut_low_value_placement_penalty: float = 0.06
    wallnut_threatened_lane_reward: float = 0.0
    wallnut_between_zombie_and_house_reward: float = 0.0
    wallnut_frontline_reward: float = 0.0
    wallnut_emergency_block_reward: float = 0.0
    wallnut_useless_penalty: float = 0.0
    cherrybomb_tactical_kill_reward: float = 0.5
    cherrybomb_tough_bonus_reward: float = 0.75
    cherrybomb_mower_save_bonus_reward: float = 0.25
    cherrybomb_wasted_penalty: float = 0.35
    cherrybomb_kill_reward: float = 0.0
    cherrybomb_heavy_zombie_bonus: float = 0.0
    cherrybomb_cluster_bonus: float = 0.0
    cherrybomb_emergency_reward: float = 0.0
    cherrybomb_zero_kill_penalty: float = 0.0
    cherrybomb_low_value_penalty: float = 0.0
    mower_risk_reduction_reward: float = 0.15
    tough_zombie_response_reward: float = 0.15
    row_danger_delta_reward: float = 0.0
    high_danger_unanswered_penalty: float = 0.0
    mower_exposure_penalty: float = 0.0
    minimum_viable_defense_reward: float = 0.0
    coach_match_reward: float = 0.02
    coach_legal_execution_reward: float = 0.01
    coach_override_penalty: float = -0.01
    coach_fusion_success_reward: float = 0.03
    coach_tactical_usefulness_reward: float = 0.01
    fusion_attempt_reward: float = 0.02
    fusion_success_reward: float = 0.50
    fusion_new_recipe_reward: float = 0.15
    fusion_recursive_reward: float = 0.20
    fusion_tier2_reward: float = 0.10
    fusion_tier3_reward: float = 0.25
    fusion_repeat_reward_multiplier: float = 0.25
    fusion_threatened_row_bonus: float = 0.15
    fusion_active_wave_bonus: float = 0.10
    fusion_defensive_value_bonus: float = 0.10
    fusion_incompatible_penalty: float = -0.10
    fusion_empty_tile_penalty: float = -0.08
    fusion_failed_penalty: float = -0.10
    fusion_bridge_error_penalty: float = -0.25
    fusion_spam_penalty: float = -0.05
    max_fusion_reward_per_episode: float = 3.0
    # Compatibility-only: absolute proximity punishment is intentionally unused.
    proximity_penalty: float = 0.01
    win_reward: float = 10.0
    loss_penalty: float = 10.0


@dataclass(frozen=True, slots=True)
class RewardAction:
    kind: str = "wait"
    plant_type: int = -1
    row: int = -1
    column: int = -1
    plant_placed: bool = False
    illegal: bool = False


def reward_action_from_result(action_result: Optional[Mapping[str, Any]]) -> RewardAction:
    if not isinstance(action_result, Mapping):
        return RewardAction()
    decoded = action_result.get("decoded") if isinstance(action_result.get("decoded"), Mapping) else {}
    placement = action_result.get("placement") if isinstance(action_result.get("placement"), Mapping) else {}
    kind = str(decoded.get("kind") or ("plant" if placement else "wait"))
    illegal = bool(action_result.get("illegalAction"))
    if kind == "fusion":
        return RewardAction(
            kind="fusion",
            plant_type=_safe_int(
                decoded.get("resultPlantType"),
                action_result.get("predictedResultType"),
                placement.get("plantType"),
                default=-1,
            ),
            row=_safe_int(decoded.get("row"), action_result.get("sourceRow"), default=-1),
            column=_safe_int(decoded.get("column"), action_result.get("sourceCol"), default=-1),
            plant_placed=bool(action_result.get("fusionSucceeded") or placement.get("success")),
            illegal=illegal,
        )
    return RewardAction(
        kind=kind,
        plant_type=_safe_int(decoded.get("plantType"), placement.get("plantType"), default=-1),
        row=_safe_int(decoded.get("row"), placement.get("row"), default=-1),
        column=_safe_int(decoded.get("column"), placement.get("column"), default=-1),
        plant_placed=bool(
            placement.get("success")
            or action_result.get("plantPlaced")
            or placement.get("plantPlaced")
        ),
        illegal=illegal,
    )


@dataclass(frozen=True, slots=True)
class PendingCherryEvent:
    row: int
    column: int
    age: int = 0
    kills: int = 0
    nearby_tough: int = 0
    nearby_buckethead: int = 0
    nearby_conehead: int = 0
    mower_risk: bool = False
    credited: bool = False

@dataclass(frozen=True, slots=True)
class FusionRewardState:
    reward_total: float = 0.0
    positive_total: float = 0.0
    capped: bool = False
    component_totals: Tuple[float, ...] = field(
        default_factory=lambda: tuple(0.0 for _ in FUSION_REWARD_COMPONENT_NAMES)
    )
    last_reward_delta: float = 0.0
    last_reward_reason: str = ""
    last_usefulness_bonus: float = 0.0
    last_source: str = ""
    recent_attempts: Tuple[Tuple[int, int, int, str, int], ...] = ()
    event_counter: int = 0
    accounted_event_ids: frozenset[str] = frozenset()
    recipes_seen_episode: frozenset[str] = frozenset()
    recipe_counts_episode: Tuple[Tuple[str, int], ...] = ()
    recipes_seen_run: frozenset[str] = frozenset()

    def component_dict(self) -> Dict[str, float]:
        return {
            name: float(self.component_totals[index])
            for index, name in enumerate(FUSION_REWARD_COMPONENT_NAMES)
        }

    def recipe_count_dict(self) -> Dict[str, int]:
        return {str(key): int(value) for key, value in self.recipe_counts_episode}


@dataclass(frozen=True, slots=True)
class RewardCompositionState:
    pending_cherry_events: Tuple[PendingCherryEvent, ...] = ()
    all_rows_peashooter_coverage_rewarded: bool = False
    all_active_threatened_rows_coverage_rewarded: bool = False
    undefended_threat_age_by_row: Tuple[int, ...] = ()
    max_undefended_threat_age_by_row: Tuple[int, ...] = ()
    undefended_threat_age_sum_by_row: Tuple[int, ...] = ()
    undefended_threat_age_count_by_row: Tuple[int, ...] = ()
    fusion: FusionRewardState = field(default_factory=FusionRewardState)

    @classmethod
    def initial(
        cls,
        rows: int,
        *,
        recipes_seen_run: Iterable[str] = (),
    ) -> "RewardCompositionState":
        zeroes = tuple(0 for _ in range(max(0, int(rows))))
        return cls(
            undefended_threat_age_by_row=zeroes,
            max_undefended_threat_age_by_row=zeroes,
            undefended_threat_age_sum_by_row=zeroes,
            undefended_threat_age_count_by_row=zeroes,
            fusion=FusionRewardState(recipes_seen_run=frozenset(str(value) for value in recipes_seen_run)),
        )


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    """Exact public component schema plus explicit exceptional adjustments."""

    components: Tuple[float, ...] = field(
        default_factory=lambda: tuple(0.0 for _ in REWARD_COMPONENT_FIELDS)
    )
    adjustments: Tuple[Tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if len(self.components) != len(REWARD_COMPONENT_FIELDS):
            raise ValueError(
                f"reward component count mismatch: {len(self.components)} != {len(REWARD_COMPONENT_FIELDS)}"
            )

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        adjustments: Optional[Mapping[str, Any]] = None,
    ) -> "RewardBreakdown":
        return cls(
            components=tuple(_safe_float(values.get(name), default=0.0) for name in REWARD_COMPONENT_FIELDS),
            adjustments=tuple(
                (str(name), _safe_float(value, default=0.0))
                for name, value in (adjustments or {}).items()
                if str(name) not in REWARD_COMPONENT_FIELDS and str(name) != "reward_total"
            ),
        )

    @property
    def reward_total(self) -> float:
        return sum(self.components) + sum(value for _name, value in self.adjustments)

    def component(self, name: str) -> float:
        return float(self.components[REWARD_COMPONENT_FIELDS.index(str(name))])

    def with_components(
        self,
        values: Mapping[str, Any],
        *,
        replace_values: bool = False,
    ) -> "RewardBreakdown":
        components = list(self.components)
        for name, raw_value in values.items():
            if name not in REWARD_COMPONENT_FIELDS:
                continue
            index = REWARD_COMPONENT_FIELDS.index(name)
            value = _safe_float(raw_value, default=0.0)
            components[index] = value if replace_values else components[index] + value
        return replace(self, components=tuple(components))

    def with_adjustment(self, name: str, value: float) -> "RewardBreakdown":
        adjustment_name = str(name)
        if adjustment_name in REWARD_COMPONENT_FIELDS or adjustment_name == "reward_total":
            raise ValueError(f"{adjustment_name!r} is not an exceptional adjustment field")
        adjustments = dict(self.adjustments)
        adjustments[adjustment_name] = float(value)
        return replace(self, adjustments=tuple(adjustments.items()))

    def to_dict(self) -> Dict[str, float]:
        payload = {
            name: float(self.components[index])
            for index, name in enumerate(REWARD_COMPONENT_FIELDS)
        }
        payload.update({name: float(value) for name, value in self.adjustments})
        payload["reward_total"] = float(self.reward_total)
        return payload


@dataclass(frozen=True, slots=True)
class RewardComposition:
    breakdown: RewardBreakdown
    state: RewardCompositionState
    event_diagnostics: Tuple[Tuple[str, Any], ...] = ()
    action_result_annotations: Tuple[Tuple[str, Any], ...] = ()
    fusion_reward_delta: float = 0.0

    def diagnostics_dict(self) -> Dict[str, Any]:
        return dict(self.event_diagnostics)

    def annotations_dict(self) -> Dict[str, Any]:
        return dict(self.action_result_annotations)


def merge_reward_components(
    breakdown: Mapping[str, Any],
    components: Mapping[str, Any],
) -> Dict[str, float]:
    """Add per-step components once and recompute total from authoritative fields.

    This is the compatibility adapter for coach/wrapper contributors.  Existing
    exceptional adjustment keys are retained and remain part of ``reward_total``.
    """

    adjustments = {
        str(key): _safe_float(value, default=0.0)
        for key, value in breakdown.items()
        if key not in REWARD_COMPONENT_FIELDS and key != "reward_total"
    }
    composed = RewardBreakdown.from_mapping(breakdown, adjustments=adjustments).with_components(components)
    payload = composed.to_dict()
    if not all(name in breakdown for name in REWARD_COMPONENT_FIELDS) and "reward_total" in breakdown:
        # Lightweight coach/test callers historically provide only a total.
        # Preserve that compatibility while complete environment breakdowns
        # use the strict component-sum invariant above.
        payload["reward_total"] = _safe_float(
            breakdown.get("reward_total"), default=0.0
        ) + sum(_safe_float(value, default=0.0) for value in components.values())
    return payload


def _row_count(facts: StepFacts, fallback_rows: int) -> int:
    return max(0, int(facts.rows or fallback_rows))


def _plant_counts_by_row(
    facts: StepFacts,
    rows: int,
    plant_type: Optional[int] = None,
) -> Dict[int, int]:
    if plant_type is None:
        return {
            row: int(facts.plant_count_by_lane.get(row, 0))
            for row in range(max(0, rows))
        }
    resolved_type = int(plant_type)
    return {
        row: int(facts.plant_count_by_type_and_lane.get((resolved_type, row), 0))
        for row in range(max(0, rows))
    }


def _lane_zombie_counts(facts: StepFacts, rows: int) -> Dict[int, int]:
    return {
        row: max(0, int(facts.lane_by_row[row].zombie_count)) if row in facts.lane_by_row else 0
        for row in range(max(0, rows))
    }


def _lane_danger_by_row(facts: StepFacts, rows: int) -> Dict[int, float]:
    return {
        row: max(0.0, float(facts.lane_by_row[row].danger)) if row in facts.lane_by_row else 0.0
        for row in range(max(0, rows))
    }


def _nearest_zombie_x_by_row(facts: StepFacts, rows: int) -> Dict[int, Optional[float]]:
    return {
        row: facts.lane_by_row[row].nearest_zombie_x if row in facts.lane_by_row else None
        for row in range(max(0, rows))
    }


def _active_threat_rows(facts: StepFacts, rows: int) -> list[int]:
    return [row for row, count in _lane_zombie_counts(facts, rows).items() if count > 0]


def _wallnut_blocker_count(facts: StepFacts, row: int) -> int:
    return sum(1 for plant in facts.plants_by_lane.get(int(row), ()) if plant.plant_type == 3)


def _meaningful_defender_counts(facts: StepFacts, rows: int) -> Dict[int, int]:
    counts = {row: 0 for row in range(max(0, rows))}
    for plant in facts.plants:
        if plant.row not in counts:
            continue
        name = plant.type_name.lower()
        if plant.plant_type in {0, 3, 1030, 1032} or any(
            token in name for token in ("pea", "shoot", "gatling", "repeater", "nut")
        ):
            counts[plant.row] += 1
    return counts


def _is_buckethead(zombie: ZombieFact) -> bool:
    name = zombie.type_name.lower()
    return zombie.zombie_type in BUCKETHEAD_TYPES or "bucket" in name or "铁桶" in name


def _is_conehead(zombie: ZombieFact) -> bool:
    name = zombie.type_name.lower()
    return (
        zombie.zombie_type in CONEHEAD_TYPES
        or "cone" in name
        or "roadblock" in name
        or "路障" in name
    )


def _is_tough(zombie: ZombieFact) -> bool:
    return (
        _is_buckethead(zombie)
        or _is_conehead(zombie)
        or int(zombie.health) >= 600
        or int(zombie.max_health) >= 600
    )


def _tough_zombies_by_row(facts: StepFacts, rows: int) -> Dict[int, Dict[str, int]]:
    counts = {
        row: {"buckethead": 0, "conehead": 0, "tough": 0}
        for row in range(max(0, rows))
    }
    for row, lane in facts.lane_by_row.items():
        if row not in counts:
            continue
        counts[row] = {
            "buckethead": int(lane.buckethead_count),
            "conehead": int(lane.conehead_count),
            "tough": int(lane.tough_zombie_count),
        }
    for zombie in facts.alive_zombies:
        if zombie.row not in counts:
            continue
        if _is_buckethead(zombie):
            counts[zombie.row]["buckethead"] += 1
        if _is_conehead(zombie):
            counts[zombie.row]["conehead"] += 1
        if _is_tough(zombie):
            counts[zombie.row]["tough"] += 1
    return counts


def _mower_risk_rows(facts: StepFacts, rows: int) -> list[int]:
    active_rows = facts.mower.active_rows
    candidates = set(range(rows)) if active_rows is None else set(active_rows)
    danger = _lane_danger_by_row(facts, rows)
    nearest = _nearest_zombie_x_by_row(facts, rows)
    defenders = _meaningful_defender_counts(facts, rows)
    zombie_counts = _lane_zombie_counts(facts, rows)
    result: list[int] = []
    for row in candidates:
        if zombie_counts.get(row, 0) <= 0:
            continue
        close = nearest.get(row)
        if danger.get(row, 0.0) >= 0.65 or (close is not None and close <= 2.0) or defenders.get(row, 0) == 0:
            result.append(row)
    return result


def _actionable_threat_rows(facts: StepFacts, rows: int) -> list[int]:
    defenders = _meaningful_defender_counts(facts, rows)
    ready = set(facts.ready_seed_types)
    tough = _tough_zombies_by_row(facts, rows)
    zombie_counts = _lane_zombie_counts(facts, rows)
    result: list[int] = []
    for row in _active_threat_rows(facts, rows):
        if defenders.get(row, 0) > 0:
            continue
        useful_ready = 0 in ready or 3 in ready
        if 2 in ready and (zombie_counts.get(row, 0) >= 2 or tough.get(row, {}).get("tough", 0) > 0):
            useful_ready = True
        if useful_ready and facts.lifecycle.gameplay_ready:
            result.append(row)
    return result


def _nearby_zombie_context(
    facts: StepFacts,
    row: int,
    column: int,
    *,
    radius: float,
) -> Dict[str, int]:
    context = {"zombies": 0, "buckethead": 0, "conehead": 0, "tough": 0}
    for zombie in facts.alive_zombies:
        if abs(zombie.row - row) > 1:
            continue
        zombie_x = zombie.x if zombie.has_position else float(column)
        if abs(zombie_x - float(column)) > radius and zombie.row != row:
            continue
        if zombie.row == row and zombie_x - float(column) > 5.0:
            continue
        context["zombies"] += 1
        context["buckethead"] += int(_is_buckethead(zombie))
        context["conehead"] += int(_is_conehead(zombie))
        context["tough"] += int(_is_tough(zombie))
    return context


def _valuable_plant_columns(facts: StepFacts, row: int) -> list[int]:
    return [
        plant.column
        for plant in facts.plants_by_lane.get(int(row), ())
        if plant.plant_type in {0, 1, 1030, 1032, 1033} and plant.column >= 0
    ]


def _role_positioning_reward(
    facts: StepFacts,
    rows: int,
    columns: int,
    plant_type: int,
    row: int,
    column: int,
    config: RewardConfig,
) -> float:
    maximum = max(0.0, float(config.role_positioning_reward))
    if maximum <= 0.0 or row < 0 or column < 0 or not 0 <= row < rows:
        return 0.0
    danger = _lane_danger_by_row(facts, rows)
    zombie_counts = _lane_zombie_counts(facts, rows)
    if int(plant_type) == 1:
        row_danger = danger.get(row, 0.0)
        if zombie_counts.get(row, 0) > 0 or row_danger >= 0.3:
            if column >= 3 and row_danger >= config.close_threat_threshold:
                return -min(0.1, maximum * 0.4)
            return 0.0
        if column <= 2:
            return maximum
        if column <= 3:
            return maximum * 0.6
        return 0.0
    if int(plant_type) == 0:
        if zombie_counts.get(row, 0) <= 0:
            return 0.0
        reward = maximum * 0.55
        nearest = _nearest_zombie_x_by_row(facts, rows).get(row)
        if nearest is not None:
            firing_distance = float(nearest) - float(column)
            if firing_distance >= 3.0:
                reward += maximum * 0.45
            elif firing_distance >= 1.5:
                reward += maximum * 0.25
            elif firing_distance < 0.5:
                reward -= maximum * 0.3
        elif column <= max(0, columns - 4):
            reward += maximum * 0.25
        if column >= max(0, columns - 2):
            reward -= maximum * 0.3
        return max(-maximum * 0.5, min(maximum, reward))
    return 0.0


def _legal_peashooter_actions_by_row(
    facts: StepFacts,
    rows: int,
    columns: int,
    legal_actions: Sequence[int],
) -> Dict[int, int]:
    counts = {row: 0 for row in range(max(0, rows))}
    cells = rows * columns
    if cells <= 0:
        return counts
    for raw_action in legal_actions:
        action = _safe_int(raw_action, default=0)
        if action <= 0:
            continue
        encoded = action - 1
        slot_index = encoded // cells
        cell = encoded % cells
        slot = facts.seed_slot(slot_index)
        if slot is None or slot.plant_type != 0:
            continue
        row = cell // columns
        if row in counts:
            counts[row] += 1
    return counts


def _is_restart_screen(observation: Mapping[str, Any]) -> bool:
    if observation.get("onPauseMenu") or observation.get("pauseMenuActive"):
        return False
    return bool(
        observation.get("onGameOverScreen")
        or observation.get("lossMenuActive")
        or (observation.get("onRestartScreen") and observation.get("gameOverTextVisible"))
    )


def _advance_pending_cherries(
    events: Sequence[PendingCherryEvent],
    kill_delta: int,
    config: RewardConfig,
) -> Tuple[Tuple[PendingCherryEvent, ...], float, float, Dict[str, int]]:
    reward = 0.0
    wasted_penalty = 0.0
    diagnostics = {"kills": 0, "zero_kill": 0, "buckethead": 0, "conehead": 0}
    active: list[PendingCherryEvent] = []
    for original in events:
        event = replace(original, age=original.age + 1)
        if kill_delta > 0 and not event.credited:
            event = replace(event, kills=event.kills + kill_delta)
            diagnostics["kills"] += kill_delta
            base = float(config.cherrybomb_tactical_kill_reward)
            if event.nearby_tough > 0 or event.nearby_buckethead > 0 or event.nearby_conehead > 0:
                base += float(config.cherrybomb_tough_bonus_reward)
                diagnostics["buckethead"] += int(event.nearby_buckethead > 0)
                diagnostics["conehead"] += int(event.nearby_conehead > 0)
            if event.mower_risk:
                base += float(config.cherrybomb_mower_save_bonus_reward)
            if event.kills >= 2 or base > float(config.cherrybomb_tactical_kill_reward):
                reward += base
                event = replace(event, credited=True)
        if event.age > 80:
            if event.kills <= 0:
                wasted_penalty -= float(config.cherrybomb_wasted_penalty)
                diagnostics["zero_kill"] += 1
            continue
        active.append(event)
    return tuple(active), reward, wasted_penalty, diagnostics


def _updated_threat_ages(
    state: RewardCompositionState,
    rows: int,
    threatened_rows: Sequence[int],
    shooter_counts: Mapping[int, int],
) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    def resized(values: Sequence[int]) -> list[int]:
        result = [int(value) for value in values[:rows]]
        result.extend(0 for _ in range(rows - len(result)))
        return result

    ages = resized(state.undefended_threat_age_by_row)
    maxima = resized(state.max_undefended_threat_age_by_row)
    sums = resized(state.undefended_threat_age_sum_by_row)
    counts = resized(state.undefended_threat_age_count_by_row)
    threatened = set(int(row) for row in threatened_rows)
    for row in range(rows):
        if row in threatened and shooter_counts.get(row, 0) == 0:
            ages[row] += 1
            maxima[row] = max(maxima[row], ages[row])
            sums[row] += ages[row]
            counts[row] += 1
        else:
            ages[row] = 0
    return tuple(ages), tuple(maxima), tuple(sums), tuple(counts)


def compose_environment_reward(
    previous: Optional[Mapping[str, Any]],
    current: Mapping[str, Any],
    action_result: Optional[Mapping[str, Any]],
    *,
    config: RewardConfig,
    state: RewardCompositionState,
    plant_types: Sequence[int] = (),
    fallback_rows: int = 5,
    fallback_columns: int = 10,
    previous_legal_actions: Sequence[int] = (),
    previous_facts: Optional[StepFacts] = None,
    current_facts: Optional[StepFacts] = None,
) -> RewardComposition:
    """Compose ordinary environment components and return the next state."""

    if previous is None:
        return RewardComposition(RewardBreakdown(), state)
    prior_facts = previous_facts or build_step_facts(previous, plant_types)
    next_facts = current_facts or build_step_facts(current, plant_types)
    rows = max(_row_count(prior_facts, fallback_rows), _row_count(next_facts, fallback_rows))
    columns = max(1, int(prior_facts.columns or fallback_columns))
    components = {name: 0.0 for name in REWARD_COMPONENT_FIELDS}
    action = reward_action_from_result(action_result)

    kill_delta = max(0, _safe_int(current.get("killCount")) - _safe_int(previous.get("killCount")))
    components["kill_reward"] = kill_delta * config.kill_reward
    pending, delayed_reward, delayed_waste, cherry_diag = _advance_pending_cherries(
        state.pending_cherry_events,
        kill_delta,
        config,
    )
    components["cherrybomb_tactical_kill_reward"] += delayed_reward
    components["cherrybomb_wasted_penalty"] += delayed_waste
    components["cherrybomb_kill_reward"] += cherry_diag["kills"] * config.cherrybomb_kill_reward
    components["cherrybomb_heavy_zombie_bonus"] += (
        cherry_diag["buckethead"] + cherry_diag["conehead"]
    ) * config.cherrybomb_heavy_zombie_bonus
    components["cherrybomb_zero_kill_penalty"] += (
        -cherry_diag["zero_kill"] * config.cherrybomb_zero_kill_penalty
    )
    components["wave_reward"] = max(
        0,
        _safe_int(current.get("wave")) - _safe_int(previous.get("wave")),
    ) * config.wave_reward
    components["plant_health_loss_penalty"] = -max(
        0,
        _safe_int(previous.get("totalPlantHealth")) - _safe_int(current.get("totalPlantHealth")),
    ) * config.plant_health_loss_penalty
    components["mower_loss_penalty"] = -max(
        0,
        prior_facts.mower.count - next_facts.mower.count,
    ) * config.mower_loss_penalty
    if action.illegal:
        components["illegal_penalty"] = -config.illegal_action_penalty

    danger_delta = next_facts.total_lane_danger - prior_facts.total_lane_danger
    if danger_delta > 0.0:
        components["danger_delta_reward"] = -danger_delta * config.danger_delta_scale
        components["row_danger_delta_reward"] = -danger_delta * config.row_danger_delta_reward
    elif danger_delta < 0.0:
        components["danger_delta_reward"] = abs(danger_delta) * config.danger_delta_scale
        components["row_danger_delta_reward"] = abs(danger_delta) * config.row_danger_delta_reward

    previous_shooters = _plant_counts_by_row(prior_facts, rows, 0)
    current_shooters = _plant_counts_by_row(next_facts, rows, 0)
    previous_sunflowers = _plant_counts_by_row(prior_facts, rows, 1)
    current_sunflowers = _plant_counts_by_row(next_facts, rows, 1)
    previous_threat_rows = _active_threat_rows(prior_facts, rows)
    current_threat_rows = _active_threat_rows(next_facts, rows)
    previous_zero_rows = [row for row in previous_threat_rows if previous_shooters.get(row, 0) == 0]
    current_zero_rows = [row for row in current_threat_rows if current_shooters.get(row, 0) == 0]

    if (
        action.plant_type == 0
        and action.plant_placed
        and not action.illegal
        and action.row >= 0
        and _lane_zombie_counts(prior_facts, rows).get(action.row, 0) > 0
        and previous_shooters.get(action.row, 0) == 0
    ):
        components["lane_response_reward"] = config.lane_response_reward
    undefended_close_rows = [
        row
        for row, count in _lane_zombie_counts(next_facts, rows).items()
        if count > 0
        and current_shooters.get(row, 0) == 0
        and _lane_danger_by_row(next_facts, rows).get(row, 0.0) >= config.close_threat_threshold
    ]
    components["undefended_threat_penalty"] = (
        -len(undefended_close_rows) * config.undefended_close_threat_penalty
    )

    previous_legal_peas = (
        _legal_peashooter_actions_by_row(
            prior_facts,
            rows,
            columns,
            previous_legal_actions,
        )
        if previous_zero_rows
        else {row: 0 for row in range(rows)}
    )
    emergency_pea_rows = [row for row in previous_zero_rows if previous_legal_peas.get(row, 0) > 0]
    placed_correct_emergency_pea = bool(
        action.plant_placed and not action.illegal and action.plant_type == 0 and action.row in emergency_pea_rows
    )
    actionable_rows = _actionable_threat_rows(prior_facts, rows)
    tough_by_row = _tough_zombies_by_row(prior_facts, rows)
    mower_risk_rows = _mower_risk_rows(prior_facts, rows)
    if action.kind == "wait" and prior_facts.lifecycle.gameplay_ready and actionable_rows:
        components["wait_while_actionable_threat_penalty"] = -config.wait_while_actionable_threat_penalty

    high_danger_rows = [
        row
        for row, danger in _lane_danger_by_row(next_facts, rows).items()
        if danger >= 0.65 and current_shooters.get(row, 0) == 0 and _wallnut_blocker_count(next_facts, row) <= 0
    ]
    if high_danger_rows:
        components["high_danger_unanswered_penalty"] = (
            -len(high_danger_rows) * config.high_danger_unanswered_penalty
        )
    mower_exposure_rows = [
        row
        for row in _mower_risk_rows(next_facts, rows)
        if current_shooters.get(row, 0) == 0 and _wallnut_blocker_count(next_facts, row) <= 0
    ]
    if mower_exposure_rows:
        components["mower_exposure_penalty"] = -len(mower_exposure_rows) * config.mower_exposure_penalty
    if current_threat_rows and all(current_shooters.get(row, 0) > 0 for row in current_threat_rows):
        components["minimum_viable_defense_reward"] = config.minimum_viable_defense_reward

    coverage_all = state.all_rows_peashooter_coverage_rewarded
    coverage_threat = state.all_active_threatened_rows_coverage_rewarded
    if action.plant_placed and not action.illegal and 0 <= action.row < rows:
        components["role_positioning_reward"] = _role_positioning_reward(
            prior_facts,
            rows,
            columns,
            action.plant_type,
            action.row,
            action.column,
            config,
        )
        if action.plant_type == 0:
            previous_row_shooters = previous_shooters.get(action.row, 0)
            if previous_row_shooters == 0 and current_shooters.get(action.row, 0) > 0:
                components["first_peashooter_in_row_reward"] = config.first_peashooter_in_row_reward
            if action.row in previous_zero_rows:
                components["first_defense_undefended_threatened_row_reward"] = (
                    config.first_defense_undefended_threatened_row_reward
                )
                components["first_defense_in_threatened_row_reward"] = config.first_defense_in_threatened_row_reward
            if action.row in previous_threat_rows and previous_row_shooters == 0:
                components["first_peashooter_threatened_row_reward"] = config.first_peashooter_threatened_row_reward
                components["threatened_lane_coverage_reward"] = config.threatened_lane_coverage_reward
            if previous_threat_rows:
                threatened_counts = {row: previous_shooters.get(row, 0) for row in previous_threat_rows}
                minimum = min(threatened_counts.values())
                if action.row in threatened_counts and previous_row_shooters == minimum:
                    components["threat_balanced_row_reward"] = config.threat_balanced_row_reward
                    components["row_balance_reward"] = config.row_balance_reward
                    if previous_row_shooters == 0:
                        components["threat_balanced_row_reward"] += config.threat_balanced_zero_defender_bonus
            if previous_row_shooters >= 2 and any(row != action.row for row in previous_zero_rows):
                components["overdefended_row_penalty"] = -config.overdefended_row_penalty
                components["overdefense_penalty"] = -config.overdefense_penalty
            useful_position = _role_positioning_reward(
                prior_facts,
                rows,
                columns,
                action.plant_type,
                action.row,
                action.column,
                config,
            )
            if useful_position > 0.0 and action.row in previous_threat_rows:
                components["useful_peashooter_position_reward"] = config.useful_peashooter_position_reward
            if sum(previous_sunflowers.values()) >= 4 and previous_row_shooters == 0 and previous_threat_rows:
                components["defense_before_extra_economy_reward"] = config.defense_before_extra_economy_reward
            if (
                rows > 0
                and not coverage_all
                and sum(1 for value in previous_shooters.values() if value > 0) < rows
                and sum(1 for value in current_shooters.values() if value > 0) >= rows
            ):
                components["all_rows_peashooter_coverage_reward"] = config.all_rows_peashooter_coverage_reward
                coverage_all = True
        elif action.plant_type == 1:
            total_previous_sunflowers = sum(previous_sunflowers.values())
            total_current_sunflowers = sum(current_sunflowers.values())
            row_danger = _lane_danger_by_row(prior_facts, rows).get(action.row, 0.0)
            if total_previous_sunflowers < 5 and action.column <= 2 and row_danger < config.close_threat_threshold:
                components["early_sunflower_reward"] = config.early_sunflower_reward
            if action.column <= 2 and row_danger < 0.3:
                components["safe_sunflower_position_reward"] = config.safe_sunflower_position_reward
            if total_current_sunflowers >= 5 and any(previous_shooters.get(row, 0) == 0 for row in range(rows)) and previous_threat_rows:
                components["sunflower_overbuild_before_defense_penalty"] = -config.sunflower_overbuild_before_defense_penalty
                components["sunflower_overbuild_penalty"] = -config.sunflower_overbuild_penalty
            if previous_zero_rows:
                components["sunflower_while_undefended_threat_penalty"] = -config.sunflower_while_undefended_threat_penalty
            if sum(previous_sunflowers.values()) >= 6 and previous_zero_rows and 0 in prior_facts.ready_seed_types:
                components["sunflower_greed_while_defense_missing_penalty"] = -config.sunflower_greed_while_defense_missing_penalty
        elif action.plant_type == 3:
            nearest = _nearest_zombie_x_by_row(prior_facts, rows).get(action.row)
            valuable_behind = any(column < action.column for column in _valuable_plant_columns(prior_facts, action.row))
            close_or_weak = action.row in previous_threat_rows and (
                nearest is None or float(nearest) - float(action.column) <= 4.0 or previous_shooters.get(action.row, 0) == 0
            )
            if close_or_weak and valuable_behind:
                components["wallnut_blocks_active_threat_reward"] = config.wallnut_blocks_active_threat_reward
                components["wallnut_frontline_reward"] = config.wallnut_frontline_reward
            if action.row in previous_threat_rows:
                components["wallnut_threatened_lane_reward"] = config.wallnut_threatened_lane_reward
            if action.row in previous_threat_rows and nearest is not None and float(action.column) < float(nearest):
                components["wallnut_between_zombie_and_house_reward"] = config.wallnut_between_zombie_and_house_reward
            if action.row in mower_risk_rows:
                components["wallnut_emergency_block_reward"] = config.wallnut_emergency_block_reward
            elif action.row not in previous_threat_rows or (nearest is not None and float(nearest) < float(action.column) - 0.5):
                components["wallnut_low_value_placement_penalty"] = -config.wallnut_low_value_placement_penalty
                components["wallnut_useless_penalty"] = -config.wallnut_useless_penalty
        elif action.plant_type == 2:
            nearby = _nearby_zombie_context(prior_facts, action.row, action.column, radius=2.75)
            under_threat = action.row in previous_threat_rows
            mower_risk = action.row in mower_risk_rows
            if nearby["zombies"] >= 2 or nearby["tough"] > 0 or mower_risk:
                components["cherrybomb_tactical_kill_reward"] += config.cherrybomb_tactical_kill_reward
                if nearby["zombies"] >= 2:
                    components["cherrybomb_cluster_bonus"] += config.cherrybomb_cluster_bonus
                if nearby["tough"] > 0 or nearby["buckethead"] > 0 or nearby["conehead"] > 0:
                    components["cherrybomb_tactical_kill_reward"] += config.cherrybomb_tough_bonus_reward
                    components["cherrybomb_heavy_zombie_bonus"] += config.cherrybomb_heavy_zombie_bonus
                if mower_risk:
                    components["cherrybomb_tactical_kill_reward"] += config.cherrybomb_mower_save_bonus_reward
                    components["cherrybomb_emergency_reward"] += config.cherrybomb_emergency_reward
            elif not under_threat:
                components["cherrybomb_wasted_penalty"] += -config.cherrybomb_wasted_penalty
                components["cherrybomb_low_value_penalty"] += -config.cherrybomb_low_value_penalty
            if nearby["zombies"] <= 0:
                components["cherrybomb_zero_kill_penalty"] += -config.cherrybomb_zero_kill_penalty
            pending = (*pending, PendingCherryEvent(
                row=action.row,
                column=action.column,
                nearby_tough=nearby["tough"],
                nearby_buckethead=nearby["buckethead"],
                nearby_conehead=nearby["conehead"],
                mower_risk=mower_risk,
            ))
        if action.row in mower_risk_rows and (action.plant_type in {0, 2, 3} or action.kind == "fusion"):
            components["mower_risk_reduction_reward"] = config.mower_risk_reduction_reward
        if tough_by_row.get(action.row, {}).get("tough", 0) > 0 and (action.plant_type in {0, 2, 3} or action.kind == "fusion"):
            multiplier = 0.5 if action.plant_type == 0 else 1.5 if action.plant_type == 2 else 1.0
            components["tough_zombie_response_reward"] = config.tough_zombie_response_reward * multiplier
        if emergency_pea_rows and not placed_correct_emergency_pea:
            components["plant_elsewhere_while_undefended_threat_penalty"] = -config.plant_elsewhere_while_undefended_threat_penalty
            if action.plant_type in {2, 3} and prior_facts.sun < 150:
                components["economy_collapse_penalty"] = -config.economy_collapse_penalty

    if (
        previous_threat_rows
        and not coverage_threat
        and any(previous_shooters.get(row, 0) == 0 for row in previous_threat_rows)
        and current_threat_rows
        and all(current_shooters.get(row, 0) > 0 for row in current_threat_rows)
    ):
        components["all_active_threatened_rows_have_peashooter_reward"] = (
            config.all_active_threatened_rows_have_peashooter_reward
        )
        coverage_threat = True
    if len(current_zero_rows) < len(previous_zero_rows):
        components["reduce_undefended_threat_reward"] = config.reduce_undefended_threat_reward

    ages, maxima, age_sums, age_counts = _updated_threat_ages(
        state,
        rows,
        current_threat_rows,
        current_shooters,
    )
    grace_steps = int(max(0, config.undefended_threat_grace_steps))
    late_count = sum(1 for row in range(rows) if ages[row] > grace_steps)
    if late_count:
        components["late_undefended_threat_penalty"] = -late_count * config.late_undefended_threat_penalty
    if bool(current.get("done")):
        terminal_hint = str(current.get("terminalHint") or "")
        if terminal_hint == "possible_win":
            components["win_loss_reward"] = config.win_reward
        elif terminal_hint == "game_over_or_loss" and _is_restart_screen(current):
            components["win_loss_reward"] = -config.loss_penalty

    next_state = replace(
        state,
        pending_cherry_events=tuple(pending),
        all_rows_peashooter_coverage_rewarded=coverage_all,
        all_active_threatened_rows_coverage_rewarded=coverage_threat,
        undefended_threat_age_by_row=ages,
        max_undefended_threat_age_by_row=maxima,
        undefended_threat_age_sum_by_row=age_sums,
        undefended_threat_age_count_by_row=age_counts,
    )
    return RewardComposition(
        breakdown=RewardBreakdown.from_mapping(components),
        state=next_state,
        event_diagnostics=(("cherry_delayed", cherry_diag),),
    )


def fusion_source_from_result(action_result: Mapping[str, Any]) -> str:
    intent_source = str(action_result.get("fusionIntentSource") or action_result.get("fusion_intent_source") or "")
    if intent_source:
        return normalize_fusion_source(intent_source)
    source = str(action_result.get("fusionExecutionSource") or "")
    if source == "model_action_mask":
        return "model"
    coach_payload = action_result.get("humanCoach") if isinstance(action_result.get("humanCoach"), Mapping) else {}
    coach_source = str(
        action_result.get("coach_command_source")
        or action_result.get("coachCommandSource")
        or coach_payload.get("source")
        or ""
    )
    coach_mode = str(coach_payload.get("commandMode") or coach_payload.get("command_mode") or "")
    if coach_mode == "assist":
        return "assist"
    if coach_source in {"human", "gui", "human_coach"}:
        return "human_coach"
    if coach_source in {"stream", "stream_coach"}:
        return "stream_coach"
    if action_result.get("coachFusionOverrideApplied") is not None or action_result.get(
        "executed_from_fresh_coach_command"
    ):
        return "human_coach"
    if isinstance(action_result.get("fusionCandidate"), Mapping) and "fusionOverrideApplied" in action_result:
        return "scripted"
    return source or "model"


def _fusion_event_from_result(
    action_result: Optional[Mapping[str, Any]],
    observation: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not isinstance(action_result, Mapping):
        return None
    decoded = action_result.get("decoded") if isinstance(action_result.get("decoded"), Mapping) else {}
    candidate = action_result.get("fusionCandidate") if isinstance(action_result.get("fusionCandidate"), Mapping) else {}
    is_fusion = (
        "fusionSucceeded" in action_result
        or "fusionAttempted" in action_result
        or "fusion_success" in action_result
        or bool(action_result.get("fusionExecutionSource"))
        or str(decoded.get("kind") or "") == "fusion"
        or bool(candidate)
        or bool(action_result.get("fusionRejectedReason"))
    )
    if not is_fusion:
        return None
    success = bool(action_result.get("fusionSucceeded") or action_result.get("fusion_success"))
    legal = bool(candidate.get("fusion_legal", success))
    bridge_method = str(action_result.get("bridgeMethodUsed") or action_result.get("bridge_method_used") or "")
    attempted = bool(success or action_result.get("fusionAttempted") or (bridge_method and bridge_method != "none"))
    changed = _safe_int(
        action_result.get("changedTileCount"),
        action_result.get("changed_tile_count"),
        default=(1 if success else 0),
    )
    reason = "" if success else str(
        action_result.get("fusionRejectedReason")
        or action_result.get("illegalReason")
        or action_result.get("bridgeResultReason")
        or "failed"
    )
    if reason in {FUSION_ILLEGAL_INCOMPATIBLE, "empty_tile"}:
        legal = False
    return {
        "source": fusion_source_from_result(action_result),
        "success": success,
        "legal": legal,
        "attempted": attempted,
        "board_changed": success and changed > 0,
        "reason": reason,
        "row": _safe_int(decoded.get("row"), action_result.get("sourceRow"), candidate.get("source_row"), default=-1),
        "col": _safe_int(decoded.get("column"), action_result.get("sourceCol"), candidate.get("source_col"), default=-1),
        "existing_plant": _safe_int(
            decoded.get("sourcePlantType"),
            action_result.get("sourcePlantType"),
            candidate.get("source_plant_type"),
            default=-1,
        ),
        "selected_seed": _safe_int(
            decoded.get("ingredientPlantType"),
            action_result.get("ingredientPlantType"),
            candidate.get("target_or_ingredient_type"),
            default=-1,
        ),
        "seed_slot": _safe_int(
            candidate.get("ingredient_seed_slot_index"),
            action_result.get("ingredientSeedSlotIndex"),
            default=-1,
        ),
        "result": action_result,
        "observation": observation if isinstance(observation, Mapping) else {},
    }


def _fusion_usefulness_components(
    event: Mapping[str, Any],
    config: RewardConfig,
    plant_types: Sequence[int],
    *,
    facts: Optional[StepFacts] = None,
) -> Dict[str, float]:
    observation = event.get("observation") if isinstance(event.get("observation"), Mapping) else {}
    snapshot = facts or build_step_facts(observation, plant_types)
    rows = max(1, int(snapshot.rows or 5))
    row = _safe_int(event.get("row"), default=-1)
    if not 0 <= row < rows:
        return {}
    threat_rows = set(_active_threat_rows(snapshot, rows))
    bonuses: Dict[str, float] = {}
    if row in threat_rows:
        bonuses["fusion_threatened_row_bonus"] = float(config.fusion_threatened_row_bonus)
    if snapshot.zombies_by_lane.get(row) or _lane_zombie_counts(snapshot, rows).get(row, 0) > 0 or _safe_int(
        observation.get("zombieCount"), default=0
    ) > 0:
        bonuses["fusion_active_wave_bonus"] = float(config.fusion_active_wave_bonus)
    shooters = _plant_counts_by_row(snapshot, rows, 0).get(row, 0)
    wallnuts = _wallnut_blocker_count(snapshot, row)
    if row in threat_rows and shooters + wallnuts <= 1:
        bonuses["fusion_defensive_value_bonus"] = float(config.fusion_defensive_value_bonus)
    return bonuses


def _fusion_components_tuple(values: Mapping[str, float]) -> Tuple[float, ...]:
    return tuple(float(values.get(name, 0.0)) for name in FUSION_REWARD_COMPONENT_NAMES)


def compose_fusion_reward(
    state: FusionRewardState,
    observation: Optional[Mapping[str, Any]],
    action_result: Optional[Mapping[str, Any]],
    *,
    config: RewardConfig,
    plant_types: Sequence[int] = (),
    facts: Optional[StepFacts] = None,
    enable_recipe_discovery_reward: bool = False,
    enable_fusion_chain_rewards: bool = False,
    enable_repeat_recipe_decay: bool = False,
) -> Tuple[FusionRewardState, float, Tuple[Tuple[str, Any], ...]]:
    event = _fusion_event_from_result(action_result, observation)
    if event is None:
        return state, 0.0, ()
    event_id = str((action_result or {}).get("fusionEventId") or (action_result or {}).get("fusion_event_id") or "")
    if event_id and event_id in state.accounted_event_ids:
        return state, 0.0, (("fusionRewardDuplicateSuppressed", True),)

    row = _safe_int(event.get("row"), default=-1)
    col = _safe_int(event.get("col"), default=-1)
    seed_slot = _safe_int(event.get("seed_slot"), default=-1)
    success = bool(event.get("success"))
    legal = bool(event.get("legal", success))
    attempted = bool(event.get("attempted", success))
    board_changed = bool(event.get("board_changed", success))
    reason = str(event.get("reason") or "")
    source = str(event.get("source") or "model")
    confirmed_success = bool(success and legal and board_changed)
    bad_reason = reason or ("no_board_change" if success and not board_changed else "failed")
    spam_reasons = {
        FUSION_ILLEGAL_INCOMPATIBLE,
        "empty_tile",
        "exception",
        "bridge_error",
        "fusion_bridge_unavailable",
        "fusion_probe_failed",
        "fusion_no_effect",
        "no_board_change",
        "failed",
    }
    key = (row, col, seed_slot, str(bad_reason or "failed"))
    same_rejection = sum(
        1 for r, c, slot, prior_reason, _step in state.recent_attempts
        if (r, c, slot, prior_reason) == key
    )
    recent_rejections = sum(1 for _r, _c, _slot, prior_reason, _step in state.recent_attempts if prior_reason)
    spam = bool(
        not confirmed_success
        and bad_reason in spam_reasons
        and (same_rejection >= 1 or (bool(bad_reason) and recent_rejections >= 3))
    )
    recent_attempts = (*state.recent_attempts, (
        row,
        col,
        seed_slot,
        "" if confirmed_success else bad_reason,
        state.event_counter,
    ))[-20:]
    component_totals = state.component_dict()

    def record(name: str, value: float) -> None:
        if value and name in component_totals:
            component_totals[name] = float(component_totals[name]) + float(value)

    positive = 0.0
    negative = 0.0
    positive_components: list[Tuple[str, float]] = []
    reasons: list[str] = []
    recipes_seen_episode = set(state.recipes_seen_episode)
    recipes_seen_run = set(state.recipes_seen_run)
    recipe_counts = state.recipe_count_dict()

    if legal and attempted:
        value = float(config.fusion_attempt_reward)
        positive += value
        positive_components.append(("fusion_attempt_reward", value))
    if confirmed_success:
        value = float(config.fusion_success_reward)
        positive += value
        positive_components.append(("fusion_success_reward", value))
        existing_plant = _safe_int(event.get("existing_plant"), default=-1)
        selected_seed = _safe_int(event.get("selected_seed"), default=-1)
        recipe_key = f"{fusion_plant_name(selected_seed)} + {fusion_plant_name(existing_plant)}"
        prior_recipe_count = int(recipe_counts.get(recipe_key, 0))
        new_recipe_episode = recipe_key not in recipes_seen_episode
        new_recipe_run = recipe_key not in recipes_seen_run
        recipe_counts[recipe_key] = prior_recipe_count + 1
        recipes_seen_episode.add(recipe_key)
        recipes_seen_run.add(recipe_key)
        result_payload = event.get("result") if isinstance(event.get("result"), Mapping) else {}
        resulting_plant = (
            result_payload.get("resultingPlantAfter")
            if isinstance(result_payload.get("resultingPlantAfter"), Mapping)
            else {}
        )
        candidate = (
            result_payload.get("fusionCandidate")
            if isinstance(result_payload.get("fusionCandidate"), Mapping)
            else {}
        )
        result_type = _safe_int(
            resulting_plant.get("plantType"),
            resulting_plant.get("type"),
            candidate.get("predicted_result_type"),
            default=-1,
        )
        result_tier = fusion_tier(result_type)
        recursive = fusion_tier(existing_plant) > 0
        if enable_recipe_discovery_reward and new_recipe_episode:
            value = float(config.fusion_new_recipe_reward)
            positive += value
            positive_components.append(("fusion_new_recipe_reward", value))
            reasons.append("new_recipe_episode")
            if new_recipe_run:
                reasons.append("new_recipe_run")
        if enable_fusion_chain_rewards and recursive:
            value = float(config.fusion_recursive_reward)
            positive += value
            positive_components.append(("fusion_recursive_reward", value))
            reasons.append("recursive")
        if enable_fusion_chain_rewards and result_tier >= 2:
            value = float(config.fusion_tier3_reward if result_tier >= 3 else config.fusion_tier2_reward)
            positive += value
            positive_components.append(("fusion_tier_reward", value))
            reasons.append(f"tier_{result_tier}")
        for name, value in _fusion_usefulness_components(
            event,
            config,
            plant_types,
            facts=facts,
        ).items():
            positive += float(value)
            positive_components.append((name, float(value)))
        if enable_repeat_recipe_decay and prior_recipe_count >= 2 and positive > 0.0:
            multiplier = max(0.0, min(1.0, float(config.fusion_repeat_reward_multiplier)))
            reduced_positive = positive * multiplier
            decay = reduced_positive - positive
            positive_components = [(name, value * multiplier) for name, value in positive_components]
            negative += decay
            record("fusion_repeat_decay", decay)
            positive = reduced_positive
            reasons.append("repeat_decay")
        reasons.append("success")
    else:
        penalty_reason = reason or ("failed" if not success else "fusion_no_effect")
        if penalty_reason == FUSION_ILLEGAL_INCOMPATIBLE:
            negative += float(config.fusion_incompatible_penalty)
            record("fusion_incompatible_penalty", config.fusion_incompatible_penalty)
        elif penalty_reason == "empty_tile":
            negative += float(config.fusion_empty_tile_penalty)
            record("fusion_empty_tile_penalty", config.fusion_empty_tile_penalty)
        elif penalty_reason in {"exception", "bridge_error", "fusion_bridge_unavailable", "fusion_probe_failed"}:
            negative += float(config.fusion_bridge_error_penalty)
            record("fusion_bridge_error_penalty", config.fusion_bridge_error_penalty)
        elif penalty_reason in {
            "cooldown_not_ready",
            "insufficient_sun",
            "fusion_disabled",
            "seed_unavailable",
            "fusion_policy_none",
            "fusion_not_available",
            "target_not_available",
            "gameplay_not_ready",
            "not_gameplay",
            "seed_selection_active",
            "terminal_or_transition_state",
            "reward_or_unlock_screen_active",
            "startup_stale_command_blocked",
            "coach_command_already_executed",
        }:
            penalty_reason = ""
        else:
            negative += float(config.fusion_failed_penalty)
            record("fusion_failed_penalty", config.fusion_failed_penalty)
        if penalty_reason:
            reasons.append(penalty_reason)
    if spam:
        negative += float(config.fusion_spam_penalty)
        record("fusion_spam_penalty", config.fusion_spam_penalty)
        reasons.append("spam")

    cap = float(config.max_fusion_reward_per_episode or 0.0)
    if positive <= 0.0:
        capped_positive = 0.0
        positive_total = state.positive_total
        capped = state.capped
    elif cap <= 0.0:
        capped_positive = positive
        positive_total = state.positive_total + positive
        capped = state.capped
    else:
        remaining = max(0.0, cap - state.positive_total)
        capped_positive = min(positive, remaining)
        positive_total = state.positive_total + capped_positive
        capped = bool(state.capped or capped_positive < positive - 1e-9 or positive_total >= cap - 1e-9)

    remaining_positive = capped_positive
    usefulness_bonus = 0.0
    for name, raw_value in positive_components:
        applied = min(max(0.0, raw_value), max(0.0, remaining_positive))
        if applied > 0.0:
            record(name, applied)
            if name in {
                "fusion_threatened_row_bonus",
                "fusion_active_wave_bonus",
                "fusion_defensive_value_bonus",
            }:
                usefulness_bonus += applied
            remaining_positive -= applied
    net = capped_positive + negative
    accounted = set(state.accounted_event_ids)
    annotations: Tuple[Tuple[str, Any], ...] = ()
    if event_id:
        accounted.add(event_id)
        annotations = (("fusionRewardApplied", True), ("fusionRewardDelta", float(net)))
    next_state = FusionRewardState(
        reward_total=state.reward_total + net,
        positive_total=positive_total,
        capped=capped,
        component_totals=_fusion_components_tuple(component_totals),
        last_reward_delta=net,
        last_reward_reason=",".join(reasons),
        last_usefulness_bonus=usefulness_bonus,
        last_source=source,
        recent_attempts=tuple(recent_attempts),
        event_counter=state.event_counter + 1,
        accounted_event_ids=frozenset(accounted),
        recipes_seen_episode=frozenset(recipes_seen_episode),
        recipe_counts_episode=tuple(sorted(recipe_counts.items())),
        recipes_seen_run=frozenset(recipes_seen_run),
    )
    return next_state, net, annotations


def fusion_reward_live_fields(state: FusionRewardState) -> Dict[str, Any]:
    totals = state.component_dict()
    return {
        "fusion_reward_total": round(float(state.reward_total), 6),
        "fusion_attempt_reward_total": round(totals["fusion_attempt_reward"], 6),
        "fusion_success_reward_total": round(totals["fusion_success_reward"], 6),
        "fusion_new_recipe_reward_total": round(totals["fusion_new_recipe_reward"], 6),
        "fusion_recursive_reward_total": round(totals["fusion_recursive_reward"], 6),
        "fusion_tier_reward_total": round(totals["fusion_tier_reward"], 6),
        "fusion_repeat_decay_total": round(totals["fusion_repeat_decay"], 6),
        "fusion_threatened_row_bonus_total": round(totals["fusion_threatened_row_bonus"], 6),
        "fusion_active_wave_bonus_total": round(totals["fusion_active_wave_bonus"], 6),
        "fusion_defensive_value_bonus_total": round(totals["fusion_defensive_value_bonus"], 6),
        "fusion_incompatible_penalty_total": round(totals["fusion_incompatible_penalty"], 6),
        "fusion_empty_tile_penalty_total": round(totals["fusion_empty_tile_penalty"], 6),
        "fusion_failed_penalty_total": round(totals["fusion_failed_penalty"], 6),
        "fusion_bridge_error_penalty_total": round(totals["fusion_bridge_error_penalty"], 6),
        "fusion_spam_penalty_total": round(totals["fusion_spam_penalty"], 6),
        "fusion_reward_capped": bool(state.capped),
        "fusion_last_reward_delta": round(float(state.last_reward_delta), 6),
        "fusion_last_reward_reason": str(state.last_reward_reason),
        "fusion_last_usefulness_bonus": round(float(state.last_usefulness_bonus), 6),
        "fusion_last_source": str(state.last_source),
    }


def compose_step_reward(
    previous: Optional[Mapping[str, Any]],
    current: Mapping[str, Any],
    action_result: Optional[Mapping[str, Any]],
    *,
    config: RewardConfig,
    state: RewardCompositionState,
    plant_types: Sequence[int] = (),
    fallback_rows: int = 5,
    fallback_columns: int = 10,
    previous_legal_actions: Sequence[int] = (),
    previous_facts: Optional[StepFacts] = None,
    current_facts: Optional[StepFacts] = None,
    enable_recipe_discovery_reward: bool = False,
    enable_fusion_chain_rewards: bool = False,
    enable_repeat_recipe_decay: bool = False,
    terminal_reward_override: Optional[float] = None,
    exceptional_adjustments: Optional[Mapping[str, float]] = None,
    additional_components: Optional[Mapping[str, float]] = None,
) -> RewardComposition:
    ordinary = compose_environment_reward(
        previous,
        current,
        action_result,
        config=config,
        state=state,
        plant_types=plant_types,
        fallback_rows=fallback_rows,
        fallback_columns=fallback_columns,
        previous_legal_actions=previous_legal_actions,
        previous_facts=previous_facts,
        current_facts=current_facts,
    )
    fusion, fusion_delta, annotations = compose_fusion_reward(
        ordinary.state.fusion,
        previous,
        action_result,
        config=config,
        plant_types=plant_types,
        facts=previous_facts,
        enable_recipe_discovery_reward=enable_recipe_discovery_reward,
        enable_fusion_chain_rewards=enable_fusion_chain_rewards,
        enable_repeat_recipe_decay=enable_repeat_recipe_decay,
    )
    breakdown = ordinary.breakdown
    if fusion_delta:
        breakdown = breakdown.with_components({"fusion_reward": fusion_delta})
    if terminal_reward_override is not None:
        breakdown = breakdown.with_components(
            {"win_loss_reward": float(terminal_reward_override)},
            replace_values=True,
        )
    if additional_components:
        breakdown = breakdown.with_components(additional_components)
    for name, value in (exceptional_adjustments or {}).items():
        breakdown = breakdown.with_adjustment(str(name), float(value))
    return RewardComposition(
        breakdown=breakdown,
        state=replace(ordinary.state, fusion=fusion),
        event_diagnostics=ordinary.event_diagnostics,
        action_result_annotations=annotations,
        fusion_reward_delta=fusion_delta,
    )


__all__ = [
    "FUSION_REWARD_COMPONENT_NAMES",
    "FusionRewardState",
    "PendingCherryEvent",
    "REWARD_COMPONENT_FIELDS",
    "REWARD_EPISODE_TOTAL_FIELDS",
    "RewardAction",
    "RewardBreakdown",
    "RewardComposition",
    "RewardCompositionState",
    "RewardConfig",
    "compose_environment_reward",
    "compose_fusion_reward",
    "compose_step_reward",
    "fusion_reward_live_fields",
    "fusion_source_from_result",
    "merge_reward_components",
    "reward_action_from_result",
]
