"""Immutable, reusable indexes for one PvZRL bridge observation.

The bridge observation remains the public contract and runtime authority.  A
``StepFacts`` instance is only a typed, immutable view over that payload: build
it once after an observation arrives, then pass it to masks, rewards,
diagnostics, fusion, encoding, and metrics instead of repeatedly scanning the
same lists.

Two plant views are intentionally distinct.  ``plants`` and its row/type
indexes contain only the bridge's primary ``plants`` list.  ``occupancy`` also
uses active, in-bounds ``visiblePlants`` as a first-hit fallback, matching the
long-standing action/fusion legality contract.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, FrozenSet, Iterable, Optional, Sequence, Tuple


Cell = Tuple[int, int]


def _safe_int(*values: Any, default: int = 0) -> int:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            continue
    return int(default)


def _safe_float(*values: Any, default: float = 0.0) -> float:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError, OverflowError):
            continue
    return float(default)


# Shared coercers for pure facts/reward/diagnostic modules.  The private names
# remain local aliases so the fact builder stays compact.
safe_int = _safe_int
safe_float = _safe_float


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_json_value(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True, default=str))
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_value(item())
        except (TypeError, ValueError):
            pass
    return repr(value)


def stable_digest(value: Any) -> str:
    """Return the deterministic SHA-256 digest used for frame/cache proofs."""

    encoded = json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ObservationIdentity:
    revision: str
    content_digest: str

    @property
    def token(self) -> str:
        return f"{self.revision}:{self.content_digest}"


def observation_identity(observation: Mapping[str, Any]) -> ObservationIdentity:
    revision_value = observation.get("frameCount")
    if revision_value is None:
        revision_value = observation.get("frame")
    if revision_value is None:
        revision_value = observation.get("time")
    revision = str(revision_value) if revision_value is not None else "unversioned"
    return ObservationIdentity(revision=revision, content_digest=stable_digest(observation))


@dataclass(frozen=True, slots=True)
class PlantFact:
    position: int
    row: int
    column: int
    compat_column: int
    plant_type: int
    type_name: str = ""
    instance_id: int = 0
    health: float = 0.0
    max_health: float = 0.0
    fusion_health_ratio: float = 1.0
    attack_cooldown: float = 0.0
    produce_cooldown: float = 0.0
    active: bool = True
    in_board_bounds: bool = True
    source: str = "plants"

    @property
    def cell(self) -> Cell:
        return self.row, self.column

    @property
    def compat_cell(self) -> Cell:
        return self.row, self.compat_column

    @property
    def health_ratio(self) -> float:
        maximum = max(1.0, float(self.max_health))
        return max(0.0, min(1.0, float(self.health) / maximum))


@dataclass(frozen=True, slots=True)
class ZombieFact:
    position: int
    row: int
    zombie_type: int
    x: float
    has_position: bool = True
    type_name: str = ""
    health: float = 0.0
    max_health: float = 0.0
    alive: bool = True


@dataclass(frozen=True, slots=True)
class LaneFact:
    row: int
    zombie_count: int = 0
    nearest_zombie_x: Optional[float] = None
    nearest_zombie_health: float = 0.0
    nearest_zombie_type: int = 0
    danger: float = 0.0
    tough_zombie_pressure_score: Optional[float] = None
    buckethead_count: int = 0
    conehead_count: int = 0
    tough_zombie_count: int = 0


@dataclass(frozen=True, slots=True)
class SeedSlotFact:
    """Normalized seed state with explicit action/fusion default semantics."""

    position: int
    slot_index: int
    plant_type: int
    usable: bool
    disabled: bool
    ready: bool
    current_cooldown: float
    full_cooldown: float
    seed_cost: int
    plant_type_name: str = ""
    cooldown_plant_type: int = -1
    cooldown_plant_type_name: str = ""
    card_instance_id: int = 0
    is_available: bool = False
    raw_cooldown: float = 0.0
    source: str = ""
    synthetic: bool = False
    fusion_usable: bool = True
    fusion_ready: bool = True
    affordable: bool = False
    valid: bool = True

    @property
    def legacy_ready(self) -> bool:
        """Match environment helpers that trust ``ready`` without a cooldown probe."""

        return bool(self.fusion_usable and not self.disabled and self.ready and self.affordable)


@dataclass(frozen=True, slots=True)
class MowerFacts:
    count: int
    logical_count: int
    visible_count: int
    stale_visible_count: int
    duplicate_rows: Tuple[int, ...]
    active_rows: Optional[FrozenSet[int]]


@dataclass(frozen=True, slots=True)
class LifecycleFacts:
    board_found: bool
    can_read_board: bool
    create_plant_found: bool
    gameplay_ready: bool
    actual_gameplay_ready: bool
    seed_selection_active: bool
    seed_selection_panel_active: bool
    blocking_reward_ui_active: bool
    done: bool
    over: bool
    more_zombies_coming: bool
    wave: int
    max_wave: int
    time: float
    screen_state: str
    terminal_hint: str


@dataclass(frozen=True, slots=True)
class SafetyFacts:
    reported_plant_count: int
    reported_visible_plant_count: int
    reported_zombie_count: int
    total_plant_health: float
    invalid_primary_plant_count: int
    invalid_zombie_count: int
    malformed_plant_count: int
    malformed_visible_plant_count: int
    malformed_zombie_count: int
    malformed_seed_slot_count: int
    duplicate_primary_cells: Tuple[Cell, ...]
    plant_count_mismatch: bool
    zombie_count_mismatch: bool


@dataclass(frozen=True, slots=True)
class StepFacts:
    identity: ObservationIdentity
    rows: int
    columns: int
    action_count: int
    sun: int
    plants: Tuple[PlantFact, ...]
    visible_plants: Tuple[PlantFact, ...]
    occupancy: Tuple[Tuple[int, int, int], ...]
    occupied_cells: FrozenSet[Cell]
    occupant_by_cell: Mapping[Cell, PlantFact]
    compat_occupant_by_cell: Mapping[Cell, PlantFact]
    primary_plant_stacks_by_cell: Mapping[Cell, Tuple[PlantFact, ...]]
    last_primary_plant_by_cell: Mapping[Cell, PlantFact]
    plants_by_lane: Mapping[int, Tuple[PlantFact, ...]]
    plants_by_type: Mapping[int, Tuple[PlantFact, ...]]
    plant_count_by_lane: Mapping[int, int]
    plant_count_by_type_and_lane: Mapping[Tuple[int, int], int]
    zombies: Tuple[ZombieFact, ...]
    alive_zombies: Tuple[ZombieFact, ...]
    zombies_by_lane: Mapping[int, Tuple[ZombieFact, ...]]
    nearest_zombie_by_lane: Mapping[int, ZombieFact]
    lanes: Tuple[LaneFact, ...]
    lane_by_row: Mapping[int, LaneFact]
    seed_slots: Tuple[SeedSlotFact, ...]
    seed_slot_by_index: Mapping[int, SeedSlotFact]
    seed_slots_by_type: Mapping[int, Tuple[SeedSlotFact, ...]]
    ready_seed_types: FrozenSet[int]
    mower: MowerFacts
    lifecycle: LifecycleFacts
    safety: SafetyFacts
    fusion_source_types: FrozenSet[int]

    @property
    def total_lane_danger(self) -> float:
        return sum(
            self.lane_by_row[row].danger
            for row in range(self.rows)
            if row in self.lane_by_row
        )

    def seed_slot(self, slot_index: int, *, positional_fallback: bool = True) -> Optional[SeedSlotFact]:
        slot = self.seed_slot_by_index.get(int(slot_index))
        if slot is not None:
            return slot
        if positional_fallback and 0 <= int(slot_index) < len(self.seed_slots):
            positional = self.seed_slots[int(slot_index)]
            return positional if positional.valid else None
        return None


def _plant_fact(raw: Mapping[str, Any], position: int, source: str) -> PlantFact:
    fusion_max_health = max(1.0, _safe_float(raw.get("maxHealth"), default=1.0))
    fusion_health = _safe_float(raw.get("health"), default=fusion_max_health)
    return PlantFact(
        position=position,
        row=_safe_int(raw.get("row"), default=-1),
        column=_safe_int(raw.get("column"), default=-1),
        compat_column=_safe_int(raw.get("column"), raw.get("col"), default=-1),
        plant_type=_safe_int(raw.get("type"), raw.get("plantType"), default=-1),
        type_name=str(raw.get("typeName") or raw.get("plantTypeName") or ""),
        instance_id=_safe_int(raw.get("instanceId"), raw.get("instanceID"), default=0),
        health=_safe_float(raw.get("health"), default=0.0),
        max_health=_safe_float(raw.get("maxHealth"), default=0.0),
        fusion_health_ratio=round(
            max(0.0, min(1.0, fusion_health / fusion_max_health)),
            4,
        ),
        attack_cooldown=_safe_float(raw.get("attackCooldown"), default=0.0),
        produce_cooldown=_safe_float(raw.get("produceCooldown"), default=0.0),
        active=bool(raw.get("activeInHierarchy", True)),
        in_board_bounds=bool(raw.get("inBoardBounds", True)),
        source=source,
    )


def _seed_slot_fact(
    raw: Mapping[str, Any],
    position: int,
    sun: int,
    *,
    fallback_plant_types: Sequence[int] = (),
    synthetic: bool = False,
    valid: bool = True,
) -> SeedSlotFact:
    cost = max(0, _safe_int(raw.get("seedCost"), default=0))
    slot_index = _safe_int(raw.get("slotIndex"), default=position)
    configured_plant_type = (
        int(fallback_plant_types[slot_index])
        if 0 <= slot_index < len(fallback_plant_types)
        else -1
    )
    return SeedSlotFact(
        position=position,
        slot_index=slot_index,
        plant_type=_safe_int(raw.get("plantType"), raw.get("type"), default=-1),
        usable=bool(raw.get("usable", False)),
        disabled=bool(raw.get("disabled", False)),
        ready=bool(raw.get("ready", False)),
        current_cooldown=_safe_float(raw.get("currentCooldown"), default=0.0),
        full_cooldown=_safe_float(raw.get("fullCooldown"), default=0.0),
        seed_cost=cost,
        plant_type_name=str(raw.get("plantTypeName") or raw.get("typeName") or ""),
        cooldown_plant_type=_safe_int(
            raw.get("plantType"),
            default=configured_plant_type,
        ),
        cooldown_plant_type_name=str(raw.get("plantTypeName") or ""),
        card_instance_id=_safe_int(raw.get("cardInstanceId"), default=0),
        is_available=bool(raw.get("isAvailable", False)),
        raw_cooldown=_safe_float(raw.get("rawCooldown"), default=0.0),
        source=str(raw.get("source") or ""),
        synthetic=bool(synthetic),
        fusion_usable=bool(raw.get("usable", True)),
        fusion_ready=bool(raw.get("ready", True)),
        affordable=int(sun) >= cost,
        valid=bool(valid),
    )


def _index_tuples(values: Iterable[Any], key: Any) -> Mapping[Any, Tuple[Any, ...]]:
    index: Dict[Any, list[Any]] = {}
    for value in values:
        index.setdefault(key(value), []).append(value)
    return MappingProxyType({item_key: tuple(items) for item_key, items in index.items()})


def build_step_facts(
    observation: Mapping[str, Any],
    fallback_plant_types: Sequence[int] = (),
    *,
    identity: Optional[ObservationIdentity] = None,
) -> StepFacts:
    """Build an immutable snapshot without mutating or retaining the input."""

    obs = observation if isinstance(observation, Mapping) else {}
    resolved_identity = identity or observation_identity(obs)
    fallback_types = tuple(int(value) for value in fallback_plant_types)
    rows = max(0, _safe_int(obs.get("rowCount"), default=0))
    columns = max(0, _safe_int(obs.get("columnCount"), default=0))
    sun = _safe_int(obs.get("sun"), default=0)

    raw_plants = obs.get("plants", [])
    plant_items = raw_plants if isinstance(raw_plants, list) else []
    malformed_plants = sum(1 for item in plant_items if not isinstance(item, Mapping))
    plants = tuple(
        _plant_fact(item, position, "plants")
        for position, item in enumerate(plant_items)
        if isinstance(item, Mapping)
    )
    raw_visible = obs.get("visiblePlants", [])
    visible_items = raw_visible if isinstance(raw_visible, list) else []
    malformed_visible = sum(1 for item in visible_items if not isinstance(item, Mapping))
    visible_plants = tuple(
        fact
        for position, item in enumerate(visible_items)
        if isinstance(item, Mapping)
        for fact in (_plant_fact(item, position, "visiblePlants"),)
        if fact.active and fact.in_board_bounds
    )

    primary_stacks = _index_tuples(plants, lambda plant: plant.cell)
    last_primary = MappingProxyType({plant.cell: plant for plant in plants})
    occupants: Dict[Cell, PlantFact] = {}
    for plant in (*plants, *visible_plants):
        occupants.setdefault(plant.cell, plant)
    occupant_by_cell = MappingProxyType(occupants)
    compat_occupants: Dict[Cell, PlantFact] = {}
    for plant in (*plants, *visible_plants):
        compat_occupants.setdefault(plant.compat_cell, plant)
    occupancy = tuple(
        (cell[0], cell[1], plant.plant_type)
        for cell, plant in occupants.items()
    )

    raw_zombies = obs.get("zombies", [])
    zombie_items = raw_zombies if isinstance(raw_zombies, list) else []
    malformed_zombies = sum(1 for item in zombie_items if not isinstance(item, Mapping))
    zombies = tuple(
        ZombieFact(
            position=position,
            row=_safe_int(item.get("row"), default=-1),
            zombie_type=_safe_int(item.get("type"), item.get("zombieType"), default=-1),
            x=_safe_float(item.get("x"), item.get("column"), default=0.0),
            has_position=item.get("x") is not None or item.get("column") is not None,
            type_name=str(item.get("typeName") or item.get("zombieTypeName") or ""),
            health=_safe_float(item.get("health"), default=0.0),
            max_health=_safe_float(item.get("maxHealth"), default=0.0),
            alive=bool(item.get("alive", True)),
        )
        for position, item in enumerate(zombie_items)
        if isinstance(item, Mapping)
    )
    alive_zombies = tuple(zombie for zombie in zombies if zombie.alive)
    zombies_by_lane = _index_tuples(
        sorted(alive_zombies, key=lambda zombie: (zombie.row, zombie.x, zombie.position)),
        lambda zombie: zombie.row,
    )
    nearest_by_lane = MappingProxyType(
        {row: lane_zombies[0] for row, lane_zombies in zombies_by_lane.items() if lane_zombies}
    )

    raw_lanes = obs.get("lanes", [])
    lane_items = raw_lanes if isinstance(raw_lanes, list) else []
    lane_by_row_mutable: Dict[int, LaneFact] = {}
    lanes_list = []
    for item in lane_items:
        if not isinstance(item, Mapping):
            continue
        row = _safe_int(item.get("row"), default=-1)
        zombie_count = max(0, _safe_int(item.get("zombieCount"), default=0))
        nearest_x = _optional_float(item.get("nearestZombieX"))
        raw_danger = _optional_float(item.get("danger"))
        if zombie_count <= 0:
            danger = 0.0
        elif raw_danger is not None:
            danger = max(0.0, raw_danger)
        elif nearest_x is not None:
            danger = max(0.0, 1.0 - nearest_x / 10.0)
        else:
            danger = 0.0
        lane = LaneFact(
            row=row,
            zombie_count=zombie_count,
            nearest_zombie_x=nearest_x,
            nearest_zombie_health=_safe_float(item.get("nearestZombieHealth"), default=0.0),
            nearest_zombie_type=_safe_int(item.get("nearestZombieType"), default=0),
            danger=danger,
            tough_zombie_pressure_score=_optional_float(item.get("toughZombiePressureScore")),
            buckethead_count=_safe_int(item.get("bucketheadCount"), item.get("buckethead_count"), default=0),
            conehead_count=_safe_int(item.get("coneheadCount"), item.get("conehead_count"), default=0),
            tough_zombie_count=_safe_int(item.get("toughZombieCount"), item.get("tough_zombie_count"), default=0),
        )
        lanes_list.append(lane)
        lane_by_row_mutable[row] = lane
    lanes = tuple(lanes_list)

    raw_slots = obs.get("seedSlots", [])
    slot_items = raw_slots if isinstance(raw_slots, list) else []
    malformed_slots = sum(1 for item in slot_items if not isinstance(item, Mapping))
    synthetic_slots = False
    if not slot_items:
        synthetic_slots = True
        slot_items = [
            {"slotIndex": index, "plantType": plant_type, "seedCost": 0}
            for index, plant_type in enumerate(fallback_types)
        ]
    seed_slots = tuple(
        _seed_slot_fact(
            item if isinstance(item, Mapping) else {},
            position,
            sun,
            fallback_plant_types=fallback_types,
            synthetic=synthetic_slots,
            valid=isinstance(item, Mapping),
        )
        for position, item in enumerate(slot_items)
    )
    slot_by_index_mutable: Dict[int, SeedSlotFact] = {}
    for slot in seed_slots:
        if slot.valid:
            slot_by_index_mutable.setdefault(slot.slot_index, slot)

    visible_mowers_value = obs.get("visibleMowers")
    if isinstance(visible_mowers_value, list):
        active_mower_rows: Optional[FrozenSet[int]] = frozenset(
            _safe_int(item.get("row"), default=-1)
            for item in visible_mowers_value
            if isinstance(item, Mapping)
            and bool(item.get("activeInHierarchy", True))
            and bool(item.get("inBoardBounds", True))
            and bool(item.get("inMowerArray", True))
            and _safe_int(item.get("row"), default=-1) >= 0
        )
    else:
        active_mower_rows = None
    if "logicalMowerCount" in obs:
        mower_count = max(0, _safe_int(obs.get("logicalMowerCount"), default=0))
    elif "visibleMowerObjectCount" in obs:
        mower_count = max(0, _safe_int(obs.get("visibleMowerObjectCount"), default=0))
    else:
        mower_count = rows
    duplicate_mower_rows = obs.get("duplicateMowerRows", [])
    duplicate_rows = tuple(
        _safe_int(value, default=-1)
        for value in (duplicate_mower_rows if isinstance(duplicate_mower_rows, list) else [])
        if _safe_int(value, default=-1) >= 0
    )

    lifecycle = LifecycleFacts(
        board_found=bool(obs.get("boardFound")),
        can_read_board=bool(obs.get("canReadBoard", True)),
        create_plant_found=bool(obs.get("createPlantFound")),
        gameplay_ready=bool(obs.get("gameplayReady")),
        actual_gameplay_ready=bool(obs.get("actualGameplayReady")),
        seed_selection_active=bool(obs.get("seedSelectionActive")),
        seed_selection_panel_active=bool(obs.get("seedSelectionPanelActive")),
        blocking_reward_ui_active=bool(obs.get("blockingRewardUiActive")),
        done=bool(obs.get("done")),
        over=bool(obs.get("over")),
        more_zombies_coming=bool(obs.get("moreZombiesComing")),
        wave=_safe_int(obs.get("wave"), default=0),
        max_wave=_safe_int(obs.get("maxWave"), default=0),
        time=_safe_float(obs.get("time"), default=0.0),
        screen_state=str(obs.get("screenState") or ""),
        terminal_hint=str(obs.get("terminalHint") or ""),
    )
    invalid_plants = sum(
        1 for plant in plants
        if not (0 <= plant.row < rows and 0 <= plant.column < columns)
    )
    invalid_zombies = sum(1 for zombie in zombies if not (0 <= zombie.row < rows))
    duplicate_cells = tuple(sorted(cell for cell, stack in primary_stacks.items() if len(stack) > 1))
    reported_plant_count = _safe_int(obs.get("plantCount"), default=len(plants))
    reported_zombie_count = _safe_int(obs.get("zombieCount"), default=len(zombies))
    plant_count_by_lane_mutable: Dict[int, int] = {}
    plant_count_by_type_and_lane_mutable: Dict[Tuple[int, int], int] = {}
    for plant in plants:
        plant_count_by_lane_mutable[plant.row] = (
            plant_count_by_lane_mutable.get(plant.row, 0) + 1
        )
        type_lane = (plant.plant_type, plant.row)
        plant_count_by_type_and_lane_mutable[type_lane] = (
            plant_count_by_type_and_lane_mutable.get(type_lane, 0) + 1
        )

    return StepFacts(
        identity=resolved_identity,
        rows=rows,
        columns=columns,
        action_count=max(0, _safe_int(obs.get("actionCount"), default=0)),
        sun=sun,
        plants=plants,
        visible_plants=visible_plants,
        occupancy=occupancy,
        occupied_cells=frozenset(occupants),
        occupant_by_cell=occupant_by_cell,
        compat_occupant_by_cell=MappingProxyType(compat_occupants),
        primary_plant_stacks_by_cell=primary_stacks,
        last_primary_plant_by_cell=last_primary,
        plants_by_lane=_index_tuples(plants, lambda plant: plant.row),
        plants_by_type=_index_tuples(plants, lambda plant: plant.plant_type),
        plant_count_by_lane=MappingProxyType(plant_count_by_lane_mutable),
        plant_count_by_type_and_lane=MappingProxyType(
            plant_count_by_type_and_lane_mutable
        ),
        zombies=zombies,
        alive_zombies=alive_zombies,
        zombies_by_lane=zombies_by_lane,
        nearest_zombie_by_lane=nearest_by_lane,
        lanes=lanes,
        lane_by_row=MappingProxyType(lane_by_row_mutable),
        seed_slots=seed_slots,
        seed_slot_by_index=MappingProxyType(slot_by_index_mutable),
        seed_slots_by_type=_index_tuples(
            (slot for slot in seed_slots if slot.valid),
            lambda slot: slot.plant_type,
        ),
        ready_seed_types=frozenset(slot.plant_type for slot in seed_slots if slot.valid and slot.legacy_ready and slot.plant_type >= 0),
        mower=MowerFacts(
            count=mower_count,
            logical_count=max(0, _safe_int(obs.get("logicalMowerCount"), default=rows)),
            visible_count=max(0, _safe_int(obs.get("visibleMowerObjectCount"), default=rows)),
            stale_visible_count=max(0, _safe_int(obs.get("staleVisibleMowerObjectCount"), default=0)),
            duplicate_rows=duplicate_rows,
            active_rows=active_mower_rows,
        ),
        lifecycle=lifecycle,
        safety=SafetyFacts(
            reported_plant_count=reported_plant_count,
            reported_visible_plant_count=_safe_int(obs.get("visiblePlantObjectCount"), default=len(visible_plants)),
            reported_zombie_count=reported_zombie_count,
            total_plant_health=_safe_float(obs.get("totalPlantHealth"), default=0.0),
            invalid_primary_plant_count=invalid_plants,
            invalid_zombie_count=invalid_zombies,
            malformed_plant_count=malformed_plants,
            malformed_visible_plant_count=malformed_visible,
            malformed_zombie_count=malformed_zombies,
            malformed_seed_slot_count=malformed_slots,
            duplicate_primary_cells=duplicate_cells,
            plant_count_mismatch=reported_plant_count != len(plants),
            zombie_count_mismatch=reported_zombie_count != len(zombies),
        ),
        fusion_source_types=frozenset(plant.plant_type for plant in occupants.values() if plant.plant_type >= 0),
    )


class StepFactsCache:
    """One-entry content-verified cache for the latest observation snapshot."""

    __slots__ = ("_key", "_facts", "_object", "hits", "misses")

    def __init__(self) -> None:
        self._key: Optional[Tuple[str, Tuple[int, ...]]] = None
        self._facts: Optional[StepFacts] = None
        self._object: Optional[Mapping[str, Any]] = None
        self.hits = 0
        self.misses = 0

    def get(
        self,
        observation: Mapping[str, Any],
        fallback_plant_types: Sequence[int] = (),
    ) -> StepFacts:
        identity = observation_identity(observation)
        fallback = tuple(int(value) for value in fallback_plant_types)
        key = identity.token, fallback
        if self._facts is not None and self._key == key:
            # The digest above verified this particular mapping.  Retain the
            # mapping itself so trusted nested consumers can reuse it safely;
            # an integer id alone is unsafe because CPython may recycle it.
            self._object = observation
            self.hits += 1
            return self._facts
        facts = build_step_facts(observation, fallback, identity=identity)
        self._key = key
        self._facts = facts
        self._object = observation
        self.misses += 1
        return facts

    def get_known(
        self,
        observation: Mapping[str, Any],
        fallback_plant_types: Sequence[int] = (),
    ) -> StepFacts:
        """Reuse an owner-verified synchronous frame without re-hashing it.

        Observation owners call ``get`` at the external boundary.  Nested mask,
        fusion, reward, encoding, and diagnostic consumers may then call this
        method while processing that same, non-mutated mapping object.
        """

        fallback = tuple(int(value) for value in fallback_plant_types)
        if (
            self._facts is not None
            and self._object is observation
            and self._key is not None
            and self._key[1] == fallback
        ):
            self.hits += 1
            return self._facts
        return self.get(observation, fallback)

    def clear(self) -> None:
        self._key = None
        self._facts = None
        self._object = None


__all__ = [
    "Cell",
    "LaneFact",
    "LifecycleFacts",
    "MowerFacts",
    "ObservationIdentity",
    "PlantFact",
    "SafetyFacts",
    "SeedSlotFact",
    "StepFacts",
    "StepFactsCache",
    "ZombieFact",
    "build_step_facts",
    "observation_identity",
    "safe_float",
    "safe_int",
    "stable_digest",
]
