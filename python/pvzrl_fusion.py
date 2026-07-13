"""Conservative fusion diagnostics and scripted assist helpers.

Fusion is intentionally modeled outside the PPO action space.  The helpers in
this module only score and validate bridge-reported candidates; the bridge
remains the final source of legality for any actual fusion execution.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

from pvzrl_observation_facts import SeedSlotFact, StepFacts, build_step_facts
from pvzrl_registry import get_plant_registry, normalize_plant_name


FUSION_POLICY_NONE = "none"
FUSION_POLICY_OBSERVE = "observe"
FUSION_POLICY_SCRIPTED = "scripted"
FUSION_POLICIES = {FUSION_POLICY_NONE, FUSION_POLICY_OBSERVE, FUSION_POLICY_SCRIPTED}
FUSION_POLICY_ALIASES = {
    "assist": FUSION_POLICY_SCRIPTED,
}

FUSION_SOURCE_MODEL = "model"
FUSION_SOURCE_SCRIPTED = "scripted"
FUSION_SOURCE_HUMAN = "human_coach"
FUSION_SOURCE_STREAM = "stream_coach"
FUSION_SOURCE_GUI = "gui"
FUSION_SOURCE_MANUAL = "manual"
FUSION_SOURCE_DEBUG = "debug"
FUSION_SOURCES = frozenset(
    {
        FUSION_SOURCE_MODEL,
        FUSION_SOURCE_SCRIPTED,
        FUSION_SOURCE_HUMAN,
        FUSION_SOURCE_STREAM,
        FUSION_SOURCE_GUI,
        FUSION_SOURCE_MANUAL,
        FUSION_SOURCE_DEBUG,
    }
)
_FUSION_SOURCE_ALIASES = MappingProxyType(
    {
        "model_action_mask": FUSION_SOURCE_MODEL,
        "policy": FUSION_SOURCE_MODEL,
        "human": FUSION_SOURCE_HUMAN,
        "human_coach": FUSION_SOURCE_HUMAN,
        "stream": FUSION_SOURCE_STREAM,
        "stream_coach": FUSION_SOURCE_STREAM,
        "mock_stream": FUSION_SOURCE_STREAM,
        "viewer": FUSION_SOURCE_STREAM,
        "gui": FUSION_SOURCE_GUI,
        "manual": FUSION_SOURCE_MANUAL,
        "scripted": FUSION_SOURCE_SCRIPTED,
        "assist": FUSION_SOURCE_SCRIPTED,
        "debug": FUSION_SOURCE_DEBUG,
    }
)

_PLANT_REGISTRY = get_plant_registry()


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    if isinstance(value, frozenset):
        return {_deep_thaw(item) for item in value}
    return value


def _required_base_plant_id(name: str) -> int:
    plant_id = _PLANT_REGISTRY.resolve_name(name)
    if plant_id is None:
        raise RuntimeError(f"canonical plant registry is missing required fusion base plant {name!r}")
    return int(plant_id)


PEASHOOTER_ID = _required_base_plant_id("Peashooter")
SUNFLOWER_ID = _required_base_plant_id("SunFlower")
CHERRYBOMB_ID = _required_base_plant_id("CherryBomb")
WALLNUT_ID = _required_base_plant_id("WallNut")

FUSION_REJECTION_REASONS = (
    "fusion_policy_none",
    "fusion_not_available",
    "fusion_not_unlocked",
    "source_not_found",
    "target_not_available",
    "incomplete_fusion_mapping",
    "not_gameplay",
    "gameplay_not_ready",
    "unsafe_state",
    "bridge_rejected",
    "low_strategic_value",
    "invalid_row_col",
    "terminal_or_transition_state",
    "seed_selection_active",
    "reward_or_unlock_screen_active",
    "action_space_mismatch",
    "exception",
)

FUSION_RESULT_NAMES: Mapping[int, str] = MappingProxyType({
    1030: "DoubleShooer",
    1031: "SunShroom",
    1032: "GatlingPea",
    1033: "TwinFlower",
    1090: "SplitPea",
})
PLANT_NAMES: Mapping[int, str] = MappingProxyType({
    **{definition.plant_type_id: definition.canonical_name for definition in _PLANT_REGISTRY.plants},
    **FUSION_RESULT_NAMES,
})


def normalize_fusion_source(value: Any) -> str:
    """Return the stable source label used by the shared fusion pipeline."""

    source = str(value or FUSION_SOURCE_MANUAL).strip().lower().replace("-", "_")
    return str(_FUSION_SOURCE_ALIASES.get(source, source if source in FUSION_SOURCES else FUSION_SOURCE_MANUAL))


def _ingredient_fallback_metadata(plant_type: int) -> Tuple[int, float]:
    definition = _PLANT_REGISTRY.get_by_id(int(plant_type))
    if definition is None:
        return 0, 0.0
    return int(definition.fallback_cost), float(definition.fallback_cooldown)


@dataclass(frozen=True, slots=True)
class FusionRecipe:
    """Immutable known-result recipe.

    Runtime CardUI values and bridge legality remain authoritative.  The cost,
    cooldown, and unlock fields describe the fallback/validation contract; they
    never override a live seed packet or a bridge rejection.
    """

    source_plant_type: int
    ingredient_plant_type: int
    result_plant_type: int
    result_plant_name: str
    reason: str
    role: str
    scripted_enabled: bool
    ingredient_fallback_cost: int
    ingredient_fallback_cooldown: float
    runtime_cost_authoritative: bool = True
    runtime_cooldown_authoritative: bool = True
    runtime_unlock_authoritative: bool = True

    @property
    def pair(self) -> Tuple[int, int]:
        return int(self.source_plant_type), int(self.ingredient_plant_type)

    def to_legacy_rule(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "predicted_result_name": self.result_plant_name,
                "predicted_result_type": int(self.result_plant_type),
                "reason": self.reason,
                "role": self.role,
                "scripted_enabled": bool(self.scripted_enabled),
            }
        )


@dataclass(frozen=True, slots=True)
class RuntimeFusionCompatibilityCase:
    """A bridge-supported pair for which Python must not invent a result."""

    first_plant_type: int
    second_plant_type: int
    reason: str

    @property
    def canonical_pair(self) -> Tuple[int, int]:
        return tuple(sorted((int(self.first_plant_type), int(self.second_plant_type))))  # type: ignore[return-value]


def _recipe(
    source: int,
    ingredient: int,
    result: int,
    result_name: str,
    reason: str,
    role: str,
    scripted_enabled: bool,
) -> FusionRecipe:
    fallback_cost, fallback_cooldown = _ingredient_fallback_metadata(ingredient)
    return FusionRecipe(
        source_plant_type=int(source),
        ingredient_plant_type=int(ingredient),
        result_plant_type=int(result),
        result_plant_name=str(result_name),
        reason=str(reason),
        role=str(role),
        scripted_enabled=bool(scripted_enabled),
        ingredient_fallback_cost=fallback_cost,
        ingredient_fallback_cooldown=fallback_cooldown,
    )


# The one authoritative result-producing recipe registry.  These IDs match the
# live hybrid game's PlantType enum and observed fusion results.  Numeric seed
# ID 7 remains the base-game Repeater seed and is never conflated with the
# recursive fusion chain.
FUSION_RECIPES: Tuple[FusionRecipe, ...] = (
    _recipe(
        PEASHOOTER_ID,
        PEASHOOTER_ID,
        1030,
        "DoubleShooer",
        "Peashooter-on-Peashooter should improve lane DPS.",
        "dps",
        True,
    ),
    _recipe(
        1030,
        PEASHOOTER_ID,
        1090,
        "SplitPea",
        "Peashooter on DoubleShooer advances the recursive pea fusion chain.",
        "dps",
        True,
    ),
    _recipe(
        1090,
        PEASHOOTER_ID,
        1032,
        "GatlingPea",
        "Peashooter on SplitPea completes the high-tier pea fusion chain.",
        "dps",
        True,
    ),
    _recipe(
        SUNFLOWER_ID,
        SUNFLOWER_ID,
        1033,
        "TwinFlower",
        "SunFlower-on-SunFlower is known economy fusion but is not a defensive assist.",
        "economy",
        False,
    ),
)
FUSION_RECIPES_BY_PAIR: Mapping[Tuple[int, int], FusionRecipe] = MappingProxyType(
    {recipe.pair: recipe for recipe in FUSION_RECIPES}
)

# These pairs are intentionally compatibility-only.  The live bridge probes and
# resolves them; Python must not infer a result plant from either relationship.
RUNTIME_ONLY_FUSION_COMPATIBILITY_CASES: Tuple[RuntimeFusionCompatibilityCase, ...] = (
    RuntimeFusionCompatibilityCase(
        SUNFLOWER_ID,
        PEASHOOTER_ID,
        "The mod exposes SunFlower/Peashooter at runtime, but no stable Python result mapping is known.",
    ),
    RuntimeFusionCompatibilityCase(
        PEASHOOTER_ID,
        CHERRYBOMB_ID,
        "The mod exposes Peashooter/CherryBomb at runtime, but no stable Python result mapping is known.",
    ),
)
RUNTIME_ONLY_FUSION_COMPATIBILITY: frozenset[Tuple[int, int]] = frozenset(
    case.canonical_pair for case in RUNTIME_ONLY_FUSION_COMPATIBILITY_CASES
)

# Deprecated compatibility view.  It is derived from FUSION_RECIPES and deeply
# immutable so callers cannot create recipe/compatibility drift.
FUSION_RULES: Mapping[Tuple[int, int], Mapping[str, Any]] = MappingProxyType(
    {recipe.pair: recipe.to_legacy_rule() for recipe in FUSION_RECIPES}
)

CONEHEAD_TYPES = {2, 12}
BUCKETHEAD_TYPES = {4, 13}


# ---------------------------------------------------------------------------
# Centralized fusion compatibility (single source of truth)
# ---------------------------------------------------------------------------
#
# This is the ONE place that decides whether an existing plant and a selected
# seed packet form a legal fusion pair.  Every consumer -- the model action
# mask, model action execution, manual coach validation, stream coach
# validation, fusion diagnostics, and fusion reward gating -- must consult the
# helpers below instead of hard-coding pairs.  Do not scatter compatibility
# checks across modules.
#
# Plant identity ids match configs/plant_registry.json.

# Fusion-result names are a separate namespace layered over base seed names.
# Result aliases intentionally win for ambiguous text such as ``Repeater``;
# numeric base seed ID 7 still resolves through the canonical plant registry.
FUSION_RESULT_NAME_TO_ID: Mapping[str, int] = MappingProxyType({
    "doubleshooer": 1030,
    "doubleshooter": 1030,
    # Compatibility alias retained for older coach/config text. The live
    # PlantType 1030 canonical name is DoubleShooer; seed ID 7 is Repeater.
    "repeater": 1030,
    "splitpea": 1090,
    "threepeater": 1090,
    "3pea": 1090,
    "gatlingpea": 1032,
    "twinflower": 1033,
    "twinsunflower": 1033,
})
_BASE_PLANT_NAME_TO_ID: Dict[str, int] = {
    key: definition.plant_type_id
    for key, definition in _PLANT_REGISTRY.by_normalized_name.items()
}
_BASE_PLANT_NAME_TO_ID.update(FUSION_RESULT_NAME_TO_ID)
PLANT_NAME_TO_ID: Mapping[str, int] = MappingProxyType(_BASE_PLANT_NAME_TO_ID)

def _build_fusion_compatibility() -> Mapping[int, frozenset[int]]:
    compatibility: Dict[int, Set[int]] = {}
    for recipe in FUSION_RECIPES:
        source, ingredient = recipe.pair
        compatibility.setdefault(source, set()).add(ingredient)
        compatibility.setdefault(ingredient, set()).add(source)
    for case in RUNTIME_ONLY_FUSION_COMPATIBILITY_CASES:
        first, second = case.canonical_pair
        compatibility.setdefault(first, set()).add(second)
        compatibility.setdefault(second, set()).add(first)
    return MappingProxyType({plant: frozenset(partners) for plant, partners in compatibility.items()})


# Deprecated compatibility view, derived solely from recipes and the explicitly
# documented runtime-only cases above.
FUSION_COMPATIBILITY: Mapping[int, frozenset[int]] = _build_fusion_compatibility()


FUSION_TIER_BY_TYPE: Mapping[int, int] = MappingProxyType({
    PEASHOOTER_ID: 0,
    1030: 1,
    1090: 2,
    1032: 3,
    SUNFLOWER_ID: 0,
    1033: 1,
})


def fusion_tier(plant: Any) -> int:
    plant_id = normalize_plant_name_or_id(plant)
    return int(FUSION_TIER_BY_TYPE.get(int(plant_id), 0)) if plant_id is not None else 0


def can_accept_fusion(plant: Any) -> bool:
    plant_id = normalize_plant_name_or_id(plant)
    return bool(plant_id is not None and _FUSION_COMPATIBILITY_SYMMETRIC.get(int(plant_id)))


def board_plant_identity_features(plant: Dict[str, Any]) -> Tuple[float, float]:
    """Two checkpoint-shape-safe identity channels for board cells.

    Base SunFlower and Peashooter retain their historical [1,0] and [0,1]
    encodings. Fused tiers and later plants receive distinct values without
    changing the observation vector length, so the 370k policy can be resumed.
    """
    plant_id = normalize_plant_name_or_id(plant)
    if plant_id == SUNFLOWER_ID:
        return 1.0, 0.0
    if plant_id == PEASHOOTER_ID:
        return 0.0, 1.0
    known = {
        1033: (0.75, 0.0),
        1030: (0.0, 0.75),
        1090: (0.0, 0.50),
        1032: (0.0, 0.25),
        WALLNUT_ID: (-0.25, 0.0),
        CHERRYBOMB_ID: (-0.50, 0.0),
        4: (-0.75, 0.0),  # Potato Mine in the base PvZ plant enum.
        6: (-1.00, 0.0),  # Chomper in the base PvZ plant enum.
    }
    if plant_id in known:
        return known[int(plant_id)]
    if plant_id is None:
        return 0.0, 0.0
    # Deterministic unknown identity; never aliases the historical generic [0,0].
    return -0.10, max(-1.0, min(1.0, (int(plant_id) % 97 + 1) / 100.0))

# Human-readable rejection reasons used by ``get_fusion_illegal_reason`` and
# surfaced verbatim by the coaches and diagnostics.
FUSION_ILLEGAL_NONE = ""
FUSION_ILLEGAL_DISABLED = "fusion_disabled"
FUSION_ILLEGAL_BRIDGE_UNAVAILABLE = "fusion_bridge_unavailable"
FUSION_ILLEGAL_INVALID_ROW = "invalid_row"
FUSION_ILLEGAL_INVALID_COL = "invalid_col"
FUSION_ILLEGAL_INVALID_SEED_SLOT = "invalid_seed_slot"
FUSION_ILLEGAL_EMPTY_TILE = "empty_tile"
FUSION_ILLEGAL_INCOMPATIBLE = "incompatible_pair"
FUSION_ILLEGAL_SEED_UNAVAILABLE = "seed_unavailable"
FUSION_ILLEGAL_COOLDOWN = "cooldown_not_ready"
FUSION_ILLEGAL_INSUFFICIENT_SUN = "insufficient_sun"


def fusion_recipe(existing_plant: Any, selected_seed: Any) -> Optional[FusionRecipe]:
    """Return an exact directional result recipe, never a compatibility guess."""

    existing = normalize_plant_name_or_id(existing_plant)
    selected = normalize_plant_name_or_id(selected_seed)
    if existing is None or selected is None:
        return None
    return FUSION_RECIPES_BY_PAIR.get((int(existing), int(selected)))


def fusion_compatibility_kind(existing_plant: Any, selected_seed: Any) -> str:
    """Classify a compatible pair without inventing a runtime-only result."""

    existing = normalize_plant_name_or_id(existing_plant)
    selected = normalize_plant_name_or_id(selected_seed)
    if existing is None or selected is None:
        return "none"
    pair = (int(existing), int(selected))
    if pair in FUSION_RECIPES_BY_PAIR or (pair[1], pair[0]) in FUSION_RECIPES_BY_PAIR:
        return "recipe"
    if tuple(sorted(pair)) in RUNTIME_ONLY_FUSION_COMPATIBILITY:
        return "runtime_only"
    return "none"


@dataclass(frozen=True, slots=True)
class FusionIntent:
    """Source-attributed, tile-scoped request consumed by every fusion path."""

    source: str
    source_row: int
    source_col: int
    ingredient_seed_slot_index: int
    source_plant_type: int
    ingredient_plant_type: int
    source_instance_id: int = 0
    ingredient_card_instance_id: int = 0
    predicted_result_type: int = -1
    predicted_result_name: str = ""
    requested_action: int = 0
    executed_action: int = 0
    source_plant_name: str = ""
    ingredient_plant_name: str = ""
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}), repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", normalize_fusion_source(self.source))
        for field_name in (
            "source_row",
            "source_col",
            "ingredient_seed_slot_index",
            "source_plant_type",
            "ingredient_plant_type",
            "source_instance_id",
            "ingredient_card_instance_id",
            "predicted_result_type",
            "requested_action",
            "executed_action",
        ):
            object.__setattr__(self, field_name, int(getattr(self, field_name)))
        metadata = _deep_thaw(self.metadata) if isinstance(self.metadata, Mapping) else {}
        object.__setattr__(self, "metadata", _deep_freeze(metadata))

    @property
    def recipe(self) -> Optional[FusionRecipe]:
        return fusion_recipe(self.source_plant_type, self.ingredient_plant_type)

    @property
    def compatibility_kind(self) -> str:
        return fusion_compatibility_kind(self.source_plant_type, self.ingredient_plant_type)

    def candidate_dict(self) -> Dict[str, Any]:
        candidate = _deep_thaw(self.metadata)
        if not isinstance(candidate, dict):
            candidate = {}
        candidate.update(
            {
                "source_plant_type": int(self.source_plant_type),
                "source_plant_name": self.source_plant_name or plant_name(self.source_plant_type),
                "source_instance_id": int(self.source_instance_id),
                "source_row": int(self.source_row),
                "source_col": int(self.source_col),
                "target_or_ingredient_type": int(self.ingredient_plant_type),
                "target_or_ingredient_name": self.ingredient_plant_name or plant_name(self.ingredient_plant_type),
                "ingredient_seed_slot_index": int(self.ingredient_seed_slot_index),
                "ingredient_card_instance_id": int(self.ingredient_card_instance_id),
                "predicted_result_type": int(self.predicted_result_type),
                "predicted_result_name": str(self.predicted_result_name),
            }
        )
        return candidate

    def bridge_command_dict(self) -> Dict[str, Any]:
        return {
            "source_instance_id": int(self.source_instance_id),
            "source_row": int(self.source_row),
            "source_col": int(self.source_col),
            "source_plant_type": int(self.source_plant_type),
            "ingredient_seed_slot_index": int(self.ingredient_seed_slot_index),
            "ingredient_plant_type": int(self.ingredient_plant_type),
            "predicted_result_type": int(self.predicted_result_type),
            "predicted_result_name": str(self.predicted_result_name),
        }


def fusion_intent_from_candidate(
    candidate: Mapping[str, Any],
    *,
    source: str,
    requested_action: int = 0,
    executed_action: int = 0,
    metadata: Optional[Mapping[str, Any]] = None,
) -> FusionIntent:
    """Compatibility factory for bridge/model/scripted/coach candidate dicts."""

    source_type = _safe_int(candidate.get("source_plant_type"), default=-1)
    ingredient_type = _safe_int(candidate.get("target_or_ingredient_type"), candidate.get("ingredient_plant_type"), default=-1)
    recipe = FUSION_RECIPES_BY_PAIR.get((source_type, ingredient_type))
    compatibility_kind = fusion_compatibility_kind(source_type, ingredient_type)
    if recipe is not None:
        predicted_type = int(recipe.result_plant_type)
        predicted_name = str(recipe.result_plant_name)
    elif compatibility_kind != "none":
        # Symmetric/reverse and runtime-only compatibility may be executable at
        # the bridge, but Python has no authoritative directional result.
        predicted_type = -1
        predicted_name = ""
    else:
        predicted_type = _safe_int(candidate.get("predicted_result_type"), default=-1)
        predicted_name = str(candidate.get("predicted_result_name") or "")
    candidate_metadata = _deep_thaw(candidate)
    if not isinstance(candidate_metadata, dict):
        candidate_metadata = {}
    if isinstance(metadata, Mapping):
        extra_metadata = _deep_thaw(metadata)
        if isinstance(extra_metadata, dict):
            candidate_metadata.update(extra_metadata)
    return FusionIntent(
        source=source,
        source_row=_safe_int(candidate.get("source_row"), default=-1),
        source_col=_safe_int(candidate.get("source_col"), candidate.get("source_column"), default=-1),
        ingredient_seed_slot_index=_safe_int(candidate.get("ingredient_seed_slot_index"), default=-1),
        source_plant_type=source_type,
        ingredient_plant_type=ingredient_type,
        source_instance_id=_safe_int(candidate.get("source_instance_id"), default=0),
        ingredient_card_instance_id=_safe_int(candidate.get("ingredient_card_instance_id"), default=0),
        predicted_result_type=predicted_type,
        predicted_result_name=predicted_name,
        requested_action=int(requested_action),
        executed_action=int(executed_action),
        source_plant_name=str(candidate.get("source_plant_name") or ""),
        ingredient_plant_name=str(candidate.get("target_or_ingredient_name") or candidate.get("ingredient_plant_name") or ""),
        metadata=MappingProxyType(candidate_metadata),
    )


@dataclass(frozen=True, slots=True)
class FusionDecision:
    """Pure legality decision reused before bridge execution by all sources."""

    intent: FusionIntent
    legal: bool
    rejection_reason: str = ""
    recipe: Optional[FusionRecipe] = None
    compatibility_kind: str = "none"
    bridge_command: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        bridge_command = _deep_thaw(self.bridge_command) if isinstance(self.bridge_command, Mapping) else {}
        object.__setattr__(self, "bridge_command", _deep_freeze(bridge_command))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.intent.source,
            "legal": bool(self.legal),
            "rejection_reason": str(self.rejection_reason),
            "compatibility_kind": str(self.compatibility_kind),
            "bridge_command": _deep_thaw(self.bridge_command),
            "candidate": self.intent.candidate_dict(),
        }


@dataclass(frozen=True, slots=True)
class FusionExecution:
    """One bridge outcome plus its exactly-once accounting identity."""

    decision: FusionDecision
    event_id: str
    attempted: bool
    success: bool
    rejection_reason: str
    result: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}), repr=False, compare=False)

    def __post_init__(self) -> None:
        result = _deep_thaw(self.result) if isinstance(self.result, Mapping) else {}
        object.__setattr__(self, "result", _deep_freeze(result))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.to_dict(),
            "event_id": str(self.event_id),
            "source": str(self.decision.intent.source),
            "attempted": bool(self.attempted),
            "success": bool(self.success),
            "rejection_reason": str(self.rejection_reason),
            "result": _deep_thaw(self.result),
        }


def _symmetric_closure(table: Mapping[int, Iterable[int]]) -> Mapping[int, frozenset[int]]:
    closure: Dict[int, Set[int]] = {}
    for source, partners in table.items():
        for partner in partners:
            closure.setdefault(int(source), set()).add(int(partner))
            closure.setdefault(int(partner), set()).add(int(source))
    return MappingProxyType({plant: frozenset(partners) for plant, partners in closure.items()})


_FUSION_COMPATIBILITY_SYMMETRIC: Mapping[int, frozenset[int]] = _symmetric_closure(FUSION_COMPATIBILITY)


def normalize_plant_name_or_id(value: Any) -> Optional[int]:
    """Convert a plant name/id/seed-slot/plant dict into a canonical plant id.

    Returns ``None`` when the value cannot be resolved to a plant id.  Accepts
    raw ints, numeric strings, canonical names/aliases, and dicts shaped like a
    seed slot (``plantType``) or a board plant (``type``).
    """

    if value is None:
        return None
    if isinstance(value, bool):  # avoid treating True/False as 1/0
        return None
    if isinstance(value, int):
        return int(value) if value >= 0 else None
    if isinstance(value, float):
        return int(value) if value >= 0 else None
    if isinstance(value, dict):
        for key in ("plantType", "type", "plant_type", "plantTypeId", "plant_type_id"):
            if value.get(key) is not None:
                resolved = normalize_plant_name_or_id(value.get(key))
                if resolved is not None:
                    return resolved
        for key in ("plantTypeName", "typeName", "name", "plant_name", "canonical_name"):
            if value.get(key):
                resolved = normalize_plant_name_or_id(value.get(key))
                if resolved is not None:
                    return resolved
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lstrip("-").isdigit():
        ivalue = int(text)
        return ivalue if ivalue >= 0 else None
    return PLANT_NAME_TO_ID.get(normalize_plant_name(text))


def are_fusion_compatible(existing_plant: Any, selected_seed: Any) -> bool:
    """Return True only if existing plant and selected seed are a legal pair.

    The relationship is symmetric: existing SunFlower + selected Peashooter and
    existing Peashooter + selected SunFlower are both legal.
    """

    existing = normalize_plant_name_or_id(existing_plant)
    selected = normalize_plant_name_or_id(selected_seed)
    if existing is None or selected is None:
        return False
    if selected in _FUSION_COMPATIBILITY_SYMMETRIC.get(existing, ()):  # type: ignore[arg-type]
        return True
    return existing in _FUSION_COMPATIBILITY_SYMMETRIC.get(selected, ())  # type: ignore[arg-type]


def fusion_compatibility_table() -> Dict[str, List[str]]:
    """Name-keyed, sorted view of the symmetric compatibility table.

    Intended for diagnostics/UI; the ids in ``FUSION_COMPATIBILITY`` remain the
    machine-readable source of truth.
    """

    table: Dict[str, List[str]] = {}
    for plant_id, partners in sorted(_FUSION_COMPATIBILITY_SYMMETRIC.items()):
        table[plant_name(plant_id)] = sorted(plant_name(partner) for partner in partners)
    return table


def plant_type_at_cell(
    observation: Dict[str, Any],
    row: int,
    col: int,
    *,
    facts: Optional[StepFacts] = None,
) -> Optional[int]:
    """Return the plant type occupying (row, col), or None if the tile is empty."""

    snapshot = facts or build_step_facts(observation if isinstance(observation, dict) else {})
    plant = snapshot.compat_occupant_by_cell.get((int(row), int(col)))
    return plant.plant_type if plant is not None and plant.plant_type >= 0 else None


def _seed_slot_payload(slot: SeedSlotFact) -> Dict[str, Any]:
    return {
        "slotIndex": slot.slot_index,
        "plantType": slot.plant_type,
        "plantTypeName": slot.plant_type_name,
        "seedCost": slot.seed_cost,
        "ready": slot.fusion_ready,
        "disabled": slot.disabled,
        "usable": slot.fusion_usable,
        "currentCooldown": slot.current_cooldown,
        "fullCooldown": slot.full_cooldown,
        "rawCooldown": slot.raw_cooldown,
        "cardInstanceId": slot.card_instance_id,
    }


def seed_slot_entry(
    observation: Dict[str, Any],
    seed_slot: int,
    plant_types: Optional[Iterable[int]] = None,
    *,
    facts: Optional[StepFacts] = None,
) -> Optional[Dict[str, Any]]:
    """Resolve a seed-slot index to its observation slot dict (by slotIndex)."""

    snapshot = facts or build_step_facts(
        observation if isinstance(observation, dict) else {},
        tuple(plant_types or ()),
    )
    slot = snapshot.seed_slot(int(seed_slot))
    return _seed_slot_payload(slot) if slot is not None else None


def seed_plant_type_for_slot(
    observation: Dict[str, Any],
    seed_slot: int,
    plant_types: Optional[Iterable[int]] = None,
    *,
    facts: Optional[StepFacts] = None,
) -> Optional[int]:
    slot = seed_slot_entry(
        observation,
        seed_slot,
        plant_types,
        facts=facts,
    )
    if not isinstance(slot, dict):
        return None
    plant_type = _safe_int(slot.get("plantType"), slot.get("type"), default=-1)
    return plant_type if plant_type >= 0 else None


def _seed_fact_resource_reason(facts: StepFacts, slot: SeedSlotFact) -> str:
    if not slot.fusion_usable or slot.disabled:
        return FUSION_ILLEGAL_SEED_UNAVAILABLE
    if not slot.fusion_ready or (
        slot.full_cooldown > 0.05 and slot.current_cooldown > 0.05
    ):
        return FUSION_ILLEGAL_COOLDOWN
    if facts.sun < slot.seed_cost:
        return FUSION_ILLEGAL_INSUFFICIENT_SUN
    return FUSION_ILLEGAL_NONE


def get_fusion_illegal_reason(
    observation: Dict[str, Any],
    row: int,
    col: int,
    seed_slot: int,
    *,
    fusion_enabled: bool = True,
    fusion_bridge_available: bool = True,
    plant_types: Optional[Iterable[int]] = None,
    check_seed_resources: bool = True,
    facts: Optional[StepFacts] = None,
) -> str:
    """Return "" if fusion is legal, else a human-readable rejection reason.

    Legality order (most fundamental first): fusion enabled -> bridge available
    -> row/col/seed-slot valid -> tile occupied -> compatible pair -> seed
    available/cooldown/sun.  Empty-tile and incompatible-pair are reported
    before transient resource reasons so callers get the stable signal.
    """

    observation = observation if isinstance(observation, dict) else {}
    snapshot = facts or build_step_facts(observation, tuple(plant_types or ()))
    if not fusion_enabled:
        return FUSION_ILLEGAL_DISABLED
    if not fusion_bridge_available:
        return FUSION_ILLEGAL_BRIDGE_UNAVAILABLE
    rows = _safe_int(observation.get("rowCount"), default=5)
    cols = _safe_int(observation.get("columnCount"), default=10)
    if not (0 <= int(row) < rows):
        return FUSION_ILLEGAL_INVALID_ROW
    if not (0 <= int(col) < cols):
        return FUSION_ILLEGAL_INVALID_COL
    slot = snapshot.seed_slot(int(seed_slot))
    if slot is None:
        return FUSION_ILLEGAL_INVALID_SEED_SLOT
    seed_type = slot.plant_type
    if seed_type < 0:
        return FUSION_ILLEGAL_INVALID_SEED_SLOT
    existing_type = plant_type_at_cell(observation, int(row), int(col), facts=snapshot)
    if existing_type is None:
        return FUSION_ILLEGAL_EMPTY_TILE
    if not are_fusion_compatible(existing_type, seed_type):
        return FUSION_ILLEGAL_INCOMPATIBLE
    if check_seed_resources:
        resource_reason = _seed_fact_resource_reason(snapshot, slot)
        if resource_reason:
            return resource_reason
    return FUSION_ILLEGAL_NONE


def is_legal_fusion_action(
    observation: Dict[str, Any],
    row: int,
    col: int,
    seed_slot: int,
    **kwargs: Any,
) -> bool:
    """Return True only if the fusion is fully legal (see get_fusion_illegal_reason)."""

    return get_fusion_illegal_reason(observation, row, col, seed_slot, **kwargs) == FUSION_ILLEGAL_NONE


def validate_fusion_intent(
    intent: FusionIntent,
    observation: Dict[str, Any],
    *,
    fusion_enabled: bool = True,
    fusion_bridge_available: bool = True,
    plant_types: Optional[Iterable[int]] = None,
    check_seed_resources: bool = True,
    precondition_rejection: str = "",
    require_known_recipe: bool = False,
    facts: Optional[StepFacts] = None,
) -> FusionDecision:
    """Pure, source-independent fusion legality decision.

    ``precondition_rejection`` carries source-specific lifecycle checks (fresh
    coach command, scripted safety, and so on) into the shared pipeline without
    changing the long-standing reason ordering inside the common validator.
    The Unity bridge still decides whether a legal decision actually succeeds.
    """

    rejection = str(precondition_rejection or "")
    observation = observation if isinstance(observation, dict) else {}
    snapshot = facts or build_step_facts(observation, tuple(plant_types or ()))
    if not rejection:
        rejection = get_fusion_illegal_reason(
            observation,
            int(intent.source_row),
            int(intent.source_col),
            int(intent.ingredient_seed_slot_index),
            fusion_enabled=bool(fusion_enabled),
            fusion_bridge_available=bool(fusion_bridge_available),
            plant_types=plant_types,
            check_seed_resources=bool(check_seed_resources),
            facts=snapshot,
        )

        # Preserve tile scoping and duplicate-slot identity.  A tile/slot that
        # changed between parsing and execution is stale even if the newly
        # observed pair happens to be compatible.
        fundamental_rejections = {
            FUSION_ILLEGAL_DISABLED,
            FUSION_ILLEGAL_BRIDGE_UNAVAILABLE,
            FUSION_ILLEGAL_INVALID_ROW,
            FUSION_ILLEGAL_INVALID_COL,
            FUSION_ILLEGAL_INVALID_SEED_SLOT,
            FUSION_ILLEGAL_EMPTY_TILE,
        }
        if rejection not in fundamental_rejections:
            actual_source = plant_type_at_cell(
                observation,
                intent.source_row,
                intent.source_col,
                facts=snapshot,
            )
            if actual_source is None or int(actual_source) != int(intent.source_plant_type):
                rejection = "source_not_found"
            else:
                if intent.source_instance_id > 0:
                    matching_instances = {
                        plant.instance_id
                        for plant in snapshot.primary_plant_stacks_by_cell.get(
                            (intent.source_row, intent.source_col),
                            (),
                        )
                    }
                    if matching_instances - {0} and intent.source_instance_id not in matching_instances:
                        rejection = "source_not_found"
                actual_ingredient = seed_plant_type_for_slot(
                    observation,
                    intent.ingredient_seed_slot_index,
                    plant_types,
                    facts=snapshot,
                )
                if not rejection and (
                    actual_ingredient is None or int(actual_ingredient) != int(intent.ingredient_plant_type)
                ):
                    rejection = "target_not_available"

    if not rejection:
        candidate_block = str(intent.metadata.get("fusion_blocked_reason") or "")
        candidate_legal = intent.metadata.get("fusion_legal")
        if candidate_block:
            rejection = candidate_block if candidate_block in FUSION_REJECTION_REASONS else "bridge_rejected"
        elif candidate_legal is False:
            rejection = "bridge_rejected"
    if not rejection and require_known_recipe and intent.recipe is None:
        rejection = "incomplete_fusion_mapping"

    return FusionDecision(
        intent=intent,
        legal=not bool(rejection),
        rejection_reason=rejection,
        recipe=intent.recipe,
        compatibility_kind=intent.compatibility_kind,
        bridge_command=MappingProxyType(intent.bridge_command_dict()),
    )


def normalize_fusion_policy(value: Any) -> str:
    policy = str(value or FUSION_POLICY_NONE).strip().lower()
    policy = FUSION_POLICY_ALIASES.get(policy, policy)
    if policy not in FUSION_POLICIES:
        raise ValueError(f"unknown fusion policy: {value!r}")
    return policy


def plant_name(plant_type: Any, fallback: str = "") -> str:
    try:
        plant_id = int(plant_type)
    except (TypeError, ValueError):
        return fallback or "unknown"
    if fallback:
        return fallback
    if plant_id in FUSION_RESULT_NAMES:
        return FUSION_RESULT_NAMES[plant_id]
    return _PLANT_REGISTRY.canonical_name(plant_id)


def default_fusion_diagnostics(policy: str = FUSION_POLICY_NONE) -> Dict[str, Any]:
    policy = normalize_fusion_policy(policy)
    return {
        "fusion_policy": policy,
        "fusion_available": False,
        "fusion_candidate_count": 0,
        "fusion_candidates": [],
        "fusion_top_candidate": None,
        # Standardized live fusion-availability / incompatibility diagnostics.
        "fusion_actions_available_count": 0,
        "fusion_candidate_tiles": [],
        "fusion_last_illegal_reason": "",
        "fusion_last_incompatible_pair": None,
        "fusion_actions_masked_empty_tile": 0,
        "fusion_actions_masked_incompatible_count": 0,
        "fusion_actions_masked_disabled_count": 0,
        "fusion_actions_masked_cooldown_count": 0,
        "fusion_actions_masked_sun_count": 0,
        "fusion_compatibility_table": fusion_compatibility_table(),
        # Fusion reward accounting (populated by the env reward policy each step).
        "fusion_reward_total": 0.0,
        "fusion_attempt_reward_total": 0.0,
        "fusion_success_reward_total": 0.0,
        "fusion_new_recipe_reward_total": 0.0,
        "fusion_recursive_reward_total": 0.0,
        "fusion_tier_reward_total": 0.0,
        "fusion_repeat_decay_total": 0.0,
        "fusion_threatened_row_bonus_total": 0.0,
        "fusion_active_wave_bonus_total": 0.0,
        "fusion_defensive_value_bonus_total": 0.0,
        "fusion_incompatible_penalty_total": 0.0,
        "fusion_empty_tile_penalty_total": 0.0,
        "fusion_failed_penalty_total": 0.0,
        "fusion_bridge_error_penalty_total": 0.0,
        "fusion_spam_penalty_total": 0.0,
        "fusion_reward_capped": False,
        "fusion_last_reward_delta": 0.0,
        "fusion_last_reward_reason": "",
        "fusion_last_usefulness_bonus": 0.0,
        "fusion_last_source": "",
        "fusion_last_attempt": None,
        "fusion_last_result": None,
        "fusion_last_execution_mode": "",
        "fusion_last_bridge_method_used": "",
        "fusion_last_bridge_result_reason": "",
        "fusion_last_duplicate_stack_detected": False,
        "last_fusion_scope": "",
        "last_fusion_changed_tile_count": 0,
        "last_fusion_non_source_tiles_changed": False,
        "last_fusion_global_side_effect": False,
        "last_executed_coach_command_id": None,
        "fusion_last_rejected_reason": "",
        "fusion_attempted_count": 0,
        "fusion_success_count": 0,
        "fusion_failed_count": 0,
        "fusion_rejected_count": 0,
        "fusion_rejected_reasons": {},
        "fusion_by_result_type": {},
        "fusion_by_source_type": {},
        "fusion_by_row": {},
        "fusion_attempts_by_pair": {},
        "fusion_successes_by_pair": {},
        "fusion_failures_by_pair": {},
        "fusion_result_counts": {},
        "fusion_depth_counts": {},
        "recursive_fusion_count": 0,
        "high_tier_fusion_count": 0,
        "highest_fusion_tier": 0,
        "fusion_under_threat_count": 0,
        "fusion_near_buckethead_count": 0,
        "fusion_near_conehead_count": 0,
        "fusion_estimated_mower_save_count": 0,
        "fusion_kills_after_use_total": 0,
        "fusion_avg_kills_after_use": 0.0,
        "fusion_bridge_error_count": 0,
        "fusion_unsafe_state_block_count": 0,
    }


def build_fusion_diagnostics(
    policy: str,
    observation: Dict[str, Any],
    *,
    bridge_probe: Optional[Dict[str, Any]] = None,
    bridge_error: Optional[str] = None,
    facts: Optional[StepFacts] = None,
) -> Dict[str, Any]:
    diag = default_fusion_diagnostics(policy)
    if diag["fusion_policy"] == FUSION_POLICY_NONE:
        return diag

    if bridge_error:
        diag["fusion_bridge_error_count"] = 1

    candidates = scan_fusion_candidates(observation, bridge_probe=bridge_probe, facts=facts)
    diag["fusion_candidates"] = candidates
    diag["fusion_candidate_count"] = len(candidates)
    legal_candidates = [candidate for candidate in candidates if bool(candidate.get("fusion_legal"))]
    diag["fusion_available"] = bool(legal_candidates)
    diag["fusion_actions_available_count"] = len(legal_candidates)
    diag["fusion_candidate_tiles"] = sorted(
        {
            (int(candidate.get("source_row", -1)), int(candidate.get("source_col", -1)))
            for candidate in legal_candidates
        }
    )
    if candidates:
        top = max(candidates, key=lambda item: float(item.get("strategic_score") or 0.0))
        diag["fusion_top_candidate"] = compact_candidate(top)
    return diag


def scan_fusion_candidates(
    observation: Dict[str, Any],
    *,
    bridge_probe: Optional[Dict[str, Any]] = None,
    facts: Optional[StepFacts] = None,
) -> List[Dict[str, Any]]:
    snapshot = facts or build_step_facts(observation)
    plants = snapshot.plants
    slots = tuple(slot for slot in snapshot.seed_slots if slot.valid)
    rows = _safe_int(observation.get("rowCount"), default=5)
    cols = _safe_int(observation.get("columnCount"), default=10)
    sun = snapshot.sun
    probe_candidates = _probe_candidate_index(bridge_probe)
    candidates: List[Dict[str, Any]] = []

    for source in plants:
        source_row = source.row
        source_col = source.column
        source_type = source.plant_type
        if not (0 <= source_row < rows and 0 <= source_col < cols):
            continue
        for slot in slots:
            ingredient_type = slot.plant_type
            slot_index = slot.slot_index
            if slot_index < 0 or ingredient_type < 0:
                continue
            rule = FUSION_RULES.get((source_type, ingredient_type))
            blocked_reason = "" if rule else "incomplete_fusion_mapping"
            fusion_legal = bool(rule)
            probe = _match_probe_candidate(
                probe_candidates,
                source.instance_id,
                source_row,
                source_col,
                source_type,
                slot_index,
                ingredient_type,
            )
            if probe is not None:
                probe_legal = bool(
                    probe.get("fusionLegal")
                    if "fusionLegal" in probe
                    else probe.get("fusion_legal")
                )
                probe_reason = str(
                    probe.get("fusionBlockedReason")
                    or probe.get("fusion_blocked_reason")
                    or probe.get("blockedReason")
                    or ""
                )
                fusion_legal = bool(rule) and probe_legal
                if probe_reason:
                    blocked_reason = probe_reason
                elif not probe_legal:
                    blocked_reason = "bridge_rejected"
                elif not rule:
                    blocked_reason = "incomplete_fusion_mapping"
                else:
                    blocked_reason = ""
            elif not slot.legacy_ready:
                fusion_legal = False
                blocked_reason = "target_not_available"
            elif rule:
                fusion_legal = False
                blocked_reason = "bridge_rejected"

            lane = _lane_context(observation, source_row, source_col, facts=snapshot)
            health_ratio = source.fusion_health_ratio
            strategic_score, reason = _strategic_score(source_type, ingredient_type, rule, lane, health_ratio)
            if rule and not bool(rule.get("scripted_enabled")) and not blocked_reason:
                fusion_legal = False
                blocked_reason = "low_strategic_value"

            candidates.append(
                {
                    "source_plant_name": source.type_name or plant_name(source_type),
                    "source_plant_type": source_type,
                    "source_instance_id": source.instance_id,
                    "source_row": source_row,
                    "source_col": source_col,
                    "target_or_ingredient_name": slot.plant_type_name or plant_name(ingredient_type),
                    "target_or_ingredient_type": ingredient_type,
                    "ingredient_seed_slot_index": slot_index,
                    "ingredient_card_instance_id": slot.card_instance_id,
                    "predicted_result_name": str((rule or {}).get("predicted_result_name") or "unknown"),
                    "predicted_result_type": _safe_int((rule or {}).get("predicted_result_type"), default=-1),
                    "fusion_legal": bool(fusion_legal and not blocked_reason),
                    "fusion_blocked_reason": blocked_reason,
                    "lane_danger_score": lane["lane_danger_score"],
                    "nearby_zombie_count": lane["nearby_zombie_count"],
                    "nearby_conehead_count": lane["nearby_conehead_count"],
                    "nearby_buckethead_count": lane["nearby_buckethead_count"],
                    "nearest_zombie_distance": lane["nearest_zombie_distance"],
                    "source_health_ratio": health_ratio,
                    "strategic_score": strategic_score,
                    "reason": reason,
                }
            )

    return sorted(candidates, key=lambda item: float(item.get("strategic_score") or 0.0), reverse=True)


def choose_scripted_fusion_candidate(diagnostics: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    policy = normalize_fusion_policy(diagnostics.get("fusion_policy"))
    if policy == FUSION_POLICY_NONE:
        return None, "fusion_policy_none"
    candidates = diagnostics.get("fusion_candidates")
    if not isinstance(candidates, list) or not candidates:
        return None, "fusion_not_available"

    best_rejection = "low_strategic_value"
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        blocked = str(candidate.get("fusion_blocked_reason") or "")
        if blocked:
            best_rejection = blocked
            continue
        if not bool(candidate.get("fusion_legal")):
            best_rejection = "bridge_rejected"
            continue
        if not _is_scripted_allowlisted(candidate):
            best_rejection = "incomplete_fusion_mapping"
            continue
        score = float(candidate.get("strategic_score") or 0.0)
        lane_danger = float(candidate.get("lane_danger_score") or 0.0)
        pressure = (
            int(candidate.get("nearby_zombie_count") or 0) >= 2
            or int(candidate.get("nearby_conehead_count") or 0) > 0
            or int(candidate.get("nearby_buckethead_count") or 0) > 0
        )
        if score >= 1.25 and (lane_danger >= 0.25 or pressure):
            return candidate, ""
        best_rejection = "low_strategic_value"
    return None, best_rejection


def validate_scripted_fusion_candidate(
    candidate: Optional[Dict[str, Any]],
    observation: Dict[str, Any],
    *,
    action_count: int,
    expected_action_count: int,
    reset_active: bool = False,
    action_already_executed: bool = False,
) -> str:
    if candidate is None:
        return "fusion_not_available"
    screen = str(observation.get("screenState") or "")
    precondition = (
        "action_space_mismatch" if action_count != expected_action_count else
        "unsafe_state" if action_already_executed or reset_active else
        "terminal_or_transition_state" if bool(observation.get("done")) or bool(observation.get("over")) else
        "seed_selection_active" if bool(observation.get("seedSelectionActive")) or screen == "seed_selection" else
        "reward_or_unlock_screen_active" if bool(observation.get("blockingRewardUiActive")) or screen in {"reward_unlock", "reward_screen", "level_complete_trophy"} else
        "not_gameplay" if not bool(observation.get("boardFound", True)) else
        "gameplay_not_ready" if not bool(observation.get("gameplayReady")) else ""
    )
    decision = validate_fusion_intent(
        fusion_intent_from_candidate(candidate, source=FUSION_SOURCE_SCRIPTED),
        observation,
        precondition_rejection=precondition,
        require_known_recipe=True,
    )
    reason = {
        FUSION_ILLEGAL_INVALID_ROW: "invalid_row_col",
        FUSION_ILLEGAL_INVALID_COL: "invalid_row_col",
        FUSION_ILLEGAL_INVALID_SEED_SLOT: "target_not_available",
        FUSION_ILLEGAL_SEED_UNAVAILABLE: "target_not_available",
        FUSION_ILLEGAL_COOLDOWN: "target_not_available",
        FUSION_ILLEGAL_INSUFFICIENT_SUN: "target_not_available",
    }.get(decision.rejection_reason, decision.rejection_reason)
    return reason or ("low_strategic_value" if float(candidate.get("strategic_score") or 0.0) < 1.25 else "")


def compact_candidate(candidate: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(candidate, dict):
        return None
    keys = (
        "source_plant_name",
        "source_plant_type",
        "source_row",
        "source_col",
        "target_or_ingredient_name",
        "target_or_ingredient_type",
        "predicted_result_name",
        "predicted_result_type",
        "fusion_legal",
        "fusion_blocked_reason",
        "lane_danger_score",
        "nearby_zombie_count",
        "nearby_conehead_count",
        "nearby_buckethead_count",
        "nearest_zombie_distance",
        "strategic_score",
        "reason",
    )
    return {key: candidate.get(key) for key in keys if key in candidate}


def apply_fusion_attempt_result(
    diagnostics: Dict[str, Any],
    candidate: Optional[Dict[str, Any]],
    result: Optional[Dict[str, Any]],
    *,
    rejected_reason: str = "",
    bridge_error: Optional[str] = None,
) -> Dict[str, Any]:
    diag = dict(diagnostics)
    reasons = Counter(diag.get("fusion_rejected_reasons") or {})
    if isinstance(candidate, dict):
        diag["fusion_last_attempt"] = compact_candidate(candidate)
    if bridge_error:
        diag["fusion_bridge_error_count"] = int(diag.get("fusion_bridge_error_count") or 0) + 1
        rejected_reason = rejected_reason or "exception"
    if rejected_reason:
        reasons[rejected_reason] += 1
        diag["fusion_rejected_count"] = int(diag.get("fusion_rejected_count") or 0) + 1
        diag["fusion_last_rejected_reason"] = rejected_reason
        diag["fusion_last_illegal_reason"] = rejected_reason
        if rejected_reason == FUSION_ILLEGAL_INCOMPATIBLE and isinstance(candidate, dict):
            diag["fusion_actions_masked_incompatible_count"] = (
                int(diag.get("fusion_actions_masked_incompatible_count") or 0) + 1
            )
            diag["fusion_last_incompatible_pair"] = {
                "existing": str(candidate.get("source_plant_name") or plant_name(candidate.get("source_plant_type"))),
                "selected": str(
                    candidate.get("target_or_ingredient_name")
                    or plant_name(candidate.get("target_or_ingredient_type"))
                ),
                "row": _safe_int(candidate.get("source_row"), default=-1),
                "col": _safe_int(candidate.get("source_col"), default=-1),
            }
        if rejected_reason in {"unsafe_state", "not_gameplay", "gameplay_not_ready", "terminal_or_transition_state"}:
            diag["fusion_unsafe_state_block_count"] = int(diag.get("fusion_unsafe_state_block_count") or 0) + 1
    if result is not None:
        diag["fusion_attempted_count"] = int(diag.get("fusion_attempted_count") or 0) + 1
        source_type = _safe_int((candidate or {}).get("source_plant_type"), default=-1)
        ingredient_type = _safe_int((candidate or {}).get("target_or_ingredient_type"), default=-1)
        pair_key = f"{plant_name(ingredient_type)} + {plant_name(source_type)}"
        _bump_dict(diag, "fusion_attempts_by_pair", pair_key)
        success = bool(
            result.get("fusionSucceeded")
            if "fusionSucceeded" in result
            else result.get("fusion_success")
        )
        if success:
            diag["fusion_success_count"] = int(diag.get("fusion_success_count") or 0) + 1
            _bump_dict(diag, "fusion_successes_by_pair", pair_key)
            _bump_dict(diag, "fusion_by_result_type", str((candidate or {}).get("predicted_result_type", -1)))
            _bump_dict(diag, "fusion_by_source_type", str((candidate or {}).get("source_plant_type", -1)))
            _bump_dict(diag, "fusion_by_row", str((candidate or {}).get("source_row", -1)))
            resulting = result.get("resultingPlantAfter") if isinstance(result.get("resultingPlantAfter"), dict) else {}
            result_type = _safe_int(
                resulting.get("plantType"),
                resulting.get("type"),
                (candidate or {}).get("predicted_result_type"),
                default=-1,
            )
            result_name = str(
                resulting.get("plantTypeName")
                or resulting.get("typeName")
                or (candidate or {}).get("predicted_result_name")
                or plant_name(result_type)
            )
            depth = fusion_tier(result_type)
            _bump_dict(diag, "fusion_result_counts", result_name)
            _bump_dict(diag, "fusion_depth_counts", f"depth_{depth}")
            diag["highest_fusion_tier"] = max(int(diag.get("highest_fusion_tier") or 0), depth)
            if fusion_tier(source_type) > 0:
                diag["recursive_fusion_count"] = int(diag.get("recursive_fusion_count") or 0) + 1
            if depth >= 2:
                diag["high_tier_fusion_count"] = int(diag.get("high_tier_fusion_count") or 0) + 1
            if float((candidate or {}).get("lane_danger_score") or 0.0) > 0.0:
                diag["fusion_under_threat_count"] = int(diag.get("fusion_under_threat_count") or 0) + 1
            if int((candidate or {}).get("nearby_buckethead_count") or 0) > 0:
                diag["fusion_near_buckethead_count"] = int(diag.get("fusion_near_buckethead_count") or 0) + 1
            if int((candidate or {}).get("nearby_conehead_count") or 0) > 0:
                diag["fusion_near_conehead_count"] = int(diag.get("fusion_near_conehead_count") or 0) + 1
            if float((candidate or {}).get("lane_danger_score") or 0.0) >= 0.6:
                diag["fusion_estimated_mower_save_count"] = int(diag.get("fusion_estimated_mower_save_count") or 0) + 1
        else:
            diag["fusion_failed_count"] = int(diag.get("fusion_failed_count") or 0) + 1
            reason = str(result.get("fusionRejectedReason") or result.get("illegalReason") or "bridge_rejected")
            _bump_dict(diag, "fusion_failures_by_pair", f"{pair_key}|{reason}")
            if not rejected_reason or reason != rejected_reason:
                reasons[reason] += 1
            diag["fusion_last_rejected_reason"] = reason
        diag["fusion_last_result"] = {
            "success": success,
            "illegalReason": result.get("illegalReason"),
            "fusionRejectedReason": result.get("fusionRejectedReason"),
            "fusionExecutionMode": str(result.get("fusionExecutionMode") or ""),
            "bridgeMethodUsed": str(result.get("bridgeMethodUsed") or ""),
            "bridgeResultReason": str(result.get("bridgeResultReason") or ""),
            "duplicateStackDetected": bool(result.get("duplicateStackDetected")),
            "fusionScope": str(result.get("fusionScope") or result.get("fusion_scope") or ""),
            "changedTileCount": _safe_int(result.get("changedTileCount"), result.get("changed_tile_count"), default=0),
            "nonSourceTilesChanged": bool(
                result.get("nonSourceTilesChanged")
                if "nonSourceTilesChanged" in result
                else result.get("non_source_tiles_changed")
            ),
            "globalFusionSideEffect": bool(
                result.get("globalFusionSideEffect")
                if "globalFusionSideEffect" in result
                else result.get("global_fusion_side_effect")
            ),
            "plantCountOnTileBefore": _safe_int(result.get("plantCountOnTileBefore"), default=0),
            "plantCountOnTileAfter": _safe_int(result.get("plantCountOnTileAfter"), default=0),
            "sourceTileOccupiedBefore": bool(result.get("sourceTileOccupiedBefore")),
            "sourcePlantBefore": result.get("sourcePlantBefore"),
            "resultingPlantAfter": result.get("resultingPlantAfter"),
        }
        diag["fusion_last_execution_mode"] = str(result.get("fusionExecutionMode") or "")
        diag["fusion_last_bridge_method_used"] = str(result.get("bridgeMethodUsed") or "")
        diag["fusion_last_bridge_result_reason"] = str(result.get("bridgeResultReason") or "")
        diag["fusion_last_duplicate_stack_detected"] = bool(result.get("duplicateStackDetected"))
        diag["last_fusion_scope"] = str(result.get("fusionScope") or result.get("fusion_scope") or "")
        diag["last_fusion_changed_tile_count"] = _safe_int(result.get("changedTileCount"), result.get("changed_tile_count"), default=0)
        diag["last_fusion_non_source_tiles_changed"] = bool(
            result.get("nonSourceTilesChanged")
            if "nonSourceTilesChanged" in result
            else result.get("non_source_tiles_changed")
        )
        diag["last_fusion_global_side_effect"] = bool(
            result.get("globalFusionSideEffect")
            if "globalFusionSideEffect" in result
            else result.get("global_fusion_side_effect")
        )
        diag["last_executed_coach_command_id"] = result.get("last_executed_coach_command_id", result.get("executed_coach_command_id"))
    diag["fusion_rejected_reasons"] = dict(sorted(reasons.items()))
    return diag


def fusion_execution_from_result(
    decision: FusionDecision,
    result: Optional[Mapping[str, Any]],
    *,
    event_id: str,
    attempted: Optional[bool] = None,
    rejection_reason: str = "",
) -> FusionExecution:
    """Normalize a bridge/pre-execution result into one accounting event."""

    payload = _deep_thaw(result) if isinstance(result, Mapping) else {}
    if not isinstance(payload, dict):
        payload = {}
    success = bool(
        payload.get("fusionSucceeded")
        if "fusionSucceeded" in payload
        else payload.get("fusion_success", False)
    )
    if attempted is None:
        attempted = bool(
            payload.get("fusionAttempted")
            if "fusionAttempted" in payload
            else bool(payload)
        )
    reason = "" if success else str(
        rejection_reason
        or payload.get("fusionRejectedReason")
        or payload.get("illegalReason")
        or payload.get("bridgeResultReason")
        or decision.rejection_reason
        or ("bridge_rejected" if attempted else "")
    )
    return FusionExecution(
        decision=decision,
        event_id=str(event_id),
        attempted=bool(attempted),
        success=bool(success),
        rejection_reason=reason,
        result=_deep_freeze(payload),
    )


def account_fusion_execution_once(
    diagnostics: Dict[str, Any],
    execution: FusionExecution,
    accounted_event_ids: MutableSet[str],
) -> Tuple[Dict[str, Any], bool]:
    """Apply attempt/success/failure counters at most once for ``event_id``.

    Returns ``(diagnostics, applied)``.  A validation rejection is observable
    but is not miscounted as a bridge attempt.  A reached bridge call is exactly
    one attempt and then exactly one success or failure.
    """

    event_id = str(execution.event_id or "").strip()
    if not event_id:
        raise ValueError("fusion accounting requires a non-empty event_id")
    if event_id in accounted_event_ids:
        return dict(diagnostics), False
    accounted_event_ids.add(event_id)

    result = _deep_thaw(execution.result) if execution.attempted else None
    rejection = "" if execution.success else str(execution.rejection_reason or "")
    updated = apply_fusion_attempt_result(
        diagnostics,
        execution.decision.intent.candidate_dict(),
        result,
        rejected_reason=rejection,
        bridge_error=(
            str(execution.result.get("bridgeError") or "")
            if execution.attempted and str(execution.result.get("bridgeError") or "")
            else None
        ),
    )
    updated["fusion_last_event_id"] = event_id
    updated["fusion_last_source"] = execution.decision.intent.source
    updated["fusion_last_attempted"] = bool(execution.attempted)
    updated["fusion_last_success"] = bool(execution.success)
    updated["fusion_accounting_applied"] = True
    return updated, True


def enforce_fusion_scope_contract(
    result: Dict[str, Any],
    *,
    require_dedicated_execution: bool = False,
    require_occupied_source: bool = False,
) -> str:
    """Reject non-tile-scoped or structurally invalid bridge fusion results.

    The mapping is updated in place for legacy callers and the rejection reason
    is returned.  Environment execution uses the tile-scope checks; the coach
    compatibility wrapper additionally requests the historical dedicated-path
    and occupied-source postconditions.
    """

    if not isinstance(result, dict):
        return ""
    placement = result.get("placement") if isinstance(result.get("placement"), dict) else {}

    def value(*keys: str, default: Any = None) -> Any:
        for payload in (result, placement):
            for key in keys:
                if key in payload and payload.get(key) is not None:
                    return payload.get(key)
        return default

    success = bool(value("fusionSucceeded", "fusion_success", "fusionOverrideApplied", default=False))
    changed_count = _safe_int(value("changedTileCount", "changed_tile_count"), default=0)
    changed_tiles = value("changedTiles", "changed_tiles", default=[])
    changed_tiles = changed_tiles if isinstance(changed_tiles, list) else []
    requested_row = _safe_int(value("requestedSourceRow", "requested_source_row"), default=-1)
    requested_col = _safe_int(value("requestedSourceCol", "requested_source_col"), default=-1)
    non_source_changed = bool(value("nonSourceTilesChanged", "non_source_tiles_changed", default=False))
    global_side_effect = bool(value("globalFusionSideEffect", "global_fusion_side_effect", default=False))
    source_tile_changed = False
    for tile in changed_tiles:
        if not isinstance(tile, dict):
            continue
        row = _safe_int(tile.get("row"), tile.get("sourceRow"), tile.get("source_row"), default=-1)
        col = _safe_int(tile.get("column"), tile.get("col"), tile.get("sourceCol"), tile.get("source_col"), default=-1)
        if row == requested_row and col == requested_col:
            source_tile_changed = True
        else:
            non_source_changed = True

    failure = ""
    bridge_reason = str(value("bridgeResultReason", "bridge_result_reason", default="") or "")
    if success and require_occupied_source and not bool(value("sourceTileOccupiedBefore", "source_tile_occupied_before")):
        failure, bridge_reason = "source_tile_not_occupied", bridge_reason or "source_tile_not_occupied"
    elif success and require_dedicated_execution and str(value("fusionExecutionMode", "fusion_execution_mode") or "") != "dedicated_fusion":
        failure, bridge_reason = "bridge_rejected", bridge_reason or "fusion_not_dedicated_path"
    elif success and bool(value("duplicateStackDetected", "duplicate_stack_detected", default=False)):
        failure, bridge_reason = "duplicate_stack_detected", bridge_reason or "duplicate_stack_detected"
    elif non_source_changed or global_side_effect or changed_count > 1:
        failure, bridge_reason = "global_fusion_side_effect", "fusion_mutated_non_source_tiles"
    elif success and (changed_count != 1 or not source_tile_changed):
        failure, bridge_reason = "fusion_no_effect", bridge_reason or "fusion_no_effect"
    if not failure:
        return ""

    updates = {
        "fusionSucceeded": False,
        "fusion_success": False,
        "fusionOverrideApplied": False,
        "coachFusionOverrideApplied": False,
        "illegalAction": True,
        "illegalReason": failure,
        "fusionRejectedReason": failure,
        "bridgeResultReason": bridge_reason,
        "bridge_result_reason": bridge_reason,
        "changedTileCount": changed_count,
        "changed_tile_count": changed_count,
        "changedTiles": changed_tiles,
        "changed_tiles": changed_tiles,
        "nonSourceTilesChanged": bool(non_source_changed),
        "non_source_tiles_changed": bool(non_source_changed),
        "globalFusionSideEffect": bool(global_side_effect or failure == "global_fusion_side_effect"),
        "global_fusion_side_effect": bool(global_side_effect or failure == "global_fusion_side_effect"),
    }
    if failure == "global_fusion_side_effect":
        updates.update({"fusionScope": "global_side_effect_detected", "fusion_scope": "global_side_effect_detected"})
    result.update(updates)
    if placement:
        placement.update(updates)
        placement["success"] = False
    return failure


def merge_episode_fusion_stats(target: Any, diagnostics: Dict[str, Any]) -> None:
    if not isinstance(diagnostics, dict):
        return
    for name in (
        "fusion_candidate_count",
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
        "recursive_fusion_count",
        "high_tier_fusion_count",
    ):
        value = int(diagnostics.get(name) or 0)
        if name == "fusion_candidate_count":
            total_name = "fusion_candidate_count_total"
            setattr(target, total_name, int(getattr(target, total_name, 0)) + value)
        else:
            setattr(target, name, int(getattr(target, name, 0)) + value)
    for name in (
        "fusion_rejected_reasons",
        "fusion_by_result_type",
        "fusion_by_source_type",
        "fusion_by_row",
        "fusion_attempts_by_pair",
        "fusion_successes_by_pair",
        "fusion_failures_by_pair",
        "fusion_result_counts",
        "fusion_depth_counts",
    ):
        current = getattr(target, name, {})
        if not isinstance(current, dict):
            current = {}
        for key, value in (diagnostics.get(name) or {}).items():
            current[str(key)] = int(current.get(str(key), 0)) + int(value)
        setattr(target, name, dict(sorted(current.items())))
    setattr(
        target,
        "highest_fusion_tier",
        max(int(getattr(target, "highest_fusion_tier", 0)), int(diagnostics.get("highest_fusion_tier") or 0)),
    )
    # Fusion reward component totals are cumulative per-episode values supplied by
    # the env each step, so take the latest (set, do not sum). fusion_reward_total
    # itself is owned by accumulate_reward_episode_totals via the reward breakdown.
    for name in (
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
    ):
        if name in diagnostics:
            try:
                setattr(target, name, float(diagnostics.get(name) or 0.0))
            except (TypeError, ValueError):
                pass
    if diagnostics.get("fusion_reward_capped"):
        setattr(target, "fusion_reward_capped", True)
    for name, default in (
        ("fusion_last_reward_delta", 0.0),
        ("fusion_last_usefulness_bonus", 0.0),
    ):
        if name in diagnostics:
            try:
                setattr(target, name, float(diagnostics.get(name) or default))
            except (TypeError, ValueError):
                pass
    for name in ("fusion_last_reward_reason", "fusion_last_source"):
        if name in diagnostics:
            setattr(target, name, str(diagnostics.get(name) or ""))
    setattr(target, "fusion_policy", str(diagnostics.get("fusion_policy") or getattr(target, "fusion_policy", "none")))


def fusion_live_fields(diagnostics: Optional[Dict[str, Any]], policy: str = FUSION_POLICY_NONE) -> Dict[str, Any]:
    diag = diagnostics if isinstance(diagnostics, dict) else default_fusion_diagnostics(policy)
    return {
        "fusion_policy": diag.get("fusion_policy", policy),
        "fusion_available": bool(diag.get("fusion_available")),
        "fusion_candidate_count": int(diag.get("fusion_candidate_count") or 0),
        "fusion_top_candidate": diag.get("fusion_top_candidate"),
        "fusion_actions_available_count": int(diag.get("fusion_actions_available_count") or 0),
        "fusion_candidate_tiles": list(diag.get("fusion_candidate_tiles") or []),
        "fusion_last_illegal_reason": str(
            diag.get("fusion_last_illegal_reason") or diag.get("fusion_last_rejected_reason") or ""
        ),
        "fusion_last_incompatible_pair": diag.get("fusion_last_incompatible_pair"),
        "fusion_actions_masked_empty_tile": int(diag.get("fusion_actions_masked_empty_tile") or 0),
        "fusion_actions_masked_incompatible_count": int(diag.get("fusion_actions_masked_incompatible_count") or 0),
        "fusion_actions_masked_disabled_count": int(diag.get("fusion_actions_masked_disabled_count") or 0),
        "fusion_actions_masked_cooldown_count": int(diag.get("fusion_actions_masked_cooldown_count") or 0),
        "fusion_actions_masked_sun_count": int(diag.get("fusion_actions_masked_sun_count") or 0),
        "fusion_compatibility_table": diag.get("fusion_compatibility_table") or fusion_compatibility_table(),
        "fusion_reward_total": float(diag.get("fusion_reward_total") or 0.0),
        "fusion_attempt_reward_total": float(diag.get("fusion_attempt_reward_total") or 0.0),
        "fusion_success_reward_total": float(diag.get("fusion_success_reward_total") or 0.0),
        "fusion_new_recipe_reward_total": float(diag.get("fusion_new_recipe_reward_total") or 0.0),
        "fusion_recursive_reward_total": float(diag.get("fusion_recursive_reward_total") or 0.0),
        "fusion_tier_reward_total": float(diag.get("fusion_tier_reward_total") or 0.0),
        "fusion_repeat_decay_total": float(diag.get("fusion_repeat_decay_total") or 0.0),
        "fusion_threatened_row_bonus_total": float(diag.get("fusion_threatened_row_bonus_total") or 0.0),
        "fusion_active_wave_bonus_total": float(diag.get("fusion_active_wave_bonus_total") or 0.0),
        "fusion_defensive_value_bonus_total": float(diag.get("fusion_defensive_value_bonus_total") or 0.0),
        "fusion_incompatible_penalty_total": float(diag.get("fusion_incompatible_penalty_total") or 0.0),
        "fusion_empty_tile_penalty_total": float(diag.get("fusion_empty_tile_penalty_total") or 0.0),
        "fusion_failed_penalty_total": float(diag.get("fusion_failed_penalty_total") or 0.0),
        "fusion_bridge_error_penalty_total": float(diag.get("fusion_bridge_error_penalty_total") or 0.0),
        "fusion_spam_penalty_total": float(diag.get("fusion_spam_penalty_total") or 0.0),
        "fusion_reward_capped": bool(diag.get("fusion_reward_capped")),
        "fusion_last_reward_delta": float(diag.get("fusion_last_reward_delta") or 0.0),
        "fusion_last_reward_reason": str(diag.get("fusion_last_reward_reason") or ""),
        "fusion_last_usefulness_bonus": float(diag.get("fusion_last_usefulness_bonus") or 0.0),
        "fusion_last_source": str(diag.get("fusion_last_source") or ""),
        "fusion_last_attempt": diag.get("fusion_last_attempt"),
        "fusion_last_result": diag.get("fusion_last_result"),
        "fusion_last_execution_mode": str(diag.get("fusion_last_execution_mode") or ""),
        "fusion_last_bridge_method_used": str(diag.get("fusion_last_bridge_method_used") or ""),
        "fusion_last_bridge_result_reason": str(diag.get("fusion_last_bridge_result_reason") or ""),
        "fusion_last_duplicate_stack_detected": bool(diag.get("fusion_last_duplicate_stack_detected")),
        "last_fusion_scope": str(diag.get("last_fusion_scope") or ""),
        "last_fusion_changed_tile_count": int(diag.get("last_fusion_changed_tile_count") or 0),
        "last_fusion_non_source_tiles_changed": bool(diag.get("last_fusion_non_source_tiles_changed")),
        "last_fusion_global_side_effect": bool(diag.get("last_fusion_global_side_effect")),
        "last_executed_coach_command_id": diag.get("last_executed_coach_command_id"),
        "fusion_last_rejected_reason": str(diag.get("fusion_last_rejected_reason") or ""),
        "fusion_attempted_count": int(diag.get("fusion_attempted_count") or 0),
        "fusion_success_count": int(diag.get("fusion_success_count") or 0),
        "fusion_failed_count": int(diag.get("fusion_failed_count") or 0),
        "fusion_rejected_count": int(diag.get("fusion_rejected_count") or 0),
        "fusion_rejected_reasons": dict(diag.get("fusion_rejected_reasons") or {}),
        "fusion_attempts_by_pair": dict(diag.get("fusion_attempts_by_pair") or {}),
        "fusion_successes_by_pair": dict(diag.get("fusion_successes_by_pair") or {}),
        "fusion_failures_by_pair": dict(diag.get("fusion_failures_by_pair") or {}),
        "fusion_result_counts": dict(diag.get("fusion_result_counts") or {}),
        "fusion_depth_counts": dict(diag.get("fusion_depth_counts") or {}),
        "recursive_fusion_count": int(diag.get("recursive_fusion_count") or 0),
        "high_tier_fusion_count": int(diag.get("high_tier_fusion_count") or 0),
        "highest_fusion_tier": int(diag.get("highest_fusion_tier") or 0),
    }


def _probe_candidate_index(bridge_probe: Optional[Dict[str, Any]]) -> Dict[Tuple[int, int, int, int, int], Dict[str, Any]]:
    if not isinstance(bridge_probe, dict):
        return {}
    raw_candidates = bridge_probe.get("fusionCandidates") or bridge_probe.get("fusion_candidates") or []
    if not isinstance(raw_candidates, list):
        return {}
    index: Dict[Tuple[int, int, int, int, int], Dict[str, Any]] = {}
    for candidate in raw_candidates:
        if not isinstance(candidate, dict):
            continue
        row = _safe_int(candidate.get("sourceRow"), candidate.get("source_row"), default=-1)
        col = _safe_int(candidate.get("sourceCol"), candidate.get("sourceColumn"), candidate.get("source_col"), default=-1)
        source_type = _safe_int(candidate.get("sourcePlantType"), candidate.get("source_plant_type"), default=-1)
        slot_index = _safe_int(candidate.get("ingredientSeedSlotIndex"), candidate.get("seedSlotIndex"), candidate.get("ingredient_seed_slot_index"), default=-1)
        ingredient_type = _safe_int(candidate.get("ingredientPlantType"), candidate.get("targetPlantType"), candidate.get("ingredient_plant_type"), default=-1)
        index[(row, col, source_type, slot_index, ingredient_type)] = candidate
    return index


def _match_probe_candidate(
    index: Dict[Tuple[int, int, int, int, int], Dict[str, Any]],
    source_instance: int,
    row: int,
    col: int,
    source_type: int,
    slot_index: int,
    ingredient_type: int,
) -> Optional[Dict[str, Any]]:
    candidate = index.get((row, col, source_type, slot_index, ingredient_type))
    if candidate is not None:
        return candidate
    for probe in index.values():
        if source_instance <= 0:
            continue
        probe_instance = _safe_int(probe.get("sourceInstanceId"), probe.get("source_instance_id"), default=0)
        if probe_instance == source_instance and _safe_int(probe.get("ingredientSeedSlotIndex"), probe.get("seedSlotIndex"), default=-2) == slot_index:
            return probe
    return None


def _is_scripted_allowlisted(candidate: Dict[str, Any]) -> bool:
    key = (
        _safe_int(candidate.get("source_plant_type"), default=-1),
        _safe_int(candidate.get("target_or_ingredient_type"), default=-1),
    )
    return bool(FUSION_RULES.get(key, {}).get("scripted_enabled"))


def _lane_context(
    observation: Dict[str, Any],
    source_row: int,
    source_col: int,
    *,
    facts: Optional[StepFacts] = None,
) -> Dict[str, Any]:
    lane = {}
    for item in observation.get("lanes", []) or []:
        if isinstance(item, dict) and _safe_int(item.get("row"), default=-1) == source_row:
            lane = item
            break
    lane_danger = _safe_float(lane.get("toughZombiePressureScore"), lane.get("danger"), default=-1.0)
    nearest_x = _safe_float(lane.get("nearestZombieX"), default=99.0)
    if lane_danger < 0.0:
        lane_danger = max(0.0, 1.0 - nearest_x / 10.0) if nearest_x < 99.0 else 0.0
    snapshot = facts or build_step_facts(observation)
    zombies = snapshot.zombies_by_lane.get(source_row, ())
    nearby = []
    for zombie in zombies:
        zx = zombie.x if zombie.has_position else nearest_x
        if zx >= float(source_col) - 0.5 and zx - float(source_col) <= 5.0:
            nearby.append(zombie)
    if not nearby and zombies:
        nearby = zombies
    coneheads = sum(1 for zombie in nearby if _is_conehead_fact(zombie))
    bucketheads = sum(1 for zombie in nearby if _is_buckethead_fact(zombie))
    return {
        "lane_danger_score": round(float(lane_danger), 4),
        "nearby_zombie_count": len(nearby),
        "nearby_conehead_count": coneheads,
        "nearby_buckethead_count": bucketheads,
        "nearest_zombie_distance": round(max(0.0, nearest_x - float(source_col)), 4) if nearest_x < 99.0 else None,
    }


def _strategic_score(
    source_type: int,
    ingredient_type: int,
    rule: Optional[Dict[str, Any]],
    lane: Dict[str, Any],
    health_ratio: float,
) -> Tuple[float, str]:
    del source_type, ingredient_type
    score = float(lane.get("lane_danger_score") or 0.0)
    score += min(1.0, 0.25 * int(lane.get("nearby_zombie_count") or 0))
    score += 0.55 * int(lane.get("nearby_conehead_count") or 0)
    score += 0.8 * int(lane.get("nearby_buckethead_count") or 0)
    distance = lane.get("nearest_zombie_distance")
    if isinstance(distance, (int, float)) and float(distance) <= 4.0:
        score += 0.35
    if health_ratio < 0.5:
        score += 0.15
    if not rule:
        return round(score, 4), "mapping unknown; observe only"
    role = str(rule.get("role") or "")
    if role == "dps":
        score += 0.35
    elif role == "economy":
        score -= 0.5
    return round(max(0.0, score), 4), str(rule.get("reason") or "known fusion mapping")


def is_conehead_zombie(zombie: Dict[str, Any]) -> bool:
    zombie_type = _safe_int(zombie.get("type"), default=-1)
    name = str(zombie.get("typeName") or "").lower()
    return zombie_type in CONEHEAD_TYPES or "cone" in name or "roadblock" in name or "路障" in name


def is_buckethead_zombie(zombie: Dict[str, Any]) -> bool:
    zombie_type = _safe_int(zombie.get("type"), default=-1)
    name = str(zombie.get("typeName") or "").lower()
    return zombie_type in BUCKETHEAD_TYPES or "bucket" in name or "铁桶" in name


def is_tough_zombie(zombie: Dict[str, Any]) -> bool:
    return (
        is_conehead_zombie(zombie)
        or is_buckethead_zombie(zombie)
        or _safe_int(zombie.get("health"), default=0) >= 600
        or _safe_int(zombie.get("maxHealth"), default=0) >= 600
    )


def _is_conehead_fact(zombie: Any) -> bool:
    name = str(zombie.type_name or "").lower()
    return zombie.zombie_type in CONEHEAD_TYPES or "cone" in name or "roadblock" in name or "路障" in name


def _is_buckethead_fact(zombie: Any) -> bool:
    name = str(zombie.type_name or "").lower()
    return zombie.zombie_type in BUCKETHEAD_TYPES or "bucket" in name or "铁桶" in name


def _is_tough_fact(zombie: Any) -> bool:
    return (
        _is_conehead_fact(zombie)
        or _is_buckethead_fact(zombie)
        or zombie.health >= 600
        or zombie.max_health >= 600
    )


def count_tough_zombies_by_row(
    observation: Dict[str, Any],
    *,
    facts: Optional[StepFacts] = None,
) -> Dict[int, Dict[str, int]]:
    snapshot = facts or build_step_facts(observation)
    rows = _safe_int(observation.get("rowCount"), default=5)
    counts = {
        row: {"buckethead": 0, "conehead": 0, "tough": 0}
        for row in range(max(0, rows))
    }
    for lane in snapshot.lanes:
        row = lane.row
        if row not in counts:
            continue
        counts[row]["buckethead"] = lane.buckethead_count
        counts[row]["conehead"] = lane.conehead_count
        counts[row]["tough"] = lane.tough_zombie_count
    for zombie in snapshot.alive_zombies:
        row = zombie.row
        if row not in counts:
            continue
        if _is_buckethead_fact(zombie):
            counts[row]["buckethead"] += 1
        if _is_conehead_fact(zombie):
            counts[row]["conehead"] += 1
        if _is_tough_fact(zombie):
            counts[row]["tough"] += 1
    return counts


def _bump_dict(payload: Dict[str, Any], field: str, key: str) -> None:
    values = dict(payload.get(field) or {})
    values[str(key)] = int(values.get(str(key), 0)) + 1
    payload[field] = dict(sorted(values.items()))


def _safe_int(*values: Any, default: int = 0) -> int:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return int(default)


def _safe_float(*values: Any, default: float = 0.0) -> float:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return float(default)
