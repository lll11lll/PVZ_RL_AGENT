"""Seed inventory diagnostics and Adventure Generalist identity features."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from pvzrl_registry import get_plant_registry


ADVENTURE_IDENTITY_PLANT_TYPE_BUCKETS = 256
ADVENTURE_IDENTITY_UNKNOWN_BUCKETS = 1
ADVENTURE_IDENTITY_ONE_HOT_WIDTH = ADVENTURE_IDENTITY_PLANT_TYPE_BUCKETS + ADVENTURE_IDENTITY_UNKNOWN_BUCKETS
ADVENTURE_IDENTITY_ROLE_NAMES = ("unknown", "economy", "attacker", "blocker", "explosive", "support", "utility")
ADVENTURE_IDENTITY_SCALAR_FEATURES_PER_SLOT = 13
ADVENTURE_IDENTITY_FEATURES_PER_SLOT = (
    ADVENTURE_IDENTITY_SCALAR_FEATURES_PER_SLOT
    + ADVENTURE_IDENTITY_ONE_HOT_WIDTH
    + len(ADVENTURE_IDENTITY_ROLE_NAMES)
)
ADVENTURE_IDENTITY_SUMMARY_FEATURES = 12


def canonical_seed(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"none", "unknown", "-1"}:
        return ""
    return text


def ordered_unique(values: Iterable[Any]) -> List[str]:
    output: List[str] = []
    seen = set()
    for value in values:
        seed = canonical_seed(value)
        if not seed or seed in seen:
            continue
        output.append(seed)
        seen.add(seed)
    return output


def missing_seeds(required: Iterable[Any], available: Iterable[Any]) -> List[str]:
    available_set = set(ordered_unique(available))
    return [seed for seed in ordered_unique(required) if seed not in available_set]


def inventory_payload(
    *,
    selected: Iterable[Any],
    unlocked: Iterable[Any],
    available: Iterable[Any],
    required_unlocked: Optional[Iterable[Any]] = None,
    required_available: Optional[Iterable[Any]] = None,
    max_seed_slots: int = 0,
    legal_action_count: int = 0,
    mask_block_reason_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    selected_list = ordered_unique(selected)
    unlocked_list = ordered_unique(unlocked)
    available_list = ordered_unique(available)
    required_unlocked_list = ordered_unique(required_unlocked or [])
    required_available_list = ordered_unique(required_available or [])
    missing_unlocked = missing_seeds(required_unlocked_list, unlocked_list)
    missing_available = missing_seeds(required_available_list, available_list)
    slot_denominator = max(1, int(max_seed_slots or len(selected_list) or len(available_list) or 1))
    available_required_total = max(1, len(required_available_list))
    unlocked_required_total = max(1, len(required_unlocked_list))
    return {
        "selected_seeds": selected_list,
        "unlocked_seeds": unlocked_list,
        "available_seeds": available_list,
        "required_unlocked": required_unlocked_list,
        "required_available": required_available_list,
        "missing_required_unlocked": missing_unlocked,
        "missing_required_available": missing_available,
        "selected_count": len(selected_list),
        "unlocked_count": len(unlocked_list),
        "available_count": len(available_list),
        "max_seed_slots": int(max_seed_slots or 0),
        "selected_slot_ratio": len(selected_list) / float(slot_denominator),
        "available_slot_ratio": len(available_list) / float(slot_denominator),
        "unlocked_slot_ratio": len(unlocked_list) / float(slot_denominator),
        "required_unlocked_ratio": (
            (len(required_unlocked_list) - len(missing_unlocked)) / float(unlocked_required_total)
            if required_unlocked_list
            else 1.0
        ),
        "required_available_ratio": (
            (len(required_available_list) - len(missing_available)) / float(available_required_total)
            if required_available_list
            else 1.0
        ),
        "legal_action_count": int(legal_action_count or 0),
        "mask_block_reason_counts": dict(sorted((mask_block_reason_counts or {}).items())),
    }


def inventory_from_runtime_sources(
    *,
    observation: Dict[str, Any],
    adventure_state: Dict[str, Any],
    context: Dict[str, Any],
    max_seed_slots: int,
    legal_action_count: int = 0,
    mask_block_reason_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    slots = observation.get("seedSlots", []) if isinstance(observation.get("seedSlots", []), list) else []
    selected_from_slots = [
        slot.get("plantTypeName") or slot.get("displayName") or slot.get("plantType")
        for slot in slots
        if isinstance(slot, dict)
    ]
    selected = (
        context.get("selected_seeds", [])
        or adventure_state.get("selectedSeedNames", [])
        or observation.get("selectedSeedNames", [])
        or selected_from_slots
    )
    unlocked = context.get("unlocked_seeds", []) or adventure_state.get("unlockedSeedNames", [])
    available = (
        adventure_state.get("availableSeedNames", [])
        or adventure_state.get("visibleSeedCardNames", [])
        or observation.get("availableSeedNames", [])
    )
    return inventory_payload(
        selected=selected,
        unlocked=unlocked,
        available=available,
        required_unlocked=context.get("required_unlocked_seeds", []),
        required_available=context.get("required_available_seeds", []),
        max_seed_slots=max_seed_slots,
        legal_action_count=legal_action_count,
        mask_block_reason_counts=mask_block_reason_counts,
    )


def adventure_identity_feature_count(max_seed_slots: int) -> int:
    return int(max_seed_slots) * ADVENTURE_IDENTITY_FEATURES_PER_SLOT + ADVENTURE_IDENTITY_SUMMARY_FEATURES


def adventure_identity_features(observation: Dict[str, Any], max_seed_slots: int) -> List[float]:
    slots = observation.get("seedSlots", []) if isinstance(observation.get("seedSlots", []), list) else []
    sun = _safe_float(observation.get("sun"), 0.0)
    unlocked_names = set(ordered_unique(observation.get("unlockedSeedNames", []) or []))
    unlocked_types = _safe_int_set(observation.get("unlockedSeedPlantTypes", []) or observation.get("visibleSeedPlantTypes", []) or [])
    selected_types: List[int] = []
    for slot in slots:
        if isinstance(slot, dict):
            selected_types.append(_safe_int(slot.get("plantType"), -1))
    duplicate_counts: Dict[int, int] = {}
    for plant_type in selected_types:
        if plant_type >= 0:
            duplicate_counts[plant_type] = duplicate_counts.get(plant_type, 0) + 1
    duplicate_seen: Dict[int, int] = {}

    features: List[float] = []
    active_count = 0
    ready_count = 0
    affordable_count = 0
    unlocked_selected_count = 0
    registry = get_plant_registry()
    for slot_index in range(max(0, int(max_seed_slots))):
        slot = slots[slot_index] if slot_index < len(slots) and isinstance(slots[slot_index], dict) else {}
        present = 1.0 if slot else 0.0
        plant_type = _safe_int(slot.get("plantType"), -1) if slot else -1
        plant_name = str(slot.get("plantTypeName") or slot.get("displayName") or "").strip() if slot else ""
        usable = 1.0 if slot.get("usable") else 0.0
        ready = 1.0 if slot.get("ready") else 0.0
        disabled = 1.0 if slot.get("disabled") else 0.0
        cost = max(0.0, _safe_float(slot.get("seedCost"), 0.0))
        affordable = 1.0 if slot and sun >= cost else 0.0
        cooldown = _safe_float(slot.get("currentCooldown"), 0.0)
        raw_cooldown = _safe_float(slot.get("rawCooldown"), cooldown)
        full = max(1e-6, _safe_float(slot.get("fullCooldown"), 0.0))
        duplicate_count = duplicate_counts.get(plant_type, 0) if plant_type >= 0 else 0
        duplicate_index = duplicate_seen.get(plant_type, 0) if plant_type >= 0 else 0
        if plant_type >= 0:
            duplicate_seen[plant_type] = duplicate_index + 1
        unlocked = bool(
            slot
            and (
                plant_name in unlocked_names
                or plant_type in unlocked_types
                or plant_name
                or plant_type >= 0
            )
        )
        if present:
            active_count += 1
        if ready:
            ready_count += 1
        if affordable:
            affordable_count += 1
        if unlocked:
            unlocked_selected_count += 1

        # ``plant_type`` is normalized once above; use the immutable index
        # directly inside this observation hot loop to avoid repeated coercion.
        definition = registry.by_id.get(plant_type)
        role = str(definition.role if definition is not None else "unknown").strip().lower()
        if role not in ADVENTURE_IDENTITY_ROLE_NAMES:
            role = "unknown"
        features.extend(
            [
                present,
                present,  # selected/populated; separate from future active-capacity if bridge exposes it.
                1.0 if unlocked else 0.0,
                usable,
                ready,
                affordable,
                disabled,
                _clip(cost / 500.0),
                _clip(cooldown / full),
                _clip(raw_cooldown / full),
                _clip((plant_type if plant_type >= 0 else 0) / 255.0),
                _clip(duplicate_count / 14.0),
                _clip(duplicate_index / 14.0),
            ]
        )
        features.extend(_plant_identity_one_hot(plant_type))
        features.extend(1.0 if name == role else 0.0 for name in ADVENTURE_IDENTITY_ROLE_NAMES)

    denom = max(1.0, float(max_seed_slots or 1))
    legal_count = len(observation.get("legalActions", []) or [])
    features.extend(
        [
            _clip(active_count / denom),
            _clip(ready_count / denom),
            _clip(affordable_count / denom),
            _clip(unlocked_selected_count / denom),
            _clip(len(duplicate_counts) / denom),
            _clip(legal_count / max(1.0, denom * 50.0 + 1.0)),
            1.0 if observation.get("seedSelectionActive") else 0.0,
            1.0 if observation.get("gameplayReady") else 0.0,
            _clip(sun / 1000.0),
            _clip(_safe_float(observation.get("wave"), 0.0) / max(1.0, _safe_float(observation.get("maxWave"), 1.0))),
            _clip(_safe_float(observation.get("plantCount"), 0.0) / 50.0),
            _clip(_safe_float(observation.get("zombieCount"), 0.0) / 50.0),
        ]
    )
    return features


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clip(value: float) -> float:
    return max(-10.0, min(10.0, float(value)))


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _safe_int_set(values: Any) -> set[int]:
    if not isinstance(values, list):
        return set()
    output: set[int] = set()
    for value in values:
        parsed = _safe_int(value, -1)
        if parsed >= 0:
            output.add(parsed)
    return output


def _plant_identity_one_hot(plant_type: int) -> List[float]:
    values = [0.0] * ADVENTURE_IDENTITY_ONE_HOT_WIDTH
    if 0 <= int(plant_type) < ADVENTURE_IDENTITY_PLANT_TYPE_BUCKETS:
        values[int(plant_type)] = 1.0
    else:
        values[-1] = 1.0
    return values
