"""Shared PvZRL action-space definitions.

The fixed legacy decoder, experimental dynamic decoder, and permanent
Adventure identity decoder intentionally keep their wait/action layouts
explicit:

* fixed: action 0 waits, placements are 1..N
* dynamic_14: placements are 0..699, action 700 waits
* adventure_14slot_identity: action 0 waits, placements are 1..700

Keeping those decoder versions explicit prevents 201-action fixed policies and
701-action policies with different wait positions or observation schemas from
being loaded into the wrong environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


ACTION_SPACE_FIXED = "fixed"
ACTION_SPACE_DYNAMIC_14 = "dynamic_14"
ACTION_SPACE_ADVENTURE_14_IDENTITY = "adventure_14slot_identity"
ACTION_SPACE_MODES = {ACTION_SPACE_FIXED, ACTION_SPACE_DYNAMIC_14, ACTION_SPACE_ADVENTURE_14_IDENTITY}

DEFAULT_ROWS = 5
DEFAULT_COLS = 10
CELLS_PER_SLOT = DEFAULT_ROWS * DEFAULT_COLS
DYNAMIC_MAX_SEED_SLOTS = 14

FIXED_OBSERVATION_VERSION = "fixed_slot_v1"
DYNAMIC_OBSERVATION_VERSION = "seed_inventory_v2"
ADVENTURE_IDENTITY_OBSERVATION_VERSION = "adventure_14slot_identity_v1"
FIXED_ACTION_DECODER_VERSION = "fixed_slot_4x50_plus_wait_v1"
DYNAMIC_ACTION_DECODER_VERSION = "max_seed_slots_14_v1"
ADVENTURE_IDENTITY_ACTION_DECODER_VERSION = "seedslot14x50_plus_wait_v1"

DYNAMIC_WAIT_ACTION = DYNAMIC_MAX_SEED_SLOTS * CELLS_PER_SLOT
DYNAMIC_ACTION_COUNT = DYNAMIC_WAIT_ACTION + 1
ADVENTURE_IDENTITY_WAIT_ACTION = 0
ADVENTURE_IDENTITY_ACTION_COUNT = DYNAMIC_MAX_SEED_SLOTS * CELLS_PER_SLOT + 1
FIXED_WAIT_ACTION = 0


@dataclass(frozen=True)
class ActionSpaceSpec:
    mode: str
    action_count: int
    max_seed_slots: int
    observation_version: str
    action_decoder_version: str
    wait_action: int
    placement_action_min: int
    placement_action_max: int
    rows: int = DEFAULT_ROWS
    cols: int = DEFAULT_COLS

    @property
    def dynamic_seed_slots(self) -> bool:
        return self.mode in {ACTION_SPACE_DYNAMIC_14, ACTION_SPACE_ADVENTURE_14_IDENTITY}

    @property
    def identity_seed_slots(self) -> bool:
        return self.mode == ACTION_SPACE_ADVENTURE_14_IDENTITY

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "action_space_mode": self.mode,
            "action_count": self.action_count,
            "max_seed_slots": self.max_seed_slots,
            "dynamic_seed_slots": self.dynamic_seed_slots,
            "identity_seed_slots": self.identity_seed_slots,
            "observation_version": self.observation_version,
            "action_decoder_version": self.action_decoder_version,
            "decoder_wait_action": self.wait_action,
            "placement_action_range": [self.placement_action_min, self.placement_action_max],
            "rows": self.rows,
            "cols": self.cols,
            "cells_per_seed_slot": self.rows * self.cols,
        }


def normalize_action_space_mode(value: Any) -> str:
    mode = str(value or ACTION_SPACE_FIXED).strip().lower()
    if mode in {"dynamic", "dynamic14", "dynamic_14"}:
        return ACTION_SPACE_DYNAMIC_14
    if mode in {
        "adventure_14slot_identity",
        "adventure_identity",
        "identity_14",
        "14slot_identity",
        "adventure_generalist_14slot",
    }:
        return ACTION_SPACE_ADVENTURE_14_IDENTITY
    if mode in {"fixed", "legacy", ""}:
        return ACTION_SPACE_FIXED
    raise ValueError(f"Unsupported action_space_mode: {value!r}")


def build_action_space_spec(
    *,
    mode: str = ACTION_SPACE_FIXED,
    plant_types: Optional[List[int]] = None,
    max_seed_slots: Optional[int] = None,
    rows: int = DEFAULT_ROWS,
    cols: int = DEFAULT_COLS,
) -> ActionSpaceSpec:
    normalized_mode = normalize_action_space_mode(mode)
    plant_slot_count = len(plant_types or [])
    cells = int(rows) * int(cols)
    if cells <= 0:
        raise ValueError(f"Invalid action-space board dimensions: rows={rows}, cols={cols}")

    if normalized_mode == ACTION_SPACE_DYNAMIC_14:
        return ActionSpaceSpec(
            mode=ACTION_SPACE_DYNAMIC_14,
            action_count=DYNAMIC_ACTION_COUNT,
            max_seed_slots=DYNAMIC_MAX_SEED_SLOTS,
            observation_version=DYNAMIC_OBSERVATION_VERSION,
            action_decoder_version=DYNAMIC_ACTION_DECODER_VERSION,
            wait_action=DYNAMIC_WAIT_ACTION,
            placement_action_min=0,
            placement_action_max=DYNAMIC_WAIT_ACTION - 1,
            rows=int(rows),
            cols=int(cols),
        )

    if normalized_mode == ACTION_SPACE_ADVENTURE_14_IDENTITY:
        return ActionSpaceSpec(
            mode=ACTION_SPACE_ADVENTURE_14_IDENTITY,
            action_count=ADVENTURE_IDENTITY_ACTION_COUNT,
            max_seed_slots=DYNAMIC_MAX_SEED_SLOTS,
            observation_version=ADVENTURE_IDENTITY_OBSERVATION_VERSION,
            action_decoder_version=ADVENTURE_IDENTITY_ACTION_DECODER_VERSION,
            wait_action=ADVENTURE_IDENTITY_WAIT_ACTION,
            placement_action_min=1,
            placement_action_max=ADVENTURE_IDENTITY_ACTION_COUNT - 1,
            rows=int(rows),
            cols=int(cols),
        )

    fixed_slots = int(max_seed_slots) if max_seed_slots is not None else plant_slot_count
    fixed_slots = max(0, fixed_slots)
    return ActionSpaceSpec(
        mode=ACTION_SPACE_FIXED,
        action_count=1 + fixed_slots * cells,
        max_seed_slots=fixed_slots,
        observation_version=FIXED_OBSERVATION_VERSION,
        action_decoder_version=FIXED_ACTION_DECODER_VERSION,
        wait_action=FIXED_WAIT_ACTION,
        placement_action_min=1,
        placement_action_max=fixed_slots * cells,
        rows=int(rows),
        cols=int(cols),
    )


def spec_from_config(config: Dict[str, Any]) -> ActionSpaceSpec:
    return build_action_space_spec(
        mode=str(config.get("action_space_mode", ACTION_SPACE_FIXED)),
        plant_types=[int(value) for value in config.get("plant_types", [])],
        max_seed_slots=int(config["max_seed_slots"]) if config.get("max_seed_slots") is not None else None,
        rows=int(config.get("row_count", config.get("rows", DEFAULT_ROWS)) or DEFAULT_ROWS),
        cols=int(config.get("column_count", config.get("cols", DEFAULT_COLS)) or DEFAULT_COLS),
    )


def action_count_for_config(config: Dict[str, Any]) -> int:
    return spec_from_config(config).action_count


def fixed_action_to_slot_cell(action: int, rows: int = DEFAULT_ROWS, cols: int = DEFAULT_COLS) -> Dict[str, int]:
    action_id = int(action)
    if action_id <= 0:
        return {"kind": 0, "slot_index": -1, "row": -1, "column": -1}
    cells = int(rows) * int(cols)
    encoded = action_id - 1
    return {
        "kind": 1,
        "slot_index": encoded // cells,
        "row": (encoded % cells) // int(cols),
        "column": encoded % int(cols),
    }


def dynamic_action_to_slot_cell(action: int, rows: int = DEFAULT_ROWS, cols: int = DEFAULT_COLS) -> Dict[str, int]:
    action_id = int(action)
    if action_id == DYNAMIC_WAIT_ACTION:
        return {"kind": 0, "slot_index": -1, "row": -1, "column": -1}
    if action_id < 0 or action_id >= DYNAMIC_WAIT_ACTION:
        return {"kind": -1, "slot_index": -1, "row": -1, "column": -1}
    cells = int(rows) * int(cols)
    return {
        "kind": 1,
        "slot_index": action_id // cells,
        "row": (action_id % cells) // int(cols),
        "column": action_id % int(cols),
    }


def adventure_identity_action_to_slot_cell(action: int, rows: int = DEFAULT_ROWS, cols: int = DEFAULT_COLS) -> Dict[str, int]:
    action_id = int(action)
    if action_id == ADVENTURE_IDENTITY_WAIT_ACTION:
        return {"kind": 0, "slot_index": -1, "row": -1, "column": -1}
    if action_id < 1 or action_id >= ADVENTURE_IDENTITY_ACTION_COUNT:
        return {"kind": -1, "slot_index": -1, "row": -1, "column": -1}
    cells = int(rows) * int(cols)
    encoded = action_id - 1
    return {
        "kind": 1,
        "slot_index": encoded // cells,
        "row": (encoded % cells) // int(cols),
        "column": encoded % int(cols),
    }


def policy_action_to_legacy_action(
    action: int,
    *,
    mode: str,
    rows: int = DEFAULT_ROWS,
    cols: int = DEFAULT_COLS,
) -> int:
    normalized_mode = normalize_action_space_mode(mode)
    action_id = int(action)
    if normalized_mode in {ACTION_SPACE_FIXED, ACTION_SPACE_ADVENTURE_14_IDENTITY}:
        return action_id
    decoded = dynamic_action_to_slot_cell(action_id, rows=rows, cols=cols)
    if int(decoded.get("kind", -1)) == 0:
        return 0
    if int(decoded.get("kind", -1)) != 1:
        return 0
    return 1 + action_id


def legacy_action_to_policy_action(
    action: int,
    *,
    mode: str,
) -> int:
    normalized_mode = normalize_action_space_mode(mode)
    action_id = int(action)
    if normalized_mode in {ACTION_SPACE_FIXED, ACTION_SPACE_ADVENTURE_14_IDENTITY}:
        return action_id
    if action_id <= 0:
        return DYNAMIC_WAIT_ACTION
    return action_id - 1


def decode_policy_action(
    action: int,
    *,
    mode: str,
    observation: Optional[Dict[str, Any]] = None,
    plant_types: Optional[List[int]] = None,
    max_seed_slots: Optional[int] = None,
    rows: int = DEFAULT_ROWS,
    cols: int = DEFAULT_COLS,
) -> Dict[str, int]:
    obs = observation if isinstance(observation, dict) else {}
    obs_rows = int(obs.get("rowCount") or rows)
    obs_cols = int(obs.get("columnCount") or cols)
    normalized_mode = normalize_action_space_mode(mode)
    if normalized_mode == ACTION_SPACE_DYNAMIC_14:
        decoded = dynamic_action_to_slot_cell(int(action), rows=obs_rows, cols=obs_cols)
    elif normalized_mode == ACTION_SPACE_ADVENTURE_14_IDENTITY:
        decoded = adventure_identity_action_to_slot_cell(int(action), rows=obs_rows, cols=obs_cols)
    else:
        decoded = fixed_action_to_slot_cell(int(action), rows=obs_rows, cols=obs_cols)

    plant_type = -1
    slot_index = int(decoded.get("slot_index", -1))
    if int(decoded.get("kind", -1)) == 1 and slot_index >= 0:
        slots = obs.get("seedSlots", []) if isinstance(obs.get("seedSlots", []), list) else []
        if 0 <= slot_index < len(slots) and isinstance(slots[slot_index], dict):
            try:
                plant_type = int(slots[slot_index].get("plantType", -1))
            except (TypeError, ValueError):
                plant_type = -1
        elif plant_types and 0 <= slot_index < len(plant_types):
            plant_type = int(plant_types[slot_index])
        elif max_seed_slots is not None and slot_index >= int(max_seed_slots):
            decoded["kind"] = -1
    decoded["plant_type"] = plant_type
    return decoded


def structural_dynamic_mask(active_seed_slots: int, rows: int = DEFAULT_ROWS, cols: int = DEFAULT_COLS) -> List[bool]:
    """Return the decoder-structural dynamic mask before live cooldown/cost filters."""

    slots = max(0, min(DYNAMIC_MAX_SEED_SLOTS, int(active_seed_slots)))
    cells = int(rows) * int(cols)
    mask = [False] * DYNAMIC_ACTION_COUNT
    for action in range(slots * cells):
        mask[action] = True
    mask[DYNAMIC_WAIT_ACTION] = True
    return mask


def structural_adventure_identity_mask(active_seed_slots: int, rows: int = DEFAULT_ROWS, cols: int = DEFAULT_COLS) -> List[bool]:
    """Return the decoder-structural 14-slot identity mask before live filters."""

    slots = max(0, min(DYNAMIC_MAX_SEED_SLOTS, int(active_seed_slots)))
    cells = int(rows) * int(cols)
    mask = [False] * ADVENTURE_IDENTITY_ACTION_COUNT
    mask[ADVENTURE_IDENTITY_WAIT_ACTION] = True
    for slot_index in range(slots):
        start = 1 + slot_index * cells
        for action in range(start, start + cells):
            mask[action] = True
    return mask
