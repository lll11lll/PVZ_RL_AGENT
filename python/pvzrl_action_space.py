"""Adventure Generalist action-space contract.

Adventure Generalist is the sole maintained policy layout.  Its action space
has one wait action followed by fourteen padded 6x10 seed-slot placement
blocks.  Five-lane Adventure boards use the same layout: their sixth-lane
actions are masked from the live bridge and their observation features are
zero-padded.

* action ``0`` waits;
* actions ``1..840`` place/fuse using seed-slot-major ordering;
* decoder version ``seedslot14x60_padded6x10_plus_wait_v2`` is checkpoint
  semantics.

The ``dynamic_seed_slots`` metadata key describes the live seed inventory; it
does not resize the fixed model action surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


ACTION_SPACE_ADVENTURE_14_IDENTITY = "adventure_14slot_identity_full_v2"
ACTION_SPACE_MODES = {ACTION_SPACE_ADVENTURE_14_IDENTITY}

DEFAULT_ROWS = 6
DEFAULT_COLS = 10
SUPPORTED_LIVE_ROWS = frozenset({5, DEFAULT_ROWS})
CELLS_PER_SLOT = DEFAULT_ROWS * DEFAULT_COLS
ADVENTURE_IDENTITY_MAX_SEED_SLOTS = 14

ADVENTURE_IDENTITY_OBSERVATION_VERSION = "adventure_14slot_identity_full_v2"
ADVENTURE_IDENTITY_ACTION_DECODER_VERSION = "seedslot14x60_padded6x10_plus_wait_v2"
ADVENTURE_IDENTITY_WAIT_ACTION = 0
ADVENTURE_IDENTITY_ACTION_COUNT = ADVENTURE_IDENTITY_MAX_SEED_SLOTS * CELLS_PER_SLOT + 1


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
        """Return the serialized Generalist inventory-capability flag."""

        return True

    @property
    def identity_seed_slots(self) -> bool:
        return True

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
    mode = str(value or ACTION_SPACE_ADVENTURE_14_IDENTITY).strip().lower()
    if mode == ACTION_SPACE_ADVENTURE_14_IDENTITY:
        return ACTION_SPACE_ADVENTURE_14_IDENTITY
    raise ValueError(
        f"Unsupported action_space_mode: {value!r}; "
        f"expected {ACTION_SPACE_ADVENTURE_14_IDENTITY!r}"
    )


def _validate_board_geometry(rows: int, cols: int) -> None:
    if not is_supported_live_board_geometry(rows, cols):
        raise ValueError(
            "Adventure Generalist requires a 5x10 or 6x10 live board under "
            "the fixed padded 6x10 model contract: "
            f"rows={rows}, cols={cols}"
        )


def is_supported_live_board_geometry(rows: int, cols: int) -> bool:
    """Return whether a runtime board fits the padded full-Adventure model."""

    return int(rows) in SUPPORTED_LIVE_ROWS and int(cols) == DEFAULT_COLS


def build_action_space_spec(
    *,
    mode: str = ACTION_SPACE_ADVENTURE_14_IDENTITY,
    plant_types: Optional[List[int]] = None,
    max_seed_slots: Optional[int] = None,
    rows: int = DEFAULT_ROWS,
    cols: int = DEFAULT_COLS,
) -> ActionSpaceSpec:
    normalize_action_space_mode(mode)
    _validate_board_geometry(rows, cols)
    if max_seed_slots is not None and int(max_seed_slots) != ADVENTURE_IDENTITY_MAX_SEED_SLOTS:
        raise ValueError(
            "Adventure Generalist requires exactly 14 seed slots: "
            f"max_seed_slots={max_seed_slots}"
        )
    # ``plant_types`` describes the current live loadout and may contain fewer
    # than fourteen entries.  It does not resize the checkpoint action space.
    _ = plant_types
    return ActionSpaceSpec(
        mode=ACTION_SPACE_ADVENTURE_14_IDENTITY,
        action_count=ADVENTURE_IDENTITY_ACTION_COUNT,
        max_seed_slots=ADVENTURE_IDENTITY_MAX_SEED_SLOTS,
        observation_version=ADVENTURE_IDENTITY_OBSERVATION_VERSION,
        action_decoder_version=ADVENTURE_IDENTITY_ACTION_DECODER_VERSION,
        wait_action=ADVENTURE_IDENTITY_WAIT_ACTION,
        placement_action_min=1,
        placement_action_max=ADVENTURE_IDENTITY_ACTION_COUNT - 1,
        rows=DEFAULT_ROWS,
        cols=DEFAULT_COLS,
    )


def spec_from_config(config: Dict[str, Any]) -> ActionSpaceSpec:
    return build_action_space_spec(
        mode=str(config.get("action_space_mode") or ACTION_SPACE_ADVENTURE_14_IDENTITY),
        plant_types=[int(value) for value in config.get("plant_types", [])],
        max_seed_slots=int(config["max_seed_slots"]) if config.get("max_seed_slots") is not None else None,
        rows=int(config.get("row_count", config.get("rows", DEFAULT_ROWS)) or DEFAULT_ROWS),
        cols=int(config.get("column_count", config.get("cols", DEFAULT_COLS)) or DEFAULT_COLS),
    )


def action_count_for_config(config: Dict[str, Any]) -> int:
    return spec_from_config(config).action_count


def adventure_identity_action_to_slot_cell(
    action: int,
    rows: int = DEFAULT_ROWS,
    cols: int = DEFAULT_COLS,
) -> Dict[str, int]:
    _validate_board_geometry(rows, cols)
    action_id = int(action)
    if action_id == ADVENTURE_IDENTITY_WAIT_ACTION:
        return {"kind": 0, "slot_index": -1, "row": -1, "column": -1}
    if action_id < 1 or action_id >= ADVENTURE_IDENTITY_ACTION_COUNT:
        return {"kind": -1, "slot_index": -1, "row": -1, "column": -1}
    encoded = action_id - 1
    return {
        "kind": 1,
        "slot_index": encoded // CELLS_PER_SLOT,
        "row": (encoded % CELLS_PER_SLOT) // DEFAULT_COLS,
        "column": encoded % DEFAULT_COLS,
    }


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
    normalize_action_space_mode(mode)
    # The decoder is checkpoint semantics.  Runtime observations may expose a
    # five-lane board, but they must never reinterpret the permanent 60-cell
    # action blocks as 50-cell blocks.
    obs = observation if isinstance(observation, dict) else {}
    decoded = adventure_identity_action_to_slot_cell(int(action), rows=rows, cols=cols)

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


def structural_adventure_identity_mask(
    active_seed_slots: int,
    rows: int = DEFAULT_ROWS,
    cols: int = DEFAULT_COLS,
) -> List[bool]:
    """Return the Generalist decoder-structural mask before live filters."""

    _validate_board_geometry(rows, cols)
    slots = max(0, min(ADVENTURE_IDENTITY_MAX_SEED_SLOTS, int(active_seed_slots)))
    mask = [False] * ADVENTURE_IDENTITY_ACTION_COUNT
    mask[ADVENTURE_IDENTITY_WAIT_ACTION] = True
    for slot_index in range(slots):
        start = 1 + slot_index * CELLS_PER_SLOT
        for action in range(start, start + CELLS_PER_SLOT):
            mask[action] = True
    return mask
