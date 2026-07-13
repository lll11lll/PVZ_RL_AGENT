"""Shared synthetic fixtures for refactor contracts and microbenchmarks.

This module is test/tool support.  It never connects to the live bridge and is
excluded from first-party runtime line statistics.
"""

from __future__ import annotations

import copy
import hashlib
import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Dict

import numpy as np

from pvzrl_action_space import ACTION_SPACE_ADVENTURE_14_IDENTITY, ACTION_SPACE_FIXED
from pvzrl_sb3 import PvZMaskedPPOEnv, PvZSB3Config


ROOT = Path(__file__).resolve().parents[1]
OBSERVATION_FIXTURE = ROOT / "python" / "fixtures" / "refactor_contracts" / "synthetic_observation.json"
STARTER_NAMES = ["SunFlower", "SunFlower", "Peashooter", "Peashooter"]
STARTER_TYPES = [1, 1, 0, 0]


def load_observation_fixture() -> Dict[str, Any]:
    return json.loads(OBSERVATION_FIXTURE.read_text(encoding="utf-8"))


def make_wrapper(
    *,
    identity: bool,
    fusion_enabled: bool = True,
    tactical_masks: bool = False,
    wallnut_tactical_mask: bool = False,
    cherrybomb_tactical_mask: bool = False,
) -> PvZMaskedPPOEnv:
    config = PvZSB3Config(
        plant_types=list(STARTER_TYPES),
        seed_list=list(STARTER_NAMES),
        action_space_mode=(ACTION_SPACE_ADVENTURE_14_IDENTITY if identity else ACTION_SPACE_FIXED),
        max_seed_slots=(14 if identity else 4),
        fusion_action_mask_enabled=bool(fusion_enabled),
        enable_board_plant_identity=bool(identity),
        tactical_masks=bool(tactical_masks),
        wallnut_tactical_mask=bool(wallnut_tactical_mask),
        cherrybomb_tactical_mask=bool(cherrybomb_tactical_mask),
    )
    # Wrapper initialization prints coach diagnostics but does not open a bridge
    # connection.  Keep test/benchmark output machine-readable.
    with redirect_stdout(StringIO()):
        return PvZMaskedPPOEnv(config)


def observation_for_wrapper(wrapper: PvZMaskedPPOEnv) -> Dict[str, Any]:
    observation = load_observation_fixture()
    observation["actionCount"] = int(wrapper.action_count)
    return observation


def dense_observation(*, slot_count: int) -> Dict[str, Any]:
    """Return a deterministic dense board used to expose repeated scans."""

    observation = load_observation_fixture()
    rows = 5
    cols = 10
    cells = rows * cols
    slot_types = [1, 0, 3, 2, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1][:slot_count]
    type_names = {
        0: "Peashooter",
        1: "SunFlower",
        2: "CherryBomb",
        3: "WallNut",
        4: "PotatoMine",
        5: "SnowPea",
        6: "Chomper",
        7: "Repeater",
        8: "PuffShroom",
        9: "SunShroom",
        10: "FumeShroom",
        11: "GraveBuster",
    }
    costs = {0: 100, 1: 50, 2: 150, 3: 50, 4: 25, 5: 175, 6: 150, 7: 200, 8: 0, 9: 25, 10: 75, 11: 75}
    observation["seedSlots"] = [
        {
            "slotIndex": index,
            "cardInstanceId": 5000 + index,
            "plantType": plant_type,
            "plantTypeName": type_names[plant_type],
            "seedCost": costs[plant_type],
            "ready": True,
            "disabled": False,
            "isAvailable": True,
            "rawCooldown": 0.0,
            "fullCooldown": 7.5,
            "currentCooldown": 0.0,
            "usable": True,
            "source": "dense_fixture",
        }
        for index, plant_type in enumerate(slot_types)
    ]

    occupied_flats = {index * 2 for index in range(25)}
    board_types = [1, 0, 1030, 1090]
    board_names = {0: "Peashooter", 1: "SunFlower", 1030: "DoubleShooer", 1090: "SplitPea"}
    plants = []
    for index, flat in enumerate(sorted(occupied_flats)):
        plant_type = board_types[index % len(board_types)]
        plants.append(
            {
                "row": flat // cols,
                "column": flat % cols,
                "type": plant_type,
                "typeName": board_names[plant_type],
                "instanceId": 6000 + index,
                "health": 250 + (index % 3) * 50,
                "maxHealth": 400,
                "attackCooldown": float(index % 4) / 4.0,
                "produceCooldown": float(index % 5),
            }
        )
    observation["plants"] = plants
    observation["visiblePlants"] = []
    observation["plantCount"] = len(plants)
    observation["visiblePlantObjectCount"] = len(plants)
    observation["totalPlantHealth"] = sum(int(plant["health"]) for plant in plants)

    zombies = []
    for index in range(30):
        row = index % rows
        zombies.append(
            {
                "row": row,
                "type": 2 + (index % 4),
                "x": 2.0 + float(index % 8),
                "health": 180 + (index % 5) * 90,
                "maxHealth": 900,
            }
        )
    observation["zombies"] = zombies
    observation["zombieCount"] = len(zombies)
    lanes = []
    for row in range(rows):
        lane_zombies = [zombie for zombie in zombies if int(zombie["row"]) == row]
        nearest = min(lane_zombies, key=lambda item: float(item["x"]))
        lanes.append(
            {
                "row": row,
                "zombieCount": len(lane_zombies),
                "nearestZombieX": float(nearest["x"]),
                "nearestZombieHealth": int(nearest["health"]),
                "nearestZombieType": int(nearest["type"]),
                "danger": max(0.0, 1.0 - float(nearest["x"]) / 10.0),
            }
        )
    observation["lanes"] = lanes

    legal_actions = [0]
    empty_flats = [flat for flat in range(cells) if flat not in occupied_flats]
    for slot_index in range(slot_count):
        legal_actions.extend(1 + slot_index * cells + flat for flat in empty_flats)
    observation["legalActions"] = legal_actions
    observation["legalActionCount"] = len(legal_actions)
    observation["actionCount"] = 1 + slot_count * cells
    observation["sun"] = 500
    return observation


def low_sun_variant(observation: Dict[str, Any]) -> Dict[str, Any]:
    updated = copy.deepcopy(observation)
    updated["sun"] = 0
    return updated


def cooldown_variant(observation: Dict[str, Any]) -> Dict[str, Any]:
    updated = copy.deepcopy(observation)
    for slot in updated.get("seedSlots", []):
        slot["ready"] = False
        slot["rawCooldown"] = 1.0
        slot["currentCooldown"] = 1.0
        slot["fullCooldown"] = 7.5
    return updated


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(values.astype("<f4", copy=False).tobytes()).hexdigest()


def mask_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(values.astype(np.uint8, copy=False).tobytes()).hexdigest()


def json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
