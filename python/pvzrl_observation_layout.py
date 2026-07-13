"""Pure observation-width contract shared by metadata and the SB3 wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from pvzrl_action_space import ActionSpaceSpec, spec_from_config
from pvzrl_seed_inventory import adventure_identity_feature_count


@dataclass(frozen=True, slots=True)
class ObservationLayout:
    global_features: int
    card_slot_count: int
    card_features: int
    cell_features: int
    lane_features: int
    seed_inventory_features: int

    @property
    def total_features(self) -> int:
        return int(
            self.global_features
            + self.card_features
            + self.cell_features
            + self.lane_features
            + self.seed_inventory_features
        )

    @property
    def shape(self) -> Tuple[int, ...]:
        return (self.total_features,)


def build_observation_layout(spec: ActionSpaceSpec) -> ObservationLayout:
    rows = int(spec.rows)
    cols = int(spec.cols)
    card_slot_count = int(spec.max_seed_slots)
    return ObservationLayout(
        global_features=12,
        card_slot_count=card_slot_count,
        card_features=5 * card_slot_count,
        cell_features=6 * rows * cols,
        lane_features=5 * rows,
        seed_inventory_features=adventure_identity_feature_count(card_slot_count),
    )


def observation_shape_for_config(config: Mapping[str, Any]) -> Tuple[int, ...]:
    spec = spec_from_config(dict(config))
    return build_observation_layout(spec).shape
