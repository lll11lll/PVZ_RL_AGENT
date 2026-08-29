"""Regression coverage for the full-Adventure Generalist v2 contract."""

from __future__ import annotations

from pvzrl_action_space import (
    ADVENTURE_IDENTITY_ACTION_COUNT,
    ADVENTURE_IDENTITY_OBSERVATION_VERSION,
    CELLS_PER_SLOT,
    DEFAULT_COLS,
    DEFAULT_ROWS,
    build_action_space_spec,
    decode_policy_action,
)
from pvzrl_adventure import RuntimeEvaluationLoadoutSelector
from pvzrl_observation_layout import build_observation_layout


def test_full_adventure_model_contract_is_padded_six_by_ten() -> None:
    spec = build_action_space_spec(rows=5, cols=10)

    assert (spec.rows, spec.cols) == (DEFAULT_ROWS, DEFAULT_COLS)
    assert spec.action_count == ADVENTURE_IDENTITY_ACTION_COUNT == 841
    assert spec.placement_action_max == 840
    assert build_observation_layout(spec).shape == (4364,)
    assert spec.observation_version == ADVENTURE_IDENTITY_OBSERVATION_VERSION

    # The fifth live lane (row 4) in slot 1 must follow slot 1's 60-cell
    # block; action 51 is permanently slot 0's sixth (padded) row.
    slot_one_fifth_lane = 1 + CELLS_PER_SLOT + 4 * DEFAULT_COLS
    assert decode_policy_action(slot_one_fifth_lane, mode=spec.mode) == {
        "kind": 1,
        "slot_index": 1,
        "row": 4,
        "column": 0,
        "plant_type": -1,
    }
    assert decode_policy_action(51, mode=spec.mode)["row"] == 5


def test_evaluation_loadout_uses_interactive_pool_cards_without_capacity_expansion() -> None:
    context: dict = {}
    selector = RuntimeEvaluationLoadoutSelector(
        configured_seed_list=["SunFlower", "SunFlower", "Peashooter", "Peashooter"],
        max_seed_slots=14,
        context=context,
    )
    selected, blocked_reason = selector(
        {
            "selectableSeedNames": [
                "SunFlower",
                "Peashooter",
                "WallNut",
                "CherryBomb",
                "Lily Pad",
                "Tangle Kelp",
            ],
            "seedBankCapacity": 4,
        },
        ["SunFlower", "SunFlower", "Peashooter", "Peashooter"],
    )

    assert blocked_reason == ""
    assert selected == ["SunFlower", "Peashooter", "LilyPad", "Tanglekelp"]
    assert context["evaluation_seed_selection_evidence"] == "interactive_cardui"
    assert context["evaluation_seed_selection_capacity"] == 4


def test_evaluation_loadout_does_not_invent_cards_when_chooser_evidence_is_empty() -> None:
    context: dict = {}
    selector = RuntimeEvaluationLoadoutSelector(
        configured_seed_list=["SunFlower", "SunFlower", "Peashooter", "Peashooter"],
        max_seed_slots=14,
        context=context,
    )
    selected, blocked_reason = selector({}, ["SunFlower", "SunFlower", "Peashooter", "Peashooter"])

    assert blocked_reason == ""
    assert selected == ["SunFlower", "SunFlower", "Peashooter", "Peashooter"]
    assert context["evaluation_seed_selection_evidence"] == "none_preserved_current_loadout"
