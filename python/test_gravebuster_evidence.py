"""Regression coverage for selectable-card unlock evidence."""

from __future__ import annotations

from collections import Counter

from pvzrl_adventure import _snapshot_unlock_state, update_unlocked_from_state
from pvzrl_adventure_generalist import _filter_supported_seed_names


def test_gravebuster_text_is_diagnostic_but_not_selectable_unlock_evidence() -> None:
    unlocked = Counter({"SunFlower": 1, "Peashooter": 1})
    state = {
        "screenState": "reward_unlock",
        "newPlantUnlockedVisible": True,
        "newPlantUnlockedName": "Gravebuster",
        "newPlantUnlockedPlantType": 239,
        "visibleRewardTexts": ["Grave Buster"],
        "unlockSnapshot": {
            "newPlantUnlockedName": "Gravebuster",
            "newPlantUnlockedPlantType": 239,
            "visibleRewardTexts": ["Grave Buster"],
            "visibleSeedCardNames": [],
            "visibleSeedPlantTypes": [],
        },
    }

    newly_unlocked = update_unlocked_from_state(unlocked, state, source="reward_unlock")
    snapshot = _snapshot_unlock_state(state, source="reward_unlock")

    assert newly_unlocked == []
    assert "Gravebuster" not in unlocked
    assert snapshot["visibleSeedCardNames"] == []
    assert "Gravebuster" in snapshot["diagnosticPlantNames"]
    assert 239 in snapshot["diagnosticPlantTypes"]
    assert _filter_supported_seed_names(["Gravebuster", "Grave Buster"]) == []


def test_cardui_seed_evidence_still_confirms_a_later_plant() -> None:
    unlocked = Counter({"SunFlower": 1, "Peashooter": 1})
    state = {
        "screenState": "seed_selection",
        "confirmedSelectableSeedCardNames": ["Chomper"],
        "confirmedSelectableSeedPlantTypes": [5],
        "visibleSeedCardNames": ["Chomper"],
        "visibleSeedPlantTypes": [5],
    }

    newly_unlocked = update_unlocked_from_state(unlocked, state, source="seed_selection")

    assert newly_unlocked == ["Chomper"]
    assert unlocked["Chomper"] == 1
