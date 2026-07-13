"""Phase 3 compatibility contracts for the shared action/fusion pipeline.

The assertions in this file intentionally use independent, exhaustive oracles
where practical.  They protect the externally visible 201/701 action layouts,
the complete mask (not a sampling of actions), fusion recipe/compatibility
semantics, rejection precedence, and one-event/one-count accounting while the
runtime paths are consolidated.
"""

from __future__ import annotations

import copy
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from itertools import product

import numpy as np
import pytest
import pvzrl_env

from pvzrl_action_space import (
    ACTION_SPACE_ADVENTURE_14_IDENTITY,
    ACTION_SPACE_DYNAMIC_14,
    ACTION_SPACE_FIXED,
    ADVENTURE_IDENTITY_ACTION_COUNT,
    DYNAMIC_ACTION_COUNT,
    build_action_space_spec,
    decode_policy_action,
    legacy_action_to_policy_action,
    policy_action_to_legacy_action,
)
from pvzrl_actions import (
    ACTION_KIND_FUSION,
    ActionResult,
    ActionValidationConfig,
    build_action_decision_cache,
    build_action_intent,
    build_action_validation_context,
    compatible_pairs_for_observation,
    validate_action_intent,
    validate_policy_action,
)
from pvzrl_fusion import (
    FUSION_COMPATIBILITY,
    FUSION_ILLEGAL_BRIDGE_UNAVAILABLE,
    FUSION_ILLEGAL_COOLDOWN,
    FUSION_ILLEGAL_DISABLED,
    FUSION_ILLEGAL_EMPTY_TILE,
    FUSION_ILLEGAL_INCOMPATIBLE,
    FUSION_ILLEGAL_INSUFFICIENT_SUN,
    FUSION_ILLEGAL_INVALID_COL,
    FUSION_ILLEGAL_INVALID_ROW,
    FUSION_ILLEGAL_INVALID_SEED_SLOT,
    FUSION_ILLEGAL_SEED_UNAVAILABLE,
    FUSION_REJECTION_REASONS,
    FUSION_RECIPES,
    FUSION_RECIPES_BY_PAIR,
    FUSION_RULES,
    RUNTIME_ONLY_FUSION_COMPATIBILITY,
    RUNTIME_ONLY_FUSION_COMPATIBILITY_CASES,
    account_fusion_execution_once,
    apply_fusion_attempt_result,
    are_fusion_compatible,
    default_fusion_diagnostics,
    fusion_compatibility_kind,
    fusion_execution_from_result,
    fusion_intent_from_candidate,
    fusion_recipe,
    get_fusion_illegal_reason,
    validate_fusion_intent,
)
from pvzrl_env import BridgeTimeoutError, PvZEnvConfig, PvZGymEnv
from pvzrl_human_coach import parse_coach_command, validate_coach_command_for_env
from pvzrl_sb3 import PvZMaskedPPOEnv, PvZSB3Config
from test_refactor_support import STARTER_NAMES, STARTER_TYPES, load_observation_fixture, make_wrapper


ROWS = 5
COLS = 10
CELLS = ROWS * COLS


def _expected_decode(action: int, *, mode: str) -> tuple[int, int, int, int]:
    """Independent policy-action decoder oracle: kind, slot, row, column."""

    if mode == ACTION_SPACE_DYNAMIC_14:
        if action == 700:
            return 0, -1, -1, -1
        encoded = action
    else:
        if action == 0:
            return 0, -1, -1, -1
        encoded = action - 1
    return 1, encoded // CELLS, (encoded % CELLS) // COLS, encoded % COLS


@pytest.mark.parametrize(
    ("mode", "slots", "action_count"),
    (
        (ACTION_SPACE_FIXED, 4, 201),
        (ACTION_SPACE_DYNAMIC_14, 14, DYNAMIC_ACTION_COUNT),
        (ACTION_SPACE_ADVENTURE_14_IDENTITY, 14, ADVENTURE_IDENTITY_ACTION_COUNT),
    ),
)
def test_every_policy_action_round_trips_and_decodes_exactly(
    mode: str,
    slots: int,
    action_count: int,
) -> None:
    spec = build_action_space_spec(
        mode=mode,
        plant_types=list(STARTER_TYPES),
        max_seed_slots=slots,
        rows=ROWS,
        cols=COLS,
    )
    assert spec.action_count == action_count
    for policy_action in range(action_count):
        legacy_action = policy_action_to_legacy_action(
            policy_action,
            mode=mode,
            rows=ROWS,
            cols=COLS,
        )
        assert legacy_action_to_policy_action(legacy_action, mode=mode) == policy_action
        decoded = decode_policy_action(
            policy_action,
            mode=mode,
            plant_types=list(range(slots)),
            max_seed_slots=slots,
            rows=ROWS,
            cols=COLS,
        )
        expected_kind, expected_slot, expected_row, expected_column = _expected_decode(
            policy_action,
            mode=mode,
        )
        assert (
            decoded["kind"],
            decoded["slot_index"],
            decoded["row"],
            decoded["column"],
        ) == (expected_kind, expected_slot, expected_row, expected_column)


def _occupied_by_cell(observation: dict) -> dict[tuple[int, int], int]:
    return {
        (int(plant["row"]), int(plant["column"])): int(plant.get("type", plant.get("plantType", -1)))
        for plant in observation.get("plants", [])
        if isinstance(plant, dict)
    }


def _expected_legacy_mask(observation: dict, *, fusion_enabled: bool) -> list[bool]:
    """Independent oracle for the complete legacy wait-plus-slot mask."""

    rows = int(observation["rowCount"])
    cols = int(observation["columnCount"])
    cells = rows * cols
    action_count = int(observation["actionCount"])
    slots = [slot for slot in observation.get("seedSlots", []) if isinstance(slot, dict)]
    bridge_legal = {int(action) for action in observation.get("legalActions", [])}
    occupied = _occupied_by_cell(observation)
    expected = [False] * action_count
    expected[0] = True
    for action in range(1, action_count):
        encoded = action - 1
        slot_index = encoded // cells
        if not 0 <= slot_index < len(slots):
            continue
        row = (encoded % cells) // cols
        column = encoded % cols
        slot = slots[slot_index]
        if int(slot.get("slotIndex", slot_index)) != slot_index:
            continue
        if not bool(slot.get("usable", False)) or bool(slot.get("disabled", False)):
            continue
        if not bool(slot.get("ready", False)):
            continue
        if float(slot.get("fullCooldown", 0.0) or 0.0) > 0.05 and float(
            slot.get("currentCooldown", 0.0) or 0.0
        ) > 0.05:
            continue
        if int(observation.get("sun", 0) or 0) < max(0, int(slot.get("seedCost", 0) or 0)):
            continue
        existing_type = occupied.get((row, column))
        if existing_type is not None:
            expected[action] = bool(
                fusion_enabled
                and are_fusion_compatible(existing_type, int(slot.get("plantType", -1)))
            )
        else:
            expected[action] = action in bridge_legal
    return expected


def _dynamic_wrapper() -> PvZMaskedPPOEnv:
    config = PvZSB3Config(
        plant_types=list(STARTER_TYPES),
        seed_list=list(STARTER_NAMES),
        action_space_mode=ACTION_SPACE_DYNAMIC_14,
        max_seed_slots=14,
        fusion_action_mask_enabled=True,
    )
    with redirect_stdout(StringIO()):
        return PvZMaskedPPOEnv(config)


def test_complete_701_entry_masks_match_independent_oracle_in_both_layouts() -> None:
    observation = load_observation_fixture()
    observation["actionCount"] = 701
    legacy_expected = _expected_legacy_mask(observation, fusion_enabled=True)

    identity = make_wrapper(identity=True, fusion_enabled=True)
    dynamic = _dynamic_wrapper()
    try:
        identity._last_observation = copy.deepcopy(observation)
        dynamic._last_observation = copy.deepcopy(observation)
        identity_mask = identity.action_masks()
        dynamic_mask = dynamic.action_masks()
        assert identity_mask.shape == (701,)
        assert dynamic_mask.shape == (701,)
        assert identity_mask.tolist() == legacy_expected

        expected_dynamic = [False] * 701
        expected_dynamic[700] = legacy_expected[0]
        for legacy_action in range(1, 701):
            expected_dynamic[legacy_action - 1] = legacy_expected[legacy_action]
        assert dynamic_mask.tolist() == expected_dynamic

        # Every policy-visible mask bit must agree with the policy/legacy index
        # adapter, not merely have the same number of enabled entries.
        for policy_action in range(701):
            legacy_action = policy_action_to_legacy_action(
                policy_action,
                mode=ACTION_SPACE_DYNAMIC_14,
            )
            assert bool(dynamic_mask[policy_action]) is bool(legacy_expected[legacy_action])
    finally:
        identity.close()
        dynamic.close()


def _validation_config(
    observation: dict,
    *,
    mode: str = ACTION_SPACE_ADVENTURE_14_IDENTITY,
    fusion_enabled: bool = True,
) -> ActionValidationConfig:
    return ActionValidationConfig(
        action_space_mode=mode,
        plant_types=tuple(STARTER_TYPES),
        max_seed_slots=14,
        rows=ROWS,
        cols=COLS,
        fusion_action_mask_enabled=fusion_enabled,
        fusion_compatible_pairs=compatible_pairs_for_observation(
            observation,
            STARTER_TYPES,
            are_fusion_compatible,
        ),
        compatibility_version="phase3_contract_v1",
    )


@pytest.mark.parametrize(
    "mode",
    (ACTION_SPACE_DYNAMIC_14, ACTION_SPACE_ADVENTURE_14_IDENTITY),
)
def test_authoritative_cache_decisions_match_every_entry_of_the_701_mask(mode: str) -> None:
    observation = load_observation_fixture()
    observation["actionCount"] = 701
    bridge_actions = list(observation["legalActions"])
    config = _validation_config(observation, mode=mode)
    cache = build_action_decision_cache(
        observation,
        config=config,
        bridge_legal_actions=bridge_actions,
    )
    expected_legacy = _expected_legacy_mask(observation, fusion_enabled=True)
    expected_policy = [False] * 701
    for policy_action in range(701):
        legacy_action = policy_action_to_legacy_action(policy_action, mode=mode)
        expected_policy[policy_action] = expected_legacy[legacy_action]
    assert list(cache.mask) == expected_policy
    assert len(cache.decisions) == 701

    # Independently rebuild every intent and validate it against the same
    # immutable context.  The complete mask must be the direct projection of
    # the authoritative decisions, not a separately maintained algorithm.
    context = build_action_validation_context(
        observation,
        config=config,
        bridge_legal_actions=bridge_actions,
    )
    for policy_action, cached in enumerate(cache.decisions):
        intent = build_action_intent(
            policy_action,
            source="execution_guard",
            mode=mode,
            observation=observation,
            plant_types=STARTER_TYPES,
            max_seed_slots=14,
            rows=ROWS,
            cols=COLS,
        )
        direct = validate_action_intent(intent, context)
        assert cached.legal is direct.legal
        assert cached.rejection_reason == direct.rejection_reason
        assert cached.resolved_action_kind == direct.resolved_action_kind
        assert cached.bridge_action == direct.bridge_action


def test_nonstandard_board_dimensions_use_exact_rows_columns_and_cell_stride() -> None:
    observation = {
        "frameCount": 77,
        "rowCount": 3,
        "columnCount": 4,
        "actionCount": 25,
        "gameplayReady": True,
        "boardFound": True,
        "canReadBoard": True,
        "seedSelectionActive": False,
        "sun": 500,
        "seedSlots": [
            {
                "slotIndex": 0,
                "plantType": 0,
                "usable": True,
                "disabled": False,
                "ready": True,
                "seedCost": 100,
                "currentCooldown": 0.0,
                "fullCooldown": 7.5,
            },
            {
                "slotIndex": 1,
                "plantType": 1,
                "usable": True,
                "disabled": False,
                "ready": True,
                "seedCost": 50,
                "currentCooldown": 0.0,
                "fullCooldown": 7.5,
            },
        ],
        "plants": [{"row": 1, "column": 2, "type": 0}],
        "visiblePlants": [],
        "legalActions": [0, 1, 4, 9, 13, 16, 21],
    }
    config = ActionValidationConfig(
        action_space_mode=ACTION_SPACE_FIXED,
        plant_types=(0, 1),
        max_seed_slots=2,
        rows=3,
        cols=4,
        fusion_action_mask_enabled=True,
        fusion_compatible_pairs=frozenset({(0, 0), (0, 1)}),
        compatibility_version="nonstandard_board_v1",
    )
    assert config.spec.action_count == 25
    cache = build_action_decision_cache(
        observation,
        config=config,
        bridge_legal_actions=observation["legalActions"],
    )
    assert len(cache.decisions) == 25
    for action in range(25):
        intent = cache.decisions[action].intent
        if action == 0:
            assert (intent.action_kind, intent.seed_slot, intent.row, intent.column) == (
                "wait",
                -1,
                -1,
                -1,
            )
            continue
        encoded = action - 1
        assert (intent.seed_slot, intent.row, intent.column) == (
            encoded // 12,
            (encoded % 12) // 4,
            encoded % 4,
        )
    occupied_action = 1 + (1 * 4 + 2)
    assert occupied_action == 7
    assert cache.decisions[occupied_action].legal
    assert cache.decisions[occupied_action].resolved_action_kind == ACTION_KIND_FUSION
    assert cache.decisions[occupied_action].existing_plant_type == 0


@pytest.mark.parametrize("observed_action_count", (10, 30))
def test_observation_action_count_mismatch_is_bounded_and_never_exposes_extra_actions(
    observed_action_count: int,
) -> None:
    observation = {
        "frameCount": 78,
        "rowCount": 3,
        "columnCount": 4,
        "actionCount": observed_action_count,
        "gameplayReady": True,
        "boardFound": True,
        "canReadBoard": True,
        "seedSelectionActive": False,
        "sun": 500,
        "seedSlots": [
            {
                "slotIndex": 0,
                "plantType": 0,
                "usable": True,
                "disabled": False,
                "ready": True,
                "seedCost": 100,
            },
            {
                "slotIndex": 1,
                "plantType": 1,
                "usable": True,
                "disabled": False,
                "ready": True,
                "seedCost": 50,
            },
        ],
        "plants": [],
        "visiblePlants": [],
        "legalActions": list(range(25)),
    }
    config = ActionValidationConfig(
        action_space_mode=ACTION_SPACE_FIXED,
        plant_types=(0, 1),
        max_seed_slots=2,
        rows=3,
        cols=4,
    )
    assert config.spec.action_count == 25
    cache = build_action_decision_cache(
        observation,
        config=config,
        bridge_legal_actions=observation["legalActions"],
    )
    assert len(cache.decisions) == observed_action_count
    assert len(cache.mask) == observed_action_count
    assert cache.context.action_count == observed_action_count
    if observed_action_count < config.spec.action_count:
        assert all(cache.decisions[action].legal for action in range(observed_action_count))
    else:
        assert all(cache.decisions[action].legal for action in range(config.spec.action_count))
        for action in range(config.spec.action_count, observed_action_count):
            assert not cache.decisions[action].legal
            assert cache.decisions[action].rejection_reason == "invalid_action_decode"


def test_all_action_sources_receive_identical_decisions_and_stable_schema() -> None:
    observation = load_observation_fixture()
    observation["actionCount"] = 701
    config = _validation_config(observation)
    sources = (
        "model",
        "random",
        "scripted",
        "human_coach",
        "stream_coach",
        "mock_coach",
        "gui",
        "manual",
        "debug",
        "adventure",
    )
    decisions = []
    for source in sources:
        decision = validate_policy_action(
            124,  # slot 2 Peashooter on the occupied Peashooter at row 2/col 3
            source=source,
            observation=observation,
            config=config,
            bridge_legal_actions=observation["legalActions"],
            bridge_command={"command": "fusion_step"},
            source_metadata={"origin": source, "nested": {"stable": True}},
        )
        assert decision.legal
        assert decision.resolved_action_kind == ACTION_KIND_FUSION
        assert decision.source == source
        assert decision.intent.bridge_command == {"command": "fusion_step"}
        assert decision.intent.source_metadata["origin"] == source
        decisions.append(decision)

    parity = {
        (
            decision.legal,
            decision.rejection_reason,
            decision.policy_action,
            decision.legacy_action,
            decision.bridge_action,
            decision.resolved_action_kind,
            decision.selected_plant_type,
            decision.existing_plant_type,
        )
        for decision in decisions
    }
    assert len(parity) == 1

    intent_payload = decisions[0].intent.to_dict()
    assert set(intent_payload) == {
        "source",
        "policy_action",
        "legacy_action",
        "bridge_action",
        "action_kind",
        "seed_slot",
        "row",
        "column",
        "decoded_action",
        "bridge_command",
        "source_metadata",
    }
    decision_payload = decisions[0].to_dict()
    assert set(decision_payload) == {
        "intent",
        "source",
        "policy_action",
        "legacy_action",
        "bridge_action",
        "action_kind",
        "legal",
        "rejection_reason",
        "frame_identity",
        "config_fingerprint",
        "selected_plant_type",
        "existing_plant_type",
        "bridge_authoritative",
        "cache_reused",
    }
    result = ActionResult.from_execution(
        decisions[0],
        {
            "illegalAction": False,
            "fusionSucceeded": True,
            "nested": {"stable": [True]},
        },
    )
    assert set(result.to_dict()) == {
        "decision",
        "executed",
        "bridge_accepted",
        "execution_result",
    }
    assert result.executed and result.bridge_accepted is True

    # Nested source metadata and commands are immutable inside the contract.
    with pytest.raises(TypeError):
        decisions[0].intent.source_metadata["origin"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        decisions[0].intent.source_metadata["nested"]["stable"] = False  # type: ignore[index]
    with pytest.raises(TypeError):
        result.execution_result["nested"]["stable"][0] = False  # type: ignore[index]


def test_action_cache_key_reuses_only_a_demonstrably_identical_frame_and_config() -> None:
    observation = load_observation_fixture()
    observation["actionCount"] = 701
    bridge_actions = list(observation["legalActions"])
    config = _validation_config(observation)

    baseline = build_action_decision_cache(
        observation,
        config=config,
        bridge_legal_actions=bridge_actions,
    )
    identical = build_action_decision_cache(
        copy.deepcopy(observation),
        config=config,
        bridge_legal_actions=list(reversed(bridge_actions)),
        source="model",
    )
    assert baseline.key == identical.key
    assert baseline.key.token == identical.key.token
    assert baseline.mask == identical.mask

    cached_intent = build_action_intent(
        124,
        source="human_coach",
        mode=config.action_space_mode,
        observation=observation,
        plant_types=STARTER_TYPES,
        max_seed_slots=14,
    )
    reused = baseline.decision_for(124, intent=cached_intent)
    assert reused is not None and reused.cache_reused and reused.source == "human_coach"
    assert reused.frame_identity == baseline.key.frame_identity
    assert reused.config_fingerprint == baseline.key.config_fingerprint
    assert baseline.decision_for(10, intent=cached_intent) is None

    changed_frame = copy.deepcopy(observation)
    changed_frame["frameCount"] = int(changed_frame["frameCount"]) + 1
    frame_cache = build_action_decision_cache(
        changed_frame,
        config=config,
        bridge_legal_actions=bridge_actions,
    )
    assert frame_cache.key.frame_identity != baseline.key.frame_identity
    assert frame_cache.key.config_fingerprint == baseline.key.config_fingerprint

    changed_state = copy.deepcopy(observation)
    changed_state["sun"] = 0
    state_cache = build_action_decision_cache(
        changed_state,
        config=config,
        bridge_legal_actions=bridge_actions,
    )
    assert state_cache.key.frame_identity != baseline.key.frame_identity
    assert state_cache.mask != baseline.mask

    bridge_cache = build_action_decision_cache(
        observation,
        config=config,
        bridge_legal_actions=[action for action in bridge_actions if action != 10],
    )
    assert bridge_cache.key.frame_identity != baseline.key.frame_identity
    assert bridge_cache.mask != baseline.mask

    tactical_cache = build_action_decision_cache(
        observation,
        config=config,
        bridge_legal_actions=bridge_actions,
        tactical_rejections={10: "tactical_contract_block"},
    )
    assert tactical_cache.key.frame_identity != baseline.key.frame_identity
    assert tactical_cache.decisions[10].rejection_reason == "tactical_contract_block"

    fusion_disabled = _validation_config(observation, fusion_enabled=False)
    config_cache = build_action_decision_cache(
        observation,
        config=fusion_disabled,
        bridge_legal_actions=bridge_actions,
    )
    assert config_cache.key.frame_identity == baseline.key.frame_identity
    assert config_cache.key.config_fingerprint != baseline.key.config_fingerprint
    assert baseline.decisions[124].legal
    assert not config_cache.decisions[124].legal
    assert config_cache.decisions[124].rejection_reason == "occupied_cell"

    no_compatibility = replace(
        config,
        fusion_compatible_pairs=frozenset(),
        compatibility_version="empty",
    )
    compatibility_cache = build_action_decision_cache(
        observation,
        config=no_compatibility,
        bridge_legal_actions=bridge_actions,
    )
    assert compatibility_cache.key.config_fingerprint != baseline.key.config_fingerprint
    assert compatibility_cache.decisions[124].rejection_reason == FUSION_ILLEGAL_INCOMPATIBLE


def test_environment_reuses_mask_decision_for_filter_and_invalidates_on_state_change() -> None:
    wrapper = make_wrapper(identity=True, fusion_enabled=True)
    try:
        base = wrapper.base
        observation = load_observation_fixture()
        observation["actionCount"] = 701
        first_mask = base.action_mask(observation)
        first_stats = base.action_cache_diagnostics()
        assert first_stats["misses"] == 1
        assert first_stats["hits"] == 0

        assert base.action_mask(copy.deepcopy(observation)) == first_mask
        second_stats = base.action_cache_diagnostics()
        assert second_stats["misses"] == 1
        assert second_stats["hits"] >= 1

        decision = base.action_decision(124, observation, source="human_coach")
        filter_decision = base.action_decision(
            124,
            observation,
            source="python_filter",
            bridge_actions=list(observation["legalActions"]),
        )
        assert decision.cache_reused and decision.source == "human_coach"
        assert filter_decision.cache_reused and filter_decision.legal
        assert filter_decision.rejection_reason == ""
        mismatched_intent = build_action_intent(
            124,
            source="gui",
            mode=ACTION_SPACE_ADVENTURE_14_IDENTITY,
            observation=observation,
            plant_types=STARTER_TYPES,
            max_seed_slots=14,
        )
        mismatch = base.action_decision(10, observation, intent=mismatched_intent)
        assert not mismatch.legal
        assert mismatch.rejection_reason == "action_identity_mismatch"
        assert not mismatch.cache_reused
        guard_stats = base.action_cache_diagnostics()
        assert guard_stats["misses"] == 1
        assert guard_stats["hits"] >= second_stats["hits"] + 2

        changed = copy.deepcopy(observation)
        changed["sun"] = 0
        changed_mask = base.action_mask(changed)
        changed_stats = base.action_cache_diagnostics()
        assert changed_stats["misses"] == 2
        assert changed_stats["key"]["frame_identity"] != first_stats["key"]["frame_identity"]
        assert changed_mask != first_mask
    finally:
        wrapper.close()


def test_environment_execution_safeguard_matches_all_701_cached_mask_bits() -> None:
    wrapper = make_wrapper(identity=True, fusion_enabled=True)
    try:
        observation = load_observation_fixture()
        observation["actionCount"] = 701
        mask = wrapper.base.action_mask(observation)
        assert len(mask) == 701
        for legacy_action, allowed in enumerate(mask):
            decision = wrapper.base.action_decision(
                legacy_action,
                observation,
                source="execution_contract",
            )
            assert decision.bridge_action == legacy_action
            assert decision.legacy_action == legacy_action
            assert decision.legal is bool(allowed)
            assert decision.cache_reused
    finally:
        wrapper.close()


def test_environment_execution_preserves_legacy_result_and_adds_structured_source_contract() -> None:
    class FakeClient:
        def __init__(self, next_observation: dict) -> None:
            self.next_observation = next_observation
            self.requests: list[tuple[str, dict]] = []

        def request(self, command: str, **payload: object) -> dict:
            self.requests.append((command, dict(payload)))
            if command == "step":
                return {
                    "action": int(payload.get("action", 0)),
                    "illegalAction": False,
                    "illegalReason": None,
                    "plantPlaced": True,
                    "costPaid": True,
                    "cooldownStarted": True,
                    "observation": copy.deepcopy(self.next_observation),
                }
            if command == "restore_game_speed":
                return {"ok": True}
            return copy.deepcopy(self.next_observation)

        def close(self) -> None:
            return None

    wrapper = make_wrapper(identity=True, fusion_enabled=True)
    base = wrapper.base
    observation = load_observation_fixture()
    observation["actionCount"] = 701
    base.previous_observation = copy.deepcopy(observation)
    base.config.step_seconds = 0.0
    base.config.seed_screen_check_interval = 10_000
    base._steps_since_seed_screen_check = 0
    fake = FakeClient(observation)
    base.client = fake  # type: ignore[assignment]
    try:
        _next, _reward, _done, _truncated, info = base.step(
            10,
            action_source="gui",
            action_source_metadata={"manual_control": True},
        )
        action_result = info["action_result"]
        # Historical bridge/result keys remain available to every consumer.
        for key in (
            "action",
            "requestedAction",
            "executedAction",
            "illegalAction",
            "illegalReason",
            "plantPlaced",
            "costPaid",
            "cooldownStarted",
        ):
            assert key in action_result
        assert action_result["actionIntent"]["source"] == "gui"
        assert action_result["actionIntent"]["source_metadata"] == {"manual_control": True}
        assert action_result["actionDecision"]["source"] == "gui"
        assert action_result["actionDecision"]["legal"] is True
        structured = action_result["structuredActionResult"]
        assert structured["decision"]["source"] == "gui"
        assert structured["executed"] is True
        assert structured["bridge_accepted"] is True
        assert any(command == "step" and payload.get("action") == 10 for command, payload in fake.requests)
    finally:
        wrapper.close()


def test_terminal_and_timeout_results_keep_the_structured_action_contract() -> None:
    class TimeoutClient:
        def request(self, command: str, **_payload: object) -> dict:
            if command == "step":
                raise BridgeTimeoutError("step", 0.01, "contract timeout")
            if command == "restore_game_speed":
                return {"ok": True}
            return {}

        def close(self) -> None:
            return None

    wrapper = make_wrapper(identity=True, fusion_enabled=True)
    base = wrapper.base
    base.config.step_seconds = 0.0
    base.config.seed_screen_check_interval = 10_000
    base._steps_since_seed_screen_check = 0
    try:
        terminal = load_observation_fixture()
        terminal.update(
            {
                "actionCount": 701,
                "screenState": "game_over_restart_screen",
                "onGameOverScreen": True,
                "gameplayReady": False,
                "done": True,
            }
        )
        base.previous_observation = terminal
        _obs, _reward, done, _truncated, info = base.step(
            10,
            action_source="debug",
            action_source_metadata={"entrypoint": "test"},
        )
        assert done
        terminal_result = info["action_result"]
        assert terminal_result["actionIntent"]["source"] == "debug"
        assert terminal_result["actionDecision"]["rejection_reason"] == "restart_screen"
        assert terminal_result["structuredActionResult"]["executed"] is False
        assert terminal_result["structuredActionResult"]["bridge_accepted"] is None

        active = load_observation_fixture()
        active["actionCount"] = 701
        base.previous_observation = active
        base.client = TimeoutClient()  # type: ignore[assignment]
        _obs, _reward, done, _truncated, info = base.step(
            10,
            action_source="random",
            action_source_metadata={"baseline_policy": "random"},
        )
        assert done and info["done_reason"] == "env_corruption"
        timeout_result = info["action_result"]
        assert timeout_result["bridgeTimeout"] is True
        assert timeout_result["actionIntent"]["source"] == "random"
        assert timeout_result["structuredActionResult"]["executed"] is True
        assert timeout_result["structuredActionResult"]["bridge_accepted"] is None
    finally:
        wrapper.close()


def test_dynamic_coach_adapter_preserves_policy_and_legacy_action_identities() -> None:
    observation = load_observation_fixture()
    observation["actionCount"] = 701
    wrapper = _dynamic_wrapper()
    try:
        wrapper._last_observation = copy.deepcopy(observation)
        wrapper.base.previous_observation = copy.deepcopy(observation)
        action_mask = wrapper.action_masks()

        # Legacy bridge action 2 is dynamic policy action 1.  Passing policy 1
        # directly into the base cache would incorrectly validate bridge action
        # 1 and reject this command.
        plant = validate_coach_command_for_env(
            parse_coach_command("plant 0 0 1", source="gui"),
            wrapper,
            observation=observation,
            action_mask=action_mask,
        )
        assert plant.legal
        assert plant.policy_action == 1
        assert plant.decoded["slot_index"] == 0
        assert plant.decoded["row"] == 0
        assert plant.decoded["column"] == 1

        # Dynamic wait 700 must resolve to legacy bridge wait 0 rather than the
        # last legacy placement identity.
        wait = validate_coach_command_for_env(
            parse_coach_command("wait", source="stream"),
            wrapper,
            observation=observation,
            action_mask=action_mask,
        )
        assert wait.legal
        assert wait.policy_action == 700
        assert wait.decoded["kind"] == 0

        occupied_plant = validate_coach_command_for_env(
            parse_coach_command("plant 2 2 3", source="gui"),
            wrapper,
            observation=observation,
            action_mask=action_mask,
        )
        assert not occupied_plant.legal
        assert occupied_plant.rejected_reason == "occupied_cell"
    finally:
        wrapper.close()


def _recipe_observation(source_type: int, ingredient_type: int) -> dict:
    return {
        "rowCount": ROWS,
        "columnCount": COLS,
        "sun": 999,
        "seedSlots": [
            {
                "slotIndex": 0,
                "plantType": ingredient_type,
                "seedCost": 0,
                "ready": True,
                "disabled": False,
                "usable": True,
                "currentCooldown": 0.0,
                "fullCooldown": 7.5,
            }
        ],
        "plants": [{"row": 2, "column": 4, "type": source_type}],
        "visiblePlants": [],
    }


def test_all_recipe_self_recursive_and_runtime_only_compatibility_pairs() -> None:
    expected_recipes = {
        (0, 0): 1030,
        (1030, 0): 1090,
        (1090, 0): 1032,
        (1, 1): 1033,
    }
    assert {pair: int(rule["predicted_result_type"]) for pair, rule in FUSION_RULES.items()} == expected_recipes

    # The two non-recipe relationships are deliberate runtime-only bridge
    # compatibility entries.  Recipe pairs and these entries form the full
    # compatibility view; no accidental broad compatibility is permitted.
    expected_undirected_pairs = {
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 1030),
        (0, 1090),
        (1, 1),
    }
    compatibility_ids = {0, 1, 2, 3, 1030, 1032, 1033, 1090}
    for left, right in product(compatibility_ids, repeat=2):
        pair = tuple(sorted((left, right)))
        assert are_fusion_compatible(left, right) is (pair in expected_undirected_pairs)

    for (source_type, ingredient_type), result_type in expected_recipes.items():
        observation = _recipe_observation(source_type, ingredient_type)
        assert get_fusion_illegal_reason(observation, 2, 4, 0) == ""
        assert are_fusion_compatible(source_type, ingredient_type)
        assert int(FUSION_RULES[(source_type, ingredient_type)]["predicted_result_type"]) == result_type

    # Public compatibility wrapper remains symmetric during the migration.
    for source, partners in FUSION_COMPATIBILITY.items():
        for partner in partners:
            assert are_fusion_compatible(source, partner)
            assert are_fusion_compatible(partner, source)


def test_live_fusion_probe_empty_or_unrelated_board_requires_peashooter_setup() -> None:
    assert pvzrl_env._select_live_fusion_probe_candidates({}) == (None, None, 0)

    unrelated = {
        "fusionCandidates": [
            {
                "fusionLegal": True,
                "sourcePlantType": 1,
                "ingredientPlantType": 1,
            }
        ]
    }
    first_legal, pea_sun, count = pvzrl_env._select_live_fusion_probe_candidates(unrelated)
    assert first_legal is unrelated["fusionCandidates"][0]
    assert pea_sun is None
    assert count == 1

    pea_sun_candidate = {
        "fusion_candidates": [
            {
                "fusion_legal": True,
                "source_plant_type": 0,
                "ingredient_plant_type": 1,
            }
        ]
    }
    first_legal, pea_sun, count = pvzrl_env._select_live_fusion_probe_candidates(pea_sun_candidate)
    assert first_legal is pea_sun is pea_sun_candidate["fusion_candidates"][0]
    assert count == 1


def test_recipe_registry_is_immutable_directional_and_derives_legacy_views() -> None:
    assert isinstance(FUSION_RECIPES, tuple)
    assert tuple(recipe.pair for recipe in FUSION_RECIPES) == tuple(FUSION_RECIPES_BY_PAIR)
    assert {
        pair: recipe.result_plant_type
        for pair, recipe in FUSION_RECIPES_BY_PAIR.items()
    } == {
        (0, 0): 1030,
        (1030, 0): 1090,
        (1090, 0): 1032,
        (1, 1): 1033,
    }
    for pair, recipe in FUSION_RECIPES_BY_PAIR.items():
        assert fusion_recipe(*pair) is recipe
        assert fusion_compatibility_kind(*pair) == "recipe"
        legacy = FUSION_RULES[pair]
        assert legacy["predicted_result_type"] == recipe.result_plant_type
        assert legacy["predicted_result_name"] == recipe.result_plant_name
        assert legacy["scripted_enabled"] is recipe.scripted_enabled
        assert set(legacy) == {
            "predicted_result_name",
            "predicted_result_type",
            "reason",
            "role",
            "scripted_enabled",
        }

    runtime_pairs = {case.canonical_pair for case in RUNTIME_ONLY_FUSION_COMPATIBILITY_CASES}
    assert runtime_pairs == set(RUNTIME_ONLY_FUSION_COMPATIBILITY) == {(0, 1), (0, 2)}
    for first, second in runtime_pairs:
        assert fusion_compatibility_kind(first, second) == "runtime_only"
        assert fusion_compatibility_kind(second, first) == "runtime_only"
        assert fusion_recipe(first, second) is None
        assert fusion_recipe(second, first) is None

    wrong_known_result = fusion_intent_from_candidate(
        {
            **_candidate(),
            "predicted_result_type": 9999,
            "predicted_result_name": "Wrong",
        },
        source="debug",
    )
    assert wrong_known_result.predicted_result_type == 1030
    assert wrong_known_result.predicted_result_name == "DoubleShooer"

    runtime_only = fusion_intent_from_candidate(
        {
            **_candidate(),
            "source_plant_type": 1,
            "target_or_ingredient_type": 0,
            "predicted_result_type": 9999,
            "predicted_result_name": "Invented",
        },
        source="debug",
    )
    assert runtime_only.compatibility_kind == "runtime_only"
    assert runtime_only.predicted_result_type == -1
    assert runtime_only.predicted_result_name == ""

    with pytest.raises(TypeError):
        FUSION_RECIPES_BY_PAIR[(0, 1)] = FUSION_RECIPES[0]  # type: ignore[index]
    with pytest.raises(TypeError):
        FUSION_RULES[(0, 0)]["predicted_result_type"] = -1  # type: ignore[index]
    with pytest.raises((AttributeError, TypeError)):
        FUSION_RECIPES[0].result_plant_type = -1  # type: ignore[misc]


def test_fusion_structured_records_are_deeply_immutable() -> None:
    candidate = {
        **_candidate(),
        "nested_candidate": {"values": [1, 2]},
    }
    metadata = {"nested_metadata": {"enabled": True}}
    intent = fusion_intent_from_candidate(
        candidate,
        source="model",
        metadata=metadata,
    )
    decision = validate_fusion_intent(intent, _recipe_observation(0, 0))
    execution = fusion_execution_from_result(
        decision,
        {
            "fusionAttempted": True,
            "fusionSucceeded": True,
            "nested_result": {"changed": [False]},
        },
        event_id="immutable-event",
    )

    with pytest.raises(TypeError):
        intent.metadata["nested_candidate"]["values"][0] = 99  # type: ignore[index]
    with pytest.raises(TypeError):
        intent.metadata["nested_metadata"]["enabled"] = False  # type: ignore[index]
    with pytest.raises(TypeError):
        decision.bridge_command["source_row"] = 4  # type: ignore[index]
    with pytest.raises(TypeError):
        execution.result["nested_result"]["changed"][0] = True  # type: ignore[index]

    # Caller-owned input must also be isolated from the record after creation.
    candidate["nested_candidate"]["values"][0] = -1
    metadata["nested_metadata"]["enabled"] = False
    assert intent.metadata["nested_candidate"]["values"][0] == 1
    assert intent.metadata["nested_metadata"]["enabled"] is True


def test_fusion_intent_validation_is_source_independent_and_schema_stable() -> None:
    observation = _recipe_observation(0, 0)
    candidate = _candidate()
    sources = (
        "model_action_mask",
        "scripted",
        "human",
        "stream",
        "mock_stream",
        "gui",
        "manual",
        "debug",
    )
    decisions = []
    for source in sources:
        intent = fusion_intent_from_candidate(
            candidate,
            source=source,
            requested_action=124,
            executed_action=124,
            metadata={"origin": source},
        )
        decision = validate_fusion_intent(intent, observation)
        assert decision.legal
        assert decision.rejection_reason == ""
        assert decision.recipe is FUSION_RECIPES_BY_PAIR[(0, 0)]
        assert decision.compatibility_kind == "recipe"
        assert decision.bridge_command["source_row"] == 2
        assert decision.bridge_command["source_col"] == 4
        assert decision.bridge_command["ingredient_seed_slot_index"] == 0
        decisions.append(decision)

    # Aliases normalize source attribution, but cannot change validation or the
    # tile-scoped bridge command.
    parity = {
        (
            decision.legal,
            decision.rejection_reason,
            decision.compatibility_kind,
            tuple(sorted(decision.bridge_command.items())),
        )
        for decision in decisions
    }
    assert len(parity) == 1
    assert [decision.intent.source for decision in decisions] == [
        "model",
        "scripted",
        "human_coach",
        "stream_coach",
        "stream_coach",
        "gui",
        "manual",
        "debug",
    ]
    assert set(decisions[0].to_dict()) == {
        "source",
        "legal",
        "rejection_reason",
        "compatibility_kind",
        "bridge_command",
        "candidate",
    }
    assert set(decisions[0].to_dict()["bridge_command"]) == {
        "source_instance_id",
        "source_row",
        "source_col",
        "source_plant_type",
        "ingredient_seed_slot_index",
        "ingredient_plant_type",
        "predicted_result_type",
        "predicted_result_name",
    }


def test_all_recipes_and_runtime_only_cases_share_one_fusion_validator() -> None:
    for recipe in FUSION_RECIPES:
        observation = _recipe_observation(recipe.source_plant_type, recipe.ingredient_plant_type)
        intent = fusion_intent_from_candidate(
            {
                "source_plant_type": recipe.source_plant_type,
                "source_row": 2,
                "source_col": 4,
                "target_or_ingredient_type": recipe.ingredient_plant_type,
                "ingredient_seed_slot_index": 0,
            },
            source="model",
        )
        decision = validate_fusion_intent(intent, observation, require_known_recipe=True)
        assert decision.legal
        assert decision.recipe is recipe
        assert decision.bridge_command["predicted_result_type"] == recipe.result_plant_type

    for first, second in RUNTIME_ONLY_FUSION_COMPATIBILITY:
        observation = _recipe_observation(first, second)
        intent = fusion_intent_from_candidate(
            {
                "source_plant_type": first,
                "source_row": 2,
                "source_col": 4,
                "target_or_ingredient_type": second,
                "ingredient_seed_slot_index": 0,
            },
            source="human",
        )
        runtime_decision = validate_fusion_intent(intent, observation)
        assert runtime_decision.legal
        assert runtime_decision.compatibility_kind == "runtime_only"
        assert runtime_decision.recipe is None
        scripted_decision = validate_fusion_intent(
            intent,
            observation,
            require_known_recipe=True,
        )
        assert not scripted_decision.legal
        assert scripted_decision.rejection_reason == "incomplete_fusion_mapping"

    illegal_observation = _recipe_observation(3, 0)
    illegal_intent = fusion_intent_from_candidate(
        {
            "source_plant_type": 3,
            "source_row": 2,
            "source_col": 4,
            "target_or_ingredient_type": 0,
            "ingredient_seed_slot_index": 0,
        },
        source="gui",
    )
    illegal_decision = validate_fusion_intent(illegal_intent, illegal_observation)
    assert not illegal_decision.legal
    assert illegal_decision.rejection_reason == FUSION_ILLEGAL_INCOMPATIBLE


@pytest.mark.parametrize(
    ("success", "reason"),
    ((True, ""), (False, "bridge_rejected")),
)
def test_fusion_accounting_event_ids_make_attempt_outcome_and_reward_gate_exactly_once(
    success: bool,
    reason: str,
) -> None:
    observation = _recipe_observation(0, 0)
    decision = validate_fusion_intent(
        fusion_intent_from_candidate(_candidate(), source="model"),
        observation,
    )
    result = {
        "fusionAttempted": True,
        "fusionSucceeded": success,
        "fusionRejectedReason": reason,
        "illegalReason": reason or None,
        "fusionScope": "source_tile",
        "changedTileCount": 1 if success else 0,
    }
    execution = fusion_execution_from_result(
        decision,
        result,
        event_id="episode-4:step-9:fusion-1",
    )
    accounted: set[str] = set()
    diagnostics, applied = account_fusion_execution_once(
        default_fusion_diagnostics("observe"),
        execution,
        accounted,
    )
    assert applied
    assert diagnostics["fusion_attempted_count"] == 1
    assert diagnostics["fusion_success_count"] == int(success)
    assert diagnostics["fusion_failed_count"] == int(not success)
    assert diagnostics["fusion_last_event_id"] == execution.event_id
    assert diagnostics["fusion_last_source"] == "model"

    repeated, applied_again = account_fusion_execution_once(
        diagnostics,
        execution,
        accounted,
    )
    assert not applied_again
    assert repeated == diagnostics
    # The boolean is the reward gate: duplicate delivery must not be eligible
    # to invoke the shared reward calculation a second time.
    reward_application_count = int(applied) + int(applied_again)
    assert reward_application_count == 1


def test_pre_execution_fusion_rejection_is_counted_once_but_never_as_an_attempt() -> None:
    observation = _recipe_observation(0, 0)
    intent = fusion_intent_from_candidate(_candidate(), source="human")
    decision = validate_fusion_intent(
        intent,
        observation,
        precondition_rejection="startup_stale_command_blocked",
    )
    assert not decision.legal
    execution = fusion_execution_from_result(
        decision,
        None,
        event_id="coach-command-17",
        attempted=False,
    )
    diagnostics, applied = account_fusion_execution_once(
        default_fusion_diagnostics("observe"),
        execution,
        set(),
    )
    assert applied
    assert diagnostics["fusion_attempted_count"] == 0
    assert diagnostics["fusion_success_count"] == 0
    assert diagnostics["fusion_failed_count"] == 0
    assert diagnostics["fusion_rejected_count"] == 1
    assert diagnostics["fusion_rejected_reasons"] == {"startup_stale_command_blocked": 1}


@pytest.mark.parametrize(
    ("raw_source", "normalized_source"),
    (
        ("model_action_mask", "model"),
        ("scripted", "scripted"),
        ("human", "human_coach"),
        ("stream", "stream_coach"),
        ("mock_stream", "stream_coach"),
        ("gui", "gui"),
        ("manual", "manual"),
        ("debug", "debug"),
    ),
)
def test_environment_fusion_adapter_is_tile_scoped_source_attributed_and_exactly_once(
    raw_source: str,
    normalized_source: str,
) -> None:
    class FusionClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        def request(self, command: str, **payload: object) -> dict:
            self.calls.append((command, dict(payload)))
            if command != "fusion_step":
                return {"ok": True}
            row = int(payload["source_row"])
            column = int(payload["source_col"])
            return {
                "fusionAttempted": True,
                "fusionSucceeded": True,
                "fusion_success": True,
                "illegalAction": False,
                "illegalReason": None,
                "fusionRejectedReason": "",
                "fusionExecutionMode": "runtime_checkmix",
                "bridgeMethodUsed": "CheckMix",
                "bridgeResultReason": "success",
                "fusionScope": "source_tile",
                "changedTileCount": 1,
                "changedTiles": [{"row": row, "column": column}],
                "nonSourceTilesChanged": False,
                "globalFusionSideEffect": False,
                "sourceTileOccupiedBefore": True,
                "plantCountOnTileBefore": 1,
                "plantCountOnTileAfter": 1,
                "resultingPlantAfter": {
                    "row": row,
                    "column": column,
                    "plantType": 1030,
                    "plantTypeName": "DoubleShooer",
                },
            }

        def close(self) -> None:
            return None

    observation = _recipe_observation(0, 0)
    client = FusionClient()
    env = PvZGymEnv(
        PvZEnvConfig(
            plant_types=[0],
            step_seconds=0.0,
            fusion_policy="observe",
            fusion_action_mask_enabled=True,
        )
    )
    env.client = client  # type: ignore[assignment]
    intent = fusion_intent_from_candidate(
        _candidate(),
        source=raw_source,
        requested_action=124,
        executed_action=124,
    )
    try:
        result, diagnostics = env._execute_fusion_intent(
            observation,
            default_fusion_diagnostics("observe"),
            intent,
        )
        assert result is not None
        fusion_calls = [payload for command, payload in client.calls if command == "fusion_step"]
        assert len(fusion_calls) == 1
        assert fusion_calls[0]["source_row"] == 2
        assert fusion_calls[0]["source_col"] == 4
        assert fusion_calls[0]["ingredient_seed_slot_index"] == 0
        assert result["fusionIntentSource"] == normalized_source
        assert result["fusionIntent"]["source"] == normalized_source
        assert result["requestedSourceRow"] == 2
        assert result["requestedSourceCol"] == 4
        assert result["changedTileCount"] == 1
        assert result["nonSourceTilesChanged"] is False
        assert result["globalFusionSideEffect"] is False
        assert result["fusionAccountingApplied"] is True
        assert diagnostics["fusion_attempted_count"] == 1
        assert diagnostics["fusion_success_count"] == 1
        assert diagnostics["fusion_failed_count"] == 0
        assert diagnostics["fusion_last_source"] == normalized_source

        first_reward = env._compose_step_reward(
            observation, observation, result, previous_legal_actions=[]
        ).breakdown.component("fusion_reward")
        second_reward = env._compose_step_reward(
            observation, observation, result, previous_legal_actions=[]
        ).breakdown.component("fusion_reward")
        assert first_reward > 0.0
        assert second_reward == 0.0
        assert result["fusionRewardApplied"] is True
        assert result["fusionRewardDuplicateSuppressed"] is True
    finally:
        env.close()


def test_model_fusion_reuses_prebuilt_action_and_seed_slot_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FusionClient:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def request(self, command: str, **payload: object) -> dict:
            assert command == "fusion_step"
            self.calls.append(dict(payload))
            return {
                "fusionAttempted": True,
                "fusionSucceeded": True,
                "fusion_success": True,
                "illegalAction": False,
                "illegalReason": None,
                "fusionRejectedReason": "",
                "fusionExecutionMode": "runtime_checkmix",
                "bridgeMethodUsed": "CheckMix",
                "bridgeResultReason": "success",
                "fusionScope": "source_tile",
                "changedTileCount": 1,
                "changedTiles": [{"row": 2, "column": 4}],
                "nonSourceTilesChanged": False,
                "globalFusionSideEffect": False,
                "sourceTileOccupiedBefore": True,
                "plantCountOnTileBefore": 1,
                "plantCountOnTileAfter": 1,
                "resultingPlantAfter": {
                    "row": 2,
                    "column": 4,
                    "plantType": 1030,
                    "plantTypeName": "DoubleShooer",
                },
            }

        def close(self) -> None:
            return None

    observation = _recipe_observation(0, 0)
    action = 1 + 2 * 10 + 4
    client = FusionClient()
    env = PvZGymEnv(
        PvZEnvConfig(
            plant_types=[0],
            step_seconds=0.0,
            fusion_policy="observe",
            fusion_action_mask_enabled=True,
        )
    )
    env.client = client  # type: ignore[assignment]
    env._step_facts_cache.get(observation, env.config.plant_types)
    monkeypatch.setattr(
        pvzrl_env,
        "decode_action",
        lambda *_args, **_kwargs: pytest.fail("model fusion raw-decoded the action"),
    )
    monkeypatch.setattr(
        pvzrl_env,
        "seed_slots_from_observation",
        lambda *_args, **_kwargs: pytest.fail("model fusion rescanned raw seed slots"),
    )
    try:
        result, diagnostics = env._maybe_execute_model_fusion(
            observation,
            default_fusion_diagnostics("observe"),
            action,
            action,
        )
        assert result is not None and result["fusionSucceeded"] is True
        assert diagnostics["fusion_success_count"] == 1
        assert len(client.calls) == 1
        assert client.calls[0]["source_row"] == 2
        assert client.calls[0]["source_col"] == 4
        assert client.calls[0]["ingredient_seed_slot_index"] == 0
    finally:
        env.close()


def test_environment_fusion_adapter_preserves_all_recipe_results_and_runtime_authority() -> None:
    class ResolvingClient:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def request(self, command: str, **payload: object) -> dict:
            assert command == "fusion_step"
            request = dict(payload)
            self.calls.append(request)
            predicted = int(request.get("predicted_result_type", -1))
            result_type = predicted if predicted >= 0 else 2000 + len(self.calls)
            return {
                "fusionAttempted": True,
                "fusionSucceeded": True,
                "fusion_success": True,
                "illegalAction": False,
                "illegalReason": None,
                "fusionRejectedReason": "",
                "fusionExecutionMode": "runtime_checkmix",
                "bridgeMethodUsed": "CheckMix",
                "bridgeResultReason": "success",
                "fusionScope": "source_tile",
                "changedTileCount": 1,
                "changedTiles": [{"row": 2, "column": 4}],
                "nonSourceTilesChanged": False,
                "globalFusionSideEffect": False,
                "sourceTileOccupiedBefore": True,
                "plantCountOnTileBefore": 1,
                "plantCountOnTileAfter": 1,
                "resultingPlantAfter": {
                    "row": 2,
                    "column": 4,
                    "plantType": result_type,
                },
            }

        def close(self) -> None:
            return None

    cases = (
        (0, 0, 1030, "DoubleShooer", "recipe"),
        (1030, 0, 1090, "SplitPea", "recipe"),
        (1090, 0, 1032, "GatlingPea", "recipe"),
        (1, 1, 1033, "TwinFlower", "recipe"),
        (1, 0, -1, "", "runtime_only"),
        (0, 2, -1, "", "runtime_only"),
    )
    client = ResolvingClient()
    env = PvZGymEnv(
        PvZEnvConfig(
            plant_types=[0, 1, 2],
            step_seconds=0.0,
            fusion_policy="observe",
            fusion_action_mask_enabled=True,
        )
    )
    env.client = client  # type: ignore[assignment]
    try:
        for source_type, ingredient_type, expected_type, expected_name, kind in cases:
            observation = _recipe_observation(source_type, ingredient_type)
            intent = fusion_intent_from_candidate(
                {
                    **_candidate(),
                    "source_plant_type": source_type,
                    "target_or_ingredient_type": ingredient_type,
                    "predicted_result_type": 9999,
                    "predicted_result_name": "CallerGuess",
                },
                source="debug",
            )
            result, diagnostics = env._execute_fusion_intent(
                observation,
                default_fusion_diagnostics("observe"),
                intent,
            )
            assert result is not None and result["fusionSucceeded"] is True
            assert intent.compatibility_kind == kind
            assert intent.predicted_result_type == expected_type
            assert intent.predicted_result_name == expected_name
            assert int(client.calls[-1]["predicted_result_type"]) == expected_type
            assert str(client.calls[-1]["predicted_result_name"]) == expected_name
            assert diagnostics["fusion_attempted_count"] == 1
            assert diagnostics["fusion_success_count"] == 1
    finally:
        env.close()


def test_failed_environment_fusion_is_counted_and_rewarded_once_across_copied_results() -> None:
    class FailureClient:
        def request(self, command: str, **_payload: object) -> dict:
            assert command == "fusion_step"
            return {
                "fusionAttempted": True,
                "fusionSucceeded": False,
                "fusion_success": False,
                "illegalAction": True,
                "illegalReason": "bridge_rejected",
                "fusionRejectedReason": "bridge_rejected",
                "fusionExecutionMode": "runtime_checkmix",
                "bridgeMethodUsed": "CheckMix",
                "bridgeResultReason": "bridge_rejected",
                "fusionScope": "source_tile",
                "changedTileCount": 0,
                "changedTiles": [],
                "nonSourceTilesChanged": False,
                "globalFusionSideEffect": False,
                "sourceTileOccupiedBefore": True,
                "plantCountOnTileBefore": 1,
                "plantCountOnTileAfter": 1,
            }

        def close(self) -> None:
            return None

    observation = _recipe_observation(0, 0)
    env = PvZGymEnv(
        PvZEnvConfig(
            plant_types=[0],
            step_seconds=0.0,
            fusion_policy="observe",
            fusion_action_mask_enabled=True,
        )
    )
    env.client = FailureClient()  # type: ignore[assignment]
    try:
        intent = fusion_intent_from_candidate(_candidate(), source="model")
        result, diagnostics = env._execute_fusion_intent(
            observation,
            default_fusion_diagnostics("observe"),
            intent,
        )
        assert result is not None
        assert diagnostics["fusion_attempted_count"] == 1
        assert diagnostics["fusion_success_count"] == 0
        assert diagnostics["fusion_failed_count"] == 1
        assert diagnostics["fusion_rejected_count"] == 1

        first_reward = env._compose_step_reward(
            observation, observation, result, previous_legal_actions=[]
        ).breakdown.component("fusion_reward")
        copied_result = copy.deepcopy(result)
        second_reward = env._compose_step_reward(
            observation, observation, copied_result, previous_legal_actions=[]
        ).breakdown.component("fusion_reward")
        assert first_reward < 0.0
        assert second_reward == 0.0
        assert result["fusionRewardApplied"] is True
        assert copied_result["fusionRewardDuplicateSuppressed"] is True
    finally:
        env.close()


def test_fusion_illegal_reason_precedence_and_external_vocabulary_are_stable() -> None:
    observation = _recipe_observation(3, 0)
    observation["seedSlots"][0].update(
        {"usable": False, "ready": False, "currentCooldown": 7.5, "seedCost": 9999}
    )
    # Structural and compatibility reasons precede transient resource reasons.
    assert get_fusion_illegal_reason(observation, -1, 99, 99, fusion_enabled=False) == FUSION_ILLEGAL_DISABLED
    assert (
        get_fusion_illegal_reason(observation, -1, 99, 99, fusion_bridge_available=False)
        == FUSION_ILLEGAL_BRIDGE_UNAVAILABLE
    )
    assert get_fusion_illegal_reason(observation, -1, 4, 0) == FUSION_ILLEGAL_INVALID_ROW
    assert get_fusion_illegal_reason(observation, 2, 99, 0) == FUSION_ILLEGAL_INVALID_COL
    assert get_fusion_illegal_reason(observation, 2, 4, 99) == FUSION_ILLEGAL_INVALID_SEED_SLOT
    assert get_fusion_illegal_reason(observation, 0, 0, 0) == FUSION_ILLEGAL_EMPTY_TILE
    assert get_fusion_illegal_reason(observation, 2, 4, 0) == FUSION_ILLEGAL_INCOMPATIBLE

    compatible = _recipe_observation(0, 0)
    compatible["seedSlots"][0]["usable"] = False
    assert get_fusion_illegal_reason(compatible, 2, 4, 0) == FUSION_ILLEGAL_SEED_UNAVAILABLE
    compatible["seedSlots"][0].update({"usable": True, "ready": False})
    assert get_fusion_illegal_reason(compatible, 2, 4, 0) == FUSION_ILLEGAL_COOLDOWN
    compatible["seedSlots"][0].update({"ready": True, "seedCost": 1000})
    compatible["sun"] = 0
    assert get_fusion_illegal_reason(compatible, 2, 4, 0) == FUSION_ILLEGAL_INSUFFICIENT_SUN

    assert FUSION_REJECTION_REASONS == (
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


def _candidate() -> dict:
    return {
        "source_plant_name": "Peashooter",
        "source_plant_type": 0,
        "source_row": 2,
        "source_col": 4,
        "target_or_ingredient_name": "Peashooter",
        "target_or_ingredient_type": 0,
        "ingredient_seed_slot_index": 0,
        "predicted_result_name": "DoubleShooer",
        "predicted_result_type": 1030,
        "fusion_legal": True,
        "fusion_blocked_reason": "",
    }


@pytest.mark.parametrize(
    ("success", "reason"),
    ((True, ""), (False, "bridge_rejected")),
)
def test_one_fusion_event_produces_exactly_one_attempt_and_one_outcome(
    success: bool,
    reason: str,
) -> None:
    result = {
        "fusionAttempted": True,
        "fusionSucceeded": success,
        "fusion_success": success,
        "fusionRejectedReason": reason,
        "illegalReason": reason or None,
        "fusionExecutionMode": "runtime_checkmix",
        "bridgeMethodUsed": "CheckMix",
        "bridgeResultReason": reason,
        "fusionScope": "source_tile",
        "changedTileCount": 1 if success else 0,
        "nonSourceTilesChanged": False,
        "globalFusionSideEffect": False,
    }
    diagnostics = apply_fusion_attempt_result(
        default_fusion_diagnostics("observe"),
        _candidate(),
        result,
        rejected_reason=reason,
    )
    assert diagnostics["fusion_attempted_count"] == 1
    assert diagnostics["fusion_success_count"] == int(success)
    assert diagnostics["fusion_failed_count"] == int(not success)
    assert diagnostics["fusion_success_count"] + diagnostics["fusion_failed_count"] == 1
    assert sum(diagnostics["fusion_attempts_by_pair"].values()) == 1
    if success:
        assert diagnostics["fusion_rejected_count"] == 0
        assert diagnostics["fusion_rejected_reasons"] == {}
        assert sum(diagnostics["fusion_successes_by_pair"].values()) == 1
    else:
        assert diagnostics["fusion_rejected_count"] == 1
        assert diagnostics["fusion_rejected_reasons"] == {reason: 1}
        assert sum(diagnostics["fusion_failures_by_pair"].values()) == 1

    # These keys are consumed by live status, episode aggregation, and GUI
    # adapters and must survive the internal pipeline consolidation.
    assert set(diagnostics["fusion_last_result"]) == {
        "success",
        "illegalReason",
        "fusionRejectedReason",
        "fusionExecutionMode",
        "bridgeMethodUsed",
        "bridgeResultReason",
        "duplicateStackDetected",
        "fusionScope",
        "changedTileCount",
        "nonSourceTilesChanged",
        "globalFusionSideEffect",
        "plantCountOnTileBefore",
        "plantCountOnTileAfter",
        "sourceTileOccupiedBefore",
        "sourcePlantBefore",
        "resultingPlantAfter",
    }
