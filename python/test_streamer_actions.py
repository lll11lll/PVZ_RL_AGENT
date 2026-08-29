from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from typing import Dict, Mapping, Optional

import pytest

from pvzrl_action_space import (
    ADVENTURE_IDENTITY_ACTION_COUNT,
    CELLS_PER_SLOT,
    adventure_identity_action_to_slot_cell,
)
from pvzrl_actions import (
    ACTION_KIND_FUSION,
    ACTION_KIND_PLACEMENT,
    ActionDecision,
    ActionIntent,
)
from pvzrl_stream_actions import (
    LEGAL,
    PERMANENTLY_INVALID,
    RESOLUTION_CURRENTLY_ILLEGAL,
    RESOLUTION_RESOLVED,
    RESOLUTION_STALE,
    RESOLUTION_UNRESOLVABLE,
    TEMPORARILY_BLOCKED,
    ViewerActionResolver,
    resolve_queued_viewer_action,
    resolve_viewer_action,
)
from pvzrl_stream_commands import BoundedViewerCommandQueue, parse_viewer_command


VIEWER_HASH = "c" * 64


def policy_action(slot: int, row: int, column: int) -> int:
    return 1 + int(slot) * CELLS_PER_SLOT + int(row) * 10 + int(column)


def decision_for(
    action_id: int,
    *,
    selected_type: int = -1,
    existing_type: int = -1,
    legal: bool = False,
    resolved_kind: str = ACTION_KIND_PLACEMENT,
    reason: str = "slot_not_usable",
    frame: str = "revision:state",
    config: str = "config:one",
) -> ActionDecision:
    decoded = adventure_identity_action_to_slot_cell(action_id)
    decoded["plant_type"] = int(selected_type)
    intent = ActionIntent(
        source="mask",
        policy_action=action_id,
        bridge_action=action_id,
        action_kind=ACTION_KIND_PLACEMENT,
        seed_slot=decoded["slot_index"],
        row=decoded["row"],
        column=decoded["column"],
        decoded_action=decoded,
        source_metadata={"raw_user": "must-be-replaced", "raw_text": "must-be-replaced"},
    )
    return ActionDecision(
        intent=intent,
        legal=legal,
        rejection_reason="" if legal else reason,
        frame_identity=frame,
        config_fingerprint=config,
        resolved_action_kind=resolved_kind,
        selected_plant_type=selected_type,
        existing_plant_type=existing_type,
        bridge_authoritative=True,
    )


class DecisionSnapshot:
    def __init__(self, *, row: int, column: int, slot_types: Optional[Mapping[int, int]] = None) -> None:
        self.row = row
        self.column = column
        self.slot_types = dict(slot_types or {})
        self.overrides: Dict[int, ActionDecision] = {}
        self.mask = [False] * ADVENTURE_IDENTITY_ACTION_COUNT

    def set(
        self,
        slot: int,
        *,
        selected_type: Optional[int] = None,
        existing_type: int = -1,
        legal: bool,
        resolved_kind: str = ACTION_KIND_PLACEMENT,
        reason: str = "slot_not_usable",
        frame: str = "revision:state",
        config: str = "config:one",
    ) -> int:
        action_id = policy_action(slot, self.row, self.column)
        selected = self.slot_types.get(slot, -1) if selected_type is None else selected_type
        self.overrides[action_id] = decision_for(
            action_id,
            selected_type=selected,
            existing_type=existing_type,
            legal=legal,
            resolved_kind=resolved_kind,
            reason=reason,
            frame=frame,
            config=config,
        )
        self.mask[action_id] = bool(legal)
        return action_id

    def __call__(self, action_id: int) -> ActionDecision:
        if action_id in self.overrides:
            return self.overrides[action_id]
        decoded = adventure_identity_action_to_slot_cell(action_id)
        return decision_for(
            action_id,
            selected_type=self.slot_types.get(decoded["slot_index"], -1),
            legal=False,
        )


def test_plant_name_uses_first_legal_duplicate_slot_in_action_id_order() -> None:
    command = parse_viewer_command("!plant pea shooter 2 3")
    snapshot = DecisionSnapshot(row=1, column=2, slot_types={0: 0, 1: 0, 2: 1})
    first = snapshot.set(0, legal=True)
    snapshot.set(1, legal=True)
    metadata = {
        "command_id": "command-1",
        "event_id": "event-1",
        "viewer_hash": VIEWER_HASH,
        "phase_generation": 4,
        "command_kind": "plant",
        "viewer_command": command.to_safe_dict(),
        "raw_user": "secret-user",
        "raw_text": "!plant pea shooter 2 3",
    }

    resolved = resolve_viewer_action(
        command,
        action_mask=snapshot.mask,
        action_decision=snapshot,
        source_metadata=metadata,
    )

    assert resolved.legal
    assert resolved.legality == LEGAL
    assert resolved.classification == RESOLUTION_RESOLVED
    assert resolved.action_id == first
    assert resolved.intent is not None and resolved.intent.source == "twitch"
    assert resolved.decision is not None and resolved.decision.resolved_action_kind == ACTION_KIND_PLACEMENT
    assert resolved.frame_identity == "revision:state"
    assert len(resolved.considered_action_ids) == 14
    assert resolved.considered_action_ids == tuple(sorted(resolved.considered_action_ids))
    assert set(resolved.intent.source_metadata) == {
        "command_id",
        "event_id",
        "viewer_hash",
        "phase_generation",
        "command_kind",
        "viewer_command",
    }
    rendered = json.dumps(resolved.to_safe_dict(), sort_keys=True)
    assert "secret-user" not in rendered
    assert "!plant" not in rendered


def test_plant_name_skips_illegal_duplicate_and_selects_next_legal_identity() -> None:
    command = parse_viewer_command("!plant peashooter 1 1")
    snapshot = DecisionSnapshot(row=0, column=0, slot_types={0: 0, 1: 0})
    snapshot.set(0, legal=False, reason="cooldown_not_ready")
    second = snapshot.set(1, legal=True)

    resolved = resolve_viewer_action(command, action_mask=snapshot.mask, action_decision=snapshot)
    assert resolved.legal
    assert resolved.action_id == second


def test_named_plant_absent_from_loadout_is_unresolvable() -> None:
    command = parse_viewer_command("!plant cherry bomb 1 1")
    snapshot = DecisionSnapshot(row=0, column=0, slot_types={0: 0, 1: 1})

    resolved = resolve_viewer_action(command, action_mask=snapshot.mask, action_decision=snapshot)
    assert not resolved.legal
    assert resolved.classification == RESOLUTION_UNRESOLVABLE
    assert resolved.reason == "plant_not_in_current_loadout"


def test_slot_command_resolves_exact_identity_and_can_select_a_fusion_ingredient() -> None:
    command = parse_viewer_command("!slot 14 5 10")
    snapshot = DecisionSnapshot(row=4, column=9, slot_types={13: 2})
    exact = snapshot.set(13, legal=True)
    resolved = resolve_viewer_action(command, action_mask=snapshot.mask, action_decision=snapshot)
    assert resolved.legal
    assert resolved.action_id == exact
    assert resolved.intent is not None and resolved.intent.seed_slot == 13

    snapshot.set(
        13,
        legal=True,
        existing_type=0,
        resolved_kind=ACTION_KIND_FUSION,
    )
    fusion_instead = resolve_viewer_action(command, action_mask=snapshot.mask, action_decision=snapshot)
    assert fusion_instead.legal
    assert fusion_instead.classification == RESOLUTION_RESOLVED
    assert fusion_instead.action_id == exact
    assert fusion_instead.decision is not None
    assert fusion_instead.decision.resolved_action_kind == ACTION_KIND_FUSION


def test_plant_command_never_turns_into_fusion() -> None:
    command = parse_viewer_command("!plant pea 1 1")
    snapshot = DecisionSnapshot(row=0, column=0, slot_types={0: 0})
    snapshot.set(0, legal=True, existing_type=0, resolved_kind=ACTION_KIND_FUSION)
    resolved = resolve_viewer_action(command, action_mask=snapshot.mask, action_decision=snapshot)
    assert not resolved.legal
    assert resolved.reason == "plant_command_requires_empty_tile"


def test_fusion_result_uses_recipe_metadata_and_ignores_other_legal_results() -> None:
    command = parse_viewer_command("!fuse twin sunflower 3 4")
    snapshot = DecisionSnapshot(row=2, column=3, slot_types={0: 0, 1: 1})
    pea_fusion = snapshot.set(
        0,
        legal=True,
        existing_type=0,
        resolved_kind=ACTION_KIND_FUSION,
    )
    twin_fusion = snapshot.set(
        1,
        legal=True,
        existing_type=1,
        resolved_kind=ACTION_KIND_FUSION,
    )

    resolved = resolve_viewer_action(command, action_mask=snapshot.mask, action_decision=snapshot)
    assert resolved.legal
    assert resolved.action_id == twin_fusion
    assert resolved.action_id != pea_fusion

    repeater_alias = parse_viewer_command("!fuse repeater 3 4")
    pea_resolved = resolve_viewer_action(
        repeater_alias,
        action_mask=snapshot.mask,
        action_decision=snapshot,
    )
    assert pea_resolved.legal
    assert pea_resolved.action_id == pea_fusion


def test_tile_fusion_uses_first_legal_action_including_runtime_only_pair() -> None:
    command = parse_viewer_command("!fuse 4 5")
    snapshot = DecisionSnapshot(row=3, column=4, slot_types={0: 0, 1: 1})
    runtime_only = snapshot.set(
        0,
        legal=True,
        existing_type=1,
        resolved_kind=ACTION_KIND_FUSION,
    )
    snapshot.set(
        1,
        legal=True,
        existing_type=1,
        resolved_kind=ACTION_KIND_FUSION,
    )

    resolved = resolve_viewer_action(command, action_mask=snapshot.mask, action_decision=snapshot)
    assert resolved.legal
    assert resolved.action_id == runtime_only


def test_fusion_requires_a_currently_legal_canonical_fusion_action() -> None:
    target = parse_viewer_command("!fuse double shooter 1 1")
    snapshot = DecisionSnapshot(row=0, column=0, slot_types={0: 0})
    snapshot.set(
        0,
        legal=False,
        existing_type=0,
        resolved_kind=ACTION_KIND_FUSION,
        reason="cooldown_not_ready",
    )
    blocked = resolve_viewer_action(target, action_mask=snapshot.mask, action_decision=snapshot)
    assert not blocked.legal
    assert blocked.classification == RESOLUTION_CURRENTLY_ILLEGAL
    assert blocked.reason == "cooldown_not_ready"
    assert blocked.legality == TEMPORARILY_BLOCKED

    empty_snapshot = DecisionSnapshot(row=0, column=0, slot_types={0: 0})
    tile = resolve_viewer_action(
        parse_viewer_command("!fuse 1 1"),
        action_mask=empty_snapshot.mask,
        action_decision=empty_snapshot,
    )
    assert not tile.legal
    assert tile.classification == RESOLUTION_CURRENTLY_ILLEGAL
    assert tile.reason == "no_currently_legal_fusion_at_tile"


def test_mask_decision_disagreement_is_stale_and_never_selected() -> None:
    command = parse_viewer_command("!slot 1 1 1")
    snapshot = DecisionSnapshot(row=0, column=0, slot_types={0: 0})
    action_id = snapshot.set(0, legal=True)
    snapshot.mask[action_id] = False

    resolved = resolve_viewer_action(command, action_mask=snapshot.mask, action_decision=snapshot)
    assert not resolved.legal
    assert resolved.classification == RESOLUTION_STALE
    assert resolved.reason == "mask_decision_mismatch"
    assert resolved.action_id is None


def test_frame_change_during_scan_and_bad_mask_are_stale() -> None:
    command = parse_viewer_command("!slot 1 1 1")
    snapshot = DecisionSnapshot(row=0, column=0, slot_types={0: 0, 1: 1})
    snapshot.set(0, legal=False, frame="frame:one")
    snapshot.set(1, legal=False, frame="frame:two")
    changed = resolve_viewer_action(command, action_mask=snapshot.mask, action_decision=snapshot)
    assert not changed.legal
    assert changed.classification == RESOLUTION_STALE
    assert changed.reason == "frame_changed_during_resolution"

    config_snapshot = DecisionSnapshot(row=0, column=0, slot_types={0: 0})
    config_snapshot.set(0, legal=False, config="config:two")
    config_changed = resolve_viewer_action(
        command,
        action_mask=config_snapshot.mask,
        action_decision=config_snapshot,
    )
    assert config_changed.classification == RESOLUTION_STALE
    assert config_changed.reason == "config_changed_during_resolution"

    bad_mask = resolve_viewer_action(command, action_mask=[True], action_decision=snapshot)
    assert bad_mask.classification == RESOLUTION_STALE
    assert bad_mask.reason == "action_mask_shape_mismatch"

    def broken(_action_id: int) -> ActionDecision:
        raise RuntimeError("bridge state changed")

    error = resolve_viewer_action(
        command,
        action_mask=[False] * ADVENTURE_IDENTITY_ACTION_COUNT,
        action_decision=broken,
    )
    assert error.classification == RESOLUTION_STALE
    assert error.reason == "action_decision_error:RuntimeError"
    assert "bridge state changed" not in json.dumps(error.to_safe_dict())


def test_queued_resolution_retains_safe_demo_metadata_and_is_immutable() -> None:
    command = parse_viewer_command("!slot 1 1 1")
    queue = BoundedViewerCommandQueue(capacity=2, ttl_seconds=10.0, dedupe_capacity=4)
    accepted = queue.enqueue(
        command,
        message_id="twitch-chat-event-1",
        viewer_hash=VIEWER_HASH,
    )
    queued = queue.peek()
    assert queued is not None and accepted.command_id == queued.command_id

    snapshot = DecisionSnapshot(row=0, column=0, slot_types={0: 0})
    action_id = snapshot.set(0, legal=True)
    resolved = resolve_queued_viewer_action(
        queued,
        action_mask=snapshot.mask,
        action_decision=snapshot,
    )
    assert resolved.legal and resolved.action_id == action_id
    assert resolved.intent is not None
    assert resolved.intent.source_metadata["event_id"] == "twitch-chat-event-1"
    assert resolved.intent.source_metadata["viewer_hash"] == VIEWER_HASH
    assert resolved.intent.source_metadata["viewer_command"]["kind"] == "slot"
    assert resolved.intent.source_metadata["viewer_command"]["seed_slot"] == 0
    with pytest.raises(FrozenInstanceError):
        resolved.action_id = 1  # type: ignore[misc]


def test_callable_resolver_reads_a_fresh_mask_for_each_command() -> None:
    command = parse_viewer_command("!slot 1 1 1")
    snapshot = DecisionSnapshot(row=0, column=0, slot_types={0: 0})
    action_id = snapshot.set(0, legal=True)
    calls = 0

    def current_mask() -> list[bool]:
        nonlocal calls
        calls += 1
        return list(snapshot.mask)

    resolver = ViewerActionResolver(action_mask=current_mask, action_decision=snapshot)
    first = resolver(command)
    second = resolver.resolve(command)
    assert calls == 2
    assert first.action_id == second.action_id == action_id
