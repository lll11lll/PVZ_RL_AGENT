"""Canonical action resolution for Streamer Mode V1 viewer commands.

The resolver does not implement game legality.  It projects one structured
viewer command onto the permanent 701-action identity, asks the environment's
canonical ``action_decision`` callback for each relevant slot-cell action, and
requires the supplied current action mask to agree with those decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

from pvzrl_action_space import (
    ADVENTURE_IDENTITY_ACTION_COUNT,
    ADVENTURE_IDENTITY_MAX_SEED_SLOTS,
    adventure_identity_action_to_slot_cell,
)
from pvzrl_actions import (
    ACTION_KIND_FUSION,
    ACTION_KIND_PLACEMENT,
    ActionDecision,
    ActionIntent,
)
from pvzrl_fusion import fusion_recipe
from pvzrl_stream_commands import (
    QueuedViewerCommand,
    ViewerCommand,
    ViewerCommandKind,
    filter_safe_action_source_metadata,
    safe_action_source_metadata,
)


RESOLUTION_RESOLVED = "resolved"
RESOLUTION_CURRENTLY_ILLEGAL = "currently_illegal"
RESOLUTION_UNRESOLVABLE = "unresolvable"
RESOLUTION_STALE = "stale"


ActionDecisionProvider = Callable[[int], Optional[ActionDecision]]


@dataclass(frozen=True, slots=True)
class ViewerActionResolution:
    """A source-attributed canonical action decision for one viewer command."""

    legal: bool
    classification: str
    reason: str
    action_id: Optional[int]
    intent: Optional[ActionIntent]
    decision: Optional[ActionDecision]
    frame_identity: str
    considered_action_ids: Tuple[int, ...] = ()

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "legal": bool(self.legal),
            "classification": self.classification,
            "reason": self.reason,
            "action_id": self.action_id,
            "frame_identity": self.frame_identity,
            "considered_action_ids": list(self.considered_action_ids),
            "intent": self.intent.to_dict() if self.intent is not None else None,
            "decision": self.decision.to_dict() if self.decision is not None else None,
        }


@lru_cache(maxsize=64)
def _tile_action_ids(row: int, column: int) -> Tuple[int, ...]:
    target_row = int(row)
    target_column = int(column)
    actions = []
    for action_id in range(1, ADVENTURE_IDENTITY_ACTION_COUNT):
        decoded = adventure_identity_action_to_slot_cell(action_id)
        if int(decoded["row"]) == target_row and int(decoded["column"]) == target_column:
            actions.append(action_id)
    return tuple(actions)


def _selected_plant_type(decision: ActionDecision) -> int:
    selected = int(decision.selected_plant_type)
    if selected >= 0:
        return selected
    try:
        return int(decision.intent.decoded_action.get("plant_type", -1))
    except (TypeError, ValueError, OverflowError):
        return -1


def _resolution(
    *,
    classification: str,
    reason: str,
    considered: Sequence[int],
    frame_identity: str = "",
    decision: Optional[ActionDecision] = None,
    source: str = "twitch",
    source_metadata: Optional[Mapping[str, Any]] = None,
) -> ViewerActionResolution:
    if decision is None:
        return ViewerActionResolution(
            legal=False,
            classification=classification,
            reason=str(reason or classification),
            action_id=None,
            intent=None,
            decision=None,
            frame_identity=str(frame_identity or ""),
            considered_action_ids=tuple(int(value) for value in considered),
        )

    safe_metadata = filter_safe_action_source_metadata(source_metadata)
    intent = replace(
        decision.intent,
        source=str(source or "twitch"),
        source_metadata=safe_metadata,
    )
    sourced_decision = decision.for_intent(intent, cache_reused=True)
    return ViewerActionResolution(
        legal=True,
        classification=RESOLUTION_RESOLVED,
        reason="",
        action_id=int(sourced_decision.policy_action),
        intent=intent,
        decision=sourced_decision,
        frame_identity=str(sourced_decision.frame_identity),
        considered_action_ids=tuple(int(value) for value in considered),
    )


def _first_rejection_reason(decisions: Sequence[ActionDecision], fallback: str) -> str:
    for decision in decisions:
        if decision.rejection_reason:
            return str(decision.rejection_reason)
    return str(fallback)


def resolve_viewer_action(
    command: ViewerCommand,
    *,
    action_mask: Sequence[bool],
    action_decision: ActionDecisionProvider,
    source: str = "twitch",
    source_metadata: Optional[Mapping[str, Any]] = None,
) -> ViewerActionResolution:
    """Resolve one command against one current mask/decision snapshot.

    The callback should close over the exact current observation and call
    ``PvZGymEnv.action_decision``.  The resolver intentionally performs no
    cooldown, sun, occupancy, lifecycle, or bridge legality checks itself.
    """

    if not isinstance(command, ViewerCommand):
        raise TypeError("command must be a ViewerCommand")
    try:
        mask = tuple(bool(value) for value in action_mask)
    except TypeError:
        return _resolution(
            classification=RESOLUTION_STALE,
            reason="action_mask_unavailable",
            considered=(),
        )
    if len(mask) != ADVENTURE_IDENTITY_ACTION_COUNT:
        return _resolution(
            classification=RESOLUTION_STALE,
            reason="action_mask_shape_mismatch",
            considered=(),
        )

    considered = _tile_action_ids(command.row, command.column)
    if len(considered) != ADVENTURE_IDENTITY_MAX_SEED_SLOTS:
        return _resolution(
            classification=RESOLUTION_STALE,
            reason="action_identity_geometry_mismatch",
            considered=considered,
        )

    decisions = []
    for action_id in considered:
        try:
            decision = action_decision(action_id)
        except Exception as exc:
            return _resolution(
                classification=RESOLUTION_STALE,
                reason=f"action_decision_error:{type(exc).__name__}",
                considered=considered,
            )
        if not isinstance(decision, ActionDecision):
            return _resolution(
                classification=RESOLUTION_STALE,
                reason="action_decision_unavailable",
                considered=considered,
            )
        if decision.policy_action != action_id or decision.bridge_action != action_id:
            return _resolution(
                classification=RESOLUTION_STALE,
                reason="action_identity_mismatch",
                considered=considered,
                frame_identity=decision.frame_identity,
            )
        decisions.append(decision)

    frame_identities = {str(decision.frame_identity) for decision in decisions if decision.frame_identity}
    if len(frame_identities) != 1:
        return _resolution(
            classification=RESOLUTION_STALE,
            reason="frame_changed_during_resolution" if frame_identities else "missing_frame_identity",
            considered=considered,
        )
    frame_identity = next(iter(frame_identities))
    config_fingerprints = {
        str(decision.config_fingerprint)
        for decision in decisions
        if decision.config_fingerprint
    }
    if len(config_fingerprints) != 1:
        return _resolution(
            classification=RESOLUTION_STALE,
            reason=(
                "config_changed_during_resolution"
                if config_fingerprints
                else "missing_config_fingerprint"
            ),
            considered=considered,
            frame_identity=frame_identity,
        )

    for action_id, decision in zip(considered, decisions):
        if bool(mask[action_id]) != bool(decision.legal):
            return _resolution(
                classification=RESOLUTION_STALE,
                reason="mask_decision_mismatch",
                considered=considered,
                frame_identity=frame_identity,
            )

    matching: list[ActionDecision] = []
    legal_matching: list[ActionDecision] = []

    if command.kind is ViewerCommandKind.PLANT:
        target_type = int(command.plant_type_id if command.plant_type_id is not None else -1)
        matching = [decision for decision in decisions if _selected_plant_type(decision) == target_type]
        if not matching:
            return _resolution(
                classification=RESOLUTION_UNRESOLVABLE,
                reason="plant_not_in_current_loadout",
                considered=considered,
                frame_identity=frame_identity,
            )
        legal_matching = [
            decision
            for decision in matching
            if decision.legal
            and mask[decision.policy_action]
            and decision.resolved_action_kind == ACTION_KIND_PLACEMENT
        ]
        if not legal_matching:
            wrong_kind = any(
                decision.legal and decision.resolved_action_kind == ACTION_KIND_FUSION
                for decision in matching
            )
            return _resolution(
                classification=RESOLUTION_CURRENTLY_ILLEGAL,
                reason=(
                    "plant_command_requires_empty_tile"
                    if wrong_kind
                    else _first_rejection_reason(matching, "plant_not_currently_legal")
                ),
                considered=considered,
                frame_identity=frame_identity,
            )

    elif command.kind is ViewerCommandKind.SLOT:
        target_slot = int(command.seed_slot if command.seed_slot is not None else -1)
        matching = [
            decision
            for decision in decisions
            if int(decision.intent.seed_slot) == target_slot
        ]
        if len(matching) != 1:
            return _resolution(
                classification=RESOLUTION_UNRESOLVABLE,
                reason="slot_action_identity_unavailable",
                considered=considered,
                frame_identity=frame_identity,
            )
        legal_matching = [
            decision
            for decision in matching
            if decision.legal
            and mask[decision.policy_action]
            and decision.resolved_action_kind == ACTION_KIND_PLACEMENT
        ]
        if not legal_matching:
            wrong_kind = bool(
                matching[0].legal and matching[0].resolved_action_kind == ACTION_KIND_FUSION
            )
            return _resolution(
                classification=RESOLUTION_CURRENTLY_ILLEGAL,
                reason=(
                    "slot_command_requires_empty_tile"
                    if wrong_kind
                    else _first_rejection_reason(matching, "slot_not_currently_legal")
                ),
                considered=considered,
                frame_identity=frame_identity,
            )

    elif command.kind in {ViewerCommandKind.FUSE_RESULT, ViewerCommandKind.FUSE_TILE}:
        fusion_decisions = [
            decision
            for decision in decisions
            if decision.resolved_action_kind == ACTION_KIND_FUSION
        ]
        if command.kind is ViewerCommandKind.FUSE_RESULT:
            target_result = int(
                command.fusion_result_type_id
                if command.fusion_result_type_id is not None
                else -1
            )
            for decision in fusion_decisions:
                recipe = fusion_recipe(
                    decision.existing_plant_type,
                    decision.selected_plant_type,
                )
                if recipe is not None and int(recipe.result_plant_type) == target_result:
                    matching.append(decision)
        else:
            matching = fusion_decisions

        legal_matching = [
            decision
            for decision in matching
            if decision.legal
            and mask[decision.policy_action]
            and decision.resolved_action_kind == ACTION_KIND_FUSION
        ]
        if not legal_matching:
            fallback = (
                "fusion_result_not_currently_legal"
                if command.kind is ViewerCommandKind.FUSE_RESULT
                else "no_currently_legal_fusion_at_tile"
            )
            return _resolution(
                classification=RESOLUTION_CURRENTLY_ILLEGAL,
                reason=_first_rejection_reason(matching, fallback),
                considered=considered,
                frame_identity=frame_identity,
            )
    else:  # defensive for data constructed outside the strict parser.
        return _resolution(
            classification=RESOLUTION_UNRESOLVABLE,
            reason="unsupported_viewer_command_kind",
            considered=considered,
            frame_identity=frame_identity,
        )

    # Candidate actions were generated in permanent action-ID order and every
    # filter above preserves that order, including duplicate identity slots.
    selected = legal_matching[0]
    return _resolution(
        classification=RESOLUTION_RESOLVED,
        reason="",
        considered=considered,
        frame_identity=frame_identity,
        decision=selected,
        source=source,
        source_metadata=source_metadata,
    )


def resolve_queued_viewer_action(
    queued: QueuedViewerCommand,
    *,
    action_mask: Sequence[bool],
    action_decision: ActionDecisionProvider,
    source: str = "twitch",
) -> ViewerActionResolution:
    return resolve_viewer_action(
        queued.command,
        action_mask=action_mask,
        action_decision=action_decision,
        source=source,
        source_metadata=safe_action_source_metadata(queued),
    )


class ViewerActionResolver:
    """Small callable wrapper for controller integration."""

    def __init__(
        self,
        *,
        action_mask: Callable[[], Sequence[bool]],
        action_decision: ActionDecisionProvider,
        source: str = "twitch",
        source_metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._action_mask = action_mask
        self._action_decision = action_decision
        self.source = str(source or "twitch")
        self.source_metadata = filter_safe_action_source_metadata(source_metadata)

    def resolve(self, command: ViewerCommand) -> ViewerActionResolution:
        return resolve_viewer_action(
            command,
            action_mask=self._action_mask(),
            action_decision=self._action_decision,
            source=self.source,
            source_metadata=self.source_metadata,
        )

    def __call__(self, command: ViewerCommand) -> ViewerActionResolution:
        return self.resolve(command)


__all__ = [
    "RESOLUTION_CURRENTLY_ILLEGAL",
    "RESOLUTION_RESOLVED",
    "RESOLUTION_STALE",
    "RESOLUTION_UNRESOLVABLE",
    "ActionDecisionProvider",
    "ViewerActionResolution",
    "ViewerActionResolver",
    "resolve_queued_viewer_action",
    "resolve_viewer_action",
]
