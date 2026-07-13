"""Pure Adventure Generalist action decoding and validation.

The bridge remains the final runtime authority.  This module owns the Python
decision that precedes a bridge call: normalize an action source into an
``ActionIntent``, validate that intent against one immutable observation
context, and expose the result as an ``ActionDecision``.  Complete masks are a
tuple of those same decisions, so diagnostics and execution safeguards do not
need independent legality implementations.

Adventure Generalist has one permanent action identity: action ``0`` waits and
actions ``1..700`` are fourteen seed-slot-major 5x10 placement/fusion blocks.
``policy_action`` and ``bridge_action`` therefore carry the same integer; no
alternate decoder identity or wait-position adapter remains.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Dict, FrozenSet, Iterable, Optional, Sequence, Tuple

from pvzrl_action_space import (
    ACTION_SPACE_ADVENTURE_14_IDENTITY,
    ADVENTURE_IDENTITY_MAX_SEED_SLOTS,
    ActionSpaceSpec,
    build_action_space_spec,
    decode_policy_action,
)
from pvzrl_observation_facts import (
    SeedSlotFact as SeedSlotFacts,
    StepFacts,
    build_step_facts,
    stable_digest,
)


ACTION_KIND_WAIT = "wait"
ACTION_KIND_PLACEMENT = "placement"
ACTION_KIND_FUSION = "fusion"
ACTION_KIND_INVALID = "invalid"


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_value(item) for item in value)
    return value


def _freeze_mapping(value: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        return MappingProxyType({})
    return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})


def _thaw_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_thaw_value(item) for item in sorted(value, key=repr)]
    return value


@dataclass(frozen=True)
class ActionIntent:
    """One normalized action request, independent of its originating source."""

    source: str
    policy_action: int
    bridge_action: int
    action_kind: str
    seed_slot: int
    row: int
    column: int
    decoded_action: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    bridge_command: Optional[Mapping[str, Any]] = None
    source_metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", str(self.source or "unknown"))
        object.__setattr__(self, "policy_action", int(self.policy_action))
        object.__setattr__(self, "bridge_action", int(self.bridge_action))
        object.__setattr__(self, "action_kind", str(self.action_kind or ACTION_KIND_INVALID))
        object.__setattr__(self, "seed_slot", int(self.seed_slot))
        object.__setattr__(self, "row", int(self.row))
        object.__setattr__(self, "column", int(self.column))
        object.__setattr__(self, "decoded_action", _freeze_mapping(self.decoded_action))
        object.__setattr__(
            self,
            "bridge_command",
            _freeze_mapping(self.bridge_command) if isinstance(self.bridge_command, Mapping) else None,
        )
        object.__setattr__(self, "source_metadata", _freeze_mapping(self.source_metadata))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "policy_action": self.policy_action,
            "bridge_action": self.bridge_action,
            "action_kind": self.action_kind,
            "seed_slot": self.seed_slot,
            "row": self.row,
            "column": self.column,
            "decoded_action": _thaw_value(self.decoded_action),
            "bridge_command": _thaw_value(self.bridge_command) if self.bridge_command is not None else None,
            "source_metadata": _thaw_value(self.source_metadata),
        }


@dataclass(frozen=True)
class ObservationFrameIdentity:
    """Proof that all Python action-decision inputs are unchanged."""

    revision: str
    state_digest: str

    @property
    def token(self) -> str:
        return f"{self.revision}:{self.state_digest}"

    def to_dict(self) -> Dict[str, str]:
        return {
            "revision": self.revision,
            "state_digest": self.state_digest,
            "token": self.token,
        }


@dataclass(frozen=True)
class ActionValidationConfig:
    """Immutable configuration inputs that can affect Python action legality."""

    action_space_mode: str = ACTION_SPACE_ADVENTURE_14_IDENTITY
    plant_types: Tuple[int, ...] = ()
    max_seed_slots: int = ADVENTURE_IDENTITY_MAX_SEED_SLOTS
    rows: int = 5
    cols: int = 10
    fusion_action_mask_enabled: bool = False
    tactical_masks: bool = False
    wallnut_tactical_mask: bool = False
    cherrybomb_tactical_mask: bool = False
    fusion_compatible_pairs: FrozenSet[Tuple[int, int]] = frozenset()
    compatibility_version: str = ""
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "plant_types", tuple(int(value) for value in self.plant_types))
        spec = build_action_space_spec(
            mode=self.action_space_mode,
            plant_types=list(self.plant_types),
            max_seed_slots=int(self.max_seed_slots),
            rows=int(self.rows),
            cols=int(self.cols),
        )
        object.__setattr__(self, "action_space_mode", spec.mode)
        object.__setattr__(self, "max_seed_slots", spec.max_seed_slots)
        object.__setattr__(self, "rows", spec.rows)
        object.__setattr__(self, "cols", spec.cols)
        object.__setattr__(
            self,
            "fusion_compatible_pairs",
            frozenset((int(existing), int(selected)) for existing, selected in self.fusion_compatible_pairs),
        )
        object.__setattr__(
            self,
            "fingerprint",
            stable_digest(
                {
                    "action_space_mode": self.action_space_mode,
                    "plant_types": self.plant_types,
                    "max_seed_slots": self.max_seed_slots,
                    "rows": self.rows,
                    "cols": self.cols,
                    "fusion_action_mask_enabled": self.fusion_action_mask_enabled,
                    "tactical_masks": self.tactical_masks,
                    "wallnut_tactical_mask": self.wallnut_tactical_mask,
                    "cherrybomb_tactical_mask": self.cherrybomb_tactical_mask,
                    "fusion_compatible_pairs": sorted(self.fusion_compatible_pairs),
                    "compatibility_version": self.compatibility_version,
                }
            ),
        )

    @property
    def spec(self) -> ActionSpaceSpec:
        return build_action_space_spec(
            mode=self.action_space_mode,
            plant_types=list(self.plant_types),
            max_seed_slots=self.max_seed_slots,
            rows=self.rows,
            cols=self.cols,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_space_mode": self.action_space_mode,
            "plant_types": list(self.plant_types),
            "max_seed_slots": self.max_seed_slots,
            "rows": self.rows,
            "cols": self.cols,
            "fusion_action_mask_enabled": bool(self.fusion_action_mask_enabled),
            "tactical_masks": bool(self.tactical_masks),
            "wallnut_tactical_mask": bool(self.wallnut_tactical_mask),
            "cherrybomb_tactical_mask": bool(self.cherrybomb_tactical_mask),
            "fusion_compatible_pairs": [list(pair) for pair in sorted(self.fusion_compatible_pairs)],
            "compatibility_version": self.compatibility_version,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class ActionValidationContext:
    """Immutable action-relevant facts for exactly one observation frame."""

    frame_identity: ObservationFrameIdentity
    facts: StepFacts
    config: ActionValidationConfig
    config_fingerprint: str
    action_count: int
    rows: int
    cols: int
    sun: int
    gameplay_ready: bool
    board_found: bool
    can_read_board: bool
    seed_selection_active: bool
    restart_screen: bool
    seed_slots: Tuple[SeedSlotFacts, ...]
    occupancy: Tuple[Tuple[int, int, int], ...]
    occupancy_by_cell: Mapping[Tuple[int, int], int]
    bridge_legal_actions: FrozenSet[int]
    tactical_rejections: Tuple[Tuple[int, str], ...] = ()
    tactical_rejection_by_action: Mapping[int, str] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "occupancy_by_cell", MappingProxyType(dict(self.occupancy_by_cell)))
        object.__setattr__(
            self,
            "tactical_rejection_by_action",
            MappingProxyType({int(action): str(reason) for action, reason in self.tactical_rejection_by_action.items()}),
        )

    @property
    def cache_key(self) -> "ActionCacheKey":
        return ActionCacheKey(
            frame_identity=self.frame_identity.token,
            config_fingerprint=self.config_fingerprint,
        )

@dataclass(frozen=True)
class ActionDecision:
    """The authoritative Python decision for an ``ActionIntent``."""

    intent: ActionIntent
    legal: bool
    rejection_reason: str
    frame_identity: str
    config_fingerprint: str
    resolved_action_kind: str
    selected_plant_type: int = -1
    existing_plant_type: int = -1
    bridge_authoritative: bool = True
    cache_reused: bool = False

    @property
    def policy_action(self) -> int:
        return self.intent.policy_action

    @property
    def bridge_action(self) -> int:
        return self.intent.bridge_action

    @property
    def source(self) -> str:
        return self.intent.source

    def for_intent(self, intent: ActionIntent, *, cache_reused: bool = True) -> "ActionDecision":
        return replace(self, intent=intent, cache_reused=bool(cache_reused))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent.to_dict(),
            "source": self.source,
            "policy_action": self.policy_action,
            "bridge_action": self.bridge_action,
            "action_kind": self.resolved_action_kind,
            "legal": bool(self.legal),
            "rejection_reason": self.rejection_reason,
            "frame_identity": self.frame_identity,
            "config_fingerprint": self.config_fingerprint,
            "selected_plant_type": self.selected_plant_type,
            "existing_plant_type": self.existing_plant_type,
            "bridge_authoritative": bool(self.bridge_authoritative),
            "cache_reused": bool(self.cache_reused),
        }


@dataclass(frozen=True)
class ActionResult:
    """A structured Python decision paired with the bridge execution result."""

    decision: ActionDecision
    execution_result: Mapping[str, Any]
    executed: bool
    bridge_accepted: Optional[bool]

    def __post_init__(self) -> None:
        object.__setattr__(self, "execution_result", _freeze_mapping(self.execution_result))

    @classmethod
    def from_execution(
        cls,
        decision: ActionDecision,
        execution_result: Optional[Mapping[str, Any]],
        *,
        executed: bool = True,
    ) -> "ActionResult":
        payload = dict(execution_result) if isinstance(execution_result, Mapping) else {}
        bridge_accepted: Optional[bool]
        if not executed or not payload:
            bridge_accepted = None
        elif bool(payload.get("bridgeTimeout")):
            bridge_accepted = None
        else:
            bridge_accepted = not bool(payload.get("illegalAction", False))
        return cls(
            decision=decision,
            execution_result=payload,
            executed=bool(executed),
            bridge_accepted=bridge_accepted,
        )

    def to_dict(self, *, include_execution_result: bool = True) -> Dict[str, Any]:
        result = {
            "decision": self.decision.to_dict(),
            "executed": bool(self.executed),
            "bridge_accepted": self.bridge_accepted,
        }
        if include_execution_result:
            result["execution_result"] = _thaw_value(self.execution_result)
        return result


@dataclass(frozen=True)
class ActionCacheKey:
    frame_identity: str
    config_fingerprint: str

    @property
    def token(self) -> str:
        return f"{self.frame_identity}|{self.config_fingerprint}"

    def to_dict(self) -> Dict[str, str]:
        return {
            "frame_identity": self.frame_identity,
            "config_fingerprint": self.config_fingerprint,
            "token": self.token,
        }


@dataclass(frozen=True)
class ActionDecisionCache:
    """Complete validated decisions for one proven frame/configuration pair."""

    key: ActionCacheKey
    context: ActionValidationContext
    decisions: Tuple[ActionDecision, ...]
    mask: Tuple[bool, ...]

    def decision_for(
        self,
        action: int,
        *,
        intent: Optional[ActionIntent] = None,
        cache_reused: bool = True,
    ) -> Optional[ActionDecision]:
        action_id = int(action)
        if not 0 <= action_id < len(self.decisions):
            return None
        decision = self.decisions[action_id]
        if intent is None:
            return replace(decision, cache_reused=bool(cache_reused))
        if intent.bridge_action != action_id:
            return None
        return decision.for_intent(intent, cache_reused=cache_reused)

    def to_dict(self, *, include_decisions: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "key": self.key.to_dict(),
            "action_count": len(self.decisions),
            "legal_action_count": sum(1 for allowed in self.mask if allowed),
            "mask": [bool(value) for value in self.mask],
        }
        if include_decisions:
            payload["decisions"] = [decision.to_dict() for decision in self.decisions]
        return payload


def compatible_pairs_for_observation(
    observation: Mapping[str, Any],
    fallback_plant_types: Sequence[int],
    compatibility: Any,
    *,
    facts: Optional[StepFacts] = None,
) -> FrozenSet[Tuple[int, int]]:
    """Freeze compatibility outcomes relevant to this frame into config input."""

    if not callable(compatibility):
        return frozenset()
    snapshot = facts or build_step_facts(observation, fallback_plant_types)
    existing_types = snapshot.fusion_source_types
    selected_types = set(snapshot.seed_slots_by_type)
    return frozenset(
        (int(existing), int(selected))
        for existing in existing_types
        for selected in selected_types
        if bool(compatibility(existing, selected))
    )


def observation_frame_identity(
    observation: Mapping[str, Any],
    *,
    bridge_legal_actions: Iterable[int],
    restart_screen: bool = False,
    tactical_rejections: Optional[Mapping[int, str]] = None,
    facts: Optional[StepFacts] = None,
) -> ObservationFrameIdentity:
    """Hash every observation value plus non-observation legality inputs.

    Hashing the complete payload is deliberately conservative.  A changed
    diagnostic-only value may force a harmless recomputation, while no
    action-relevant mutation can silently reuse an old decision.
    """

    snapshot = facts or build_step_facts(observation)
    state_digest = stable_digest(
        {
            "observation_digest": snapshot.identity.content_digest,
            "bridge_legal_actions": sorted({int(action) for action in bridge_legal_actions}),
            "restart_screen": bool(restart_screen),
            # Direct callers may supply non-derived tactical decisions, so they
            # are part of the proof.  The environment deliberately builds its
            # preliminary key with no tactical mapping, checks for a hit, and
            # derives the mapping only on a miss from this same observation and
            # the separately fingerprinted validation configuration.
            "tactical_rejections": sorted(
                (int(action), str(reason))
                for action, reason in (tactical_rejections or {}).items()
            ),
        }
    )
    return ObservationFrameIdentity(revision=snapshot.identity.revision, state_digest=state_digest)


def build_action_validation_context(
    observation: Mapping[str, Any],
    *,
    config: ActionValidationConfig,
    bridge_legal_actions: Iterable[int],
    restart_screen: bool = False,
    tactical_rejections: Optional[Mapping[int, str]] = None,
    facts: Optional[StepFacts] = None,
) -> ActionValidationContext:
    bridge_actions = frozenset(int(action) for action in bridge_legal_actions)
    tactical = tuple(
        sorted((int(action), str(reason)) for action, reason in (tactical_rejections or {}).items())
    )
    snapshot = facts or build_step_facts(observation, config.plant_types)
    rows = snapshot.rows
    cols = snapshot.columns
    slots = snapshot.seed_slots
    # Model-facing width is checkpoint semantics, never a mutable observation
    # hint.  Startup validation reports bridge/config mismatches separately;
    # the mask and decision cache must always retain the protected 701 entries.
    action_count = config.spec.action_count
    occupancy = snapshot.occupancy
    config_fingerprint = config.fingerprint
    return ActionValidationContext(
        frame_identity=observation_frame_identity(
            observation,
            bridge_legal_actions=bridge_actions,
            restart_screen=restart_screen,
            tactical_rejections=dict(tactical),
            facts=snapshot,
        ),
        facts=snapshot,
        config=config,
        config_fingerprint=config_fingerprint,
        action_count=action_count,
        rows=rows,
        cols=cols,
        sun=snapshot.sun,
        gameplay_ready=snapshot.lifecycle.gameplay_ready,
        board_found=snapshot.lifecycle.board_found,
        can_read_board=snapshot.lifecycle.can_read_board,
        seed_selection_active=snapshot.lifecycle.seed_selection_active,
        restart_screen=bool(restart_screen),
        seed_slots=slots,
        occupancy=occupancy,
        occupancy_by_cell={cell: plant.plant_type for cell, plant in snapshot.occupant_by_cell.items()},
        bridge_legal_actions=bridge_actions,
        tactical_rejections=tactical,
        tactical_rejection_by_action=dict(tactical),
    )


def build_action_intent(
    policy_action: int,
    *,
    source: str,
    mode: str = ACTION_SPACE_ADVENTURE_14_IDENTITY,
    observation: Optional[Mapping[str, Any]] = None,
    plant_types: Sequence[int] = (),
    max_seed_slots: Optional[int] = None,
    rows: int = 5,
    cols: int = 10,
    bridge_command: Optional[Mapping[str, Any]] = None,
    source_metadata: Optional[Mapping[str, Any]] = None,
) -> ActionIntent:
    spec = build_action_space_spec(
        mode=mode,
        plant_types=list(plant_types),
        max_seed_slots=max_seed_slots,
        rows=rows,
        cols=cols,
    )
    return _build_action_intent_with_spec(
        policy_action,
        source=source,
        spec=spec,
        observation=observation,
        plant_types=plant_types,
        max_seed_slots=max_seed_slots,
        rows=rows,
        cols=cols,
        bridge_command=bridge_command,
        source_metadata=source_metadata,
    )


def _build_action_intent_with_spec(
    policy_action: int,
    *,
    source: str,
    spec: ActionSpaceSpec,
    observation: Optional[Mapping[str, Any]],
    plant_types: Sequence[int],
    max_seed_slots: Optional[int],
    rows: int,
    cols: int,
    bridge_command: Optional[Mapping[str, Any]] = None,
    source_metadata: Optional[Mapping[str, Any]] = None,
) -> ActionIntent:
    action_id = int(policy_action)
    obs = dict(observation) if isinstance(observation, Mapping) else {}
    decoded = decode_policy_action(
        action_id,
        mode=spec.mode,
        observation=obs,
        plant_types=list(plant_types),
        max_seed_slots=max_seed_slots,
        rows=rows,
        cols=cols,
    )
    decoded_kind = _safe_int(decoded.get("kind"), -1)
    if action_id == spec.wait_action and decoded_kind == 0:
        action_kind = ACTION_KIND_WAIT
    elif decoded_kind == 1:
        action_kind = ACTION_KIND_PLACEMENT
    else:
        action_kind = ACTION_KIND_INVALID
    return ActionIntent(
        source=source,
        policy_action=action_id,
        bridge_action=action_id,
        action_kind=action_kind,
        seed_slot=_safe_int(decoded.get("slot_index"), -1),
        row=_safe_int(decoded.get("row"), -1),
        column=_safe_int(decoded.get("column"), -1),
        decoded_action=decoded,
        bridge_command=bridge_command,
        source_metadata=source_metadata or {},
    )


def _decision(
    intent: ActionIntent,
    context: ActionValidationContext,
    *,
    legal: bool,
    reason: str = "",
    resolved_action_kind: Optional[str] = None,
    selected_plant_type: int = -1,
    existing_plant_type: int = -1,
) -> ActionDecision:
    return ActionDecision(
        intent=intent,
        legal=bool(legal),
        rejection_reason=str(reason or ""),
        frame_identity=context.frame_identity.token,
        config_fingerprint=context.config_fingerprint,
        resolved_action_kind=resolved_action_kind or intent.action_kind,
        selected_plant_type=int(selected_plant_type),
        existing_plant_type=int(existing_plant_type),
    )


def validate_action_intent(
    intent: ActionIntent,
    context: ActionValidationContext,
) -> ActionDecision:
    """Pure placement/wait validation with the established rejection ordering.

    Ordering is externally visible in diagnostics and deliberately mirrors the
    pre-refactor ``PvZGymEnv._python_action_filter`` implementation.
    """

    if intent.action_kind == ACTION_KIND_WAIT:
        return _decision(intent, context, legal=True, resolved_action_kind=ACTION_KIND_WAIT)
    if context.restart_screen:
        return _decision(intent, context, legal=False, reason="restart_screen")
    if not context.gameplay_ready:
        return _decision(intent, context, legal=False, reason="gameplay_not_ready")
    if not context.board_found or not context.can_read_board:
        return _decision(intent, context, legal=False, reason="board_not_readable")
    if context.seed_selection_active:
        return _decision(intent, context, legal=False, reason="seed_selection_active")
    if context.rows <= 0 or context.cols <= 0:
        return _decision(intent, context, legal=False, reason="invalid_board_dimensions")
    if intent.action_kind != ACTION_KIND_PLACEMENT:
        return _decision(intent, context, legal=False, reason="invalid_action_decode")
    if not (0 <= intent.row < context.rows and 0 <= intent.column < context.cols):
        return _decision(intent, context, legal=False, reason="target_out_of_bounds")
    if not (0 <= intent.seed_slot < len(context.seed_slots)):
        return _decision(intent, context, legal=False, reason="seed_slot_index_out_of_range")

    slot = context.seed_slots[intent.seed_slot]
    selected_type = slot.plant_type
    if slot.slot_index != intent.seed_slot:
        return _decision(
            intent,
            context,
            legal=False,
            reason="seed_slot_index_mismatch",
            selected_plant_type=selected_type,
        )
    if not slot.usable:
        return _decision(
            intent,
            context,
            legal=False,
            reason="slot_not_usable",
            selected_plant_type=selected_type,
        )
    if slot.disabled:
        return _decision(
            intent,
            context,
            legal=False,
            reason="slot_disabled",
            selected_plant_type=selected_type,
        )
    if not slot.ready:
        return _decision(
            intent,
            context,
            legal=False,
            reason="cooldown_not_ready",
            selected_plant_type=selected_type,
        )
    if slot.full_cooldown > 0.05 and slot.current_cooldown > 0.05:
        return _decision(
            intent,
            context,
            legal=False,
            reason="cooldown_not_ready",
            selected_plant_type=selected_type,
        )
    if context.sun < slot.seed_cost:
        return _decision(
            intent,
            context,
            legal=False,
            reason="insufficient_sun",
            selected_plant_type=selected_type,
        )

    existing_type = context.occupancy_by_cell.get((intent.row, intent.column), -1)
    if existing_type >= 0:
        if not context.config.fusion_action_mask_enabled:
            return _decision(
                intent,
                context,
                legal=False,
                reason="occupied_cell",
                selected_plant_type=selected_type,
                existing_plant_type=existing_type,
            )
        if (existing_type, selected_type) not in context.config.fusion_compatible_pairs:
            return _decision(
                intent,
                context,
                legal=False,
                reason="incompatible_pair",
                selected_plant_type=selected_type,
                existing_plant_type=existing_type,
            )
        tactical_reason = context.tactical_rejection_by_action.get(intent.policy_action, "")
        if tactical_reason:
            return _decision(
                intent,
                context,
                legal=False,
                reason=tactical_reason,
                resolved_action_kind=ACTION_KIND_FUSION,
                selected_plant_type=selected_type,
                existing_plant_type=existing_type,
            )
        return _decision(
            intent,
            context,
            legal=True,
            resolved_action_kind=ACTION_KIND_FUSION,
            selected_plant_type=selected_type,
            existing_plant_type=existing_type,
        )

    if intent.bridge_action not in context.bridge_legal_actions:
        return _decision(
            intent,
            context,
            legal=False,
            reason="bridge_legal_actions_missing",
            selected_plant_type=selected_type,
        )
    tactical_reason = context.tactical_rejection_by_action.get(intent.policy_action, "")
    if tactical_reason:
        return _decision(
            intent,
            context,
            legal=False,
            reason=tactical_reason,
            selected_plant_type=selected_type,
        )
    return _decision(intent, context, legal=True, selected_plant_type=selected_type)


def validate_policy_action(
    policy_action: int,
    *,
    source: str,
    observation: Mapping[str, Any],
    config: ActionValidationConfig,
    bridge_legal_actions: Iterable[int],
    restart_screen: bool = False,
    tactical_rejections: Optional[Mapping[int, str]] = None,
    bridge_command: Optional[Mapping[str, Any]] = None,
    source_metadata: Optional[Mapping[str, Any]] = None,
) -> ActionDecision:
    """Stable convenience API for coaches, GUI/manual tools, and scripts."""

    context = build_action_validation_context(
        observation,
        config=config,
        bridge_legal_actions=bridge_legal_actions,
        restart_screen=restart_screen,
        tactical_rejections=tactical_rejections,
    )
    intent = build_action_intent(
        policy_action,
        source=source,
        mode=config.action_space_mode,
        observation=observation,
        plant_types=config.plant_types,
        max_seed_slots=config.max_seed_slots,
        rows=config.rows,
        cols=config.cols,
        bridge_command=bridge_command,
        source_metadata=source_metadata,
    )
    return validate_action_intent(intent, context)


def build_action_decision_cache(
    observation: Mapping[str, Any],
    *,
    config: ActionValidationConfig,
    bridge_legal_actions: Iterable[int],
    restart_screen: bool = False,
    tactical_rejections: Optional[Mapping[int, str]] = None,
    source: str = "mask",
    context: Optional[ActionValidationContext] = None,
) -> ActionDecisionCache:
    if context is None:
        context = build_action_validation_context(
            observation,
            config=config,
            bridge_legal_actions=bridge_legal_actions,
            restart_screen=restart_screen,
            tactical_rejections=tactical_rejections,
        )
    elif context.config is not config and context.config_fingerprint != config.fingerprint:
        raise ValueError("action decision cache context/config fingerprint mismatch")
    decisions = []
    spec = config.spec
    for action in range(context.action_count):
        intent = _build_action_intent_with_spec(
            action,
            source=source,
            spec=spec,
            observation=observation,
            plant_types=config.plant_types,
            max_seed_slots=config.max_seed_slots,
            rows=config.rows,
            cols=config.cols,
        )
        decisions.append(validate_action_intent(intent, context))
    frozen_decisions = tuple(decisions)
    return ActionDecisionCache(
        key=context.cache_key,
        context=context,
        decisions=frozen_decisions,
        mask=tuple(bool(decision.legal) for decision in frozen_decisions),
    )


__all__ = [
    "ACTION_KIND_FUSION",
    "ACTION_KIND_INVALID",
    "ACTION_KIND_PLACEMENT",
    "ACTION_KIND_WAIT",
    "ActionCacheKey",
    "ActionDecision",
    "ActionDecisionCache",
    "ActionIntent",
    "ActionResult",
    "ActionValidationConfig",
    "ActionValidationContext",
    "ObservationFrameIdentity",
    "SeedSlotFacts",
    "build_action_decision_cache",
    "build_action_intent",
    "build_action_validation_context",
    "compatible_pairs_for_observation",
    "observation_frame_identity",
    "validate_action_intent",
    "validate_policy_action",
]
