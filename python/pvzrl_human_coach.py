"""Local human-coach command parsing, validation, and override support.

This module is intentionally bridge-light. It turns trusted local/mock coach
commands into already-validated PvZRL policy actions, then lets the normal env
step path execute those actions. Twitch/YouTube adapters and voting live in a
future stream-coach layer.
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
from collections import deque

from pvzrl_action_space import (
    ACTION_SPACE_DYNAMIC_14,
    ActionSpaceSpec,
    build_action_space_spec,
    decode_policy_action,
    legacy_action_to_policy_action,
    normalize_action_space_mode,
)
from pvzrl_fusion import (
    FUSION_ILLEGAL_EMPTY_TILE,
    FUSION_ILLEGAL_INCOMPATIBLE,
    are_fusion_compatible,
    plant_name as fusion_plant_name,
    plant_type_at_cell,
    seed_plant_type_for_slot,
)
from pvzrl_file_tail import IncrementalLineTailReader


COACH_COMMANDS = {"plant", "fuse", "wait", "defend", "economy"}
COACH_COMMAND_MODES = {"override", "assist", "coach_only", "viewer_suggestion"}
COACH_MATCH_REWARD = 0.02
COACH_EXECUTED_REWARD = 0.01
COACH_OVERRIDE_PENALTY = -0.01
COACH_FUSION_SUCCESS_REWARD = 0.03
COACH_TACTICAL_USEFULNESS_REWARD = 0.01
COACH_REWARD_MATCH_COMPONENT = "coach_match_reward"
COACH_REWARD_LEGAL_EXECUTION_COMPONENT = "coach_legal_execution_reward"
COACH_REWARD_OVERRIDE_PENALTY_COMPONENT = "coach_override_penalty"
COACH_REWARD_FUSION_SUCCESS_COMPONENT = "coach_fusion_success_reward"
COACH_REWARD_TACTICAL_USEFULNESS_COMPONENT = "coach_tactical_usefulness_reward"

COACH_REJECTION_FUSION_DISABLED = "fusion_disabled"
COACH_REJECTION_FUSION_BRIDGE_UNAVAILABLE = "fusion_bridge_unavailable"
COACH_REJECTION_FUSION_PROBE_FAILED = "fusion_probe_failed"
COACH_REJECTION_FUSION_SOURCE_NOT_FOUND = "fusion_source_not_found"
COACH_REJECTION_FUSION_TARGET_NOT_AVAILABLE = "fusion_target_not_available"
COACH_REJECTION_FUSION_BRIDGE_REJECTED = "fusion_bridge_rejected"
COACH_REJECTION_FUSION_INVALID_STATE = "fusion_invalid_state"
COACH_REJECTION_FUSION_DIRECT_NOT_IMPLEMENTED = "fusion_direct_command_not_implemented"
# Centralized compatibility rejections (shared with the model mask via pvzrl_fusion).
COACH_REJECTION_FUSION_INCOMPATIBLE = FUSION_ILLEGAL_INCOMPATIBLE  # "incompatible_pair"
COACH_REJECTION_FUSION_EMPTY_TILE = FUSION_ILLEGAL_EMPTY_TILE  # "empty_tile"
COACH_REJECTION_INSUFFICIENT_SUN = "insufficient_sun"
COACH_REJECTION_COOLDOWN_NOT_READY = "cooldown_not_ready"
COACH_REJECTION_SLOT_NOT_USABLE = "slot_not_usable"
COACH_REJECTION_SLOT_DISABLED = "slot_disabled"
COACH_REJECTION_OCCUPIED_CELL = "occupied_cell"
COACH_REJECTION_PENDING_COMMAND = "pending_command"

COACH_PENDING_RETRY_REASONS: Set[str] = {
    COACH_REJECTION_INSUFFICIENT_SUN,
    COACH_REJECTION_COOLDOWN_NOT_READY,
}


_NEXT_COACH_COMMAND_ID = 0


def _next_coach_command_id() -> int:
    global _NEXT_COACH_COMMAND_ID
    _NEXT_COACH_COMMAND_ID += 1
    return int(_NEXT_COACH_COMMAND_ID)


@dataclass(frozen=True)
class CoachCommand:
    kind: str
    seed_index: Optional[int] = None
    row: Optional[int] = None
    col: Optional[int] = None
    raw_text: str = ""
    timestamp: float = field(default_factory=time.time)
    source: str = "human"
    valid_syntax: bool = True
    rejected_reason: str = ""
    coach_command_id: int = field(default_factory=_next_coach_command_id)

    def normalized_key(self) -> Tuple[Any, ...]:
        return (self.coach_command_id, self.kind, self.seed_index, self.row, self.col)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CoachActionValidation:
    command: CoachCommand
    legal: bool
    policy_action: Optional[int] = None
    rejected_reason: str = ""
    decoded: Dict[str, Any] = field(default_factory=dict)
    bridge_command: Optional[Dict[str, Any]] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["command"] = self.command.to_dict()
        return payload


@dataclass
class CoachDecision:
    enabled: bool
    ppo_action: int
    selected_action: int
    command: Optional[CoachCommand] = None
    validation: Optional[CoachActionValidation] = None
    selected_bridge_command: Optional[Dict[str, Any]] = None
    selected_action_label: str = ""
    override_applied: bool = False
    coach_match: bool = False
    rejected: bool = False
    rejected_reason: str = ""
    reward_components: Dict[str, float] = field(default_factory=dict)
    reward_delta: float = 0.0
    event: str = "no_command"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "ppo_action": int(self.ppo_action),
            "selected_action": int(self.selected_action),
            "selected_bridge_command": dict(self.selected_bridge_command) if isinstance(self.selected_bridge_command, dict) else None,
            "selected_action_label": str(self.selected_action_label or ""),
            "override_applied": bool(self.override_applied),
            "coach_match": bool(self.coach_match),
            "rejected": bool(self.rejected),
            "rejected_reason": str(self.rejected_reason),
            "reward_components": {str(key): float(value) for key, value in (self.reward_components or {}).items()},
            "reward_delta": float(self.reward_delta),
            "event": str(self.event),
            "timestamp": float(self.timestamp),
            "command": self.command.to_dict() if self.command is not None else None,
            "validation": self.validation.to_dict() if self.validation is not None else None,
        }


@dataclass
class HumanCoachStats:
    enabled: bool = False
    platform: str = "mock"
    last_command: Optional[Dict[str, Any]] = None
    last_action: Optional[Union[int, str]] = None
    last_vote_count: int = 0
    last_rejected_command: Optional[Dict[str, Any]] = None
    last_rejected_reason: str = ""
    last_error: str = ""
    top_commands: List[Dict[str, Any]] = field(default_factory=list)
    override_count: int = 0
    match_count: int = 0
    rejected_count: int = 0
    legal_execution_count: int = 0
    fusion_attempt_count: int = 0
    fusion_success_count: int = 0
    fusion_failure_count: int = 0
    tactical_useful_count: int = 0
    fusion_probe_count: int = 0
    fusion_probe_available_count: int = 0
    fusion_probe_unavailable_count: int = 0
    last_fusion_probe_reason: str = ""
    last_fusion_probe_diagnostics: Dict[str, Any] = field(default_factory=dict)
    fusion_bridge_available: Optional[bool] = None
    fusion_bridge_enabled: bool = False
    last_fusion_command: Optional[Dict[str, Any]] = None
    last_fusion_result: str = ""
    last_fusion_execution_mode: str = ""
    last_fusion_bridge_method_used: str = ""
    last_fusion_bridge_result_reason: str = ""
    last_fusion_duplicate_stack_detected: bool = False
    last_fusion_source_tile_occupied_before: bool = False
    last_fusion_plant_count_on_tile_before: int = 0
    last_fusion_plant_count_on_tile_after: int = 0
    last_fusion_source_plant_before: Optional[Dict[str, Any]] = None
    last_fusion_resulting_plant_after: Optional[Dict[str, Any]] = None
    last_fusion_predicted_result_resolution_source: str = ""
    last_fusion_mix_lookup_found: bool = False
    last_fusion_mix_lookup_key: str = ""
    last_fusion_pre_source_type: int = -1
    last_fusion_pre_source_name: str = ""
    last_fusion_ingredient_type: int = -1
    last_fusion_ingredient_name: str = ""
    last_fusion_post_result_type: int = -1
    last_fusion_post_result_name: str = ""
    last_fusion_no_effect_reason: str = ""
    last_fusion_scope: str = ""
    last_fusion_changed_tile_count: int = 0
    last_fusion_non_source_tiles_changed: bool = False
    last_fusion_global_side_effect: bool = False
    pending_coach_command: Optional[Dict[str, Any]] = None
    selected_bridge_command: Optional[Dict[str, Any]] = None
    last_executed_coach_command_id: Optional[int] = None
    coach_command_queue_cleared_on_reset: bool = True
    startup_command_blocked: bool = False
    coach_match_reward_total: float = 0.0
    coach_legal_execution_reward_total: float = 0.0
    coach_override_penalty_total: float = 0.0
    coach_fusion_success_reward_total: float = 0.0
    coach_tactical_usefulness_reward_total: float = 0.0
    reward_total: float = 0.0

    def live_status_fields(self) -> Dict[str, Any]:
        fields = {
            "human_coach_enabled": bool(self.enabled),
            "human_coach_platform": self.platform,
            "human_coach_last_command": self.last_command,
            "human_coach_last_action": self.last_action,
            "human_coach_last_vote_count": int(self.last_vote_count),
            "human_coach_override_count": int(self.override_count),
            "human_coach_match_count": int(self.match_count),
            "human_coach_rejected_count": int(self.rejected_count),
            "human_coach_legal_execution_count": int(self.legal_execution_count),
            "human_coach_fusion_attempt_count": int(self.fusion_attempt_count),
            "human_coach_fusion_success_count": int(self.fusion_success_count),
            "human_coach_fusion_failure_count": int(self.fusion_failure_count),
            "human_coach_tactical_useful_count": int(self.tactical_useful_count),
            "human_coach_fusion_probe_count": int(self.fusion_probe_count),
            "human_coach_fusion_probe_available_count": int(self.fusion_probe_available_count),
            "human_coach_fusion_probe_unavailable_count": int(self.fusion_probe_unavailable_count),
            "human_coach_last_fusion_probe_reason": str(self.last_fusion_probe_reason or ""),
            "human_coach_last_fusion_probe_diagnostics": dict(self.last_fusion_probe_diagnostics or {}),
            "human_coach_match_reward_total": float(self.coach_match_reward_total),
            "human_coach_legal_execution_reward_total": float(self.coach_legal_execution_reward_total),
            "human_coach_override_penalty_total": float(self.coach_override_penalty_total),
            "human_coach_fusion_success_reward_total": float(self.coach_fusion_success_reward_total),
            "human_coach_tactical_usefulness_reward_total": float(self.coach_tactical_usefulness_reward_total),
            "human_coach_reward_total": float(self.reward_total),
            "human_coach_last_rejected_command": self.last_rejected_command,
            "human_coach_last_rejected_reason": str(self.last_rejected_reason or ""),
            "human_coach_last_error": str(self.last_error or ""),
            "human_coach_top_commands": list(self.top_commands),
            "fusion_bridge_enabled": bool(self.fusion_bridge_enabled),
            "fusion_bridge_available": self.fusion_bridge_available,
            "fusion_last_command": self.last_fusion_command,
            "fusion_last_result": str(self.last_fusion_result or ""),
            "fusion_last_execution_mode": str(self.last_fusion_execution_mode or ""),
            "fusion_last_bridge_method_used": str(self.last_fusion_bridge_method_used or ""),
            "fusion_last_bridge_result_reason": str(self.last_fusion_bridge_result_reason or ""),
            "fusion_last_duplicate_stack_detected": bool(self.last_fusion_duplicate_stack_detected),
            "fusion_last_source_tile_occupied_before": bool(self.last_fusion_source_tile_occupied_before),
            "fusion_last_plant_count_on_tile_before": int(self.last_fusion_plant_count_on_tile_before),
            "fusion_last_plant_count_on_tile_after": int(self.last_fusion_plant_count_on_tile_after),
            "fusion_last_source_plant_before": self.last_fusion_source_plant_before,
            "fusion_last_resulting_plant_after": self.last_fusion_resulting_plant_after,
            "fusion_last_predicted_result_resolution_source": str(self.last_fusion_predicted_result_resolution_source or ""),
            "fusion_last_mix_lookup_found": bool(self.last_fusion_mix_lookup_found),
            "fusion_last_mix_lookup_key": str(self.last_fusion_mix_lookup_key or ""),
            "fusion_last_pre_source_type": int(self.last_fusion_pre_source_type),
            "fusion_last_pre_source_name": str(self.last_fusion_pre_source_name or ""),
            "fusion_last_ingredient_type": int(self.last_fusion_ingredient_type),
            "fusion_last_ingredient_name": str(self.last_fusion_ingredient_name or ""),
            "fusion_last_post_result_type": int(self.last_fusion_post_result_type),
            "fusion_last_post_result_name": str(self.last_fusion_post_result_name or ""),
            "fusion_last_no_effect_reason": str(self.last_fusion_no_effect_reason or ""),
            "last_fusion_scope": str(self.last_fusion_scope or ""),
            "last_fusion_changed_tile_count": int(self.last_fusion_changed_tile_count),
            "last_fusion_non_source_tiles_changed": bool(self.last_fusion_non_source_tiles_changed),
            "last_fusion_global_side_effect": bool(self.last_fusion_global_side_effect),
            "fusion_last_scope": str(self.last_fusion_scope or ""),
            "fusion_last_changed_tile_count": int(self.last_fusion_changed_tile_count),
            "fusion_last_non_source_tiles_changed": bool(self.last_fusion_non_source_tiles_changed),
            "fusion_last_global_side_effect": bool(self.last_fusion_global_side_effect),
            "pending_coach_command": self.pending_coach_command,
            "selected_bridge_command": self.selected_bridge_command,
            "last_executed_coach_command_id": self.last_executed_coach_command_id,
            "coach_command_queue_cleared_on_reset": bool(self.coach_command_queue_cleared_on_reset),
            "startup_command_blocked": bool(self.startup_command_blocked),
        }
        fields.update(
            {
                "stream_coach_enabled": bool(self.enabled),
                "stream_coach_platform": self.platform,
                "stream_coach_active_viewers_estimate": None,
                "stream_coach_last_command": self.last_command,
                "stream_coach_last_action": self.last_action,
                "stream_coach_last_vote_count": int(self.last_vote_count),
                "stream_coach_top_commands": list(self.top_commands),
                "stream_coach_override_count": int(self.override_count),
                "stream_coach_match_count": int(self.match_count),
                "stream_coach_rejected_count": int(self.rejected_count),
                "stream_coach_legal_execution_count": int(self.legal_execution_count),
                "stream_coach_fusion_attempt_count": int(self.fusion_attempt_count),
                "stream_coach_fusion_success_count": int(self.fusion_success_count),
                "stream_coach_fusion_failure_count": int(self.fusion_failure_count),
                "stream_coach_tactical_useful_count": int(self.tactical_useful_count),
                "stream_coach_fusion_probe_count": int(self.fusion_probe_count),
                "stream_coach_fusion_probe_available_count": int(self.fusion_probe_available_count),
                "stream_coach_fusion_probe_unavailable_count": int(self.fusion_probe_unavailable_count),
                "stream_coach_last_fusion_probe_reason": str(self.last_fusion_probe_reason or ""),
                "stream_coach_last_fusion_probe_diagnostics": dict(self.last_fusion_probe_diagnostics or {}),
                "stream_coach_match_reward_total": float(self.coach_match_reward_total),
                "stream_coach_legal_execution_reward_total": float(self.coach_legal_execution_reward_total),
                "stream_coach_override_penalty_total": float(self.coach_override_penalty_total),
                "stream_coach_fusion_success_reward_total": float(self.coach_fusion_success_reward_total),
                "stream_coach_tactical_usefulness_reward_total": float(self.coach_tactical_usefulness_reward_total),
                "stream_coach_reward_total": float(self.reward_total),
                "stream_coach_last_rejected_command": self.last_rejected_command,
                "stream_coach_last_rejected_reason": str(self.last_rejected_reason or ""),
                "stream_coach_last_error": str(self.last_error or ""),
                "stream_fusion_bridge_enabled": bool(self.fusion_bridge_enabled),
                "stream_fusion_bridge_available": self.fusion_bridge_available,
                "stream_fusion_last_command": self.last_fusion_command,
                "stream_fusion_last_result": str(self.last_fusion_result or ""),
                "stream_fusion_last_execution_mode": str(self.last_fusion_execution_mode or ""),
                "stream_fusion_last_bridge_method_used": str(self.last_fusion_bridge_method_used or ""),
                "stream_fusion_last_bridge_result_reason": str(self.last_fusion_bridge_result_reason or ""),
                "stream_fusion_last_duplicate_stack_detected": bool(self.last_fusion_duplicate_stack_detected),
                "stream_fusion_last_source_tile_occupied_before": bool(self.last_fusion_source_tile_occupied_before),
                "stream_fusion_last_plant_count_on_tile_before": int(self.last_fusion_plant_count_on_tile_before),
                "stream_fusion_last_plant_count_on_tile_after": int(self.last_fusion_plant_count_on_tile_after),
                "stream_fusion_last_source_plant_before": self.last_fusion_source_plant_before,
                "stream_fusion_last_resulting_plant_after": self.last_fusion_resulting_plant_after,
                "stream_fusion_last_predicted_result_resolution_source": str(self.last_fusion_predicted_result_resolution_source or ""),
                "stream_fusion_last_mix_lookup_found": bool(self.last_fusion_mix_lookup_found),
                "stream_fusion_last_mix_lookup_key": str(self.last_fusion_mix_lookup_key or ""),
                "stream_fusion_last_pre_source_type": int(self.last_fusion_pre_source_type),
                "stream_fusion_last_pre_source_name": str(self.last_fusion_pre_source_name or ""),
                "stream_fusion_last_ingredient_type": int(self.last_fusion_ingredient_type),
                "stream_fusion_last_ingredient_name": str(self.last_fusion_ingredient_name or ""),
                "stream_fusion_last_post_result_type": int(self.last_fusion_post_result_type),
                "stream_fusion_last_post_result_name": str(self.last_fusion_post_result_name or ""),
                "stream_fusion_last_no_effect_reason": str(self.last_fusion_no_effect_reason or ""),
                "stream_fusion_last_scope": str(self.last_fusion_scope or ""),
                "stream_fusion_last_changed_tile_count": int(self.last_fusion_changed_tile_count),
                "stream_fusion_last_non_source_tiles_changed": bool(self.last_fusion_non_source_tiles_changed),
                "stream_fusion_last_global_side_effect": bool(self.last_fusion_global_side_effect),
            }
        )
        return fields


@dataclass
class HumanCoachConfig:
    enabled: bool = False
    log_path: Optional[Union[str, Path]] = None
    command_path: Optional[Union[str, Path]] = None
    reward_enabled: bool = False
    fusion_enabled: bool = False
    platform: str = "mock"
    command_mode: str = "override"
    match_reward: float = COACH_MATCH_REWARD
    legal_execution_reward: float = COACH_EXECUTED_REWARD
    override_penalty: float = COACH_OVERRIDE_PENALTY
    fusion_success_reward: float = COACH_FUSION_SUCCESS_REWARD
    tactical_usefulness_reward: float = COACH_TACTICAL_USEFULNESS_REWARD


class HumanCoachJSONLLogger:
    def __init__(self, path: Optional[Union[str, Path]]) -> None:
        self.path = Path(path) if path else None

    def log(self, event: str, **payload: Any) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {"timestamp": time.time(), "event": event, **payload}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")


class QueueCoachCommandSource:
    def __init__(self, commands: Optional[Iterable[Union[str, CoachCommand]]] = None) -> None:
        self._queue: Deque[Union[str, CoachCommand]] = deque(commands or [])

    def submit(self, command: Union[str, CoachCommand]) -> None:
        self._queue.append(command)

    def poll(self) -> Optional[Union[str, CoachCommand]]:
        if not self._queue:
            return None
        return self._queue.popleft()

    def clear_pending(self) -> int:
        cleared = len(self._queue)
        self._queue.clear()
        return int(cleared)

    def clear_to_end(self) -> int:
        return self.clear_pending()


class FileCoachCommandSource:
    """Consume newly appended local mock commands from a plain text or JSONL file."""

    def __init__(self, path: Union[str, Path], *, start_at_end: bool = True) -> None:
        self.path = Path(path)
        self._tail = IncrementalLineTailReader(self.path, start_at_end=start_at_end)
        self._offset = self._tail.offset
        self._last_error = ""
        self._queue: Deque[str] = deque()

    def poll(self) -> Optional[str]:
        if self._queue:
            return self._queue.popleft()
        self._last_error = ""
        for line in self._tail.read_lines():
            stripped = line.strip()
            if stripped.startswith("{"):
                try:
                    json.loads(stripped)
                except json.JSONDecodeError:
                    self._tail.note_malformed_record("json_decode_error")
                    self._last_error = "json_decode_error"
                    continue
            text = _command_text_from_line(line)
            if text:
                self._queue.append(text)
        self._offset = self._tail.offset
        if not self._last_error and self._tail.last_error:
            self._last_error = self._tail.last_error
        if not self._queue:
            return None
        return self._queue.popleft()

    def clear_pending(self) -> int:
        cleared = len(self._queue)
        self._queue.clear()
        return int(cleared)

    def clear_to_end(self) -> int:
        cleared = self.clear_pending()
        self._tail.clear_to_end()
        if self._tail.last_clear_had_bytes:
            cleared += 1
        self._offset = self._tail.offset
        return int(cleared)


CommandSource = Union[QueueCoachCommandSource, FileCoachCommandSource, Callable[[], Optional[Union[str, CoachCommand]]]]


class HumanCoachOverrideHook:
    """Select a validated coach override before the environment executes a step."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        source: Optional[CommandSource] = None,
        log_path: Optional[Union[str, Path]] = None,
        reward_enabled: bool = False,
        fusion_enabled: bool = False,
        platform: str = "mock",
        command_mode: str = "override",
        match_reward: float = COACH_MATCH_REWARD,
        legal_execution_reward: float = COACH_EXECUTED_REWARD,
        override_penalty: float = COACH_OVERRIDE_PENALTY,
        fusion_success_reward: float = COACH_FUSION_SUCCESS_REWARD,
        tactical_usefulness_reward: float = COACH_TACTICAL_USEFULNESS_REWARD,
    ) -> None:
        self.enabled = bool(enabled)
        self.source = source
        self.reward_enabled = bool(reward_enabled)
        self.fusion_enabled = bool(fusion_enabled)
        normalized_mode = str(command_mode or "override").strip().lower()
        self.command_mode = normalized_mode if normalized_mode in COACH_COMMAND_MODES else "override"
        self.match_reward = float(match_reward)
        self.legal_execution_reward = float(legal_execution_reward)
        self.override_penalty = float(override_penalty)
        self.fusion_success_reward = float(fusion_success_reward)
        self.tactical_usefulness_reward = float(tactical_usefulness_reward)
        self.stats = HumanCoachStats(enabled=bool(enabled), platform=str(platform or "mock"))
        self.stats.fusion_bridge_enabled = bool(fusion_enabled)
        self.logger = HumanCoachJSONLLogger(log_path)
        self._pending_command: Optional[CoachCommand] = None
        self._pending_reason: str = ""
        self._issued_command_ids: Set[int] = set()

    @classmethod
    def from_config(cls, config: HumanCoachConfig, source: Optional[CommandSource] = None) -> "HumanCoachOverrideHook":
        command_source = source
        if command_source is None and config.command_path:
            command_source = FileCoachCommandSource(config.command_path)
        return cls(
            enabled=bool(config.enabled),
            source=command_source,
            log_path=config.log_path,
            reward_enabled=bool(config.reward_enabled),
            fusion_enabled=bool(config.fusion_enabled),
            platform=str(config.platform or "mock"),
            command_mode=str(config.command_mode or "override"),
            match_reward=float(config.match_reward),
            legal_execution_reward=float(config.legal_execution_reward),
            override_penalty=float(config.override_penalty),
            fusion_success_reward=float(config.fusion_success_reward),
            tactical_usefulness_reward=float(config.tactical_usefulness_reward),
        )

    def poll_command(self) -> Optional[CoachCommand]:
        if not self.enabled or self.source is None:
            return None
        raw: Optional[Union[str, CoachCommand]]
        if callable(self.source):
            raw = self.source()
        else:
            raw = self.source.poll()
        if raw is None:
            return None
        if isinstance(raw, CoachCommand):
            return self._ensure_command_id(raw)
        return parse_coach_command(str(raw), source=self.stats.platform)

    @staticmethod
    def _ensure_command_id(command: CoachCommand) -> CoachCommand:
        try:
            command_id = int(getattr(command, "coach_command_id", 0) or 0)
        except (TypeError, ValueError):
            command_id = 0
        if command_id > 0:
            return command
        return replace(command, coach_command_id=_next_coach_command_id())

    def clear_pending_state(
        self,
        *,
        clear_source: bool = True,
        reason: str = "reset",
        preserve_startup_blocked: bool = False,
    ) -> bool:
        stale_detected = self._pending_command is not None or isinstance(self.stats.selected_bridge_command, dict)
        self._pending_command = None
        self._pending_reason = ""
        self.stats.pending_coach_command = None
        self.stats.selected_bridge_command = None
        if clear_source and self.source is not None and not callable(self.source):
            clear_fn = getattr(self.source, "clear_to_end", None) or getattr(self.source, "clear_pending", None)
            if callable(clear_fn):
                try:
                    stale_detected = bool(int(clear_fn() or 0) > 0 or stale_detected)
                except Exception:
                    stale_detected = True
        self.stats.coach_command_queue_cleared_on_reset = True
        self.stats.startup_command_blocked = bool(
            stale_detected or (preserve_startup_blocked and self.stats.startup_command_blocked)
        )
        self.logger.log(
            "coach_command_queue_cleared",
            reason=str(reason or "reset"),
            stale_detected=bool(stale_detected),
        )
        return bool(stale_detected)

    def select_action(self, env: Any, ppo_action: int) -> CoachDecision:
        ppo_action = int(ppo_action)
        if not self.enabled:
            return CoachDecision(enabled=False, ppo_action=ppo_action, selected_action=ppo_action, event="disabled")
        latest_command = self.poll_command()
        if latest_command is not None:
            command = latest_command
            self._pending_command = None
            self._pending_reason = ""
        else:
            command = self._pending_command
        if command is None:
            self.stats.pending_coach_command = None
            self.stats.selected_bridge_command = None
            if self.command_mode == "coach_only":
                observation = getattr(env, "_last_observation", None)
                if not isinstance(observation, dict):
                    observation = {}
                wait_action = self._wait_action_for_env(env, observation=observation, fallback=ppo_action)
                return CoachDecision(
                    enabled=True,
                    ppo_action=ppo_action,
                    selected_action=int(wait_action),
                    override_applied=int(wait_action) != ppo_action,
                    coach_match=int(wait_action) == ppo_action,
                    selected_action_label="coach_only_wait",
                    event="coach_only_wait",
                )
            return CoachDecision(enabled=True, ppo_action=ppo_action, selected_action=ppo_action, event="no_command")
        command = self._ensure_command_id(command)

        command_id = int(command.coach_command_id)
        if command_id in self._issued_command_ids:
            reason = "coach_command_already_executed"
            self._pending_command = None
            self._pending_reason = ""
            self.stats.pending_coach_command = None
            self.stats.selected_bridge_command = None
            self.stats.rejected_count += 1
            self.stats.last_command = command.to_dict()
            self.stats.last_rejected_command = command.to_dict()
            self.stats.last_rejected_reason = str(reason)
            self.stats.last_error = str(reason)
            self.stats.last_vote_count = 0
            self.logger.log("coach_rejected", command=command.to_dict(), ppo_action=ppo_action, rejected_reason=reason)
            return CoachDecision(
                enabled=True,
                ppo_action=ppo_action,
                selected_action=ppo_action,
                command=command,
                rejected=True,
                rejected_reason=reason,
                event="coach_rejected",
            )

        observation = getattr(env, "_last_observation", None)
        if not isinstance(observation, dict):
            observation = {}
        action_mask = None
        try:
            action_mask = env.action_masks()
        except Exception:
            action_mask = None

        validation = validate_coach_command_for_env(
            command,
            env,
            observation=observation,
            action_mask=action_mask,
            force_fusion_enabled=self.fusion_enabled,
        )
        self._record_fusion_probe_stats(validation)
        if not validation.legal or validation.policy_action is None:
            reason = validation.rejected_reason or command.rejected_reason or "illegal_command"
            if _is_pending_retry_reason(command, reason):
                self._pending_command = command
                self._pending_reason = str(reason)
                self.stats.pending_coach_command = command.to_dict()
                self.stats.selected_bridge_command = None
                wait_action = self._wait_action_for_env(env, observation=observation, fallback=ppo_action)
                pending_validation = _validation(
                    command,
                    False,
                    policy_action=int(wait_action),
                    rejected_reason=str(reason),
                    decoded={"kind": "wait"},
                    diagnostics={
                        "pending": True,
                        "pending_reason": str(reason),
                        "pending_command_kind": str(command.kind),
                    },
                )
                match = int(wait_action) == int(ppo_action)
                override = not match
                self.stats.last_command = command.to_dict()
                self.stats.last_action = int(wait_action)
                self.stats.last_vote_count = 1
                self.stats.last_rejected_reason = f"{COACH_REJECTION_PENDING_COMMAND}:{reason}"
                self.stats.last_error = self.stats.last_rejected_reason
                self.logger.log(
                    "coach_pending",
                    command=command.to_dict(),
                    validation=pending_validation.to_dict(),
                    ppo_action=ppo_action,
                    pending_reason=reason,
                )
                return CoachDecision(
                    enabled=True,
                    ppo_action=ppo_action,
                    selected_action=int(wait_action),
                    command=command,
                    validation=pending_validation,
                    selected_action_label=f"pending_wait:{reason}",
                    override_applied=override,
                    coach_match=match,
                    rejected=False,
                    rejected_reason=str(reason),
                    event="coach_pending",
                )

            self._pending_command = None
            self._pending_reason = ""
            self.stats.pending_coach_command = None
            self.stats.selected_bridge_command = None
            self.stats.rejected_count += 1
            self.stats.last_command = command.to_dict()
            self.stats.last_rejected_command = command.to_dict()
            self.stats.last_rejected_reason = str(reason)
            self.stats.last_error = str(reason)
            self.stats.last_vote_count = 0
            self.logger.log(
                "coach_rejected",
                command=command.to_dict(),
                validation=validation.to_dict(),
                ppo_action=ppo_action,
                rejected_reason=reason,
            )
            return CoachDecision(
                enabled=True,
                ppo_action=ppo_action,
                selected_action=ppo_action,
                command=command,
                validation=validation,
                rejected=True,
                rejected_reason=reason,
                event="coach_rejected",
            )

        self._pending_command = None
        self._pending_reason = ""
        self.stats.pending_coach_command = None
        selected_action = int(validation.policy_action)
        selected_bridge_command = dict(validation.bridge_command) if isinstance(validation.bridge_command, dict) else None
        if self.command_mode in {"assist", "viewer_suggestion"}:
            self._issued_command_ids.add(int(command.coach_command_id))
            self.stats.selected_bridge_command = None
            self.stats.last_command = command.to_dict()
            self.stats.last_action = int(ppo_action)
            self.stats.last_vote_count = 1
            self.stats.last_rejected_reason = ""
            self.stats.last_error = ""
            decision = CoachDecision(
                enabled=True,
                ppo_action=ppo_action,
                selected_action=ppo_action,
                command=command,
                validation=validation,
                selected_action_label=f"suggestion:{selected_action}",
                override_applied=False,
                coach_match=selected_action == ppo_action,
                event="coach_suggestion",
            )
            self.logger.log("coach_suggestion", command_mode=self.command_mode, decision=decision.to_dict())
            return decision
        if selected_bridge_command is not None:
            selected_bridge_command["coach_command_id"] = int(command.coach_command_id)
            selected_bridge_command["coach_command_timestamp"] = float(command.timestamp)
            selected_bridge_command["coach_command_source"] = str(command.source or "")
            selected_bridge_command["executed_from_fresh_coach_command"] = True
        self._issued_command_ids.add(int(command.coach_command_id))
        self.stats.selected_bridge_command = dict(selected_bridge_command) if isinstance(selected_bridge_command, dict) else None
        match = selected_action == ppo_action
        override = not match
        reward_components = self._base_reward_components(match=match, override=override, legal_execution=True)
        reward_delta = sum(reward_components.values())
        if reward_components:
            self.stats.coach_legal_execution_reward_total += reward_components.get(COACH_REWARD_LEGAL_EXECUTION_COMPONENT, 0.0)
            self.stats.coach_match_reward_total += reward_components.get(COACH_REWARD_MATCH_COMPONENT, 0.0)
            self.stats.coach_override_penalty_total += reward_components.get(COACH_REWARD_OVERRIDE_PENALTY_COMPONENT, 0.0)
        if match:
            self.stats.match_count += 1
        if override:
            self.stats.override_count += 1
        self.stats.legal_execution_count += 1
        self.stats.reward_total += reward_delta
        self.stats.last_command = command.to_dict()
        self.stats.last_action = "fusion_step" if selected_bridge_command else selected_action
        self.stats.last_vote_count = 1
        self.stats.last_rejected_reason = ""
        self.stats.last_error = ""
        if command.kind == "fuse":
            self.stats.last_fusion_command = command.to_dict()
            self.stats.last_fusion_result = "pending"
        event = "coach_match" if match else "coach_override"
        action_label = "fusion_step" if selected_bridge_command else f"policy_action:{selected_action}"
        decision = CoachDecision(
            enabled=True,
            ppo_action=ppo_action,
            selected_action=selected_action,
            command=command,
            validation=validation,
            selected_bridge_command=selected_bridge_command,
            selected_action_label=action_label,
            override_applied=override,
            coach_match=match,
            reward_components=reward_components,
            reward_delta=reward_delta,
            event=event,
        )
        self.logger.log(event, decision=decision.to_dict())
        return decision

    def _wait_action_for_env(self, env: Any, *, observation: Dict[str, Any], fallback: int) -> int:
        action_spec = getattr(env, "action_spec", None)
        mode = getattr(action_spec, "mode", "fixed")
        rows = _safe_int(
            observation.get("rowCount"),
            getattr(env, "rows", None),
            getattr(action_spec, "rows", None),
            default=5,
        )
        cols = _safe_int(
            observation.get("columnCount"),
            getattr(env, "cols", None),
            getattr(action_spec, "cols", None),
            default=10,
        )
        max_seed_slots_raw = getattr(action_spec, "max_seed_slots", None)
        max_seed_slots = None
        if max_seed_slots_raw is not None:
            parsed_slots = _safe_int(max_seed_slots_raw, default=0)
            if parsed_slots > 0:
                max_seed_slots = int(parsed_slots)
        config = getattr(env, "config", None)
        plant_types = list(getattr(config, "plant_types", []) or [])
        try:
            spec = build_action_space_spec(
                mode=mode,
                plant_types=[int(value) for value in plant_types],
                max_seed_slots=max_seed_slots,
                rows=int(max(1, rows)),
                cols=int(max(1, cols)),
            )
            return int(spec.wait_action)
        except Exception:
            return int(fallback)

    def step_env(self, env: Any, ppo_action: int) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        decision = self.select_action(env, ppo_action)
        coach_context = {
            "enabled": bool(decision.enabled),
            "rewardEnabled": bool(self.reward_enabled),
            "event": str(decision.event),
            "overrideApplied": bool(decision.override_applied),
            "coachMatch": bool(decision.coach_match),
            "selectedAction": int(decision.selected_action),
            "ppoAction": int(decision.ppo_action),
            "command": decision.command.to_dict() if decision.command is not None else None,
            "validation": decision.validation.to_dict() if decision.validation is not None else None,
        }
        try:
            obs, reward, terminated, truncated, info = env.step(
                decision.selected_action,
                coach_bridge_command=decision.selected_bridge_command,
                coach_context=coach_context,
            )
        except TypeError:
            obs, reward, terminated, truncated, info = env.step(decision.selected_action)
        decision_bonus = self.apply_step_outcome(decision, info if isinstance(info, dict) else {})
        if decision_bonus:
            reward = float(reward) + float(decision_bonus)
        if isinstance(info, dict):
            info["human_coach"] = decision.to_dict()
            info["stream_coach"] = self.live_status_fields()
        return obs, float(reward), bool(terminated), bool(truncated), info

    def live_status_fields(self) -> Dict[str, Any]:
        self.stats.enabled = bool(self.enabled)
        fields = self.stats.live_status_fields()
        fields["human_coach_command_mode"] = self.command_mode
        return fields

    def apply_step_outcome(self, decision: CoachDecision, info: Dict[str, Any]) -> float:
        components = dict(decision.reward_components or {})
        for key in (
            COACH_REWARD_MATCH_COMPONENT,
            COACH_REWARD_LEGAL_EXECUTION_COMPONENT,
            COACH_REWARD_OVERRIDE_PENALTY_COMPONENT,
            COACH_REWARD_FUSION_SUCCESS_COMPONENT,
            COACH_REWARD_TACTICAL_USEFULNESS_COMPONENT,
        ):
            components[key] = float(components.get(key, 0.0))
        if decision.event == "coach_suggestion":
            self.stats.selected_bridge_command = None
            return 0.0
        if (
            decision.command is not None
            and decision.validation is not None
            and decision.validation.legal
        ):
            self.stats.last_executed_coach_command_id = int(decision.command.coach_command_id)
        self.stats.selected_bridge_command = None
        if decision.command is not None and decision.command.kind == "fuse" and decision.validation is not None and decision.validation.legal:
            self.stats.fusion_attempt_count += 1
            action_result = info.get("action_result") if isinstance(info, dict) else {}
            if not isinstance(action_result, dict):
                action_result = {}
            placement = action_result.get("placement") if isinstance(action_result.get("placement"), dict) else {}
            self._validate_fusion_postconditions(action_result, placement)
            self.stats.last_fusion_execution_mode = str(
                action_result.get("fusionExecutionMode")
                or placement.get("fusionExecutionMode")
                or ""
            )
            self.stats.last_fusion_bridge_method_used = str(
                action_result.get("bridgeMethodUsed")
                or placement.get("bridgeMethodUsed")
                or ""
            )
            self.stats.last_fusion_bridge_result_reason = str(
                action_result.get("bridgeResultReason")
                or placement.get("bridgeResultReason")
                or ""
            )
            self.stats.last_fusion_duplicate_stack_detected = bool(
                action_result.get("duplicateStackDetected")
                or placement.get("duplicateStackDetected")
            )
            self.stats.last_fusion_source_tile_occupied_before = bool(
                action_result.get("sourceTileOccupiedBefore")
                if "sourceTileOccupiedBefore" in action_result
                else placement.get("sourceTileOccupiedBefore")
            )
            self.stats.last_fusion_plant_count_on_tile_before = _safe_int(
                action_result.get("plantCountOnTileBefore"),
                placement.get("plantCountOnTileBefore"),
                default=0,
            )
            self.stats.last_fusion_plant_count_on_tile_after = _safe_int(
                action_result.get("plantCountOnTileAfter"),
                placement.get("plantCountOnTileAfter"),
                default=0,
            )
            source_before = action_result.get("sourcePlantBefore")
            if not isinstance(source_before, dict):
                source_before = placement.get("sourcePlantBefore")
            resulting_after = action_result.get("resultingPlantAfter")
            if not isinstance(resulting_after, dict):
                resulting_after = placement.get("resultingPlantAfter")
            self.stats.last_fusion_source_plant_before = dict(source_before) if isinstance(source_before, dict) else None
            self.stats.last_fusion_resulting_plant_after = dict(resulting_after) if isinstance(resulting_after, dict) else None
            self.stats.last_fusion_predicted_result_resolution_source = str(
                action_result.get("predictedResultResolutionSource")
                or placement.get("predictedResultResolutionSource")
                or ""
            )
            self.stats.last_fusion_mix_lookup_found = bool(
                action_result.get("mixLookupFound")
                if "mixLookupFound" in action_result
                else placement.get("mixLookupFound")
            )
            self.stats.last_fusion_mix_lookup_key = str(
                action_result.get("mixLookupKey")
                or placement.get("mixLookupKey")
                or ""
            )
            self.stats.last_fusion_pre_source_type = _safe_int(
                action_result.get("preSourceType"),
                placement.get("preSourceType"),
                default=-1,
            )
            self.stats.last_fusion_pre_source_name = str(
                action_result.get("preSourceName")
                or placement.get("preSourceName")
                or ""
            )
            self.stats.last_fusion_ingredient_type = _safe_int(
                action_result.get("ingredientType"),
                placement.get("ingredientType"),
                default=-1,
            )
            self.stats.last_fusion_ingredient_name = str(
                action_result.get("ingredientName")
                or placement.get("ingredientName")
                or ""
            )
            self.stats.last_fusion_post_result_type = _safe_int(
                action_result.get("postResultType"),
                placement.get("postResultType"),
                default=-1,
            )
            self.stats.last_fusion_post_result_name = str(
                action_result.get("postResultName")
                or placement.get("postResultName")
                or ""
            )
            self.stats.last_fusion_no_effect_reason = str(
                action_result.get("noEffectReason")
                or placement.get("noEffectReason")
                or ""
            )
            self.stats.last_fusion_scope = str(
                action_result.get("fusionScope")
                or action_result.get("fusion_scope")
                or placement.get("fusionScope")
                or placement.get("fusion_scope")
                or ""
            )
            self.stats.last_fusion_changed_tile_count = _safe_int(
                action_result.get("changedTileCount"),
                action_result.get("changed_tile_count"),
                placement.get("changedTileCount"),
                placement.get("changed_tile_count"),
                default=0,
            )
            self.stats.last_fusion_non_source_tiles_changed = bool(
                action_result.get("nonSourceTilesChanged")
                if "nonSourceTilesChanged" in action_result
                else action_result.get("non_source_tiles_changed")
                if "non_source_tiles_changed" in action_result
                else placement.get("nonSourceTilesChanged")
                if "nonSourceTilesChanged" in placement
                else placement.get("non_source_tiles_changed")
            )
            self.stats.last_fusion_global_side_effect = bool(
                action_result.get("globalFusionSideEffect")
                if "globalFusionSideEffect" in action_result
                else action_result.get("global_fusion_side_effect")
                if "global_fusion_side_effect" in action_result
                else placement.get("globalFusionSideEffect")
                if "globalFusionSideEffect" in placement
                else placement.get("global_fusion_side_effect")
            )
            fusion_success = bool(
                action_result.get("fusionSucceeded")
                or action_result.get("fusion_success")
                or action_result.get("fusionOverrideApplied")
            )
            if fusion_success:
                self.stats.fusion_success_count += 1
                self.stats.last_fusion_result = "success"
                self.stats.last_rejected_reason = ""
                self.stats.last_error = ""
                if self.reward_enabled:
                    components[COACH_REWARD_FUSION_SUCCESS_COMPONENT] += self.fusion_success_reward
            else:
                self.stats.fusion_failure_count += 1
                self.stats.last_fusion_result = "failed"
                failure_reason = str(
                    action_result.get("fusionRejectedReason")
                    or action_result.get("illegalReason")
                    or self.stats.last_fusion_bridge_result_reason
                    or "bridge_rejected"
                )
                self.stats.last_rejected_reason = failure_reason
                self.stats.last_error = failure_reason
        tactical_useful = self._is_tactical_useful(decision, info)
        if tactical_useful:
            self.stats.tactical_useful_count += 1
            if self.reward_enabled:
                components[COACH_REWARD_TACTICAL_USEFULNESS_COMPONENT] += self.tactical_usefulness_reward

        outcome_reward = sum(float(value) for value in components.values())
        if decision.command is None and outcome_reward == 0.0:
            decision.reward_components = components
            decision.reward_delta = 0.0
            return 0.0
        self.stats.coach_fusion_success_reward_total += components.get(COACH_REWARD_FUSION_SUCCESS_COMPONENT, 0.0)
        self.stats.coach_tactical_usefulness_reward_total += components.get(COACH_REWARD_TACTICAL_USEFULNESS_COMPONENT, 0.0)
        baseline_reward = float(decision.reward_delta or 0.0)
        additional_reward = outcome_reward - baseline_reward
        if additional_reward != 0.0:
            self.stats.reward_total += additional_reward
        decision.reward_components = components
        decision.reward_delta = outcome_reward
        if isinstance(info, dict):
            breakdown = info.get("reward_breakdown")
            if not isinstance(breakdown, dict):
                breakdown = {}
            for key, value in components.items():
                try:
                    breakdown[key] = float(breakdown.get(key) or 0.0) + float(value)
                except (TypeError, ValueError):
                    breakdown[key] = float(value)
            try:
                breakdown["reward_total"] = float(breakdown.get("reward_total") or 0.0) + float(outcome_reward)
            except (TypeError, ValueError):
                breakdown["reward_total"] = float(outcome_reward)
            info["reward_breakdown"] = breakdown
            if decision.command is not None and decision.command.kind == "fuse":
                info["human_coach_fusion_success"] = bool(
                    decision.reward_components.get(COACH_REWARD_FUSION_SUCCESS_COMPONENT, 0.0) > 0.0
                )
                info["human_coach_tactical_useful"] = bool(tactical_useful)
        self.logger.log(
            "coach_step_outcome",
            decision=decision.to_dict(),
            tactical_useful=bool(tactical_useful),
        )
        return float(outcome_reward)

    @staticmethod
    def _validate_fusion_postconditions(action_result: Dict[str, Any], placement: Dict[str, Any]) -> None:
        if not isinstance(action_result, dict):
            return
        if not isinstance(placement, dict):
            placement = {}

        reported_success = bool(
            action_result.get("fusionSucceeded")
            or action_result.get("fusion_success")
            or action_result.get("fusionOverrideApplied")
        )
        if not reported_success:
            return

        source_tile_occupied_before = bool(
            action_result.get("sourceTileOccupiedBefore")
            if "sourceTileOccupiedBefore" in action_result
            else placement.get("sourceTileOccupiedBefore")
        )
        fusion_execution_mode = str(
            action_result.get("fusionExecutionMode")
            or placement.get("fusionExecutionMode")
            or ""
        )
        duplicate_stack_detected = bool(
            action_result.get("duplicateStackDetected")
            or placement.get("duplicateStackDetected")
        )
        changed_tile_count = _safe_int(
            action_result.get("changedTileCount"),
            action_result.get("changed_tile_count"),
            placement.get("changedTileCount"),
            placement.get("changed_tile_count"),
            default=0,
        )
        changed_tiles = action_result.get("changedTiles")
        if not isinstance(changed_tiles, list):
            changed_tiles = action_result.get("changed_tiles")
        if not isinstance(changed_tiles, list):
            changed_tiles = placement.get("changedTiles")
        if not isinstance(changed_tiles, list):
            changed_tiles = placement.get("changed_tiles")
        if not isinstance(changed_tiles, list):
            changed_tiles = []
        requested_source_row = _safe_int(
            action_result.get("requestedSourceRow"),
            action_result.get("requested_source_row"),
            placement.get("requestedSourceRow"),
            placement.get("requested_source_row"),
            default=-1,
        )
        requested_source_col = _safe_int(
            action_result.get("requestedSourceCol"),
            action_result.get("requested_source_col"),
            placement.get("requestedSourceCol"),
            placement.get("requested_source_col"),
            default=-1,
        )
        non_source_tiles_changed = bool(
            action_result.get("nonSourceTilesChanged")
            if "nonSourceTilesChanged" in action_result
            else action_result.get("non_source_tiles_changed")
            if "non_source_tiles_changed" in action_result
            else placement.get("nonSourceTilesChanged")
            if "nonSourceTilesChanged" in placement
            else placement.get("non_source_tiles_changed")
        )
        global_fusion_side_effect = bool(
            action_result.get("globalFusionSideEffect")
            if "globalFusionSideEffect" in action_result
            else action_result.get("global_fusion_side_effect")
            if "global_fusion_side_effect" in action_result
            else placement.get("globalFusionSideEffect")
            if "globalFusionSideEffect" in placement
            else placement.get("global_fusion_side_effect")
        )

        changed_tile_matches_requested = False
        if changed_tile_count == 1 and isinstance(changed_tiles, list) and changed_tiles:
            first_tile = changed_tiles[0]
            if isinstance(first_tile, dict):
                changed_row = _safe_int(
                    first_tile.get("row"),
                    first_tile.get("sourceRow"),
                    first_tile.get("source_row"),
                    default=-1,
                )
                changed_col = _safe_int(
                    first_tile.get("column"),
                    first_tile.get("col"),
                    first_tile.get("sourceCol"),
                    first_tile.get("source_col"),
                    default=-1,
                )
                changed_tile_matches_requested = (
                    requested_source_row >= 0
                    and requested_source_col >= 0
                    and changed_row == requested_source_row
                    and changed_col == requested_source_col
                )

        if not non_source_tiles_changed and isinstance(changed_tiles, list):
            for tile in changed_tiles:
                if not isinstance(tile, dict):
                    continue
                tile_row = _safe_int(
                    tile.get("row"),
                    tile.get("sourceRow"),
                    tile.get("source_row"),
                    default=-1,
                )
                tile_col = _safe_int(
                    tile.get("column"),
                    tile.get("col"),
                    tile.get("sourceCol"),
                    tile.get("source_col"),
                    default=-1,
                )
                if tile_row != requested_source_row or tile_col != requested_source_col:
                    non_source_tiles_changed = True
                    break

        failure_reason = ""
        bridge_result_reason = str(
            action_result.get("bridgeResultReason")
            or placement.get("bridgeResultReason")
            or ""
        )
        if not source_tile_occupied_before:
            failure_reason = "source_tile_not_occupied"
            bridge_result_reason = bridge_result_reason or failure_reason
        elif fusion_execution_mode != "dedicated_fusion":
            failure_reason = "bridge_rejected"
            bridge_result_reason = bridge_result_reason or "fusion_not_dedicated_path"
        elif duplicate_stack_detected:
            failure_reason = "duplicate_stack_detected"
            bridge_result_reason = bridge_result_reason or failure_reason
        elif non_source_tiles_changed or global_fusion_side_effect or changed_tile_count > 1:
            failure_reason = "global_fusion_side_effect"
            bridge_result_reason = "fusion_mutated_non_source_tiles"
        elif changed_tile_count != 1 or not changed_tile_matches_requested:
            failure_reason = "fusion_no_effect"
            bridge_result_reason = bridge_result_reason or failure_reason

        if not failure_reason:
            return

        action_result["fusionSucceeded"] = False
        action_result["fusion_success"] = False
        action_result["fusionOverrideApplied"] = False
        action_result["illegalAction"] = True
        action_result["illegalReason"] = failure_reason
        action_result["fusionRejectedReason"] = failure_reason
        action_result["bridgeResultReason"] = bridge_result_reason
        action_result["bridge_result_reason"] = bridge_result_reason
        action_result["changedTileCount"] = changed_tile_count
        action_result["changed_tile_count"] = changed_tile_count
        action_result["nonSourceTilesChanged"] = bool(non_source_tiles_changed)
        action_result["non_source_tiles_changed"] = bool(non_source_tiles_changed)
        action_result["globalFusionSideEffect"] = bool(
            failure_reason == "global_fusion_side_effect" or global_fusion_side_effect
        )
        action_result["global_fusion_side_effect"] = bool(
            failure_reason == "global_fusion_side_effect" or global_fusion_side_effect
        )
        if failure_reason == "global_fusion_side_effect":
            action_result["fusionScope"] = "global_side_effect_detected"
            action_result["fusion_scope"] = "global_side_effect_detected"
        if placement:
            placement["success"] = False
            placement["illegalReason"] = failure_reason
            placement["bridgeResultReason"] = bridge_result_reason
            placement["bridge_result_reason"] = bridge_result_reason
            placement["changedTileCount"] = changed_tile_count
            placement["changed_tile_count"] = changed_tile_count
            placement["nonSourceTilesChanged"] = bool(non_source_tiles_changed)
            placement["non_source_tiles_changed"] = bool(non_source_tiles_changed)
            placement["globalFusionSideEffect"] = bool(
                failure_reason == "global_fusion_side_effect" or global_fusion_side_effect
            )
            placement["global_fusion_side_effect"] = bool(
                failure_reason == "global_fusion_side_effect" or global_fusion_side_effect
            )
            if failure_reason == "global_fusion_side_effect":
                placement["fusionScope"] = "global_side_effect_detected"
                placement["fusion_scope"] = "global_side_effect_detected"

    def _base_reward_components(self, *, match: bool, override: bool, legal_execution: bool) -> Dict[str, float]:
        components = {
            COACH_REWARD_MATCH_COMPONENT: 0.0,
            COACH_REWARD_LEGAL_EXECUTION_COMPONENT: 0.0,
            COACH_REWARD_OVERRIDE_PENALTY_COMPONENT: 0.0,
            COACH_REWARD_FUSION_SUCCESS_COMPONENT: 0.0,
            COACH_REWARD_TACTICAL_USEFULNESS_COMPONENT: 0.0,
        }
        if not self.reward_enabled:
            return components
        if legal_execution:
            components[COACH_REWARD_LEGAL_EXECUTION_COMPONENT] = self.legal_execution_reward
        if match:
            components[COACH_REWARD_MATCH_COMPONENT] = self.match_reward
        if override:
            components[COACH_REWARD_OVERRIDE_PENALTY_COMPONENT] = self.override_penalty
        return components

    def _record_fusion_probe_stats(self, validation: CoachActionValidation) -> None:
        diagnostics = validation.diagnostics if isinstance(validation.diagnostics, dict) else {}
        if not diagnostics:
            return
        self.stats.last_fusion_probe_diagnostics = _compact_fusion_probe_diagnostics(diagnostics)
        if "fusion_probe_available" in diagnostics:
            self.stats.fusion_probe_count += 1
            probe_available = bool(diagnostics.get("fusion_probe_available"))
            self.stats.fusion_bridge_available = probe_available
            if probe_available:
                self.stats.fusion_probe_available_count += 1
            else:
                self.stats.fusion_probe_unavailable_count += 1
            self.stats.last_fusion_probe_reason = str(diagnostics.get("fusion_probe_reason") or "")

    def _is_tactical_useful(self, decision: CoachDecision, info: Dict[str, Any]) -> bool:
        if not decision.enabled or decision.command is None or decision.validation is None or not decision.validation.legal:
            return False
        breakdown = info.get("reward_breakdown") if isinstance(info, dict) else {}
        if not isinstance(breakdown, dict):
            breakdown = {}
        for key in (
            "lane_response_reward",
            "threat_balanced_row_reward",
            "first_defense_undefended_threatened_row_reward",
            "first_peashooter_threatened_row_reward",
            "all_active_threatened_rows_have_peashooter_reward",
            "mower_risk_reduction_reward",
            "tough_zombie_response_reward",
            "wallnut_blocks_active_threat_reward",
            "cherrybomb_tactical_kill_reward",
            "reduce_undefended_threat_reward",
        ):
            try:
                if float(breakdown.get(key) or 0.0) > 0.0:
                    return True
            except (TypeError, ValueError):
                continue
        lane_diag = info.get("lane_diagnostics") if isinstance(info, dict) else {}
        if isinstance(lane_diag, dict):
            if bool(lane_diag.get("lane_response_reward_applied")) or bool(lane_diag.get("tough_zombie_response")):
                return True
            mower_saves = lane_diag.get("mower_saves_estimated_by_row")
            if isinstance(mower_saves, dict):
                for value in mower_saves.values():
                    try:
                        if int(value) > 0:
                            return True
                    except (TypeError, ValueError):
                        continue
        if decision.command.kind == "fuse":
            action_result = info.get("action_result") if isinstance(info, dict) else {}
            if isinstance(action_result, dict) and bool(
                action_result.get("fusionSucceeded")
                or action_result.get("fusion_success")
                or action_result.get("fusionOverrideApplied")
            ):
                return True
            candidate_diag = decision.validation.diagnostics if isinstance(decision.validation.diagnostics, dict) else {}
            return bool(
                float(candidate_diag.get("lane_danger_score") or 0.0) >= 0.35
                or int(candidate_diag.get("nearby_buckethead_count") or 0) > 0
                or int(candidate_diag.get("nearby_conehead_count") or 0) > 0
                or int(candidate_diag.get("nearby_zombie_count") or 0) >= 2
            )
        return False


def parse_coach_command(text: str, *, timestamp: Optional[float] = None, source: str = "human") -> CoachCommand:
    raw_text = str(text or "").strip()
    ts = time.time() if timestamp is None else float(timestamp)
    if not raw_text:
        return _invalid_command(raw_text, ts, source, "empty_command")
    normalized = raw_text
    if normalized.startswith("!"):
        normalized = normalized[1:].strip()
    if not normalized:
        return _invalid_command(raw_text, ts, source, "empty_command")
    kind = ""
    parts: List[str] = []
    if "(" in normalized:
        open_index = normalized.find("(")
        close_index = normalized.rfind(")")
        if open_index <= 0 or close_index != len(normalized) - 1:
            return _invalid_command(raw_text, ts, source, "malformed_command")
        kind = normalized[:open_index].strip().lower()
        arg_text = normalized[open_index + 1 : close_index].strip()
        parts = [part for part in re.split(r"[\s,]+", arg_text) if part]
    else:
        tokens = normalized.split()
        if not tokens:
            return _invalid_command(raw_text, ts, source, "empty_command")
        kind = str(tokens[0]).strip().lower()
        parts = [str(token).strip() for token in tokens[1:] if str(token).strip()]
    if kind not in COACH_COMMANDS:
        return _invalid_command(raw_text, ts, source, "unknown_command")
    try:
        if kind in {"plant", "fuse"}:
            if len(parts) != 3:
                return _invalid_command(raw_text, ts, source, f"{kind}_expects_seed_row_col")
            return CoachCommand(
                kind=kind,
                seed_index=int(parts[0]),
                row=int(parts[1]),
                col=int(parts[2]),
                raw_text=raw_text,
                timestamp=ts,
                source=source,
            )
        if kind == "defend":
            if len(parts) != 1:
                return _invalid_command(raw_text, ts, source, "defend_expects_row")
            return CoachCommand(kind=kind, row=int(parts[0]), raw_text=raw_text, timestamp=ts, source=source)
        if kind in {"economy", "wait"}:
            if len(parts) != 0:
                return _invalid_command(raw_text, ts, source, f"{kind}_expects_no_args")
            return CoachCommand(kind=kind, raw_text=raw_text, timestamp=ts, source=source)
    except ValueError:
        return _invalid_command(raw_text, ts, source, "non_integer_argument")
    return _invalid_command(raw_text, ts, source, "unknown_command")


def command_to_policy_action(
    command: CoachCommand,
    *,
    action_space_mode: str,
    rows: int = 5,
    cols: int = 10,
    max_seed_slots: Optional[int] = None,
    plant_types: Optional[Sequence[int]] = None,
) -> Tuple[Optional[int], str]:
    if not command.valid_syntax:
        return None, command.rejected_reason or "invalid_syntax"
    mode = normalize_action_space_mode(action_space_mode)
    spec = build_action_space_spec(
        mode=mode,
        plant_types=[int(value) for value in (plant_types or [])],
        max_seed_slots=max_seed_slots,
        rows=int(rows),
        cols=int(cols),
    )
    if command.kind == "wait":
        return int(spec.wait_action), ""
    if command.kind not in {"plant", "fuse"}:
        return None, "high_level_command_requires_observation"
    return _encode_slot_cell_action(
        int(command.seed_index if command.seed_index is not None else -1),
        int(command.row if command.row is not None else -1),
        int(command.col if command.col is not None else -1),
        spec=spec,
    )


def validate_coach_command(
    command: CoachCommand,
    *,
    action_space_mode: str,
    observation: Optional[Dict[str, Any]] = None,
    action_mask: Optional[Any] = None,
    plant_types: Optional[Sequence[int]] = None,
    max_seed_slots: Optional[int] = None,
    rows: int = 5,
    cols: int = 10,
    fusion_enabled: bool = False,
    fusion_bridge_probe: Optional[
        Callable[
            [CoachCommand, Dict[str, Any], Sequence[int], ActionSpaceSpec],
            Tuple[Optional[Dict[str, Any]], str, Dict[str, Any]],
        ]
    ] = None,
) -> CoachActionValidation:
    observation = observation if isinstance(observation, dict) else {}
    rows = int(observation.get("rowCount") or rows)
    cols = int(observation.get("columnCount") or cols)
    plant_types = list(plant_types or [])
    spec = build_action_space_spec(
        mode=action_space_mode,
        plant_types=[int(value) for value in plant_types],
        max_seed_slots=max_seed_slots,
        rows=rows,
        cols=cols,
    )
    if not command.valid_syntax:
        return _validation(command, False, rejected_reason=command.rejected_reason or "invalid_syntax")

    if command.kind == "fuse":
        bounds_reason = _validate_seed_cell_bounds(command, spec, observation, plant_types)
        if bounds_reason:
            return _validation(command, False, rejected_reason=bounds_reason)
        if not fusion_enabled:
            return _validation(command, False, rejected_reason=COACH_REJECTION_FUSION_DISABLED)
        compat_reason, compat_diag = _fusion_compatibility_rejection(command, observation, plant_types)
        if compat_reason:
            _log_fusion_rejection(command, compat_reason, compat_diag)
            return _validation(command, False, rejected_reason=compat_reason, diagnostics=compat_diag)
        if fusion_bridge_probe is None:
            return _validation(command, False, rejected_reason=COACH_REJECTION_FUSION_DIRECT_NOT_IMPLEMENTED)
        bridge_command, rejected_reason, diagnostics = fusion_bridge_probe(command, observation, plant_types, spec)
        if bridge_command is None:
            return _validation(
                command,
                False,
                rejected_reason=rejected_reason or COACH_REJECTION_FUSION_BRIDGE_REJECTED,
                diagnostics=diagnostics,
            )
        wait_validation = _validate_policy_action(command, int(spec.wait_action), spec, observation, action_mask, plant_types)
        if not wait_validation.legal:
            return _validation(
                command,
                False,
                policy_action=wait_validation.policy_action,
                rejected_reason=wait_validation.rejected_reason or "wait_action_illegal_for_fusion",
                decoded=wait_validation.decoded,
                diagnostics=diagnostics,
            )
        decoded = dict(wait_validation.decoded or {})
        decoded["kind"] = "fusion"
        return _validation(
            command,
            True,
            policy_action=int(spec.wait_action),
            decoded=decoded,
            bridge_command=bridge_command,
            diagnostics=diagnostics,
        )

    if command.kind == "plant":
        bounds_reason = _validate_seed_cell_bounds(command, spec, observation, plant_types)
        if bounds_reason:
            return _validation(command, False, rejected_reason=bounds_reason)
        block_reason = _placement_block_reason_for_command(command, observation)
        if block_reason:
            return _validation(command, False, rejected_reason=block_reason)
        action, reason = _encode_slot_cell_action(
            int(command.seed_index if command.seed_index is not None else -1),
            int(command.row if command.row is not None else -1),
            int(command.col if command.col is not None else -1),
            spec=spec,
        )
        if action is None:
            return _validation(command, False, rejected_reason=reason)
        return _validate_policy_action(command, action, spec, observation, action_mask, plant_types)

    if command.kind == "wait":
        return _validate_policy_action(command, int(spec.wait_action), spec, observation, action_mask, plant_types)

    if command.kind == "defend":
        row = int(command.row if command.row is not None else -1)
        if not (0 <= row < rows):
            return _validation(command, False, rejected_reason="row_out_of_bounds")
        action = _choose_high_level_action(
            kind="defend",
            row=row,
            spec=spec,
            observation=observation,
            action_mask=action_mask,
            plant_types=plant_types,
        )
        if action is None:
            return _validation(command, False, rejected_reason="no_legal_defense_action")
        return _validate_policy_action(command, action, spec, observation, action_mask, plant_types)

    if command.kind == "economy":
        action = _choose_high_level_action(
            kind="economy",
            row=None,
            spec=spec,
            observation=observation,
            action_mask=action_mask,
            plant_types=plant_types,
        )
        if action is None:
            return _validation(command, False, rejected_reason="no_legal_economy_action")
        return _validate_policy_action(command, action, spec, observation, action_mask, plant_types)

    return _validation(command, False, rejected_reason="unknown_command")


def validate_coach_command_for_env(
    command: CoachCommand,
    env: Any,
    *,
    observation: Optional[Dict[str, Any]] = None,
    action_mask: Optional[Any] = None,
    force_fusion_enabled: bool = False,
) -> CoachActionValidation:
    action_spec = getattr(env, "action_spec", None)
    mode = getattr(action_spec, "mode", "fixed")
    rows = int(getattr(env, "rows", getattr(action_spec, "rows", 5)) or 5)
    cols = int(getattr(env, "cols", getattr(action_spec, "cols", 10)) or 10)
    max_seed_slots = int(getattr(action_spec, "max_seed_slots", 0) or 0) or None
    config = getattr(env, "config", None)
    plant_types = list(getattr(config, "plant_types", []) or [])
    fusion_policy = str(getattr(config, "fusion_policy", "none") or "none")
    config_fusion_enabled = bool(getattr(config, "human_coach_fusion_enabled", False))
    fusion_enabled = bool(force_fusion_enabled or config_fusion_enabled or fusion_policy != "none")
    fusion_probe = build_env_fusion_probe(env)
    if fusion_enabled and fusion_probe is None:
        def _unavailable_probe(
            _command: CoachCommand,
            _observation: Dict[str, Any],
            _plant_types: Sequence[int],
            _spec: ActionSpaceSpec,
        ) -> Tuple[Optional[Dict[str, Any]], str, Dict[str, Any]]:
            return None, COACH_REJECTION_FUSION_BRIDGE_UNAVAILABLE, {
                "fusion_probe_available": False,
                "fusion_probe_reason": COACH_REJECTION_FUSION_BRIDGE_UNAVAILABLE,
            }

        fusion_probe = _unavailable_probe
    return validate_coach_command(
        command,
        action_space_mode=mode,
        observation=observation,
        action_mask=action_mask,
        plant_types=plant_types,
        max_seed_slots=max_seed_slots,
        rows=rows,
        cols=cols,
        fusion_enabled=fusion_enabled,
        fusion_bridge_probe=fusion_probe,
    )


def human_coach_live_status_defaults(enabled: bool = False, platform: str = "mock") -> Dict[str, Any]:
    return HumanCoachStats(enabled=enabled, platform=platform).live_status_fields()


def human_coach_live_status_from_hook(hook: Any, *, enabled: bool = False, platform: str = "mock") -> Dict[str, Any]:
    if hook is not None and hasattr(hook, "live_status_fields"):
        try:
            return dict(hook.live_status_fields())
        except Exception:
            pass
    return human_coach_live_status_defaults(enabled=enabled, platform=platform)


def _invalid_command(raw_text: str, timestamp: float, source: str, reason: str) -> CoachCommand:
    return CoachCommand(
        kind="invalid",
        raw_text=raw_text,
        timestamp=timestamp,
        source=source,
        valid_syntax=False,
        rejected_reason=reason,
    )


def _encode_slot_cell_action(seed_index: int, row: int, col: int, *, spec: ActionSpaceSpec) -> Tuple[Optional[int], str]:
    if seed_index < 0 or seed_index >= int(spec.max_seed_slots):
        return None, "seed_index_out_of_bounds"
    if row < 0 or row >= int(spec.rows):
        return None, "row_out_of_bounds"
    if col < 0 or col >= int(spec.cols):
        return None, "col_out_of_bounds"
    cells = int(spec.rows) * int(spec.cols)
    encoded = seed_index * cells + row * int(spec.cols) + col
    action = encoded if spec.mode == ACTION_SPACE_DYNAMIC_14 else encoded + 1
    if not (0 <= action < int(spec.action_count)):
        return None, "action_out_of_bounds"
    if action != int(spec.wait_action) and not (int(spec.placement_action_min) <= action <= int(spec.placement_action_max)):
        return None, "action_outside_placement_range"
    return int(action), ""


def _validate_seed_cell_bounds(
    command: CoachCommand,
    spec: ActionSpaceSpec,
    observation: Dict[str, Any],
    plant_types: Sequence[int],
) -> str:
    seed_index = int(command.seed_index if command.seed_index is not None else -1)
    row = int(command.row if command.row is not None else -1)
    col = int(command.col if command.col is not None else -1)
    if seed_index < 0 or seed_index >= int(spec.max_seed_slots):
        return "seed_index_out_of_bounds"
    active_slots = _active_seed_slot_count(observation, plant_types, int(spec.max_seed_slots))
    if active_slots is not None and seed_index >= active_slots:
        return "seed_index_inactive"
    if row < 0 or row >= int(spec.rows):
        return "row_out_of_bounds"
    if col < 0 or col >= int(spec.cols):
        return "col_out_of_bounds"
    return ""


def _active_seed_slot_count(
    observation: Dict[str, Any],
    plant_types: Sequence[int],
    max_seed_slots: int,
) -> Optional[int]:
    slots = observation.get("seedSlots")
    if isinstance(slots, list):
        return max(0, min(max_seed_slots, len(slots)))
    raw_count = observation.get("seedSlotCount")
    if raw_count is not None:
        try:
            return max(0, min(max_seed_slots, int(raw_count)))
        except (TypeError, ValueError):
            return None
    if plant_types:
        return max(0, min(max_seed_slots, len(plant_types)))
    return None


def _placement_block_reason_for_command(command: CoachCommand, observation: Dict[str, Any]) -> str:
    seed_index = int(command.seed_index if command.seed_index is not None else -1)
    row = int(command.row if command.row is not None else -1)
    col = int(command.col if command.col is not None else -1)
    slot = _seed_slot_for_index(observation, seed_index=seed_index)
    slot_block_reason = _seed_slot_block_reason(observation, slot)
    if slot_block_reason:
        return slot_block_reason
    if _cell_occupied_for_validation(observation, row=row, col=col):
        return COACH_REJECTION_OCCUPIED_CELL
    return ""


def _seed_slot_for_index(observation: Dict[str, Any], *, seed_index: int) -> Optional[Dict[str, Any]]:
    slots = observation.get("seedSlots")
    if not isinstance(slots, list):
        return None
    if not (0 <= int(seed_index) < len(slots)):
        return None
    slot = slots[int(seed_index)]
    if not isinstance(slot, dict):
        return None
    return slot


def _seed_slot_block_reason(observation: Dict[str, Any], slot: Optional[Dict[str, Any]]) -> str:
    if not isinstance(slot, dict):
        return ""
    if not bool(slot.get("usable", True)):
        return COACH_REJECTION_SLOT_NOT_USABLE
    if bool(slot.get("disabled", False)):
        return COACH_REJECTION_SLOT_DISABLED
    if not bool(slot.get("ready", True)):
        return COACH_REJECTION_COOLDOWN_NOT_READY
    current_cooldown = _safe_float(slot.get("currentCooldown"), default=0.0)
    full_cooldown = _safe_float(slot.get("fullCooldown"), default=0.0)
    if full_cooldown > 0.05 and current_cooldown > 0.05:
        return COACH_REJECTION_COOLDOWN_NOT_READY
    sun = max(0, _safe_int(observation.get("sun"), default=0))
    cost = max(0, _safe_int(slot.get("seedCost"), default=0))
    if sun < cost:
        return COACH_REJECTION_INSUFFICIENT_SUN
    return ""


def _fusion_compatibility_rejection(
    command: CoachCommand,
    observation: Dict[str, Any],
    plant_types: Sequence[int],
) -> Tuple[str, Dict[str, Any]]:
    """Centralized existing-plant/seed compatibility gate for fuse commands.

    Returns (reason, diagnostics).  Uses the shared pvzrl_fusion compatibility
    table so manual and stream coaches reject exactly the pairs the model action
    mask blocks.  Transient resource blocks (cooldown/sun) are intentionally
    deferred to the bridge-probe path so they can become pending retries.
    """

    row = int(command.row if command.row is not None else -1)
    col = int(command.col if command.col is not None else -1)
    seed_index = int(command.seed_index if command.seed_index is not None else -1)
    selected_type = seed_plant_type_for_slot(observation, seed_index, plant_types)
    existing_type = plant_type_at_cell(observation, row, col)
    has_board = isinstance(observation.get("plants"), list) or isinstance(
        observation.get("visiblePlants"), list
    )
    diagnostics: Dict[str, Any] = {
        "fusion_existing_plant_type": int(existing_type) if existing_type is not None else -1,
        "fusion_existing_plant_name": fusion_plant_name(existing_type) if existing_type is not None else "",
        "fusion_selected_seed_type": int(selected_type) if selected_type is not None else -1,
        "fusion_selected_seed_name": fusion_plant_name(selected_type) if selected_type is not None else "",
        "fusion_row": row,
        "fusion_col": col,
    }
    if existing_type is None:
        # Only assert emptiness when the board is readable; otherwise defer to bridge.
        if has_board:
            return COACH_REJECTION_FUSION_EMPTY_TILE, diagnostics
        return "", diagnostics
    if selected_type is None:
        return "", diagnostics  # cannot resolve the seed's plant type; let the bridge decide
    if not are_fusion_compatible(existing_type, selected_type):
        diagnostics["fusion_incompatible_pair"] = {
            "existing": diagnostics["fusion_existing_plant_name"],
            "selected": diagnostics["fusion_selected_seed_name"],
            "row": row,
            "col": col,
        }
        return COACH_REJECTION_FUSION_INCOMPATIBLE, diagnostics
    return "", diagnostics


def _log_fusion_rejection(command: CoachCommand, reason: str, diagnostics: Dict[str, Any]) -> None:
    existing = str(diagnostics.get("fusion_existing_plant_name") or "") or "none"
    selected = str(diagnostics.get("fusion_selected_seed_name") or "") or "none"
    # Logged to stderr so machine-readable stdout (e.g. live-status JSON) stays clean.
    print(
        f"[coach] fusion rejected: {reason} existing={existing} selected={selected} "
        f"row={diagnostics.get('fusion_row')} col={diagnostics.get('fusion_col')}",
        file=sys.stderr,
    )


def _cell_occupied_for_validation(observation: Dict[str, Any], *, row: int, col: int) -> bool:
    for key in ("plants", "visiblePlants"):
        values = observation.get(key)
        if not isinstance(values, list):
            continue
        for plant in values:
            if not isinstance(plant, dict):
                continue
            if key == "visiblePlants" and (
                not bool(plant.get("activeInHierarchy", True))
                or not bool(plant.get("inBoardBounds", True))
            ):
                continue
            prow = _safe_int(plant.get("row"), default=-1)
            pcol = _safe_int(plant.get("column"), plant.get("col"), default=-1)
            if prow == int(row) and pcol == int(col):
                return True
    return False


def _validate_policy_action(
    command: CoachCommand,
    action: int,
    spec: ActionSpaceSpec,
    observation: Dict[str, Any],
    action_mask: Optional[Any],
    plant_types: Sequence[int],
) -> CoachActionValidation:
    action = int(action)
    if action < 0 or action >= int(spec.action_count):
        return _validation(command, False, policy_action=action, rejected_reason="action_out_of_bounds")
    if not _has_legality_signal(action_mask, observation):
        if action == int(spec.wait_action):
            decoded = decode_policy_action(
                action,
                mode=spec.mode,
                observation=observation,
                plant_types=list(plant_types),
                max_seed_slots=spec.max_seed_slots,
                rows=spec.rows,
                cols=spec.cols,
            )
            return _validation(command, True, policy_action=action, decoded=decoded)
        return _validation(command, False, policy_action=action, rejected_reason="missing_legality_signal")
    if not _policy_action_allowed(action, spec, observation, action_mask):
        return _validation(command, False, policy_action=action, rejected_reason="illegal_action")
    decoded = decode_policy_action(
        action,
        mode=spec.mode,
        observation=observation,
        plant_types=list(plant_types),
        max_seed_slots=spec.max_seed_slots,
        rows=spec.rows,
        cols=spec.cols,
    )
    if int(decoded.get("kind", -1)) == -1:
        return _validation(command, False, policy_action=action, rejected_reason="invalid_action_decode")
    return _validation(command, True, policy_action=action, decoded=decoded)


def _choose_high_level_action(
    *,
    kind: str,
    row: Optional[int],
    spec: ActionSpaceSpec,
    observation: Dict[str, Any],
    action_mask: Optional[Any],
    plant_types: Sequence[int],
) -> Optional[int]:
    candidates: List[Tuple[int, int, int]] = []
    for action in sorted(_legal_policy_actions(spec, observation, action_mask)):
        if action == int(spec.wait_action):
            continue
        decoded = decode_policy_action(
            action,
            mode=spec.mode,
            observation=observation,
            plant_types=list(plant_types),
            max_seed_slots=spec.max_seed_slots,
            rows=spec.rows,
            cols=spec.cols,
        )
        if int(decoded.get("kind", -1)) != 1:
            continue
        decoded_row = int(decoded.get("row", -1))
        plant_type = int(decoded.get("plant_type", -1))
        col = int(decoded.get("column", -1))
        if kind == "defend":
            if row is not None and decoded_row != row:
                continue
            priority = {0: 0, 3: 1, 2: 2, 1: 4}.get(plant_type, 3)
            candidates.append((priority, col, action))
        elif kind == "economy" and plant_type == 1:
            candidates.append((decoded_row, col, action))
    if not candidates:
        return None
    candidates.sort()
    return int(candidates[0][2])


def _legal_policy_actions(spec: ActionSpaceSpec, observation: Dict[str, Any], action_mask: Optional[Any]) -> Set[int]:
    result: Set[int] = set()
    if action_mask is not None:
        try:
            for index, allowed in enumerate(action_mask):
                if bool(allowed):
                    result.add(int(index))
        except TypeError:
            pass
    raw_legal = observation.get("legalActions")
    if isinstance(raw_legal, list):
        for raw_action in raw_legal:
            try:
                result.add(int(legacy_action_to_policy_action(int(raw_action), mode=spec.mode)))
            except Exception:
                continue
    return {action for action in result if 0 <= action < int(spec.action_count)}


def _policy_action_allowed(
    action: int,
    spec: ActionSpaceSpec,
    observation: Dict[str, Any],
    action_mask: Optional[Any],
) -> bool:
    if action_mask is not None:
        try:
            if 0 <= action < len(action_mask):
                return bool(action_mask[action])
        except TypeError:
            pass
    return action in _legal_policy_actions(spec, observation, action_mask)


def _has_legality_signal(action_mask: Optional[Any], observation: Dict[str, Any]) -> bool:
    if action_mask is not None:
        try:
            return len(action_mask) > 0
        except TypeError:
            return False
    return isinstance(observation.get("legalActions"), list)


def _is_pending_retry_reason(command: CoachCommand, reason: str) -> bool:
    if command.kind not in {"plant", "fuse"}:
        return False
    normalized = str(reason or "").strip().lower()
    return normalized in COACH_PENDING_RETRY_REASONS


def _validation(
    command: CoachCommand,
    legal: bool,
    *,
    policy_action: Optional[int] = None,
    rejected_reason: str = "",
    decoded: Optional[Dict[str, Any]] = None,
    bridge_command: Optional[Dict[str, Any]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> CoachActionValidation:
    return CoachActionValidation(
        command=command,
        legal=bool(legal),
        policy_action=policy_action,
        rejected_reason=str(rejected_reason or ""),
        decoded=dict(decoded or {}),
        bridge_command=bridge_command,
        diagnostics=dict(diagnostics or {}),
    )


def build_env_fusion_probe(
    env: Any,
) -> Optional[
    Callable[
        [CoachCommand, Dict[str, Any], Sequence[int], ActionSpaceSpec],
        Tuple[Optional[Dict[str, Any]], str, Dict[str, Any]],
    ]
]:
    base = getattr(env, "base", None)
    client = getattr(base, "client", None) if base is not None else None
    if client is None or not hasattr(client, "request"):
        return None

    def _probe(
        command: CoachCommand,
        observation: Dict[str, Any],
        plant_types: Sequence[int],
        _spec: ActionSpaceSpec,
    ) -> Tuple[Optional[Dict[str, Any]], str, Dict[str, Any]]:
        diagnostics: Dict[str, Any] = {
            "fusion_probe_available": False,
            "fusion_probe_reason": "",
            "requested_seed_slot_index": int(command.seed_index if command.seed_index is not None else -1),
            "requested_row": int(command.row if command.row is not None else -1),
            "requested_col": int(command.col if command.col is not None else -1),
        }
        try:
            probe = client.request("fusion_probe", return_observation=False)
            diagnostics["fusion_probe_available"] = True
        except Exception as exc:
            diagnostics["fusion_probe_reason"] = str(exc)
            return None, COACH_REJECTION_FUSION_PROBE_FAILED, diagnostics
        seed_index = int(command.seed_index if command.seed_index is not None else -1)
        row = int(command.row if command.row is not None else -1)
        col = int(command.col if command.col is not None else -1)
        probe_candidates = probe.get("fusionCandidates") if isinstance(probe, dict) else None
        if isinstance(probe_candidates, list):
            diagnostics["fusion_probe_candidate_count"] = int(len(probe_candidates))
        source = _find_source_plant_at_cell(observation, row=row, col=col)
        if source is None:
            diagnostics["fusion_probe_reason"] = COACH_REJECTION_FUSION_SOURCE_NOT_FOUND
            diagnostics["source_found"] = False
            return None, COACH_REJECTION_FUSION_SOURCE_NOT_FOUND, diagnostics
        diagnostics["source_found"] = True
        ingredient_type, slot = _ingredient_type_for_seed_slot(
            observation,
            plant_types=plant_types,
            seed_index=seed_index,
        )
        resolved_seed_slot_index = int(seed_index)
        if isinstance(slot, dict):
            resolved_seed_slot_index = _safe_int(
                slot.get("slotIndex"),
                slot.get("seedSlotIndex"),
                slot.get("index"),
                default=seed_index,
            )
        source_type = _safe_int(source.get("type"), source.get("plantType"), default=-1)
        source_name = str(source.get("typeName") or source.get("plantTypeName") or "")
        ingredient_name = ""
        if isinstance(slot, dict):
            ingredient_name = str(slot.get("plantTypeName") or "")
        diagnostics["source_plant_type"] = int(source_type)
        diagnostics["source_plant_name"] = source_name
        diagnostics["ingredient_plant_type"] = int(ingredient_type)
        diagnostics["ingredient_plant_name"] = ingredient_name
        diagnostics["resolved_seed_slot_index"] = int(resolved_seed_slot_index)
        diagnostics["duplicate_slot_fallback_attempted"] = False
        diagnostics["duplicate_slot_fallback_applied"] = False
        diagnostics["duplicate_slot_fallback_from_seed_slot_index"] = None
        diagnostics["duplicate_slot_fallback_to_seed_slot_index"] = None
        diagnostics["duplicate_slot_fallback_from_runtime_slot_index"] = None
        diagnostics["duplicate_slot_fallback_to_runtime_slot_index"] = None
        if ingredient_type < 0:
            diagnostics["fusion_probe_reason"] = COACH_REJECTION_FUSION_TARGET_NOT_AVAILABLE
            return None, COACH_REJECTION_FUSION_TARGET_NOT_AVAILABLE, diagnostics

        source_instance_id = _safe_int(source.get("instanceId"), source.get("instanceID"), default=0)
        candidate = _find_probe_fusion_candidate(
            probe=probe if isinstance(probe, dict) else {},
            row=row,
            col=col,
            source_type=source_type,
            source_instance_id=source_instance_id,
            ingredient_seed_slot_index=seed_index,
            ingredient_runtime_slot_index=resolved_seed_slot_index,
            ingredient_plant_type=ingredient_type,
        )
        slot_block_reason = _seed_slot_block_reason(observation, slot)
        initial_candidate = candidate if isinstance(candidate, dict) else None
        slot_probe_rows = _build_fusion_slot_probe_rows(
            observation,
            probe=probe if isinstance(probe, dict) else {},
            requested_seed_slot_index=seed_index,
            requested_runtime_slot_index=resolved_seed_slot_index,
            source_row=row,
            source_col=col,
            source_type=source_type,
            source_instance_id=source_instance_id,
            ingredient_plant_type=ingredient_type,
            initial_candidate=initial_candidate,
        )
        if slot_probe_rows:
            diagnostics["candidate_slots_checked"] = [_public_slot_probe_row(item) for item in slot_probe_rows]

        selected_probe_row: Optional[Dict[str, Any]] = None
        fallback_candidates: List[Dict[str, Any]] = []
        for row_item in slot_probe_rows:
            if int(row_item.get("seed_slot_index", -1)) == int(seed_index):
                selected_probe_row = row_item
                break
        if selected_probe_row is None:
            for row_item in slot_probe_rows:
                if int(row_item.get("runtime_seed_slot_index", -1)) == int(resolved_seed_slot_index):
                    selected_probe_row = row_item
                    break

        if selected_probe_row is not None:
            diagnostics["selected_slot_probe"] = _public_slot_probe_row(selected_probe_row)
            selected_reason = str(selected_probe_row.get("slot_block_reason") or "")
            selected_legal = bool(selected_probe_row.get("fusion_legal"))
            if selected_reason or not selected_legal:
                diagnostics["duplicate_slot_fallback_attempted"] = True
                for row_item in slot_probe_rows:
                    if int(row_item.get("seed_slot_index", -1)) == int(selected_probe_row.get("seed_slot_index", -1)):
                        continue
                    if str(row_item.get("slot_block_reason") or ""):
                        continue
                    if not bool(row_item.get("fusion_legal")):
                        continue
                    fallback_candidates.append(row_item)
                if fallback_candidates:
                    fallback = fallback_candidates[0]
                    diagnostics["duplicate_slot_fallback_applied"] = True
                    diagnostics["duplicate_slot_fallback_from_seed_slot_index"] = int(
                        selected_probe_row.get("seed_slot_index", -1)
                    )
                    diagnostics["duplicate_slot_fallback_to_seed_slot_index"] = int(
                        fallback.get("seed_slot_index", -1)
                    )
                    diagnostics["duplicate_slot_fallback_from_runtime_slot_index"] = int(
                        selected_probe_row.get("runtime_seed_slot_index", -1)
                    )
                    diagnostics["duplicate_slot_fallback_to_runtime_slot_index"] = int(
                        fallback.get("runtime_seed_slot_index", -1)
                    )
                    selected_probe_row = fallback

        selected_candidate = (
            selected_probe_row.get("_candidate_obj")
            if isinstance(selected_probe_row, dict)
            else candidate
        )
        matched_seed_slot_index = _safe_int(
            selected_probe_row.get("runtime_seed_slot_index") if isinstance(selected_probe_row, dict) else None,
            selected_candidate.get("ingredientSeedSlotIndex") if isinstance(selected_candidate, dict) else None,
            selected_candidate.get("seedSlotIndex") if isinstance(selected_candidate, dict) else None,
            selected_candidate.get("ingredient_seed_slot_index") if isinstance(selected_candidate, dict) else None,
            default=resolved_seed_slot_index,
        )
        if matched_seed_slot_index < 0:
            matched_seed_slot_index = int(resolved_seed_slot_index)
        diagnostics["matched_seed_slot_index"] = int(matched_seed_slot_index)
        diagnostics["seed_slot_index_mismatch"] = bool(int(seed_index) != int(matched_seed_slot_index))

        blocked_reason = ""
        legal = False
        if isinstance(selected_probe_row, dict):
            blocked_reason = str(selected_probe_row.get("slot_block_reason") or selected_probe_row.get("fusion_blocked_reason") or "")
            legal = bool(selected_probe_row.get("fusion_legal")) and not blocked_reason
        elif isinstance(selected_candidate, dict):
            legal = bool(
                selected_candidate.get("fusionLegal")
                if "fusionLegal" in selected_candidate
                else selected_candidate.get("fusion_legal")
            )
            blocked_reason = str(
                selected_candidate.get("fusionBlockedReason")
                or selected_candidate.get("fusion_blocked_reason")
                or selected_candidate.get("blockedReason")
                or ""
            )
        elif slot_block_reason:
            blocked_reason = str(slot_block_reason)

        if not legal:
            rejection_reason = blocked_reason or COACH_REJECTION_FUSION_BRIDGE_REJECTED
            diagnostics["fusion_probe_reason"] = rejection_reason
            diagnostics["bridge_rejection_reason"] = rejection_reason
            diagnostics["requested_slot_block_reason"] = str(slot_block_reason or "")
            diagnostics["fusion_probe_candidate"] = _compact_probe_candidate(selected_candidate) if isinstance(selected_candidate, dict) else None
            diagnostics["fusion_probe_nearby_candidates"] = _nearby_fusion_probe_candidates(
                probe=probe if isinstance(probe, dict) else {},
                row=row,
                col=col,
                ingredient_seed_slot_index=seed_index,
                ingredient_runtime_slot_index=resolved_seed_slot_index,
            )
            return None, rejection_reason, diagnostics

        predicted_result_type = _safe_int(
            selected_candidate.get("predictedResultType") if isinstance(selected_candidate, dict) else None,
            selected_candidate.get("resultPlantType") if isinstance(selected_candidate, dict) else None,
            selected_candidate.get("predicted_result_type") if isinstance(selected_candidate, dict) else None,
            default=-1,
        )
        predicted_result_name = ""
        if isinstance(selected_candidate, dict):
            predicted_result_name = str(
                selected_candidate.get("predictedResultName")
                or selected_candidate.get("resultPlantName")
                or selected_candidate.get("predicted_result_name")
                or ""
            )
        predicted_result_resolution_source = ""
        mix_lookup_found = False
        mix_lookup_key = ""
        if isinstance(selected_candidate, dict):
            predicted_result_resolution_source = str(
                selected_candidate.get("predictedResultResolutionSource")
                or selected_candidate.get("predicted_result_resolution_source")
                or ""
            )
            mix_lookup_found = bool(
                selected_candidate.get("mixLookupFound")
                if "mixLookupFound" in selected_candidate
                else selected_candidate.get("mix_lookup_found")
            )
            mix_lookup_key = str(
                selected_candidate.get("mixLookupKey")
                or selected_candidate.get("mix_lookup_key")
                or ""
            )
        candidate_payload = {
            "source_plant_type": source_type,
            "source_row": row,
            "source_col": col,
            "source_instance_id": source_instance_id,
            "ingredient_seed_slot_index": int(matched_seed_slot_index),
            "target_or_ingredient_type": ingredient_type,
            "predicted_result_type": predicted_result_type,
            "predicted_result_name": predicted_result_name,
            "predicted_result_resolution_source": predicted_result_resolution_source,
            "mix_lookup_found": bool(mix_lookup_found),
            "mix_lookup_key": mix_lookup_key,
            "fusion_legal": True,
            "fusion_blocked_reason": "",
        }
        if isinstance(selected_candidate, dict):
            candidate_payload.update(_fusion_tactical_metrics(observation, row=row, col=col, candidate=selected_candidate))
        diagnostics.update(
            {
                "fusion_probe_reason": "",
                "fusion_candidate": dict(candidate_payload),
                "bridge_rejection_reason": "",
                "lane_danger_score": float(candidate_payload.get("lane_danger_score") or 0.0),
                "nearby_zombie_count": int(candidate_payload.get("nearby_zombie_count") or 0),
                "nearby_buckethead_count": int(candidate_payload.get("nearby_buckethead_count") or 0),
                "nearby_conehead_count": int(candidate_payload.get("nearby_conehead_count") or 0),
                "estimated_mower_save": bool(candidate_payload.get("estimated_mower_save")),
                "predicted_result_resolution_source": predicted_result_resolution_source,
                "mix_lookup_found": bool(mix_lookup_found),
                "mix_lookup_key": mix_lookup_key,
            }
        )
        bridge_command = {
            "command": "fusion_step",
            "coach_command_id": int(command.coach_command_id),
            "coach_command_timestamp": float(command.timestamp),
            "coach_command_source": str(command.source or ""),
            "executed_from_fresh_coach_command": True,
            "source_instance_id": source_instance_id,
            "source_row": row,
            "source_col": col,
            "source_plant_type": source_type,
            "ingredient_seed_slot_index": int(matched_seed_slot_index),
            "ingredient_plant_type": ingredient_type,
            "predicted_result_type": predicted_result_type,
            "predicted_result_name": predicted_result_name,
            "candidate": candidate_payload,
        }
        return bridge_command, "", diagnostics

    return _probe


def _find_source_plant_at_cell(observation: Dict[str, Any], *, row: int, col: int) -> Optional[Dict[str, Any]]:
    plants = observation.get("plants")
    if not isinstance(plants, list):
        return None
    for plant in plants:
        if not isinstance(plant, dict):
            continue
        prow = _safe_int(plant.get("row"), default=-1)
        pcol = _safe_int(plant.get("column"), plant.get("col"), default=-1)
        if prow == row and pcol == col:
            return plant
    return None


def _ingredient_type_for_seed_slot(
    observation: Dict[str, Any],
    *,
    plant_types: Sequence[int],
    seed_index: int,
) -> Tuple[int, Optional[Dict[str, Any]]]:
    slots = observation.get("seedSlots")
    slot_obj: Optional[Dict[str, Any]] = None
    ingredient_type = -1
    if isinstance(slots, list) and 0 <= seed_index < len(slots):
        raw_slot = slots[seed_index]
        if isinstance(raw_slot, dict):
            slot_obj = raw_slot
            ingredient_type = _safe_int(raw_slot.get("plantType"), default=-1)
    if ingredient_type < 0 and 0 <= seed_index < len(plant_types):
        try:
            ingredient_type = int(plant_types[seed_index])
        except (TypeError, ValueError):
            ingredient_type = -1
    return ingredient_type, slot_obj


def _build_fusion_slot_probe_rows(
    observation: Dict[str, Any],
    *,
    probe: Dict[str, Any],
    requested_seed_slot_index: int,
    requested_runtime_slot_index: int,
    source_row: int,
    source_col: int,
    source_type: int,
    source_instance_id: int,
    ingredient_plant_type: int,
    initial_candidate: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    slots = observation.get("seedSlots")
    if not isinstance(slots, list):
        return []
    sun_available = max(0, _safe_int(observation.get("sun"), default=0))
    rows: List[Dict[str, Any]] = []
    for list_index, slot in enumerate(slots):
        if not isinstance(slot, dict):
            continue
        slot_plant_type = _safe_int(slot.get("plantType"), default=-1)
        if slot_plant_type != int(ingredient_plant_type):
            continue
        runtime_slot_index = _safe_int(
            slot.get("slotIndex"),
            slot.get("seedSlotIndex"),
            slot.get("index"),
            default=list_index,
        )
        slot_block_reason = _seed_slot_block_reason(observation, slot)
        seed_cost = max(0, _safe_int(slot.get("seedCost"), default=0))
        candidate = initial_candidate if int(list_index) == int(requested_seed_slot_index) else None
        if candidate is None:
            candidate = _find_probe_fusion_candidate(
                probe=probe if isinstance(probe, dict) else {},
                row=source_row,
                col=source_col,
                source_type=source_type,
                source_instance_id=source_instance_id,
                ingredient_seed_slot_index=int(list_index),
                ingredient_runtime_slot_index=int(runtime_slot_index),
                ingredient_plant_type=int(ingredient_plant_type),
            )
        fusion_legal = False
        fusion_blocked_reason = ""
        if isinstance(candidate, dict):
            fusion_legal = bool(candidate.get("fusionLegal") if "fusionLegal" in candidate else candidate.get("fusion_legal"))
            fusion_blocked_reason = str(
                candidate.get("fusionBlockedReason")
                or candidate.get("fusion_blocked_reason")
                or candidate.get("blockedReason")
                or ""
            )
        if not fusion_legal and not fusion_blocked_reason and candidate is None:
            fusion_blocked_reason = COACH_REJECTION_FUSION_BRIDGE_REJECTED
        rows.append(
            {
                "seed_slot_index": int(list_index),
                "runtime_seed_slot_index": int(runtime_slot_index),
                "requested_seed_slot": bool(int(list_index) == int(requested_seed_slot_index)),
                "requested_runtime_slot": bool(int(runtime_slot_index) == int(requested_runtime_slot_index)),
                "plant_type": int(slot_plant_type),
                "plant_name": str(slot.get("plantTypeName") or ""),
                "ready": bool(slot.get("ready", True)),
                "usable": bool(slot.get("usable", True)),
                "disabled": bool(slot.get("disabled", False)),
                "current_cooldown": float(_safe_float(slot.get("currentCooldown"), default=0.0)),
                "full_cooldown": float(_safe_float(slot.get("fullCooldown"), default=0.0)),
                "seed_cost": int(seed_cost),
                "sun_available": int(sun_available),
                "affordable": bool(sun_available >= seed_cost),
                "slot_block_reason": str(slot_block_reason or ""),
                "candidate_found": bool(isinstance(candidate, dict)),
                "fusion_legal": bool(fusion_legal),
                "fusion_blocked_reason": str(fusion_blocked_reason or ""),
                "candidate": _compact_probe_candidate(candidate) if isinstance(candidate, dict) else None,
                "_candidate_obj": candidate,
            }
        )
    rows.sort(key=lambda item: (0 if item.get("requested_seed_slot") else 1, int(item.get("seed_slot_index", 0))))
    return rows


def _public_slot_probe_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "seed_slot_index": int(row.get("seed_slot_index", -1)),
        "runtime_seed_slot_index": int(row.get("runtime_seed_slot_index", -1)),
        "requested_seed_slot": bool(row.get("requested_seed_slot")),
        "requested_runtime_slot": bool(row.get("requested_runtime_slot")),
        "plant_type": int(row.get("plant_type", -1)),
        "plant_name": str(row.get("plant_name") or ""),
        "ready": bool(row.get("ready")),
        "usable": bool(row.get("usable")),
        "disabled": bool(row.get("disabled")),
        "current_cooldown": float(_safe_float(row.get("current_cooldown"), default=0.0)),
        "full_cooldown": float(_safe_float(row.get("full_cooldown"), default=0.0)),
        "seed_cost": int(row.get("seed_cost", 0)),
        "sun_available": int(row.get("sun_available", 0)),
        "affordable": bool(row.get("affordable")),
        "slot_block_reason": str(row.get("slot_block_reason") or ""),
        "candidate_found": bool(row.get("candidate_found")),
        "fusion_legal": bool(row.get("fusion_legal")),
        "fusion_blocked_reason": str(row.get("fusion_blocked_reason") or ""),
        "candidate": row.get("candidate") if isinstance(row.get("candidate"), dict) else None,
    }


def _compact_fusion_probe_diagnostics(diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(diagnostics, dict):
        return {}
    keys = (
        "fusion_probe_available",
        "fusion_probe_reason",
        "bridge_rejection_reason",
        "requested_seed_slot_index",
        "requested_row",
        "requested_col",
        "resolved_seed_slot_index",
        "matched_seed_slot_index",
        "seed_slot_index_mismatch",
        "source_found",
        "source_plant_type",
        "source_plant_name",
        "ingredient_plant_type",
        "ingredient_plant_name",
        "predicted_result_resolution_source",
        "mix_lookup_found",
        "mix_lookup_key",
        "requested_slot_block_reason",
        "duplicate_slot_fallback_attempted",
        "duplicate_slot_fallback_applied",
        "duplicate_slot_fallback_from_seed_slot_index",
        "duplicate_slot_fallback_to_seed_slot_index",
        "duplicate_slot_fallback_from_runtime_slot_index",
        "duplicate_slot_fallback_to_runtime_slot_index",
        "fusion_probe_candidate_count",
    )
    payload = {key: _jsonable(diagnostics.get(key)) for key in keys if key in diagnostics}
    selected_slot_probe = diagnostics.get("selected_slot_probe")
    if isinstance(selected_slot_probe, dict):
        payload["selected_slot_probe"] = _jsonable(selected_slot_probe)
    slot_rows = diagnostics.get("candidate_slots_checked")
    if isinstance(slot_rows, list):
        payload["candidate_slots_checked"] = [_jsonable(item) for item in slot_rows[:8] if isinstance(item, dict)]
    candidate = diagnostics.get("fusion_probe_candidate")
    if isinstance(candidate, dict):
        payload["fusion_probe_candidate"] = _jsonable(candidate)
    return payload


def _find_probe_fusion_candidate(
    *,
    probe: Dict[str, Any],
    row: int,
    col: int,
    source_type: int,
    source_instance_id: int,
    ingredient_seed_slot_index: int,
    ingredient_runtime_slot_index: int,
    ingredient_plant_type: int,
) -> Optional[Dict[str, Any]]:
    raw_candidates = probe.get("fusionCandidates") or probe.get("fusion_candidates") or []
    if not isinstance(raw_candidates, list):
        return None
    slot_candidates: Set[int] = {int(ingredient_seed_slot_index)}
    if int(ingredient_runtime_slot_index) >= 0:
        slot_candidates.add(int(ingredient_runtime_slot_index))
    best_same_instance: Optional[Dict[str, Any]] = None
    best_same_cell_same_types: Optional[Dict[str, Any]] = None
    for candidate in raw_candidates:
        if not isinstance(candidate, dict):
            continue
        c_row = _safe_int(candidate.get("sourceRow"), candidate.get("source_row"), default=-1)
        c_col = _safe_int(
            candidate.get("sourceCol"),
            candidate.get("sourceColumn"),
            candidate.get("source_col"),
            default=-1,
        )
        c_source_type = _safe_int(candidate.get("sourcePlantType"), candidate.get("source_plant_type"), default=-1)
        c_slot = _safe_int(
            candidate.get("ingredientSeedSlotIndex"),
            candidate.get("seedSlotIndex"),
            candidate.get("ingredient_seed_slot_index"),
            default=-1,
        )
        c_ingredient = _safe_int(
            candidate.get("ingredientPlantType"),
            candidate.get("targetPlantType"),
            candidate.get("ingredient_plant_type"),
            default=-1,
        )
        same_cell = c_row == row and c_col == col
        source_type_match = not (c_source_type >= 0 and source_type >= 0 and c_source_type != source_type)
        ingredient_type_match = not (c_ingredient >= 0 and ingredient_plant_type >= 0 and c_ingredient != ingredient_plant_type)
        slot_match = c_slot in slot_candidates
        if same_cell and slot_match and source_type_match and ingredient_type_match:
            return candidate
        c_source_instance = _safe_int(
            candidate.get("sourceInstanceId"),
            candidate.get("source_instance_id"),
            default=0,
        )
        if (
            best_same_instance is None
            and source_instance_id > 0
            and c_source_instance == source_instance_id
            and slot_match
            and source_type_match
            and ingredient_type_match
        ):
            best_same_instance = candidate
        if best_same_cell_same_types is None and same_cell and source_type_match and ingredient_type_match:
            best_same_cell_same_types = candidate
    if best_same_instance is not None:
        return best_same_instance
    return best_same_cell_same_types


def _nearby_fusion_probe_candidates(
    *,
    probe: Dict[str, Any],
    row: int,
    col: int,
    ingredient_seed_slot_index: int,
    ingredient_runtime_slot_index: int,
) -> List[Dict[str, Any]]:
    raw_candidates = probe.get("fusionCandidates") or probe.get("fusion_candidates") or []
    if not isinstance(raw_candidates, list):
        return []
    slot_candidates: Set[int] = {int(ingredient_seed_slot_index)}
    if int(ingredient_runtime_slot_index) >= 0:
        slot_candidates.add(int(ingredient_runtime_slot_index))
    matches: List[Dict[str, Any]] = []
    for candidate in raw_candidates:
        if not isinstance(candidate, dict):
            continue
        c_row = _safe_int(candidate.get("sourceRow"), candidate.get("source_row"), default=-1)
        c_col = _safe_int(
            candidate.get("sourceCol"),
            candidate.get("sourceColumn"),
            candidate.get("source_col"),
            default=-1,
        )
        if c_row != int(row) or c_col != int(col):
            continue
        c_slot = _safe_int(
            candidate.get("ingredientSeedSlotIndex"),
            candidate.get("seedSlotIndex"),
            candidate.get("ingredient_seed_slot_index"),
            default=-1,
        )
        if c_slot not in slot_candidates:
            continue
        matches.append(_compact_probe_candidate(candidate))
        if len(matches) >= 3:
            break
    return matches


def _compact_probe_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source_row": _safe_int(candidate.get("sourceRow"), candidate.get("source_row"), default=-1),
        "source_col": _safe_int(
            candidate.get("sourceCol"),
            candidate.get("sourceColumn"),
            candidate.get("source_col"),
            default=-1,
        ),
        "source_plant_type": _safe_int(candidate.get("sourcePlantType"), candidate.get("source_plant_type"), default=-1),
        "ingredient_seed_slot_index": _safe_int(
            candidate.get("ingredientSeedSlotIndex"),
            candidate.get("seedSlotIndex"),
            candidate.get("ingredient_seed_slot_index"),
            default=-1,
        ),
        "ingredient_plant_type": _safe_int(
            candidate.get("ingredientPlantType"),
            candidate.get("targetPlantType"),
            candidate.get("ingredient_plant_type"),
            default=-1,
        ),
        "predicted_result_type": _safe_int(
            candidate.get("predictedResultType"),
            candidate.get("resultPlantType"),
            candidate.get("predicted_result_type"),
            default=-1,
        ),
        "predicted_result_name": str(
            candidate.get("predictedResultName")
            or candidate.get("resultPlantName")
            or candidate.get("predicted_result_name")
            or ""
        ),
        "predicted_result_resolution_source": str(
            candidate.get("predictedResultResolutionSource")
            or candidate.get("predicted_result_resolution_source")
            or ""
        ),
        "mix_lookup_found": bool(
            candidate.get("mixLookupFound") if "mixLookupFound" in candidate else candidate.get("mix_lookup_found")
        ),
        "mix_lookup_key": str(
            candidate.get("mixLookupKey")
            or candidate.get("mix_lookup_key")
            or ""
        ),
        "fusion_legal": bool(
            candidate.get("fusionLegal") if "fusionLegal" in candidate else candidate.get("fusion_legal")
        ),
        "fusion_blocked_reason": str(
            candidate.get("fusionBlockedReason")
            or candidate.get("fusion_blocked_reason")
            or candidate.get("blockedReason")
            or ""
        ),
    }


def _fusion_tactical_metrics(
    observation: Dict[str, Any],
    *,
    row: int,
    col: int,
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    lanes = observation.get("lanes")
    lane = {}
    if isinstance(lanes, list):
        for item in lanes:
            if not isinstance(item, dict):
                continue
            if _safe_int(item.get("row"), default=-1) == row:
                lane = item
                break
    lane_danger_score = _safe_float(lane.get("toughZombiePressureScore"), lane.get("danger"), default=0.0)
    nearby_zombie_count = 0
    nearby_buckethead_count = 0
    nearby_conehead_count = 0
    zombies = observation.get("zombies")
    if isinstance(zombies, list):
        for zombie in zombies:
            if not isinstance(zombie, dict):
                continue
            if _safe_int(zombie.get("row"), default=-1) != row:
                continue
            if not bool(zombie.get("alive", True)):
                continue
            zombie_x = _safe_float(zombie.get("x"), default=999.0)
            if zombie_x < float(col) - 0.75 or zombie_x - float(col) > 5.25:
                continue
            nearby_zombie_count += 1
            zombie_type = _safe_int(zombie.get("type"), default=-1)
            zombie_name = str(zombie.get("typeName") or "").lower()
            if zombie_type in {4, 13} or "bucket" in zombie_name:
                nearby_buckethead_count += 1
            if zombie_type in {2, 12} or "cone" in zombie_name or "roadblock" in zombie_name:
                nearby_conehead_count += 1
    predicted_score = _safe_float(candidate.get("strategicScore"), candidate.get("strategic_score"), default=0.0)
    under_threat = lane_danger_score >= 0.25 or nearby_zombie_count > 0
    estimated_mower_save = lane_danger_score >= 0.6 or nearby_buckethead_count > 0
    return {
        "lane_danger_score": round(lane_danger_score, 4),
        "nearby_zombie_count": int(nearby_zombie_count),
        "nearby_buckethead_count": int(nearby_buckethead_count),
        "nearby_conehead_count": int(nearby_conehead_count),
        "strategic_score": round(predicted_score, 4),
        "under_threat": bool(under_threat),
        "estimated_mower_save": bool(estimated_mower_save),
    }


def _safe_int(*values: Any, default: int = 0) -> int:
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return int(default)


def _safe_float(*values: Any, default: float = 0.0) -> float:
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return float(default)


def _command_text_from_line(line: str) -> str:
    text = str(line or "").strip()
    if not text:
        return ""
    if text in {"}", "]"}:
        return ""
    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            # Ignore malformed JSONL records (including partial writes).
            return ""
        if isinstance(payload, dict):
            for key in ("parser_command", "raw_text", "text", "command"):
                value = payload.get(key)
                if value:
                    return str(value).strip()
        return ""
    return text


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value
