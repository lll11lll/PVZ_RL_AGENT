"""Typed assisted human-in-the-loop command and intervention primitives."""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


LAWN_ROWS = 5
LAWN_COLS = 9
SEED_PACKET_SLOTS = 14


class AssistedCommandType(str, Enum):
    PLANT = "PLANT"
    FUSE = "FUSE"
    REMOVE = "REMOVE"
    BOOST = "BOOST"
    SAVE_SUN = "SAVE_SUN"
    PAUSE_AGENT = "PAUSE_AGENT"
    RESUME_AGENT = "RESUME_AGENT"
    FORCE_EVAL = "FORCE_EVAL"


class AssistedCommandStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"


class AssistedExecutionMode(str, Enum):
    OVERRIDE = "override"
    ASSIST = "assist"
    COACH_ONLY = "coach_only"
    VIEWER_SUGGESTION = "viewer_suggestion"


@dataclass
class AssistedCoachCommand:
    command_type: AssistedCommandType
    source: str = "dashboard"
    user: str = "local"
    row: Optional[int] = None
    col: Optional[int] = None
    target: str = ""
    status: AssistedCommandStatus = AssistedCommandStatus.PENDING
    command_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["command_type"] = self.command_type.value
        payload["status"] = self.status.value
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AssistedCoachCommand":
        return cls(
            command_type=AssistedCommandType(str(payload.get("command_type") or "").upper()),
            source=str(payload.get("source") or "dashboard"),
            user=str(payload.get("user") or "local"),
            row=_optional_int(payload.get("row")),
            col=_optional_int(payload.get("col")),
            target=str(payload.get("target") or ""),
            status=AssistedCommandStatus(str(payload.get("status") or "pending").lower()),
            command_id=str(payload.get("command_id") or uuid.uuid4().hex[:12]),
            created_at=float(payload.get("created_at") or time.time()),
            updated_at=float(payload.get("updated_at") or time.time()),
            note=str(payload.get("note") or ""),
        )

    def display_text(self) -> str:
        parts = [self.command_type.value]
        if self.row is not None:
            parts.append(str(self.row))
        if self.col is not None:
            parts.append(str(self.col))
        if self.target:
            parts.append(self.target)
        return " ".join(parts)


@dataclass(frozen=True)
class AssistedValidationResult:
    valid: bool
    reason: str = ""
    backend_command: str = ""
    backend_supported: bool = False


class AssistedCommandValidator:
    """Validate UI/source commands independently from queueing and execution."""

    POSITION_COMMANDS = {
        AssistedCommandType.PLANT,
        AssistedCommandType.FUSE,
        AssistedCommandType.REMOVE,
        AssistedCommandType.BOOST,
    }
    TARGET_COMMANDS = {
        AssistedCommandType.PLANT,
        AssistedCommandType.FUSE,
        AssistedCommandType.BOOST,
    }

    @classmethod
    def validate(cls, command: AssistedCoachCommand) -> AssistedValidationResult:
        if command.command_type in cls.POSITION_COMMANDS:
            if command.row is None or not 0 <= command.row < LAWN_ROWS:
                return AssistedValidationResult(False, f"row must be between 0 and {LAWN_ROWS - 1}")
            if command.col is None or not 0 <= command.col < LAWN_COLS:
                return AssistedValidationResult(False, f"column must be between 0 and {LAWN_COLS - 1}")
        if command.command_type in cls.TARGET_COMMANDS:
            if not command.target.strip():
                return AssistedValidationResult(False, "plant/seed target is required")
            if command.command_type in {AssistedCommandType.PLANT, AssistedCommandType.FUSE}:
                try:
                    seed_slot = int(command.target)
                except ValueError:
                    return AssistedValidationResult(False, "seed target must be an integer slot")
                if not 0 <= seed_slot < SEED_PACKET_SLOTS:
                    return AssistedValidationResult(
                        False,
                        f"seed target must be between 0 and {SEED_PACKET_SLOTS - 1}",
                    )
        backend_command = cls.backend_command(command)
        return AssistedValidationResult(
            True,
            backend_command=backend_command or "",
            backend_supported=bool(backend_command),
        )

    @staticmethod
    def backend_command(command: AssistedCoachCommand) -> Optional[str]:
        """Serialize supported commands to the established coach parser syntax."""
        if command.command_type in {AssistedCommandType.PLANT, AssistedCommandType.FUSE}:
            keyword = command.command_type.value.lower()
            return f"{keyword} {int(command.target)} {command.row} {command.col}"
        if command.command_type == AssistedCommandType.SAVE_SUN:
            return "economy"
        return None


class AssistedCommandQueue:
    """Thread-safe moderation queue suitable for local UI and future chat sources."""

    def __init__(self) -> None:
        self._commands: List[AssistedCoachCommand] = []
        self._lock = threading.RLock()

    def submit(self, command: AssistedCoachCommand) -> AssistedCoachCommand:
        result = AssistedCommandValidator.validate(command)
        if not result.valid:
            raise ValueError(result.reason)
        with self._lock:
            self._commands.append(command)
        return command

    def all(self) -> List[AssistedCoachCommand]:
        with self._lock:
            return list(self._commands)

    def get(self, command_id: str) -> Optional[AssistedCoachCommand]:
        with self._lock:
            return next((item for item in self._commands if item.command_id == command_id), None)

    def set_status(
        self,
        command_id: str,
        status: AssistedCommandStatus,
        note: str = "",
    ) -> AssistedCoachCommand:
        with self._lock:
            command = self.get(command_id)
            if command is None:
                raise KeyError(command_id)
            command.status = status
            command.updated_at = time.time()
            if note:
                command.note = note
            return command

    def modify(self, command_id: str, replacement: AssistedCoachCommand) -> AssistedCoachCommand:
        result = AssistedCommandValidator.validate(replacement)
        if not result.valid:
            raise ValueError(result.reason)
        with self._lock:
            current = self.get(command_id)
            if current is None:
                raise KeyError(command_id)
            current.command_type = replacement.command_type
            current.source = replacement.source
            current.user = replacement.user
            current.row = replacement.row
            current.col = replacement.col
            current.target = replacement.target
            current.status = AssistedCommandStatus.PENDING
            current.updated_at = time.time()
            current.note = "modified"
            return current

    def counts(self) -> Dict[str, int]:
        with self._lock:
            return {
                status.value: sum(item.status == status for item in self._commands)
                for status in AssistedCommandStatus
            }


class InterventionJSONLLogger:
    """Append stable human-intervention records for later training analysis."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def log(
        self,
        *,
        run_id: str,
        episode_id: int,
        step: int,
        mode: str,
        model_action: Any,
        human_command: Any,
        command_source: str,
        status: str,
        board_state_summary: Optional[Dict[str, Any]] = None,
        reward_before: Optional[float] = None,
        reward_after: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        record = {
            "timestamp": time.time(),
            "run_id": str(run_id or "unknown"),
            "episode_id": int(episode_id),
            "step": int(step),
            "mode": str(mode or "unknown"),
            "model_action": model_action,
            "human_command": human_command,
            "command_source": str(command_source or "unknown"),
            "board_state_summary": dict(board_state_summary or {}),
            "reward_before": reward_before,
            "reward_after": reward_after,
            "status": str(status),
            "metadata": dict(metadata or {}),
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock, self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        except OSError:
            # Intervention logging must never terminate an active environment step.
            pass
        return record


def _optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)


def queue_rows(commands: Iterable[AssistedCoachCommand]) -> List[Tuple[str, ...]]:
    """Return stable UI rows without coupling queue state to Tkinter."""
    return [
        (
            item.command_id,
            time.strftime("%H:%M:%S", time.localtime(item.created_at)),
            f"{item.source}/{item.user}",
            item.command_type.value,
            "" if item.row is None else str(item.row),
            "" if item.col is None else str(item.col),
            item.target,
            item.status.value,
        )
        for item in commands
    ]
