"""Shared JSONL queue sink for dashboard coach and fusion commands."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass(frozen=True)
class CoachQueueCommand:
    command: str
    source: str = "gui"
    parser_command: str = ""

    def payload(self, timestamp: str) -> dict[str, str]:
        command = str(self.command or "").strip()
        if not command:
            raise ValueError("coach command is empty")
        source = str(self.source or "gui")
        payload = {
            "timestamp": str(timestamp),
            "source": source,
            "command": command,
        }
        parser_command = str(self.parser_command or "").strip()
        if parser_command and parser_command != command:
            payload["parser_command"] = parser_command
        return payload


class CoachCommandSink:
    def __init__(
        self,
        path: Path,
        *,
        timestamp_factory: Callable[[], str] = utc_timestamp,
    ) -> None:
        self.path = Path(path)
        self.timestamp_factory = timestamp_factory

    def append(self, commands: Iterable[CoachQueueCommand]) -> int:
        rows = tuple(commands)
        if not rows:
            return 0
        timestamp = self.timestamp_factory()
        payloads = [command.payload(timestamp) for command in rows]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for payload in payloads:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
        return len(payloads)
