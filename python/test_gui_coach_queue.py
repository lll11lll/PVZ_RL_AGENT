from __future__ import annotations

import json
from pathlib import Path

import pytest

from pvzrl_gui_coach import CoachCommandSink, CoachQueueCommand


def test_coach_queue_sink_preserves_single_and_batch_jsonl_schema(tmp_path: Path) -> None:
    path = tmp_path / "coach.jsonl"
    sink = CoachCommandSink(path, timestamp_factory=lambda: "2026-07-12T22:00:00Z")

    count = sink.append(
        [
            CoachQueueCommand("place 0 1 2", source="gui", parser_command="place 0 1 2"),
            CoachQueueCommand("fuse 1 2 3", source="gui_fusion"),
        ]
    )

    assert count == 2
    assert [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] == [
        {
            "command": "place 0 1 2",
            "source": "gui",
            "timestamp": "2026-07-12T22:00:00Z",
        },
        {
            "command": "fuse 1 2 3",
            "source": "gui_fusion",
            "timestamp": "2026-07-12T22:00:00Z",
        },
    ]


def test_coach_queue_sink_records_distinct_parser_command(tmp_path: Path) -> None:
    path = tmp_path / "coach.jsonl"
    sink = CoachCommandSink(path, timestamp_factory=lambda: "fixed")

    sink.append([CoachQueueCommand("Plant Sunflower", parser_command="place 0 1 2")])

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "command": "Plant Sunflower",
        "parser_command": "place 0 1 2",
        "source": "gui",
        "timestamp": "fixed",
    }


def test_coach_queue_sink_rejects_empty_commands_before_writing(tmp_path: Path) -> None:
    path = tmp_path / "coach.jsonl"
    sink = CoachCommandSink(path)

    with pytest.raises(ValueError, match="coach command is empty"):
        sink.append([CoachQueueCommand("  ")])

    assert not path.exists()
