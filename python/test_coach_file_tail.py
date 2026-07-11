from __future__ import annotations

import json
import os
import sys
from pathlib import Path


PYTHON_DIR = Path(__file__).resolve().parent
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from pvzrl_file_tail import IncrementalLineTailReader
from pvzrl_human_coach import FileCoachCommandSource
from pvzrl_stream_coach import JsonlCoachCommandSource


def _stream_record(command: str, *, message_id: str = "m1") -> str:
    return json.dumps(
        {
            "message_id": message_id,
            "selected_command": {"command": command},
            "timestamp": 1.0,
        },
        ensure_ascii=False,
    )


def test_incremental_reader_does_not_reread_incomplete_record(tmp_path: Path) -> None:
    path = tmp_path / "commands.txt"
    path.write_bytes("éco".encode("utf-8"))
    reader = IncrementalLineTailReader(path, start_at_end=False)

    assert reader.read_lines() == []
    first_bytes_read = reader.diagnostics()["bytes_read"]
    assert first_bytes_read == path.stat().st_size
    assert reader.pending_bytes == path.stat().st_size
    assert reader.offset == 0
    assert reader.read_offset == path.stat().st_size

    assert reader.read_lines() == []
    assert reader.diagnostics()["bytes_read"] == first_bytes_read

    with path.open("ab") as handle:
        handle.write("nomie\r\n".encode("utf-8"))
    assert reader.read_lines() == ["économie"]
    assert reader.pending_bytes == 0
    assert reader.offset == path.stat().st_size


def test_human_source_retains_partial_json_and_skips_complete_malformed_record(tmp_path: Path) -> None:
    path = tmp_path / "human.jsonl"
    path.write_text('{"command":"plant 0 2 4"', encoding="utf-8")
    source = FileCoachCommandSource(path, start_at_end=False)

    assert source.poll() is None
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write("}\r\n")
        handle.write("{not json}\r\n")
        handle.write(json.dumps({"command": "wait"}) + "\r\n")

    assert source.poll() == "plant 0 2 4"
    assert source.poll() == "wait"
    assert source.poll() is None
    assert source._tail.diagnostics()["malformed_record_count"] == 1
    assert source._last_error == "json_decode_error"


def test_stream_source_retains_partial_json_until_newline(tmp_path: Path) -> None:
    path = tmp_path / "stream.jsonl"
    path.write_text('{"selected_command":{"command":"wait"}', encoding="utf-8")
    source = JsonlCoachCommandSource(path, start_at_end=False)

    assert source.poll_latest() is None
    assert source._tail is not None
    assert source._tail.pending_bytes == path.stat().st_size
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write("}\r\n")

    command = source.poll_latest()
    assert command is not None
    assert command.command == "wait"
    assert source._tail.diagnostics()["malformed_record_count"] == 0


def test_complete_malformed_stream_record_advances_with_diagnostics(tmp_path: Path) -> None:
    path = tmp_path / "stream.jsonl"
    path.write_text("{not json}\n" + _stream_record("wait") + "\n", encoding="utf-8")
    source = JsonlCoachCommandSource(path, start_at_end=False)

    command = source.poll_latest()
    assert command is not None
    assert command.command == "wait"
    diagnostics = source.get_diagnostics()
    assert diagnostics["stream_source_last_error"] == "json_decode_error"
    assert source._tail is not None
    assert source._tail.diagnostics()["malformed_record_count"] == 1
    assert source.poll_latest() is None


def test_clear_to_end_discards_completion_of_partial_record(tmp_path: Path) -> None:
    human_path = tmp_path / "human.jsonl"
    human_path.write_text('{"command":"wait"', encoding="utf-8")
    human = FileCoachCommandSource(human_path, start_at_end=False)
    assert human.poll() is None
    assert human.clear_to_end() == 1
    with human_path.open("a", encoding="utf-8") as handle:
        handle.write("}\n")
        handle.write("plant 0 2 4\n")
    assert human.poll() == "plant 0 2 4"
    assert human.poll() is None

    stream_path = tmp_path / "stream.jsonl"
    stream_path.write_text('{"selected_command":{"command":"wait"}', encoding="utf-8")
    stream = JsonlCoachCommandSource(stream_path, start_at_end=False)
    assert stream.poll_latest() is None
    assert stream.clear_to_end() == 1
    with stream_path.open("a", encoding="utf-8") as handle:
        handle.write("}\n")
        handle.write(_stream_record("wait", message_id="fresh") + "\n")
    command = stream.poll_latest()
    assert command is not None
    assert command.command == "wait"


def test_start_at_end_clears_existing_partial_record(tmp_path: Path) -> None:
    path = tmp_path / "stream.jsonl"
    path.write_text('{"selected_command":{"command":"plant"}', encoding="utf-8")
    source = JsonlCoachCommandSource(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("}\n")
        handle.write(_stream_record("wait", message_id="fresh") + "\n")

    command = source.poll_latest()
    assert command is not None
    assert command.command == "wait"
    assert source.poll_latest() is None


def test_path_replacement_is_detected_even_when_new_file_is_larger(tmp_path: Path) -> None:
    path = tmp_path / "stream.jsonl"
    path.write_text(_stream_record("wait", message_id="old") + "\n", encoding="utf-8")
    source = JsonlCoachCommandSource(path, start_at_end=False)
    assert source.poll_latest() is not None

    replacement = tmp_path / "replacement.jsonl"
    replacement.write_text(
        _stream_record("wait", message_id="new-1")
        + "\n"
        + _stream_record("plant", message_id="new-2")
        + "\n",
        encoding="utf-8",
    )
    assert replacement.stat().st_size > path.stat().st_size
    os.replace(replacement, path)

    command = source.poll_latest()
    assert command is not None
    assert command.command == "plant"
    assert source._tail is not None
    assert source._tail.diagnostics()["rotation_count"] >= 1


def test_same_inode_rewrite_uses_content_anchor_fallback(tmp_path: Path) -> None:
    path = tmp_path / "human.txt"
    path.write_text("wait\n", encoding="utf-8")
    source = FileCoachCommandSource(path, start_at_end=False)
    assert source.poll() == "wait"

    path.write_text("plant 0 2 4\nwait\n", encoding="utf-8")
    assert source.poll() == "plant 0 2 4"
    assert source.poll() == "wait"
    assert source._tail.diagnostics()["truncation_count"] >= 1


def test_same_size_rewrite_with_unchanged_tail_uses_head_anchor(tmp_path: Path) -> None:
    path = tmp_path / "anchored.txt"
    old_record = (b"a" * 128) + (b"same-tail" * 24) + b"\n"
    new_record = (b"b" * 128) + (b"same-tail" * 24) + b"\n"
    assert len(old_record) == len(new_record)
    path.write_bytes(old_record)
    reader = IncrementalLineTailReader(path, start_at_end=False)
    assert reader.read_lines() == [old_record[:-1].decode("ascii")]

    path.write_bytes(new_record)
    assert reader.read_lines() == [new_record[:-1].decode("ascii")]
    assert reader.diagnostics()["truncation_count"] >= 1


def test_reads_are_bounded_and_oversized_partial_is_discarded_to_boundary(tmp_path: Path) -> None:
    path = tmp_path / "oversized.txt"
    path.write_bytes(b"abcdefghijkl")
    reader = IncrementalLineTailReader(
        path,
        start_at_end=False,
        max_read_bytes=4,
        max_pending_bytes=8,
    )
    assert reader.read_lines() == []
    assert reader.diagnostics()["bytes_read"] == 4
    assert reader.offset == 0
    assert reader.read_lines() == []
    assert reader.diagnostics()["bytes_read"] == 8
    assert reader.read_lines() == []
    assert reader.diagnostics()["oversized_record_count"] == 1
    assert reader.discarding_cleared_partial is True

    with path.open("ab") as handle:
        handle.write(b"\nwait\n")
    assert reader.read_lines() == []
    assert reader.read_lines() == ["wait"]


def test_utf8_json_and_crlf_are_preserved(tmp_path: Path) -> None:
    human_path = tmp_path / "human.txt"
    human_path.write_bytes("défendre 3\r\n".encode("utf-8"))
    human = FileCoachCommandSource(human_path, start_at_end=False)
    assert human.poll() == "défendre 3"

    stream_path = tmp_path / "stream.jsonl"
    payload = {"user": "Renée", "message_id": "utf8", "message": "!wait", "timestamp": 2.0}
    stream_path.write_bytes((json.dumps(payload, ensure_ascii=False) + "\r\n").encode("utf-8"))
    stream = JsonlCoachCommandSource(stream_path, start_at_end=False)
    messages = stream.drain_messages()
    assert len(messages) == 1
    assert messages[0].display_name == "Renée"
    assert messages[0].text == "!wait"
