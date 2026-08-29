from __future__ import annotations

import io
import json
import threading
from dataclasses import FrozenInstanceError, dataclass
from typing import Any, Dict, List, Optional

import pytest

from pvzrl_stream_commands import (
    BoundedViewerCommandQueue,
    ViewerCommandController,
    ViewerCommandKind,
    ViewerCommandParser,
    parse_viewer_command,
)
from pvzrl_streamer_source import StreamSourceMessage
from pvzrl_streamer_logging import BufferedStreamerEventLogger
from train_ppo import RotatingTextStream, TeeStream


VIEWER_A = "a" * 64
VIEWER_B = "b" * 64


class FakeClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value

    def set(self, value: float) -> None:
        self.value = float(value)


class FakeSource:
    def __init__(self, messages: Optional[List[StreamSourceMessage]] = None) -> None:
        self.messages = list(messages or [])
        self.started = False
        self.stopped = False
        self.accepting = True
        self.phase_changes: List[tuple[bool, str]] = []

    def start(self) -> None:
        self.started = True

    def stop(self, timeout_seconds: float = 5.0) -> bool:
        self.stopped = True
        return True

    def drain_messages(self, max_items: Optional[int] = None) -> List[StreamSourceMessage]:
        limit = len(self.messages) if max_items is None else min(len(self.messages), int(max_items))
        result = self.messages[:limit]
        del self.messages[:limit]
        return result

    def clear(self) -> int:
        count = len(self.messages)
        self.messages.clear()
        return count

    def set_accepting(self, accepting: bool, *, reason: str = "phase_change") -> int:
        cleared = self.clear()
        self.accepting = bool(accepting)
        self.phase_changes.append((bool(accepting), reason))
        return cleared

    def get_diagnostics(self) -> Dict[str, Any]:
        return {"accepting": self.accepting, "depth": len(self.messages)}


def source_message(event_id: str, text: str, *, viewer_hash: str = VIEWER_A, received: float = 0.0) -> StreamSourceMessage:
    return StreamSourceMessage(
        platform="twitch",
        delivery_id=f"delivery-{event_id}",
        event_id=event_id,
        viewer_hash=viewer_hash,
        command_text=text,
        received_monotonic=received,
    )


@pytest.mark.parametrize(
    ("text", "kind", "row", "column", "specific_field", "specific_value"),
    [
        ("!plant SunFlower 1 1", ViewerCommandKind.PLANT, 0, 0, "plant_type_id", 1),
        ("  !PLANT   sun flower   5   10  ", ViewerCommandKind.PLANT, 4, 9, "canonical_plant_name", "SunFlower"),
        ("!plant Wall-Nut 2 3", ViewerCommandKind.PLANT, 1, 2, "plant_type_id", 3),
        ("!slot 1 1 1", ViewerCommandKind.SLOT, 0, 0, "seed_slot", 0),
        ("!SLOT 14 5 10", ViewerCommandKind.SLOT, 4, 9, "seed_slot", 13),
        ("!fuse Double Shooter 1 2", ViewerCommandKind.FUSE_RESULT, 0, 1, "fusion_result_type_id", 1030),
        ("!FUSE repeater 3 4", ViewerCommandKind.FUSE_RESULT, 2, 3, "canonical_fusion_result_name", "DoubleShooer"),
        ("!fuse twin sunflower 5 10", ViewerCommandKind.FUSE_RESULT, 4, 9, "fusion_result_type_id", 1033),
        ("!fuse 1 10", ViewerCommandKind.FUSE_TILE, 0, 9, "fusion_result_type_id", None),
    ],
)
def test_strict_parser_accepts_only_v1_forms_and_converts_once(
    text: str,
    kind: ViewerCommandKind,
    row: int,
    column: int,
    specific_field: str,
    specific_value: Any,
) -> None:
    command = parse_viewer_command(text)
    assert command.kind is kind
    assert (command.row, command.column) == (row, column)
    assert getattr(command, specific_field) == specific_value
    assert "raw" not in command.to_safe_dict()


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("", "empty"),
        ("   ", "empty"),
        ("plant pea 1 1", "not_command"),
        ("!wait", "unknown_command"),
        ("!defend", "unknown_command"),
        ("!plant", "wrong_argument_count"),
        ("!plant pea 1", "wrong_argument_count"),
        ("!plant unknown 1 1", "unknown_plant"),
        ("!plant pea 0 1", "row_out_of_range"),
        ("!plant pea 7 1", "row_out_of_range"),
        ("!plant pea 1 0", "column_out_of_range"),
        ("!plant pea 1 11", "column_out_of_range"),
        ("!slot 0 1 1", "slot_out_of_range"),
        ("!slot 15 1 1", "slot_out_of_range"),
        ("!slot 1 1", "wrong_argument_count"),
        ("!slot 1 1 1 extra", "wrong_argument_count"),
        ("!fuse", "wrong_argument_count"),
        ("!fuse sunflower 1 1", "unknown_fusion_result"),
        ("!fuse 1030 1 1", "unknown_fusion_result"),
        ("!fuse unknown 1 1", "unknown_fusion_result"),
        ("!plant pea\n1 1", "invalid_character"),
        ("!plant pea\t1 1", "invalid_character"),
        ("!plant pеa 1 1", "invalid_character"),  # Cyrillic e.
        ("!plant pea😀 1 1", "invalid_character"),
        ("!plant pea;shutdown 1 1", "invalid_character"),
        ("!plant $(whoami) 1 1", "invalid_character"),
        ("!plant 'pea' 1 1", "invalid_character"),
        ("!plant(pea,1,1)", "invalid_character"),
    ],
)
def test_strict_parser_rejects_malformed_untrusted_input(text: str, reason: str) -> None:
    outcome = ViewerCommandParser().try_parse(text)
    assert not outcome.accepted
    assert outcome.command is None
    assert outcome.reason == reason
    if text:
        assert text not in json.dumps(outcome.to_safe_dict())


def test_parser_enforces_configurable_maximum_and_records_are_immutable() -> None:
    parser = ViewerCommandParser(max_message_length=20)
    assert parser.try_parse("!plant pea 1 1").accepted
    outcome = parser.try_parse("!plant peashooter 1 1")
    assert not outcome.accepted
    assert outcome.reason == "message_too_long"

    command = parse_viewer_command("!slot 2 3 4")
    with pytest.raises(FrozenInstanceError):
        command.row = 4  # type: ignore[misc]


def test_queue_is_fifo_bounded_rejects_newest_and_deduplicates() -> None:
    clock = FakeClock()
    queue = BoundedViewerCommandQueue(capacity=2, ttl_seconds=10.0, dedupe_capacity=4, monotonic=clock)
    first = parse_viewer_command("!slot 1 1 1")
    second = parse_viewer_command("!slot 2 1 1")
    third = parse_viewer_command("!slot 3 1 1")

    outcome1 = queue.enqueue(first, message_id="event-1", viewer_hash=VIEWER_A)
    outcome2 = queue.enqueue(second, message_id="event-2", viewer_hash=VIEWER_B)
    rejected = queue.enqueue(third, message_id="event-3", viewer_hash=VIEWER_A)
    duplicate_rejected = queue.enqueue(third, message_id="event-3", viewer_hash=VIEWER_A)

    assert [outcome1.status, outcome2.status] == ["accepted", "accepted"]
    assert rejected.status == "capacity_rejected"
    assert duplicate_rejected.status == "duplicate"
    assert queue.peek() is not None
    assert queue.peek().event_id == "event-1"
    assert queue.pop_head(outcome1.command_id, status="selected")[0].event_id == "event-1"  # type: ignore[index]
    assert queue.peek().event_id == "event-2"  # type: ignore[union-attr]
    assert queue.snapshot().counters == {
        "accepted": 2,
        "capacity_rejected": 1,
        "duplicate": 1,
        "selected": 1,
    }


def test_queue_ttl_phase_isolation_and_safe_persistence_shape() -> None:
    clock = FakeClock(5.0)
    queue = BoundedViewerCommandQueue(capacity=3, ttl_seconds=2.0, dedupe_capacity=8, monotonic=clock)
    command = parse_viewer_command("!plant cherry bomb 2 3")
    accepted = queue.enqueue(command, message_id="chat-event-1", viewer_hash=VIEWER_A)
    queued = queue.peek()
    assert queued is not None
    assert queued.received_monotonic == 5.0
    assert queued.expires_monotonic == 7.0
    assert queued.event_id == "chat-event-1"

    safe_payload = json.dumps(queued.to_safe_dict(), sort_keys=True)
    assert "chat-event-1" in safe_payload
    assert "cherry bomb" not in safe_payload.lower()
    assert "raw_user" not in safe_payload

    old_generation = queue.phase_generation
    snapshot = queue.begin_phase("evaluate")
    assert snapshot.phase_generation == old_generation + 1
    assert snapshot.depth == 0
    stale = queue.enqueue(
        command,
        message_id="chat-event-2",
        viewer_hash=VIEWER_A,
        phase_generation=old_generation,
    )
    assert stale.status == "phase_rejected"
    duplicate_across_phase = queue.enqueue(command, message_id="chat-event-1", viewer_hash=VIEWER_A)
    assert duplicate_across_phase.status == "duplicate"


def test_queue_thread_safety_never_exceeds_capacity() -> None:
    queue = BoundedViewerCommandQueue(capacity=8, ttl_seconds=5.0, dedupe_capacity=32)
    command = parse_viewer_command("!fuse 1 1")
    outcomes: List[str] = []
    lock = threading.Lock()

    def publish(index: int) -> None:
        status = queue.enqueue(
            command,
            message_id=f"concurrent-{index}",
            viewer_hash=VIEWER_A,
        ).status
        with lock:
            outcomes.append(status)

    threads = [threading.Thread(target=publish, args=(index,)) for index in range(24)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2.0)

    assert all(not thread.is_alive() for thread in threads)
    assert len(queue) == 8
    assert outcomes.count("accepted") == 8
    assert outcomes.count("capacity_rejected") == 16


@dataclass(frozen=True)
class FakeResolution:
    legal: bool
    classification: str
    reason: str
    action_id: Optional[int]
    frame_identity: str = "frame:1"


def test_controller_scans_fifo_and_executes_at_most_one_per_two_seconds() -> None:
    clock = FakeClock()
    source = FakeSource(
        [
            source_message("bad", "hello streamer"),
            source_message("unresolvable", "!slot 1 1 1"),
            source_message("illegal", "!slot 2 1 1"),
            source_message("legal-1", "!slot 3 1 1"),
            source_message("legal-2", "!slot 4 1 1"),
        ]
    )
    queue = BoundedViewerCommandQueue(capacity=8, ttl_seconds=30.0, dedupe_capacity=16, monotonic=clock)
    controller = ViewerCommandController(source=source, queue=queue, monotonic=clock)
    executed: List[int] = []

    def resolve(command: Any) -> FakeResolution:
        if command.seed_slot == 0:
            return FakeResolution(False, "unresolvable", "slot_missing", None)
        if command.seed_slot == 1:
            return FakeResolution(False, "currently_illegal", "cooldown_not_ready", None)
        return FakeResolution(True, "resolved", "", 1 + int(command.seed_slot) * 50)

    def execute(_queued: Any, resolution: FakeResolution) -> str:
        executed.append(int(resolution.action_id))  # type: ignore[arg-type]
        return "ok"

    first_tick = controller.tick(resolve, execute_command=execute, now=0.0)
    assert first_tick.opportunity_opened
    assert executed == [101]
    assert [outcome.status for outcome in first_tick.outcomes] == [
        "parse_rejected",
        "accepted",
        "accepted",
        "accepted",
        "accepted",
        "unresolvable",
        "currently_illegal",
        "dispatched",
    ]
    parse_rejected, accepted, _accepted_illegal, _accepted_legal, _accepted_later, dropped, _illegal, dispatched = first_tick.outcomes
    assert parse_rejected.event_id == "bad"
    assert parse_rejected.viewer_hash == VIEWER_A
    assert parse_rejected.command is None
    assert accepted.event_id == "unresolvable"
    assert accepted.command is not None and accepted.command.seed_slot == 0
    assert dropped.event_id == "unresolvable"
    assert dropped.viewer_hash == VIEWER_A
    assert dropped.command is not None and dropped.command.to_safe_dict()["kind"] == "slot"
    assert dispatched.event_id == "legal-1"
    assert dispatched.command is not None and dispatched.command.seed_slot == 2
    rendered_outcomes = json.dumps([outcome.to_safe_dict() for outcome in first_tick.outcomes])
    assert "hello streamer" not in rendered_outcomes
    assert "!slot" not in rendered_outcomes
    assert len(queue) == 1

    clock.set(1.999)
    not_due = controller.tick(resolve, execute_command=execute)
    assert not not_due.opportunity_opened
    assert executed == [101]
    assert len(queue) == 1

    clock.set(2.0)
    second_tick = controller.tick(resolve, execute_command=execute)
    assert second_tick.opportunity_opened
    assert executed == [101, 151]
    assert len(queue) == 0

    source.messages.extend(
        [source_message("late-1", "!slot 5 1 1", received=100.0), source_message("late-2", "!slot 6 1 1", received=100.0)]
    )
    clock.set(100.0)
    controller.tick(resolve, execute_command=execute)
    assert len(executed) == 3
    assert len(queue) == 1
    clock.set(100.1)
    controller.tick(resolve, execute_command=execute)
    assert len(executed) == 3
    assert controller.next_opportunity_monotonic == 102.0


def test_controller_drops_expired_commands_and_phase_gates_source() -> None:
    clock = FakeClock()
    source = FakeSource([source_message("old", "!slot 1 1 1", received=0.0)])
    queue = BoundedViewerCommandQueue(capacity=4, ttl_seconds=2.0, dedupe_capacity=8, monotonic=clock)
    controller = ViewerCommandController(source=source, queue=queue, monotonic=clock)
    controller.poll_source(now=0.0)
    assert len(queue) == 1

    clock.set(3.0)
    tick = controller.tick(lambda _command: FakeResolution(True, "resolved", "", 1))
    assert tick.opportunity_opened
    assert [outcome.status for outcome in tick.outcomes] == ["expired"]
    assert tick.outcomes[0].event_id == "old"
    assert tick.outcomes[0].viewer_hash == VIEWER_A
    assert tick.outcomes[0].command is not None
    assert tick.selected is None

    source.messages.append(source_message("eval-leak", "!slot 2 1 1", received=3.0))
    before = queue.phase_generation
    snapshot = controller.begin_phase("evaluate", accepting=False, now=3.0)
    assert snapshot.phase_generation == before + 1
    assert source.messages == []
    assert source.phase_changes[-1] == (False, "evaluate")
    disabled = controller.tick(lambda _command: FakeResolution(True, "resolved", "", 1), now=3.0)
    assert not disabled.opportunity_opened
    assert len(queue) == 0


def test_controller_defers_source_drain_while_parsed_fifo_is_full() -> None:
    clock = FakeClock()
    source = FakeSource(
        [
            source_message("queued-upstream-1", "!slot 2 1 1"),
            source_message("queued-upstream-2", "!slot 3 1 1"),
        ]
    )
    queue = BoundedViewerCommandQueue(
        capacity=1,
        ttl_seconds=10.0,
        dedupe_capacity=8,
        monotonic=clock,
    )
    assert queue.enqueue(
        parse_viewer_command("!slot 1 1 1"),
        message_id="already-parsed",
        viewer_hash=VIEWER_A,
        received_monotonic=0.0,
    ).status == "accepted"
    controller = ViewerCommandController(source=source, queue=queue, monotonic=clock)

    assert controller.poll_source(now=0.0) == ()
    assert len(source.messages) == 2
    assert queue.snapshot().counters["source_poll_deferred_queue_full"] == 1

    queued = queue.peek()
    assert queued is not None
    queue.pop_head(queued.command_id, status="test_removed", now=0.0)
    outcomes = controller.poll_source(now=0.0)
    assert [outcome.status for outcome in outcomes] == ["accepted"]
    assert len(source.messages) == 1


def test_controller_and_queue_diagnostics_never_persist_raw_identity_or_chat() -> None:
    clock = FakeClock()
    raw_identity = "RawDisplayNameSecret"
    raw_text = "!plant sun flower 1 1"
    source = FakeSource([source_message("privacy-event", raw_text, viewer_hash=VIEWER_A)])
    controller = ViewerCommandController(
        source=source,
        queue=BoundedViewerCommandQueue(capacity=4, ttl_seconds=5.0, dedupe_capacity=8, monotonic=clock),
        monotonic=clock,
    )
    controller.poll_source(now=0.0)
    rendered = json.dumps(controller.diagnostics(), sort_keys=True)
    assert raw_identity not in rendered
    assert raw_text not in rendered
    assert "privacy-event" not in rendered
    assert VIEWER_A not in rendered


def test_controller_source_failure_and_close_are_bounded_and_nonfatal() -> None:
    class BrokenSource(FakeSource):
        def drain_messages(self, max_items: Optional[int] = None) -> List[StreamSourceMessage]:
            raise RuntimeError("must not leak this detail")

    source = BrokenSource()
    controller = ViewerCommandController(source=source)
    outcome = controller.poll_source()
    assert outcome[0].status == "source_error"
    assert outcome[0].reason == "source_error:RuntimeError"
    assert "must not leak" not in json.dumps(outcome[0].to_safe_dict())
    assert controller.close(timeout_seconds=0.1)
    assert source.stopped


def test_disconnect_between_selection_and_execution_rejects_the_popped_command() -> None:
    class EpochSource(FakeSource):
        def __init__(self) -> None:
            super().__init__()
            self.epoch = 1

        def get_diagnostics(self) -> Dict[str, Any]:
            return {
                "stream_source_accepting": self.accepting,
                "stream_source_gate_epoch": self.epoch,
            }

    clock = FakeClock()
    source = EpochSource()
    controller = ViewerCommandController(source=source, monotonic=clock)
    controller.begin_phase("stream_train", accepting=True, now=0.0)
    source.messages.append(source_message("race", "!slot 1 1 1"))
    tick = controller.tick(
        lambda _command: FakeResolution(True, "resolved", "", 1),
        now=0.0,
    )
    assert tick.selected is not None

    source.epoch += 1
    source.accepting = False
    stale = controller.validate_selected_for_execution(tick.selected, now=0.0)
    assert stale is not None
    assert stale.status == "source_stale"
    assert stale.reason == "source_not_accepting"
    assert controller.queue.snapshot().counters["execution_gate_rejected"] == 1


def test_streamer_event_log_rotates_and_strips_raw_identity(tmp_path) -> None:
    path = tmp_path / "streamer_events.jsonl"
    logger = BufferedStreamerEventLogger(
        path,
        flush_records=1,
        max_bytes=1024,
        backup_count=2,
    )
    for index in range(40):
        logger.append(
            {
                "event": "viewer_command",
                "command_id": f"command-{index}",
                "viewer_hash": VIEWER_A,
                "username": "must-not-persist",
                "nested": [
                    {
                        "chatter_user_login": "nested-identity",
                        "access_token": "nested-token",
                        "message": {"text": "raw-chat"},
                        "safe_status": "accepted",
                    }
                ],
                "padding": "x" * 80,
            }
        )
    logger.close()
    files = list(tmp_path.glob("streamer_events.jsonl*"))
    assert 1 <= len(files) <= 3
    rendered = "".join(file.read_text(encoding="utf-8") for file in files)
    assert "must-not-persist" not in rendered
    assert "nested-identity" not in rendered
    assert "nested-token" not in rendered
    assert "raw-chat" not in rendered
    assert "safe_status" in rendered
    assert VIEWER_A in rendered


def test_streamer_console_log_rotation_is_bounded(tmp_path) -> None:
    path = tmp_path / "streamer.log"
    stream = RotatingTextStream(path, max_bytes=1024, backup_count=2)
    try:
        for index in range(80):
            stream.write(f"line-{index}-" + ("x" * 64) + "\n")
            stream.flush()
    finally:
        stream.close()
    files = list(tmp_path.glob("streamer.log*"))
    assert 1 <= len(files) <= 3
    assert all(file.stat().st_size <= 1100 for file in files)


def test_tee_stream_write_is_safe_after_rotating_handle_close(tmp_path) -> None:
    path = tmp_path / "streamer.log"
    stream = RotatingTextStream(path, max_bytes=1024, backup_count=2)
    proxy = TeeStream(io.StringIO(), stream)
    stream.close()

    assert proxy.write("\u001b[0m") == len("\u001b[0m")
    proxy.flush()
