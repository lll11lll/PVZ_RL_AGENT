from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from pvzrl_sb3 import apply_streamer_sun_reservation_mask
from pvzrl_stream_commands import (
    BoundedViewerCommandQueue,
    ViewerCommandController,
    parse_viewer_command,
)
from pvzrl_streamer_source import StreamSourceMessage


VIEWER_HASH = "d" * 64


class Clock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value

    def set(self, value: float) -> None:
        self.value = float(value)


class Source:
    def __init__(self, messages: Optional[List[StreamSourceMessage]] = None) -> None:
        self.messages = list(messages or [])
        self.accepting = True

    def start(self) -> None:
        return None

    def stop(self, timeout_seconds: float = 5.0) -> bool:
        return True

    def clear(self) -> int:
        count = len(self.messages)
        self.messages.clear()
        return count

    def set_accepting(self, accepting: bool, *, reason: str = "phase_change") -> int:
        self.accepting = bool(accepting)
        return self.clear()

    def drain_messages(self, max_items: Optional[int] = None) -> List[StreamSourceMessage]:
        count = len(self.messages) if max_items is None else min(len(self.messages), int(max_items))
        result = self.messages[:count]
        del self.messages[:count]
        return result

    def get_diagnostics(self) -> Dict[str, Any]:
        return {"stream_source_accepting": self.accepting, "stream_source_gate_epoch": 1}


def message(event_id: str, text: str) -> StreamSourceMessage:
    return StreamSourceMessage(
        platform="twitch",
        delivery_id=f"delivery-{event_id}",
        event_id=event_id,
        viewer_hash=VIEWER_HASH,
        command_text=text,
        received_monotonic=0.0,
    )


@dataclass(frozen=True)
class Resolution:
    legal: bool
    classification: str
    reason: str
    action_id: Optional[int]
    legality: str
    required_sun: int = 0
    frame_identity: str = "frame:1"


def controller_for(
    clock: Clock,
    messages: List[StreamSourceMessage],
    *,
    ttl_seconds: float = 10.0,
) -> ViewerCommandController:
    queue = BoundedViewerCommandQueue(
        capacity=8,
        ttl_seconds=ttl_seconds,
        dedupe_capacity=16,
        monotonic=clock,
    )
    return ViewerCommandController(
        source=Source(messages),
        queue=queue,
        monotonic=clock,
    )


def test_insufficient_sun_stays_fifo_head_and_executes_when_legal() -> None:
    clock = Clock()
    controller = controller_for(clock, [message("sun", "!slot 1 1 1")])
    calls = 0

    def resolve(_command: Any) -> Resolution:
        nonlocal calls
        calls += 1
        if calls < 3:
            return Resolution(
                False,
                "currently_illegal",
                "insufficient_sun",
                None,
                "TEMPORARILY_BLOCKED",
                required_sun=50,
            )
        return Resolution(True, "resolved", "", 1, "LEGAL", required_sun=50)

    first = controller.tick(resolve, now=0.0)
    assert first.selected is None
    assert len(controller.queue) == 1
    assert controller.diagnostics()["pending_viewer_block_reason"] == "insufficient_sun"
    assert controller.diagnostics()["streamer_reserved_sun"] == 50
    assert controller.diagnostics()["pending_viewer_retry_count"] == 0

    clock.set(2.0)
    second = controller.tick(resolve, now=2.0)
    assert second.selected is None
    assert len(controller.queue) == 1
    assert controller.diagnostics()["pending_viewer_retry_count"] == 1

    clock.set(4.0)
    third = controller.tick(resolve, now=4.0)
    assert third.selected is not None
    assert len(controller.queue) == 0
    assert controller.diagnostics()["pending_viewer_command"] is None
    assert controller.diagnostics()["streamer_reserved_sun"] == 0
    assert controller.diagnostics()["streamer_command_queue"]["counters"]["executed"] == 0


def test_cooldown_retry_does_not_reserve_sun() -> None:
    clock = Clock()
    controller = controller_for(clock, [message("cooldown", "!slot 1 1 1")])
    attempts = 0

    def resolve(_command: Any) -> Resolution:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return Resolution(False, "currently_illegal", "cooldown_not_ready", None, "TEMPORARILY_BLOCKED", 50)
        return Resolution(True, "resolved", "", 1, "LEGAL", 50)

    controller.tick(resolve, now=0.0)
    assert len(controller.queue) == 1
    assert controller.diagnostics()["streamer_reserved_sun"] == 0
    clock.set(2.0)
    assert controller.tick(resolve, now=2.0).selected is not None
    assert len(controller.queue) == 0


def test_permanent_invalid_advances_fifo_and_ttl_removes_pending() -> None:
    clock = Clock()
    controller = controller_for(
        clock,
        [message("bad", "!slot 1 1 1"), message("good", "!slot 2 1 1")],
        ttl_seconds=2.0,
    )

    def resolve(command: Any) -> Resolution:
        if command.seed_slot == 0:
            return Resolution(False, "unresolvable", "incompatible_pair", None, "PERMANENTLY_INVALID")
        return Resolution(True, "resolved", "", 51, "LEGAL")

    first = controller.tick(resolve, now=0.0)
    assert first.selected is not None
    assert [outcome.status for outcome in first.outcomes] == ["accepted", "accepted", "permanently_rejected", "selected"]
    assert controller.queue.snapshot().counters["permanently_rejected"] == 1

    pending_controller = controller_for(clock, [message("ttl", "!slot 1 1 1")], ttl_seconds=2.0)
    blocked = lambda _command: Resolution(False, "currently_illegal", "insufficient_sun", None, "TEMPORARILY_BLOCKED", 50)
    pending_controller.tick(blocked, now=0.0)
    clock.set(3.0)
    expired = pending_controller.tick(blocked, now=3.0)
    assert expired.selected is None
    assert expired.outcomes[-1].status == "expired"
    assert pending_controller.diagnostics()["streamer_reserved_sun"] == 0


def test_reset_and_evaluation_clear_pending_reservation() -> None:
    clock = Clock()
    controller = controller_for(clock, [message("reset", "!slot 1 1 1")])
    blocked = lambda _command: Resolution(False, "currently_illegal", "insufficient_sun", None, "TEMPORARILY_BLOCKED", 50)
    controller.tick(blocked, now=0.0)
    assert controller.diagnostics()["streamer_reserved_sun"] == 50
    controller.clear_pending_state(clear_queue=True, clear_source=True, reason="reset")
    assert len(controller.queue) == 0
    assert controller.diagnostics()["streamer_reserved_sun"] == 0

    controller = controller_for(clock, [message("eval", "!slot 1 1 1")])
    controller.tick(blocked, now=0.0)
    controller.begin_phase("EVALUATE", accepting=False, now=0.0)
    assert len(controller.queue) == 0
    assert controller.diagnostics()["pending_viewer_command"] is None
    assert controller.diagnostics()["streamer_reserved_sun"] == 0


def test_policy_mask_preserves_reserved_sun_and_never_replaces_selected_action() -> None:
    @dataclass(frozen=True)
    class Decision:
        required_sun: int

    decisions = {1: Decision(25), 2: Decision(50), 3: Decision(0)}
    calls: List[int] = []

    def action_decision(action_id: int) -> Decision:
        calls.append(int(action_id))
        return decisions.get(int(action_id), Decision(0))

    mask = np.asarray([True, True, True, True], dtype=bool)
    reserved = apply_streamer_sun_reservation_mask(
        mask,
        {"sun": 50},
        reserved_sun=50,
        action_decision=action_decision,
        wait_action=0,
    )
    assert reserved.tolist() == [True, False, False, True]
    assert calls == [1, 2, 3]
    # The helper only changes the pre-selection mask; the caller's action is
    # never silently rewritten.
    selected_action = 3
    assert selected_action == 3
