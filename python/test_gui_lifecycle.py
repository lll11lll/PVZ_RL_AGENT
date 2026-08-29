from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

import pvzrl_gui_process
from pvzrl_gui_status import classify_live_health
from pvzrl_gui import (
    LOG_DRAIN_MAX_ITEMS,
    LOG_HISTORY_MAX_CHARS,
    LOG_HISTORY_MAX_LINES,
    LIVE_MAX_AGE_SECONDS,
    STALE_MAX_AGE_SECONDS,
    PvZDashboard,
)


class FakeRoot:
    def __init__(self) -> None:
        self.callbacks: dict[str, Callable[[], None]] = {}
        self.canceled: list[str] = []
        self.destroyed = False
        self._next_id = 0

    def after(self, _delay_ms: int, callback: Callable[[], None]) -> str:
        self._next_id += 1
        callback_id = f"after-{self._next_id}"
        self.callbacks[callback_id] = callback
        return callback_id

    def after_cancel(self, callback_id: str) -> None:
        self.canceled.append(callback_id)
        self.callbacks.pop(callback_id, None)

    def destroy(self) -> None:
        self.destroyed = True


class FakeVar:
    def __init__(self) -> None:
        self.value = ""

    def set(self, value: str) -> None:
        self.value = str(value)

    def get(self) -> str:
        return self.value


class FakeText:
    def __init__(self) -> None:
        self.configure_calls: list[dict[str, Any]] = []
        self.delete_calls: list[tuple[str, str]] = []
        self.insert_calls: list[tuple[str, str]] = []
        self.see_calls: list[str] = []

    def configure(self, **kwargs: Any) -> None:
        self.configure_calls.append(dict(kwargs))

    def delete(self, start: str, end: str) -> None:
        self.delete_calls.append((start, end))

    def insert(self, index: str, text: str) -> None:
        self.insert_calls.append((index, text))

    def see(self, index: str) -> None:
        self.see_calls.append(index)


class CountingVar(FakeVar):
    def __init__(self, value: str = "") -> None:
        super().__init__()
        self.value = value
        self.set_count = 0

    def set(self, value: str) -> None:
        self.set_count += 1
        super().set(value)


class ImmediateExitProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.terminate_calls = 0
        self.kill_calls = 0

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.returncode = 0

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return int(self.returncode or 0)


class TimeoutThenKillProcess(ImmediateExitProcess):
    def terminate(self) -> None:
        self.terminate_calls += 1

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            raise subprocess.TimeoutExpired("test", timeout)
        return int(self.returncode)


class NeverExitProcess(ImmediateExitProcess):
    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.kill_calls += 1

    def wait(self, timeout: float | None = None) -> int:
        raise subprocess.TimeoutExpired("test", timeout)


def _bare_dashboard() -> PvZDashboard:
    dashboard = PvZDashboard.__new__(PvZDashboard)
    dashboard.root = FakeRoot()
    dashboard.log_queue = queue.Queue(maxsize=10)
    dashboard._log_queue_put_lock = threading.Lock()
    dashboard._log_queue_drop_lock = threading.Lock()
    dashboard._log_queue_dropped_items = 0
    dashboard.log_history = []
    dashboard.log_history_chars = 0
    dashboard.log_dropped_lines = 0
    dashboard.log_text = None
    dashboard._poll_after_id = "poll-existing"
    dashboard._log_after_id = "log-existing"
    dashboard._close_after_id = None
    dashboard._closing = False
    dashboard._destroyed = False
    dashboard._close_deadline = 0.0
    dashboard._reader_thread = None
    dashboard._stopper_thread = None
    dashboard._stopping_process = None
    dashboard.active_process = None
    dashboard.active_process_name = ""
    dashboard.active_process_started_at = None
    dashboard.active_process_started_wall_time = None
    dashboard.process_lifecycle_state = "OFFLINE"
    dashboard.process_lifecycle_detail = ""
    dashboard.live_writer_warning_emitted = False
    dashboard.process_status_var = FakeVar()
    dashboard.process_lifecycle_var = FakeVar()
    dashboard.last_good_status = None
    dashboard.last_live_health = ""
    dashboard.launch_buttons = []
    dashboard.stop_buttons = []
    return dashboard


def _status_dashboard(path: Path) -> PvZDashboard:
    dashboard = PvZDashboard.__new__(PvZDashboard)
    dashboard.live_status_path = path
    return dashboard


def test_close_reuses_process_stop_and_cancels_callbacks() -> None:
    dashboard = _bare_dashboard()
    process = ImmediateExitProcess()
    dashboard.active_process = process
    dashboard.active_process_name = "training"

    dashboard._on_close()
    dashboard._on_close()

    assert process.terminate_calls == 1
    assert process.kill_calls == 0
    assert dashboard._destroyed is True
    assert dashboard.root.destroyed is True
    assert "poll-existing" in dashboard.root.canceled
    assert "log-existing" in dashboard.root.canceled
    assert dashboard.active_process is None


def test_close_drains_bounded_queue_before_destroy() -> None:
    dashboard = _bare_dashboard()
    dashboard.log_queue = queue.Queue(maxsize=2000)
    for index in range(2000):
        dashboard.log_queue.put_nowait(f"line {index}\n")
    dashboard._on_close()
    assert dashboard.root.destroyed is True
    assert dashboard.log_queue.empty()


def test_close_hard_deadline_destroys_root_when_process_never_exits() -> None:
    dashboard = _bare_dashboard()
    process = NeverExitProcess()
    dashboard.active_process = process
    dashboard.active_process_name = "training"

    dashboard._on_close()
    assert dashboard._stopper_thread is not None
    dashboard._stopper_thread.join(timeout=1.0)
    dashboard._close_deadline = time.monotonic() - 1.0
    dashboard._poll_close_cleanup()

    assert process.terminate_calls == 1
    assert process.kill_calls >= 1
    assert dashboard._destroyed is True
    assert dashboard.root.destroyed is True
    assert dashboard._close_after_id is None


def test_stop_escalates_only_after_grace_timeout() -> None:
    dashboard = _bare_dashboard()
    process = TimeoutThenKillProcess()
    process.terminate()
    dashboard._wait_then_kill("training", process)
    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.poll() == -9


def test_begin_stop_is_idempotent_for_same_process() -> None:
    dashboard = _bare_dashboard()
    process = ImmediateExitProcess()
    dashboard.active_process = process
    dashboard.active_process_name = "training"

    dashboard._begin_process_stop("training", process)
    dashboard._begin_process_stop("training", process)
    assert dashboard._stopper_thread is not None
    dashboard._stopper_thread.join(timeout=1.0)

    assert process.terminate_calls == 1
    assert process.kill_calls == 0


def test_stale_exit_event_does_not_clear_newer_process() -> None:
    dashboard = _bare_dashboard()
    current = ImmediateExitProcess()
    stale = ImmediateExitProcess()
    dashboard.active_process = current
    dashboard.active_process_name = "current"

    dashboard._handle_process_exit("stale", stale, 0)

    assert dashboard.active_process is current
    assert dashboard.active_process_name == "current"
    assert dashboard.process_status_var.value == ""


def test_second_launch_is_rejected_without_starting_subprocess() -> None:
    dashboard = _bare_dashboard()
    dashboard.active_process = ImmediateExitProcess()
    dashboard.active_process_name = "current"
    messages: list[str] = []
    dashboard._append_log = messages.append

    dashboard.launch_process("replacement", ["python", "replacement.py"])

    assert dashboard.active_process_name == "current"
    assert messages == ["ERROR: Cannot launch replacement; current is still running.\n"]


def test_launch_process_preserves_popen_contract(tmp_path: Path, monkeypatch: Any) -> None:
    dashboard = _bare_dashboard()
    dashboard.project_root = tmp_path
    dashboard.repo_root = tmp_path
    dashboard.active_run_var = FakeVar()
    captured: dict[str, Any] = {}

    class Process(ImmediateExitProcess):
        stdout = None

    def fake_popen(command: list[str], **kwargs: Any) -> Process:
        captured["command"] = list(command)
        captured["kwargs"] = dict(kwargs)
        return Process()

    monkeypatch.setattr(pvzrl_gui_process.subprocess, "Popen", fake_popen)
    command = [
        "python",
        "train.py",
        "--run-dir",
        "runs/snapshot",
        "--live-status-path",
        "runs/live_status.json",
    ]

    dashboard.launch_process("training", command)
    assert dashboard._reader_thread is not None
    dashboard._reader_thread.join(timeout=1.0)

    assert captured["command"] == command
    assert captured["kwargs"]["cwd"] == str(tmp_path)
    assert captured["kwargs"]["env"]["PYTHONUNBUFFERED"] == "1"
    assert captured["kwargs"]["stdout"] is subprocess.PIPE
    assert captured["kwargs"]["stderr"] is subprocess.STDOUT
    assert captured["kwargs"]["text"] is True
    assert captured["kwargs"]["bufsize"] == 1
    assert dashboard.active_run_path == str((tmp_path / "runs/snapshot").resolve())
    assert dashboard.process_status_var.value == "STARTING: training"


def test_fresh_blocked_streamer_status_prevents_duplicate_backend_launch() -> None:
    dashboard = _bare_dashboard()
    dashboard.last_live_health = "BLOCKED_SEED_SELECTION"
    dashboard.last_good_status = {
        "status": "blocked",
        "streamer_v1_enabled": True,
        "run_mode": "adventure_generalist_14slot_train",
        "active_run": "runs/external_streamer",
    }
    messages: list[str] = []
    dashboard._append_log = messages.append

    dashboard.launch_process("duplicate", ["python", "train.py"])

    assert dashboard.active_process is None
    assert dashboard.process_lifecycle_state == "ERROR"
    assert any("another backend appears active" in message for message in messages)


def test_launch_freshly_reads_the_command_status_target_before_popen(
    tmp_path: Path, monkeypatch: Any
) -> None:
    dashboard = _bare_dashboard()
    dashboard.project_root = tmp_path
    dashboard.live_status_path = tmp_path / "previous.json"
    dashboard._resolve_text_path = lambda raw: (tmp_path / raw).resolve()
    target = tmp_path / "target" / "live_status.json"
    target.parent.mkdir(parents=True)
    target.write_text(
        json.dumps(
            {
                "status": "running",
                "run_mode": "adventure_generalist_14slot_train",
                "active_run": "runs/external-target",
            }
        ),
        encoding="utf-8",
    )
    messages: list[str] = []
    dashboard._append_log = messages.append

    def forbidden_popen(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("duplicate launch reached Popen")

    monkeypatch.setattr(pvzrl_gui_process.subprocess, "Popen", forbidden_popen)
    dashboard.launch_process(
        "duplicate",
        ["python", "train.py", "--live-status-path", "target/live_status.json"],
    )

    assert dashboard.active_process is None
    assert any("another backend appears active" in message for message in messages)


def test_intentional_nonzero_stop_is_offline_but_spontaneous_exit_is_error() -> None:
    stopped = _bare_dashboard()
    stopped_process = ImmediateExitProcess()
    stopped_process.returncode = 1
    stopped.active_process = stopped_process
    stopped.active_process_name = "training"
    stopped._stopping_process = stopped_process
    stopped._handle_process_exit("training", stopped_process, 1)
    assert stopped.process_lifecycle_state == "OFFLINE"
    assert stopped.process_lifecycle_detail == "training stopped"

    crashed = _bare_dashboard()
    crashed_process = ImmediateExitProcess()
    crashed_process.returncode = 1
    crashed.active_process = crashed_process
    crashed.active_process_name = "training"
    crashed._handle_process_exit("training", crashed_process, 1)
    assert crashed.process_lifecycle_state == "ERROR"
    assert crashed.process_lifecycle_detail == "training exited 1"


def test_poll_and_log_callbacks_are_tracked_and_bounded() -> None:
    dashboard = _bare_dashboard()
    dashboard._poll_after_id = None
    dashboard._log_after_id = None
    dashboard._refresh_diagnostics = lambda auto: None

    dashboard._poll()
    assert dashboard._poll_after_id in dashboard.root.callbacks

    dashboard.log_queue = queue.Queue(maxsize=LOG_DRAIN_MAX_ITEMS + 25)
    for index in range(LOG_DRAIN_MAX_ITEMS + 25):
        dashboard.log_queue.put_nowait(f"line {index}\n")
    dashboard._drain_log_queue()
    assert dashboard.log_queue.qsize() == 25
    assert dashboard._log_after_id in dashboard.root.callbacks


def test_log_queue_and_retained_history_are_bounded_with_drop_accounting() -> None:
    dashboard = _bare_dashboard()
    dashboard.log_queue = queue.Queue(maxsize=2)
    assert dashboard._queue_log_item("first\n")
    assert dashboard._queue_log_item("second\n")
    assert not dashboard._queue_log_item("third\n")
    assert dashboard._queue_log_item("critical\n", critical=True)
    dashboard._consume_log_queue(10, 1.0)
    assert any("log queue dropped 2 item(s)" in line for line in dashboard.log_history)

    dashboard._append_log("x\n" * (LOG_HISTORY_MAX_LINES + 20))
    assert len(dashboard.log_history) <= LOG_HISTORY_MAX_LINES
    assert dashboard.log_history_chars <= LOG_HISTORY_MAX_CHARS
    assert dashboard.log_dropped_lines >= 20


def test_log_rollover_updates_widget_incrementally_without_full_rebuild() -> None:
    dashboard = _bare_dashboard()
    dashboard.log_text = FakeText()
    dashboard.log_history = ["old\n"] * LOG_HISTORY_MAX_LINES
    dashboard.log_history_chars = sum(len(line) for line in dashboard.log_history)
    dashboard.log_pause_autoscroll_var = FakeVar()

    dashboard._append_log("new\n" * 25)

    assert ("1.0", "26.0") in dashboard.log_text.delete_calls
    assert ("1.0", "end") not in dashboard.log_text.delete_calls
    assert dashboard.log_text.insert_calls[0][0] == "1.0"
    assert dashboard.log_text.insert_calls[-1] == ("end", "new\n" * 25)
    assert len(dashboard.log_history) == LOG_HISTORY_MAX_LINES


def test_oversized_incoming_log_text_never_bypasses_retention_in_widget() -> None:
    dashboard = _bare_dashboard()
    dashboard.log_text = FakeText()
    dashboard.log_pause_autoscroll_var = FakeVar()

    dashboard._append_log("x" * (LOG_HISTORY_MAX_CHARS + 1))

    assert dashboard.log_history == []
    assert dashboard.log_history_chars == 0
    assert dashboard.log_dropped_lines == 1
    assert not any(index == "end" and len(text) > LOG_HISTORY_MAX_CHARS for index, text in dashboard.log_text.insert_calls)


def test_oversized_incoming_log_batch_inserts_only_retained_suffix() -> None:
    dashboard = _bare_dashboard()
    dashboard.log_text = FakeText()
    dashboard.log_pause_autoscroll_var = FakeVar()
    incoming = "".join(f"line {index}\n" for index in range(LOG_HISTORY_MAX_LINES + 25))

    dashboard._append_log(incoming)

    retained = "".join(f"line {index}\n" for index in range(25, LOG_HISTORY_MAX_LINES + 25))
    assert dashboard.log_history == retained.splitlines(keepends=True)
    assert dashboard.log_text.insert_calls[-1] == ("end", retained)


def test_partial_line_rollover_rebuilds_from_bounded_history() -> None:
    dashboard = _bare_dashboard()
    dashboard.log_text = FakeText()
    dashboard.log_filter_var = FakeVar()
    dashboard.log_severity_var = FakeVar()
    dashboard.log_severity_var.value = "All"
    dashboard.log_pause_autoscroll_var = FakeVar()
    dashboard.log_history = ["partial"] + [f"line {index}\n" for index in range(LOG_HISTORY_MAX_LINES - 1)]
    dashboard.log_history_chars = sum(len(line) for line in dashboard.log_history)

    dashboard._append_log("new line\n")

    expected = dashboard._log_drop_notice() + "".join(dashboard.log_history)
    assert ("1.0", "end") in dashboard.log_text.delete_calls
    assert dashboard.log_text.insert_calls[-1] == ("1.0", expected)
    assert dashboard.log_history[0] == "line 0\n"


def test_identical_live_and_diagnostics_labels_skip_tcl_writes(tmp_path: Path) -> None:
    dashboard = PvZDashboard.__new__(PvZDashboard)
    dashboard.live_status_path = tmp_path / "live_status.json"
    dashboard.live_status_var = CountingVar()
    dashboard.diagnostics_status_var = CountingVar()
    dashboard.last_live_parse_error = ""
    dashboard.last_good_status = {"status": "running"}
    dashboard.last_good_read_time = 100.0
    info = {
        "path": dashboard.live_status_path,
        "exists": True,
        "size": 20,
        "mtime": 100.0,
        "age": 1.0,
        "health": "LIVE",
        "parse_error": "",
    }

    dashboard._set_live_status(info)
    dashboard._set_diagnostics_status(info, using_last_good=False)
    dashboard._set_live_status(dict(info))
    dashboard._set_diagnostics_status(dict(info), using_last_good=False)

    assert dashboard.live_status_var.set_count == 1
    assert dashboard.diagnostics_status_var.set_count == 1


def test_filtered_log_bursts_schedule_only_one_full_refresh() -> None:
    dashboard = _bare_dashboard()
    dashboard.log_text = FakeText()
    dashboard.log_filter_var = FakeVar()
    dashboard.log_filter_var.value = "needle"
    dashboard.log_severity_var = FakeVar()
    dashboard.log_severity_var.value = "All"
    dashboard.log_pause_autoscroll_var = FakeVar()
    dashboard._last_log_view_refresh_at = time.monotonic()
    dashboard._log_view_after_id = None
    refresh_calls: list[bool] = []
    dashboard._refresh_log_view = lambda: refresh_calls.append(True)

    dashboard._append_log("needle one\n")
    callback_id = dashboard._log_view_after_id
    dashboard._append_log("needle two\n")

    assert callback_id is not None
    assert dashboard._log_view_after_id == callback_id
    assert len(dashboard.root.callbacks) == 1
    dashboard.root.callbacks[callback_id]()
    assert refresh_calls == [True]


def test_unchanged_live_status_uses_cached_parse_and_recomputes_age(tmp_path: Path) -> None:
    path = tmp_path / "live_status.json"
    path.write_text(json.dumps({"status": "running", "value": 1}), encoding="utf-8")
    dashboard = _status_dashboard(path)

    first, first_info = dashboard._read_live_status_file()
    second, second_info = dashboard._read_live_status_file()

    assert first == second == {"status": "running", "value": 1}
    assert not first_info.get("unchanged", False)
    assert second_info["unchanged"] is True
    assert second_info["health"] == "LIVE"


def test_status_age_overrides_old_blocked_payload(tmp_path: Path) -> None:
    path = tmp_path / "live_status.json"
    path.write_text(json.dumps({"blocked_reason": "post_win_next_screen_timeout"}), encoding="utf-8")
    old = time.time() - STALE_MAX_AGE_SECONDS - 5.0
    os.utime(path, (old, old))
    dashboard = _status_dashboard(path)

    _, info = dashboard._read_live_status_file()
    assert info["health"] == "DEAD"
    assert classify_live_health(
        LIVE_MAX_AGE_SECONDS + 0.01,
        {"blocked_reason": "post_win_timeout"},
    ) == "STALE"


def test_health_aliases_preserve_case_insensitive_empty_fallbacks(tmp_path: Path) -> None:
    assert classify_live_health(
        0.0,
        {"blocked_reason": "", "adventure": {"blocked_reason": "post_win_next_screen_timeout"}},
    ) == "BLOCKED_POST_WIN"
    assert classify_live_health(
        0.0,
        {"Blocked_Reason": "post_win_next_screen_timeout"},
    ) == "BLOCKED_POST_WIN"


def test_status_index_reuses_case_insensitive_alias_map() -> None:
    dashboard = PvZDashboard.__new__(PvZDashboard)
    payload = {"Primary": "", "Nested": {"Value": 7}}

    assert dashboard._lookup_path(payload, "nested.value") == 7
    first_index = dashboard._normalized_status_index
    assert dashboard._first_value(payload, ["primary", "nested.value"]) == 7
    assert dashboard._status_index_for(payload["Nested"]) is first_index


def test_unchanged_coach_fields_do_not_rewrite_string_vars() -> None:
    dashboard = PvZDashboard.__new__(PvZDashboard)
    payload = {
        "human_coach_enabled": True,
        "stream_coach_enabled": True,
        "stream_coach_mode": "mock",
        "fusion_bridge_enabled": True,
    }
    before = set(vars(dashboard))
    dashboard._set_coach_live_fields(payload)
    created = [name for name in vars(dashboard) if name not in before and name.endswith("_var")]
    for name in created:
        current = getattr(dashboard, name).get()
        setattr(dashboard, name, CountingVar(str(current)))

    dashboard._set_coach_live_fields(payload)
    assert sum(getattr(dashboard, name).set_count for name in created) == 0

    changed = dict(payload)
    changed["human_coach_enabled"] = False
    dashboard._set_coach_live_fields(changed)
    assert dashboard.human_coach_enabled_status_var.set_count == 1


def test_malformed_cache_recovers_after_rotation(tmp_path: Path) -> None:
    path = tmp_path / "live_status.json"
    path.write_text("{bad", encoding="utf-8")
    dashboard = _status_dashboard(path)

    payload, malformed = dashboard._read_live_status_file()
    _, unchanged = dashboard._read_live_status_file()
    assert payload is None
    assert malformed["health"] == "MALFORMED"
    assert unchanged["unchanged"] is True

    replacement = tmp_path / "replacement.json"
    replacement.write_text(json.dumps({"status": "recovered"}), encoding="utf-8")
    os.replace(replacement, path)
    recovered, info = dashboard._read_live_status_file()
    assert recovered == {"status": "recovered"}
    assert info["health"] == "LIVE"
    assert not info.get("unchanged", False)


def test_non_object_live_status_is_cached_as_malformed(tmp_path: Path) -> None:
    path = tmp_path / "live_status.json"
    path.write_text("[]", encoding="utf-8")
    dashboard = _status_dashboard(path)

    payload, malformed = dashboard._read_live_status_file()
    cached, unchanged = dashboard._read_live_status_file()

    assert payload is cached is None
    assert malformed["health"] == unchanged["health"] == "MALFORMED"
    assert malformed["parse_error"] == "Expected JSON object, got list"
    assert unchanged["unchanged"] is True


def test_unchanged_payload_skips_full_render_until_view_state_changes() -> None:
    dashboard = PvZDashboard.__new__(PvZDashboard)
    render_calls: list[tuple[object, str, bool]] = []
    empty_calls: list[str] = []
    dashboard._render = lambda payload, health, using_last_good: render_calls.append(
        (payload, health, using_last_good)
    )
    dashboard._render_no_status = lambda health: empty_calls.append(health)
    payload = {"status": "running"}

    dashboard._render_diagnostics_payload(payload, "LIVE", False)
    dashboard._render_diagnostics_payload(payload, "LIVE", False)
    dashboard._render_diagnostics_payload(payload, "STALE", False)
    dashboard._render_diagnostics_payload(dict(payload), "STALE", False)
    dashboard._render_diagnostics_payload({**payload, "updated_at": 10.0}, "STALE", False)
    dashboard._render_diagnostics_payload({**payload, "updated_at": 11.0}, "STALE", False)
    coach_payload = {**payload, "updated_at": 11.0, "coach": {"stream_coach_last_poll_age_seconds": 1.0}}
    dashboard._render_diagnostics_payload(coach_payload, "STALE", False)
    dashboard._render_diagnostics_payload(
        {**coach_payload, "coach": {"stream_coach_last_poll_age_seconds": 2.0}},
        "STALE",
        False,
    )
    dashboard._render_diagnostics_payload({"status": "changed"}, "STALE", False)
    dashboard._render_diagnostics_payload(None, "MISSING", False)
    dashboard._render_diagnostics_payload(None, "MISSING", False)

    assert len(render_calls) == 5
    assert [call[1] for call in render_calls] == ["LIVE", "STALE", "STALE", "STALE", "STALE"]
    assert empty_calls == ["MISSING"]
