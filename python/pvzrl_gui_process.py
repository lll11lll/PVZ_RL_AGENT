"""Subprocess lifecycle and bounded log draining for the Tk dashboard."""

from __future__ import annotations

import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, List

import tkinter as tk


LOG_POLL_MS = 100
LOG_BACKLOG_POLL_MS = 1
LOG_DRAIN_MAX_ITEMS = 250
LOG_DRAIN_BUDGET_SECONDS = 0.012
LOG_QUEUE_MAX_ITEMS = 10_000
STOP_GRACE_SECONDS = 5.0
STOP_KILL_WAIT_SECONDS = 2.0


class ProcessLogMixin:
    def launch_process(self, name: str, command: List[str]) -> None:
        if self.active_process is not None:
            self._append_log(f"ERROR: Cannot launch {name}; {self.active_process_name} is still running.\n")
            return
        self._append_log(f"\nStarting subprocess: {name}\n")
        model_path = self._command_arg(command, "--model-path")
        resume_model_path = self._command_arg(command, "--resume-model-path")
        run_dir = self._command_arg(command, "--run-dir")
        live_status_arg = self._command_arg(command, "--live-status-path")
        if model_path:
            self._append_log(f"Selected model path: {model_path}\n")
        if resume_model_path:
            self._append_log(f"Selected resume model path: {resume_model_path}\n")
        if live_status_arg:
            try:
                resolved_live_status = self._resolve_text_path(live_status_arg)
            except (OSError, ValueError):
                resolved_live_status = Path(live_status_arg)
            self._append_log(f"[gui] live status path: {resolved_live_status}\n")
        if run_dir:
            self._append_log(f"Selected run directory: {run_dir}\n")
            self.active_run_path = str(self._resolve_text_path(run_dir))
            self.active_run_var.set(f"Active run: {self.active_run_path}")
        elif ("--adventure-eval" in command or "--adventure-generalist-eval" in command) and model_path:
            self.active_run_path = str(self._resolve_text_path(model_path).parent)
            self.active_run_var.set(f"Active run: {self.active_run_path}")
            self._append_log(f"Inferred Adventure run directory: {self.active_run_path}\n")
        else:
            self.active_run_path = ""
            self.active_run_var.set("Active run: unknown")
        self._append_log(f"Launching with cwd: {self.project_root}\n")
        self._append_log(f"$ {subprocess.list2cmdline(command)}\n")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            process = subprocess.Popen(
                command,
                cwd=str(self.project_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            self._append_log(f"ERROR: Failed to launch {name}: {exc}\n")
            self.process_status_var.set("Idle")
            return

        self.active_process = process
        self.active_process_name = name
        self.active_process_started_at = time.monotonic()
        self.live_writer_warning_emitted = False
        self.process_status_var.set(f"Running: {name}")
        self._set_running(True)
        self._reader_thread = threading.Thread(
            target=self._read_process_output,
            args=(name, process),
            name=f"{name} log reader",
            daemon=True,
        )
        self._reader_thread.start()

    def _command_arg(self, command: List[str], flag: str) -> str:
        try:
            index = command.index(flag)
        except ValueError:
            return ""
        next_index = index + 1
        if next_index >= len(command):
            return ""
        return command[next_index]

    def _read_process_output(self, name: str, process: subprocess.Popen[str]) -> None:
        try:
            if process.stdout is not None:
                for line in process.stdout:
                    self._queue_log_item(line)
        finally:
            try:
                exit_code = process.wait()
            except OSError:
                exit_code = process.poll()
                if exit_code is None:
                    exit_code = -1
            self._queue_log_item(("process_exit", name, process, exit_code), critical=True)

    def stop_active_process(self) -> None:
        process = self.active_process
        if process is None:
            self._append_log("No active process to stop.\n")
            return
        exit_code = process.poll()
        if exit_code is not None:
            self._handle_process_exit(self.active_process_name or "process", process, int(exit_code))
            return
        name = self.active_process_name or "process"
        self._begin_process_stop(name, process)

    def _begin_process_stop(self, name: str, process: subprocess.Popen[str]) -> None:
        if self._stopping_process is process:
            return
        self._stopping_process = process
        for button in self.stop_buttons:
            button.configure(state="disabled")
        try:
            process.terminate()
            self._append_log(f"[{name}] terminate() sent.\n")
        except OSError as exc:
            self._append_log(f"ERROR: Failed to terminate {name}: {exc}\n")
        self._stopper_thread = threading.Thread(
            target=self._wait_then_kill,
            args=(name, process),
            name=f"{name} stopper",
            daemon=True,
        )
        self._stopper_thread.start()

    def _wait_then_kill(self, name: str, process: subprocess.Popen[str]) -> None:
        try:
            process.wait(timeout=STOP_GRACE_SECONDS)
            self._queue_log_item(f"[{name}] process exited after terminate().\n")
        except subprocess.TimeoutExpired:
            self._queue_log_item(f"[{name}] terminate() timed out; kill() used.\n")
            try:
                process.kill()
                process.wait(timeout=STOP_KILL_WAIT_SECONDS)
            except subprocess.TimeoutExpired:
                self._queue_log_item(
                    f"ERROR: {name} did not exit within {STOP_KILL_WAIT_SECONDS:.1f}s after kill().\n"
                )
            except OSError as exc:
                self._queue_log_item(f"ERROR: Failed to kill {name}: {exc}\n")
        except OSError as exc:
            self._queue_log_item(f"ERROR: Failed while waiting for {name} to stop: {exc}\n")

    def _queue_log_item(self, item: Any, *, critical: bool = False) -> bool:
        with self._log_queue_put_lock:
            try:
                self.log_queue.put_nowait(item)
                return True
            except queue.Full:
                if not critical:
                    with self._log_queue_drop_lock:
                        self._log_queue_dropped_items += 1
                    return False
                try:
                    self.log_queue.get_nowait()
                except queue.Empty:
                    pass
                else:
                    with self._log_queue_drop_lock:
                        self._log_queue_dropped_items += 1
                try:
                    self.log_queue.put_nowait(item)
                    return True
                except queue.Full:
                    with self._log_queue_drop_lock:
                        self._log_queue_dropped_items += 1
                    return False

    def _take_log_queue_drop_count(self) -> int:
        with self._log_queue_drop_lock:
            dropped = int(self._log_queue_dropped_items)
            self._log_queue_dropped_items = 0
        return dropped

    def _schedule_after(self, attribute: str, delay_ms: int, callback: Any) -> None:
        self._cancel_after(attribute)
        if self._destroyed:
            return
        try:
            callback_id = self.root.after(delay_ms, callback)
        except tk.TclError:
            return
        setattr(self, attribute, callback_id)

    def _cancel_after(self, attribute: str) -> None:
        callback_id = getattr(self, attribute, None)
        if callback_id is None:
            return
        setattr(self, attribute, None)
        try:
            self.root.after_cancel(callback_id)
        except (tk.TclError, ValueError):
            pass

    def _cancel_scheduled_callbacks(self, *, include_close: bool = True) -> None:
        self._cancel_after("_poll_after_id")
        self._cancel_after("_log_after_id")
        if include_close:
            self._cancel_after("_close_after_id")

    def _drain_log_queue(self) -> None:
        self._log_after_id = None
        if self._closing or self._destroyed:
            return
        self._consume_log_queue(LOG_DRAIN_MAX_ITEMS, LOG_DRAIN_BUDGET_SECONDS)
        delay_ms = LOG_BACKLOG_POLL_MS if not self.log_queue.empty() else LOG_POLL_MS
        self._schedule_after("_log_after_id", delay_ms, self._drain_log_queue)

    def _consume_log_queue(self, max_items: int, budget_seconds: float) -> int:
        started_at = time.monotonic()
        processed = 0
        pending_text: List[str] = []
        dropped = self._take_log_queue_drop_count()
        if dropped:
            pending_text.append(f"[gui] log queue dropped {dropped} item(s) while producers outpaced the UI.\n")
        while processed < max_items and time.monotonic() - started_at < budget_seconds:
            try:
                item = self.log_queue.get_nowait()
            except queue.Empty:
                break
            processed += 1
            if isinstance(item, tuple) and item and item[0] == "process_exit":
                if pending_text:
                    self._append_log("".join(pending_text))
                    pending_text.clear()
                _, name, process, exit_code = item
                self._handle_process_exit(str(name), process, int(exit_code))
            else:
                pending_text.append(str(item))
        if pending_text:
            self._append_log("".join(pending_text))
        return processed

    def _handle_process_exit(self, name: str, process: subprocess.Popen[str], exit_code: int) -> None:
        if self.active_process is not process and self._stopping_process is not process:
            return
        self._append_log(f"[{name}] exited with code {exit_code}\n")
        if self.active_process is process:
            self.active_process = None
            self.active_process_name = ""
            self.active_process_started_at = None
            self._stopping_process = None
            self.live_writer_warning_emitted = False
            self.process_status_var.set("Idle")
            if not self._closing:
                self._set_running(False)
        if self._reader_thread is not None and not self._reader_thread.is_alive():
            self._reader_thread = None
        if self._stopper_thread is not None and not self._stopper_thread.is_alive():
            self._stopper_thread = None

    def _set_running(self, running: bool) -> None:
        state = "disabled" if running else "normal"
        for button in self.launch_buttons:
            button.configure(state=state)
        for button in self.stop_buttons:
            button.configure(state="normal" if running else "disabled")
