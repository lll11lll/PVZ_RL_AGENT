"""Bounded incremental tailing for append-only coach command files."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


DEFAULT_MAX_READ_BYTES = 1024 * 1024
DEFAULT_MAX_PENDING_BYTES = 4 * DEFAULT_MAX_READ_BYTES
_ANCHOR_BYTES = 128


class IncrementalLineTailReader:
    """Read complete appended UTF-8 lines without rereading partial records.

    The reader follows path replacement and in-place truncation/rewrite. File
    identity is the primary signal; bounded prefix/offset anchors provide a
    fallback for platforms with weak file IDs and for same-inode rewrites.
    """

    def __init__(
        self,
        path: Union[str, Path],
        *,
        start_at_end: bool = True,
        max_read_bytes: int = DEFAULT_MAX_READ_BYTES,
        max_pending_bytes: int = DEFAULT_MAX_PENDING_BYTES,
    ) -> None:
        self.path = Path(path)
        self.max_read_bytes = max(1, int(max_read_bytes))
        self.max_pending_bytes = max(self.max_read_bytes, int(max_pending_bytes))
        # `offset` is the committed position after the last complete record.
        # `_read_offset` may be ahead while an incomplete trailing record is
        # buffered, avoiding repeated I/O without logically consuming it.
        self.offset = 0
        self._read_offset = 0
        self._pending = b""
        self._discard_until_newline = False
        self._identity: Optional[Tuple[int, int]] = None
        self._head_anchor: Optional[Tuple[int, bytes]] = None
        self._anchor: Optional[Tuple[int, bytes]] = None
        self.rotation_count = 0
        self.truncation_count = 0
        self.decode_error_count = 0
        self.malformed_record_count = 0
        self.oversized_record_count = 0
        self.lines_emitted = 0
        self.bytes_read = 0
        self.last_error = ""
        self.last_clear_had_bytes = False

        try:
            stat = self.path.stat()
        except (FileNotFoundError, OSError) as exc:
            if not isinstance(exc, FileNotFoundError):
                self.last_error = str(exc)
            return
        self._identity = self._file_identity(stat)
        if start_at_end:
            self.offset = int(stat.st_size)
            self._read_offset = self.offset
            self._discard_until_newline = self._ends_with_partial_record(int(stat.st_size))
        self._refresh_anchor()

    @property
    def pending_bytes(self) -> int:
        return len(self._pending)

    @property
    def read_offset(self) -> int:
        return int(self._read_offset)

    @property
    def discarding_cleared_partial(self) -> bool:
        return bool(self._discard_until_newline)

    def note_malformed_record(self, reason: str = "malformed_record") -> None:
        self.malformed_record_count += 1
        self.last_error = str(reason or "malformed_record")

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "offset": int(self.offset),
            "read_offset": int(self._read_offset),
            "pending_bytes": int(len(self._pending)),
            "discarding_cleared_partial": bool(self._discard_until_newline),
            "rotation_count": int(self.rotation_count),
            "truncation_count": int(self.truncation_count),
            "decode_error_count": int(self.decode_error_count),
            "malformed_record_count": int(self.malformed_record_count),
            "oversized_record_count": int(self.oversized_record_count),
            "lines_emitted": int(self.lines_emitted),
            "bytes_read": int(self.bytes_read),
            "last_error": str(self.last_error or ""),
        }

    def read_lines(self) -> List[str]:
        stat = self._prepare_for_read()
        if stat is None or int(stat.st_size) <= self._read_offset:
            return []

        try:
            with self.path.open("rb") as handle:
                handle.seek(self._read_offset)
                chunk = handle.read(self.max_read_bytes)
        except OSError as exc:
            self.last_error = str(exc)
            return []
        if not chunk:
            return []

        chunk_start = self._read_offset
        self._read_offset += len(chunk)
        self.bytes_read += len(chunk)
        if self._discard_until_newline:
            newline = chunk.find(b"\n")
            if newline < 0:
                self.offset = self._read_offset
                self._refresh_anchor()
                return []
            chunk = chunk[newline + 1 :]
            self.offset = chunk_start + newline + 1
            self._discard_until_newline = False

        data = self._pending + chunk
        parts = data.split(b"\n")
        self._pending = parts.pop() if parts else data
        self.offset = self._read_offset - len(self._pending)
        if len(self._pending) > self.max_pending_bytes:
            self._pending = b""
            self._discard_until_newline = True
            self.offset = self._read_offset
            self.oversized_record_count += 1
            self.last_error = "record_too_large"

        lines: List[str] = []
        for raw in parts:
            if raw.endswith(b"\r"):
                raw = raw[:-1]
            try:
                line = raw.decode("utf-8")
            except UnicodeDecodeError:
                self.decode_error_count += 1
                self.last_error = "utf8_decode_error"
                continue
            lines.append(line)
        self.lines_emitted += len(lines)
        self._refresh_anchor()
        return lines

    def clear_to_end(self) -> int:
        """Discard queued/unread bytes and return nonblank logical records skipped."""

        stat = self._prepare_for_read()
        current_has_bytes = bool(self._pending)
        current_nonblank = bool(self._pending.strip())
        had_bytes = bool(self._pending)
        self._pending = b""
        skipped = 0
        scanned_new_bytes = False

        if stat is not None and int(stat.st_size) > self.offset:
            try:
                with self.path.open("rb") as handle:
                    handle.seek(self._read_offset)
                    while True:
                        chunk = handle.read(self.max_read_bytes)
                        if not chunk:
                            break
                        scanned_new_bytes = True
                        had_bytes = True
                        self._read_offset += len(chunk)
                        self.bytes_read += len(chunk)
                        segments = chunk.split(b"\n")
                        for segment in segments[:-1]:
                            current_has_bytes = current_has_bytes or bool(segment)
                            current_nonblank = current_nonblank or bool(segment.rstrip(b"\r").strip())
                            if current_nonblank:
                                skipped += 1
                            current_has_bytes = False
                            current_nonblank = False
                        tail = segments[-1]
                        current_has_bytes = current_has_bytes or bool(tail)
                        current_nonblank = current_nonblank or bool(tail.strip())
            except OSError as exc:
                self.last_error = str(exc)
                self.last_clear_had_bytes = had_bytes
                return int(skipped)

        if current_nonblank:
            skipped += 1
        self.offset = self._read_offset
        if had_bytes or scanned_new_bytes:
            self._discard_until_newline = bool(current_has_bytes)
        self.last_clear_had_bytes = bool(had_bytes)
        self._refresh_anchor()
        return int(skipped)

    def _prepare_for_read(self) -> Optional[os.stat_result]:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            self.offset = 0
            self._read_offset = 0
            self._pending = b""
            self._discard_until_newline = False
            self._identity = None
            self._head_anchor = None
            self._anchor = None
            return None
        except OSError as exc:
            self.last_error = str(exc)
            return None

        identity = self._file_identity(stat)
        if self._identity is not None and identity is not None and identity != self._identity:
            self._reset_for_replacement()
            self.rotation_count += 1
        elif int(stat.st_size) < self._read_offset:
            self._reset_for_replacement()
            self.truncation_count += 1
        elif self._read_offset > 0 and self._anchor is not None:
            anchors_match = self._anchor_matches()
            if anchors_match is None:
                return None
            if not anchors_match:
                self._reset_for_replacement()
                self.truncation_count += 1
        self._identity = identity
        return stat

    def _reset_for_replacement(self) -> None:
        self.offset = 0
        self._read_offset = 0
        self._pending = b""
        self._discard_until_newline = False
        self._head_anchor = None
        self._anchor = None

    @staticmethod
    def _file_identity(stat: os.stat_result) -> Optional[Tuple[int, int]]:
        inode = int(getattr(stat, "st_ino", 0) or 0)
        if inode <= 0:
            return None
        return int(getattr(stat, "st_dev", 0) or 0), inode

    def _ends_with_partial_record(self, size: int) -> bool:
        if size <= 0:
            return False
        try:
            with self.path.open("rb") as handle:
                handle.seek(size - 1)
                return handle.read(1) != b"\n"
        except OSError as exc:
            self.last_error = str(exc)
            return False

    def _anchor_matches(self) -> Optional[bool]:
        if self._anchor is None and self._head_anchor is None:
            return True
        try:
            with self.path.open("rb") as handle:
                if self._head_anchor is not None:
                    head_length, expected_head = self._head_anchor
                    actual_head = handle.read(head_length)
                    if hashlib.sha256(actual_head).digest() != expected_head:
                        return False
                if self._anchor is None:
                    return True
                anchor_offset, expected = self._anchor
                handle.seek(anchor_offset)
                actual = handle.read(min(_ANCHOR_BYTES, self._read_offset - anchor_offset))
        except OSError as exc:
            self.last_error = str(exc)
            return None
        return hashlib.sha256(actual).digest() == expected

    def _refresh_anchor(self) -> None:
        if self._read_offset <= 0 or not self.path.exists():
            self._head_anchor = None
            self._anchor = None
            return
        anchor_offset = max(0, self._read_offset - _ANCHOR_BYTES)
        try:
            with self.path.open("rb") as handle:
                head_length = min(_ANCHOR_BYTES, self._read_offset)
                head_data = handle.read(head_length)
                handle.seek(anchor_offset)
                data = handle.read(self._read_offset - anchor_offset)
        except OSError as exc:
            self.last_error = str(exc)
            return
        self._head_anchor = (head_length, hashlib.sha256(head_data).digest())
        self._anchor = (anchor_offset, hashlib.sha256(data).digest())
