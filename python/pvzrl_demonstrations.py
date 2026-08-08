"""Bounded, mask-aware Streamer behavior-cloning demonstrations.

The PPO rollout buffer is intentionally not reused for demonstrations.  A
viewer transition has a different behavior policy and must remain outside the
on-policy sample stream.  This module stores only successful, explicitly
eligible viewer decisions and persists them independently from SB3 model
checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence
import uuid
import zipfile

import numpy as np


DEMONSTRATION_FORMAT_VERSION = 1
MAX_DEMONSTRATION_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_DEMONSTRATION_RECORDS = 16_384
MAX_DEMONSTRATION_OBSERVATION_ELEMENTS = 1_000_000
MAX_DEMONSTRATION_ACTIONS = 100_000
MAX_DEMONSTRATION_METADATA_BYTES_PER_RECORD = 64 * 1024
MAX_DEMONSTRATION_METADATA_BYTES_TOTAL = 128 * 1024 * 1024
DEMONSTRATION_ARCHIVE_MEMBERS = frozenset(
    {
        "format_version.npy",
        "capacity.npy",
        "observation_shape.npy",
        "action_count.npy",
        "observations.npy",
        "action_masks.npy",
        "actions.npy",
        "metadata_json.npy",
        "total_added.npy",
        "total_evicted.npy",
        "persist_count.npy",
    }
)


class DemonstrationValidationError(ValueError):
    """Raised when a demonstration would violate the policy contract."""


def _npy_header(
    archive: zipfile.ZipFile,
    name: str,
) -> tuple[tuple[int, ...], np.dtype[Any], bool, int]:
    try:
        with archive.open(name, "r") as handle:
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
            elif version in {(2, 0), (3, 0)}:
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
            else:
                raise DemonstrationValidationError(f"unsupported NPY version in {name}: {version}")
            data_offset = int(handle.tell())
    except (KeyError, OSError, ValueError, EOFError) as exc:
        raise DemonstrationValidationError(f"invalid demonstration member {name}") from exc
    return tuple(int(value) for value in shape), np.dtype(dtype), bool(fortran), data_offset


def _validated_member_header(
    archive: zipfile.ZipFile,
    member: zipfile.ZipInfo,
) -> tuple[tuple[int, ...], np.dtype[Any]]:
    shape, dtype, fortran, data_offset = _npy_header(archive, member.filename)
    if fortran or dtype.hasobject or any(int(value) < 0 for value in shape):
        raise DemonstrationValidationError(
            f"demonstration member {member.filename} has an unsafe array header"
        )
    try:
        element_count = math.prod(shape) if shape else 1
        body_bytes = int(element_count) * int(dtype.itemsize)
    except (OverflowError, ValueError) as exc:
        raise DemonstrationValidationError(
            f"demonstration member {member.filename} has an unsafe array size"
        ) from exc
    if body_bytes < 0 or data_offset + body_bytes != int(member.file_size):
        raise DemonstrationValidationError(
            f"demonstration member {member.filename} body size does not match its header"
        )
    return shape, dtype


def _validate_npz_archive(
    source: Path,
    *,
    capacity: Optional[int],
    expected_observation_shape: Optional[Sequence[int]],
    expected_action_count: Optional[int],
) -> None:
    """Reject oversized/corrupt sidecars before NumPy allocates their arrays."""

    if source.stat().st_size > MAX_DEMONSTRATION_ARCHIVE_BYTES:
        raise DemonstrationValidationError("demonstration archive exceeds compressed-byte limit")
    try:
        with zipfile.ZipFile(source, "r") as archive:
            members = archive.infolist()
            member_names = {member.filename for member in members}
            if (
                len(members) != len(DEMONSTRATION_ARCHIVE_MEMBERS)
                or len(member_names) != len(members)
                or member_names != DEMONSTRATION_ARCHIVE_MEMBERS
            ):
                raise DemonstrationValidationError("demonstration archive member set is invalid")
            if any(member.flag_bits & 0x1 for member in members):
                raise DemonstrationValidationError("encrypted demonstration archives are not supported")
            if sum(int(member.file_size) for member in members) > MAX_DEMONSTRATION_ARCHIVE_BYTES:
                raise DemonstrationValidationError("demonstration archive exceeds uncompressed-byte limit")
            headers = {
                member.filename: _validated_member_header(archive, member)
                for member in members
            }
    except (OSError, zipfile.BadZipFile) as exc:
        raise DemonstrationValidationError("invalid demonstration archive") from exc

    int64_scalar_members = {
        "format_version.npy",
        "capacity.npy",
        "action_count.npy",
        "total_added.npy",
        "total_evicted.npy",
        "persist_count.npy",
    }
    for name in int64_scalar_members:
        if headers[name] != ((1,), np.dtype(np.int64)):
            raise DemonstrationValidationError(f"demonstration member {name} has an invalid schema")
    observation_shape_header, observation_shape_dtype = headers["observation_shape.npy"]
    if (
        len(observation_shape_header) != 1
        or observation_shape_header[0] <= 0
        or observation_shape_header[0] > 8
        or observation_shape_dtype != np.dtype(np.int64)
    ):
        raise DemonstrationValidationError("persisted observation-shape metadata is invalid")

    observation_shape, observation_dtype = headers["observations.npy"]
    mask_shape, mask_dtype = headers["action_masks.npy"]
    action_shape, action_dtype = headers["actions.npy"]
    metadata_shape, metadata_dtype = headers["metadata_json.npy"]

    record_count = observation_shape[0] if observation_shape else -1
    selected_limit = min(
        MAX_DEMONSTRATION_RECORDS,
        int(capacity) if capacity is not None else MAX_DEMONSTRATION_RECORDS,
    )
    if record_count < 0 or record_count > selected_limit:
        raise DemonstrationValidationError("persisted demonstration count exceeds configured capacity")
    if action_shape != (record_count,) or metadata_shape != (record_count,):
        raise DemonstrationValidationError("persisted demonstration record counts disagree")
    if len(mask_shape) != 2 or mask_shape[0] != record_count:
        raise DemonstrationValidationError("persisted demonstration mask shape is invalid")
    if len(observation_shape) < 2:
        raise DemonstrationValidationError("persisted demonstration observation shape is invalid")
    if any(value <= 0 for value in observation_shape[1:]) or math.prod(
        observation_shape[1:]
    ) > MAX_DEMONSTRATION_OBSERVATION_ELEMENTS:
        raise DemonstrationValidationError("persisted demonstration observation shape is unsafe")
    if expected_observation_shape is not None and observation_shape[1:] != tuple(
        int(value) for value in expected_observation_shape
    ):
        raise DemonstrationValidationError("persisted observation shape mismatch")
    if expected_action_count is not None and mask_shape[1] != int(expected_action_count):
        raise DemonstrationValidationError("persisted action count mismatch")
    if mask_shape[1] <= 0 or mask_shape[1] > MAX_DEMONSTRATION_ACTIONS:
        raise DemonstrationValidationError("persisted action count is unsafe")
    if observation_dtype != np.dtype(np.float32):
        raise DemonstrationValidationError("persisted observation dtype is invalid")
    if mask_dtype != np.dtype(bool) or action_dtype != np.dtype(np.int64):
        raise DemonstrationValidationError("persisted action arrays have invalid dtypes")
    if metadata_dtype.kind not in {"U", "S"}:
        raise DemonstrationValidationError("persisted demonstration metadata dtype is invalid")
    if metadata_dtype.itemsize > MAX_DEMONSTRATION_METADATA_BYTES_PER_RECORD:
        raise DemonstrationValidationError("persisted demonstration metadata record is too large")
    if record_count * metadata_dtype.itemsize > MAX_DEMONSTRATION_METADATA_BYTES_TOTAL:
        raise DemonstrationValidationError("persisted demonstration metadata is too large")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported demonstration metadata value: {type(value).__name__}")


def _metadata_copy(metadata: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """Return a JSON-safe deep copy and reject non-structured metadata."""

    payload = dict(metadata or {})
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=_json_default,
        )
    except (TypeError, ValueError) as exc:
        raise DemonstrationValidationError(f"demonstration metadata is not JSON-safe: {exc}") from exc
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise DemonstrationValidationError("demonstration metadata must encode to an object")
    return decoded


@dataclass(slots=True)
class DemonstrationRecord:
    """One supervised masked action decision."""

    observation: np.ndarray
    action_mask: np.ndarray
    action: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "DemonstrationRecord":
        return DemonstrationRecord(
            observation=np.asarray(self.observation, dtype=np.float32).copy(),
            action_mask=np.asarray(self.action_mask, dtype=bool).copy(),
            action=int(self.action),
            metadata=_metadata_copy(self.metadata),
        )


@dataclass(frozen=True, slots=True)
class DemonstrationBatch:
    """A dense NumPy batch ready for the existing MaskablePPO policy."""

    observations: np.ndarray
    action_masks: np.ndarray
    actions: np.ndarray
    metadata: tuple[dict[str, Any], ...]


class DemonstrationBuffer:
    """A bounded FIFO/ring of validated viewer demonstrations.

    Observations are the exact model-facing float32 vectors used at decision
    time.  Masks are copied before the environment step so the supervised
    target uses the same legality surface as the executed viewer action.
    """

    def __init__(
        self,
        capacity: int,
        *,
        observation_shape: Optional[Sequence[int]] = None,
        action_count: Optional[int] = None,
        persist_path: Optional[Path | str] = None,
    ) -> None:
        if int(capacity) <= 0:
            raise ValueError("demonstration capacity must be positive")
        self.capacity = int(capacity)
        self.observation_shape = (
            tuple(int(value) for value in observation_shape)
            if observation_shape is not None
            else None
        )
        if self.observation_shape is not None and any(value <= 0 for value in self.observation_shape):
            raise ValueError("observation_shape must contain only positive dimensions")
        self.action_count = int(action_count) if action_count is not None else None
        if self.action_count is not None and self.action_count <= 0:
            raise ValueError("action_count must be positive")
        self.persist_path = Path(persist_path) if persist_path is not None else None
        self._records: list[DemonstrationRecord] = []
        self.total_added = 0
        self.total_evicted = 0
        self.persist_count = 0
        self._dirty_additions = 0

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterable[DemonstrationRecord]:
        for record in self._records:
            yield record.copy()

    @property
    def dirty_additions(self) -> int:
        return int(self._dirty_additions)

    def _validate_arrays(
        self,
        observation: np.ndarray | Sequence[float],
        action_mask: np.ndarray | Sequence[bool],
        action: int,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        obs = np.asarray(observation, dtype=np.float32)
        mask = np.asarray(action_mask, dtype=bool)
        action_id = int(action)

        if obs.ndim == 0:
            raise DemonstrationValidationError("demonstration observation must have at least one dimension")
        if not np.all(np.isfinite(obs)):
            raise DemonstrationValidationError("demonstration observation contains a non-finite value")
        if mask.ndim != 1:
            raise DemonstrationValidationError(
                f"demonstration action mask must be one-dimensional, got shape={mask.shape}"
            )

        if self.observation_shape is None:
            self.observation_shape = tuple(int(value) for value in obs.shape)
        if tuple(obs.shape) != tuple(self.observation_shape):
            raise DemonstrationValidationError(
                "demonstration observation shape mismatch: "
                f"expected={self.observation_shape} actual={tuple(obs.shape)}"
            )

        if self.action_count is None:
            self.action_count = int(mask.shape[0])
        if int(mask.shape[0]) != int(self.action_count):
            raise DemonstrationValidationError(
                "demonstration action mask width mismatch: "
                f"expected={self.action_count} actual={mask.shape[0]}"
            )
        if action_id < 0 or action_id >= int(self.action_count):
            raise DemonstrationValidationError(
                f"demonstration action is out of range: action={action_id} action_count={self.action_count}"
            )
        if not bool(mask[action_id]):
            raise DemonstrationValidationError(
                f"demonstration action was masked at execution time: action={action_id}"
            )
        return obs.copy(), mask.copy(), action_id

    def add(
        self,
        observation: np.ndarray | Sequence[float],
        action_mask: np.ndarray | Sequence[bool],
        action: int,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> DemonstrationRecord:
        obs, mask, action_id = self._validate_arrays(observation, action_mask, action)
        record = DemonstrationRecord(
            observation=obs,
            action_mask=mask,
            action=action_id,
            metadata=_metadata_copy(metadata),
        )
        if len(self._records) >= self.capacity:
            self._records.pop(0)
            self.total_evicted += 1
        self._records.append(record)
        self.total_added += 1
        self._dirty_additions += 1
        return record.copy()

    def add_if_eligible(
        self,
        observation: np.ndarray | Sequence[float],
        action_mask: np.ndarray | Sequence[bool],
        transition: Mapping[str, Any],
    ) -> Optional[DemonstrationRecord]:
        """Add only an explicitly successful viewer-controlled transition.

        The caller's transition classifier is authoritative.  Truthy or
        inferred values are deliberately insufficient: the three booleans
        below must be literal ``True`` so rejected/ambiguous records cannot
        silently become positive labels.
        """

        if transition.get("viewer_controlled") is not True:
            return None
        if transition.get("demo_eligible") is not True:
            return None
        if transition.get("execution_succeeded") is not True:
            raise DemonstrationValidationError(
                "demo_eligible viewer transition must also declare execution_succeeded=true"
            )
        if "executed_action" not in transition:
            raise DemonstrationValidationError("eligible viewer transition is missing executed_action")

        metadata = _metadata_copy(transition.get("demonstration"))
        for key in (
            "schema_version",
            "behavior_source",
            "proposed_policy_action",
            "executed_action",
            "execution_status",
            "execution_succeeded",
        ):
            if key in transition:
                metadata.setdefault(key, transition[key])
        return self.add(
            observation,
            action_mask,
            int(transition["executed_action"]),
            metadata=metadata,
        )

    def sample(self, batch_size: int, rng: np.random.Generator) -> DemonstrationBatch:
        if len(self._records) == 0:
            raise ValueError("cannot sample from an empty demonstration buffer")
        if int(batch_size) <= 0:
            raise ValueError("demonstration batch_size must be positive")
        size = int(batch_size)
        replace = len(self._records) < size
        indices = np.asarray(rng.choice(len(self._records), size=size, replace=replace)).reshape(-1)
        selected = [self._records[int(index)] for index in indices]
        return DemonstrationBatch(
            observations=np.stack([record.observation for record in selected]).astype(np.float32, copy=False),
            action_masks=np.stack([record.action_mask for record in selected]).astype(bool, copy=False),
            actions=np.asarray([record.action for record in selected], dtype=np.int64),
            metadata=tuple(_metadata_copy(record.metadata) for record in selected),
        )

    def update_episode_outcome(
        self,
        episode_id: Any,
        outcome: str,
        *,
        training_cycle: Any = None,
        outcome_metadata: Optional[Mapping[str, Any]] = None,
    ) -> int:
        """Attach a later episode result to retained demonstrations."""

        target = str(episode_id)
        updates = 0
        extra = _metadata_copy(outcome_metadata)
        for record in self._records:
            if str(record.metadata.get("episode_id", "")) != target:
                continue
            if training_cycle is not None and str(
                record.metadata.get("training_cycle", "")
            ) != str(training_cycle):
                continue
            record.metadata["episode_outcome"] = str(outcome or "unknown")
            if extra:
                record.metadata["episode_outcome_metadata"] = extra
            updates += 1
        if updates:
            self._dirty_additions += 1
        return updates

    def records(self) -> tuple[DemonstrationRecord, ...]:
        return tuple(record.copy() for record in self._records)

    def save(self, path: Optional[Path | str] = None) -> Path:
        target = Path(path) if path is not None else self.persist_path
        if target is None:
            raise ValueError("no demonstration persistence path was configured")
        target.parent.mkdir(parents=True, exist_ok=True)

        obs_shape = tuple(self.observation_shape or ())
        action_count = int(self.action_count or 0)
        if self._records:
            observations = np.stack([record.observation for record in self._records]).astype(np.float32, copy=False)
            action_masks = np.stack([record.action_mask for record in self._records]).astype(bool, copy=False)
            actions = np.asarray([record.action for record in self._records], dtype=np.int64)
        else:
            observations = np.empty((0, *obs_shape), dtype=np.float32)
            action_masks = np.empty((0, action_count), dtype=bool)
            actions = np.empty((0,), dtype=np.int64)
        metadata_json = np.asarray(
            [
                json.dumps(
                    record.metadata,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    default=_json_default,
                )
                for record in self._records
            ],
            dtype=np.str_,
        )

        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    format_version=np.asarray([DEMONSTRATION_FORMAT_VERSION], dtype=np.int64),
                    capacity=np.asarray([self.capacity], dtype=np.int64),
                    observation_shape=np.asarray(obs_shape, dtype=np.int64),
                    action_count=np.asarray([action_count], dtype=np.int64),
                    observations=observations,
                    action_masks=action_masks,
                    actions=actions,
                    metadata_json=metadata_json,
                    total_added=np.asarray([self.total_added], dtype=np.int64),
                    total_evicted=np.asarray([self.total_evicted], dtype=np.int64),
                    persist_count=np.asarray([self.persist_count + 1], dtype=np.int64),
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
        self.persist_path = target
        self.persist_count += 1
        self._dirty_additions = 0
        return target

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        capacity: Optional[int] = None,
        expected_observation_shape: Optional[Sequence[int]] = None,
        expected_action_count: Optional[int] = None,
    ) -> "DemonstrationBuffer":
        source = Path(path)
        _validate_npz_archive(
            source,
            capacity=capacity,
            expected_observation_shape=expected_observation_shape,
            expected_action_count=expected_action_count,
        )
        with np.load(source, allow_pickle=False) as payload:
            version = int(np.asarray(payload["format_version"]).reshape(-1)[0])
            if version != DEMONSTRATION_FORMAT_VERSION:
                raise DemonstrationValidationError(
                    f"unsupported demonstration format_version={version}"
                )
            persisted_capacity = int(np.asarray(payload["capacity"]).reshape(-1)[0])
            selected_capacity = int(capacity) if capacity is not None else persisted_capacity
            obs_shape = tuple(int(value) for value in np.asarray(payload["observation_shape"]).tolist())
            action_count = int(np.asarray(payload["action_count"]).reshape(-1)[0])
            observations = np.asarray(payload["observations"], dtype=np.float32)
            masks = np.asarray(payload["action_masks"], dtype=bool)
            actions = np.asarray(payload["actions"], dtype=np.int64)
            metadata_raw = np.asarray(payload["metadata_json"], dtype=np.str_).tolist()
            total_added = int(np.asarray(payload["total_added"]).reshape(-1)[0])
            total_evicted = int(np.asarray(payload["total_evicted"]).reshape(-1)[0])
            persist_count = int(np.asarray(payload["persist_count"]).reshape(-1)[0])

        if not 1 <= persisted_capacity <= MAX_DEMONSTRATION_RECORDS:
            raise DemonstrationValidationError("persisted demonstration capacity is invalid")
        if not 1 <= selected_capacity <= MAX_DEMONSTRATION_RECORDS:
            raise DemonstrationValidationError("requested demonstration capacity is invalid")
        if len(obs_shape) == 0 or len(obs_shape) > 8 or any(value <= 0 for value in obs_shape):
            raise DemonstrationValidationError("persisted observation shape metadata is invalid")
        if tuple(observations.shape[1:]) != obs_shape:
            raise DemonstrationValidationError(
                "persisted observation shape metadata disagrees with observations"
            )
        if not 1 <= action_count <= MAX_DEMONSTRATION_ACTIONS:
            raise DemonstrationValidationError("persisted action count metadata is invalid")
        if masks.ndim != 2 or int(masks.shape[1]) != action_count:
            raise DemonstrationValidationError(
                "persisted action count metadata disagrees with masks"
            )
        if total_added < observations.shape[0] or total_evicted < 0 or persist_count < 0:
            raise DemonstrationValidationError("persisted demonstration counters are invalid")

        expected_shape = (
            tuple(int(value) for value in expected_observation_shape)
            if expected_observation_shape is not None
            else None
        )
        if expected_shape is not None and obs_shape != expected_shape:
            raise DemonstrationValidationError(
                f"persisted observation shape mismatch: expected={expected_shape} actual={obs_shape}"
            )
        if expected_action_count is not None and action_count != int(expected_action_count):
            raise DemonstrationValidationError(
                "persisted action count mismatch: "
                f"expected={int(expected_action_count)} actual={action_count}"
            )
        if not (
            observations.shape[0]
            == masks.shape[0]
            == actions.shape[0]
            == len(metadata_raw)
        ):
            raise DemonstrationValidationError("persisted demonstration arrays have inconsistent lengths")

        buffer = cls(
            selected_capacity,
            observation_shape=obs_shape or expected_shape,
            action_count=action_count or expected_action_count,
            persist_path=source,
        )
        records: list[DemonstrationRecord] = []
        for index in range(observations.shape[0]):
            try:
                metadata = json.loads(str(metadata_raw[index]))
            except json.JSONDecodeError as exc:
                raise DemonstrationValidationError(
                    f"invalid persisted demonstration metadata at index={index}: {exc}"
                ) from exc
            if not isinstance(metadata, dict):
                raise DemonstrationValidationError(
                    f"persisted demonstration metadata at index={index} is not an object"
                )
            obs, mask, action = buffer._validate_arrays(
                observations[index], masks[index], int(actions[index])
            )
            records.append(
                DemonstrationRecord(
                    observation=obs,
                    action_mask=mask,
                    action=action,
                    metadata=_metadata_copy(metadata),
                )
            )
        if len(records) > buffer.capacity:
            dropped = len(records) - buffer.capacity
            records = records[-buffer.capacity :]
            total_evicted += dropped
        buffer._records = records
        buffer.total_added = max(total_added, len(records))
        buffer.total_evicted = max(0, total_evicted)
        buffer.persist_count = max(0, persist_count)
        buffer._dirty_additions = 0
        return buffer


__all__ = [
    "DEMONSTRATION_FORMAT_VERSION",
    "DemonstrationBatch",
    "DemonstrationBuffer",
    "DemonstrationRecord",
    "DemonstrationValidationError",
]
