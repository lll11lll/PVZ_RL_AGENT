"""Bounded, read-only artifact discovery for the PvZRL Runs & Models GUI.

The index deliberately does not expose a "latest" or implicit-selection API.
Callers receive deterministic entries with explicit paths and must require a
user choice before using a checkpoint for resume, evaluation, or Streamer V1.

Only small JSON summaries are read.  Model archives are never opened or
hashed, and directory traversal, JSON work, retained results, and issue output
are all bounded so this module is safe to call from a background GUI worker.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from pvzrl_gui_config import FULL_ADVENTURE_CONTRACT, inspect_model_compatibility


ROLE_BASELINE = "BASELINE"
ROLE_CURRENT = "CURRENT"
ROLE_BEST = "BEST"
ROLE_CHECKPOINT = "CHECKPOINT"
ROLE_MODEL = "MODEL"

ROLE_ORDER = (
    ROLE_CURRENT,
    ROLE_BEST,
    ROLE_BASELINE,
    ROLE_CHECKPOINT,
    ROLE_MODEL,
)

METADATA_VALID = "VALID"
METADATA_MISSING = "MISSING"
METADATA_MALFORMED = "MALFORMED"
METADATA_UNREADABLE = "UNREADABLE"
METADATA_TOO_LARGE = "TOO_LARGE"
METADATA_LIMIT_REACHED = "LIMIT_REACHED"

_CHECKPOINT_STEPS_RE = re.compile(r"(?:^|[_-])(\d+)_steps(?:\.zip)$", re.IGNORECASE)
_RUN_MARKERS = frozenset(
    {
        "resolved_config.json",
        "summary.json",
        "adventure_training_progress.json",
        "adventure_progression_results.json",
        "streamer_state.json",
        "streamer_cycles.jsonl",
        "performance_summary.json",
    }
)
_EVALUATION_FILES = frozenset(
    {
        "evaluation.json",
        "adventure_progression_results.json",
        "streamer_state.json",
    }
)
_PROGRESSION_FILES = frozenset(
    {
        "adventure_training_progress.json",
        "adventure_progression_results.json",
        "curriculum_state.json",
        "streamer_state.json",
    }
)
_RECORD_FILES = frozenset({"baseline.json", "streamer_checkpoint.json"})


@dataclass(frozen=True, slots=True)
class ArtifactScanLimits:
    """Hard limits for one artifact-index refresh."""

    max_depth: int = 12
    max_entries: int = 20_000
    max_models: int = 500
    max_runs: int = 500
    max_json_reads: int = 2_000
    max_json_bytes: int = 1_048_576
    max_summaries_per_run: int = 32
    max_issues: int = 200

    def validate(self) -> None:
        for name in (
            "max_entries",
            "max_models",
            "max_runs",
            "max_json_reads",
            "max_json_bytes",
            "max_summaries_per_run",
            "max_issues",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.max_depth) < 0:
            raise ValueError("max_depth must be non-negative")


@dataclass(frozen=True, slots=True)
class ArtifactIssue:
    code: str
    path: str
    message: str


@dataclass(frozen=True, slots=True)
class ModelCompatibilitySummary:
    """Exact metadata-only compatibility with the maintained v2 contract."""

    compatible: bool
    blocked_reason: str
    details: str
    metadata_status: str
    metadata_path: str
    metadata_error: str
    metadata_version: Optional[int]
    model_family: str
    action_space_mode: str
    action_count: Optional[int]
    action_decoder_version: str
    observation_version: str
    observation_shape: Tuple[int, ...]
    max_seed_slots: Optional[int]
    dynamic_seed_slots: Optional[bool]
    identity_seed_slots: Optional[bool]
    decoder_wait_action: Optional[int]
    placement_action_range: Tuple[int, ...]
    rows: Optional[int]
    cols: Optional[int]
    cells_per_seed_slot: Optional[int]


@dataclass(frozen=True, slots=True)
class EvaluationArtifactSummary:
    path: str
    scope: str
    status: str
    win_rate: Optional[float]
    avg_reward: Optional[float]
    episodes: Optional[int]
    adventure_start_level: Optional[int]
    next_adventure_level: Optional[int]
    streamer_cycle: Optional[int]
    error: str = ""


@dataclass(frozen=True, slots=True)
class ProgressionArtifactSummary:
    path: str
    status: str
    current_level: Optional[int]
    frontier_level: Optional[int]
    next_adventure_level: Optional[int]
    latest_episode: Optional[int]
    latest_attempt: Optional[int]
    cleared_levels: Tuple[int, ...]
    blocked_reason: str
    error: str = ""


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    path: str
    relative_path: str
    name: str
    run_dir: str
    role: str
    roles: Tuple[str, ...]
    size_bytes: Optional[int]
    modified_at: str
    artifact_timestamp: str
    timesteps: Optional[int]
    timesteps_source: str
    adventure_level: Optional[int]
    adventure_level_source: str
    compatibility: ModelCompatibilitySummary
    evaluations: Tuple[EvaluationArtifactSummary, ...] = field(default_factory=tuple)
    progressions: Tuple[ProgressionArtifactSummary, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class RunArtifact:
    path: str
    relative_path: str
    modified_at: str
    status: str
    run_mode: str
    model_family: str
    model_paths: Tuple[str, ...]
    evaluations: Tuple[EvaluationArtifactSummary, ...] = field(default_factory=tuple)
    progressions: Tuple[ProgressionArtifactSummary, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class ArtifactIndex:
    root: str
    models: Tuple[ModelArtifact, ...]
    runs: Tuple[RunArtifact, ...]
    issues: Tuple[ArtifactIssue, ...]
    truncated: bool
    entries_examined: int
    directories_examined: int
    json_files_read: int
    dropped_issue_count: int = 0


@dataclass(frozen=True, slots=True)
class _JsonResult:
    value: Optional[Dict[str, Any]]
    status: str
    error: str = ""


@dataclass(frozen=True, slots=True)
class _FileInfo:
    path: Path
    size_bytes: Optional[int]
    modified_ns: int
    modified_at: str


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if result == result and result not in {float("inf"), float("-inf")} else None


def _optional_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    if value in (0, 1):
        return bool(value)
    return None


def _int_tuple(value: Any) -> Tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    result = []
    for item in value:
        converted = _optional_int(item)
        if converted is None:
            return ()
        result.append(converted)
    return tuple(result)


def _utc_timestamp(timestamp: float) -> str:
    try:
        return datetime.fromtimestamp(timestamp, timezone.utc).isoformat().replace("+00:00", "Z")
    except (OSError, OverflowError, ValueError):
        return ""


def _absolute(path: Path) -> str:
    # Traversal never follows symlinks, so lexical normalization is sufficient
    # here and avoids an expensive filesystem resolution for every comparison.
    return os.path.abspath(os.fspath(path))


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.normpath(_absolute(path)))


def _relative(path: Path, root: Path) -> str:
    try:
        return Path(_absolute(path)).relative_to(Path(_absolute(root))).as_posix()
    except ValueError:
        return _absolute(path)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path_key = _path_key(path)
        root_key = _path_key(root)
        return os.path.commonpath((path_key, root_key)) == root_key
    except ValueError:
        return False


class _ScanState:
    def __init__(self, root: Path, limits: ArtifactScanLimits) -> None:
        self.root = root
        self.limits = limits
        self.issues: list[ArtifactIssue] = []
        self.dropped_issue_count = 0
        self.truncated = False
        self.entries_examined = 0
        self.directories_examined = 0
        self.json_files_read = 0
        self.json_cache: Dict[str, _JsonResult] = {}

    def issue(self, code: str, path: Path, message: str) -> None:
        if len(self.issues) < self.limits.max_issues:
            self.issues.append(ArtifactIssue(code=code, path=_absolute(path), message=str(message)[:1000]))
        else:
            self.dropped_issue_count += 1

    def read_json(self, path: Path) -> _JsonResult:
        key = _path_key(path)
        cached = self.json_cache.get(key)
        if cached is not None:
            return cached
        if self.json_files_read >= self.limits.max_json_reads:
            self.truncated = True
            result = _JsonResult(None, METADATA_LIMIT_REACHED, "JSON read limit reached")
            self.json_cache[key] = result
            self.issue("json_read_limit", path, result.error)
            return result
        self.json_files_read += 1
        try:
            size = path.stat().st_size
            if size > self.limits.max_json_bytes:
                result = _JsonResult(
                    None,
                    METADATA_TOO_LARGE,
                    f"JSON file is {size} bytes; limit is {self.limits.max_json_bytes}",
                )
            else:
                with path.open("rb") as handle:
                    raw = handle.read(self.limits.max_json_bytes + 1)
                if len(raw) > self.limits.max_json_bytes:
                    result = _JsonResult(None, METADATA_TOO_LARGE, "JSON grew beyond the read limit")
                else:
                    try:
                        decoded = raw.decode("utf-8")
                        value = json.loads(decoded)
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        result = _JsonResult(None, METADATA_MALFORMED, f"invalid_json:{exc}")
                    else:
                        result = (
                            _JsonResult(dict(value), METADATA_VALID)
                            if isinstance(value, dict)
                            else _JsonResult(None, METADATA_MALFORMED, "invalid_json:not_object")
                        )
        except OSError as exc:
            result = _JsonResult(None, METADATA_UNREADABLE, f"read_failed:{exc}")
        self.json_cache[key] = result
        if result.value is None:
            self.issue(f"json_{result.status.lower()}", path, result.error)
        return result


def _walk_bounded(state: _ScanState) -> tuple[list[Path], Dict[str, set[str]], Dict[str, _FileInfo]]:
    root = state.root
    directories: list[Path] = []
    names_by_directory: Dict[str, set[str]] = {}
    files: Dict[str, _FileInfo] = {}
    queue: list[tuple[Path, int]] = [(root, 0)]
    queue_index = 0

    while queue_index < len(queue):
        directory, depth = queue[queue_index]
        queue_index += 1
        directories.append(directory)
        state.directories_examined += 1
        directory_key = _path_key(directory)
        names = names_by_directory.setdefault(directory_key, set())
        try:
            iterator = os.scandir(directory)
        except OSError as exc:
            state.issue("directory_unreadable", directory, str(exc))
            continue
        children: list[tuple[str, Path, bool, bool]] = []
        with iterator:
            for entry in iterator:
                if state.entries_examined >= state.limits.max_entries:
                    state.truncated = True
                    state.issue("entry_limit", directory, "Filesystem entry limit reached")
                    break
                state.entries_examined += 1
                path = Path(entry.path)
                names.add(entry.name.lower())
                try:
                    is_directory = entry.is_dir(follow_symlinks=False)
                    is_file = entry.is_file(follow_symlinks=False)
                except OSError as exc:
                    state.issue("entry_unreadable", path, str(exc))
                    continue
                children.append((entry.name.casefold(), path, is_directory, is_file))
        children.sort(key=lambda item: (item[0], item[1].name))
        for _name, path, is_directory, is_file in children:
            if is_directory:
                if depth >= state.limits.max_depth:
                    state.truncated = True
                    state.issue("depth_limit", path, f"Directory depth limit {state.limits.max_depth} reached")
                    continue
                queue.append((path, depth + 1))
            elif is_file:
                try:
                    stat = path.stat()
                    info = _FileInfo(
                        path=path,
                        size_bytes=int(stat.st_size),
                        modified_ns=int(stat.st_mtime_ns),
                        modified_at=_utc_timestamp(float(stat.st_mtime)),
                    )
                except OSError as exc:
                    state.issue("file_stat_failed", path, str(exc))
                    info = _FileInfo(path=path, size_bytes=None, modified_ns=0, modified_at="")
                files[_path_key(path)] = info
        if state.entries_examined >= state.limits.max_entries:
            break
    return directories, names_by_directory, files


def _under_checkpoint_directory(path: Path, root: Path) -> bool:
    try:
        parts = tuple(
            part.lower()
            for part in Path(_absolute(path)).relative_to(Path(_absolute(root))).parts
        )
    except ValueError:
        parts = tuple(part.lower() for part in path.parts)
    return "checkpoints" in parts


def _looks_like_model(path: Path, file_names: Mapping[str, set[str]], referenced: set[str], root: Path) -> bool:
    if path.suffix.lower() != ".zip":
        return False
    if _path_key(path) in referenced:
        return True
    name = path.name.lower()
    if name in {"model.zip", "final_model.zip"} or _CHECKPOINT_STEPS_RE.search(name):
        return True
    if _under_checkpoint_directory(path, root):
        return True
    return "model_metadata.json" in file_names.get(_path_key(path.parent), set())


def _record_path_candidates(text: str, record_path: Path, experiment_dir: Path, root: Path) -> Iterable[Path]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ()
    candidate = Path(cleaned)
    if candidate.is_absolute():
        return (candidate,)
    ordered = (
        root.parent / candidate,
        experiment_dir / candidate,
        record_path.parent / candidate,
        candidate.absolute(),
    )
    result: list[Path] = []
    seen: set[str] = set()
    for value in ordered:
        key = _path_key(value)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return tuple(result)


def _experiment_dir_for_record(record_path: Path) -> Path:
    if record_path.name.lower() == "baseline.json" and record_path.parent.name.lower() == "checkpoints":
        return record_path.parent.parent
    if record_path.name.lower() == "streamer_checkpoint.json":
        parent = record_path.parent
        if parent.name.lower() in {"current", "best"} and parent.parent.name.lower() == "checkpoints":
            return parent.parent.parent
    return record_path.parent


def _load_role_records(
    state: _ScanState,
    record_paths: Sequence[Path],
) -> tuple[Dict[str, set[str]], Dict[str, list[tuple[str, Path, Dict[str, Any]]]]]:
    roles: Dict[str, set[str]] = {}
    payloads: Dict[str, list[tuple[str, Path, Dict[str, Any]]]] = {}
    for record_path in sorted(record_paths, key=lambda path: _relative(path, state.root).casefold()):
        result = state.read_json(record_path)
        if result.value is None:
            continue
        payload = result.value
        filename = record_path.name.lower()
        declared_role = str(payload.get("role") or "").strip().upper()
        if filename == "baseline.json":
            role = ROLE_BASELINE if declared_role in {"", ROLE_BASELINE} else ""
            path_fields = ("model_path",)
        else:
            role = declared_role if declared_role in {ROLE_CURRENT, ROLE_BEST} else ""
            path_fields = ("model_path", "compatibility_alias")
        if not role:
            state.issue("streamer_role_record_invalid", record_path, f"Unexpected role {declared_role!r}")
            continue
        experiment_dir = _experiment_dir_for_record(record_path)
        matched = False
        for field_name in path_fields:
            for candidate in _record_path_candidates(
                str(payload.get(field_name) or ""), record_path, experiment_dir, state.root
            ):
                key = _path_key(candidate)
                roles.setdefault(key, set()).add(role)
                payloads.setdefault(key, []).append((role, record_path, payload))
                matched = True
        if not matched:
            state.issue("streamer_role_record_path_missing", record_path, "Role record has no model path")
    return roles, payloads


def _path_roles(path: Path, root: Path, record_roles: Mapping[str, set[str]]) -> Tuple[str, ...]:
    roles = set(record_roles.get(_path_key(path), set()))
    try:
        parts = tuple(
            part.lower()
            for part in Path(_absolute(path)).relative_to(Path(_absolute(root))).parts
        )
    except ValueError:
        parts = tuple(part.lower() for part in path.parts)
    if "checkpoints" in parts:
        index = parts.index("checkpoints")
        child = parts[index + 1] if index + 1 < len(parts) else ""
        if child == "current":
            roles.add(ROLE_CURRENT)
        elif child == "best":
            roles.add(ROLE_BEST)
        else:
            roles.add(ROLE_CHECKPOINT)
    if not roles:
        roles.add(ROLE_MODEL)
    return tuple(role for role in ROLE_ORDER if role in roles)


def _is_run_directory(path: Path, root: Path, names: set[str]) -> bool:
    if names.intersection(_RUN_MARKERS):
        return True
    if "model_metadata.json" not in names:
        return False
    if not names.intersection({"model.zip", "final_model.zip"}):
        return False
    return not _under_checkpoint_directory(path, root)


def _nearest_run(path: Path, root: Path, run_paths: Mapping[str, Path]) -> Optional[Path]:
    current = path
    while _is_within(current, root):
        candidate = run_paths.get(_path_key(current))
        if candidate is not None:
            return candidate
        if current == root or current.parent == current:
            break
        current = current.parent
    return None


def _metadata_candidates(model_path: Path, run_dir: Optional[Path]) -> Tuple[Path, ...]:
    candidates = [model_path.parent / "model_metadata.json"]
    if model_path.parent.name.lower() == "checkpoints":
        candidates.append(model_path.parent.parent / "model_metadata.json")
    if run_dir is not None:
        candidates.append(run_dir / "model_metadata.json")
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = _path_key(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return tuple(unique)


def _compatibility_mapping(report: Any) -> Dict[str, Any]:
    if isinstance(report, Mapping):
        return dict(report)
    to_dict = getattr(report, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        return dict(value) if isinstance(value, Mapping) else {}
    result: Dict[str, Any] = {}
    for name in (
        "compatible",
        "ok",
        "blocked_reason",
        "details",
        "metadata_path",
        "metadata_inferred",
        "model_metadata",
        "actual",
        "expected",
        "env_metadata",
    ):
        if hasattr(report, name):
            result[name] = getattr(report, name)
    return result


def _build_compatibility(
    state: _ScanState,
    model_path: Path,
    run_dir: Optional[Path],
) -> tuple[ModelCompatibilitySummary, Dict[str, Any]]:
    metadata_path: Optional[Path] = None
    metadata_result = _JsonResult(None, METADATA_MISSING, "Canonical model_metadata.json is missing")
    for candidate in _metadata_candidates(model_path, run_dir):
        try:
            exists = candidate.is_file()
        except OSError:
            exists = False
        if not exists:
            continue
        metadata_path = candidate
        metadata_result = state.read_json(candidate)
        break
    metadata = dict(metadata_result.value or {})
    if metadata_path is None:
        metadata_status = METADATA_MISSING
        metadata_error = metadata_result.error
        blocked_reason = "missing_model_metadata"
        details = f"No canonical model_metadata.json found for {model_path}."
        compatible = False
    elif metadata_result.value is None:
        metadata_status = metadata_result.status
        metadata_error = metadata_result.error
        blocked_reason = {
            METADATA_MALFORMED: "invalid_model_metadata",
            METADATA_UNREADABLE: "unreadable_model_metadata",
            METADATA_TOO_LARGE: "model_metadata_too_large",
            METADATA_LIMIT_REACHED: "artifact_scan_json_limit",
        }.get(metadata_status, "invalid_model_metadata")
        details = metadata_error
        compatible = False
    else:
        metadata_status = METADATA_VALID
        metadata_error = ""
        try:
            report = inspect_model_compatibility(
                model_path,
                contract=FULL_ADVENTURE_CONTRACT,
            )
            report_mapping = _compatibility_mapping(report)
            compatible = bool(report_mapping.get("compatible", report_mapping.get("ok", False)))
            blocked_reason = str(report_mapping.get("blocked_reason") or "")
            details = str(report_mapping.get("details") or "")
            reported_path = str(report_mapping.get("metadata_path") or "").strip()
            if reported_path:
                metadata_path = Path(reported_path)
        except Exception as exc:  # Defensive GUI boundary; scanner must keep returning other artifacts.
            compatible = False
            blocked_reason = "model_compatibility_inspection_failed"
            details = f"{type(exc).__name__}: {exc}"[:1000]
            state.issue("model_compatibility_inspection_failed", model_path, details)

    summary = ModelCompatibilitySummary(
        compatible=compatible,
        blocked_reason=blocked_reason,
        details=details,
        metadata_status=metadata_status,
        metadata_path=_absolute(metadata_path) if metadata_path is not None else "",
        metadata_error=metadata_error,
        metadata_version=_optional_int(metadata.get("metadata_version")),
        model_family=str(metadata.get("model_family") or ""),
        action_space_mode=str(metadata.get("action_space_mode") or ""),
        action_count=_optional_int(metadata.get("action_count")),
        action_decoder_version=str(metadata.get("action_decoder_version") or ""),
        observation_version=str(metadata.get("observation_version") or ""),
        observation_shape=_int_tuple(metadata.get("observation_shape")),
        max_seed_slots=_optional_int(metadata.get("max_seed_slots")),
        dynamic_seed_slots=_optional_bool(metadata.get("dynamic_seed_slots")),
        identity_seed_slots=_optional_bool(metadata.get("identity_seed_slots")),
        decoder_wait_action=_optional_int(metadata.get("decoder_wait_action")),
        placement_action_range=_int_tuple(metadata.get("placement_action_range")),
        rows=_optional_int(metadata.get("rows")),
        cols=_optional_int(metadata.get("cols")),
        cells_per_seed_slot=_optional_int(metadata.get("cells_per_seed_slot")),
    )
    return summary, metadata


def _evaluation_summary(
    path: Path,
    scope: str,
    payload: Optional[Mapping[str, Any]],
    error: str = "",
) -> EvaluationArtifactSummary:
    data = dict(payload or {})
    nested = data.get("summary")
    summary = dict(nested) if isinstance(nested, Mapping) else data
    episodes = _optional_int(
        summary.get(
            "episodes_completed",
            summary.get("episodes", data.get("evaluation_episodes_completed", data.get("levels_completed"))),
        )
    )
    return EvaluationArtifactSummary(
        path=_absolute(path),
        scope=scope,
        status=str(data.get("status") or ("unreadable" if error else "")),
        win_rate=_optional_float(summary.get("win_rate")),
        avg_reward=_optional_float(summary.get("avg_reward", summary.get("mean_reward"))),
        episodes=episodes,
        adventure_start_level=_optional_int(
            data.get("adventure_start_level", summary.get("adventure_start_level"))
        ),
        next_adventure_level=_optional_int(
            data.get("next_adventure_level", summary.get("next_adventure_level"))
        ),
        streamer_cycle=_optional_int(data.get("streamer_cycle", data.get("training_cycle"))),
        error=error,
    )


def _evaluation_summaries_from_json(
    path: Path,
    result: _JsonResult,
) -> Tuple[EvaluationArtifactSummary, ...]:
    if result.value is None:
        return (_evaluation_summary(path, path.stem, None, result.error),)
    payload = result.value
    summaries: list[EvaluationArtifactSummary] = []
    has_direct_metrics = isinstance(payload.get("summary"), Mapping) or any(
        key in payload for key in ("win_rate", "avg_reward", "mean_reward")
    )
    if has_direct_metrics:
        summaries.append(_evaluation_summary(path, path.stem, payload))
    for key in ("baseline_evaluation", "current_evaluation", "best_evaluation"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            summaries.append(_evaluation_summary(path, key.removesuffix("_evaluation"), value))
    return tuple(summaries)


def _progression_summary(path: Path, result: _JsonResult) -> ProgressionArtifactSummary:
    if result.value is None:
        return ProgressionArtifactSummary(
            path=_absolute(path),
            status="unreadable",
            current_level=None,
            frontier_level=None,
            next_adventure_level=None,
            latest_episode=None,
            latest_attempt=None,
            cleared_levels=(),
            blocked_reason="",
            error=result.error,
        )
    payload = result.value
    latest = payload.get("latest") if isinstance(payload.get("latest"), Mapping) else {}
    levels = payload.get("levels") if isinstance(payload.get("levels"), list) else []
    last_level: Mapping[str, Any] = levels[-1] if levels and isinstance(levels[-1], Mapping) else {}
    cleared = _int_tuple(payload.get("cleared_levels"))
    return ProgressionArtifactSummary(
        path=_absolute(path),
        status=str(payload.get("status") or ""),
        current_level=_optional_int(
            payload.get(
                "current_level",
                payload.get("frontier_level", latest.get("level", last_level.get("level"))),
            )
        ),
        frontier_level=_optional_int(payload.get("frontier_level")),
        next_adventure_level=_optional_int(payload.get("next_adventure_level")),
        latest_episode=_optional_int(latest.get("episode", payload.get("episode"))),
        latest_attempt=_optional_int(latest.get("attempt", payload.get("attempt"))),
        cleared_levels=cleared,
        blocked_reason=str(
            payload.get("blocked_reason")
            or payload.get("post_win_blocked_reason")
            or payload.get("frontier_replay_blocked_reason")
            or ""
        ),
    )


def _summary_files_for_run(
    run_dir: Path,
    candidate_paths: Sequence[Path],
    limit: int,
) -> Tuple[Path, ...]:
    paths = [path for path in candidate_paths if _is_within(path, run_dir)]
    paths.sort(key=lambda path: (_relative(path, run_dir).casefold(), _relative(path, run_dir)))
    return tuple(paths[:limit])


def _run_payload(
    state: _ScanState,
    run_dir: Path,
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for filename in (
        "resolved_config.json",
        "model_metadata.json",
        "adventure_training_progress.json",
        "streamer_state.json",
        "summary.json",
    ):
        path = run_dir / filename
        if not path.is_file():
            continue
        result = state.read_json(path)
        if result.value is not None:
            for key, value in result.value.items():
                merged.setdefault(key, value)
    return merged


def _record_for_model(
    model_path: Path,
    record_payloads: Mapping[str, list[tuple[str, Path, Dict[str, Any]]]],
    roles: Sequence[str],
) -> Optional[tuple[str, Path, Dict[str, Any]]]:
    candidates = record_payloads.get(_path_key(model_path), [])
    order = {role: index for index, role in enumerate(roles)}
    ranked = sorted(candidates, key=lambda item: (order.get(item[0], len(order)), _absolute(item[1])))
    return ranked[0] if ranked else None


def _model_timesteps(
    model_path: Path,
    metadata: Mapping[str, Any],
    run_payload: Mapping[str, Any],
    record: Optional[tuple[str, Path, Dict[str, Any]]],
) -> tuple[Optional[int], str]:
    if record is not None:
        value = _optional_int(record[2].get("model_steps"))
        if value is not None:
            return value, _absolute(record[1]) + ":model_steps"
    match = _CHECKPOINT_STEPS_RE.search(model_path.name)
    if match:
        return int(match.group(1)), "checkpoint_filename"
    for source_name, payload in (("summary", run_payload), ("metadata", metadata)):
        for key in ("model_steps", "num_timesteps", "total_timesteps"):
            value = _optional_int(payload.get(key))
            if value is not None:
                return value, f"{source_name}:{key}"
    return None, ""


def _model_level(
    progressions: Sequence[ProgressionArtifactSummary],
    record: Optional[tuple[str, Path, Dict[str, Any]]],
) -> tuple[Optional[int], str]:
    if record is not None:
        payload = record[2]
        training = payload.get("training_metrics")
        if isinstance(training, Mapping):
            value = _optional_int(training.get("next_adventure_level"))
            if value is not None:
                return value, _absolute(record[1]) + ":training_metrics.next_adventure_level"
        evaluation = payload.get("evaluation")
        if isinstance(evaluation, Mapping):
            for key in ("next_adventure_level", "adventure_start_level"):
                value = _optional_int(evaluation.get(key))
                if value is not None:
                    return value, _absolute(record[1]) + f":evaluation.{key}"
    for summary in progressions:
        for key, value in (
            ("next_adventure_level", summary.next_adventure_level),
            ("current_level", summary.current_level),
            ("frontier_level", summary.frontier_level),
        ):
            if value is not None:
                return value, summary.path + f":{key}"
    return None, ""


def _artifact_timestamp(
    metadata: Mapping[str, Any],
    record: Optional[tuple[str, Path, Dict[str, Any]]],
    modified_at: str,
) -> str:
    if record is not None:
        for key in ("saved_at", "captured_at"):
            value = str(record[2].get(key) or "").strip()
            if value:
                return value
    value = str(metadata.get("created_at") or "").strip()
    return value or modified_at


def scan_run_artifacts(
    runs_root: Path | str,
    *,
    limits: Optional[ArtifactScanLimits] = None,
) -> ArtifactIndex:
    """Return a deterministic, bounded index rooted at ``runs_root``.

    The function performs no writes and never selects a model.  Compatibility
    is metadata-only; an actual ``MaskablePPO.load`` remains the runtime proof
    before a checkpoint is executed.
    """

    selected_limits = limits or ArtifactScanLimits()
    selected_limits.validate()
    root = Path(runs_root)
    try:
        root = root.resolve(strict=False)
    except OSError:
        root = root.absolute()
    state = _ScanState(root, selected_limits)
    if not root.exists():
        state.issue("root_missing", root, "Runs root does not exist")
        return ArtifactIndex(
            root=_absolute(root),
            models=(),
            runs=(),
            issues=tuple(state.issues),
            truncated=False,
            entries_examined=0,
            directories_examined=0,
            json_files_read=0,
        )
    if not root.is_dir():
        state.issue("root_not_directory", root, "Runs root is not a directory")
        return ArtifactIndex(
            root=_absolute(root),
            models=(),
            runs=(),
            issues=tuple(state.issues),
            truncated=False,
            entries_examined=0,
            directories_examined=0,
            json_files_read=0,
        )

    directories, names_by_directory, files = _walk_bounded(state)
    file_infos = tuple(files.values())
    record_paths = tuple(
        info.path for info in file_infos if info.path.name.lower() in _RECORD_FILES
    )
    record_roles, record_payloads = _load_role_records(state, record_paths)
    referenced_paths = set(record_roles)
    model_infos = [
        info
        for info in file_infos
        if _looks_like_model(info.path, names_by_directory, referenced_paths, root)
    ]
    model_infos.sort(
        key=lambda info: (-info.modified_ns, _relative(info.path, root).casefold(), _relative(info.path, root))
    )
    if len(model_infos) > selected_limits.max_models:
        state.truncated = True
        state.issue(
            "model_result_limit",
            root,
            f"Found {len(model_infos)} models; retaining {selected_limits.max_models}",
        )
        model_infos = model_infos[: selected_limits.max_models]

    run_candidates = [
        directory
        for directory in directories
        if _is_run_directory(
            directory,
            root,
            names_by_directory.get(_path_key(directory), set()),
        )
    ]
    run_candidates.sort(key=lambda path: (_relative(path, root).casefold(), _relative(path, root)))
    if len(run_candidates) > selected_limits.max_runs:
        state.truncated = True
        state.issue(
            "run_result_limit",
            root,
            f"Found {len(run_candidates)} runs; retaining {selected_limits.max_runs}",
        )
        run_candidates = run_candidates[: selected_limits.max_runs]
    run_paths = {_path_key(path): path for path in run_candidates}

    evaluation_paths = tuple(
        info.path for info in file_infos if info.path.name.lower() in _EVALUATION_FILES
    )
    progression_paths = tuple(
        info.path for info in file_infos if info.path.name.lower() in _PROGRESSION_FILES
    )
    run_evaluations: Dict[str, Tuple[EvaluationArtifactSummary, ...]] = {}
    run_progressions: Dict[str, Tuple[ProgressionArtifactSummary, ...]] = {}
    run_payloads: Dict[str, Dict[str, Any]] = {}
    for run_dir in run_candidates:
        run_key = _path_key(run_dir)
        evaluation_summaries: list[EvaluationArtifactSummary] = []
        for path in _summary_files_for_run(
            run_dir,
            evaluation_paths,
            selected_limits.max_summaries_per_run,
        ):
            evaluation_summaries.extend(_evaluation_summaries_from_json(path, state.read_json(path)))
            if len(evaluation_summaries) >= selected_limits.max_summaries_per_run:
                break
        run_evaluations[run_key] = tuple(
            evaluation_summaries[: selected_limits.max_summaries_per_run]
        )
        run_progressions[run_key] = tuple(
            _progression_summary(path, state.read_json(path))
            for path in _summary_files_for_run(
                run_dir,
                progression_paths,
                selected_limits.max_summaries_per_run,
            )
        )
        run_payloads[run_key] = _run_payload(state, run_dir)

    model_artifacts: list[ModelArtifact] = []
    models_by_run: Dict[str, list[str]] = {}
    for info in model_infos:
        run_dir = _nearest_run(info.path.parent, root, run_paths)
        run_key = _path_key(run_dir) if run_dir is not None else ""
        roles = _path_roles(info.path, root, record_roles)
        record = _record_for_model(info.path, record_payloads, roles)
        compatibility, metadata = _build_compatibility(state, info.path, run_dir)
        progressions = run_progressions.get(run_key, ())
        evaluations = list(run_evaluations.get(run_key, ()))
        if record is not None and isinstance(record[2].get("evaluation"), Mapping):
            record_summary = _evaluation_summary(
                record[1], record[0].lower(), record[2]["evaluation"]
            )
            if record_summary not in evaluations:
                evaluations.insert(0, record_summary)
        timesteps, timesteps_source = _model_timesteps(
            info.path,
            metadata,
            run_payloads.get(run_key, {}),
            record,
        )
        adventure_level, level_source = _model_level(progressions, record)
        artifact = ModelArtifact(
            path=_absolute(info.path),
            relative_path=_relative(info.path, root),
            name=info.path.name,
            run_dir=_absolute(run_dir) if run_dir is not None else "",
            role=roles[0],
            roles=roles,
            size_bytes=info.size_bytes,
            modified_at=info.modified_at,
            artifact_timestamp=_artifact_timestamp(metadata, record, info.modified_at),
            timesteps=timesteps,
            timesteps_source=timesteps_source,
            adventure_level=adventure_level,
            adventure_level_source=level_source,
            compatibility=compatibility,
            evaluations=tuple(evaluations[: selected_limits.max_summaries_per_run]),
            progressions=tuple(progressions[: selected_limits.max_summaries_per_run]),
        )
        model_artifacts.append(artifact)
        if run_key:
            models_by_run.setdefault(run_key, []).append(artifact.path)

    run_artifacts: list[tuple[int, RunArtifact]] = []
    for run_dir in run_candidates:
        run_key = _path_key(run_dir)
        payload = run_payloads.get(run_key, {})
        # A directory's timestamp is adequate for index ordering.  Walking all
        # scanned files once per run would make work bounded only by the product
        # of both limits and is unnecessarily slow on large Windows run trees.
        try:
            run_stat = run_dir.stat()
            modified_ns = int(run_stat.st_mtime_ns)
            modified_at = _utc_timestamp(float(run_stat.st_mtime))
        except OSError as exc:
            state.issue("run_stat_failed", run_dir, str(exc))
            modified_ns = 0
            modified_at = ""
        run_artifacts.append(
            (
                modified_ns,
                RunArtifact(
                    path=_absolute(run_dir),
                    relative_path=_relative(run_dir, root),
                    modified_at=modified_at,
                    status=str(payload.get("status") or ""),
                    run_mode=str(payload.get("run_mode") or payload.get("mode") or ""),
                    model_family=str(payload.get("model_family") or ""),
                    model_paths=tuple(models_by_run.get(run_key, ())),
                    evaluations=run_evaluations.get(run_key, ()),
                    progressions=run_progressions.get(run_key, ()),
                ),
            )
        )
    run_artifacts.sort(
        key=lambda item: (-item[0], item[1].relative_path.casefold(), item[1].relative_path)
    )
    state.issues.sort(key=lambda issue: (issue.path.casefold(), issue.code, issue.message))
    return ArtifactIndex(
        root=_absolute(root),
        models=tuple(model_artifacts),
        runs=tuple(artifact for _modified, artifact in run_artifacts),
        issues=tuple(state.issues),
        truncated=state.truncated,
        entries_examined=state.entries_examined,
        directories_examined=state.directories_examined,
        json_files_read=state.json_files_read,
        dropped_issue_count=state.dropped_issue_count,
    )


# Readable aliases for consumers; all three names preserve the no-selection API.
scan_artifacts = scan_run_artifacts
build_artifact_index = scan_run_artifacts


__all__ = [
    "ArtifactIndex",
    "ArtifactIssue",
    "ArtifactScanLimits",
    "EvaluationArtifactSummary",
    "ModelArtifact",
    "ModelCompatibilitySummary",
    "ProgressionArtifactSummary",
    "ROLE_BASELINE",
    "ROLE_BEST",
    "ROLE_CHECKPOINT",
    "ROLE_CURRENT",
    "ROLE_MODEL",
    "RunArtifact",
    "build_artifact_index",
    "scan_artifacts",
    "scan_run_artifacts",
]
