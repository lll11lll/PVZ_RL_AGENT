"""Stage-based Adventure model routing for PvZRL."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pvzrl_model_metadata import apply_model_metadata_defaults, normalized_seed_list


@dataclass(frozen=True)
class ModelStage:
    stage_id: str
    family: str
    model_path: Path
    level_start: int
    level_end: int
    seed_list: List[str]
    plant_types: List[int]
    requires_unlocked: List[str] = field(default_factory=list)
    requires_available: List[str] = field(default_factory=list)
    action_space_mode: str = "fixed"

    def contains_level(self, level: int) -> bool:
        return self.level_start <= int(level) <= self.level_end

    def to_config_overlay(self) -> Dict[str, Any]:
        return {
            "model_family": self.family,
            "model_path": str(self.model_path),
            "seed_list": list(self.seed_list),
            "plant_types": list(self.plant_types),
            "action_space_mode": self.action_space_mode,
            "max_seed_slots": 14 if self.action_space_mode == "dynamic_14" else len(self.plant_types),
        }


@dataclass
class RouterDecision:
    ok: bool
    stage: Optional[ModelStage] = None
    blocked_reason: str = ""
    level: int = 0
    level_label: str = ""
    missing_required_unlocked: List[str] = field(default_factory=list)
    missing_required_available: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "blocked_reason": self.blocked_reason,
            "level": self.level,
            "level_label": self.level_label,
            "stage": self.stage.stage_id if self.stage else "",
            "family": self.stage.family if self.stage else "",
            "model_path": str(self.stage.model_path) if self.stage else "",
            "seed_list": list(self.stage.seed_list) if self.stage else [],
            "plant_types": list(self.stage.plant_types) if self.stage else [],
            "requires_unlocked": list(self.stage.requires_unlocked) if self.stage else [],
            "requires_available": list(self.stage.requires_available) if self.stage else [],
            "missing_required_unlocked": list(self.missing_required_unlocked),
            "missing_required_available": list(self.missing_required_available),
        }


class ModelRouter:
    def __init__(self, stages: List[ModelStage], source_path: Optional[Path] = None):
        self.stages = list(stages)
        self.source_path = source_path

    @classmethod
    def from_file(cls, path: Path) -> "ModelRouter":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(parse_schedule(data, base_dir=path.parent), source_path=path)

    def select_stage(
        self,
        *,
        level: Any,
        unlocked_seeds: Iterable[Any],
        available_seeds: Iterable[Any],
    ) -> RouterDecision:
        numeric_level = parse_adventure_level(level)
        stage = next((candidate for candidate in self.stages if candidate.contains_level(numeric_level)), None)
        if stage is None:
            return RouterDecision(
                ok=False,
                blocked_reason=f"no_stage_for_level:{level_label(numeric_level)}",
                level=numeric_level,
                level_label=level_label(numeric_level),
            )
        unlocked_set = set(_ordered_unique(unlocked_seeds))
        available_set = set(_ordered_unique(available_seeds))
        missing_unlocked = [seed for seed in stage.requires_unlocked if seed not in unlocked_set]
        if missing_unlocked:
            return RouterDecision(
                ok=False,
                stage=stage,
                blocked_reason=f"missing_required_unlocked:{missing_unlocked[0]}",
                level=numeric_level,
                level_label=level_label(numeric_level),
                missing_required_unlocked=missing_unlocked,
            )
        missing_available = [seed for seed in stage.requires_available if seed not in available_set]
        if missing_available:
            return RouterDecision(
                ok=False,
                stage=stage,
                blocked_reason=f"missing_required_available:{missing_available[0]}",
                level=numeric_level,
                level_label=level_label(numeric_level),
                missing_required_available=missing_available,
            )
        return RouterDecision(ok=True, stage=stage, level=numeric_level, level_label=level_label(numeric_level))

    def detect_level(self, adventure_state: Dict[str, Any], fallback_level: int) -> int:
        for key in ("currentAdventureLevel", "adventureLevel", "level", "currentLevel"):
            value = adventure_state.get(key)
            if value in (None, ""):
                continue
            try:
                parsed = parse_adventure_level(value)
            except ValueError:
                continue
            if parsed > 0:
                return parsed
        return int(fallback_level)


def parse_schedule(data: Dict[str, Any], base_dir: Path) -> List[ModelStage]:
    raw_stages = data.get("stages", [])
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError("model schedule must include a non-empty stages list")
    stages: List[ModelStage] = []
    for index, raw in enumerate(raw_stages):
        if not isinstance(raw, dict):
            raise ValueError(f"stage index {index} is not an object")
        start, end = parse_level_range(raw.get("levels") or raw.get("level_range") or raw.get("range"))
        model_path = Path(str(raw.get("model_path") or ""))
        if model_path and not model_path.is_absolute():
            repo_relative = Path.cwd() / model_path
            schedule_relative = base_dir / model_path
            model_path = repo_relative if repo_relative.exists() else schedule_relative
        plant_types = [int(value) for value in raw.get("plant_types", [])]
        seed_list = normalized_seed_list(raw.get("seed_list", []))
        if not seed_list or not plant_types:
            raise ValueError(f"stage {raw.get('id', index)!r} must define seed_list and plant_types")
        stages.append(
            ModelStage(
                stage_id=str(raw.get("id") or raw.get("name") or f"stage_{index}"),
                family=str(raw.get("family") or raw.get("model_family") or raw.get("id") or f"stage_{index}"),
                model_path=model_path,
                level_start=start,
                level_end=end,
                seed_list=seed_list,
                plant_types=plant_types,
                requires_unlocked=_ordered_unique(raw.get("requires_unlocked", [])),
                requires_available=_ordered_unique(raw.get("requires_available", [])),
                action_space_mode=str(raw.get("action_space_mode") or "fixed"),
            )
        )
    return stages


def parse_level_range(value: Any) -> Tuple[int, int]:
    if isinstance(value, dict):
        start = parse_adventure_level(value.get("start"))
        end = parse_adventure_level(value.get("end", value.get("start")))
        return min(start, end), max(start, end)
    if isinstance(value, list) and value:
        start = parse_adventure_level(value[0])
        end = parse_adventure_level(value[-1])
        return min(start, end), max(start, end)
    text = str(value or "").strip()
    for separator in ("..", " through ", "-", "to"):
        if separator == "-" and text.count("-") == 1 and text.replace("-", "").isdigit():
            continue
        if separator in text:
            parts = [part.strip() for part in text.split(separator) if part.strip()]
            if len(parts) == 2:
                start = parse_adventure_level(parts[0])
                end = parse_adventure_level(parts[1])
                return min(start, end), max(start, end)
    level = parse_adventure_level(text)
    return level, level


def parse_adventure_level(value: Any) -> int:
    if isinstance(value, int):
        return max(1, int(value))
    text = str(value or "").strip()
    if not text:
        raise ValueError("empty Adventure level")
    if text.isdigit():
        return max(1, int(text))
    if "-" in text:
        world_text, level_text = text.split("-", 1)
        world = int(world_text)
        level = int(level_text)
        return max(1, (world - 1) * 10 + level)
    raise ValueError(f"Invalid Adventure level: {value!r}")


def level_label(level: int) -> str:
    numeric = max(1, int(level))
    world = (numeric - 1) // 10 + 1
    level_in_world = (numeric - 1) % 10 + 1
    return f"{world}-{level_in_world}"


def stage_config(base_config: Dict[str, Any], stage: ModelStage) -> Dict[str, Any]:
    config = dict(base_config)
    config.update(stage.to_config_overlay())
    return apply_model_metadata_defaults(config)


def _ordered_unique(values: Iterable[Any]) -> List[str]:
    output: List[str] = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        output.append(text)
        seen.add(text)
    return output
