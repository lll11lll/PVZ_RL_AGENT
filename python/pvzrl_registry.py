"""Canonical, immutable plant metadata for PvZRL.

Runtime ``CardUI`` values remain authoritative.  This registry supplies stable
names, aliases, fallback metadata, and training/display annotations when live
game metadata is unavailable.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLANT_REGISTRY_PATH = PROJECT_ROOT / "configs" / "plant_registry.json"
_DEFAULT_PLANT_REGISTRY_CACHE_KEY = str(DEFAULT_PLANT_REGISTRY_PATH.resolve())


class PlantRegistryError(ValueError):
    """Raised when canonical plant metadata is malformed or ambiguous."""


def normalize_plant_name(value: Any) -> str:
    """Return the compatibility name key used by existing seed-list parsing."""

    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return value


def _require_nonempty_text(
    entry: Mapping[str, Any],
    key: str,
    *,
    plant_index: int,
    collection: str = "plants",
) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PlantRegistryError(f"{collection}[{plant_index}].{key} must be a non-empty string")
    return value.strip()


def _require_int(entry: Mapping[str, Any], key: str, *, plant_index: int, minimum: int = 0) -> int:
    value = entry.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PlantRegistryError(f"plants[{plant_index}].{key} must be an integer >= {minimum}")
    return int(value)


def _require_number(entry: Mapping[str, Any], key: str, *, plant_index: int, minimum: float = 0.0) -> float:
    value = entry.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) < minimum:
        raise PlantRegistryError(f"plants[{plant_index}].{key} must be a number >= {minimum}")
    return float(value)


def _require_bool(entry: Mapping[str, Any], key: str, *, plant_index: int, default: bool = False) -> bool:
    value = entry.get(key, default)
    if not isinstance(value, bool):
        raise PlantRegistryError(f"plants[{plant_index}].{key} must be a boolean")
    return bool(value)


@dataclass(frozen=True, slots=True)
class PlantDefinition:
    canonical_name: str
    aliases: Tuple[str, ...]
    plant_type_id: int
    fallback_cost: int
    fallback_cooldown: float
    category: str
    role: str
    description: str
    source_of_metadata: str
    enabled_for_training: bool
    bridge_fallback_enabled: bool
    unlock_metadata: Mapping[str, Any]
    training_flags: Mapping[str, Any]
    fusion_metadata: Mapping[str, Any]
    gui_display_metadata: Mapping[str, Any]
    _legacy_entry: Mapping[str, Any] = field(repr=False, compare=False)

    def to_legacy_dict(self) -> dict[str, Any]:
        """Return the mutable dict shape historically exposed by ``pvzrl_env``."""

        return _deep_thaw(self._legacy_entry)


@dataclass(frozen=True, slots=True)
class PlantPreset:
    key: str
    display_name: str
    seed_names: Tuple[str, ...]
    plant_type_ids: Tuple[int, ...]

    @property
    def seed_csv(self) -> str:
        return ",".join(self.seed_names)

    @property
    def plant_type_csv(self) -> str:
        return ",".join(str(plant_type) for plant_type in self.plant_type_ids)


@dataclass(frozen=True, slots=True)
class PlantRegistry:
    source_path: Path
    version: int
    notes: str
    plants: Tuple[PlantDefinition, ...]
    gui_presets: Tuple[PlantPreset, ...]
    content_sha256: str
    _by_id: Mapping[int, PlantDefinition] = field(repr=False, compare=False)
    _by_name: Mapping[str, PlantDefinition] = field(repr=False, compare=False)
    _presets_by_key: Mapping[str, PlantPreset] = field(repr=False, compare=False)
    _legacy_payload: Mapping[str, Any] = field(repr=False, compare=False)

    @property
    def by_id(self) -> Mapping[int, PlantDefinition]:
        return self._by_id

    @property
    def by_normalized_name(self) -> Mapping[str, PlantDefinition]:
        return self._by_name

    def get_by_id(self, plant_type_id: int) -> Optional[PlantDefinition]:
        try:
            return self._by_id.get(int(plant_type_id))
        except (TypeError, ValueError):
            return None

    def get_by_name(self, value: Any) -> Optional[PlantDefinition]:
        key = normalize_plant_name(value)
        return self._by_name.get(key) if key else None

    def resolve_name(self, value: Any) -> Optional[int]:
        definition = self.get_by_name(value)
        return definition.plant_type_id if definition is not None else None

    def canonical_name(self, plant_type_id: int, fallback: Optional[str] = None) -> str:
        definition = self.get_by_id(plant_type_id)
        if definition is not None:
            return definition.canonical_name
        return str(plant_type_id) if fallback is None else str(fallback)

    def enabled_for_training(self) -> Tuple[PlantDefinition, ...]:
        return tuple(definition for definition in self.plants if definition.enabled_for_training)

    def get_gui_preset(self, key: str) -> Optional[PlantPreset]:
        return self._presets_by_key.get(str(key).strip().lower())

    def require_gui_preset(self, key: str) -> PlantPreset:
        preset = self.get_gui_preset(key)
        if preset is None:
            raise PlantRegistryError(f"unknown GUI plant preset: {key!r}")
        return preset

    def to_legacy_payload(self) -> dict[str, Any]:
        return _deep_thaw(self._legacy_payload)


def _canonical_payload_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _build_registry(payload: Mapping[str, Any], source_path: Path) -> PlantRegistry:
    if not isinstance(payload, Mapping):
        raise PlantRegistryError("plant registry root must be a JSON object")

    version = payload.get("version", 0)
    if isinstance(version, bool) or not isinstance(version, int) or version < 0:
        raise PlantRegistryError("plant registry version must be a non-negative integer")
    notes = payload.get("notes", "")
    if not isinstance(notes, str):
        raise PlantRegistryError("plant registry notes must be a string")
    raw_plants = payload.get("plants", [])
    if not isinstance(raw_plants, list):
        raise PlantRegistryError("plant registry plants must be a list")

    definitions = []
    by_id: dict[int, PlantDefinition] = {}
    by_name: dict[str, PlantDefinition] = {}
    for plant_index, raw_entry in enumerate(raw_plants):
        if not isinstance(raw_entry, Mapping):
            raise PlantRegistryError(f"plants[{plant_index}] must be a JSON object")

        canonical_name = _require_nonempty_text(raw_entry, "canonical_name", plant_index=plant_index)
        plant_type_id = _require_int(raw_entry, "plant_type_id", plant_index=plant_index)
        fallback_cost = _require_int(raw_entry, "cost", plant_index=plant_index)
        fallback_cooldown = _require_number(raw_entry, "cooldown", plant_index=plant_index)
        role = str(raw_entry.get("role") or "unknown").strip().lower() or "unknown"
        category = str(raw_entry.get("category") or role).strip().lower() or "unknown"
        description = str(raw_entry.get("description") or "")
        source_of_metadata = str(raw_entry.get("source_of_metadata") or "")
        enabled_for_training = _require_bool(
            raw_entry,
            "enabled_for_training",
            plant_index=plant_index,
        )
        bridge_fallback_enabled = _require_bool(
            raw_entry,
            "bridge_fallback_enabled",
            plant_index=plant_index,
            default=False,
        )

        raw_aliases = raw_entry.get("aliases", [])
        if not isinstance(raw_aliases, list) or any(not isinstance(alias, str) for alias in raw_aliases):
            raise PlantRegistryError(f"plants[{plant_index}].aliases must be a list of strings")
        aliases = tuple(alias.strip() for alias in raw_aliases if alias.strip())

        if plant_type_id in by_id:
            raise PlantRegistryError(f"duplicate plant_type_id {plant_type_id}")

        raw_training_flags = raw_entry.get("training_flags", {})
        if not isinstance(raw_training_flags, Mapping):
            raise PlantRegistryError(f"plants[{plant_index}].training_flags must be a JSON object")
        training_flags = dict(raw_training_flags)
        training_flags.setdefault("enabled_for_training", enabled_for_training)
        training_flags.setdefault("bridge_fallback_enabled", bridge_fallback_enabled)
        for metadata_key in ("unlock_metadata", "fusion_metadata", "gui_display_metadata"):
            metadata_value = raw_entry.get(metadata_key, {})
            if not isinstance(metadata_value, Mapping):
                raise PlantRegistryError(f"plants[{plant_index}].{metadata_key} must be a JSON object")
        unlock_metadata = dict(raw_entry.get("unlock_metadata", {}))
        unlock_metadata.setdefault("authority", "runtime_card_ui")
        fusion_metadata = dict(raw_entry.get("fusion_metadata", {}))
        fusion_metadata.setdefault("identity_namespace", "base_seed")
        gui_display_metadata = dict(raw_entry.get("gui_display_metadata", {}))
        gui_display_metadata.setdefault("display_name", canonical_name)
        gui_display_metadata.setdefault("registry_order", plant_index)

        definition = PlantDefinition(
            canonical_name=canonical_name,
            aliases=aliases,
            plant_type_id=plant_type_id,
            fallback_cost=fallback_cost,
            fallback_cooldown=fallback_cooldown,
            category=category,
            role=role,
            description=description,
            source_of_metadata=source_of_metadata,
            enabled_for_training=enabled_for_training,
            bridge_fallback_enabled=bridge_fallback_enabled,
            unlock_metadata=_deep_freeze(unlock_metadata),
            training_flags=_deep_freeze(training_flags),
            fusion_metadata=_deep_freeze(fusion_metadata),
            gui_display_metadata=_deep_freeze(gui_display_metadata),
            _legacy_entry=_deep_freeze(dict(raw_entry)),
        )
        definitions.append(definition)
        by_id[plant_type_id] = definition

        for raw_name in (canonical_name, *aliases):
            name_key = normalize_plant_name(raw_name)
            if not name_key:
                continue
            previous = by_name.get(name_key)
            if previous is not None and previous.plant_type_id != plant_type_id:
                raise PlantRegistryError(
                    f"plant name/alias {raw_name!r} is ambiguous between "
                    f"{previous.plant_type_id} and {plant_type_id}"
                )
            by_name[name_key] = definition

    raw_presets = payload.get("gui_presets", [])
    if not isinstance(raw_presets, list):
        raise PlantRegistryError("plant registry gui_presets must be a list")
    presets: list[PlantPreset] = []
    presets_by_key: dict[str, PlantPreset] = {}
    preset_display_names: set[str] = set()
    for preset_index, raw_preset in enumerate(raw_presets):
        if not isinstance(raw_preset, Mapping):
            raise PlantRegistryError(f"gui_presets[{preset_index}] must be a JSON object")
        key = _require_nonempty_text(
            raw_preset,
            "key",
            plant_index=preset_index,
            collection="gui_presets",
        ).lower()
        display_name = _require_nonempty_text(
            raw_preset,
            "display_name",
            plant_index=preset_index,
            collection="gui_presets",
        )
        raw_seed_names = raw_preset.get("seed_list", [])
        if not isinstance(raw_seed_names, list) or not raw_seed_names or any(
            not isinstance(seed_name, str) or not seed_name.strip() for seed_name in raw_seed_names
        ):
            raise PlantRegistryError(f"gui_presets[{preset_index}].seed_list must be a non-empty list of names")
        if key in presets_by_key:
            raise PlantRegistryError(f"duplicate GUI preset key {key!r}")
        display_key = display_name.strip().lower()
        if display_key in preset_display_names:
            raise PlantRegistryError(f"duplicate GUI preset display_name {display_name!r}")
        definitions_for_preset: list[PlantDefinition] = []
        for seed_name in raw_seed_names:
            definition = by_name.get(normalize_plant_name(seed_name))
            if definition is None:
                raise PlantRegistryError(
                    f"gui_presets[{preset_index}].seed_list references unknown plant {seed_name!r}"
                )
            definitions_for_preset.append(definition)
        preset = PlantPreset(
            key=key,
            display_name=display_name,
            seed_names=tuple(definition.canonical_name for definition in definitions_for_preset),
            plant_type_ids=tuple(definition.plant_type_id for definition in definitions_for_preset),
        )
        presets.append(preset)
        presets_by_key[key] = preset
        preset_display_names.add(display_key)

    canonical_bytes = _canonical_payload_bytes(payload)
    return PlantRegistry(
        source_path=source_path,
        version=int(version),
        notes=notes,
        plants=tuple(definitions),
        gui_presets=tuple(presets),
        content_sha256=hashlib.sha256(canonical_bytes).hexdigest(),
        _by_id=MappingProxyType(by_id),
        _by_name=MappingProxyType(by_name),
        _presets_by_key=MappingProxyType(presets_by_key),
        _legacy_payload=_deep_freeze(dict(payload)),
    )


@lru_cache(maxsize=None)
def _load_registry_cached(resolved_path: str) -> PlantRegistry:
    path = Path(resolved_path)
    if not path.exists():
        return _build_registry({"version": 0, "plants": []}, path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PlantRegistryError(f"invalid plant registry JSON at {path}: {exc}") from exc
    return _build_registry(payload, path)


def get_plant_registry(path: Path = DEFAULT_PLANT_REGISTRY_PATH) -> PlantRegistry:
    """Load and validate a registry once per resolved path."""

    candidate = Path(path)
    cache_key = (
        _DEFAULT_PLANT_REGISTRY_CACHE_KEY
        if candidate == DEFAULT_PLANT_REGISTRY_PATH
        else str(candidate.resolve())
    )
    return _load_registry_cached(cache_key)


def clear_plant_registry_cache() -> None:
    """Test/development hook for explicitly reloading edited registry files."""

    _load_registry_cached.cache_clear()
