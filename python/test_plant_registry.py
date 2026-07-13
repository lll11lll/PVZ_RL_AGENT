"""Phase 2 parity locks for canonical plant metadata and bridge generation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from pvzrl_env import (
    PLANT_REGISTRY_PATH,
    load_plant_registry,
    plant_type_name,
    registry_entries,
    registry_entry_by_type,
    resolve_seed_list,
)
from pvzrl_fusion import normalize_plant_name_or_id, plant_name as fusion_plant_name
from pvzrl_registry import PlantRegistryError, get_plant_registry


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts" / "generate_bridge_registry.py"
GENERATED_CSHARP = ROOT / "src" / "PvZRLBridge" / "GeneratedPlantRegistry.cs"
BRIDGE_SOURCE_DIR = ROOT / "src" / "PvZRLBridge"
BUILD_SCRIPT = ROOT / "scripts" / "build_bridge.ps1"
GUI_SOURCE = ROOT / "python" / "pvzrl_gui.py"


def test_registry_is_cached_deeply_immutable_and_source_compatible() -> None:
    registry = get_plant_registry()
    assert registry is get_plant_registry()
    assert registry.source_path == PLANT_REGISTRY_PATH.resolve()
    assert registry.version == 1
    assert len(registry.plants) == 12
    assert isinstance(registry.plants, tuple)
    assert registry.get_by_id(0).canonical_name == "Peashooter"  # type: ignore[union-attr]
    assert registry.get_by_name("Sun Flower").plant_type_id == 1  # type: ignore[union-attr]
    assert registry.resolve_name("pea") == 0
    assert registry.canonical_name(9999) == "9999"

    with pytest.raises(TypeError):
        registry.by_id[99] = registry.plants[0]  # type: ignore[index]
    with pytest.raises(TypeError):
        registry.plants[0].training_flags["enabled_for_training"] = False  # type: ignore[index]
    assert isinstance(registry.plants[0].aliases, tuple)
    assert registry.plants[0].unlock_metadata["authority"] == "runtime_card_ui"
    assert registry.plants[0].fusion_metadata["identity_namespace"] == "base_seed"
    assert registry.plants[0].gui_display_metadata["display_name"] == "SunFlower"

    generalist = registry.require_gui_preset("adventure_generalist_initial_loadout")
    assert generalist.seed_names == ("SunFlower", "SunFlower", "Peashooter", "Peashooter")
    assert generalist.plant_type_ids == (1, 1, 0, 0)

    source_payload = json.loads(PLANT_REGISTRY_PATH.read_text(encoding="utf-8"))
    assert registry.to_legacy_payload() == source_payload
    assert load_plant_registry() == source_payload


def test_registry_path_cache_parses_once_and_validates_ambiguity(tmp_path: Path) -> None:
    custom = tmp_path / "registry.json"
    custom.write_text(PLANT_REGISTRY_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    first = get_plant_registry(custom)
    custom.write_text('{"version":99,"plants":[]}', encoding="utf-8")
    second = get_plant_registry(custom)
    assert second is first
    assert second.version == 1
    assert len(second.plants) == 12

    ambiguous = tmp_path / "ambiguous.json"
    payload = {
        "version": 1,
        "plants": [
            {
                "canonical_name": "Alpha Plant",
                "aliases": [],
                "plant_type_id": 1,
                "cost": 10,
                "cooldown": 1.0,
                "role": "utility",
                "enabled_for_training": True,
            },
            {
                "canonical_name": "Beta Plant",
                "aliases": ["AlphaPlant"],
                "plant_type_id": 2,
                "cost": 20,
                "cooldown": 2.0,
                "role": "utility",
                "enabled_for_training": True,
            },
        ],
    }
    ambiguous.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PlantRegistryError, match="ambiguous"):
        get_plant_registry(ambiguous)


def test_pvzrl_env_forwarders_preserve_mutable_shapes_and_seed_resolution() -> None:
    entries = registry_entries()
    assert [int(entry["plant_type_id"]) for entry in entries] == [1, 0, 3, 2, 4, 5, 6, 7, 8, 9, 10, 11]
    assert resolve_seed_list(["Sun Flower", "Pea", "7", "Peashooter:2"]) == [1, 0, 7, 0, 0]
    assert resolve_seed_list(["WallNut"]) == [3]
    assert plant_type_name(7) == "Repeater"
    assert plant_type_name(1030) == "1030"

    mutable = registry_entry_by_type(0)
    assert mutable is not None
    mutable["canonical_name"] = "changed locally"
    assert registry_entry_by_type(0)["canonical_name"] == "Peashooter"  # type: ignore[index]


def test_training_flag_does_not_filter_required_compatibility_plants() -> None:
    registry = get_plant_registry()
    assert registry.get_by_id(3).enabled_for_training is False  # type: ignore[union-attr]
    assert registry.get_by_id(2).enabled_for_training is False  # type: ignore[union-attr]
    assert resolve_seed_list(["WallNut", "CherryBomb"]) == [3, 2]


def test_fusion_result_identity_overlay_remains_separate_from_seed_registry() -> None:
    registry = get_plant_registry()
    assert registry.resolve_name("Repeater") == 7
    assert normalize_plant_name_or_id("Repeater") == 1030
    assert fusion_plant_name(7) == "Repeater"
    assert fusion_plant_name(1030) == "DoubleShooer"
    assert normalize_plant_name_or_id("SplitPea") == 1090
    assert fusion_plant_name(1090) == "SplitPea"


def test_gui_generalist_loadout_is_derived_from_named_registry_preset() -> None:
    gui = GUI_SOURCE.read_text(encoding="utf-8")
    assert 'require_gui_preset("adventure_generalist_initial_loadout")' in gui
    assert 'require_gui_preset("four_slot_current")' not in gui
    assert 'require_gui_preset("four_slot_duplicate")' not in gui
    assert '"SunFlower,SunFlower,Peashooter,Peashooter"' not in gui


def test_generated_bridge_registry_is_deterministic_and_complete(tmp_path: Path) -> None:
    checked = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert checked.returncode == 0, checked.stderr or checked.stdout

    regenerated = tmp_path / "GeneratedPlantRegistry.cs"
    generated = subprocess.run(
        [sys.executable, str(GENERATOR), "--output", str(regenerated)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert generated.returncode == 0, generated.stderr or generated.stdout
    assert regenerated.read_bytes() == GENERATED_CSHARP.read_bytes()

    registry = get_plant_registry()
    source = GENERATED_CSHARP.read_text(encoding="utf-8")
    assert f'public const string SourceSha256 = "{registry.content_sha256}";' in source
    assert source.count("new GeneratedPlantMetadata(") == len(registry.plants)
    assert source.count("bridgeFallbackEnabled: true") == 2
    for definition in registry.plants:
        assert f"[{definition.plant_type_id}] = new GeneratedPlantMetadata(" in source
        assert f'canonicalName: "{definition.canonical_name}"' in source


def test_bridge_uses_generated_fallbacks_after_runtime_card_sources() -> None:
    bridge = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(BRIDGE_SOURCE_DIR.glob("*.cs"), key=lambda path: path.name)
    )
    build = BUILD_SCRIPT.read_text(encoding="utf-8")
    get_cost_start = bridge.index("private PlantCostInfo GetPlantCost")
    get_cost_end = bridge.index("private static PlantCostInfo GetFallbackPlantCost", get_cost_start)
    get_cost_body = bridge[get_cost_start:get_cost_end]
    assert get_cost_body.index("CachedPlantCosts") < get_cost_body.index("TryReadPlantCostFromCards")
    assert get_cost_body.index("TryReadPlantCostFromCards") < get_cost_body.rindex("GetFallbackPlantCost")
    assert "GeneratedPlantRegistry.TryGetBridgeFallbackCost" in bridge
    assert "PlantType.SunFlower => 50" not in bridge
    assert "PlantType.Peashooter => 100" not in bridge
    assert "generate_bridge_registry.py" in build
    assert "GeneratedPlantRegistry.cs" in build
