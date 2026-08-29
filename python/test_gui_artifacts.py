from __future__ import annotations

import json
import os
from pathlib import Path

from pvzrl_gui_artifacts import (
    ArtifactScanLimits,
    ROLE_BASELINE,
    ROLE_BEST,
    ROLE_CHECKPOINT,
    ROLE_CURRENT,
    ROLE_MODEL,
    scan_run_artifacts,
)


def _full_v2_metadata(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "metadata_version": 1,
        "model_family": "ppo_adventure_generalist_14slot_identity_full_v2",
        "seed_list": ["SunFlower", "SunFlower", "Peashooter", "Peashooter"],
        "plant_types": [1, 1, 0, 0],
        "action_space_mode": "adventure_14slot_identity_full_v2",
        "action_count": 841,
        "max_seed_slots": 14,
        "dynamic_seed_slots": True,
        "identity_seed_slots": True,
        "observation_version": "adventure_14slot_identity_full_v2",
        "action_decoder_version": "seedslot14x60_padded6x10_plus_wait_v2",
        "decoder_wait_action": 0,
        "placement_action_range": [1, 840],
        "rows": 6,
        "cols": 10,
        "cells_per_seed_slot": 60,
        "observation_shape": [4364],
        "created_at": "2026-08-12T12:00:00Z",
    }
    payload.update(updates)
    return payload


def _write_model(path: Path, metadata: dict[str, object] | str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"synthetic-model\n")
    if metadata is not None:
        metadata_path = path.parent / "model_metadata.json"
        metadata_path.write_text(
            metadata if isinstance(metadata, str) else json.dumps(metadata),
            encoding="utf-8",
        )
    return path


def _by_relative(index) -> dict[str, object]:
    return {model.relative_path: model for model in index.models}


def test_index_classifies_canonical_streamer_roles_and_keeps_explicit_paths(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    baseline = _write_model(runs / "baseline_source" / "model.zip", _full_v2_metadata())

    experiment = runs / "streamer_experiment"
    (experiment / "streamer_state.json").parent.mkdir(parents=True, exist_ok=True)
    (experiment / "streamer_state.json").write_text(
        json.dumps({"status": "running", "mode": "STREAM_TRAIN", "next_adventure_level": 4}),
        encoding="utf-8",
    )
    checkpoint_root = experiment / "checkpoints"
    current_alias = _write_model(
        checkpoint_root / "current" / "model.zip", _full_v2_metadata()
    )
    current_version = _write_model(
        checkpoint_root / "current" / "versions" / "v-a" / "model.zip",
        _full_v2_metadata(),
    )
    best_alias = _write_model(checkpoint_root / "best" / "model.zip", _full_v2_metadata())
    best_version = _write_model(
        checkpoint_root / "best" / "versions" / "v-b" / "model.zip",
        _full_v2_metadata(),
    )
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    (checkpoint_root / "baseline.json").write_text(
        json.dumps({"role": "BASELINE", "model_path": str(baseline.resolve())}),
        encoding="utf-8",
    )
    (current_alias.parent / "streamer_checkpoint.json").write_text(
        json.dumps(
            {
                "role": "CURRENT",
                "model_path": str(current_version.resolve()),
                "compatibility_alias": str(current_alias.resolve()),
                "model_steps": 25_000,
                "saved_at": "2026-08-12T13:00:00Z",
                "training_metrics": {"next_adventure_level": 5},
            }
        ),
        encoding="utf-8",
    )
    (best_alias.parent / "streamer_checkpoint.json").write_text(
        json.dumps(
            {
                "role": "BEST",
                "model_path": str(best_version.resolve()),
                "compatibility_alias": str(best_alias.resolve()),
                "model_steps": 20_000,
                "saved_at": "2026-08-12T12:30:00Z",
                "evaluation": {
                    "adventure_start_level": 4,
                    "next_adventure_level": 5,
                    "summary": {"win_rate": 0.75, "avg_reward": 12.5},
                },
            }
        ),
        encoding="utf-8",
    )

    ordinary_run = runs / "ordinary"
    ordinary_run.mkdir(parents=True)
    (ordinary_run / "summary.json").write_text(
        json.dumps({"model_steps": 900, "run_mode": "adventure_generalist_14slot_train"}),
        encoding="utf-8",
    )
    (ordinary_run / "model_metadata.json").write_text(
        json.dumps(_full_v2_metadata()), encoding="utf-8"
    )
    checkpoint = _write_model(
        ordinary_run / "checkpoints" / "ppo_pvz_900_steps.zip"
    )
    ordinary_model = _write_model(ordinary_run / "model.zip")

    index = scan_run_artifacts(runs)
    models = _by_relative(index)

    assert models["baseline_source/model.zip"].role == ROLE_BASELINE
    assert models["streamer_experiment/checkpoints/current/model.zip"].role == ROLE_CURRENT
    assert models[
        "streamer_experiment/checkpoints/current/versions/v-a/model.zip"
    ].role == ROLE_CURRENT
    assert models["streamer_experiment/checkpoints/best/model.zip"].role == ROLE_BEST
    assert models[
        "streamer_experiment/checkpoints/best/versions/v-b/model.zip"
    ].role == ROLE_BEST
    assert models["ordinary/checkpoints/ppo_pvz_900_steps.zip"].role == ROLE_CHECKPOINT
    assert models["ordinary/model.zip"].role == ROLE_MODEL

    current = models["streamer_experiment/checkpoints/current/model.zip"]
    assert current.path == str(current_alias.resolve())
    assert current.timesteps == 25_000
    assert current.adventure_level == 5
    assert current.run_dir == str(experiment.resolve())
    assert models["ordinary/checkpoints/ppo_pvz_900_steps.zip"].path == str(checkpoint.resolve())
    assert models["ordinary/model.zip"].path == str(ordinary_model.resolve())
    assert not hasattr(index, "selected_model")


def test_malformed_metadata_is_reported_without_aborting_other_models(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    broken = _write_model(runs / "broken" / "model.zip", "{ definitely-not-json")
    valid = _write_model(runs / "valid" / "model.zip", _full_v2_metadata())

    index = scan_run_artifacts(runs)
    models = {model.path: model for model in index.models}

    assert models[str(broken.resolve())].compatibility.compatible is False
    assert models[str(broken.resolve())].compatibility.metadata_status == "MALFORMED"
    assert models[str(broken.resolve())].compatibility.blocked_reason == "invalid_model_metadata"
    assert "invalid_json" in models[str(broken.resolve())].compatibility.metadata_error
    assert models[str(valid.resolve())].compatibility.compatible is True
    assert any(
        Path(issue.path).parts[-2:] == ("broken", "model_metadata.json")
        for issue in index.issues
    )


def test_historical_v1_checkpoint_is_explicitly_incompatible(tmp_path: Path) -> None:
    model = _write_model(
        tmp_path / "runs" / "historical_v1" / "model.zip",
        {
            "metadata_version": 1,
            "model_family": "ppo_adventure_generalist_14slot_identity_v1",
            "action_space_mode": "adventure_14slot_identity_v1",
            "action_count": 701,
            "max_seed_slots": 14,
            "dynamic_seed_slots": True,
            "identity_seed_slots": True,
            "observation_version": "adventure_14slot_identity_v1",
            "action_decoder_version": "seedslot14x50_plus_wait_v1",
            "decoder_wait_action": 0,
            "placement_action_range": [1, 700],
            "rows": 5,
            "cols": 10,
            "cells_per_seed_slot": 50,
            "observation_shape": [4297],
        },
    )

    entry = scan_run_artifacts(tmp_path / "runs").models[0]

    assert entry.path == str(model.resolve())
    assert entry.compatibility.compatible is False
    assert entry.compatibility.blocked_reason
    assert entry.compatibility.action_count == 701
    assert entry.compatibility.observation_shape == (4297,)
    assert entry.compatibility.rows == 5


def test_full_adventure_v2_compatibility_reports_every_fixed_semantic_dimension(
    tmp_path: Path,
) -> None:
    model = _write_model(tmp_path / "runs" / "full_v2" / "model.zip", _full_v2_metadata())
    run = model.parent
    (run / "summary.json").write_text(
        json.dumps({"status": "complete", "model_steps": 123_456}), encoding="utf-8"
    )
    (run / "adventure_training_progress.json").write_text(
        json.dumps(
            {
                "status": "running",
                "current_level": 7,
                "frontier_level": 7,
                "cleared_levels": [1, 2, 3, 4, 5, 6],
                "latest": {"episode": 19, "attempt": 2, "level": 7},
            }
        ),
        encoding="utf-8",
    )

    entry = scan_run_artifacts(tmp_path / "runs").models[0]
    compatibility = entry.compatibility

    assert compatibility.compatible is True
    assert compatibility.metadata_status == "VALID"
    assert compatibility.model_family == "ppo_adventure_generalist_14slot_identity_full_v2"
    assert compatibility.action_space_mode == "adventure_14slot_identity_full_v2"
    assert compatibility.action_count == 841
    assert compatibility.action_decoder_version == "seedslot14x60_padded6x10_plus_wait_v2"
    assert compatibility.observation_version == "adventure_14slot_identity_full_v2"
    assert compatibility.observation_shape == (4364,)
    assert compatibility.max_seed_slots == 14
    assert compatibility.dynamic_seed_slots is True
    assert compatibility.identity_seed_slots is True
    assert compatibility.decoder_wait_action == 0
    assert compatibility.placement_action_range == (1, 840)
    assert (compatibility.rows, compatibility.cols, compatibility.cells_per_seed_slot) == (6, 10, 60)
    assert entry.timesteps == 123_456
    assert entry.timesteps_source == "summary:model_steps"
    assert entry.adventure_level == 7
    assert entry.progressions[0].cleared_levels == (1, 2, 3, 4, 5, 6)


def test_ordering_is_deterministic_and_uses_path_as_the_equal_timestamp_tiebreaker(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "runs"
    zeta = _write_model(runs / "zeta" / "model.zip", _full_v2_metadata())
    alpha = _write_model(runs / "Alpha" / "model.zip", _full_v2_metadata())
    same_timestamp = 1_800_000_000
    os.utime(alpha, (same_timestamp, same_timestamp))
    os.utime(zeta, (same_timestamp, same_timestamp))

    first = scan_run_artifacts(runs)
    second = scan_run_artifacts(runs)

    first_paths = [entry.relative_path for entry in first.models]
    assert first_paths == ["Alpha/model.zip", "zeta/model.zip"]
    assert [entry.relative_path for entry in second.models] == first_paths
    assert [run.relative_path for run in second.runs] == [
        run.relative_path for run in first.runs
    ]


def test_scan_limits_bound_depth_results_json_bytes_and_issue_retention(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _write_model(runs / "one" / "model.zip", _full_v2_metadata())
    _write_model(runs / "two" / "model.zip", _full_v2_metadata())
    deep = runs / "a" / "b" / "c"
    _write_model(deep / "model.zip", _full_v2_metadata())

    index = scan_run_artifacts(
        runs,
        limits=ArtifactScanLimits(
            max_depth=1,
            max_entries=100,
            max_models=1,
            max_runs=1,
            max_json_reads=10,
            max_json_bytes=10_000,
            max_summaries_per_run=2,
            max_issues=2,
        ),
    )

    assert index.truncated is True
    assert len(index.models) == 1
    assert len(index.runs) == 1
    assert len(index.issues) <= 2
    assert index.entries_examined <= 100
    assert index.json_files_read <= 10
