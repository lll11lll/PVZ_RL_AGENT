"""Ordered transition evidence for the Phase 5 lifecycle shadow classifier."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest

from pvzrl_lifecycle import (
    LifecycleClassification,
    LifecycleContext,
    classify_lifecycle,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "refactor_contracts"
    / "lifecycle_ordered_sequences_phase5.json"
)
REQUIRED_SEQUENCES = {
    "startup_loading_seed_gameplay",
    "possible_win_trophy_reward",
    "post_win_same_level_seed_fresh",
    "loss_restart_seed_ready",
    "action_freeze_env_corruption_reset_ready",
    "corruption_reset_ready",
    "post_win_unlock_adventure_advancement",
    "stable_wrong_level_block",
}


def _fixture() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _observation(payload: dict[str, Any], frame: dict[str, Any]) -> dict[str, Any]:
    observation = copy.deepcopy(payload["defaults"])
    observation.update(copy.deepcopy(payload["observation_templates"][frame["template"]]))
    observation.update(copy.deepcopy(frame.get("observation", {})))
    return observation


def _projection(
    classification: LifecycleClassification,
    field_names: list[str],
) -> list[Any]:
    values: list[Any] = []
    for name in field_names:
        value = getattr(classification, name)
        values.append(list(value) if isinstance(value, tuple) else value)
    return values


@pytest.mark.parametrize(
    "sequence",
    _fixture()["sequences"],
    ids=lambda sequence: sequence["name"],
)
def test_ordered_lifecycle_transition_contracts(sequence: dict[str, Any]) -> None:
    """Every frame locks the complete classifier projection, order, and delta."""

    payload = _fixture()
    field_names = payload["classification_fields"]
    delta_fields = payload["delta_fields"]
    phases: list[str] = []
    legacy_actions: list[str] = []
    accumulated_delta = {name: 0 for name in delta_fields}

    for index, frame in enumerate(sequence["frames"]):
        observation = _observation(payload, frame)
        classification = classify_lifecycle(
            observation,
            context=LifecycleContext(**frame.get("context", {})),
            fallback_rows=5,
            start_sun=500,
        )
        expected = payload["classification_contracts"][frame["contract"]]
        assert _projection(classification, field_names) == expected, (
            f"{sequence['name']} frame {index} ({frame['name']}) changed its "
            "complete LifecycleClassification projection"
        )
        phases.append(classification.phase)
        legacy_actions.append(frame["legacy_action"])
        for name, value in frame.get("delta", {}).items():
            assert name in accumulated_delta
            accumulated_delta[name] += int(value)

    assert phases == sequence["expected_phase_order"]
    assert legacy_actions == sequence["expected_legacy_actions"]
    assert accumulated_delta == sequence["final_delta"]


def test_fixture_covers_every_lifecycle_projection_field_exactly_once() -> None:
    payload = _fixture()
    assert {sequence["name"] for sequence in payload["sequences"]} == REQUIRED_SEQUENCES
    runtime_fields = [field.name for field in fields(LifecycleClassification)]
    assert payload["classification_fields"] == runtime_fields
    assert len(runtime_fields) == len(set(runtime_fields))
    width = len(runtime_fields)
    used_contracts = {
        frame["contract"]
        for sequence in payload["sequences"]
        for frame in sequence["frames"]
    }
    assert used_contracts == set(payload["classification_contracts"])
    assert all(
        len(contract) == width
        for contract in payload["classification_contracts"].values()
    )


def test_ordered_evidence_has_auditable_sanitized_provenance() -> None:
    payload = _fixture()
    provenance = payload["provenance"]
    referenced = {
        frame["provenance"]
        for sequence in payload["sequences"]
        for frame in sequence["frames"]
    }
    assert referenced <= set(provenance)

    for name in referenced:
        evidence = provenance[name]
        source = str(evidence["source"])
        start, end = evidence["line_range"]
        assert not Path(source).is_absolute()
        assert 1 <= int(start) <= int(end)
        local_source = ROOT / source
        if local_source.exists():
            assert int(end) <= len(local_source.read_text(encoding="utf-8").splitlines())
        else:
            # Runtime logs are deliberately gitignored; their path, range, and
            # captured excerpt digest remain sufficient to audit the fixture.
            assert evidence["kind"] == "sanitized_live"

        if evidence["kind"] == "sanitized_live":
            assert source.startswith("runs/")
            assert evidence["sanitization"]
            assert re.fullmatch(r"[0-9a-f]{64}", evidence["source_excerpt_sha256"])


def test_sanitized_frames_contain_no_secret_or_machine_local_fields() -> None:
    payload = _fixture()
    forbidden_key_fragments = ("password", "secret", "token", "cookie", "model_path")
    local_path = re.compile(r"^[A-Za-z]:[\\/]")

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                lowered = str(key).lower()
                assert not any(fragment in lowered for fragment in forbidden_key_fragments)
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str):
            assert local_path.match(value) is None

    visit(payload["defaults"])
    visit(payload["observation_templates"])
    for sequence in payload["sequences"]:
        for frame in sequence["frames"]:
            visit(frame.get("observation", {}))
