# PvZRL Adventure Generalist Refactor Plan

Status: implementation is in final integration and verification. The obsolete product path has been removed from source/configuration/GUI/tests; repository-wide validation and live Generalist acceptance remain the finish line.

Adventure Generalist is the sole maintained training and evaluation path. Legacy fixed/specialist mode was removed during the repository refactor.

## Objective

Reduce PvZRL to one coherent train/eval/checkpoint surface while preserving the reusable environment, masks, observations, action execution, fusion, rewards, bridge, lifecycle, progression, coaches, diagnostics, and GUI infrastructure required by Adventure Generalist.

The protected compatibility target is:

| Contract | Required value |
| --- | --- |
| Model family | `ppo_adventure_generalist_14slot_identity_v1` |
| Action mode | `adventure_14slot_identity` |
| Actions | `701` (`0` wait, `1..700` placement/fusion) |
| Decoder | `seedslot14x50_plus_wait_v1` |
| Observation | `adventure_14slot_identity_v1`, shape `(4297,)` |
| Board/capacity | 5x10, 14 identity slots |
| Protected checkpoint | `runs/ppo_adventure_generalist_14slot_identity_v1_20260627_172727/checkpoints/ppo_pvz_370000_steps.zip` |
| Initial identity loadout | `SunFlower,SunFlower,Peashooter,Peashooter` / `[1,1,0,0]` |

## Scope classification

The pre-removal classification is retained in `docs/FIXED_MODE_REMOVAL_INVENTORY.md`.

### Generalist-required

- `python/pvzrl_adventure_generalist.py`
- `python/pvzrl_generalist_progression.py`
- Adventure attempt/lifecycle orchestration in `python/pvzrl_adventure.py`
- Generalist CLI/config/metadata/checkpoint dispatch in `python/train_ppo.py` and `python/pvzrl_config.py`
- Generalist GUI train/resume/eval and model-selection workflows
- the protected 370k checkpoint and adjacent metadata

### Shared reusable infrastructure

- base environment, bridge client, reset state machine, and lifecycle predicates;
- 14-slot observation construction, `StepFacts`, action masks, cached legality, and action execution;
- fusion recipes/probes/execution, rewards, metrics, watchdog, telemetry, and live status;
- registry generation, coach inputs, Tk process/status/queue infrastructure;
- C# bridge server, DTOs, Unity dispatch, observation, placement, seed UI, Adventure UI, fusion, and reset;
- bridge-free tests, C# lifecycle harness, and synthetic benchmarks that exercise maintained behavior.

### Removed product-specific surface

- alternate training/evaluation dispatch and CLI selectors;
- level-specific startup gates, loadouts, model selection, and reporting;
- alternate action/observation layouts and action-ID adapters;
- obsolete configs, schedules, GUI controls, callbacks, metadata branches, tests, fixtures, and runnable documentation;
- compatibility wrappers whose only consumer was the removed path.

### Investigate-before-delete rule

Generic filenames are not proof that code is obsolete. `pvzrl_adventure.py`, environment lifecycle helpers, telemetry builders, and bridge operations are retained when the Generalist runner consumes them. Before deleting an unclear symbol, search definitions, imports, call sites, configuration keys, serialized outputs, tests, and protected checkpoint metadata.

## Incremental implementation sequence

### 1. Inventory and safety boundary

- Record the classification and protected contract.
- Preserve hashes and paths for the protected checkpoint, metadata, installed bridge DLL, and local profile before live work.
- Stop any task that assumes the removed path remains a compatibility invariant.

Exit: inventory exists and no deletion begins without dependency classification.

### 2. Remove obsolete files and dispatch

- Delete unused configs, schedules, router/backfill utilities, and dedicated artifacts.
- Reduce `train_ppo.py`, `pvzrl_config.py`, and environment run-mode validation to Generalist train/eval.
- Remove unsupported flags and reject old run-mode/action-mode strings.
- Preserve config precedence: explicit CLI > JSON > Generalist mode default > global default.

Exit: only two run modes are constructible and no import points to deleted modules.

### 3. Collapse core contracts

- Make `pvzrl_action_space.py` identity-only: 701 actions, wait 0, actions 1..700.
- Remove action-ID conversion and obsolete intent fields.
- Pin environment/SB3 validation to 14 slots and 5x10 geometry.
- Preserve `dynamic_seed_slots=true` as checkpoint metadata describing inventory capability, not another decoder.
- Keep observation shape/version and board identity behavior byte-for-byte compatible with the protected model.

Exit: exact action/observation contract tests pass and the protected checkpoint loads.

### 4. Simplify GUI and local coaching

- Expose only Generalist fresh train, resume, evaluation, diagnostics, runs/models, and coach workflows.
- Remove obsolete tabs, presets, argv builders, and status projections.
- Keep human-over-crowd precedence, dry-run defaults, intervention logging, and bounded shutdown.

Exit: command snapshots contain only supported flags and withdrawn Tk construction/shutdown passes.

### 5. Replace tests and fixtures

- Delete tests whose sole purpose was removed checkpoint/layout compatibility.
- Replace them with negative rejection tests for unsupported modes plus exact Generalist contract tests.
- Preserve regression coverage for masks, identity observation, progression, timeouts, fusion/recursive fusion, rewards, watchdog, telemetry, bridge lifecycle, and GUI.
- Update deterministic hashes only after reviewing the intended schema change.

Exit: focused suites and full `pytest` pass without imports or fixtures for removed code.

### 6. Rewrite documentation

- Update `AGENTS.md`, configuration, architecture, learning, bridge, coach, roadmap, refactor plan/report, and README material.
- Retain concise historical removal statements only where they prevent accidental reintroduction.
- Remove runnable unsupported commands and unsupported compatibility promises.

Exit: repository Markdown and source scans find no obsolete support language outside the deliberate inventory/removal statement.

### 7. Verification

Bridge-free gates:

```powershell
python .\python\train_ppo.py --check-deps
python -m compileall -q python
python -m pytest -q
python .\python\train_ppo.py --config configs\ppo_adventure_generalist_14slot_identity_v1.json --metadata-dry-run --model-path runs\ppo_adventure_generalist_14slot_identity_v1_20260627_172727\checkpoints\ppo_pvz_370000_steps.zip
```

Bridge gates:

```powershell
.\scripts\build_bridge.ps1
.\scripts\test_bridge_lifecycle.ps1
python .\python\benchmark_hotpaths.py --samples 50 --rounds 5 --json-out runs\benchmarks\generalist_refactor.json
.\scripts\benchmark_bridge_observation.ps1 -OutputPath runs\benchmarks\generalist_bridge.json
```

Live gates are stateful and must be run separately from the correct profile/screen:

1. Protected checkpoint loads with 701 actions, `(4297,)`, and timestep 370000.
2. Fresh Generalist training reaches model creation and at least one rollout/checkpoint in a new run directory.
3. Resume loads that compatible model, advances timesteps, and writes only to a second run directory.
4. Generalist evaluation loads the protected model and reaches Adventure gameplay.
5. Progression/startup validation, action masks, fusion chain, reward totals, lifecycle/reset, status, and GUI remain functional.
6. No source checkpoint, adjacent metadata, installed recovery DLL, or profile is unintentionally modified.

## Acceptance matrix

| Gate | Acceptance condition |
| --- | --- |
| Imports/config | No reference to deleted files, flags, schedules, or run modes. |
| Action | `Discrete(701)`, wait `0`, placements/fusions `1..700`; inactive slots masked. |
| Observation | Version `adventure_14slot_identity_v1`, exact shape `(4297,)`. |
| Checkpoint | Protected 370k metadata validation and actual CPU `MaskablePPO.load` pass. |
| Training | Fresh start produces new artifacts; resume advances in an isolated destination. |
| Evaluation | Starts from a validated Adventure boundary and does not mutate source model. |
| Progression | Frontier/replay/unlock transitions remain reducer-driven and inspectable. |
| Fusion/reward | Known recursive identities, tile scope, event deduplication, and exact additive totals pass. |
| GUI | Launches with maintained workflows only; command and shutdown tests pass. |
| Bridge | Build is clean under the repository warning policy and lifecycle harness passes. |
| Documentation | No unsupported runnable path remains; required removal statement is present. |

## Rollback and stop rules

- Version control is the recovery mechanism; do not keep large dead implementations behind flags.
- Never overwrite the protected checkpoint during resume or evaluation.
- Do not force progression when profile, bridge, UI, and wrapper level identities disagree.
- Stop live work and restore the protected installed DLL if the bridge/game becomes unstable.
- A metadata dry run is not live acceptance, and a process start is not a completed training/evaluation gate.
- Record unresolved disagreements or performance drift as pending evidence, not as success.
