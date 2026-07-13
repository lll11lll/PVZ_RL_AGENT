# Fixed/Specialist Removal Inventory

Date: 2026-07-13
Inventory baseline: `0afae6e4683eef6fa9836d143597b28d42332e6c`
Status: pre-removal classification; no implementation is authorized by classification alone

Adventure Generalist is the sole maintained training and evaluation path. Legacy fixed/specialist mode was removed during the repository refactor.

This inventory separates obsolete product paths from code that Adventure Generalist still uses. The protected contract is the 370k Generalist checkpoint: 701 actions, wait action `0`, placement actions `1..700`, decoder `seedslot14x50_plus_wait_v1`, observation version `adventure_14slot_identity_v1`, observation shape `(4297,)`, 14 identity slots, and duplicate startup loadout `SunFlower,SunFlower,Peashooter,Peashooter` / `[1,1,0,0]`.

## 1. Adventure-Generalist required

- `configs/ppo_adventure_generalist_14slot_identity_v1.json`: the sole tracked train/eval configuration.
- `python/pvzrl_adventure_generalist.py`: Generalist training wrapper, startup validation, unlock-aware seed selection, resume handoff, and episode/progression transitions.
- `python/pvzrl_generalist_progression.py`: pure frontier, replay, unlock, and attempt reducer.
- `python/pvzrl_adventure.py`: `run_adventure_eval`, lifecycle/post-win helpers, Adventure status construction, and progression artifacts used by Generalist evaluation and training.
- `python/train_ppo.py` Generalist surfaces:
  - strict Generalist checkpoint validation;
  - Generalist config resolution;
  - environment construction;
  - training/resume source immutability guards;
  - episode metrics, performance accumulation, callbacks, checkpoint writes, and runtime live status;
  - the normal single-model Generalist evaluation branch;
  - metadata dry-run with an actual `MaskablePPO.load`.
- `python/pvzrl_model_metadata.py`: canonical metadata generation, checkpoint/parent lookup, strict validation, and live compatibility projection for current Generalist models.
- `python/pvzrl_action_space.py`: the identity `ActionSpaceSpec`, 701-action constants, identity decoder, policy/bridge index mapping, and identity structural mask.
- `python/pvzrl_observation_layout.py` and `python/pvzrl_seed_inventory.py`: exact `(4297,)` identity observation calculation and features.
- `python/pvzrl_sb3.py`: `PvZSB3Config`, `PvZMaskedPPOEnv`, identity encoding, direct-copy mask projection, coach arbitration, reward accounting, and episode summaries.
- `python/pvzrl_gui.py` plus extracted GUI modules: Generalist train, resume, evaluation, coach, diagnostics, runs, models, logs, and safe process shutdown.

## 2. Shared reusable infrastructure

These areas must be cleaned of obsolete branches, not deleted.

- `python/pvzrl_env.py`: bridge client, current observation, seed/UI automation, action execution, fusion adapter, lifecycle/reset guards, recovery, and runtime state.
- `python/pvzrl_actions.py`: immutable action intents/decisions/results and the per-frame legality cache. Its mode abstraction can become identity-only.
- `python/pvzrl_observation_facts.py`, `python/pvzrl_lifecycle.py`, `python/pvzrl_runtime_state.py`: immutable facts, pure lifecycle classification, and reset/watchdog state.
- `python/pvzrl_fusion.py`, `python/pvzrl_rewards.py`, lane diagnostics, telemetry, human/stream/assisted coach modules, and file tailing.
- `python/train_ppo.py::train`: shared implementation used by Generalist; remove its fixed branches while retaining callback, metrics, resume, checkpoint, TensorBoard, and status behavior.
- `python/train_ppo.py::write_eval_live_status`: currently serves Generalist compatibility-failure publication; retain or migrate that use before deleting fixed-eval construction.
- `python/pvzrl_adventure.py::run_adventure_eval`: retain the evaluator; remove only legacy generic/fixed routing and staged specialist selection.
- `src/PvZRLBridge/` and bridge build/lifecycle/benchmark scripts: no fixed/Level-3 identifiers were found; the bridge is shared.
- Registry presets and slot×cell helpers whose names mention four-slot/fixed require caller-level classification. Generalist uses the duplicate four-card startup loadout and the same `action - 1` slot×cell arithmetic, so reusable logic must be renamed or narrowed before old wrappers are removed.

## 3. Fixed-mode-only obsolete code

### Whole files and tracked artifacts

- `python/backfill_model_metadata.py`
- `python/pvzrl_model_router.py`
- `configs/model_schedule.json`
- `configs/ppo_sunflower_peashooter.json`
- `configs/ppo_sunflower_peashooter_wallnut_cherrybomb.json`
- `configs/ppo_sunflower_peashooter_wallnut_cherrybomb_v2.json`
- `docs/LEVEL3_SPECIALIST_TRAINING_REPORT.md`

### `python/train_ppo.py`

- Fixed default configuration, Level-3 reward defaults, seed-derived specialist family/run-directory helpers.
- `fixed_train`, `fixed_eval`, `level3_specialist`, and generic pre-Generalist `adventure_eval` resolution/dispatch.
- Level-3 seed/action/start-state validation and fixed action-count checks.
- Staged `ModelRouter` evaluation and router dry-run.
- `EvalLog` and the fixed evaluator/reporting constellation: `evaluate`, `run_eval_episode`, fixed evaluation live-status/placeholder/table/diagnostic/summary/output helpers.
- Compatibility wrappers used only by the removed fixed evaluator after caller cleanup.

### Environment/runtime

- `RUN_MODE_FIXED_TRAIN`, `RUN_MODE_LEVEL3_SPECIALIST`, and the public legacy `RUN_MODE_ADVENTURE_EVAL` path.
- `_is_level3_specialist_mode`, `_is_fixed_level_mode`, `level3_specialist_start_state`, `level3_start` reset branches, fixed replay/reset flags, and fixed-only hard-reset fallback behavior.
- Fixed terminal/replay fields and bridge aliases in `pvzrl_runtime_state.py` after proving no Generalist consumer remains.

### Metadata/action/observation

- Fixed legacy metadata inference and `allow_missing_model_metadata` compatibility behavior.
- Fixed/specialist metadata fields generated only to compare against four-slot checkpoints. An extra historical `incompatible_with_4slot_specialist` field in the protected 370k metadata remains tolerated but is no longer generated or enforced.
- `fixed` and experimental `dynamic_14` public action-space modes, decoders, observation layouts, and mask-shift branches. Generalist identity metadata must continue to report `dynamic_seed_slots=true` because that value is part of the protected contract.

### GUI

- Generic fixed train/resume/eval command builders and their dormant variables.
- Legacy generic Adventure-eval command builder and staged/specialist model fields.
- Level-3 command builder, presets, variables, and snapshots.
- Any model discovery rule that accepts a checkpoint without current Generalist metadata.

## 4. CLI and configuration surface to remove

- `--train`, `--eval`, `--level3-train`, `--level3-eval`, `--target-level`
- `--adventure`, `--adventure-eval`
- Fixed/specialist/generic-Adventure `--run-mode` choices
- `--model-schedule`, `--router-dry-run`, `--dry-run-level`, `--dry-run-unlocked-seeds`, `--dry-run-available-seeds`
- `--allow-missing-model-metadata`
- `--experimental-dynamic-seed-slots` and user-selectable non-identity action-space branches
- Fixed-only config fields such as `target_level`, `legacy_max_steps`, `experimental_dynamic_seed_slots`, and generated `incompatible_with_4slot_specialist`

Generalist-owned flags, checkpoint paths, resume paths, Adventure progression limits, identity/loadout fields, reward/fusion/watchdog controls, coach controls, bridge controls, and artifact paths remain.

## 5. Tests and fixtures to delete or replace

- Delete the Level-3-only reset handoff test and fixed-only model schedule/config artifacts.
- Replace `python/test_model_metadata_compatibility.py` with identity-only Generalist checkpoint compatibility coverage.
- Rewrite fixed/dynamic cases in `test_phase2_metadata_contracts.py`, `test_phase3_action_fusion_contracts.py`, `test_refactor_contracts.py`, and `test_refactor_support.py` around the sole identity contract.
- Replace fixed/Level-3/router cases in `test_resolved_config.py` with Generalist train/eval/resume precedence and invalid-obsolete-flag checks.
- Remove fixed `EvalLog`/summary assertions from `test_fusion_reward_policy.py` while retaining reward composition and Generalist episode aggregation coverage.
- Reduce `test_gui_commands.py` snapshots to Generalist fresh train, resume, evaluation, and maintained coach modes.
- Remove fixed benchmark cases and fixed model fixtures; retain deterministic identity mask/vector/reward/fusion/status hashes.
- Remove obsolete namespace placeholders (`level3_train`, `level3_eval`, generic `adventure_eval`) from Generalist tests.

Lifecycle fixtures that happen to use level number 3 are not automatically fixed-mode fixtures. Keep them when they model generic Adventure reward/unlock/replay transitions; rename contexts only when a specialist assumption is actually present.

## 6. Documentation to update

- Rewrite `AGENTS.md`, `docs/REFACTOR_PLAN.md`, and `docs/REFACTOR_REPORT.md` around the sole maintained path and include the required policy sentence verbatim.
- Remove fixed/specialist command forms, checkpoint claims, compatibility tables, pending Level-3 gates, model schedules, and fixed live-status ownership.
- Delete the dedicated Level-3 report.
- Audit `docs/CONFIGURATION.md`, `docs/ADVENTURE_GENERALIST.md`, `docs/RLPVZ_TECHNICAL_WIKI.md`, PPO guides, fusion/watchdog guide, and older roadmaps so none presents fixed mode as supported. Historical refactor evidence may say that the path was removed, but must not retain runnable instructions.

## 7. Adventure-Generalist risks and stop rules

- Never change the protected 701/4297 decoder or observation contract while deleting 201/357 compatibility.
- Keep wait action `0`; do not inherit the removed `dynamic_14` wait-at-700 shift.
- Keep identity masks index-preserving and keep 14-slot duplicate identity semantics.
- Preserve strict model family, seed/loadout order, plant IDs, metadata lookup, and actual model-load checks.
- Preserve source-checkpoint immutability and write resumed artifacts only to the requested new run directory.
- Preserve `run_adventure_eval`, progression transitions, post-win evidence, reward composition, fusion/recursive fusion, coach arbitration, watchdog diagnostics, and atomic status publication.
- Treat the 2026-07-13 current-source live run as a functional Generalist proof, not a zero-disagreement proof: it completed 408 steps with exit code 0 and no bridge errors, but recorded two illegal/mask-bridge fusion disagreements (`fusion_no_effect`, `source_not_found`).
- After every removal slice, run focused tests plus the full bridge-free suite. Before declaring completion, load the protected 370k model and exercise Generalist fresh training, resume, evaluation, progression, GUI, masks, fusion, and rewards.

## 8. Incremental removal order

1. Remove tracked fixed configs, staged model routing, Level-3 preflight, obsolete public CLI routes, and their tests.
2. Remove dead fixed GUI state/commands and narrow model discovery to Generalist metadata.
3. Reduce train/eval/config/status/callback code to Generalist while retaining shared metrics and compatibility failure publication.
4. Reduce action, observation, metadata, environment, SB3, coach, benchmark, and fixtures to the identity contract.
5. Rewrite documentation and run repository-wide obsolete-reference/import checks.
6. Complete bridge-free checkpoint/train/resume/eval tests, then perform protected live Generalist and GUI validation.
