# PvZRL Adventure Generalist Refactor Report

Status: repository-wide removal and the final bridge-free/bridge verification gates completed on 2026-07-13. A new post-removal live gameplay rollout was not run; that limitation is recorded below.

Adventure Generalist is the sole maintained training and evaluation path. Legacy fixed/specialist mode was removed during the repository refactor.

Streamer Mode V1 was added later as an overlay on this settled Generalist architecture. It does not reopen a fixed/specialist path or alter the protected 701-action / 4,297-observation contract. The 2026-07-13 counts and live evidence below remain historical refactor evidence; they do not include the later Streamer implementation.

## Executive result

PvZRL now has one maintained policy surface:

| Field | Current contract |
| --- | --- |
| Run modes | `adventure_generalist_14slot_train`, `adventure_generalist_14slot_eval` |
| Model family | `ppo_adventure_generalist_14slot_identity_v1` |
| Action mode | `adventure_14slot_identity` |
| Action space | 701 actions; wait `0`; placement/fusion `1..700` |
| Decoder | `seedslot14x50_plus_wait_v1` |
| Observation | `adventure_14slot_identity_v1`, shape `(4297,)` |
| Seed capacity | 14 identity slots on a 5x10 board |
| Protected model | `runs/ppo_adventure_generalist_14slot_identity_v1_20260627_172727/checkpoints/ppo_pvz_370000_steps.zip` |
| Protected timestep | 370000 |

Shared environment, bridge, reset, masks, observation facts, fusion, rewards, progression, coaches, telemetry, diagnostics, and Tk infrastructure were retained. Product-specific dispatch, layouts, adapters, schedules, controls, fixtures, and documentation were removed instead of hidden behind deprecated switches.

## Post-refactor Streamer V1 ownership update (2026-08-08)

Streamer V1 reuses the maintained architecture rather than introducing a second training product:

| Concern | Current owner and boundary |
| --- | --- |
| Train/evaluate loop | `pvzrl_streamer.py` coordinates existing `train_ppo.train` and `run_adventure_eval`; `STREAM_TRAIN`/`EVALUATE` are phases, not new run modes. |
| Twitch input | `pvzrl_twitch.py` implements read-only EventSub WebSocket `channel.chat.message`; `pvzrl_streamer_source.py` keeps Twitch and deterministic mock sources interchangeable. |
| Commands/actions | `pvzrl_stream_commands.py` owns strict parsing, bounded FIFO, TTL, cadence, dedupe, and phase generation. `pvzrl_stream_actions.py` resolves through the current canonical action-decision cache and mask. |
| PPO/BC | `pvzrl_streamer_ppo.py` excludes viewer transitions/log probabilities from policy rollouts and cuts GAE at interventions. `pvzrl_demonstrations.py` owns bounded atomic observation/mask/action records and masked BC uses the existing policy/optimizer. |
| Environment/bridge | `pvzrl_sb3.py` classifies the actual canonical execution. The existing `PvZGymEnv`, action/fusion code, reward composition, and C# protocol remain authoritative. |
| Checkpoints | The configured 500k Generalist source is immutable BASELINE. CURRENT and BEST use immutable hash-addressed generations plus atomic role records and repairable aliases; protected BEST promotes by win rate then average reward only for same-start-level protocol-compatible evaluations, retaining ties. |
| Progression | Baseline evaluation, cycle training, and current evaluation pass expected/next Adventure levels sequentially and retain strict live identity validation. This is not profile-restored same-level evaluation. |
| Status/logging | Existing atomic live status receives compact Streamer/queue/Twitch/BC/evaluation/level fields. Streamer JSONL excludes raw chat identity and full observations; the bounded demonstration `.npz` separately stores learning inputs. |

The existing local mock crowd-coach/voting path remains a separate legacy coach tool. Streamer V1 is FIFO with no voting and refuses simultaneous legacy action overrides.

Bridge-free verification added focused EventSub, parser/FIFO, action/fusion resolution, intervention-aware PPO/GAE, BC persistence, cycle/checkpoint, resume, evaluation-isolation, and phase-race tests. The local June 21 500k baseline was actual-loaded on CPU during Streamer implementation with 701 actions, observation `(4297,)`, and `num_timesteps=500000`. That is a compatibility/load proof, not live Unity, Twitch delivery, or completed episode evidence.

No full six-hour credentialed Twitch/game endurance result is claimed by this source update. `docs/STREAMER_MODE.md` provides both the bridge-free six-hour soak and exact live six-hour command, the artifacts/counters to inspect, and the sequential-evaluation limitation. OBS, game/process restart supervision, public services, and the future autonomous control arm remain out of scope.

## Scope inventory

The deletion classification was completed before implementation and is retained in `docs/FIXED_MODE_REMOVAL_INVENTORY.md`.

The inventory separated:

1. Adventure-Generalist-required code;
2. shared reusable infrastructure;
3. obsolete product-only code;
4. unclear dependencies that required call-site, serialization, and checkpoint investigation.

This prevented broad deletion of shared code merely because it lived in an older module or had a generic filename.

## Implemented changes

### Files and routing

Removed unused schedule/router/backfill and obsolete example configuration files. Train/eval dispatch now recognizes only Generalist train and eval. Unsupported mode strings fail early instead of falling through to another behavior.

`python/pvzrl_adventure.py` remains because the Generalist evaluator consumes its lifecycle, attempt, transition, timeout, and live-status infrastructure. Its generic name does not represent a third supported mode.

### Configuration and CLI

- Reduced typed run-mode configuration to Generalist train/eval.
- Removed level-target, alternate-action-layout, metadata-bypass, and model-schedule branches.
- Kept one resolver and the precedence rule `explicit CLI > JSON > Generalist mode default > global default`.
- Kept fresh/resume separation: resume must load a compatible source and write to a new destination.
- The tracked Generalist JSON remains the canonical default.

### Action and observation contracts

- `pvzrl_action_space.py` now defines one 701-action identity layout.
- Removed alternate decoder math and policy/bridge action-ID conversion.
- Removed obsolete action intent/result fields whose only purpose was cross-layout translation.
- Pinned validation to 14 slots and 5x10 geometry.
- Prevented bridge active-slot counts from resizing or reclassifying the permanent policy mask.
- Kept `dynamic_seed_slots=true` as protected serialized inventory-capability metadata, not as another decoder.
- Preserved the exact 4,297-feature identity observation.

### Environment and SB3

- `PvZEnvConfig`, `PvZGymEnv`, and `PvZSB3Config` accept only Generalist train/eval.
- Defaults now use `SunFlower,SunFlower,Peashooter,Peashooter`, IDs `[1,1,0,0]`, 14 slots, and a 5x10 board.
- Removed level-specific startup/reset and post-win handling.
- Preserved shared action cache, masks, bridge execution, lifecycle, watchdog, coach arbitration, episode accounting, fusion, and reward behavior.
- SB3 masks copy the sole identity-aligned raw mask directly; action IDs remain unchanged at the bridge boundary.

### Metadata and protected checkpoint

- Metadata compatibility now protects only the maintained Generalist family and contract.
- Removed cross-family routing and obsolete checkpoint fixtures.
- Retained fail-fast checks for action count/mode, decoder, observation version/shape, slot capacity/identity, wait/placement range, and board geometry.
- Actual `MaskablePPO.load` remains the final bridge-free proof.

### GUI and coaches

- Removed obsolete GUI controls, presets, command builders, and model filters.
- The dashboard now presents Generalist fresh train, resume, evaluation, runs/models, coach, and diagnostics workflows.
- Preserved local human-coach, mock crowd-coach, assisted-coach, intervention logging, dry-run/apply safety, file-tail handling, and bounded child-process shutdown.
- Human coach continues to take precedence over crowd coach when both are enabled.

### Tests and fixtures

- Removed tests and fixture sections dedicated solely to deleted layouts/checkpoints.
- Added negative tests that reject unsupported run/action modes.
- Added exact Generalist environment and SB3 contract tests.
- Kept progression, lifecycle, timeout, fusion, reward, mask, observation, telemetry, coach, GUI, bridge, and registry regressions.
- Updated deterministic hashes only for reviewed removals such as deleted episode/status fields.

### Documentation

Updated the root operating guide, configuration, Generalist architecture, bridge, coach, learning, readiness, roadmap, README, and refactor documents. Runnable unsupported commands and compatibility promises were removed. The deliberate inventory remains the historical deletion record.

## Focused automated evidence

The following focused results were produced while the working tree was being integrated:

| Slice | Result |
| --- | --- |
| Generalist action/fusion refactor | 46 focused tests passed |
| Environment/lifecycle refactor | 41 environment/lifecycle tests passed |
| Reward/fusion/observation around environment | 39 tests plus 10 subtests passed |
| Action/fusion pipeline after environment collapse | 32 tests passed |
| SB3, masks, fusion, rewards, lifecycle, and observation | 75 tests passed |
| New SB3 Generalist configuration contract | 13 tests included in the passing SB3 slice |
| Generalist benchmark smoke | completed successfully |

The benchmark smoke retained:

- action count `701`;
- observation vector size `4297`;
- identity mask and vector contract outputs;
- current tactical-mask, fusion, reward, facts, status, and GUI-poll paths.

Final settled-tree evidence is recorded below; the focused results remain useful slice-level provenance.

## Protected checkpoint evidence

After the removal, the current protected checkpoint passed metadata validation and actual CPU `MaskablePPO.load` with:

```text
actions=701
observation=(4297,)
num_timesteps=370000
decoder=seedslot14x50_plus_wait_v1
observation_version=adventure_14slot_identity_v1
```

The source checkpoint hash recorded for protected live work was:

```text
DFD2B800D5B4BB24772BE868E4CD31A320E07AEC14C2D67D2CA509EAB49C0B5B
```

The adjacent metadata hash was:

```text
378CE2B592570AE7A5EFE96F7E85AAAB5EAA5E563B26155308E659E54484AD8F
```

The post-removal hashes remained exactly the values above.

## Live evidence retained from the refactor

Two live Generalist observations are important and must not be conflated:

1. A clean-boundary protected 370k evaluation (`generalist_eval_recursive_fix_clean_20260712_210447`) started from Level 6 seed selection, selected the duplicate initial loadout, and completed a 547-step classified loss at wave 9. It recorded 45 fusion attempts, 45 successes, no illegal action, bridge/reset error, mask/bridge disagreement, or rejection reason.
2. The later current-source validation (`runs/manual_phase8/current_source_live_20260713_162551/generalist_eval_current_source`) completed with exit zero after 408 steps, a Level-6 loss at wave 8, and no bridge/reset error. It recorded 30 fusion attempts, 28 successes, and two rejected/no-effect outcomes (`fusion_no_effect`, `source_not_found`), producing two illegal-action/mask-disagreement records.

The later run proves the current source can load and execute the protected model end to end, but it is not a zero-disagreement acceptance run. The two rejected outcomes remain explicit evidence for follow-up; they must not be summarized away as full live fusion acceptance.

The installed game bridge was restored after protected live work to recovery hash:

```text
5643EC37984762AB72FEA3C50E87FCC466B905C5CF046C307FDAA6E0CE42A0F4
```

The protected profile hash remained:

```text
3DD53960383292C4CCD31C3112CB663252A91956E4E21AC5F67BFC6186CD7BBC
```

No game process or bridge listener was left running after that validation.

## Final settled-tree verification

| Gate | Result |
| --- | --- |
| Dependency readiness | `train_ppo.py --check-deps` passed. |
| Python compile/full suite | `compileall` passed; `308 passed, 10 subtests passed`. |
| Retained executable compatibility scripts | All 9 passed. |
| Fresh/resume/eval entrypoint starts | 3 bridge-free integration tests passed using the real parser/config and protected metadata. Fresh reached `learn(reset_num_timesteps=True)`; resume loaded the 370k source and reached `learn(reset_num_timesteps=False)` in an isolated destination; evaluation validated metadata and reached `run_adventure_eval`. |
| Source immutability | Resume integration preserved source hash and timestamp; the protected model and metadata SHA-256 values remained unchanged. |
| Protected checkpoint | Actual CPU `MaskablePPO.load` passed with 701 actions, `(4297,)`, timestep 370000, and the maintained decoder/observation versions. |
| Progression/masks/fusion/rewards | Full and focused suites passed, including recursive fusion and exact reward/contract snapshots. |
| Registry/bridge | Generated registry check passed; bridge built with no warnings; lifecycle harness passed 58,553 checks. |
| GUI | Withdrawn Tk smoke passed with `Train`, `Eval`, `Coach`, `Diagnostics`, and `Runs/Models`; only Generalist train/eval command builders are exposed. |
| Repository scan | No active deleted-mode imports, dispatch, flags, configs, schedules, GUI controls, status branches, or runnable documentation remain. Old strings occur only in deliberate rejection/filter tests, the removal inventory, and ordinary Adventure level-number fixtures. |

This verification intentionally did not start or mutate the installed game. The two live runs in the previous section predate the final removal. A new live rollout, full live train/resume artifact cycle, and full live transition matrix remain optional workstation acceptance work when the game is at a validated Adventure boundary; they are not represented as completed here.

## Acceptance status

| Gate | Current report status |
| --- | --- |
| Inventory/classification | Complete |
| Source/config/GUI/test removal | Complete; final repository scan passed |
| 701 action and decoder contract | Pass |
| 4,297 observation contract | Pass |
| Masks/fusion/rewards | Automated pass; pre-removal live run retains two disagreement records |
| Progression/lifecycle | Automated pass; new post-removal live transition matrix not run |
| Protected 370k load | Pass on final source; hashes unchanged |
| Fresh train/resume/eval start | Bridge-free entrypoint pass; live rollout cycle not run |
| GUI | Withdrawn construction/workflow smoke pass |
| Bridge | Build and 58,553-check lifecycle harness pass |
| Full pytest | `308 passed, 10 subtests passed` |
| Documentation | Rewritten; tracked and ignored-workspace obsolete-reference audit passed |

The repository refactor is accepted for the requested source-removal and bridge-free start gates. Live gameplay evidence remains explicitly bounded to the pre-removal runs above; no post-removal live completion claim is made.
