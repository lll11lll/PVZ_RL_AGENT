# PvZRL Refactor Execution Report

Status: automated/local-Tk validation and partial real-game Phase 8 validation complete; overall goal acceptance remains incomplete
Report started: 2026-07-11
Baseline commit: `0cbc4d90ff68b31f5a0fed92d7508243c1d0293f`

## Executive status

The repository-wide implementation, every bridge-free/local-Tk validation phase, and a bounded final-source real-game gate are complete. Separate compatibility, concurrency, refactoring-quality, performance, and final-diff reviews were run, and their high-confidence findings were resolved where the environment allowed.

The overall goal is not labeled complete. The final bridge was exercised in the real game for startup, menu/seed/gameplay entry, observation and action semantics, repeated loss/retry reset, fusion, and a corrected protected Generalist evaluation. Short fresh/resumed fixed training, fixed Level-3 evaluation, win/timeout/reward/unlock/replay/advancement traces, live game-backed GUI operation, and measured live latency remain outstanding. The cross-run approximately-5% performance gate also remains unresolved, and first-party runtime code increased from 48,127 to 54,821 physical lines rather than achieving a net reduction.

## Baseline repository state

- Branch: `main`, tracking `origin/main`.
- Baseline working tree: clean.
- Baseline revision: `0cbc4d90ff68b31f5a0fed92d7508243c1d0293f`.
- Platform: Windows/PowerShell.
- Python: 3.14.0 at `C:\Python314\python.exe`.
- NumPy: 2.4.2.
- Gymnasium: 1.2.3.
- Stable-Baselines3 / sb3-contrib: 2.8.0.
- Protected models load on CPU.
- Bridge compiler: Visual Studio Roslyn C# 4.6.0 targeting installed .NET 6.0.6 reference assemblies.
- .NET SDK: unavailable; the repository's Roslyn build script is authoritative in this environment.

## Inspected systems

- All 15 non-test Python modules, `BridgeMod.cs`, `build_bridge.ps1`, configs, README, and relevant architecture/Adventure/bridge/coach documents.
- All nine executable Python regression scripts and pytest collection.
- Protected and fixed model metadata plus actual checkpoint loads.
- Bridge request queue, timeout race, server/client shutdown, seed cache, observation construction, legal action generation, fusion execution, and warning sites.
- Base environment stepping, masking, rewards, fusion accounting, lifecycle/reset, corruption diagnostics, and registry reads.
- SB3 observation encoding, action translation, coach arbitration, watchdog logging, and episode summaries.
- Adventure/generalist progression, status building, curriculum, replay, and startup identity.
- GUI command construction, subprocess/log lifecycle, live-status polling/health/rendering, and coach queue surfaces.

## Baseline code statistics

The baseline uses the same exclusions as the supplied audit: first-party non-test Python modules, `src/PvZRLBridge/BridgeMod.cs`, and `scripts/build_bridge.ps1`; no tests, docs, models, runs, caches, generated output, or build artifacts.

| Measure | Baseline |
| --- | ---: |
| Runtime files | 17 |
| Runtime lines | 48,127 |
| Python test files | 9 |
| Python test lines | 6,154 |
| Pytest-discovered tests | 17 tests + 7 subtests |
| Script regression runners | 9 |

Largest verified units:

- `python/pvzrl_env.py`: 11,658 lines; `_reset_state_machine()` 972; `step()` 486.
- `src/PvZRLBridge/BridgeMod.cs`: 11,488 lines.
- `python/train_ppo.py`: 4,610 lines; `build_config()` 489.
- `python/pvzrl_gui.py`: 4,448 lines; `_render()` 260.
- `python/pvzrl_adventure.py`: `run_adventure_eval()` 588; `build_live_status()` 539.
- `python/pvzrl_sb3.py`: `step()` 703.

Final added/removed/net runtime lines, test lines, documentation lines, functions/classes removed, duplicate families consolidated, wrappers remaining, files split/deleted, and moved-versus-eliminated code are recorded in the Phase 8 statistics below.

## Baseline commands and results

| Command | Result |
| --- | --- |
| `python .\python\train_ppo.py --check-deps` | PASS: PPO readiness yes |
| `python -m compileall -q python` | PASS |
| `python -m pytest -q` | PASS: 17 tests and 7 subtests in 1.22s |
| All nine `python/test_*.py` script runners | PASS |
| `.\scripts\build_bridge.ps1` | PASS with three warnings |
| June 27 protected 370k metadata dry-run | PASS, 701 actions |
| June 27 protected 370k `MaskablePPO.load` | PASS, observation `(4297,)`, timestep 370000 |
| June 21 370k metadata/load control | PASS, same action/observation contract |
| Fixed four-slot utility metadata dry-run | PASS, 201 actions |
| Fixed utility `MaskablePPO.load` | PASS, observation `(357,)`, timestep 350368 |

### Bridge warnings before refactoring

1. `CS8604` at `BridgeMod.cs:5735`: nullable cached `CardUI` passed to `BuildSeedSlotDto`.
2. `CS8604` at `BridgeMod.cs:5748`: same unsafe retry path.
3. `CS0649` at `BridgeMod.cs:55`: `_sunSpawnCompensationApplyCount` is never assigned.

These are not suppressed. The nullable paths will be corrected. The serialized sun-compensation counter is externally consumed, so its compatibility key must remain while its deprecated constant semantics are made explicit.

## Pre-existing failures and environmental limits

No automated Python or bridge-build failures were present at baseline.

Not yet verified live:

- Game/bridge connection and current source DLL loading.
- Real observation, placement, legal/illegal/self/recursive fusion, reset, and seed selection.
- Startup, win, loss, timeout, reward/unlock, replay, and Adventure advancement transitions.
- Short fixed/generalist rollout, resumed training, and fixed/Adventure evaluation.
- Live bridge/Python step latency and rollout SPS.
- Interactive GUI launch/stop/render smoke.

The baseline source DLL hash differs from the installed May bridge DLL. A live gate must install the newly built DLL deliberately and preserve the previous DLL as a recovery point before making runtime claims.

## Confirmed defects

### Lifecycle and concurrency

- A timed-out bridge request remains in `_pending` and may mutate a later episode on Unity's main thread.
- Timeout versus dispatch has no atomic ownership state, so a client can receive timeout while the request executes.
- `StopServer()` does not reject enqueue atomically, close tracked clients, complete pending requests, prevent post-stop dispatch, or join the server thread.
- Client `ReadLine()` can remain blocked until its socket is externally closed.

### Seed cache and command files

- Cached seed-slot retry paths can pass null/stale cards and do not verify that the compacted cache entry's `SlotIndex` matches the requested slot.
- Stream JSONL tailing consumes malformed/incomplete trailing records and advances to EOF; the human tailer correctly retains them. Both tailers only detect size-shrink rotation.

### Configuration and metadata

- Four argparse defaults incorrectly take precedence over JSON without explicit CLI input.
- `model_schedule.json` points to missing root-relative models.
- Registry metadata is reparsed repeatedly and conflicts with hard-coded Python/GUI/C# tables.

### Rewards and diagnostics

- Four delayed CherryBomb lane-diagnostic fields always serialize as zero because `cherry_delayed_diag` is not in that function's locals.
- Mask generation and fusion diagnostics repeat across MaskablePPO, coach arbitration, base execution safeguards, lane diagnostics, and live status.
- Reward/lane/mask/fusion code repeatedly scans the same observation lists.

### GUI and hot-path I/O

- Window close bypasses the existing bounded terminate/kill flow and destroys Tk immediately.
- Scheduled Tk callback IDs are not retained/cancelled.
- Log draining is unbounded per tick; an unbounded queue plus per-line filtered-history rebuild can freeze the UI under bursts.
- Unchanged status is stat/read/parsed/render-normalized every second.
- Very old blocked payloads remain `BLOCKED_*` instead of becoming stale/dead and suppress writer warnings.
- Fixed training live status and configured watchdog paths perform expensive serialization/persistence at step frequency.

### Fusion audit correction

The current bridge has no active `CheckMix` probe. Fusion uses mouse/reflection attempts. A method attempt without observed mutation can be provisionally successful, charge sun/start cooldown, and fail only in the later postcondition check. This is isolated from Phase 1 and requires dedicated fixtures before correction.

## Verified duplication and hot-path candidates

- Plant registry: repeated `pvzrl_env` JSON reads, separate seed-inventory cache, fusion name/ID tables, GUI loadouts, SB3/C# fallback costs.
- Fusion: independent recipe and compatibility tables plus model/scripted/human/stream branches.
- Masks: wrapper mask, coach masks, base step revalidation, lane diagnostics, and status diagnostics.
- Observation facts: repeated plant/zombie/lane/slot/occupancy scans across masks, rewards, diagnostics, fusion, and encoding.
- Metrics/status: duplicated fields, coercion, aggregation, atomic writers, aliases, and schema building across environment, SB3, trainer, Adventure, coach, and GUI.
- Configuration: overlapping argparse, JSON, GUI, dataclass, bridge, and metadata defaults.
- Bridge: repeated occupancy scans inside slot/cell loops, per-row zombie filtering/sorting, repeated seed/UI/object scans, and seed cost/cooldown collection rebuilding.

## Performance benchmark record

Committed harness: `python/benchmark_hotpaths.py`. The durable baseline summary is `python/fixtures/refactor_contracts/phase0_benchmark_baseline.json`; the command also writes the full machine-local result to ignored `runs/benchmarks/phase0_python_baseline.json`. The synthetic dense fixture uses a 5x10 board with 25 plants, 30 zombies, five lanes, and ready/affordable slots; fixed mode uses 4 slots/201 actions and identity mode 14 slots/701 actions. Its representative live payload is about 42.6 KB. Samples use `perf_counter_ns`, warmups, disabled GC during sampling, and nearest-rank p95.

| Operation | Median ms | p95 ms |
| --- | ---: | ---: |
| Action mask, fixed four-slot dense | 1.860 | 3.521 |
| Action mask, identity 14-slot dense | 11.253 | 12.812 |
| Action mask, identity low-sun | 2.637 | 3.292 |
| Action mask, identity all-cooldown | 1.513 | 1.557 |
| Action mask, identity fusion disabled | 8.677 | 9.120 |
| Observation encode, fixed sparse | 0.206 | 0.220 |
| Observation encode, identity sparse | 0.538 | 0.555 |
| Reward breakdown, dense | 0.272 | 0.286 |
| Fusion candidate scan, dense | 5.810 | 6.119 |
| Adventure live-status build including mask | 21.821 | 22.579 |
| Live-status JSON serialization | 0.176 | 0.191 |
| Atomic live-status flush/fsync/replace | 4.249 | 4.581 |
| Unchanged GUI stat/read/parse | 0.408 | 0.620 |
| Candidate `(mtime_ns, size)` signature only | 0.029 | 0.033 |

This is deliberately dense scan pressure, not live rollout evidence. It excludes bridge latency, Unity observation construction, PPO inference, full Tk widget rendering, Python environment-step latency, and SPS. Before/after claims require this exact committed harness, fixture hashes, samples, and rounds.

## Phase execution log

### Phase 0 - baseline and behavior locks

Status: complete.

Completed:

- Read both supplied texts fully.
- Verified repository instructions, docs, configs, build path, architecture, size, and warnings.
- Ran the complete current bridge-free automated baseline.
- Validated and loaded both 370k models and the fixed specialist control.
- Recorded immutable contracts, audit deviations, defects, risks, rollback boundaries, and initial performance measurements.
- Made the two refactor deliverables trackable without exposing the rest of ignored `docs/`.
- Added portable synthetic observation, model-contract, and benchmark-baseline fixtures.
- Added nine focused compatibility tests covering action encodings, exact observation vectors, masks, fusion, reward values, serialized schemas, bridge DTOs, and protected model metadata facts.
- Added the repeatable Python hot-path microbenchmark and recorded its exact fixture hashes, method, limitations, median, and p95 values.
- Reran the full gate: dependency check and compileall passed; pytest passed 26 tests plus 7 subtests; all nine legacy script runners passed; the bridge built with the three recorded baseline warnings; protected generalist and fixed control checkpoint loads retained `(701, (4297,), 370000)` and `(201, (357,), 350368)` respectively.
- Completed an independent diff review. Its three portability/robustness findings were corrected, and the re-review found the Phase 0 boundary safe to commit.

Rollback boundary: local commit `773c228` (`test: lock PvZRL refactor baseline`).

### Phase 1 - confirmed defects and lifecycle safety

Status: complete.

Implemented:

- Replaced the bridge's unowned concurrent queue entries with request IDs, monotonic deadlines, and atomic `Queued -> Dispatching -> Completed` or `Queued -> Canceled` ownership. Timeout cancellation can win only before Unity dispatch; once dispatch owns a request, the client receives the real completion rather than an early timeout.
- Made bridge shutdown atomically stop enqueue and client registration, cancel every queued request with `server_stopping`, stop the listener, close active client sockets to unblock reads, prevent later `OnUpdate` dispatch, bounded-join the listener thread, and bounded-wait for client-worker drain with actionable warnings.
- Added a deterministic standalone C# lifecycle harness covering deadlines, expired step/fusion/configure/reset commands, timeout/dispatch and stop/dispatch races, queued cancellation, rejection after stop, and bounded client drain.
- Rebuilt seed-slot placement lookup by explicit `SlotIndex`, rejected missing/stale/null cached cards after one authoritative probe rebuild, and removed both nullable `CardUI` compiler paths.
- Retained `sunSpawnCompensationApplyCount` as an explicit deprecated constant-zero compatibility field, eliminating the dead-field warning without changing its serialized value.
- Added one bounded binary incremental line tailer shared by human and mock-stream coach sources. It separates committed and read offsets, retains incomplete UTF-8/CRLF records, resumes them exactly once, skips complete malformed records with diagnostics, detects replacement/truncation/same-inode rewrite, preserves clear/start-at-end behavior, and bounds reads and oversized pending records.
- Unified GUI explicit-stop and window-close process handling. Close cancels tracked Tk callbacks, requests terminate, waits without blocking Tk, escalates after the grace period, and enforces one hard deadline. At that deadline it makes a final kill attempt, records a still-alive failure, then completes bounded thread/log cleanup and destroys the Tk root instead of waiting forever.
- Bounded the GUI producer queue, per-tick drain count/time, retained log lines/chars, and visible drop accounting. Unchanged live-status files now reuse parsed payloads by file signature, while missing/malformed/replaced files recover normally.
- Made live-status age authoritative over old `blocked_reason` values, so stale/dead writers can no longer remain indefinitely `BLOCKED_*`.
- Passed CherryBomb delayed event diagnostics explicitly from reward calculation into lane diagnostics; delayed kills, zero-kill expiry, buckethead credit, and conehead credit no longer serialize as unconditional zero.

Proven deletions:

- Removed the disabled commented Level-3 start gate and unused environment helpers `_is_active_gameplay`, `_is_confirmed_terminal_or_transition`, `_playable_board_has_episode_residue`, `_wait_for_reset_playable`, and `_zombie_proximity_penalty`.
- Removed unused Adventure post-win helpers `clear_post_win_screens`, `_wait_for_unlock_or_reward_screen`, and `_wait_for_next_seed_selection_or_menu`, plus generalist `_set_hard_blocked`.
- Removed unused GUI `row_panel_lines`, human-coach private compatibility aliases `_build_env_fusion_probe` and `_seed_slot_ready_for_use`, model-metadata `compatibility_or_raise`, and trainer helpers `model_config_candidates`, `load_model_run_config`, and `plant_types_from_model_metadata`.
- Each deletion had definition-only repository, import, call-site, configuration, serialized-key, documentation, and checkpoint/metadata searches before removal. Compatibility wrappers still used by reset, diagnostics, router, local mock-stream, CLI, or serialized consumers remain.

Current gate results:

- `python -m compileall -q python`: PASS.
- `python -m pytest -q`: PASS, 47 tests plus 7 subtests after the final focused additions.
- All nine retained script regression entry points: PASS.
- `scripts/test_bridge_lifecycle.ps1`: PASS, 7,051 deterministic checks and a zero-warning bridge build.
- Headless Tk construction/close smoke: PASS.
- Real child-process GUI close smoke: PASS; the 60-second child was terminated, reaped, and the dashboard destroyed without an orphan.
- Fixed and protected metadata dry-runs: PASS with 201/701 actions and no warnings.
- Actual protected generalist and fixed control loads: PASS with `(701, (4297,), 370000)` and `(201, (357,), 350368)`.
- Independent review: PASS with no blocking findings. It directly checked request ownership, shutdown, seed caching, DTO/wire compatibility, tail offsets/anchors, GUI lifecycle, Cherry propagation, and deletion evidence.

Residual Phase 1 limitation: the deterministic C# harness exercises the request queue/state and active-client registry primitives but not a complete listener with a worker blocked in `ReadLine()`. The reviewed production path closes every socket that wins atomic registration, which is the operation that unblocks that read; a live bridge connection remains part of the final environment gate.

Rollback boundary: local commit `55426dc` (`fix: harden PvZRL lifecycle boundaries`).

### Phase 2 - canonical metadata and resolved configuration

Status: complete.

Implemented:

- Added `pvzrl_registry.py` as the one parsed, validated, deeply immutable Python plant registry. Its 12 definitions expose canonical names, aliases, IDs, fallback cost/cooldown, category/role, unlock/training/fusion/GUI metadata containers, training eligibility, and explicit bridge-fallback eligibility. Default-path lookups use a pre-resolved cache key so the cache itself does not add filesystem resolution to observation encoding.
- Retained the existing `pvzrl_env` registry APIs as defensive-copy forwarding adapters. Seed-list resolution, display names, Adventure identity features, GUI profiles, Level 3 IDs/loadout, and base-plant fusion identities now derive from indexed canonical definitions and named GUI presets. Fusion-result IDs remain an explicit separate overlay until the authoritative fusion-recipe migration in Phase 3.
- Added deterministic bridge generation from `configs/plant_registry.json`. `build_bridge.ps1` regenerates and compiles `GeneratedPlantRegistry.cs`; the bridge still reads cached/live `CardUI` cost first and uses generated values only for the two pre-existing limited fallbacks. The manually synchronized C# cost switch is gone.
- Added a deeply immutable `ResolvedRunConfig` with typed optimization, environment, seed/action, Adventure, fusion, coach, diagnostics, artifacts, bridge, model-contract, and reward sections plus immutable value provenance. `build_config()` remains the flat-dictionary compatibility adapter, so existing consumers and `resolved_config.json` retain their keys and JSON value shapes.
- Replaced scattered scalar/enable precedence branches with `ConfigResolver`: `explicit CLI > JSON > mode default > global default`. The four Adventure parser defaults that previously masked JSON now use an unset state; explicit run-mode aliases resolve before JSON; runtime dispatch uses the resolved mode; stream mode/platform aliases stay coherent; JSON/CLI `plant_types` must match canonical seed-slot order; and tracked metadata reasons survive resolution. Quick-wait, Level 3, and Adventure Generalist defaults are mode defaults rather than accidental parser values.
- Added actionable warnings for recognized ignored legacy fields. `enable_fusion_diagnostics` is identified as a no-op; unused `proximity_penalty` warns in both top-level and nested reward shapes while remaining in resolved output during the compatibility window. `docs/CONFIGURATION.md` documents falsey values, provenance, aliases, legacy warnings, mode defaults, GUI launch behavior, and the adapter window.
- Extended model compatibility checks across metadata version, identity-slot semantics, exact loaded observation shape, declared observation shape (including malformed values), wait action, placement range, rows, columns, and cells per slot. Metadata and the SB3 wrapper now share one pure observation-layout calculator; runtime environment metadata still records the actual Gym shape as final proof. Existing metadata without the additive shape field remains compatible when the loaded model proves the shape.
- Repaired both root-relative paths in `configs/model_schedule.json`, corrected the utility artifact/run choice, and labeled the tracked file as a locally validated example whose model artifacts are intentionally gitignored. Both router stages now select, load, and validate from the repository root.

Compatibility evidence:

- The generated registry is byte-deterministic and hash-linked to its JSON source; every canonical entry appears once, only the original SunFlower/Peashooter bridge fallbacks are enabled, and no manual C# fallback switch remains.
- The full pytest gate passes 111 tests plus 7 subtests. The reviewer's five-file registry/configuration/metadata contract gate passes 72 tests. Coverage includes deep immutability, parse-once behavior, named GUI presets, fusion-overlay separation, generator staleness, CardUI-first order, falsey and cross-mode precedence, dispatch routing, aliases, typed provenance/round trips, legacy warnings, plant-slot alignment, optional schedule portability, and every enforced model axis.
- All nine retained standalone regression entry points pass after the final resolver and dispatch changes.
- Both repaired schedule stages pass end-to-end `--router-dry-run`, including actual model load and `(201, (357,))` compatibility.
- The protected June 27 checkpoint passes `--metadata-dry-run` with `(701, (4297,), 370000)`; the fixed control passes with `(201, (357,), 350368)`. Neither checkpoint or adjacent metadata file was modified.
- The bridge builds with zero warnings, and the lifecycle harness still passes all 7,051 checks.
- The Phase 0 benchmark contract hashes remain identical. A final five-round/50-sample confirmation shows identity observation encoding at `0.53770 -> 0.48600 ms` (`-9.6%`), fixed encoding at `0.20600 -> 0.19005 ms` (`-7.7%`), dense identity masks at `11.25345 -> 11.00010 ms` (`-2.3%`), fusion scans at `5.81045 -> 5.67800 ms` (`-2.3%`), live-status construction at `21.82135 -> 21.53770 ms` (`-1.3%`), and unchanged-status GUI polling at `0.40765 -> 0.04055 ms` (`-90.1%`). Earlier short reruns showed uniform host-load variance, so the higher-sample confirmation is the retained comparison.
- Independent review found and drove fixes for GUI/fusion copies, ignored legacy fields, clean-clone schedule portability, run-mode and stream-alias precedence, main dispatch, JSON plant IDs, metadata shape drift/malformed values, shared observation width, provenance, and tracked metadata values. The final re-review found no other Phase 2 blocker; its last symmetric fixed-mode conflict was then fixed and regression-tested.

Residual Phase 2 boundaries:

- Runtime `CardUI` metadata remains authoritative and still needs the final live-game gate. The generator supplies fallbacks only; it does not replace Unity discovery.
- The flat configuration adapter deliberately remains while later phases migrate runtime consumers. GUI widget-only state remains in `pvzrl_gui.py`; plant presets come from the registry and launched values enter the same CLI resolver.
- Fusion-result identity and recipes remain separate until Phase 3 so this phase does not change recursive/self-fusion behavior.
- The schedule is runnable in this checkout and explicitly labeled as requiring local gitignored artifacts; it is not represented as a portable model bundle.

Rollback boundary: local commit `87aee4a` (`refactor: establish canonical PvZRL metadata`).

### Phase 3 - authoritative action and fusion pipeline

Status: complete.

Implemented:

- Added `pvzrl_actions.py` with immutable `ActionIntent`, `ActionDecision`, and `ActionResult` records. Policy, legacy, and bridge identities remain explicit; source metadata and execution payloads are deeply immutable; the bridge remains the final runtime authority.
- Replaced separate mask, diagnostic, Python-filter, coach, and execution decisions with one pure validator and one complete decision cache. Cache reuse requires a content digest of the full observation and bridge legal actions plus a fingerprint of every legality-affecting configuration value. A mutated observation with the same frame counter, changed bridge actions, changed tactical input, or changed fusion/configuration state invalidates the cache.
- Preserved fixed, dynamic-14, and Adventure identity action layouts. The dynamic coach adapter now converts policy action `0..700` to the correct legacy bridge identity while retaining the original policy intent; regression coverage specifically locks dynamic wait `700 -> 0` and placement `1 -> 2`.
- Traced live model, random-baseline, scripted-baseline/fusion, human, stream/mock, GUI/manual, debug CLI, and Adventure-model paths. GUI/manual/debug sources survive coach parsing and SB3 arbitration; coach match/pending context remains in source metadata even when the model selected the same action.
- Replaced overlapping fusion result and compatibility tables with immutable `FusionRecipe` records for Peashooter self-fusion, both recursive pea upgrades, and Twin SunFlower. Compatibility is derived from recipes plus two explicit runtime-only relationships (SunFlower/Peashooter and Peashooter/CherryBomb), for which Python deliberately invents no result.
- Routed model, scripted, human, stream/mock, GUI, manual, and debug fusion requests through one source-parameterized validation/execution adapter. A shared scope/postcondition contract replaces the previous environment and coach copies; the bridge's tile-scoped mutation result remains authoritative.
- Added stable fusion event IDs. Diagnostics, coach/stream outcomes, success/failure counts, and fusion reward application each suppress duplicate handling of the same event. A pre-execution rejection remains observable but is not misreported as a bridge attempt.
- Retained the exact five-key `FUSION_RULES` compatibility view and the existing private Python-filter/fusion entry-point wrappers during the one-phase adapter window. Removed the old model/scripted/coach execution bodies, duplicated coach resource/occupancy validation, duplicated fusion postcondition validator, unused tactical-mask branch, and zero-caller action/cache helpers.

Compatibility and test evidence:

- Full pytest gate: PASS, 148 tests plus 10 subtests. The dedicated Phase 3 contract file passes all 37 tests.
- Phase 3 contract coverage exhaustively checks all 201/701 policy identities, every bit in both 701 layouts, non-5x10 stride behavior, observation/spec action-count mismatch, cache reuse/invalidation, all action/fusion sources, deep immutability, exact schemas/reasons, all recipes and runtime-only cases, recursive IDs, tile scope, and exactly-once attempt/outcome/reward handling.
- All nine retained standalone regression entry points: PASS.
- Dependency check and `compileall`: PASS.
- Bridge lifecycle/build gate: PASS, 7,051 deterministic checks and zero compiler warnings.
- Protected June 27 generalist metadata dry-run and actual load: PASS with `(701, (4297,), 370000)`. Fixed control: PASS with `(201, (357,), 350368)`. Neither checkpoint nor adjacent metadata was changed.
- The benchmark contract hashes and 433-bit legal-mask count are unchanged. The five-round/50-sample final run records dense identity mask median/p95 `11.253/12.812 -> 1.371/1.426 ms`; fixed mask `1.860/3.521 -> 0.713/0.797`; low-sun identity mask `2.637/3.292 -> 1.189/1.280`; and no-fusion identity mask `8.677/9.120 -> 1.334/1.396`. A forced cold identity miss is `9.488/11.111 ms`; same-frame reuse is the measured hot path.
- Other medians remain within the benchmark policy: fixed/identity encoding `0.206/0.538 -> 0.205/0.534 ms`, reward `0.272 -> 0.272`, fusion scan `5.810 -> 5.685`, live-status serialization `0.176 -> 0.161`, atomic write `4.249 -> 4.321`, and unchanged GUI poll `0.408 -> 0.037`. Live-status construction falls from `21.821 -> 3.014 ms` because it reuses the same proven mask decision.

Measured deviation and residual boundary:

- The supplied Phase 3 estimate targeted an 800-1,500 runtime-line reduction. The final pre-commit physical runtime diff is `+2,656/-935` (net `+1,721`), with `+1,663` test lines. This target is not met and is not being relabeled as met. The increase comes from explicit immutable action/fusion records, cache proofs, source/result schemas, and exactly-once contracts; old execution and validation bodies were removed rather than retained beside them. Later phases must deliver overall repository reduction without compressing readable safety code merely to hit a number.
- Fusion diagnostics still classify occupied tile/slot pairs through the shared fusion validator after the action cache is built. Phase 4 will replace that remaining scan with immutable per-observation facts while preserving its compatibility-first reason ordering.
- Live bridge/game execution remains intentionally unclaimed at this boundary. The newly built DLL is not installed until the final live gate can preserve the current installed bridge as a recovery point and test placement, legal/illegal/recursive fusion, reset, transitions, short train/resume, and evaluation together.

Independent review found and drove fixes for mismatched cached intent/bridge identities, dynamic coach policy-versus-legacy indexing, implicit `plant`-to-fusion conversion, missing structured terminal/timeout results, caller-supplied recipe-result drift, runtime-only result invention, and missing failed-event copy replay. The final focused rerun is green. The review also confirmed the remaining fusion-diagnostics scan is a Phase 4 facts/index concern rather than a second execution authority.

Rollback boundary: local commit `a147e93` (`refactor: unify PvZRL action and fusion pipelines`).

### Phase 4 - per-observation facts, rewards, metrics, and hot-path I/O

Status: complete.

Implemented:

- Added immutable `StepFacts` snapshots and an owner/content-verified one-entry cache. Plants, zombies, lanes, occupancy, seed slots, mower state, lifecycle signals, safety facts, and row/type counts are built once at the observation boundary and reused by action validation, masks, encoding, fusion, rewards, lane diagnostics, safety diagnostics, and episode metrics. Positional versus explicit seed-slot identities remain separate where the legacy contracts require them.
- Replaced the environment's duplicated reward capture/apply state with one immutable `RewardCompositionState` and one pure reward compositor. The captured `a147e93` replay fixture locks every component and hidden-state transition at `1e-9`; terminal fusion reward is applied exactly once, including terminal restart returns.
- Extracted lane diagnostics and environment-safety diagnostics into pure compositors. Six complete 94-field lane payloads retain their captured hashes and exact value types. Safety composition retains mower-loss ownership and was differential-tested across randomized lifecycle/cooldown transitions, malformed slots, duplicate explicit slot IDs, registry-name fallbacks, and board-refresh paths.
- Kept execution, safety, and reward snapshot ownership explicit. A periodic forced seed probe may advance the pre-action execution frame, but reward and lane deltas remain paired with the stored previous observation, its facts, and its legal actions. Normal, timeout, and terminal/restart paths share that boundary and do not rebuild the prior frame.
- Removed the old reward-state mirrors, lane/safety implementation bodies, temporary action/fusion adapters, duplicate environment analytics, and redundant readiness/accessor paths after call-site and schema coverage. Python wrappers that still form a documented compatibility boundary remain.
- Centralized episode CSV/JSON coercion and writes, plus live-status throttling and atomic replacement. Ordinary status attempts use a lazy builder, failed replacement does not advance throttle state, forced terminal writes bypass suppression, and episode-significant keys remain part of the change signature.
- Made watchdog normal-state timing records bounded and lightweight while preserving board/cooldown change booleans. Hashes, detailed differences, corruption evidence, timeout bundles, and safety context are persisted only when anomalous or explicitly verbose; safety and corruption classifications remain distinct.
- Removed repeated raw seed-slot/model-fusion decoding, eager SB3 lane fallbacks, duplicate live-status mask calculation, and per-action diagnostic cloning. The hot path now projects from cached decisions and immutable facts without changing public masks or payloads.

Compatibility and test evidence:

- Full pytest gate: PASS, 197 tests plus 10 subtests. The final independent correctness audit reran 99 focused/adversarial tests plus 10 subtests and differential-fuzzed 2,000 valid/malformed cooldown snapshots and 2,000 watchdog board/cooldown signatures with exact legacy parity.
- Component reward replay: PASS at `1e-9`; terminal fusion reward is exactly `0.62` once, and the forced-probe regression retains the stored-frame danger delta `-0.007` with distinct prior/current legal-action projections.
- Lane diagnostics: PASS for all six captured 94-field payloads. Environment-safety composition: PASS for 500 randomized lifecycle transitions plus the larger independent malformed-slot projection audit.
- All nine retained standalone regression entry points, dependency check, and `compileall`: PASS. The unchanged bridge lifecycle/build gate remains PASS at 7,051 deterministic checks and zero compiler warnings.
- Protected June 27 generalist and fixed control loads remain PASS with `(701, (4297,), 370000)` and `(201, (357,), 350368)` respectively. No checkpoint, adjacent metadata, or progress artifact was modified.
- External contracts remain exact: dense observation `8d0260e4...`; identity mask `80f460d9...` with 433 legal actions; identity vector `8c2d3547...` at width 4,297; GUI-default tactical mask `661d4005...` with 414 legal actions; tactical diagnostics `bbdc7ea9...`; live-status recursive keys/types `53e6489a...`.
- The final bridge-free benchmark is `runs/benchmarks/phase4_python_final.json`, 50 samples by five rounds. Median/p95 changed as follows: fixed mask `0.77440/0.86370 -> 0.73295/0.78910 ms`; dense identity mask `1.50830/1.61350 -> 1.42240/1.49070`; reward composition with prebuilt shared facts `0.29335/0.57580 -> 0.16905/0.20250`; fusion scan `6.13360/7.05530 -> 3.00595/3.54450`; live-status build `3.28150/3.49770 -> 1.57945/1.77260`; serialization `0.17825/0.18920 -> 0.15785/0.17730`; unchanged GUI status parse `0.04605/0.06070 -> 0.03685/0.05590`. Identity encoding median is effectively flat (`0.60850 -> 0.61105 ms`). Atomic-write median is also flat (`4.98750 -> 4.92480 ms`).
- Reward timing above is the incremental compositor with prebuilt facts, not a standalone end-to-end reward claim. Dense facts build is `1.21930/1.35960 ms`; content-verified reuse is `0.71490/0.83070 ms`, and the snapshot cost is amortized across masks, rewards, fusion, encoding, and diagnostics.
- Of 500 ordinary live-status attempts, 49 built/wrote and 451 were suppressed; one forced final write also completed. Lazy suppressed attempts measure `0.00050 ms` median.
- Independent correctness and performance audits both report PASS. They directly drove fixes for terminal fusion reward, tactical/fusion fact reuse, raw slot rescans, cache owner safety, localized names, structured terminal results, telemetry replacement ordering, watchdog truthfulness, forced-probe snapshot pairing, and exact cooldown projection semantics.

Measured deviation and residual boundary:

- The supplied Phase 4 estimate targeted a 900-1,500 runtime-line reduction. Relative to `a147e93`, existing runtime files are `+1,205/-2,604` (net `-1,399`), while the five new pure facts/reward/diagnostic/telemetry modules add 3,147 physical lines, for total runtime `+4,352/-2,604` (net `+1,748`). Test and fixture code is `+2,713/-16`; benchmark tooling is `+175/-5`. The runtime target is not met and is not being relabeled as met. The large environment deletion is real, but explicit immutable schemas, pure compositors, and adversarial compatibility boundaries dominate this phase's physical total.
- The benchmark is bridge-free and excludes Unity step latency, PPO inference/rollout SPS, and complete Tk rendering. Current source was not installed into the game during this phase, so no live placement, fusion, reset, or training claim is added here.
- Phase 5 begins with shadow lifecycle classification and explicit state records. Existing reset/progression behavior remains authoritative until recorded traces match; checkpoints, frontier, replay streaks, unlock state, and progress artifacts remain protected.

Rollback boundary: local commit `30b8910` (`refactor: consolidate PvZRL facts rewards and telemetry`).

### Phase 5 - environment, reset, wrapper, and Adventure state

Status: complete for source and automated compatibility gates. Live game execution is deferred to the final environment gate because no game process or bridge listener was available and the installed May DLL remains intentionally untouched.

Implemented:

- Added the pure `LifecycleClassification` shadow model with separate base and Adventure vocabulary, exact legacy done projections, reset context, timeout/corruption interpretation, transition identity, and authoritative-versus-transient level mismatch fields. Randomized comparisons locked legacy base and Adventure classification before any lifecycle owner changed.
- Added ordered trace fixtures for startup/loading/seed/gameplay, possible win/trophy/reward, same-level replay, loss/restart, action-freeze-to-corruption recovery, corruption reset, post-win Adventure advancement, and stable wrong-level blocking. Twenty-six frames lock every lifecycle field, legacy action order, reset/progression deltas, and sanitized source provenance.
- Corrected two demonstrated lifecycle defects with focused rollback commits: base reset no longer evaluates an undefined orphan `level3_start_state`, while Level-3 preflight remains at the trainer boundary; an episode-facing `action_freeze` now hands the base reset machine the supported `env_corruption` recovery reason.
- Added explicit `ResetRuntimeState`, `EpisodeRuntimeState`, `WatchdogRuntimeState`, and immutable Generalist progression records. Phase 3's immutable `FusionIntent`, `FusionDecision`, and `FusionExecution` remain the one fusion record family. Core wrapper episode counters and reset/Adventure initialization now have one owner; specialized lane, fusion, reward, coach, and curriculum counters stay separate where their schemas differ.
- Established wrapper/base observation ownership with content/revision identity. Reset, Adventure start, step return, and mask fallback all adopt one observation; divergent copies fail before policy masking or bridge execution. Adventure terminal effects set `transition_pending`, so masks and steps remain unavailable until a fresh board is adopted.
- Promoted the differential-tested Generalist reducer from shadow to canonical progression ownership without changing the serialized `AdventureGeneralistProgress`, attempt, or level log schemas. Replay/collection hooks receive the legacy-equivalent provisional win state before external effects, then the final immutable state commits after the effect result. This preserves frontier, streak, attempt, cleared/mastered levels, unlock ordering, replay recovery, max-level clamping, maintenance isolation, and checkpoint warm-start behavior.
- Consolidated only the proven-equivalent polling leaf used by popup dismissal, board discovery, and gameplay readiness. The reset dispatcher, post-win unlock collector, same-level replay, Adventure preparation, and Generalist startup recovery retain explicit strategy order because simultaneous-signal priorities differ materially.
- Updated the bridge-free benchmark to establish observations through the production ownership boundary. Direct `_last_observation` injection remains a test-only compatibility path and is no longer mismeasured as the production mask hot path.

Compatibility and test evidence:

- Full pytest gate: PASS, 264 tests plus 10 subtests. The seven Phase 5 files pass 67 focused tests. The retained corruption, fusion-chain, Generalist identity, timeout, fusion compatibility, fusion reward, human coach, metadata, and stream coach entry points all exit zero.
- Independent focused reviews differential-tested reset payloads across eight fixed/Adventure terminal paths, progression across 48,384 shadow cases and hook-visible live-wrapper states, episode initialization across 10,000 randomized states, watchdog summaries through 1,000 actions, and observation ownership across reset/terminal/test-only paths. Review found and drove the reset handoff fixes, shallow progression immutability fix, and provisional pre-effect progression boundary.
- Dependency readiness and `compileall`: PASS. Bridge build: PASS with zero warnings. Lifecycle harness: PASS, 7,051 checks.
- Protected metadata dry-runs and actual CPU loads remain PASS: Generalist `(701, (4297,), 370000)` and fixed control `(201, (357,), 350368)`. The June 21 control also remains inventoried. All six checkpoint/model and adjacent metadata mtimes predate this refactor; current SHA-256 values are recorded by the gate and no protected artifact or progress file was written.

| Protected artifact | Bytes | UTC mtime | SHA-256 |
| --- | ---: | --- | --- |
| Fixed `model.zip` | 1,081,313 | 2026-05-07T23:53:03.4743374Z | `8a4de3fcb8b3691119fbd3203bc30d25526b82ba1f3814b840447cb20de521eb` |
| Fixed `model_metadata.json` | 715 | 2026-05-07T23:53:03.4873399Z | `ce77348031b44f733f48cd5117fb155e1eec4b892d3367e729532e1c452cd3fa` |
| June 21 370k checkpoint | 7,795,814 | 2026-06-21T14:34:29.0185485Z | `a8f42c7cb88c2e334be39bc0cc114c624efd5f5d50f1d97f48a52b8a27e5abc6` |
| June 21 metadata | 2,282 | 2026-06-21T14:15:03.4839261Z | `677fb7c771a5c54e177d5c8ce87b5421e185e163fd781b2cf7c62e492c9be3db` |
| June 27 protected 370k checkpoint | 7,869,579 | 2026-06-28T14:07:08.3413716Z | `dfd2b800d5b4bb24772be868e4cd31a320e07aec14c2d67d2ca509eab49c0b5b` |
| June 27 metadata | 2,028 | 2026-06-27T21:27:30.0912885Z | `378ce2b592570ae7a5efe96f7e85aaab5eaa5e563b26155308e659e54484ad8f` |
- External hashes remain exact: dense observation `8d0260e4...`; identity mask `80f460d9...` with 433 legal actions; identity vector `8c2d3547...` at width 4,297; GUI tactical mask `661d4005...` with 414 legal actions; tactical diagnostics `bbdc7ea9...`; live-status recursive keys/types `53e6489a...`. Live-status throttling remains 49 builds/writes and 451 suppressions across 500 ordinary attempts, plus one forced final write.
- Final bridge-free benchmark: `runs/benchmarks/phase5_python_final.json`, 50 samples by five rounds. Phase 4 to Phase 5 median/p95 is fixed mask `0.73295/0.78910 -> 0.72625/0.77500 ms`; dense identity mask `1.42240/1.49070 -> 1.40665/1.47730`; reward with prebuilt facts `0.16905/0.20250 -> 0.13475/0.15580`; fusion scan `3.00595/3.54450 -> 2.57805/3.13220`; live-status build `1.57945/1.77260 -> 1.49960/1.82680`; facts build `1.21930/1.35960 -> 0.99900/1.16100`; verified facts reuse `0.71490/0.83070 -> 0.70670/0.79640`. No deterministic contract changed and no unexplained median regression remains.

Measured deviation and residual boundary:

- The supplied Phase 5 estimate targeted an 800-1,400 runtime-line reduction. Relative to `30b8910`, the three changed target runtime files are `+734/-390` (net `+344`); the three new lifecycle/progression/state modules add 1,074 lines, for total runtime `+1,808/-390` (net `+1,418`). Phase 5 tests add 1,669 lines, fixtures add 777, and benchmark tooling is net `+12`. The target is not met and is not being relabeled as met. Explicit state schemas, trace fixtures, compatibility projections, and adversarial boundary tests outweigh the removed scalar/initialization duplication.
- `action_durations` remains an unbounded per-episode list exactly as before, cleared on fixed reset and Adventure start. Five wrapper terminal-to-reset handoff latches remain outside the state records. They are parity-tested but remain a future consolidation candidate only if a trace-backed change materially improves ownership.
- Large high-level loops remain intentionally separate: reset, unlock collection, same-level replay, Adventure preparation, and Generalist recovery use different action priorities when signals conflict. The nominal ordered traces do not justify collapsing them into one generic dispatcher.
- No `PlantsVsZombiesRH`/MelonLoader process was running and port 32323 was closed. The newly built DLL is `f0216d69...`; the installed recovery DLL remains `5643ec37...`. Source was not copied into `Game Files/Mods`, so live startup, win/loss/timeout, post-win unlock, same-level replay, Adventure advancement, short train/resume, and evaluation are not claimed here. The final live gate will first preserve the installed DLL, run `scripts\build_bridge.ps1 -CopyToMods`, launch `Game Files\PlantsVsZombiesRH.exe`, and execute observation/legal-action, placement/fusion, reset/terminal, seed-selection, short train/resume, and fixed/Adventure evaluation commands before restoring or accepting the new install.

Rollback boundaries, in order: `e36b4cd` shadow classifier; `fd31df6` reset handoff defects; `d58b0bd` progression shadow; `2220228` ordered traces; `ca76308` observation ownership; `f793aca` reset state; `084fcab` polling leaf; `0632401` episode/watchdog state; `bee529c` canonical progression; `b259bec` benchmark ownership.

Later phase results, live verification, final independent reviews, code statistics, deferred work, remaining duplication/risks, and exact rollback commits will be appended rather than inferred in advance.

### Phase 6 - C# bridge decomposition and observation optimization

Status: complete for source and automated compatibility gates. Live IL2CPP placement, fusion, reset, and seed-selection checks remain deferred because no game process or bridge listener was available; the installed recovery DLL was not modified.

Implemented:

- Split the 11,887-line bridge monolith in two move-only commits. Server/request state and DTOs moved first, followed by partial `BridgeMod` files for server, commands, reset, fusion, placement, observation, Adventure UI, seed/UI, and runtime control. Exact reconstruction checks proved the original 10,742 moved method-body lines retained their order before optimization.
- Updated `scripts/build_bridge.ps1` to generate the plant registry first, enumerate every top-level bridge `.cs` source in ordinal filename order, and compile with Roslyn `/deterministic+`. After the independent audit found that sorted inputs alone did not make the binary reproducible, two consecutive unchanged builds produced identical DLL and PDB hashes. The final post-cleanup DLL hash is `5ece6e03d6d9234e1c4fb96cab8b6730d8efb514a87d93d9388d28ea9d730507`.
- Centralized assembly metadata in `BridgeMod.cs`; extracted DTO and server files no longer duplicate Melon or friend-assembly attributes.
- Replaced slot-by-row-by-column plant scans with one bounded occupied-cell index per observation. Invalid and out-of-bounds source entries are excluded before key construction, avoiding integer-key collisions. The lifecycle harness differential-tests 500 randomized boards cell by cell.
- Replaced per-row zombie filtering, sorting, and repeated classification with a single lane-accumulator pass. The implementation preserves alive filtering, stable nearest-zombie tie behavior, cone/bucket/tough classification, nullable empty-lane values, and the legacy float-sum result. Five hundred randomized exact comparisons cover every lane field.
- Consolidated seed compatibility counts, minimum positive costs, sorted slot DTOs, and slot-index lookup into one cache population pass. Cache invalidation clears list and dictionary views together; cached IL2CPP wrappers are validated for object presence, active hierarchy, and stable instance identity before reuse.
- Shared one lazy `CardUI` snapshot between observation cost and cooldown projections, resolved duplicate configured plant types once, and reused the original seed-probe scan for immediate cache refresh while retaining original scan order and first-card-per-type selection. Per-card stale-wrapper failures remain isolated.
- Added a pure bridge observation benchmark with exact legacy/indexed cell and lane projection checks. The final artifact is `runs/benchmarks/phase6_bridge_pure_final.json`; it explicitly excludes Unity scans, `CheckBox`, IL2CPP lifetime, sockets, and live bridge latency.
- Strengthened the lifecycle harness from 7,051 to 58,553 checks. The 122-property top-level observation contract remains locked, and nested DTO public JSON fields are now recursively fingerprinted at `c0d34a11cb12f53e0e0a84d82bf35022b73bf3eaeb9e83706bc6e4b3b005b6b5`.

Compatibility and test evidence:

- Full pytest gate: PASS, 264 tests plus 10 subtests. All nine retained standalone regression entrypoints, dependency readiness, and `compileall` pass. A split-source inspection test was updated to follow the same deterministic multi-file source set as the build instead of assuming `GetPlantCost` remained in `BridgeMod.cs`.
- Bridge build: PASS with zero warnings. Lifecycle harness: PASS, 58,553 checks. Consecutive unchanged builds produce byte-identical DLL and PDB files.
- Protected Generalist and fixed metadata dry-runs remain compatible at 701 and 201 actions. Actual CPU `MaskablePPO.load` remains `(701, (4297,), 370000)` and `(201, (357,), 350368)`. All six protected checkpoint/metadata byte sizes, mtimes, and SHA-256 values exactly match the Phase 5 inventory.
- Python contract hashes remain exact: dense observation `8d0260e4...`; identity mask `80f460d9...` with 433 legal actions; identity vector `8c2d3547...` at width 4,297; GUI tactical mask `661d4005...` with 414 legal actions; tactical diagnostics `bbdc7ea9...`; live-status recursive keys/types `53e6489a...`.
- Two 50-by-5 Python benchmark reruns show no contract drift. The repeated medians are fixed mask `0.6745/0.6752 ms`, identity mask `1.32795/1.29465`, reward `0.1412/0.1379`, fusion scan `2.69095/2.63505`, live-status build `1.51005/1.5053`, facts build `1.1203/1.10735`, facts reuse `0.7137/0.70865`, and unchanged GUI status parse `0.0371/0.0369`. The facts-build median is 10.8-12.1% above the older Phase 5 sample even though Phase 6 changed no Python runtime path; the two same-source reruns cluster within 1.2%, so this is recorded as cross-run environment drift rather than attributed to the C# optimization.
- Pure C# helper benchmark final median/p95: legacy occupancy `0.276796/0.424751 ms`, indexed occupancy including set construction `0.003684/0.003913` (75.1x/108.5x); legacy lane filter/sort `0.022237/0.0241235`, one-pass lanes `0.004547/0.004784` (4.9x/5.0x). Exact occupancy and lane signatures match before timing.
- Independent source and benchmark reviews found and drove fixes for deterministic compiler output, recursive nested-DTO coverage, per-cell benchmark equivalence, checksum anti-elision, and duplicate assembly metadata. The final audit reports no remaining automated correctness blocker.

Measured deviation and residual boundary:

- The supplied Phase 6 estimate targeted a 1,000-1,800 production-line reduction after the move-only split. Phase 5 had 12,099 physical bridge source lines; move-only decomposition added 256 lines of file headers and partial-class scaffolding; optimization and helper extraction added another 251; benchmark access added one friend-assembly line; final assembly-metadata cleanup removes 8, leaving 12,599 lines, or net `+500`. This target is not met and is not relabeled as met. The phase delivered coherent file ownership, stronger deterministic/schema proofs, and measured hot-path improvements, but not source reduction.
- `CreatePlant.CheckBox` remains evaluated per candidate. The brief explicitly requires benchmarking it before caching; no live Unity/IL2CPP timing was available, and caching its dynamic answer without lifetime/UI-state evidence would risk stale legality. Broad Unity scans remain in reset, seed/UI discovery, and diagnostic/command paths whose ordering and mutation safety differ; they were not collapsed without live evidence.
- Sharing a single observation `CardUI` snapshot means one transient scan failure affects both cost and cooldown fallback within that observation, whereas the legacy code could attempt independent scans. The synchronous main-thread path, per-card exception isolation, invalidation rules, and fallback behavior keep this low risk, but it remains an explicit live IL2CPP gate.
- No `PlantsVsZombiesRH`/Melon process was running and port 32323 was closed. The built deterministic DLL is `5ece6e03...`; the installed recovery DLL remains `5643ec37...`. Source was not copied into `Game Files/Mods`, so live placement, legal/illegal/recursive fusion, reset, seed-selection timing, bridge latency, and `CheckBox` timing are not claimed here.

Rollback boundaries, in order: `cb9cc41` server/DTO move-only split; `f63d63c` bridge partial move-only split; `13d20d5` occupancy/lane optimization; `a184889` seed compatibility caches; `35834cb` shared observation card snapshot; `3827a7f` seed-probe scan reuse; `bb3f6d9` pure benchmark; `c7f53db` split-source contract test; `0f58b3c` deterministic build; `070a60d` recursive DTO schema; `2698bc9` assembly metadata cleanup.

### Phase 7 - GUI, status normalization, coaches, and process control

Status: complete for source, automated compatibility gates, and a local Tk lifecycle smoke. Live game-backed GUI execution remains part of the final environment-specific gate because no game process or bridge listener is available and the installed recovery DLL remains untouched.

Implemented:

- Extracted GUI command construction, process/log control, status reading/normalization, diagnostics rendering, and coach queue writes into focused mixins/modules without redesigning the existing tabs or controls. Ten exact argv snapshots cover fixed train/resume/eval, Adventure evaluation, Generalist fresh/resume/eval, Level-3 train/eval, and the combined coach/stream/fusion configuration.
- Added one `LiveStatusReader` that tracks `mtime_ns` and size, caches successfully parsed content, reports malformed and missing files without discarding the last good state, and recalculates age/health even when content is unchanged. Stale age now takes precedence over an old `blocked_reason`, while case-insensitive and empty-value compatibility fallbacks remain intact.
- Added `NormalizedStatusIndex` so case-insensitive aliases are indexed once per normalized payload instead of rebuilding lower-cased dictionaries at each lookup. Semantic render keys suppress unchanged/equal payload rendering while ignoring volatile timestamp and stream-poll-age values; key presence remains part of the compatibility contract.
- Suppressed unchanged Tcl `StringVar` writes in live, diagnostics, coach, and fusion views. Status parsing, normalization, render decisions, and age/health updates are independently testable; the Tk-facing layer remains intentionally thin.
- Extracted bounded subprocess/process lifecycle handling. Launch, second-launch rejection, natural/stale exits, terminate/kill fallback, idempotent stop, queue polling, and shutdown callback cancellation are covered without weakening the production `Popen` contract or moving widget updates off the Tk thread.
- Kept log dequeue work bounded at 250 records and a 12 ms dequeue budget per tick, with retained history capped at 5,000 lines and 1 MB. Unfiltered complete-line rollover is incremental, filtered refreshes are coalesced, oversized single writes and batches cannot bypass retention, and partial-line history boundaries fall back to a bounded rebuild to preserve exact widget/history parity.
- Added a shared GUI JSONL queue sink for coach and fusion commands, including source, mode, and command identifiers. Human and stream coach command metadata now share one canonical alias table, and valid file-coach JSON records are parsed once. Existing shared partial-record tailing, malformed-record isolation, queue clearing, local mock-stream behavior, moderation fields, and compatibility aliases remain covered.

Compatibility and test evidence:

- Full pytest gate: PASS, 292 tests plus 10 subtests. The independent Phase 7 correctness audit reran 36 focused GUI tests, checked every emitted command flag against current argparse help, exercised 5,000 real-Tk rollover iterations, and found no remaining blocker. All nine retained standalone regression entrypoints, dependency readiness, and `compileall` pass.
- A withdrawn real `tk.Tk()` dashboard creates all required train/eval/log widgets, builds representative commands, schedules both polling callbacks, processes one event cycle, and closes through the production shutdown path. The independent audit repeated real-Tk construction and shutdown successfully. This is a local GUI lifecycle smoke, not a claim of a live bridge/game session.
- Bridge build: PASS with zero warnings. Lifecycle harness: PASS, 58,553 checks. The deterministic DLL remains `5ece6e03...`; the installed recovery DLL remains `5643ec37...` and was not replaced.
- Protected Generalist and fixed metadata dry-runs remain compatible at 701 and 201 actions. Actual CPU model loads remain `(701, (4297,), 370000)` and `(201, (357,), 350368)`. The six protected checkpoint and metadata hashes, sizes, and mtimes remain identical to the Phase 5 inventory.
- Python contract hashes remain exact: dense observation `8d0260e4...`; identity mask `80f460d9...` with 433 legal actions; identity vector `8c2d3547...` at width 4,297; GUI tactical mask `661d4005...` with 414 legal actions; tactical diagnostics `bbdc7ea9...`; live-status recursive keys/types `53e6489a...`. The new normalized GUI alias projection is locked at `400cd2f1...`.
- Final bridge-free benchmark: `runs/benchmarks/phase7_python_final.json`, 50 samples by five rounds. Hot-cache case-insensitive lookup improves from `9.44235/11.1399 ms` median/p95 for the legacy repeated-map surrogate to `0.16040/0.23450 ms` for the prebuilt index (58.9x median). Same-object render-key checks are `0.00020/0.00030 ms`; semantically equal fresh payloads are `0.45265/0.52870 ms`. Unchanged status reads including signature, health, and cached-state handling are `0.06540/0.07290 ms`. The no-op-widget log surrogate improves from `0.04390/0.05040 ms` for a retained-history rebuild to `0.00410/0.00450 ms` for incremental rollover (10.7x median). Unchanged coach-view application is `0.06230/0.06580 ms`.
- The benchmark deliberately forces compatibility-alias misses in the casefold comparison and excludes index construction; it is a hot-cache lookup result, not end-to-end rendering. Log timing uses a no-op text widget, and the 12 ms dequeue budget excludes the final Tk update. Full Tk rendering, live bridge latency, PPO inference, and rollout throughput remain unmeasured.

Measured deviation and residual boundary:

- The supplied Phase 7 estimate targeted a 700-1,200 runtime-line reduction. Relative to the Phase 6 documentation boundary `14aa99e`, production Python is `+2,293/-1,868` (net `+425`), tests are `+606/-4`, and benchmark tooling is `+153`. The target is not met and is not relabeled as met. The 1,862-line `pvzrl_gui.py` deletion is principally coherent relocation into focused modules, while the measurable gains come from cached status/index work, suppressed Tcl writes, and bounded incremental logs.
- Compatibility status emission still expands a large alias surface across top-level, `coach`, `human_coach`, and `stream_coach` payloads; the final synthetic payload is 42,591 bytes. Removing those aliases before all consumers migrate would violate the required adapter window, so Phase 7 normalizes them at the reader and leaves writer contraction deferred.
- Numeric coercion remains independently implemented at several runtime boundaries, and the extracted process/view modules are pragmatic Tk mixins rather than fully pure typed controllers. Further consolidation is deferred because it would cross stable command, status, and serialized compatibility boundaries without a demonstrated Phase 7 defect.
- The log benchmarks do not measure complete Tcl/Tk cost, and the dequeue time cap is not a strict total-tick latency cap. Status writers can still produce expensive compatibility payloads; this phase prevents unchanged GUI reparsing/rerendering but does not claim their removal.
- No online service or network dependency was added. Local human-coach, file-tail, moderation, stream/mock queue, and intervention-log behavior remain local and compatible.
- No `PlantsVsZombiesRH`/MelonLoader process or bridge listener was available. Current source was not copied to `Game Files/Mods`, so live GUI connection, observation, placement, fusion, reset, transitions, short train/resume, and evaluation are not claimed here.

Rollback boundaries, in order: `14e1a3d` status reader; `6373f82` command snapshots; `b066a31` command construction; `943da70` command-module cleanup; `ec9cea3` process edge contracts; `29a2e50` process control; `26d654b` bounded logs; `c3d2f0b` normalized status index; `755c481` coach queue writes; `8fa20c3` coach command metadata; `1c80881` diagnostics view; `ed199e2` GUI benchmarks; `b8048bd` oversized-log retention; `d3389c4` partial-line rollover and status-write parity.

### Phase 8 - full validation and independent review

Status: complete for every automated/local-Tk gate and the recorded partial real-game gate. Overall goal acceptance remains incomplete because fixed training/evaluation and several terminal/progression traces remain unexecuted, the repeat benchmark did not close the cross-run 5% policy gate, and physical runtime volume increased rather than decreased.

Final implementation and review fixes:

- The concurrency review found that a child which continued to report alive after `kill()` could renew the GUI close deadline indefinitely. Commit `34dabc1` makes the deadline hard: one final kill attempt is followed by an actionable still-alive record, bounded 0.2-second thread joins, bounded log drain, callback cancellation, and root destruction. A `NeverExitProcess` regression locks the behavior.
- The refactoring-quality review found 260 lines of unmounted GUI tab builders/options, an expired private action-filter adapter, four additional private Python helpers, and a 300-line dead C# mix/reflection/reset/seed/observation helper graph. Tests now inspect the mounted Training tab and call the canonical action decision/status health paths. Commits `409138a` and `80376cf` remove 636 runtime lines and add one, net `-635`; full Python, bridge, script, and real-Tk gates remain green.
- The final ownership map now names the split bridge, pure facts/reward/lifecycle modules, Generalist reducer, GUI command/process/status/view/queue modules, and shared file tailer rather than attributing those systems to their former monoliths.
- A final post-report simplification pass made `pvzrl_lifecycle.py` the pure lifecycle predicate authority (`43ce840`, runtime net `-209`), removed a dormant dashboard/coach shell (`07714b3`, `-157`), derived redundant reward schema and seed projections (`32c9e20`, `-39`), unified evaluation reducers without the initially measured reporting slowdown (`1cb533f`, runtime `-63`, tests `+24`), shared exact live-fusion test setup (`c2bd74e`, `-33`), and removed the duplicate bridge type-count projector (`dd563de`, `-15`). Together with the two-line mask fast path, this reduces final runtime by another 518 physical lines. Every slice received a separate clean review.
- No further high-confidence bulk deletion remains after this second sweep. Python production imports are acyclic; no private Python no-caller helper or C# private identifier-singleton helper remains. Closing the overall volume gap would require another substantive redesign of live-compatible state schemas and large reset/step loops, not safe dead-code cleanup.
- The first retained live Generalist attempt exposed a real recursive-fusion identity defect: Python predicted plant `1031` after DoubleShooer plus Peashooter while the game produced `1090`, yielding five `fusion_result_mismatch`/mask-bridge disagreements. Authoritative live results and the game enum identify `1030` as DoubleShooer, `1090` as SplitPea, `1032` as GatlingPea, and `1031` as SunShroom. Commit `c7d38bc` corrects the registry/feature chain and adds bridge-free empty/unrelated-board candidate regressions; the failed attempt is retained rather than hidden.

Final automated and compatibility evidence:

- Dependency readiness and `compileall`: PASS. Full pytest: PASS, 295 tests plus 10 subtests. All nine retained standalone regression entrypoints exit zero after the final recursive-fusion correction.
- Bridge build: PASS with zero warnings. Three consecutive current-source builds are byte-identical. Final DLL SHA-256 is `3b849aa40505e40bc01ecf54bc2c0ff40b2970cb1d33a237d76e42ff0185de5f`; PDB SHA-256 is `f0e21eff77ccf8593b876b69a0aba01468429d1b158e2702e19d3597ea7922e5`. The lifecycle harness passes 58,553 checks.
- A withdrawn real `tk.Tk()` dashboard builds the mounted Training, Evaluation, Coach, Diagnostics, Runs/Models, status, and log surfaces, schedules both pollers, processes an event cycle, and closes through the production callback. Command, status, queue, process, rollover, and never-exits lifecycle coverage remains green.
- Both protected metadata dry-runs return `compatible=true`, `ok=true`, empty compatibility-warning lists, and no blocked reason. The fixed invocation also emits the known `IgnoredLegacyConfigWarning` for the retired `enable_fusion_diagnostics` key in its resolved legacy configuration. Actual CPU loads remain Generalist `(701, (4297,), 370000)` and fixed `(201, (357,), 350368)`. The June 21 Generalist control also remains loadable.
- All six protected model/metadata sizes, mtimes, and SHA-256 values exactly match the Phase 5 table. The installed recovery DLL was preserved before the live gate, the deterministic final DLL was installed temporarily for validation, and the original DLL was restored and reverified at `5643ec37984762ab72fea3c50e87fcc466b905c5cf046c307fdaa6e0ce42a0f4`.
- Current deterministic Python contracts are: dense observation `c341506a...`; identity mask `80f460d9...` with 433 legal actions; identity vector `8c2d3547...` at width 4,297; GUI tactical mask `661d4005...` with 414 legal actions; tactical diagnostics `26a09c13...`; recursive live-status schema `5d826a00...`; GUI alias projection `400cd2f1...`; corrected fusion-candidate snapshot `bac8d8a6...`. The dense/diagnostic/status hashes intentionally include the corrected recursive fusion identities; the mask/vector/GUI alias contracts remain unchanged. Bridge top-level observation and recursive nested DTO contracts remain 122 properties and `c0d34a11...`.

Partial real-game evidence:

- The installed recovery DLL was copied to `runs/manual_phase8/final_live_gate/PvZRLBridge.recovery.dll` and verified at SHA-256 `5643ec37984762ab72fea3c50e87fcc466b905c5cf046c307fdaa6e0ce42a0f4` before the deterministic final DLL was installed. The final bridge loaded and accepted connections on port 32323; after the gate the recovery DLL was restored to `Game Files/Mods`, its hash was reverified, and the listener was closed.
- Adventure startup passed through popup dismissal, the reward/menu state, Adventure selection, Level 6 seed selection, and gameplay with the duplicate `SunFlower,SunFlower,Peashooter,Peashooter` loadout. The corrected-source startup is persisted in `runs/manual_phase8/final_live_gate/recursive_fix_startup_state_smoke.log`.
- The environment smoke obtained a structured observation and 201 legal actions, preserved occupied-cell exclusion, accepted a legal teacher action, advanced on wait, placed on the requested tile, and rejected an invalid action safely.
- A soft reset requested during active wave 7 was rejected by the production safety guard. This is expected safety behavior, not a reset failure. Three real losses exercised `LoseMenuBtn.TryAgain`, seed selection, Let's Rock, and reset acceptance on clean four-slot wave-0 boards with no plants and five mowers (`reset_after_fusion_patch.log`, `reset_for_generalist_eval.log`, and `recursive_fix_reset_for_eval.log`). A separate seed-selection flow accepted a clean two-slot board (`reset_before_coach_scope_rerun.log`); it is not mislabeled as a TryAgain reset.
- Live fusion semantics passed through the dedicated fusion bridge command. Peashooter plus SunFlower produced plant ID `1000` (`PeaSunFlower`) with exactly one changed tile; an empty source was rejected as `source_not_found`; occupied normal placement remained blocked; and the control plant was unchanged. The initial empty-board diagnostic was corrected so setup state is informational rather than a false failure.
- The live coach fusion-scope test exposed and fixed a bare-`PvZGymEnv.client` compatibility defect. Its rerun selected one fresh `fusion_step`, mutated only the requested tile, and preserved the control plant; the bridge-free coach suite now covers both wrapped and bare environment boundaries.
- The first protected Generalist run is a retained FAIL for fusion compatibility, not final evidence: `generalist_eval_20260712_203311` recorded five illegal `fusion_result_mismatch` events and five mask/bridge disagreements. After the correction, a 311-step run joined an already active wave-3 board and proved 13/13 fusion success with no disagreement, but was not accepted as a clean episode.
- The final protected 370,000-step Generalist run started at the Level-6 seed-selection boundary, auto-selected the exact duplicate loadout, and entered a clean wave-0 board with 201 legal actions, no plants, and five mowers. `generalist_eval_recursive_fix_clean_20260712_210447` completed a classified loss in 547 policy steps with process exit zero, final wave 9, two mowers lost, 45 attempted/45 successful/0 failed fusions, zero illegal actions, zero bridge errors, zero reset failures, zero mask/bridge disagreements, and no rejection reasons. Result counts were runtime-only/unknown `-1`: 3, DoubleShooer `1030`: 8, SplitPea `1090`: 7, GatlingPea `1032`: 6, and TwinFlower `1033`: 21. The shell exit and run path are persisted in `generalist_eval_recursive_fix_clean.log`; no anomaly diagnostics file was produced because no anomaly occurred.
- The protected fixed Level-3 evaluator was invoked from an isolated output directory and correctly failed fast at the actual Level-6 boundary with `blocked_reason=not_at_level3_specialist_start_state`. WallNut and CherryBomb were not unlocked in this profile, so fresh/resumed four-slot fixed training was not started against an incompatible live seed bank.
- Short fresh/resumed fixed training, fixed Level-3 evaluation at Level 3, win/timeout/reward/unlock/replay/advancement traces, live game-backed GUI operation, and measured live bridge/step latency remain unverified.
- Final cleanup reverified the installed recovery hash and a closed port 32323, but Windows continued to expose an elevated `PlantsVsZombiesRH.exe` process record (PID 23116, no window/listener). Normal stop, forced `Stop-Process`, and `taskkill /F` could not remove it because access was denied. Manual elevated-process cleanup remains required and is not mislabeled as a stopped-process proof.

Independent review disposition:

- Compatibility reviewer: no blocker after 242 focused tests plus 10 subtests, protected loads/dry-runs/hashes, fusion/reward/schema coverage, and the 58,553-check bridge harness.
- Concurrency reviewer: the unbounded GUI close finding was fixed; 65 focused deadline, queue, tailer, process, callback, and shutdown tests plus the bridge harness leave no high-confidence concurrency blocker. One arbitrarily large newline-free child record can still make a single Tk drain item exceed the nominal time budget, so no strict total-tick claim is made.
- Refactoring-quality reviewer: code ownership, import direction, pure calculation boundaries, bridge partials, and focused GUI mixins pass; no service/factory/web proliferation or commented superseded body remains. The review explicitly rejects a full-goal claim because net runtime volume increased.
- Performance reviewer: targeted GUI and C# improvements are credible and every deterministic contract matches, but the repository-wide no-regression gate is not closed. Two quiet-intent same-source Phase 8 runs still show broad, inconsistent host-level slowdowns on paths untouched by Phase 7/8.
- Final-diff reviewer: changed files remain inside the planned config/docs/Python/test/build/bridge surfaces. No binary diff, tracked run/model/checkpoint/build/cache artifact, credential pattern, URL, email, user-specific path, dependency/environment/deployment change, deleted test file, merge, or history rewrite was found. Protected hashes and the installed recovery DLL are exact; `git diff --check` is clean.

Final performance record:

- The current-source confirmation is `runs/benchmarks/phase8_post_reduction_final.json`, 50 samples by five rounds. It records the current hashes above and the unchanged 49 writes/451 suppressions plus one forced final write. Historical Phase 8 repeats remain `runs/benchmarks/phase8_python_final_run1.json` and `runs/benchmarks/phase8_python_final.json`.
- The fixed/identity SB3 mask projection now copies index-preserving raw masks directly while retaining the shifted `dynamic_14` loop. An alternating exact-output A/B measures fixed projection `0.01960 -> 0.00850 ms` median (2.31x) and identity projection `0.39195 -> 0.03970 ms` (9.87x). In the full current benchmark, fixed/identity mask medians are `0.84485/1.30230 ms`; compared with the immediately preceding same-machine lifecycle-source run they are 6.0% and 22.4% lower, but that cross-process comparison is supporting evidence rather than a causal claim.
- Evaluation aggregation was benchmarked after consolidation because the first canonical-row version doubled a 100-log synthetic workload. The retained typed-record path is performance-neutral: old `2.3314 s`, new `2.3038 s` across 200 complete summaries (1.2% faster), with 1,000 randomized whole-summary differentials and exact malformed-input parity.
- GUI case-insensitive status lookup measures 200 mixed-case alias lookups per operation with index construction outside timing. Across the two repeats, the legacy repeated-map surrogate is `10.40030/13.70210` and `11.46040/19.79760 ms` median/p95; the hot cached index is `0.16425/0.16760` and `0.19930/0.20720 ms`, a 57.5-63.3x median improvement.
- The 5,000-line no-op-widget log surrogate is `0.04610/0.05500 -> 0.00450/0.00480 ms` in run 1 and `0.05940/0.08640 -> 0.00550/0.00610 ms` in run 2, a 10.2-10.8x median improvement. Same-object render-key checks are `0.00030 ms`; equal fresh payload comparisons are `0.48155/0.55820` and `0.59505/0.64550 ms`.
- The final C# pure-helper benchmark remains exact and unaffected by dead-code removal: indexed occupancy including set construction improves 75.1x/108.5x median/p95, and one-pass lanes improve 4.9x/5.0x. It excludes Unity scans, `CheckBox`, IL2CPP lifetime, sockets, and live bridge latency.
- The two Python repeats do not prove repository-wide no-regression. Against the mean of the two stable Phase 6 runs, their mean medians are slower by 17.1% fixed mask, 14.4% dense identity mask, 27.4% identity cooldown, 25.6% identity encoding, 23.8% reward composition, 17.4% facts build, 97.1% unchanged-status read, 126.8% status signature, and 43.9% atomic write. The Phase 8 repeats also disagree materially on several p95/filesystem/status values. Phase 7/8 did not change the non-GUI timed implementations, so the broad movement is consistent with host/cache/load drift, but the fixed-order benchmark lacks interleaving, confidence intervals, and power-state metadata needed to establish causality. The requested approximately-5% cross-run performance acceptance is therefore recorded as unmet rather than explained away.
- No benchmark includes full Tk rendering, Unity/IL2CPP, live bridge/environment-step latency, PPO inference, rollout SPS, or actual game `CheckBox` timing.

Final code statistics, using the exact baseline exclusions:

| Measure | Baseline | Final | Change |
| --- | ---: | ---: | ---: |
| Runtime files | 17 | 47 | +30 |
| Runtime physical lines | 48,127 | 54,821 | +6,694 (+13.91%) |
| Python runtime lines | 36,562 | 42,655 | +6,093 |
| C# runtime lines, excluding generated registry | 11,488 | 12,070 | +582 |
| Build-script lines | 77 | 96 | +19 |
| Raw runtime diff | - | +25,374 / -18,680 | +6,694 |
| Python test files | 9 | 32 | +23 |
| Python test physical lines | 6,154 | 14,057 | +7,903 |
| Raw Python test diff | - | +7,945 / -42 | +7,903 |
| Documentation diff (`REFACTOR_PLAN.md`, `REFACTOR_REPORT.md`, root `AGENTS.md`) | - | +1,248 / -0 | +1,248 |
| Python function definitions | 1,043 | 1,294 | +251 |
| Python class definitions | 59 | 135 | +76 |
| Confirmed function/method names removed | - | 98 | 84 Python + 14 C# |
| Confirmed class names removed | - | 1 | one internal C# helper class |
| Subsystem duplicate families consolidated | - | 22 | semantic count; no textual-clone claim |
| Explicit compatibility wrapper/view/projection surfaces remaining | - | 12 | retained where consumers cannot be proven migrated |
| Source units split | - | 2 | bridge monolith and GUI dashboard |
| Runtime files deleted | - | 0 | source was split/additive; dead bodies were removed in place |

The raw removal count is not a code-reduction claim: 10,742 C# method-body lines were moved during the bridge split, and most of Phase 7's 1,862-line GUI deletion was relocated into focused modules. The final physical total is authoritative. The 22 consolidated duplicate families cover request ownership, JSONL tailing, registry/configuration, action decisions, fusion validation/execution, per-observation facts, rewards and reward schema, lane/safety diagnostics, episode telemetry and evaluation reducers, live-status writing, lifecycle classification, Generalist progression, bridge occupancy/lanes/cards/type counts, GUI status/commands/process/logs, coach queue/aliases, and live fusion-test setup. The 12 narrowly counted compatibility surfaces are four registry projections, reward/reset adapters, two deprecated fusion views, three SB3 state projections, and the compatibility fusion-candidate factory.

Measured acceptance deviation:

- The initial audit expected 5,000-8,000 fewer runtime lines and named 4,800 lines/10% as its completion threshold. The final repository instead has 6,694 more runtime lines. This is not relabeled as reduction. The refactor deleted large duplicated bodies and improved ownership, but immutable schemas, pure compositors, compatibility projections, split scaffolding, stronger state models, and runtime diagnostics outweighed those deletions physically.
- The coordinating goal text says not to sacrifice correctness, compatibility, diagnostics, or readability to hit an arbitrary deletion target. After the independent dead-code sweep, no further high-confidence bulk deletion remains. Meeting the original numeric target now would require a new, high-risk redesign rather than evidence-backed cleanup.
- Large residual units remain: `pvzrl_env.py` is 9,966 lines, its reset state machine remains about 988 lines, both environment/SB3 step methods exceed 700 lines, and bridge Seed/UI and Reset remain 2,865 and 2,198 lines. `pvzrl_lifecycle.py` is now the 649-line pure predicate authority rather than a shadow copy. Numeric coercion and compatibility-heavy live-status emission remain distributed.

Live-game validation record and exact remaining commands:

The final-source install/startup and bounded environment checks above have now run, and the original installed recovery DLL has been restored and reverified. The commands below are retained as reproducibility evidence or remain required where explicitly marked.

| Live gate | Status |
| --- | --- |
| Final DLL load and bridge connection | PASS |
| Popup/menu/seed/gameplay entry | PASS |
| Observation, legality, wait, placement, invalid-action safety | PASS |
| Active-wave reset safety guard | PASS, safely blocked as designed |
| Real-loss retry/reset to clean gameplay | PASS, three TryAgain resets to clean four-slot boards; separate two-slot seed-selection flow also passed |
| Dedicated live fusion semantics and tile scope | PASS |
| Coach fusion scope through bare environment boundary | PASS |
| Protected Generalist Level-6 evaluation | PASS after correcting a retained mismatch: clean-start 547-step classified loss, 45/45 fusions, zero illegal/disagreement/bridge/reset failures, zero exit |
| Fixed Level-3 evaluation | BLOCKED as designed at current Level 6 |
| Fresh fixed training and resume | PENDING; required plants are not unlocked in the current profile |
| Win, timeout, reward/unlock, replay, and advancement traces | PENDING |
| Live game-backed GUI interaction | PENDING |
| Live bridge/step latency and rollout SPS | PENDING |

```powershell
$tag = Get-Date -Format 'yyyyMMdd_HHmmss'
$manualRoot = Join-Path 'runs/manual_phase8' $tag
New-Item -ItemType Directory -Force $manualRoot | Out-Null
$installed = 'Game Files/Mods/PvZRLBridge.dll'
$backup = Join-Path $manualRoot 'PvZRLBridge.recovery.dll'
Copy-Item -LiteralPath $installed -Destination $backup
.\scripts\build_bridge.ps1 -CopyToMods
Start-Process -FilePath '.\Game Files\PlantsVsZombiesRH.exe'
Test-NetConnection 127.0.0.1 -Port 32323
```

Observed: the installed DLL hashed to `f6be2f86...`, the bridge loaded, and port 32323 accepted connections. The preserved recovery DLL was restored after the bounded gate.

The following live commands have incompatible state preconditions and are not a sequential recipe. Reset/relaunch to the required clean gameplay, loss/retry, or seed-selection boundary before each command. In particular, the reset command requires a transition-safe loss/seed path; the production guard correctly blocks destructive reset during an active wave.

```powershell
python .\python\pvzrl_env.py --smoke-test --wait-for-board --wait-gameplay-ready --quick-wait --auto-select-seeds --seed-list SunFlower,SunFlower,Peashooter,Peashooter --start-sun 9990
python .\python\pvzrl_env.py --fusion-semantics-test --wait-for-board --wait-gameplay-ready --quick-wait --auto-select-seeds --seed-list SunFlower,SunFlower,Peashooter,Peashooter --start-sun 9990
python .\python\pvzrl_env.py --coach-fusion-scope-test --wait-for-board --wait-gameplay-ready --quick-wait --auto-select-seeds --seed-list SunFlower,SunFlower,Peashooter,Peashooter --start-sun 9990
python .\python\pvzrl_env.py --reset-state-machine-test --auto-select-seeds --seed-list SunFlower,SunFlower,Peashooter,Peashooter --quick-wait
python .\python\pvzrl_env.py --auto-select-seeds-test --episodes 2 --seed-list SunFlower,SunFlower,Peashooter,Peashooter --quick-wait
```

Observed for the bounded smoke/fusion/coach/reset checks: structured observations and legal actions, wait advancement, normal placement, safe invalid-action rejection, exact empty-source rejection, one tile-scoped Peashooter-plus-SunFlower fusion, control-plant preservation, and repeated loss-to-seed-to-clean-board resets. Dedicated incompatible/self/recursive recipe scenarios and the two-episode command remain part of the broader live matrix.

```powershell
$fixedRun = Join-Path $manualRoot 'fixed_fresh'
$resumeRun = Join-Path $manualRoot 'fixed_resume'
python .\python\train_ppo.py --run-mode fixed_train --config configs\ppo_sunflower_peashooter_wallnut_cherrybomb.json --run-dir $fixedRun --total-timesteps 512 --n-steps 128 --batch-size 64 --checkpoint-freq 128 --quick-wait --wait-gameplay-ready
python .\python\train_ppo.py --run-mode fixed_train --config configs\ppo_sunflower_peashooter_wallnut_cherrybomb.json --resume-model-path (Join-Path $fixedRun 'model.zip') --run-dir $resumeRun --total-timesteps 512 --n-steps 128 --batch-size 64 --checkpoint-freq 128 --quick-wait --wait-gameplay-ready
python .\python\train_ppo.py --level3-eval --target-level 3 --model-path python\runs\ppo_4slot_sunflower_peashooter_wallnut_cherrybomb_20260507_130623\model.zip --episodes 1 --seed-list SunFlower,Peashooter,WallNut,CherryBomb --plant-types 1,0,3,2 --fusion-policy none --tactical-masks --wallnut-tactical-mask --cherrybomb-tactical-mask --quick-wait --wait-gameplay-ready
```

These commands remain pending. They require the exact live bank `SunFlower,Peashooter,WallNut,CherryBomb`; the validation profile exposed only SunFlower and Peashooter at Level 6. The Level-3 command was run from an isolated directory and correctly failed fast with `blocked_reason=not_at_level3_specialist_start_state`. Once the prerequisites exist, fresh training must collect at least one rollout under `$fixedRun`, resume must advance a newly written model under `$resumeRun`, and fixed evaluation must complete one real terminal episode without writing the protected source model.

```powershell
python .\python\pvzrl_env.py --adventure-state-smoke --duration-seconds 180 --auto-select-seeds --seed-list SunFlower,SunFlower,Peashooter,Peashooter --quick-wait
$generalistRun = Join-Path $manualRoot 'generalist_level6'
python .\python\train_ppo.py --config configs\ppo_adventure_generalist_14slot_identity_v1.json --adventure-generalist-eval --model-path runs\ppo_adventure_generalist_14slot_identity_v1_20260627_172727\checkpoints\ppo_pvz_370000_steps.zip --run-dir $generalistRun --live-status-path (Join-Path $generalistRun 'live_status.json') --adventure-start-level 6 --max-adventure-levels 1 --max-attempts-per-level 1 --advance-on-wins 1 --quick-wait --wait-gameplay-ready
python .\python\pvzrl_gui.py --live-status-path runs\live_status.json
```

Observed after the recursive-identity correction: the clean-boundary Generalist command loaded 701 model actions/4,297 observations, entered wave 0 with 201 currently legal actions, and completed a 547-step real loss with 45/45 successful fusions, no illegal action, no bridge/reset error, no mask/bridge disagreement, and process exit zero. The longer Adventure state smoke, live game-backed GUI, and win/timeout/reward/unlock/replay/advancement coverage remain pending; each must be retained as an inspectable trace before those gates can be accepted.

If any live check fails, stop the game, wait for bridge port 32323 to close, restore the preserved install, and verify its session backup hash before further diagnosis:

```powershell
$backupHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $backup).Hash
Get-Process PlantsVsZombiesRH -ErrorAction SilentlyContinue | Stop-Process
$deadline = (Get-Date).AddSeconds(15)
do {
    $listening = (Test-NetConnection 127.0.0.1 -Port 32323 -WarningAction SilentlyContinue).TcpTestSucceeded
    if ($listening) { Start-Sleep -Milliseconds 250 }
} while ($listening -and (Get-Date) -lt $deadline)
if ($listening) { throw 'Bridge listener still open; do not overwrite the installed DLL.' }
Copy-Item -LiteralPath $backup -Destination $installed -Force
$restoredHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $installed).Hash
if ($restoredHash -ne $backupHash) { throw "Installed bridge restore hash mismatch: $restoredHash != $backupHash" }
```

Final rollback boundaries added in Phase 8: `34dabc1` bounded hard close; `409138a` dead GUI/Python/C# compatibility paths; `80376cf` expired helper adapters; `8c06886` empty-board-aware live fusion diagnostics; `62b197b` bare live coach environment support; `c7d38bc` corrected live recursive fusion identities and bridge-free setup regression. The earlier phase rollback commits remain listed in their phase sections.
