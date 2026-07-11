# PvZRL Refactor Execution Report

Status: in progress
Report started: 2026-07-11
Baseline commit: `0cbc4d90ff68b31f5a0fed92d7508243c1d0293f`

## Executive status

The repository-wide refinement is active. The initial audit has been checked against the live checkout by separate architecture, compatibility, bridge/concurrency, GUI/process, and performance investigations. Phase 0 behavior locks and benchmark tooling are complete; production behavior changes begin in Phase 1.

Automated baseline status is green. Live-game behavior has not yet been claimed as verified because the game/bridge were not running during baseline capture and the DLL installed under `Game Files/Mods` predates the current source build.

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

Final added/removed/net runtime lines, test lines, documentation lines, functions/classes removed, duplicate blocks consolidated, wrappers remaining, files split/deleted, and moved-versus-eliminated code will be recorded after Phase 8.

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

Rollback boundary: local commit subject `test: lock PvZRL refactor baseline`; its hash will be recorded in the final phase table.

Later phase results, benchmark comparisons, live verification, independent review findings, deferred work, remaining duplication/risks, and exact rollback commits will be appended rather than inferred in advance.
