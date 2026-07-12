# PvZRL Repository Refinement Plan

Status: in progress
Plan date: 2026-07-11
Verified baseline commit: `0cbc4d90ff68b31f5a0fed92d7508243c1d0293f`

## Objective and boundaries

Refine the working local PvZRL application across Python, the MelonLoader bridge, configuration, diagnostics, and Tkinter without changing supported behavior or checkpoint semantics. The work must reduce duplicated rule implementations and unnecessary hot-path work while improving lifecycle, concurrency, testing, and ownership.

This is not a greenfield rewrite. It does not add a website, web service, public stream adapter, remote game-session service, or production networking layer. Models, checkpoints, datasets, recordings, assets, generated runs, and recovery points are protected.

The supplied `PvZRL Internal Architecture Refinement Plan` is the initial roadmap. Findings below are verified against the current checkout rather than copied forward as assumptions.

## Verified ownership map

| Area | Current owner and boundary |
| --- | --- |
| Game state | Plants vs. Zombies owns authoritative plants, zombies, waves, cooldowns, sun, progression, and spawning. Unity objects are read or mutated only from the bridge's Melon `OnUpdate` thread. |
| Bridge request state | `src/PvZRLBridge/BridgeMod.cs` owns the localhost TCP server, request queue, Unity-thread dispatch, configuration, observation DTOs, UI/seed caches, placement, fusion, reset, and speed control. |
| Base environment | `python/pvzrl_env.py` owns bridge calls, lifecycle/reset acceptance, Python legality safeguards, rewards, corruption detection, fusion accounting, and terminal classification. |
| Action decisions | `python/pvzrl_actions.py` owns immutable policy/legacy/bridge action intents, pure Python legality decisions, frame/config cache proofs, and structured execution results. The base environment owns the cache and the bridge remains final authority. |
| RL adapter | `python/pvzrl_sb3.py` owns numeric observation encoding, policy-action translation, MaskablePPO masks, coach arbitration, watchdog diagnostics, episode counters, and summaries. |
| Adventure | `python/pvzrl_adventure.py` owns inference progression and transitions. `python/pvzrl_adventure_generalist.py` owns training curriculum, replay/frontier decisions, unlock state, loadouts, and strict startup identity. |
| GUI | `python/pvzrl_gui.py` owns Tk state, command construction, subprocess control, log display, live-status reading/rendering, and local moderation UI. It does not own gameplay state. |
| Coach inputs | `python/pvzrl_human_coach.py`, `python/pvzrl_stream_coach.py`, and `python/pvzrl_assisted_coach.py` own local parsing, validation, queueing, aggregation, and intervention records. Human coach precedence is intentional. |
| Configuration | CLI and JSON flow through `ConfigResolver` into a deeply immutable typed `ResolvedRunConfig`; `build_config()` provides the flat compatibility adapter for runtime dictionaries, metadata, bridge configuration, and GUI-launched commands. |
| Artifacts | Trainer and Adventure writers emit live status, episode JSONL/CSV, progress, TensorBoard, metadata, checkpoints, and watchdog bundles. GUI consumes a compatibility-heavy view of live status. |

## Verified execution paths

- Fixed training: `train_ppo.py -> build_config -> train -> DummyVecEnv/Monitor -> PvZMaskedPPOEnv -> MaskablePPO.learn -> ExperimentCallback`.
- Fixed evaluation: `train_ppo.py -> evaluate -> metadata validation -> MaskablePPO.load -> run_eval_episode`.
- Adventure evaluation: `train_ppo.py -> adventure_evaluate -> run_adventure_eval`, optionally through `ModelRouter`.
- Action: source-attributed policy intent -> one cached pure decision -> mask/coach/diagnostic/execution projections -> policy-to-legacy bridge conversion -> final bridge validation -> structured result -> reward and episode accounting.
- Fusion: immutable recipe/runtime-only compatibility registry -> source-attributed fusion intent -> one validation/execution adapter -> bridge mouse/reflection execution -> shared tile-scope contract -> event-ID-deduplicated diagnostics and reward accounting.
- Plant deletion remains outside the policy/manual action surface. Destruction is limited to reset, stale-object cleanup, and board recovery.

## Immutable compatibility contracts

Unless a separately documented defect correction has focused regression coverage, preserve all of the following.

### Actions

- Fixed mode: wait action `0`; placement `1 + slot * 50 + row * 10 + column`.
- Adventure identity mode: exactly 14 slots, 5 rows, 10 columns, 701 actions; wait `0`; placement range `1..700`; slot 13, row 4, column 9 is action 700.
- Dynamic-14 legacy mode retains its distinct wait/action ordering.
- Preserve action counts, ordering, decoder versions, wait identity, seed-slot order, and bridge action/response shapes.

### Observations and models

- Protected model family: `ppo_adventure_generalist_14slot_identity_v1`.
- Protected checkpoint: `runs/ppo_adventure_generalist_14slot_identity_v1_20260627_172727/checkpoints/ppo_pvz_370000_steps.zip`.
- Protected model contract: `Discrete(701)`, observation shape `(4297,)`, decoder `seedslot14x50_plus_wait_v1`, observation version `adventure_14slot_identity_v1`.
- Keep the older June 21 370k checkpoint inventoried as an additional compatibility control.
- Fixed control model: `python/runs/ppo_4slot_sunflower_peashooter_wallnut_cherrybomb_20260507_130623/model.zip`, with `Discrete(201)`, observation shape `(357,)`, and exact four-slot order.
- Preserve model metadata inference rules, checkpoint loading, resume semantics, and observation feature meanings.

### Legality and fusion

- Wait is always legal.
- Python execution safeguards remain even after mask reuse; cached decisions may be reused only for a demonstrably unchanged observation frame and configuration fingerprint.
- Bridge-side validation remains final runtime authority.
- Preserve occupied-cell fusion exposure, self-fusions, compatibility-only cases, tile scoping, recursive results, source attribution, costs, cooldowns, reasons, and one-time reward/count accounting.
- Required recursive identities include Repeater `1030`, Threepeater `1031`, Gatling Pea `1032`, and Twin SunFlower `1033`.

### Rewards and serialized artifacts

- Preserve reward coefficients, component meanings, `REWARD_COMPONENT_FIELDS`, episode totals, coach deltas, positive fusion cap semantics, and negative penalties.
- Preserve episode-summary keys, CSV headers/types, JSONL records, live-status compatibility keys, atomic writes, and timeout/freeze evidence.
- A live-status schema version may be added only while retaining current keys through an adapter window.

### Adventure and GUI

- Preserve frontier level, replay streaks, seed capacity, unlock state, progress files, strict startup validation, same-level replay, difficulty, and checkpoint continuation.
- Preserve GUI controls, presets, command serialization, moderation semantics, status health behavior except confirmed defects, and bounded process-stop behavior.

## Audit corrections and newly verified defects

1. The explicitly protected 370k checkpoint is the June 27 run, not merely any 370k artifact. Both June 27 and June 21 models currently load.
2. The bridge has no live `CheckMix` call. Fusion is mouse/reflection driven; attempted methods can report provisional success without observed mutation, and sun/cooldown may be charged before the final postcondition failure. Fusion execution changes are isolated from lifecycle work.
3. Four argparse defaults (`advance_on_wins`, `max_adventure_levels`, `max_attempts_per_level`, `adventure_start_level`) currently override JSON even when the user did not explicitly supply those flags. This is a confirmed configuration-precedence defect.
4. `lane_diagnostics()` references a nonexistent local `cherry_delayed_diag`, so four serialized delayed CherryBomb counters are always zero even when reward calculation observed delayed results.
5. GUI polling does reparse unchanged live status, but existing panel setters already suppress identical text-widget rewrites. The primary measured costs are JSON parsing, normalization/render computation, and unconditional `StringVar` updates.
6. GUI live health checks `blocked_reason` before file age, so arbitrarily old blocked payloads remain `BLOCKED_*` and suppress stale-writer warnings.
7. `model_schedule.json` paths are non-runnable from the repository root. The corresponding models live under `python/runs`, and the utility model's actual run/timestep differs from the stale schedule entry.
8. `docs/ADVENTURE_GENERALIST.md` contains a metadata-dry-run example that omits the duplicate generalist seed list and currently fails compatibility validation.

## Ordered phases and gates

| Phase | Scope | Current status | Required gate before next phase |
| --- | --- | --- | --- |
| 0 | Baseline, behavior locks, fixtures, benchmarks, plan/report | Complete | Full existing Python baseline, bridge build, checkpoint loads, new contract tests, benchmark record |
| 1 | Request deadlines/shutdown, seed cache warnings, shared JSONL tailing, GUI close, confirmed defects/dead code | Complete | Deterministic stale-request, shutdown, partial-record, GUI lifecycle tests; zero-warning bridge build; full baseline |
| 2 | Immutable plant registry, generated bridge fallbacks, typed resolved configuration, schedule repair | Complete | Registry parity, precedence matrix, model/action/observation metadata snapshots |
| 3 | Authoritative action intent/decision/result and fusion recipe/validation/execution pipeline | Complete | Exhaustive encode/decode, 701 masks, source parity, legal/illegal/self/recursive fusion contracts |
| 4 | Per-observation facts, reward/metric consolidation, watchdog/live-status I/O | Complete | Component reward replay at `1e-9`, identical external schemas, before/after benchmarks |
| 5 | Explicit episode/reset/progression/watchdog state and shadow lifecycle classification | Pending | Recorded lifecycle trace equivalence and live startup/win/loss/timeout/replay/Adventure checks |
| 6 | Move-only bridge decomposition, then observation/occupancy/lane optimization | Pending | Build after every split, DTO snapshots, zero warnings, live placement/fusion/reset/seed checks |
| 7 | GUI/process/status/coach separation and polling/log-drain optimization | Pending | Command snapshots, malformed/stale/unchanged status, bounded logs, callback/process lifecycle, interactive smoke |
| 8 | Full validation, independent reviews, final benchmarks/statistics/report | Pending | Every automated gate green; environment-specific omissions documented with exact commands |

## Risk controls and rollback boundaries

- One local commit per coherent phase when the phase is green. No push, merge, force operation, history rewrite, or branch change.
- Phase 0 is test/tool/documentation only and can be reverted without runtime impact.
- Phase 1 bridge lifecycle changes remain separate from fusion execution and observation optimization.
- Registry forwarding APIs remain during the compatibility window.
- Action/fusion wrappers remain until every source migrates and parity tests pass.
- Pure reward/metric extraction, aggregation consolidation, and I/O throttling are separate diff groups.
- Lifecycle state records begin in shadow mode; existing reset logic remains authoritative until traces match.
- Bridge file splitting is move-only before any optimization, and moved lines do not count as code reduction.
- GUI process, status reader, and rendering changes remain separable.
- Stop a phase at a failed compatibility gate, diagnose the regression, and restore the last green boundary before proceeding.

## Validation matrix

Always run from the repository root.

```powershell
python .\python\train_ppo.py --check-deps
python -m compileall -q python
python -m pytest -q
python .\python\test_adventure_corruption_trackers.py
python .\python\test_adventure_fusion_chain_diagnostics.py
python .\python\test_adventure_generalist_14slot_identity.py
python .\python\test_adventure_timeout_semantics.py
python .\python\test_fusion_compatibility.py
python .\python\test_fusion_reward_policy.py
python .\python\test_human_coach.py
python .\python\test_model_metadata_compatibility.py
python .\python\test_stream_coach.py
.\scripts\build_bridge.ps1
```

Checkpoint gates additionally run metadata dry-run plus actual `MaskablePPO.load` for the protected generalist and fixed control models. Live gates require installing the newly built bridge DLL, starting the bundled game, and exercising observation, placement, fusion, reset, transitions, short training/resume, and evaluation without modifying source checkpoints.

## Performance policy

Record median and p95 for action masks, observation encoding, reward calculation, fusion scans, environment/bridge steps where live execution is possible, live-status construction/serialization/write frequency, watchdog persistence, and unchanged GUI polling. Retain an optimization only when it simplifies ownership, removes demonstrated I/O/allocation pressure, or improves its target materially without an unexplained greater-than-5% regression elsewhere.

## Stop rule

Continue through the phases while compatibility gates pass and high-confidence duplication remains. Do not rewrite stable behavior merely to enlarge the diff or hit a line target. Completion requires the full final validation and independent reviews, not just recommendations or green unit tests.
