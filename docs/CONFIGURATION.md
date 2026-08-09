# PvZRL Configuration

PvZRL supports one product path: Adventure Generalist training and evaluation.

The canonical tracked Generalist configuration is:

```text
configs/ppo_adventure_generalist_14slot_identity_v1.json
```

Streamer Mode V1 is a Generalist-training overlay with a safe tracked example:

```text
configs/streamer_v1.example.json
```

The example contains environment-variable names but no Twitch credentials. See `docs/STREAMER_MODE.md` before a live run.

## Run modes

Exactly two run modes are accepted:

```text
adventure_generalist_14slot_train
adventure_generalist_14slot_eval
```

Use the matching CLI selectors:

```text
--adventure-generalist-train
--adventure-generalist-eval
```

`--run-mode` accepts the same two values. Unsupported mode strings fail during resolution; there is no fallback into another trainer or evaluator.

`--streamer-v1` does not add a third run mode. It is valid only with `adventure_generalist_14slot_train`; its cycle manager calls the existing Generalist trainer and evaluator for `STREAM_TRAIN` and `EVALUATE` phases.

## Resolution precedence

Every field resolves in this order:

```text
explicit CLI value > JSON configuration > Generalist mode default > global default
```

An argparse value of `None` means not supplied. Explicit falsey values still override JSON. `ConfigResolver` records the winning source for inspectable `resolved_config.json` output.

`ResolvedRunConfig` is the immutable typed view. `train_ppo.build_config()` is the flat dictionary/artifact adapter used by runtime consumers. Do not add another resolver.

## Permanent model contract

The following are not tunable run variants:

| Field | Required value |
| --- | --- |
| `model_family` | `ppo_adventure_generalist_14slot_identity_v1` |
| `action_space_mode` | `adventure_14slot_identity` |
| `max_seed_slots` | `14` |
| action count | `701` |
| wait/placement range | `0` / `1..700` |
| `action_decoder_version` | `seedslot14x50_plus_wait_v1` |
| `observation_version` | `adventure_14slot_identity_v1` |
| observation shape | `(4297,)` |
| board geometry | 5 rows x 10 columns |

The initial identity loadout is:

```json
{
  "seed_list": ["SunFlower", "SunFlower", "Peashooter", "Peashooter"],
  "initial_loadout": ["SunFlower", "SunFlower", "Peashooter", "Peashooter"]
}
```

Duplicates and order are intentional. The current live loadout may grow with unlocks, but it never resizes the policy action or observation spaces.

## Important configuration groups

### Training

- `total_timesteps`, `learning_rate`, `n_steps`, `batch_size`, `gamma`, `gae_lambda`, `ent_coef`, `clip_range`
- `checkpoint_freq`, `run_dir`
- `resume_model_path` for explicit compatible continuation

Fresh training has no resume source. Resume requires a compatible Generalist model and writes new artifacts to the requested destination; it never overwrites the source model.

### Adventure lifecycle and progression

- `adventure_start_level`
- `max_adventure_levels`, `max_attempts_per_level`, `advance_on_wins`
- `adventure_soft_max_steps`, `adventure_hard_max_steps`
- `adventure_final_wave_extension`
- `adventure_generalist_strict_startup_validation`
- `adventure_replay_cleared_levels`
- frontier/recent/maintenance sample probabilities and frontier win-streak requirement

Startup validation compares wrapper, bridge, profile, UI, seed-selection, and gameplay identity. Configuration cannot authorize forcing a mismatched level.

### Seed curriculum

- `unlock_aware_seed_curriculum`
- `seed_curriculum`
- `unlock_introduction_delay`
- `new_plant_min_inclusion_prob`
- `core_seed_names` (default `SunFlower,Peashooter`; unique CORE identities)
- `new_unlock_guarantee_episodes` (default `4`; completed eligible inclusions per new unlock)
- `infer_capacity_from_unlocks`
- `allow_weak_unlocked_capacity_fallback`
- `randomize_seed_order`

The protected checkpoint starts with ordered duplicate identities. Change order/randomization only when the current model contract and experiment explicitly permit it.

### Fusion and tactical behavior

- `fusion_policy`: `none`, `observe`, or `scripted` (`assist` aliases scripted)
- `fusion_action_mask_enabled`
- `enable_board_plant_identity`
- fusion chain/discovery/repeat rewards
- fusion/later-plant/coach curriculum toggles and probabilities
- `tactical_masks`, `wallnut_tactical_mask`, `cherrybomb_tactical_mask`

The bridge/game is final fusion authority. Configuration may enable prediction, masks, scripted selection, or reward accounting; it cannot invent result identities.

### Watchdog and diagnostics

- `enable_action_watchdog`
- `action_timeout_seconds`
- `save_freeze_debug_bundle`
- `action_diagnostics_path`, `freeze_debug_dir`
- `live_status_path`

Status and diagnostics paths should normally live beneath the run directory. Writes are bounded/atomic where implemented.

### Coaches

Human coach and crowd coach use local command/log paths. Human coach takes precedence when both are enabled. Crowd coach is dry-run unless apply is explicitly enabled. Intervention logging and fusion probes remain local.

### Streamer Mode V1

Streamer V1 owns its intervention collector and therefore rejects simultaneous legacy human-coach or crowd-coach overrides.

| Key | Default | Meaning |
| --- | --- | --- |
| `streamer_v1_enabled` | `false` | Enable the Streamer cycle overlay. |
| `streamer_platform` | `twitch` | `twitch` EventSub or deterministic `mock` source. |
| `streamer_baseline_checkpoint` | required | Repository-relative or runtime path to a compatible Generalist baseline. |
| `streamer_intervention_interval_seconds` | `2.0` | Monotonic interval between opportunities; at most one viewer action per opportunity. Valid range: `0.1..3600` seconds. |
| `streamer_command_ttl_seconds` | `10.0` | Time before a queued parsed command expires. |
| `streamer_command_queue_capacity` | `256` | Bounded FIFO capacity; a full queue rejects the newest command. |
| `streamer_message_max_chars` | `256` | Maximum parsed command length. |
| `streamer_policy_steps_per_cycle` | `25000` | Policy-owned PPO transitions required before evaluation. |
| `streamer_checkpoint_policy_steps` | `5000` | Safe post-optimizer CURRENT save interval. |
| `streamer_evaluation_episodes` | `50` | Exact autonomous completed episodes per evaluation. |
| `streamer_max_cycles` | `0` | Cycle limit; zero is unlimited. |
| `streamer_endurance_hours` | `0.0` | Elapsed-time target checked at safe cycle boundaries; zero is unlimited. |
| `streamer_bc_enabled` | `true` | Enable masked BC from proven viewer executions during training. |
| `streamer_bc_coefficient` | `0.01` | Non-negative BC contribution to the minimized PPO loss. |
| `streamer_demonstration_capacity` | `4096` | Maximum retained observation/mask/action demonstrations. |
| `streamer_demonstration_persist_every` | `512` | New demonstrations accumulated before the bounded `.npz` is atomically rewritten; cannot exceed capacity. |
| `streamer_bc_batch_size` | `32` | Demonstration samples per BC loss. |
| `streamer_bc_update_frequency` | `1` | PPO training-call interval for BC. |
| `streamer_bc_min_demonstrations` | `8` | Minimum retained demonstrations before BC starts. |
| `streamer_mock_script` | empty | JSONL path required for `mock`; not consulted for Twitch. |

Streamer mode defaults `n_steps=500` and `batch_size=50` when neither CLI nor JSON overrides them. `streamer_policy_steps_per_cycle` must be divisible by `n_steps`, and `n_steps` must be divisible by `batch_size`; this prevents Stable-Baselines3 from overshooting an exact cycle boundary.

The default Twitch environment-variable-name settings are:

```text
streamer_twitch_client_id_env        = PVZRL_TWITCH_CLIENT_ID
streamer_twitch_access_token_env     = PVZRL_TWITCH_USER_ACCESS_TOKEN
streamer_twitch_broadcaster_id_env   = PVZRL_TWITCH_BROADCASTER_USER_ID
streamer_twitch_user_id_env          = PVZRL_TWITCH_EVENTSUB_USER_ID
streamer_viewer_hash_secret_env      = PVZRL_TWITCH_VIEWER_HASH_SECRET
```

Only names belong in configuration; secret values stay in the process environment. Client ID, user access token, broadcaster ID, and a 16–4,096-byte viewer-hash secret are required for Twitch. EventSub user ID is optional and otherwise resolves from token validation. The token must include `user:read:chat`.

The baseline is actual-loaded and checked against the permanent contract before any evaluation or training. Its SHA-256 is pinned to the experiment directory. Reusing an experiment directory with another baseline, a changed baseline-evaluation protocol, missing CURRENT/state, incompatible demonstration data, or contradictory cycle metadata fails closed. CURRENT and BEST use immutable hash-addressed generations plus atomic role records; their conventional `model.zip` files are repairable aliases and only the active and immediately prior generations are retained.

Adventure progression remains sequential across the baseline evaluation, train cycle, and current evaluation. `adventure_start_level` is the initial expected live level; each phase persists and hands forward `next_adventure_level`. Strict profile/UI/bridge validation remains authoritative, including on resume. Evaluations starting at different Adventure levels are recorded as `UNKNOWN` comparisons and cannot promote BEST.

## Command forms

Check dependencies:

```powershell
python .\python\train_ppo.py --check-deps
```

Fresh training:

```powershell
python .\python\train_ppo.py `
  --config configs\ppo_adventure_generalist_14slot_identity_v1.json `
  --adventure-generalist-train `
  --run-dir runs\manual_generalist\fresh
```

Resume into a separate destination:

```powershell
python .\python\train_ppo.py `
  --config configs\ppo_adventure_generalist_14slot_identity_v1.json `
  --adventure-generalist-train `
  --resume-model-path runs\manual_generalist\fresh\model.zip `
  --run-dir runs\manual_generalist\resume
```

Evaluate the protected model:

```powershell
python .\python\train_ppo.py `
  --config configs\ppo_adventure_generalist_14slot_identity_v1.json `
  --adventure-generalist-eval `
  --model-path runs\ppo_adventure_generalist_14slot_identity_v1_20260627_172727\checkpoints\ppo_pvz_370000_steps.zip `
  --run-dir runs\manual_generalist\eval
```

Bridge-free metadata and actual CPU model-load check:

```powershell
python .\python\train_ppo.py `
  --config configs\ppo_adventure_generalist_14slot_identity_v1.json `
  --metadata-dry-run `
  --model-path runs\ppo_adventure_generalist_14slot_identity_v1_20260627_172727\checkpoints\ppo_pvz_370000_steps.zip
```

Use `python .\python\train_ppo.py --help` as the executable authority for optional tuning flags.

Start Streamer V1 from the safe example:

```powershell
python .\python\train_ppo.py `
  --config .\configs\streamer_v1.example.json `
  --streamer-v1 `
  --run-dir .\runs\streamer_v1\live `
  --live-status-path .\runs\streamer_v1\live\live_status.json `
  --quick-wait --wait-gameplay-ready
```

Set Twitch credential values in the named environment variables first. For a one-cycle local smoke, add `--streamer-max-cycles 1`. To resume, use the same experiment directory and baseline; the Streamer state/CURRENT records replace the ordinary `--resume-model-path` handoff.

## Artifacts

Each run should contain its own resolved configuration, model metadata, logs, progress metrics, checkpoints/model, live status, and diagnostics as applicable. A Streamer experiment additionally owns `streamer_state.json`, `streamer_cycles.jsonl`, BASELINE/CURRENT/BEST records, per-cycle train/evaluation directories, compact Streamer events, and the bounded demonstration `.npz`. Most run artifacts are intentionally ignored. Treat models, profiles, installed bridge DLLs, demonstrations, and live logs as user data.

Before protected live work, record hashes for the source checkpoint/metadata, installed DLL, and profile. Confirm them again after the run.
