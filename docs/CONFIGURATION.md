# PvZRL Configuration

PvZRL supports one product path: Adventure Generalist training and evaluation.

The canonical tracked configuration is:

```text
configs/ppo_adventure_generalist_14slot_identity_v1.json
```

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

## Artifacts

Each run should contain its own resolved configuration, model metadata, logs, progress metrics, checkpoints/model, live status, and diagnostics as applicable. Most run artifacts are intentionally ignored. Treat models, profiles, installed bridge DLLs, and live logs as user data.

Before protected live work, record hashes for the source checkpoint/metadata, installed DLL, and profile. Confirm them again after the run.
