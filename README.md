# PvZRL

PvZRL is a local Windows reinforcement-learning and automation system for `PlantsVsZombiesRH`. A MelonLoader C# bridge exposes structured Unity state on localhost; Python builds a Gymnasium environment and trains/evaluates a MaskablePPO Adventure policy.

Adventure Generalist is the sole maintained training and evaluation path. Legacy fixed/specialist mode was removed during the repository refactor.

## Maintained contract

| Item | Value |
| --- | --- |
| Model family | `ppo_adventure_generalist_14slot_identity_v1` |
| Run modes | `adventure_generalist_14slot_train`, `adventure_generalist_14slot_eval` |
| Action mode | `adventure_14slot_identity` |
| Actions | 701: wait `0`, placement/fusion `1..700` |
| Decoder | `seedslot14x50_plus_wait_v1` |
| Observation | `adventure_14slot_identity_v1`, shape `(4297,)` |
| Board/capacity | 5x10, 14 identity slots |
| Initial loadout | `SunFlower,SunFlower,Peashooter,Peashooter` |
| Protected checkpoint | `runs/ppo_adventure_generalist_14slot_identity_v1_20260627_172727/checkpoints/ppo_pvz_370000_steps.zip` |

The action and observation spaces never resize with the current seed bank. Inactive slot blocks remain encoded and masked.

## Features

- Adventure Generalist fresh training, compatible resume, and evaluation
- strict startup identity validation across wrapper, bridge, profile, UI, and gameplay
- unlock-aware seed curriculum and pure frontier/replay progression reducer
- one cached legality pipeline shared by policy, coaches, diagnostics, and execution
- tile-scoped known and runtime-probed fusion with recursive identity/reward tracking
- additive reward components, episode metrics, TensorBoard, live status, and watchdog bundles
- local human coach, mock crowd coach, assisted-coach interventions, and Tk dashboard
- localhost-only Unity bridge with bounded request ownership/deadlines and reset/UI operations
- bridge-free Python tests, deterministic C# lifecycle harness, and synthetic benchmarks

This is not a hosted service or public streaming integration. The bridge binds to `127.0.0.1` by default.

## Requirements

- Windows
- a prepared Python environment with packages from `requirements-ppo.txt`
- the bundled `PlantsVsZombiesRH`/MelonLoader assemblies for bridge builds and live use
- workstation Visual Studio/.NET paths expected by the bridge scripts

Check the active Python environment:

```powershell
python .\python\train_ppo.py --check-deps
```

## Bridge-free verification

```powershell
python -m compileall -q python
python -m pytest -q
```

Validate and actually load the protected checkpoint on CPU:

```powershell
python .\python\train_ppo.py `
  --config configs\ppo_adventure_generalist_14slot_identity_v1.json `
  --metadata-dry-run `
  --model-path runs\ppo_adventure_generalist_14slot_identity_v1_20260627_172727\checkpoints\ppo_pvz_370000_steps.zip
```

Expected model contract: 701 actions, observation `(4297,)`, timestep 370000.

## Bridge build

```powershell
.\scripts\build_bridge.ps1
.\scripts\test_bridge_lifecycle.ps1
```

The build regenerates `src/PvZRLBridge/GeneratedPlantRegistry.cs` from `configs/plant_registry.json` and writes `src/PvZRLBridge/bin/Release/net6.0/PvZRLBridge.dll`.

`build_bridge.ps1 -CopyToMods` changes the installed game. Record the installed DLL hash first and restore or explicitly accept the result after protected live testing.

## Fresh training

Live commands are stateful. Select the intended profile and place the game at a clean Adventure boundary first.

```powershell
python .\python\train_ppo.py `
  --config configs\ppo_adventure_generalist_14slot_identity_v1.json `
  --adventure-generalist-train `
  --run-dir runs\manual_generalist\fresh `
  --quick-wait --wait-gameplay-ready
```

## Resume

Resume loads a compatible Generalist model and must write to a different destination:

```powershell
python .\python\train_ppo.py `
  --config configs\ppo_adventure_generalist_14slot_identity_v1.json `
  --adventure-generalist-train `
  --resume-model-path runs\manual_generalist\fresh\model.zip `
  --run-dir runs\manual_generalist\resume `
  --quick-wait --wait-gameplay-ready
```

## Evaluation

```powershell
python .\python\train_ppo.py `
  --config configs\ppo_adventure_generalist_14slot_identity_v1.json `
  --adventure-generalist-eval `
  --model-path runs\ppo_adventure_generalist_14slot_identity_v1_20260627_172727\checkpoints\ppo_pvz_370000_steps.zip `
  --run-dir runs\manual_generalist\eval `
  --quick-wait --wait-gameplay-ready
```

Evaluation performs masked inference and Adventure progression orchestration; it does not update or overwrite the source model.

## Dashboard

```powershell
python .\python\pvzrl_gui.py --live-status-path runs\live_status.json
```

The Tk dashboard exposes Generalist fresh train, resume, evaluation, runs/models, coach, and diagnostics workflows. It launches child processes and reads status; it does not directly control Unity gameplay.

## Architecture

```mermaid
flowchart LR
    CLI["Generalist CLI / JSON"] --> Resolver["ConfigResolver"]
    Resolver --> Runner["train_ppo"]
    Runner --> PPO["AdventureGeneralistTrainingEnv / evaluation"]
    PPO --> Env["PvZMaskedPPOEnv + PvZGymEnv"]
    Env --> Bridge["localhost C# bridge"]
    Bridge --> Game["Unity game"]
    Env --> Artifacts["metrics / status / diagnostics / checkpoints"]
    Artifacts --> GUI["Tk dashboard"]
    Coach["local coach files"] --> PPO
```

## Repository map

| Path | Purpose |
| --- | --- |
| `python/train_ppo.py` | CLI/config resolution, training, resume, evaluation, artifacts |
| `python/pvzrl_adventure_generalist.py` | Generalist training wrapper and curriculum effects |
| `python/pvzrl_generalist_progression.py` | Pure frontier/replay/unlock reducer |
| `python/pvzrl_adventure.py` | Adventure evaluation, transitions, timeout, status |
| `python/pvzrl_env.py` | Bridge client, base environment, execution/reset boundary |
| `python/pvzrl_sb3.py` | Gymnasium/MaskablePPO adapter |
| `python/pvzrl_action_space.py`, `python/pvzrl_actions.py` | 701-action contract and legality cache |
| `python/pvzrl_observation_*.py`, `python/pvzrl_seed_inventory.py` | 4,297-feature contract and facts |
| `python/pvzrl_fusion.py`, `python/pvzrl_rewards.py` | Fusion and reward composition |
| `python/pvzrl_gui*.py` | Tk dashboard, commands, process/status/coach support |
| `src/PvZRLBridge/` | MelonLoader bridge and Unity operations |
| `configs/ppo_adventure_generalist_14slot_identity_v1.json` | Canonical run configuration |
| `scripts/` | Bridge generation/build/harness/benchmark scripts |
| `docs/` | Architecture, configuration, operations, learning, and refactor evidence |

## Benchmarks

```powershell
python .\python\benchmark_hotpaths.py --samples 50 --rounds 5 --json-out runs\benchmarks\manual_python.json
.\scripts\benchmark_bridge_observation.ps1 -OutputPath runs\benchmarks\manual_bridge.json
```

The Python benchmark is synthetic. It does not measure Unity, socket latency, rollout SPS, or complete Tk rendering.

## Documentation

- [Agent operating guide](AGENTS.md)
- [Configuration](docs/CONFIGURATION.md)
- [Removal inventory](docs/FIXED_MODE_REMOVAL_INVENTORY.md)
- [Refactor plan](docs/REFACTOR_PLAN.md)
- [Refactor report](docs/REFACTOR_REPORT.md)

## Safety and evidence

- Treat models, profiles, installed DLLs, and ignored run artifacts as user data.
- Never overwrite a resume/evaluation source checkpoint.
- Do not force Adventure progression when identity evidence disagrees.
- The bridge/game is final placement/fusion authority.
- A metadata load is not a live-game test; a process start is not a rollout; a synthetic benchmark is not live performance.
- Preserve inspectable `blocked_reason`, rejection, timeout, reset, fusion, and reward evidence instead of summarizing failures away.
