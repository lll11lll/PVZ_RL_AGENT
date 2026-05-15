# PvZRL: Plants vs. Zombies Reinforcement Learning

**PvZRL** trains and evaluates reinforcement learning agents to play Plants vs. Zombies using **Stable-Baselines3 MaskablePPO**. The system integrates a live Unity IL2CPP game build with a C# MelonLoader bridge, a Python Gymnasium-style environment, and an interactive dashboard for experiment management.

## Project Status

PvZRL is an **active research and development project**. Core systems for fixed-level training, Adventure-mode progression, and specialist/generalist evaluation are functional. The **Adventure Generalist framework** (14-slot identity-aware policy) represents the current foundation for early-game progression and is actively evolving. Some subsystems, particularly curriculum logic and advanced unlock handling, remain experimental.

## Key Features

### Game Integration
- **Unity IL2CPP Bridge**: MelonLoader C# mod reads live game state and executes plant placements
- **TCP/JSON Protocol**: Language-neutral communication between game process and Python trainer
- **Live State Access**: Direct access to Board, Plant, Zombie, CardUI, and gameplay lifecycle objects

### Learning Infrastructure
- **Gymnasium Environment**: Standard Gym-style interface with `reset()` and `step()` 
- **Stable-Baselines3 MaskablePPO**: Multi-seed parallel training with legal action masks
- **Dynamic Action Spaces**: Fixed slots, dynamic slots, and identity-aware 14-slot modes
- **Legal Action Masking**: Enforces cooldowns, sun economy, cell occupancy, terrain rules, and Adventure constraints

### Observation and Reward
- **Structured State Encoding**: Global sun, wave counters, seed inventory, board state, lane threat signals, and seed identity features
- **Extensive Reward Shaping**: Kill/wave bonuses, win/loss, lane coverage, threat response, economy management, and role positioning rewards
- **Episode Diagnostics**: Detailed per-step and per-episode telemetry (kills, waves, plants placed, rewards by component, terminal reason)

### Training Modes
- **Fixed-Level Specialist Training**: Dedicated PPO agents for individual levels with fixed seed sets and action spaces
- **Adventure Evaluation**: Load trained models and run inference through Adventure progression with unlock tracking
- **Adventure Generalist Training** (14-slot): Scratch training for a single policy that adapts to changing seed packets and loadouts during early Adventure

### Model Management
- **Compatibility Checking**: Metadata-driven validation ensures models load only into compatible configurations
- **Versioning**: Action decoder, observation encoding, and action space mode explicitly tracked
- **Run Artifacts**: Configuration, action maps, episode metrics, checkpoints, and live diagnostics per training run

### Dashboard and Diagnostics
- **Tkinter GUI**: Launch training, evaluation, or Adventure runs with preset profiles
- **Live Status Monitoring**: Real-time episode progress, reward tracking, legal action validation, threat detection
- **Run Browser**: View saved runs, inspect configurations, and download model checkpoints
- **Diagnostics Panel**: Cooldown state, board occupancy, lane threat profiles, unlock status, seed inventory

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Unity IL2CPP Game                     │
│         (PlantsVsZombiesRH, MelonLoader loader)        │
└──────────────────────┬──────────────────────────────────┘
                       │ TCP / JSON
┌──────────────────────▼──────────────────────────────────┐
│          MelonLoader C# Bridge Mod                      │
│   (Board state reads, plant placement commands)         │
└──────────────────────┬──────────────────────────────────┘
                       │ localhost:32323
┌──────────────────────▼──────────────────────────────────┐
│     Python Gymnasium Environment (pvzrl_env.py)        │
│  (Reset, step, observe, reward, legal mask generation) │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  SB3 MaskablePPO Adapter (pvzrl_sb3.py)                │
│  (Observation encoding, action decoding, masking)       │
└──────────────────────┬──────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
┌───────▼────────────┐     ┌──────────▼──────────────┐
│ Training Loop      │     │ Dashboard + Diagnostics│
│ (train_ppo.py)     │     │ (pvzrl_gui.py)         │
│ - Config loading   │     │ - Run launcher         │
│ - Metadata setup   │     │ - Live monitoring      │
│ - PPO training     │     │ - Artifact browser     │
│ - Checkpoint save  │     │ - Episode inspection   │
└────────────────────┘     └────────────────────────┘
```

## Quick Start

### Prerequisites

- **Python 3.8+** with `gymnasium`, `numpy`, `stable-baselines3`, `sb3-contrib`
- **Plants vs. Zombies RH** (shipped IL2CPP build in `Game Files/`)
- **MelonLoader** installed and working with the game
- **.NET SDK** or compatible Roslyn compiler for building the C# bridge

### Building the Bridge

```powershell
.\scripts\build_bridge.ps1 -CopyToMods
```

This compiles the C# MelonLoader mod (`PvZRLBridge.dll`) and copies it to `Game Files/Mods`.

### Verifying Game Integration

1. Start the game: `Game Files/PlantsVsZombiesRH.exe`
2. Navigate to Adventure mode and select plants
3. Once the board is visible, run:

```powershell
python .\python\pvzrl_env.py --wait-for-board --wait-gameplay-ready --quick-wait
```

The bridge is ready if the board state prints successfully.

### Training a Specialist Model

Run the GUI dashboard:

```powershell
python .\python\pvzrl_gui.py --live-status-path runs\live_status.json
```

From the **Fixed-Level Training** tab:
- Select preset profile (e.g., "2-slot stable: SunFlower, Peashooter")
- Adjust hyperparameters if needed
- Click **Launch Training**

Alternatively, from the command line:

```powershell
python .\python\train_ppo.py `
  --run-mode fixed_train `
  --config configs/ppo_sunflower_peashooter.json `
  --total-timesteps 25000
```

### Training Adventure Generalist (14-slot)

```powershell
python .\python\pvzrl_gui.py --live-status-path runs\live_status.json
```

Select the **Adventure Generalist** tab and click **Launch Training**. The policy will train on early Adventure levels with dynamic seed unlocks and loadout changes.

### Evaluating a Model

Use the **Adventure Eval** tab in the GUI to load a trained model and run inference through Adventure mode. The system will:
- Load the model and validate metadata compatibility
- Handle seed selection and progression automatically
- Track unlocks and advancement
- Save episode metrics to the run directory

## Directory Structure

```
PvZRL/
├── README.md                    # This file
├── requirements-ppo.txt         # Python dependencies
│
├── scripts/
│   └── build_bridge.ps1        # C# bridge compiler
│
├── configs/
│   ├── plant_registry.json
│   ├── model_schedule.json
│   ├── ppo_sunflower_peashooter.json
│   ├── ppo_sunflower_peashooter_wallnut_cherrybomb.json
│   └── ppo_adventure_generalist_14slot_identity_v1.json
│
├── docs/
│   ├── PvZRL_Learning_Guide.md          # Architecture & concepts
│   ├── PvZRLBridge.md                   # Bridge integration guide
│   ├── ImplementationProcess.md         # Development history
│   ├── PPO_Readiness.md                 # Training readiness checklist
│   ├── ADVENTURE_GENERALIST.md          # 14-slot policy design
│   ├── AdventureAndFusionRoadmap.md    # Roadmap & future work
│   ├── PPO_RL_Explained.md              # RL concepts
│   └── Maintainability_Refactor_Roadmap.md
│
├── python/
│   ├── train_ppo.py                     # Main training entrypoint
│   ├── pvzrl_env.py                     # Gymnasium environment
│   ├── pvzrl_sb3.py                     # SB3 MaskablePPO adapter
│   ├── pvzrl_adventure.py               # Adventure evaluation utilities
│   ├── pvzrl_adventure_generalist.py    # 14-slot training environment
│   ├── pvzrl_gui.py                     # Tkinter dashboard
│   ├── pvzrl_action_space.py            # Action space definitions
│   ├── pvzrl_seed_inventory.py          # Seed inventory encoding
│   ├── pvzrl_model_metadata.py          # Compatibility checking
│   ├── pvzrl_model_router.py            # Model resolution
│   ├── pvzrl_fusion.py                  # Fusion mechanics (experimental)
│   ├── backfill_model_metadata.py       # Metadata tooling
│   ├── test_*.py                        # Unit and integration tests
│   └── runs/                            # Checkpoints from local training
│
├── runs/
│   ├── live_status.json                 # Current training metrics
│   ├── ppo_*/                           # Saved run artifacts
│   │   ├── config.json
│   │   ├── model_metadata.json
│   │   ├── action_map.txt
│   │   ├── command_used.txt
│   │   ├── episode_metrics.csv
│   │   ├── checkpoints/
│   │   └── tensorboard/
│   └── diagnostics/
│
└── Game Files/
    ├── PlantsVsZombiesRH.exe
    ├── PlantsVsZombiesRH_Data/
    │   ├── Managed/
    │   ├── Plugins/
    │   └── Resources/
    ├── MelonLoader/
    │   ├── net6/
    │   ├── Dependencies/
    │   └── Logs/
    └── Mods/
        └── PvZRLBridge/                 # Compiled bridge mod
```

## Core Components

### `pvzrl_env.py` — Gymnasium Environment
Implements the game abstraction layer:
- `reset()`: Connects to bridge, waits for board, initializes game state
- `step(action)`: Places plant or waits, collects reward, checks terminal conditions
- `observe()`: Reads structured board state (sun, plants, zombies, waves, cooldowns)
- `action_mask()`: Computes legal actions based on sun, cooldowns, occupancy, terrain
- Supports fixed-level, Adventure, and specialist modes

### `pvzrl_sb3.py` — SB3 Adapter
Bridges Gymnasium environment to Stable-Baselines3:
- Converts observations to fixed-shape numpy arrays
- Implements action masking for MaskablePPO
- Decodes policy actions to game commands
- Adds identity features for generalist policies
- Supports observation versioning and action decoder routing

### `pvzrl_action_space.py` — Action Space Management
Defines and encodes action spaces:
- **Fixed mode**: Position-based, 2–50 slots × (row × col) + wait
- **Dynamic mode**: Expands/contracts with available slots
- **Adventure 14-slot Identity**: Permanent 701-action interface with slot identity observation
- Metadata-driven action decoder resolution
- Legal action mask structural helpers

### `pvzrl_adventure_generalist.py` — Generalist Training
Specialized training environment for 14-slot Adventure policy:
- Wraps fixed environment with Adventure-aware reset/episode logic
- Manages seed unlocks and loadout changes between episodes
- Implements frontier and curriculum progression
- Tracks checkpoints and unlock history
- Generates diagnostics for Adventure-specific events

### `train_ppo.py` — Training Entrypoint
Orchestrates training loop:
- Config loading and resolution
- Model metadata creation and validation
- SB3 environment setup and observation/action routing
- MaskablePPO training with checkpointing
- Run artifact organization and logging

### `pvzrl_gui.py` — Dashboard
Tkinter-based control center:
- **Fixed-Level Training**: Preset profiles, hyperparameter editing
- **Adventure Evaluation**: Model loading, progression tracking, diagnostics
- **Adventure Generalist**: Scratch training launch and monitoring
- **Live Status Panel**: Real-time episode metrics, reward components, threat state
- **Run Browser**: Artifact inspection, checkpoint download
- **Diagnostics View**: Cooldown state, board occupancy, lane analysis

### `pvzrl_model_metadata.py` — Model Compatibility
Ensures correct model-environment pairing:
- Metadata schema: action count, action space mode, seed list, observation version, decoder version, max slots
- Compatibility checking: Validates loaded models against environment configuration
- Migration support: Handles legacy config formats
- Detailed error reporting for incompatibilities

## Supported Plants and Mechanics

### Current Plants
- **SunFlower** (ID 1): Sun generator, 50 cost, 7.5s cooldown
- **Peashooter** (ID 0): Single-lane attacker, 100 cost, 7.5s cooldown
- **WallNut** (ID 3): Defensive blocker, 50 cost, 24s cooldown
- **CherryBomb** (ID 2): Area damage, variable cost, 24s cooldown

### Supported Actions
- **Wait**: No action, advance game state
- **Plant Placement**: Select plant from seed inventory and place at valid (row, column)

### Action Masking Rules
- **Cooldown**: Plant action masked if seed packet is on cooldown
- **Sun Cost**: Plant action masked if current sun < plant cost
- **Occupancy**: Plant action masked if target cell is occupied
- **Terrain**: Plant action masked if cell is invalid (e.g., off-board, inaccessible)
- **Adventure Constraints**: Slot identity and loadout changes impact available actions

## Observation and Reward

### Observation Components
- **Global state**: Current sun, wave number, episode steps
- **Board state**: 5×10 grid with plant IDs, zombie positions, health values
- **Seed inventory**: Availability, cooldown state, cost of each slot (fixed or dynamic)
- **Seed identity** (generalist): Explicit plant ID for each active slot
- **Lane threats**: Nearest zombie position, count, health per lane
- **Lifecycle**: Game status (active, win pending, loss pending, reset pending)

### Reward Shaping
Rewards are composed of multiple weighted components:
- **Kill reward**: Bonus per zombie defeated
- **Wave reward**: Bonus per wave cleared
- **Win/loss reward**: Terminal event bonuses
- **Illegal penalty**: Cost for attempted invalid actions
- **Threat response reward**: Bonus for defending threatened lanes
- **Lane coverage reward**: Bonus for having defense in all threatened rows
- **Economy reward**: Bonus for timely sun generation and defense prioritization
- **Position reward**: Bonus for correct plant role positioning (SunFlower safe, Peashooter defensive)
- Additional specialized rewards for WallNut/CherryBomb tactics and Adventure-specific progression

Reward coefficients are config-driven and can be tuned per experiment.

## Running Tests

Unit and integration tests validate core behaviors:

```powershell
# Adventure Generalist schema and curriculum
python .\python\test_adventure_generalist_14slot_identity.py

# Adventure timeout and corruption handling
python .\python\test_adventure_timeout_semantics.py
python .\python\test_adventure_corruption_trackers.py

# Model metadata compatibility
python .\python\test_model_metadata_compatibility.py
```

## Configuration

Training and evaluation are configured via JSON config files in `configs/`:

```json
{
  "policy": "MlpPolicy",
  "total_timesteps": 25000,
  "learning_rate": 0.0003,
  "n_steps": 512,
  "batch_size": 64,
  "gamma": 0.99,
  "gae_lambda": 0.95,
  "ent_coef": 0.01,
  "clip_range": 0.2,
  "max_steps": 1000,
  "step_seconds": 0.05,
  "game_speed": 4.0,
  "start_sun": 500,
  "seed": 12345,
  "auto_select_seeds": true,
  "seed_list": "SunFlower,Peashooter",
  "plant_types": [1, 0],
  "checkpoint_freq": 5000
}
```

Key config fields:
- **SB3 hyperparameters**: `learning_rate`, `n_steps`, `batch_size`, `gamma`, `gae_lambda`, `ent_coef`, `clip_range`
- **Game settings**: `max_steps`, `step_seconds`, `game_speed`, `start_sun`, `board_timeout`, `gameplay_ready_timeout`
- **Action/observation schema**: `action_space_mode`, `seed_list`, `plant_types`, `action_decoder_version`, `observation_version`
- **Reward weights**: Per-component coefficients for shaping
- **Checkpointing**: `checkpoint_freq` in timesteps

See individual configs in `configs/` for complete examples.

## Advanced Topics

### Action Space Modes
- **fixed**: Traditional position-based actions, 101–201 actions depending on slots
- **dynamic_14**: Dynamic slot allocation up to 14 plants, 751 actions
- **adventure_14slot_identity**: Permanent 14-slot with explicit slot identity features, 701 actions

### Observation Versions
- **default**: Global state + board + seed availability (backward compatible)
- **adventure_14slot_identity_v1**: Adds fixed 14-slot seed identity encoding for generalist policies

### Seed Inventory Encoding
- **v2**: Compact fixed-width representation of available seeds with costs and cooldown status
- **adventure_identity**: Extended with 14-slot identity features (plant type per slot)

### Legal Action Generation
The bridge computes a base legal action set. Python applies additional safety filters:
- Removes actions for slots that are not in the current loadout (Adventure)
- Masks duplicate slots consistently
- Validates cell bounds and terrain from board state

### Adventure Progression and Unlocks
- **Frontier tracking**: Maintains the highest cleared level and unlocked plants
- **Seed curriculum**: Dynamically updates available seed packets based on unlocks
- **Loadout sampling**: May select subset of unlocked plants or enforce curriculum progression
- **Terminal conditions**: Win/loss detected, Adventure menus handled, progression recorded

### Fusion Mechanics (Experimental)
Early-stage fusion support:
- `pvzrl_fusion.py` defines fusion policies: `none`, `observe`, `scripted`, `assist`
- Fusion rules and candidate validation
- Diagnostics tracking fusion attempts and outcomes
- Currently not integrated into main training loop; reserved for future work

## Documentation

Detailed guides are in `docs/`:

- **[PvZRL_Learning_Guide.md](docs/PvZRL_Learning_Guide.md)**: Conceptual overview of architecture and project goals
- **[PvZRLBridge.md](docs/PvZRLBridge.md)**: Bridge build, proof-of-control, board readiness, and integration checklist
- **[PPO_Readiness.md](docs/PPO_Readiness.md)**: Training validation steps and readiness criteria
- **[ADVENTURE_GENERALIST.md](docs/ADVENTURE_GENERALIST.md)**: 14-slot policy design, curriculum, metadata, and implementation details
- **[ImplementationProcess.md](docs/ImplementationProcess.md)**: Development history and decision rationale
- **[AdventureAndFusionRoadmap.md](docs/AdventureAndFusionRoadmap.md)**: Planned work and future directions

## Known Limitations

- **UI-driven progression**: Adventure mode relies on detecting end-game screens and menu navigation; some edge cases may not be handled
- **Real-time synchronization**: Game runs at real speed; fast action sequences may experience timing sensitivity
- **IL2CPP metadata dependency**: Bridge relies on Unity's IL2CPP metadata and may break across game updates
- **Fusion mechanics**: Only partially implemented; not yet integrated into main training pipelines
- **Curriculum learning**: Current progression logic is conservative and scaffolded; advanced replay and non-frontier sampling are experimental

## Contributing

This is an active research project. Areas for contribution:

- **Curriculum refinement**: Improved seed unlock and loadout sampling strategies
- **Reward tuning**: Specialized coefficients for different game phases or plant combinations
- **Generalization**: Multi-level or multi-plant-set training beyond Adventure Generalist
- **Fusion integration**: Complete fusion mechanics and training support
- **Bridge robustness**: Better error handling and edge case detection in IL2CPP read/write operations
- **Testing**: Additional integration tests and edge case validation

## License

See LICENSE file (if present) for project licensing details.

## References

- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/
- **Gymnasium**: https://gymnasium.farama.org/
- **MelonLoader**: https://melonwiki.xyz/
- **Plants vs. Zombies**: Original game by PopCap Games

---

**Last Updated**: May 2026  
**Project Status**: Active Development
