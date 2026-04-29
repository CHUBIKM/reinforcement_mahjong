# Riichi Mahjong Engine with RL Training

A high-performance Riichi Mahjong rule engine with C++ core and Python reinforcement learning training support.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Python RL Training Layer                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ rl/trainer   │  │ rl/adapter   │  │   rl_policy (CLI)    │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │ pybind11 bindings
┌────────────────────────────┴────────────────────────────────────┐
│                    Python Compatibility Layer                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  engine.py (thin wrapper) + scoring.py                     │  │
│  └────────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │ pybind11
┌────────────────────────────┴────────────────────────────────────┐
│                      C++ Core Engine (Zero Python deps)          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────────┐ │
│  │  types   │ │   tile   │ │  hand    │ │        yaku        │ │
│  └──────────┘ └──────────┘ └──────────┘ └────────────────────┘ │
│  ┌──────────┐ ┌───────────────────────────────────────────────┐ │
│  │ scoring  │ │                  engine                       │ │
│  └──────────┘ └───────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

- **C++ Core** (`cpp/`): Modular rule engine with zero Python dependencies
  - `engine.hpp/cpp`: `RiichiEngine` class, game state machine
  - `yaku.hpp/cpp`: Yaku analysis (役种判定)
  - `scoring.hpp/cpp`: Han/fu calculation, point resolution
  - `hand_analysis.hpp/cpp`: Agari/tenpai detection, hand decomposition
  - `tile_utils.hpp/cpp`: Tile utilities, lookup tables
  - `types.hpp`: Core types (Action, PlayerState, StepResult, etc.)

- **Python Layer** (`mahjong/`):
  - `engine.py`: Thin wrapper providing `_SyncList` and `_PlayerStateProxy` for compatibility
  - `scoring.py`: Delegates to C++ scoring functions
  - `rules.py`: `RuleProfile` configuration dataclass
  - `rl/adapter.py`: RL adapter (observation encoder, action codec, mask builder)
  - `rl/trainer.py`: Stable PPO training/evaluation entrypoints
  - `replay.py`: Structured replay export/import

- **Bindings** (`cpp/src/bindings.cpp`): pybind11 bindings (only file with Python dependency)

## Quick Start

### 1) Install dependencies (requires C++ compiler)

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.10+
- C++17 compatible compiler (gcc/clang/MSVC)
- CMake 3.18+
- pybind11 2.12+
- scikit-build-core 0.10+

### 2) Build and install C++ engine

```bash
pip install -e .
```

This compiles the C++ core and installs the module locally in editable mode.

### 3) Run engine random self-test

```bash
# Legacy entrypoint (uses C++ engine internally)
python riichi_engine.py

# New debug entrypoint (C++ engine only)
python debug_cpp_engine.py --seed 42
```

**Debug modes:**
```bash
python debug_cpp_engine.py                # Full random game
python debug_cpp_engine.py --step         # Step-by-step with pause
python debug_cpp_engine.py --obs          # Test observation encoding
python debug_cpp_engine.py --parity 100   # C++ vs Python parity test
python debug_cpp_engine.py --rl           # RL adapter integration test
```

### 4) Train (default config file)

```bash
python rl_policy.py train --config configs/train.toml
```

Override selected values from CLI:

```bash
python rl_policy.py train --config configs/train.toml --updates 400 --lr 3e-4 --envs 128
```

**Training optimizations:**
- C++ engine uses `std::array<int,34>` for hand34 (stack allocation, cache-friendly)
- Zero-copy observation via `get_obs_array()` (writes directly to numpy buffer)
- `logging_enabled = False` during training to skip event log overhead
- Efficient phase state machine with minimal Python roundtrips
- **Multi-process data collection**: Use `python -m mahjong.rl.trainer_mp` for CPU-bound parallelization

### Multi-Process Data Collection

**Issue:** By default, Python's GIL limits training to single-core CPU utilization, even with multiple environments.

**Solution:** The trainer module includes multiprocessing support:

```bash
# Use multiprocessing for data collection (uses all CPU cores)
python -m mahjong.rl.trainer_mp train --config configs/train.toml

# Or test the multiprocessing approach standalone
python test_mp_collect.py --num-envs 64 --steps 10000 --mode both
```

**How it works:**
- Main process: GPU inference (model forward/backward passes)
- Worker processes: Parallel environment stepping on CPU cores
- Communication: Observations → Main → GPU inference → Actions → Workers

**Performance:**
- Sequential: ~1-2k steps/sec (single core)
- Multi-process: ~8-16k steps/sec (depends on CPU cores)
- Typical speedup: 4-8x on 8-16 core systems

## RTX 5090 Optimization

For RTX 5090 (32GB VRAM, high memory bandwidth), recommended hyperparameters:

```toml
# configs/train_rtx5090.toml
num_updates = 2000
num_envs = 256              # More environments for better parallelism
target_transitions = 32768  # Larger batch size
ppo_epochs = 3
ppo_batch_size = 8192       # Larger PPO mini-batches
lr = 0.0003
hidden = 2048               # Larger network (more capacity)
gamma = 0.99
lam = 0.95
clip_eps = 0.2
vf_coef = 0.5
ent_coef = 0.01
step_penalty = 0.001
reward_scale = 0.000125      # 1/8000
shaping_coef = 0.05
log_every = 10

[rules]
# Use default rules or customize
```

**Expected throughput:**
- Multi-process data collection: ~15-20k steps/sec
- Total training time (2000 updates): ~6-8 hours
- VRAM usage: ~8-12GB (well within 32GB capacity)

**Training command:**
```bash
python -m mahjong.rl.trainer_mp train --config configs/train_rtx5090.toml
```

### 5) Evaluate

```bash
python rl_policy.py eval --episodes 8 --weights ppo_riichi.pt
```

## Tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

**Test coverage:**
- `test_engine.py`: Core engine logic, phase transitions, legal actions
- `test_scoring.py`: Han/fu calculation, point resolution
- `test_yaku.py`: Yaku detection across various hand patterns
- `test_rl_adapter.py`: RL integration, observation encoding, action masking

## API Usage

### Basic Engine Usage

```python
from mahjong.engine import RiichiEngine
from mahjong.rules import RuleProfile
from _mahjong_cpp import ActionType, Phase

config = RuleProfile()  # Default: mahjongsoul_common rules
eng = RiichiEngine(seed=42, config=config)
eng.reset(dealer=0)

# Game loop
while not eng.done:
    if eng.phase == Phase.DISCARD:
        actions = eng.legal_actions()
        action = choose_action(actions)  # Your policy here
        result = eng.apply_action(action)
    elif eng.phase == Phase.RESPONSE:
        actions = eng.legal_actions()
        action = choose_response(actions)  # Pon/Chi/Ron/Pass
        result = eng.apply_action(action)

print(f"Game ended: {result.reason}, winner: {result.winner}")
print(f"Final scores: {eng.scores}")
```

### RL Integration

```python
import numpy as np
from mahjong.engine import RiichiEngine
from mahjong.rl.adapter import obs_encoder, mask_builder, action_from_index
from _mahjong_cpp import OBS_DIM

config = RuleProfile()
eng = RiichiEngine(seed=42, config=config)
eng.reset(dealer=0)

# Disable logging for training performance
eng.logging_enabled = False

seat = 0  # Your agent's seat
obs = np.zeros(OBS_DIM, dtype=np.float32)

while not eng.done:
    if eng.phase == Phase.DRAW:
        tile = eng.draw()
        continue

    # Zero-copy observation
    eng.get_obs_array(seat, obs)

    # Build legal action mask
    legal_actions = eng.legal_actions()
    mask = mask_builder(obs, legal_actions, seat, eng.phase, eng.pending_discard)

    # Select action (replace with your policy)
    valid_indices = np.where(mask)[0]
    action_idx = np.random.choice(valid_indices)
    action = action_from_index(action_idx, obs, legal_actions, eng.phase, eng.pending_discard)

    result = eng.apply_action(action)
```

### Observation Encoding

**OBS_DIM = 126** (34 + 34 + 34 + 4 + 4 + 8 + 4 + 4 + 4)

| Offset | Size | Description |
|--------|------|-------------|
| 0-33   | 34   | hand34 (tile counts) |
| 34-67  | 34   | river_history (last 2 discards per player) |
| 68-101 | 34   | meld_history (sequence/triplet/kan indicators) |
| 102-105 | 4   | phase_onehot (DRAW/DISCARD/RESPONSE/END) |
| 106-109 | 4   | seat_onehot |
| 110-117 | 8   | scalars (turn, live_wall_len, dead_wall_len, riichi_sticks, honba, dora_count, ippatsu[2]) |
| 118-121 | 4   | dora_pad (dora indicator onehot) |
| 122-125 | 4   | scores (normalized) |
| 126-129 | 4   | riichi_declared (per player) |

## Rule System

**Supported rules** (configurable via `RuleProfile`):

| Rule | Description |
|------|-------------|
| `RIICHI` | 立直声明和一發 |
| `ANKAN` / `MINKAN` / `KAKAN` | 暗杠/明杠/加杠 |
| `CHANKAN` | 抢杠 |
| `KYUUSHU_KYUUHAI` | 九种九牌 |
| `SUUFON_RENDA` | 四风连打 |
| `SUUCHA_RIICHI` | 四家立直 |
| `SUUKAN_SANRA` | 四杠散了 |
| `NAGASHI_MANGAN` | 流局满贯 |
| `NOTEN_BAPPU` | 流局罚点 |

**Yaku supported:**
- 1 han: 门前清自摸和, 断幺九, 役牌, 一杯口 (closed only)
- 2 han: 对对和, 三暗刻, 混老头, 小三元, 七对子, 三色同顺 (closed), 一气通贯 (closed), 混全带幺九
- 3 han: 纯全带幺九 (closed), 混一色
- 6 han: 清一色
- 13 han: 国士无双

## Performance Notes

- **C++ core**: ~100x faster than pure Python engine for self-play
- **Stack allocation**: `std::array<int,34>` eliminates heap allocations for hand34
- **Zero-copy**: `get_obs_array()` writes directly to pre-allocated numpy buffer
- **Training mode**: `logging_enabled = False` skips event log overhead (~15% speedup)
- **Device selection**: Automatic: `cuda -> mps -> cpu`

## License

MIT License

## Contributing

This project uses a modular C++ architecture with clear separation between core logic and Python bindings. The C++ core has zero Python dependencies, enabling independent testing and reuse.
