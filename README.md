# Mahjong Engine + RL (Refactored)

This project is now organized into subsystems:

- `mahjong/engine.py`: single-hand Riichi engine (state machine + legal actions + event log)
- `mahjong/scoring.py`: han/fu point resolution and payment distribution
- `mahjong/rules.py`: `RuleProfile` for ruleset toggles (`mahjongsoul_common` default)
- `mahjong/rl/adapter.py`: RL adapter (`obs_encoder`, `action codec`, `mask_builder`)
- `mahjong/rl/trainer.py`: stable `train(config)` / `evaluate(config)` entrypoints
- `mahjong/replay.py`: structured replay export/import

Legacy entrypoints are preserved:

- `riichi_engine.py`
- `rl_policy.py`

## Quick Start

### 1) Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2) Run engine random self-test

```bash
python riichi_engine.py
```

### 3) Train (default config file)

```bash
python rl_policy.py train --config configs/train.toml
```

Override selected values from CLI:

```bash
python rl_policy.py train --config configs/train.toml --updates 400 --lr 3e-4 --envs 128
```

### 4) Evaluate

```bash
python rl_policy.py eval --episodes 8 --weights ppo_riichi.pt
```

## Tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## Notes

- Default ruleset profile is `RuleProfile(name="mahjongsoul_common")`.
- Training device selection is automatic: `cuda -> mps -> cpu`.
- If `torch` is missing, training/evaluation raises a clear runtime error.
- Training hyperparameters can be persisted in `configs/train.toml` with inline comments.
- `fu` now uses full itemized calculation in engine-side resolution (base fu, menzen ron, tsumo fu, pair fu, wait fu, triplet/kan fu, and rounding rules including pinfu-tsumo/open-ron minima).
- Rule system now includes: `RIICHI`, `ANKAN/MINKAN/KAKAN`, `CHANKAN`, `KYUUSHU_KYUUHAI`, `SUUFON_RENDA`, `SUUCHA_RIICHI`, `SUUKAN_SANRA`, exhaustive draw `NOTEN_BAPPU`, and `NAGASHI_MANGAN`.
