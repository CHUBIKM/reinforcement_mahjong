#!/usr/bin/env python3
"""Small CLI for comparing sequential and multiprocessing collection."""

from __future__ import annotations

import argparse
import time

from mahjong.rl.trainer import TrainConfig
from mahjong.rl.trainer_mp import train_mp


def _run(mode: str, cfg: TrainConfig) -> None:
    use_mp = mode == "mp"
    start = time.perf_counter()
    train_mp(cfg, use_multiprocessing=use_mp)
    elapsed = max(time.perf_counter() - start, 1e-9)
    print(f"{mode}: {cfg.target_transitions / elapsed:.1f} target transitions/sec over {elapsed:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Mahjong RL collection modes")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--mode", choices=("seq", "mp", "both"), default="both")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cfg = TrainConfig(
        num_updates=1,
        num_envs=args.num_envs,
        target_transitions=args.steps,
        ppo_epochs=1,
        ppo_batch_size=min(2048, max(1, args.steps)),
        device=args.device,
        log_every=1,
    )

    if args.mode in ("seq", "both"):
        _run("seq", cfg)
    if args.mode in ("mp", "both"):
        _run("mp", cfg)


if __name__ == "__main__":
    main()
