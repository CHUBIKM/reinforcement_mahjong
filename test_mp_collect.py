#!/usr/bin/env python3
"""Small CLI for comparing sequential and multiprocessing collection."""

from __future__ import annotations

import argparse
import time

from mahjong.rl.trainer import TrainConfig
from mahjong.rl.trainer_mp import train_mp


def _run(mode: str, cfg: TrainConfig) -> None:
    use_mp = mode == "mp"
    mode_text = "多进程采样" if use_mp else "单进程顺序采样"
    start = time.perf_counter()
    train_mp(cfg, use_multiprocessing=use_mp)
    elapsed = max(time.perf_counter() - start, 1e-9)
    print(f"{mode_text}：平均吞吐={cfg.target_transitions / elapsed:.1f}目标样本/秒，总耗时={elapsed:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="对比日麻强化学习训练的单进程与多进程采样速度")
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
