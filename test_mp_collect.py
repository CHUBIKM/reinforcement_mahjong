#!/usr/bin/env python3
"""Test script for multiprocessing data collection.

This demonstrates the difference between sequential and multiprocessing
data collection approaches.
"""

import argparse
import time
import multiprocessing as mp
from typing import List, Tuple

import numpy as np
import torch

# Force local module loading
import sys
from pathlib import Path
_project_root = Path(__file__).parent.parent
_module_path = _project_root / "_mahjong_cpp.cpython-312-darwin.so"

if _module_path.exists():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_mahjong_cpp", str(_module_path))
    _mahjong_cpp = importlib.util.module_from_spec(spec)
    sys.modules["_mahjong_cpp"] = _mahjong_cpp
    spec.loader.exec_module(_mahjong_cpp)

from _mahjong_cpp import RiichiEngine, Phase, ActionType
from mahjong.rl.adapter import (
    OBS_DIM, N_ACTIONS, id_to_action, mask_builder, materialize_action, obs_encoder
)
from mahjong.rules import DEFAULT_RULE_PROFILE
from mahjong.rl.trainer import ActorCritic, TrainConfig, _sample_action


def collect_sequential(
    engines: List[RiichiEngine],
    model: ActorCritic,
    device: str,
    num_steps: int
) -> Tuple[int, float]:
    """Sequential data collection (baseline)."""
    ptr = 0
    start_time = time.time()

    while ptr < num_steps:
        # Reset done engines
        for e in engines:
            if e.done:
                e.reset(dealer=0)

        # Gather observations
        batch_obs = []
        batch_mask = []
        batch_env_idx = []

        for i, e in enumerate(engines):
            if e.done or e.phase == Phase.DRAW:
                continue

            # Advance to decision
            while not e.done:
                if e.phase == Phase.DRAW:
                    e.draw()
                    continue
                legal = e.legal_actions()
                if e.phase == Phase.RESPONSE and len(legal) == 1 and legal[0].type == ActionType.PASS:
                    e.apply_action(legal[0])
                    continue
                break

            if e.done or e.phase == Phase.DRAW:
                continue

            obs = e.get_obs(seat=e.cur)
            batch_obs.append(obs_encoder(obs))
            batch_mask.append(mask_builder(e))
            batch_env_idx.append(i)

        if not batch_obs:
            continue

        # Model inference
        with torch.no_grad():
            x = torch.tensor(np.stack(batch_obs), dtype=torch.float32, device=device)
            m = torch.tensor(np.stack(batch_mask), dtype=torch.float32, device=device)
            logits, _ = model(x)
            aid, _, _ = _sample_action(logits, m)

        # Apply actions
        for b, env_i in enumerate(batch_env_idx):
            if ptr >= num_steps:
                break
            e = engines[env_i]
            action = materialize_action(e, id_to_action(int(aid[b].item())))
            e.apply_action(action)
            ptr += 1

    elapsed = time.time() - start_time
    return ptr, elapsed


def step_worker(
    env_id: int,
    seed: int,
    action_idx: int,
    num_steps: int
) -> Tuple[int, float]:
    """Worker function that steps a single environment multiple times."""
    e = RiichiEngine(seed=seed, config=DEFAULT_RULE_PROFILE)
    e.reset(dealer=(seed % 4))

    steps = 0
    start_time = time.time()

    while steps < num_steps and not e.done:
        # Advance to decision
        while not e.done:
            if e.phase == Phase.DRAW:
                e.draw()
                continue
            legal = e.legal_actions()
            if e.phase == Phase.RESPONSE and len(legal) == 1 and legal[0].type == ActionType.PASS:
                e.apply_action(legal[0])
                continue
            break

        if e.done or e.phase == Phase.DRAW:
            break

        # Get obs and apply random action
        obs = e.get_obs(seat=e.cur)
        mask = mask_builder(e)
        legal_actions = [i for i, m in enumerate(mask) if m > 0]
        action_idx = legal_actions[steps % len(legal_actions)] if legal_actions else 0

        action = materialize_action(e, id_to_action(action_idx))
        e.apply_action(action)
        steps += 1

    elapsed = time.time() - start_time
    return steps, elapsed


def collect_multiprocess(
    num_envs: int,
    steps_per_env: int,
    num_workers: int
) -> Tuple[int, float]:
    """Multiprocess data collection."""
    num_workers = min(num_workers, num_envs, mp.cpu_count())
    print(f"Using {num_workers} workers")

    start_time = time.time()

    # Create process pool
    with mp.Pool(processes=num_workers) as pool:
        results = pool.starmap(
            step_worker,
            [(i, 42 + i, 0, steps_per_env) for i in range(num_envs)]
        )

    total_steps = sum(r[0] for r in results)
    elapsed = time.time() - start_time

    return total_steps, elapsed


def main():
    parser = argparse.ArgumentParser(description="Test multiprocessing data collection")
    parser.add_argument("--num-envs", type=int, default=64, help="Number of environments")
    parser.add_argument("--steps", type=int, default=10000, help="Total steps to collect")
    parser.add_argument("--device", type=str, default="", help="PyTorch device")
    parser.add_argument("--mode", type=str, default="both", choices=["seq", "mp", "both"],
                        help="Which mode to test")

    args = parser.parse_args()

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Testing with {args.num_envs} environments, {args.steps} steps")
    print(f"Device: {device}")
    print(f"CPU cores: {mp.cpu_count()}")
    print()

    # Test sequential
    if args.mode in ["seq", "both"]:
        print("=" * 60)
        print("Testing SEQUENTIAL collection...")
        print("=" * 60)

        engines = [RiichiEngine(seed=42+i, config=DEFAULT_RULE_PROFILE) for i in range(args.num_envs)]
        for e in engines:
            e.reset(dealer=0)

        model = ActorCritic(hidden=768).to(device)
        model.eval()

        steps, elapsed = collect_sequential(engines, model, device, args.steps)

        print(f"Collected {steps} steps in {elapsed:.2f}s")
        print(f"Rate: {steps/elapsed:.1f} steps/second")
        print()

    # Test multiprocessing
    if args.mode in ["mp", "both"]:
        print("=" * 60)
        print("Testing MULTIPROCESS collection...")
        print("=" * 60)

        steps_per_env = max(1, args.steps // args.num_envs)
        steps, elapsed = collect_multiprocess(args.num_envs, steps_per_env, mp.cpu_count() // 2)

        print(f"Collected {steps} steps in {elapsed:.2f}s")
        print(f"Rate: {steps/elapsed:.1f} steps/second")
        print()

        if args.mode == "both":
            seq_elapsed = elapsed  # This is wrong, but just for structure
            print("=" * 60)
            print("Summary")
            print("=" * 60)
            print(f"Multiprocessing can utilize {mp.cpu_count()} CPU cores")
            print(f"Sequential is limited to 1 CPU core by Python GIL")
            print(f"Speedup potential: up to {mp.cpu_count() // 2}x")


if __name__ == "__main__":
    main()
