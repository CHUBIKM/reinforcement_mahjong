"""Multi-process parallel data collection for Riichi Mahjong RL training.

This module provides CPU-bound parallelization for environment stepping
while keeping GPU inference in the main process. This is essential for
maximizing CPU utilization on multi-core systems during training.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from functools import partial
import queue

import numpy as np

# Force local module loading
import sys
from pathlib import Path
_project_root = Path(__file__).parent.parent.parent
_module_path = _project_root / "_mahjong_cpp.cpython-312-darwin.so"

if _module_path.exists():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_mahjong_cpp", str(_module_path))
    _mahjong_cpp = importlib.util.module_from_spec(spec)
    sys.modules["_mahjong_cpp"] = _mahjong_cpp
    spec.loader.exec_module(_mahjong_cpp)

from _mahjong_cpp import RiichiEngine, Action, ActionType, Phase
from mahjong.rl.adapter import (
    OBS_DIM, N_ACTIONS, action_to_id, id_to_action,
    mask_builder, materialize_action, obs_encoder
)
from mahjong.rules import RuleProfile


@dataclass
class StepRequest:
    """Request from main process to worker: apply these actions."""
    env_ids: List[int]
    action_indices: List[int]
    actors: List[int]


@dataclass
class StepResult:
    """Result of stepping environments."""
    obs: np.ndarray
    mask: np.ndarray
    action_idx: int
    reward: float
    done: bool
    actor: int
    env_id: int


@dataclass
class WorkerInit:
    """Initialization data for a worker process."""
    worker_id: int
    env_configs: List[Tuple[int, int]]  # [(env_id, seed), ...]
    rules: RuleProfile


@dataclass
class WorkerState:
    """Current state of environments in a worker."""
    engine_states: List[Dict[str, Any]]
    env_ids: List[int]


def _engine_to_dict(e: RiichiEngine, env_id: int) -> Dict[str, Any]:
    """Serialize engine state to dict for inter-process transfer."""
    obs = e.get_obs(seat=e.cur)
    return {
        "env_id": env_id,
        "done": e.done,
        "phase": e.phase,
        "cur": e.cur,
        "hand34": list(obs["hand34"]),
        "actor": e.cur,
    }


def _worker_process(
    init: WorkerInit,
    request_queue: mp.Queue,
    result_queue: mp.Queue
) -> None:
    """Worker process that manages a subset of environments.

    The worker:
    1. Initializes its environments
    2. Waits for step requests from main process
    3. Executes actions and returns results
    """
    # Create engines
    engines = []
    env_ids = []
    for env_id, seed in init.env_configs:
        e = RiichiEngine(seed=seed, config=init.rules)
        e.reset(dealer=(seed + env_id) % 4)
        engines.append(e)
        env_ids.append(env_id)

    # Map env_id to engine index
    env_id_to_idx = {eid: i for i, eid in enumerate(env_ids)}

    while True:
        try:
            req = request_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        if req is None:  # Shutdown signal
            break

        # Process step request
        results = []
        for env_id, action_idx, actor in zip(req.env_ids, req.action_indices, req.actors):
            idx = env_id_to_idx[env_id]
            e = engines[idx]

            # Skip if done
            if e.done:
                continue

            # Execute action
            action = materialize_action(e, id_to_action(action_idx))
            step_res = e.apply_action(action)

            # Get next observation
            if not e.done and e.phase != Phase.DRAW:
                # Advance to next decision point
                auto_pass = 0
                while not e.done:
                    if e.phase == Phase.DRAW:
                        e.draw()
                        continue
                    legal = e.legal_actions()
                    if e.phase == Phase.RESPONSE and len(legal) == 1 and legal[0].type == ActionType.PASS:
                        e.apply_action(legal[0])
                        auto_pass += 1
                        continue
                    break

            # Get observation if we're at a decision point
            if not e.done and e.phase != Phase.DRAW:
                obs = e.get_obs(seat=e.cur)
                obs_encoded = obs_encoder(obs)
                mask = mask_builder(e)
                results.append(StepResult(
                    obs=obs_encoded,
                    mask=mask,
                    action_idx=action_idx,
                    reward=0.0,  # Will be computed by main process
                    done=step_res.done,
                    actor=actor,
                    env_id=env_id
                ))

        result_queue.put(results)


def _hand_shape_score(hand34: List[int]) -> float:
    """Compute hand shape score for reward shaping."""
    s = 0.0
    for t in range(34):
        c = hand34[t]
        if c >= 2:
            s += 0.12
        if c >= 3:
            s += 0.10
    for base in (0, 9, 18):
        for i in range(8):
            if hand34[base + i] > 0 and hand34[base + i + 1] > 0:
                s += 0.03
        for i in range(7):
            if hand34[base + i] > 0 and hand34[base + i + 2] > 0:
                s += 0.015
    for t in range(27, 34):
        if hand34[t] == 1:
            s -= 0.05
    return s


def collect_batch_multiprocess(
    model,  # ActorCritic
    cfg: Any,  # TrainConfig
    device: str,
    num_workers: Optional[int] = None
) -> Tuple[Any, Dict[str, int]]:
    """Collect training batch using multiprocessing for CPU-bound env stepping.

    This splits environments across worker processes to utilize all CPU cores.
    The main process handles GPU inference.

    Args:
        model: PyTorch ActorCritic model
        cfg: Training configuration
        device: PyTorch device
        num_workers: Number of worker processes (default: num_cpu_cores // 2)

    Returns:
        Tuple of (Batch, stats)
    """
    import torch

    # Determine number of workers
    num_cpus = mp.cpu_count()
    num_workers = num_workers or (num_cpus // 2)
    num_workers = min(num_workers, cfg.num_envs, 16)

    print(f"Using {num_workers} worker processes (out of {num_cpus} CPUs)")

    # Prepare buffers
    target_transitions = cfg.target_transitions
    obs_buf = np.zeros((target_transitions, OBS_DIM), dtype=np.float32)
    mask_buf = np.zeros((target_transitions, N_ACTIONS), dtype=np.float32)
    act_buf = np.zeros((target_transitions,), dtype=np.int64)
    logp_buf = np.zeros((target_transitions,), dtype=np.float32)
    rew_buf = np.zeros((target_transitions,), dtype=np.float32)
    done_buf = np.zeros((target_transitions,), dtype=np.float32)
    val_buf = np.zeros((target_transitions,), dtype=np.float32)
    env_buf = np.zeros((target_transitions,), dtype=np.int64)

    ptr = 0
    global_stats = {"steps": 0, "auto_pass": 0, "done": {"tsumo": 0, "ron": 0, "ryuukyoku": 0}}

    # Split environments among workers
    envs_per_worker = cfg.num_envs // num_workers
    worker_envs: List[List[Tuple[int, int]]] = []
    for w in range(num_workers):
        start = w * envs_per_worker
        end = start + envs_per_worker if w < num_workers - 1 else cfg.num_envs
        env_configs = [(i, cfg.seed + i) for i in range(start, end)]
        worker_envs.append(env_configs)

    # Initialize workers
    from mahjong.rl.trainer import _sample_action, _reward_from_step

    request_queues: List[mp.Queue] = []
    result_queues: List[mp.Queue] = []
    processes: List[mp.Process] = []

    for w in range(num_workers):
        req_q = mp.Queue(maxsize=10)
        res_q = mp.Queue(maxsize=10)
        request_queues.append(req_q)
        result_queues.append(res_q)

        init = WorkerInit(
            worker_id=w,
            env_configs=worker_envs[w],
            rules=cfg.rules
        )

        p = mp.Process(target=_worker_process, args=(init, req_q, res_q))
        p.start()
        processes.append(p)

    # Track engine states in main process for observation gathering
    engines = [RiichiEngine(seed=cfg.seed + i, config=cfg.rules) for i in range(cfg.num_envs)]
    for i, e in enumerate(engines):
        e.reset(dealer=(cfg.seed + i) % 4)

    try:
        while ptr < target_transitions:
            # Reset done engines
            for i, e in enumerate(engines):
                if e.done:
                    e.reset(dealer=(cfg.seed + i) % 4)

            # Collect observations from all engines
            batch_obs = []
            batch_mask = []
            batch_actor = []
            batch_pre_shape = []
            batch_env_ids = []

            for i, e in enumerate(engines):
                if e.done or e.phase == Phase.DRAW:
                    continue

                # Advance to decision point
                while not e.done:
                    if e.phase == Phase.DRAW:
                        e.draw()
                        continue
                    legal = e.legal_actions()
                    if e.phase == Phase.RESPONSE and len(legal) == 1 and legal[0].type == ActionType.PASS:
                        e.apply_action(legal[0])
                        global_stats["auto_pass"] += 1
                        continue
                    break

                if e.done or e.phase == Phase.DRAW:
                    continue

                actor = e.cur
                obs = e.get_obs(seat=actor)
                batch_obs.append(obs_encoder(obs))
                batch_mask.append(mask_builder(e))
                batch_actor.append(actor)
                batch_pre_shape.append(_hand_shape_score(obs["hand34"]))
                batch_env_ids.append(i)

            if not batch_obs:
                continue

            # Model inference
            with torch.no_grad():
                x = torch.tensor(np.stack(batch_obs), dtype=torch.float32, device=device)
                m = torch.tensor(np.stack(batch_mask), dtype=torch.float32, device=device)
                logits, values = model(x)
                aid, logp, _ = _sample_action(logits, m)

            # Send step requests to workers
            worker_requests: Dict[int, StepRequest] = {w: StepRequest([], [], []) for w in range(num_workers)}
            for b, env_i in enumerate(batch_env_ids):
                worker_idx = env_i // envs_per_worker
                if worker_idx >= num_workers:
                    worker_idx = num_workers - 1
                worker_requests[worker_idx].env_ids.append(env_i)
                worker_requests[worker_idx].action_indices.append(int(aid[b].item()))
                worker_requests[worker_idx].actors.append(batch_actor[b])

            for w in range(num_workers):
                if worker_requests[w].env_ids:
                    request_queues[w].put(worker_requests[w])

            # Collect results from workers
            for w in range(num_workers):
                if not worker_requests[w].env_ids:
                    continue
                results = result_queues[w].get()

                for res in results:
                    # Get corresponding pre-shape score
                    b_idx = batch_env_ids.index(res.env_id)
                    shape_delta = _hand_shape_score(res.obs) - batch_pre_shape[b_idx]

                    # Compute reward (simplified - workers don't compute it)
                    reward = -cfg.step_penalty + cfg.shaping_coef * shape_delta
                    if res.done:
                        # For simplicity, use 0 as final reward for now
                        # (real implementation would need to track engine states)
                        pass

                    # Store in buffers
                    obs_buf[ptr] = res.obs
                    mask_buf[ptr] = res.mask
                    act_buf[ptr] = res.action_idx
                    logp_buf[ptr] = float(logp[b_idx].item())
                    rew_buf[ptr] = reward
                    done_buf[ptr] = 1.0 if res.done else 0.0
                    val_buf[ptr] = float(values[b_idx].item())
                    env_buf[ptr] = res.env_id

                    ptr += 1
                    global_stats["steps"] += 1

                    if ptr >= target_transitions:
                        break

                if ptr >= target_transitions:
                    break

    finally:
        # Shutdown workers
        for q in request_queues:
            q.put(None)
        for p in processes:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()

    # Import Batch class
    from mahjong.rl.trainer import Batch

    return (
        Batch(
            obs=obs_buf[:ptr],
            mask=mask_buf[:ptr],
            act=act_buf[:ptr],
            logp=logp_buf[:ptr],
            rew=rew_buf[:ptr],
            done=done_buf[:ptr],
            val=val_buf[:ptr],
            env_id=env_buf[:ptr],
            size=ptr,
        ),
        global_stats,
    )
