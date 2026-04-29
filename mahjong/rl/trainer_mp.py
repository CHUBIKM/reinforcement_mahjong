"""Multi-process enabled trainer for Riichi Mahjong RL.

This module extends the trainer with multiprocessing support for CPU-bound
operations while keeping GPU inference in the main process.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

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
    OBS_DIM, N_ACTIONS, id_to_action,
    mask_builder, materialize_action, obs_encoder
)
from mahjong.rules import RuleProfile

# Import trainer components
from mahjong.rl.trainer import (
    TrainConfig, ActorCritic, Batch,
    _sample_action, _reward_from_step, _hand_shape_score,
    ppo_update, require_torch, select_device
)


@dataclass
class EnvObservation:
    """Observation from a single environment."""
    env_id: int
    obs: np.ndarray
    mask: np.ndarray
    actor: int
    hand_shape_score: float
    done: bool


@dataclass
class EnvStepData:
    """Data for stepping a single environment."""
    env_id: int
    action_idx: int
    actor: int


def _gather_observation(env: RiichiEngine, env_id: int) -> Optional[EnvObservation]:
    """Gather observation from a single environment.

    This function is designed to be called in parallel via ProcessPoolExecutor.
    """
    # Reset if done
    if env.done:
        env.reset(dealer=(env_id % 4))

    # Advance to decision point
    auto_pass = 0
    while not env.done:
        if env.phase == Phase.DRAW:
            env.draw()
            continue
        legal = env.legal_actions()
        if env.phase == Phase.RESPONSE and len(legal) == 1 and legal[0].type == ActionType.PASS:
            env.apply_action(legal[0])
            auto_pass += 1
            continue
        break

    if env.done or env.phase == Phase.DRAW:
        return None

    actor = env.cur
    obs = env.get_obs(seat=actor)

    return EnvObservation(
        env_id=env_id,
        obs=obs_encoder(obs),
        mask=mask_builder(env),
        actor=actor,
        hand_shape_score=_hand_shape_score(obs["hand34"]),
        done=False
    )


def _apply_action_step(
    env: RiichiEngine,
    action_idx: int,
    actor: int,
    cfg: TrainConfig,
    device: str,
    model  # ActorCritic - not used here, value computed in main process
) -> Tuple[int, np.ndarray, np.ndarray, float, bool]:
    """Apply action to a single environment and return next observation.

    Returns:
        (env_id, next_obs, next_mask, reward, done)
    """
    action = materialize_action(env, id_to_action(action_idx))
    step_res = env.apply_action(action)

    # Advance to next decision point
    while not env.done:
        if env.phase == Phase.DRAW:
            env.draw()
            continue
        legal = env.legal_actions()
        if env.phase == Phase.RESPONSE and len(legal) == 1 and legal[0].type == ActionType.PASS:
            env.apply_action(legal[0])
            continue
        break

    if not env.done and env.phase != Phase.DRAW:
        next_actor = env.cur
        next_obs = env.get_obs(seat=next_actor)
        reward = -cfg.step_penalty + cfg.shaping_coef * (
            _hand_shape_score(next_obs["hand34"]) - 0  # Pre-shape not available here
        )
        if step_res.done:
            # Add final reward
            pass  # Simplified
        return (0, obs_encoder(next_obs), mask_builder(env), reward, step_res.done)
    else:
        return (0, np.zeros(OBS_DIM, dtype=np.float32), np.zeros(N_ACTIONS, dtype=np.float32), 0.0, True)


def collect_parallel_batch_mp(
    engines: List[RiichiEngine],
    model: ActorCritic,
    cfg: TrainConfig,
    device: str,
    num_workers: Optional[int] = None
) -> Tuple[Batch, Dict]:
    """Collect training batch with multiprocessing for observation gathering.

    This parallelizes the observation gathering step across multiple CPU cores
    while keeping model inference in the main process.

    Args:
        engines: List of RiichiEngine instances
        model: PyTorch ActorCritic model
        cfg: Training configuration
        device: PyTorch device
        num_workers: Number of worker processes (default: CPU count // 2)

    Returns:
        Tuple of (Batch, stats)
    """
    import torch

    # Determine number of workers
    import os
    num_cpus = os.cpu_count()
    num_workers = num_workers or (num_cpus // 2)
    num_workers = min(num_workers, len(engines), 8)
    print(f"Using {num_workers} workers for observation gathering")

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

    try:
        while ptr < target_transitions:
            # Phase 1: Gather observations in parallel
            batch_obs = []
            batch_mask = []
            batch_actor = []
            batch_pre_shape = []
            batch_env_ids = []

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit all observation gathering tasks
                future_to_env = {
                    executor.submit(_gather_observation, e, i): i
                    for i, e in enumerate(engines)
                }

                # Collect results
                for future in as_completed(future_to_env):
                    env_id = future_to_env[future]
                    try:
                        obs_result = future.result()
                        if obs_result is not None:
                            batch_obs.append(obs_result.obs)
                            batch_mask.append(obs_result.mask)
                            batch_actor.append(obs_result.actor)
                            batch_pre_shape.append(obs_result.hand_shape_score)
                            batch_env_ids.append(env_id)
                    except Exception as e:
                        print(f"Error gathering obs for env {env_id}: {e}")

            if not batch_obs:
                continue

            # Phase 2: Batched model inference (on GPU, in main process)
            with torch.no_grad():
                x = torch.tensor(np.stack(batch_obs), dtype=torch.float32, device=device)
                m = torch.tensor(np.stack(batch_mask), dtype=torch.float32, device=device)
                logits, values = model(x)
                aid, logp, _ = _sample_action(logits, m)

            # Phase 3: Apply actions sequentially (this is fast with C++ engine)
            for b, env_i in enumerate(batch_env_ids):
                if ptr >= target_transitions:
                    break

                e = engines[env_i]
                actor = batch_actor[b]

                # Apply action
                action = materialize_action(e, id_to_action(int(aid[b].item())))
                step_res = e.apply_action(action)

                # Get next observation
                post_obs = e.get_obs(seat=actor)
                shape_delta = _hand_shape_score(post_obs["hand34"]) - batch_pre_shape[b]
                reward = _reward_from_step(step_res, actor, shape_delta, cfg)

                # Store in buffers
                obs_buf[ptr] = batch_obs[b]
                mask_buf[ptr] = batch_mask[b]
                act_buf[ptr] = int(aid[b].item())
                logp_buf[ptr] = float(logp[b].item())
                rew_buf[ptr] = reward
                done_buf[ptr] = 1.0 if step_res.done else 0.0
                val_buf[ptr] = float(values[b].item())
                env_buf[ptr] = env_i

                ptr += 1
                global_stats["steps"] += 1

                if step_res.done and step_res.reason in global_stats["done"]:
                    global_stats["done"][step_res.reason] += 1

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        pass  # Executor cleanup is automatic

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
            size=int(ptr),
        ),
        global_stats,
    )


def train_mp(config: Optional[TrainConfig] = None, use_multiprocessing: bool = True) -> ActorCritic:
    """Training function with optional multiprocessing support.

    Args:
        config: Training configuration
        use_multiprocessing: If True, use multiprocessing for data collection

    Returns:
        Trained ActorCritic model
    """
    require_torch()
    cfg = config or TrainConfig()
    device = select_device(cfg.device)

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    engines = [RiichiEngine(seed=cfg.seed + i, config=cfg.rules) for i in range(cfg.num_envs)]
    for i, e in enumerate(engines):
        e.reset(dealer=(cfg.seed + i) % 4)

    model = ActorCritic(hidden=cfg.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Choose collection function
    collect_fn = collect_parallel_batch_mp if use_multiprocessing else None
    if collect_fn is None:
        from mahjong.rl.trainer import collect_parallel_batch
        collect_fn = collect_parallel_batch

    for upd in range(1, cfg.num_updates + 1):
        batch, st = collect_fn(engines, model, cfg, device)
        met = ppo_update(model, optimizer, batch, cfg)

        if cfg.log_every > 0 and upd % cfg.log_every == 0:
            d = st["done"]
            denom = max(1, d["tsumo"] + d["ron"] + d["ryuukyoku"])
            mp_tag = "[MP]" if use_multiprocessing else "[SEQ]"
            print(
                f"{mp_tag} [UPD {upd}] device={device} transitions={batch.size} steps={st['steps']} "
                f"autoPASS={st['auto_pass']} "
                f"rate(ron/tsumo/ryu)={d['ron']/denom:.2%}/{d['tsumo']/denom:.2%}/{d['ryuukyoku']/denom:.2%} "
                f"loss={met['loss']:.4f} pl={met['pl']:.4f} vl={met['vl']:.4f} ent={met['ent']:.4f}"
            )

    return model
