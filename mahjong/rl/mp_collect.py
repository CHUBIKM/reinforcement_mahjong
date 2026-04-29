"""Simplified multiprocessing data collection.

This is a practical implementation that uses a process pool to handle
environment stepping while the main process manages GPU inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import multiprocessing as mp

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


@dataclass
class EnvState:
    """Serialized environment state for inter-process transfer."""
    env_id: int
    done: bool
    phase: Phase
    cur: int
    hand34: List[int]
    river: List[List[int]]
    melds: List[Dict[str, Any]]
    riichi_declared: List[bool]
    riichi_turn: List[int]
    dora_indicators: List[int]
    wall: List[int]
    dead_wall: List[int]
    scores: List[int]
    honba: int
    riichi_sticks: int


@dataclass
class StepInput:
    """Input for stepping an environment."""
    env_state: EnvState
    action_idx: int
    actor: int


@dataclass
class StepOutput:
    """Output from stepping an environment."""
    env_id: int
    obs: np.ndarray
    mask: np.ndarray
    action_idx: int
    reward: float
    done: bool
    reason: str
    actor: int
    next_env_state: Optional[EnvState]


def _serialize_engine(e: RiichiEngine, env_id: int) -> EnvState:
    """Serialize engine state for inter-process transfer."""
    obs = e.get_obs(seat=0)
    return EnvState(
        env_id=env_id,
        done=e.done,
        phase=e.phase,
        cur=e.cur,
        hand34=list(obs["hand34"]),
        river=[list(r) for r in e.river],
        melds=[{"type": m.type, "tiles": list(m.tiles)} for m in e.melds],
        riichi_declared=list(e.riichi_declared),
        riichi_turn=list(e.riichi_turn),
        dora_indicators=list(e.dora_indicators),
        wall=list(e.wall),
        dead_wall=list(e.dead_wall),
        scores=list(e.scores),
        honba=e.honba,
        riichi_sticks=e.riichi_sticks,
    )


def _deserialize_engine(state: EnvState, rules: RuleProfile) -> RiichiEngine:
    """Recreate engine from serialized state."""
    # Note: This is a simplified version. Full implementation would need
    # to reconstruct all engine state properly.
    # For now, we'll create a fresh engine and reset it.
    e = RiichiEngine(seed=state.env_id, config=rules)
    # TODO: Properly restore engine state
    return e


def _step_env_worker(inp: StepInput, rules: RuleProfile) -> StepOutput:
    """Worker function to step a single environment.

    Args:
        inp: StepInput containing state and action
        rules: Rule configuration

    Returns:
        StepOutput with results
    """
    # Reconstruct engine (simplified - just create fresh for now)
    e = RiichiEngine(seed=inp.env_state.env_id, config=rules)
    e.reset(dealer=(inp.env_state.env_id % 4))

    # Execute action
    action = materialize_action(e, id_to_action(inp.action_idx))
    step_res = e.apply_action(action)

    # Get next observation
    if not e.done and e.phase != Phase.DRAW:
        # Advance to decision point
        while not e.done:
            if e.phase == Phase.DRAW:
                e.draw()
                continue
            legal = e.legal_actions()
            if e.phase == Phase.RESPONSE and len(legal) == 1 and legal[0].type == ActionType.PASS:
                e.apply_action(legal[0])
                continue
            break

    obs = e.get_obs(seat=e.cur) if not e.done else None

    return StepOutput(
        env_id=inp.env_state.env_id,
        obs=obs_encoder(obs) if obs is not None else np.zeros(OBS_DIM, dtype=np.float32),
        mask=mask_builder(e) if obs is not None else np.zeros(N_ACTIONS, dtype=np.float32),
        action_idx=inp.action_idx,
        reward=0.0,  # Computed by main process
        done=step_res.done,
        reason=step_res.reason if hasattr(step_res, 'reason') else "unknown",
        actor=inp.actor,
        next_env_state=_serialize_engine(e, inp.env_state.env_id) if not e.done else None,
    )


def collect_batch_mp(
    engines: List[RiichiEngine],
    model,
    cfg: Any,  # TrainConfig
    device: str,
    num_workers: Optional[int] = None
) -> Tuple[Any, Dict[str, int]]:
    """Collect training batch using multiprocessing.

    This uses a process pool to parallelize environment stepping.

    Args:
        engines: List of RiichiEngine instances
        model: PyTorch ActorCritic model
        cfg: Training configuration
        device: PyTorch device
        num_workers: Number of worker processes

    Returns:
        Tuple of (Batch, stats)
    """
    import torch
    from mahjong.rl.trainer import _sample_action, _reward_from_step, _hand_shape_score

    # Determine number of workers
    num_cpus = mp.cpu_count()
    num_workers = num_workers or (num_cpus // 2)
    num_workers = min(num_workers, len(engines), 8)  # Cap at 8 workers
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

    # Create process pool
    with mp.Pool(processes=num_workers) as pool:
        while ptr < target_transitions:
            # Reset done engines
            for i, e in enumerate(engines):
                if e.done:
                    e.reset(dealer=(cfg.seed + i) % 4)

            # Collect observations (this is fast with C++ engine)
            batch_obs = []
            batch_mask = []
            batch_actor = []
            batch_pre_shape = []
            batch_env_ids = []

            for i, e in enumerate(engines):
                if e.done or e.phase == Phase.DRAW:
                    continue

                # Advance to decision point
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

                global_stats["auto_pass"] += auto_pass

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

            # Model inference (batched, on GPU)
            with torch.no_grad():
                x = torch.tensor(np.stack(batch_obs), dtype=torch.float32, device=device)
                m = torch.tensor(np.stack(batch_mask), dtype=torch.float32, device=device)
                logits, values = model(x)
                aid, logp, _ = _sample_action(logits, m)

            # Prepare step inputs for multiprocessing
            step_inputs = []
            for b, env_i in enumerate(batch_env_ids):
                e = engines[env_i]
                state = _serialize_engine(e, env_i)
                step_inputs.append(StepInput(
                    env_state=state,
                    action_idx=int(aid[b].item()),
                    actor=batch_actor[b]
                ))

            # Execute steps in parallel
            step_func = partial(_step_env_worker, rules=cfg.rules)
            step_outputs = pool.map(step_func, step_inputs)

            # Process results
            for out in step_outputs:
                if ptr >= target_transitions:
                    break

                # Compute reward
                b_idx = batch_env_ids.index(out.env_id)
                shape_delta = _hand_shape_score(out.obs) - batch_pre_shape[b_idx]
                reward = -cfg.step_penalty + cfg.shaping_coef * shape_delta
                if out.done:
                    # Simplified - use 0 for final reward
                    # (real implementation would compute from score_delta)
                    pass

                # Store in buffers
                obs_buf[ptr] = out.obs
                mask_buf[ptr] = out.mask
                act_buf[ptr] = out.action_idx
                logp_buf[ptr] = float(logp[b_idx].item())
                rew_buf[ptr] = reward
                done_buf[ptr] = 1.0 if out.done else 0.0
                val_buf[ptr] = float(values[b_idx].item())
                env_buf[ptr] = out.env_id

                ptr += 1
                global_stats["steps"] += 1

                # Update engine state in main process
                if out.next_env_state is not None:
                    engines[out.env_id] = _deserialize_engine(out.next_env_state, cfg.rules)

                # Track done reasons
                if out.done:
                    reason_map = {"tsumo": "tsumo", "ron": "ron", "ryuukyoku": "ryuukyoku"}
                    reason = reason_map.get(out.reason, "other")
                    if reason in global_stats["done"]:
                        global_stats["done"][reason] += 1

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
