"""Multi-process trainer for Riichi Mahjong RL.

Workers own their C++ engines for the lifetime of the collection loop.  The
main process only sends action ids and receives numpy observations/rewards, so
pybind engine objects never need to be pickled.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mahjong.engine import ActionType, Phase, RiichiEngine
from mahjong.rl.adapter import OBS_DIM, N_ACTIONS, id_to_action, mask_builder, materialize_action, obs_encoder
from mahjong.rl.trainer import (
    ActorCritic,
    Batch,
    TrainConfig,
    _hand_shape_score,
    _reward_from_step,
    _sample_action,
    collect_parallel_batch,
    load_train_config,
    ppo_update,
    require_torch,
    select_device,
    torch,
)


@dataclass(frozen=True)
class WorkerEnvConfig:
    env_id: int
    seed: int
    dealer: int


@dataclass
class WorkerObservation:
    worker_id: int
    env_id: int
    obs: np.ndarray
    mask: np.ndarray
    actor: int
    pre_shape: float


@dataclass(frozen=True)
class WorkerAction:
    env_id: int
    action_idx: int
    actor: int
    pre_shape: float


@dataclass
class WorkerStepResult:
    env_id: int
    reward: float
    done: bool
    reason: str


def _advance_until_decision(engine: RiichiEngine) -> int:
    auto_pass = 0
    while not engine.done:
        if engine.phase == Phase.DRAW:
            engine.draw()
            continue
        legal = engine.legal_actions()
        if engine.phase == Phase.RESPONSE and len(legal) == 1 and legal[0].type == ActionType.PASS:
            engine.apply_action(legal[0])
            auto_pass += 1
            continue
        break
    return auto_pass


def _worker_loop(
    worker_id: int,
    env_configs: List[WorkerEnvConfig],
    cfg: TrainConfig,
    request_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    engines: Dict[int, RiichiEngine] = {}
    dealers: Dict[int, int] = {}
    for env_cfg in env_configs:
        engine = RiichiEngine(seed=env_cfg.seed, config=cfg.rules)
        engine.reset(dealer=env_cfg.dealer)
        engines[env_cfg.env_id] = engine
        dealers[env_cfg.env_id] = env_cfg.dealer

    while True:
        msg = request_queue.get()
        kind = msg[0]

        if kind == "shutdown":
            return

        if kind == "observe":
            observations: List[WorkerObservation] = []
            auto_pass = 0
            for env_id, engine in engines.items():
                if engine.done:
                    engine.reset(dealer=dealers[env_id])

                auto_pass += _advance_until_decision(engine)
                if engine.done or engine.phase == Phase.DRAW:
                    continue

                actor = int(engine.cur)
                raw_obs = engine.get_obs(seat=actor)
                observations.append(
                    WorkerObservation(
                        worker_id=worker_id,
                        env_id=env_id,
                        obs=obs_encoder(raw_obs),
                        mask=mask_builder(engine),
                        actor=actor,
                        pre_shape=_hand_shape_score(raw_obs["hand34"]),
                    )
                )
            result_queue.put(("observe", worker_id, observations, {"auto_pass": auto_pass}))
            continue

        if kind == "step":
            step_results: List[WorkerStepResult] = []
            for action_req in msg[1]:
                engine = engines[action_req.env_id]
                action = materialize_action(engine, id_to_action(action_req.action_idx))
                step_res = engine.apply_action(action)
                post_obs = engine.get_obs(seat=action_req.actor)
                shape_delta = _hand_shape_score(post_obs["hand34"]) - action_req.pre_shape
                step_results.append(
                    WorkerStepResult(
                        env_id=action_req.env_id,
                        reward=_reward_from_step(step_res, action_req.actor, shape_delta, cfg),
                        done=bool(step_res.done),
                        reason=str(step_res.reason),
                    )
                )
            result_queue.put(("step", worker_id, step_results))
            continue

        result_queue.put(("error", worker_id, f"unknown worker message: {kind}"))


def _split_envs(cfg: TrainConfig, num_workers: int) -> List[List[WorkerEnvConfig]]:
    worker_envs: List[List[WorkerEnvConfig]] = [[] for _ in range(num_workers)]
    for env_id in range(cfg.num_envs):
        worker_envs[env_id % num_workers].append(
            WorkerEnvConfig(env_id=env_id, seed=cfg.seed + env_id, dealer=(cfg.seed + env_id) % 4)
        )
    return worker_envs


def _get_mp_context() -> mp.context.BaseContext:
    if "fork" in mp.get_all_start_methods():
        return mp.get_context("fork")
    return mp.get_context()


def collect_parallel_batch_mp(
    engines: List[RiichiEngine],
    model: ActorCritic,
    cfg: TrainConfig,
    device: str,
    num_workers: Optional[int] = None,
) -> Tuple[Batch, Dict[str, Any]]:
    del engines  # Workers own their own engines; kept for API compatibility.
    require_torch()

    cpu_count = os.cpu_count() or 1
    requested_workers = num_workers or cfg.num_workers or cpu_count
    worker_count = max(1, min(requested_workers, cfg.num_envs, 16))
    started_at = time.perf_counter()

    obs_buf = np.zeros((cfg.target_transitions, OBS_DIM), dtype=np.float32)
    mask_buf = np.zeros((cfg.target_transitions, N_ACTIONS), dtype=np.float32)
    act_buf = np.zeros((cfg.target_transitions,), dtype=np.int64)
    logp_buf = np.zeros((cfg.target_transitions,), dtype=np.float32)
    rew_buf = np.zeros((cfg.target_transitions,), dtype=np.float32)
    done_buf = np.zeros((cfg.target_transitions,), dtype=np.float32)
    val_buf = np.zeros((cfg.target_transitions,), dtype=np.float32)
    env_buf = np.zeros((cfg.target_transitions,), dtype=np.int64)

    ctx = _get_mp_context()
    result_queue = ctx.Queue()
    request_queues: List[mp.Queue] = []
    processes: List[mp.Process] = []

    for worker_id, env_configs in enumerate(_split_envs(cfg, worker_count)):
        request_queue = ctx.Queue()
        proc = ctx.Process(target=_worker_loop, args=(worker_id, env_configs, cfg, request_queue, result_queue))
        proc.start()
        request_queues.append(request_queue)
        processes.append(proc)

    stats: Dict[str, Any] = {
        "steps": 0,
        "auto_pass": 0,
        "done": {"tsumo": 0, "ron": 0, "ryuukyoku": 0},
        "multiprocessing": True,
        "workers": worker_count,
    }
    ptr = 0

    try:
        while ptr < cfg.target_transitions:
            for request_queue in request_queues:
                request_queue.put(("observe",))

            observations: List[WorkerObservation] = []
            for _ in processes:
                try:
                    kind, worker_id, payload, worker_stats = result_queue.get(timeout=60)
                except queue.Empty as exc:
                    raise RuntimeError("timed out waiting for multiprocessing observations") from exc
                if kind == "error":
                    raise RuntimeError(str(payload))
                if kind != "observe":
                    raise RuntimeError(f"unexpected worker response during observe: {kind}")
                del worker_id
                observations.extend(payload)
                stats["auto_pass"] += int(worker_stats.get("auto_pass", 0))

            if not observations:
                continue

            observations = observations[: cfg.target_transitions - ptr]
            x = torch.tensor(np.stack([o.obs for o in observations]), dtype=torch.float32, device=device)
            m = torch.tensor(np.stack([o.mask for o in observations]), dtype=torch.float32, device=device)

            with torch.no_grad():
                logits, values = model(x)
                aid, logp, _ = _sample_action(logits, m)

            by_worker: Dict[int, List[WorkerAction]] = {}
            decision_by_env: Dict[int, Tuple[int, WorkerObservation, float, float]] = {}
            for i, observation in enumerate(observations):
                action_idx = int(aid[i].item())
                decision_by_env[observation.env_id] = (
                    action_idx,
                    observation,
                    float(logp[i].item()),
                    float(values[i].item()),
                )
                by_worker.setdefault(observation.worker_id, []).append(
                    WorkerAction(
                        env_id=observation.env_id,
                        action_idx=action_idx,
                        actor=observation.actor,
                        pre_shape=observation.pre_shape,
                    )
                )

            for worker_id, actions in by_worker.items():
                request_queues[worker_id].put(("step", actions))

            for _ in by_worker:
                try:
                    kind, worker_id, payload = result_queue.get(timeout=60)
                except queue.Empty as exc:
                    raise RuntimeError("timed out waiting for multiprocessing step results") from exc
                if kind == "error":
                    raise RuntimeError(str(payload))
                if kind != "step":
                    raise RuntimeError(f"unexpected worker response during step: {kind}")
                del worker_id

                for step_result in payload:
                    if ptr >= cfg.target_transitions:
                        break
                    action_idx, observation, old_logp, value = decision_by_env[step_result.env_id]

                    obs_buf[ptr] = observation.obs
                    mask_buf[ptr] = observation.mask
                    act_buf[ptr] = action_idx
                    logp_buf[ptr] = old_logp
                    rew_buf[ptr] = float(step_result.reward)
                    done_buf[ptr] = 1.0 if step_result.done else 0.0
                    val_buf[ptr] = value
                    env_buf[ptr] = step_result.env_id

                    ptr += 1
                    stats["steps"] += 1
                    if step_result.done and step_result.reason in stats["done"]:
                        stats["done"][step_result.reason] += 1

    finally:
        for request_queue in request_queues:
            request_queue.put(("shutdown",))
        for proc in processes:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
        for request_queue in request_queues:
            request_queue.close()
        result_queue.close()

    elapsed = max(time.perf_counter() - started_at, 1e-9)
    stats["collect_seconds"] = elapsed
    stats["steps_per_second"] = stats["steps"] / elapsed

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
        stats,
    )


def train_mp(config: Optional[TrainConfig] = None, use_multiprocessing: bool = True) -> ActorCritic:
    require_torch()
    cfg = config or TrainConfig()
    device = select_device(cfg.device)

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    model = ActorCritic(hidden=cfg.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    engines = []
    collect_fn = collect_parallel_batch_mp
    if not use_multiprocessing:
        engines = [RiichiEngine(seed=cfg.seed + i, config=cfg.rules) for i in range(cfg.num_envs)]
        for i, engine in enumerate(engines):
            engine.reset(dealer=(cfg.seed + i) % 4)
        collect_fn = collect_parallel_batch

    for upd in range(1, cfg.num_updates + 1):
        try:
            batch, st = collect_fn(engines, model, cfg, device)
        except RuntimeError as exc:
            if not use_multiprocessing:
                raise
            print(f"[MP] collection failed ({exc}); falling back to sequential collection")
            engines = [RiichiEngine(seed=cfg.seed + i, config=cfg.rules) for i in range(cfg.num_envs)]
            for i, engine in enumerate(engines):
                engine.reset(dealer=(cfg.seed + i) % 4)
            collect_fn = collect_parallel_batch
            batch, st = collect_fn(engines, model, cfg, device)

        met = ppo_update(model, optimizer, batch, cfg)

        if cfg.log_every > 0 and (upd == 1 or upd % cfg.log_every == 0):
            d = st["done"]
            denom = max(1, d["tsumo"] + d["ron"] + d["ryuukyoku"])
            mp_tag = "[MP]" if st.get("multiprocessing") else "[SEQ]"
            worker_text = f" workers={st.get('workers', 1)}" if st.get("multiprocessing") else ""
            speed_text = ""
            if "collect_seconds" in st:
                speed_text = f" collect={st['collect_seconds']:.2f}s collect_sps={st['steps_per_second']:.0f}"
            print(
                f"{mp_tag} [UPD {upd}] device={device}{worker_text} transitions={batch.size} steps={st['steps']} "
                f"autoPASS={st['auto_pass']} "
                f"rate(ron/tsumo/ryu)={d['ron']/denom:.2%}/{d['tsumo']/denom:.2%}/{d['ryuukyoku']/denom:.2%} "
                f"loss={met['loss']:.4f} pl={met['pl']:.4f} vl={met['vl']:.4f} ent={met['ent']:.4f}"
                f"{speed_text}"
            )

    return model


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Mahjong RL policy with optional multiprocessing")
    sub = parser.add_subparsers(dest="cmd", required=False)

    train_parser = sub.add_parser("train", help="run PPO training")
    train_parser.add_argument("--config", type=str, default="configs/train.toml")
    train_parser.add_argument("--updates", type=int, default=None)
    train_parser.add_argument("--envs", type=int, default=None)
    train_parser.add_argument("--workers", type=int, default=None)
    train_parser.add_argument("--transitions", type=int, default=None)
    train_parser.add_argument("--device", type=str, default=None)
    train_parser.add_argument("--lr", type=float, default=None)
    train_parser.add_argument("--epochs", type=int, default=None)
    train_parser.add_argument("--batch-size", type=int, default=None)
    train_parser.add_argument("--log-every", type=int, default=None)
    train_parser.add_argument("--save", type=str, default="ppo_riichi.pt")
    train_parser.add_argument("--no-mp", action="store_true", help="use the sequential collector")

    return parser


def main() -> None:
    args = _build_parser().parse_args()
    cmd = args.cmd or "train"
    if cmd != "train":
        raise SystemExit(f"unsupported command: {cmd}")

    cfg = TrainConfig()
    cfg_path = Path(args.config) if args.config else None
    if cfg_path and cfg_path.exists():
        cfg = load_train_config(str(cfg_path), base=cfg)
    elif cfg_path and str(cfg_path) != "configs/train.toml":
        raise SystemExit(f"Train config not found: {cfg_path}")

    if args.updates is not None:
        cfg.num_updates = args.updates
    if args.envs is not None:
        cfg.num_envs = args.envs
    if args.workers is not None:
        cfg.num_workers = args.workers
    if args.transitions is not None:
        cfg.target_transitions = args.transitions
    if args.device is not None:
        cfg.device = args.device
    if args.lr is not None:
        cfg.lr = args.lr
    if args.epochs is not None:
        cfg.ppo_epochs = args.epochs
    if args.batch_size is not None:
        cfg.ppo_batch_size = args.batch_size
    if args.log_every is not None:
        cfg.log_every = args.log_every

    model = train_mp(cfg, use_multiprocessing=not args.no_mp)
    torch.save(model.state_dict(), args.save)
    print(f"saved to {args.save}")


if __name__ == "__main__":
    main()
