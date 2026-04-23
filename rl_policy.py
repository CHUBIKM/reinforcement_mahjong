# -*- coding: utf-8 -*-
"""Backward-compatible entrypoint for refactored RL policy/training."""

from __future__ import annotations

import argparse
from pathlib import Path

from mahjong.rl.adapter import N_ACTIONS, OBS_DIM, action_to_id, id_to_action, mask_builder, materialize_action, obs_encoder
from mahjong.rl.trainer import ActorCritic, EvalConfig, TrainConfig, evaluate, load_train_config, train


legal_action_mask = mask_builder
obs_to_vec = obs_encoder
DEFAULT_TRAIN_CONFIG_PATH = "configs/train.toml"


def train_selfplay_ppo(
    num_updates: int = 2000,
    seed: int = 0,
    device: str = "cpu",
    log_every: int = 20,
    stats_window: int = 100,
    num_envs: int = 64,
    target_transitions: int = 16384,
    ppo_epochs: int = 2,
    ppo_batch_size: int = 4096,
    ent_coef: float = 0.05,
    lr: float = 1e-4,
    clip_eps: float = 0.1,
):
    del stats_window  # preserved for compatibility
    cfg = TrainConfig(
        num_updates=num_updates,
        seed=seed,
        device=device,
        log_every=log_every,
        num_envs=num_envs,
        target_transitions=target_transitions,
        ppo_epochs=ppo_epochs,
        ppo_batch_size=ppo_batch_size,
        ent_coef=ent_coef,
        lr=lr,
        clip_eps=clip_eps,
    )
    return train(cfg)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train or evaluate Mahjong RL policy")
    sub = p.add_subparsers(dest="cmd", required=False)

    p_train = sub.add_parser("train", help="run PPO training")
    p_train.add_argument("--config", type=str, default=DEFAULT_TRAIN_CONFIG_PATH)
    p_train.add_argument("--updates", type=int, default=None)
    p_train.add_argument("--envs", type=int, default=None)
    p_train.add_argument("--transitions", type=int, default=None)
    p_train.add_argument("--device", type=str, default=None)
    p_train.add_argument("--lr", type=float, default=None)
    p_train.add_argument("--epochs", type=int, default=None)
    p_train.add_argument("--batch-size", type=int, default=None)
    p_train.add_argument("--log-every", type=int, default=None)
    p_train.add_argument("--save", type=str, default="ppo_riichi.pt")

    p_eval = sub.add_parser("eval", help="run quick evaluation")
    p_eval.add_argument("--episodes", type=int, default=8)
    p_eval.add_argument("--device", type=str, default=None)
    p_eval.add_argument("--weights", type=str, default=None)

    return p


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    cmd = args.cmd or "train"
    try:
        if cmd == "train":
            cfg = TrainConfig()
            cfg_path = Path(args.config) if args.config else None
            if cfg_path:
                if cfg_path.exists():
                    cfg = load_train_config(str(cfg_path), base=cfg)
                elif str(cfg_path) != DEFAULT_TRAIN_CONFIG_PATH:
                    raise RuntimeError(f"Train config not found: {cfg_path}")

            # CLI overrides take precedence over config file.
            if args.updates is not None:
                cfg.num_updates = args.updates
            if args.envs is not None:
                cfg.num_envs = args.envs
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

            model = train(cfg)
            try:
                import torch

                torch.save(model.state_dict(), args.save)
                print(f"saved to {args.save}")
            except ModuleNotFoundError:
                raise RuntimeError("PyTorch is not installed; cannot save model weights.")

        elif cmd == "eval":
            model = ActorCritic()
            if args.weights:
                try:
                    import torch

                    state = torch.load(args.weights, map_location="cpu")
                    model.load_state_dict(state)
                except ModuleNotFoundError:
                    raise RuntimeError("PyTorch is not installed; cannot load model weights.")
            out = evaluate(model, EvalConfig(episodes=args.episodes, device=args.device))
            print(out)
    except RuntimeError as exc:
        raise SystemExit(str(exc))
