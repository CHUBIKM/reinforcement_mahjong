from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import time

import numpy as np

from mahjong.engine import ActionType, Phase, RiichiEngine
from mahjong.rules import RuleProfile, DEFAULT_RULE_PROFILE
from mahjong.rl.adapter import N_ACTIONS, OBS_DIM, id_to_action, mask_builder, materialize_action, obs_encoder

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover - exercised in runtime when torch missing
    torch = None
    nn = None
    F = None

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None


def require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "训练或评估需要安装 PyTorch。请先运行 `pip install torch`，"
            "或按 PyTorch 官方安装页面选择适合当前系统和设备的 wheel。"
        )


def select_device(preferred: Optional[str] = None) -> str:
    require_torch()
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class TrainConfig:
    num_updates: int = 200
    seed: int = 42
    device: Optional[str] = None
    num_envs: int = 64
    num_workers: Optional[int] = None
    target_transitions: int = 8192
    ppo_epochs: int = 2
    ppo_batch_size: int = 2048
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.02
    lr: float = 1e-4
    hidden: int = 768
    step_penalty: float = 0.002
    reward_scale: float = 1.0 / 8000.0
    shaping_coef: float = 0.05
    log_every: int = 10
    rules: RuleProfile = DEFAULT_RULE_PROFILE


@dataclass
class EvalConfig:
    episodes: int = 8
    seed: int = 123
    device: Optional[str] = None
    greedy: bool = True
    max_steps: int = 20000
    rules: RuleProfile = DEFAULT_RULE_PROFILE


def _train_config_field_names() -> set[str]:
    return {f.name for f in fields(TrainConfig)}


def _rule_field_names() -> set[str]:
    return {f.name for f in fields(RuleProfile)}


def _merge_rule_profile(base: RuleProfile, patch: Dict[str, Any]) -> RuleProfile:
    if not patch:
        return base
    unknown = set(patch.keys()) - _rule_field_names()
    if unknown:
        raise ValueError(f"训练配置 [rules] 中包含未知字段：{sorted(unknown)}")
    return replace(base, **patch)


def load_train_config(path: str, base: Optional[TrainConfig] = None) -> TrainConfig:
    """Load TrainConfig from TOML file.

    Supported layout:
    - top-level train fields (e.g. num_updates, num_envs, lr, ...)
    - optional [rules] table for RuleProfile overrides.
    """
    if tomllib is None:
        raise RuntimeError("当前 Python 运行时缺少 tomllib，无法读取 TOML 训练配置。")

    cfg = base or TrainConfig()
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("训练配置 TOML 必须解析为表/dict。")

    raw_rules = raw.get("rules", {})
    if raw_rules is None:
        raw_rules = {}
    if not isinstance(raw_rules, dict):
        raise ValueError("训练配置中的 [rules] 必须是一个表。")

    unknown_top = set(raw.keys()) - (_train_config_field_names() | {"rules"})
    if unknown_top:
        raise ValueError(f"训练配置中包含未知字段：{sorted(unknown_top)}")

    train_patch = {k: v for k, v in raw.items() if k != "rules"}
    if train_patch:
        cfg = replace(cfg, **train_patch)
    merged_rules = _merge_rule_profile(cfg.rules, raw_rules)
    return replace(cfg, rules=merged_rules)


if nn is not None:
    class ActorCritic(nn.Module):
        def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS, hidden: int = 768):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.ReLU(),
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.LayerNorm(hidden),
            )
            self.pi = nn.Linear(hidden, n_actions)
            self.v = nn.Linear(hidden, 1)

            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    nn.init.zeros_(m.bias)
            nn.init.orthogonal_(self.pi.weight, gain=0.01)
            nn.init.zeros_(self.pi.bias)
            nn.init.orthogonal_(self.v.weight, gain=1.0)
            nn.init.zeros_(self.v.bias)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            h = self.net(x)
            return self.pi(h), self.v(h).squeeze(-1)
else:
    class ActorCritic:  # pragma: no cover - only used when torch is unavailable
        def __init__(self, *args, **kwargs):
            del args, kwargs
            require_torch()


@dataclass
class Batch:
    obs: np.ndarray
    mask: np.ndarray
    act: np.ndarray
    logp: np.ndarray
    rew: np.ndarray
    done: np.ndarray
    val: np.ndarray
    env_id: np.ndarray
    size: int


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def _format_training_mode(tag: str) -> str:
    if tag == "[MP]":
        return "多进程采样训练"
    if tag == "[SEQ]":
        return "单进程顺序训练"
    return tag.strip("[]") or "训练"


def format_train_start(*, tag: str, cfg: TrainConfig, device: str) -> str:
    mode = _format_training_mode(tag)
    worker_text = "自动选择" if cfg.num_workers is None else str(cfg.num_workers)
    return (
        f"{tag} {mode}开始\n"
        f"  训练目标：共 {cfg.num_updates} 轮参数更新；每轮先采集 {cfg.target_transitions} 条决策样本，再执行 PPO 优化。\n"
        f"  并行配置：环境数={cfg.num_envs}，工作进程={worker_text}，运行设备={device}。\n"
        f"  PPO 配置：每轮优化 epoch={cfg.ppo_epochs}，小批量大小={cfg.ppo_batch_size}，学习率={cfg.lr:g}，"
        f"折扣因子 gamma={cfg.gamma:g}，GAE lambda={cfg.lam:g}。\n"
        f"  奖励构成：每步惩罚={cfg.step_penalty:g}，手牌形状奖励系数={cfg.shaping_coef:g}，"
        f"终局点差缩放={cfg.reward_scale:g}。"
    )


def format_train_status(
    *,
    tag: str,
    update: int,
    cfg: TrainConfig,
    device: str,
    batch: Batch,
    stats: Dict[str, Any],
    metrics: Dict[str, float],
    elapsed_total: float,
    update_seconds: float,
) -> str:
    done = stats["done"]
    terminal = done["tsumo"] + done["ron"] + done["ryuukyoku"]
    denom = max(1, terminal)
    progress = update / max(1, cfg.num_updates)
    eta = (elapsed_total / max(1, update)) * max(0, cfg.num_updates - update)
    total_sps = batch.size / max(update_seconds, 1e-9)

    timing_parts = []
    if "collect_seconds" in stats:
        timing_parts.append(f"采集耗时={stats['collect_seconds']:.2f}s")
        timing_parts.append(f"采集速度={stats.get('steps_per_second', 0):.0f}步/秒")
    if "update_seconds" in stats:
        timing_parts.append(f"优化耗时={stats['update_seconds']:.2f}s")
    timing_parts.append(f"本轮总耗时={update_seconds:.2f}s")
    timing_parts.append(f"本轮吞吐={total_sps:.0f}样本/秒")

    reward_mean = float(np.mean(batch.rew[: batch.size])) if batch.size else 0.0
    reward_std = float(np.std(batch.rew[: batch.size])) if batch.size else 0.0
    ron_rate = done["ron"] / denom
    tsumo_rate = done["tsumo"] / denom
    ryuukyoku_rate = done["ryuukyoku"] / denom
    mode = _format_training_mode(tag)
    worker_line = f"，工作进程={stats.get('workers', 1)}" if stats.get("multiprocessing") else ""
    timing_text = "，".join(timing_parts)

    return (
        f"{tag} {mode}状态：第 {update}/{cfg.num_updates} 轮更新（进度 {progress:.1%}）\n"
        f"  运行配置：设备={device}{worker_line}，环境数={cfg.num_envs}，本轮目标样本={cfg.target_transitions}。\n"
        f"  数据采集：本轮实际决策样本={batch.size}，环境动作步数={stats['steps']}，"
        f"自动 PASS 次数={stats['auto_pass']}（只有 PASS 可选时自动跳过）。\n"
        f"  终局统计：本轮终局数={terminal}；荣和={done['ron']}（{ron_rate:.2%}），"
        f"自摸={done['tsumo']}（{tsumo_rate:.2%}），流局={done['ryuukyoku']}（{ryuukyoku_rate:.2%}）。\n"
        f"  奖励统计：平均奖励={reward_mean:.4f}，奖励标准差={reward_std:.4f}；"
        f"奖励包含每步惩罚、手牌形状变化奖励和终局点差奖励。\n"
        f"  PPO 优化：总损失={metrics['loss']:.4f}，策略损失={metrics['pl']:.4f}，"
        f"价值损失={metrics['vl']:.4f}，策略熵={metrics['ent']:.4f}（熵越高表示动作分布越分散）。\n"
        f"  耗时统计：{timing_text}；累计训练={_format_duration(elapsed_total)}，预计剩余={_format_duration(eta)}。"
    )


def _hand_shape_score(hand34: List[int]) -> float:
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


def _sample_action(logits: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    neg_inf = torch.finfo(logits.dtype).min
    masked_logits = torch.where(mask > 0, logits, torch.tensor(neg_inf, dtype=logits.dtype, device=logits.device))
    dist = torch.distributions.Categorical(logits=masked_logits)
    act = dist.sample()
    return act, dist.log_prob(act), dist.entropy()


def _reward_from_step(step_res, actor: int, shaping_delta: float, cfg: TrainConfig) -> float:
    r = -cfg.step_penalty + cfg.shaping_coef * shaping_delta
    if step_res.done:
        r += cfg.reward_scale * float(step_res.score_delta[actor])
    return float(r)


def collect_parallel_batch(engines: List[RiichiEngine], model: ActorCritic, cfg: TrainConfig, device: str) -> Tuple[Batch, Dict]:
    obs_buf = np.zeros((cfg.target_transitions, OBS_DIM), dtype=np.float32)
    mask_buf = np.zeros((cfg.target_transitions, N_ACTIONS), dtype=np.float32)
    act_buf = np.zeros((cfg.target_transitions,), dtype=np.int64)
    logp_buf = np.zeros((cfg.target_transitions,), dtype=np.float32)
    rew_buf = np.zeros((cfg.target_transitions,), dtype=np.float32)
    done_buf = np.zeros((cfg.target_transitions,), dtype=np.float32)
    val_buf = np.zeros((cfg.target_transitions,), dtype=np.float32)
    env_buf = np.zeros((cfg.target_transitions,), dtype=np.int64)

    ptr = 0
    started_at = time.perf_counter()
    st = {"steps": 0, "auto_pass": 0, "done": {"tsumo": 0, "ron": 0, "ryuukyoku": 0}}

    while ptr < cfg.target_transitions:
        for i, e in enumerate(engines):
            if e.done:
                e.reset(dealer=(cfg.seed + i) % 4)
            st["auto_pass"] += _advance_until_decision(e)

        batch_env: List[int] = []
        batch_obs: List[np.ndarray] = []
        batch_mask: List[np.ndarray] = []
        batch_actor: List[int] = []
        pre_shape: List[float] = []

        for i, e in enumerate(engines):
            if e.done or e.phase == Phase.DRAW:
                continue
            actor = e.cur
            obs = e.get_obs(seat=actor)
            batch_env.append(i)
            batch_obs.append(obs_encoder(obs))
            batch_mask.append(mask_builder(e))
            batch_actor.append(actor)
            pre_shape.append(_hand_shape_score(obs["hand34"]))

        if not batch_env:
            continue

        x = torch.tensor(np.stack(batch_obs), dtype=torch.float32, device=device)
        m = torch.tensor(np.stack(batch_mask), dtype=torch.float32, device=device)

        with torch.no_grad():
            logits, values = model(x)
            aid, logp, _ = _sample_action(logits, m)

        for b, env_i in enumerate(batch_env):
            e = engines[env_i]
            actor = batch_actor[b]

            action = materialize_action(e, id_to_action(int(aid[b].item())))
            step_res = e.apply_action(action)
            post_obs = e.get_obs(seat=actor)
            shape_delta = _hand_shape_score(post_obs["hand34"]) - pre_shape[b]
            reward = _reward_from_step(step_res, actor, shape_delta, cfg)

            obs_buf[ptr] = batch_obs[b]
            mask_buf[ptr] = batch_mask[b]
            act_buf[ptr] = int(aid[b].item())
            logp_buf[ptr] = float(logp[b].item())
            rew_buf[ptr] = reward
            done_buf[ptr] = 1.0 if step_res.done else 0.0
            val_buf[ptr] = float(values[b].item())
            env_buf[ptr] = env_i

            ptr += 1
            st["steps"] += 1
            if step_res.done and step_res.reason in st["done"]:
                st["done"][step_res.reason] += 1
            if ptr >= cfg.target_transitions:
                break

    elapsed = max(time.perf_counter() - started_at, 1e-9)
    st["collect_seconds"] = elapsed
    st["steps_per_second"] = st["steps"] / elapsed

    return (
        Batch(
            obs=obs_buf,
            mask=mask_buf,
            act=act_buf,
            logp=logp_buf,
            rew=rew_buf,
            done=done_buf,
            val=val_buf,
            env_id=env_buf,
            size=int(ptr),
        ),
        st,
    )


def ppo_update(model: ActorCritic, optimizer: torch.optim.Optimizer, batch: Batch, cfg: TrainConfig) -> Dict[str, float]:
    if batch.size == 0:
        return {"loss": 0.0, "pl": 0.0, "vl": 0.0, "ent": 0.0}

    T = batch.size
    device = next(model.parameters()).device

    obs = torch.tensor(batch.obs[:T], dtype=torch.float32, device=device)
    mask = torch.tensor(batch.mask[:T], dtype=torch.float32, device=device)
    act = torch.tensor(batch.act[:T], dtype=torch.int64, device=device)
    old_logp = torch.tensor(batch.logp[:T], dtype=torch.float32, device=device)
    rew = torch.tensor(batch.rew[:T], dtype=torch.float32, device=device)
    done = torch.tensor(batch.done[:T], dtype=torch.float32, device=device)
    val = torch.tensor(batch.val[:T], dtype=torch.float32, device=device)
    env_id = batch.env_id[:T]

    adv = torch.zeros(T, dtype=torch.float32, device=device)
    gae_by_env = {int(e): 0.0 for e in np.unique(env_id)}
    next_value_by_env = {int(e): 0.0 for e in np.unique(env_id)}

    # Correct GAE for interleaved multi-env transitions.
    for t in reversed(range(T)):
        e = int(env_id[t])
        nonterminal = 1.0 - done[t]
        delta = rew[t] + cfg.gamma * float(next_value_by_env[e]) * nonterminal - val[t]
        gae = delta + cfg.gamma * cfg.lam * nonterminal * float(gae_by_env[e])
        gae_by_env[e] = float(gae)
        next_value_by_env[e] = float(val[t])
        adv[t] = gae

    ret = adv + val
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    idxs = np.arange(T)
    metrics = {"loss": [], "pl": [], "vl": [], "ent": []}
    for _ in range(cfg.ppo_epochs):
        np.random.shuffle(idxs)
        for start in range(0, T, cfg.ppo_batch_size):
            mb = idxs[start:start + cfg.ppo_batch_size]
            mb_obs = obs[mb]
            mb_mask = mask[mb]
            mb_act = act[mb]
            mb_old_logp = old_logp[mb]
            mb_adv = adv[mb]
            mb_ret = ret[mb]

            logits, value = model(mb_obs)
            neg_inf = torch.finfo(logits.dtype).min
            masked_logits = torch.where(mb_mask > 0, logits, torch.tensor(neg_inf, dtype=logits.dtype, device=device))
            dist = torch.distributions.Categorical(logits=masked_logits)
            logp = dist.log_prob(mb_act)
            entropy = dist.entropy().mean()

            ratio = torch.exp(logp - mb_old_logp)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(value, mb_ret)
            loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            metrics["loss"].append(float(loss.item()))
            metrics["pl"].append(float(policy_loss.item()))
            metrics["vl"].append(float(value_loss.item()))
            metrics["ent"].append(float(entropy.item()))

    return {k: (sum(v) / len(v) if v else 0.0) for k, v in metrics.items()}


def train(config: Optional[TrainConfig] = None) -> ActorCritic:
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
    train_started_at = time.perf_counter()
    if cfg.log_every > 0:
        print(format_train_start(tag="[SEQ]", cfg=cfg, device=device))

    for upd in range(1, cfg.num_updates + 1):
        update_started_at = time.perf_counter()
        batch, st = collect_parallel_batch(engines, model, cfg, device)
        update_started = time.perf_counter()
        met = ppo_update(model, optimizer, batch, cfg)
        update_seconds = time.perf_counter() - update_started
        total_update_seconds = time.perf_counter() - update_started_at
        st["update_seconds"] = update_seconds

        if cfg.log_every > 0 and (upd == 1 or upd % cfg.log_every == 0 or upd == cfg.num_updates):
            print(
                format_train_status(
                    tag="[SEQ]",
                    update=upd,
                    cfg=cfg,
                    device=device,
                    batch=batch,
                    stats=st,
                    metrics=met,
                    elapsed_total=time.perf_counter() - train_started_at,
                    update_seconds=total_update_seconds,
                )
            )

    return model


def evaluate(model: ActorCritic, config: Optional[EvalConfig] = None) -> Dict[str, float]:
    require_torch()
    cfg = config or EvalConfig()
    device = select_device(cfg.device)

    model = model.to(device)
    model.eval()

    reasons = {"tsumo": 0, "ron": 0, "ryuukyoku": 0, "other": 0}
    score_sum = 0.0

    for ep in range(cfg.episodes):
        e = RiichiEngine(seed=cfg.seed + ep, config=cfg.rules)
        e.reset(dealer=(cfg.seed + ep) % 4)

        steps = 0
        while not e.done and steps < cfg.max_steps:
            steps += 1
            _advance_until_decision(e)
            if e.done:
                break
            actor = e.cur
            obs = obs_encoder(e.get_obs(seat=actor))
            mask = mask_builder(e)

            x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            m = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, _ = model(x)
                neg_inf = torch.finfo(logits.dtype).min
                masked_logits = torch.where(m > 0, logits, torch.tensor(neg_inf, dtype=logits.dtype, device=device))
                if cfg.greedy:
                    aid = int(torch.argmax(masked_logits, dim=-1).item())
                else:
                    dist = torch.distributions.Categorical(logits=masked_logits)
                    aid = int(dist.sample().item())
            res = e.apply_action(materialize_action(e, id_to_action(aid)))
            if res.done:
                reasons[res.reason if res.reason in reasons else "other"] += 1
                score_sum += float(res.score_delta[0])
                break
        else:
            reasons["other"] += 1

    total = max(1, cfg.episodes)
    return {
        "episodes": float(cfg.episodes),
        "avg_score_delta_seat0": score_sum / total,
        "rate_tsumo": reasons["tsumo"] / total,
        "rate_ron": reasons["ron"] / total,
        "rate_ryuukyoku": reasons["ryuukyoku"] / total,
    }
