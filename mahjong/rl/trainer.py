from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

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
            "PyTorch is required for training/evaluation. Install it first, e.g. `pip install torch` "
            "(or follow the official wheel selector for your OS/device)."
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
        raise ValueError(f"Unknown rules keys in train config: {sorted(unknown)}")
    return replace(base, **patch)


def load_train_config(path: str, base: Optional[TrainConfig] = None) -> TrainConfig:
    """Load TrainConfig from TOML file.

    Supported layout:
    - top-level train fields (e.g. num_updates, num_envs, lr, ...)
    - optional [rules] table for RuleProfile overrides.
    """
    if tomllib is None:
        raise RuntimeError("tomllib is unavailable in this Python runtime; cannot read TOML train config.")

    cfg = base or TrainConfig()
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Train config TOML must parse to a table/dict.")

    raw_rules = raw.get("rules", {})
    if raw_rules is None:
        raw_rules = {}
    if not isinstance(raw_rules, dict):
        raise ValueError("[rules] in train config must be a table.")

    unknown_top = set(raw.keys()) - (_train_config_field_names() | {"rules"})
    if unknown_top:
        raise ValueError(f"Unknown train config keys: {sorted(unknown_top)}")

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

    for upd in range(1, cfg.num_updates + 1):
        batch, st = collect_parallel_batch(engines, model, cfg, device)
        met = ppo_update(model, optimizer, batch, cfg)

        if cfg.log_every > 0 and upd % cfg.log_every == 0:
            d = st["done"]
            denom = max(1, d["tsumo"] + d["ron"] + d["ryuukyoku"])
            print(
                f"[UPD {upd}] device={device} transitions={batch.size} steps={st['steps']} autoPASS={st['auto_pass']} "
                f"rate(ron/tsumo/ryu)={d['ron']/denom:.2%}/{d['tsumo']/denom:.2%}/{d['ryuukyoku']/denom:.2%} "
                f"loss={met['loss']:.4f} pl={met['pl']:.4f} vl={met['vl']:.4f} ent={met['ent']:.4f}"
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
