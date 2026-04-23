from __future__ import annotations

from typing import Dict

import numpy as np

from mahjong.engine import Action, ActionType, Phase, RiichiEngine


# Fixed action space
# 0..33   DISCARD(tile)
# 34      TSUMO
# 35      PASS
# 36      RON
# 37      PON
# 38..40  CHI pattern (left/mid/right)
# 41..74  KAN(tile)
# 75..108 RIICHI+DISCARD(tile)
# 109     ABORTIVE_DRAW (九种九牌)
N_ACTIONS = 110


def action_to_id(a: Action) -> int:
    if a.type == ActionType.DISCARD:
        return int(a.tile)
    if a.type == ActionType.TSUMO:
        return 34
    if a.type == ActionType.PASS:
        return 35
    if a.type == ActionType.RON:
        return 36
    if a.type == ActionType.PON:
        return 37
    if a.type == ActionType.CHI:
        t = int(a.tile) if a.tile is not None else None
        use = (a.info or {}).get("use")
        if t is None or not use or len(use) != 2:
            raise ValueError("CHI action requires tile/use")
        u0, u1 = sorted((int(use[0]), int(use[1])))
        if u0 == t - 2 and u1 == t - 1:
            return 38
        if u0 == t - 1 and u1 == t + 1:
            return 39
        if u0 == t + 1 and u1 == t + 2:
            return 40
        raise ValueError(f"invalid chi use: tile={t} use={use}")
    if a.type == ActionType.KAN:
        return 41 + int(a.tile)
    if a.type == ActionType.RIICHI:
        return 75 + int(a.tile)
    if a.type == ActionType.ABORTIVE_DRAW:
        return 109
    raise ValueError(f"unsupported action type: {a.type}")


def id_to_action(aid: int) -> Action:
    if 0 <= aid <= 33:
        return Action(ActionType.DISCARD, tile=aid)
    if aid == 34:
        return Action(ActionType.TSUMO)
    if aid == 35:
        return Action(ActionType.PASS)
    if aid == 36:
        return Action(ActionType.RON)
    if aid == 37:
        return Action(ActionType.PON)
    if 38 <= aid <= 40:
        return Action(ActionType.CHI, info={"chi_pattern": aid - 38})
    if 41 <= aid <= 74:
        return Action(ActionType.KAN, tile=aid - 41, info={"kan_type": "ANKAN"})
    if 75 <= aid <= 108:
        return Action(ActionType.RIICHI, tile=aid - 75)
    if aid == 109:
        return Action(ActionType.ABORTIVE_DRAW)
    raise ValueError(f"illegal action id: {aid}")


def mask_builder(engine: RiichiEngine) -> np.ndarray:
    mask = np.zeros((N_ACTIONS,), dtype=np.float32)
    for a in engine.legal_actions():
        mask[action_to_id(a)] = 1.0
    return mask


def materialize_action(engine: RiichiEngine, action: Action) -> Action:
    pd = engine.pending_discard
    if action.type in (ActionType.RON, ActionType.PON):
        if not pd:
            return Action(ActionType.PASS)
        tile = pd["tile"]
        frm = pd["player"]
        if action.type == ActionType.RON:
            return Action(ActionType.RON, tile=tile, info={"from": frm})
        return Action(ActionType.PON, tile=tile, info={"from": frm})

    if action.type == ActionType.CHI:
        if not pd:
            return Action(ActionType.PASS)
        t = pd["tile"]
        frm = pd["player"]
        actor = engine.cur
        if actor != (frm + 1) % 4 or t < 0 or t > 26:
            return Action(ActionType.PASS)
        pat = (action.info or {}).get("chi_pattern", 1)
        if pat == 0:
            use = [t - 2, t - 1]
        elif pat == 1:
            use = [t - 1, t + 1]
        else:
            use = [t + 1, t + 2]
        if min(use) < 0 or max(use) > 26:
            return Action(ActionType.PASS)
        if (use[0] // 9) != (t // 9) or (use[1] // 9) != (t // 9):
            return Action(ActionType.PASS)
        return Action(ActionType.CHI, tile=t, info={"from": frm, "use": use})

    if action.type == ActionType.KAN:
        if engine.phase == Phase.RESPONSE and pd:
            return Action(ActionType.KAN, tile=pd["tile"], info={"from": pd["player"], "kan_type": "MINKAN"})
        # DISCARD phase: choose concrete legal KAN variant by tile.
        for a in engine.legal_actions():
            if a.type == ActionType.KAN and a.tile == action.tile:
                return a
        return action

    return action


def _phase_one_hot(phase: Phase) -> np.ndarray:
    # DRAW/DISCARD/RESPONSE/END
    v = np.zeros((4,), dtype=np.float32)
    if str(phase) == str(Phase.DRAW):
        v[0] = 1.0
    elif str(phase) == str(Phase.DISCARD):
        v[1] = 1.0
    elif str(phase) == str(Phase.RESPONSE):
        v[2] = 1.0
    else:
        v[3] = 1.0
    return v


def obs_encoder(obs: Dict) -> np.ndarray:
    hand34 = np.array(obs["hand34"], dtype=np.float32)

    rivers = obs["rivers"]
    flat_river = np.fromiter((int(t) for r in rivers for t in r), dtype=np.int64)
    river_hist = np.bincount(flat_river, minlength=34).astype(np.float32) if flat_river.size else np.zeros((34,), dtype=np.float32)

    melds = obs.get("melds", [])
    flat_meld = np.fromiter((int(t) for pm in melds for _, ts in pm for t in ts), dtype=np.int64)
    meld_hist = np.bincount(flat_meld, minlength=34).astype(np.float32) if flat_meld.size else np.zeros((34,), dtype=np.float32)

    phase_oh = _phase_one_hot(obs["phase"])
    seat = int(obs["seat"])
    seat_oh = np.zeros((4,), dtype=np.float32)
    seat_oh[seat] = 1.0

    cur = float(obs["cur"])
    turn = float(obs["turn"])
    live_len = float(obs["live_wall_len"])
    dead_len = float(obs["dead_wall_len"])
    last_discard = -1.0 if obs["last_discard"] is None else float(obs["last_discard"])
    last_discarder = -1.0 if obs["last_discarder"] is None else float(obs["last_discarder"])

    dora = obs.get("dora_indicators", [])
    dora_pad = np.full((4,), -1.0, dtype=np.float32)
    for i in range(min(4, len(dora))):
        dora_pad[i] = float(dora[i])

    scores = np.array(obs.get("scores", [25000, 25000, 25000, 25000]), dtype=np.float32) / 10000.0
    riichi_declared = np.array(obs.get("riichi_declared", [False] * 4), dtype=np.float32)
    riichi_sticks = float(obs.get("riichi_sticks", 0))
    honba = float(obs.get("honba", 0))

    return np.concatenate(
        [
            hand34,
            river_hist,
            meld_hist,
            phase_oh,
            seat_oh,
            np.array([cur, turn, live_len, dead_len, last_discard, last_discarder, riichi_sticks, honba], dtype=np.float32),
            dora_pad,
            scores,
            riichi_declared,
        ],
        axis=0,
    )


OBS_DIM = 34 + 34 + 34 + 4 + 4 + 8 + 4 + 4 + 4
