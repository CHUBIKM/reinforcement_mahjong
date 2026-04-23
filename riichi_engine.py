# -*- coding: utf-8 -*-
"""Backward-compatible entrypoint for the refactored Mahjong engine."""

from mahjong.engine import *  # noqa: F401,F403


if __name__ == "__main__":
    eng = RiichiEngine(seed=42, config=GameConfig())
    eng.reset(dealer=0)

    print("=== 初始牌局（仅自测展示，真实对局中他家手牌不可见）===")
    for p in eng.players:
        print(f"玩家{p.seat} 手牌：{hand_to_str(p.hand34)}")

    res = eng.play_random(verbose=False)

    print("\n=== 对局结果 ===")
    print(res)

    for p in eng.players:
        river_str = " ".join(tile_to_str(t) for t in p.river)
        print(f"玩家{p.seat} 河：{river_str}")
