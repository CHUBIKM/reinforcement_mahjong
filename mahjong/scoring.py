from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import Dict, List, Tuple


def _ceil100(x: int) -> int:
    return int(ceil(x / 100.0) * 100)


def dora_from_indicator(indicator: int) -> int:
    """Convert dora indicator tile index to actual dora tile index."""
    if 0 <= indicator <= 26:
        base = (indicator // 9) * 9
        pos = indicator % 9
        return base + ((pos + 1) % 9)
    if 27 <= indicator <= 30:  # winds
        return 27 + ((indicator - 27 + 1) % 4)
    if 31 <= indicator <= 33:  # dragons
        return 31 + ((indicator - 31 + 1) % 3)
    raise ValueError(f"invalid indicator: {indicator}")


def count_dora(hand34: List[int], dora_indicators: List[int]) -> int:
    dora_tiles = [dora_from_indicator(t) for t in dora_indicators]
    total = 0
    for t in dora_tiles:
        total += hand34[t]
    return total


@dataclass
class PointResult:
    """Resolved point transfer for one terminal result."""

    score_delta: List[int]
    payments: Dict[str, int] = field(default_factory=dict)
    level: str = "none"


def point_level(han: int, fu: int, kazoe_yakuman: bool = True, kiriage_mangan: bool = False) -> str:
    if han >= 13:
        return "yakuman"
    if han >= 11:
        return "sanbaiman"
    if han >= 8:
        return "baiman"
    if han >= 6:
        return "haneman"
    if han >= 5:
        return "mangan"
    if han == 4 and fu >= 40:
        return "mangan"
    if han == 3 and fu >= 70:
        return "mangan"
    if kiriage_mangan and ((han == 4 and fu == 30) or (han == 3 and fu == 60)):
        return "mangan"
    return "regular"


def _base_points(han: int, fu: int, kazoe_yakuman: bool = True, kiriage_mangan: bool = False) -> int:
    level = point_level(han, fu, kazoe_yakuman=kazoe_yakuman, kiriage_mangan=kiriage_mangan)
    if level == "yakuman":
        return 8000
    if level == "sanbaiman":
        return 6000
    if level == "baiman":
        return 4000
    if level == "haneman":
        return 3000
    if level == "mangan":
        return 2000
    return min(2000, fu * (2 ** (han + 2)))


def resolve_ron(
    winner: int,
    loser: int,
    han: int,
    fu: int,
    dealer: int,
    honba: int = 0,
    riichi_sticks: int = 0,
    kazoe_yakuman: bool = True,
    kiriage_mangan: bool = False,
) -> PointResult:
    base = _base_points(han=han, fu=fu, kazoe_yakuman=kazoe_yakuman, kiriage_mangan=kiriage_mangan)
    is_dealer = winner == dealer
    ron_points = _ceil100(base * (6 if is_dealer else 4))
    honba_bonus = honba * 300

    delta = [0, 0, 0, 0]
    total_gain = ron_points + honba_bonus + (riichi_sticks * 1000)
    total_loss = ron_points + honba_bonus

    delta[winner] += total_gain
    delta[loser] -= total_loss

    return PointResult(
        score_delta=delta,
        payments={
            "ron": ron_points,
            "honba": honba_bonus,
            "riichi_sticks": riichi_sticks * 1000,
        },
        level=point_level(han=han, fu=fu, kazoe_yakuman=kazoe_yakuman, kiriage_mangan=kiriage_mangan),
    )


def resolve_tsumo(
    winner: int,
    han: int,
    fu: int,
    dealer: int,
    honba: int = 0,
    riichi_sticks: int = 0,
    kazoe_yakuman: bool = True,
    kiriage_mangan: bool = False,
) -> PointResult:
    base = _base_points(han=han, fu=fu, kazoe_yakuman=kazoe_yakuman, kiriage_mangan=kiriage_mangan)
    is_dealer = winner == dealer

    delta = [0, 0, 0, 0]
    if is_dealer:
        pay_each = _ceil100(base * 2)
        for p in range(4):
            if p == winner:
                continue
            delta[p] -= pay_each + honba * 100
            delta[winner] += pay_each + honba * 100
        payments = {"from_each": pay_each, "honba_each": honba * 100, "riichi_sticks": riichi_sticks * 1000}
    else:
        pay_non_dealer = _ceil100(base)
        pay_dealer = _ceil100(base * 2)
        for p in range(4):
            if p == winner:
                continue
            if p == dealer:
                amt = pay_dealer + honba * 100
            else:
                amt = pay_non_dealer + honba * 100
            delta[p] -= amt
            delta[winner] += amt
        payments = {
            "from_dealer": pay_dealer,
            "from_non_dealer": pay_non_dealer,
            "honba_each": honba * 100,
            "riichi_sticks": riichi_sticks * 1000,
        }

    delta[winner] += riichi_sticks * 1000

    return PointResult(
        score_delta=delta,
        payments=payments,
        level=point_level(han=han, fu=fu, kazoe_yakuman=kazoe_yakuman, kiriage_mangan=kiriage_mangan),
    )
