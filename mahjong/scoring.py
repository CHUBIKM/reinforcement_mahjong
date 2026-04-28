from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from _mahjong_cpp import (
    count_dora as _count_dora,
    resolve_ron as _resolve_ron,
    resolve_tsumo as _resolve_tsumo,
)


def dora_from_indicator(indicator: int) -> int:
    """Convert dora indicator tile index to actual dora tile index."""
    if 0 <= indicator <= 26:
        base = (indicator // 9) * 9
        pos = indicator % 9
        return base + ((pos + 1) % 9)
    if 27 <= indicator <= 30:
        return 27 + ((indicator - 27 + 1) % 4)
    if 31 <= indicator <= 33:
        return 31 + ((indicator - 31 + 1) % 3)
    raise ValueError(f"invalid indicator: {indicator}")


def count_dora(hand34: List[int], dora_indicators: List[int]) -> int:
    return _count_dora(hand34, dora_indicators)


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
    pr = _resolve_ron(winner, loser, han, fu, dealer, honba, riichi_sticks, kazoe_yakuman, kiriage_mangan)
    return PointResult(
        score_delta=list(pr.score_delta),
        payments=dict(pr.payments),
        level=pr.level,
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
    pr = _resolve_tsumo(winner, han, fu, dealer, honba, riichi_sticks, kazoe_yakuman, kiriage_mangan)
    return PointResult(
        score_delta=list(pr.score_delta),
        payments=dict(pr.payments),
        level=pr.level,
    )
