from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuleProfile:
    """Rule toggles for single-hand Riichi Mahjong.

    Default profile aims to match common Mahjong Soul table settings.
    """

    name: str = "mahjongsoul_common"
    allow_multi_ron: bool = False
    use_dead_wall: bool = True
    dead_wall_size: int = 14
    enable_kan: bool = True
    max_kan_per_hand: int = 4

    # Special draws / table abortive draws
    enable_nagashi_mangan: bool = True
    enable_kyuushu_kyuuhai: bool = True
    enable_suufon_renda: bool = True
    enable_suucha_riichi: bool = True
    enable_suukan_sanra: bool = True

    # Scoring toggles
    aka_dora_count: int = 3
    ura_dora_enabled: bool = True
    kazoe_yakuman: bool = True
    kiriage_mangan: bool = False

    # Riichi / furiten behavior
    enforce_furiten: bool = True
    enforce_same_turn_furiten: bool = True
    enforce_riichi_furiten: bool = True


DEFAULT_RULE_PROFILE = RuleProfile()
