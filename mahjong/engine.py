# riichi_engine.py
# -*- coding: utf-8 -*-

"""
更完善的日麻（立直麻将）MVP引擎（用于 RL 起步）：
- 牌山：开局一次性生成 136 张并洗牌（可复现 seed），之后顺序摸牌
- 4人配牌13张
- 回合：摸牌 -> 自摸判定 -> 弃牌 -> 其他玩家荣和判定 -> 轮转
- 终局：自摸 / 荣和 / 牌山耗尽流局
- 放铳归因：记录放铳者（谁打出致胜牌）与荣和者（谁荣和）
- 提供合法弃牌 mask（34维）与基础 obs（dict）

注意：
- 这里只判定“结构和牌”（4面子1雀头 / 七对 / 国士），不算役、不算番，不做点数结算
- 不包含鸣牌/立直/宝牌/岭上/振听等；后续可以逐步加
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Set
import random
from enum import Enum

from mahjong.rules import RuleProfile, DEFAULT_RULE_PROFILE
from mahjong.scoring import count_dora, resolve_ron, resolve_tsumo
from mahjong.replay import Replay


# ============================================================
# 一、牌的编码（34种牌型）
# ------------------------------------------------------------
# 0 -  8  : 1m - 9m (万子 manzu)
# 9 - 17  : 1p - 9p (筒子 pinzu)
# 18 - 26 : 1s - 9s (索子 souzu)
# 27 - 33 : 字牌 honors
#           27:东 28:南 29:西 30:北 31:白 32:发 33:中
#
# 每种牌有 4 张实体牌，所以全牌山一共 34*4 = 136 张。
# ============================================================

SUIT_NAMES = ["m", "p", "s"]
HONOR_NAMES = ["东", "南", "西", "北", "白", "发", "中"]


def tile_to_str(t: int) -> str:
    """将 0..33 的牌型索引转换为可读字符串。"""
    if 0 <= t <= 26:
        suit = SUIT_NAMES[t // 9]
        num = (t % 9) + 1
        return f"{num}{suit}"
    if 27 <= t <= 33:
        return HONOR_NAMES[t - 27]
    return f"<?>({t})"


def hand_to_str(hand34: List[int]) -> str:
    """将 34维计数手牌打印为紧凑字符串（仅用于调试）。"""
    parts = []
    for s in range(3):
        nums = []
        for i in range(9):
            t = s * 9 + i
            nums += [str(i + 1)] * hand34[t]
        if nums:
            parts.append("".join(nums) + SUIT_NAMES[s])

    honors = []
    for i in range(7):
        t = 27 + i
        honors += [HONOR_NAMES[i]] * hand34[t]
    if honors:
        parts.append("".join(honors))

    return " ".join(parts) if parts else "(空)"


def make_wall(rng: random.Random) -> List[int]:
    """
    生成并洗牌完整牌山（136张）。

    重要：必须“开局一次性生成并洗牌”，之后按顺序摸牌（pop）。
    不要每次摸牌临时随机生成，否则会破坏“每张牌最多4张”的物理约束，
    也无法定义剩余牌分布、王牌/宝牌/岭上等规则。
    """
    wall: List[int] = []
    for t in range(34):
        wall.extend([t] * 4)
    rng.shuffle(wall)
    return wall


def hand34_add(hand: List[int], t: int) -> None:
    """手牌加入一张 t（34计数）。"""
    hand[t] += 1


def hand34_remove(hand: List[int], t: int) -> None:
    """手牌移除一张 t（34计数），若没有则报错。"""
    if hand[t] <= 0:
        raise ValueError(f"手里没有 {tile_to_str(t)}，无法移除。")
    hand[t] -= 1


def copy_hand(hand: List[int]) -> List[int]:
    """复制 34维计数手牌（避免递归判定改坏原数据）。"""
    return list(hand)


# ============================================================
# 二、和牌判定（Agari）：结构判定，不含役/番/符
# ------------------------------------------------------------
# 支持：
# - 标准形：4面子 + 1雀头（顺子/刻子）
# - 七对子
# - 国士无双
# ============================================================

KOKUSHI_SET = set([0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33])


def is_kokushi(hand34: List[int]) -> bool:
    """
    国士无双：
    - 13种幺九字每种至少1张
    - 其中一种>=2张（作对子）
    - 不允许出现其他牌
    """
    dup = False
    for t in KOKUSHI_SET:
        if hand34[t] == 0:
            return False
        if hand34[t] >= 2:
            dup = True

    for t in range(34):
        if t not in KOKUSHI_SET and hand34[t] != 0:
            return False

    return dup


def is_chiitoi(hand34: List[int]) -> bool:
    """
    七对子：
    - 恰好 7 个对子
    - 任何牌只能是 0 或 2（不能有1/3/4）
    """
    pairs = 0
    for c in hand34:
        if c == 2:
            pairs += 1
        elif c == 0:
            pass
        else:
            return False
    return pairs == 7


def can_form_melds(counts: List[int], start: int = 0) -> bool:
    """
    递归拆面子：counts 是否能完全拆成若干面子（刻子/顺子），不含雀头。
    """
    i = start
    while i < 34 and counts[i] == 0:
        i += 1
    if i == 34:
        return True

    # 尝试刻子
    if counts[i] >= 3:
        counts[i] -= 3
        if can_form_melds(counts, i):
            counts[i] += 3
            return True
        counts[i] += 3

    # 尝试顺子（仅数牌）
    if 0 <= i <= 26:
        suit = i // 9
        pos = i % 9
        if pos <= 6:
            a, b, c = i, i + 1, i + 2
            if (b // 9) == suit and (c // 9) == suit and counts[b] > 0 and counts[c] > 0:
                counts[a] -= 1
                counts[b] -= 1
                counts[c] -= 1
                if can_form_melds(counts, i):
                    counts[a] += 1
                    counts[b] += 1
                    counts[c] += 1
                    return True
                counts[a] += 1
                counts[b] += 1
                counts[c] += 1

    return False


def is_standard_agari(hand34: List[int]) -> bool:
    """
    标准形：
    - 枚举雀头（任意牌 count>=2）
    - 去掉雀头后，剩余必须可完全拆成面子
    """
    for t in range(34):
        if hand34[t] >= 2:
            counts = copy_hand(hand34)
            counts[t] -= 2
            if can_form_melds(counts, 0):
                return True
    return False


def is_agari(hand34: List[int]) -> bool:
    """综合判定（输入必须是14张）。"""
    if sum(hand34) != 14:
        return False
    if is_kokushi(hand34):
        return True
    if is_chiitoi(hand34):
        return True
    return is_standard_agari(hand34)


def count_yaochu_types(hand34: List[int]) -> int:
    yaochu = set([0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33])
    return sum(1 for t in yaochu if hand34[t] > 0)


def is_tenpai(hand34: List[int]) -> bool:
    """Tenpai check for a 13-tile hand via brute-force agari wait scan."""
    if sum(hand34) != 13:
        return False
    for t in range(34):
        if hand34[t] >= 4:
            continue
        tmp = copy_hand(hand34)
        tmp[t] += 1
        if is_agari(tmp):
            return True
    return False

# ============================================================
# 三、役种判定（Yaku）：在“结构和牌”基础上，识别常见役种并给出番数
# ------------------------------------------------------------
# 说明（重要）：
# 1) 当前引擎还没有实现鸣牌/立直/宝牌/计符/点数结算，所以这里先做：
#    - 役种识别（返回役种列表）
#    - 番数（han）仅按“门清（无副露）”默认番数计算
# 2) 役种判定依赖“面子拆分”（4面子+1雀头）。为减少歧义：
#    - 我们枚举所有可能的拆分，只要存在一种拆分满足某役种条件，就认为该役成立。
# 3) 七对子、国士无双属于特殊和牌形式，不需要面子拆分。
# ------------------------------------------------------------
# 目前支持的役种（门清番数）：
# - 门前清自摸和(1)
# - 断幺九(1)
# - 役牌(1)：白/发/中 + 场风 + 自风（需要刻子）
# - 一杯口(1)
# - 三色同顺(2)
# - 一气通贯(2)
# - 对对和(2)
# - 三暗刻(2)
# - 混一色(3)
# - 清一色(6)
# - 混全带幺九(2)
# - 纯全带幺九(3)
# - 混老头(2)
# - 小三元(2)
# - 七对子(2)
# - 国士无双(役满，标记为 13 番仅作占位)
#
# 不支持/暂不实现：平和、二杯口、三杠子、岭上开花、抢杠、海底/河底、立直/一发等
# ============================================================

TERMINALS = set([0, 8, 9, 17, 18, 26])
HONORS = set(range(27, 34))
DRAGONS = set([31, 32, 33])


def _is_terminal_or_honor(t: int) -> bool:
    return t in HONORS or t in TERMINALS


def _tile_suit(t: int) -> Optional[int]:
    """返回花色：0=m,1=p,2=s；字牌返回 None"""
    if 0 <= t <= 26:
        return t // 9
    return None


def _meld_type(meld: Tuple[str, List[int]]) -> str:
    """meld 形如 ("seq", [a,b,c]) 或 ("trip", [t,t,t])"""
    return meld[0]


def _gen_standard_decompositions(hand34: List[int]) -> List[Tuple[int, List[Tuple[str, List[int]]]]]:
    """
    枚举所有标准形拆分：返回列表，每个元素是 (雀头牌型, 面子列表)
    面子列表元素：
      - ("seq", [a,b,c]) 顺子
      - ("trip", [t,t,t]) 刻子

    注意：这会产生一定数量的拆分，但用于 MVP 够用。
    """
    if sum(hand34) != 14:
        return []

    results: List[Tuple[int, List[Tuple[str, List[int]]]]] = []

    def backtrack(counts: List[int], start: int, melds: List[Tuple[str, List[int]]]):
        # 找到下一张还没用完的牌
        i = start
        while i < 34 and counts[i] == 0:
            i += 1
        if i == 34:
            # 所有牌都被拆完，得到一个拆分
            results.append((pair_tile, [m for m in melds]))
            return

        # 尝试刻子
        if counts[i] >= 3:
            counts[i] -= 3
            melds.append(("trip", [i, i, i]))
            backtrack(counts, i, melds)
            melds.pop()
            counts[i] += 3

        # 尝试顺子（仅数牌）
        if 0 <= i <= 26:
            suit = i // 9
            pos = i % 9
            if pos <= 6:
                a, b, c = i, i + 1, i + 2
                if (b // 9) == suit and (c // 9) == suit and counts[b] > 0 and counts[c] > 0:
                    counts[a] -= 1
                    counts[b] -= 1
                    counts[c] -= 1
                    melds.append(("seq", [a, b, c]))
                    backtrack(counts, i, melds)
                    melds.pop()
                    counts[a] += 1
                    counts[b] += 1
                    counts[c] += 1

    # 枚举雀头
    for t in range(34):
        if hand34[t] >= 2:
            counts = copy_hand(hand34)
            counts[t] -= 2
            pair_tile = t
            backtrack(counts, 0, [])

    return results


def _count_suits_in_hand(hand34: List[int]) -> Tuple[set, bool]:
    """返回：出现过的数牌花色集合 + 是否含字牌"""
    suits = set()
    has_honor = False
    for t, c in enumerate(hand34):
        if c <= 0:
            continue
        s = _tile_suit(t)
        if s is None:
            has_honor = True
        else:
            suits.add(s)
    return suits, has_honor


def _is_tanyao(hand34: List[int]) -> bool:
    """断幺九：不含幺九/字牌"""
    for t, c in enumerate(hand34):
        if c <= 0:
            continue
        if _is_terminal_or_honor(t):
            return False
    return True


def _yakuhai_from_melds(melds: List[Tuple[str, List[int]]], seat_wind: int, round_wind: int) -> List[str]:
    """役牌：白/发/中 + 场风 + 自风（刻子成立）"""
    names = []
    for mtype, tiles in melds:
        if mtype != "trip":
            continue
        t = tiles[0]
        if t in DRAGONS:
            names.append(f"役牌·{tile_to_str(t)}")
        if t == seat_wind:
            names.append("役牌·自风")
        if t == round_wind:
            names.append("役牌·场风")
    return names


def _is_toitoi(melds: List[Tuple[str, List[int]]]) -> bool:
    return all(_meld_type(m) == "trip" for m in melds)


def _is_sanankou(melds: List[Tuple[str, List[int]]]) -> bool:
    """
    三暗刻：三组刻子。
    注：严格规则里“明暗刻”与和牌方式有关（荣和会把荣和牌那组刻子算明刻等细节）。
    MVP 先简化为：手牌拆分中刻子数量 >=3 即认为满足。
    """
    return sum(1 for m in melds if _meld_type(m) == "trip") >= 3


def _seq_key(seq: List[int]) -> Tuple[int, int]:
    """顺子键： (suit, start_pos)"""
    a = seq[0]
    return (_tile_suit(a) or -1, a % 9)


def _is_iipeikou(melds: List[Tuple[str, List[int]]]) -> bool:
    """一杯口：同花色同起点的顺子出现两次（门清限定；目前引擎默认门清）"""
    seqs = [m[1] for m in melds if _meld_type(m) == "seq"]
    keys = [_seq_key(s) for s in seqs]
    for k in set(keys):
        if keys.count(k) >= 2:
            return True
    return False


def _is_sanshoku_doujun(melds: List[Tuple[str, List[int]]]) -> bool:
    """三色同顺：三门（m/p/s）各有同一数字起点的顺子"""
    seqs = [m[1] for m in melds if _meld_type(m) == "seq"]
    # 记录每个起点pos(0..6)在各花色是否出现
    seen = {pos: set() for pos in range(7)}
    for s in seqs:
        a = s[0]
        suit = _tile_suit(a)
        if suit is None:
            continue
        pos = a % 9
        if 0 <= pos <= 6:
            seen[pos].add(suit)
    return any(len(seen[pos]) == 3 for pos in range(7))


def _is_ittsuu(melds: List[Tuple[str, List[int]]]) -> bool:
    """一气通贯：同一花色内 123/456/789 三组顺子齐全"""
    seqs = [m[1] for m in melds if _meld_type(m) == "seq"]
    by_suit = {0: set(), 1: set(), 2: set()}
    for s in seqs:
        a = s[0]
        suit = _tile_suit(a)
        if suit is None:
            continue
        by_suit[suit].add(a % 9)  # 记录顺子起点 0..6
    for suit in (0, 1, 2):
        if 0 in by_suit[suit] and 3 in by_suit[suit] and 6 in by_suit[suit]:
            return True
    return False


def _each_meld_has_terminal_or_honor(pair_tile: int, melds: List[Tuple[str, List[int]]], allow_honor: bool) -> bool:
    """用于混全带幺九/纯全带幺九：每一组（含雀头）都必须含幺九/字牌"""
    # 雀头检查
    if allow_honor:
        if not _is_terminal_or_honor(pair_tile):
            return False
    else:
        # 纯全：只允许终端，不允许字牌
        if pair_tile in HONORS:
            return False
        if pair_tile not in TERMINALS:
            return False

    for mtype, tiles in melds:
        if allow_honor:
            if not any(_is_terminal_or_honor(t) for t in tiles):
                return False
        else:
            # 纯全：每组必须含终端，且不能含字牌
            if any(t in HONORS for t in tiles):
                return False
            if not any(t in TERMINALS for t in tiles):
                return False

    return True


def _is_honroutou(hand34: List[int], melds: List[Tuple[str, List[int]]]) -> bool:
    """混老头：全部由幺九牌+字牌组成，且必须是对对和系（不允许顺子）。"""
    # 全部牌都是幺九字
    for t, c in enumerate(hand34):
        if c <= 0:
            continue
        if not _is_terminal_or_honor(t):
            return False
    # 不允许顺子
    return all(_meld_type(m) == "trip" for m in melds)


def _is_shousangen(pair_tile: int, melds: List[Tuple[str, List[int]]]) -> bool:
    """小三元：三元牌中两副刻子 + 一副雀头"""
    trip_dragons = 0
    has_pair_dragon = pair_tile in DRAGONS
    for mtype, tiles in melds:
        if mtype == "trip" and tiles[0] in DRAGONS:
            trip_dragons += 1
    return trip_dragons == 2 and has_pair_dragon


def analyze_yaku(
    hand34: List[int],
    win_type: str,  # "tsumo" or "ron"
    seat_wind: int,
    round_wind: int,
    is_closed: bool = True,
) -> Tuple[List[Tuple[str, int]], int]:
    """
    识别役种并返回 ([(役名, 番数), ...], 总番数)。

    参数：
    - hand34: 14张的和牌手牌（对 ron 来说，必须是“包含荣和牌后的14张”）
    - win_type: "tsumo" or "ron"
    - seat_wind/round_wind: 自风/场风（用 27..30 表示东南西北）
    - is_closed: 是否门清（当前引擎无鸣牌，默认 True）

    说明：
    - 对于役满，这里用 13 番作为占位（方便 RL 先跑通），后续可改为单独标记 yakuman。
    """
    yakus: List[Tuple[str, int]] = []

    # 特殊和牌：国士/七对
    if is_kokushi(hand34):
        yakus.append(("国士无双(役满)", 13))
        return yakus, sum(h for _, h in yakus)
    if is_chiitoi(hand34):
        yakus.append(("七对子", 2))
        # 七对子也可能同时满足断幺/清一色等，但严格规则下七对子与若干役可叠加。
        # MVP 为了简单，我们仍允许叠加基础色役/断幺。
        if _is_tanyao(hand34):
            yakus.append(("断幺九", 1))
        suits, has_honor = _count_suits_in_hand(hand34)
        if len(suits) == 1:
            if has_honor:
                yakus.append(("混一色", 3))
            else:
                yakus.append(("清一色", 6))
        return yakus, sum(h for _, h in yakus)

    # 标准形：枚举拆分
    decomps = _gen_standard_decompositions(hand34)
    if not decomps:
        return yakus, 0

    # 门前清自摸和
    if win_type == "tsumo" and is_closed:
        yakus.append(("门前清自摸和", 1))

    # 断幺九
    if _is_tanyao(hand34):
        yakus.append(("断幺九", 1))

    # 清/混一色
    suits, has_honor = _count_suits_in_hand(hand34)
    if len(suits) == 1:
        if has_honor:
            yakus.append(("混一色", 3))
        else:
            yakus.append(("清一色", 6))

    # 以下役需要看“拆分”（因为跟面子类型有关）
    # 我们采用“存在某个拆分满足”就成立的方式。

    # 役牌：每个刻子都可能产生多个役牌（白发中/自风/场风）
    yakuhai_names = set()
    for pair_tile, melds in decomps:
        for name in _yakuhai_from_melds(melds, seat_wind, round_wind):
            yakuhai_names.add(name)
    for name in sorted(yakuhai_names):
        yakus.append((name, 1))

    # 对对和
    if any(_is_toitoi(melds) for _, melds in decomps):
        yakus.append(("对对和", 2))

    # 三暗刻（简化版）
    if any(_is_sanankou(melds) for _, melds in decomps):
        yakus.append(("三暗刻", 2))

    # 一杯口（门清限定；我们默认门清）
    if is_closed and any(_is_iipeikou(melds) for _, melds in decomps):
        yakus.append(("一杯口", 1))

    # 三色同顺
    if any(_is_sanshoku_doujun(melds) for _, melds in decomps):
        yakus.append(("三色同顺", 2 if is_closed else 1))

    # 一气通贯
    if any(_is_ittsuu(melds) for _, melds in decomps):
        yakus.append(("一气通贯", 2 if is_closed else 1))

    # 混全带幺九 / 纯全带幺九
    if any(_each_meld_has_terminal_or_honor(pair_tile, melds, allow_honor=True) for pair_tile, melds in decomps):
        yakus.append(("混全带幺九", 2 if is_closed else 1))
    if any(_each_meld_has_terminal_or_honor(pair_tile, melds, allow_honor=False) for pair_tile, melds in decomps):
        yakus.append(("纯全带幺九", 3 if is_closed else 2))

    # 混老头
    if any(_is_honroutou(hand34, melds) for _, melds in decomps):
        yakus.append(("混老头", 2))

    # 小三元
    if any(_is_shousangen(pair_tile, melds) for pair_tile, melds in decomps):
        yakus.append(("小三元", 2))

    total_han = sum(h for _, h in yakus)
    return yakus, total_han


# ============================================================
# 四、数据结构
# ============================================================

@dataclass
class PlayerState:
    """单局中单个玩家状态（后续扩展鸣牌/立直等字段）。"""
    seat: int
    hand34: List[int] = field(default_factory=lambda: [0] * 34)
    river: List[int] = field(default_factory=list)  # 弃牌序列（顺序重要）
    melds: List[Tuple[str, List[int]]] = field(default_factory=list)  # 副露占位
    riichi_declared: bool = False
    riichi_turn: Optional[int] = None


@dataclass
class StepResult:
    """
    step() 的结果：
    - done: 是否终局
    - reason: "continue"/"tsumo"/"ron"/"ryuukyoku"
    - winner / winners:
        * tsumo：winner=自摸者
        * ron：winner=（allow_multi_ron=False时）第一荣和者；winners 记录所有荣和者
    - loser:
        * ron：放铳者 seat
    """
    done: bool
    reason: str
    winner: Optional[int] = None
    winners: List[int] = field(default_factory=list)
    loser: Optional[int] = None
    score_delta: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    han: int = 0
    fu: int = 0
    yaku_list: List[Tuple[str, int]] = field(default_factory=list)
    payments: Dict[str, int] = field(default_factory=dict)
    flags: Dict[str, bool] = field(default_factory=dict)
    info: Dict = field(default_factory=dict)


# ============================================================
# 五、引擎配置 / 动作类型 / 阶段机（稳健性关键）
# ------------------------------------------------------------
# 设计目标：
# - 将“规则引擎（环境）”与“策略（AI）”彻底解耦
# - 所有可执行动作统一为 Action，所有合法性由 legal_actions() 控制
# - 用 Phase 明确一步处于哪个阶段，避免把流程写死在一个 step() 里
# - 引擎内部维护事件日志 event_log，便于调试与复现
# ============================================================


GameConfig = RuleProfile


class Phase(str, Enum):
    """对局阶段（状态机）。"""
    DRAW = "DRAW"                 # 当前玩家需要摸牌（或岭上摸牌）
    DISCARD = "DISCARD"           # 当前玩家需要弃牌 / 自摸 / 暗杠 / 立直(后续)
    RESPONSE = "RESPONSE"         # 其他玩家对弃牌进行响应（ron/pon/chi/kan）
    END = "END"                   # 终局


class ActionType(str, Enum):
    """统一动作类型。"""
    DISCARD = "DISCARD"           # 弃牌
    TSUMO = "TSUMO"               # 自摸和
    RON = "RON"                   # 荣和
    PASS = "PASS"                 # 响应阶段放弃
    # 下面为扩展位：鸣牌/杠（目前先提供结构与合法性校验框架）
    CHI = "CHI"
    PON = "PON"
    KAN = "KAN"                   # 明杠/暗杠/加杠统一记为 KAN（具体类型由 info 区分）
    RIICHI = "RIICHI"
    ABORTIVE_DRAW = "ABORTIVE_DRAW"


@dataclass(frozen=True)
class Action:
    """一次动作。tile 为主要牌型参数，info 用于携带额外信息（如顺子组合/杠类型等）。"""
    type: ActionType
    tile: Optional[int] = None
    info: Dict = field(default_factory=dict)


# ============================================================
# 六、引擎主体（单局）
# ============================================================

class RiichiEngine:
    """
    更完善的 MVP 引擎：
    - 仍然只做“摸/打/和/流局”的最小闭环
    - 新增：荣和（ron）与放铳归因
    """
    def __init__(self, seed: int = 0, config: Optional[GameConfig] = None):
        """
        稳健引擎初始化：
        - config：规则开关集合（可扩展）
        - 采用 Phase 状态机驱动流程
        - 采用 Action 统一动作编码，合法性由 legal_actions() 约束
        """
        self.rng = random.Random(seed)
        self.config = config or DEFAULT_RULE_PROFILE

        self.players: List[PlayerState] = [PlayerState(i) for i in range(4)]

        # live_wall：可摸的活牌山；dead_wall：王牌区（含宝牌指示与岭上牌）
        self.live_wall: List[int] = []
        self.dead_wall: List[int] = []
        self.kan_count: int = 0  # 本局杠次数
        self.dora_indicators: List[int] = []  # 宝牌指示牌（牌型索引），按翻出顺序

        self.cur: int = 0
        self.turn: int = 0
        self.done: bool = False
        self.dealer: int = 0

        # 当前阶段机
        self.phase: Phase = Phase.DRAW

        # 响应阶段：待响应的弃牌信息
        self.pending_discard: Optional[Dict] = None

        # 记录最近一次动作信息（方便调试/归因）
        self.last_draw: Optional[int] = None
        self.last_discard: Optional[int] = None
        self.last_discarder: Optional[int] = None

        # 事件日志：用于复现/调试/单元测试
        self.event_log: List[Dict] = []

        # 场风/自风（MVP：默认东场；自风按座位相对庄家变化，可随 dealer 轮转时重算）
        self.round_wind: int = 27  # 东场
        self.seat_winds: List[int] = [27, 28, 29, 30]  # 东南西北
        self.scores: List[int] = [25000, 25000, 25000, 25000]
        self.honba: int = 0
        self.riichi_sticks: int = 0
        self.ippatsu_active: List[bool] = [False, False, False, False]
        self.same_turn_furiten: List[Set[int]] = [set(), set(), set(), set()]
        self.first_discards: List[Optional[int]] = [None, None, None, None]
        self.open_call_happened: bool = False
        self.discard_was_called: List[bool] = [False, False, False, False]

    # -----------------------------
    # 初始化
    # -----------------------------

    def reset(self, dealer: int = 0) -> None:
        """开局：生成牌山、清空状态、配牌13张、设置庄家为当前玩家。"""
        full_wall = make_wall(self.rng)

        self.kan_count = 0
        self.dora_indicators = []
        self.event_log = []

        self.phase = Phase.DRAW
        self.pending_discard = None

        self.cur = dealer
        self.dealer = dealer
        self.turn = 0
        self.done = False
        self.last_draw = None
        self.last_discard = None
        self.last_discarder = None
        self.ippatsu_active = [False, False, False, False]
        self.same_turn_furiten = [set(), set(), set(), set()]
        self.first_discards = [None, None, None, None]
        self.open_call_happened = False
        self.discard_was_called = [False, False, False, False]

        for p in self.players:
            p.hand34 = [0] * 34
            p.river = []
            p.melds = []
            p.riichi_declared = False
            p.riichi_turn = None

        if self.config.use_dead_wall:
            # 末尾切出王牌区（14张），剩余为活牌山
            self.dead_wall = full_wall[-self.config.dead_wall_size:]
            self.live_wall = full_wall[:-self.config.dead_wall_size]
            # 初始翻出第 1 张宝牌指示牌
            self._reveal_dora_indicator(index=0)
        else:
            self.dead_wall = []
            self.live_wall = full_wall

        # 配牌：每人 13 张
        for _ in range(13):
            for i in range(4):
                t = self.live_wall.pop()
                hand34_add(self.players[i].hand34, t)

        # 自风跟座位相对庄家的位置变化：庄=东，其下家=南，对家=西，上家=北
        self.seat_winds = [0, 0, 0, 0]
        for offset in range(4):
            seat = (dealer + offset) % 4
            self.seat_winds[seat] = 27 + offset

        self.phase = Phase.DRAW
        self._log_event({"type": "RESET", "dealer": dealer, "round_wind": self.round_wind, "seat_winds": list(self.seat_winds)})

    def _open_shape_melds(self, seat: int) -> List[Tuple[str, List[int]]]:
        """Return exposed melds as shape melds (kan counted as one meld)."""
        out: List[Tuple[str, List[int]]] = []
        for mtype, tiles in self.players[seat].melds:
            if mtype == "chi":
                out.append(("seq", list(tiles[:3])))
            elif mtype == "pon":
                out.append(("trip", [tiles[0], tiles[0], tiles[0]]))
            elif mtype in ("minkan", "kakan", "ankan"):
                out.append(("kan", [tiles[0], tiles[0], tiles[0], tiles[0]]))
        return out

    def _gen_concealed_decompositions(
        self,
        hand34: List[int],
        target_melds: int,
    ) -> List[Tuple[int, List[Tuple[str, List[int]]]]]:
        """Enumerate decompositions of concealed tiles into target_melds + one pair."""
        if sum(hand34) != (target_melds * 3 + 2):
            return []
        results: List[Tuple[int, List[Tuple[str, List[int]]]]] = []

        def backtrack(counts: List[int], start: int, melds: List[Tuple[str, List[int]]], pair_tile: int) -> None:
            if len(melds) > target_melds:
                return
            i = start
            while i < 34 and counts[i] == 0:
                i += 1
            if i == 34:
                if len(melds) == target_melds:
                    results.append((pair_tile, [m for m in melds]))
                return

            if counts[i] >= 3:
                counts[i] -= 3
                melds.append(("trip", [i, i, i]))
                backtrack(counts, i, melds, pair_tile)
                melds.pop()
                counts[i] += 3

            if 0 <= i <= 26:
                suit = i // 9
                pos = i % 9
                if pos <= 6:
                    a, b, c = i, i + 1, i + 2
                    if (b // 9) == suit and (c // 9) == suit and counts[b] > 0 and counts[c] > 0:
                        counts[a] -= 1
                        counts[b] -= 1
                        counts[c] -= 1
                        melds.append(("seq", [a, b, c]))
                        backtrack(counts, i, melds, pair_tile)
                        melds.pop()
                        counts[a] += 1
                        counts[b] += 1
                        counts[c] += 1

        for t in range(34):
            if hand34[t] >= 2:
                counts = copy_hand(hand34)
                counts[t] -= 2
                backtrack(counts, 0, [], t)

        return results

    def _is_valuable_pair(self, tile: int, seat_wind: int, round_wind: int) -> int:
        fu = 0
        if tile in DRAGONS:
            fu += 2
        if tile == seat_wind:
            fu += 2
        if tile == round_wind:
            fu += 2
        return fu

    def _trip_kan_fu(self, tile: int, *, is_open: bool, is_kan: bool) -> int:
        is_yaochu = _is_terminal_or_honor(tile)
        if is_kan:
            if is_open:
                return 16 if is_yaochu else 8
            return 32 if is_yaochu else 16
        if is_open:
            return 4 if is_yaochu else 2
        return 8 if is_yaochu else 4

    def _wait_fu_for_seq(self, seq: List[int], win_tile: int) -> int:
        if win_tile not in seq:
            return 0
        a, b, c = seq
        if win_tile == b:
            return 2  # kanchan
        start = a % 9
        if win_tile == a and start == 6:
            return 2  # penchan 7 on 789
        if win_tile == c and start == 0:
            return 2  # penchan 3 on 123
        return 0  # ryanmen

    def _calculate_fu(
        self,
        *,
        winner: int,
        hand34: List[int],
        win_type: str,
        win_tile: Optional[int],
        seat_wind: int,
        round_wind: int,
    ) -> int:
        if is_kokushi(hand34):
            return 0
        if is_chiitoi(hand34):
            return 25

        open_shape = self._open_shape_melds(winner)
        target_melds = 4 - len(open_shape)
        if target_melds < 0:
            return 30
        decomps = self._gen_concealed_decompositions(hand34, target_melds=target_melds)
        if not decomps:
            return 30

        is_closed = self._is_closed_hand(winner)
        best_fu = 20

        open_meld_fu = 0
        for mtype, tiles in self.players[winner].melds:
            if mtype == "pon":
                open_meld_fu += self._trip_kan_fu(tiles[0], is_open=True, is_kan=False)
            elif mtype in ("minkan", "kakan"):
                open_meld_fu += self._trip_kan_fu(tiles[0], is_open=True, is_kan=True)
            elif mtype == "ankan":
                open_meld_fu += self._trip_kan_fu(tiles[0], is_open=False, is_kan=True)

        for pair_tile, concealed_melds in decomps:
            pair_fu = self._is_valuable_pair(pair_tile, seat_wind=seat_wind, round_wind=round_wind)
            base_without_wait = 20
            if win_type == "ron" and is_closed:
                base_without_wait += 10  # menzen ron
            base_without_wait += open_meld_fu

            # concealed triplet fu (ron on shanpon turns that triplet open)
            seq_melds: List[List[int]] = []
            trip_melds: List[List[int]] = []
            for mtype, tiles in concealed_melds:
                if mtype == "seq":
                    seq_melds.append(tiles)
                else:
                    trip_melds.append(tiles)

            # Candidate winning component assignments (for ambiguous decompositions).
            wait_candidates: List[Tuple[str, int]] = []
            if win_tile is not None and pair_tile == win_tile:
                wait_candidates.append(("pair", -1))
            if win_tile is not None:
                for i, seq in enumerate(seq_melds):
                    if win_tile in seq:
                        wait_candidates.append(("seq", i))
                for i, trip in enumerate(trip_melds):
                    if win_tile in trip:
                        wait_candidates.append(("trip", i))
            if not wait_candidates:
                wait_candidates.append(("none", -1))

            for kind, idx in wait_candidates:
                meld_fu = 0
                for i, trip in enumerate(trip_melds):
                    ron_trip_open = (win_type == "ron" and kind == "trip" and idx == i)
                    meld_fu += self._trip_kan_fu(trip[0], is_open=ron_trip_open, is_kan=False)

                wait_fu = 0
                if kind == "pair":
                    wait_fu += 2  # tanki
                elif kind == "seq":
                    wait_fu += self._wait_fu_for_seq(seq_melds[idx], win_tile if win_tile is not None else -1)
                # shanpon -> 0

                fu = base_without_wait + pair_fu + meld_fu + wait_fu
                if win_type == "tsumo":
                    fu += 2
                    # pinfu tsumo special: fixed 20-fu
                    if (
                        is_closed
                        and open_meld_fu == 0
                        and pair_fu == 0
                        and meld_fu == 0
                        and wait_fu == 0
                    ):
                        fu = 20

                if fu != 25:
                    if win_type == "ron" and fu == 20:
                        fu = 30  # open pinfu ron minimum
                    fu = ((fu + 9) // 10) * 10
                best_fu = max(best_fu, fu)

        return best_fu

    def _yaku_info_for_win(
        self,
        winner: int,
        win_type: str,
        winning_hand34: List[int],
        win_tile: Optional[int],
    ) -> Dict:
        """Analyze yaku and derive scoring fields for StepResult."""
        yakus, total_han = analyze_yaku(
            hand34=winning_hand34,
            win_type=win_type,
            seat_wind=self.seat_winds[winner],
            round_wind=self.round_wind,
            is_closed=self._is_closed_hand(winner),
        )
        dora_han = count_dora(winning_hand34, self.dora_indicators)
        if dora_han > 0:
            yakus.append(("宝牌", dora_han))
        total_han += dora_han
        fu = self._calculate_fu(
            winner=winner,
            hand34=winning_hand34,
            win_type=win_type,
            win_tile=win_tile,
            seat_wind=self.seat_winds[winner],
            round_wind=self.round_wind,
        )
        return {
            "yaku": yakus,
            "han": total_han,
            "fu": fu,
            "seat_wind": self.seat_winds[winner],
            "round_wind": self.round_wind,
        }


    def _log_event(self, evt: Dict) -> None:
        """写入事件日志（用于复现/调试/单元测试）。"""
        self.event_log.append(evt)

    def _reveal_dora_indicator(self, index: int) -> None:
        """翻出第 index 张宝牌指示牌（按杠次数逐步翻）。"""
        if not self.config.use_dead_wall:
            return
        # 约定指示牌在 dead_wall 的固定位置：从末端向前第 5、7、9、11 张
        positions = [5, 7, 9, 11]
        if index < 0 or index >= len(positions):
            return
        pos_from_end = positions[index]
        if pos_from_end > len(self.dead_wall):
            return
        indicator = self.dead_wall[-pos_from_end]
        if len(self.dora_indicators) <= index:
            self.dora_indicators.append(indicator)
            self._log_event({"type": "DORA_REVEAL", "index": index, "indicator": indicator})

    def _draw_from_rinshan(self) -> int:
        """岭上摸牌：从王牌区的岭上牌摸（MVP：用 dead_wall 的末端依次取）。"""
        if not self.config.use_dead_wall or len(self.dead_wall) == 0:
            raise RuntimeError("没有王牌区，无法岭上摸牌")
        t = self.dead_wall.pop()
        hand34_add(self.players[self.cur].hand34, t)
        self.last_draw = t
        self._log_event({"type": "DRAW_RINSHAN", "player": self.cur, "tile": t})
        return t

    def _is_closed_hand(self, seat: int) -> bool:
        # ankan keeps hand closed for yaku purposes.
        return all(m[0] in ("ankan",) for m in self.players[seat].melds)

    def _riichi_discard_candidates(self, seat: int) -> List[int]:
        p = self.players[seat]
        if p.riichi_declared or not self._is_closed_hand(seat):
            return []
        if self.scores[seat] < 1000:
            return []
        cands: List[int] = []
        for t in range(34):
            if p.hand34[t] <= 0:
                continue
            tmp = copy_hand(p.hand34)
            tmp[t] -= 1
            if is_tenpai(tmp):
                cands.append(t)
        return cands

    def _is_furiten(self, seat: int, tile: int) -> bool:
        if not self.config.enforce_furiten:
            return False
        permanent = tile in self.players[seat].river
        same_turn = self.config.enforce_same_turn_furiten and tile in self.same_turn_furiten[seat]
        riichi_furiten = self.config.enforce_riichi_furiten and self.players[seat].riichi_declared and same_turn
        return permanent or same_turn or riichi_furiten

    def _cancel_ippatsu(self) -> None:
        self.ippatsu_active = [False, False, False, False]

    def _should_abort_suucha_riichi(self) -> bool:
        return self.config.enable_suucha_riichi and all(p.riichi_declared for p in self.players)

    def _should_abort_suufon_renda(self) -> bool:
        if not self.config.enable_suufon_renda or self.open_call_happened:
            return False
        if any(t is None for t in self.first_discards):
            return False
        first = self.first_discards[0]
        if first is None or not (27 <= first <= 30):
            return False
        return all(t == first for t in self.first_discards)

    def _should_abort_suukan_sanra(self) -> bool:
        return self.config.enable_suukan_sanra and self.kan_count >= 4

    def _finalize_abortive_draw(self, reason: str, info: Optional[Dict] = None) -> StepResult:
        self.done = True
        self.phase = Phase.END
        payload = {"turn": self.turn}
        if info:
            payload.update(info)
        self._log_event({"type": "END", "reason": reason.upper(), **payload})
        return StepResult(done=True, reason=reason, info=payload)

    def _calc_noten_bappu_delta(self) -> Tuple[List[int], List[int]]:
        tenpai = [i for i in range(4) if is_tenpai(self.players[i].hand34)]
        noten = [i for i in range(4) if i not in tenpai]
        delta = [0, 0, 0, 0]
        if len(tenpai) in (0, 4):
            return delta, tenpai

        total = 3000
        gain = total // len(tenpai)
        loss = total // len(noten)
        for t in tenpai:
            delta[t] += gain
        for n in noten:
            delta[n] -= loss
        return delta, tenpai

    def _resolve_exhaustive_draw(self) -> StepResult:
        # Nagashi mangan priority over regular exhaustive draw payments.
        if self.config.enable_nagashi_mangan:
            qualifiers = []
            for seat in range(4):
                if self.discard_was_called[seat]:
                    continue
                river = self.players[seat].river
                if not river:
                    continue
                if all(_is_terminal_or_honor(t) for t in river):
                    qualifiers.append(seat)
            if qualifiers:
                winner = qualifiers[0]
                pr = resolve_tsumo(
                    winner=winner,
                    han=5,
                    fu=30,
                    dealer=self.dealer,
                    honba=self.honba,
                    riichi_sticks=self.riichi_sticks,
                    kazoe_yakuman=self.config.kazoe_yakuman,
                    kiriage_mangan=self.config.kiriage_mangan,
                )
                for i in range(4):
                    self.scores[i] += pr.score_delta[i]
                self.done = True
                self.phase = Phase.END
                self._log_event({"type": "END", "reason": "NAGASHI_MANGAN", "winner": winner, "turn": self.turn})
                return StepResult(
                    done=True,
                    reason="nagashi_mangan",
                    winner=winner,
                    winners=[winner],
                    score_delta=list(pr.score_delta),
                    han=5,
                    fu=30,
                    yaku_list=[("流局满贯", 5)],
                    payments=dict(pr.payments),
                    info={"turn": self.turn, "winner": winner},
                )

        delta, tenpai = self._calc_noten_bappu_delta()
        for i in range(4):
            self.scores[i] += delta[i]
        self.done = True
        self.phase = Phase.END
        self._log_event({"type": "END", "reason": "RYUUKYOKU", "turn": self.turn, "tenpai": tenpai})
        return StepResult(
            done=True,
            reason="ryuukyoku",
            score_delta=delta,
            payments={"noten_bappu_total": 3000, "tenpai_count": len(tenpai)},
            info={"turn": self.turn, "tenpai": tenpai},
        )

    # -----------------------------
    # 基础动作：摸 / 弃
    # -----------------------------

    def draw(self) -> int:
        """当前玩家从活牌山摸一张牌。"""
        if self.done or self.phase == Phase.END:
            raise RuntimeError("对局已结束，不能摸牌。")
        if self.phase != Phase.DRAW:
            raise RuntimeError(f"当前阶段为 {self.phase}，不能摸牌")
        if len(self.live_wall) == 0:
            raise RuntimeError("活牌山已空，不能摸牌。")

        t = self.live_wall.pop()
        hand34_add(self.players[self.cur].hand34, t)
        self.same_turn_furiten[self.cur].clear()
        self.last_draw = t
        self._log_event({"type": "DRAW", "player": self.cur, "tile": t, "live_wall": len(self.live_wall)})

        # 摸牌后进入弃牌阶段
        self.phase = Phase.DISCARD
        return t

    def legal_discard_mask(self) -> List[int]:
        """返回 34维合法弃牌 mask（手里有则1，否则0）。"""
        hand = self.players[self.cur].hand34
        return [1 if hand[t] > 0 else 0 for t in range(34)]

    def legal_discards(self) -> List[int]:
        """返回当前玩家所有合法弃牌列表。"""
        hand = self.players[self.cur].hand34
        return [t for t in range(34) if hand[t] > 0]

    def _chi_options(self, tile: int) -> List[Tuple[int, int]]:
        """返回对 tile 可行的吃牌两张搭配（返回 (a,b) 表示用 a 和 b 吃进 tile）。"""
        if tile is None:
            return []
        if tile < 0 or tile > 26:
            return []  # 字牌不能吃
        suit = tile // 9
        pos = tile % 9
        opts: List[Tuple[int, int]] = []
        # (tile-2, tile-1, tile)
        if pos >= 2:
            a, b = tile - 2, tile - 1
            if a // 9 == suit and b // 9 == suit:
                opts.append((a, b))
        # (tile-1, tile, tile+1)
        if 1 <= pos <= 7:
            a, b = tile - 1, tile + 1
            if a // 9 == suit and b // 9 == suit:
                opts.append((a, b))
        # (tile, tile+1, tile+2)
        if pos <= 6:
            a, b = tile + 1, tile + 2
            if a // 9 == suit and b // 9 == suit:
                opts.append((a, b))
        return opts

    def _apply_pon(self, actor: int, tile: int, discarder: int) -> None:
        """执行碰：actor 用两张 tile + 弃牌 tile 组成明刻，随后由 actor 进入 DISCARD。"""
        self._mark_discard_called(discarder=discarder, tile=tile)
        # 从 actor 手牌移除两张
        hand34_remove(self.players[actor].hand34, tile)
        hand34_remove(self.players[actor].hand34, tile)
        # 记录副露
        self.players[actor].melds.append(("pon", [tile, tile, tile]))
        self._cancel_ippatsu()
        self._log_event({"type": "PON", "player": actor, "from": discarder, "tile": tile})
        # 清空 pending，切到 actor 弃牌
        self.pending_discard = None
        self.cur = actor
        self.phase = Phase.DISCARD

    def _apply_chi(self, actor: int, tile: int, use_a: int, use_b: int, discarder: int) -> None:
        """执行吃：actor 用 use_a/use_b + 弃牌 tile 组成顺子，随后由 actor 进入 DISCARD。"""
        self._mark_discard_called(discarder=discarder, tile=tile)
        hand34_remove(self.players[actor].hand34, use_a)
        hand34_remove(self.players[actor].hand34, use_b)
        seq = sorted([use_a, tile, use_b])
        self.players[actor].melds.append(("chi", seq))
        self._cancel_ippatsu()
        self._log_event({"type": "CHI", "player": actor, "from": discarder, "tile": tile, "use": [use_a, use_b]})
        self.pending_discard = None
        self.cur = actor
        self.phase = Phase.DISCARD

    def _mark_discard_called(self, discarder: int, tile: int) -> None:
        if self.players[discarder].river and self.players[discarder].river[-1] == tile:
            self.players[discarder].river.pop()
            self.discard_was_called[discarder] = True
            self.open_call_happened = True

    def _find_pon_meld_idx(self, seat: int, tile: int) -> Optional[int]:
        for i, m in enumerate(self.players[seat].melds):
            if m[0] == "pon" and len(m[1]) == 3 and all(x == tile for x in m[1]):
                return i
        return None

    def _collect_chankan_ronners(self, kan_actor: int, tile: int) -> List[int]:
        winners: List[int] = []
        for offset in (1, 2, 3):
            p = (kan_actor + offset) % 4
            tmp = copy_hand(self.players[p].hand34)
            tmp[tile] += 1
            if is_agari(tmp) and not self._is_furiten(p, tile):
                winners.append(p)
        return winners

    def _apply_minkan(self, actor: int, tile: int, discarder: int) -> StepResult:
        self._mark_discard_called(discarder=discarder, tile=tile)
        for _ in range(3):
            hand34_remove(self.players[actor].hand34, tile)
        self.players[actor].melds.append(("minkan", [tile, tile, tile, tile]))
        self.kan_count += 1
        self._cancel_ippatsu()
        self._log_event({"type": "KAN", "player": actor, "tile": tile, "kan_type": "MINKAN", "kan_count": self.kan_count})
        if self._should_abort_suukan_sanra():
            return self._finalize_abortive_draw("suukan_sanra", {"kan_count": self.kan_count})
        self._reveal_dora_indicator(index=min(self.kan_count, 3))
        self.pending_discard = None
        self.cur = actor
        self.phase = Phase.DISCARD
        self._draw_from_rinshan()
        self.phase = Phase.DISCARD
        return StepResult(done=False, reason="continue", info={"turn": self.turn, "claim": {"type": "minkan", "player": actor, "tile": tile}})

    def legal_actions(self) -> List[Action]:
        """返回当前阶段下的所有合法动作（稳健性核心：策略只能从这里选）。"""
        if self.done or self.phase == Phase.END:
            return []

        actions: List[Action] = []

        if self.phase == Phase.DISCARD:
            # 允许声明自摸（若已和牌）
            if is_agari(self.players[self.cur].hand34):
                actions.append(Action(ActionType.TSUMO))

            # 九种九牌（第一巡自家第一摸牌后）
            if (
                self.config.enable_kyuushu_kyuuhai
                and len(self.players[self.cur].river) == 0
                and all(len(p.river) == 0 for p in self.players)
                and not self.open_call_happened
                and count_yaochu_types(self.players[self.cur].hand34) >= 9
            ):
                actions.append(Action(ActionType.ABORTIVE_DRAW))

            # 合法弃牌
            for t in self.legal_discards():
                actions.append(Action(ActionType.DISCARD, tile=t))

            # 立直：用 ActionType.RIICHI + tile 表示“宣言并切该牌”
            for t in self._riichi_discard_candidates(self.cur):
                actions.append(Action(ActionType.RIICHI, tile=t))

            # 暗杠：四张同牌（仅提供最小闭环）
            if self.config.enable_kan and self.kan_count < self.config.max_kan_per_hand:
                hand = self.players[self.cur].hand34
                for t in range(34):
                    if hand[t] == 4:
                        actions.append(Action(ActionType.KAN, tile=t, info={"kan_type": "ANKAN"}))
                for t in range(34):
                    if hand[t] <= 0:
                        continue
                    if self._find_pon_meld_idx(self.cur, t) is not None:
                        actions.append(Action(ActionType.KAN, tile=t, info={"kan_type": "KAKAN"}))

            return actions

        if self.phase == Phase.RESPONSE:
            if not self.pending_discard:
                return [Action(ActionType.PASS)]

            tile = self.pending_discard["tile"]
            discarder = self.pending_discard["player"]
            actor = self.pending_discard["actor"]

            # PASS 总是合法
            actions.append(Action(ActionType.PASS))

            # RON：手牌 + 该弃牌能和（任何响应者都可）
            tmp = copy_hand(self.players[actor].hand34)
            tmp[tile] += 1
            if is_agari(tmp) and not self._is_furiten(actor, tile):
                actions.append(Action(ActionType.RON, tile=tile, info={"from": discarder}))

            # 若已有人声明了 pon/chi，则后续响应者只允许 RON/PASS（等待是否被截胡）
            if self.pending_discard.get("claim_made", False):
                return actions

            # PON：除弃牌者外，任一响应者手里有两张同牌即可碰
            if (not self.players[actor].riichi_declared) and actor != discarder and self.players[actor].hand34[tile] >= 2:
                actions.append(Action(ActionType.PON, tile=tile, info={"from": discarder}))

            # MINKAN：响应弃牌时，手里有三张同牌可明杠
            if (
                self.config.enable_kan
                and self.kan_count < self.config.max_kan_per_hand
                and (not self.players[actor].riichi_declared)
                and actor != discarder
                and self.players[actor].hand34[tile] >= 3
            ):
                actions.append(Action(ActionType.KAN, tile=tile, info={"from": discarder, "kan_type": "MINKAN"}))

            # CHI：只能下家吃，且只能吃数牌，且必须存在可行顺子组合
            if (not self.players[actor].riichi_declared) and actor == (discarder + 1) % 4 and 0 <= tile <= 26:
                for a, b in self._chi_options(tile):
                    if self.players[actor].hand34[a] > 0 and self.players[actor].hand34[b] > 0:
                        actions.append(Action(ActionType.CHI, tile=tile, info={"from": discarder, "use": [a, b]}))

            return actions

        # DRAW 阶段通常由引擎自动摸牌，这里不暴露动作
        return actions

    def discard(self, tile: int) -> None:
        """当前玩家弃牌，并进入响应阶段。"""
        if self.done or self.phase == Phase.END:
            raise RuntimeError("对局已结束，不能弃牌。")
        if self.phase != Phase.DISCARD:
            raise RuntimeError(f"当前阶段为 {self.phase}，不能弃牌")

        if tile < 0 or tile >= 34:
            raise ValueError("tile 必须在 0..33")

        hand = self.players[self.cur].hand34
        if hand[tile] <= 0:
            raise ValueError(f"非法弃牌：手里没有 {tile_to_str(tile)}")

        hand34_remove(hand, tile)
        self.players[self.cur].river.append(tile)
        if len(self.players[self.cur].river) == 1:
            self.first_discards[self.cur] = tile

        self.last_discard = tile
        self.last_discarder = self.cur

        self.turn += 1
        self.last_draw = None

        self._log_event({"type": "DISCARD", "player": self.cur, "tile": tile, "turn": self.turn})

        # 进入响应阶段：从下家开始依次响应
        discarder = self.cur
        responders = [(discarder + 1) % 4, (discarder + 2) % 4, (discarder + 3) % 4]
        self.pending_discard = {
            "player": discarder,
            "tile": tile,
            "responders": responders,
            "actor": responders[0],
            "passes": set(),
            "ronners": [],
            # 鸣牌声明（用于优先级裁决）：
            # - pon_claim: (player, Action) 可能有多个，按顺时针距离取最优
            # - minkan_claim: (player, Action) 明杠声明
            # - chi_claim: (player, Action) 只可能是下家
            "minkan_claims": [],
            "pon_claims": [],
            "chi_claim": None,
            # 一旦有人声明 pon/chi，则后续响应者只允许 RON/PASS（模拟“声明后等待是否被ron截胡”）
            "claim_made": False,
        }
        self.phase = Phase.RESPONSE

        # 当前行动者切换到第一个响应者
        self.cur = responders[0]

    # -----------------------------
    # 新增：荣和判定与放铳归因
    # -----------------------------

    def _check_ron_after_discard(self, discarded_tile: int, discarder: int) -> List[int]:
        """
        在 discarder 打出 discarded_tile 后，检查其他三家是否能“荣和”。

        判定方法（MVP）：
        - 对每个其他玩家 p：
          临时构造 hand_p + discarded_tile（即 hand34[t] += 1）
          若结构上 is_agari(14张) 则认为“可以荣和”

        返回：所有可以荣和的玩家 seat 列表（按顺时针顺序）
        """
        ronners: List[int] = []

        # 日麻通常从“放铳者下家开始”顺时针响应（头跳等细则另说）
        for offset in (1, 2, 3):
            p = (discarder + offset) % 4
            tmp = copy_hand(self.players[p].hand34)
            tmp[discarded_tile] += 1  # 临时吃进这张弃牌作为荣和牌
            if is_agari(tmp):
                ronners.append(p)

        return ronners

    # -----------------------------
    # RL 友好 step：摸 -> 自摸判定 -> 弃牌 -> 荣和判定 -> 轮转
    # -----------------------------

    def apply_action(self, action: Action) -> StepResult:
        """执行一个动作，并在必要时推进阶段机。"""
        if self.done or self.phase == Phase.END:
            return StepResult(done=True, reason="already_done", info={"turn": self.turn})

        # DISCARD 阶段：弃牌 / 自摸 / 暗杠
        if self.phase == Phase.DISCARD:
            if action.type == ActionType.TSUMO:
                if not is_agari(self.players[self.cur].hand34):
                    raise ValueError("当前手牌不满足和牌，不能自摸")
                self.done = True
                self.phase = Phase.END
                hand = copy_hand(self.players[self.cur].hand34)
                yaku_info = self._yaku_info_for_win(self.cur, "tsumo", hand, self.last_draw)
                pr = resolve_tsumo(
                    winner=self.cur,
                    han=yaku_info["han"],
                    fu=yaku_info["fu"],
                    dealer=self.dealer,
                    honba=self.honba,
                    riichi_sticks=self.riichi_sticks,
                    kazoe_yakuman=self.config.kazoe_yakuman,
                    kiriage_mangan=self.config.kiriage_mangan,
                )
                for i in range(4):
                    self.scores[i] += pr.score_delta[i]
                info = {
                    "turn": self.turn,
                    "win_type": "tsumo",
                    "winner": self.cur,
                    "winner_hand34": hand,
                    "dora_indicators": list(self.dora_indicators),
                    "score_delta": list(pr.score_delta),
                    "payments": dict(pr.payments),
                    "level": pr.level,
                }
                info.update(yaku_info)
                self._log_event({"type": "END", "reason": "TSUMO", "winner": self.cur})
                flags = {
                    "riichi": self.players[self.cur].riichi_declared,
                    "ippatsu": self.ippatsu_active[self.cur],
                    "rinshan": bool(self.last_draw is not None and self.event_log and self.event_log[-1].get("type") == "DRAW_RINSHAN"),
                    "haitei": len(self.live_wall) == 0,
                    "houtei": False,
                    "chankan": False,
                }
                return StepResult(
                    done=True,
                    reason="tsumo",
                    winner=self.cur,
                    winners=[self.cur],
                    score_delta=list(pr.score_delta),
                    han=int(yaku_info["han"]),
                    fu=int(yaku_info["fu"]),
                    yaku_list=list(yaku_info["yaku"]),
                    payments=dict(pr.payments),
                    flags=flags,
                    info=info,
                )

            if action.type == ActionType.KAN:
                if not self.config.enable_kan:
                    raise ValueError("规则未启用杠")
                if self.kan_count >= self.config.max_kan_per_hand:
                    raise ValueError("已达到本局最大杠次数")
                t = action.tile
                if t is None:
                    raise ValueError("KAN 需要指定 tile")
                kan_type = str((action.info or {}).get("kan_type", "ANKAN")).upper()

                if kan_type == "KAKAN":
                    meld_idx = self._find_pon_meld_idx(self.cur, t)
                    if meld_idx is None or self.players[self.cur].hand34[t] <= 0:
                        raise ValueError("加杠需要已有碰子且手里有第四张")
                    chankan_winners = self._collect_chankan_ronners(self.cur, t)
                    if chankan_winners:
                        return self._finalize_ron(chankan_winners, discarder=self.cur, tile=t, chankan=True)
                    hand34_remove(self.players[self.cur].hand34, t)
                    self.players[self.cur].melds[meld_idx] = ("kakan", [t, t, t, t])
                    self.kan_count += 1
                    self._cancel_ippatsu()
                    self._log_event({"type": "KAN", "player": self.cur, "tile": t, "kan_type": "KAKAN", "kan_count": self.kan_count})
                    if self._should_abort_suukan_sanra():
                        return self._finalize_abortive_draw("suukan_sanra", {"kan_count": self.kan_count})
                    self._reveal_dora_indicator(index=min(self.kan_count, 3))
                    self._draw_from_rinshan()
                    self.phase = Phase.DISCARD
                    return StepResult(done=False, reason="continue", info={"turn": self.turn, "kan": {"player": self.cur, "tile": t, "type": "KAKAN"}})

                # default ANKAN in DISCARD phase
                if self.players[self.cur].hand34[t] != 4:
                    raise ValueError("暗杠需要手里恰好四张同牌")
                for _ in range(4):
                    hand34_remove(self.players[self.cur].hand34, t)
                self.players[self.cur].melds.append(("ankan", [t, t, t, t]))
                self.kan_count += 1
                self._cancel_ippatsu()
                self._log_event({"type": "KAN", "player": self.cur, "tile": t, "kan_type": "ANKAN", "kan_count": self.kan_count})
                if self._should_abort_suukan_sanra():
                    return self._finalize_abortive_draw("suukan_sanra", {"kan_count": self.kan_count})
                self._reveal_dora_indicator(index=min(self.kan_count, 3))
                self._draw_from_rinshan()
                self.phase = Phase.DISCARD
                return StepResult(done=False, reason="continue", info={"turn": self.turn, "kan": {"player": self.cur, "tile": t, "type": "ANKAN"}})

            if action.type == ActionType.ABORTIVE_DRAW:
                if action not in self.legal_actions():
                    raise ValueError("非法途中流局动作")
                return self._finalize_abortive_draw("kyuushu_kyuuhai", {"player": self.cur})

            if action.type == ActionType.RIICHI:
                t = action.tile
                if t is None:
                    raise ValueError("RIICHI 需要指定弃牌 tile")
                if action not in self.legal_actions():
                    raise ValueError("非法立直动作")
                p = self.players[self.cur]
                p.riichi_declared = True
                p.riichi_turn = self.turn
                self.ippatsu_active[self.cur] = True
                self.riichi_sticks += 1
                self.scores[self.cur] -= 1000
                self._log_event({"type": "RIICHI", "player": self.cur, "tile": t, "turn": self.turn})
                self.discard(t)
                return StepResult(done=False, reason="continue", info={"turn": self.turn, "riichi": {"player": self.cur, "tile": t}})

            if action.type == ActionType.DISCARD:
                if action.tile is None:
                    raise ValueError("DISCARD 需要 tile")
                # 校验是否是合法动作
                if action not in self.legal_actions():
                    raise ValueError("非法弃牌动作")
                self.discard(action.tile)
                return StepResult(done=False, reason="continue", info={"turn": self.turn})

            raise ValueError(f"当前阶段 {self.phase} 不支持动作 {action.type}")

        # RESPONSE 阶段：处理 PASS / RON / PON / CHI / MINKAN
        if self.phase == Phase.RESPONSE:
            if not self.pending_discard:
                raise RuntimeError("RESPONSE 阶段但没有 pending_discard")

            actor = self.pending_discard["actor"]
            if self.cur != actor:
                raise ValueError("当前不是该玩家的响应回合")

            # 校验动作在合法集合中
            legal_now = self.legal_actions()
            had_ron_option = any(a.type == ActionType.RON for a in legal_now)
            if action not in legal_now:
                raise ValueError("非法响应动作")

            if action.type == ActionType.RON:
                self.pending_discard["ronners"].append(actor)
                self._log_event({"type": "RON_DECLARE", "winner": actor, "from": self.pending_discard["player"], "tile": self.pending_discard["tile"]})
                # 头跳：不允许一炮多响时，立即结算
                if not self.config.allow_multi_ron:
                    return self._finalize_ron([actor])

            elif action.type == ActionType.PON:
                # 记录碰声明，但先不立刻执行：等待其他家是否 RON 截胡
                self.pending_discard["pon_claims"].append((actor, action))
                self.pending_discard["claim_made"] = True
                self._log_event({"type": "PON_DECLARE", "player": actor, "from": self.pending_discard["player"], "tile": self.pending_discard["tile"]})

            elif action.type == ActionType.KAN:
                self.pending_discard["minkan_claims"].append((actor, action))
                self.pending_discard["claim_made"] = True
                self._log_event({"type": "MINKAN_DECLARE", "player": actor, "from": self.pending_discard["player"], "tile": self.pending_discard["tile"]})

            elif action.type == ActionType.CHI:
                # 记录吃声明（只可能下家），同样等待是否被 RON 截胡
                self.pending_discard["chi_claim"] = (actor, action)
                self.pending_discard["claim_made"] = True
                self._log_event({"type": "CHI_DECLARE", "player": actor, "from": self.pending_discard["player"], "tile": self.pending_discard["tile"], "use": action.info.get("use")})

            elif action.type == ActionType.PASS:
                self.pending_discard["passes"].add(actor)
                if had_ron_option:
                    self.same_turn_furiten[actor].add(self.pending_discard["tile"])
                self._log_event({"type": "PASS", "player": actor})

            else:
                raise ValueError(f"RESPONSE 阶段不支持动作 {action.type}")

            responders = self.pending_discard["responders"]
            idx = responders.index(actor)
            if idx + 1 < len(responders):
                next_actor = responders[idx + 1]
                self.pending_discard["actor"] = next_actor
                self.cur = next_actor
                return StepResult(done=False, reason="continue", info={"turn": self.turn})

            # 所有人响应完：优先级裁决
            discarder = self.pending_discard["player"]
            tile = self.pending_discard["tile"]
            ronners = list(self.pending_discard["ronners"])

            # 1) 有 ron：结算（允许一炮多响则全部）
            if ronners:
                return self._finalize_ron(ronners)

            # 2) 无 ron：若有 minkan/pon 声明，按顺时针距离最近者成立（responders 顺序即距离顺序）
            minkan_claims = self.pending_discard.get("minkan_claims", [])
            pon_claims = self.pending_discard.get("pon_claims", [])
            if minkan_claims or pon_claims:
                chosen = None
                for r in responders:
                    for p, a in minkan_claims:
                        if p == r:
                            chosen = ("minkan", p, a)
                            break
                    if chosen is not None:
                        break
                    for p, a in pon_claims:
                        if p == r:
                            chosen = ("pon", p, a)
                            break
                    if chosen is not None:
                        break
                if chosen is None and minkan_claims:
                    chosen = ("minkan", minkan_claims[0][0], minkan_claims[0][1])
                if chosen is None and pon_claims:
                    chosen = ("pon", pon_claims[0][0], pon_claims[0][1])
                if chosen is None:
                    raise RuntimeError("claim resolution failed")

                ctype, chosen_player, _ = chosen
                if ctype == "minkan":
                    return self._apply_minkan(actor=chosen_player, tile=tile, discarder=discarder)

                self._apply_pon(actor=chosen_player, tile=tile, discarder=discarder)
                return StepResult(done=False, reason="continue", info={"turn": self.turn, "claim": {"type": "pon", "player": chosen_player, "tile": tile}})

            # 3) 无 ron/pon：若有 chi 声明则执行
            chi_claim = self.pending_discard.get("chi_claim", None)
            if chi_claim is not None:
                chi_player, chi_action = chi_claim
                use = chi_action.info.get("use") or []
                if len(use) != 2:
                    raise RuntimeError("CHI 声明缺少 use 信息")
                self._apply_chi(actor=chi_player, tile=tile, use_a=use[0], use_b=use[1], discarder=discarder)
                return StepResult(done=False, reason="continue", info={"turn": self.turn, "claim": {"type": "chi", "player": chi_player, "tile": tile, "use": use}})

            # 4) 无人鸣牌：进入下一家摸牌阶段
            self.pending_discard = None
            next_player = (discarder + 1) % 4
            self.cur = next_player
            self.phase = Phase.DRAW

            if self._should_abort_suufon_renda():
                return self._finalize_abortive_draw("suufon_renda", {"first_discards": list(self.first_discards)})
            if self._should_abort_suucha_riichi():
                return self._finalize_abortive_draw("suucha_riichi")

            # 活牌山空则荒牌流局
            if len(self.live_wall) == 0:
                return self._resolve_exhaustive_draw()

            return StepResult(done=False, reason="continue", info={"turn": self.turn})

        raise ValueError(f"未处理的阶段 {self.phase}")

    def _finalize_ron(
        self,
        winners: List[int],
        *,
        discarder: Optional[int] = None,
        tile: Optional[int] = None,
        chankan: bool = False,
    ) -> StepResult:
        """结算荣和（含役种/符番/点数结算）。"""
        if discarder is None:
            discarder = self.pending_discard["player"] if self.pending_discard else None
        if tile is None:
            tile = self.pending_discard["tile"] if self.pending_discard else None
        if discarder is None or tile is None:
            raise RuntimeError("无法结算荣和：缺少 pending_discard")

        self.done = True
        self.phase = Phase.END
        self.pending_discard = None

        winners_yaku: Dict[int, Dict] = {}
        aggregate_delta = [0, 0, 0, 0]
        payments: Dict[str, int] = {}
        han = 0
        fu = 0
        yaku_list: List[Tuple[str, int]] = []
        for w in winners:
            tmp = copy_hand(self.players[w].hand34)
            tmp[tile] += 1
            winners_yaku[w] = self._yaku_info_for_win(w, "ron", tmp, tile)
            y = winners_yaku[w]
            pr = resolve_ron(
                winner=w,
                loser=discarder,
                han=y["han"],
                fu=y["fu"],
                dealer=self.dealer,
                honba=self.honba,
                riichi_sticks=self.riichi_sticks if w == winners[0] else 0,
                kazoe_yakuman=self.config.kazoe_yakuman,
                kiriage_mangan=self.config.kiriage_mangan,
            )
            for i in range(4):
                aggregate_delta[i] += pr.score_delta[i]
            if w == winners[0]:
                payments = dict(pr.payments)
                han = int(y["han"])
                fu = int(y["fu"])
                yaku_list = list(y["yaku"])

        for i in range(4):
            self.scores[i] += aggregate_delta[i]

        info = {
            "turn": self.turn,
            "win_type": "ron",
            "discarded_tile": tile,
            "discarder": discarder,
            "winners": list(winners),
            "winners_yaku": winners_yaku,
            "dora_indicators": list(self.dora_indicators),
            "score_delta": list(aggregate_delta),
            "chankan": chankan,
        }

        self._log_event({"type": "END", "reason": "RON", "discarder": discarder, "winners": list(winners), "tile": tile})
        flags = {
            "riichi": self.players[winners[0]].riichi_declared,
            "ippatsu": self.ippatsu_active[winners[0]],
            "rinshan": False,
            "haitei": False,
            "houtei": len(self.live_wall) == 0,
            "chankan": chankan,
        }
        return StepResult(
            done=True,
            reason="ron",
            winner=winners[0],
            winners=list(winners),
            loser=discarder,
            score_delta=list(aggregate_delta),
            han=han,
            fu=fu,
            yaku_list=yaku_list,
            payments=payments,
            flags=flags,
            info=info,
        )


    def step(self, discard_tile: int) -> StepResult:
        """兼容旧接口：在 DISCARD 阶段执行一次弃牌动作。推荐新代码使用 apply_action()。"""
        if self.phase == Phase.DRAW:
            self.draw()
        if self.phase != Phase.DISCARD:
            raise RuntimeError(f"step(discard_tile) 只能在 DISCARD 阶段调用，但当前为 {self.phase}")
        return self.apply_action(Action(ActionType.DISCARD, tile=discard_tile))

    # -----------------------------
    # Observation（不完全信息：只给自己手牌 + 公共信息）
    # -----------------------------

    def get_obs(self, seat: Optional[int] = None) -> Dict:
        """
        观测（给 RL 用）：
        - seat=None 默认当前行动玩家
        - 自己手牌（34维计数）
        - 四家河（弃牌序列，公共信息）
        - 当前轮到谁、剩余牌数等
        """
        if seat is None:
            seat = self.cur

        obs = {
            "seat": seat,
            "cur": self.cur,
            "turn": self.turn,
            "live_wall_len": len(self.live_wall),
            "dead_wall_len": len(self.dead_wall),
            "dora_indicators": list(self.dora_indicators),
            "phase": self.phase,
            "hand34": copy_hand(self.players[seat].hand34),
            "rivers": [list(p.river) for p in self.players],
            "melds": [list(p.melds) for p in self.players],
            "scores": list(self.scores),
            "riichi_declared": [p.riichi_declared for p in self.players],
            "riichi_sticks": self.riichi_sticks,
            "honba": self.honba,
            "last_discard": self.last_discard,
            "last_discarder": self.last_discarder,
        }
        return obs

    def validate_invariants(self) -> None:
        """Strong consistency checks for wall/hand accounting and phase sanity."""
        counts = [0] * 34
        for t in self.live_wall:
            counts[t] += 1
        for t in self.dead_wall:
            counts[t] += 1
        for p in self.players:
            for t, c in enumerate(p.hand34):
                counts[t] += c
            for _, tiles in p.melds:
                for t in tiles:
                    counts[t] += 1
            for t in p.river:
                counts[t] += 1
        for t, c in enumerate(counts):
            if c != 4:
                raise AssertionError(f"tile {t} count invariant broken: {c} != 4")
        if self.done and self.phase != Phase.END:
            raise AssertionError("done game must be in END phase")

    def export_replay(self) -> Replay:
        return Replay(events=list(self.event_log))

    # -----------------------------
    # Demo：随机对局自测（策略=随机弃牌）
    # -----------------------------

    def play_random(self, max_steps: int = 20000, verbose: bool = False) -> StepResult:
        """随机对局自测：按阶段机自动推进，动作从 legal_actions() 随机采样。"""
        steps = 0
        while steps < max_steps and not self.done:
            steps += 1

            if self.phase == Phase.DRAW:
                if len(self.live_wall) == 0:
                    return self._resolve_exhaustive_draw()
                t = self.draw()
                if verbose:
                    print(f"\n[玩家{self.cur}] 摸牌：{tile_to_str(t)} | 活牌剩余={len(self.live_wall)}")
                    print(f"手牌：{hand_to_str(self.players[self.cur].hand34)}")

            elif self.phase in (Phase.DISCARD, Phase.RESPONSE):
                acts = self.legal_actions()
                if not acts:
                    raise RuntimeError(f"阶段 {self.phase} 没有合法动作，状态机卡死")

                a = self.rng.choice(acts)
                if verbose:
                    if self.phase == Phase.DISCARD:
                        if a.type == ActionType.DISCARD:
                            print(f"[玩家{self.cur}] 弃牌：{tile_to_str(a.tile)}")
                        elif a.type == ActionType.TSUMO:
                            print(f"[玩家{self.cur}] 宣告自摸")
                        elif a.type == ActionType.KAN:
                            print(f"[玩家{self.cur}] 宣告暗杠：{tile_to_str(a.tile)}")
                    else:
                        if a.type == ActionType.RON:
                            print(f"[玩家{self.cur}] 宣告荣和（对 {tile_to_str(a.tile)}）")
                        else:
                            print(f"[玩家{self.cur}] PASS")

                res = self.apply_action(a)
                if res.done:
                    if verbose:
                        print(f"\n=== 终局: {res.reason} ===")
                        print(res.info)
                    return res

            else:
                break

        return StepResult(done=self.done, reason="max_steps", info={"turn": self.turn, "phase": self.phase, "events": len(self.event_log)})


# ============================================================
# 主程序：运行自测
# ============================================================

if __name__ == "__main__":
    eng = RiichiEngine(seed=42, config=GameConfig(allow_multi_ron=False, use_dead_wall=True, enable_kan=True))
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
