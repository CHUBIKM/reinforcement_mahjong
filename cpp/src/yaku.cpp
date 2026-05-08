#include "mahjong/yaku.hpp"
#include "mahjong/tile_utils.hpp"

#include <algorithm>
#include <array>
#include <map>
#include <set>

namespace mahjong {

namespace {

using YakuList = std::vector<std::pair<std::string, int>>;

bool is_trip_like(const ShapeMeld& meld) {
    return meld.first == "trip" || meld.first == "kan";
}

bool is_sequence(const ShapeMeld& meld) {
    return meld.first == "seq";
}

bool has_sequence(const std::vector<ShapeMeld>& melds) {
    return std::any_of(melds.begin(), melds.end(), is_sequence);
}

bool all_tiles_match(const Hand34& hand34, bool (*pred)(int)) {
    for (int t = 0; t < 34; ++t) {
        if (hand34[t] > 0 && !pred(t)) return false;
    }
    return true;
}

bool is_terminal(int tile) {
    return tile >= 0 && tile < 34 && IS_TERMINAL[tile];
}

bool is_honor_tile(int tile) {
    return tile >= 0 && tile < 34 && IS_HONOR[tile];
}

bool is_green_tile(int tile) {
    return tile == 19 || tile == 20 || tile == 21 ||
           tile == 23 || tile == 25 || tile == 32;
}

int valuable_pair_fu_for_yaku(int tile, int seat_wind, int round_wind) {
    int fu = 0;
    if (IS_DRAGON[tile]) fu += 2;
    if (tile == seat_wind) fu += 2;
    if (tile == round_wind) fu += 2;
    return fu;
}

bool contains_tile(const std::vector<int>& tiles, int tile) {
    return std::find(tiles.begin(), tiles.end(), tile) != tiles.end();
}

bool is_ryanmen_wait(const std::vector<int>& seq, int win_tile) {
    if (win_tile < 0 || !contains_tile(seq, win_tile)) return false;
    int start = seq[0] % 9;
    if (win_tile == seq[1]) return false;              // kanchan
    if (win_tile == seq[0] && start == 6) return false; // penchan 7-8 waiting 9
    if (win_tile == seq[2] && start == 0) return false; // penchan 1-2 waiting 3
    return true;
}

std::vector<ShapeMeld> fixed_shape_melds(const std::vector<Meld>& fixed_melds) {
    std::vector<ShapeMeld> out;
    for (const auto& m : fixed_melds) {
        if (m.type == "chi") {
            auto tiles = m.tiles;
            std::sort(tiles.begin(), tiles.end());
            if (tiles.size() >= 3) {
                out.emplace_back("seq", std::vector<int>{tiles[0], tiles[1], tiles[2]});
            }
        } else if (m.type == "pon") {
            if (!m.tiles.empty()) {
                int t = m.tiles[0];
                out.emplace_back("trip", std::vector<int>{t, t, t});
            }
        } else if (m.type == "minkan" || m.type == "kakan" || m.type == "ankan") {
            if (!m.tiles.empty()) {
                int t = m.tiles[0];
                out.emplace_back("kan", std::vector<int>{t, t, t, t});
            }
        }
    }
    return out;
}

std::vector<Decomposition> decompositions_with_fixed_melds(
    const Hand34& hand34,
    const std::vector<Meld>& fixed_melds) {
    if (fixed_melds.empty()) return gen_standard_decompositions(hand34);

    auto fixed = fixed_shape_melds(fixed_melds);
    int target_melds = 4 - static_cast<int>(fixed.size());
    if (target_melds < 0) return {};

    auto concealed = copy_hand(hand34);
    for (const auto& m : fixed_melds) {
        int remove_count = (m.type == "chi" || m.type == "pon" ||
                            m.type == "minkan" || m.type == "kakan" ||
                            m.type == "ankan") ? 3 : 0;
        for (int i = 0; i < remove_count && i < static_cast<int>(m.tiles.size()); ++i) {
            if (m.tiles[i] < 0 || m.tiles[i] >= 34 || concealed[m.tiles[i]] <= 0) {
                return {};
            }
            concealed[m.tiles[i]] -= 1;
        }
    }

    std::vector<Decomposition> combined;
    for (const auto& [pair_tile, concealed_melds] :
         gen_concealed_decompositions(concealed, target_melds)) {
        auto melds = fixed;
        melds.insert(melds.end(), concealed_melds.begin(), concealed_melds.end());
        combined.emplace_back(pair_tile, melds);
    }
    return combined;
}

int iipeikou_pair_count(const std::vector<ShapeMeld>& melds) {
    std::map<std::pair<int, int>, int> counts;
    for (const auto& [mtype, tiles] : melds) {
        if (mtype != "seq") continue;
        auto s = tile_suit(tiles[0]);
        if (!s.has_value()) continue;
        counts[{s.value(), tiles[0] % 9}] += 1;
    }

    int pairs = 0;
    for (const auto& [_, count] : counts) {
        pairs += count / 2;
    }
    return pairs;
}

bool is_pinfu_shape(int pair_tile, const std::vector<ShapeMeld>& melds,
                    int seat_wind, int round_wind, int win_tile) {
    if (valuable_pair_fu_for_yaku(pair_tile, seat_wind, round_wind) > 0) return false;
    if (!std::all_of(melds.begin(), melds.end(), is_sequence)) return false;
    if (win_tile < 0) return true;
    if (pair_tile == win_tile) return false;
    for (const auto& [_, tiles] : melds) {
        if (is_ryanmen_wait(tiles, win_tile)) return true;
    }
    return false;
}

bool is_sanshoku_doukou(const std::vector<ShapeMeld>& melds) {
    std::array<std::set<int>, 9> seen;
    for (const auto& meld : melds) {
        if (!is_trip_like(meld)) continue;
        int t = meld.second[0];
        auto suit = tile_suit(t);
        if (!suit.has_value()) continue;
        seen[t % 9].insert(suit.value());
    }
    for (const auto& suits : seen) {
        if (suits.size() == 3) return true;
    }
    return false;
}

int kan_count_from_melds(const std::vector<ShapeMeld>& melds) {
    return static_cast<int>(std::count_if(melds.begin(), melds.end(),
                                          [](const ShapeMeld& m) { return m.first == "kan"; }));
}

int dragon_trip_count(const std::vector<ShapeMeld>& melds) {
    int count = 0;
    for (const auto& meld : melds) {
        if (is_trip_like(meld) && IS_DRAGON[meld.second[0]]) count += 1;
    }
    return count;
}

int wind_trip_count(const std::vector<ShapeMeld>& melds) {
    int count = 0;
    for (const auto& meld : melds) {
        if (is_trip_like(meld)) {
            int t = meld.second[0];
            if (t >= 27 && t <= 30) count += 1;
        }
    }
    return count;
}

bool is_chuuren_shape(const Hand34& hand34, int win_tile, int* yakuman_value) {
    auto [suits, has_honor] = count_suits_in_hand(hand34);
    if (has_honor || suits.size() != 1) return false;

    int suit = *suits.begin();
    int base = suit * 9;
    static const std::array<int, 9> required = {3, 1, 1, 1, 1, 1, 1, 1, 3};
    for (int i = 0; i < 9; ++i) {
        if (hand34[base + i] < required[i]) return false;
    }

    *yakuman_value = 13;
    if (win_tile >= base && win_tile < base + 9 &&
        hand34[win_tile] == required[win_tile - base] + 1) {
        *yakuman_value = 26;
    }
    return true;
}

void add_unique_yakuman(YakuList& yakus, const std::string& name, int han) {
    auto it = std::find_if(yakus.begin(), yakus.end(),
                           [&](const auto& y) { return y.first == name; });
    if (it == yakus.end()) yakus.emplace_back(name, han);
}

YakuList collect_yakuman_yaku(const Hand34& hand34,
                              const std::vector<Decomposition>& decomps,
                              const std::string& win_type,
                              bool is_closed,
                              int win_tile) {
    YakuList yakus;

    if (all_tiles_match(hand34, is_honor_tile)) {
        add_unique_yakuman(yakus, "字一色(役满)", 13);
    }
    if (all_tiles_match(hand34, is_terminal)) {
        add_unique_yakuman(yakus, "清老头(役满)", 13);
    }
    if (all_tiles_match(hand34, is_green_tile)) {
        add_unique_yakuman(yakus, "绿一色(役满)", 13);
    }

    int chuuren_han = 0;
    if (is_closed && is_chuuren_shape(hand34, win_tile, &chuuren_han)) {
        add_unique_yakuman(yakus, chuuren_han == 26 ? "纯正九莲宝灯(双倍役满)" : "九莲宝灯(役满)", chuuren_han);
    }

    for (const auto& [pair_tile, melds] : decomps) {
        if (dragon_trip_count(melds) == 3) {
            add_unique_yakuman(yakus, "大三元(役满)", 13);
        }

        int wind_trips = wind_trip_count(melds);
        if (wind_trips == 4) {
            add_unique_yakuman(yakus, "大四喜(双倍役满)", 26);
        } else if (wind_trips == 3 && pair_tile >= 27 && pair_tile <= 30) {
            add_unique_yakuman(yakus, "小四喜(役满)", 13);
        }

        if (is_closed && std::all_of(melds.begin(), melds.end(), is_trip_like)) {
            if (win_tile >= 0 && pair_tile == win_tile) {
                add_unique_yakuman(yakus, "四暗刻单骑(双倍役满)", 26);
            } else if (win_type == "tsumo" || win_tile < 0) {
                add_unique_yakuman(yakus, "四暗刻(役满)", 13);
            }
        }

        if (kan_count_from_melds(melds) == 4) {
            add_unique_yakuman(yakus, "四杠子(役满)", 13);
        }
    }

    return yakus;
}

std::pair<YakuList, int> finish_yakus(YakuList yakus) {
    int total = 0;
    for (const auto& [_, han] : yakus) total += han;
    return {yakus, total};
}

std::pair<YakuList, int> analyze_yaku_impl(
    const std::vector<Meld>& fixed_melds,
    const Hand34& hand34,
    const std::string& win_type,
    int seat_wind,
    int round_wind,
    bool is_closed,
    int win_tile) {

    YakuList yakus;
    YakuList chiitoi_candidate;
    int chiitoi_total = -1;

    if (fixed_melds.empty() && is_kokushi(hand34)) {
        bool thirteen_wait = win_tile >= 0 && hand34[win_tile] == 2;
        yakus.emplace_back(thirteen_wait ? "国士无双十三面(双倍役满)" : "国士无双(役满)",
                           thirteen_wait ? 26 : 13);
        return finish_yakus(yakus);
    }

    if (fixed_melds.empty() && is_chiitoi(hand34)) {
        auto special_yakuman = collect_yakuman_yaku(hand34, {}, win_type, is_closed, win_tile);
        if (!special_yakuman.empty()) return finish_yakus(special_yakuman);

        chiitoi_candidate.emplace_back("七对子", 2);
        if (is_tanyao(hand34)) {
            chiitoi_candidate.emplace_back("断幺九", 1);
        }
        if (all_tiles_match(hand34, is_terminal_or_honor)) {
            chiitoi_candidate.emplace_back("混老头", 2);
        }

        auto [suits, has_honor] = count_suits_in_hand(hand34);
        if (suits.size() == 1) {
            chiitoi_candidate.emplace_back(has_honor ? "混一色" : "清一色", has_honor ? 3 : 6);
        }
        chiitoi_total = 0;
        for (const auto& [_, han] : chiitoi_candidate) chiitoi_total += han;
    }

    auto decomps = decompositions_with_fixed_melds(hand34, fixed_melds);
    if (decomps.empty()) {
        if (chiitoi_total >= 0) return {chiitoi_candidate, chiitoi_total};
        return {yakus, 0};
    }

    auto yakuman = collect_yakuman_yaku(hand34, decomps, win_type, is_closed, win_tile);
    if (!yakuman.empty()) return finish_yakus(yakuman);

    if (win_type == "tsumo" && is_closed) {
        yakus.emplace_back("门前清自摸和", 1);
    }

    if (is_closed) {
        bool any_pinfu = false;
        for (const auto& [pair_tile, melds] : decomps) {
            if (is_pinfu_shape(pair_tile, melds, seat_wind, round_wind, win_tile)) {
                any_pinfu = true;
                break;
            }
        }
        if (any_pinfu) yakus.emplace_back("平和", 1);
    }

    if (is_tanyao(hand34)) {
        yakus.emplace_back("断幺九", 1);
    }

    auto [suits, has_honor] = count_suits_in_hand(hand34);
    if (suits.size() == 1) {
        if (has_honor) {
            yakus.emplace_back("混一色", is_closed ? 3 : 2);
        } else {
            yakus.emplace_back("清一色", is_closed ? 6 : 5);
        }
    }

    std::set<std::string> yakuhai_names;
    for (const auto& [pair_tile, melds] : decomps) {
        (void)pair_tile;
        for (const auto& name : yakuhai_from_melds(melds, seat_wind, round_wind)) {
            yakuhai_names.insert(name);
        }
    }
    for (const auto& name : yakuhai_names) {
        yakus.emplace_back(name, 1);
    }

    bool any_toitoi = false;
    bool any_sanankou = false;
    bool any_sanshoku_doukou = false;
    bool any_sankantsu = false;
    for (const auto& [_, melds] : decomps) {
        if (is_toitoi(melds)) any_toitoi = true;
        if (is_sanankou(melds)) any_sanankou = true;
        if (is_sanshoku_doukou(melds)) any_sanshoku_doukou = true;
        if (kan_count_from_melds(melds) >= 3) any_sankantsu = true;
    }
    if (any_toitoi) yakus.emplace_back("对对和", 2);
    if (any_sanankou) yakus.emplace_back("三暗刻", 2);
    if (any_sanshoku_doukou) yakus.emplace_back("三色同刻", 2);
    if (any_sankantsu) yakus.emplace_back("三杠子", 2);

    if (is_closed) {
        bool any_ryanpeikou = false;
        bool any_iipeikou = false;
        for (const auto& [_, melds] : decomps) {
            int pairs = iipeikou_pair_count(melds);
            if (pairs >= 2) any_ryanpeikou = true;
            if (pairs >= 1) any_iipeikou = true;
        }
        if (any_ryanpeikou) {
            yakus.emplace_back("二杯口", 3);
        } else if (any_iipeikou) {
            yakus.emplace_back("一杯口", 1);
        }
    }

    bool any_sanshoku = false;
    for (const auto& [_, melds] : decomps) {
        if (is_sanshoku_doujun(melds)) { any_sanshoku = true; break; }
    }
    if (any_sanshoku) yakus.emplace_back("三色同顺", is_closed ? 2 : 1);

    bool any_ittsuu = false;
    for (const auto& [_, melds] : decomps) {
        if (is_ittsuu(melds)) { any_ittsuu = true; break; }
    }
    if (any_ittsuu) yakus.emplace_back("一气通贯", is_closed ? 2 : 1);

    bool any_chanta = false;
    bool any_junchan = false;
    for (const auto& [pair_tile, melds] : decomps) {
        if (!has_sequence(melds)) continue;
        if (each_meld_has_terminal_or_honor(pair_tile, melds, true)) any_chanta = true;
        if (each_meld_has_terminal_or_honor(pair_tile, melds, false)) any_junchan = true;
    }
    if (any_junchan) {
        yakus.emplace_back("纯全带幺九", is_closed ? 3 : 2);
    } else if (any_chanta) {
        yakus.emplace_back("混全带幺九", is_closed ? 2 : 1);
    }

    bool any_honroutou = false;
    for (const auto& [_, melds] : decomps) {
        if (is_honroutou(hand34, melds)) { any_honroutou = true; break; }
    }
    if (any_honroutou) yakus.emplace_back("混老头", 2);

    bool any_shousangen = false;
    for (const auto& [pair_tile, melds] : decomps) {
        if (is_shousangen(pair_tile, melds)) { any_shousangen = true; break; }
    }
    if (any_shousangen) yakus.emplace_back("小三元", 2);

    auto standard = finish_yakus(yakus);
    if (chiitoi_total > standard.second) {
        return {chiitoi_candidate, chiitoi_total};
    }
    return standard;
}

}  // namespace

// ============================================================
// Helper predicates
// ============================================================

std::pair<std::set<int>, bool> count_suits_in_hand(const Hand34& hand34) {
    std::set<int> suits;
    bool has_honor = false;
    for (int t = 0; t < 34; ++t) {
        if (hand34[t] <= 0) continue;
        auto s = tile_suit(t);
        if (s.has_value()) {
            suits.insert(s.value());
        } else {
            has_honor = true;
        }
    }
    return {suits, has_honor};
}

bool is_tanyao(const Hand34& hand34) {
    for (int t = 0; t < 34; ++t) {
        if (hand34[t] <= 0) continue;
        if (is_terminal_or_honor(t)) return false;
    }
    return true;
}

std::vector<std::string> yakuhai_from_melds(
    const std::vector<ShapeMeld>& melds, int seat_wind, int round_wind) {
    std::vector<std::string> names;
    for (const auto& meld : melds) {
        if (!is_trip_like(meld)) continue;
        int t = meld.second[0];
        if (IS_DRAGON[t]) {
            names.push_back("役牌·" + tile_to_str(t));
        }
        if (t == seat_wind) {
            names.push_back("役牌·自风");
        }
        if (t == round_wind) {
            names.push_back("役牌·场风");
        }
    }
    return names;
}

bool is_toitoi(const std::vector<ShapeMeld>& melds) {
    return std::all_of(melds.begin(), melds.end(), is_trip_like);
}

bool is_sanankou(const std::vector<ShapeMeld>& melds) {
    int trips = 0;
    for (const auto& meld : melds) {
        if (is_trip_like(meld)) trips += 1;
    }
    return trips >= 3;
}

static std::pair<int, int> seq_key(const std::vector<int>& seq) {
    int a = seq[0];
    auto s = tile_suit(a);
    return {s.value_or(-1), a % 9};
}

bool is_iipeikou(const std::vector<ShapeMeld>& melds) {
    return iipeikou_pair_count(melds) >= 1;
}

bool is_sanshoku_doujun(const std::vector<ShapeMeld>& melds) {
    std::array<std::set<int>, 7> seen;
    for (const auto& [mtype, tiles] : melds) {
        if (mtype != "seq") continue;
        int a = tiles[0];
        auto s = tile_suit(a);
        if (!s.has_value()) continue;
        int pos = a % 9;
        if (pos >= 0 && pos <= 6) {
            seen[pos].insert(s.value());
        }
    }
    for (int pos = 0; pos < 7; ++pos) {
        if (seen[pos].size() == 3) return true;
    }
    return false;
}

bool is_ittsuu(const std::vector<ShapeMeld>& melds) {
    std::array<std::set<int>, 3> by_suit;
    for (const auto& [mtype, tiles] : melds) {
        if (mtype != "seq") continue;
        int a = tiles[0];
        auto s = tile_suit(a);
        if (!s.has_value()) continue;
        by_suit[s.value()].insert(a % 9);
    }
    for (int suit = 0; suit < 3; ++suit) {
        if (by_suit[suit].count(0) && by_suit[suit].count(3) && by_suit[suit].count(6)) {
            return true;
        }
    }
    return false;
}

bool each_meld_has_terminal_or_honor(int pair_tile, const std::vector<ShapeMeld>& melds, bool allow_honor) {
    if (allow_honor) {
        if (!is_terminal_or_honor(pair_tile)) return false;
    } else {
        if (IS_HONOR[pair_tile]) return false;
        if (!IS_TERMINAL[pair_tile]) return false;
    }

    for (const auto& [_, tiles] : melds) {
        if (allow_honor) {
            bool has = false;
            for (int t : tiles) {
                if (is_terminal_or_honor(t)) { has = true; break; }
            }
            if (!has) return false;
        } else {
            for (int t : tiles) {
                if (IS_HONOR[t]) return false;
            }
            bool has_terminal = false;
            for (int t : tiles) {
                if (IS_TERMINAL[t]) { has_terminal = true; break; }
            }
            if (!has_terminal) return false;
        }
    }
    return true;
}

bool is_honroutou(const Hand34& hand34, const std::vector<ShapeMeld>& melds) {
    for (int t = 0; t < 34; ++t) {
        if (hand34[t] <= 0) continue;
        if (!is_terminal_or_honor(t)) return false;
    }
    return std::all_of(melds.begin(), melds.end(), is_trip_like);
}

bool is_shousangen(int pair_tile, const std::vector<ShapeMeld>& melds) {
    int trip_dragons = 0;
    bool has_pair_dragon = IS_DRAGON[pair_tile];
    for (const auto& meld : melds) {
        if (is_trip_like(meld) && IS_DRAGON[meld.second[0]]) {
            trip_dragons += 1;
        }
    }
    return trip_dragons == 2 && has_pair_dragon;
}

// ============================================================
// Main yaku analysis
// ============================================================

std::pair<std::vector<std::pair<std::string, int>>, int> analyze_yaku(
    const Hand34& hand34,
    const std::string& win_type,
    int seat_wind,
    int round_wind,
    bool is_closed,
    int win_tile) {
    return analyze_yaku_impl({}, hand34, win_type, seat_wind, round_wind, is_closed, win_tile);
}

std::pair<std::vector<std::pair<std::string, int>>, int> analyze_yaku_with_melds(
    const std::vector<Meld>& fixed_melds,
    const Hand34& hand34,
    const std::string& win_type,
    int seat_wind,
    int round_wind,
    bool is_closed,
    int win_tile) {
    return analyze_yaku_impl(fixed_melds, hand34, win_type, seat_wind, round_wind, is_closed, win_tile);
}

}  // namespace mahjong
