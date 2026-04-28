#include "mahjong/yaku.hpp"
#include "mahjong/tile_utils.hpp"

#include <algorithm>

namespace mahjong {

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
    for (const auto& [mtype, tiles] : melds) {
        if (mtype != "trip") continue;
        int t = tiles[0];
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
    return std::all_of(melds.begin(), melds.end(),
                       [](const ShapeMeld& m) { return m.first == "trip"; });
}

bool is_sanankou(const std::vector<ShapeMeld>& melds) {
    int trips = 0;
    for (const auto& [mtype, _] : melds) {
        if (mtype == "trip") trips += 1;
    }
    return trips >= 3;
}

static std::pair<int, int> seq_key(const std::vector<int>& seq) {
    int a = seq[0];
    auto s = tile_suit(a);
    return {s.value_or(-1), a % 9};
}

bool is_iipeikou(const std::vector<ShapeMeld>& melds) {
    std::vector<std::pair<int, int>> keys;
    for (const auto& [mtype, tiles] : melds) {
        if (mtype == "seq") {
            keys.push_back(seq_key(tiles));
        }
    }
    for (size_t i = 0; i < keys.size(); ++i) {
        for (size_t j = i + 1; j < keys.size(); ++j) {
            if (keys[i] == keys[j]) return true;
        }
    }
    return false;
}

bool is_sanshoku_doujun(const std::vector<ShapeMeld>& melds) {
    // For each starting position 0..6, track which suits have a seq there
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
    // Check pair
    if (allow_honor) {
        if (!is_terminal_or_honor(pair_tile)) return false;
    } else {
        if (IS_HONOR[pair_tile]) return false;
        if (!IS_TERMINAL[pair_tile]) return false;
    }

    // Check each meld
    for (const auto& [mtype, tiles] : melds) {
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
    return std::all_of(melds.begin(), melds.end(),
                       [](const ShapeMeld& m) { return m.first == "trip"; });
}

bool is_shousangen(int pair_tile, const std::vector<ShapeMeld>& melds) {
    int trip_dragons = 0;
    bool has_pair_dragon = IS_DRAGON[pair_tile];
    for (const auto& [mtype, tiles] : melds) {
        if (mtype == "trip" && IS_DRAGON[tiles[0]]) {
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
    bool is_closed) {

    std::vector<std::pair<std::string, int>> yakus;

    // Special hands: kokushi / chiitoi
    if (is_kokushi(hand34)) {
        yakus.emplace_back("国士无双(役满)", 13);
        int total = 0;
        for (auto& [_, h] : yakus) total += h;
        return {yakus, total};
    }

    if (is_chiitoi(hand34)) {
        yakus.emplace_back("七对子", 2);
        if (is_tanyao(hand34)) {
            yakus.emplace_back("断幺九", 1);
        }
        auto [suits, has_honor] = count_suits_in_hand(hand34);
        if (suits.size() == 1) {
            if (has_honor) {
                yakus.emplace_back("混一色", 3);
            } else {
                yakus.emplace_back("清一色", 6);
            }
        }
        int total = 0;
        for (auto& [_, h] : yakus) total += h;
        return {yakus, total};
    }

    // Standard form: enumerate decompositions
    auto decomps = gen_standard_decompositions(hand34);
    if (decomps.empty()) {
        return {yakus, 0};
    }

    // Menzen tsumo
    if (win_type == "tsumo" && is_closed) {
        yakus.emplace_back("门前清自摸和", 1);
    }

    // Tanyao
    if (is_tanyao(hand34)) {
        yakus.emplace_back("断幺九", 1);
    }

    // Honitsu / Chinitsu
    auto [suits, has_honor] = count_suits_in_hand(hand34);
    if (suits.size() == 1) {
        if (has_honor) {
            yakus.emplace_back("混一色", 3);
        } else {
            yakus.emplace_back("清一色", 6);
        }
    }

    // Yakuhai (from any decomposition)
    std::set<std::string> yakuhai_names;
    for (const auto& [pair_tile, melds] : decomps) {
        for (const auto& name : yakuhai_from_melds(melds, seat_wind, round_wind)) {
            yakuhai_names.insert(name);
        }
    }
    for (const auto& name : yakuhai_names) {
        yakus.emplace_back(name, 1);
    }

    // Toitoi
    bool any_toitoi = false;
    for (const auto& [_, melds] : decomps) {
        if (is_toitoi(melds)) { any_toitoi = true; break; }
    }
    if (any_toitoi) yakus.emplace_back("对对和", 2);

    // Sanankou
    bool any_sanankou = false;
    for (const auto& [_, melds] : decomps) {
        if (is_sanankou(melds)) { any_sanankou = true; break; }
    }
    if (any_sanankou) yakus.emplace_back("三暗刻", 2);

    // Iipeikou
    if (is_closed) {
        bool any_iipeikou = false;
        for (const auto& [_, melds] : decomps) {
            if (is_iipeikou(melds)) { any_iipeikou = true; break; }
        }
        if (any_iipeikou) yakus.emplace_back("一杯口", 1);
    }

    // Sanshoku doujun
    bool any_sanshoku = false;
    for (const auto& [_, melds] : decomps) {
        if (is_sanshoku_doujun(melds)) { any_sanshoku = true; break; }
    }
    if (any_sanshoku) yakus.emplace_back("三色同顺", is_closed ? 2 : 1);

    // Ittsuu
    bool any_ittsuu = false;
    for (const auto& [_, melds] : decomps) {
        if (is_ittsuu(melds)) { any_ittsuu = true; break; }
    }
    if (any_ittsuu) yakus.emplace_back("一气通贯", is_closed ? 2 : 1);

    // Chanta / Junchan
    bool any_chanta = false;
    bool any_junchan = false;
    for (const auto& [pair_tile, melds] : decomps) {
        if (each_meld_has_terminal_or_honor(pair_tile, melds, true)) any_chanta = true;
        if (each_meld_has_terminal_or_honor(pair_tile, melds, false)) any_junchan = true;
    }
    if (any_chanta) yakus.emplace_back("混全带幺九", is_closed ? 2 : 1);
    if (any_junchan) yakus.emplace_back("纯全带幺九", is_closed ? 3 : 2);

    // Honroutou
    bool any_honroutou = false;
    for (const auto& [_, melds] : decomps) {
        if (is_honroutou(hand34, melds)) { any_honroutou = true; break; }
    }
    if (any_honroutou) yakus.emplace_back("混老头", 2);

    // Shousangen
    bool any_shousangen = false;
    for (const auto& [pair_tile, melds] : decomps) {
        if (is_shousangen(pair_tile, melds)) { any_shousangen = true; break; }
    }
    if (any_shousangen) yakus.emplace_back("小三元", 2);

    int total_han = 0;
    for (const auto& [_, h] : yakus) total_han += h;
    return {yakus, total_han};
}

}  // namespace mahjong
