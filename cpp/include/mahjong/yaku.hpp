#pragma once

#include <set>
#include <string>
#include <utility>
#include <vector>

#include "mahjong/hand_analysis.hpp"

namespace mahjong {

// Yaku analysis: returns (yaku_list, total_han)
std::pair<std::vector<std::pair<std::string, int>>, int> analyze_yaku(
    const std::vector<int>& hand34,
    const std::string& win_type,   // "tsumo" or "ron"
    int seat_wind,
    int round_wind,
    bool is_closed = true
);

// Yaku helper predicates (exposed for testing)
bool is_tanyao(const std::vector<int>& hand34);
std::vector<std::string> yakuhai_from_melds(
    const std::vector<ShapeMeld>& melds, int seat_wind, int round_wind);
bool is_toitoi(const std::vector<ShapeMeld>& melds);
bool is_sanankou(const std::vector<ShapeMeld>& melds);
bool is_iipeikou(const std::vector<ShapeMeld>& melds);
bool is_sanshoku_doujun(const std::vector<ShapeMeld>& melds);
bool is_ittsuu(const std::vector<ShapeMeld>& melds);
bool each_meld_has_terminal_or_honor(int pair_tile, const std::vector<ShapeMeld>& melds, bool allow_honor);
bool is_honroutou(const std::vector<int>& hand34, const std::vector<ShapeMeld>& melds);
bool is_shousangen(int pair_tile, const std::vector<ShapeMeld>& melds);
std::pair<std::set<int>, bool> count_suits_in_hand(const std::vector<int>& hand34);

}  // namespace mahjong
