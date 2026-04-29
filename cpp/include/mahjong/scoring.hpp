#pragma once

#include <map>
#include <string>
#include <vector>

#include "mahjong/hand_analysis.hpp"
#include "mahjong/types.hpp"

namespace mahjong {

struct PointResult {
    std::vector<int> score_delta = {0, 0, 0, 0};
    std::map<std::string, int> payments;
    std::string level = "none";
};

// Dora
int dora_from_indicator(int indicator);
int count_dora(const Hand34& hand34, const std::vector<int>& dora_indicators);

// Score level & base points
std::string point_level(int han, int fu, bool kazoe_yakuman = true, bool kiriage_mangan = false);
int base_points(int han, int fu, bool kazoe_yakuman = true, bool kiriage_mangan = false);

// Payment resolution
PointResult resolve_ron(int winner, int loser, int han, int fu, int dealer,
                        int honba = 0, int riichi_sticks = 0,
                        bool kazoe_yakuman = true, bool kiriage_mangan = false);
PointResult resolve_tsumo(int winner, int han, int fu, int dealer,
                          int honba = 0, int riichi_sticks = 0,
                          bool kazoe_yakuman = true, bool kiriage_mangan = false);

// Fu calculation
int calculate_fu(const std::vector<Meld>& open_melds,
                 const Hand34& hand34,
                 const std::string& win_type,
                 int win_tile,
                 int seat_wind, int round_wind,
                 bool is_closed);

}  // namespace mahjong
