#pragma once

#include <string>

namespace mahjong {

struct RuleConfig {
    std::string name = "mahjongsoul_common";
    bool allow_multi_ron = false;
    bool use_dead_wall = true;
    int dead_wall_size = 14;
    bool enable_kan = true;
    int max_kan_per_hand = 4;

    bool enable_nagashi_mangan = true;
    bool enable_kyuushu_kyuuhai = true;
    bool enable_suufon_renda = true;
    bool enable_suucha_riichi = true;
    bool enable_suukan_sanra = true;

    int aka_dora_count = 3;
    bool ura_dora_enabled = true;
    bool kazoe_yakuman = true;
    bool kiriage_mangan = false;

    bool enforce_furiten = true;
    bool enforce_same_turn_furiten = true;
    bool enforce_riichi_furiten = true;
};

}  // namespace mahjong
