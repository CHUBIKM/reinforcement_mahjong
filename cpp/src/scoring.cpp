#include "mahjong/scoring.hpp"
#include "mahjong/tile_utils.hpp"

#include <cmath>
#include <algorithm>

namespace mahjong {

static int ceil100(int x) {
    return static_cast<int>(std::ceil(x / 100.0) * 100.0);
}

int dora_from_indicator(int indicator) {
    if (0 <= indicator && indicator <= 26) {
        int base = (indicator / 9) * 9;
        int pos = indicator % 9;
        return base + ((pos + 1) % 9);
    }
    if (27 <= indicator && indicator <= 30) {
        return 27 + ((indicator - 27 + 1) % 4);
    }
    if (31 <= indicator && indicator <= 33) {
        return 31 + ((indicator - 31 + 1) % 3);
    }
    throw std::invalid_argument("invalid indicator: " + std::to_string(indicator));
}

int count_dora(const Hand34& hand34, const std::vector<int>& dora_indicators) {
    int total = 0;
    for (int ind : dora_indicators) {
        int dora_tile = dora_from_indicator(ind);
        total += hand34[dora_tile];
    }
    return total;
}

std::string point_level(int han, int fu, bool kazoe_yakuman, bool kiriage_mangan) {
    if (han >= 13) return "yakuman";
    if (han >= 11) return "sanbaiman";
    if (han >= 8)  return "baiman";
    if (han >= 6)  return "haneman";
    if (han >= 5)  return "mangan";
    if (han == 4 && fu >= 40) return "mangan";
    if (han == 3 && fu >= 70) return "mangan";
    if (kiriage_mangan && ((han == 4 && fu == 30) || (han == 3 && fu == 60))) {
        return "mangan";
    }
    return "regular";
}

int base_points(int han, int fu, bool kazoe_yakuman, bool kiriage_mangan) {
    auto level = point_level(han, fu, kazoe_yakuman, kiriage_mangan);
    if (level == "yakuman")   return 8000;
    if (level == "sanbaiman") return 6000;
    if (level == "baiman")    return 4000;
    if (level == "haneman")   return 3000;
    if (level == "mangan")    return 2000;
    return std::min(2000, fu * (1 << (han + 2)));
}

PointResult resolve_ron(int winner, int loser, int han, int fu, int dealer,
                        int honba, int riichi_sticks,
                        bool kazoe_yakuman, bool kiriage_mangan) {
    int base = base_points(han, fu, kazoe_yakuman, kiriage_mangan);
    bool is_dealer = (winner == dealer);
    int ron_points = ceil100(base * (is_dealer ? 6 : 4));
    int honba_bonus = honba * 300;

    std::vector<int> delta(4, 0);
    int total_gain = ron_points + honba_bonus + (riichi_sticks * 1000);
    int total_loss = ron_points + honba_bonus;

    delta[winner] += total_gain;
    delta[loser]  -= total_loss;

    return PointResult{
        delta,
        {{"ron", ron_points}, {"honba", honba_bonus}, {"riichi_sticks", riichi_sticks * 1000}},
        point_level(han, fu, kazoe_yakuman, kiriage_mangan)
    };
}

PointResult resolve_tsumo(int winner, int han, int fu, int dealer,
                          int honba, int riichi_sticks,
                          bool kazoe_yakuman, bool kiriage_mangan) {
    int base = base_points(han, fu, kazoe_yakuman, kiriage_mangan);
    bool is_dealer = (winner == dealer);

    std::vector<int> delta(4, 0);
    std::map<std::string, int> payments;

    if (is_dealer) {
        int pay_each = ceil100(base * 2);
        for (int p = 0; p < 4; ++p) {
            if (p == winner) continue;
            delta[p] -= pay_each + honba * 100;
            delta[winner] += pay_each + honba * 100;
        }
        payments = {{"from_each", pay_each}, {"honba_each", honba * 100}, {"riichi_sticks", riichi_sticks * 1000}};
    } else {
        int pay_non_dealer = ceil100(base);
        int pay_dealer = ceil100(base * 2);
        for (int p = 0; p < 4; ++p) {
            if (p == winner) continue;
            int amt = (p == dealer) ? pay_dealer + honba * 100 : pay_non_dealer + honba * 100;
            delta[p] -= amt;
            delta[winner] += amt;
        }
        payments = {{"from_dealer", pay_dealer}, {"from_non_dealer", pay_non_dealer},
                     {"honba_each", honba * 100}, {"riichi_sticks", riichi_sticks * 1000}};
    }

    delta[winner] += riichi_sticks * 1000;

    return PointResult{
        delta,
        payments,
        point_level(han, fu, kazoe_yakuman, kiriage_mangan)
    };
}

// ============================================================
// Fu calculation
// ============================================================

namespace {

int valuable_pair_fu(int tile, int seat_wind, int round_wind) {
    int fu = 0;
    if (IS_DRAGON[tile]) fu += 2;
    if (tile == seat_wind) fu += 2;
    if (tile == round_wind) fu += 2;
    return fu;
}

int trip_kan_fu(int tile, bool is_open, bool is_kan) {
    bool yaochu = is_terminal_or_honor(tile);
    if (is_kan) {
        if (is_open) return yaochu ? 16 : 8;
        return yaochu ? 32 : 16;
    }
    if (is_open) return yaochu ? 4 : 2;
    return yaochu ? 8 : 4;
}

int wait_fu_for_seq(const std::vector<int>& seq, int win_tile) {
    bool found = false;
    for (int t : seq) { if (t == win_tile) { found = true; break; } }
    if (!found) return 0;

    int a = seq[0], b = seq[1], c = seq[2];
    if (win_tile == b) return 2;  // kanchan
    int start = a % 9;
    if (win_tile == a && start == 6) return 2;  // penchan 7 on 789
    if (win_tile == c && start == 0) return 2;  // penchan 3 on 123
    return 0;  // ryanmen
}

std::vector<ShapeMeld> open_shape_melds(const std::vector<Meld>& melds) {
    std::vector<ShapeMeld> out;
    for (const auto& m : melds) {
        if (m.type == "chi") {
            out.emplace_back("seq", std::vector<int>(m.tiles.begin(), m.tiles.begin() + 3));
        } else if (m.type == "pon") {
            out.emplace_back("trip", std::vector<int>{m.tiles[0], m.tiles[0], m.tiles[0]});
        } else if (m.type == "minkan" || m.type == "kakan" || m.type == "ankan") {
            out.emplace_back("kan", std::vector<int>{m.tiles[0], m.tiles[0], m.tiles[0], m.tiles[0]});
        }
    }
    return out;
}

}  // anonymous namespace

int calculate_fu(const std::vector<Meld>& open_melds_vec,
                 const Hand34& hand34,
                 const std::string& win_type,
                 int win_tile,
                 int seat_wind, int round_wind,
                 bool is_closed) {
    if (is_kokushi(hand34)) return 0;
    if (is_chiitoi(hand34)) return 25;

    auto open_shape = open_shape_melds(open_melds_vec);
    int target_melds = 4 - static_cast<int>(open_shape.size());
    if (target_melds < 0) return 30;

    // Build concealed hand by subtracting open meld tiles
    auto concealed = copy_hand(hand34);
    for (const auto& m : open_melds_vec) {
        if (m.type == "ankan") {
            for (int t : m.tiles) hand34_remove(concealed, t);
        }
    }

    auto decomps = gen_concealed_decompositions(concealed, target_melds);
    if (decomps.empty()) return 30;

    int best_fu = 20;

    // Open meld fu
    int open_meld_fu = 0;
    for (const auto& m : open_melds_vec) {
        if (m.type == "pon") {
            open_meld_fu += trip_kan_fu(m.tiles[0], true, false);
        } else if (m.type == "minkan" || m.type == "kakan") {
            open_meld_fu += trip_kan_fu(m.tiles[0], true, true);
        } else if (m.type == "ankan") {
            open_meld_fu += trip_kan_fu(m.tiles[0], false, true);
        }
    }

    for (const auto& [pair_tile, concealed_melds] : decomps) {
        int pair_fu = valuable_pair_fu(pair_tile, seat_wind, round_wind);
        int base_without_wait = 20;
        if (win_type == "ron" && is_closed) base_without_wait += 10;
        base_without_wait += open_meld_fu;

        std::vector<std::vector<int>> seq_melds;
        std::vector<std::vector<int>> trip_melds;
        for (const auto& [mtype, tiles] : concealed_melds) {
            if (mtype == "seq") seq_melds.push_back(tiles);
            else trip_melds.push_back(tiles);
        }

        // Candidate winning component assignments
        struct WaitCand { std::string kind; int idx; };
        std::vector<WaitCand> wait_candidates;
        if (win_tile >= 0 && pair_tile == win_tile) {
            wait_candidates.push_back({"pair", -1});
        }
        if (win_tile >= 0) {
            for (int i = 0; i < static_cast<int>(seq_melds.size()); ++i) {
                for (int t : seq_melds[i]) { if (t == win_tile) { wait_candidates.push_back({"seq", i}); break; } }
            }
            for (int i = 0; i < static_cast<int>(trip_melds.size()); ++i) {
                for (int t : trip_melds[i]) { if (t == win_tile) { wait_candidates.push_back({"trip", i}); break; } }
            }
        }
        if (wait_candidates.empty()) {
            wait_candidates.push_back({"none", -1});
        }

        for (const auto& [kind, idx] : wait_candidates) {
            int meld_fu = 0;
            for (int i = 0; i < static_cast<int>(trip_melds.size()); ++i) {
                bool ron_trip_open = (win_type == "ron" && kind == "trip" && idx == i);
                meld_fu += trip_kan_fu(trip_melds[i][0], ron_trip_open, false);
            }

            int wait_fu = 0;
            if (kind == "pair") {
                wait_fu += 2;  // tanki
            } else if (kind == "seq") {
                wait_fu += wait_fu_for_seq(seq_melds[idx], win_tile);
            }

            int fu = base_without_wait + pair_fu + meld_fu + wait_fu;
            if (win_type == "tsumo") {
                fu += 2;
                if (is_closed && open_meld_fu == 0 && pair_fu == 0 && meld_fu == 0 && wait_fu == 0) {
                    fu = 20;
                }
            }

            if (fu != 25) {
                if (win_type == "ron" && fu == 20) fu = 30;
                fu = ((fu + 9) / 10) * 10;
            }
            best_fu = std::max(best_fu, fu);
        }
    }

    return best_fu;
}

}  // namespace mahjong
