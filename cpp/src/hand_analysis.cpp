#include "mahjong/hand_analysis.hpp"
#include "mahjong/tile_utils.hpp"

#include <numeric>

namespace mahjong {

bool is_kokushi(const std::vector<int>& hand34) {
    bool dup = false;
    for (int t = 0; t < 34; ++t) {
        if (IS_KOKUSHI[t]) {
            if (hand34[t] == 0) return false;
            if (hand34[t] >= 2) dup = true;
        } else {
            if (hand34[t] != 0) return false;
        }
    }
    return dup;
}

bool is_chiitoi(const std::vector<int>& hand34) {
    int pairs = 0;
    for (int c : hand34) {
        if (c == 2) {
            pairs += 1;
        } else if (c == 0) {
            continue;
        } else {
            return false;
        }
    }
    return pairs == 7;
}

bool can_form_melds(std::vector<int>& counts, int start) {
    int i = start;
    while (i < 34 && counts[i] == 0) i += 1;
    if (i == 34) return true;

    // Try triplet
    if (counts[i] >= 3) {
        counts[i] -= 3;
        if (can_form_melds(counts, i)) {
            counts[i] += 3;
            return true;
        }
        counts[i] += 3;
    }

    // Try sequence (only for suited tiles)
    if (i <= 26) {
        int suit = i / 9;
        int pos = i % 9;
        if (pos <= 6) {
            int a = i, b = i + 1, c = i + 2;
            if ((b / 9) == suit && (c / 9) == suit && counts[b] > 0 && counts[c] > 0) {
                counts[a] -= 1;
                counts[b] -= 1;
                counts[c] -= 1;
                if (can_form_melds(counts, i)) {
                    counts[a] += 1;
                    counts[b] += 1;
                    counts[c] += 1;
                    return true;
                }
                counts[a] += 1;
                counts[b] += 1;
                counts[c] += 1;
            }
        }
    }

    return false;
}

bool is_standard_agari(const std::vector<int>& hand34) {
    for (int t = 0; t < 34; ++t) {
        if (hand34[t] >= 2) {
            auto counts = copy_hand(hand34);
            counts[t] -= 2;
            if (can_form_melds(counts, 0)) return true;
        }
    }
    return false;
}

bool is_agari(const std::vector<int>& hand34) {
    int total = 0;
    for (int c : hand34) total += c;
    if (total != 14) return false;
    if (is_kokushi(hand34)) return true;
    if (is_chiitoi(hand34)) return true;
    return is_standard_agari(hand34);
}

int count_yaochu_types(const std::vector<int>& hand34) {
    int count = 0;
    for (int t = 0; t < 34; ++t) {
        if (IS_KOKUSHI[t] && hand34[t] > 0) count += 1;
    }
    return count;
}

bool is_tenpai(const std::vector<int>& hand34) {
    int total = 0;
    for (int c : hand34) total += c;
    if (total != 13) return false;
    for (int t = 0; t < 34; ++t) {
        if (hand34[t] >= 4) continue;
        auto tmp = copy_hand(hand34);
        tmp[t] += 1;
        if (is_agari(tmp)) return true;
    }
    return false;
}

// ============================================================
// Decomposition enumeration
// ============================================================

std::vector<Decomposition> gen_standard_decompositions(const std::vector<int>& hand34) {
    int total = 0;
    for (int c : hand34) total += c;
    if (total != 14) return {};

    std::vector<Decomposition> results;

    std::function<void(std::vector<int>&, int, std::vector<ShapeMeld>&, int)> backtrack;
    backtrack = [&](std::vector<int>& counts, int start, std::vector<ShapeMeld>& melds, int pair_tile) {
        int i = start;
        while (i < 34 && counts[i] == 0) i += 1;
        if (i == 34) {
            results.emplace_back(pair_tile, melds);
            return;
        }

        // Try triplet
        if (counts[i] >= 3) {
            counts[i] -= 3;
            melds.emplace_back("trip", std::vector<int>{i, i, i});
            backtrack(counts, i, melds, pair_tile);
            melds.pop_back();
            counts[i] += 3;
        }

        // Try sequence
        if (i <= 26) {
            int suit = i / 9;
            int pos = i % 9;
            if (pos <= 6) {
                int a = i, b = i + 1, c = i + 2;
                if ((b / 9) == suit && (c / 9) == suit && counts[b] > 0 && counts[c] > 0) {
                    counts[a] -= 1;
                    counts[b] -= 1;
                    counts[c] -= 1;
                    melds.emplace_back("seq", std::vector<int>{a, b, c});
                    backtrack(counts, i, melds, pair_tile);
                    melds.pop_back();
                    counts[a] += 1;
                    counts[b] += 1;
                    counts[c] += 1;
                }
            }
        }
    };

    for (int t = 0; t < 34; ++t) {
        if (hand34[t] >= 2) {
            auto counts = copy_hand(hand34);
            counts[t] -= 2;
            std::vector<ShapeMeld> melds;
            backtrack(counts, 0, melds, t);
        }
    }

    return results;
}

std::vector<Decomposition> gen_concealed_decompositions(const std::vector<int>& hand34, int target_melds) {
    int total = 0;
    for (int c : hand34) total += c;
    if (total != target_melds * 3 + 2) return {};

    std::vector<Decomposition> results;

    std::function<void(std::vector<int>&, int, std::vector<ShapeMeld>&, int)> backtrack;
    backtrack = [&](std::vector<int>& counts, int start, std::vector<ShapeMeld>& melds, int pair_tile) {
        if (static_cast<int>(melds.size()) > target_melds) return;
        int i = start;
        while (i < 34 && counts[i] == 0) i += 1;
        if (i == 34) {
            if (static_cast<int>(melds.size()) == target_melds) {
                results.emplace_back(pair_tile, melds);
            }
            return;
        }

        // Try triplet
        if (counts[i] >= 3) {
            counts[i] -= 3;
            melds.emplace_back("trip", std::vector<int>{i, i, i});
            backtrack(counts, i, melds, pair_tile);
            melds.pop_back();
            counts[i] += 3;
        }

        // Try sequence
        if (i <= 26) {
            int suit = i / 9;
            int pos = i % 9;
            if (pos <= 6) {
                int a = i, b = i + 1, c = i + 2;
                if ((b / 9) == suit && (c / 9) == suit && counts[b] > 0 && counts[c] > 0) {
                    counts[a] -= 1;
                    counts[b] -= 1;
                    counts[c] -= 1;
                    melds.emplace_back("seq", std::vector<int>{a, b, c});
                    backtrack(counts, i, melds, pair_tile);
                    melds.pop_back();
                    counts[a] += 1;
                    counts[b] += 1;
                    counts[c] += 1;
                }
            }
        }
    };

    for (int t = 0; t < 34; ++t) {
        if (hand34[t] >= 2) {
            auto counts = copy_hand(hand34);
            counts[t] -= 2;
            std::vector<ShapeMeld> melds;
            backtrack(counts, 0, melds, t);
        }
    }

    return results;
}

}  // namespace mahjong
