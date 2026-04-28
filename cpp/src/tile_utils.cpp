#include "mahjong/tile_utils.hpp"

#include <algorithm>
#include <stdexcept>

namespace mahjong {

std::string tile_to_str(int t) {
    static const char* suit_names[] = {"m", "p", "s"};
    static const char* honor_names[] = {"东", "南", "西", "北", "白", "发", "中"};

    if (0 <= t && t <= 26) {
        int num = (t % 9) + 1;
        return std::to_string(num) + suit_names[t / 9];
    }
    if (27 <= t && t <= 33) {
        return honor_names[t - 27];
    }
    return "<?>(" + std::to_string(t) + ")";
}

std::string hand_to_str(const std::vector<int>& hand34) {
    static const char* suit_names[] = {"m", "p", "s"};
    static const char* honor_names[] = {"东", "南", "西", "北", "白", "发", "中"};

    std::vector<std::string> parts;

    for (int s = 0; s < 3; ++s) {
        std::string nums;
        for (int i = 0; i < 9; ++i) {
            int t = s * 9 + i;
            nums.append(static_cast<size_t>(hand34[t]), static_cast<char>('1' + i));
        }
        if (!nums.empty()) {
            parts.push_back(nums + suit_names[s]);
        }
    }

    std::string honors;
    for (int i = 0; i < 7; ++i) {
        int t = 27 + i;
        for (int c = 0; c < hand34[t]; ++c) {
            honors += honor_names[i];
        }
    }
    if (!honors.empty()) {
        parts.push_back(honors);
    }

    if (parts.empty()) {
        return "(空)";
    }

    std::string out;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i > 0) out += " ";
        out += parts[i];
    }
    return out;
}

void hand34_add(std::vector<int>& hand, int t) {
    hand[t] += 1;
}

void hand34_remove(std::vector<int>& hand, int t) {
    if (hand[t] <= 0) {
        throw std::invalid_argument("手里没有 " + tile_to_str(t) + "，无法移除。");
    }
    hand[t] -= 1;
}

std::vector<int> copy_hand(const std::vector<int>& hand) {
    return std::vector<int>(hand);
}

std::vector<int> make_wall(std::mt19937& rng) {
    std::vector<int> wall;
    wall.reserve(136);
    for (int t = 0; t < 34; ++t) {
        wall.insert(wall.end(), 4, t);
    }
    std::shuffle(wall.begin(), wall.end(), rng);
    return wall;
}

bool is_terminal_or_honor(int t) {
    if (t < 0 || t >= 34) return false;
    return IS_TERMINAL_OR_HONOR[t];
}

std::optional<int> tile_suit(int t) {
    if (0 <= t && t <= 26) {
        return t / 9;
    }
    return std::nullopt;
}

std::vector<std::pair<int, int>> chi_options(int tile) {
    std::vector<std::pair<int, int>> opts;
    if (tile < 0 || tile > 26) return opts;  // honor tiles cannot be chied

    int suit = tile / 9;
    int pos = tile % 9;

    // (tile-2, tile-1, tile)
    if (pos >= 2) {
        int a = tile - 2, b = tile - 1;
        if (a / 9 == suit && b / 9 == suit) {
            opts.emplace_back(a, b);
        }
    }
    // (tile-1, tile, tile+1)
    if (pos >= 1 && pos <= 7) {
        int a = tile - 1, b = tile + 1;
        if (a / 9 == suit && b / 9 == suit) {
            opts.emplace_back(a, b);
        }
    }
    // (tile, tile+1, tile+2)
    if (pos <= 6) {
        int a = tile + 1, b = tile + 2;
        if (a / 9 == suit && b / 9 == suit) {
            opts.emplace_back(a, b);
        }
    }
    return opts;
}

}  // namespace mahjong
