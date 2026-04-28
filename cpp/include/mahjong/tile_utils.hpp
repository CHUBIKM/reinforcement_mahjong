#pragma once

#include <array>
#include <random>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mahjong {

// Lookup tables (replace std::set for O(1) access)
inline constexpr std::array<bool, 34> IS_TERMINAL = {
    true,  false, false, false, false, false, false, false, true,   // 0-8: manzu
    true,  false, false, false, false, false, false, false, true,   // 9-17: pinzu
    true,  false, false, false, false, false, false, false, true,   // 18-26: souzu
    false, false, false, false, false, false, false                 // 27-33: honors
};

inline constexpr std::array<bool, 34> IS_HONOR = {
    false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false,
    true,  true,  true,  true,  true,  true,  true
};

inline constexpr std::array<bool, 34> IS_DRAGON = {
    false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false,
    false, false, false, false, true,  true,  true
};

inline constexpr std::array<bool, 34> IS_KOKUSHI = {
    true,  false, false, false, false, false, false, false, true,
    true,  false, false, false, false, false, false, false, true,
    true,  false, false, false, false, false, false, false, true,
    true,  true,  true,  true,  true,  true,  true
};

inline constexpr std::array<bool, 34> IS_TERMINAL_OR_HONOR = {
    true,  false, false, false, false, false, false, false, true,
    true,  false, false, false, false, false, false, false, true,
    true,  false, false, false, false, false, false, false, true,
    true,  true,  true,  true,  true,  true,  true
};

// String conversion
std::string tile_to_str(int t);
std::string hand_to_str(const std::vector<int>& hand34);

// Hand manipulation
void hand34_add(std::vector<int>& hand, int t);
void hand34_remove(std::vector<int>& hand, int t);
std::vector<int> copy_hand(const std::vector<int>& hand);

// Wall generation
std::vector<int> make_wall(std::mt19937& rng);

// Tile classification
bool is_terminal_or_honor(int t);
std::optional<int> tile_suit(int t);

// Chi options: returns pairs (a, b) such that a, b + discarded tile form a sequence
std::vector<std::pair<int, int>> chi_options(int tile);

}  // namespace mahjong
