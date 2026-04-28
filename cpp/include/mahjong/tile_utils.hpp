#pragma once

#include <array>
#include <random>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mahjong {

// Lookup tables (O(1) access, replace std::set)
inline constexpr std::array<bool, 34> IS_TERMINAL = {
    true,  false, false, false, false, false, false, false, true,
    true,  false, false, false, false, false, false, false, true,
    true,  false, false, false, false, false, false, false, true,
    false, false, false, false, false, false, false
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
std::string hand_to_str(const std::array<int, 34>& hand34);

// Hand manipulation
void hand34_add(std::array<int, 34>& hand, int t);
void hand34_remove(std::array<int, 34>& hand, int t);
std::array<int, 34> copy_hand(const std::array<int, 34>& hand);

// Wall generation
std::vector<int> make_wall(std::mt19937& rng);

// Tile classification
bool is_terminal_or_honor(int t);
std::optional<int> tile_suit(int t);

// Chi options
std::vector<std::pair<int, int>> chi_options(int tile);

}  // namespace mahjong
