#pragma once

#include <array>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace mahjong {

using Hand34 = std::array<int, 34>;

// Shape meld: ("seq", [a,b,c]) or ("trip", [t,t,t]) or ("kan", [t,t,t,t])
using ShapeMeld = std::pair<std::string, std::vector<int>>;
// Decomposition: (pair_tile, list_of_melds)
using Decomposition = std::pair<int, std::vector<ShapeMeld>>;

// Basic hand checks
bool is_kokushi(const Hand34& hand34);
bool is_chiitoi(const Hand34& hand34);
bool can_form_melds(Hand34& counts, int start = 0);
bool is_standard_agari(const Hand34& hand34);
bool is_agari(const Hand34& hand34);
int count_yaochu_types(const Hand34& hand34);
bool is_tenpai(const Hand34& hand34);

// Decomposition enumeration
std::vector<Decomposition> gen_standard_decompositions(const Hand34& hand34);
std::vector<Decomposition> gen_concealed_decompositions(const Hand34& hand34, int target_melds);

}  // namespace mahjong
