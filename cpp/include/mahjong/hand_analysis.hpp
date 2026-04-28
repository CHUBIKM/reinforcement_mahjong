#pragma once

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace mahjong {

// Shape meld representation: ("seq", [a,b,c]) or ("trip", [t,t,t]) or ("kan", [t,t,t,t])
using ShapeMeld = std::pair<std::string, std::vector<int>>;
// Decomposition: (pair_tile, list_of_melds)
using Decomposition = std::pair<int, std::vector<ShapeMeld>>;

// Basic hand checks
bool is_kokushi(const std::vector<int>& hand34);
bool is_chiitoi(const std::vector<int>& hand34);
bool can_form_melds(std::vector<int>& counts, int start = 0);
bool is_standard_agari(const std::vector<int>& hand34);
bool is_agari(const std::vector<int>& hand34);
int count_yaochu_types(const std::vector<int>& hand34);
bool is_tenpai(const std::vector<int>& hand34);

// Decomposition enumeration
std::vector<Decomposition> gen_standard_decompositions(const std::vector<int>& hand34);
std::vector<Decomposition> gen_concealed_decompositions(const std::vector<int>& hand34, int target_melds);

}  // namespace mahjong
