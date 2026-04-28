#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <unordered_set>
#include <variant>
#include <vector>

namespace mahjong {

// ============================================================
// Enumerations
// ============================================================

enum class Phase {
    DRAW,
    DISCARD,
    RESPONSE,
    END,
};

enum class ActionType {
    DISCARD,
    TSUMO,
    RON,
    PASS,
    CHI,
    PON,
    KAN,
    RIICHI,
    ABORTIVE_DRAW,
};

std::string phase_name(Phase p);
std::string action_type_name(ActionType t);

// ============================================================
// Action
// ============================================================

struct ActionInfo {
    std::optional<int> from;                // discarder seat (for ron/pon/chi/minkan)
    std::vector<int> use;                   // tiles used for chi [a, b]
    std::optional<int> chi_pattern;         // chi pattern index
    std::optional<std::string> kan_type;    // "ANKAN", "KAKAN", "MINKAN"

    bool operator==(const ActionInfo& o) const {
        return from == o.from && use == o.use &&
               chi_pattern == o.chi_pattern && kan_type == o.kan_type;
    }
};

struct Action {
    ActionType type = ActionType::PASS;
    int tile = -1;
    ActionInfo info;

    bool operator==(const Action& o) const {
        return type == o.type && tile == o.tile && info == o.info;
    }
};

// ============================================================
// Meld
// ============================================================

struct Meld {
    std::string type;               // "chi", "pon", "ankan", "minkan", "kakan"
    std::vector<int> tiles;

    bool operator==(const Meld& o) const {
        return type == o.type && tiles == o.tiles;
    }
};

// ============================================================
// PlayerState
// ============================================================

struct PlayerState {
    int seat = 0;
    std::vector<int> hand34;                        // 34-element count vector
    std::vector<int> river;
    std::vector<Meld> melds;
    bool riichi_declared = false;
    int riichi_turn = -1;

    PlayerState() : hand34(34, 0) {}
    explicit PlayerState(int s) : seat(s), hand34(34, 0) {}
};

// ============================================================
// StepResult
// ============================================================

using InfoValue = std::variant<int, std::string, bool, std::vector<int>,
                               std::vector<std::pair<std::string, int>>>;

struct StepResult {
    bool done = false;
    std::string reason = "continue";
    int winner = -1;
    std::vector<int> winners;
    int loser = -1;
    std::vector<int> score_delta = {0, 0, 0, 0};
    int han = 0;
    int fu = 0;
    std::vector<std::pair<std::string, int>> yaku_list;
    std::map<std::string, int> payments;
    std::map<std::string, bool> flags;
    std::map<std::string, InfoValue> info;
};

// ============================================================
// PendingDiscard (internal to engine)
// ============================================================

struct PendingDiscard {
    int player = -1;                 // discarder
    int tile = -1;
    std::vector<int> responders;
    int actor = -1;                  // current responder
    std::unordered_set<int> passes;
    std::vector<int> ronners;
    std::vector<std::pair<int, Action>> minkan_claims;
    std::vector<std::pair<int, Action>> pon_claims;
    std::pair<int, Action> chi_claim = {-1, Action{}};
    bool claim_made = false;
};

// ============================================================
// Observation (strongly-typed return for get_obs)
// ============================================================

struct Observation {
    int seat = 0;
    int cur = 0;
    int turn = 0;
    int live_wall_len = 0;
    int dead_wall_len = 0;
    std::vector<int> dora_indicators;
    Phase phase = Phase::DRAW;
    std::vector<int> hand34;
    std::vector<std::vector<int>> rivers;         // 4 players
    std::vector<std::vector<Meld>> melds;         // 4 players
    std::vector<int> scores;
    std::vector<bool> riichi_declared;
    int riichi_sticks = 0;
    int honba = 0;
    int last_discard = -1;
    int last_discarder = -1;
};

// ============================================================
// GameEvent (for event_log, replaces py::dict)
// ============================================================

struct GameEvent {
    std::string type;
    std::map<std::string, InfoValue> data;
};

}  // namespace mahjong
