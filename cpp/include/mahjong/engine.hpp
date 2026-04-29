#pragma once

#include <random>
#include <unordered_set>
#include <vector>

#include "mahjong/types.hpp"
#include "mahjong/rule_config.hpp"
#include "mahjong/hand_analysis.hpp"

namespace mahjong {

class RiichiEngine {
public:
    explicit RiichiEngine(int seed = 0, RuleConfig config = RuleConfig{});

    RuleConfig config;
    std::vector<PlayerState> players;
    std::vector<int> live_wall;
    std::vector<int> dead_wall;
    int kan_count = 0;
    std::vector<int> dora_indicators;

    int cur = 0;
    int turn = 0;
    bool done = false;
    int dealer = 0;
    Phase phase = Phase::DRAW;

    int last_draw = -1;
    int last_discard = -1;
    int last_discarder = -1;

    int round_wind = 27;
    std::vector<int> seat_winds = {27, 28, 29, 30};
    std::vector<int> scores = {25000, 25000, 25000, 25000};
    int honba = 0;
    int riichi_sticks = 0;
    std::vector<bool> ippatsu_active = {false, false, false, false};
    std::vector<std::unordered_set<int>> same_turn_furiten;
    std::vector<int> first_discards = {-1, -1, -1, -1};
    bool open_call_happened = false;
    std::vector<bool> discard_was_called = {false, false, false, false};

    // Public for RL adapter access (materialize_action reads pending_discard)
    std::optional<PendingDiscard> pending_discard;

    // Logging toggle — set false during training to skip event_log overhead
    bool logging_enabled = true;

    // Core API
    void reset(int dealer = 0);
    int draw();
    void discard_tile(int tile);
    std::vector<Action> legal_actions();
    StepResult apply_action(const Action& action);
    StepResult step(int discard_tile);
    Observation get_obs(int seat = -1) const;

    // Zero-copy observation: writes directly into a pre-allocated float buffer.
    // Buffer must have at least OBS_DIM floats. Returns number of floats written.
    int get_obs_array(int seat, float* buf) const;

    void validate_invariants() const;
    std::vector<GameEvent> export_replay() const;
    StepResult play_random(int max_steps = 20000, bool verbose = false);

    // Internal helpers (public for testing)
    bool should_abort_suufon_renda() const;
    std::map<std::string, InfoValue> yaku_info_for_win(int winner, const std::string& win_type,
                                                        const Hand34& winning_hand34, int win_tile);

    // Observation dimension for RL
    static constexpr int OBS_DIM = 34 + 34 + 34 + 4 + 4 + 8 + 4 + 4 + 4;

private:
    std::mt19937 rng_;
    std::vector<GameEvent> event_log_;

    void log_event(GameEvent evt);
    void reveal_dora_indicator(int index);
    int draw_from_rinshan();
    bool is_closed_hand(int seat) const;
    std::vector<int> riichi_discard_candidates(int seat) const;
    bool is_furiten(int seat, int tile) const;
    void cancel_ippatsu();
    bool should_abort_suucha_riichi() const;
    bool should_abort_suukan_sanra() const;

    StepResult finalize_abortive_draw(const std::string& reason);
    std::pair<std::vector<int>, std::vector<int>> calc_noten_bappu_delta();
    StepResult resolve_exhaustive_draw();

    void apply_pon(int actor, int tile, int discarder);
    void apply_chi(int actor, int tile, int use_a, int use_b, int discarder);
    StepResult apply_minkan(int actor, int tile, int discarder);
    void mark_discard_called(int discarder, int tile);
    int find_pon_meld_idx(int seat, int tile) const;
    std::vector<int> collect_chankan_ronners(int kan_actor, int tile);

    StepResult finalize_ron(const std::vector<int>& winners,
                            int discarder = -1, int tile = -1, bool chankan = false);
};

}  // namespace mahjong
