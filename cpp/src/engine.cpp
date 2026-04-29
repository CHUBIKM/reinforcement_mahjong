#include "mahjong/engine.hpp"
#include "mahjong/tile_utils.hpp"
#include "mahjong/hand_analysis.hpp"
#include "mahjong/yaku.hpp"
#include "mahjong/scoring.hpp"

#include <algorithm>
#include <stdexcept>

namespace mahjong {

RiichiEngine::RiichiEngine(int seed, RuleConfig config)
    : config(std::move(config)),
      rng_(static_cast<uint32_t>(seed)),
      same_turn_furiten(4) {
    players.resize(4);
    for (int i = 0; i < 4; ++i) players[i] = PlayerState(i);
    reset(0);
}

// ============================================================
// Reset
// ============================================================

void RiichiEngine::reset(int dealer_arg) {
    auto full_wall = make_wall(rng_);

    kan_count = 0;
    dora_indicators.clear();
    event_log_.clear();
    pending_discard.reset();

    cur = dealer_arg;
    dealer = dealer_arg;
    turn = 0;
    done = false;
    last_draw = -1;
    last_discard = -1;
    last_discarder = -1;
    ippatsu_active = {false, false, false, false};
    same_turn_furiten = std::vector<std::unordered_set<int>>(4);
    first_discards = {-1, -1, -1, -1};
    open_call_happened = false;
    discard_was_called = {false, false, false, false};

    for (auto& p : players) {
        p.hand34.fill(0);
        p.river.clear();
        p.melds.clear();
        p.riichi_declared = false;
        p.riichi_turn = -1;
    }

    if (config.use_dead_wall) {
        if (config.dead_wall_size < 0 || config.dead_wall_size > static_cast<int>(full_wall.size())) {
            throw std::runtime_error("dead_wall_size out of range");
        }
        dead_wall.assign(full_wall.end() - config.dead_wall_size, full_wall.end());
        live_wall.assign(full_wall.begin(), full_wall.end() - config.dead_wall_size);
        reveal_dora_indicator(0);
    } else {
        dead_wall.clear();
        live_wall = std::move(full_wall);
    }

    for (int i = 0; i < 13; ++i) {
        for (int p = 0; p < 4; ++p) {
            int t = live_wall.back();
            live_wall.pop_back();
            hand34_add(players[p].hand34, t);
        }
    }

    seat_winds = {0, 0, 0, 0};
    for (int offset = 0; offset < 4; ++offset) {
        int seat = (dealer + offset) % 4;
        seat_winds[seat] = 27 + offset;
    }

    phase = Phase::DRAW;
    log_event({"RESET", {{"dealer", dealer}, {"round_wind", round_wind}}});
}

// ============================================================
// Logging
// ============================================================

void RiichiEngine::log_event(GameEvent evt) {
    if (logging_enabled) event_log_.push_back(std::move(evt));
}

void RiichiEngine::reveal_dora_indicator(int index) {
    if (!config.use_dead_wall) return;
    static const int positions[] = {5, 7, 9, 11};
    if (index < 0 || index >= 4) return;
    int pos_from_end = positions[index];
    if (pos_from_end > static_cast<int>(dead_wall.size())) return;
    int indicator = dead_wall[dead_wall.size() - pos_from_end];
    if (static_cast<int>(dora_indicators.size()) <= index) {
        dora_indicators.push_back(indicator);
        log_event({"DORA_REVEAL", {{"index", index}, {"indicator", indicator}}});
    }
}

int RiichiEngine::draw_from_rinshan() {
    if (!config.use_dead_wall || dead_wall.empty()) {
        throw std::runtime_error("没有王牌区，无法岭上摸牌");
    }
    int t = dead_wall.back();
    dead_wall.pop_back();
    hand34_add(players[cur].hand34, t);
    last_draw = t;
    log_event({"DRAW_RINSHAN", {{"player", cur}, {"tile", t}}});
    return t;
}

// ============================================================
// Hand helpers
// ============================================================

bool RiichiEngine::is_closed_hand(int seat) const {
    return std::all_of(players[seat].melds.begin(), players[seat].melds.end(),
                       [](const Meld& m) { return m.type == "ankan"; });
}

std::vector<int> RiichiEngine::riichi_discard_candidates(int seat) const {
    const auto& p = players[seat];
    if (p.riichi_declared || !is_closed_hand(seat)) return {};
    if (scores[seat] < 1000) return {};
    std::vector<int> cands;
    for (int t = 0; t < 34; ++t) {
        if (p.hand34[t] <= 0) continue;
        auto tmp = copy_hand(p.hand34);
        tmp[t] -= 1;
        if (is_tenpai(tmp)) cands.push_back(t);
    }
    return cands;
}

bool RiichiEngine::is_furiten(int seat, int tile) const {
    if (!config.enforce_furiten) return false;
    bool permanent = std::find(players[seat].river.begin(), players[seat].river.end(), tile) != players[seat].river.end();
    bool same_turn = config.enforce_same_turn_furiten && same_turn_furiten[seat].count(tile);
    bool riichi_furiten = config.enforce_riichi_furiten && players[seat].riichi_declared && same_turn;
    return permanent || same_turn || riichi_furiten;
}

void RiichiEngine::cancel_ippatsu() {
    ippatsu_active = {false, false, false, false};
}

bool RiichiEngine::should_abort_suucha_riichi() const {
    return config.enable_suucha_riichi && std::all_of(players.begin(), players.end(),
                                                       [](const PlayerState& p) { return p.riichi_declared; });
}

bool RiichiEngine::should_abort_suufon_renda() const {
    if (!config.enable_suufon_renda || open_call_happened) return false;
    if (std::any_of(first_discards.begin(), first_discards.end(), [](int t) { return t < 0; })) return false;
    int first = first_discards[0];
    if (first < 27 || first > 30) return false;
    return std::all_of(first_discards.begin(), first_discards.end(), [first](int t) { return t == first; });
}

bool RiichiEngine::should_abort_suukan_sanra() const {
    return config.enable_suukan_sanra && kan_count >= 4;
}

// ============================================================
// Abortive draw / exhaustive draw
// ============================================================

StepResult RiichiEngine::finalize_abortive_draw(const std::string& reason) {
    done = true;
    phase = Phase::END;
    log_event({"END", {{"reason", reason}, {"turn", turn}}});
    return StepResult{true, reason, -1, {}, -1, {0,0,0,0}, 0, 0, {}, {}, {}, {{"turn", turn}}};
}

std::pair<std::vector<int>, std::vector<int>> RiichiEngine::calc_noten_bappu_delta() {
    std::vector<int> tenpai, noten;
    for (int i = 0; i < 4; ++i) {
        if (is_tenpai(players[i].hand34)) tenpai.push_back(i);
        else noten.push_back(i);
    }
    std::vector<int> delta(4, 0);
    if (tenpai.empty() || tenpai.size() == 4) return {delta, tenpai};
    int total = 3000;
    int gain = total / static_cast<int>(tenpai.size());
    int loss = total / static_cast<int>(noten.size());
    for (int t : tenpai) delta[t] += gain;
    for (int n : noten) delta[n] -= loss;
    return {delta, tenpai};
}

StepResult RiichiEngine::resolve_exhaustive_draw() {
    if (config.enable_nagashi_mangan) {
        std::vector<int> qualifiers;
        for (int seat = 0; seat < 4; ++seat) {
            if (discard_was_called[seat]) continue;
            const auto& river = players[seat].river;
            if (river.empty()) continue;
            if (std::all_of(river.begin(), river.end(), [](int t) { return is_terminal_or_honor(t); })) {
                qualifiers.push_back(seat);
            }
        }
        if (!qualifiers.empty()) {
            int winner = qualifiers[0];
            auto pr = resolve_tsumo(winner, 5, 30, dealer, honba, riichi_sticks,
                                    config.kazoe_yakuman, config.kiriage_mangan);
            for (int i = 0; i < 4; ++i) scores[i] += pr.score_delta[i];
            done = true;
            phase = Phase::END;
            log_event({"END", {{"reason", std::string("NAGASHI_MANGAN")}, {"winner", winner}, {"turn", turn}}});
            return StepResult{true, "nagashi_mangan", winner, {winner}, -1,
                              pr.score_delta, 5, 30, {{"流局满贯", 5}}, pr.payments, {},
                              {{"turn", turn}, {"winner", winner}}};
        }
    }

    auto [delta, tenpai] = calc_noten_bappu_delta();
    for (int i = 0; i < 4; ++i) scores[i] += delta[i];
    done = true;
    phase = Phase::END;
    log_event({"END", {{"reason", std::string("RYUUKYOKU")}, {"turn", turn}}});
    std::map<std::string, int> payments;
    payments["noten_bappu_total"] = 3000;
    payments["tenpai_count"] = static_cast<int>(tenpai.size());
    return StepResult{true, "ryuukyoku", -1, {}, -1, delta, 0, 0, {}, payments, {},
                      {{"turn", turn}, {"tenpai", tenpai}}};
}

// ============================================================
// Draw / Discard
// ============================================================

int RiichiEngine::draw() {
    if (done || phase == Phase::END) throw std::runtime_error("对局已结束，不能摸牌。");
    if (phase != Phase::DRAW) throw std::runtime_error("当前阶段不能摸牌");
    if (live_wall.empty()) throw std::runtime_error("活牌山已空，不能摸牌。");

    int t = live_wall.back();
    live_wall.pop_back();
    hand34_add(players[cur].hand34, t);
    same_turn_furiten[cur].clear();
    last_draw = t;
    log_event({"DRAW", {{"player", cur}, {"tile", t}, {"live_wall", static_cast<int>(live_wall.size())}}});
    phase = Phase::DISCARD;
    return t;
}

void RiichiEngine::discard_tile(int tile) {
    if (done || phase == Phase::END) throw std::runtime_error("对局已结束，不能弃牌。");
    if (phase != Phase::DISCARD) throw std::runtime_error("当前阶段不能弃牌");
    if (tile < 0 || tile >= 34) throw std::invalid_argument("tile 必须在 0..33");
    auto& hand = players[cur].hand34;
    if (hand[tile] <= 0) throw std::invalid_argument("非法弃牌：手里没有 " + tile_to_str(tile));

    hand34_remove(hand, tile);
    players[cur].river.push_back(tile);
    if (static_cast<int>(players[cur].river.size()) == 1) first_discards[cur] = tile;

    last_discard = tile;
    last_discarder = cur;
    turn += 1;
    last_draw = -1;
    log_event({"DISCARD", {{"player", cur}, {"tile", tile}, {"turn", turn}}});

    int discarder = cur;
    std::vector<int> responders = {(discarder + 1) % 4, (discarder + 2) % 4, (discarder + 3) % 4};

    PendingDiscard pd;
    pd.player = discarder;
    pd.tile = tile;
    pd.responders = responders;
    pd.actor = responders[0];
    pending_discard = pd;
    phase = Phase::RESPONSE;
    cur = responders[0];
}

// ============================================================
// Meld application
// ============================================================

void RiichiEngine::mark_discard_called(int discarder, int tile) {
    auto& river = players[discarder].river;
    if (!river.empty() && river.back() == tile) {
        river.pop_back();
        discard_was_called[discarder] = true;
        open_call_happened = true;
    }
}

void RiichiEngine::apply_pon(int actor, int tile, int discarder) {
    mark_discard_called(discarder, tile);
    hand34_remove(players[actor].hand34, tile);
    hand34_remove(players[actor].hand34, tile);
    players[actor].melds.push_back({"pon", {tile, tile, tile}});
    cancel_ippatsu();
    log_event({"PON", {{"player", actor}, {"from", discarder}, {"tile", tile}}});
    pending_discard.reset();
    cur = actor;
    phase = Phase::DISCARD;
}

void RiichiEngine::apply_chi(int actor, int tile, int use_a, int use_b, int discarder) {
    mark_discard_called(discarder, tile);
    hand34_remove(players[actor].hand34, use_a);
    hand34_remove(players[actor].hand34, use_b);
    std::vector<int> seq = {use_a, tile, use_b};
    std::sort(seq.begin(), seq.end());
    players[actor].melds.push_back({"chi", seq});
    cancel_ippatsu();
    log_event({"CHI", {{"player", actor}, {"from", discarder}, {"tile", tile}}});
    pending_discard.reset();
    cur = actor;
    phase = Phase::DISCARD;
}

StepResult RiichiEngine::apply_minkan(int actor, int tile, int discarder) {
    mark_discard_called(discarder, tile);
    for (int i = 0; i < 3; ++i) hand34_remove(players[actor].hand34, tile);
    players[actor].melds.push_back({"minkan", {tile, tile, tile, tile}});
    kan_count += 1;
    cancel_ippatsu();
    log_event({"KAN", {{"player", actor}, {"tile", tile}, {"kan_type", std::string("MINKAN")}, {"kan_count", kan_count}}});
    if (should_abort_suukan_sanra()) {
        return finalize_abortive_draw("suukan_sanra");
    }
    reveal_dora_indicator(std::min(kan_count, 3));
    pending_discard.reset();
    cur = actor;
    phase = Phase::DISCARD;
    draw_from_rinshan();
    phase = Phase::DISCARD;
    return StepResult{false, "continue", -1, {}, -1, {0,0,0,0}, 0, 0, {}, {}, {}, {}};
}

int RiichiEngine::find_pon_meld_idx(int seat, int tile) const {
    for (int i = 0; i < static_cast<int>(players[seat].melds.size()); ++i) {
        const auto& m = players[seat].melds[i];
        if (m.type == "pon" && m.tiles.size() == 3 &&
            std::all_of(m.tiles.begin(), m.tiles.end(), [tile](int t) { return t == tile; })) {
            return i;
        }
    }
    return -1;
}

std::vector<int> RiichiEngine::collect_chankan_ronners(int kan_actor, int tile) {
    std::vector<int> winners;
    for (int offset = 1; offset <= 3; ++offset) {
        int p = (kan_actor + offset) % 4;
        auto tmp = copy_hand(players[p].hand34);
        tmp[tile] += 1;
        if (is_agari(tmp) && !is_furiten(p, tile)) {
            winners.push_back(p);
        }
    }
    return winners;
}

// ============================================================
// Yaku info
// ============================================================

std::map<std::string, InfoValue> RiichiEngine::yaku_info_for_win(
    int winner, const std::string& win_type,
    const Hand34& winning_hand34, int win_tile) {

    auto [yakus, total_han] = analyze_yaku(
        winning_hand34, win_type, seat_winds[winner], round_wind, is_closed_hand(winner));

    int dora_han = count_dora(winning_hand34, dora_indicators);
    if (dora_han > 0) {
        yakus.emplace_back("宝牌", dora_han);
        total_han += dora_han;
    }

    int fu = calculate_fu(
        players[winner].melds, winning_hand34, win_type,
        win_tile, seat_winds[winner], round_wind, is_closed_hand(winner));

    std::map<std::string, InfoValue> result;
    result["yaku"] = yakus;
    result["han"] = total_han;
    result["fu"] = fu;
    result["seat_wind"] = seat_winds[winner];
    result["round_wind"] = round_wind;
    return result;
}

// ============================================================
// Legal actions
// ============================================================

std::vector<Action> RiichiEngine::legal_actions() {
    if (done || phase == Phase::END) return {};

    std::vector<Action> actions;

    if (phase == Phase::DISCARD) {
        // Tsumo
        if (is_agari(players[cur].hand34)) {
            actions.push_back({ActionType::TSUMO});
        }

        // Kyuushu kyuuhai
        if (config.enable_kyuushu_kyuuhai &&
            players[cur].river.empty() &&
            std::all_of(players.begin(), players.end(),
                        [](const PlayerState& p) { return p.river.empty(); }) &&
            !open_call_happened &&
            count_yaochu_types(players[cur].hand34) >= 9) {
            actions.push_back({ActionType::ABORTIVE_DRAW});
        }

        // Discards
        for (int t = 0; t < 34; ++t) {
            if (players[cur].hand34[t] > 0) {
                actions.push_back({ActionType::DISCARD, t});
            }
        }

        // Riichi
        for (int t : riichi_discard_candidates(cur)) {
            actions.push_back({ActionType::RIICHI, t});
        }

        // Kan
        if (config.enable_kan && kan_count < config.max_kan_per_hand) {
            const auto& hand = players[cur].hand34;
            for (int t = 0; t < 34; ++t) {
                if (hand[t] == 4) {
                    Action a{ActionType::KAN, t};
                    a.info.kan_type = "ANKAN";
                    actions.push_back(a);
                }
            }
            for (int t = 0; t < 34; ++t) {
                if (hand[t] <= 0) continue;
                if (find_pon_meld_idx(cur, t) >= 0) {
                    Action a{ActionType::KAN, t};
                    a.info.kan_type = "KAKAN";
                    actions.push_back(a);
                }
            }
        }
        return actions;
    }

    if (phase == Phase::RESPONSE) {
        if (!pending_discard) return {{ActionType::PASS}};

        int tile = pending_discard->tile;
        int discarder = pending_discard->player;
        int actor = pending_discard->actor;

        // PASS is always legal
        actions.push_back({ActionType::PASS});

        // RON
        {
            auto tmp = copy_hand(players[actor].hand34);
            tmp[tile] += 1;
            if (is_agari(tmp) && !is_furiten(actor, tile)) {
                Action a{ActionType::RON, tile};
                a.info.from = discarder;
                actions.push_back(a);
            }
        }

        // If claim already made, only RON/PASS allowed
        if (pending_discard->claim_made) return actions;

        // PON
        if (!players[actor].riichi_declared && actor != discarder && players[actor].hand34[tile] >= 2) {
            Action a{ActionType::PON, tile};
            a.info.from = discarder;
            actions.push_back(a);
        }

        // MINKAN
        if (config.enable_kan && kan_count < config.max_kan_per_hand &&
            !players[actor].riichi_declared && actor != discarder &&
            players[actor].hand34[tile] >= 3) {
            Action a{ActionType::KAN, tile};
            a.info.from = discarder;
            a.info.kan_type = "MINKAN";
            actions.push_back(a);
        }

        // CHI
        if (!players[actor].riichi_declared && actor == (discarder + 1) % 4 && tile >= 0 && tile <= 26) {
            for (auto [a, b] : chi_options(tile)) {
                if (players[actor].hand34[a] > 0 && players[actor].hand34[b] > 0) {
                    Action act{ActionType::CHI, tile};
                    act.info.from = discarder;
                    act.info.use = {a, b};
                    actions.push_back(act);
                }
            }
        }

        return actions;
    }

    return actions;
}

// ============================================================
// Apply action
// ============================================================

StepResult RiichiEngine::apply_action(const Action& action) {
    if (done || phase == Phase::END) {
        return StepResult{true, "already_done", -1, {}, -1, {0,0,0,0}, 0, 0, {}, {}, {}, {{"turn", turn}}};
    }

    if (phase == Phase::DRAW) {
        throw std::invalid_argument("当前是 DRAW 阶段，需要先调用 draw() 摸牌");
    }

    if (phase == Phase::DISCARD) {
        if (action.type == ActionType::TSUMO) {
            if (!is_agari(players[cur].hand34)) {
                throw std::invalid_argument("当前手牌不满足和牌，不能自摸");
            }
            done = true;
            phase = Phase::END;
            auto hand = copy_hand(players[cur].hand34);
            auto yaku_info = yaku_info_for_win(cur, "tsumo", hand, last_draw);

            int han_val = std::get<int>(yaku_info["han"]);
            int fu_val = std::get<int>(yaku_info["fu"]);
            auto pr = resolve_tsumo(cur, han_val, fu_val, dealer, honba, riichi_sticks,
                                    config.kazoe_yakuman, config.kiriage_mangan);
            for (int i = 0; i < 4; ++i) scores[i] += pr.score_delta[i];

            log_event({"END", {{"reason", std::string("TSUMO")}, {"winner", cur}}});

            auto yakus = std::get<std::vector<std::pair<std::string,int>>>(yaku_info["yaku"]);
            std::map<std::string, bool> flags;
            flags["riichi"] = players[cur].riichi_declared;
            flags["ippatsu"] = ippatsu_active[cur];
            flags["haitei"] = live_wall.empty();
            flags["houtei"] = false;
            flags["chankan"] = false;
            bool is_rinshan = false;
            if (!event_log_.empty()) {
                auto& last = event_log_.back();
                if (last.type == "DRAW_RINSHAN") is_rinshan = true;
            }
            flags["rinshan"] = is_rinshan;

            std::map<std::string, InfoValue> info;
            info["turn"] = turn;
            info["win_type"] = std::string("tsumo");
            info["winner"] = cur;

            return StepResult{true, "tsumo", cur, {cur}, -1,
                              pr.score_delta, han_val, fu_val, yakus, pr.payments, flags, info};
        }

        if (action.type == ActionType::KAN) {
            if (!config.enable_kan) throw std::invalid_argument("规则未启用杠");
            if (kan_count >= config.max_kan_per_hand) throw std::invalid_argument("已达到本局最大杠次数");
            int t = action.tile;
            if (t < 0) throw std::invalid_argument("KAN 需要指定 tile");
            std::string kt = action.info.kan_type.value_or("ANKAN");

            if (kt == "KAKAN") {
                int meld_idx = find_pon_meld_idx(cur, t);
                if (meld_idx < 0 || players[cur].hand34[t] <= 0) {
                    throw std::invalid_argument("加杠需要已有碰子且手里有第四张");
                }
                auto chankan_winners = collect_chankan_ronners(cur, t);
                if (!chankan_winners.empty()) {
                    return finalize_ron(chankan_winners, cur, t, true);
                }
                hand34_remove(players[cur].hand34, t);
                players[cur].melds[meld_idx] = {"kakan", {t, t, t, t}};
                kan_count += 1;
                cancel_ippatsu();
                log_event({"KAN", {{"player", cur}, {"tile", t}, {"kan_type", std::string("KAKAN")}, {"kan_count", kan_count}}});
                if (should_abort_suukan_sanra()) return finalize_abortive_draw("suukan_sanra");
                reveal_dora_indicator(std::min(kan_count, 3));
                draw_from_rinshan();
                phase = Phase::DISCARD;
                return StepResult{false, "continue", -1, {}, -1, {0,0,0,0}, 0, 0, {}, {}, {}, {}};
            }

            // ANKAN
            if (players[cur].hand34[t] != 4) {
                throw std::invalid_argument("暗杠需要手里恰好四张同牌");
            }
            for (int i = 0; i < 4; ++i) hand34_remove(players[cur].hand34, t);
            players[cur].melds.push_back({"ankan", {t, t, t, t}});
            kan_count += 1;
            cancel_ippatsu();
            log_event({"KAN", {{"player", cur}, {"tile", t}, {"kan_type", std::string("ANKAN")}, {"kan_count", kan_count}}});
            if (should_abort_suukan_sanra()) return finalize_abortive_draw("suukan_sanra");
            reveal_dora_indicator(std::min(kan_count, 3));
            draw_from_rinshan();
            phase = Phase::DISCARD;
            return StepResult{false, "continue", -1, {}, -1, {0,0,0,0}, 0, 0, {}, {}, {}, {}};
        }

        if (action.type == ActionType::ABORTIVE_DRAW) {
            // Validate it's in legal_actions
            auto legal = legal_actions();
            bool found = false;
            for (const auto& a : legal) {
                if (a.type == ActionType::ABORTIVE_DRAW) { found = true; break; }
            }
            if (!found) throw std::invalid_argument("非法途中流局动作");
            return finalize_abortive_draw("kyuushu_kyuuhai");
        }

        if (action.type == ActionType::RIICHI) {
            int t = action.tile;
            if (t < 0) throw std::invalid_argument("RIICHI 需要指定弃牌 tile");
            auto legal = legal_actions();
            bool found = false;
            for (const auto& a : legal) {
                if (a.type == ActionType::RIICHI && a.tile == t) { found = true; break; }
            }
            if (!found) throw std::invalid_argument("非法立直动作");

            auto& p = players[cur];
            p.riichi_declared = true;
            p.riichi_turn = turn;
            ippatsu_active[cur] = true;
            riichi_sticks += 1;
            scores[cur] -= 1000;
            log_event({"RIICHI", {{"player", cur}, {"tile", t}, {"turn", turn}}});
            discard_tile(t);
            return StepResult{false, "continue", -1, {}, -1, {0,0,0,0}, 0, 0, {}, {}, {}, {}};
        }

        if (action.type == ActionType::DISCARD) {
            int t = action.tile;
            if (t < 0) throw std::invalid_argument("DISCARD 需要 tile");
            auto legal = legal_actions();
            bool found = false;
            for (const auto& a : legal) {
                if (a.type == ActionType::DISCARD && a.tile == t) { found = true; break; }
            }
            if (!found) throw std::invalid_argument("非法弃牌动作");
            discard_tile(t);
            return StepResult{false, "continue", -1, {}, -1, {0,0,0,0}, 0, 0, {}, {}, {}, {}};
        }

        throw std::invalid_argument("当前阶段不支持该动作");
    }

    if (phase == Phase::RESPONSE) {
        if (!pending_discard) throw std::runtime_error("RESPONSE 阶段但没有 pending_discard");

        int actor = pending_discard->actor;
        if (cur != actor) throw std::invalid_argument("当前不是该玩家的响应回合");

        auto legal_now = legal_actions();
        bool had_ron_option = false;
        for (const auto& a : legal_now) {
            if (a.type == ActionType::RON) { had_ron_option = true; break; }
        }

        // Validate action is legal
        bool is_legal = false;
        for (const auto& a : legal_now) {
            if (a == action) { is_legal = true; break; }
        }
        if (!is_legal) throw std::invalid_argument("非法响应动作");

        if (action.type == ActionType::RON) {
            pending_discard->ronners.push_back(actor);
            log_event({"RON_DECLARE", {{"winner", actor}, {"from", pending_discard->player}, {"tile", pending_discard->tile}}});
            if (!config.allow_multi_ron) {
                return finalize_ron({actor});
            }
        } else if (action.type == ActionType::PON) {
            pending_discard->pon_claims.emplace_back(actor, action);
            pending_discard->claim_made = true;
            log_event({"PON_DECLARE", {{"player", actor}, {"from", pending_discard->player}, {"tile", pending_discard->tile}}});
        } else if (action.type == ActionType::KAN) {
            pending_discard->minkan_claims.emplace_back(actor, action);
            pending_discard->claim_made = true;
            log_event({"MINKAN_DECLARE", {{"player", actor}, {"from", pending_discard->player}, {"tile", pending_discard->tile}}});
        } else if (action.type == ActionType::CHI) {
            pending_discard->chi_claim = {actor, action};
            pending_discard->claim_made = true;
            log_event({"CHI_DECLARE", {{"player", actor}, {"from", pending_discard->player}, {"tile", pending_discard->tile}}});
        } else if (action.type == ActionType::PASS) {
            pending_discard->passes.insert(actor);
            if (had_ron_option) {
                same_turn_furiten[actor].insert(pending_discard->tile);
            }
            log_event({"PASS", {{"player", actor}}});
        } else {
            throw std::invalid_argument("RESPONSE 阶段不支持该动作");
        }

        const auto& responders = pending_discard->responders;
        int idx = -1;
        for (int i = 0; i < static_cast<int>(responders.size()); ++i) {
            if (responders[i] == actor) { idx = i; break; }
        }
        if (idx + 1 < static_cast<int>(responders.size())) {
            int next_actor = responders[idx + 1];
            pending_discard->actor = next_actor;
            cur = next_actor;
            return StepResult{false, "continue", -1, {}, -1, {0,0,0,0}, 0, 0, {}, {}, {}, {}};
        }

        // All responders have acted: resolve priority
        int discarder = pending_discard->player;
        int tile = pending_discard->tile;
        auto ronners = pending_discard->ronners;

        // 1) RON wins
        if (!ronners.empty()) {
            return finalize_ron(ronners);
        }

        // 2) Minkan/Pon claims
        const auto& minkan_claims = pending_discard->minkan_claims;
        const auto& pon_claims = pending_discard->pon_claims;
        if (!minkan_claims.empty() || !pon_claims.empty()) {
            std::tuple<std::string, int, Action> chosen;
            bool found = false;
            for (int r : responders) {
                for (const auto& [p, a] : minkan_claims) {
                    if (p == r) { chosen = {"minkan", p, a}; found = true; break; }
                }
                if (found) break;
                for (const auto& [p, a] : pon_claims) {
                    if (p == r) { chosen = {"pon", p, a}; found = true; break; }
                }
                if (found) break;
            }
            if (!found && !minkan_claims.empty()) {
                chosen = {"minkan", minkan_claims[0].first, minkan_claims[0].second};
                found = true;
            }
            if (!found && !pon_claims.empty()) {
                chosen = {"pon", pon_claims[0].first, pon_claims[0].second};
                found = true;
            }

            if (found) {
                const auto& [ctype, chosen_player, _] = chosen;
                if (ctype == "minkan") {
                    return apply_minkan(chosen_player, tile, discarder);
                }
                apply_pon(chosen_player, tile, discarder);
                return StepResult{false, "continue", -1, {}, -1, {0,0,0,0}, 0, 0, {}, {}, {}, {}};
            }
        }

        // 3) Chi claim
        if (pending_discard->chi_claim.first >= 0) {
            auto [chi_player, chi_action] = pending_discard->chi_claim;
            const auto& use = chi_action.info.use;
            if (use.size() != 2) throw std::runtime_error("CHI 声明缺少 use 信息");
            apply_chi(chi_player, tile, use[0], use[1], discarder);
            return StepResult{false, "continue", -1, {}, -1, {0,0,0,0}, 0, 0, {}, {}, {}, {}};
        }

        // 4) No claims: next player draws
        pending_discard.reset();
        int next_player = (discarder + 1) % 4;
        cur = next_player;
        phase = Phase::DRAW;

        if (should_abort_suufon_renda()) {
            return finalize_abortive_draw("suufon_renda");
        }
        if (should_abort_suucha_riichi()) {
            return finalize_abortive_draw("suucha_riichi");
        }
        if (live_wall.empty()) {
            return resolve_exhaustive_draw();
        }

        return StepResult{false, "continue", -1, {}, -1, {0,0,0,0}, 0, 0, {}, {}, {}, {}};
    }

    throw std::invalid_argument("未处理的阶段");
}

// ============================================================
// Finalize ron
// ============================================================

StepResult RiichiEngine::finalize_ron(const std::vector<int>& winners,
                                       int discarder, int tile, bool chankan) {
    if (discarder < 0 && pending_discard) discarder = pending_discard->player;
    if (tile < 0 && pending_discard) tile = pending_discard->tile;
    if (discarder < 0 || tile < 0) throw std::runtime_error("无法结算荣和：缺少信息");

    done = true;
    phase = Phase::END;
    pending_discard.reset();

    std::vector<int> aggregate_delta(4, 0);
    std::map<std::string, int> payments;
    int han = 0, fu = 0;
    std::vector<std::pair<std::string, int>> yaku_list;

    for (int w : winners) {
        auto tmp = copy_hand(players[w].hand34);
        tmp[tile] += 1;
        auto y = yaku_info_for_win(w, "ron", tmp, tile);
        int w_han = std::get<int>(y["han"]);
        int w_fu = std::get<int>(y["fu"]);
        auto w_yakus = std::get<std::vector<std::pair<std::string,int>>>(y["yaku"]);

        auto pr = resolve_ron(w, discarder, w_han, w_fu, dealer, honba,
                              w == winners[0] ? riichi_sticks : 0,
                              config.kazoe_yakuman, config.kiriage_mangan);
        for (int i = 0; i < 4; ++i) aggregate_delta[i] += pr.score_delta[i];
        if (w == winners[0]) {
            payments = pr.payments;
            han = w_han;
            fu = w_fu;
            yaku_list = w_yakus;
        }
    }

    for (int i = 0; i < 4; ++i) scores[i] += aggregate_delta[i];

    log_event({"END", {{"reason", std::string("RON")}, {"discarder", discarder}, {"tile", tile}}});

    std::map<std::string, bool> flags;
    flags["riichi"] = players[winners[0]].riichi_declared;
    flags["ippatsu"] = ippatsu_active[winners[0]];
    flags["rinshan"] = false;
    flags["haitei"] = false;
    flags["houtei"] = live_wall.empty();
    flags["chankan"] = chankan;

    std::map<std::string, InfoValue> info;
    info["turn"] = turn;
    info["win_type"] = std::string("ron");
    info["chankan"] = chankan;

    return StepResult{true, "ron", winners[0], winners, discarder,
                      aggregate_delta, han, fu, yaku_list, payments, flags, info};
}

// ============================================================
// Step (legacy)
// ============================================================

StepResult RiichiEngine::step(int discard_tile) {
    if (phase == Phase::DRAW) draw();
    if (phase != Phase::DISCARD) throw std::runtime_error("step() 只能在 DISCARD 阶段调用");
    return apply_action({ActionType::DISCARD, discard_tile});
}

// ============================================================
// Observation
// ============================================================

Observation RiichiEngine::get_obs(int seat) const {
    if (seat < 0) seat = cur;

    Observation obs;
    obs.seat = seat;
    obs.cur = cur;
    obs.turn = turn;
    obs.live_wall_len = static_cast<int>(live_wall.size());
    obs.dead_wall_len = static_cast<int>(dead_wall.size());
    obs.dora_indicators = dora_indicators;
    obs.phase = phase;
    obs.hand34 = copy_hand(players[seat].hand34);
    for (int i = 0; i < 4; ++i) {
        obs.rivers.push_back(players[i].river);
        obs.melds.push_back(players[i].melds);
    }
    obs.scores = scores;
    for (int i = 0; i < 4; ++i) obs.riichi_declared.push_back(players[i].riichi_declared);
    obs.riichi_sticks = riichi_sticks;
    obs.honba = honba;
    obs.last_discard = last_discard;
    obs.last_discarder = last_discarder;
    return obs;
}

int RiichiEngine::get_obs_array(int seat, float* buf) const {
    if (seat < 0) seat = cur;
    float* p = buf;

    // hand34 (34)
    for (int t = 0; t < 34; ++t) *p++ = static_cast<float>(players[seat].hand34[t]);

    // river histogram (34)
    Hand34 river_hist = {};
    for (int i = 0; i < 4; ++i)
        for (int t : players[i].river) river_hist[t] += 1;
    for (int t = 0; t < 34; ++t) *p++ = static_cast<float>(river_hist[t]);

    // meld histogram (34)
    Hand34 meld_hist = {};
    for (int i = 0; i < 4; ++i)
        for (const auto& m : players[i].melds)
            for (int t : m.tiles) meld_hist[t] += 1;
    for (int t = 0; t < 34; ++t) *p++ = static_cast<float>(meld_hist[t]);

    // phase one-hot (4): DRAW/DISCARD/RESPONSE/END
    *p++ = (phase == Phase::DRAW) ? 1.0f : 0.0f;
    *p++ = (phase == Phase::DISCARD) ? 1.0f : 0.0f;
    *p++ = (phase == Phase::RESPONSE) ? 1.0f : 0.0f;
    *p++ = (phase == Phase::END) ? 1.0f : 0.0f;

    // seat one-hot (4)
    for (int i = 0; i < 4; ++i) *p++ = (i == seat) ? 1.0f : 0.0f;

    // scalars (8)
    *p++ = static_cast<float>(cur);
    *p++ = static_cast<float>(turn);
    *p++ = static_cast<float>(live_wall.size());
    *p++ = static_cast<float>(dead_wall.size());
    *p++ = static_cast<float>(last_discard);
    *p++ = static_cast<float>(last_discarder);
    *p++ = static_cast<float>(riichi_sticks);
    *p++ = static_cast<float>(honba);

    // dora indicators padded (4)
    for (int i = 0; i < 4; ++i)
        *p++ = (i < static_cast<int>(dora_indicators.size())) ? static_cast<float>(dora_indicators[i]) : -1.0f;

    // scores normalized (4)
    for (int i = 0; i < 4; ++i) *p++ = static_cast<float>(scores[i]) / 10000.0f;

    // riichi_declared (4)
    for (int i = 0; i < 4; ++i) *p++ = players[i].riichi_declared ? 1.0f : 0.0f;

    return static_cast<int>(p - buf);
}

// ============================================================
// Validate
// ============================================================

void RiichiEngine::validate_invariants() const {
    std::vector<int> counts(34, 0);
    for (int t : live_wall) counts[t] += 1;
    for (int t : dead_wall) counts[t] += 1;
    for (const auto& p : players) {
        for (int t = 0; t < 34; ++t) counts[t] += p.hand34[t];
        for (const auto& m : p.melds) {
            for (int t : m.tiles) counts[t] += 1;
        }
        for (int t : p.river) counts[t] += 1;
    }
    for (int t = 0; t < 34; ++t) {
        if (counts[t] != 4) {
            throw std::runtime_error("tile " + std::to_string(t) + " count invariant broken: " + std::to_string(counts[t]));
        }
    }
    if (done && phase != Phase::END) {
        throw std::runtime_error("done game must be in END phase");
    }
}

// ============================================================
// Export replay
// ============================================================

std::vector<GameEvent> RiichiEngine::export_replay() const {
    return event_log_;
}

// ============================================================
// Play random
// ============================================================

StepResult RiichiEngine::play_random(int max_steps, bool verbose) {
    int steps = 0;
    while (steps < max_steps && !done) {
        steps += 1;

        if (phase == Phase::DRAW) {
            if (live_wall.empty()) return resolve_exhaustive_draw();
            int t = draw();
            if (verbose) {
                // Could print here but C++ stdout in RL loop is rarely wanted
                (void)t;
            }
        } else if (phase == Phase::DISCARD || phase == Phase::RESPONSE) {
            auto acts = legal_actions();
            if (acts.empty()) throw std::runtime_error("阶段没有合法动作，状态机卡死");

            std::uniform_int_distribution<int> dist(0, static_cast<int>(acts.size()) - 1);
            const Action& a = acts[dist(rng_)];
            auto res = apply_action(a);
            if (res.done) return res;
        } else {
            break;
        }
    }

    return StepResult{done, "max_steps", -1, {}, -1, {0,0,0,0}, 0, 0, {}, {}, {},
                      {{"turn", turn}, {"phase", phase_name(phase)}}};
}

}  // namespace mahjong
