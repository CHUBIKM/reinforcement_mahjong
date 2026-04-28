#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "mahjong/engine.hpp"
#include "mahjong/tile_utils.hpp"
#include "mahjong/hand_analysis.hpp"
#include "mahjong/yaku.hpp"
#include "mahjong/scoring.hpp"

#include <map>
#include <string>
#include <variant>
#include <vector>

namespace py = pybind11;

using namespace mahjong;

// ============================================================
// RuleConfig from Python RuleProfile
// ============================================================

static RuleConfig parse_rule_config(const py::object& obj) {
    RuleConfig cfg;
    if (obj.is_none()) return cfg;

    auto read_str = [&](const char* key, std::string def) {
        if (py::hasattr(obj, key)) return py::cast<std::string>(obj.attr(key));
        return def;
    };
    auto read_bool = [&](const char* key, bool def) {
        if (py::hasattr(obj, key)) return py::cast<bool>(obj.attr(key));
        return def;
    };
    auto read_int = [&](const char* key, int def) {
        if (py::hasattr(obj, key)) return py::cast<int>(obj.attr(key));
        return def;
    };

    cfg.name = read_str("name", cfg.name);
    cfg.allow_multi_ron = read_bool("allow_multi_ron", cfg.allow_multi_ron);
    cfg.use_dead_wall = read_bool("use_dead_wall", cfg.use_dead_wall);
    cfg.dead_wall_size = read_int("dead_wall_size", cfg.dead_wall_size);
    cfg.enable_kan = read_bool("enable_kan", cfg.enable_kan);
    cfg.max_kan_per_hand = read_int("max_kan_per_hand", cfg.max_kan_per_hand);

    cfg.enable_nagashi_mangan = read_bool("enable_nagashi_mangan", cfg.enable_nagashi_mangan);
    cfg.enable_kyuushu_kyuuhai = read_bool("enable_kyuushu_kyuuhai", cfg.enable_kyuushu_kyuuhai);
    cfg.enable_suufon_renda = read_bool("enable_suufon_renda", cfg.enable_suufon_renda);
    cfg.enable_suucha_riichi = read_bool("enable_suucha_riichi", cfg.enable_suucha_riichi);
    cfg.enable_suukan_sanra = read_bool("enable_suukan_sanra", cfg.enable_suukan_sanra);

    cfg.aka_dora_count = read_int("aka_dora_count", cfg.aka_dora_count);
    cfg.ura_dora_enabled = read_bool("ura_dora_enabled", cfg.ura_dora_enabled);
    cfg.kazoe_yakuman = read_bool("kazoe_yakuman", cfg.kazoe_yakuman);
    cfg.kiriage_mangan = read_bool("kiriage_mangan", cfg.kiriage_mangan);

    cfg.enforce_furiten = read_bool("enforce_furiten", cfg.enforce_furiten);
    cfg.enforce_same_turn_furiten = read_bool("enforce_same_turn_furiten", cfg.enforce_same_turn_furiten);
    cfg.enforce_riichi_furiten = read_bool("enforce_riichi_furiten", cfg.enforce_riichi_furiten);
    return cfg;
}

// ============================================================
// Helpers: convert C++ types to Python
// ============================================================

static py::list melds_to_py(const std::vector<Meld>& melds) {
    py::list out;
    for (const auto& m : melds) {
        out.append(py::make_tuple(m.type, py::cast(m.tiles)));
    }
    return out;
}

static std::vector<Meld> py_to_melds(const py::object& obj) {
    std::vector<Meld> out;
    auto seq = py::cast<py::list>(obj);
    for (auto item : seq) {
        auto t = py::cast<py::tuple>(item);
        auto mtype = py::cast<std::string>(t[0]);
        auto tiles = py::cast<std::vector<int>>(t[1]);
        out.push_back({mtype, tiles});
    }
    return out;
}

static py::object info_value_to_py(const InfoValue& val) {
    return std::visit([](const auto& v) -> py::object {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, int>) {
            return py::cast(v);
        } else if constexpr (std::is_same_v<T, std::string>) {
            return py::cast(v);
        } else if constexpr (std::is_same_v<T, bool>) {
            return py::cast(v);
        } else if constexpr (std::is_same_v<T, std::vector<int>>) {
            return py::cast(v);
        } else if constexpr (std::is_same_v<T, std::vector<std::pair<std::string, int>>>) {
            py::list l;
            for (const auto& [name, han] : v) {
                l.append(py::make_tuple(name, han));
            }
            return l;
        }
    }, val);
}

static py::dict obs_to_py(const Observation& obs) {
    py::dict d;
    d["seat"] = obs.seat;
    d["cur"] = obs.cur;
    d["turn"] = obs.turn;
    d["live_wall_len"] = obs.live_wall_len;
    d["dead_wall_len"] = obs.dead_wall_len;
    d["dora_indicators"] = obs.dora_indicators;
    d["phase"] = phase_name(obs.phase);
    d["hand34"] = obs.hand34;
    py::list rivers;
    for (const auto& r : obs.rivers) rivers.append(py::cast(r));
    d["rivers"] = rivers;
    py::list melds_list;
    for (const auto& ms : obs.melds) melds_list.append(melds_to_py(ms));
    d["melds"] = melds_list;
    d["scores"] = obs.scores;
    d["riichi_declared"] = obs.riichi_declared;
    d["riichi_sticks"] = obs.riichi_sticks;
    d["honba"] = obs.honba;
    d["last_discard"] = obs.last_discard;
    d["last_discarder"] = obs.last_discarder;
    return d;
}

static py::dict step_result_info_to_py(const std::map<std::string, InfoValue>& info) {
    py::dict d;
    for (const auto& [key, val] : info) {
        d[key.c_str()] = info_value_to_py(val);
    }
    return d;
}

static py::dict pending_discard_to_py(const std::optional<PendingDiscard>& pd) {
    if (!pd) return py::dict();
    py::dict d;
    d["player"] = pd->player;
    d["tile"] = pd->tile;
    d["actor"] = pd->actor;
    d["claim_made"] = pd->claim_made;

    py::list ronners;
    for (int r : pd->ronners) ronners.append(r);
    d["ronners"] = ronners;

    py::list responders;
    for (int r : pd->responders) responders.append(r);
    d["responders"] = responders;

    py::list pon_claims;
    for (const auto& [seat, act] : pd->pon_claims) {
        py::dict claim;
        claim["player"] = seat;
        claim["action"] = act;
        pon_claims.append(claim);
    }
    d["pon_claims"] = pon_claims;

    py::list minkan_claims;
    for (const auto& [seat, act] : pd->minkan_claims) {
        py::dict claim;
        claim["player"] = seat;
        claim["action"] = act;
        minkan_claims.append(claim);
    }
    d["minkan_claims"] = minkan_claims;

    if (pd->chi_claim.first >= 0) {
        py::dict claim;
        claim["player"] = pd->chi_claim.first;
        claim["action"] = pd->chi_claim.second;
        d["chi_claim"] = claim;
    }

    py::list passes;
    for (int p : pd->passes) passes.append(p);
    d["passes"] = passes;

    return d;
}

static py::list events_to_py(const std::vector<GameEvent>& events) {
    py::list out;
    for (const auto& evt : events) {
        py::dict d;
        d["type"] = evt.type;
        for (const auto& [key, val] : evt.data) {
            d[key.c_str()] = info_value_to_py(val);
        }
        out.append(d);
    }
    return out;
}

// ============================================================
// Action Python constructor helper
// ============================================================

static Action make_action(ActionType type, py::object tile_obj, py::object info_obj) {
    Action a;
    a.type = type;
    if (!tile_obj.is_none()) a.tile = py::cast<int>(tile_obj);

    if (!info_obj.is_none()) {
        py::dict info = py::cast<py::dict>(info_obj);
        if (info.contains("from")) a.info.from = py::cast<int>(info["from"]);
        if (info.contains("use")) a.info.use = py::cast<std::vector<int>>(info["use"]);
        if (info.contains("chi_pattern")) a.info.chi_pattern = py::cast<int>(info["chi_pattern"]);
        if (info.contains("kan_type")) a.info.kan_type = py::cast<std::string>(info["kan_type"]);
    }
    return a;
}

static py::dict action_info_to_py(const Action& a) {
    py::dict info;
    if (a.info.from.has_value()) info["from"] = a.info.from.value();
    if (!a.info.use.empty()) info["use"] = a.info.use;
    if (a.info.chi_pattern.has_value()) info["chi_pattern"] = a.info.chi_pattern.value();
    if (a.info.kan_type.has_value()) info["kan_type"] = a.info.kan_type.value();
    return info;
}

// ============================================================
// Hand34 conversion helpers
// ============================================================

static Hand34 vec_to_hand34(const std::vector<int>& v) {
    Hand34 h = {};
    for (size_t i = 0; i < std::min(v.size(), size_t(34)); ++i) h[i] = v[i];
    return h;
}

// ============================================================
// PYBIND11_MODULE
// ============================================================

PYBIND11_MODULE(_mahjong_cpp, m) {
    m.doc() = "C++ Riichi Mahjong engine with pybind11 bindings";

    // Constants
    m.attr("OBS_DIM") = RiichiEngine::OBS_DIM;

    // Phase enum
    py::enum_<Phase>(m, "Phase")
        .value("DRAW", Phase::DRAW)
        .value("DISCARD", Phase::DISCARD)
        .value("RESPONSE", Phase::RESPONSE)
        .value("END", Phase::END)
        .def("__repr__", [](Phase p) { return phase_name(p); });

    // ActionType enum
    py::enum_<ActionType>(m, "ActionType")
        .value("DISCARD", ActionType::DISCARD)
        .value("TSUMO", ActionType::TSUMO)
        .value("RON", ActionType::RON)
        .value("PASS", ActionType::PASS)
        .value("CHI", ActionType::CHI)
        .value("PON", ActionType::PON)
        .value("KAN", ActionType::KAN)
        .value("RIICHI", ActionType::RIICHI)
        .value("ABORTIVE_DRAW", ActionType::ABORTIVE_DRAW)
        .def("__repr__", [](ActionType t) { return action_type_name(t); });

    // Action
    py::class_<Action>(m, "Action")
        .def(py::init([](ActionType type, py::object tile, py::object info) {
            return make_action(type, tile, info);
        }), py::arg("type"), py::arg("tile") = py::none(), py::arg("info") = py::dict())
        .def_readonly("type", [](const Action& a) -> ActionType { return a.type; })
        .def_property_readonly("tile", [](const Action& a) -> py::object {
            return a.tile >= 0 ? py::cast(a.tile) : py::none();
        })
        .def_property_readonly("info", &action_info_to_py)
        .def("__eq__", [](const Action& a, const Action& b) { return a == b; })
        .def("__hash__", [](const Action& a) {
            size_t h = static_cast<size_t>(a.type) * 31 + static_cast<size_t>(a.tile);
            return h;
        })
        .def("__repr__", [](const Action& a) {
            std::string s = action_type_name(a.type);
            if (a.tile >= 0) s += "(" + tile_to_str(a.tile) + ")";
            return s;
        });

    // Meld
    py::class_<Meld>(m, "Meld")
        .def_readonly("type", &Meld::type)
        .def_readonly("tiles", &Meld::tiles);

    // PlayerState
    py::class_<PlayerState>(m, "PlayerState")
        .def_readonly("seat", &PlayerState::seat)
        .def_property("hand34",
                      [](const PlayerState& p) -> py::list { py::list l; for (int c : p.hand34) l.append(c); return l; },
                      [](PlayerState& p, py::list v) { for (int i = 0; i < 34 && i < py::len(v); ++i) p.hand34[i] = py::cast<int>(v[i]); })
        .def_property("river",
                      [](const PlayerState& p) -> py::list { py::list l; for (int t : p.river) l.append(t); return l; },
                      [](PlayerState& p, py::list v) { p.river.clear(); for (auto x : v) p.river.push_back(py::cast<int>(x)); })
        .def_property("melds",
                      [](const PlayerState& p) -> py::list { return melds_to_py(p.melds); },
                      [](PlayerState& p, py::list v) { p.melds = py_to_melds(v); })
        .def_readonly("riichi_declared", &PlayerState::riichi_declared)
        .def_readonly("riichi_turn", &PlayerState::riichi_turn);

    // StepResult
    py::class_<StepResult>(m, "StepResult")
        .def_readonly("done", &StepResult::done)
        .def_readonly("reason", &StepResult::reason)
        .def_property_readonly("winner", [](const StepResult& r) -> py::object {
            return r.winner >= 0 ? py::cast(r.winner) : py::none();
        })
        .def_readonly("winners", &StepResult::winners)
        .def_property_readonly("loser", [](const StepResult& r) -> py::object {
            return r.loser >= 0 ? py::cast(r.loser) : py::none();
        })
        .def_readonly("score_delta", &StepResult::score_delta)
        .def_readonly("han", &StepResult::han)
        .def_readonly("fu", &StepResult::fu)
        .def_property_readonly("yaku_list", [](const StepResult& r) -> py::list {
            py::list l;
            for (const auto& [name, han] : r.yaku_list) l.append(py::make_tuple(name, han));
            return l;
        })
        .def_readonly("payments", &StepResult::payments)
        .def_readonly("flags", &StepResult::flags)
        .def_property_readonly("info", [](const StepResult& r) -> py::dict {
            return step_result_info_to_py(r.info);
        })
        .def("__repr__", [](const StepResult& r) {
            return "<StepResult reason=" + r.reason + " done=" + (r.done ? "True" : "False") + ">";
        });

    // RuleConfig
    py::class_<RuleConfig>(m, "RuleConfig")
        .def(py::init<>())
        .def_readonly("name", &RuleConfig::name)
        .def_readonly("allow_multi_ron", &RuleConfig::allow_multi_ron)
        .def_readonly("use_dead_wall", &RuleConfig::use_dead_wall)
        .def_readonly("dead_wall_size", &RuleConfig::dead_wall_size)
        .def_readonly("enable_kan", &RuleConfig::enable_kan)
        .def_readonly("max_kan_per_hand", &RuleConfig::max_kan_per_hand)
        .def_readonly("enable_nagashi_mangan", &RuleConfig::enable_nagashi_mangan)
        .def_readonly("enable_kyuushu_kyuuhai", &RuleConfig::enable_kyuushu_kyuuhai)
        .def_readonly("enable_suufon_renda", &RuleConfig::enable_suufon_renda)
        .def_readonly("enable_suucha_riichi", &RuleConfig::enable_suucha_riichi)
        .def_readonly("enable_suukan_sanra", &RuleConfig::enable_suukan_sanra)
        .def_readonly("aka_dora_count", &RuleConfig::aka_dora_count)
        .def_readonly("ura_dora_enabled", &RuleConfig::ura_dora_enabled)
        .def_readonly("kazoe_yakuman", &RuleConfig::kazoe_yakuman)
        .def_readonly("kiriage_mangan", &RuleConfig::kiriage_mangan)
        .def_readonly("enforce_furiten", &RuleConfig::enforce_furiten)
        .def_readonly("enforce_same_turn_furiten", &RuleConfig::enforce_same_turn_furiten)
        .def_readonly("enforce_riichi_furiten", &RuleConfig::enforce_riichi_furiten);

    // RiichiEngine
    py::class_<RiichiEngine>(m, "RiichiEngine")
        .def(py::init([](int seed, py::object config_obj) {
            RuleConfig cfg = parse_rule_config(config_obj);
            return new RiichiEngine(seed, cfg);
        }), py::arg("seed") = 0, py::arg("config") = py::none())
        .def("reset", &RiichiEngine::reset, py::arg("dealer") = 0)
        .def("draw", &RiichiEngine::draw)
        .def("legal_discard_mask", [](RiichiEngine& e) -> py::list {
            py::list l;
            const auto& hand = e.players[e.cur].hand34;
            for (int t = 0; t < 34; ++t) l.append(hand[t] > 0 ? 1 : 0);
            return l;
        })
        .def("legal_discards", [](RiichiEngine& e) -> std::vector<int> {
            std::vector<int> out;
            const auto& hand = e.players[e.cur].hand34;
            for (int t = 0; t < 34; ++t) {
                if (hand[t] > 0) out.push_back(t);
            }
            return out;
        })
        .def("legal_actions", &RiichiEngine::legal_actions)
        .def("discard", &RiichiEngine::discard_tile)
        .def("apply_action", &RiichiEngine::apply_action)
        .def("step", &RiichiEngine::step)
        .def("get_obs", [](RiichiEngine& e, py::object seat_obj) -> py::dict {
            int seat = seat_obj.is_none() ? -1 : py::cast<int>(seat_obj);
            return obs_to_py(e.get_obs(seat));
        }, py::arg("seat") = py::none())
        .def("get_obs_array", [](RiichiEngine& e, int seat) -> py::array_t<float> {
            auto buf = py::array_t<float>(RiichiEngine::OBS_DIM);
            auto info = buf.request();
            e.get_obs_array(seat, static_cast<float*>(info.ptr));
            return buf;
        }, py::arg("seat"))
        .def("validate_invariants", &RiichiEngine::validate_invariants)
        .def("export_replay", [](RiichiEngine& e) -> py::list {
            return events_to_py(e.export_replay());
        })
        .def("play_random", [](RiichiEngine& e, int max_steps, bool verbose) -> StepResult {
            return e.play_random(max_steps, verbose);
        }, py::arg("max_steps") = 20000, py::arg("verbose") = false)
        .def("_should_abort_suufon_renda", &RiichiEngine::should_abort_suufon_renda)
        .def("_yaku_info_for_win", [](RiichiEngine& e, int winner, const std::string& win_type,
                                       const std::vector<int>& hand34_vec, py::object win_tile_obj) -> py::dict {
            int win_tile = win_tile_obj.is_none() ? -1 : py::cast<int>(win_tile_obj);
            Hand34 h34 = vec_to_hand34(hand34_vec);
            auto info = e.yaku_info_for_win(winner, win_type, h34, win_tile);
            py::dict d;
            for (const auto& [key, val] : info) {
                d[key.c_str()] = info_value_to_py(val);
            }
            return d;
        })
        .def_readonly("config", &RiichiEngine::config)
        .def_readonly("players", &RiichiEngine::players)
        .def_readonly("live_wall", &RiichiEngine::live_wall)
        .def_readonly("dead_wall", &RiichiEngine::dead_wall)
        .def_readonly("kan_count", &RiichiEngine::kan_count)
        .def_readonly("dora_indicators", &RiichiEngine::dora_indicators)
        .def_readonly("cur", &RiichiEngine::cur)
        .def_readonly("turn", &RiichiEngine::turn)
        .def_readonly("done", &RiichiEngine::done)
        .def_readonly("dealer", &RiichiEngine::dealer)
        .def_property_readonly("phase", [](const RiichiEngine& e) -> std::string {
            return phase_name(e.phase);
        })
        .def_property_readonly("pending_discard", [](const RiichiEngine& e) -> py::dict {
            return pending_discard_to_py(e.pending_discard);
        })
        .def_readwrite("logging_enabled", &RiichiEngine::logging_enabled)
        .def_readonly("scores", &RiichiEngine::scores)
        .def_readonly("honba", &RiichiEngine::honba)
        .def_readonly("riichi_sticks", &RiichiEngine::riichi_sticks);

    // Free functions
    m.def("tile_to_str", &tile_to_str);
    m.def("hand_to_str", &hand_to_str);
    m.def("is_kokushi", [](const std::vector<int>& v) -> bool {
        return is_kokushi(vec_to_hand34(v));
    });
    m.def("is_chiitoi", [](const std::vector<int>& v) -> bool {
        return is_chiitoi(vec_to_hand34(v));
    });
    m.def("is_standard_agari", [](const std::vector<int>& v) -> bool {
        return is_standard_agari(vec_to_hand34(v));
    });
    m.def("is_agari", [](const std::vector<int>& v) -> bool {
        return is_agari(vec_to_hand34(v));
    });
    m.def("is_tenpai", [](const std::vector<int>& v) -> bool {
        return is_tenpai(vec_to_hand34(v));
    });
    m.def("count_yaochu_types", [](const std::vector<int>& v) -> int {
        return count_yaochu_types(vec_to_hand34(v));
    });
    m.def("analyze_yaku", [](const std::vector<int>& hand34_vec, const std::string& win_type,
                              int seat_wind, int round_wind, bool is_closed) -> py::tuple {
        Hand34 h34 = vec_to_hand34(hand34_vec);
        auto [yakus, total_han] = analyze_yaku(h34, win_type, seat_wind, round_wind, is_closed);
        py::list yaku_list;
        for (const auto& [name, han] : yakus) {
            yaku_list.append(py::make_tuple(name, han));
        }
        return py::make_tuple(yaku_list, total_han);
    }, py::arg("hand34"), py::arg("win_type"), py::arg("seat_wind"),
       py::arg("round_wind"), py::arg("is_closed") = true);
}
