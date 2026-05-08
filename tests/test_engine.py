import unittest

from mahjong.engine import Action, ActionType, Phase, RiichiEngine


def hand_from_tiles(tiles):
    hand = [0] * 34
    for tile in tiles:
        hand[tile] += 1
    return hand


class EngineTests(unittest.TestCase):
    def test_random_play_keeps_tile_invariants(self):
        eng = RiichiEngine(seed=7)
        eng.reset(dealer=0)
        eng.validate_invariants()
        res = eng.play_random(max_steps=20000, verbose=False)
        self.assertTrue(res.done)
        eng.validate_invariants()
        self.assertEqual(len(res.score_delta), 4)

    def test_riichi_action_exists_for_tenpai_discard(self):
        eng = RiichiEngine(seed=1)
        eng.reset(dealer=0)
        seat = eng.cur
        hand = [0] * 34
        hand[0] = 3
        hand[1] = 3
        hand[2] = 3
        hand[3] = 3
        hand[4] = 2
        eng.players[seat].hand34 = hand
        eng.phase = Phase.DISCARD

        acts = eng.legal_actions()
        riichi_tiles = sorted(a.tile for a in acts if a.type == ActionType.RIICHI)
        self.assertIn(4, riichi_tiles)

    def test_step_result_score_fields_on_tsumo(self):
        eng = RiichiEngine(seed=2)
        eng.reset(dealer=0)
        seat = eng.cur
        hand = [0] * 34
        hand[0] = 3
        hand[1] = 3
        hand[2] = 3
        hand[3] = 3
        hand[4] = 2
        eng.players[seat].hand34 = hand
        eng.phase = Phase.DISCARD

        res = eng.apply_action(Action(ActionType.TSUMO))
        self.assertTrue(res.done)
        self.assertEqual(res.reason, "tsumo")
        self.assertEqual(len(res.score_delta), 4)
        self.assertGreaterEqual(res.han, 1)

    def test_kyuushu_kyuuhai_action_available(self):
        eng = RiichiEngine(seed=9)
        eng.reset(dealer=0)
        eng.phase = Phase.DISCARD
        seat = eng.cur

        hand = [0] * 34
        for t in [0, 8, 9, 17, 18, 26, 27, 28, 29]:
            hand[t] = 1
        hand[1] = 5
        eng.players[seat].hand34 = hand
        eng.players[seat].river = []
        for p in eng.players:
            p.river = []

        acts = eng.legal_actions()
        self.assertTrue(any(a.type == ActionType.ABORTIVE_DRAW for a in acts))

    def test_minkan_from_response(self):
        eng = RiichiEngine(seed=10)
        eng.reset(dealer=0)
        eng.phase = Phase.DISCARD
        eng.cur = 0
        eng.players[0].hand34 = [0] * 34
        eng.players[0].hand34[4] = 1
        eng.players[1].hand34 = [0] * 34
        eng.players[1].hand34[4] = 3
        eng.players[2].hand34 = [0] * 34
        eng.players[3].hand34 = [0] * 34

        eng.apply_action(Action(ActionType.DISCARD, tile=4))
        self.assertEqual(eng.phase, Phase.RESPONSE)
        self.assertEqual(eng.cur, 1)

        minkan = next(a for a in eng.legal_actions() if a.type == ActionType.KAN)
        res = eng.apply_action(minkan)  # declare
        self.assertFalse(res.done)
        self.assertEqual(eng.phase, Phase.RESPONSE)
        self.assertEqual(eng.cur, 2)

        eng.apply_action(Action(ActionType.PASS))  # actor 2
        res = eng.apply_action(Action(ActionType.PASS))  # actor 3 -> resolve minkan
        self.assertFalse(res.done)
        self.assertEqual(eng.phase, Phase.DISCARD)
        self.assertEqual(eng.cur, 1)
        self.assertTrue(any(m[0] == "minkan" for m in eng.players[1].melds))
        self.assertTrue(eng.discard_was_called[0])

    def test_kakan_can_be_chankaned(self):
        eng = RiichiEngine(seed=11)
        eng.reset(dealer=0)
        eng.phase = Phase.DISCARD
        eng.cur = 0

        eng.players[0].hand34 = [0] * 34
        eng.players[0].hand34[4] = 1
        eng.players[0].melds = [("pon", [4, 4, 4])]

        eng.players[1].hand34 = [0] * 34
        eng.players[1].hand34[0] = 3
        eng.players[1].hand34[1] = 3
        eng.players[1].hand34[2] = 3
        eng.players[1].hand34[3] = 3
        eng.players[1].hand34[4] = 1

        eng.players[2].hand34 = [0] * 34
        eng.players[3].hand34 = [0] * 34

        res = eng.apply_action(Action(ActionType.KAN, tile=4, info={"kan_type": "KAKAN"}))
        self.assertTrue(res.done)
        self.assertEqual(res.reason, "ron")
        self.assertEqual(res.loser, 0)
        self.assertIn(1, res.winners)
        self.assertTrue(res.flags.get("chankan", False))

    def test_suufon_renda_detection(self):
        eng = RiichiEngine(seed=12)
        eng.reset(dealer=0)
        eng.first_discards = [27, 27, 27, 27]
        eng.open_call_happened = False
        self.assertTrue(eng._should_abort_suufon_renda())

    def test_fu_pinfu_tsumo_is_20(self):
        eng = RiichiEngine(seed=13)
        eng.reset(dealer=0)
        seat = eng.cur
        hand = [0] * 34
        for t in [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14]:
            hand[t] += 1
        hand[19] = 2  # 2s pair (non-value)
        eng.players[seat].hand34 = hand
        eng.last_draw = 14  # 6p
        y = eng._yaku_info_for_win(seat, "tsumo", hand, 14)
        self.assertEqual(y["fu"], 20)

    def test_fu_pinfu_ron_is_30(self):
        eng = RiichiEngine(seed=14)
        eng.reset(dealer=0)
        seat = eng.cur
        hand = [0] * 34
        for t in [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14]:
            hand[t] += 1
        hand[19] = 2
        eng.players[seat].hand34 = hand
        y = eng._yaku_info_for_win(seat, "ron", hand, 14)
        self.assertEqual(y["fu"], 30)

    def test_fu_tanki_yakuhai_pair_ron(self):
        eng = RiichiEngine(seed=15)
        eng.reset(dealer=0)
        seat = eng.cur
        hand = [0] * 34
        for t in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12]:
            hand[t] += 1
        hand[31] = 2  # white dragon pair, tanki win on white
        eng.players[seat].hand34 = hand
        y = eng._yaku_info_for_win(seat, "ron", hand, 31)
        self.assertEqual(y["fu"], 40)

    def test_open_hand_ron_is_legal(self):
        eng = RiichiEngine(seed=16)
        eng.reset(dealer=0)
        eng.phase = Phase.DISCARD
        eng.cur = 0

        eng.players[0].hand34 = hand_from_tiles([26])
        eng.players[1].hand34 = hand_from_tiles([1, 2, 3, 12, 13, 14, 24, 25, 22, 22])
        eng.players[1].melds = [("pon", [31, 31, 31])]
        eng.players[2].hand34 = [0] * 34
        eng.players[3].hand34 = [0] * 34

        eng.apply_action(Action(ActionType.DISCARD, tile=26))
        self.assertTrue(any(a.type == ActionType.RON for a in eng.legal_actions()))

    def test_no_yaku_ron_is_not_legal(self):
        eng = RiichiEngine(seed=160)
        eng.reset(dealer=0)
        eng.phase = Phase.DISCARD
        eng.cur = 0
        eng.live_wall = [5]

        eng.players[0].hand34 = hand_from_tiles([3])
        eng.players[1].hand34 = hand_from_tiles([1, 2, 9, 10, 11, 12, 13, 14, 18, 19, 20, 31, 31])
        eng.players[2].hand34 = [0] * 34
        eng.players[3].hand34 = [0] * 34

        eng.apply_action(Action(ActionType.DISCARD, tile=3))
        self.assertFalse(any(a.type == ActionType.RON for a in eng.legal_actions()))

    def test_houtei_can_supply_only_yaku_for_ron(self):
        eng = RiichiEngine(seed=161)
        eng.reset(dealer=0)
        eng.phase = Phase.DISCARD
        eng.cur = 0
        eng.live_wall = []

        eng.players[0].hand34 = hand_from_tiles([3])
        eng.players[1].hand34 = hand_from_tiles([1, 2, 9, 10, 11, 12, 13, 14, 18, 19, 20, 31, 31])
        eng.players[2].hand34 = [0] * 34
        eng.players[3].hand34 = [0] * 34

        eng.apply_action(Action(ActionType.DISCARD, tile=3))
        ron = next(a for a in eng.legal_actions() if a.type == ActionType.RON)
        res = eng.apply_action(ron)

        self.assertTrue(res.done)
        self.assertIn(("河底捞鱼", 1), res.yaku_list)
        self.assertTrue(res.flags.get("houtei", False))

    def test_open_hand_tsumo_is_legal(self):
        eng = RiichiEngine(seed=17)
        eng.reset(dealer=0)
        eng.phase = Phase.DISCARD
        eng.cur = 1

        eng.players[1].hand34 = hand_from_tiles([1, 2, 3, 12, 13, 14, 24, 25, 26, 22, 22])
        eng.players[1].melds = [("pon", [31, 31, 31])]

        self.assertTrue(any(a.type == ActionType.TSUMO for a in eng.legal_actions()))

    def test_kan_meld_hand_tsumo_is_legal(self):
        eng = RiichiEngine(seed=18)
        eng.reset(dealer=0)
        eng.phase = Phase.DISCARD
        eng.cur = 2

        eng.players[2].hand34 = hand_from_tiles([1, 2, 3, 12, 13, 14, 24, 25, 26, 22, 22])
        eng.players[2].melds = [("minkan", [31, 31, 31, 31])]

        self.assertTrue(any(a.type == ActionType.TSUMO for a in eng.legal_actions()))

    def test_non_dealer_kyuushu_kyuuhai_action_available(self):
        eng = RiichiEngine(seed=19)
        eng.reset(dealer=0)
        eng.phase = Phase.DISCARD
        eng.cur = 1
        eng.open_call_happened = False

        for p in eng.players:
            p.river = []
        eng.players[0].river = [5]

        hand = [0] * 34
        for t in [0, 8, 9, 17, 18, 26, 27, 28, 29]:
            hand[t] = 1
        hand[1] = 5
        eng.players[1].hand34 = hand

        self.assertTrue(any(a.type == ActionType.ABORTIVE_DRAW for a in eng.legal_actions()))

    def test_riichi_player_must_discard_last_draw(self):
        eng = RiichiEngine(seed=20)
        eng.reset(dealer=0)
        eng.phase = Phase.DISCARD
        eng.cur = 0
        eng.players[0].riichi_declared = True
        eng.last_draw = 13
        eng.players[0].hand34 = hand_from_tiles([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 13])

        discards = [a.tile for a in eng.legal_actions() if a.type == ActionType.DISCARD]
        self.assertEqual(discards, [13])

    def test_permanent_furiten_blocks_all_waits(self):
        eng = RiichiEngine(seed=21)
        eng.reset(dealer=0)
        eng.phase = Phase.DISCARD
        eng.cur = 0

        eng.players[0].hand34 = hand_from_tiles([3])
        eng.players[1].hand34 = hand_from_tiles([1, 2, 9, 10, 11, 12, 13, 14, 18, 19, 20, 31, 31])
        eng.players[1].river = [0]
        eng.players[2].hand34 = [0] * 34
        eng.players[3].hand34 = [0] * 34

        eng.apply_action(Action(ActionType.DISCARD, tile=3))
        self.assertFalse(any(a.type == ActionType.RON for a in eng.legal_actions()))

    def test_riichi_missed_ron_furiten_persists_after_draw(self):
        eng = RiichiEngine(seed=22)
        eng.reset(dealer=0)
        eng.phase = Phase.DISCARD
        eng.cur = 0

        tenpai = hand_from_tiles([1, 2, 9, 10, 11, 12, 13, 14, 18, 19, 20, 31, 31])
        eng.players[0].hand34 = hand_from_tiles([3, 4])
        eng.players[1].hand34 = list(tenpai)
        eng.players[1].riichi_declared = True
        eng.players[2].hand34 = [0] * 34
        eng.players[3].hand34 = [0] * 34

        eng.apply_action(Action(ActionType.DISCARD, tile=3))
        self.assertTrue(any(a.type == ActionType.RON for a in eng.legal_actions()))
        eng.apply_action(Action(ActionType.PASS))
        eng.apply_action(Action(ActionType.PASS))
        eng.apply_action(Action(ActionType.PASS))

        eng.cur = 1
        eng.phase = Phase.DRAW
        eng.draw()

        eng.players[1].hand34 = list(tenpai)
        eng.players[0].hand34 = hand_from_tiles([0])
        eng.phase = Phase.DISCARD
        eng.cur = 0
        eng.apply_action(Action(ActionType.DISCARD, tile=0))
        self.assertFalse(any(a.type == ActionType.RON for a in eng.legal_actions()))

    def test_riichi_ippatsu_pinfu_tsumo_are_scored(self):
        eng = RiichiEngine(seed=23)
        eng.reset(dealer=0)
        eng.phase = Phase.DISCARD
        eng.cur = 0
        eng.dora_indicators = []
        eng.players[0].riichi_declared = True
        eng.ippatsu_active = [True, False, False, False]
        eng.last_draw = 14
        eng.players[0].hand34 = hand_from_tiles([0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 19, 19])

        res = eng.apply_action(Action(ActionType.TSUMO))
        names = [name for name, _ in res.yaku_list]

        self.assertTrue(res.done)
        self.assertIn("立直", names)
        self.assertIn("一发", names)
        self.assertIn("门前清自摸和", names)
        self.assertIn("平和", names)
        self.assertEqual(res.han, 6)

    def test_chankan_is_counted_as_yaku(self):
        eng = RiichiEngine(seed=24)
        eng.reset(dealer=0)
        eng.phase = Phase.DISCARD
        eng.cur = 0
        eng.dora_indicators = []

        eng.players[0].hand34 = hand_from_tiles([3])
        eng.players[0].melds = [("pon", [3, 3, 3])]

        eng.players[1].hand34 = hand_from_tiles([1, 2, 9, 10, 11, 12, 13, 14, 18, 19, 20, 31, 31])
        eng.players[2].hand34 = [0] * 34
        eng.players[3].hand34 = [0] * 34

        res = eng.apply_action(Action(ActionType.KAN, tile=3, info={"kan_type": "KAKAN"}))
        names = [name for name, _ in res.yaku_list]

        self.assertTrue(res.done)
        self.assertIn("抢杠", names)
        self.assertTrue(res.flags.get("chankan", False))


if __name__ == "__main__":
    unittest.main()
