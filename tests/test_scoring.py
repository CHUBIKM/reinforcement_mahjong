import unittest

from mahjong.engine import analyze_yaku
from mahjong.scoring import dora_from_indicator, resolve_ron, resolve_tsumo


def hand_from_tiles(tiles):
    hand = [0] * 34
    for tile in tiles:
        hand[tile] += 1
    return hand


class ScoringTests(unittest.TestCase):
    def test_dora_indicator_mapping(self):
        self.assertEqual(dora_from_indicator(8), 0)   # 9m -> 1m
        self.assertEqual(dora_from_indicator(30), 27) # 北 -> 东
        self.assertEqual(dora_from_indicator(33), 31) # 中 -> 白

    def test_resolve_ron_dealer(self):
        pr = resolve_ron(winner=0, loser=1, han=3, fu=40, dealer=0)
        self.assertEqual(len(pr.score_delta), 4)
        self.assertGreater(pr.score_delta[0], 0)
        self.assertLess(pr.score_delta[1], 0)

    def test_resolve_tsumo_non_dealer(self):
        pr = resolve_tsumo(winner=1, han=2, fu=30, dealer=0)
        self.assertEqual(len(pr.score_delta), 4)
        self.assertGreater(pr.score_delta[1], 0)
        self.assertLess(pr.score_delta[0], 0)

    def test_kazoe_yakuman_toggle(self):
        pr = resolve_ron(winner=0, loser=1, han=13, fu=30, dealer=0, kazoe_yakuman=False)
        self.assertEqual(pr.level, "sanbaiman")
        self.assertEqual(pr.payments["ron"], 36000)

    def test_junchan_does_not_stack_chanta(self):
        hand = hand_from_tiles([0, 1, 2, 6, 7, 8, 9, 10, 11, 15, 16, 17, 26, 26])
        yakus, total = analyze_yaku(hand, "ron", 27, 27, True, 26)
        names = [name for name, _ in yakus]

        self.assertIn("纯全带幺九", names)
        self.assertNotIn("混全带幺九", names)
        self.assertEqual(total, 3)

    def test_pinfu_requires_ryanmen_and_non_value_pair(self):
        hand = hand_from_tiles([0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 19, 19])
        yakus, total = analyze_yaku(hand, "ron", 27, 27, True, 14)
        names = [name for name, _ in yakus]

        self.assertIn("平和", names)
        self.assertIn("一气通贯", names)
        self.assertEqual(total, 3)

    def test_ryanpeikou_does_not_stack_iipeikou(self):
        hand = hand_from_tiles([0, 0, 1, 1, 2, 2, 12, 12, 13, 13, 14, 14, 24, 24])
        yakus, total = analyze_yaku(hand, "ron", 27, 27, True, 24)
        names = [name for name, _ in yakus]

        self.assertIn("二杯口", names)
        self.assertNotIn("一杯口", names)
        self.assertEqual(total, 3)

    def test_honroutou_does_not_stack_chanta(self):
        hand = hand_from_tiles([0, 0, 0, 8, 8, 8, 27, 27, 27, 31, 31, 31, 33, 33])
        yakus, _ = analyze_yaku(hand, "ron", 28, 28, True, 0)
        names = [name for name, _ in yakus]

        self.assertIn("混老头", names)
        self.assertIn("对对和", names)
        self.assertNotIn("混全带幺九", names)

    def test_daisangen_is_natural_yakuman(self):
        hand = hand_from_tiles([0, 0, 0, 1, 1, 31, 31, 31, 32, 32, 32, 33, 33, 33])
        yakus, total = analyze_yaku(hand, "ron", 27, 27, True, 31)
        names = [name for name, _ in yakus]

        self.assertEqual(names, ["大三元(役满)"])
        self.assertEqual(total, 13)

    def test_natural_yakuman_ignores_kazoe_toggle(self):
        pr = resolve_ron(
            winner=0,
            loser=1,
            han=13,
            fu=0,
            dealer=2,
            kazoe_yakuman=False,
            yakuman_count=1,
        )

        self.assertEqual(pr.level, "yakuman")
        self.assertEqual(pr.payments["ron"], 32000)

    def test_double_yakuman_payment(self):
        pr = resolve_ron(
            winner=0,
            loser=1,
            han=26,
            fu=0,
            dealer=2,
            kazoe_yakuman=False,
            yakuman_count=2,
        )

        self.assertEqual(pr.level, "yakuman")
        self.assertEqual(pr.payments["ron"], 64000)


if __name__ == "__main__":
    unittest.main()
