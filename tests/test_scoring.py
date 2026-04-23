import unittest

from mahjong.scoring import dora_from_indicator, resolve_ron, resolve_tsumo


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


if __name__ == "__main__":
    unittest.main()
