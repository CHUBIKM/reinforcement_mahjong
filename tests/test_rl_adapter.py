import unittest

from mahjong.engine import RiichiEngine
from mahjong.rl.adapter import OBS_DIM, N_ACTIONS, mask_builder, obs_encoder


class RLAdapterTests(unittest.TestCase):
    def test_obs_dim_and_mask_dim(self):
        eng = RiichiEngine(seed=42)
        eng.reset(dealer=0)
        obs = eng.get_obs(seat=eng.cur)
        vec = obs_encoder(obs)
        mask = mask_builder(eng)
        self.assertEqual(vec.shape[0], OBS_DIM)
        self.assertEqual(mask.shape[0], N_ACTIONS)


if __name__ == "__main__":
    unittest.main()
