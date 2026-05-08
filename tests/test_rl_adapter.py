import unittest

import numpy as np

from mahjong.engine import Phase, RiichiEngine
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

    def test_phase_one_hot_matches_cpp_observation_array(self):
        eng = RiichiEngine(seed=43)
        eng.reset(dealer=0)

        for phase in (Phase.DRAW, Phase.DISCARD, Phase.RESPONSE, Phase.END):
            eng.phase = phase
            vec = obs_encoder(eng.get_obs(seat=eng.cur))
            arr = eng.get_obs_array(eng.cur)
            np.testing.assert_array_equal(vec[102:106], arr[102:106])


if __name__ == "__main__":
    unittest.main()
