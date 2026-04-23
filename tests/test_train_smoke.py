import importlib.util
import unittest


HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class TrainSmokeTests(unittest.TestCase):
    def test_train_smoke_cpu(self):
        from mahjong.rl.trainer import TrainConfig, train

        cfg = TrainConfig(
            num_updates=1,
            num_envs=2,
            target_transitions=64,
            ppo_epochs=1,
            ppo_batch_size=32,
            device="cpu",
            log_every=1,
        )
        model = train(cfg)
        self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
