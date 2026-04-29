import importlib.util
import multiprocessing as mp
import unittest


HAS_TORCH = importlib.util.find_spec("torch") is not None


def _multiprocessing_available() -> bool:
    try:
        ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
        q = ctx.Queue()
        q.close()
        q.join_thread()
        return True
    except (OSError, PermissionError):
        return False


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

    @unittest.skipUnless(_multiprocessing_available(), "multiprocessing queues are unavailable")
    def test_train_mp_collect_smoke_cpu(self):
        from mahjong.rl.adapter import N_ACTIONS, OBS_DIM
        from mahjong.rl.trainer import ActorCritic, TrainConfig
        from mahjong.rl.trainer_mp import collect_parallel_batch_mp

        cfg = TrainConfig(
            num_updates=1,
            num_envs=2,
            target_transitions=8,
            ppo_epochs=1,
            ppo_batch_size=4,
            hidden=32,
            device="cpu",
            log_every=1,
        )
        model = ActorCritic(hidden=cfg.hidden)
        batch, stats = collect_parallel_batch_mp([], model, cfg, "cpu", num_workers=2)
        self.assertEqual(batch.size, 8)
        self.assertEqual(batch.obs.shape, (8, OBS_DIM))
        self.assertEqual(batch.mask.shape, (8, N_ACTIONS))
        self.assertTrue(stats["multiprocessing"])
        self.assertEqual(stats["workers"], 2)


if __name__ == "__main__":
    unittest.main()
