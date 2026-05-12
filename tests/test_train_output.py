import unittest

import numpy as np

from mahjong.rl.trainer import Batch, TrainConfig, format_train_start, format_train_status


class TrainOutputTests(unittest.TestCase):
    def test_train_start_uses_chinese_explanations(self):
        cfg = TrainConfig(num_updates=3, num_envs=2, target_transitions=8, ppo_epochs=1, ppo_batch_size=4)

        text = format_train_start(tag="[SEQ]", cfg=cfg, device="cpu")

        self.assertIn("单进程顺序训练开始", text)
        self.assertIn("训练目标：共 3 轮参数更新", text)
        self.assertIn("奖励构成：", text)

    def test_train_status_uses_chinese_metric_labels(self):
        cfg = TrainConfig(num_updates=5, num_envs=2, target_transitions=4)
        batch = Batch(
            obs=np.zeros((4, 1), dtype=np.float32),
            mask=np.zeros((4, 1), dtype=np.float32),
            act=np.zeros((4,), dtype=np.int64),
            logp=np.zeros((4,), dtype=np.float32),
            rew=np.array([0.1, -0.1, 0.3, 0.0], dtype=np.float32),
            done=np.zeros((4,), dtype=np.float32),
            val=np.zeros((4,), dtype=np.float32),
            env_id=np.zeros((4,), dtype=np.int64),
            size=4,
        )
        stats = {
            "steps": 4,
            "auto_pass": 2,
            "done": {"tsumo": 1, "ron": 1, "ryuukyoku": 0},
            "collect_seconds": 0.5,
            "steps_per_second": 8.0,
            "update_seconds": 0.25,
        }
        metrics = {"loss": 1.0, "pl": 0.2, "vl": 1.5, "ent": 0.7}

        text = format_train_status(
            tag="[MP]",
            update=2,
            cfg=cfg,
            device="cpu",
            batch=batch,
            stats=stats,
            metrics=metrics,
            elapsed_total=10.0,
            update_seconds=1.0,
        )

        self.assertIn("多进程采样训练状态：第 2/5 轮更新", text)
        self.assertIn("数据采集：", text)
        self.assertIn("终局统计：", text)
        self.assertIn("奖励统计：", text)
        self.assertIn("PPO 优化：总损失=1.0000", text)
        self.assertIn("耗时统计：", text)
        self.assertIn("采集速度=8步/秒", text)
        self.assertNotIn("reward=", text)
        self.assertNotIn("loss=", text)
        self.assertNotIn("collect_sps=", text)


if __name__ == "__main__":
    unittest.main()
