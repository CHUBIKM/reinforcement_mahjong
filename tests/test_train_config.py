import tempfile
import unittest
from pathlib import Path

from mahjong.rl.trainer import TrainConfig, load_train_config


class TrainConfigTests(unittest.TestCase):
    def test_load_train_config_with_rules_override(self):
        toml_text = """
num_updates = 12
num_envs = 8
target_transitions = 1024
lr = 0.0003

[rules]
allow_multi_ron = true
enable_suucha_riichi = false
""".strip()

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "train.toml"
            p.write_text(toml_text, encoding="utf-8")
            cfg = load_train_config(str(p), base=TrainConfig())

        self.assertEqual(cfg.num_updates, 12)
        self.assertEqual(cfg.num_envs, 8)
        self.assertEqual(cfg.target_transitions, 1024)
        self.assertAlmostEqual(cfg.lr, 0.0003)
        self.assertTrue(cfg.rules.allow_multi_ron)
        self.assertFalse(cfg.rules.enable_suucha_riichi)

    def test_load_train_config_rejects_unknown_keys(self):
        toml_text = "unknown_field = 1\n"
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.toml"
            p.write_text(toml_text, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_train_config(str(p), base=TrainConfig())


if __name__ == "__main__":
    unittest.main()
