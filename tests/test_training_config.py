import unittest
from pathlib import Path

from src.training_config import (
    ConfigError,
    load_yaml_config,
    resolve_learning_rate,
    resolve_optimizer_spec,
    validate_training_config,
)


class TrainingConfigTests(unittest.TestCase):
    def test_rejects_sklearn_config_with_clear_message(self):
        cfg = {
            "experiment_name": "baseline_sklearn",
            "model": {"architecture": "sklearn_rf", "num_classes": 38},
            "data": {},
            "training": {"optimizer": "adam"},
        }

        with self.assertRaisesRegex(ConfigError, "sklearn_rf.*not supported by src/train.py"):
            validate_training_config(cfg)

    def test_resolves_sgd_optimizer_from_config(self):
        cfg = {
            "experiment_name": "ablation_optimizer",
            "model": {"architecture": "resnet18", "num_classes": 38},
            "data": {},
            "training": {"optimizer": "sgd", "lr_unfrozen": 0.0001, "momentum": 0.8},
        }

        validate_training_config(cfg)
        spec = resolve_optimizer_spec(cfg, stage="unfrozen")

        self.assertEqual(spec.name, "sgd")
        self.assertEqual(spec.lr, 0.0001)
        self.assertEqual(spec.options["momentum"], 0.8)

    def test_resolves_frozen_learning_rate_before_unfreeze(self):
        cfg = {
            "experiment_name": "resnet18_finetune_default",
            "model": {"architecture": "resnet18", "num_classes": 38},
            "data": {},
            "training": {"optimizer": "adam", "lr_frozen": 0.001, "lr_unfrozen": 0.0001},
        }

        self.assertEqual(resolve_learning_rate(cfg, stage="frozen"), 0.001)
        self.assertEqual(resolve_learning_rate(cfg, stage="unfrozen"), 0.0001)

    def test_smoke_config_is_cpu_safe_and_offline(self):
        cfg = load_yaml_config(Path("experiments/configs/smoke_test.yaml"))

        validate_training_config(cfg)

        self.assertEqual(cfg["model"]["architecture"], "simple_cnn")
        self.assertFalse(cfg["model"].get("pretrained", False))
        self.assertEqual(cfg["model"]["num_classes"], 2)
        self.assertEqual(cfg["model"]["epochs"], 1)
        self.assertEqual(cfg["data"]["num_workers"], 0)
        self.assertEqual(cfg["data"]["split_dir"], "data/smoke/splits")

    def test_plantvillage_smoke_config_uses_real_split_path_without_pretrained_download(self):
        cfg = load_yaml_config(Path("experiments/configs/plantvillage_smoke.yaml"))

        validate_training_config(cfg)

        self.assertEqual(cfg["model"]["architecture"], "simple_cnn")
        self.assertFalse(cfg["model"].get("pretrained", False))
        self.assertEqual(cfg["model"]["num_classes"], 15)
        self.assertEqual(cfg["model"]["epochs"], 1)
        self.assertEqual(cfg["data"]["split_dir"], "data/splits")
        self.assertEqual(cfg["data"]["num_workers"], 0)


if __name__ == "__main__":
    unittest.main()
