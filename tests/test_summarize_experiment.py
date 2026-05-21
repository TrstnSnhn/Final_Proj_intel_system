import json
import tempfile
import unittest
from pathlib import Path

from src.summarize_experiment import ExperimentSummaryError, build_experiment_summary, format_summary


class SummarizeExperimentTests(unittest.TestCase):
    def test_builds_concise_summary_from_eval_json_and_training_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_summary = root / "eval.json"
            training_log = root / "train.csv"
            eval_summary.write_text(
                json.dumps(
                    {
                        "num_samples": 10,
                        "correct": 8,
                        "accuracy": 0.8,
                        "per_class": {
                            "Tomato_healthy": {"support": 6, "correct": 6},
                            "Potato___healthy": {"support": 4, "correct": 2},
                        },
                    }
                ),
                encoding="utf-8",
            )
            training_log.write_text(
                "epoch,train_loss,train_acc,val_loss,val_acc,val_macro_f1,lr,optimizer,stage,time_sec\n"
                "1,1.2,0.4,1.0,0.5,0.45,0.001,adam,unfrozen,12.5\n"
                "2,0.8,0.7,0.6,0.8,0.75,0.001,adam,unfrozen,25.0\n",
                encoding="utf-8",
            )

            summary = build_experiment_summary(eval_summary, training_log)
            text = format_summary(summary)

            self.assertEqual(summary["epochs"], 2)
            self.assertEqual(summary["best_val_loss"], 0.6)
            self.assertEqual(summary["best_val_loss_epoch"], 2)
            self.assertEqual(summary["eval_samples"], 10)
            self.assertEqual(summary["eval_correct"], 8)
            self.assertIn("Accuracy: 80.00%", text)
            self.assertIn("Best val_loss: 0.6000 at epoch 2", text)

    def test_rejects_missing_eval_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            training_log = root / "train.csv"
            training_log.write_text("epoch,val_loss\n1,0.5\n", encoding="utf-8")

            with self.assertRaisesRegex(ExperimentSummaryError, "Evaluation summary not found"):
                build_experiment_summary(root / "missing.json", training_log)


if __name__ == "__main__":
    unittest.main()
