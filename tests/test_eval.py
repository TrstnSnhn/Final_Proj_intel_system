import tempfile
import unittest
from pathlib import Path

from src.eval import EvaluationInputError, validate_evaluation_inputs


class EvaluationInputTests(unittest.TestCase):
    def test_rejects_missing_checkpoint_before_loading_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            class_map = root / "classes.json"
            data_dir = root / "test"
            class_map.write_text('{"classes": ["Apple___healthy"]}', encoding="utf-8")
            data_dir.mkdir()

            with self.assertRaisesRegex(EvaluationInputError, "Checkpoint file not found"):
                validate_evaluation_inputs(root / "missing.pt", class_map, data_dir)

    def test_rejects_missing_dataset_split_before_loading_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "model.pt"
            class_map = root / "classes.json"
            checkpoint.write_bytes(b"placeholder")
            class_map.write_text('{"classes": ["Apple___healthy"]}', encoding="utf-8")

            with self.assertRaisesRegex(EvaluationInputError, "Dataset split not found"):
                validate_evaluation_inputs(checkpoint, class_map, root / "missing-split")


if __name__ == "__main__":
    unittest.main()
