import tempfile
import unittest
from pathlib import Path

from src.validate_dataset import DatasetValidationError, validate_dataset


class DatasetValidationTests(unittest.TestCase):
    def test_validates_split_dataset_and_counts_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ("train", "val", "test"):
                for class_name in ("Apple___healthy", "Tomato___Early_blight"):
                    class_dir = root / split / class_name
                    class_dir.mkdir(parents=True)
                    (class_dir / f"{split}.jpg").write_bytes(b"not decoded during validation")

            summary = validate_dataset(root, layout="split")

            self.assertEqual(summary.layout, "split")
            self.assertEqual(summary.total_images, 6)
            self.assertEqual(summary.class_count, 2)
            self.assertEqual(summary.splits["train"].class_counts["Apple___healthy"], 1)

    def test_rejects_missing_dataset_path(self):
        with self.assertRaisesRegex(DatasetValidationError, "Dataset path does not exist"):
            validate_dataset(Path("missing-dataset"), layout="raw")

    def test_rejects_empty_class_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Apple___healthy").mkdir()

            with self.assertRaisesRegex(DatasetValidationError, "empty class"):
                validate_dataset(root, layout="raw")


if __name__ == "__main__":
    unittest.main()
