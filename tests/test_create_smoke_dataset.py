import tempfile
import unittest
from pathlib import Path

from src.create_smoke_dataset import SMOKE_CLASSES, create_smoke_dataset
from src.validate_dataset import validate_dataset


class SmokeDatasetTests(unittest.TestCase):
    def test_creates_split_imagefolder_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "smoke" / "splits"

            create_smoke_dataset(root, overwrite=False, image_size=32)
            summary = validate_dataset(root, layout="split")

            self.assertEqual(summary.layout, "split")
            self.assertEqual(summary.class_names, list(SMOKE_CLASSES))
            self.assertEqual(summary.class_count, 2)
            self.assertGreater(summary.splits["train"].total_images, 0)
            self.assertGreater(summary.splits["val"].total_images, 0)
            self.assertGreater(summary.splits["test"].total_images, 0)


if __name__ == "__main__":
    unittest.main()
