import tempfile
import unittest
from pathlib import Path

from src.data_pipeline import split_dataset
from src.validate_dataset import validate_dataset


class DataPipelineSplitTests(unittest.TestCase):
    def test_split_dataset_writes_to_requested_split_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw" / "plantvillage"
            split_dir = root / "splits"
            for class_name in ("Apple___healthy", "Tomato___Early_blight"):
                class_dir = raw_dir / class_name
                class_dir.mkdir(parents=True)
                for index in range(10):
                    (class_dir / f"{index:02d}.jpg").write_bytes(b"image placeholder")

            split_dataset(data_dir=raw_dir, split_dir=split_dir, seed=42)

            summary = validate_dataset(split_dir, layout="split")
            self.assertEqual(summary.class_names, ["Apple___healthy", "Tomato___Early_blight"])
            self.assertEqual(summary.total_images, 20)


if __name__ == "__main__":
    unittest.main()
