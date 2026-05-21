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

    def test_split_dataset_skips_non_image_files_and_reports_supported_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw" / "plantvillage"
            split_dir = root / "splits"
            for class_name in ("Apple___healthy", "Tomato___Early_blight"):
                class_dir = raw_dir / class_name
                class_dir.mkdir(parents=True)
                for index in range(12):
                    (class_dir / f"{index:02d}.jpg").write_bytes(b"image placeholder")
                (class_dir / "metadata.txt").write_text("not an image", encoding="utf-8")

            result = split_dataset(data_dir=raw_dir, split_dir=split_dir, seed=42)

            copied_non_images = [
                path
                for path in split_dir.rglob("*")
                if path.is_file() and path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            ]
            summary = validate_dataset(split_dir, layout="split")
            self.assertEqual(copied_non_images, [])
            self.assertEqual(summary.total_images, 24)
            self.assertEqual(result.split_counts, {"train": 16, "val": 4, "test": 4})
            self.assertEqual(result.skipped_unsupported, 2)

    def test_split_dataset_is_deterministic_for_same_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw" / "plantvillage"
            first_split = root / "splits-a"
            second_split = root / "splits-b"
            for class_name in ("Apple___healthy", "Tomato___Early_blight"):
                class_dir = raw_dir / class_name
                class_dir.mkdir(parents=True)
                for index in range(12):
                    (class_dir / f"{index:02d}.jpg").write_bytes(b"image placeholder")

            split_dataset(data_dir=raw_dir, split_dir=first_split, seed=42)
            split_dataset(data_dir=raw_dir, split_dir=second_split, seed=42)

            first_paths = sorted(path.relative_to(first_split) for path in first_split.rglob("*.jpg"))
            second_paths = sorted(path.relative_to(second_split) for path in second_split.rglob("*.jpg"))
            self.assertEqual(first_paths, second_paths)

    def test_split_dataset_rejects_class_with_no_supported_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw" / "plantvillage"
            split_dir = root / "splits"
            class_dir = raw_dir / "Apple___healthy"
            class_dir.mkdir(parents=True)
            (class_dir / "metadata.txt").write_text("not an image", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "No supported image files"):
                split_dataset(data_dir=raw_dir, split_dir=split_dir, seed=42)


if __name__ == "__main__":
    unittest.main()
