import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from data.get_data import find_imagefolder_root, main


class GetDataTests(unittest.TestCase):
    def test_find_imagefolder_root_prefers_kaggle_color_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            download_root = Path(tmp)
            color_root = download_root / "plantvillage dataset" / "color"
            for class_name in ("Apple___healthy", "Tomato___Early_blight"):
                class_dir = color_root / class_name
                class_dir.mkdir(parents=True)
                (class_dir / "leaf.jpg").write_bytes(b"image placeholder")

            self.assertEqual(find_imagefolder_root(download_root), color_root)

    def test_main_rejects_target_outside_data_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                exit_code = main(["--target-dir", str(Path(tmp) / "plantvillage")])

            self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
