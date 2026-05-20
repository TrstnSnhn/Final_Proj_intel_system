import json
import tempfile
import unittest
from pathlib import Path

from src.class_mapping import class_names_from_mapping, load_class_names, save_class_mapping


class ClassMappingTests(unittest.TestCase):
    def test_saves_and_loads_imagefolder_class_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "classes.json"
            save_class_mapping({"Apple___healthy": 0, "Tomato___Early_blight": 1}, path)

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["classes"], ["Apple___healthy", "Tomato___Early_blight"])
            self.assertEqual(load_class_names(path), ["Apple___healthy", "Tomato___Early_blight"])

    def test_orders_class_names_by_index(self):
        self.assertEqual(
            class_names_from_mapping({"Tomato___Early_blight": 1, "Apple___healthy": 0}),
            ["Apple___healthy", "Tomato___Early_blight"],
        )


if __name__ == "__main__":
    unittest.main()
