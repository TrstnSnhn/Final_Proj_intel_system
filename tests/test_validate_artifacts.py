import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.validate_artifacts import ArtifactValidationError, resolve_artifact_paths, validate_artifacts


class ValidateArtifactsTests(unittest.TestCase):
    def write_class_map(self, path: Path, classes: list[str] | None = None) -> None:
        payload = {"classes": classes or ["Tomato_healthy", "Tomato_Late_blight"]}
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_missing_checkpoint_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            class_map = root / "classes.json"
            self.write_class_map(class_map)

            with self.assertRaisesRegex(ArtifactValidationError, "Checkpoint artifact is missing"):
                validate_artifacts(root / "missing.pt", class_map)

    def test_missing_class_map_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"placeholder")

            with self.assertRaisesRegex(ArtifactValidationError, "Class map artifact is missing"):
                validate_artifacts(checkpoint, root / "missing.json")

    def test_invalid_class_map_json_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "model.pt"
            class_map = root / "classes.json"
            checkpoint.write_bytes(b"placeholder")
            class_map.write_text("{not-json", encoding="utf-8")

            with self.assertRaisesRegex(ArtifactValidationError, "Class map is invalid"):
                validate_artifacts(checkpoint, class_map)

    def test_valid_artifacts_return_basic_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "model.pt"
            class_map = root / "classes.json"
            checkpoint.write_bytes(b"placeholder")
            self.write_class_map(class_map)

            summary = validate_artifacts(checkpoint, class_map, expected_classes=2)

            self.assertEqual(summary.class_count, 2)
            self.assertEqual(summary.checkpoint_size_bytes, len(b"placeholder"))
            self.assertEqual(summary.class_map_size_bytes, class_map.stat().st_size)
            self.assertEqual(summary.sample_classes, ["Tomato_healthy", "Tomato_Late_blight"])

    def test_expected_class_count_mismatch_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "model.pt"
            class_map = root / "classes.json"
            checkpoint.write_bytes(b"placeholder")
            self.write_class_map(class_map, ["Tomato_healthy"])

            with self.assertRaisesRegex(ArtifactValidationError, "Expected 2 classes but found 1"):
                validate_artifacts(checkpoint, class_map, expected_classes=2)

    def test_environment_overrides_are_used_without_printing_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "private-model.pt"
            class_map = root / "private-classes.json"

            with patch.dict(
                "os.environ",
                {
                    "PLANTGUARD_CHECKPOINT_PATH": str(checkpoint),
                    "PLANTGUARD_CLASS_MAP_PATH": str(class_map),
                },
            ):
                resolved_checkpoint, resolved_class_map = resolve_artifact_paths(None, None)

            self.assertEqual(resolved_checkpoint, checkpoint)
            self.assertEqual(resolved_class_map, class_map)


if __name__ == "__main__":
    unittest.main()
