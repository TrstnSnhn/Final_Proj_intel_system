import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.prepare_artifact_handoff import ArtifactHandoffError, prepare_artifact_handoff


def write_class_map(path: Path, classes: list[str] | None = None) -> None:
    path.write_text(json.dumps({"classes": classes or ["Pepper_healthy", "Tomato_healthy"]}), encoding="utf-8")


class PrepareArtifactHandoffTests(unittest.TestCase):
    def test_missing_checkpoint_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            class_map = root / "classes.json"
            write_class_map(class_map)

            with self.assertRaisesRegex(ArtifactHandoffError, "Checkpoint artifact is missing"):
                prepare_artifact_handoff(
                    checkpoint_path=root / "missing.pt",
                    class_map_path=class_map,
                    output_dir=root / "handoff",
                    expected_classes=2,
                )

    def test_missing_class_map_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"checkpoint")

            with self.assertRaisesRegex(ArtifactHandoffError, "Class map artifact is missing"):
                prepare_artifact_handoff(
                    checkpoint_path=checkpoint,
                    class_map_path=root / "missing_classes.json",
                    output_dir=root / "handoff",
                    expected_classes=2,
                )

    def test_valid_artifacts_create_export_folder_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "model.pt"
            class_map = root / "classes.json"
            output_dir = root / "handoff"
            checkpoint.write_bytes(b"checkpoint")
            write_class_map(class_map)

            manifest_path = prepare_artifact_handoff(
                checkpoint_path=checkpoint,
                class_map_path=class_map,
                output_dir=output_dir,
                expected_classes=2,
            )

            self.assertEqual(manifest_path, output_dir / "manifest.json")
            self.assertTrue((output_dir / "model.pt").exists())
            self.assertTrue((output_dir / "classes.json").exists())

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["class_count"], 2)
            self.assertEqual(manifest["recommended_repo_id"], "TrstnSnhn/plantguard-simplecnn-15class")
            files = {item["role"]: item for item in manifest["files"]}
            self.assertEqual(files["checkpoint"]["filename"], "model.pt")
            self.assertEqual(files["class_map"]["filename"], "classes.json")
            self.assertEqual(files["checkpoint"]["sha256"], hashlib.sha256(b"checkpoint").hexdigest())


if __name__ == "__main__":
    unittest.main()
