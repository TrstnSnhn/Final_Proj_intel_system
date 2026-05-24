from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Sequence

try:
    from .validate_artifacts import ArtifactValidationError, resolve_artifact_paths, validate_artifacts
except ImportError:  # pragma: no cover - supports running as python src/prepare_artifact_handoff.py
    from validate_artifacts import ArtifactValidationError, resolve_artifact_paths, validate_artifacts


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifact_handoff"
DEFAULT_REPO_ID = "TrstnSnhn/plantguard-simplecnn-15class"
MANIFEST_FILENAME = "manifest.json"


class ArtifactHandoffError(ValueError):
    """Raised when a local artifact handoff bundle cannot be prepared."""


def prepare_artifact_handoff(
    checkpoint_path: str | Path,
    class_map_path: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    expected_classes: int | None = 15,
    recommended_repo_id: str = DEFAULT_REPO_ID,
) -> Path:
    try:
        summary = validate_artifacts(
            checkpoint_path=checkpoint_path,
            class_map_path=class_map_path,
            expected_classes=expected_classes,
        )
    except ArtifactValidationError as exc:
        raise ArtifactHandoffError(str(exc)) from exc

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    checkpoint_copy = destination / summary.checkpoint_path.name
    class_map_copy = destination / summary.class_map_path.name
    shutil.copy2(summary.checkpoint_path, checkpoint_copy)
    shutil.copy2(summary.class_map_path, class_map_copy)

    manifest = {
        "format": "plantguard.huggingface_artifact_handoff.v1",
        "recommended_repo_id": recommended_repo_id,
        "class_count": summary.class_count,
        "sample_classes": summary.sample_classes,
        "files": [
            _manifest_file("checkpoint", checkpoint_copy),
            _manifest_file("class_map", class_map_copy),
        ],
        "upload_status": "not_uploaded",
    }
    manifest_path = destination / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a local, ignored PlantGuard artifact handoff bundle. No upload is performed."
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to checkpoint artifact. Defaults to PLANTGUARD_CHECKPOINT_PATH or the local baseline path.",
    )
    parser.add_argument(
        "--class-map",
        default=None,
        help="Path to class mapping JSON. Defaults to PLANTGUARD_CLASS_MAP_PATH or the local baseline path.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Ignored local export folder for handoff files. Default: artifact_handoff/",
    )
    parser.add_argument(
        "--expected-classes",
        type=int,
        default=15,
        help="Expected class count for the baseline artifact pair.",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Documentation-only target Hugging Face model repo ID.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    checkpoint, class_map = resolve_artifact_paths(args.checkpoint, args.class_map)

    try:
        manifest_path = prepare_artifact_handoff(
            checkpoint_path=checkpoint,
            class_map_path=class_map,
            output_dir=args.output_dir,
            expected_classes=args.expected_classes,
            recommended_repo_id=args.repo_id,
        )
    except ArtifactHandoffError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("PlantGuard artifact handoff bundle prepared.")
    print(f"Output: {_safe_path(Path(args.output_dir))}")
    print(f"Manifest: {_safe_path(manifest_path)}")
    print("No upload was performed.")
    return 0


def _manifest_file(role: str, path: Path) -> dict[str, str | int]:
    return {
        "role": role,
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except (OSError, ValueError):
        return path.name


if __name__ == "__main__":
    raise SystemExit(main())
