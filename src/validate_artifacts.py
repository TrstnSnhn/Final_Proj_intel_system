from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

try:
    from .class_mapping import load_class_names
except ImportError:  # pragma: no cover - supports running as python src/validate_artifacts.py
    from class_mapping import load_class_names


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "experiments/checkpoints/plantvillage_baseline_simple_cnn_best.pt"
DEFAULT_CLASS_MAP_PATH = PROJECT_ROOT / "experiments/checkpoints/plantvillage_baseline_simple_cnn_classes.json"
SAMPLE_CLASS_LIMIT = 5


class ArtifactValidationError(ValueError):
    """Raised when required model artifacts are missing or invalid."""


@dataclass(frozen=True)
class ArtifactSummary:
    checkpoint_path: Path
    class_map_path: Path
    checkpoint_size_bytes: int
    class_map_size_bytes: int
    class_count: int
    sample_classes: list[str]


def resolve_artifact_paths(checkpoint: str | Path | None, class_map: str | Path | None) -> tuple[Path, Path]:
    resolved_checkpoint = Path(
        checkpoint or os.environ.get("PLANTGUARD_CHECKPOINT_PATH") or DEFAULT_CHECKPOINT_PATH
    )
    resolved_class_map = Path(class_map or os.environ.get("PLANTGUARD_CLASS_MAP_PATH") or DEFAULT_CLASS_MAP_PATH)
    return resolved_checkpoint, resolved_class_map


def validate_artifacts(
    checkpoint_path: str | Path,
    class_map_path: str | Path,
    expected_classes: int | None = None,
) -> ArtifactSummary:
    checkpoint = Path(checkpoint_path)
    class_map = Path(class_map_path)

    if not checkpoint.exists() or not checkpoint.is_file():
        raise ArtifactValidationError(f"Checkpoint artifact is missing: {_safe_path(checkpoint)}")
    if not class_map.exists() or not class_map.is_file():
        raise ArtifactValidationError(f"Class map artifact is missing: {_safe_path(class_map)}")

    try:
        class_names = load_class_names(class_map)
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        raise ArtifactValidationError(f"Class map is invalid: {exc}") from exc

    class_count = len(class_names)
    if expected_classes is not None and class_count != expected_classes:
        raise ArtifactValidationError(f"Expected {expected_classes} classes but found {class_count}.")

    return ArtifactSummary(
        checkpoint_path=checkpoint,
        class_map_path=class_map,
        checkpoint_size_bytes=checkpoint.stat().st_size,
        class_map_size_bytes=class_map.stat().st_size,
        class_count=class_count,
        sample_classes=class_names[:SAMPLE_CLASS_LIMIT],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate PlantGuard model artifacts without loading the full model."
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
        "--expected-classes",
        type=int,
        default=None,
        help="Optional expected class count.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    checkpoint, class_map = resolve_artifact_paths(args.checkpoint, args.class_map)

    try:
        summary = validate_artifacts(
            checkpoint_path=checkpoint,
            class_map_path=class_map,
            expected_classes=args.expected_classes,
        )
    except ArtifactValidationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("PlantGuard artifact validation passed.")
    print(f"Checkpoint: {_safe_path(summary.checkpoint_path)} ({summary.checkpoint_size_bytes} bytes)")
    print(f"Class map: {_safe_path(summary.class_map_path)} ({summary.class_map_size_bytes} bytes)")
    print(f"Classes: {summary.class_count}")
    print("Sample classes: " + ", ".join(summary.sample_classes))
    return 0


def _safe_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return path.name
    except OSError:
        return path.name


if __name__ == "__main__":
    raise SystemExit(main())
