from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_NAMES = ("train", "val", "test")
DatasetLayout = Literal["auto", "raw", "split"]


class DatasetValidationError(ValueError):
    """Raised when a dataset folder cannot be used by PlantGuard."""


@dataclass(frozen=True)
class SplitSummary:
    name: str
    path: Path
    class_counts: dict[str, int]

    @property
    def total_images(self) -> int:
        return sum(self.class_counts.values())

    @property
    def empty_classes(self) -> list[str]:
        return [class_name for class_name, count in self.class_counts.items() if count == 0]


@dataclass(frozen=True)
class DatasetSummary:
    layout: str
    path: Path
    splits: dict[str, SplitSummary]

    @property
    def class_names(self) -> list[str]:
        names: set[str] = set()
        for split in self.splits.values():
            names.update(split.class_counts)
        return sorted(names)

    @property
    def class_count(self) -> int:
        return len(self.class_names)

    @property
    def total_images(self) -> int:
        return sum(split.total_images for split in self.splits.values())


def validate_dataset(path: str | Path, layout: DatasetLayout = "auto") -> DatasetSummary:
    root = Path(path)
    if not root.exists():
        raise DatasetValidationError(f"Dataset path does not exist: {root}")
    if not root.is_dir():
        raise DatasetValidationError(f"Dataset path is not a directory: {root}")

    resolved_layout = _resolve_layout(root, layout)
    if resolved_layout == "raw":
        summary = DatasetSummary(layout="raw", path=root, splits={"raw": _scan_class_folder("raw", root)})
    else:
        split_summaries = {split: _scan_class_folder(split, root / split) for split in SPLIT_NAMES}
        _validate_split_class_consistency(split_summaries)
        summary = DatasetSummary(layout="split", path=root, splits=split_summaries)

    _raise_for_invalid_summary(summary)
    return summary


def format_summary(summary: DatasetSummary) -> str:
    lines = [
        f"Dataset: {summary.path}",
        f"Layout: {summary.layout}",
        f"Classes: {summary.class_count}",
        f"Total images: {summary.total_images}",
    ]
    for split_name, split in summary.splits.items():
        lines.append(f"{split_name}: {split.total_images} images")
        for class_name, count in split.class_counts.items():
            lines.append(f"  {class_name}: {count}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a PlantGuard dataset folder without training.")
    parser.add_argument("path", help="Dataset root, e.g. data/raw/plantvillage or data/splits.")
    parser.add_argument(
        "--layout",
        choices=("auto", "raw", "split"),
        default="auto",
        help="Use raw for class-folder roots, split for train/val/test roots.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = validate_dataset(args.path, layout=args.layout)
    except DatasetValidationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(format_summary(summary))
    return 0


def _resolve_layout(root: Path, layout: DatasetLayout) -> Literal["raw", "split"]:
    if layout == "raw":
        return "raw"
    if layout == "split":
        missing = [split for split in SPLIT_NAMES if not (root / split).is_dir()]
        if missing:
            raise DatasetValidationError(f"Missing dataset split folder(s): {', '.join(missing)}")
        return "split"

    if all((root / split).is_dir() for split in SPLIT_NAMES):
        return "split"
    return "raw"


def _scan_class_folder(split_name: str, root: Path) -> SplitSummary:
    if not root.exists():
        raise DatasetValidationError(f"Dataset split not found: {root}")
    if not root.is_dir():
        raise DatasetValidationError(f"Dataset split is not a directory: {root}")

    class_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not class_dirs:
        raise DatasetValidationError(f"No class folders found in {root}")

    counts = {
        class_dir.name: sum(1 for file_path in class_dir.rglob("*") if _is_image_file(file_path))
        for class_dir in class_dirs
    }
    return SplitSummary(split_name, root, counts)


def _validate_split_class_consistency(splits: dict[str, SplitSummary]) -> None:
    expected = set(splits["train"].class_counts)
    for split_name, split in splits.items():
        actual = set(split.class_counts)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            details = []
            if missing:
                details.append(f"missing in {split_name}: {', '.join(missing)}")
            if extra:
                details.append(f"extra in {split_name}: {', '.join(extra)}")
            raise DatasetValidationError("Split class folders do not match: " + "; ".join(details))


def _raise_for_invalid_summary(summary: DatasetSummary) -> None:
    empty = [
        f"{split_name}/{class_name}"
        for split_name, split in summary.splits.items()
        for class_name in split.empty_classes
    ]
    if empty:
        raise DatasetValidationError(f"Found empty class folder(s): {', '.join(empty)}")
    if summary.total_images == 0:
        raise DatasetValidationError("Dataset contains no supported image files.")


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


if __name__ == "__main__":
    raise SystemExit(main())
