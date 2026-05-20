from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image

DATA_ROOT = Path(__file__).resolve().parent
RAW_DIR = DATA_ROOT / "raw"
TARGET_DIR = RAW_DIR / "plantvillage"
DEFAULT_DATASET = "abdallahalidev/plantvillage-dataset"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class DatasetDownloadError(RuntimeError):
    """Raised when the PlantVillage dataset cannot be prepared safely."""


def find_imagefolder_root(download_root: Path) -> Path:
    root = Path(download_root)
    candidates = [
        root / "plantvillage dataset" / "color",
        root / "PlantVillage",
        root / "color",
        root,
    ]
    for candidate in candidates:
        if _has_class_folders(candidate):
            return candidate

    matches = sorted(
        (path for path in root.rglob("*") if path.is_dir() and _has_class_folders(path)),
        key=lambda path: (len(path.parts), str(path).lower()),
    )
    if matches:
        return matches[0]
    raise DatasetDownloadError(f"Could not find class folders under {root}")


def copy_imagefolder(source_root: Path, target_dir: Path = TARGET_DIR, overwrite: bool = False) -> None:
    source = find_imagefolder_root(source_root)
    _ensure_safe_raw_target(target_dir)
    if target_dir.exists() and any(target_dir.iterdir()):
        if not overwrite:
            raise DatasetDownloadError(f"Target already exists and is not empty: {target_dir}")
        shutil.rmtree(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)
    for class_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        if _count_images(class_dir) == 0:
            continue
        shutil.copytree(class_dir, target_dir / class_dir.name, dirs_exist_ok=True)


def validate_images(root: Path) -> tuple[int, int]:
    total, removed = 0, 0
    for p in root.rglob("*"):
        if not _is_image_file(p):
            continue
        total += 1
        if p.stat().st_size == 0:
            p.unlink(missing_ok=True)
            removed += 1
            continue
        try:
            with Image.open(p) as img:
                img.verify()
        except Exception:
            p.unlink(missing_ok=True)
            removed += 1
    return total, removed


def summarize(root: Path) -> None:
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    counts = {d.name: _count_images(d) for d in class_dirs}
    print(f"Classes: {len(class_dirs)}")
    print(f"Total images: {sum(counts.values())}")
    for k, v in counts.items():
        print(f"{k}: {v}")


def download_dataset(dataset: str) -> Path:
    try:
        import kagglehub  # type: ignore

        return Path(kagglehub.dataset_download(dataset))
    except Exception as exc:
        raise DatasetDownloadError(
            "Failed to download dataset. Ensure Kaggle credentials are configured."
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare PlantVillage in PlantGuard's raw ImageFolder layout.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--target-dir", default=str(TARGET_DIR), help="Normalized raw dataset output folder.")
    parser.add_argument(
        "--source-dir",
        help="Use an already downloaded/extracted dataset folder instead of downloading with KaggleHub.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing target folder.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned source and target without writing.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    target_dir = Path(args.target_dir)

    if args.dry_run:
        print(f"Dataset: {args.dataset}")
        print(f"Target: {target_dir}")
        if args.source_dir:
            print(f"Source: {Path(args.source_dir)}")
        else:
            print("Source: KaggleHub download cache")
        print("Dry run only. No network request or file write was performed.")
        return 0

    try:
        _ensure_safe_raw_target(target_dir)
        if target_dir.exists() and any(target_dir.iterdir()) and not args.overwrite and not args.source_dir:
            print("Dataset already exists. Skipping download.")
        elif args.source_dir:
            copy_imagefolder(Path(args.source_dir), target_dir=target_dir, overwrite=args.overwrite)
        else:
            download_root = download_dataset(args.dataset)
            copy_imagefolder(download_root, target_dir=target_dir, overwrite=args.overwrite)

        total, removed = validate_images(target_dir)
        print(f"Validated files: {total}, removed corrupted/empty: {removed}")
        summarize(target_dir)
    except DatasetDownloadError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


def _has_class_folders(root: Path) -> bool:
    return root.exists() and root.is_dir() and any(_count_images(path) > 0 for path in root.iterdir() if path.is_dir())


def _count_images(root: Path) -> int:
    return sum(1 for path in root.rglob("*") if _is_image_file(path))


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _ensure_safe_raw_target(target_dir: Path) -> None:
    raw_root = RAW_DIR.resolve()
    resolved = target_dir.resolve()
    if raw_root not in (resolved, *resolved.parents):
        raise DatasetDownloadError(f"Refusing to overwrite target outside {raw_root}: {target_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
