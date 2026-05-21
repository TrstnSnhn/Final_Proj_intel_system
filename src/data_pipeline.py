from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

try:
    from .config import (
        RAW_DIR,
        SPLIT_DIR,
        IMAGE_SIZE,
        BATCH_SIZE,
        NUM_WORKERS,
        IMAGENET_MEAN,
        IMAGENET_STD,
    )
    from .validate_dataset import IMAGE_EXTENSIONS
except ImportError:  # pragma: no cover - supports running as python src/data_pipeline.py
    from config import (
        RAW_DIR,
        SPLIT_DIR,
        IMAGE_SIZE,
        BATCH_SIZE,
        NUM_WORKERS,
        IMAGENET_MEAN,
        IMAGENET_STD,
    )
    from validate_dataset import IMAGE_EXTENSIONS


@dataclass(frozen=True)
class SplitDatasetResult:
    split_counts: dict[str, int]
    skipped_unsupported: int


def get_split_dirs(split_dir: Path | str = SPLIT_DIR) -> tuple[Path, Path, Path]:
    root = Path(split_dir)
    return root / "train", root / "val", root / "test"


def get_transforms(mode: str = "train", augmentation: bool = True):
    if mode == "train" and augmentation:
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomCrop(IMAGE_SIZE),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.3),
                transforms.RandomRotation(15),
                transforms.ColorJitter(0.2, 0.2, 0.2),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def split_dataset(
    data_dir: Path = RAW_DIR,
    split_dir: Path | str = SPLIT_DIR,
    seed: int = 42,
    overwrite: bool = True,
) -> SplitDatasetResult:
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing raw dataset directory: {data_dir}")

    train_dir, val_dir, test_dir = get_split_dirs(split_dir)
    existing_splits = [split for split in (train_dir, val_dir, test_dir) if split.exists()]
    if existing_splits and not overwrite:
        existing = ", ".join(str(path) for path in existing_splits)
        raise FileExistsError(f"Split output already exists: {existing}")

    for split in (train_dir, val_dir, test_dir):
        if split.exists():
            shutil.rmtree(split)
        split.mkdir(parents=True, exist_ok=True)

    split_counts = {"train": 0, "val": 0, "test": 0}
    skipped_unsupported = 0
    for class_dir in sorted([d for d in data_dir.iterdir() if d.is_dir()]):
        files = sorted(path for path in class_dir.iterdir() if path.is_file())
        images = [path for path in files if _is_supported_image_file(path)]
        skipped_unsupported += len(files) - len(images)
        if not images:
            raise ValueError(f"No supported image files found in class folder: {class_dir}")

        train_files, tmp_files = train_test_split(images, test_size=0.30, random_state=seed)
        val_files, test_files = train_test_split(tmp_files, test_size=0.50, random_state=seed)

        for split_name, output_dir, file_list in [
            ("train", train_dir, train_files),
            ("val", val_dir, val_files),
            ("test", test_dir, test_files),
        ]:
            out = output_dir / class_dir.name
            out.mkdir(parents=True, exist_ok=True)
            for src in file_list:
                shutil.copy2(src, out / src.name)
            split_counts[split_name] += len(file_list)

    return SplitDatasetResult(split_counts=split_counts, skipped_unsupported=skipped_unsupported)


def get_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    augmentation: bool = True,
    split_dir: Path | str = SPLIT_DIR,
):
    train_dir, val_dir, test_dir = get_split_dirs(split_dir)
    train_ds = ImageFolder(train_dir, transform=get_transforms("train", augmentation=augmentation))
    val_ds = ImageFolder(val_dir, transform=get_transforms("val"))
    test_ds = ImageFolder(test_dir, transform=get_transforms("test"))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def _count_images(split_dir: Path) -> int:
    return sum(1 for p in split_dir.rglob("*") if _is_supported_image_file(p))


def _is_supported_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["split", "summary"], default="summary")
    parser.add_argument("--raw-dir", default=str(RAW_DIR), help="Raw class-folder dataset root.")
    parser.add_argument("--split-dir", default=str(SPLIT_DIR), help="Output split root.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="Replace existing split folders before splitting. This is the default behavior.",
    )
    args = parser.parse_args()

    split_root = Path(args.split_dir)
    result = None
    if args.action == "split":
        result = split_dataset(
            data_dir=Path(args.raw_dir),
            split_dir=split_root,
            seed=args.seed,
            overwrite=args.overwrite,
        )

    train_dir, val_dir, test_dir = get_split_dirs(split_root)
    print(f"Train: {_count_images(train_dir)}")
    print(f"Val: {_count_images(val_dir)}")
    print(f"Test: {_count_images(test_dir)}")
    if result is not None:
        print(f"Skipped unsupported files: {result.skipped_unsupported}")


if __name__ == "__main__":
    main()
