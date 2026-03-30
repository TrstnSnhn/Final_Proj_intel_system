from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from config import (
    RAW_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


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


def split_dataset(data_dir: Path = RAW_DIR, seed: int = 42) -> None:
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing raw dataset directory: {data_dir}")

    for split in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        if split.exists():
            shutil.rmtree(split)
        split.mkdir(parents=True, exist_ok=True)

    for class_dir in sorted([d for d in data_dir.iterdir() if d.is_dir()]):
        images = [p for p in class_dir.iterdir() if p.is_file()]
        train_files, tmp_files = train_test_split(images, test_size=0.30, random_state=seed)
        val_files, test_files = train_test_split(tmp_files, test_size=0.50, random_state=seed)

        for split_dir, file_list in [(TRAIN_DIR, train_files), (VAL_DIR, val_files), (TEST_DIR, test_files)]:
            out = split_dir / class_dir.name
            out.mkdir(parents=True, exist_ok=True)
            for src in file_list:
                shutil.copy2(src, out / src.name)


def get_dataloaders(batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS, augmentation: bool = True):
    train_ds = ImageFolder(TRAIN_DIR, transform=get_transforms("train", augmentation=augmentation))
    val_ds = ImageFolder(VAL_DIR, transform=get_transforms("val"))
    test_ds = ImageFolder(TEST_DIR, transform=get_transforms("test"))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def _count_images(split_dir: Path) -> int:
    return sum(1 for p in split_dir.rglob("*") if p.is_file())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["split", "summary"], default="summary")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.action == "split":
        split_dataset(seed=args.seed)

    print(f"Train: {_count_images(TRAIN_DIR)}")
    print(f"Val: {_count_images(VAL_DIR)}")
    print(f"Test: {_count_images(TEST_DIR)}")


if __name__ == "__main__":
    main()
