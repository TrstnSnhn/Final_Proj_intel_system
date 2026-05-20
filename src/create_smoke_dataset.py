from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw


SMOKE_CLASSES = ("Smoke___green_leaf", "Smoke___red_leaf")
SPLIT_COUNTS = {"train": 4, "val": 2, "test": 2}
DEFAULT_OUTPUT = Path("data/smoke/splits")


class SmokeDatasetError(ValueError):
    """Raised when the smoke dataset cannot be safely created."""


def create_smoke_dataset(
    output_dir: str | Path = DEFAULT_OUTPUT,
    overwrite: bool = False,
    image_size: int = 96,
) -> Path:
    output = Path(output_dir)
    if output.exists():
        if not overwrite:
            raise SmokeDatasetError(f"Output already exists: {output}. Pass --overwrite to recreate it.")
        _remove_existing_output(output)

    output.mkdir(parents=True, exist_ok=True)
    for split_name, count in SPLIT_COUNTS.items():
        for class_index, class_name in enumerate(SMOKE_CLASSES):
            class_dir = output / split_name / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for image_index in range(count):
                image = _make_image(class_index, image_index, image_size)
                image.save(class_dir / f"{class_name}_{image_index:02d}.png")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a tiny generated ImageFolder dataset for smoke tests.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output split root.")
    parser.add_argument("--overwrite", action="store_true", help="Delete and recreate the output folder.")
    parser.add_argument("--image-size", type=int, default=96, help="Generated square image size.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output = create_smoke_dataset(args.output, overwrite=args.overwrite, image_size=args.image_size)
    except SmokeDatasetError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Created smoke dataset at {output}")
    return 0


def _make_image(class_index: int, image_index: int, image_size: int) -> Image.Image:
    if image_size < 16:
        raise SmokeDatasetError("image_size must be at least 16 pixels.")

    base_colors = ((62, 142, 73), (180, 74, 62))
    accent_colors = ((230, 244, 216), (250, 222, 214))
    base = base_colors[class_index]
    accent = accent_colors[class_index]
    image = Image.new("RGB", (image_size, image_size), base)
    draw = ImageDraw.Draw(image)
    margin = 8 + image_index
    draw.ellipse((margin, margin, image_size - margin, image_size - margin), fill=accent)
    draw.line((image_size // 2, margin, image_size // 2, image_size - margin), fill=base, width=3)
    return image


def _remove_existing_output(output: Path) -> None:
    parts = set(output.parts)
    if "smoke" not in parts:
        raise SmokeDatasetError(f"Refusing to overwrite non-smoke output path: {output}")
    shutil.rmtree(output)


if __name__ == "__main__":
    raise SystemExit(main())
