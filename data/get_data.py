from __future__ import annotations

from pathlib import Path
import argparse

try:
    from PIL import Image
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Pillow is not installed. Run `python -m pip install -r requirements.txt` and retry."
    ) from exc

DATA_ROOT = Path(__file__).resolve().parent
RAW_DIR = DATA_ROOT / "raw"
TARGET_DIR = RAW_DIR / "plantvillage"


def validate_images(root: Path) -> tuple[int, int]:
    total, removed = 0, 0
    for p in root.rglob("*"):
        if not p.is_file():
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
    counts = {d.name: len(list(d.glob("*"))) for d in class_dirs}
    print(f"Classes: {len(class_dirs)}")
    print(f"Total images: {sum(counts.values())}")
    for k, v in counts.items():
        print(f"{k}: {v}")


def download_dataset(dataset: str) -> None:
    try:
        import kagglehub  # type: ignore

        path = Path(kagglehub.dataset_download(dataset))
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        TARGET_DIR.mkdir(parents=True, exist_ok=True)
        # copy tree content into target dir
        for src in path.rglob("*"):
            if src.is_file():
                rel = src.relative_to(path)
                dst = TARGET_DIR / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(src.read_bytes())
    except Exception as exc:
        raise RuntimeError(
            "Failed to download dataset. Ensure Kaggle credentials are configured."
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="abdallahalidev/plantvillage-dataset")
    args = parser.parse_args()

    if TARGET_DIR.exists() and any(TARGET_DIR.iterdir()):
        print("Dataset already exists. Skipping download.")
    else:
        download_dataset(args.dataset)

    total, removed = validate_images(TARGET_DIR)
    print(f"Validated files: {total}, removed corrupted/empty: {removed}")
    summarize(TARGET_DIR)


if __name__ == "__main__":
    main()
