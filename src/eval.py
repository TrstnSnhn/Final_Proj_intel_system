from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from .class_mapping import infer_class_map_path, load_class_names
    from .infer import _load_model
    from .training_config import SUPPORTED_ARCHITECTURES
except ImportError:  # pragma: no cover - supports running as python src/eval.py
    from class_mapping import infer_class_map_path, load_class_names
    from infer import _load_model
    from training_config import SUPPORTED_ARCHITECTURES


class EvaluationInputError(ValueError):
    """Raised before evaluation when required files or folders are missing."""


def validate_evaluation_inputs(
    checkpoint_path: str | Path, class_map_path: str | Path, data_dir: str | Path
) -> None:
    checkpoint = Path(checkpoint_path)
    class_map = Path(class_map_path)
    dataset = Path(data_dir)

    if not checkpoint.exists():
        raise EvaluationInputError(f"Checkpoint file not found: {checkpoint}")
    if not checkpoint.is_file():
        raise EvaluationInputError(f"Checkpoint path is not a file: {checkpoint}")
    if not class_map.exists():
        raise EvaluationInputError(f"Class mapping file not found: {class_map}")
    if not class_map.is_file():
        raise EvaluationInputError(f"Class mapping path is not a file: {class_map}")
    if not dataset.exists():
        raise EvaluationInputError(f"Dataset split not found: {dataset}")
    if not dataset.is_dir():
        raise EvaluationInputError(f"Dataset split path is not a directory: {dataset}")


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    data_dir: str | Path,
    class_map_path: str | Path | None = None,
    architecture: str = "resnet18",
    batch_size: int = 32,
    num_workers: int = 0,
    device_name: str = "cpu",
) -> dict[str, Any]:
    try:
        import torch
        from torch.utils.data import DataLoader
        from torchvision.datasets import ImageFolder
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch and torchvision are required for evaluation.") from exc

    try:
        from .data_pipeline import get_transforms
    except ImportError:  # pragma: no cover
        from data_pipeline import get_transforms

    checkpoint = Path(checkpoint_path)
    class_map = Path(class_map_path) if class_map_path else infer_class_map_path(checkpoint)
    dataset_path = Path(data_dir)
    validate_evaluation_inputs(checkpoint, class_map, dataset_path)

    class_names = load_class_names(class_map)
    dataset = ImageFolder(dataset_path, transform=get_transforms("test"))
    if dataset.classes != class_names:
        raise EvaluationInputError(
            "Dataset class folders do not match class mapping. "
            f"dataset={dataset.classes}, class_map={class_names}"
        )

    model = _load_model(architecture, checkpoint, num_classes=len(class_names), device_name=device_name)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    device = torch.device(device_name)

    correct = 0
    total = 0
    per_class = {name: {"support": 0, "correct": 0} for name in class_names}

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            predictions = outputs.argmax(1)
            matches = predictions.eq(labels)
            correct += int(matches.sum().item())
            total += int(labels.numel())

            for label, match in zip(labels.detach().cpu().tolist(), matches.detach().cpu().tolist()):
                class_name = class_names[int(label)]
                per_class[class_name]["support"] += 1
                per_class[class_name]["correct"] += int(bool(match))

    accuracy = correct / total if total else 0.0
    return {
        "status": "evaluated",
        "checkpoint": str(checkpoint),
        "class_map": str(class_map),
        "data_dir": str(dataset_path),
        "architecture": architecture,
        "num_samples": total,
        "correct": correct,
        "accuracy": accuracy,
        "per_class": per_class,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a PlantGuard checkpoint on an ImageFolder split.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained .pt checkpoint.")
    parser.add_argument("--data-dir", required=True, help="Path to a class-folder split, e.g. data/splits/test.")
    parser.add_argument(
        "--class-map",
        default=None,
        help="Path to class mapping JSON. Defaults to a sibling *_classes.json file.",
    )
    parser.add_argument(
        "--architecture",
        default="resnet18",
        choices=sorted(SUPPORTED_ARCHITECTURES),
        help="Model architecture used by the checkpoint.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = evaluate_checkpoint(
            checkpoint_path=args.checkpoint,
            data_dir=args.data_dir,
            class_map_path=args.class_map,
            architecture=args.architecture,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device_name=args.device,
        )
    except (EvaluationInputError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    text = json.dumps(payload, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
