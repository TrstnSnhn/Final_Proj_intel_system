from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

try:
    from .class_mapping import infer_class_map_path, load_class_names
    from .training_config import SUPPORTED_ARCHITECTURES
except ImportError:  # pragma: no cover - supports running as python src/infer.py
    from class_mapping import infer_class_map_path, load_class_names
    from training_config import SUPPORTED_ARCHITECTURES


def format_topk_predictions(
    class_names: Sequence[str], probabilities: Sequence[float], top_k: int
) -> list[dict[str, float | int | str]]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")
    if len(class_names) != len(probabilities):
        raise ValueError("class_names and probabilities must have the same length.")
    if top_k > len(class_names):
        raise ValueError("top_k cannot exceed the number of classes.")

    ranked = sorted(enumerate(probabilities), key=lambda item: (-float(item[1]), item[0]))
    return [
        {
            "rank": rank,
            "class_name": class_names[index],
            "confidence": float(probability),
        }
        for rank, (index, probability) in enumerate(ranked[:top_k], start=1)
    ]


def preprocess_image(image_path: str | Path):
    try:
        from PIL import Image
        from torchvision import transforms
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pillow and torchvision are required for image preprocessing.") from exc

    try:
        from .config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
    except ImportError:  # pragma: no cover - supports running as python src/infer.py
        from config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

    image = Path(image_path)
    if not image.exists():
        raise FileNotFoundError(f"Image file not found: {image}")
    if not image.is_file():
        raise ValueError(f"Image path is not a file: {image}")

    try:
        with Image.open(image) as img:
            rgb = img.convert("RGB")
            transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.CenterCrop(IMAGE_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            )
            return transform(rgb).unsqueeze(0)
    except Exception as exc:
        raise ValueError(f"Could not read image file: {image}") from exc


def predict_image(
    checkpoint_path: str | Path,
    image_path: str | Path,
    class_map_path: str | Path | None = None,
    architecture: str = "resnet18",
    top_k: int = 3,
    device_name: str = "cpu",
) -> list[dict[str, float | int | str]]:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")
    if not checkpoint.is_file():
        raise ValueError(f"Checkpoint path is not a file: {checkpoint}")

    resolved_class_map = Path(class_map_path) if class_map_path else infer_class_map_path(checkpoint)
    class_names = load_class_names(resolved_class_map)
    model = _load_model(architecture, checkpoint, num_classes=len(class_names), device_name=device_name)
    image_tensor = preprocess_image(image_path)

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for inference.") from exc

    device = torch.device(device_name)
    with torch.no_grad():
        logits = model(image_tensor.to(device))
        probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().tolist()

    return format_topk_predictions(class_names, probabilities, top_k)


def _load_model(architecture: str, checkpoint: Path, num_classes: int, device_name: str):
    if architecture not in SUPPORTED_ARCHITECTURES:
        supported = ", ".join(sorted(SUPPORTED_ARCHITECTURES))
        raise ValueError(f"Unsupported architecture: {architecture}. Supported: {supported}.")

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for inference.") from exc

    try:
        from .models.cnn_from_scratch import SimpleCNN
        from .models.resnet_finetune import ResNet18Classifier
    except ImportError:  # pragma: no cover - supports running as python src/infer.py
        from models.cnn_from_scratch import SimpleCNN
        from models.resnet_finetune import ResNet18Classifier

    if architecture == "simple_cnn":
        model = SimpleCNN(num_classes)
    elif architecture == "resnet18":
        model = ResNet18Classifier(num_classes, pretrained=False)
    else:  # validate above keeps this defensive.
        raise ValueError(f"Unsupported architecture: {architecture}")

    device = torch.device(device_name)
    checkpoint_payload = torch.load(checkpoint, map_location=device)
    state_dict = checkpoint_payload.get("model_state_dict", checkpoint_payload)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PlantGuard image inference from a trained checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained .pt checkpoint.")
    parser.add_argument("--image", required=True, help="Path to a leaf image.")
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
    parser.add_argument("--top-k", type=int, default=3, help="Number of predictions to return.")
    parser.add_argument("--device", default="cpu", help="Torch device, for example cpu or cuda.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        predictions = predict_image(
            checkpoint_path=args.checkpoint,
            image_path=args.image,
            class_map_path=args.class_map,
            architecture=args.architecture,
            top_k=args.top_k,
            device_name=args.device,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({"predictions": predictions}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
