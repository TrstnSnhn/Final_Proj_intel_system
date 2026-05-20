from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


def class_names_from_mapping(class_to_idx: Mapping[str, int]) -> list[str]:
    if not class_to_idx:
        raise ValueError("Class mapping cannot be empty.")

    pairs: list[tuple[int, str]] = []
    for class_name, index in class_to_idx.items():
        if not isinstance(class_name, str) or not class_name:
            raise ValueError("Class names must be non-empty strings.")
        if not isinstance(index, int) or index < 0:
            raise ValueError("Class indices must be non-negative integers.")
        pairs.append((index, class_name))

    pairs.sort()
    expected = list(range(len(pairs)))
    actual = [index for index, _ in pairs]
    if actual != expected:
        raise ValueError("Class indices must be contiguous and start at 0.")

    return [class_name for _, class_name in pairs]


def save_class_mapping(class_to_idx: Mapping[str, int], path: str | Path) -> Path:
    destination = Path(path)
    class_names = class_names_from_mapping(class_to_idx)
    payload = {
        "format": "plantguard.class_mapping.v1",
        "classes": class_names,
        "class_to_idx": dict(sorted(class_to_idx.items(), key=lambda item: item[1])),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def load_class_names(path: str | Path) -> list[str]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Class mapping file not found: {source}")

    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return _validate_class_names(payload)

    if not isinstance(payload, dict):
        raise ValueError("Class mapping must be a JSON object or list.")

    if "classes" in payload:
        return _validate_class_names(payload["classes"])

    if "class_to_idx" in payload:
        return class_names_from_mapping(payload["class_to_idx"])

    if payload and all(isinstance(value, int) for value in payload.values()):
        return class_names_from_mapping(payload)

    raise ValueError("Class mapping must include 'classes' or 'class_to_idx'.")


def infer_class_map_path(checkpoint_path: str | Path) -> Path:
    checkpoint = Path(checkpoint_path)
    stem = checkpoint.stem
    if stem.endswith("_best"):
        stem = stem[: -len("_best")]
    return checkpoint.with_name(f"{stem}_classes.json")


def _validate_class_names(class_names: object) -> list[str]:
    if not isinstance(class_names, list) or not class_names:
        raise ValueError("'classes' must be a non-empty list.")
    if not all(isinstance(class_name, str) and class_name for class_name in class_names):
        raise ValueError("Class names must be non-empty strings.")
    return list(class_names)
