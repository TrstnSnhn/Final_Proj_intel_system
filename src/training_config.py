from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUPPORTED_ARCHITECTURES = {"simple_cnn", "resnet18"}
UNSUPPORTED_ARCHITECTURES = {
    "sklearn_rf": (
        "sklearn_rf is not supported by src/train.py. It is a planned classical "
        "ML baseline and needs a separate feature-extraction trainer before it "
        "can run."
    )
}
SUPPORTED_OPTIMIZERS = {"adam", "sgd"}


class ConfigError(ValueError):
    """Raised when an experiment config cannot be run safely."""


@dataclass(frozen=True)
class OptimizerSpec:
    name: str
    lr: float
    options: dict[str, Any]


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ConfigError("PyYAML is required to load experiment configs.") from exc

    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ConfigError(f"Config file must contain a YAML mapping: {config_path}")
    return cfg


def validate_training_config(cfg: dict[str, Any]) -> None:
    experiment_name = cfg.get("experiment_name")
    if not isinstance(experiment_name, str) or not experiment_name.strip():
        raise ConfigError("Config must define a non-empty experiment_name.")

    model_cfg = _require_mapping(cfg, "model")
    training_cfg = _require_mapping(cfg, "training")
    architecture = model_cfg.get("architecture")

    if architecture in UNSUPPORTED_ARCHITECTURES:
        raise ConfigError(UNSUPPORTED_ARCHITECTURES[architecture])
    if architecture not in SUPPORTED_ARCHITECTURES:
        supported = ", ".join(sorted(SUPPORTED_ARCHITECTURES))
        raise ConfigError(f"Unsupported architecture: {architecture}. Supported: {supported}.")

    num_classes = model_cfg.get("num_classes", 38)
    if not isinstance(num_classes, int) or num_classes <= 1:
        raise ConfigError("model.num_classes must be an integer greater than 1.")

    optimizer = str(training_cfg.get("optimizer", "adam")).lower()
    if optimizer not in SUPPORTED_OPTIMIZERS:
        supported = ", ".join(sorted(SUPPORTED_OPTIMIZERS))
        raise ConfigError(f"Unsupported optimizer: {optimizer}. Supported: {supported}.")

    resolve_learning_rate(cfg, stage="unfrozen")
    if architecture == "resnet18":
        resolve_learning_rate(cfg, stage="frozen")


def resolve_learning_rate(cfg: dict[str, Any], stage: str) -> float:
    training_cfg = _require_mapping(cfg, "training")
    if stage == "frozen":
        value = training_cfg.get("lr_frozen", training_cfg.get("lr_unfrozen", 1e-3))
    elif stage in {"unfrozen", "default"}:
        value = training_cfg.get("lr_unfrozen", training_cfg.get("lr_frozen", 1e-3))
    else:
        raise ConfigError(f"Unknown training stage: {stage}")

    try:
        lr = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Learning rate for stage '{stage}' must be numeric.") from exc

    if lr <= 0:
        raise ConfigError(f"Learning rate for stage '{stage}' must be greater than 0.")
    return lr


def resolve_optimizer_spec(cfg: dict[str, Any], stage: str) -> OptimizerSpec:
    training_cfg = _require_mapping(cfg, "training")
    optimizer = str(training_cfg.get("optimizer", "adam")).lower()
    if optimizer not in SUPPORTED_OPTIMIZERS:
        supported = ", ".join(sorted(SUPPORTED_OPTIMIZERS))
        raise ConfigError(f"Unsupported optimizer: {optimizer}. Supported: {supported}.")

    options: dict[str, Any] = {}
    if "weight_decay" in training_cfg:
        options["weight_decay"] = float(training_cfg["weight_decay"])

    if optimizer == "sgd":
        options["momentum"] = float(training_cfg.get("momentum", 0.9))
        if "nesterov" in training_cfg:
            options["nesterov"] = bool(training_cfg["nesterov"])

    return OptimizerSpec(name=optimizer, lr=resolve_learning_rate(cfg, stage), options=options)


def _require_mapping(cfg: dict[str, Any], key: str) -> dict[str, Any]:
    value = cfg.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"Config must define a '{key}' mapping.")
    return value
