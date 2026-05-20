from __future__ import annotations

import argparse
import time
from collections.abc import Iterable

try:
    from .class_mapping import save_class_mapping
    from .training_config import (
        ConfigError,
        load_yaml_config,
        resolve_optimizer_spec,
        validate_training_config,
    )
except ImportError:  # pragma: no cover - supports running as python src/train.py
    from class_mapping import save_class_mapping
    from training_config import (
        ConfigError,
        load_yaml_config,
        resolve_optimizer_spec,
        validate_training_config,
    )


def build_model(cfg):
    arch = cfg["model"]["architecture"]
    num_classes = cfg["model"].get("num_classes", 38)
    if arch == "simple_cnn":
        try:
            from .models.cnn_from_scratch import SimpleCNN
        except ImportError:  # pragma: no cover
            from models.cnn_from_scratch import SimpleCNN

        return SimpleCNN(num_classes)
    if arch == "resnet18":
        try:
            from .models.resnet_finetune import ResNet18Classifier
        except ImportError:  # pragma: no cover
            from models.resnet_finetune import ResNet18Classifier

        return ResNet18Classifier(num_classes, cfg["model"].get("pretrained", True))
    raise ValueError(f"Unsupported architecture: {arch}")


def build_optimizer(parameters: Iterable, cfg, stage: str):
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for training.") from exc

    params = list(parameters)
    if not params:
        raise ConfigError(f"No trainable parameters found for training stage '{stage}'.")

    spec = resolve_optimizer_spec(cfg, stage)
    if spec.name == "adam":
        return torch.optim.Adam(params, lr=spec.lr, **spec.options)
    if spec.name == "sgd":
        return torch.optim.SGD(params, lr=spec.lr, **spec.options)
    raise ConfigError(f"Unsupported optimizer: {spec.name}")


def build_scheduler(optimizer, cfg):
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for training.") from exc

    training_cfg = cfg["training"]
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=training_cfg.get("scheduler_patience", 3),
        factor=training_cfg.get("scheduler_factor", 0.1),
    )


def trainable_parameters(model):
    return (parameter for parameter in model.parameters() if parameter.requires_grad)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses, preds, labels = [], [], []
    from tqdm import tqdm

    try:
        from .utils.metrics import compute_accuracy
    except ImportError:  # pragma: no cover
        from utils.metrics import compute_accuracy

    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        preds.extend(out.argmax(1).detach().cpu().tolist())
        labels.extend(y.detach().cpu().tolist())
    return sum(losses) / len(losses), compute_accuracy(preds, labels)


def validate(model, loader, criterion, device):
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for validation.") from exc

    try:
        from .utils.metrics import compute_accuracy, compute_f1
    except ImportError:  # pragma: no cover
        from utils.metrics import compute_accuracy, compute_f1

    model.eval()
    losses, preds, labels = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            losses.append(loss.item())
            preds.extend(out.argmax(1).detach().cpu().tolist())
            labels.extend(y.detach().cpu().tolist())
    return sum(losses) / len(losses), compute_accuracy(preds, labels), compute_f1(preds, labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    validate_training_config(cfg)

    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for training.") from exc

    try:
        from .config import CHECKPOINTS_DIR, LOGS_DIR
        from .data_pipeline import get_dataloaders
        from .utils.logger import CSVLogger
        from .utils.seed import set_seed
    except ImportError:  # pragma: no cover - supports running as python src/train.py
        from config import CHECKPOINTS_DIR, LOGS_DIR
        from data_pipeline import get_dataloaders
        from utils.logger import CSVLogger
        from utils.seed import set_seed

    set_seed(cfg["training"].get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _ = get_dataloaders(
        batch_size=cfg["data"].get("batch_size", 32),
        num_workers=cfg["data"].get("num_workers", 2),
        augmentation=cfg["data"].get("augmentation", True),
        split_dir=cfg["data"].get("split_dir", "data/splits"),
    )

    model = build_model(cfg).to(device)
    stage = "unfrozen"
    if cfg["model"]["architecture"] == "resnet18":
        model.freeze_backbone()
        stage = "frozen"

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(trainable_parameters(model), cfg, stage=stage)
    scheduler = build_scheduler(optimizer, cfg)

    logger = CSVLogger(LOGS_DIR / f"{cfg['experiment_name']}.csv")
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    save_class_mapping(
        train_loader.dataset.class_to_idx,
        CHECKPOINTS_DIR / f"{cfg['experiment_name']}_classes.json",
    )

    best_val_loss = float("inf")
    patience = cfg["training"].get("early_stopping_patience", 5)
    stale = 0
    total_epochs = cfg["model"].get(
        "epochs",
        cfg["model"].get("freeze_epochs", 5) + cfg["model"].get("unfreeze_epochs", 15),
    )

    start = time.time()
    for epoch in range(1, total_epochs + 1):
        if cfg["model"]["architecture"] == "resnet18" and epoch == cfg["model"].get("freeze_epochs", 5) + 1:
            model.unfreeze_backbone()
            stage = "unfrozen"
            optimizer = build_optimizer(trainable_parameters(model), cfg, stage=stage)
            scheduler = build_scheduler(optimizer, cfg)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
            "lr": optimizer.param_groups[0]["lr"],
            "optimizer": cfg["training"].get("optimizer", "adam"),
            "stage": stage,
            "time_sec": time.time() - start,
        }
        logger.log(row)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stale = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "architecture": cfg["model"]["architecture"],
                    "num_classes": cfg["model"].get("num_classes", 38),
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                CHECKPOINTS_DIR / f"{cfg['experiment_name']}_best.pt",
            )
        else:
            stale += 1
            if stale >= patience:
                print("Early stopping triggered")
                break

    logger.close()
    print(f"Training complete. Best val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    try:
        main()
    except (ConfigError, RuntimeError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
