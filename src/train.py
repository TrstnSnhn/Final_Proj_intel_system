from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from config import CHECKPOINTS_DIR, LOGS_DIR
from data_pipeline import get_dataloaders
from models.cnn_from_scratch import SimpleCNN
from models.resnet_finetune import ResNet18Classifier
from utils.logger import CSVLogger
from utils.metrics import compute_accuracy, compute_f1
from utils.seed import set_seed


def build_model(cfg):
    arch = cfg["model"]["architecture"]
    num_classes = cfg["model"].get("num_classes", 38)
    if arch == "simple_cnn":
        return SimpleCNN(num_classes)
    if arch == "resnet18":
        return ResNet18Classifier(num_classes, cfg["model"].get("pretrained", True))
    raise ValueError(f"Unsupported architecture: {arch}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses, preds, labels = [], [], []
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


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    losses, preds, labels = [], [], []
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

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["training"].get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _ = get_dataloaders(
        batch_size=cfg["data"].get("batch_size", 32),
        num_workers=cfg["data"].get("num_workers", 2),
        augmentation=cfg["data"].get("augmentation", True),
    )

    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"].get("lr_unfrozen", 1e-3))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=cfg["training"].get("scheduler_patience", 3),
        factor=cfg["training"].get("scheduler_factor", 0.1),
    )

    logger = CSVLogger(LOGS_DIR / f"{cfg['experiment_name']}.csv")
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience = cfg["training"].get("early_stopping_patience", 5)
    stale = 0
    total_epochs = cfg["model"].get("epochs", cfg["model"].get("freeze_epochs", 5) + cfg["model"].get("unfreeze_epochs", 15))

    if cfg["model"]["architecture"] == "resnet18":
        model.freeze_backbone()

    start = time.time()
    for epoch in range(1, total_epochs + 1):
        if cfg["model"]["architecture"] == "resnet18" and epoch == cfg["model"].get("freeze_epochs", 5) + 1:
            model.unfreeze_backbone()
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"].get("lr_unfrozen", 1e-4))

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
            "time_sec": time.time() - start,
        }
        logger.log(row)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stale = 0
            torch.save(model.state_dict(), CHECKPOINTS_DIR / f"{cfg['experiment_name']}_best.pt")
        else:
            stale += 1
            if stale >= patience:
                print("Early stopping triggered")
                break

    logger.close()
    print(f"Training complete. Best val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    main()
