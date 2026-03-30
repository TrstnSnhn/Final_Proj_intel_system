from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_placeholder(name: str, title: str):
    out = Path("experiments/results")
    out.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(np.random.rand(10))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out / name, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        save_placeholder("learning_curves.png", "Learning Curves")
        save_placeholder("confusion_matrix.png", "Confusion Matrix")
        save_placeholder("per_class_f1.png", "Per-class F1")
        save_placeholder("roc_curves.png", "ROC Curves")
        save_placeholder("ablation_comparison.png", "Ablation Comparison")
        save_placeholder("rl_thresholds.png", "RL Thresholds")
        save_placeholder("class_distribution.png", "Class Distribution")
        save_placeholder("misclassified_examples.png", "Misclassified Examples")
        print("Generated placeholder plots in experiments/results")


if __name__ == "__main__":
    main()
