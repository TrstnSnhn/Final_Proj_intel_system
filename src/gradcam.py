from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=12)
    args = parser.parse_args()

    out = Path("experiments/results")
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(3, 4, figsize=(10, 7))
    for i, axi in enumerate(ax.flat):
        axi.text(0.5, 0.5, f"Sample {i+1}", ha="center", va="center")
        axi.axis("off")
    fig.suptitle(f"Grad-CAM placeholders ({args.num_samples} samples)")
    plt.tight_layout()
    plt.savefig(out / "gradcam_samples.png", dpi=150)
    print("Saved experiments/results/gradcam_samples.png")


if __name__ == "__main__":
    main()
