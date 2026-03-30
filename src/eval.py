from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--model", default="resnet18")
    args = parser.parse_args()

    results_path = Path("experiments/results/eval_summary.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "status": "placeholder_evaluation",
        "all_models": args.all,
        "selected_model": args.model,
        "note": "Run after training checkpoints are available.",
    }
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved placeholder evaluation summary to {results_path}")


if __name__ == "__main__":
    main()
