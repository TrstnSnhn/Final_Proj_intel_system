#!/usr/bin/env bash
set -e

echo "=== Step 0: Environment Check ==="
python - <<'PY'
import importlib
import subprocess
import sys

required_modules = [
    "torch",
    "torchvision",
    "numpy",
    "pandas",
    "sklearn",
    "matplotlib",
    "seaborn",
    "PIL",
    "yaml",
    "tqdm",
    "kagglehub",
]

missing = []
for name in required_modules:
    try:
        importlib.import_module(name)
    except Exception:
        missing.append(name)

if missing:
    print(f"Missing Python packages detected: {', '.join(missing)}")
    print("Installing dependencies from requirements.txt...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
else:
    print("All required Python packages are available.")
PY

echo "=== Step 1: Download Data ==="
python data/get_data.py

echo "=== Step 2: Preprocess & Split ==="
python src/data_pipeline.py --action split --seed 42

echo "=== Step 3: Train From-Scratch CNN ==="
python src/train.py --config experiments/configs/baseline_scratch.yaml

echo "=== Step 4: Train ResNet18 Fine-Tune ==="
python src/train.py --config experiments/configs/resnet18_default.yaml

echo "=== Step 5: Train NLP Classifier ==="
python src/nlp_pipeline.py --action train

echo "=== Step 6: Run RL Threshold Agent ==="
python src/rl_agent.py --episodes 300

echo "=== Step 7: Evaluate All Models ==="
python src/eval.py --all

echo "=== Step 8: Generate Grad-CAM ==="
python src/gradcam.py --num-samples 12

echo "=== Step 9: Generate All Plots ==="
python src/utils/visualization.py --all

echo "=== Done! Results in experiments/results/ ==="
