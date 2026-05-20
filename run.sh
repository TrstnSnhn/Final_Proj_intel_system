#!/usr/bin/env bash
set -e

echo "=== Step 1: Download Data ==="
python data/get_data.py

echo "=== Step 2: Preprocess & Split ==="
python src/data_pipeline.py --action split --seed 42

echo "=== Step 3: Skipping Classical sklearn Baseline ==="
echo "baseline_sklearn.yaml is documented but not implemented by src/train.py yet."

echo "=== Step 4: Train From-Scratch CNN ==="
python src/train.py --config experiments/configs/baseline_scratch.yaml

echo "=== Step 5: Train ResNet18 Fine-Tune ==="
python src/train.py --config experiments/configs/resnet18_default.yaml

echo "=== Step 6: Train NLP Classifier ==="
python src/nlp_pipeline.py --action train

echo "=== Step 7: Run RL Threshold Agent ==="
python src/rl_agent.py --episodes 300

echo "=== Step 8: Generate Placeholder Evaluation Summary ==="
python src/eval.py --all

echo "=== Step 9: Generate Placeholder Grad-CAM Figure ==="
python src/gradcam.py --num-samples 12

echo "=== Step 10: Generate Placeholder Plots ==="
python src/utils/visualization.py --all

echo "=== Done! Results in experiments/results/ ==="
