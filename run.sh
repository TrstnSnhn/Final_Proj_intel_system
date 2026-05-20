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

echo "=== Step 8: Evaluation Requires a Trained Checkpoint ==="
echo "After training, run for example:"
echo "python src/eval.py --checkpoint experiments/checkpoints/resnet18_finetune_default_best.pt --data-dir data/splits/test --class-map experiments/checkpoints/resnet18_finetune_default_classes.json"

echo "=== Step 9: Grad-CAM Planned for Phase 2B ==="
echo "src/gradcam.py is intentionally a non-generating placeholder until real checkpoint-based Grad-CAM is implemented."

echo "=== Step 10: Visualization Planned After Real Evaluation ==="
echo "src/utils/visualization.py is intentionally a non-generating placeholder until metrics-backed plots are implemented."

echo "=== Done! Results in experiments/results/ ==="
