# Model Card: PlantGuard AI Plant Disease Classifier

## Model Details
- Current baseline: SimpleCNN trained from scratch
- Framework: PyTorch
- Dataset variant: local PlantVillage subset with 15 classes
- Dataset split: 14,440 train, 3,097 validation, 3,101 test images
- Config: `experiments/configs/plantvillage_baseline_simple_cnn.yaml`
- Version: 0.2-baseline-readiness

The older ResNet18 configs are still present for future experiments, but the current verified baseline uses SimpleCNN to avoid pretrained weight downloads and keep the workflow CPU-safe.

See `experiments/configs/README.md` for the current distinction between verified 15-class configs and legacy/future-work 38-class configs.

## Intended Use
PlantGuard is an educational screening tool for plant disease classification from leaf photos. It is intended for a portfolio-ready Intelligent Systems project, not production agricultural diagnosis.

## Out of Scope
- Definitive crop or disease diagnosis
- Field-ready agronomy recommendations
- Safety-critical or commercial decision-making
- Claims about real-world generalization

## Evaluation Snapshot
The Phase 2F baseline run used a two-epoch CPU SimpleCNN training pass on the local 15-class PlantVillage split.

- Test samples: 3,101
- Correct predictions: 2,525
- Accuracy: 81.43%

This result is a baseline workflow check only. It confirms that dataset validation, training, checkpoint creation, evaluation, and inference connect end to end. It should not be presented as final model quality.

## Artifacts
Generated artifacts stay local and are ignored by git:

- `experiments/checkpoints/plantvillage_baseline_simple_cnn_best.pt`
- `experiments/checkpoints/plantvillage_baseline_simple_cnn_classes.json`
- `experiments/logs/plantvillage_baseline_simple_cnn.csv`
- `experiments/results/plantvillage_baseline_eval_summary.json`

## Caveats
- PlantVillage is lab-curated and may not represent real field images.
- The local dataset variant has 15 classes, while older configs still target the full 38-class PlantVillage setup.
- The current baseline is intentionally conservative and not tuned.
- ResNet and Grad-CAM work remain future phases.
- A future web demo should use a locally recreated checkpoint or an explicit artifact handoff; no checkpoint is committed in this repository.
