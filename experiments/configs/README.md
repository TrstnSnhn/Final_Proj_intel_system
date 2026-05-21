# Experiment Config Guide

PlantGuard keeps experiment configs in YAML so dataset, model, and optimizer choices are explicit and reproducible. Running tests does not require the real dataset, checkpoints, or full training.

## Verified Local Configs

These configs are verified against the current local 15-class PlantVillage dataset variant:

- `plantvillage_smoke.yaml`: one-epoch SimpleCNN workflow check. Use it to verify that the real split can train, write a checkpoint, write a class map, evaluate, and run inference. It is not a model-quality run.
- `plantvillage_baseline_simple_cnn.yaml`: conservative SimpleCNN baseline for the local 15-class PlantVillage split. The Phase 2F CPU run produced the current documented baseline workflow result.

Both configs use `pretrained: false`, `num_workers: 0`, and `data/splits`, so they are safe for CPU/local validation and do not trigger pretrained weight downloads.

## Synthetic Smoke Config

- `smoke_test.yaml`: tiny generated two-class dataset workflow check using `data/smoke/splits`. It proves the pipeline plumbing without PlantVillage.

## Legacy and Future-Work Configs

These configs are retained from the original/future experiment plan and are not verified against the current local 15-class dataset:

- `baseline_scratch.yaml`: SimpleCNN, 38 classes, longer training plan.
- `resnet18_default.yaml`: ResNet18, 38 classes, uses pretrained weights.
- `ablation_lr.yaml`: ResNet18, 38 classes, uses pretrained weights.
- `ablation_no_augment.yaml`: ResNet18, 38 classes, uses pretrained weights.
- `ablation_optimizer.yaml`: ResNet18, 38 classes, uses pretrained weights.
- `baseline_sklearn.yaml`: planned sklearn random-forest baseline. It is marked `not_implemented` and is rejected by `src/train.py`.

ResNet configs may trigger pretrained weight downloads when training constructs the model with `pretrained: true`. Do not run them in offline or CPU-only validation unless they are first adjusted and reviewed.

## Artifact Conventions

Generated artifacts are recreated locally and ignored by git:

- Checkpoint: `experiments/checkpoints/<experiment_name>_best.pt`
- Class mapping: `experiments/checkpoints/<experiment_name>_classes.json`
- Training log: `experiments/logs/<experiment_name>.csv`
- Evaluation summary: `experiments/results/<experiment_name>_eval_summary.json`

Do not commit generated checkpoints, logs, metrics, result JSON, datasets, caches, or virtual environments.
