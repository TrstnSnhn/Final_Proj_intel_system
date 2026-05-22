# PlantGuard Model Artifacts

PlantGuard's Flask demo needs a trained checkpoint and a class mapping JSON at runtime. These files are generated locally and are intentionally ignored by git.

## Required Runtime Artifacts

Current baseline artifacts:

```text
experiments/checkpoints/plantvillage_baseline_simple_cnn_best.pt
experiments/checkpoints/plantvillage_baseline_simple_cnn_classes.json
```

The checkpoint contains the SimpleCNN model weights. The class map defines the label order used by inference. Both must come from the same training run.

## Why They Are Not Committed

Model artifacts can be large, are generated outputs, and may need a separate release or hosting policy. Keeping them out of git prevents accidental repository bloat and avoids mixing source code with runtime artifacts.

Do not commit:

- `experiments/checkpoints/`
- `*.pt`
- `*.pth`
- `*.ckpt`
- generated logs, metrics, or evaluation JSON
- dataset files
- secrets, tokens, or `.env` files

## Local Default Paths

The Flask app and artifact validator use these default paths:

```text
experiments/checkpoints/plantvillage_baseline_simple_cnn_best.pt
experiments/checkpoints/plantvillage_baseline_simple_cnn_classes.json
```

For local development, recreate or place artifacts at those paths.

## Environment Variable Overrides

For deployment or alternate local layouts, set:

```text
PLANTGUARD_CHECKPOINT_PATH
PLANTGUARD_CLASS_MAP_PATH
```

Configure these through the hosting platform or shell session. Do not create or commit `.env` files unless a future phase explicitly adds a safe `.env.example` template.

## Recreate The Baseline Artifacts Locally

After the PlantVillage dataset has been placed, split, and validated, run:

```powershell
python src\train.py --config experiments\configs\plantvillage_baseline_simple_cnn.yaml
```

Expected generated files:

```text
experiments/checkpoints/plantvillage_baseline_simple_cnn_best.pt
experiments/checkpoints/plantvillage_baseline_simple_cnn_classes.json
```

This training command is not part of deployment validation and should not be run unless a training phase is explicitly approved.

## Validate Artifacts Locally

Validate the default local baseline artifacts without loading the full model:

```powershell
python src\validate_artifacts.py --expected-classes 15
```

Validate explicit paths:

```powershell
python src\validate_artifacts.py `
  --checkpoint path\to\model.pt `
  --class-map path\to\classes.json `
  --expected-classes 15
```

The validator checks:

- checkpoint file exists
- class map file exists
- class map JSON is valid
- class map contains at least one class
- optional expected class count matches

It does not require the dataset, training, Docker, browser automation, or loading the full model.

## Future Hosting Options

Preferred future option: Hugging Face model repository.

- Fits the planned Hugging Face Spaces Docker deployment path.
- Keeps model artifacts separate from application source code.
- Lets the app download or mount artifacts in a controlled deployment step.

Fallback future option: GitHub Release assets.

- Simple to attach versioned checkpoint/class-map files to a release.
- Useful if Hugging Face model hosting is not approved.
- The app would still need an approved download or mount strategy.

No artifact upload has been performed yet. Any upload to Hugging Face, GitHub Releases, object storage, or another host requires explicit approval.

## Docker Reminder

The Docker image does not include the checkpoint or class map. A future deployment must provide artifacts by one approved method:

- mount/copy files to the default paths
- set `PLANTGUARD_CHECKPOINT_PATH` and `PLANTGUARD_CLASS_MAP_PATH`
- add a startup download step from an approved artifact host

If artifacts are missing, the app must fail clearly and must not fake predictions.
