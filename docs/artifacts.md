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
- Keeps the main GitHub source repository clean while allowing the Space to fetch approved runtime artifacts.

Fallback future option: GitHub Release assets.

- Simple to attach versioned checkpoint/class-map files to a release.
- Useful if Hugging Face model hosting is not approved.
- The app would still need an approved download or mount strategy.

Not recommended for the main GitHub repository: committing checkpoint/class-map files directly. The local baseline checkpoint is small enough for a deliberate artifact handoff, but it is still generated output and should stay out of the source repo unless explicitly approved.

No artifact upload has been performed yet. Any upload to Hugging Face, GitHub Releases, object storage, or another host requires explicit approval.

See `docs/huggingface_artifact_handoff.md` for the local packaging helper, Hugging Face model repository placeholder, and upload templates.

## Docker Reminder

The Docker image does not include the checkpoint or class map. A future deployment must provide artifacts by one approved method:

- mount/copy files to the default paths
- set `PLANTGUARD_CHECKPOINT_PATH` and `PLANTGUARD_CLASS_MAP_PATH`
- add a startup download step from an approved artifact host

If artifacts are missing, the app must fail clearly and must not fake predictions.

## Hugging Face Spaces Runtime Plan

For the planned Docker Space, provide the baseline checkpoint and class map by one approved method:

1. Upload the files to a separate Hugging Face model repository.
2. Add an approved startup download or mount step for the Space.
3. Set `PLANTGUARD_CHECKPOINT_PATH` and `PLANTGUARD_CLASS_MAP_PATH` if the files are not placed at the default paths.
4. Validate the files with `python src\validate_artifacts.py --expected-classes 15` before enabling predictions.

The Space should expose the Flask app on port `7860`. `GET /healthz` should work without model artifacts, while `POST /predict` and `POST /api/predict` require the checkpoint and class map.
