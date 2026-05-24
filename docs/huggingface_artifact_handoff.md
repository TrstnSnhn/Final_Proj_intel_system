# Hugging Face Artifact Handoff

PlantGuard's planned Hugging Face Spaces deployment needs the baseline checkpoint and class map at runtime. These artifacts are generated locally and must stay out of the main GitHub source repository unless explicitly approved.

No Hugging Face model repository has been created, and no artifact upload has been performed.

## Recommended Model Repository

Use this confirmed model artifact repository ID:

```text
TrstnSnhn/plantguard-simplecnn-15class
```

The planned Hugging Face Space app host is:

```text
TrstnSnhn/PlantGuard
```

The Space repository is the app host. The model repository is the artifact host for the checkpoint and class map. The model repository has not been created by this phase unless you create it manually outside this repo, and no artifact upload has been performed.

Suggested visibility:

- Public: use when the baseline is ready to be shown as a portfolio demo and the artifact files are approved for public release.
- Private: use while testing artifact handoff, startup download behavior, and Space configuration.

## Files To Upload

Upload exactly this matched artifact pair:

```text
experiments/checkpoints/plantvillage_baseline_simple_cnn_best.pt
experiments/checkpoints/plantvillage_baseline_simple_cnn_classes.json
```

The checkpoint and class map must come from the same training run. Do not upload logs, generated evaluation JSON, datasets, `.env` files, tokens, or local cache folders.

## Prepare A Local Handoff Bundle

The local helper validates the artifacts, copies them into an ignored `artifact_handoff/` folder, and writes a manifest with file names, sizes, SHA256 hashes, class count, and sample class names.

```powershell
.\.venv\Scripts\python src\prepare_artifact_handoff.py --expected-classes 15
```

The helper does not upload anything and does not require a Hugging Face account or token.

Validate the source artifacts before preparing a handoff bundle:

```powershell
.\.venv\Scripts\python src\validate_artifacts.py --expected-classes 15
```

Expected local output:

```text
artifact_handoff/
  manifest.json
  plantvillage_baseline_simple_cnn_best.pt
  plantvillage_baseline_simple_cnn_classes.json
```

`artifact_handoff/` is ignored by git and should not be committed.

## Model Repository README Outline

When the model repository is created, its README/model card should include:

- Model name: `PlantGuard SimpleCNN 15-class baseline`
- Intended use: educational plant leaf disease screening demo
- Dataset variant: local 15-class PlantVillage variant
- Source project: PlantGuard Flask/PyTorch demo
- Required files:
  - `plantvillage_baseline_simple_cnn_best.pt`
  - `plantvillage_baseline_simple_cnn_classes.json`
- Baseline status: workflow baseline, not final model quality
- Known limitations: lab-curated data, not a definitive diagnosis, deployment artifact for demo use
- Expected consuming app environment variables:
  - `PLANTGUARD_CHECKPOINT_PATH`
  - `PLANTGUARD_CLASS_MAP_PATH`

## Manual Upload Option

After approval, create the Hugging Face model repository through the Hugging Face UI, then upload only the checkpoint and class map from `artifact_handoff/`.

Do not paste or commit tokens. Authentication must stay outside this repository.

## CLI Upload Template

Do not run this until the repository ID and upload approval are confirmed. Verify the current Hugging Face CLI syntax before upload.

```powershell
huggingface-cli upload TrstnSnhn/plantguard-simplecnn-15class `
  artifact_handoff\plantvillage_baseline_simple_cnn_best.pt `
  plantvillage_baseline_simple_cnn_best.pt `
  --repo-type model

huggingface-cli upload TrstnSnhn/plantguard-simplecnn-15class `
  artifact_handoff\plantvillage_baseline_simple_cnn_classes.json `
  plantvillage_baseline_simple_cnn_classes.json `
  --repo-type model
```

Python API template, also for later approval only:

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    repo_id="TrstnSnhn/plantguard-simplecnn-15class",
    repo_type="model",
    path_or_fileobj="artifact_handoff/plantvillage_baseline_simple_cnn_best.pt",
    path_in_repo="plantvillage_baseline_simple_cnn_best.pt",
)
api.upload_file(
    repo_id="TrstnSnhn/plantguard-simplecnn-15class",
    repo_type="model",
    path_or_fileobj="artifact_handoff/plantvillage_baseline_simple_cnn_classes.json",
    path_in_repo="plantvillage_baseline_simple_cnn_classes.json",
)
```

## Future Space Configuration

The Hugging Face Space must provide the artifacts by download, mount, or placement at runtime. If the files are not placed at the default app paths, configure:

```text
PLANTGUARD_CHECKPOINT_PATH
PLANTGUARD_CLASS_MAP_PATH
```

The Docker image alone is not enough for predictions. `GET /healthz` can work without artifacts, but `POST /predict` and `POST /api/predict` require the checkpoint and class map.
