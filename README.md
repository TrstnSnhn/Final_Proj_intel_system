# PlantGuard AI

PlantGuard AI is an Intelligent Systems finals project for plant disease screening from leaf images. The project is being revived from a school-project scaffold into a reliable, portfolio-ready application.

## Current Status

Current work focuses on turning the validated machine-learning workflow and local Flask demo into a deployment-ready portfolio project.

What exists now:

- PyTorch image-classification training scaffold.
- Simple CNN and ResNet18 model definitions.
- PlantVillage dataset download and split scripts.
- Dataset validation CLI.
- Experiment YAML configs.
- Checkpoint and class mapping save/load support.
- Real checkpoint-based evaluation CLI.
- Minimal Flask web demo for local image upload, preview, and prediction.
- Basic NLP and RL project scaffolding.
- Grad-CAM and visualization placeholders that fail honestly until implemented.
- A command-line inference entrypoint that works once a trained checkpoint and class mapping exist.

What is not implemented yet:

- No deployed web UI.
- No mobile app.
- No desktop app.
- No live deployment or approved model artifact host.
- No committed trained checkpoint.
- No real Grad-CAM output tied to a trained checkpoint.

## Runtime Target

PlantGuard targets **Python 3.11**.

The ML dependencies are intentionally pinned conservatively for Phase 1. Do not broadly upgrade PyTorch, Torchvision, or related ML packages without testing compatibility first.

## Tech Stack

- Python 3.11
- PyTorch
- Torchvision
- scikit-learn
- pandas and NumPy
- matplotlib and seaborn
- Pillow
- PyYAML
- Flask
- KaggleHub
- Jupyter notebooks

## Setup

Create and activate a Python 3.11 virtual environment:

If `py -3.11 --version` fails on Windows, install Python 3.11 first:

```powershell
winget install --id Python.Python.3.11 -e --source winget
py -3.11 --version
```

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional development tools:

```powershell
python -m pip install -r requirements-dev.txt
```

On macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional development tools:

```bash
python -m pip install -r requirements-dev.txt
```

## Safe Validation

These commands do not download the dataset, train the full model, or deploy anything:

```powershell
python --version
python -m py_compile src\*.py src\models\*.py src\utils\*.py data\get_data.py
python -m unittest discover -s tests
python src\validate_dataset.py --help
python src\infer.py --help
python src\train.py --help
python src\validate_artifacts.py --help
python src\eval.py --help
```

If Ruff is installed:

```powershell
ruff check .
```

## Dataset

The intended dataset is PlantVillage. The primary configured source is the Kaggle dataset `abdallahalidev/plantvillage-dataset`, accessed through KaggleHub. Kaggle credentials, if used, must be configured outside this repository. Do not commit `kaggle.json`, `.env`, API tokens, downloaded data, split data, checkpoints, or generated results.

Dataset download is intentionally not part of the safe validation flow. Preview the planned KaggleHub source and local target without network access or writes:

```powershell
python data\get_data.py --dry-run
```

If KaggleHub is configured and you intentionally want to download and normalize the dataset into PlantGuard's raw ImageFolder layout, run:

```powershell
python data\get_data.py
```

If you manually download or extract PlantVillage first, normalize the extracted archive into the expected raw folder with:

```powershell
python data\get_data.py --source-dir C:\path\to\extracted\plantvillage --overwrite
```

The script looks for the Kaggle `plantvillage dataset/color/` folder when present, because PlantGuard trains on RGB leaf images. It writes the normalized class folders under `data/raw/plantvillage/`, which is ignored by git.

Expected raw dataset layout:

```text
data/raw/plantvillage/
  Apple___healthy/
    image-1.jpg
  Tomato___Early_blight/
    image-2.jpg
```

If placing files manually, copy the class folders themselves into `data/raw/plantvillage/`. Do not leave an extra wrapper folder such as `data/raw/plantvillage/plantvillage dataset/color/`, because training expects class names directly under the raw root.

After splitting, training expects:

```text
data/splits/
  train/
    <class-name>/
  val/
    <class-name>/
  test/
    <class-name>/
```

Dataset folders are ignored by git. Keep `data/raw/`, `data/splits/`, `data/smoke/`, and any large generated artifacts local.

Validate the raw dataset folder without training:

```powershell
python src\validate_dataset.py data\raw\plantvillage --layout raw
```

Create the train/validation/test split:

```powershell
python src\data_pipeline.py --action split --raw-dir data\raw\plantvillage --split-dir data\splits --seed 42 --overwrite
```

Validate split folders:

```powershell
python src\validate_dataset.py data\splits --layout split
```

## Training

Training requires the dataset to exist under `data/raw/plantvillage/` and then be split into train/validation/test folders.

See `experiments/configs/README.md` before choosing a config. The verified local configs are `plantvillage_smoke.yaml` and `plantvillage_baseline_simple_cnn.yaml`; older 38-class and ResNet configs are retained as legacy/future-work configs and are not verified against the current local 15-class dataset.

Example commands:

```powershell
python src\data_pipeline.py --action split --raw-dir data\raw\plantvillage --split-dir data\splits --seed 42 --overwrite
python src\train.py --config experiments\configs\plantvillage_baseline_simple_cnn.yaml
```

The split command copies supported image files only (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`) and skips harmless non-image files in class folders. Its printed split counts match the supported-image counts used by validation, training, and evaluation. If `data/splits/` was created before this behavior existed, rerun the split command with `--overwrite`.

Trained checkpoints and class mappings are written to `experiments/checkpoints/`. That folder is ignored by git because model artifacts are generated runtime files and should not be committed by accident. See `docs/artifacts.md` for the artifact handoff and validation workflow.

The `baseline_sklearn.yaml` config is marked as not implemented. It documents a planned classical ML baseline, but `src/train.py` is a PyTorch trainer and does not run `sklearn_rf` yet.

For real-data pipeline preparation without a full training run, `experiments/configs/plantvillage_smoke.yaml` is available. It is a one-epoch CPU-friendly SimpleCNN smoke config using `data/splits`. It is only for verifying that real PlantVillage splits can feed training and produce artifacts; it does not demonstrate model quality.

For the local 15-class PlantVillage variant used in Phase 2F, use the conservative SimpleCNN baseline config:

```powershell
python src\train.py --config experiments\configs\plantvillage_baseline_simple_cnn.yaml
```

This baseline uses `data/splits`, 15 classes, 2 epochs, batch size 16, and no pretrained weights. The verified local dataset contains 20,638 supported images split into 14,440 train, 3,097 validation, and 3,101 test images. In the Phase 2F CPU baseline run, evaluation reported 2,525 correct out of 3,101 test images, or 81.43% accuracy. Treat this as a baseline workflow result only, not final model quality.

Generated checkpoints, logs, and metrics are ignored by git:

```text
experiments/checkpoints/<experiment_name>_best.pt
experiments/checkpoints/<experiment_name>_classes.json
experiments/logs/<experiment_name>.csv
experiments/results/<experiment_name>_eval_summary.json
```

## Evaluation

Evaluate a trained checkpoint against a class-folder split:

```powershell
python src\eval.py `
  --checkpoint experiments\checkpoints\resnet18_finetune_default_best.pt `
  --class-map experiments\checkpoints\resnet18_finetune_default_classes.json `
  --data-dir data\splits\test `
  --architecture resnet18 `
  --output experiments\results\eval_summary.json
```

Evaluation currently reports total accuracy and per-class support/correct counts. It fails clearly if the checkpoint, class mapping, or dataset split is missing.

Evaluate the local SimpleCNN baseline with:

```powershell
python eval.py `
  --checkpoint experiments\checkpoints\plantvillage_baseline_simple_cnn_best.pt `
  --class-map experiments\checkpoints\plantvillage_baseline_simple_cnn_classes.json `
  --data-dir data\splits\test `
  --architecture simple_cnn `
  --batch-size 16 `
  --num-workers 0 `
  --output experiments\results\plantvillage_baseline_eval_summary.json
```

Summarize a generated training log and evaluation summary without loading the dataset or model:

```powershell
python src\summarize_experiment.py `
  --eval-summary experiments\results\plantvillage_baseline_eval_summary.json `
  --training-log experiments\logs\plantvillage_baseline_simple_cnn.csv
```

## Tiny Smoke Workflow

The smoke workflow proves the technical pipeline without downloading PlantVillage or training a useful model. It generates a tiny two-class image dataset under `data/smoke/`, trains for one CPU-safe epoch, writes a checkpoint and class mapping, evaluates the checkpoint, and runs one inference command.

Create and validate the smoke dataset:

```powershell
python src\create_smoke_dataset.py --output data\smoke\splits --overwrite
python src\validate_dataset.py data\smoke\splits --layout split
```

Run the smoke training config:

```powershell
python src\train.py --config experiments\configs\smoke_test.yaml
```

Evaluate and infer:

```powershell
python eval.py `
  --checkpoint experiments\checkpoints\smoke_test_best.pt `
  --class-map experiments\checkpoints\smoke_test_classes.json `
  --data-dir data\smoke\splits\test `
  --architecture simple_cnn `
  --batch-size 2 `
  --num-workers 0 `
  --output experiments\results\smoke_eval_summary.json

python src\infer.py `
  --checkpoint experiments\checkpoints\smoke_test_best.pt `
  --class-map experiments\checkpoints\smoke_test_classes.json `
  --image data\smoke\splits\test\Smoke___green_leaf\Smoke___green_leaf_00.png `
  --architecture simple_cnn `
  --top-k 2
```

This smoke workflow only proves that the dataset, training, checkpoint, evaluation, and inference plumbing works. It does not prove model quality.

## Inference

Once a trained checkpoint exists, use:

```powershell
python src\infer.py `
  --checkpoint experiments\checkpoints\resnet18_finetune_default_best.pt `
  --class-map experiments\checkpoints\resnet18_finetune_default_classes.json `
  --image path\to\leaf.jpg `
  --architecture resnet18 `
  --top-k 3
```

If `--class-map` is omitted, the script looks for a sibling `*_classes.json` file next to the checkpoint.

If no checkpoint exists, inference fails honestly with a clear error. It does not fake predictions.

For the local SimpleCNN baseline, use:

```powershell
python src\infer.py `
  --checkpoint experiments\checkpoints\plantvillage_baseline_simple_cnn_best.pt `
  --class-map experiments\checkpoints\plantvillage_baseline_simple_cnn_classes.json `
  --image path\to\leaf.jpg `
  --architecture simple_cnn `
  --top-k 3
```

## Local Web Demo

The Flask web demo supports local upload-and-predict testing with a simple responsive UI. Deployment is not done yet.

Deployment planning lives in `docs/deployment.md`. The current recommendation is to prepare a Hugging Face Spaces Docker deployment path, with a local-only screenshot demo as the fallback until model artifact hosting is approved.

Deployment scaffolding exists for future Docker-based hosting, but no live deployment has been performed. The Docker image still needs the ignored checkpoint and class map to be provided at runtime.

The web demo expects these local artifacts, which are ignored by git. See `docs/artifacts.md` for artifact validation and future hosting options:

```text
experiments/checkpoints/plantvillage_baseline_simple_cnn_best.pt
experiments/checkpoints/plantvillage_baseline_simple_cnn_classes.json
```

Run the local web app:

```powershell
python -m flask --app web.app run
```

Then open the local Flask URL shown in the terminal. Upload rules are intentionally conservative:

- Supported file types: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`
- Maximum file size: 5 MB
- One image per request
- Temporary upload handling only, no database and no permanent repo storage

The UI includes a client-side image preview before submission, an empty state, clear inline error states, and top-3 prediction cards with confidence bars. The app uses the existing `src.infer.predict_image` path with `simple_cnn`. If the checkpoint or class mapping is missing, it fails clearly and tells you to recreate the baseline checkpoint. The result is a baseline educational screening output, not a definitive diagnosis and not final model quality.

For production-style serving in a future deployment, use Gunicorn instead of Flask's development server:

```bash
gunicorn --bind 0.0.0.0:${PORT:-7860} web.app:app
```

A lightweight health check is available at `GET /healthz`. It confirms the app process is alive without loading the model or exposing artifact paths.

Validate local artifacts before running or deploying the demo:

```powershell
python src\validate_artifacts.py --expected-classes 15
```

Optional model artifact path overrides:

```powershell
$env:PLANTGUARD_CHECKPOINT_PATH="C:\path\to\model.pt"
$env:PLANTGUARD_CLASS_MAP_PATH="C:\path\to\classes.json"
python -m flask --app web.app run
```

### Screenshots

These screenshots were captured from the local Flask demo. They document browser validation only; deployment is still pending.

| Default state | Prediction result |
| --- | --- |
| ![PlantGuard web demo default state](docs/screenshots/web-demo-default.png) | ![PlantGuard web demo prediction result](docs/screenshots/web-demo-result.png) |

| Error state | Mobile layout |
| --- | --- |
| ![PlantGuard web demo error state](docs/screenshots/web-demo-error.png) | ![PlantGuard web demo mobile layout](docs/screenshots/web-demo-mobile.png) |

## Placeholder Outputs

Some existing files in `experiments/results/` and the notebooks are still scaffolding from the original academic project. Treat them as placeholders until the project is trained and evaluated against a real checkpoint.

`src/gradcam.py` and `src/utils/visualization.py` are intentionally non-generating placeholders. They do not create fake outputs.

## Limitations

- Predictions are educational screening output, not definitive agricultural diagnosis.
- Performance is not validated until real evaluation metrics are generated.
- PlantVillage is lab-curated and may not generalize to field images.
- Grad-CAM and metrics-backed visualization are still planned work.
- Deployment, model artifact hosting, and additional portfolio polish are planned for later phases.

## Recommended Next Phase

Next deployment work should approve an artifact host and validate the Docker image/container flow without committing checkpoints or deploying until explicitly approved.
