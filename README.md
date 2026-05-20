# PlantGuard AI

PlantGuard AI is an Intelligent Systems finals project for plant disease screening from leaf images. The project is being revived from a school-project scaffold into a reliable, portfolio-ready application.

## Current Status

Phase 1 focuses on stabilizing the machine-learning foundation before adding a web UI or deployment setup.

What exists now:

- PyTorch image-classification training scaffold.
- Simple CNN and ResNet18 model definitions.
- PlantVillage dataset download and split scripts.
- Dataset validation CLI.
- Experiment YAML configs.
- Checkpoint and class mapping save/load support.
- Real checkpoint-based evaluation CLI.
- Basic NLP and RL project scaffolding.
- Grad-CAM and visualization placeholders that fail honestly until implemented.
- A command-line inference entrypoint that works once a trained checkpoint and class mapping exist.

What is not implemented yet:

- No web UI.
- No mobile app.
- No desktop app.
- No deployment configuration.
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
python src\eval.py --help
```

If Ruff is installed:

```powershell
ruff check .
```

## Dataset

The intended dataset is PlantVillage. The download script uses KaggleHub and requires local Kaggle credentials configured outside the repository.

Dataset download is intentionally not part of the safe validation flow:

```powershell
python data\get_data.py
```

Only run that command when you intentionally want to download the dataset.

Expected raw dataset layout:

```text
data/raw/plantvillage/
  Apple___healthy/
    image-1.jpg
  Tomato___Early_blight/
    image-2.jpg
```

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

Dataset folders are ignored by git. Keep `data/raw/`, `data/splits/`, and any large generated artifacts local.

Validate the raw dataset folder without training:

```powershell
python src\validate_dataset.py data\raw\plantvillage --layout raw
```

Validate split folders:

```powershell
python src\validate_dataset.py data\splits --layout split
```

## Training

Training requires the dataset to exist under `data/raw/plantvillage/` and then be split into train/validation/test folders.

Example commands:

```powershell
python src\data_pipeline.py --action split --seed 42
python src\train.py --config experiments\configs\resnet18_default.yaml
```

Trained checkpoints and class mappings are written to `experiments/checkpoints/`. That folder is ignored by git because model artifacts are usually large and should not be committed by accident.

The `baseline_sklearn.yaml` config is marked as not implemented. It documents a planned classical ML baseline, but `src/train.py` is a PyTorch trainer and does not run `sklearn_rf` yet.

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

## Placeholder Outputs

Some existing files in `experiments/results/` and the notebooks are still scaffolding from the original academic project. Treat them as placeholders until the project is trained and evaluated against a real checkpoint.

`src/gradcam.py` and `src/utils/visualization.py` are intentionally non-generating placeholders. They do not create fake outputs.

## Limitations

- Predictions are educational screening output, not definitive agricultural diagnosis.
- Performance is not validated until real evaluation metrics are generated.
- PlantVillage is lab-curated and may not generalize to field images.
- Grad-CAM and metrics-backed visualization are still planned work.
- Web UI, deployment, and portfolio polish are planned for later phases.

## Recommended Next Phase

Phase 2 should clean up dependencies and structure, add more tests, and prepare the project for a small web inference UI without over-engineering the original finals-project scope.
