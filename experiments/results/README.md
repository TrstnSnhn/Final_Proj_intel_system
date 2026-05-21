# Experiment Results

The committed files in this folder are scaffold artifacts from the original finals project. Treat them as placeholders until the training and evaluation pipeline is run with a real checkpoint.

Real evaluation output should be regenerated locally after:

1. Downloading the dataset with explicit approval.
2. Training or providing a checkpoint.
3. Running the evaluation and visualization scripts against that checkpoint.

Generated result images and JSON files are ignored by git by default so future local runs do not accidentally commit large or misleading artifacts.

## Artifact Naming

Use the experiment name from the YAML config when regenerating local artifacts:

- `experiments/checkpoints/<experiment_name>_best.pt`
- `experiments/checkpoints/<experiment_name>_classes.json`
- `experiments/logs/<experiment_name>.csv`
- `experiments/results/<experiment_name>_eval_summary.json`

For the current SimpleCNN baseline, the expected generated evaluation path is `experiments/results/plantvillage_baseline_eval_summary.json`.

## Local Summary Helper

After training and evaluation have produced local artifacts, summarize them without loading the dataset or model:

```powershell
python src\summarize_experiment.py `
  --eval-summary experiments\results\plantvillage_baseline_eval_summary.json `
  --training-log experiments\logs\plantvillage_baseline_simple_cnn.csv
```

The summary is printed to the terminal only. It does not create or commit generated outputs.
