# Data: PlantVillage

- **Dataset**: PlantVillage Dataset
- **Primary source**: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
- **Alternative**: https://github.com/spMohanty/PlantVillage-Dataset
- **License**: CC0 / Public Domain (verify at source)

## Download

The download helper uses KaggleHub and requires Kaggle credentials configured outside this repository. Do not commit `kaggle.json`, `.env`, API tokens, downloaded data, checkpoints, or generated results.

Preview the planned dataset and target without network access or writes:

```bash
python data/get_data.py --dry-run
```

Download through KaggleHub only when you intentionally want the dataset locally:

```bash
python data/get_data.py
```

If you manually download or extract PlantVillage first, normalize the extracted archive into the expected raw folder:

```bash
python data/get_data.py --source-dir /path/to/extracted/plantvillage --overwrite
```

The script looks for the Kaggle `plantvillage dataset/color/` folder when present and copies class folders into `data/raw/plantvillage/`. It validates image files after copying and removes corrupt or empty image files inside the ignored target folder.

## Expected Layout

Raw data should use one folder per class:

```text
data/raw/plantvillage/
  Apple___healthy/
    image-1.jpg
  Tomato___Early_blight/
    image-2.jpg
```

Do not leave an extra wrapper folder under `data/raw/plantvillage/`; training expects class names directly under the raw root.

After splitting, training uses:

```text
data/splits/
  train/<class-name>/
  val/<class-name>/
  test/<class-name>/
```

Validate a raw dataset without training:

```bash
python src/validate_dataset.py data/raw/plantvillage --layout raw
```

Create split folders:

```bash
python src/data_pipeline.py --action split --raw-dir data/raw/plantvillage --split-dir data/splits --seed 42
```

Validate split folders:

```bash
python src/validate_dataset.py data/splits --layout split
```

For a one-epoch real-data pipeline smoke run after the split exists, use:

```bash
python src/train.py --config experiments/configs/plantvillage_smoke.yaml
```

That smoke config is not a final training recipe and does not prove model quality.
