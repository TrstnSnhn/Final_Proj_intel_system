# Data: PlantVillage

- **Dataset**: PlantVillage Dataset
- **Primary source**: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
- **Alternative**: https://github.com/spMohanty/PlantVillage-Dataset
- **License**: CC0 / Public Domain (verify at source)

## Download
```bash
python data/get_data.py
```

The script downloads/extracts data into `data/raw/plantvillage/` (folder-per-class) and validates corrupt/empty files.

## Expected Layout

Raw data should use one folder per class:

```text
data/raw/plantvillage/
  Apple___healthy/
  Tomato___Early_blight/
```

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

Validate split folders:

```bash
python src/validate_dataset.py data/splits --layout split
```
