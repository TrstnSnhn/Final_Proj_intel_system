from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "plantvillage"
SPLIT_DIR = DATA_DIR / "splits"
TRAIN_DIR = SPLIT_DIR / "train"
VAL_DIR = SPLIT_DIR / "val"
TEST_DIR = SPLIT_DIR / "test"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 38
NUM_WORKERS = 2
SEED = 42

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
LOGS_DIR = EXPERIMENTS_DIR / "logs"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
CHECKPOINTS_DIR = EXPERIMENTS_DIR / "checkpoints"
