"""
Central configuration: paths, thresholds, class names.
Everything is relative to the project root so the service is self-contained.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Paths
DATA_DIR = PROJECT_ROOT / "data"
CANDIDATE_TILES_DIR = DATA_DIR / "candidate_tiles"
EVAL_SET_DIR = DATA_DIR / "eval_set"
EVAL_LABELS_PATH = DATA_DIR / "eval_labels.csv"
MODEL_PATH = PROJECT_ROOT / "model" / "classifier.pt"
TILE_STORE_DIR = PROJECT_ROOT / "tile_store"
DB_PATH = PROJECT_ROOT / "results.db"

# Model
CLASS_NAMES = sorted([
    "AnnualCrop",
    "Forest",
    "Highway",
    "Industrial",
    "Residential",
    "River",
    "SeaLake",
])
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = 64  # EuroSAT tiles are 64x64

# Confidence
CONFIDENCE_THRESHOLD = 0.5  # below this needs_review = True

MODEL_VERSION = "resnet18-ft-v1"