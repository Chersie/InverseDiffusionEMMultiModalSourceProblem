from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NAIVE_DIR = PROJECT_ROOT / "NaiveSolution"
CHERSIE_DIR = PROJECT_ROOT / "Chersie"

DOCS_DIR = PROJECT_ROOT / "docs"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

DATA_RAW_DIR = DATA_DIR / "raw"
DATA_INTERIM_DIR = DATA_DIR / "interim"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_EXTERNAL_DIR = DATA_DIR / "external"
DATA_ML_DIR = DATA_DIR / "ml"
DATA_ML_DATASETS_DIR = DATA_ML_DIR / "datasets"
DATA_ML_FEATURES_DIR = DATA_ML_DIR / "features"
DATA_ML_SPLITS_DIR = DATA_ML_DIR / "splits"

MODELS_TRACKING_DIR = MODELS_DIR / "tracking"
MODELS_TRAINING_DIR = MODELS_DIR / "training"
MODELS_ARTIFACTS_DIR = MODELS_DIR / "artifacts"

LIBRARY_FAST_DIR = CHERSIE_DIR / "FieldsFast0.5"
LIBRARY_SLOW_DIR = CHERSIE_DIR / "Fields0.5"


def ensure_scaffold_dirs() -> None:
    """Ensure new architecture directories exist."""
    for directory in (
        DOCS_DIR,
        DATA_DIR,
        DATA_RAW_DIR,
        DATA_INTERIM_DIR,
        DATA_PROCESSED_DIR,
        DATA_EXTERNAL_DIR,
        DATA_ML_DIR,
        DATA_ML_DATASETS_DIR,
        DATA_ML_FEATURES_DIR,
        DATA_ML_SPLITS_DIR,
        MODELS_DIR,
        MODELS_TRACKING_DIR,
        MODELS_TRAINING_DIR,
        MODELS_ARTIFACTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
