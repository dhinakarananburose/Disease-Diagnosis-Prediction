"""
Project Configuration for Disease Diagnosis Prediction.
Defines directory paths, dataset specifications, feature groupings, and target definitions.
"""

from pathlib import Path

# Project Directory Structure using pathlib
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Dataset File Path
DATASET_PATH = RAW_DATA_DIR / "heart_disease.csv"

# Target Columns
TARGET_COLUMN = "num"
BINARY_TARGET_COLUMN = "target"
ORIGINAL_TARGET = "num"
BINARY_TARGET = "target"

# Train/Test Split Constants
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Feature Groupings based on dataset inspection (11 raw columns)
NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalch", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "restecg"]
BOOLEAN_FEATURES = ["fbs", "exang"]

# Model Feature Matrix Columns (10 features, excluding 'num' and 'target')
MODEL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES

# Expected Raw Dataset Columns
RAW_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalch",
    "exang",
    "oldpeak",
    "num",
]
