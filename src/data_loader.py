"""
Data Loader Module for Disease Diagnosis Prediction.
Provides reusable functions for loading raw dataset, schema validation,
in-memory data quality handling (converting 0 values to NaN),
binary target creation, feature/target separation, and train-test splitting.
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATASET_PATH,
    RAW_COLUMNS,
    MODEL_FEATURES,
    ORIGINAL_TARGET,
    BINARY_TARGET,
    TEST_SIZE,
    RANDOM_STATE,
)


def load_raw_data(data_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Loads raw heart disease dataset from CSV file without modification.

    Parameters
    ----------
    data_path : Path, optional
        Path to raw CSV file. Defaults to DATASET_PATH from config.

    Returns
    -------
    pd.DataFrame
        Raw dataset DataFrame.
    """
    path = Path(data_path) if data_path else DATASET_PATH
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset file not found at: {path}")

    df = pd.read_csv(path)
    validate_dataset(df)
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """
    Validates dataset structure and column existence.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.

    Raises
    ------
    ValueError
        If required columns are missing or DataFrame is empty.
    """
    if df is None or df.empty:
        raise ValueError("Dataset is empty or None.")

    missing_cols = [col for col in RAW_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    if len(df) != 918:
        pass


def create_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates binary target column 'target' where target = (num > 0).astype(int).
    Preserves original 'num' column without mutating the input DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing original 'num' column.

    Returns
    -------
    pd.DataFrame
        New DataFrame copy containing both 'num' and 'target'.
    """
    if ORIGINAL_TARGET not in df.columns:
        raise ValueError(f"Target creation failed: '{ORIGINAL_TARGET}' column missing.")

    df_copy = df.copy()
    df_copy[BINARY_TARGET] = (df_copy[ORIGINAL_TARGET] > 0).astype(int)
    return df_copy


def prepare_model_dataframe(data_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Loads raw data, creates binary target, and applies in-memory data quality handling:
    - Converts trestbps == 0 -> np.nan (1 row)
    - Converts chol == 0 -> np.nan (172 rows)
    - Preserves negative oldpeak values (13 rows)
    - Preserves original 'num' column
    Does NOT modify the raw CSV file on disk.

    Parameters
    ----------
    data_path : Path, optional
        Path to raw CSV file.

    Returns
    -------
    pd.DataFrame
        In-memory modeling DataFrame with data quality conversions applied.
    """
    df_raw = load_raw_data(data_path)
    df_target = create_binary_target(df_raw)

    # Create in-memory copy for modeling quality handling
    df_model = df_target.copy()

    # Convert trestbps == 0 to NaN (if any exist)
    df_model.loc[df_model["trestbps"] == 0, "trestbps"] = np.nan

    # Convert chol == 0 to NaN (172 observations)
    df_model.loc[df_model["chol"] == 0, "chol"] = np.nan

    return df_model


def load_model_data(data_path: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads modeling DataFrame, extracts feature matrix X and target vector y.
    Excludes 'num' and 'target' from feature matrix X.

    Parameters
    ----------
    data_path : Path, optional
        Path to raw CSV file.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        (X, y) tuple where X contains 10 model features and y is binary target Series.
    """
    df_model = prepare_model_dataframe(data_path)

    X = df_model[MODEL_FEATURES].copy()
    y = df_model[BINARY_TARGET].copy()

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits feature matrix X and target y into training and testing sets.
    Uses stratified splitting by default to preserve target class proportions.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    test_size : float, optional
        Proportion of dataset for test split (default 0.20).
    random_state : int, optional
        Random seed for reproducibility (default 42).
    stratify : bool, optional
        Whether to perform stratified splitting based on y (default True).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        (X_train, X_test, y_train, y_test)
    """
    stratify_target = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )
    return X_train, X_test, y_train, y_test
