"""
Prediction Module for Disease Diagnosis Prediction (Phase 10).
Provides a reusable, leakage-safe inference pipeline for single and batch predictions.
Uses the persisted scikit-learn pipeline (models/final_model.joblib).
Handles input validation, schema consistency, chol=0 missing value treatment,
and produces structured prediction results and positive-class probabilities.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import json
import joblib
import numpy as np
import pandas as pd

from src.config import (
    MODELS_DIR,
    MODEL_FEATURES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    BOOLEAN_FEATURES,
)


def load_persisted_model(model_path: Optional[Path] = None):
    """
    Loads the persisted trained scikit-learn pipeline artifact.

    Parameters
    ----------
    model_path : Path, optional
        Path to joblib model artifact. Defaults to MODELS_DIR / 'final_model.joblib'.

    Returns
    -------
    Pipeline
        Loaded scikit-learn Pipeline instance.

    Raises
    ------
    FileNotFoundError
        If model artifact does not exist at specified path.
    """
    path = Path(model_path) if model_path else MODELS_DIR / "final_model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Persisted model artifact not found at: {path}")
    
    return joblib.load(path)


def load_model_metadata(metadata_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Loads metadata associated with the persisted model artifact.

    Parameters
    ----------
    metadata_path : Path, optional
        Path to JSON metadata file. Defaults to MODELS_DIR / 'model_metadata.json'.

    Returns
    -------
    dict
        Metadata dictionary.
    """
    path = Path(metadata_path) if metadata_path else MODELS_DIR / "model_metadata.json"
    if not path.exists():
        return {"model_name": "Tuned Support Vector Machine"}
    
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_prediction_input(
    input_data: Union[Dict[str, Any], pd.DataFrame, List[Dict[str, Any]]],
    is_batch: bool = False,
) -> pd.DataFrame:
    """
    Validates prediction input structure, feature completeness, data types, and values.

    Parameters
    ----------
    input_data : Dict, pd.DataFrame, or List of Dicts
        Input features for single or batch prediction.
    is_batch : bool, optional
        Whether the input is expected to be a batch DataFrame (default False).

    Returns
    -------
    pd.DataFrame
        Validated DataFrame copy.

    Raises
    ------
    ValueError
        If input is empty, missing required features, or contains invalid data types/values.
    TypeError
        If input is not of expected type.
    """
    if input_data is None:
        raise ValueError("Prediction input cannot be None.")

    if isinstance(input_data, dict):
        if not input_data:
            raise ValueError("Prediction input dictionary cannot be empty.")
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        if input_data.empty:
            raise ValueError("Prediction input DataFrame cannot be empty.")
        df = input_data.copy()
    elif isinstance(input_data, list):
        if not input_data:
            raise ValueError("Prediction input list cannot be empty.")
        if not all(isinstance(item, dict) for item in input_data):
            raise TypeError("All items in input list must be dictionaries.")
        df = pd.DataFrame(input_data)
    else:
        raise TypeError(
            f"Unsupported prediction input type: {type(input_data)}. "
            "Expected dict, pandas DataFrame, or list of dicts."
        )

    # 1. Check required 10 features exist
    missing_cols = [col for col in MODEL_FEATURES if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Input is missing required feature(s): {missing_cols}")

    # 2. Validate Numerical Features
    for col in NUMERICAL_FEATURES:
        series = df[col]
        for idx, val in series.items():
            if pd.isna(val):
                continue  # Missing values are handled by imputer in pipeline
            if not isinstance(val, (int, float, np.number)):
                # Try converting numeric strings
                try:
                    num_val = float(val)
                except (ValueError, TypeError):
                    raise ValueError(
                        f"Invalid numeric value '{val}' for feature '{col}' at index {idx}. "
                        "Must be a valid integer or float."
                    )
            else:
                num_val = float(val)

            # Enforce numerical range constraints
            if col == "age" and not (1 <= num_val <= 120):
                raise ValueError(
                    f"Invalid value {num_val} for feature 'age' at index {idx}. Must be between 1 and 120."
                )
            elif col == "trestbps" and not (0 <= num_val <= 300):
                raise ValueError(
                    f"Invalid value {num_val} for feature 'trestbps' at index {idx}. Must be between 0 and 300."
                )
            elif col == "chol" and not (0 <= num_val <= 1500):
                raise ValueError(
                    f"Invalid value {num_val} for feature 'chol' at index {idx}. Must be between 0 and 1500."
                )
            elif col == "thalch" and not (1 <= num_val <= 250):
                raise ValueError(
                    f"Invalid value {num_val} for feature 'thalch' at index {idx}. Must be between 1 and 250."
                )
            elif col == "oldpeak" and not (-10.0 <= num_val <= 15.0):
                raise ValueError(
                    f"Invalid value {num_val} for feature 'oldpeak' at index {idx}. Must be between -10.0 and 15.0."
                )

    # 3. Validate Boolean Features
    valid_bool_values = {True, False, 1, 0, 1.0, 0.0, "true", "false", "True", "False", "1", "0"}
    for col in BOOLEAN_FEATURES:
        series = df[col]
        for idx, val in series.items():
            if pd.isna(val):
                continue
            if val not in valid_bool_values and not isinstance(val, (bool, np.bool_)):
                raise ValueError(
                    f"Invalid boolean value '{val}' for feature '{col}' at index {idx}. "
                    "Must be boolean (True/False) or 0/1."
                )

    # 4. Validate Categorical Features
    ALLOWED_CP = {"typical angina", "atypical angina", "non-anginal", "asymptomatic"}
    ALLOWED_RESTECG = {"normal", "st-t abnormality", "lv hypertrophy"}

    for col in CATEGORICAL_FEATURES:
        series = df[col]
        for idx, val in series.items():
            if pd.isna(val):
                continue
            if isinstance(val, (list, dict, set, tuple)):
                raise ValueError(
                    f"Invalid categorical value '{val}' for feature '{col}' at index {idx}. "
                    "Categorical features cannot be complex data structures."
                )
            if col == "cp" and val not in ALLOWED_CP:
                raise ValueError(
                    f"Invalid categorical value '{val}' for feature 'cp' at index {idx}. "
                    f"Must be one of {ALLOWED_CP}."
                )
            if col == "restecg" and val not in ALLOWED_RESTECG:
                raise ValueError(
                    f"Invalid categorical value '{val}' for feature 'restecg' at index {idx}. "
                    f"Must be one of {ALLOWED_RESTECG}."
                )

    return df


def prepare_prediction_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares input DataFrame for model inference:
    - Preserves column ordering matching MODEL_FEATURES
    - Converts chol == 0 -> np.nan (treats chol=0 as missing/unrecorded, matching training workflow)
    - Converts trestbps == 0 -> np.nan (if present)

    Parameters
    ----------
    df : pd.DataFrame
        Validated input DataFrame.

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame ready for pipeline inference.
    """
    df_prep = df[MODEL_FEATURES].copy()

    # Convert numeric values to float where applicable
    for col in NUMERICAL_FEATURES:
        df_prep[col] = pd.to_numeric(df_prep[col], errors="coerce")

    # In-memory data quality handling matching training workflow
    df_prep.loc[df_prep["chol"] == 0, "chol"] = np.nan
    df_prep.loc[df_prep["trestbps"] == 0, "trestbps"] = np.nan

    return df_prep


def predict_single(
    input_dict: Dict[str, Any],
    model: Optional[Any] = None,
    model_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Generates a structured prediction for a single raw observation.

    Parameters
    ----------
    input_dict : Dict[str, Any]
        Dictionary containing all 10 required feature values.
    model : Pipeline, optional
        Pre-loaded scikit-learn Pipeline instance.
    model_path : Path, optional
        Path to model artifact if model is not provided.

    Returns
    -------
    Dict[str, Any]
        Structured result containing:
        - 'predicted_class': int (0 or 1)
        - 'predicted_probability': float (predicted probability for the positive class)
        - 'model_name': str
    """
    df_validated = validate_prediction_input(input_dict, is_batch=False)
    df_prep = prepare_prediction_dataframe(df_validated)

    if model is None:
        model = load_persisted_model(model_path)

    pred_class = int(model.predict(df_prep)[0])
    pred_prob = float(model.predict_proba(df_prep)[0, 1])

    metadata = load_model_metadata()
    model_name = metadata.get("model_name", "Tuned Support Vector Machine")

    return {
        "predicted_class": pred_class,
        "predicted_probability": pred_prob,
        "model_name": model_name,
    }


def predict_batch(
    input_df: pd.DataFrame,
    model: Optional[Any] = None,
    model_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Generates predictions for a batch of raw observations in a DataFrame.
    Does NOT mutate the original input DataFrame.

    Parameters
    ----------
    input_df : pd.DataFrame
        DataFrame containing required raw features.
    model : Pipeline, optional
        Pre-loaded scikit-learn Pipeline instance.
    model_path : Path, optional
        Path to model artifact if model is not provided.

    Returns
    -------
    pd.DataFrame
        New DataFrame copy containing original columns plus:
        - 'predicted_class': int
        - 'predicted_probability': float
    """
    df_validated = validate_prediction_input(input_df, is_batch=True)
    df_prep = prepare_prediction_dataframe(df_validated)

    if model is None:
        model = load_persisted_model(model_path)

    preds = model.predict(df_prep).astype(int)
    probs = model.predict_proba(df_prep)[:, 1].astype(float)

    # Return new DataFrame copy without mutating original input_df
    result_df = input_df.copy()
    result_df["predicted_class"] = preds
    result_df["predicted_probability"] = probs

    return result_df
