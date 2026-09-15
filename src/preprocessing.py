"""
Preprocessing Module for Disease Diagnosis Prediction.
Provides scikit-learn ColumnTransformer and Pipeline construction functions.
Implements leakage-safe imputation (median with missing indicator for numeric,
most_frequent for categorical/boolean), scaling (StandardScaler), and encoding (OneHotEncoder).
Ensures preprocessing transformers are fit strictly on training data within pipelines.
"""

from typing import Optional
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.base import BaseEstimator

from src.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, BOOLEAN_FEATURES


def _cast_boolean_to_int(X):
    """Utility function to cast boolean inputs to integer 0/1 array."""
    return X.astype(int)


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs a leakage-safe ColumnTransformer for feature preprocessing:
    - Numerical features: SimpleImputer(strategy='median', add_indicator=True) -> StandardScaler()
      (add_indicator=True preserves the predictive signal of missing/unrecorded values)
    - Categorical features: SimpleImputer(strategy='most_frequent') -> OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    - Boolean features: SimpleImputer(strategy='most_frequent') -> FunctionTransformer(_cast_boolean_to_int)

    Returns
    -------
    ColumnTransformer
        Unfitted ColumnTransformer instance.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    boolean_transformer = Pipeline(
        steps=[
            ("bool_to_int", FunctionTransformer(_cast_boolean_to_int, feature_names_out="one-to-one")),
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("bool", boolean_transformer, BOOLEAN_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def build_pipeline(model: Optional[BaseEstimator] = None) -> Pipeline:
    """
    Constructs a full scikit-learn Pipeline with preprocessing and optional estimator.

    Parameters
    ----------
    model : BaseEstimator, optional
        Scikit-learn compatible classifier. Defaults to None.

    Returns
    -------
    Pipeline
        Unfitted scikit-learn Pipeline.
    """
    preprocessor = build_preprocessor()
    steps = [("preprocessor", preprocessor)]

    if model is not None:
        steps.append(("model", model))

    return Pipeline(steps=steps)
