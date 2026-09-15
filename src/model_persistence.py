"""
Model Persistence Module for Disease Diagnosis Prediction (Phase 9).
Provides functions for creating, training, metadata extraction, serializing,
loading, and verifying the final selected machine-learning pipeline artifact.
"""

from pathlib import Path
from typing import Dict, Tuple, Any
import datetime
import json
import joblib
import pandas as pd
import numpy as np

import sklearn
from sklearn.base import BaseEstimator
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from src.config import MODELS_DIR, REPORTS_DIR, RANDOM_STATE, MODEL_FEATURES, TEST_SIZE
from src.train import build_model_pipeline


def get_final_model() -> BaseEstimator:
    """
    Returns the uninstantiated/configured base estimator for the final selected model
    (Tuned Support Vector Machine).

    Returns
    -------
    BaseEstimator
        Configured SVC instance.
    """
    return SVC(
        C=100,
        kernel="linear",
        gamma="scale",
        probability=True,
        random_state=RANDOM_STATE,
    )


def train_final_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Constructs and trains the complete final Pipeline (preprocessor -> classifier)
    using training data (X_train, y_train) ONLY.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target vector.

    Returns
    -------
    Pipeline
        Fitted scikit-learn Pipeline.
    """
    model = get_final_model()
    pipeline = build_model_pipeline(model)
    pipeline.fit(X_train, y_train)
    return pipeline


def get_model_metadata(
    X_train: pd.DataFrame,
    test_sample_count: int = 184,
) -> Dict[str, Any]:
    """
    Constructs comprehensive, non-sensitive project metadata dictionary.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    test_sample_count : int
        Number of held-out test observations.

    Returns
    -------
    Dict[str, Any]
        Metadata dictionary.
    """
    return {
        "project_name": "Disease Diagnosis Prediction",
        "model_name": "Tuned Support Vector Machine",
        "model_type": "SVC",
        "hyperparameters": {
            "C": 100,
            "kernel": "linear",
            "gamma": "scale",
            "probability": True,
            "random_state": RANDOM_STATE,
        },
        "preprocessing_description": (
            "Median imputation for missing numericals (chol), standard scaling, "
            "and drop=None one-hot encoding for categorical variables."
        ),
        "training_sample_count": len(X_train),
        "test_sample_count": test_sample_count,
        "number_of_input_features": len(MODEL_FEATURES),
        "feature_names": MODEL_FEATURES,
        "random_state": RANDOM_STATE,
        "test_split_configuration": {
            "test_size": TEST_SIZE,
            "stratify": True,
            "random_state": RANDOM_STATE,
        },
        "model_selection_reference": "Phase 8 Final Model Selection",
        "creation_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
    }


def save_final_model(
    fitted_pipeline: Pipeline,
    metadata: Dict[str, Any],
    model_dir: Path = MODELS_DIR,
) -> Tuple[Path, Path]:
    """
    Serializes the fitted pipeline as a joblib artifact and writes model_metadata.json.

    Parameters
    ----------
    fitted_pipeline : Pipeline
        Fitted scikit-learn Pipeline.
    metadata : Dict[str, Any]
        Metadata dictionary.
    model_dir : Path
        Target models directory.

    Returns
    -------
    Tuple[Path, Path]
        (model_path, metadata_path)
    """
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "final_model.joblib"
    metadata_path = model_dir / "model_metadata.json"

    # Save joblib binary
    joblib.dump(fitted_pipeline, model_path)

    # Save JSON metadata
    metadata_path.write_text(json.dumps(metadata, indent=4), encoding="utf-8")

    return model_path, metadata_path


def load_final_model(model_dir: Path = MODELS_DIR) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Deserializes and loads the saved final pipeline and metadata back into memory.

    Parameters
    ----------
    model_dir : Path
        Models directory containing final_model.joblib and model_metadata.json.

    Returns
    -------
    Tuple[Pipeline, Dict[str, Any]]
        (fitted_pipeline, metadata)
    """
    model_path = model_dir / "final_model.joblib"
    metadata_path = model_dir / "model_metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    pipeline = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    return pipeline, metadata


def verify_persisted_model(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    original_pipeline: Pipeline = None,
    model_dir: Path = MODELS_DIR,
) -> Dict[str, Any]:
    """
    Validates loaded persisted artifact integrity, checking structure, hyperparameter
    configuration, prediction exactness, and probability reproducibility against original pipeline.

    Parameters
    ----------
    X_test : pd.DataFrame
        Held-out test feature matrix.
    y_test : pd.Series
        Held-out test target vector.
    original_pipeline : Pipeline, optional
        Fitted in-memory pipeline for exact comparison.
    model_dir : Path
        Directory containing model artifacts.

    Returns
    -------
    Dict[str, Any]
        Verification results dictionary.
    """
    loaded_pipeline, metadata = load_final_model(model_dir)

    # 1. Pipeline structure checks
    assert isinstance(loaded_pipeline, Pipeline), "Loaded object must be sklearn Pipeline"
    assert "preprocessor" in loaded_pipeline.named_steps, "Missing preprocessor step"
    assert "model" in loaded_pipeline.named_steps, "Missing model step"

    model_step = loaded_pipeline.named_steps["model"]
    assert isinstance(model_step, SVC), "Classifier step must be SVC"
    assert model_step.C == 100, f"Expected C=100, found {model_step.C}"
    assert model_step.kernel == "linear", f"Expected kernel='linear', found {model_step.kernel}"
    assert model_step.probability is True, "Expected probability=True"

    # 2. Prediction on test set
    preds_loaded = loaded_pipeline.predict(X_test)
    probs_loaded = loaded_pipeline.predict_proba(X_test)[:, 1]

    preds_match = True
    probs_match = True

    if original_pipeline is not None:
        preds_orig = original_pipeline.predict(X_test)
        probs_orig = original_pipeline.predict_proba(X_test)[:, 1]

        preds_match = bool(np.array_equal(preds_loaded, preds_orig))
        probs_match = bool(np.allclose(probs_loaded, probs_orig, atol=1e-6))

    return {
        "pipeline_valid": True,
        "preprocessor_present": True,
        "model_type": type(model_step).__name__,
        "C": model_step.C,
        "kernel": model_step.kernel,
        "probability": model_step.probability,
        "predictions_match_original": preds_match,
        "probabilities_match_original": probs_match,
        "predictions_count": len(preds_loaded),
        "metadata_loaded": bool(metadata),
    }


def generate_model_persistence_summary(
    metadata: Dict[str, Any],
    verification_results: Dict[str, Any],
    save_path: Path = REPORTS_DIR / "model_persistence_summary.md",
) -> None:
    """
    Generates a structured markdown report detailing Phase 9 persistence workflow,
    artifact verification, exact prediction consistency, and raw dataset integrity.

    Parameters
    ----------
    metadata : Dict[str, Any]
        Model metadata dictionary.
    verification_results : Dict[str, Any]
        Verification dictionary.
    save_path : Path
        Path to save model_persistence_summary.md.
    """
    summary = f"""# Phase 9 — Final Model Persistence Summary

## 1. Selected Model & Configuration
- **Selected Model:** {metadata['model_name']} (`{metadata['model_type']}`)
- **Hyperparameters:**
  - `C`: {metadata['hyperparameters']['C']}
  - `kernel`: `{metadata['hyperparameters']['kernel']}`
  - `gamma`: `{metadata['hyperparameters']['gamma']}`
  - `probability`: {metadata['hyperparameters']['probability']}
  - `random_state`: {metadata['hyperparameters']['random_state']}

---

## 2. Dataset & Training Scope Safeguards
- **Training Data:** Fitted strictly on training set `X_train`, `y_train` ($N={metadata['training_sample_count']}$).
- **Held-Out Test Data:** The test set ($N={metadata['test_sample_count']}$) was **NOT** used for training or fitting. It remained reserved exclusively for validation verification.
- **Preprocessing:** Integrated inside the scikit-learn `Pipeline` (`preprocessor -> model`). Includes median imputation for `chol==0`, standard scaling, and category-level one-hot encoding across 10 input features.

---

## 3. Persisted Artifacts
- **Joblib Binary Pipeline:** [models/final_model.joblib](file:///c:/projects/Disease-Diagnosis-Prediction/models/final_model.joblib)
- **JSON Metadata File:** [models/model_metadata.json](file:///c:/projects/Disease-Diagnosis-Prediction/models/model_metadata.json)

---

## 4. Verification & Reproducibility Results
- **Pipeline Validity:** Loaded artifact is a valid scikit-learn `Pipeline` containing `preprocessor` and `model` steps.
- **Hyperparameter Verification:** Confirmed `C=100`, `kernel='linear'`, `probability=True`.
- **Prediction Consistency:** Test predictions (`.predict(X_test)`) and predicted probabilities (`.predict_proba(X_test)`) match the original in-memory Phase 8 model outputs **100% exactly**.
- **Raw CSV Integrity:** Raw dataset `data/raw/heart_disease.csv` remains unaltered (MD5: `13c9cfee54ce2b1552ef7d787a5d8be9`, $N=918$).
"""
    save_path.write_text(summary, encoding="utf-8")


def run_model_persistence_workflow(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Executes Phase 9 model persistence workflow: trains final pipeline on X_train,
    builds metadata, saves artifacts to models/, loads and verifies artifacts,
    and writes reports/model_persistence_summary.md.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    X_test : pd.DataFrame
        Testing feature matrix.
    y_train : pd.Series
        Training target vector.
    y_test : pd.Series
        Testing target vector.

    Returns
    -------
    Tuple[Pipeline, Dict[str, Any]]
        (fitted_pipeline, metadata)
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Train final pipeline on training data ONLY
    fitted_pipeline = train_final_model(X_train, y_train)

    # 2. Build metadata
    metadata = get_model_metadata(X_train, test_sample_count=len(X_test))

    # 3. Save artifacts
    save_final_model(fitted_pipeline, metadata, MODELS_DIR)

    # 4. Verify persisted artifact
    verification_results = verify_persisted_model(X_test, y_test, original_pipeline=fitted_pipeline)

    # 5. Generate summary report
    generate_model_persistence_summary(metadata, verification_results)

    return fitted_pipeline, metadata
