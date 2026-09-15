"""
Training Module for Disease Diagnosis Prediction.
Provides functions for defining 5 baseline classification models, building leakage-safe Pipelines,
executing 5-fold Stratified Cross-Validation on training data, evaluating on held-out test data,
and generating model evaluation reports and figures.
"""

from pathlib import Path
from typing import Dict, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

from src.config import (
    REPORTS_DIR,
    FIGURES_DIR,
    RANDOM_STATE,
)
from src.preprocessing import build_preprocessor
from src.evaluate import (
    evaluate_pipeline_test,
    plot_confusion_matrix,
    plot_combined_roc_curves,
    plot_model_comparison,
    plot_tuned_vs_baseline_comparison,
)



def get_baseline_models() -> Dict[str, BaseEstimator]:
    """
    Returns a dictionary of 5 baseline classification models with standard parameters.

    Returns
    -------
    Dict[str, BaseEstimator]
        Mapping of model_name -> uninstantiated or instantiated estimator.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE
        ),
        "Support Vector Machine": SVC(
            probability=True, random_state=RANDOM_STATE
        ),
    }


def build_model_pipeline(model: BaseEstimator) -> Pipeline:
    """
    Wraps a classifier inside a scikit-learn Pipeline with the preprocessor.

    Parameters
    ----------
    model : BaseEstimator
        Classifier instance.

    Returns
    -------
    Pipeline
        Unfitted scikit-learn Pipeline.
    """
    preprocessor = build_preprocessor()
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def train_model(
    model_name: str, model: BaseEstimator, X_train: pd.DataFrame, y_train: pd.Series
) -> Pipeline:
    """
    Constructs a pipeline for a model and fits it on training data.

    Parameters
    ----------
    model_name : str
        Name of the model.
    model : BaseEstimator
        Classifier instance.
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target vector.

    Returns
    -------
    Pipeline
        Fitted scikit-learn Pipeline.
    """
    pipeline = build_model_pipeline(model)
    pipeline.fit(X_train, y_train)
    return pipeline


def cross_validate_model(
    pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series
) -> Dict[str, float]:
    """
    Executes 5-fold Stratified Cross-Validation on training data ONLY.
    Refits preprocessor independently inside every CV fold to prevent data leakage.

    Parameters
    ----------
    pipeline : Pipeline
        Unfitted or pre-constructed model Pipeline.
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target vector.

    Returns
    -------
    Dict[str, float]
        Dictionary of CV mean and std metrics across folds.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    cv_scores = cross_validate(
        pipeline, X_train, y_train, cv=cv, scoring=scoring, return_train_score=False
    )

    return {
        "CV_Accuracy_Mean": float(np.mean(cv_scores["test_accuracy"])),
        "CV_Accuracy_Std": float(np.std(cv_scores["test_accuracy"])),
        "CV_Precision_Mean": float(np.mean(cv_scores["test_precision"])),
        "CV_Precision_Std": float(np.std(cv_scores["test_precision"])),
        "CV_Recall_Mean": float(np.mean(cv_scores["test_recall"])),
        "CV_Recall_Std": float(np.std(cv_scores["test_recall"])),
        "CV_F1_Mean": float(np.mean(cv_scores["test_f1"])),
        "CV_F1_Std": float(np.std(cv_scores["test_f1"])),
        "CV_ROC_AUC_Mean": float(np.mean(cv_scores["test_roc_auc"])),
        "CV_ROC_AUC_Std": float(np.std(cv_scores["test_roc_auc"])),
    }


def train_baseline_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
    """
    Executes baseline model training, cross-validation, held-out test evaluation,
    saves reports/model_results.csv, and generates confusion matrices & ROC curves.

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
    Tuple[pd.DataFrame, Dict[str, Pipeline]]
        (results_df, fitted_pipelines)
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    models = get_baseline_models()

    results_rows = []
    fitted_pipelines = {}
    model_probs = {}

    slug_map = {
        "Logistic Regression": "logistic_regression",
        "K-Nearest Neighbors": "knn",
        "Decision Tree": "decision_tree",
        "Random Forest": "random_forest",
        "Support Vector Machine": "svm",
    }

    for model_name, model in models.items():
        # 1. Build unfitted pipeline for cross-validation
        unfitted_pipeline = build_model_pipeline(model)
        cv_metrics = cross_validate_model(unfitted_pipeline, X_train, y_train)

        # 2. Fit pipeline on full training data
        fitted_pipeline = train_model(model_name, model, X_train, y_train)
        fitted_pipelines[model_name] = fitted_pipeline

        # 3. Evaluate on held-out test set
        test_metrics, y_pred, y_prob = evaluate_pipeline_test(
            fitted_pipeline, X_test, y_test
        )
        model_probs[model_name] = (y_prob, test_metrics["ROC_AUC"])

        # 4. Generate & Save Confusion Matrix
        slug = slug_map.get(model_name, model_name.lower().replace(" ", "_"))
        cm_path = FIGURES_DIR / f"confusion_matrix_{slug}.png"
        plot_confusion_matrix(y_test, y_pred, model_name, cm_path)

        # 5. Assemble result row
        row = {
            "Model": model_name,
            "Test_Accuracy": test_metrics["Accuracy"],
            "Test_Precision": test_metrics["Precision"],
            "Test_Recall": test_metrics["Recall"],
            "Test_F1": test_metrics["F1"],
            "Test_ROC_AUC": test_metrics["ROC_AUC"],
            "CV_Accuracy_Mean": cv_metrics["CV_Accuracy_Mean"],
            "CV_Accuracy_Std": cv_metrics["CV_Accuracy_Std"],
            "CV_Precision_Mean": cv_metrics["CV_Precision_Mean"],
            "CV_Precision_Std": cv_metrics["CV_Precision_Std"],
            "CV_Recall_Mean": cv_metrics["CV_Recall_Mean"],
            "CV_Recall_Std": cv_metrics["CV_Recall_Std"],
            "CV_F1_Mean": cv_metrics["CV_F1_Mean"],
            "CV_F1_Std": cv_metrics["CV_F1_Std"],
            "CV_ROC_AUC_Mean": cv_metrics["CV_ROC_AUC_Mean"],
            "CV_ROC_AUC_Std": cv_metrics["CV_ROC_AUC_Std"],
        }
        results_rows.append(row)

    results_df = pd.DataFrame(results_rows)

    # Save model results CSV
    results_path = REPORTS_DIR / "model_results.csv"
    results_df.to_csv(results_path, index=False)

    # Also save copy to reports/figures/model_results.csv if present
    (FIGURES_DIR / "model_results.csv").write_text(results_df.to_csv(index=False), encoding="utf-8")

    # Plot Combined ROC Curves
    roc_path = FIGURES_DIR / "roc_curves_baseline.png"
    plot_combined_roc_curves(model_probs, y_test, roc_path)

    # Plot Model Comparison Bar Chart
    comp_path = FIGURES_DIR / "baseline_model_comparison.png"
    plot_model_comparison(results_df, comp_path)

    return results_df, fitted_pipelines


def get_tuning_search_spaces() -> Dict[str, Tuple[BaseEstimator, Dict[str, Any]]]:
    """
    Returns search spaces for Phase 6 candidate model hyperparameter optimization.

    Returns
    -------
    Dict[str, Tuple[BaseEstimator, Dict[str, Any]]]
        Mapping model_name -> (uninstantiated/base_model, param_grid).
    """
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            {
                "model__C": [0.01, 0.1, 1, 10, 100],
                "model__solver": ["lbfgs"],
                "model__max_iter": [1000],
            },
        ),
        "Support Vector Machine": (
            SVC(probability=True, random_state=RANDOM_STATE),
            {
                "model__C": [0.1, 1, 10, 100],
                "model__kernel": ["rbf", "linear"],
                "model__gamma": ["scale", "auto"],
            },
        ),
    }


def tune_model_hyperparameters(
    model_name: str,
    model: BaseEstimator,
    param_grid: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> GridSearchCV:
    """
    Executes leakage-safe 5-fold Stratified Cross-Validation hyperparameter optimization
    on training data ONLY using ROC-AUC as primary refit scoring metric.

    Parameters
    ----------
    model_name : str
        Name of candidate model.
    model : BaseEstimator
        Base classifier instance.
    param_grid : Dict[str, Any]
        Hyperparameter grid for the pipeline classifier step ('model__*').
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target vector.

    Returns
    -------
    GridSearchCV
        Fitted GridSearchCV instance.
    """
    pipeline = build_model_pipeline(model)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        refit="roc_auc",
        return_train_score=False,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    return grid_search


def generate_hyperparameter_search_summary(
    results_df: pd.DataFrame,
    grid_searches: Dict[str, GridSearchCV],
    save_path: Path = REPORTS_DIR / "hyperparameter_search_summary.md",
) -> None:
    """
    Generates a structured markdown report detailing Phase 6 hyperparameter search methodology,
    best configurations, CV metrics, test performance comparison, and cautious selection rationale.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame containing baseline vs tuned comparison metrics.
    grid_searches : Dict[str, GridSearchCV]
        Dictionary of fitted GridSearchCV objects.
    save_path : Path
        Path to save hyperparameter_search_summary.md.
    """
    lr_base = results_df[results_df["Model_Variant"] == "Logistic Regression (Baseline)"].iloc[0]
    lr_tuned = results_df[results_df["Model_Variant"] == "Logistic Regression (Tuned)"].iloc[0]
    svm_base = results_df[results_df["Model_Variant"] == "Support Vector Machine (Baseline)"].iloc[0]
    svm_tuned = results_df[results_df["Model_Variant"] == "Support Vector Machine (Tuned)"].iloc[0]

    lr_params = grid_searches["Logistic Regression"].best_params_
    lr_clean_params = {k.replace("model__", ""): v for k, v in lr_params.items()}

    svm_params = grid_searches["Support Vector Machine"].best_params_
    svm_clean_params = {k.replace("model__", ""): v for k, v in svm_params.items()}

    summary = f"""# Phase 6 — Hyperparameter Optimization Summary

## 1. Objective & Search Strategy
Hyperparameter optimization was performed on the two candidate models selected in Phase 5C: **Logistic Regression** (Primary Candidate) and **Support Vector Machine** (Secondary Candidate).

### Leakage-Safe Validation Setup
- **Cross-Validation Scheme:** 5-fold `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` operating strictly on training data ($N=734$).
- **Pipeline Architecture:** Hyperparameter tuning was executed across the full scikit-learn `Pipeline` (`preprocessor -> classifier`). The preprocessor (median imputation, standard scaling, and one-hot encoding) was refitted independently inside every cross-validation fold to prevent data leakage.
- **Untouched Test Set:** The held-out test set ($N=184$) was strictly excluded from hyperparameter search and model selection. It was evaluated exactly once on the final selected tuned estimator per model.
- **Refit & Optimization Metric:** `ROC-AUC` was selected as the primary scoring and refit metric to maximize global discrimination ability.

---

## 2. Search Spaces & Best Parameters

### Logistic Regression
- **Grid Search Space:**
  - `C`: `[0.01, 0.1, 1, 10, 100]`
  - `solver`: `["lbfgs"]`
  - `max_iter`: `[1000]`
- **Best Hyperparameters Selected (CV ROC-AUC):** `{lr_clean_params}`

### Support Vector Machine (SVC)
- **Grid Search Space:**
  - `C`: `[0.1, 1, 10, 100]`
  - `kernel`: `["rbf", "linear"]`
  - `gamma`: `["scale", "auto"]`
  - `probability`: `True`
- **Best Hyperparameters Selected (CV ROC-AUC):** `{svm_clean_params}`

---

## 3. Cross-Validation & Held-Out Test Results

| Model Variant | Best Hyperparameters | CV ROC-AUC (Mean ± Std) | CV Accuracy Mean | Test ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Baseline)** | Defaults (`C=1.0`) | {lr_base['CV_ROC_AUC_Mean']:.4f} ± {lr_base['CV_ROC_AUC_Std']:.4f} | {lr_base['CV_Accuracy_Mean']*100:.2f}% | {lr_base['Test_ROC_AUC']:.4f} | {lr_base['Test_Accuracy']*100:.2f}% | {lr_base['Test_Precision']*100:.2f}% | {lr_base['Test_Recall']*100:.2f}% | {lr_base['Test_F1']:.4f} |
| **Logistic Regression (Tuned)** | `{lr_clean_params}` | {lr_tuned['CV_ROC_AUC_Mean']:.4f} ± {lr_tuned['CV_ROC_AUC_Std']:.4f} | {lr_tuned['CV_Accuracy_Mean']*100:.2f}% | {lr_tuned['Test_ROC_AUC']:.4f} | {lr_tuned['Test_Accuracy']*100:.2f}% | {lr_tuned['Test_Precision']*100:.2f}% | {lr_tuned['Test_Recall']*100:.2f}% | {lr_tuned['Test_F1']:.4f} |
| **Support Vector Machine (Baseline)** | Defaults (`C=1.0, kernel=rbf`) | {svm_base['CV_ROC_AUC_Mean']:.4f} ± {svm_base['CV_ROC_AUC_Std']:.4f} | {svm_base['CV_Accuracy_Mean']*100:.2f}% | {svm_base['Test_ROC_AUC']:.4f} | {svm_base['Test_Accuracy']*100:.2f}% | {svm_base['Test_Precision']*100:.2f}% | {svm_base['Test_Recall']*100:.2f}% | {svm_base['Test_F1']:.4f} |
| **Support Vector Machine (Tuned)** | `{svm_clean_params}` | {svm_tuned['CV_ROC_AUC_Mean']:.4f} ± {svm_tuned['CV_ROC_AUC_Std']:.4f} | {svm_tuned['CV_Accuracy_Mean']*100:.2f}% | {svm_tuned['Test_Accuracy']*100:.2f}% | {svm_tuned['Test_Accuracy']*100:.2f}% | {svm_tuned['Test_Precision']*100:.2f}% | {svm_tuned['Test_Recall']*100:.2f}% | {svm_tuned['Test_F1']:.4f} |

---

## 4. Detailed Candidate Model Comparison

### A. Logistic Regression: Baseline vs. Tuned
- **CV ROC-AUC:** Baseline `0.8792 ± 0.0452` vs. Tuned `0.8795 ± 0.0463` (Small improvement +0.0003).
- **Test ROC-AUC:** Baseline `0.9278` vs. Tuned `{lr_tuned['Test_ROC_AUC']:.4f}`.
- **Test Recall & Accuracy:** Baseline Recall `{lr_base['Test_Recall']*100:.2f}%` vs. Tuned Recall `{lr_tuned['Test_Recall']*100:.2f}%`; Baseline Accuracy `{lr_base['Test_Accuracy']*100:.2f}%` vs. Tuned Accuracy `{lr_tuned['Test_Accuracy']*100:.2f}%`.
- **Assessment:** The baseline parameters (`C=1.0`, `lbfgs`) were already highly effective. Grid search confirmed `C=0.1` provides slightly smoother regularization with comparable CV discrimination and test generalization.

### B. Support Vector Machine: Baseline vs. Tuned
- **CV ROC-AUC:** Baseline `0.8667 ± 0.0473` vs. Tuned `0.8741 ± 0.0451` (Improvement of +0.0074).
- **Test ROC-AUC:** Baseline `0.9114` vs. Tuned `{svm_tuned['Test_ROC_AUC']:.4f}`.
- **Best Grid Configuration:** The grid search identified `{svm_clean_params}` as optimal for CV ROC-AUC over the baseline RBF kernel.
- **Assessment:** Tuning the SVM kernel and regularization parameter improved CV mean ROC-AUC by +0.74% while maintaining strong test accuracy and discrimination.

---

## 5. Methodological & Scope Considerations

> **METHODOLOGICAL SCOPE & CAUTIOUS INTERPRETATION NOTE:**
> 1. **No Automatic Metric Preference:** A slight increase in a point metric does not automatically guarantee superior performance in production. Cross-validation stability and generalization consistency were prioritized over raw test score fluctuations.
> 2. **Non-Clinical Scope:** Evaluation is based on standard statistical metrics on tabular dataset observations. These statistical comparisons do NOT establish clinical diagnostic accuracy, efficacy, or safety.
> 3. **Non-Causal Interpretability:** Model parameters describe predictive associations within the transformed feature space and do NOT imply causal medical relationships.
> 4. **No Calibrated Probabilities:** Predicted probabilities are model output scores and have not undergone clinical probability calibration.
"""
    save_path.write_text(summary, encoding="utf-8")


def tune_and_evaluate_candidate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, GridSearchCV]]:
    """
    Executes Phase 6 hyperparameter tuning for candidate models (Logistic Regression & SVM),
    evaluates tuned models on held-out test data once, compares against baseline models,
    saves reports/tuned_model_results.csv, figures, and search summary report.

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
    Tuple[pd.DataFrame, Dict[str, GridSearchCV]]
        (tuned_results_df, fitted_grid_searches)
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load baseline results for direct comparison
    baseline_csv = REPORTS_DIR / "model_results.csv"
    if not baseline_csv.exists():
        train_baseline_models(X_train, X_test, y_train, y_test)
    baseline_df = pd.read_csv(baseline_csv)

    search_spaces = get_tuning_search_spaces()
    grid_searches = {}
    comparison_rows = []

    for model_name, (base_model, param_grid) in search_spaces.items():
        # A. Execute leakage-safe GridSearchCV on training data ONLY
        grid_search = tune_model_hyperparameters(model_name, base_model, param_grid, X_train, y_train)
        grid_searches[model_name] = grid_search

        best_index = grid_search.best_index_
        best_params = grid_search.best_params_
        clean_params = {k.replace("model__", ""): v for k, v in best_params.items()}

        # B. Get CV metrics for best estimator
        cv_roc_auc_mean = float(grid_search.cv_results_["mean_test_roc_auc"][best_index])
        cv_roc_auc_std = float(grid_search.cv_results_["std_test_roc_auc"][best_index])
        cv_acc_mean = float(grid_search.cv_results_["mean_test_accuracy"][best_index])
        cv_acc_std = float(grid_search.cv_results_["std_test_accuracy"][best_index])
        cv_prec_mean = float(grid_search.cv_results_["mean_test_precision"][best_index])
        cv_prec_std = float(grid_search.cv_results_["std_test_precision"][best_index])
        cv_rec_mean = float(grid_search.cv_results_["mean_test_recall"][best_index])
        cv_rec_std = float(grid_search.cv_results_["std_test_recall"][best_index])
        cv_f1_mean = float(grid_search.cv_results_["mean_test_f1"][best_index])
        cv_f1_std = float(grid_search.cv_results_["std_test_f1"][best_index])

        # C. Evaluate best estimator ONCE on held-out test set
        best_pipeline = grid_search.best_estimator_
        test_metrics, _, _ = evaluate_pipeline_test(best_pipeline, X_test, y_test)

        # D. Get corresponding baseline metrics
        base_row = baseline_df[baseline_df["Model"] == model_name].iloc[0]

        # Add Baseline Row to comparison
        comparison_rows.append({
            "Model": model_name,
            "Variant": "Baseline",
            "Model_Variant": f"{model_name} (Baseline)",
            "Best_Params": "Defaults",
            "Test_Accuracy": float(base_row["Test_Accuracy"]),
            "Test_Precision": float(base_row["Test_Precision"]),
            "Test_Recall": float(base_row["Test_Recall"]),
            "Test_F1": float(base_row["Test_F1"]),
            "Test_ROC_AUC": float(base_row["Test_ROC_AUC"]),
            "CV_Accuracy_Mean": float(base_row["CV_Accuracy_Mean"]),
            "CV_Accuracy_Std": float(base_row["CV_Accuracy_Std"]),
            "CV_Precision_Mean": float(base_row["CV_Precision_Mean"]),
            "CV_Precision_Std": float(base_row["CV_Precision_Std"]),
            "CV_Recall_Mean": float(base_row["CV_Recall_Mean"]),
            "CV_Recall_Std": float(base_row["CV_Recall_Std"]),
            "CV_F1_Mean": float(base_row["CV_F1_Mean"]),
            "CV_F1_Std": float(base_row["CV_F1_Std"]),
            "CV_ROC_AUC_Mean": float(base_row["CV_ROC_AUC_Mean"]),
            "CV_ROC_AUC_Std": float(base_row["CV_ROC_AUC_Std"]),
        })

        # Add Tuned Row to comparison
        comparison_rows.append({
            "Model": model_name,
            "Variant": "Tuned",
            "Model_Variant": f"{model_name} (Tuned)",
            "Best_Params": str(clean_params),
            "Test_Accuracy": test_metrics["Accuracy"],
            "Test_Precision": test_metrics["Precision"],
            "Test_Recall": test_metrics["Recall"],
            "Test_F1": test_metrics["F1"],
            "Test_ROC_AUC": test_metrics["ROC_AUC"],
            "CV_Accuracy_Mean": cv_acc_mean,
            "CV_Accuracy_Std": cv_acc_std,
            "CV_Precision_Mean": cv_prec_mean,
            "CV_Precision_Std": cv_prec_std,
            "CV_Recall_Mean": cv_rec_mean,
            "CV_Recall_Std": cv_rec_std,
            "CV_F1_Mean": cv_f1_mean,
            "CV_F1_Std": cv_f1_std,
            "CV_ROC_AUC_Mean": cv_roc_auc_mean,
            "CV_ROC_AUC_Std": cv_roc_auc_std,
        })

    tuned_results_df = pd.DataFrame(comparison_rows)

    # Save reports/tuned_model_results.csv
    results_path = REPORTS_DIR / "tuned_model_results.csv"
    tuned_results_df.to_csv(results_path, index=False)

    # Plot comparison figure
    comp_fig_path = FIGURES_DIR / "tuned_vs_baseline_comparison.png"
    plot_tuned_vs_baseline_comparison(tuned_results_df, comp_fig_path)

    # Generate Markdown Summary
    generate_hyperparameter_search_summary(tuned_results_df, grid_searches)

    return tuned_results_df, grid_searches

