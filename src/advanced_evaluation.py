"""
Advanced Evaluation Module for Disease Diagnosis Prediction (Phase 7).
Provides functions for deep evaluation of tuned candidate models:
- Confusion Matrix & Classification Metrics (TN, FP, FN, TP, Specificity, FPR, FNR)
- ROC Curves & ROC-AUC
- Precision-Recall Curves & Average Precision Scores
- Training-only Out-of-Fold Threshold Analysis (0.10 to 0.90)
- Training-only Out-of-Fold Probability Calibration & Brier Scores
- Comprehensive Advanced Evaluation Summary table generation
"""

from pathlib import Path
from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

from src.config import REPORTS_DIR, FIGURES_DIR, RANDOM_STATE
from src.train import build_model_pipeline, train_model
from src.evaluate import evaluate_pipeline_test


def get_tuned_candidate_models() -> Dict[str, BaseEstimator]:
    """
    Returns the two tuned candidate models selected in Phase 6.

    Returns
    -------
    Dict[str, BaseEstimator]
        Mapping model_name -> configured estimator instance.
    """
    return {
        "Tuned Logistic Regression": LogisticRegression(
            C=0.1, solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE
        ),
        "Tuned Support Vector Machine": SVC(
            C=100, kernel="linear", gamma="scale", probability=True, random_state=RANDOM_STATE
        ),
    }


def evaluate_tuned_confusion_matrices(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    save_csv: bool = True,
    save_fig: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
    """
    Evaluates tuned candidate models on held-out test data, computing exact confusion matrix
    counts (TN, FP, FN, TP) and diagnostic error metrics (Recall/Sensitivity, Specificity, FPR, FNR).

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
    save_csv : bool
        Whether to save reports/tuned_confusion_matrix_summary.csv.
    save_fig : bool
        Whether to save reports/figures/tuned_confusion_matrices.png.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Pipeline]]
        (summary_df, fitted_pipelines)
    """
    candidates = get_tuned_candidate_models()
    rows = []
    fitted_pipelines = {}

    for name, model in candidates.items():
        pipeline = train_model(name, model, X_train, y_train)
        fitted_pipelines[name] = pipeline

        y_pred = pipeline.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0) # Sensitivity
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = f1_score(y_test, y_pred, zero_division=0)

        rows.append({
            "Model": name,
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "Sensitivity": rec,
            "Specificity": spec,
            "False_Positive_Rate": fpr,
            "False_Negative_Rate": fnr,
            "F1": f1,
        })

    summary_df = pd.DataFrame(rows)

    if save_csv:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(REPORTS_DIR / "tuned_confusion_matrix_summary.csv", index=False)

    if save_fig:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        labels = ["No Disease (0)", "Disease (1)"]

        for idx, (name, pipeline) in enumerate(fitted_pipelines.items()):
            y_pred = pipeline.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            ax = axes[idx]
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=labels,
                yticklabels=labels,
                ax=ax,
                annot_kws={"size": 14, "weight": "bold"},
            )
            ax.set_title(f"Confusion Matrix — {name}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Predicted Class", fontsize=10)
            ax.set_ylabel("Actual Class", fontsize=10)

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "tuned_confusion_matrices.png", dpi=300)
        plt.close(fig)

    return summary_df, fitted_pipelines


def plot_tuned_roc_curves(
    fitted_pipelines: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: Path = FIGURES_DIR / "tuned_roc_curves.png",
) -> Dict[str, float]:
    """
    Generates combined ROC curves on held-out test data for tuned candidate models.

    Parameters
    ----------
    fitted_pipelines : Dict[str, Pipeline]
        Dictionary of fitted tuned Pipelines.
    X_test : pd.DataFrame
        Held-out test feature matrix.
    y_test : pd.Series
        Held-out test target vector.
    save_path : Path
        Path to save ROC curve figure PNG.

    Returns
    -------
    Dict[str, float]
        Mapping model_name -> Test_ROC_AUC score.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    auc_scores = {}

    for name, pipeline in fitted_pipelines.items():
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_prob = pipeline.decision_function(X_test)

        auc = roc_auc_score(y_test, y_prob)
        auc_scores[name] = auc
        fpr, tpr, _ = roc_curve(y_test, y_prob)

        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{name} (AUC = {auc:.4f})",
        )

    ax.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.5000)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.set_title("ROC Curves — Tuned Candidate Models (Held-Out Test Set)", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

    return auc_scores


def plot_tuned_precision_recall_curves(
    fitted_pipelines: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: Path = FIGURES_DIR / "tuned_precision_recall_curves.png",
) -> Dict[str, float]:
    """
    Generates Precision-Recall curves and computes Average Precision scores on held-out test data.

    Parameters
    ----------
    fitted_pipelines : Dict[str, Pipeline]
        Dictionary of fitted tuned Pipelines.
    X_test : pd.DataFrame
        Held-out test feature matrix.
    y_test : pd.Series
        Held-out test target vector.
    save_path : Path
        Path to save PR curve figure PNG.

    Returns
    -------
    Dict[str, float]
        Mapping model_name -> Average_Precision score.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ap_scores = {}

    baseline_ratio = y_test.mean()

    for name, pipeline in fitted_pipelines.items():
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_prob = pipeline.decision_function(X_test)

        ap = average_precision_score(y_test, y_prob)
        ap_scores[name] = ap
        precision, recall, _ = precision_recall_curve(y_test, y_prob)

        ax.plot(
            recall,
            precision,
            linewidth=2,
            label=f"{name} (AP = {ap:.4f})",
        )

    ax.axhline(baseline_ratio, color="k", linestyle="--", label=f"No Skill Baseline ({baseline_ratio:.3f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (Sensitivity)", fontsize=11)
    ax.set_ylabel("Precision (PPV)", fontsize=11)
    ax.set_title("Precision-Recall Curves — Tuned Candidate Models (Test Set)", fontsize=12, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

    return ap_scores


def analyze_threshold_tradeoffs(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    thresholds: list = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
    save_csv: bool = True,
    save_fig: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Performs leakage-safe threshold analysis using 5-fold Out-of-Fold (OoF) predictions
    on training data ONLY. Evaluates metrics at decision thresholds from 0.10 to 0.90.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target vector.
    thresholds : list
        List of decision threshold cutoff values.
    save_csv : bool
        Whether to save reports/threshold_analysis.csv.
    save_fig : bool
        Whether to save reports/figures/threshold_tradeoff.png.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, np.ndarray]]
        (threshold_df, oof_probabilities_dict)
    """
    candidates = get_tuned_candidate_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    oof_probs = {}

    for name, base_model in candidates.items():
        unfitted_pipeline = build_model_pipeline(base_model)
        y_prob_oof = cross_val_predict(
            unfitted_pipeline, X_train, y_train, cv=cv, method="predict_proba"
        )[:, 1]
        oof_probs[name] = y_prob_oof

        for t in thresholds:
            y_pred_t = (y_prob_oof >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_train, y_pred_t).ravel()

            acc = accuracy_score(y_train, y_pred_t)
            prec = precision_score(y_train, y_pred_t, zero_division=0)
            rec = recall_score(y_train, y_pred_t, zero_division=0) # Sensitivity
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
            fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = f1_score(y_train, y_pred_t, zero_division=0)

            rows.append({
                "Model": name,
                "Threshold": t,
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "Sensitivity": rec,
                "Specificity": spec,
                "False_Positive_Rate": fpr,
                "False_Negative_Rate": fnr,
                "F1": f1,
            })

    df_threshold = pd.DataFrame(rows)

    if save_csv:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        df_threshold.to_csv(REPORTS_DIR / "threshold_analysis.csv", index=False)

    if save_fig:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

        for idx, (name, _) in enumerate(candidates.items()):
            ax = axes[idx]
            model_df = df_threshold[df_threshold["Model"] == name]

            ax.plot(model_df["Threshold"], model_df["Sensitivity"], "o-", label="Sensitivity (Recall)", color="#e74c3c", linewidth=2)
            ax.plot(model_df["Threshold"], model_df["Specificity"], "s-", label="Specificity", color="#2ecc71", linewidth=2)
            ax.plot(model_df["Threshold"], model_df["Precision"], "^-", label="Precision", color="#3498db", linewidth=2)
            ax.plot(model_df["Threshold"], model_df["F1"], "d--", label="F1 Score", color="#9b59b6", linewidth=2)
            ax.axvline(0.50, color="gray", linestyle=":", label="Default Threshold (0.50)")

            ax.set_title(f"Threshold Trade-offs — {name} (Training OoF)", fontsize=11, fontweight="bold")
            ax.set_xlabel("Decision Threshold", fontsize=10)
            ax.set_ylabel("Metric Score", fontsize=10)
            ax.set_ylim([0.0, 1.05])
            ax.legend(loc="lower left", fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "threshold_tradeoff.png", dpi=300)
        plt.close(fig)

    return df_threshold, oof_probs


def analyze_probability_calibration(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    oof_probs: Dict[str, np.ndarray] = None,
    save_fig: bool = True,
) -> Dict[str, float]:
    """
    Evaluates probability calibration using training-only Out-of-Fold predictions,
    computing Brier scores and reliability curves.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target vector.
    oof_probs : Dict[str, np.ndarray], optional
        Pre-computed Out-of-Fold probabilities.
    save_fig : bool
        Whether to save reports/figures/calibration_curves.png.

    Returns
    -------
    Dict[str, float]
        Mapping model_name -> Brier Score.
    """
    candidates = get_tuned_candidate_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    if oof_probs is None:
        oof_probs = {}
        for name, base_model in candidates.items():
            unfitted_pipeline = build_model_pipeline(base_model)
            oof_probs[name] = cross_val_predict(
                unfitted_pipeline, X_train, y_train, cv=cv, method="predict_proba"
            )[:, 1]

    brier_scores = {}

    for name, probs in oof_probs.items():
        brier = brier_score_loss(y_train, probs)
        brier_scores[name] = brier

    if save_fig:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 6.5))

        ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated Baseline")

        for name, probs in oof_probs.items():
            brier = brier_scores[name]
            prob_true, prob_pred = calibration_curve(y_train, probs, n_bins=5, strategy="uniform")
            ax.plot(
                prob_pred,
                prob_true,
                "s-",
                linewidth=2,
                label=f"{name} (Brier = {brier:.4f})",
            )

        ax.set_xlabel("Mean Predicted Probability", fontsize=11)
        ax.set_ylabel("Fraction of Positives", fontsize=11)
        ax.set_title("Probability Calibration Curves — Training Out-of-Fold Predictions", fontsize=12, fontweight="bold")
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "calibration_curves.png", dpi=300)
        plt.close(fig)

    return brier_scores


def generate_advanced_evaluation_summary(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    save_csv: bool = True,
) -> pd.DataFrame:
    """
    Compiles reports/advanced_evaluation_summary.csv containing all baseline vs tuned metrics,
    confusion matrix details, ROC-AUC, Average Precision, Brier score, and CV vs test consistency.

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
    save_csv : bool
        Whether to save reports/advanced_evaluation_summary.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with comprehensive comparison across tuned candidates.
    """
    # 1. Run confusion matrix evaluation on test set
    cm_df, fitted_pipelines = evaluate_tuned_confusion_matrices(X_train, X_test, y_train, y_test, save_csv=True, save_fig=True)

    # 2. Run ROC curve evaluation on test set
    roc_scores = plot_tuned_roc_curves(fitted_pipelines, X_test, y_test)

    # 3. Run PR curve evaluation on test set
    ap_scores = plot_tuned_precision_recall_curves(fitted_pipelines, X_test, y_test)

    # 4. Run threshold analysis on training OoF
    df_threshold, oof_probs = analyze_threshold_tradeoffs(X_train, y_train, save_csv=True, save_fig=True)

    # 5. Run calibration analysis on training OoF
    brier_scores = analyze_probability_calibration(X_train, y_train, oof_probs=oof_probs, save_fig=True)

    # Load Phase 6 tuned results CSV for CV metrics
    tuned_csv = REPORTS_DIR / "tuned_model_results.csv"
    if tuned_csv.exists():
        tuned_res_df = pd.read_csv(tuned_csv)
    else:
        tuned_res_df = pd.DataFrame()

    summary_rows = []
    candidates = get_tuned_candidate_models()

    for name in candidates.keys():
        cm_row = cm_df[cm_df["Model"] == name].iloc[0]

        # Extract CV metrics from Phase 6 tuned results if present
        cv_roc_mean = np.nan
        cv_roc_std = np.nan
        if not tuned_res_df.empty:
            variant_match = tuned_res_df[tuned_res_df["Model_Variant"].str.contains(name.replace("Tuned ", ""))]
            if not variant_match.empty:
                tuned_sub = variant_match[variant_match["Variant"] == "Tuned"]
                if not tuned_sub.empty:
                    cv_roc_mean = float(tuned_sub["CV_ROC_AUC_Mean"].values[0])
                    cv_roc_std = float(tuned_sub["CV_ROC_AUC_Std"].values[0])

        summary_rows.append({
            "Model": name,
            "CV_ROC_AUC_Mean": cv_roc_mean,
            "CV_ROC_AUC_Std": cv_roc_std,
            "Test_ROC_AUC": float(roc_scores[name]),
            "Test_Accuracy": float(cm_row["Accuracy"]),
            "Test_Precision": float(cm_row["Precision"]),
            "Test_Recall": float(cm_row["Recall"]),
            "Test_Specificity": float(cm_row["Specificity"]),
            "Test_F1": float(cm_row["F1"]),
            "Test_FPR": float(cm_row["False_Positive_Rate"]),
            "Test_FNR": float(cm_row["False_Negative_Rate"]),
            "Average_Precision": float(ap_scores[name]),
            "Brier_Score": float(brier_scores[name]),
        })

    adv_summary_df = pd.DataFrame(summary_rows)

    if save_csv:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        adv_summary_df.to_csv(REPORTS_DIR / "advanced_evaluation_summary.csv", index=False)

    return adv_summary_df


def run_advanced_evaluation(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Executes all Phase 7 advanced evaluation analyses and generates reports/figures.

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
    pd.DataFrame
        Comprehensive advanced evaluation summary DataFrame.
    """
    return generate_advanced_evaluation_summary(X_train, X_test, y_train, y_test, save_csv=True)
