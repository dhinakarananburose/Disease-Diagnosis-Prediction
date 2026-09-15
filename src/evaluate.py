"""
Evaluation Module for Disease Diagnosis Prediction.
Provides reusable functions for computing classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC),
evaluating model pipelines on test data, plotting confusion matrices, combined ROC curves, and model comparison charts.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline

from src.config import FIGURES_DIR


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None
) -> Dict[str, float]:
    """
    Computes classification performance metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True target labels.
    y_pred : np.ndarray
        Predicted target labels (0 or 1).
    y_prob : np.ndarray, optional
        Predicted class probabilities or decision function scores.

    Returns
    -------
    Dict[str, float]
        Dictionary containing Accuracy, Precision, Recall, F1, and ROC_AUC metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    roc_auc = np.nan
    if y_prob is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            roc_auc = np.nan

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC_AUC": roc_auc,
    }


def evaluate_pipeline_test(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Evaluates a fitted model Pipeline on held-out test data.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted scikit-learn Pipeline.
    X_test : pd.DataFrame
        Held-out test feature matrix.
    y_test : pd.Series
        Held-out test target vector.

    Returns
    -------
    Tuple[Dict[str, float], np.ndarray, np.ndarray]
        (metrics_dict, y_pred, y_prob)
    """
    y_pred = pipeline.predict(X_test)

    y_prob = None
    if hasattr(pipeline, "predict_proba"):
        try:
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        except (AttributeError, IndexError):
            y_prob = None

    if y_prob is None and hasattr(pipeline, "decision_function"):
        try:
            y_prob = pipeline.decision_function(X_test)
        except AttributeError:
            y_prob = None

    metrics = compute_classification_metrics(y_test, y_pred, y_prob)
    return metrics, y_pred, y_prob


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str, save_path: Path
) -> None:
    """
    Plots and saves confusion matrix figure.

    Parameters
    ----------
    y_true : np.ndarray
        True target labels.
    y_pred : np.ndarray
        Predicted target labels.
    model_name : str
        Model name for title.
    save_path : Path
        Path to save figure PNG.
    """
    cm = confusion_matrix(y_true, y_pred)
    labels = ["No Heart Disease (0)", "Heart Disease (1)"]

    fig, ax = plt.subplots(figsize=(6, 5))
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
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Class", fontsize=11)
    ax.set_ylabel("Actual Class", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_combined_roc_curves(
    model_probs: Dict[str, Tuple[np.ndarray, float]],
    y_test: pd.Series,
    save_path: Path,
) -> None:
    """
    Plots combined ROC curves for all baseline models on test set.

    Parameters
    ----------
    model_probs : Dict[str, Tuple[np.ndarray, float]]
        Dictionary mapping model_name -> (y_prob, roc_auc_score).
    y_test : pd.Series
        True test target labels.
    save_path : Path
        Path to save figure PNG.
    """
    fig, ax = plt.subplots(figsize=(8, 6.5))

    for model_name, (y_prob, auc_score) in model_probs.items():
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            ax.plot(
                fpr,
                tpr,
                linewidth=2,
                label=f"{model_name} (AUC = {auc_score:.3f})",
            )

    ax.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.set_title("Combined ROC Curves — Baseline Classification Models", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_model_comparison(results_df: pd.DataFrame, save_path: Path) -> None:
    """
    Plots comparison bar chart across models and key metrics.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame containing model results.
    save_path : Path
        Path to save figure PNG.
    """
    metrics = ["Test_Accuracy", "Test_Precision", "Test_Recall", "Test_F1", "Test_ROC_AUC"]
    plot_df = pd.melt(
        results_df,
        id_vars=["Model"],
        value_vars=metrics,
        var_name="Metric",
        value_name="Score",
    )
    plot_df["Metric"] = plot_df["Metric"].str.replace("Test_", "")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=plot_df,
        x="Model",
        y="Score",
        hue="Metric",
        palette="Set2",
        ax=ax,
    )
    ax.set_title("Baseline Model Performance Comparison (Held-Out Test Set)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Metric Score", fontsize=11)
    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower right", fontsize=10)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_tuned_vs_baseline_comparison(results_df: pd.DataFrame, save_path: Path) -> None:
    """
    Plots a comparison bar chart comparing baseline vs tuned candidate models.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame containing baseline and tuned model evaluation results.
    save_path : Path
        Path to save figure PNG.
    """
    metrics = ["Test_Accuracy", "Test_Precision", "Test_Recall", "Test_F1", "Test_ROC_AUC"]
    plot_df = pd.melt(
        results_df,
        id_vars=["Model_Variant"],
        value_vars=metrics,
        var_name="Metric",
        value_name="Score",
    )
    plot_df["Metric"] = plot_df["Metric"].str.replace("Test_", "")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=plot_df,
        x="Model_Variant",
        y="Score",
        hue="Metric",
        palette="Blues_d",
        ax=ax,
    )
    ax.set_title("Tuned vs. Baseline Candidate Model Comparison (Held-Out Test Set)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Metric Score", fontsize=11)
    ax.set_xlabel("Model Variant", fontsize=11)
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower right", fontsize=10)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

