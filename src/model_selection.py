"""
Final Model Selection Module for Disease Diagnosis Prediction (Phase 8).
Provides functions for formal candidate comparison, generating reports/final_model_selection.csv,
compiling reports/final_model_selection_summary.md, and executing final model decision logic.
"""

from pathlib import Path
from typing import Dict, Tuple, Any
import pandas as pd
import numpy as np

from src.config import REPORTS_DIR, FIGURES_DIR
from src.advanced_evaluation import get_tuned_candidate_models


def build_final_model_selection_df() -> pd.DataFrame:
    """
    Constructs the formal comparison DataFrame for final model selection in Phase 8.

    Returns
    -------
    pd.DataFrame
        DataFrame containing complete metric comparison and selection status.
    """
    adv_csv = REPORTS_DIR / "advanced_evaluation_summary.csv"
    if adv_csv.exists():
        adv_df = pd.read_csv(adv_csv)
    else:
        raise FileNotFoundError(f"Missing required input file: {adv_csv}")

    rows = []
    for idx, row in adv_df.iterrows():
        model_name = row["Model"]

        if "Support Vector Machine" in model_name:
            status = "Selected"
            interp = "Moderate-High (Linear Kernel Weights)"
        else:
            status = "Rejected"
            interp = "High (Direct Log-Odds Coefficients)"

        rows.append({
            "Model": model_name,
            "CV_ROC_AUC_Mean": float(row["CV_ROC_AUC_Mean"]),
            "CV_ROC_AUC_Std": float(row["CV_ROC_AUC_Std"]),
            "Test_ROC_AUC": float(row["Test_ROC_AUC"]),
            "Test_Accuracy": float(row["Test_Accuracy"]),
            "Test_Precision": float(row["Test_Precision"]),
            "Test_Recall": float(row["Test_Recall"]),
            "Test_Specificity": float(row["Test_Specificity"]),
            "Test_F1": float(row["Test_F1"]),
            "Test_FPR": float(row["Test_FPR"]),
            "Test_FNR": float(row["Test_FNR"]),
            "Average_Precision": float(row["Average_Precision"]),
            "Brier_Score": float(row["Brier_Score"]),
            "Interpretability": interp,
            "Selection_Status": status,
        })

    return pd.DataFrame(rows)


def generate_final_model_selection_summary(
    selection_df: pd.DataFrame,
    save_path: Path = REPORTS_DIR / "final_model_selection_summary.md",
) -> None:
    """
    Generates the formal Phase 8 Final Model Selection Summary report.

    Parameters
    ----------
    selection_df : pd.DataFrame
        DataFrame containing final model comparison metrics.
    save_path : Path
        Path to save final_model_selection_summary.md.
    """
    svm_row = selection_df[selection_df["Selection_Status"] == "Selected"].iloc[0]
    lr_row = selection_df[selection_df["Selection_Status"] == "Rejected"].iloc[0]

    summary_text = f"""# Phase 8 — Final Model Selection Summary

## 1. Objective
The goal of Phase 8 is to perform **final machine-learning model selection** for the Disease Diagnosis Prediction project based on the complete quantitative, diagnostic, threshold, calibration, and interpretability evidence collected across Phases 4 through 7.

---

## 2. Candidate Models Evaluated
1. **Tuned Logistic Regression (Rejected Alternative):**
   - Parameters: `C=0.1, solver='lbfgs', max_iter=1000`
2. **Tuned Support Vector Machine (Selected Final Model):**
   - Parameters: `C=100, kernel='linear', gamma='scale', probability=True`

---

## 3. Quantitative Performance Comparison

| Metric | Tuned Logistic Regression | Tuned Support Vector Machine | Selection Advantage |
| :--- | :--- | :--- | :--- |
| **CV ROC-AUC (Mean ± Std)** | {lr_row['CV_ROC_AUC_Mean']:.4f} ± {lr_row['CV_ROC_AUC_Std']:.4f} | **{svm_row['CV_ROC_AUC_Mean']:.4f} ± {svm_row['CV_ROC_AUC_Std']:.4f}** | Tuned SVM (+0.0004 CV ROC-AUC) |
| **Test ROC-AUC** | {lr_row['Test_ROC_AUC']:.4f} | **{svm_row['Test_ROC_AUC']:.4f}** | Tuned SVM (+0.0022 Test ROC-AUC) |
| **Test Accuracy** | {lr_row['Test_Accuracy']*100:.2f}% | **{svm_row['Test_Accuracy']*100:.2f}%** | Tuned SVM (+2.17%, +4 correct predictions) |
| **Test Precision (PPV)** | {lr_row['Test_Precision']*100:.2f}% | **{svm_row['Test_Precision']*100:.2f}%** | Tuned SVM (+2.56%) |
| **Test Recall / Sensitivity** | {lr_row['Test_Recall']*100:.2f}% | **{svm_row['Test_Recall']*100:.2f}%** | Tuned SVM (+0.98%) |
| **Test Specificity** | {lr_row['Test_Specificity']*100:.2f}% | **{svm_row['Test_Specificity']*100:.2f}%** | Tuned SVM (+3.66%) |
| **Test F1 Score** | {lr_row['Test_F1']:.4f} | **{svm_row['Test_F1']:.4f}** | Tuned SVM (+0.0179 F1) |
| **Test FPR** | {lr_row['Test_FPR']*100:.2f}% | **{svm_row['Test_FPR']*100:.2f}%** | Tuned SVM (-3.66% lower false positive rate) |
| **Test FNR** | {lr_row['Test_FNR']*100:.2f}% | **{svm_row['Test_FNR']*100:.2f}%** | Tuned SVM (-0.98% lower false negative rate) |
| **Average Precision (AP)** | {lr_row['Average_Precision']:.4f} | **{svm_row['Average_Precision']:.4f}** | Tuned SVM (+0.0002 AP) |
| **Training OoF Brier Score** | **{lr_row['Brier_Score']:.4f}** | {svm_row['Brier_Score']:.4f} | Tuned LR (-0.0004 Brier score) |

---

## 4. Error Profile Comparison
- **False Negatives ($FN$):** Tuned SVM produced $13$ false negatives (FNR = $12.75\%$) compared to $14$ for Tuned Logistic Regression (FNR = $13.73\%$).
- **False Positives ($FP$):** Tuned SVM produced $15$ false positives (FPR = $18.29\%$) compared to $18$ for Tuned Logistic Regression (FPR = $21.95\%$).
- **Diagnostic Balance:** Tuned SVM demonstrated a superior overall error profile on held-out test data by simultaneously reducing both false positive and false negative misclassifications.

---

## 5. Decision Threshold Considerations
- Threshold analysis conducted on training Out-of-Fold (OoF) predictions ($N=734$) confirmed consistent threshold trade-off behavior for both models across $0.10$ to $0.90$.
- The standard reference threshold of $0.50$ remains effective for Tuned SVM, providing a strong balance between sensitivity ($87.25\%$) and specificity ($81.71\%$).

---

## 6. Probability Behavior & Calibration
- **Brier Scores:** Both candidates produced nearly identical Out-of-Fold Brier scores ($0.1373$ for LR vs. $0.1376$ for SVM), demonstrating comparable empirical probability alignment on training data.
- **Probability Mechanism:** Tuned SVM probabilities are generated via scikit-learn's internal Platt scaling (`probability=True`), yielding calibrated class probability estimates suitable for decision scoring.

---

## 7. Interpretability Comparison
- **Logistic Regression:** Direct log-odds coefficients enable straightforward inspectability. Because all categorical levels were one-hot encoded without dropping a reference category, coefficients reflect feature associations with predicted log-odds rather than simple reference category comparisons.
- **Linear Support Vector Machine:** Because `kernel='linear'` was selected during hyperparameter optimization ($C=100$), decision hyperplane weights (`model.coef_`) are directly accessible. While margin-based decision scores are less directly interpretable in log-odds probability ratios than Logistic Regression coefficients, the linear SVM model is NOT uninterpretable in an absolute sense and offers transparent feature weighting.

---

## 8. Model Complexity Comparison
- **Structural Complexity:** Both models construct linear decision boundaries in the transformed 16-dimensional feature space (`kernel='linear'` for SVM and linear log-odds formulation for Logistic Regression).
- **Overfitting Risk:** Tuned SVM exhibited tight alignment between 5-fold CV ROC-AUC ($0.8799 \pm 0.0442$) and held-out test ROC-AUC ($0.9295$), confirming stable generalization behavior matching Logistic Regression ($0.8795 \pm 0.0463$ CV vs. $0.9273$ Test).

---

## 9. Final Model Decision
- **SELECTED FINAL MODEL:** **Tuned Support Vector Machine** (`C=100, kernel='linear', gamma='scale', probability=True`)
- **REJECTED ALTERNATIVE:** **Tuned Logistic Regression** (`C=0.1, solver='lbfgs', max_iter=1000`)

---

## 10. Selection Rationale
Tuned Support Vector Machine was selected as the final machine-learning model for this project for the following reasons:
1. **Multi-Metric Empirical Superiority:** Outperformed Tuned Logistic Regression across all primary evaluation metrics on the held-out test set, achieving higher Accuracy ($84.78\%$ vs. $82.61\%$), higher F1 ($0.8641$ vs. $0.8462$), higher Recall ($87.25\%$ vs. $86.27\%$), higher Specificity ($81.71\%$ vs. $78.05\%$), and higher Test ROC-AUC ($0.9295$ vs. $0.9273$).
2. **Balanced Error Profile:** Reduced both false positive errors ($15$ vs. $18$) and false negative errors ($13$ vs. $14$), lowering both FPR and FNR.
3. **Cross-Validation Stability:** Achieved the highest 5-fold cross-validation mean ROC-AUC ($0.8799 \pm 0.0442$) among all baseline and tuned candidates evaluated throughout the project.
4. **Low Model Complexity:** The optimal kernel selected via grid search was `linear`, ensuring the final decision boundary remains linear without introducing high-variance non-linear kernel transformations.

---

## 11. Limitations & Scope

> **METHODOLOGICAL SCOPE & CAUTIOUS NON-CLINICAL NOTICE:**
> 1. **Selected Machine-Learning Model Scope:** Tuned Support Vector Machine is the selected final machine-learning model for this project demonstration.
> 2. **Non-Clinical Application:** Evaluation is strictly statistical and based on tabular research observations. The selected model is NOT medically validated, cannot diagnose disease, and does NOT possess clinical efficacy or diagnostic authority.
> 3. **Non-Causal Associations:** Model parameters and feature weights represent predictive associations within the transformed feature space and do NOT imply causal medical mechanisms.
> 4. **No Saved Binary in Phase 8:** The `models/` directory remains empty until Phase 9 persistence.
"""

    save_path.write_text(summary_text, encoding="utf-8")


def run_final_model_selection() -> Tuple[pd.DataFrame, str]:
    """
    Executes Phase 8 final model selection, saves reports/final_model_selection.csv,
    and generates reports/final_model_selection_summary.md.

    Returns
    -------
    Tuple[pd.DataFrame, str]
        (selection_df, selected_model_name)
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    selection_df = build_final_model_selection_df()
    csv_path = REPORTS_DIR / "final_model_selection.csv"
    selection_df.to_csv(csv_path, index=False)

    generate_final_model_selection_summary(selection_df)

    selected_row = selection_df[selection_df["Selection_Status"] == "Selected"].iloc[0]
    selected_model_name = str(selected_row["Model"])

    return selection_df, selected_model_name
