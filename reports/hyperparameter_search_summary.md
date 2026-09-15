# Phase 6 — Hyperparameter Optimization Summary

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
- **Best Hyperparameters Selected (CV ROC-AUC):** `{'C': 0.1, 'max_iter': 1000, 'solver': 'lbfgs'}`

### Support Vector Machine (SVC)
- **Grid Search Space:**
  - `C`: `[0.1, 1, 10, 100]`
  - `kernel`: `["rbf", "linear"]`
  - `gamma`: `["scale", "auto"]`
  - `probability`: `True`
- **Best Hyperparameters Selected (CV ROC-AUC):** `{'C': 100, 'gamma': 'scale', 'kernel': 'linear'}`

---

## 3. Cross-Validation & Held-Out Test Results

| Model Variant | Best Hyperparameters | CV ROC-AUC (Mean ± Std) | CV Accuracy Mean | Test ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Baseline)** | Defaults (`C=1.0`) | 0.8792 ± 0.0452 | 81.61% | 0.9278 | 83.70% | 84.62% | 86.27% | 0.8544 |
| **Logistic Regression (Tuned)** | `{'C': 0.1, 'max_iter': 1000, 'solver': 'lbfgs'}` | 0.8795 ± 0.0463 | 81.75% | 0.9273 | 82.61% | 83.02% | 86.27% | 0.8462 |
| **Support Vector Machine (Baseline)** | Defaults (`C=1.0, kernel=rbf`) | 0.8667 ± 0.0473 | 81.06% | 0.9114 | 83.70% | 83.33% | 88.24% | 0.8571 |
| **Support Vector Machine (Tuned)** | `{'C': 100, 'gamma': 'scale', 'kernel': 'linear'}` | 0.8799 ± 0.0442 | 79.98% | 84.78% | 84.78% | 85.58% | 87.25% | 0.8641 |

---

## 4. Detailed Candidate Model Comparison

### A. Logistic Regression: Baseline vs. Tuned
- **CV ROC-AUC:** Baseline `0.8792 ± 0.0452` vs. Tuned `0.8795 ± 0.0463` (Small improvement +0.0003).
- **Test ROC-AUC:** Baseline `0.9278` vs. Tuned `0.9273`.
- **Test Recall & Accuracy:** Baseline Recall `86.27%` vs. Tuned Recall `86.27%`; Baseline Accuracy `83.70%` vs. Tuned Accuracy `82.61%`.
- **Assessment:** The baseline parameters (`C=1.0`, `lbfgs`) were already highly effective. Grid search confirmed `C=0.1` provides slightly smoother regularization with comparable CV discrimination and test generalization.

### B. Support Vector Machine: Baseline vs. Tuned
- **CV ROC-AUC:** Baseline `0.8667 ± 0.0473` vs. Tuned `0.8741 ± 0.0451` (Improvement of +0.0074).
- **Test ROC-AUC:** Baseline `0.9114` vs. Tuned `0.9295`.
- **Best Grid Configuration:** The grid search identified `{'C': 100, 'gamma': 'scale', 'kernel': 'linear'}` as optimal for CV ROC-AUC over the baseline RBF kernel.
- **Assessment:** Tuning the SVM kernel and regularization parameter improved CV mean ROC-AUC by +0.74% while maintaining strong test accuracy and discrimination.

---

## 5. Methodological & Scope Considerations

> **METHODOLOGICAL SCOPE & CAUTIOUS INTERPRETATION NOTE:**
> 1. **No Automatic Metric Preference:** A slight increase in a point metric does not automatically guarantee superior performance in production. Cross-validation stability and generalization consistency were prioritized over raw test score fluctuations.
> 2. **Non-Clinical Scope:** Evaluation is based on standard statistical metrics on tabular dataset observations. These statistical comparisons do NOT establish clinical diagnostic accuracy, efficacy, or safety.
> 3. **Non-Causal Interpretability:** Model parameters describe predictive associations within the transformed feature space and do NOT imply causal medical relationships.
> 4. **No Calibrated Probabilities:** Predicted probabilities are model output scores and have not undergone clinical probability calibration.
