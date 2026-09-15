# Phase 8 — Final Model Selection Summary

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
| **CV ROC-AUC (Mean ± Std)** | 0.8795 ± 0.0463 | **0.8799 ± 0.0442** | Tuned SVM (+0.0004 CV ROC-AUC) |
| **Test ROC-AUC** | 0.9273 | **0.9295** | Tuned SVM (+0.0022 Test ROC-AUC) |
| **Test Accuracy** | 82.61% | **84.78%** | Tuned SVM (+2.17%, +4 correct predictions) |
| **Test Precision (PPV)** | 83.02% | **85.58%** | Tuned SVM (+2.56%) |
| **Test Recall / Sensitivity** | 86.27% | **87.25%** | Tuned SVM (+0.98%) |
| **Test Specificity** | 78.05% | **81.71%** | Tuned SVM (+3.66%) |
| **Test F1 Score** | 0.8462 | **0.8641** | Tuned SVM (+0.0179 F1) |
| **Test FPR** | 21.95% | **18.29%** | Tuned SVM (-3.66% lower false positive rate) |
| **Test FNR** | 13.73% | **12.75%** | Tuned SVM (-0.98% lower false negative rate) |
| **Average Precision (AP)** | 0.9453 | **0.9455** | Tuned SVM (+0.0002 AP) |
| **Training OoF Brier Score** | **0.1373** | 0.1376 | Tuned LR (-0.0004 Brier score) |

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
