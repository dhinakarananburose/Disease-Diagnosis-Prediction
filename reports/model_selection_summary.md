# Model Selection Summary

## 1. Objective
This report provides a multi-criteria evidence-based comparison and candidate model selection for **Disease Diagnosis Prediction Using Clinical Data and Classification Models**.

## 2. Models Evaluated
Five baseline classification architectures were evaluated within leakage-safe scikit-learn `Pipeline` objects:
1. **Logistic Regression** (`LogisticRegression(max_iter=1000, random_state=42)`)
2. **K-Nearest Neighbors** (`KNeighborsClassifier(n_neighbors=5)`)
3. **Decision Tree** (`DecisionTreeClassifier(random_state=42)`)
4. **Random Forest** (`RandomForestClassifier(n_estimators=100, random_state=42)`)
5. **Support Vector Machine** (`SVC(probability=True, random_state=42)`)

## 3. Test Performance Comparison
Evaluated on the held-out test set ($N=184$, $80/20$ stratified split):
- **Logistic Regression:** Accuracy = $83.70\%$, Precision = $84.62\%$, Recall = $86.27\%$, F1 = $85.44\%$, ROC-AUC = **$0.9278$**.
- **K-Nearest Neighbors:** Accuracy = **$84.24\%$**, Precision = $84.11\%$, Recall = **$88.24\%$**, F1 = **$86.12\%$**, ROC-AUC = $0.8998$.
- **Support Vector Machine:** Accuracy = $83.70\%$, Precision = $83.33\%$, Recall = **$88.24\%$**, F1 = $85.71\%$, ROC-AUC = $0.9114$.
- **Random Forest:** Accuracy = $81.52\%$, Precision = $82.69\%$, Recall = $84.31\%$, F1 = $83.50\%$, ROC-AUC = $0.9112$.
- **Decision Tree:** Accuracy = $71.20\%$, Precision = $73.79\%$, Recall = $74.51\%$, F1 = $74.15\%$, ROC-AUC = $0.7079$.

## 4. Cross-Validation Comparison
Evaluated across 5-fold Stratified Cross-Validation on $X_{train}$ ($N=734$):
- **Logistic Regression:** CV Accuracy = **$81.61\% \pm 4.46\%$**, CV Precision = $82.83\%$, CV Recall = $84.72\%$, CV F1 = $83.69\%$, CV ROC-AUC = **$0.8792$**.
- **Support Vector Machine:** CV Accuracy = $81.06\% \pm 4.06\%$, CV Precision = $81.62\%$, CV Recall = $85.71\%$, CV F1 = $83.47\%$, CV ROC-AUC = $0.8667$.
- **Random Forest:** CV Accuracy = $80.66\% \pm 4.49\%$, CV Precision = $82.11\%$, CV Recall = $83.99\%$, CV F1 = $82.90\%$, CV ROC-AUC = $0.8578$.
- **K-Nearest Neighbors:** CV Accuracy = $79.84\% \pm 4.48\%$, CV Precision = $79.10\%$, CV Recall = $86.95\%$, CV F1 = $82.75\%$, CV ROC-AUC = $0.8411$.
- **Decision Tree:** CV Accuracy = $70.71\% \pm 3.30\%$, CV Precision = $73.89\%$, CV Recall = $72.66\%$, CV F1 = $73.23\%$, CV ROC-AUC = $0.7047$.

## 5. Error Analysis
Evaluated confusion matrix elements ($TN$, $FP$, $FN$, $TP$) and diagnostic error rates on the test set ($N=184$):
- **Logistic Regression:** $TN=66$, $FP=16$, $FN=14$, $TP=88$, Sensitivity = $86.27\%$, Specificity = **$80.49\%$**, FNR = $13.73\%$, FPR = **$19.51\%$**.
- **K-Nearest Neighbors:** $TN=65$, $FP=17$, $FN=12$, $TP=90$, Sensitivity = **$88.24\%$**, Specificity = $79.27\%$, FNR = **$11.76\%$**, FPR = $20.73\%$.
- **Support Vector Machine:** $TN=64$, $FP=18$, $FN=12$, $TP=90$, Sensitivity = **$88.24\%$**, Specificity = $78.05\%$, FNR = **$11.76\%$**, FPR = $21.95\%$.
- **Random Forest:** $TN=64$, $FP=18$, $FN=16$, $TP=86$, Sensitivity = $84.31\%$, Specificity = $78.05\%$, FNR = $15.69\%$, FPR = $21.95\%$.
- **Decision Tree:** $TN=55$, $FP=27$, $FN=26$, $TP=76$, Sensitivity = $74.51\%$, Specificity = $67.07\%$, FNR = $25.49\%$, FPR = $32.93\%$.

> **CAUTIOUS ERROR TRADE-OFF NOTE:** False negatives represent observations belonging to class 1 that were predicted as class 0. False positives represent observations belonging to class 0 that were predicted as class 1. For a screening-oriented application, false negatives may be particularly consequential; however, the appropriate error trade-off depends on the intended clinical use and requires clinical validation.

## 6. Generalization Analysis
Evaluated performance gaps ($	ext{Test Metric} - 	ext{CV Metric Mean}$):
- **Logistic Regression:** Accuracy Gap = $+2.09\%$, F1 Gap = $+1.74\%$, ROC-AUC Gap = $+4.86\%$.
- **Support Vector Machine:** Accuracy Gap = $+2.63\%$, F1 Gap = $+2.24\%$, ROC-AUC Gap = $+4.47\%$.
- **Random Forest:** Accuracy Gap = $+0.86\%$, F1 Gap = $+0.59\%$, ROC-AUC Gap = $+5.34\%$.
- **Decision Tree:** Accuracy Gap = $+0.48\%$, F1 Gap = $+0.92\%$, ROC-AUC Gap = $+0.32\%$.
- **K-Nearest Neighbors:** Accuracy Gap = $+4.40\%$, F1 Gap = $+3.37\%$, ROC-AUC Gap = $+5.87\%$.

> **OBSERVED PERFORMANCE CONSISTENCY NOTE:** The difference between held-out test performance and cross-validation mean provides a useful comparison of observed performance across evaluation procedures, but these gaps alone do not establish overfitting.

## 7. Interpretability Analysis
> **NON-CAUSAL INTERPRETATION & ONE-HOT ENCODING METHODOLOGICAL NOTE:**
> The Logistic Regression coefficients indicate the direction and magnitude of association of each transformed feature with the model's predicted log-odds of class 1. Because all observed categorical levels are one-hot encoded, categorical coefficient interpretation should be treated cautiously and should not be interpreted as a simple comparison against an omitted reference category.
> Random Forest feature importance reflects the feature's contribution to the fitted tree ensemble and does not establish causality.

Key multi-model feature overlap observations:
- `cat__cp_asymptomatic`: Among the transformed features, this feature has a relatively large absolute coefficient in the fitted Logistic Regression model and high model feature importance in Random Forest.
- `bool__exang`: Exercise-induced angina (`bool__exang`): Among the transformed features, this feature has a relatively large absolute coefficient in the fitted Logistic Regression model and high model feature importance in Random Forest.
- `num__oldpeak`: ST depression induced by exercise (`num__oldpeak`): Among the transformed features, this feature has a relatively large absolute coefficient in the fitted Logistic Regression model and high model feature importance in Random Forest.
- `num__thalch`: Maximum heart rate achieved (`num__thalch`): Among the transformed features, this feature has a relatively large absolute coefficient in the fitted Logistic Regression model and the highest model feature importance in Random Forest.

## 8. Primary Candidate Model
**PRIMARY CANDIDATE MODEL:** **Logistic Regression**

**Justification:**
1. **Highest Discrimination:** Achieved the highest test ROC-AUC (**$0.9278$**) and highest cross-validation mean ROC-AUC (**$0.8792$**).
2. **Highest CV Accuracy:** Achieved the highest 5-fold CV mean accuracy (**$81.61\% \pm 4.46\%$**).
3. **Lowest False Positive Rate:** Achieved the highest test specificity (**$80.49\%$**) and lowest test FPR (**$19.51\%$**), providing a balanced diagnostic profile ($86.27\%$ sensitivity, $14$ FNs).
4. **Mathematical Interpretability:** Direct inspection of log-odds coefficients enables transparent feature analysis and model auditing.
5. **Phase 6 Suitability:** Well-suited for probability calibration, decision threshold adjustment, and hyperparameter tuning (`C`, `penalty`, `solver`).

## 9. Secondary Candidate Model
**SECONDARY CANDIDATE MODEL:** **Support Vector Machine (SVC)**

**Justification:**
1. **Tied-Highest Sensitivity:** Achieved tied-highest test sensitivity (**$88.24\%$**, $12$ FNs) alongside KNN, with lower FNR ($11.76\%$).
2. **Strong Discrimination:** Achieved high test ROC-AUC (**$0.9114$**) and strong test accuracy ($83.70\%$).
3. **Consistent Cross-Validation:** Achieved strong CV mean accuracy ($81.06\% \pm 4.06\%$) and CV ROC-AUC ($0.8667$).
4. **Non-Linear Alternative:** Provides a robust non-linear decision boundary alternative to linear Logistic Regression for Phase 6 hyperparameter tuning (`C`, `gamma`, `kernel`).

## 10. Models Not Selected
1. **K-Nearest Neighbors (KNN):**
   - **Reason for Exclusion:** Although KNN achieved high test accuracy ($84.24\%$) and recall ($88.24\%$), it exhibited lower CV mean accuracy ($79.84\%$), larger ROC-AUC generalization gap ($+5.87\%$), and lacks parametric global model interpretability.
2. **Random Forest:**
   - **Reason for Exclusion:** Solid test ROC-AUC ($0.9112$), but lower baseline test accuracy ($81.52\%$) and test F1 ($83.50\%$) compared to Logistic Regression ($83.70\%$) and SVM ($83.70\%$).
3. **Decision Tree:**
   - **Reason for Exclusion:** Substantially lower test accuracy ($71.20\%$), lower CV mean accuracy ($70.71\%$), and weak ROC-AUC ($0.7079$) due to single-tree variance and instability.

## 11. Selection Rationale
Candidate selection was conducted using a multi-criteria evidence-based framework without combining metrics into an arbitrary weighted score.

**Key Trade-off Analysis:**
- **Discrimination vs. Sensitivity:** Logistic Regression provides superior global discrimination (ROC-AUC $0.9278$) and specificity ($80.49\%$), whereas SVM prioritizes sensitivity ($88.24\%$) with slightly lower specificity ($78.05\%$).
- **Parametric Interpretability vs. Non-Linear Boundaries:** Logistic Regression offers linear log-odds transparency, while SVM captures complex non-linear feature interactions via RBF kernel space.
- **Cross-Validation Stability:** Both Logistic Regression ($81.61\%$) and SVM ($81.06\%$) demonstrated superior cross-validation mean performance compared to KNN ($79.84\%$) and Decision Tree ($70.71\%$).

## 12. Limitations
1. **Single Train/Test Split:** Evaluation is based on a single 80/20 stratified split ($184$ test observations), resulting in sample variance for point estimates.
2. **Default Hyperparameters:** Baseline models were evaluated with standard default parameter settings without prior hyperparameter optimization.
3. **Clinical Scope:** Statistical evaluation on tabular research data does not establish clinical efficacy or diagnostic validity.

## 13. Recommendation for Phase 6
1. **Primary Focus:** Advance **Logistic Regression** (Primary Candidate) and **Support Vector Machine** (Secondary Candidate) to Phase 6.
2. **Tuning Objectives:**
   - Conduct systematic hyperparameter tuning (`C`, `penalty`, `solver` for Logistic Regression; `C`, `gamma`, `kernel` for SVM).
   - Perform probability calibration and decision threshold optimization to evaluate sensitivity/specificity trade-offs.
   - Re-evaluate tuned candidates against Phase 4/5 baseline benchmarks.
