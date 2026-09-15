# Disease Diagnosis Prediction Using Clinical Data and Classification Models

**Author**: Advanced Agentic Coding & Machine Learning Team  
**Project**: Disease Diagnosis Prediction  
**Status**: Final Verified Technical Report (Phases 1–14 Complete)  
**Artifact Version**: 1.0.0  

---

## Abstract
This report presents the complete end-to-end design, implementation, evaluation, persistence, API integration, and containerization of a machine-learning system for disease diagnosis prediction based on clinical data. Using a dataset of 918 raw clinical observations with 10 predictor features, we established a leakage-safe preprocessing pipeline combining median and most-frequent imputation, standard scaling, and one-hot encoding. Five classification algorithms—Logistic Regression, K-Nearest Neighbors (KNN), Decision Tree, Random Forest, and Support Vector Machine (SVM)—were evaluated across 5-fold Stratified Cross-Validation and a held-out stratified test set (80/20 split, $N_{\text{train}}=734$, $N_{\text{test}}=184$). 

Hyperparameter optimization via `GridSearchCV` on training data selected a Tuned Support Vector Machine ($C=100$, kernel=`linear`, `gamma=`scale`, `probability=`True`) as the final model. On the held-out test set, the final tuned SVM achieved an **Accuracy of 84.78%**, **Precision of 85.58%**, **Recall of 87.25%**, **Specificity of 81.71%**, **F1-score of 0.8641**, and **ROC-AUC of 0.9295** (Confusion Matrix: $\text{TN}=67, \text{FP}=15, \text{FN}=13, \text{TP}=89$). The complete pipeline was persisted as a Joblib artifact (`models/final_model.joblib`), integrated into a reusable prediction module (`src/predict.py`), exposed via a production-structured FastAPI REST application (`app/main.py`), validated with 142 automated unit and integration tests, and containerized using Docker (`python:3.11-slim`).

> **Mandatory Disclaimer**: This machine-learning application provides predictions based on historical clinical dataset patterns and is not a clinically validated diagnostic system.

---

## Keywords
Clinical Machine Learning, Heart Disease Prediction, Support Vector Machine, Preprocessing Pipeline, Model Persistence, FastAPI, Docker Containerization, Model Evaluation.

---

## 1. Introduction
Predictive modeling using machine learning in healthcare provides opportunities to assist clinical decision-making by identifying risk patterns in patient data. However, deploying machine learning in biomedical contexts requires rigorous methodology to prevent data leakage, evaluate model generalization, interpret feature associations, and ensure reproducible software deployment.

This project implements a complete, production-structured machine learning application for predicting the presence of heart disease based on patient clinical characteristics. The codebase follows strict software engineering and data science best practices, progressing through data loading, exploratory data analysis, pipeline construction, baseline model comparison, model interpretability, error analysis, hyperparameter tuning, advanced evaluation, model persistence, reusable inference module design, web API development, comprehensive testing, and containerization.

---

## 2. Problem Statement
Given a vector of 10 clinical predictor variables $\mathbf{x} = [x_1, x_2, \dots, x_{10}]^T$ representing patient demographic, physiological, and diagnostic measurements, the objective is to learn a mapping $f: \mathbf{x} \mapsto y$ where $y \in \{0, 1\}$ represents the presence ($y=1$) or absence ($y=0$) of heart disease.

---

## 3. Objectives
1. Perform thorough exploratory data analysis (EDA) to evaluate distributions, missingness, and feature-target relationships.
2. Construct a leakage-safe scikit-learn preprocessing pipeline fitting transformers strictly on training data.
3. Train and compare five baseline classification algorithms under 5-fold Stratified Cross-Validation and held-out test evaluation.
4. Conduct interpretability analysis to examine model feature associations and importance scores.
5. Perform error, threshold, and probability calibration analysis on candidate models.
6. Execute leakage-safe hyperparameter optimization using `GridSearchCV` on training data.
7. Select a single final model based on multi-metric quantitative evaluation.
8. Persist the complete trained pipeline artifact (`models/final_model.joblib`) and metadata.
9. Implement a reusable prediction module (`src/predict.py`) handling single and batch inference.
10. Develop a production-structured FastAPI application (`app/main.py`).
11. Conduct comprehensive test validation (142 automated tests).
12. Package the application into a lightweight, production-ready Docker container (`Dockerfile`).

---

## 4. Scope & Limitations Boundaries
- **In-Scope**: Open-source clinical dataset analysis, scikit-learn pipeline engineering, model selection, Joblib artifact persistence, FastAPI REST API development, `pytest` validation suite, and Docker containerization.
- **Out-of-Scope**: Prospective clinical trials, direct electronic health record (EHR) integration, cloud infrastructure deployment, and medical diagnostic claims.

---

## 5. Dataset Description
The dataset consists of 918 observations across 11 original columns sourced from clinical heart disease records. 

- **Original Target**: `num` (integer values 0, 1, 2, 3, 4 representing disease severity stage).
- **Binary Target Creation**: A binary target `target` was created according to:
  $$ \text{target} = \begin{cases} 1 & \text{if } \text{num} > 0 \\ 0 & \text{if } \text{num} = 0 \end{cases} $$
- **Class Distribution**:
  - Class 0 (No Heart Disease): 410 observations (44.66%)
  - Class 1 (Heart Disease Present): 508 observations (55.34%)

### Feature Specifications
The 10 predictor variables are grouped into numerical, categorical, and boolean types:

| Feature | Type | Data Representation / Range | Description |
| :--- | :--- | :--- | :--- |
| `age` | Numerical | 28 to 77 years | Patient age in years |
| `trestbps` | Numerical | 80 to 200 mm Hg | Resting blood pressure |
| `chol` | Numerical | 0 to 603 mg/dl | Serum cholesterol (0 = unrecorded) |
| `thalch` | Numerical | 60 to 202 bpm | Maximum heart rate achieved |
| `oldpeak` | Numerical | -2.6 to 6.2 mm | ST depression induced by exercise |
| `sex` | Categorical | 'Male', 'Female' (or 1, 0) | Patient sex |
| `cp` | Categorical | 'typical angina', 'atypical angina', 'non-anginal', 'asymptomatic' | Chest pain type |
| `restecg` | Categorical | 'normal', 'ST-T wave abnormality', 'left ventricular hypertrophy' | Resting ECG results |
| `fbs` | Boolean | True/False (or 1/0) | Fasting blood sugar > 120 mg/dl |
| `exang` | Boolean | True/False (or 1/0) | Exercise induced angina |

---

## 6. Data Quality Assessment
- **Raw CSV Integrity**: Hash MD5 `13c9cfee54ce2b1552ef7d787a5d8be9`, exactly 918 rows.
- **Missing Values in Raw CSV**: 0 null/NaN entries in the raw CSV file on disk.
- **Duplicate Rows**: 0 duplicate rows.
- **Cholesterol Zero Treatment**: `chol == 0` occurs in 172 observations. Rather than dropping observations or modifying the raw CSV on disk, `chol == 0` is treated as unrecorded/missing and mapped to `np.nan` strictly in the in-memory modeling DataFrame, enabling median imputation within the pipeline.
- **Blood Pressure Zero Count**: `trestbps == 0` count is 0 (minimum observed trestbps is 80 mm Hg).
- **Negative Oldpeak Values**: `oldpeak < 0` occurs in 12 observations (minimum -2.6 mm). These observations represent valid clinical electrocardiographic readings and were fully preserved.

---

## 7. Exploratory Data Analysis (EDA)

### Key Univariate & Bivariate Findings
- **Age**: Mean age of 53.5 years ($\text{SD} = 9.4$, range 28–77).
- **Chest Pain (`cp`)**: 54.0% of patients present with asymptomatic chest pain.
- **Exercise Angina (`exang`)**: Present in 40.3% of observations.
- **Strongest Target Correlations**:
  - `oldpeak`: $+0.40$
  - `thalch`: $-0.40$
  - `age`: $+0.28$
- **Multicollinearity Check**: Pairwise absolute correlations among numerical predictors were all below $0.40$, indicating absence of severe multicollinearity.
- **Outlier Assessment (IQR Method)**:
  - `chol`: 173 outliers (including 172 zero values)
  - `trestbps`: 28 outliers
  - `oldpeak`: 16 outliers
  - `thalch`: 2 outliers
  - `age`: 0 outliers  
  *All outlier observations were preserved to reflect real-world clinical variance.*

### Referenced EDA Artifacts
- `reports/figures/target_distribution.png`
- `reports/figures/numerical_distributions.png`
- `reports/figures/numerical_boxplots.png`
- `reports/figures/categorical_distributions.png`
- `reports/figures/feature_target_relationships.png`
- `reports/figures/correlation_heatmap.png`
- `reports/figures/outlier_analysis.png`

---

## 8. Data Preprocessing & Train/Test Split

### Stratified Train/Test Split
To preserve class proportions, a stratified split was executed (`TEST_SIZE = 0.20`, `RANDOM_STATE = 42`):
- **Training Set ($N_{\text{train}} = 734$)**: 406 positive (55.31%), 328 negative (44.69%)
- **Held-Out Test Set ($N_{\text{test}} = 184$)**: 102 positive (55.43%), 82 negative (44.57%)

### Pipeline Architecture (`ColumnTransformer`)
Preprocessing transformers were fit strictly on `X_train`:
1. **Numerical Pipeline**:
   - `SimpleImputer(strategy='median', add_indicator=True)` (preserves predictive signal of unrecorded values)
   - `StandardScaler()`
2. **Categorical Pipeline**:
   - `SimpleImputer(strategy='most_frequent')`
   - `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`
3. **Boolean Pipeline**:
   - `FunctionTransformer(_cast_boolean_to_int)`
   - `SimpleImputer(strategy='most_frequent')`

---

## 9. Baseline Classification Models
Five candidate algorithms were wrapped into scikit-learn Pipelines and evaluated using default parameters:
1. Logistic Regression (`solver='lbfgs'`, `max_iter=1000`)
2. K-Nearest Neighbors (`n_neighbors=5`)
3. Decision Tree (`random_state=42`)
4. Random Forest (`n_estimators=100`, `random_state=42`)
5. Support Vector Machine (`C=1.0`, `kernel='rbf'`, `probability=True`, `random_state=42`)

### Baseline Test Performance Metrics

| Model | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 83.70% | 84.62% | 86.27% | 0.8544 | **0.9278** |
| **K-Nearest Neighbors** | 84.24% | 84.11% | 88.24% | 0.8612 | 0.8998 |
| **Decision Tree** | 71.20% | 73.79% | 74.51% | 0.7415 | 0.7079 |
| **Random Forest** | 81.52% | 82.69% | 84.31% | 0.8350 | 0.9112 |
| **Support Vector Machine** | 83.70% | 83.33% | 88.24% | 0.8571 | 0.9114 |

---

## 10. Cross-Validation Analysis
5-fold Stratified Cross-Validation was conducted on training data ($N_{\text{train}} = 734$) only:

| Model | CV Accuracy Mean ± Std | CV Precision Mean ± Std | CV Recall Mean ± Std | CV F1 Mean ± Std | CV ROC-AUC Mean ± Std |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.8161 ± 0.0446** | 0.8283 ± 0.0549 | 0.8472 ± 0.0152 | 0.8369 ± 0.0338 | **0.8792 ± 0.0452** |
| **KNN** | 0.7984 ± 0.0448 | 0.7910 ± 0.0515 | 0.8695 ± 0.0296 | 0.8275 ± 0.0344 | 0.8411 ± 0.0521 |
| **Decision Tree** | 0.7071 ± 0.0330 | 0.7389 ± 0.0257 | 0.7266 ± 0.0492 | 0.7323 ± 0.0360 | 0.7047 ± 0.0316 |
| **Random Forest** | 0.8066 ± 0.0449 | 0.8211 ± 0.0607 | 0.8399 ± 0.0106 | 0.8290 ± 0.0310 | 0.8578 ± 0.0482 |
| **SVM** | 0.8106 ± 0.0406 | 0.8162 ± 0.0574 | 0.8571 ± 0.0128 | 0.8347 ± 0.0275 | 0.8667 ± 0.0473 |

### Referenced Figures
- `reports/figures/baseline_model_comparison.png`
- `reports/figures/roc_curves_baseline.png`

---

## 11. Model Interpretability (Phase 5A)

### Logistic Regression Feature Coefficients
Top observed coefficient weights in the fitted linear model:
- `cat__cp_asymptomatic`: $+0.9671$ (Strongest positive association with target)
- `bool__exang`: $+0.9395$
- `num__oldpeak`: $+0.5283$
- `cat__cp_typical angina`: $-0.7248$ (Strongest negative association with target)
- `num__thalch`: $-0.3384$

*Note: The `OneHotEncoder` uses `drop=None`, so coefficients represent model weights across encoded indicator features rather than contrasts against a single omitted baseline category.*

### Random Forest Feature Importances
Top Gini impurity reduction feature importances:
- `num__thalch`: $0.1371$
- `num__age`: $0.1288$
- `cat__cp_asymptomatic`: $0.1220$
- `num__oldpeak`: $0.1154$
- `num__chol`: $0.1121$

### Referenced Figures
- `reports/figures/logistic_regression_coefficients.png`
- `reports/figures/random_forest_feature_importance.png`

---

## 12. Error Profile & Confusion Matrix Analysis

### Baseline Held-Out Test Confusion Matrices ($N_{\text{test}}=184$)

| Model | TN | FP | FN | TP | Sensitivity | Specificity | FNR | FPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 66 | 16 | 14 | 88 | 86.27% | 80.49% | 13.73% | 19.51% |
| **KNN** | 65 | 17 | 12 | 90 | 88.24% | 79.27% | 11.76% | 20.73% |
| **Decision Tree** | 55 | 27 | 26 | 76 | 74.51% | 67.07% | 25.49% | 32.93% |
| **Random Forest** | 64 | 18 | 16 | 86 | 84.31% | 78.05% | 15.69% | 21.95% |
| **SVM** | 64 | 18 | 12 | 90 | 88.24% | 78.05% | 11.76% | 21.95% |

### Referenced Figures
- `reports/figures/confusion_matrix_logistic_regression.png`
- `reports/figures/confusion_matrix_svm.png`
- `reports/figures/model_error_rates.png`
- `reports/figures/test_vs_cv_performance.png`

---

## 13. Baseline Candidate Model Selection (Phase 5C)
Based on comparative performance, two candidate models were advanced to hyperparameter optimization:
1. **Primary Candidate**: **Logistic Regression** (highest baseline test ROC-AUC 0.9278, highest CV accuracy 81.61%, highest specificity 80.49%, lowest FPR 19.51%).
2. **Secondary Candidate**: **Support Vector Machine** (highest baseline test recall 88.24%, lowest FNR 11.76%, strong test ROC-AUC 0.9114).

*Rejection Rationale*: KNN was rejected due to lower CV accuracy (79.84%) and larger test/CV AUC gap. Decision Tree and Random Forest exhibited lower test accuracy and discrimination.

---

## 14. Hyperparameter Optimization (Phase 6)
Hyperparameter tuning was conducted using `GridSearchCV` with 5-fold StratifiedKFold on training data ($N_{\text{train}}=734$) only.

### Selected Hyperparameters
- **Tuned Logistic Regression**: `C=0.1`, `solver='lbfgs'`, `max_iter=1000`
- **Tuned Support Vector Machine**: `C=100`, `kernel='linear'`, `gamma='scale'`, `probability=True`, `random_state=42`

### Baseline vs. Tuned Performance Comparison

| Model | Variant | CV ROC-AUC | Test ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Baseline | 0.8792 | 0.9278 | 83.70% | 84.62% | 86.27% | 0.8544 |
| **Logistic Regression** | Tuned | 0.8795 | 0.9273 | 82.61% | 83.02% | 86.27% | 0.8462 |
| **SVM** | Baseline | 0.8667 | 0.9114 | 83.70% | 83.33% | 88.24% | 0.8571 |
| **SVM** | **Tuned** | **0.8799** | **0.9295** | **84.78%** | **85.58%** | **87.25%** | **0.8641** |

### Referenced Figures
- `reports/figures/tuned_vs_baseline_comparison.png`

---

## 15. Advanced Evaluation (Phase 7)

### Detailed Metrics of Tuned Candidate Models

| Metric | Tuned Logistic Regression | Tuned Support Vector Machine |
| :--- | :---: | :---: |
| **True Negatives (TN)** | 64 | **67** |
| **False Positives (FP)** | 18 | **15** |
| **False Negatives (FN)** | 14 | **13** |
| **True Positives (TP)** | 88 | **89** |
| **Test Accuracy** | 82.61% | **84.78%** |
| **Test Precision** | 83.02% | **85.58%** |
| **Test Recall (Sensitivity)** | 86.27% | **87.25%** |
| **Test Specificity** | 78.05% | **81.71%** |
| **Test False Positive Rate (FPR)** | 21.95% | **18.29%** |
| **Test False Negative Rate (FNR)** | 13.73% | **12.75%** |
| **Test F1-Score** | 0.8462 | **0.8641** |
| **Test ROC-AUC** | 0.9273 | **0.9295** |
| **Average Precision (PR-AUC)** | 0.9453 | **0.9455** |
| **Out-of-Fold Brier Score** | **0.1373** | 0.1376 |

### Threshold Analysis & Calibration
- **Threshold Sweeps (0.10 to 0.90)**: Evaluated on out-of-fold training predictions. Lowering the decision threshold to 0.30 increased sensitivity to ~92.7% while lowering specificity. The default 0.50 threshold was retained as the standard reference.
- **Probability Calibration**: Both tuned models produced low out-of-fold Brier scores (~0.137), indicating reasonable probability alignment.

### Referenced Figures
- `reports/figures/tuned_confusion_matrices.png`
- `reports/figures/tuned_roc_curves.png`
- `reports/figures/tuned_precision_recall_curves.png`
- `reports/figures/threshold_tradeoff.png`
- `reports/figures/calibration_curves.png`

---

## 16. Final Model Selection (Phase 8)

### Selected Production Model
**Tuned Support Vector Machine (`SVC`)**
- `C = 100`
- `kernel = "linear"`
- `gamma = "scale"`
- `probability = True`
- `random_state = 42`

### Multi-Metric Decision Rationale
1. **Highest Overall Accuracy**: 84.78% (vs 82.61% for Tuned LR).
2. **Highest Discrimination**: Test ROC-AUC of 0.9295 and Average Precision of 0.9455.
3. **Superior Error Profile**: Achieved both fewer false positives ($\text{FP}=15$ vs $18$) and fewer false negatives ($\text{FN}=13$ vs $14$) than Tuned Logistic Regression.
4. **Strong Specificity & Sensitivity Balance**: Specificity of 81.71% and Sensitivity of 87.25%.
5. **Robust Cross-Validation Stability**: CV ROC-AUC of $0.8799 \pm 0.0442$.

---

## 17. Model Persistence (Phase 9)
The final model was fit on the full training dataset ($N_{\text{train}}=734$) and persisted:
- **Model Artifact**: `models/final_model.joblib` (11.7 KB)
- **Metadata JSON**: `models/model_metadata.json` (1.1 KB)

*Data Separation Rule*: The held-out test set ($N_{\text{test}}=184$) was NOT included during final model fitting to preserve evaluation integrity.

---

## 18. Reusable Prediction Pipeline (Phase 10)
Implemented in `src/predict.py`:
- `load_persisted_model()`: Loads Joblib pipeline artifact.
- `validate_prediction_input()`: Validates required features, numerical types, and boolean formats.
- `prepare_prediction_dataframe()`: Converts `chol == 0` to `np.nan` for pipeline median imputation and enforces column ordering matching `MODEL_FEATURES`.
- `predict_single()`: Returns single observation prediction dictionary.
- `predict_batch()`: Processes pandas DataFrames without mutating original input payloads.

---

## 19. FastAPI Application Integration (Phase 11)
Implemented in `app/main.py` and `app/schemas.py`:
- **Lifespan Startup**: Model artifact loaded once at startup via `@asynccontextmanager` and attached to `app.state.model`.
- **Endpoints**:
  - `GET /` — API metadata info
  - `GET /health` — Service health and model load status
  - `POST /predict` — Single prediction endpoint
  - `POST /predict/batch` — Batch prediction endpoint
- **Error Handling**: Custom exception handlers convert `ValueError` and `TypeError` exceptions into HTTP 422 JSON error responses without exposing stack traces.

---

## 20. Dockerization (Phase 13)
Containerized using Docker and Docker Compose:
- **Dockerfile**: Based on `python:3.11-slim`, configures non-root user `appuser` (UID 1000), exposes port 8000, and starts `uvicorn app.main:app --host 0.0.0.0 --port 8000`.
- **.dockerignore**: Excludes development caches, virtual environments, and notebooks while preserving `app/`, `src/`, `models/`, and `requirements.txt`.
- **docker-compose.yml**: Configures service `api` mapping port `8000:8000`.

---

## 21. Testing and Validation (Phase 12)
Automated testing conducted via `pytest`:
- **Total Test Count**: **142 passed, 0 failed** in 43.12s.
- **Coverage**: Pipeline unit tests (100), API integration tests (37), Docker configuration tests (5).
- **Artifact SHA-256 Hashes**:
  - `models/final_model.joblib`: `16081825890dc9a3fae9c8fd2216272379ee7ac0f7712072f983a0bb897a1568`
  - `models/model_metadata.json`: `5ec6e595a0c7d5d0dc899611b1f3f3480c1792564c2113e6e39478329467c836`
- **Raw CSV MD5 Hash**: `13c9cfee54ce2b1552ef7d787a5d8be9` (918 rows, unmodified).

---

## 22. System Architecture

### End-to-End Data & Request Flow
```
User / HTTP Client
        │
        ▼
   FastAPI REST API (app/main.py)
        │
        ▼
   Pydantic Schema Validation (app/schemas.py)
        │
        ▼
   Inference Pipeline Module (src/predict.py)
        │
        ▼
   Persisted Scikit-Learn Pipeline (models/final_model.joblib)
   ├── Preprocessor (SimpleImputer, StandardScaler, OneHotEncoder)
   └── Classifier (Tuned Linear Support Vector Machine)
        │
        ▼
   Structured JSON Prediction Response
   {"predicted_class": 1, "predicted_probability": 0.8245, "model_name": "Tuned Support Vector Machine"}
```

### Docker Container Structure
```
Docker Container (disease-diagnosis-api)
 ├── Base OS: Debian Slim (python:3.11-slim)
 ├── Security: User appuser (UID 1000)
 ├── Server: Uvicorn (0.0.0.0:8000)
 ├── Application: app/ main.py & schemas.py
 ├── Logic: src/ predict.py & config.py
 └── Artifacts: models/ final_model.joblib
```

---

## 23. Technical Limitations
1. **Sample Size**: Dataset is limited to 918 observations.
2. **Single Dataset Split**: Evaluation relies on a single 80/20 stratified split and 5-fold CV.
3. **Non-Clinical Validation**: The model has not undergone external prospective clinical validation.
4. **Missing Data Imputation**: 172 cholesterol zero values relied on median imputation.
5. **Exploratory Thresholding**: Decision threshold analysis was exploratory; default 0.50 threshold was retained.
6. **Non-Causal Associations**: Feature coefficients reflect mathematical model associations, not causal medical effects.

---

## 24. Ethical & Responsible AI Considerations
- **Non-Clinical Scope**: Machine-learning predictions must serve as technical analytical demonstrations rather than standalone clinical diagnostic tools.
- **Error Risks**: False negatives (FNR 12.75%) and false positives (FPR 18.29%) carry clinical implications; human expert oversight is essential.
- **Data Privacy**: No patient personally identifiable information (PII) is contained in the open-source dataset.

---

## 25. Conclusion
This project successfully designed, validated, persisted, and containerized a machine-learning application for disease diagnosis prediction. The Tuned Support Vector Machine achieved 84.78% test accuracy and 0.9295 ROC-AUC. Combined with a reusable prediction module, FastAPI service, 142 automated tests, and Docker containerization, the codebase provides a complete, reproducible ML engineering workflow.

---

## 26. Future Work
- External prospective validation on independent multi-center clinical cohorts.
- Integration of clinical decision threshold tuning in consultation with medical domain experts.
- Addition of model monitoring, drift detection, and cloud infrastructure deployment (Kubernetes/AWS/GCP).
- Implementation of API authentication, rate limiting, and role-based access control.

---

## 27. References / Dataset Source
1. Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988). *Heart Disease Data Set*. UCI Machine Learning Repository.
2. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
3. FastAPI Documentation: https://fastapi.tiangolo.com/

---

## 28. Appendix: Technical Reference Summary

### Final Model Hyperparameters
- Algorithm: `sklearn.svm.SVC`
- `C`: `100`
- `kernel`: `"linear"`
- `gamma`: `"scale"`
- `probability`: `True`
- `random_state`: `42`

### Final Model Metrics Summary
- **Accuracy**: 84.78%
- **Precision**: 85.58%
- **Recall**: 87.25%
- **Specificity**: 81.71%
- **F1-Score**: 0.8641
- **ROC-AUC**: 0.9295
- **Average Precision**: 0.9455
- **Brier Score**: 0.1376

### Docker Build & Run Quick Reference
```bash
# Build Docker image
docker build -t disease-diagnosis-api .

# Run container via Docker Compose
docker compose up -d

# Stop container
docker compose down
```
