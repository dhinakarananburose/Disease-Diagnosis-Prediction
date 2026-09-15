# Portfolio Project Case Study: Disease Diagnosis Prediction

## 1. Project Title
**Disease Diagnosis Prediction Using Clinical Data and Classification Models**

## 2. One-Line Summary
A production-structured, end-to-end machine learning system providing heart disease risk classification with leak-free preprocessing, hyperparameter-tuned Support Vector Machine, FastAPI microservice, and Docker deployment.

## 3. Problem Statement
Cardiovascular disease remains a leading cause of mortality worldwide. Early diagnostic screening from clinical measurements (blood pressure, cholesterol, heart rate, electrocardiogram indicators) can assist healthcare workflows. The challenge is constructing an automated diagnostic prediction model that maintains rigorous statistical leakage isolation, robust probability calibration, and reproducible REST API deployment.

## 4. Solution Overview
Developed a full-stack ML solution using 918 patient observations. The system implements:
- Stratified 80/20 train-test splitting and 5-fold Stratified Cross-Validation on training data.
- Leakage-safe preprocessing pipeline (`StandardScaler` + `OneHotEncoder` fitted strictly on `X_train`).
- Benchmarking of 5 classification models (Logistic Regression, KNN, Decision Tree, Random Forest, SVM).
- Hyperparameter optimization via 5-fold Stratified `GridSearchCV`.
- Model persistence of the complete pipeline (`final_model.joblib`).
- Production FastAPI microservice (`app/main.py`) exposing REST endpoints.
- Dockerized environment (`python:3.11-slim`) verified with 142 automated tests.

## 5. Technologies Used
- **Core ML**: Python 3.11, `scikit-learn`, `pandas`, `numpy`, `joblib`
- **REST API**: `FastAPI`, `Uvicorn`, `Pydantic`
- **Testing & Quality**: `pytest`, `httpx`
- **Containerization**: `Docker`, `Docker Compose`

## 6. Dataset Summary
- **Source**: Cleveland / UCI Heart Disease dataset ($N=918$).
- **Features**: 10 clinical predictors (5 numerical: `age`, `trestbps`, `chol`, `thalch`, `oldpeak`; 3 categorical: `sex`, `cp`, `restecg`; 2 boolean: `fbs`, `exang`).
- **Target**: Binary classification (`0` = No Heart Disease, `1` = Heart Disease; 508 positive, 410 negative).

## 7. Machine Learning Methodology
- **Data Quality Handling**: Suspicious zero-cholesterol values (`chol == 0`, $N=172$) treated as missing and imputed via median inside the training pipeline. Negative `oldpeak` values preserved based on clinical validity.
- **Leakage Prevention**: All transformations (imputation, scaling, encoding) fitted exclusively on `X_train` ($N=734$) and applied to held-out test set `X_test` ($N=184$).

## 8. Model Benchmarking & Tuning
Evaluated 5 baseline models on the held-out test set ($N=184$):
- **Logistic Regression**: Accuracy 83.70%, Recall 86.27%, ROC-AUC 0.9278
- **KNN**: Accuracy 84.24%, Recall 88.24%, ROC-AUC 0.8998
- **Decision Tree**: Accuracy 71.20%, Recall 74.51%, ROC-AUC 0.7079
- **Random Forest**: Accuracy 81.52%, Recall 84.31%, ROC-AUC 0.9112
- **Support Vector Machine**: Accuracy 83.70%, Recall 88.24%, ROC-AUC 0.9114

Hyperparameter tuning on `X_train` via 5-fold `GridSearchCV` selected **Tuned Support Vector Machine** ($C=100$, linear kernel, `probability=True`).

## 9. Final Model Performance (Held-Out Test Set, $N=184$)
- **Accuracy**: 84.78%
- **Precision**: 85.58%
- **Recall (Sensitivity)**: 87.25%
- **Specificity**: 81.71%
- **F1 Score**: 0.8641
- **ROC-AUC**: 0.9295
- **Average Precision**: 0.9455
- **Confusion Matrix**: $\text{TN}=67, \text{FP}=15, \text{FN}=13, \text{TP}=89$

## 10. API & Containerization
Exposes RESTful endpoints:
- `GET /health`: Health status & model load verification.
- `POST /predict`: Single observation inference.
- `POST /predict/batch`: Batch observation inference.

Packaged in lightweight Docker image (`python:3.11-slim`) running Uvicorn under non-root execution (`appuser`). Verified exact numerical output consistency (0% class discrepancy, $<10^{-6}$ probability variance between local and containerized execution).

## 11. Key Learning Outcomes & Engineering Rigor
- Strict isolation between development (`X_train`) and evaluation (`X_test`) sets to eliminate data leakage.
- Pipeline encapsulation bundling transformers and estimator into a single joblib object.
- Test-driven validation passing 142 automated tests covering schema integrity, edge cases, batch processing, and container health.

## 12. Non-Clinical Disclaimer
*This project is an engineering prototype for demonstrating machine learning capabilities and software design principles. It is not a clinically validated medical device and must not be used for clinical decision-making.*
