# Disease Diagnosis Prediction Using Clinical Data and Classification Models

Production-structured machine learning application for heart disease risk classification. The project implements a leakage-safe preprocessing pipeline, hyperparameter-tuned Support Vector Machine (SVC) classifier, reusable prediction inference module, FastAPI REST microservice, and Docker containerization.

> [!WARNING]
> **Non-Clinical Disclaimer**: This application is a machine-learning research prototype built for demonstrating software design and predictive modeling workflows. It is **not a clinically validated diagnostic system** and must not be used for medical diagnosis or clinical decision-making.

---

## Project Performance Summary (Held-Out Test Set, $N=184$)

| Metric | Result |
| :--- | :--- |
| **Selected Final Model** | Tuned Support Vector Machine (`SVC`, $C=100$, linear kernel) |
| **Test Accuracy** | **84.78%** (156 / 184) |
| **Test Precision** | **85.58%** |
| **Test Recall (Sensitivity)** | **87.25%** (89 / 102 positive cases) |
| **Test Specificity** | **81.71%** (67 / 82 negative cases) |
| **Test F1 Score** | **0.8641** |
| **Test ROC-AUC** | **0.9295** |
| **Average Precision** | **0.9455** |
| **Test Confusion Matrix** | $\text{TN}=67, \text{FP}=15, \text{FN}=13, \text{TP}=89$ |

*Note: All evaluation metrics were computed on a held-out test dataset ($N=184$) that was completely isolated from preprocessing fitting, baseline training, and hyperparameter tuning.*

---

## System Architecture

```
HTTP Client / User
       │
       ▼
FastAPI Application (app/main.py)
       │
       ▼
Pydantic Schema Validation (app/schemas.py)
       │
       ▼
Inference Pipeline (src/predict.py)
       │
       ▼
Leakage-Safe Preprocessor (StandardScaler + OneHotEncoder)
       │
       ▼
Tuned Support Vector Machine Classifier (SVC, C=100, Linear)
       │
       ▼
Structured Prediction & Probability JSON Response
```

### Containerized Boundary

```
Docker Container (python:3.11-slim)
 ├── FastAPI (Uvicorn on 0.0.0.0:8000)
 ├── Input Validation & Inference Engine
 ├── Leakage-Safe Preprocessing Pipeline
 └── Persisted Model Artifact (models/final_model.joblib)
```

System architecture visual diagram available at [`reports/figures/system_architecture.png`](file:///c:/projects/Disease-Diagnosis-Prediction/reports/figures/system_architecture.png).

---

## Dataset & Publication Notice

The project utilizes the **Cleveland / UCI Heart Disease dataset** ($N=918$ observations, 11 original columns).

> [!NOTE]
> **Dataset Redistribution & Exclusion Policy**: The local raw CSV (`data/raw/heart_disease.csv`) is **intentionally excluded from public Git tracking** via `.gitignore` because explicit redistribution permissions for the exact local CSV file were not independently established.
> The repository includes all original project source code, pipeline scripts, Jupyter notebooks, technical reports, automated unit tests, persisted model artifacts, FastAPI endpoints, and Docker deployment configurations required to run and test the application.

---

## Dataset Setup & Reproducibility Instructions

To run local dataset training, data loading, or exploratory data analysis notebooks:

1. Obtain the Cleveland / UCI Heart Disease dataset from the official public source:
   - [UCI Machine Learning Repository — Heart Disease Data Set](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
2. Save the raw dataset CSV file locally to:
   ```
   data/raw/heart_disease.csv
   ```
3. Confirm dataset structure:
   - **Observations**: 918 rows
   - **Original Columns (11)**: `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalch`, `exang`, `oldpeak`, `num`
4. Execute the project test suite or scripts:
   ```bash
   pytest -q
   ```

*Note: The persisted model artifact (`models/final_model.joblib`) is included in the repository, so the FastAPI REST service and prediction pipeline function fully out-of-the-box without requiring local dataset downloading or retraining.*

---

## Repository Structure

```
Disease-Diagnosis-Prediction/
├── app/                        # FastAPI Web Microservice
│   ├── __init__.py
│   ├── main.py                 # FastAPI application routes & startup logic
│   └── schemas.py              # Pydantic input/output validation schemas
├── src/                        # Machine Learning Source Code Modules
│   ├── __init__.py
│   ├── config.py               # Paths, random seeds, and global constants
│   ├── data_loader.py          # Data ingestion and verification
│   ├── preprocessing.py       # Leakage-safe preprocessing transformers
│   ├── train.py                # Model training and baseline benchmarking
│   ├── evaluate.py             # Evaluation metrics and matrix calculations
│   ├── model_analysis.py       # Interpretability and coefficient extraction
│   ├── advanced_evaluation.py  # ROC curves, precision-recall, calibration
│   ├── model_selection.py      # Candidate selection logic
│   ├── model_persistence.py    # Pipeline serialization & metadata logging
│   └── predict.py              # Production inference prediction engine
├── data/
│   ├── raw/
│   │   └── .gitkeep            # Data directory placeholder (CSV excluded from Git)
│   └── processed/
│       └── .gitkeep            # Processed data directory placeholder
├── models/
│   ├── final_model.joblib      # Persisted scikit-learn pipeline artifact (~54.5 KB)
│   └── model_metadata.json     # Model hyperparameters & training metadata
├── notebooks/                  # Step-by-Step Jupyter Notebooks (Phases 1-9)
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_model_analysis.ipynb
│   ├── 06_hyperparameter_tuning.ipynb
│   ├── 07_advanced_evaluation.ipynb
│   ├── 08_model_selection.ipynb
│   └── 09_model_persistence.ipynb
├── reports/                    # Technical Documentation & Reports
│   ├── final_project_report.md # 33-section comprehensive technical report
│   ├── final_metrics_table.csv # Structured benchmark results
│   ├── project_configuration_summary.md
│   ├── public_repository_manifest.md # Public repository file classification
│   ├── license_and_publication_status.md # License and publication status report
│   ├── figures/                # Visualizations, ROC curves, confusion matrices
│   ├── github_description.md
│   ├── portfolio_project_description.md
│   ├── resume_project_entry.md
│   ├── linkedin_project_description.md
│   ├── github_screenshot_plan.md
│   ├── github_publication_checklist.md
│   └── github_portfolio_preparation_summary.md
├── tests/                      # Automated Pytest Test Suite
│   ├── conftest.py             # Pytest fixtures and mock client setups
│   ├── test_data_loader.py     # Ingestion & raw CSV integrity tests
│   ├── test_preprocessing.py   # Leakage isolation & transformer tests
│   ├── test_train.py           # Training logic & model fitting tests
│   ├── test_evaluate.py        # Metric calculation tests
│   ├── test_model_persistence.py # Joblib serialization & reproducibility
│   ├── test_predict.py         # Single & batch inference tests
│   ├── test_api.py             # FastAPI REST endpoint & schema tests
│   └── test_docker.py          # Docker configuration & parity tests
├── Dockerfile                  # Production multi-stage Docker build
├── docker-compose.yml          # Container service orchestration
├── .dockerignore               # Container build exclusions
├── .gitignore                # Git exclusions (data/raw CSV, caches, virtualenvs)
├── requirements.txt            # Python dependency pins
├── LICENSE                     # MIT License (Copyright (c) 2026 Dhinakaran Anburose)
└── README.md                   # Project overview & documentation
```

---

## Local Development & Installation

### Prerequisites
- Python 3.11, 3.12, or 3.13
- Git

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/your-username/disease-diagnosis-prediction.git
cd disease-diagnosis-prediction
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Automated Test Suite
```bash
pytest -q
```
All **142 automated tests** should pass cleanly without errors.

---

## Docker Deployment Guide

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) or Docker Engine (v20.10+)
- Docker Compose (v2.0+)

### 1. Build Docker Image
```bash
docker build -t disease-diagnosis-api .
```

### 2. Launch API via Docker Compose
```bash
docker compose up -d
```
The application will launch in a non-root container exposing port `8000`.

### 3. Verify Health Endpoint
```bash
curl http://localhost:8000/health
```

### 4. Stop Container Service
```bash
docker compose down
```

---

## REST API Documentation

### Interactive OpenAPI Documentation
Interactive Swagger documentation is served live at `http://localhost:8000/docs` when the container or application is running.

### API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | API metadata and system status |
| `GET` | `/health` | Service health check and model loading verification |
| `POST` | `/predict` | Predict heart disease class and probability for a single observation |
| `POST` | `/predict/batch` | Predict heart disease class and probability for a list of observations |

---

### Input Schema (`PatientData`)

| Feature | Type | Constraints / Valid Values | Description |
| :--- | :--- | :--- | :--- |
| `age` | `integer` | $20 \le \text{age} \le 100$ | Age in years |
| `sex` | `string` | `"Male"`, `"Female"` | Biological sex |
| `cp` | `string` | `"typical angina"`, `"atypical angina"`, `"non-anginal pain"`, `"asymptomatic"` | Chest pain type |
| `trestbps` | `number` | $80 \le \text{trestbps} \le 220$ | Resting blood pressure (mm Hg) |
| `chol` | `number` | $0 \le \text{chol} \le 600$ | Serum cholesterol (mg/dl; $0$ imputed as missing) |
| `fbs` | `boolean` | `true`, `false` | Fasting blood sugar > 120 mg/dl |
| `restecg` | `string` | `"normal"`, `"ST-T wave abnormality"`, `"left ventricular hypertrophy"` | Resting ECG results |
| `thalch` | `number` | $60 \le \text{thalch} \le 220$ | Maximum heart rate achieved |
| `exang` | `boolean` | `true`, `false` | Exercise induced angina |
| `oldpeak` | `number` | $-3.0 \le \text{oldpeak} \le 10.0$ | ST depression induced by exercise relative to rest |

---

### Example Single Prediction Request (`POST /predict`)

#### cURL Request:
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
    "age": 58,
    "sex": "Male",
    "cp": "asymptomatic",
    "trestbps": 140,
    "chol": 240,
    "fbs": false,
    "restecg": "normal",
    "thalch": 130,
    "exang": true,
    "oldpeak": 2.5
  }'
```

#### JSON Response (`PredictionResponse`):
```json
{
  "predicted_class": 1,
  "predicted_probability": 0.8842,
  "model_name": "Tuned Support Vector Machine"
}
```
*Interpretation: Class `1` indicates positive prediction for Heart Disease.*

---

## Machine Learning Methodology Highlights

1. **Dataset Integrity**: 918 observations from the Cleveland / UCI dataset with binary target transformation ($\text{target} = (\text{num} > 0)$).
2. **Leakage Isolation**: Split into $N_{\text{train}}=734$ and $N_{\text{test}}=184$ using stratified splitting (`random_state=42`). Preprocessing parameters (StandardScaler mean/std, OneHotEncoder categories, median imputers) were fitted **strictly on `X_train`**.
3. **Model Selection**: Evaluated 5 baseline models. Hyperparameter optimization via 5-fold Stratified `GridSearchCV` selected Linear SVM ($C=100$) over Logistic Regression, KNN, Decision Trees, and Random Forests.
4. **Model Persistence**: Serialized fitted preprocessor and classifier pipeline into `models/final_model.joblib` with matching SHA-256 metadata verification.

---

## Methodological Limitations

1. **Dataset Size**: Evaluated on $N=918$ observations; generalization to broader clinical populations requires larger prospective datasets.
2. **Missing Value Treatment**: Zero-cholesterol values ($N=172$) were treated as unrecorded and imputed via training median; prospective clinical data should minimize unrecorded fields.
3. **Exploratory Thresholding**: Model decision boundary uses default probability threshold ($0.50$). Optimal clinical thresholds must be selected based on clinical cost-benefit requirements.
4. **Non-Causal Associations**: Feature coefficients and importances reflect statistical associations within the fitted model and do not establish direct medical causality.

---

## License & Source Provenance

- **Original Project Code & Materials**: Released under the **[MIT License](file:///c:/projects/Disease-Diagnosis-Prediction/LICENSE)** (Copyright (c) 2026 Dhinakaran Anburose).
- **Third-Party Clinical Dataset**: The raw clinical dataset originates from the Cleveland / UCI Heart Disease dataset. It is **intentionally excluded from this repository** because dataset redistribution rights were not independently established for the local CSV format. Users acquiring the raw dataset directly from the UCI Machine Learning Repository must comply with its applicable source terms. The MIT License applies exclusively to original project source code, microservice implementation, Docker configuration, and original technical documentation.
