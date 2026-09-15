# Project Configuration Summary

## 1. Technical Stack & Environment
- **Operating System**: Windows / Linux (Dockerized)
- **Python Version**: Python 3.11 / 3.12 / 3.13
- **Core ML & Data Libraries**:
  - `pandas` (>= 2.2.0)
  - `numpy` (>= 2.0.0)
  - `scikit-learn` (>= 1.6.0)
  - `joblib` (>= 1.4.0)
  - `matplotlib` (>= 3.10.0)
  - `seaborn` (>= 0.13.0)
- **Web API Libraries**:
  - `fastapi` (>= 0.115.0)
  - `uvicorn` (>= 0.30.0)
  - `pydantic` (>= 2.9.0)
- **Testing & HTTP Client Libraries**:
  - `pytest` (>= 9.0.0)
  - `httpx` (>= 0.27.0)
  - `httpx2` (>= 2.13.0)

---

## 2. Final Persisted Model Specifications
- **Selected Algorithm**: Tuned Support Vector Machine (`SVC`)
- **Persisted Pipeline Artifact**: `models/final_model.joblib` (11.7 KB)
- **Metadata File**: `models/model_metadata.json` (1.1 KB)
- **Hyperparameters**:
  - `C`: `100`
  - `kernel`: `"linear"`
  - `gamma`: `"scale"`
  - `probability`: `True`
  - `random_state`: `42`
- **Preprocessing Included**:
  - Numerical: `SimpleImputer(strategy='median', add_indicator=True)` + `StandardScaler()`
  - Categorical: `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`
  - Boolean: `FunctionTransformer(_cast_boolean_to_int)` + `SimpleImputer(strategy='most_frequent')`

---

## 3. Web API Architecture
- **Framework**: FastAPI
- **Entrypoint**: `app/main.py`
- **Schemas**: `app/schemas.py`
- **Endpoints**:
  - `GET /` — API metadata info
  - `GET /health` — Service health and model artifact load status
  - `POST /predict` — Single observation prediction endpoint
  - `POST /predict/batch` — Batch prediction endpoint
- **Model Loading**: Loaded once during application startup via FastAPI lifespan context manager (`@asynccontextmanager`) and attached to `app.state.model`.

---

## 4. Containerization Architecture
- **Container Technology**: Docker & Docker Compose
- **Base Image**: `python:3.11-slim`
- **Configuration Files**:
  - `Dockerfile` — Slim base image, non-root user `appuser` (UID 1000), uvicorn production entrypoint on `0.0.0.0:8000`.
  - `.dockerignore` — Excludes VCS, virtual environments, development notebooks, and reports while preserving `app/`, `src/`, `models/`, and `requirements.txt`.
  - `docker-compose.yml` — Service `api` mapping port `8000:8000`.

---

## 5. Automated Test Suite Architecture
- **Framework**: `pytest`
- **Test Modules**:
  - `tests/test_pipeline.py` — 100 unit & pipeline tests (data loading, preprocessing, baseline models, tuning, evaluation, model selection, persistence, predict module).
  - `tests/test_api.py` — 37 API integration & validation tests (endpoints, validation, boundary edge cases, batch, consistency, repeated requests, artifact SHA-256 integrity, raw dataset MD5 integrity).
  - `tests/test_docker.py` — 5 Docker configuration & path resolution tests.
- **Total Test Count**: 142 tests (100% passing, 0 failures).

---

## 6. Directory Structure
```
Disease-Diagnosis-Prediction/
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
├── data/
│   ├── raw/
│   │   └── heart_disease.csv
│   └── processed/
├── models/
│   ├── final_model.joblib
│   └── model_metadata.json
├── notebooks/
│   ├── 01_data_loading_and_inspection.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_data_preprocessing.ipynb
│   ├── 04_baseline_classification_models.ipynb
│   ├── 05_model_evaluation.ipynb
│   ├── 06_hyperparameter_optimization.ipynb
│   ├── 07_advanced_evaluation.ipynb
│   ├── 08_final_model_selection.ipynb
│   ├── 09_model_persistence.ipynb
│   └── 10_prediction_pipeline.ipynb
├── reports/
│   ├── figures/
│   ├── advanced_evaluation_summary.csv
│   ├── api_integration_summary.md
│   ├── api_test_results.csv
│   ├── application_testing_summary.md
│   ├── confusion_matrix_summary.csv
│   ├── data_quality_report.csv
│   ├── docker_test_results.csv
│   ├── dockerization_summary.md
│   ├── feature_importance_comparison.csv
│   ├── final_metrics_table.csv
│   ├── final_model_selection.csv
│   ├── final_model_selection_summary.md
│   ├── final_project_report.md
│   ├── hyperparameter_search_summary.md
│   ├── logistic_regression_coefficients.csv
│   ├── model_generalization_comparison.csv
│   ├── model_persistence_summary.md
│   ├── model_results.csv
│   ├── model_selection_summary.md
│   ├── prediction_pipeline_summary.md
│   ├── project_configuration_summary.md
│   ├── random_forest_feature_importance.csv
│   ├── threshold_analysis.csv
│   ├── tuned_confusion_matrix_summary.csv
│   └── tuned_model_results.csv
├── src/
│   ├── __init__.py
│   ├── advanced_evaluation.py
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── model_analysis.py
│   ├── model_persistence.py
│   ├── model_selection.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── train.py
├── tests/
│   ├── test_api.py
│   ├── test_docker.py
│   └── test_pipeline.py
├── .dockerignore
├── docker-compose.yml
├── Dockerfile
├── README.md
└── requirements.txt
```
