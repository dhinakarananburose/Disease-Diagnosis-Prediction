# Phase 11 — FastAPI Application Integration Summary

## 1. Executive Summary
Phase 11 integrates a production-structured REST API built with FastAPI and Pydantic (`app/main.py`, `app/schemas.py`). The API exposes the persisted machine-learning model (`models/final_model.joblib`) using the inference module implemented in `src/predict.py`.

> **Disclaimer**: The API exposes machine-learning predictions from the trained model. It is not a clinically validated diagnostic system.

---

## 2. API Architecture & Structure
The application structure is organized as follows:

```
app/
├── __init__.py
├── main.py          # FastAPI application, lifespan model loading, and HTTP route handlers
└── schemas.py       # Pydantic request and response models
```

### Key Architectural Principles:
1. **Separation of Concerns**: Endpoint routing and HTTP payload validation are handled in `app/`, while model loading, data cleaning (`chol=0` conversion to `NaN`), feature ordering, and model execution are encapsulated in `src/predict.py`.
2. **Startup Model Loading**: The model artifact is loaded once during application startup via FastAPI lifespan context (`@asynccontextmanager`) and stored in application state (`app.state.model`), preventing redundant disk I/O on every request.
3. **Zero Retraining**: The API operates strictly in inference mode using `models/final_model.joblib`. No model retraining, hyperparameter tuning, or dataset modification occurs.

---

## 3. API Endpoints

### 1. Root Information Endpoint (`GET /`)
- **Summary**: API metadata.
- **Response Format**:
```json
{
    "name": "Disease Diagnosis Prediction API",
    "version": "1.0.0",
    "model": "Tuned Support Vector Machine"
}
```

### 2. Health Check Endpoint (`GET /health`)
- **Summary**: Service health and model artifact loading verification.
- **Response Format**:
```json
{
    "status": "healthy",
    "model_loaded": true
}
```

### 3. Single Prediction Endpoint (`POST /predict`)
- **Summary**: Accepts 10 raw feature values for a single observation and returns class prediction and positive-class probability.
- **Request Body (JSON)**:
```json
{
    "age": 55,
    "sex": "Male",
    "cp": "asymptomatic",
    "trestbps": 140,
    "chol": 250,
    "fbs": false,
    "restecg": "normal",
    "thalch": 150,
    "exang": false,
    "oldpeak": 1.2
}
```
- **Response Body (JSON)**:
```json
{
    "predicted_class": 1,
    "predicted_probability": 0.8245,
    "model_name": "Tuned Support Vector Machine"
}
```

### 4. Batch Prediction Endpoint (`POST /predict/batch`)
- **Summary**: Accepts a list of prediction feature records and returns predictions for each record.
- **Request Body (JSON)**:
```json
{
    "records": [
        {
            "age": 55,
            "sex": "Male",
            "cp": "asymptomatic",
            "trestbps": 140,
            "chol": 250,
            "fbs": false,
            "restecg": "normal",
            "thalch": 150,
            "exang": false,
            "oldpeak": 1.2
        }
    ]
}
```
- **Response Body (JSON)**:
```json
{
    "predictions": [
        {
            "predicted_class": 1,
            "predicted_probability": 0.8245,
            "model_name": "Tuned Support Vector Machine"
        }
    ]
}
```

---

## 4. Input Validation & Error Handling

- **Pydantic Validation**: `PredictionInput` schema enforces required fields and valid data types.
- **Pipeline Preprocessing Integration**: Input dictionaries are passed to `src.predict.predict_single()`, which converts `chol == 0` to `np.nan` for pipeline median imputation and enforces column ordering matching `MODEL_FEATURES`.
- **Exception Handlers**: Custom FastAPI exception handlers capture `ValueError` and `TypeError` exceptions from validation/prediction modules and return structured `422 Unprocessable Entity` JSON responses.
- **Zero Stack Trace Exposure**: Stack traces are caught internally and never exposed to API consumers.

---

## 5. Testing & Verification Summary

A comprehensive test suite was created in `tests/test_api.py` using FastAPI's `TestClient`:
1. `GET /`: Validated status 200 and root response schema.
2. `GET /health`: Validated status 200, `status='healthy'`, and `model_loaded=True`.
3. `POST /predict` (valid): Validated status 200, binary class (`{0, 1}`), positive-class probability in `[0.0, 1.0]`, and model name.
4. `POST /predict` (invalid): Validated HTTP 422 error code and clean JSON error response for missing features, invalid numeric types, and invalid categorical structures.
5. **Exact Match Verification**: Confirmed that `POST /predict` outputs match `src.predict.predict_single()` **100% exactly**.
6. **Zero Retraining Verification**: Verified that the API loads `models/final_model.joblib` cleanly without altering artifacts.
7. **Regression Integrity**: All 103 existing tests in `tests/test_pipeline.py` remain 100% passing.

---

## 6. Limitations & Scope

1. **Schema Requirement**: The API expects the 10 raw features of the heart disease dataset (`age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalch`, `exang`, `oldpeak`). Extra columns like `ca`, `thal`, or `slope` are not part of the model schema.
2. **Interactive OpenAPI Docs**: Standard OpenAPI documentation is automatically available at `/docs` and `/openapi.json`.
3. **Non-Clinical System Disclaimer**: The API exposes machine-learning predictions from the trained model. It is not a clinically validated diagnostic system.
