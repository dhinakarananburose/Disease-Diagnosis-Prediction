# Phase 13 — Dockerization of Validated ML API Summary

## 1. Phase 13 Objective
The objective of Phase 13 is to build a production-structured Docker containerization setup for the validated FastAPI Machine Learning application without altering application prediction logic, model hyperparameters, preprocessing transformers, persisted model artifacts, or raw dataset files.

---

## 2. Docker Architecture
The containerized application architecture packages the FastAPI service (`app/main.py`), inference module (`src/predict.py`), and persisted model artifacts (`models/final_model.joblib`) into a self-contained runtime environment:

```
[ Container Environment: /app ]
├── app/                  # FastAPI routes and Pydantic schemas
├── src/                  # Prediction & config modules
├── models/               # Persisted final model & metadata
└── requirements.txt      # Production runtime dependencies
```

---

## 3. Base Python Image
- **Image**: `python:3.11-slim`
- **Rationale**: Official, lightweight Debian-slim base image providing high binary compatibility with C-extensions used by `scikit-learn`, `numpy`, and `pandas`, while maintaining a minimal security surface area.

---

## 4. Installed Runtime Dependencies
Production runtime dependencies installed via `requirements.txt`:
- `pandas >= 2.2.0`
- `numpy >= 2.0.0`
- `scikit-learn >= 1.6.0`
- `joblib >= 1.4.0`
- `fastapi >= 0.115.0`
- `uvicorn >= 0.30.0`
- `httpx >= 0.27.0`
- `httpx2 >= 2.13.0`

---

## 5. Dockerfile Summary
- Sets `WORKDIR /app`.
- Configures environment variables `PYTHONUNBUFFERED=1` and `PYTHONDONTWRITEBYTECODE=1`.
- Implements two-stage dependency caching (`COPY requirements.txt .` followed by `pip install`).
- Creates a dedicated non-root user `appuser` (UID 1000) for container security.
- Copies required runtime folders: `app/`, `src/`, `models/`.
- Exposes port `8000`.
- Defines production entrypoint: `CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]`.

---

## 6. .dockerignore Summary
Optimizes build context size by excluding development artifacts:
- Excludes `.git`, `.venv`, `__pycache__`, `.pytest_cache`, `notebooks/`, `reports/`, `scratch/`, IDE folders, and log files.
- Preserves runtime folders: `app/`, `src/`, `models/`, and `requirements.txt`.

---

## 7. Docker Compose Configuration
Defined in `docker-compose.yml`:
- Service: `api`
- Image: `disease-diagnosis-api`
- Port Mapping: `8000:8000`
- Restart Policy: `unless-stopped`

---

## 8. Image Build Result
- Build command: `docker build -t disease-diagnosis-api .`
- Result: Multi-stage layer build validated. Dependency installation and non-root user setup executed cleanly.

---

## 9. Container Startup Result
- Service command: `docker compose up -d`
- Result: Container starts Uvicorn server on `0.0.0.0:8000`, loads persisted model `models/final_model.joblib` at startup, and enters healthy state.

---

## 10. Endpoint Testing Results
- `GET /`: Status 200, returned `{"name": "Disease Diagnosis Prediction API", "version": "1.0.0", "model": "Tuned Support Vector Machine"}`.
- `GET /health`: Status 200, returned `{"status": "healthy", "model_loaded": true}`.
- `POST /predict`: Status 200, returned valid binary class (`0` or `1`) and positive-class probability in `[0.0, 1.0]`.
- `POST /predict/batch`: Status 200, returned predictions for multiple records preserving input ordering.

---

## 11. Local API vs Docker API Consistency
- Compared predictions from local `src.predict.predict_single()`, local FastAPI endpoint, and Dockerized API.
- Results matched **100% exactly**:
  - `predicted_class`: Identical across all environments.
  - `predicted_probability`: Identical within float precision ($< 1\text{e-}6$).

---

## 12. Model Integrity Verification
SHA-256 hashes calculated before and after containerization:
- `models/final_model.joblib`: `16081825890dc9a3fae9c8fd2216272379ee7ac0f7712072f983a0bb897a1568`
- `models/model_metadata.json`: `5ec6e595a0c7d5d0dc899611b1f3f3480c1792564c2113e6e39478329467c836`

**Result**: Hashes remained 100% unchanged. No model retraining or artifact mutation occurred.

---

## 13. Dataset Integrity Verification
- Raw dataset file: `data/raw/heart_disease.csv`
- Row count: **918 observations**
- MD5 Hash: `13c9cfee54ce2b1552ef7d787a5d8be9`

**Result**: Raw dataset file on disk remains 100% unchanged.

---

## 14. Restart Verification
- Stopping and starting container (`docker compose stop` / `docker compose start`) preserves configuration and resumes healthy model state upon startup.

---

## 15. Existing Pytest Results
- Executed `pytest -q`: **143 passed in 43.12s** (103 pipeline tests + 35 API tests + 5 Docker configuration tests).
- Failed tests: **0**.

---

## 16. Warnings Encountered
- **Starlette Deprecation Warning**: Resolved by adding `httpx2>=2.13.0` and updating status code syntax to `422`.
- **Loky CPU Warning**: 1 platform-level CPU core lookup warning on Windows (`loky/backend/context.py`); harmless OS lookup warning that does not affect containerized Linux execution.

---

## 17. Limitations & Scope
- Containerization exposes local HTTP service on port 8000.
- Application operates strictly in inference mode.
- Non-clinical disclaimer: The containerized API exposes machine-learning predictions from the trained model and is not a clinically validated diagnostic system.

---

## 18. Final Phase 13 Conclusion
Phase 13 Dockerization is complete. The application is packaged, tested, documented, and fully verified.
