# Phase 16.2.1 — Docker Dependency Version Consistency Fix Report

## 1. Root Cause
In `requirements.txt`, the scikit-learn dependency was specified with a floating lower bound: `scikit-learn>=1.6.0`. During Docker image build (`docker build`), `pip` resolved and installed the latest available version (`1.9.1`). Because the persisted model (`models/final_model.joblib`) was serialized with scikit-learn `1.6.1`, the version mismatch triggered `InconsistentVersionWarning` upon loading inside the container and failed the model loading validation, causing the `/health` endpoint to report `{"status": "unhealthy", "model_loaded": false}`.

## 2. Previous Docker scikit-learn Version
`1.9.1`

## 3. Correct Model scikit-learn Version
`1.6.1` (as recorded in `models/model_metadata.json`)

## 4. Dependency Change
Modified `requirements.txt` to strictly pin scikit-learn version:
```diff
-scikit-learn>=1.6.0
+scikit-learn==1.6.1
```

## 5. Docker Rebuild Result
The Docker container image `disease-diagnosis-api:latest` was rebuilt cleanly without cache using:
`docker build --no-cache -t disease-diagnosis-api .`
The container was started via:
`docker compose up -d`
Status: **Container `disease-diagnosis-api-container` started successfully and is running.**

## 6. Installed Docker scikit-learn Version
Verified inside the running container via `docker exec disease-diagnosis-api-container python -c "import sklearn; print(sklearn.__version__)"`:
**`1.6.1`**

## 7. Health Response
Endpoint `GET http://localhost:8000/health`:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## 8. Prediction Response
Endpoint `POST http://localhost:8000/predict` with synthetic input (`age`: 55, `chol`: 250, `cp`: `"asymptomatic"`, `exang`: false, `fbs`: false, `oldpeak`: 1.2, `restecg`: `"normal"`, `sex`: `"Male"`, `thalch`: 150, `trestbps`: 140):
```json
{
  "predicted_class": 1,
  "predicted_probability": 0.6461265923693491,
  "model_name": "Tuned Support Vector Machine"
}
```

## 9. Model Integrity Hashes
- **`models/final_model.joblib` SHA-256:** `16081825890dc9a3fae9c8fd2216272379ee7ac0f7712072f983a0bb897a1568` (Verified Unchanged)
- **`models/model_metadata.json` SHA-256:** `5ec6e595a0c7d5d0dc899611b1f3f3480c1792564c2113e6e39478329467c836` (Verified Unchanged)

## 10. Dataset Integrity
- **`data/raw/heart_disease.csv` MD5:** `13c9cfee54ce2b1552ef7d787a5d8be9` (Verified Unchanged)
- **Raw CSV Data Rows:** 918 rows (+ 1 header line = 919 lines) (Verified Unchanged)

## 11. pytest Result
Ran `pytest -q`:
**`143 passed, 1 warning in 93.71s`**

## 12. Docker Test Result
Ran `pytest tests/test_docker.py -v`:
**`5 passed in 0.12s`**

## 13. Git Changes
`git status --short`:
```
 M requirements.txt
?? reports/docker_dependency_fix_report.md
?? reports/screenshot_capture_status.md
?? reports/screenshots/
```

`git diff -- requirements.txt Dockerfile docker-compose.yml`:
```diff
diff --git a/requirements.txt b/requirements.txt
index 2b6b41e..83f7895 100644
--- a/requirements.txt
+++ b/requirements.txt
@@ -1,6 +1,6 @@
 pandas>=2.2.0
 numpy>=2.0.0
-scikit-learn>=1.6.0
+scikit-learn==1.6.1
 joblib>=1.4.0
 pytest>=9.0.0
 matplotlib>=3.10.0
```

## 14. Resume Screenshot Capture Status
**YES.** The Docker-served FastAPI application is running healthily on `http://localhost:8000` with the correct model loaded. Phase 16.2 live screenshot capture can resume.
