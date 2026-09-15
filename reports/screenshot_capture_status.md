# Phase 16.2 — Real Screenshot Capture & Visual Presentation Report

## 1. Master Screenshot Inventory & Capture Status

| Filename | Source Type | Description / View | Capture Status | Intended README / Portfolio Location | Manual Capture Instruction | Security / PII Audit |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `01_target_distribution.png` | Static Figure | Class balance plot (410 negative, 508 positive) | **Copied & Verified** (`reports/screenshots/01_target_distribution.png`) | Section 2 / Portfolio EDA | N/A (Static asset) | Passed (No PII) |
| `02_correlation_heatmap.png` | Static Figure | Pairwise correlation matrix across numerical features | **Copied & Verified** (`reports/screenshots/02_correlation_heatmap.png`) | Section 2 / README Figure 3 | N/A (Static asset) | Passed (No PII) |
| `03_model_comparison.png` | Static Figure | Test set metric comparison across 5 baseline models | **Copied & Verified** (`reports/screenshots/03_model_comparison.png`) | Section 2 / Portfolio Model Selection | N/A (Static asset) | Passed (No PII) |
| `04_roc_curves.png` | Static Figure | ROC Curves comparison (Tuned SVM AUC = 0.9295) | **Copied & Verified** (`reports/screenshots/04_roc_curves.png`) | Section 2 / README Figure 1 | N/A (Static asset) | Passed (No PII) |
| `05_confusion_matrix.png` | Static Figure | Confusion matrix heatmaps ($\text{TN}=67, \text{FP}=15, \text{FN}=13, \text{TP}=89$) | **Copied & Verified** (`reports/screenshots/05_confusion_matrix.png`) | Section 2 / README Figure 2 | N/A (Static asset) | Passed (No PII) |
| `06_system_architecture.png` | Static Figure | System Architecture flow & Docker container boundary | **Copied & Verified** (`reports/screenshots/06_system_architecture.png`) | Section 3 / README Figure 4 | N/A (Static asset) | Passed (No PII) |
| `07_fastapi_swagger.png` | Live Application | Interactive OpenAPI documentation (`/docs`) showing GET/POST endpoints | **Live Response Verified** (Manual capture recommended) | Section 6 / Portfolio API | Launch `uvicorn app.main:app --port 8000`, open `http://localhost:8000/docs` in browser, capture screen | Passed (No PII/Secrets) |
| `08_api_health.png` | Live Application | `GET /health` response (`status: "healthy"`, `model_loaded: true`) | **Live Response Verified** (Empirical JSON verified) | Section 6 / README API Health | Run `curl http://localhost:8000/health` or open in browser | Passed (No PII/Secrets) |
| `09_api_prediction.png` | Live Application | `POST /predict` single observation response (`predicted_class: 1`, `probability: 0.9145`) | **Live Response Verified** (Empirical JSON verified) | Section 6 / Portfolio Predict | Execute `POST /predict` in Swagger UI or cURL, capture response window | Passed (No PII/Secrets) |
| `10_api_batch_prediction.png` | Live Application | `POST /predict/batch` response with multiple synthetic records | **Live Response Verified** (Empirical JSON verified) | Optional / Portfolio Appendix | Execute `POST /predict/batch` in Swagger UI or cURL | Passed (No PII/Secrets) |
| `11_docker_running.png` | Live Container | Terminal execution showing `docker compose up -d` & container logs | **Command Verified** (Manual capture recommended) | Section 5 / Portfolio Deployment | Run `docker compose up -d` in terminal, capture terminal window | Passed (No PII/Secrets) |

---

## 2. Launch Commands & API Endpoint Instructions

### Local API Server Launch Command
```bash
python -m uvicorn app.main:app --reload --port 8000
```
*Alternative (with PYTHONPATH):*
```bash
$env:PYTHONPATH="."; python -m uvicorn app.main:app --port 8000
```

### Docker Container Launch Command
```bash
docker build -t disease-diagnosis-api .
docker compose up -d
```

### Exact Target Endpoint URLs
- **OpenAPI Interactive Documentation**: `http://localhost:8000/docs`
- **Root Info Endpoint**: `http://localhost:8000/`
- **Health Check Endpoint**: `http://localhost:8000/health`
- **Single Prediction Endpoint**: `http://localhost:8000/predict`
- **Batch Prediction Endpoint**: `http://localhost:8000/predict/batch`

---

## 3. Empirical Live API Verification Results

### 1. `GET /health` Verified Output
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 2. `POST /predict` Verified Output
**Input Synthetic Payload:**
```json
{
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
}
```
**Empirical Output:**
```json
{
  "predicted_class": 1,
  "predicted_probability": 0.9144626118701925,
  "model_name": "Tuned Support Vector Machine"
}
```

### 3. `POST /predict/batch` Verified Output
**Input Synthetic Payload:**
```json
{
  "records": [
    {
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
    },
    {
      "age": 42,
      "sex": "Female",
      "cp": "typical angina",
      "trestbps": 120,
      "chol": 190,
      "fbs": false,
      "restecg": "normal",
      "thalch": 165,
      "exang": false,
      "oldpeak": 0.0
    }
  ]
}
```
**Empirical Output:**
```json
{
  "predictions": [
    {
      "predicted_class": 1,
      "predicted_probability": 0.9144626118701925,
      "model_name": "Tuned Support Vector Machine"
    },
    {
      "predicted_class": 0,
      "predicted_probability": 0.03190398896181622,
      "model_name": "Tuned Support Vector Machine"
    }
  ]
}
```

---

## 4. Prioritized README Visual Recommendations (6 Core Visuals)

For maximum recruiter impact without cluttering the README:
1. **System Architecture**: `reports/figures/system_architecture.png` (Visual 4 in README)
2. **ROC Curves Comparison**: `reports/figures/tuned_roc_curves.png` (Visual 1 in README)
3. **Confusion Matrices Heatmap**: `reports/figures/tuned_confusion_matrices.png` (Visual 2 in README)
4. **Correlation Heatmap**: `reports/figures/correlation_heatmap.png` (Visual 3 in README)
5. **FastAPI Swagger UI**: Browser screenshot of `http://localhost:8000/docs` (`07_fastapi_swagger.png`)
6. **API Prediction Response**: Real cURL / Postman / Swagger response (`09_api_prediction.png`)

---

## 5. Security & Privacy Audit
- **Status**: **PASS (Clean)**.
- All live API tests used synthetic clinical example values. Zero real patient PII, personal file paths, tokens, or credentials were included or generated.
