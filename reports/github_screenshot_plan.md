# GitHub & Portfolio Visual Screenshot Capture Plan

To showcase the application visually on GitHub and portfolio platforms, capture the following 9 recommended screenshots manually.

---

### Screenshot 1: FastAPI Swagger UI (`/docs`)
- **Recommended Filename**: `docs_swagger_ui.png`
- **Target URL / View**: `http://localhost:8000/docs`
- **What to Capture**: OpenAPI interactive documentation showing `GET /`, `GET /health`, `POST /predict`, and `POST /predict/batch` endpoints expanded.
- **Purpose**: Demonstrates RESTful API design, interactive documentation, and OpenAPI standard compliance.

---

### Screenshot 2: API Health Check (`GET /health`)
- **Recommended Filename**: `api_health_check.png`
- **Target URL / View**: `http://localhost:8000/health` (via browser or Postman/cURL)
- **What to Capture**: JSON response showing `status: "healthy"`, model load status, and timestamp.
- **Purpose**: Proves microservice health monitoring and successful model initialization.

---

### Screenshot 3: Single Prediction Response (`POST /predict`)
- **Recommended Filename**: `api_single_predict.png`
- **Target URL / View**: Swagger UI or Postman showing `POST /predict` request & response
- **What to Capture**: JSON payload with 10 clinical inputs and returned `predicted_class`, `predicted_probability`, and `model_name`.
- **Purpose**: Demonstrates working prediction endpoint and probability output.

---

### Screenshot 4: Docker Container Execution
- **Recommended Filename**: `docker_container_running.png`
- **Target URL / View**: Terminal executing `docker compose up` or Docker Desktop UI
- **What to Capture**: Active Docker container log output showing Uvicorn running on `0.0.0.0:8000`.
- **Purpose**: Proves functional containerization and Uvicorn orchestration.

---

### Screenshot 5: Feature-Target Relationships & EDA
- **Recommended Filename**: `eda_feature_relationships.png`
- **Target File**: `reports/figures/feature_target_relationships.png`
- **What to Capture**: Rendered EDA visualization comparing age, chest pain type, and max heart rate against disease status.
- **Purpose**: Highlights clinical data exploration and statistical analysis skills.

---

### Screenshot 6: Correlation Heatmap
- **Recommended Filename**: `eda_correlation_heatmap.png`
- **Target File**: `reports/figures/correlation_heatmap.png`
- **What to Capture**: Pairwise correlation heatmap of numerical predictors.
- **Purpose**: Shows multicollinearity check and feature understanding.

---

### Screenshot 7: Model Comparison ROC Curves
- **Recommended Filename**: `model_comparison_roc.png`
- **Target File**: `reports/figures/tuned_roc_curves.png` or `reports/figures/roc_curves_baseline.png`
- **What to Capture**: ROC curves comparing baseline and tuned classification models.
- **Purpose**: Demonstrates rigorous model evaluation and discrimination analysis.

---

### Screenshot 8: Final Model Confusion Matrix
- **Recommended Filename**: `final_svm_confusion_matrix.png`
- **Target File**: `reports/figures/tuned_confusion_matrices.png`
- **What to Capture**: Confusion matrix heatmap for the tuned Linear SVM ($N=184$).
- **Purpose**: Visualizes false positives, false negatives, sensitivity, and specificity.

---

### Screenshot 9: Automated Test Suite Execution (`pytest`)
- **Recommended Filename**: `pytest_execution_pass.png`
- **Target URL / View**: Terminal executing `pytest -q`
- **What to Capture**: Clean green test output showing `142 passed` in terminal.
- **Purpose**: Validates software quality, pipeline test coverage, and API reliability.
