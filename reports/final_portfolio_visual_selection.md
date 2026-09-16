# Final Portfolio Visual Selection & Curation Report

This report defines the curated visual hierarchy, asset placement guidelines, and prioritized presentation sequences for presenting the **Disease Diagnosis Prediction** project across public GitHub, technical portfolio websites, and detailed engineering case studies.

---

## 1. Master Visual Inventory & Classification

### Primary Visuals

#### 1. Interactive Web Dashboard & Single Prediction
- **Exact Filename / Resource**: `reports/screenshots/09_api_prediction.png` / `frontend/index.html` (`http://localhost:8000/ui`)
- **Purpose**: Demonstrates the interactive browser UI for single patient risk classification, clinical input forms, live connection status, predicted probability bar, and model metadata.
- **What a Recruiter Verifies**: Full product integration, clean frontend/backend integration, input validation, and user-centric design without relying on third-party UI frameworks.
- **Recommended Placement**: GitHub README (Hero section) / Portfolio website / Technical case study
- **Priority**: **Primary**

#### 2. System Architecture & Processing Boundary
- **Exact Filename / Resource**: `reports/figures/system_architecture.png` (also `reports/screenshots/06_system_architecture.png`)
- **Purpose**: Illustrates the end-to-end data processing pipeline, HTTP request routing, Pydantic schema validation, inference pipeline, preprocessor, and Docker container isolation.
- **What a Recruiter Verifies**: Production software architecture, clean separation of concerns, and Docker containerization and non-root execution.
- **Recommended Placement**: GitHub README / Portfolio website / Technical case study
- **Priority**: **Primary**

#### 3. Tuned Model ROC Curves Discrimination
- **Exact Filename / Resource**: `reports/figures/tuned_roc_curves.png` (also `reports/screenshots/04_roc_curves.png`)
- **Purpose**: Evaluates receiver operating characteristic (ROC) discrimination curves for tuned candidate models on the held-out test set ($N=184$).
- **What a Recruiter Verifies**: Rigorous machine learning evaluation, candidate benchmarking, and test-set discrimination performance (Linear SVM ROC-AUC = `0.9295`).
- **Recommended Placement**: GitHub README / Portfolio website / Technical case study
- **Priority**: **Primary**

#### 4. Final Model Confusion Matrix Heatmap
- **Exact Filename / Resource**: `reports/figures/tuned_confusion_matrices.png` (also `reports/screenshots/05_confusion_matrix.png`)
- **Purpose**: Visualizes classification performance breakdown across True Negatives ($67$), False Positives ($15$), False Negatives ($13$), and True Positives ($89$).
- **What a Recruiter Verifies**: Quantitative evaluation on isolated held-out test set ($N=184$), sensitivity ($87.25\%$), and specificity ($81.71\%$).
- **Recommended Placement**: GitHub README / Portfolio website / Technical case study
- **Priority**: **Primary**

#### 5. FastAPI Interactive Swagger / OpenAPI Documentation
- **Exact Filename / Resource**: `reports/screenshots/07_fastapi_swagger.png` (`http://localhost:8000/docs`)
- **Purpose**: Displays interactive OpenAPI documentation for all REST API endpoints (`/`, `/health`, `/predict`, `/predict/batch`).
- **What a Recruiter Verifies**: RESTful API design standards, self-documenting endpoints, and Pydantic schema validation compliance.
- **Recommended Placement**: Portfolio website / Technical case study
- **Priority**: **Primary**

---

### Secondary Visuals

#### 6. Batch Prediction Workspace
- **Exact Filename / Resource**: `reports/screenshots/10_api_batch_prediction.png`
- **Purpose**: Demonstrates bulk JSON array prediction processing (`POST /predict/batch`) and tabular batch output format.
- **What a Recruiter Verifies**: Batch inference capability and structured batch prediction output.
- **Recommended Placement**: Portfolio website / Technical case study
- **Priority**: **Secondary**

#### 7. Docker Container Runtime Execution
- **Exact Filename / Resource**: `reports/screenshots/11_docker_running_health.png`
- **Purpose**: Proves non-root container orchestration, Uvicorn execution, and container startup logs.
- **What a Recruiter Verifies**: Practical DevOps containerization skills, Docker Compose configuration, and non-root security compliance (`appuser`).
- **Recommended Placement**: Portfolio website / Technical case study
- **Priority**: **Secondary**

#### 8. API Health Check Endpoint
- **Exact Filename / Resource**: `reports/screenshots/08_api_health.png` (`http://localhost:8000/health`)
- **Purpose**: Shows microservice health monitoring output (`status: "healthy"`, `model_loaded: true`).
- **What a Recruiter Verifies**: Microservice production readiness and automated model loading verification.
- **Recommended Placement**: Technical case study / Resume support
- **Priority**: **Secondary**

---

### Supporting Analytics

#### 9. Feature Correlation Matrix
- **Exact Filename / Resource**: `reports/figures/correlation_heatmap.png` (also `reports/screenshots/02_correlation_heatmap.png`)
- **Purpose**: Shows pairwise correlation analysis across numerical predictors.
- **What a Recruiter Verifies**: Multicollinearity verification ($|r| < 0.40$) and thorough exploratory data analysis.
- **Recommended Placement**: GitHub README / Technical case study
- **Priority**: **Supporting**

#### 10. Dataset Target Distribution
- **Exact Filename / Resource**: `reports/figures/target_distribution.png` (also `reports/screenshots/01_target_distribution.png`)
- **Purpose**: Displays clinical dataset class balance (410 negative, 508 positive cases out of $N=918$).
- **What a Recruiter Verifies**: Data understanding, stratification rationale, and class imbalance assessment.
- **Recommended Placement**: Technical case study
- **Priority**: **Supporting**

#### 11. Baseline Model Performance Comparison
- **Exact Filename / Resource**: `reports/figures/baseline_model_comparison.png` (also `reports/screenshots/03_model_comparison.png`)
- **Purpose**: Compares baseline test-set metrics across 5 classification algorithms prior to hyperparameter optimization.
- **What a Recruiter Verifies**: Algorithm selection methodology, baseline benchmarking rigor, and justification for model selection.
- **Recommended Placement**: Technical case study
- **Priority**: **Supporting**

---

## 2. Recommended Visual Presentation Sequences

### Sequence A: GitHub README (5 High-Impact Visuals)
1. **Interactive Web Dashboard**: `frontend/index.html` (`http://localhost:8000/ui`)
2. **System Architecture Diagram**: `reports/figures/system_architecture.png`
3. **Tuned ROC Curves Discrimination**: `reports/figures/tuned_roc_curves.png`
4. **Tuned Confusion Matrix Heatmap**: `reports/figures/tuned_confusion_matrices.png`
5. **Feature Correlation Heatmap**: `reports/figures/correlation_heatmap.png`

### Sequence B: Portfolio Website Case Study (7 Visuals)
1. **Interactive Web Dashboard**: `frontend/index.html` (`http://localhost:8000/ui`)
2. **System Architecture Diagram**: `reports/figures/system_architecture.png`
3. **Tuned ROC Curves**: `reports/figures/tuned_roc_curves.png`
4. **Tuned Confusion Matrix**: `reports/figures/tuned_confusion_matrices.png`
5. **FastAPI Interactive Swagger**: `reports/screenshots/07_fastapi_swagger.png`
6. **Batch Prediction Workspace**: `reports/screenshots/10_api_batch_prediction.png`
7. **Docker Container Runtime**: `reports/screenshots/11_docker_running_health.png`

### Sequence C: Comprehensive Technical Case Study (9 Visuals)
1. **Target Distribution**: `reports/figures/target_distribution.png`
2. **Correlation Heatmap**: `reports/figures/correlation_heatmap.png`
3. **Baseline Model Comparison**: `reports/figures/baseline_model_comparison.png`
4. **Tuned ROC Curves**: `reports/figures/tuned_roc_curves.png`
5. **Tuned Confusion Matrix**: `reports/figures/tuned_confusion_matrices.png`
6. **System Architecture**: `reports/figures/system_architecture.png`
7. **FastAPI Swagger Docs**: `reports/screenshots/07_fastapi_swagger.png`
8. **Batch Prediction Workspace**: `reports/screenshots/10_api_batch_prediction.png`
9. **Docker Container Runtime**: `reports/screenshots/11_docker_running_health.png`

---

## 3. Do Not Use (Visual Exclusion List)

The following assets are **excluded from public portfolio presentation** to prevent redundancy, visual clutter, or unhelpful technical detail:

1. **Individual Single-Model Baseline Confusion Matrices**:
   - `confusion_matrix_knn.png`
   - `confusion_matrix_logistic_regression.png`
   - `confusion_matrix_decision_tree.png`
   - `confusion_matrix_random_forest.png`
   - `confusion_matrix_svm.png`
   - *Reason*: Redundant with the consolidated final matrix `tuned_confusion_matrices.png`.
2. **Raw Terminal Log Excerpts**:
   - Non-stylized terminal error traces or unformatted pip installation logs.
   - *Reason*: Provides low visual value and distracts from software engineering highlights.
3. **Redundant Intermediate Plots**:
   - Raw distribution plots duplicated across intermediate notebook phases.
   - *Reason*: Kept in exploratory notebooks (`notebooks/`); omitted from primary portfolio documentation for conciseness.

---

## 4. Methodological Framing & Non-Clinical Integrity

To maintain strict scientific and technical accuracy across all portfolio presentations:

- **Approved Terminology**: Use `predicted class`, `predicted probability`, `held-out test set ($N=184$)`, `model performance`, `FastAPI inference service`, `interactive web dashboard`.
- **Prohibited Claims**: Do NOT claim that screenshots or model predictions prove clinical validity, diagnostic efficacy, calibration perfection, or real-world patient safety.
- **Required Framing**: Always present the system as an **engineering research prototype** demonstrating end-to-end software architecture and machine learning best practices.
