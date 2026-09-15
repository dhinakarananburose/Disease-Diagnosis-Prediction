# Resume Project Bullet Points

### Option A — Full Machine Learning & Software Engineering Focus
**Disease Diagnosis Prediction ML API** | *Python, scikit-learn, FastAPI, Docker, pytest*
- Engineered an end-to-end clinical diagnosis prediction pipeline ($N=918$) with leak-free preprocessing, benchmarked 5 algorithms, and tuned a Linear SVM classifier achieving **84.78% test accuracy** and **0.9295 ROC-AUC**.
- Built a RESTful FastAPI service (`POST /predict`, `POST /predict/batch`) featuring Pydantic request validation, automated model persistence (`joblib`), and non-root Docker containerization (`python:3.11-slim`).
- Implemented a robust automated testing suite (`pytest`) with 142 unit, integration, and container parity tests achieving 100% test pass rate and verifying numerical consistency ($<10^{-6}$ probability delta).

---

### Option B — Concise Data Science Focus
**Clinical Disease Classification Microservice** | *scikit-learn, FastAPI, Docker*
- Developed a leak-safe ML classification engine on clinical data ($N=918$), improving test set recall to **87.25%** and F1 score to **0.8641** through 5-fold Stratified `GridSearchCV` tuning of a Linear SVM.
- Deployed persisted pipeline model (`final_model.joblib`) via containerized FastAPI REST API with automated health checks, Docker Compose orchestration, and comprehensive test coverage (142 tests).
