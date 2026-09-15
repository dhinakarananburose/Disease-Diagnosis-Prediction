# Phase 9 — Final Model Persistence Summary

## 1. Selected Model & Configuration
- **Selected Model:** Tuned Support Vector Machine (`SVC`)
- **Hyperparameters:**
  - `C`: 100
  - `kernel`: `linear`
  - `gamma`: `scale`
  - `probability`: True
  - `random_state`: 42

---

## 2. Dataset & Training Scope Safeguards
- **Training Data:** Fitted strictly on training set `X_train`, `y_train` ($N=734$).
- **Held-Out Test Data:** The test set ($N=184$) was **NOT** used for training or fitting. It remained reserved exclusively for validation verification.
- **Preprocessing:** Integrated inside the scikit-learn `Pipeline` (`preprocessor -> model`). Includes median imputation for `chol==0`, standard scaling, and category-level one-hot encoding across 10 input features.

---

## 3. Persisted Artifacts
- **Joblib Binary Pipeline:** [models/final_model.joblib](file:///c:/projects/Disease-Diagnosis-Prediction/models/final_model.joblib)
- **JSON Metadata File:** [models/model_metadata.json](file:///c:/projects/Disease-Diagnosis-Prediction/models/model_metadata.json)

---

## 4. Verification & Reproducibility Results
- **Pipeline Validity:** Loaded artifact is a valid scikit-learn `Pipeline` containing `preprocessor` and `model` steps.
- **Hyperparameter Verification:** Confirmed `C=100`, `kernel='linear'`, `probability=True`.
- **Prediction Consistency:** Test predictions (`.predict(X_test)`) and predicted probabilities (`.predict_proba(X_test)`) match the original in-memory Phase 8 model outputs **100% exactly**.
- **Raw CSV Integrity:** Raw dataset `data/raw/heart_disease.csv` remains unaltered (MD5: `13c9cfee54ce2b1552ef7d787a5d8be9`, $N=918$).
