# Phase 10 — Reusable Prediction Pipeline Summary

## 1. Executive Summary
Phase 10 implements a reusable, robust, and leakage-safe prediction/inference module in `src/predict.py`. The prediction pipeline loads the persisted scikit-learn pipeline artifact (`models/final_model.joblib`) created in Phase 9, which combines feature preprocessing and the tuned Support Vector Machine (SVC) classifier.

> **Disclaimer**: This pipeline provides machine-learning predictions based on the trained model and is not a clinically validated diagnostic system.

---

## 2. Input Schema
The prediction pipeline operates strictly on the 10 raw features present in the heart disease dataset. No additional features (such as `ca`, `thal`, or `slope`) are expected or accepted.

| Feature Name | Data Type | Feature Group | Description / Expected Categories |
| :--- | :--- | :--- | :--- |
| `age` | Numeric (int/float) | Numerical | Age in years |
| `sex` | Categorical (str/int) | Categorical | Sex ('Male', 'Female', 1, 0) |
| `cp` | Categorical (str) | Categorical | Chest pain type ('typical angina', 'atypical angina', 'non-anginal', 'asymptomatic') |
| `trestbps` | Numeric (int/float) | Numerical | Resting blood pressure (mm Hg) |
| `chol` | Numeric (int/float) | Numerical | Serum cholesterol (mg/dl); 0 indicates unrecorded |
| `fbs` | Boolean (bool/int) | Boolean | Fasting blood sugar > 120 mg/dl (True/False or 1/0) |
| `restecg` | Categorical (str) | Categorical | Resting ECG ('normal', 'ST-T wave abnormality', 'left ventricular hypertrophy') |
| `thalch` | Numeric (int/float) | Numerical | Maximum heart rate achieved |
| `exang` | Boolean (bool/int) | Boolean | Exercise induced angina (True/False or 1/0) |
| `oldpeak` | Numeric (int/float) | Numerical | ST depression induced by exercise relative to rest |

---

## 3. Model Loading Architecture
Model persistence loading is handled by `load_persisted_model()`:
- **Artifact Path**: `models/final_model.joblib`
- **Metadata Path**: `models/model_metadata.json`
- **Loaded Pipeline Structure**:
  - `preprocessor`: `ColumnTransformer` (SimpleImputer, StandardScaler, OneHotEncoder, FunctionTransformer)
  - `model`: `SVC(C=100, kernel="linear", gamma="scale", probability=True, random_state=42)`

The model artifact is loaded once and reused across single or batch predictions. No model retraining or manual feature preprocessing takes place outside the persisted pipeline.

---

## 4. Prediction Workflow & Output Schema

### Single Observation Inference (`predict_single`)
Accepts a dictionary containing all 10 required raw features:
```python
sample_input = {
    "age": 55,
    "sex": "Male",
    "cp": "asymptomatic",
    "trestbps": 140,
    "chol": 250,
    "fbs": False,
    "restecg": "normal",
    "thalch": 150,
    "exang": False,
    "oldpeak": 1.2
}
result = predict_single(sample_input)
```

**Output Schema**:
```json
{
    "predicted_class": 1,
    "predicted_probability": 0.8245,
    "model_name": "Tuned Support Vector Machine"
}
```
- `predicted_class`: Binary integer (0 = No Disease, 1 = Disease Present).
- `predicted_probability`: Float representing the predicted probability for the positive class (class 1).
- `model_name`: Descriptive name of the model from metadata.

### Batch Inference (`predict_batch`)
Accepts a pandas DataFrame, validates required features, converts `chol=0` to `NaN`, applies column reordering, generates predictions, and appends `predicted_class` and `predicted_probability` columns.
- **Non-mutation Guarantee**: The input DataFrame remains 100% unmutated. A new DataFrame copy containing original features and prediction outputs is returned.

---

## 5. Input Validation Rules
Input validation is performed by `validate_prediction_input()` prior to inference:
1. **Completeness**: All 10 required feature columns must be present in the input.
2. **Numeric Compatibility**: Numerical features (`age`, `trestbps`, `chol`, `thalch`, `oldpeak`) are checked for numeric compatibility. Non-numeric strings or invalid objects trigger a descriptive `ValueError`.
3. **Boolean Format**: Boolean features (`fbs`, `exang`) must be booleans or 0/1 integers.
4. **Categorical Validation**: Complex objects (lists, dicts) passed to categorical fields trigger immediate validation errors.
5. **Empty Inputs**: Empty dictionaries, empty DataFrames, or `None` values are rejected with explicit error messages.

---

## 6. Cholesterol Zero (`chol=0`) Handling
In the original raw dataset, `chol == 0` represents missing or unrecorded serum cholesterol values (172 observations).

To maintain 100% consistency with the training workflow data cleaning:
- During pre-processing in `prepare_prediction_dataframe()`, any observation with `chol == 0` is automatically converted to `np.nan`.
- When passed to the persisted model pipeline, the numerical `SimpleImputer(strategy='median', add_indicator=True)` step imputes the missing cholesterol value using the median learned strictly during model training (223.0 mg/dl).
- The raw dataset file on disk is never modified.

---

## 7. Reproducibility Verification
Verification was conducted using held-out test dataset observations ($N=184$):
1. Predictions were generated directly using `loaded_pipeline.predict(X_test)` and `loaded_pipeline.predict_proba(X_test)[:, 1]`.
2. Predictions were generated via `src/predict.py` `predict_batch(X_test)`.
3. **Verification Result**:
   - Class predictions matched **100% exactly** ($\text{array\_equal} == \text{True}$).
   - Positive-class probabilities matched **100% exactly** ($\text{allclose} \text{ atol}=1\text{e-}6$).

---

## 8. Limitations & Operational Guidelines
1. **Schema Dependency**: The pipeline strictly expects the 10 features defined in the project schema.
2. **Feature Ordering**: `src/predict.py` automatically handles arbitrary dictionary key orderings by enforcing consistent feature ordering prior to preprocessor execution.
3. **Non-Clinical Disclaimer**: This machine-learning pipeline is created for technical prediction based on historical dataset patterns and is not a clinically validated diagnostic system.
