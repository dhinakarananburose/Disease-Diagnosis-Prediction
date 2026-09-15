# Phase 17.1 & 17.2 — API Input Validation Hardening & Canonical Categorical Fix Report

## 1. Validation Weakness Found & Addressed
- **Numerical Validation:** Pre-publication testing revealed that the API accepted non-physiological, logically invalid input values (such as `age: -5`, `trestbps: -1`, or `chol: -1`) with HTTP 200 responses.
- **Categorical Alignment Fix (Phase 17.2):** Audit of `models/final_model.joblib` revealed that the persisted `OneHotEncoder` was fitted on the exact raw dataset vocabulary:
  - `restecg`: `["normal", "st-t abnormality", "lv hypertrophy"]`
  - `cp`: `["asymptomatic", "non-anginal", "typical angina", "atypical angina"]`
  
  The intermediate schema allowed verbose aliases (`"ST-T wave abnormality"`, `"left ventricular hypertrophy"`) which do NOT exist in the raw dataset or OneHotEncoder vocabulary. Because `handle_unknown="ignore"`, those verbose strings were silently encoded as all-zero unknown categories (`[0, 0, 0]`), altering prediction probabilities.

## 2. Validation Rules Enforced

### Pydantic Schemas ([`app/schemas.py`](file:///c:/projects/Disease-Diagnosis-Prediction/app/schemas.py)) & Module Validation ([`src/predict.py`](file:///c:/projects/Disease-Diagnosis-Prediction/src/predict.py)):
- **`age`:** `1 <= age <= 120` (Pydantic `ge=1, le=120`). Rejects negative or zero ages.
- **`trestbps`:** `0 <= trestbps <= 300` (Pydantic `ge=0, le=300`). Rejects negative blood pressures.
- **`chol`:** `0 <= chol <= 1500` (Pydantic `ge=0, le=1500`). Rejects negative cholesterol. `chol=0` remains accepted (converted to `np.nan` missing value by the pipeline).
- **`thalch`:** `1 <= thalch <= 250` (Pydantic `ge=1, le=250`). Rejects negative or zero heart rates.
- **`oldpeak`:** `-10.0 <= oldpeak <= 15.0` (Pydantic `ge=-10.0, le=15.0`). Rejects extreme non-physiological ST depressions while preserving valid negative values observed down to `-2.6` in the raw dataset.
- **`cp` (Chest Pain Type):** Restricted strictly to canonical categories: `Literal["asymptomatic", "non-anginal", "typical angina", "atypical angina"]`.
- **`restecg` (Resting ECG):** Restricted strictly to canonical fitted categories: `Literal["normal", "st-t abnormality", "lv hypertrophy"]`.
- **Existing Representations Preserved:** `sex`, `fbs`, and `exang` existing accepted representations were preserved without modification.

## 3. Valid Boundary Cases Tested
Verified that the following valid boundary payloads return HTTP 200 and produce valid predictions:
- `age`: 28 (min dataset age) and 77 (max dataset age)
- `trestbps`: 80 and 200
- `chol`: 0 (unrecorded/missing value handled by imputer) and 603
- `thalch`: 60 and 202
- `oldpeak`: -2.6 (negative ST depression present in 12 raw rows) and 6.2
- `restecg`: `"normal"`, `"st-t abnormality"`, `"lv hypertrophy"`
- `cp`: `"asymptomatic"`, `"non-anginal"`, `"typical angina"`, `"atypical angina"`

## 4. Invalid Cases Tested
Verified that out-of-bounds numeric values, invalid categorical strings, and non-canonical verbose aliases return HTTP 422:
- `age`: -5 and 0
- `trestbps`: -1
- `chol`: -1
- `thalch`: 0
- `oldpeak`: -100 and 100
- `cp`: `"invalid_chest_pain"`
- `restecg`: `"invalid_rest_ecg"`
- `restecg` verbose aliases: `"ST-T wave abnormality"` (HTTP 422) and `"left ventricular hypertrophy"` (HTTP 422)

## 5. Test Suite Result
Ran full test suite (`pytest tests/ -q`):
**`170 passed, 1 warning in 74.61s`** (All unit and integration tests passing).

## 6. API Verification Result
Container HTTP endpoints tested against live Docker deployment (`http://localhost:8000`):
1. **`GET /health`:** Returned `HTTP 200`: `{"status": "healthy", "model_loaded": true}`
2. **Canonical `restecg` Values:** All 3 canonical `restecg` values (`"normal"`, `"st-t abnormality"`, `"lv hypertrophy"`) returned `HTTP 200`.
3. **Invalid `restecg` Aliases:** Both verbose aliases (`"ST-T wave abnormality"`, `"left ventricular hypertrophy"`) returned `HTTP 422`.
4. **`POST /predict` (Validated Synthetic Payload):** Returned `HTTP 200` with exact previously validated predictions:
   ```json
   {
     "predicted_class": 1,
     "predicted_probability": 0.6461265923693491,
     "model_name": "Tuned Support Vector Machine"
   }
   ```

## 7. Model & Metadata Hash Verification
- **`models/final_model.joblib` SHA-256:** `16081825890dc9a3fae9c8fd2216272379ee7ac0f7712072f983a0bb897a1568` (VERIFIED UNCHANGED)
- **`models/model_metadata.json` SHA-256:** `5ec6e595a0c7d5d0dc899611b1f3f3480c1792564c2113e6e39478329467c836` (VERIFIED UNCHANGED)

## 8. Files Modified
- [`app/schemas.py`](file:///c:/projects/Disease-Diagnosis-Prediction/app/schemas.py): Updated `restecg` `Literal` to exact canonical values `["normal", "st-t abnormality", "lv hypertrophy"]`.
- [`src/predict.py`](file:///c:/projects/Disease-Diagnosis-Prediction/src/predict.py): Updated `ALLOWED_RESTECG` to `{"normal", "st-t abnormality", "lv hypertrophy"}`.
- [`tests/test_api.py`](file:///c:/projects/Disease-Diagnosis-Prediction/tests/test_api.py): Updated categorical test suites to test canonical values (HTTP 200) and reject verbose aliases (HTTP 422).
- [`reports/api_validation_hardening_report.md`](file:///c:/projects/Disease-Diagnosis-Prediction/reports/api_validation_hardening_report.md): Updated phase completion report artifact.

## 9. Model Retraining Confirmation
**CONFIRMED:** The ML model was NOT retrained. `models/final_model.joblib` and `models/model_metadata.json` remain strictly untouched.

## 10. Git Operations Confirmation
**CONFIRMED:** No `git commit` or `git push` operations were executed.
