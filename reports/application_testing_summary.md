# Phase 12 — Application / API Testing & Validation Summary

## 1. Test Objectives
The objective of Phase 12 is to conduct comprehensive application-level testing and validation of the FastAPI prediction service (`app/main.py`). Testing evaluated functional correctness, schema validation, edge/boundary handling, batch execution, reproducibility, artifact integrity, and robustness against invalid inputs.

---

## 2. API Endpoints Tested
The following endpoints were systematically tested:
- `GET /` — API metadata endpoint
- `GET /health` — Service health and model load status
- `POST /predict` — Single observation prediction endpoint
- `POST /predict/batch` — Batch prediction endpoint

---

## 3. Functional Tests
- **GET /**: Verified HTTP 200 response, returning `{"name": "Disease Diagnosis Prediction API", "version": "1.0.0", "model": "Tuned Support Vector Machine"}`.
- **GET /health**: Verified HTTP 200 response with `{"status": "healthy", "model_loaded": true}` confirming the model artifact is loaded and ready.
- **POST /predict**: Verified HTTP 200 response containing `predicted_class` ($\in \{0, 1\}$), `predicted_probability` ($\in [0.0, 1.0]$), and `model_name`.
- **POST /predict/batch**: Verified HTTP 200 response containing array of prediction records.

---

## 4. Valid-Input Tests
Evaluated multiple valid payload variations:
- Typical valid record with default 10 raw feature values.
- Different categorical value combinations:
  - `cp`: `'typical angina'`, `'atypical angina'`, `'non-anginal'`, `'asymptomatic'`
  - `sex`: `'Male'`, `'Female'`, `0`, `1`
  - `restecg`: `'normal'`, `'ST-T wave abnormality'`, `'left ventricular hypertrophy'`
- Cholesterol variations: `chol > 0` (e.g. 280 mg/dl) and `chol == 0` (unrecorded cholesterol).
- Boolean variations: `fbs` and `exang` as `True`, `False`, `1`, `0`.
- All valid inputs were accepted cleanly with HTTP 200.

---

## 5. Invalid-Input Tests
Verified clean rejection of malformed or incomplete payloads without server crashes or stack trace exposure:
- **Missing Features**: Individually omitted each of the 10 required features (`age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalch`, `exang`, `oldpeak`). All 10 returned HTTP 422 Unprocessable Entity with descriptive error messages.
- **Invalid Numerical Types**: Non-numeric strings (e.g., `"fifty-five"`) returned HTTP 422.
- **Invalid Categoricals**: Complex list/dict structures passed to categorical fields returned HTTP 422.
- **Invalid Booleans**: Unrecognized boolean strings returned HTTP 422.
- **Empty JSON & Nulls**: Empty JSON `{}` and `null` values for required fields returned HTTP 422 cleanly.

---

## 6. Boundary Tests
Tested dataset-observed numerical boundaries based on observed raw data ranges:
- `age`: 28 (min) to 77 (max)
- `trestbps`: 80 (min) to 200 (max)
- `chol`: 0 (min / unrecorded) to 603 (max)
- `thalch`: 60 (min) to 202 (max)
- `oldpeak`: -2.6 (min) to 6.2 (max)
- **Negative Oldpeak Values**: Verified that negative ST depression values present in the raw dataset (e.g. -2.6, -1.5, -0.5) execute cleanly and produce valid predictions without rejection.

---

## 7. Batch Tests
Tested `POST /predict/batch`:
- Verified batch sizes of 1, 5, and 10 records.
- Verified output array length matches input record count.
- Confirmed that predictions preserve input record order.
- Verified that original request payload records remain 100% unmutated.
- Verified that empty batch lists (`records: []`) or batches containing invalid records return HTTP 422 cleanly.

---

## 8. Prediction Consistency Tests
Compared prediction outputs across held-out test data samples:
- `POST /predict` vs `src.predict.predict_single()`: Produced **100% exact numerical match** for `predicted_class`, `predicted_probability`, and `model_name`.
- `POST /predict/batch` vs `src.predict.predict_batch()`: Produced **100% exact numerical match** across all batch items.

---

## 9. Repeated-Request Tests
- Sent 10 consecutive identical HTTP requests to `POST /predict`.
- Confirmed that `predicted_class` and `predicted_probability` remained 100% identical across all 10 calls.
- Confirmed zero model retraining or artifact modification took place during inference.

---

## 10. Artifact Integrity
Model artifact SHA-256 hashes calculated before and after API testing:
- `models/final_model.joblib`: `16081825890dc9a3fae9c8fd2216272379ee7ac0f7712072f983a0bb897a1568`
- `models/model_metadata.json`: `5ec6e595a0c7d5d0dc899611b1f3f3480c1792564c2113e6e39478329467c836`

**Result**: Hashes remained 100% unchanged.

---

## 11. Raw Data Integrity
- `data/raw/heart_disease.csv` row count: **918 rows**
- MD5 hash: `13c9cfee54ce2b1552ef7d787a5d8be9`

**Result**: Raw dataset file on disk remains completely intact and unmodified.

---

## 12. Dependency / Warning Status
- **Resolved Warning**: Installed `httpx2` dependency to eliminate the `StarletteDeprecationWarning` regarding TestClient httpx usage.
- **Platform Warning**: 1 platform-level warning (`loky/backend/context.py` physical core lookup on Windows) remains active; this is a harmless OS environment warning and does not impact application correctness.

---

## 13. Limitations & Operational Scope
- The API is scoped strictly to the 10 raw features of the heart disease dataset.
- Input validation rejects malformed requests cleanly with HTTP 422 JSON error messages.
- The service runs in local inference mode and does not implement authentication, rate limiting, or cloud infrastructure.

---

## 14. Final Testing Conclusion
Application behavior was validated under the tested scenarios. A total of 138 unit and integration tests (103 pipeline tests + 35 API tests) passed with 0 failures.
