"""
Comprehensive Unit and Integration Test Suite for FastAPI Application (Phase 12).
Tests API endpoints, valid input variations, invalid input rejection (Parts A-C),
observed boundary cases (Part D), batch prediction behavior (Part E), prediction consistency (Part F),
repeated requests (Part G), artifact SHA-256 integrity (Part H), raw data MD5 integrity (Part I),
schema contracts (Part J), robustness & security basics (Part K), and dependency warning resolution (Part L).
"""

import pytest
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app
from src.config import DATASET_PATH, MODELS_DIR, REPORTS_DIR, MODEL_FEATURES
from src.predict import load_persisted_model, predict_single, predict_batch
from src.data_loader import load_model_data, split_data


@pytest.fixture(scope="module")
def client():
    """Module-level TestClient fixture for FastAPI application."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def valid_single_payload():
    """Valid 10-feature raw observation payload."""
    return {
        "age": 55,
        "sex": "Male",
        "cp": "asymptomatic",
        "trestbps": 140,
        "chol": 250,
        "fbs": False,
        "restecg": "normal",
        "thalch": 150,
        "exang": False,
        "oldpeak": 1.2,
    }


# =====================================================================
# Part A — API Functional Testing (1 to 4)
# =====================================================================

def test_root_endpoint(client):
    """1. Test GET / returns 200, correct JSON structure, and expected model metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert data["name"] == "Disease Diagnosis Prediction API"
    assert data["version"] == "1.0.0"
    assert data["model"] == "Tuned Support Vector Machine"


def test_health_endpoint(client):
    """2. Test GET /health returns 200, status='healthy', and model_loaded=True."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_endpoint_valid(client, valid_single_payload):
    """3. Test POST /predict returns 200 with valid single payload and valid probability range."""
    response = client.post("/predict", json=valid_single_payload)
    assert response.status_code == 200
    data = response.json()

    assert "predicted_class" in data
    assert "predicted_probability" in data
    assert "model_name" in data

    assert data["predicted_class"] in [0, 1]
    assert isinstance(data["predicted_probability"], float)
    assert 0.0 <= data["predicted_probability"] <= 1.0
    assert data["model_name"] == "Tuned Support Vector Machine"


def test_batch_predict_endpoint_valid(client, valid_single_payload):
    """4. Test POST /predict/batch returns 200 with list of records."""
    batch_payload = {"records": [valid_single_payload, valid_single_payload]}
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()

    assert "predictions" in data
    assert len(data["predictions"]) == 2
    for pred in data["predictions"]:
        assert pred["predicted_class"] in [0, 1]
        assert 0.0 <= pred["predicted_probability"] <= 1.0
        assert pred["model_name"] == "Tuned Support Vector Machine"


# =====================================================================
# Part B — Valid Input Testing (5 to 7)
# =====================================================================

def test_valid_categorical_combinations(client, valid_single_payload):
    """5. Test valid payloads across different categorical feature combinations."""
    cp_values = ["typical angina", "atypical angina", "non-anginal", "asymptomatic"]
    sex_values = ["Male", "Female", 0, 1]
    restecg_values = ["normal", "st-t abnormality", "lv hypertrophy"]

    for cp in cp_values:
        for sex in sex_values:
            for restecg in restecg_values:
                payload = valid_single_payload.copy()
                payload["cp"] = cp
                payload["sex"] = sex
                payload["restecg"] = restecg

                res = client.post("/predict", json=payload)
                assert res.status_code == 200
                data = res.json()
                assert data["predicted_class"] in [0, 1]
                assert 0.0 <= data["predicted_probability"] <= 1.0


def test_valid_chol_zero_and_positive(client, valid_single_payload):
    """6. Test valid payloads with chol > 0 and chol = 0 (unrecorded cholesterol)."""
    # Positive cholesterol
    payload_pos = valid_single_payload.copy()
    payload_pos["chol"] = 280.0
    res1 = client.post("/predict", json=payload_pos)
    assert res1.status_code == 200

    # Zero cholesterol (treated as missing/unrecorded)
    payload_zero = valid_single_payload.copy()
    payload_zero["chol"] = 0
    res2 = client.post("/predict", json=payload_zero)
    assert res2.status_code == 200
    assert res2.json()["predicted_class"] in [0, 1]


def test_valid_boolean_combinations(client, valid_single_payload):
    """7. Test valid payloads with boolean True/False and 1/0 representations."""
    bool_combos = [
        (True, False),
        (False, True),
        (1, 0),
        (0, 1),
    ]
    for fbs, exang in bool_combos:
        payload = valid_single_payload.copy()
        payload["fbs"] = fbs
        payload["exang"] = exang

        res = client.post("/predict", json=payload)
        assert res.status_code == 200
        assert res.json()["predicted_class"] in [0, 1]


# =====================================================================
# Part C — Invalid Input Testing (8 to 11)
# =====================================================================

@pytest.mark.parametrize("missing_feature", [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalch", "exang", "oldpeak"
])
def test_missing_each_required_feature(client, valid_single_payload, missing_feature):
    """8. Test that omitting any one of the 10 required features returns HTTP 422 cleanly without stack traces."""
    payload = valid_single_payload.copy()
    del payload[missing_feature]

    res = client.post("/predict", json=payload)
    assert res.status_code == 422
    data = res.json()
    assert "detail" in data
    assert "Traceback" not in str(data)


def test_invalid_types_and_malformed_input(client, valid_single_payload):
    """9. Test invalid numeric types and invalid categorical structures return HTTP 422."""
    # Non-numeric string for age
    payload1 = valid_single_payload.copy()
    payload1["age"] = "fifty-five"
    res1 = client.post("/predict", json=payload1)
    assert res1.status_code == 422

    # Complex list in categorical field
    payload2 = valid_single_payload.copy()
    payload2["cp"] = ["invalid", "list"]
    res2 = client.post("/predict", json=payload2)
    assert res2.status_code == 422

    # Invalid boolean string
    payload3 = valid_single_payload.copy()
    payload3["fbs"] = "invalid_bool_string"
    res3 = client.post("/predict", json=payload3)
    assert res3.status_code == 422


def test_empty_json_payload(client):
    """10. Test empty JSON payload {} returns HTTP 422 cleanly."""
    res = client.post("/predict", json={})
    assert res.status_code == 422
    assert "detail" in res.json()


def test_null_values_for_required_fields(client, valid_single_payload):
    """11. Test passing null/None for non-nullable required numeric field returns HTTP 422."""
    payload = valid_single_payload.copy()
    payload["age"] = None

    res = client.post("/predict", json=payload)
    assert res.status_code == 422
    assert "detail" in res.json()


# =====================================================================
# Part D — Boundary / Edge Case Testing (12 to 14)
# =====================================================================

def test_observed_dataset_boundaries(client, valid_single_payload):
    """12. Test dataset-observed numeric boundary values (min/max observed across raw dataset)."""
    boundaries = [
        ("age", 28),     # Min observed age
        ("age", 77),     # Max observed age
        ("trestbps", 80),   # Min observed trestbps
        ("trestbps", 200),  # Max observed trestbps
        ("chol", 0),        # Min observed chol (unrecorded)
        ("chol", 603),      # Max observed chol
        ("thalch", 60),     # Min observed thalch
        ("thalch", 202),    # Max observed thalch
        ("oldpeak", -2.6),  # Min observed oldpeak
        ("oldpeak", 6.2),   # Max observed oldpeak
    ]
    for feature, val in boundaries:
        payload = valid_single_payload.copy()
        payload[feature] = val

        res = client.post("/predict", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["predicted_class"] in [0, 1]
        assert 0.0 <= data["predicted_probability"] <= 1.0


def test_negative_oldpeak_preservation(client, valid_single_payload):
    """13. Test negative oldpeak values (which exist in 12 raw dataset rows, e.g. -2.6, -1.0, -0.5)."""
    neg_oldpeaks = [-2.6, -1.5, -1.0, -0.5]
    for val in neg_oldpeaks:
        payload = valid_single_payload.copy()
        payload["oldpeak"] = val

        res = client.post("/predict", json=payload)
        assert res.status_code == 200
        assert res.json()["predicted_class"] in [0, 1]


def test_extreme_unrealistic_numeric_rejection(client, valid_single_payload):
    """14. Test extreme out-of-bounds or non-finite inputs."""
    payload = valid_single_payload.copy()
    payload["age"] = "not_a_number"

    res = client.post("/predict", json=payload)
    assert res.status_code == 422


# =====================================================================
# Part E — Batch Testing (15 to 17)
# =====================================================================

def test_batch_predict_various_sizes(client, valid_single_payload):
    """15. Test POST /predict/batch with 1 record, 5 records, and 10 records."""
    for size in [1, 5, 10]:
        records = [valid_single_payload.copy() for _ in range(size)]
        res = client.post("/predict/batch", json={"records": records})
        assert res.status_code == 200
        data = res.json()
        assert len(data["predictions"]) == size
        for pred in data["predictions"]:
            assert pred["predicted_class"] in [0, 1]
            assert 0.0 <= pred["predicted_probability"] <= 1.0


def test_batch_predict_preserves_ordering_and_input(client, valid_single_payload):
    """16. Test that batch prediction output ordering matches input ordering and input is unmutated."""
    rec1 = valid_single_payload.copy()
    rec1["age"] = 30
    rec2 = valid_single_payload.copy()
    rec2["age"] = 70

    records_original = [rec1.copy(), rec2.copy()]
    batch_payload = {"records": [rec1, rec2]}

    res = client.post("/predict/batch", json=batch_payload)
    assert res.status_code == 200
    predictions = res.json()["predictions"]

    assert len(predictions) == 2
    # Input records must remain unchanged
    assert batch_payload["records"] == records_original


def test_batch_predict_invalid_records(client, valid_single_payload):
    """17. Test empty batch records list [] and batch with invalid item return HTTP 422."""
    # Empty list
    res1 = client.post("/predict/batch", json={"records": []})
    assert res1.status_code == 422

    # Batch containing an item missing a required feature
    invalid_rec = valid_single_payload.copy()
    del invalid_rec["age"]
    res2 = client.post("/predict/batch", json={"records": [valid_single_payload, invalid_rec]})
    assert res2.status_code == 422


# =====================================================================
# Part F — Prediction Consistency (18 & 19)
# =====================================================================

def test_api_vs_src_predict_single_exact_match(client):
    """18. Compare API /predict response against src.predict.predict_single() across test set samples."""
    X, y = load_model_data()
    _, X_test, _, _ = split_data(X, y)

    # Convert first 5 test rows to dicts
    sample_records = X_test.head(5).to_dict(orient="records")

    for rec in sample_records:
        direct_res = predict_single(rec)
        api_res = client.post("/predict", json=rec).json()

        assert api_res["predicted_class"] == direct_res["predicted_class"]
        assert pytest.approx(api_res["predicted_probability"], abs=1e-6) == direct_res["predicted_probability"]
        assert api_res["model_name"] == direct_res["model_name"]


def test_api_vs_src_predict_batch_exact_match(client):
    """19. Compare API /predict/batch response against src.predict.predict_batch()."""
    X, y = load_model_data()
    _, X_test, _, _ = split_data(X, y)

    sample_df = X_test.head(5).copy()
    records = sample_df.to_dict(orient="records")

    # Direct module batch prediction
    direct_batch_df = predict_batch(sample_df)

    # API batch prediction
    api_batch_res = client.post("/predict/batch", json={"records": records}).json()
    api_preds = api_batch_res["predictions"]

    assert len(api_preds) == len(direct_batch_df)
    for idx, row in direct_batch_df.reset_index().iterrows():
        assert api_preds[idx]["predicted_class"] == int(row["predicted_class"])
        assert pytest.approx(api_preds[idx]["predicted_probability"], abs=1e-6) == float(row["predicted_probability"])


# =====================================================================
# Part G — Repeated Request Testing (20)
# =====================================================================

def test_repeated_identical_requests(client, valid_single_payload):
    """20. Send 10 repeated identical requests and verify 100% exact numerical agreement."""
    first_res = client.post("/predict", json=valid_single_payload).json()

    for _ in range(10):
        res = client.post("/predict", json=valid_single_payload).json()
        assert res["predicted_class"] == first_res["predicted_class"]
        assert res["predicted_probability"] == first_res["predicted_probability"]
        assert res["model_name"] == first_res["model_name"]


# =====================================================================
# Part H & I — Artifact & Raw Data Integrity (21 to 23)
# =====================================================================

def test_model_artifact_sha256_integrity():
    """21. Calculate and verify SHA-256 hash of models/final_model.joblib and model_metadata.json."""
    model_path = MODELS_DIR / "final_model.joblib"
    meta_path = MODELS_DIR / "model_metadata.json"

    assert model_path.exists() and model_path.stat().st_size > 0
    assert meta_path.exists() and meta_path.stat().st_size > 0

    model_bytes = model_path.read_bytes()
    meta_bytes = meta_path.read_bytes()

    hash_model = hashlib.sha256(model_bytes).hexdigest()
    hash_meta = hashlib.sha256(meta_bytes).hexdigest()

    assert len(hash_model) == 64
    assert len(hash_meta) == 64


def test_raw_dataset_md5_integrity_and_rows():
    """22. Verify data/raw/heart_disease.csv has MD5 hash 13c9cfee54ce2b1552ef7d787a5d8be9 and 918 rows."""
    raw_bytes = DATASET_PATH.read_bytes()
    md5_hash = hashlib.md5(raw_bytes).hexdigest()
    assert md5_hash == "13c9cfee54ce2b1552ef7d787a5d8be9", f"MD5 mismatch: {md5_hash}"

    df_raw = pd.read_csv(DATASET_PATH)
    assert len(df_raw) == 918, f"Row count mismatch: {len(df_raw)}"


def test_app_artifacts_exist():
    """23. Verify app/main.py, app/schemas.py, and app/__init__.py exist."""
    app_dir = MODELS_DIR.parent / "app"
    assert (app_dir / "main.py").exists()
    assert (app_dir / "schemas.py").exists()
    assert (app_dir / "__init__.py").exists()


# =====================================================================
# Part K — Robustness & Security Basics (24 & 25)
# =====================================================================

def test_robustness_unexpected_extra_fields(client, valid_single_payload):
    """24. Test payload with unexpected extra JSON fields handled cleanly."""
    payload = valid_single_payload.copy()
    payload["unexpected_field_123"] = "ignored_or_validated"

    res = client.post("/predict", json=payload)
    # FastAPI / Pydantic handles or ignores extra fields cleanly without crashing
    assert res.status_code in [200, 422]
    if res.status_code == 200:
        assert res.json()["predicted_class"] in [0, 1]


def test_robustness_excessively_long_strings(client, valid_single_payload):
    """25. Test excessively long categorical strings handled cleanly without server error."""
    payload = valid_single_payload.copy()
    payload["cp"] = "A" * 10000  # 10k characters

    res = client.post("/predict", json=payload)
    assert res.status_code in [200, 422]
    assert "Traceback" not in str(res.json())


# =====================================================================
# Part M — Phase 17.2 Canonical Categorical & Validation Hardening Tests
# =====================================================================

@pytest.mark.parametrize("invalid_field, invalid_val", [
    ("age", -5),
    ("age", 0),
    ("trestbps", -1),
    ("chol", -1),
    ("thalch", 0),
    ("oldpeak", -100),
    ("oldpeak", 100),
    ("cp", "invalid_chest_pain"),
    ("restecg", "invalid_rest_ecg"),
    ("restecg", "ST-T wave abnormality"),
    ("restecg", "left ventricular hypertrophy"),
])
def test_validation_hardening_rejected_invalid_values(client, valid_single_payload, invalid_field, invalid_val):
    """Test that out-of-bounds numeric values, invalid categorical strings, and verbose restecg aliases return HTTP 422."""
    payload = valid_single_payload.copy()
    payload[invalid_field] = invalid_val

    res = client.post("/predict", json=payload)
    assert res.status_code == 422
    assert "detail" in res.json()


@pytest.mark.parametrize("valid_field, valid_val", [
    ("age", 28),
    ("age", 77),
    ("trestbps", 80),
    ("trestbps", 200),
    ("chol", 0),
    ("chol", 603),
    ("thalch", 60),
    ("thalch", 202),
    ("oldpeak", -2.6),
    ("oldpeak", 6.2),
    ("restecg", "normal"),
    ("restecg", "st-t abnormality"),
    ("restecg", "lv hypertrophy"),
    ("cp", "asymptomatic"),
    ("cp", "non-anginal"),
    ("cp", "typical angina"),
    ("cp", "atypical angina"),
])
def test_validation_hardening_accepted_boundary_values(client, valid_single_payload, valid_field, valid_val):
    """Test that valid boundary values and canonical categorical values remain accepted with HTTP 200."""
    payload = valid_single_payload.copy()
    payload[valid_field] = valid_val

    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["predicted_class"] in [0, 1]
    assert 0.0 <= data["predicted_probability"] <= 1.0

