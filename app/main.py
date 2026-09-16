"""
FastAPI Application Entry Point for Disease Diagnosis Prediction (Phase 11).
Exposes HTTP endpoints for API metadata, health status, and single/batch predictions.
Uses src.predict and persisted artifact models/final_model.joblib.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from fastapi import FastAPI, Request, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from app.schemas import (
    PredictionInput,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    RootResponse,
)
from src.predict import (
    load_persisted_model,
    load_model_metadata,
    predict_single,
    predict_batch,
)

# Application Lifespan Handler for Startup Model Loading
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager that loads the persisted model artifact once at startup.
    """
    try:
        app.state.model = load_persisted_model()
        app.state.metadata = load_model_metadata()
        app.state.model_loaded = True
    except Exception as e:
        app.state.model = None
        app.state.metadata = {}
        app.state.model_loaded = False
    yield


app = FastAPI(
    title="Disease Diagnosis Prediction API",
    description=(
        "Production-structured REST API exposing machine-learning predictions "
        "from the trained final model pipeline. "
        "This pipeline provides machine-learning predictions based on the trained model "
        "and is not a clinically validated diagnostic system."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Mount frontend static directory if present
frontend_path = Path(__file__).resolve().parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/ui", StaticFiles(directory=str(frontend_path), html=True), name="ui")



@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handles ValueErrors from input validation without exposing stack traces."""
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)},
    )


@app.exception_handler(TypeError)
async def type_error_handler(request: Request, exc: TypeError):
    """Handles TypeErrors cleanly without exposing stack traces."""
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)},
    )


@app.get("/", response_model=RootResponse, summary="API Root Info")
async def root() -> RootResponse:
    """Returns basic API name, version, and model information."""
    metadata = getattr(app.state, "metadata", {})
    model_name = metadata.get("model_name", "Tuned Support Vector Machine")
    return RootResponse(
        name="Disease Diagnosis Prediction API",
        version="1.0.0",
        model=model_name,
    )


@app.get("/health", response_model=HealthResponse, summary="Health Check")
async def health() -> HealthResponse:
    """Verifies service health and confirms the persisted model is loaded."""
    model_loaded = getattr(app.state, "model_loaded", False)
    if not model_loaded or getattr(app.state, "model", None) is None:
        # Attempt loading if not loaded during lifespan startup (e.g. testing fallback)
        try:
            app.state.model = load_persisted_model()
            app.state.metadata = load_model_metadata()
            app.state.model_loaded = True
            model_loaded = True
        except Exception:
            model_loaded = False

    status_str = "healthy" if model_loaded else "unhealthy"
    return HealthResponse(status=status_str, model_loaded=model_loaded)


@app.post("/predict", response_model=PredictionResponse, summary="Single Prediction")
async def predict(request_data: PredictionInput) -> PredictionResponse:
    """
    Accepts 10 raw feature values for a single observation and returns:
    - predicted_class (0 or 1)
    - predicted_probability (positive-class probability)
    - model_name
    """
    input_dict = request_data.model_dump()
    model = getattr(app.state, "model", None)
    if model is None:
        model = load_persisted_model()
        app.state.model = model

    result = predict_single(input_dict, model=model)
    return PredictionResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictionResponse, summary="Batch Prediction")
async def predict_batch_endpoint(batch_request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Accepts a list of prediction records and returns predictions for each record.
    Does not mutate the original request data.
    """
    if not batch_request.records:
        raise ValueError("Batch request records list cannot be empty.")

    records_list = [r.model_dump() for r in batch_request.records]
    input_df = pd.DataFrame(records_list)

    model = getattr(app.state, "model", None)
    if model is None:
        model = load_persisted_model()
        app.state.model = model

    batch_res_df = predict_batch(input_df, model=model)

    metadata = getattr(app.state, "metadata", {})
    model_name = metadata.get("model_name", "Tuned Support Vector Machine")

    predictions = []
    for _, row in batch_res_df.iterrows():
        predictions.append(
            PredictionResponse(
                predicted_class=int(row["predicted_class"]),
                predicted_probability=float(row["predicted_probability"]),
                model_name=model_name,
            )
        )

    return BatchPredictionResponse(predictions=predictions)
