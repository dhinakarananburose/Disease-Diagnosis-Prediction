"""
Pydantic Schemas for Disease Diagnosis Prediction API.
Defines request and response data models for single and batch predictions,
health checks, and root info endpoints.
"""

from typing import Union, List, Optional
from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    """Schema for a single prediction request containing the 10 required raw features."""

    age: Union[int, float] = Field(..., description="Age in years")
    sex: Union[str, int] = Field(..., description="Sex ('Male', 'Female', 1, 0)")
    cp: str = Field(
        ...,
        description="Chest pain type ('typical angina', 'atypical angina', 'non-anginal', 'asymptomatic')",
    )
    trestbps: Union[int, float] = Field(
        ..., description="Resting blood pressure in mm Hg"
    )
    chol: Union[int, float] = Field(
        ..., description="Serum cholesterol in mg/dl (0 indicates unrecorded)"
    )
    fbs: Union[bool, int, float, str] = Field(
        ..., description="Fasting blood sugar > 120 mg/dl (True/False or 1/0)"
    )
    restecg: str = Field(
        ...,
        description="Resting ECG ('normal', 'ST-T wave abnormality', 'left ventricular hypertrophy')",
    )
    thalch: Union[int, float] = Field(
        ..., description="Maximum heart rate achieved"
    )
    exang: Union[bool, int, float, str] = Field(
        ..., description="Exercise induced angina (True/False or 1/0)"
    )
    oldpeak: Union[int, float] = Field(
        ..., description="ST depression induced by exercise relative to rest"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionResponse(BaseModel):
    """Schema for a single prediction response."""

    predicted_class: int = Field(
        ..., description="Predicted class label (0 = No Disease, 1 = Disease Present)"
    )
    predicted_probability: float = Field(
        ..., description="Model predicted probability for the positive class (class 1)"
    )
    model_name: str = Field(..., description="Name of the persisted model used for inference")


class BatchPredictionRequest(BaseModel):
    """Schema for a batch prediction request."""

    records: List[PredictionInput] = Field(
        ..., description="List of prediction feature input records"
    )


class BatchPredictionResponse(BaseModel):
    """Schema for a batch prediction response."""

    predictions: List[PredictionResponse] = Field(
        ..., description="List of prediction results"
    )


class HealthResponse(BaseModel):
    """Schema for API health status."""

    status: str = Field(..., description="Service status ('healthy')")
    model_loaded: bool = Field(..., description="Whether the model artifact is loaded")


class RootResponse(BaseModel):
    """Schema for API root metadata."""

    name: str = Field(..., description="API name")
    version: str = Field(..., description="API version")
    model: str = Field(..., description="Model name")
