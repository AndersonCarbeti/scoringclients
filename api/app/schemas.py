from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="Feature vector as key/value pairs")

class BatchPredictRequest(BaseModel):
    items: List[PredictRequest]

class PredictResponse(BaseModel):
    proba_default: float
    predicted_class: int = Field(..., description="1 = default risk, 0 = no default risk")
    decision: str = Field(..., description="APPROVED or REFUSED (based on threshold)")
    threshold: float
    model_source: str = Field(..., description="mlflow_uri or local_path")

class BatchPredictItemResponse(BaseModel):
    proba_default: Optional[float] = None
    predicted_class: Optional[int] = None
    decision: Optional[str] = None
    error: Optional[str] = None

class BatchPredictResponse(BaseModel):
    threshold: float
    model_source: str
    predictions: List[BatchPredictItemResponse]

class ModelInfoResponse(BaseModel):
    model_source: str
    has_signature: bool
    input_columns: Optional[List[str]] = None
