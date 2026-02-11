from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

import pandas as pd
from fastapi import FastAPI, HTTPException

from .config import get_settings
from .model_loader import features_to_dataframe, load_model
from .schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)

logger = logging.getLogger("pret_a_depenser_api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Pret a depenser — Credit Scoring API", version="1.0.0")

def _decision_from_proba(proba_default: float, threshold: float) -> Dict[str, Any]:
    predicted_class = int(proba_default >= threshold)  # 1 => risky/default
    decision = "REFUSED" if predicted_class == 1 else "APPROVED"
    return {"predicted_class": predicted_class, "decision": decision}

def _extract_first(value: Any) -> float:
    if hasattr(value, "iloc"):
        if getattr(value, "ndim", 1) == 2:
            return float(value.iloc[0, -1])
        return float(value.iloc[0])
    if isinstance(value, (list, tuple)):
        first = value[0]
        if isinstance(first, (list, tuple)):
            return float(first[-1])
        return float(first)
    try:
        return float(value[0])
    except Exception:
        return float(value)

def _predict_proba(df: pd.DataFrame) -> float:
    loaded = load_model()
    model = loaded.model
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)
        if hasattr(proba, "shape") and len(getattr(proba, "shape", [])) == 2:
            return float(proba[0, -1])
        return _extract_first(proba)

    y = model.predict(df)
    return _extract_first(y)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    loaded = load_model()
    return ModelInfoResponse(
        model_source=loaded.source,
        has_signature=bool(loaded.input_columns),
        input_columns=loaded.input_columns,
    )

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    s = get_settings()
    loaded = load_model()
    try:
        if not loaded.input_columns:
            raise ValueError("Input columns are not configured. Set INPUT_COLUMNS_PATH or add model signature.")
        df = features_to_dataframe(req.features, loaded.input_columns)
        proba = _predict_proba(df)
        if not (0.0 <= proba <= 1.0):
            raise ValueError(f"Model returned proba outside [0,1]: {proba}")
        dec = _decision_from_proba(proba, s.threshold)
        return PredictResponse(
            proba_default=proba,
            predicted_class=dec["predicted_class"],
            decision=dec["decision"],
            threshold=s.threshold,
            model_source=loaded.source,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail="Internal prediction error")

@app.get("/")
def root():
    return {"service": "pret-a-depenser-scoring-api", "status": "ok", "docs": "/docs", "health": "/health"}

@app.post("/predict-batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest):
    s = get_settings()
    loaded = load_model()
    outs = []
    for item in req.items:
        try:
            if not loaded.input_columns:
                raise ValueError("Input columns are not configured. Set INPUT_COLUMNS_PATH or add model signature.")
            df = features_to_dataframe(item.features, loaded.input_columns)
            proba = _predict_proba(df)
            if not (0.0 <= proba <= 1.0):
                raise ValueError(f"Model returned proba outside [0,1]: {proba}")
            dec = _decision_from_proba(proba, s.threshold)
            outs.append(
                {
                    "proba_default": proba,
                    "predicted_class": dec["predicted_class"],
                    "decision": dec["decision"],
                }
            )
        except Exception as e:
            outs.append({"error": str(e)})

    return {"threshold": s.threshold, "model_source": loaded.source, "predictions": outs}

@app.get("/predict-by-id/{client_id}", response_model=PredictResponse)
def predict_by_id(client_id: int):
    s = get_settings()
    loaded = load_model()
    if not s.clients_csv_path:
        raise HTTPException(status_code=400, detail="CLIENTS_CSV_PATH is not configured.")
    try:
        df = pd.read_csv(s.clients_csv_path)
        if s.client_id_col not in df.columns:
            raise ValueError(f"CLIENT_ID_COL '{s.client_id_col}' not found in CSV columns.")
        row = df[df[s.client_id_col] == client_id]
        if row.empty:
            raise ValueError(f"Client id {client_id} not found.")
        if not loaded.input_columns:
            raise ValueError("Input columns are not configured. Set INPUT_COLUMNS_PATH or add model signature.")
        if s.client_id_col in loaded.input_columns:
            features = row.iloc[0].to_dict()
        else:
            features = row.drop(columns=[s.client_id_col]).iloc[0].to_dict()
        df_one = features_to_dataframe(features, loaded.input_columns)
        proba = _predict_proba(df_one)
        dec = _decision_from_proba(proba, s.threshold)
        return PredictResponse(
            proba_default=proba,
            predicted_class=dec["predicted_class"],
            decision=dec["decision"],
            threshold=s.threshold,
            model_source=loaded.source,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Prediction by id error")
        raise HTTPException(status_code=500, detail="Internal prediction error")
