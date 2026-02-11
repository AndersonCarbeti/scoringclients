from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
import mlflow
from mlflow.pyfunc import PyFuncModel
import mlflow.sklearn
import joblib

from .config import get_settings

@dataclass(frozen=True)
class LoadedModel:
    model: object
    source: str
    input_columns: Optional[List[str]]

def _infer_input_columns(pyfunc_model: PyFuncModel) -> Optional[List[str]]:
    try:
        schema = pyfunc_model.metadata.get_input_schema()
        if schema is None:
            return None
        return [c.name for c in schema.inputs]  # type: ignore[attr-defined]
    except Exception:
        return None

def _load_input_columns(path_str: Optional[str]) -> Optional[List[str]]:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text())
            if isinstance(data, list):
                return [str(x) for x in data]
        elif path.suffix.lower() in {".csv", ".txt"}:
            text = path.read_text().strip()
            if "," in text:
                return [c.strip() for c in text.split(",") if c.strip()]
            return [c.strip() for c in text.splitlines() if c.strip()]
    except Exception:
        return None
    return None

def _load_mlflow_model(uri: str) -> object:
    # Prefer native sklearn flavor. In this project, pyfunc may wrap a model
    # configured with predict_fn=predict, which can break probability serving.
    try:
        return mlflow.sklearn.load_model(uri)
    except Exception:
        return mlflow.pyfunc.load_model(uri)

@lru_cache(maxsize=1)
def load_model() -> LoadedModel:
    s = get_settings()

    if s.model_uri:
        model = _load_mlflow_model(s.model_uri)
        cols = _load_input_columns(s.input_columns_path)
        if cols is None and isinstance(model, PyFuncModel):
            cols = _infer_input_columns(model)
        return LoadedModel(model=model, source=f"mlflow_uri:{s.model_uri}", input_columns=cols)

    if s.local_model_path:
        local_path = Path(s.local_model_path)
        if local_path.is_file() and local_path.suffix.lower() in {".joblib", ".pkl", ".pickle"}:
            model = joblib.load(local_path)
            cols = _load_input_columns(s.input_columns_path)
            return LoadedModel(model=model, source=f"local_file:{s.local_model_path}", input_columns=cols)

        model = _load_mlflow_model(s.local_model_path)
        cols = _load_input_columns(s.input_columns_path)
        if cols is None and isinstance(model, PyFuncModel):
            cols = _infer_input_columns(model)
        return LoadedModel(model=model, source=f"local_path:{s.local_model_path}", input_columns=cols)

    raise RuntimeError("No model configured. Set MODEL_URI or LOCAL_MODEL_PATH.")

def features_to_dataframe(features: dict, input_columns: Optional[List[str]]) -> pd.DataFrame:
    df = pd.DataFrame([features])

    if input_columns:
        missing = [c for c in input_columns if c not in df.columns]
        extra = [c for c in df.columns if c not in input_columns]
        if missing:
            raise ValueError(f"Missing required features: {missing[:20]}{'...' if len(missing) > 20 else ''}")
        if extra:
            raise ValueError(f"Unexpected extra features: {extra[:20]}{'...' if len(extra) > 20 else ''}")
        df = df[input_columns]

    return df
