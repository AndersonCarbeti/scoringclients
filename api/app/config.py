from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

def _getenv(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.getenv(name, default)
    return val if val is not None and str(val).strip() != "" else None

BASE_DIR = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Settings:
    threshold: float
    fn_cost: float
    fp_cost: float
    model_uri: Optional[str]
    local_model_path: Optional[str]
    input_columns_path: Optional[str]
    clients_csv_path: Optional[str]
    client_id_col: str

def get_settings() -> Settings:
    threshold = float(_getenv("THRESHOLD", "0.402"))
    fn_cost = float(_getenv("FN_COST", "10"))
    fp_cost = float(_getenv("FP_COST", "1"))

    model_uri = _getenv("MODEL_URI")

    local_model_path = _getenv("LOCAL_MODEL_PATH")
    if local_model_path is None:
        fallback = BASE_DIR.parent / "artifacts" / "model_champion"
        if fallback.exists():
            local_model_path = str(fallback)

    input_columns_path = _getenv("INPUT_COLUMNS_PATH")
    if input_columns_path is None:
        default_cols = BASE_DIR / "data" / "input_columns.json"
        if default_cols.exists():
            input_columns_path = str(default_cols)

    clients_csv_path = _getenv("CLIENTS_CSV_PATH")
    if clients_csv_path is None:
        # Prefer the larger sample if available
        default_clients = BASE_DIR.parent / "data" / "samples" / "echantillon_clients.csv"
        if default_clients.exists():
            clients_csv_path = str(default_clients)
        else:
            fallback_clients = BASE_DIR / "data" / "clients_sample.csv"
            if fallback_clients.exists():
                clients_csv_path = str(fallback_clients)
    client_id_col = _getenv("CLIENT_ID_COL", "SK_ID_CURR") or "SK_ID_CURR"

    return Settings(
        threshold=threshold,
        fn_cost=fn_cost,
        fp_cost=fp_cost,
        model_uri=model_uri,
        local_model_path=local_model_path,
        input_columns_path=input_columns_path,
        clients_csv_path=clients_csv_path,
        client_id_col=client_id_col,
    )
