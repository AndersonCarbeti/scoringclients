import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.model_loader import load_model

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

pytestmark = pytest.mark.integration


def _set_env():
    os.environ.setdefault("INPUT_COLUMNS_PATH", str(DATA_DIR / "input_columns.json"))
    joblib_model = ROOT / "outputs" / "models" / "pipeline_lgbm.joblib"
    os.environ["LOCAL_MODEL_PATH"] = str(joblib_model)
    os.environ.setdefault("CLIENTS_CSV_PATH", str(DATA_DIR / "clients_sample.csv"))


def test_predict_joblib_model():
    if os.getenv("RUN_INTEGRATION") != "1":
        pytest.skip("Set RUN_INTEGRATION=1 to run joblib integration test")

    pytest.importorskip("lightgbm")

    _set_env()
    load_model.cache_clear()
    client = TestClient(app)

    payload = json.loads((DATA_DIR / "sample_predict.json").read_text())
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["proba_default"] <= 1.0
