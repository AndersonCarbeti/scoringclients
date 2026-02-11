from pathlib import Path
import json

import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.model_loader import LoadedModel, load_model

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def _set_env():
    import os

    os.environ.setdefault("INPUT_COLUMNS_PATH", str(DATA_DIR / "input_columns.json"))
    # Default to the local MLflow model shipped in the repo (sibling folder)
    default_model = ROOT.parent / "artifacts" / "model_champion"
    os.environ.setdefault("LOCAL_MODEL_PATH", str(default_model))
    os.environ.setdefault("CLIENTS_CSV_PATH", str(DATA_DIR / "clients_sample.csv"))


class _FakeModel:
    def predict_proba(self, df):
        return [[0.2, 0.8]]


def _client(monkeypatch=None):
    _set_env()
    load_model.cache_clear()

    cols = json.loads((DATA_DIR / "input_columns.json").read_text())
    fake = LoadedModel(model=_FakeModel(), source="fake", input_columns=cols)

    if monkeypatch is not None:
        monkeypatch.setattr("app.model_loader.load_model", lambda: fake)
        monkeypatch.setattr("app.main.load_model", lambda: fake)

    return TestClient(app)


def test_health(monkeypatch):
    client = _client(monkeypatch)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_model_info(monkeypatch):
    client = _client(monkeypatch)
    resp = client.get("/model-info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["has_signature"] is True
    assert isinstance(data["input_columns"], list)


def test_predict(monkeypatch):
    client = _client(monkeypatch)
    payload = json.loads((DATA_DIR / "sample_predict.json").read_text())
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["proba_default"] <= 1.0
    assert body["decision"] in {"APPROVED", "REFUSED"}


def test_predict_batch(monkeypatch):
    client = _client(monkeypatch)
    payload = json.loads((DATA_DIR / "sample_batch.json").read_text())
    resp = client.post("/predict-batch", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "predictions" in body and len(body["predictions"]) == 2


def test_predict_by_id(monkeypatch):
    client = _client(monkeypatch)
    df = pd.read_csv(DATA_DIR / "clients_sample.csv")
    client_id = int(df["SK_ID_CURR"].iloc[0])
    resp = client.get(f"/predict-by-id/{client_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["proba_default"] <= 1.0


def test_predict_missing_feature_returns_400(monkeypatch):
    client = _client(monkeypatch)
    payload = {"features": {"SK_ID_CURR": 100001}}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 400
