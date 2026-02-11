import json
from pathlib import Path

import requests
import streamlit as st
import pandas as pd

BASE_URL_DEFAULT = "http://localhost:8000"
DATA_DIR = Path(__file__).resolve().parent
CLIENTS_CSV_DEFAULT = DATA_DIR.parent / "data" / "samples" / "echantillon_clients.csv"

st.set_page_config(page_title="Pret a depenser — API Test", layout="centered")

st.title("Pret a depenser — API Test")

base_url = st.text_input("Base URL", value=BASE_URL_DEFAULT)

st.subheader("Health")
if st.button("Check /health"):
    try:
        r = requests.get(f"{base_url}/health", timeout=10)
        st.json(r.json())
    except Exception as e:
        st.error(str(e))

st.subheader("Model Info")
if st.button("Check /model-info"):
    try:
        r = requests.get(f"{base_url}/model-info", timeout=10)
        st.json(r.json())
    except Exception as e:
        st.error(str(e))

st.subheader("Predict (single)")

sample_predict_path = DATA_DIR / "sample_predict.json"
if sample_predict_path.exists():
    sample_predict = json.loads(sample_predict_path.read_text())
else:
    sample_predict = {"features": {}}

payload_text = st.text_area("Payload JSON", value=json.dumps(sample_predict, indent=2), height=240)
if st.button("Call /predict"):
    try:
        payload = json.loads(payload_text)
        r = requests.post(f"{base_url}/predict", json=payload, timeout=20)
        st.json(r.json())
    except Exception as e:
        st.error(str(e))

st.subheader("Predict batch")

sample_batch_path = DATA_DIR / "sample_batch.json"
if sample_batch_path.exists():
    sample_batch = json.loads(sample_batch_path.read_text())
else:
    sample_batch = {"items": []}

batch_text = st.text_area("Batch JSON", value=json.dumps(sample_batch, indent=2), height=240)
if st.button("Call /predict-batch"):
    try:
        payload = json.loads(batch_text)
        r = requests.post(f"{base_url}/predict-batch", json=payload, timeout=30)
        st.json(r.json())
    except Exception as e:
        st.error(str(e))

st.subheader("Predict by id")
clients_csv = st.text_input("Clients CSV path", value=str(CLIENTS_CSV_DEFAULT))

client_ids = []
try:
    if Path(clients_csv).exists():
        df_clients = pd.read_csv(clients_csv, usecols=["SK_ID_CURR"])
        client_ids = df_clients["SK_ID_CURR"].dropna().astype(int).tolist()
except Exception as e:
    st.warning(f"Impossible de lire le CSV clients: {e}")

if client_ids:
    client_id = st.selectbox("Client ID (from CSV)", options=client_ids[:500])
else:
    client_id = st.number_input("Client ID", min_value=0, value=100001, step=1)

if st.button("Call /predict-by-id"):
    try:
        r = requests.get(f"{base_url}/predict-by-id/{int(client_id)}", timeout=20)
        st.json(r.json())
    except Exception as e:
        st.error(str(e))
