# Projet 7 — Pret a depenser (scoring credit)

Ce dossier est structure pour la soutenance et regroupe :
- notebooks de modelisation
- artefacts modele + MLflow
- API FastAPI
- data drift Evidently
- CI/CD

## Structure

- `notebooks/` : notebooks modele + api_test
- `data/raw/` : datasets d origine
- `data/processed/` : datasets prepares
- `data/samples/` : echantillons clients pour l API
- `artifacts/` : modele champion + threshold/input columns
- `mlruns/` : tracking MLflow local
- `outputs/` : rapports (shap/evidently), exports
- `api/` : service FastAPI + tests + streamlit
- `.github/workflows/` : CI/CD
- `slides/` : supports de soutenance

## API

Voir `api/README.md`.

## Validation locale soutenance

1) Preparer environnement :
```bash
cd /Users/andersoncarbeti/Projet_7_final/Projet_7_final_final/api
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

2) Lancer tests :
```bash
pytest -q
```

3) Lancer API :
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

4) Verifier endpoints (nouveau terminal) :
```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/model-info
curl -s http://localhost:8000/predict -H 'Content-Type: application/json' -d @/Users/andersoncarbeti/Projet_7_final/Projet_7_final_final/api/sample_predict.json
curl -s http://localhost:8000/predict-batch -H 'Content-Type: application/json' -d @/Users/andersoncarbeti/Projet_7_final/Projet_7_final_final/api/sample_batch.json
curl -s http://localhost:8000/predict-by-id/100001
```

5) Lancer Streamlit :
```bash
cd /Users/andersoncarbeti/Projet_7_final/Projet_7_final_final/api
source .venv/bin/activate
streamlit run streamlit_app.py
```

URL attendue : `http://localhost:8501`

Reglages Streamlit :
- Base URL: `http://localhost:8000`
- Clients CSV path: `/Users/andersoncarbeti/Projet_7_final/Projet_7_final_final/data/samples/echantillon_clients.csv`

## CI

Le workflow CI est dans `.github/workflows/ci.yml`.

## Data drift

Rapport Evidently : `outputs/reports/evidently/data_drift_train_vs_dataframe_test.html`
