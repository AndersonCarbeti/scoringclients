# Checklist soutenance — Projet 7 (Pret a depenser)

## API
- Endpoints a demontrer: `/health`, `/model-info`, `/predict`, `/predict-batch`, `/predict-by-id/{client_id}`
- Fichier principal: `/Users/andersoncarbeti/Projet_7_final/Projet_7_final_final/api/app/main.py`
- Exemples JSON: `/Users/andersoncarbeti/Projet_7_final/Projet_7_final_final/api/sample_predict.json`, `/Users/andersoncarbeti/Projet_7_final/Projet_7_final_final/api/sample_batch.json`
- Colonnes d entree: `/Users/andersoncarbeti/Projet_7_final/Projet_7_final_final/api/data/input_columns.json`
- Seuil final: `/Users/andersoncarbeti/Projet_7_final/Projet_7_final_final/api/data/threshold.json`

## Tests API (local)
```bash
cd /Users/andersoncarbeti/Projet_7_final/Projet_7_final_final/api
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q
```

## Validation endpoints (local)
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
curl -s http://localhost:8000/health
curl -s http://localhost:8000/model-info
curl -s http://localhost:8000/predict -H 'Content-Type: application/json' -d @/Users/andersoncarbeti/Projet_7_final/Projet_7_final_final/api/sample_predict.json
curl -s http://localhost:8000/predict-batch -H 'Content-Type: application/json' -d @/Users/andersoncarbeti/Projet_7_final/Projet_7_final_final/api/sample_batch.json
curl -s http://localhost:8000/predict-by-id/100001
```

## Streamlit (local)
```bash
cd /Users/andersoncarbeti/Projet_7_final/Projet_7_final_final/api
source .venv/bin/activate
streamlit run streamlit_app.py
```
- URL: `http://localhost:8501`
- Base URL: `http://localhost:8000`
- CSV clients: `/Users/andersoncarbeti/Projet_7_final/Projet_7_final_final/data/samples/echantillon_clients.csv`

## Notebook et modele
- Notebook principal: `/Users/andersoncarbeti/Projet_7_final/Projet_7_final_final/notebooks/Carbeti_Anderson_Projet7_Modele_Scoring_Credit_HomeCredit.ipynb`
- A verifier dans le resume: AUC, accuracy, score metier (FN/FP), seuil optimal, SHAP global et local

## MLflow
- Registry local: `/Users/andersoncarbeti/Projet_7_final/Projet_7_final_final/mlruns/models/home_credit_default_model/meta.yaml`
- Alias champion present
- Modele exporte: `/Users/andersoncarbeti/Projet_7_final/Projet_7_final_final/artifacts/model_champion`

## Data drift
- Rapport Evidently: `/Users/andersoncarbeti/Projet_7_final/Projet_7_final_final/outputs/reports/evidently/data_drift_train_vs_dataframe_test.html`

## CI/CD
- CI: `/Users/andersoncarbeti/Projet_7_final/Projet_7_final_final/.github/workflows/ci.yml`
- Deploy: `/Users/andersoncarbeti/Projet_7_final/Projet_7_final_final/.github/workflows/deploy.yml`

## Slides
- Cibles: problematique, modelisation, seuil metier, MLOps, API + Streamlit, drift, limites, Q&A
