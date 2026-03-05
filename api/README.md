# Pret a depenser — API de credit scoring (FastAPI)

Ce dépôt fournit une **API FastAPI** de scoring crédit, avec :
- chargement du modèle via **MLflow** (URI `models:/...@champion`) ou **fallback** sur un modèle local,
- validation stricte des features par **liste d’input columns**,
- application du **seuil métier** (coût FN/FP) pour décider *APPROVED* vs *REFUSED*,
- **tests** (pytest) et exemples JSON prêts a l’emploi.

> Important : je ne connais pas tes versions exactes de packages.  
> Après installation locale, exécute `pip freeze > requirements.txt` pour figer les versions.

---

## 1) Structure

```
app/
  main.py
  config.py
  model_loader.py
  schemas.py
data/
  input_columns.json
  clients_sample.csv
  threshold.json
tests/
  test_api.py
sample_predict.json
sample_batch.json
outputs/
  models/pipeline_lgbm.joblib
```

---

## 2) Configuration

Variables d’environnement (valeurs par defaut si les fichiers existent) :
- `MODEL_URI=models:/home_credit_default_model@champion`
- `LOCAL_MODEL_PATH=../artifacts/model_champion`
- `INPUT_COLUMNS_PATH=data/input_columns.json`
- `CLIENTS_CSV_PATH=../data/samples/echantillon_clients.csv`
- `CLIENT_ID_COL=SK_ID_CURR`
- `THRESHOLD=0.402` (seuil metier optimise, calcule dans le notebook)
- `FN_COST=10`
- `FP_COST=1`

Note: le seuil metier est **pre-calcule** dans le notebook a partir des couts FN/FP.  
L’API applique simplement ce seuil (pas de recalcul en production).

---

## 3) Lancer en local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 4) Exemples d’appels

```bash
curl -s http://localhost:8000/health

curl -s http://localhost:8000/model-info

curl -s http://localhost:8000/predict \\
  -H 'Content-Type: application/json' \\
  -d @sample_predict.json

curl -s http://localhost:8000/predict-batch \\
  -H 'Content-Type: application/json' \\
  -d @sample_batch.json

curl -s http://localhost:8000/predict-by-id/100001
```

## 4b) Notebook de test API

Notebook minimal : `/Users/andersoncarbeti/Projet_7_final/Projet_7_final_final/api/notebooks/api_test.ipynb`  
Prerequis : `pip install requests`

## 4c) Mini interface Streamlit

Fichier : `/Users/andersoncarbeti/Projet_7_final/Projet_7_final_final/api/streamlit_app.py`  
Installation : `pip install -r requirements-dev.txt`  
Execution :
```bash
streamlit run streamlit_app.py
```

## 5) Tests

```bash
pytest -q
```

## 6) Tests d’integration (modele reel)

Option A (MLflow + model_champion, dependances completes) :
```bash
./scripts/setup_integration_env.sh
RUN_INTEGRATION=1 python -m pytest -q -m integration
```

Option B (modele local .joblib, sans MLflow) :
```bash
RUN_INTEGRATION=1 python -m pytest -q -m integration
```
Note: l’integration .joblib necessite `lightgbm` si le pipeline l’utilise.

## 7) Deploiement (Render / Railway)

Les fichiers de deploiement sont inclus :
- `render.yaml` (racine du repo)
- `railway.json` (racine du repo)
- `api/Dockerfile`

### Render
1. Connecter le repo GitHub a Render.
2. Render detecte `render.yaml` automatiquement.
3. Lancer le deploy puis verifier : `https://<votre-service>.onrender.com/health`

### Railway
1. Connecter le repo GitHub a Railway.
2. Railway utilisera `railway.json` et `api/Dockerfile`.
3. Verifier : `https://<votre-service>.up.railway.app/health`

> Note: l URL finale depend de votre compte/projet Render ou Railway.
