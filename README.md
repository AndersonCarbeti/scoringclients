# Projet 7 — Prêt à dépenser (Credit Scoring)

Ce dépôt regroupe l’ensemble des livrables pour la soutenance : notebooks, API, artefacts modèle, MLflow, CI/CD, Streamlit et rapports Data Drift Evidently.

---

## Structure du projet

- `notebooks/` : notebooks de modélisation et tests API  
- `data/raw/` : jeux de données originaux  
- `data/processed/` : jeux de données préparés  
- `data/samples/` : échantillons clients pour l’API  
- `artifacts/` : modèle champion + seuil métier + colonnes d’entrée  
- `mlruns/` : tracking MLflow local  
- `outputs/` : rapports SHAP, Evidently et exports  
- `api/` : service FastAPI + tests + Streamlit  
- `.github/workflows/` : CI/CD  
- `slides/` : supports de présentation  

---

## API Locale — Commandes Terminal

```bash
# 1. Aller dans le dossier API
cd /Users/andersoncarbeti/Projet_7_final/Projet_7_final_final/api
```
```bash
# 2. Créer et activer l'environnement virtuel
python3 -m venv .venv
source .venv/bin/activate

# 3. Installer les dépendances dev
pip install -r requirements-dev.txt

# 4. Lancer les tests unitaires
pytest -q

# 5. Lancer l’API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 6. Tester les endpoints
curl -s http://localhost:8000/health
curl -s http://localhost:8000/model-info
curl -s http://localhost:8000/predict -H "Content-Type: application/json" -d @data/sample_predict.json
curl -s http://localhost:8000/predict-batch -H "Content-Type: application/json" -d @data/sample_batch.json
curl -s http://localhost:8000/predict-by-id/100001

# 7. Lancer Streamlit local
streamlit run streamlit_app.py
# URL attendue : http://localhost:8501
# Base URL Streamlit : http://localhost:8000
# CSV clients : data/samples/echantillon_clients.csv
```

---

## Tests API Cloud (Render)

```bash
curl -s https://scoringclients.onrender.com/health
curl -s https://scoringclients.onrender.com/model-info
curl -s https://scoringclients.onrender.com/predict -H "Content-Type: application/json" -d @data/sample_predict.json
curl -s https://scoringclients.onrender.com/predict-batch -H "Content-Type: application/json" -d @data/sample_batch.json
curl -s https://scoringclients.onrender.com/predict-by-id/100001
# URL API : https://scoringclients.onrender.com
```

---

## CI/CD

```bash
# Workflow CI
.github/workflows/ci.yml
# Actions GitHub
https://github.com/AndersonCarbeti/scoringclients/actions
```

---

## MLflow UI (local)

```bash
cd /Users/andersoncarbeti/Projet_7_final/Projet_7_final_final/scoringclients
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
# URL : http://127.0.0.1:5000
# Visualise toutes les expériences et le modèle champion
```

---

## Data Drift Evidently

- Rapport HTML : `outputs/reports/evidently/data_drift_train_vs_dataframe_test.html`  
- Analyse la stabilité des features entre train et test  

Résumé :
- Dataset Drift is NOT detected  
- Threshold drift : 0.5  
- Nombre de colonnes : 236  
- Colonnes drifted : 18 (7.63%)

---

## Démo courte soutenance
1. Montrer API locale : endpoints + tests  
2. Montrer CI/CD et workflow GitHub  
3. Montrer API cloud Render  
4. Montrer MLflow UI et rapport Evidently  
5. Montrer Streamlit pour le scoring client

---

## Notes importantes
- Seuil métier : 0.402  
- Score FN/FP documenté (10:1)  
- Endpoints clés : `/predict`, `/predict-batch`, `/predict-by-id/{client_id}`  
- JSON exemples : `sample_predict.json` et `sample_batch.json`

