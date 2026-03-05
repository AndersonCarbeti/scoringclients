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

## API Locale

### 1. Préparer l'environnement
```bash
cd /Users/andersoncarbeti/Projet_7_final/Projet_7_final_final/api
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
