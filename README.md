# -----------------------------------------------------------
# 1️⃣ Préparer l'environnement API local
# -----------------------------------------------------------
cd /Users/andersoncarbeti/Projet_7_final/Projet_7_final_final/api
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

# -----------------------------------------------------------
# 2️⃣ Lancer les tests unitaires
# -----------------------------------------------------------
pytest -q

# -----------------------------------------------------------
# 3️⃣ Lancer l’API localement
# -----------------------------------------------------------
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# -----------------------------------------------------------
# 4️⃣ Tester les endpoints API locaux
# -----------------------------------------------------------
curl -s http://localhost:8000/health
curl -s http://localhost:8000/model-info
curl -s http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d @/Users/andersoncarbeti/Projet_7_final/Projet_7_final_final/api/data/sample_predict.json
curl -s http://localhost:8000/predict-batch \
     -H "Content-Type: application/json" \
     -d @/Users/andersoncarbeti/Projet_7_final/Projet_7_final_final/api/data/sample_batch.json
curl -s http://localhost:8000/predict-by-id/100001

# -----------------------------------------------------------
# 5️⃣ Lancer Streamlit local pour tester l’API
# -----------------------------------------------------------
cd /Users/andersoncarbeti/Projet_7_final/Projet_7_final_final/api
source .venv/bin/activate
streamlit run streamlit_app.py

# URL attendue : http://localhost:8501
# Base URL Streamlit : http://localhost:8000
# CSV clients : /Users/andersoncarbeti/Projet_7_final/Projet_7_final_final/data/samples/echantillon_clients.csv

# -----------------------------------------------------------
# 6️⃣ Tests API Cloud (Render)
# -----------------------------------------------------------
curl -s https://scoringclients.onrender.com/health
curl -s https://scoringclients.onrender.com/model-info
curl -s https://scoringclients.onrender.com/predict \
     -H "Content-Type: application/json" \
     -d @/Users/andersoncarbeti/Projet_7_final/Projet_7_final_final/api/data/sample_predict.json
curl -s https://scoringclients.onrender.com/predict-batch \
     -H "Content-Type: application/json" \
     -d @/Users/andersoncarbeti/Projet_7_final/Projet_7_final_final/api/data/sample_batch.json
curl -s https://scoringclients.onrender.com/predict-by-id/100001

# URL API cloud : https://scoringclients.onrender.com

# -----------------------------------------------------------
# 7️⃣ CI/CD
# -----------------------------------------------------------
# Workflow CI : .github/workflows/ci.yml
# Actions GitHub : https://github.com/AndersonCarbeti/scoringclients/actions

# -----------------------------------------------------------
# 8️⃣ MLflow UI (local)
# -----------------------------------------------------------
cd /Users/andersoncarbeti/Projet_7_final/Projet_7_final_final/scoringclients
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000

# URL : http://127.0.0.1:5000
# Montre toutes les expériences et le modèle champion

# -----------------------------------------------------------
# 9️⃣ Data Drift Evidently
# -----------------------------------------------------------
# Rapport HTML : outputs/reports/evidently/data_drift_train_vs_dataframe_test.html
# Analyse la stabilité des features entre train et test
# Résumé :
#   - Dataset Drift is NOT detected
#   - Threshold drift : 0.5
#   - Nombre de colonnes : 236
#   - Colonnes drifted : 18 (7.63%)

# -----------------------------------------------------------
# 10️⃣ Démo courte soutenance
# -----------------------------------------------------------
# 1. API locale : endpoints + tests
# 2. CI/CD : workflow GitHub Actions
# 3. API Cloud Render
# 4. MLflow UI + rapport Evidently
# 5. Streamlit : scoring client

# -----------------------------------------------------------
# Notes importantes
# -----------------------------------------------------------
# - Seuil métier : 0.402
# - Score FN/FP : 10:1
# - Endpoints clés : /predict, /predict-batch, /predict-by-id/{client_id}
# - JSON exemples : sample_predict.json, sample_batch.json
