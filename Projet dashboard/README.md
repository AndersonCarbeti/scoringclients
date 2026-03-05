# Projet dashboard - credit scoring

Ce dossier contient uniquement les livrables lies au dashboard et a la veille technique.
Aucun fichier existant du projet principal n est deplace.

## Fichiers

- `app_dashboard.py` : dashboard Streamlit principal.
- `requirements.txt` : dependances du dashboard.
- `veille_technique.md` : feuille de route veille technique.
- `veille_note_modele.md` : modele de note hebdomadaire.
- `deploiement.md` : guide de deploiement dashboard.
- `.streamlit/config.toml` : theme accessible en mode clair.

## Fonctions couvertes

- Score client avec seuil metier et interpretation simple.
- Affichage descriptif du client.
- Comparaison client vs population ou groupe similaire (filtre).
- Analyse bivariee sur 2 variables numeriques.
- Simulation what-if avec rescoring via API.
- Vue proxy d importance locale et globale des variables.

## Lancement local

1. Installation
```bash
cd /Users/andersoncarbeti/Projet_7_final/Projet_7_final_final/scoringclients
python3 -m venv .venv
source .venv/bin/activate
pip install -r api/requirements-dev.txt
pip install -r "Projet dashboard/requirements.txt"
```

2. API (terminal 1)
```bash
cd /Users/andersoncarbeti/Projet_7_final/Projet_7_final_final/scoringclients/api
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

3. Dashboard (terminal 2)
```bash
cd /Users/andersoncarbeti/Projet_7_final/Projet_7_final_final/scoringclients
streamlit run "Projet dashboard/app_dashboard.py"
```

## A noter

- La section "importance locale/globale" est une proxy statistique (pas SHAP natif).
- Si tu veux, on peut brancher SHAP directement depuis le pipeline modele au prochain passage.
