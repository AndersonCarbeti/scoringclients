# Veille technique - mission dashboard credit scoring

## 1) Objectif

Mettre en place une veille actionnable pour maintenir un dashboard:
- compréhensible pour non experts,
- conforme aux exigences d accessibilite,
- deployable et maintenable en production,
- aligne avec les bonnes pratiques d explicabilite IA.

## 2) Axes de veille prioritaires

1. Explicabilite
- SHAP local/global pour expliquer chaque decision.
- Limites et risques d interpretation pour le front-office.

2. Accessibilite (WCAG 2.2 AA)
- Contraste des couleurs, focus clavier, labels explicites.
- Alternatives textuelles aux graphiques.

3. Dashboard engineering
- Streamlit vs Dash vs Bokeh: compromis developpement/maintenabilite.
- Patterns de separation UI/API et gestion des erreurs.

4. MLOps
- Data drift / prediction drift.
- Observabilite API (latence, erreurs) et audit des decisions.

## 3) Sources de reference

- W3C WCAG 2.2.
- Documentation Streamlit, Dash, Bokeh.
- Documentation SHAP.
- Documentation MLflow et Evidently.
- FastAPI production best practices.

## 4) Rythme

- Point hebdomadaire: 30-45 minutes.
- Sortie: 1 note de veille courte + 1 action decisee.

## 5) Format de sortie attendu

- Sujet veille
- Date
- 3 sources max
- Impact projet (2-4 lignes)
- Decision (Go / No-Go)
- Action concrete (owner + deadline)

## 6) Backlog initial

- [ ] Integrer SHAP dans le dashboard (local + global).
- [ ] Rediger checklist WCAG specifique aux graphiques du projet.
- [ ] Definir SLO API (latence p95, taux d erreur).
- [ ] Ajouter suivi drift et periodicite de revue.
- [ ] Formaliser runbook incident dashboard/API.
