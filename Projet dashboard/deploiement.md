# Deploiement du dashboard

## Option 1 - Streamlit Community Cloud

- Avantage: tres rapide pour une demo.
- Limite: dependance au repo public/parametrage secrets.

Etapes:
1. Pousser le repo sur GitHub.
2. Creer une app Streamlit et pointer vers `Projet dashboard/app_dashboard.py`.
3. Configurer `API URL` cible dans l interface.

## Option 2 - Render (Web Service)

- Avantage: plus proche d un usage production.
- Limite: configuration plus complete.

Exemple commande de demarrage:
```bash
streamlit run "Projet dashboard/app_dashboard.py" --server.port $PORT --server.address 0.0.0.0
```

## Configuration minimale

- Variable d environnement recommandee: URL API publique.
- Si API et dashboard sont deployes separement, verifier CORS/API reachability.

## Checklist avant demo

- [ ] URL dashboard accessible.
- [ ] Endpoint API `/health` OK.
- [ ] Prediction par id fonctionne.
- [ ] What-if fonctionne.
- [ ] Contraste et lisibilite verifies.
