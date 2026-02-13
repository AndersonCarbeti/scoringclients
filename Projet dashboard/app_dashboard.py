from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    px = None
    go = None

LOGGER = logging.getLogger("dashboard_scoring")
logging.basicConfig(level=logging.INFO)

ROOT_DIR = Path(__file__).resolve().parents[1]
URL_API_PAR_DEFAUT = "http://localhost:8000"
CSV_CLIENTS_PAR_DEFAUT = ROOT_DIR / "data" / "samples" / "echantillon_clients.csv"
COLONNE_ID_CLIENT = "SK_ID_CURR"

COULEUR_ACCEPTE = "#1b5e20"
COULEUR_REFUSE = "#b71c1c"
COULEUR_NEUTRE = "#455a64"

# Hypothese minimale: seules ces variables sont modifiables en what-if par un conseiller.
VARIABLES_ACTIONNABLES = {
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "AMT_INCOME_TOTAL",
}

# Exclusion explicite des variables sensibles et non modifiables.
MOTS_SENSIBLES = [
    "GENDER",
    "SEXE",
    "ETHNIC",
    "RACE",
    "ORIGIN",
    "NATIONALITY",
    "RELIGION",
    "FAMILY_STATUS",
    "CODE_GENDER",
]


def _appeler_api_get(url: str, timeout: int = 20) -> dict[str, Any]:
    try:
        reponse = requests.get(url, timeout=timeout)
        reponse.raise_for_status()
        return reponse.json()
    except Exception as exc:  # pragma: no cover
        LOGGER.exception("Erreur appel GET API: %s", url)
        raise RuntimeError("Le service de scoring est indisponible temporairement.") from exc


def _appeler_api_post(url: str, payload: dict[str, Any], timeout: int = 20) -> dict[str, Any]:
    try:
        reponse = requests.post(url, json=payload, timeout=timeout)
        reponse.raise_for_status()
        return reponse.json()
    except Exception as exc:  # pragma: no cover
        LOGGER.exception("Erreur appel POST API: %s", url)
        raise RuntimeError("Le recalcul du score a echoue. Veuillez reessayer.") from exc


@st.cache_data(show_spinner=False)
def charger_clients(chemin_csv: str) -> pd.DataFrame:
    return pd.read_csv(chemin_csv)


@st.cache_data(show_spinner=False)
def charger_info_modele(url_api: str) -> dict[str, Any]:
    return _appeler_api_get(f"{url_api}/model-info")


@st.cache_data(show_spinner=False)
def scorer_par_id(url_api: str, client_id: int) -> dict[str, Any]:
    return _appeler_api_get(f"{url_api}/predict-by-id/{client_id}")


def scorer_par_features(url_api: str, features: dict[str, Any]) -> dict[str, Any]:
    return _appeler_api_post(f"{url_api}/predict", payload={"features": features})


@st.cache_data(show_spinner=False)
def calculer_bornes(df: pd.DataFrame, colonnes: list[str]) -> dict[str, dict[str, float]]:
    bornes: dict[str, dict[str, float]] = {}
    for col in colonnes:
        if col not in df.columns:
            continue
        serie = pd.to_numeric(df[col], errors="coerce").dropna()
        if serie.empty:
            continue
        min_q = float(serie.quantile(0.01))
        max_q = float(serie.quantile(0.99))
        if min_q == max_q:
            min_q = float(serie.min())
            max_q = float(serie.max())
        bornes[col] = {
            "min": min_q,
            "max": max_q,
            "mediane": float(serie.median()),
        }
    return bornes


def libelle_decision(decision_api: str) -> str:
    if decision_api == "APPROVED":
        return "Accepte"
    if decision_api == "REFUSED":
        return "Refuse"
    return "Inconnu"


def couleur_decision(decision_api: str) -> str:
    if decision_api == "APPROVED":
        return COULEUR_ACCEPTE
    if decision_api == "REFUSED":
        return COULEUR_REFUSE
    return COULEUR_NEUTRE


def marge_au_seuil(proba: float, seuil: float) -> float:
    return abs(proba - seuil)


def colonnes_numeriques(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def colonnes_categorielles(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]


def extraire_features_ligne(ligne_client: pd.Series, colonnes_modele: list[str]) -> dict[str, Any]:
    ligne_dict = ligne_client.to_dict()
    features: dict[str, Any] = {}
    for col in colonnes_modele:
        if col in ligne_dict:
            valeur = ligne_dict[col]
            features[col] = None if pd.isna(valeur) else valeur
    return features


def est_sensible(nom_colonne: str) -> bool:
    nom_upper = nom_colonne.upper()
    return any(mot in nom_upper for mot in MOTS_SENSIBLES)


def variables_modifiables(colonnes_modele: list[str], ligne_client: pd.Series) -> list[str]:
    disponibles = []
    for col in colonnes_modele:
        if col not in ligne_client.index:
            continue
        if est_sensible(col):
            continue
        if col not in VARIABLES_ACTIONNABLES:
            continue
        if not pd.api.types.is_numeric_dtype(type(ligne_client[col])):
            try:
                float(ligne_client[col])
            except Exception:
                continue
        disponibles.append(col)
    return sorted(disponibles)


def construire_groupe_comparaison(df: pd.DataFrame, colonne_filtre: str, valeur_filtre: str) -> pd.DataFrame:
    if colonne_filtre == "Aucun filtre" or colonne_filtre not in df.columns:
        return df
    masque = df[colonne_filtre].fillna("MANQUANT").astype(str) == valeur_filtre
    sous_groupe = df[masque]
    return sous_groupe if not sous_groupe.empty else df


def ecarts_client_vs_groupe(df_groupe: pd.DataFrame, ligne_client: pd.Series, top_n: int = 12) -> pd.DataFrame:
    cols_num = [c for c in colonnes_numeriques(df_groupe) if c != COLONNE_ID_CLIENT]
    if not cols_num:
        return pd.DataFrame()

    base = df_groupe[cols_num].replace([np.inf, -np.inf], np.nan)
    moyennes = base.mean(numeric_only=True)
    ecarts_types = base.std(numeric_only=True).replace(0, np.nan)

    valeurs_client = pd.to_numeric(ligne_client[cols_num], errors="coerce")
    z_abs = ((valeurs_client - moyennes) / ecarts_types).abs().fillna(0).sort_values(ascending=False)

    return pd.DataFrame(
        {
            "Variable": z_abs.index,
            "Ecart normalise": z_abs.values,
            "Valeur client": valeurs_client[z_abs.index].values,
            "Moyenne groupe": moyennes[z_abs.index].values,
        }
    ).head(top_n)


def afficher_jauge_score(proba: float, seuil: float, decision_api: str, zone_grise: float) -> None:
    st.subheader("Decision de scoring")

    if go is None:
        st.progress(float(proba))
        st.write(f"Probabilite de defaut: {proba:.2%}")
        st.write(f"Seuil de decision: {seuil:.3f}")
        return

    bas_gris = max(0.0, seuil - zone_grise)
    haut_gris = min(1.0, seuil + zone_grise)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=proba,
            number={"valueformat": ".2%"},
            delta={"reference": seuil, "valueformat": ".2%"},
            title={"text": "Probabilite de defaut"},
            gauge={
                "axis": {"range": [0, 1], "tickformat": ".0%"},
                "bar": {"color": couleur_decision(decision_api)},
                "steps": [
                    {"range": [0, bas_gris], "color": "#d8f3dc"},
                    {"range": [bas_gris, haut_gris], "color": "#e9ecef"},
                    {"range": [haut_gris, 1], "color": "#ffccd5"},
                ],
                "threshold": {
                    "line": {"color": "#111111", "width": 4},
                    "thickness": 0.85,
                    "value": seuil,
                },
            },
        )
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)


def afficher_resume_decision(proba: float, seuil: float, decision_api: str, zone_grise: float) -> None:
    decision = libelle_decision(decision_api)
    marge = marge_au_seuil(proba, seuil)
    revue_manuelle = marge < zone_grise

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Probabilite de defaut",
        f"{proba:.2%}",
        help="Une probabilite n'est pas une certitude.",
    )
    c2.metric("Seuil de decision", f"{seuil:.3f}")
    c3.metric("Decision", decision)
    c4.metric("Marge au seuil", f"{marge:.3f}")

    if revue_manuelle:
        st.warning("Zone grise detectee: recommander une revue manuelle du dossier.")
    elif decision_api == "REFUSED":
        st.error("Risque de defaut eleve selon le modele.")
    else:
        st.success("Risque de defaut faible selon le modele.")

    st.info("Aide: une probabilite n'est pas une certitude. Elle represente un risque estime par le modele.")


def afficher_ecarts_non_causaux(df_groupe: pd.DataFrame, ligne_client: pd.Series) -> None:
    st.subheader("Ecarts client vs groupe de comparaison (non causal)")
    st.caption(
        "Ce bloc n'est PAS une explication du modele. Il montre seulement des ecarts statistiques "
        "entre le client et son groupe de comparaison."
    )

    df_ecarts = ecarts_client_vs_groupe(df_groupe, ligne_client, top_n=12)
    if df_ecarts.empty:
        st.info("Aucune variable numerique disponible pour calculer les ecarts.")
        return

    st.bar_chart(df_ecarts.set_index("Variable")["Ecart normalise"])
    st.dataframe(df_ecarts, use_container_width=True, hide_index=True)

    st.warning("Disclaimer non causal: ces ecarts ne prouvent pas une relation cause-effet.")


def afficher_profil_client(ligne_client: pd.Series, colonnes_affichees: list[str]) -> None:
    st.subheader("Profil client")
    presentes = [c for c in colonnes_affichees if c in ligne_client.index]
    if not presentes:
        st.info("Aucune variable selectionnee pour le profil.")
        return

    tableau = pd.DataFrame({"Variable": presentes, "Valeur": [ligne_client[c] for c in presentes]})
    st.dataframe(tableau, use_container_width=True, hide_index=True)


def afficher_comparaison_univariee(df_groupe: pd.DataFrame, ligne_client: pd.Series, variable: str, top_k_categories: int) -> None:
    st.subheader("Comparaison client vs groupe")

    if variable not in df_groupe.columns:
        st.info("Variable indisponible dans le groupe de comparaison.")
        return

    if pd.api.types.is_numeric_dtype(df_groupe[variable]):
        serie = pd.to_numeric(df_groupe[variable], errors="coerce").dropna()
        if serie.empty:
            st.info("Pas de donnees exploitables pour cette variable.")
            return

        valeur_client = ligne_client.get(variable)
        percentile = np.nan
        if pd.notna(valeur_client):
            percentile = float((serie < valeur_client).mean() * 100)

        if px is not None:
            fig = px.histogram(
                x=serie,
                nbins=30,
                labels={"x": variable, "y": "Nombre de clients"},
                title=f"Distribution de {variable} dans le groupe",
            )
            if pd.notna(valeur_client):
                fig.add_vline(x=float(valeur_client), line_width=3, line_color="#111111")
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(serie)

        if pd.notna(valeur_client):
            st.write(f"Valeur client: `{valeur_client}` | Percentile dans le groupe: `{percentile:.1f}%`")
    else:
        serie_cat = df_groupe[variable].fillna("MANQUANT").astype(str)
        comptes = serie_cat.value_counts()

        top = comptes.head(top_k_categories)
        reste = comptes.iloc[top_k_categories:].sum()
        if reste > 0:
            top.loc["Autres"] = reste

        df_cat = top.rename_axis("Categorie").reset_index(name="Nombre")
        if px is not None:
            fig = px.bar(
                df_cat,
                x="Categorie",
                y="Nombre",
                labels={"Categorie": variable, "Nombre": "Nombre de clients"},
                title=f"Repartition de {variable} (top {top_k_categories})",
            )
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(top)

        st.write(f"Categorie client: `{ligne_client.get(variable, 'MANQUANT')}`")


def afficher_bivariee(df_groupe: pd.DataFrame, ligne_client: pd.Series, x_col: str, y_col: str) -> None:
    st.subheader("Analyse bivariee")

    if x_col not in df_groupe.columns or y_col not in df_groupe.columns:
        st.info("Selection invalide pour l'analyse bivariee.")
        return

    echantillon = df_groupe[[x_col, y_col]].dropna()
    if echantillon.empty:
        st.info("Aucun point disponible pour l'analyse bivariee.")
        return

    if not pd.api.types.is_numeric_dtype(echantillon[x_col]) or not pd.api.types.is_numeric_dtype(echantillon[y_col]):
        st.info("Selectionnez deux variables numeriques.")
        return

    echantillon = echantillon.sample(min(1500, len(echantillon)), random_state=42)
    valeur_x = ligne_client.get(x_col)
    valeur_y = ligne_client.get(y_col)

    if px is not None:
        fig = px.scatter(
            echantillon,
            x=x_col,
            y=y_col,
            opacity=0.5,
            labels={x_col: x_col, y_col: y_col},
            title=f"Nuage de points: {x_col} vs {y_col}",
        )
        if pd.notna(valeur_x) and pd.notna(valeur_y):
            fig.add_scatter(
                x=[float(valeur_x)],
                y=[float(valeur_y)],
                mode="markers",
                marker=dict(size=12, color="#111111"),
                name="Client",
            )
        fig.update_layout(height=390, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.scatter_chart(echantillon, x=x_col, y=y_col)


def afficher_what_if(
    url_api: str,
    ligne_client: pd.Series,
    colonnes_modele: list[str],
    df_clients: pd.DataFrame,
    proba_avant: float,
    decision_avant_api: str,
) -> None:
    st.subheader("Simulation what-if")
    st.caption("Seules les variables actionnables et non sensibles sont modifiables.")

    features_initiales = extraire_features_ligne(ligne_client, colonnes_modele)
    modifiables = variables_modifiables(colonnes_modele, ligne_client)

    if not modifiables:
        st.info("Aucune variable actionnable disponible pour la simulation.")
        return

    bornes = calculer_bornes(df_clients, modifiables)
    choix_par_defaut = modifiables[: min(4, len(modifiables))]
    selection = st.multiselect(
        "Variables a modifier",
        options=modifiables,
        default=choix_par_defaut,
        help="Les variables sensibles (ex: genre, origine) sont exclues automatiquement.",
    )

    with st.form("formulaire_what_if"):
        valeurs_modifiees = features_initiales.copy()
        erreurs: list[str] = []

        for col in selection:
            valeur_initiale = features_initiales.get(col)
            if col not in bornes:
                erreurs.append(f"Bornes indisponibles pour {col}.")
                continue

            borne_min = bornes[col]["min"]
            borne_max = bornes[col]["max"]
            valeur_defaut = float(valeur_initiale) if valeur_initiale is not None else bornes[col]["mediane"]
            pas = max((borne_max - borne_min) / 1000, 0.01)

            valeurs_modifiees[col] = st.number_input(
                f"{col}",
                min_value=float(borne_min),
                max_value=float(borne_max),
                value=float(np.clip(valeur_defaut, borne_min, borne_max)),
                step=float(pas),
                help=f"Intervalle autorise: [{borne_min:.3f}, {borne_max:.3f}]",
            )

        soumettre = st.form_submit_button("Recalculer le score")

    if soumettre:
        if erreurs:
            for err in erreurs:
                st.error(err)
            return

        try:
            pred_apres = scorer_par_features(url_api, valeurs_modifiees)
        except Exception:
            st.error("Le recalcul what-if a echoue. Verifiez les valeurs saisies puis reessayez.")
            return

        proba_apres = float(pred_apres["proba_default"])
        decision_apres_api = str(pred_apres["decision"])

        delta_proba = proba_apres - proba_avant
        decision_avant = libelle_decision(decision_avant_api)
        decision_apres = libelle_decision(decision_apres_api)

        st.success("Simulation terminee")
        c1, c2, c3 = st.columns(3)
        c1.metric("Probabilite avant", f"{proba_avant:.2%}")
        c2.metric("Probabilite apres", f"{proba_apres:.2%}", delta=f"{delta_proba:+.2%}")
        c3.metric("Evolution decision", f"{decision_avant} -> {decision_apres}")

        with st.expander("Voir le detail des resultats"):
            st.json(
                {
                    "avant": {
                        "proba_defaut": round(proba_avant, 6),
                        "decision": decision_avant,
                    },
                    "apres": {
                        "proba_defaut": round(proba_apres, 6),
                        "decision": decision_apres,
                    },
                    "delta_proba": round(delta_proba, 6),
                }
            )


def initialiser_application() -> None:
    st.set_page_config(page_title="Dashboard scoring credit", layout="wide")
    st.title("Pret a depenser - Dashboard de credit scoring")
    st.caption("Presentation du score client pour un public non technique")


def main() -> None:
    initialiser_application()

    with st.sidebar:
        st.header("Navigation")
        section = st.radio(
            "Aller a",
            options=[
                "Synthese decision",
                "Comparaison client",
                "Simulation what-if",
                "Parametres",
            ],
        )

        st.header("Configuration")
        url_api = st.text_input("URL API", value=URL_API_PAR_DEFAUT)
        chemin_csv = st.text_input("Chemin du fichier clients", value=str(CSV_CLIENTS_PAR_DEFAUT))
        zone_grise = st.slider("Largeur zone grise", min_value=0.0, max_value=0.10, value=0.03, step=0.005)
        top_k_categories = st.slider("Nombre max de categories affichees", min_value=5, max_value=30, value=12, step=1)

    try:
        with st.spinner("Chargement des donnees clients..."):
            df_clients = charger_clients(chemin_csv)
    except Exception:
        st.error("Impossible de lire le fichier clients. Verifiez le chemin puis reessayez.")
        return

    if COLONNE_ID_CLIENT not in df_clients.columns:
        st.error(f"La colonne obligatoire {COLONNE_ID_CLIENT} est absente du fichier clients.")
        return

    try:
        with st.spinner("Connexion au service de scoring..."):
            info_modele = charger_info_modele(url_api)
            colonnes_modele = info_modele.get("input_columns") or []
    except Exception:
        st.error("Impossible de recuperer les informations du modele depuis l'API.")
        return

    liste_clients = sorted(df_clients[COLONNE_ID_CLIENT].dropna().astype(int).unique().tolist())
    client_id = st.selectbox("Selection du client", options=liste_clients)

    ligne_client = df_clients[df_clients[COLONNE_ID_CLIENT] == client_id].iloc[0]

    try:
        with st.spinner("Calcul du score client..."):
            prediction = scorer_par_id(url_api, int(client_id))
    except Exception:
        st.error("Impossible de calculer le score du client selectionne.")
        return

    proba = float(prediction["proba_default"])
    seuil = float(prediction["threshold"])
    decision_api = str(prediction["decision"])

    st.markdown("---")

    if section == "Synthese decision":
        afficher_resume_decision(proba, seuil, decision_api, zone_grise)
        afficher_jauge_score(proba, seuil, decision_api, zone_grise)

        colonnes_num = [c for c in colonnes_numeriques(df_clients) if c != COLONNE_ID_CLIENT]
        colonnes_cat = [c for c in colonnes_categorielles(df_clients) if c != COLONNE_ID_CLIENT]
        valeurs_par_defaut = (colonnes_num[:6] + colonnes_cat[:4])[:10]
        selection_profil = st.multiselect(
            "Variables du profil a afficher",
            options=[c for c in df_clients.columns if c != COLONNE_ID_CLIENT],
            default=valeurs_par_defaut,
        )
        afficher_profil_client(ligne_client, selection_profil)

    elif section == "Comparaison client":
        st.subheader("Definition du groupe de comparaison")
        colonnes_cat = [c for c in colonnes_categorielles(df_clients) if c != COLONNE_ID_CLIENT]

        options_filtre = ["Aucun filtre"] + colonnes_cat[:20]
        filtre_col = st.selectbox("Colonne de filtre", options=options_filtre)
        if filtre_col != "Aucun filtre":
            valeurs_filtre = sorted(df_clients[filtre_col].fillna("MANQUANT").astype(str).unique().tolist())
            filtre_val = st.selectbox("Valeur du filtre", options=valeurs_filtre)
        else:
            filtre_val = "TOUS"

        groupe = construire_groupe_comparaison(df_clients, filtre_col, filtre_val)
        st.info(f"Taille du groupe de comparaison: {len(groupe):,} clients")

        cols_num = [c for c in colonnes_numeriques(df_clients) if c != COLONNE_ID_CLIENT]
        cols_cat = [c for c in colonnes_categorielles(df_clients) if c != COLONNE_ID_CLIENT]
        options_comp = cols_num[:80] + cols_cat[:20]
        variable = st.selectbox("Variable a comparer", options=options_comp)

        afficher_comparaison_univariee(groupe, ligne_client, variable, top_k_categories)

        if len(cols_num) >= 2:
            c1, c2 = st.columns(2)
            with c1:
                x_col = st.selectbox("Variable X (bivariee)", options=cols_num, index=0)
            with c2:
                y_col = st.selectbox("Variable Y (bivariee)", options=cols_num, index=min(1, len(cols_num) - 1))
            afficher_bivariee(groupe, ligne_client, x_col, y_col)

        afficher_ecarts_non_causaux(groupe, ligne_client)

    elif section == "Simulation what-if":
        afficher_resume_decision(proba, seuil, decision_api, zone_grise)
        afficher_what_if(
            url_api=url_api,
            ligne_client=ligne_client,
            colonnes_modele=colonnes_modele,
            df_clients=df_clients,
            proba_avant=proba,
            decision_avant_api=decision_api,
        )

    else:
        st.subheader("Parametres d'utilisation")
        st.write("Utilisez le menu de gauche pour naviguer entre les vues du dashboard.")
        st.write("URL API actuelle:", url_api)
        st.write("Fichier clients actuel:", chemin_csv)
        st.write("Zone grise actuelle:", zone_grise)

    st.markdown("---")
    st.caption(
        "Accessibilite: la couleur n'est jamais le seul signal. Les decisions, seuils et marges sont aussi affiches en texte."
    )


if __name__ == "__main__":
    main()
