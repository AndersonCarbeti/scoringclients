from __future__ import annotations

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

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_CLIENTS_CSV = ROOT_DIR / "data" / "samples" / "echantillon_clients.csv"
APPROVED_COLOR = "#1b5e20"
REFUSED_COLOR = "#b71c1c"
NEUTRAL_COLOR = "#455a64"
CLIENT_ID_COL = "SK_ID_CURR"


# ------------------------- API helpers -------------------------
def _api_get(url: str, timeout: int = 20) -> dict[str, Any]:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _api_post(url: str, payload: dict[str, Any], timeout: int = 20) -> dict[str, Any]:
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


@st.cache_data(show_spinner=False)
def load_clients(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


@st.cache_data(show_spinner=False)
def get_model_info(base_url: str) -> dict[str, Any]:
    return _api_get(f"{base_url}/model-info")


def get_prediction_by_id(base_url: str, client_id: int) -> dict[str, Any]:
    return _api_get(f"{base_url}/predict-by-id/{client_id}")


def get_prediction_from_features(base_url: str, features: dict[str, Any]) -> dict[str, Any]:
    return _api_post(f"{base_url}/predict", payload={"features": features})


# ------------------------- Domain helpers -------------------------
def decision_label(decision: str) -> str:
    if decision == "APPROVED":
        return "Credit likely approved"
    if decision == "REFUSED":
        return "Credit likely refused"
    return "Unknown decision"


def decision_color(decision: str) -> str:
    if decision == "APPROVED":
        return APPROVED_COLOR
    if decision == "REFUSED":
        return REFUSED_COLOR
    return NEUTRAL_COLOR


def compute_distance(proba: float, threshold: float) -> float:
    return abs(proba - threshold)


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]


def to_features(row: pd.Series, input_columns: list[str], client_id_col: str = CLIENT_ID_COL) -> dict[str, Any]:
    row_dict = row.to_dict()
    features: dict[str, Any] = {}
    for col in input_columns:
        if col == client_id_col and col in row_dict:
            features[col] = row_dict[col]
        elif col in row_dict:
            value = row_dict[col]
            features[col] = None if pd.isna(value) else value
    return features


def make_similarity_group(df: pd.DataFrame, filter_col: str, filter_value: str) -> pd.DataFrame:
    if filter_col == "(No filter)" or filter_col not in df.columns:
        return df
    mask = df[filter_col].fillna("MISSING").astype(str) == filter_value
    group = df[mask]
    return group if not group.empty else df


def local_global_importance_proxy(df: pd.DataFrame, client_row: pd.Series, n_top: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    nums = [c for c in numeric_columns(df) if c != CLIENT_ID_COL]
    if not nums:
        return pd.DataFrame(), pd.DataFrame()

    base = df[nums].replace([np.inf, -np.inf], np.nan)
    means = base.mean(numeric_only=True)
    stds = base.std(numeric_only=True).replace(0, np.nan)

    client_vals = pd.to_numeric(client_row[nums], errors="coerce")
    z_abs = ((client_vals - means) / stds).abs().fillna(0).sort_values(ascending=False)

    local_df = pd.DataFrame({
        "feature": z_abs.index,
        "local_impact_proxy": z_abs.values,
        "client_value": client_vals[z_abs.index].values,
        "population_mean": means[z_abs.index].values,
    }).head(n_top)

    global_proxy = stds.fillna(0).sort_values(ascending=False)
    global_df = pd.DataFrame(
        {
            "feature": global_proxy.index,
            "global_impact_proxy": global_proxy.values,
        }
    ).head(n_top)

    return local_df, global_df


# ------------------------- UI blocks -------------------------
def draw_score_gauge(proba: float, threshold: float, decision: str) -> None:
    st.subheader("1) Credit score and threshold")

    if go is None:
        st.progress(float(proba))
        st.write(f"Default probability: {proba:.3f}")
        st.write(f"Threshold: {threshold:.3f}")
        return

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=proba,
            number={"valueformat": ".2%"},
            delta={"reference": threshold},
            title={"text": "Default probability"},
            gauge={
                "axis": {"range": [0, 1], "tickformat": ".0%"},
                "bar": {"color": decision_color(decision)},
                "steps": [
                    {"range": [0, threshold], "color": "#d8f3dc"},
                    {"range": [threshold, 1], "color": "#ffccd5"},
                ],
                "threshold": {
                    "line": {"color": "#111111", "width": 4},
                    "thickness": 0.85,
                    "value": threshold,
                },
            },
        )
    )
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)


def draw_interpretation(proba: float, threshold: float, decision: str) -> None:
    st.subheader("2) Decision interpretation")
    distance = compute_distance(proba, threshold)
    label = decision_label(decision)

    st.markdown(
        f"""
**Decision**: <span style=\"color:{decision_color(decision)}; font-weight:700\">{label}</span>  
**Default probability**: `{proba:.3f}`  
**Distance to threshold**: `{distance:.3f}`  
**Rule applied**: `REFUSED` when `proba_default >= threshold`.
""",
        unsafe_allow_html=True,
    )

    if distance < 0.03:
        st.warning("Near-threshold case: advisor should provide additional context to the client.")
    elif decision == "REFUSED":
        st.error("High default risk estimated by model.")
    else:
        st.success("Low default risk estimated by model.")


def draw_importance_proxy(df: pd.DataFrame, client_row: pd.Series) -> None:
    st.subheader("3) Feature importance (proxy)")
    st.caption(
        "Proxy view: local impact is based on client deviation from population (z-score), "
        "global impact is based on feature spread in population."
    )

    local_df, global_df = local_global_importance_proxy(df, client_row, n_top=12)
    if local_df.empty:
        st.info("No numeric features available for proxy importance.")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Local (client level)**")
        local_chart = local_df.set_index("feature")["local_impact_proxy"]
        st.bar_chart(local_chart)
        st.dataframe(local_df, use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Global (portfolio level)**")
        global_chart = global_df.set_index("feature")["global_impact_proxy"]
        st.bar_chart(global_chart)
        st.dataframe(global_df, use_container_width=True, hide_index=True)


def draw_client_profile(client_row: pd.Series, selected_columns: list[str]) -> None:
    st.subheader("4) Client profile")
    available = [c for c in selected_columns if c in client_row.index]
    if not available:
        st.info("No profile column selected.")
        return

    profile_df = pd.DataFrame({"Feature": available, "Value": [client_row[c] for c in available]})
    st.dataframe(profile_df, use_container_width=True, hide_index=True)


def draw_univariate_comparison(pop_df: pd.DataFrame, client_row: pd.Series, feature: str) -> None:
    st.subheader("5) Client vs comparison group")
    if feature not in pop_df.columns:
        st.info("Feature not available in dataset.")
        return

    if pd.api.types.is_numeric_dtype(pop_df[feature]):
        valid = pop_df[[feature]].dropna()
        if valid.empty:
            st.info("No data available for this feature.")
            return

        if px is not None:
            fig = px.histogram(valid, x=feature, nbins=30, opacity=0.85)
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            hist_vals, bin_edges = np.histogram(valid[feature], bins=30)
            hist_df = pd.DataFrame({"bin_start": bin_edges[:-1], "count": hist_vals})
            st.bar_chart(hist_df.set_index("bin_start"))

        client_value = client_row.get(feature)
        if pd.notna(client_value):
            pct = (valid[feature] < client_value).mean() * 100
            st.write(f"Client value: `{client_value}` | Percentile in comparison group: `{pct:.1f}%`")
    else:
        counts = pop_df[feature].fillna("MISSING").astype(str).value_counts().head(20)
        st.bar_chart(counts)
        client_value = str(client_row.get(feature, "MISSING"))
        st.write(f"Client category: `{client_value}`")


def draw_bivariate(pop_df: pd.DataFrame, client_row: pd.Series, x_col: str, y_col: str) -> None:
    st.subheader("6) Bivariate analysis")
    if x_col not in pop_df.columns or y_col not in pop_df.columns:
        st.info("Invalid feature selection.")
        return

    sample = pop_df[[x_col, y_col]].dropna()
    if sample.empty:
        st.info("No points available for bivariate view.")
        return

    if not pd.api.types.is_numeric_dtype(sample[x_col]) or not pd.api.types.is_numeric_dtype(sample[y_col]):
        st.info("Select two numeric features.")
        return

    sampled = sample.sample(min(1500, len(sample)), random_state=42)
    if px is not None:
        fig = px.scatter(sampled, x=x_col, y=y_col, opacity=0.55)
        client_x = client_row.get(x_col)
        client_y = client_row.get(y_col)
        if pd.notna(client_x) and pd.notna(client_y):
            fig.add_scatter(x=[client_x], y=[client_y], mode="markers", marker=dict(size=12, color="#111111"), name="Client")
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.scatter_chart(sampled, x=x_col, y=y_col)


def draw_rescoring(base_url: str, client_row: pd.Series, input_columns: list[str], editable_cols: list[str]) -> None:
    st.subheader("7) What-if simulation")
    st.caption("Change selected variables and recompute score through API /predict.")

    original = to_features(client_row, input_columns=input_columns)
    editable_available = [c for c in editable_cols if c in original][:40]
    default_edit = editable_available[:8]
    selected_editable = st.multiselect("Editable numeric features", options=editable_available, default=default_edit)

    with st.form("what_if_form"):
        edited = original.copy()
        for col in selected_editable:
            val = edited.get(col)
            if isinstance(val, (int, float, np.number)) and val is not None:
                edited[col] = st.number_input(col, value=float(val), key=f"edit_{col}")
        submitted = st.form_submit_button("Run what-if scoring")

    if submitted:
        try:
            pred = get_prediction_from_features(base_url=base_url, features=edited)
            st.success("What-if scoring complete")
            c1, c2, c3 = st.columns(3)
            c1.metric("New default probability", f"{float(pred['proba_default']):.2%}")
            c2.metric("New decision", str(pred["decision"]))
            c3.metric("Threshold", f"{float(pred['threshold']):.3f}")
            with st.expander("Raw API response"):
                st.json(pred)
        except Exception as exc:
            st.error(f"Rescoring error: {exc}")


def main() -> None:
    st.set_page_config(page_title="Credit scoring dashboard", layout="wide")
    st.title("Pret a depenser - Dashboard credit scoring")
    st.caption("Prototype V1: transparence de decision pour charges de relation client")

    with st.sidebar:
        st.header("Configuration")
        base_url = st.text_input("API URL", value=DEFAULT_API_URL)
        csv_path = st.text_input("Clients CSV", value=str(DEFAULT_CLIENTS_CSV))

    try:
        df_clients = load_clients(csv_path)
    except Exception as exc:
        st.error(f"Cannot load clients CSV: {exc}")
        st.stop()

    if CLIENT_ID_COL not in df_clients.columns:
        st.error(f"Required ID column missing: {CLIENT_ID_COL}")
        st.stop()

    try:
        model_info = get_model_info(base_url)
        input_columns = model_info.get("input_columns") or []
    except Exception as exc:
        st.error(f"Cannot load API model info: {exc}")
        st.stop()

    client_ids = sorted(df_clients[CLIENT_ID_COL].dropna().astype(int).unique().tolist())
    selected_id = st.selectbox("Client ID", options=client_ids)
    client_row = df_clients[df_clients[CLIENT_ID_COL] == selected_id].iloc[0]

    try:
        pred = get_prediction_by_id(base_url, int(selected_id))
    except Exception as exc:
        st.error(f"Cannot score selected client: {exc}")
        st.stop()

    proba = float(pred["proba_default"])
    threshold = float(pred["threshold"])
    decision = str(pred["decision"])

    top_left, top_right = st.columns(2)
    with top_left:
        draw_score_gauge(proba=proba, threshold=threshold, decision=decision)
    with top_right:
        draw_interpretation(proba=proba, threshold=threshold, decision=decision)

    draw_importance_proxy(df=df_clients, client_row=client_row)

    nums = [c for c in numeric_columns(df_clients) if c != CLIENT_ID_COL]
    cats = [c for c in categorical_columns(df_clients) if c != CLIENT_ID_COL]

    profile_defaults = (nums[:6] + cats[:4])[:10]
    profile_cols = st.multiselect(
        "Profile variables",
        options=[c for c in df_clients.columns if c != CLIENT_ID_COL],
        default=profile_defaults,
    )
    draw_client_profile(client_row=client_row, selected_columns=profile_cols)

    st.markdown("---")
    st.subheader("Comparison group filter")
    filter_options = ["(No filter)"] + cats[:20]
    group_col = st.selectbox("Filter column", options=filter_options)
    if group_col != "(No filter)":
        values = sorted(df_clients[group_col].fillna("MISSING").astype(str).unique().tolist())
        group_value = st.selectbox("Filter value", options=values)
    else:
        group_value = "ALL"

    group_df = make_similarity_group(df_clients, group_col, group_value)
    st.caption(f"Comparison group size: {len(group_df):,} clients")

    feature_options = nums[:80] + cats[:20]
    if feature_options:
        uni_feature = st.selectbox("Feature for comparison", options=feature_options)
        draw_univariate_comparison(pop_df=group_df, client_row=client_row, feature=uni_feature)

    if len(nums) >= 2:
        c1, c2 = st.columns(2)
        with c1:
            x_col = st.selectbox("Bivariate X", options=nums, index=0)
        with c2:
            y_col = st.selectbox("Bivariate Y", options=nums, index=min(1, len(nums) - 1))
        draw_bivariate(pop_df=group_df, client_row=client_row, x_col=x_col, y_col=y_col)

    draw_rescoring(base_url=base_url, client_row=client_row, input_columns=input_columns, editable_cols=nums)

    st.markdown("---")
    st.caption(
        "Accessibility note: color is never the only carrier of information; values, thresholds and labels are displayed in text. "
        "For full WCAG compliance, include keyboard-only tests and contrast audits in QA."
    )


if __name__ == "__main__":
    main()
