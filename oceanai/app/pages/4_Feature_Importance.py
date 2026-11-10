"""
Page 4: Feature Attribution & Native TreeSHAP Analysis
=====================================================
Explains decision drivers across ALL 16 features using exact TreeSHAP attributions,
permutation drops, split gain, and grouped category shares.
"""

import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Feature Importance - Ocean AI", page_icon="📊", layout="wide")

OCEANAI_DIR = Path(__file__).resolve().parent.parent.parent
METRICS_DIR = OCEANAI_DIR / "reports" / "metrics"
FIGURES_DIR = OCEANAI_DIR / "reports" / "figures"


@st.cache_data
def load_feature_importance():
    f_csv = METRICS_DIR / "feature_importance_stage_a_m2.csv"
    f_json = METRICS_DIR / "feature_importance_grouped_shares.json"
    
    # Fallback to feature_importance.csv if stage_a_m2 not yet finished
    if not f_csv.exists():
        f_csv = METRICS_DIR / "feature_importance.csv"
        
    df_imp = pd.read_csv(f_csv) if f_csv.exists() else None
    grouped = None
    if f_json.exists():
        with open(f_json, "r", encoding="utf-8") as f:
            grouped = json.load(f)
            
    return df_imp, grouped


st.title("📊 Feature Attribution & Native TreeSHAP Analysis")
st.markdown("""
To dissect model decision pathways for **Model 2 (16 Features)**, we evaluate three complementary explainability paradigms:
1. **XGBoost Split Gain:** Normalised improvement in tree objective contributed by each feature split.
2. **Out-of-Time Test Permutation Importance:** Drop in test set ROC-AUC when each feature is independently shuffled (5 repeats).
3. **Native TreeSHAP Values:** Exact Shapley attributions computed directly via XGBoost's C++ booster (`pred_contribs=True`).
""")

df_imp, grouped = load_feature_importance()
if df_imp is None:
    st.error("Feature importance data not found. Run train_models.py first.")
    st.stop()

# 1. Grouped Shares
st.header("1. Grouped Category Predictive Shares")
if grouped:
    g_rows = []
    g_rows.append({
        "Category": "Spatial Geography",
        "Features Included": ", ".join(f"`{f}`" for f in grouped["Spatial_Features"]["Features"]),
        "Count": grouped["Spatial_Features"]["Count"],
        "Split Gain Share": f"{grouped['Spatial_Features']['Split_Gain_Share_Pct']:.2f}%",
        "Permutation Drop Share": f"{grouped['Spatial_Features']['Permutation_Drop_Share_Pct']:.2f}%",
        "TreeSHAP Share": f"{grouped['Spatial_Features']['TreeSHAP_Share_Pct']:.2f}%"
    })
    g_rows.append({
        "Category": "Monsoon Seasonality",
        "Features Included": ", ".join(f"`{f}`" for f in grouped["Seasonal_Features"]["Features"]),
        "Count": grouped["Seasonal_Features"]["Count"],
        "Split Gain Share": f"{grouped['Seasonal_Features']['Split_Gain_Share_Pct']:.2f}%",
        "Permutation Drop Share": f"{grouped['Seasonal_Features']['Permutation_Drop_Share_Pct']:.2f}%",
        "TreeSHAP Share": f"{grouped['Seasonal_Features']['TreeSHAP_Share_Pct']:.2f}%"
    })
    g_rows.append({
        "Category": "Dynamic Ocean Physics",
        "Features Included": ", ".join(f"`{f}`" for f in grouped["Ocean_Physical_Features"]["Features"]),
        "Count": grouped["Ocean_Physical_Features"]["Count"],
        "Split Gain Share": f"{grouped['Ocean_Physical_Features']['Split_Gain_Share_Pct']:.2f}%",
        "Permutation Drop Share": f"{grouped['Ocean_Physical_Features']['Permutation_Drop_Share_Pct']:.2f}%",
        "TreeSHAP Share": f"{grouped['Ocean_Physical_Features']['TreeSHAP_Share_Pct']:.2f}%"
    })
    df_g = pd.DataFrame(g_rows)
    st.dataframe(df_g)

st.markdown("---")

# 2. Complete 16-Feature Importance Table
st.header("2. Complete 16-Feature Importance Ranking")
st.dataframe(df_imp.style.format({
    "Normalized_Split_Gain": "{:.4f}",
    "Permutation_Drop_AUC": "{:.4f}",
    "Normalized_Permutation": "{:.4f}",
    "Mean_Abs_TreeSHAP": "{:.4f}",
    "Normalized_TreeSHAP": "{:.4f}"
}))

st.markdown("---")

# 3. Interactive Visualizations
st.header("3. Tripartite Explainability Comparison")
c1, c2 = st.columns(2)

with c1:
    fig_shap = px.bar(
        df_imp.sort_values("Normalized_TreeSHAP", ascending=True),
        x="Normalized_TreeSHAP",
        y="Feature",
        orientation="h",
        title="Mean |TreeSHAP| Attributions (Normalized Share)",
        color="Normalized_TreeSHAP",
        color_continuous_scale="Viridis",
        labels={"Normalized_TreeSHAP": "TreeSHAP Share", "Feature": "Feature"}
    )
    st.plotly_chart(fig_shap, use_container_width=True)

with c2:
    fig_gain = px.bar(
        df_imp.sort_values("Normalized_Split_Gain", ascending=True),
        x="Normalized_Split_Gain",
        y="Feature",
        orientation="h",
        title="XGBoost Split Gain (Normalized Share)",
        color="Normalized_Split_Gain",
        color_continuous_scale="Cividis",
        labels={"Normalized_Split_Gain": "Split Gain Share", "Feature": "Feature"}
    )
    st.plotly_chart(fig_gain, use_container_width=True)

st.subheader("Publication Diagnostic Visualizations")
c_p1, c_p2 = st.columns(2)
if (FIGURES_DIR / "shap_summary_stage_a.png").exists():
    c_p1.image(str(FIGURES_DIR / "shap_summary_stage_a.png"), caption="TreeSHAP Feature Attributions Bar Chart", use_container_width=True)
if (FIGURES_DIR / "feature_importance_comparison.png").exists():
    c_p2.image(str(FIGURES_DIR / "feature_importance_comparison.png"), caption="Tripartite Comparison: Gain vs Permutation vs TreeSHAP", use_container_width=True)
