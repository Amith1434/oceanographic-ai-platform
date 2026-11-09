"""
Page 4: Feature Importance & Native TreeSHAP Analysis
=====================================================
Explains decision drivers using exact TreeSHAP attributions and permutation drops.
"""

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
    imp_csv = METRICS_DIR / "feature_importance.csv"
    if imp_csv.exists():
        return pd.read_csv(imp_csv)
    return None


st.title("📊 Feature Attribution & Native TreeSHAP Analysis")
st.markdown("""
To understand how individual features contribute to predictions in **Model 2 (Full Model)**, we evaluate:
1. **Native TreeSHAP Values:** Exact Shapley attributions computed directly via XGBoost's C++ booster (`pred_contribs=True`).
2. **Permutation Importance:** Drop in test set ROC-AUC when each feature is independently shuffled.
3. **Gain Importance:** Relative improvement in tree node purity contributed by each feature.
""")

df_imp = load_feature_importance()
if df_imp is None:
    st.error("Feature importance data not found. Run train_models.py first.")
    st.stop()

# Classify categories
spatial_cols = ["lat_bin", "lon_bin", "region_id", "dist_to_coast_km", "sin_month", "cos_month"]
df_imp["Domain"] = df_imp["Feature"].apply(lambda x: "Spatial / Seasonal" if x in spatial_cols else "Ocean Physical")

st.subheader("TreeSHAP and Permutation Importance Ranking")
st.dataframe(df_imp.style.format({
    "Mean_Abs_SHAP": "{:.4f}",
    "Permutation_ROC_AUC_Drop": "{:.4f}",
    "StageA_Gain_Pct": "{:.2f}%",
    "StageB_Gain_Pct": "{:.2f}%"
}))

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    fig_bar = px.bar(
        df_imp.sort_values("Mean_Abs_SHAP", ascending=True),
        x="Mean_Abs_SHAP",
        y="Feature",
        color="Domain",
        orientation="h",
        color_discrete_map={"Spatial / Seasonal": "#1f77b4", "Ocean Physical": "#2ca02c"},
        title="Mean |TreeSHAP Value| (Impact on Stage A Occurrence)",
        labels={"Mean_Abs_SHAP": "Mean |SHAP| (Log-Odds Impact)", "Feature": "Predictor"}
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    fig_perm = px.bar(
        df_imp.sort_values("Permutation_ROC_AUC_Drop", ascending=True),
        x="Permutation_ROC_AUC_Drop",
        y="Feature",
        color="Domain",
        orientation="h",
        color_discrete_map={"Spatial / Seasonal": "#1f77b4", "Ocean Physical": "#2ca02c"},
        title="Permutation Importance (ROC-AUC Drop on 2025 Test Set)",
        labels={"Permutation_ROC_AUC_Drop": "Test ROC-AUC Drop", "Feature": "Predictor"}
    )
    st.plotly_chart(fig_perm, use_container_width=True)

st.markdown("""
### Analytical Summary:
- **Spatial Coordinates Dominance:** `lon_bin` and `lat_bin` dominate TreeSHAP and permutation drops. Shuffling `lon_bin` collapses model ROC-AUC by **+0.1759**, confirming that spatial position is the overriding predictive factor.
- **Vessel Proximity Factor:** `dist_to_coast_km` is the 2nd most critical feature, reflecting fuel economics, operational vessel range, and the concentration of trawlers along shallow coastal shelf breaks.
- **Monsoon Cycles:** `sin_month` and `cos_month` account for ~20% of decision splits, capturing the annual monsoon closure seasons.
- **Ocean Physics:** Individual physical features (SST, salinity, currents, anomalies) exhibit small permutation drops (<0.013), indicating they modulate predictions locally but do not dictate general fleet distribution.
""")

st.subheader("Publication Diagnostic Plots")
im_c1, im_c2 = st.columns(2)
if (FIGURES_DIR / "shap_summary_stage_a.png").exists():
    im_c1.image(str(FIGURES_DIR / "shap_summary_stage_a.png"), caption="TreeSHAP Feature Attribution Bar Chart", use_container_width=True)
if (FIGURES_DIR / "feature_importance_comparison.png").exists():
    im_c2.image(str(FIGURES_DIR / "feature_importance_comparison.png"), caption="XGBoost Relative Gain: Stage A vs Stage B", use_container_width=True)
