"""
Page 3: Scientific Ablation Study
=================================
Rigorous 5-model ablation experiment across Pooled, Arabian Sea, and Bay of Bengal panels
with month-block bootstrap 95% confidence intervals on delta metrics.
"""

import json
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Ablation Study - Ocean AI", page_icon="⚖️", layout="wide")

OCEANAI_DIR = Path(__file__).resolve().parent.parent.parent
METRICS_DIR = OCEANAI_DIR / "reports" / "metrics"
FIGURES_DIR = OCEANAI_DIR / "reports" / "figures"


@st.cache_data
def load_ablation_data():
    f_csv = METRICS_DIR / "ablation_study_expanded.csv"
    f_json = METRICS_DIR / "ablation_bootstrap_cis.json"
    
    df_abl = pd.read_csv(f_csv) if f_csv.exists() else None
    cis = None
    if f_json.exists():
        with open(f_json, "r", encoding="utf-8") as f:
            cis = json.load(f)
    return df_abl, cis


st.title("⚖️ The Scientific Ablation Study: Does Ocean Physics Add Value?")
st.markdown(r"""
### The Central Scientific Hypothesis
> **Do physical oceanographic conditions (sea temperature, salinity, currents, anomalies, lags) help predict where and how much apparent commercial fishing happens, beyond spatial position and seasonal monsoons?**

To answer this conclusively, we evaluated a 5-model ablation hierarchy across three geographic panels on out-of-time test data (**2025**):
- **M1 (Spatial + Seasonal, 6 feats):** `lat_bin`, `lon_bin`, `region_id`, `dist_to_coast_km`, `sin_month`, `cos_month`.
- **M2 (Spatial + Seasonal + Ocean, 16 feats):** M1 + `thetao`, `so`, `uo`, `vo`, `speed`, `sst_anomaly`, `so_anomaly`, `sst_lag1`, `so_lag1`, `speed_lag1`.
- **M2b (Spatial + Seasonal + Anomalies, 8 feats):** M1 + `sst_anomaly`, `so_anomaly`.
- **M3 (Spatial + Seasonal + Persistence, 8 feats):** M1 + `hours_lag1`, `is_fishing_lag1`.
- **M4 (Full + Persistence + Ocean, 18 feats):** M3 + All Ocean Variables.
""")

df_abl, cis = load_ablation_data()
if df_abl is None:
    st.error("Ablation study results not found. Run train_models.py first.")
    st.stop()

# 1. Expanded Ablation Table
st.header("1. Out-of-Time Test (2025) Performance Across Geographic Panels")

panel_choice = st.selectbox(
    "Filter by Geographic Basin:",
    ["All Panels", "Pooled (All Basins)", "Arabian Sea", "Bay of Bengal"],
    index=0
)

view_df = df_abl.copy()
if panel_choice != "All Panels":
    view_df = view_df[view_df["Panel"] == panel_choice].copy()

st.dataframe(view_df.style.format({
    "Num_Features": "{:d}",
    "StageA_Test_ROC_AUC": "{:.4f}",
    "StageA_Test_PR_AUC": "{:.4f}",
    "StageA_Test_F1": "{:.4f}",
    "StageA_Val_Threshold": "{:.2f}",
    "StageB_Test_R2": "{:.4f}",
    "StageB_Test_RMSE": "{:.4f}",
    "StageB_Test_Spearman": "{:.4f}"
}))

st.markdown("---")

# 2. Month-Block Bootstrap Deltas
st.header("2. Rigorous Month-Block Bootstrap Confidence Intervals (M2 − M1)")
st.markdown(r"""
To evaluate whether the difference between Model 2 (+Ocean) and Model 1 (Base) is statistically meaningful,
we computed **1,000 month-block bootstrap iterations** on the 2025 test set:
""")

if cis:
    roc_ci = cis.get("Delta_ROC_AUC_M2_minus_M1", {})
    r2_ci = cis.get("Delta_R2_StageB_M2_minus_M1", {})
    
    c1, c2 = st.columns(2)
    c1.metric(
        "Stage A Δ ROC-AUC (M2 − M1)",
        f"{roc_ci.get('Mean', 0):+.4f}",
        f"95% CI: [{roc_ci.get('CI_95_Lower', 0):+.4f}, {roc_ci.get('CI_95_Upper', 0):+.4f}]",
        delta_color="off"
    )
    c2.metric(
        "Stage B Δ R² (M2 − M1)",
        f"{r2_ci.get('Mean', 0):+.4f}",
        f"95% CI: [{r2_ci.get('CI_95_Lower', 0):+.4f}, {r2_ci.get('CI_95_Upper', 0):+.4f}]",
        delta_color="off"
    )

st.markdown("---")

# 3. Mechanistic Synthesis
st.header("3. Mechanistic Explanation for the Null / Marginal Finding")
st.info(r"""
### Why Dynamic Ocean Physics Adds Negligible Out-of-Time Predictive Value:
1. **Sovereign EEZs & Continental Shelf Bathymetry:** Commercial fishing fleets are constrained by maritime boundaries, exclusive economic zones, and distance to landing ports. Vessel range and ice-hold duration govern spatial presence far more than surface temperature.
2. **Statutory Monsoon Trawl Bans:** India enforces uniform 47–61 day mechanized trawl bans (June–July on the West coast, April–May on the East coast). Seasonal harmonics ($\sin, \cos$) completely capture this regulatory calendar.
3. **Collinearity & Overfitting:** Sea surface temperature and salinity co-vary strongly with latitude, longitude, and calendar month. Adding 10 collinear features increases model complexity without introducing novel orthogonal variance, causing minor out-of-time test degradation.
4. **Honest Scientific Reporting:** Transparently reporting this marginal/null finding is a scientifically robust result. It demonstrates that commercial fishing is governed by human institutions and geography rather than passive oceanographic drift.
""")

st.subheader("Publication Diagnostic Visualizations")
fig_ablation = FIGURES_DIR / "ablation_roc_pr_curves.png"
if fig_ablation.exists():
    st.image(str(fig_ablation), caption="Stage A ROC and Precision-Recall Curves (Model 1 vs Model 2 on 2025 Test Set)", use_container_width=True)
