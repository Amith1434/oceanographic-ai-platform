"""
Page 5: Oceanographic Anomaly Detection & Econometric Analysis
=============================================================
Evaluates marine heatwave proxies (P90 / z > 2 SST anomalies) and multivariate
hydrodynamic disruptions (Isolation Forest), resolving temporal confounding via TWFE.
"""

import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Anomaly Analysis - Ocean AI", page_icon="🔥", layout="wide")

OCEANAI_DIR = Path(__file__).resolve().parent.parent.parent
METRICS_DIR = OCEANAI_DIR / "reports" / "metrics"
FIGURES_DIR = OCEANAI_DIR / "reports" / "figures"


@st.cache_data
def load_all_anomaly_metrics():
    f_shares = METRICS_DIR / "anomaly_shares_by_year_region.csv"
    f_within = METRICS_DIR / "anomaly_within_year_region.csv"
    f_twfe = METRICS_DIR / "anomaly_fixed_effects_regression.json"
    f_boot = METRICS_DIR / "anomaly_bootstrap_cis.json"
    
    df_shares = pd.read_csv(f_shares) if f_shares.exists() else None
    df_within = pd.read_csv(f_within) if f_within.exists() else None
    
    if df_within is not None and not df_within.empty:
        # Compute percentage shifts relative to normal months
        df_within["Warm_Diff_Hours_Pct"] = (df_within["Warm_Hours_Ratio"] - 1.0) * 100
        df_within["Iso_Diff_Hours_Pct"] = (df_within["Iso_Hours_Ratio"] - 1.0) * 100
        df_within["Basin"] = df_within["Region"].str.replace("_", " ").str.title()
        
    twfe = None
    if f_twfe.exists():
        with open(f_twfe, "r", encoding="utf-8") as f:
            twfe = json.load(f)
            
    boot = None
    if f_boot.exists():
        with open(f_boot, "r", encoding="utf-8") as f:
            boot = json.load(f)
            
    return df_shares, df_within, twfe, boot


st.title("🔥 Marine Heatwaves, Dynamic Extremes & Fleet Behavior")
st.markdown(r"""
How does apparent commercial fishing activity shift during anomalous oceanographic regimes?
We evaluate two contrasting frameworks:
1. **Univariate Warm Anomalies (Marine Heatwave Proxy):** Cell-months exceeding the 90th percentile SST anomaly (**$+0.528^\circ\text{C}$**) or standardized $z > 2.0$.
2. **Multivariate Dynamic Disruptions:** Unsupervised `IsolationForest` fitted on standardized `[sst_anomaly, so_anomaly, speed]` (5% contamination rate).
""")

# Causal & Resolution Caveat Expander
with st.expander("⚠️ Critical Methodological Caveats on Anomaly Attribution", expanded=True):
    st.markdown(r"""
    - **Observational Correlation, Not Direct Causation:** All cross-domain relationships are observational associations consistent with ecological and physical hypotheses. No controlled clinical experiment is possible in open-ocean fisheries.
    - **Temporal Resolution Limit:** Data are monthly mean averages. Monthly reanalysis cannot resolve transient synoptic weather events, squall lines, or multi-day tropical depressions that keep vessels docked for 48–72 hours.
    - **The Temporal Confounder Discovery:** In naive unconditioned pooling, warm anomalies appear to exhibit +35.9% more fishing hours because warm years (2023, 2024) coincide with post-2022 GFW satellite reception expansions. Controlling for year and cell fixed effects reveals the true modest within-cell marginal effect.
    """)

df_shares, df_within, twfe, boot = load_all_anomaly_metrics()
if df_shares is None or df_within is None:
    st.error("Anomaly metrics not found. Run anomalies.py first.")
    st.stop()

# 1. Two-Way Fixed Effects Regression
st.header("1. Econometric Two-Way Fixed Effects (TWFE) Analysis")
st.markdown(r"""
To eliminate confounding from spatial stationarity (cell-level fixed effects) and macro-temporal satellite constellation expansions (month-year fixed effects), we estimate:
$$\log(1 + \text{hours}_{c, t}) = \alpha_c + \gamma_t + \beta \cdot \text{Anomaly}_{c, t} + \epsilon_{c, t}$$
with standard errors clustered at the grid-cell level ($N = 1,255,968$ cell-months across 14,952 unique cells).
""")

if twfe:
    twfe_rows = []
    for spec_name, s in twfe.items():
        twfe_rows.append({
            "Specification": spec_name,
            "Dependent Variable": s.get("Outcome"),
            "Regressor": s.get("Regressor"),
            "Coefficient (β)": s.get("Coefficient"),
            "Clustered SE": s.get("Std_Error"),
            "t-statistic": s.get("t_statistic"),
            "p-value": f"{s.get('p_value'):.2e}",
            "95% CI Lower": s.get("CI_95_Lower"),
            "95% CI Upper": s.get("CI_95_Upper")
        })
    df_twfe = pd.DataFrame(twfe_rows)
    st.dataframe(df_twfe.style.format({
        "Coefficient (β)": "{:+.4f}",
        "Clustered SE": "{:.4f}",
        "t-statistic": "{:+.2f}",
        "95% CI Lower": "{:+.4f}",
        "95% CI Upper": "{:+.4f}"
    }), use_container_width=True)

st.markdown("---")

# 2. Within-Year Regional Contrast
st.header("2. Within-Year Anomaly Comparisons (Eliminating Satellite Jump Confounding)")
st.markdown(r"""
Comparing anomalous vs normal months **strictly within the same calendar year and basin** demonstrates the true behavioral response:
In **4 out of 7 years**, warm SST anomalies actually exhibited **lower** fishing effort than normal months!
""")

tab_within_tbl, tab_within_chart = st.tabs(["📊 Tabular Within-Year Contrast", "📈 Interactive Trajectory Chart"])

with tab_within_tbl:
    display_cols = {
        "Basin": "Basin",
        "Year": "Year",
        "Warm_Norm_Mean_Hours": "Normal Mean Hrs",
        "Warm_Anom_Mean_Hours": "Warm Anom Mean Hrs",
        "Warm_Hours_Diff": "Hours Diff",
        "Warm_Diff_Hours_Pct": "Effort Shift (%)",
        "Warm_Norm_Fished_Pct": "Normal Fished (%)",
        "Warm_Anom_Fished_Pct": "Warm Fished (%)",
        "Warm_Fished_Diff": "Fished Rate Diff (%)"
    }
    df_disp = df_within[list(display_cols.keys())].rename(columns=display_cols)
    st.dataframe(df_disp.style.format({
        "Normal Mean Hrs": "{:.2f} hrs",
        "Warm Anom Mean Hrs": "{:.2f} hrs",
        "Hours Diff": "{:+.2f} hrs",
        "Effort Shift (%)": "{:+.1f}%",
        "Normal Fished (%)": "{:.2f}%",
        "Warm Fished (%)": "{:.2f}%",
        "Fished Rate Diff (%)": "{:+.2f}%"
    }), use_container_width=True)

with tab_within_chart:
    fig_within = px.bar(
        df_within,
        x="Year",
        y="Warm_Diff_Hours_Pct",
        color="Basin",
        barmode="group",
        title="Within-Year Fishing Effort Shift (%) during Warm SST Anomalies",
        labels={"Warm_Diff_Hours_Pct": "Effort Shift vs Normal Months (%)", "Year": "Calendar Year"}
    )
    fig_within.add_hline(y=0, line_dash="dash", line_color="black")
    st.plotly_chart(fig_within, use_container_width=True)

st.markdown("---")

# 3. 1,000-Rep Month-Block Bootstrap CIs
st.header("3. Month-Block Bootstrap 95% Confidence Intervals")
st.markdown(r"""
To account for temporal autocorrelation across space, we report 1,000-rep month-block bootstrap confidence intervals:
""")

if boot:
    c1, c2, c3, c4 = st.columns(4)
    w_h = boot.get("Warm_Anomaly_Hours_Diff", {})
    w_r = boot.get("Warm_Anomaly_Rate_Diff_Pct", {})
    i_h = boot.get("Iso_Disruption_Hours_Diff", {})
    i_r = boot.get("Iso_Disruption_Rate_Diff_Pct", {})
    
    c1.metric("Warm Hours Shift", f"{w_h.get('Mean', 0):+.2f} hrs", f"95% CI: [{w_h.get('CI_95_Lower', 0):+.2f}, {w_h.get('CI_95_Upper', 0):+.2f}]")
    c2.metric("Warm Fished Rate Shift", f"{w_r.get('Mean', 0):+.2f}%", f"95% CI: [{w_r.get('CI_95_Lower', 0):+.2f}%, {w_r.get('CI_95_Upper', 0):+.2f}%]")
    c3.metric("IsoForest Hours Shift", f"{i_h.get('Mean', 0):+.2f} hrs", f"95% CI: [{i_h.get('CI_95_Lower', 0):+.2f}, {i_h.get('CI_95_Upper', 0):+.2f}]")
    c4.metric("IsoForest Fished Rate Shift", f"{i_r.get('Mean', 0):+.2f}%", f"95% CI: [{i_r.get('CI_95_Lower', 0):+.2f}%, {i_r.get('CI_95_Upper', 0):+.2f}]")

st.markdown("---")

# 4. Diagnostic Figures
st.header("4. Publication Diagnostic Figures")
col_f1, col_f2 = st.columns(2)
if (FIGURES_DIR / "anomaly_shares_by_year.png").exists():
    col_f1.image(str(FIGURES_DIR / "anomaly_shares_by_year.png"), caption="Annual Frequency of Anomaly Regimes by Basin", use_container_width=True)
if (FIGURES_DIR / "sst_anomaly_distribution.png").exists():
    col_f2.image(str(FIGURES_DIR / "sst_anomaly_distribution.png"), caption="Distribution of SST Anomalies and P90 Threshold (+0.528°C)", use_container_width=True)

if (FIGURES_DIR / "anomaly_spatial_frequency.png").exists():
    st.image(str(FIGURES_DIR / "anomaly_spatial_frequency.png"), caption="Spatial Recurrence Frequency of Warm SST and Isolation Forest Anomalies (2019-2025)", use_container_width=True)
