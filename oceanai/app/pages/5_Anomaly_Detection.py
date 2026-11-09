"""
Page 5: Marine Heatwaves & Anomaly Detection
============================================
Evaluates how commercial fishing effort responds to anomalous oceanographic regimes.
"""

from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Anomalies - Ocean AI", page_icon="🔥", layout="wide")

OCEANAI_DIR = Path(__file__).resolve().parent.parent.parent
METRICS_DIR = OCEANAI_DIR / "reports" / "metrics"
FIGURES_DIR = OCEANAI_DIR / "reports" / "figures"


@st.cache_data
def load_anomaly_metrics():
    f1 = METRICS_DIR / "anomaly_fishing_impact.csv"
    f2 = METRICS_DIR / "regional_anomaly_impact.csv"
    s1 = pd.read_csv(f1) if f1.exists() else None
    s2 = pd.read_csv(f2) if f2.exists() else None
    return s1, s2


st.title("🔥 Marine Heatwaves & Ocean Anomaly Detection")
st.markdown("""
### Cross-Domain Analytical Objective
> **How does apparent commercial fishing activity shift during anomalous ocean conditions (marine warm anomalies vs multivariate dynamic disruptions)?**

We test two contrasting anomaly frameworks:
1. **Univariate Warm Anomalies (Marine Heatwave Proxy):** Cell-months exceeding the 90th percentile SST anomaly (**$+0.528^\circ\text{C}$**) or standardized $z > 2.0$.
2. **Multivariate Dynamic Disruptions:** Unsupervised `IsolationForest` fitted on standardized `[sst_anomaly, so_anomaly, speed]` (5% contamination rate).
""")

s1, s2 = load_anomaly_metrics()
if s1 is None or s2 is None:
    st.error("Anomaly metrics not found. Run anomalies.py first.")
    st.stop()

st.header("1. Cross-Domain Shift in Apparent Fishing Effort")
st.dataframe(s1.style.format({
    "Total_Cell_Months": "{:,}",
    "Fished_Cells_Pct": "{:.2f}%",
    "Mean_Hours_All": "{:.2f} hrs",
    "Median_Hours_All": "{:.1f} hrs",
    "Mean_Hours_Fished_Only": "{:.2f} hrs",
    "Total_Hours": "{:,.0f} hrs"
}))

st.markdown("---")

st.header("2. Two Contrasting Fleet Behavioral Regimes")
col_w, col_i = st.columns(2)

with col_w:
    st.subheader("☀️ Regime 1: Warm SST Anomalies")
    st.markdown("""
    - **Fished Occurrence Rate:** Jumps from **20.50% to 26.12%** (+5.6 percentage points).
    - **Mean Apparent Effort:** Surges from **11.32 to 15.39 hours/cell** (+35.9% increase).
    - **Significance:** Mann-Whitney U test $p < 10^{-15}$.
    - **Physical Mechanism:** Warmer surface temperatures compress pelagic fish shoals (e.g., sardines, tuna, mackerel) into localized upwelling thermal fronts, attracting intensified commercial search and harvesting effort.
    """)

with col_i:
    st.subheader("🌪️ Regime 2: Multivariate Dynamic Disruptions")
    st.markdown("""
    - **Fished Occurrence Rate:** Drops from **21.36% to 15.43%** (-6.0 percentage points).
    - **Mean Apparent Effort:** Plunges from **11.96 to 7.40 hours/cell** (-38.1% drop).
    - **Physical Mechanism:** Compound anomalies (strong current shear combined with extreme freshwater flood pulses and anomalous temperatures) generate severe sea states and navigation hazards that suppress vessel departures from port.
    """)

st.markdown("---")

st.header("3. Regional Breakdown (Arabian Sea vs Bay of Bengal)")
st.dataframe(s2.style.format({
    "Cell_Months": "{:,}",
    "Fished_Pct": "{:.2f}%",
    "Mean_Hours": "{:.2f} hrs",
    "Total_Hours": "{:,.0f} hrs"
}))

st.subheader("Publication Visualizations")
c1, c2 = st.columns(2)
if (FIGURES_DIR / "sst_anomaly_distribution.png").exists():
    c1.image(str(FIGURES_DIR / "sst_anomaly_distribution.png"), caption="Distribution of SST Anomalies and 90th Percentile Threshold", use_container_width=True)
if (FIGURES_DIR / "fishing_hours_anomaly_comparison.png").exists():
    c2.image(str(FIGURES_DIR / "fishing_hours_anomaly_comparison.png"), caption="Fishing Effort Distribution in Normal vs Warm Anomalous Months", use_container_width=True)
if (FIGURES_DIR / "anomaly_spatial_frequency.png").exists():
    st.image(str(FIGURES_DIR / "anomaly_spatial_frequency.png"), caption="Spatial Frequency of Marine Warm Anomalies (2019-2025)", use_container_width=True)
