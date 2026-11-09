"""
Oceanographic AI Platform - Home & Executive Dashboard
======================================================
Interactive landing page for the Indian Ocean Cross-Domain Modeling Study.
"""

import json
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Oceanographic AI Platform",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
APP_DIR = Path(__file__).resolve().parent
OCEANAI_DIR = APP_DIR.parent
DATA_DIR = OCEANAI_DIR / "data" / "processed" / "integrated"
METRICS_DIR = OCEANAI_DIR / "reports" / "metrics"
FIGURES_DIR = OCEANAI_DIR / "reports" / "figures"


@st.cache_data
def load_summary_metrics():
    summary_csv = METRICS_DIR / "integration_summary.csv"
    if summary_csv.exists():
        return pd.read_csv(summary_csv)
    return None


@st.cache_data
def load_model_metrics():
    model_json = METRICS_DIR / "model_benchmark_metrics.json"
    if model_json.exists():
        with open(model_json, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# Sidebar
st.sidebar.title("🌊 Ocean AI Platform")
st.sidebar.markdown(
    "**Indian Ocean Cross-Domain Study**  \n"
    "Copernicus Marine ⨂ Global Fishing Watch ⨂ OBIS"
)
st.sidebar.info("""
**Geographical Scope:**
- **Arabian Sea:** Lon 50–78°E, Lat 0–25°N
- **Bay of Bengal:** Lon 78–100°E, Lat 0–23°N
- **Grid:** Standardized 0.25° × 0.25°
- **Period:** 2019-01 to 2025-12 (84 Months)
""")

st.title("🌊 Oceanographic AI Platform")
st.subheader("Data Integration → ML Modeling → Anomaly Detection → Biodiversity Insights")

# Prominent Caveats Alert Banner
st.error("""
🚨 **Critical Analytical Caveats & Scientific Disclaimers:**
1. **GFW 2022 Coverage Jump:** The +310.96% increase in apparent fishing in the Bay of Bengal from 2022 onward reflects an AIS satellite constellation upgrade, NOT real fleet growth. `year` is strictly excluded from all model feature spaces.
2. **Malacca Strait Incursion:** The Bay of Bengal bounding box includes the Andaman Sea and northern entrance to the Malacca Strait, explaining the 4.1°N, 99.9°E hotspot and foreign flag states (MYS, IDN, THA).
3. **OBIS UNESCO Sundarbans Survey:** 39.34% of all OBIS records (100% of *Setipinna tenuifilis*) originate from a single 2-day eDNA cruise (8–9 Jan 2023, 19 sites). eDNA reads denote molecular presence, NOT abundance. Species Distribution Modeling (SDM) is omitted to avoid predicting cruise tracks.
4. **Association ≠ Causation:** Cross-domain findings describe statistical associations between ocean state and fishing detections, not causal mechanisms.
""")

st.markdown("---")

# KPI Summary Cards
summary_df = load_summary_metrics()
model_metrics = load_model_metrics()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Spatial Grid Resolution", "0.25° (~28 km)")
col2.metric("Total Harmonized Panel", "1,255,968 cell-months")
col3.metric("Zero-Inflation Rate", "78.94% (0 hrs)")
if model_metrics:
    col4.metric("Best Occurrence ROC-AUC", f"{model_metrics['stage_a']['XGBoost_M1']['ROC_AUC']:.4f}")
else:
    col4.metric("Temporal Window", "84 Months (2019–2025)")

st.markdown("---")

tab_overview, tab_roadmap, tab_arch = st.tabs([
    "📋 Project Overview & Findings",
    "🗓️ 5-Day Delivery Roadmap",
    "🏗️ System Architecture"
])

with tab_overview:
    st.markdown("""
    ### Research Question
    > **Do ocean physical conditions (potential temperature, salinity, currents) help predict where and how much apparent commercial fishing happens in a given month?**
    
    ### Key Empirical Conclusions
    1. **Two-Stage Hurdle Superiority:** Because fishing activity is heavily zero-inflated (78.94% zero) and right-skewed, single-stage models fail. A two-stage hurdle approach (Occurrence Stage A + Intensity Stage B) achieves strong generalization on out-of-time test data (2025).
    2. **Scientific Ablation Result (Spatial Dominance):**
       - **Model 1 (Location + Seasonality only)** achieves a test **ROC-AUC of 0.8065** in Stage A and **$R^2$ of 0.2064** in Stage B, easily outperforming historical Climatology (ROC-AUC 0.7672) and Persistence (ROC-AUC 0.7024).
       - **Model 2 (Adding Ocean Variables: SST, Salinity, Velocity, Anomalies, Lags)** achieves a test **ROC-AUC of 0.7992** ($\Delta = -0.0074$).
       - **Scientific Implication:** Spatial boundaries (EEZs, continental shelf edges, distance to ports) and seasonal monsoon weather rhythms dominate fishing effort. Monthly ocean physical variables add negligible predictive signal and can introduce slight out-of-time variance. Reporting this finding honestly is a scientifically valid result.
    3. **Cross-Domain Anomaly Shifts:**
       - **Warm SST Anomalies (>90th percentile, $+0.53^\circ\text{C}$):** Fishing occurrence increases from **20.5% to 26.1%**, and mean hours rise by **+35.9%** ($p < 10^{-15}$). Pelagic shoaling along thermal edges concentrates effort.
       - **Multivariate Ocean Disruptions (Isolation Forest):** Fishing occurrence drops from **21.4% to 15.4%**, and mean hours plunge by **-38.1%**, as severe hydrodynamic turbulence impedes vessel departures.
    """)

with tab_roadmap:
    st.markdown("""
    ### 5-Day Strategic Delivery Plan
    | Day | Focus Area | Deliverable Status |
    | :---: | :--- | :--- |
    | **Day 1** | **Grid Integration & Harmonization** | ✅ **COMPLETE** — Coarsened Copernicus to 0.25°, binned GFW, zero-filled water cells. 1.25M rows. |
    | **Day 2** | **Feature Engineering & Benchmarks** | ✅ **COMPLETE** — Cyclical seasonality, cell climatology, anomalies, 1-mo lags, distance to coast via cKDTree. Strict year exclusion. |
    | **Day 3** | **Two-Stage ML & Ablation Study** | ✅ **COMPLETE** — Chronological split (2019-2023 train, 2024 val, 2025 test). Evaluated Baselines vs M1 vs M2. Native TreeSHAP. |
    | **Day 4** | **Ocean Anomaly Detection** | ✅ **COMPLETE** — 90th percentile SST heatwave proxy, Isolation Forest, Mann-Whitney U test on fishing shifts. |
    | **Day 5** | **Biodiversity Layer & Dashboard** | ✅ **COMPLETE** — OBIS environmental envelopes, UNESCO eDNA flag, 7-page interactive Streamlit dashboard, comprehensive documentation. |
    """)

with tab_arch:
    st.markdown("""
    ### Modular Pipeline Architecture
    ```
    Copernicus Marine Reanalysis (0.083°, 84 mo) ──┐
                                                    ├──> src/integrate.py  ──> integrated.parquet (0.25° Grid)
    Global Fishing Watch (AIS Effort, 84 mo) ───────┘
                                                              │
                                                              ▼
    scipy.spatial.cKDTree Land Mask Coastline ─────> src/features.py   ──> features.parquet
                                                              │
                                       ┌──────────────────────┴──────────────────────┐
                                       ▼                                             ▼
                              src/train_models.py                           src/anomalies.py
                        (Baselines + M1/M2 Ablation)                  (P90 SST / IsolationForest)
                                       │                                             │
                                       ▼                                             ▼
                               models/ & metrics/                            metrics/ & figures/
                                       │                                             │
    OBIS Occurrences (2015-2025) ──────┼─────────────────────────────────────────────┘
                                       ▼
                               src/obis_env.py
                        (Envelopes & eDNA Flagging)
                                       │
                                       ▼
                       Interactive Streamlit Dashboard (app/)
                   (Pages 1 to 7: Maps, ML, Ablation, SHAP, Anomalies, OBIS, Limitations)
    ```
    """)

st.markdown("---")
st.caption("Use the left navigation sidebar to explore individual analytical sections.")
