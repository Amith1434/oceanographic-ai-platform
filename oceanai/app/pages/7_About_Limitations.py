"""
Page 7: Analytical Caveats, Data Limitations & Scientific Ethics
================================================================
Comprehensive documentation of all caveats, assumptions, and guardrails.
"""

from pathlib import Path
import streamlit as st

st.set_page_config(page_title="About & Limitations - Ocean AI", page_icon="📖", layout="wide")

st.title("📖 Analytical Caveats, Data Limitations & Scientific Ethics")
st.markdown("""
A core principle of scientific machine learning is rigorous transparency regarding the limits of observational data.
This section details all known caveats, systemic biases, and inferential boundaries of this study.
""")

st.header("1. Core Caveats Checklist")

with st.expander("🚨 1. GFW 2022 Coverage Jump (Satellite Artifact)", expanded=True):
    st.markdown("""
    - **The Artifact:** In 2022, apparent fishing hours in the Bay of Bengal jumped by **+310.96%** compared to 2021.
    - **Physical vs Observational Reality:** This surge does **not** reflect a tripling of real fishing vessels. Rather, it coincides with Global Fishing Watch updating their AIS satellite constellation reception algorithms and expanding low-earth orbit satellite receiver feeds.
    - **Strict Methodological Guardrail:** Calendar `year` was **strictly excluded** from all model feature spaces. If `year` had been included, tree models would have learned a fictitious temporal shortcut, destroying generalization to pre-2022 or post-2025 regimes.
    """)

with st.expander("📍 2. Malacca Strait Incursion & Foreign Flag Hotspots", expanded=True):
    st.markdown("""
    - **Bounding Box Geometry:** The Bay of Bengal query box spans Lon 78°E to 100°E, Lat 0°N to 23°N.
    - **The Hotspot:** The extreme fishing hotspot detected near **4.1°N, 99.9°E** sits at the northern entrance of the Malacca Strait in the Andaman Sea.
    - **Flag Composition:** This explains why foreign Southeast Asian flags (Malaysia, Indonesia, Thailand) appear prominently in Bay of Bengal data alongside regional Indian Ocean coastal states.
    """)

with st.expander("🚢 3. Apparent Fishing vs True Catch / Real Abundance", expanded=True):
    st.markdown("""
    - **AIS Tracking Scope:** GFW detects *apparent* fishing effort based on machine learning classification of vessel speed and turning patterns from AIS transponders.
    - **Coverage Bias:** AIS is mandated primarily for larger industrial vessels (>12–15 meters, or >300 gross tonnage). The vast artisanal, small-scale motorized, and traditional non-AIS fleets of India, Bangladesh, and Sri Lanka are largely invisible to AIS.
    - **Effort ≠ Catch:** High apparent hours indicate vessel presence, not necessarily high harvest volume or healthy fish stocks.
    """)

with st.expander("🧬 4. OBIS eDNA Expedition Concentration & SDM Exclusion", expanded=True):
    st.markdown("""
    - **Expedition Concentration:** **39.34% of all OBIS records** (476 of 1,210) belong to *Setipinna tenuifilis*, and **100%** of these were collected during a single 2-day UNESCO Sundarbans survey cruise (8–9 January 2023) across 17–19 sample sites.
    - **eDNA Scope:** Environmental DNA measures shed cellular DNA in water samples. Positive eDNA detections confirm taxon sequence presence but do not equate to individual counts, biomass, or resident population density.
    - **Why SDM Was Omitted:** Fitting Species Distribution Models (SDMs) with presence-only data from one cruise track would produce extreme pseudo-absence artifacts and model the researcher's boat trajectory. This layer remains strictly descriptive.
    """)

with st.expander("🔬 5. Exploratory Associations vs Causal Inference", expanded=True):
    st.markdown("""
    - **Inferential Scope:** All cross-domain relationships reported (e.g., shifts in fishing hours during SST warm anomalies) are **exploratory observational associations**, NOT causal proofs.
    - **Confounding Factors:** Fuel prices, market demand, geopolitical port access, naval enforcement, and national trawl bans are major latent drivers that cannot be observed from satellite oceanography alone.
    """)

st.markdown("---")

st.header("2. Reproducibility & Codebase Access")
st.markdown("""
All code, models, and outputs are packaged in the isolated `oceanai/` directory:
- **Pipeline Orchestrator:** `python oceanai/src/integrate.py` → `features.py` → `train_models.py` → `anomalies.py` → `obis_env.py`
- **Dashboard Command:** `streamlit run oceanai/app/Home.py`
- **Automation:** `make -C oceanai pipeline`
""")
