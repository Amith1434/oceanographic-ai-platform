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

with st.expander("🌊 3. Why Data Extends South of India down to 0°N (The Equator)", expanded=True):
    st.markdown("""
    - **Bounding Box Retrieval:** The original data ingestion configured retrieval bounding boxes starting at **0°N** (`min_lat: 0` in `config/regions.json`).
    - **Geographic Reality:** India's southern tip (Kanyakumari) is at **~8.1°N**, and Sri Lanka ends at **~5.9°N**. The open water between 0°N and 6°–8°N is technically the **Equatorial Indian Ocean**.
    - **Data Presence:** Because rectangular bounding boxes were downloaded from Copernicus and GFW, data was pulled for the entire rectangular swath down to the equator. In fact, **over 50% of the raw data rows** lie in this 0°–8°N band.
    - **Visualization:** In the Ocean Map page, users can choose between viewing the full rectangular retrieval box (0°N–25°N) or toggling to **Strict Marine Basins (Lat ≥ 6°N)** to exclude these open equatorial waters.
    """)

with st.expander("🚢 4. Apparent Fishing vs True Catch / Real Abundance", expanded=True):
    st.markdown("""
    - **AIS Tracking Scope:** GFW detects *apparent* fishing effort based on machine learning classification of vessel speed and turning patterns from AIS transponders.
    - **Coverage Bias:** AIS is mandated primarily for larger industrial vessels (>12–15 meters, or >300 gross tonnage). The vast artisanal, small-scale motorized, and traditional non-AIS fleets of India, Bangladesh, and Sri Lanka are largely invisible to AIS.
    - **Effort ≠ Catch:** High apparent hours indicate vessel presence, not necessarily high harvest volume or healthy fish stocks.
    """)

with st.expander("🧬 5. OBIS eDNA Expedition Concentration & SDM Exclusion", expanded=True):
    st.markdown("""
    - **Expedition Identity:** "UNESCO eDNA expedition in The Sundarbans (Bangladesh): February 2023", sampled on **8–9 January 2023** across **19 sampling sites** in the **Sundarbans West Wildlife Sanctuary (Bangladesh)**.
    - **Bounding Box & DOI:** Lat 21.703521°N to 21.828443°N, Lon 89.195000°E to 89.284180°E. DOI: `10.25607/rt4e3h`.
    - **Expedition Concentration:** Exactly **39.34% of all OBIS records** (476 of 1,210) belong to *Setipinna tenuifilis* (Common hairfin anchovy), and **100%** of these were collected during this single 2-day UNESCO survey cruise.
    - **eDNA Scope:** Environmental DNA measures shed cellular DNA in water samples. Positive eDNA detections confirm taxon sequence presence but do not equate to individual counts, biomass, or resident population density.
    - **Why SDM Was Omitted:** Fitting Species Distribution Models (SDMs) with presence-only data from one cruise track would produce extreme pseudo-absence artifacts and model the researcher's boat trajectory. This layer remains strictly descriptive.
    - **Estuarine Resolution Caveat:** The 0.25° marine grid is coarse relative to deltaic mangrove creek networks; matched values represent conditions at the nearest ocean margin.
    """)

with st.expander("🔬 6. Exploratory Associations vs Causal Inference", expanded=True):
    st.markdown("""
    - **Inferential Scope:** All cross-domain relationships reported (e.g., shifts in fishing hours during SST warm anomalies) are **exploratory observational associations**, NOT causal proofs.
    - **Temporal Resolution:** Monthly averages cannot resolve short-term synoptic storms or transient depressions that dock fleets for 48–72 hours.
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
