"""
Page 6: Descriptive Marine Biodiversity Layer (OBIS)
=====================================================
Explores realized environmental ranges for focal marine species and flags survey artifacts.
"""

from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Biodiversity - Ocean AI", page_icon="🦈", layout="wide")

OCEANAI_DIR = Path(__file__).resolve().parent.parent.parent
METRICS_DIR = OCEANAI_DIR / "reports" / "metrics"
FIGURES_DIR = OCEANAI_DIR / "reports" / "figures"


@st.cache_data
def load_obis_envelopes():
    f = METRICS_DIR / "obis_species_envelopes.csv"
    if f.exists():
        return pd.read_csv(f)
    return None


st.title("🦈 Descriptive Marine Biodiversity Layer (OBIS)")
st.markdown("""
Extracts realized environmental ranges (Copernicus temperature, salinity, currents) and apparent fishing overlap
for marine biodiversity observations recorded in OBIS (2015–2025).
""")

# Prominent Expedition Caveat
st.error("""
🚩 **CRITICAL SAMPLING ARTIFACT: UNESCO Sundarbans eDNA Expedition**
- **39.34% of all OBIS records** in the entire dataset (476 of 1,210 observations) belong to a single species: *Setipinna tenuifilis* (Common hairfin anchovy).
- **100% of these 476 records** were collected during **ONE 2-day research expedition** (UNESCO Sundarbans, 8–9 January 2023) across 17–19 sampling sites in the northern Bay of Bengal delta.
- **eDNA reads ≠ Abundance:** Environmental DNA detects molecular sequence signatures in water samples, not direct animal counts or population density.
- **Why SDM is Omitted:** Fitting a Species Distribution Model (MaxEnt/GAM) on this presence-only dataset would simply model the research boat's GPS track, creating severe pseudo-absence artifacts. This layer is strictly descriptive.
""")

df_env = load_obis_envelopes()
if df_env is None:
    st.error("OBIS envelopes not found. Run obis_env.py first.")
    st.stop()

st.header("1. Realized Environmental Envelopes Summary")
st.dataframe(df_env.style.format({
    "Total_Records": "{:,}",
    "UNESCO_eDNA_Share_Pct": "{:.1f}%",
    "Temp_Median": "{:.2f} °C",
    "Temp_IQR": "{:.2f} °C",
    "Temp_Min": "{:.2f} °C",
    "Temp_Max": "{:.2f} °C",
    "Salinity_Median": "{:.2f} PSU",
    "Salinity_IQR": "{:.2f} PSU",
    "Salinity_Min": "{:.2f} PSU",
    "Salinity_Max": "{:.2f} PSU",
    "Speed_Median": "{:.3f} m/s",
    "Speed_IQR": "{:.3f} m/s",
    "Co_occurring_Fishing_Pct": "{:.1f}%",
    "Mean_Fishing_Hours_In_Cell": "{:.2f} hrs"
}))

st.markdown("---")

st.header("2. Focal Species Profiles")
sp1, sp2, sp3 = st.columns(3)

with sp1:
    st.subheader("🦈 Blacktip Reef Shark")
    st.caption("*Carcharhinus melanopterus* (595 records)")
    st.markdown("""
    - **Thermal Envelope:** 29.59 °C median (IQR: 1.02 °C)
    - **Salinity:** 34.42 PSU (High marine salinity)
    - **Co-occurrence with Fishing:** **15.3%**
    - **Habitat:** Shallow coral atolls (Lakshadweep, Maldives). Sightings cluster around marine protected dive spots, with modest overlap with coastal gillnet/trawl fisheries.
    """)

with sp2:
    st.subheader("🐋 Whale Shark")
    st.caption("*Rhincodon typus* (139 records)")
    st.markdown("""
    - **Thermal Envelope:** 29.62 °C median (IQR: 1.25 °C)
    - **Salinity:** 34.46 PSU (Open oceanic waters)
    - **Co-occurrence with Fishing:** **7.4%**
    - **Habitat:** Migratory pelagic filter feeder. Distributed along Gujarat coast and open offshore migration corridors with low commercial fishing co-occurrence.
    """)

with sp3:
    st.subheader("🐟 Common Hairfin Anchovy")
    st.caption("*Setipinna tenuifilis* (476 records - Bay of Bengal only)")
    st.markdown("""
    - **Thermal Envelope:** 28.55 °C median (IQR: 0.68 °C)
    - **Salinity:** **22.49 PSU** (Estuarine brackish plume)
    - **Co-occurrence with Fishing:** **0.0%**
    - **Habitat:** Ganges-Brahmaputra deltaic mangrove channels. Zero AIS fishing co-occurrence occurs because nearshore deltaic fisheries are dominated by small artisanal non-AIS craft.
    """)

st.subheader("Diagnostic Figures")
if (FIGURES_DIR / "obis_environmental_envelopes.png").exists():
    st.image(str(FIGURES_DIR / "obis_environmental_envelopes.png"), caption="Species Environmental Range Envelopes (Potential Temperature, Salinity, Current Speed)", use_container_width=True)

col_f1, col_f2 = st.columns(2)
if (FIGURES_DIR / "obis_fishing_overlay_map.png").exists():
    col_f1.image(str(FIGURES_DIR / "obis_fishing_overlay_map.png"), caption="Spatial Overlay: OBIS Species vs GFW Fishing Hotspots", use_container_width=True)
if (FIGURES_DIR / "unesco_edna_sundarbans_map.png").exists():
    col_f2.image(str(FIGURES_DIR / "unesco_edna_sundarbans_map.png"), caption="UNESCO Sundarbans eDNA Expedition Sampling Sites", use_container_width=True)
