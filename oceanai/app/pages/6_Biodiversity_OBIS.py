"""
Page 6: Descriptive Marine Biodiversity Layer (OBIS)
=====================================================
Presents realized environmental ranges for focal marine species,
quantifies match quality across 4 categories, and flags the UNESCO eDNA artifact.
"""

from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Biodiversity - Ocean AI", page_icon="🦈", layout="wide")

OCEANAI_DIR = Path(__file__).resolve().parent.parent.parent
METRICS_DIR = OCEANAI_DIR / "reports" / "metrics"
FIGURES_DIR = OCEANAI_DIR / "reports" / "figures"


@st.cache_data
def load_obis_data():
    f_match = METRICS_DIR / "obis_match_quality.csv"
    f_env = METRICS_DIR / "obis_species_envelopes.csv"
    
    df_match = pd.read_csv(f_match) if f_match.exists() else None
    df_env = pd.read_csv(f_env) if f_env.exists() else None
    return df_match, df_env


st.title("🦈 Descriptive Marine Biodiversity Layer (OBIS)")
st.markdown("""
Extracts realized environmental ranges (Copernicus temperature, salinity, currents) and commercial fishing overlap
for 1,210 marine biodiversity observations recorded in the Ocean Biodiversity Information System (OBIS, 2015–2025).
""")

# Top Caveat Banner
st.error("""
🚩 **CRITICAL SAMPLING ARTIFACT: UNESCO Sundarbans eDNA Expedition (Bangladesh)**
- **Sampling Identity:** "UNESCO eDNA expedition in The Sundarbans (Bangladesh): February 2023", sampled on **8–9 January 2023** across **19 sampling sites** in the **Sundarbans West Wildlife Sanctuary (Bangladesh)**.
- **Bounding Box:** Lat 21.703521°N to 21.828443°N, Lon 89.195000°E to 89.284180°E.
- **Dataset Impact:** Exactly **476 out of 1,210 records (39.34% of the entire OBIS dataset, and 100% of Setipinna tenuifilis records)** originate from this single 2-day research cruise (DOI: `10.25607/rt4e3h`).
- **Why Species Distribution Modeling (SDM) was Omitted:** Fitting an SDM or MaxEnt algorithm on this presence-only dataset would inadvertently model the research vessel's itinerary rather than biological habitat suitability. OBIS is presented strictly as a descriptive layer with realized environmental envelopes.
- **Estuarine Resolution Caveat:** The 0.25° grid (~28 km) is coarse relative to narrow mangrove creeks, so environmental matches reflect the nearest ocean cell.
""")

df_match, df_env = load_obis_data()
if df_match is None or df_env is None:
    st.error("OBIS metrics not found. Run obis_env.py first.")
    st.stop()

# 1. Matching Quality Classification
st.header("1. Copernicus Ocean Matching Quality (4-Category Classification)")
st.markdown("""
To prevent data contamination from temporal extrapolation, records are strictly audited into four categories:
1. `exact_match`: Point falls in an active ocean grid cell in the exact observed month (2019–2025).
2. `nearest_cell_same_month`: Point falls on land/estuary but matches a marine cell within $\\le 0.5^\\circ$ in the same month.
3. `no_match_pre2019`: Records prior to Copernicus coverage (2015–2018) — **strictly dropped** (no all-months imputation).
4. `no_match_far`: Records farther than $> 0.5^\\circ$ from any ocean cell — **strictly dropped**.
""")

st.dataframe(df_match.style.format({
    "Total_Records": "{:,}",
    "Pre_2019": "{:,}",
    "Exact_Match": "{:,}",
    "Nearest_Same_Month": "{:,}",
    "Far_Dropped": "{:,}",
    "Dropped_Records": "{:,}",
    "Used_Records": "{:,}",
    "Used_Share_Pct": "{:.2f}%"
}))

st.markdown("---")

# 2. Realized Environmental Envelopes
st.header("2. Realized Environmental Envelopes & Commercial Fishing Co-occurrence")

# Exclude eDNA Toggle
exclude_edna = st.checkbox(
    "Exclude UNESCO Sundarbans eDNA Expedition (*Setipinna tenuifilis*)",
    value=False,
    help="Toggle to inspect only pelagic and reef shark species free from the estuarine eDNA expedition artifact."
)

plot_env = df_env.copy()
if exclude_edna:
    plot_env = plot_env[plot_env["Species"] != "Setipinna tenuifilis"].copy()

st.dataframe(plot_env.style.format({
    "Total_Records": "{:,}",
    "Matched_Records_Used": "{:,}",
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
    "Co_occurring_Fishing_Pct": "{:.2f}%",
    "Mean_Fishing_Hours_In_Cell": "{:.2f} hrs"
}))

st.markdown("---")

# 3. Species Profiles
st.header("3. Focal Species Profiles")
sp1, sp2, sp3 = st.columns(3)

with sp1:
    st.subheader("🦈 Blacktip Reef Shark")
    st.caption("*Carcharhinus melanopterus* (595 total, 544 used)")
    st.markdown("""
    - **Thermal Envelope:** 29.59 °C median (IQR: 0.88 °C)
    - **Salinity:** 34.42 PSU (High marine salinity)
    - **Commercial Fishing Co-occurrence:** **14.71%**
    - **Mean Fishing in Fished Cells:** 39.87 hrs
    - **Habitat:** Shallow coral atolls (Lakshadweep, Maldives). Encounters modest overlap with coastal gillnet and artisanal fisheries.
    """)

with sp2:
    st.subheader("🐋 Whale Shark")
    st.caption("*Rhincodon typus* (139 total, 120 used)")
    st.markdown("""
    - **Thermal Envelope:** 29.62 °C median (IQR: 1.13 °C)
    - **Salinity:** 34.46 PSU (Open oceanic waters)
    - **Commercial Fishing Co-occurrence:** **7.50%**
    - **Mean Fishing in Fished Cells:** 3.23 hrs
    - **Habitat:** Migratory pelagic filter feeder. Distributed along Gujarat coast and open offshore migration corridors.
    """)

with sp3:
    st.subheader("🐟 Common Hairfin Anchovy")
    st.caption("*Setipinna tenuifilis* (476 total, 476 used - Bay of Bengal only)")
    if exclude_edna:
        st.info("Excluded via filter toggle above.")
    else:
        st.markdown("""
        - **Thermal Envelope:** 24.97 °C median (IQR: 0.56 °C)
        - **Salinity:** **27.66 PSU** (Estuarine brackish runoff)
        - **Commercial Fishing Co-occurrence:** **0.00%**
        - **Habitat:** Sundarbans mangrove creeks. 100% of records from UNESCO eDNA survey (Jan 2023). Zero AIS commercial fishing co-occurrence.
        """)

st.markdown("---")

# 4. Diagnostic Figures
st.header("4. Publication Diagnostic Visualizations")
if (FIGURES_DIR / "obis_environmental_envelopes.png").exists():
    st.image(str(FIGURES_DIR / "obis_environmental_envelopes.png"), caption="Realized Environmental Ranges (Temperature, Salinity, Current Speed)", use_container_width=True)

col_f1, col_f2 = st.columns(2)
if (FIGURES_DIR / "obis_fishing_overlay_map.png").exists():
    col_f1.image(str(FIGURES_DIR / "obis_fishing_overlay_map.png"), caption="Spatial Co-occurrence: Species vs Mean Apparent Fishing Grounds", use_container_width=True)
if (FIGURES_DIR / "unesco_edna_sundarbans_map.png").exists():
    col_f2.image(str(FIGURES_DIR / "unesco_edna_sundarbans_map.png"), caption="UNESCO Sundarbans eDNA Sampling Track (Bangladesh, 8–9 Jan 2023)", use_container_width=True)
