"""
Page 1: Interactive Gridded Ocean Explorer (0.25° Resolution)
=============================================================
Visualizes harmonized Copernicus physical oceanography and GFW apparent fishing activity.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Ocean Map - Ocean AI", page_icon="🗺️", layout="wide")

OCEANAI_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = OCEANAI_DIR / "data" / "processed" / "integrated"


@st.cache_data
def load_grid_data():
    parquet_path = DATA_DIR / "features.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return None


st.title("🗺️ Gridded Ocean & Fisheries Explorer")
st.markdown("Explore unified physical ocean variables and apparent commercial fishing effort across the **0.25° marine grid**.")

df = load_grid_data()
if df is None:
    st.error("Dataset not found. Please run the integration pipeline first.")
    st.stop()

# Filter Controls
col_reg, col_yr, col_mo = st.columns(3)
with col_reg:
    reg_sel = st.selectbox("Select Marine Region:", ["Combined Indian Ocean", "Arabian Sea", "Bay of Bengal"])
with col_yr:
    year_sel = st.slider("Select Year:", 2019, 2025, 2024)
with col_mo:
    month_sel = st.slider("Select Calendar Month:", 1, 12, 10)

mo_str = f"{year_sel}-{month_sel:02d}-01"
subset = df[df["month"] == mo_str].copy()

if reg_sel == "Arabian Sea":
    subset = subset[subset["region"] == "arabian_sea"]
elif reg_sel == "Bay of Bengal":
    subset = subset[subset["region"] == "bay_of_bengal"]

# KPI Metrics
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Valid Ocean Cells", f"{len(subset):,}")
fished_cnt = (subset["hours"] > 0).sum()
fished_pct = (subset["hours"] > 0).mean() * 100
kpi2.metric("Fished Cells (hours > 0)", f"{fished_cnt:,} ({fished_pct:.1f}%)")
kpi3.metric("Mean Surface Temp", f"{subset['thetao'].mean():.2f} °C")
kpi4.metric("Total Apparent Effort", f"{subset['hours'].sum():,.1f} hrs")

st.markdown("---")

var_choice = st.radio(
    "Select Display Variable:",
    [
        "Apparent Fishing Hours [log1p(hours)]",
        "Potential Temperature (°C)",
        "Salinity (PSU)",
        "Current Speed (m/s)",
        "SST Anomaly (°C)"
    ],
    horizontal=True
)

col_map = {
    "Apparent Fishing Hours [log1p(hours)]": ("log_hours", "Blues"),
    "Potential Temperature (°C)": ("thetao", "Viridis"),
    "Salinity (PSU)": ("so", "YlGnBu"),
    "Current Speed (m/s)": ("speed", "Plasma"),
    "SST Anomaly (°C)": ("sst_anomaly", "RdBu_r")
}

val_col, c_scale = col_map[var_choice]

fig = px.scatter(
    subset,
    x="lon_bin",
    y="lat_bin",
    color=val_col,
    color_continuous_scale=c_scale,
    title=f"{var_choice} — {year_sel}-{month_sel:02d} (0.25° Grid Cells)",
    labels={"lon_bin": "Longitude (°E)", "lat_bin": "Latitude (°N)", val_col: var_choice},
    opacity=0.85,
    height=600
)
fig.update_traces(marker=dict(size=5, symbol="square"))
st.plotly_chart(fig, use_container_width=True)

# Geographic Caveat Alert Box
st.warning("""
📌 **Geographic Note on High-Intensity Fishing Hotspots:**
- **Malacca Strait Incursion:** The Bay of Bengal retrieval box extends to 100°E and includes the entrance to the Malacca Strait and Andaman Sea. This explains the prominent high-intensity cluster near **4.1°N, 99.9°E** and the presence of Southeast Asian flag states (Malaysia, Indonesia, Thailand).
- **Zero-Inflation Mechanism:** Cells rendered in pale blue/white denote zero detected AIS fishing hours, correctly zero-filled from GFW detection logs.
""")
