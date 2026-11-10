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
    reg_sel = st.selectbox("Select Marine Region:", ["Both Regions (Arabian Sea + Bay of Bengal)", "Arabian Sea", "Bay of Bengal"])
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

# Geographic Extent & Zero-Filter Controls
col_ext, col_zero = st.columns(2)
with col_ext:
    lat_extent = st.radio(
        "Geographic Latitude Range:",
        [
            "Full Retrieval Box (0°N – 25°N)",
            "Strict Marine Basins (Lat ≥ 6°N, excludes open Equatorial Indian Ocean)"
        ],
        horizontal=True,
        help="The raw Copernicus and GFW datasets were downloaded using rectangular bounding boxes starting at 0°N (the Equator). Traditional Arabian Sea and Bay of Bengal limits begin north of Sri Lanka/Maldives (~6°–8°N)."
    )

if "Strict Marine Basins" in lat_extent:
    subset = subset[subset["lat_bin"] >= 6.0].copy()

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

is_fishing_var = "Fishing" in var_choice
hide_zeros = False
if is_fishing_var:
    hide_zeros = st.checkbox(
        "Hide Zero-Effort Cells (Show only cells with detected fishing > 0 hours)",
        value=True,
        help="When checked, cells with 0 fishing hours are not drawn, so only active fishing hotspots are highlighted rather than coloring the entire ocean canvas."
    )

col_map = {
    "Apparent Fishing Hours [log1p(hours)]": ("log_hours", "YlOrRd"),
    "Potential Temperature (°C)": ("thetao", "Viridis"),
    "Salinity (PSU)": ("so", "YlGnBu"),
    "Current Speed (m/s)": ("speed", "Plasma"),
    "SST Anomaly (°C)": ("sst_anomaly", "RdBu_r")
}

val_col, c_scale = col_map[var_choice]

plot_subset = subset.copy()
if is_fishing_var and hide_zeros:
    plot_subset = plot_subset[plot_subset["hours"] > 0].copy()

fig = px.scatter(
    plot_subset,
    x="lon_bin",
    y="lat_bin",
    color=val_col,
    color_continuous_scale=c_scale,
    title=f"{var_choice} — {year_sel}-{month_sel:02d} ({len(plot_subset):,} plotted 0.25° cells)",
    labels={"lon_bin": "Longitude (°E)", "lat_bin": "Latitude (°N)", val_col: var_choice},
    opacity=0.85,
    height=600
)
fig.update_traces(marker=dict(size=5, symbol="square"))
st.plotly_chart(fig, use_container_width=True)

# Geographic Caveat Alert Box
st.info("""
ℹ️ **Why is there data south of India down to 0°N (the Equator)?**
- **Rectangular Bounding Box Effect:** During the original data ingestion, the retrieval coordinates in `config/regions.json` were configured from **0°N to 25°N** (Arabian Sea) and **0°N to 23°N** (Bay of Bengal).
- **Equatorial Indian Ocean Inclusion:** Geographically, India ends at Cape Comorin (~8.1°N) and Sri Lanka ends at ~5.9°N. The ocean strip between 0°N (the Equator) and 6°–8°N is technically the **Equatorial Indian Ocean**. Because rectangular bounding boxes were requested from Copernicus and GFW, data was downloaded for this entire equatorial strip (**over 50% of the raw data rows** lie in this 0°–8°N band).
- **Use the toggle above** (`Strict Marine Basins (Lat ≥ 6°N)`) to exclude these open equatorial waters if you only want to focus on the enclosed coastal basins!
""")

st.warning("""
📌 **Additional Hotspots & Caveats:**
- **Malacca Strait Incursion:** The Bay of Bengal retrieval box extends eastward to 100°E, incorporating the entrance to the Malacca Strait and Andaman Sea. This explains the extreme hotspot near **4.1°N, 99.9°E** and the presence of Southeast Asian flag states (Malaysia, Indonesia, Thailand).
- **Zero-Filling:** In GFW, ocean cells without an AIS detection represent **0.0 hours** (zero fishing effort), not missing data. With "Hide Zero-Effort Cells" checked, only genuine fishing detections are plotted.
""")
