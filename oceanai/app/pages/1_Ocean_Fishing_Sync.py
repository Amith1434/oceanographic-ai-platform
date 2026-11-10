"""
1_Ocean_Fishing_Sync.py
-----------------------
Page 1: Cross-Domain Synchronized Explorer: Ocean Physics ⨂ Apparent Fishing Effort
Interactive side-by-side synchronized spatial maps and cross-domain correlation diagnostics.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import spearmanr, pearsonr
import streamlit as st

st.set_page_config(page_title="Ocean ↔ Fishing Sync - Ocean AI", page_icon="🧭", layout="wide")

OCEANAI_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = OCEANAI_DIR / "data" / "processed" / "integrated"
METRICS_DIR = OCEANAI_DIR / "reports" / "metrics"


@st.cache_data
def load_features_sample():
    features_path = DATA_DIR / "features.parquet"
    if not features_path.exists():
        return None
    # Read core columns needed for interactive mapping
    cols = [
        "month", "lat_bin", "lon_bin", "region", "region_id", "hours", "log_hours", "is_fishing",
        "thetao", "so", "uo", "vo", "speed", "sst_anomaly", "so_anomaly", "dist_to_coast_km"
    ]
    return pd.read_parquet(features_path, columns=cols)


st.title("🧭 Cross-Domain Synchronized Explorer: Ocean ↔ Fishing")
st.markdown(r"""
Coupled spatio-temporal view comparing **Copernicus Physical Oceanography** directly alongside **Global Fishing Watch (GFW)** apparent commercial fishing effort across identical **0.25° grid cells** and monthly time-steps (2019–2025).
""")

# Top Caveat Expander
with st.expander("ℹ️ Essential Spatial & Activity Caveats (Click to expand)", expanded=False):
    st.markdown(r"""
    - **Zero-Filling Principle:** In GFW, ocean cells with no recorded AIS detections represent **0.0 fishing hours** (absence of activity), not missing data (`NaN`). Missing Copernicus data represents the continental land-mask.
    - **The 0°N Boundary:** Bounding boxes extend to the Equator ($0^\circ\text{N}$). Mainland India terminates at $\approx 8.1^\circ\text{N}$ and Sri Lanka at $\approx 5.9^\circ\text{N}$. Waters between $0^\circ\text{N}$ and $8^\circ\text{N}$ represent the international open Equatorial Indian Ocean.
    - **2022 Satellite Expansion Artifact:** Apparent fishing effort in the Bay of Bengal jumps +310.96% from 2022 onward due to satellite AIS constellation expansions (Spire/exactEarth) and Class-B transponder adoption, not sudden fish biomass growth.
    - **Malacca Strait Entrance:** The Bay of Bengal box extends eastward to $100^\circ\text{E}$, capturing the dense shipping and trawling corridor at the mouth of the Malacca Strait ($4.1^\circ\text{N}, 99.9^\circ\text{E}$).
    """)

df_full = load_features_sample()
if df_full is None:
    st.error("Dataset not found at `data/processed/integrated/features.parquet`. Please run feature engineering first.")
    st.stop()

# Interactive Filter Controls
st.subheader("1. Spatial & Temporal Selectors")
c_reg, c_yr, c_mo, c_lat = st.columns([2, 1.5, 1.5, 2.5])

with c_reg:
    reg_choice = st.selectbox(
        "Marine Basin:",
        ["All Basins (Arabian Sea + Bay of Bengal)", "Arabian Sea Only", "Bay of Bengal Only"],
        index=0
    )

with c_yr:
    year_sel = st.selectbox("Year:", sorted(df_full["month"].dt.year.unique()), index=5) # Default 2024

with c_mo:
    month_sel = st.slider("Month:", min_value=1, max_value=12, value=10) # Default October (post-monsoon)

with c_lat:
    lat_choice = st.radio(
        "Latitude Boundary Filter:",
        ["Full Retrieval Box (0°N – 25°N)", "Coastal & Basin Shelves (Lat ≥ 6°N)"],
        index=0,
        help="Filter out open equatorial waters south of 6°N to focus on coastal EEZs."
    )

# Filter Data Slice
sel_month_str = f"{year_sel}-{month_sel:02d}-01"
subset = df_full[df_full["month"] == sel_month_str].copy()

if reg_choice == "Arabian Sea Only":
    subset = subset[subset["region"] == "arabian_sea"].copy()
elif reg_choice == "Bay of Bengal Only":
    subset = subset[subset["region"] == "bay_of_bengal"].copy()

if "Lat ≥ 6°N" in lat_choice:
    subset = subset[subset["lat_bin"] >= 6.0].copy()

if subset.empty:
    st.warning("No ocean cells match the selected filter criteria. Please broaden your selection.")
    st.stop()

# Variable Selectors
c_var1, c_var2, c_opt = st.columns([2, 2, 2])
with c_var1:
    ocean_var = st.selectbox(
        "Ocean Physical Layer (Left Map):",
        [
            ("thetao", "Sea Surface Temperature (°C)", "Viridis"),
            ("so", "Sea Surface Salinity (PSU)", "YlGnBu"),
            ("speed", "Current Velocity Speed (m/s)", "Plasma"),
            ("sst_anomaly", "SST Anomaly relative to 2019-2023 Clim (°C)", "RdBu_r"),
            ("so_anomaly", "Salinity Anomaly (PSU)", "BrBG_r"),
            ("dist_to_coast_km", "Distance to Coast (km)", "Blues_r")
        ],
        format_func=lambda x: x[1],
        index=0
    )

with c_var2:
    fish_var = st.selectbox(
        "Fisheries Layer (Right Map):",
        [
            ("hours", "Apparent Fishing Effort (Raw Hours)", "YlOrRd"),
            ("log_hours", "Log Apparent Effort [log1p(hours)]", "YlOrRd"),
            ("is_fishing", "Binary Occurrence (Fished vs Unfished)", "Blues")
        ],
        format_func=lambda x: x[1],
        index=1
    )

with c_opt:
    hide_zeros = st.checkbox(
        "Hide Zero-Effort Cells on Fishing Map",
        value=True,
        help="When checked, only ocean cells with active fishing (> 0 hours) are plotted on the fisheries map."
    )

# Summary KPIs
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
fished_cells = int((subset["hours"] > 0).sum())
tot_cells = len(subset)
fished_pct = (fished_cells / tot_cells) * 100 if tot_cells > 0 else 0
tot_hrs = float(subset["hours"].sum())
mean_hrs_active = float(subset[subset["hours"] > 0]["hours"].mean()) if fished_cells > 0 else 0.0

kpi1.metric("Active Ocean Cells", f"{tot_cells:,}")
kpi2.metric("Fished Cells Rate", f"{fished_cells:,} ({fished_pct:.1f}%)")
kpi3.metric("Total Apparent Effort", f"{tot_hrs:,.1f} hrs")
kpi4.metric("Mean Effort in Fished Cells", f"{mean_hrs_active:.1f} hrs/cell")

st.markdown("---")

# 2. Side-by-Side Synchronized Plotly Maps
st.subheader(f"2. Spatial Distribution ({year_sel}-{month_sel:02d})")
col_left, col_right = st.columns(2)

lon_min, lon_max = subset["lon_bin"].min() - 1, subset["lon_bin"].max() + 1
lat_min, lat_max = subset["lat_bin"].min() - 1, subset["lat_bin"].max() + 1

with col_left:
    st.markdown(f"**🌊 Ocean Physics:** {ocean_var[1]}")
    fig_ocean = px.scatter(
        subset,
        x="lon_bin",
        y="lat_bin",
        color=ocean_var[0],
        color_continuous_scale=ocean_var[2],
        labels={"lon_bin": "Longitude (°E)", "lat_bin": "Latitude (°N)", ocean_var[0]: ocean_var[1]},
        hover_data={
            "lat_bin": ":.2f",
            "lon_bin": ":.2f",
            ocean_var[0]: ":.3f",
            "hours": ":.1f",
            "dist_to_coast_km": ":.1f"
        },
        height=520
    )
    fig_ocean.update_traces(marker=dict(size=4.5, symbol="square"))
    fig_ocean.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(range=[lon_min, lon_max]),
        yaxis=dict(range=[lat_min, lat_max])
    )
    st.plotly_chart(fig_ocean, use_container_width=True)

with col_right:
    st.markdown(f"**⚓ Commercial Fishing:** {fish_var[1]}")
    fish_subset = subset[subset["hours"] > 0].copy() if hide_zeros else subset.copy()
    
    fig_fish = px.scatter(
        fish_subset,
        x="lon_bin",
        y="lat_bin",
        color=fish_var[0],
        color_continuous_scale=fish_var[2],
        labels={"lon_bin": "Longitude (°E)", "lat_bin": "Latitude (°N)", fish_var[0]: fish_var[1]},
        hover_data={
            "lat_bin": ":.2f",
            "lon_bin": ":.2f",
            "hours": ":.1f",
            "log_hours": ":.2f",
            "thetao": ":.2f",
            "dist_to_coast_km": ":.1f"
        },
        height=520
    )
    fig_fish.update_traces(marker=dict(size=4.5, symbol="square"))
    fig_fish.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(range=[lon_min, lon_max]),
        yaxis=dict(range=[lat_min, lat_max])
    )
    st.plotly_chart(fig_fish, use_container_width=True)

st.markdown("---")

# 3. Synchronized Cross-Domain Statistical Coupling
st.subheader("3. Cross-Domain Correlation Diagnostics (This Time Slice)")

c_stat1, c_stat2 = st.columns([1.5, 2])

with c_stat1:
    x_vals = subset[ocean_var[0]].dropna()
    y_vals = subset.loc[x_vals.index, "hours"]
    
    p_corr, p_val = pearsonr(x_vals, y_vals) if len(x_vals) > 2 else (0.0, 1.0)
    s_corr, s_val = spearmanr(x_vals, y_vals) if len(x_vals) > 2 else (0.0, 1.0)
    
    st.markdown(f"""
    **Bivariate Association:**
    - **Predictor:** `{ocean_var[0]}` ({ocean_var[1]})
    - **Response:** `hours` (Apparent Fishing Hours)
    - **Pearson Correlation ($r$):** `{p_corr:+.4f}` ($p = {p_val:.2e}$)
    - **Spearman Rank Correlation ($\\rho$):** `{s_corr:+.4f}` ($p = {s_val:.2e}$)
    
    > **Interpretation:**  
    > Across single-month snapshots, surface physical variables exhibit modest rank correlation with fishing effort ($|\\rho| \\approx 0.05 - 0.25$). Fixed geographic coordinates (`lat_bin`, `lon_bin`, `dist_to_coast_km`) consistently account for the dominant share of spatial variance.
    """)

with c_stat2:
    # Hexbin / scatter plot on active rows
    active_sub = subset[subset["hours"] > 0]
    if len(active_sub) > 10:
        fig_scatter = px.scatter(
            active_sub,
            x=ocean_var[0],
            y="log_hours",
            color="dist_to_coast_km",
            color_continuous_scale="Viridis",
            labels={
                ocean_var[0]: ocean_var[1],
                "log_hours": "Apparent Fishing [log1p(hours)]",
                "dist_to_coast_km": "Dist to Coast (km)"
            },
            title=f"{ocean_var[1]} vs Apparent Fishing Hours (Fished Cells Only)",
            opacity=0.7,
            height=350
        )
        fig_scatter.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Fewer than 10 active fishing cells in this slice to compute scatter plot.")
