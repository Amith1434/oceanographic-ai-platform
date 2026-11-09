"""
OBIS Marine Biodiversity Environmental Matching & Realized Envelopes
=====================================================================
Matches OBIS marine species occurrence records (2015-2025) with Copernicus physical
ocean conditions (potential temperature, salinity, currents) and GFW apparent fishing.

Key Scientific & Analytical Guardrails:
1. Prominently flags and separates the UNESCO Sundarbans eDNA expedition:
   - 476 of 1,210 records (39.34% of total dataset; 100% of Setipinna tenuifilis)
     originate from a single 2-day eDNA survey (8-9 Jan 2023, 19 sampling sites).
   - eDNA reads denote molecular sequence presence, NOT individual abundance.
2. Explicitly omits Species Distribution Modeling (SDM):
   - With 3 species and presence-only data concentrated on a single expedition track,
     SDMs would predict the research vessel's cruise track rather than true habitat suitability.
3. Generates descriptive realized environmental envelopes and spatial fishing overlays.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# Local module import
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_oceanai_dir, get_project_root, load_config, set_seed, setup_logger

logger = setup_logger("obis_env")

# Headless plotting setup
LOCAL_CACHE_DIR = Path(__file__).resolve().parent.parent / ".matplotlib_cache"
LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(LOCAL_CACHE_DIR)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_and_flag_obis(config: dict, repo_root: Path) -> pd.DataFrame:
    """Load cleaned OBIS occurrences and annotate the UNESCO Sundarbans eDNA expedition."""
    obis_rel_path = config["obis"]["path"]
    csv_path = repo_root / obis_rel_path
    logger.info(f"Loading cleaned OBIS occurrences from: {csv_path}")
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing OBIS CSV at: {csv_path}")
        
    df_obis = pd.read_csv(csv_path)
    
    # Parse event dates
    df_obis["event_dt"] = pd.to_datetime(df_obis["eventDate"], errors="coerce", utc=True)
    df_obis["year_val"] = df_obis["event_dt"].dt.year.fillna(df_obis["date_year"])
    df_obis["month_dt"] = df_obis["event_dt"].dt.to_period("M").dt.to_timestamp()
    
    # Grid binning to 0.25° cells
    df_obis["lat_bin"] = np.round(np.floor(df_obis["decimalLatitude"] / 0.25) * 0.25, 2)
    df_obis["lon_bin"] = np.round(np.floor(df_obis["decimalLongitude"] / 0.25) * 0.25, 2)
    
    # Flag the UNESCO Sundarbans eDNA expedition
    edna_dates = config["obis"]["edna_expedition"]["dates"]
    edna_species = config["obis"]["edna_expedition"]["species"]
    
    is_edna = (df_obis["species"] == edna_species) & (
        df_obis["event_dt"].dt.strftime("%Y-%m-%d").isin(edna_dates)
    )
    df_obis["is_unesco_edna_expedition"] = is_edna.astype(int)
    
    edna_count = is_edna.sum()
    edna_share = edna_count / len(df_obis) * 100
    logger.info(f"Total OBIS records: {len(df_obis):,}")
    logger.info(f"UNESCO Sundarbans eDNA records flagged: {edna_count:,} ({edna_share:.2f}% of total OBIS dataset)")
    
    return df_obis


def match_ocean_and_fisheries(df_obis: pd.DataFrame, ocean_features_path: Path) -> pd.DataFrame:
    """Match each OBIS record with physical ocean variables and apparent fishing effort."""
    logger.info(f"Loading oceanographic and fisheries features from: {ocean_features_path.name}")
    if not ocean_features_path.exists():
        raise FileNotFoundError(f"Missing {ocean_features_path}. Run features.py first.")
        
    ocean_df = pd.read_parquet(ocean_features_path)
    
    # 1. Exact match on (month, lat_bin, lon_bin)
    logger.info("Performing spatio-temporal matching on (month, lat_bin, lon_bin)...")
    cols_to_pull = ["month", "lat_bin", "lon_bin", "thetao", "so", "speed", "sst_anomaly", "hours", "is_fishing"]
    
    merged = df_obis.merge(
        ocean_df[cols_to_pull],
        left_on=["month_dt", "lat_bin", "lon_bin"],
        right_on=["month", "lat_bin", "lon_bin"],
        how="left"
    )
    
    # 2. For coastal edge occurrences where exact month or cell is masked, interpolate via spatial nearest-neighbor
    unmatched_mask = merged["thetao"].isna()
    if unmatched_mask.sum() > 0:
        logger.info(f"Interpolating nearest ocean conditions for {unmatched_mask.sum():,} coastal fringe occurrences via cKDTree...")
        cell_means = ocean_df.groupby(["lat_bin", "lon_bin"])[["thetao", "so", "speed", "sst_anomaly", "hours", "is_fishing"]].mean().reset_index()
        tree = cKDTree(cell_means[["lat_bin", "lon_bin"]].values)
        
        unmatched_coords = merged.loc[unmatched_mask, ["lat_bin", "lon_bin"]].values
        _, idxs = tree.query(unmatched_coords)
        
        for col in ["thetao", "so", "speed", "sst_anomaly", "hours", "is_fishing"]:
            merged.loc[unmatched_mask, col] = cell_means.iloc[idxs][col].values
            
    # Standardize column naming
    matched = merged.rename(columns={
        "thetao": "temperature_c",
        "so": "salinity_psu",
        "speed": "current_speed_ms",
        "hours": "apparent_fishing_hours",
        "year_val": "year"
    })
    
    logger.info("Spatio-temporal matching completed successfully.")
    return matched


def compute_environmental_envelopes(df_matched: pd.DataFrame):
    """Compute realized environmental envelopes (median, IQR, ranges) per species."""
    logger.info("Calculating realized species environmental envelopes...")
    
    stats_list = []
    for sp in sorted(df_matched["species"].unique()):
        sub = df_matched[df_matched["species"] == sp]
        common = sub["common_name"].iloc[0] if "common_name" in sub and pd.notna(sub["common_name"].iloc[0]) else sp
        is_edna_share = sub["is_unesco_edna_expedition"].mean() * 100
        
        stats_list.append({
            "Species": sp,
            "Common_Name": common,
            "Total_Records": len(sub),
            "UNESCO_eDNA_Share_Pct": is_edna_share,
            "Temp_Median": float(sub["temperature_c"].median()),
            "Temp_IQR": float(sub["temperature_c"].quantile(0.75) - sub["temperature_c"].quantile(0.25)),
            "Temp_Min": float(sub["temperature_c"].min()),
            "Temp_Max": float(sub["temperature_c"].max()),
            "Salinity_Median": float(sub["salinity_psu"].median()),
            "Salinity_IQR": float(sub["salinity_psu"].quantile(0.75) - sub["salinity_psu"].quantile(0.25)),
            "Salinity_Min": float(sub["salinity_psu"].min()),
            "Salinity_Max": float(sub["salinity_psu"].max()),
            "Speed_Median": float(sub["current_speed_ms"].median()),
            "Speed_IQR": float(sub["current_speed_ms"].quantile(0.75) - sub["current_speed_ms"].quantile(0.25)),
            "Co_occurring_Fishing_Pct": float((sub["apparent_fishing_hours"] > 0).mean() * 100),
            "Mean_Fishing_Hours_In_Cell": float(sub["apparent_fishing_hours"].mean())
        })
        
    df_env = pd.DataFrame(stats_list)
    logger.info("Species Environmental Envelopes Summary:")
    for _, r in df_env.iterrows():
        logger.info(f"  {r['Species']} ({r['Common_Name']}): Temp={r['Temp_Median']:.1f}°C, Salinity={r['Salinity_Median']:.1f}PSU, Fishing Overlap={r['Co_occurring_Fishing_Pct']:.1f}%, eDNA={r['UNESCO_eDNA_Share_Pct']:.1f}%")
        
    return df_env


def generate_biodiversity_figures(df_matched: pd.DataFrame, ocean_features_path: Path, fig_dir: Path):
    """Generate multi-panel environmental envelope boxplots and spatial fishing overlay maps."""
    logger.info(f"Generating biodiversity figures in: {fig_dir}...")
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Multi-panel Environmental Envelopes Boxplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    species_order = sorted(df_matched["species"].unique())
    short_names = [s.split()[0][0] + ". " + s.split()[-1] for s in species_order]
    
    # Potential Temperature
    data_t = [df_matched[df_matched["species"] == s]["temperature_c"].dropna() for s in species_order]
    b1 = axes[0].boxplot(data_t, tick_labels=short_names, patch_artist=True)
    for patch in b1["boxes"]:
        patch.set_facecolor("#e6550d")
        patch.set_alpha(0.75)
    axes[0].set_title("Potential Temperature (°C)", fontweight="bold", fontsize=11)
    axes[0].set_ylabel("°C")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    
    # Salinity
    data_s = [df_matched[df_matched["species"] == s]["salinity_psu"].dropna() for s in species_order]
    b2 = axes[1].boxplot(data_s, tick_labels=short_names, patch_artist=True)
    for patch in b2["boxes"]:
        patch.set_facecolor("#3182bd")
        patch.set_alpha(0.75)
    axes[1].set_title("Salinity (PSU)", fontweight="bold", fontsize=11)
    axes[1].set_ylabel("PSU")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    
    # Current Speed
    data_v = [df_matched[df_matched["species"] == s]["current_speed_ms"].dropna() for s in species_order]
    b3 = axes[2].boxplot(data_v, tick_labels=short_names, patch_artist=True)
    for patch in b3["boxes"]:
        patch.set_facecolor("#31a354")
        patch.set_alpha(0.75)
    axes[2].set_title("Current Speed (m/s)", fontweight="bold", fontsize=11)
    axes[2].set_ylabel("m/s")
    axes[2].grid(axis="y", linestyle="--", alpha=0.5)
    
    plt.suptitle("Realized Environmental Envelopes per Species (OBIS 2015-2025)\n(Note: Setipinna reflects UNESCO Sundarbans eDNA survey)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / "obis_environmental_envelopes.png", dpi=300)
    plt.close()
    
    # 2. Spatial Overlay: Fishing Hotspots with Overlaid OBIS Occurrences
    ocean_df = pd.read_parquet(ocean_features_path)
    mean_fishing = ocean_df.groupby(["lat_bin", "lon_bin"])["hours"].mean().reset_index()
    
    plt.figure(figsize=(12, 7))
    sc = plt.scatter(
        mean_fishing["lon_bin"], mean_fishing["lat_bin"],
        c=np.log1p(mean_fishing["hours"]), cmap="Blues",
        s=10, marker="s", alpha=0.7, label="Mean Apparent Fishing Intensity [log1p(hours)]"
    )
    plt.colorbar(sc, label="Log1p(Mean Monthly Fishing Hours)")
    
    markers = {
        "Carcharhinus melanopterus": ("o", "red", "Blacktip Reef Shark"),
        "Rhincodon typus": ("^", "darkorange", "Whale Shark"),
        "Setipinna tenuifilis": ("*", "magenta", "Setipinna (UNESCO eDNA Cruise)")
    }
    
    for sp, (marker, color, label) in markers.items():
        sub = df_matched[df_matched["species"] == sp]
        if len(sub) > 0:
            plt.scatter(
                sub["decimalLongitude"], sub["decimalLatitude"],
                marker=marker, color=color, edgecolor="black", s=55,
                alpha=0.85, label=f"{label} (n={len(sub)})"
            )
            
    plt.xlabel("Longitude (°E)", fontsize=11)
    plt.ylabel("Latitude (°N)", fontsize=11)
    plt.title("Spatial Overlay: OBIS Marine Species Occurrences vs GFW Apparent Fishing Hotspots", fontsize=12, fontweight="bold")
    plt.legend(loc="lower left", fontsize=9, framealpha=0.9)
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(fig_dir / "obis_fishing_overlay_map.png", dpi=300)
    plt.close()
    
    # 3. Focus Map: UNESCO Sundarbans eDNA Sampling Sites
    seti_sub = df_matched[df_matched["species"] == "Setipinna tenuifilis"]
    plt.figure(figsize=(8, 6))
    plt.scatter(seti_sub["decimalLongitude"], seti_sub["decimalLatitude"], c="#7a0177", s=80, edgecolor="black", alpha=0.9, label="eDNA Station (8-9 Jan 2023)")
    plt.title("UNESCO Sundarbans eDNA Expedition (39.34% of OBIS Dataset)\nSampling Stations for Setipinna tenuifilis", fontsize=11, fontweight="bold")
    plt.xlabel("Longitude (°E)", fontsize=10)
    plt.ylabel("Latitude (°N)", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_dir / "unesco_edna_sundarbans_map.png", dpi=300)
    plt.close()
    
    logger.info("Biodiversity figures generated successfully.")


def main():
    parser = argparse.ArgumentParser(description="Match OBIS biodiversity records with ocean conditions and GFW fishing.")
    parser.add_argument("--features-file", type=str, default=None, help="Path to features.parquet.")
    parser.add_argument("--config", type=str, default=None, help="Path to regions.yaml configuration.")
    args = parser.parse_args()
    
    config = load_config(Path(args.config) if args.config else None)
    seed = config["splits"]["random_seed"]
    set_seed(seed)
    
    repo_root = get_project_root()
    oceanai_dir = get_oceanai_dir()
    
    features_path = Path(args.features_file) if args.features_file else oceanai_dir / "data" / "processed" / "integrated" / "features.parquet"
    out_dir = oceanai_dir / "data" / "processed" / "integrated"
    metrics_dir = oceanai_dir / "reports" / "metrics"
    figures_dir = oceanai_dir / "reports" / "figures"
    
    df_obis = load_and_flag_obis(config, repo_root)
    df_matched = match_ocean_and_fisheries(df_obis, features_path)
    df_env = compute_environmental_envelopes(df_matched)
    
    # Save Matched Dataset & Envelopes CSV
    matched_csv = out_dir / "obis_matched_ocean.csv"
    keep_cols = [
        "id", "species", "common_name", "eventDate", "year", "decimalLatitude", "decimalLongitude",
        "lat_bin", "lon_bin", "is_unesco_edna_expedition", "temperature_c", "salinity_psu",
        "current_speed_ms", "sst_anomaly", "apparent_fishing_hours", "is_fishing"
    ]
    df_matched[keep_cols].to_csv(matched_csv, index=False)
    logger.info(f"Saved matched OBIS records to: {matched_csv}")
    
    env_csv = metrics_dir / "obis_species_envelopes.csv"
    df_env.to_csv(env_csv, index=False)
    logger.info(f"Saved species environmental envelopes to: {env_csv}")
    
    # Generate Figures
    generate_biodiversity_figures(df_matched, features_path, figures_dir)
    
    logger.info("=== SCIENTIFIC CAVEAT CONFIRMATION ===")
    logger.info("Species Distribution Modeling (SDM) was deliberately skipped.")
    logger.info("39.34% of all records originate from one single 2-day eDNA cruise (UNESCO Sundarbans).")
    logger.info("Fitting SDMs under this extreme sampling bias would model cruise tracks rather than true habitat suitability.")


if __name__ == "__main__":
    main()
