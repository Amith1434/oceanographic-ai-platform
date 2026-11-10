"""
OBIS Marine Biodiversity Environmental Matching & Realized Envelopes
=====================================================================
Matches OBIS marine species occurrence records (2015-2025) with Copernicus physical
ocean conditions (potential temperature, salinity, currents) and GFW apparent fishing.

Key Scientific & Analytical Guardrails:
1. Prominently flags and documents the UNESCO Sundarbans eDNA expedition in Bangladesh:
   - Dataset: "UNESCO eDNA expedition in The Sundarbans (Bangladesh): February 2023"
   - Dates: 8-9 Jan 2023, 19 sampling sites, Sundarbans West Wildlife Sanctuary (Bangladesh)
   - DOI: 10.25607/rt4e3h
   - Exactly 476 of 1,210 records (39.34% of total dataset; 100% of Setipinna tenuifilis)
   - eDNA reads denote molecular sequence presence, NOT individual abundance.
2. Rigorous 4-category Spatio-Temporal Match Classification:
   - 'exact_match': 2019+ and exact cell-month exists in Copernicus ocean grid.
   - 'nearest_cell_same_month': 2019+, within 0.5° of an ocean cell in that SAME month.
   - 'no_match_pre2019': year < 2019 (Copernicus/GFW period begins in 2019).
   - 'no_match_far': distance > 0.5° from any ocean cell in that month.
   - ONLY 'exact_match' and 'nearest_cell_same_month' are used for environmental envelopes
     and fishing overlap statistics. Hours are NEVER imputed with all-month averages.
3. Explicitly omits Species Distribution Modeling (SDM) to avoid modeling research cruise tracks.
4. Caveat: The 0.25° grid cannot resolve narrow estuarine channels; Setipinna salinity is nearest-cell
   and is consistent with, but not verified as, estuarine runoff.
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
    """Load cleaned OBIS occurrences and annotate the UNESCO Sundarbans (Bangladesh) eDNA expedition."""
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
    
    # Flag the UNESCO Sundarbans (Bangladesh) eDNA expedition
    edna_dates = config["obis"]["edna_expedition"]["dates"]
    edna_species = config["obis"]["edna_expedition"]["species"]
    
    is_edna = (df_obis["species"] == edna_species) & (
        df_obis["event_dt"].dt.strftime("%Y-%m-%d").isin(edna_dates)
    )
    df_obis["is_unesco_edna_expedition"] = is_edna.astype(int)
    
    edna_count = int(is_edna.sum())
    edna_share = float(edna_count / len(df_obis) * 100)
    
    # Assertions to ensure data integrity
    assert edna_count == 476, f"Expected 476 eDNA records, got {edna_count}"
    assert abs(edna_share - 39.3388) < 0.01, f"Expected 39.34% share, got {edna_share:.2f}%"
    
    edna_subset = df_obis[is_edna]
    edna_bbox = {
        "lat_min": float(edna_subset["decimalLatitude"].min()),
        "lat_max": float(edna_subset["decimalLatitude"].max()),
        "lon_min": float(edna_subset["decimalLongitude"].min()),
        "lon_max": float(edna_subset["decimalLongitude"].max()),
    }
    
    logger.info(f"Total OBIS records: {len(df_obis):,}")
    logger.info(f"UNESCO Sundarbans (Bangladesh) eDNA records: {edna_count:,} ({edna_share:.2f}% of total OBIS dataset)")
    logger.info(f"Sundarbans eDNA Bounding Box: Lat {edna_bbox['lat_min']:.6f}–{edna_bbox['lat_max']:.6f}°N, Lon {edna_bbox['lon_min']:.6f}–{edna_bbox['lon_max']:.6f}°E")
    
    return df_obis, edna_bbox


def match_ocean_and_fisheries(df_obis: pd.DataFrame, ocean_features_path: Path):
    """
    Match each OBIS record with physical ocean variables and apparent fishing effort
    using a rigorous 4-category classification:
    - exact_match: 2019+ and cell-month exists in ocean grid
    - nearest_cell_same_month: 2019+, within 0.5° of an ocean cell in that SAME month
    - no_match_pre2019: year < 2019
    - no_match_far: distance > 0.5° or no valid ocean cell
    """
    logger.info(f"Loading oceanographic and fisheries features from: {ocean_features_path.name}")
    if not ocean_features_path.exists():
        raise FileNotFoundError(f"Missing {ocean_features_path}. Run features.py first.")
        
    ocean_df = pd.read_parquet(ocean_features_path)
    ocean_cells = ocean_df[["month", "lat_bin", "lon_bin", "thetao", "so", "speed", "sst_anomaly", "hours", "is_fishing"]].drop_duplicates()
    
    # Initialize match classification and result columns
    df_matched = df_obis.copy()
    df_matched["match_category"] = "no_match_far"
    df_matched["temperature_c"] = np.nan
    df_matched["salinity_psu"] = np.nan
    df_matched["current_speed_ms"] = np.nan
    df_matched["sst_anomaly_val"] = np.nan
    df_matched["apparent_fishing_hours"] = np.nan
    df_matched["is_fishing_matched"] = np.nan
    df_matched["match_dist_deg"] = np.nan
    
    # 1. Pre-2019 records
    pre2019_mask = df_matched["year_val"] < 2019
    df_matched.loc[pre2019_mask, "match_category"] = "no_match_pre2019"
    
    # 2. Exact Spatio-temporal matching for 2019+
    post2019_indices = df_matched.index[~pre2019_mask].tolist()
    post2019_df = df_matched.loc[post2019_indices]
    
    merged_exact = post2019_df.merge(
        ocean_cells,
        left_on=["month_dt", "lat_bin", "lon_bin"],
        right_on=["month", "lat_bin", "lon_bin"],
        how="left"
    )
    
    exact_mask = ~merged_exact["thetao"].isna()
    exact_post_indices = post2019_df.index[exact_mask.values].tolist()
    
    df_matched.loc[exact_post_indices, "match_category"] = "exact_match"
    df_matched.loc[exact_post_indices, "temperature_c"] = merged_exact.loc[exact_mask, "thetao"].values
    df_matched.loc[exact_post_indices, "salinity_psu"] = merged_exact.loc[exact_mask, "so"].values
    df_matched.loc[exact_post_indices, "current_speed_ms"] = merged_exact.loc[exact_mask, "speed"].values
    df_matched.loc[exact_post_indices, "sst_anomaly_val"] = merged_exact.loc[exact_mask, "sst_anomaly"].values
    df_matched.loc[exact_post_indices, "apparent_fishing_hours"] = merged_exact.loc[exact_mask, "hours"].values
    df_matched.loc[exact_post_indices, "is_fishing_matched"] = merged_exact.loc[exact_mask, "is_fishing"].values
    df_matched.loc[exact_post_indices, "match_dist_deg"] = 0.0
    
    # 3. Nearest-cell matching within SAME MONTH (<= 0.5 degrees)
    unmatched_post_indices = post2019_df.index[(~exact_mask).values].tolist()
    logger.info(f"Exact matches: {len(exact_post_indices):,}. Checking nearest-cell in same month for {len(unmatched_post_indices):,} occurrences...")
    
    months_with_unmatched = df_matched.loc[unmatched_post_indices, "month_dt"].dropna().unique()
    
    for m in months_with_unmatched:
        sub_unmatched = df_matched.loc[unmatched_post_indices]
        sub_m = sub_unmatched[sub_unmatched["month_dt"] == m]
        sub_ocean = ocean_cells[ocean_cells["month"] == m]
        
        if len(sub_ocean) == 0:
            continue
            
        tree = cKDTree(sub_ocean[["lat_bin", "lon_bin"]].values)
        dists, idxs = tree.query(sub_m[["lat_bin", "lon_bin"]].values)
        
        for idx_row, dist_deg, ocean_idx in zip(sub_m.index, dists, idxs):
            if dist_deg <= 0.5:
                matched_row = sub_ocean.iloc[ocean_idx]
                df_matched.loc[idx_row, "match_category"] = "nearest_cell_same_month"
                df_matched.loc[idx_row, "temperature_c"] = matched_row["thetao"]
                df_matched.loc[idx_row, "salinity_psu"] = matched_row["so"]
                df_matched.loc[idx_row, "current_speed_ms"] = matched_row["speed"]
                df_matched.loc[idx_row, "sst_anomaly_val"] = matched_row["sst_anomaly"]
                df_matched.loc[idx_row, "apparent_fishing_hours"] = matched_row["hours"]
                df_matched.loc[idx_row, "is_fishing_matched"] = matched_row["is_fishing"]
                df_matched.loc[idx_row, "match_dist_deg"] = dist_deg
            else:
                df_matched.loc[idx_row, "match_category"] = "no_match_far"
                df_matched.loc[idx_row, "match_dist_deg"] = dist_deg
                
    df_matched["year"] = df_matched["year_val"]
    
    # Build match quality table
    species_order = sorted(df_matched["species"].unique())
    quality_rows = []
    for sp in species_order:
        sub = df_matched[df_matched["species"] == sp]
        common = sub["common_name"].iloc[0] if "common_name" in sub and pd.notna(sub["common_name"].iloc[0]) else sp
        n_total = len(sub)
        n_pre2019 = int((sub["match_category"] == "no_match_pre2019").sum())
        n_exact = int((sub["match_category"] == "exact_match").sum())
        n_nearest = int((sub["match_category"] == "nearest_cell_same_month").sum())
        n_far = int((sub["match_category"] == "no_match_far").sum())
        n_dropped = n_pre2019 + n_far
        n_used = n_exact + n_nearest
        
        quality_rows.append({
            "Species": sp,
            "Common_Name": common,
            "Total_Records": n_total,
            "Pre_2019": n_pre2019,
            "Exact_Match": n_exact,
            "Nearest_Same_Month": n_nearest,
            "Far_Dropped": n_far,
            "Dropped_Records": n_dropped,
            "Used_Records": n_used,
            "Used_Share_Pct": round(n_used / n_total * 100, 2)
        })
        
    df_quality = pd.DataFrame(quality_rows)
    logger.info("Match Quality Summary by Species:")
    for _, r in df_quality.iterrows():
        logger.info(f"  {r['Species']}: Total={r['Total_Records']}, Pre2019={r['Pre_2019']}, Exact={r['Exact_Match']}, Nearest={r['Nearest_Same_Month']}, Dropped={r['Dropped_Records']}, Used={r['Used_Records']}")
        
    return df_matched, df_quality


def compute_environmental_envelopes(df_matched: pd.DataFrame):
    """
    Compute realized environmental envelopes (median, IQR, ranges) and fishing overlap
    using ONLY validated matches (exact_match and nearest_cell_same_month).
    """
    logger.info("Calculating realized species environmental envelopes on used matches only...")
    used_mask = df_matched["match_category"].isin(["exact_match", "nearest_cell_same_month"])
    df_used = df_matched[used_mask].copy()
    
    stats_list = []
    for sp in sorted(df_matched["species"].unique()):
        sub_all = df_matched[df_matched["species"] == sp]
        sub = df_used[df_used["species"] == sp]
        common = sub_all["common_name"].iloc[0] if "common_name" in sub_all and pd.notna(sub_all["common_name"].iloc[0]) else sp
        is_edna_share = sub_all["is_unesco_edna_expedition"].mean() * 100
        
        if len(sub) == 0:
            continue
            
        stats_list.append({
            "Species": sp,
            "Common_Name": common,
            "Total_Records": len(sub_all),
            "Matched_Records_Used": len(sub),
            "UNESCO_eDNA_Share_Pct": round(is_edna_share, 2),
            "Temp_Median": round(float(sub["temperature_c"].median()), 2),
            "Temp_IQR": round(float(sub["temperature_c"].quantile(0.75) - sub["temperature_c"].quantile(0.25)), 2),
            "Temp_Min": round(float(sub["temperature_c"].min()), 2),
            "Temp_Max": round(float(sub["temperature_c"].max()), 2),
            "Salinity_Median": round(float(sub["salinity_psu"].median()), 2),
            "Salinity_IQR": round(float(sub["salinity_psu"].quantile(0.75) - sub["salinity_psu"].quantile(0.25)), 2),
            "Salinity_Min": round(float(sub["salinity_psu"].min()), 2),
            "Salinity_Max": round(float(sub["salinity_psu"].max()), 2),
            "Speed_Median": round(float(sub["current_speed_ms"].median()), 3),
            "Speed_IQR": round(float(sub["current_speed_ms"].quantile(0.75) - sub["current_speed_ms"].quantile(0.25)), 3),
            "Co_occurring_Fishing_Pct": round(float((sub["apparent_fishing_hours"] > 0).mean() * 100), 2),
            "Mean_Fishing_Hours_In_Cell": round(float(sub["apparent_fishing_hours"].mean()), 2)
        })
        
    df_env = pd.DataFrame(stats_list)
    logger.info("Realized Environmental Envelopes Summary (Matched Records Only):")
    for _, r in df_env.iterrows():
        logger.info(f"  {r['Species']} ({r['Common_Name']}): n_matched={r['Matched_Records_Used']}/{r['Total_Records']}, Temp={r['Temp_Median']}°C, Salinity={r['Salinity_Median']}PSU, Fishing Overlap={r['Co_occurring_Fishing_Pct']}%, eDNA Share={r['UNESCO_eDNA_Share_Pct']}%")
        
    return df_env


def generate_biodiversity_figures(df_matched: pd.DataFrame, ocean_features_path: Path, fig_dir: Path, edna_bbox: dict):
    """Generate multi-panel environmental envelope boxplots and spatial fishing overlay maps."""
    logger.info(f"Generating biodiversity figures in: {fig_dir}...")
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    used_mask = df_matched["match_category"].isin(["exact_match", "nearest_cell_same_month"])
    df_used = df_matched[used_mask].copy()
    
    # 1. Multi-panel Environmental Envelopes Boxplots (Used Records Only)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    species_order = sorted(df_used["species"].unique())
    short_names = [s.split()[0][0] + ". " + s.split()[-1] for s in species_order]
    
    # Potential Temperature
    data_t = [df_used[df_used["species"] == s]["temperature_c"].dropna() for s in species_order]
    b1 = axes[0].boxplot(data_t, tick_labels=short_names, patch_artist=True)
    for patch in b1["boxes"]:
        patch.set_facecolor("#e6550d")
        patch.set_alpha(0.75)
    axes[0].set_title("Potential Temperature (°C)", fontweight="bold", fontsize=11)
    axes[0].set_ylabel("°C")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    
    # Salinity
    data_s = [df_used[df_used["species"] == s]["salinity_psu"].dropna() for s in species_order]
    b2 = axes[1].boxplot(data_s, tick_labels=short_names, patch_artist=True)
    for patch in b2["boxes"]:
        patch.set_facecolor("#3182bd")
        patch.set_alpha(0.75)
    axes[1].set_title("Salinity (PSU)", fontweight="bold", fontsize=11)
    axes[1].set_ylabel("PSU")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    
    # Current Speed
    data_v = [df_used[df_used["species"] == s]["current_speed_ms"].dropna() for s in species_order]
    b3 = axes[2].boxplot(data_v, tick_labels=short_names, patch_artist=True)
    for patch in b3["boxes"]:
        patch.set_facecolor("#31a354")
        patch.set_alpha(0.75)
    axes[2].set_title("Current Speed (m/s)", fontweight="bold", fontsize=11)
    axes[2].set_ylabel("m/s")
    axes[2].grid(axis="y", linestyle="--", alpha=0.5)
    
    plt.suptitle("Realized Environmental Envelopes (Matched 2019-2025 Ocean Cells)\n(Note: Setipinna reflects UNESCO Sundarbans eDNA survey in Bangladesh)", fontsize=12, fontweight="bold")
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
        "Setipinna tenuifilis": ("*", "magenta", "Setipinna (UNESCO Sundarbans eDNA, Bangladesh)")
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
    
    # 3. Focus Map: UNESCO Sundarbans (Bangladesh) eDNA Sampling Sites
    seti_sub = df_matched[df_matched["species"] == "Setipinna tenuifilis"]
    plt.figure(figsize=(8, 6))
    plt.scatter(seti_sub["decimalLongitude"], seti_sub["decimalLatitude"], c="#7a0177", s=80, edgecolor="black", alpha=0.9, label="eDNA Station (8-9 Jan 2023)")
    plt.title(
        f"UNESCO eDNA Expedition: Sundarbans West Wildlife Sanctuary (Bangladesh)\n"
        f"Lat {edna_bbox['lat_min']:.4f}°–{edna_bbox['lat_max']:.4f}°N, Lon {edna_bbox['lon_min']:.4f}°–{edna_bbox['lon_max']:.4f}°E (n={len(seti_sub)})",
        fontsize=11, fontweight="bold"
    )
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
    
    df_obis, edna_bbox = load_and_flag_obis(config, repo_root)
    df_matched, df_quality = match_ocean_and_fisheries(df_obis, features_path)
    df_env = compute_environmental_envelopes(df_matched)
    
    # Save Matched Dataset & Envelopes CSV
    matched_csv = out_dir / "obis_matched_ocean.csv"
    keep_cols = [
        "id", "species", "common_name", "eventDate", "year", "decimalLatitude", "decimalLongitude",
        "lat_bin", "lon_bin", "match_category", "match_dist_deg", "is_unesco_edna_expedition",
        "temperature_c", "salinity_psu", "current_speed_ms", "sst_anomaly_val",
        "apparent_fishing_hours", "is_fishing_matched"
    ]
    df_matched[keep_cols].to_csv(matched_csv, index=False)
    logger.info(f"Saved matched OBIS records to: {matched_csv}")
    
    quality_csv = metrics_dir / "obis_match_quality.csv"
    df_quality.to_csv(quality_csv, index=False)
    logger.info(f"Saved OBIS match quality table to: {quality_csv}")
    
    env_csv = metrics_dir / "obis_species_envelopes.csv"
    df_env.to_csv(env_csv, index=False)
    logger.info(f"Saved species environmental envelopes to: {env_csv}")
    
    # Generate Figures
    generate_biodiversity_figures(df_matched, features_path, figures_dir, edna_bbox)
    
    logger.info("=== SCIENTIFIC CAVEAT CONFIRMATION ===")
    logger.info("Species Distribution Modeling (SDM) was deliberately skipped.")
    logger.info("39.34% of all records originate from one single 2-day eDNA cruise (UNESCO Sundarbans, Bangladesh).")
    logger.info("Fitting SDMs under this extreme sampling bias would model cruise tracks rather than true habitat suitability.")


if __name__ == "__main__":
    main()
