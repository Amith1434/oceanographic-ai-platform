"""
Feature Engineering Pipeline
============================
Transforms raw harmonized grid cells into machine-learning ready features:
1. Cyclical seasonality harmonics: sin(2π·m/12), cos(2π·m/12).
2. Spatial geometry: distance to coast (km) via cKDTree on land-mask cells.
3. Cell-level monthly climatological anomalies for SST and salinity.
4. One-month autoregressive lags for ocean state and apparent fishing.
5. Strict guardrail: Calendar year is NEVER included in feature spaces to prevent
   exploiting the 2022 GFW AIS satellite coverage jump artifact.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import xarray as xr

# Local module import
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_oceanai_dir, get_project_root, load_config, set_seed, setup_logger

logger = setup_logger("features")


def compute_distance_to_coast(df_integrated: pd.DataFrame, config: dict, repo_root: Path) -> pd.DataFrame:
    """Extract land-mask boundaries from original Copernicus grids and compute distance to coast in km."""
    logger.info("Extracting Copernicus land mask cells for distance-to-coast calculation...")
    
    land_coords = []
    for reg_key, reg_cfg in config["regions"].items():
        nc_path = repo_root / reg_cfg["copernicus_path"]
        ds = xr.open_dataset(nc_path).isel(depth=0, time=0)
        coarsen_factor = config["grid"]["coarsen_factor"]
        ds_coarse = ds.coarsen(latitude=coarsen_factor, longitude=coarsen_factor, boundary="trim").mean()
        grid_df = ds_coarse["thetao"].to_dataframe().reset_index()
        land = grid_df[grid_df["thetao"].isna()][["latitude", "longitude"]].values
        if len(land) > 0:
            land_coords.append(land)
            
    all_land = np.vstack(land_coords)
    logger.info(f"Constructing KD-Tree on {len(all_land):,} land-mask boundary points...")
    tree = cKDTree(all_land)
    
    unique_cells = df_integrated[["lat_bin", "lon_bin"]].drop_duplicates().copy()
    coords = unique_cells[["lat_bin", "lon_bin"]].values
    
    # Query distance in degrees, convert to kilometers (1° ≈ 111.12 km)
    distances_deg, _ = tree.query(coords)
    unique_cells["dist_to_coast_km"] = distances_deg * 111.12
    
    merged = df_integrated.merge(unique_cells, on=["lat_bin", "lon_bin"], how="left")
    logger.info(f"Distance to coast computed: min={merged['dist_to_coast_km'].min():.1f}km, mean={merged['dist_to_coast_km'].mean():.1f}km, max={merged['dist_to_coast_km'].max():.1f}km.")
    return merged


def engineer_features(df: pd.DataFrame, config: dict, repo_root: Path) -> pd.DataFrame:
    """Generate all spatial, seasonal, oceanographic, anomaly, and lag features."""
    logger.info("Starting multi-domain feature construction...")
    df = df.copy()
    
    # 1. Temporal & Cyclical Harmonics
    df["month"] = pd.to_datetime(df["month"])
    df["year"] = df["month"].dt.year
    df["cal_month"] = df["month"].dt.month
    
    # Cyclical monsoon harmonics
    df["sin_month"] = np.sin(2 * np.pi * df["cal_month"] / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * df["cal_month"] / 12.0)
    
    # 2. Region Categorical Indicator
    df["region_id"] = (df["region"] == "bay_of_bengal").astype(int)
    
    # 3. Spatial Geometry: Distance to Coast
    df = compute_distance_to_coast(df, config, repo_root)
    
    # 4. Cell-Level Monthly Climatologies & Physical Anomalies (TRAIN-PERIOD ONLY: 2019-2023)
    logger.info("Computing cell-specific monthly climatologies from TRAIN PERIOD (2019-2023) to prevent leakage...")
    train_mask = df["year"] <= 2023
    clim_train = (
        df[train_mask]
        .groupby(["region", "lat_bin", "lon_bin", "cal_month"])[["thetao", "so"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    clim_train.columns = [
        "region", "lat_bin", "lon_bin", "cal_month",
        "clim_thetao_mean", "clim_thetao_std",
        "clim_so_mean", "clim_so_std"
    ]
    # Replace zero or NaN standard deviations with fallback 1.0
    clim_train["clim_thetao_std"] = clim_train["clim_thetao_std"].fillna(1.0).replace(0.0, 1.0)
    clim_train["clim_so_std"] = clim_train["clim_so_std"].fillna(1.0).replace(0.0, 1.0)
    
    # Merge train climatologies onto all years (train, val, test)
    df = df.merge(clim_train, on=["region", "lat_bin", "lon_bin", "cal_month"], how="left")
    
    # Model features (leak-free: relative to 2019-2023 baseline)
    df["sst_anomaly"] = df["thetao"] - df["clim_thetao_mean"]
    df["so_anomaly"] = df["so"] - df["clim_so_mean"]
    df["sst_zscore"] = df["sst_anomaly"] / df["clim_thetao_std"]
    
    # Also compute full-period climatologies and anomalies for descriptive anomaly mapping only
    logger.info("Computing separate full-period (2019-2025) anomalies for descriptive mapping...")
    clim_full = (
        df.groupby(["region", "lat_bin", "lon_bin", "cal_month"])[["thetao", "so"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    clim_full.columns = [
        "region", "lat_bin", "lon_bin", "cal_month",
        "clim_thetao_mean_fullperiod", "clim_thetao_std_fullperiod",
        "clim_so_mean_fullperiod", "clim_so_std_fullperiod"
    ]
    clim_full["clim_thetao_std_fullperiod"] = clim_full["clim_thetao_std_fullperiod"].fillna(1.0).replace(0.0, 1.0)
    clim_full["clim_so_std_fullperiod"] = clim_full["clim_so_std_fullperiod"].fillna(1.0).replace(0.0, 1.0)
    
    df = df.merge(clim_full, on=["region", "lat_bin", "lon_bin", "cal_month"], how="left")
    df["sst_anomaly_fullperiod"] = df["thetao"] - df["clim_thetao_mean_fullperiod"]
    df["so_anomaly_fullperiod"] = df["so"] - df["clim_so_mean_fullperiod"]
    df["sst_zscore_fullperiod"] = df["sst_anomaly_fullperiod"] / df["clim_thetao_std_fullperiod"]
    
    # 5. One-Month Lagged Features
    logger.info("Calculating 1-month autoregressive lags for ocean state and fishing...")
    df = df.sort_values(["region", "lat_bin", "lon_bin", "month"]).reset_index(drop=True)
    
    grouped_cells = df.groupby(["region", "lat_bin", "lon_bin"])
    df["sst_lag1"] = grouped_cells["thetao"].shift(1)
    df["so_lag1"] = grouped_cells["so"].shift(1)
    df["speed_lag1"] = grouped_cells["speed"].shift(1)
    df["hours_lag1"] = grouped_cells["hours"].shift(1)
    
    # Impute first month (2019-01) with cell climatological values / 0 to preserve data
    df["sst_lag1"] = df["sst_lag1"].fillna(df["clim_thetao_mean"])
    df["so_lag1"] = df["so_lag1"].fillna(df["clim_so_mean"])
    df["speed_lag1"] = df["speed_lag1"].fillna(df["speed"])
    df["hours_lag1"] = df["hours_lag1"].fillna(0.0)
    df["is_fishing_lag1"] = (df["hours_lag1"] > 0).astype(int)
    
    # 6. Targets
    df["is_fishing"] = (df["hours"] > 0).astype(int)
    df["log_hours"] = np.log1p(df["hours"])
    
    # Drop raw intermediate climatology columns
    df = df.drop(columns=[
        "clim_thetao_mean", "clim_thetao_std", "clim_so_mean", "clim_so_std",
        "clim_thetao_mean_fullperiod", "clim_thetao_std_fullperiod",
        "clim_so_mean_fullperiod", "clim_so_std_fullperiod"
    ])
    
    # 7. Methodological Guardrail Verification
    feature_cols = config["features"]["spatial_seasonal"] + config["features"]["ocean_physical"]
    assert "year" not in feature_cols, "CRITICAL ERROR: 'year' must not be in the model feature list!"
    logger.info("Guardrail verified: 'year' is strictly excluded from feature sets (avoids 2022 coverage jump artifact).")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Engineer multi-domain features on integrated grid dataset.")
    parser.add_argument("--input-file", type=str, default=None, help="Input integrated.parquet path.")
    parser.add_argument("--output-file", type=str, default=None, help="Output features.parquet path.")
    parser.add_argument("--config", type=str, default=None, help="Path to regions.yaml configuration.")
    args = parser.parse_args()
    
    set_seed(42)
    config = load_config(Path(args.config) if args.config else None)
    repo_root = get_project_root()
    oceanai_dir = get_oceanai_dir()
    
    input_path = Path(args.input_file) if args.input_file else oceanai_dir / "data" / "processed" / "integrated" / "integrated.parquet"
    output_path = Path(args.output_file) if args.output_file else oceanai_dir / "data" / "processed" / "integrated" / "features.parquet"
    
    if not input_path.exists():
        raise FileNotFoundError(f"Missing {input_path}. Please run integrate.py first.")
        
    logger.info(f"Loading integrated table from: {input_path}")
    df_integrated = pd.read_parquet(input_path)
    
    df_featured = engineer_features(df_integrated, config, repo_root)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_featured.to_parquet(output_path, index=False, compression="snappy")
    logger.info(f"Saved featured dataset ({len(df_featured):,} rows, {len(df_featured.columns)} columns) to: {output_path}")


if __name__ == "__main__":
    main()
