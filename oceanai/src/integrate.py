"""
Data Integration Pipeline: Copernicus Reanalysis + Global Fishing Watch
========================================================================
Harmonizes monthly physical oceanographic variables with AIS-derived apparent
fishing effort onto a unified 0.25° spatial grid for the Arabian Sea and Bay of Bengal.

Key Rules:
1. Slices shallowest surface layer from Copernicus (depth = 0).
2. Calculates current velocity magnitude speed = sqrt(uo^2 + vo^2).
3. Coarsens Copernicus by 3x3 factor to match ~0.25° resolution.
4. Aggregates GFW apparent fishing hours per cell-month across all flags.
   Does NOT sum vesselIDs to prevent duplicate counts.
5. Zero-filling: Water cells with no GFW detection represent zero fishing hours,
   NOT missing values.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

# Local module import
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_oceanai_dir, get_project_root, load_config, set_seed, setup_logger

logger = setup_logger("integrate")


def process_copernicus(nc_path: Path, coarsen_factor: int = 3, depth_idx: int = 0) -> pd.DataFrame:
    """Load, slice surface level, calculate current speed, coarsen, and bin Copernicus ocean data."""
    logger.info(f"Loading Copernicus physical reanalysis from: {nc_path.name}")
    if not nc_path.exists():
        raise FileNotFoundError(f"Missing Copernicus NetCDF: {nc_path}")
        
    ds = xr.open_dataset(nc_path).isel(depth=depth_idx)
    
    # Calculate surface current speed
    ds["speed"] = np.hypot(ds.uo, ds.vo)
    
    # Coarsen from ~0.083° to ~0.25° via block averaging
    ds = ds.coarsen(latitude=coarsen_factor, longitude=coarsen_factor, boundary="trim").mean()
    
    # Convert to dataframe and drop land mask cells
    df_oc = ds[["thetao", "so", "uo", "vo", "speed"]].to_dataframe().reset_index()
    df_oc = df_oc.dropna(subset=["thetao"]).copy()
    
    # Standardize time and spatial binning
    df_oc["month"] = pd.to_datetime(df_oc["time"]).dt.to_period("M").dt.to_timestamp()
    df_oc["lat_bin"] = np.round(np.floor(df_oc["latitude"] / 0.25) * 0.25, 2)
    df_oc["lon_bin"] = np.round(np.floor(df_oc["longitude"] / 0.25) * 0.25, 2)
    
    # Aggregate to cell-month
    aggregated = df_oc.groupby(
        ["month", "lat_bin", "lon_bin"], as_index=False
    )[["thetao", "so", "uo", "vo", "speed"]].mean()
    
    logger.info(f"Copernicus coarsened: {len(aggregated):,} valid ocean cell-months.")
    return aggregated


def process_gfw(csv_path: Path) -> pd.DataFrame:
    """Load and aggregate GFW fishing hours per 0.25° cell-month."""
    logger.info(f"Loading GFW fishing activity from: {csv_path.name}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing GFW CSV: {csv_path}")
        
    df_gfw = pd.read_csv(csv_path)
    
    df_gfw["month"] = pd.to_datetime(df_gfw["date"]).dt.to_period("M").dt.to_timestamp()
    df_gfw["lat_bin"] = np.round(np.floor(df_gfw["lat"] / 0.25) * 0.25, 2)
    df_gfw["lon_bin"] = np.round(np.floor(df_gfw["lon"] / 0.25) * 0.25, 2)
    
    # Sum hours across all flags (never sum vesselIDs)
    aggregated = df_gfw.groupby(
        ["month", "lat_bin", "lon_bin"], as_index=False
    )["hours"].sum()
    
    logger.info(f"GFW binned: {len(aggregated):,} cells with detected fishing activity.")
    return aggregated


def integrate_region(reg_key: str, reg_cfg: dict, repo_root: Path, coarsen_factor: int, depth_idx: int) -> pd.DataFrame:
    """Perform zero-filled integration for a single geographic region."""
    logger.info(f"=== Integrating Region: {reg_cfg['name']} ({reg_key}) ===")
    
    copernicus_path = repo_root / reg_cfg["copernicus_path"]
    gfw_path = repo_root / reg_cfg["gfw_path"]
    
    df_oc = process_copernicus(copernicus_path, coarsen_factor=coarsen_factor, depth_idx=depth_idx)
    df_gfw = process_gfw(gfw_path)
    
    # Left join ocean grid onto fishing detections
    logger.info("Merging ocean grid with fishing activity (Left Join + Zero-Filling)...")
    merged = df_oc.merge(df_gfw, how="left", on=["month", "lat_bin", "lon_bin"])
    
    # Zero-filling: No GFW detection = 0 apparent fishing hours
    merged["hours"] = merged["hours"].fillna(0.0)
    merged["region"] = reg_key
    
    fished_cells = (merged["hours"] > 0).sum()
    fished_pct = (merged["hours"] > 0).mean() * 100
    logger.info(f"Region {reg_key} total cells: {len(merged):,} | Fished cells: {fished_cells:,} ({fished_pct:.2f}%)")
    logger.info(f"Region {reg_key} total apparent hours: {merged['hours'].sum():,.2f}")
    
    return merged


def main():
    parser = argparse.ArgumentParser(description="Build integrated 0.25° Copernicus + GFW dataset.")
    parser.add_argument("--config", type=str, default=None, help="Path to regions.yaml configuration.")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory.")
    args = parser.parse_args()
    
    set_seed(42)
    config = load_config(Path(args.config) if args.config else None)
    repo_root = get_project_root()
    oceanai_dir = get_oceanai_dir()
    
    out_dir = Path(args.output_dir) if args.output_dir else oceanai_dir / "data" / "processed" / "integrated"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_dir = oceanai_dir / "reports" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    coarsen_factor = config["grid"]["coarsen_factor"]
    depth_idx = config["grid"]["depth_level"]
    
    region_dfs = []
    summary_rows = []
    
    for reg_key, reg_cfg in config["regions"].items():
        df_reg = integrate_region(reg_key, reg_cfg, repo_root, coarsen_factor, depth_idx)
        
        # Save regional parquet
        reg_parquet = out_dir / f"{reg_key}.parquet"
        df_reg.to_parquet(reg_parquet, index=False, compression="snappy")
        logger.info(f"Saved {reg_key} parquet to: {reg_parquet}")
        
        region_dfs.append(df_reg)
        summary_rows.append({
            "region": reg_key,
            "name": reg_cfg["name"],
            "total_cell_months": len(df_reg),
            "fished_cells": int((df_reg["hours"] > 0).sum()),
            "zero_cells": int((df_reg["hours"] == 0).sum()),
            "fished_pct": float((df_reg["hours"] > 0).mean() * 100),
            "zero_pct": float((df_reg["hours"] == 0).mean() * 100),
            "total_fishing_hours": float(df_reg["hours"].sum()),
            "mean_fishing_hours": float(df_reg["hours"].mean()),
            "max_fishing_hours": float(df_reg["hours"].max())
        })
        
    # Combine into master integrated panel
    unified_df = pd.concat(region_dfs, ignore_index=True)
    unified_df = unified_df.sort_values(["region", "lat_bin", "lon_bin", "month"]).reset_index(drop=True)
    
    master_parquet = out_dir / "integrated.parquet"
    unified_df.to_parquet(master_parquet, index=False, compression="snappy")
    logger.info(f"Successfully wrote unified master panel to: {master_parquet}")
    
    # Save integration summary CSV
    summary_rows.append({
        "region": "all_combined",
        "name": "Combined Indian Ocean",
        "total_cell_months": len(unified_df),
        "fished_cells": int((unified_df["hours"] > 0).sum()),
        "zero_cells": int((unified_df["hours"] == 0).sum()),
        "fished_pct": float((unified_df["hours"] > 0).mean() * 100),
        "zero_pct": float((unified_df["hours"] == 0).mean() * 100),
        "total_fishing_hours": float(unified_df["hours"].sum()),
        "mean_fishing_hours": float(unified_df["hours"].mean()),
        "max_fishing_hours": float(unified_df["hours"].max())
    })
    
    df_summary = pd.DataFrame(summary_rows)
    summary_csv = metrics_dir / "integration_summary.csv"
    df_summary.to_csv(summary_csv, index=False)
    logger.info(f"Saved integration metrics summary to: {summary_csv}")
    
    logger.info(f"Integration complete. Total Records: {len(unified_df):,}. Zero-inflation: {unified_df['hours'].eq(0).mean()*100:.2f}%.")


if __name__ == "__main__":
    main()
