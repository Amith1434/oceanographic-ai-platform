
"""
Build monthly Copernicus + GFW modelling datasets.

Spatial resolution: 0.25 degrees
Temporal resolution: Monthly
Period: January 2019 - December 2025

OBIS remains separate for descriptive biodiversity analysis.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import xarray as xr


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = ROOT / "config" / "regions.json"

OUTPUT_DIR = ROOT / "data" / "processed" / "integrated"

GRID_DEGREES = 0.25

START_DATE = "2019-01-01"
END_DATE = "2025-12-31"

EXPECTED_MONTHS = 84

OCEAN_VARIABLES = [
    "thetao",
    "so",
    "uo",
    "vo",
    "current_speed"
]


# ============================================================
# 2. LOAD CONFIGURATION
# ============================================================

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

REGIONS = CONFIG["regions"]


# ============================================================
# 3. COMMON GRID FUNCTION
# ============================================================

def add_grid_indices(
    frame,
    lat_col,
    lon_col
):
    """
    Assign lower-left coordinates to a common
    0.25-degree grid.
    """

    out = frame.copy()

    out["lat_bin_id"] = np.floor(
        out[lat_col].astype(float) / GRID_DEGREES + 1e-9
    ).astype("int32")

    out["lon_bin_id"] = np.floor(
        out[lon_col].astype(float) / GRID_DEGREES + 1e-9
    ).astype("int32")

    out["lat_bin"] = (
        out["lat_bin_id"] * GRID_DEGREES
    ).astype("float32")

    out["lon_bin"] = (
        out["lon_bin_id"] * GRID_DEGREES
    ).astype("float32")

    return out


# ============================================================
# 4. LOAD AND AGGREGATE COPERNICUS
# ============================================================

def load_copernicus_monthly(
    nc_path,
    bounds
):
    """
    Load surface-level ocean variables.

    Aggregate daily/submonthly observations into
    monthly 0.25-degree ocean grid cells.
    """

    print("\nLoading Copernicus:", nc_path)

    if not nc_path.exists():
        raise FileNotFoundError(nc_path)

    monthly_parts = []

    with xr.open_dataset(nc_path) as source:

        required_coords = [
            "time",
            "depth",
            "latitude",
            "longitude"
        ]

        for coord in required_coords:
            if coord not in source.coords:
                raise ValueError(
                    f"Missing Copernicus coordinate: {coord}"
                )

        required_vars = [
            "thetao",
            "so",
            "uo",
            "vo"
        ]

        for var in required_vars:
            if var not in source.data_vars:
                raise ValueError(
                    f"Missing Copernicus variable: {var}"
                )

        # Select shallowest available depth.
        surface = (
            source.sortby("depth")
            .isel(depth=0, drop=True)
        )

        # Calculate current speed if not already present.
        if "current_speed" not in surface.data_vars:

            surface = surface.assign(
                current_speed=np.hypot(
                    surface["uo"],
                    surface["vo"]
                )
            )

        # Keep the requested period.
        surface = surface.sel(
            time=slice(START_DATE, END_DATE)
        )

        # Keep only the required variables.
        surface = surface[OCEAN_VARIABLES]

        # Monthly mean.
        monthly_ds = surface.resample(
            time="1MS"
        ).mean()

        print("Processing monthly ocean observations...")

        for timestamp in monthly_ds.time.values:

            month_ds = monthly_ds.sel(
                time=timestamp
            )

            frame = month_ds.to_dataframe().reset_index()

            # Restrict to region boundaries.
            frame = frame[
                (frame["latitude"] >= bounds["min_lat"]) &
                (frame["latitude"] <= bounds["max_lat"]) &
                (frame["longitude"] >= bounds["min_lon"]) &
                (frame["longitude"] <= bounds["max_lon"])
            ].copy()

            # Retain valid ocean cells.
            # thetao NaN represents unavailable water data,
            # including land-mask cells.
            frame = frame.dropna(
                subset=["thetao"]
            )

            if frame.empty:
                continue

            frame["month"] = pd.Timestamp(
                timestamp
            ).to_period("M").to_timestamp()

            frame = add_grid_indices(
                frame,
                "latitude",
                "longitude"
            )

            # Aggregate Copernicus observations
            # into 0.25-degree grid cells.
            grouped = frame.groupby(
                [
                    "month",
                    "lat_bin_id",
                    "lon_bin_id",
                    "lat_bin",
                    "lon_bin"
                ],
                as_index=False
            )[OCEAN_VARIABLES].mean()

            monthly_parts.append(grouped)

    if not monthly_parts:
        raise ValueError(
            f"No valid Copernicus ocean data found: {nc_path}"
        )

    ocean = pd.concat(
        monthly_parts,
        ignore_index=True
    )

    ocean = ocean.groupby(
        [
            "month",
            "lat_bin_id",
            "lon_bin_id",
            "lat_bin",
            "lon_bin"
        ],
        as_index=False
    )[OCEAN_VARIABLES].mean()

    ocean = ocean.sort_values(
        ["month", "lat_bin_id", "lon_bin_id"]
    ).reset_index(drop=True)

    print("Copernicus monthly rows:", len(ocean))
    print("Unique ocean cells:", ocean[
        ["lat_bin_id", "lon_bin_id"]
    ].drop_duplicates().shape[0])

    return ocean


# ============================================================
# 5. LOAD AND AGGREGATE GFW
# ============================================================

def load_gfw_monthly(
    csv_path,
    bounds
):
    """
    Aggregate fishing hours by month and 0.25-degree cell.

    Hours are summed across flags.
    Vessel IDs are NOT summed.
    """

    print("\nLoading GFW:", csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    gfw = pd.read_csv(csv_path)

    required = [
        "date",
        "lat",
        "lon",
        "hours"
    ]

    missing = [
        col for col in required
        if col not in gfw.columns
    ]

    if missing:
        raise ValueError(
            f"GFW file missing columns: {missing}"
        )

    # Convert columns.
    gfw["date"] = pd.to_datetime(
        gfw["date"],
        errors="coerce",
        utc=True
    )

    for col in ["lat", "lon", "hours"]:
        gfw[col] = pd.to_numeric(
            gfw[col],
            errors="coerce"
        )

    # Remove invalid records.
    gfw = gfw.dropna(
        subset=["date", "lat", "lon", "hours"]
    ).copy()

    # Remove negative fishing hours.
    gfw = gfw[gfw["hours"] >= 0].copy()

    # Restrict date range.
    gfw = gfw[
        (gfw["date"] >= pd.Timestamp(START_DATE, tz="UTC")) &
        (gfw["date"] < pd.Timestamp("2026-01-01", tz="UTC"))
    ].copy()

    # Restrict geographic boundaries.
    gfw = gfw[
        (gfw["lat"] >= bounds["min_lat"]) &
        (gfw["lat"] <= bounds["max_lat"]) &
        (gfw["lon"] >= bounds["min_lon"]) &
        (gfw["lon"] <= bounds["max_lon"])
    ].copy()

    if gfw.empty:
        raise ValueError(
            f"No valid GFW records found: {csv_path}"
        )

    gfw["month"] = (
        gfw["date"]
        .dt.tz_convert(None)
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    gfw = add_grid_indices(
        gfw,
        "lat",
        "lon"
    )

    # Sum hours across flags.
    gfw_monthly = gfw.groupby(
        [
            "month",
            "lat_bin_id",
            "lon_bin_id",
            "lat_bin",
            "lon_bin"
        ],
        as_index=False
    ).agg(
        fishing_hours=("hours", "sum"),
        gfw_records=("hours", "size")
    )

    print("GFW monthly rows:", len(gfw_monthly))
    print(
        "Total fishing hours:",
        round(gfw_monthly["fishing_hours"].sum(), 2)
    )

    return gfw_monthly


# ============================================================
# 6. BUILD COMPLETE MONTHLY OCEAN PANEL
# ============================================================

def build_integrated_region(region_key, region_info):

    print("\n" + "=" * 65)
    print("BUILDING:", region_info["name"].upper())
    print("=" * 65)

    bounds = region_info

    copernicus_path = (
        ROOT
        / "data"
        / "processed"
        / "copernicus"
        / region_key
        / "copernicus_features_2019_2025.nc"
    )

    gfw_path = (
        ROOT
        / "data"
        / "processed"
        / "gfw"
        / region_key
        / "gfw_clean.csv"
    )

    # Load both domains independently.
    ocean = load_copernicus_monthly(
        copernicus_path,
        bounds
    )

    fishing = load_gfw_monthly(
        gfw_path,
        bounds
    )

    # --------------------------------------------------------
    # Create the full ocean-cell x month panel.
    # --------------------------------------------------------

    cells = ocean[
        [
            "lat_bin_id",
            "lon_bin_id",
            "lat_bin",
            "lon_bin"
        ]
    ].drop_duplicates()

    months = pd.DataFrame({
        "month": pd.date_range(
            START_DATE,
            END_DATE,
            freq="MS"
        )
    })

    # Every valid ocean cell gets 84 monthly rows.
    panel = cells.merge(
        months,
        how="cross"
    )

    # Add monthly Copernicus variables.
    panel = panel.merge(
        ocean,
        on=[
            "month",
            "lat_bin_id",
            "lon_bin_id",
            "lat_bin",
            "lon_bin"
        ],
        how="left",
        validate="one_to_one"
    )

    # Add monthly GFW fishing activity.
    panel = panel.merge(
        fishing,
        on=[
            "month",
            "lat_bin_id",
            "lon_bin_id",
            "lat_bin",
            "lon_bin"
        ],
        how="left",
        validate="one_to_one"
    )

    # No GFW row in a valid ocean cell means
    # no detected fishing activity in that cell-month.
    panel["fishing_hours"] = (
        panel["fishing_hours"].fillna(0)
    )

    panel["gfw_records"] = (
        panel["gfw_records"].fillna(0).astype("int32")
    )

    # Region identifier.
    panel["region"] = region_key

    # Fishing occurrence target.
    panel["fishing_present"] = (
        panel["fishing_hours"] > 0
    ).astype("int8")

    # Stable time features.
    panel["year"] = panel["month"].dt.year
    panel["month_number"] = panel["month"].dt.month

    # Sort.
    panel = panel.sort_values(
        ["month", "lat_bin_id", "lon_bin_id"]
    ).reset_index(drop=True)

    return panel, ocean, fishing


# ============================================================
# 7. VALIDATION
# ============================================================

def validate_integrated(
    panel,
    ocean,
    fishing,
    region_key
):

    print("\nVALIDATION:", region_key.upper())

    expected_cells = ocean[
        ["lat_bin_id", "lon_bin_id"]
    ].drop_duplicates().shape[0]

    expected_rows = expected_cells * EXPECTED_MONTHS

    actual_months = panel["month"].nunique()

    # GFW records that did not match a Copernicus ocean cell.
    ocean_keys = ocean[
        ["month", "lat_bin_id", "lon_bin_id"]
    ].drop_duplicates()

    fishing_keys = fishing[
        ["month", "lat_bin_id", "lon_bin_id"]
    ].drop_duplicates()

    matched = fishing_keys.merge(
        ocean_keys,
        on=["month", "lat_bin_id", "lon_bin_id"],
        how="inner"
    )

    unmatched_gfw = len(fishing_keys) - len(matched)

    summary = {
        "region": region_key,
        "copernicus_rows": len(ocean),
        "gfw_rows": len(fishing),
        "unique_ocean_cells": expected_cells,
        "expected_months": EXPECTED_MONTHS,
        "actual_months": actual_months,
        "expected_panel_rows": expected_rows,
        "actual_panel_rows": len(panel),
        "zero_fishing_rows": int(
            (panel["fishing_hours"] == 0).sum()
        ),
        "positive_fishing_rows": int(
            (panel["fishing_hours"] > 0).sum()
        ),
        "total_fishing_hours": float(
            panel["fishing_hours"].sum()
        ),
        "unmatched_gfw_cell_months": unmatched_gfw,
        "missing_thetao": int(
            panel["thetao"].isna().sum()
        ),
        "missing_so": int(
            panel["so"].isna().sum()
        ),
        "missing_current_speed": int(
            panel["current_speed"].isna().sum()
        ),
        "duplicate_cell_months": int(
            panel.duplicated(
                ["month", "lat_bin_id", "lon_bin_id"]
            ).sum()
        )
    }

    print("\nPanel shape:", panel.shape)
    print("Expected rows:", expected_rows)
    print("Actual rows:", len(panel))
    print("Months:", actual_months)
    print("Unique ocean cells:", expected_cells)
    print("Zero fishing rows:", summary["zero_fishing_rows"])
    print("Positive fishing rows:", summary["positive_fishing_rows"])
    print("Unmatched GFW cell-months:", unmatched_gfw)
    print("Duplicate cell-months:", summary["duplicate_cell_months"])

    if actual_months != EXPECTED_MONTHS:
        raise ValueError(
            f"Expected 84 months, found {actual_months}"
        )

    if len(panel) != expected_rows:
        raise ValueError(
            "Integrated panel row count does not match "
            "ocean cells x 84 months."
        )

    if summary["duplicate_cell_months"] != 0:
        raise ValueError("Duplicate cell-month records found.")

    return summary


# ============================================================
# 8. MAIN EXECUTION
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    all_regions = []
    validation_results = []

    for region_key, region_info in REGIONS.items():

        panel, ocean, fishing = build_integrated_region(
            region_key,
            region_info
        )

        summary = validate_integrated(
            panel,
            ocean,
            fishing,
            region_key
        )

        validation_results.append(summary)

        # Save individual regional dataset.
        output_path = (
            OUTPUT_DIR / f"{region_key}.parquet"
        )

        panel.to_parquet(
            output_path,
            index=False
        )

        print("\nSaved:", output_path)

        all_regions.append(panel)

    # --------------------------------------------------------
    # Combined dataset.
    # --------------------------------------------------------

    combined = pd.concat(
        all_regions,
        ignore_index=True
    )

    combined_path = (
        OUTPUT_DIR / "integrated.parquet"
    )

    combined.to_parquet(
        combined_path,
        index=False
    )

    print("\nSaved combined dataset:", combined_path)

    # --------------------------------------------------------
    # Validation summary.
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        validation_results
    )

    summary_path = (
        OUTPUT_DIR / "validation_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print("Saved validation summary:", summary_path)

    # --------------------------------------------------------
    # Final report.
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("INTEGRATION COMPLETED")
    print("=" * 65)

    print("Combined dataset shape:", combined.shape)

    print("\nRows by region:")
    print(combined.groupby("region").size())

    print("\nMonth range:")
    print(combined["month"].min(), "to", combined["month"].max())

    print("\nFinal output directory:")
    print(OUTPUT_DIR)

    print("\nGenerated files:")
    print("1. arabian_sea.parquet")
    print("2. bay_of_bengal.parquet")
    print("3. integrated.parquet")
    print("4. validation_summary.csv")


if __name__ == "__main__":
    main()
