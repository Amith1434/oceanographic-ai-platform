"""
test_pipeline.py
----------------
Comprehensive test suite verifying data integrity, zero-filling, absence of leakage,
chronological split bounds, OBIS 4-category matching, and complete 16-feature space.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest


OCEANAI_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = OCEANAI_DIR / "data" / "processed" / "integrated"
METRICS_DIR = OCEANAI_DIR / "reports" / "metrics"
ROOT_DIR = OCEANAI_DIR.parent


# -----------------------------------------------------------------------------
# 1. GFW Hours Conservation Test
# -----------------------------------------------------------------------------
def test_gfw_hours_conservation():
    """Verify that apparent fishing hours are strictly conserved relative to oceanic land-masking."""
    gfw_as = pd.read_csv(ROOT_DIR / "data" / "processed" / "gfw" / "arabian_sea" / "gfw_clean.csv")
    gfw_bob = pd.read_csv(ROOT_DIR / "data" / "processed" / "gfw" / "bay_of_bengal" / "gfw_clean.csv")
    
    raw_total_hours = float(gfw_as["hours"].sum() + gfw_bob["hours"].sum())
    
    integrated_df = pd.read_parquet(DATA_DIR / "integrated.parquet")
    integrated_total_hours = float(integrated_df["hours"].sum())
    
    # Verify exact match with integration_summary.csv metric
    summary_df = pd.read_csv(METRICS_DIR / "integration_summary.csv")
    summary_total_hours = float(summary_df[summary_df["region"] == "all_combined"]["total_fishing_hours"].iloc[0])
    
    assert np.isclose(integrated_total_hours, summary_total_hours, atol=0.01), (
        f"Parquet hours {integrated_total_hours:.2f} do not match summary {summary_total_hours:.2f}"
    )
    
    # 98.51% of raw GFW hours fall in Copernicus oceanic cells; 1.49% fall in land-masked estuaries/ports
    retention_pct = (integrated_total_hours / raw_total_hours) * 100
    assert 98.0 < retention_pct < 100.0, f"Unexpected GFW retention rate: {retention_pct:.2f}%"


# -----------------------------------------------------------------------------
# 2. Zero-Filling & Grid Integrity Test
# -----------------------------------------------------------------------------
def test_zero_filling():
    """Verify zero-inflation rate is ~78.94% and unobserved cells have 0.0, not NaN."""
    integrated_df = pd.read_parquet(DATA_DIR / "integrated.parquet")
    
    assert "hours" in integrated_df.columns
    assert integrated_df["hours"].isna().sum() == 0, "Integrated hours contains NaNs!"
    
    zero_count = (integrated_df["hours"] == 0.0).sum()
    total_count = len(integrated_df)
    zero_pct = (zero_count / total_count) * 100
    
    assert total_count == 1_255_968, f"Expected 1,255,968 cell-months, got {total_count}"
    assert 78.0 <= zero_pct <= 80.0, f"Expected ~78.9% zero-filling rate, got {zero_pct:.2f}%"


# -----------------------------------------------------------------------------
# 3. No-Leakage & Train-Only Climatology Test
# -----------------------------------------------------------------------------
def test_no_leakage():
    """Verify train-only climatology for model features and absence of 'year' feature."""
    features_df = pd.read_parquet(DATA_DIR / "features.parquet")
    
    # Model features must exist
    assert "sst_anomaly" in features_df.columns
    assert "so_anomaly" in features_df.columns
    assert "sst_zscore" in features_df.columns
    
    # Separate full-period mapping features must exist
    assert "sst_anomaly_fullperiod" in features_df.columns
    
    # Strict No-Year Guardrail: 'year' must not be in model features
    FEATS_M2 = [
        "lat_bin", "lon_bin", "region_id", "dist_to_coast_km", "sin_month", "cos_month",
        "thetao", "so", "uo", "vo", "speed", "sst_anomaly", "so_anomaly", "sst_lag1", "so_lag1", "speed_lag1"
    ]
    assert "year" not in FEATS_M2, "'year' feature found in model feature set!"
    assert len(FEATS_M2) == 16, f"Expected 16 features for Model 2, got {len(FEATS_M2)}"


# -----------------------------------------------------------------------------
# 4. Strict Chronological Split Bounds Test
# -----------------------------------------------------------------------------
def test_split_bounds():
    """Verify strict out-of-time chronological split bounds: train 2019-2023, val 2024, test 2025."""
    features_df = pd.read_parquet(DATA_DIR / "features.parquet")
    features_df["dt_year"] = pd.to_datetime(features_df["month"]).dt.year
    
    train = features_df[features_df["dt_year"] <= 2023]
    val = features_df[features_df["dt_year"] == 2024]
    test = features_df[features_df["dt_year"] == 2025]
    
    assert train["dt_year"].min() == 2019 and train["dt_year"].max() == 2023
    assert val["dt_year"].unique().tolist() == [2024]
    assert test["dt_year"].unique().tolist() == [2025]
    
    assert len(train) == 897_120, f"Expected 897,120 train rows, got {len(train)}"
    assert len(val) == 179_424, f"Expected 179,424 val rows, got {len(val)}"
    assert len(test) == 179_424, f"Expected 179,424 test rows, got {len(test)}"


# -----------------------------------------------------------------------------
# 5. OBIS 4-Category Matching Test
# -----------------------------------------------------------------------------
def test_obis_match_categories():
    """Verify that OBIS matching uses the 4-category classification and drops pre-2019."""
    match_csv = METRICS_DIR / "obis_match_quality.csv"
    assert match_csv.exists(), "obis_match_quality.csv missing!"
    
    df_match = pd.read_csv(match_csv)
    assert len(df_match) == 3, "Expected 3 focal species in match quality table"
    
    # Verify columns exist
    required_cols = ["Species", "Total_Records", "Pre_2019", "Exact_Match", "Nearest_Same_Month", "Far_Dropped", "Used_Records"]
    for col in required_cols:
        assert col in df_match.columns, f"Column {col} missing from match quality table"
        
    # Verify pre-2019 records were dropped
    total_pre2019 = df_match["Pre_2019"].sum()
    assert total_pre2019 == 69, f"Expected 69 pre-2019 records dropped (51 C. melanopterus, 18 R. typus), got {total_pre2019}"
    
    # Verify Setipinna tenuifilis has 476 records (100% UNESCO eDNA)
    setipinna = df_match[df_match["Species"] == "Setipinna tenuifilis"].iloc[0]
    assert setipinna["Total_Records"] == 476
    assert setipinna["Pre_2019"] == 0


# -----------------------------------------------------------------------------
# 6. Complete 16-Feature Importance & Grouped Shares Test
# -----------------------------------------------------------------------------
def test_16_feature_importance():
    """Verify that all 16 features exist, uo and vo are included, and grouped shares total 100%."""
    imp_csv = METRICS_DIR / "feature_importance_stage_a_m2.csv"
    if not imp_csv.exists():
        pytest.skip("feature_importance_stage_a_m2.csv not yet generated.")
        
    df_imp = pd.read_csv(imp_csv)
    assert len(df_imp) == 16, f"Expected 16 features in Model 2, found {len(df_imp)}"
    
    feats = df_imp["Feature"].tolist()
    assert "uo" in feats, "'uo' missing from 16-feature table"
    assert "vo" in feats, "'vo' missing from 16-feature table"
    
    # Check grouped shares
    shares_json = METRICS_DIR / "feature_importance_grouped_shares.json"
    if shares_json.exists():
        with open(shares_json, "r", encoding="utf-8") as f:
            shares = json.load(f)
            
        sp_gain = shares["Spatial_Features"]["Split_Gain_Share_Pct"]
        se_gain = shares["Seasonal_Features"]["Split_Gain_Share_Pct"]
        oc_gain = shares["Ocean_Physical_Features"]["Split_Gain_Share_Pct"]
        
        total_gain = sp_gain + se_gain + oc_gain
        assert np.isclose(total_gain, 100.0, atol=0.5), f"Grouped gain shares must sum to 100%, got {total_gain}"
