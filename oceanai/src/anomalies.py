"""
Anomaly Detection & Cross-Domain Fisheries Impact
=================================================
Identifies univariate marine warm anomalies (marine heatwave proxy) and multivariate
ocean state disruptions (Isolation Forest), then rigorously quantifies apparent fishing effort shifts.

Addresses Confounding and Autocorrelation (Review Updates):
1. Analyzes anomaly frequency across calendar years and regions to document secular warming trends
   and coverage coincidence.
2. Performs WITHIN-YEAR and WITHIN-REGION comparisons to remove secular coverage jump confounding.
3. Implements Two-Way Fixed Effects (TWFE) regressions (Cell FE + Year-Month FE) with clustered standard
   errors by cell on demeaned log1p(hours) and is_fishing.
4. Implements per-cell linear trend removal (detrended SST anomalies).
5. Computes block-bootstrap confidence intervals (1,000 iterations resampled by monthly time-blocks)
   to account for spatial autocorrelation.
6. Replaces causal assertions with "consistent with" hypotheses, removes unsupported species names,
   and caveats that monthly reanalysis cannot resolve episodic storms.
"""

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.ensemble import IsolationForest
import statsmodels.api as sm

# Local module import
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_oceanai_dir, load_config, set_seed, setup_logger

logger = setup_logger("anomalies")

# Headless plotting setup
LOCAL_CACHE_DIR = Path(__file__).resolve().parent.parent / ".matplotlib_cache"
LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(LOCAL_CACHE_DIR)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def detect_anomalies(df: pd.DataFrame, seed: int):
    """Detect univariate warm anomalies, detrended anomalies, and multivariate dynamic disruptions."""
    logger.info("Detecting ocean physical anomalies...")
    df = df.copy()
    
    # 1. Univariate Warm Anomalies (from leak-free train climatology)
    p90_sst = float(df["sst_anomaly"].quantile(0.90))
    logger.info(f"Basin 90th percentile SST anomaly threshold: +{p90_sst:.3f}°C")
    
    df["warm_anomaly_p90"] = (df["sst_anomaly"] > p90_sst).astype(int)
    df["warm_anomaly_z2"] = (df["sst_zscore"] > 2.0).astype(int)
    df["warm_anomaly"] = ((df["sst_anomaly"] > p90_sst) | (df["sst_zscore"] > 2.0)).astype(int)
    
    warm_count = df["warm_anomaly"].sum()
    warm_pct = df["warm_anomaly"].mean() * 100
    logger.info(f"Warm anomalous cell-months (Train-clim baseline): {warm_count:,} ({warm_pct:.2f}%)")
    
    # 2. Per-Cell Detrended SST Anomalies
    logger.info("Computing per-cell detrended SST anomalies (removing 2019-2025 secular warming trend)...")
    df = df.sort_values(["region", "lat_bin", "lon_bin", "month"]).reset_index(drop=True)
    
    # Add integer time step t = 0..83
    months_seq = {m: i for i, m in enumerate(sorted(df["month"].unique()))}
    df["time_idx"] = df["month"].map(months_seq).astype(float)
    
    # Fast cell-level linear detrending: fit slope b = cov(t, sst) / var(t)
    t_vals = df["time_idx"].values
    t_mean = 41.5 # (84 - 1) / 2
    t_var = np.var(np.arange(84))
    
    cell_groups = df.groupby(["region", "lat_bin", "lon_bin"])
    # b_cell = sum((t - t_mean) * (sst - sst_mean)) / (84 * t_var)
    df["t_dev"] = df["time_idx"] - t_mean
    df["sst_dev"] = cell_groups["thetao"].transform(lambda x: x - x.mean())
    df["cov_t_sst"] = df["t_dev"] * df["sst_dev"]
    cell_slopes = df.groupby(["region", "lat_bin", "lon_bin"])["cov_t_sst"].transform("sum") / (84.0 * t_var)
    df["sst_linear_trend"] = cell_slopes * df["t_dev"]
    df["sst_detrended_anomaly"] = df["sst_anomaly"] - df["sst_linear_trend"]
    
    p90_detrended = float(df["sst_detrended_anomaly"].quantile(0.90))
    df["warm_anomaly_detrended"] = (df["sst_detrended_anomaly"] > p90_detrended).astype(int)
    
    # Drop temporary detrending calculation columns
    df = df.drop(columns=["t_dev", "sst_dev", "cov_t_sst", "sst_linear_trend"])
    
    # 3. Multivariate Dynamic Disruption via Isolation Forest
    logger.info("Fitting unsupervised Isolation Forest on [sst_anomaly, so_anomaly, speed]...")
    iso = IsolationForest(
        n_estimators=100,
        max_samples=10000,
        contamination=0.05,
        random_state=seed,
        n_jobs=2
    )
    sample_train = df[df["year"] <= 2023][["sst_anomaly", "so_anomaly", "speed"]].sample(n=min(50000, len(df)), random_state=seed)
    iso.fit(sample_train)
    iso_preds = iso.predict(df[["sst_anomaly", "so_anomaly", "speed"]].values)
    df["iso_anomaly"] = (iso_preds == -1).astype(int)
    
    iso_count = df["iso_anomaly"].sum()
    logger.info(f"Isolation Forest flagged disruptions: {iso_count:,} ({df['iso_anomaly'].mean()*100:.2f}%)")
    
    return df, p90_sst


def compute_anomaly_shares(df: pd.DataFrame, out_dir: Path):
    """Compute and save share of warm-anomaly and IsolationForest cell-months by year and region."""
    logger.info("Computing anomaly shares by year and region...")
    records = []
    
    for (reg, yr), sub in df.groupby(["region", "year"]):
        records.append({
            "Region": reg,
            "Year": int(yr),
            "Total_Cell_Months": len(sub),
            "Warm_Anomaly_Share_Pct": round(float(sub["warm_anomaly"].mean() * 100), 2),
            "Warm_Detrended_Share_Pct": round(float(sub["warm_anomaly_detrended"].mean() * 100), 2),
            "Iso_Disruption_Share_Pct": round(float(sub["iso_anomaly"].mean() * 100), 2),
            "Mean_Fishing_Hours": round(float(sub["hours"].mean()), 2),
            "Fished_Cells_Pct": round(float(sub["is_fishing"].mean() * 100), 2)
        })
        
    df_shares = pd.DataFrame(records)
    csv_path = out_dir / "anomaly_shares_by_year_region.csv"
    df_shares.to_csv(csv_path, index=False)
    logger.info(f"Saved anomaly shares to: {csv_path}")
    return df_shares


def compute_within_year_comparisons(df: pd.DataFrame, out_dir: Path):
    """Compute within-year and within-region comparisons for warm vs normal and IsoForest disruptions."""
    logger.info("Performing within-year and within-region anomaly comparisons...")
    records = []
    
    for (reg, yr), sub in df.groupby(["region", "year"]):
        # Warm vs Normal
        norm_warm = sub[sub["warm_anomaly"] == 0]
        anom_warm = sub[sub["warm_anomaly"] == 1]
        
        rate_norm_w = float(norm_warm["is_fishing"].mean() * 100) if len(norm_warm) > 0 else 0.0
        rate_anom_w = float(anom_warm["is_fishing"].mean() * 100) if len(anom_warm) > 0 else 0.0
        mean_norm_w = float(norm_warm["hours"].mean()) if len(norm_warm) > 0 else 0.0
        mean_anom_w = float(anom_warm["hours"].mean()) if len(anom_warm) > 0 else 0.0
        
        # IsoForest Disruption vs Normal
        norm_iso = sub[sub["iso_anomaly"] == 0]
        anom_iso = sub[sub["iso_anomaly"] == 1]
        
        rate_norm_i = float(norm_iso["is_fishing"].mean() * 100) if len(norm_iso) > 0 else 0.0
        rate_anom_i = float(anom_iso["is_fishing"].mean() * 100) if len(anom_iso) > 0 else 0.0
        mean_norm_i = float(norm_iso["hours"].mean()) if len(norm_iso) > 0 else 0.0
        mean_anom_i = float(anom_iso["hours"].mean()) if len(anom_iso) > 0 else 0.0
        
        records.append({
            "Region": reg,
            "Year": int(yr),
            "Warm_Norm_Fished_Pct": round(rate_norm_w, 2),
            "Warm_Anom_Fished_Pct": round(rate_anom_w, 2),
            "Warm_Fished_Diff": round(rate_anom_w - rate_norm_w, 2),
            "Warm_Fished_Ratio": round(rate_anom_w / rate_norm_w, 3) if rate_norm_w > 0 else np.nan,
            "Warm_Norm_Mean_Hours": round(mean_norm_w, 2),
            "Warm_Anom_Mean_Hours": round(mean_anom_w, 2),
            "Warm_Hours_Diff": round(mean_anom_w - mean_norm_w, 2),
            "Warm_Hours_Ratio": round(mean_anom_w / mean_norm_w, 3) if mean_norm_w > 0 else np.nan,
            "Iso_Norm_Fished_Pct": round(rate_norm_i, 2),
            "Iso_Anom_Fished_Pct": round(rate_anom_i, 2),
            "Iso_Fished_Diff": round(rate_anom_i - rate_norm_i, 2),
            "Iso_Fished_Ratio": round(rate_anom_i / rate_norm_i, 3) if rate_norm_i > 0 else np.nan,
            "Iso_Norm_Mean_Hours": round(mean_norm_i, 2),
            "Iso_Anom_Mean_Hours": round(mean_anom_i, 2),
            "Iso_Hours_Diff": round(mean_anom_i - mean_norm_i, 2),
            "Iso_Hours_Ratio": round(mean_anom_i / mean_norm_i, 3) if mean_norm_i > 0 else np.nan,
        })
        
    df_within = pd.DataFrame(records)
    csv_path = out_dir / "anomaly_within_year_region.csv"
    df_within.to_csv(csv_path, index=False)
    logger.info(f"Saved within-year anomaly comparisons to: {csv_path}")
    return df_within


def run_fixed_effects_regressions(df: pd.DataFrame, out_dir: Path):
    """
    Run Two-Way Fixed Effects (TWFE) regressions on demeaned data:
    Y_tilde = Y - Y_cell_mean - Y_month_mean + Y_grand_mean
    Clusters standard errors by cell_id to account for spatial autocorrelation.
    """
    logger.info("Estimating Two-Way Fixed Effects (TWFE) panel regressions with clustered standard errors...")
    df = df.copy()
    df["cell_id"] = df["region"] + "_" + df["lat_bin"].astype(str) + "_" + df["lon_bin"].astype(str)
    df["month_id"] = df["month"].astype(str)
    cluster_groups = df["cell_id"].astype("category").cat.codes.values
    
    results_dict = {}
    
    # Models to estimate:
    # 1. log_hours on warm_anomaly
    # 2. is_fishing on warm_anomaly
    # 3. log_hours on warm_anomaly_detrended
    # 4. log_hours on iso_anomaly
    # 5. is_fishing on iso_anomaly
    
    models = [
        ("warm_anomaly_on_log_hours", "log_hours", "warm_anomaly"),
        ("warm_anomaly_on_is_fishing", "is_fishing", "warm_anomaly"),
        ("warm_detrended_on_log_hours", "log_hours", "warm_anomaly_detrended"),
        ("iso_anomaly_on_log_hours", "log_hours", "iso_anomaly"),
        ("iso_anomaly_on_is_fishing", "is_fishing", "iso_anomaly"),
    ]
    
    cell_grouped = df.groupby("cell_id")
    month_grouped = df.groupby("month_id")
    
    for name, y_var, x_var in models:
        y = df[y_var].values
        x = df[x_var].values
        
        y_cell_m = cell_grouped[y_var].transform("mean").values
        x_cell_m = cell_grouped[x_var].transform("mean").values
        
        y_month_m = month_grouped[y_var].transform("mean").values
        x_month_m = month_grouped[x_var].transform("mean").values
        
        y_tilde = y - y_cell_m - y_month_m + y.mean()
        x_tilde = x - x_cell_m - x_month_m + x.mean()
        
        ols_mod = sm.OLS(y_tilde, x_tilde)
        res = ols_mod.fit(cov_type="cluster", cov_kwds={"groups": cluster_groups})
        
        ci = res.conf_int()[0]
        results_dict[name] = {
            "Outcome": y_var,
            "Regressor": x_var,
            "Coefficient": round(float(res.params[0]), 5),
            "Std_Error": round(float(res.bse[0]), 5),
            "t_statistic": round(float(res.tvalues[0]), 3),
            "p_value": float(res.pvalues[0]),
            "CI_95_Lower": round(float(ci[0]), 5),
            "CI_95_Upper": round(float(ci[1]), 5),
            "N_Obs": int(res.nobs),
            "N_Clusters": len(np.unique(cluster_groups))
        }
        logger.info(f"TWFE {name}: Coef={res.params[0]:.4f} (SE={res.bse[0]:.4f}, p={res.pvalues[0]:.2e}, 95% CI=[{ci[0]:.4f}, {ci[1]:.4f}])")
        
    json_path = out_dir / "anomaly_fixed_effects_regression.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)
    logger.info(f"Saved TWFE regression results to: {json_path}")
    return results_dict


def compute_block_bootstrap_cis(df: pd.DataFrame, out_dir: Path, n_reps=1000, seed=42):
    """
    Compute 95% bootstrap confidence intervals for anomaly effect sizes
    using month-block resampling to preserve spatial correlation within each month.
    Pre-aggregates by month for instant, vectorized computation across 1,000 replications.
    """
    logger.info(f"Computing month-block bootstrap confidence intervals ({n_reps:,} replications)...")
    np.random.seed(seed)
    
    unique_months = df["month"].unique()
    n_months = len(unique_months)
    
    month_stats = []
    for m in unique_months:
        sub = df[df["month"] == m]
        w0 = sub[sub["warm_anomaly"] == 0]
        w1 = sub[sub["warm_anomaly"] == 1]
        i0 = sub[sub["iso_anomaly"] == 0]
        i1 = sub[sub["iso_anomaly"] == 1]
        
        month_stats.append({
            "month": m,
            "w0_hours_sum": float(w0["hours"].sum()),
            "w0_count": len(w0),
            "w0_fish_sum": float(w0["is_fishing"].sum()),
            "w1_hours_sum": float(w1["hours"].sum()),
            "w1_count": len(w1),
            "w1_fish_sum": float(w1["is_fishing"].sum()),
            "i0_hours_sum": float(i0["hours"].sum()),
            "i0_count": len(i0),
            "i0_fish_sum": float(i0["is_fishing"].sum()),
            "i1_hours_sum": float(i1["hours"].sum()),
            "i1_count": len(i1),
            "i1_fish_sum": float(i1["is_fishing"].sum()),
        })
        
    m_df = pd.DataFrame(month_stats)
    
    boot_diff_hours_warm = []
    boot_diff_rate_warm = []
    boot_diff_hours_iso = []
    boot_diff_rate_iso = []
    
    for _ in range(n_reps):
        sampled_indices = np.random.choice(n_months, size=n_months, replace=True)
        sample = m_df.iloc[sampled_indices]
        
        w0_h_mean = sample["w0_hours_sum"].sum() / max(1, sample["w0_count"].sum())
        w1_h_mean = sample["w1_hours_sum"].sum() / max(1, sample["w1_count"].sum())
        w0_f_rate = sample["w0_fish_sum"].sum() / max(1, sample["w0_count"].sum()) * 100
        w1_f_rate = sample["w1_fish_sum"].sum() / max(1, sample["w1_count"].sum()) * 100
        
        i0_h_mean = sample["i0_hours_sum"].sum() / max(1, sample["i0_count"].sum())
        i1_h_mean = sample["i1_hours_sum"].sum() / max(1, sample["i1_count"].sum())
        i0_f_rate = sample["i0_fish_sum"].sum() / max(1, sample["i0_count"].sum()) * 100
        i1_f_rate = sample["i1_fish_sum"].sum() / max(1, sample["i1_count"].sum()) * 100
        
        boot_diff_hours_warm.append(w1_h_mean - w0_h_mean)
        boot_diff_rate_warm.append(w1_f_rate - w0_f_rate)
        boot_diff_hours_iso.append(i1_h_mean - i0_h_mean)
        boot_diff_rate_iso.append(i1_f_rate - i0_f_rate)
        
    ci_dict = {
        "Warm_Anomaly_Hours_Diff": {
            "Mean": round(float(np.mean(boot_diff_hours_warm)), 3),
            "CI_95_Lower": round(float(np.percentile(boot_diff_hours_warm, 2.5)), 3),
            "CI_95_Upper": round(float(np.percentile(boot_diff_hours_warm, 97.5)), 3)
        },
        "Warm_Anomaly_Rate_Diff_Pct": {
            "Mean": round(float(np.mean(boot_diff_rate_warm)), 3),
            "CI_95_Lower": round(float(np.percentile(boot_diff_rate_warm, 2.5)), 3),
            "CI_95_Upper": round(float(np.percentile(boot_diff_rate_warm, 97.5)), 3)
        },
        "Iso_Disruption_Hours_Diff": {
            "Mean": round(float(np.mean(boot_diff_hours_iso)), 3),
            "CI_95_Lower": round(float(np.percentile(boot_diff_hours_iso, 2.5)), 3),
            "CI_95_Upper": round(float(np.percentile(boot_diff_hours_iso, 97.5)), 3)
        },
        "Iso_Disruption_Rate_Diff_Pct": {
            "Mean": round(float(np.mean(boot_diff_rate_iso)), 3),
            "CI_95_Lower": round(float(np.percentile(boot_diff_rate_iso, 2.5)), 3),
            "CI_95_Upper": round(float(np.percentile(boot_diff_rate_iso, 97.5)), 3)
        }
    }
    
    json_path = out_dir / "anomaly_bootstrap_cis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ci_dict, f, indent=2)
    logger.info(f"Saved block bootstrap CIs to: {json_path}")
    return ci_dict


def generate_anomaly_figures(df: pd.DataFrame, df_shares: pd.DataFrame, fig_dir: Path, p90_sst: float):
    """Generate diagnostic figures for the anomaly analysis."""
    logger.info(f"Generating anomaly figures in: {fig_dir}...")
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Anomaly Shares by Year and Region (Documenting Secular Confounding)
    plt.figure(figsize=(10, 5))
    for reg, col in [("arabian_sea", "#1f77b4"), ("bay_of_bengal", "#ff7f0e")]:
        sub = df_shares[df_shares["Region"] == reg]
        plt.plot(sub["Year"], sub["Warm_Anomaly_Share_Pct"], marker="o", linewidth=2.2, color=col, label=f"{reg.replace('_', ' ').title()} (Warm Anom %)")
        plt.plot(sub["Year"], sub["Iso_Disruption_Share_Pct"], marker="s", linestyle="--", linewidth=1.8, color=col, alpha=0.7, label=f"{reg.replace('_', ' ').title()} (Iso Disrupt %)")
    plt.xlabel("Year", fontsize=11)
    plt.ylabel("Share of Basin Cell-Months Flagged (%)", fontsize=11)
    plt.title("Secular Frequency of Ocean Anomalies by Basin (2019-2025)\n(Notice warm anomaly clustering in post-2022 period)", fontsize=11, fontweight="bold")
    plt.legend(fontsize=9)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_dir / "anomaly_shares_by_year.png", dpi=300)
    plt.close()
    
    # 2. SST Anomaly Distribution
    plt.figure(figsize=(9, 4))
    plt.hist(df["sst_anomaly"], bins=60, color="#3470a3", edgecolor="black", alpha=0.75, density=True)
    plt.axvline(p90_sst, color="red", linestyle="--", linewidth=2, label=f"90th Percentile (+{p90_sst:.2f}°C)")
    plt.xlabel("SST Anomaly (°C relative to monthly cell climatology)")
    plt.ylabel("Density")
    plt.title("Distribution of Monthly Sea Surface Temperature Anomalies")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_dir / "sst_anomaly_distribution.png", dpi=300)
    plt.close()
    
    # 3. Fishing Effort Comparison across Regimes (Boxplot)
    plt.figure(figsize=(11, 5))
    plt.subplot(1, 2, 1)
    data_w = [
        np.log1p(df[df["warm_anomaly"] == 0]["hours"]),
        np.log1p(df[df["warm_anomaly"] == 1]["hours"])
    ]
    plt.boxplot(data_w, tick_labels=["Normal SST", "Warm Anomaly\n(P90 / Z>2)"], patch_artist=True)
    plt.ylabel("log1p(Apparent Fishing Hours)")
    plt.title("Thermal Regime Comparison")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    
    plt.subplot(1, 2, 2)
    data_i = [
        np.log1p(df[df["iso_anomaly"] == 0]["hours"]),
        np.log1p(df[df["iso_anomaly"] == 1]["hours"])
    ]
    plt.boxplot(data_i, tick_labels=["Normal Ocean", "Multivariate\nDisruption"], patch_artist=True)
    plt.ylabel("log1p(Apparent Fishing Hours)")
    plt.title("Hydrodynamic Regime Comparison")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.suptitle("Apparent Fishing Effort Across Environmental Regimes", fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / "fishing_hours_anomaly_comparison.png", dpi=300)
    plt.close()
    
    # 4. Spatial Frequency Map of Anomalies
    cell_freq = df.groupby(["lat_bin", "lon_bin"])[["warm_anomaly", "iso_anomaly"]].mean().reset_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sc1 = axes[0].scatter(cell_freq["lon_bin"], cell_freq["lat_bin"], c=cell_freq["warm_anomaly"]*100, cmap="YlOrRd", s=12, marker="s")
    axes[0].set_title("Warm SST Anomaly Frequency (% Months)", fontweight="bold")
    axes[0].set_xlabel("Longitude (°E)")
    axes[0].set_ylabel("Latitude (°N)")
    plt.colorbar(sc1, ax=axes[0], label="% Months Flagged")
    axes[0].grid(True, linestyle=":", alpha=0.4)
    
    sc2 = axes[1].scatter(cell_freq["lon_bin"], cell_freq["lat_bin"], c=cell_freq["iso_anomaly"]*100, cmap="PuBuGn", s=12, marker="s")
    axes[1].set_title("Multivariate Disruption Frequency (% Months)", fontweight="bold")
    axes[1].set_xlabel("Longitude (°E)")
    axes[1].set_ylabel("Latitude (°N)")
    plt.colorbar(sc2, ax=axes[1], label="% Months Flagged")
    axes[1].grid(True, linestyle=":", alpha=0.4)
    
    plt.suptitle("Spatial Frequency of Oceanographic Anomalies (2019-2025)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / "anomaly_spatial_frequency.png", dpi=300)
    plt.close()
    logger.info("Anomaly figures generated successfully.")


def main():
    parser = argparse.ArgumentParser(description="Ocean anomaly detection and rigorous cross-domain fisheries impact.")
    parser.add_argument("--features-file", type=str, default=None, help="Path to features.parquet.")
    parser.add_argument("--config", type=str, default=None, help="Path to regions.yaml configuration.")
    args = parser.parse_args()
    
    config = load_config(Path(args.config) if args.config else None)
    seed = config["splits"]["random_seed"]
    set_seed(seed)
    
    oceanai_dir = get_oceanai_dir()
    features_path = Path(args.features_file) if args.features_file else oceanai_dir / "data" / "processed" / "integrated" / "features.parquet"
    out_dir = oceanai_dir / "data" / "processed" / "integrated"
    metrics_dir = oceanai_dir / "reports" / "metrics"
    figures_dir = oceanai_dir / "reports" / "figures"
    
    if not features_path.exists():
        raise FileNotFoundError(f"Missing {features_path}. Run features.py first.")
        
    logger.info(f"Loading feature dataset from: {features_path}")
    df_features = pd.read_parquet(features_path)
    
    df_anom, p90_sst = detect_anomalies(df_features, seed)
    
    # Save Anomalies Table
    anom_parquet = out_dir / "anomalies.parquet"
    df_anom.to_parquet(anom_parquet, index=False, compression="snappy")
    logger.info(f"Saved flagged anomaly panel to: {anom_parquet}")
    
    # 1. Shares by Year and Region
    df_shares = compute_anomaly_shares(df_anom, metrics_dir)
    
    # 2. Within-Year Comparisons
    compute_within_year_comparisons(df_anom, metrics_dir)
    
    # 3. Two-Way Fixed Effects Panel Regressions
    run_fixed_effects_regressions(df_anom, metrics_dir)
    
    # 4. Month-Block Bootstrap Confidence Intervals
    compute_block_bootstrap_cis(df_anom, metrics_dir, n_reps=1000, seed=seed)
    
    # 5. Diagnostic Figures
    generate_anomaly_figures(df_anom, df_shares, figures_dir, p90_sst)
    
    logger.info("=== STEP 5 ANOMALY ANALYSIS COMPLETE ===")


if __name__ == "__main__":
    main()
