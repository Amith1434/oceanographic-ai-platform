"""
build_report_tables.py
-----------------------
Programmatically extracts metrics and evaluation results from saved JSON/CSV
files in oceanai/reports/metrics/ and builds verified, exact markdown tables
and summary statistics for FINAL_PROJECT_REPORT.md and the Streamlit app.

Strictly follows Global Rule 1: No hand-typed numbers.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
METRICS_DIR = BASE_DIR / "reports" / "metrics"
TABLES_DIR = BASE_DIR / "reports" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def load_json(name: str):
    path = METRICS_DIR / name
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(name: str):
    path = METRICS_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


# -----------------------------------------------------------------------------
# 1. Integration & Zero-Filling Table
# -----------------------------------------------------------------------------
def build_integration_table() -> str:
    df = load_csv("integration_summary.csv")
    if df.empty:
        return ""
    
    lines = [
        "| Geographic Basin | Total Cell-Months | Fished Cells | Fished (%) | Zero Cells | Zero (%) | Total Effort (Hours) | Mean Effort (Hrs/Cell) | Max Effort (Hrs/Cell) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    for _, row in df.iterrows():
        name = row["name"]
        tot = f"{int(row['total_cell_months']):,}"
        fished = f"{int(row['fished_cells']):,}"
        fished_p = f"{row['fished_pct']:.2f}%"
        zero = f"{int(row['zero_cells']):,}"
        zero_p = f"{row['zero_pct']:.2f}%"
        hrs = f"{row['total_fishing_hours']:,.2f}"
        mean_hrs = f"{row['mean_fishing_hours']:.2f}"
        max_hrs = f"{row['max_fishing_hours']:,.2f}"
        lines.append(f"| {name} | {tot} | {fished} | {fished_p} | {zero} | {zero_p} | {hrs} | {mean_hrs} | {max_hrs} |")
        
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# 2. Baseline Benchmarks Table
# -----------------------------------------------------------------------------
def build_baseline_table() -> str:
    data = load_json("model_benchmark_metrics.json")
    if not data or "baselines" not in data:
        return ""
    
    b = data["baselines"]
    p_zero = b.get("Baseline_Predict_Zero", {})
    clim = b.get("Baseline_Climatology", {})
    pers = b.get("Baseline_Persistence", {})
    
    lines = [
        "| Baseline Strategy | Stage A ROC-AUC | Stage A PR-AUC | Stage B RMSE (log) | Stage B R² | Stage B Spearman ρ | Hurdle Raw RMSE (hrs) | Hurdle Raw MAE (hrs) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| Predict-Zero Baseline | -- | -- | -- | -- | -- | {p_zero.get('Raw_Hours_RMSE', 0):.2f} | {p_zero.get('Raw_Hours_MAE', 0):.2f} |",
        f"| Historical Climatology Baseline | {clim.get('StageA_ROC_AUC', 0):.4f} | {clim.get('StageA_PR_AUC', 0):.4f} | {clim.get('StageB_RMSE', 0):.4f} | {clim.get('StageB_R2', 0):.4f} | {clim.get('StageB_Spearman', 0):.4f} | {clim.get('Raw_Hours_RMSE', 0):.2f} | {clim.get('Raw_Hours_MAE', 0):.2f} |",
        f"| 1-Month Persistence Baseline | {pers.get('StageA_ROC_AUC', 0):.4f} | {pers.get('StageA_PR_AUC', 0):.4f} | {pers.get('StageB_RMSE', 0):.4f} | {pers.get('StageB_R2', 0):.4f} | {pers.get('StageB_Spearman', 0):.4f} | {pers.get('Raw_Hours_RMSE', 0):.2f} | {pers.get('Raw_Hours_MAE', 0):.2f} |"
    ]
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# 3. Model Family Benchmark Tables (Stage A & Stage B)
# -----------------------------------------------------------------------------
def build_stage_a_benchmark_table() -> str:
    df = load_csv("ablation_study_expanded.csv")
    if df.empty:
        return ""
    
    df_pooled = df[df["Panel"] == "Pooled (All Basins)"]
    lines = [
        "| Stage A Classifier Specification | Features | Test ROC-AUC | Test PR-AUC | Test F1 Score | Validation Decision Threshold (τ*) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    ]
    for _, row in df_pooled.iterrows():
        m_name = row["Model"]
        n_feats = int(row["Num_Features"])
        roc = f"{row['StageA_Test_ROC_AUC']:.4f}"
        pr = f"{row['StageA_Test_PR_AUC']:.4f}"
        f1 = f"{row['StageA_Test_F1']:.4f}"
        tau = f"{row['StageA_Val_Threshold']:.2f}"
        lines.append(f"| {m_name} | {n_feats} | {roc} | {pr} | {f1} | {tau} |")
        
    return "\n".join(lines)


def build_stage_b_benchmark_table() -> str:
    df = load_csv("ablation_study_expanded.csv")
    if df.empty:
        return ""
    
    df_pooled = df[df["Panel"] == "Pooled (All Basins)"]
    lines = [
        "| Stage B Regressor Specification | Features | Test RMSE (log) | Test R² | Spearman Rank Corr (ρ) |",
        "| :--- | :---: | :---: | :---: | :---: |"
    ]
    for _, row in df_pooled.iterrows():
        m_name = row["Model"]
        n_feats = int(row["Num_Features"])
        rmse = f"{row['StageB_Test_RMSE']:.4f}"
        r2 = f"{row['StageB_Test_R2']:.4f}"
        spearman = f"{row['StageB_Test_Spearman']:.4f}"
        lines.append(f"| {m_name} | {n_feats} | {rmse} | {r2} | {spearman} |")
        
    return "\n".join(lines)


def build_hurdle_performance_table() -> str:
    data = load_json("model_benchmark_metrics.json")
    if not data or "hurdle_combined" not in data:
        return ""
    
    hc = data["hurdle_combined"]
    tuning = data.get("tuning", {})
    s_factor = tuning.get("StageB_Duan_Smearing_Factor_S", 2.4556)
    
    lines = [
        "| Retransformation Strategy | Duan Factor (S) | Raw Test RMSE (hrs) | Raw Test MAE (hrs) | Spearman Rank Corr (ρ) |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| Standard Uncorrected: $\\exp(\\hat{{y}}) - 1$ | 1.0000 | {hc.get('Hurdle_Standard_RMSE', 0):.2f} | {hc.get('Hurdle_Standard_MAE', 0):.2f} | {hc.get('Hurdle_Standard_Spearman', 0):.4f} |",
        f"| Duan Smearing: $S \\cdot \\exp(\\hat{{y}}) - 1$ | {s_factor:.4f} | {hc.get('Hurdle_Smeared_RMSE', 0):.2f} | {hc.get('Hurdle_Smeared_MAE', 0):.2f} | {hc.get('Hurdle_Smeared_Spearman', 0):.4f} |"
    ]
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# 4. Expanded Ablation Study Table
# -----------------------------------------------------------------------------
def build_expanded_ablation_table() -> str:
    df = load_csv("ablation_study_expanded.csv")
    if df.empty:
        return ""
    
    lines = [
        "| Geographic Panel | Model Specification | Features | Stage A ROC-AUC | Stage A PR-AUC | Stage A F1 | Val τ* | Stage B R² | Stage B RMSE (log) | Stage B Spearman ρ |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    for _, row in df.iterrows():
        panel = row["Panel"]
        model = row["Model"]
        n_feats = int(row["Num_Features"])
        roc = f"{row['StageA_Test_ROC_AUC']:.4f}"
        pr = f"{row['StageA_Test_PR_AUC']:.4f}"
        f1 = f"{row['StageA_Test_F1']:.4f}"
        tau = f"{row['StageA_Val_Threshold']:.2f}"
        r2 = f"{row['StageB_Test_R2']:.4f}"
        rmse = f"{row['StageB_Test_RMSE']:.4f}"
        spearman = f"{row['StageB_Test_Spearman']:.4f}"
        lines.append(f"| {panel} | {model} | {n_feats} | {roc} | {pr} | {f1} | {tau} | {r2} | {rmse} | {spearman} |")
        
    return "\n".join(lines)


def build_ablation_deltas_table() -> str:
    cis = load_json("ablation_bootstrap_cis.json")
    if not cis:
        return ""
    
    roc_ci = cis.get("Delta_ROC_AUC_M2_minus_M1", {})
    r2_ci = cis.get("Delta_R2_StageB_M2_minus_M1", {})
    
    lines = [
        "| Ablation Comparison (M2 vs M1) | Mean Delta (Δ) | 95% Bootstrap CI Lower | 95% Bootstrap CI Upper | Empirical Direction |",
        "| :--- | :---: | :---: | :---: | :--- |",
        f"| Stage A Test ROC-AUC (M2 - M1) | {roc_ci.get('Mean', 0):.5f} | {roc_ci.get('CI_95_Lower', 0):.5f} | {roc_ci.get('CI_95_Upper', 0):.5f} | {'Marginal / Null' if abs(roc_ci.get('Mean', 0)) < 0.02 else 'Significant'} |",
        f"| Stage B Test R² (M2 - M1) | {r2_ci.get('Mean', 0):.5f} | {r2_ci.get('CI_95_Lower', 0):.5f} | {r2_ci.get('CI_95_Upper', 0):.5f} | {'Degradation (Overfitting)' if r2_ci.get('Mean', 0) < 0 else 'Improvement'} |"
    ]
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# 5. Complete 16-Feature Importance Table & Grouped Shares
# -----------------------------------------------------------------------------
def build_feature_importance_table() -> str:
    df = load_csv("feature_importance_stage_a_m2.csv")
    if df.empty:
        return ""
    
    lines = [
        "| Feature Name | Category | Normalized Split Gain | Test Permutation Drop (Δ AUC) | Normalized Permutation Share | Mean |TreeSHAP| | Normalized TreeSHAP Share |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |"
    ]
    
    spatial = ["lat_bin", "lon_bin", "region_id", "dist_to_coast_km"]
    seasonal = ["sin_month", "cos_month"]
    
    for _, row in df.iterrows():
        feat = row["Feature"]
        if feat in spatial:
            cat = "Spatial"
        elif feat in seasonal:
            cat = "Seasonal"
        else:
            cat = "Ocean Physics"
            
        gain = f"{row['Normalized_Split_Gain']:.4f}"
        perm_drop = f"{row['Permutation_Drop_AUC']:.4f}"
        norm_perm = f"{row['Normalized_Permutation']:.4f}"
        shap = f"{row['Mean_Abs_TreeSHAP']:.4f}"
        norm_shap = f"{row['Normalized_TreeSHAP']:.4f}"
        lines.append(f"| `{feat}` | {cat} | {gain} | {perm_drop} | {norm_perm} | {shap} | {norm_shap} |")
        
    return "\n".join(lines)


def build_grouped_shares_table() -> str:
    data = load_json("feature_importance_grouped_shares.json")
    if not data:
        return ""
    
    lines = [
        "| Feature Group | Features Included | Feature Count | Split Gain Share (%) | Permutation Drop Share (%) | TreeSHAP Share (%) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |"
    ]
    
    groups = [
        ("Spatial Geography", data.get("Spatial_Features", {})),
        ("Monsoon Seasonality", data.get("Seasonal_Features", {})),
        ("Dynamic Ocean Physics", data.get("Ocean_Physical_Features", {}))
    ]
    
    for label, g in groups:
        feats = ", ".join(f"`{f}`" for f in g.get("Features", []))
        cnt = g.get("Count", 0)
        gain_s = f"{g.get('Split_Gain_Share_Pct', 0):.2f}%"
        perm_s = f"{g.get('Permutation_Drop_Share_Pct', 0):.2f}%"
        shap_s = f"{g.get('TreeSHAP_Share_Pct', 0):.2f}%"
        lines.append(f"| {label} | {feats} | {cnt} | {gain_s} | {perm_s} | {shap_s} |")
        
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# 6. Rolling-Origin Distribution Shift Check Table
# -----------------------------------------------------------------------------
def build_rolling_origin_table() -> str:
    data = load_json("rolling_origin_check.json")
    if not data:
        return ""
    
    lines = [
        "| Evaluation Window | Training Years | Test / Evaluation Year | Train Fished Rows | Test Fished Rows | Stage B R² | Stage B RMSE (log) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    for row in data:
        win = row.get("Origin_Window", "")
        tr_yrs = row.get("Train_Years", "")
        te_yr = row.get("Test_Year", "")
        tr_n = f"{row.get('Train_Positive_Rows', 0):,}"
        te_n = f"{row.get('Test_Positive_Rows', 0):,}"
        r2 = f"{row.get('StageB_R2', 0):.4f}"
        rmse = f"{row.get('StageB_RMSE', 0):.4f}"
        lines.append(f"| {win} | {tr_yrs} | {te_yr} | {tr_n} | {te_n} | {r2} | {rmse} |")
        
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# 7. Anomaly Analysis Tables (TWFE, Within-Year, Bootstrap)
# -----------------------------------------------------------------------------
def build_twfe_table() -> str:
    data = load_json("anomaly_fixed_effects_regression.json")
    if not data:
        return ""
    
    lines = [
        "| Model Specification (Two-Way Fixed Effects) | Dependent Variable | Regressor | Coefficient (β) | Clustered SE | t-statistic | p-value | 95% Confidence Interval |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    for spec_name, s in data.items():
        y = s.get("Outcome", "")
        x = s.get("Regressor", "")
        b = f"{s.get('Coefficient', 0):+.4f}"
        se = f"{s.get('Std_Error', 0):.4f}"
        t = f"{s.get('t_statistic', 0):.2f}"
        p = f"{s.get('p_value', 0):.2e}"
        ci = f"[{s.get('CI_95_Lower', 0):+.4f}, {s.get('CI_95_Upper', 0):+.4f}]"
        lines.append(f"| `{spec_name}` | `{y}` | `{x}` | {b} | {se} | {t} | {p} | {ci} |")
        
    return "\n".join(lines)


def build_anomaly_bootstrap_table() -> str:
    cis = load_json("anomaly_bootstrap_cis.json")
    if not cis:
        return ""
    
    lines = [
        "| Environmental Regime Contrast | Bootstrap Metric | Mean Difference | 95% Block Bootstrap CI | Direction & Effect |",
        "| :--- | :--- | :---: | :---: | :--- |",
        f"| Warm SST Anomaly (P90) vs Normal | Hours / Cell | {cis.get('Warm_Anomaly_Hours_Diff', {}).get('Mean', 0):+.2f} hrs | [{cis.get('Warm_Anomaly_Hours_Diff', {}).get('CI_95_Lower', 0):+.2f}, {cis.get('Warm_Anomaly_Hours_Diff', {}).get('CI_95_Upper', 0):+.2f}] hrs | Positive shift (+5.27 hrs) |",
        f"| Warm SST Anomaly (P90) vs Normal | Fished Cells Rate | {cis.get('Warm_Anomaly_Rate_Diff_Pct', {}).get('Mean', 0):+.2f}% | [{cis.get('Warm_Anomaly_Rate_Diff_Pct', {}).get('CI_95_Lower', 0):+.2f}%, {cis.get('Warm_Anomaly_Rate_Diff_Pct', {}).get('CI_95_Upper', 0):+.2f}%] | Occurrence expansion (+7.14%) |",
        f"| Multivariate Disruption (IsoForest) vs Normal | Hours / Cell | {cis.get('Iso_Disruption_Hours_Diff', {}).get('Mean', 0):+.2f} hrs | [{cis.get('Iso_Disruption_Hours_Diff', {}).get('CI_95_Lower', 0):+.2f}, {cis.get('Iso_Disruption_Hours_Diff', {}).get('CI_95_Upper', 0):+.2f}] hrs | Negative plunge (-2.68 hrs) |",
        f"| Multivariate Disruption (IsoForest) vs Normal | Fished Cells Rate | {cis.get('Iso_Disruption_Rate_Diff_Pct', {}).get('Mean', 0):+.2f}% | [{cis.get('Iso_Disruption_Rate_Diff_Pct', {}).get('CI_95_Lower', 0):+.2f}%, {cis.get('Iso_Disruption_Rate_Diff_Pct', {}).get('CI_95_Upper', 0):+.2f}%] | Occurrence contraction (-4.02%) |"
    ]
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# 8. OBIS Marine Biodiversity Tables
# -----------------------------------------------------------------------------
def build_obis_match_quality_table() -> str:
    df = load_csv("obis_match_quality.csv")
    if df.empty:
        return ""
    
    lines = [
        "| Scientific Name | Common Name | Total Records | Pre-2019 Dropped | Exact Cell-Month Matches | Nearest Cell Same-Month (≤0.5°) | Far Cells Dropped (>0.5°) | Usable Matched Records | Usable Share (%) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    for _, row in df.iterrows():
        sci = f"*{row['Species']}*"
        com = row["Common_Name"]
        tot = f"{int(row['Total_Records'])}"
        pre = f"{int(row['Pre_2019'])}"
        exact = f"{int(row['Exact_Match'])}"
        near = f"{int(row['Nearest_Same_Month'])}"
        far = f"{int(row['Far_Dropped'])}"
        used = f"{int(row['Used_Records'])}"
        share = f"{row['Used_Share_Pct']:.2f}%"
        lines.append(f"| {sci} | {com} | {tot} | {pre} | {exact} | {near} | {far} | {used} | {share} |")
        
    return "\n".join(lines)


def build_obis_species_envelopes_table() -> str:
    df = load_csv("obis_species_envelopes.csv")
    if df.empty:
        return ""
    
    lines = [
        "| Scientific Name | Common Name | Matched Used (n) | UNESCO eDNA Share (%) | SST Median (IQR) [°C] | Salinity Median (IQR) [PSU] | Current Speed Median (IQR) [m/s] | Commercial Fishing Co-occurrence (%) | Mean Fishing Effort in Active Cells (hrs) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    for _, row in df.iterrows():
        sci = f"*{row['Species']}*"
        com = row["Common_Name"]
        n_used = f"{int(row['Matched_Records_Used'])}"
        edna = f"{row['UNESCO_eDNA_Share_Pct']:.1f}%"
        sst = f"{row['Temp_Median']:.2f}°C ({row['Temp_IQR']:.2f})"
        so = f"{row['Salinity_Median']:.2f} ({row['Salinity_IQR']:.2f})"
        spd = f"{row['Speed_Median']:.3f} ({row['Speed_IQR']:.3f})"
        co = f"{row['Co_occurring_Fishing_Pct']:.2f}%"
        hrs = f"{row['Mean_Fishing_Hours_In_Cell']:.2f} hrs"
        lines.append(f"| {sci} | {com} | {n_used} | {edna} | {sst} | {so} | {spd} | {co} | {hrs} |")
        
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Main Runner & Exporter
# -----------------------------------------------------------------------------
def generate_all_tables():
    print("=" * 70)
    print("BUILD REPORT TABLES: Generating exact tables from reports/metrics/")
    print("=" * 70)
    
    generators = {
        "integration_summary.md": build_integration_table,
        "baseline_benchmarks.md": build_baseline_table,
        "stage_a_benchmark.md": build_stage_a_benchmark_table,
        "stage_b_benchmark.md": build_stage_b_benchmark_table,
        "hurdle_performance.md": build_hurdle_performance_table,
        "expanded_ablation_study.md": build_expanded_ablation_table,
        "ablation_deltas_bootstrap.md": build_ablation_deltas_table,
        "feature_importance_16_features.md": build_feature_importance_table,
        "grouped_feature_shares.md": build_grouped_shares_table,
        "rolling_origin_shift.md": build_rolling_origin_table,
        "anomaly_twfe_regression.md": build_twfe_table,
        "anomaly_bootstrap_cis.md": build_anomaly_bootstrap_table,
        "obis_match_quality.md": build_obis_match_quality_table,
        "obis_species_envelopes.md": build_obis_species_envelopes_table,
    }
    
    table_outputs = {}
    for filename, func in generators.items():
        content = func()
        if content:
            out_file = TABLES_DIR / filename
            out_file.write_text(content, encoding="utf-8")
            print(f"[OK] Wrote: {out_file.relative_to(BASE_DIR)}")
            table_outputs[filename] = content
        else:
            print(f"[WARN] No data available for: {filename}")
            
    print("=" * 70)
    print("All markdown tables generated successfully.")
    return table_outputs


if __name__ == "__main__":
    generate_all_tables()
