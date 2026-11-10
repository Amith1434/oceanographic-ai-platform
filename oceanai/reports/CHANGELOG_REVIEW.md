# Oceanographic AI Platform — Review Round Hardening Changelog

**Date:** November 10, 2026  
**Auditor / Review Objective:** Hardening pipeline integrity, eliminating look-ahead leakage, de-confounding anomaly analysis, correcting biodiversity provenance, and establishing strict reproducibility (Zero Hand-Typed Numbers).

---

## 1. Executive Summary of Changes

| Review Item | Original Status / Finding | Hardened Implementation / Resolution | Status |
| :--- | :--- | :--- | :---: |
| **Feature Importance (Step 1)** | 14 features (`uo`, `vo` missing); split gain typed to sum to exactly 1.000; "67.2%" quoted in text without exact feature breakdown. | All 16 features evaluated. Computed normalized split gain, 5-repeat test-set permutation drops ($\Delta$ ROC-AUC), and native TreeSHAP attributions via XGBoost C++ API. Grouped shares calculated for Spatial (4), Seasonal (2), and Ocean Physics (10). | **FIXED** |
| **Sundarbans eDNA Metadata (Step 2)** | Misattributed to "Indian Sundarbans" with incorrect bounding box. | Corrected to **"UNESCO eDNA expedition in The Sundarbans (Bangladesh): February 2023"**, sampled 8–9 Jan 2023 across 19 sites in the Sundarbans West Wildlife Sanctuary (Bangladesh), actual bounding box (Lat 21.703521–21.828443°N, Lon 89.195000–89.284180°E), DOI: `10.25607/rt4e3h`. Exactly 476 records (39.34% of OBIS, 100% of *Setipinna tenuifilis*). Estuarine resolution caveat added. | **FIXED** |
| **OBIS Matching (Step 3)** | Pre-2019 records imputed using all-months mean, conflating different time periods. | Implemented strict 4-category classification (`exact_match`, `nearest_cell_same_month` $\le 0.5^\circ$, `no_match_pre2019`, `no_match_far`). Pre-2019 records (69 total) and distant records (1 total) strictly dropped. Match quality table and realized environmental envelopes regenerated on valid records only. | **FIXED** |
| **Climatology Leakage (Step 4)** | Climatological anomalies computed over full 2019–2025 period, leaking test-set statistics into training features. | Model features (`sst_anomaly`, `so_anomaly`, `sst_zscore`) recomputed strictly on train-only climatology (2019–2023). Separate `*_fullperiod` features saved solely for interactive map rendering. Lags partitioned without leakage. | **FIXED** |
| **Anomaly Confounding (Step 5)** | Unconditioned pooling suggested warm anomalies had +47% more fishing hours, confounded with GFW 2022 satellite expansion. | Conducted within-year regional comparisons (showing 4 of 7 years had negative fishing differentials). Implemented Two-Way Fixed Effects (TWFE, Cell FE + Month FE with clustered SE) demonstrating modest marginal effect ($\beta = +0.0630$ on log-hours). Evaluated detrended SST anomalies and 1,000-rep month-block bootstrap CIs. Softened causal claims to observational associations consistent with hypotheses. | **FIXED** |
| **Model Evaluation (Step 6)** | Decision threshold fixed at 0.5; log retransformation uncorrected; baselines lacked raw hours error; ablation limited to 2 models. | Added Predict-Zero baseline (Raw MAE = 17.25 hrs, outperforming Climatology and Persistence). Tuned decision threshold on 2024 validation set ($\tau^* = 0.16$, Val F1 = 0.5884). Computed Duan smearing factor ($S = 2.4556$). Executed expanded 5-model ablation (M1, M2, M2b, M3, M4) across Pooled, Arabian Sea, and Bay of Bengal panels with bootstrap CIs on deltas. Evaluated rolling-origin distribution shifts. | **FIXED** |
| **Report Tone & Verification (Step 7)** | Potential overclaims; hand-typed numbers in markdown tables. | Softened causal language. Built `oceanai/reports/build_report_tables.py` to programmatically extract and format all markdown tables directly from saved JSON/CSV metrics files. Zero hand-typed numbers. | **FIXED** |
| **Walkthrough Notebook (Step 8)** | 12 tab characters and 1 form-feed corrupted LaTeX keywords (`\text`, `\frac`, `\approx`). | Cleaned all markdown cells; added automated assertions ensuring 0 tabs, form-feeds, or control characters remain. | **FIXED** |
| **Interactive Dashboard (Step 9)** | Missing side-by-side synchronized explorer, match quality table, and within-year anomaly plots. | Added dedicated `1_Ocean_Fishing_Sync.py` with synchronized Plotly maps, tooltips, and correlation scatter. Added match quality table, exclude-eDNA toggle, within-year anomaly charts, and caveat expanders across all pages. | **FIXED** |
| **Testing & Automation (Step 10)** | Test suite missing; Makefile lacked unified verification. | Created `oceanai/tests/test_pipeline.py` with pytest covering hours conservation, zero-filling, no leakage, split bounds, match categories, 16 features. Updated `Makefile` with `make all`. | **FIXED** |

---

## 2. Quantitative Comparison: Old vs. Hardened Metrics

### 2.1 Baseline Benchmarks (2025 Test Set)

| Metric | Historical Climatology | 1-Month Persistence | Predict-Zero Baseline |
| :--- | :---: | :---: | :---: |
| **Stage A Test ROC-AUC** | 0.7672 | 0.7024 | -- |
| **Stage A Test PR-AUC** | 0.4457 | 0.4051 | -- |
| **Stage B Test $R^2$** | -0.0183 | 0.0194 | -- |
| **Stage B Test RMSE (log)** | 1.5810 | 1.5514 | -- |
| **Combined Hurdle Raw RMSE** | 113.88 hrs | 114.21 hrs | **113.06 hrs** |
| **Combined Hurdle Raw MAE** | 17.98 hrs | 18.12 hrs | **17.25 hrs** |

> **Key Insight:** Because 78.94% of ocean cells host zero fishing activity, the naive Predict-Zero baseline achieves a lower raw MAE (17.25 hrs) than either Climatology (17.98 hrs) or Persistence (18.12 hrs).

### 2.2 Model 2 Validation Tuning & Hurdle Performance

| Metric | Standard (Uncorrected) | Duan Smeared ($S = 2.4556$) |
| :--- | :---: | :---: |
| **Decision Threshold ($\tau^*$)** | 0.16 (Val F1 = 0.5884) | 0.16 (Val F1 = 0.5884) |
| **Raw Test RMSE** | 107.39 hrs | **101.12 hrs** (5.8% reduction) |
| **Raw Test MAE** | **17.49 hrs** | 19.18 hrs |
| **Spearman Rank Correlation ($\rho$)** | 0.4710 | **0.4711** |

### 2.3 Expanded 5-Model Ablation Study Across Geographic Panels

| Panel | Model Specification | Features | Stage A ROC-AUC | Stage A PR-AUC | Stage A F1 | Stage B $R^2$ | Stage B RMSE (log) | Stage B Spearman $\rho$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pooled** | M1 (Spatial+Seasonal) | 6 | **0.8051** | **0.4941** | 0.5633 | **0.2060** | **1.3960** | **0.3792** |
| **Pooled** | M2 (Spatial+Seasonal+Ocean) | 16 | 0.7984 | 0.4901 | 0.5587 | 0.1768 | 1.4214 | 0.3641 |
| **Pooled** | M2b (Spatial+Seasonal+Anomalies) | 8 | 0.7936 | 0.4909 | 0.5451 | 0.1799 | 1.4188 | 0.3664 |
| **Pooled** | M3 (Spatial+Seasonal+Persistence) | 8 | **0.8430** | **0.6364** | 0.5948 | **0.3052** | **1.3060** | **0.4410** |
| **Pooled** | M4 (Full+Persistence+Ocean) | 18 | 0.8389 | 0.6279 | 0.5892 | 0.2951 | 1.3154 | 0.4319 |
| **Arabian Sea** | M1 (Spatial+Seasonal) | 6 | **0.7759** | **0.4843** | 0.5547 | **0.1335** | **1.3688** | 0.3110 |
| **Arabian Sea** | M2 (Spatial+Seasonal+Ocean) | 16 | 0.7719 | 0.4698 | 0.5510 | 0.1208 | 1.3788 | **0.3167** |
| **Arabian Sea** | M3 (Spatial+Seasonal+Persistence) | 8 | **0.8108** | **0.5748** | 0.5780 | **0.1944** | **1.3199** | **0.3612** |
| **Arabian Sea** | M4 (Full+Persistence+Ocean) | 18 | 0.8094 | 0.5786 | 0.5787 | 0.1889 | 1.3243 | 0.3596 |
| **Bay of Bengal**| M1 (Spatial+Seasonal) | 6 | **0.8807** | **0.6335** | 0.6265 | **0.3296** | **1.4122** | **0.4985** |
| **Bay of Bengal**| M2 (Spatial+Seasonal+Ocean) | 16 | 0.8525 | 0.5913 | 0.5755 | 0.2663 | 1.4773 | 0.4710 |
| **Bay of Bengal**| M3 (Spatial+Seasonal+Persistence) | 8 | **0.8979** | **0.7395** | 0.6424 | **0.4533** | **1.2752** | **0.5550** |
| **Bay of Bengal**| M4 (Full+Persistence+Ocean) | 18 | 0.8862 | 0.7209 | 0.6362 | 0.4387 | 1.2921 | 0.5525 |

### 2.4 OBIS 4-Category Matching Quality

| Species | Total Records | Pre-2019 Dropped | Exact Matches | Nearest Same-Month ($\le 0.5^\circ$) | Far Dropped ($> 0.5^\circ$) | Usable Records | Usable Share (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| *Carcharhinus melanopterus* | 595 | 51 | 541 | 3 | 0 | 544 | 91.43% |
| *Rhincodon typus* | 139 | 18 | 119 | 1 | 1 | 120 | 86.33% |
| *Setipinna tenuifilis* | 476 | 0 | 126 | 350 | 0 | 476 | 100.00% |
| **Total** | **1,210** | **69** | **786** | **354** | **1** | **1,140** | **94.21%** |

### 2.5 Two-Way Fixed Effects (TWFE) Anomaly Regressions

| Specification | Dependent Variable | Regressor | Coefficient ($\beta$) | Clustered SE | $t$-statistic | $p$-value | 95% Confidence Interval |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `warm_anomaly_on_log_hours` | `log_hours` | `warm_anomaly` | **+0.0630** | 0.0043 | 14.58 | $3.81 \times 10^{-48}$ | [+0.0545, +0.0715] |
| `warm_anomaly_on_is_fishing` | `is_fishing` | `warm_anomaly` | **+0.0212** | 0.0012 | 17.01 | $7.48 \times 10^{-65}$ | [+0.0187, +0.0236] |
| `warm_detrended_on_log_hours` | `log_hours` | `warm_anomaly_detrended` | **+0.0528** | 0.0044 | 11.99 | $3.91 \times 10^{-33}$ | [+0.0442, +0.0614] |
| `iso_anomaly_on_log_hours` | `log_hours` | `iso_anomaly` | **-0.1185** | 0.0048 | -24.74 | $3.94 \times 10^{-135}$ | [-0.1279, -0.1091] |
| `iso_anomaly_on_is_fishing` | `is_fishing` | `iso_anomaly` | **-0.0322** | 0.0015 | -22.02 | $2.31 \times 10^{-107}$ | [-0.0351, -0.0293] |

---

## 3. Items Maintained / Guardrails Preserved

1. **Strict Random Seed:** Seed 42 maintained across all random number generators.
2. **Absolute "No-Year" Feature Rule:** `year` remains strictly excluded from all machine learning feature spaces.
3. **Temporal Split Structure:** Train 2019–2023 (60 months), Val 2024 (12 months), Test 2025 (12 months).
4. **Descriptive Biodiversity Policy:** No Species Distribution Modeling (MaxEnt/GAM) performed on OBIS records due to the 39.34% single-cruise eDNA sampling artifact.
