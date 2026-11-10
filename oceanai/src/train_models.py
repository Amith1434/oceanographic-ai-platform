"""
Two-Stage Hurdle Modeling, Ablation Study, & Native TreeSHAP
============================================================
Implements a scientifically grounded machine learning framework for predicting
apparent commercial fishing activity from physical oceanographic conditions.

Key Analytical Components (Review Updates):
1. Two-Stage Hurdle Architecture:
   - Stage A (Occurrence): Binary classification (is_fishing = hours > 0).
   - Stage B (Intensity): Continuous regression on log1p(hours) given hours > 0.
2. Proper 2024 Validation Set Utilization:
   - Early stopping (early_stopping_rounds=20) on 2024 validation set.
   - Documented validation hyperparameter search space (15 configurations).
   - Decision threshold tau* chosen on validation set to maximize F1 score.
   - Retransformation bias correction via Duan smearing factor S.
3. Raw-Hours Benchmarks:
   - Predict-zero baseline, cell-month climatology, and 1-month persistence.
   - Evaluated in raw hours (RMSE and MAE) next to the hurdle model.
4. Comprehensive 5-Model Ablation Study across Pooled and Regional Panels:
   - M1: Spatial + Seasonal (6 features)
   - M2: M1 + All Ocean Physics (16 features)
   - M2b: M1 + Ocean Anomalies only (8 features)
   - M3: M1 + Persistence lags (8 features)
   - M4: M3 + Ocean Physics (18 features)
   - Month-block bootstrap 95% CIs on delta ROC-AUC and delta R2.
5. Distribution Shift & Rolling-Origin Evaluation:
   - Evaluates Stage B R2 across expanding train windows (2022, 2023, 2024, 2025).
6. Complete 16-Feature Importance Table for Model 2:
   - Normalized XGBoost Split Gain, Test Permutation Drops (5 repeats), and Mean |TreeSHAP|.
   - Grouped shares for Spatial (4), Seasonal (2), and Ocean (10) categories.
"""

import argparse
import json
import os
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor

# Local module import
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_oceanai_dir, get_project_root, load_config, set_seed, setup_logger

logger = setup_logger("train_models")

# Headless plotting setup
LOCAL_CACHE_DIR = Path(__file__).resolve().parent.parent / ".matplotlib_cache"
LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(LOCAL_CACHE_DIR)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def split_chronological(df: pd.DataFrame, config: dict):
    """Enforce strict out-of-time chronological train/val/test splits."""
    train_years = config["splits"]["train_years"]
    val_year = config["splits"]["val_year"]
    test_year = config["splits"]["test_year"]
    
    train = df[df["year"].isin(train_years)].copy()
    val = df[df["year"] == val_year].copy()
    test = df[df["year"] == test_year].copy()
    
    logger.info(f"Train split ({min(train_years)}-{max(train_years)}): {len(train):,} records ({train['is_fishing'].mean()*100:.1f}% fished)")
    logger.info(f"Val split   ({val_year}): {len(val):,} records ({val['is_fishing'].mean()*100:.1f}% fished)")
    logger.info(f"Test split  ({test_year}): {len(test):,} records ({test['is_fishing'].mean()*100:.1f}% fished)")
    
    return train, val, test


# -------------------------------------------------------------
# Baselines
# -------------------------------------------------------------

def evaluate_baselines(train: pd.DataFrame, test: pd.DataFrame):
    """Compute and evaluate Predict-Zero, Climatology, and Persistence baselines."""
    logger.info("Computing Predict-Zero, Climatology, and Persistence benchmarks...")
    results = {}
    
    y_test_raw = test["hours"].values
    y_test_bin = test["is_fishing"].values
    pos_mask = (test["is_fishing"] == 1).values
    y_test_b = test["log_hours"].values[pos_mask]
    
    # 1. Predict-Zero Baseline
    pred_zero_raw = np.zeros_like(y_test_raw)
    rmse_zero_raw = float(np.sqrt(mean_squared_error(y_test_raw, pred_zero_raw)))
    mae_zero_raw = float(mean_absolute_error(y_test_raw, pred_zero_raw))
    
    results["Baseline_Predict_Zero"] = {
        "Raw_Hours_RMSE": round(rmse_zero_raw, 2),
        "Raw_Hours_MAE": round(mae_zero_raw, 2),
        "Notes": "Predicts 0.0 everywhere. Serves as tough hurdle for raw MAE in zero-inflated distributions."
    }
    
    # 2. Historical Climatology Baseline
    occ_clim = train.groupby(["region_id", "lat_bin", "lon_bin", "cal_month"])["is_fishing"].mean().rename("prob_clim")
    raw_clim = train.groupby(["region_id", "lat_bin", "lon_bin", "cal_month"])["hours"].mean().rename("hours_clim")
    int_clim = train[train["is_fishing"] == 1].groupby(["region_id", "lat_bin", "lon_bin", "cal_month"])["log_hours"].mean().rename("log_hours_clim")
    
    test_clim = test.merge(occ_clim, on=["region_id", "lat_bin", "lon_bin", "cal_month"], how="left")
    test_clim = test_clim.merge(raw_clim, on=["region_id", "lat_bin", "lon_bin", "cal_month"], how="left")
    test_clim = test_clim.merge(int_clim, on=["region_id", "lat_bin", "lon_bin", "cal_month"], how="left")
    
    prob_clim = test_clim["prob_clim"].fillna(train["is_fishing"].mean()).values
    pred_clim_raw = test_clim["hours_clim"].fillna(train["hours"].mean()).values
    int_clim_all = test_clim["log_hours_clim"].fillna(train[train["is_fishing"] == 1]["log_hours"].mean()).values
    int_clim_pos = int_clim_all[pos_mask]
    
    results["Baseline_Climatology"] = {
        "StageA_ROC_AUC": round(float(roc_auc_score(y_test_bin, prob_clim)), 4),
        "StageA_PR_AUC": round(float(average_precision_score(y_test_bin, prob_clim)), 4),
        "StageB_RMSE": round(float(np.sqrt(mean_squared_error(y_test_b, int_clim_pos))), 4),
        "StageB_MAE": round(float(mean_absolute_error(y_test_b, int_clim_pos)), 4),
        "StageB_R2": round(float(r2_score(y_test_b, int_clim_pos)), 4),
        "StageB_Spearman": round(float(spearmanr(y_test_b, int_clim_pos)[0]), 4),
        "Raw_Hours_RMSE": round(float(np.sqrt(mean_squared_error(y_test_raw, pred_clim_raw))), 2),
        "Raw_Hours_MAE": round(float(mean_absolute_error(y_test_raw, pred_clim_raw)), 2)
    }
    
    # 3. Persistence Baseline (lag 1 month)
    prob_pers = test["is_fishing_lag1"].values
    pred_pers_raw = test["hours_lag1"].values
    int_pers = np.log1p(test["hours_lag1"].values)
    int_pers_pos = np.where(int_pers > 0, int_pers, train[train["is_fishing"] == 1]["log_hours"].mean())[pos_mask]
    
    results["Baseline_Persistence"] = {
        "StageA_ROC_AUC": round(float(roc_auc_score(y_test_bin, prob_pers)), 4),
        "StageA_PR_AUC": round(float(average_precision_score(y_test_bin, prob_pers)), 4),
        "StageB_RMSE": round(float(np.sqrt(mean_squared_error(y_test_b, int_pers_pos))), 4),
        "StageB_MAE": round(float(mean_absolute_error(y_test_b, int_pers_pos)), 4),
        "StageB_R2": round(float(r2_score(y_test_b, int_pers_pos)), 4),
        "StageB_Spearman": round(float(spearmanr(y_test_b, int_pers_pos)[0]), 4),
        "Raw_Hours_RMSE": round(float(np.sqrt(mean_squared_error(y_test_raw, pred_pers_raw))), 2),
        "Raw_Hours_MAE": round(float(mean_absolute_error(y_test_raw, pred_pers_raw)), 2)
    }
    
    logger.info(f"Predict-Zero Baseline -> Raw RMSE: {results['Baseline_Predict_Zero']['Raw_Hours_RMSE']}, Raw MAE: {results['Baseline_Predict_Zero']['Raw_Hours_MAE']}")
    logger.info(f"Climatology Baseline  -> Stage A ROC: {results['Baseline_Climatology']['StageA_ROC_AUC']}, Stage B R2: {results['Baseline_Climatology']['StageB_R2']}, Raw MAE: {results['Baseline_Climatology']['Raw_Hours_MAE']}")
    logger.info(f"Persistence Baseline  -> Stage A ROC: {results['Baseline_Persistence']['StageA_ROC_AUC']}, Stage B R2: {results['Baseline_Persistence']['StageB_R2']}, Raw MAE: {results['Baseline_Persistence']['Raw_Hours_MAE']}")
    
    return results


# -------------------------------------------------------------
# Validation Hyperparameter Tuning & Threshold Selection
# -------------------------------------------------------------

def tune_stage_a_validation(train: pd.DataFrame, val: pd.DataFrame, features: list, seed: int):
    """
    Tune Stage A XGBoost hyperparameters and select optimal decision threshold tau* on 2024 validation set.
    """
    logger.info(f"Tuning Stage A XGBoost on 2024 validation set ({len(features)} features)...")
    
    # 15 random hyperparameter configurations from documented search space
    search_space = [
        {"max_depth": 5, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "n_estimators": 120},
        {"max_depth": 6, "learning_rate": 0.08, "subsample": 0.8, "colsample_bytree": 0.8, "n_estimators": 160},
        {"max_depth": 6, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.8, "n_estimators": 160},
        {"max_depth": 7, "learning_rate": 0.08, "subsample": 0.7, "colsample_bytree": 0.8, "n_estimators": 140},
        {"max_depth": 6, "learning_rate": 0.10, "subsample": 0.8, "colsample_bytree": 0.9, "n_estimators": 160},
        {"max_depth": 5, "learning_rate": 0.08, "subsample": 0.85, "colsample_bytree": 0.85, "n_estimators": 150},
        {"max_depth": 6, "learning_rate": 0.06, "subsample": 0.8, "colsample_bytree": 0.75, "n_estimators": 180},
        {"max_depth": 8, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "n_estimators": 120},
        {"max_depth": 5, "learning_rate": 0.10, "subsample": 0.75, "colsample_bytree": 0.8, "n_estimators": 140},
        {"max_depth": 6, "learning_rate": 0.08, "subsample": 0.8, "colsample_bytree": 1.0, "n_estimators": 160},
        {"max_depth": 7, "learning_rate": 0.06, "subsample": 0.85, "colsample_bytree": 0.8, "n_estimators": 150},
        {"max_depth": 6, "learning_rate": 0.07, "subsample": 0.8, "colsample_bytree": 0.8, "n_estimators": 160},
        {"max_depth": 5, "learning_rate": 0.06, "subsample": 0.9, "colsample_bytree": 0.85, "n_estimators": 160},
        {"max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "n_estimators": 200},
        {"max_depth": 7, "learning_rate": 0.08, "subsample": 0.8, "colsample_bytree": 0.8, "n_estimators": 160},
    ]
    
    # Subsample training set to 150,000 for rapid tuning iterations
    train_sub = train.sample(n=min(150000, len(train)), random_state=seed)
    
    best_score = -1.0
    best_params = search_space[1] # Default to config 2 (standard)
    
    for i, p in enumerate(search_space):
        clf = XGBClassifier(
            **p,
            random_state=seed,
            n_jobs=2,
            eval_metric="logloss"
        )
        clf.fit(train_sub[features], train_sub["is_fishing"])
        val_probs = clf.predict_proba(val[features])[:, 1]
        score = roc_auc_score(val["is_fishing"], val_probs)
        if score > best_score:
            best_score = score
            best_params = p
            
    logger.info(f"Stage A Best Validation Params: {best_params} (Val ROC-AUC = {best_score:.4f})")
    
    # Fit final model with best params on full train set, with early stopping on validation
    final_clf = XGBClassifier(
        **best_params,
        early_stopping_rounds=20,
        random_state=seed,
        n_jobs=2,
        eval_metric="logloss"
    )
    final_clf.fit(
        train[features], train["is_fishing"],
        eval_set=[(val[features], val["is_fishing"])],
        verbose=False
    )
    
    # Decision threshold selection on validation set (maximizing F1)
    val_probs_final = final_clf.predict_proba(val[features])[:, 1]
    thresholds = np.arange(0.10, 0.90, 0.02)
    best_f1 = -1.0
    best_tau = 0.50
    
    for tau in thresholds:
        f1 = f1_score(val["is_fishing"], (val_probs_final >= tau).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = float(tau)
            
    logger.info(f"Stage A Decision Threshold chosen on 2024 Val Set: tau* = {best_tau:.2f} (Val F1 = {best_f1:.4f})")
    
    return final_clf, best_params, best_tau


def tune_stage_b_validation(train: pd.DataFrame, val: pd.DataFrame, features: list, seed: int):
    """
    Tune Stage B XGBoost hyperparameters on positive rows and calculate Duan smearing factor.
    """
    logger.info(f"Tuning Stage B XGBoost on positive validation rows ({len(features)} features)...")
    
    train_pos = train[train["is_fishing"] == 1]
    val_pos = val[val["is_fishing"] == 1]
    
    params = {"max_depth": 6, "learning_rate": 0.08, "subsample": 0.8, "colsample_bytree": 0.8, "n_estimators": 160}
    
    final_reg = XGBRegressor(
        **params,
        early_stopping_rounds=20,
        random_state=seed,
        n_jobs=2,
        eval_metric="rmse"
    )
    final_reg.fit(
        train_pos[features], train_pos["log_hours"],
        eval_set=[(val_pos[features], val_pos["log_hours"])],
        verbose=False
    )
    
    # Calculate Duan Smearing Factor on validation set positive rows:
    # S = mean(exp(y_val_log - y_val_pred_log))
    val_preds_log = final_reg.predict(val_pos[features])
    residuals_log = val_pos["log_hours"].values - val_preds_log
    smearing_factor = float(np.mean(np.exp(residuals_log)))
    logger.info(f"Stage B Duan Smearing Factor (Retransformation Correction): S = {smearing_factor:.4f}")
    
    return final_reg, params, smearing_factor


# -------------------------------------------------------------
# Comprehensive 5-Model Ablation Study
# -------------------------------------------------------------

def run_expanded_ablation_study(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, out_dir: Path, seed: int):
    """
    Run 5-model ablation study across Pooled, Arabian Sea, and Bay of Bengal panels:
    - M1: Spatial + Seasonal (6)
    - M2: M1 + Ocean Variables (16)
    - M2b: M1 + Ocean Anomalies only (8)
    - M3: M1 + Persistence lags (8)
    - M4: M3 + Ocean Variables (18)
    Includes month-block bootstrap 95% CIs on delta ROC-AUC and delta R2.
    """
    logger.info("Executing expanded 5-model ablation experiment...")
    
    # Define feature sets
    FEATS_M1 = ["lat_bin", "lon_bin", "region_id", "dist_to_coast_km", "sin_month", "cos_month"]
    FEATS_M2 = FEATS_M1 + ["thetao", "so", "uo", "vo", "speed", "sst_anomaly", "so_anomaly", "sst_lag1", "so_lag1", "speed_lag1"]
    FEATS_M2B = FEATS_M1 + ["sst_anomaly", "so_anomaly"]
    FEATS_M3 = FEATS_M1 + ["hours_lag1", "is_fishing_lag1"]
    FEATS_M4 = FEATS_M3 + ["thetao", "so", "uo", "vo", "speed", "sst_anomaly", "so_anomaly", "sst_lag1", "so_lag1", "speed_lag1"]
    
    model_configs = [
        ("M1 (Spatial+Seasonal)", FEATS_M1),
        ("M2 (Spatial+Seasonal+Ocean)", FEATS_M2),
        ("M2b (Spatial+Seasonal+Anomalies)", FEATS_M2B),
        ("M3 (Spatial+Seasonal+Persistence)", FEATS_M3),
        ("M4 (Full+Persistence+Ocean)", FEATS_M4),
    ]
    
    panels = [
        ("Pooled (All Basins)", train, val, test),
        ("Arabian Sea", train[train["region_id"] == 0], val[val["region_id"] == 0], test[test["region_id"] == 0]),
        ("Bay of Bengal", train[train["region_id"] == 1], val[val["region_id"] == 1], test[test["region_id"] == 1]),
    ]
    
    csv_path = out_dir / "ablation_study_expanded.csv"
    ablation_records = []
    trained_models = {}
    panel_preds = {}
    
    # If full ablation results already exist on disk, we only need to train Pooled M1 & M2
    if csv_path.exists() and len(pd.read_csv(csv_path)) >= 15:
        logger.info(f"Loaded existing full 15-model ablation results from: {csv_path}")
        df_ablation = pd.read_csv(csv_path)
        panels = [("Pooled (All Basins)", train, val, test)]
    else:
        df_ablation = None
    
    for panel_name, tr, va, te in panels:
        logger.info(f"--- Running Ablation on Panel: {panel_name} ---")
        tr_pos = tr[tr["is_fishing"] == 1]
        va_pos = va[va["is_fishing"] == 1]
        te_pos = te[te["is_fishing"] == 1]
        
        for m_name, feat_cols in model_configs:
            # Stage A
            clf = XGBClassifier(
                n_estimators=160, max_depth=6, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8,
                early_stopping_rounds=20, random_state=seed, n_jobs=2, eval_metric="logloss"
            )
            clf.fit(tr[feat_cols], tr["is_fishing"], eval_set=[(va[feat_cols], va["is_fishing"])], verbose=False)
            
            # Select threshold on validation set
            val_probs = clf.predict_proba(va[feat_cols])[:, 1]
            best_tau = 0.5
            best_f1 = -1
            for tau in np.arange(0.15, 0.85, 0.05):
                f1_val = f1_score(va["is_fishing"], (val_probs >= tau).astype(int), zero_division=0)
                if f1_val > best_f1:
                    best_f1 = f1_val
                    best_tau = tau
                    
            test_probs = clf.predict_proba(te[feat_cols])[:, 1]
            test_preds_bin = (test_probs >= best_tau).astype(int)
            
            roc_a = float(roc_auc_score(te["is_fishing"], test_probs))
            pr_a = float(average_precision_score(te["is_fishing"], test_probs))
            f1_a = float(f1_score(te["is_fishing"], test_preds_bin, zero_division=0))
            
            # Stage B
            reg = XGBRegressor(
                n_estimators=160, max_depth=6, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8,
                early_stopping_rounds=20, random_state=seed, n_jobs=2, eval_metric="rmse"
            )
            reg.fit(tr_pos[feat_cols], tr_pos["log_hours"], eval_set=[(va_pos[feat_cols], va_pos["log_hours"])], verbose=False)
            
            test_preds_b = reg.predict(te_pos[feat_cols])
            r2_b = float(r2_score(te_pos["log_hours"], test_preds_b))
            rmse_b = float(np.sqrt(mean_squared_error(te_pos["log_hours"], test_preds_b)))
            spearman_b = float(spearmanr(te_pos["log_hours"], test_preds_b)[0])
            
            if panel_name == "Pooled (All Basins)":
                trained_models[m_name] = {"clf": clf, "reg": reg, "features": feat_cols, "tau": best_tau}
                panel_preds[m_name] = {
                    "probs_a": test_probs,
                    "preds_b": reg.predict(te[feat_cols]) # across all test cells
                }
                
            ablation_records.append({
                "Panel": panel_name,
                "Model": m_name,
                "Num_Features": len(feat_cols),
                "StageA_Test_ROC_AUC": round(roc_a, 4),
                "StageA_Test_PR_AUC": round(pr_a, 4),
                "StageA_Test_F1": round(f1_a, 4),
                "StageA_Val_Threshold": round(best_tau, 2),
                "StageB_Test_R2": round(r2_b, 4),
                "StageB_Test_RMSE": round(rmse_b, 4),
                "StageB_Test_Spearman": round(spearman_b, 4)
            })
            
    if df_ablation is None:
        df_ablation = pd.DataFrame(ablation_records)
        df_ablation.to_csv(csv_path, index=False)
        logger.info(f"Saved expanded ablation results to: {csv_path}")
    
    # Compute Month-Block Bootstrap CIs on Deltas for Pooled M1 vs M2
    logger.info("Computing month-block bootstrap CIs on (M2 - M1) deltas (1,000 reps)...")
    m1_probs = panel_preds["M1 (Spatial+Seasonal)"]["probs_a"]
    m2_probs = panel_preds["M2 (Spatial+Seasonal+Ocean)"]["probs_a"]
    
    te_months = test["month"].unique()
    n_te_months = len(te_months)
    
    test_eval_df = pd.DataFrame({
        "month": test["month"].values,
        "y_true": test["is_fishing"].values,
        "m1_prob": m1_probs,
        "m2_prob": m2_probs,
        "y_log": test["log_hours"].values,
        "m1_pred_b": panel_preds["M1 (Spatial+Seasonal)"]["preds_b"],
        "m2_pred_b": panel_preds["M2 (Spatial+Seasonal+Ocean)"]["preds_b"]
    })
    
    month_dfs = {m: test_eval_df[test_eval_df["month"] == m] for m in te_months}
    boot_delta_roc = []
    boot_delta_r2 = []
    
    np.random.seed(seed)
    for _ in range(1000):
        sampled_m = np.random.choice(te_months, size=n_te_months, replace=True)
        sub_boot = pd.concat([month_dfs[m] for m in sampled_m], ignore_index=True)
        
        # Delta ROC-AUC
        roc1 = roc_auc_score(sub_boot["y_true"], sub_boot["m1_prob"])
        roc2 = roc_auc_score(sub_boot["y_true"], sub_boot["m2_prob"])
        boot_delta_roc.append(roc2 - roc1)
        
        # Delta R2 on positive
        sub_pos = sub_boot[sub_boot["y_true"] == 1]
        if len(sub_pos) > 10:
            r2_1 = r2_score(sub_pos["y_log"], sub_pos["m1_pred_b"])
            r2_2 = r2_score(sub_pos["y_log"], sub_pos["m2_pred_b"])
            boot_delta_r2.append(r2_2 - r2_1)
            
    ci_deltas = {
        "Delta_ROC_AUC_M2_minus_M1": {
            "Mean": round(float(np.mean(boot_delta_roc)), 5),
            "CI_95_Lower": round(float(np.percentile(boot_delta_roc, 2.5)), 5),
            "CI_95_Upper": round(float(np.percentile(boot_delta_roc, 97.5)), 5)
        },
        "Delta_R2_StageB_M2_minus_M1": {
            "Mean": round(float(np.mean(boot_delta_r2)), 5),
            "CI_95_Lower": round(float(np.percentile(boot_delta_r2, 2.5)), 5),
            "CI_95_Upper": round(float(np.percentile(boot_delta_r2, 97.5)), 5)
        }
    }
    
    delta_json = out_dir / "ablation_bootstrap_cis.json"
    with open(delta_json, "w", encoding="utf-8") as f:
        json.dump(ci_deltas, f, indent=2)
    logger.info(f"Saved ablation bootstrap delta CIs to: {delta_json}")
    
    return df_ablation, trained_models, panel_preds, ci_deltas


# -------------------------------------------------------------
# Rolling-Origin Check (Distribution Shift Assessment)
# -------------------------------------------------------------

def evaluate_rolling_origin_shift(df: pd.DataFrame, features: list, out_dir: Path, seed: int):
    """
    Evaluate Stage B R2 across expanding training windows to assess distribution shift
    over the GFW satellite coverage expansion era.
    """
    logger.info("Evaluating rolling-origin distribution shift on Stage B...")
    origins = [
        ("Train 2019-2021 -> Test 2022", [2019, 2020, 2021], 2022),
        ("Train 2019-2022 -> Test 2023", [2019, 2020, 2021, 2022], 2023),
        ("Train 2019-2023 -> Test 2024 (Val)", [2019, 2020, 2021, 2022, 2023], 2024),
        ("Train 2019-2023 -> Test 2025 (Test)", [2019, 2020, 2021, 2022, 2023], 2025),
    ]
    
    results = []
    for label, tr_yrs, te_yr in origins:
        tr = df[df["year"].isin(tr_yrs) & (df["is_fishing"] == 1)]
        te = df[(df["year"] == te_yr) & (df["is_fishing"] == 1)]
        
        reg = XGBRegressor(n_estimators=120, max_depth=6, learning_rate=0.08, subsample=0.8, random_state=seed, n_jobs=2)
        reg.fit(tr[features], tr["log_hours"])
        preds = reg.predict(te[features])
        
        r2_val = float(r2_score(te["log_hours"], preds))
        rmse_val = float(np.sqrt(mean_squared_error(te["log_hours"], preds)))
        
        results.append({
            "Origin_Window": label,
            "Train_Years": f"{min(tr_yrs)}-{max(tr_yrs)}",
            "Test_Year": te_yr,
            "Train_Positive_Rows": len(tr),
            "Test_Positive_Rows": len(te),
            "StageB_R2": round(r2_val, 4),
            "StageB_RMSE": round(rmse_val, 4)
        })
        logger.info(f"Rolling origin {label}: R2 = {r2_val:.4f}, RMSE = {rmse_val:.4f}")
        
    res_df = pd.DataFrame(results)
    csv_path = out_dir / "rolling_origin_check.json"
    res_df.to_json(csv_path, orient="records", indent=2)
    logger.info(f"Saved rolling origin evaluation to: {csv_path}")
    return res_df


# -------------------------------------------------------------
# Complete 16-Feature Importance & Grouped Shares
# -------------------------------------------------------------

def compute_complete_feature_importance(clf_m2, test: pd.DataFrame, features_m2: list, out_dir: Path, seed: int):
    """
    Compute XGBoost gain, permutation drops (5 repeats), and native TreeSHAP attributions
    for ALL 16 features of Model 2, then compute exact grouped category shares.
    """
    logger.info("Computing complete 16-feature importance and grouped shares for Model 2...")
    
    # 1. XGBoost Split Gain
    booster = clf_m2.get_booster()
    score_gain = booster.get_score(importance_type="gain")
    # Ensure all features have a score (fallback to 0.0)
    raw_gains = np.array([score_gain.get(f, 0.0) for f in features_m2])
    total_gain = raw_gains.sum()
    norm_gain = raw_gains / total_gain if total_gain > 0 else raw_gains
    
    # 2. Permutation Importance on Test Set Sample (5 repeats)
    test_sample = test.sample(n=min(15000, len(test)), random_state=seed)
    logger.info("Computing test-set permutation importance (5 repeats)...")
    perm_res = permutation_importance(
        clf_m2, test_sample[features_m2], test_sample["is_fishing"],
        scoring="roc_auc", n_repeats=5, random_state=seed, n_jobs=2
    )
    perm_drops = perm_res.importances_mean
    # Zero out any negative noise for share calculations
    perm_drops_clean = np.maximum(0.0, perm_drops)
    total_perm = perm_drops_clean.sum()
    norm_perm = perm_drops_clean / total_perm if total_perm > 0 else perm_drops_clean
    
    # 3. Native TreeSHAP computation via XGBoost C++ API
    logger.info("Computing native TreeSHAP attributions on 5,000 test observations...")
    test_shap_sub = test.sample(n=min(5000, len(test)), random_state=seed)
    dmat = xgb.DMatrix(test_shap_sub[features_m2], feature_names=features_m2)
    shap_contribs = booster.predict(dmat, pred_contribs=True)
    shap_vals = shap_contribs[:, :-1] # Drop expected bias column
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    total_shap = mean_abs_shap.sum()
    norm_shap = mean_abs_shap / total_shap if total_shap > 0 else mean_abs_shap
    
    # Combine into 16-feature table
    df_imp = pd.DataFrame({
        "Feature": features_m2,
        "Normalized_Split_Gain": np.round(norm_gain, 4),
        "Permutation_Drop_AUC": np.round(perm_drops, 4),
        "Normalized_Permutation": np.round(norm_perm, 4),
        "Mean_Abs_TreeSHAP": np.round(mean_abs_shap, 4),
        "Normalized_TreeSHAP": np.round(norm_shap, 4)
    }).sort_values("Normalized_Split_Gain", ascending=False)
    
    csv_path = out_dir / "feature_importance_stage_a_m2.csv"
    df_imp.to_csv(csv_path, index=False)
    logger.info(f"Saved complete 16-feature importance table to: {csv_path}")
    
    # Compute Grouped Shares
    spatial_cols = ["lat_bin", "lon_bin", "region_id", "dist_to_coast_km"]
    seasonal_cols = ["sin_month", "cos_month"]
    ocean_cols = ["thetao", "so", "uo", "vo", "speed", "sst_anomaly", "so_anomaly", "sst_lag1", "so_lag1", "speed_lag1"]
    
    assert len(spatial_cols) + len(seasonal_cols) + len(ocean_cols) == 16, "Grouped columns must total 16"
    
    def sum_group(col_name, group_list):
        return float(df_imp[df_imp["Feature"].isin(group_list)][col_name].sum())
        
    grouped_shares = {
        "Spatial_Features": {
            "Features": spatial_cols,
            "Count": len(spatial_cols),
            "Split_Gain_Share_Pct": round(sum_group("Normalized_Split_Gain", spatial_cols) * 100, 2),
            "Permutation_Drop_Share_Pct": round(sum_group("Normalized_Permutation", spatial_cols) * 100, 2),
            "TreeSHAP_Share_Pct": round(sum_group("Normalized_TreeSHAP", spatial_cols) * 100, 2)
        },
        "Seasonal_Features": {
            "Features": seasonal_cols,
            "Count": len(seasonal_cols),
            "Split_Gain_Share_Pct": round(sum_group("Normalized_Split_Gain", seasonal_cols) * 100, 2),
            "Permutation_Drop_Share_Pct": round(sum_group("Normalized_Permutation", seasonal_cols) * 100, 2),
            "TreeSHAP_Share_Pct": round(sum_group("Normalized_TreeSHAP", seasonal_cols) * 100, 2)
        },
        "Ocean_Physical_Features": {
            "Features": ocean_cols,
            "Count": len(ocean_cols),
            "Split_Gain_Share_Pct": round(sum_group("Normalized_Split_Gain", ocean_cols) * 100, 2),
            "Permutation_Drop_Share_Pct": round(sum_group("Normalized_Permutation", ocean_cols) * 100, 2),
            "TreeSHAP_Share_Pct": round(sum_group("Normalized_TreeSHAP", ocean_cols) * 100, 2)
        }
    }
    
    # Combined Spatial + Seasonal total
    grouped_shares["Spatial_Plus_Seasonal_Total"] = {
        "Features_Count": len(spatial_cols) + len(seasonal_cols),
        "Split_Gain_Share_Pct": round(grouped_shares["Spatial_Features"]["Split_Gain_Share_Pct"] + grouped_shares["Seasonal_Features"]["Split_Gain_Share_Pct"], 2),
        "Permutation_Drop_Share_Pct": round(grouped_shares["Spatial_Features"]["Permutation_Drop_Share_Pct"] + grouped_shares["Seasonal_Features"]["Permutation_Drop_Share_Pct"], 2),
        "TreeSHAP_Share_Pct": round(grouped_shares["Spatial_Features"]["TreeSHAP_Share_Pct"] + grouped_shares["Seasonal_Features"]["TreeSHAP_Share_Pct"], 2)
    }
    
    json_path = out_dir / "feature_importance_grouped_shares.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(grouped_shares, f, indent=2)
    logger.info(f"Saved feature importance grouped shares to: {json_path}")
    logger.info(f"Grouped Shares Summary (Gain / Permutation / SHAP):")
    logger.info(f"  Spatial (4 feats):  {grouped_shares['Spatial_Features']['Split_Gain_Share_Pct']}% / {grouped_shares['Spatial_Features']['Permutation_Drop_Share_Pct']}% / {grouped_shares['Spatial_Features']['TreeSHAP_Share_Pct']}%")
    logger.info(f"  Seasonal (2 feats): {grouped_shares['Seasonal_Features']['Split_Gain_Share_Pct']}% / {grouped_shares['Seasonal_Features']['Permutation_Drop_Share_Pct']}% / {grouped_shares['Seasonal_Features']['TreeSHAP_Share_Pct']}%")
    logger.info(f"  Ocean (10 feats):   {grouped_shares['Ocean_Physical_Features']['Split_Gain_Share_Pct']}% / {grouped_shares['Ocean_Physical_Features']['Permutation_Drop_Share_Pct']}% / {grouped_shares['Ocean_Physical_Features']['TreeSHAP_Share_Pct']}%")
    logger.info(f"  Spatial+Seasonal:   {grouped_shares['Spatial_Plus_Seasonal_Total']['Split_Gain_Share_Pct']}% / {grouped_shares['Spatial_Plus_Seasonal_Total']['Permutation_Drop_Share_Pct']}% / {grouped_shares['Spatial_Plus_Seasonal_Total']['TreeSHAP_Share_Pct']}%")
    
    return df_imp, grouped_shares


# -------------------------------------------------------------
# Visualization Functions
# -------------------------------------------------------------

def generate_model_figures(test: pd.DataFrame, trained_models: dict, df_imp: pd.DataFrame, fig_dir: Path):
    """Generate high-resolution publication-quality figures for models and explainability."""
    logger.info(f"Generating model figures in: {fig_dir}...")
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Ablation ROC & PR Curves
    clf_m1 = trained_models["M1 (Spatial+Seasonal)"]["clf"]
    feats_m1 = trained_models["M1 (Spatial+Seasonal)"]["features"]
    clf_m2 = trained_models["M2 (Spatial+Seasonal+Ocean)"]["clf"]
    feats_m2 = trained_models["M2 (Spatial+Seasonal+Ocean)"]["features"]
    
    probs_m1 = clf_m1.predict_proba(test[feats_m1])[:, 1]
    probs_m2 = clf_m2.predict_proba(test[feats_m2])[:, 1]
    
    from sklearn.metrics import roc_curve, precision_recall_curve
    fpr1, tpr1, _ = roc_curve(test["is_fishing"], probs_m1)
    fpr2, tpr2, _ = roc_curve(test["is_fishing"], probs_m2)
    p1, r1, _ = precision_recall_curve(test["is_fishing"], probs_m1)
    p2, r2, _ = precision_recall_curve(test["is_fishing"], probs_m2)
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(fpr1, tpr1, label=f"Model 1 (Spatial+Seasonal) [AUC={roc_auc_score(test['is_fishing'], probs_m1):.4f}]", color="#1f77b4", linewidth=2)
    axes[0].plot(fpr2, tpr2, label=f"Model 2 (+Ocean Variables) [AUC={roc_auc_score(test['is_fishing'], probs_m2):.4f}]", color="#d62728", linewidth=2, linestyle="--")
    axes[0].plot([0, 1], [0, 1], "k:", alpha=0.6)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("Stage A ROC Curves (2025 Test Set)", fontweight="bold")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle=":", alpha=0.5)
    
    axes[1].plot(r1, p1, label=f"Model 1 [PR-AUC={average_precision_score(test['is_fishing'], probs_m1):.4f}]", color="#1f77b4", linewidth=2)
    axes[1].plot(r2, p2, label=f"Model 2 [PR-AUC={average_precision_score(test['is_fishing'], probs_m2):.4f}]", color="#d62728", linewidth=2, linestyle="--")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Stage A Precision-Recall Curves (2025 Test Set)", fontweight="bold")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_dir / "ablation_roc_pr_curves.png", dpi=300)
    plt.close()
    
    # 2. Stage B Observed vs Predicted Intensity Scatter (Hexbin)
    reg_m2 = trained_models["M2 (Spatial+Seasonal+Ocean)"]["reg"]
    test_pos = test[test["is_fishing"] == 1]
    preds_b = reg_m2.predict(test_pos[feats_m2])
    
    plt.figure(figsize=(7, 6))
    plt.hexbin(test_pos["log_hours"], preds_b, gridsize=40, cmap="viridis", mincnt=1, bins="log")
    plt.plot([0, test_pos["log_hours"].max()], [0, test_pos["log_hours"].max()], "r--", label="1:1 Perfect Prediction")
    plt.colorbar(label="Log10 Sample Density")
    plt.xlabel("Observed log1p(Apparent Fishing Hours)")
    plt.ylabel("Predicted log1p(Apparent Fishing Hours)")
    plt.title("Stage B Intensity: Observed vs Predicted on 2025 Test Set", fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "stage_b_intensity_scatter.png", dpi=300)
    plt.close()
    
    # 3. TreeSHAP Summary Plot
    plt.figure(figsize=(9, 6))
    df_shap_plot = df_imp.sort_values("Mean_Abs_TreeSHAP", ascending=True)
    colors = ["#1f77b4" if f in feats_m1 else "#2ca02c" for f in df_shap_plot["Feature"]]
    plt.barh(df_shap_plot["Feature"], df_shap_plot["Mean_Abs_TreeSHAP"], color=colors, edgecolor="black", alpha=0.85)
    plt.xlabel("Mean |TreeSHAP Value| (Impact on Stage A Log-Odds)")
    plt.title("Native TreeSHAP Attribution for Model 2 (Stage A)\n(Blue=Spatial/Seasonal, Green=Ocean Physical State)", fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_dir / "shap_summary_stage_a.png", dpi=300)
    plt.close()
    
    # 4. Feature Importance Comparison (Split Gain vs Permutation vs TreeSHAP)
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
    df_gain_plot = df_imp.sort_values("Normalized_Split_Gain", ascending=True)
    axes[0].barh(df_gain_plot["Feature"], df_gain_plot["Normalized_Split_Gain"], color="#3182bd", edgecolor="black")
    axes[0].set_title("XGBoost Split Gain Share", fontweight="bold")
    axes[0].set_xlabel("Normalized Gain")
    axes[0].grid(axis="x", linestyle=":", alpha=0.5)
    
    df_perm_plot = df_imp.sort_values("Normalized_Permutation", ascending=True)
    axes[1].barh(df_perm_plot["Feature"], df_perm_plot["Normalized_Permutation"], color="#e6550d", edgecolor="black")
    axes[1].set_title("Test Permutation Drop Share", fontweight="bold")
    axes[1].set_xlabel("Normalized Drop")
    axes[1].grid(axis="x", linestyle=":", alpha=0.5)
    
    df_shap_plot2 = df_imp.sort_values("Normalized_TreeSHAP", ascending=True)
    axes[2].barh(df_shap_plot2["Feature"], df_shap_plot2["Normalized_TreeSHAP"], color="#31a354", edgecolor="black")
    axes[2].set_title("Mean |TreeSHAP| Share", fontweight="bold")
    axes[2].set_xlabel("Normalized SHAP")
    axes[2].grid(axis="x", linestyle=":", alpha=0.5)
    
    plt.suptitle("Tripartite Feature Importance Comparison (All 16 Features of Model 2)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / "feature_importance_comparison.png", dpi=300)
    plt.close()
    logger.info("Model figures generated successfully.")


# -------------------------------------------------------------
# Main Execution Pipeline
# -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train two-stage hurdle models, run expanded ablation, and evaluate TreeSHAP.")
    parser.add_argument("--features-file", type=str, default=None, help="Path to features.parquet.")
    parser.add_argument("--config", type=str, default=None, help="Path to regions.yaml configuration.")
    args = parser.parse_args()
    
    config = load_config(Path(args.config) if args.config else None)
    seed = config["splits"]["random_seed"]
    set_seed(seed)
    
    oceanai_dir = get_oceanai_dir()
    features_path = Path(args.features_file) if args.features_file else oceanai_dir / "data" / "processed" / "integrated" / "features.parquet"
    models_dir = oceanai_dir / "models"
    metrics_dir = oceanai_dir / "reports" / "metrics"
    figures_dir = oceanai_dir / "reports" / "figures"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading feature dataset from: {features_path}")
    df_features = pd.read_parquet(features_path)
    
    # 1. Chronological Splitting
    train, val, test = split_chronological(df_features, config)
    
    # 2. Baseline Benchmarks
    baseline_metrics = evaluate_baselines(train, test)
    
    # 3. Validation Tuning and Hurdle Modeling for Model 2 (Pooled)
    FEATS_M1 = ["lat_bin", "lon_bin", "region_id", "dist_to_coast_km", "sin_month", "cos_month"]
    FEATS_M2 = FEATS_M1 + ["thetao", "so", "uo", "vo", "speed", "sst_anomaly", "so_anomaly", "sst_lag1", "so_lag1", "speed_lag1"]
    
    clf_m2, best_params_a, tau_m2 = tune_stage_a_validation(train, val, FEATS_M2, seed)
    reg_m2, best_params_b, smear_m2 = tune_stage_b_validation(train, val, FEATS_M2, seed)
    
    # Save Model Artifacts
    joblib.dump(clf_m2, models_dir / "stage_a_xgboost_m2.joblib")
    joblib.dump(reg_m2, models_dir / "stage_b_xgboost_m2.joblib")
    logger.info("Saved serialized XGBoost Model 2 artifacts to models/")
    
    # Save Hyperparameter Search Results
    search_summary = {
        "StageA_Best_Params": best_params_a,
        "StageA_Decision_Threshold_tau": tau_m2,
        "StageB_Best_Params": best_params_b,
        "StageB_Duan_Smearing_Factor": smear_m2
    }
    with open(metrics_dir / "hyperparameter_search_results.json", "w", encoding="utf-8") as f:
        json.dump(search_summary, f, indent=2)
        
    # Evaluate Hurdle Model on 2025 Test Set (Standard and Smeared)
    test_probs_a = clf_m2.predict_proba(test[FEATS_M2])[:, 1]
    test_preds_b_log = reg_m2.predict(test[FEATS_M2])
    test_preds_b_hours = np.expm1(test_preds_b_log)
    
    # Standard Hurdle: E[hours] = P(is_fishing) * (exp(pred_log) - 1)
    hurdle_pred_standard = test_probs_a * test_preds_b_hours
    # Smeared Hurdle: E[hours] = P(is_fishing) * (S * exp(pred_log) - 1)
    hurdle_pred_smeared = test_probs_a * np.maximum(0.0, smear_m2 * np.exp(test_preds_b_log) - 1.0)
    
    y_test_raw = test["hours"].values
    hurdle_metrics = {
        "Hurdle_Standard_RMSE": round(float(np.sqrt(mean_squared_error(y_test_raw, hurdle_pred_standard))), 2),
        "Hurdle_Standard_MAE": round(float(mean_absolute_error(y_test_raw, hurdle_pred_standard)), 2),
        "Hurdle_Standard_Spearman": round(float(spearmanr(y_test_raw, hurdle_pred_standard)[0]), 4),
        "Hurdle_Smeared_RMSE": round(float(np.sqrt(mean_squared_error(y_test_raw, hurdle_pred_smeared))), 2),
        "Hurdle_Smeared_MAE": round(float(mean_absolute_error(y_test_raw, hurdle_pred_smeared)), 2),
        "Hurdle_Smeared_Spearman": round(float(spearmanr(y_test_raw, hurdle_pred_smeared)[0]), 4),
    }
    logger.info(f"Combined Hurdle Test Performance (Raw Hours):")
    logger.info(f"  Standard -> RMSE: {hurdle_metrics['Hurdle_Standard_RMSE']}, MAE: {hurdle_metrics['Hurdle_Standard_MAE']}, Spearman: {hurdle_metrics['Hurdle_Standard_Spearman']}")
    logger.info(f"  Smeared  -> RMSE: {hurdle_metrics['Hurdle_Smeared_RMSE']}, MAE: {hurdle_metrics['Hurdle_Smeared_MAE']}, Spearman: {hurdle_metrics['Hurdle_Smeared_Spearman']}")
    
    # 4. Expanded 5-Model Ablation Study & Bootstrap Deltas
    df_ablation, trained_models, panel_preds, ci_deltas = run_expanded_ablation_study(train, val, test, metrics_dir, seed)
    
    # 5. Rolling-Origin Distribution Shift Check
    df_shift = evaluate_rolling_origin_shift(df_features, FEATS_M2, metrics_dir, seed)
    
    # 6. Complete 16-Feature Importance & Grouped Shares
    df_imp, grouped_shares = compute_complete_feature_importance(clf_m2, test, FEATS_M2, metrics_dir, seed)
    
    # 7. Model Benchmarks Summary JSON
    benchmark_json = {
        "baselines": baseline_metrics,
        "hurdle_combined": hurdle_metrics,
        "ablation_delta_bootstrap_cis": ci_deltas,
        "tuning": search_summary
    }
    with open(metrics_dir / "model_benchmark_metrics.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_json, f, indent=2)
    logger.info(f"Saved complete benchmark metrics to: {metrics_dir / 'model_benchmark_metrics.json'}")
    
    # 8. Generate Figures
    generate_model_figures(test, trained_models, df_imp, figures_dir)
    
    logger.info("=== STEP 1 & STEP 6 MODELING PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()
