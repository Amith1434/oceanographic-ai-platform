"""
Two-Stage Hurdle Modeling, Ablation Study, & Native TreeSHAP
============================================================
Implements a scientifically grounded machine learning framework for predicting
apparent commercial fishing activity from physical oceanographic conditions.

Key Components:
1. Two-Stage Hurdle Architecture:
   - Stage A (Occurrence): Binary classification (is_fishing = hours > 0).
   - Stage B (Intensity): Continuous regression on log1p(hours) given hours > 0.
2. Chronological Out-of-Time Evaluation:
   - Train: 2019–2023 (60 months)
   - Val:   2024 (12 months)
   - Test:  2025 (12 months)
   - Prevents spatial and temporal autocorrelation data leakage.
3. Benchmarks to Beat:
   - Climatology (historical cell-month average)
   - Persistence (one-month autoregressive lag)
4. Scientific Ablation Study:
   - Model 1: Location + Seasonality only
   - Model 2: Location + Seasonality + Ocean physical state
5. Explainability:
   - Native XGBoost TreeSHAP computation directly via booster C++ API (pred_contribs=True)
   - Permutation importance on test set.
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
    """Compute and evaluate Climatology and Persistence baselines on test set."""
    logger.info("Computing Climatology & Persistence benchmarks on 2025 Test Set...")
    results = {}
    
    # 1. Historical Climatology Baseline
    occ_clim = train.groupby(["region_id", "lat_bin", "lon_bin", "cal_month"])["is_fishing"].mean().rename("prob_clim")
    test_clim = test.merge(occ_clim, on=["region_id", "lat_bin", "lon_bin", "cal_month"], how="left")
    global_occ = train["is_fishing"].mean()
    prob_clim = test_clim["prob_clim"].fillna(global_occ).values
    pred_occ_clim = (prob_clim >= 0.5).astype(int)
    
    train_pos = train[train["is_fishing"] == 1]
    int_clim = train_pos.groupby(["region_id", "lat_bin", "lon_bin", "cal_month"])["log_hours"].mean().rename("log_hours_clim")
    test_clim = test_clim.merge(int_clim, on=["region_id", "lat_bin", "lon_bin", "cal_month"], how="left")
    global_int = train_pos["log_hours"].mean()
    int_clim_all = test_clim["log_hours_clim"].fillna(global_int).values
    
    test_pos_mask = (test["is_fishing"] == 1).values
    y_test_b = test["log_hours"].values[test_pos_mask]
    y_pred_b_clim = int_clim_all[test_pos_mask]
    sp_clim, _ = spearmanr(y_test_b, y_pred_b_clim)
    
    results["Baseline_Climatology"] = {
        "StageA_ROC_AUC": float(roc_auc_score(test["is_fishing"], prob_clim)),
        "StageA_PR_AUC": float(average_precision_score(test["is_fishing"], prob_clim)),
        "StageA_F1": float(f1_score(test["is_fishing"], pred_occ_clim)),
        "StageA_Accuracy": float(accuracy_score(test["is_fishing"], pred_occ_clim)),
        "StageB_RMSE": float(np.sqrt(mean_squared_error(y_test_b, y_pred_b_clim))),
        "StageB_MAE": float(mean_absolute_error(y_test_b, y_pred_b_clim)),
        "StageB_R2": float(r2_score(y_test_b, y_pred_b_clim)),
        "StageB_Spearman": float(sp_clim),
    }
    
    # 2. Persistence Baseline (One-month autoregressive lag)
    prob_pers = test["is_fishing_lag1"].values
    int_pers = np.log1p(test["hours_lag1"].values)
    int_pers_pos = np.where(int_pers > 0, int_pers, global_int)[test_pos_mask]
    sp_pers, _ = spearmanr(y_test_b, int_pers_pos)
    
    results["Baseline_Persistence"] = {
        "StageA_ROC_AUC": float(roc_auc_score(test["is_fishing"], prob_pers)),
        "StageA_PR_AUC": float(average_precision_score(test["is_fishing"], prob_pers)),
        "StageA_F1": float(f1_score(test["is_fishing"], prob_pers)),
        "StageA_Accuracy": float(accuracy_score(test["is_fishing"], prob_pers)),
        "StageB_RMSE": float(np.sqrt(mean_squared_error(y_test_b, int_pers_pos))),
        "StageB_MAE": float(mean_absolute_error(y_test_b, int_pers_pos)),
        "StageB_R2": float(r2_score(y_test_b, int_pers_pos)),
        "StageB_Spearman": float(sp_pers),
    }
    
    for b_name, b_met in results.items():
        logger.info(f"  {b_name} -> Stage A ROC-AUC: {b_met['StageA_ROC_AUC']:.4f} | Stage B RMSE: {b_met['StageB_RMSE']:.4f}, R²: {b_met['StageB_R2']:.4f}")
        
    return results, prob_clim, prob_pers


# -------------------------------------------------------------
# Stage A: Occurrence Models
# -------------------------------------------------------------

def train_stage_a(train: pd.DataFrame, test: pd.DataFrame, feats_m1: list, feats_m2: list, seed: int):
    """Train and evaluate Stage A Occurrence classifiers."""
    logger.info("=== STAGE A: OCCURRENCE (Fishing Detected? hours > 0) ===")
    
    y_train = train["is_fishing"].values
    y_test = test["is_fishing"].values
    
    experiments = [
        ("Logistic_Regression_M1", Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=300, random_state=seed))]), feats_m1),
        ("Logistic_Regression_M2", Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=300, random_state=seed))]), feats_m2),
        ("Random_Forest_M1", RandomForestClassifier(n_estimators=60, max_depth=12, max_samples=0.25, n_jobs=-1, random_state=seed), feats_m1),
        ("Random_Forest_M2", RandomForestClassifier(n_estimators=60, max_depth=12, max_samples=0.25, n_jobs=-1, random_state=seed), feats_m2),
        ("XGBoost_M1", XGBClassifier(n_estimators=160, max_depth=6, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=seed, eval_metric="logloss"), feats_m1),
        ("XGBoost_M2", XGBClassifier(n_estimators=160, max_depth=6, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=seed, eval_metric="logloss"), feats_m2),
    ]
    
    results = {}
    fitted_models = {}
    test_probs = {}
    
    for name, model, feats in experiments:
        logger.info(f"Training {name} with {len(feats)} features...")
        X_train = train[feats].values
        X_test = test[feats].values
        
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)
        
        roc_auc = float(roc_auc_score(y_test, probs))
        pr_auc = float(average_precision_score(y_test, probs))
        f1 = float(f1_score(y_test, preds))
        acc = float(accuracy_score(y_test, preds))
        prec = float(precision_score(y_test, preds, zero_division=0))
        rec = float(recall_score(y_test, preds, zero_division=0))
        
        results[name] = {
            "ROC_AUC": roc_auc,
            "PR_AUC": pr_auc,
            "F1": f1,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "Features": feats
        }
        fitted_models[name] = model
        test_probs[name] = probs
        
        logger.info(f"  {name} -> ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | F1: {f1:.4f} | Acc: {acc:.4f} | Recall: {rec:.4f}")
        
    return results, fitted_models, test_probs


# -------------------------------------------------------------
# Stage B: Intensity Models
# -------------------------------------------------------------

def train_stage_b(train: pd.DataFrame, test: pd.DataFrame, feats_m1: list, feats_m2: list, seed: int):
    """Train and evaluate Stage B Intensity regressors on positive cells (hours > 0)."""
    logger.info("=== STAGE B: INTENSITY (log1p(hours) given hours > 0) ===")
    
    train_pos = train[train["is_fishing"] == 1].copy()
    test_pos = test[test["is_fishing"] == 1].copy()
    
    y_train = train_pos["log_hours"].values
    y_test = test_pos["log_hours"].values
    
    logger.info(f"Training on {len(train_pos):,} positive rows; Testing on {len(test_pos):,} positive rows.")
    
    experiments = [
        ("Linear_Regression_M1", Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0))]), feats_m1),
        ("Linear_Regression_M2", Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0))]), feats_m2),
        ("Random_Forest_M1", RandomForestRegressor(n_estimators=60, max_depth=12, max_samples=0.25, n_jobs=-1, random_state=seed), feats_m1),
        ("Random_Forest_M2", RandomForestRegressor(n_estimators=60, max_depth=12, max_samples=0.25, n_jobs=-1, random_state=seed), feats_m2),
        ("XGBoost_M1", XGBRegressor(n_estimators=160, max_depth=6, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=seed), feats_m1),
        ("XGBoost_M2", XGBRegressor(n_estimators=160, max_depth=6, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=seed), feats_m2),
    ]
    
    results = {}
    fitted_models = {}
    test_preds = {}
    
    for name, model, feats in experiments:
        logger.info(f"Training {name} with {len(feats)} features...")
        X_train = train_pos[feats].values
        X_test = test_pos[feats].values
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))
        sp, _ = spearmanr(y_test, preds)
        
        results[name] = {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "Spearman": float(sp),
            "Features": feats
        }
        fitted_models[name] = model
        test_preds[name] = preds
        
        logger.info(f"  {name} -> RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f} | Spearman: {sp:.4f}")
        
    return results, fitted_models, test_preds


# -------------------------------------------------------------
# Native XGBoost TreeSHAP & Permutation Importance
# -------------------------------------------------------------

def compute_native_shap_and_importance(xgb_a_m2, xgb_b_m2, test: pd.DataFrame, feats_m2: list, seed: int):
    """Compute exact Shapley values using XGBoost C++ booster and Scikit-Learn permutation importance."""
    logger.info("Computing native TreeSHAP values and Permutation Importance for Model 2...")
    
    np.random.seed(seed)
    sample_size = min(10000, len(test))
    test_sample = test.sample(n=sample_size, random_state=seed).copy()
    X_sample = test_sample[feats_m2].values
    
    # 1. Native XGBoost TreeSHAP on Stage A
    dmat = xgb.DMatrix(X_sample, feature_names=feats_m2)
    shap_matrix = xgb_a_m2.get_booster().predict(dmat, pred_contribs=True)
    # The last column is the bias/base value
    shap_values = shap_matrix[:, :-1]
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    
    # 2. Permutation Importance Drop on Test Set
    perm = permutation_importance(xgb_a_m2, X_sample, test_sample["is_fishing"].values, n_repeats=3, random_state=seed, scoring="roc_auc")
    
    df_imp = pd.DataFrame({
        "Feature": feats_m2,
        "StageA_Gain_Pct": (xgb_a_m2.feature_importances_ / xgb_a_m2.feature_importances_.sum()) * 100,
        "StageB_Gain_Pct": (xgb_b_m2.feature_importances_ / xgb_b_m2.feature_importances_.sum()) * 100,
        "Mean_Abs_SHAP": mean_abs_shap,
        "Permutation_ROC_AUC_Drop": perm.importances_mean
    })
    
    df_imp = df_imp.sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
    logger.info("TreeSHAP and Permutation ranking completed.")
    return df_imp, shap_values, X_sample, feats_m2


# -------------------------------------------------------------
# Publication Figures Generation
# -------------------------------------------------------------

def generate_report_figures(
    test: pd.DataFrame,
    test_preds_b: dict,
    df_imp: pd.DataFrame,
    all_metrics: dict,
    shap_values: np.ndarray,
    X_sample: np.ndarray,
    feats: list,
    fig_dir: Path
):
    """Generate high-resolution (300 DPI) publication-ready diagnostic charts."""
    logger.info(f"Rendering publication figures to: {fig_dir}...")
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Ablation ROC-AUC Bar Chart
    plt.figure(figsize=(10, 5))
    models = ["Climatology", "Persistence", "Logistic M1", "Logistic M2", "RF M1", "RF M2", "XGBoost M1", "XGBoost M2"]
    roc_scores = [
        all_metrics["baselines"]["Baseline_Climatology"]["StageA_ROC_AUC"],
        all_metrics["baselines"]["Baseline_Persistence"]["StageA_ROC_AUC"],
        all_metrics["stage_a"]["Logistic_Regression_M1"]["ROC_AUC"],
        all_metrics["stage_a"]["Logistic_Regression_M2"]["ROC_AUC"],
        all_metrics["stage_a"]["Random_Forest_M1"]["ROC_AUC"],
        all_metrics["stage_a"]["Random_Forest_M2"]["ROC_AUC"],
        all_metrics["stage_a"]["XGBoost_M1"]["ROC_AUC"],
        all_metrics["stage_a"]["XGBoost_M2"]["ROC_AUC"],
    ]
    colors = ["#7f7f7f", "#bcbd22", "#aec7e8", "#1f77b4", "#c5b0d5", "#9467bd", "#ffbb78", "#d62728"]
    bars = plt.bar(np.arange(len(models)), roc_scores, color=colors, edgecolor="black", alpha=0.9)
    plt.xticks(np.arange(len(models)), models, rotation=30, ha="right", fontsize=10)
    plt.ylabel("Test ROC-AUC (2025)", fontsize=11)
    plt.ylim(0.5, 0.95)
    plt.title("Ablation Study: Test Set ROC-AUC Across Baselines and Models\n(Comparing Spatial+Seasonal vs Ocean Physical Predictors)", fontsize=12, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width()/2, h + 0.008, f"{h:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / "ablation_roc_pr_curves.png", dpi=300)
    plt.close()
    
    # 2. Native TreeSHAP Summary Bar Plot
    plt.figure(figsize=(9, 6))
    top_shap = df_imp.sort_values("Mean_Abs_SHAP", ascending=True)
    spatial_cols = ["lat_bin", "lon_bin", "region_id", "dist_to_coast_km", "sin_month", "cos_month"]
    c_list = ["#1f77b4" if f in spatial_cols else "#2ca02c" for f in top_shap["Feature"]]
    plt.barh(top_shap["Feature"], top_shap["Mean_Abs_SHAP"], color=c_list, edgecolor="black", alpha=0.85)
    plt.xlabel("Mean |SHAP Value| (Impact on Log-Odds of Occurrence)", fontsize=11)
    plt.title("Native TreeSHAP Feature Attribution (Stage A XGBoost Model 2)\n(Blue = Spatial/Seasonal, Green = Ocean Physical)", fontsize=12, fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_dir / "shap_summary_stage_a.png", dpi=300)
    plt.close()
    
    # 3. Stage B Predicted vs Observed Intensity
    test_pos = test[test["is_fishing"] == 1].copy()
    y_true_b = test_pos["log_hours"].values
    y_pred_b = test_preds_b["XGBoost_M2"]
    plt.figure(figsize=(7, 6))
    plt.hexbin(y_true_b, y_pred_b, gridsize=40, cmap="viridis", mincnt=1, bins="log")
    plt.plot([0, y_true_b.max()], [0, y_true_b.max()], "r--", linewidth=1.5, label="1:1 Perfect Prediction")
    plt.colorbar(label="Log10 Count")
    plt.xlabel("Observed log1p(Apparent Fishing Hours)", fontsize=11)
    plt.ylabel("Predicted log1p(Apparent Fishing Hours) [XGBoost M2]", fontsize=11)
    plt.title(f"Stage B (Intensity): Observed vs Predicted on Test Set (2025)\nR² = {all_metrics['stage_b']['XGBoost_M2']['R2']:.3f}, Spearman = {all_metrics['stage_b']['XGBoost_M2']['Spearman']:.3f}", fontsize=11, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "stage_b_intensity_scatter.png", dpi=300)
    plt.close()
    
    # 4. Feature Importance Comparison Bar Chart
    plt.figure(figsize=(10, 6))
    ind = np.arange(len(df_imp))
    width = 0.35
    plt.barh(ind - width/2, df_imp["StageA_Gain_Pct"], width, label="Stage A (Occurrence) Gain %", color="#3182bd", edgecolor="black")
    plt.barh(ind + width/2, df_imp["StageB_Gain_Pct"], width, label="Stage B (Intensity) Gain %", color="#fd8d3c", edgecolor="black")
    plt.yticks(ind, df_imp["Feature"])
    plt.xlabel("Relative XGBoost Gain (%)", fontsize=11)
    plt.title("XGBoost Relative Feature Importance Across Stages (Model 2)", fontsize=12, fontweight="bold")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_dir / "feature_importance_comparison.png", dpi=300)
    plt.close()
    
    logger.info("Publication figures rendered successfully.")


def main():
    parser = argparse.ArgumentParser(description="Train two-stage hurdle models, run ablation study, and compute TreeSHAP.")
    parser.add_argument("--input-file", type=str, default=None, help="Input features.parquet path.")
    parser.add_argument("--config", type=str, default=None, help="Path to regions.yaml configuration.")
    args = parser.parse_args()
    
    config = load_config(Path(args.config) if args.config else None)
    seed = config["splits"]["random_seed"]
    set_seed(seed)
    
    oceanai_dir = get_oceanai_dir()
    input_path = Path(args.input_file) if args.input_file else oceanai_dir / "data" / "processed" / "integrated" / "features.parquet"
    models_dir = oceanai_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = oceanai_dir / "reports" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = oceanai_dir / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Missing {input_path}. Run features.py first.")
        
    logger.info(f"Loading feature table from: {input_path}")
    df = pd.read_parquet(input_path)
    
    # Chronological Split
    train, val, test = split_chronological(df, config)
    
    feats_m1 = config["features"]["spatial_seasonal"]
    feats_m2 = config["features"]["spatial_seasonal"] + config["features"]["ocean_physical"]
    
    # 1. Benchmarks
    baselines, prob_clim, prob_pers = evaluate_baselines(train, test)
    
    # 2. Stage A Occurrence
    res_a, models_a, test_probs_a = train_stage_a(train, test, feats_m1, feats_m2, seed)
    
    # 3. Stage B Intensity
    res_b, models_b, test_preds_b = train_stage_b(train, test, feats_m1, feats_m2, seed)
    
    # 4. TreeSHAP & Permutation Importance
    df_imp, shap_values, X_sample, feats_list = compute_native_shap_and_importance(
        models_a["XGBoost_M2"], models_b["XGBoost_M2"], test, feats_m2, seed
    )
    
    # 5. Full End-to-End Test Evaluation
    logger.info("Evaluating full combined two-stage predictions on all 2025 Test Cells...")
    p_occ = test_probs_a["XGBoost_M2"]
    X_test_b_all = test[feats_m2].values
    log_int_all = np.maximum(0, models_b["XGBoost_M2"].predict(X_test_b_all))
    pred_hours_expected = p_occ * np.expm1(log_int_all)
    actual_hours = test["hours"].values
    
    combined_rmse = float(np.sqrt(mean_squared_error(actual_hours, pred_hours_expected)))
    combined_mae = float(mean_absolute_error(actual_hours, pred_hours_expected))
    combined_sp, _ = spearmanr(actual_hours, pred_hours_expected)
    logger.info(f"Combined Two-Stage Test RMSE: {combined_rmse:.2f} hrs | MAE: {combined_mae:.2f} hrs | Spearman: {combined_sp:.4f}")
    
    # 6. Save Metrics & Ablation Deltas
    ablation_summary = {
        "baselines": baselines,
        "stage_a": res_a,
        "stage_b": res_b,
        "combined": {
            "Test_RMSE": combined_rmse,
            "Test_MAE": combined_mae,
            "Test_Spearman": float(combined_sp)
        },
        "ablation_delta": {
            "StageA_ROC_AUC_Delta_XGB": float(res_a["XGBoost_M2"]["ROC_AUC"] - res_a["XGBoost_M1"]["ROC_AUC"]),
            "StageA_PR_AUC_Delta_XGB": float(res_a["XGBoost_M2"]["PR_AUC"] - res_a["XGBoost_M1"]["PR_AUC"]),
            "StageB_R2_Delta_XGB": float(res_b["XGBoost_M2"]["R2"] - res_b["XGBoost_M1"]["R2"]),
            "StageB_RMSE_Delta_XGB": float(res_b["XGBoost_M1"]["RMSE"] - res_b["XGBoost_M2"]["RMSE"]),
        }
    }
    
    metrics_json = metrics_dir / "model_benchmark_metrics.json"
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(ablation_summary, f, indent=2)
    logger.info(f"Saved benchmark metrics to: {metrics_json}")
    
    df_imp.to_csv(metrics_dir / "feature_importance.csv", index=False)
    
    # Save Key Trained Models
    joblib.dump(models_a["XGBoost_M1"], models_dir / "stage_a_xgboost_m1.joblib")
    joblib.dump(models_a["XGBoost_M2"], models_dir / "stage_a_xgboost_m2.joblib")
    joblib.dump(models_b["XGBoost_M1"], models_dir / "stage_b_xgboost_m1.joblib")
    joblib.dump(models_b["XGBoost_M2"], models_dir / "stage_b_xgboost_m2.joblib")
    logger.info(f"Serialized trained models to: {models_dir}")
    
    # Generate Figures
    generate_report_figures(test, test_preds_b, df_imp, ablation_summary, shap_values, X_sample, feats_list, figures_dir)
    
    logger.info("=== ABLATION STUDY SCIENTIFIC CONCLUSION ===")
    logger.info(f"Stage A ROC-AUC: Model 1 (Spatial+Seasonal) = {res_a['XGBoost_M1']['ROC_AUC']:.4f} vs Model 2 (+Ocean) = {res_a['XGBoost_M2']['ROC_AUC']:.4f} (Delta: {ablation_summary['ablation_delta']['StageA_ROC_AUC_Delta_XGB']:+.4f})")
    logger.info(f"Stage B R²:      Model 1 (Spatial+Seasonal) = {res_b['XGBoost_M1']['R2']:.4f} vs Model 2 (+Ocean) = {res_b['XGBoost_M2']['R2']:.4f} (Delta: {ablation_summary['ablation_delta']['StageB_R2_Delta_XGB']:+.4f})")


if __name__ == "__main__":
    main()
