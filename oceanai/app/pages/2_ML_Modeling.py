"""
Page 2: Two-Stage Machine Learning Models & Evaluation
======================================================
Presents the hurdle modeling architecture, test performance, and baseline comparisons.
"""

import json
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ML Modeling - Ocean AI", page_icon="🤖", layout="wide")

OCEANAI_DIR = Path(__file__).resolve().parent.parent.parent
METRICS_DIR = OCEANAI_DIR / "reports" / "metrics"
FIGURES_DIR = OCEANAI_DIR / "reports" / "figures"


@st.cache_data
def load_metrics():
    metrics_path = METRICS_DIR / "model_benchmark_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


st.title("🤖 Two-Stage Machine Learning Architecture")
st.markdown("""
Apparent fishing hours exhibit severe **zero-inflation (78.94% zero)** and extreme **right-skewness** (maximum hours reach >7,000 hrs/month).
A standard single-stage regression model conflates absence with low-intensity effort. We implement a **Two-Stage Hurdle Model** evaluated on an out-of-time chronological test split (**2025**):
1. **Stage A (Occurrence):** Binary classifier predicting whether commercial fishing is present ($\text{hours} > 0$).
2. **Stage B (Intensity):** Continuous regressor predicting $\log(1 + \text{hours})$ in cells where fishing was detected.
""")

metrics = load_metrics()
if metrics is None:
    st.error("Model metrics not found. Please train models first.")
    st.stop()

# 1. Comparative Tables
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Stage A: Occurrence (Test Set: 2025)")
    rows_a = []
    # Baselines
    rows_a.append({"Model": "Baseline: Historical Climatology", "Type": "Baseline", "ROC-AUC": metrics["baselines"]["Baseline_Climatology"]["StageA_ROC_AUC"], "PR-AUC": metrics["baselines"]["Baseline_Climatology"]["StageA_PR_AUC"], "F1": metrics["baselines"]["Baseline_Climatology"]["StageA_F1"], "Accuracy": metrics["baselines"]["Baseline_Climatology"]["StageA_Accuracy"]})
    rows_a.append({"Model": "Baseline: Persistence (Lag 1)", "Type": "Baseline", "ROC-AUC": metrics["baselines"]["Baseline_Persistence"]["StageA_ROC_AUC"], "PR-AUC": metrics["baselines"]["Baseline_Persistence"]["StageA_PR_AUC"], "F1": metrics["baselines"]["Baseline_Persistence"]["StageA_F1"], "Accuracy": metrics["baselines"]["Baseline_Persistence"]["StageA_Accuracy"]})
    
    for m_key, m_val in metrics["stage_a"].items():
        m_type = "Model 1 (Spatial+Season)" if "M1" in m_key else "Model 2 (+Ocean)"
        clean_name = m_key.replace("_M1", " [M1]").replace("_M2", " [M2]").replace("_", " ")
        rows_a.append({"Model": clean_name, "Type": m_type, "ROC-AUC": m_val["ROC_AUC"], "PR-AUC": m_val["PR_AUC"], "F1": m_val["F1"], "Accuracy": m_val["Accuracy"]})
        
    df_a = pd.DataFrame(rows_a)
    st.dataframe(df_a.style.format({"ROC-AUC": "{:.4f}", "PR-AUC": "{:.4f}", "F1": "{:.4f}", "Accuracy": "{:.4f}"}))

with col_b:
    st.subheader("Stage B: Intensity (log1p hours, given hours > 0)")
    rows_b = []
    rows_b.append({"Model": "Baseline: Historical Climatology", "Type": "Baseline", "RMSE": metrics["baselines"]["Baseline_Climatology"]["StageB_RMSE"], "MAE": metrics["baselines"]["Baseline_Climatology"]["StageB_MAE"], "R²": metrics["baselines"]["Baseline_Climatology"]["StageB_R2"], "Spearman": metrics["baselines"]["Baseline_Climatology"]["StageB_Spearman"]})
    rows_b.append({"Model": "Baseline: Persistence (Lag 1)", "Type": "Baseline", "RMSE": metrics["baselines"]["Baseline_Persistence"]["StageB_RMSE"], "MAE": metrics["baselines"]["Baseline_Persistence"]["StageB_MAE"], "R²": metrics["baselines"]["Baseline_Persistence"]["StageB_R2"], "Spearman": metrics["baselines"]["Baseline_Persistence"]["StageB_Spearman"]})
    
    for m_key, m_val in metrics["stage_b"].items():
        m_type = "Model 1 (Spatial+Season)" if "M1" in m_key else "Model 2 (+Ocean)"
        clean_name = m_key.replace("_M1", " [M1]").replace("_M2", " [M2]").replace("_", " ")
        rows_b.append({"Model": clean_name, "Type": m_type, "RMSE": m_val["RMSE"], "MAE": m_val["MAE"], "R²": m_val["R2"], "Spearman": m_val["Spearman"]})
        
    df_b = pd.DataFrame(rows_b)
    st.dataframe(df_b.style.format({"RMSE": "{:.4f}", "MAE": "{:.4f}", "R²": "{:.4f}", "Spearman": "{:.4f}"}))

st.markdown("---")

# 2. Combined Expected Value Performance
st.subheader("Combined Two-Stage Expected Value Evaluation (Full 2025 Test Set)")
comb = metrics["combined"]
c1, c2, c3 = st.columns(3)
c1.metric("Overall Test RMSE", f"{comb['Test_RMSE']:.2f} hrs", help="Root Mean Squared Error on all 2025 test cells including zero-effort cells.")
c2.metric("Overall Test MAE", f"{comb['Test_MAE']:.2f} hrs", help="Mean Absolute Error across all test cells.")
c3.metric("Overall Spearman Rank Correlation", f"{comb['Test_Spearman']:.4f}", help="Monotonic ranking correlation of predicted vs observed fishing hours.")

st.markdown("""
$$\mathbb{E}[\text{Hours}] = \hat{P}(\text{Occurrence}) \times \left(\exp(\hat{y}_{\text{Intensity}}) - 1\right)$$
""")

# Display Stage B Diagnostic Plot
scatter_fig = FIGURES_DIR / "stage_b_intensity_scatter.png"
if scatter_fig.exists():
    st.image(str(scatter_fig), caption="Observed vs Predicted log1p(Fishing Hours) on 2025 Test Set (XGBoost Model 2)", use_container_width=True)
