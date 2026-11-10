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
def load_all_model_metrics():
    metrics_path = METRICS_DIR / "model_benchmark_metrics.json"
    ablation_path = METRICS_DIR / "ablation_study_expanded.csv"
    
    metrics = None
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
            
    df_abl = pd.read_csv(ablation_path) if ablation_path.exists() else None
    return metrics, df_abl


st.title("🤖 Two-Stage Machine Learning Architecture")
st.markdown(r"""
Apparent fishing hours exhibit severe **zero-inflation (78.94% zero)** and extreme **right-skewness** (maximum hours reach >7,000 hrs/month).
A standard single-stage regression model conflates absence with low-intensity effort. We implement a **Two-Stage Hurdle Model** evaluated on an out-of-time chronological test split (**2025**):
1. **Stage A (Occurrence):** Binary classifier predicting whether commercial fishing is present ($\text{hours} > 0$).
2. **Stage B (Intensity):** Continuous regressor predicting $\log(1 + \text{hours})$ in cells where fishing was detected.
""")

metrics, df_abl = load_all_model_metrics()
if metrics is None:
    st.error("Model metrics not found. Please train models first.")
    st.stop()

# 1. Comparative Tables
col_a, col_b = st.columns(2)

clim = metrics.get("baselines", {}).get("Baseline_Climatology", {})
pers = metrics.get("baselines", {}).get("Baseline_Persistence", {})

with col_a:
    st.subheader("Stage A: Occurrence (2025 Test Set)")
    rows_a = []
    
    # Baselines
    if clim:
        rows_a.append({
            "Model": "Baseline: Historical Climatology",
            "Features": "--",
            "ROC-AUC": f"{clim.get('StageA_ROC_AUC', 0):.4f}",
            "PR-AUC": f"{clim.get('StageA_PR_AUC', 0):.4f}",
            "F1 Score": "N/A (Prob)",
            "Val τ*": "--"
        })
    if pers:
        rows_a.append({
            "Model": "Baseline: 1-Month Persistence",
            "Features": "--",
            "ROC-AUC": f"{pers.get('StageA_ROC_AUC', 0):.4f}",
            "PR-AUC": f"{pers.get('StageA_PR_AUC', 0):.4f}",
            "F1 Score": "N/A (Prob)",
            "Val τ*": "--"
        })
        
    # Trained Models from Ablation Table
    if df_abl is not None and not df_abl.empty:
        df_pooled = df_abl[df_abl["Panel"] == "Pooled (All Basins)"]
        for _, r in df_pooled.iterrows():
            rows_a.append({
                "Model": r["Model"],
                "Features": f"{int(r['Num_Features'])} feats",
                "ROC-AUC": f"{r['StageA_Test_ROC_AUC']:.4f}",
                "PR-AUC": f"{r['StageA_Test_PR_AUC']:.4f}",
                "F1 Score": f"{r['StageA_Test_F1']:.4f}",
                "Val τ*": f"{r['StageA_Val_Threshold']:.2f}"
            })
            
    df_a = pd.DataFrame(rows_a)
    st.dataframe(df_a, use_container_width=True)

with col_b:
    st.subheader("Stage B: Intensity (log1p hours, given hours > 0)")
    rows_b = []
    
    # Baselines
    if clim:
        rows_b.append({
            "Model": "Baseline: Historical Climatology",
            "Features": "--",
            "RMSE (log)": f"{clim.get('StageB_RMSE', 0):.4f}",
            "R²": f"{clim.get('StageB_R2', 0):.4f}",
            "Spearman ρ": f"{clim.get('StageB_Spearman', 0):.4f}"
        })
    if pers:
        rows_b.append({
            "Model": "Baseline: 1-Month Persistence",
            "Features": "--",
            "RMSE (log)": f"{pers.get('StageB_RMSE', 0):.4f}",
            "R²": f"{pers.get('StageB_R2', 0):.4f}",
            "Spearman ρ": f"{pers.get('StageB_Spearman', 0):.4f}"
        })
        
    # Trained Models from Ablation Table
    if df_abl is not None and not df_abl.empty:
        df_pooled = df_abl[df_abl["Panel"] == "Pooled (All Basins)"]
        for _, r in df_pooled.iterrows():
            rows_b.append({
                "Model": r["Model"],
                "Features": f"{int(r['Num_Features'])} feats",
                "RMSE (log)": f"{r['StageB_Test_RMSE']:.4f}",
                "R²": f"{r['StageB_Test_R2']:.4f}",
                "Spearman ρ": f"{r['StageB_Test_Spearman']:.4f}"
            })
            
    df_b = pd.DataFrame(rows_b)
    st.dataframe(df_b, use_container_width=True)

st.markdown("---")

# 2. Combined Expected Value Performance
st.subheader("Combined Two-Stage Expected Value Evaluation (Full 2025 Test Set)")
st.markdown(r"""
$$\mathbb{E}[\text{Hours}] = \hat{P}(\text{Occurrence}) \times \left(S \cdot \exp(\hat{y}_{\text{Intensity}}) - 1\right)$$
""")

hc = metrics.get("hurdle_combined", {})
p_zero = metrics.get("baselines", {}).get("Baseline_Predict_Zero", {})
tuning = metrics.get("tuning", {})
s_factor = tuning.get("StageB_Duan_Smearing_Factor", tuning.get("StageB_Duan_Smearing_Factor_S", 2.4556))

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Predict-Zero Baseline MAE",
    f"{p_zero.get('Raw_Hours_MAE', 17.25):.2f} hrs",
    help="Predicting 0.0 everywhere achieves 17.25 hrs MAE due to 78.94% zero-inflation."
)
c2.metric(
    "Hurdle Standard (Uncorrected)",
    f"{hc.get('Hurdle_Standard_RMSE', 107.39):.2f} hrs",
    f"MAE: {hc.get('Hurdle_Standard_MAE', 17.49):.2f} hrs",
    help="Uncorrected exponentiation exp(y) - 1."
)
c3.metric(
    "Hurdle Duan Smeared (S=2.46)",
    f"{hc.get('Hurdle_Smeared_RMSE', 101.12):.2f} hrs",
    f"MAE: {hc.get('Hurdle_Smeared_MAE', 19.18):.2f} hrs",
    help="Non-parametric retransformation smearing reduces RMSE by 5.8%."
)
c4.metric(
    "Hurdle Spearman Rank (ρ)",
    f"{hc.get('Hurdle_Smeared_Spearman', 0.4711):.4f}",
    help="Monotonic ranking correlation of predicted vs observed fishing hours."
)

st.markdown("---")

# Display Stage B Diagnostic Plot
scatter_fig = FIGURES_DIR / "stage_b_intensity_scatter.png"
if scatter_fig.exists():
    st.subheader("Stage B Intensity Hexbin Scatter Plot")
    st.image(str(scatter_fig), caption="Observed vs Predicted log1p(Fishing Hours) on 2025 Test Set (XGBoost Model 2)", use_container_width=True)
