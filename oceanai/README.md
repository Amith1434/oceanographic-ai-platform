# Oceanographic AI Platform (`oceanai/`)

A cross-domain machine learning and intelligence platform for the northern Indian Ocean, bridging physical oceanography (**Copernicus Marine Service**), commercial fisheries dynamics (**Global Fishing Watch**), and marine biodiversity records (**OBIS**).

---

## 1. What Was Done and Why

### Step 1: 0.25° Spatial Grid Integration (`src/integrate.py`)
- **What was done:** Harmonized Copernicus Reanalysis NetCDF datasets (`cmems_mod_glo_phy_my_0.083deg_P1M-m`) and GFW AIS-derived apparent fishing logs into a single 0.25° monthly panel spanning 2019 through 2025 across the Arabian Sea (50°–78°E, 0°–25°N) and Bay of Bengal (78°–100°E, 0°–23°N).
- **Why it was done this way:**
  - Copernicus native resolution is 0.083°, while GFW is coarser. Coarsening Copernicus by a factor of 3 (3×3 block averaging) creates standardized 0.25° cells (~28 km).
  - Shallowest surface depth layer (`depth = 0`) was selected to correspond to surface fishing operations, and velocity magnitude was computed as $speed = \sqrt{uo^2 + vo^2}$.
  - GFW apparent hours were summed across flags per cell-month without summing `vesselIDs` (to avoid double-counting multi-flag vessels).
  - **Zero-Filling:** GFW logs events only when fishing is detected. An ocean cell with no detection represents **zero apparent fishing hours**, not missing data. Skipping this would introduce severe selection bias where models only see fished water.

### Step 2: Multi-Domain Feature Engineering (`src/features.py`)
- **What was done:** Engineered cyclical monsoon harmonics ($\sin(2\pi m / 12)$, $\cos(2\pi m / 12)$), spatial distance to coast via `cKDTree` on Copernicus land cells, monthly cell climatologies, physical anomalies (SST anomaly, Salinity anomaly, SST z-score), and 1-month autoregressive lags.
- **Why it was done this way:**
  - Indian Ocean fisheries are governed by annual monsoon wind reversals and national regulatory closures (e.g. India's annual 61-day trawl bans). Harmonic trigonometric terms represent this cycle without boundary discontinuities.
  - Continental shelf width and fuel costs strictly limit vessel range, making `dist_to_coast_km` an essential physical-economic predictor.
  - **Critical Guardrail:** Calendar `year` was **strictly excluded** from feature spaces to prevent models from learning the 2022 GFW satellite constellation coverage jump (+310.96% in Bay of Bengal).

### Step 3 & 4: Two-Stage Hurdle Modeling & Scientific Ablation (`src/train_models.py`)
- **What was done:**
  - Evaluated on an out-of-time chronological test split: **Train (2019–2023)**, **Validation (2024)**, **Test (2025)**. Random splitting was strictly avoided to prevent spatial and temporal autocorrelation leakage.
  - Built a two-stage hurdle framework:
    - **Stage A (Occurrence):** Predicts if fishing is present ($hours > 0$) using Logistic Regression, Random Forest, and XGBoost.
    - **Stage B (Intensity):** Predicts $\log(1 + hours)$ on positive cells using Ridge Regression, Random Forest, and XGBoost.
  - Compared against strict benchmarks: **Historical Climatology** and **Persistence**.
  - Conducted a scientific ablation study:
    - **Model 1:** Location + Seasonality only (6 features).
    - **Model 2:** Location + Seasonality + Ocean Variables (16 features).
  - Computed exact Shapley attributions using **native XGBoost TreeSHAP** (`pred_contribs=True`) and Scikit-Learn test permutation importance.
- **Why it was done this way:**
  - Apparent fishing is zero-inflated (78.94% zeros) and extreme-right skewed, causing single-stage regression models to fail.
  - The ablation delta isolates the true scientific contribution of physical ocean conditions.
  - **Empirical Scientific Finding:** Location (`lon_bin`, `lat_bin`, `dist_to_coast_km`) and seasonal monsoons account for **67.2% of decision gain** (Model 1 Stage A ROC-AUC: **0.8065**). Adding ocean variables yields ROC-AUC **0.7992** ($\Delta = -0.0074$). Reporting this marginal/null effect transparently is a valid scientific finding: national EEZs, bathymetry, and regulatory bans override ocean physical fluctuations.

### Step 5: Marine Heatwaves & Anomaly Detection (`src/anomalies.py`)
- **What was done:** Identified warm SST anomalies exceeding the 90th percentile threshold ($+0.528^\circ\text{C}$) or $z > 2.0$, and fitted an unsupervised `IsolationForest` on $(SST_{anomaly}, Salinity_{anomaly}, Speed)$. Quantified shifts in fishing hours using Mann-Whitney U testing.
- **Why it was done this way:**
  - Provides a second cross-domain result: evaluates whether marine heatwave conditions disrupt or stimulate fleet activity.
  - **Finding:** Under warm anomalies, fishing occurrence rises from **20.5% to 26.1%**, and mean hours increase by **+35.9%** ($p < 10^{-15}$), likely driven by pelagic stock compression along thermal fronts. Conversely, multivariate compound disruptions suppress vessel departure by **-38.1%** due to rough sea states.

### Step 6: Descriptive Biodiversity Layer & eDNA Flagging (`src/obis_env.py`)
- **What was done:** Matched 1,141 OBIS occurrences (2015–2025) with Copernicus physics and GFW hotspots. Extracted realized environmental envelopes (median, IQR, ranges) for *Carcharhinus melanopterus*, *Rhincodon typus*, and *Setipinna tenuifilis*.
- **Why it was done this way:**
  - **UNESCO Sundarbans eDNA Flag:** 39.34% of all OBIS records (100% of Setipinna) come from a single 2-day expedition (8–9 Jan 2023, 19 sampling sites).
  - eDNA denotes molecular sequence presence, NOT individual abundance.
  - **SDM Omission:** Fitting Species Distribution Models on this dataset would simply model the research boat's GPS track. Keeping this layer strictly descriptive prevents severe sampling-bias artifacts.

### Step 7: Interactive Streamlit Dashboard (`app/`)
- **What was done:** Built a multi-page interactive Streamlit dashboard (`Home.py` + 7 modular pages) presenting maps, model metrics, ablation deltas, TreeSHAP values, anomaly shifts, biodiversity envelopes, and project limitations.

---

## 2. Directory Structure

```
oceanai/
  config/
    regions.yaml                          # Spatial bounding boxes, paths, grid size, splits
  src/
    utils.py                              # Logging, fixed random seeds, config loading, cKDTree
    integrate.py                          # Step 1: 0.25° Copernicus + GFW integration & zero-filling
    features.py                           # Step 2: Multi-domain feature engineering (strict no-year rule)
    train_models.py                       # Step 3 & 4: Two-stage hurdle ML, baselines, ablation, TreeSHAP
    anomalies.py                          # Step 5: SST anomalies, Isolation Forest, fishing shifts
    obis_env.py                           # Step 6: Realized environmental matching & eDNA flagging
  data/processed/integrated/              # Output parquet & CSV tables (1.25M records)
  models/                                 # Serialized trained XGBoost models (joblib)
  reports/
    figures/                              # High-resolution (300 DPI) publication figures
    metrics/                              # Benchmark metrics JSON and CSV tables
  app/
    Home.py                               # Executive dashboard landing page
    pages/
      1_Ocean_Map.py                      # Interactive gridded 0.25° ocean map
      2_ML_Modeling.py                    # Hurdle framework & baseline comparison
      3_Ablation_Study.py                 # Controlled ablation (Model 1 vs Model 2)
      4_Feature_Importance.py             # Native TreeSHAP & permutation drops
      5_Anomaly_Detection.py              # Marine heatwaves & fishing effort shifts
      6_Biodiversity_OBIS.py              # Realized environmental ranges & eDNA caveat
      7_About_Limitations.py              # Known caveats & inferential boundaries
  requirements.txt                        # Python dependencies
  Makefile                                # Pipeline automation
  README.md                               # Project documentation
```

---

## 3. Mandatory Analytical Caveats

1. **GFW 2022 AIS Coverage Jump:**
   The +310.96% increase in apparent fishing hours in the Bay of Bengal starting in 2022 is an observational satellite constellation upgrade artifact, NOT an actual threefold fleet increase. Calendar `year` is strictly omitted from model training.
2. **Malacca Strait Incursion:**
   The Bay of Bengal bounding box extends to 100°E and incorporates the Andaman Sea and northern entrance to the Malacca Strait. This explains the extreme hotspot at **4.1°N, 99.9°E** and the presence of Southeast Asian flags (MYS, IDN, THA).
3. **Apparent Effort ≠ True Abundance:**
   GFW tracks AIS transponders (typically larger industrial vessels). Artisanal and small mechanized craft without AIS are invisible. Furthermore, high apparent hours indicate vessel search/trawl duration, not fish biomass.
4. **UNESCO Sundarbans eDNA Expedition:**
   39.34% of all OBIS records (476 of 1,210) originate from a single 2-day cruise in the Sundarbans mangrove delta (8–9 Jan 2023). eDNA reads denote molecular presence, not individual abundance. Species Distribution Modeling (SDM) is omitted to avoid predicting research cruise tracks.
5. **Exploratory Associations vs Causal Inference:**
   All cross-domain relationships reported are observational associations and do not imply direct causal mechanisms. Latent socioeconomic drivers (fuel subsidies, export prices, maritime border enforcement) play major unobserved roles.

---

## 4. How to Run

### Installation
Ensure Python 3.10+ is available and install dependencies:
```bash
pip install -r oceanai/requirements.txt
```

### Running the Complete Pipeline
You can run each step standalone with CLI arguments, or use the Makefile:
```bash
# Option A: Run via Makefile
make -C oceanai pipeline

# Option B: Run each step standalone
python oceanai/src/integrate.py
python oceanai/src/features.py
python oceanai/src/train_models.py
python oceanai/src/anomalies.py
python oceanai/src/obis_env.py
```

### Launching the Streamlit Dashboard
```bash
streamlit run oceanai/app/Home.py
```
*(Navigate through Pages 1 to 7 via the left sidebar).*
