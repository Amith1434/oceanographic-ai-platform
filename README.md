
AI-Driven Unified Data Platform for Oceanographic, Fisheries and Molecular Biodiversity Insights
1. Project Overview
The Oceanographic AI Platform explores oceanographic conditions, marine biodiversity, fisheries activity, and molecular biodiversity information across selected regions of the Indian Ocean.

The project uses three independent primary data pipelines:

Copernicus Marine Service – oceanographic and environmental data.
Global Fishing Watch (GFW) – AIS-derived apparent fishing activity.
Ocean Biodiversity Information System (OBIS) – marine species occurrence observations.
Each dataset is collected, validated, preprocessed, analyzed, and visualized independently before evaluating whether meaningful integration is technically and scientifically justified.

Guiding principle: Collect accurately, process independently, analyze meaningfully, and integrate only when justified by the data.

2. Project Objectives
Analyze oceanographic variables such as temperature, salinity, and ocean currents.
Examine marine species occurrence and geographical distribution.
Analyze apparent fishing activity and identify areas with relatively high fishing intensity.
Explore molecular biodiversity reference information and potential DNA-barcode coverage gaps.
Develop and evaluate machine-learning or other analytical features.
Build an interactive dashboard for exploring the independent datasets.
Evaluate possible cross-dataset integration only after the individual pipelines are established.
3. Geographical Scope
The project focuses on two rectangular retrieval regions:

Region	Minimum longitude	Maximum longitude	Minimum latitude	Maximum latitude
Arabian Sea	50°E	78°E	0°N	25°N
Bay of Bengal	78°E	100°E	0°N	23°N
These are bounding boxes used for retrieval and filtering, not exact ocean-boundary polygons. Gridded data can include coastal or land-mask cells.

4. Dataset Sources, Periods and Current Status
Dataset	Period	Temporal resolution	Current status
Copernicus Marine	2019-01 to 2025-12	Monthly	Downloaded, combined, feature-engineered and inspected
Global Fishing Watch	2019-01 to 2025-12	Monthly	Downloaded for both regions; validated and cleaned
OBIS	2015-01 to 2025-12	Occurrence dates	Downloaded and initially cleaned
Molecular references	To be determined	To be determined	Not yet collected
4.1 Copernicus Marine – Oceanographic Data
Source: https://marine.copernicus.eu/
Selected dataset: cmems_mod_glo_phy_my_0.083deg_P1M-m
Product: Global Ocean Physics Reanalysis
Dataset version used: 202311

Selected variables:

Variable	Description	Unit
thetao	Potential temperature	°C
so	Sea-water salinity	Practical salinity units
uo	Eastward sea-water velocity	m/s
vo	Northward sea-water velocity	m/s
Collection configuration:

Period: January 2019 – December 2025.
Temporal resolution: Monthly.
Spatial resolution: approximately 0.083°.
Depth selection: approximately 0–5 m; downloaded files contain four shallow depth levels.
Regions: Arabian Sea and Bay of Bengal.
Format: NetCDF.
Completed work:

Downloaded all seven years for both regions: 14 annual NetCDF files.
Verified all annual files.
Confirmed 12 monthly time steps per annual file and 84 monthly time steps per region.
Combined annual files into one regional NetCDF file per region.
Engineered two additional variables:
current_speed = sqrt(uo² + vo²) in m/s.
current_direction, calculated from uo and vo as degrees counterclockwise from east, representing the direction the current flows toward.
Inspected variable ranges, percentiles, extreme values and local neighbourhoods.
Investigated unusually low coastal salinity values. Examined neighbourhoods showed spatially coherent values, so these observations were retained rather than automatically removed. Freshwater influence is a plausible interpretation, not a confirmed explanation.
Processed outputs:

data/processed/copernicus/
├── arabian_sea/
│   ├── copernicus_2019_2025.nc
│   └── copernicus_features_2019_2025.nc
└── bay_of_bengal/
    ├── copernicus_2019_2025.nc
    └── copernicus_features_2019_2025.nc
Each region has 84 monthly observations, four depth levels and the four original variables. Feature-engineered files additionally contain current speed and current direction.

Interpretation notes:

thetao is potential temperature, not automatically equivalent to sea-surface temperature.
Current direction is the direction the current flows toward, measured counterclockwise from east.
Land-mask cells account for much of the missing gridded data.
Extreme coastal salinity values should be interpreted with geographic and product context rather than discarded solely because they are unusual.
4.2 Global Fishing Watch – Fisheries Activity
Source: https://globalfishingwatch.org/
Selected dataset: public-global-fishing-effort:latest
Dataset returned by API: public-global-fishing-effort:v5.0
API: GFW 4Wings report endpoint
Metric: AIS-derived apparent fishing hours, aggregated by spatial grid cell, month and flag where available.

Collection configuration:

Period: January 2019 – December 2025.
Temporal resolution: Monthly.
Spatial resolution: LOW (0.1° grid, as configured in the API requests).
Regions: Arabian Sea and Bay of Bengal.
Retrieval: one region-month request at a time.
Raw format: JSON.
Consolidated format: CSV.
Completed work:

Obtained and tested API authentication.
Verified the API using small test requests.
Tested both regions for October 2021.
Retrieved 84 monthly periods for each region.
Saved monthly raw JSON responses.
Consolidated each region's monthly records into a CSV.
Validated date coverage, coordinates, missing values, duplicate rows, flags, fishing hours and vessel-ID counts.
Investigated the largest apparent fishing-hour records. These records were associated with multiple vessel IDs and were retained; they were not automatically treated as errors.
Created cleaned analysis-ready CSV copies, replacing missing flag values with Unknown while preserving original consolidated CSVs and raw JSON.
Downloaded data summary:

Metric	Arabian Sea	Bay of Bengal
Period	2019–2025	2019–2025
Monthly periods	84	84
Records	684,285	255,414
Unique grid cells	48,171	21,935
Distinct non-missing flag codes	49	91
Missing flag values in original CSV	26,291	17,958
Missing values after cleaning	0	0
Duplicate rows in audit	0	0
Total apparent fishing hours	9,315,764.61	5,637,766.36
Total records across both regions: 939,699.

Processed outputs:

data/processed/gfw/
├── arabian_sea/
│   ├── gfw_2019_2025.csv
│   └── gfw_clean.csv
└── bay_of_bengal/
    ├── gfw_2019_2025.csv
    └── gfw_clean.csv
Raw monthly JSON files are stored under:

data/raw/gfw/
├── arabian_sea/
│   ├── 2019-01.json
│   └── ...
└── bay_of_bengal/
    ├── 2019-01.json
    └── ...
Data interpretation and quality notes:

GFW apparent fishing hours are not direct measurements of catch, fish abundance, or all fishing activity.
AIS-based data has coverage and detection limitations; activity not represented in AIS-derived products should not be assumed absent.
Missing flag values are retained and represented as Unknown only in cleaned analysis copies.
The largest records are aggregated observations involving multiple vessel IDs. They have been retained pending further analytical review.
vesselIDs is a count of vessel IDs represented in an aggregate record, not a vessel identity list.
Total apparent fishing hours are sums of returned aggregate records and should be interpreted within the dataset's aggregation methodology.
4.3 OBIS – Marine Biodiversity Data
Source: https://obis.org/

OBIS provides marine biodiversity observations, including species occurrence records and geographical information.

Collection configuration:

Period: 2015–2025.
Regions: Arabian Sea and Bay of Bengal.
Format: CSV.
Selected species: three.
Selected species:

Common name	Scientific name	Selection
Blacktip Reef Shark	Carcharhinus melanopterus	Primary species
Whale Shark	Rhincodon typus	Secondary species
Indian Threadfin Bream	Setipinna tenuifilis	Bay of Bengal case study
Downloaded occurrence counts:

Species	Arabian Sea	Bay of Bengal
Blacktip Reef Shark	469	126
Whale Shark	107	32
Indian Threadfin Bream	0	476
Total	576	634
Total downloaded records: 1,210.

Completed preprocessing:

Date parsing and validation.
Duplicate ID checking.
Missing-value inspection.
Latitude and longitude validation.
Bounding-box filtering.
Date-range filtering.
Removal of invalid records where applicable.
Fixed mixed-format date parsing with pd.to_datetime(..., format="mixed", errors="coerce", utc=True).
Processed output:

data/processed/obis/
└── obis_occurrences_clean.csv
Occurrence records describe reported observations, not direct estimates of species population or abundance. OBIS sampling effort and geographic reporting bias must be considered during analysis.

4.4 Molecular Biodiversity Sources – Planned
Potential sources:

NCBI
BOLD Systems
The molecular reference-data pipeline has not yet been collected or processed. It is planned as a separate workstream to examine DNA barcode/reference availability and potential taxonomic coverage gaps. Molecular reference coverage is not equivalent to species occurrence or abundance.

5. Current Architecture and Development Approach
Each source has an independent pipeline:

Ingestion – retrieve source data and preserve raw files.
Validation – check schema, dates, coordinates, missingness, duplicates and coverage.
Preprocessing – produce analysis-ready data while preserving raw source data.
EDA and analysis – investigate patterns within each domain.
Visualization – build domain-specific charts and maps.
AI/ML evaluation – implement a justified analytical feature and evaluate it.
Only after the individual pipelines are understood will the team assess possible integration using compatible spatial and temporal units.

Potential integration work, if supported by the data, includes shared geographic references, compatible temporal aggregation, combined maps, environmental and biodiversity comparisons, and fishing-pressure comparisons. Integration is not assumed to be successful in advance.

6. Planned Analytical and AI Features
6.1 Oceanographic analysis
Monthly and seasonal temperature patterns.
Salinity distribution and variability.
Current speed and direction patterns.
Regional and temporal comparisons.
Investigation of unusual environmental observations.
6.2 Fisheries analysis
Fishing activity and apparent fishing-hour maps.
Spatial fishing-intensity hotspots.
Monthly, seasonal and annual patterns.
Flag-wise comparisons, including records with unknown flags.
Regional comparisons.
6.3 Biodiversity analysis
Species occurrence maps.
Temporal occurrence patterns.
Comparison of selected species across regions.
Analysis of sampling and reporting limitations.
6.4 Molecular biodiversity coverage analysis
Reference sequence or barcode availability for selected taxa.
Taxonomic coverage and gaps.
Reference availability comparisons across selected species.
This work remains planned and depends on source access and data feasibility.

6.5 Machine learning
A model or analytical feature will be selected after EDA and a clear problem definition. Potential directions include oceanographic anomaly detection, habitat-suitability or species-occurrence modelling (if observations and environmental features support a defensible design), and fishing activity pattern analysis.

Models must have an appropriate baseline, evaluation metrics and validation strategy. Spatial and temporal leakage must be considered. A model should not be presented as validated merely because it produces predictions.

6.6 Exploratory monitoring-priority analysis
A combined monitoring-priority indicator may be explored only if the independent datasets prove compatible. Any such indicator would be an exploratory screening aid, not a validated conservation or fisheries-management metric.

7. Technology Stack
Area	Technology
Programming	Python
Data processing	Pandas, NumPy
Oceanographic data	xarray, NetCDF
Copernicus retrieval	copernicusmarine
GFW retrieval	requests, GFW 4Wings API
OBIS retrieval	OBIS API / CSV
Visualization	Matplotlib, Plotly
Machine learning	Scikit-learn
Interactive dashboard	Streamlit
Geographical visualization	Folium / Plotly Maps
Data formats	CSV, JSON, NetCDF; Parquet may be considered
Development	Visual Studio Code
Version control	Git and GitHub
Shared data handoff	Team-managed cloud drive
8. Current Repository Structure
The structure below reflects the implemented data pipelines. Additional files and modules may exist as development continues.

oceanographic-ai-platform/
├── config/
│   ├── regions.json
│   └── settings.py
├── data/
│   ├── raw/
│   │   ├── obis/
│   │   ├── copernicus/
│   │   │   ├── arabian_sea/
│   │   │   └── bay_of_bengal/
│   │   └── gfw/
│   │       ├── arabian_sea/
│   │       └── bay_of_bengal/
│   └── processed/
│       ├── obis/
│       │   └── obis_occurrences_clean.csv
│       ├── copernicus/
│       │   ├── arabian_sea/
│       │   └── bay_of_bengal/
│       └── gfw/
│           ├── arabian_sea/
│           └── bay_of_bengal/
├── src/
│   ├── ingestion/
│   │   ├── download_obis.py
│   │   ├── download_copernicus.py
│   │   ├── inspect_copernicus.py
│   │   ├── test_copernicus.py
│   │   ├── verify_copernicus.py
│   │   ├── test_gfw.py
│   │   └── download_gfw.py
│   ├── preprocessing/
│   │   ├── clean_obis.py
│   │   ├── prepare_copernicus.py
│   │   ├── engineer_copernicus_features.py
│   │   ├── check_copernicus_outliers.py
│   │   ├── check_copernicus_neighborhood.py
│   │   ├── inspect_gfw.py
│   │   ├── validate_gfw.py
│   │   ├── audit_gfw.py
│   │   ├── check_gfw_outliers.py
│   │   └── clean_gfw.py
│   ├── models/
│   └── ...
├── app/
├── notebooks/
├── reports/
├── requirements.txt
├── README.md
└── .gitignore
Do not commit API tokens or .env files. Large raw and processed datasets are intended to be shared through the team's agreed cloud-drive location rather than duplicated in Git.

9. Development Roadmap and Status
Phase 1 – Data collection

Establish geographical regions.

Explore OBIS data access.

Select marine species.

Download OBIS occurrence records.

Register and test Copernicus Marine access.

Download Copernicus data for 2019–2025 in both regions.

Verify Copernicus annual files.

Obtain and test GFW API access.

Retrieve GFW data for 2019–2025 in both regions.
Collect molecular biodiversity reference data.
Phase 2 – Data preprocessing and validation

Initial OBIS cleaning.

Combine and prepare Copernicus regional NetCDF files.

Engineer Copernicus current speed and direction.

Inspect Copernicus ranges and investigate selected extremes.

Consolidate GFW monthly responses into regional CSV files.

Validate GFW date coverage, coordinates, missing values and duplicates.

Audit GFW flags, fishing hours and vessel-ID counts.

Create cleaned GFW analysis copies with missing flags labelled Unknown.
Complete any additional domain-specific preprocessing required by EDA and modelling.
Finalize shared data handoff and reproducibility checks.
Phase 3 – Independent exploratory data analysis
Oceanographic variable analysis.
Marine biodiversity distribution analysis.
Fishing activity and hotspot analysis.
Molecular biodiversity coverage analysis.
Generate individual charts, maps and reports.
Phase 4 – AI and machine learning
Define the exact ML problem and target.
Establish baseline methods.
Prepare features and suitable train/validation/test strategy.
Train and evaluate selected models.
Perform spatial or temporal validation where appropriate.
Document limitations and model results.
Phase 5 – Dashboard development
Build the Streamlit application.
Create the oceanography section.
Create the fisheries section.
Create the biodiversity section.
Add model outputs and evaluation.
Add interactive geographical visualizations.
Phase 6 – Integration and final evaluation
Assess compatibility between datasets.
Evaluate whether a shared spatial grid is appropriate.
Explore combined geographical visualizations.
Develop an integrated analytical feature if feasible.
Document limitations and final results.
10. Current Project Status
Current stage: Data collection and initial preprocessing are substantially complete. The project is transitioning into independent EDA.

Completed
Project structure and two geographic retrieval regions established.
OBIS species selected; 1,210 occurrence records downloaded and initially cleaned.
Copernicus data downloaded for both regions across 2019–2025, verified, combined and feature-engineered.
GFW data downloaded for both regions across 2019–2025.
GFW monthly coverage verified: 84 months per region.
GFW regional CSVs validated and cleaned: 939,699 total records.
Missing GFW flags preserved as Unknown in cleaned copies.
Initial outlier investigations completed for Copernicus and GFW.
Immediate next steps
Transfer the agreed raw and processed datasets to shared cloud storage for both team members.
Verify that the other team member can open the shared files.
Begin independent EDA for GFW, Copernicus and OBIS.
Select the AI/ML feature after EDA findings and data suitability are clear.
Not yet completed
Molecular reference-data collection and analysis.
Full EDA and analytical reports.
Model development and evaluation.
Streamlit dashboard.
Cross-dataset integration and final evaluation.
11. Team Data-Sharing and Reproducibility
The project uses large files, especially monthly GFW JSON responses and Copernicus NetCDF files. Share them through a controlled team cloud-drive folder.

Recommended handoff structure:

Oceanographic_Project_Data/
├── raw/
│   ├── obis/
│   ├── copernicus/
│   │   ├── arabian_sea/
│   │   └── bay_of_bengal/
│   └── gfw/
│       ├── arabian_sea/
│       └── bay_of_bengal/
└── processed/
    ├── obis/
    ├── copernicus/
    │   ├── arabian_sea/
    │   └── bay_of_bengal/
    └── gfw/
        ├── arabian_sea/
        └── bay_of_bengal/
Recommended practices:

Share both raw and processed data so the other team member can reproduce and inspect the pipeline.
Keep original raw files unchanged.
Share repository code and requirements.txt separately from large data files.
Never upload .env, API tokens, passwords or other credentials.
Add a data inventory containing each file path, source, region, date coverage, format and purpose.
After upload, verify file counts and sizes and open representative files from the shared location.
Agree on one canonical processed dataset per region to avoid teammates editing different copies.
Avoid simultaneous edits to the same large files or scripts without coordination.
12. Important Data Considerations
OBIS occurrence counts are not estimates of actual species populations.
GFW apparent fishing activity is not total fishing catch or all fishing activity.
Copernicus variables are gridded oceanographic product values and must be interpreted with their resolution and product characteristics in mind.
thetao is potential temperature, not automatically sea-surface temperature.
Rectangular bounding boxes can include land cells and are not exact ocean boundaries.
Missing values must be examined before model training.
Different sources have different spatial resolutions, temporal coverage, sampling biases and definitions.
Missing GFW flag values are retained as Unknown in cleaned copies.
High GFW apparent fishing-hour aggregates have not been removed solely for being extreme.
Molecular reference availability does not represent actual species abundance.
Any integrated priority index must be described as exploratory unless independently validated.
API terms, attribution requirements and permitted-use conditions must be followed when sharing or publishing derived data.
13. Project Philosophy
The primary goal is not simply to combine multiple marine datasets into one interface. The project aims to build reliable and reproducible data pipelines, conduct meaningful analysis in each domain, and use machine learning only where it provides measurable analytical value.

The guiding principle: Collect accurately, process independently, analyze meaningfully, and integrate only when justified by the data.

0