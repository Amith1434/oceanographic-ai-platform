# Oceanographic AI Platform Makefile
PYTHON := python

.PHONY: all integrate features train anomalies obis pipeline tables test app clean help

all: pipeline tables test
	@echo "All pipelines, report tables, and tests completed successfully."

help:
	@echo "Available commands:"
	@echo "  make all         - Run full pipeline, build report tables, and execute test suite"
	@echo "  make integrate   - Run Step 1 data integration (Copernicus + GFW on 0.25 deg grid)"
	@echo "  make features    - Run Step 2 feature engineering (harmonics, lags, train-only climatologies)"
	@echo "  make train       - Run Step 3 & 4 hurdle modeling, 5-model ablation, TreeSHAP, and rolling origin"
	@echo "  make anomalies   - Run Step 5 TWFE regressions, within-year contrasts, and bootstrap CIs"
	@echo "  make obis        - Run Step 6 OBIS 4-category matching and UNESCO Sundarbans eDNA layer"
	@echo "  make tables      - Run programmatic extraction of metrics into exact report tables"
	@echo "  make test        - Run automated pytest suite verifying conservation, splits, and features"
	@echo "  make pipeline    - Run full end-to-end data and modeling pipeline sequentially"
	@echo "  make app         - Launch interactive Streamlit dashboard"

integrate:
	$(PYTHON) src/integrate.py

features:
	$(PYTHON) src/features.py

train:
	$(PYTHON) src/train_models.py

anomalies:
	$(PYTHON) src/anomalies.py

obis:
	$(PYTHON) src/obis_env.py

tables:
	$(PYTHON) reports/build_report_tables.py

test:
	pytest tests/ -v

pipeline: integrate features train anomalies obis
	@echo "End-to-end pipeline finished successfully."

app:
	streamlit run app/Home.py
