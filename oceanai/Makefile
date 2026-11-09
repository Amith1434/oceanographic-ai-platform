# Oceanographic AI Platform Makefile
PYTHON := python

.PHONY: all integrate features train anomalies obis pipeline app clean help

help:
	@echo "Available commands:"
	@echo "  make integrate   - Run Step 1 data integration (Copernicus + GFW on 0.25 deg grid)"
	@echo "  make features    - Run Step 2 feature engineering (harmonics, lags, anomalies, distance to coast)"
	@echo "  make train       - Run Step 3 & 4 two-stage hurdle modeling, ablation study, and TreeSHAP"
	@echo "  make anomalies   - Run Step 5 marine heatwave and multivariate anomaly detection"
	@echo "  make obis        - Run Step 6 OBIS realized environmental matching and eDNA flagging"
	@echo "  make pipeline    - Run full end-to-end pipeline sequentially"
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

pipeline: integrate features train anomalies obis
	@echo "End-to-end pipeline finished successfully."

app:
	streamlit run app/Home.py
