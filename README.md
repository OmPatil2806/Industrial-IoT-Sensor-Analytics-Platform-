# Industrial IoT Sensor Analytics Platform

## Overview

An end-to-end Data Engineering and Machine Learning platform that turns raw industrial sensor data into 24-hour failure warnings and business decisions, moving a plant from reactive to predictive maintenance.

Plants usually repair machines after they break, which means unplanned downtime, emergency repairs and sensor data too messy to trust. This project builds the full path from raw sensor readings to a prioritised maintenance plan with its cost impact:

- an automated, tested data pipeline (Bronze, Silver, Gold) that ingests, cleans and validates sensor data
- a DuckDB star-schema warehouse for analysis
- ML models that predict failures 24 hours ahead, name the failing component and flag unusual sensor behaviour
- a Streamlit dashboard for operations, maintenance and management
- a business case with savings, payback and where to focus first

**Data:** [Microsoft Azure Predictive Maintenance dataset](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance) (synthetic). 100 machines with hourly readings for 2015: 876,100 sensor rows, 3,919 errors, 3,286 component replacements and 761 component failures. Added on top: plants and production lines (Pune, Chennai), component costs in INR (`config/settings.yaml`), and injected data-quality issues (stuck sensors, gaps, spikes, duplicates) to prove the cleaning works.

**Tech stack:** Python 3.11, pandas, Parquet (PyArrow), DuckDB, scikit-learn, MLflow, Streamlit, Altair, pytest, ruff.

---

## Architecture

```
 RAW DATA ──► BRONZE ──────► SILVER ───────► GOLD ──────────► DATA MODEL ──────► DASHBOARD
 5 CSV files  raw Parquet    cleaned,        features,        DuckDB star        Streamlit,
 + reference  + lineage,     validated,      labels, KPIs,    schema + views     6 pages
 data         idempotent     quality flags   ML dataset             ▲                │
                                                  │                 │ predictions    ▼
                                                  └──► MACHINE LEARNING ───► BUSINESS INSIGHTS
                                                       failure 24 h,         scenarios, payback,
                                                       component, anomaly    report, focus areas
                                                       (MLflow tracking)

 Every stage ends with automated quality checks. One command runs everything: iiot run
```

### 1. Data acquisition
- Downloads the 5 source CSV files from Kaggle (also committed in `data/raw/`).
- **Data contract:** 45 automated checks (files, columns, row counts, date ranges, machine IDs, allowed values) confirm the source is complete before any processing.
- **Data-quality injection:** a reproducible script adds stuck sensors, missing values, impossible spikes and duplicate rows to a copy of the telemetry. A manifest records exactly what was injected. The original files are never modified.
- **Reference data:** assigns each machine to a plant and production line, and builds a component cost table (unplanned failure vs planned maintenance, downtime hours).

### 2. Bronze layer (raw landing)
- Stores an unchanged copy of all 7 sources as Parquet, with every column kept as text (schema-on-read), so nothing is lost or silently converted.
- Adds lineage columns to every row: source file, source line, ingestion time, batch ID.
- **Idempotent:** each file's SHA-256 hash is logged, so re-runs skip unchanged files. Row counts are reconciled on every load.
- A DuckDB report verifies every table: it exists, is readable, and has the expected rows and columns.

### 3. Silver layer (cleaned and validated)
- Type casting, timestamp standardisation and consistent naming; unparseable values are counted.
- Deduplication, physical-limit validation (spikes become null) and stuck-sensor detection.
- A complete hourly grid per machine. Short gaps (up to 3 h) are filled with the 24 h mean; longer gaps are left empty.
- A quality flag per sensor value: `ok`, `filled` or `missing`.
- **Proven, not assumed:** the quality report checks the cleaning against the injection manifest and the clean original data.

### 4. Gold layer (ML and business ready)
- **Features** (one row per machine every 3 hours): 3 h and 24 h rolling sensor statistics, trends, data-quality shares, error counts, hours since each component was replaced, machine attributes. Features only look backwards.
- **Labels:** failure within 24 h, failure per component, failed component and hours to failure. Labels only look forwards.
- **ML dataset** with a leakage-safe, time-based split.
- **KPIs** per machine, line and plant by month: failures, MTBF, MTTR, downtime, availability and cost.
- Checks: a leakage test, labels recomputed independently, and KPIs reconciled with Silver.

### 5. Data model (DuckDB star schema)
Built from SQL scripts into `data/warehouse/iiot.duckdb`.

| Type | Tables |
|---|---|
| Dimensions | `dim_machine`, `dim_date`, `dim_sensor`, `dim_component`, `dim_error_type` |
| Facts | `fact_sensor_reading`, `fact_failure`, `fact_maintenance`, `fact_error`, `fact_machine_month`, `fact_prediction` |
| Views | `v_fleet_monthly`, `v_machine_health`, `v_component_reliability`, `v_line_performance`, `v_machine_risk_now` |

Six model checks run after every build: unique keys, foreign-key references, date keys, reconciliation with Silver and Gold, and the views.

### 6. Machine learning (scikit-learn + MLflow)

| Model | Question | Algorithm |
|---|---|---|
| Failure prediction | Will this machine fail within 24 hours? | HistGradientBoosting classifier |
| Component diagnosis | Which component will fail? | Multi-output gradient boosting |
| Anomaly detection | Is the sensor behaviour unusual right now? | Isolation Forest |

How it is built:
- **Time-based split:** fit on Jan–Jul 2015, tune and choose thresholds on Aug–Sep, refit on Jan–Sep, then test once on Oct–Dec.
- **Baselines first:** every model must beat a simple rule to be registered as `@champion`.
- **Business metrics:** a false alarm costs one inspection (₹20,000); a caught failure saves unplanned cost minus planned cost. The alert threshold is the one with the highest net saving on validation.
- **Explanations:** permutation importance, and up to 3 plain-language reasons per alert (for example "vibration 24 h mean 48.9 vs normal 40.3 (+21%)").
- **MLflow:** tracks every run (parameters, metrics, charts, model cards, git commit, data fingerprint) and the model registry. Every `iiot run` is tracked as well.
- Predictions for every machine every 3 hours are written to `fact_prediction`; an ML report runs 6 checks (champions, data unchanged, predictions complete, thresholds applied, reproducible, test results match).

### 7. Dashboard (Streamlit + Altair)
Reads the warehouse through short-lived read-only connections, so it never blocks a pipeline run.

| Page | Shows |
|---|---|
| Fleet overview | Availability, failures, downtime, MTBF, MTTR and cost; monthly trends; lines and components |
| Machines at risk | Replay any moment of the test period: machines ranked by risk, the component, the reasons, and what happened next |
| Machine detail | One machine's risk timeline and sensor history with failures and maintenance marked |
| Maintenance planner | Prioritised to-do list with savings per action; CSV export |
| Business case | What-if calculator: catch rate, false alarms and costs give net saving, payback and break-even live |
| Model & data health | Quality checks per layer, models vs baselines, pipeline runs from MLflow |

### 8. Business insights
Turns the warehouse and the model test results into a business case: scenarios, break-even catch rate, sensitivity, focus areas (lines, machine age, models, components), warning signatures per component and a 3-month pilot plan. Output: [reports/business_insights.md](reports/business_insights.md) with charts.

---

## Results

### Machine learning (test months Oct–Dec 2015, never seen in training)

| Model | Result | Baseline |
|---|---|---|
| Failure within 24 h | 166 of 171 failures caught, 1 false alarm, ₹6.41 cr net saving (97% of possible) | error5 rule: 43 of 171 caught, 40 false alarms, ₹2.16 cr |
| Component diagnosis | Right component ranked first 100% of the time | Most frequent component: 42% |
| Anomaly detection | 13.0% of rows flagged in the 48 h before a failure vs 1.1% normally (12x lift) | Designed 1% flag rate |

Each component has its own warning signature: comp1 voltage + error1, comp2 rotation + error3, comp3 pressure + error4, comp4 vibration + error5. Leakage was checked four ways. The data are synthetic with strong patterns, so a real plant would score lower.

### Business case (per year)

| Scenario | Failures caught | Net saving | Downtime avoided | Payback |
|---|---|---|---|---|
| Conservative | 50% | ₹12.57 cr | 2,432 h | under 1 month |
| Expected | 70% | ₹18.30 cr | 3,404 h | under 1 month |
| As tested | 97% | ₹26.27 cr | 4,721 h | under 1 month |

- Today: 719 unplanned failures cost ₹41.48 cr and 11,154 hours of downtime a year.
- Break-even: the programme (₹50 lakh a year) pays for itself once it catches 1.9% of failures.
- Focus first on CHENNAI-L1 (8.85 failures per machine, 25% of failure cost), machines aged 15+ years (9.0 failures a year vs 5.5 under 10 years) and model1 (10.9 failures per machine vs 5.4 for model4).

### Engineering
- One command runs the full pipeline (11 steps, about 5 minutes); every stage ends with a quality report.
- 459 automated tests (pytest) and ruff lint/format checks.

---

## Running the project

### Setup
```bash
git clone https://github.com/OmPatil2806/Industrial-IoT-Sensor-Analytics-Platform-.git
cd Industrial-IoT-Sensor-Analytics-Platform-
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate         # macOS / Linux
pip install -r requirements-dev.txt
pip install -e .
```

### Run
```bash
iiot run              # full pipeline: data → Bronze → Silver → Gold → warehouse → ML → insights
iiot dashboard        # open http://localhost:8501 (keep the terminal open)
pytest                # run all tests
ruff check .          # lint
```

### All commands

| Command | Purpose |
|---|---|
| `iiot run` | Full pipeline (`--skip-ml` stops after the warehouse, `--no-tracking` skips MLflow) |
| `iiot data prepare` | Download, validate, inject issues, build reference data |
| `iiot bronze ingest` / `iiot bronze report` | Load and verify the Bronze layer |
| `iiot silver build` / `iiot silver report` | Build and check the Silver layer |
| `iiot gold build` / `iiot gold report` | Build and check the Gold layer |
| `iiot model build` / `iiot model report` | Build and check the DuckDB warehouse |
| `iiot ml train` / `iiot ml score` / `iiot ml report` | Train, score and check the ML models |
| `iiot insights build` / `iiot insights report` | Build the business case and report |
| `iiot dashboard --port 8501` | Start the dashboard |
| `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --workers 1` | MLflow UI at http://127.0.0.1:5000 |

Settings (paths, sensor limits, costs, ML) are in `config/settings.yaml`. Design details are in [docs/architecture.md](docs/architecture.md).

---

## Author

Om Patil: [GitHub @OmPatil2806](https://github.com/OmPatil2806)
