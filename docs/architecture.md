# Architecture & Design Decisions

This document explains **how** the Industrial IoT Sensor Analytics Platform is built and **why** each design choice was made. For the problem statement and a high-level overview, see the [README](../README.md).

---

## 1. Goals

| Goal | How the architecture supports it |
|---|---|
| **Trustworthy data** | Layered pipeline with validation and a data-quality report at the Silver layer |
| **Reproducibility** | Raw data is never modified; every layer can be rebuilt from the layer before it; random processes use fixed seeds |
| **Early failure detection** | Gold-layer features feed anomaly-detection and failure-prediction models |
| **Business decisions** | Star-schema data model and a business-insights layer translate predictions into cost and downtime |
| **Maintainability** | Central configuration, structured logging, automated tests, one module per layer |

---

## 2. End-to-End Data Flow

```
 Kaggle: Azure PdM dataset (5 CSV files)
        │  download
        ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │ data/raw/        original CSVs (never modified)                 │
 │                  + telemetry copy with injected quality issues  │
 │                  + business reference data (costs, plants)      │
 └─────────────────────────────────────────────────────────────────┘
        │  ingest as-is + metadata
        ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │ data/bronze/     Parquet, one table per source file             │
 └─────────────────────────────────────────────────────────────────┘
        │  clean · validate · deduplicate · standardise
        ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │ data/silver/     trusted tables + data-quality report           │
 └─────────────────────────────────────────────────────────────────┘
        │  aggregate · engineer features · label · compute KPIs
        ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │ data/gold/       feature table, KPI tables, star schema         │
 └─────────────────────────────────────────────────────────────────┘
        │                                  │
        ▼                                  ▼
  Machine learning ── predictions ──► fact_predictions
  (models/)                                │
                                           ▼
                                 Dashboard & business insights
                                 (dashboard/, reports/)
```

---

## 3. Layer Details

### 3.1 Sensor Data (Source)

**Dataset:** Microsoft Azure Predictive Maintenance (Kaggle). It covers 100 machines with hourly records for 2015.

| File | Grain | Key columns |
|---|---|---|
| `PdM_telemetry.csv` | machine × hour | `datetime`, `machineID`, `volt`, `rotate`, `pressure`, `vibration` |
| `PdM_errors.csv` | error event | `datetime`, `machineID`, `errorID` |
| `PdM_maint.csv` | component replacement | `datetime`, `machineID`, `comp` |
| `PdM_failures.csv` | component failure | `datetime`, `machineID`, `failure` |
| `PdM_machines.csv` | machine | `machineID`, `model`, `age` |

**Hybrid enhancements:**
- **Data-quality injection.** The source data is unusually clean. Real sensor feeds are not. A reproducible script (fixed seed, rates in `config/settings.yaml`) writes a *separate copy* of the telemetry with missing values, impossible spikes and duplicate rows. The original files stay untouched, and the injection is documented, so it is transparent and not hidden.
- **Business reference data.** The dataset has no costs or plant structure. A small reference table adds plant, production line, downtime cost per hour and repair cost per component. These values are illustrative assumptions and are configurable.

### 3.2 Bronze Layer: Raw Landing

| | |
|---|---|
| **Input** | `data/raw/*.csv` |
| **Output** | `data/bronze/<table>.parquet` |
| **Transformations** | None to the data itself. Adds `_source_file`, `_ingested_at`, `_batch_id`. |

**Why:** Bronze is a faithful, queryable copy of the source. If a downstream bug is found, Silver and Gold can be rebuilt from Bronze without downloading again, and Bronze always shows what the source actually sent.

### 3.3 Silver Layer: Cleaned & Validated

| | |
|---|---|
| **Input** | Bronze tables |
| **Output** | `data/silver/<table>.parquet` + data-quality report |

Transformations, in order:
1. **Standardise:** snake_case column names, parsed timestamps, correct numeric types.
2. **Deduplicate:** one row per `(machine_id, datetime)`.
3. **Validate:** readings outside the physical limits in `config/settings.yaml` → null (a sensor glitch is not a real measurement).
4. **Regularise:** each machine on a complete hourly grid. Short gaps are interpolated and long gaps stay null.
5. **Enrich:** join machine master data (model, age) and reference data (plant, line).

**Data-quality report:** rows in/out, duplicates removed, values nulled per rule, and remaining nulls. Quality problems are measured and reported, not silently fixed.

### 3.4 Gold Layer: Business & ML Ready

| Table | Purpose |
|---|---|
| `features` | One row per machine × hour. Rolling statistics (e.g. 3h / 24h mean and std), trends, error counts, time since last maintenance, and labels. |
| KPI tables | Uptime, downtime, MTBF, MTTR and availability per machine / model / line |
| Star schema | See 3.5 |

**Labels:**
- `fails_within_24h`: 1 if a failure occurs in the next `ml.prediction_horizon_hours`.
- `remaining_useful_life_hours`: hours until the next failure.

### 3.5 Data Model: Star Schema

| Table | Type | Grain |
|---|---|---|
| `dim_machine` | Dimension | machine (model, age, plant, line) |
| `dim_date` | Dimension | calendar day |
| `dim_sensor` | Dimension | sensor (name, unit, limits) |
| `dim_component` | Dimension | component (repair cost) |
| `fact_sensor_hourly` | Fact | machine × hour × sensor statistics |
| `fact_errors` | Fact | error event |
| `fact_maintenance` | Fact | failure / replacement event (downtime, cost) |
| `fact_predictions` | Fact | machine × hour model outputs |

**Why a star schema:** it is the standard model for analytics. It keeps dashboard queries simple (facts joined to small dimensions), makes business questions easy to express in SQL, and separates *what happened* (facts) from *descriptive context* (dimensions). DuckDB queries the Parquet files directly, so no database server is needed.

### 3.6 Machine Learning

| Model | Task | Output |
|---|---|---|
| Anomaly detection | Unsupervised (Isolation Forest) | anomaly score per machine-hour |
| Failure prediction | Binary classification (gradient boosting) | probability of failure within 24h |
| Remaining useful life | Regression | hours until failure |

**Evaluation rules:**
- **Time-based split** at `ml.train_end_date`: train on the past and test on the future. A random split would leak future information into training and overstate performance.
- **Imbalance-aware metrics:** failures are rare, so precision, recall, F1 and PR-AUC matter more than accuracy.
- **Explainability:** feature importance / SHAP, so maintenance teams understand *why* a machine is flagged.

### 3.7 Dashboard

Streamlit app reading from the Gold layer and the star schema:
Executive Overview · Machine Health · Sensor Explorer · Maintenance Planner · Data Quality.

### 3.8 Business Insights

Translates model output into decisions: avoidable downtime and cost (using the configured business assumptions), reliability KPIs, which sensor patterns come before which failures, and a prioritised maintenance list.

---

## 4. Cross-Cutting Concerns

| Concern | Implementation |
|---|---|
| **Configuration** | `config/settings.yaml`, loaded and validated by `iiot.config`. No hard-coded paths, limits or costs. Use `IIOT_CONFIG` to point at another file. |
| **Logging** | `iiot.utils.logger.get_logger(__name__)`. Writes to console and `logs/pipeline.log` (rotating, 5 MB × 3). Level is set in config. |
| **Testing** | pytest. Tests use temporary config copies and never touch real data or logs. |
| **Code quality** | ruff for linting and formatting (config in `pyproject.toml`). |
| **Version control** | Data, models, logs and secrets are excluded via `.gitignore`. Only code and documentation are committed. |

---

## 5. Key Design Decisions

| Decision | Alternatives considered | Reason |
|---|---|---|
| **Medallion architecture** (Bronze/Silver/Gold) | Single cleaning script | Each layer has one responsibility, can be rebuilt independently, and is easy to debug and explain |
| **Parquet** for storage | CSV, a database server | Columnar, compressed, typed, fast to read. Industry standard for data lakes. |
| **DuckDB** for SQL | PostgreSQL, SQLite | Runs in-process with no server, and queries Parquet directly. Fast analytical SQL on a laptop. |
| **pandas** for processing | PySpark | The dataset (~1M rows) fits in memory. The layered design could move to Spark later without changing the architecture. |
| **Public dataset + documented enhancements** | Fully simulated data | Real-world structure gives credibility. Documented enhancements fill the gaps (data-quality issues, costs) transparently. |
| **YAML config + typed loader** | Constants in code | One place to change thresholds and assumptions. Invalid values fail fast with clear errors. |
| **Time-based train/test split** | Random split | Matches how the model is used in production (predicting the future) and avoids leakage |

---

## 6. Module Map

| Module | Status | Responsibility |
|---|---|---|
| `iiot.config` | ✅ Done | Load and validate settings |
| `iiot.utils.logger` | ✅ Done | Project-wide logging |
| `iiot.ingestion` | ⏳ Phase 2 | Download dataset, inject quality issues, build reference data |
| `iiot.bronze` | ⏳ Phase 3 | Raw → Bronze |
| `iiot.silver` | ⏳ Phase 4 | Bronze → Silver + data-quality report |
| `iiot.gold` | ⏳ Phase 5 | Features, labels, KPIs |
| `iiot.data_model` | ⏳ Phase 6 | Star schema |
| `iiot.ml` | ⏳ Phase 7 | Train, evaluate and score models |
| `dashboard/` | ⏳ Phase 8 | Streamlit app |
| `reports/` | ⏳ Phase 9 | Business insights report |
