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
- **Data contract** (`iiot.ingestion.validate_raw`). 45 checks run before any processing: files present, expected columns, row-count ranges, no nulls, timestamps parse and fall in the expected range, valid machine IDs (exactly 100 machines), numeric sensors and allowed category values. A failed check stops the pipeline.
- **Data-quality injection** (`iiot.ingestion.inject_issues`). The source data is unusually clean. Real sensor feeds are not. A reproducible script (fixed seed, rates in `config/settings.yaml`) writes `PdM_telemetry_dirty.csv`, a *separate copy* of the telemetry, with four problem types:

  | Problem | Rate / size | Simulates |
  |---|---|---|
  | Stuck sensor | 200 episodes of 3–12 h | a frozen sensor repeating its last value |
  | Missing values | 1% of sensor cells | dropped readings |
  | Spikes | 0.2% of sensor cells, always outside the validation limits | sensor glitches (`-999`, `9999`) |
  | Duplicate rows | 0.5% of rows, right after the original | gateway retries |

  Missing values and spikes never land on stuck cells or each other, so counts are exact. `injection_manifest.json` records exactly what was injected. The original file stays untouched and the injection is documented, so it is transparent, not hidden.
- **Business reference data** (`iiot.ingestion.reference_data`). The dataset has no costs or plant structure.
  - `ref_machine_location.csv`: machines shuffled with a fixed seed and split evenly across production lines (Pune: 3 lines, Chennai: 2 lines, 20 machines each).
  - `ref_component_costs.csv`: per component, the cost of an **unplanned failure** (repair cost × emergency premium + unplanned downtime × hourly downtime cost) vs **planned maintenance** (repair cost + planned downtime × hourly cost), and the saving if a failure is prevented.

  All values are illustrative assumptions, configurable in `config/settings.yaml`.

All four Phase 2 steps run with one command: `iiot data prepare`.

### 3.2 Bronze Layer: Raw Landing

| | |
|---|---|
| **Input** | 7 files in `data/raw/`: the dirty telemetry (the sensor feed), errors, maintenance, failures, machines, machine location, component costs |
| **Output** | `data/bronze/<table>.parquet`, `_ingestion_log.json`, `bronze_report.json` |
| **Transformations** | None to the data itself. Adds `_source_file`, `_source_line`, `_ingested_at`, `_batch_id`. |
| **Code** | `iiot.bronze.ingest` (one file), `iiot.bronze.pipeline` (batch), `iiot.bronze.report` (verification) |

Design rules:
- **Schema-on-read:** every source column is stored as **text** exactly as received; empty cells stay `""`. Converting types here would crash on, or silently null out, malformed values. Silver does the typing, where problems are counted and reported.
- **Lineage:** `_source_line` is the line number in the source CSV (header = line 1), so any downstream value can be traced to the exact line it came from.
- **Reconciliation:** CSV data lines, rows read and Parquet rows must be equal, or the load fails.
- **Atomic writes:** each table is written to a temporary file and renamed, so a failed load never leaves a half-written table.
- **Idempotency:** each source file's SHA-256 hash is recorded in `_ingestion_log.json`. Unchanged files are skipped; `--force` reloads everything. Because Phase 2 is deterministic, re-running the full pipeline skips every Bronze table.
- **Verification:** `bronze_report.json` is produced with DuckDB SQL queries: table exists, is readable, row count and columns match the log, and all rows come from one batch. A raw file changed since loading is flagged as *stale*.
- **Why the dirty telemetry:** in this project it plays the role of the plant's sensor feed. The clean original stays in `data/raw/` as ground truth.

**Why:** Bronze is a faithful, queryable copy of the source. If a downstream bug is found, Silver and Gold can be rebuilt from Bronze without downloading again, and Bronze always shows what the source actually sent.

### 3.3 Silver Layer: Cleaned & Validated

| | |
|---|---|
| **Input** | Bronze tables |
| **Output** | `data/silver/<table>.parquet` (telemetry, machines, errors, maintenance, failures, component_costs) + `silver_quality_report.json` |
| **Code** | `iiot.silver.telemetry` (typing), `iiot.silver.cleaning` (rules), `iiot.silver.tables` (event/master tables), `iiot.silver.report` (build + proofs) |

Telemetry transformations, in order:
1. **Standardise:** snake_case names (`machine_id`, `timestamp`), typed columns. Empty and unparseable values are counted separately; rows without a valid timestamp or machine ID are dropped and counted.
2. **Deduplicate:** one row per `(machine_id, timestamp)`, keeping the first copy in source order. Duplicates whose values differ from the kept row are counted as *conflicting*.
3. **Validate:** readings outside the physical limits in `config/settings.yaml` → null (a sensor glitch is not a real measurement).
4. **Stuck sensors:** runs of 3+ identical consecutive hourly readings for one machine and sensor → keep the first value, null the repeats (a frozen sensor is not measuring).
5. **Regularise:** each machine on a complete hourly grid; off-hour timestamps are dropped and counted.
6. **Fill short gaps:** gaps of ≤ `silver.max_fill_hours` (3) with valid readings on both sides get the mean of the valid readings in a centred `silver.fill_window_hours` (24) window. Longer gaps stay null.
7. **Quality flags:** each sensor value is marked `ok`, `filled` or `missing`, so later phases can down-weight or exclude filled values.

Event and master tables: typed and validated against known machines, error IDs and components (from config and the data contract). Invalid rows are dropped and counted by reason (each row once); a machine without a location is kept but flagged. Machines are enriched with plant, city and production line.

**Data-quality report with two proofs:**
1. **Manifest check:** the cleaning must find exactly what `injection_manifest.json` says was injected (duplicates, spikes per sensor, stuck runs and repeats).
2. **Ground-truth check:** every `ok` value must equal the clean original; the error of filled values is recorded next to linear interpolation.

On the real data both pass. A proof is skipped (not failed) when its input file is missing.

**Decision record: gap filling.** The original plan was linear interpolation. Measured against the clean original data, hourly readings in this dataset vary so much from hour to hour that a straight line between neighbours is a poor estimate. The 24 h window mean was about 17% more accurate for every sensor (e.g. volt MAE 12.30 vs 14.74), so it replaced linear interpolation. The report keeps both numbers so the decision stays verifiable.

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
| **Command line** | `iiot run` for the full pipeline, or `iiot <group> <command>` (`iiot.cli`), e.g. `iiot data prepare`, `iiot bronze ingest`, `iiot silver build`. All steps live in one registry; each logs progress and timing, and a failed step stops the steps after it with a non-zero exit code. |
| **Version control** | Code, documentation and the 5 source CSVs are committed. Generated data (dirty copy, reference data, Bronze/Silver/Gold), models, logs and secrets are excluded via `.gitignore`, because they can be regenerated with one command. |

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
| **Gap filling: 24 h window mean** | Linear interpolation | Measured 17% lower error against the clean original data (see 3.3); filled values are flagged either way |
| **Time-based train/test split** | Random split | Matches how the model is used in production (predicting the future) and avoids leakage |

---

## 6. Module Map

| Module | Status | Responsibility |
|---|---|---|
| `iiot.config` | ✅ Done | Load and validate settings |
| `iiot.utils.logger` | ✅ Done | Project-wide logging |
| `iiot.cli` | ✅ Done | `iiot` command: `run`, `data`, `bronze` and `silver` groups (later phases add groups) |
| `iiot.ingestion` | ✅ Done | Download dataset, data contract, inject quality issues, build reference data |
| `iiot.bronze` | ✅ Done | Raw → Bronze: loader, idempotent batch pipeline, DuckDB report |
| `iiot.silver` | ✅ Done | Bronze → Silver: typing, cleaning rules, event/master tables, quality report with proofs |
| `iiot.gold` | ⏳ Phase 5 | Features, labels, KPIs |
| `iiot.data_model` | ⏳ Phase 6 | Star schema |
| `iiot.ml` | ⏳ Phase 7 | Train, evaluate and score models |
| `dashboard/` | ⏳ Phase 8 | Streamlit app |
| `reports/` | ⏳ Phase 9 | Business insights report |
