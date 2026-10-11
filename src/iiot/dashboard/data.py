"""Every query the dashboard runs, in one place (plain functions, no Streamlit).

All reads go to the warehouse through a SHORT-LIVED READ-ONLY connection: each
function opens, queries and closes, so a running dashboard never blocks
`iiot run` from rebuilding the warehouse. The Streamlit app adds caching on top.

Predictions are shown only for the out-of-sample period (from the train/test
split date), because earlier rows were used to train the models. The data end on
1 Jan 2016, so pages pick an "as of" time inside that period and replay what the
models said then.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import iiot.ml  # noqa: F401  (sets MLflow's quiet-logging options before MLflow is imported)
from iiot.config import get_settings
from iiot.model.warehouse import connect, table_counts, view_names, warehouse_path
from iiot.pipeline_tracking import report_files

MODEL_CARDS = {
    "failure": "failure_24h_model_card.json",
    "component": "component_model_card.json",
    "anomaly": "anomaly_model_card.json",
}


def query(sql: str, params: list | None = None, path: Path | None = None) -> pd.DataFrame:
    """Run one query on a short-lived read-only connection."""
    with connect(path) as con:
        return con.execute(sql, params or []).df()


def warehouse_exists(path: Path | None = None) -> bool:
    return Path(path or warehouse_path()).exists()


# ------------------------------------------------------------------ status


def status(path: Path | None = None) -> dict:
    """What the warehouse holds: table sizes, views, data and prediction periods."""
    with connect(path) as con:
        counts = table_counts(con)
        views = view_names(con)
        first, last = con.execute(
            "SELECT min(timestamp), max(timestamp) FROM fact_sensor_reading"
        ).fetchone()
        p_first, p_last = con.execute(
            "SELECT min(timestamp), max(timestamp) FROM fact_prediction WHERE out_of_sample"
        ).fetchone()
    return {
        "tables": counts,
        "views": views,
        "data_from": first,
        "data_to": last,
        "has_predictions": counts.get("fact_prediction", 0) > 0,
        "predictions_from": p_first,
        "predictions_to": p_last,
    }


def report_status() -> pd.DataFrame:
    """One row per quality report: layer, passed, generated_at in UTC (missing reports included)."""
    rows = []
    for layer, file in report_files().items():
        if file.exists():
            report = json.loads(file.read_text(encoding="utf-8"))
            generated = report.get("generated_at")
            # The raw validation report has no timestamp: use the file's time instead.
            when = (
                pd.Timestamp(generated)
                if generated
                else pd.Timestamp(file.stat().st_mtime, unit="s", tz="UTC")
            )
            rows.append(
                {
                    "layer": layer,
                    "passed": bool(report.get("passed")),
                    "generated_at": when,
                    "file": str(file),
                }
            )
        else:
            rows.append({"layer": layer, "passed": None, "generated_at": None, "file": str(file)})
    return pd.DataFrame(rows)


def report_checks(report: dict) -> pd.DataFrame:
    """The individual checks inside one quality report: check, passed.

    Layers write their reports differently: a list of results (raw), one entry per table
    (Bronze), *_check entries (Silver) or a "checks" mapping (Gold, warehouse, ML)."""
    rows = []
    if isinstance(report.get("results"), list):
        rows = [
            {
                "check": f"{r.get('table', '')}: {r.get('check', '')}".strip(": "),
                "passed": r["passed"],
            }
            for r in report["results"]
        ]
    elif isinstance(report.get("checks"), dict):
        rows = [
            {"check": name, "passed": bool(c.get("passed")) if isinstance(c, dict) else bool(c)}
            for name, c in report["checks"].items()
        ]
    else:
        rows = [
            {"check": name, "passed": bool(value["passed"])}
            for name, value in report.items()
            if name.endswith("_check") and isinstance(value, dict) and "passed" in value
        ]
        for table, value in (report.get("tables") or {}).items():
            if isinstance(value, dict) and "passed" in value:
                rows.append({"check": f"table {table}", "passed": bool(value["passed"])})
    return pd.DataFrame(rows, columns=["check", "passed"])


def load_report(layer: str) -> dict | None:
    file = report_files()[layer]
    return json.loads(file.read_text(encoding="utf-8")) if file.exists() else None


def pipeline_runs(mlruns_dir: Path | None = None, limit: int = 20) -> pd.DataFrame:
    """The latest `iiot run` executions recorded in MLflow (newest first)."""
    import mlflow
    from mlflow import MlflowClient

    from iiot.ml.tracking import tracking_uri
    from iiot.pipeline_tracking import EXPERIMENT

    mlruns_dir = Path(mlruns_dir or get_settings().paths.mlruns)
    columns = ["started", "minutes", "passed", "steps", "git_commit", "status"]
    if not (mlruns_dir / "mlflow.db").exists():
        return pd.DataFrame(columns=columns)
    mlflow.set_tracking_uri(tracking_uri(mlruns_dir))
    experiment = MlflowClient().get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        return pd.DataFrame(columns=columns)
    runs = MlflowClient().search_runs(
        [experiment.experiment_id], order_by=["attributes.start_time DESC"], max_results=limit
    )
    return pd.DataFrame(
        [
            {
                "started": pd.Timestamp(r.info.start_time, unit="ms", tz="UTC"),
                "minutes": (r.info.end_time - r.info.start_time) / 60000
                if r.info.end_time
                else None,
                "passed": bool(r.data.metrics.get("pipeline_passed", 0)),
                "steps": len(r.data.params.get("steps", "").split(",")),
                "git_commit": r.data.tags.get("git_commit", ""),
                "status": r.info.status,
            }
            for r in runs
        ],
        columns=columns,
    )


def model_cards(models_dir: Path | None = None) -> dict[str, dict]:
    """The model cards written by the last training (empty if not trained yet)."""
    models_dir = Path(models_dir or get_settings().paths.models)
    cards = {}
    for key, name in MODEL_CARDS.items():
        file = models_dir / name
        if file.exists():
            cards[key] = json.loads(file.read_text(encoding="utf-8"))
    return cards


# ------------------------------------------------------------------ fleet


PLANT_FILTER = "(CAST(? AS VARCHAR) IS NULL OR m.plant_id = ?)"


def plants(path: Path | None = None) -> pd.DataFrame:
    return query(
        "SELECT DISTINCT plant_id, plant_name FROM dim_machine ORDER BY plant_id", path=path
    )


def fleet_kpis(path: Path | None = None, plant_id: str | None = None) -> dict:
    """Headline numbers for the whole analysis period (one plant, or all if None)."""
    row = query(
        f"""
        SELECT CAST(sum(k.failures) AS INTEGER)             AS failures,
               CAST(sum(k.planned_maintenances) AS INTEGER) AS planned_maintenances,
               sum(k.downtime_h)                            AS downtime_h,
               1 - sum(k.downtime_h) / sum(k.period_hours)  AS availability,
               (sum(k.period_hours) - sum(k.downtime_h)) / nullif(sum(k.failures), 0) AS mtbf_h,
               sum(k.unplanned_downtime_h) / nullif(sum(k.failures), 0)              AS mttr_h,
               sum(k.failure_cost)                          AS failure_cost,
               sum(k.maintenance_cost)                      AS maintenance_cost,
               sum(k.total_cost)                            AS total_cost,
               count(DISTINCT k.machine_id)                 AS machines
        FROM fact_machine_month AS k
        JOIN dim_machine AS m USING (machine_id)
        WHERE {PLANT_FILTER}
        """,
        [plant_id, plant_id],
        path=path,
    )
    return {col: row[col].iloc[0].item() for col in row.columns}


def fleet_monthly(path: Path | None = None, plant_id: str | None = None) -> pd.DataFrame:
    """Monthly failures, downtime, availability and cost (one plant, or all if None).

    Same rules as v_fleet_monthly: ratios are recomputed from the summed hours."""
    return query(
        f"""
        SELECT k.year_month,
               CAST(sum(k.failures) AS INTEGER)            AS failures,
               sum(k.downtime_h)                           AS downtime_h,
               1 - sum(k.downtime_h) / sum(k.period_hours) AS availability,
               sum(k.failure_cost)                         AS failure_cost,
               sum(k.maintenance_cost)                     AS maintenance_cost,
               sum(k.total_cost)                           AS total_cost
        FROM fact_machine_month AS k
        JOIN dim_machine AS m USING (machine_id)
        WHERE {PLANT_FILTER}
        GROUP BY k.year_month
        ORDER BY k.year_month
        """,
        [plant_id, plant_id],
        path=path,
    )


def line_performance(path: Path | None = None) -> pd.DataFrame:
    return query("SELECT * FROM v_line_performance ORDER BY availability_rank", path=path)


def component_reliability(path: Path | None = None) -> pd.DataFrame:
    return query("SELECT * FROM v_component_reliability ORDER BY failure_cost DESC", path=path)


def machine_health(path: Path | None = None) -> pd.DataFrame:
    return query("SELECT * FROM v_machine_health ORDER BY machine_id", path=path)


def machines(path: Path | None = None) -> pd.DataFrame:
    return query("SELECT * FROM dim_machine ORDER BY machine_id", path=path)


# ------------------------------------------------------------------ predictions


def prediction_times(path: Path | None = None) -> pd.Series:
    """Every out-of-sample scoring time (every 3 hours), for the "as of" picker."""
    return query(
        "SELECT DISTINCT timestamp FROM fact_prediction WHERE out_of_sample ORDER BY timestamp",
        path=path,
    )["timestamp"]


def daily_alerts(path: Path | None = None) -> pd.DataFrame:
    """Per out-of-sample day: machines with a failure alert and machines with an anomaly flag."""
    return query(
        """
        SELECT CAST(timestamp AS DATE) AS day,
               count(DISTINCT machine_id) FILTER (WHERE alert)        AS machines_alerted,
               count(DISTINCT machine_id) FILTER (WHERE anomaly_flag) AS machines_flagged
        FROM fact_prediction
        WHERE out_of_sample
        GROUP BY day
        ORDER BY day
        """,
        path=path,
    )


def latest_alert_time(path: Path | None = None) -> pd.Timestamp | None:
    """The last out-of-sample scoring time with at least one alert (a useful default)."""
    value = query(
        "SELECT max(timestamp) AS t FROM fact_prediction WHERE out_of_sample AND alert",
        path=path,
    )["t"].iloc[0]
    return None if pd.isna(value) else pd.Timestamp(value)


def failures_between(
    start: pd.Timestamp, end: pd.Timestamp, path: Path | None = None
) -> pd.DataFrame:
    """Failure events (machine, time, components) with start < time <= end."""
    return query(
        """
        SELECT machine_id, timestamp, string_agg(component, '+' ORDER BY component) AS components
        FROM fact_failure
        WHERE timestamp > ? AND timestamp <= ?
        GROUP BY machine_id, timestamp
        ORDER BY timestamp, machine_id
        """,
        [pd.Timestamp(start).to_pydatetime(), pd.Timestamp(end).to_pydatetime()],
        path=path,
    )


def risk_as_of(as_of: pd.Timestamp, path: Path | None = None) -> pd.DataFrame:
    """Each machine's latest out-of-sample prediction at or before `as_of`, riskiest first."""
    return query(
        """
        WITH latest AS (
            SELECT max(timestamp) AS t FROM fact_prediction
            WHERE out_of_sample AND timestamp <= ?
        )
        SELECT rank() OVER (ORDER BY p.failure_risk DESC, p.anomaly_score DESC) AS risk_rank,
               p.machine_id, m.model, m.age, m.plant_id, m.line_id, p.timestamp AS scored_at,
               p.failure_risk, p.alert, p.predicted_component,
               p.comp1_risk, p.comp2_risk, p.comp3_risk, p.comp4_risk,
               p.anomaly_score, p.anomaly_flag, p.reason_1, p.reason_2, p.reason_3
        FROM fact_prediction AS p
        JOIN latest ON p.timestamp = latest.t
        JOIN dim_machine AS m USING (machine_id)
        WHERE p.out_of_sample
        ORDER BY risk_rank, p.machine_id
        """,
        [pd.Timestamp(as_of).to_pydatetime()],
        path=path,
    )


def maintenance_plan(as_of: pd.Timestamp, path: Path | None = None) -> pd.DataFrame:
    """Actions for `as_of`: replace the predicted component on alerted machines (highest
    risk and saving first), then inspect machines whose sensors are flagged as unusual."""
    plan = risk_as_of(as_of, path)
    plan = plan[plan["alert"] | plan["anomaly_flag"]].copy()
    if plan.empty:
        return plan.assign(priority=[], action=[], saving_if_acted=[], alert_since=[])
    savings = query(
        "SELECT component AS predicted_component, saving_if_prevented FROM dim_component",
        path=path,
    )
    since = query(
        """
        SELECT machine_id, min(timestamp) AS alert_since FROM fact_prediction
        WHERE out_of_sample AND alert AND timestamp <= ? AND timestamp > ? - INTERVAL 24 HOUR
        GROUP BY machine_id
        """,
        [pd.Timestamp(as_of).to_pydatetime()] * 2,
        path=path,
    )
    plan = plan.merge(savings, on="predicted_component", how="left").merge(
        since, on="machine_id", how="left"
    )
    plan["action"] = plan.apply(
        lambda r: f"replace {r['predicted_component']}" if r["alert"] else "inspect sensors",
        axis=1,
    )
    plan["saving_if_acted"] = plan["saving_if_prevented"].where(plan["alert"], 0.0)
    plan = plan.sort_values(
        ["alert", "failure_risk", "saving_if_acted", "anomaly_score"],
        ascending=[False, False, False, False],
        ignore_index=True,
    )
    plan.insert(0, "priority", range(1, len(plan) + 1))
    return plan.drop(columns=["saving_if_prevented", "risk_rank"])


# ------------------------------------------------------------------ one machine


def machine_sensors(
    machine_id: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
    path: Path | None = None,
    daily: bool = False,
) -> pd.DataFrame:
    """Sensor readings of one machine (long format: timestamp, sensor, value, quality).

    Hourly by default; with daily=True one daily mean per sensor (quality "daily mean"),
    which keeps long windows light enough to chart."""
    params = [
        int(machine_id),
        pd.Timestamp(start).to_pydatetime(),
        pd.Timestamp(end).to_pydatetime(),
    ]
    if daily:
        sql = """
            SELECT date_trunc('day', timestamp) AS timestamp, sensor, avg(value) AS value,
                   'daily mean' AS quality
            FROM fact_sensor_reading
            WHERE machine_id = ? AND timestamp BETWEEN ? AND ?
            GROUP BY ALL
            ORDER BY sensor, timestamp
        """
    else:
        sql = """
            SELECT timestamp, sensor, value, quality FROM fact_sensor_reading
            WHERE machine_id = ? AND timestamp BETWEEN ? AND ?
            ORDER BY sensor, timestamp
        """
    return query(sql, params, path=path)


def machine_events(
    machine_id: int, start: pd.Timestamp, end: pd.Timestamp, path: Path | None = None
) -> pd.DataFrame:
    """Failures, maintenance and errors of one machine: timestamp, kind, detail."""
    params = [
        int(machine_id),
        pd.Timestamp(start).to_pydatetime(),
        pd.Timestamp(end).to_pydatetime(),
    ]
    return query(
        """
        SELECT timestamp, 'failure' AS kind, component AS detail FROM fact_failure
        WHERE machine_id = ? AND timestamp BETWEEN ? AND ?
        UNION ALL
        SELECT timestamp, 'maintenance', maintenance_type || ': ' || component FROM fact_maintenance
        WHERE machine_id = ? AND timestamp BETWEEN ? AND ?
        UNION ALL
        SELECT timestamp, 'error', error_id FROM fact_error
        WHERE machine_id = ? AND timestamp BETWEEN ? AND ?
        ORDER BY timestamp, kind, detail
        """,
        params * 3,
        path=path,
    )


def machine_predictions(
    machine_id: int, start: pd.Timestamp, end: pd.Timestamp, path: Path | None = None
) -> pd.DataFrame:
    """One machine's out-of-sample predictions between start and end."""
    return query(
        """
        SELECT timestamp, failure_risk, alert, predicted_component, anomaly_score, anomaly_flag,
               reason_1, reason_2, reason_3
        FROM fact_prediction
        WHERE machine_id = ? AND out_of_sample AND timestamp BETWEEN ? AND ?
        ORDER BY timestamp
        """,
        [int(machine_id), pd.Timestamp(start).to_pydatetime(), pd.Timestamp(end).to_pydatetime()],
        path=path,
    )
