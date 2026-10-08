"""Build the whole Gold layer and check it: data/gold/gold_report.json.

`build_gold()` builds sensor features, event features, labels + ML dataset and
KPIs, then runs these checks on the real data:

    no_leakage          for a sample of machines, every reading and event AFTER a
                        cut-off time is changed / added; all features up to the
                        cut-off must stay exactly the same
    labels_correct      fails_within_<H>h recomputed independently (from each
                        failure backwards) must match the label column on every row
    split_integrity     every train label window ends before the split date, every
                        test row is on or after it
    label_balance       train and test both contain positives (rate reported)
    feature_completeness  no feature is empty in more than MAX_EMPTY_SHARE of rows
    kpi_reconciliation  KPI failures and component failures equal the Silver
                        failures in the KPI period; machine totals equal the sum of
                        their months

`passed` is true only if every check passes.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from iiot.config import get_settings
from iiot.gold import event_features, kpis, labels, sensor_features
from iiot.utils.logger import get_logger

logger = get_logger("iiot.gold.report")

REPORT_FILE_NAME = "gold_report.json"
MAX_EMPTY_SHARE = 0.05
LEAKAGE_SAMPLE_MACHINES = 3
KEYS = ["machine_id", "timestamp"]


def check_no_leakage(
    silver_telemetry: pd.DataFrame,
    errors: pd.DataFrame,
    maintenance: pd.DataFrame,
    machines: pd.DataFrame,
    sample_machines: int = LEAKAGE_SAMPLE_MACHINES,
) -> dict:
    """Change the future and verify the features of the past do not move."""
    settings = get_settings()
    sensors = list(settings.sensors)
    ids = sorted(silver_telemetry["machine_id"].dropna().unique())[:sample_machines]
    tel = silver_telemetry[silver_telemetry["machine_id"].isin(ids)]
    before, _ = sensor_features.build_sensor_features(tel, sensors, settings.gold)
    # The cut-off is ON a feature row, so even a 1-hour look-ahead from that row
    # reaches changed data and is detected.
    row_times = before["timestamp"].sort_values().reset_index(drop=True)
    cutoff = row_times.iloc[len(row_times) // 2]

    future = tel["timestamp"] > cutoff
    changed_tel = tel.copy()
    changed_tel.loc[future, sensors] = changed_tel.loc[future, sensors] + 1000.0
    after, _ = sensor_features.build_sensor_features(changed_tel, sensors, settings.gold)
    past = before["timestamp"] <= cutoff
    sensor_ok = (
        before[past]
        .reset_index(drop=True)
        .equals(after[after["timestamp"] <= cutoff].reset_index(drop=True))
    )

    window = max(settings.gold.window_hours)
    components = list(settings.business.components)
    spine = before[KEYS]
    late = cutoff + pd.Timedelta(hours=1)
    extra_errors = pd.DataFrame(
        {"machine_id": pd.array(ids, dtype="Int64"), "timestamp": late, "error_id": "error1"}
    ).astype({"timestamp": errors["timestamp"].dtype})
    extra_maint = pd.DataFrame(
        {"machine_id": pd.array(ids, dtype="Int64"), "timestamp": late, "component": components[0]}
    ).astype({"timestamp": maintenance["timestamp"].dtype})
    ev_before, _ = event_features.build_event_features(
        spine, errors, maintenance, machines, window, components
    )
    ev_after, _ = event_features.build_event_features(
        spine,
        pd.concat([errors, extra_errors], ignore_index=True),
        pd.concat([maintenance, extra_maint], ignore_index=True),
        machines,
        window,
        components,
    )
    ev_past = ev_before["timestamp"] <= cutoff
    event_ok = ev_before[ev_past].equals(ev_after[ev_past])
    return {
        "passed": bool(sensor_ok and event_ok),
        "machines": [int(m) for m in ids],
        "cutoff": str(cutoff),
        "rows_compared": int(past.sum()),
        "sensor_features_unchanged": bool(sensor_ok),
        "event_features_unchanged": bool(event_ok),
    }


def check_labels(label_table: pd.DataFrame, failures: pd.DataFrame, horizon_hours: int) -> dict:
    """Recompute the target from each failure backwards and compare on every row."""
    target = f"fails_within_{horizon_hours}h"
    rows = label_table[KEYS].reset_index()
    rows["timestamp"] = rows["timestamp"].astype("datetime64[ns]")
    events = failures[KEYS].drop_duplicates().rename(columns={"timestamp": "failure_time"})
    events["failure_time"] = events["failure_time"].astype("datetime64[ns]")
    pairs = rows.merge(events, on="machine_id")
    hit = (pairs["failure_time"] > pairs["timestamp"]) & (
        pairs["failure_time"] <= pairs["timestamp"] + pd.Timedelta(hours=horizon_hours)
    )
    positive_rows = set(pairs.loc[hit, "index"])
    expected = label_table.index.isin(list(positive_rows)).astype(int)
    mismatches = int((expected != label_table[target].to_numpy()).sum())
    return {
        "passed": mismatches == 0,
        "rows_checked": len(label_table),
        "positives_expected": int(expected.sum()),
        "positives_labelled": int(label_table[target].sum()),
        "mismatches": mismatches,
    }


def check_split(dataset: pd.DataFrame, train_end: pd.Timestamp, horizon_hours: int) -> dict:
    split = dataset["split"].astype(str)
    train_ts, test_ts = (
        dataset.loc[split == "train", "timestamp"],
        dataset.loc[split == "test", "timestamp"],
    )
    train_ok = bool((train_ts + pd.Timedelta(hours=horizon_hours) < train_end).all())
    test_ok = bool((test_ts >= train_end).all())
    return {
        "passed": train_ok and test_ok and len(train_ts) > 0 and len(test_ts) > 0,
        "split_date": str(train_end.date()),
        "last_train_row": str(train_ts.max()),
        "first_test_row": str(test_ts.min()),
        "train_rows": len(train_ts),
        "test_rows": len(test_ts),
    }


def check_balance(dataset: pd.DataFrame, horizon_hours: int) -> dict:
    target = f"fails_within_{horizon_hours}h"
    result: dict = {}
    for split in ("train", "test"):
        part = dataset[dataset["split"].astype(str) == split]
        result[split] = {
            "rows": len(part),
            "positives": int(part[target].sum()),
            "positive_rate": round(float(part[target].mean()), 4) if len(part) else 0.0,
        }
    result["passed"] = result["train"]["positives"] > 0 and result["test"]["positives"] > 0
    return result


def check_completeness(dataset: pd.DataFrame, feature_columns: list[str]) -> dict:
    shares = dataset[feature_columns].isna().mean().sort_values(ascending=False)
    too_empty = {c: round(float(s), 4) for c, s in shares.items() if s > MAX_EMPTY_SHARE}
    return {
        "passed": not too_empty,
        "max_allowed_empty_share": MAX_EMPTY_SHARE,
        "features_checked": len(feature_columns),
        "most_empty": {c: round(float(s), 5) for c, s in shares.head(3).items()},
        "too_empty": too_empty,
    }


def check_kpis(tables: dict[str, pd.DataFrame], failures: pd.DataFrame, period: tuple) -> dict:
    start, end = period
    in_period = failures[(failures["timestamp"] >= start) & (failures["timestamp"] < end)]
    monthly, total = tables["kpi_machine_monthly"], tables["kpi_machine_total"]
    expected_events = in_period.groupby(["machine_id", "timestamp"]).ngroups
    by_machine = monthly.groupby("machine_id")[["failures", "total_cost", "downtime_h"]].sum()
    totals = total.set_index("machine_id")[["failures", "total_cost", "downtime_h"]]
    machines_consistent = bool(np.allclose(by_machine.sort_index(), totals.sort_index()))
    return {
        "passed": int(monthly["failures"].sum()) == expected_events
        and int(monthly["components_failed"].sum()) == len(in_period)
        and machines_consistent,
        "failure_events": int(monthly["failures"].sum()),
        "failure_events_in_silver": expected_events,
        "component_failures": int(monthly["components_failed"].sum()),
        "component_failures_in_silver": len(in_period),
        "machine_totals_match_months": machines_consistent,
    }


def build_gold(silver_dir: Path | None = None, gold_dir: Path | None = None) -> dict:
    """Build every Gold table, run all checks, write and return the report."""
    settings = get_settings()
    silver_dir = Path(silver_dir or settings.paths.silver)
    gold_dir = Path(gold_dir or settings.paths.gold)
    horizon = settings.ml.prediction_horizon_hours

    _, sensor_stats = sensor_features.build(silver_dir, gold_dir)
    _, event_stats = event_features.build(silver_dir, gold_dir)
    dataset, label_stats, dataset_stats = labels.build(silver_dir, gold_dir)
    kpi_tables, kpi_stats = kpis.build(silver_dir, gold_dir)

    silver = {
        n: pd.read_parquet(silver_dir / f"{n}.parquet")
        for n in ("telemetry", "errors", "maintenance", "failures", "machines")
    }
    label_table = pd.read_parquet(gold_dir / "labels.parquet")
    label_columns = set(label_table.columns) - set(KEYS)
    feature_columns = [c for c in dataset.columns if c not in label_columns | set(KEYS) | {"split"}]
    period_start = silver["telemetry"]["timestamp"].min()
    period_end = silver["telemetry"]["timestamp"].max().normalize()

    checks = {
        "no_leakage": check_no_leakage(
            silver["telemetry"], silver["errors"], silver["maintenance"], silver["machines"]
        ),
        "labels_correct": check_labels(label_table, silver["failures"], horizon),
        "split_integrity": check_split(dataset, pd.Timestamp(settings.ml.train_end_date), horizon),
        "label_balance": check_balance(dataset, horizon),
        "feature_completeness": check_completeness(dataset, feature_columns),
        "kpi_reconciliation": check_kpis(
            kpi_tables, silver["failures"], (period_start, period_end)
        ),
    }
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "passed": all(c["passed"] for c in checks.values()),
        "tables": {
            "sensor_features": {"rows": sensor_stats.rows_out, "features": sensor_stats.features},
            "event_features": {"rows": event_stats.rows, "features": event_stats.features},
            "labels": {"rows": label_stats.rows_out, "positives": label_stats.positives},
            "ml_dataset": {"rows": dataset_stats.rows, "features": dataset_stats.features},
            **{name: {"rows": n} for name, n in kpi_stats.rows.items()},
        },
        "checks": checks,
    }
    out = gold_dir / REPORT_FILE_NAME
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _log_summary(report, out)
    return report


def show_report(gold_dir: Path | None = None) -> dict:
    """Log the summary of the latest gold_report.json and return the report."""
    path = Path(gold_dir or get_settings().paths.gold) / REPORT_FILE_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `iiot gold build` first.")
    report = json.loads(path.read_text(encoding="utf-8"))
    logger.info("Report generated at %s", report["generated_at"])
    _log_summary(report, path)
    return report


def _log_summary(report: dict, out: Path) -> None:
    for name, check in report["checks"].items():
        (logger.info if check["passed"] else logger.error)(
            "%-22s %s", name, "PASS" if check["passed"] else "FAIL"
        )
    (logger.info if report["passed"] else logger.error)(
        "Gold report: %s -> %s", "PASSED" if report["passed"] else "FAILED", out
    )
