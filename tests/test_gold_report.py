"""Tests for iiot.gold.report: each check passes on good data and FAILS on broken data."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from iiot.gold import report as gold_report
from iiot.gold.report import (
    check_balance,
    check_completeness,
    check_kpis,
    check_labels,
    check_no_leakage,
    check_split,
    show_report,
)

QUALITY = ["ok", "filled", "missing"]


def silver_telemetry(n_machines: int = 3, hours: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    parts = []
    for m in range(1, n_machines + 1):
        part = pd.DataFrame(
            {
                "machine_id": pd.array([m] * hours, dtype="Int64"),
                "timestamp": pd.date_range("2015-01-01 06:00", periods=hours, freq="h"),
            }
        )
        for s, mean in [("volt", 170), ("rotate", 446), ("pressure", 100), ("vibration", 40)]:
            part[s] = rng.normal(mean, 5, hours)
            part[f"{s}_quality"] = pd.Categorical(["ok"] * hours, QUALITY)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def event_table(value_col: str, rows: list[tuple[int, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "machine_id": pd.array([r[0] for r in rows], dtype="Int64"),
            "timestamp": pd.to_datetime([r[1] for r in rows]).astype("datetime64[us]"),
            value_col: [r[2] for r in rows],
        }
    )


MACHINES = pd.DataFrame(
    {
        "machine_id": pd.array([1, 2, 3], dtype="Int64"),
        "model": ["model1"] * 3,
        "age": pd.array([5, 6, 7], dtype="Int64"),
        "plant_id": ["P"] * 3,
        "line_id": ["P-L1"] * 3,
    }
)
ERRORS = event_table("error_id", [(1, "2015-01-02 10:00", "error1")])
MAINT = event_table("component", [(1, "2014-12-01 06:00", "comp1")])


# --- no_leakage ------------------------------------------------------------------


def test_no_leakage_passes_for_real_features():
    result = check_no_leakage(silver_telemetry(), ERRORS, MAINT, MACHINES)
    assert result["passed"]
    assert result["sensor_features_unchanged"] and result["event_features_unchanged"]


def test_no_leakage_fails_for_a_leaky_feature(monkeypatch):
    real = gold_report.sensor_features.build_sensor_features

    def leaky(silver, sensors, gold):
        out, stats = real(silver, sensors, gold)
        # "peeks" at the next reading: a classic leakage bug
        nxt = (
            silver.sort_values(["machine_id", "timestamp"]).groupby("machine_id")["volt"].shift(-1)
        )
        lookup = silver.assign(nxt=nxt.to_numpy()).set_index(["machine_id", "timestamp"])["nxt"]
        out["volt_mean_3h"] = lookup.reindex(
            pd.MultiIndex.from_frame(out[["machine_id", "timestamp"]])
        ).to_numpy()
        return out, stats

    monkeypatch.setattr(gold_report.sensor_features, "build_sensor_features", leaky)
    result = check_no_leakage(silver_telemetry(), ERRORS, MAINT, MACHINES)
    assert not result["passed"] and not result["sensor_features_unchanged"]


# --- labels_correct --------------------------------------------------------------


def label_table(fail_flags: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "machine_id": pd.array([1] * len(fail_flags), dtype="Int64"),
            "timestamp": pd.date_range("2015-01-01", periods=len(fail_flags), freq="3h"),
            "fails_within_24h": np.array(fail_flags, dtype=np.int8),
        }
    )


FAILURES = event_table("component", [(1, "2015-01-01 12:00", "comp1")])


def test_labels_correct_passes():
    # rows at 0,3,6,9 h see the 12:00 failure; the row at 12:00 does not
    result = check_labels(label_table([1, 1, 1, 1, 0, 0]), FAILURES, 24)
    assert result["passed"] and result["mismatches"] == 0


def test_labels_correct_fails_on_a_wrong_label():
    result = check_labels(label_table([1, 1, 1, 1, 1, 0]), FAILURES, 24)
    assert not result["passed"] and result["mismatches"] == 1


# --- split, balance, completeness -----------------------------------------------


def dataset(rows: list[tuple[str, str, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime([r[0] for r in rows]),
            "split": pd.Categorical([r[1] for r in rows]),
            "fails_within_24h": [r[2] for r in rows],
        }
    )


def test_split_passes_and_fails():
    split = pd.Timestamp("2015-10-01")
    good = dataset([("2015-09-29 21:00", "train", 0), ("2015-10-01 00:00", "test", 0)])
    assert check_split(good, split, 24)["passed"]
    leaky = dataset([("2015-09-30 12:00", "train", 0), ("2015-10-01 00:00", "test", 0)])
    assert not check_split(leaky, split, 24)["passed"]  # its label window reaches October


def test_balance_needs_positives_in_both_splits():
    good = dataset(
        [("2015-01-01", "train", 1), ("2015-11-01", "test", 1), ("2015-11-02", "test", 0)]
    )
    result = check_balance(good, 24)
    assert result["passed"] and result["test"]["positive_rate"] == 0.5
    no_test_positive = dataset([("2015-01-01", "train", 1), ("2015-11-01", "test", 0)])
    assert not check_balance(no_test_positive, 24)["passed"]


def test_completeness_flags_mostly_empty_features():
    df = pd.DataFrame({"a": [1.0] * 100, "b": [np.nan] * 10 + [1.0] * 90})
    result = check_completeness(df, ["a", "b"])
    assert not result["passed"] and result["too_empty"] == {"b": 0.1}
    assert check_completeness(df, ["a"])["passed"]


# --- kpi_reconciliation ---------------------------------------------------------


def kpi_tables(failures: list[int]) -> dict[str, pd.DataFrame]:
    monthly = pd.DataFrame(
        {
            "machine_id": [1, 1],
            "failures": failures,
            "components_failed": failures,
            "total_cost": [10.0, 20.0],
            "downtime_h": [1.0, 2.0],
        }
    )
    total = pd.DataFrame(
        {"machine_id": [1], "failures": [sum(failures)], "total_cost": [30.0], "downtime_h": [3.0]}
    )
    return {"kpi_machine_monthly": monthly, "kpi_machine_total": total}


def test_kpi_reconciliation_passes_and_fails():
    failures = event_table(
        "component", [(1, "2015-01-05 06:00", "comp1"), (1, "2015-02-05 06:00", "comp2")]
    )
    period = (pd.Timestamp("2015-01-01"), pd.Timestamp("2015-03-01"))
    assert check_kpis(kpi_tables([1, 1]), failures, period)["passed"]
    assert not check_kpis(kpi_tables([1, 0]), failures, period)["passed"]


# --- report file -----------------------------------------------------------------


def test_show_report(tmp_path):
    (tmp_path / "gold_report.json").write_text(
        json.dumps({"generated_at": "x", "passed": True, "checks": {"a": {"passed": True}}}),
        encoding="utf-8",
    )
    assert show_report(tmp_path)["passed"]


def test_show_report_without_build_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="iiot gold build"):
        show_report(tmp_path)
