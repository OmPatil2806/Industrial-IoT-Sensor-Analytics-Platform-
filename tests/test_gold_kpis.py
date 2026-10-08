"""Tests for iiot.gold.kpis (reliability, availability and cost KPIs)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from iiot.gold.kpis import (
    aggregate,
    build,
    build_kpis,
    classify_events,
    period_hours_by_month,
)

START = pd.Timestamp("2015-01-01 06:00")
END = pd.Timestamp("2015-03-01 00:00")

COSTS = pd.DataFrame(
    {
        "component": ["comp1", "comp2"],
        "unplanned_downtime_hours": [8.0, 12.0],
        "planned_downtime_hours": [2.0, 3.0],
        "unplanned_failure_cost": [100.0, 200.0],
        "planned_maintenance_cost": [10.0, 20.0],
    }
)
MACHINES = pd.DataFrame(
    {
        "machine_id": pd.array([1, 2], dtype="Int64"),
        "model": ["model1", "model2"],
        "plant_id": ["PUNE", "PUNE"],
        "line_id": ["PUNE-L1", "PUNE-L2"],
    }
)


def records(rows: list[tuple[int, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "machine_id": pd.array([r[0] for r in rows], dtype="Int64"),
            "timestamp": pd.to_datetime([r[1] for r in rows]).astype("datetime64[ns]"),
            "component": [r[2] for r in rows],
        }
    )


def test_period_hours_by_month():
    hours = period_hours_by_month(START, END)
    assert hours.tolist() == [31 * 24 - 6, 28 * 24]  # January starts at 06:00


def test_repairs_of_failures_are_not_counted_as_planned():
    failures = records([(1, "2015-01-10 06:00", "comp1")])
    maintenance = records([(1, "2015-01-10 06:00", "comp1"), (1, "2015-01-20 06:00", "comp2")])
    fail_ev, planned_ev, stats = classify_events(failures, maintenance, COSTS, START, END)
    assert stats.repair_records_excluded == 1
    assert len(fail_ev) == 1 and len(planned_ev) == 1
    assert planned_ev.iloc[0]["downtime_h"] == 3.0 and planned_ev.iloc[0]["cost"] == 20.0


def test_simultaneous_failures_are_one_event():
    failures = records([(1, "2015-01-10 06:00", "comp1"), (1, "2015-01-10 06:00", "comp2")])
    fail_ev, _, stats = classify_events(failures, records([]), COSTS, START, END)
    assert stats.failure_events == 1 and stats.component_failures == 2
    event = fail_ev.iloc[0]
    assert event["components"] == 2
    assert event["downtime_h"] == 12.0  # repaired in parallel: the longest repair
    assert event["cost"] == 300.0  # both components paid for


def test_events_outside_the_period_are_excluded():
    maintenance = records([(1, "2014-06-01 06:00", "comp1"), (1, "2015-01-05 06:00", "comp1")])
    _, planned_ev, stats = classify_events(records([]), maintenance, COSTS, START, END)
    assert len(planned_ev) == 1 and stats.outside_period_excluded == 1


def test_unknown_component_raises():
    with pytest.raises(ValueError, match="comp9"):
        classify_events(records([(1, "2015-01-10", "comp9")]), records([]), COSTS, START, END)


def test_machine_monthly_kpis():
    failures = records([(1, "2015-01-10 06:00", "comp1"), (1, "2015-01-20 06:00", "comp2")])
    maintenance = records([(1, "2015-02-05 06:00", "comp1")])
    tables, _ = build_kpis(failures, maintenance, MACHINES, COSTS, START, END)
    monthly = tables["kpi_machine_monthly"]
    assert len(monthly) == 4  # 2 machines x 2 months, including months without events
    jan = monthly[(monthly["machine_id"] == 1) & (monthly["month"] == "2015-01")].iloc[0]
    hours = 31 * 24 - 6
    assert jan["failures"] == 2 and jan["unplanned_downtime_h"] == 20.0
    assert jan["availability"] == pytest.approx(1 - 20 / hours)
    assert jan["mtbf_h"] == pytest.approx((hours - 20) / 2)
    assert jan["mttr_h"] == pytest.approx(10.0)
    assert jan["failure_cost"] == 300.0 and jan["total_cost"] == 300.0
    feb = monthly[(monthly["machine_id"] == 1) & (monthly["month"] == "2015-02")].iloc[0]
    assert feb["planned_maintenances"] == 1 and feb["planned_downtime_h"] == 2.0
    assert np.isnan(feb["mtbf_h"]) and np.isnan(feb["mttr_h"])  # no failures


def test_aggregates_recompute_ratios_from_sums():
    failures = records([(1, "2015-01-10 06:00", "comp2")])
    tables, _ = build_kpis(failures, records([]), MACHINES, COSTS, START, END)
    line_or_plant = tables["kpi_plant_monthly"]
    jan = line_or_plant[line_or_plant["month"] == "2015-01"].iloc[0]
    hours = 2 * (31 * 24 - 6)  # two machines
    assert jan["period_hours"] == hours
    assert jan["availability"] == pytest.approx(1 - 12 / hours)  # not an average of ratios


def test_totals_are_consistent_across_tables():
    failures = records([(1, "2015-01-10 06:00", "comp1"), (2, "2015-02-10 06:00", "comp2")])
    tables, stats = build_kpis(failures, records([]), MACHINES, COSTS, START, END)
    for name in (
        "kpi_machine_monthly",
        "kpi_machine_total",
        "kpi_line_monthly",
        "kpi_plant_monthly",
    ):
        assert tables[name]["failures"].sum() == 2, name
        assert tables[name]["total_cost"].sum() == 300.0, name
    assert stats.rows == {
        "kpi_machine_monthly": 4,
        "kpi_machine_total": 2,
        "kpi_line_monthly": 4,
        "kpi_plant_monthly": 2,
    }


def test_aggregate_handles_zero_failures():
    monthly = build_kpis(records([]), records([]), MACHINES, COSTS, START, END)[0][
        "kpi_machine_monthly"
    ]
    total = aggregate(monthly.assign(all="x"), ["all"]).iloc[0]
    assert total["availability"] == 1.0 and np.isnan(total["mtbf_h"])


def test_build_without_silver_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="iiot silver build"):
        build(tmp_path / "silver", tmp_path / "gold")
