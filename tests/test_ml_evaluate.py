"""Tests for iiot.ml.evaluate: metrics checked against numbers worked out by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from iiot.ml.evaluate import (
    Costs,
    choose_threshold,
    evaluate,
    inspections,
    load_costs,
    threshold_candidates,
)

TARGET = "fails_within_24h"
STEP = pd.Timedelta(hours=3)
DAY = pd.Timedelta(hours=24)
COSTS = Costs(savings={"comp1": 300.0, "comp2": 100.0}, false_alarm_cost=50.0)
FAILURES = {  # machine 1, 10 days of rows every 3 h
    pd.Timestamp("2015-01-05"): "comp1",
    pd.Timestamp("2015-01-09"): "comp1+comp2",
}


def rows() -> pd.DataFrame:
    """Labels like Gold: a row is positive if a failure follows within 24 h."""
    t = pd.date_range("2015-01-01", periods=80, freq="3h")
    df = pd.DataFrame({"machine_id": 1, "timestamp": t, TARGET: 0, "failed_component": "none"})
    df["hours_to_failure"] = np.nan
    for failed_at, component in FAILURES.items():
        window = (t < failed_at) & (t >= failed_at - DAY)
        df.loc[window, TARGET] = 1
        df.loc[window, "failed_component"] = component
        df.loc[window, "hours_to_failure"] = (failed_at - t[window]).total_seconds() / 3600
    return df


def scores_for(df: pd.DataFrame, alert_times: list[str]) -> np.ndarray:
    return df["timestamp"].isin(pd.to_datetime(alert_times)).to_numpy().astype(float)


def run(df, scores, threshold=1.0):
    return evaluate(df, scores, threshold, COSTS, TARGET, STEP)


def test_metrics_worked_out_by_hand():
    df = rows()
    # One alert 12 h before the first failure; false alerts on day 2 (twice, within one
    # inspection) and on day 7. The second failure is missed.
    alerts = ["2015-01-04 12:00", "2015-01-02 00:00", "2015-01-02 03:00", "2015-01-07 06:00"]
    m = run(df, scores_for(df, alerts))
    assert (m["positives"], m["alerts"]) == (16, 4)
    assert m["precision"] == pytest.approx(1 / 4)
    assert m["recall"] == pytest.approx(1 / 16)
    assert (m["failure_events"], m["events_caught"], m["event_recall"]) == (2, 1, 0.5)
    assert m["median_lead_time_h"] == 12
    assert (m["inspections"], m["false_alarms"]) == (3, 2)
    assert m["caught_saving"] == 300  # comp1
    assert m["false_alarm_cost"] == 100
    assert m["net_saving"] == 200
    assert m["potential_saving"] == 700  # comp1 + (comp1 + comp2)
    assert m["saving_captured"] == pytest.approx(200 / 700)
    machine_months = 80 * 3 / (365.25 * 24 / 12)
    assert m["false_alarms_per_machine_month"] == pytest.approx(2 / machine_months)


def test_perfect_and_empty_models():
    df = rows()
    perfect = run(df, df[TARGET].to_numpy().astype(float))
    assert perfect["pr_auc"] == 1.0 and perfect["roc_auc"] == 1.0
    assert perfect["net_saving"] == perfect["potential_saving"] == 700
    assert perfect["median_lead_time_h"] == 24  # first alert on the first positive row
    silent = run(df, np.zeros(len(df)))
    assert (silent["alerts"], silent["events_caught"], silent["net_saving"]) == (0, 0, 0)
    assert silent["pr_auc"] == pytest.approx(16 / 80)  # no skill = share of positives


def test_alerting_all_the_time_pays_for_an_inspection_every_24h():
    df = rows()
    m = run(df, np.ones(len(df)))
    assert m["events_caught"] == 2
    assert m["inspections"] == 10  # 10 days, one inspection per day
    assert m["false_alarms"] == 8  # only the inspections on days 4 and 8 find a failure
    assert m["net_saving"] == 700 - 8 * 50


def test_inspection_covers_the_next_24_hours():
    df = rows()
    t = df["timestamp"]
    covered = (t >= "2015-01-01") & (t < "2015-01-02")  # 8 alerts within 24 h
    assert len(inspections(df, covered.to_numpy(), TARGET, DAY)) == 1
    plus_next_day = covered | (t == pd.Timestamp("2015-01-02"))  # exactly 24 h later
    assert len(inspections(df, plus_next_day.to_numpy(), TARGET, DAY)) == 2


def test_inspections_are_counted_per_machine():
    df = pd.concat([rows(), rows().assign(machine_id=2)], ignore_index=True)
    alerts = (df["timestamp"] == pd.Timestamp("2015-01-02")).to_numpy()
    assert len(inspections(df, alerts, TARGET, DAY)) == 2


def test_choose_threshold_maximises_net_saving_and_prefers_fewer_alerts():
    df = rows()
    scores = np.where(df[TARGET] == 1, 0.9, 0.1)
    scores[df["timestamp"] == pd.Timestamp("2015-01-02")] = 0.95  # one costly false alarm
    threshold, m = choose_threshold(df, scores, COSTS, TARGET, STEP)
    assert threshold == 0.9  # 0.95 alone catches nothing; 0.1 alerts everywhere
    assert (m["events_caught"], m["false_alarms"], m["net_saving"]) == (2, 1, 650)


def test_choose_threshold_never_alerts_when_every_alert_loses_money():
    df = rows()
    scores = np.where(df[TARGET] == 1, 0.1, 0.9)  # a model that is always wrong
    expensive = Costs(savings=COSTS.savings, false_alarm_cost=10_000)
    threshold, m = choose_threshold(df, scores, expensive, TARGET, STEP)
    assert threshold > scores.max()
    assert (m["alerts"], m["net_saving"]) == (0, 0)


def test_fast_threshold_search_agrees_with_full_evaluation():
    df = pd.concat([rows(), rows().assign(machine_id=2)], ignore_index=True)
    scores = np.random.default_rng(1).random(len(df)) + 0.5 * df[TARGET].to_numpy()
    _, chosen = choose_threshold(df, scores, COSTS, TARGET, STEP)
    best = max(run(df, scores, t)["net_saving"] for t in threshold_candidates(scores))
    assert chosen["net_saving"] == best


def test_threshold_candidates_are_limited_and_include_never_alert():
    scores = np.random.default_rng(0).random(10_000)
    candidates = threshold_candidates(scores)
    assert len(candidates) <= 201
    assert candidates[-1] > scores.max()


def test_wrong_number_of_scores_is_rejected():
    with pytest.raises(ValueError, match="scores"):
        run(rows(), np.zeros(3))


def test_load_costs_reads_silver(tmp_path):
    pd.DataFrame({"component": ["comp1"], "saving_if_prevented": [320000]}).to_parquet(
        tmp_path / "component_costs.parquet"
    )
    costs = load_costs(tmp_path)
    assert costs.savings == {"comp1": 320000.0}
    assert costs.false_alarm_cost == 20000


def test_load_costs_without_silver_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="iiot silver build"):
        load_costs(tmp_path)
