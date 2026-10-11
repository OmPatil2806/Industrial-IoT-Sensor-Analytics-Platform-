"""Tests for iiot.dashboard.charts: every chart has tooltips and follows the colour rules."""

from __future__ import annotations

import pandas as pd

from iiot.dashboard import charts

MONTHLY = pd.DataFrame(
    {
        "year_month": ["2015-01", "2015-02"],
        "failures": [5, 3],
        "availability": [0.98, 0.99],
        "failure_cost": [3e6, 2e6],
        "maintenance_cost": [1e6, 1e6],
    }
)


def spec(chart) -> dict:
    return chart.to_dict()


def test_monthly_charts_use_one_measure_one_colour_and_tooltips():
    for chart in (
        charts.monthly_bars(MONTHLY, "failures", "Failures", "failures"),
        charts.monthly_line(MONTHLY, "availability", "Availability", "availability", ".2%"),
    ):
        s = spec(chart)
        assert s["mark"]["color"] == charts.SERIES[0]
        assert "tooltip" in s["encoding"]
        assert "y2" not in s["encoding"] and "resolve" not in s  # no second axis


def test_cost_chart_splits_parts_in_palette_order_with_a_legend():
    s = spec(charts.monthly_cost(MONTHLY))
    color = s["encoding"]["color"]
    assert color["scale"]["range"] == [charts.SERIES[1], charts.SERIES[0]]
    assert color["scale"]["domain"] == ["unplanned", "planned"]
    assert color["legend"] is not None
    data = next(iter(s["datasets"].values()))
    assert sorted(row["lakh"] for row in data) == [10.0, 10.0, 20.0, 30.0]


def test_ranked_bars_have_labels_and_tooltips():
    df = pd.DataFrame({"line": ["A", "B"], "value": [1.5, 2.5]})
    s = spec(charts.ranked_bars(df, "line", "value", "Title", "unit"))
    bars, labels = s["layer"]
    assert bars["mark"]["type"] == "bar" and labels["mark"]["type"] == "text"
    assert bars["encoding"]["y"]["sort"] == "-x"
    assert "tooltip" in bars["encoding"] and "text" in labels["encoding"]
