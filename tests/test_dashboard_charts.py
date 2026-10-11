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
        "total_cost": [4e6, 3e6],
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


EVENTS = pd.DataFrame(
    {
        "timestamp": pd.to_datetime(["2015-01-05 06:00", "2015-01-05 06:00", "2015-01-03 07:00"]),
        "kind": ["failure", "maintenance", "error"],
        "detail": ["comp1", "failure_repair: comp1", "error3"],
    }
)
DOMAIN = ["2015-01-01T00:00:00", "2015-01-10T00:00:00"]


def test_risk_timeline_shows_threshold_and_events_without_a_legend():
    predictions = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2015-01-04 06:00", "2015-01-04 09:00"]),
            "failure_risk": [0.1, 0.95],
            "predicted_component": ["comp1", "comp1"],
            "reason_1": [None, "voltage 24 h mean 190.0 vs normal 170.0 (+12%)"],
        }
    )
    s = spec(charts.risk_timeline(predictions, EVENTS, 0.9, DOMAIN))
    line, rule, events = s["layer"]
    assert line["encoding"]["y"]["scale"]["domain"] == [0, 1]
    assert "tooltip" in line["encoding"] and "tooltip" in events["encoding"]
    assert events["encoding"]["color"]["legend"] is None
    rows = s["datasets"][events["data"]["name"]]
    assert sorted(r["kind"] for r in rows) == ["failure", "maintenance"]  # errors are not drawn


def test_sensor_panels_one_chart_per_sensor_with_its_own_scale():
    sensors = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2015-01-04"] * 4),
            "sensor": ["volt", "rotate", "pressure", "vibration"],
            "value": [170.0, 450.0, 100.0, 40.0],
            "quality": ["ok"] * 4,
        }
    )
    s = spec(charts.sensor_panels(sensors, EVENTS, DOMAIN))
    titles = [panel["layer"][0]["title"] for panel in s["vconcat"]]
    assert titles == ["Voltage", "Rotation", "Pressure", "Vibration"]
    assert all(p["layer"][0]["encoding"]["y"]["scale"]["zero"] is False for p in s["vconcat"])


def test_saving_curve_marks_the_chosen_rate_and_the_break_even():
    df = pd.DataFrame({"catch_rate": [0.0, 0.5, 1.0], "net_saving_cr": [-0.5, 10.0, 20.5]})
    spec = charts.saving_curve(df, chosen=0.5, break_even=0.0238).to_dict()
    line, point, rule = spec["layer"]
    assert line["mark"]["type"] == "line" and line["mark"]["color"] == charts.SERIES[0]
    assert point["mark"]["type"] == "point" and rule["mark"]["type"] == "rule"
    rows = next(iter(spec["datasets"].values()))
    assert [r["mark"] for r in rows] == ["", "break-even", "chosen", ""]
    assert all(layer["encoding"]["tooltip"] for layer in spec["layer"])
    assert len(charts.saving_curve(df, 0.5, None).to_dict()["datasets"]) == 1
