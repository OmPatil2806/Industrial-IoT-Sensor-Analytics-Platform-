"""Tests for iiot.insights: the business-case arithmetic (worked by hand) and the build."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from iiot.insights import calculations as calc
from iiot.insights import report
from iiot.insights.build import AS_TESTED, INSIGHTS_FILE_NAME, build_insights, show_insights
from iiot.model.warehouse import build_warehouse
from tests.test_dashboard_data import write_predictions

COMPONENTS = pd.DataFrame(
    {
        "component": ["comp1", "comp2"],
        "unplanned_downtime_hours": [8.0, 10.0],
        "planned_downtime_hours": [2.0, 3.0],
        "saving_if_prevented": [300_000.0, 400_000.0],
    }
)
PLANNED_HOURS = COMPONENTS.set_index("component")["planned_downtime_hours"]


def year_of_events() -> calc.FailureEvents:
    """Three events in one year on 2 machines: (comp1+comp2), comp1, comp2."""
    failures = pd.DataFrame(
        {
            "machine_id": [1, 1, 1, 2],
            "timestamp": pd.to_datetime(["2015-01-05"] * 2 + ["2015-03-01", "2015-05-01"]),
            "component": ["comp1", "comp2", "comp1", "comp2"],
        }
    )
    return calc.failure_events(failures, COMPONENTS, 2 * calc.HOURS_PER_YEAR, machines=2)


def test_failure_events_group_components_of_one_failure():
    year = year_of_events()
    assert year.events == 3
    assert year.component_failures.to_dict() == {"comp1": 2, "comp2": 2}
    # savings add up per event: 700k + 300k + 400k
    assert year.saving_if_caught == 1_400_000
    # downtime: longest repair - longest planned job: (10-3) + (8-2) + (10-3)
    assert year.downtime_avoided_h == 20
    assert year.years == pytest.approx(1.0)


def test_scenario_by_hand():
    r = calc.scenario(
        "half",
        catch_rate=0.5,
        false_alarms_per_catch=2.0,
        year=year_of_events(),
        false_alarm_cost=10_000,
        program_cost=100_000,
        inspection_hours=2,
        planned_hours=PLANNED_HOURS,
    )
    assert r.failures_caught == pytest.approx(1.5)
    assert r.false_alarms == pytest.approx(3.0)
    assert r.gross_saving == pytest.approx(700_000)
    assert r.false_alarm_cost == pytest.approx(30_000)
    assert r.net_saving == pytest.approx(700_000 - 30_000 - 100_000)
    assert r.downtime_avoided_h == pytest.approx(10)
    assert r.availability_gain == pytest.approx(100 * 10 / (2 * 8760))
    assert r.payback_months == pytest.approx(100_000 / (670_000 / 12))
    assert r.inspections_per_week == pytest.approx(4.5 / 52)
    # 4.5 inspections x 2 h + planned replacements: 1 comp1 x 2 h + 1 comp2 x 3 h
    assert r.technician_hours_per_week == pytest.approx((9 + 2 + 3) / 52)
    assert r.parts_per_quarter == pytest.approx({"comp1": 0.25, "comp2": 0.25})
    assert r.to_dict()["name"] == "half"


def test_scenario_scales_to_a_year():
    half_year = calc.failure_events(
        pd.DataFrame(
            {"machine_id": [1], "timestamp": [pd.Timestamp("2015-01-05")], "component": ["comp1"]}
        ),
        COMPONENTS,
        machine_hours=calc.HOURS_PER_YEAR / 2,
        machines=1,
    )
    r = calc.scenario("all", 1.0, 0.0, half_year, 10_000, 0, 2, PLANNED_HOURS)
    assert r.failures_caught == pytest.approx(2)
    assert r.gross_saving == pytest.approx(600_000)


def test_payback_never_when_alarms_cost_more_than_savings():
    r = calc.scenario("bad", 0.5, 100.0, year_of_events(), 10_000, 100_000, 2, PLANNED_HOURS)
    assert r.net_saving < 0
    assert r.payback_months is None


def test_break_even_catch_rate():
    year = year_of_events()
    # per event: 1.4M / 3 - 2 x 10k; times 3 events = 1.34M a year
    rate = calc.break_even_catch_rate(year, 2.0, 10_000, program_cost=134_000)
    assert rate == pytest.approx(0.1)
    at_break_even = calc.scenario("b", rate, 2.0, year, 10_000, 134_000, 2, PLANNED_HOURS)
    assert at_break_even.net_saving == pytest.approx(0, abs=1e-6)
    assert calc.break_even_catch_rate(year, 2.0, 10_000, program_cost=2_000_000) is None
    assert calc.break_even_catch_rate(year, 100.0, 10_000, program_cost=0) is None


def test_sensitivity_matches_scenarios():
    year = year_of_events()
    grid = calc.sensitivity(year, [0.5, 1.0], [10_000, 50_000], 2.0, 100_000)
    assert grid.shape == (2, 2)
    for rate in (0.5, 1.0):
        for cost in (10_000, 50_000):
            r = calc.scenario("s", rate, 2.0, year, cost, 100_000, 2, PLANNED_HOURS)
            assert grid.loc[rate, cost] == pytest.approx(r.net_saving)


@pytest.mark.parametrize(
    ("age", "band"), [(0, "0-4 years"), (4, "0-4 years"), (9, "5-9 years"), (20, "15+ years")]
)
def test_age_band(age, band):
    assert calc.age_band(age) == band


HEALTH = pd.DataFrame(
    {
        "machine_id": [1, 2, 3],
        "model": ["model1", "model1", "model2"],
        "age": [18, 3, 7],
        "line_id": ["A", "A", "B"],
        "failures": [4, 0, 2],
        "failure_cost": [400.0, 0.0, 100.0],
        "total_cost": [450.0, 10.0, 120.0],
    }
)


def test_focus_groups():
    models = calc.focus_groups(HEALTH, "model")
    assert models.to_dict("records")[0] == {
        "model": "model1",
        "machines": 2,
        "failures": 4,
        "total_cost": 460.0,
        "failures_per_machine": 2.0,
        "cost_per_machine": 230.0,
    }
    ages = calc.focus_groups(HEALTH, "age_band")
    assert list(ages["age_band"]) == ["0-4 years", "5-9 years", "15+ years"]  # empty band left out


def test_focus_machines():
    worst, share = calc.focus_machines(HEALTH, top=1)
    assert list(worst["machine_id"]) == [1]
    assert share == pytest.approx(400 / 500)


def test_focus_lines():
    lines = pd.DataFrame(
        {
            "plant_name": ["P", "P"],
            "line_id": ["A", "B"],
            "machines": [2, 2],
            "failures": [2, 6],
            "failures_per_machine": [1.0, 3.0],
            "cost_per_machine": [1.0, 3.0],
            "failure_cost": [100.0, 300.0],
            "availability": [0.99, 0.97],
        }
    )
    out = calc.focus_lines(lines)
    assert list(out["line_id"]) == ["B", "A"]
    assert list(out["failure_cost_share"]) == [0.75, 0.25]
    assert list(out["above_fleet_average"]) == [True, False]  # fleet: 8 / 4 = 2


def test_focus_components():
    out = calc.focus_components(
        pd.DataFrame(
            {
                "component": ["comp1", "comp2"],
                "failures": [2, 0],
                "failure_cost": [100.0, 0.0],
                "saving_if_all_prevented": [60.0, 0.0],
            }
        )
    )
    assert out.loc[0, "cost_per_failure"] == 50
    assert pd.isna(out.loc[1, "cost_per_failure"])  # no division by zero
    assert list(out["failure_cost_share"]) == [1.0, 0.0]


@pytest.mark.parametrize(
    ("reason", "signal"),
    [
        ("rotation 24 h mean 380.0 vs normal 450.0 (-16%)", "rotation"),
        ("error5 count 24 h 2 vs normal 0", "error5"),
        ("errors 24 h 3 vs normal 0", "errors (all types)"),
        ("comp2 days since replaced 400 vs normal 80", "comp2 age"),
    ],
)
def test_reason_signal(reason, signal):
    assert calc.reason_signal(reason) == signal


def test_warning_signatures_count_each_signal_once_per_alert():
    alerts = pd.DataFrame(
        {
            "predicted_component": ["comp1", "comp1", "comp2"],
            "reason_1": ["volt 3 h mean 190", "volt 24 h mean 185", "rotation 3 h mean 380"],
            "reason_2": ["volt 24 h mean 185", "error1 count 24 h 1", None],
            "reason_3": [None, None, None],
        }
    )
    out = calc.warning_signatures(alerts, top=2)
    comp1 = out[out["component"] == "comp1"].set_index("signal")["share_of_alerts"]
    assert comp1.to_dict() == {"volt": 1.0, "error1": 0.5}
    assert out[out["component"] == "comp2"]["signal"].tolist() == ["rotation"]


# ------------------------------------------------------------------ build


@pytest.fixture
def scored(silver, tmp_path):
    write_predictions(silver.parent / "gold")
    path = tmp_path / "scored.duckdb"
    build_warehouse(silver, path, gold_dir=silver.parent / "gold")
    return path


@pytest.fixture
def models_dir(tmp_path):
    d = tmp_path / "models"
    d.mkdir()
    card = {
        "test": {"events_caught": 9, "failure_events": 10, "false_alarms": 3, "net_saving": 1.0}
    }
    (d / "failure_24h_model_card.json").write_text(json.dumps(card), encoding="utf-8")
    return d


def test_build_insights(scored, models_dir, tmp_path):
    reports = tmp_path / "reports"
    insights = build_insights(scored, models_dir=models_dir, reports_dir=reports)

    assert json.loads((reports / INSIGHTS_FILE_NAME).read_text(encoding="utf-8")) == json.loads(
        json.dumps(insights, default=str)
    )
    # one failure event (comp1 + comp2 on machine 1) in 738 h on 2 machines
    base = insights["base_year"]
    years = 738 / 8760
    assert base["failure_events"] == 1
    assert base["component_failures"] == {"comp1": 1, "comp2": 1}
    assert base["saving_if_every_failure_caught"] == pytest.approx(700_000 / years)
    assert base["downtime_avoidable_h"] == pytest.approx((10 - 3) / years)

    names = [s["name"] for s in insights["scenarios"]]
    assert names == ["conservative", "expected", AS_TESTED]
    tested = insights["scenarios"][-1]
    assert tested["catch_rate"] == pytest.approx(0.9)
    assert tested["false_alarms_per_catch"] == pytest.approx(3 / 9)

    grid = insights["sensitivity"]
    assert len(grid["net_saving"]) == len(grid["catch_rates"])
    assert len(grid["net_saving"][0]) == len(grid["false_alarm_costs"])
    assert insights["focus"]["costliest_10_share_of_failure_cost"] == pytest.approx(1.0)
    assert [m["machine_id"] for m in insights["focus"]["costliest_machines"]] == [1, 2]
    assert insights["warning_signatures"] == [
        {"component": "comp2", "signal": "rotation", "share_of_alerts": 1.0}
    ]
    assert insights["test_period"]["events_caught"] == 9


def test_build_insights_needs_a_trained_model(scored, tmp_path):
    with pytest.raises(FileNotFoundError, match="iiot ml train"):
        build_insights(scored, models_dir=tmp_path / "empty", reports_dir=tmp_path / "r")


# ------------------------------------------------------------------ report


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (262_700_000, "₹26.27 cr"),
        (5_000_000, "₹50.0 lakh"),
        (20_000, "₹20,000"),
        (-2e7, "₹-2.00 cr"),
    ],
)
def test_inr(value, text):
    assert report.inr(value) == text


def test_months_and_plural():
    assert report.months(None) == "never"
    assert report.months(0.3) == "under 1 month"
    assert report.months(14.25) == "14.2 months"
    assert report.plural(1, "false alarm") == "1 false alarm"
    assert report.plural(3, "false alarm") == "3 false alarms"


def test_write_report(scored, models_dir, tmp_path):
    reports = tmp_path / "reports"
    insights = build_insights(scored, models_dir=models_dir, reports_dir=reports)
    out = report.write_report(reports_dir=reports)  # reads insights.json

    md = out.read_text(encoding="utf-8")
    for name in report.FIGURES:
        assert (reports / "figures" / f"{name}.png").stat().st_size > 0
        assert f"(figures/{name}.png)" in md
    expected = next(s for s in insights["scenarios"] if s["name"] == "expected")
    assert f"**{report.inr(expected['net_saving'])} a year**" in md
    assert "Start with PUNE-L1" in md  # the only line with a failure
    assert "comp2 | rotation (100%)" in md
    assert "with 3 false alarms." in md
    # unchanged insights give byte-identical charts
    first = (reports / "figures" / "lines.png").read_bytes()
    report.write_report(insights, reports_dir=reports)
    assert (reports / "figures" / "lines.png").read_bytes() == first


def test_write_report_needs_insights(tmp_path):
    with pytest.raises(FileNotFoundError, match="insights"):
        report.write_report(reports_dir=tmp_path)


def test_show_insights(scored, models_dir, tmp_path):
    reports = tmp_path / "reports"
    with pytest.raises(FileNotFoundError, match="iiot insights build"):
        show_insights(reports)
    built = build_insights(scored, models_dir=models_dir, reports_dir=reports)
    assert show_insights(reports) == json.loads(json.dumps(built, default=str))
