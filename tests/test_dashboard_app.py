"""Smoke tests: open the dashboard with Streamlit's test runner (no browser needed)."""

from __future__ import annotations

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from iiot import config
from iiot.cli import DASHBOARD_APP


@pytest.fixture
def point_settings_at(write_config, monkeypatch, tmp_path):
    """Use a settings file whose warehouse and models folders are temporary."""

    def _point(warehouse_dir) -> None:
        path = write_config(
            {
                "paths": {
                    "warehouse": str(warehouse_dir),
                    "models": str(tmp_path / "models"),
                    "mlruns": str(tmp_path / "mlruns"),
                }
            }
        )
        monkeypatch.setenv("IIOT_CONFIG", str(path))
        config.get_settings.cache_clear()
        st.cache_data.clear()

    yield _point
    config.get_settings.cache_clear()
    st.cache_data.clear()


def run_app() -> AppTest:
    app = AppTest.from_file(str(DASHBOARD_APP), default_timeout=30)
    app.run()
    return app


def test_fleet_overview_is_the_default_page(built, point_settings_at):
    path, _ = built
    point_settings_at(path.parent)
    app = run_app()
    assert not app.exception
    assert app.title[0].value == "📊 Fleet overview"
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Failure events"] == "1"
    assert metrics["Total cost"] == "₹10.5 lakh"


def test_plant_filter_changes_the_numbers(built, point_settings_at):
    path, _ = built
    point_settings_at(path.parent)
    app = run_app()
    app.radio[0].set_value("Chennai Plant").run()  # machine 2: no failures
    assert not app.exception
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Failure events"] == "0"
    assert metrics["Availability"] == "100.00%"
    assert metrics["MTBF"] == "-"  # undefined without failures


def test_health_page(built, point_settings_at):
    path, _ = built
    point_settings_at(path.parent)
    app = run_app()
    app.switch_page("views/health.py").run()
    assert not app.exception
    assert app.title[0].value == "✅ Model & data health"
    metrics = {m.label: m.value for m in app.metric}
    assert (metrics["Machines"], metrics["Failures"]) == ("2", "2")
    assert any("not been trained" in i.value for i in app.info)  # no model cards yet
    assert any("No tracked pipeline runs" in i.value for i in app.info)


def test_home_page_without_a_warehouse_explains_what_to_run(tmp_path, point_settings_at):
    point_settings_at(tmp_path / "empty")
    app = run_app()
    assert not app.exception
    assert "iiot run" in app.warning[0].value


@pytest.fixture
def scored_warehouse(silver, tmp_path):
    """The conftest warehouse plus predictions (machine 1 alerted for comp2)."""
    from iiot.model.warehouse import build_warehouse
    from tests.test_dashboard_data import write_predictions

    write_predictions(silver.parent / "gold")
    path = tmp_path / "scored" / "iiot.duckdb"
    build_warehouse(silver, path, gold_dir=silver.parent / "gold")
    return path


def test_risk_page_replays_the_latest_alert(scored_warehouse, point_settings_at):
    point_settings_at(scored_warehouse.parent)
    app = run_app()
    app.switch_page("views/risk.py").run()
    assert not app.exception
    assert app.title[0].value == "🚨 Machines at risk"
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Failure alerts"] == "1"
    assert metrics["Unusual sensors"] == "2"
    assert metrics["Component most at risk"] == "comp2"
    assert metrics["Saving at stake"] == "₹3.8 lakh"
    assert app.selectbox[0].value == "00:00"  # the latest alert time: 2 Jan 00:00


def test_risk_page_without_predictions_explains_what_to_run(built, point_settings_at):
    path, _ = built
    point_settings_at(path.parent)
    app = run_app()
    app.switch_page("views/risk.py").run()
    assert not app.exception
    assert "iiot ml train" in app.warning[0].value


def test_machine_detail_page(scored_warehouse, point_settings_at, tmp_path):
    import json

    (tmp_path / "models").mkdir(exist_ok=True)
    (tmp_path / "models" / "failure_24h_model_card.json").write_text(json.dumps({"threshold": 0.9}))
    point_settings_at(scored_warehouse.parent)
    app = run_app()
    app.switch_page("views/machine.py").run()
    assert not app.exception
    assert app.title[0].value == "🔧 Machine detail"
    assert app.selectbox[0].value == 1  # the costliest machine
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Failures (2015)"] == "1"
    assert metrics["Last failure"] == "05 Jan"


def test_planner_page_lists_actions_and_offers_a_csv(scored_warehouse, point_settings_at):
    point_settings_at(scored_warehouse.parent)
    app = run_app()
    app.switch_page("views/planner.py").run()
    assert not app.exception
    assert app.title[0].value == "🗓️ Maintenance planner"
    metrics = {m.label: m.value for m in app.metric}
    assert (metrics["Replacements"], metrics["Inspections"]) == ("1", "1")
    assert metrics["Saving if done in time"] == "₹3.8 lakh"
    assert metrics["Lines involved"] == "2"
    assert len(app.get("download_button")) == 1


def test_planner_page_when_nothing_is_flagged(scored_warehouse, point_settings_at):
    point_settings_at(scored_warehouse.parent)
    app = run_app()
    app.switch_page("views/planner.py").run()
    app.selectbox[0].set_value("21:00")  # 1 Jan 21:00: before the alert
    app.date_input[0].set_value(pd.Timestamp("2015-01-01").date()).run()
    assert not app.exception
    assert "Nothing to do" in app.success[0].value


def test_business_case_page(built, point_settings_at):
    path, _ = built
    point_settings_at(path.parent)
    app = run_app()
    app.switch_page("views/business.py").run()
    assert not app.exception
    assert app.title[0].value == "💼 Business case"
    assert app.radio[0].value == "expected"  # no model card: no "as tested"
    assert app.slider[0].value == 70
    metrics = {m.label: m.value for m in app.metric}
    # one failure event (saving ₹7 lakh) in 738 h, scaled to a year, at 70% caught
    years = 738 / 8760
    gross, alarms = 0.7 * 700_000 / years, 0.7 * 0.5 / years * 20_000
    assert metrics["Net saving a year"] == f"₹{(gross - alarms - 5_000_000) / 1e5:,.1f} lakh"

    app.slider[0].set_value(100).run()  # catch everything
    assert not app.exception
    gross = 700_000 / years
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Net saving a year"] == f"₹{(gross - alarms / 0.7 - 5_000_000) / 1e5:,.1f} lakh"

    app.radio[0].set_value("conservative").run()  # resets the sliders to that preset
    assert app.slider[0].value == 50
