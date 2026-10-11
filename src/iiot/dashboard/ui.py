"""Shared Streamlit helpers for the dashboard pages: cached data, formatting, colours.

Data functions are wrapped with st.cache_data (5-minute TTL), so moving between
pages is instant while a rebuilt warehouse still shows up within minutes.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from iiot.dashboard import data

CACHE_SECONDS = 300

# Validated colour-blind-safe categorical palette, assigned in this fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # blue, orange, aqua, yellow
MUTED = "#898781"

STATUS_LABEL = {True: "✅ PASS", False: "❌ FAIL", None: "⚪ not run yet"}


def cached(func):
    """Cache a data-layer function's result for CACHE_SECONDS."""
    return st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)(func)


status = cached(data.status)
report_status = cached(data.report_status)
model_cards = cached(data.model_cards)
load_report = cached(data.load_report)
pipeline_runs = cached(data.pipeline_runs)
plants = cached(data.plants)
fleet_kpis = cached(data.fleet_kpis)
fleet_monthly = cached(data.fleet_monthly)
line_performance = cached(data.line_performance)
component_reliability = cached(data.component_reliability)
machine_health = cached(data.machine_health)
machines = cached(data.machines)
prediction_times = cached(data.prediction_times)
risk_as_of = cached(data.risk_as_of)
daily_alerts = cached(data.daily_alerts)
latest_alert_time = cached(data.latest_alert_time)
failures_between = cached(data.failures_between)
maintenance_plan = cached(data.maintenance_plan)
machine_sensors = cached(data.machine_sensors)
machine_events = cached(data.machine_events)
machine_predictions = cached(data.machine_predictions)


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def base_year():
    """Failure events of the analysis period (for the business case)."""
    from iiot.insights.build import base_year as _base_year

    return _base_year()


def require_warehouse() -> None:
    """Stop the page with instructions if the warehouse has not been built yet."""
    if not data.warehouse_exists():
        st.warning(
            "The warehouse has not been built yet. In the project folder, run:\n\n"
            "```\niiot run\n```\n\nthen refresh this page."
        )
        st.stop()


def require_predictions() -> None:
    """Stop the page with instructions if the ML models have not been scored yet."""
    require_warehouse()
    if not status()["has_predictions"]:
        st.warning(
            "There are no ML predictions yet. Train and score the models with:\n\n"
            "```\niiot ml train\n```\n\nthen refresh this page."
        )
        st.stop()


def inr(value: float) -> str:
    """Indian-style amount: crore above 1 crore, lakh above 1 lakh, else rupees."""
    if value is None or pd.isna(value):
        return "-"
    if abs(value) >= 1e7:
        return f"₹{value / 1e7:,.2f} cr"
    if abs(value) >= 1e5:
        return f"₹{value / 1e5:,.1f} lakh"
    return f"₹{value:,.0f}"


def pct(value: float, digits: int = 1) -> str:
    return "-" if value is None or pd.isna(value) else f"{100 * value:.{digits}f}%"


def hours(value: float, digits: int = 0) -> str:
    return "-" if value is None or pd.isna(value) else f"{value:,.{digits}f} h"


def as_of_picker() -> pd.Timestamp:
    """Date + 3-hourly time inside the out-of-sample period (default: the latest alert)."""
    times = prediction_times()
    default = latest_alert_time() or times.max()
    c1, c2 = st.columns([2, 1])
    day = c1.date_input(
        "As of date",
        value=default.date(),
        min_value=times.min().date(),
        max_value=times.max().date(),
    )
    hours = [f"{h:02d}:00" for h in range(0, 24, 3)]
    hour = c2.selectbox("Time", hours, index=hours.index(f"{default.hour // 3 * 3:02d}:00"))
    return pd.Timestamp(dt.datetime.combine(day, dt.time(int(hour[:2]))))
