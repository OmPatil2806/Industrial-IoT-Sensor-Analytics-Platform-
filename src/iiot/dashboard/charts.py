"""Altair charts for the dashboard (shown with Streamlit's theme, which follows light/dark).

Rules (shared with every chart in the project):
- one measure per chart (no second y-axis); trends as lines, magnitudes as bars from zero
- a single blue for magnitude; categorical colours only when series must be told
  apart, in the fixed validated order (blue, orange, aqua, yellow), with a legend
- every mark has a hover tooltip with its exact value
"""

from __future__ import annotations

import altair as alt
import pandas as pd

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated colour-blind-safe order
MUTED = "#898781"
# Months as categories (Jan..Dec): a continuous time axis would add an extra tick.
MONTH_X = alt.X("month(month):O", title=None, axis=alt.Axis(labelAngle=0, grid=False))
MONTH_TOOLTIP = alt.Tooltip("yearmonth(month):T", title="Month", format="%b %Y")
HEIGHT = 220


def _months(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(month=pd.to_datetime(df["year_month"] + "-01"))


def monthly_bars(
    df: pd.DataFrame, field: str, title: str, unit: str, fmt: str = ",.0f"
) -> alt.Chart:
    """One bar per month for one measure."""
    return (
        alt.Chart(_months(df), title=title, height=HEIGHT)
        .mark_bar(color=SERIES[0], cornerRadiusEnd=3)
        .encode(
            x=MONTH_X,
            y=alt.Y(f"{field}:Q", title=unit),
            tooltip=[MONTH_TOOLTIP, alt.Tooltip(f"{field}:Q", title=title, format=fmt)],
        )
    )


def monthly_line(df: pd.DataFrame, field: str, title: str, unit: str, fmt: str) -> alt.Chart:
    """A trend line with points for one measure (the y-axis need not start at zero)."""
    return (
        alt.Chart(_months(df), title=title, height=HEIGHT)
        .mark_line(color=SERIES[0], strokeWidth=2, point=alt.OverlayMarkDef(size=40))
        .encode(
            x=MONTH_X,
            y=alt.Y(
                f"{field}:Q", title=unit, scale=alt.Scale(zero=False), axis=alt.Axis(format=fmt)
            ),
            tooltip=[MONTH_TOOLTIP, alt.Tooltip(f"{field}:Q", title=title, format=fmt)],
        )
    )


def monthly_cost(df: pd.DataFrame) -> alt.Chart:
    """Cost per month, split into unplanned failures and planned maintenance (lakh INR)."""
    parts = {"failure_cost": "unplanned", "maintenance_cost": "planned"}
    long = _months(df).melt(
        id_vars=["month"], value_vars=list(parts), var_name="part", value_name="cost"
    )
    long["part"] = long["part"].map(parts)
    long["lakh"] = long["cost"] / 1e5
    return (
        alt.Chart(long, title="Cost per month", height=HEIGHT)
        .mark_bar()
        .encode(
            x=MONTH_X,
            y=alt.Y("lakh:Q", title="lakh INR", stack=True),
            color=alt.Color(
                "part:N",
                scale=alt.Scale(domain=list(parts.values()), range=[SERIES[1], SERIES[0]]),
                legend=alt.Legend(title=None, orient="top", direction="horizontal"),
            ),
            order=alt.Order("part:N", sort="descending"),
            tooltip=[
                MONTH_TOOLTIP,
                alt.Tooltip("part:N", title="Cost"),
                alt.Tooltip("lakh:Q", title="lakh INR", format=",.1f"),
            ],
        )
    )


def ranked_bars(
    df: pd.DataFrame, category: str, field: str, title: str, unit: str, fmt: str = ",.1f"
) -> alt.Chart:
    """Horizontal bars, largest at the top, with the value written at the bar end."""
    base = alt.Chart(df, title=title, height=alt.Step(30)).encode(
        y=alt.Y(f"{category}:N", sort="-x", title=None),
        x=alt.X(f"{field}:Q", title=unit),
        tooltip=[
            alt.Tooltip(f"{category}:N", title=category.replace("_", " ")),
            alt.Tooltip(f"{field}:Q", title=unit, format=fmt),
        ],
    )
    bars = base.mark_bar(color=SERIES[0], cornerRadiusEnd=3)
    labels = base.mark_text(align="left", dx=4).encode(text=alt.Text(f"{field}:Q", format=fmt))
    return bars + labels


def daily_alerts(df: pd.DataFrame, selected_day: pd.Timestamp) -> alt.Chart:
    """Machines with a failure alert per day, with the selected day marked."""
    bars = (
        alt.Chart(df, title="Machines with a failure alert, per day", height=160)
        .mark_bar(color=SERIES[0])
        .encode(
            x=alt.X("day:T", title=None, axis=alt.Axis(format="%d %b", grid=False)),
            y=alt.Y("machines_alerted:Q", title="machines", axis=alt.Axis(tickMinStep=1)),
            tooltip=[
                alt.Tooltip("day:T", title="Day", format="%a %d %b %Y"),
                alt.Tooltip("machines_alerted:Q", title="Machines alerted"),
                alt.Tooltip("machines_flagged:Q", title="Machines with unusual sensors"),
            ],
        )
    )
    marker = (
        alt.Chart(pd.DataFrame({"day": [pd.Timestamp(selected_day)]}))
        .mark_rule(color=MUTED, strokeDash=[4, 3], strokeWidth=1.5)
        .encode(x="day:T")
    )
    return bars + marker


SENSOR_TITLES = {
    "volt": "Voltage",
    "rotate": "Rotation",
    "pressure": "Pressure",
    "vibration": "Vibration",
}


def _event_rules(events: pd.DataFrame, legend: bool = True) -> alt.Chart:
    """Vertical rules: failures (dark, dashed) and maintenance (muted, dotted)."""
    marks = events[events["kind"].isin(["failure", "maintenance"])].drop_duplicates(
        ["timestamp", "kind"]
    )
    return (
        alt.Chart(marks)
        .mark_rule(strokeWidth=1.5)
        .encode(
            x="timestamp:T",
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(domain=["failure", "maintenance"], range=["#52514e", MUTED]),
                legend=alt.Legend(title=None, orient="top") if legend else None,
            ),
            strokeDash=alt.StrokeDash(
                "kind:N",
                scale=alt.Scale(domain=["failure", "maintenance"], range=[[6, 3], [2, 3]]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("timestamp:T", title="When", format="%a %d %b %Y, %H:%M"),
                alt.Tooltip("kind:N", title="Event"),
            ],
        )
    )


def risk_timeline(
    predictions: pd.DataFrame, events: pd.DataFrame, threshold: float, domain: list
) -> alt.LayerChart:
    """Predicted failure risk every 3 hours, the alert threshold, failures and maintenance."""
    line = (
        alt.Chart(predictions, title="Predicted risk of failure within 24 h", height=170)
        .mark_line(color=SERIES[0], strokeWidth=1.5, interpolate="step-after")
        .encode(
            x=alt.X("timestamp:T", title=None, scale=alt.Scale(domain=domain)),
            y=alt.Y(
                "failure_risk:Q",
                title="risk",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%"),
            ),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Scored at", format="%a %d %b %Y, %H:%M"),
                alt.Tooltip("failure_risk:Q", title="Risk", format=".1%"),
                alt.Tooltip("predicted_component:N", title="Component"),
                alt.Tooltip("reason_1:N", title="Main reason"),
            ],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"threshold": [threshold]}))
        .mark_rule(color=MUTED, strokeDash=[2, 2])
        .encode(
            y="threshold:Q",
            tooltip=[alt.Tooltip("threshold:Q", title="Alert threshold", format=".1%")],
        )
    )
    return alt.layer(line, rule, _event_rules(events, legend=False))


def sensor_panels(sensors: pd.DataFrame, events: pd.DataFrame, domain: list) -> alt.VConcatChart:
    """One small chart per sensor (own scale), with failures and maintenance marked."""
    panels = []
    for sensor, title in SENSOR_TITLES.items():
        part = sensors[sensors["sensor"] == sensor]
        line = (
            alt.Chart(part, title=title, height=110)
            .mark_line(color=SERIES[0], strokeWidth=1)
            .encode(
                x=alt.X("timestamp:T", title=None, scale=alt.Scale(domain=domain)),
                y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False)),
                tooltip=[
                    alt.Tooltip("timestamp:T", title="When", format="%a %d %b %Y, %H:%M"),
                    alt.Tooltip("value:Q", title=title, format=",.1f"),
                    alt.Tooltip("quality:N", title="Quality"),
                ],
            )
        )
        panels.append(alt.layer(line, _event_rules(events)))
    return alt.vconcat(*panels).resolve_scale(color="shared")
