"""Static charts for the business report (PNG, same style as the ML charts).

One measure per chart, one y-axis, values written on the bars, the validated
palette in fixed order, and a legend whenever there are two or more series.
"""

from __future__ import annotations

import pandas as pd
from matplotlib.figure import Figure

from iiot.ml.charts import (
    AXIS,
    INK_MUTED,
    INK_SECONDARY,
    _figure,  # noqa: PLC2701 - shared house style
)

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # blue, orange, aqua, yellow
DE_EMPHASISED = "#c3c2b7"
CRORE = 1e7


def _label_bars(ax, values, texts, horizontal: bool) -> None:
    for i, (value, text) in enumerate(zip(values, texts, strict=True)):
        xy, offset = ((value, i), (4, 0)) if horizontal else ((i, value), (0, 3))
        ax.annotate(
            text,
            xy,
            xytext=offset,
            textcoords="offset points",
            ha="left" if horizontal else "center",
            va="center" if horizontal else "bottom",
            color=INK_SECONDARY,
            fontsize=8,
        )


def _hbar_figure(title: str, rows: int) -> tuple[Figure, object]:
    fig, ax = _figure(title)
    fig.set_size_inches(6.4, 1.4 + 0.42 * rows)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelcolor=INK_SECONDARY)
    return fig, ax


def scenario_savings(scenarios: list[dict]) -> Figure:
    """Net annual saving (crore INR) per scenario."""
    names = [s["name"] for s in scenarios][::-1]
    values = [s["net_saving"] / CRORE for s in scenarios][::-1]
    fig, ax = _hbar_figure("Net saving per year by scenario", len(names))
    labels = [
        f"{n} ({s['catch_rate']:.0%} caught)" for n, s in zip(names, scenarios[::-1], strict=True)
    ]
    ax.barh(labels, values, color=SERIES[0], height=0.6)
    _label_bars(ax, values, [f"INR {v:,.2f} cr" for v in values], horizontal=True)
    ax.set_xlim(min(0, min(values)), max(values) * 1.25)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_xlabel("net saving per year (crore INR)")
    fig.tight_layout()
    return fig


def sensitivity(sens: dict) -> Figure:
    """Net annual saving against the catch rate, one line per false-alarm cost."""
    fig, ax = _figure(
        f"Net saving vs share of failures caught ({sens['false_alarms_per_catch']:g} "
        "false alarms per catch)"
    )
    rates = [100 * r for r in sens["catch_rates"]]
    for j, (cost, colour) in enumerate(zip(sens["false_alarm_costs"], SERIES, strict=False)):
        values = [row[j] / CRORE for row in sens["net_saving"]]
        ax.plot(
            rates,
            values,
            color=colour,
            linewidth=2,
            marker="o",
            markersize=3,
            label=f"INR {cost:,.0f} per false alarm",
        )
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
    ax.set_xlabel("failures caught in time (%)")
    ax.set_ylabel("net saving per year (crore INR)")
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK_SECONDARY)
    fig.tight_layout()
    return fig


def lines(lines: list[dict]) -> Figure:
    """Failures per machine by production line, above the fleet average highlighted."""
    df = pd.DataFrame(lines).sort_values("failures_per_machine")
    fleet = df["failures"].sum() / df["machines"].sum()
    fig, ax = _hbar_figure("Failures per machine by production line", len(df))
    colours = [SERIES[0] if a else DE_EMPHASISED for a in df["above_fleet_average"]]
    ax.barh(df["line_id"], df["failures_per_machine"], color=colours, height=0.6)
    _label_bars(
        ax, df["failures_per_machine"], [f"{v:.2f}" for v in df["failures_per_machine"]], True
    )
    ax.axvline(fleet, color=INK_MUTED, linewidth=1, linestyle=":")
    ax.annotate(
        f"fleet average {fleet:.2f} (blue: above it)",
        (fleet, len(df) - 0.5),
        xytext=(4, 0),
        textcoords="offset points",
        color=INK_MUTED,
        fontsize=8,
        va="bottom",
    )
    ax.set_xlim(0, df["failures_per_machine"].max() * 1.2)
    ax.set_xlabel("failures per machine in the year")
    fig.tight_layout()
    return fig


def age_bands(bands: list[dict]) -> Figure:
    """Failures per machine by machine age."""
    df = pd.DataFrame(bands)
    fig, ax = _figure("Failures per machine by machine age")
    ax.bar(df["age_band"], df["failures_per_machine"], color=SERIES[0], width=0.6)
    _label_bars(
        ax,
        df["failures_per_machine"],
        [
            f"{v:.1f} ({m} machines)"
            for v, m in zip(df["failures_per_machine"], df["machines"], strict=True)
        ],
        horizontal=False,
    )
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, df["failures_per_machine"].max() * 1.2)
    ax.set_xlabel("machine age")
    ax.set_ylabel("failures per machine in the year")
    fig.tight_layout()
    return fig


def components(comps: list[dict]) -> Figure:
    """Share of the failure cost by component."""
    df = pd.DataFrame(comps).sort_values("failure_cost_share")
    fig, ax = _hbar_figure("Share of the failure cost by component", len(df))
    values = 100 * df["failure_cost_share"]
    ax.barh(df["component"], values, color=SERIES[0], height=0.6)
    _label_bars(
        ax,
        values,
        [f"{v:.0f}% ({n} failures)" for v, n in zip(values, df["failures"], strict=True)],
        horizontal=True,
    )
    ax.set_xlim(0, values.max() * 1.35)
    ax.set_xlabel("share of the failure cost (%)")
    fig.tight_layout()
    return fig
