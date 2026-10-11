"""Charts logged with each ML run (saved as PNG in MLflow, so they are static).

Style: the project's validated colour-blind-safe palette in fixed order (blue,
then orange), a light surface, recessive grid and axes, thin marks, one y-axis per
chart, a legend whenever there are two or more series, and text in ink colours.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import precision_recall_curve

SERIES = ["#2a78d6", "#eb6834"]  # categorical slots 1 and 2, validated for colour blindness
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
LAKH = 1e5


def _figure(title: str) -> tuple[Figure, Axes]:
    # A bare Figure (not pyplot) needs no display backend and never changes the backend
    # of the session, so notebooks that import this module still show their charts.
    fig = Figure(figsize=(6.4, 4.0), dpi=120)
    ax = fig.add_subplot()
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, fontweight="bold", loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def pr_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    model_label: str,
    baseline_points: dict[str, tuple[float, float]],
    title: str,
) -> Figure:
    """Precision-recall curve of the model, with baselines as single points."""
    precision, recall, _ = precision_recall_curve(y_true, scores)
    fig, ax = _figure(title)
    ax.plot(recall, precision, color=SERIES[0], linewidth=2, label=model_label)
    for (name, (r, p)), colour in zip(baseline_points.items(), SERIES[1:], strict=False):
        ax.scatter(
            [r], [p], s=64, color=colour, edgecolor=SURFACE, linewidth=2, zorder=3, label=name
        )
        ax.annotate(
            name,
            (r, p),
            xytext=(8, -4),
            textcoords="offset points",
            color=INK_SECONDARY,
            fontsize=8,
        )
    chance = float(np.mean(y_true))
    ax.axhline(chance, color=INK_MUTED, linewidth=1, linestyle=":")
    ax.annotate(
        f"no skill ({chance:.1%} positive)",
        (1.0, chance),
        xytext=(0, 4),
        textcoords="offset points",
        ha="right",
        color=INK_MUTED,
        fontsize=8,
    )
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("recall (share of failure rows flagged)")
    ax.set_ylabel("precision (share of alerts that are real)")
    ax.legend(frameon=False, fontsize=8, loc="lower left", labelcolor=INK_SECONDARY)
    fig.tight_layout()
    return fig


def net_saving_curve(curve: pd.DataFrame, chosen: float, title: str) -> Figure:
    """Net saving (lakh INR) against the alert threshold, with the chosen threshold marked."""
    shown = curve[curve["threshold"] <= 1.0].sort_values("threshold")  # drop "never alert"
    fig, ax = _figure(title)
    ax.plot(shown["threshold"], shown["net_saving"] / LAKH, color=SERIES[0], linewidth=2)
    ax.axhline(0, color=AXIS, linewidth=1)
    best = curve.loc[curve["threshold"] == chosen, "net_saving"].iloc[0] / LAKH
    ax.axvline(chosen, color=INK_MUTED, linewidth=1, linestyle="--")
    ax.annotate(
        f"chosen {chosen:.3f}: INR {best:,.1f} lakh",
        (chosen, best),
        xytext=(-6, -14),
        textcoords="offset points",
        ha="right",
        color=INK_SECONDARY,
        fontsize=8,
    )
    ax.set_xlabel("alert threshold (predicted probability of failure within 24 h)")
    ax.set_ylabel("net saving (lakh INR)")
    fig.tight_layout()
    return fig


def importance_bars(table: pd.DataFrame, label_col: str, title: str) -> Figure:
    """Horizontal bars of PR-AUC lost per input or group, largest at the top."""
    table = table.sort_values("importance")
    fig, ax = _figure(title)
    fig.set_size_inches(6.4, 0.5 + 0.32 * len(table) + 1.2)
    ax.barh(table[label_col], table["importance"], color=SERIES[0], height=0.6)
    for y, value in enumerate(table["importance"]):
        ax.annotate(
            f"{value:.3f}",
            (max(value, 0), y),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            color=INK_SECONDARY,
            fontsize=8,
        )
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(table["importance"].max(), 0.01) * 1.15)
    ax.set_xlabel("PR-AUC lost when shuffled")
    ax.tick_params(axis="y", labelcolor=INK_SECONDARY)
    fig.tight_layout()
    return fig


SEQUENTIAL = ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def confusion_heatmap(matrix: pd.DataFrame, title: str) -> Figure:
    """Counts of actual (rows) vs predicted (columns), one blue scale, counts in each cell."""
    from matplotlib.colors import LinearSegmentedColormap

    fig, ax = _figure(title)
    fig.set_size_inches(5.2, 4.4)
    cmap = LinearSegmentedColormap.from_list("blue", SEQUENTIAL)
    values = matrix.to_numpy()
    ax.imshow(values, cmap=cmap, vmin=0, vmax=max(values.max(), 1))
    ax.grid(False)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns)
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    for (i, j), value in np.ndenumerate(values):
        dark = value > 0.55 * values.max()
        ax.annotate(
            f"{value:,}",
            (j, i),
            ha="center",
            va="center",
            color=SURFACE if dark else INK,
            fontsize=9,
        )
    ax.set_xlabel("diagnosed (highest risk)")
    ax.set_ylabel("actually failed")
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


def flag_rate_bars(rates: dict[str, float], design_rate: float, title: str) -> Figure:
    """Share of rows flagged per period before a failure, with the designed normal rate."""
    fig, ax = _figure(title)
    labels, values = list(rates), [100 * v for v in rates.values()]
    ax.bar(labels, values, color=SERIES[0], width=0.6)
    for x, value in enumerate(values):
        ax.annotate(
            f"{value:.1f}%",
            (x, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            color=INK_SECONDARY,
            fontsize=8,
        )
    ax.axhline(100 * design_rate, color=INK_MUTED, linewidth=1, linestyle=":")
    ax.annotate(
        f"dotted line: {design_rate:.0%} of rows flagged in normal operation, by design",
        (0.99, 0.97),
        xycoords="axes fraction",
        ha="right",
        va="top",
        color=INK_MUTED,
        fontsize=8,
    )
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, max(values + [100 * design_rate]) * 1.2)
    ax.set_xlabel("time before the next failure")
    ax.set_ylabel("rows flagged (%)")
    fig.tight_layout()
    return fig
