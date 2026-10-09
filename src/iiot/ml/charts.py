"""Charts logged with each ML run (saved as PNG in MLflow, so they are static).

Style: the project's validated colour-blind-safe palette in fixed order (blue,
then orange), a light surface, recessive grid and axes, thin marks, one y-axis per
chart, a legend whenever there are two or more series, and text in ink colours.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # no display needed: charts are written to files

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

SERIES = ["#2a78d6", "#eb6834"]  # categorical slots 1 and 2, validated for colour blindness
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
LAKH = 1e5


def _figure(title: str) -> tuple[Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=120)
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
