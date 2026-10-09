"""Evaluate any failure model the way the business uses it.

A model gives every row (machine, time) a risk score; rows with a score at or
above the threshold raise an ALERT. Failures are rare (~2% of rows), so accuracy
would be misleading (always saying "no failure" is 98% accurate). We report:

Ranking (threshold-free)
    pr_auc      average precision: how well risky rows are ranked above safe ones.
                A model with no skill scores the positive rate (~0.02).
    roc_auc

Rows (at the threshold)
    precision, recall, f1 on the fails_within_<H>h label

Failure events (what maintenance cares about)
    A failure event is one machine failing at one time (several components can
    fail together). Its positive rows are the rows in the H hours before it.
    events_caught       events with at least one alert in those rows
    event_recall        events_caught / failure_events
    median_lead_time_h  hours from the first alert to the failure (caught events)

Inspections
    An alert sends a technician to inspect the machine; the machine then counts as
    checked for the next H hours, so alerts in that time cost nothing extra. An
    alert after that is a new inspection. (Counting one inspection per run of
    alerts instead would let a model alert all the time for the price of one.)
    inspections                 inspections triggered by the alerts
    false_alarms                inspections with no failure in the next H hours
    false_alarms_per_machine_month

Money (INR, illustrative costs from config)
    caught_saving     sum of saving_if_prevented of the components of caught events
                      (unplanned failure cost - planned maintenance cost)
    false_alarm_cost  false_alarms x ml.false_alarm_cost
    net_saving        caught_saving - false_alarm_cost
    potential_saving  the saving if every event were caught with no false alarm
    saving_captured   net_saving / potential_saving

`choose_threshold` picks the threshold with the highest net saving; it must be
called on validation data only, never on test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from iiot.config import get_settings

HOURS_PER_MONTH = 365.25 * 24 / 12
MAX_THRESHOLD_CANDIDATES = 200


@dataclass(frozen=True)
class Costs:
    savings: dict[str, float]  # component -> saving_if_prevented
    false_alarm_cost: float


def load_costs(silver_dir: Path | None = None) -> Costs:
    """Savings per component from Silver component_costs, false-alarm cost from config."""
    settings = get_settings()
    path = Path(silver_dir or settings.paths.silver) / "component_costs.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `iiot silver build` first.")
    costs = pd.read_parquet(path)
    savings = dict(zip(costs["component"], costs["saving_if_prevented"].astype(float), strict=True))
    return Costs(savings=savings, false_alarm_cost=settings.ml.false_alarm_cost)


def failure_events(df: pd.DataFrame, alerts: np.ndarray, target: str) -> pd.DataFrame:
    """One row per failure event in `df`: caught?, lead time, failed components."""
    positive = df[target].to_numpy() == 1
    rows = df.loc[positive, ["machine_id", "timestamp", "hours_to_failure", "failed_component"]]
    rows = rows.assign(
        failure_time=rows["timestamp"] + pd.to_timedelta(rows["hours_to_failure"], unit="h"),
        alert=alerts[positive],
    )
    rows["alert_time"] = rows["timestamp"].where(rows["alert"])
    events = rows.groupby(["machine_id", "failure_time"], as_index=False).agg(
        caught=("alert", "any"),
        first_alert=("alert_time", "min"),
        components=("failed_component", "first"),
    )
    events["lead_time_h"] = (
        events["failure_time"] - events["first_alert"]
    ).dt.total_seconds() / 3600
    events["components"] = events["components"].astype(str)
    return events


def inspections(
    df: pd.DataFrame, alerts: np.ndarray, target: str, cooldown: pd.Timedelta
) -> np.ndarray:
    """The label of each inspection: True if a failure followed it, False for a false alarm.

    Per machine, the first alert triggers an inspection; alerts within `cooldown`
    after an inspection are covered by it; the next alert after that triggers the
    next inspection.
    """
    machines = df["machine_id"].to_numpy()[alerts]
    times = df["timestamp"].to_numpy()[alerts]
    y = df[target].to_numpy()[alerts]
    order = np.lexsort((times, machines))
    machines, times, y = machines[order], times[order], y[order]
    starts = np.flatnonzero(np.r_[True, machines[1:] != machines[:-1]])
    ends = np.r_[starts[1:], len(machines)]
    labels = []
    for start, end in zip(starts, ends, strict=True):
        machine_times = times[start:end]
        i = 0
        while i < len(machine_times):
            labels.append(bool(y[start + i]))
            i = int(np.searchsorted(machine_times, machine_times[i] + cooldown, side="left"))
    return np.array(labels, dtype=bool)


def _event_savings(events: pd.DataFrame, costs: Costs) -> np.ndarray:
    """Saving if each event had been prevented: the sum over its failed components."""
    return np.array(
        [sum(costs.savings[c] for c in comps.split("+")) for comps in events["components"]],
        dtype=float,
    )


def _step() -> pd.Timedelta:
    """The row spacing of the dataset (Gold writes a row every feature_step_hours)."""
    return pd.Timedelta(hours=get_settings().gold.feature_step_hours)


def evaluate(
    df: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    costs: Costs,
    target: str,
    step: pd.Timedelta | None = None,
    cooldown: pd.Timedelta | None = None,
) -> dict:
    """All metrics of one model on one part of the data."""
    step = step or _step()
    cooldown = cooldown or pd.Timedelta(hours=get_settings().ml.prediction_horizon_hours)
    scores = np.asarray(scores, dtype=float)
    if len(scores) != len(df):
        raise ValueError(f"{len(scores)} scores for {len(df)} rows")
    y = df[target].to_numpy().astype(int)
    alerts = scores >= threshold
    tp = int((alerts & (y == 1)).sum())
    n_alerts, positives = int(alerts.sum()), int(y.sum())
    precision = tp / n_alerts if n_alerts else 0.0
    recall = tp / positives if positives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    both_classes = 0 < positives < len(y)

    events = failure_events(df, alerts, target)
    event_saving = pd.Series(_event_savings(events, costs))
    inspected = inspections(df, alerts, target, cooldown)
    false_alarms = int((~inspected).sum())
    span = df["timestamp"].max() - df["timestamp"].min() + step
    machine_months = df["machine_id"].nunique() * (span / pd.Timedelta(hours=HOURS_PER_MONTH))

    caught_saving = float(event_saving[events["caught"]].sum())
    false_alarm_cost = false_alarms * costs.false_alarm_cost
    potential = float(event_saving.sum())
    net = caught_saving - false_alarm_cost
    caught = events[events["caught"]]
    return {
        "threshold": float(threshold),
        "rows": len(df),
        "positives": positives,
        "alerts": n_alerts,
        "pr_auc": float(average_precision_score(y, scores)) if both_classes else float("nan"),
        "roc_auc": float(roc_auc_score(y, scores)) if both_classes else float("nan"),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "failure_events": len(events),
        "events_caught": int(events["caught"].sum()),
        "event_recall": float(events["caught"].mean()) if len(events) else 0.0,
        "median_lead_time_h": float(caught["lead_time_h"].median()) if len(caught) else 0.0,
        "inspections": len(inspected),
        "false_alarms": false_alarms,
        "false_alarms_per_machine_month": false_alarms / machine_months if machine_months else 0.0,
        "caught_saving": caught_saving,
        "false_alarm_cost": false_alarm_cost,
        "net_saving": net,
        "potential_saving": potential,
        "saving_captured": net / potential if potential else 0.0,
    }


def threshold_candidates(scores: np.ndarray) -> np.ndarray:
    """Distinct scores (at most MAX_THRESHOLD_CANDIDATES, spread over the high quantiles),
    plus one above the highest score, which means "never alert"."""
    scores = np.asarray(scores, dtype=float)
    unique = np.unique(scores)
    if len(unique) > MAX_THRESHOLD_CANDIDATES:
        # Alerts are rare, so the useful thresholds are in the top few percent of scores.
        quantiles = 1 - np.geomspace(0.5, 1e-4, MAX_THRESHOLD_CANDIDATES)
        unique = np.unique(np.quantile(scores, quantiles))
    return np.append(unique, scores.max() + 1.0)


def choose_threshold(
    df: pd.DataFrame,
    scores: np.ndarray,
    costs: Costs,
    target: str,
    step: pd.Timedelta | None = None,
    cooldown: pd.Timedelta | None = None,
) -> tuple[float, dict]:
    """The threshold with the highest net saving. Use on VALIDATION data only.

    Ties are broken towards the higher threshold (fewer alerts for the same money).
    Returns the threshold and its metrics.
    """
    scores = np.asarray(scores, dtype=float)
    cooldown = cooldown or pd.Timedelta(hours=get_settings().ml.prediction_horizon_hours)
    # Everything that does not depend on the threshold is computed once.
    positive = df[target].to_numpy() == 1
    events = failure_events(df, np.zeros(len(df), dtype=bool), target)
    event_of_row = (
        df.loc[positive, ["machine_id"]]
        .assign(
            failure_time=df.loc[positive, "timestamp"]
            + pd.to_timedelta(df.loc[positive, "hours_to_failure"], unit="h")
        )
        .merge(events[["machine_id", "failure_time"]].reset_index(), how="left")["index"]
        .to_numpy()
    )
    event_saving = _event_savings(events, costs)

    best_threshold, best_net = None, -np.inf
    for threshold in threshold_candidates(scores):
        alerts = scores >= threshold
        caught = np.bincount(event_of_row, weights=alerts[positive], minlength=len(events)) > 0
        false_alarms = int((~inspections(df, alerts, target, cooldown)).sum())
        net = event_saving[caught].sum() - false_alarms * costs.false_alarm_cost
        if net >= best_net:
            best_threshold, best_net = float(threshold), net
    return best_threshold, evaluate(df, scores, best_threshold, costs, target, step, cooldown)
