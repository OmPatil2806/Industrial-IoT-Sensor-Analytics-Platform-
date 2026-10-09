"""Baselines every failure model must beat.

    never-fail   never raises an alert: the "do nothing" option. Its net saving is
                 0, and its PR-AUC equals the share of positive rows.
    error5-rule  alert when the machine raised error5 in the last 24 hours: the
                 rule a maintenance engineer would write first (Phase 6 showed that
                 51% of error5 events are followed by a failure within 48 hours).

Baselines are fixed rules: nothing is tuned, so evaluating them on the test set
is safe and gives the bar the models are compared against at the end.

Usage:
    from iiot.ml.baselines import run_baselines
    results = run_baselines()   # evaluates and logs one MLflow run per baseline
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from iiot.ml import tracking
from iiot.ml.data import Splits, load_splits
from iiot.ml.evaluate import Costs, evaluate, load_costs
from iiot.utils.logger import get_logger

logger = get_logger("iiot.ml.baselines")

EVALUATED_PARTS = ("validation", "test")


@dataclass(frozen=True)
class NeverFail:
    name: str = "never-fail"
    description: str = "never raise an alert"
    threshold: float = 1.0

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))


@dataclass(frozen=True)
class ErrorRule:
    column: str = "error5_count_24h"
    name: str = "error5-rule"
    description: str = "alert if error5 was raised in the last 24 hours"
    threshold: float = 1.0

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.column].to_numpy(dtype=float)


BASELINES = (NeverFail(), ErrorRule())


def log_metrics(metrics_by_part: dict[str, dict]) -> None:
    """Log every numeric metric as <part>_<metric>, e.g. validation_pr_auc."""
    prefix = {"validation": "val", "test": "test"}
    mlflow.log_metrics(
        {
            f"{prefix.get(part, part)}_{name}": value
            for part, metrics in metrics_by_part.items()
            for name, value in metrics.items()
            if isinstance(value, int | float) and not math.isnan(value)
        }
    )


def run_baselines(
    splits: Splits | None = None,
    costs: Costs | None = None,
    mlruns_dir: Path | None = None,
) -> dict[str, dict[str, dict]]:
    """Evaluate each baseline on validation and test and log one MLflow run each."""
    splits = splits or load_splits()
    costs = costs or load_costs()
    results: dict[str, dict[str, dict]] = {}
    for baseline in BASELINES:
        results[baseline.name] = {
            part: evaluate(
                splits.data[part],
                baseline.score(splits.X(part)),
                baseline.threshold,
                costs,
                splits.target,
            )
            for part in EVALUATED_PARTS
        }
        with tracking.start_run(baseline.name, model_type="baseline", mlruns_dir=mlruns_dir):
            tracking.log_splits(splits)
            mlflow.log_params(
                {
                    "rule": baseline.description,
                    "threshold": baseline.threshold,
                    "false_alarm_cost": costs.false_alarm_cost,
                }
            )
            log_metrics(results[baseline.name])
            mlflow.log_dict(results[baseline.name], "metrics.json")
        _log_summary(baseline.name, results[baseline.name])
    return results


def _log_summary(name: str, metrics_by_part: dict[str, dict]) -> None:
    for part, m in metrics_by_part.items():
        logger.info(
            "%-12s %-10s PR-AUC %.3f | events caught %d/%d | false alarms %d "
            "(%.2f per machine-month) | net saving INR %s",
            name,
            part,
            m["pr_auc"],
            m["events_caught"],
            m["failure_events"],
            m["false_alarms"],
            m["false_alarms_per_machine_month"],
            f"{m['net_saving']:,.0f}",
        )
