"""Anomaly detection: is this machine's sensor behaviour unusual? (No labels needed.)

Unlike the failure model, this model never sees a failure label. It learns what
NORMAL sensor behaviour looks like and scores how far each row is from it, so it
can also flag problems of a kind that never happened before.

Model: scikit-learn pipeline
    SimpleImputer (median)  ->  StandardScaler  ->  IsolationForest
Inputs: the 20 sensor features (mean, variability and trend per sensor and window).
Errors, component age and machine data are left out on purpose: the anomaly score
is about how the machine behaves physically. Data-quality shares are left out too.
All machines share the same normal sensor levels, so one scaler fits the fleet.

Training data: FIT rows of HEALTHY periods only, at least HEALTHY_MARGIN_H hours
before any failure (or with no failure ahead).

Threshold (also label-free): the anomaly score that only FLAG_RATE of healthy
VALIDATION rows exceed, i.e. about 1% of normal rows are flagged by design.

Labels are used only to check the model afterwards, on TEST:
    flag rate by time before a failure (0-24 h, 24-48 h, 48-72 h, 72-168 h, normal)
    lift_48h        flag rate in the 48 h before a failure / flag rate in normal times
    event_recall_48h  failure events with at least one flag in the 48 h before
    pr_auc          ranking quality of the anomaly score for fails_within_24h
The model is registered as `iiot-anomaly` and aliased `champion` if lift_48h >= MIN_LIFT.

Usage:
    from iiot.ml.anomaly import train_anomaly_model
    card = train_anomaly_model()
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from iiot.config import get_settings
from iiot.ml import charts, tracking
from iiot.ml.baselines import log_metrics
from iiot.ml.data import Splits, fingerprint, load_splits
from iiot.ml.explain import feature_group
from iiot.ml.failure_model import CHAMPION_ALIAS, SKOPS_TRUSTED_TYPES, model_signature
from iiot.utils.logger import get_logger

logger = get_logger("iiot.ml.anomaly")

MODEL_NAME = "iiot-anomaly"
MODEL_CARD_FILE_NAME = "anomaly_model_card.json"
SENSOR_GROUPS = ("voltage", "rotation", "pressure", "vibration")
HEALTHY_MARGIN_H = 168  # a week before any failure counts as healthy
FLAG_RATE = 0.01
MIN_LIFT = 3.0
N_ESTIMATORS = 200
BUCKETS = [  # (label, lower bound exclusive, upper bound inclusive) of hours to failure
    ("0-24 h", 0, 24),
    ("24-48 h", 24, 48),
    ("48-72 h", 48, 72),
    ("72-168 h", 72, HEALTHY_MARGIN_H),
]
NORMAL = "normal (>168 h)"
# Isolation Forest stores its trees in sklearn's Tree type; trusted for the same reason as
# the gradient-boosting types: the model file is produced by this pipeline, never downloaded.
TRUSTED_TYPES = [*SKOPS_TRUSTED_TYPES, "sklearn.tree._tree.Tree", "numpy.dtype"]


def sensor_features(features: list[str]) -> list[str]:
    """Sensor statistics only: no errors, component age, machine data or data-quality shares."""
    return [
        f for f in features if feature_group(f) in SENSOR_GROUPS and not f.endswith("_share_24h")
    ]


def healthy(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Rows at least HEALTHY_MARGIN_H hours before any failure, or with no failure ahead."""
    htf = df["hours_to_failure"]
    return df[(df[target] == 0) & (htf.isna() | (htf > HEALTHY_MARGIN_H))]


def make_model(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("forest", IsolationForest(n_estimators=N_ESTIMATORS, random_state=seed)),
        ]
    )


def anomaly_score(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Higher = more unusual (the negated Isolation Forest score)."""
    return -model.score_samples(X)


def time_bucket(hours_to_failure: pd.Series) -> pd.Series:
    """Which period before a failure each row is in (NORMAL if more than a week away)."""
    out = pd.Series(NORMAL, index=hours_to_failure.index, dtype=object)
    for label, lo, hi in BUCKETS:
        out[(hours_to_failure > lo) & (hours_to_failure <= hi)] = label
    return out


def rate_key(bucket: str) -> str:
    """Metric name of a bucket's flag rate, e.g. "0-24 h" -> "flag_rate_0_24h"."""
    if bucket == NORMAL:
        return "flag_rate_normal"
    return "flag_rate_" + bucket.split(" ")[0].replace("-", "_") + "h"


def evaluate_anomalies(df: pd.DataFrame, scores: np.ndarray, threshold: float, target: str) -> dict:
    """Label-based check of a label-free model: are flags concentrated before failures?"""
    flags = scores >= threshold
    bucket = time_bucket(df["hours_to_failure"]).to_numpy()
    rates = {
        label: float(flags[bucket == label].mean()) if (bucket == label).any() else 0.0
        for label in [b[0] for b in BUCKETS] + [NORMAL]
    }
    before_48h = np.isin(bucket, ["0-24 h", "24-48 h"])
    normal_rate = rates[NORMAL]
    rate_48h = float(flags[before_48h].mean()) if before_48h.any() else 0.0

    within_48h = df.loc[before_48h, ["machine_id", "timestamp", "hours_to_failure"]].assign(
        failure_time=lambda d: d["timestamp"] + pd.to_timedelta(d["hours_to_failure"], unit="h"),
        flag=flags[before_48h],
    )
    events = within_48h.groupby(["machine_id", "failure_time"])["flag"].any()
    y = df[target].to_numpy()
    return {
        "threshold": float(threshold),
        "rows": len(df),
        "flags": int(flags.sum()),
        **{rate_key(k): v for k, v in rates.items() if k != NORMAL},
        "flag_rate_normal": normal_rate,
        "flag_rate_48h": rate_48h,
        "lift_48h": rate_48h / normal_rate if normal_rate else float("inf"),
        "failure_events": len(events),
        "event_recall_48h": float(events.mean()) if len(events) else 0.0,
        "pr_auc": float(average_precision_score(y, scores))
        if 0 < y.sum() < len(y)
        else float("nan"),
    }


def train_anomaly_model(
    splits: Splits | None = None,
    mlruns_dir: Path | None = None,
    models_dir: Path | None = None,
) -> dict:
    """Fit on healthy rows, set a label-free threshold, check against failures, register."""
    settings = get_settings()
    splits = splits or load_splits()
    seed = settings.ml.random_seed
    models_dir = Path(models_dir or settings.paths.models)
    inputs = sensor_features(splits.features)

    def scores(model: Pipeline, part: str) -> np.ndarray:
        return anomaly_score(model, splits.data[part][inputs])

    with tracking.start_run(
        "anomaly-detection", model_type="anomaly", mlruns_dir=mlruns_dir
    ) as run:
        tracking.log_splits(splits)
        train_rows = healthy(splits.data["fit"], splits.target)
        model = make_model(seed).fit(train_rows[inputs])

        healthy_val = healthy(splits.data["validation"], splits.target)
        threshold = float(np.quantile(anomaly_score(model, healthy_val[inputs]), 1 - FLAG_RATE))
        val_metrics = evaluate_anomalies(
            splits.data["validation"], scores(model, "validation"), threshold, splits.target
        )
        test_scores = scores(model, "test")
        test_metrics = evaluate_anomalies(
            splits.data["test"], test_scores, threshold, splits.target
        )
        promoted = test_metrics["lift_48h"] >= MIN_LIFT

        mlflow.log_params(
            {
                "algorithm": "IsolationForest",
                "n_estimators": N_ESTIMATORS,
                "inputs": len(inputs),
                "training_rows": len(train_rows),
                "healthy_margin_h": HEALTHY_MARGIN_H,
                "flag_rate_on_healthy_validation": FLAG_RATE,
                "threshold": threshold,
                "min_lift": MIN_LIFT,
                "random_seed": seed,
            }
        )
        log_metrics({"validation": _finite(val_metrics), "test": _finite(test_metrics)})
        mlflow.set_tag("beats_baselines", str(promoted))
        mlflow.log_figure(
            charts.flag_rate_bars(
                {b: test_metrics[rate_key(b)] for b in [*(b[0] for b in BUCKETS), NORMAL]},
                FLAG_RATE,
                "Test set: rows flagged as unusual, by time before a failure",
            ),
            "charts/flag_rate_by_time_to_failure_test.png",
        )

        info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=model_signature(splits.data["development"][inputs], test_scores),
            registered_model_name=MODEL_NAME,
            skops_trusted_types=TRUSTED_TYPES,
        )
        version = info.registered_model_version
        if promoted:
            MlflowClient().set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, version)
            logger.info("Registered %s version %s as @%s", MODEL_NAME, version, CHAMPION_ALIAS)
        else:
            logger.warning(
                "%s version %s: lift %.1f below %.1f; not promoted",
                MODEL_NAME,
                version,
                test_metrics["lift_48h"],
                MIN_LIFT,
            )

        card = {
            "model_name": MODEL_NAME,
            "version": version,
            "alias": CHAMPION_ALIAS if promoted else None,
            "run_id": run.info.run_id,
            "trained_at": datetime.now(UTC).isoformat(),
            "git_commit": tracking.git_commit(),
            "question": "Is this machine's sensor behaviour unusual compared with normal?",
            "algorithm": "SimpleImputer + StandardScaler + IsolationForest (scikit-learn)",
            "inputs": inputs,
            "training_rows": len(train_rows),
            "training_rule": f"healthy fit rows, more than {HEALTHY_MARGIN_H} h before a failure",
            "threshold": threshold,
            "threshold_rule": f"flags {FLAG_RATE:.0%} of healthy validation rows; no labels used",
            "data": {"fingerprint": fingerprint(splits.data["development"])},
            "validation": val_metrics,
            "test": test_metrics,
            "promoted": promoted,
            "notes": [
                "Labels are only used to evaluate the model, not to train it or set the threshold.",
                "Flags are a prompt to look at the machine, not a failure prediction.",
            ],
        }
        mlflow.log_dict(card, "model_card.json")

    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / MODEL_CARD_FILE_NAME).write_text(
        json.dumps(card, indent=2, default=str), encoding="utf-8"
    )
    logger.info(
        "TEST anomalies: %.1f%% of rows flagged in the 48 h before a failure vs %.1f%% in normal "
        "times (lift %.1f); %.0f%% of failures had a flag in the 48 h before",
        100 * test_metrics["flag_rate_48h"],
        100 * test_metrics["flag_rate_normal"],
        test_metrics["lift_48h"],
        100 * test_metrics["event_recall_48h"],
    )
    return card


def _finite(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if not (isinstance(v, float) and not np.isfinite(v))}
