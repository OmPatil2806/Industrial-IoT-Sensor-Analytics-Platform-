"""Score every machine and time with the @champion models: data/gold/predictions.parquet.

Rows are the Gold feature rows (sensor_features + event_features), so the latest
hours of data are scored too (the ML dataset drops them because their labels are
not known yet). Per row:

    failure_risk          probability of a failure within 24 h (iiot-failure-24h)
    alert                 failure_risk >= the model's threshold
    predicted_component   component with the highest risk (iiot-component)
    comp1_risk..comp4_risk
    anomaly_score, anomaly_flag   (iiot-anomaly; flag = score >= its threshold)
    reason_1..reason_3    why the alert was raised (alert rows only, see iiot.ml.explain)
    out_of_sample         True from ml.train_end_date: the models never saw these rows.
                          Earlier rows were used for training, so their scores look
                          better than the models really are; reports use only
                          out-of-sample rows.
    *_model_version       which registered model version produced the column

`score_all()` writes the file and rebuilds the warehouse, so the scores appear in
fact_prediction and v_machine_risk_now.

Usage:
    from iiot.ml.score import score_all
    score_all()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient

from iiot.config import get_settings
from iiot.ml import anomaly, component_model, failure_model, tracking
from iiot.ml.data import Splits, load_splits
from iiot.ml.explain import NormalProfile, explain_rows
from iiot.utils.io import write_parquet_atomic
from iiot.utils.logger import get_logger

logger = get_logger("iiot.ml.score")

PREDICTIONS_FILE_NAME = "predictions.parquet"
KEYS = ["machine_id", "timestamp"]
COMPONENTS = ["comp1", "comp2", "comp3", "comp4"]
REASONS = ["reason_1", "reason_2", "reason_3"]


@dataclass(frozen=True)
class Champion:
    """A registered @champion model with its version and model card."""

    model: object
    version: str
    card: dict


def load_champion(name: str) -> Champion:
    """Load the @champion of a registered model (call tracking.setup first)."""
    version = MlflowClient().get_model_version_by_alias(name, failure_model.CHAMPION_ALIAS)
    model = mlflow.sklearn.load_model(f"models:/{name}@{failure_model.CHAMPION_ALIAS}")
    card = mlflow.artifacts.load_dict(f"runs:/{version.run_id}/model_card.json")
    return Champion(model=model, version=str(version.version), card=card)


def load_champions(mlruns_dir: Path | None = None) -> dict[str, Champion]:
    tracking.setup(mlruns_dir)
    names = {
        "failure": failure_model.MODEL_NAME,
        "component": component_model.MODEL_NAME,
        "anomaly": anomaly.MODEL_NAME,
    }
    return {key: load_champion(name) for key, name in names.items()}


def load_feature_rows(gold_dir: Path | None = None) -> pd.DataFrame:
    """Every Gold feature row (sensor + event features), latest hours included."""
    gold_dir = Path(gold_dir or get_settings().paths.gold)
    paths = [gold_dir / f"{t}.parquet" for t in ("sensor_features", "event_features")]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `iiot gold build` first.")
    sensor, event = (pd.read_parquet(p) for p in paths)
    return sensor.merge(event, on=KEYS, how="inner").sort_values(KEYS, ignore_index=True)


def score_rows(
    rows: pd.DataFrame, champions: dict[str, Champion], profile: NormalProfile
) -> pd.DataFrame:
    """Apply the three champion models (and alert reasons) to feature rows."""
    failure, comp, anom = champions["failure"], champions["component"], champions["anomaly"]
    out = rows[KEYS].reset_index(drop=True)
    X = rows[failure.card["features"]].reset_index(drop=True)

    out["failure_risk"] = failure.model.predict_proba(X)[:, 1]
    out["alert"] = out["failure_risk"] >= failure.card["threshold"]

    risks = component_model.component_risk(comp.model, rows[comp.card["features"]], COMPONENTS)
    risks = risks.reset_index(drop=True)
    out["predicted_component"] = risks.idxmax(axis=1)
    for c in COMPONENTS:
        out[f"{c}_risk"] = risks[c]

    out["anomaly_score"] = anomaly.anomaly_score(anom.model, rows[anom.card["inputs"]])
    out["anomaly_flag"] = out["anomaly_score"] >= anom.card["threshold"]

    for col in REASONS:
        out[col] = pd.Series(pd.NA, index=out.index, dtype="string")
    alerts = out["alert"].to_numpy()
    if alerts.any():
        reasons = explain_rows(failure.model, X[alerts], out.loc[alerts, "machine_id"], profile)
        for col in REASONS:
            out.loc[alerts, col] = reasons[col].astype("string").to_numpy()

    out["failure_model_version"] = failure.version
    out["component_model_version"] = comp.version
    out["anomaly_model_version"] = anom.version
    return out


def score_all(
    splits: Splits | None = None,
    gold_dir: Path | None = None,
    mlruns_dir: Path | None = None,
    rebuild_warehouse: bool = True,
) -> pd.DataFrame:
    """Score all feature rows, write predictions.parquet and (optionally) rebuild the warehouse."""
    settings = get_settings()
    gold_dir = Path(gold_dir or settings.paths.gold)
    splits = splits or load_splits(gold_dir)
    champions = load_champions(mlruns_dir)
    profile = NormalProfile.from_rows(splits.data["development"], splits.features, splits.target)

    rows = load_feature_rows(gold_dir)
    predictions = score_rows(rows, champions, profile)
    split_at = pd.Timestamp(settings.ml.train_end_date)
    predictions["out_of_sample"] = predictions["timestamp"] >= split_at

    path = gold_dir / PREDICTIONS_FILE_NAME
    write_parquet_atomic(predictions, path)
    recent = predictions[predictions["out_of_sample"]]
    logger.info(
        "Scored %s rows -> %s (%s out of sample: %s alerts, %s anomaly flags)",
        f"{len(predictions):,}",
        path,
        f"{len(recent):,}",
        f"{int(recent['alert'].sum()):,}",
        f"{int(recent['anomaly_flag'].sum()):,}",
    )
    if rebuild_warehouse:
        from iiot.model.report import build_model

        build_model(gold_dir=gold_dir)
    return predictions
