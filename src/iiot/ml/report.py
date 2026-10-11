"""Check the ML models and their predictions: models/ml_report.json.

    champions            iiot-failure-24h, iiot-component and iiot-anomaly each have an
                         @champion that beat its baseline when it was trained
    data_unchanged       the champions were trained on the current data (the data
                         fingerprint in each model card equals today's)
    predictions_complete one prediction per Gold feature row, unique keys, no empty
                         scores, made by the current champion versions
    thresholds_applied   alert = failure_risk >= threshold, anomaly_flag = score >= threshold
    reproducible         re-scoring a random sample with the champions gives the same scores
    test_metrics_match   the stored test-period predictions give exactly the test results
                         in the failure model's card (alerts, events caught, false alarms)

`passed` is true only if every check passes.

Usage:
    from iiot.ml.report import build_ml_report
    report = build_ml_report()
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from iiot.config import get_settings
from iiot.ml.data import Splits, fingerprint, load_splits
from iiot.ml.evaluate import evaluate, load_costs
from iiot.ml.explain import NormalProfile
from iiot.ml.score import (
    KEYS,
    PREDICTIONS_FILE_NAME,
    Champion,
    load_champions,
    load_feature_rows,
    score_rows,
)
from iiot.utils.logger import get_logger

logger = get_logger("iiot.ml.report")

REPORT_FILE_NAME = "ml_report.json"
SAMPLE_ROWS = 2000
SCORE_COLUMNS = [
    "failure_risk",
    "comp1_risk",
    "comp2_risk",
    "comp3_risk",
    "comp4_risk",
    "anomaly_score",
]


def check_champions(champions: dict[str, Champion]) -> dict:
    flags = {
        "failure": champions["failure"].card.get("beats_baselines"),
        "component": champions["component"].card.get("beats_baseline"),
        "anomaly": champions["anomaly"].card.get("promoted"),
    }
    return {
        "passed": all(flags.values()),
        "versions": {k: c.version for k, c in champions.items()},
        "beat_baseline": flags,
    }


def check_data_unchanged(champions: dict[str, Champion], splits: Splits) -> dict:
    today = fingerprint(splits.data["development"])
    trained_on = {k: c.card["data"]["fingerprint"] for k, c in champions.items()}
    stale = [k for k, f in trained_on.items() if f != today]
    return {"passed": not stale, "current_fingerprint": today, "stale_models": stale}


def check_complete(predictions: pd.DataFrame, rows: pd.DataFrame, champions: dict) -> dict:
    duplicates = int(predictions.duplicated(KEYS).sum())
    empty = int(predictions[SCORE_COLUMNS].isna().sum().sum())
    expected = rows[KEYS].merge(predictions[KEYS], on=KEYS, how="left", indicator=True)
    missing = int((expected["_merge"] == "left_only").sum())
    versions_ok = all(
        set(predictions[f"{k}_model_version"].astype(str)) == {c.version}
        for k, c in champions.items()
    )
    return {
        "passed": duplicates == 0
        and empty == 0
        and missing == 0
        and len(predictions) == len(rows)
        and versions_ok,
        "rows": len(predictions),
        "feature_rows": len(rows),
        "missing_rows": missing,
        "duplicate_keys": duplicates,
        "empty_scores": empty,
        "current_versions": versions_ok,
    }


def check_thresholds(predictions: pd.DataFrame, champions: dict[str, Champion]) -> dict:
    alert_threshold = champions["failure"].card["threshold"]
    anomaly_threshold = champions["anomaly"].card["threshold"]
    wrong_alerts = int(
        (predictions["alert"] != (predictions["failure_risk"] >= alert_threshold)).sum()
    )
    wrong_flags = int(
        (predictions["anomaly_flag"] != (predictions["anomaly_score"] >= anomaly_threshold)).sum()
    )
    reasons_ok = bool(predictions.loc[predictions["alert"], "reason_1"].notna().all())
    return {
        "passed": wrong_alerts == 0 and wrong_flags == 0 and reasons_ok,
        "wrong_alerts": wrong_alerts,
        "wrong_anomaly_flags": wrong_flags,
        "every_alert_has_a_reason": reasons_ok,
    }


def check_reproducible(
    predictions: pd.DataFrame,
    rows: pd.DataFrame,
    champions: dict[str, Champion],
    profile: NormalProfile,
    seed: int,
) -> dict:
    sample = rows.sample(min(SAMPLE_ROWS, len(rows)), random_state=seed)
    again = score_rows(sample, champions, profile)
    stored = again[KEYS].merge(predictions, on=KEYS, how="left")
    max_diff = float(
        np.nanmax(np.abs(again[SCORE_COLUMNS].to_numpy() - stored[SCORE_COLUMNS].to_numpy()))
    )
    same_alerts = bool((again["alert"].to_numpy() == stored["alert"].to_numpy()).all())
    return {
        "passed": max_diff < 1e-9 and same_alerts,
        "rows_rescored": len(sample),
        "max_difference": max_diff,
    }


def check_test_metrics(predictions: pd.DataFrame, splits: Splits, champion: Champion) -> dict:
    test = splits.data["test"]
    scored = test[KEYS].merge(predictions[[*KEYS, "failure_risk"]], on=KEYS, how="left")
    metrics = evaluate(
        test,
        scored["failure_risk"].to_numpy(),
        champion.card["threshold"],
        load_costs(),
        splits.target,
    )
    card = champion.card["test"]
    compared = ["alerts", "events_caught", "failure_events", "false_alarms", "net_saving"]
    differences = {
        k: {"card": card[k], "predictions": metrics[k]} for k in compared if card[k] != metrics[k]
    }
    return {
        "passed": not differences,
        **{k: metrics[k] for k in compared},
        "differences": differences,
    }


def build_ml_report(
    splits: Splits | None = None,
    gold_dir: Path | None = None,
    mlruns_dir: Path | None = None,
    models_dir: Path | None = None,
) -> dict:
    """Run every check on the champions and predictions; write and return the report."""
    settings = get_settings()
    gold_dir = Path(gold_dir or settings.paths.gold)
    models_dir = Path(models_dir or settings.paths.models)
    path = gold_dir / PREDICTIONS_FILE_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Score the models first (iiot.ml.score).")
    splits = splits or load_splits(gold_dir)
    champions = load_champions(mlruns_dir)
    predictions = pd.read_parquet(path)
    rows = load_feature_rows(gold_dir)
    profile = NormalProfile.from_rows(splits.data["development"], splits.features, splits.target)

    checks = {
        "champions": check_champions(champions),
        "data_unchanged": check_data_unchanged(champions, splits),
        "predictions_complete": check_complete(predictions, rows, champions),
        "thresholds_applied": check_thresholds(predictions, champions),
        "reproducible": check_reproducible(
            predictions, rows, champions, profile, settings.ml.random_seed
        ),
        "test_metrics_match": check_test_metrics(predictions, splits, champions["failure"]),
    }
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "passed": all(c["passed"] for c in checks.values()),
        "checks": checks,
    }
    models_dir.mkdir(parents=True, exist_ok=True)
    out = models_dir / REPORT_FILE_NAME
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _log_summary(report, out)
    return report


def show_ml_report(models_dir: Path | None = None) -> dict:
    """Log the summary of the latest ml_report.json and return the report."""
    path = Path(models_dir or get_settings().paths.models) / REPORT_FILE_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Build the ML report first.")
    report = json.loads(path.read_text(encoding="utf-8"))
    logger.info("Report generated at %s", report["generated_at"])
    _log_summary(report, path)
    return report


def _log_summary(report: dict, out: Path) -> None:
    for name, check in report["checks"].items():
        (logger.info if check["passed"] else logger.error)(
            "%-22s %s", name, "PASS" if check["passed"] else "FAIL"
        )
    (logger.info if report["passed"] else logger.error)(
        "ML report: %s -> %s", "PASSED" if report["passed"] else "FAILED", out
    )
