"""Failure prediction: will this machine fail within the next 24 hours?

Training follows the rules in iiot.ml.data and iiot.ml.evaluate:

1. Fit one gradient-boosting model per setting in the grid on the FIT rows, and
   choose each one's alert threshold on VALIDATION (highest net saving). Each
   candidate is a nested MLflow run.
2. Pick the candidate with the highest validation net saving (ties: higher
   PR-AUC, then the simpler setting, which comes first in the grid).
3. Sanity check: the same model trained on shuffled labels must score no better
   than chance on validation. If it does, the pipeline is leaking information.
4. Refit the chosen setting on all DEVELOPMENT rows, keep the validation
   threshold, and evaluate ONCE on TEST, next to the baselines.
5. Log parameters, metrics, charts and a model card to MLflow, register the model
   as `iiot-failure-24h`, and give it the alias `champion` only if it beats both
   baselines on test (higher net saving and higher PR-AUC than the error5 rule).

The model card is also written to models/failure_24h_model_card.json.

Usage:
    from iiot.ml.failure_model import train_failure_model
    result = train_failure_model()
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
from mlflow.models import ModelSignature, infer_signature
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from iiot.config import get_settings
from iiot.ml import charts, tracking
from iiot.ml.baselines import BASELINES, log_metrics
from iiot.ml.data import Splits, fingerprint, load_splits
from iiot.ml.evaluate import Costs, choose_threshold, evaluate, load_costs, net_saving_curve
from iiot.utils.logger import get_logger

logger = get_logger("iiot.ml.failure_model")

MODEL_NAME = "iiot-failure-24h"
CHAMPION_ALIAS = "champion"
MODEL_CARD_FILE_NAME = "failure_24h_model_card.json"
ALGORITHM = "HistGradientBoostingClassifier (scikit-learn)"
FIXED_PARAMS = {
    "max_iter": 300,
    "l2_regularization": 1.0,
    "categorical_features": "from_dtype",
    "early_stopping": False,
}
# Simplest first: on a tie the earlier setting wins.
DEFAULT_GRID = [
    {"learning_rate": 0.05, "max_leaf_nodes": 15},
    {"learning_rate": 0.1, "max_leaf_nodes": 15},
    {"learning_rate": 0.05, "max_leaf_nodes": 31},
    {"learning_rate": 0.1, "max_leaf_nodes": 31},
]
# MLflow saves scikit-learn models with skops, which only loads types marked as
# trusted. These are the internals of the gradient-boosting model; we trust them
# because the model files are produced by this pipeline, never downloaded.
SKOPS_TRUSTED_TYPES = [
    "functools.partial",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "sklearn.utils.validation.check_array",
]
SHUFFLED_TOLERANCE = 3.0  # shuffled-label PR-AUC must stay below 3x the positive rate


def model_signature(X: pd.DataFrame, scores: np.ndarray | pd.DataFrame) -> ModelSignature:
    """Inputs and output of the model, recorded in MLflow.

    Built from a sample with categories as text and every number as a float
    (counts can be missing in new data), so MLflow can describe all 40 inputs.
    """
    sample = X.head(100).copy()
    for col in sample.columns:
        is_category = isinstance(sample[col].dtype, pd.CategoricalDtype)
        sample[col] = sample[col].astype(str if is_category else "float64")
    return infer_signature(sample, scores[:100])


def make_model(params: dict, seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**FIXED_PARAMS, **params, random_state=seed)


def _scores(model: HistGradientBoostingClassifier, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def shuffled_label_check(splits: Splits, params: dict, seed: int) -> dict:
    """Train on shuffled labels: validation PR-AUC must stay near the positive rate."""
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(splits.y("fit").to_numpy())
    model = make_model(params, seed).fit(splits.X("fit"), shuffled)
    y_val = splits.y("validation")
    pr_auc = float(average_precision_score(y_val, _scores(model, splits.X("validation"))))
    chance = float(y_val.mean())
    return {
        "passed": pr_auc < SHUFFLED_TOLERANCE * chance,
        "shuffled_label_val_pr_auc": pr_auc,
        "chance_pr_auc": chance,
    }


def train_failure_model(
    splits: Splits | None = None,
    costs: Costs | None = None,
    mlruns_dir: Path | None = None,
    models_dir: Path | None = None,
    grid: list[dict] | None = None,
) -> dict:
    """Tune, check, refit, evaluate and register the failure model. Returns a summary."""
    settings = get_settings()
    splits = splits or load_splits()
    costs = costs or load_costs()
    grid = grid or DEFAULT_GRID
    seed = settings.ml.random_seed
    models_dir = Path(models_dir or settings.paths.models)
    val, test = splits.data["validation"], splits.data["test"]

    with tracking.start_run("failure-24h", model_type="failure_24h", mlruns_dir=mlruns_dir) as run:
        tracking.log_splits(splits)

        candidates = []
        for i, params in enumerate(grid, 1):
            with mlflow.start_run(
                run_name=f"candidate-{i}", nested=True, tags={"candidate": str(i)}
            ):
                model = make_model(params, seed).fit(splits.X("fit"), splits.y("fit"))
                val_scores = _scores(model, splits.X("validation"))
                threshold, val_metrics = choose_threshold(val, val_scores, costs, splits.target)
                mlflow.log_params(params)
                log_metrics({"validation": val_metrics})
            candidates.append((i, params, threshold, val_metrics, val_scores))
            logger.info(
                "candidate %d %s: validation PR-AUC %.4f, net saving INR %s at threshold %.3f",
                i,
                params,
                val_metrics["pr_auc"],
                f"{val_metrics['net_saving']:,.0f}",
                threshold,
            )
        i, params, threshold, val_metrics, val_scores = max(
            candidates, key=lambda c: (c[3]["net_saving"], c[3]["pr_auc"], -c[0])
        )
        logger.info("Chosen: candidate %d %s, threshold %.3f", i, params, threshold)

        sanity = shuffled_label_check(splits, params, seed)
        (logger.info if sanity["passed"] else logger.error)(
            "Shuffled-label check: validation PR-AUC %.4f (chance %.4f) -> %s",
            sanity["shuffled_label_val_pr_auc"],
            sanity["chance_pr_auc"],
            "PASS" if sanity["passed"] else "FAIL: the pipeline may be leaking information",
        )

        final = make_model(params, seed).fit(splits.X("development"), splits.y("development"))
        test_scores = _scores(final, splits.X("test"))
        test_metrics = evaluate(test, test_scores, threshold, costs, splits.target)
        baselines = {
            b.name: evaluate(test, b.score(splits.X("test")), b.threshold, costs, splits.target)
            for b in BASELINES
        }
        rule = baselines["error5-rule"]
        beats_baselines = bool(
            test_metrics["net_saving"] > max(b["net_saving"] for b in baselines.values())
            and test_metrics["pr_auc"] > rule["pr_auc"]
        )

        mlflow.log_params(
            {
                **{f"model_{k}": v for k, v in {**FIXED_PARAMS, **params}.items()},
                "chosen_candidate": i,
                "threshold": threshold,
                "threshold_rule": "highest net saving on validation",
                "false_alarm_cost": costs.false_alarm_cost,
                "random_seed": seed,
            }
        )
        log_metrics({"validation": val_metrics, "test": test_metrics})
        log_metrics({f"test_{name}": m for name, m in baselines.items()})
        mlflow.log_metrics(
            {
                "sanity_shuffled_label_val_pr_auc": sanity["shuffled_label_val_pr_auc"],
                "beats_baselines": float(beats_baselines),
            }
        )
        mlflow.set_tag("beats_baselines", str(beats_baselines))

        pr = charts.pr_curve(
            splits.y("test").to_numpy(),
            test_scores,
            "failure model",
            {"error5 rule": (rule["recall"], rule["precision"])},
            "Test set (Oct-Dec 2015): precision vs recall",
        )
        mlflow.log_figure(pr, "charts/pr_curve_test.png")
        curve = net_saving_curve(val, val_scores, costs, splits.target)
        saving = charts.net_saving_curve(
            curve, threshold, "Validation (Aug-Sep 2015): net saving by alert threshold"
        )
        mlflow.log_figure(saving, "charts/net_saving_vs_threshold_validation.png")

        info = mlflow.sklearn.log_model(
            final,
            name="model",
            signature=model_signature(splits.X("development"), test_scores),
            registered_model_name=MODEL_NAME,
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )
        version = info.registered_model_version
        if beats_baselines:
            MlflowClient().set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, version)
            logger.info("Registered %s version %s as @%s", MODEL_NAME, version, CHAMPION_ALIAS)
        else:
            logger.warning(
                "%s version %s does NOT beat the baselines on test; not promoted to @%s",
                MODEL_NAME,
                version,
                CHAMPION_ALIAS,
            )

        card = {
            "model_name": MODEL_NAME,
            "version": version,
            "alias": CHAMPION_ALIAS if beats_baselines else None,
            "run_id": run.info.run_id,
            "trained_at": datetime.now(UTC).isoformat(),
            "git_commit": tracking.git_commit(),
            "question": (
                f"Will the machine fail within the next "
                f"{settings.ml.prediction_horizon_hours} hours?"
            ),
            "algorithm": ALGORITHM,
            "params": {**FIXED_PARAMS, **params},
            "features": splits.features,
            "excluded_features": list(settings.ml.exclude_features),
            "data": {
                "fingerprint": fingerprint(splits.data["development"]),
                "splits": splits.summary(),
            },
            "threshold": threshold,
            "threshold_rule": "highest net saving on validation",
            "false_alarm_cost_inr": costs.false_alarm_cost,
            "validation": val_metrics,
            "test": test_metrics,
            "test_baselines": baselines,
            "beats_baselines": beats_baselines,
            "sanity_check": sanity,
            "notes": [
                "Data are the synthetic Microsoft Azure Predictive Maintenance dataset: failures "
                "follow strong error and wear patterns, so near-perfect scores are expected and "
                "would not carry over unchanged to a real plant.",
                "Costs are illustrative assumptions from config/settings.yaml.",
                "The threshold was chosen on validation with the model fitted on the fit rows "
                "and kept unchanged for the model refitted on all development rows.",
            ],
        }
        mlflow.log_dict(card, "model_card.json")

    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / MODEL_CARD_FILE_NAME).write_text(
        json.dumps(card, indent=2, default=str), encoding="utf-8"
    )
    _log_summary(test_metrics, baselines)
    return card


def _log_summary(model: dict, baselines: dict[str, dict]) -> None:
    for name, m in {"failure model": model, **baselines}.items():
        logger.info(
            "TEST %-13s PR-AUC %.3f | events caught %3d/%d | false alarms %3d | "
            "net saving INR %s (%.0f%% of possible)",
            name,
            m["pr_auc"],
            m["events_caught"],
            m["failure_events"],
            m["false_alarms"],
            f"{m['net_saving']:,.0f}",
            100 * m["saving_captured"],
        )
