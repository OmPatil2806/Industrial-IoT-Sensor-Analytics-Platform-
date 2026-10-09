"""Component diagnosis: which component will fail, so the right spare part is sent.

One gradient-boosting classifier per component (comp1..comp4), each predicting
"this component fails within 24 h", wrapped in scikit-learn's
MultiOutputClassifier so the four are trained, saved and loaded as one model.
Each row gets four risks; the component with the highest risk is the diagnosis.

Diagnosis is judged on the rows before a real failure (the rows where it is used):
    top1_accuracy      the top-ranked component is one of the components that failed
    exact_set_accuracy the components with risk >= 0.5 are exactly the ones that
                       failed (2-component failures need both)
    pr_auc_<comp>      ranking quality of each component's risk over all rows
Baseline: always name the component that failed most often in the fit rows.

Training uses the same settings as the chosen failure model's first grid entry
(fit on FIT, checked on VALIDATION, refit on DEVELOPMENT, evaluated once on TEST).
The model is registered as `iiot-component` and aliased `champion` only if it
beats the baseline on test.

Usage:
    from iiot.ml.component_model import train_component_model
    card = train_component_model()
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
from sklearn.metrics import average_precision_score
from sklearn.multioutput import MultiOutputClassifier

from iiot.config import get_settings
from iiot.ml import charts, tracking
from iiot.ml.baselines import log_metrics
from iiot.ml.data import Splits, fingerprint, load_splits
from iiot.ml.failure_model import (
    CHAMPION_ALIAS,
    DEFAULT_GRID,
    FIXED_PARAMS,
    SKOPS_TRUSTED_TYPES,
    model_signature,
)
from iiot.ml.failure_model import make_model as make_classifier
from iiot.utils.logger import get_logger

logger = get_logger("iiot.ml.component_model")

MODEL_NAME = "iiot-component"
MODEL_CARD_FILE_NAME = "component_model_card.json"
PARAMS = DEFAULT_GRID[0]
SET_THRESHOLD = 0.5


def component_labels(splits: Splits) -> list[str]:
    """The per-component label columns, e.g. comp1_fails_within_24h."""
    return [c for c in splits.labels if c.startswith("comp") and "_fails_within_" in c]


def component_name(label: str) -> str:
    return label.split("_")[0]


def make_model(seed: int) -> MultiOutputClassifier:
    return MultiOutputClassifier(make_classifier(PARAMS, seed))


def component_risk(
    model: MultiOutputClassifier, X: pd.DataFrame, components: list[str]
) -> pd.DataFrame:
    """Risk of each component failing within the horizon, one column per component."""
    risks = [p[:, 1] for p in model.predict_proba(X)]
    return pd.DataFrame(np.column_stack(risks), columns=components, index=X.index)


def evaluate_diagnosis(
    df: pd.DataFrame, risk: pd.DataFrame, labels: list[str], target: str
) -> dict:
    """Diagnosis metrics on the rows before a real failure; ranking per component on all rows."""
    components = [component_name(lbl) for lbl in labels]
    before_failure = df[target].to_numpy() == 1
    actual = df.loc[before_failure, labels].to_numpy() == 1
    r = risk.to_numpy()[before_failure]
    top = r.argmax(axis=1)
    metrics = {
        "rows_before_failure": int(before_failure.sum()),
        "top1_accuracy": float(actual[np.arange(len(top)), top].mean()) if len(top) else 0.0,
        "exact_set_accuracy": float(((r >= SET_THRESHOLD) == actual).all(axis=1).mean())
        if len(top)
        else 0.0,
    }
    for comp, label in zip(components, labels, strict=True):
        y = df[label].to_numpy()
        if 0 < y.sum() < len(y):
            metrics[f"pr_auc_{comp}"] = float(average_precision_score(y, risk[comp]))
    return metrics


def confusion(df: pd.DataFrame, risk: pd.DataFrame, labels: list[str], target: str) -> pd.DataFrame:
    """Single-component failures: actual component (rows) vs top-ranked component (columns)."""
    before = df[target].to_numpy() == 1
    actual = df.loc[before, labels].to_numpy() == 1
    single = actual.sum(axis=1) == 1
    components = [component_name(lbl) for lbl in labels]
    truth = np.array(components)[actual[single].argmax(axis=1)]
    predicted = np.array(components)[risk.to_numpy()[before][single].argmax(axis=1)]
    table = pd.crosstab(pd.Series(truth, name="actual"), pd.Series(predicted, name="predicted"))
    return table.reindex(index=components, columns=components, fill_value=0)


def frequency_baseline(splits: Splits, labels: list[str], part: str) -> pd.DataFrame:
    """Every row gets each component's share of the failures in the fit rows."""
    fit = splits.data["fit"]
    shares = fit.loc[fit[splits.target] == 1, labels].mean().to_numpy()
    components = [component_name(lbl) for lbl in labels]
    n = len(splits.data[part])
    return pd.DataFrame(np.tile(shares, (n, 1)), columns=components)


def train_component_model(
    splits: Splits | None = None,
    mlruns_dir: Path | None = None,
    models_dir: Path | None = None,
) -> dict:
    """Train, evaluate and register the component diagnosis model. Returns its card."""
    settings = get_settings()
    splits = splits or load_splits()
    seed = settings.ml.random_seed
    models_dir = Path(models_dir or settings.paths.models)
    labels = component_labels(splits)
    components = [component_name(lbl) for lbl in labels]

    def metrics_for(model, part: str) -> dict:
        risk = component_risk(model, splits.X(part), components)
        return evaluate_diagnosis(splits.data[part], risk, labels, splits.target)

    with tracking.start_run(
        "component-diagnosis", model_type="component", mlruns_dir=mlruns_dir
    ) as run:
        tracking.log_splits(splits)
        checked = make_model(seed).fit(splits.X("fit"), splits.data["fit"][labels])
        val_metrics = metrics_for(checked, "validation")

        final = make_model(seed).fit(splits.X("development"), splits.data["development"][labels])
        test = splits.data["test"]
        test_risk = component_risk(final, splits.X("test"), components)
        test_metrics = evaluate_diagnosis(test, test_risk, labels, splits.target)
        baseline = evaluate_diagnosis(
            test, frequency_baseline(splits, labels, "test"), labels, splits.target
        )
        beats_baseline = test_metrics["top1_accuracy"] > baseline["top1_accuracy"]
        matrix = confusion(test, test_risk, labels, splits.target)

        mlflow.log_params(
            {
                **{f"model_{k}": v for k, v in {**FIXED_PARAMS, **PARAMS}.items()},
                "components": ",".join(components),
                "set_threshold": SET_THRESHOLD,
                "baseline": "most frequent failed component in the fit rows",
                "random_seed": seed,
            }
        )
        log_metrics({"validation": val_metrics, "test": test_metrics, "test_baseline": baseline})
        mlflow.set_tag("beats_baselines", str(beats_baseline))
        mlflow.log_figure(
            charts.confusion_heatmap(matrix, "Test set: actual vs diagnosed component"),
            "charts/confusion_test.png",
        )
        mlflow.log_text(matrix.to_csv(), "confusion_test.csv")

        sample = splits.X("development")
        info = mlflow.sklearn.log_model(
            final,
            name="model",
            signature=model_signature(sample, component_risk(final, sample.head(100), components)),
            registered_model_name=MODEL_NAME,
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )
        version = info.registered_model_version
        if beats_baseline:
            MlflowClient().set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, version)
            logger.info("Registered %s version %s as @%s", MODEL_NAME, version, CHAMPION_ALIAS)
        else:
            logger.warning(
                "%s version %s does NOT beat the baseline; not promoted", MODEL_NAME, version
            )

        card = {
            "model_name": MODEL_NAME,
            "version": version,
            "alias": CHAMPION_ALIAS if beats_baseline else None,
            "run_id": run.info.run_id,
            "trained_at": datetime.now(UTC).isoformat(),
            "git_commit": tracking.git_commit(),
            "question": "If this machine fails, which component will it be?",
            "algorithm": "MultiOutputClassifier of HistGradientBoostingClassifier (scikit-learn)",
            "params": {**FIXED_PARAMS, **PARAMS},
            "components": components,
            "features": splits.features,
            "data": {"fingerprint": fingerprint(splits.data["development"])},
            "validation": val_metrics,
            "test": test_metrics,
            "test_baseline": baseline,
            "beats_baseline": beats_baseline,
            "confusion_test": matrix.to_dict("index"),
        }
        mlflow.log_dict(card, "model_card.json")

    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / MODEL_CARD_FILE_NAME).write_text(
        json.dumps(card, indent=2, default=str), encoding="utf-8"
    )
    logger.info(
        "TEST diagnosis: top-1 accuracy %.1f%% (baseline %.1f%%), exact component set %.1f%%",
        100 * test_metrics["top1_accuracy"],
        100 * baseline["top1_accuracy"],
        100 * test_metrics["exact_set_accuracy"],
    )
    return card
