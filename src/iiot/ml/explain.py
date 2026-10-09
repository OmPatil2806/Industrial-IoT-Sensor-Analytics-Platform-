"""Explain the failure model: what it relies on, and why each alert was raised.

Global: permutation importance
    Shuffle one input (or one group of inputs) on the TEST set and measure how much
    the PR-AUC drops. A big drop means the model relies on it. Because inputs
    overlap (e.g. errors_total_24h and the five error counts), single-feature
    importance understates shared signal, so importance is also measured per
    GROUP: voltage, rotation, pressure, vibration, errors, component wear, machine.

Local: the reasons for one alert
    For each group, the alert row's values are replaced by that machine's NORMAL
    values (its median over healthy development rows: no failure in the next 24 h)
    and the row is scored again. The groups whose normal values lower the predicted
    risk the most are the reasons, best first (up to three, each lowering the risk
    by at least MIN_RISK_DROP). Each reason is described by the input in that group
    that is furthest from the machine's normal, for example:
        "vibration 24 h mean 48.9 vs normal 40.3 (+21%)"
        "error5 raised 2 times in the last 24 h (normal 0)"
        "comp2 last replaced 1,250 h ago (normal 410 h)"
    This uses the model itself, so the reasons are the model's reasons, not just
    unusual readings.

Usage:
    from iiot.ml.explain import explain_champion
    summary = explain_champion()   # logs to the champion model's MLflow run
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

from iiot.config import get_settings
from iiot.ml import charts, tracking
from iiot.ml.data import Splits, load_splits
from iiot.ml.failure_model import CHAMPION_ALIAS, MODEL_NAME
from iiot.utils.logger import get_logger

logger = get_logger("iiot.ml.explain")

SENSORS = ("volt", "rotate", "pressure", "vibration")
SENSOR_NAMES = {
    "volt": "voltage",
    "rotate": "rotation",
    "pressure": "pressure",
    "vibration": "vibration",
}
LOCAL_GROUPS = ("voltage", "rotation", "pressure", "vibration", "errors", "component wear")
MIN_RISK_DROP = 0.05  # a group must lower the risk by 5 points to count as a reason
TOP_REASONS = 3
GROUP_REPEATS = 5
FEATURE_REPEATS = 3
NO_REASON = "no single sensor, error or component stands out"


def feature_group(feature: str) -> str:
    prefix = feature.split("_")[0]
    if prefix in SENSOR_NAMES:
        return SENSOR_NAMES[prefix]
    if feature.startswith("error"):
        return "errors"
    if feature.startswith("hours_since_"):
        return "component wear"
    if feature in ("model", "age"):
        return "machine"
    return "other"


def feature_groups(features: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for f in features:
        groups.setdefault(feature_group(f), []).append(f)
    return groups


def _risk(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------- global importance


def group_importance(
    model, X: pd.DataFrame, y: pd.Series, repeats: int = GROUP_REPEATS, seed: int = 42
) -> pd.DataFrame:
    """PR-AUC drop when all inputs of a group are shuffled together (same row order)."""
    X = X.reset_index(drop=True)
    base = average_precision_score(y, _risk(model, X))
    rng = np.random.default_rng(seed)
    rows = []
    for group, cols in feature_groups(list(X.columns)).items():
        drops = []
        for _ in range(repeats):
            order = rng.permutation(len(X))
            shuffled = X.copy()
            for col in cols:
                shuffled[col] = X[col].iloc[order].reset_index(drop=True)
            drops.append(base - average_precision_score(y, _risk(model, shuffled)))
        rows.append(
            {
                "group": group,
                "importance": float(np.mean(drops)),
                "std": float(np.std(drops)),
                "features": len(cols),
            }
        )
    return pd.DataFrame(rows).sort_values("importance", ascending=False, ignore_index=True)


def feature_importance(
    model, X: pd.DataFrame, y: pd.Series, repeats: int = FEATURE_REPEATS, seed: int = 42
) -> pd.DataFrame:
    """PR-AUC drop when one input is shuffled (scikit-learn permutation importance)."""
    result = permutation_importance(
        model, X, y, scoring="average_precision", n_repeats=repeats, random_state=seed
    )
    return pd.DataFrame(
        {
            "feature": X.columns,
            "group": [feature_group(f) for f in X.columns],
            "importance": result.importances_mean,
            "std": result.importances_std,
        }
    ).sort_values("importance", ascending=False, ignore_index=True)


# ---------------------------------------------------------------- local reasons


@dataclass(frozen=True)
class NormalProfile:
    """Each machine's normal value of every numeric input (healthy development rows)."""

    median: pd.DataFrame  # index machine_id, one column per numeric feature
    std: pd.DataFrame

    @classmethod
    def from_rows(cls, df: pd.DataFrame, features: list[str], target: str) -> NormalProfile:
        numeric = [f for f in features if feature_group(f) in LOCAL_GROUPS]
        healthy = df.loc[df[target] == 0, ["machine_id", *numeric]]
        grouped = healthy.groupby("machine_id")[numeric]
        fleet_std = healthy[numeric].std().replace(0, np.nan).fillna(1.0)
        std = grouped.std().replace(0, np.nan).fillna(fleet_std)
        return cls(median=grouped.median(), std=std)


def describe(feature: str, value: float, normal: float) -> str:
    """Plain-language description of one input against the machine's normal value."""
    if feature.startswith("error"):
        what = "errors (all types)" if feature.startswith("errors_total") else feature.split("_")[0]
        return f"{what} raised {value:.0f} times in the last 24 h (normal {normal:.0f})"
    if feature.startswith("hours_since_"):
        component = feature.split("_")[2]
        return f"{component} last replaced {value:,.0f} h ago (normal {normal:,.0f} h)"
    parts = feature.split("_")
    sensor = SENSOR_NAMES.get(parts[0], parts[0])
    if parts[1] == "trend":
        return f"{sensor} trend (3 h vs 24 h mean) {value:+.1f} (normal {normal:+.1f})"
    stat = {"mean": "mean", "std": "variability"}.get(parts[1], parts[1])
    window = parts[2].replace("h", " h")
    change = f" ({100 * (value / normal - 1):+.0f}%)" if normal else ""
    return f"{sensor} {window} {stat} {value:.1f} vs normal {normal:.1f}{change}"


def _input_name(feature: str) -> str:
    """Short name of the input behind a reason: error1..5, errors, a sensor or a component."""
    if feature.startswith("errors_total"):
        return "errors (all types)"
    if feature.startswith("error") or feature.startswith("hours_since_"):
        return feature.split("_")[0] if feature.startswith("error") else feature.split("_")[2]
    return SENSOR_NAMES.get(feature.split("_")[0], feature)


def reason_inputs_by_component(true_alerts: pd.DataFrame) -> pd.DataFrame:
    """Share of each component's alerts that name each input among their reasons."""
    alerts = true_alerts.reset_index(drop=True).rename_axis("alert").reset_index()
    alerts["failed_component"] = alerts["failed_component"].astype(str)
    named = alerts.melt(
        id_vars=["alert", "failed_component"],
        value_vars=[c for c in alerts.columns if c.endswith("_input")],
        value_name="input",
    ).dropna(subset=["input"])
    # An input named twice in one alert (e.g. two error types) counts once for that alert.
    named = named.drop_duplicates(subset=["alert", "input"])
    counts = named.groupby(["failed_component", "input"]).size().unstack(fill_value=0)
    alerts_per_component = alerts["failed_component"].value_counts()
    return counts.div(alerts_per_component.loc[counts.index], axis=0).round(3)


def _describable(cols: list[str]) -> list[str]:
    """Inputs that describe the machine (data-quality shares are not a physical reason)."""
    return [c for c in cols if not c.endswith("_share_24h")]


def explain_rows(
    model,
    X: pd.DataFrame,
    machine_ids: pd.Series,
    profile: NormalProfile,
    top: int = TOP_REASONS,
) -> pd.DataFrame:
    """Up to `top` reasons per row: reason_k (text), reason_k_group and risk_drop_k."""
    X = X.reset_index(drop=True)
    machines = machine_ids.reset_index(drop=True)
    groups = {g: cols for g, cols in feature_groups(list(X.columns)).items() if g in LOCAL_GROUPS}
    risk = _risk(model, X)
    drops = {}
    for group, cols in groups.items():
        normal = X.copy()
        normal[cols] = profile.median.loc[machines, cols].to_numpy()
        drops[group] = risk - _risk(model, normal)
    drops = pd.DataFrame(drops)

    out = []
    for i in range(len(X)):
        ranked = drops.iloc[i].sort_values(ascending=False)
        row: dict = {"risk": float(risk[i])}
        reasons = [(g, d) for g, d in ranked.items() if d >= MIN_RISK_DROP][:top]
        for k in range(1, top + 1):
            row[f"reason_{k}"] = row[f"reason_{k}_group"] = row[f"reason_{k}_input"] = None
            row[f"risk_drop_{k}"] = np.nan
        for k, (group, drop) in enumerate(reasons, 1):
            cols = _describable(groups[group])
            machine = machines.iloc[i]
            z = (X.loc[i, cols] - profile.median.loc[machine, cols]).abs() / profile.std.loc[
                machine, cols
            ]
            feature = z.astype(float).idxmax()
            row[f"reason_{k}"] = describe(
                feature, float(X.loc[i, feature]), float(profile.median.loc[machine, feature])
            )
            row[f"reason_{k}_group"] = group
            row[f"reason_{k}_input"] = _input_name(feature)
            row[f"risk_drop_{k}"] = float(drop)
        if not reasons:
            row["reason_1"] = NO_REASON
        out.append(row)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- champion


def explain_champion(splits: Splits | None = None, mlruns_dir: Path | None = None) -> dict:
    """Explain the @champion failure model on the test set; log to its MLflow run."""
    settings = get_settings()
    splits = splits or load_splits()
    seed = settings.ml.random_seed
    tracking.setup(mlruns_dir)
    version = MlflowClient().get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}")
    threshold = mlflow.artifacts.load_dict(f"runs:/{version.run_id}/model_card.json")["threshold"]

    X, y = splits.X("test"), splits.y("test")
    groups = group_importance(model, X, y, seed=seed)
    features = feature_importance(model, X, y, seed=seed)

    test = splits.data["test"]
    alerts = _risk(model, X) >= threshold
    profile = NormalProfile.from_rows(splits.data["development"], splits.features, splits.target)
    reasons = explain_rows(model, X[alerts], test.loc[alerts, "machine_id"], profile)
    alert_rows = pd.concat(
        [
            test.loc[alerts, ["machine_id", "timestamp", splits.target, "failed_component"]]
            .reset_index(drop=True)
            .rename(columns={splits.target: "failure_followed"}),
            reasons,
        ],
        axis=1,
    )
    true_alerts = alert_rows[alert_rows["failure_followed"] == 1]
    by_component = reason_inputs_by_component(true_alerts)

    with mlflow.start_run(run_id=version.run_id):
        mlflow.log_figure(
            charts.importance_bars(
                groups,
                "group",
                "Input groups the model relies on (test set)",
            ),
            "explain/importance_by_group_test.png",
        )
        mlflow.log_figure(
            charts.importance_bars(
                features.head(15),
                "feature",
                "Top 15 inputs (test set)",
            ),
            "explain/importance_by_feature_test.png",
        )
        mlflow.log_text(groups.to_csv(index=False), "explain/importance_by_group_test.csv")
        mlflow.log_text(features.to_csv(index=False), "explain/importance_by_feature_test.csv")
        mlflow.log_text(alert_rows.to_csv(index=False), "explain/alert_reasons_test.csv")
        mlflow.log_text(by_component.to_csv(), "explain/reason_inputs_by_component_test.csv")
        mlflow.set_tag("explained", "true")

    summary = {
        "model_version": version.version,
        "run_id": version.run_id,
        "group_importance": groups.to_dict("records"),
        "top_features": features.head(10).to_dict("records"),
        "alerts_explained": len(alert_rows),
        "alerts_without_reason": int((alert_rows["reason_1"] == NO_REASON).sum()),
        "reason_inputs_by_component": by_component.to_dict("index"),
    }
    _log_summary(groups, by_component, alert_rows)
    return summary


def _log_summary(groups: pd.DataFrame, by_component: pd.DataFrame, alerts: pd.DataFrame) -> None:
    for g in groups.itertuples():
        logger.info("importance %-15s PR-AUC drop %.4f (+/- %.4f)", g.group, g.importance, g.std)
    for component, shares in by_component.iterrows():
        if "+" in component:
            continue  # multi-component failures mix the patterns of their components
        top = shares.sort_values(ascending=False).head(3)
        logger.info(
            "%s failures are flagged by: %s",
            component,
            ", ".join(f"{name} ({share:.0%})" for name, share in top.items()),
        )
    for row in alerts[alerts["failure_followed"] == 1].head(3).itertuples():
        named = [r for r in (row.reason_1, row.reason_2, row.reason_3) if isinstance(r, str)]
        logger.info(
            "example: machine %s at %s -> %s", row.machine_id, row.timestamp, "; ".join(named)
        )
