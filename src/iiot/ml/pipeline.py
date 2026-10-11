"""Run the whole ML phase in order: `iiot ml train`.

    baselines -> failure model -> explanations -> component model -> anomaly model
    -> score every row (and rebuild the warehouse) -> ML report

Each model step logs its own MLflow run and registers its model; a model becomes
@champion only if it beats its baseline. The ML report then checks the champions
and their predictions. Returns the report.
"""

from __future__ import annotations

from iiot.ml.anomaly import train_anomaly_model
from iiot.ml.baselines import run_baselines
from iiot.ml.component_model import train_component_model
from iiot.ml.data import load_splits
from iiot.ml.evaluate import load_costs
from iiot.ml.explain import explain_champion
from iiot.ml.failure_model import train_failure_model
from iiot.ml.report import build_ml_report
from iiot.ml.score import score_all
from iiot.utils.logger import get_logger

logger = get_logger("iiot.ml.pipeline")


def train_all() -> dict:
    """Train, explain, score and check every model. Returns the ML report."""
    splits = load_splits()
    costs = load_costs()
    steps = [
        ("baselines", lambda: run_baselines(splits, costs)),
        ("failure model", lambda: train_failure_model(splits, costs)),
        ("explanations", lambda: explain_champion(splits)),
        ("component model", lambda: train_component_model(splits)),
        ("anomaly model", lambda: train_anomaly_model(splits)),
        ("scoring", lambda: score_all(splits)),
    ]
    for i, (name, step) in enumerate(steps, 1):
        logger.info("ML %d/%d: %s", i, len(steps) + 1, name)
        step()
    logger.info("ML %d/%d: report", len(steps) + 1, len(steps) + 1)
    return build_ml_report(splits)
