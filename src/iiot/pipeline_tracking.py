"""Track each `iiot run` in MLflow (experiment `iiot-data-pipeline`).

Every full pipeline run becomes one MLflow run with:
    params    the steps that ran
    metrics   <step>_seconds and <step>_passed for each step, pipeline_passed,
              <layer>_report_passed for each quality report, and rows per table
              of the Gold layer and the warehouse
    artifacts the quality reports (raw, Bronze, Silver, Gold, warehouse, ML)
    tags      git commit
So MLflow shows the whole project: when the data was last processed, whether
every check passed, and how long each step took - next to the ML experiments.

MLflow is imported only when tracking starts, so other commands stay fast.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from iiot.config import get_settings
from iiot.utils.logger import get_logger

logger = get_logger("iiot.pipeline_tracking")

EXPERIMENT = "iiot-data-pipeline"


def report_files() -> dict[str, Path]:
    """Where each layer writes its quality report."""
    paths = get_settings().paths
    return {
        "raw": paths.raw / "raw_validation_report.json",
        "bronze": paths.bronze / "bronze_report.json",
        "silver": paths.silver / "silver_quality_report.json",
        "gold": paths.gold / "gold_report.json",
        "warehouse": paths.warehouse / "model_report.json",
        "ml": paths.models / "ml_report.json",
    }


def report_metrics(name: str, report: dict) -> dict[str, float]:
    """Pass/fail and row counts from one quality report."""
    metrics = {}
    if "passed" in report:
        metrics[f"{name}_report_passed"] = float(bool(report["passed"]))
    for table, value in (report.get("tables") or {}).items():
        rows = value.get("rows") if isinstance(value, dict) else value
        if isinstance(rows, int | float):
            metrics[f"rows_{name}_{table}"] = float(rows)
    return metrics


@contextmanager
def track_pipeline(steps: list[str]) -> Iterator[Callable[[str, float, bool], None]]:
    """Yield a `record(step, seconds, passed)` function; log everything at the end.

    The run is written through MlflowClient with an explicit run ID instead of being
    the "active" run, so the ML steps inside the pipeline can start their own runs.
    """
    import time

    from mlflow import MlflowClient

    from iiot.ml import tracking

    experiment_id = tracking.setup(experiment=EXPERIMENT)
    client = MlflowClient()
    run = client.create_run(
        experiment_id, run_name="iiot-run", tags={"git_commit": tracking.git_commit()}
    )
    run_id = run.info.run_id
    client.log_param(run_id, "steps", ",".join(steps))
    results: dict[str, bool] = {}

    def record(step: str, seconds: float, passed: bool) -> None:
        results[step] = passed
        now = int(time.time() * 1000)
        client.log_metric(run_id, f"{step}_seconds", seconds, timestamp=now)
        client.log_metric(run_id, f"{step}_passed", float(passed), timestamp=now)

    status = "FAILED"
    try:
        yield record
        status = "FINISHED"
    finally:
        for name, path in report_files().items():
            if path.exists():
                client.log_artifact(run_id, str(path), "reports")
                report = json.loads(path.read_text(encoding="utf-8"))
                for key, value in report_metrics(name, report).items():
                    client.log_metric(run_id, key, value)
        passed = len(results) == len(steps) and all(results.values())
        client.log_metric(run_id, "pipeline_passed", float(passed))
        client.set_terminated(run_id, status)
        logger.info("Pipeline run tracked in MLflow experiment '%s'", EXPERIMENT)
