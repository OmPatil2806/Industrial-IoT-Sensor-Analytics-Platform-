"""Tests for iiot.pipeline_tracking (MLflow in a temporary folder)."""

from __future__ import annotations

import json

import mlflow

from iiot import pipeline_tracking
from iiot.ml import tracking
from iiot.pipeline_tracking import report_metrics, track_pipeline


def test_report_metrics_read_pass_flags_and_row_counts():
    report = {"passed": True, "tables": {"telemetry": {"rows": 10, "passed": True}, "kpi": 3}}
    assert report_metrics("gold", report) == {
        "gold_report_passed": 1.0,
        "rows_gold_telemetry": 10.0,
        "rows_gold_kpi": 3.0,
    }


def test_report_metrics_read_net_savings_from_the_insights():
    insights = {
        "scenarios": [
            {"name": "expected", "net_saving": 5.0},
            {"name": "as tested", "net_saving": 7},
        ]
    }
    assert report_metrics("insights", insights) == {
        "net_saving_expected": 5.0,
        "net_saving_as_tested": 7.0,
    }


def test_pipeline_run_is_logged_with_steps_and_reports(tmp_path, monkeypatch):
    report = tmp_path / "gold_report.json"
    report.write_text(json.dumps({"passed": True, "tables": {"labels": {"rows": 5}}}))
    monkeypatch.setattr(pipeline_tracking, "report_files", lambda: {"gold": report})
    original_setup = tracking.setup
    monkeypatch.setattr(
        tracking, "setup", lambda experiment=None: original_setup(tmp_path / "mlruns", experiment)
    )

    with track_pipeline(["silver-build", "gold-build"]) as record:
        record("silver-build", 1.5, True)
        # A step inside the pipeline (like the ML training) can start its own MLflow run.
        with mlflow.start_run(run_name="inner-ml-run"):
            mlflow.log_metric("x", 1.0)
        record("gold-build", 2.0, False)

    runs = mlflow.search_runs(experiment_names=[pipeline_tracking.EXPERIMENT])
    run = runs[runs["tags.mlflow.runName"] == "iiot-run"].iloc[0]
    assert run["status"] == "FINISHED"
    assert run["params.steps"] == "silver-build,gold-build"
    assert run["metrics.silver-build_passed"] == 1.0
    assert run["metrics.gold-build_passed"] == 0.0
    assert run["metrics.gold-build_seconds"] == 2.0
    assert run["metrics.pipeline_passed"] == 0.0
    assert run["metrics.gold_report_passed"] == 1.0
    assert run["metrics.rows_gold_labels"] == 5.0
    artifacts = mlflow.MlflowClient().list_artifacts(run["run_id"], "reports")
    assert [a.path for a in artifacts] == ["reports/gold_report.json"]
