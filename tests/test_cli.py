"""Tests for the `iiot` command-line interface (steps are replaced by fakes)."""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from iiot import cli


@pytest.fixture
def calls(monkeypatch):
    """Replace every data step with a fake that records its name."""

    class Calls(list):
        report: type
        silver_passed: bool
        gold_passed: bool
        model_passed: bool
        ml_passed: bool
        tracked: list

    recorded = Calls()

    class Report:
        passed = True

    monkeypatch.setattr(
        cli.download,
        "download_dataset",
        lambda force=False: recorded.append(f"download(force={force})"),
    )
    monkeypatch.setattr(
        cli.validate_raw, "validate_raw", lambda: recorded.append("validate") or Report()
    )
    monkeypatch.setattr(cli.inject_issues, "run", lambda: recorded.append("inject"))
    monkeypatch.setattr(cli.reference_data, "run", lambda: recorded.append("reference"))
    monkeypatch.setattr(
        cli.bronze_pipeline,
        "ingest_all",
        lambda force=False: recorded.append(f"bronze-ingest(force={force})"),
    )
    monkeypatch.setattr(
        cli.bronze_report, "build_report", lambda: recorded.append("bronze-report") or Report()
    )
    monkeypatch.setattr(
        cli.silver_report,
        "build_silver",
        lambda: recorded.append("silver-build") or {"passed": recorded.silver_passed},
    )
    monkeypatch.setattr(
        cli.silver_report,
        "show_report",
        lambda: recorded.append("silver-report") or {"passed": recorded.silver_passed},
    )
    monkeypatch.setattr(
        cli.gold_report,
        "build_gold",
        lambda: recorded.append("gold-build") or {"passed": recorded.gold_passed},
    )
    monkeypatch.setattr(
        cli.gold_report,
        "show_report",
        lambda: recorded.append("gold-report") or {"passed": recorded.gold_passed},
    )
    monkeypatch.setattr(
        cli.model_report,
        "build_model",
        lambda: recorded.append("model-build") or {"passed": recorded.model_passed},
    )
    monkeypatch.setattr(
        cli.model_report,
        "show_report",
        lambda: recorded.append("model-report") or {"passed": recorded.model_passed},
    )

    def fake_ml(name):
        def step(args):
            recorded.append(name)
            if not recorded.ml_passed:
                raise cli.StepFailed("ML checks failed")

        return step

    for name in ("ml-train", "ml-score", "ml-report"):
        monkeypatch.setitem(cli.STEPS, name, (fake_ml(name), cli.STEPS[name][1]))

    @contextmanager
    def fake_tracker(steps):
        run = {"steps": steps, "results": {}}
        recorded.tracked.append(run)
        yield lambda step, seconds, passed: run["results"].__setitem__(step, passed)

    monkeypatch.setattr(cli.pipeline_tracking, "track_pipeline", fake_tracker)
    recorded.ml_passed = True
    recorded.tracked = []
    recorded.silver_passed = True
    recorded.model_passed = True
    recorded.gold_passed = True
    recorded.report = Report
    return recorded


def test_prepare_runs_all_steps_in_order(calls):
    assert cli.main(["data", "prepare"]) == 0
    assert calls == ["download(force=False)", "validate", "inject", "reference"]


def test_prepare_force_download(calls):
    assert cli.main(["data", "prepare", "--force-download"]) == 0
    assert calls[0] == "download(force=True)"


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (["download"], ["download(force=False)"]),
        (["download", "--force"], ["download(force=True)"]),
        (["validate"], ["validate"]),
        (["inject"], ["inject"]),
        (["reference"], ["reference"]),
    ],
)
def test_single_steps(calls, command, expected):
    assert cli.main(["data", *command]) == 0
    assert calls == expected


def test_failed_validation_stops_the_pipeline(calls):
    calls.report.passed = False
    assert cli.main(["data", "prepare"]) == 1
    assert calls == ["download(force=False)", "validate"]


def test_missing_file_returns_error_code(calls, monkeypatch):
    def missing():
        raise FileNotFoundError("PdM_telemetry.csv not found")

    monkeypatch.setattr(cli.inject_issues, "run", missing)
    assert cli.main(["data", "inject"]) == 1


def test_download_error_stops_the_pipeline(calls, monkeypatch):
    def failing(force=False):
        raise cli.download.DownloadError("no credentials")

    monkeypatch.setattr(cli.download, "download_dataset", failing)
    assert cli.main(["data", "prepare"]) == 1
    assert calls == []


def test_unknown_command_exits_with_usage_error():
    with pytest.raises(SystemExit) as exc:
        cli.main(["data", "explode"])
    assert exc.value.code == 2


def test_run_executes_the_full_pipeline_in_order(calls):
    assert cli.main(["run"]) == 0
    assert calls == [
        "download(force=False)",
        "validate",
        "inject",
        "reference",
        "bronze-ingest(force=False)",
        "bronze-report",
        "silver-build",
        "gold-build",
        "model-build",
        "ml-train",
    ]


def test_run_force_flags_are_independent(calls):
    assert cli.main(["run", "--force-bronze"]) == 0
    assert calls[0] == "download(force=False)"
    assert calls[4] == "bronze-ingest(force=True)"
    calls.clear()
    assert cli.main(["run", "--force-download"]) == 0
    assert calls[0] == "download(force=True)"
    assert calls[4] == "bronze-ingest(force=False)"


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (["ingest"], ["bronze-ingest(force=False)"]),
        (["ingest", "--force"], ["bronze-ingest(force=True)"]),
        (["report"], ["bronze-report"]),
    ],
)
def test_bronze_commands(calls, command, expected):
    assert cli.main(["bronze", *command]) == 0
    assert calls == expected


def test_failed_bronze_report_returns_error(calls):
    calls.report.passed = False
    assert cli.main(["bronze", "report"]) == 1


def test_bronze_error_stops_the_pipeline(calls, monkeypatch):
    def failing(force=False):
        raise cli.BronzeError("row count mismatch")

    monkeypatch.setattr(cli.bronze_pipeline, "ingest_all", failing)
    assert cli.main(["run"]) == 1
    assert calls == ["download(force=False)", "validate", "inject", "reference"]


@pytest.mark.parametrize(
    ("command", "expected"),
    [(["build"], ["silver-build"]), (["report"], ["silver-report"])],
)
def test_silver_commands(calls, command, expected):
    assert cli.main(["silver", *command]) == 0
    assert calls == expected


@pytest.mark.parametrize("command", ["build", "report"])
def test_failed_silver_checks_return_error(calls, command):
    calls.silver_passed = False
    assert cli.main(["silver", command]) == 1


def test_failed_silver_build_fails_the_full_run(calls):
    calls.silver_passed = False
    assert cli.main(["run"]) == 1
    assert calls[-1] == "silver-build"  # Gold is not built on top of failed Silver


def test_silver_report_without_build_returns_error(calls, monkeypatch):
    def missing():
        raise FileNotFoundError("silver_quality_report.json not found")

    monkeypatch.setattr(cli.silver_report, "show_report", missing)
    assert cli.main(["silver", "report"]) == 1


@pytest.mark.parametrize(
    ("command", "expected"),
    [(["build"], ["gold-build"]), (["report"], ["gold-report"])],
)
def test_gold_commands(calls, command, expected):
    assert cli.main(["gold", *command]) == 0
    assert calls == expected


@pytest.mark.parametrize("command", ["build", "report"])
def test_failed_gold_checks_return_error(calls, command):
    calls.gold_passed = False
    assert cli.main(["gold", command]) == 1


def test_failed_gold_build_fails_the_full_run(calls):
    calls.gold_passed = False
    assert cli.main(["run"]) == 1
    assert calls[-1] == "gold-build"


@pytest.mark.parametrize(
    ("command", "expected"),
    [(["build"], ["model-build"]), (["report"], ["model-report"])],
)
def test_model_commands(calls, command, expected):
    assert cli.main(["model", *command]) == 0
    assert calls == expected


@pytest.mark.parametrize("command", ["build", "report"])
def test_failed_model_checks_return_error(calls, command):
    calls.model_passed = False
    assert cli.main(["model", command]) == 1


def test_failed_model_build_fails_the_full_run(calls):
    calls.model_passed = False
    assert cli.main(["run"]) == 1
    assert calls[-1] == "model-build"


def test_locked_warehouse_returns_error_code(calls, monkeypatch):
    def locked():
        raise cli.WarehouseLockedError("iiot.duckdb is open in DBeaver")

    monkeypatch.setattr(cli.model_report, "build_model", locked)
    assert cli.main(["model", "build"]) == 1


@pytest.mark.parametrize(
    ("command", "expected"),
    [(["train"], ["ml-train"]), (["score"], ["ml-score"]), (["report"], ["ml-report"])],
)
def test_ml_commands(calls, command, expected):
    assert cli.main(["ml", *command]) == 0
    assert calls == expected
    assert calls.tracked == []  # only `iiot run` is tracked in MLflow


def test_failed_ml_checks_fail_the_full_run(calls):
    calls.ml_passed = False
    assert cli.main(["run"]) == 1
    assert calls[-1] == "ml-train"


def test_run_skip_ml_stops_after_the_warehouse(calls):
    assert cli.main(["run", "--skip-ml"]) == 0
    assert calls[-1] == "model-build"


def test_run_is_tracked_with_every_step_result(calls):
    assert cli.main(["run"]) == 0
    [run] = calls.tracked
    assert run["steps"] == cli.FULL_PIPELINE
    assert run["results"] == {step: True for step in cli.FULL_PIPELINE}


def test_failed_step_is_tracked_as_failed(calls):
    calls.gold_passed = False
    assert cli.main(["run"]) == 1
    [run] = calls.tracked
    assert run["results"]["gold-build"] is False
    assert "model-build" not in run["results"]


def test_run_without_tracking(calls):
    assert cli.main(["run", "--no-tracking"]) == 0
    assert calls.tracked == []


def test_dashboard_command_starts_streamlit_from_the_project_folder(monkeypatch):
    seen = {}

    def fake_call(command, cwd):
        seen.update(command=command, cwd=cwd)
        return 0

    monkeypatch.setattr(cli.subprocess, "call", fake_call)
    assert cli.main(["dashboard", "--port", "8600"]) == 0
    assert seen["command"][1:4] == ["-m", "streamlit", "run"]
    assert seen["command"][4] == str(cli.DASHBOARD_APP)
    assert seen["command"][-2:] == ["--server.port", "8600"]
    assert seen["cwd"] == cli.PROJECT_ROOT
