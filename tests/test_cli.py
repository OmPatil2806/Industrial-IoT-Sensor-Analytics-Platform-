"""Tests for the `iiot` command-line interface (steps are replaced by fakes)."""

from __future__ import annotations

import pytest

from iiot import cli


@pytest.fixture
def calls(monkeypatch):
    """Replace every data step with a fake that records its name."""

    class Calls(list):
        report: type

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
