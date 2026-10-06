"""Raw data contract: check the downloaded source files before any processing.

A data contract states what we expect from a source (files, columns, types,
value ranges, row counts). If the source changes or a download is corrupted,
the pipeline stops here with a clear report instead of failing later.

Usage:
    python -m iiot.ingestion.validate_raw

Exit code 0 = all checks passed, 1 = at least one check failed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from iiot.config import get_settings
from iiot.utils.logger import get_logger

logger = get_logger("iiot.ingestion.validate_raw")

REPORT_FILE_NAME = "raw_validation_report.json"


@dataclass(frozen=True)
class TableContract:
    """Expectations for one source file (keyed like settings.dataset.files)."""

    columns: tuple[str, ...]
    min_rows: int
    max_rows: int
    date_column: str | None = None
    min_date: str | None = None
    max_date: str | None = None
    numeric_columns: tuple[str, ...] = ()
    categories: dict[str, frozenset[str]] = field(default_factory=dict)
    unique_columns: tuple[str, ...] = ()
    machine_count: int | None = None  # exact number of distinct machineIDs expected


COMPONENTS = frozenset({"comp1", "comp2", "comp3", "comp4"})

# Contract for the Microsoft Azure Predictive Maintenance dataset.
# Row-count ranges are deliberately loose around the known sizes.
CONTRACTS: dict[str, TableContract] = {
    "telemetry": TableContract(
        columns=("datetime", "machineID", "volt", "rotate", "pressure", "vibration"),
        min_rows=800_000,
        max_rows=1_000_000,
        date_column="datetime",
        min_date="2015-01-01",
        max_date="2016-01-02",
        numeric_columns=("volt", "rotate", "pressure", "vibration"),
        machine_count=100,
    ),
    "errors": TableContract(
        columns=("datetime", "machineID", "errorID"),
        min_rows=1_000,
        max_rows=10_000,
        date_column="datetime",
        min_date="2015-01-01",
        max_date="2016-01-02",
        categories={"errorID": frozenset({f"error{i}" for i in range(1, 6)})},
    ),
    "maintenance": TableContract(
        columns=("datetime", "machineID", "comp"),
        min_rows=1_000,
        max_rows=10_000,
        date_column="datetime",
        min_date="2014-01-01",  # maintenance history starts before the telemetry
        max_date="2016-01-02",
        categories={"comp": COMPONENTS},
    ),
    "failures": TableContract(
        columns=("datetime", "machineID", "failure"),
        min_rows=100,
        max_rows=5_000,
        date_column="datetime",
        min_date="2015-01-01",
        max_date="2016-01-02",
        categories={"failure": COMPONENTS},
    ),
    "machines": TableContract(
        columns=("machineID", "model", "age"),
        min_rows=100,
        max_rows=100,
        numeric_columns=("age",),
        categories={"model": frozenset({"model1", "model2", "model3", "model4"})},
        unique_columns=("machineID",),
        machine_count=100,
    ),
}


@dataclass
class CheckResult:
    table: str
    check: str
    passed: bool
    detail: str


@dataclass
class ValidationReport:
    results: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed]

    def add(self, table: str, check: str, passed: bool, detail: str) -> None:
        self.results.append(CheckResult(table, check, bool(passed), detail))

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "checks_run": len(self.results),
            "checks_failed": len(self.failures),
            "results": [asdict(r) for r in self.results],
        }


def parse_datetimes(values: pd.Series) -> pd.Series:
    """Parse timestamps fast as ISO 8601, falling back to per-value parsing for other
    formats (some copies of the dataset use e.g. "1/1/2015 6:00:00 AM").
    Unparseable values become NaT."""
    parsed = pd.to_datetime(values, format="ISO8601", errors="coerce")
    retry = parsed.isna() & values.notna()
    if retry.any():
        parsed[retry] = pd.to_datetime(values[retry], format="mixed", errors="coerce")
    return parsed


def _check_table(name: str, df: pd.DataFrame, contract: TableContract, report: ValidationReport):
    missing = [c for c in contract.columns if c not in df.columns]
    report.add(
        name,
        "columns",
        not missing,
        f"missing: {missing}" if missing else f"all {len(contract.columns)} present",
    )
    if missing:
        return  # remaining checks depend on these columns

    rows = len(df)
    report.add(
        name,
        "row_count",
        contract.min_rows <= rows <= contract.max_rows,
        f"{rows:,} rows (expected {contract.min_rows:,}-{contract.max_rows:,})",
    )

    nulls = int(df[list(contract.columns)].isna().sum().sum())
    report.add(name, "no_nulls", nulls == 0, f"{nulls:,} null values")

    if contract.date_column:
        dates = parse_datetimes(df[contract.date_column])
        bad = int(dates.isna().sum())
        report.add(name, "dates_parse", bad == 0, f"{bad:,} unparseable timestamps")
        valid = dates.dropna()
        if not valid.empty:
            lo, hi = valid.min(), valid.max()
            in_range = lo >= pd.Timestamp(contract.min_date) and hi <= pd.Timestamp(
                contract.max_date
            )
            report.add(
                name,
                "date_range",
                in_range,
                f"{lo} to {hi} (allowed {contract.min_date} to {contract.max_date})",
            )

    machine_ids = pd.to_numeric(df["machineID"], errors="coerce")
    bad_ids = int(machine_ids.isna().sum() + (machine_ids <= 0).sum())
    report.add(name, "machine_id_valid", bad_ids == 0, f"{bad_ids:,} invalid machineIDs")
    if contract.machine_count is not None:
        distinct = int(machine_ids.nunique())
        report.add(
            name,
            "machine_count",
            distinct == contract.machine_count,
            f"{distinct} machines (expected {contract.machine_count})",
        )

    for col in contract.numeric_columns:
        non_numeric = int(
            pd.to_numeric(df[col], errors="coerce").isna().sum() - df[col].isna().sum()
        )
        report.add(name, f"numeric:{col}", non_numeric == 0, f"{non_numeric:,} non-numeric values")

    for col, allowed in contract.categories.items():
        unexpected = sorted(set(df[col].dropna().astype(str)) - allowed)
        report.add(
            name,
            f"allowed_values:{col}",
            not unexpected,
            f"unexpected: {unexpected[:5]}" if unexpected else "all values allowed",
        )

    for col in contract.unique_columns:
        dupes = int(df[col].duplicated().sum())
        report.add(name, f"unique:{col}", dupes == 0, f"{dupes:,} duplicates")


def validate_raw(
    raw_dir: Path | None = None,
    contracts: dict[str, TableContract] | None = None,
    write_report: bool = True,
) -> ValidationReport:
    """Validate every source file against its contract and return the report."""
    settings = get_settings()
    raw_dir = raw_dir or settings.paths.raw
    contracts = contracts or CONTRACTS
    report = ValidationReport()

    for name, contract in contracts.items():
        path = raw_dir / settings.dataset.files[name]
        exists = path.exists()
        report.add(name, "file_exists", exists, str(path.name))
        if not exists:
            continue
        df = pd.read_csv(path)
        _check_table(name, df, contract, report)

    for r in report.results:
        log = logger.info if r.passed else logger.error
        log("%-4s %-12s %-26s %s", "PASS" if r.passed else "FAIL", r.table, r.check, r.detail)

    summary = f"{len(report.results) - len(report.failures)}/{len(report.results)} checks passed"
    if report.passed:
        logger.info("Raw data contract: %s", summary)
    else:
        logger.error("Raw data contract FAILED: %s", summary)

    if write_report:
        out = raw_dir / REPORT_FILE_NAME
        out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        logger.info("Report written to %s", out)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate raw source files (data contract).")
    parser.parse_args(argv)
    return 0 if validate_raw().passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
