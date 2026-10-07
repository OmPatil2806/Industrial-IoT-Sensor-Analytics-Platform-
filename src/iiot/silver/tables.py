"""Silver event and master tables (everything except telemetry).

    Bronze table(s)               Silver table      Columns
    machines + machine_location   machines          machine_id, model, age, plant_id,
                                                    plant_name, city, line_id
    errors                        errors            timestamp, machine_id, error_id
    maintenance                   maintenance       timestamp, machine_id, component
    failures                      failures          timestamp, machine_id, component
    component_costs               component_costs   component + numeric cost columns

Event rows are kept only if they are valid; every dropped row is counted by
reason so the data-quality report shows exactly what was removed:
    bad_timestamp     timestamp missing or not parseable
    unknown_machine   machine_id not in the machines table
    unknown_value     error_id / component not in the allowed list
    duplicate         an identical event (same timestamp, machine and value) seen earlier

Lineage columns (_batch_id, _source_line) are kept on every table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from iiot.config import get_settings
from iiot.ingestion.validate_raw import CONTRACTS, parse_datetimes
from iiot.silver.io import read_bronze, write_silver
from iiot.utils.logger import get_logger

logger = get_logger("iiot.silver.tables")

LINEAGE = ["_batch_id", "_source_line"]
COST_NUMERIC_COLUMNS = [
    "repair_cost",
    "unplanned_repair_cost",
    "unplanned_downtime_hours",
    "planned_downtime_hours",
    "downtime_cost_per_hour",
    "unplanned_failure_cost",
    "planned_maintenance_cost",
    "saving_if_prevented",
]


@dataclass
class TableStats:
    table: str
    rows_in: int = 0
    rows_out: int = 0
    dropped: dict[str, int] = field(default_factory=dict)  # rows removed, by reason
    flagged: dict[str, int] = field(default_factory=dict)  # rows kept, but with a problem


@dataclass(frozen=True)
class EventSpec:
    bronze_table: str
    value_source: str  # column name in Bronze
    value_name: str  # column name in Silver
    allowed: frozenset[str]


def _text(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("").str.strip()


def _blank_to_na(s: pd.Series) -> pd.Series:
    text = _text(s)
    return text.mask(text == "")


def _to_int(s: pd.Series) -> pd.Series:
    numbers = pd.to_numeric(_blank_to_na(s), errors="coerce")
    return numbers.where(numbers == numbers.round()).astype("Int64")


def build_machines(
    machines: pd.DataFrame, location: pd.DataFrame
) -> tuple[pd.DataFrame, TableStats]:
    """One row per machine: model and age, plus plant and production line."""
    stats = TableStats("machines", rows_in=len(machines))
    allowed_models = CONTRACTS["machines"].categories["model"]
    df = pd.DataFrame(
        {
            "machine_id": _to_int(machines["machineID"]),
            "model": _text(machines["model"]),
            "age": _to_int(machines["age"]),
            **{c: machines[c].to_numpy() for c in LINEAGE if c in machines},
        }
    )
    checks = {
        "bad_machine_id": df["machine_id"].isna() | (df["machine_id"] <= 0),
        "unknown_value": ~df["model"].isin(allowed_models),
        "bad_age": df["age"].isna() | (df["age"] < 0),
    }
    df, stats.dropped = _drop_invalid(df, checks)
    duplicate = df["machine_id"].duplicated()
    stats.dropped["duplicate"] = int(duplicate.sum())
    df = df[~duplicate]

    loc = pd.DataFrame(
        {
            "machine_id": _to_int(location["machineID"]),
            **{c: _text(location[c]) for c in ("plant_id", "plant_name", "city", "line_id")},
        }
    ).drop_duplicates("machine_id")
    df = df.merge(loc, on="machine_id", how="left", validate="one_to_one")
    stats.flagged["no_location"] = int(df["line_id"].isna().sum())
    columns = ["machine_id", "model", "age", "plant_id", "plant_name", "city", "line_id"]
    df = df[columns + [c for c in LINEAGE if c in df]].sort_values("machine_id")
    stats.rows_out = len(df)
    return df.reset_index(drop=True), stats


def build_events(
    bronze: pd.DataFrame, spec: EventSpec, known_machines: set[int], name: str
) -> tuple[pd.DataFrame, TableStats]:
    """Type, validate and deduplicate an event table (errors, maintenance, failures)."""
    stats = TableStats(name, rows_in=len(bronze))
    df = pd.DataFrame(
        {
            "timestamp": parse_datetimes(_blank_to_na(bronze["datetime"])),
            "machine_id": _to_int(bronze["machineID"]),
            spec.value_name: _text(bronze[spec.value_source]),
            **{c: bronze[c].to_numpy() for c in LINEAGE if c in bronze},
        }
    )
    checks = {
        "bad_timestamp": df["timestamp"].isna(),
        "unknown_machine": ~df["machine_id"].isin(known_machines).fillna(False),
        "unknown_value": ~df[spec.value_name].isin(spec.allowed),
    }
    df, stats.dropped = _drop_invalid(df, checks)
    keys = ["timestamp", "machine_id", spec.value_name]
    duplicate = df.duplicated(keys)
    stats.dropped["duplicate"] = int(duplicate.sum())
    df = df[~duplicate].sort_values(keys).reset_index(drop=True)
    stats.rows_out = len(df)
    return df, stats


def build_component_costs(bronze: pd.DataFrame) -> tuple[pd.DataFrame, TableStats]:
    stats = TableStats("component_costs", rows_in=len(bronze))
    df = pd.DataFrame({"component": _text(bronze["component"])})
    for col in COST_NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(_blank_to_na(bronze[col]), errors="coerce").astype("float64")
    df["currency"] = _text(bronze["currency"])
    for col in LINEAGE:
        if col in bronze:
            df[col] = bronze[col].to_numpy()
    checks = {
        "bad_value": df[COST_NUMERIC_COLUMNS].isna().any(axis=1)
        | (df[COST_NUMERIC_COLUMNS] < 0).any(axis=1),
        "unknown_value": df["component"] == "",
    }
    df, stats.dropped = _drop_invalid(df, checks)
    duplicate = df["component"].duplicated()
    stats.dropped["duplicate"] = int(duplicate.sum())
    df = df[~duplicate].reset_index(drop=True)
    stats.rows_out = len(df)
    return df, stats


def _drop_invalid(
    df: pd.DataFrame, checks: dict[str, pd.Series]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop rows failing any check; each row is counted once, under its first failed check."""
    dropped: dict[str, int] = {}
    keep = pd.Series(True, index=df.index)
    for reason, bad in checks.items():
        newly_bad = keep & bad.fillna(True).astype(bool)
        dropped[reason] = int(newly_bad.sum())
        keep &= ~newly_bad
    return df[keep].copy(), dropped


def event_specs() -> dict[str, EventSpec]:
    components = frozenset(get_settings().business.components)
    errors = CONTRACTS["errors"].categories["errorID"]
    return {
        "errors": EventSpec("errors", "errorID", "error_id", errors),
        "maintenance": EventSpec("maintenance", "comp", "component", components),
        "failures": EventSpec("failures", "failure", "component", components),
    }


def build_tables(
    bronze_dir: Path | None = None, silver_dir: Path | None = None
) -> dict[str, TableStats]:
    """Build every non-telemetry Silver table from Bronze and write it to data/silver/."""
    results: dict[str, TableStats] = {}

    machines, results["machines"] = build_machines(
        read_bronze("machines", bronze_dir), read_bronze("machine_location", bronze_dir)
    )
    write_silver(machines, "machines", silver_dir)
    known = set(machines["machine_id"].astype(int))

    for name, spec in event_specs().items():
        events, results[name] = build_events(
            read_bronze(spec.bronze_table, bronze_dir), spec, known, name
        )
        write_silver(events, name, silver_dir)

    costs, results["component_costs"] = build_component_costs(
        read_bronze("component_costs", bronze_dir)
    )
    write_silver(costs, "component_costs", silver_dir)

    for stats in results.values():
        dropped = {k: v for k, v in stats.dropped.items() if v}
        flagged = {k: v for k, v in stats.flagged.items() if v}
        logger.info(
            "%-16s %6s rows in -> %6s rows out%s%s",
            stats.table,
            f"{stats.rows_in:,}",
            f"{stats.rows_out:,}",
            f"  dropped: {dropped}" if dropped else "",
            f"  flagged: {flagged}" if flagged else "",
        )
    return results
