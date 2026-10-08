"""Gold business KPIs: reliability, availability and cost.

Events (from Silver, inside the observed period of the telemetry):
    failure event        all component failures of one machine at one time.
                         Unplanned downtime = the LONGEST repair among the failed
                         components (they are repaired in parallel); cost = the SUM
                         of their unplanned failure costs.
    planned maintenance  replacements that are NOT the repair of a failure (a
                         maintenance record with the same machine, time and component
                         as a failure is that failure's repair, so it is not counted
                         again). Grouped per machine and time like failures, with
                         planned downtime and planned maintenance costs.

KPIs per machine and month (downtime is booked in the month the event starts):
    period_hours          hours of the month inside the observed period
    failures, components_failed, planned_maintenances
    unplanned_downtime_h, planned_downtime_h, downtime_h
    availability          1 - downtime_h / period_hours
    mtbf_h                (period_hours - downtime_h) / failures  (empty without failures)
    mttr_h                unplanned_downtime_h / failures         (empty without failures)
    failure_cost, maintenance_cost, total_cost

Line, plant and whole-period tables are aggregated by SUMMING the counts,
hours and costs and then recomputing the ratios (never by averaging ratios).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from iiot.config import get_settings
from iiot.utils.io import write_parquet_atomic
from iiot.utils.logger import get_logger

logger = get_logger("iiot.gold.kpis")

ADDITIVE = [
    "period_hours",
    "failures",
    "components_failed",
    "planned_maintenances",
    "unplanned_downtime_h",
    "planned_downtime_h",
    "downtime_h",
    "failure_cost",
    "maintenance_cost",
    "total_cost",
]
TABLES = {
    "kpi_machine_monthly": ["machine_id", "model", "plant_id", "line_id", "month"],
    "kpi_machine_total": ["machine_id", "model", "plant_id", "line_id"],
    "kpi_line_monthly": ["plant_id", "line_id", "month"],
    "kpi_plant_monthly": ["plant_id", "month"],
}


@dataclass
class KpiStats:
    failure_events: int = 0
    component_failures: int = 0
    planned_events: int = 0
    repair_records_excluded: int = 0
    outside_period_excluded: int = 0
    rows: dict[str, int] = field(default_factory=dict)


def _events(
    records: pd.DataFrame, costs: pd.DataFrame, hours_col: str, cost_col: str
) -> pd.DataFrame:
    """Group component records into events per (machine, time): max downtime, summed cost."""
    merged = records.merge(costs[["component", hours_col, cost_col]], on="component", how="left")
    if merged[hours_col].isna().any():
        unknown = sorted(merged.loc[merged[hours_col].isna(), "component"].unique())
        raise ValueError(f"No cost data for components: {unknown}")
    return merged.groupby(["machine_id", "timestamp"], as_index=False).agg(
        components=("component", "size"), downtime_h=(hours_col, "max"), cost=(cost_col, "sum")
    )


def classify_events(
    failures: pd.DataFrame,
    maintenance: pd.DataFrame,
    costs: pd.DataFrame,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, KpiStats]:
    """Return (failure_events, planned_events) inside [period_start, period_end)."""
    stats = KpiStats()
    in_period = lambda df: (df["timestamp"] >= period_start) & (df["timestamp"] < period_end)  # noqa: E731
    stats.outside_period_excluded = int(
        (~in_period(failures)).sum() + (~in_period(maintenance)).sum()
    )
    failures, maintenance = failures[in_period(failures)], maintenance[in_period(maintenance)]

    keys = ["machine_id", "timestamp", "component"]
    repair = maintenance.merge(
        failures[keys].drop_duplicates(), on=keys, how="left", indicator=True
    )
    is_repair = (repair["_merge"] == "both").to_numpy()
    stats.repair_records_excluded = int(is_repair.sum())
    planned = maintenance[~is_repair]

    failure_events = _events(failures, costs, "unplanned_downtime_hours", "unplanned_failure_cost")
    planned_events = _events(planned, costs, "planned_downtime_hours", "planned_maintenance_cost")
    stats.failure_events, stats.planned_events = len(failure_events), len(planned_events)
    stats.component_failures = len(failures)
    return failure_events, planned_events, stats


def period_hours_by_month(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Hours of each calendar month inside [start, end)."""
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    return pd.Series(1, index=hours).groupby(hours.to_period("M")).sum().rename("period_hours")


def add_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        df["availability"] = 1 - df["downtime_h"] / df["period_hours"]
        failures = df["failures"].where(df["failures"] > 0)
        df["mtbf_h"] = (df["period_hours"] - df["downtime_h"]) / failures
        df["mttr_h"] = df["unplanned_downtime_h"] / failures
    return df


def machine_monthly(
    failure_events: pd.DataFrame,
    planned_events: pd.DataFrame,
    machines: pd.DataFrame,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
) -> pd.DataFrame:
    """One row per machine and month, including months without any event."""
    months = period_hours_by_month(period_start, period_end)
    grid = machines[["machine_id", "model", "plant_id", "line_id"]].merge(
        months.rename_axis("month").reset_index(), how="cross"
    )

    def monthly(events: pd.DataFrame, prefix: str) -> pd.DataFrame:
        e = events.assign(month=events["timestamp"].dt.to_period("M"))
        return e.groupby(["machine_id", "month"], as_index=False).agg(
            **{
                f"{prefix}_events": ("timestamp", "size"),
                f"{prefix}_components": ("components", "sum"),
                f"{prefix}_downtime": ("downtime_h", "sum"),
                f"{prefix}_cost": ("cost", "sum"),
            }
        )

    df = grid.merge(monthly(failure_events, "f"), on=["machine_id", "month"], how="left").merge(
        monthly(planned_events, "p"), on=["machine_id", "month"], how="left"
    )
    df = df.fillna({c: 0 for c in df.columns if c[:2] in ("f_", "p_")})
    out = df[["machine_id", "model", "plant_id", "line_id", "month", "period_hours"]].copy()
    out["failures"] = df["f_events"].astype(int)
    out["components_failed"] = df["f_components"].astype(int)
    out["planned_maintenances"] = df["p_events"].astype(int)
    out["unplanned_downtime_h"] = df["f_downtime"]
    out["planned_downtime_h"] = df["p_downtime"]
    out["downtime_h"] = out["unplanned_downtime_h"] + out["planned_downtime_h"]
    out["failure_cost"] = df["f_cost"]
    out["maintenance_cost"] = df["p_cost"]
    out["total_cost"] = out["failure_cost"] + out["maintenance_cost"]
    out["month"] = out["month"].astype(str)
    return add_ratios(out)


def aggregate(monthly: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Sum the additive columns over `keys`, then recompute the ratios."""
    summed = monthly.groupby(keys, as_index=False, observed=True)[ADDITIVE].sum()
    return add_ratios(summed)


def build_kpis(
    failures: pd.DataFrame,
    maintenance: pd.DataFrame,
    machines: pd.DataFrame,
    costs: pd.DataFrame,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
) -> tuple[dict[str, pd.DataFrame], KpiStats]:
    failure_events, planned_events, stats = classify_events(
        failures, maintenance, costs, period_start, period_end
    )
    monthly = machine_monthly(failure_events, planned_events, machines, period_start, period_end)
    tables = {"kpi_machine_monthly": monthly}
    for name, keys in TABLES.items():
        if name != "kpi_machine_monthly":
            tables[name] = aggregate(monthly, keys)
    stats.rows = {name: len(df) for name, df in tables.items()}
    return tables, stats


def build(
    silver_dir: Path | None = None, gold_dir: Path | None = None
) -> tuple[dict[str, pd.DataFrame], KpiStats]:
    """Silver events -> data/gold/kpi_*.parquet."""
    settings = get_settings()
    silver_dir = Path(silver_dir or settings.paths.silver)
    gold_dir = Path(gold_dir or settings.paths.gold)
    inputs = {}
    for name in ("failures", "maintenance", "machines", "component_costs", "telemetry"):
        path = silver_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `iiot silver build` first.")
        columns = ["timestamp"] if name == "telemetry" else None
        inputs[name] = pd.read_parquet(path, columns=columns)

    # Observed period: from the first telemetry reading up to midnight of the last day,
    # so a few trailing hours never form a tiny extra month (for this dataset:
    # 2015-01-01 06:00 to 2016-01-01 00:00, 12 calendar months).
    period_start = inputs["telemetry"]["timestamp"].min()
    period_end = inputs["telemetry"]["timestamp"].max().normalize()
    tables, stats = build_kpis(
        inputs["failures"],
        inputs["maintenance"],
        inputs["machines"],
        inputs["component_costs"],
        period_start,
        period_end,
    )
    for name, df in tables.items():
        write_parquet_atomic(df, gold_dir / f"{name}.parquet")

    total = aggregate(tables["kpi_machine_monthly"].assign(all="all"), ["all"]).iloc[0]
    currency = inputs["component_costs"]["currency"].iloc[0]
    logger.info(
        "KPIs (%s to %s): %d failure events (%d component failures), %d planned maintenances "
        "(%d repair records not double-counted)",
        period_start,
        period_end,
        stats.failure_events,
        stats.component_failures,
        stats.planned_events,
        stats.repair_records_excluded,
    )
    logger.info(
        "  fleet availability %.2f%%, MTBF %.0f h, MTTR %.1f h, downtime %s h, total cost %s %s",
        100 * total["availability"],
        total["mtbf_h"],
        total["mttr_h"],
        f"{total['downtime_h']:,.0f}",
        currency,
        f"{total['total_cost']:,.0f}",
    )
    logger.info("  tables: %s", stats.rows)
    return tables, stats
