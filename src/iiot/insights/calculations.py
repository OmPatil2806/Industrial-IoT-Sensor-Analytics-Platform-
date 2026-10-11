"""The business-case arithmetic, as pure functions on small tables (all tested by hand).

Conventions (the same as the Gold KPIs and the model cards, so numbers reconcile):
- a failure EVENT is all components of one machine failing at one time
- its cost is the SUM of its components' unplanned failure costs, and the saving if
  it is caught in time is the SUM of their saving_if_prevented
  (unplanned failure cost - planned maintenance cost)
- its downtime is the LONGEST repair (repairs run in parallel); caught in time, it
  becomes the longest planned replacement instead

A scenario says what share of failure events is caught in time (catch_rate) and how
many false alarms come with each caught failure. Every false alarm costs one
inspection (ml.false_alarm_cost). The base year is the analysis period (2015).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

HOURS_PER_YEAR = 365 * 24
WEEKS_PER_YEAR = 52
QUARTERS_PER_YEAR = 4
AGE_BANDS = [
    (0, 4, "0-4 years"),
    (5, 9, "5-9 years"),
    (10, 14, "10-14 years"),
    (15, 200, "15+ years"),
]


@dataclass(frozen=True)
class FailureEvents:
    """Per failure event of the base year: its saving if caught and the downtime avoided."""

    events: int
    component_failures: pd.Series  # component -> failures in the year
    saving_if_caught: float  # sum over all events
    downtime_avoided_h: float  # sum over all events
    machine_hours: float  # machines x hours in the base year
    years: float  # length of the base year in years (about 1)


def failure_events(
    failures: pd.DataFrame, components: pd.DataFrame, machine_hours: float, machines: int
) -> FailureEvents:
    """failures: machine_id, timestamp, component; components: dim_component rows."""
    costs = components.set_index("component")
    f = failures.join(costs, on="component")
    events = f.groupby(["machine_id", "timestamp"]).agg(
        saving=("saving_if_prevented", "sum"),
        unplanned_h=("unplanned_downtime_hours", "max"),
        planned_h=("planned_downtime_hours", "max"),
    )
    return FailureEvents(
        events=len(events),
        component_failures=f["component"].value_counts().sort_index(),
        saving_if_caught=float(events["saving"].sum()),
        downtime_avoided_h=float((events["unplanned_h"] - events["planned_h"]).sum()),
        machine_hours=float(machine_hours),
        years=float(machine_hours) / machines / HOURS_PER_YEAR,
    )


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    catch_rate: float
    false_alarms_per_catch: float
    failures_caught: float
    false_alarms: float
    gross_saving: float
    false_alarm_cost: float
    program_cost: float
    net_saving: float
    downtime_avoided_h: float
    availability_gain: float  # percentage points of fleet availability
    payback_months: float | None  # None when the programme never pays back
    inspections_per_week: float
    technician_hours_per_week: float
    parts_per_quarter: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


def scenario(
    name: str,
    catch_rate: float,
    false_alarms_per_catch: float,
    year: FailureEvents,
    false_alarm_cost: float,
    program_cost: float,
    inspection_hours: float,
    planned_hours: pd.Series,
) -> ScenarioResult:
    """Annual effect of predictive maintenance for one scenario.

    planned_hours: component -> technician hours for a planned replacement."""
    per_year = 1 / year.years
    caught = catch_rate * year.events * per_year
    false_alarms = caught * false_alarms_per_catch
    gross = catch_rate * year.saving_if_caught * per_year
    alarm_cost = false_alarms * false_alarm_cost
    net = gross - alarm_cost - program_cost
    monthly_benefit = (gross - alarm_cost) / 12
    replaced = catch_rate * year.component_failures * per_year
    replacement_hours = float((replaced * planned_hours.reindex(replaced.index)).sum())
    inspections = caught + false_alarms
    return ScenarioResult(
        name=name,
        catch_rate=catch_rate,
        false_alarms_per_catch=false_alarms_per_catch,
        failures_caught=caught,
        false_alarms=false_alarms,
        gross_saving=gross,
        false_alarm_cost=alarm_cost,
        program_cost=program_cost,
        net_saving=net,
        downtime_avoided_h=catch_rate * year.downtime_avoided_h * per_year,
        availability_gain=100 * catch_rate * year.downtime_avoided_h / year.machine_hours,
        payback_months=program_cost / monthly_benefit if monthly_benefit > 0 else None,
        inspections_per_week=inspections / WEEKS_PER_YEAR,
        technician_hours_per_week=(inspections * inspection_hours + replacement_hours)
        / WEEKS_PER_YEAR,
        parts_per_quarter={c: float(n) / QUARTERS_PER_YEAR for c, n in replaced.items()},
    )


def break_even_catch_rate(
    year: FailureEvents, false_alarms_per_catch: float, false_alarm_cost: float, program_cost: float
) -> float | None:
    """Smallest catch rate at which the programme pays for itself (None if never)."""
    per_event_net = (
        (year.saving_if_caught / year.events - false_alarms_per_catch * false_alarm_cost)
        * year.events
        / year.years
    )
    if per_event_net <= 0:
        return None
    rate = program_cost / per_event_net
    return rate if rate <= 1 else None


def sensitivity(
    year: FailureEvents,
    catch_rates: list[float],
    false_alarm_costs: list[float],
    false_alarms_per_catch: float,
    program_cost: float,
) -> pd.DataFrame:
    """Net annual saving for every catch rate (rows) and false-alarm cost (columns)."""
    per_year = 1 / year.years
    table = {
        cost: [
            r * per_year * (year.saving_if_caught - year.events * false_alarms_per_catch * cost)
            - program_cost
            for r in catch_rates
        ]
        for cost in false_alarm_costs
    }
    return pd.DataFrame(table, index=pd.Index(catch_rates, name="catch_rate"))


# ------------------------------------------------------------------ where to focus


def age_band(age: int) -> str:
    return next(label for lo, hi, label in AGE_BANDS if lo <= age <= hi)


def focus_lines(lines: pd.DataFrame) -> pd.DataFrame:
    """Production lines with their share of the failure cost; above the fleet average flagged."""
    out = lines[
        [
            "plant_name",
            "line_id",
            "machines",
            "failures",
            "failures_per_machine",
            "cost_per_machine",
            "failure_cost",
            "availability",
        ]
    ].copy()
    out["failure_cost_share"] = out["failure_cost"] / out["failure_cost"].sum()
    fleet = out["failures"].sum() / out["machines"].sum()
    out["above_fleet_average"] = out["failures_per_machine"] > fleet
    return out.sort_values("failures_per_machine", ascending=False, ignore_index=True)


def focus_groups(health: pd.DataFrame, by: str) -> pd.DataFrame:
    """Failures and cost per machine for a machine attribute ("model" or "age_band")."""
    df = health.assign(age_band=health["age"].map(age_band))
    out = df.groupby(by).agg(
        machines=("machine_id", "size"),
        failures=("failures", "sum"),
        total_cost=("total_cost", "sum"),
    )
    out["failures_per_machine"] = out["failures"] / out["machines"]
    out["cost_per_machine"] = out["total_cost"] / out["machines"]
    order = [b[2] for b in AGE_BANDS] if by == "age_band" else sorted(out.index)
    return out.reindex([o for o in order if o in out.index]).reset_index()


def focus_machines(health: pd.DataFrame, top: int = 10) -> tuple[pd.DataFrame, float]:
    """The costliest machines and their share of the fleet's failure cost."""
    ranked = health.sort_values("total_cost", ascending=False)
    worst = ranked.head(top)[["machine_id", "model", "age", "line_id", "failures", "total_cost"]]
    share = float(ranked.head(top)["failure_cost"].sum() / ranked["failure_cost"].sum())
    return worst.reset_index(drop=True), share


def focus_components(components: pd.DataFrame) -> pd.DataFrame:
    """Failures, cost per failure and share of failure cost and of the possible saving."""
    out = components[["component", "failures", "failure_cost", "saving_if_all_prevented"]].copy()
    out["cost_per_failure"] = out["failure_cost"] / out["failures"].replace(0, np.nan)
    out["failure_cost_share"] = out["failure_cost"] / out["failure_cost"].sum()
    out["saving_share"] = out["saving_if_all_prevented"] / out["saving_if_all_prevented"].sum()
    return out.sort_values("failure_cost", ascending=False, ignore_index=True)


SIGNAL_WORDS = ("voltage", "rotation", "pressure", "vibration")


def reason_signal(reason: str) -> str:
    """The input named by an alert reason: a sensor, an error code or a component's age."""
    first = reason.split(" ")[0]
    if first == "errors":
        return "errors (all types)"
    if first.startswith("comp"):
        return f"{first} age"
    return first


def warning_signatures(alerts: pd.DataFrame, top: int = 2) -> pd.DataFrame:
    """For each predicted component, the inputs most often named in its alerts' reasons.

    alerts: predicted_component, reason_1, reason_2, reason_3 (out-of-sample alert rows)."""
    rows = []
    for component, group in alerts.groupby("predicted_component"):
        named = group[["reason_1", "reason_2", "reason_3"]].apply(
            lambda r: {reason_signal(x) for x in r if isinstance(x, str) and x}, axis=1
        )
        counts = pd.Series([s for signals in named for s in signals]).value_counts()
        share = counts / len(group)
        for signal, value in share.head(top).items():
            rows.append({"component": component, "signal": signal, "share_of_alerts": float(value)})
    return pd.DataFrame(rows, columns=["component", "signal", "share_of_alerts"])
