"""Compute every business insight from the warehouse and the models: reports/insights.json.

Inputs (all produced by `iiot run`): the warehouse (KPIs, failures, components,
lines, machines, predictions) and the failure model card (test results, used for
the "as tested" scenario). Assumptions come from config/settings.yaml (`insights`,
`ml.false_alarm_cost`, `business`).

Usage:
    from iiot.insights.build import build_insights
    insights = build_insights()
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from iiot.config import get_settings
from iiot.dashboard import data
from iiot.insights import calculations as calc
from iiot.utils.logger import get_logger

logger = get_logger("iiot.insights")

INSIGHTS_FILE_NAME = "insights.json"
AS_TESTED = "as tested"
SENSITIVITY_CATCH_RATES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SENSITIVITY_ALARM_COSTS = [10_000, 20_000, 30_000, 50_000]
# Failures inside the analysis period (as the KPIs and views count them).
PERIOD_FAILURES = """
    WITH period AS (
        SELECT min(timestamp) AS period_start, date_trunc('day', max(timestamp)) AS period_end
        FROM fact_sensor_reading
    )
    SELECT f.machine_id, f.timestamp, f.component
    FROM fact_failure AS f, period AS p
    WHERE f.timestamp >= p.period_start AND f.timestamp < p.period_end
"""


def _records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records", date_format="iso"))


def build_insights(
    path: Path | None = None,
    models_dir: Path | None = None,
    reports_dir: Path | None = None,
) -> dict:
    """Compute the business case and where to focus; write and return insights.json."""
    settings = get_settings()
    assumptions = settings.insights
    alarm_cost = settings.ml.false_alarm_cost
    reports_dir = Path(reports_dir or settings.paths.reports)

    cards = data.model_cards(models_dir)
    if "failure" not in cards:
        raise FileNotFoundError("No failure model card. Run `iiot ml train` first.")
    tested = cards["failure"]["test"]

    components = data.query("SELECT * FROM dim_component ORDER BY component", path=path)
    hours = data.query(
        "SELECT sum(period_hours) AS h, count(DISTINCT machine_id) AS m FROM fact_machine_month",
        path=path,
    ).iloc[0]
    year = calc.failure_events(
        data.query(PERIOD_FAILURES, path=path), components, hours["h"], int(hours["m"])
    )
    planned_hours = components.set_index("component")["planned_downtime_hours"]

    scenarios = {
        name: (s.catch_rate, s.false_alarms_per_catch) for name, s in assumptions.scenarios.items()
    }
    scenarios[AS_TESTED] = (
        tested["events_caught"] / tested["failure_events"],
        tested["false_alarms"] / max(tested["events_caught"], 1),
    )
    results = [
        calc.scenario(
            name,
            rate,
            alarms,
            year,
            alarm_cost,
            assumptions.program_cost_per_year,
            assumptions.inspection_hours,
            planned_hours,
        )
        for name, (rate, alarms) in scenarios.items()
    ]
    reference = assumptions.scenarios.get("expected") or next(iter(assumptions.scenarios.values()))
    grid = calc.sensitivity(
        year,
        SENSITIVITY_CATCH_RATES,
        SENSITIVITY_ALARM_COSTS,
        reference.false_alarms_per_catch,
        assumptions.program_cost_per_year,
    )

    health = data.machine_health(path)
    machines, top_share = calc.focus_machines(health)
    alerts = data.query(
        """
        SELECT predicted_component, reason_1, reason_2, reason_3 FROM fact_prediction
        WHERE out_of_sample AND alert
        """,
        path=path,
    )
    insights = {
        "generated_at": datetime.now(UTC).isoformat(),
        "baseline": data.fleet_kpis(path),
        "base_year": {
            "years": year.years,
            "failure_events": year.events,
            "component_failures": {k: int(v) for k, v in year.component_failures.items()},
            "saving_if_every_failure_caught": year.saving_if_caught / year.years,
            "downtime_avoidable_h": year.downtime_avoided_h / year.years,
        },
        "scenarios": [r.to_dict() for r in results],
        "break_even_catch_rate": calc.break_even_catch_rate(
            year, reference.false_alarms_per_catch, alarm_cost, assumptions.program_cost_per_year
        ),
        "sensitivity": {
            "false_alarms_per_catch": reference.false_alarms_per_catch,
            "catch_rates": SENSITIVITY_CATCH_RATES,
            "false_alarm_costs": SENSITIVITY_ALARM_COSTS,
            "net_saving": grid.to_numpy().tolist(),
        },
        "focus": {
            "lines": _records(calc.focus_lines(data.line_performance(path))),
            "models": _records(calc.focus_groups(health, "model")),
            "age_bands": _records(calc.focus_groups(health, "age_band")),
            "costliest_machines": _records(machines),
            "costliest_10_share_of_failure_cost": top_share,
            "components": _records(calc.focus_components(data.component_reliability(path))),
        },
        "warning_signatures": _records(calc.warning_signatures(alerts)),
        "test_period": {
            k: tested[k] for k in ("events_caught", "failure_events", "false_alarms", "net_saving")
        },
        "assumptions": {
            "currency": settings.business.currency,
            "false_alarm_cost": alarm_cost,
            "program_cost_per_year": assumptions.program_cost_per_year,
            "inspection_hours": assumptions.inspection_hours,
            "downtime_cost_per_hour": settings.business.downtime_cost_per_hour,
            "scenarios": {
                k: {"catch_rate": v[0], "false_alarms_per_catch": v[1]}
                for k, v in scenarios.items()
            },
        },
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / INSIGHTS_FILE_NAME
    out.write_text(json.dumps(insights, indent=2, default=str), encoding="utf-8")
    _log_summary(insights, out)
    return insights


def _log_summary(insights: dict, out: Path) -> None:
    for s in insights["scenarios"]:
        logger.info(
            "%-12s catch %3.0f%%: net saving INR %6.2f cr/year, downtime avoided %5.0f h, "
            "payback %s",
            s["name"],
            100 * s["catch_rate"],
            s["net_saving"] / 1e7,
            s["downtime_avoided_h"],
            "never" if s["payback_months"] is None else f"{s['payback_months']:.1f} months",
        )
    logger.info("Insights written to %s", out)
