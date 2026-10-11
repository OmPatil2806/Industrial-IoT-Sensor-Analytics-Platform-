"""Write the business report: reports/business_insights.md with charts in reports/figures/.

Everything in the report comes from reports/insights.json (see iiot.insights.build),
so the text, tables and charts always agree. The report is committed to the
repository, so it renders on GitHub.

Usage:
    from iiot.insights.report import write_report
    write_report()            # reads reports/insights.json
"""

from __future__ import annotations

import json
from pathlib import Path

from iiot.config import get_settings
from iiot.insights import figures
from iiot.insights.build import INSIGHTS_FILE_NAME
from iiot.utils.logger import get_logger

logger = get_logger("iiot.insights.report")

REPORT_FILE_NAME = "business_insights.md"
FIGURES_DIR = "figures"
YOUNG_BANDS = ("0-4 years", "5-9 years")


def inr(value: float) -> str:
    """Indian-style amount: crore above 1 crore, lakh above 1 lakh, else rupees."""
    if abs(value) >= 1e7:
        return f"₹{value / 1e7:,.2f} cr"
    if abs(value) >= 1e5:
        return f"₹{value / 1e5:,.1f} lakh"
    return f"₹{value:,.0f}"


def pct(value: float, digits: int = 0) -> str:
    return f"{100 * value:.{digits}f}%"


def months(value: float | None) -> str:
    if value is None:
        return "never"
    return "under 1 month" if value < 1 else f"{value:.1f} months"


def plural(n: float, word: str) -> str:
    return f"{n:g} {word}" + ("" if n == 1 else "s")


def table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _scenario(insights: dict, name: str) -> dict:
    by_name = {s["name"]: s for s in insights["scenarios"]}
    return by_name.get(name) or insights["scenarios"][0]


def _young_vs_old(bands: list[dict]) -> tuple[dict, float]:
    """The oldest age band, and failures per machine of machines under 10 years."""
    young = [b for b in bands if b["age_band"] in YOUNG_BANDS]
    machines = sum(b["machines"] for b in young)
    rate = sum(b["failures"] for b in young) / machines if machines else float("nan")
    return bands[-1], rate


def _signatures(insights: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for s in insights["warning_signatures"]:
        out.setdefault(s["component"], []).append(s)
    return out


def render(insights: dict) -> str:
    """The report as markdown (figures are referenced as figures/<name>.png)."""
    base, year = insights["baseline"], insights["base_year"]
    expected = _scenario(insights, "expected")
    conservative = _scenario(insights, "conservative")
    focus = insights["focus"]
    hot_lines = [ln for ln in focus["lines"] if ln["above_fleet_average"]]
    top_line = focus["lines"][0]
    oldest, young_rate = _young_vs_old(focus["age_bands"])
    by_rate = sorted(focus["models"], key=lambda m: m["failures_per_machine"])
    best_model, worst_model = by_rate[0], by_rate[-1]
    top_component = focus["components"][0]
    signatures = _signatures(insights)
    break_even = insights["break_even_catch_rate"]
    a = insights["assumptions"]
    parts = expected["parts_per_quarter"]

    md = [
        "# Industrial IoT Sensor Analytics: business insights",
        "",
        f"_Generated {insights['generated_at'][:10]} from `reports/insights.json` "
        "(`iiot.insights`). All amounts in INR; 1 cr = 1 crore = ₹1,00,00,000._",
        "",
        "## Executive summary",
        "",
        f"In the year analysed, the fleet of {base['machines']} machines had "
        f"**{base['failures']} unplanned failures** costing **{inr(base['failure_cost'])}** "
        f"and **{base['downtime_h']:,.0f} hours** of downtime. Predicting failures 24 hours "
        "ahead turns most of them into planned replacements, which are cheaper and faster.",
        "",
        table(
            ["Key number", "Value"],
            [
                ["Unplanned failure cost today", f"{inr(base['failure_cost'])} a year"],
                [
                    f"Net saving, expected case ({pct(expected['catch_rate'])} caught)",
                    f"**{inr(expected['net_saving'])} a year**",
                ],
                [
                    "Downtime avoided, expected case",
                    f"{expected['downtime_avoided_h']:,.0f} h a year "
                    f"(+{expected['availability_gain']:.2f} pts availability)",
                ],
                ["Payback of the programme", months(expected["payback_months"])],
                [
                    "Break-even: share of failures to catch",
                    pct(break_even, 1) if break_even is not None else "never",
                ],
            ],
        ),
        "",
        "**Recommendations**",
        "",
        f"1. **Roll out the 24-hour failure alerts.** Even the conservative case "
        f"({pct(conservative['catch_rate'])} caught, {conservative['false_alarms_per_catch']:g} "
        f"false alarm per catch) saves {inr(conservative['net_saving'])} a year; the "
        f"programme pays for itself once it catches {pct(break_even, 1)} of failures."
        if break_even is not None
        else "1. **Do not roll out yet:** the programme does not pay back under these assumptions.",
        f"2. **Start with {top_line['line_id']}** ({top_line['plant_name']}): "
        f"{top_line['failures_per_machine']:.2f} failures per machine and "
        f"{pct(top_line['failure_cost_share'])} of the failure cost. "
        f"{len(hot_lines)} lines are above the fleet average: "
        f"{', '.join(ln['line_id'] for ln in hot_lines)}.",
        f"3. **Prioritise older machines and {worst_model['model']}.** Machines aged "
        f"{oldest['age_band']} fail {oldest['failures_per_machine']:.1f} times a year, against "
        f"{young_rate:.1f} for machines under 10 years; {worst_model['model']} fails "
        f"{worst_model['failures_per_machine']:.1f} times per machine against "
        f"{best_model['failures_per_machine']:.1f} for {best_model['model']}. "
        "Review them for overhaul or replacement.",
        "4. **Plan spare parts by quarter.** In the expected case about "
        + ", ".join(f"{v:.0f} {c}" for c, v in parts.items())
        + f" replacements a quarter; {top_component['component']} alone is "
        f"{pct(top_component['failure_cost_share'])} of the failure cost.",
        "5. **Put the warning signs into inspection checklists** "
        "(see [Warning signatures](#warning-signatures)): each component fails "
        "with its own recognisable pattern.",
        "",
        "## Today: what failures cost",
        "",
        table(
            ["Measure", "Value"],
            [
                ["Machines", base["machines"]],
                ["Unplanned failures", base["failures"]],
                [
                    "Component failures",
                    ", ".join(f"{c} {n}" for c, n in year["component_failures"].items()),
                ],
                ["Downtime", f"{base['downtime_h']:,.0f} h"],
                ["Fleet availability", pct(base["availability"], 2)],
                ["Mean time between failures", f"{base['mtbf_h']:,.0f} h"],
                ["Mean time to repair", f"{base['mttr_h']:.1f} h"],
                ["Failure cost", inr(base["failure_cost"])],
                ["Planned maintenance cost", inr(base["maintenance_cost"])],
                [
                    "Saving if every failure were caught in time",
                    inr(year["saving_if_every_failure_caught"]),
                ],
            ],
        ),
        "",
        "## Business case",
        "",
        "Each scenario sets the share of failures caught 24 hours ahead and how many false "
        f"alarms come with each catch. A caught failure becomes a planned replacement; each "
        f"false alarm costs one inspection ({inr(a['false_alarm_cost'])}); the programme "
        f'costs {inr(a["program_cost_per_year"])} a year. "As tested" uses what the model '
        "actually achieved on the test months it never saw.",
        "",
        "![Net saving per year by scenario](figures/scenario_savings.png)",
        "",
        table(
            ["Per year", *[s["name"] for s in insights["scenarios"]]],
            [
                ["Failures caught", *[pct(s["catch_rate"]) for s in insights["scenarios"]]],
                [
                    "False alarms per catch",
                    *[f"{s['false_alarms_per_catch']:.2f}" for s in insights["scenarios"]],
                ],
                [
                    "Failures caught (count)",
                    *[f"{s['failures_caught']:.0f}" for s in insights["scenarios"]],
                ],
                ["Gross saving", *[inr(s["gross_saving"]) for s in insights["scenarios"]]],
                ["False-alarm cost", *[inr(s["false_alarm_cost"]) for s in insights["scenarios"]]],
                ["Programme cost", *[inr(s["program_cost"]) for s in insights["scenarios"]]],
                ["**Net saving**", *[f"**{inr(s['net_saving'])}**" for s in insights["scenarios"]]],
                [
                    "Downtime avoided",
                    *[f"{s['downtime_avoided_h']:,.0f} h" for s in insights["scenarios"]],
                ],
                [
                    "Availability gain",
                    *[f"+{s['availability_gain']:.2f} pts" for s in insights["scenarios"]],
                ],
                ["Payback", *[months(s["payback_months"]) for s in insights["scenarios"]]],
            ],
        ),
        "",
        "### How sure is the saving?",
        "",
        "Net saving for other catch rates and inspection costs "
        f"({insights['sensitivity']['false_alarms_per_catch']:g} false alarms per catch). "
        + (
            "It stays positive across the whole range: the result does not hinge on the "
            "model being as good as it tested."
            if min(min(row) for row in insights["sensitivity"]["net_saving"]) > 0
            else "Below some catch rates the programme loses money; see the table."
        ),
        "",
        "![Net saving vs share of failures caught](figures/sensitivity.png)",
        "",
        table(
            [
                "Caught",
                *[f"{inr(c)} / false alarm" for c in insights["sensitivity"]["false_alarm_costs"]],
            ],
            [
                [pct(r), *[inr(v) for v in row]]
                for r, row in zip(
                    insights["sensitivity"]["catch_rates"],
                    insights["sensitivity"]["net_saving"],
                    strict=True,
                )
            ],
        ),
        "",
        "## Where to focus",
        "",
        "### Production lines",
        "",
        "![Failures per machine by production line](figures/lines.png)",
        "",
        table(
            ["Line", "Plant", "Failures per machine", "Share of failure cost", "Availability"],
            [
                [
                    f"**{ln['line_id']}**" if ln["above_fleet_average"] else ln["line_id"],
                    ln["plant_name"],
                    f"{ln['failures_per_machine']:.2f}",
                    pct(ln["failure_cost_share"], 1),
                    pct(ln["availability"], 2),
                ]
                for ln in focus["lines"]
            ],
        ),
        "",
        "Lines in bold are above the fleet average.",
        "",
        "### Machine age and model",
        "",
        "![Failures per machine by machine age](figures/age_bands.png)",
        "",
        table(
            ["Model", "Machines", "Failures per machine", "Cost per machine"],
            [
                [
                    m["model"],
                    m["machines"],
                    f"{m['failures_per_machine']:.1f}",
                    inr(m["cost_per_machine"]),
                ]
                for m in focus["models"]
            ],
        ),
        "",
        "### Costliest machines",
        "",
        f"The 10 costliest machines cause "
        f"{pct(focus['costliest_10_share_of_failure_cost'], 1)} of the failure cost "
        "(an even spread would be 10%).",
        "",
        table(
            ["Machine", "Model", "Age", "Line", "Failures", "Total cost"],
            [
                [
                    m["machine_id"],
                    m["model"],
                    m["age"],
                    m["line_id"],
                    m["failures"],
                    inr(m["total_cost"]),
                ]
                for m in focus["costliest_machines"]
            ],
        ),
        "",
        "### Components",
        "",
        "![Share of the failure cost by component](figures/components.png)",
        "",
        table(
            [
                "Component",
                "Failures",
                "Cost per failure",
                "Share of failure cost",
                "Share of possible saving",
            ],
            [
                [
                    c["component"],
                    c["failures"],
                    inr(c["cost_per_failure"]),
                    pct(c["failure_cost_share"], 1),
                    pct(c["saving_share"], 1),
                ]
                for c in focus["components"]
            ],
        ),
        "",
        "## Warning signatures",
        "",
        "The inputs most often named in the reasons for out-of-sample alerts, per predicted "
        "component. Technicians can check these first.",
        "",
        table(
            ["Component", "Warning signs (share of its alerts)"],
            [
                [c, "; ".join(f"{s['signal']} ({pct(s['share_of_alerts'])})" for s in sigs)]
                for c, sigs in signatures.items()
            ],
        ),
        "",
        "## What it means for operations (expected case)",
        "",
        table(
            ["Measure", "Value"],
            [
                ["Inspections per week", f"{expected['inspections_per_week']:.1f}"],
                [
                    "Technician hours per week (inspections + planned replacements)",
                    f"{expected['technician_hours_per_week']:.0f} h",
                ],
                [
                    "Planned replacements per quarter",
                    ", ".join(f"{c} {v:.0f}" for c, v in parts.items()),
                ],
            ],
        ),
        "",
        "## Next steps: a 3-month pilot",
        "",
        f"1. **Scope:** the lines above the fleet average "
        f"({', '.join(ln['line_id'] for ln in hot_lines)}), starting with "
        f"{top_line['line_id']}.",
        "2. **Run:** send the 24-hour alerts and their reasons to the line's maintenance "
        "team every shift; log each inspection as a catch or a false alarm.",
        f"3. **Go / no-go:** continue to all lines if the pilot catches at least "
        f"{pct(conservative['catch_rate'])} of failures with no more than "
        f"{conservative['false_alarms_per_catch']:g} false alarm per catch (the conservative "
        "case, which already pays back).",
        "4. **Keep the model honest:** retrain when the data drifts (`iiot run` checks the "
        "data fingerprint) and track every run in MLflow.",
        "",
        "## Assumptions and limits",
        "",
        f"- Costs per component come from `config/settings.yaml` (parts, labour and "
        f"{inr(a['downtime_cost_per_hour'])} per hour of downtime); a false alarm costs "
        f"{inr(a['false_alarm_cost'])} and {a['inspection_hours']:g} technician hours; the "
        f"programme costs {inr(a['program_cost_per_year'])} a year.",
        "- A failure event is all components of one machine failing at one time; its downtime "
        "is the longest repair, which becomes the longest planned replacement when caught.",
        "- The data are Microsoft's synthetic Azure Predictive Maintenance dataset (one year, "
        '100 machines). Its failures follow strong patterns, so the "as tested" result '
        f"({pct(_scenario(insights, 'as tested')['catch_rate'])} caught) is better than a "
        "real plant would see: plan with the expected case.",
        f"- Test period: {insights['test_period']['events_caught']} of "
        f"{insights['test_period']['failure_events']} failures caught with "
        f"{plural(insights['test_period']['false_alarms'], 'false alarm')}.",
        "",
    ]
    return "\n".join(md)


FIGURES = {
    "scenario_savings": lambda i: figures.scenario_savings(i["scenarios"]),
    "sensitivity": lambda i: figures.sensitivity(i["sensitivity"]),
    "lines": lambda i: figures.lines(i["focus"]["lines"]),
    "age_bands": lambda i: figures.age_bands(i["focus"]["age_bands"]),
    "components": lambda i: figures.components(i["focus"]["components"]),
}


def write_report(insights: dict | None = None, reports_dir: Path | None = None) -> Path:
    """Write business_insights.md and its figures; return the report path."""
    reports_dir = Path(reports_dir or get_settings().paths.reports)
    if insights is None:
        source = reports_dir / INSIGHTS_FILE_NAME
        if not source.exists():
            raise FileNotFoundError(f"{source} not found. Build the insights first.")
        insights = json.loads(source.read_text(encoding="utf-8"))
    figures_dir = reports_dir / FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)
    for name, make in FIGURES.items():
        # No "Software" metadata, so unchanged charts give byte-identical files.
        make(insights).savefig(figures_dir / f"{name}.png", metadata={"Software": None})
    out = reports_dir / REPORT_FILE_NAME
    out.write_text(render(insights), encoding="utf-8")
    logger.info("Business report written to %s (%s figures)", out, len(FIGURES))
    return out
