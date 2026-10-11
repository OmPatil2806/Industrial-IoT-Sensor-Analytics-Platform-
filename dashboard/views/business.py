"""Business case: what predictive maintenance is worth, with your own assumptions."""

import pandas as pd
import streamlit as st

from iiot.config import get_settings
from iiot.dashboard import charts, ui
from iiot.insights import calculations as calc

st.title("💼 Business case")
ui.require_warehouse()
st.caption(
    "What catching failures 24 hours ahead is worth in a year like 2015. A caught failure "
    "becomes a planned replacement (cheaper, shorter downtime); every false alarm costs one "
    "inspection. Pick a starting scenario, then move the sliders. The full write-up is in "
    "`reports/business_insights.md`."
)

settings = get_settings()
assumptions = settings.insights
year, planned_hours = ui.base_year()

presets = {
    name: (s.catch_rate, s.false_alarms_per_catch) for name, s in assumptions.scenarios.items()
}
tested = ui.model_cards().get("failure", {}).get("test")
if tested and tested.get("failure_events"):
    presets["as tested"] = (
        tested["events_caught"] / tested["failure_events"],
        tested["false_alarms"] / max(tested["events_caught"], 1),
    )
names = list(presets)
default = names.index("expected") if "expected" in names else 0
preset = st.radio(
    "Start from",
    names,
    index=default,
    horizontal=True,
    help="the scenarios in config/settings.yaml; 'as tested' is what the model achieved "
    "on the test months",
)
start_rate, start_alarms = presets[preset]

# Keys depend on the preset, so picking another preset resets the sliders to it.
c = st.columns(4)
catch_rate = (
    c[0].slider(
        "Failures caught (%)",
        1,
        100,
        round(100 * start_rate),
        key=f"rate_{preset}",
        help="share of failure events flagged at least 24 h ahead",
    )
    / 100
)
alarms = c[1].slider(
    "False alarms per catch",
    0.0,
    3.0,
    round(start_alarms, 2),
    step=0.01,
    key=f"alarms_{preset}",
    help="inspections that find nothing, per failure caught",
)
alarm_cost = c[2].number_input(
    "Cost of a false alarm (₹)",
    0,
    500_000,
    int(settings.ml.false_alarm_cost),
    step=5_000,
    key=f"cost_{preset}",
)
program_lakh = c[3].number_input(
    "Programme cost (₹ lakh a year)",
    0.0,
    1_000.0,
    assumptions.program_cost_per_year / 1e5,
    step=5.0,
    key=f"program_{preset}",
    help="software, sensors and people running it",
)
program_cost = program_lakh * 1e5


def run(name: str, rate: float, per_catch: float) -> calc.ScenarioResult:
    return calc.scenario(
        name,
        rate,
        per_catch,
        year,
        alarm_cost,
        program_cost,
        assumptions.inspection_hours,
        planned_hours,
    )


mine = run("your scenario", catch_rate, alarms)
break_even = calc.break_even_catch_rate(year, alarms, alarm_cost, program_cost)

m = st.columns(4)
m[0].metric(
    "Net saving a year",
    ui.inr(mine.net_saving),
    help="saving on caught failures - false-alarm inspections - programme cost",
)
m[1].metric(
    "Downtime avoided",
    ui.hours(mine.downtime_avoided_h),
    help=f"+{mine.availability_gain:.2f} percentage points of fleet availability",
)
m[2].metric(
    "Payback",
    "never"
    if mine.payback_months is None
    else ("under 1 month" if mine.payback_months < 1 else f"{mine.payback_months:.1f} months"),
)
m[3].metric(
    "Break-even",
    "never" if break_even is None else ui.pct(break_even),
    help="the share of failures to catch for the programme to pay for itself",
)
m = st.columns(4)
m[0].metric(
    "Failures caught",
    f"{mine.failures_caught:,.0f}",
    help=f"of {year.events / year.years:,.0f} failure events a year",
)
m[1].metric("False alarms", f"{mine.false_alarms:,.0f}")
m[2].metric("Inspections a week", f"{mine.inspections_per_week:.1f}")
m[3].metric(
    "Technician hours a week",
    ui.hours(mine.technician_hours_per_week),
    help="inspections plus planned replacements",
)

rates = [r / 100 for r in range(0, 101)]
grid = calc.sensitivity(year, rates, [alarm_cost], alarms, program_cost)
curve = pd.DataFrame({"catch_rate": grid.index, "net_saving_cr": grid[alarm_cost] / 1e7})
st.subheader("Net saving per year vs share of failures caught")
st.caption("The dot is your scenario; the dashed line is the break-even catch rate.")
st.altair_chart(charts.saving_curve(curve, catch_rate, break_even), use_container_width=True)

left, right = st.columns([3, 2])
with left:
    st.subheader("Compared with the scenarios")
    rows = [run(n, r, a) for n, (r, a) in presets.items()] + [mine]
    st.dataframe(
        pd.DataFrame(
            {
                "Scenario": [r.name for r in rows],
                "Caught": [ui.pct(r.catch_rate, 0) for r in rows],
                "False alarms per catch": [f"{r.false_alarms_per_catch:.2f}" for r in rows],
                "Net saving": [ui.inr(r.net_saving) for r in rows],
                "Downtime avoided": [ui.hours(r.downtime_avoided_h) for r in rows],
            }
        ),
        hide_index=True,
    )
    st.caption("All rows use the false-alarm and programme costs set above.")
with right:
    st.subheader("Planned replacements a quarter")
    st.dataframe(
        pd.DataFrame(
            {
                "Component": list(mine.parts_per_quarter),
                "Replacements": [f"{v:.0f}" for v in mine.parts_per_quarter.values()],
            }
        ),
        hide_index=True,
    )
