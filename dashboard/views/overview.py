"""Fleet overview: how is the fleet doing, and where does the money go?"""

import streamlit as st

from iiot.dashboard import charts, ui

st.title("📊 Fleet overview")
ui.require_warehouse()

plants = ui.plants()
choice = st.radio(
    "Plant",
    ["All plants", *plants["plant_name"]],
    horizontal=True,
    label_visibility="collapsed",
)
plant_id = (
    None if choice == "All plants" else plants.set_index("plant_name").loc[choice, "plant_id"]
)
st.caption(
    "Analysis period: 2015 (12 months). "
    "Costs are illustrative assumptions from config/settings.yaml."
)

# ---- headline numbers
k = ui.fleet_kpis(plant_id=plant_id)
unplanned_share = k["failure_cost"] / k["total_cost"] if k["total_cost"] else None
row1, row2 = st.columns(3), st.columns(3)
row1[0].metric("Availability", ui.pct(k["availability"], 2), help="1 − downtime / machine-hours")
row1[1].metric("Failure events", f"{k['failures']:,}", help=f"across {k['machines']} machines")
row1[2].metric(
    "Total cost",
    ui.inr(k["total_cost"]),
    help=f"{ui.pct(unplanned_share, 0)} from unplanned failures",
)
row2[0].metric(
    "Downtime", f"{k['downtime_h']:,.0f} h", help="unplanned repairs + planned maintenance"
)
# MTBF and MTTR are undefined without failures.
row2[1].metric(
    "MTBF", ui.hours(k["mtbf_h"], 0), help="mean operating machine-hours between failures"
)
row2[2].metric("MTTR", ui.hours(k["mttr_h"], 1), help="mean hours to repair a failure")

# ---- monthly trends (one measure per chart)
monthly = ui.fleet_monthly(plant_id=plant_id)
t1, t2, t3 = st.columns(3)
t1.altair_chart(
    charts.monthly_bars(monthly, "failures", "Failure events per month", "failure events"),
    use_container_width=True,
)
t2.altair_chart(
    charts.monthly_line(monthly, "availability", "Availability per month", "availability", ".2%"),
    use_container_width=True,
)
t3.altair_chart(charts.monthly_cost(monthly), use_container_width=True)
st.caption(
    "Unplanned failures cost about as much as all planned maintenance together: "
    "this is the cost predictive maintenance can reduce."
)

# ---- production lines and components (whole fleet)
st.subheader("Production lines")
lines = ui.line_performance()
lines["line"] = lines["line_id"]
lines["cost_per_machine_lakh"] = lines["cost_per_machine"] / 1e5
l1, l2 = st.columns(2)
l1.altair_chart(
    charts.ranked_bars(lines, "line", "failures_per_machine", "Failures per machine", "failures"),
    use_container_width=True,
)
l2.altair_chart(
    charts.ranked_bars(lines, "line", "cost_per_machine_lakh", "Cost per machine", "lakh INR"),
    use_container_width=True,
)

st.subheader("Components")
comps = ui.component_reliability()
comps["failure_cost_lakh"] = comps["failure_cost"] / 1e5
comps["saving_lakh"] = comps["saving_if_all_prevented"] / 1e5
k1, k2 = st.columns(2)
k1.altair_chart(
    charts.ranked_bars(comps, "component", "failure_cost_lakh", "Failure cost", "lakh INR"),
    use_container_width=True,
)
k2.altair_chart(
    charts.ranked_bars(
        comps, "component", "saving_lakh", "Saving if every failure were prevented", "lakh INR"
    ),
    use_container_width=True,
)

# ---- machines that cost the most
st.subheader("Costliest machines")
health = ui.machine_health()
if plant_id:
    health = health[health["plant_id"] == plant_id]
top = health.sort_values("total_cost", ascending=False).head(10)
top = top.assign(total_cost_lakh=top["total_cost"] / 1e5)
st.dataframe(
    top[
        [
            "machine_id",
            "model",
            "age",
            "line_id",
            "failures",
            "availability",
            "total_cost_lakh",
            "errors_last_30d",
        ]
    ],
    hide_index=True,
    column_config={
        "machine_id": "Machine",
        "model": "Model",
        "age": st.column_config.NumberColumn("Age (years)"),
        "line_id": "Line",
        "failures": "Failures",
        "availability": st.column_config.NumberColumn("Availability", format="percent"),
        "total_cost_lakh": st.column_config.ProgressColumn(
            "Total cost (lakh INR)",
            format="₹%.1f",
            min_value=0,
            max_value=float(health["total_cost"].max() / 1e5),
        ),
        "errors_last_30d": "Errors (last 30 days)",
    },
)
