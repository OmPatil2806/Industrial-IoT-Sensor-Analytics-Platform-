"""Maintenance planner: what to do first at a chosen moment, and what it is worth."""

import streamlit as st

from iiot.dashboard import ui

st.title("🗓️ Maintenance planner")
ui.require_predictions()
st.caption(
    "The to-do list the models would have produced at a moment in the test period (Oct–Dec 2015). "
    "Failure alerts come first (replace the predicted component), then machines whose sensors "
    "look unusual (inspect them)."
)

as_of = ui.as_of_picker()
plan = ui.maintenance_plan(as_of)

if plan.empty:
    st.success(f"Nothing to do at {as_of:%a %d %b %Y, %H:%M}: no alerts and no unusual sensors.")
    st.stop()

replace = plan[plan["alert"]]
inspect = plan[~plan["alert"]]
scored_at = plan["scored_at"].iloc[0]

m = st.columns(4)
m[0].metric("Replacements", len(replace), help="machines with a failure alert")
m[1].metric("Inspections", len(inspect), help="machines with unusual sensors but no alert")
m[2].metric(
    "Saving if done in time",
    ui.inr(replace["saving_if_acted"].sum()),
    help="unplanned failure cost minus planned maintenance cost, per replaced component",
)
m[3].metric("Lines involved", plan["line_id"].nunique())

st.subheader(f"To-do list at {scored_at:%a %d %b %Y, %H:%M}")
todo = plan.assign(
    action=plan["action"].map(lambda a: ("🔴 " if a.startswith("replace") else "🟡 ") + a),
    saving_lakh=plan["saving_if_acted"] / 1e5,
    reason_1=plan["reason_1"].fillna(""),
    reason_2=plan["reason_2"].fillna(""),
)
st.dataframe(
    todo[
        [
            "priority",
            "action",
            "machine_id",
            "line_id",
            "failure_risk",
            "alert_since",
            "saving_lakh",
            "reason_1",
            "reason_2",
        ]
    ],
    hide_index=True,
    column_config={
        "priority": "Priority",
        "action": "Action",
        "machine_id": "Machine",
        "line_id": "Line",
        "failure_risk": st.column_config.ProgressColumn(
            "Failure risk (24 h)", format="percent", min_value=0, max_value=1
        ),
        "alert_since": st.column_config.DatetimeColumn("Alert since", format="ddd DD MMM, HH:mm"),
        "saving_lakh": st.column_config.NumberColumn("Saving (lakh INR)", format="₹%.1f"),
        "reason_1": "Main reason",
        "reason_2": "Reason 2",
    },
)

csv = plan.drop(columns=["scored_at"]).assign(scored_at=scored_at).to_csv(index=False)
st.download_button(
    "⬇️ Download the to-do list (CSV)",
    csv,
    file_name=f"maintenance_plan_{scored_at:%Y%m%d_%H%M}.csv",
    mime="text/csv",
)

left, right = st.columns(2)
with left:
    st.subheader("Spare parts to prepare")
    if replace.empty:
        st.write("No replacements needed.")
    else:
        parts = (
            replace.groupby("predicted_component")
            .agg(units=("machine_id", "size"), saving=("saving_if_acted", "sum"))
            .assign(saving=lambda d: d["saving"] / 1e5)
            .reset_index()
            .sort_values("units", ascending=False)
        )
        st.dataframe(
            parts,
            hide_index=True,
            column_config={
                "predicted_component": "Component",
                "units": "Units",
                "saving": st.column_config.NumberColumn("Saving (lakh INR)", format="₹%.1f"),
            },
        )
with right:
    st.subheader("Work per production line")
    work = (
        plan.groupby("line_id")
        .agg(
            replacements=("alert", "sum"),
            inspections=("alert", lambda a: int((~a).sum())),
        )
        .reset_index()
        .sort_values(["replacements", "inspections"], ascending=False)
    )
    st.dataframe(
        work,
        hide_index=True,
        column_config={
            "line_id": "Line",
            "replacements": "Replacements",
            "inspections": "Inspections",
        },
    )

st.caption(
    "Savings use the illustrative costs in config/settings.yaml. Inspections have no saving "
    "attached: unusual sensors are a prompt to look, not a failure prediction."
)
