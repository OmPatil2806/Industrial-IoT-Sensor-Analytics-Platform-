"""Machines at risk: what the models said at a chosen time, and what happened next."""

import datetime as dt

import pandas as pd
import streamlit as st

from iiot.dashboard import charts, ui

st.title("🚨 Machines at risk")
ui.require_predictions()
st.caption(
    "Replay of the test period (Oct–Dec 2015), which the models never saw in training. "
    "Pick a moment to see which machines the models flagged then, and why."
)

# ---- "as of" picker: a date and one of the 3-hourly scoring times
times = ui.prediction_times()
default = ui.latest_alert_time() or times.max()
c1, c2 = st.columns([2, 1])
day = c1.date_input(
    "As of date",
    value=default.date(),
    min_value=times.min().date(),
    max_value=times.max().date(),
)
hours = [f"{h:02d}:00" for h in range(0, 24, 3)]
hour = c2.selectbox("Time", hours, index=hours.index(f"{default.hour // 3 * 3:02d}:00"))
as_of = pd.Timestamp(dt.datetime.combine(day, dt.time(int(hour[:2]))))

risk = ui.risk_as_of(as_of)
if risk.empty:
    st.info("No predictions at or before this time. Pick a later date.")
    st.stop()
scored_at = risk["scored_at"].iloc[0]
alerts = risk[risk["alert"]]
flagged = risk[risk["anomaly_flag"]]

# ---- headline numbers for that moment
m = st.columns([1, 1, 1.7, 1.2])  # room for two tied components, e.g. "comp1, comp4"
m[0].metric(
    "Failure alerts",
    len(alerts),
    help="machines whose risk of failure within 24 h is above the threshold",
)
m[1].metric(
    "Unusual sensors",
    len(flagged),
    help="machines flagged by the anomaly model (no failure labels used)",
)
counts = alerts["predicted_component"].value_counts()
m[2].metric(
    "Component most at risk",
    ", ".join(sorted(counts[counts == counts.max()].index)) if len(counts) else "-",
    help="the component predicted most often among the alerted machines",
)
plan = ui.maintenance_plan(as_of)
m[3].metric(
    "Saving at stake",
    ui.inr(plan["saving_if_acted"].sum()) if len(plan) else ui.inr(0),
    help="if every alerted machine is serviced before it fails",
)

st.altair_chart(charts.daily_alerts(ui.daily_alerts(), as_of.normalize()), use_container_width=True)

# ---- ranked machines
st.subheader(f"Machines ranked by risk at {scored_at:%a %d %b %Y, %H:%M}")
only_attention = st.toggle("Only machines that need attention", value=True)
shown = risk[risk["alert"] | risk["anomaly_flag"]] if only_attention else risk
if shown.empty:
    st.success("No machine needs attention at this time.")
else:
    table = shown.fillna({"reason_1": "", "reason_2": "", "reason_3": ""}).assign(
        status=shown.apply(
            lambda r: (
                "🔴 failure alert"
                if r["alert"]
                else ("🟡 unusual sensors" if r["anomaly_flag"] else "🟢 normal")
            ),
            axis=1,
        )
    )
    st.dataframe(
        table[
            [
                "risk_rank",
                "machine_id",
                "line_id",
                "model",
                "status",
                "failure_risk",
                "predicted_component",
                "reason_1",
                "reason_2",
                "reason_3",
            ]
        ],
        hide_index=True,
        column_config={
            "risk_rank": "Rank",
            "machine_id": "Machine",
            "line_id": "Line",
            "model": "Model",
            "status": "Status",
            "failure_risk": st.column_config.ProgressColumn(
                "Failure risk (24 h)", format="percent", min_value=0, max_value=1
            ),
            "predicted_component": "Component",
            "reason_1": "Main reason",
            "reason_2": "Reason 2",
            "reason_3": "Reason 3",
        },
    )

# ---- what actually happened next (possible because this is a replay)
st.subheader("What happened in the next 24 hours")
actual = ui.failures_between(scored_at, scored_at + pd.Timedelta(hours=24))
if actual.empty:
    st.write("No machine failed in the 24 hours after this moment.")
else:
    caught = actual["machine_id"].isin(alerts["machine_id"])
    st.write(
        f"**{len(actual)}** failure(s) followed; **{int(caught.sum())}** of them were on machines "
        "that had a failure alert at this moment."
    )
    st.dataframe(
        actual.assign(alerted=caught.map({True: "✅ yes", False: "❌ no"})),
        hide_index=True,
        column_config={
            "machine_id": "Machine",
            "timestamp": st.column_config.DatetimeColumn("Failed at", format="ddd DD MMM, HH:mm"),
            "components": "Components",
            "alerted": "Alerted in advance",
        },
    )
