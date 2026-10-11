"""Machine detail: one machine's health, sensors, predictions and events over a window."""

import datetime as dt

import pandas as pd
import streamlit as st

from iiot.dashboard import charts, ui

st.title("🔧 Machine detail")
ui.require_warehouse()

health = ui.machine_health()
status = ui.status()
first, last = status["data_from"].date(), status["data_to"].date()

c1, c2 = st.columns([1, 2])
default_machine = int(health.sort_values("cost_rank").iloc[0]["machine_id"])  # costliest
machine_id = c1.selectbox(
    "Machine",
    health["machine_id"].tolist(),
    index=health["machine_id"].tolist().index(default_machine),
    format_func=lambda m: f"Machine {m}",
)
window = c2.date_input(
    "Window",
    value=(max(first, last - dt.timedelta(days=31)), last),
    min_value=first,
    max_value=last,
)
if len(window) != 2:
    st.info("Pick an end date for the window.")
    st.stop()
start = pd.Timestamp(window[0])
end = pd.Timestamp(window[1]) + pd.Timedelta(hours=23)
domain = [start.isoformat(), end.isoformat()]

# ---- health card for the whole year
h = health.set_index("machine_id").loc[machine_id]
st.caption(
    f"{h['model']}, {h['age']} years old, line {h['line_id']} · "
    f"cost rank {int(h['cost_rank'])} of {len(health)}"
)
m = st.columns(5)
m[0].metric("Failures (2015)", int(h["failures"]))
m[1].metric("Availability", ui.pct(h["availability"], 2))
m[2].metric("Total cost", ui.inr(h["total_cost"]))
m[3].metric("Errors (last 30 days)", int(h["errors_last_30d"]))
last_failure = h["last_failure_at"]
m[4].metric(
    "Last failure",
    "-" if pd.isna(last_failure) else f"{last_failure:%d %b}",
    help=None if pd.isna(last_failure) else f"{last_failure:%a %d %b %Y, %H:%M}",
)

events = ui.machine_events(machine_id, start, end)

# ---- predicted risk (out-of-sample period only)
if status["has_predictions"]:
    predictions = ui.machine_predictions(machine_id, start, end)
    cards = ui.model_cards()
    if predictions.empty:
        st.info(
            "Predictions are shown for the test period only (from "
            f"{status['predictions_from']:%d %b %Y}), which the models never saw. "
            "Move the window into that period to see them."
        )
    elif "failure" in cards:
        st.altair_chart(
            charts.risk_timeline(predictions, events, cards["failure"]["threshold"], domain),
            use_container_width=True,
        )
        st.caption(
            "Dotted horizontal line: alert threshold. Dashed vertical lines: failures; "
            "dotted vertical lines: maintenance."
        )

# ---- sensors
# Hourly readings are noisy: daily means show the build-up before a failure better.
# Zoom in to 14 days or less to see every hour.
daily = (end - start) > pd.Timedelta(days=14)
sensors = ui.machine_sensors(machine_id, start, end, daily=daily)
st.subheader("Sensors")
st.caption(
    ("Daily means (pick 14 days or less for hourly readings). " if daily else "Hourly readings. ")
    + "Dashed lines: failures; dotted lines: maintenance."
)
st.altair_chart(charts.sensor_panels(sensors, events, domain), use_container_width=True)

# ---- events
st.subheader("Events in the window")
if events.empty:
    st.write("No failures, maintenance or errors in this window.")
else:
    icons = {"failure": "🔴 failure", "maintenance": "🔧 maintenance", "error": "⚠️ error"}
    st.dataframe(
        events.assign(kind=events["kind"].map(icons)).sort_values("timestamp", ascending=False),
        hide_index=True,
        column_config={
            "timestamp": st.column_config.DatetimeColumn("When", format="ddd DD MMM YYYY, HH:mm"),
            "kind": "Event",
            "detail": "Detail",
        },
    )
