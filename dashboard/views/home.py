"""Platform status: what data, checks and models are behind the dashboard."""

import pandas as pd
import streamlit as st

from iiot.dashboard import ui

st.title("🏭 Industrial IoT Sensor Analytics")
st.caption(
    "Predictive maintenance for 100 machines in 2 plants (Pune, Chennai). "
    "Data: Microsoft Azure Predictive Maintenance dataset (synthetic); costs are "
    "illustrative assumptions from config/settings.yaml."
)
ui.require_warehouse()

status = ui.status()
st.subheader("Data")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Machines", f"{status['tables']['dim_machine']:,}")
c2.metric("Sensor readings", f"{status['tables']['fact_sensor_reading']:,}")
c3.metric("Failures", f"{status['tables']['fact_failure']:,}")
c4.metric("Days of data", f"{(status['data_to'] - status['data_from']).days:,}")
st.caption(
    f"Hourly sensor data from {status['data_from']:%d %b %Y} to {status['data_to']:%d %b %Y}."
)

st.subheader("Quality checks")
reports = ui.report_status()
reports["result"] = reports["passed"].map(
    lambda p: ui.STATUS_LABEL[None if pd.isna(p) else bool(p)]
)
st.dataframe(
    reports[["layer", "result", "generated_at"]],
    hide_index=True,
    column_config={
        "layer": "Layer",
        "result": "Result",
        "generated_at": st.column_config.DatetimeColumn(
            "Last checked (UTC)", format="YYYY-MM-DD HH:mm"
        ),
    },
)

st.subheader("Machine learning")
cards = ui.model_cards()
if not status["has_predictions"] or not cards:
    st.info("The ML models have not been trained yet. Run `iiot ml train` (or `iiot run`).")
else:
    f, c, a = cards["failure"]["test"], cards["component"]["test"], cards["anomaly"]["test"]
    rule = cards["failure"]["test_baselines"]["error5-rule"]
    st.caption(
        f"Test period {status['predictions_from']:%d %b %Y} – {status['predictions_to']:%d %b %Y}: "
        "data the models never saw during training."
    )
    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Failures caught (24 h ahead)",
        f"{f['events_caught']} / {f['failure_events']}",
        f"{f['events_caught'] - rule['events_caught']:+d} vs error5 rule",
    )
    m2.metric("Right component named", ui.pct(c["top1_accuracy"], 0))
    m3.metric(
        "Net saving",
        ui.inr(f["net_saving"]),
        f"{ui.inr(f['net_saving'] - rule['net_saving'])} vs error5 rule",
    )
    st.caption(
        f"Models: failure-24h v{cards['failure']['version']}, "
        f"component v{cards['component']['version']}, anomaly v{cards['anomaly']['version']} "
        f"(all @champion). Anomaly flags: {ui.pct(a['flag_rate_48h'])} of rows in the 48 h "
        f"before a failure vs {ui.pct(a['flag_rate_normal'])} in normal operation."
    )
