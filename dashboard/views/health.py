"""Model & data health: what data, checks, models and pipeline runs are behind the dashboard."""

import pandas as pd
import streamlit as st

from iiot.dashboard import data, ui

st.title("✅ Model & data health")
st.caption(
    "Predictive maintenance for 100 machines in 2 plants (Pune, Chennai). "
    "Data: Microsoft Azure Predictive Maintenance dataset (synthetic); costs are "
    "illustrative assumptions from config/settings.yaml."
)
ui.require_warehouse()
status = ui.status()

# ---- data
st.subheader("Data")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Machines", f"{status['tables']['dim_machine']:,}")
c2.metric("Sensor readings", f"{status['tables']['fact_sensor_reading']:,}")
c3.metric("Failures", f"{status['tables']['fact_failure']:,}")
c4.metric("Days of data", f"{(status['data_to'] - status['data_from']).days:,}")
st.caption(
    f"Hourly sensor data from {status['data_from']:%d %b %Y} to {status['data_to']:%d %b %Y}."
)

# ---- quality checks: one row per layer, details on demand
st.subheader("Quality checks")
reports = ui.report_status()
details = {layer: ui.load_report(layer) for layer in reports["layer"]}
reports["checks"] = [
    "" if details[layer] is None else f"{int(c['passed'].sum())}/{len(c)}"
    for layer, c in (
        (layer, data.report_checks(details[layer] or {})) for layer in reports["layer"]
    )
]
reports["result"] = reports["passed"].map(
    lambda p: ui.STATUS_LABEL[None if pd.isna(p) else bool(p)]
)
st.dataframe(
    reports[["layer", "result", "checks", "generated_at"]],
    hide_index=True,
    column_config={
        "layer": "Layer",
        "result": "Result",
        "checks": "Checks passed",
        "generated_at": st.column_config.DatetimeColumn(
            "Last checked (UTC)", format="YYYY-MM-DD HH:mm"
        ),
    },
)
available = [layer for layer, r in details.items() if r is not None]
layer = st.selectbox("Show the checks of", available, index=max(len(available) - 1, 0))
if layer:
    checks = data.report_checks(details[layer])
    st.dataframe(
        checks.assign(passed=checks["passed"].map(ui.STATUS_LABEL)),
        hide_index=True,
        column_config={"check": "Check", "passed": "Result"},
    )

# ---- models
st.subheader("Models (test period: data never seen in training)")
cards = ui.model_cards()
if not cards:
    st.info("The ML models have not been trained yet. Run `iiot ml train` (or `iiot run`).")
else:
    rows = []  # (registered name, card, test result, baseline, beat its baseline?)
    if "failure" in cards:
        f = cards["failure"]["test"]
        rule = cards["failure"]["test_baselines"]["error5-rule"]
        result = f"{f['events_caught']}/{f['failure_events']} failures caught, "
        result += f"{f['false_alarms']} false alarm(s), {ui.inr(f['net_saving'])} saved"
        baseline = f"error5 rule: {rule['events_caught']}/{rule['failure_events']} caught, "
        baseline += ui.inr(rule["net_saving"])
        rows.append(
            (
                "iiot-failure-24h",
                cards["failure"],
                result,
                baseline,
                cards["failure"].get("beats_baselines"),
            )
        )
    if "component" in cards:
        c = cards["component"]
        result = f"right component first {ui.pct(c['test']['top1_accuracy'], 0)}"
        baseline = f"most frequent component: {ui.pct(c['test_baseline']['top1_accuracy'], 0)}"
        rows.append(("iiot-component", c, result, baseline, c.get("beats_baseline")))
    if "anomaly" in cards:
        a = cards["anomaly"]
        result = f"{ui.pct(a['test']['flag_rate_48h'])} of rows flagged 48 h before a failure"
        baseline = f"{ui.pct(a['test']['flag_rate_normal'])} flagged in normal operation"
        rows.append(("iiot-anomaly", a, result, baseline, a.get("promoted")))
    table = pd.DataFrame(
        [
            {
                "Model": name,
                "Version": card["version"],
                "Champion": "✅ yes" if beats else "❌ no",
                "Test result": result,
                "Baseline": baseline,
                "Trained (UTC)": pd.Timestamp(card["trained_at"]),
                "Git commit": card["git_commit"],
            }
            for name, card, result, baseline, beats in rows
        ]
    )
    st.dataframe(
        table,
        hide_index=True,
        column_config={
            "Trained (UTC)": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
        },
    )
    st.caption(
        "A model becomes @champion only if it beats its baseline on the test period. "
        "Every run, chart and model version is in MLflow: "
        "`mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --workers 1`."
    )

# ---- pipeline runs
st.subheader("Pipeline runs (`iiot run`)")
runs = ui.pipeline_runs()
if runs.empty:
    st.info("No tracked pipeline runs yet. Every `iiot run` is recorded in MLflow.")
else:
    st.dataframe(
        runs.assign(passed=runs["passed"].map(ui.STATUS_LABEL)),
        hide_index=True,
        column_config={
            "started": st.column_config.DatetimeColumn("Started (UTC)", format="YYYY-MM-DD HH:mm"),
            "minutes": st.column_config.NumberColumn("Minutes", format="%.1f"),
            "passed": "Result",
            "steps": "Steps",
            "git_commit": "Git commit",
            "status": "Status",
        },
    )
