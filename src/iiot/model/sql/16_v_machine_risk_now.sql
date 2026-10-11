-- v_machine_risk_now: each machine's latest prediction, riskiest first (for the dashboard).
-- "Now" is the last scored time in the data. risk_rank 1 = the machine to look at first.
-- Empty until the ML models are trained and scored.
CREATE VIEW v_machine_risk_now AS
SELECT
    p.machine_id, m.model, m.age, m.plant_id, m.line_id,
    p.timestamp            AS scored_at,
    p.failure_risk, p.alert,
    p.predicted_component, p.anomaly_score, p.anomaly_flag,
    p.reason_1, p.reason_2, p.reason_3,
    h.last_failure_at, h.errors_last_30d,
    rank() OVER (ORDER BY p.failure_risk DESC, p.anomaly_score DESC) AS risk_rank
FROM fact_prediction AS p
JOIN dim_machine AS m USING (machine_id)
LEFT JOIN v_machine_health AS h USING (machine_id)
WHERE p.timestamp = (SELECT max(timestamp) FROM fact_prediction)
ORDER BY risk_rank, p.machine_id;
