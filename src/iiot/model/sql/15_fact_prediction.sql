-- fact_prediction: the ML models' output per machine and time (every 3 hours).
-- Source: Gold `predictions` (iiot.ml.score); empty until the models are trained and scored.
--   failure_risk / alert            iiot-failure-24h: probability of a failure within 24 h
--   predicted_component, compN_risk iiot-component: which component is at risk
--   anomaly_score / anomaly_flag    iiot-anomaly: unusual sensor behaviour (no labels used)
--   reason_1..3                     why an alert was raised (alert rows only)
--   out_of_sample                   TRUE from the train/test split date: the models never
--                                   saw these rows. Earlier rows were used for training,
--                                   so judge the models on out-of-sample rows only.
CREATE TABLE fact_prediction (
    machine_id               INTEGER   NOT NULL REFERENCES dim_machine (machine_id),
    date_key                 INTEGER   NOT NULL REFERENCES dim_date (date_key),
    timestamp                TIMESTAMP NOT NULL,
    failure_risk             DOUBLE    NOT NULL CHECK (failure_risk BETWEEN 0 AND 1),
    alert                    BOOLEAN   NOT NULL,
    predicted_component      VARCHAR   NOT NULL REFERENCES dim_component (component),
    comp1_risk               DOUBLE    NOT NULL,
    comp2_risk               DOUBLE    NOT NULL,
    comp3_risk               DOUBLE    NOT NULL,
    comp4_risk               DOUBLE    NOT NULL,
    anomaly_score            DOUBLE    NOT NULL,
    anomaly_flag             BOOLEAN   NOT NULL,
    reason_1                 VARCHAR,
    reason_2                 VARCHAR,
    reason_3                 VARCHAR,
    out_of_sample            BOOLEAN   NOT NULL,
    failure_model_version    VARCHAR   NOT NULL,
    component_model_version  VARCHAR   NOT NULL,
    anomaly_model_version    VARCHAR   NOT NULL,
    PRIMARY KEY (machine_id, timestamp)
);

INSERT INTO fact_prediction
SELECT machine_id,
       CAST(strftime(timestamp, '%Y%m%d') AS INTEGER),
       timestamp, failure_risk, alert, predicted_component,
       comp1_risk, comp2_risk, comp3_risk, comp4_risk,
       anomaly_score, anomaly_flag, reason_1, reason_2, reason_3, out_of_sample,
       failure_model_version, component_model_version, anomaly_model_version
FROM src_predictions
ORDER BY machine_id, timestamp;
