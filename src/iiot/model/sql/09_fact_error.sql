-- fact_error: one row per machine error event.
-- Source: Silver `errors`.
CREATE TABLE fact_error (
    machine_id  INTEGER   NOT NULL REFERENCES dim_machine (machine_id),
    date_key    INTEGER   NOT NULL REFERENCES dim_date (date_key),
    timestamp   TIMESTAMP NOT NULL,
    error_id    VARCHAR   NOT NULL REFERENCES dim_error_type (error_id),
    PRIMARY KEY (machine_id, timestamp, error_id)
);

INSERT INTO fact_error
SELECT machine_id, CAST(strftime(timestamp, '%Y%m%d') AS INTEGER), timestamp, error_id
FROM src_errors
ORDER BY machine_id, timestamp, error_id;
