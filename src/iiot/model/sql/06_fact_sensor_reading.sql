-- fact_sensor_reading: one row per machine, hour and sensor (long format).
-- Source: Silver `telemetry` (cleaned; quality = ok / filled / missing).
-- Missing readings are KEPT (value NULL, quality 'missing') so gaps stay visible.
--
-- No primary-key or foreign-key constraints on this table, on purpose: DuckDB builds
-- an index for each one, which on ~3.5 million rows made the build 7x slower and the
-- file 5x bigger (measured: 13.5 s / 163 MB vs 1.9 s / 35 MB). Uniqueness of
-- (machine_id, timestamp, sensor) and every reference to dim_machine, dim_date and
-- dim_sensor are verified by the model checks instead, as large warehouses usually do.
-- The smaller fact tables declare their keys and DuckDB enforces them.
CREATE TABLE fact_sensor_reading (
    machine_id  INTEGER   NOT NULL,   -- -> dim_machine
    date_key    INTEGER   NOT NULL,   -- -> dim_date
    timestamp   TIMESTAMP NOT NULL,
    sensor      VARCHAR   NOT NULL,   -- -> dim_sensor
    value       DOUBLE,
    quality     VARCHAR   NOT NULL CHECK (quality IN ('ok', 'filled', 'missing'))
);

INSERT INTO fact_sensor_reading
SELECT machine_id, CAST(strftime(timestamp, '%Y%m%d') AS INTEGER), timestamp, sensor, value, quality
FROM (
    SELECT machine_id, timestamp, 'volt' AS sensor, volt AS value,
           CAST(volt_quality AS VARCHAR) AS quality FROM src_telemetry
    UNION ALL
    SELECT machine_id, timestamp, 'rotate', rotate, CAST(rotate_quality AS VARCHAR) FROM src_telemetry
    UNION ALL
    SELECT machine_id, timestamp, 'pressure', pressure, CAST(pressure_quality AS VARCHAR) FROM src_telemetry
    UNION ALL
    SELECT machine_id, timestamp, 'vibration', vibration, CAST(vibration_quality AS VARCHAR) FROM src_telemetry
);
