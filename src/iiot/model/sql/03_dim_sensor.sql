-- dim_sensor: one row per sensor, with the valid range used by the Silver range check.
-- Source: `sensors` in config/settings.yaml.
CREATE TABLE dim_sensor (
    sensor     VARCHAR PRIMARY KEY,
    valid_min  DOUBLE NOT NULL,
    valid_max  DOUBLE NOT NULL,
    CHECK (valid_min < valid_max)
);

INSERT INTO dim_sensor
SELECT sensor, valid_min, valid_max
FROM src_sensors
ORDER BY sensor;
