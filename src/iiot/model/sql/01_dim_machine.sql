-- dim_machine: one row per machine, with its plant and production line.
-- Source: Silver `machines` (model and age joined with the plant layout).
CREATE TABLE dim_machine (
    machine_id  INTEGER PRIMARY KEY,
    model       VARCHAR NOT NULL,
    age         INTEGER,
    plant_id    VARCHAR,
    plant_name  VARCHAR,
    city        VARCHAR,
    line_id     VARCHAR
);

INSERT INTO dim_machine
SELECT machine_id, model, age, plant_id, plant_name, city, line_id
FROM src_machines
ORDER BY machine_id;
