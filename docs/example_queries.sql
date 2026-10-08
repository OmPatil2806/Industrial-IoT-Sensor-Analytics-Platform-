-- Example queries for the warehouse: data/warehouse/iiot.duckdb
--
-- Each query answers one business question. Run them in a notebook
-- (notebooks/02_explore_warehouse.ipynb), in Python:
--     from iiot.model.warehouse import connect
--     connect().sql("SELECT * FROM v_fleet_monthly").show()
-- or in any SQL tool that supports DuckDB (e.g. DBeaver; open the file read-only).
--
-- All queries are run by tests/test_model_views.py, so they always work with the
-- current schema.


-- Q1. How did the fleet perform month by month?
SELECT year_month, failures, downtime_h,
       round(availability * 100, 2)  AS availability_pct,
       round(mtbf_h)                 AS mtbf_h,
       round(total_cost / 1e5, 1)    AS total_cost_lakh_inr
FROM v_fleet_monthly
ORDER BY year_month;


-- Q2. Which 10 machines cost the most, and where are they?
SELECT cost_rank, machine_id, model, age, line_id, failures,
       round(availability * 100, 2) AS availability_pct,
       round(total_cost / 1e5, 1)   AS total_cost_lakh_inr
FROM v_machine_health
ORDER BY cost_rank
LIMIT 10;


-- Q3. Which components cause the most failure cost, and what could prevention save?
SELECT component, failures, machines_affected,
       round(unplanned_share * 100, 1)        AS unplanned_pct,
       round(mean_days_between_failures, 1)   AS mean_days_between_failures,
       round(failure_cost / 1e5, 1)           AS failure_cost_lakh_inr,
       round(saving_if_all_prevented / 1e5, 1) AS saving_if_all_prevented_lakh_inr
FROM v_component_reliability
ORDER BY failure_cost DESC;


-- Q4. Which production lines perform best and worst?
SELECT availability_rank, plant_name, line_id, machines, failures,
       round(availability * 100, 2)    AS availability_pct,
       round(failures_per_machine, 2)  AS failures_per_machine,
       round(cost_per_machine / 1e5, 1) AS cost_per_machine_lakh_inr
FROM v_line_performance
ORDER BY availability_rank;


-- Q5. Do machine models differ in reliability? (star join: fact + dim_machine)
SELECT m.model,
       count(DISTINCT m.machine_id)                                AS machines,
       CAST(sum(k.failures) AS INTEGER)                            AS failures,
       round(sum(k.failures) / count(DISTINCT m.machine_id), 2)    AS failures_per_machine,
       round((1 - sum(k.downtime_h) / sum(k.period_hours)) * 100, 2) AS availability_pct
FROM fact_machine_month AS k
JOIN dim_machine AS m USING (machine_id)
GROUP BY m.model
ORDER BY failures_per_machine DESC;


-- Q6. Do older machines fail more often?
SELECT CASE WHEN age < 5 THEN '0-4 years'
            WHEN age < 10 THEN '5-9 years'
            WHEN age < 15 THEN '10-14 years'
            ELSE '15+ years' END                  AS age_band,
       count(*)                                   AS machines,
       round(avg(failures), 2)                    AS avg_failures_per_machine,
       round(avg(total_cost) / 1e5, 1)            AS avg_cost_lakh_inr
FROM v_machine_health
GROUP BY age_band
ORDER BY min(age);


-- Q7. Do the sensors behave differently in the 24 hours before a failure?
-- Compares good-quality readings in the 24 h before each failure event with all other
-- good-quality readings. This is the signal the ML model will learn from.
WITH failure_events AS (
    SELECT DISTINCT machine_id, timestamp AS failed_at FROM fact_failure
),
readings AS (
    SELECT r.sensor, r.value,
           EXISTS (
               SELECT 1 FROM failure_events AS f
               WHERE f.machine_id = r.machine_id
                 AND r.timestamp >= f.failed_at - INTERVAL 24 HOUR
                 AND r.timestamp < f.failed_at
           ) AS before_failure
    FROM fact_sensor_reading AS r
    WHERE r.quality = 'ok'
)
SELECT sensor,
       round(avg(value) FILTER (WHERE NOT before_failure), 2) AS normal_avg,
       round(avg(value) FILTER (WHERE before_failure), 2)     AS before_failure_avg,
       round((avg(value) FILTER (WHERE before_failure)
              / avg(value) FILTER (WHERE NOT before_failure) - 1) * 100, 1) AS change_pct
FROM readings
GROUP BY sensor
ORDER BY abs(change_pct) DESC;


-- Q8. Which error codes are early warnings?
-- Share of each error that is followed by a failure on the same machine within 48 hours.
WITH failure_events AS (
    SELECT DISTINCT machine_id, timestamp AS failed_at FROM fact_failure
),
flagged AS (
    SELECT e.error_id,
           EXISTS (
               SELECT 1 FROM failure_events AS f
               WHERE f.machine_id = e.machine_id
                 AND f.failed_at > e.timestamp
                 AND f.failed_at <= e.timestamp + INTERVAL 48 HOUR
           ) AS followed_by_failure
    FROM fact_error AS e
)
SELECT error_id,
       count(*)                                                     AS errors,
       count(*) FILTER (WHERE followed_by_failure)                  AS followed_by_failure,
       round(avg(CAST(followed_by_failure AS INTEGER)) * 100, 1)    AS followed_by_failure_pct
FROM flagged
GROUP BY error_id
ORDER BY followed_by_failure_pct DESC;


-- Q9. Do failures happen more on some weekdays? (star join: fact + dim_date)
SELECT d.day_of_week, d.day_name,
       count(DISTINCT (f.machine_id, f.timestamp)) AS failure_events
FROM fact_failure AS f
JOIN dim_date AS d USING (date_key)
GROUP BY d.day_of_week, d.day_name
ORDER BY d.day_of_week;


-- Q10. How good is the sensor data? Share of readings per quality flag and sensor.
SELECT sensor,
       count(*)                                                         AS readings,
       round(avg(CAST(quality = 'ok' AS INTEGER)) * 100, 3)             AS ok_pct,
       round(avg(CAST(quality = 'filled' AS INTEGER)) * 100, 3)         AS filled_pct,
       round(avg(CAST(quality = 'missing' AS INTEGER)) * 100, 3)        AS missing_pct
FROM fact_sensor_reading
GROUP BY sensor
ORDER BY sensor;
