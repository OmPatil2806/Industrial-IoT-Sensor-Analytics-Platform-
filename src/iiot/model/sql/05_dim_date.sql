-- dim_date: one row per calendar day, covering every date that appears in the data
-- (maintenance history starts in 2014, before the telemetry).
-- date_key is the date as a number, e.g. 20150101, used by the fact tables.
CREATE TABLE dim_date (
    date_key      INTEGER PRIMARY KEY,
    date          DATE NOT NULL UNIQUE,
    year          INTEGER,
    quarter       INTEGER,
    month         INTEGER,
    month_name    VARCHAR,
    year_month    VARCHAR,
    iso_week      INTEGER,
    day_of_month  INTEGER,
    day_of_week   INTEGER,   -- 1 = Monday ... 7 = Sunday
    day_name      VARCHAR,
    is_weekend    BOOLEAN
);

INSERT INTO dim_date
SELECT
    CAST(strftime(d, '%Y%m%d') AS INTEGER),
    d,
    year(d),
    quarter(d),
    month(d),
    monthname(d),
    strftime(d, '%Y-%m'),
    weekofyear(d),
    day(d),
    isodow(d),
    dayname(d),
    isodow(d) >= 6
FROM (SELECT CAST(date AS DATE) AS d FROM src_dates)
ORDER BY d;
