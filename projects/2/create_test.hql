CREATE TEMPORARY EXTERNAL TABLE hw2_test(
    id INT,
    if1 INT,
    if2 INT,
    if3 INT,
    if4 INT,
    if5 INT,
    if6 INT,
    if7 INT,
    if8 INT,
    if9 INT,
    if10 INT,
    if11 INT,
    if12 INT,
    if13 INT,
    cf1 STRING,
    cf2 STRING,
    cf3 STRING,
    cf4 STRING,
    cf5 STRING,
    cf6 STRING,
    cf7 STRING,
    cf8 STRING,
    cf9 STRING,
    cf10 STRING,
    cf11 STRING,
    cf12 STRING,
    cf13 STRING,
    cf14 STRING,
    cf15 STRING,
    cf16 STRING,
    cf17 STRING,
    cf18 STRING,
    cf19 STRING,
    cf20 STRING,
    cf21 STRING,
    cf22 STRING,
    cf23 STRING,
    cf24 STRING,
    cf25 STRING,
    cf26 STRING,
    day_number STRING
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
LOCATION '/datasets/criteo/criteo_test_large_features';