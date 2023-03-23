ADD FILE projects/2/2.joblib;
ADD FILE projects/2/model.py;
ADD FILE projects/2/predict.py;

-- SELECT id, prediction
-- FROM (
--   SELECT id, pred
--   FROM hw2_test
--   WHERE if1 > 20 AND if1 < 40
-- ) filtered_data;

-- SELECT * from hw2_test LIMIT 10;

INSERT INTO TABLE hw2_pred
SELECT TRANSFORM(id, nvl(if1, -1), 
nvl(if2, -1), nvl(if3, -1), nvl(if4, -1), 
nvl(if5, -1), nvl(if6, -1), nvl(if7, -1), nvl(if8, -1), 
nvl(if9, -1), nvl(if10, -1), nvl(if11, -1), nvl(if12, -1), 
nvl(if13, -1), cf1, cf2, cf3, cf4, cf5, cf6, cf7, cf8, cf9, 
cf10, cf11, cf12, cf13, cf14, cf15, cf16, cf17, cf18, cf19, 
cf20, cf21, cf22, cf23, cf24, cf25, cf26, day_number) 
USING "predict.py" AS pred, id FROM hw2_test
WHERE if1 > 20 AND if1 < 40;

-- INSERT INTO TABLE hw2_pred
-- SELECT id, pred
-- FROM 
-- WHERE if1 > 20 AND if1 < 40;
