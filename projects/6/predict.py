import os
import sys
import argparse


# Construct the argument parser
ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("--test-in", required=True)
ap.add_argument("--pred-out", required=True)
ap.add_argument("--sklearn-model-in", required=True)
args = vars(ap.parse_args())

SPARK_HOME = "/usr/lib/spark3"
PYSPARK_PYTHON = "/opt/conda/envs/dsenv/bin/python"
os.environ["PYSPARK_PYTHON"]= PYSPARK_PYTHON
os.environ["PYSPARK_DRIVER_PYTHON"]= PYSPARK_PYTHON
os.environ["SPARK_HOME"] = SPARK_HOME

PYSPARK_HOME = os.path.join(SPARK_HOME, "python/lib")
sys.path.insert(0, os.path.join(PYSPARK_HOME, "py4j-0.10.9.3-src.zip"))
sys.path.insert(0, os.path.join(PYSPARK_HOME, "pyspark.zip"))

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
import pandas as pd

df = pd.read_json(args['test_in']).fillna( {"reviewText": "missingreview"})

from pyspark.sql import functions as F

models = load(args['sklearn-model-in'])

clf = spark.sparkContext.broadcast(models)

@F.pandas_udf("double")
def predict(*cols):
    X = cols[0]
    # Make predictions and select probabilities of positive class (1).
    X.columns = ['reviewText']
    predictions = clf.value.predict_proba(X)[:, 1]
    # Return Pandas Series of predictions.
    return pd.Series(predictions)

df = df.withColumn("predictions", predict(*df.columns))
df.select("id", "predictions").write.csv(args['pred-out'])