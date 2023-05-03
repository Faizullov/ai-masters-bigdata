import os
import sys
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--pred-out", dest="pred_out", type=str)
parser.add_argument("--test-in", dest="test_in", type=str)
parser.add_argument("--sklearn-model-in", dest="sklearn_model_in", type=str)

args = parser.parse_args()

SPARK_HOME = "/usr/lib/spark3"
PYSPARK_PYTHON = "/opt/conda/envs/dsenv/bin/python"
os.environ["PYSPARK_PYTHON"]= PYSPARK_PYTHON
os.environ["PYSPARK_DRIVER_PYTHON"]= PYSPARK_PYTHON
os.environ["SPARK_HOME"] = SPARK_HOME

PYSPARK_HOME = os.path.join(SPARK_HOME, "python/lib")
sys.path.insert(0, os.path.join(PYSPARK_HOME, "py4j-0.10.9.3-src.zip"))
sys.path.insert(0, os.path.join(PYSPARK_HOME, "pyspark.zip"))

from pyspark import SparkConf
from pyspark.sql import SparkSession

conf = SparkConf()

spark = SparkSession.builder.config(conf=conf).appName("Spark SQL").getOrCreate()

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
import pandas as pd
from joblib import load

df = spark.read.json(args.test_in).fillna( {"reviewText": "missingreview"})

from pyspark.sql import functions as F

models = load(args.sklearn_model_in)

clf = spark.sparkContext.broadcast(models)

@F.pandas_udf("double")
def predict(*cols):
    X = cols[0]
    # Make predictions and select probabilities of positive class (1).
    X.columns = ['reviewText']
    predictions = clf.value.predict_proba(X)[:, 1]
    # Return Pandas Series of predictions.
    return pd.Series(predictions)

df = df.withColumn("predictions", predict('reviewText'))
df.select("id", "predictions").write.csv(args.pred_out)