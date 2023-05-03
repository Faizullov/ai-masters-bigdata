import os
import sys
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--train-in", dest="train_in", type=str)
parser.add_argument("--sklearn-model-out", dest="sklearn_model_out", type=str)

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

from joblib import dump

model = make_pipeline(
    HashingVectorizer(n_features=150),
    LogisticRegression()
)

table_df = pd.read_json(args.train_in, lines=True)

models = model.fit(table_df['reviewText'], table_df['label'])

dump(models, args.sklearn_model_out)