import os
import sys

SPARK_HOME = "/usr/lib/spark3"
PYSPARK_PYTHON = "/opt/conda/envs/dsenv/bin/python"
os.environ["PYSPARK_PYTHON"]= PYSPARK_PYTHON
os.environ["PYSPARK_DRIVER_PYTHON"]= PYSPARK_PYTHON
os.environ["SPARK_HOME"] = SPARK_HOME

PYSPARK_HOME = os.path.join(SPARK_HOME, "python/lib")
sys.path.insert(0, os.path.join(PYSPARK_HOME, "py4j-0.10.9.3-src.zip"))
sys.path.insert(0, os.path.join(PYSPARK_HOME, "pyspark.zip"))

from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark.sql.functions import *

from pyspark.sql import SparkSession
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import split, expr
from pyspark.sql.functions import concat, lit
from pyspark import SparkContext, SparkConf

conf = SparkConf()
spark = SparkSession.builder.config(conf=conf).appName("Pagerank").getOrCreate()

graph_schema = StructType([
    StructField("user_id", IntegerType(), False),
    StructField("follower_id", IntegerType(), False)
])

dist_schema = StructType([
    StructField("vertex", IntegerType(), False),
    StructField("distance", IntegerType(), False),
    StructField("prev", StringType(), False),
])

def shortest_path(v_from, v_to, dataset_path=None):

    edges = spark.read.csv(dataset_path, sep="\t", schema=graph_schema) 
    edges.cache()
    distances = spark.createDataFrame([(v_from, 0, "")], dist_schema)
    d = 0
    
    cnt = False
    while True:
        
        to_lit = ","
        if not cnt:
            to_lit = ""
            cnt = True
        candidates = (distances
                      .join(edges.alias("edges"), distances.vertex==edges.follower_id)
                      .select(col("edges.user_id").alias("vertex"), (distances.distance + 1).alias("distance"), \
                              (concat(distances.prev, lit(f"{to_lit}"), distances.vertex).alias("prev"))) 
                     ).cache()
        
        new_distances = (distances
                         .join(candidates, on="vertex", how="full_outer")
                         .select("vertex", candidates.prev.alias("prev"),
                                 when(
                                     distances.distance.isNotNull(), distances.distance
                                 ).otherwise(
                                     candidates.distance
                                 ).alias("distance"))
                        ).persist()
        
        tmp = new_distances.where(new_distances.distance==d+1)
        count = tmp.count()
        if count > 0:
            d += 1            
            distances = candidates
        else:
            return "CAN'T FIND"
        
        target = (new_distances
                  .where(new_distances.vertex == v_to)
                 ).count()
        
        if  target > 0:
            return tmp.select("prev").collect()


d = shortest_path(sys.argv[1], sys.argv[2], sys.argv[3])
df = spark.createDataFrame(eval(f"{d}"))
df = df.withColumn('prev', concat(df.prev, lit(f",{sys.argv[2]}")))
df = df.dropDuplicates()
ln = len(d[0].prev.split(',')) + 1
df = df.select(df.prev).alias('prev')
df = df.select(split(df.prev, ',').alias('prev'))
for i in range(ln):
    df = df.withColumn('prev_' + str(i), expr('prev[' + str(i) + ']'))

df = df.drop("prev")
df.write.mode('overwrite').option("header", "false").csv(sys.argv[4])
df.show()

spark.stop()