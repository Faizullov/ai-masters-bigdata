#!/opt/conda/envs/dsenv/bin/python

import sys
from pyspark.sql.types import StructType, StructField, IntegerType
from pyspark.sql.functions import *

from pyspark.sql import SparkSession
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import split, expr
from pyspark.sql.functions import concat, lit

spark = SparkSession.builder.enableHiveSupport().master("local[2]").getOrCreate()

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
            break  
        
        target = (new_distances
                  .where(new_distances.vertex == v_to)
                 ).count()
        
        if  target > 0:
            return tmp.select("prev").collect()
            break 

    return d

print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(sys.argv[1])
print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
d = shortest_path(sys.argv[1], sys.argv[2], sys.argv[3])
# tmp_str = d.to_string()
print(f"{d}")
df = spark.createDataFrame(eval(f"{d}"))
df = df.withColumn('prev', concat(df.prev, lit(f",{sys.argv[2]}")))
df = df.dropDuplicates()
ln = len(d[0].prev.split(',')) + 1
print(ln)
df = df.select(df.prev).alias('prev')
df = df.select(split(df.prev, ',').alias('prev'))
for i in range(ln):
    df = df.withColumn('prev_' + str(i), expr('prev[' + str(i) + ']'))

df = df.drop("prev")
df.write.mode('overwrite').option("header", "false").csv("/user/Faizullov/Faizullov_hw3_output")
df.show()

spark.stop()