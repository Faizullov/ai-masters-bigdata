#!/opt/conda/envs/dsenv/bin/python

import sys
from pyspark.sql.types import StructType, StructField, IntegerType
from pyspark.sql.functions import *

from pyspark.sql import SparkSession
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
            return tmp.select("prev").collect()[0].prev + f",{v_to}"
            break 

    return d

d = shortest_path(sys.argv[1], sys.argv[2], sys.argv[3])

d.write.format('csv').save(sys.argv[4])