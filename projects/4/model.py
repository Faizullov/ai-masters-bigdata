from pyspark.sql.types import *
from pyspark.ml.feature import *
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline


tokenizer_review = Tokenizer(inputCol="reviewText", outputCol="words_review")
tokenizer_summary = Tokenizer(inputCol="summary", outputCol="words_summary")
hasher_review = HashingTF(numFeatures=100, inputCol="words_review", outputCol="review_vector")
hasher_summary = HashingTF(numFeatures=10, inputCol="words_summary", outputCol="summary_vector")
assembler = VectorAssembler(inputCols=["review_vector"],outputCol="features")
train_data, test_data = df2.randomSplit([0.7, 0.3], seed=123)
lr = LinearRegression(regParam=0.0, labelCol='overall', featuresCol='features')

pipeline = Pipeline(stages = [
    tokenizer_review,
#     tokenizer_summary,
    hasher_review,
#     hasher_summary,
    assembler,
    lr
])
