from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, regexp_replace, split, when, isnan, count, avg, from_unixtime
)
from pyspark.sql.types import IntegerType, FloatType
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# 1. Create Spark Session
spark = SparkSession.builder \
    .appName("ALSRecommendationSystem") \
    .master("local[*]") \
    .config("spark.driver.memory", "32g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# 2. Load data
ratings_path = r"C:\Amazon Review Dataset\ardataset\Books_rating.csv"
ratings_df = spark.read.csv(ratings_path, header=True, inferSchema=True)

# 3. Rename columns
renamed_columns = {
    "Id": "id",
    "Title": "title",
    "Price": "price",
    "User_id": "user_id",
    "profileName": "profile_name",
    "review/helpfulness": "r_helpfulness",
    "review/score": "r_score",
    "review/time": "r_time",
    "review/summary": "r_summary",
    "review/text": "r_review"
}
for original, new_name in renamed_columns.items():
    if original in ratings_df.columns:
        ratings_df = ratings_df.withColumnRenamed(original, new_name)

# 4. Basic checks
required_cols = ["user_id", "id", "r_score"]
for c in required_cols:
    if c not in ratings_df.columns:
        raise ValueError(f"Missing required column: {c}")

# 5. Clean data
ratings_df = ratings_df.na.drop(subset=["user_id", "id", "r_score"])
ratings_df = ratings_df.withColumn(
    "price",
    regexp_replace(col("price"), "[^0-9.]", "").cast(FloatType())
)
# helpful votes
ratings_df = ratings_df.withColumn(
    "helpful_votes",
    split(col("r_helpfulness"), "/")[0].cast(IntegerType())
).withColumn(
    "total_votes",
    split(col("r_helpfulness"), "/")[1].cast(IntegerType())
)
ratings_df = ratings_df.withColumn(
    "helpfulness_ratio",
    when(col("total_votes") == 0, 0.0).otherwise(col("helpful_votes") / col("total_votes"))
)
ratings_df = ratings_df.withColumn("r_score", col("r_score").cast(IntegerType()))
ratings_df = ratings_df.withColumn("r_time_ts", from_unixtime(col("r_time")).cast("timestamp"))

# 6. Convert user_id and id to numeric
user_indexer = StringIndexer(inputCol="user_id", outputCol="userIndex", handleInvalid="skip")
item_indexer = StringIndexer(inputCol="id", outputCol="itemIndex", handleInvalid="skip")

indexed_df = user_indexer.fit(ratings_df).transform(ratings_df)
indexed_df = item_indexer.fit(indexed_df).transform(indexed_df)

# 7. Create ALS dataframe
als_df = indexed_df.select("userIndex", "itemIndex", "r_score")

# Drop any null or NaN in r_score
als_df = als_df.filter((col("r_score").isNotNull()) & (~isnan(col("r_score"))))

# (a) Check for duplicates; optionally average them
als_df = als_df.groupBy("userIndex", "itemIndex").agg(avg("r_score").alias("r_score"))

# (b) Filter out users/items with fewer than 2 ratings
user_freq = als_df.groupBy("userIndex").agg(count("*").alias("user_count"))
valid_users = user_freq.filter(col("user_count") >= 2).select("userIndex")
item_freq = als_df.groupBy("itemIndex").agg(count("*").alias("item_count"))
valid_items = item_freq.filter(col("item_count") >= 2).select("itemIndex")

als_df = als_df.join(valid_users, on="userIndex", how="inner")
als_df = als_df.join(valid_items, on="itemIndex", how="inner")

# 8. Train/test split
train_df, test_df = als_df.randomSplit([0.8, 0.2], seed=42)

# 9. Define ALS 
als = ALS(
    rank=5,            # smaller rank help stability
    maxIter=10,
    regParam=0.2,      
    userCol="userIndex",
    itemCol="itemIndex",
    ratingCol="r_score",
    coldStartStrategy="drop"
)

als_model = als.fit(train_df)

# 10. Predictions and evaluation
predictions = als_model.transform(test_df)

evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="r_score",
    predictionCol="prediction"
)
rmse = evaluator.evaluate(predictions)
print(f"Root-mean-square error (RMSE) = {rmse:.2f}")

spark.stop()
