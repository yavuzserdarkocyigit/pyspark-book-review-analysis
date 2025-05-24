from concurrent.futures import ThreadPoolExecutor
import time
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, MinMaxScaler, Tokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier, LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, regexp_replace, split, from_unixtime, year, month

# Create Spark session
spark = SparkSession.builder \
    .appName("AmazonBooksBinaryClassification") \
    .master("local[*]") \
    .config("spark.driver.memory", "32g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

# Load dataset
ratings_path = r"C:\Amazon Review Dataset\ardataset\Books_rating.csv"
ratings_df = spark.read.csv(ratings_path, header=True, inferSchema=True)

# Rename columns to match expected names
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

# Apply renaming
for original, new_name in renamed_columns.items():
    if original in ratings_df.columns:
        ratings_df = ratings_df.withColumnRenamed(original, new_name)

# Check if all expected columns are present after renaming
expected_columns = list(renamed_columns.values())
actual_columns = ratings_df.columns

missing_columns = set(expected_columns) - set(actual_columns)
extra_columns = set(actual_columns) - set(expected_columns)

if missing_columns:
    print(f"Missing columns after renaming: {missing_columns}")
if extra_columns:
    print(f"Unexpected columns after renaming: {extra_columns}")

print("Columns after renaming:", ratings_df.columns)

# Data Cleaning and Preprocessing

# Convert 'price' column to numeric
ratings_df = ratings_df.withColumn("price", regexp_replace(col("price"), "[^0-9.]", "").cast("float"))

# Process 'r_helpfulness' column into numerator and denominator
ratings_df = ratings_df.withColumn("helpful_votes", split(col("r_helpfulness"), "/")[0].cast("int"))
ratings_df = ratings_df.withColumn("total_votes", split(col("r_helpfulness"), "/")[1].cast("int"))
ratings_df = ratings_df.withColumn("helpfulness_ratio", col("helpful_votes") / col("total_votes"))

# Convert 'r_time' to year and month
ratings_df = ratings_df.withColumn("review_date", from_unixtime(col("r_time")))
ratings_df = ratings_df.withColumn("review_year", year(col("review_date")))
ratings_df = ratings_df.withColumn("review_month", month(col("review_date")))

# Drop rows with missing values in required columns
ratings_df = ratings_df.na.drop(subset=["r_review", "price", "r_score"])


price_vector_assembler = VectorAssembler(inputCols=["price"], outputCol="price_vector")
ratings_df = price_vector_assembler.transform(ratings_df)

# Create binary label column (1-3 => 0, 4-5 => 1)
ratings_df = ratings_df.withColumn("label", (col("r_score") >= 4).cast("int"))

# Drop unnecessary columns
ratings_df = ratings_df.drop("id", "title", "user_id", "profile_name", "r_time", "r_helpfulness", "r_summary")

# Define preprocessing steps
tokenizer = Tokenizer(inputCol="r_review", outputCol="tokens")
stop_remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
count_vectorizer = CountVectorizer(inputCol="filtered_tokens", outputCol="count_features", vocabSize=5000, minDF=5)
tf_idf = IDF(inputCol="count_features", outputCol="tfidf_features")
scaler = MinMaxScaler(inputCol="price_vector", outputCol="scaled_price")

# Handle NaN in helpfulness_ratio
ratings_df = ratings_df.filter(col("helpfulness_ratio").isNotNull())

# Combine features
vector_assembler = VectorAssembler(inputCols=["scaled_price", "tfidf_features", "helpfulness_ratio"], outputCol="features")

# Define the pipeline
pipeline = Pipeline(stages=[tokenizer, stop_remover, count_vectorizer, tf_idf, scaler, vector_assembler])

# Process data through the pipeline
processed_df = pipeline.fit(ratings_df).transform(ratings_df)

processed_df.show(5)


# Split data into training and test sets
train_df, test_df = processed_df.randomSplit([0.8, 0.2], seed=42)

# Define models
models = {
    "Logistic Regression": LogisticRegression(featuresCol="features", labelCol="label", maxIter=10),
    "Random Forest": RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=10),
    "Decision Tree": DecisionTreeClassifier(featuresCol="features", labelCol="label"),
    "Linear SVC": LinearSVC(featuresCol="features", labelCol="label", maxIter=10)
}

# Function to train and evaluate a single model, measuring time
def train_and_evaluate(model_name, model):
    start_time = time.time()
    trained_model = model.fit(train_df)
    predictions = trained_model.transform(test_df)

    # Calculate evaluation metrics
    evaluator_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    evaluator_accuracy = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    evaluator_precision = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
    evaluator_recall = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

    auc = evaluator_auc.evaluate(predictions)
    accuracy = evaluator_accuracy.evaluate(predictions)
    precision = evaluator_precision.evaluate(predictions)
    recall = evaluator_recall.evaluate(predictions)
    f1 = evaluator_f1.evaluate(predictions)

    duration = time.time() - start_time
    return model_name, auc, accuracy, precision, recall, f1, duration

# Train and evaluate models in parallel
results = []
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(train_and_evaluate, name, model): name for name, model in models.items()}
    for future in futures:
        name, auc, accuracy, precision, recall, f1, duration = future.result()
        results.append((name, auc, accuracy, precision, recall, f1, duration))

# Print individual model results
print("Individual Model Results:")
for name, auc, accuracy, precision, recall, f1, duration in results:
    print(f"{name}:")
    print(f"  AUC: {auc:.2f}")
    print(f"  Accuracy: {accuracy:.2f}")
    print(f"  Precision: {precision:.2f}")
    print(f"  Recall: {recall:.2f}")
    print(f"  F1 Score: {f1:.2f}")
    print(f"  Time: {duration:.2f} seconds")

# Calculate total parallel time (equal to the longest single model duration)
parallel_time = max(duration for _, _, _, _, _, _, duration in results)
print(f"\nTotal time for parallel execution: {parallel_time:.2f} seconds")
