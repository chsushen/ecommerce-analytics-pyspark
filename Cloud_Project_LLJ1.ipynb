# =========================================
# Real-Time E-Commerce Analytics (PySpark)
# =========================================

# Install (uncomment if needed)
# !pip install pyspark kagglehub pandas matplotlib

import kagglehub
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, expr

print("\n=== STEP 1: DATA INGESTION ===")

# Download dataset
path = kagglehub.dataset_download(
    "mkechinov/ecommerce-behavior-data-from-multi-category-store"
)
print("Dataset Path:", path)

# Load file
files = os.listdir(path)
file_path = os.path.join(path, files[0])

# Load limited rows for performance
df = pd.read_csv(file_path, nrows=100000)
print("\nData Loaded:", df.shape)

# -----------------------------------------
print("\n=== STEP 2: DATA CLEANING ===")

df['category_code'] = df['category_code'].fillna("unknown")
df['brand'] = df['brand'].fillna("unknown")
df['event_time'] = pd.to_datetime(df['event_time'])

print(df.head())

# -----------------------------------------
print("\n=== STEP 3: SPARK PROCESSING ===")

spark = SparkSession.builder.appName("EcommerceAnalytics").getOrCreate()
spark_df = spark.createDataFrame(df)

print("\nSample Spark Data:")
spark_df.show(5)

# -----------------------------------------
print("\n=== STEP 4: ANALYTICS ===")

print("\nEvent Type Count:")
spark_df.groupBy("event_type").count().show()

print("\nTop Products:")
spark_df.groupBy("product_id") \
    .count() \
    .orderBy("count", ascending=False) \
    .show(10)

print("\nTop Users:")
spark_df.groupBy("user_id") \
    .count() \
    .orderBy("count", ascending=False) \
    .show(10)

# -----------------------------------------
print("\n=== STEP 5: REAL-TIME SIMULATION ===")

chunks = [df[i:i+5000] for i in range(0, len(df), 5000)]

for i, chunk in enumerate(chunks[:5]):
    print(f"\n--- Batch {i+1} ---")
    spark_chunk = spark.createDataFrame(chunk)
    spark_chunk.groupBy("event_type").count().show()
    time.sleep(1)

# -----------------------------------------
print("\n=== STEP 6: DASHBOARD (CHARTS) ===")

event_counts = df["event_type"].value_counts()
event_counts.plot(kind='bar', title="User Activity")
plt.show()

top_products = df["product_id"].value_counts().head(10)
top_products.plot(kind='bar', title="Top Products")
plt.show()

# -----------------------------------------
print("\n=== STEP 7: UNIFIED CUSTOMER VIEW ===")

transactions = df[['user_id', 'product_id', 'price']].copy()
transactions['purchase'] = 1

reviews = df[['user_id', 'product_id']].copy()
reviews['rating'] = 4.0  # dummy rating

spark_transactions = spark.createDataFrame(transactions) \
    .withColumnRenamed("price", "transaction_price")

spark_reviews = spark.createDataFrame(reviews)

customer_view = spark_df.join(
    spark_transactions,
    on=['user_id', 'product_id'],
    how='left'
).join(
    spark_reviews,
    on=['user_id', 'product_id'],
    how='left'
)

print("\nUnified Customer View:")
customer_view.show(5)

print("\nCustomer Insights:")
customer_view.groupBy("user_id").agg(
    avg("transaction_price").alias("avg_spent"),
    avg("rating").alias("avg_rating")
).show(10)

# -----------------------------------------
print("\n=== STEP 8: SECURITY ===")

customer_view_secured = customer_view.withColumn(
    "user_id_masked",
    expr("user_id % 1000")
)

print("\nMasked IDs:")
customer_view_secured.select("user_id", "user_id_masked").show(5)

secure_view = customer_view_secured.drop("user_id")

print("\nSecure View:")
secure_view.show(5)

dashboard_view = secure_view.select(
    "product_id", "event_type", "transaction_price", "rating", "user_id_masked"
)

print("\nDashboard View:")
dashboard_view.show(5)

print("\n=== PROJECT COMPLETED SUCCESSFULLY ===")
