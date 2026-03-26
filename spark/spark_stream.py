from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    from_json,
    current_timestamp,
    date_format
)

from schema import flow_schema


# ----------------------------------------------------
# Create Spark Session
# ----------------------------------------------------

spark = (
    SparkSession.builder
    .appName("KafkaSparkStreaming")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ----------------------------------------------------
# Read stream from Kafka
# ----------------------------------------------------

kafka_df = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", "kafka:9092")
    .option("subscribe", "unsw-events")
    .option("startingOffsets", "earliest")
    .option("maxOffsetsPerTrigger", 100)   # limit batch size
    .option("failOnDataLoss", "false")
    .load()
)


# ----------------------------------------------------
# Convert Kafka binary value → string
# ----------------------------------------------------

json_df = kafka_df.select(
    col("value").cast("string").alias("json")
)

debug = json_df.writeStream \
    .format("console") \
    .option("truncate", False) \
    .start()
# ----------------------------------------------------
# Parse JSON using schema
# ----------------------------------------------------

parsed_df = (
    json_df
    .select(from_json(col("json"), flow_schema).alias("data"))
    .select("data.*")
)



# ----------------------------------------------------
# Add processing timestamp
# ----------------------------------------------------

with_time_df = parsed_df.withColumn(
    "processing_time",
    current_timestamp()
)


# ----------------------------------------------------
# Partition columns
# ----------------------------------------------------

final_df = (
    with_time_df
    .withColumn("date", date_format(col("processing_time"), "yyyy-MM-dd"))
    .withColumn("hour", date_format(col("processing_time"), "HH"))
)


# ----------------------------------------------------
# Write stream to Parquet
# ----------------------------------------------------

query = (
    final_df.writeStream
    .format("parquet")
    .outputMode("append")
    .option("path", "/app/output/unsw_stream")
    .option("checkpointLocation", "/app/checkpoints/unsw_stream")
    .partitionBy("date", "hour")
    .trigger(processingTime="5 seconds")
    .start()
)


# ----------------------------------------------------
# Keep streaming running
# ----------------------------------------------------

query.awaitTermination()