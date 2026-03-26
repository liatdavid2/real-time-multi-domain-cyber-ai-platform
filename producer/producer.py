import json
import time
import pandas as pd

from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable


KAFKA_BROKER = "kafka:9092"
TOPIC = "unsw-events"
BATCH_SIZE = 100


# Load dataset
df = pd.read_parquet("/data/UNSW_Flow.parquet")


# Use binary label as the training label
if "binary_label" in df.columns:
    df["label"] = df["binary_label"]


# Remove other labels if exist
if "attack_label" in df.columns:
    df = df.drop(columns=["attack_label"])

if "binary_label" in df.columns:
    df = df.drop(columns=["binary_label"])


producer = None

# Retry until Kafka is ready
while producer is None:
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BROKER,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            linger_ms=50,
            batch_size=16384,
            acks="all",
            retries=5
        )
        print("Connected to Kafka")
    except NoBrokersAvailable:
        print("Kafka not ready, retrying in 3 seconds...")
        time.sleep(3)


batch = []

for _, row in df.iterrows():

    batch.append(row.to_dict())

    if len(batch) >= BATCH_SIZE:

        for event in batch:
            producer.send(TOPIC, event)

        producer.flush()

        print(f"Sent batch of {BATCH_SIZE} events")

        batch = []

        # simulate streaming pressure
        time.sleep(0.5)


# Send remaining events
if batch:
    for event in batch:
        producer.send(TOPIC, event)

    producer.flush()


print("Finished sending events")
producer.close()