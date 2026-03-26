# Real-Time Network Traffic Classification (Kafka, Spark, MLflow + Model Registry, Auto-Retraining)

An AI-powered firewall for real-time network traffic classification:

- API that classifies network traffic in real time, deciding whether to ALLOW, ALERT, or BLOCK
- Detects cyber attacks in real time from streaming network traffic
- Automatically retrains on new data
- Monitors model performance and data drift
- Performs automatic rollback when degradation is detected


## 🎥 System Demo

### Real-Time Inference API
Real-time AI firewall deciding ALLOW / ALERT / BLOCK using Rules and ML, trained on the UNSW-NB15 dataset.
[![Watch](docs/images/SwaggerAPI.png)](https://youtu.be/0rco39ZGqtI)

### MLflow Training & Promotion
Model retraining and promotion to production based on performance metrics.
[![Watch](docs/images/demo_thumbnail.png)](https://youtu.be/H85fJlSiw10)

### Monitoring, Drift & Auto-Rollback
Continuous monitoring with drift detection and automatic rollback on performance degradation.
[![Watch](docs/images/monitoring_drift.png)](https://youtu.be/X_ocS6ZOEkY)

## Key Highlights

- Real-time ingestion of network events with Kafka
- Distributed processing with Spark Structured Streaming
- Time-partitioned Parquet storage for scalable retraining
- Time-based evaluation to reduce data leakage
- Threshold optimization with a high-recall cybersecurity focus
- Automatic retraining when new partitions arrive
- MLflow tracking and Model Registry promotion flow
- Monitoring, drift detection, and auto-rollback
- Real-time inference with FastAPI

---

## Problem

Modern security systems generate massive volumes of network events that must be processed reliably in real time for monitoring, analytics, and machine learning.

This architecture is designed for horizontal scalability using Kafka ingestion, Spark streaming, and partitioned storage.

## Dataset

This project uses the **UNSW-NB15 network flow dataset**, a cybersecurity dataset containing detailed network traffic features such as ports, packet counts, bytes, and flow duration.

## Input

Network flow records streamed as events into a Kafka topic.

## Output

Processed events stored as **partitioned Parquet files** by date and hour for scalable analytics.

---

# Architecture

```
          +----------------------+
          |   UNSW-NB15 Dataset  |
          |  Network Flow Events |
          +----------+-----------+
                     |
                     v
          +----------------------+
          |     Kafka Producer   |
          |  Streams JSON events |
          +----------+-----------+
                     |
                     v
          +----------------------+
          |      Kafka Topic     |
          |  Distributed Queue   |
          +----------+-----------+
                     |
                     v
          +----------------------+
          | Spark Structured     |
          | Streaming Engine     |
          |                      |
          | Micro-batch: 100     |
          | Trigger: 5 seconds   |
          | Checkpointing        |
          +----------+-----------+
                     |
                     v
          +----------------------+
          | Partitioned Parquet  |
          |  Data Lake Storage   |
          | date / hour          |
          +----------+-----------+
                     |
          +----------+-----------+
          |                      |
          v                      v
+----------------------+   +----------------------+
| Retraining Watcher   |   |   Monitoring System  |
| Monitors partitions  |   |                      |
| Detects new data     |   | Loads Production     |
|                      |   | Model Metrics        |
+----------+-----------+   |                      |
           |               | Computes Drift:      |
           v               | - Mean               |
+----------------------+   | - KS Test            |
|   Model Training     |   | - PSI                |
|     XGBoost Model    |   |                      |
| + Calibration        |   | Weighted Drift       |
| + Threshold Tuning   |   | (Feature Importance) |
+----------+-----------+   |                      |
           |               | Decision Engine:     |
           v               | - Error Rate         |
+----------------------+   | - Drift Score        |
|        MLflow        |   |                      |
|  Experiment Tracking |   | Auto Rollback        |
|                      |   | (Model Registry)     |
| Params / Metrics     |   +----------+-----------+
| Threshold / PR Curve |              |
| Model Versions       |              |
| Drift Statistics     |              |
+----------+-----------+              |
           |                          |
           v                          v
+----------------------+   +----------------------+
|  Model Registry      |<--| Rollback Controller  |
|  Staging → Production|   | Switch to prev model |
|  Auto Promotion      |   +----------+-----------+
+----------+-----------+              |
           |                          |
           v                          |
+----------------------+              |
|   Versioned Models   |--------------+
|  Model Artifacts     |
| metrics / features   |
+----------+-----------+
           |
           v
+----------------------+
|   Inference API      |
|  FastAPI Service     |
|                      |
| Loads model + thresh |
| Handles JSON input   |
| Feature processing   |
| Real-time prediction |
+----------+-----------+
           |
           v
+----------------------+
|   API Response       |
| prediction / score   |
| decision (ALLOW/BLOCK)|
+----------------------+
```
---

## Notebooks (EDA)

The `notebooks/` directory contains exploratory data analysis used to understand the dataset before training.

Includes:

* Feature and label distributions
* Class imbalance analysis
* Time-based behavior (by hour/date)

Example:
`01_dataset_familiarization_unsw_nb15.ipynb` – initial dataset exploration and validation.

---
## Data Lake Structure

Processed events are stored as **partitioned Parquet files** in a data lake layout.

```text
output/
  unsw_stream/
    date=YYYY-MM-DD/
      hour=HH/
        part-xxxxx.parquet
```

## Partitioning

Data is partitioned by **date** and **hour** based on the processing timestamp.

## File Format

Data is stored in **Parquet**, a columnar format optimized for large-scale analytics.

## Streaming Writes

Spark Structured Streaming continuously appends new files to the correct partition for each micro-batch.

---

# Pipeline Stages

**Dataset**
Structured network flow records used to simulate real-time network telemetry.

**Kafka Producer**
Reads rows from the dataset and streams them as JSON events to Kafka.

**Kafka Topic**
Acts as a durable event queue buffering incoming data.

**Spark Structured Streaming**
Consumes events from Kafka and processes them in micro-batches.

**Parquet Storage**
Writes processed events into a partitioned data lake for efficient querying.

---

# Streaming Configuration

**Batch size**
maxOffsetsPerTrigger = 100 events per batch.

**Processing interval**
trigger(processingTime = 5 seconds).

**Fault tolerance**
Spark checkpoints store offsets to allow recovery after failures.

---
## Why This Architecture Scales

**Kafka ingestion layer**
Kafka can ingest very large streams of events using distributed brokers and topic partitions.

**Decoupled producer and consumer**
Kafka buffers events so producers and processors can scale independently.

**Parallel stream processing**
Spark processes events in parallel across multiple cores or machines.

**Micro-batch streaming**
Spark handles data in small batches which stabilizes processing under high load.

**Backpressure control**
`maxOffsetsPerTrigger` limits how many events are processed per batch.

**Fault tolerance**
Checkpointing allows Spark to recover from failures without losing data.

**Scalable storage**
Partitioned Parquet storage supports efficient querying on large datasets.

---
## Model Training

### Data

Training uses processed flow records derived from the **UNSW-NB15 dataset**, a widely used benchmark for network intrusion detection.

The data is generated by the **Kafka → Spark streaming pipeline** and stored as partitioned Parquet files:

```
/app/output/unsw_stream/
```

Spark continuously writes the processed events into time-based partitions:

```
output/unsw_stream/
    date=2026-03-16/
        hour=13/
        hour=14/
    date=2026-03-17/
    date=2026-03-18/
```

The partition hierarchy is organized by **date → hour**.
Each actual data partition corresponds to a specific date and hour combination, for example:

```
date=2026-03-16/hour=13
date=2026-03-16/hour=14
```

### Model

The system trains a **XGBoost classifier** with balanced class weights to handle the class imbalance typical in intrusion detection datasets.

### Training Process

1. Load Parquet partitions produced by Spark
2. Validate required columns
3. Split data using time-based partitioning:
   - Train on past partitions (hour = t)
   - Test on the next unseen partition (hour = t+1)
4. Train the XGBoost model
5. Evaluate predictions

### Example Training Output

```
Training on partition: /app/output/unsw_stream/date=2026-03-21/hour=13
Loading full date: /app/output/unsw_stream/date=2026-03-21
Loading /app/output/unsw_stream/date=2026-03-21/hour=13
Loading /app/output/unsw_stream/date=2026-03-21/hour=14

Train hours: [13]
Test hour: 14

Chosen threshold: 0.9958533048629761
Precision at threshold: 0.7332082551594746

Classification report:
              precision    recall  f1-score   support

           0       1.00      0.97      0.98     24755
           1       0.73      0.96      0.83      2045

    accuracy                           0.97     26800
   macro avg       0.86      0.96      0.91     26800
weighted avg       0.98      0.97      0.97     26800

Confusion matrix:
[[24044   711]
 [   92  1953]]

Model version 8 moved to Staging
Model version 8 promoted to Production
```

---

## Imbalanced Data Handling

The dataset is naturally imbalanced (benign ≫ attack), reflecting real-world network traffic.

Class balancing techniques (such as oversampling or undersampling) were intentionally avoided because they can distort the data distribution and lead to unrealistic model behavior, especially increasing false positives in production.

Instead, imbalance is handled using:

* Threshold optimization
* Precision-recall tradeoff

This ensures the model remains aligned with real-world conditions.

---

## Threshold Optimization

Instead of using a default classification threshold (0.5), the model selects a threshold based on the precision-recall curve.

Goal:

* Maintain high recall (~0.95) to minimize missed attacks
* Ensure minimum precision (~0.70) to control false positives

This aligns with cybersecurity requirements where detecting attacks is more critical than avoiding false alarms.

---

### MLflow Model Registry (Staging → Production)

![MLflow Model Registry Staging](docs/images/mlflow_registry_stage.png)

The latest model version is automatically promoted to Production, while previous versions remain in Staging for comparison and rollback.
---

### Model Artifacts

Each training run stores versioned artifacts:

```
models/
   2026-03-16-15-02-50/
       intrusion_model.joblib
       intrusion_model.metrics.json
       intrusion_model.features.json
```
---

## Automatic Retraining

The system includes an automatic retraining mechanism triggered by new incoming data partitions.

### How it works

* Spark streaming writes processed data into **time-based partitions**:

  ```
  /app/output/unsw_stream/date=YYYY-MM-DD/hour=HH
  ```

* A dedicated **training service (watcher)** continuously monitors the output directory.

* When a new partition is detected:

  * The system **automatically triggers model retraining**
  * Training runs on the **latest completed partition** (to avoid partial data)

---

### Demo

#### 1. Start the training watcher

```bash
docker compose up training
```

---

#### 2. Simulate new incoming data

```bash
docker exec -it spark bash
cp -r /app/output/unsw_stream/date=2026-03-19 /app/output/unsw_stream/date=2026-03-20
```

---

#### 3. Automatic retraining is triggered

## Retraining Flow

The automatic retraining mechanism is implemented via a lightweight watcher service.

1. **Container startup**

```python
CMD ["python", "retrain_watcher.py"]
```

The training container runs a watcher script continuously.

---

2. **Monitoring loop**

```python
while True:
    partitions = set(get_partitions())
    new_partitions = partitions - known_partitions
```

The system continuously scans for new data partitions.
New folders act as the **trigger** for retraining.

---

3. **New data detection**

```python
if new_partitions:
    print("New partitions detected:", new_partitions)
```

When new data is detected, the system initiates retraining.

---

4. **Selecting stable data**

```python
sorted_parts = sorted(partitions)
previous_partition = sorted_parts[-2]
```

Training is performed on the **latest completed partition**,
avoiding partially written data.

---

5. **Triggering training**

```python
subprocess.run(["python", "train.py"])
```

This command executes the training pipeline and produces a new model.

---

### Training Strategy

* The system does not train on the newest partition
* **It trains on the previous (fully completed) partition**

This ensures the model is trained only on stable, fully written data, avoiding partial or incomplete streaming inputs.

---

## MLflow Tracking

The project uses MLflow for experiment tracking, enabling full visibility into model training, evaluation, and data versioning.


### MLflow UI

### MLflow Experiments Overview

![MLflow Experiments](docs/images/mlflow_experiments.png)

The MLflow dashboard provides a comparison of multiple training runs, including accuracy and F1-score across different model versions.

Consistent performance (~0.97 accuracy and F1) across runs indicates stable training behavior after fixing data leakage and applying time-based evaluation.

---

## Monitoring Metrics

- **Accuracy / F1**
  - Accuracy: ~0.97  
  - Error rate: ~2.8%  
  → Strong performance with realistic errors (no data leakage)

- **Inference Latency**
  - ~0.2 sec per batch  
  → Reasonable for offline batch inference

- **Data Drift**
  - drift_sbytes ≈ 2.9K  
  - drift_dbytes ≈ 21K  
  - drift_sload ≈ 6.6M  
  - drift_stcpb ≈ 147M  

  → Significant differences between train and test data  
  → The test distribution is not identical to training  

  **Interpretation:**
  - ✔ Realistic: reflects changing conditions in streaming environments  
  - ❗ Challenging: makes the prediction task harder

All metrics are logged to MLflow under **offline monitoring mode**.

## MLflow Monitoring Dashboard

### Model Metrics Overview

![MLflow Metrics](docs/images/mlflow_monitoring_metrics.png)

### Full Metrics Table

![MLflow Full Metrics](docs/images/mlflow_monitoring_full_metrics.png)
---

## Imbalanced Data Handling & Cyber-Security Metrics

The dataset is highly imbalanced, with significantly more normal traffic than attack samples. This reflects real-world network conditions, where malicious events are rare.

Instead of applying class balancing techniques (e.g., oversampling or undersampling), the model handles imbalance using:

* **Class weighting (`scale_pos_weight`)** during training
* **Threshold tuning** to control the trade-off between recall and precision

Class balancing was intentionally avoided because it can distort the data distribution and lead to unrealistic performance, especially increasing false positives in production.

### Evaluation Strategy

In cybersecurity, missing an attack (false negative) is far more critical than raising false alerts. Therefore, model evaluation is focused on the **attack class (positive class)** rather than global averages.

The promotion criteria prioritize:

* **High Recall (≥ 0.95)** – to minimize missed attacks
* **Minimum Precision (≥ 0.70)** – to control alert noise
* **Latency constraints** – to ensure real-time performance

Metrics such as macro or weighted averages are not used for decision-making, as they can hide poor performance on the minority attack class.

This approach ensures that the model aligns with real-world security requirements rather than optimizing for generic ML metrics.

---

# Inference API

FastAPI service for real-time network traffic classification, inspired by firewall systems.

### POST `/predict`
---

# Examples (Execution Order)

## 1. Normal Traffic (ML)

### Request

```json
{
  "sttl": 31,
  "dttl": 29,
  "ct_state_ttl": 0,
  "dload": 500000
}
```

### Response

```json
{
  "prediction": 0,
  "ml_score": 0.0022239708341658115,
  "decision": "ALLOW",
  "decision_source": "ML",
  "attack_hypothesis": [],
  "reasons": []
}
```

---

## 2. Attack (ML)

### Request

```json
{
  "sttl": 254,
  "dttl": 252,
  "ct_state_ttl": 1,
  "dbytes": 0
}
```

### Response

```json
{
  "prediction": 1,
  "ml_score": 0.9312506318092346,
  "decision": "BLOCK",
  "decision_source": "ML",
  "attack_hypothesis": [],
  "reasons": []
}
```

---

## 3. DoS Attack (RULE)

### Request

```json
{
  "spkts": 200,
  "sload": 120000,
  "sintpkt": 0.0005
}
```

### Response

```json
{
  "decision": "BLOCK",
  "decision_source": "RULE",
  "attack_hypothesis": ["DoS"],
  "reasons": [
    "High traffic rate that may overload the system."
  ],
  "explanations": [
    "This rule is used to detect Denial of Service behavior. It looks for very high packet volume, high traffic load, and very short time between packets. Together, these signals may indicate an attempt to flood the target and reduce its availability."
  ]
}
```

---

## 4. Reconnaissance (RULE)

### Request

```json
{
  "spkts": 60,
  "ct_src_dport_ltm": 5,
  "sintpkt": 0.01
}
```

### Response

```json
{
  "decision": "ALERT",
  "decision_source": "RULE",
  "attack_hypothesis": ["Reconnaissance"],
  "reasons": [
    "Suspicious scanning behavior indicating reconnaissance activity."
  ],
  "explanations": [
    "This rule detects reconnaissance activity such as port scanning or probing. It looks for a high number of packets sent, multiple destination ports, and short intervals between packets."
  ]
}
```
---

# Decision Flow

1. Rules are evaluated first
2. If no rule matches → ML model runs

---

## Monitoring, Drift Detection & Auto-Rollback

This system includes a **production monitoring layer** that tracks both model performance and data distribution in real time.

### What it does

* Monitors **error rate** from MLflow
* Detects **data drift** using:

  * Mean difference
  * KS test
  * PSI
* Computes **weighted drift score** using XGBoost feature importance

### Decision Logic

* **Rollback triggered if:**

  * High error rate
  * OR drift + performance degradation

* **No action if:**

  * Drift exists but model performance is stable

### Auto-Rollback

Automatically switches to the previous stable model via MLflow Model Registry.

```text
[DRIFT SUMMARY]
Weighted drift score: 0.60

[DECISION] ISSUE DETECTED
→ ACTION: ROLLBACK
```

---

## Why it matters

* Prevents silent model degradation
* Handles changing network behavior (e.g. new attacks)
* Enables **self-healing ML systems**

---
## Notes

* Model is loaded dynamically from MLflow
* Uses calibrated probabilities + learned threshold
* Supports partial inputs (missing fields are filled automatically)
---
## Cybersecurity Policy (Rule Engine)
This system includes a rule-based policy layer that detects known attack patterns based on network flow features from the UNSW-NB15 dataset.
---

## Attack Types & Feature-Based Indicators

**DoS**  
Very high traffic sent in a short time to overload a system and make it unavailable.
Indicators: high `spkts` (source packets), `dpkts` (destination packets), high `sload` (source bytes/sec) / `dload` (destination bytes/sec), very low `sintpkt` (source inter-packet time) / `dintpkt` (destination inter-packet time), high `ct_srv_src` (connections per service from source), high `ct_dst_ltm` (connections to destination over time).

---

**Reconnaissance**  
Scanning activity used to find open ports, services, or network structure.
Indicators: short `dur` (connection duration), low `sbytes` (bytes sent) / `dbytes` (bytes received), high `ct_dst_ltm` (connections to destination), high `ct_srv_dst` (connections per service to destination), high `ct_dst_src_ltm` (connections between same src-dst), multiple ports via `ct_src_dport_ltm` (connections per destination port).

---

**Exploits**  
Attempts to take advantage of system or application vulnerabilities.
Indicators: `trans_depth` (HTTP transaction depth), `res_bdy_len` (response body size), `ct_flw_http_mthd` (HTTP methods count), `is_ftp_login` (FTP login attempt flag), abnormal `tcprtt` (TCP round-trip time), unusual `sttl` (source TTL) / `dttl` (destination TTL), high `ct_state_ttl` (state-TTL combinations).

---

**Fuzzers**  
Sending random or malformed data to break or confuse a system.
Indicators: high `spkts` (source packets) with low `sbytes` (bytes sent), small `smeansz` (average source packet size), small `dmeansz` (average destination packet size), high `sjit` (source jitter) / `djit` (destination jitter).

---

**Backdoors**  
Hidden communication that allows ongoing unauthorized access to a system.
Indicators: long `dur` (connection duration), low but stable `sbytes` (bytes sent) / `dbytes` (bytes received), repeated connections via `ct_src_ltm` (connections from source over time), stable `tcprtt` (round-trip time).

---

**Analysis**  
Probing activity used to understand how a system behaves and find weak points.
Indicators: moderate `dur` (connection duration), high `ct_dst_ltm` (connections to destination), moderate `trans_depth` (application interaction depth), variable `tcprtt` (round-trip time).

---

**Shellcode**  
Attempts to send and execute malicious code on the target system.
Indicators: `trans_depth` (application interaction), abnormal `res_bdy_len` (response size), very low `tcprtt` (round-trip time), low `synack` (SYN-ACK time), inconsistent `sttl` (source TTL) / `dttl` (destination TTL).

---

**Worms**  
Malware that spreads automatically across many systems or ports.
Indicators: high `ct_dst_ltm` (connections to destination), high `ct_srv_dst` (connections per service to destination), high `ct_dst_src_ltm` (connections between src-dst), multiple ports via `ct_src_dport_ltm` (destination ports per source), high `ct_dst_sport_ltm` (source ports per destination).

---

**Generic**  
Unusual traffic that does not match a specific attack but looks abnormal.
Indicators: abnormal ratio between `sbytes` (bytes sent) and `dbytes` (bytes received), unusual `smeansz` (avg source packet size) / `dmeansz` (avg destination packet size), high `sjit` (source jitter) / `djit` (destination jitter).

---

**Normal Traffic**  
Regular network activity with stable and expected behavior.
Indicators: balanced `sbytes` (bytes sent) / `dbytes` (bytes received), stable `sintpkt` (source inter-packet time) / `dintpkt` (destination inter-packet time), low `ct_*` (connection counters), consistent `tcprtt` (round-trip time).

---
## Handling Large-Scale Data

Using all historical data for training is often not feasible due to memory, latency, and compute constraints.

This project applies a **time-aware training strategy**:

- Training is performed on recent data partitions (by hour)
- Evaluation is done on the next unseen time window
- Data is incrementally accumulated and retrained periodically
---

## Deployment

The system runs using **Docker Compose** and can be deployed on distributed infrastructure such as Kubernetes or cloud platforms.

---

## Fault Tolerance

Spark **checkpointing** stores Kafka offsets and streaming state, allowing the pipeline to resume processing after failures without data loss.

---

## Backpressure Control

Backpressure is controlled using `maxOffsetsPerTrigger`, which limits how many events Spark consumes per micro-batch and prevents overload during traffic spikes.

---
## Quick Start

### Clean Environment and Rebuild

```bash
# stop containers and remove volumes
docker compose down -v

# rebuild images from scratch
docker compose build --no-cache

# start the full pipeline
docker compose up
```

### View Logs

```bash
docker compose logs spark
docker compose logs producer
```

### Clean Streaming Output

If the streaming job needs to be restarted from a clean state:

```bash
docker exec -it spark bash
rm -rf /app/output/*
rm -rf /app/checkpoints/*
exit
```

### Inspect Data with Spark

You can inspect the generated Parquet data using the Spark shell.

```bash
docker exec -it spark bash
/opt/spark/bin/pyspark
```

Example query:

```python
df = spark.read.parquet("/app/output/unsw_stream")
df.show(2)
```

### Run Model Training

After the streaming pipeline has produced data, the training job can be executed:

```bash
docker compose down training
docker compose build training
docker compose up training
```

The training job reads the generated Parquet partitions and trains the intrusion detection model.
