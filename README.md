# Real-Time Multi-Domain Cyber AI Platform

## Overview

This project is a multi-domain machine learning platform for cybersecurity, designed to train and deploy models across different domains such as:

* Malware detection (static PE analysis – EMBER dataset)
* Network anomaly detection (streaming via Spark + Kafka)

The system supports modular training pipelines and allows selecting which domain to train using a unified interface.

---
## System Demo

### BOTH Training + MLflow Production Promotion

This demo showcases a multi-domain training pipeline running in `both` mode.  
It trains two independent models: one for network attacks classification and one for malware classification, each using its own data pipeline and feature engineering.

At the end of the run, all experiments are logged to MLflow, and the best-performing model is automatically promoted to `Production`.


[![Watch](docs/images/training-both-demo.png)](https://youtu.be/viFWanNg0rY)
---

## Key Idea

A single training service supports multiple domains:

```
User → run.py (--mode) → Domain Pipeline → Model Training → Evaluation → MLflow Registry
```

---

## Supported Modes

```bash
python run.py --mode <mode>
```

| Mode     | Description                           |
| -------- | ------------------------------------- |
| networks | Train network anomaly detection model |
| malwares | Train malware classification model    |
| both     | Train both pipelines sequentially     |

---

## System Architecture

### Offline Training Pipeline

```
Data → Feature Engineering → Clean Features → Train/Val/Test Split
     → Model Training (XGBoost)
     → Evaluation → Feature Importance
     → MLflow Logging → Model Registry
```

### Infrastructure

* Docker Compose
* MLflow (tracking + registry)
* Kafka (streaming)
* Spark (network pipeline)

---

## Project Structure

```
training/
├── malwares/
│   ├── train.py
│   ├── features.py
│
├── networks/
│   ├── train.py
│
├── common/
│   ├── utils.py
│
├── run.py
```

---

## Running the Training

From the infrastructure folder:

```bash
docker compose run training python run.py --mode malwares
```

---

## Example Training Output

```
Mode selected: malwares
Running MALWARES training...

Loading data/ember_2017/ember_2017.jsonl
Loaded samples: 15927

Label distribution:
1    10405
0     5522

Train shape: (11148, 22)
Val shape: (2389, 22)
Test shape: (2390, 22)
```

---

## Model Performance

```
Accuracy: 0.93
ROC AUC: 0.9745

Class 0 (Benign):
Precision: 0.92 | Recall: 0.88 | F1: 0.90

Class 1 (Malware):
Precision: 0.94 | Recall: 0.96 | F1: 0.95
```

---

## Feature Importance (Top Signals)

```
Top 10 Most Important Features (XGBoost):

general.has_debug                              0.317
general.has_relocations                        0.081
header.optional.sizeof_heap_commit             0.072
general.exports                                0.066
header.optional.major_subsystem_version        0.059
header.optional.major_operating_system_version 0.044
header.optional.minor_operating_system_version 0.041
general.has_resources                          0.037
general.imports                                0.031
header.optional.minor_subsystem_version        0.030
```

### Interpretation

These features reflect structural properties of PE files:

* Debug flags → uncommon in production binaries
* Imports/exports → behavioral signatures
* Memory layout → low-level execution patterns

The model learns meaningful signals aligned with real malware characteristics.

---

## Model Lifecycle (MLflow)

* Automatic model registration
* Versioning (v1, v2, v3, ...)
* Promotion to production

Example:

```
Model 4 → Production
```
---
## Running Network Training

From the infrastructure folder:

```bash
docker compose run training python run.py --mode networks
```
---

## Example Training Output (Networks)

```text
Mode selected: networks
Running NETWORKS training...

Loading full date: /app/output/unsw_stream/date=2026-03-21
Loading /app/output/unsw_stream/date=2026-03-21/hour=13
Loading /app/output/unsw_stream/date=2026-03-21/hour=14

Train hours: [13]
Test hour: 14
```

---

## Model Behavior (Probability Distribution)

```text
min: 0.0022
max: 0.9335
mean: 0.1009

Chosen threshold: 0.6
Precision at threshold: 1.0
```

### Interpretation

* Most traffic is low-risk (low probability)
* High-confidence anomalies are clearly separated
* Threshold tuning prioritizes high recall with strong precision

---

## Feature Importance (Network Model)

```text
Top 10 Most Important Features (XGBoost - Networks):

sttl              0.677
ct_state_ttl      0.304
ct_dst_ltm        0.004
dttl              0.002
ct_src_ltm        0.001
sbytes            0.001
load_ratio        0.001
byte_ratio        0.001
smeansz           0.0009
sjit              0.0007
```

### Interpretation

The model relies heavily on TTL-based features:

* `sttl`, `dttl` → packet routing characteristics
* `ct_state_ttl` → connection state + TTL behavior
* `ct_*` features → connection frequency patterns

These signals are highly effective for detecting:

* Network scanning
* Spoofing
* Abnormal routing behavior

---

## Model Performance

```text
Accuracy: 0.97

Class 0 (Normal):
Precision: 1.00 | Recall: 0.97 | F1: 0.98

Class 1 (Attack):
Precision: 0.72 | Recall: 1.00 | F1: 0.84
```

---

## Confusion Matrix

```text
[[23957   798]
 [    2  2043]]
```

### Interpretation

* Very high recall (almost no missed attacks)
* Some false positives (acceptable in security systems)

---

## Model Lifecycle (MLflow + Promotion Logic)

```text
Model version 6 → Production
```

### Promotion Strategy

The model is promoted only if:

* Recall ≥ previous production model
* Precision does not degrade significantly
* Latency constraints are met

---

## Deployment Artifacts

* Model saved locally:

```text
/app/models/<timestamp>/intrusion_model.joblib
```

* Uploaded to S3:

```text
s3://intrusion-ml-models/models/intrusion/<timestamp>
```

* Latest model pointer:

```text
models/intrusion/latest/model.joblib
```
