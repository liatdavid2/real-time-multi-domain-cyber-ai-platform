# Real-Time Multi-Domain Cyber AI Platform

## Overview

This project is a multi-domain machine learning platform for cybersecurity, designed to train and deploy models across different domains such as:

* Malware detection (static PE analysis – EMBER dataset)
* Network anomaly detection (streaming via Spark + Kafka)

The system supports modular training pipelines and allows selecting which domain to train using a unified interface.

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
