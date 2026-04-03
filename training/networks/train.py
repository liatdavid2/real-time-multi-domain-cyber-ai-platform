import json
import os
import time
from pathlib import Path
from typing import Any
import time

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import boto3
from datetime import datetime

from networks.config import (
    DATA_PATH,
    MODELS_DIR,
    LABEL_COLUMN,
    FEATURE_COLUMNS,
    TEST_SIZE,
    RANDOM_STATE,
    N_ESTIMATORS,
)
from networks.evaluate import evaluate_model
from networks.features import build_features
from networks.utils import make_model_version_path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_NAME = "networks_classification_model"

def upload_directory_to_s3(local_dir: str, bucket: str, prefix: str):
    import os
    s3 = boto3.client("s3")

    for root, _, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_dir)

            s3_key = f"{prefix}/{relative_path}".replace("\\", "/")

            s3.upload_file(local_path, bucket, s3_key)

    print(f"Uploaded directory to s3://{bucket}/{prefix}")


def save_train_distribution(df, feature_columns, path="train_stats.json"):
    stats = {}

    for col in feature_columns:
        values = df[col].dropna().values

        stats[col] = {
            "mean": float(values.mean()),
            "sample": values[:1000].tolist() 
        }

    with open(path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved train distribution → {path}")

def find_best_threshold(y_true, y_prob, min_recall=0.75):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    best_threshold = 0.5
    best_precision = 0

    for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
        if r >= min_recall and p > best_precision:
            best_precision = p
            best_threshold = t

    return float(best_threshold), float(best_precision)


def load_data() -> pd.DataFrame:
    partition = os.getenv("TRAIN_PARTITION")
    base_path = Path(DATA_PATH)

    if partition:
        data_path = base_path / partition
        print(f"Training on partition: {data_path}")

        if not data_path.exists():
            raise FileNotFoundError(f"Partition path does not exist: {data_path}")

        if "hour=" in partition:
            date_path = data_path.parent
        else:
            date_path = data_path
    else:
        date_dirs = sorted(base_path.glob("date=*"))
        if not date_dirs:
            raise FileNotFoundError(f"No date partitions found under: {base_path}")
        date_path = date_dirs[-1]

    print(f"Loading full date: {date_path}")

    dfs = []
    for hour_dir in sorted(date_path.glob("hour=*")):
        print(f"Loading {hour_dir}")

        df = pd.read_parquet(hour_dir)

        hour = hour_dir.name.split("=")[1]
        date = date_path.name.split("=")[1]

        df["hour"] = int(hour)
        df["date"] = date

        dfs.append(df)

    if not dfs:
        raise ValueError(f"No hour partitions found under: {date_path}")

    df = pd.concat(dfs, ignore_index=True)
    return df


def validate_columns(df: pd.DataFrame) -> None:
    required_columns = set(FEATURE_COLUMNS + [LABEL_COLUMN])
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    positive_count = int(y_train.sum())
    negative_count = int(len(y_train) - positive_count)

    if positive_count == 0:
        raise ValueError("Training labels contain no positive samples.")

    pos_weight = negative_count / positive_count

    model = XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="logloss",
        use_label_encoder=False,
        tree_method="hist"
    )

    model.fit(X_train, y_train)
    return model



def save_artifacts(
    model: Any,
    metrics: dict,
    feature_columns: list[str],
    output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_path)

    metrics_path = output_path.with_suffix(".metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    features_path = output_path.with_suffix(".features.json")
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(feature_columns, f, indent=2)


def should_promote_model(
    metrics: dict,
    latency: float,
    prod_metrics: dict | None,
) -> tuple[bool, list[str]]:
    reasons = []

    recall = metrics.get("recall", 0.0)
    precision = metrics.get("precision", 0.0)

    # --- Cyber security priorities ---
    if recall < 0.95:
        reasons.append(f"Recall too low: {recall:.4f} < 0.95")

    if precision < 0.70:
        reasons.append(f"Precision too low: {precision:.4f} < 0.70")

    # latency constraint
    if latency > 0.5:
        reasons.append(f"Latency too high: {latency:.4f} > 0.5")

    # --- Compare to current production ---
    if prod_metrics:
        prod_recall = prod_metrics.get("recall", 0.0)
        prod_precision = prod_metrics.get("precision", 0.0)

        # must not degrade recall
        if recall < prod_recall:
            reasons.append(
                f"Recall worse than production: {recall:.4f} < {prod_recall:.4f}"
            )

        # optional: don't degrade precision too much
        if precision + 0.02 < prod_precision:
            reasons.append(
                f"Precision dropped too much: {precision:.4f} < {prod_precision:.4f}"
            )

    return len(reasons) == 0, reasons


def register_and_promote_model(
    client: MlflowClient,
    metrics: dict,
    latency: float,
) -> None:
    latest_versions = client.get_latest_versions(MODEL_NAME, stages=["None"])

    if not latest_versions:
        print("No new model version found in Registry")
        return

    latest = latest_versions[0]

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=latest.version,
        stage="Staging",
        archive_existing_versions=False,
    )
    print(f"Model version {latest.version} moved to Staging")

    prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    prod_metrics = None

    if prod_versions:
        prod_run_id = prod_versions[0].run_id
        prod_run = client.get_run(prod_run_id)

        prod_metrics = {
            "recall": prod_run.data.metrics.get("recall", 0.0),
            "precision": prod_run.data.metrics.get("precision", 0.0),
        }

        print(f"Current Production Metrics: {prod_metrics}")

    if prod_versions:
        prod_run_id = prod_versions[0].run_id
        prod_run = client.get_run(prod_run_id)
        prod_f1 = prod_run.data.metrics.get("f1")
        print(f"Current Production F1: {prod_f1}")

    promote, reasons = should_promote_model(
    metrics=metrics,
    latency=latency,
    prod_metrics=prod_metrics,
    )

    if promote:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=latest.version,
            stage="Production",
            archive_existing_versions=True,
        )
        print(f"Model version {latest.version} promoted to Production")
    else:
        print("Model NOT promoted to Production")
        for reason in reasons:
            print(f"Promotion check failed: {reason}")


def main() -> None:
    total_start = time.time()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment(MODEL_NAME)

    partition = os.getenv("TRAIN_PARTITION")
    print("USE_S3_DATA =", os.getenv("USE_S3_DATA"))

    df = load_data()
    validate_columns(df)

    df = build_features(df)

    feature_columns = FEATURE_COLUMNS + [
        "bytes_total",
        "pkts_total",
        "byte_ratio",
        "pkt_ratio",
        "load_ratio",
        "ttl_diff",
        "jit_total",
        "mean_size_total",
    ]

    df = df.sort_values(["hour", "stime"])

    unique_hours = sorted(df["hour"].unique())
    if len(unique_hours) <= 1:
        raise ValueError("Not enough hours for hour-based split")

    train_hours = unique_hours[:-1]
    test_hour = unique_hours[-1]

    train_df = df[df["hour"].isin(train_hours)]
    test_df = df[df["hour"] == test_hour]

    print(f"Train hours: {train_hours}")
    print(f"Test hour: {test_hour}")



    X_full = train_df[feature_columns]
    y_full = train_df[LABEL_COLUMN]

    X_train, X_val, y_train, y_val = train_test_split(
        X_full,
        y_full,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_full
    )

    X_test = test_df[feature_columns]
    y_test = test_df[LABEL_COLUMN]

    with mlflow.start_run():

        # -----------------------------
        # Params & Tags
        # -----------------------------
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("partition", partition or "full")
        mlflow.log_param("train_hours", str(train_hours))
        mlflow.log_param("test_hour", str(test_hour))

        mlflow.set_tag("model_type", "XGBoost")
        mlflow.set_tag("feature_version", "v1")
        mlflow.set_tag("data_partition", partition or "full")
        mlflow.set_tag("split_type", "hour_based")
        mlflow.set_tag("monitoring_mode", "offline")

        # -----------------------------
        # Train model
        # -----------------------------
        base_model = train_model(X_train, y_train)

        model = CalibratedClassifierCV(
            base_model,
            method="sigmoid",
            cv="prefit"
        )

        model.fit(X_val, y_val)

        # -----------------------------
        # Predictions
        # -----------------------------
        y_prob = model.predict_proba(X_test)[:, 1]
        print("=== PROBA STATS BEFORE THRESHOLD ===")
        print("min:", y_prob.min())
        print("max:", y_prob.max())
        print("mean:", y_prob.mean())
        y_val_prob = model.predict_proba(X_val)[:, 1]

        print("Sample probabilities:", y_prob[:10])

        threshold, precision_at_threshold = find_best_threshold(
            y_val,
            y_val_prob,
            min_recall=0.75
        )
        threshold = max(0.2, min(threshold, 0.6))


        print(f"Chosen threshold: {threshold}")
        print(f"Precision at threshold: {precision_at_threshold}")

        y_pred = (y_prob > threshold).astype(int)

        # -----------------------------
        # Log params
        # -----------------------------

        importance = base_model.feature_importances_

        feat_imp = pd.DataFrame({
            "feature": X_train.columns,
            "importance": importance
        }).sort_values("importance", ascending=False)

        print("\nTop 10 Most Important Features (XGBoost - Networks):")
        print("----------------------------------------------------")
        print(feat_imp.head(10).to_string(index=False))

        print("\nInterpretation:")
        print("These features contributed the most to detecting anomalous network behavior.")
        print("Higher importance indicates stronger influence on identifying malicious or abnormal traffic patterns.")

        top_features = feat_imp.head(10)
        mlflow.log_dict(
            top_features.to_dict(orient="records"),
            "feature_importance_top10.json"
        )
        mlflow.log_param("calibration", "sigmoid")
        mlflow.log_param("calibration_split", 0.2)
        mlflow.log_param("threshold", threshold)

        # -----------------------------
        # Metrics
        # -----------------------------
        mlflow.log_metric("precision_at_threshold", precision_at_threshold)

        metrics = evaluate_model(y_test, y_pred)
        metrics["partition"] = partition or "full"

        # latency
        sample = X_test.iloc[: min(100, len(X_test))]
        start = time.time()
        _ = model.predict_proba(sample)
        latency = (time.time() - start) / max(len(sample), 1)
        mlflow.log_metric("inference_latency_sec", latency)

        # error rate
        error_rate = float((y_pred != y_test).mean())
        mlflow.log_metric("error_rate", error_rate)

        # prediction distribution
        unique_preds, counts = np.unique(y_pred, return_counts=True)
        for pred_class, count in zip(unique_preds, counts):
            mlflow.log_metric(f"pred_class_{int(pred_class)}", int(count))

        # drift (basic mean drift for logging)
        for col in feature_columns:
            drift = abs(X_full[col].mean() - X_test[col].mean())
            mlflow.log_metric(f"drift_{col}", float(drift))

        # log evaluation metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)

        # -----------------------------
        # Save artifacts
        # -----------------------------

        # 1. Train distribution (for monitoring)
        train_stats_path = "/tmp/train_stats.json"
        save_train_distribution(X_full, feature_columns, train_stats_path)
        train_stats_path = "/tmp/train_stats.json"
        mlflow.log_artifact(train_stats_path, artifact_path="drift")

        # 2. Metrics JSON
        metrics_path = "/tmp/metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact(metrics_path)

        # 3. Features JSON
        features_path = "/tmp/features.json"
        with open(features_path, "w", encoding="utf-8") as f:
            json.dump(feature_columns, f, indent=2)
        mlflow.log_artifact(features_path)

        # -----------------------------
        # Save & Register model
        # -----------------------------
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        # -----------------------------
        # Promote model
        # -----------------------------
        client = MlflowClient()
        register_and_promote_model(
            client=client,
            metrics=metrics,
            latency=latency,
        )

    output_path = make_model_version_path(MODELS_DIR)
    save_artifacts(
        model=model,
        metrics=metrics,
        feature_columns=feature_columns,
        output_path=output_path
    )

    # -----------------------------
    # Upload model to S3
    # -----------------------------
    BUCKET = "intrusion-ml-models"
    S3_PREFIX = "models/intrusion"

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")

    upload_directory_to_s3(
        local_dir=str(output_path.parent),
        bucket=BUCKET,
        prefix=f"{S3_PREFIX}/{timestamp}"
    )

    s3 = boto3.client("s3")

    s3.upload_file(
        str(output_path),
        BUCKET,
        f"{S3_PREFIX}/latest/model.joblib"
    )

    print("Uploaded model version + latest pointer")


    print("=== Proba stats: ===")
    print("min:", y_prob.min())
    print("max:", y_prob.max())
    print("mean:", y_prob.mean())
    print("=== METRICS ===")
    print(metrics)


    print(f"Saved model to: {output_path}")
    print(f"Total pipeline time: {time.time() - total_start:.2f} seconds")


if __name__ == "__main__":
    main()