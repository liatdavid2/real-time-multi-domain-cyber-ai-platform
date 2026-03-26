import time
import os
import mlflow
from mlflow.tracking import MlflowClient
from rollback import rollback_to_previous
import datetime
from drift import mean_drift, ks_drift, population_stability_index
import numpy as np
import json
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
)

MODEL_NAME = "intrusion_model"
CHECK_INTERVAL = 30
ERROR_RATE_THRESHOLD = 0.1

DATA_PATH = Path("/app/output")  # parquet from Spark


# -----------------------------
# Load train distribution from MLflow
# -----------------------------
def get_train_stats(client):
    try:
        # -----------------------------
        # Get production model run
        # -----------------------------
        versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])

        if not versions:
            print("[DRIFT] No production model")
            return None

        run_id = versions[0].run_id
        print(f"[DEBUG] Using production run_id = {run_id}")

        # -----------------------------
        # Direct download (no listing!)
        # -----------------------------
        artifact_path = "drift/train_stats.json"

        try:
            local_path = client.download_artifacts(run_id, artifact_path)
            print(f"[DRIFT] Loaded from {artifact_path}")
        except Exception as e:
            print(f"[DRIFT] train_stats.json not found: {e}")
            return None

        # -----------------------------
        # Load JSON
        # -----------------------------
        with open(local_path, "r") as f:
            stats = json.load(f)

        print(f"[DRIFT] Loaded train stats from run {run_id}")
        return stats

    except Exception as e:
        print(f"[DRIFT ERROR] {e}")
        return None


# -----------------------------
# Load current production data
# -----------------------------
def load_current_data():
    files = sorted(DATA_PATH.rglob("*.parquet"))

    if not files:
        print("[DATA] No parquet files found")
        return None

    latest = files[-1]
    print(f"[DATA] loading current data from {latest}")

    df = pd.read_parquet(latest)
    return df


# -----------------------------
# MLflow metrics
# -----------------------------
def get_production_metrics(client):
    versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    if not versions:
        return None

    run_id = versions[0].run_id
    run = client.get_run(run_id)

    return run.data.metrics


# -----------------------------
# Drift computation
# -----------------------------
def compute_drift_weighted(client):
    train_stats = get_train_stats(client)

    if train_stats is None:
        print("[DRIFT] skipped (no train stats)")
        return False

    current_df = load_current_data()

    if current_df is None:
        print("[DRIFT] skipped (no current data)")
        return False

    # -----------------------------
    # Load feature importance
    # -----------------------------
    feature_importance = {}
    try:
        versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        run_id = versions[0].run_id
        run = client.get_run(run_id)

        feature_importance = run.data.tags.get("feature_importance")
        if feature_importance:
            feature_importance = json.loads(feature_importance)
        else:
            feature_importance = {}
    except Exception:
        feature_importance = {}

    drift_results = []

    for col, train_info in train_stats.items():
        if col not in current_df.columns:
            continue

        current_values = current_df[col].dropna().values
        train_values = np.array(train_info["sample"])

        if len(current_values) < 50:
            continue

        mean_flag, _ = mean_drift(train_info["mean"], np.mean(current_values))
        ks_flag, _ = ks_drift(train_values, current_values)
        psi_flag, _ = population_stability_index(train_values, current_values)

        votes = sum([mean_flag, ks_flag, psi_flag])
        feature_drift = votes >= 2

        importance = feature_importance.get(col, 1.0)

        drift_results.append((feature_drift, importance))

    if not drift_results:
        print("[DRIFT] no features evaluated")
        return False

    # -----------------------------
    # Weighted decision
    # -----------------------------
    weighted_drift = sum(imp for drift, imp in drift_results if drift)
    total_importance = sum(imp for _, imp in drift_results)

    drift_ratio = weighted_drift / (total_importance + 1e-8)

    total_drift = drift_ratio > 0.4

    # -----------------------------
    # Clean summary
    # -----------------------------
    drifted = sum(1 for d, _ in drift_results if d)
    total = len(drift_results)

    print("\n[DRIFT SUMMARY]")
    print(f"Features drifted: {drifted}/{total}")
    print(f"Weighted drift score: {drift_ratio:.3f}")
    print(f"Drift detected: {total_drift}")

    return total_drift

# -----------------------------
# Main check
# -----------------------------
def check_and_rollback():
    print(f"\n[MONITOR] {datetime.datetime.now()} running check...")

    client = MlflowClient()

    metrics = get_production_metrics(client)

    if not metrics:
        print("[MONITOR] No production model found")
        return

    error_rate = metrics.get("error_rate", 0.0)

    print(f"[MONITOR] error_rate={error_rate:.3f} (threshold={ERROR_RATE_THRESHOLD})")

    # -------- DRIFT --------
    drift_detected = compute_drift_weighted(client)

    # -------- DECISION --------
    rollback = False
    reason = None

    if error_rate > ERROR_RATE_THRESHOLD:
        rollback = True
        reason = "High error rate"

    elif drift_detected and error_rate > 0.02:
        rollback = True
        reason = "Drift + performance degradation"

    # -------- ACTION --------
    if rollback:
        print("\n[DECISION] ISSUE DETECTED")
        print(f"Reason: {reason}")
        print("→ ACTION: ROLLBACK\n")
        rollback_to_previous(client)

    else:
        print("\n[DECISION] System stable")
        print("→ ACTION: NO ROLLBACK\n")


# -----------------------------
# Loop
# -----------------------------
if __name__ == "__main__":
    while True:
        check_and_rollback()
        time.sleep(CHECK_INTERVAL)