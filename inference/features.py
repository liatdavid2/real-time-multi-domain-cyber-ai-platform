import json
import os
import pandas as pd
from mlflow.tracking import MlflowClient
from model import load_model
from training.features import build_features

_client = MlflowClient()

FEATURE_NAMES = None


def load_feature_names():
    global FEATURE_NAMES

    if FEATURE_NAMES is not None:
        return FEATURE_NAMES

    # load model (to get run_id)
    model, _ = load_model()

    # get production model info
    versions = _client.get_latest_versions("intrusion_model", stages=["Production"])
    model_info = versions[0]

    run_id = model_info.run_id

    # download artifact
    local_path = _client.download_artifacts(run_id, "features.json")

    print(f"[INFO] Loaded features from MLflow run: {run_id}")
    print(f"[INFO] Features path: {local_path}")

    with open(local_path, "r") as f:
        FEATURE_NAMES = json.load(f)

    return FEATURE_NAMES


def build_features_from_json(data: dict):
    df = pd.DataFrame([data])
    df = df.apply(pd.to_numeric, errors="coerce")

    # load train stats
    stats_path = _client.download_artifacts(
        _client.get_latest_versions("intrusion_model", stages=["Production"])[0].run_id,
        "drift/train_stats.json"
    )

    with open(stats_path) as f:
        stats = json.load(f)

    # fill missing with TRAIN mean
    for col in df.columns:
        if df[col].isna().any() and col in stats:
            df[col] = df[col].fillna(stats[col]["mean"])

    df = build_features(df)

    feature_names = load_feature_names()

    for col in feature_names:
        if col not in df:
            df[col] = 0

    df = df[feature_names]

    return df