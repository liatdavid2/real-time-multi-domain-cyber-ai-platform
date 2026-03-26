import threading

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import boto3
import joblib
import json
import os

MODEL_NAME = "intrusion_model"
MODEL_STAGE = "Production"

_client = MlflowClient()

_model = None
_current_version = None
_threshold = 0.5
_lock = threading.Lock()


def _get_production_model_info():
    versions = _client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
    if not versions:
        raise RuntimeError(f"No model found in stage '{MODEL_STAGE}'")
    return versions[0]


def load_model():
    global _model, _current_version, _threshold

    use_s3 = os.getenv("USE_S3", "false").lower() == "true"

    # ==================================================
    # S3 MODE (AWS)
    # ==================================================
    if use_s3:
        MODEL_PATH = "/app/models/model.joblib"
        METRICS_PATH = "/app/models/metrics.json"

        os.makedirs("/app/models", exist_ok=True)

        bucket = os.getenv("MODEL_BUCKET", "intrusion-ml-models")
        model_key = os.getenv("MODEL_KEY", "models/intrusion/latest/model.joblib")
        metrics_key = os.getenv("MODEL_METRICS_KEY", "models/intrusion/latest/metrics.json")

        if _model is None:
            print("[MODEL] Loading from S3")



            try:
                s3 = boto3.client("s3")

                if not os.path.exists(MODEL_PATH):
                    print(f"[S3] Downloading model from s3://{bucket}/{model_key}")
                    s3.download_file(bucket, model_key, MODEL_PATH)

                    try:
                        s3.download_file(bucket, metrics_key, METRICS_PATH)
                    except Exception as e:
                        print(f"[WARN] Metrics not found in S3: {e}")

                _model = joblib.load(MODEL_PATH)

                _threshold = 0.5
                if os.path.exists(METRICS_PATH):
                    try:
                        with open(METRICS_PATH) as f:
                            _threshold = json.load(f).get("threshold", 0.5)
                    except Exception as e:
                        print(f"[WARN] Failed reading metrics: {e}")

                _current_version = "s3-latest"
                print("[MODEL] Loaded successfully from S3")

            except Exception as e:
                raise RuntimeError(f"S3 model loading failed: {e}")

        return _model, _threshold

    # ==================================================
    # MLflow MODE (ONLY if USE_S3=false)
    # ==================================================
    try:
        model_info = _get_production_model_info()
    except Exception as e:
        if _model is not None:
            print(f"[WARN] MLflow unavailable, using cached model v{_current_version}: {e}")
            return _model, _threshold
        raise RuntimeError(f"MLflow unavailable and no model loaded: {e}")

    if _model is None or str(model_info.version) != str(_current_version):
        with _lock:
            if _model is None or str(model_info.version) != str(_current_version):
                print(f"[INFO] Loading model version {model_info.version}")

                model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

                try:
                    _model = mlflow.sklearn.load_model(model_uri)
                    _current_version = str(model_info.version)

                    run = _client.get_run(model_info.run_id)
                    _threshold = float(run.data.params.get("threshold", 0.5))

                    print(f"[INFO] Model loaded successfully (version={_current_version})")
                    print(f"[INFO] Threshold loaded: {_threshold}")

                except Exception as e:
                    if _model is not None:
                        print(f"[WARN] Failed loading new model, using old version {_current_version}: {e}")
                        return _model, _threshold
                    raise RuntimeError(f"Failed to load model: {e}")

    return _model, _threshold

def get_model_version():
    return _current_version