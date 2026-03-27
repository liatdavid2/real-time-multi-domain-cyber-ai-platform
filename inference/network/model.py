import threading
import mlflow
from mlflow.tracking import MlflowClient

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

    try:
        model_info = _get_production_model_info()
    except Exception as e:
        if _model is not None:
            print(f"[WARN] MLflow unavailable, using cached model v{_current_version}: {e}")
            return _model, _threshold
        raise RuntimeError(f"MLflow unavailable and no model loaded: {e}")

    # reload only if version changed
    if _model is None or str(model_info.version) != str(_current_version):
        with _lock:
            if _model is None or str(model_info.version) != str(_current_version):
                print(f"[INFO] Loading model version {model_info.version}")

                model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

                try:
                    _model = mlflow.sklearn.load_model(model_uri)
                    _current_version = str(model_info.version)

                    # Load threshold from MLflow
                    run = _client.get_run(model_info.run_id)
                    _threshold = float(run.data.params.get("threshold", 0.5))
                    _threshold = max(0.2, min(_threshold, 0.6))

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