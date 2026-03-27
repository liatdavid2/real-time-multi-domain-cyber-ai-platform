from datetime import datetime
from pathlib import Path

def make_model_version_path(models_dir: str | Path) -> Path:

    models_dir = Path(models_dir)

    # full timestamp
    run_id = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")

    run_dir = models_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir / "intrusion_model.joblib"