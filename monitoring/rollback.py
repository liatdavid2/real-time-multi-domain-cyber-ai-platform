from mlflow.tracking import MlflowClient

MODEL_NAME = "intrusion_model"


def rollback_to_previous(client: MlflowClient):
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")

    archived = [
        v for v in versions if v.current_stage == "Archived"
    ]

    if not archived:
        print("No archived model for rollback")
        return False

    archived = sorted(
        archived,
        key=lambda v: int(v.version),
        reverse=True
    )

    target = archived[0]

    print(f"[ROLLBACK] switching to version {target.version}")

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=target.version,
        stage="Production",
        archive_existing_versions=True,
    )

    print("[ROLLBACK] done")
    return True