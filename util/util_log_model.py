import os

import mlflow
import mlflow.data
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from pandas import DataFrame

TRACKING_SERVER_URL = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
EXPERIMENT_NAME = "sprint2_experiment"
RUN_NAME = "sprint2_experiment_run"


def util_log_artifacts(
    artifact_path: list[str] | str,
    experiment_name: str = EXPERIMENT_NAME,
    run_name: str = RUN_NAME,
) -> str:
    """Log a file or all files from a directory to an MLflow run."""
    if isinstance(artifact_path, list):
        for path in artifact_path:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Artifact path does not exist: {path}"
                )
    else:
        if not os.path.exists(artifact_path):
            raise FileNotFoundError(
                f"Artifact path does not exist: {artifact_path}"
            )

    mlflow.set_tracking_uri(TRACKING_SERVER_URL)
    mlflow.set_registry_uri(TRACKING_SERVER_URL)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        for path in (
            artifact_path
            if isinstance(artifact_path, list)
            else [artifact_path]
        ):
            if os.path.isdir(path):
                mlflow.log_artifacts(
                    path, artifact_path=os.path.basename(path)
                )
            else:
                mlflow.log_artifact(path)

        return run.info.run_id


def util_log_model(
    model,
    params,
    metrics,
    requirements,
    model_name,
    artifact_path,
    X_val: DataFrame,
    art_path: str | list[str] = None,
    run_name: str = RUN_NAME,
) -> str:
    mlflow.set_tracking_uri(TRACKING_SERVER_URL)
    mlflow.set_registry_uri(TRACKING_SERVER_URL)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(artifact_path, artifact_path="model")

        if art_path:
            if isinstance(art_path, list):
                for path in art_path:
                    mlflow.log_artifact(path, artifact_path="model")
            else:
                mlflow.log_artifact(art_path, artifact_path="model")

        dataset = mlflow.data.from_pandas(X_val, name="train_dataset")
        mlflow.log_input(dataset, context="training")

        X_val_10 = X_val.head(10)
        input_example = X_val_10.to_dict(orient="records")
        mlflow.log_params(params)

        predictions = pd.DataFrame(
            model.predict(X_val_10),
            columns=[
                "price"
            ],  # <-- Имя вашей целевой переменной / предсказания
        )

        signature = infer_signature(X_val_10, predictions)

        mlflow.sklearn.log_model(
            sk_model=model,  # Передаем наш Pipeline
            artifact_path="model",
            pip_requirements=requirements,
            registered_model_name=model_name,
            input_example=input_example,
            signature=signature,
        )

        return run.info.run_id
