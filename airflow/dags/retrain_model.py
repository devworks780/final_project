"""Ежемесячное воспроизводимое обновление рекомендательной модели."""

from datetime import datetime, timedelta, timezone

from airflow.operators.bash import BashOperator

from airflow import DAG

PROJECT_ROOT = "/opt/project"

with DAG(
    dag_id="retrain_bank_recommendations",
    description="DVC pull, feature generation, model selection and API reload",
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    schedule="0 3 1 * *",
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "ml-team",
        "retries": 2,
        "retry_delay": timedelta(minutes=10),
    },
    tags=["ml", "dvc", "recommendations"],
) as dag:
    pull_source = BashOperator(
        task_id="pull_source_data",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            "python -m dvc pull "
            "dvc_data/data/origin/train_ver2.csv.zip.dvc"
        ),
    )

    reproduce_and_push = BashOperator(
        task_id="reproduce_and_push",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            "PYTHON_BIN=python ./scripts/run_dvc_pipeline.sh --push"
        ),
        execution_timeout=timedelta(hours=6),
    )

    reload_api = BashOperator(
        task_id="reload_api_model",
        bash_command=(
            'python -c "import os, urllib.request; '
            "request=urllib.request.Request('http://api:8000/reload', method='POST', "
            "headers={'X-Reload-Token': os.environ['MODEL_RELOAD_TOKEN']}); "
            'print(urllib.request.urlopen(request, timeout=30).read().decode())"'
        ),
    )

    pull_source >> reproduce_and_push >> reload_api
