#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# shellcheck disable=SC1091
source "$PROJECT_ROOT/load_env.sh"

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Не найден Python окружения: $PYTHON_BIN" >&2
    exit 1
fi

MLFLOW_HOST="${MLFLOW_HOST:-127.0.0.1}"
MLFLOW_PORT="${MLFLOW_PORT:-5001}"
MLFLOW_DB_PATH="$PROJECT_ROOT/dvc_data/data/mlflow_db/mlflow.db"
MLFLOW_DB_DVC_FILE="$MLFLOW_DB_PATH.dvc"
MLFLOW_BACKEND_STORE_URI="${MLFLOW_BACKEND_STORE_URI:-sqlite:///$MLFLOW_DB_PATH}"
MLFLOW_ARTIFACT_ROOT="${MLFLOW_ARTIFACT_ROOT:-s3://${S3_BUCKET_NAME}/final_project/mlflow-artifacts}"

# DVC восстанавливает только управляемую SQLite-базу. Пользовательский URI,
# например PostgreSQL, не требует локального файла и не затрагивается.
if [[ "$MLFLOW_BACKEND_STORE_URI" == "sqlite:///$MLFLOW_DB_PATH" ]]; then
    mkdir -p "$(dirname "$MLFLOW_DB_PATH")"
    if [[ ! -f "$MLFLOW_DB_PATH" ]]; then
        if [[ ! -f "$MLFLOW_DB_DVC_FILE" ]]; then
            echo "Нет локальной MLflow DB и метафайла $MLFLOW_DB_DVC_FILE" >&2
            echo "Сначала восстановите .dvc-файл из Git или задайте MLFLOW_BACKEND_STORE_URI." >&2
            exit 1
        fi
        echo "Локальная MLflow DB отсутствует — восстанавливаю снимок из DVC/S3..."
        "$PYTHON_BIN" -m dvc pull "$MLFLOW_DB_DVC_FILE"
    fi
fi

exec "$PYTHON_BIN" -m mlflow server \
    --host "$MLFLOW_HOST" \
    --port "$MLFLOW_PORT" \
    --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" \
    --default-artifact-root "$MLFLOW_ARTIFACT_ROOT"
