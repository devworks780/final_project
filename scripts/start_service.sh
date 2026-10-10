#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Не найден Python окружения: $PYTHON_BIN" >&2
    exit 1
fi
if [[ ! -f dvc_data/data/baseline/model.json ]]; then
    echo "Не найдена модель. Выполните dvc pull или dvc repro train_baseline." >&2
    exit 1
fi

export MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/dvc_data/data/baseline/model.json}"
exec "$PYTHON_BIN" -m uvicorn app.main:app \
    --host "${API_HOST:-127.0.0.1}" \
    --port "${API_PORT:-8000}"
