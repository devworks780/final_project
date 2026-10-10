#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# shellcheck disable=SC1091
source "$PROJECT_ROOT/load_env.sh"

if [[ ! -f dvc_data/data/baseline/model.json ]]; then
    "$PROJECT_ROOT/.venv/bin/python" -m dvc pull \
        dvc_data/data/baseline/model.json
fi

export AIRFLOW_UID="${AIRFLOW_UID:-$(id -u)}"
docker compose --env-file /dev/null up --build -d

echo "API:        http://localhost:8000/docs"
echo "Prometheus: http://localhost:9090"
echo "Grafana:    http://localhost:3000"
echo "MLflow:     http://localhost:5001"
echo "Airflow:    http://localhost:8080"
