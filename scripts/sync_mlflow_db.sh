#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# shellcheck disable=SC1091
source "$PROJECT_ROOT/load_env.sh"

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
DB_PATH="$PROJECT_ROOT/dvc_data/data/mlflow_db/mlflow.db"
DVC_FILE="$DB_PATH.dvc"

if [[ ! -f "$DB_PATH" ]]; then
    echo "Не найдена MLflow DB: $DB_PATH" >&2
    exit 1
fi

# Снимок работающей SQLite-базы может быть несогласованным. На macOS lsof
# позволяет обнаружить открытый MLflow DB-файл до запуска dvc add.
if command -v lsof >/dev/null 2>&1 && lsof "$DB_PATH" >/dev/null 2>&1; then
    echo "MLflow DB сейчас открыта процессом. Сначала остановите MLflow через Ctrl+C." >&2
    exit 1
fi

"$PYTHON_BIN" - "$DB_PATH" <<'PY'
import sqlite3
import sys

path = sys.argv[1]
with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
    result = connection.execute("PRAGMA integrity_check").fetchone()[0]
if result != "ok":
    raise SystemExit(f"SQLite integrity_check failed: {result}")
print("SQLite integrity_check: ok")
PY

"$PYTHON_BIN" -m dvc add "$DB_PATH"
"$PYTHON_BIN" -m dvc push "$DVC_FILE"

echo "MLflow DB сохранена в DVC remote. Добавьте обновлённый метафайл в Git:"
echo "  git add dvc_data/data/mlflow_db/mlflow.db.dvc"
