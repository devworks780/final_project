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

export PATH="$(dirname "$PYTHON_BIN"):$PATH"
"$PYTHON_BIN" -m dvc repro

if [[ "${1:-}" == "--push" ]]; then
    "$PYTHON_BIN" -m dvc push
fi
