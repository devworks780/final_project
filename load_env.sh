#!/usr/bin/env bash
# Этот файл нужно подключать через: source ./load_env.sh

if [[ -n "${ZSH_VERSION:-}" ]]; then
    SCRIPT_PATH="${(%):-%N}"
else
    SCRIPT_PATH="${BASH_SOURCE[0]}"
fi
PROJECT_ROOT="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "Не найден файл $ENV_FILE" >&2
    return 1 2>/dev/null || exit 1
fi

while IFS='=' read -r raw_key raw_value || [[ -n "$raw_key" ]]; do
    raw_key="${raw_key%$'\r'}"
    raw_value="${raw_value%$'\r'}"
    key="$(printf '%s' "$raw_key" | sed -E 's/^[[:space:]]+|[[:space:]]+$//g')"
    [[ -z "$key" || "$key" == \#* || "$key" == \;* || "$key" == \[* ]] && continue
    if [[ ! "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
        echo "Некорректное имя переменной в $ENV_FILE: $key" >&2
        return 1 2>/dev/null || exit 1
    fi
    value="$(printf '%s' "$raw_value" | sed -E 's/^[[:space:]]+|[[:space:]]+$//g')"
    if [[ "$value" == \"*\" && "$value" == *\" ]]; then
        value="${value:1:${#value}-2}"
    elif [[ "$value" == \'*\' && "$value" == *\' ]]; then
        value="${value:1:${#value}-2}"
    fi
    export "$key=$value"
done < "$ENV_FILE"

export MLFLOW_S3_ENDPOINT_URL="${MLFLOW_S3_ENDPOINT_URL:-https://storage.yandexcloud.net}"
export AWS_ENDPOINT_URL_S3="${AWS_ENDPOINT_URL_S3:-$MLFLOW_S3_ENDPOINT_URL}"
