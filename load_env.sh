#!/bin/bash
# Экспорт переменных окружения из файла .env

ENV_FILE="$(dirname "$0")/.env"

while IFS='=' read -r key value; do
    # Пропускаем пустые строки, комментарии и секции [...]
    [ -z "$key" ] && continue
    case "$key" in
        \#*) continue ;;
        \[*) continue ;;
    esac
    # Убираем пробелы вокруг ключа и значения
    key="${key// /}"
    value="${value// /}"
    export "$key=$value"
done < "$ENV_FILE"
