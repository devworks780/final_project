#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="$PROJECT_ROOT/dvc_data/data/eda/plots"
TARGET_DIR="$PROJECT_ROOT/docs/images"

PLOTS=(
    monthly_observations.png
    product_prevalence.png
    product_additions.png
    missingness.png
    monthly_spikes.png
)

mkdir -p "$TARGET_DIR"

for plot in "${PLOTS[@]}"; do
    if [[ ! -f "$SOURCE_DIR/$plot" ]]; then
        echo "Не найден EDA-график: $SOURCE_DIR/$plot" >&2
        echo "Сначала выполните dvc pull или dvc repro." >&2
        exit 1
    fi
    cp "$SOURCE_DIR/$plot" "$TARGET_DIR/$plot"
done

echo "Статичные графики README обновлены в $TARGET_DIR"
