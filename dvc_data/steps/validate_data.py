"""Быстрая проверка структуры исходного датасета Santander перед EDA.

Шаг проверяет заголовок целиком, а значения — на настраиваемой выборке первых
строк. Полный подсчёт пропусков, дубликатов и распределений выполняет этап EDA.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
from logging_utils import get_step_logger, run_logged_step

LOGGER = get_step_logger("validate")

PRODUCT_COLUMNS = [
    "ind_ahor_fin_ult1",
    "ind_aval_fin_ult1",
    "ind_cco_fin_ult1",
    "ind_cder_fin_ult1",
    "ind_cno_fin_ult1",
    "ind_ctju_fin_ult1",
    "ind_ctma_fin_ult1",
    "ind_ctop_fin_ult1",
    "ind_ctpp_fin_ult1",
    "ind_deco_fin_ult1",
    "ind_deme_fin_ult1",
    "ind_dela_fin_ult1",
    "ind_ecue_fin_ult1",
    "ind_fond_fin_ult1",
    "ind_hip_fin_ult1",
    "ind_plan_fin_ult1",
    "ind_pres_fin_ult1",
    "ind_reca_fin_ult1",
    "ind_tjcr_fin_ult1",
    "ind_valo_fin_ult1",
    "ind_viv_fin_ult1",
    "ind_nomina_ult1",
    "ind_nom_pens_ult1",
    "ind_recibo_ult1",
]

REQUIRED_COLUMNS = ["fecha_dato", "ncodpers", *PRODUCT_COLUMNS]


def parse_args() -> argparse.Namespace:
    """Прочитать пути и размер проверочной выборки из командной строки."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dvc_data/data/origin/train_ver2.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dvc_data/data/validation/validation.json"),
    )
    parser.add_argument("--sample-rows", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    """Сформировать JSON-отчёт и остановить DVC при критичной ошибке данных."""
    args = parse_args()
    LOGGER.info("Input: %s; sample rows: %s", args.input, args.sample_rows)
    if not args.input.is_file():
        raise FileNotFoundError(f"Source dataset not found: {args.input}")

    # Заголовок читается стандартным csv-модулем без загрузки большого файла.
    # Это позволяет проверить обязательные и повторяющиеся колонки заранее.
    with args.input.open(newline="", encoding="utf-8") as source:
        header = next(csv.reader(source))

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(header))
    duplicate_columns = sorted({name for name in header if header.count(name) > 1})

    # Для быстрой проверки значений достаточно продуктовых колонок, даты и ID.
    # skipinitialspace нужен из-за пробелов после запятых в исходном CSV.
    sample = pd.read_csv(
        args.input,
        nrows=args.sample_rows,
        usecols=REQUIRED_COLUMNS,
        skipinitialspace=True,
        low_memory=False,
    )
    parsed_dates = pd.to_datetime(sample["fecha_dato"], errors="coerce")
    customer_ids = pd.to_numeric(sample["ncodpers"], errors="coerce")
    product_values = sample[PRODUCT_COLUMNS].apply(pd.to_numeric, errors="coerce")
    # Пропуски считаются отдельно и здесь допустимы. Ошибкой являются только
    # непустые значения вне бинарного множества {0, 1}.
    invalid_product_values = int(
        (~product_values.isin([0, 1]) & product_values.notna()).sum().sum()
    )

    checks = {
        "file_exists": True,
        "file_size_bytes": args.input.stat().st_size,
        "column_count": len(header),
        "missing_required_columns": missing_columns,
        "duplicate_columns": duplicate_columns,
        "sample_rows_checked": len(sample),
        "invalid_dates_in_sample": int(parsed_dates.isna().sum()),
        "missing_customer_ids_in_sample": int(customer_ids.isna().sum()),
        "invalid_product_values_in_sample": invalid_product_values,
    }
    # Единый статус удобен как контракт между этапами validate и eda.
    checks["status"] = (
        "passed"
        if not missing_columns
        and not duplicate_columns
        and checks["invalid_dates_in_sample"] == 0
        and checks["missing_customer_ids_in_sample"] == 0
        and invalid_product_values == 0
        else "failed"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(checks, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    LOGGER.info(
        "Validation status: %s; checked rows: %s; report: %s",
        checks["status"],
        len(sample),
        args.output,
    )
    # Отчёт записывается и при ошибке, чтобы причину можно было изучить после
    # остановки пайплайна.
    if checks["status"] != "passed":
        raise ValueError(f"Dataset validation failed; see {args.output}")


if __name__ == "__main__":
    run_logged_step("validate", main)
