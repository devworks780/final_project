"""Выполнить потоковый EDA большого датасета банковских продуктов.

Файл объёмом более 2 ГБ читается блоками. Между блоками хранятся только
агрегаты и данные двух соседних месяцев, необходимые для расчёта новых
подключений продуктов.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from logging_utils import get_step_logger, run_logged_step
from validate_data import PRODUCT_COLUMNS

LOGGER = get_step_logger("eda")

PRODUCT_NAMES = {
    "ind_ahor_fin_ult1": "Savings account",
    "ind_aval_fin_ult1": "Guarantees",
    "ind_cco_fin_ult1": "Current account",
    "ind_cder_fin_ult1": "Derivatives account",
    "ind_cno_fin_ult1": "Payroll account",
    "ind_ctju_fin_ult1": "Junior account",
    "ind_ctma_fin_ult1": "Más particular account",
    "ind_ctop_fin_ult1": "Particular account",
    "ind_ctpp_fin_ult1": "Particular plus account",
    "ind_deco_fin_ult1": "Short-term deposit",
    "ind_deme_fin_ult1": "Medium-term deposit",
    "ind_dela_fin_ult1": "Long-term deposit",
    "ind_ecue_fin_ult1": "e-account",
    "ind_fond_fin_ult1": "Funds",
    "ind_hip_fin_ult1": "Mortgage",
    "ind_plan_fin_ult1": "Pension plan",
    "ind_pres_fin_ult1": "Loan",
    "ind_reca_fin_ult1": "Taxes",
    "ind_tjcr_fin_ult1": "Credit card",
    "ind_valo_fin_ult1": "Securities",
    "ind_viv_fin_ult1": "Home account",
    "ind_nomina_ult1": "Payroll",
    "ind_nom_pens_ult1": "Pensions",
    "ind_recibo_ult1": "Direct debit",
}

CATEGORY_COLUMNS = [
    "sexo",
    "segmento",
    "ind_actividad_cliente",
    "pais_residencia",
]
NUMERIC_COLUMNS = ["age", "renta"]


def parse_args() -> argparse.Namespace:
    """Прочитать стандартные DVC-пути и при необходимости их переопределить."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dvc_data/data/origin/train_ver2.csv"),
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("dvc_data/data/validation/validation.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("dvc_data/data/eda"))
    parser.add_argument("--params", default="params.yaml", type=Path)
    return parser.parse_args()


def save_bar(
    frame: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    ylabel: str,
    output: Path,
    horizontal: bool = False,
) -> None:
    """Сохранить вертикальную или горизонтальную столбчатую диаграмму."""
    plt.figure(figsize=(11, 6))
    if horizontal:
        plt.barh(frame[x], frame[y], color="#4C78A8")
        plt.xlabel(ylabel)
        plt.ylabel("")
        plt.gca().invert_yaxis()
    else:
        plt.bar(frame[x], frame[y], color="#4C78A8")
        plt.ylabel(ylabel)
        plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.grid(axis="x" if horizontal else "y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def iqr_flags(series: pd.Series) -> tuple[pd.Series, float, float]:
    """Найти выбросы по правилу Тьюки и вернуть флаги вместе с границами."""
    clean = series.dropna()
    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (series.lt(lower) | series.gt(upper)).fillna(False), lower, upper


def save_spike_plot(anomalies: pd.DataFrame, output: Path) -> None:
    """Показать месячную динамику и выделить найденные IQR-выбросы."""
    figure, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    x = np.arange(len(anomalies))

    for axis, value, flag, title, ylabel in [
        (
            axes[0],
            "records",
            "records_spike",
            "Monthly observation volume and detected spikes",
            "Records",
        ),
        (
            axes[1],
            "total_additions",
            "additions_spike",
            "Monthly product additions and detected spikes",
            "Additions",
        ),
    ]:
        axis.plot(x, anomalies[value], marker="o", color="#4C78A8")
        flagged = anomalies.loc[anomalies[flag]]
        axis.scatter(
            flagged.index,
            flagged[value],
            color="#E45756",
            s=70,
            label="IQR outlier",
            zorder=3,
        )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        if not flagged.empty:
            axis.legend()

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(anomalies["fecha_dato"], rotation=45, ha="right")
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main() -> None:
    """Посчитать EDA-агрегаты и сохранить таблицы, метрики и графики."""
    args = parse_args()
    # EDA запускается только после успешной проверки входного контракта.
    validation = json.loads(args.validation.read_text(encoding="utf-8"))
    if validation.get("status") != "passed":
        raise ValueError("EDA requires a successful validation report")

    params = yaml.safe_load(args.params.read_text(encoding="utf-8")) or {}
    eda_params = params.get("eda", {})
    chunk_size = int(eda_params.get("chunk_size", 100_000))
    top_categories = int(eda_params.get("top_categories", 10))
    sample_per_chunk = int(eda_params.get("sample_per_chunk", 2_000))
    random_state = int(eda_params.get("random_state", 42))
    LOGGER.info(
        "Input: %s; chunk size: %s; sample per chunk: %s; random state: %s",
        args.input,
        chunk_size,
        sample_per_chunk,
        random_state,
    )

    output_dir = args.output_dir
    plots_dir = output_dir / "plots"
    tables_dir = output_dir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Эти структуры содержат только агрегаты и поэтому почти не растут вместе
    # с количеством строк исходного файла.
    header = pd.read_csv(args.input, nrows=0).columns.tolist()
    missing_counts = pd.Series(0, index=header, dtype="int64")
    product_totals = pd.Series(0, index=PRODUCT_COLUMNS, dtype="int64")
    monthly_rows: Counter[str] = Counter()
    monthly_customers: Counter[str] = Counter()
    category_counts = {column: Counter() for column in CATEGORY_COLUMNS}
    numeric_samples: dict[str, list[np.ndarray]] = {
        column: [] for column in NUMERIC_COLUMNS
    }
    customer_ids: set[int] = set()
    total_rows = 0

    # Для переходов t-1 -> t нужны продуктовые состояния текущего и предыдущего
    # месяца. Исходный датасет уже упорядочен по fecha_dato.
    current_month: str | None = None
    current_frames: list[pd.DataFrame] = []
    previous_products: pd.DataFrame | None = None
    additions_by_month: list[dict[str, int | str]] = []
    duplicates_by_month: list[dict[str, int | str]] = []

    def finish_month(month: str, frames: list[pd.DataFrame]) -> pd.DataFrame:
        """Завершить месячный снимок и сравнить его с предыдущим месяцем."""
        nonlocal previous_products
        month_data = pd.concat(frames, ignore_index=True)
        month_record_count = len(month_data)

        # ID клиента должен быть уникальным внутри месяца. Полные повторы ищем
        # по стабильному pandas-хешу всей исходной строки, рассчитанному в chunk.
        duplicate_key_rows = int(month_data.duplicated("ncodpers").sum())
        full_duplicate_rows = int(month_data.duplicated("_row_hash").sum())
        duplicates_by_month.append(
            {
                "fecha_dato": month,
                "records": len(month_data),
                "duplicate_customer_month_rows": duplicate_key_rows,
                "full_duplicate_rows": full_duplicate_rows,
            }
        )
        # Даже при нарушении уникальности расчёт остаётся детерминированным:
        # для клиента используется последняя запись месяца, а число повторов
        # сохраняется в отдельном диагностическом артефакте.
        month_data = month_data.drop_duplicates("ncodpers", keep="last")
        month_data = month_data.set_index("ncodpers").sort_index()
        products = month_data[PRODUCT_COLUMNS].astype("uint8")
        monthly_customers[month] = len(products)

        additions: dict[str, int | str] = {"fecha_dato": month}
        if previous_products is None:
            # Для первого снимка предыдущего состояния нет, поэтому считать
            # все имеющиеся продукты новыми было бы ошибкой.
            additions.update({column: 0 for column in PRODUCT_COLUMNS})
        else:
            # Сравниваются только клиенты, присутствующие в обоих месяцах.
            # Новый продукт: раньше 0, теперь 1. Снятие продукта не является
            # положительной меткой рекомендательной задачи.
            common = products.index.intersection(previous_products.index)
            current_common = products.loc[common]
            previous_common = previous_products.loc[common]
            new_products = current_common.gt(previous_common) & current_common.eq(1)
            additions.update(
                {column: int(new_products[column].sum()) for column in PRODUCT_COLUMNS}
            )
        additions["total_additions"] = int(
            sum(int(additions[column]) for column in PRODUCT_COLUMNS)
        )
        additions_by_month.append(additions)
        previous_products = products
        LOGGER.info(
            "Month %s: %s records, %s customers, %s additions, %s duplicate keys",
            month,
            month_record_count,
            len(products),
            additions["total_additions"],
            duplicate_key_rows,
        )
        return products

    # Потоковое чтение удерживает потребление памяти на приемлемом уровне.
    reader = pd.read_csv(
        args.input,
        chunksize=chunk_size,
        skipinitialspace=True,
        low_memory=False,
    )
    for chunk_number, chunk in enumerate(reader):
        total_rows += len(chunk)
        missing_counts = missing_counts.add(chunk.isna().sum(), fill_value=0)

        ids = pd.to_numeric(chunk["ncodpers"], errors="coerce").dropna().astype("int64")
        customer_ids.update(ids.tolist())

        # Продукты приводятся к компактному uint8: это заметно уменьшает память
        # при хранении месячного снимка примерно на миллион клиентов.
        products = (
            chunk[PRODUCT_COLUMNS]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .clip(0, 1)
            .astype("uint8")
        )
        # Хеш всей строки позволяет искать полные повторы, не сохраняя в памяти
        # все 48 исходных колонок для целого месяца.
        row_hashes = pd.util.hash_pandas_object(chunk, index=False).to_numpy()
        product_totals = product_totals.add(products.sum(), fill_value=0)

        for column in CATEGORY_COLUMNS:
            if column == "ind_actividad_cliente":
                numeric_values = pd.to_numeric(chunk[column], errors="coerce")
                values = numeric_values.map(
                    lambda value: "<missing>" if pd.isna(value) else str(int(value))
                )
            else:
                values = chunk[column].fillna("<missing>").astype(str).str.strip()
            category_counts[column].update(values.value_counts().to_dict())

        # Квантили возраста и дохода оцениваются по фиксированной выборке из
        # каждого блока. Счётчики строк, пропусков и продуктов остаются точными.
        rng = np.random.default_rng(random_state + chunk_number)
        for column in NUMERIC_COLUMNS:
            values = pd.to_numeric(chunk[column], errors="coerce").dropna().to_numpy()
            if len(values):
                take = min(sample_per_chunk, len(values))
                numeric_samples[column].append(
                    values[rng.choice(len(values), size=take, replace=False)]
                )

        # Граница месяца может проходить внутри блока, поэтому месячные части
        # накапливаются до появления следующей даты.
        month_values = chunk["fecha_dato"].astype(str)
        for month, positions in month_values.groupby(
            month_values, sort=False
        ).groups.items():
            if current_month is not None and month != current_month:
                finish_month(current_month, current_frames)
                current_frames = []
            if current_month is not None and month < current_month:
                raise ValueError(
                    "Rows must be ordered by fecha_dato for transition EDA"
                )
            current_month = month
            month_slice = pd.DataFrame(
                {
                    "ncodpers": pd.to_numeric(
                        chunk.loc[positions, "ncodpers"], errors="raise"
                    ).astype("int64"),
                    "_row_hash": row_hashes[chunk.index.get_indexer(positions)],
                    **{
                        column: products.loc[positions, column].to_numpy()
                        for column in PRODUCT_COLUMNS
                    },
                }
            )
            current_frames.append(month_slice)
            monthly_rows[month] += len(month_slice)

    if current_month is not None:
        finish_month(current_month, current_frames)

    months = sorted(monthly_rows)
    missingness = pd.DataFrame(
        {
            "column": header,
            "missing_count": missing_counts.reindex(header).astype("int64").values,
        }
    )
    missingness["missing_rate"] = missingness["missing_count"] / total_rows
    missingness = missingness.sort_values("missing_rate", ascending=False)
    missingness.to_csv(tables_dir / "missingness.csv", index=False)

    product_prevalence = pd.DataFrame(
        {
            "product": PRODUCT_COLUMNS,
            "product_name": [PRODUCT_NAMES[column] for column in PRODUCT_COLUMNS],
            "owned_records": product_totals.astype("int64").values,
        }
    )
    product_prevalence["ownership_rate"] = (
        product_prevalence["owned_records"] / total_rows
    )
    product_prevalence = product_prevalence.sort_values(
        "ownership_rate", ascending=False
    )
    product_prevalence.to_csv(tables_dir / "product_prevalence.csv", index=False)

    monthly = pd.DataFrame(
        {
            "fecha_dato": months,
            "records": [monthly_rows[month] for month in months],
            "unique_customers": [monthly_customers[month] for month in months],
        }
    )
    monthly.to_csv(tables_dir / "monthly_coverage.csv", index=False)

    additions = pd.DataFrame(additions_by_month)
    additions.to_csv(tables_dir / "monthly_product_additions.csv", index=False)
    duplicates = pd.DataFrame(duplicates_by_month)
    duplicates.to_csv(tables_dir / "duplicate_analysis.csv", index=False)
    product_additions = pd.DataFrame(
        {
            "product": PRODUCT_COLUMNS,
            "product_name": [PRODUCT_NAMES[column] for column in PRODUCT_COLUMNS],
            "additions": [int(additions[column].sum()) for column in PRODUCT_COLUMNS],
        }
    ).sort_values("additions", ascending=False)
    product_additions.to_csv(tables_dir / "product_additions.csv", index=False)

    # Выбросы ищутся в относительном изменении к предыдущему месяцу, а не в
    # абсолютном уровне. Это выделяет резкие скачки на растущем временном ряду.
    anomalies = monthly.merge(
        additions[["fecha_dato", "total_additions"]], on="fecha_dato", how="left"
    )
    anomalies["records_change_rate"] = anomalies["records"].pct_change()
    anomalies["additions_change_rate"] = anomalies["total_additions"].pct_change()
    records_flags, records_lower, records_upper = iqr_flags(
        anomalies["records_change_rate"]
    )
    # Первый месяц имеет 0 подключений, поэтому изменение следующего месяца
    # бесконечно. Оба стартовых значения исключаются из оценки IQR-границ.
    additions_flags, additions_lower, additions_upper = iqr_flags(
        anomalies.loc[anomalies.index > 1, "additions_change_rate"]
    )
    anomalies["records_spike"] = records_flags
    anomalies["additions_spike"] = additions_flags.reindex(
        anomalies.index, fill_value=False
    )
    anomalies.to_csv(tables_dir / "monthly_anomalies.csv", index=False)

    category_summary: dict[str, list[dict[str, int | str]]] = {}
    for column, counter in category_counts.items():
        table = pd.DataFrame(counter.most_common(), columns=[column, "count"])
        table["rate"] = table["count"] / total_rows
        table.to_csv(tables_dir / f"{column}_distribution.csv", index=False)
        category_summary[column] = table.head(top_categories).to_dict("records")

    # Объединяем небольшие воспроизводимые подвыборки только после завершения
    # чтения файла; это не требует загрузки исходных числовых колонок целиком.
    numeric_summary: dict[str, dict[str, float | int]] = {}
    for column, samples in numeric_samples.items():
        values = np.concatenate(samples) if samples else np.array([], dtype=float)
        values = values[np.isfinite(values)]
        numeric_summary[column] = {
            "sample_size": len(values),
            "median": float(np.median(values)) if len(values) else float("nan"),
            "q01": float(np.quantile(values, 0.01)) if len(values) else float("nan"),
            "q99": float(np.quantile(values, 0.99)) if len(values) else float("nan"),
        }

    save_bar(
        monthly,
        "fecha_dato",
        "records",
        "Monthly observation volume",
        "Records",
        plots_dir / "monthly_observations.png",
    )
    save_bar(
        product_prevalence.head(15),
        "product_name",
        "ownership_rate",
        "Most frequently held products",
        "Share of monthly customer records",
        plots_dir / "product_prevalence.png",
        horizontal=True,
    )
    save_spike_plot(anomalies, plots_dir / "monthly_spikes.png")
    save_bar(
        missingness.head(15),
        "column",
        "missing_rate",
        "Columns with the most missing values",
        "Missing-value rate",
        plots_dir / "missingness.png",
        horizontal=True,
    )
    save_bar(
        product_additions.head(15),
        "product_name",
        "additions",
        "Most frequently added products",
        "New product additions",
        plots_dir / "product_additions.png",
        horizontal=True,
    )

    top_owned = product_prevalence.iloc[0]
    top_added = product_additions.iloc[0]
    max_missing = missingness.iloc[0]
    total_additions = int(product_additions["additions"].sum())
    duplicate_customer_month_rows = int(
        duplicates["duplicate_customer_month_rows"].sum()
    )
    full_duplicate_rows = int(duplicates["full_duplicate_rows"].sum())
    records_spike_months = anomalies.loc[
        anomalies["records_spike"], "fecha_dato"
    ].tolist()
    additions_spike_months = anomalies.loc[
        anomalies["additions_spike"], "fecha_dato"
    ].tolist()
    # Одно наблюдение может иметь несколько новых продуктов. Поэтому N - число
    # положительных меток даёт консервативную нижнюю границу доли месяцев без
    # подключений, а не точную долю таких месяцев.
    zero_addition_share = float(
        max(0, (sum(monthly["unique_customers"].iloc[1:]) - total_additions))
        / max(1, sum(monthly["unique_customers"].iloc[1:]))
    )
    conclusions = [
        (
            f"The dataset contains {total_rows:,} monthly records for "
            f"{len(customer_ids):,} customers across {len(months)} snapshots."
        ),
        (
            f"The most prevalent product is {top_owned['product_name']} "
            f"({top_owned['ownership_rate']:.1%} of records)."
        ),
        (
            f"The most frequently added product is {top_added['product_name']} "
            f"({int(top_added['additions']):,} observed additions)."
        ),
        (
            f"The highest missing-value rate is in {max_missing['column']} "
            f"({max_missing['missing_rate']:.1%})."
        ),
        (
            f"Positive additions are sparse: at least {zero_addition_share:.1%} of "
            "customer-month observations have no product addition when each addition "
            "is counted as a separate positive label."
        ),
        (
            f"Duplicate analysis found {duplicate_customer_month_rows:,} repeated "
            f"customer-month keys and {full_duplicate_rows:,} full-row duplicates."
        ),
        (
            "IQR analysis flagged record-volume spikes in "
            f"{', '.join(records_spike_months) or 'no months'} and addition spikes in "
            f"{', '.join(additions_spike_months) or 'no months'}."
        ),
    ]

    # summary.json объявлен DVC-метрикой: основные результаты доступны через
    # `dvc metrics show` без чтения всех CSV-таблиц.
    summary = {
        "rows": total_rows,
        "unique_customers": len(customer_ids),
        "date_min": months[0],
        "date_max": months[-1],
        "monthly_snapshots": len(months),
        "product_columns": len(PRODUCT_COLUMNS),
        "total_observed_product_additions": total_additions,
        "duplicate_customer_month_rows": duplicate_customer_month_rows,
        "full_duplicate_rows": full_duplicate_rows,
        "records_spike_months": records_spike_months,
        "additions_spike_months": additions_spike_months,
        "spike_detection": {
            "method": "Tukey IQR on month-over-month change rates",
            "records_change_lower_bound": records_lower,
            "records_change_upper_bound": records_upper,
            "additions_change_lower_bound": additions_lower,
            "additions_change_upper_bound": additions_upper,
        },
        "top_owned_product": str(top_owned["product"]),
        "top_owned_product_rate": float(top_owned["ownership_rate"]),
        "top_added_product": str(top_added["product"]),
        "top_added_product_count": int(top_added["additions"]),
        "max_missing_column": str(max_missing["column"]),
        "max_missing_rate": float(max_missing["missing_rate"]),
        "numeric_sample_summary": numeric_summary,
        "top_categories": category_summary,
        "conclusions": conclusions,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    LOGGER.info(
        "EDA artifacts saved to %s; processed %s rows and %s customers",
        output_dir,
        total_rows,
        len(customer_ids),
    )


if __name__ == "__main__":
    run_logged_step("eda", main)
