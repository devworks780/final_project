"""Подготовить компактные признаки и временные выборки для baseline-модели.

Вместо сохранения всех многомиллионных переходов этап агрегирует статистики
популярности по месяцу, сегменту и продукту. Подробные клиентские строки
сохраняются только для validation/test месяцев, нужных для честной оценки.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from logging_utils import get_step_logger, run_logged_step
from validate_data import PRODUCT_COLUMNS

LOGGER = get_step_logger("prepare_features")


def parse_args() -> argparse.Namespace:
    """Прочитать стандартные пути DVC-шагa."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dvc_data/data/origin/train_ver2.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dvc_data/data/features"),
    )
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    return parser.parse_args()


def normalize_segment(values: pd.Series) -> pd.Series:
    """Свести исходные подписи сегментов к устойчивым коротким категориям."""
    cleaned = values.fillna("UNKNOWN").astype(str).str.strip()
    return cleaned.str.split(" - ", n=1).str[0].replace("", "UNKNOWN")


def main() -> None:
    """Построить агрегаты переходов и клиентские validation/test выборки."""
    args = parse_args()
    params = yaml.safe_load(args.params.read_text(encoding="utf-8"))
    model_params = params["model"]
    validation_month = model_params["temporal_validation_month"]
    test_month = model_params["temporal_test_month"]
    chunk_size = int(params.get("eda", {}).get("chunk_size", 100_000))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    usecols = ["fecha_dato", "ncodpers", "segmento", *PRODUCT_COLUMNS]
    current_month: str | None = None
    current_parts: list[pd.DataFrame] = []
    previous: pd.DataFrame | None = None
    transition_rows: list[dict[str, int | str]] = []
    evaluation_frames: dict[str, pd.DataFrame] = {}

    def finish_month(month: str, parts: list[pd.DataFrame]) -> None:
        """Сравнить готовый месячный снимок с предыдущим снимком."""
        nonlocal previous
        current = pd.concat(parts, ignore_index=True)
        current = current.drop_duplicates("ncodpers", keep="last")
        current = current.set_index("ncodpers").sort_index()

        if previous is not None:
            common = current.index.intersection(previous.index)
            before = previous.loc[common]
            after = current.loc[common]
            additions = after[PRODUCT_COLUMNS].gt(before[PRODUCT_COLUMNS]) & after[
                PRODUCT_COLUMNS
            ].eq(1)

            # Популярность считается как доля подключений среди клиентов,
            # которым продукт действительно можно было рекомендовать.
            eligible = before[PRODUCT_COLUMNS].eq(0)
            segments = before["segmento"]
            for segment in sorted(segments.unique()):
                mask = segments.eq(segment).to_numpy()
                segment_additions = additions.iloc[mask]
                segment_eligible = eligible.iloc[mask]
                for product in PRODUCT_COLUMNS:
                    transition_rows.append(
                        {
                            "target_month": month,
                            "segmento": segment,
                            "product": product,
                            "additions": int(segment_additions[product].sum()),
                            "eligible": int(segment_eligible[product].sum()),
                        }
                    )

            if month in {validation_month, test_month}:
                evaluation = pd.DataFrame(index=common)
                evaluation.index.name = "ncodpers"
                evaluation["segmento"] = segments
                for product in PRODUCT_COLUMNS:
                    evaluation[f"owned__{product}"] = before[product].astype("uint8")
                    evaluation[f"target__{product}"] = additions[product].astype(
                        "uint8"
                    )
                evaluation_frames[month] = evaluation.reset_index()

            LOGGER.info(
                "Transition %s: %s common customers, %s additions",
                month,
                len(common),
                int(additions.to_numpy().sum()),
            )
        previous = current

    reader = pd.read_csv(
        args.input,
        usecols=usecols,
        chunksize=chunk_size,
        skipinitialspace=True,
        low_memory=False,
    )
    for chunk in reader:
        products = (
            chunk[PRODUCT_COLUMNS]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .clip(0, 1)
            .astype("uint8")
        )
        normalized = pd.DataFrame(
            {
                "fecha_dato": chunk["fecha_dato"].astype(str),
                "ncodpers": pd.to_numeric(chunk["ncodpers"], errors="raise").astype(
                    "int64"
                ),
                "segmento": normalize_segment(chunk["segmento"]),
            },
            index=chunk.index,
        )
        normalized[PRODUCT_COLUMNS] = products

        for month, positions in normalized.groupby(
            "fecha_dato", sort=False
        ).groups.items():
            if current_month is not None and month != current_month:
                finish_month(current_month, current_parts)
                current_parts = []
            if current_month is not None and month < current_month:
                raise ValueError("Rows must be ordered by fecha_dato")
            current_month = month
            current_parts.append(normalized.loc[positions].drop(columns="fecha_dato"))

    if current_month is not None:
        finish_month(current_month, current_parts)

    missing_months = {validation_month, test_month} - evaluation_frames.keys()
    if missing_months:
        raise ValueError(f"Evaluation months are missing: {sorted(missing_months)}")

    stats = pd.DataFrame(transition_rows)
    stats.to_parquet(args.output_dir / "transition_stats.parquet", index=False)
    evaluation_frames[validation_month].to_parquet(
        args.output_dir / "validation.parquet", index=False
    )
    evaluation_frames[test_month].to_parquet(
        args.output_dir / "test.parquet", index=False
    )

    summary = {
        "transition_months": int(stats["target_month"].nunique()),
        "segments": sorted(stats["segmento"].unique().tolist()),
        "products": len(PRODUCT_COLUMNS),
        "validation_month": validation_month,
        "validation_rows": len(evaluation_frames[validation_month]),
        "validation_positive_labels": int(
            evaluation_frames[validation_month][
                [f"target__{product}" for product in PRODUCT_COLUMNS]
            ]
            .to_numpy()
            .sum()
        ),
        "test_month": test_month,
        "test_rows": len(evaluation_frames[test_month]),
        "test_positive_labels": int(
            evaluation_frames[test_month][
                [f"target__{product}" for product in PRODUCT_COLUMNS]
            ]
            .to_numpy()
            .sum()
        ),
        "generated_features": [
            "24 current-product ownership flags",
            "24 next-month product-addition targets",
            "normalized customer segment",
            "monthly global product-addition rate",
            "monthly segment product-addition rate",
            "candidate eligibility count",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Feature artifacts saved to %s", args.output_dir)


if __name__ == "__main__":
    run_logged_step("prepare_features", main)
