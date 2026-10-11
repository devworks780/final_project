"""Подобрать popularity baseline и зарегистрировать эксперименты в MLflow."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from joblib import dump
from logging_utils import get_step_logger, run_logged_step
from recommendation_metrics import ranking_metrics, top_k_indices
from validate_data import PRODUCT_COLUMNS

LOGGER = get_step_logger("train_baseline")


def parse_args() -> argparse.Namespace:
    """Прочитать каталоги признаков, модели и файл параметров."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features-dir", type=Path, default=Path("dvc_data/data/features")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("dvc_data/data/baseline")
    )
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    return parser.parse_args()


def resolve_env(value: str) -> str:
    """Подставить переменные окружения в строку конфигурации."""
    return os.path.expandvars(value)


def build_popularity_scores(
    stats: pd.DataFrame,
    cutoff_month: str,
    window: int,
    decay: float,
    smoothing: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str]]:
    """Оценить глобальные и сегментные вероятности подключения продукта."""
    history = stats.loc[stats["target_month"] < cutoff_month].copy()
    months = sorted(history["target_month"].unique())[-window:]
    history = history.loc[history["target_month"].isin(months)]
    month_age = {month: len(months) - index - 1 for index, month in enumerate(months)}
    history["weight"] = history["target_month"].map(
        lambda month: decay ** month_age[month]
    )
    history["weighted_additions"] = history["additions"] * history["weight"]
    history["weighted_eligible"] = history["eligible"] * history["weight"]

    global_aggregates = history.groupby("product", sort=False)[
        ["weighted_additions", "weighted_eligible"]
    ].sum()
    global_rates = np.array(
        [
            (global_aggregates.loc[product, "weighted_additions"] + smoothing)
            / (global_aggregates.loc[product, "weighted_eligible"] + 2 * smoothing)
            for product in PRODUCT_COLUMNS
        ],
        dtype="float64",
    )

    segment_rates: dict[str, np.ndarray] = {}
    for segment, segment_frame in history.groupby("segmento"):
        aggregates = segment_frame.groupby("product", sort=False)[
            ["weighted_additions", "weighted_eligible"]
        ].sum()
        segment_rates[str(segment)] = np.array(
            [
                (aggregates.loc[product, "weighted_additions"] + smoothing)
                / (aggregates.loc[product, "weighted_eligible"] + 2 * smoothing)
                for product in PRODUCT_COLUMNS
            ],
            dtype="float64",
        )
    return global_rates, segment_rates, months


def recommend(
    frame: pd.DataFrame,
    global_rates: np.ndarray,
    segment_rates: dict[str, np.ndarray],
    segment_weight: float,
    k: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Построить top-k, обязательно исключив уже подключённые продукты."""
    owned_columns = [f"owned__{product}" for product in PRODUCT_COLUMNS]
    target_columns = [f"target__{product}" for product in PRODUCT_COLUMNS]
    owned = frame[owned_columns].to_numpy(dtype="uint8")
    targets = frame[target_columns].to_numpy(dtype="uint8")
    segments = frame["segmento"].astype(str).to_numpy()
    recommendations = np.empty((len(frame), k), dtype="uint8")

    for start in range(0, len(frame), batch_size):
        stop = min(start + batch_size, len(frame))
        batch_segments = segments[start:stop]
        scores = np.tile(global_rates, (stop - start, 1))
        if segment_weight > 0:
            for segment in np.unique(batch_segments):
                mask = batch_segments == segment
                local_rates = segment_rates.get(segment, global_rates)
                scores[mask] = (1 - segment_weight) * global_rates + (
                    segment_weight * local_rates
                )
        scores[owned[start:stop].astype(bool)] = -np.inf
        recommendations[start:stop] = top_k_indices(scores, k)
    return targets, recommendations


def evaluate_configuration(
    frame: pd.DataFrame,
    global_rates: np.ndarray,
    segment_rates: dict[str, np.ndarray],
    segment_weight: float,
    k: int,
    batch_size: int,
) -> dict[str, float]:
    """Оценить один вариант baseline на готовой временной выборке."""
    targets, recommendations = recommend(
        frame,
        global_rates,
        segment_rates,
        segment_weight,
        k,
        batch_size,
    )
    metrics = ranking_metrics(targets, recommendations)
    metrics["product_coverage_at_k"] = float(
        len(np.unique(recommendations)) / len(PRODUCT_COLUMNS)
    )
    return metrics


def configure_mlflow(config: dict[str, object]) -> bool:
    """Настроить локальный tracking store и S3-каталог артефактов."""
    # Переменная используется для безопасного локального воспроизведения без
    # внешних side effects; штатное значение из params.yaml остаётся включённым.
    if os.getenv("DISABLE_MLFLOW_LOGGING") == "1" or not bool(
        config.get("enabled", True)
    ):
        LOGGER.warning("MLflow logging is disabled for this run")
        return False
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI", resolve_env(str(config["tracking_uri"]))
    )
    artifact_location = resolve_env(str(config["artifact_location"]))
    if "$" in artifact_location:
        raise ValueError(
            "MLflow artifact_location contains unresolved environment variables"
        )
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = str(config["experiment_name"])
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
    mlflow.set_experiment(experiment_name)
    return True


def main() -> None:
    """Выполнить grid search, тест лучшего baseline и сохранить модель."""
    args = parse_args()
    params = yaml.safe_load(args.params.read_text(encoding="utf-8"))
    model_params = params["model"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stats = pd.read_parquet(args.features_dir / "transition_stats.parquet")
    validation = pd.read_parquet(args.features_dir / "validation.parquet")
    test = pd.read_parquet(args.features_dir / "test.parquet")
    validation_month = str(model_params["temporal_validation_month"])
    test_month = str(model_params["temporal_test_month"])
    k = int(model_params["recommendation_k"])
    smoothing = float(model_params["rate_smoothing"])
    batch_size = int(model_params["evaluation_batch_size"])
    mlflow_enabled = configure_mlflow(params["mlflow"])

    configurations = list(
        itertools.product(
            model_params["popularity_windows"],
            model_params["recency_decays"],
            model_params["segment_weights"],
        )
    )
    LOGGER.info("Grid search: %s configurations", len(configurations))
    search_rows: list[dict[str, float | int]] = []

    parent_context = (
        mlflow.start_run(run_name="baseline_grid_search") if mlflow_enabled else None
    )
    try:
        for window, decay, segment_weight in configurations:
            global_rates, segment_rates, used_months = build_popularity_scores(
                stats,
                validation_month,
                int(window),
                float(decay),
                smoothing,
            )
            metrics = evaluate_configuration(
                validation,
                global_rates,
                segment_rates,
                float(segment_weight),
                k,
                batch_size,
            )
            row: dict[str, float | int] = {
                "window": int(window),
                "decay": float(decay),
                "segment_weight": float(segment_weight),
                **metrics,
            }
            search_rows.append(row)
            if mlflow_enabled:
                with mlflow.start_run(
                    run_name=(f"baseline_w{window}_d{decay}_s{segment_weight}"),
                    nested=True,
                ):
                    mlflow.log_params(
                        {
                            "window": window,
                            "recency_decay": decay,
                            "segment_weight": segment_weight,
                            "recommendation_k": k,
                            "validation_month": validation_month,
                            "history_months": ",".join(used_months),
                        }
                    )
                    mlflow.log_metrics(metrics)
            LOGGER.info(
                "window=%s decay=%s segment_weight=%s MAP@%s=%.6f",
                window,
                decay,
                segment_weight,
                k,
                metrics["map_at_k_all"],
            )

        search = pd.DataFrame(search_rows).sort_values(
            ["map_at_k_all", "ndcg_at_k_positive"], ascending=False
        )
        best = search.iloc[0]
        best_params = {
            "window": int(best["window"]),
            "recency_decay": float(best["decay"]),
            "segment_weight": float(best["segment_weight"]),
            "recommendation_k": k,
            "rate_smoothing": smoothing,
        }

        test_global, test_segments, test_history = build_popularity_scores(
            stats,
            test_month,
            best_params["window"],
            best_params["recency_decay"],
            smoothing,
        )
        test_metrics = evaluate_configuration(
            test,
            test_global,
            test_segments,
            best_params["segment_weight"],
            k,
            batch_size,
        )

        # Для веб-сервиса модель дообучается на всех доступных переходах после
        # честной test-оценки. Test-метрики относятся к версии, обученной до мая.
        production_global, production_segments, production_history = (
            build_popularity_scores(
                stats,
                "9999-12-31",
                best_params["window"],
                best_params["recency_decay"],
                smoothing,
            )
        )
        model = {
            "model_type": "segment_recency_popularity",
            "product_columns": PRODUCT_COLUMNS,
            "best_params": best_params,
            "trained_transition_months": production_history,
            "global_scores": production_global.tolist(),
            "segment_scores": {
                segment: scores.tolist()
                for segment, scores in production_segments.items()
            },
        }
        search.to_csv(args.output_dir / "search_results.csv", index=False)
        (args.output_dir / "model.json").write_text(
            json.dumps(model, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        # Бинарная копия является самостоятельным модельным артефактом для
        # проверки проекта и загрузки из Python без повторного JSON-парсинга.
        dump(model, args.output_dir / "model.pkl")
        result = {
            "best_params": best_params,
            "validation_metrics": {
                key: float(best[key])
                for key in search.columns
                if key not in {"window", "decay", "segment_weight"}
            },
            "test_metrics": test_metrics,
            "validation_month": validation_month,
            "test_month": test_month,
            "test_training_months": test_history,
        }
        (args.output_dir / "metrics.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        if mlflow_enabled:
            mlflow.log_params(
                {f"best_{key}": value for key, value in best_params.items()}
            )
            mlflow.log_metrics(
                {f"test_{key}": value for key, value in test_metrics.items()}
            )
            if os.getenv("DISABLE_MLFLOW_ARTIFACT_UPLOAD") != "1":
                mlflow.log_artifacts(str(args.output_dir), artifact_path="baseline")
                mlflow.log_artifact(
                    str(args.features_dir / "summary.json"),
                    artifact_path="feature_metadata",
                )
            else:
                LOGGER.warning(
                    "MLflow artifact upload is disabled; parameters and metrics "
                    "were written to the tracking database only"
                )
        LOGGER.info("Best params: %s; test metrics: %s", best_params, test_metrics)
    finally:
        if parent_context is not None:
            parent_context.__exit__(None, None, None)


if __name__ == "__main__":
    run_logged_step("train_baseline", main)
