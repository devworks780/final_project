"""Векторизованные метрики top-k для multilabel-рекомендаций."""

from __future__ import annotations

import numpy as np


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Вернуть индексы k лучших продуктов в порядке убывания score."""
    candidates = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    candidate_scores = np.take_along_axis(scores, candidates, axis=1)
    order = np.argsort(-candidate_scores, axis=1)
    return np.take_along_axis(candidates, order, axis=1)


def ranking_metrics(
    targets: np.ndarray, recommendations: np.ndarray
) -> dict[str, float]:
    """Посчитать MAP, recall и NDCG для готовых top-k рекомендаций."""
    hits = np.take_along_axis(targets, recommendations, axis=1).astype(float)
    positives = targets.sum(axis=1)
    positive_mask = positives > 0
    k = recommendations.shape[1]

    precision_at_rank = np.cumsum(hits, axis=1) / np.arange(1, k + 1)
    ap_denominator = np.minimum(positives, k)
    average_precision = np.divide(
        (precision_at_rank * hits).sum(axis=1),
        ap_denominator,
        out=np.zeros(len(targets), dtype=float),
        where=ap_denominator > 0,
    )
    recall = np.divide(
        hits.sum(axis=1),
        positives,
        out=np.zeros(len(targets), dtype=float),
        where=positive_mask,
    )

    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (hits * discounts).sum(axis=1)
    ideal_lengths = np.minimum(positives, k).astype(int)
    ideal_dcg = np.array([discounts[:length].sum() for length in ideal_lengths])
    ndcg = np.divide(
        dcg,
        ideal_dcg,
        out=np.zeros(len(targets), dtype=float),
        where=ideal_dcg > 0,
    )

    return {
        "map_at_k_all": float(average_precision.mean()),
        "map_at_k_positive": float(average_precision[positive_mask].mean()),
        "recall_at_k_positive": float(recall[positive_mask].mean()),
        "ndcg_at_k_positive": float(ndcg[positive_mask].mean()),
        "positive_customer_rate": float(positive_mask.mean()),
    }
