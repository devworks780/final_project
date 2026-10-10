"""Загрузка popularity-модели и построение рекомендаций."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Recommendation:
    """Один продукт и его оценка вероятности подключения."""

    product: str
    score: float


class PopularityRecommender:
    """Потокобезопасная read-only модель сегментной популярности."""

    def __init__(self, model_path: Path) -> None:
        payload = json.loads(model_path.read_text(encoding="utf-8"))
        self.model_path = model_path
        self.model_type = str(payload["model_type"])
        self.products = tuple(payload["product_columns"])
        self.global_scores = tuple(float(value) for value in payload["global_scores"])
        self.segment_scores = {
            str(segment): tuple(float(value) for value in values)
            for segment, values in payload["segment_scores"].items()
        }
        self.default_k = int(payload["best_params"]["recommendation_k"])
        self.segment_weight = float(payload["best_params"]["segment_weight"])
        if len(self.products) != len(self.global_scores):
            raise ValueError("Model contains different numbers of products and scores")

    @property
    def segments(self) -> tuple[str, ...]:
        """Вернуть известные модели клиентские сегменты."""
        return tuple(sorted(self.segment_scores))

    def recommend(
        self,
        segment: str,
        owned_products: set[str],
        k: int | None = None,
    ) -> list[Recommendation]:
        """Вернуть top-k доступных продуктов, исключив уже подключённые."""
        unknown = owned_products.difference(self.products)
        if unknown:
            raise ValueError(f"Unknown products: {', '.join(sorted(unknown))}")
        limit = min(k or self.default_k, len(self.products) - len(owned_products))
        segment_values = self.segment_scores.get(segment, self.global_scores)
        weighted_scores = [
            (1 - self.segment_weight) * global_score
            + self.segment_weight * segment_score
            for global_score, segment_score in zip(
                self.global_scores, segment_values, strict=True
            )
        ]
        ranked = sorted(
            (
                Recommendation(product, weighted_scores[index])
                for index, product in enumerate(self.products)
                if product not in owned_products
            ),
            key=lambda item: (-item.score, item.product),
        )
        return ranked[:limit]
