"""Pydantic-схемы HTTP API."""

from pydantic import BaseModel, Field, field_validator


class RecommendationRequest(BaseModel):
    """Данные клиента, необходимые baseline-модели."""

    customer_id: int | None = Field(default=None, gt=0)
    segment: str = Field(default="UNKNOWN", max_length=20)
    owned_products: list[str] = Field(default_factory=list, max_length=24)
    k: int = Field(default=7, ge=1, le=24)

    @field_validator("segment")
    @classmethod
    def normalize_segment(cls, value: str) -> str:
        """Привести сегмент к формату обучающей выборки."""
        normalized = value.strip().upper()
        return normalized or "UNKNOWN"

    @field_validator("owned_products")
    @classmethod
    def products_must_be_unique(cls, value: list[str]) -> list[str]:
        """Запретить двусмысленные повторяющиеся продукты."""
        if len(value) != len(set(value)):
            raise ValueError("owned_products must not contain duplicates")
        return value


class ProductRecommendation(BaseModel):
    """Рекомендованный продукт с model score."""

    product: str
    score: float


class RecommendationResponse(BaseModel):
    """Ответ рекомендательного сервиса."""

    customer_id: int | None
    segment_used: str
    model_type: str
    recommendations: list[ProductRecommendation]


class HealthResponse(BaseModel):
    """Состояние приложения и загруженной модели."""

    status: str
    model_loaded: bool
    products: int


class ReloadResponse(BaseModel):
    """Результат безопасной горячей перезагрузки модели."""

    status: str
    model_type: str
    products: int
