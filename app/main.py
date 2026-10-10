"""FastAPI-приложение рекомендательной системы."""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

from app.recommender import PopularityRecommender
from app.schemas import (
    HealthResponse,
    ProductRecommendation,
    RecommendationRequest,
    RecommendationResponse,
    ReloadResponse,
)

LOGGER = logging.getLogger("recommendation_service")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

REQUESTS = Counter(
    "recommendation_http_requests_total",
    "HTTP requests processed by the recommendation service",
    ("method", "path", "status"),
)
LATENCY = Histogram(
    "recommendation_http_request_duration_seconds",
    "HTTP request latency",
    ("method", "path"),
)
RECOMMENDATIONS = Counter(
    "recommendation_products_returned_total",
    "Products returned to clients",
    ("product", "segment"),
)
MODEL_READY = Gauge(
    "recommendation_model_ready",
    "Whether the recommendation model is loaded",
)


def model_path() -> Path:
    """Получить путь модели из окружения или использовать DVC output."""
    return Path(os.getenv("MODEL_PATH", "dvc_data/data/baseline/model.json")).resolve()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загрузить модель один раз до приёма трафика."""
    path = model_path()
    LOGGER.info("Loading model from %s", path)
    app.state.recommender = PopularityRecommender(path)
    MODEL_READY.set(1)
    try:
        yield
    finally:
        MODEL_READY.set(0)


app = FastAPI(
    title="Bank Product Recommendation API",
    version="1.0.0",
    description="Top-7 recommendations with already-owned products excluded.",
    lifespan=lifespan,
)
app.mount("/metrics", make_asgi_app())


@app.middleware("http")
async def observe_request(request: Request, call_next):
    """Записать latency, код ответа и структурированное сообщение в лог."""
    started = time.perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")
    path = getattr(route, "path", request.url.path)
    duration = time.perf_counter() - started
    REQUESTS.labels(request.method, path, str(response.status_code)).inc()
    LATENCY.labels(request.method, path).observe(duration)
    LOGGER.info(
        "method=%s path=%s status=%s duration_ms=%.2f",
        request.method,
        path,
        response.status_code,
        duration * 1000,
    )
    return response


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Проверить процесс и факт загрузки модели."""
    recommender = request.app.state.recommender
    return HealthResponse(
        status="ok",
        model_loaded=True,
        products=len(recommender.products),
    )


@app.post("/reload", response_model=ReloadResponse)
def reload_model(request: Request) -> ReloadResponse:
    """Атомарно заменить модель после успешного переобучения."""
    expected_token = os.getenv("MODEL_RELOAD_TOKEN")
    supplied_token = request.headers.get("X-Reload-Token")
    if not expected_token or supplied_token != expected_token:
        raise HTTPException(status_code=403, detail="Invalid reload token")
    new_model = PopularityRecommender(model_path())
    request.app.state.recommender = new_model
    LOGGER.info("Model reloaded from %s", new_model.model_path)
    return ReloadResponse(
        status="reloaded",
        model_type=new_model.model_type,
        products=len(new_model.products),
    )


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(
    payload: RecommendationRequest, request: Request
) -> RecommendationResponse:
    """Построить персонализированный по сегменту top-k."""
    recommender: PopularityRecommender = request.app.state.recommender
    segment = payload.segment if payload.segment in recommender.segments else "UNKNOWN"
    try:
        result = recommender.recommend(
            segment=segment,
            owned_products=set(payload.owned_products),
            k=payload.k,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    for item in result:
        RECOMMENDATIONS.labels(item.product, segment).inc()
    return RecommendationResponse(
        customer_id=payload.customer_id,
        segment_used=segment,
        model_type=recommender.model_type,
        recommendations=[
            ProductRecommendation(product=item.product, score=item.score)
            for item in result
        ],
    )
