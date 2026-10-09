"""Общее файловое логирование для исполняемых шагов DVC."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

LOG_PATH = Path(__file__).resolve().parents[1] / "dvc_steps.log"
T = TypeVar("T")


def get_step_logger(step_name: str) -> logging.Logger:
    """Вернуть настроенный logger, дописывающий сообщения в общий файл."""
    logger = logging.getLogger(f"dvc_steps.{step_name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Повторный вызов в одном процессе не должен добавлять второй handler и
    # дублировать каждую строку лога.
    if not logger.handlers:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)

    return logger


def run_logged_step(step_name: str, action: Callable[[], T]) -> T:
    """Выполнить шаг, записав начало, длительность и возможную ошибку."""
    logger = get_step_logger(step_name)
    started_at = time.monotonic()
    logger.info("START")
    try:
        result = action()
    except Exception:
        # logger.exception добавляет traceback, полезный при падении dvc repro.
        logger.exception("FAILED after %.2f s", time.monotonic() - started_at)
        raise
    logger.info("SUCCESS in %.2f s", time.monotonic() - started_at)
    return result
