"""
scoring_weights.py

Загрузка конфигурации весов скоринга из JSON-файла.
Позволяет менять веса без передеплоя приложения.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

_DEFAULT_WEIGHTS: Final[dict[str, int]] = {
    "wrong_exact_match": 1000,
    "name_exact_match": 500,
    "partial_text_match": 200,
    "tag_primary": 10,
    "tag_primary_bonus": 1,
    "tag_expanded": 2,
}

_CONFIG_PATH = Path("config") / "scoring_weights.json"


@lru_cache(maxsize=1)
def _load_scoring_weights_cached() -> dict[str, int]:
    """
    Загружает веса из config/scoring_weights.json (с кешированием в lru_cache).

    Если файл отсутствует или повреждён, возвращает значения по умолчанию
    и логирует предупреждение. Дополнительные ключи в файле разрешены,
    но не используются.
    """
    if not _CONFIG_PATH.is_file():
        logger.warning(
            "Scoring weights config not found at %s, using defaults.",
            _CONFIG_PATH,
        )
        return _DEFAULT_WEIGHTS.copy()

    try:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            data = json.load(f)
        for key in _DEFAULT_WEIGHTS:
            if key not in data:
                raise ValueError(f"Missing key '{key}' in scoring_weights.json")
            if not isinstance(data[key], int):
                raise ValueError(f"Key '{key}' must be integer, got " f"{type(data[key]).__name__}")
        result = {key: data[key] for key in _DEFAULT_WEIGHTS}
        logger.info("Loaded scoring weights from %s", _CONFIG_PATH)
        return result
    except (json.JSONDecodeError, OSError, ValueError, TypeError) as e:
        logger.error(
            "Failed to load scoring weights: %s. Using defaults.",
            e,
        )
        return _DEFAULT_WEIGHTS.copy()


def load_scoring_weights() -> dict[str, int]:
    """Возвращает веса скоринга (кешируются через lru_cache)."""
    return _load_scoring_weights_cached()


def reset_scoring_weights_cache() -> None:
    """Сбрасывает кеш весов; для тестов и hot-reload."""
    _load_scoring_weights_cached.cache_clear()


def get_scoring_weight(name: str) -> int:
    """Возвращает значение веса по имени."""
    weights = load_scoring_weights()
    if name not in weights:
        logger.warning(
            "Ключ %r не найден в scoring_weights.json. "
            "Правило получит вес 0 и не будет отобрано. "
            "Добавь ключ в конфиг.",
            name,
        )
        return 0
    return weights[name]
