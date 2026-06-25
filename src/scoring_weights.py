"""
scoring_weights.py

Загрузка конфигурации весов скоринга из JSON-файла.
Позволяет менять веса без передеплоя приложения.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Final

logger = logging.getLogger(__name__)

_DEFAULT_WEIGHTS: Final[Dict[str, int]] = {
    "wrong_exact_match": 1000,
    "name_exact_match": 500,
    "partial_text_match": 200,
    "tag_primary": 10,
    "tag_primary_bonus": 1,
    "tag_expanded": 2,
}

_WEIGHTS_CACHE: Dict[str, int] | None = None
_CONFIG_PATH = Path("config") / "scoring_weights.json"


def load_scoring_weights() -> Dict[str, int]:
    """
    Загружает веса из config/scoring_weights.json.
    Если файл отсутствует или повреждён, возвращает значения по умолчанию
    и логирует предупреждение.
    """
    global _WEIGHTS_CACHE
    if _WEIGHTS_CACHE is not None:
        return _WEIGHTS_CACHE

    if not _CONFIG_PATH.is_file():
        logger.warning(
            "Scoring weights config not found at %s, using defaults.",
            _CONFIG_PATH,
        )
        _WEIGHTS_CACHE = _DEFAULT_WEIGHTS.copy()
        return _WEIGHTS_CACHE

    try:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            data = json.load(f)
        # Проверяем, что все ключи присутствуют и значения — int
        for key in _DEFAULT_WEIGHTS:
            if key not in data:
                raise ValueError(f"Missing key '{key}' in scoring_weights.json")
            if not isinstance(data[key], int):
                raise ValueError(f"Key '{key}' must be integer, got {type(data[key]).__name__}")
        # Дополнительные ключи в файле разрешены, но не используются
        _WEIGHTS_CACHE = {key: data[key] for key in _DEFAULT_WEIGHTS}
        logger.info("Loaded scoring weights from %s", _CONFIG_PATH)
    except Exception as e:
        logger.error("Failed to load scoring weights: %s. Using defaults.", e)
        _WEIGHTS_CACHE = _DEFAULT_WEIGHTS.copy()

    return _WEIGHTS_CACHE


# D-7: добавлено предупреждение при запросе неизвестного ключа
def get_scoring_weight(name: str) -> int:
    """Возвращает значение веса по имени."""
    weights = load_scoring_weights()
    if name not in weights:
        logger.warning(
            "Ключ %r не найден в scoring_weights.json. "
            "Правило получит вес 0 и не будет отобрано. "
            "Добавь ключ в конфиг.", name
        )
        return 0
    return weights[name]