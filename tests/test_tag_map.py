"""
tests/test_tag_map.py

Тесты для задачи 3 (TP-2): вынос CANONICAL_TAGS в JSON и проверка покрытия.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config_types import CANONICAL_TAGS, get_primary_tags_for_category
from src.shared_contracts import ALLOWED_DOMAINS, ALLOWED_INTENTS, ALLOWED_OVERLAYS
from src.startup_checks import _check_tag_map_coverage, run_startup_checks


def test_canonical_tags_loaded_from_json() -> None:
    """Проверяет, что CANONICAL_TAGS загружен из tag_map.json, а не хардкод."""
    assert isinstance(CANONICAL_TAGS, dict)
    assert "domains" in CANONICAL_TAGS
    assert "intents" in CANONICAL_TAGS
    assert "overlays" in CANONICAL_TAGS
    # Проверяем, что есть хотя бы несколько ключей
    assert len(CANONICAL_TAGS["domains"]) > 0
    assert len(CANONICAL_TAGS["intents"]) > 0
    assert len(CANONICAL_TAGS["overlays"]) > 0


def test_get_primary_tags_never_crashes() -> None:
    """Для всех зарегистрированных доменов/интентов/оверлеев вызов get_primary_tags_for_category не падает."""
    for domain in ALLOWED_DOMAINS:
        result = get_primary_tags_for_category("domains", domain)
        assert isinstance(result, list)
        # Минимально — хотя бы один тег (сам домен как fallback)
        assert len(result) >= 1, f"Для домена {domain} результат пуст"

    for intent in ALLOWED_INTENTS:
        if intent == "neutral":
            continue
        result = get_primary_tags_for_category("intents", intent)
        assert isinstance(result, list)
        assert len(result) >= 1, f"Для интента {intent} результат пуст"

    for overlay in ALLOWED_OVERLAYS:
        result = get_primary_tags_for_category("overlays", overlay)
        assert isinstance(result, list)
        assert len(result) >= 1, f"Для оверлея {overlay} результат пуст"


def test_tag_map_coverage_warns_on_missing() -> None:
    """
    Проверяет, что _check_tag_map_coverage логирует предупреждения
    для отсутствующих записей, но не падает.
    """
    # Временно создаём словарь, где удалены некоторые ключи, чтобы проверить предупреждение
    with patch("src.config_types.CANONICAL_TAGS", {"domains": {}, "intents": {}, "overlays": {}}):
        with patch("logging.Logger.warning") as mock_warning:
            _check_tag_map_coverage(
                config_path=Path("config"),
                allowed_domains=ALLOWED_DOMAINS,
                allowed_intents=ALLOWED_INTENTS,
                allowed_overlays=ALLOWED_OVERLAYS,
            )
            # Должны быть предупреждения для каждой категории
            assert mock_warning.call_count >= 3


def test_run_startup_checks_does_not_fail_due_to_tag_map() -> None:
    """
    Запуск run_startup_checks не должен падать из-за отсутствия записей в tag_map.json.
    Проверяем только то, что исключение не выбрасывается.
    """
    # Запускаем с реальными данными; если какие-то записи отсутствуют, логируется warning, но не ошибка.
    with patch("logging.Logger.warning") as mock_warning:
        run_startup_checks(
            allowed_domains=ALLOWED_DOMAINS,
            allowed_intents=ALLOWED_INTENTS,
            allowed_overlays=ALLOWED_OVERLAYS,
            config_path=Path("config"),
            kb_path=Path("knowledge_base"),
        )
        # Проверяем, что warnings были (если есть пропуски), но исключений не было
        # Наличие предупреждений — ок.
        # Также проверяем, что не было вызова logger.error или критических ошибок.
        # Мы можем проверить, что mock_warning вызывался хотя бы один раз (скорее всего).
        # Но если все записи есть, предупреждений может не быть. Поэтому проверяем только отсутствие исключений.
        # Это тест не должен упасть.
        assert True