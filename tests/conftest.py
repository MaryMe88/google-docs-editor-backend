"""
conftest.py

Общие фикстуры для тестов.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

if TYPE_CHECKING:
    # Импортируются только для аннотаций типов.
    # Во время выполнения фикстуры импортируют эти классы локально,
    # чтобы глобальный мок sentence-transformers успел примениться.
    from src.config_types import AudienceProfile
    from src.prompt_builder import PromptBuilder

# ---------------------------------------------------------------------------
# ГЛОБАЛЬНЫЙ МОК sentence-transformers ДО ИМПОРТА ОСТАЛЬНЫХ МОДУЛЕЙ
# ---------------------------------------------------------------------------
# Создаём мок-класс для SentenceTransformer


class MockSentenceTransformer:
    def __init__(self, model_name, **kwargs):
        pass

    def encode(self, texts, **kwargs):
        # Возвращаем нулевые эмбеддинги
        if isinstance(texts, str):
            texts = [texts]
        return np.zeros((len(texts), 384), dtype=np.float32)


# Подменяем модуль в sys.modules, чтобы импорт возвращал мок
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["sentence_transformers"].SentenceTransformer = MockSentenceTransformer

# Также подменяем сам класс, чтобы при импорте из модуля получали мок
# Это необходимо для случаев, когда импорт делается через from sentence_transformers import SentenceTransformer
import sentence_transformers

sentence_transformers.SentenceTransformer = MockSentenceTransformer

# ---------------------------------------------------------------------------
# Устанавливаем фейковый API-ключ для тестов
# ---------------------------------------------------------------------------
os.environ["OPENROUTER_API_KEY"] = "test-key"
os.environ["PYTEST_RUNNING"] = "true"

# ---------------------------------------------------------------------------
# Пути
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
KB_PATH = PROJECT_ROOT / "knowledge_base"
CONFIG_PATH = PROJECT_ROOT / "config"

# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------


@pytest.fixture
def kb_path() -> Path:
    return KB_PATH


@pytest.fixture
def config_path() -> Path:
    return CONFIG_PATH


@pytest.fixture
def builder() -> PromptBuilder:
    # Импортируем здесь, чтобы мок уже был применён
    from src.prompt_builder import PromptBuilder

    return PromptBuilder(config_path=CONFIG_PATH, kb_path=KB_PATH)


@pytest.fixture
def sample_audience() -> AudienceProfile:
    from src.config_types import AudienceProfile

    return AudienceProfile(
        kind="b2b",
        expertise="pro",
        formality="neutral",
        description="Менеджеры по продукту",
    )


@pytest.fixture
def sample_text() -> str:
    return (
        "Наш сервис является самым лучшим на рынке. "
        "Мы осуществляем проведение анализа данных. "
        "В целом, это очень эффективное решение."
    )


# ---------------------------------------------------------------------------
# Фикстуры для KB-3 (золотой набор)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def knowledge_base() -> Any:
    """Загружает базу знаний один раз для всех тестов."""
    from src.prompt_builder import load_knowledge_base

    if not KB_PATH.exists():
        pytest.skip(f"Knowledge base directory not found: {KB_PATH}")
    return load_knowledge_base(KB_PATH, load_all=True)


@pytest.fixture(scope="session")
def golden_set() -> list[dict[str, Any]]:
    """Загружает golden_set.json из корня проекта или папки tests."""
    golden_path = Path(__file__).parent / "golden_set.json"
    if not golden_path.exists():
        golden_path = PROJECT_ROOT / "golden_set.json"
    if not golden_path.exists():
        pytest.skip("golden_set.json not found")
    with open(golden_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["tests"]


def load_json(path: Path) -> dict[str, Any]:
    """Утилита для загрузки JSON в тестах."""
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Настройка пропуска интеграционных тестов (SEC-07)
# ---------------------------------------------------------------------------


def pytest_configure(config):
    """Читаем INTEGRATION_TESTS_ENABLED и сохраняем в конфиг."""
    enabled = os.getenv("INTEGRATION_TESTS_ENABLED", "").lower() in ("true", "1", "yes")
    config._integration_enabled = enabled


def pytest_collection_modifyitems(config, items):
    """Пропускаем интеграционные тесты, если флаг не установлен."""
    enabled = config._integration_enabled
    skip_integration = pytest.mark.skip(reason="INTEGRATION_TESTS_ENABLED not set to true")
    for item in items:
        if item.get_closest_marker("integration") and not enabled:
            item.add_marker(skip_integration)


# ---------------------------------------------------------------------------
# Фикстура для сброса глобального состояния SemanticIndex
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_semantic_index():
    """
    Сбрасывает глобальное состояние SemanticIndex перед каждым тестом.
    Предотвращает ленивую инициализацию индекса с реальной моделью.
    """
    import src.semantic_index as si

    si._global_index = None
    si._entries_for_index = None
    yield


# ---------------------------------------------------------------------------
# Фикстура для мока _semantic_rerank (дополнительная защита)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="session")
def mock_semantic_rerank():
    """
    Мокает _semantic_rerank, чтобы он не выполнял реальный поиск по индексу,
    а возвращал исходный список записей без изменений.
    """
    with patch(
        "src.knowledge_retrieval._semantic_rerank",
        side_effect=lambda entries, query, *args, **kwargs: entries,
    ):
        yield
