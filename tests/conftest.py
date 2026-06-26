"""
conftest.py

Общие фикстуры для тестов.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.prompt_builder import AudienceProfile, PromptBuilder, load_knowledge_base
from src.knowledge_retrieval import FallbackStage

# Устанавливаем фейковый API-ключ для тестов, чтобы lifespan не падал
os.environ["OPENROUTER_API_KEY"] = "test-key"

# Устанавливаем флаг, чтобы приложение знало, что запущены тесты
# (используется для отключения rate limiting в тестах)
os.environ["PYTEST_RUNNING"] = "true"


# ============================================================================
# Пути
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KB_PATH = PROJECT_ROOT / "knowledge_base"
CONFIG_PATH = PROJECT_ROOT / "config"


# ============================================================================
# Фикстуры
# ============================================================================


@pytest.fixture
def kb_path() -> Path:
    return KB_PATH


@pytest.fixture
def config_path() -> Path:
    return CONFIG_PATH


@pytest.fixture
def builder() -> PromptBuilder:
    return PromptBuilder(config_path=CONFIG_PATH, kb_path=KB_PATH)


@pytest.fixture
def sample_audience() -> AudienceProfile:
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


# ============================================================================
# Фикстуры для KB-3 (золотой набор)
# ============================================================================


@pytest.fixture(scope="session")
def knowledge_base() -> Any:
    """Загружает базу знаний один раз для всех тестов."""
    if not KB_PATH.exists():
        pytest.skip(f"Knowledge base directory not found: {KB_PATH}")
    return load_knowledge_base(KB_PATH)


@pytest.fixture(scope="session")
def golden_set() -> List[Dict[str, Any]]:
    """Загружает golden_set.json из корня проекта или папки tests."""
    # Сначала ищем в tests/
    golden_path = Path(__file__).parent / "golden_set.json"
    if not golden_path.exists():
        # Затем в корне проекта
        golden_path = PROJECT_ROOT / "golden_set.json"
    if not golden_path.exists():
        pytest.skip("golden_set.json not found")
    with open(golden_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["tests"]


def load_json(path: Path) -> Dict[str, Any]:
    """Утилита для загрузки JSON в тестах."""
    return json.loads(path.read_text(encoding="utf-8"))


# ============================================================================
# Настройка пропуска интеграционных тестов (SEC-07)
# ============================================================================

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