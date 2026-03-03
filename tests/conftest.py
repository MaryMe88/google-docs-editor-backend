"""
conftest.py

Общие фикстуры для тестов.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from src.prompt_builder import AudienceProfile, PromptBuilder


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


def load_json(path: Path) -> Dict[str, Any]:
    """Утилита для загрузки JSON в тестах."""
    return json.loads(path.read_text(encoding="utf-8"))
