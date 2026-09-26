"""
config_types.py

Dataclasses, enum'ы и инфраструктурные типы для конфигурирования PromptBuilder.

Содержит:
- Domain types — RuleEntry, KnowledgeBase, CoreConfig и т.д.
- LimitsConfig — лимиты выдачи и кандидатов
- KnowledgeLevel — режим включения блоков знаний
- KnowledgeBlockPlan — описание блока для budget-aware сборки
- BlockBudget — бюджет одного блока KB
- KnowledgeBudget — совокупный бюджет всех блоков
- KnowledgeBudgetManager — вычисляет бюджет
- CachePolicy — политика инвалидации кэша
- FileCache — кэш-менеджер с поддержкой TTL/mtime
- Tag constants — CANONICAL_TAGS, KNOWN_TAGS, get_*_tags_for_category
- Explainability structures — FeatureResolutionResult, AssemblyBlockDiagnostics, AssemblyTrace
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Re-exports для обратной совместимости (итерация 7 дорожной карты).
# Код проекта исторически ожидает эти имена из src.config_types.
from src.config_models.cache import (  # noqa: F401
    CachePolicy,
    FileCache,
)
from src.config_models.domain import (  # noqa: F401
    AudienceProfile,
    CoreConfig,
    DomainConfig,
    EditorialTechniqueEntry,
    FlatEntry,
    IntentConfig,
    KnowledgeBase,
    OverlayConfig,
    RuleEntry,
    StructuralEntry,
)
from src.config_models.explainability import (  # noqa: F401
    AssemblyBlockDiagnostics,
    AssemblyTrace,
    FeatureResolutionResult,
)
from src.config_models.knowledge_budget import (  # noqa: F401
    BlockBudget,
    KnowledgeBlockPlan,
    KnowledgeBudget,
    KnowledgeBudgetManager,
    KnowledgeLevel,
    blocks_allowed_at_level,
)
from src.config_models.limits import (  # noqa: F401
    LimitsConfig,
)
from src.reason_codes import ReasonCode  # noqa: F401

logger = logging.getLogger(__name__)

# ============================================================================
# Domain types (moved to src.config_models.domain)
# ============================================================================


# ============================================================================
# LimitsConfig — лимиты выдачи и кандидатов (moved to src.config_models.limits)
# ============================================================================


# ============================================================================
# KnowledgeLevel — режим включения блоков знаний (moved to src.config_models.knowledge_budget)
# ============================================================================


# ============================================================================
# CachePolicy и FileCache (moved to src.config_models.cache)
# ============================================================================


# ============================================================================
# Tag constants и helpers
# ============================================================================


def _load_canonical_tags() -> dict[str, dict[str, Any]]:
    """
    Загружает CANONICAL_TAGS из config/tag_map.json.
    Файл ищется относительно корня проекта (два уровня выше этого модуля).
    При отсутствии файла возвращает пустой словарь и логирует предупреждение.
    """
    tag_map_path = Path(__file__).parent.parent / "config" / "tag_map.json"
    if not tag_map_path.exists():
        logger.warning(
            "tag_map.json not found at %s, CANONICAL_TAGS will be empty. "
            "Tag-based retrieval will degrade to fallback.",
            tag_map_path,
        )
        return {}
    try:
        with open(tag_map_path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        # Файл есть, но невалидный JSON или недоступен для чтения.
        # Это не критично: CANONICAL_TAGS просто останется пустым.
        logger.error("Failed to load tag_map.json: %s", e)
        return {}


CANONICAL_TAGS: dict[str, dict[str, Any]] = _load_canonical_tags()

KB_TAGS_STRICT_VALIDATION: bool = False


def _normalize_tag_local(tag: str) -> str:
    """Локальная нормализация тега без импорта tag_registry (для bootstrap)."""
    return tag.lower().replace("-", "_").replace(" ", "_")


def _normalize_tags_local(tags: list[str]) -> list[str]:
    return [_normalize_tag_local(tag) for tag in tags if isinstance(tag, str)]


def _resolve_normalizers() -> (
    tuple[
        Callable[[str], str],
        Callable[[list[str]], list[str]],
    ]
):
    """Resolve (normalize_tag, normalize_tags) from tag_registry or use local fallbacks."""
    try:
        from src.tag_registry import normalize_tag, normalize_tags
    except ImportError:
        return _normalize_tag_local, _normalize_tags_local
    return normalize_tag, normalize_tags


def _build_known_tags_from_canonical() -> set[str]:
    """Строит множество всех canonical тегов."""
    tags: set[str] = set()

    for category_data in CANONICAL_TAGS.values():
        for tag_data in category_data.values():
            if isinstance(tag_data, dict):
                for tag_list in tag_data.values():
                    if isinstance(tag_list, list):
                        tags.update(
                            _normalize_tag_local(tag) for tag in tag_list if isinstance(tag, str)
                        )

    return tags


KNOWN_TAGS: set[str] = _build_known_tags_from_canonical()


def get_canonical_tags_for_category(category: str, value: str) -> list[str]:
    """Возвращает primary + expanded теги для категории/значения."""
    normalize_tag, normalize_tags = _resolve_normalizers()

    norm_value = normalize_tag(value)
    data = CANONICAL_TAGS.get(category, {}).get(norm_value)

    if isinstance(data, dict):
        return normalize_tags(data.get("primary", []) + data.get("expanded", []))
    if isinstance(data, list):
        return normalize_tags(data)
    return normalize_tags([norm_value])


def get_primary_tags_for_category(category: str, value: str) -> list[str]:
    """Возвращает primary теги."""
    normalize_tag, normalize_tags = _resolve_normalizers()

    norm_value = normalize_tag(value)
    data = CANONICAL_TAGS.get(category, {}).get(norm_value)

    if isinstance(data, dict):
        return normalize_tags(data.get("primary", []))
    if isinstance(data, list):
        return normalize_tags(data)
    return normalize_tags([norm_value])


def get_expanded_tags_for_category(category: str, value: str) -> list[str]:
    """Возвращает expanded теги."""
    normalize_tag, normalize_tags = _resolve_normalizers()

    norm_value = normalize_tag(value)
    data = CANONICAL_TAGS.get(category, {}).get(norm_value)

    if isinstance(data, dict):
        return normalize_tags(data.get("expanded", []))
    return []


# ============================================================================
# Explainability structures (moved to src.config_models.explainability)
# ============================================================================
