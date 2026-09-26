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

import logging

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
from src.config_models.tags import (  # noqa: F401
    CANONICAL_TAGS,
    KB_TAGS_STRICT_VALIDATION,
    KNOWN_TAGS,
    get_canonical_tags_for_category,
    get_expanded_tags_for_category,
    get_primary_tags_for_category,
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
# Tag constants и helpers (moved to src.config_models.tags)
# ============================================================================
