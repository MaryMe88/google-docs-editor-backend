"""
config_types.py — публичный фасад конфигурационных типов.

Содержит только реэкспорты из src.config_models.* и ReasonCode
из src.reason_codes для обратной совместимости.

Сами определения типов живут в тематических модулях:
- src.config_models.domain — домены, интенты, оверлеи, аудитория.
- src.config_models.limits — LimitsConfig.
- src.config_models.knowledge_budget — KnowledgeLevel, бюджеты.
- src.config_models.cache — CachePolicy, FileCache.
- src.config_models.tags — CANONICAL_TAGS, KNOWN_TAGS, helpers.
- src.config_models.explainability — FeatureResolutionResult,
  AssemblyBlockDiagnostics, AssemblyTrace.

Не добавляйте сюда новые определения: расширяйте соответствующий
модуль в src/config_models/ и реэкспортируйте при необходимости.
"""

from __future__ import annotations

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
