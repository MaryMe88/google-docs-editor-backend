# src/prompt_builder/__init__.py
"""
Публичный API пакета prompt_builder.

Здесь реэкспортируются только публичные имена (без `_`-префикса).
Приватные вспомогательные функции доступны напрямую из подмодулей,
например:
    from src.prompt_builder.kb_rendering import _process_kb_block
    from src.prompt_builder.normalization import _is_incompatible_intent

Это позволяет тестам и внутреннему коду использовать их, не превращая
в часть публичного API пакета.
"""

from .normalization import (
    normalize_intent,
    normalize_overlays,
    normalize_string_list,
)

from .defaults import (
    ALLOWED_KB_LIMIT_KEYS,
    ALLOWED_EDIT_LEVELS,
    KB_LIMIT_MIN,
    KB_LIMIT_MAX,
)

from .config_loaders import (
    load_json_file,
    load_core_config,
    load_domain_config,
    load_intent_config,
    load_overlay_config,
    load_overlay_configs,
    load_output_format,
)

from .kb_loading import load_knowledge_base

from .kb_rendering import (
    KBBlockConfig,
    KB_BLOCK_REGISTRY,
    DEFAULT_CANDIDATE_LIMIT,
    ProcessContext,
)

from .feature_resolution import resolve_prompt_features

from .builder import PromptBuilder, KnowledgeBlockRequest

# Реэкспорт из config_types для обратной совместимости
from src.config_types import (
    AudienceProfile,
    CoreConfig,
    DomainConfig,
    IntentConfig,
    KnowledgeBase,
    KnowledgeBudget,
    KnowledgeBudgetManager,
    KnowledgeLevel,
    LimitsConfig,
    OverlayConfig,
    BlockBudget,
    AssemblyBlockDiagnostics,
    AssemblyTrace,
    FeatureResolutionResult,
    CachePolicy,
    FileCache,
    get_canonical_tags_for_category,
    get_primary_tags_for_category,
)

# Реэкспорт из shared_contracts
from src.shared_contracts import (
    ALLOWED_DOMAINS,
    ALLOWED_INTENTS,
    ALLOWED_OUTPUT_MODES,
    ALLOWED_OVERLAYS,
)

# Реэкспорт из reason_codes
from src.reason_codes import ReasonCode

# Реэкспорт из registry
from src.registry import (
    CANONICAL_FEATURE_ALIASES,
    KNOWN_FEATURE_ALIASES,
    get_features_from_tags,
    check_alias_consistency,
)

__all__ = [
    # Основной класс и dataclass запроса
    "PromptBuilder",
    "KnowledgeBlockRequest",
    # Загрузчики конфигов
    "load_core_config",
    "load_domain_config",
    "load_intent_config",
    "load_overlay_config",
    "load_overlay_configs",
    "load_output_format",
    # Загрузка KB
    "load_knowledge_base",
    "KBBlockConfig",
    "KB_BLOCK_REGISTRY",
    "DEFAULT_CANDIDATE_LIMIT",
    "ProcessContext",
    # Нормализация
    "normalize_intent",
    "normalize_overlays",
    "normalize_string_list",
    # Разрешение фич
    "resolve_prompt_features",
    "get_features_from_tags",
    "check_alias_consistency",
    # Типы из config_types
    "AudienceProfile",
    "CoreConfig",
    "DomainConfig",
    "IntentConfig",
    "OverlayConfig",
    "KnowledgeBase",
    "KnowledgeBudget",
    "KnowledgeBudgetManager",
    "KnowledgeLevel",
    "LimitsConfig",
    "BlockBudget",
    "CachePolicy",
    "FileCache",
    "FeatureResolutionResult",
    "AssemblyBlockDiagnostics",
    "AssemblyTrace",
    # Константы
    "ALLOWED_DOMAINS",
    "ALLOWED_INTENTS",
    "ALLOWED_OVERLAYS",
    "ALLOWED_OUTPUT_MODES",
    "ReasonCode",
    "CANONICAL_FEATURE_ALIASES",
    "KNOWN_FEATURE_ALIASES",
    "get_canonical_tags_for_category",
    "get_primary_tags_for_category",
    # Границы лимитов
    "ALLOWED_KB_LIMIT_KEYS",
    "ALLOWED_EDIT_LEVELS",
    "KB_LIMIT_MIN",
    "KB_LIMIT_MAX",
]