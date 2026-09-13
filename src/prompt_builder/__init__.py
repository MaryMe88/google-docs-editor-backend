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

# Реэкспорт из config_types для обратной совместимости
from src.config_types import (
    AssemblyBlockDiagnostics,
    AssemblyTrace,
    AudienceProfile,
    BlockBudget,
    CachePolicy,
    CoreConfig,
    DomainConfig,
    FeatureResolutionResult,
    FileCache,
    IntentConfig,
    KnowledgeBase,
    KnowledgeBudget,
    KnowledgeBudgetManager,
    KnowledgeLevel,
    LimitsConfig,
    OverlayConfig,
    get_canonical_tags_for_category,
    get_primary_tags_for_category,
)

# Реэкспорт из reason_codes
from src.reason_codes import ReasonCode

# Реэкспорт из registry
from src.registry import (
    CANONICAL_FEATURE_ALIASES,
    KNOWN_FEATURE_ALIASES,
    check_alias_consistency,
    get_features_from_tags,
)

# Реэкспорт из shared_contracts
from src.shared_contracts import (
    ALLOWED_DOMAINS,
    ALLOWED_INTENTS,
    ALLOWED_OUTPUT_MODES,
    ALLOWED_OVERLAYS,
)

from .builder import KnowledgeBlockRequest, PromptBuilder
from .config_loaders import (
    load_core_config,
    load_domain_config,
    load_intent_config,
    load_json_file,
    load_output_format,
    load_overlay_config,
    load_overlay_configs,
)
from .defaults import (
    ALLOWED_EDIT_LEVELS,
    ALLOWED_KB_LIMIT_KEYS,
    KB_LIMIT_MAX,
    KB_LIMIT_MIN,
)
from .feature_resolution import resolve_prompt_features
from .kb_loading import load_knowledge_base
from .kb_rendering import (
    DEFAULT_CANDIDATE_LIMIT,
    KB_BLOCK_REGISTRY,
    KBBlockConfig,
    ProcessContext,
)
from .normalization import (
    normalize_intent,
    normalize_overlays,
    normalize_string_list,
)

# RUF022: __all__ намеренно сгруппирован по назначению (с комментариями),
# а не отсортирован по алфавиту — читаемость важнее машинной сортировки.
__all__ = [  # noqa: RUF022
    # Основной класс и dataclass запроса
    "PromptBuilder",
    "KnowledgeBlockRequest",
    # Загрузчики конфигов
    "load_core_config",
    "load_domain_config",
    "load_intent_config",
    "load_json_file",
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
