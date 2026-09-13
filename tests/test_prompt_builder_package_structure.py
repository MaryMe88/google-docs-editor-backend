# tests/test_prompt_builder_package_structure.py
"""
Проверяет, что публичный API пакета src.prompt_builder не потерял
ни одного публичного имени после рефакторинга из монолитного файла в пакет.

Приватные имена (с `_`-префиксом) намеренно НЕ входят в публичный API
пакета — они доступны из соответствующих подмодулей напрямую, например:
    from src.prompt_builder.kb_rendering import _process_kb_block
    from src.prompt_builder.kb_loading import _load_kb_file
"""

from __future__ import annotations

import src.prompt_builder as pb

# Список публичных имён, которые должны быть доступны через src.prompt_builder.
EXPECTED_PUBLIC_NAMES = [
    # Класс PromptBuilder и dataclass запроса
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
    # Функции нормализации
    "normalize_intent",
    "normalize_overlays",
    "normalize_string_list",
    # Разрешение фич
    "resolve_prompt_features",
    "get_features_from_tags",
    "check_alias_consistency",
    # Типы из config_types, доступные через prompt_builder
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
    # Константы из shared_contracts
    "ALLOWED_DOMAINS",
    "ALLOWED_INTENTS",
    "ALLOWED_OVERLAYS",
    "ALLOWED_OUTPUT_MODES",
    # ReasonCode
    "ReasonCode",
]


def test_prompt_builder_is_package():
    """src.prompt_builder должен быть пакетом (иметь __path__)."""
    assert hasattr(pb, "__path__"), "src.prompt_builder не является пакетом"


def test_all_expected_names_are_available():
    """Все ожидаемые публичные имена должны быть доступны из src.prompt_builder."""
    missing = [name for name in EXPECTED_PUBLIC_NAMES if not hasattr(pb, name)]
    assert not missing, f"Отсутствуют имена в публичном API: {missing}"


def test_prompt_builder_class_is_class():
    """PromptBuilder должен быть классом."""
    assert isinstance(pb.PromptBuilder, type), "PromptBuilder не является классом"


def test_load_domain_config_is_callable():
    """load_domain_config должна быть функцией (callable)."""
    assert callable(pb.load_domain_config), "load_domain_config не является callable"


def test_resolve_prompt_features_is_callable():
    """resolve_prompt_features должна быть функцией."""
    assert callable(pb.resolve_prompt_features), (
        "resolve_prompt_features не является callable"
    )


def test_kb_block_registry_is_list():
    """KB_BLOCK_REGISTRY должен быть списком."""
    assert isinstance(pb.KB_BLOCK_REGISTRY, list), (
        "KB_BLOCK_REGISTRY не является списком"
    )
