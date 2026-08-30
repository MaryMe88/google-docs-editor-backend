# tests/test_prompt_builder_package_structure.py
"""
Проверяет, что публичный API пакета src.prompt_builder не потерял
ни одного имени после рефакторинга из монолитного файла в пакет.
"""

from __future__ import annotations

import src.prompt_builder as pb


# Список имён, которые были доступны в монолитном prompt_builder.py
# и должны оставаться доступными через src.prompt_builder.
EXPECTED_PUBLIC_NAMES = [
    # Класс PromptBuilder
    "PromptBuilder",
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
    # Функции нормализации
    "normalize_intent",
    "normalize_overlays",
    "normalize_string_list",
    # Разрешение фич
    "resolve_prompt_features",
    "get_features_from_tags",
    "check_alias_consistency",
    # Вспомогательные функции (использовались в тестах)
    "_collect_retrieval_tags",
    "_process_kb_block",
    "_has_few_shot_pair",
    "_format_few_shot_example",
    "_select_few_shot_examples",
    "_derive_seed",
    "_get_confidence_note",
    "_unpack_retrieval_result",
    "_append_rule_entries",
    "_append_structural_entries",
    "_append_editorial_entries",
    "_append_case_study_entries",
    "_append_evaluation_techniques",
    "_append_glossary",
    "_append_nkrj",
    "_warn_if_empty_retrieval",
    "_TAG_TO_FEATURE",
    "_build_overlay_slug_map",
    "_add_activation_reason",
    "_add_suppression_reason",
    "_add_recognized_alias",
    "_add_ignored_unknown",
    # Типы из config_types, которые были доступны через prompt_builder
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
    """Все ожидаемые имена должны быть доступны из src.prompt_builder."""
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
    assert callable(pb.resolve_prompt_features), "resolve_prompt_features не является callable"


def test_kb_block_registry_is_list():
    """KB_BLOCK_REGISTRY должен быть списком."""
    assert isinstance(pb.KB_BLOCK_REGISTRY, list), "KB_BLOCK_REGISTRY не является списком"


# Дополнительно: можно проверить, что нет неожиданного отсутствия
# важных внутренних функций, которые использовались в тестах.
def test_private_helpers_available():
    """Проверка вспомогательных функций с подчёркиванием."""
    private_helpers = [
        "_load_kb_file",
        "_is_incompatible_intent",
        "_is_incompatible_overlay",
        "_normalize_overlay_ref",
        "_make_default_overlay_config",
        "_DEFAULT_DOMAIN_CONFIG",
        "ALLOWED_KB_LIMIT_KEYS",
        "ALLOWED_EDIT_LEVELS",
        "KB_LIMIT_MIN",
        "KB_LIMIT_MAX",
    ]
    missing = [name for name in private_helpers if not hasattr(pb, name)]
    assert not missing, f"Отсутствуют приватные имена: {missing}"