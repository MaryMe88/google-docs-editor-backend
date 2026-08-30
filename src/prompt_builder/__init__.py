# src/prompt_builder/__init__.py
"""
Публичный API пакета prompt_builder.
"""

from .normalization import (
    normalize_intent,
    normalize_overlays,
    normalize_string_list,
    _is_incompatible_intent,
    _is_incompatible_overlay,
    _normalize_overlay_ref,
)

from .defaults import (
    ALLOWED_KB_LIMIT_KEYS,
    ALLOWED_EDIT_LEVELS,
    KB_LIMIT_MIN,
    KB_LIMIT_MAX,
    _DEFAULT_DOMAIN_CONFIG,
    _make_default_overlay_config,
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

from .kb_loading import _load_kb_file, load_knowledge_base

from .kb_rendering import (
    KBBlockConfig,
    KB_BLOCK_REGISTRY,
    DEFAULT_CANDIDATE_LIMIT,
    _collect_retrieval_tags,
    _append_rule_entries,
    _append_structural_entries,
    _append_editorial_entries,
    _append_case_study_entries,
    _append_evaluation_techniques,
    _append_glossary,
    _append_nkrj,
    _warn_if_empty_retrieval,
    _process_kb_block,
    _has_few_shot_pair,
    _format_few_shot_example,
    _select_few_shot_examples,
    _derive_seed,
    _unpack_retrieval_result,
    _get_confidence_note,
)

from .feature_resolution import (
    resolve_prompt_features,
    _build_overlay_slug_map,
    _add_activation_reason,
    _add_suppression_reason,
    _add_recognized_alias,
    _add_ignored_unknown,
    _TAG_TO_FEATURE,
)

from .builder import PromptBuilder

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

# Реэкспорт из shared_contracts (были доступны в старом prompt_builder)
from src.shared_contracts import (
    ALLOWED_DOMAINS,
    ALLOWED_INTENTS,
    ALLOWED_OUTPUT_MODES,
    ALLOWED_OVERLAYS,
)

# Реэкспорт из reason_codes
from src.reason_codes import ReasonCode

# Реэкспорт из registry (если нужен в тестах)
from src.registry import (
    CANONICAL_FEATURE_ALIASES,
    KNOWN_FEATURE_ALIASES,
    get_features_from_tags,
    check_alias_consistency,
)