# src/prompt_builder/_patchable.py
"""
Модуль-прослойка для объектов, используемых в builder.py.
Вынесен отдельно, чтобы избежать циклического импорта через __init__.py.
Все импорты — из подмодулей пакета, без зависимостей от __init__.py.
"""

from .config_loaders import (
    load_core_config,
    load_domain_config,
    load_intent_config,
    load_overlay_config,
    load_output_format,
)
from .kb_loading import load_knowledge_base
from .kb_rendering import (
    KB_BLOCK_REGISTRY,
    _process_kb_block,
    _collect_retrieval_tags,
    _append_glossary,
    _append_nkrj,
    _append_evaluation_techniques,
    _derive_seed,
)
from .feature_resolution import resolve_prompt_features
