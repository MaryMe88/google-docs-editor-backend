# src/prompt_builder/_patchable.py
# ruff: noqa: F401
"""
Модуль-прослойка для объектов, используемых в builder.py.
Вынесен отдельно, чтобы избежать циклического импорта через __init__.py.
Все импорты — из подмодулей пакета, без зависимостей от __init__.py.
"""

from .config_loaders import (
    load_core_config,
    load_domain_config,
    load_intent_config,
    load_output_format,
    load_overlay_config,
)
from .feature_resolution import resolve_prompt_features
from .kb_loading import load_knowledge_base
from .kb_rendering import (
    KB_BLOCK_REGISTRY,
    _append_evaluation_techniques,
    _append_glossary,
    _append_nkrj,
    _collect_retrieval_tags,
    _derive_seed,
    _process_kb_block,
)
