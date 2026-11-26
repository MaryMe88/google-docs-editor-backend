# src/prompt_builder/_patchable.py
# ruff: noqa: F401
"""
Слой ре-экспорта символов, используемых builder.py.

Историческая причина выделения: избежать циклического импорта
через __init__.py. builder.py импортирует эти 14 символов
из четырёх подмодулей пакета (config_loaders, feature_resolution,
kb_loading, kb_rendering) через один компактный блок
`from ._patchable import (...)`, а не четырьмя отдельными блоками.

Тестовая инфраструктура патчит имена в namespace builder
(src.prompt_builder.builder.<symbol>), а не здесь. Наличие
этого модуля гарантирует, что имена load_knowledge_base,
_process_kb_block, KB_BLOCK_REGISTRY, load_domain_config
и остальные связаны в builder в одном явном месте.

Удаление модуля возможно технически (заменить один импорт на четыре),
но не даёт функционального выигрыша и увеличивает diff без причины.
Решение принято в итерации 16.4 карты №3: оставить с обоснованием.
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
