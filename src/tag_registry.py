"""
tag_registry.py — публичный фасад: алиасы, нормализация, канонические имена.

Итерация 6.2 карты №2: реализации перенесены в src.config_models.tag_constants,
чтобы разорвать цикл tag_registry ↔ config_types ↔ config_models.tags.
Имена реэкспортируются для обратной совместимости:
все внешние callers (src/startup_checks.py, src/knowledge_retrieval.py,
src/contracts.py, src/registry.py, src/prompt_builder/config_loaders.py,
scripts/audit_tags.py, тесты) импортируют из src.tag_registry.
"""

from __future__ import annotations

from src.config_models.tag_constants import (  # noqa: F401
    _ALIAS_MAP,
    build_known_tags,
    get_canonical_tag_names,
    normalize_tag,
    normalize_tags,
)
