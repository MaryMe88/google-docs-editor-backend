"""
config_models.tag_constants

Чистые данные и функции нормализации тегов — БЕЗ top-level импортов из src.*.

Итерация 6.2 карты №2: разрыв цикла
tag_registry ↔ config_types ↔ config_models.tags.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

_ALIAS_MAP: dict[str, str] = {
    # ---- домены / теги ----
    "anti_ai": "antiai",
    "anti-ai": "antiai",
    "antiai": "antiai",
    "de_ai": "deai",
    "de-ai": "deai",
    "deai": "deai",
    "nora_gal": "noragal",
    "nora-gal": "noragal",
    "noragal": "noragal",
    "marketing_push": "marketingpush",
    "marketing-push": "marketingpush",
    "marketingpush": "marketingpush",
    "info_style": "infostyle",
    "info-style": "infostyle",
    "infostyle": "infostyle",
    "final_check": "finalcheck",
    "final-check": "finalcheck",
    "finalcheck": "finalcheck",
    "non_marketing": "nonmarketing",
    "non-marketing": "nonmarketing",
    "nonmarketing": "nonmarketing",
    # ---- оверлеи (snake_case — каноническая форма совпадает с именем файла) ----
    # Эти записи гарантируют идемпотентность:
    #   normalize_tag("finalcheck_full") == "finalcheck_full"
    # Без них normalize_tag стрипает "_" и возвращает "finalcheckfull",
    # что не совпадает с именем файла → FileNotFoundError при загрузке.
    "finalcheck_full": "finalcheck_full",
    "finalcheck-full": "finalcheck_full",
    "finalcheckfull": "finalcheck_full",
    "finalcheck_light": "finalcheck_light",
    "finalcheck-light": "finalcheck_light",
    "finalchecklight": "finalcheck_light",
    "no_overwrite": "no_overwrite",
    "no-overwrite": "no_overwrite",
    "nooverwrite": "no_overwrite",
}


def normalize_tag(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"tag must be str, got {type(value)!r}")
    raw = value.strip().lower()
    if not raw:
        return ""
    if raw in _ALIAS_MAP:
        return _ALIAS_MAP[raw]
    compact = re.sub(r"[^a-z0-9]+", "", raw)
    return _ALIAS_MAP.get(compact, compact)


def normalize_tags(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        normalized = normalize_tag(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def build_known_tags(mapping: dict[str, Any]) -> set[str]:
    known: set[str] = set()

    def _visit(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(key, str):
                    normalized_key = normalize_tag(key)
                    if normalized_key:
                        known.add(normalized_key)
                _visit(value)
        elif isinstance(obj, list):
            for item in obj:
                _visit(item)
        elif isinstance(obj, str):
            normalized_value = normalize_tag(obj)
            if normalized_value:
                known.add(normalized_value)

    _visit(mapping)
    return known


def get_canonical_tag_names(
    canonical_tags: dict[str, dict[str, Any]] | None = None,
) -> set[str]:
    """
    Возвращает множество канонических primary-тегов из CANONICAL_TAGS (без алиасов).
    Используется в startup-проверках для сверки с реальными тегами в KB.

    canonical_tags: если None — lazy-импорт CANONICAL_TAGS из config_models.tags
    (единственная зависимость, lazy, в одну сторону — не создаёт load-цикл).
    """
    if canonical_tags is None:
        from src.config_models.tags import CANONICAL_TAGS

        canonical_tags = CANONICAL_TAGS

    known_aliases = {
        "antillm",
        "antiplastic",
        "taiga",
        "socialnorms",
        "basicedit",
        "cleanup",
        "story",
    }

    tags: set[str] = set()
    for category_data in canonical_tags.values():
        if not isinstance(category_data, dict):
            continue
        for payload in category_data.values():
            if isinstance(payload, dict):
                for tag in payload.get("primary", []):
                    if isinstance(tag, str):
                        norm = normalize_tag(tag)
                        if norm == "basicedit":
                            norm = "editorial"
                        if norm not in known_aliases:
                            tags.add(norm)
            elif isinstance(payload, list):
                for tag in payload:
                    if isinstance(tag, str):
                        norm = normalize_tag(tag)
                        if norm == "basicedit":
                            norm = "editorial"
                        if norm not in known_aliases:
                            tags.add(norm)
    return tags
