"""
tag_registry.py

Единый реестр нормализации тегов для доменов, интентов и оверлеев.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Set

CANONICAL_TAG_ALIASES: Dict[str, str] = {
    "anti_ai": "antiai",
    "anti-ai": "antiai",
    "anti ai": "antiai",
    "de_ai": "deai",
    "de-ai": "deai",
    "de ai": "deai",
    "nora_gal": "noragal",
    "nora-gal": "noragal",
    "nora gal": "noragal",
    "info_style": "infostyle",
    "info-style": "infostyle",
    "info style": "infostyle",
    "fact_check": "factcheck",
    "fact-check": "factcheck",
    "fact check": "factcheck",
    "final_check": "finalcheck",
    "final-check": "finalcheck",
    "final check": "finalcheck",
    "marketing_push": "marketingpush",
    "marketing-push": "marketingpush",
    "marketing push": "marketingpush",
    "non_marketing": "nonmarketing",
    "non-marketing": "nonmarketing",
    "non marketing": "nonmarketing",
}


def normalize_tag(tag: str) -> str:
    """Нормализует один тег к каноническому виду."""
    value = tag.strip().lower()
    return CANONICAL_TAG_ALIASES.get(value, value)


def normalize_tags(tags: Iterable[str]) -> List[str]:
    """
    Нормализует список тегов, удаляя дубликаты с сохранением порядка.
    Нестрочные значения пропускаются.
    """
    seen: Set[str] = set()
    result: List[str] = []

    for raw in tags:
        if not isinstance(raw, str):
            continue

        normalized = normalize_tag(raw)
        if normalized in seen:
            continue

        seen.add(normalized)
        result.append(normalized)

    return result


def build_known_tags(canonical_tags: Dict[str, Dict[str, Any]]) -> Set[str]:
    """
    Строит множество всех известных нормализованных тегов
    из структуры canonical tags.
    """
    known: Set[str] = set()

    for category_map in canonical_tags.values():
        if not isinstance(category_map, dict):
            continue

        for payload in category_map.values():
            if isinstance(payload, dict):
                for key in ("primary", "expanded"):
                    value = payload.get(key)
                    if value is None:
                        continue

                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, str):
                                known.add(normalize_tag(item))
                    elif isinstance(value, str):
                        known.add(normalize_tag(value))

            elif isinstance(payload, list):
                for item in payload:
                    if isinstance(item, str):
                        known.add(normalize_tag(item))

    for alias, target in CANONICAL_TAG_ALIASES.items():
        known.add(normalize_tag(alias))
        known.add(normalize_tag(target))

    return known
