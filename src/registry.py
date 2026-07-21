# src/registry.py
"""
Канонический registry для известных значений, алиасов и тегов.
Единый источник истины для runtime, validation и tests.
"""
from __future__ import annotations

from typing import Dict, List, Set, Final

from src.shared_contracts import ALLOWED_INTENTS, ALLOWED_OVERLAYS
from src.tag_registry import normalize_tag

# ---------------------------------------------------------------------------
# Canonical feature aliases (фичи и их синонимы)
# ---------------------------------------------------------------------------
CANONICAL_FEATURE_ALIASES: Final[Dict[str, List[str]]] = {
    "storytelling": ["storytelling", "story", "narrative"],
    "marketing": ["marketing", "marketingpush", "sales", "promo"],
    "antiai": ["deai", "antiai", "anti-llm", "humanize", "antiplastic"],
    "rhetoric": ["rhetoric", "persuasion", "figures"],
    "nkrj": ["nkrj", "taiga", "socialnorms"],
    "editorial": ["editorial", "editing", "noragal", "cleanup", "readerfirst", "basic_edit"],
}

# Обратный маппинг: тег -> фича
_TAG_TO_FEATURE: Dict[str, str] = {}
for feature, aliases in CANONICAL_FEATURE_ALIASES.items():
    for alias in aliases:
        _TAG_TO_FEATURE[alias] = feature

KNOWN_FEATURE_ALIASES: Final[Set[str]] = set(_TAG_TO_FEATURE.keys())

# ---------------------------------------------------------------------------
# Функции для работы с алиасами
# ---------------------------------------------------------------------------
def get_feature_for_tag(tag: str) -> str | None:
    """Возвращает имя фичи для данного тега (нормализованного) или None."""
    return _TAG_TO_FEATURE.get(tag)

def get_features_from_tags(tags: List[str]) -> Set[str]:
    """Возвращает множество фич, соответствующих переданным тегам."""
    features = set()
    for tag in tags:
        norm = normalize_tag(tag)
        if norm in _TAG_TO_FEATURE:
            features.add(_TAG_TO_FEATURE[norm])
    return features

# ---------------------------------------------------------------------------
# Проверка согласованности с конфигами (для валидации)
# ---------------------------------------------------------------------------
def get_known_intents() -> Set[str]:
    """Возвращает множество известных интентов (из shared_contracts)."""
    return set(ALLOWED_INTENTS)

def get_known_overlays() -> Set[str]:
    """Возвращает множество известных оверлеев (из shared_contracts)."""
    return set(ALLOWED_OVERLAYS)

# ---------------------------------------------------------------------------
# Проверка согласованности алиасов с тегами (для валидации)
# ---------------------------------------------------------------------------
def check_alias_consistency() -> List[str]:
    """
    Проверяет, что все алиасы присутствуют в KNOWN_TAGS.
    Возвращает список предупреждений.
    """
    from src.config_types import KNOWN_TAGS
    warnings = []
    for alias in KNOWN_FEATURE_ALIASES:
        if alias not in KNOWN_TAGS:
            warnings.append(f"Alias '{alias}' not found in KNOWN_TAGS.")
    return warnings