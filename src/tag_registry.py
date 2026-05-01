from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Set

_ALIAS_MAP = {
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



def normalize_tags(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen: Set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        normalized = normalize_tag(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result



def build_known_tags(mapping: Dict[str, Any]) -> Set[str]:
    known: Set[str] = set()

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
