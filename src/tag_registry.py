from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Set

# ---------------------------------------------------------------------------
# Алиас-карта: любое написание → каноническая форма.
#
# Правило именования оверлеев/доменов/интентов:
#   - Имена файлов в config/overlays/*.json используют snake_case.
#   - normalize_tag по умолчанию убирает подчёркивания ([^a-z0-9]+ → ""),
#     что ломает идемпотентность для snake_case-имён.
#   - Поэтому все snake_case-имена ОБЯЗАНЫ быть зарегистрированы здесь
#     явно, чтобы normalize_tag("finalcheck_full") == "finalcheck_full".
#
# При добавлении нового файла config/overlays/my_overlay.json:
#   1. Добавь строку  "my_overlay": "my_overlay"  в раздел OVERLAYS ниже.
#   2. Если у имени есть альтернативные написания — добавь их тоже.
# ---------------------------------------------------------------------------
_ALIAS_MAP: Dict[str, str] = {
    # ---- домены / теги ----
    "anti_ai":         "antiai",
    "anti-ai":         "antiai",
    "antiai":          "antiai",
    "de_ai":           "deai",
    "de-ai":           "deai",
    "deai":            "deai",
    "nora_gal":        "noragal",
    "nora-gal":        "noragal",
    "noragal":         "noragal",
    "marketing_push":  "marketingpush",
    "marketing-push":  "marketingpush",
    "marketingpush":   "marketingpush",
    "info_style":      "infostyle",
    "info-style":      "infostyle",
    "infostyle":       "infostyle",
    "final_check":     "finalcheck",
    "final-check":     "finalcheck",
    "finalcheck":      "finalcheck",
    "non_marketing":   "nonmarketing",
    "non-marketing":   "nonmarketing",
    "nonmarketing":    "nonmarketing",

    # ---- оверлеи (snake_case — каноническая форма совпадает с именем файла) ----
    # Эти записи гарантируют идемпотентность:
    #   normalize_tag("finalcheck_full") == "finalcheck_full"
    # Без них normalize_tag стрипает "_" и возвращает "finalcheckfull",
    # что не совпадает с именем файла → FileNotFoundError при загрузке.
    "finalcheck_full":   "finalcheck_full",
    "finalcheck-full":   "finalcheck_full",
    "finalcheckfull":    "finalcheck_full",

    "finalcheck_light":  "finalcheck_light",
    "finalcheck-light":  "finalcheck_light",
    "finalchecklight":   "finalcheck_light",

    "no_overwrite":      "no_overwrite",
    "no-overwrite":      "no_overwrite",
    "nooverwrite":       "no_overwrite",
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