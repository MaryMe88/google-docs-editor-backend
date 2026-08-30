# src/prompt_builder/normalization.py
"""
Модуль нормализации входных параметров: intent, overlays, а также
вспомогательные функции для работы с префиксами ссылок.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Set

from src.shared_contracts import ALLOWED_INTENTS, ALLOWED_OVERLAYS

logger = logging.getLogger(__name__)


def normalize_intent(intent: Optional[str]) -> Optional[str]:
    if intent is None or intent == "neutral":
        return None
    normalized = intent.lower().strip()
    if normalized not in ALLOWED_INTENTS:
        logger.warning(f"Unknown intent '{normalized}', treating as neutral.")
        return None
    return normalized


def normalize_overlays(
    overlays: Sequence[str],
    *,
    allowed_overlays: Optional[Set[str]] = None,
) -> List[str]:
    """
    Нормализует overlay-имена.

    allowed_overlays=None -> старое поведение (фильтр по ALLOWED_OVERLAYS).
    allowed_overlays={...} -> фильтр по расширенному множеству,
    используется внутри resolve_prompt_features, где overlay_configs
    уже переданы явно и являются доверенным источником.
    """
    effective_allowed = ALLOWED_OVERLAYS if allowed_overlays is None else allowed_overlays
    result: List[str] = []
    for ov in overlays:
        norm = ov.lower().strip()
        if not norm:
            continue
        if norm in effective_allowed:
            result.append(norm)
        else:
            logger.warning(f"Unknown overlay '{norm}', ignoring.")
    return result


def normalize_string_list(value: List[str]) -> List[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if isinstance(item, str):
            stripped = item.strip()
            if stripped:
                result.append(stripped.lower())
    return result


def _is_incompatible_intent(
    effective_intent: Optional[str],
    incompatible_intents: tuple,
) -> bool:
    """Проверяет, есть ли effective_intent в списке несовместимых интентов с учётом префикса intent:."""
    if not effective_intent:
        return False
    for item in incompatible_intents:
        if isinstance(item, str):
            if item.startswith("intent:"):
                if effective_intent == item[7:]:
                    return True
            else:
                if effective_intent == item:
                    return True
    return False


def _is_incompatible_overlay(overlay: str, incompatible_overlays: tuple) -> bool:
    """Проверяет, есть ли overlay в списке несовместимых оверлеев с учётом префикса overlay:."""
    for item in incompatible_overlays:
        if isinstance(item, str):
            if item.startswith("overlay:"):
                if overlay == item[8:]:
                    return True
            else:
                if overlay == item:
                    return True
    return False


def _normalize_overlay_ref(ref: str) -> str:
    """Убирает префикс 'overlay:' из ссылки, если он есть."""
    if ref.startswith("overlay:"):
        return ref[8:]
    return ref