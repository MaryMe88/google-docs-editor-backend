# src/reason_codes.py
"""
Канонические reason codes для explainability feature resolution и prompt assembly.
Эти коды стабильны и используются для тестирования и диагностики.
"""

from enum import Enum
from typing import Final, Set


class ReasonCode(str, Enum):
    """
    Перечисление всех возможных причин (reason codes) для принятия решений
    в feature resolution и сборке промпта.
    Используются для объяснения, почему фича включена/выключена или блок добавлен/пропущен.
    """

    # ------------------------------------------------------------------
    # Feature activation reasons (фича включена)
    # ------------------------------------------------------------------
    DOMAIN_ALLOWS_STORYTELLING = "domain_allows_storytelling"
    DOMAIN_ALLOWS_MARKETING = "domain_allows_marketing"
    RECOGNIZED_STORYTELLING_ALIAS = "recognized_storytelling_alias"
    RECOGNIZED_MARKETING_ALIAS = "recognized_marketing_alias"
    RECOGNIZED_ANTIAI_ALIAS = "recognized_antiai_alias"
    RECOGNIZED_RHETORIC_ALIAS = "recognized_rhetoric_alias"
    RECOGNIZED_NKRJ_ALIAS = "recognized_nkrj_alias"
    RECOGNIZED_EDITORIAL_ALIAS = "recognized_editorial_alias"

    # ------------------------------------------------------------------
    # Feature suppression reasons (фича выключена)
    # ------------------------------------------------------------------
    SUPPRESSED_BY_DOMAIN_PRIORITY = "suppressed_by_domain_priority"
    SUPPRESSED_BY_DOMAIN_RULE = "suppressed_by_domain_rule"
    SUPPRESSED_BY_OVERLAY_RULE = "suppressed_by_overlay_rule"
    SUPPRESSED_BY_OVERLAY_CONFLICT = "suppressed_by_overlay_conflict"
    SUPPRESSED_BY_DOMAIN_INCOMPATIBLE_INTENT = "suppressed_by_domain_incompatible_intent"
    SUPPRESSED_BY_DOMAIN_INCOMPATIBLE_OVERLAY = "suppressed_by_domain_incompatible_overlay"
    DOMAIN_DENIES_STORYTELLING = "domain_denies_storytelling"
    DOMAIN_DENIES_MARKETING = "domain_denies_marketing"
    NO_RECOGNIZED_ALIAS = "no_recognized_alias"

    # ------------------------------------------------------------------
    # Intent/Overlay processing reasons
    # ------------------------------------------------------------------
    INTENT_NORMALIZED = "intent_normalized"
    INTENT_NEUTRAL = "intent_neutral"
    INTENT_IGNORED_UNKNOWN = "ignored_unknown_intent"
    OVERLAY_IGNORED_UNKNOWN = "ignored_unknown_overlay"
    OVERLAY_NORMALIZED = "overlay_normalized"

    # ------------------------------------------------------------------
    # Assembly block reasons (почему блок включён или пропущен)
    # ------------------------------------------------------------------
    BLOCK_ELIGIBLE = "block_eligible"
    BLOCK_INELIGIBLE_FEATURE_DISABLED = "block_ineligible_feature_disabled"
    BLOCK_INELIGIBLE_BUDGET_DISABLED = "block_ineligible_budget_disabled"
    BLOCK_INELIGIBLE_KB_EMPTY = "block_ineligible_kb_empty"
    BLOCK_INELIGIBLE_KB_UNAVAILABLE = "block_ineligible_kb_unavailable"
    BLOCK_INCLUDED = "block_included"
    BLOCK_SKIPPED = "block_skipped"
    BLOCK_EMPTY_AFTER_BUILD = "block_empty_after_build"

    # ------------------------------------------------------------------
    # Validation / invariant reasons
    # ------------------------------------------------------------------
    VALIDATION_FEATURE_WITHOUT_REASON = "validation_feature_without_reason"
    VALIDATION_SUPPRESSED_WITHOUT_REASON = "validation_suppressed_without_reason"


# Множество всех activation reasons (для быстрой проверки)
ACTIVATION_REASONS: Final[Set[str]] = {
    ReasonCode.DOMAIN_ALLOWS_STORYTELLING,
    ReasonCode.DOMAIN_ALLOWS_MARKETING,
    ReasonCode.RECOGNIZED_STORYTELLING_ALIAS,
    ReasonCode.RECOGNIZED_MARKETING_ALIAS,
    ReasonCode.RECOGNIZED_ANTIAI_ALIAS,
    ReasonCode.RECOGNIZED_RHETORIC_ALIAS,
    ReasonCode.RECOGNIZED_NKRJ_ALIAS,
    ReasonCode.RECOGNIZED_EDITORIAL_ALIAS,
}

# Множество всех suppression reasons
SUPPRESSION_REASONS: Final[Set[str]] = {
    ReasonCode.SUPPRESSED_BY_DOMAIN_PRIORITY,
    ReasonCode.SUPPRESSED_BY_DOMAIN_RULE,
    ReasonCode.SUPPRESSED_BY_OVERLAY_RULE,
    ReasonCode.SUPPRESSED_BY_OVERLAY_CONFLICT,
    ReasonCode.SUPPRESSED_BY_DOMAIN_INCOMPATIBLE_INTENT,
    ReasonCode.SUPPRESSED_BY_DOMAIN_INCOMPATIBLE_OVERLAY,
    ReasonCode.DOMAIN_DENIES_STORYTELLING,
    ReasonCode.DOMAIN_DENIES_MARKETING,
    ReasonCode.NO_RECOGNIZED_ALIAS,
}