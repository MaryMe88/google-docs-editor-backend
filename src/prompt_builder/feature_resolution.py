# src/prompt_builder/feature_resolution.py
"""
Разрешение фич (feature flags) и объяснимость (explainability).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Set

from src.config_types import (
    DomainConfig,
    FeatureResolutionResult,
    IntentConfig,
    KnowledgeLevel,
    OverlayConfig,
    get_primary_tags_for_category,
)
from src.reason_codes import ReasonCode
from src.registry import CANONICAL_FEATURE_ALIASES, get_features_from_tags
from src.shared_contracts import ALLOWED_OVERLAYS

from .normalization import (
    normalize_intent,
    normalize_overlays,
    _is_incompatible_intent,
    _is_incompatible_overlay,
    _normalize_overlay_ref,
)

logger = logging.getLogger(__name__)

# Создаём карту тег -> фича
_TAG_TO_FEATURE = {alias: feature for feature, aliases in CANONICAL_FEATURE_ALIASES.items() for alias in aliases}


# ---------------------------------------------------------------------------
# Вспомогательные функции для explainability
# ---------------------------------------------------------------------------
def _add_activation_reason(
    result: FeatureResolutionResult,
    feature: str,
    reason: str,
) -> None:
    if feature not in result.activation_reasons:
        result.activation_reasons[feature] = []
    result.activation_reasons[feature].append(reason)
    if feature not in result.activated_features:
        result.activated_features.append(feature)


def _add_suppression_reason(
    result: FeatureResolutionResult,
    feature: str,
    reason: str,
) -> None:
    if feature not in result.suppression_reasons:
        result.suppression_reasons[feature] = []
    result.suppression_reasons[feature].append(reason)
    if feature not in result.suppressed_features:
        result.suppressed_features.append(feature)


def _add_recognized_alias(
    result: FeatureResolutionResult,
    feature: str,
    alias: str,
) -> None:
    if feature not in result.recognized_aliases:
        result.recognized_aliases[feature] = []
    if alias not in result.recognized_aliases[feature]:
        result.recognized_aliases[feature].append(alias)


def _add_ignored_unknown(
    result: FeatureResolutionResult,
    value: str,
) -> None:
    if value not in result.ignored_unknown_values:
        result.ignored_unknown_values.append(value)


def _build_overlay_slug_map(
    overlays: Sequence[str],
    overlay_configs: Sequence[OverlayConfig],
) -> Dict[str, OverlayConfig]:
    """Сопоставляет слаг оверлея (как в config/overlays/<slug>.json) с его конфигом."""
    mapping: Dict[str, OverlayConfig] = {}
    for slug, cfg in zip(overlays, overlay_configs):
        mapping[slug] = cfg
    for cfg in overlay_configs:
        mapping.setdefault(cfg.name, cfg)
    return mapping


# ---------------------------------------------------------------------------
# Главная функция разрешения фич
# ---------------------------------------------------------------------------
def resolve_prompt_features(
    domain: str,
    intent: Optional[str],
    overlays: Sequence[str],
    domain_config: DomainConfig,
    intent_config: Optional[IntentConfig],
    overlay_configs: List[OverlayConfig],
    knowledge_level: Optional[KnowledgeLevel] = None,
) -> Dict[str, Any]:
    """
    Единый канонический источник feature flags с explainability.
    Возвращает dict с полями, включая диагностические.
    """
    # 1. Нормализация
    norm_intent = normalize_intent(intent)
    known_overlay_names = {
        cfg.name.lower().strip()
        for cfg in overlay_configs
        if isinstance(getattr(cfg, "name", None), str) and cfg.name.strip()
    }
    norm_overlays = normalize_overlays(
        overlays,
        allowed_overlays=set(ALLOWED_OVERLAYS) | known_overlay_names,
    )

    effective_intent = norm_intent
    effective_overlays = list(norm_overlays)
    suppressed_layers = []
    warnings = []

    # 2. Создаём результат с explainability
    result = FeatureResolutionResult(
        tags=[],
        effective_intent=effective_intent,
        effective_overlays=effective_overlays,
        suppressed_layers=suppressed_layers,
        warnings=warnings,
        storytelling_enabled=False,
        marketing_enabled=False,
        antiai_enabled=False,
        rhetoric_enabled=False,
        nkrj_enabled=False,
        editorial_enabled=False,
        activated_features=[],
        suppressed_features=[],
        activation_reasons={},
        suppression_reasons={},
        recognized_aliases={},
        ignored_unknown_values=[],
    )

    # Фиксируем unknown intent
    if intent and not norm_intent:
        _add_ignored_unknown(result, intent)

    # Фиксируем unknown overlays
    for ov in overlays:
        if ov.lower().strip() not in ALLOWED_OVERLAYS:
            _add_ignored_unknown(result, ov)

    # 3. Базовые теги
    tags = [domain]
    if effective_intent:
        tags.append(effective_intent)
    tags.extend(effective_overlays)

    # 4. Проверка несовместимости интента с доменом
    if _is_incompatible_intent(effective_intent, domain_config.incompatible_intents):
        suppressed_layers.append(f"intent '{effective_intent}' suppressed by domain '{domain}'")
        warnings.append(f"Intent '{effective_intent}' incompatible with domain '{domain}', ignoring.")
        _add_suppression_reason(result, "intent", ReasonCode.SUPPRESSED_BY_DOMAIN_INCOMPATIBLE_INTENT)
        suppressed_intent = effective_intent
        effective_intent = None
        tags = [t for t in tags if t != suppressed_intent]

    # 5. Проверка несовместимости оверлеев с доменом
    for overlay in list(effective_overlays):
        if _is_incompatible_overlay(overlay, domain_config.incompatible_overlays):
            effective_overlays.remove(overlay)
            suppressed_layers.append(f"overlay '{overlay}' suppressed by domain '{domain}'")
            warnings.append(f"Overlay '{overlay}' incompatible with domain '{domain}', removed.")
            _add_suppression_reason(result, f"overlay:{overlay}", ReasonCode.SUPPRESSED_BY_DOMAIN_INCOMPATIBLE_OVERLAY)
            tags = [t for t in tags if t != overlay]

    # ======================================================================
    # 5.5. Явные suppresses между оверлеями
    # ======================================================================
    overlay_map = _build_overlay_slug_map(effective_overlays, overlay_configs)
    suppressed_by_overlay: Set[str] = set()

    for ov in list(effective_overlays):
        cfg = overlay_map.get(ov)
        if not cfg or not cfg.suppresses:
            continue
        for target in cfg.suppresses:
            target_name = _normalize_overlay_ref(target)
            if target_name in effective_overlays and target_name != ov:
                suppressed_by_overlay.add(target_name)
                suppressed_layers.append(
                    f"overlay '{target_name}' suppressed by overlay '{ov}' (explicit suppress)"
                )
                warnings.append(
                    f"Overlay '{target_name}' explicitly suppressed by '{ov}'."
                )
                _add_suppression_reason(
                    result,
                    f"overlay:{target_name}",
                    ReasonCode.SUPPRESSED_BY_OVERLAY_RULE,
                )
                tags = [t for t in tags if t != target_name]

    if suppressed_by_overlay:
        effective_overlays = [
            ov for ov in effective_overlays if ov not in suppressed_by_overlay
        ]
    # ======================================================================

    # ======================================================================
    # 6. Конфликты между оверлеями
    # ======================================================================
    overlay_map = _build_overlay_slug_map(effective_overlays, overlay_configs)
    conflicts_to_resolve = []
    for ov in effective_overlays:
        cfg = overlay_map.get(ov)
        if cfg and cfg.conflicts_with:
            for conflict in cfg.conflicts_with:
                conflict_name = _normalize_overlay_ref(conflict)
                if conflict_name in effective_overlays and conflict_name != ov:
                    conflicts_to_resolve.append((ov, conflict_name))

    for ov, conflict in conflicts_to_resolve:
        if ov not in effective_overlays or conflict not in effective_overlays:
            continue

        cfg_ov = overlay_map.get(ov)
        cfg_conflict = overlay_map.get(conflict)

        if cfg_ov is None or cfg_conflict is None:
            continue

        if any(_normalize_overlay_ref(s) == conflict for s in cfg_ov.suppresses):
            effective_overlays.remove(conflict)
            suppressed_layers.append(f"overlay '{conflict}' suppressed by overlay '{ov}' (explicit suppress)")
            warnings.append(f"Overlay '{conflict}' explicitly suppressed by '{ov}'.")
            _add_suppression_reason(result, f"overlay:{conflict}", ReasonCode.SUPPRESSED_BY_OVERLAY_RULE)
            tags = [t for t in tags if t != conflict]
            continue
        if any(_normalize_overlay_ref(s) == ov for s in cfg_conflict.suppresses):
            effective_overlays.remove(ov)
            suppressed_layers.append(f"overlay '{ov}' suppressed by overlay '{conflict}' (explicit suppress)")
            warnings.append(f"Overlay '{ov}' explicitly suppressed by '{conflict}'.")
            _add_suppression_reason(result, f"overlay:{ov}", ReasonCode.SUPPRESSED_BY_OVERLAY_RULE)
            tags = [t for t in tags if t != ov]
            continue

        if cfg_ov.priority > cfg_conflict.priority:
            effective_overlays.remove(conflict)
            suppressed_layers.append(f"overlay '{conflict}' suppressed due to conflict with '{ov}' (higher priority)")
            warnings.append(f"Overlay conflict: '{conflict}' removed (priority {cfg_conflict.priority}) < '{ov}' (priority {cfg_ov.priority}).")
            _add_suppression_reason(result, f"overlay:{conflict}", ReasonCode.SUPPRESSED_BY_OVERLAY_CONFLICT)
            tags = [t for t in tags if t != conflict]
        elif cfg_conflict.priority > cfg_ov.priority:
            effective_overlays.remove(ov)
            suppressed_layers.append(f"overlay '{ov}' suppressed due to conflict with '{conflict}' (higher priority)")
            warnings.append(f"Overlay conflict: '{ov}' removed (priority {cfg_ov.priority}) < '{conflict}' (priority {cfg_conflict.priority}).")
            _add_suppression_reason(result, f"overlay:{ov}", ReasonCode.SUPPRESSED_BY_OVERLAY_CONFLICT)
            tags = [t for t in tags if t != ov]
        else:
            # SEC-патч 3.2: Fallback вместо raise ValueError при равном priority.
            # Детерминированно побеждает оверлей, который встречается раньше в списке.
            if effective_overlays.index(ov) < effective_overlays.index(conflict):
                effective_overlays.remove(conflict)
                suppressed_layers.append(
                    f"overlay '{conflict}' suppressed due to conflict with '{ov}' (equal priority, deterministic fallback)"
                )
                warnings.append(
                    f"Overlay conflict: '{conflict}' removed (equal priority, fallback)."
                )
                _add_suppression_reason(result, f"overlay:{conflict}", ReasonCode.SUPPRESSED_BY_OVERLAY_CONFLICT)
                tags = [t for t in tags if t != conflict]
            else:
                effective_overlays.remove(ov)
                suppressed_layers.append(
                    f"overlay '{ov}' suppressed due to conflict with '{conflict}' (equal priority, deterministic fallback)"
                )
                warnings.append(
                    f"Overlay conflict: '{ov}' removed (equal priority, fallback)."
                )
                _add_suppression_reason(result, f"overlay:{ov}", ReasonCode.SUPPRESSED_BY_OVERLAY_CONFLICT)
                tags = [t for t in tags if t != ov]
    # ======================================================================

    # 7. Получаем фичи из тегов
    all_tags = [domain]
    if effective_intent:
        all_tags.append(effective_intent)
    for overlay in effective_overlays:
        all_tags.append(overlay)
        all_tags.extend(get_primary_tags_for_category("overlays", overlay))
    features = get_features_from_tags(all_tags)

    # 8. Базовые флаги с explainability
    # Storytelling
    storytelling_recognized = any(
        tag in _TAG_TO_FEATURE and _TAG_TO_FEATURE[tag] == "storytelling"
        for tag in all_tags
    )
    if domain_config.allow_storytelling and "storytelling" in features:
        result.storytelling_enabled = True
        _add_activation_reason(result, "storytelling", ReasonCode.DOMAIN_ALLOWS_STORYTELLING)
        if storytelling_recognized:
            _add_activation_reason(result, "storytelling", ReasonCode.RECOGNIZED_STORYTELLING_ALIAS)
        for tag in all_tags:
            if tag in _TAG_TO_FEATURE and _TAG_TO_FEATURE[tag] == "storytelling":
                _add_recognized_alias(result, "storytelling", tag)
    elif "storytelling" in features:
        _add_suppression_reason(result, "storytelling", ReasonCode.DOMAIN_DENIES_STORYTELLING)
    elif not storytelling_recognized:
        _add_suppression_reason(result, "storytelling", ReasonCode.NO_RECOGNIZED_ALIAS)

    # Marketing
    marketing_recognized = any(
        tag in _TAG_TO_FEATURE and _TAG_TO_FEATURE[tag] == "marketing"
        for tag in all_tags
    )
    if domain_config.allow_marketing and "marketing" in features:
        result.marketing_enabled = True
        _add_activation_reason(result, "marketing", ReasonCode.DOMAIN_ALLOWS_MARKETING)
        if marketing_recognized:
            _add_activation_reason(result, "marketing", ReasonCode.RECOGNIZED_MARKETING_ALIAS)
        for tag in all_tags:
            if tag in _TAG_TO_FEATURE and _TAG_TO_FEATURE[tag] == "marketing":
                _add_recognized_alias(result, "marketing", tag)
    elif "marketing" in features:
        _add_suppression_reason(result, "marketing", ReasonCode.DOMAIN_DENIES_MARKETING)
    elif not marketing_recognized:
        _add_suppression_reason(result, "marketing", ReasonCode.NO_RECOGNIZED_ALIAS)

    # anti-ai
    antiai_recognized = any(
        tag in _TAG_TO_FEATURE and _TAG_TO_FEATURE[tag] == "antiai"
        for tag in all_tags
    )
    if "antiai" in features:
        result.antiai_enabled = True
        _add_activation_reason(result, "antiai", ReasonCode.RECOGNIZED_ANTIAI_ALIAS)
        for tag in all_tags:
            if tag in _TAG_TO_FEATURE and _TAG_TO_FEATURE[tag] == "antiai":
                _add_recognized_alias(result, "antiai", tag)
    elif not antiai_recognized:
        _add_suppression_reason(result, "antiai", ReasonCode.NO_RECOGNIZED_ALIAS)

    # rhetoric
    rhetoric_recognized = any(
        tag in _TAG_TO_FEATURE and _TAG_TO_FEATURE[tag] == "rhetoric"
        for tag in all_tags
    )
    if "rhetoric" in features:
        result.rhetoric_enabled = True
        _add_activation_reason(result, "rhetoric", ReasonCode.RECOGNIZED_RHETORIC_ALIAS)
        for tag in all_tags:
            if tag in _TAG_TO_FEATURE and _TAG_TO_FEATURE[tag] == "rhetoric":
                _add_recognized_alias(result, "rhetoric", tag)
    elif not rhetoric_recognized:
        _add_suppression_reason(result, "rhetoric", ReasonCode.NO_RECOGNIZED_ALIAS)

    # nkrj
    nkrj_recognized = any(
        tag in _TAG_TO_FEATURE and _TAG_TO_FEATURE[tag] == "nkrj"
        for tag in all_tags
    )
    if "nkrj" in features:
        result.nkrj_enabled = True
        _add_activation_reason(result, "nkrj", ReasonCode.RECOGNIZED_NKRJ_ALIAS)
        for tag in all_tags:
            if tag in _TAG_TO_FEATURE and _TAG_TO_FEATURE[tag] == "nkrj":
                _add_recognized_alias(result, "nkrj", tag)
    elif not nkrj_recognized:
        _add_suppression_reason(result, "nkrj", ReasonCode.NO_RECOGNIZED_ALIAS)

    # editorial
    editorial_recognized = any(
        tag in _TAG_TO_FEATURE and _TAG_TO_FEATURE[tag] == "editorial"
        for tag in all_tags
    )
    if "editorial" in features:
        result.editorial_enabled = True
        _add_activation_reason(result, "editorial", ReasonCode.RECOGNIZED_EDITORIAL_ALIAS)
        for tag in all_tags:
            if tag in _TAG_TO_FEATURE and _TAG_TO_FEATURE[tag] == "editorial":
                _add_recognized_alias(result, "editorial", tag)
    elif not editorial_recognized:
        _add_suppression_reason(result, "editorial", ReasonCode.NO_RECOGNIZED_ALIAS)

    # 8.5 Принудительное включение при FULL уровне
    if knowledge_level == KnowledgeLevel.FULL:
        if domain_config.allow_storytelling and not result.storytelling_enabled:
            result.storytelling_enabled = True
            _add_activation_reason(result, "storytelling", ReasonCode.FULL_LEVEL_OVERRIDE)
        if domain_config.allow_marketing and not result.marketing_enabled:
            result.marketing_enabled = True
            _add_activation_reason(result, "marketing", ReasonCode.FULL_LEVEL_OVERRIDE)
        if not result.editorial_enabled:
            result.editorial_enabled = True
            _add_activation_reason(result, "editorial", ReasonCode.FULL_LEVEL_OVERRIDE)
        if not result.rhetoric_enabled:
            result.rhetoric_enabled = True
            _add_activation_reason(result, "rhetoric", ReasonCode.FULL_LEVEL_OVERRIDE)
        if not result.nkrj_enabled:
            result.nkrj_enabled = True
            _add_activation_reason(result, "nkrj", ReasonCode.FULL_LEVEL_OVERRIDE)

    # 9. Применяем suppress правила (первая итерация)
    if "storytelling" in domain_config.suppresses:
        result.storytelling_enabled = False
        suppressed_layers.append("storytelling suppressed by domain")
        warnings.append("Storytelling disabled by domain 'suppresses' rule.")
        _add_suppression_reason(result, "storytelling", ReasonCode.SUPPRESSED_BY_DOMAIN_RULE)
    if "marketing" in domain_config.suppresses or "marketingpush" in domain_config.suppresses:
        result.marketing_enabled = False
        suppressed_layers.append("marketing suppressed by domain")
        warnings.append("Marketing disabled by domain 'suppresses' rule.")
        _add_suppression_reason(result, "marketing", ReasonCode.SUPPRESSED_BY_DOMAIN_RULE)

    for cfg in overlay_configs:
        if cfg.name in effective_overlays:
            if "storytelling" in cfg.suppresses:
                result.storytelling_enabled = False
                suppressed_layers.append(f"storytelling suppressed by overlay '{cfg.name}'")
                warnings.append(f"Storytelling disabled by overlay '{cfg.name}' suppress rule.")
                _add_suppression_reason(result, "storytelling", ReasonCode.SUPPRESSED_BY_OVERLAY_RULE)
            if "marketing" in cfg.suppresses or "marketingpush" in cfg.suppresses:
                result.marketing_enabled = False
                suppressed_layers.append(f"marketing suppressed by overlay '{cfg.name}'")
                warnings.append(f"Marketing disabled by overlay '{cfg.name}' suppress rule.")
                _add_suppression_reason(result, "marketing", ReasonCode.SUPPRESSED_BY_OVERLAY_RULE)

    # 9.5 Применяем suppress правила интента
    if intent_config:
        for suppressed in intent_config.suppresses:
            if suppressed == "storytelling":
                result.storytelling_enabled = False
                if "storytelling" not in result.suppressed_features:
                    result.suppressed_features.append("storytelling")
                _add_suppression_reason(result, "storytelling", ReasonCode.SUPPRESSED_BY_INTENT_RULE)
            elif suppressed in ("marketing", "marketingpush"):
                result.marketing_enabled = False
                if "marketing" not in result.suppressed_features:
                    result.suppressed_features.append("marketing")
                _add_suppression_reason(result, "marketing", ReasonCode.SUPPRESSED_BY_INTENT_RULE)
            elif suppressed == "antiai":
                result.antiai_enabled = False
                if "antiai" not in result.suppressed_features:
                    result.suppressed_features.append("antiai")
                _add_suppression_reason(result, "antiai", ReasonCode.SUPPRESSED_BY_INTENT_RULE)
            elif suppressed == "rhetoric":
                result.rhetoric_enabled = False
                if "rhetoric" not in result.suppressed_features:
                    result.suppressed_features.append("rhetoric")
                _add_suppression_reason(result, "rhetoric", ReasonCode.SUPPRESSED_BY_INTENT_RULE)
            elif suppressed == "nkrj":
                result.nkrj_enabled = False
                if "nkrj" not in result.suppressed_features:
                    result.suppressed_features.append("nkrj")
                _add_suppression_reason(result, "nkrj", ReasonCode.SUPPRESSED_BY_INTENT_RULE)
            elif suppressed == "editorial":
                result.editorial_enabled = False
                if "editorial" not in result.suppressed_features:
                    result.suppressed_features.append("editorial")
                _add_suppression_reason(result, "editorial", ReasonCode.SUPPRESSED_BY_INTENT_RULE)

    # 10. Итоговые теги: уникальные
    final_tags = list(dict.fromkeys(tags))
    result.tags = final_tags
    result.effective_intent = effective_intent
    result.effective_overlays = effective_overlays
    result.suppressed_layers = suppressed_layers
    result.warnings = warnings

    return result.to_dict()