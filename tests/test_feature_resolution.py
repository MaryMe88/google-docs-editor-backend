# tests/test_feature_resolution.py
"""Unit-тесты для feature_resolution.py (новые хелперы)."""

from __future__ import annotations

import pytest

from src.config_types import (
    DomainConfig,
    FeatureResolutionResult,
    OverlayConfig,
    KnowledgeLevel,
)
from src.prompt_builder.feature_resolution import (
    _prepare_resolution_context,
    _apply_domain_incompatibility,
    _apply_overlay_suppressions,
    _resolve_overlay_conflicts,
    _apply_full_level_overrides,
    _apply_suppress_rules,
    _activate_features_from_tags,
    _activate_storytelling,
    _activate_marketing,
    _activate_antiai,
    _activate_rhetoric,
    _activate_nkrj,
    _activate_editorial,
)
from src.reason_codes import ReasonCode


def test_prepare_resolution_context_unknown_intent():
    """
    Проверяет, что неизвестный intent нормализуется в None
    и попадает в ignored_unknown_values.
    """
    result, effective_intent, effective_overlays, tags, _, _ = (
        _prepare_resolution_context("blog", "unknown", [], [])
    )
    assert effective_intent is None
    assert "unknown" in result.ignored_unknown_values


def test_apply_domain_incompatibility_removes_intent():
    """Проверяет, что несовместимый интент удаляется и добавляется suppression_reason."""
    domain_config = DomainConfig(
        name="blog",
        system_rules="",
        tone="neutral",
        allow_storytelling=False,
        allow_marketing=False,
        incompatible_intents=["analytical"],
    )
    result = FeatureResolutionResult(
        tags=[],
        effective_intent="",
        effective_overlays=[],
        suppressed_layers=[],
        warnings=[],
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
    effective_intent = "analytical"
    effective_overlays = []
    tags = ["blog", "analytical"]
    suppressed_layers = []
    warnings = []

    new_intent, new_overlays, new_tags = _apply_domain_incompatibility(
        result, "blog", domain_config,
        effective_intent, effective_overlays, tags,
        suppressed_layers, warnings,
    )
    assert new_intent is None
    assert "analytical" not in new_tags
    assert len(result.suppressed_features) == 1
    assert ReasonCode.SUPPRESSED_BY_DOMAIN_INCOMPATIBLE_INTENT in result.suppression_reasons["intent"]


def test_resolve_overlay_conflicts_higher_priority_wins():
    """При конфликте оверлеев побеждает тот, у кого выше priority."""
    ov1 = OverlayConfig(
        name="infostyle",
        priority=10,
        instructions=[],
        suppresses=[],
        conflicts_with=["editorial"],
    )
    ov2 = OverlayConfig(
        name="editorial",
        priority=5,
        instructions=[],
        suppresses=[],
        conflicts_with=["infostyle"],
    )
    effective_overlays = ["infostyle", "editorial"]
    tags = ["infostyle", "editorial"]
    result = FeatureResolutionResult(
        tags=[],
        effective_intent="",
        effective_overlays=[],
        suppressed_layers=[],
        warnings=[],
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
    new_overlays, new_tags = _resolve_overlay_conflicts(
        result, effective_overlays, [ov1, ov2], tags,
        result.suppressed_layers, result.warnings,
    )
    assert "editorial" not in new_overlays
    assert "editorial" not in new_tags
    assert "infostyle" in new_overlays


def test_apply_full_level_overrides_enables_features():
    """При FULL уровне все фичи, разрешённые доменом, принудительно включаются."""
    domain_config = DomainConfig(
        name="blog",
        system_rules="",
        tone="neutral",
        allow_storytelling=True,
        allow_marketing=True,
    )
    result = FeatureResolutionResult(
        tags=[],
        effective_intent="",
        effective_overlays=[],
        suppressed_layers=[],
        warnings=[],
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
    _apply_full_level_overrides(result, KnowledgeLevel.FULL, domain_config)
    assert result.storytelling_enabled is True
    assert result.marketing_enabled is True
    assert result.editorial_enabled is True
    assert result.rhetoric_enabled is True
    assert result.nkrj_enabled is True


def test_apply_suppress_rules_domain_suppresses_storytelling():
    """Проверяет, что suppress из домена выключает storytelling."""
    domain_config = DomainConfig(
        name="blog",
        system_rules="",
        tone="neutral",
        allow_storytelling=True,
        allow_marketing=True,
        suppresses=["storytelling"],
    )
    result = FeatureResolutionResult(
        tags=[],
        effective_intent="",
        effective_overlays=[],
        suppressed_layers=[],
        warnings=[],
        storytelling_enabled=True,
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
    _apply_suppress_rules(
        result, domain_config, None, [], [],
        result.suppressed_layers, result.warnings,
    )
    assert result.storytelling_enabled is False
    assert "storytelling" in result.suppressed_features


# ---------------------------------------------------------------------------
# Тесты для функций активации отдельных фич
# ---------------------------------------------------------------------------

def test_activate_storytelling_enabled():
    domain_config = DomainConfig(
        name="blog", system_rules="", tone="neutral",
        allow_storytelling=True, allow_marketing=False,
    )
    result = FeatureResolutionResult(
        tags=[],
        effective_intent="",
        effective_overlays=[],
        suppressed_layers=[],
        warnings=[],
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
    all_tags = ["blog", "storytelling"]
    _activate_storytelling(result, all_tags, domain_config)
    assert result.storytelling_enabled is True
    assert "storytelling" in result.activation_reasons
    assert "storytelling" in result.recognized_aliases


def test_activate_marketing_disabled_by_domain():
    domain_config = DomainConfig(
        name="blog", system_rules="", tone="neutral",
        allow_storytelling=False, allow_marketing=False,
    )
    result = FeatureResolutionResult(
        tags=[],
        effective_intent="",
        effective_overlays=[],
        suppressed_layers=[],
        warnings=[],
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
    all_tags = ["blog", "marketing"]
    _activate_marketing(result, all_tags, domain_config)
    assert result.marketing_enabled is False
    assert ReasonCode.DOMAIN_DENIES_MARKETING in result.suppression_reasons.get("marketing", [])


def test_activate_antiai_enabled():
    result = FeatureResolutionResult(
        tags=[],
        effective_intent="",
        effective_overlays=[],
        suppressed_layers=[],
        warnings=[],
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
    all_tags = ["deai"]
    _activate_antiai(result, all_tags)
    assert result.antiai_enabled is True
    assert "antiai" in result.activation_reasons


def test_activate_rhetoric_no_tag():
    result = FeatureResolutionResult(
        tags=[],
        effective_intent="",
        effective_overlays=[],
        suppressed_layers=[],
        warnings=[],
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
    all_tags = ["blog"]
    _activate_rhetoric(result, all_tags)
    assert result.rhetoric_enabled is False
    assert ReasonCode.NO_RECOGNIZED_ALIAS in result.suppression_reasons.get("rhetoric", [])


def test_activate_nkrj_enabled():
    result = FeatureResolutionResult(
        tags=[],
        effective_intent="",
        effective_overlays=[],
        suppressed_layers=[],
        warnings=[],
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
    all_tags = ["nkrj"]
    _activate_nkrj(result, all_tags)
    assert result.nkrj_enabled is True


def test_activate_editorial_enabled():
    result = FeatureResolutionResult(
        tags=[],
        effective_intent="",
        effective_overlays=[],
        suppressed_layers=[],
        warnings=[],
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
    all_tags = ["editorial"]
    _activate_editorial(result, all_tags)
    assert result.editorial_enabled is True


# ---------------------------------------------------------------------------
# Тесты для _apply_overlay_suppressions
# ---------------------------------------------------------------------------

def test_apply_overlay_suppressions_removes_suppressed_overlay():
    """Проверяет, что явный suppress удаляет целевой оверлей."""
    ov1 = OverlayConfig(
        name="landing",
        instructions=[],
        priority=70,
        suppresses=["pressrelease"],
        conflicts_with=[],
    )
    ov2 = OverlayConfig(
        name="pressrelease",
        instructions=[],
        priority=70,
        suppresses=[],
        conflicts_with=[],
    )
    result = FeatureResolutionResult(
        tags=[],
        effective_intent="",
        effective_overlays=[],
        suppressed_layers=[],
        warnings=[],
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
    effective_overlays = ["landing", "pressrelease"]
    tags = ["landing", "pressrelease"]
    suppressed_layers = []
    warnings = []

    new_overlays, new_tags = _apply_overlay_suppressions(
        result, effective_overlays, [ov1, ov2], tags,
        suppressed_layers, warnings,
    )
    assert "pressrelease" not in new_overlays
    assert "pressrelease" not in new_tags
    assert "landing" in new_overlays
    assert any("pressrelease" in layer for layer in suppressed_layers)


def test_apply_overlay_suppressions_no_suppress_if_target_not_present():
    """Проверяет, что suppress не применяется, если целевой оверлей отсутствует."""
    ov1 = OverlayConfig(
        name="landing",
        instructions=[],
        priority=70,
        suppresses=["pressrelease"],
        conflicts_with=[],
    )
    result = FeatureResolutionResult(
        tags=[],
        effective_intent="",
        effective_overlays=[],
        suppressed_layers=[],
        warnings=[],
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
    effective_overlays = ["landing"]
    tags = ["landing"]
    new_overlays, new_tags = _apply_overlay_suppressions(
        result, effective_overlays, [ov1], tags,
        [], [],
    )
    assert "landing" in new_overlays
    assert len(new_overlays) == 1
    assert not result.suppression_reasons  # нет подавлений