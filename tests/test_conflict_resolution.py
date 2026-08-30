"""Тесты для разрешения конфликтов конфигурации (Итерация 5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.prompt_builder import PromptBuilder, resolve_prompt_features
from src.reason_codes import ReasonCode
from src.config_types import IntentConfig, OverlayConfig


@pytest.fixture
def builder() -> PromptBuilder:
    return PromptBuilder(config_path=Path("config"), kb_path=Path("knowledge_base"))


@pytest.fixture
def builder_with_mock(monkeypatch: pytest.MonkeyPatch) -> PromptBuilder:
    def mock_normalize_overlays(overlays, **kwargs):
        return list(overlays)
    monkeypatch.setattr("src.prompt_builder.normalize_overlays", mock_normalize_overlays)
    return PromptBuilder(config_path=Path("config"), kb_path=Path("knowledge_base"))


# ============================================================================
# Тесты из Итерации 3 (domain.incompatible_*)
# ============================================================================

def test_deai_removes_infostyle(builder: PromptBuilder) -> None:
    """Домен deai должен удалять оверлей infostyle."""
    domain_config = builder.get_domain_config("deai")
    overlay_configs = [builder.get_overlay_config("infostyle")]
    result = resolve_prompt_features(
        domain="deai",
        intent=None,
        overlays=["infostyle"],
        domain_config=domain_config,
        intent_config=None,
        overlay_configs=overlay_configs,
    )
    assert "infostyle" not in result["effective_overlays"]
    assert any("infostyle" in layer for layer in result["suppressed_layers"])
    suppression_reasons = result["suppression_reasons"]
    assert any(
        ReasonCode.SUPPRESSED_BY_DOMAIN_INCOMPATIBLE_OVERLAY in reasons
        for reasons in suppression_reasons.values()
    )


def test_nora_gal_removes_infostyle(builder: PromptBuilder) -> None:
    """Домен nora_gal должен удалять infostyle."""
    domain_config = builder.get_domain_config("nora_gal")
    overlay_configs = [builder.get_overlay_config("infostyle")]
    result = resolve_prompt_features(
        domain="nora_gal",
        intent=None,
        overlays=["infostyle"],
        domain_config=domain_config,
        intent_config=None,
        overlay_configs=overlay_configs,
    )
    assert "infostyle" not in result["effective_overlays"]


def test_nora_gal_soft_removes_infostyle(builder: PromptBuilder) -> None:
    """Домен nora_gal_soft должен удалять infostyle."""
    domain_config = builder.get_domain_config("nora_gal_soft")
    overlay_configs = [builder.get_overlay_config("infostyle")]
    result = resolve_prompt_features(
        domain="nora_gal_soft",
        intent=None,
        overlays=["infostyle"],
        domain_config=domain_config,
        intent_config=None,
        overlay_configs=overlay_configs,
    )
    assert "infostyle" not in result["effective_overlays"]


def test_fiction_removes_analytical_intent(builder: PromptBuilder) -> None:
    """Домен fiction должен удалять intent analytical."""
    domain_config = builder.get_domain_config("fiction")
    result = resolve_prompt_features(
        domain="fiction",
        intent="analytical",
        overlays=["base"],
        domain_config=domain_config,
        intent_config=None,
        overlay_configs=[builder.get_overlay_config("base")],
    )
    assert result["effective_intent"] is None
    assert any("analytical" in layer for layer in result["suppressed_layers"])
    suppression_reasons = result["suppression_reasons"]
    assert any(
        ReasonCode.SUPPRESSED_BY_DOMAIN_INCOMPATIBLE_INTENT in reasons
        for reasons in suppression_reasons.values()
    )


def test_fiction_allows_storytelling_intent(builder: PromptBuilder) -> None:
    """Домен fiction должен разрешать storytelling intent."""
    domain_config = builder.get_domain_config("fiction")
    result = resolve_prompt_features(
        domain="fiction",
        intent="storytelling",
        overlays=["base"],
        domain_config=domain_config,
        intent_config=None,
        overlay_configs=[builder.get_overlay_config("base")],
    )
    assert result["effective_intent"] == "storytelling"


def test_domain_incompatibility_does_not_affect_other_overlays(builder: PromptBuilder) -> None:
    """Проверка, что deai не удаляет другие оверлеи (например, base)."""
    domain_config = builder.get_domain_config("deai")
    overlay_configs = [builder.get_overlay_config("infostyle"), builder.get_overlay_config("base")]
    result = resolve_prompt_features(
        domain="deai",
        intent=None,
        overlays=["infostyle", "base"],
        domain_config=domain_config,
        intent_config=None,
        overlay_configs=overlay_configs,
    )
    assert "infostyle" not in result["effective_overlays"]
    assert "base" in result["effective_overlays"]


# ============================================================================
# Тесты из Итерации 4 (intent.suppresses)
# ============================================================================

def test_intent_suppresses_storytelling(builder: PromptBuilder) -> None:
    """Intent может подавлять storytelling через suppresses."""
    intent_config = IntentConfig(
        name="test_intent",
        instructions=(),
        priority=50,
        suppresses=("storytelling",),
        conflicts_with=(),
    )
    domain_config = builder.get_domain_config("blog")
    overlay_configs = [builder.get_overlay_config("base")]
    result = resolve_prompt_features(
        domain="blog",
        intent="test_intent",
        overlays=["base"],
        domain_config=domain_config,
        intent_config=intent_config,
        overlay_configs=overlay_configs,
    )
    assert result["storytelling_enabled"] is False
    suppression_reasons = result["suppression_reasons"]
    assert any(
        ReasonCode.SUPPRESSED_BY_INTENT_RULE in reasons
        for reasons in suppression_reasons.values()
    )


def test_intent_suppresses_marketing(builder: PromptBuilder) -> None:
    """Intent может подавлять marketing."""
    intent_config = IntentConfig(
        name="test_intent",
        instructions=(),
        priority=50,
        suppresses=("marketing",),
        conflicts_with=(),
    )
    domain_config = builder.get_domain_config("marketing")
    overlay_configs = [builder.get_overlay_config("base")]
    result = resolve_prompt_features(
        domain="marketing",
        intent="test_intent",
        overlays=["base"],
        domain_config=domain_config,
        intent_config=intent_config,
        overlay_configs=overlay_configs,
    )
    assert result["marketing_enabled"] is False
    suppression_reasons = result["suppression_reasons"]
    assert any(
        ReasonCode.SUPPRESSED_BY_INTENT_RULE in reasons
        for reasons in suppression_reasons.values()
    )


def test_intent_suppresses_overlay_feature(builder: PromptBuilder) -> None:
    """Проверяем, что suppresses интента не влияет на overlay (только на фичи)."""
    intent_config = IntentConfig(
        name="test_intent",
        instructions=(),
        priority=50,
        suppresses=("infostyle",),
        conflicts_with=(),
    )
    domain_config = builder.get_domain_config("blog")
    overlay_configs = [builder.get_overlay_config("infostyle")]
    result = resolve_prompt_features(
        domain="blog",
        intent="test_intent",
        overlays=["infostyle"],
        domain_config=domain_config,
        intent_config=intent_config,
        overlay_configs=overlay_configs,
    )
    assert "infostyle" in result["effective_overlays"]
    assert "infostyle" not in result["suppression_reasons"]


def test_intent_suppresses_multiple_features(builder: PromptBuilder) -> None:
    """Intent может подавлять несколько фич одновременно."""
    intent_config = IntentConfig(
        name="test_intent",
        instructions=(),
        priority=50,
        suppresses=("storytelling", "marketing"),
        conflicts_with=(),
    )
    domain_config = builder.get_domain_config("marketing")
    overlay_configs = [builder.get_overlay_config("base")]
    result = resolve_prompt_features(
        domain="marketing",
        intent="test_intent",
        overlays=["base"],
        domain_config=domain_config,
        intent_config=intent_config,
        overlay_configs=overlay_configs,
    )
    assert result["storytelling_enabled"] is False
    assert result["marketing_enabled"] is False
    suppression_reasons = result["suppression_reasons"]
    assert any(ReasonCode.SUPPRESSED_BY_INTENT_RULE in reasons
               for reasons in suppression_reasons.get("storytelling", []))
    assert any(ReasonCode.SUPPRESSED_BY_INTENT_RULE in reasons
               for reasons in suppression_reasons.get("marketing", []))


def test_intent_suppresses_does_not_affect_other_features(builder: PromptBuilder) -> None:
    """Подавление storytelling не должно влиять на marketing, если не указано."""
    intent_config = IntentConfig(
        name="test_intent",
        instructions=(),
        priority=50,
        suppresses=("storytelling",),
        conflicts_with=(),
    )
    domain_config = builder.get_domain_config("marketing")
    overlay_configs = [builder.get_overlay_config("base")]
    result = resolve_prompt_features(
        domain="marketing",
        intent="test_intent",
        overlays=["base"],
        domain_config=domain_config,
        intent_config=intent_config,
        overlay_configs=overlay_configs,
    )
    assert result["storytelling_enabled"] is False
    assert result["marketing_enabled"] is True


# ============================================================================
# Тесты для Итерации 5 (overlay conflicts и priority)
# ============================================================================

def test_overlay_conflict_higher_priority_wins(builder_with_mock: PromptBuilder) -> None:
    """При конфликте оверлеев побеждает тот, у кого выше priority."""
    overlay_high = OverlayConfig(
        name="high_priority",
        instructions=(),
        priority=100,
        suppresses=(),
        conflicts_with=("low_priority",),
    )
    overlay_low = OverlayConfig(
        name="low_priority",
        instructions=(),
        priority=50,
        suppresses=(),
        conflicts_with=("high_priority",),
    )
    domain_config = builder_with_mock.get_domain_config("blog")
    result = resolve_prompt_features(
        domain="blog",
        intent=None,
        overlays=["low_priority", "high_priority"],
        domain_config=domain_config,
        intent_config=None,
        overlay_configs=[overlay_low, overlay_high],
    )
    assert "high_priority" in result["effective_overlays"]
    assert "low_priority" not in result["effective_overlays"]
    assert any("low_priority" in layer for layer in result["suppressed_layers"])


def test_overlay_conflict_resolution_independent_of_input_order(builder_with_mock: PromptBuilder) -> None:
    """Порядок overlays во входном списке не влияет на результат."""
    overlay_high = OverlayConfig(
        name="high_priority",
        instructions=(),
        priority=100,
        suppresses=(),
        conflicts_with=("low_priority",),
    )
    overlay_low = OverlayConfig(
        name="low_priority",
        instructions=(),
        priority=50,
        suppresses=(),
        conflicts_with=("high_priority",),
    )
    domain_config = builder_with_mock.get_domain_config("blog")
    result1 = resolve_prompt_features(
        domain="blog",
        intent=None,
        overlays=["high_priority", "low_priority"],
        domain_config=domain_config,
        intent_config=None,
        overlay_configs=[overlay_high, overlay_low],
    )
    result2 = resolve_prompt_features(
        domain="blog",
        intent=None,
        overlays=["low_priority", "high_priority"],
        domain_config=domain_config,
        intent_config=None,
        overlay_configs=[overlay_low, overlay_high],
    )
    assert "high_priority" in result1["effective_overlays"]
    assert "low_priority" not in result1["effective_overlays"]
    assert "high_priority" in result2["effective_overlays"]
    assert "low_priority" not in result2["effective_overlays"]


def test_overlay_conflict_equal_priority_uses_deterministic_fallback(builder_with_mock: PromptBuilder) -> None:
    """При конфликте с равными приоритетами и без явного победителя — детерминированный fallback."""
    overlay_a = OverlayConfig(
        name="overlay_a",
        instructions=(),
        priority=50,
        suppresses=(),
        conflicts_with=("overlay_b",),
    )
    overlay_b = OverlayConfig(
        name="overlay_b",
        instructions=(),
        priority=50,
        suppresses=(),
        conflicts_with=("overlay_a",),
    )
    domain_config = builder_with_mock.get_domain_config("blog")
    result = resolve_prompt_features(
        domain="blog",
        intent=None,
        overlays=["overlay_a", "overlay_b"],
        domain_config=domain_config,
        intent_config=None,
        overlay_configs=[overlay_a, overlay_b],
    )
    # Детерминированный fallback: побеждает первый в списке (overlay_a)
    assert "overlay_a" in result["effective_overlays"]
    assert "overlay_b" not in result["effective_overlays"]
    # Проверяем наличие предупреждения о равных приоритетах
    assert any("equal priority" in w for w in result["warnings"])


def test_explicit_suppression_wins_over_priority(builder_with_mock: PromptBuilder) -> None:
    """Явное suppresses (domain или overlay) побеждает сравнение priority."""
    overlay_high = OverlayConfig(
        name="high_priority",
        instructions=(),
        priority=100,
        suppresses=("low_priority",),
        conflicts_with=("low_priority",),
    )
    overlay_low = OverlayConfig(
        name="low_priority",
        instructions=(),
        priority=50,
        suppresses=(),
        conflicts_with=("high_priority",),
    )
    domain_config = builder_with_mock.get_domain_config("blog")
    result = resolve_prompt_features(
        domain="blog",
        intent=None,
        overlays=["low_priority", "high_priority"],
        domain_config=domain_config,
        intent_config=None,
        overlay_configs=[overlay_low, overlay_high],
    )
    assert "high_priority" in result["effective_overlays"]
    assert "low_priority" not in result["effective_overlays"]
    assert any("low_priority" in layer and "suppressed" in layer for layer in result["suppressed_layers"])


def test_equal_priority_conflict_uses_deterministic_fallback_with_mock_overlays(builder_with_mock: PromptBuilder) -> None:
    """Искусственные оверлеи с равными приоритетами (70) — детерминированный fallback."""
    overlay_a = OverlayConfig(
        name="overlay_a",
        instructions=(),
        priority=70,
        suppresses=(),
        conflicts_with=("overlay_b",),
    )
    overlay_b = OverlayConfig(
        name="overlay_b",
        instructions=(),
        priority=70,
        suppresses=(),
        conflicts_with=("overlay_a",),
    )
    domain_config = builder_with_mock.get_domain_config("blog")
    result = resolve_prompt_features(
        domain="blog",
        intent=None,
        overlays=["overlay_a", "overlay_b"],
        domain_config=domain_config,
        intent_config=None,
        overlay_configs=[overlay_a, overlay_b],
    )
    # Детерминированный fallback: побеждает первый в списке (overlay_a)
    assert "overlay_a" in result["effective_overlays"]
    assert "overlay_b" not in result["effective_overlays"]
    # Проверяем наличие предупреждения о равных приоритетах
    assert any("equal priority" in w for w in result["warnings"])


# ============================================================================
# Тест с реальными оверлеями (с принудительной перезагрузкой)
# ============================================================================

def test_real_genre_overlays_conflict_with_suppress() -> None:
    """Реальные жанровые оверлеи с равным priority (70) и явным suppress должны разрешаться без ошибки."""
    # Создаём свежий билдер, чтобы исключить кэширование
    builder = PromptBuilder(config_path=Path("config"), kb_path=Path("knowledge_base"))
    # Принудительно перезагружаем все конфиги
    builder.reload_configs()
    # Дополнительно очищаем кэш оверлеев на всякий случай
    builder._overlay_cache.clear()

    domain_config = builder.get_domain_config("genre")
    landing_cfg = builder.get_overlay_config("landing")
    pressrelease_cfg = builder.get_overlay_config("pressrelease")

    overlay_configs = [landing_cfg, pressrelease_cfg]
    result = resolve_prompt_features(
        domain="genre",
        intent=None,
        overlays=["landing", "pressrelease"],
        domain_config=domain_config,
        intent_config=None,
        overlay_configs=overlay_configs,
    )

    # Проверяем, что landing остался, pressrelease удалён
    assert "landing" in result["effective_overlays"]
    assert "pressrelease" not in result["effective_overlays"]
    # Проверяем, что причина подавления — SUPPRESSED_BY_OVERLAY_RULE (явный suppress)
    suppression_reasons = result["suppression_reasons"]
    assert "overlay:pressrelease" in suppression_reasons
    assert any(
        ReasonCode.SUPPRESSED_BY_OVERLAY_RULE in reasons
        for reasons in suppression_reasons.values()
    )


# ============================================================================
# НОВЫЕ ТЕСТЫ: префиксованный формат incompatible_* (Итерация 5)
# ============================================================================

def test_domain_incompatible_overlay_with_prefix() -> None:
    """
    Проверяет, что домен с incompatible_overlays: ["overlay:infostyle"]
    корректно подавляет оверлей infostyle.
    """
    builder = PromptBuilder(config_path=Path("config"), kb_path=Path("knowledge_base"))
    domain_config = builder.get_domain_config("deai")  # deai.json содержит "overlay:infostyle"
    overlay_configs = [builder.get_overlay_config("infostyle")]

    result = resolve_prompt_features(
        domain="deai",
        intent=None,
        overlays=["infostyle"],
        domain_config=domain_config,
        intent_config=None,
        overlay_configs=overlay_configs,
    )

    assert "infostyle" not in result["effective_overlays"]
    suppression_reasons = result["suppression_reasons"]
    assert any(
        ReasonCode.SUPPRESSED_BY_DOMAIN_INCOMPATIBLE_OVERLAY in reasons
        for reasons in suppression_reasons.values()
    )


def test_domain_incompatible_intent_with_prefix() -> None:
    """
    Проверяет, что домен с incompatible_intents: ["intent:analytical"]
    корректно подавляет интент analytical.
    """
    builder = PromptBuilder(config_path=Path("config"), kb_path=Path("knowledge_base"))
    domain_config = builder.get_domain_config("fiction")  # fiction.json содержит "intent:analytical"
    overlay_configs = [builder.get_overlay_config("base")]

    result = resolve_prompt_features(
        domain="fiction",
        intent="analytical",
        overlays=["base"],
        domain_config=domain_config,
        intent_config=None,
        overlay_configs=overlay_configs,
    )

    assert result["effective_intent"] is None
    suppression_reasons = result["suppression_reasons"]
    assert any(
        ReasonCode.SUPPRESSED_BY_DOMAIN_INCOMPATIBLE_INTENT in reasons
        for reasons in suppression_reasons.values()
    )


def test_domain_incompatible_overlay_without_prefix_still_works() -> None:
    """
    Проверяет обратную совместимость: старый формат без префикса всё ещё работает.
    Используем домен nora_gal, где incompatible_overlays: ["infostyle"] (без префикса).
    """
    builder = PromptBuilder(config_path=Path("config"), kb_path=Path("knowledge_base"))
    domain_config = builder.get_domain_config("nora_gal")  # в nora_gal.json без префикса
    overlay_configs = [builder.get_overlay_config("infostyle")]

    result = resolve_prompt_features(
        domain="nora_gal",
        intent=None,
        overlays=["infostyle"],
        domain_config=domain_config,
        intent_config=None,
        overlay_configs=overlay_configs,
    )

    assert "infostyle" not in result["effective_overlays"]
    suppression_reasons = result["suppression_reasons"]
    assert any(
        ReasonCode.SUPPRESSED_BY_DOMAIN_INCOMPATIBLE_OVERLAY in reasons
        for reasons in suppression_reasons.values()
    )