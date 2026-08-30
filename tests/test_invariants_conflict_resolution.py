"""Инварианты разрешения конфликтов: проверка explainability и детерминизма."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.prompt_builder import PromptBuilder, resolve_prompt_features
from src.reason_codes import ACTIVATION_REASONS, SUPPRESSION_REASONS, ReasonCode
from src.config_types import IntentConfig, OverlayConfig


@pytest.fixture
def builder() -> PromptBuilder:
    return PromptBuilder(config_path=Path("config"), kb_path=Path("knowledge_base"))


@pytest.fixture
def builder_with_mock(monkeypatch: pytest.MonkeyPatch) -> PromptBuilder:
    """Builder с подменой normalize_overlays для искусственных оверлеев."""
    def mock_normalize_overlays(overlays, **kwargs):
        return list(overlays)
    monkeypatch.setattr("src.prompt_builder.normalize_overlays", mock_normalize_overlays)
    return PromptBuilder(config_path=Path("config"), kb_path=Path("knowledge_base"))


def test_every_suppressed_layer_has_reason_code(builder: PromptBuilder) -> None:
    """Каждый подавленный слой должен иметь хотя бы один suppression reason."""
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
    assert any("infostyle" in layer for layer in result["suppressed_layers"])
    assert "overlay:infostyle" in result["suppression_reasons"]
    reasons = result["suppression_reasons"]["overlay:infostyle"]
    assert any(r in SUPPRESSION_REASONS for r in reasons)


def test_every_activated_feature_has_activation_reason(builder: PromptBuilder) -> None:
    """Каждая активированная фича должна иметь activation reason."""
    domain_config = builder.get_domain_config("blog")
    intent_config = IntentConfig(
        name="storytelling",
        instructions=(),
        priority=50,
        suppresses=(),
        conflicts_with=(),
    )
    overlay_configs = [builder.get_overlay_config("base")]
    result = resolve_prompt_features(
        domain="blog",
        intent="storytelling",
        overlays=["base"],
        domain_config=domain_config,
        intent_config=intent_config,
        overlay_configs=overlay_configs,
    )
    assert result["storytelling_enabled"] is True
    assert "storytelling" in result["activation_reasons"]
    reasons = result["activation_reasons"]["storytelling"]
    assert any(r in ACTIVATION_REASONS for r in reasons)


def test_effective_overlays_do_not_contain_suppressed_overlays(builder: PromptBuilder) -> None:
    """effective_overlays не должны содержать подавленные оверлеи."""
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


def test_final_tags_do_not_contain_suppressed_layers(builder: PromptBuilder) -> None:
    """Финальные теги не должны содержать подавленные слои."""
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
    assert "infostyle" not in result["tags"]
    assert "deai" in result["tags"]


def test_resolution_independent_of_overlay_order(builder_with_mock: PromptBuilder) -> None:
    """Порядок оверлеев во входном списке не влияет на результат."""
    overlay_high = OverlayConfig(
        name="high",
        instructions=(),
        priority=100,
        suppresses=(),
        conflicts_with=("low",),
    )
    overlay_low = OverlayConfig(
        name="low",
        instructions=(),
        priority=50,
        suppresses=(),
        conflicts_with=("high",),
    )
    domain_config = builder_with_mock.get_domain_config("blog")
    result1 = resolve_prompt_features(
        domain="blog",
        intent=None,
        overlays=["high", "low"],
        domain_config=domain_config,
        intent_config=None,
        overlay_configs=[overlay_high, overlay_low],
    )
    result2 = resolve_prompt_features(
        domain="blog",
        intent=None,
        overlays=["low", "high"],
        domain_config=domain_config,
        intent_config=None,
        overlay_configs=[overlay_low, overlay_high],
    )
    assert result1["effective_overlays"] == ["high"]
    assert result2["effective_overlays"] == ["high"]


def test_resolution_deterministic_for_same_config(builder: PromptBuilder) -> None:
    """При одинаковой конфигурации результат должен быть детерминированным."""
    domain_config = builder.get_domain_config("blog")
    overlay_configs = [builder.get_overlay_config("base")]
    result1 = resolve_prompt_features(
        domain="blog",
        intent=None,
        overlays=["base"],
        domain_config=domain_config,
        intent_config=None,
        overlay_configs=overlay_configs,
    )
    result2 = resolve_prompt_features(
        domain="blog",
        intent=None,
        overlays=["base"],
        domain_config=domain_config,
        intent_config=None,
        overlay_configs=overlay_configs,
    )
    assert result1["effective_intent"] == result2["effective_intent"]
    assert result1["effective_overlays"] == result2["effective_overlays"]
    assert result1["suppressed_layers"] == result2["suppressed_layers"]
    assert result1["storytelling_enabled"] == result2["storytelling_enabled"]
    assert result1["marketing_enabled"] == result2["marketing_enabled"]