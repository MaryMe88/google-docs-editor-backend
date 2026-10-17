"""Тесты для src.model_policy.

Проверяют:
- DEFAULT_MODELS покрывает всех провайдеров;
- allowlist содержит только openrouter-модели;
- validate_explicit_model пропускает валидные пары;
- validate_explicit_model отвергает нативные провайдеры;
- validate_explicit_model отвергает неизвестные модели;
- validate_explicit_model пропускает model=None;
- get_default_model возвращает правильные значения;
- get_default_model бросает ValueError для неизвестного провайдера.
"""

from __future__ import annotations

import pytest

from src.model_policy import (
    ALLOWED_OPENROUTER_MODELS,
    DEFAULT_MODELS,
    ModelPolicyError,
    get_allowed_models_for_provider,
    get_default_model,
    is_allowed_pair,
    validate_explicit_model,
)
from src.provider_registry import LLMProvider


def test_default_models_covers_all_providers() -> None:
    """Для каждого провайдера из LLMProvider есть default-модель."""
    for provider in LLMProvider:
        assert provider.value in DEFAULT_MODELS


def test_allowed_openrouter_models_only_openrouter() -> None:
    """Allowlist содержит только модели для openrouter-провайдера."""
    assert "deepseek/deepseek-v4-flash" in ALLOWED_OPENROUTER_MODELS
    assert "openai/gpt-5" in ALLOWED_OPENROUTER_MODELS
    # Модели, которых нет — должны отсутствовать
    assert "sonar-pro" not in ALLOWED_OPENROUTER_MODELS
    assert "gpt-4o-mini" not in ALLOWED_OPENROUTER_MODELS


def test_validate_explicit_model_accepts_valid_pair() -> None:
    """Разрешённая пара (openrouter, gemini-3-pro) проходит без ошибок."""
    validate_explicit_model("openrouter", "google/gemini-3-pro-preview")


def test_validate_explicit_model_accepts_none() -> None:
    """model=None валиден для любого провайдера."""
    for provider in LLMProvider:
        validate_explicit_model(provider.value, None)


def test_validate_explicit_model_rejects_native_provider() -> None:
    """Явная модель для нативного провайдера отклоняется."""
    with pytest.raises(ModelPolicyError, match="only supported for provider"):
        validate_explicit_model("openai", "gpt-5")


def test_validate_explicit_model_rejects_unknown_model() -> None:
    """Неизвестная модель для openrouter отклоняется."""
    with pytest.raises(ModelPolicyError, match="not allowed for provider"):
        validate_explicit_model("openrouter", "evil-model")


def test_get_default_model_returns_correct() -> None:
    """get_default_model возвращает ожидаемые значения."""
    assert get_default_model("openrouter") == "deepseek/deepseek-v4-flash"
    assert get_default_model("openai") == "gpt-4o-mini"


def test_get_default_model_raises_for_unknown() -> None:
    """get_default_model бросает ValueError для неизвестного провайдера."""
    with pytest.raises(ValueError, match="No default model for provider"):
        get_default_model("unknown-provider")


def test_is_allowed_pair_returns_true_for_all_allowed() -> None:
    """is_allowed_pair возвращает True для каждой разрешённой пары."""
    for model in ALLOWED_OPENROUTER_MODELS:
        assert is_allowed_pair("openrouter", model) is True


def test_is_allowed_pair_returns_false_for_native_provider() -> None:
    """is_allowed_pair возвращает False для нативных провайдеров."""
    assert is_allowed_pair("openai", "gpt-5") is False
    assert is_allowed_pair("anthropic", "claude-sonnet-4") is False


def test_get_allowed_models_for_provider_openrouter() -> None:
    """Для openrouter возвращается полный allowlist."""
    assert get_allowed_models_for_provider("openrouter") == ALLOWED_OPENROUTER_MODELS


def test_get_allowed_models_for_provider_native_empty() -> None:
    """Для нативных провайдеров — пустое множество."""
    assert get_allowed_models_for_provider("openai") == frozenset()
