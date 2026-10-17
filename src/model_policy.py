"""Единая policy провайдеров и моделей для LLM API.

Единственный источник истины для:
- default-модели каждого провайдера (используется, когда клиент
  не передал model явно);
- allowlist пар (provider, model), которые можно передать явно.

Правила:
- model=None разрешён всегда, для любого провайдера из LLMProvider;
- явный model разрешён только для provider="openrouter" и только
  из ALLOWED_OPENROUTER_MODELS;
- явный model для нативных провайдеров (openai/anthropic/perplexity)
  отклоняется: клиенту следует использовать openrouter.
"""

from __future__ import annotations

from typing import Final

from src.provider_registry import LLMProvider

# Default-модели. Используются, когда клиент передал model=None.
DEFAULT_MODELS: Final[dict[str, str]] = {
    LLMProvider.PERPLEXITY.value: "sonar-pro",
    LLMProvider.OPENAI.value: "gpt-4o-mini",
    LLMProvider.ANTHROPIC.value: "claude-3-5-sonnet-20241022",
    LLMProvider.OPENROUTER.value: "deepseek/deepseek-v4-flash",
}

# Модели, доступные для явного выбора через OpenRouter.
ALLOWED_OPENROUTER_MODELS: Final[frozenset[str]] = frozenset(
    {
        "deepseek/deepseek-v4-flash",
        "openai/gpt-5",
        "openai/gpt-5-mini",
        "openai/gpt-5.2",
        "anthropic/claude-sonnet-4",
        "anthropic/claude-sonnet-4.5",
        "google/gemini-2.5-pro",
        "google/gemini-3-pro-preview",
        "google/gemini-3-flash-preview",
    }
)


class ModelPolicyError(ValueError):
    """Явно выбранная модель не разрешена policy."""


def get_default_model(provider: str) -> str:
    """Возвращает default-модель для провайдера.

    Raises:
        ValueError: если для провайдера нет default-модели.
    """
    try:
        return DEFAULT_MODELS[provider]
    except KeyError as exc:
        raise ValueError(f"No default model for provider: {provider!r}") from exc


def is_allowed_pair(provider: str, model: str) -> bool:
    """True, если модель можно явно передать с этим провайдером."""
    if provider != LLMProvider.OPENROUTER.value:
        return False
    return model in ALLOWED_OPENROUTER_MODELS


def get_allowed_models_for_provider(provider: str) -> frozenset[str]:
    """Модели, разрешённые для явного выбора у данного провайдера."""
    if provider == LLMProvider.OPENROUTER.value:
        return ALLOWED_OPENROUTER_MODELS
    return frozenset()


def validate_explicit_model(provider: str, model: str | None) -> None:
    """Проверяет пару (provider, model), если model передан явно.

    - model is None: OK (используется default).
    - provider не openrouter: ошибка с подсказкой использовать openrouter.
    - model не из allowlist: ошибка со списком разрешённых моделей.

    Raises:
        ModelPolicyError: если пара запрещена.
    """
    if model is None:
        return
    if provider != LLMProvider.OPENROUTER.value:
        raise ModelPolicyError(
            "Explicit model selection is only supported for provider="
            f"{LLMProvider.OPENROUTER.value!r}. Got provider={provider!r}. "
            "Use provider='openrouter' to choose a specific model, "
            "or omit 'model' to use the provider default."
        )
    if model not in ALLOWED_OPENROUTER_MODELS:
        allowed = ", ".join(sorted(ALLOWED_OPENROUTER_MODELS))
        raise ModelPolicyError(
            f"Model {model!r} is not allowed for provider='openrouter'. "
            f"Allowed models: {allowed}"
        )


__all__ = [
    "ALLOWED_OPENROUTER_MODELS",
    "DEFAULT_MODELS",
    "ModelPolicyError",
    "get_allowed_models_for_provider",
    "get_default_model",
    "is_allowed_pair",
    "validate_explicit_model",
]
