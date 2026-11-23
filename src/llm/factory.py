"""Фабрика LLM-клиентов по провайдеру.

Выделено из src/llm_client.py в итерации 17.1.a карты №3.
"""

from __future__ import annotations

import os
from collections.abc import Callable

from src.model_policy import get_default_model
from src.provider_registry import LLMProvider

from ._constants import _DEFAULT_MAX_TOKENS
from .base import BaseLLMClient
from .config import LLMConfig
from .providers import (
    AnthropicClient,
    OpenAIClient,
    OpenRouterClient,
    PerplexityClient,
)


def create_llm_client(
    provider: LLMProvider = LLMProvider.PERPLEXITY,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    timeout: float = 60.0,
    max_retries: int = 4,
) -> BaseLLMClient:
    """Создаёт клиент LLM по провайдеру.

    Args:
        provider: Провайдер; по умолчанию Perplexity.
        model: Модель; None — из src/model_policy.py.
        api_key: Ключ; None — из переменной окружения провайдера.
        temperature: Температура генерации.
        max_tokens: Лимит токенов ответа.
        timeout: Таймаут HTTP-запроса в секундах.
        max_retries: Максимум попыток в generate.

    Returns:
        Настроенный клиент провайдера.

    Raises:
        ValueError: Ключ не найден или провайдер не поддерживается.
    """
    # SEC (model_policy): единый источник default-моделей — src/model_policy.py.
    # Issue #22: openrouter/auto returns reasoning models that consume
    # the entire output budget. Explicit model keeps behavior predictable.

    env_keys = {
        LLMProvider.PERPLEXITY: "PERPLEXITY_API_KEY",
        LLMProvider.OPENAI: "OPENAI_API_KEY",
        LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
        LLMProvider.OPENROUTER: "OPENROUTER_API_KEY",
    }

    if model is None:
        model = get_default_model(provider.value)

    if api_key is None:
        env_key = env_keys.get(provider)
        if env_key is not None:
            api_key = os.getenv(env_key)

    if not api_key:
        env_key = env_keys.get(provider, "UNKNOWN_API_KEY")
        raise ValueError(f"API key not provided and not found in environment variable {env_key}")

    config = LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
    )

    client_classes: dict[LLMProvider, Callable[[LLMConfig], BaseLLMClient]] = {
        LLMProvider.PERPLEXITY: PerplexityClient,
        LLMProvider.OPENAI: OpenAIClient,
        LLMProvider.ANTHROPIC: AnthropicClient,
        LLMProvider.OPENROUTER: OpenRouterClient,
    }

    client_class = client_classes.get(provider)
    if client_class is None:
        raise ValueError(f"Unsupported provider: {provider}")

    return client_class(config)
