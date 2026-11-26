"""LLM-клиенты: фасад над пакетом src/llm/.

Итерация 17.1.b карты №3. Исторический модуль ``src.llm_client``
оставлен как совместимый фасад: все публичные имена реэкспортируются
из пакета ``src.llm``, весь код перенесён туда.

Monkeypatch-пути вида ``src.llm_client.X`` продолжат работать для
классов и функций, которые тесты мокают *на самом фасаде*; для
патчей, затрагивающих внутренние вызовы между модулями пакета
(например, ``fallback.create_llm_client``), тесты мигрированы на
пути ``src.llm.<module>.<name>``.
"""

from __future__ import annotations

# `import asyncio` сохранён для совместимости с
# patch("src.llm_client.asyncio.sleep") в тестах: патч фактически
# заменяет атрибут глобального модуля asyncio, что работает и для
# нового пакета.
import asyncio  # noqa: F401  # нужен для patch("src.llm_client.asyncio.sleep")

from dotenv import load_dotenv

from src.llm import (
    _CHARS_PER_TOKEN,
    _DEFAULT_MAX_TOKENS,
    _MAX_MAX_TOKENS,
    _MIN_MAX_TOKENS,
    _RESPONSE_BUDGET_MULTIPLIER,
    AnthropicClient,
    BaseLLMClient,
    LLMAPIError,
    LLMConfig,
    LLMError,
    LLMFallbackError,
    LLMInvalidResponseError,
    LLMProvider,
    LLMRateLimitError,
    LLMResponse,
    LLMTimeoutError,
    OpenAIClient,
    OpenRouterClient,
    PerplexityClient,
    _backoff_with_jitter,
    _build_fallback_error,
    _classify_api_status,
    _classify_cause_if_llm_error,
    _classify_error,
    _collect_response_diagnostics,
    _extract_upstream_status,
    _format_diagnostics,
    _looks_like_context_limit,
    _OpenAICompatibleClient,
    _resolve_openrouter_reasoning_enabled,
    _resolve_provider_max_tokens,
    _safe_int,
    _safe_str,
    _try_provider,
    call_with_fallback,
    create_llm_client,
    estimate_max_tokens,
    generate_text,
)

# Семантика оригинала: llm_client.load_dotenv() на импорте модуля.
load_dotenv()

__all__ = [
    "_CHARS_PER_TOKEN",
    "_DEFAULT_MAX_TOKENS",
    "_MAX_MAX_TOKENS",
    "_MIN_MAX_TOKENS",
    "_RESPONSE_BUDGET_MULTIPLIER",
    "AnthropicClient",
    "BaseLLMClient",
    "LLMAPIError",
    "LLMConfig",
    "LLMError",
    "LLMFallbackError",
    "LLMInvalidResponseError",
    "LLMProvider",
    "LLMRateLimitError",
    "LLMResponse",
    "LLMTimeoutError",
    "OpenAIClient",
    "OpenRouterClient",
    "PerplexityClient",
    "_OpenAICompatibleClient",
    "_backoff_with_jitter",
    "_build_fallback_error",
    "_classify_api_status",
    "_classify_cause_if_llm_error",
    "_classify_error",
    "_collect_response_diagnostics",
    "_extract_upstream_status",
    "_format_diagnostics",
    "_looks_like_context_limit",
    "_resolve_openrouter_reasoning_enabled",
    "_resolve_provider_max_tokens",
    "_safe_int",
    "_safe_str",
    "_try_provider",
    "call_with_fallback",
    "create_llm_client",
    "estimate_max_tokens",
    "generate_text",
]
