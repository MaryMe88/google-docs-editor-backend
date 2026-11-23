"""Пакет LLM-клиентов (итерация 17.1.a карты №3).

Публичный API пакета: реэкспортирует всё, что исторически было доступно
через ``src.llm_client``. На шаге 17.1.b ``llm_client.py`` станет тонким
фасадом над этим пакетом для сохранения совместимости.
"""

from __future__ import annotations

from src.provider_registry import LLMProvider

from ._constants import (
    _CHARS_PER_TOKEN,
    _DEFAULT_MAX_TOKENS,
    _MAX_MAX_TOKENS,
    _MIN_MAX_TOKENS,
    _RESPONSE_BUDGET_MULTIPLIER,
)
from .base import BaseLLMClient
from .classification import (
    _classify_api_status,
    _classify_cause_if_llm_error,
    _classify_error,
    _extract_upstream_status,
    _looks_like_context_limit,
)
from .config import LLMConfig, LLMResponse
from .diagnostics import (
    _backoff_with_jitter,
    _collect_response_diagnostics,
    _format_diagnostics,
    _safe_int,
    _safe_str,
)
from .errors import (
    LLMAPIError,
    LLMError,
    LLMFallbackError,
    LLMInvalidResponseError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from .factory import create_llm_client
from .fallback import (
    _build_fallback_error,
    _resolve_provider_max_tokens,
    _try_provider,
    call_with_fallback,
    generate_text,
)
from .providers import (
    AnthropicClient,
    OpenAIClient,
    OpenRouterClient,
    PerplexityClient,
    _OpenAICompatibleClient,
    _resolve_openrouter_reasoning_enabled,
)
from .tokens import estimate_max_tokens

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
