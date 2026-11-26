"""Классификация ошибок LLM для LLMFallbackError.

Выделено из src/llm_client.py в итерации 17.1.a карты №3.
"""

from __future__ import annotations

import httpx

from .errors import (
    LLMAPIError,
    LLMError,
    LLMInvalidResponseError,
    LLMRateLimitError,
    LLMTimeoutError,
)


def _extract_upstream_status(error: LLMError) -> int | None:
    """Извлекает HTTP статус из ошибки, если он присутствует."""
    if hasattr(error, "status_code") and isinstance(error.status_code, int):
        return error.status_code
    cause = error.__cause__
    if cause is not None and hasattr(cause, "status_code") and isinstance(cause.status_code, int):
        return cause.status_code
    return None


def _classify_cause_if_llm_error(error: LLMError) -> str | None:
    """Рекурсивная классификация по __cause__, если cause — LLMError. Иначе None."""
    cause = error.__cause__
    if cause is None:
        return None
    if isinstance(cause, LLMInvalidResponseError):
        return "invalid_response"
    if isinstance(cause, LLMError):
        return _classify_error(cause)
    return None


def _looks_like_context_limit(message: str) -> bool:
    """True, если сообщение содержит фразы про context limit."""
    phrases = (
        "context length",
        "context window",
        "maximum context",
        "max tokens",
        "token limit",
        "too many tokens",
    )
    lowered = message.lower()
    return any(phrase in lowered for phrase in phrases)


def _classify_api_status(error: LLMAPIError) -> str:
    """Классифицирует LLMAPIError по status_code и тексту сообщения."""
    status = _extract_upstream_status(error)
    if status == 429:
        return "rate_limit"
    if status == 401 or status == 403:
        return "authentication"
    if status == 413:
        return "context_limit"
    if status and 500 <= status < 600:
        return "upstream_error"
    if status == 400 and _looks_like_context_limit(str(error)):
        return "context_limit"
    return "upstream_error"


def _classify_error(error: LLMError) -> str:
    """Классифицирует ошибку для LLMFallbackError."""
    if isinstance(error, LLMInvalidResponseError):
        return "invalid_response"

    cause_class = _classify_cause_if_llm_error(error)
    if cause_class is not None:
        return cause_class

    if isinstance(error, LLMRateLimitError):
        return "rate_limit"
    if isinstance(error, LLMTimeoutError):
        return "timeout"
    if isinstance(error, LLMAPIError):
        return _classify_api_status(error)
    if isinstance(error.__cause__, httpx.TimeoutException):
        return "timeout"
    return "unknown"
