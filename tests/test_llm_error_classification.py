"""Характеристические тесты для _classify_error (итерация 18, PR-6).

Фиксируют текущую классификацию ошибок до рефакторинга.
Порядок проверок и поведение с __cause__ — часть контракта.
"""

from __future__ import annotations

import httpx
import pytest

from src.llm_client import (
    LLMAPIError,
    LLMError,
    LLMInvalidResponseError,
    LLMRateLimitError,
    LLMTimeoutError,
    _classify_error,
)


def _with_cause(error: LLMError, cause: BaseException) -> LLMError:
    """Устанавливает __cause__ без raise-from (для тестов)."""
    error.__cause__ = cause
    return error


# --- direct types ----------------------------------------------------------


def test_classify_invalid_response_direct() -> None:
    assert _classify_error(LLMInvalidResponseError("EMPTY_CONTENT")) == "invalid_response"


def test_classify_rate_limit_direct() -> None:
    assert _classify_error(LLMRateLimitError("429")) == "rate_limit"


def test_classify_timeout_direct() -> None:
    assert _classify_error(LLMTimeoutError("timed out")) == "timeout"


# --- APIError by status ----------------------------------------------------


def test_classify_api_429_rate_limit() -> None:
    assert _classify_error(LLMAPIError("429", status_code=429)) == "rate_limit"


def test_classify_api_401_authentication() -> None:
    assert _classify_error(LLMAPIError("401", status_code=401)) == "authentication"


def test_classify_api_403_authentication() -> None:
    assert _classify_error(LLMAPIError("403", status_code=403)) == "authentication"


def test_classify_api_413_context_limit() -> None:
    assert _classify_error(LLMAPIError("413", status_code=413)) == "context_limit"


def test_classify_api_500_upstream_error() -> None:
    assert _classify_error(LLMAPIError("500", status_code=500)) == "upstream_error"


def test_classify_api_503_upstream_error() -> None:
    assert _classify_error(LLMAPIError("503", status_code=503)) == "upstream_error"


# --- APIError status 400: message inspection -------------------------------


@pytest.mark.parametrize(
    "phrase",
    [
        "context length",
        "context window",
        "maximum context",
        "max tokens",
        "token limit",
        "too many tokens",
    ],
)
def test_classify_api_400_context_phrases(phrase: str) -> None:
    msg = f"bad request: {phrase} exceeded"
    assert _classify_error(LLMAPIError(msg, status_code=400)) == "context_limit"


def test_classify_api_400_generic_upstream_error() -> None:
    assert _classify_error(LLMAPIError("bad request", status_code=400)) == "upstream_error"


# --- cause chain -----------------------------------------------------------


def test_classify_api_error_caused_by_invalid_response() -> None:
    error = _with_cause(
        LLMAPIError("api", status_code=500),
        LLMInvalidResponseError("EMPTY_CONTENT"),
    )
    assert _classify_error(error) == "invalid_response"


def test_classify_api_error_caused_by_rate_limit() -> None:
    error = _with_cause(LLMAPIError("api", status_code=500), LLMRateLimitError("429"))
    assert _classify_error(error) == "rate_limit"


def test_classify_rate_limit_with_upstream_cause_recurses() -> None:
    """__cause__ проверяется раньше типа: rate_limit с 5xx-cause → upstream_error."""
    error = _with_cause(LLMRateLimitError("429"), LLMAPIError("500", status_code=500))
    assert _classify_error(error) == "upstream_error"


def test_classify_base_error_with_httpx_timeout_cause() -> None:
    error = _with_cause(LLMError("wrapped"), httpx.TimeoutException("timeout"))
    assert _classify_error(error) == "timeout"


def test_classify_base_error_without_cause_unknown() -> None:
    assert _classify_error(LLMError("bare")) == "unknown"


def test_classify_two_level_recursion() -> None:
    inner = LLMAPIError("500", status_code=500)
    mid = _with_cause(LLMError("mid"), inner)
    outer = _with_cause(LLMError("outer"), mid)
    assert _classify_error(outer) == "upstream_error"
