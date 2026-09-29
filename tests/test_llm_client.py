"""
tests/test_llm_client.py
========================
Тесты для llm_client, проверяющие обработку ошибок и fallback.

Запуск:
    pytest tests/test_llm_client.py -v
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.context_budget import LLMContextLimitError
from src.llm_client import (
    _CHARS_PER_TOKEN,
    _MAX_MAX_TOKENS,
    _MIN_MAX_TOKENS,
    LLMAPIError,
    LLMConfig,
    LLMError,
    LLMFallbackError,
    LLMInvalidResponseError,
    LLMProvider,
    LLMResponse,
    LLMTimeoutError,
    OpenRouterClient,
    _build_fallback_error,
    _collect_response_diagnostics,
    _format_diagnostics,
    _resolve_provider_max_tokens,
    _safe_int,
    _safe_str,
    _try_provider,
    call_with_fallback,
    estimate_max_tokens,
)


# ---------------------------------------------------------------------------
# Helper для создания клиента с ошибкой
# ---------------------------------------------------------------------------
def make_failing_client(error: Exception) -> MagicMock:
    """Создаёт клиент, который при generate выбрасывает error."""
    client = MagicMock()
    client.generate = AsyncMock(side_effect=error)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


# ---------------------------------------------------------------------------
# Адаптивный max_tokens
# ---------------------------------------------------------------------------
def test_estimate_max_tokens_short_prompt_uses_floor() -> None:
    """Короткий промпт не опускает лимит ниже нижней границы."""
    assert estimate_max_tokens("") == _MIN_MAX_TOKENS
    assert estimate_max_tokens("Привет") == _MIN_MAX_TOKENS


def test_estimate_max_tokens_long_prompt_scales_up() -> None:
    """Длинный промпт повышает лимит выше нижней границы."""
    prompt = "а" * (_MIN_MAX_TOKENS * _CHARS_PER_TOKEN * 2)
    result = estimate_max_tokens(prompt)
    assert result > _MIN_MAX_TOKENS
    assert result <= _MAX_MAX_TOKENS


def test_estimate_max_tokens_caps_at_ceiling() -> None:
    """Очень длинный промпт не превышает верхнюю границу."""
    prompt = "а" * (_MAX_MAX_TOKENS * _CHARS_PER_TOKEN * 10)
    assert estimate_max_tokens(prompt) == _MAX_MAX_TOKENS


@pytest.mark.asyncio
async def test_call_with_fallback_empty_providers() -> None:
    """При пустом списке провайдеров должно выбрасываться LLMError."""
    with pytest.raises(LLMError, match="No providers specified"):
        await call_with_fallback(
            prompt="test prompt",
            providers=[],
        )


@pytest.mark.asyncio
async def test_call_with_fallback_unknown_provider() -> None:
    """При неизвестном провайдере выбрасывается LLMFallbackError с kind=configuration."""
    with pytest.raises(LLMFallbackError) as exc_info:
        await call_with_fallback(
            prompt="test prompt",
            providers=["unknown_xyz_provider"],
            max_retries_per_provider=0,
        )
    assert exc_info.value.kind == "configuration"
    assert "unknown_xyz_provider" in exc_info.value.unknown_providers


@pytest.mark.asyncio
async def test_call_with_fallback_mixed_unknown_and_known() -> None:
    """Неизвестные провайдеры пропускаются, если все неизвестны -> configuration."""
    with pytest.raises(LLMFallbackError) as exc_info:
        await call_with_fallback(
            prompt="test",
            providers=["unknown1", "unknown2"],
            max_retries_per_provider=0,
        )
    assert exc_info.value.kind == "configuration"
    assert sorted(exc_info.value.unknown_providers) == ["unknown1", "unknown2"]


# ---------------------------------------------------------------------------
# Исправленные тесты для fallback с сохранением первичной ошибки
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fallback_primary_error_preserved_when_skipping_unconfigured() -> None:
    """
    Сценарий: первый провайдер (openrouter) возвращает LLMAPIError с 503,
    остальные провайдеры ненастроены (ValueError).
    Должен возникнуть LLMFallbackError с primary_provider == "openrouter"
    и kind == "upstream_error", skipped_providers содержит остальных.
    """
    openrouter_error = LLMAPIError("upstream temporarily unavailable", status_code=503)
    openrouter_client = make_failing_client(openrouter_error)

    def fake_create_llm_client(provider: LLMProvider, **kwargs):
        if provider == LLMProvider.OPENROUTER:
            return openrouter_client
        raise ValueError("provider not configured")

    with (
        patch(
            "src.llm_client.create_llm_client",
            side_effect=fake_create_llm_client,
        ),
        pytest.raises(LLMFallbackError) as exc_info,
    ):
        await call_with_fallback(
            prompt="test",
            providers=["openrouter", "anthropic", "openai", "perplexity"],
            model="some-model",
            max_retries_per_provider=1,
        )

    error = exc_info.value
    assert error.provider == "openrouter"
    assert error.kind == "upstream_error"
    assert error.upstream_status == 503
    assert error.skipped_providers == ("anthropic", "openai", "perplexity")
    assert error.unknown_providers == ()
    assert error.prompt_length == 4
    assert openrouter_client.generate.await_count == 1


@pytest.mark.asyncio
async def test_fallback_all_providers_unconfigured_raises_configuration() -> None:
    """
    Все провайдеры ненастроены → primary_error None → kind = "configuration".
    """

    def fake_create_llm_client(provider: LLMProvider, **kwargs):
        raise ValueError("Missing API key")

    with (
        patch(
            "src.llm_client.create_llm_client",
            side_effect=fake_create_llm_client,
        ),
        pytest.raises(LLMFallbackError) as exc_info,
    ):
        await call_with_fallback(
            prompt="test",
            providers=["openrouter", "anthropic", "openai"],
            max_retries_per_provider=1,
        )

    error = exc_info.value
    assert error.provider is None
    assert error.kind == "configuration"
    assert error.upstream_status is None
    assert error.skipped_providers == ("openrouter", "anthropic", "openai")
    assert error.unknown_providers == ()


@pytest.mark.asyncio
async def test_fallback_http_429_classified_as_rate_limit() -> None:
    """HTTP 429 → kind = rate_limit."""
    rate_limit_error = LLMAPIError("Rate limit", status_code=429)
    rate_limit_client = make_failing_client(rate_limit_error)

    def fake_create_llm_client(provider: LLMProvider, **kwargs):
        if provider == LLMProvider.OPENROUTER:
            return rate_limit_client
        raise ValueError("Missing")

    with (
        patch(
            "src.llm_client.create_llm_client",
            side_effect=fake_create_llm_client,
        ),
        pytest.raises(LLMFallbackError) as exc_info,
    ):
        await call_with_fallback(
            prompt="test",
            providers=["openrouter", "anthropic"],
            max_retries_per_provider=1,
        )

    assert exc_info.value.kind == "rate_limit"
    assert rate_limit_client.generate.await_count == 1


@pytest.mark.asyncio
async def test_fallback_timeout_classified_as_timeout() -> None:
    """LLMTimeoutError → kind = timeout."""
    timeout_error = LLMTimeoutError("Timeout")
    timeout_client = make_failing_client(timeout_error)

    def fake_create_llm_client(provider: LLMProvider, **kwargs):
        if provider == LLMProvider.OPENROUTER:
            return timeout_client
        raise ValueError("Missing")

    with (
        patch(
            "src.llm_client.create_llm_client",
            side_effect=fake_create_llm_client,
        ),
        pytest.raises(LLMFallbackError) as exc_info,
    ):
        await call_with_fallback(
            prompt="test",
            providers=["openrouter", "anthropic"],
            max_retries_per_provider=1,
        )

    assert exc_info.value.kind == "timeout"
    assert timeout_client.generate.await_count == 1


@pytest.mark.asyncio
async def test_fallback_http_413_classified_as_context_limit() -> None:
    """HTTP 413 → kind = context_limit."""
    context_error = LLMAPIError("Too large", status_code=413)
    context_client = make_failing_client(context_error)

    def fake_create_llm_client(provider: LLMProvider, **kwargs):
        if provider == LLMProvider.OPENROUTER:
            return context_client
        raise ValueError("Missing")

    with (
        patch(
            "src.llm_client.create_llm_client",
            side_effect=fake_create_llm_client,
        ),
        pytest.raises(LLMFallbackError) as exc_info,
    ):
        await call_with_fallback(
            prompt="test",
            providers=["openrouter", "anthropic"],
            max_retries_per_provider=1,
        )

    assert exc_info.value.kind == "context_limit"
    assert context_client.generate.await_count == 1


@pytest.mark.asyncio
async def test_fallback_http_400_without_context_words_not_context_limit() -> None:
    """HTTP 400 без упоминания контекста → upstream_error, не context_limit."""
    bad_request_error = LLMAPIError("Bad request", status_code=400)
    bad_request_client = make_failing_client(bad_request_error)

    def fake_create_llm_client(provider: LLMProvider, **kwargs):
        if provider == LLMProvider.OPENROUTER:
            return bad_request_client
        raise ValueError("Missing")

    with (
        patch(
            "src.llm_client.create_llm_client",
            side_effect=fake_create_llm_client,
        ),
        pytest.raises(LLMFallbackError) as exc_info,
    ):
        await call_with_fallback(
            prompt="test",
            providers=["openrouter", "anthropic"],
            max_retries_per_provider=1,
        )

    assert exc_info.value.kind == "upstream_error"
    assert bad_request_client.generate.await_count == 1


@pytest.mark.asyncio
async def test_fallback_model_passed_only_to_first_provider() -> None:
    """
    Проверяем, что model передаётся только первому провайдеру, остальным — None.
    Первый провайдер падает с LLMError, второй — ненастроен.
    """
    calls = []

    def fake_create_llm_client(provider: LLMProvider, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            # Первый провайдер возвращает клиент, который при generate падает
            client = make_failing_client(LLMError("First provider failed"))
            return client
        else:
            # Второй провайдер ненастроен
            raise ValueError("Missing key")

    with (
        patch(
            "src.llm_client.create_llm_client",
            side_effect=fake_create_llm_client,
        ),
        pytest.raises(LLMFallbackError),
    ):
        await call_with_fallback(
            prompt="test",
            providers=["openrouter", "anthropic"],
            model="gpt-4",
            max_retries_per_provider=0,
        )

    # Первый вызов должен получить model="gpt-4"
    assert calls[0]["model"] == "gpt-4"
    # Второй вызов (anthropic) должен получить model=None
    assert calls[1]["model"] is None


# ---------------------------------------------------------------------------
# НОВЫЕ ТЕСТЫ ДЛЯ ХЕЛПЕРОВ call_with_fallback
# ---------------------------------------------------------------------------


def test_resolve_provider_max_tokens_explicit():
    """Явно переданный max_tokens возвращается без вычислений."""
    result = _resolve_provider_max_tokens(
        provider_name="openrouter",
        model=None,
        prompt="test",
        source_text=None,
        explicit_max_tokens=1000,
    )
    assert result == 1000


def test_resolve_provider_max_tokens_with_source():
    """При наличии source_text вычисляется бюджет."""
    with (
        patch("src.llm_client.get_context_profile_from_env") as mock_profile,
        patch("src.llm_client.resolve_context_budget") as mock_budget,
    ):
        mock_profile.return_value = MagicMock(
            context_window=8192, safety_margin=512, mode="observe"
        )
        mock_budget.return_value = MagicMock(
            effective_output_tokens=768, was_capped=False, mode="observe"
        )

        result = _resolve_provider_max_tokens(
            provider_name="openrouter",
            model=None,
            prompt="test",
            source_text="source",
            explicit_max_tokens=None,
        )
        assert result == 768


def test_resolve_provider_max_tokens_zero_budget_skips():
    """Если effective_output_tokens <= 0, возвращается None."""
    with (
        patch("src.llm_client.get_context_profile_from_env") as mock_profile,
        patch("src.llm_client.resolve_context_budget") as mock_budget,
    ):
        mock_profile.return_value = MagicMock(
            context_window=8192, safety_margin=512, mode="enforce"
        )
        mock_budget.return_value = MagicMock(effective_output_tokens=0)

        result = _resolve_provider_max_tokens(
            provider_name="openrouter",
            model=None,
            prompt="test",
            source_text="source",
            explicit_max_tokens=None,
        )
        assert result is None


def test_resolve_provider_max_tokens_context_limit_skips():
    """LLMContextLimitError приводит к пропуску провайдера (None)."""
    with (
        patch("src.llm_client.get_context_profile_from_env") as mock_profile,
        patch("src.llm_client.resolve_context_budget") as mock_budget,
    ):
        mock_profile.return_value = MagicMock(
            context_window=8192, safety_margin=512, mode="enforce"
        )
        # Правильное создание исключения со всеми обязательными аргументами
        error = LLMContextLimitError(
            provider="openrouter",
            model="auto",
            input_tokens_estimate=1000,
            requested_output_tokens=100,
            available_output_tokens=0,
            context_window=8192,
            reason="insufficient_output_budget",
            mode="enforce",
        )
        mock_budget.side_effect = error

        result = _resolve_provider_max_tokens(
            provider_name="openrouter",
            model=None,
            prompt="test",
            source_text="source",
            explicit_max_tokens=None,
        )
        assert result is None


def test_resolve_provider_max_tokens_fallback_estimate():
    """Без source_text и без explicit используется estimate_max_tokens."""
    with patch("src.llm_client.estimate_max_tokens", return_value=999):
        result = _resolve_provider_max_tokens(
            provider_name="openrouter",
            model=None,
            prompt="test",
            source_text=None,
            explicit_max_tokens=None,
        )
        assert result == 999


@pytest.mark.asyncio
async def test_try_provider_success():
    """Успешный вызов провайдера возвращает LLMResponse."""
    mock_response = MagicMock(spec=LLMResponse)
    mock_client = AsyncMock()
    mock_client.generate = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("src.llm_client.create_llm_client", return_value=mock_client):
        response = await _try_provider(
            provider_enum=LLMProvider.OPENROUTER,
            model=None,
            prompt="test",
            temperature=0.3,
            max_retries=1,
            max_tokens=100,
        )
        assert response is mock_response


@pytest.mark.asyncio
async def test_try_provider_value_error():
    """ValueError (нет ключа) пробрасывается дальше."""
    with (
        patch("src.llm_client.create_llm_client", side_effect=ValueError("Missing key")),
        pytest.raises(ValueError),
    ):
        await _try_provider(
            provider_enum=LLMProvider.OPENROUTER,
            model=None,
            prompt="test",
            temperature=0.3,
            max_retries=1,
            max_tokens=100,
        )


@pytest.mark.asyncio
async def test_try_provider_llm_error():
    """LLMError пробрасывается."""
    with patch("src.llm_client.create_llm_client", side_effect=LLMError("API error")):
        with pytest.raises(LLMError):
            await _try_provider(
                provider_enum=LLMProvider.OPENROUTER,
                model=None,
                prompt="test",
                temperature=0.3,
                max_retries=1,
                max_tokens=100,
            )


def test_build_fallback_error_with_primary():
    """С первичной ошибкой заполняются все поля."""
    primary = LLMAPIError("Bad request", status_code=400)
    error = _build_fallback_error(
        primary_error=primary,
        primary_provider="openrouter",
        skipped_providers=["anthropic"],
        unknown_providers=["foo"],
        prompt_length=10,
    )
    assert error.kind == "upstream_error"  # 400 без context -> upstream_error
    assert error.provider == "openrouter"
    assert error.upstream_status == 400
    assert error.skipped_providers == ("anthropic",)
    assert error.unknown_providers == ("foo",)
    assert error.prompt_length == 10


def test_build_fallback_error_no_primary():
    """Без первичной ошибки kind = configuration."""
    error = _build_fallback_error(
        primary_error=None,
        primary_provider=None,
        skipped_providers=["openrouter"],
        unknown_providers=["foo"],
        prompt_length=5,
    )
    assert error.kind == "configuration"
    assert error.provider is None
    assert error.upstream_status is None
    assert error.skipped_providers == ("openrouter",)
    assert error.unknown_providers == ("foo",)


# ============================================================================
# Issue #22: safe diagnostics for invalid OpenRouter responses.
# ============================================================================


def _make_openrouter_client(
    *,
    max_tokens: int = 6000,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> OpenRouterClient:
    """Тестовый OpenRouterClient с фиксированной конфигурацией."""
    config = LLMConfig(
        provider=LLMProvider.OPENROUTER,
        model="openrouter/auto",
        api_key="test-key",
        max_tokens=max_tokens,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )
    return OpenRouterClient(config)


# --- _safe_str / _safe_int --------------------------------------------------


def test_diag_safe_str_none() -> None:
    assert _safe_str(None) is None


def test_diag_safe_str_short_string() -> None:
    assert _safe_str("hello") == "hello"


def test_diag_safe_str_long_string_truncated() -> None:
    result = _safe_str("x" * 100, max_len=10)
    assert result == "x" * 10 + "..."


def test_diag_safe_str_rejects_dict_and_list() -> None:
    assert _safe_str({"k": "v"}) is None
    assert _safe_str(["a", "b"]) is None
    assert _safe_str(("tuple",)) is None


def test_diag_safe_str_rejects_bool() -> None:
    assert _safe_str(True) is None
    assert _safe_str(False) is None


def test_diag_safe_str_accepts_numbers() -> None:
    assert _safe_str(42) == "42"
    assert _safe_str(3.5) == "3.5"


def test_diag_safe_int_rejects_bool() -> None:
    assert _safe_int(True) is None
    assert _safe_int(False) is None


def test_diag_safe_int_accepts_int() -> None:
    assert _safe_int(42) == 42


def test_diag_safe_int_accepts_float() -> None:
    assert _safe_int(3.7) == 3


def test_diag_safe_int_rejects_other_types() -> None:
    assert _safe_int("42") is None
    assert _safe_int(None) is None
    assert _safe_int([1]) is None


# --- _collect_response_diagnostics ------------------------------------------


def test_diag_collect_not_a_dict() -> None:
    diag = _collect_response_diagnostics(None, reason_code="X", requested_model="m", max_tokens=100)
    assert diag["reason_code"] == "X"
    assert diag["data_type"] == "NoneType"


def test_diag_collect_full_response() -> None:
    data = {
        "id": "resp-123",
        "model": "openai/gpt-4o-mini",
        "provider": "OpenAI",
        "choices": [
            {
                "finish_reason": "length",
                "message": {"content": "", "reasoning": "some reasoning here"},
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "completion_tokens_details": {"reasoning_tokens": 40},
        },
    }
    diag = _collect_response_diagnostics(
        data, reason_code="EMPTY_CONTENT", requested_model="auto", max_tokens=768
    )
    assert diag["response_id"] == "resp-123"
    assert diag["actual_model"] == "openai/gpt-4o-mini"
    assert diag["provider"] == "OpenAI"
    assert diag["choices_count"] == 1
    assert diag["finish_reason"] == "length"
    assert diag["has_message"] is True
    assert diag["content_type"] == "str"
    assert diag["content_length"] == 0
    assert diag["has_reasoning"] is True
    assert diag["reasoning_length"] == len("some reasoning here")
    assert diag["prompt_tokens"] == 100
    assert diag["completion_tokens"] == 50
    assert diag["reasoning_tokens"] == 40
    # Содержимое не должно попасть в диагностику:
    assert "some reasoning here" not in str(diag)


def test_diag_collect_choices_not_list() -> None:
    diag = _collect_response_diagnostics(
        {"choices": "not-a-list"}, reason_code="X", requested_model="m", max_tokens=100
    )
    assert diag["choices_count"] is None


def test_diag_collect_top_level_error() -> None:
    data = {
        "error": {
            "code": "invalid_api_key",
            "type": "authentication_error",
            "message": "SECRET_ERROR_MESSAGE_TEXT",
        },
        "choices": [],
    }
    diag = _collect_response_diagnostics(data, reason_code="X", requested_model="m", max_tokens=100)
    assert diag["has_top_level_error"] is True
    assert diag["error_code"] == "invalid_api_key"
    assert diag["error_type"] == "authentication_error"
    assert "SECRET_ERROR_MESSAGE_TEXT" not in str(diag)


# --- _format_diagnostics ----------------------------------------------------


def test_diag_format_with_dict() -> None:
    result = _format_diagnostics({"a": 1, "b": "x"}, attempt=2)
    assert "attempt=2" in result
    assert "a=1" in result
    assert "b=x" in result


def test_diag_format_none() -> None:
    assert _format_diagnostics(None, attempt=1) == "[attempt=1]"


# --- parse_response ---------------------------------------------------------


def test_diag_parse_valid_content() -> None:
    client = _make_openrouter_client()
    data = {"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]}
    resp = client.parse_response(data)
    assert resp.content == "hello"


def test_diag_parse_empty_string() -> None:
    client = _make_openrouter_client()
    data = {"choices": [{"message": {"content": ""}}]}
    with pytest.raises(LLMInvalidResponseError) as exc:
        client.parse_response(data)
    assert exc.value.reason_code == "EMPTY_CONTENT"
    assert exc.value.diagnostics is not None
    assert exc.value.diagnostics["reason_code"] == "EMPTY_CONTENT"


def test_diag_parse_whitespace_only() -> None:
    client = _make_openrouter_client()
    data = {"choices": [{"message": {"content": "   \n  "}}]}
    with pytest.raises(LLMInvalidResponseError) as exc:
        client.parse_response(data)
    assert exc.value.reason_code == "EMPTY_CONTENT"


def test_diag_parse_content_none() -> None:
    client = _make_openrouter_client()
    data = {"choices": [{"message": {"content": None}}]}
    with pytest.raises(LLMInvalidResponseError) as exc:
        client.parse_response(data)
    assert exc.value.reason_code == "EMPTY_CONTENT"


def test_diag_parse_missing_choices() -> None:
    client = _make_openrouter_client()
    with pytest.raises(LLMInvalidResponseError) as exc:
        client.parse_response({})
    assert exc.value.reason_code == "MISSING_CHOICES"


def test_diag_parse_empty_choices() -> None:
    client = _make_openrouter_client()
    with pytest.raises(LLMInvalidResponseError) as exc:
        client.parse_response({"choices": []})
    assert exc.value.reason_code == "MISSING_CHOICES"


def test_diag_parse_missing_message() -> None:
    client = _make_openrouter_client()
    data = {"choices": [{"finish_reason": "stop"}]}
    with pytest.raises(LLMInvalidResponseError) as exc:
        client.parse_response(data)
    assert exc.value.reason_code == "MISSING_MESSAGE"


def test_diag_parse_top_level_error() -> None:
    client = _make_openrouter_client()
    data = {
        "error": {"code": "x", "type": "y", "message": "SECRET_MSG"},
        "choices": [],
    }
    with pytest.raises(LLMInvalidResponseError) as exc:
        client.parse_response(data)
    assert exc.value.reason_code == "MISSING_CHOICES"
    assert exc.value.diagnostics is not None
    assert exc.value.diagnostics["has_top_level_error"] is True
    assert "SECRET_MSG" not in str(exc.value.diagnostics)


def test_diag_parse_length_finish_reason_empty_content() -> None:
    """Ключевой сценарий issue #22: finish_reason=length + пустой content."""
    client = _make_openrouter_client()
    reasoning_text = "thinking-step " * 50
    data = {
        "model": "some-reasoning-model",
        "choices": [
            {
                "finish_reason": "length",
                "message": {"content": "", "reasoning": reasoning_text},
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 768,
            "completion_tokens_details": {"reasoning_tokens": 750},
        },
    }
    with pytest.raises(LLMInvalidResponseError) as exc:
        client.parse_response(data)
    assert exc.value.reason_code == "EMPTY_CONTENT"
    diag = exc.value.diagnostics
    assert diag is not None
    assert diag["finish_reason"] == "length"
    assert diag["has_reasoning"] is True
    assert diag["reasoning_length"] == len(reasoning_text)
    assert diag["reasoning_tokens"] == 750
    assert "thinking-step" not in str(diag)


def test_diag_parse_malformed() -> None:
    client = _make_openrouter_client()
    # choices[0] не dict -> TypeError при проверке "message" in choice
    with pytest.raises(LLMInvalidResponseError) as exc:
        client.parse_response({"choices": [None]})
    assert exc.value.reason_code == "MALFORMED_RESPONSE"


# --- generate retry behavior ------------------------------------------------


@pytest.mark.asyncio
async def test_diag_generate_recovers_after_first_invalid() -> None:
    """Первая попытка invalid, вторая успешная."""
    client = _make_openrouter_client(max_tokens=100, max_retries=3, retry_delay=0.0)
    good = LLMResponse(content="ok", model="m", provider="openrouter")
    queue: list[object] = [
        LLMInvalidResponseError("EMPTY_CONTENT", diagnostics={"reason_code": "EMPTY_CONTENT"}),
        good,
    ]

    async def fake_call_api(prompt: str) -> LLMResponse:
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item  # type: ignore[return-value]

    try:
        with patch.object(client, "call_api", side_effect=fake_call_api):
            result = await client.generate("prompt")
        assert result is good
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_diag_generate_raises_after_all_invalid() -> None:
    """Все попытки invalid -> LLMInvalidResponseError прокидывается наружу."""
    client = _make_openrouter_client(max_tokens=100, max_retries=2, retry_delay=0.0)

    async def fake_call_api(prompt: str) -> LLMResponse:
        raise LLMInvalidResponseError("EMPTY_CONTENT", diagnostics={"reason_code": "EMPTY_CONTENT"})

    try:
        with patch.object(client, "call_api", side_effect=fake_call_api):
            with pytest.raises(LLMInvalidResponseError):
                await client.generate("prompt")
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_diag_generate_does_not_log_user_data(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Ключевой safety-тест: пользовательские данные не попадают в логи."""
    caplog.set_level(logging.WARNING)

    secret_prompt = "SECRET_PROMPT_TEXT_XYZ123"
    secret_reasoning = "SECRET_REASONING_TEXT_ABC456"

    client = _make_openrouter_client(max_tokens=100, max_retries=1, retry_delay=0.0)
    bad_data = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {"content": "", "reasoning": secret_reasoning},
            }
        ],
    }

    async def fake_post(*args: object, **kwargs: object) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = bad_data
        return mock_resp

    try:
        with patch.object(client.client, "post", side_effect=fake_post):
            with pytest.raises(LLMInvalidResponseError):
                await client.generate(secret_prompt)
    finally:
        await client.close()

    log_text = "\n".join(rec.message for rec in caplog.records)
    assert secret_prompt not in log_text
    assert secret_reasoning not in log_text
    # Диагностика при этом действительно залогирована:
    assert "reason_code=EMPTY_CONTENT" in log_text


# --- fallback integration ---------------------------------------------------


@pytest.mark.asyncio
async def test_diag_fallback_kind_is_invalid_response() -> None:
    """LLMInvalidResponseError -> kind=invalid_response в LLMFallbackError."""
    invalid = LLMInvalidResponseError("EMPTY_CONTENT", diagnostics={"reason_code": "EMPTY_CONTENT"})
    invalid_client = make_failing_client(invalid)

    def fake_create(provider: LLMProvider, **kwargs: object) -> object:
        if provider == LLMProvider.OPENROUTER:
            return invalid_client
        raise ValueError("Missing")

    with (
        patch("src.llm_client.create_llm_client", side_effect=fake_create),
        pytest.raises(LLMFallbackError) as exc_info,
    ):
        await call_with_fallback(
            prompt="test",
            providers=["openrouter", "anthropic"],
            max_retries_per_provider=1,
        )

    assert exc_info.value.kind == "invalid_response"


def test_diag_invalid_response_maps_to_502() -> None:
    """kind=invalid_response -> HTTPException 502 (контракт не изменился)."""
    from src.error_mapping import _llm_error_to_http_exception

    err = LLMFallbackError("invalid", kind="invalid_response")
    exc = _llm_error_to_http_exception(err)
    assert exc.status_code == 502


# --- robustness to unexpected field types -----------------------------------


@pytest.mark.parametrize(
    "weird_data",
    [
        None,
        "string",
        123,
        [],
        {"choices": "not-a-list"},
        {"choices": None},
        {"choices": [None]},
        {"choices": [{}]},
        {"choices": [{"message": None}]},
        {"choices": [{"message": {"content": 12345}}]},
        {"choices": [{"message": {"content": []}}]},
        {"choices": [{"message": {"content": {"nested": "dict"}}}]},
        {"choices": [{"message": {"content": "", "reasoning": 999}}]},
        {"choices": [{"message": {"content": ""}}], "usage": "not-a-dict"},
        {"choices": [{"message": {"content": ""}}], "usage": {"prompt_tokens": "not-int"}},
        {"choices": [{"message": {"content": ""}}], "error": "string-error"},
        {"choices": [{"message": {"content": ""}}], "provider": {"nested": "dict"}},
    ],
)
def test_diag_collect_never_raises(weird_data: object) -> None:
    """_collect_response_diagnostics не падает на любых неожиданных данных."""
    result = _collect_response_diagnostics(
        weird_data, reason_code="X", requested_model="m", max_tokens=100
    )
    assert isinstance(result, dict)
    assert result["reason_code"] == "X"
