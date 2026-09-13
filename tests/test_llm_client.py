"""
tests/test_llm_client.py
========================
Тесты для llm_client, проверяющие обработку ошибок и fallback.

Запуск:
    pytest tests/test_llm_client.py -v
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.context_budget import LLMContextLimitError
from src.llm_client import (
    _CHARS_PER_TOKEN,
    _MAX_MAX_TOKENS,
    _MIN_MAX_TOKENS,
    LLMAPIError,
    LLMError,
    LLMFallbackError,
    LLMProvider,
    LLMResponse,
    LLMTimeoutError,
    _build_fallback_error,
    _resolve_provider_max_tokens,
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

    with patch(
        "src.llm_client.create_llm_client",
        side_effect=fake_create_llm_client,
    ), pytest.raises(LLMFallbackError) as exc_info:
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

    with patch(
        "src.llm_client.create_llm_client",
        side_effect=fake_create_llm_client,
    ), pytest.raises(LLMFallbackError) as exc_info:
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

    with patch(
        "src.llm_client.create_llm_client",
        side_effect=fake_create_llm_client,
    ), pytest.raises(LLMFallbackError) as exc_info:
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

    with patch(
        "src.llm_client.create_llm_client",
        side_effect=fake_create_llm_client,
    ), pytest.raises(LLMFallbackError) as exc_info:
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

    with patch(
        "src.llm_client.create_llm_client",
        side_effect=fake_create_llm_client,
    ), pytest.raises(LLMFallbackError) as exc_info:
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

    with patch(
        "src.llm_client.create_llm_client",
        side_effect=fake_create_llm_client,
    ), pytest.raises(LLMFallbackError) as exc_info:
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

    with patch(
        "src.llm_client.create_llm_client",
        side_effect=fake_create_llm_client,
    ), pytest.raises(LLMFallbackError):
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
    with patch(
        "src.llm_client.create_llm_client", side_effect=ValueError("Missing key")
    ), pytest.raises(ValueError):
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
