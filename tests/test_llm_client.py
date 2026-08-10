"""
tests/test_llm_client.py
========================
Тесты для llm_client, проверяющие обработку ошибок и fallback.

Запуск:
    pytest tests/test_llm_client.py -v
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.llm_client import (
    LLMError,
    LLMFallbackError,
    LLMAPIError,
    LLMTimeoutError,
    call_with_fallback,
    estimate_max_tokens,
    _MIN_MAX_TOKENS,
    _MAX_MAX_TOKENS,
    _CHARS_PER_TOKEN,
    LLMProvider,
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
    ):
        with pytest.raises(LLMFallbackError) as exc_info:
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
    ):
        with pytest.raises(LLMFallbackError) as exc_info:
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
    ):
        with pytest.raises(LLMFallbackError) as exc_info:
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
    ):
        with pytest.raises(LLMFallbackError) as exc_info:
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
    ):
        with pytest.raises(LLMFallbackError) as exc_info:
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
    ):
        with pytest.raises(LLMFallbackError) as exc_info:
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
    ):
        with pytest.raises(LLMFallbackError):
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
