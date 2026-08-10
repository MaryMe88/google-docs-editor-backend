"""
tests/test_llm_client.py
========================
Тесты для llm_client, проверяющие обработку ошибок и fallback.

Запуск:
    pytest tests/test_llm_client.py -v
"""
from __future__ import annotations

import pytest

from src.llm_client import (
    LLMError,
    call_with_fallback,
    estimate_max_tokens,
    _MIN_MAX_TOKENS,
    _MAX_MAX_TOKENS,
    _CHARS_PER_TOKEN,
)


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
    from src.llm_client import LLMFallbackError

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
    from src.llm_client import LLMFallbackError

    with pytest.raises(LLMFallbackError) as exc_info:
        await call_with_fallback(
            prompt="test",
            providers=["unknown1", "unknown2"],
            max_retries_per_provider=0,
        )
    assert exc_info.value.kind == "configuration"
    assert sorted(exc_info.value.unknown_providers) == ["unknown1", "unknown2"]


# ---------------------------------------------------------------------------
# Тесты для fallback с сохранением первичной ошибки
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fallback_primary_error_preserved_when_skipping_unconfigured() -> None:
    """
    Сценарий: первый провайдер (openrouter) возвращает LLMAPIError с 503,
    остальные провайдеры ненастроены (ValueError).
    Должен возникнуть LLMFallbackError с primary_provider == "openrouter"
    и kind == "upstream_error", skipped_providers содержит остальных.
    """
    from unittest.mock import AsyncMock, patch
    from src.llm_client import call_with_fallback, LLMAPIError, LLMFallbackError

    with patch("src.llm_client.create_llm_client") as mock_create:
        # Первый вызов — openrouter, возвращаем клиент, который при generate выдаёт ошибку
        mock_client_openrouter = AsyncMock()
        mock_client_openrouter.generate.side_effect = LLMAPIError("OpenRouter error", status_code=503)

        # Последующие вызовы — выбрасывают ValueError (ненастроены)
        mock_create.side_effect = [
            mock_client_openrouter,  # первый вызов: openrouter
            ValueError("Missing API key"),  # anthropic
            ValueError("Missing API key"),  # openai
            ValueError("Missing API key"),  # perplexity
        ]

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
        assert "anthropic" in error.skipped_providers
        assert "openai" in error.skipped_providers
        assert "perplexity" in error.skipped_providers
        assert error.unknown_providers == ()
        assert error.prompt_length == 4


@pytest.mark.asyncio
async def test_fallback_all_providers_unconfigured_raises_configuration() -> None:
    """
    Все провайдеры ненастроены → primary_error None → kind = "configuration".
    """
    from unittest.mock import patch
    from src.llm_client import call_with_fallback, LLMFallbackError

    with patch("src.llm_client.create_llm_client") as mock_create:
        mock_create.side_effect = [
            ValueError("Missing API key"),
            ValueError("Missing API key"),
            ValueError("Missing API key"),
        ]

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
        assert len(error.skipped_providers) == 3
        assert error.unknown_providers == ()


@pytest.mark.asyncio
async def test_fallback_http_429_classified_as_rate_limit() -> None:
    """HTTP 429 → kind = rate_limit."""
    from unittest.mock import patch, AsyncMock
    from src.llm_client import call_with_fallback, LLMAPIError, LLMFallbackError

    mock_client = AsyncMock()
    mock_client.generate.side_effect = LLMAPIError("Rate limit", status_code=429)

    with patch("src.llm_client.create_llm_client") as mock_create:
        mock_create.side_effect = [mock_client, ValueError("Missing")]

        with pytest.raises(LLMFallbackError) as exc_info:
            await call_with_fallback(
                prompt="test",
                providers=["openrouter", "anthropic"],
                max_retries_per_provider=1,
            )

        assert exc_info.value.kind == "rate_limit"


@pytest.mark.asyncio
async def test_fallback_timeout_classified_as_timeout() -> None:
    """LLMTimeoutError → kind = timeout."""
    from unittest.mock import patch, AsyncMock
    from src.llm_client import call_with_fallback, LLMTimeoutError, LLMFallbackError

    mock_client = AsyncMock()
    mock_client.generate.side_effect = LLMTimeoutError("Timeout")

    with patch("src.llm_client.create_llm_client") as mock_create:
        mock_create.side_effect = [mock_client, ValueError("Missing")]

        with pytest.raises(LLMFallbackError) as exc_info:
            await call_with_fallback(
                prompt="test",
                providers=["openrouter", "anthropic"],
                max_retries_per_provider=1,
            )

        assert exc_info.value.kind == "timeout"


@pytest.mark.asyncio
async def test_fallback_http_413_classified_as_context_limit() -> None:
    """HTTP 413 → kind = context_limit."""
    from unittest.mock import patch, AsyncMock
    from src.llm_client import call_with_fallback, LLMAPIError, LLMFallbackError

    mock_client = AsyncMock()
    mock_client.generate.side_effect = LLMAPIError("Too large", status_code=413)

    with patch("src.llm_client.create_llm_client") as mock_create:
        mock_create.side_effect = [mock_client, ValueError("Missing")]

        with pytest.raises(LLMFallbackError) as exc_info:
            await call_with_fallback(
                prompt="test",
                providers=["openrouter", "anthropic"],
                max_retries_per_provider=1,
            )

        assert exc_info.value.kind == "context_limit"


@pytest.mark.asyncio
async def test_fallback_http_400_without_context_words_not_context_limit() -> None:
    """HTTP 400 без упоминания контекста → upstream_error, не context_limit."""
    from unittest.mock import patch, AsyncMock
    from src.llm_client import call_with_fallback, LLMAPIError, LLMFallbackError

    mock_client = AsyncMock()
    mock_client.generate.side_effect = LLMAPIError("Bad request", status_code=400)

    with patch("src.llm_client.create_llm_client") as mock_create:
        mock_create.side_effect = [mock_client, ValueError("Missing")]

        with pytest.raises(LLMFallbackError) as exc_info:
            await call_with_fallback(
                prompt="test",
                providers=["openrouter", "anthropic"],
                max_retries_per_provider=1,
            )

        assert exc_info.value.kind == "upstream_error"


@pytest.mark.asyncio
async def test_fallback_model_passed_only_to_first_provider() -> None:
    """
    Проверяем, что model передаётся только первому провайдеру, остальным — None.
    Первый провайдер падает с LLMError, второй — ненастроен.
    """
    from unittest.mock import patch, AsyncMock
    from src.llm_client import call_with_fallback, LLMError, LLMFallbackError

    calls = []

    def side_effect(*args, **kwargs):
        calls.append((args, kwargs))
        if len(calls) == 1:
            # Первый провайдер — возвращаем клиент, который при generate падает
            mock_client = AsyncMock()
            mock_client.generate.side_effect = LLMError("First provider failed")
            return mock_client
        else:
            # Второй провайдер — ненастроен
            raise ValueError("Missing key")

    with patch("src.llm_client.create_llm_client", side_effect=side_effect):
        with pytest.raises(LLMFallbackError):
            await call_with_fallback(
                prompt="test",
                providers=["openrouter", "anthropic"],
                model="gpt-4",
                max_retries_per_provider=0,
            )

    # Первый вызов должен получить model="gpt-4"
    assert calls[0][1]["model"] == "gpt-4"
    # Второй вызов (anthropic) должен получить model=None
    assert calls[1][1]["model"] is None
