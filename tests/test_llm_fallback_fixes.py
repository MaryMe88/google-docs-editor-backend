"""Тесты фиксов 1-3: fallback при отсутствии ключа, null-content, rate-limit key.

Фикс 1: call_with_fallback пропускает провайдера, у которого нет API-ключа
        (create_llm_client бросает ValueError), и переходит к следующему.
Фикс 2: _OpenAICompatibleClient.parse_response фейлит LLMError при content=null
        или пустом content, вместо того чтобы вернуть None и упасть позже.
Фикс 3 (обновлён): ключ rate-limit берётся из реального IP клиента (request.client.host),
        а не из заголовка X-Forwarded-For (который может быть подделан).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.llm_client import (
    LLMError,
    LLMResponse,
    OpenAIClient,
    LLMConfig,
    call_with_fallback,
)
from src.provider_registry import LLMProvider


# ===========================================================================
# Фикс 1: fallback при отсутствии ключа (ValueError -> skip -> next provider)
# ===========================================================================
def _make_mock_client(content: str):
    """Async-context-manager клиент, чей generate() возвращает LLMResponse."""
    resp = LLMResponse(
        content=content, model="m", provider="openrouter", tokens_used=1
    )
    client = AsyncMock()
    client.generate = AsyncMock(return_value=resp)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


@pytest.mark.asyncio
async def test_fallback_skips_provider_without_key() -> None:
    """Первый провайдер без ключа (ValueError) пропускается, берётся второй."""
    good_client = _make_mock_client("Готовый текст.")

    def fake_create(provider, **kwargs):
        # Первый провайдер имитирует отсутствие ключа.
        if provider == LLMProvider.OPENAI:
            raise ValueError("API key not provided for OPENAI_API_KEY")
        return good_client

    with patch("src.llm_client.create_llm_client", side_effect=fake_create):
        resp = await call_with_fallback(
            prompt="тест",
            providers=["openai", "openrouter"],
            max_retries_per_provider=1,
        )
    assert resp.content == "Готовый текст."


@pytest.mark.asyncio
async def test_fallback_all_providers_without_key_raises_llmerror() -> None:
    """Если у всех провайдеров нет ключа — поднимается LLMError, не ValueError."""

    def fake_create(provider, **kwargs):
        raise ValueError("API key not provided")

    with patch("src.llm_client.create_llm_client", side_effect=fake_create):
        with pytest.raises(LLMError):
            await call_with_fallback(
                prompt="тест",
                providers=["openai", "anthropic"],
                max_retries_per_provider=1,
            )


# ===========================================================================
# Фикс 2: parse_response фейлит на content=null / пустом content
# ===========================================================================
def _openai_client() -> OpenAIClient:
    config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model="gpt-4o-mini",
        api_key="test-key",
    )
    return OpenAIClient(config)


def test_parse_response_null_content_raises() -> None:
    """content=null должен приводить к LLMError, а не к LLMResponse(content=None)."""
    client = _openai_client()
    data = {"choices": [{"message": {"content": None}, "finish_reason": "stop"}]}
    with pytest.raises(LLMError, match="empty or non-text"):
        client.parse_response(data)


def test_parse_response_empty_content_raises() -> None:
    """Пустая строка / пробелы тоже считаются невалидным ответом."""
    client = _openai_client()
    data = {"choices": [{"message": {"content": "   "}, "finish_reason": "stop"}]}
    with pytest.raises(LLMError, match="empty or non-text"):
        client.parse_response(data)


def test_parse_response_valid_content_ok() -> None:
    """Нормальный текст парсится без ошибок."""
    client = _openai_client()
    data = {
        "choices": [{"message": {"content": "Привет"}, "finish_reason": "stop"}],
        "usage": {"total_tokens": 5},
    }
    resp = client.parse_response(data)
    assert resp.content == "Привет"
    assert resp.tokens_used == 5


# ===========================================================================
# Фикс 3: rate-limit key_func использует реальный IP клиента (без X-Forwarded-For)
# ===========================================================================
def _fake_request(headers: dict, client_host: str | None):
    req = MagicMock()
    req.headers = headers
    if client_host is None:
        req.client = None
    else:
        req.client = MagicMock()
        req.client.host = client_host
    return req


def test_client_ip_key_uses_remote_address() -> None:
    """Ключ rate-limit должен быть реальным IP клиента (request.client.host)."""
    from src.main import _client_ip_key

    req = _fake_request(
        {"X-Forwarded-For": "203.0.113.7, 10.0.0.1"}, client_host="10.0.0.1"
    )
    assert _client_ip_key(req) == "10.0.0.1"


def test_client_ip_key_without_client() -> None:
    """Если request.client отсутствует, возвращается адрес get_remote_address (обычно '127.0.0.1')."""
    from src.main import _client_ip_key

    req = _fake_request({}, client_host=None)
    # get_remote_address вернёт '127.0.0.1' для пустого client
    assert _client_ip_key(req) == "127.0.0.1"


def test_client_ip_key_ignores_forwarded_for() -> None:
    """Проверяем, что подделанный X-Forwarded-For не влияет на ключ."""
    from src.main import _client_ip_key

    req_a = _fake_request({"X-Forwarded-For": "1.1.1.1"}, client_host="10.0.0.1")
    req_b = _fake_request({"X-Forwarded-For": "2.2.2.2"}, client_host="10.0.0.1")
    assert _client_ip_key(req_a) == _client_ip_key(req_b)