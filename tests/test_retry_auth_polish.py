"""Тесты фиксов 4-6: retry без лишнего sleep, парсер, soft-auth warning.

Фикс 4: generate() не спит перед выходом из цикла на последней попытке.
Фикс 5: _parse_text_and_report сохраняет отчёт при перевёрнутом порядке
        маркеров и штатно разбирает обычный порядок.
Фикс 6: verify_api_key логирует предупреждение (один раз) в soft-mode.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.llm_client import (
    LLMConfig,
    LLMError,
    LLMRateLimitError,
    OpenAIClient,
)
from src.provider_registry import LLMProvider


# ===========================================================================
# Фикс 4: нет лишнего sleep на последней попытке
# ===========================================================================
def _client(max_retries: int) -> OpenAIClient:
    return OpenAIClient(
        LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini",
            api_key="test-key",
            max_retries=max_retries,
            retry_delay=1.0,
        )
    )


@pytest.mark.asyncio
async def test_no_sleep_after_last_attempt() -> None:
    """При N попытках sleep вызывается ровно N-1 раз (не N)."""
    client = _client(max_retries=3)

    with patch.object(
        client, "call_api", AsyncMock(side_effect=LLMRateLimitError("429"))
    ), patch("src.llm_client.asyncio.sleep", AsyncMock()) as mock_sleep:
        with pytest.raises(LLMError, match="Failed after 3 attempts"):
            await client.generate("prompt")

    assert mock_sleep.await_count == 2  # 3 попытки -> 2 паузы между ними


@pytest.mark.asyncio
async def test_single_attempt_no_sleep() -> None:
    """При max_retries=1 sleep не вызывается вовсе."""
    client = _client(max_retries=1)

    with patch.object(
        client, "call_api", AsyncMock(side_effect=LLMRateLimitError("429"))
    ), patch("src.llm_client.asyncio.sleep", AsyncMock()) as mock_sleep:
        with pytest.raises(LLMError):
            await client.generate("prompt")

    assert mock_sleep.await_count == 0


# ===========================================================================
# Фикс 5: парсер сохраняет отчёт при любом порядке маркеров
# ===========================================================================
def test_parse_normal_order() -> None:
    from src.main import _parse_text_and_report

    raw = "===ТЕКСТ===\nОтредактированный текст.\n===ОТЧЁТ===\nмаркеры и ИП"
    text, report = _parse_text_and_report(raw)
    assert text == "Отредактированный текст."
    assert report == "маркеры и ИП"


def test_parse_reversed_order_keeps_report() -> None:
    from src.main import _parse_text_and_report

    raw = "===ОТЧЁТ===\nмаркеры и ИП\n===ТЕКСТ===\nОтредактированный текст."
    text, report = _parse_text_and_report(raw)
    assert text == "Отредактированный текст."
    assert report == "маркеры и ИП"  # раньше терялся -> None


def test_parse_text_only_no_report() -> None:
    from src.main import _parse_text_and_report

    raw = "===ТЕКСТ===\nТолько текст без отчёта."
    text, report = _parse_text_and_report(raw)
    assert text == "Только текст без отчёта."
    assert report is None


def test_parse_no_markers_returns_all_as_text() -> None:
    from src.main import _parse_text_and_report

    raw = "Ответ модели без маркеров."
    text, report = _parse_text_and_report(raw)
    assert text == "Ответ модели без маркеров."
    assert report is None


# ===========================================================================
# Фикс 6: soft-auth пишет предупреждение (один раз)
# ===========================================================================
def test_soft_auth_logs_warning_once(caplog) -> None:
    import src.auth as auth_mod

    # Сбрасываем флаг, чтобы тест был детерминированным.
    auth_mod._soft_auth_warned = False

    with patch.dict("os.environ", {}, clear=False):
        # Убеждаемся, что ключ не задан.
        import os

        os.environ.pop("API_SECRET_KEY", None)
        with caplog.at_level("WARNING"):
            auth_mod.verify_api_key(x_api_key=None)  # первый вызов -> warning
            auth_mod.verify_api_key(x_api_key=None)  # второй -> без нового warning

    warnings = [r for r in caplog.records if "API_SECRET_KEY" in r.message]
    assert len(warnings) == 1


def test_configured_auth_rejects_wrong_key() -> None:
    """С заданным ключом неверный X-API-Key даёт 401 (регресс-проверка)."""
    import os

    from fastapi import HTTPException

    import src.auth as auth_mod

    with patch.dict(os.environ, {"API_SECRET_KEY": "correct-key"}):
        with pytest.raises(HTTPException) as exc:
            auth_mod.verify_api_key(x_api_key="wrong-key")
        assert exc.value.status_code == 401
