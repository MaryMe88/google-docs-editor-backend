"""Характеристические тесты для BaseLLMClient.generate (итерация 18, PR-6).

Фиксируют retry-семантику generate до рефакторинга:
- 4xx (не 429) → немедленный raise без retry;
- 5xx на всех попытках → generic LLMError("Failed after N attempts").
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.llm_client import (
    LLMAPIError,
    LLMConfig,
    LLMError,
    OpenAIClient,
)
from src.provider_registry import LLMProvider


def _make_openai_client(max_retries: int) -> OpenAIClient:
    return OpenAIClient(
        LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini",
            api_key="test-key",
            max_retries=max_retries,
            retry_delay=0.0,
        )
    )


@pytest.mark.asyncio
async def test_generate_4xx_raises_without_retry() -> None:
    """LLMAPIError(status=400) — raise сразу, без sleep, без повторов."""
    client = _make_openai_client(max_retries=3)
    try:
        with (
            patch.object(
                client,
                "call_api",
                AsyncMock(side_effect=LLMAPIError("bad", status_code=400)),
            ) as mock_call,
            patch("src.llm_client.asyncio.sleep", AsyncMock()) as mock_sleep,
        ):
            with pytest.raises(LLMAPIError):
                await client.generate("prompt")
        assert mock_call.await_count == 1
        assert mock_sleep.await_count == 0
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_generate_5xx_exhausts_retries() -> None:
    """LLMAPIError(status=500) — N попыток, N-1 sleep, общий LLMError."""
    client = _make_openai_client(max_retries=3)
    try:
        with (
            patch.object(
                client,
                "call_api",
                AsyncMock(side_effect=LLMAPIError("err", status_code=500)),
            ) as mock_call,
            patch("src.llm_client.asyncio.sleep", AsyncMock()) as mock_sleep,
        ):
            with pytest.raises(LLMError, match="Failed after 3 attempts"):
                await client.generate("prompt")
        assert mock_call.await_count == 3
        assert mock_sleep.await_count == 2
    finally:
        await client.close()
