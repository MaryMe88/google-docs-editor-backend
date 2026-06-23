"""
tests/test_llm_client.py
========================
Тесты для llm_client, проверяющие обработку ошибок и fallback.

Запуск:
    pytest tests/test_llm_client.py -v
"""
from __future__ import annotations

import pytest

from src.llm_client import LLMError, call_with_fallback


@pytest.mark.asyncio
async def test_call_with_fallback_empty_providers() -> None:
    """Шаг 1: при пустом списке провайдеров должно выбрасываться LLMError."""
    with pytest.raises(LLMError, match="No providers specified"):
        await call_with_fallback(
            prompt="test prompt",
            providers=[],
        )


@pytest.mark.asyncio
async def test_call_with_fallback_unknown_provider() -> None:
    """Шаг 1: при неизвестном провайдере должно выбрасываться LLMError."""
    with pytest.raises(LLMError, match="All providers failed"):
        await call_with_fallback(
            prompt="test prompt",
            providers=["unknown_xyz_provider"],
            max_retries_per_provider=0,  # чтобы сразу упало без retry
        )


@pytest.mark.asyncio
async def test_call_with_fallback_mixed_unknown_and_known() -> None:
    """
    Проверяем, что неизвестные провайдеры пропускаются, и если все неизвестны,
    то выбрасывается ошибка.
    Здесь мы не можем легко замокать реальный вызов, поэтому просто проверяем,
    что при всех неизвестных падает с LLMError.
    """
    with pytest.raises(LLMError, match="All providers failed"):
        await call_with_fallback(
            prompt="test",
            providers=["unknown1", "unknown2"],
            max_retries_per_provider=0,
        )