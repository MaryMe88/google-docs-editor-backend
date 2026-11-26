"""Конфигурация клиента и нормализованный ответ LLM.

Выделено из src/llm_client.py в итерации 17.1.a карты №3.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.provider_registry import LLMProvider


@dataclass(frozen=True)
class LLMConfig:
    """Конфигурация клиента LLM."""

    provider: LLMProvider
    model: str
    api_key: str
    temperature: float = 0.3
    max_tokens: int = 6000
    timeout: float = 60.0
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass(frozen=True)
class LLMResponse:
    """Нормализованный ответ от LLM."""

    content: str
    model: str
    provider: str
    tokens_used: int | None = None
    finish_reason: str | None = None
