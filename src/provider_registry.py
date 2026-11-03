"""Реестр LLM-провайдеров. Не импортирует HTTP-клиенты."""

from enum import Enum


class LLMProvider(str, Enum):
    """Перечисление поддерживаемых LLM-провайдеров."""

    PERPLEXITY = "perplexity"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
