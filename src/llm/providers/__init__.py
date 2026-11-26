"""Провайдеры LLM: OpenAI-совместимые + Anthropic.

Выделено из src/llm_client.py в итерации 17.1.a карты №3.
"""

from __future__ import annotations

from .anthropic import AnthropicClient
from .openai import OpenAIClient
from .openai_compat import _OpenAICompatibleClient
from .openrouter import OpenRouterClient, _resolve_openrouter_reasoning_enabled
from .perplexity import PerplexityClient

__all__ = [
    "AnthropicClient",
    "OpenAIClient",
    "OpenRouterClient",
    "PerplexityClient",
    "_OpenAICompatibleClient",
    "_resolve_openrouter_reasoning_enabled",
]
