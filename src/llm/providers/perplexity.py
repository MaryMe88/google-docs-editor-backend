"""Клиент Perplexity (OpenAI-совместимый API).

Выделено из src/llm_client.py в итерации 17.1.a карты №3.
"""

from __future__ import annotations

from .openai_compat import _OpenAICompatibleClient


class PerplexityClient(_OpenAICompatibleClient):
    """Клиент Perplexity (OpenAI-совместимый API)."""

    API_URL = "https://api.perplexity.ai/chat/completions"
