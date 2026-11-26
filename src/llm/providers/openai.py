"""Клиент OpenAI Chat Completions API.

Выделено из src/llm_client.py в итерации 17.1.a карты №3.
"""

from __future__ import annotations

from .openai_compat import _OpenAICompatibleClient


class OpenAIClient(_OpenAICompatibleClient):
    """Клиент OpenAI Chat Completions API."""

    API_URL = "https://api.openai.com/v1/chat/completions"
