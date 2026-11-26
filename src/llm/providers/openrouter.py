"""Клиент OpenRouter; отключает reasoning для экономии бюджета.

Выделено из src/llm_client.py в итерации 17.1.a карты №3.
"""

from __future__ import annotations

import os
from typing import Any

from .openai_compat import _OpenAICompatibleClient


def _resolve_openrouter_reasoning_enabled() -> bool:
    """Return True if reasoning should be enabled.

    Default is False (reasoning disabled) to prevent the model from
    consuming the entire output budget on chain-of-thought.
    Env variable OPENROUTER_REASONING_ENABLED=true enables it back.
    """
    raw = os.getenv("OPENROUTER_REASONING_ENABLED", "false")
    return raw.lower() in ("1", "true", "yes", "on")


class OpenRouterClient(_OpenAICompatibleClient):
    """Клиент OpenRouter; отключает reasoning для экономии бюджета."""

    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def _build_headers(self) -> dict[str, str]:
        headers = super()._build_headers()
        headers["HTTP-Referer"] = os.getenv("OPENROUTER_SITE_URL", "")
        headers["X-Title"] = os.getenv("OPENROUTER_APP_NAME", "text-editor-api")
        return headers

    def _build_payload(self, prompt: str) -> dict[str, Any]:
        payload = super()._build_payload(prompt)
        if not _resolve_openrouter_reasoning_enabled():
            # OpenRouter native: reasoning disabled for all
            # supporting models. Prevents budget drain on
            # chain-of-thought.
            payload["reasoning"] = {"enabled": False}
        return payload
