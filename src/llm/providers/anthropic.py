"""Клиент Anthropic Messages API.

Выделено из src/llm_client.py в итерации 17.1.a карты №3.
"""

from __future__ import annotations

from typing import Any

import httpx

from ..base import BaseLLMClient
from ..config import LLMResponse
from ..errors import (
    LLMAPIError,
    LLMInvalidResponseError,
    LLMRateLimitError,
    LLMTimeoutError,
)


class AnthropicClient(BaseLLMClient):
    """Клиент Anthropic Messages API."""

    API_URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    async def call_api(self, prompt: str) -> LLMResponse:
        """Отправляет запрос в Messages API и нормализует ответ.

        Raises:
            LLMRateLimitError: HTTP 429.
            LLMAPIError: Прочие HTTP-ошибки и сетевые сбои.
            LLMTimeoutError: Таймаут запроса.
        """
        headers = {
            "x-api-key": self.config.api_key,
            "anthropic-version": self.API_VERSION,
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

        try:
            response = await self.client.post(
                self.API_URL,
                json=payload,
                headers=headers,
            )

            if response.status_code == 429:
                raise LLMRateLimitError("Rate limit exceeded")

            if response.status_code >= 400:
                error_detail = self.extract_error_message(response)
                raise LLMAPIError(
                    f"API error: {error_detail}",
                    status_code=response.status_code,
                )

            data = response.json()
            return self.parse_response(data)

        except httpx.TimeoutException as error:
            raise LLMTimeoutError("Request timed out") from error
        except httpx.HTTPError as error:
            raise LLMAPIError(f"HTTP error: {error}") from error

    def parse_response(self, data: dict[str, Any]) -> LLMResponse:
        """Собирает текстовые блоки Anthropic в один LLMResponse.

        Raises:
            LLMInvalidResponseError: Нет текстовых блоков или ответ повреждён.
        """
        try:
            content_blocks = data.get("content", [])
            text_chunks: list[str] = []

            if not isinstance(content_blocks, list):
                raise LLMInvalidResponseError("MALFORMED_RESPONSE")

            for block in content_blocks:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text = block.get("text")
                    if isinstance(text, str):
                        text_chunks.append(text)

            content = "\n".join(text_chunks).strip()

            if not content:
                raise LLMInvalidResponseError("EMPTY_CONTENT")

            usage = data.get("usage", {})
            tokens_used = None
            if isinstance(usage, dict):
                input_tokens = usage.get("input_tokens")
                output_tokens = usage.get("output_tokens")
                if isinstance(input_tokens, int) and isinstance(output_tokens, int):
                    tokens_used = input_tokens + output_tokens

            actual_model = data.get("model", self.config.model)

            return LLMResponse(
                content=content,
                model=actual_model,
                provider=self.config.provider.value,
                tokens_used=tokens_used,
                finish_reason=data.get("stop_reason"),
            )

        except (KeyError, IndexError, TypeError) as error:
            raise LLMInvalidResponseError("MALFORMED_RESPONSE") from error
