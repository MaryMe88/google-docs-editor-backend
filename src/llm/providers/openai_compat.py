"""OpenAI-совместимый клиент (базовый для Perplexity/OpenAI/OpenRouter).

Выделено из src/llm_client.py в итерации 17.1.a карты №3.
"""

from __future__ import annotations

from typing import Any

import httpx

from ..base import BaseLLMClient
from ..config import LLMResponse
from ..diagnostics import _collect_response_diagnostics
from ..errors import (
    LLMAPIError,
    LLMInvalidResponseError,
    LLMRateLimitError,
    LLMTimeoutError,
)


class _OpenAICompatibleClient(BaseLLMClient):
    API_URL: str = ""

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(self, prompt: str) -> dict[str, Any]:
        return {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

    async def call_api(self, prompt: str) -> LLMResponse:
        try:
            response = await self.client.post(
                self.API_URL,
                json=self._build_payload(prompt),
                headers=self._build_headers(),
            )

            if response.status_code == 429:
                raise LLMRateLimitError("Rate limit exceeded")

            if response.status_code >= 400:
                error_detail = self.extract_error_message(response)
                raise LLMAPIError(
                    f"API error: {error_detail}",
                    status_code=response.status_code,
                )

            return self.parse_response(response.json())

        except httpx.TimeoutException as error:
            raise LLMTimeoutError("Request timed out") from error
        except httpx.HTTPError as error:
            raise LLMAPIError(f"HTTP error: {error}") from error

    def parse_response(self, data: dict[str, Any]) -> LLMResponse:
        def _fail(reason_code: str) -> LLMInvalidResponseError:
            diagnostics = _collect_response_diagnostics(
                data,
                reason_code=reason_code,
                requested_model=self.config.model,
                max_tokens=self.config.max_tokens,
            )
            return LLMInvalidResponseError(reason_code, diagnostics=diagnostics)

        try:
            if (
                "choices" not in data
                or not isinstance(data["choices"], list)
                or len(data["choices"]) == 0
            ):
                raise _fail("MISSING_CHOICES")

            choice = data["choices"][0]
            if "message" not in choice or not isinstance(choice["message"], dict):
                raise _fail("MISSING_MESSAGE")

            message = choice["message"]
            content = message.get("content")
            finish_reason = choice.get("finish_reason")
            reasoning = message.get("reasoning")
            has_reasoning = isinstance(reasoning, str) and bool(reasoning.strip())

            # Issue #22: reasoning-модель израсходовала весь выходной
            # бюджет на chain-of-thought и обрубилась по лимиту,
            # оставив content пустым. Это более специфичный случай,
            # чем общий EMPTY_CONTENT.
            content_empty = content is None or (isinstance(content, str) and not content.strip())
            if content_empty and finish_reason == "length" and has_reasoning:
                raise _fail("REASONING_ONLY")

            if content is None:
                raise _fail("EMPTY_CONTENT")

            if not isinstance(content, str):
                raise _fail("NON_TEXT_CONTENT")

            if not content.strip():
                raise _fail("EMPTY_CONTENT")

            tokens_used = None
            if "usage" in data and isinstance(data["usage"], dict):
                tokens_used = data["usage"].get("total_tokens")

            actual_model = data.get("model", self.config.model)

            return LLMResponse(
                content=content,
                model=actual_model,
                provider=self.config.provider.value,
                tokens_used=tokens_used,
                finish_reason=finish_reason,
            )

        except (KeyError, IndexError, TypeError) as error:
            raise _fail("MALFORMED_RESPONSE") from error
