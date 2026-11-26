"""Базовый async-клиент LLM с retry-логикой.

Выделено из src/llm_client.py в итерации 17.1.a карты №3.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

import httpx

from ._constants import _CHARS_PER_TOKEN
from .config import LLMConfig, LLMResponse
from .diagnostics import _backoff_with_jitter, _format_diagnostics
from .errors import (
    LLMAPIError,
    LLMError,
    LLMInvalidResponseError,
    LLMRateLimitError,
    LLMTimeoutError,
)

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Базовый async-клиент для LLM."""

    def __init__(self, config: LLMConfig) -> None:
        """Инициализирует клиент с заданной конфигурацией и общим httpx-клиентом.

        Args:
            config: Конфигурация провайдера, модели и политики повторов.
        """
        self.config = config
        self.client = httpx.AsyncClient(timeout=config.timeout, trust_env=False)

    @staticmethod
    def extract_error_message(response: httpx.Response) -> str:
        """Извлекает человекочитаемое сообщение об ошибке из ответа провайдера.

        Returns:
            Текст из поля error.message; при отсутствии — HTTP-статус ответа.
        """
        try:
            error_data = response.json()
            if "error" in error_data:
                err = error_data["error"]
                if isinstance(err, dict):
                    return err.get("message", "API error")
                if isinstance(err, str):
                    return err
                return "API error"
            return f"HTTP {response.status_code}"
        except (ValueError, KeyError, TypeError):
            # Response body не является корректным JSON-объектом с ожидаемой
            # структурой. Возвращаем fallback-сообщение по HTTP-статусу.
            return f"HTTP {response.status_code}"

    async def __aenter__(self) -> BaseLLMClient:
        """Входит в асинхронный контекст; возвращает сам клиент."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Выходит из асинхронного контекста и закрывает httpx-клиент."""
        await self.client.aclose()

    async def close(self) -> None:
        """Закрывает httpx-клиент; идемпотентно."""
        await self.client.aclose()

    def _sleep_delay_for(self, attempt: int) -> float | None:
        if attempt + 1 >= self.config.max_retries:
            return None
        return _backoff_with_jitter(self.config.retry_delay, attempt)

    async def _retry_rate_limit(self, attempt: int) -> None:
        """Логирует rate limit и выжидает backoff (если это не последняя попытка)."""
        delay = self._sleep_delay_for(attempt)
        if delay is not None:
            logger.warning(
                "Rate limit hit, retrying in %.2f seconds",
                delay,
                extra={"attempt": attempt + 1},
            )
            await asyncio.sleep(delay)

    async def _retry_timeout(self, attempt: int) -> None:
        """Логирует таймаут и выжидает backoff."""
        delay = self._sleep_delay_for(attempt)
        if delay is not None:
            logger.warning(
                "Timeout, retrying in %.2f seconds",
                delay,
                extra={"attempt": attempt + 1},
            )
            await asyncio.sleep(delay)

    async def _retry_server_error(self, error: LLMAPIError, attempt: int) -> None:
        """5xx — логирует и выжидает backoff; иначе — raise немедленно."""
        if not (error.status_code and 500 <= error.status_code < 600):
            raise error
        delay = self._sleep_delay_for(attempt)
        if delay is not None:
            logger.warning(
                "Server error %s, retrying in %.2f seconds",
                error.status_code,
                delay,
                extra={"attempt": attempt + 1},
            )
            await asyncio.sleep(delay)

    async def _retry_invalid_response(self, error: LLMInvalidResponseError, attempt: int) -> None:
        """Retry при наличии delay; на последней попытке — log error и raise."""
        delay = self._sleep_delay_for(attempt)
        diag_str = _format_diagnostics(error.diagnostics, attempt=attempt + 1)
        if delay is not None:
            logger.warning(
                "Invalid response (empty or malformed), retrying in %.2f seconds %s",
                delay,
                diag_str,
            )
            await asyncio.sleep(delay)
            return
        logger.error(
            "Invalid response (empty or malformed), giving up %s",
            diag_str,
        )
        raise error

    async def generate(self, prompt: str) -> LLMResponse:
        """Выполняет запрос к провайдеру с повторами и экспоненциальной задержкой.

        Повторяет запрос при rate limit, таймауте, 5xx и некорректном ответе.
        На 4xx (кроме 429) поднимает LLMAPIError сразу, без повтора.

        Args:
            prompt: Пользовательский текст для генерации.

        Returns:
            Нормализованный ответ LLMResponse.

        Raises:
            LLMError: Все попытки исчерпаны без успеха.
            LLMAPIError: 4xx-ошибка (кроме rate limit) без повтора.
        """
        attempt = 0
        last_error: Exception | None = None

        while attempt < self.config.max_retries:
            try:
                logger.info(
                    "LLM request attempt %s/%s",
                    attempt + 1,
                    self.config.max_retries,
                    extra={
                        "provider": self.config.provider.value,
                        "model": self.config.model,
                        "prompt_length": len(prompt),
                        "max_tokens": self.config.max_tokens,
                    },
                )

                response = await self.call_api(prompt)

                logger.info(
                    "LLM request successful",
                    extra={
                        "provider": self.config.provider.value,
                        "model": self.config.model,
                        "actual_model": response.model,
                        "prompt_length": len(prompt),
                        "estimated_input_tokens": len(prompt) // _CHARS_PER_TOKEN,
                        "max_tokens": self.config.max_tokens,
                        "response_length": len(response.content),
                        "tokens_used": response.tokens_used,
                        "finish_reason": response.finish_reason,
                    },
                )
                return response

            except LLMRateLimitError as error:
                last_error = error
                await self._retry_rate_limit(attempt)

            except (LLMTimeoutError, httpx.TimeoutException) as error:
                last_error = error
                await self._retry_timeout(attempt)

            except LLMAPIError as error:
                last_error = error
                await self._retry_server_error(error, attempt)

            except LLMInvalidResponseError as error:
                last_error = error
                await self._retry_invalid_response(error, attempt)

            attempt += 1

        logger.error(
            "All %s attempts failed",
            self.config.max_retries,
            extra={"last_error": type(last_error).__name__ if last_error else None},
        )
        raise LLMError(f"Failed after {self.config.max_retries} attempts") from last_error

    @abstractmethod
    async def call_api(self, prompt: str) -> LLMResponse:
        """Отправляет запрос к API провайдера; реализуется в подклассах."""
