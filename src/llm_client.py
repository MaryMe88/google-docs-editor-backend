"""
llm_client.py

Модуль для взаимодействия с LLM API.
Поддерживает нескольких провайдеров через абстракцию,
обработку ошибок, retry-логику и логирование.
"""

from __future__ import annotations

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Type

import httpx
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Поддерживаемые провайдеры LLM."""

    PERPLEXITY = "perplexity"
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    ANTHROPIC = "anthropic"


@dataclass(frozen=True)
class LLMConfig:
    """Конфигурация для LLM-клиента."""

    provider: LLMProvider
    model: str
    api_key: str
    temperature: float = 0.3
    max_tokens: int = 4000
    timeout: float = 60.0
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass(frozen=True)
class LLMResponse:
    """Ответ от LLM."""

    content: str
    model: str
    provider: str
    tokens_used: Optional[int] = None
    finish_reason: Optional[str] = None


class LLMError(Exception):
    """Базовое исключение для ошибок LLM."""


class LLMAPIError(LLMError):
    """Ошибка API (4xx, 5xx)."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class LLMTimeoutError(LLMError):
    """Таймаут при запросе к LLM."""


class LLMRateLimitError(LLMError):
    """Превышен лимит запросов."""


class BaseLLMClient(ABC):
    """Абстрактный базовый класс для LLM-клиентов."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = httpx.AsyncClient(timeout=config.timeout)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._client.aclose()

    async def close(self) -> None:
        await self._client.aclose()

    async def generate(self, prompt: str) -> LLMResponse:
        attempt = 0
        last_error: Optional[Exception] = None

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
                    },
                )

                response = await self._call_api(prompt)

                logger.info(
                    "LLM request successful",
                    extra={
                        "provider": self.config.provider.value,
                        "model": self.config.model,
                        "response_length": len(response.content),
                        "tokens_used": response.tokens_used,
                    },
                )
                return response

            except LLMRateLimitError as error:
                last_error = error
                delay = self.config.retry_delay * (2 ** attempt)
                logger.warning(
                    "Rate limit hit, retrying in %ss",
                    delay,
                    extra={"attempt": attempt + 1},
                )
                await asyncio.sleep(delay)

            except (LLMTimeoutError, httpx.TimeoutException) as error:
                last_error = error
                logger.warning(
                    "Timeout, retrying in %ss",
                    self.config.retry_delay,
                    extra={"attempt": attempt + 1},
                )
                await asyncio.sleep(self.config.retry_delay)

            except LLMAPIError as error:
                last_error = error
                if error.status_code and 500 <= error.status_code < 600:
                    logger.warning(
                        "Server error %s, retrying",
                        error.status_code,
                        extra={"attempt": attempt + 1},
                    )
                    await asyncio.sleep(self.config.retry_delay)
                else:
                    raise

            attempt += 1

        logger.error(
            "All %s attempts failed",
            self.config.max_retries,
            extra={"last_error": str(last_error)},
        )
        raise LLMError(f"Failed after {self.config.max_retries} attempts") from last_error

    @abstractmethod
    async def _call_api(self, prompt: str) -> LLMResponse:
        raise NotImplementedError

    @staticmethod
    def _extract_error_message(response: httpx.Response) -> str:
        try:
            error_data = response.json()
            error = error_data.get("error")
            if isinstance(error, dict):
                return error.get("message", str(error))
            if error is not None:
                return str(error)
            return response.text
        except Exception:  # noqa: BLE001
            return response.text


class PerplexityClient(BaseLLMClient):
    """Клиент для Perplexity API."""

    API_URL = "https://api.perplexity.ai/chat/completions"

    async def _call_api(self, prompt: str) -> LLMResponse:
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        try:
            response = await self._client.post(self.API_URL, json=payload, headers=headers)
            if response.status_code == 429:
                raise LLMRateLimitError("Rate limit exceeded")
            if response.status_code >= 400:
                raise LLMAPIError(
                    f"API error: {self._extract_error_message(response)}",
                    status_code=response.status_code,
                )
            data = response.json()
            return self._parse_response(data)
        except httpx.TimeoutException as error:
            raise LLMTimeoutError("Request timed out") from error
        except httpx.HTTPError as error:
            raise LLMAPIError(f"HTTP error: {error}") from error

    def _parse_response(self, data: Dict[str, Any]) -> LLMResponse:
        try:
            return LLMResponse(
                content=data["choices"][0]["message"]["content"],
                model=self.config.model,
                provider=self.config.provider.value,
                tokens_used=data.get("usage", {}).get("total_tokens"),
                finish_reason=data["choices"][0].get("finish_reason"),
            )
        except (KeyError, IndexError) as error:
            raise LLMError(f"Failed to parse response: {error}") from error


class OpenAIClient(BaseLLMClient):
    """Клиент для OpenAI API."""

    API_URL = "https://api.openai.com/v1/chat/completions"

    async def _call_api(self, prompt: str) -> LLMResponse:
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        try:
            response = await self._client.post(self.API_URL, json=payload, headers=headers)
            if response.status_code == 429:
                raise LLMRateLimitError("Rate limit exceeded")
            if response.status_code >= 400:
                raise LLMAPIError(
                    f"API error: {self._extract_error_message(response)}",
                    status_code=response.status_code,
                )
            data = response.json()
            return self._parse_response(data)
        except httpx.TimeoutException as error:
            raise LLMTimeoutError("Request timed out") from error
        except httpx.HTTPError as error:
            raise LLMAPIError(f"HTTP error: {error}") from error

    def _parse_response(self, data: Dict[str, Any]) -> LLMResponse:
        try:
            return LLMResponse(
                content=data["choices"][0]["message"]["content"],
                model=self.config.model,
                provider=self.config.provider.value,
                tokens_used=data.get("usage", {}).get("total_tokens"),
                finish_reason=data["choices"][0].get("finish_reason"),
            )
        except (KeyError, IndexError) as error:
            raise LLMError(f"Failed to parse response: {error}") from error


class OpenRouterClient(BaseLLMClient):
    """Клиент для OpenRouter API."""

    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    async def _call_api(self, prompt: str) -> LLMResponse:
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://example.com"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "text-editor-docs"),
        }
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": "You are a helpful writing assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        try:
            response = await self._client.post(self.API_URL, json=payload, headers=headers)
            if response.status_code == 429:
                raise LLMRateLimitError("Rate limit exceeded")
            if response.status_code >= 400:
                raise LLMAPIError(
                    f"API error: {self._extract_error_message(response)}",
                    status_code=response.status_code,
                )
            data = response.json()
            return self._parse_response(data)
        except httpx.TimeoutException as error:
            raise LLMTimeoutError("Request timed out") from error
        except httpx.HTTPError as error:
            raise LLMAPIError(f"HTTP error: {error}") from error

    def _parse_response(self, data: Dict[str, Any]) -> LLMResponse:
        try:
            return LLMResponse(
                content=data["choices"][0]["message"]["content"],
                model=self.config.model,
                provider=self.config.provider.value,
                tokens_used=data.get("usage", {}).get("total_tokens"),
                finish_reason=data["choices"][0].get("finish_reason"),
            )
        except (KeyError, IndexError) as error:
            raise LLMError(f"Failed to parse response: {error}") from error


def create_llm_client(
    provider: LLMProvider = LLMProvider.PERPLEXITY,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 4000,
    timeout: float = 60.0,
    max_retries: int = 3,
) -> BaseLLMClient:
    """Фабрика LLM-клиентов."""
    default_models = {
        LLMProvider.PERPLEXITY: "sonar-pro",
        LLMProvider.OPENAI: "gpt-4o-mini",
        LLMProvider.OPENROUTER: "openrouter/auto",
        LLMProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
    }
    env_keys = {
        LLMProvider.PERPLEXITY: "PERPLEXITY_API_KEY",
        LLMProvider.OPENAI: "OPENAI_API_KEY",
        LLMProvider.OPENROUTER: "OPENROUTER_API_KEY",
        LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
    }
    client_classes: Dict[LLMProvider, Type[BaseLLMClient]] = {
        LLMProvider.PERPLEXITY: PerplexityClient,
        LLMProvider.OPENAI: OpenAIClient,
        LLMProvider.OPENROUTER: OpenRouterClient,
    }

    if provider not in client_classes:
        raise ValueError(
            f"Unsupported provider: {provider.value}. Supported: "
            f"{', '.join(item.value for item in client_classes)}"
        )

    resolved_model = model or default_models[provider]
    resolved_api_key = api_key or os.getenv(env_keys[provider])
    if not resolved_api_key:
        raise ValueError(
            "API key not provided and not found in environment variable "
            f"{env_keys[provider]}"
        )

    config = LLMConfig(
        provider=provider,
        model=resolved_model,
        api_key=resolved_api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
    )
    return client_classes[provider](config)


async def generate_text(
    prompt: str,
    provider: LLMProvider = LLMProvider.PERPLEXITY,
    model: Optional[str] = None,
    temperature: float = 0.3,
) -> str:
    """Упрощённый helper для генерации текста."""
    async with create_llm_client(
        provider=provider,
        model=model,
        temperature=temperature,
    ) as client:
        response = await client.generate(prompt)
    return response.content
