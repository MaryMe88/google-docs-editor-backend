"""
llm_client.py

Модуль для взаимодействия с LLM API.
Поддерживает различных провайдеров через абстракцию,
обработку ошибок, retry-логику и логирование.
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

import httpx
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# Настраиваем логирование
logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================


class LLMProvider(str, Enum):
    """Поддерживаемые провайдеры LLM."""
    PERPLEXITY = "perplexity"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"


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
    pass


class LLMAPIError(LLMError):
    """Ошибка API (4xx, 5xx)."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class LLMTimeoutError(LLMError):
    """Таймаут при запросе к LLM."""
    pass


class LLMRateLimitError(LLMError):
    """Превышен лимит запросов."""
    pass


# ============================================================================
# Abstract Base Client
# ============================================================================


class BaseLLMClient(ABC):
    """
    Абстрактный базовый класс для LLM-клиентов.
    Все конкретные клиенты должны реализовывать _call_api().
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = httpx.AsyncClient(timeout=config.timeout)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._client.aclose()

    async def close(self):
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

            except LLMRateLimitError as e:
                last_error = e
                delay = self.config.retry_delay * (2 ** attempt)
                logger.warning(
                    "Rate limit hit, retrying in %ss",
                    delay,
                    extra={"attempt": attempt + 1},
                )
                time.sleep(delay)

            except (LLMTimeoutError, httpx.TimeoutException) as e:
                last_error = e
                logger.warning(
                    "Timeout, retrying in %ss",
                    self.config.retry_delay,
                    extra={"attempt": attempt + 1},
                )
                time.sleep(self.config.retry_delay)

            except LLMAPIError as e:
                last_error = e
                if e.status_code and 500 <= e.status_code < 600:
                    logger.warning(
                        "Server error %s, retrying",
                        e.status_code,
                        extra={"attempt": attempt + 1},
                    )
                    time.sleep(self.config.retry_delay)
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
        pass


# ============================================================================
# Perplexity Client
# ============================================================================


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
                error_detail = self._extract_error_message(response)
                raise LLMAPIError(
                    f"API error: {error_detail}",
                    status_code=response.status_code,
                )

            response.raise_for_status()
            data = response.json()
            return self._parse_response(data)

        except httpx.TimeoutException as e:
            raise LLMTimeoutError("Request timed out") from e

        except httpx.HTTPError as e:
            raise LLMAPIError(f"HTTP error: {str(e)}") from e

    def _parse_response(self, data: Dict[str, Any]) -> LLMResponse:
        try:
            content = data["choices"][0]["message"]["content"]
            finish_reason = data["choices"][0].get("finish_reason")

            tokens_used = None
            if "usage" in data:
                tokens_used = data["usage"].get("total_tokens")

            return LLMResponse(
                content=content,
                model=self.config.model,
                provider=self.config.provider.value,
                tokens_used=tokens_used,
                finish_reason=finish_reason,
            )

        except (KeyError, IndexError) as e:
            raise LLMError(f"Failed to parse response: {str(e)}") from e

    def _extract_error_message(self, response: httpx.Response) -> str:
        try:
            error_data = response.json()
            if "error" in error_data:
                if isinstance(error_data["error"], dict):
                    return error_data["error"].get("message", str(error_data["error"]))
                return str(error_data["error"])
            return response.text
        except Exception:
            return response.text


# ============================================================================
# OpenAI Client
# ============================================================================


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
            "messages": [{"role": "system", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        try:
            response = await self._client.post(self.API_URL, json=payload, headers=headers)

            if response.status_code == 429:
                raise LLMRateLimitError("Rate limit exceeded")

            if response.status_code >= 400:
                error_detail = self._extract_error_message(response)
                raise LLMAPIError(
                    f"API error: {error_detail}",
                    status_code=response.status_code,
                )

            response.raise_for_status()
            data = response.json()
            return self._parse_response(data)

        except httpx.TimeoutException as e:
            raise LLMTimeoutError("Request timed out") from e

        except httpx.HTTPError as e:
            raise LLMAPIError(f"HTTP error: {str(e)}") from e

    def _parse_response(self, data: Dict[str, Any]) -> LLMResponse:
        try:
            content = data["choices"][0]["message"]["content"]
            finish_reason = data["choices"][0].get("finish_reason")
            tokens_used = data.get("usage", {}).get("total_tokens")

            return LLMResponse(
                content=content,
                model=self.config.model,
                provider=self.config.provider.value,
                tokens_used=tokens_used,
                finish_reason=finish_reason,
            )

        except (KeyError, IndexError) as e:
            raise LLMError(f"Failed to parse response: {str(e)}") from e

    def _extract_error_message(self, response: httpx.Response) -> str:
        try:
            error_data = response.json()
            if "error" in error_data:
                return error_data["error"].get("message", str(error_data["error"]))
            return response.text
        except Exception:
            return response.text


# ============================================================================
# OpenRouter Client
# ============================================================================


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
            response = await self._client.post(
                self.API_URL,
                json=payload,
                headers=headers,
            )

            if response.status_code == 429:
                raise LLMRateLimitError("Rate limit exceeded")

            if response.status_code >= 400:
                error_detail = self._extract_error_message(response)
                raise LLMAPIError(
                    f"API error: {error_detail}",
                    status_code=response.status_code,
                )

            response.raise_for_status()
            data = response.json()
            return self._parse_response(data)

        except httpx.TimeoutException as e:
            raise LLMTimeoutError("Request timed out") from e

        except httpx.HTTPError as e:
            raise LLMAPIError(f"HTTP error: {str(e)}") from e

    def _parse_response(self, data: Dict[str, Any]) -> LLMResponse:
        try:
            content = data["choices"][0]["message"]["content"]
            finish_reason = data["choices"][0].get("finish_reason")
            tokens_used = data.get("usage", {}).get("total_tokens")

            return LLMResponse(
                content=content,
                model=self.config.model,
                provider=self.config.provider.value,
                tokens_used=tokens_used,
                finish_reason=finish_reason,
            )
        except (KeyError, IndexError) as e:
            raise LLMError(f"Failed to parse response: {str(e)}") from e

    def _extract_error_message(self, response: httpx.Response) -> str:
        try:
            error_data = response.json()
            if "error" in error_data:
                err = error_data["error"]
                if isinstance(err, dict):
                    return err.get("message", str(err))
                return str(err)
            return response.text
        except Exception:
            return response.text


# ============================================================================
# Factory Function
# ============================================================================


def create_llm_client(
    provider: LLMProvider = LLMProvider.PERPLEXITY,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 4000,
    timeout: float = 60.0,
    max_retries: int = 3,
) -> BaseLLMClient:
    # дефолтные модели
    default_models = {
        LLMProvider.PERPLEXITY: "sonar-pro",
        LLMProvider.OPENAI: "gpt-4",
        LLMProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
        LLMProvider.OPENROUTER: "openrouter/auto",
    }

    # env-переменные с ключами
    env_keys = {
        LLMProvider.PERPLEXITY: "PERPLEXITY_API_KEY",
        LLMProvider.OPENAI: "OPENAI_API_KEY",
        LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
        LLMProvider.OPENROUTER: "OPENROUTER_API_KEY",
    }

    if model is None:
        model = default_models.get(provider)
        if model is None:
            raise ValueError(f"No default model for provider: {provider}")

    if api_key is None:
        env_key = env_keys.get(provider)
        if env_key:
            api_key = os.getenv(env_key)
        if not api_key:
            raise ValueError(
                f"API key not provided and not found in environment variable {env_key}"
            )

    config = LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
    )

    client_classes = {
        LLMProvider.PERPLEXITY: PerplexityClient,
        LLMProvider.OPENAI: OpenAIClient,
        LLMProvider.OPENROUTER: OpenRouterClient,
    }

    client_class = client_classes.get(provider)
    if client_class is None:
        raise ValueError(f"Unsupported provider: {provider}")

    return client_class(config)


# ============================================================================
# Convenience Function
# ============================================================================


async def generate_text(
    prompt: str,
    provider: LLMProvider = LLMProvider.PERPLEXITY,
    model: Optional[str] = None,
    temperature: float = 0.3,
) -> str:
    async with create_llm_client(
        provider=provider,
        model=model,
        temperature=temperature,
    ) as client:
        response = await client.generate(prompt)
        return response.content


if __name__ == "__main__":
    import asyncio

    async def test_client():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        test_prompt = "Скажи одну фразу о том, что редактор работает."

        try:
            async with create_llm_client(
                provider=LLMProvider.OPENROUTER,
                model="openrouter/auto",
                temperature=0.3,
            ) as client:
                response = await client.generate(test_prompt)
                print("=== Результат ===")
                print(response.content)
                print(f"\nМодель: {response.model}")
                print(f"Провайдер: {response.provider}")
        except LLMError as e:
            print(f"Ошибка LLM: {e}")

    asyncio.run(test_client())
