"""
llm_client.py

Клиенты для работы с LLM API:
- Perplexity
- OpenAI
- OpenRouter
- Anthropic

Поддерживает:
- единый async-интерфейс
- retry-логику
- контекстный менеджер async with
- нормализованный ответ LLMResponse
"""

from __future__ import annotations

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Поддерживаемые LLM-провайдеры."""

    PERPLEXITY = "perplexity"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"


@dataclass(frozen=True)
class LLMConfig:
    """Конфигурация клиента LLM."""

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
    """Нормализованный ответ от LLM."""

    content: str
    model: str
    provider: str
    tokens_used: Optional[int] = None
    finish_reason: Optional[str] = None


class LLMError(Exception):
    """Базовая ошибка LLM-слоя."""


class LLMAPIError(LLMError):
    """Ошибка API провайдера."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMTimeoutError(LLMError):
    """Таймаут запроса к LLM."""


class LLMRateLimitError(LLMError):
    """Rate limit от провайдера."""


class BaseLLMClient(ABC):
    """Базовый async-клиент для LLM."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.client = httpx.AsyncClient(timeout=config.timeout)

    async def __aenter__(self) -> "BaseLLMClient":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.client.aclose()

    async def close(self) -> None:
        """Явно закрывает HTTP-клиент."""
        await self.client.aclose()

    async def generate(self, prompt: str) -> LLMResponse:
        """Генерирует ответ с retry-логикой."""
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
                response = await self.call_api(prompt)
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
                delay = self.config.retry_delay * (2**attempt)
                logger.warning(
                    "Rate limit hit, retrying in %s seconds",
                    delay,
                    extra={"attempt": attempt + 1},
                )
                await asyncio.sleep(delay)

            except (LLMTimeoutError, httpx.TimeoutException) as error:
                last_error = error
                logger.warning(
                    "Timeout, retrying in %s seconds",
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
            extra={"last_error": str(last_error) if last_error else None},
        )
        raise LLMError(f"Failed after {self.config.max_retries} attempts") from last_error

    @abstractmethod
    async def call_api(self, prompt: str) -> LLMResponse:
        """Выполняет вызов API провайдера."""


class PerplexityClient(BaseLLMClient):
    """Клиент Perplexity API."""

    API_URL = "https://api.perplexity.ai/chat/completions"

    async def call_api(self, prompt: str) -> LLMResponse:
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
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

    def parse_response(self, data: Dict[str, Any]) -> LLMResponse:
        try:
            content = data["choices"][0]["message"]["content"]
            finish_reason = data["choices"][0].get("finish_reason")
            tokens_used = None
            if "usage" in data and isinstance(data["usage"], dict):
                tokens_used = data["usage"].get("total_tokens")

            return LLMResponse(
                content=content,
                model=self.config.model,
                provider=self.config.provider.value,
                tokens_used=tokens_used,
                finish_reason=finish_reason,
            )
        except (KeyError, IndexError, TypeError) as error:
            raise LLMError(f"Failed to parse response: {error}") from error

    @staticmethod
    def extract_error_message(response: httpx.Response) -> str:
        try:
            error_data = response.json()
            if "error" in error_data:
                if isinstance(error_data["error"], dict):
                    return error_data["error"].get("message", str(error_data["error"]))
                return str(error_data["error"])
            return response.text
        except Exception:
            return response.text


class OpenAIClient(BaseLLMClient):
    """Клиент OpenAI API."""

    API_URL = "https://api.openai.com/v1/chat/completions"

    async def call_api(self, prompt: str) -> LLMResponse:
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful writing assistant.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
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

    def parse_response(self, data: Dict[str, Any]) -> LLMResponse:
        try:
            content = data["choices"][0]["message"]["content"]
            finish_reason = data["choices"][0].get("finish_reason")
            tokens_used = None
            if "usage" in data and isinstance(data["usage"], dict):
                tokens_used = data["usage"].get("total_tokens")

            return LLMResponse(
                content=content,
                model=self.config.model,
                provider=self.config.provider.value,
                tokens_used=tokens_used,
                finish_reason=finish_reason,
            )
        except (KeyError, IndexError, TypeError) as error:
            raise LLMError(f"Failed to parse response: {error}") from error

    @staticmethod
    def extract_error_message(response: httpx.Response) -> str:
        try:
            error_data = response.json()
            if "error" in error_data:
                if isinstance(error_data["error"], dict):
                    return error_data["error"].get("message", str(error_data["error"]))
                return str(error_data["error"])
            return response.text
        except Exception:
            return response.text


class OpenRouterClient(BaseLLMClient):
    """Клиент OpenRouter API."""

    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    async def call_api(self, prompt: str) -> LLMResponse:
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://example.com"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "text-editor-api"),
        }
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful writing assistant.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
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

    def parse_response(self, data: Dict[str, Any]) -> LLMResponse:
        try:
            content = data["choices"][0]["message"]["content"]
            finish_reason = data["choices"][0].get("finish_reason")
            tokens_used = None
            if "usage" in data and isinstance(data["usage"], dict):
                tokens_used = data["usage"].get("total_tokens")

            return LLMResponse(
                content=content,
                model=self.config.model,
                provider=self.config.provider.value,
                tokens_used=tokens_used,
                finish_reason=finish_reason,
            )
        except (KeyError, IndexError, TypeError) as error:
            raise LLMError(f"Failed to parse response: {error}") from error

    @staticmethod
    def extract_error_message(response: httpx.Response) -> str:
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


class AnthropicClient(BaseLLMClient):
    """Клиент Anthropic API."""

    API_URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    async def call_api(self, prompt: str) -> LLMResponse:
        headers = {
            "x-api-key": self.config.api_key,
            "anthropic-version": self.API_VERSION,
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
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

    def parse_response(self, data: Dict[str, Any]) -> LLMResponse:
        try:
            content_blocks = data.get("content", [])
            text_chunks = []

            if isinstance(content_blocks, list):
                for block in content_blocks:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text")
                        if isinstance(text, str):
                            text_chunks.append(text)

            content = "\n".join(text_chunks).strip()
            if not content:
                raise LLMError("Anthropic response does not contain text content")

            usage = data.get("usage", {})
            tokens_used = None
            if isinstance(usage, dict):
                input_tokens = usage.get("input_tokens")
                output_tokens = usage.get("output_tokens")
                if isinstance(input_tokens, int) and isinstance(output_tokens, int):
                    tokens_used = input_tokens + output_tokens

            return LLMResponse(
                content=content,
                model=self.config.model,
                provider=self.config.provider.value,
                tokens_used=tokens_used,
                finish_reason=data.get("stop_reason"),
            )
        except (KeyError, IndexError, TypeError) as error:
            raise LLMError(f"Failed to parse response: {error}") from error

    @staticmethod
    def extract_error_message(response: httpx.Response) -> str:
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


def create_llm_client(
    provider: LLMProvider = LLMProvider.PERPLEXITY,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    apikey: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 4000,
    timeout: float = 60.0,
    max_retries: int = 3,
) -> BaseLLMClient:
    """
    Фабрика LLM-клиента.

    Важно:
    - поддерживает оба имени параметра: api_key и apikey (для обратной совместимости);
    - при отсутствии API key бросает ValueError;
    - возвращает объект, пригодный для `async with`.
    """
    # Обратная совместимость с старым именем аргумента
    if api_key is None and apikey is not None:
        api_key = apikey

    default_models = {
        LLMProvider.PERPLEXITY: "sonar-pro",
        LLMProvider.OPENAI: "gpt-4o-mini",
        LLMProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
        LLMProvider.OPENROUTER: "openrouter/auto",
    }

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
        if env_key is not None:
            api_key = os.getenv(env_key)

    if not api_key:
        env_key = env_keys.get(provider, "UNKNOWN_API_KEY")
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
        LLMProvider.ANTHROPIC: AnthropicClient,
        LLMProvider.OPENROUTER: OpenRouterClient,
    }

    client_class = client_classes.get(provider)
    if client_class is None:
        raise ValueError(f"Unsupported provider: {provider}")

    return client_class(config)


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


if __name__ == "__main__":
    import asyncio as _asyncio

    async def _test_client() -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        test_prompt = "Напиши короткий дружелюбный абзац о пользе хорошей редактуры."

        try:
            async with create_llm_client(
                provider=LLMProvider.OPENROUTER,
                model="openrouter/auto",
                temperature=0.3,
            ) as client:
                response = await client.generate(test_prompt)
                print()
                print(response.content)
                print(f"\nModel: {response.model}")
                print(f"Provider: {response.provider}")
                print(f"Tokens used: {response.tokens_used}")
        except Exception as error:  # noqa: BLE001
            print(f"LLM error: {error}")

    _asyncio.run(_test_client())