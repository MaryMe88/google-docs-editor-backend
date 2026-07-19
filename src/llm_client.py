"""
llm_client.py

Клиенты для работы с LLM API:
- Perplexity
- OpenAI
- OpenRouter
- Anthropic

Поддерживает:
- единый async-интерфейс
- retry-логику с экспоненциальным backoff и jitter
- контекстный менеджер async with
- нормализованный ответ LLMResponse
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv

from src.provider_registry import LLMProvider

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMConfig:
    """Конфигурация клиента LLM."""

    provider: LLMProvider
    model: str
    api_key: str
    temperature: float = 0.3
    max_tokens: int = 6000
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


def _backoff_with_jitter(base_delay: float, attempt: int) -> float:
    """
    Экспоненциальный backoff с полным jitter.

    Формула: uniform(0, base_delay * 2^attempt).
    Full jitter лучше рассеивает повторные запросы при пиковой нагрузке,
    чем равномерный или additive jitter.

    SEC: используется random.uniform — намеренно, не secrets.
    Jitter не является security-critical: его цель — рассеять
    нагрузку при retry, а не генерировать непредсказуемые токены.
    Криптографическая стойкость здесь не требуется.
    """
    cap = base_delay * (2 ** attempt)
    return random.uniform(0, cap)  # noqa: S311


# ---------------------------------------------------------------------------
# Адаптивный расчёт max_tokens
# ---------------------------------------------------------------------------
# Базовый дефолт поднят, потому что ответы в режиме text_and_report
# могут включать и сам текст, и краткий отчёт; на длинных промптах 4000
# токенов иногда оказывается маловато.
_DEFAULT_MAX_TOKENS = 6000
_MIN_MAX_TOKENS = 1536
_MAX_MAX_TOKENS = 12000

# Грубая эвристика для смешанного RU/EN текста.
_CHARS_PER_TOKEN = 4

# Во многих задачах редактуры ответ по длине сопоставим с входом, а иногда
# даже длиннее из-за более явных формулировок. Даём запас.
_RESPONSE_BUDGET_MULTIPLIER = 1.35
_RESPONSE_BUDGET_FLOOR = 2048


def estimate_max_tokens(prompt: str) -> int:
    """Оценивает разумный max_tokens исходя из длины промпта.

    Возвращает значение, ограниченное диапазоном
    [``_MIN_MAX_TOKENS``, ``_MAX_MAX_TOKENS``]. Цель — уменьшить риск
    обрезки длинных ответов и не раздувать лимит на коротких запросах.

    Args:
        prompt: финальный текст промпта, отправляемый модели.

    Returns:
        Рекомендуемое значение max_tokens.
    """
    prompt_length = max(len(prompt), 0)
    estimated_input_tokens = prompt_length // _CHARS_PER_TOKEN

    response_budget = int(estimated_input_tokens * _RESPONSE_BUDGET_MULTIPLIER)
    response_budget = max(response_budget, _RESPONSE_BUDGET_FLOOR)
    response_budget = max(response_budget, _DEFAULT_MAX_TOKENS)

    return min(max(response_budget, _MIN_MAX_TOKENS), _MAX_MAX_TOKENS)


class BaseLLMClient(ABC):
    """Базовый async-клиент для LLM."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.client = httpx.AsyncClient(timeout=config.timeout, trust_env=False)

    @staticmethod
    def extract_error_message(response: httpx.Response) -> str:
        """Извлекает человекочитаемое сообщение об ошибке из ответа провайдера."""
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
        except Exception:  # noqa: BLE001
            return f"HTTP {response.status_code}"

    async def __aenter__(self) -> "BaseLLMClient":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.client.aclose()

    async def close(self) -> None:
        """Явно закрывает HTTP-клиент."""
        await self.client.aclose()

    def _sleep_delay_for(self, attempt: int) -> Optional[float]:
        """Возвращает задержку перед следующей попыткой или None."""
        if attempt + 1 >= self.config.max_retries:
            return None
        return _backoff_with_jitter(self.config.retry_delay, attempt)

    async def generate(self, prompt: str) -> LLMResponse:
        """Генерирует ответ с retry-логикой и jitter."""
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
                        "max_tokens": self.config.max_tokens,
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
                        "finish_reason": response.finish_reason,
                    },
                )
                return response

            except LLMRateLimitError as error:
                last_error = error
                delay = self._sleep_delay_for(attempt)
                if delay is not None:
                    logger.warning(
                        "Rate limit hit, retrying in %.2f seconds",
                        delay,
                        extra={"attempt": attempt + 1},
                    )
                    await asyncio.sleep(delay)

            except (LLMTimeoutError, httpx.TimeoutException) as error:
                last_error = error
                delay = self._sleep_delay_for(attempt)
                if delay is not None:
                    logger.warning(
                        "Timeout, retrying in %.2f seconds",
                        delay,
                        extra={"attempt": attempt + 1},
                    )
                    await asyncio.sleep(delay)

            except LLMAPIError as error:
                last_error = error
                if error.status_code and 500 <= error.status_code < 600:
                    delay = self._sleep_delay_for(attempt)
                    if delay is not None:
                        logger.warning(
                            "Server error %s, retrying in %.2f seconds",
                            error.status_code,
                            delay,
                            extra={"attempt": attempt + 1},
                        )
                        await asyncio.sleep(delay)
                else:
                    raise

            attempt += 1

        logger.error(
            "All %s attempts failed",
            self.config.max_retries,
            extra={"last_error": type(last_error).__name__ if last_error else None},
        )
        raise LLMError(f"Failed after {self.config.max_retries} attempts") from last_error

    @abstractmethod
    async def call_api(self, prompt: str) -> LLMResponse:
        """Выполняет вызов API провайдера."""


class _OpenAICompatibleClient(BaseLLMClient):
    """Базовый клиент для провайдеров с OpenAI-совместимым API."""

    API_URL: str = ""

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(self, prompt: str) -> Dict[str, Any]:
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

    def parse_response(self, data: Dict[str, Any]) -> LLMResponse:
        try:
            content = data["choices"][0]["message"]["content"]
            finish_reason = data["choices"][0].get("finish_reason")

            tokens_used = None
            if "usage" in data and isinstance(data["usage"], dict):
                tokens_used = data["usage"].get("total_tokens")

            if not isinstance(content, str) or not content.strip():
                raise LLMError("Provider returned empty or non-text content")

            return LLMResponse(
                content=content,
                model=self.config.model,
                provider=self.config.provider.value,
                tokens_used=tokens_used,
                finish_reason=finish_reason,
            )

        except (KeyError, IndexError, TypeError) as error:
            raise LLMError(f"Failed to parse response: {error}") from error


class PerplexityClient(_OpenAICompatibleClient):
    """Клиент Perplexity API."""

    API_URL = "https://api.perplexity.ai/chat/completions"


class OpenAIClient(_OpenAICompatibleClient):
    """Клиент OpenAI API."""

    API_URL = "https://api.openai.com/v1/chat/completions"


class OpenRouterClient(_OpenAICompatibleClient):
    """Клиент OpenRouter API."""

    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def _build_headers(self) -> Dict[str, str]:
        headers = super()._build_headers()
        headers["HTTP-Referer"] = os.getenv("OPENROUTER_SITE_URL", "")
        headers["X-Title"] = os.getenv("OPENROUTER_APP_NAME", "text-editor-api")
        return headers


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

            return self.parse_response(response.json())

        except httpx.TimeoutException as error:
            raise LLMTimeoutError("Request timed out") from error
        except httpx.HTTPError as error:
            raise LLMAPIError(f"HTTP error: {error}") from error

    def parse_response(self, data: Dict[str, Any]) -> LLMResponse:
        try:
            content_blocks = data.get("content", [])
            text_chunks: List[str] = []

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


def create_llm_client(
    provider: LLMProvider = LLMProvider.PERPLEXITY,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    apikey: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    timeout: float = 60.0,
    max_retries: int = 3,
) -> BaseLLMClient:
    """
    Фабрика LLM-клиента.

    Важно:
    - поддерживает оба имени параметра: api_key и apikey;
    - при отсутствии API key бросает ValueError;
    - возвращает объект, пригодный для `async with`.
    """
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


async def call_with_fallback(
    prompt: str,
    providers: List[str],
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_retries_per_provider: int = 1,
    max_tokens: Optional[int] = None,
) -> LLMResponse:
    """Последовательно пробует провайдеров из списка.

    Возвращает первый успешный ответ. При исчерпании всех провайдеров
    поднимает LLMError с причиной последнего сбоя.
    """
    if max_tokens is None:
        max_tokens = estimate_max_tokens(prompt)

    if not providers:
        raise LLMError("No providers specified. Cannot execute LLM call.")

    last_error: Optional[Exception] = None

    for provider_name in providers:
        try:
            provider_enum = LLMProvider(provider_name)
        except ValueError:
            logger.warning("Unknown provider %r, skipping.", provider_name)
            continue

        try:
            logger.info(
                "call_with_fallback: trying provider=%s model=%s max_tokens=%s",
                provider_name,
                model,
                max_tokens,
            )
            async with create_llm_client(
                provider=provider_enum,
                model=model,
                temperature=temperature,
                max_retries=max_retries_per_provider,
                max_tokens=max_tokens,
            ) as client:
                response = await client.generate(prompt)
            logger.info(
                "call_with_fallback: success with provider=%s finish_reason=%s",
                provider_name,
                response.finish_reason,
            )
            return response

        except ValueError:
            logger.warning(
                "call_with_fallback: provider=%s unavailable (missing key or model), skipping",
                provider_name,
            )
            last_error = LLMError(f"Provider {provider_name} is not configured")

        except LLMError as error:
            logger.warning(
                "call_with_fallback: provider=%s failed: %s",
                provider_name,
                type(error).__name__,
            )
            last_error = error

    raise last_error or LLMError("All providers failed")


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
                print(f"Finish reason: {response.finish_reason}")
        except Exception as error:  # noqa: BLE001
            print(f"LLM error: {error}")

    _asyncio.run(_test_client())
