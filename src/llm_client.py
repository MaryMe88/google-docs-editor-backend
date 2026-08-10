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
from typing import Any, Dict, List, Optional, Tuple

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


class LLMFallbackError(LLMError):
    """Ошибка при переборе провайдеров: все попытки неудачны."""

    def __init__(
        self,
        message: str,
        *,
        provider: Optional[str] = None,
        primary_error: Optional[LLMError] = None,
        skipped_providers: Tuple[str, ...] = (),
        unknown_providers: Tuple[str, ...] = (),
        prompt_length: int = 0,
        upstream_status: Optional[int] = None,
        kind: str = "unknown",
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.primary_error = primary_error
        self.skipped_providers = skipped_providers
        self.unknown_providers = unknown_providers
        self.prompt_length = prompt_length
        self.upstream_status = upstream_status
        self.kind = kind


class LLMInvalidResponseError(LLMError):
    """Провайдер вернул HTTP-успех, но ответ не годится для текстовой генерации."""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        if reason_code in ("EMPTY_CONTENT", "NON_TEXT_CONTENT"):
            message = f"Provider returned empty or non-text content ({reason_code})"
        else:
            message = f"Invalid LLM response: {reason_code}"
        super().__init__(message)


def _extract_upstream_status(error: LLMError) -> Optional[int]:
    """Извлекает HTTP статус из ошибки, если он присутствует."""
    if hasattr(error, "status_code") and isinstance(error.status_code, int):
        return error.status_code
    cause = error.__cause__
    if cause is not None and hasattr(cause, "status_code") and isinstance(cause.status_code, int):
        return cause.status_code
    return None


def _classify_error(error: LLMError) -> str:
    """Классифицирует ошибку для LLMFallbackError."""
    # Проверяем на invalid_response (приоритет выше общего upstream_error)
    if isinstance(error, LLMInvalidResponseError):
        return "invalid_response"
    # Проверяем цепочку причин
    if error.__cause__:
        cause = error.__cause__
        if isinstance(cause, LLMInvalidResponseError):
            return "invalid_response"
        if isinstance(cause, LLMError):
            return _classify_error(cause)

    if isinstance(error, LLMRateLimitError):
        return "rate_limit"
    if isinstance(error, LLMTimeoutError):
        return "timeout"
    if isinstance(error, LLMAPIError):
        status = _extract_upstream_status(error)
        # Проверяем rate_limit по статусу 429 (может быть обёрнут в LLMAPIError)
        if status == 429:
            return "rate_limit"
        if status == 401 or status == 403:
            return "authentication"
        if status == 413:
            return "context_limit"
        if status and 500 <= status < 600:
            return "upstream_error"
        if status == 400:
            msg = str(error).lower()
            if any(phrase in msg for phrase in (
                "context length", "context window", "maximum context",
                "max tokens", "token limit", "too many tokens"
            )):
                return "context_limit"
        return "upstream_error"
    # Проверяем цепочку причин для timeout
    if error.__cause__:
        cause = error.__cause__
        if isinstance(cause, httpx.TimeoutException):
            return "timeout"
    return "unknown"


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
_DEFAULT_MAX_TOKENS = 6000
_MIN_MAX_TOKENS = 1536
_MAX_MAX_TOKENS = 12000

_CHARS_PER_TOKEN = 4
_RESPONSE_BUDGET_MULTIPLIER = 1.35


def estimate_max_tokens(prompt: str) -> int:
    """Оценивает разумный max_tokens исходя из длины промпта.

    Для коротких промптов возвращает нижнюю границу. Для длинных —
    масштабирует бюджет ответа пропорционально оценке входа, но
    ограничивает результат диапазоном [_MIN_MAX_TOKENS, _MAX_MAX_TOKENS].
    """
    prompt_length = max(len(prompt), 0)
    estimated_input_tokens = prompt_length // _CHARS_PER_TOKEN
    response_budget = int(estimated_input_tokens * _RESPONSE_BUDGET_MULTIPLIER)

    if response_budget < _MIN_MAX_TOKENS:
        return _MIN_MAX_TOKENS

    if response_budget > _MAX_MAX_TOKENS:
        return _MAX_MAX_TOKENS

    return response_budget


class BaseLLMClient(ABC):
    """Базовый async-клиент для LLM."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.client = httpx.AsyncClient(timeout=config.timeout, trust_env=False)

    @staticmethod
    def extract_error_message(response: httpx.Response) -> str:
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
        except Exception:
            return f"HTTP {response.status_code}"

    async def __aenter__(self) -> BaseLLMClient:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.client.aclose()

    async def close(self) -> None:
        await self.client.aclose()

    def _sleep_delay_for(self, attempt: int) -> Optional[float]:
        if attempt + 1 >= self.config.max_retries:
            return None
        return _backoff_with_jitter(self.config.retry_delay, attempt)

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
        pass


class _OpenAICompatibleClient(BaseLLMClient):
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
            if "choices" not in data or not isinstance(data["choices"], list) or len(data["choices"]) == 0:
                raise LLMInvalidResponseError("MISSING_CHOICES")

            choice = data["choices"][0]
            if "message" not in choice or not isinstance(choice["message"], dict):
                raise LLMInvalidResponseError("MISSING_MESSAGE")

            content = choice["message"].get("content")

            if content is None:
                raise LLMInvalidResponseError("EMPTY_CONTENT")

            if not isinstance(content, str):
                raise LLMInvalidResponseError("NON_TEXT_CONTENT")

            if not content.strip():
                raise LLMInvalidResponseError("EMPTY_CONTENT")

            finish_reason = choice.get("finish_reason")

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
            raise LLMInvalidResponseError("MALFORMED_RESPONSE") from error


class PerplexityClient(_OpenAICompatibleClient):
    API_URL = "https://api.perplexity.ai/chat/completions"


class OpenAIClient(_OpenAICompatibleClient):
    API_URL = "https://api.openai.com/v1/chat/completions"


class OpenRouterClient(_OpenAICompatibleClient):
    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def _build_headers(self) -> Dict[str, str]:
        headers = super()._build_headers()
        headers["HTTP-Referer"] = os.getenv("OPENROUTER_SITE_URL", "")
        headers["X-Title"] = os.getenv("OPENROUTER_APP_NAME", "text-editor-api")
        return headers


class AnthropicClient(BaseLLMClient):
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

    def parse_response(self, data: Dict[str, Any]) -> LLMResponse:
        try:
            content_blocks = data.get("content", [])
            text_chunks: List[str] = []

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

            return LLMResponse(
                content=content,
                model=self.config.model,
                provider=self.config.provider.value,
                tokens_used=tokens_used,
                finish_reason=data.get("stop_reason"),
            )

        except (KeyError, IndexError, TypeError) as error:
            raise LLMInvalidResponseError("MALFORMED_RESPONSE") from error


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


async def call_with_fallback(
    prompt: str,
    providers: List[str],
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_retries_per_provider: int = 1,
    max_tokens: Optional[int] = None,
) -> LLMResponse:
    if max_tokens is None:
        max_tokens = estimate_max_tokens(prompt)

    if not providers:
        raise LLMError("No providers specified. Cannot execute LLM call.")

    primary_error: Optional[LLMError] = None
    primary_provider: Optional[str] = None
    skipped_providers: List[str] = []
    unknown_providers: List[str] = []

    for idx, provider_name in enumerate(providers):
        try:
            provider_enum = LLMProvider(provider_name)
        except ValueError:
            logger.warning("Unknown provider %r, skipping.", provider_name)
            unknown_providers.append(provider_name)
            continue

        model_for_this = model if idx == 0 else None

        try:
            logger.info(
                "call_with_fallback: trying provider=%s model=%s max_tokens=%s",
                provider_name,
                model_for_this,
                max_tokens,
            )
            async with create_llm_client(
                provider=provider_enum,
                model=model_for_this,
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
            skipped_providers.append(provider_name)
            continue

        except LLMError as error:
            logger.warning(
                "call_with_fallback: provider=%s failed: %s",
                provider_name,
                type(error).__name__,
            )
            if primary_error is None:
                primary_error = error
                primary_provider = provider_name
            continue

    # После исчерпания всех провайдеров — всегда выбрасываем LLMFallbackError
    kind = "unknown"
    upstream_status = None
    response_reason_code = None
    primary_error_type = None

    if primary_error is not None:
        kind = _classify_error(primary_error)
        upstream_status = _extract_upstream_status(primary_error)
        primary_error_type = type(primary_error).__name__
        if isinstance(primary_error, LLMInvalidResponseError):
            response_reason_code = primary_error.reason_code

        logger.warning(
            "All providers exhausted. primary_provider=%s, error_kind=%s, "
            "upstream_status=%s, primary_error_type=%s, response_reason_code=%s, "
            "skipped_providers=%s, unknown_providers=%s, prompt_length=%d",
            primary_provider,
            kind,
            upstream_status,
            primary_error_type,
            response_reason_code,
            skipped_providers,
            unknown_providers,
            len(prompt),
        )
    else:
        kind = "configuration"
        logger.warning(
            "All providers skipped or unknown. skipped_providers=%s, unknown_providers=%s, prompt_length=%d",
            skipped_providers,
            unknown_providers,
            len(prompt),
        )

    raise LLMFallbackError(
        f"All providers failed. Last error from {primary_provider if primary_provider else 'none'}",
        provider=primary_provider,
        primary_error=primary_error,
        skipped_providers=tuple(skipped_providers),
        unknown_providers=tuple(unknown_providers),
        prompt_length=len(prompt),
        upstream_status=upstream_status,
        kind=kind,
    )


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
        except Exception as error:
            print(f"LLM error: {error}")

    _asyncio.run(_test_client())
