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
    
    Все конкретные клиенты (Perplexity, OpenAI и т.д.) должны наследоваться
    от этого класса и реализовывать метод _call_api().
    """
    
    def __init__(self, config: LLMConfig):
        """
        Инициализирует LLM-клиент.
        
        Args:
            config: Конфигурация клиента
        """
        self.config = config
        self._client = httpx.AsyncClient(timeout=config.timeout)
    
    async def __aenter__(self):
        """Поддержка async context manager."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Закрывает HTTP-клиент при выходе из контекста."""
        await self._client.aclose()
    
    async def close(self):
        """Явное закрытие HTTP-клиента."""
        await self._client.aclose()
    
    async def generate(self, prompt: str) -> LLMResponse:
        """
        Генерирует ответ на промпт с retry-логикой.
        
        Args:
            prompt: Текст промпта
            
        Returns:
            Ответ от LLM
            
        Raises:
            LLMError: При ошибках генерации
        """
        attempt = 0
        last_error: Optional[Exception] = None
        
        while attempt < self.config.max_retries:
            try:
                logger.info(
                    f"LLM request attempt {attempt + 1}/{self.config.max_retries}",
                    extra={
                        "provider": self.config.provider.value,
                        "model": self.config.model,
                        "prompt_length": len(prompt),
                    }
                )
                
                response = await self._call_api(prompt)
                
                logger.info(
                    "LLM request successful",
                    extra={
                        "provider": self.config.provider.value,
                        "model": self.config.model,
                        "response_length": len(response.content),
                        "tokens_used": response.tokens_used,
                    }
                )
                
                return response
            
            except LLMRateLimitError as e:
                # Rate limit — ждём дольше перед повтором
                last_error = e
                delay = self.config.retry_delay * (2 ** attempt)
                logger.warning(
                    f"Rate limit hit, retrying in {delay}s",
                    extra={"attempt": attempt + 1}
                )
                time.sleep(delay)
            
            except (LLMTimeoutError, httpx.TimeoutException) as e:
                # Timeout — повторяем с небольшой задержкой
                last_error = e
                logger.warning(
                    f"Timeout, retrying in {self.config.retry_delay}s",
                    extra={"attempt": attempt + 1}
                )
                time.sleep(self.config.retry_delay)
            
            except LLMAPIError as e:
                # API ошибка — если 5xx, можно повторить
                last_error = e
                if e.status_code and 500 <= e.status_code < 600:
                    logger.warning(
                        f"Server error {e.status_code}, retrying",
                        extra={"attempt": attempt + 1}
                    )
                    time.sleep(self.config.retry_delay)
                else:
                    # 4xx — не повторяем
                    raise
            
            attempt += 1
        
        # Все попытки исчерпаны
        logger.error(
            f"All {self.config.max_retries} attempts failed",
            extra={"last_error": str(last_error)}
        )
        raise LLMError(f"Failed after {self.config.max_retries} attempts") from last_error
    
    @abstractmethod
    async def _call_api(self, prompt: str) -> LLMResponse:
        """
        Выполняет запрос к API конкретного провайдера.
        
        Должен быть реализован в подклассах.
        
        Args:
            prompt: Текст промпта
            
        Returns:
            Ответ от LLM
            
        Raises:
            LLMError: При ошибках API
        """
        pass


# ============================================================================
# Perplexity Client
# ============================================================================


class PerplexityClient(BaseLLMClient):
    """
    Клиент для Perplexity API.
    
    Документация: https://docs.perplexity.ai/
    """
    
    API_URL = "https://api.perplexity.ai/chat/completions"
    
    async def _call_api(self, prompt: str) -> LLMResponse:
        """Выполняет запрос к Perplexity API."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "user", "content": prompt}
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
            
            # Обработка rate limit
            if response.status_code == 429:
                raise LLMRateLimitError("Rate limit exceeded")
            
            # Обработка других ошибок
            if response.status_code >= 400:
                error_detail = self._extract_error_message(response)
                raise LLMAPIError(
                    f"API error: {error_detail}",
                    status_code=response.status_code
                )
            
            response.raise_for_status()
            data = response.json()
            
            return self._parse_response(data)
        
        except httpx.TimeoutException as e:
            raise LLMTimeoutError("Request timed out") from e
        
        except httpx.HTTPError as e:
            raise LLMAPIError(f"HTTP error: {str(e)}") from e
    
    def _parse_response(self, data: Dict[str, Any]) -> LLMResponse:
        """Парсит ответ от Perplexity API."""
        try:
            content = data["choices"][0]["message"]["content"]
            finish_reason = data["choices"][0].get("finish_reason")
            
            # Perplexity возвращает usage в некоторых случаях
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
        """Извлекает сообщение об ошибке из ответа."""
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
# OpenAI Client (для совместимости)
# ============================================================================


class OpenAIClient(BaseLLMClient):
    """
    Клиент для OpenAI API (совместимость).
    
    Можно использовать для тестирования или как альтернативу.
    """
    
    API_URL = "https://api.openai.com/v1/chat/completions"
    
    async def _call_api(self, prompt: str) -> LLMResponse:
        """Выполняет запрос к OpenAI API."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": prompt}
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
                    status_code=response.status_code
                )
            
            response.raise_for_status()
            data = response.json()
            
            return self._parse_response(data)
        
        except httpx.TimeoutException as e:
            raise LLMTimeoutError("Request timed out") from e
        
        except httpx.HTTPError as e:
            raise LLMAPIError(f"HTTP error: {str(e)}") from e
    
    def _parse_response(self, data: Dict[str, Any]) -> LLMResponse:
        """Парсит ответ от OpenAI API."""
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
        """Извлекает сообщение об ошибке из ответа."""
        try:
            error_data = response.json()
            if "error" in error_data:
                return error_data["error"].get("message", str(error_data["error"]))
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
    """
    Фабричная функция для создания LLM-клиента.
    
    Args:
        provider: Провайдер LLM
        model: Имя модели (если None, используется дефолтная для провайдера)
        api_key: API ключ (если None, берётся из переменных окружения)
        temperature: Температура генерации
        max_tokens: Максимальное количество токенов
        timeout: Таймаут запроса в секундах
        max_retries: Количество повторных попыток при ошибках
        
    Returns:
        Экземпляр LLM-клиента
        
    Raises:
        ValueError: Если не указан API ключ или неизвестный провайдер
    """
    # Определяем дефолтные модели
    default_models = {
        LLMProvider.PERPLEXITY: "sonar-pro",
        LLMProvider.OPENAI: "gpt-4",
        LLMProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
    }
    
    # Определяем имена переменных окружения для ключей
    env_keys = {
        LLMProvider.PERPLEXITY: "PERPLEXITY_API_KEY",
        LLMProvider.OPENAI: "OPENAI_API_KEY",
        LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
    }
    
    # Получаем модель
    if model is None:
        model = default_models.get(provider)
        if model is None:
            raise ValueError(f"No default model for provider: {provider}")
    
    # Получаем API ключ
    if api_key is None:
        env_key = env_keys.get(provider)
        if env_key:
            api_key = os.getenv(env_key)
        
        if not api_key:
            raise ValueError(
                f"API key not provided and not found in environment variable {env_key}"
            )
    
    # Создаём конфигурацию
    config = LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
    )
    
    # Создаём клиента
    client_classes = {
        LLMProvider.PERPLEXITY: PerplexityClient,
        LLMProvider.OPENAI: OpenAIClient,
    }
    
    client_class = client_classes.get(provider)
    if client_class is None:
        raise ValueError(f"Unsupported provider: {provider}")
    
    return client_class(config)


# ============================================================================
# Convenience Function (упрощённая функция для быстрого использования)
# ============================================================================


async def generate_text(
    prompt: str,
    provider: LLMProvider = LLMProvider.PERPLEXITY,
    model: Optional[str] = None,
    temperature: float = 0.3,
) -> str:
    """
    Упрощённая функция для быстрой генерации текста.
    
    Args:
        prompt: Промпт для генерации
        provider: Провайдер LLM
        model: Имя модели
        temperature: Температура генерации
        
    Returns:
        Сгенерированный текст
        
    Raises:
        LLMError: При ошибках генерации
    """
    async with create_llm_client(
        provider=provider,
        model=model,
        temperature=temperature
    ) as client:
        response = await client.generate(prompt)
        return response.content


# ============================================================================
# Main (для тестирования)
# ============================================================================


if __name__ == "__main__":
    import asyncio
    
    async def test_client():
        """Простой тест клиента."""
        # Настройка логирования для теста
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        test_prompt = """Ты — редактор текстов.
        
Исправь этот текст:
\"\"\"
Наш сервис является самым лучшым на рынке.
\"\"\"
"""
        
        try:
            # Вариант 1: Через фабрику с context manager
            async with create_llm_client(
                provider=LLMProvider.PERPLEXITY,
                temperature=0.3
            ) as client:
                response = await client.generate(test_prompt)
                print("=== Результат ===")
                print(response.content)
                print(f"\nМодель: {response.model}")
                print(f"Провайдер: {response.provider}")
                if response.tokens_used:
                    print(f"Токенов использовано: {response.tokens_used}")
            
            # Вариант 2: Упрощённая функция
            # result = await generate_text(test_prompt)
            # print(result)
        
        except LLMError as e:
            print(f"Ошибка LLM: {e}")
    
    # Запускаем тест
    asyncio.run(test_client())
