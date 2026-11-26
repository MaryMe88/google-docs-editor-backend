"""Классы ошибок LLM-слоя.

Выделено из src/llm_client.py в итерации 17.1.a карты №3.
"""

from __future__ import annotations

from typing import Any


class LLMError(Exception):
    """Базовая ошибка LLM-слоя."""


class LLMAPIError(LLMError):
    """Ошибка API провайдера."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        """Инициализирует ошибку API провайдера.

        Args:
            message: Текст ошибки.
            status_code: HTTP-статус ответа, если известен.
        """
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
        provider: str | None = None,
        primary_error: LLMError | None = None,
        skipped_providers: tuple[str, ...] = (),
        unknown_providers: tuple[str, ...] = (),
        prompt_length: int = 0,
        upstream_status: int | None = None,
        kind: str = "unknown",
    ) -> None:
        """Инициализирует ошибку fallback с техническими метаданными попытки.

        Args:
            message: Текст ошибки.
            provider: Имя провайдера, на котором произошёл отказ.
            primary_error: Исходная ошибка основного провайдера.
            skipped_providers: Провайдеры, пропущенные из-за отсутствия ключа.
            unknown_providers: Неизвестные имена провайдеров.
            prompt_length: Длина исходного промпта в символах.
            upstream_status: HTTP-статус последней ошибки вышестоящего сервиса.
            kind: Классификация ошибки (rate_limit, timeout и т. п.).
        """
        super().__init__(message)
        self.provider = provider
        self.primary_error = primary_error
        self.skipped_providers = skipped_providers
        self.unknown_providers = unknown_providers
        self.prompt_length = prompt_length
        self.upstream_status = upstream_status
        self.kind = kind


class LLMInvalidResponseError(LLMError):
    """Провайдер вернул HTTP-успех, но ответ не годится для текстовой генерации.

    ``diagnostics`` — безопасные технические метаданные (id, модель, типы,
    длины, finish_reason, счётчики токенов). Никогда не содержит
    пользовательского текста, промпта или содержимого ответа.
    """

    def __init__(
        self,
        reason_code: str,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        """Инициализирует ошибку некорректного ответа.

        Args:
            reason_code: Код причины (REASONING_ONLY, EMPTY_CONTENT, ...).
            diagnostics: Безопасные технические метаданные ответа без текста.
        """
        self.reason_code = reason_code
        self.diagnostics = diagnostics
        if reason_code == "REASONING_ONLY":
            message = (
                "Provider consumed the entire output budget on reasoning "
                "without producing content (REASONING_ONLY)"
            )
        elif reason_code in ("EMPTY_CONTENT", "NON_TEXT_CONTENT"):
            message = f"Provider returned empty or non-text content ({reason_code})"
        else:
            message = f"Invalid LLM response: {reason_code}"
        super().__init__(message)
