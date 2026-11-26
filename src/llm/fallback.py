"""Перебор провайдеров с fallback и накоплением ошибок.

Выделено из src/llm_client.py в итерации 17.1.a карты №3.
"""

from __future__ import annotations

import logging

from src.context_budget import (
    LLMContextLimitError,
    get_context_profile_from_env,
    resolve_context_budget,
)
from src.provider_registry import LLMProvider

from .classification import _classify_error, _extract_upstream_status
from .config import LLMResponse
from .errors import LLMError, LLMFallbackError
from .factory import create_llm_client
from .tokens import estimate_max_tokens

logger = logging.getLogger(__name__)


def _resolve_provider_max_tokens(
    provider_name: str,
    model: str | None,
    prompt: str,
    source_text: str | None,
    explicit_max_tokens: int | None,
) -> int | None:
    """
    Вычисляет max_tokens для провайдера.
    Возвращает None, если провайдер не может обработать запрос (бюджет <= 0).
    """
    if explicit_max_tokens is not None:
        return explicit_max_tokens

    if source_text is not None:
        try:
            profile = get_context_profile_from_env(provider_name, model)
            budget = resolve_context_budget(
                provider=provider_name,
                model=model,
                prompt=prompt,
                source_text=source_text,
                context_window=profile.context_window,
                safety_margin=profile.safety_margin,
                mode=profile.mode,
            )
            logger.info(
                "Context budget for %s: input_tokens=%d, requested=%d, "
                "effective=%d, capped=%s, mode=%s",
                provider_name,
                budget.input_tokens_estimate,
                budget.requested_output_tokens,
                budget.effective_output_tokens,
                budget.was_capped,
                budget.mode,
            )
            if budget.effective_output_tokens <= 0:
                logger.warning(
                    "Provider %s has zero effective output tokens, skipping",
                    provider_name,
                )
                return None
            return budget.effective_output_tokens
        except LLMContextLimitError as error:
            logger.warning(
                "Provider %s cannot handle request: %s (mode=%s)",
                provider_name,
                error.reason,
                getattr(error, "mode", "enforce"),
            )
            return None

    # Если source_text отсутствует, используем эвристику
    return estimate_max_tokens(prompt)


async def _try_provider(
    provider_enum: LLMProvider,
    model: str | None,
    prompt: str,
    temperature: float,
    max_retries: int,
    max_tokens: int,
) -> LLMResponse:
    """
    Пытается выполнить запрос к одному провайдеру.
    Возвращает LLMResponse или выбрасывает исключение.
    """
    provider_name = provider_enum.value
    logger.info(
        "call_with_fallback: trying provider=%s model=%s max_tokens=%s prompt_length=%d",
        provider_name,
        model,
        max_tokens,
        len(prompt),
    )

    async with create_llm_client(
        provider=provider_enum,
        model=model,
        temperature=temperature,
        max_retries=max_retries,
        max_tokens=max_tokens,
    ) as client:
        response = await client.generate(prompt)

    logger.info(
        "call_with_fallback: success with provider=%s finish_reason=%s",
        provider_name,
        response.finish_reason,
    )
    return response


def _build_fallback_error(
    primary_error: LLMError | None,
    primary_provider: str | None,
    skipped_providers: list[str],
    unknown_providers: list[str],
    prompt_length: int,
) -> LLMFallbackError:
    """
    Строит LLMFallbackError с классификацией и метаданными.
    """
    if primary_error is not None:
        kind = _classify_error(primary_error)
        upstream_status = _extract_upstream_status(primary_error)
        logger.warning(
            "All providers exhausted. primary_provider=%s, error_kind=%s, "
            "upstream_status=%s, skipped_providers=%s, unknown_providers=%s, "
            "prompt_length=%d",
            primary_provider,
            kind,
            upstream_status,
            skipped_providers,
            unknown_providers,
            prompt_length,
        )
        last_provider = primary_provider if primary_provider else "none"
        return LLMFallbackError(
            f"All providers failed. Last error from {last_provider}",
            provider=primary_provider,
            primary_error=primary_error,
            skipped_providers=tuple(skipped_providers),
            unknown_providers=tuple(unknown_providers),
            prompt_length=prompt_length,
            upstream_status=upstream_status,
            kind=kind,
        )
    else:
        logger.warning(
            "All providers skipped or unknown. "
            "skipped_providers=%s, unknown_providers=%s, prompt_length=%d",
            skipped_providers,
            unknown_providers,
            prompt_length,
        )
        return LLMFallbackError(
            "All providers failed (no primary error)",
            provider=None,
            primary_error=None,
            skipped_providers=tuple(skipped_providers),
            unknown_providers=tuple(unknown_providers),
            prompt_length=prompt_length,
            upstream_status=None,
            kind="configuration",
        )


async def call_with_fallback(
    prompt: str,
    providers: list[str],
    model: str | None = None,
    temperature: float = 0.3,
    max_retries_per_provider: int = 1,
    max_tokens: int | None = None,
    source_text: str | None = None,
) -> LLMResponse:
    """
    Выполняет запрос к LLM с перебором провайдеров.

    Args:
        prompt: Полный промпт
        providers: Список имён провайдеров в порядке приоритета
        model: Модель для первого провайдера (для остальных — default)
        temperature: Температура
        max_retries_per_provider: Количество повторных попыток на провайдера
        max_tokens: Явный max_tokens (если None — вычисляется автоматически)
        source_text: Исходный текст пользователя (для оценки выходного бюджета)

    Returns:
        LLMResponse от успешного провайдера

    Raises:
        LLMFallbackError: Если все провайдеры не смогли обработать запрос
    """
    if not providers:
        raise LLMError("No providers specified. Cannot execute LLM call.")

    primary_error: LLMError | None = None
    primary_provider: str | None = None
    skipped_providers: list[str] = []
    unknown_providers: list[str] = []

    for idx, provider_name in enumerate(providers):
        try:
            provider_enum = LLMProvider(provider_name)
        except ValueError:
            unknown_providers.append(provider_name)
            continue

        model_for_this = model if idx == 0 else None

        provider_max_tokens = _resolve_provider_max_tokens(
            provider_name=provider_name,
            model=model_for_this,
            prompt=prompt,
            source_text=source_text,
            explicit_max_tokens=max_tokens,
        )

        if provider_max_tokens is None:
            skipped_providers.append(provider_name)
            continue

        try:
            response = await _try_provider(
                provider_enum=provider_enum,
                model=model_for_this,
                prompt=prompt,
                temperature=temperature,
                max_retries=max_retries_per_provider,
                max_tokens=provider_max_tokens,
            )
            return response

        except ValueError:
            logger.warning(
                "call_with_fallback: provider=%s unavailable " "(missing key or model), skipping",
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

        except Exception as error:
            # Осознанно широкий перехват: последний рубеж перед fallback.
            # Неожиданные ошибки одного провайдера не должны ломать
            # перебор остальных.
            logger.warning(
                "call_with_fallback: provider=%s raised unexpected error: %s",
                provider_name,
                type(error).__name__,
            )
            if primary_error is None:
                primary_error = LLMError(f"Unexpected error: {error}")
                primary_provider = provider_name
            continue

    raise _build_fallback_error(
        primary_error=primary_error,
        primary_provider=primary_provider,
        skipped_providers=skipped_providers,
        unknown_providers=unknown_providers,
        prompt_length=len(prompt),
    )


async def generate_text(
    prompt: str,
    provider: LLMProvider = LLMProvider.PERPLEXITY,
    model: str | None = None,
    temperature: float = 0.3,
) -> str:
    """Удобная обёртка: создаёт клиент, генерирует текст и закрывает клиент.

    Args:
        prompt: Пользовательский текст.
        provider: Провайдер; по умолчанию Perplexity.
        model: Модель; None — из model_policy.
        temperature: Температура генерации.

    Returns:
        Сгенерированный текст.
    """
    async with create_llm_client(
        provider=provider,
        model=model,
        temperature=temperature,
    ) as client:
        response = await client.generate(prompt)
        return response.content
