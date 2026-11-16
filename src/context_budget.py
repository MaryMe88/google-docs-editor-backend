"""
context_budget.py

Модуль для управления контекстным бюджетом LLM.

Содержит:
- ModelContextProfile — профиль провайдера/модели с context window
- ContextBudget — результат расчёта бюджета
- LLMContextLimitError — исключение для раннего отказа
- resolve_context_budget() — чистая функция расчёта бюджета
- estimate_input_tokens() — оценка входных токенов
- estimate_edit_output_tokens() — оценка выходных токенов для редактуры
- get_context_profile_from_env() — загрузка профиля из переменных окружения
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Literal, cast

# ---------------------------------------------------------------------------
# Константы по умолчанию
# ---------------------------------------------------------------------------
DEFAULT_CONTEXT_WINDOW: int = 16384
DEFAULT_SAFETY_MARGIN: int = 1024
MIN_USEFUL_OUTPUT_TOKENS: int = 512

# Оценка токенов для русского текста (более консервативно, чем 4)
CHARS_PER_INPUT_TOKEN: float = 3.5
CHARS_PER_OUTPUT_TOKEN: float = 3.5

# Лимиты для выходного бюджета
#
# MIN_EDIT_OUTPUT_TOKENS увеличен с 768 до 4096 (issue #22):
# reasoning-модели (например, deepseek/deepseek-v4.1-flash через
# OpenRouter) тратят весь выходной бюджет на chain-of-thought и
# оставляют 0 токенов на финальный ответ. 4096 даёт запас и на
# reasoning (~1000-1500 токенов), и на сам текст.
#
# Для не-reasoning моделей стоимость не растёт: max_tokens — это
# лимит, а не цель. Модель завершает ответ раньше и биллятся за
# фактические токены.
#
# Env-override: MIN_EDIT_OUTPUT_TOKENS (целое положительное число).
_DEFAULT_MIN_EDIT_OUTPUT_TOKENS: int = 4096


def _load_min_edit_output_tokens() -> int:
    """Читает MIN_EDIT_OUTPUT_TOKENS из env с fallback на дефолт.

    Возвращает дефолт, если переменная не задана, не парсится в int
    или не положительна. Ошибки env не критичны для старта.
    """
    raw = os.getenv("MIN_EDIT_OUTPUT_TOKENS")
    if raw is None:
        return _DEFAULT_MIN_EDIT_OUTPUT_TOKENS
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_MIN_EDIT_OUTPUT_TOKENS
    if value <= 0:
        return _DEFAULT_MIN_EDIT_OUTPUT_TOKENS
    return value


MIN_EDIT_OUTPUT_TOKENS: int = _load_min_edit_output_tokens()
MAX_EDIT_OUTPUT_TOKENS: int = 8192
EDIT_OUTPUT_MULTIPLIER: float = 1.35

# Режимы работы
ContextBudgetMode = Literal["observe", "cap", "enforce"]
DEFAULT_MODE: ContextBudgetMode = "observe"


# ---------------------------------------------------------------------------
# Контракты
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelContextProfile:
    """
    Профиль контекстного окна для конкретного провайдера/модели.

    Attributes:
        provider: Имя провайдера (openrouter, openai, anthropic, perplexity)
        model: Имя модели (None означает default для провайдера)
        context_window: Максимальный размер контекста в токенах
        safety_margin: Запас, который резервируется для безопасности
        mode: Режим работы (observe, cap, enforce)
    """

    provider: str
    model: str | None
    context_window: int
    safety_margin: int
    mode: ContextBudgetMode


@dataclass(frozen=True)
class ContextBudget:
    """
    Результат расчёта контекстного бюджета.

    Attributes:
        input_tokens_estimate: Оценка размера промпта в токенах
        requested_output_tokens: Запрошенный размер выходного бюджета
        effective_output_tokens: Реальный выходной бюджет (может быть меньше запрошенного)
        context_window: Используемый размер контекстного окна
        safety_margin: Используемый запас
        was_capped: Был ли уменьшен выходной бюджет
        available_output_tokens: Доступное место под ответ
        mode: Режим работы
    """

    input_tokens_estimate: int
    requested_output_tokens: int
    effective_output_tokens: int
    context_window: int
    safety_margin: int
    was_capped: bool
    available_output_tokens: int
    mode: ContextBudgetMode


class LLMContextLimitError(Exception):
    """
    Исключение, выбрасываемое, когда запрос не помещается в контекстное окно
    и режим == "enforce".

    Содержит все необходимые поля для диагностики и логирования,
    без раскрытия пользовательского контента.
    """

    def __init__(
        self,
        provider: str,
        model: str | None,
        input_tokens_estimate: int,
        requested_output_tokens: int,
        available_output_tokens: int,
        context_window: int,
        reason: str,
        mode: str = "enforce",
    ) -> None:
        """Инициализирует ошибку превышения контекстного окна.

        Args:
            provider: Имя провайдера.
            model: Имя модели или None.
            input_tokens_estimate: Оценка токенов на вход.
            requested_output_tokens: Запрошенный лимит токенов ответа.
            available_output_tokens: Доступный лимит токенов ответа.
            context_window: Размер контекстного окна модели.
            reason: Причина отказа.
            mode: Режим обработки (enforce по умолчанию).
        """
        self.provider = provider
        self.model = model
        self.input_tokens_estimate = input_tokens_estimate
        self.requested_output_tokens = requested_output_tokens
        self.available_output_tokens = available_output_tokens
        self.context_window = context_window
        self.reason = reason
        self.mode = mode

        message = (
            f"Request exceeds context window for {provider}/"
            f"{model or 'default'}. "
            f"Input: {input_tokens_estimate} tokens, "
            f"available output: {available_output_tokens} tokens, "
            f"requested: {requested_output_tokens} tokens. "
            f"Reason: {reason}"
        )
        super().__init__(message)


# ---------------------------------------------------------------------------
# Функции оценки токенов
# ---------------------------------------------------------------------------


def estimate_input_tokens(prompt: str, chars_per_token: float = CHARS_PER_INPUT_TOKEN) -> int:
    """
    Консервативно оценивает размер промпта в токенах.

    Args:
        prompt: Полный промпт
        chars_per_token: Количество символов на один токен
            (по умолчанию 3.5 для русского текста)

    Returns:
        Оценка количества токенов (округление вверх)
    """
    if not prompt:
        return 0
    return math.ceil(len(prompt) / chars_per_token)


def estimate_edit_output_tokens(
    source_text: str,
    chars_per_token: float = CHARS_PER_OUTPUT_TOKEN,
) -> int:
    """
    Оценивает необходимый объём отредактированного текста в токенах.

    Формула:
        source_tokens = len(source_text) / chars_per_token
        requested = source_tokens * EDIT_OUTPUT_MULTIPLIER
        effective = max(MIN_EDIT_OUTPUT_TOKENS, min(requested, MAX_EDIT_OUTPUT_TOKENS))

    Args:
        source_text: Исходный текст пользователя
        chars_per_token: Количество символов на один токен

    Returns:
        Оценка необходимого выходного бюджета в токенах
    """
    if not source_text:
        return MIN_EDIT_OUTPUT_TOKENS

    source_tokens = math.ceil(len(source_text) / chars_per_token)
    requested = int(source_tokens * EDIT_OUTPUT_MULTIPLIER)

    if requested < MIN_EDIT_OUTPUT_TOKENS:
        return MIN_EDIT_OUTPUT_TOKENS
    if requested > MAX_EDIT_OUTPUT_TOKENS:
        return MAX_EDIT_OUTPUT_TOKENS
    return requested


# ---------------------------------------------------------------------------
# Загрузка профиля из переменных окружения
# ---------------------------------------------------------------------------


def _normalize_model_name_for_env(model: str | None) -> str:
    """Нормализует имя модели для использования в имени переменной окружения."""
    if not model:
        return ""
    # Заменяем /, -, . на _ и приводим к верхнему регистру
    normalized = re.sub(r"[^a-zA-Z0-9]", "_", model)
    return normalized.upper()


def get_context_profile_from_env(
    provider: str,
    model: str | None = None,
) -> ModelContextProfile:
    """
    Загружает профиль контекста из переменных окружения.

    Приоритет:
        1. Специфичная переменная для модели: {PROVIDER}_{MODEL}_CONTEXT_WINDOW
        2. Общая переменная провайдера: {PROVIDER}_CONTEXT_WINDOW
        3. Дефолт: DEFAULT_CONTEXT_WINDOW

    Аналогично для safety_margin:
        1. {PROVIDER}_{MODEL}_SAFETY_MARGIN
        2. {PROVIDER}_SAFETY_MARGIN
        3. LLM_CONTEXT_SAFETY_MARGIN
        4. DEFAULT_SAFETY_MARGIN

    Режим:
        1. {PROVIDER}_BUDGET_MODE
        2. LLM_CONTEXT_BUDGET_MODE
        3. DEFAULT_MODE

    Args:
        provider: Имя провайдера (должно быть в верхнем регистре для env)
        model: Имя модели (опционально)

    Returns:
        ModelContextProfile с заполненными значениями
    """
    provider_upper = provider.upper()
    model_norm = _normalize_model_name_for_env(model)

    # Контекстное окно
    if model_norm:
        specific_var = f"{provider_upper}_{model_norm}_CONTEXT_WINDOW"
        context_window = os.getenv(specific_var)
        if context_window is not None:
            try:
                return ModelContextProfile(
                    provider=provider,
                    model=model,
                    context_window=int(context_window),
                    safety_margin=_get_safety_margin(provider_upper, model_norm),
                    mode=_get_mode(provider_upper),
                )
            except ValueError:
                pass  # fall through

    general_var = f"{provider_upper}_CONTEXT_WINDOW"
    context_window = os.getenv(general_var)
    if context_window is not None:
        try:
            return ModelContextProfile(
                provider=provider,
                model=model,
                context_window=int(context_window),
                safety_margin=_get_safety_margin(provider_upper, model_norm),
                mode=_get_mode(provider_upper),
            )
        except ValueError:
            pass

    # Дефолт
    return ModelContextProfile(
        provider=provider,
        model=model,
        context_window=DEFAULT_CONTEXT_WINDOW,
        safety_margin=_get_safety_margin(provider_upper, model_norm),
        mode=_get_mode(provider_upper),
    )


def _get_safety_margin(provider_upper: str, model_norm: str) -> int:
    """Внутренняя функция для получения safety margin с приоритетами."""
    if model_norm:
        specific_var = f"{provider_upper}_{model_norm}_SAFETY_MARGIN"
        value = os.getenv(specific_var)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                pass

    general_var = f"{provider_upper}_SAFETY_MARGIN"
    value = os.getenv(general_var)
    if value is not None:
        try:
            return int(value)
        except ValueError:
            pass

    global_var = "LLM_CONTEXT_SAFETY_MARGIN"
    value = os.getenv(global_var)
    if value is not None:
        try:
            return int(value)
        except ValueError:
            pass

    return DEFAULT_SAFETY_MARGIN


def _get_mode(provider_upper: str) -> ContextBudgetMode:
    """Внутренняя функция для получения режима с приоритетами."""
    specific_var = f"{provider_upper}_BUDGET_MODE"
    mode = os.getenv(specific_var)
    if mode is not None:
        if mode in ("observe", "cap", "enforce"):
            return cast(ContextBudgetMode, mode)

    global_var = "LLM_CONTEXT_BUDGET_MODE"
    mode = os.getenv(global_var)
    if mode is not None:
        if mode in ("observe", "cap", "enforce"):
            return cast(ContextBudgetMode, mode)

    return DEFAULT_MODE


# ---------------------------------------------------------------------------
# Основная функция расчёта бюджета
# ---------------------------------------------------------------------------


def _resolve_budget_inputs(
    *,
    provider: str,
    model: str | None,
    context_window: int | None,
    safety_margin: int | None,
    mode: ContextBudgetMode | None,
) -> tuple[int, int, ContextBudgetMode]:
    """Дополняет параметры бюджета значениями из env, если что-то не задано."""
    if context_window is None or safety_margin is None or mode is None:
        profile = get_context_profile_from_env(provider, model)
        if context_window is None:
            context_window = profile.context_window
        if safety_margin is None:
            safety_margin = profile.safety_margin
        if mode is None:
            mode = profile.mode
    return context_window, safety_margin, mode


def _raise_if_input_too_large(
    *,
    provider: str,
    model: str | None,
    input_tokens: int,
    requested_output: int,
    available: int,
    context_window: int,
    safety_margin: int,
    mode: ContextBudgetMode,
) -> None:
    """В режиме enforce выбрасывает LLMContextLimitError, если промпт не помещается."""
    if input_tokens < context_window - safety_margin:
        return
    if mode == "enforce":
        raise LLMContextLimitError(
            provider=provider,
            model=model,
            input_tokens_estimate=input_tokens,
            requested_output_tokens=requested_output,
            available_output_tokens=available,
            context_window=context_window,
            reason="input_too_large",
            mode=mode,
        )


def _raise_if_output_too_small(
    *,
    provider: str,
    model: str | None,
    input_tokens: int,
    requested_output: int,
    available: int,
    context_window: int,
    mode: ContextBudgetMode,
) -> None:
    """В режиме enforce выбрасывает LLMContextLimitError, если available < MIN_USEFUL_OUTPUT_TOKENS."""
    if available >= MIN_USEFUL_OUTPUT_TOKENS:
        return
    if mode == "enforce":
        raise LLMContextLimitError(
            provider=provider,
            model=model,
            input_tokens_estimate=input_tokens,
            requested_output_tokens=requested_output,
            available_output_tokens=available,
            context_window=context_window,
            reason="insufficient_output_budget",
            mode=mode,
        )


def _compute_effective_output(requested_output: int, available: int) -> tuple[int, bool]:
    """Возвращает (effective_output, was_capped)."""
    if available < 0:
        return 0, True
    if requested_output <= available:
        return requested_output, False
    return available, True


def resolve_context_budget(
    *,
    provider: str,
    model: str | None,
    prompt: str,
    source_text: str,
    context_window: int | None = None,
    safety_margin: int | None = None,
    mode: ContextBudgetMode | None = None,
) -> ContextBudget:
    """
    Рассчитывает итоговый контекстный бюджет.

    Алгоритм:
    1. Если context_window или safety_margin не переданы, загружаем из env.
    2. Оцениваем входные токены.
    3. Оцениваем запрошенный выходной бюджет (на основе source_text).
    4. available = context_window - input_tokens - safety_margin.
    5. Если input_tokens >= context_window - safety_margin и mode == "enforce" — raise.
    6. Если available < MIN_USEFUL_OUTPUT_TOKENS и mode == "enforce" — raise.
    7. Иначе effective = min(requested_output, available) или 0 при available < 0.

    Raises:
        LLMContextLimitError: Если запрос не помещается и режим == "enforce".
    """
    context_window, safety_margin, mode = _resolve_budget_inputs(
        provider=provider,
        model=model,
        context_window=context_window,
        safety_margin=safety_margin,
        mode=mode,
    )

    input_tokens = estimate_input_tokens(prompt)
    requested_output = estimate_edit_output_tokens(source_text)
    available = context_window - input_tokens - safety_margin

    _raise_if_input_too_large(
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        requested_output=requested_output,
        available=available,
        context_window=context_window,
        safety_margin=safety_margin,
        mode=mode,
    )
    _raise_if_output_too_small(
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        requested_output=requested_output,
        available=available,
        context_window=context_window,
        mode=mode,
    )

    effective_output, was_capped = _compute_effective_output(requested_output, available)

    return ContextBudget(
        input_tokens_estimate=input_tokens,
        requested_output_tokens=requested_output,
        effective_output_tokens=effective_output,
        context_window=context_window,
        safety_margin=safety_margin,
        was_capped=was_capped,
        available_output_tokens=available,
        mode=mode,
    )
