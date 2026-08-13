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
from typing import Optional, Literal

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
MIN_EDIT_OUTPUT_TOKENS: int = 1500
MAX_EDIT_OUTPUT_TOKENS: int = 4096
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
    model: Optional[str]
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
        model: Optional[str],
        input_tokens_estimate: int,
        requested_output_tokens: int,
        available_output_tokens: int,
        context_window: int,
        reason: str,
        mode: str = "enforce",
    ) -> None:
        self.provider = provider
        self.model = model
        self.input_tokens_estimate = input_tokens_estimate
        self.requested_output_tokens = requested_output_tokens
        self.available_output_tokens = available_output_tokens
        self.context_window = context_window
        self.reason = reason
        self.mode = mode

        message = (
            f"Request exceeds context window for {provider}/{model or 'default'}. "
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
        chars_per_token: Количество символов на один токен (по умолчанию 3.5 для русского текста)

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

def _normalize_model_name_for_env(model: Optional[str]) -> str:
    """Нормализует имя модели для использования в имени переменной окружения."""
    if not model:
        return ""
    # Заменяем /, -, . на _ и приводим к верхнему регистру
    normalized = re.sub(r"[^a-zA-Z0-9]", "_", model)
    return normalized.upper()


def get_context_profile_from_env(
    provider: str,
    model: Optional[str] = None,
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
            return mode

    global_var = "LLM_CONTEXT_BUDGET_MODE"
    mode = os.getenv(global_var)
    if mode is not None:
        if mode in ("observe", "cap", "enforce"):
            return mode

    return DEFAULT_MODE


# ---------------------------------------------------------------------------
# Основная функция расчёта бюджета
# ---------------------------------------------------------------------------

def resolve_context_budget(
    *,
    provider: str,
    model: Optional[str],
    prompt: str,
    source_text: str,
    context_window: Optional[int] = None,
    safety_margin: Optional[int] = None,
    mode: Optional[ContextBudgetMode] = None,
) -> ContextBudget:
    """
    Рассчитывает итоговый контекстный бюджет.

    Алгоритм:
    1. Если context_window или safety_margin не переданы, загружаем из env (через get_context_profile_from_env).
    2. Оцениваем входные токены.
    3. Оцениваем запрошенный выходной бюджет (на основе source_text).
    4. Вычисляем доступное место под ответ:
         available = context_window - input_tokens_estimate - safety_margin
    5. Если available < MIN_USEFUL_OUTPUT_TOKENS:
         - в режиме enforce — выбрасываем LLMContextLimitError
         - в режиме observe/cap — логируем предупреждение, но не выбрасываем (будет обработано выше)
    6. Иначе effective = min(requested_output_tokens, available).

    Args:
        provider: Имя провайдера
        model: Имя модели (None = default)
        prompt: Полный промпт
        source_text: Исходный текст пользователя
        context_window: Размер контекстного окна (если None — берём из env)
        safety_margin: Запас безопасности (если None — берём из env)
        mode: Режим работы (если None — берём из env)

    Returns:
        ContextBudget с результатами расчёта

    Raises:
        LLMContextLimitError: Если запрос не помещается в контекстное окно и режим == "enforce"
    """
    # Загружаем профиль из env, если не переданы параметры
    if context_window is None or safety_margin is None or mode is None:
        profile = get_context_profile_from_env(provider, model)
        if context_window is None:
            context_window = profile.context_window
        if safety_margin is None:
            safety_margin = profile.safety_margin
        if mode is None:
            mode = profile.mode

    input_tokens = estimate_input_tokens(prompt)
    requested_output = estimate_edit_output_tokens(source_text)

    available = context_window - input_tokens - safety_margin

    # Проверка: если сам промпт не помещается
    if input_tokens >= context_window - safety_margin:
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
        # В режимах observe/cap — просто возвращаем бюджет с available < 0
        # Это позволит вызывающему коду залогировать и принять решение

    # Проверка: недостаточно места для полезного ответа
    if available < MIN_USEFUL_OUTPUT_TOKENS:
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
        # В режимах observe/cap — продолжаем, но effective_output будет = available (если доступно > 0)
        # или 0, если available отрицательный.

    # Определяем эффективный выходной бюджет
    if available < 0:
        # Даже если режим observe, мы не можем дать отрицательный max_tokens
        effective_output = 0
        was_capped = True
    elif requested_output <= available:
        effective_output = requested_output
        was_capped = False
    else:
        effective_output = available
        was_capped = True

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
