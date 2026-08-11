"""
tests/test_context_budget.py

Unit-тесты для модуля context_budget.
Покрывают все сценарии расчёта бюджета.
"""

import pytest

from src.context_budget import (
    resolve_context_budget,
    estimate_input_tokens,
    estimate_edit_output_tokens,
    LLMContextLimitError,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_SAFETY_MARGIN,
    MIN_USEFUL_OUTPUT_TOKENS,
    MIN_EDIT_OUTPUT_TOKENS,
    MAX_EDIT_OUTPUT_TOKENS,
)


# ---------------------------------------------------------------------------
# Тесты для estimate_input_tokens
# ---------------------------------------------------------------------------

def test_estimate_input_tokens_empty() -> None:
    assert estimate_input_tokens("") == 0


def test_estimate_input_tokens_short() -> None:
    # "Привет" = 6 символов / 3.5 = 1.71 → округление вверх = 2
    assert estimate_input_tokens("Привет") == 2


def test_estimate_input_tokens_long() -> None:
    text = "а" * 100
    expected = (100 + 3.5 - 1) // 3.5  # ceil
    assert estimate_input_tokens(text) == int(expected)


# ---------------------------------------------------------------------------
# Тесты для estimate_edit_output_tokens
# ---------------------------------------------------------------------------

def test_estimate_edit_output_tokens_empty() -> None:
    assert estimate_edit_output_tokens("") == MIN_EDIT_OUTPUT_TOKENS


def test_estimate_edit_output_tokens_short_text() -> None:
    # 10 символов / 3.5 = 2.85 → * 1.35 = 3.86 → ceil = 4
    # Но MIN_EDIT_OUTPUT_TOKENS = 768, поэтому результат = 768
    result = estimate_edit_output_tokens("а" * 10)
    assert result == MIN_EDIT_OUTPUT_TOKENS


def test_estimate_edit_output_tokens_medium_text() -> None:
    # 1000 символов / 3.5 = 285.7 → * 1.35 = 385.7 → ceil = 386
    # 386 > 768? Нет, значит MIN_EDIT_OUTPUT_TOKENS = 768
    result = estimate_edit_output_tokens("а" * 1000)
    assert result == MIN_EDIT_OUTPUT_TOKENS


def test_estimate_edit_output_tokens_long_text() -> None:
    # 5000 символов / 3.5 = 1428.6 → * 1.35 = 1928.6 → ceil = 1929
    result = estimate_edit_output_tokens("а" * 5000)
    assert MIN_EDIT_OUTPUT_TOKENS < result < MAX_EDIT_OUTPUT_TOKENS
    assert result == 1929


def test_estimate_edit_output_tokens_very_long_text() -> None:
    # 20000 символов / 3.5 = 5714.3 → * 1.35 = 7714.3 → но MAX = 4096
    result = estimate_edit_output_tokens("а" * 20000)
    assert result == MAX_EDIT_OUTPUT_TOKENS


# ---------------------------------------------------------------------------
# Тесты для resolve_context_budget
# ---------------------------------------------------------------------------

def test_resolve_budget_normal() -> None:
    """Нормальный случай: всё помещается, output не обрезается."""
    prompt = "а" * 1000  # ~286 токенов
    source = "б" * 1000  # ~286 токенов → requested ~386, но MIN=768
    budget = resolve_context_budget(
        provider="openrouter",
        model="auto",
        prompt=prompt,
        source_text=source,
        context_window=8192,
        safety_margin=512,
        mode="observe",  # явно указываем режим, но он не влияет на результат
    )

    assert budget.input_tokens_estimate == estimate_input_tokens(prompt)
    assert budget.requested_output_tokens == estimate_edit_output_tokens(source)
    assert budget.effective_output_tokens == budget.requested_output_tokens
    assert budget.was_capped is False
    assert budget.available_output_tokens > budget.requested_output_tokens


def test_resolve_budget_capped() -> None:
    """Случай, когда запрошенный output больше доступного."""
    prompt = "а" * 20000  # ~5714 токенов
    source = "б" * 20000  # ~5714 токенов → requested ~7714, но MAX=4096
    context_window = 8192
    safety_margin = 512

    budget = resolve_context_budget(
        provider="openrouter",
        model="auto",
        prompt=prompt,
        source_text=source,
        context_window=context_window,
        safety_margin=safety_margin,
        mode="observe",
    )

    # input_tokens ~5714, доступно: 8192 - 5714 - 512 = 1966
    # requested ~4096, effective = 1966
    assert budget.was_capped is True
    assert budget.effective_output_tokens <= budget.available_output_tokens
    assert budget.effective_output_tokens < budget.requested_output_tokens


def test_resolve_budget_insufficient_output_raises() -> None:
    """Случай, когда доступно меньше MIN_USEFUL_OUTPUT_TOKENS и режим enforce."""
    prompt = "а" * 26000  # ~7429 токенов
    source = "б" * 100
    context_window = 8192
    safety_margin = 512

    with pytest.raises(LLMContextLimitError) as exc_info:
        resolve_context_budget(
            provider="openrouter",
            model="auto",
            prompt=prompt,
            source_text=source,
            context_window=context_window,
            safety_margin=safety_margin,
            mode="enforce",  # <-- явно включаем enforce
        )

    error = exc_info.value
    assert error.provider == "openrouter"
    assert error.reason == "insufficient_output_budget"
    assert error.available_output_tokens < MIN_USEFUL_OUTPUT_TOKENS


def test_resolve_budget_input_too_large_raises() -> None:
    """Случай, когда сам промпт не помещается и режим enforce."""
    prompt = "а" * 30000  # ~8571 токенов
    source = "б" * 100
    context_window = 8192
    safety_margin = 512

    with pytest.raises(LLMContextLimitError) as exc_info:
        resolve_context_budget(
            provider="openrouter",
            model="auto",
            prompt=prompt,
            source_text=source,
            context_window=context_window,
            safety_margin=safety_margin,
            mode="enforce",  # <-- явно включаем enforce
        )

    error = exc_info.value
    assert error.provider == "openrouter"
    assert error.reason == "input_too_large"


def test_resolve_budget_with_custom_constants() -> None:
    """Проверяем, что можно переопределить context_window и safety_margin."""
    prompt = "а" * 5000
    source = "б" * 5000

    budget = resolve_context_budget(
        provider="openai",
        model="gpt-4",
        prompt=prompt,
        source_text=source,
        context_window=16384,
        safety_margin=2048,
        mode="observe",
    )

    assert budget.context_window == 16384
    assert budget.safety_margin == 2048
    assert budget.input_tokens_estimate == estimate_input_tokens(prompt)
    assert budget.requested_output_tokens == estimate_edit_output_tokens(source)


def test_resolve_budget_uses_defaults() -> None:
    """Проверяем, что функция использует дефолтные значения, если не переданы."""
    prompt = "а" * 100
    source = "б" * 100

    budget = resolve_context_budget(
        provider="openrouter",
        model=None,
        prompt=prompt,
        source_text=source,
        mode="observe",
    )

    assert budget.context_window == DEFAULT_CONTEXT_WINDOW
    assert budget.safety_margin == DEFAULT_SAFETY_MARGIN


def test_resolve_budget_does_not_log_sensitive_data() -> None:
    """Проверяем, что исключение не содержит пользовательский контент (в режиме enforce)."""
    prompt = "Конфиденциальные данные: секретный код 12345"
    source = "Ещё более секретный текст"

    with pytest.raises(LLMContextLimitError) as exc_info:
        resolve_context_budget(
            provider="openrouter",
            model="auto",
            prompt=prompt,
            source_text=source,
            context_window=100,
            safety_margin=10,
            mode="enforce",  # <-- явно включаем enforce
        )

    error = exc_info.value
    # Убеждаемся, что пользовательский контент не попал в сообщение
    assert "секретный" not in str(error)
    assert "12345" not in str(error)
    assert "Конфиденциальные" not in str(error)


def test_resolve_budget_provider_model_fields() -> None:
    """Проверяем, что provider и model правильно сохраняются в ошибке (в режиме enforce)."""
    provider = "anthropic"
    model = "claude-3-sonnet"

    # Нормальный случай (не должен выбрасывать)
    budget = resolve_context_budget(
        provider=provider,
        model=model,
        prompt="а" * 100,
        source_text="б" * 100,
        mode="observe",
    )
    # В ContextBudget нет полей provider/model, но они не нужны для результата

    # Ошибка должна содержать provider и model
    with pytest.raises(LLMContextLimitError) as exc_info:
        resolve_context_budget(
            provider=provider,
            model=model,
            prompt="а" * 30000,
            source_text="б" * 100,
            context_window=8192,
            safety_margin=512,
            mode="enforce",  # <-- явно включаем enforce
        )

    error = exc_info.value
    assert error.provider == provider
    assert error.model == model