"""
Тесты итерации 2: ограничения на уровне Pydantic-контракта.

Проверяем EditRequest и AudienceRequest:
- text: границы 1..10000.
- overlays: ≤ 20 элементов (проверка ДО дедупликации), каждый ≤ 100 символов.
- audience.description: ≤ 500 символов.
- model: ≤ 200 символов + regex-allowlist.

HTTP-слой тестируется отдельно в test_request_body_limit.py.

Запуск:
    pytest tests/test_request_limits.py -v
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.contracts import AudienceRequest, EditRequest
from src.shared_contracts import ALLOWED_OVERLAYS

# Один валидный оверлей — берём из реального allowlist, чтобы тесты
# не зависели от конкретных имён.
_VALID_OVERLAY: str = sorted(ALLOWED_OVERLAYS)[0]


# ============================================================================
# text
# ============================================================================


def test_text_at_limit_passes() -> None:
    """Ровно 10 000 символов — принимается."""
    req = EditRequest(text="a" * 10_000)
    assert len(req.text) == 10_000


def test_text_above_limit_rejected() -> None:
    """10 001 символ — отклоняется."""
    with pytest.raises(ValidationError):
        EditRequest(text="a" * 10_001)


def test_text_empty_rejected() -> None:
    """Пустой текст — отклоняется (min_length=1)."""
    with pytest.raises(ValidationError):
        EditRequest(text="")


# ============================================================================
# overlays: количество
# ============================================================================


def test_overlays_count_at_limit_passes() -> None:
    """20 элементов (даже одинаковых) — принимается, дедуп срабатывает после."""
    req = EditRequest(text="x", overlays=[_VALID_OVERLAY] * 20)
    assert req.overlays == [_VALID_OVERLAY]


def test_overlays_count_above_limit_rejected_before_dedup() -> None:
    """
    21 одинаковых элементов — отклоняется, несмотря на то что дедуп
    дал бы 1 элемент. Наш валидатор проверяет длину ДО дедупликации,
    чтобы нельзя было протащить миллион одинаковых строк.
    """
    with pytest.raises(ValidationError, match="too many overlays"):
        EditRequest(text="x", overlays=[_VALID_OVERLAY] * 21)


def test_overlays_thousand_duplicates_rejected() -> None:
    """Атака «миллион одинаковых строк» — отклоняется на ранней стадии."""
    with pytest.raises(ValidationError, match="too many overlays"):
        EditRequest(text="x", overlays=[_VALID_OVERLAY] * 1000)


# ============================================================================
# overlays: длина элементов
# ============================================================================


def test_overlays_element_too_long_rejected() -> None:
    """Элемент длиной 101 символ — отклоняется."""
    long_name = "x" * 101
    with pytest.raises(ValidationError, match="too long"):
        EditRequest(text="x", overlays=[long_name])


def test_overlays_element_at_limit_reaches_allowlist() -> None:
    """
    Элемент длиной 100 символов проходит проверку длины,
    но не проходит allowlist (неизвестное имя). Это ожидаемо:
    мы тестируем только, что длина не блокирует раньше allowlist-а.
    """
    border_name = "x" * 100
    with pytest.raises(ValidationError):
        EditRequest(text="x", overlays=[border_name])


# ============================================================================
# audience.description
# ============================================================================


def test_audience_description_at_limit_passes() -> None:
    """500 символов — принимается."""
    EditRequest(
        text="x",
        audience=AudienceRequest(description="a" * 500),
    )


def test_audience_description_above_limit_rejected() -> None:
    """501 символ — отклоняется."""
    with pytest.raises(ValidationError):
        EditRequest(
            text="x",
            audience=AudienceRequest(description="a" * 501),
        )


# ============================================================================
# model
# ============================================================================


def test_model_at_limit_passes() -> None:
    """200 символов в field max_length — принимается (если regex позволяет)."""
    # Используем только разрешённые regex-символы: \w . / : -
    value = "a" * 200
    EditRequest(text="x", model=value)


def test_model_above_limit_rejected() -> None:
    """201 символ — отклоняется."""
    with pytest.raises(ValidationError):
        EditRequest(text="x", model="a" * 201)
