"""Характеристические тесты для _check_trace_block_invariants (итерация 18, PR-4).

Фиксируют текущее поведение проверки трассировки сборки блоков:
какие диагностики ведут к warning, какие — молча пропускаются.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.reason_codes import ReasonCode
from src.startup_checks import _check_trace_block_invariants


def _diag(
    *,
    name: str = "block",
    eligible: bool = True,
    included: bool = True,
    reason_codes: list[str] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        eligible=eligible,
        included=included,
        reason_codes=[] if reason_codes is None else reason_codes,
    )


def _trace(*diags: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(blocks=list(diags))


def test_empty_trace_no_warnings(caplog: pytest.LogCaptureFixture) -> None:
    """Пустой trace — ни одного warning."""
    caplog.set_level("WARNING")
    _check_trace_block_invariants(_trace())
    assert caplog.text == ""


def test_included_not_eligible_warns(caplog: pytest.LogCaptureFixture) -> None:
    """Блок включён, но не eligible → warning «included but not eligible»."""
    caplog.set_level("WARNING")
    _check_trace_block_invariants(_trace(_diag(name="b1", eligible=False, included=True)))
    assert "included but not eligible" in caplog.text
    assert "b1" in caplog.text


def test_eligible_not_included_without_reason_warns(caplog: pytest.LogCaptureFixture) -> None:
    """Блок eligible, не включён, reason_codes пуст → warning «without reason»."""
    caplog.set_level("WARNING")
    _check_trace_block_invariants(
        _trace(_diag(name="b2", eligible=True, included=False, reason_codes=[]))
    )
    assert "without reason" in caplog.text
    assert "b2" in caplog.text


def test_eligible_not_included_feature_disabled_no_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Блок eligible, не включён по BLOCK_INELIGIBLE_FEATURE_DISABLED → без warning."""
    caplog.set_level("WARNING")
    _check_trace_block_invariants(
        _trace(
            _diag(
                name="b3",
                eligible=True,
                included=False,
                reason_codes=[ReasonCode.BLOCK_INELIGIBLE_FEATURE_DISABLED],
            )
        )
    )
    assert caplog.text == ""


def test_eligible_not_included_other_reason_no_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Блок eligible, не включён с «непонятной» причиной — молча пропускается.

    Фиксирует текущее поведение: outer-условие пропускает только случай
    BLOCK_INELIGIBLE_FEATURE_DISABLED, но и inner не срабатывает для
    непустых reason_codes. Такое поведение сохраняем как есть.
    """
    caplog.set_level("WARNING")
    _check_trace_block_invariants(
        _trace(
            _diag(
                name="b4",
                eligible=True,
                included=False,
                reason_codes=["SOME_OTHER_REASON"],
            )
        )
    )
    assert caplog.text == ""
