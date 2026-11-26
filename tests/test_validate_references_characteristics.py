"""Характеристические тесты для _validate_references_in_configs (итерация 18, PR-10).

Фиксируют текущее поведение до рефакторинга:
- префиксные ссылки intent:/overlay:/feature: — валидация или raise;
- беспрефиксные ссылки — warning при совпадении с реестром, иначе raise;
- обход полей домена (4), intent (2), overlay (2).

Проверяет, что в warning-сообщениях подставляется фактическое имя ссылки
(f-string), а не литерал '{name}'. Исправлено в PR-11.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from src.startup_checks import _validate_references_in_configs


def _cfg(**kwargs):
    """Минимальный fake-объект с 4 полями-ссылками (duck-typing)."""
    defaults = {
        "suppresses": (),
        "conflicts_with": (),
        "incompatible_intents": (),
        "incompatible_overlays": (),
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


@pytest.fixture
def patched_registries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Подменяет реестры и _is_valid_feature для контролируемых имён."""
    monkeypatch.setattr("src.startup_checks.providers.ALLOWED_OVERLAYS", {"myoverlay"})
    monkeypatch.setattr("src.startup_checks.providers.ALLOWED_INTENTS", {"myintent"})
    monkeypatch.setattr(
        "src.startup_checks.providers._is_valid_feature",
        lambda name: name in {"myfeature"},
    )


# --- unprefixed: warning-ветки (3) + raise (1) -----------------------------


def test_unprefixed_overlay_warns_with_actual_name(
    caplog: pytest.LogCaptureFixture, patched_registries
) -> None:
    """myoverlay in ALLOWED_OVERLAYS → warning с actual name в подсказке."""
    caplog.set_level(logging.WARNING)
    _validate_references_in_configs({"blog": _cfg(suppresses=["myoverlay"])}, {}, {})
    assert "Unprefixed overlay reference 'myoverlay'" in caplog.text
    assert "consider using 'overlay:myoverlay'" in caplog.text


def test_unprefixed_intent_warns_with_actual_name(
    caplog: pytest.LogCaptureFixture, patched_registries
) -> None:
    """myintent in ALLOWED_INTENTS → warning с actual name в подсказке."""
    caplog.set_level(logging.WARNING)
    _validate_references_in_configs({"blog": _cfg(suppresses=["myintent"])}, {}, {})
    assert "Unprefixed intent reference 'myintent'" in caplog.text
    assert "consider using 'intent:myintent'" in caplog.text


def test_unprefixed_feature_warns_with_actual_name(
    caplog: pytest.LogCaptureFixture, patched_registries
) -> None:
    """myfeature → _is_valid_feature True → warning с actual name в подсказке."""
    caplog.set_level(logging.WARNING)
    _validate_references_in_configs({"blog": _cfg(suppresses=["myfeature"])}, {}, {})
    assert "Unprefixed feature reference 'myfeature'" in caplog.text
    assert "consider using 'feature:myfeature'" in caplog.text


def test_unprefixed_unknown_raises(patched_registries) -> None:
    """Неизвестное имя без префикса → ValueError."""
    with pytest.raises(ValueError, match="not found in any registry"):
        _validate_references_in_configs({"blog": _cfg(suppresses=["unknown"])}, {}, {})


# --- prefixed: valid (3) + invalid (3) -------------------------------------


def test_prefixed_overlay_valid_no_raise(patched_registries) -> None:
    _validate_references_in_configs({"blog": _cfg(suppresses=["overlay:myoverlay"])}, {}, {})


def test_prefixed_overlay_invalid_raises(patched_registries) -> None:
    with pytest.raises(ValueError, match="'overlay:missing' does not exist"):
        _validate_references_in_configs({"blog": _cfg(suppresses=["overlay:missing"])}, {}, {})


def test_prefixed_intent_valid_no_raise(patched_registries) -> None:
    _validate_references_in_configs({"blog": _cfg(suppresses=["intent:myintent"])}, {}, {})


def test_prefixed_intent_invalid_raises(patched_registries) -> None:
    with pytest.raises(ValueError, match="'intent:missing' does not exist"):
        _validate_references_in_configs({"blog": _cfg(suppresses=["intent:missing"])}, {}, {})


def test_prefixed_feature_valid_no_raise(patched_registries) -> None:
    _validate_references_in_configs({"blog": _cfg(suppresses=["feature:myfeature"])}, {}, {})


def test_prefixed_feature_invalid_raises(patched_registries) -> None:
    with pytest.raises(ValueError, match="'feature:missing' does not exist"):
        _validate_references_in_configs({"blog": _cfg(suppresses=["feature:missing"])}, {}, {})


# --- циклы по категориям ---------------------------------------------------


@pytest.mark.parametrize(
    "field",
    ["suppresses", "conflicts_with", "incompatible_intents", "incompatible_overlays"],
)
def test_domain_all_fields_validated(field: str, patched_registries) -> None:
    """Все 4 поля DomainConfig валидируются; невалидная ссылка → raise."""
    domains = {"blog": _cfg(**{field: ["unknown"]})}
    with pytest.raises(ValueError, match=f"Invalid {field} reference"):
        _validate_references_in_configs(domains, {}, {})


def test_intent_loop_validates(patched_registries) -> None:
    """intents.suppresses с невалидной ссылкой → raise с «intent <name>»."""
    intents = {"myintent": _cfg(suppresses=["unknown"])}
    with pytest.raises(ValueError, match="Invalid suppresses reference in intent myintent"):
        _validate_references_in_configs({}, intents, {})


def test_overlay_loop_validates(patched_registries) -> None:
    """overlays.conflicts_with с невалидной ссылкой → raise с «overlay <name>»."""
    overlays = {"myoverlay": _cfg(conflicts_with=["unknown"])}
    with pytest.raises(ValueError, match="Invalid conflicts_with reference in overlay myoverlay"):
        _validate_references_in_configs({}, {}, overlays)
