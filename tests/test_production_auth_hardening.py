"""
tests/test_production_auth_hardening.py
========================================

Итерация 1 дорожной карты остаточных security-работ (issue #35).

Закрепляет два аспекта production-auth, не покрытых
существующими тестами:

1. Утечка API-ключа в логи. Ни при успешной аутентификации, ни при
   неверном ключе, ни при soft-mode значение ключа не должно
   появляться в логах.

2. Инварианты OpenAPI-схемы. Ключевые эндпоинты и обязательные поля
   в контрактах остались на месте. Без полного снапшота — только
   существенные инварианты, чтобы не ломаться от добавления новых
   эндпоинтов.

Существующие тесты auth (см. tests/test_main.py, tests/test_contracts.py,
tests/test_api_smoke.py, tests/test_retry_auth_polish.py) покрывают
поведение auth. Здесь — только утечки и инварианты схемы.
"""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from src.main import app

# ---------------------------------------------------------------------------
# 1. Ключ не утекает в логи
# ---------------------------------------------------------------------------

# Значение-маркер: если оно появится в логах — тест упадёт с понятным
# сообщением. Формат похож на настоящий секрет, но это тестовое значение.
_LEAK_PROBE_KEY = "leak-probe-secret-9c1c4e5b-2f8a-4d3c-a1b6-7e8d9f0a2b3c"


def _assert_key_not_logged(caplog: pytest.LogCaptureFixture) -> None:
    """Общий ассерт: ни один лог-рекорд не содержит значение ключа."""
    leak_messages = [
        rec.getMessage()
        for rec in caplog.records
        if _LEAK_PROBE_KEY in rec.getMessage()
    ]
    assert not leak_messages, f"API key leaked into logs: {leak_messages!r}"


def test_correct_api_key_not_logged(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Правильный ключ в заголовке не попадает в логи."""
    monkeypatch.setenv("API_SECRET_KEY", _LEAK_PROBE_KEY)
    caplog.clear()

    with caplog.at_level(logging.DEBUG):
        with TestClient(app) as client:
            resp = client.get(
                "/health",
                headers={"X-API-Key": _LEAK_PROBE_KEY},
            )

    # /health может вернуть 200 или 503 — оба допустимы.
    assert resp.status_code in (200, 503), resp.text
    _assert_key_not_logged(caplog)


def test_wrong_api_key_not_logged(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Неверный ключ тоже не попадает в логи."""
    monkeypatch.setenv("API_SECRET_KEY", _LEAK_PROBE_KEY)
    caplog.clear()

    with caplog.at_level(logging.DEBUG):
        with TestClient(app) as client:
            resp = client.get(
                "/health",
                headers={"X-API-Key": "definitely-wrong-key"},
            )

    assert resp.status_code == 401
    _assert_key_not_logged(caplog)


def test_missing_api_key_not_logged(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Отсутствие ключа в запросе не логируется со значением."""
    monkeypatch.setenv("API_SECRET_KEY", _LEAK_PROBE_KEY)
    caplog.clear()

    with caplog.at_level(logging.DEBUG):
        with TestClient(app) as client:
            resp = client.get("/health")

    assert resp.status_code == 401
    _assert_key_not_logged(caplog)


def test_soft_mode_warning_does_not_leak_key(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Soft-mode warning не содержит значения ключа (ключа и нет)."""
    import src.auth as auth_mod

    auth_mod._soft_auth_warned = False
    monkeypatch.delenv("API_SECRET_KEY", raising=False)
    caplog.clear()

    with caplog.at_level(logging.WARNING):
        auth_mod.verify_api_key(x_api_key=None)

    # Убедимся, что warning вообще записан.
    warnings = [r for r in caplog.records if "API_SECRET_KEY" in r.getMessage()]
    assert warnings, "soft-mode warning was not logged"

    _assert_key_not_logged(caplog)


# ---------------------------------------------------------------------------
# 2. Инварианты OpenAPI
# ---------------------------------------------------------------------------


def _openapi() -> dict:
    return app.openapi()


def test_openapi_contains_required_paths() -> None:
    """Ключевые пути зарегистрированы."""
    paths = _openapi()["paths"]
    for required_path in ("/", "/livez", "/health", "/api/edit"):
        assert required_path in paths, f"missing path: {required_path}"


def test_edit_endpoint_is_post_only() -> None:
    """/api/edit принимает только POST."""
    edit_ops = _openapi()["paths"]["/api/edit"]
    assert "post" in edit_ops, "POST /api/edit not declared"
    assert "get" not in edit_ops, "GET /api/edit must not be declared"
    assert "put" not in edit_ops, "PUT /api/edit must not be declared"
    assert "delete" not in edit_ops, "DELETE /api/edit must not be declared"


def test_edit_request_schema_requires_text_with_limits() -> None:
    """EditRequest.text — обязательное поле с разумными лимитами."""
    schemas = _openapi()["components"]["schemas"]
    assert "EditRequest" in schemas, "EditRequest schema missing"

    edit_request = schemas["EditRequest"]
    props = edit_request["properties"]
    required = edit_request.get("required", [])

    assert "text" in props, "EditRequest.text missing"
    assert "text" in required, "EditRequest.text must be required"

    text_schema = props["text"]
    min_len = text_schema.get("minLength")
    max_len = text_schema.get("maxLength")
    assert min_len is not None and min_len >= 1, f"unexpected minLength: {min_len}"
    assert max_len is not None and max_len <= 20_000, f"unexpected maxLength: {max_len}"


def test_edit_response_schema_has_edited_text() -> None:
    """EditResponse содержит обязательное поле edited_text."""
    schemas = _openapi()["components"]["schemas"]
    assert "EditResponse" in schemas, "EditResponse schema missing"

    edit_response = schemas["EditResponse"]
    props = edit_response["properties"]
    required = edit_response.get("required", [])

    assert "edited_text" in props, "EditResponse.edited_text missing"
    assert "edited_text" in required, "EditResponse.edited_text must be required"


def test_health_response_schema_has_status() -> None:
    """HealthResponse содержит обязательное поле status."""
    schemas = _openapi()["components"]["schemas"]
    assert "HealthResponse" in schemas, "HealthResponse schema missing"

    health_response = schemas["HealthResponse"]
    props = health_response["properties"]
    required = health_response.get("required", [])

    assert "status" in props, "HealthResponse.status missing"
    assert "status" in required, "HealthResponse.status must be required"
