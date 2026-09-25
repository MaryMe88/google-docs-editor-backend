"""
tests/test_api_smoke.py
=======================
Итерация 3 дорожной карты: smoke-тесты API.

Минимальный набор проверок, фиксирующий контракты FastAPI-приложения
до будущего разделения main.py на слои (итерация 8).

Запуск:
    pytest tests/test_api_smoke.py -v
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.main import app

# -----------------------------------------------------------------------------
# 1. Импорт приложения
# -----------------------------------------------------------------------------


def test_app_import_exposes_fastapi_instance() -> None:
    """src.main:app — это FastAPI, импорт не падает."""
    assert isinstance(app, FastAPI)


def test_import_main_does_not_load_semantic_model() -> None:
    """
    Импорт src.main в изолированном интерпретаторе не должен тянуть
    sentence_transformers (тяжёлую модель). Модель загружается только
    при первом запросе с deep_semantic_search=true.
    """
    project_root = Path(__file__).resolve().parent.parent
    code = (
        "import sys\n"
        "sys.path.insert(0, '.')\n"
        "import src.main  # noqa: F401\n"
        "assert 'sentence_transformers' not in sys.modules, "
        "'sentence_transformers was imported by src.main'\n"
        "print('ok')\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(project_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"subprocess failed (rc={result.returncode})\n"
        f"stdout: {result.stdout!r}\n"
        f"stderr: {result.stderr!r}"
    )
    assert "ok" in result.stdout


# -----------------------------------------------------------------------------
# 2. /livez и /health
# -----------------------------------------------------------------------------


def test_livez_returns_alive_without_openrouter() -> None:
    """/livez отвечает 200 без реального OpenRouter."""
    with TestClient(app) as client:
        resp = client.get("/livez")
    assert resp.status_code == 200
    assert resp.json() == {"status": "alive"}


def test_health_returns_expected_structure() -> None:
    """/health возвращает ожидаемую структуру HealthResponse."""
    from src.main import invalidate_provider_cache

    invalidate_provider_cache()

    with TestClient(app) as client:
        resp = client.get("/health")

    assert resp.status_code in (200, 503)
    data = resp.json()

    expected_keys = {
        "status",
        "version",
        "available_domains",
        "available_intents",
        "available_overlays",
        "available_providers",
        "provider_status",
        "deep_check",
        "contract_version",
    }
    assert expected_keys <= set(data.keys())
    assert data["status"] in ("ok", "degraded")
    assert data["deep_check"] is False
    assert isinstance(data["provider_status"], dict)
    assert isinstance(data["available_domains"], list)
    assert isinstance(data["available_providers"], list)


# -----------------------------------------------------------------------------
# 3. /api/edit — отклонение невалидного запроса
# -----------------------------------------------------------------------------


def test_edit_rejects_empty_body() -> None:
    """Пустое тело /api/edit → 422 (Pydantic-валидация)."""
    with TestClient(app) as client:
        resp = client.post("/api/edit", json={})
    assert resp.status_code == 422


def test_edit_rejects_missing_text_field() -> None:
    """Отсутствие обязательного поля text → 422."""
    with TestClient(app) as client:
        resp = client.post("/api/edit", json={"domain": "basic_edit"})
    assert resp.status_code == 422


def test_edit_rejects_wrong_type_for_text() -> None:
    """text должен быть строкой; число → 422."""
    with TestClient(app) as client:
        resp = client.post(
            "/api/edit",
            json={"text": 12345, "domain": "basic_edit"},
        )
    assert resp.status_code == 422


# -----------------------------------------------------------------------------
# 4. Аутентификация
# -----------------------------------------------------------------------------


def test_wrong_api_key_returns_401_with_expected_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Неверный X-API-Key → 401, фиксированный формат ошибки, WWW-Authenticate.
    """
    monkeypatch.setenv("API_SECRET_KEY", "correct-key-for-test")
    with TestClient(app) as client:
        resp = client.get("/health", headers={"X-API-Key": "wrong-key"})
    assert resp.status_code == 401
    data = resp.json()
    assert data["detail"] == "Invalid API key."
    assert resp.headers.get("WWW-Authenticate") == "ApiKey"


def test_missing_api_key_returns_401(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Отсутствие X-API-Key при заданном API_SECRET_KEY → 401.
    """
    monkeypatch.setenv("API_SECRET_KEY", "correct-key-for-test")
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 401
    data = resp.json()
    assert data["detail"] == "Missing API key."
    assert resp.headers.get("WWW-Authenticate") == "ApiKey"
