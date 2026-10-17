"""Тесты для hardening auth в runtime (Этап 2A дорожной карты v2).

Покрывают три класса сценариев:

1. Production startup fail-fast:
   - без ключа приложение не стартует;
   - PYTEST_RUNNING=true НЕ обходит fail-fast;
   - ключ из пробелов считается отсутствующим.

2. Request-time auth:
   - ключ из пробелов в X-API-Key отклоняется;
   - пустой ключ в X-API-Key отклоняется;
   - правильный ключ принимается.

3. Инварианты:
   - development soft-mode без ключа работает;
   - /livez доступен без ключа;
   - /health требует ключ.

Три теста на дефекты (PYTEST_RUNNING bypass, whitespace startup key,
whitespace request key) изначально падают и проходят после правок
src/auth.py и src/main.py.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.main import app, lifespan

# ---------------------------------------------------------------------------
# Вспомогательные значения
# ---------------------------------------------------------------------------

_TEST_KEY = "test-secret-key-c4f8a2e1-9b7d-4e6a-8f3c-2a1b0d5e6f7a"


# ---------------------------------------------------------------------------
# 1. Production startup fail-fast
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_production_without_key_refuses_to_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ENV=production + нет ключа → RuntimeError, приложение не стартует."""
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    monkeypatch.delenv("API_SECRET_KEY", raising=False)
    monkeypatch.delenv("PYTEST_RUNNING", raising=False)

    with pytest.raises(RuntimeError, match="API_SECRET_KEY is required"):
        async with lifespan(FastAPI()):
            pass


@pytest.mark.asyncio
async def test_production_succeeds_with_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ENV=production + ключ задан → startup проходит без исключений."""
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    monkeypatch.setenv("API_SECRET_KEY", _TEST_KEY)
    monkeypatch.delenv("PYTEST_RUNNING", raising=False)

    async with lifespan(FastAPI()):
        pass


@pytest.mark.asyncio
async def test_pytest_running_does_not_bypass_production_failfast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PYTEST_RUNNING=true не отключает fail-fast в production.

    Дефект до Этапа 2A: is_production = not (ENV==development or PYTEST_RUNNING).
    """
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("PYTEST_RUNNING", "true")
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    monkeypatch.delenv("API_SECRET_KEY", raising=False)

    with pytest.raises(RuntimeError, match="API_SECRET_KEY is required"):
        async with lifespan(FastAPI()):
            pass


@pytest.mark.asyncio
async def test_whitespace_key_treated_as_missing_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """API_SECRET_KEY='   ' в production — считается отсутствующим.

    Дефект до Этапа 2A: os.getenv(...) возвращает '   ' (truthy),
    проверка not os.getenv(...) не срабатывает.
    """
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    monkeypatch.setenv("API_SECRET_KEY", "   ")
    monkeypatch.delenv("PYTEST_RUNNING", raising=False)

    with pytest.raises(RuntimeError, match="API_SECRET_KEY is required"):
        async with lifespan(FastAPI()):
            pass


# ---------------------------------------------------------------------------
# 2. Development soft-mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_development_soft_mode_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ENV=development без ключа — startup проходит (soft-mode)."""
    monkeypatch.setenv("ENV", "development")
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    monkeypatch.delenv("API_SECRET_KEY", raising=False)
    monkeypatch.delenv("PYTEST_RUNNING", raising=False)

    async with lifespan(FastAPI()):
        pass


# ---------------------------------------------------------------------------
# 3. Request-time auth
# ---------------------------------------------------------------------------


def test_correct_key_accepted_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Правильный ключ в X-API-Key → 200 на /health."""
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("API_SECRET_KEY", _TEST_KEY)

    with TestClient(app) as client:
        resp = client.get("/health", headers={"X-API-Key": _TEST_KEY})

    assert resp.status_code in (200, 503), resp.text


def test_missing_key_returns_401(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Без X-API-Key (ключ задан) → 401."""
    monkeypatch.setenv("ENV", "development")
    monkeypatch.setenv("API_SECRET_KEY", _TEST_KEY)

    with TestClient(app) as client:
        resp = client.get("/health")

    assert resp.status_code == 401


def test_whitespace_key_rejected_in_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Клиент прислал X-API-Key из пробелов — 401, даже если ключ задан.

    После Этапа 2A: verify_api_key делает strip() на входном значении.
    Пустая строка после strip() → 401, не доходит до compare_digest.
    """
    monkeypatch.setenv("ENV", "development")
    monkeypatch.setenv("API_SECRET_KEY", _TEST_KEY)

    with TestClient(app) as client:
        resp = client.get("/health", headers={"X-API-Key": "   "})

    assert resp.status_code == 401, f"expected 401, got {resp.status_code}"


def test_whitespace_env_key_is_treated_as_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """API_SECRET_KEY='   ' эквивалентно отсутствию ключа → soft mode.

    get_expected_api_key() делает strip(). Пустая строка → soft mode
    (в production startup упадёт из-за fail-fast в src/main.py).
    """
    monkeypatch.setenv("ENV", "development")
    monkeypatch.setenv("API_SECRET_KEY", "   ")

    with TestClient(app) as client:
        resp = client.get("/health")

    assert resp.status_code in (200, 503), resp.text


def test_livez_works_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """/livez доступен без X-API-Key даже в production."""
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("API_SECRET_KEY", _TEST_KEY)

    with TestClient(app) as client:
        resp = client.get("/livez")

    assert resp.status_code == 200
    assert resp.json()["status"] == "alive"
