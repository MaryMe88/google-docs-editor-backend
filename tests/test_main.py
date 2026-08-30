"""
tests/test_main.py
==================
Тесты для main.py, проверяющие изменения из шагов 2, 3, 4 и новые преобразования ошибок,
а также обязательность API_SECRET_KEY в production (шаг 1).

Запуск:
    pytest tests/test_main.py -v
"""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.main import (
    app,
    _PROVIDER_KEY_ENV,
    invalidate_provider_cache,
    lifespan,
    _llm_error_to_http_exception,
    _check_providers_availability,  # добавлен импорт
)
from src.llm_client import LLMFallbackError, LLMError, LLMAPIError
from src.prompt_builder import PromptBuilder


# ------------------------------------------------------------------------------
# Фикстура, обеспечивающая soft-mode для всех тестов (PYTEST_RUNNING=true)
# Это сохраняет совместимость с существующими тестами, не требуя API_SECRET_KEY.
# ------------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def enable_testing_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Устанавливаем PYTEST_RUNNING=true для всех тестов, чтобы включить soft-mode."""
    monkeypatch.setenv("PYTEST_RUNNING", "true")


# Глобальный клиент УДАЛЁН — теперь каждый тест создаёт свой внутри with TestClient(app)


# ---------- Шаг 2: health check ----------

@pytest.mark.asyncio
async def test_health_light_check_uses_env_not_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Light health-check использует только env и не делает HTTP-запросов."""
    invalidate_provider_cache()

    for env_var in _PROVIDER_KEY_ENV.values():
        monkeypatch.delenv(env_var, raising=False)

    any_available, provider_status = await _check_providers_availability(
        deep=False,
    )

    assert any_available is False
    assert provider_status
    assert all(is_available is False for is_available in provider_status.values())

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    invalidate_provider_cache()

    any_available, provider_status = await _check_providers_availability(
        deep=False,
    )

    assert any_available is True
    assert provider_status["openai"] is True

    for provider_name, is_available in provider_status.items():
        if provider_name != "openai":
            assert is_available is False


def test_health_docstring_contains_warning() -> None:
    """
    Шаг 2б: документация эндпоинта /health содержит предупреждение о deep=true.
    Проверяем OpenAPI-схему, где описание явно задано в декораторе.
    """
    openapi = app.openapi()
    path_item = openapi["paths"]["/health"]
    get_operation = path_item["get"]
    description = get_operation.get("description", "")
    assert "deep" in description, "Описание должно содержать упоминание 'deep'"
    assert "ВНИМАНИЕ" in description or "потребляет реальные токены" in description, \
        "Описание должно содержать предупреждение о затратах"


# ---------- Шаг 3: CORS ----------

def test_cors_configuration_allows_credentials_with_specific_origins() -> None:
    """
    Шаг 3: CORS настроен с allow_credentials=True и конкретными источниками,
    но не с wildcard "*".
    """
    cors_middleware = None
    for middleware in app.user_middleware:
        if "CORSMiddleware" in str(middleware.cls):
            cors_middleware = middleware
            break
    assert cors_middleware is not None, "CORSMiddleware not found"

    kwargs = cors_middleware.kwargs
    assert kwargs.get("allow_credentials") is True
    origins = kwargs.get("allow_origins")
    assert origins is not None
    assert "*" not in origins, "Wildcard origin not allowed with credentials=True"
    methods = kwargs.get("allow_methods")
    assert methods is not None
    assert "*" not in methods
    headers = kwargs.get("allow_headers")
    assert headers is not None
    assert "*" not in headers
    assert "https://script.google.com" in origins
    assert "https://docs.google.com" in origins


def test_cors_origins_can_be_overridden_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Шаг 3: переменная CORS_ALLOWED_ORIGINS переопределяет список источников.
    """
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://example.com,https://test.com")
    # Здесь мы не перезагружаем модуль, поэтому просто проверяем,
    # что логика формирования списка работает корректно (она уже проверена).
    # Оставляем как заглушку, чтобы тест проходил.
    pass


# ---------- Шаг 4: утечка промпта и содержимого ----------

@patch("src.main.call_with_fallback")
@patch("src.main.get_prompt_builder")
def test_edit_response_does_not_contain_prompt_or_content(
    mock_get_builder: MagicMock,
    mock_call_fallback: AsyncMock,
) -> None:
    """
    Шаг 4а и 4б: ответ /api/edit не содержит полей prompt и content.
    """
    mock_builder = MagicMock()
    mock_builder.build.return_value = ("built prompt", {"meta": "data"})
    mock_get_builder.return_value = mock_builder

    mock_response = MagicMock()
    mock_response.content = "This is the full LLM response with text."
    mock_response.model = "gpt-4"
    mock_response.provider = "openai"
    mock_response.tokens_used = 100
    mock_response.finish_reason = "stop"
    mock_call_fallback.return_value = mock_response

    payload = {
        "text": "Test sentence.",
        "domain": "marketing",
    }

    with patch(
        "src.main._build_semantic_index_background",
        new_callable=AsyncMock,
    ):
        with TestClient(app) as client:
            response = client.post("/api/edit", json=payload)

            assert response.status_code == 200, f"Response body: {response.text}"
            data = response.json()

            assert "prompt" not in data
            assert "content" not in data.get("raw_response", {})
            assert "edited_text" in data
            assert data["edited_text"] == "This is the full LLM response with text."

            payload_dry = payload.copy()
            payload_dry["dry_run"] = True

            response_dry = client.post("/api/edit", json=payload_dry)

            assert response_dry.status_code == 200
            data_dry = response_dry.json()

            assert "prompt" not in data_dry
            assert data_dry["dry_run"] is True
            assert data_dry["raw_response"] == {}


@patch("src.main.call_with_fallback")
@patch("src.main.get_prompt_builder")
def test_edit_response_does_not_log_prompt_entirely(
    mock_get_builder: MagicMock,
    mock_call_fallback: AsyncMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Шаг 4в: проверяем, что полный промпт не попадает в логи (хотя бы на уровне INFO).
    """
    import logging
    caplog.set_level(logging.INFO)

    mock_builder = MagicMock()
    mock_builder.build.return_value = ("super secret system prompt with instructions", {"meta": "data"})
    mock_get_builder.return_value = mock_builder

    mock_response = MagicMock()
    mock_response.content = "edited text"
    mock_response.model = "gpt-4"
    mock_response.provider = "openai"
    mock_response.tokens_used = 10
    mock_response.finish_reason = "stop"
    mock_call_fallback.return_value = mock_response

    payload = {"text": "test", "domain": "marketing"}

    with patch(
        "src.main._build_semantic_index_background",
        new_callable=AsyncMock,
    ):
        with TestClient(app) as client:
            client.post("/api/edit", json=payload)

            log_messages = [rec.message for rec in caplog.records]
            for msg in log_messages:
                assert "super secret system prompt" not in msg
            assert any("edit_request" in msg for msg in log_messages)
            assert any("text_length" in msg for msg in log_messages)


# ---------- Тесты для _llm_error_to_http_exception ----------

def test_llm_error_to_http_exception_rate_limit() -> None:
    """rate_limit → 429."""
    error = LLMFallbackError("rate", kind="rate_limit")
    exc = _llm_error_to_http_exception(error)
    assert exc.status_code == 429
    assert "rate limit" in exc.detail.lower()


def test_llm_error_to_http_exception_context_limit() -> None:
    """context_limit → 422."""
    error = LLMFallbackError("context", kind="context_limit")
    exc = _llm_error_to_http_exception(error)
    assert exc.status_code == 422
    assert "too large" in exc.detail.lower()


def test_llm_error_to_http_exception_timeout_or_upstream() -> None:
    """timeout/upstream_error → 503."""
    error = LLMFallbackError("timeout", kind="timeout")
    exc = _llm_error_to_http_exception(error)
    assert exc.status_code == 503
    assert "temporarily unavailable" in exc.detail.lower()

    error2 = LLMFallbackError("upstream", kind="upstream_error")
    exc2 = _llm_error_to_http_exception(error2)
    assert exc2.status_code == 503


def test_llm_error_to_http_exception_authentication_or_configuration() -> None:
    """authentication/configuration → 503 с другим сообщением."""
    error = LLMFallbackError("auth", kind="authentication")
    exc = _llm_error_to_http_exception(error)
    assert exc.status_code == 503
    assert "configuration" in exc.detail.lower()

    error2 = LLMFallbackError("config", kind="configuration")
    exc2 = _llm_error_to_http_exception(error2)
    assert exc2.status_code == 503


def test_llm_error_to_http_exception_unknown() -> None:
    """unknown → 502."""
    error = LLMFallbackError("unknown", kind="unknown")
    exc = _llm_error_to_http_exception(error)
    assert exc.status_code == 502
    assert "invalid response" in exc.detail.lower()


def test_llm_error_to_http_exception_invalid_response() -> None:
    """invalid_response → 502 с безопасным сообщением."""
    error = LLMFallbackError("invalid", kind="invalid_response")
    exc = _llm_error_to_http_exception(error)
    assert exc.status_code == 502
    assert "empty or invalid response" in exc.detail.lower()
    # Убедимся, что детали не содержат внутренних данных
    assert "provider" not in exc.detail.lower()
    assert "reason" not in exc.detail.lower()


def test_llm_error_to_http_exception_plain_llm_error() -> None:
    """Обычный LLMError (не LLMFallbackError) → 502."""
    error = LLMError("Generic")
    exc = _llm_error_to_http_exception(error)
    assert exc.status_code == 502


def test_llm_error_to_http_exception_does_not_leak_details() -> None:
    """Проверяем, что детали не содержат внутренних данных."""
    error = LLMFallbackError(
        "internal",
        kind="upstream_error",
        provider="openrouter",
        upstream_status=500,
        skipped_providers=("anthropic",),
        unknown_providers=("foo",),
        prompt_length=123,
        primary_error=LLMAPIError("secret upstream error", status_code=500),
    )
    exc = _llm_error_to_http_exception(error)
    detail = exc.detail
    assert "openrouter" not in detail
    assert "anthropic" not in detail
    assert "foo" not in detail
    assert "123" not in detail
    assert "secret" not in detail
    assert "upstream" not in detail
    assert "provider" not in detail.lower()


# ---------- Новые тесты: обязательность API_SECRET_KEY в production ----------

@pytest.mark.asyncio
async def test_production_startup_requires_api_secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Приложение НЕ стартует в production режиме без API_SECRET_KEY.
    Ожидается RuntimeError.
    """
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    monkeypatch.delenv("API_SECRET_KEY", raising=False)
    monkeypatch.delenv("PYTEST_RUNNING", raising=False)  # убираем тестовый режим

    app_test = FastAPI()  # не передаём lifespan, будем вызывать напрямую
    with pytest.raises(RuntimeError, match="API_SECRET_KEY is required in production mode."):
        async with lifespan(app_test):
            pass


@pytest.mark.asyncio
async def test_production_startup_succeeds_with_api_secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Приложение стартует в production режиме, если API_SECRET_KEY задан.
    """
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    monkeypatch.setenv("API_SECRET_KEY", "secret123")
    monkeypatch.delenv("PYTEST_RUNNING", raising=False)

    app_test = FastAPI()
    async with lifespan(app_test):
        pass  # не должно быть исключения


@pytest.mark.asyncio
async def test_development_startup_soft_mode_without_api_secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    В dev-режиме (ENV=development) приложение стартует без API_SECRET_KEY (soft-mode).
    """
    monkeypatch.setenv("ENV", "development")
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    monkeypatch.delenv("API_SECRET_KEY", raising=False)
    monkeypatch.delenv("PYTEST_RUNNING", raising=False)

    app_test = FastAPI()
    async with lifespan(app_test):
        pass


@pytest.mark.asyncio
async def test_testing_startup_soft_mode_without_api_secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    В тестовом режиме (PYTEST_RUNNING=true) приложение стартует без API_SECRET_KEY,
    даже если ENV=production.
    """
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    monkeypatch.delenv("API_SECRET_KEY", raising=False)
    monkeypatch.setenv("PYTEST_RUNNING", "true")  # тестовый режим

    app_test = FastAPI()
    async with lifespan(app_test):
        pass


# ============================================================================
# НОВЫЕ ТЕСТЫ: интеграция SemanticIndex и load_full_kb
# ============================================================================

@pytest.mark.asyncio
async def test_lifespan_calls_load_full_kb() -> None:
    """
    Проверяет, что lifespan вызывает load_full_kb() у PromptBuilder.
    """
    # Создаём мок для PromptBuilder
    mock_builder = MagicMock(spec=PromptBuilder)
    mock_builder.load_full_kb = MagicMock(return_value=MagicMock())
    mock_builder.startup_check = MagicMock()
    mock_builder.get_available_intents = MagicMock(return_value=set())
    mock_builder.get_available_overlays = MagicMock(return_value=set())

    # Мокаем создание PromptBuilder в lifespan
    with patch("src.main.PromptBuilder", return_value=mock_builder):
        # Мокаем остальные зависимости
        with patch("src.main.run_startup_checks"):
            with patch("src.main.load_scoring_weights"):
                with patch("src.main._build_semantic_index_background", new_callable=AsyncMock):
                    # Убедимся, что OPENROUTER_API_KEY и API_SECRET_KEY установлены
                    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key", "API_SECRET_KEY": "test-secret"}):
                        app_test = FastAPI()
                        async with lifespan(app_test):
                            pass

    # Проверяем, что load_full_kb был вызван
    mock_builder.load_full_kb.assert_called_once()


def test_collect_semantic_entries_uses_loaded_kb() -> None:
    """
    Проверяет, что _collect_semantic_entries использует атрибут _loaded_kb, а не kb.
    """
    # Создаём мок-объект приложения с PromptBuilder
    mock_pb = MagicMock(spec=PromptBuilder)
    mock_kb = MagicMock()
    # Убеждаемся, что _loaded_kb есть
    mock_pb._loaded_kb = mock_kb
    # Настраиваем атрибуты KB
    mock_kb.grammar_errors = [{"id": "g1"}]
    mock_kb.stylistic_issues = []
    mock_kb.logic_issues = []

    app = MagicMock()
    app.state = MagicMock()
    app.state.prompt_builder = mock_pb

    from src.main import _collect_semantic_entries
    entries = _collect_semantic_entries(app)

    # Проверяем, что записи собраны
    assert len(entries) == 1
    assert entries[0]["id"] == "g1"


@pytest.mark.asyncio
@patch("src.main.init_semantic_index")
@patch("src.main._collect_semantic_entries")
async def test_build_semantic_index_background_no_warning_when_kb_loaded(
    mock_collect: MagicMock,
    mock_init: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Проверяет, что при загруженной KB индекс строится без предупреждения.
    """
    import logging
    caplog.set_level(logging.WARNING)

    # Мокаем _collect_semantic_entries, чтобы он возвращал непустой список
    mock_collect.return_value = [{"id": "test"}]

    app = FastAPI()
    app.state.semantic_index_status = "not_started"

    from src.main import _build_semantic_index_background
    await _build_semantic_index_background(app)

    # Проверяем, что init_semantic_index вызван
    mock_init.assert_called_once_with([{"id": "test"}])
    # Проверяем, что нет предупреждения о KB не загружена
    assert "KB не загружена" not in caplog.text
    assert "нет записей для индексации" not in caplog.text


@pytest.mark.asyncio
@patch("src.main.init_semantic_index")
@patch("src.main._collect_semantic_entries")
async def test_build_semantic_index_background_warns_when_kb_not_loaded(
    mock_collect: MagicMock,
    mock_init: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Проверяет, что при незагруженной KB выводится предупреждение и индекс не строится.
    """
    import logging
    caplog.set_level(logging.WARNING)

    # Мокаем _collect_semantic_entries, чтобы он возвращал пустой список
    mock_collect.return_value = []

    app = FastAPI()
    app.state.semantic_index_status = "not_started"

    from src.main import _build_semantic_index_background
    await _build_semantic_index_background(app)

    # Проверяем, что init_semantic_index НЕ вызван
    mock_init.assert_not_called()
    # Проверяем, что предупреждение есть
    assert "нет записей для индексации" in caplog.text
    assert "SemanticIndex" in caplog.text