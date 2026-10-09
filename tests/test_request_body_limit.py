"""
Тесты итерации 2: ASGI middleware BodySizeLimitMiddleware.

Две группы:

1. Unit-тесты через прямой вызов middleware — полный контроль над
   scope и receive. Здесь проверяем и chunked, и «Content-Length
   больше лимита — не читаем тело», и заголовки 413-ответа.

2. Интеграционные тесты через TestClient — проверяем, что middleware
   действительно включён в app и не пропускает большие запросы.

Запуск:
    pytest tests/test_request_body_limit.py -v
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.middleware.body_size import (
    DEFAULT_MAX_BODY_SIZE,
    BodySizeLimitMiddleware,
)

# ============================================================================
# Вспомогательная обвязка для прямого вызова middleware
# ============================================================================


async def _run_middleware(
    *,
    max_body_size: int,
    content_length: bytes | None,
    messages: list[dict[str, Any]],
    path: str = "/api/edit",
) -> tuple[list[dict[str, Any]], int]:
    """
    Прогоняет middleware на фейковом ASGI-приложении.

    Аргументы:
        max_body_size: лимит в байтах.
        content_length: значение заголовка Content-Length или None.
        messages: последовательность сообщений, которые вернёт receive().
        path: путь запроса (для логов).

    Возвращает:
        (список отправленных сообщений, число вызовов receive).
    """
    headers: list[tuple[bytes, bytes]] = []
    if content_length is not None:
        headers.append((b"content-length", content_length))

    scope: dict[str, Any] = {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": headers,
        "query_string": b"",
    }

    sent: list[dict[str, Any]] = []
    receive_idx = [0]

    async def receive() -> dict[str, Any]:
        i = receive_idx[0]
        if i >= len(messages):
            return {"type": "http.disconnect"}
        receive_idx[0] += 1
        return messages[i]

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    async def dummy_app(scope_: Any, receive_: Any, send_: Any) -> None:
        """Просто читает тело до конца и отвечает 200."""
        while True:
            msg = await receive_()
            if msg["type"] == "http.request" and not msg.get("more_body", False):
                break
            if msg["type"] == "http.disconnect":
                return
        await send_(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [],
            }
        )
        await send_({"type": "http.response.body", "body": b"ok"})

    middleware = BodySizeLimitMiddleware(dummy_app, max_body_size=max_body_size)
    await middleware(scope, receive, send)
    return sent, receive_idx[0]


def _status_of(sent: list[dict[str, Any]]) -> int:
    """Достать HTTP-статус из первого response.start-сообщения."""
    for msg in sent:
        if msg["type"] == "http.response.start":
            return int(msg["status"])
    raise AssertionError(f"no http.response.start in {sent!r}")


def _headers_of(sent: list[dict[str, Any]]) -> dict[bytes, bytes]:
    for msg in sent:
        if msg["type"] == "http.response.start":
            return dict(msg.get("headers", []))
    raise AssertionError(f"no http.response.start in {sent!r}")


# ============================================================================
# Unit-тесты middleware
# ============================================================================


@pytest.mark.asyncio
async def test_small_body_passes() -> None:
    """Тело 100 байт при лимите 1 KB — проходит, downstream отвечает 200."""
    sent, receive_calls = await _run_middleware(
        max_body_size=1024,
        content_length=b"100",
        messages=[
            {"type": "http.request", "body": b"x" * 100, "more_body": False},
        ],
    )
    assert _status_of(sent) == 200
    assert receive_calls == 1


@pytest.mark.asyncio
async def test_body_at_limit_passes() -> None:
    """Тело ровно в лимит — проходит (условие 'строго больше')."""
    sent, _ = await _run_middleware(
        max_body_size=1024,
        content_length=b"1024",
        messages=[
            {"type": "http.request", "body": b"x" * 1024, "more_body": False},
        ],
    )
    assert _status_of(sent) == 200


@pytest.mark.asyncio
async def test_content_length_above_limit_rejected_without_reading_body() -> None:
    """
    Content-Length > лимита — 413, и тело НЕ читается (receive не вызывается).
    Это важная оптимизация: не тратим память на большое тело.
    """
    sent, receive_calls = await _run_middleware(
        max_body_size=1024,
        content_length=b"9999999",
        messages=[],  # вообще не должно читаться
    )
    assert _status_of(sent) == 413
    assert receive_calls == 0


@pytest.mark.asyncio
async def test_413_response_shape() -> None:
    """Тело ответа при 413 — ожидаемый JSON, заголовок Connection: close."""
    sent, _ = await _run_middleware(
        max_body_size=1024,
        content_length=b"9999999",
        messages=[],
    )
    headers = _headers_of(sent)
    assert headers.get(b"connection") == b"close"
    assert headers.get(b"content-type") == b"application/json"

    body = b""
    for msg in sent:
        if msg["type"] == "http.response.body":
            body += msg.get("body", b"")
    assert body == b'{"detail":"Request body too large."}'


@pytest.mark.asyncio
async def test_invalid_content_length_passed_through() -> None:
    """
    Невалидный Content-Length (не число) — middleware не отклоняет,
    передаёт downstream. Валидация заголовков — не его работа.
    """
    sent, _ = await _run_middleware(
        max_body_size=1024,
        content_length=b"abc",
        messages=[
            {"type": "http.request", "body": b"ok", "more_body": False},
        ],
    )
    assert _status_of(sent) == 200


@pytest.mark.asyncio
async def test_chunked_body_above_limit_rejected() -> None:
    """
    Без Content-Length, два чанка по 200 KB, лимит 300 KB.
    Middleware считает потоково и отклоняет на втором чанке.
    """
    sent, _ = await _run_middleware(
        max_body_size=300 * 1024,
        content_length=None,
        messages=[
            {"type": "http.request", "body": b"x" * (200 * 1024), "more_body": True},
            {"type": "http.request", "body": b"x" * (200 * 1024), "more_body": False},
        ],
    )
    assert _status_of(sent) == 413


@pytest.mark.asyncio
async def test_chunked_body_within_limit_passes() -> None:
    """Chunked-тело в пределах лимита — проходит."""
    sent, _ = await _run_middleware(
        max_body_size=300 * 1024,
        content_length=None,
        messages=[
            {"type": "http.request", "body": b"x" * (100 * 1024), "more_body": True},
            {"type": "http.request", "body": b"x" * (100 * 1024), "more_body": False},
        ],
    )
    assert _status_of(sent) == 200


# ============================================================================
# Логирование
# ============================================================================


@pytest.mark.asyncio
async def test_body_not_in_logs_on_rejection(caplog: pytest.LogCaptureFixture) -> None:
    """Тело запроса НЕ попадает в логи, даже при отклонении."""
    secret = b"top-secret-payload-content"
    with caplog.at_level(logging.WARNING, logger="src.middleware.body_size"):
        await _run_middleware(
            max_body_size=4,
            content_length=None,
            messages=[
                {"type": "http.request", "body": secret, "more_body": False},
            ],
        )
    assert secret.decode() not in caplog.text


@pytest.mark.asyncio
async def test_warning_logged_with_metadata(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """При отклонении пишется WARNING с method/path/received/limit."""
    with caplog.at_level(logging.WARNING, logger="src.middleware.body_size"):
        await _run_middleware(
            max_body_size=1024,
            content_length=b"9999999",
            messages=[],
            path="/api/edit",
        )
    assert "Request body too large" in caplog.text
    assert "/api/edit" in caplog.text
    assert "9999999" in caplog.text  # received
    assert "1024" in caplog.text  # limit


# ============================================================================
# Константа лимита
# ============================================================================


def test_default_limit_is_256_kb() -> None:
    """Дефолтный лимит — ровно 256 KB."""
    assert DEFAULT_MAX_BODY_SIZE == 256 * 1024


# ============================================================================
# Интеграционные тесты через TestClient
# ============================================================================


def test_small_request_passes_through_middleware() -> None:
    """
    Небольшой POST на несуществующий путь → 404 (роутер ответил),
    а не 413 (middleware пропустил).
    """
    with TestClient(app) as client:
        resp = client.post("/definitely-not-a-route", content=b"x" * 100)
    assert resp.status_code != 413
    assert resp.status_code == 404


def test_large_request_rejected_by_middleware_integration() -> None:
    """
    POST на несуществующий путь с телом > 256 KB → 413.
    Middleware работает раньше роутинга, поэтому 404 не успевает.
    """
    big = b"x" * (300 * 1024)
    with TestClient(app) as client:
        resp = client.post("/definitely-not-a-route", content=big)
    assert resp.status_code == 413
    assert resp.json() == {"detail": "Request body too large."}
