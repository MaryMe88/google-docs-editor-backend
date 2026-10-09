"""
ASGI middleware: reject HTTP requests with an oversized body.

Работает на уровне ASGI (между uvicorn и FastAPI), поэтому:
- покрывает и Content-Length, и chunked transfer-encoding;
- не читает тело целиком в память при превышении лимита;
- не ломает downstream: тело проходит через receive() порциями.

Ограничение задаётся в байтах. Значение по умолчанию —
DEFAULT_MAX_BODY_SIZE (256 KB), достаточное с большим запасом
для валидных запросов (текст до 10 000 символов UTF-8 ≈ 20 KB).

Запросы с телом больше лимита отклоняются со статусом 413 и
заголовком Connection: close. Тело запроса в логи не выводится.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MAX_BODY_SIZE: int = 256 * 1024  # 256 KB

Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]
Scope = dict[str, Any]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


class _BodyTooLarge(Exception):
    """
    Внутреннее исключение для прерывания чтения тела.

    Не должно утекать наружу: middleware ловит его и превращает
    в HTTP-ответ 413. Downstream (FastAPI) увидит исключение как
    обычное, если не успел его перехватить, — middleware ловит
    раньше.
    """

    def __init__(self, received: int) -> None:
        super().__init__(f"Body too large: {received} bytes")
        self.received = received


class BodySizeLimitMiddleware:
    """
    ASGI middleware, отклоняющий запросы с телом больше max_body_size.

    Стратегия:

    1. Если в scope есть валидный Content-Length и он больше лимита —
       сразу отвечаем 413, не читая тело.
    2. Иначе оборачиваем receive() в счётчик. Считаем длину каждой
       порции body. При превышении лимита — выбрасываем внутреннее
       исключение и отвечаем 413.
    3. Невалидный Content-Length (не парсится как int) — не отклоняем,
       передаём downstream. Это не наша задача — валидация заголовков.
    4. Non-HTTP scope (lifespan, websocket) — пропускаем без изменений.

    Тело в логи не выводится. Логируются только method, path,
    received_bytes, limit — для отладки и мониторинга.
    """

    def __init__(
        self,
        app: ASGIApp,
        max_body_size: int = DEFAULT_MAX_BODY_SIZE,
    ) -> None:
        if max_body_size <= 0:
            raise ValueError("max_body_size must be a positive integer")
        self.app = app
        self.max_body_size = max_body_size

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        content_length = self._parse_content_length(scope)
        if content_length is not None and content_length > self.max_body_size:
            await self._send_413(scope, send, content_length)
            return

        received_bytes = 0

        async def limited_receive() -> dict[str, Any]:
            nonlocal received_bytes
            message = await receive()
            if message.get("type") == "http.request":
                body = message.get("body", b"") or b""
                received_bytes += len(body)
                if received_bytes > self.max_body_size:
                    raise _BodyTooLarge(received_bytes)
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _BodyTooLarge as exc:
            await self._send_413(scope, send, exc.received)

    @staticmethod
    def _parse_content_length(scope: Scope) -> int | None:
        """
        Возвращает Content-Length как int или None.

        None возвращается и когда заголовка нет, и когда он невалиден:
        в обоих случаях дальше работает потоковый счётчик.
        """
        for name, value in scope.get("headers", ()):
            if name == b"content-length":
                try:
                    parsed = int(value)
                except (TypeError, ValueError):
                    return None
                return parsed if parsed >= 0 else None
        return None

    async def _send_413(
        self,
        scope: Scope,
        send: Send,
        received: int,
    ) -> None:
        logger.warning(
            "Request body too large: method=%s path=%s received=%d limit=%d",
            scope.get("method", "?"),
            scope.get("path", "?"),
            received,
            self.max_body_size,
        )
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"connection", b"close"),
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": b'{"detail":"Request body too large."}',
            }
        )
