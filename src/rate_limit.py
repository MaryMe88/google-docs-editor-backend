"""
rate_limit

Rate limiting infrastructure для FastAPI-приложения: Limiter,
ключ по IP-адресу клиента, строка лимита.

Выделено из src.main в итерации 8 дорожной карты, потому что
@limiter.limit(...) на роутере требует доступа к Limiter на этапе
импорта модуля, что исключает его нахождение в main.py при вынесении
/api/edit в отдельный router.
"""

from __future__ import annotations

import os

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

_is_testing = os.getenv("PYTEST_RUNNING", "false").lower() == "true"
RATE_LIMIT = "1000/minute" if _is_testing else "10/minute"
# /health вызывается внутренними healthcheck-ами Render + диагностикой.
# В prod даём запас, чтобы healthcheck не попадал под лимит /api/edit.
HEALTH_RATE_LIMIT = "1000/minute" if _is_testing else "60/minute"


def _client_ip_key(request: Request) -> str:
    return get_remote_address(request)


# headers_enabled=True — slowapi отдаёт Retry-After и X-RateLimit-*
# в 429-ответах. Клиент (Apps Script) может подождать Retry-After сек.
limiter = Limiter(key_func=_client_ip_key, headers_enabled=True)
