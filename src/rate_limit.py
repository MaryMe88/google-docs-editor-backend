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


def _client_ip_key(request: Request) -> str:
    return get_remote_address(request)


limiter = Limiter(key_func=_client_ip_key)
