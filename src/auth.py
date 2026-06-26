from __future__ import annotations

import os
import secrets
from typing import Optional

from fastapi import Header, HTTPException, status


def verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> None:
    """
    Проверяет X-API-Key header.

    - Если переменная окружения API_SECRET_KEY не задана — пропускает (soft mode).
    - Если задана — требует наличия заголовка и сравнивает ключи.
    - При несовпадении или отсутствии заголовка возвращает 401 Unauthorized.
    """
    expected = os.environ.get("API_SECRET_KEY", "")
    if not expected:
        # Ключ не задан — пропускаем (для совместимости с деплоями без ключа)
        return
    if x_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    if not secrets.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )