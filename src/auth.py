from __future__ import annotations

import logging
import os
import secrets
from typing import Optional

from fastapi import Header, HTTPException, status

logger = logging.getLogger(__name__)

# Одноразовое предупреждение о soft-auth: чтобы не спамить лог на каждый
# запрос, пишем его максимум один раз за время жизни процесса.
_soft_auth_warned = False


def verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> None:
    """
    Проверяет X-API-Key header.

    - Если переменная окружения API_SECRET_KEY не задана — пропускает (soft mode).
    - Если задана — требует наличия заголовка и сравнивает ключи.
    - При несовпадении или отсутствии заголовка возвращает 401 Unauthorized.
    """
    global _soft_auth_warned
    expected = os.environ.get("API_SECRET_KEY", "")
    if not expected:
        # Ключ не задан — пропускаем (для совместимости с деплоями без ключа).
        # Сигнализируем один раз, чтобы открытый доступ не остался
        # незамеченным на production.
        if not _soft_auth_warned:
            logger.warning(
                "API_SECRET_KEY не задан: защищённые эндпоинты работают без "
                "аутентификации (soft mode). Задайте API_SECRET_KEY для защиты "
                "в production."
            )
            _soft_auth_warned = True
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