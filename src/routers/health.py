"""
routers.health

Health endpoints: /, /livez, /health.

Выделено из src.main в итерации 8 дорожной карты.
"""

from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse

from src.auth import verify_api_key
from src.contracts import CONTRACT_VERSION, HealthResponse
from src.prompt_builder import PromptBuilder
from src.rate_limit import HEALTH_RATE_LIMIT, limiter
from src.services.provider_health import _check_providers_availability
from src.shared_contracts import ALLOWED_DOMAINS

router = APIRouter()


def _get_pb(request: Request) -> PromptBuilder:
    """Возвращает PromptBuilder, инициализированный в lifespan."""
    prompt_builder = getattr(request.app.state, "prompt_builder", None)
    if prompt_builder is None:
        raise RuntimeError("PromptBuilder is not initialized")
    return prompt_builder


@router.get("/")
async def root() -> dict:
    return {"status": "ok"}


@router.get("/livez")
async def liveness_check() -> dict:
    return {"status": "alive"}


@router.get(
    "/health",
    response_model=HealthResponse,
    dependencies=[Depends(verify_api_key)],
    description="""
Проверка состояния сервиса.

- deep=false (по умолчанию): проверяет только наличие API-ключей в env.
- deep=true: выполняет реальный тестовый запрос к каждому LLM-провайдеру.
  ВНИМАНИЕ: deep=true потребляет реальные токены и может тарифицироваться.
  Использовать только для диагностики, не в автоматическом мониторинге.
""",
)
@limiter.limit(HEALTH_RATE_LIMIT)
async def health_check(
    request: Request,
    response: Response,
    deep: bool = False,
) -> Response:
    if deep and not os.getenv("API_SECRET_KEY"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API_SECRET_KEY is required for deep health check.",
        )

    builder = _get_pb(request)
    any_available, provider_status = await _check_providers_availability(deep=deep)

    health = HealthResponse(
        status="ok" if any_available else "degraded",
        version="1.0.0",
        available_domains=sorted(ALLOWED_DOMAINS),
        available_intents=list(builder.get_available_intents()),
        available_overlays=list(builder.get_available_overlays()),
        available_providers=[provider for provider, ok in provider_status.items() if ok],
        provider_status=provider_status,
        deep_check=deep,
        contract_version=CONTRACT_VERSION,
    )

    status_code = status.HTTP_200_OK if any_available else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(
        content=health.model_dump(),
        status_code=status_code,
    )
