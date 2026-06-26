from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from src.auth import verify_api_key
from src.config_types import AudienceProfile
from src.contracts import CONTRACT_VERSION, EditRequest, EditResponse, HealthResponse
from src.llm_client import LLMError, call_with_fallback, create_llm_client
from src.prompt_builder import PromptBuilder
from src.provider_registry import LLMProvider
from src.shared_contracts import (
    ALLOWED_DOMAINS,
    ALLOWED_INTENTS,
    ALLOWED_OVERLAYS,
    ALLOWED_PROVIDERS,
)
from src.startup_checks import run_startup_checks
from src.scoring_weights import load_scoring_weights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Кэш доступности провайдеров (A-5)
# ---------------------------------------------------------------------------
@dataclass
class _ProviderCacheEntry:
    available: bool
    checked_at: float = field(default_factory=time.monotonic)

    def is_fresh(self, ttl: float) -> bool:
        return (time.monotonic() - self.checked_at) < ttl


_provider_cache: dict[str, _ProviderCacheEntry] = {}
_PROVIDER_CACHE_TTL = 60.0


def _get_cached_availability(provider: str) -> bool | None:
    entry = _provider_cache.get(provider)
    if entry and entry.is_fresh(_PROVIDER_CACHE_TTL):
        return entry.available
    return None


def _set_cached_availability(provider: str, available: bool) -> None:
    _provider_cache[provider] = _ProviderCacheEntry(available=available)


def invalidate_provider_cache() -> None:
    """Принудительная инвалидация кэша (для тестов и диагностики)."""
    _provider_cache.clear()


# ---------------------------------------------------------------------------
# Маппинг провайдер → переменная окружения для light‑check
# ---------------------------------------------------------------------------
_PROVIDER_KEY_ENV: Dict[str, str] = {
    "perplexity": "PERPLEXITY_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
_CORS_ORIGINS_RAW = os.getenv("CORS_ALLOWED_ORIGINS", "")
_CORS_ORIGINS: list[str] = (
    [origin.strip() for origin in _CORS_ORIGINS_RAW.split(",") if origin.strip()]
    if _CORS_ORIGINS_RAW
    else ["https://script.google.com", "https://docs.google.com"]
)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up text editor service...")
    prompt_builder = PromptBuilder()

    await asyncio.to_thread(prompt_builder.startup_check)
    await asyncio.to_thread(
        run_startup_checks,
        ALLOWED_DOMAINS,
        ALLOWED_INTENTS,
        ALLOWED_OVERLAYS,
        Path("config"),
        Path("knowledge_base"),
    )
    await asyncio.to_thread(load_scoring_weights)

    logger.info("PromptBuilder initialized successfully")
    app.state.prompt_builder = prompt_builder
    yield
    logger.info("Shutting down text editor service...")


app = FastAPI(
    title="Text Editor API",
    description="API для редактирования текстов с помощью LLM",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000
    log_entry = {
        "timestamp": time.time(),
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "duration_ms": round(duration_ms, 2),
        "client_ip": request.client.host if request.client else None,
    }
    logger.info(json.dumps(log_entry, ensure_ascii=False))
    return response


def get_prompt_builder() -> PromptBuilder:
    prompt_builder = getattr(app.state, "prompt_builder", None)
    if prompt_builder is None:
        raise RuntimeError("PromptBuilder is not initialized")
    return prompt_builder


# ---------------------------------------------------------------------------
# Проверка провайдеров (глубокая и лёгкая)
# ---------------------------------------------------------------------------
async def _check_provider_deep(provider_name: str) -> bool:
    try:
        provider_enum = LLMProvider(provider_name)
        async with create_llm_client(
            provider=provider_enum,
            model=None,
            temperature=0.0,
            timeout=5.0,
            max_retries=1,
            max_tokens=1,
        ) as client:
            await client.generate("ping")
            return True
    except Exception as e:
        logger.debug(f"Deep check failed for {provider_name}: {e}")
        return False


async def _check_providers_availability(deep: bool = False) -> Tuple[bool, Dict[str, bool]]:
    """
    Возвращает (any_available, {provider: available}).
    При deep=False использует кэш (TTL 60 сек) и light‑check (наличие API‑ключа).
    При deep=True выполняет реальные запросы и обновляет кэш.
    """
    results: Dict[str, bool] = {}
    for provider in ALLOWED_PROVIDERS:
        if not deep:
            cached = _get_cached_availability(provider)
            if cached is not None:
                results[provider] = cached
                continue
            # light‑check – только по env
            env_var = _PROVIDER_KEY_ENV.get(provider.lower())
            available = bool(os.getenv(env_var)) if env_var else False
            _set_cached_availability(provider, available)
            results[provider] = available
        else:
            available = await _check_provider_deep(provider)
            _set_cached_availability(provider, available)
            results[provider] = available

    any_available = any(results.values())
    return any_available, results


# ---------------------------------------------------------------------------
# Эндпоинты
# ---------------------------------------------------------------------------
@app.get("/")
async def root() -> dict:
    return {"status": "ok", "version": "1.0.0"}


# ИСПРАВЛЕНИЕ: добавляем явное описание в декоратор, чтобы гарантировать наличие в OpenAPI
@app.get(
    "/health",
    response_model=HealthResponse,
    description="""
Проверка состояния сервиса.

- deep=false (по умолчанию): проверяет только наличие API-ключей в env.
- deep=true: выполняет реальный тестовый запрос к каждому LLM-провайдеру.
  ВНИМАНИЕ: deep=true потребляет реальные токены и может тарифицироваться.
  Использовать только для диагностики, не в автоматическом мониторинге.
""",
    dependencies=[Depends(verify_api_key)],
)
async def health_check(deep: bool = False) -> Response:
    builder = get_prompt_builder()
    any_available, provider_status = await _check_providers_availability(deep=deep)

    health = HealthResponse(
        status="ok" if any_available else "degraded",
        version="1.0.0",
        available_domains=sorted(ALLOWED_DOMAINS),
        available_intents=list(builder.get_available_intents()),
        available_overlays=list(builder.get_available_overlays()),
        available_providers=[p for p, ok in provider_status.items() if ok],
        provider_status=provider_status,
        deep_check=deep,
        contract_version=CONTRACT_VERSION,
    )

    status_code = status.HTTP_200_OK if any_available else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(
        content=health.model_dump(),
        status_code=status_code,
    )


def _log_edit_request_meta(request: EditRequest, retrieval_meta: Optional[Dict] = None) -> None:
    log_data = {
        "event": "edit_request",
        "domain": request.domain,
        "intent": request.intent,
        "overlays": request.overlays,
        "provider": request.provider,
        "output_mode": request.output_mode,
        "dry_run": request.dry_run,
        "text_length": len(request.text),
        "include_knowledge": request.include_knowledge,
        "include_few_shot": request.include_few_shot,
    }
    if retrieval_meta:
        log_data["retrieval_meta"] = retrieval_meta
    logger.info(json.dumps(log_data, ensure_ascii=False))


@app.post("/api/edit", response_model=EditResponse, dependencies=[Depends(verify_api_key)])
async def edit_text(request: EditRequest) -> EditResponse:
    # Валидация domain/intent/overlays теперь выполняется в Pydantic (A-1)
    # Ручные проверки УДАЛЕНЫ.
    try:
        audience: Optional[AudienceProfile] = None
        if request.audience is not None:
            audience = AudienceProfile(
                kind=request.audience.kind,
                expertise=request.audience.expertise,
                formality=request.audience.formality,
                description=request.audience.description,
            )

        prompt_builder = get_prompt_builder()

        if request.dry_run:
            prompt, retrieval_meta = prompt_builder.build(
                text=request.text,
                domain=request.domain,
                intent=request.intent,
                audience=audience,
                overlays=request.overlays,
                output_mode=request.output_mode,
                include_knowledge=request.include_knowledge,
                include_few_shot=request.include_few_shot,
                include_retrieval_meta=True,
            )
            _log_edit_request_meta(request, retrieval_meta)
            return EditResponse(
                edited_text=request.text,
                report=None,
                provider=request.provider,
                model=request.model,
                dry_run=True,
                usage={},
                raw_response={},
                retrieval_meta=retrieval_meta,
            )

        prompt, retrieval_meta = prompt_builder.build(
            text=request.text,
            domain=request.domain,
            intent=request.intent,
            audience=audience,
            overlays=request.overlays,
            output_mode=request.output_mode,
            include_knowledge=request.include_knowledge,
            include_few_shot=request.include_few_shot,
            include_retrieval_meta=True,
        )
        providers_to_try = [request.provider] + [
            p for p in sorted(ALLOWED_PROVIDERS) if p != request.provider
        ]
        response = await call_with_fallback(
            prompt=prompt,
            providers=providers_to_try,
            model=request.model,
            temperature=request.temperature,
            max_retries_per_provider=2,
        )

        edited_text = response.content
        report: Optional[str] = None
        if request.output_mode == "text_and_report":
            edited_text, report = _parse_text_and_report(response.content)

        _log_edit_request_meta(request, retrieval_meta)
        return EditResponse(
            edited_text=edited_text,
            report=report,
            model=response.model,
            provider=response.provider,
            dry_run=False,
            usage={"tokens_used": response.tokens_used},
            raw_response={
                "finish_reason": response.finish_reason,
            },
            retrieval_meta=retrieval_meta if request.include_retrieval_meta else None,
        )

    except LLMError as error:
        logger.error("LLM error: %s", error, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM service temporarily unavailable. Try again later.",
        ) from error
    except FileNotFoundError as error:
        logger.error("Config file not found: %s", error, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Service configuration error. Contact support.",
        ) from error
    except HTTPException:
        raise
    except ValidationError as error:
        logger.error("Validation error: %s", error, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=error.errors(),
        ) from error
    except Exception as error:
        logger.error("Unexpected error: %s", error, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error.",
        ) from error


# ---------------------------------------------------------------------------
# Парсинг ответа LLM с маркерами ТЕКСТ/ОТЧЁТ (A-4)
# ---------------------------------------------------------------------------
_MARKER_TEXT = re.compile(r"={2,}\s*ТЕКСТ\s*={2,}", re.IGNORECASE)
_MARKER_REPORT = re.compile(r"={2,}\s*ОТЧЁТ\s*={2,}", re.IGNORECASE)


def _parse_text_and_report(raw: str) -> Tuple[str, Optional[str]]:
    """
    Разбирает ответ LLM на отредактированный текст и отчёт.
    Устойчив к вариациям маркеров (пробелы, регистр).
    """
    parts_by_text = _MARKER_TEXT.split(raw, maxsplit=1)
    if len(parts_by_text) < 2:
        logger.warning(
            "Маркер ТЕКСТ не найден в ответе LLM. "
            "Возвращаем весь ответ как текст. Длина: %d символов", len(raw)
        )
        return raw.strip(), None

    after_text_marker = parts_by_text[1]
    parts_by_report = _MARKER_REPORT.split(after_text_marker, maxsplit=1)

    edited_text = parts_by_report[0].strip()
    report = parts_by_report[1].strip() if len(parts_by_report) > 1 else None

    if not edited_text:
        logger.warning("Блок ТЕКСТ найден, но содержимое пустое.")

    return edited_text, report