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

# slowapi
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.auth import verify_api_key
from src.config_types import AudienceProfile
from src.contracts import CONTRACT_VERSION, EditRequest, EditResponse, HealthResponse
from src.llm_client import LLMError, call_with_fallback, create_llm_client
from src.output_guard import (
    find_placeholder_leaks,
    harden_prompt_against_placeholders,
    has_placeholder_leak,
)
from src.prompt_builder import PromptBuilder
from src.provider_registry import LLMProvider
from src.semantic_index import init_semantic_index
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

_is_testing = os.getenv("PYTEST_RUNNING", "false").lower() == "true"
_rate_limit = "1000/minute" if _is_testing else "10/minute"


# ---------------------------------------------------------------------------
# Кэш доступности провайдеров
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
    _provider_cache.clear()


_PROVIDER_KEY_ENV: Dict[str, str] = {
    "perplexity": "PERPLEXITY_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

_CORS_ORIGINS_RAW = os.getenv("CORS_ALLOWED_ORIGINS", "")
_CORS_ORIGINS: list[str] = (
    [origin.strip() for origin in _CORS_ORIGINS_RAW.split(",") if origin.strip()]
    if _CORS_ORIGINS_RAW
    else ["https://script.google.com", "https://docs.google.com"]
)


# ---------------------------------------------------------------------------
# Вспомогательные функции для семантического индекса
# ---------------------------------------------------------------------------
def _collect_semantic_entries(app: FastAPI) -> list[dict]:
    """
    Собирает все записи из базы знаний, сохранённой в app.state.kb.
    Если kb отсутствует, возвращает пустой список.
    """
    kb = getattr(app.state, "kb", None)
    if kb is None:
        logger.warning("SemanticIndex: kb не загружен, пропускаем сбор записей")
        return []

    all_entries = []
    for attr in ("grammar_errors", "stylistic_issues", "logic_issues"):
        entries = getattr(kb, attr, [])
        if isinstance(entries, list):
            all_entries.extend(entries)
    return all_entries


async def _build_semantic_index_background(app: FastAPI) -> None:
    """Фоновая задача построения семантического индекса."""
    if app.state.semantic_index_status != "not_started":
        logger.info("SemanticIndex: уже запущен или завершён, пропускаем")
        return

    app.state.semantic_index_status = "building"
    logger.info("SemanticIndex: фоновое построение индекса начато")

    try:
        all_entries = _collect_semantic_entries(app)
        if not all_entries:
            logger.warning("SemanticIndex: нет записей для индексации, индекс не строится")
            app.state.semantic_index_status = "ready"
            return

        await asyncio.to_thread(init_semantic_index, all_entries)
        app.state.semantic_index_status = "ready"
        logger.info("SemanticIndex: фоновое построение индекса завершено успешно")
    except Exception as e:
        logger.error("SemanticIndex: ошибка при построении индекса: %s", e, exc_info=True)
        app.state.semantic_index_status = "failed"
        app.state.semantic_index_error = str(e)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up text editor service...")

    # SEC-05: Проверка обязательных переменных окружения
    _required_env = ["OPENROUTER_API_KEY"]
    _missing = [k for k in _required_env if not os.getenv(k)]
    if _missing:
        logger.critical("Missing required env variables: %s. Refusing to start.", _missing)
        raise RuntimeError(f"Missing required env variables: {_missing}")

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

    # Сохраняем kb в состояние приложения для использования в фоновой задаче
    try:
        app.state.kb = prompt_builder.kb
        logger.info("SemanticIndex: kb загружен")
    except AttributeError:
        logger.warning("PromptBuilder не содержит атрибут kb, семантический индекс не будет построен")
        app.state.kb = None

    # Инициализация состояния индекса
    app.state.semantic_index_status = "not_started"
    app.state.semantic_index_task = None
    app.state.semantic_index_error = None

    # Запускаем фоновую задачу (не блокирует старт)
    task = asyncio.create_task(_build_semantic_index_background(app))
    app.state.semantic_index_task = task

    yield

    logger.info("Shutting down text editor service...")
    if app.state.semantic_index_task:
        app.state.semantic_index_task.cancel()
        try:
            await app.state.semantic_index_task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="Text Editor API",
    description="API для редактирования текстов с помощью LLM",
    version="1.0.0",
    lifespan=lifespan,
)

# Настройка rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
    forwarded_for = request.headers.get("X-Forwarded-For")
    client_ip = forwarded_for.split(",")[0].strip() if forwarded_for else (
        request.client.host if request.client else None
    )
    log_entry = {
        "timestamp": time.time(),
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "duration_ms": round(duration_ms, 2),
        "client_ip": client_ip,
    }
    logger.info(json.dumps(log_entry, ensure_ascii=False))
    return response


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    return response


def get_prompt_builder() -> PromptBuilder:
    prompt_builder = getattr(app.state, "prompt_builder", None)
    if prompt_builder is None:
        raise RuntimeError("PromptBuilder is not initialized")
    return prompt_builder


# ---------------------------------------------------------------------------
# Проверка провайдеров
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
    results: Dict[str, bool] = {}
    for provider in ALLOWED_PROVIDERS:
        if not deep:
            cached = _get_cached_availability(provider)
            if cached is not None:
                results[provider] = cached
                continue
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
    return {"status": "ok"}


@app.get("/livez")
async def liveness_check() -> dict:
    return {"status": "alive"}


@app.get(
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


def _log_edit_request_meta(body: EditRequest, retrieval_meta: Optional[Dict] = None) -> None:
    log_data = {
        "event": "edit_request",
        "domain": body.domain,
        "intent": body.intent,
        "overlays": body.overlays,
        "provider": body.provider,
        "output_mode": body.output_mode,
        "dry_run": body.dry_run,
        "text_length": len(body.text),
        "include_knowledge": body.include_knowledge,
        "include_few_shot": body.include_few_shot,
    }
    if retrieval_meta:
        log_data["retrieval_meta"] = retrieval_meta
    logger.info(json.dumps(log_data, ensure_ascii=False))


class PlaceholderLeakError(Exception):
    """Служебные плейсхолдеры остались в ответе LLM после всех попыток.

    Поднимается, когда в финальном тексте по-прежнему присутствуют токены
    вроде ``[PERSON_NAME]``/``[ADDRESS]`` даже после повторной генерации.
    Обрабатывается как fail-closed: пользователю возвращается ошибка,
    а не текст со служебными маркерами.
    """

    def __init__(self, leaks: list[str]) -> None:
        self.leaks = leaks
        super().__init__(f"Placeholder tokens leaked into output: {leaks}")


def _split_edit_output(raw: str, output_mode: str) -> Tuple[str, Optional[str]]:
    """Разбирает сырой ответ LLM на текст и (опционально) отчёт.

    В режиме ``text_and_report`` парсит блоки, иначе возвращает весь
    ответ как текст.
    """
    if output_mode == "text_and_report":
        return _parse_text_and_report(raw)
    return raw, None


async def _generate_clean_edit(
    prompt: str,
    providers: list[str],
    body: EditRequest,
) -> Tuple[Any, str, Optional[str]]:
    """Генерирует отредактированный текст с защитой от плейсхолдеров.

    Сначала выполняется обычный вызов LLM. Если в финальном тексте (а в
    режиме отчёта — также в блоке отчёта) обнаружены служебные
    плейсхолдеры, выполняется одна повторная попытка с усиленным
    промптом. Если и после неё токены остаются, поднимается
    ``PlaceholderLeakError`` (поведение fail-closed) — пользователю такой
    текст не отдаётся.

    Args:
        prompt: собранный промпт для LLM.
        providers: список провайдеров в порядке приоритета.
        body: исходный запрос (нужны output_mode, model, temperature).

    Returns:
        Кортеж ``(response, edited_text, report)`` для успешного чистого
        результата.

    Raises:
        PlaceholderLeakError: если плейсхолдеры не удалось устранить.
        LLMError: если все провайдеры недоступны.
    """
    response = await call_with_fallback(
        prompt=prompt,
        providers=providers,
        model=body.model,
        temperature=body.temperature,
        max_retries_per_provider=2,
    )
    edited_text, report = _split_edit_output(response.content, body.output_mode)

    # Проверяем именно пользовательский текст и отчёт, а не сырой ответ:
    # легальные блоки-заголовки режима text_and_report детектор не трогает.
    if not (has_placeholder_leak(edited_text) or has_placeholder_leak(report or "")):
        return response, edited_text, report

    logger.warning(
        "Guard: обнаружены плейсхолдеры в ответе LLM, выполняем повторную попытку"
    )
    hardened_prompt = harden_prompt_against_placeholders(prompt)
    response = await call_with_fallback(
        prompt=hardened_prompt,
        providers=providers,
        model=body.model,
        # Понижаем температуру, чтобы модель точнее следовала инструкции.
        temperature=min(body.temperature, 0.2),
        max_retries_per_provider=2,
    )
    edited_text, report = _split_edit_output(response.content, body.output_mode)

    leaks = find_placeholder_leaks(edited_text) + find_placeholder_leaks(report or "")
    if leaks:
        # Не логируем содержимое текста (может содержать PII) — только токены.
        logger.error("Guard: плейсхолдеры остались после повторной попытки: %s", leaks)
        raise PlaceholderLeakError(sorted(set(leaks)))

    return response, edited_text, report


@app.post("/api/edit", response_model=EditResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit(_rate_limit)
async def edit_text(request: Request, body: EditRequest) -> EditResponse:
    try:
        audience: Optional[AudienceProfile] = None
        if body.audience is not None:
            audience = AudienceProfile(
                kind=body.audience.kind,
                expertise=body.audience.expertise,
                formality=body.audience.formality,
                description=body.audience.description,
            )

        prompt_builder = get_prompt_builder()

        if body.dry_run:
            prompt, retrieval_meta = prompt_builder.build(
                text=body.text,
                domain=body.domain,
                intent=body.intent,
                audience=audience,
                overlays=body.overlays,
                output_mode=body.output_mode,
                include_knowledge=body.include_knowledge,
                include_few_shot=body.include_few_shot,
                include_retrieval_meta=True,
            )
            _log_edit_request_meta(body, retrieval_meta)
            return EditResponse(
                edited_text=body.text,
                report=None,
                provider=body.provider,
                model=body.model,
                dry_run=True,
                usage={},
                raw_response={},
                retrieval_meta=retrieval_meta,
            )

        prompt, retrieval_meta = prompt_builder.build(
            text=body.text,
            domain=body.domain,
            intent=body.intent,
            audience=audience,
            overlays=body.overlays,
            output_mode=body.output_mode,
            include_knowledge=body.include_knowledge,
            include_few_shot=body.include_few_shot,
            include_retrieval_meta=True,
        )
        providers_to_try = [body.provider] + [
            p for p in sorted(ALLOWED_PROVIDERS) if p != body.provider
        ]

        response, edited_text, report = await _generate_clean_edit(
            prompt=prompt,
            providers=providers_to_try,
            body=body,
        )

        _log_edit_request_meta(body, retrieval_meta)
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
            retrieval_meta=retrieval_meta if body.include_retrieval_meta else None,
        )

    except PlaceholderLeakError as error:
        logger.error("Placeholder guard blocked response: %s", error.leaks)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "The editor could not produce a clean result. "
                "Please try again."
            ),
        ) from error
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
# Парсинг ответа LLM с маркерами ТЕКСТ/ОТЧЁТ
# ---------------------------------------------------------------------------
_MARKER_TEXT = re.compile(r"={2,}\s*ТЕКСТ\s*={2,}", re.IGNORECASE)
_MARKER_REPORT = re.compile(r"={2,}\s*ОТЧЁТ\s*={2,}", re.IGNORECASE)


def _parse_text_and_report(raw: str) -> Tuple[str, Optional[str]]:
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