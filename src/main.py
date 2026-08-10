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
from typing import Any, Dict, Optional, Tuple

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.auth import verify_api_key
from src.config_types import AudienceProfile
from src.contracts import CONTRACT_VERSION, EditRequest, EditResponse, HealthResponse
from src.llm_client import LLMError, call_with_fallback, create_llm_client, LLMFallbackError  # NEW: added LLMFallbackError
from src.output_guard import (
    find_placeholder_leaks,
    harden_prompt_against_placeholders,
    has_placeholder_leak,
)
from src.prompt_builder import PromptBuilder
from src.provider_registry import LLMProvider
from src.scoring_weights import load_scoring_weights
from src.semantic_index import init_semantic_index
from src.shared_contracts import (
    ALLOWED_DOMAINS,
    ALLOWED_INTENTS,
    ALLOWED_OVERLAYS,
    ALLOWED_PROVIDERS,
)
from src.startup_checks import run_startup_checks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_is_testing = os.getenv("PYTEST_RUNNING", "false").lower() == "true"
_rate_limit = "1000/minute" if _is_testing else "10/minute"


def _client_ip_key(request: Request) -> str:
    """Ключ rate-limit по реальному IP клиента за reverse-proxy.

    На Render (и любом прокси) ``request.client.host`` равен IP прокси и
    одинаков для всех клиентов — тогда лимит становится глобальным.
    Берём первый IP из ``X-Forwarded-For`` (ближайший к клиенту),
    а при его отсутствии — fallback на стандартный get_remote_address.

    Замечание по безопасности: заголовок X-Forwarded-For клиент может
    подделать, если запрос идёт не через доверенный прокси. На Render
    входящий трафик всегда проходит через их прокси, который
    перезаписывает этот заголовок, поэтому для текущего деплоя это приемлемо.
    """
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        first_ip = forwarded_for.split(",")[0].strip()
        if first_ip:
            return first_ip
    return get_remote_address(request)


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
    """Собирает все записи из базы знаний, сохранённой в app.state.kb."""
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
    except Exception as error:
        logger.error(
            "SemanticIndex: ошибка при построении индекса: %s",
            error,
            exc_info=True,
        )
        app.state.semantic_index_status = "failed"
        app.state.semantic_index_error = str(error)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up text editor service...")

    _required_env = ["OPENROUTER_API_KEY"]
    _missing = [key for key in _required_env if not os.getenv(key)]
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

    try:
        app.state.kb = prompt_builder.kb
        logger.info("SemanticIndex: kb загружен")
    except AttributeError:
        logger.warning(
            "PromptBuilder не содержит атрибут kb, семантический индекс не будет построен"
        )
        app.state.kb = None

    app.state.semantic_index_status = "not_started"
    app.state.semantic_index_task = None
    app.state.semantic_index_error = None

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

limiter = Limiter(key_func=_client_ip_key)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000
    forwarded_for = request.headers.get("X-Forwarded-For")
    client_ip = (
        forwarded_for.split(",")[0].strip()
        if forwarded_for
        else (request.client.host if request.client else None)
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
    except Exception as error:
        logger.debug("Deep check failed for %s: %s", provider_name, error)
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
        available_providers=[provider for provider, ok in provider_status.items() if ok],
        provider_status=provider_status,
        deep_check=deep,
        contract_version=CONTRACT_VERSION,
    )

    status_code = (
        status.HTTP_200_OK
        if any_available
        else status.HTTP_503_SERVICE_UNAVAILABLE
    )
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


class InvalidLLMOutputError(Exception):
    """LLM вернула ответ, не соответствующий ожидаемому формату."""

    def __init__(self, reasons: list[str]) -> None:
        self.reasons = reasons
        super().__init__(f"Invalid LLM output: {reasons}")


def _split_edit_output(raw: str, output_mode: str) -> Tuple[str, Optional[str]]:
    """Разбирает сырой ответ LLM на текст и (опционально) отчёт."""
    if output_mode == "text_and_report":
        return _parse_text_and_report(raw)
    return raw, None


def _looks_like_report_instead_of_text(text: str) -> bool:
    """Эвристика: ответ похож на анализ/отчёт, а не на отредактированный текст."""
    normalized = text.strip().lower()
    if not normalized:
        return True

    report_signals = [
        'count the "не x, а y" occurrences',
        "count the",
        "so we have",
        "this is a marker",
        "also, the use of",
        "маркеры:",
        "исходный ип:",
        "итоговый ип:",
        "нужен второй проход",
    ]
    return any(signal in normalized for signal in report_signals)


def _validate_edit_output(
    *,
    raw_content: str,
    edited_text: str,
    report: Optional[str],
    output_mode: str,
) -> list[str]:
    """Возвращает список причин, по которым ответ LLM невалиден."""
    reasons: list[str] = []

    if has_placeholder_leak(edited_text):
        reasons.extend(find_placeholder_leaks(edited_text))

    if output_mode == "text_and_report":
        has_text_marker = _MARKER_TEXT.search(raw_content) is not None

        if has_text_marker and not edited_text.strip():
            reasons.append("EMPTY_TEXT_BLOCK")

        if not has_text_marker and _looks_like_report_instead_of_text(edited_text):
            reasons.append("REPORT_INSTEAD_OF_TEXT")

        if has_placeholder_leak(report or ""):
            reasons.extend(find_placeholder_leaks(report or ""))

    return sorted(set(reasons))


async def _generate_clean_edit(
    prompt: str,
    providers: list[str],
    body: EditRequest,
) -> Tuple[Any, str, Optional[str]]:
    """Генерирует отредактированный текст с защитой от плейсхолдеров и сломанного формата."""
    response = await call_with_fallback(
        prompt=prompt,
        providers=providers,
        model=body.model,
        temperature=body.temperature,
        max_retries_per_provider=2,
    )
    edited_text, report = _split_edit_output(response.content, body.output_mode)

    reasons = _validate_edit_output(
        raw_content=response.content,
        edited_text=edited_text,
        report=report,
        output_mode=body.output_mode,
    )
    if not reasons:
        return response, edited_text, report

    logger.warning(
        "Guard: ответ LLM невалиден, выполняем повторную попытку. Причины: %s",
        reasons,
    )

    hardened_prompt = harden_prompt_against_placeholders(prompt)
    hardened_prompt += (
        "\n\nКритично: строго соблюдай формат ответа. "
        "Если запрошен режим text_and_report, сначала выведи блок "
        "===ТЕКСТ=== с полным отредактированным текстом, затем блок "
        "===ОТЧЁТ===. Не выводи один только анализ, список маркеров или "
        "служебные пояснения вместо текста."
    )

    response = await call_with_fallback(
        prompt=hardened_prompt,
        providers=providers,
        model=body.model,
        temperature=min(body.temperature, 0.2),
        max_retries_per_provider=2,
    )
    edited_text, report = _split_edit_output(response.content, body.output_mode)

    reasons = _validate_edit_output(
        raw_content=response.content,
        edited_text=edited_text,
        report=report,
        output_mode=body.output_mode,
    )
    if reasons:
        logger.error(
            "Guard: ответ LLM остался невалидным после повторной попытки: %s",
            reasons,
        )
        raise InvalidLLMOutputError(reasons)

    return response, edited_text, report


# NEW: функция преобразования LLMError в HTTPException
def _llm_error_to_http_exception(error: LLMError) -> HTTPException:
    """Преобразует LLMError в HTTPException с безопасным сообщением."""
    if isinstance(error, LLMFallbackError):
        kind = error.kind
        if kind == "rate_limit":
            return HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="LLM provider rate limit reached. Please try again later.",
            )
        if kind == "context_limit":
            return HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="The text or editing instructions are too large. Please shorten them.",
            )
        if kind in ("timeout", "upstream_error"):
            return HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LLM service is temporarily unavailable. Please try again later.",
            )
        if kind in ("authentication", "configuration"):
            return HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LLM service configuration is temporarily unavailable.",
            )
        # unknown
        return HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM service returned an invalid response. Please try again later.",
        )
    # Обычный LLMError (не LLMFallbackError)
    return HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="LLM service returned an invalid response. Please try again later.",
    )


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
            provider for provider in sorted(ALLOWED_PROVIDERS)
            if provider != body.provider
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

    except InvalidLLMOutputError as error:
        logger.error("Output guard blocked response: %s", error.reasons)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "The editor could not produce a valid formatted result. "
                "Please try again."
            ),
        ) from error
    except LLMError as error:
        # CHANGED: логируем и преобразуем через новую функцию
        if isinstance(error, LLMFallbackError):
            logger.warning(
                "LLMFallbackError: provider=%s kind=%s upstream_status=%s skipped=%s unknown=%s prompt_length=%d",
                error.provider,
                error.kind,
                error.upstream_status,
                error.skipped_providers,
                error.unknown_providers,
                error.prompt_length,
            )
        else:
            logger.error("LLM error: %s", error, exc_info=True)
        raise _llm_error_to_http_exception(error) from error
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
    """Разбирает ответ LLM на отредактированный текст и отчёт.

    Поддерживает оба порядка маркеров:
    - штатный ``===ТЕКСТ=== ... ===ОТЧЁТ=== ...``;
    - перевёрнутый ``===ОТЧЁТ=== ... ===ТЕКСТ=== ...``.

    Если маркер ТЕКСТ отсутствует, сохраняем обратную совместимость:
    весь ответ возвращается как текст.
    """
    text_match = _MARKER_TEXT.search(raw)
    report_match = _MARKER_REPORT.search(raw)

    if text_match is None:
        logger.warning(
            "Маркер ТЕКСТ не найден в ответе LLM. "
            "Возвращаем весь ответ как текст. Длина: %d символов",
            len(raw),
        )
        return raw.strip(), None

    if report_match is None:
        edited_text = raw[text_match.end():].strip()
        if not edited_text:
            logger.warning("Блок ТЕКСТ найден, но содержимое пустое.")
        return edited_text, None

    if text_match.start() < report_match.start():
        edited_text = raw[text_match.end():report_match.start()].strip()
        report = raw[report_match.end():].strip()
    else:
        logger.warning(
            "Маркеры ТЕКСТ/ОТЧЁТ идут в перевёрнутом порядке — "
            "разбираем с учётом этого, отчёт сохраняется."
        )
        edited_text = raw[text_match.end():].strip()
        report = raw[report_match.end():text_match.start()].strip()

    if not edited_text:
        logger.warning("Блок ТЕКСТ найден, но содержимое пустое.")

    return edited_text, (report or None)
