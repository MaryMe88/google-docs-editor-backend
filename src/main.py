import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from src.error_mapping import (  # noqa: F401
    InvalidLLMOutputError,
    _llm_error_to_http_exception,
)
from src.middleware.body_size import BodySizeLimitMiddleware
from src.prompt_builder import PromptBuilder
from src.rate_limit import (
    _client_ip_key,  # noqa: F401
    limiter,
)
from src.routers.edit import router as edit_router
from src.routers.health import router as health_router
from src.scoring_weights import load_scoring_weights
from src.semantic_index import set_semantic_entries
from src.services.edit_service import _parse_text_and_report  # noqa: F401
from src.services.provider_health import (  # noqa: F401
    _PROVIDER_KEY_ENV,
    _check_providers_availability,
    invalidate_provider_cache,
)
from src.shared_contracts import (
    ALLOWED_DOMAINS,
    ALLOWED_INTENTS,
    ALLOWED_OVERLAYS,
)
from src.startup_checks import StartupCheckParams, run_startup_checks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# _client_ip_key moved to src.rate_limit (iteration 8, step 3).
# Re-export is in the top import block.


# ---------------------------------------------------------------------------
# Кэш доступности провайдеров
# ---------------------------------------------------------------------------
# Provider cache and _PROVIDER_KEY_ENV moved to
# src.services.provider_health (iteration 8, step 1a).
# Re-exports are declared in the top import block below.

# ---------------------------------------------------------------------------
# SEC-патч 2.1: Строгий allowlist для CORS
# ---------------------------------------------------------------------------
_ALLOWED_GOOGLE_ORIGINS: frozenset[str] = frozenset(
    {
        "https://script.google.com",
        "https://docs.google.com",
    }
)
_extra_origins = {o.strip() for o in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",") if o.strip()}
_unexpected = _extra_origins - _ALLOWED_GOOGLE_ORIGINS
if _unexpected:
    logger.warning("Игнорирую неожиданные CORS origins из ENV: %s", _unexpected)
_CORS_ORIGINS: list[str] = sorted(_ALLOWED_GOOGLE_ORIGINS)


# ---------------------------------------------------------------------------
# Вспомогательные функции для семантического индекса
# ---------------------------------------------------------------------------
def _collect_semantic_entries(app: FastAPI) -> list[dict[str, Any]]:
    """Собирает индексируемые записи из заранее загруженной полной KB."""
    prompt_builder = getattr(app.state, "prompt_builder", None)
    if prompt_builder is None:
        logger.warning("SemanticIndex: PromptBuilder не инициализирован")
        return []

    knowledge_base = getattr(prompt_builder, "_loaded_kb", None)
    if knowledge_base is None:
        logger.warning("SemanticIndex: KB не загружена в PromptBuilder")
        return []

    all_entries: list[dict[str, Any]] = []
    for attribute_name in (
        "grammar_errors",
        "stylistic_issues",
        "logic_issues",
    ):
        entries = getattr(knowledge_base, attribute_name, [])
        if isinstance(entries, list):
            all_entries.extend(entry for entry in entries if isinstance(entry, dict))

    return all_entries


# ---------------------------------------------------------------------------
# Lifespan (фоновое построение индекса УДАЛЕНО)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up text editor service...")

    _required_env = ["OPENROUTER_API_KEY"]
    _missing = [key for key in _required_env if not os.getenv(key)]
    if _missing:
        logger.critical("Missing required env variables: %s. Refusing to start.", _missing)
        raise RuntimeError(f"Missing required env variables: {_missing}")

    is_testing_now = os.getenv("PYTEST_RUNNING", "false").lower() == "true"
    is_production = not (os.getenv("ENV", "").lower() == "development" or is_testing_now)
    if is_production and not os.getenv("API_SECRET_KEY"):
        logger.critical("API_SECRET_KEY is required in production mode. Refusing to start.")
        raise RuntimeError("API_SECRET_KEY is required in production mode.")

    prompt_builder = PromptBuilder()

    await asyncio.to_thread(prompt_builder.startup_check)

    startup_params = StartupCheckParams(
        allowed_domains=ALLOWED_DOMAINS,
        allowed_intents=ALLOWED_INTENTS,
        allowed_overlays=ALLOWED_OVERLAYS,
        config_path=Path("config"),
        kb_path=Path("knowledge_base"),
    )
    await asyncio.to_thread(run_startup_checks, startup_params)
    await asyncio.to_thread(load_scoring_weights)

    logger.info("PromptBuilder initialized successfully")
    app.state.prompt_builder = prompt_builder

    try:
        prompt_builder.load_full_kb()
        logger.info("Полная KB загружена для SemanticIndex")
        all_entries = _collect_semantic_entries(app)
        set_semantic_entries(all_entries)
        logger.info(
            "SemanticIndex: записи сохранены, индекс будет построен при "
            "первом запросе с deep_semantic_search=True"
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        logger.error(
            "Не удалось загрузить полную KB для SemanticIndex: %s",
            error,
            exc_info=True,
        )
        raise RuntimeError("Failed to load knowledge base required for SemanticIndex.") from error

    yield

    logger.info("Shutting down text editor service...")


app = FastAPI(
    title="Text Editor API",
    description="API для редактирования текстов с помощью LLM",
    version="1.0.0",
    lifespan=lifespan,
)

# limiter imported from src.rate_limit (iteration 8, step 3).
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)

app.add_middleware(BodySizeLimitMiddleware)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000
    client_ip = request.client.host if request.client else None
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
    response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
    return response


app.include_router(health_router)
app.include_router(edit_router)


# get_prompt_builder moved to src.routers.edit
# (iteration 8, step 3).


# ---------------------------------------------------------------------------
# Проверка провайдеров
# ---------------------------------------------------------------------------
# _check_provider_deep and _check_providers_availability moved to
# src.services.provider_health (iteration 8, step 1a).
# Re-exports are declared in the top import block below.


# ---------------------------------------------------------------------------
# Эндпоинты
# ---------------------------------------------------------------------------
# Health endpoints (/, /livez, /health) moved to src.routers.health
# (iteration 8, step 1b).


# Edit helpers (_log_edit_request_meta, _split_edit_output,
# _looks_like_report_instead_of_text, _validate_edit_output,
# _generate_clean_edit, _build_audience_from_request,
# _build_dry_run_response) moved to src.services.edit_service
# (iteration 8, step 2). Re-exports are in the top import block.


# /api/edit endpoint moved to src.routers.edit (iteration 8, step 3).
# Re-export of _parse_text_and_report — in the top import block.
# Router is registered via app.include_router(edit_router).
