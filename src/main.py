from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from src.config_types import AudienceProfile
from src.contracts import CONTRACT_VERSION, EditRequest, EditResponse, HealthResponse
from src.llm_client import LLMError, LLMProvider, LLMResponse, call_with_fallback, create_llm_client
from src.prompt_builder import PromptBuilder
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


_provider_cache: Dict[str, Any] = {}
_PROVIDER_CACHE_TTL = 60


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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


def _supported_providers() -> Set[str]:
    return set(ALLOWED_PROVIDERS)


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
    global _provider_cache
    now = time.time()
    if deep and _provider_cache.get("_timestamp", 0) + _PROVIDER_CACHE_TTL > now:
        cached = _provider_cache.get("_data")
        if cached:
            return cached
    availability: Dict[str, bool] = {}
    for provider_name in ALLOWED_PROVIDERS:
        if deep:
            available = await _check_provider_deep(provider_name)
        else:
            try:
                provider_enum = LLMProvider(provider_name)
                async with create_llm_client(
                    provider=provider_enum,
                    model=None,
                    temperature=0.0,
                    timeout=5.0,
                    max_retries=1,
                ) as client:
                    available = True
            except Exception as e:
                logger.debug(f"Light check failed for {provider_name}: {e}")
                available = False
        availability[provider_name] = available
    any_available = any(availability.values())
    if deep:
        _provider_cache = {
            "_timestamp": now,
            "_data": (any_available, availability),
        }
    return any_available, availability


@app.get("/")
async def root() -> dict:
    return {"status": "ok", "version": "1.0.0"}


# PR-3 (НП-3): подключён response_model=HealthResponse; ответ строится через
# HealthResponse.model_dump() + JSONResponse, чтобы сохранить 503 при degraded.
# available_domains добавлен в ответ (был в модели, но не возвращался).
@app.get("/health", response_model=HealthResponse)
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


async def _call_llm_with_provider(
    prompt: str,
    provider_name: str,
    model: Optional[str],
    temperature: float,
) -> LLMResponse:
    provider_enum = LLMProvider(provider_name)
    async with create_llm_client(
        provider=provider_enum,
        model=model,
        temperature=temperature,
    ) as client:
        return await client.generate(prompt)


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


# PR-2 (НП-2): подключён response_model=EditResponse; обе ветки возвращают
# EditResponse(...) вместо dict — Pydantic валидирует ответ на выходе.
@app.post("/api/edit", response_model=EditResponse)
async def edit_text(request: EditRequest) -> EditResponse:
    if request.domain not in ALLOWED_DOMAINS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid domain: {request.domain}. Allowed: {sorted(ALLOWED_DOMAINS)}",
        )
    if request.intent is not None and request.intent not in ALLOWED_INTENTS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid intent: {request.intent}. Allowed: {sorted(ALLOWED_INTENTS)}",
        )
    for overlay in request.overlays:
        if overlay not in ALLOWED_OVERLAYS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid overlay: {overlay}. Allowed: {sorted(ALLOWED_OVERLAYS)}",
            )

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
                prompt=prompt,
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
            prompt=prompt,
            model=response.model,
            provider=response.provider,
            dry_run=False,
            usage={"tokens_used": response.tokens_used},
            raw_response={
                "finish_reason": response.finish_reason,
                "content": response.content,
            },
            retrieval_meta=retrieval_meta if request.include_retrieval_meta else None,
        )

    except LLMError as error:
        logger.error("LLM error: %s", error, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM generation failed: {error}",
        ) from error
    except FileNotFoundError as error:
        logger.error("Config file not found: %s", error, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Configuration error: {error}",
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
            detail=f"Internal server error: {error}",
        ) from error


_TEXT_MARKERS: Tuple[str, ...] = (
    "===ТЕКСТ===",
    "=== ТЕКСТ ===",
    "## ТЕКСТ",
    "**ТЕКСТ**",
    "ТЕКСТ:",
)

_REPORT_MARKERS: Tuple[str, ...] = (
    "===ОТЧЁТ===",
    "=== ОТЧЁТ ===",
    "===ОТЧЕТ===",
    "=== ОТЧЕТ ===",
    "## ОТЧЁТ",
    "## ОТЧЕТ",
    "**ОТЧЁТ**",
    "**ОТЧЕТ**",
    "ОТЧЁТ:",
    "ОТЧЕТ:",
)


def _find_marker(content_upper: str, markers: Tuple[str, ...]) -> Optional[str]:
    for marker in markers:
        if marker.upper() in content_upper:
            return marker
    return None


def _parse_text_and_report(content: str) -> Tuple[str, Optional[str]]:
    content_upper = content.upper()
    text_marker = _find_marker(content_upper, _TEXT_MARKERS)
    report_marker = _find_marker(content_upper, _REPORT_MARKERS)

    if text_marker and report_marker:
        text_pat = re.compile(re.escape(text_marker), re.IGNORECASE)
        report_pat = re.compile(re.escape(report_marker), re.IGNORECASE)
        text_match = text_pat.search(content)
        report_match = report_pat.search(content)
        if text_match and report_match:
            edited_text = content[text_match.end():report_match.start()].strip()
            report = content[report_match.end():].strip() or None
            return edited_text, report

    if text_marker and not report_marker:
        text_pat = re.compile(re.escape(text_marker), re.IGNORECASE)
        text_match = text_pat.search(content)
        if text_match:
            return content[text_match.end():].strip(), None

    logger.warning("parse_text_and_report: no markers found, returning whole response as text")
    return content.strip(), None
