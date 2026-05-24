from __future__ import annotations

import logging
import re
from contextlib import asynccontextmanager
from typing import Optional, Set, Tuple

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.config_types import AudienceProfile
from src.contracts import EditRequest
from src.llm_client import LLMError, LLMProvider, create_llm_client
from src.prompt_builder import PromptBuilder
from src.shared_contracts import ALLOWED_PROVIDERS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up text editor service...")
    prompt_builder = PromptBuilder()

    try:
        prompt_builder.startup_check()
        logger.info("PromptBuilder initialized successfully")
    except Exception as error:
        logger.warning("PromptBuilder startup check failed: %s", error)

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


def get_prompt_builder() -> PromptBuilder:
    prompt_builder = getattr(app.state, "prompt_builder", None)
    if prompt_builder is None:
        raise RuntimeError("PromptBuilder is not initialized")
    return prompt_builder


def _supported_providers() -> Set[str]:
    return set(ALLOWED_PROVIDERS)


@app.get("/")
async def root() -> dict:
    return {"status": "ok", "version": "1.0.0"}


@app.get("/health")
async def health_check() -> dict:
    builder = get_prompt_builder()
    return {
        "status": "ok",
        "version": "1.0.0",
        "available_providers": sorted(_supported_providers()),
        "available_intents": builder.get_available_intents(),
        "available_overlays": builder.get_available_overlays(),
    }


@app.post("/api/edit")
async def edit_text(request: EditRequest) -> dict:
    logger.info(
        "Received edit request",
        extra={
            "text_length": len(request.text),
            "domain": request.domain,
            "intent": request.intent,
            "output_mode": request.output_mode,
            "provider": request.provider,
        },
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
        prompt = prompt_builder.build(
            text=request.text,
            domain=request.domain,
            intent=request.intent,
            audience=audience,
            overlays=request.overlays,
            output_mode=request.output_mode,
            include_knowledge=request.include_knowledge,
        )

        if request.dry_run:
            return {
                "edited_text": request.text,
                "report": None,
                "prompt": prompt,
                "provider": request.provider,
                "model": request.model,
                "dry_run": True,
                "usage": {},
                "raw_response": {},
            }

        provider_enum = LLMProvider(request.provider)
        async with create_llm_client(
            provider=provider_enum,
            model=request.model,
            temperature=request.temperature,
        ) as client:
            response = await client.generate(prompt)

        edited_text = response.content
        report: Optional[str] = None

        if request.output_mode == "text_and_report":
            edited_text, report = _parse_text_and_report(response.content)

        return {
            "edited_text": edited_text,
            "report": report,
            "prompt": prompt,
            "model": response.model,
            "provider": response.provider,
            "dry_run": False,
            "usage": {"tokens_used": response.tokens_used},
            "raw_response": {
                "finish_reason": response.finish_reason,
                "content": response.content,
            },
        }

    except LLMError as error:
        logger.error("LLM error: %s", error, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
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

    logger.warning(
        "parse_text_and_report: no markers found, returning whole response as text"
    )
    return content.strip(), None