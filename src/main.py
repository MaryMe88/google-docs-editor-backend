"""
main.py

FastAPI сервер для редактора текстов.
Принимает запросы из Google Docs, обрабатывает через LLM и возвращает результат.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Set

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from src.llm_client import LLMError, LLMProvider, create_llm_client
from src.prompt_builder import AudienceProfile, PromptBuilder
from src.tag_registry import normalize_tag

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

APP_VERSION = "1.0.0"


def _allowed_intents() -> Set[str]:
    return {"analytical", "storytelling", "marketingpush"}


def _allowed_overlays() -> Set[str]:
    return {"infostyle", "factcheck", "recommendations", "finalcheck"}


def _allowed_providers() -> Set[str]:
    return {"perplexity", "openai", "openrouter"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Управление жизненным циклом приложения.
    Инициализация и очистка ресурсов.
    """
    logger.info("Starting up text editor service...")

    try:
        app.state.prompt_builder = PromptBuilder()
        logger.info("PromptBuilder initialized successfully")
    except Exception as error:  # noqa: BLE001
        logger.error("Failed to initialize PromptBuilder: %s", error)
        raise

    yield

    logger.info("Shutting down text editor service...")


app = FastAPI(
    title="Text Editor API",
    description="API для редактирования текстов с помощью LLM",
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AudienceRequest(BaseModel):
    """Модель профиля аудитории в запросе."""

    kind: str = Field(default="b2b", description="Тип аудитории: b2b, b2c, mixed")
    expertise: str = Field(default="pro", description="Уровень: novice, pro, expert")
    formality: str = Field(default="neutral", description="Формальность: casual, neutral, formal")
    description: str = Field(default="", description="Дополнительное описание")

    @validator("kind")
    def validate_kind(cls, value: str) -> str:
        allowed = {"b2b", "b2c", "mixed", "custom"}
        normalized = value.strip().lower()
        if normalized not in allowed:
            raise ValueError(f"kind must be one of {sorted(allowed)}")
        return normalized

    @validator("expertise")
    def validate_expertise(cls, value: str) -> str:
        allowed = {"novice", "pro", "expert"}
        normalized = value.strip().lower()
        if normalized not in allowed:
            raise ValueError(f"expertise must be one of {sorted(allowed)}")
        return normalized

    @validator("formality")
    def validate_formality(cls, value: str) -> str:
        allowed = {"casual", "neutral", "formal"}
        normalized = value.strip().lower()
        if normalized not in allowed:
            raise ValueError(f"formality must be one of {sorted(allowed)}")
        return normalized


class EditRequest(BaseModel):
    """Модель запроса на редактирование текста."""

    text: str = Field(..., min_length=1, description="Текст для редактирования")
    domain: str = Field(
        default="marketing",
        description="Домен текста: marketing, blog, fiction, basic_edit, logic_edit",
    )
    intent: Optional[str] = Field(
        default=None,
        description=(
            "Цель обработки: analytical (точнее+логичнее), "
            "storytelling (увлекательнее), marketingpush (продающим)"
        ),
    )
    audience: Optional[AudienceRequest] = Field(
        default=None,
        description="Профиль аудитории",
    )
    overlays: List[str] = Field(
        default_factory=list,
        description=(
            "Дополнительные режимы: infostyle, factcheck, "
            "recommendations, finalcheck"
        ),
    )
    output_mode: str = Field(
        default="text_only",
        description="Формат вывода: text_only, text_and_report",
    )
    provider: str = Field(
        default="openrouter",
        description="LLM провайдер: openrouter, perplexity, openai",
    )
    model: Optional[str] = Field(
        default=None,
        description="Модель LLM (например, openrouter/auto или deepseek/deepseek-chat)",
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Температура генерации",
    )

    @validator("domain")
    def validate_domain(cls, value: str) -> str:
        return value.strip().lower()

    @validator("intent")
    def validate_intent(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None

        normalized = normalize_tag(value)
        allowed = _allowed_intents()
        if normalized not in allowed:
            raise ValueError(
                f"intent '{value}' not allowed. Allowed: {sorted(allowed)}"
            )
        return normalized

    @validator("overlays")
    def validate_overlays(cls, value: List[str]) -> List[str]:
        allowed = _allowed_overlays()
        normalized_overlays = [normalize_tag(item) for item in value]

        for overlay in normalized_overlays:
            if overlay not in allowed:
                raise ValueError(
                    f"overlay '{overlay}' not allowed. Allowed: {sorted(allowed)}"
                )

        return normalized_overlays

    @validator("output_mode")
    def validate_output_mode(cls, value: str) -> str:
        allowed = {"text_only", "text_and_report"}
        normalized = value.strip().lower()
        if normalized not in allowed:
            raise ValueError(f"output_mode must be one of {sorted(allowed)}")
        return normalized

    @validator("provider")
    def validate_provider(cls, value: str) -> str:
        normalized = value.strip().lower()
        allowed = _allowed_providers()
        if normalized not in allowed:
            raise ValueError(f"provider must be one of {sorted(allowed)}")
        return normalized

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Наш сервис является самым лучшим на рынке.",
                "domain": "marketing",
                "intent": "marketing_push",
                "audience": {
                    "kind": "b2b",
                    "expertise": "pro",
                    "formality": "neutral",
                    "description": "Менеджеры отделов продаж",
                },
                "overlays": ["info_style", "final_check"],
                "output_mode": "text_only",
                "provider": "openrouter",
                "model": "openrouter/auto",
            },
        }


class EditResponse(BaseModel):
    """Модель ответа с отредактированным текстом."""

    edited_text: str = Field(..., description="Отредактированный текст")
    report: Optional[str] = Field(
        default=None,
        description="Отчёт о правках (если запрошен)",
    )
    model: str = Field(..., description="Использованная модель")
    provider: str = Field(..., description="Использованный провайдер")
    tokens_used: Optional[int] = Field(
        default=None,
        description="Количество использованных токенов",
    )


class HealthResponse(BaseModel):
    """Модель ответа health check."""

    status: str
    version: str


class ErrorResponse(BaseModel):
    """Модель ответа с ошибкой."""

    error: str
    detail: Optional[str] = None


@app.get("/", response_model=HealthResponse)
async def root() -> HealthResponse:
    """Корневой эндпоинт."""
    return HealthResponse(status="ok", version=APP_VERSION)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check эндпоинт."""
    return HealthResponse(status="ok", version=APP_VERSION)


@app.post(
    "/api/edit",
    response_model=EditResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Некорректный запрос"},
        500: {"model": ErrorResponse, "description": "Внутренняя ошибка сервера"},
    },
)
async def edit_text(request: EditRequest) -> EditResponse:
    """
    Редактирует текст с помощью LLM.
    """
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
        if request.audience:
            audience = AudienceProfile(
                kind=request.audience.kind,
                expertise=request.audience.expertise,
                formality=request.audience.formality,
                description=request.audience.description,
            )

        prompt_builder: PromptBuilder = app.state.prompt_builder
        prompt = prompt_builder.build(
            text=request.text,
            domain=request.domain,
            intent=request.intent,
            audience=audience,
            overlays=request.overlays,
            output_mode=request.output_mode,
            include_knowledge=True,
        )

        logger.info(
            "Prompt built successfully",
            extra={"prompt_length": len(prompt)},
        )

        provider_enum = LLMProvider(request.provider)

        async with create_llm_client(
            provider=provider_enum,
            model=request.model,
            temperature=request.temperature,
        ) as client:
            response = await client.generate(prompt)

        logger.info(
            "LLM generation successful",
            extra={
                "response_length": len(response.content),
                "tokens_used": response.tokens_used,
            },
        )

        edited_text = response.content
        report: Optional[str] = None

        if request.output_mode == "text_and_report":
            edited_text, report = _parse_text_and_report(response.content)

        return EditResponse(
            edited_text=edited_text,
            report=report,
            model=response.model,
            provider=response.provider,
            tokens_used=response.tokens_used,
        )

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

    except Exception as error:  # noqa: BLE001
        logger.error("Unexpected error: %s", error, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {error}",
        ) from error


@app.post("/api/quick-edit")
async def quick_edit(text: str, audience_type: str = "b2b") -> dict:
    """
    Упрощённый эндпоинт для быстрого редактирования.
    Для обратной совместимости с MVP.
    """
    request = EditRequest(
        text=text,
        domain="marketing",
        audience=AudienceRequest(kind=audience_type),
        output_mode="text_only",
    )

    response = await edit_text(request)
    return {"edited_text": response.edited_text}


def _parse_text_and_report(content: str) -> tuple[str, Optional[str]]:
    """
    Парсит ответ в режиме text_and_report.

    Ожидается формат:
    ===ТЕКСТ===
    [текст]

    ===ОТЧЁТ===
    [отчёт]
    """
    text_marker = "===ТЕКСТ==="
    report_marker = "===ОТЧЁТ==="

    if text_marker in content and report_marker in content:
        parts = content.split(report_marker, maxsplit=1)
        text_part = parts[0].replace(text_marker, "").strip()
        report_part = parts[1].strip() if len(parts) > 1 else None
        return text_part, report_part

    return content.strip(), None


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
