"""
main.py

FastAPI сервер для редактора текстов.
Принимает запросы, собирает промпт через PromptBuilder,
отправляет его в LLM и возвращает результат.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, List, Optional, Set, Tuple

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


def _call_builder_method(builder: PromptBuilder, *names: str) -> Any:
    """
    Вызывает первый существующий метод у PromptBuilder.
    Нужен для мягкой совместимости с legacy-именами.
    """
    for name in names:
        method = getattr(builder, name, None)
        if callable(method):
            return method()
    raise AttributeError(
        f"PromptBuilder does not have any of methods: {', '.join(names)}"
    )


def _supported_providers() -> Set[str]:
    """
    Возвращает список провайдеров, реально поддерживаемых llm_client.py.
    """
    return {
        LLMProvider.PERPLEXITY.value,
        LLMProvider.OPENAI.value,
        LLMProvider.OPENROUTER.value,
        LLMProvider.ANTHROPIC.value,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Инициализация и завершение ресурсов приложения.
    """
    logger.info("Starting up text editor service...")

    try:
        prompt_builder = PromptBuilder()

        # Принудительно прогреваем основные зависимости,
        # чтобы ошибки конфигов упали на старте, а не в первом запросе.
        _call_builder_method(prompt_builder, "get_core_config", "getcoreconfig")
        _call_builder_method(prompt_builder, "get_knowledge_base", "getknowledgebase")

        app.state.prompt_builder = prompt_builder
        logger.info("PromptBuilder initialized successfully")
    except Exception as error:  # noqa: BLE001
        logger.error("Failed to initialize PromptBuilder: %s", error)
        raise

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
    """
    Возвращает инициализированный PromptBuilder из состояния приложения.
    """
    prompt_builder = getattr(app.state, "prompt_builder", None)
    if prompt_builder is None:
        raise RuntimeError("PromptBuilder is not initialized")
    return prompt_builder


def get_available_intents() -> Set[str]:
    """
    Возвращает доступные intents из PromptBuilder в нормализованном виде.
    """
    builder = get_prompt_builder()
    values = _call_builder_method(
        builder,
        "get_available_intents",
        "getavailableintents",
    )
    return {normalize_tag(item) for item in values}


def get_available_overlays() -> Set[str]:
    """
    Возвращает доступные overlays из PromptBuilder в нормализованном виде.
    """
    builder = get_prompt_builder()
    values = _call_builder_method(
        builder,
        "get_available_overlays",
        "getavailableoverlays",
    )
    return {normalize_tag(item) for item in values}


class AudienceRequest(BaseModel):
    """
    Профиль аудитории, приходящий в API-запросе.
    """

    kind: str = Field(default="b2b", description="Тип аудитории: b2b, b2c, mixed, custom")
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
    """
    Основной запрос на редактирование текста.
    """

    text: str = Field(..., min_length=1, description="Текст для редактирования")
    domain: str = Field(
        default="marketing",
        description="Домен текста, например: marketing, blog, fiction",
    )
    intent: Optional[str] = Field(
        default=None,
        description="Intent из config/intents",
    )
    audience: Optional[AudienceRequest] = Field(
        default=None,
        description="Профиль аудитории",
    )
    overlays: List[str] = Field(
        default_factory=list,
        description="Список overlays из config/overlays",
    )
    output_mode: str = Field(
        default="text_only",
        description="Формат вывода: text_only или text_and_report",
    )
    provider: str = Field(
        default="openrouter",
        description="LLM provider: perplexity, openai, openrouter, anthropic",
    )
    model: Optional[str] = Field(
        default=None,
        description="Опционально: явное имя модели",
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
        available = get_available_intents()

        if normalized not in available:
            raise ValueError(
                f"intent '{value}' not found in config/intents. "
                f"Available: {sorted(available)}"
            )

        return normalized

    @validator("overlays")
    def validate_overlays(cls, value: List[str]) -> List[str]:
        available = get_available_overlays()
        normalized_values = [normalize_tag(item) for item in value]

        invalid = [item for item in normalized_values if item not in available]
        if invalid:
            raise ValueError(
                f"overlays {invalid} not found in config/overlays. "
                f"Available: {sorted(available)}"
            )

        return normalized_values

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
        allowed = _supported_providers()
        if normalized not in allowed:
            raise ValueError(f"provider must be one of {sorted(allowed)}")
        return normalized

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Наш сервис является самым лучшим на рынке.",
                "domain": "marketing",
                "intent": "storytelling",
                "audience": {
                    "kind": "b2b",
                    "expertise": "pro",
                    "formality": "neutral",
                    "description": "Менеджеры отделов продаж",
                },
                "overlays": ["infostyle"],
                "output_mode": "text_only",
                "provider": "openrouter",
                "model": "openrouter/auto",
                "temperature": 0.3,
            }
        }


class EditResponse(BaseModel):
    """
    Ответ API с отредактированным текстом.
    """

    edited_text: str = Field(..., description="Отредактированный текст")
    report: Optional[str] = Field(default=None, description="Отчёт о правках")
    model: str = Field(..., description="Использованная модель")
    provider: str = Field(..., description="Использованный провайдер")
    tokens_used: Optional[int] = Field(default=None, description="Количество токенов")


class HealthResponse(BaseModel):
    """
    Ответ health-check.
    """

    status: str
    version: str


class ErrorResponse(BaseModel):
    """
    Структура ошибки.
    """

    error: str
    detail: Optional[str] = None


@app.get("/", response_model=HealthResponse)
async def root() -> HealthResponse:
    """
    Корневой endpoint.
    """
    return HealthResponse(status="ok", version="1.0.0")


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health-check endpoint.
    """
    return HealthResponse(status="ok", version="1.0.0")


@app.post(
    "/api/edit",
    response_model=EditResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Некорректный запрос"},
        422: {"model": ErrorResponse, "description": "Ошибка валидации"},
        500: {"model": ErrorResponse, "description": "Внутренняя ошибка сервера"},
    },
)
async def edit_text(request: EditRequest) -> EditResponse:
    """
    Основной endpoint редактирования текста через LLM.
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
                "provider": response.provider,
                "model": response.model,
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

    except HTTPException:
        raise

    except Exception as error:  # noqa: BLE001
        logger.error("Unexpected error: %s", error, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {error}",
        ) from error


@app.post("/api/quick-edit")
async def quick_edit(text: str, audience_type: str = "b2b") -> dict:
    """
    Упрощённый endpoint для быстрого редактирования.
    Оставлен для обратной совместимости.
    """
    request = EditRequest(
        text=text,
        domain="marketing",
        audience=AudienceRequest(kind=audience_type),
        output_mode="text_only",
    )

    response = await edit_text(request)
    return {"edited_text": response.edited_text}


def _parse_text_and_report(content: str) -> Tuple[str, Optional[str]]:
    """
    Разбирает ответ режима text_and_report.

    Ожидаемый формат:
    ===ТЕКСТ===
    ...
    ===ОТЧЁТ===
    ...
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