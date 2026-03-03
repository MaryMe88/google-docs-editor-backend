"""
main.py

FastAPI сервер для редактора текстов.
Принимает запросы из Google Docs, обрабатывает через LLM и возвращает результат.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from src.llm_client import (
    LLMError,
    LLMProvider,
    create_llm_client,
)
from src.prompt_builder import AudienceProfile, PromptBuilder

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Lifespan Management (инициализация и очистка ресурсов)
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Управление жизненным циклом приложения.
    Инициализация и очистка ресурсов.
    """
    # Startup
    logger.info("Starting up text editor service...")
    
    # Инициализируем PromptBuilder (проверяем конфиги)
    try:
        app.state.prompt_builder = PromptBuilder()
        logger.info("PromptBuilder initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize PromptBuilder: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down text editor service...")


# ============================================================================
# FastAPI App
# ============================================================================


app = FastAPI(
    title="Text Editor API",
    description="API для редактирования текстов с помощью LLM",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware (для вызовов из Google Apps Script)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене укажи конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Request/Response Models
# ============================================================================


class AudienceRequest(BaseModel):
    """Модель профиля аудитории в запросе."""
    kind: str = Field(default="b2b", description="Тип аудитории: b2b, b2c, mixed")
    expertise: str = Field(default="pro", description="Уровень: novice, pro, expert")
    formality: str = Field(default="neutral", description="Формальность: casual, neutral, formal")
    description: str = Field(default="", description="Дополнительное описание")
    
    @validator("kind")
    def validate_kind(cls, v):
        allowed = ["b2b", "b2c", "mixed", "custom"]
        if v not in allowed:
            raise ValueError(f"kind must be one of {allowed}")
        return v
    
    @validator("expertise")
    def validate_expertise(cls, v):
        allowed = ["novice", "pro", "expert"]
        if v not in allowed:
            raise ValueError(f"expertise must be one of {allowed}")
        return v
    
    @validator("formality")
    def validate_formality(cls, v):
        allowed = ["casual", "neutral", "formal"]
        if v not in allowed:
            raise ValueError(f"formality must be one of {allowed}")
        return v


class EditRequest(BaseModel):
    """Модель запроса на редактирование текста."""
    text: str = Field(..., min_length=1, description="Текст для редактирования")
    domain: str = Field(default="marketing", description="Домен текста: marketing, blog, fiction")
    
    intent: Optional[str] = Field(
        default=None, 
        description="Цель обработки: analytical (точнее+логичнее), storytelling (увлекательнее), marketing_push (продающим)"
    )
    
    audience: Optional[AudienceRequest] = Field(default=None, description="Профиль аудитории")
    
    overlays: List[str] = Field(
        default_factory=list, 
        description="Дополнительные режимы: infostyle (жёсткий инфостиль), factcheck (отчёт по фактам), recommendations (советы)"
    )
    
    output_mode: str = Field(default="text_only", description="Формат вывода: text_only, text_and_report")
    
    # LLM параметры
    provider: str = Field(default="perplexity", description="LLM провайдер")
    model: Optional[str] = Field(default=None, description="Модель LLM")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="Температура генерации")
    
    @validator("intent")
    def validate_intent(cls, v):
        if v is not None:
            allowed = ["analytical", "storytelling", "marketing_push"]
            if v not in allowed:
                raise ValueError(f"intent must be one of {allowed} or None")
        return v
    
    @validator("overlays")
    def validate_overlays(cls, v):
        allowed = ["infostyle", "factcheck", "recommendations", "final_check"]
        for overlay in v:
            if overlay not in allowed:
                raise ValueError(f"overlay '{overlay}' not allowed. Allowed: {allowed}")
        return v
    
    @validator("output_mode")
    def validate_output_mode(cls, v):
        allowed = ["text_only", "text_and_report"]
        if v not in allowed:
            raise ValueError(f"output_mode must be one of {allowed}")
        return v
    
    @validator("provider")
    def validate_provider(cls, v):
        allowed = ["perplexity", "openai", "anthropic"]
        if v not in allowed:
            raise ValueError(f"provider must be one of {allowed}")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Наш сервис является самым лучшим на рынке.",
                "domain": "marketing",
                "intent": "analytical",
                "audience": {
                    "kind": "b2b",
                    "expertise": "pro",
                    "formality": "neutral",
                    "description": "Менеджеры отделов продаж"
                },
                "overlays": ["infostyle"],
                "output_mode": "text_only"
            }
        }


class EditResponse(BaseModel):
    """Модель ответа с отредактированным текстом."""
    edited_text: str = Field(..., description="Отредактированный текст")
    report: Optional[str] = Field(default=None, description="Отчёт о правках (если запрошен)")
    model: str = Field(..., description="Использованная модель")
    provider: str = Field(..., description="Использованный провайдер")
    tokens_used: Optional[int] = Field(default=None, description="Количество использованных токенов")


class HealthResponse(BaseModel):
    """Модель ответа health check."""
    status: str
    version: str


class ErrorResponse(BaseModel):
    """Модель ответа с ошибкой."""
    error: str
    detail: Optional[str] = None


# ============================================================================
# Endpoints
# ============================================================================


@app.get("/", response_model=HealthResponse)
async def root():
    """Корневой эндпоинт."""
    return HealthResponse(status="ok", version="1.0.0")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check эндпоинт."""
    return HealthResponse(status="ok", version="1.0.0")


@app.post(
    "/api/edit",
    response_model=EditResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Некорректный запрос"},
        500: {"model": ErrorResponse, "description": "Внутренняя ошибка сервера"},
    }
)
async def edit_text(request: EditRequest):
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
        }
    )
    
    try:
        # 1. Конвертируем audience в нужный формат
        audience = None
        if request.audience:
            audience = AudienceProfile(
                kind=request.audience.kind,
                expertise=request.audience.expertise,
                formality=request.audience.formality,
                description=request.audience.description,
            )
        
        # 2. Собираем промпт
        prompt_builder = app.state.prompt_builder
        prompt = prompt_builder.build(
            text=request.text,
            domain=request.domain,
            intent=request.intent,
            audience=audience,
            overlays=request.overlays,
            output_mode=request.output_mode,
            include_knowledge=True,  # <── добавили явный флаг
        )
        
        logger.info(
            "Prompt built successfully",
            extra={"prompt_length": len(prompt)}
        )
        
        # 3. Вызываем LLM
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
            }
        )
        
        # 4. Парсим ответ (если нужен отчёт)
        edited_text = response.content
        report = None
        
        if request.output_mode == "text_and_report":
            edited_text, report = _parse_text_and_report(response.content)
        
        # 5. Возвращаем результат
        return EditResponse(
            edited_text=edited_text,
            report=report,
            model=response.model,
            provider=response.provider,
            tokens_used=response.tokens_used,
        )

    
    except LLMError as e:
        logger.error(f"LLM error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LLM generation failed: {str(e)}"
        )
    
    except FileNotFoundError as e:
        logger.error(f"Config file not found: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Configuration error: {str(e)}"
        )
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/api/quick-edit")
async def quick_edit(text: str, audience_type: str = "b2b"):
    """
    Упрощённый эндпоинт для быстрого редактирования.
    Для обратной совместимости с MVP.
    
    Args:
        text: Текст для редактирования
        audience_type: Тип аудитории (b2b, b2c, mixed)
        
    Returns:
        Отредактированный текст
    """
    request = EditRequest(
        text=text,
        domain="marketing",
        audience=AudienceRequest(kind=audience_type),
        output_mode="text_only"
    )
    
    response = await edit_text(request)
    return {"edited_text": response.edited_text}


# ============================================================================
# Helper Functions
# ============================================================================


def _parse_text_and_report(content: str) -> tuple[str, Optional[str]]:
    """
    Парсит ответ в режиме text_and_report.
    
    Ожидается формат:
    ===ТЕКСТ===
    [текст]
    
    ===ОТЧЁТ===
    [отчёт]
    
    Args:
        content: Ответ от LLM
        
    Returns:
        Кортеж (текст, отчёт)
    """
    text_marker = "===ТЕКСТ==="
    report_marker = "===ОТЧЁТ==="
    
    if text_marker in content and report_marker in content:
        parts = content.split(report_marker)
        text_part = parts[0].replace(text_marker, "").strip()
        report_part = parts[1].strip() if len(parts) > 1 else None
        return text_part, report_part
    
    # Если маркеров нет — считаем, что всё — текст
    return content.strip(), None


# ============================================================================
# Run (для локального запуска)
# ============================================================================


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
