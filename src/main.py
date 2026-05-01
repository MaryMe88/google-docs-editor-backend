from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Callable, List, Optional, Set

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
    for name in names:
        method = getattr(builder, name, None)
        if callable(method):
            return method()
    raise AttributeError(
        f"PromptBuilder does not have any of methods: {', '.join(names)}"
    )


def _supported_providers() -> Set[str]:
    return {
        LLMProvider.PERPLEXITY.value,
        LLMProvider.OPENAI.value,
        LLMProvider.OPENROUTER.value,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up text editor service...")

    try:
        prompt_builder = PromptBuilder()

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
    prompt_builder = getattr(app.state, "prompt_builder", None)
    if prompt_builder is None:
        raise RuntimeError("PromptBuilder is not initialized")
    return prompt_builder


def get_available_intents() -> Set[str]:
    builder = get_prompt_builder()
    values = _call_builder_method(
        builder,
        "get_available_intents",
        "getavailableintents",
    )
    return {normalize_tag(item) for item in values}


def get_available_overlays() -> Set[str]:
    builder = get_prompt_builder()
    values = _call_builder_method(
        builder,
        "get_available_overlays",
        "getavailableoverlays",
    )
    return {normalize_tag(item) for item in values}


class AudienceRequest(BaseModel):
    kind: str = Field(default="b2b")
    expertise: str = Field(default="pro")
    formality: str = Field(default="neutral")
    description: str = Field(default="")

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
    text: str = Field(..., min_length=1)
    domain: str = Field(default="marketing")
    intent: Optional[str] = Field(default=None)
    audience: Optional[AudienceRequest] = Field(default=None)
    overlays: List[str] = Field(default_factory=list)
    output_mode: str = Field(default="text_only")
    provider: str = Field(default="openrouter")
    model: Optional[str] = Field(default=None)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)

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


class EditResponse(BaseModel):
    edited_text: str
    report: Optional[str] = None
    model: str
    provider: str
    tokens_used: Optional[int] = None