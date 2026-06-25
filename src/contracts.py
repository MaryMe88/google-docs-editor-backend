from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from src.shared_contracts import (
    ALLOWED_DOMAINS,
    ALLOWED_EXPERTISE,
    ALLOWED_FORMALITY,
    ALLOWED_INTENTS,
    ALLOWED_KIND,
    ALLOWED_OUTPUT_MODES,
    ALLOWED_OVERLAYS,
    ALLOWED_PROVIDERS,
)
from src.tag_registry import normalize_tag

# Версия контракта между бекендом и клиентом (Google Apps Script).
# При несовместимых изменениях увеличивать мажорную версию.
# 1.1.0 — добавлено поле report в EditResponse;
#          HealthResponse приведён в соответствие с реальным ответом /health.
CONTRACT_VERSION: str = "1.1.0"


class AudienceRequest(BaseModel):
    kind: str = Field(default="b2b")
    expertise: str = Field(default="pro")
    formality: str = Field(default="neutral")
    description: str = Field(default="")

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in ALLOWED_KIND:
            raise ValueError(f"kind must be one of {sorted(ALLOWED_KIND)}")
        return normalized

    @field_validator("expertise")
    @classmethod
    def validate_expertise(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in ALLOWED_EXPERTISE:
            raise ValueError(
                f"expertise must be one of {sorted(ALLOWED_EXPERTISE)}"
            )
        return normalized

    @field_validator("formality")
    @classmethod
    def validate_formality(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in ALLOWED_FORMALITY:
            raise ValueError(
                f"formality must be one of {sorted(ALLOWED_FORMALITY)}"
            )
        return normalized


class EditRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    domain: str = Field(default="marketing")
    intent: Optional[str] = Field(default=None)
    audience: Optional[AudienceRequest] = Field(default=None)
    overlays: List[str] = Field(default_factory=list)
    output_mode: str = Field(default="text_only")
    provider: str = Field(default="openrouter")
    model: Optional[str] = Field(default=None)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    include_knowledge: bool = Field(default=True)
    include_retrieval_meta: bool = Field(default=False)
    include_few_shot: bool = Field(default=True)
    dry_run: bool = Field(default=False)

    # ИЗМЕНЕНИЕ A-1: валидаторы теперь проверяют допустимость и нормализуют через normalize_tag
    @field_validator("domain")
    @classmethod
    def validate_domain(cls, v: str) -> str:
        """Нормализует и проверяет domain через ALLOWED_DOMAINS."""
        normalized = normalize_tag(v)
        if normalized not in ALLOWED_DOMAINS:
            raise ValueError(f"Unknown domain: {v!r}. Allowed: {sorted(ALLOWED_DOMAINS)}")
        return normalized

    @field_validator("intent")
    @classmethod
    def validate_intent(cls, v: Optional[str]) -> Optional[str]:
        """Нормализует и проверяет intent через ALLOWED_INTENTS."""
        if v is None or not v.strip():
            return None
        normalized = normalize_tag(v)
        if normalized not in ALLOWED_INTENTS:
            raise ValueError(f"Unknown intent: {v!r}. Allowed: {sorted(ALLOWED_INTENTS)}")
        return normalized

    @field_validator("overlays", mode="before")
    @classmethod
    def validate_overlays(cls, v: list) -> list[str]:
        """Нормализует, проверяет через ALLOWED_OVERLAYS и дедуплицирует."""
        result = []
        seen = set()
        for item in v:
            normalized = normalize_tag(str(item))
            if normalized not in ALLOWED_OVERLAYS:
                raise ValueError(f"Unknown overlay: {item!r}. Allowed: {sorted(ALLOWED_OVERLAYS)}")
            if normalized not in seen:
                seen.add(normalized)
                result.append(normalized)
        return result

    @field_validator("output_mode")
    @classmethod
    def validate_output_mode(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in ALLOWED_OUTPUT_MODES:
            raise ValueError(
                f"output_mode must be one of {sorted(ALLOWED_OUTPUT_MODES)}"
            )
        return normalized

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in ALLOWED_PROVIDERS:
            raise ValueError(
                f"provider must be one of {sorted(ALLOWED_PROVIDERS)}"
            )
        return normalized


class EditResponse(BaseModel):
    """Ответ эндпоинта POST /api/edit.

    Поле report заполняется только в режиме output_mode='text_and_report'.
    В остальных режимах report=None.

    Поле prompt удалено из ответа (Шаг 4) — внутренняя информация не раскрывается клиенту.
    """

    edited_text: str
    report: Optional[str] = None  # PR-2 (НП-2): добавлено для режима text_and_report
    # prompt: str  # УДАЛЕНО: не возвращаем промпт клиенту
    provider: Optional[str] = None
    model: Optional[str] = None
    dry_run: bool = False
    usage: Dict[str, Any] = Field(default_factory=dict)
    raw_response: Dict[str, Any] = Field(default_factory=dict)
    retrieval_meta: Optional[Dict[str, Any]] = Field(default=None)


class PromptResponse(BaseModel):
    prompt: str


class HealthResponse(BaseModel):
    """Ответ эндпоинта GET /health.

    PR-3 (НП-3): приведён в соответствие с реальным ответом:
    - добавлены provider_status, deep_check, version
    - available_domains добавлен явно (был в модели, но не возвращался эндпоинтом)
    """

    status: str
    version: str = "1.0.0"
    available_domains: List[str]
    available_intents: List[str]
    available_overlays: List[str]
    available_providers: List[str]
    provider_status: Dict[str, bool]
    deep_check: bool = False
    contract_version: str = CONTRACT_VERSION