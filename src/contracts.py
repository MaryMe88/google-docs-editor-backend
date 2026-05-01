from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

from src.tag_registry import normalize_tag

_ALLOWED_DOMAINS = {"marketing", "blog", "deai"}
_ALLOWED_INTENTS = {"storytelling", "noragal", "deai", "neutral"}
_ALLOWED_OVERLAYS = {"logic", "factcheck", "infostyle", "marketingpush"}
_ALLOWED_OUTPUT_MODES = {"text_only", "text_and_report"}
_ALLOWED_PROVIDERS = {"openrouter", "perplexity", "openai"}
_ALLOWED_KIND = {"b2b", "b2c", "mixed", "custom"}
_ALLOWED_EXPERTISE = {"novice", "pro", "expert"}
_ALLOWED_FORMALITY = {"casual", "neutral", "formal"}


class AudienceRequest(BaseModel):
    kind: str = Field(default="b2b")
    expertise: str = Field(default="pro")
    formality: str = Field(default="neutral")
    description: str = Field(default="")

    @validator("kind")
    def validate_kind(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in _ALLOWED_KIND:
            raise ValueError(f"kind must be one of {sorted(_ALLOWED_KIND)}")
        return normalized

    @validator("expertise")
    def validate_expertise(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in _ALLOWED_EXPERTISE:
            raise ValueError(f"expertise must be one of {sorted(_ALLOWED_EXPERTISE)}")
        return normalized

    @validator("formality")
    def validate_formality(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in _ALLOWED_FORMALITY:
            raise ValueError(f"formality must be one of {sorted(_ALLOWED_FORMALITY)}")
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
    include_knowledge: bool = Field(default=True)
    dry_run: bool = Field(default=False)

    @validator("domain")
    def validate_domain(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in _ALLOWED_DOMAINS:
            raise ValueError(f"domain must be one of {sorted(_ALLOWED_DOMAINS)}")
        return normalized

    @validator("intent")
    def validate_intent(cls, value: Optional[str]) -> Optional[str]:
        if value is None or not value.strip():
            return None
        normalized = normalize_tag(value)
        if normalized not in _ALLOWED_INTENTS:
            raise ValueError(f"intent must be one of {sorted(_ALLOWED_INTENTS)}")
        return normalized

    @validator("overlays")
    def validate_overlays(cls, value: List[str]) -> List[str]:
        normalized_values: List[str] = []
        seen = set()
        for item in value:
            normalized = normalize_tag(item)
            if normalized not in _ALLOWED_OVERLAYS:
                raise ValueError(f"overlay must be one of {sorted(_ALLOWED_OVERLAYS)}")
            if normalized not in seen:
                seen.add(normalized)
                normalized_values.append(normalized)
        return normalized_values

    @validator("output_mode")
    def validate_output_mode(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in _ALLOWED_OUTPUT_MODES:
            raise ValueError(f"output_mode must be one of {sorted(_ALLOWED_OUTPUT_MODES)}")
        return normalized

    @validator("provider")
    def validate_provider(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in _ALLOWED_PROVIDERS:
            raise ValueError(f"provider must be one of {sorted(_ALLOWED_PROVIDERS)}")
        return normalized


class EditResponse(BaseModel):
    edited_text: str
    prompt: str
    provider: Optional[str] = None
    model: Optional[str] = None
    dry_run: bool = False
    usage: Dict[str, Any] = Field(default_factory=dict)
    raw_response: Dict[str, Any] = Field(default_factory=dict)


class PromptResponse(BaseModel):
    prompt: str


class HealthResponse(BaseModel):
    status: str
    available_domains: List[str]
    available_intents: List[str]
    available_overlays: List[str]
    available_providers: List[str]
