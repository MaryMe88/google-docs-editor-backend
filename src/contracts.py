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

    @field_validator("domain")
    @classmethod
    def validate_domain(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in ALLOWED_DOMAINS:
            raise ValueError(f"domain must be one of {sorted(ALLOWED_DOMAINS)}")
        return normalized

    @field_validator("intent")
    @classmethod
    def validate_intent(cls, value: Optional[str]) -> Optional[str]:
        if value is None or not value.strip():
            return None

        normalized = normalize_tag(value)
        if normalized not in ALLOWED_INTENTS:
            raise ValueError(f"intent must be one of {sorted(ALLOWED_INTENTS)}")
        return normalized

    @field_validator("overlays")
    @classmethod
    def validate_overlays(cls, value: List[str]) -> List[str]:
        normalized_values: List[str] = []
        seen = set()

        for item in value:
            normalized = normalize_tag(item)
            if normalized not in ALLOWED_OVERLAYS:
                raise ValueError(
                    f"overlay must be one of {sorted(ALLOWED_OVERLAYS)}"
                )

            if normalized not in seen:
                seen.add(normalized)
                normalized_values.append(normalized)

        return normalized_values

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