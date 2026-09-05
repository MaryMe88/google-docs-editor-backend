import re
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

# SEC-патч 3.3: строгий allowlist для поля model.
_MODEL_NAME_RE = re.compile(r"^[\w./:-]{1,200}$")


class AudienceRequest(BaseModel):
    kind: str = Field(default="b2b")
    expertise: str = Field(default="pro")
    formality: str = Field(default="neutral")
    # SEC: ограничение длины предотвращает prompt injection через свободное текстовое поле,
    # которое попадает в LLM-промпт через _build_audience_block без санитизации.
    description: str = Field(default="", max_length=500)

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
    # SEC: max_length предотвращает передачу произвольной строки напрямую
    # в payload["model"] к LLM API без ограничений.
    model: Optional[str] = Field(default=None, max_length=200)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    include_knowledge: bool = Field(default=True)
    include_retrieval_meta: bool = Field(default=False)
    include_few_shot: bool = Field(default=True)
    dry_run: bool = Field(default=False)
    # Углублённая семантическая проверка: включает re-ranking правил KB через sentence-transformers.
    # При включении запрос становится медленнее, но точнее для творческих и композиционных режимов.
    deep_semantic_search: bool = Field(default=False)

    @field_validator("domain")
    @classmethod
    def validate_domain(cls, v: str) -> str:
        """Нормализует и проверяет domain через ALLOWED_DOMAINS.

        Используем strip().lower() — НЕ normalize_tag(), потому что
        normalize_tag удаляет подчёркивания («basic_edit» → «basicedit»),
        тогда как имена доменов содержат значимые подчёркивания.
        """
        normalized = v.strip().lower()
        if normalized not in ALLOWED_DOMAINS:
            raise ValueError(f"Unknown domain: {v!r}. Allowed: {sorted(ALLOWED_DOMAINS)}")
        return normalized

    @field_validator("intent")
    @classmethod
    def validate_intent(cls, v: Optional[str]) -> Optional[str]:
        """Нормализует и проверяет intent через ALLOWED_INTENTS.

        Используем strip().lower() — НЕ normalize_tag(), по той же причине:
        имена интентов содержат значимые подчёркивания (fix_flow, add_hooks и т.д.).
        """
        if v is None or not v.strip():
            return None
        normalized = v.strip().lower()
        if normalized not in ALLOWED_INTENTS:
            raise ValueError(f"Unknown intent: {v!r}. Allowed: {sorted(ALLOWED_INTENTS)}")
        return normalized

    @field_validator("overlays", mode="before")
    @classmethod
    def validate_overlays(cls, v: list) -> list[str]:
        """Нормализует, проверяет через ALLOWED_OVERLAYS и дедуплицирует.

        Для оверлеев normalize_tag() уместен: у них есть алиасы и
        варианты написания (finalcheck / final_check / FinalCheck).
        """
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

    # SEC-патч 3.3: Allowlist для поля model.
    @field_validator("model")
    @classmethod
    def validate_model(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        # Убираем лишние пробелы? Обычно модель не должна содержать пробелы.
        v = v.strip()
        if not _MODEL_NAME_RE.match(v):
            raise ValueError("model contains disallowed characters")
        return v


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