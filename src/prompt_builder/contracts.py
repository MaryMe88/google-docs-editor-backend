"""Контракты для PromptBuilder: запрос, опции и результат сборки.

Итерация 15.1 карты №3: только типы, runtime-путь build() не меняется.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from src.config_types import AudienceProfile, KnowledgeLevel


@dataclass(frozen=True, kw_only=True)
class BuildOptions:
    """Опции сборки промпта, соответствующие keyword-only аргументам build()."""

    include_knowledge: bool = True
    include_few_shot: bool = True
    knowledge_level: KnowledgeLevel = KnowledgeLevel.STANDARD
    token_budget: int | None = None
    include_retrieval_meta: bool = False
    few_shot_seed: int | None = None
    deep_semantic_search: bool = False


@dataclass(frozen=True)
class BuildRequest:
    """Полный запрос на сборку промпта."""

    text: str
    domain: str
    intent: str | None = None
    audience: AudienceProfile | None = None
    overlays: Sequence[str] | None = None
    output_mode: str = "text_only"
    options: BuildOptions = field(default_factory=BuildOptions)


@dataclass(frozen=True)
class BuildResult:
    """Результат сборки промпта: строка + опциональные метаданные retrieval."""

    prompt: str
    retrieval_meta: dict[str, Any] | None = None
