"""
config_types.py

Типы данных для конфигурации PromptBuilder.
Вынесены в отдельный модуль, чтобы не засорять prompt_builder.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class AudienceProfile:
    """Профиль аудитории."""

    kind: str        # "b2b" | "b2c" | "mixed" | "custom"
    expertise: str   # "novice" | "pro" | "expert"
    formality: str   # "casual" | "neutral" | "formal"
    description: str = ""


@dataclass
class LimitsConfig:
    """
    Лимиты выдачи и кандидатов для всех блоков knowledge base.

    Вынесены из __init__ PromptBuilder, чтобы сигнатура оставалась чистой.
    Изменяй только нужные поля, остальные берут дефолты:

        limits = LimitsConfig(grammar=5, style=5)
        builder = PromptBuilder(limits=limits)
    """

    # --- Лимиты выдачи (сколько записей попадает в промпт) ---
    grammar: int = 10
    style: int = 10
    logic: int = 8
    composition: int = 6
    cohesion: int = 6
    composition_errors: int = 6
    storytelling: int = 4
    marketing: int = 4
    rhetoric: int = 4
    editorial: int = 6
    glossary: int = 10
    stop_words_category: int = 8
    stop_words_items: int = 5

    # --- Лимиты кандидатов (None = рассматривать все записи KB) ---
    grammar_candidates: Optional[int] = None
    style_candidates: Optional[int] = None
    logic_candidates: Optional[int] = None
    storytelling_candidates: Optional[int] = None
    marketing_candidates: Optional[int] = None
    rhetoric_candidates: Optional[int] = None
