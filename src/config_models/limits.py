"""
config_models.limits

LimitsConfig — лимиты выдачи и кандидатов для всех блоков KB.

Выделено из src.config_types в итерации 7 дорожной карты.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LimitsConfig:
    """
    Лимиты выдачи и кандидатов для всех блоков KB.
    Параметры *_candidates задают, сколько записей рассматривается
    перед ранжированием (None = все).
    """

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
    stop_words_category: int = 15
    stop_words_items: int = 5
    nkrj: int = 4
    casestudy: int = 4
    evaluation_techniques: int = 8

    grammar_candidates: int | None = None
    style_candidates: int | None = None
    logic_candidates: int | None = None
    storytelling_candidates: int | None = None
    marketing_candidates: int | None = None
    rhetoric_candidates: int | None = None
