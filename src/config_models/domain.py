"""
config_models.domain

Domain types: TypedDict'ы RuleEntry, StructuralEntry,
EditorialTechniqueEntry, FlatEntry; dataclass'ы CoreConfig, DomainConfig,
IntentConfig, OverlayConfig, AudienceProfile; контейнер KnowledgeBase
и sentinel _MISSING.

Выделено из src.config_types в итерации 7 дорожной карты.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict


# Sentinel для KnowledgeBase.get(): отличает «default не передан»
# от «default передан как None/False/0/""/{}».
_MISSING = object()


class RuleEntry(TypedDict, total=False):
    """Запись с правилом исправления (грамматика, стиль, логика)."""

    wrong: str
    correct: str
    rule: str
    description: str
    tags: list[str]
    category: str


class StructuralEntry(TypedDict, total=False):
    """Структурная запись (фреймворк, шаблон, приём)."""

    name: str
    description: str
    when_to_use: str | list[str]
    rule: str
    steps: list[dict[str, Any]]
    sections: list[dict[str, Any]]
    tags: list[str]


class EditorialTechniqueEntry(TypedDict, total=False):
    """Редакторский приём."""

    id: str
    name: str
    category: str
    description: str
    when_to_use: list[str]
    how_to_apply: list[str]
    example_wrong: str
    example_correct: str
    example_explanation: str
    tags: list[str]
    source: dict[str, Any]


FlatEntry = dict[str, Any]


@dataclass(frozen=True)
class CoreConfig:
    """Базовая конфигурация редактора."""

    role: str
    priorities: str
    basic_audit_instructions: tuple
    forbidden: tuple
    ip_ceiling: float = 2.5


@dataclass(frozen=True)
class DomainConfig:
    """Конфигурация домена."""

    name: str
    system_rules: str
    tone: str
    allow_storytelling: bool = True
    allow_marketing: bool = True
    tasks: tuple = field(default_factory=tuple)
    constraints: tuple = field(default_factory=tuple)
    ip_ceiling: float | None = None
    kb_limits: dict[str, int] = field(default_factory=dict)
    priority: int = 100
    suppresses: tuple = field(default_factory=tuple)
    conflicts_with: tuple = field(default_factory=tuple)
    incompatible_intents: tuple = field(default_factory=tuple)
    incompatible_overlays: tuple = field(default_factory=tuple)
    edit_level: str = "processing"


@dataclass(frozen=True)
class IntentConfig:
    """Конфигурация цели обработки."""

    name: str
    instructions: list[str]
    priority: int = 50
    suppresses: tuple = field(default_factory=tuple)
    conflicts_with: tuple = field(default_factory=tuple)


@dataclass(frozen=True)
class OverlayConfig:
    """Конфигурация оверлея."""

    name: str
    instructions: tuple
    conflicts_with: tuple = field(default_factory=tuple)
    priority: int = 70
    suppresses: tuple = field(default_factory=tuple)


@dataclass(frozen=True)
class AudienceProfile:
    """Профиль аудитории."""

    kind: str
    expertise: str
    formality: str
    description: str = ""


class KnowledgeBase:
    """
    Динамическая база знаний.

    Записи хранятся в _blocks по ключу — имени блока (например, "grammar_errors").
    Обратная совместимость: доступ к старым атрибутам (grammar_errors, stylistic_issues и т.д.)
    реализован через __getattr__, поэтому существующий код не ломается.

    Методы:
        get(key, default=<sentinel>) — получить блок по ключу.
            Если default не передан, возвращается пустой список
            (обратная совместимость со старым поведением).
            Явно переданные None, False, 0, "", {} возвращаются как есть,
            без подмены на [] — как в dict.get().
        register(key, data) — установить блок.
        keys() — список всех ключей.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Создаёт KnowledgeBase из именованных аргументов.
        Каждый аргумент становится блоком с соответствующим именем.
        """
        self._blocks: dict[str, Any] = {}
        for key, value in kwargs.items():
            self._blocks[key] = value

    # FIX 3.1: явно переданный falsy default больше не подменяется на [].
    def get(self, key: str, default: Any = _MISSING) -> Any:
        """Возвращает блок по ключу или default.

        Если default не передан, возвращается пустой список (обратная
        совместимость). Явно переданные None, False, 0, "", {} возвращаются
        без подмены — как в dict.get().
        """
        if key in self._blocks:
            return self._blocks[key]
        if default is _MISSING:
            return []
        return default

    def register(self, key: str, data: Any) -> None:
        """Регистрирует (перезаписывает) блок с именем key."""
        self._blocks[key] = data

    def keys(self) -> set[str]:
        """Возвращает множество имён блоков."""
        return set(self._blocks.keys())

    def __getattr__(self, name: str) -> Any:
        """
        Обеспечивает обратную совместимость:
        kb.grammar_errors → self._blocks["grammar_errors"]
        Если ключа нет, возбуждается AttributeError.
        """
        try:
            return self._blocks[name]
        except KeyError:
            raise AttributeError(f"KnowledgeBase has no block '{name}'")

    def __repr__(self) -> str:
        return f"KnowledgeBase(blocks={list(self._blocks.keys())})"
