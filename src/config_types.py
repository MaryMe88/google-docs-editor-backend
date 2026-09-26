"""
config_types.py

Dataclasses, enum'ы и инфраструктурные типы для конфигурирования PromptBuilder.

Содержит:
- Domain types — RuleEntry, KnowledgeBase, CoreConfig и т.д.
- LimitsConfig — лимиты выдачи и кандидатов
- KnowledgeLevel — режим включения блоков знаний
- KnowledgeBlockPlan — описание блока для budget-aware сборки
- BlockBudget — бюджет одного блока KB
- KnowledgeBudget — совокупный бюджет всех блоков
- KnowledgeBudgetManager — вычисляет бюджет
- CachePolicy — политика инвалидации кэша
- FileCache — кэш-менеджер с поддержкой TTL/mtime
- Tag constants — CANONICAL_TAGS, KNOWN_TAGS, get_*_tags_for_category
- Explainability structures — FeatureResolutionResult, AssemblyBlockDiagnostics, AssemblyTrace
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Generic,
    TypeVar,
)

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

# Импорт для обратной совместимости: код проекта исторически ожидает,
# что ReasonCode доступен через config_types (используется в тестах и
# в модулях, импортирующих ReasonCode через config_types).
from src.reason_codes import ReasonCode  # noqa: F401

logger = logging.getLogger(__name__)
V = TypeVar("V")

# Sentinel для KnowledgeBase.get(): отличает «default не передан»
# от «default передан как None/False/0/""/{}».
_MISSING = object()


# ============================================================================
# Domain types — TypedDict и dataclass'ы для конфигов и базы знаний
# ============================================================================


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


# ============================================================================
# Dataclass'ы для конфигов
# ============================================================================


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


# ============================================================================
# KnowledgeBase — динамический контейнер блоков
# ============================================================================


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


# ============================================================================
# LimitsConfig — лимиты выдачи и кандидатов
# ============================================================================


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


# ============================================================================
# KnowledgeLevel — режим включения блоков знаний
# ============================================================================


class KnowledgeLevel(str, Enum):
    """
    Режим включения блоков базы знаний в промпт.

    NONE — база знаний не включается совсем.
    CORE — только обязательные блоки: grammar, style, stop_words.
    STANDARD — CORE + logic, composition, cohesion, composition_errors,
    nkrj, glossary.
    FULL — все доступные блоки, включая storytelling, marketing,
    rhetoric, editorial, casestudy, evaluation_techniques.
    """

    NONE = "none"
    CORE = "core"
    STANDARD = "standard"
    FULL = "full"


KNOWLEDGE_BUDGET_CHARS: dict[KnowledgeLevel, int] = {
    KnowledgeLevel.NONE: 0,
    KnowledgeLevel.CORE: 4_000,
    KnowledgeLevel.STANDARD: 10_000,
    KnowledgeLevel.FULL: 16_000,
}

_LEVEL_BLOCKS: dict[KnowledgeLevel, set[str]] = {
    KnowledgeLevel.NONE: set(),
    KnowledgeLevel.CORE: {"grammar", "style", "stop_words"},
    KnowledgeLevel.STANDARD: {
        "grammar",
        "style",
        "stop_words",
        "logic",
        "composition",
        "cohesion",
        "composition_errors",
        "nkrj",
        "glossary",
    },
    KnowledgeLevel.FULL: {
        "grammar",
        "style",
        "stop_words",
        "logic",
        "composition",
        "cohesion",
        "composition_errors",
        "nkrj",
        "glossary",
        "storytelling",
        "marketing",
        "rhetoric",
        "editorial",
        "casestudy",
        "evaluation_techniques",
    },
}


def blocks_allowed_at_level(level: KnowledgeLevel) -> set[str]:
    """Возвращает множество имён блоков, разрешённых на данном уровне."""
    return _LEVEL_BLOCKS.get(level, set())


# ============================================================================
# KnowledgeBlockPlan
# ============================================================================


@dataclass
class KnowledgeBlockPlan:
    """
    Описание одного блока знаний для budget-aware сборки.

    Атрибуты:
        name: Идентификатор блока.
        priority: Порядок включения (меньше = важнее).
        min_level: Минимальный KnowledgeLevel для включения.
        mandatory: Если True — включается всегда при level >= min_level.
        estimated_chars: Оценка размера блока в символах. Вычисляется лениво.
        builder: Callable без аргументов, возвращающий str блока.
        enable_condition: Дополнительное runtime-условие включения.
    """

    name: str
    priority: int
    min_level: KnowledgeLevel
    mandatory: bool = False
    estimated_chars: int = 0
    builder: Callable[[], str] | None = field(default=None, repr=False)
    enable_condition: bool = True


# ============================================================================
# BlockBudget, KnowledgeBudget, KnowledgeBudgetManager
# ============================================================================


@dataclass(frozen=True)
class BlockBudget:
    """
    Бюджет одного блока KB.

    Атрибуты:
        entry_limit: Максимальное количество записей для выдачи.
        char_budget: Мягкий лимит символов (None = без ограничений).
        enabled: Блок разрешён к включению в промпт.
    """

    entry_limit: int
    char_budget: int | None
    enabled: bool = True

    @property
    def charbudget(self) -> int | None:
        """Совместимость со старым legacy-кодом, использующим charbudget."""
        return self.char_budget


class KnowledgeBudget:
    """
    Совокупный бюджет всех блоков KB для одного вызова build().
    Реализован как dict-like объект: budget.get("grammar") → BlockBudget.
    Атрибуты grammar, style, logic, ... — шорткаты для читаемости.
    """

    _BLOCK_NAMES = (
        "grammar",
        "style",
        "logic",
        "composition",
        "cohesion",
        "composition_errors",
        "storytelling",
        "marketing",
        "rhetoric",
        "editorial",
        "glossary",
        "stop_words",
        "nkrj",
        "casestudy",
        "evaluation_techniques",
    )

    def __init__(self, budgets: dict[str, BlockBudget]) -> None:
        self._budgets = budgets

    def get(self, block_name: str) -> BlockBudget | None:
        """Возвращает BlockBudget по имени блока."""
        return self._budgets.get(block_name)

    def disable(self, key: str) -> None:
        """
        Отключает блок знаний по имени.
        Если блок не найден, молча игнорирует (не бросает исключение).
        """
        if key not in self._budgets:
            return
        old = self._budgets[key]
        # BlockBudget frozen, создаём новый с enabled=False
        self._budgets[key] = BlockBudget(
            entry_limit=old.entry_limit,
            char_budget=old.char_budget,
            enabled=False,
        )

    def __getattr__(self, name: str) -> BlockBudget:
        if name.startswith("_"):
            raise AttributeError(name)
        block_budget = self._budgets.get(name)
        if block_budget is None:
            raise AttributeError(f"Block '{name}' not in KnowledgeBudget")
        return block_budget

    def __repr__(self) -> str:
        return f"KnowledgeBudget({self._budgets!r})"


class KnowledgeBudgetManager:
    """
    Вычисляет KnowledgeBudget из LimitsConfig и KnowledgeLevel.

    Если token_budget задан — равномерно распределяет char_budget по блокам.
    Иначе — char_budget = None (без ограничений), entry_limit из LimitsConfig.
    Блоки, не разрешённые на текущем KnowledgeLevel, получают enabled=False.
    """

    def __init__(self, token_budget: int | None = None) -> None:
        """
        Args:
            token_budget: Приблизительный лимит токенов под блок «База знаний».
            1 токен ≈ 4 символа (heuristic). None = без ограничений.
        """
        self._token_budget = token_budget
        self._char_budget: int | None = token_budget * 4 if token_budget is not None else None

    def allocate(
        self,
        limits: LimitsConfig,
        active_blocks: set[str] | None = None,
        level: KnowledgeLevel = KnowledgeLevel.FULL,
    ) -> KnowledgeBudget:
        """
        Вычисляет и возвращает KnowledgeBudget.

        Args:
            limits: LimitsConfig с лимитами выдачи.
            active_blocks: Блоки, которые реально будут собираться.
            level: Текущий KnowledgeLevel для фильтрации блоков.
        """
        allowed = blocks_allowed_at_level(level)
        effective_active = active_blocks or set(KnowledgeBudget._BLOCK_NAMES)

        enabled_set = (
            effective_active & allowed if level != KnowledgeLevel.FULL else effective_active
        )

        n_enabled = len(enabled_set) or 1
        per_block_chars: int | None = (
            self._char_budget // n_enabled if self._char_budget is not None else None
        )

        def _blk(name: str, entry_limit: int) -> BlockBudget:
            # Блок отключается, если его entry_limit равен нулю
            is_enabled = (level == KnowledgeLevel.FULL or name in allowed) and entry_limit > 0
            return BlockBudget(
                entry_limit=entry_limit,
                char_budget=per_block_chars if is_enabled else None,
                enabled=is_enabled,
            )

        return KnowledgeBudget(
            {
                "grammar": _blk("grammar", limits.grammar),
                "style": _blk("style", limits.style),
                "logic": _blk("logic", limits.logic),
                "composition": _blk("composition", limits.composition),
                "cohesion": _blk("cohesion", limits.cohesion),
                "composition_errors": _blk("composition_errors", limits.composition_errors),
                "storytelling": _blk("storytelling", limits.storytelling),
                "marketing": _blk("marketing", limits.marketing),
                "rhetoric": _blk("rhetoric", limits.rhetoric),
                "editorial": _blk("editorial", limits.editorial),
                "glossary": _blk("glossary", limits.glossary),
                "stop_words": _blk("stop_words", limits.stop_words_category),
                "nkrj": _blk("nkrj", limits.nkrj),
                "casestudy": _blk("casestudy", limits.casestudy),
                "evaluation_techniques": _blk(
                    "evaluation_techniques", limits.evaluation_techniques
                ),
            }
        )


# ============================================================================
# CachePolicy и FileCache
# ============================================================================


@dataclass
class CachePolicy:
    """
    Политика инвалидации кэша.

    Атрибуты:
        check_mtime: Инвалидировать при изменении mtime файла.
        ttl_seconds: Время жизни кэша в секундах (None = без TTL).

    Рекомендуемые режимы:
        prod: CachePolicy(check_mtime=True)
        dev: CachePolicy(check_mtime=True, ttl_seconds=30)
        test: CachePolicy(check_mtime=False, ttl_seconds=None)
    """

    check_mtime: bool = True
    ttl_seconds: float | None = None


@dataclass
class _CacheEntry(Generic[V]):
    """Внутренняя запись кэша."""

    value: V
    path: Path | None
    loaded_at: float
    mtime_at_load: float | None


class FileCache:
    """
    Кэш файловых данных с поддержкой TTL и mtime-инвалидации.

    Использование:
        cache = FileCache(policy=CachePolicy(check_mtime=True))
        data = cache.get_or_load("key", path, loader_fn, *loader_args)
    """

    def __init__(self, policy: CachePolicy | None = None) -> None:
        self._policy = policy or CachePolicy(check_mtime=True)
        self._store: dict[str, _CacheEntry[Any]] = {}

    def _is_valid(self, entry: _CacheEntry[Any]) -> bool:
        """Проверяет актуальность записи кэша."""
        now = time.monotonic()

        if self._policy.ttl_seconds is not None:
            if now - entry.loaded_at > self._policy.ttl_seconds:
                return False

        if self._policy.check_mtime and entry.path is not None:
            try:
                current_mtime = entry.path.stat().st_mtime
                if entry.mtime_at_load is None or current_mtime != entry.mtime_at_load:
                    return False
            except OSError:
                return False

        return True

    def get_or_load(
        self,
        key: str,
        path: Path | None,
        loader: Callable[..., V],
        *loader_args: Any,
    ) -> V:
        """
        Возвращает закэшированное значение или загружает через loader(*loader_args).

        Args:
            key: Ключ кэша (уникальный идентификатор значения).
            path: Путь к файлу для mtime-инвалидации (None = без mtime).
            loader: Callable, возвращающий значение.
            loader_args: Позиционные аргументы для loader.
        """
        entry = self._store.get(key)
        if entry is not None and self._is_valid(entry):
            return entry.value

        value = loader(*loader_args)

        mtime: float | None = None
        if path is not None and self._policy.check_mtime:
            with suppress(OSError):
                mtime = path.stat().st_mtime

        self._store[key] = _CacheEntry(
            value=value,
            path=path,
            loaded_at=time.monotonic(),
            mtime_at_load=mtime,
        )
        return value

    def get_or_load_multi(
        self,
        key: str,
        paths: list[Path],
        loader: Callable[..., V],
        *loader_args: Any,
    ) -> V:
        """
        Кэширует результат loader с инвалидацией по нескольким файлам.
        Инвалидируется если изменился mtime любого из paths.
        """
        entry = self._store.get(key)

        if entry is not None:
            if self._policy.ttl_seconds is not None:
                if time.monotonic() - entry.loaded_at > self._policy.ttl_seconds:
                    entry = None

            if entry is not None and self._policy.check_mtime and entry.mtime_at_load is not None:
                try:
                    current_max = max(
                        (path.stat().st_mtime for path in paths if path.exists()),
                        default=0.0,
                    )
                    if current_max != entry.mtime_at_load:
                        entry = None
                except OSError:
                    entry = None

            if entry is not None:
                return entry.value

        value = loader(*loader_args)

        try:
            max_mtime: float | None = max(
                (path.stat().st_mtime for path in paths if path.exists()),
                default=None,
            )
        except OSError:
            max_mtime = None

        self._store[key] = _CacheEntry(
            value=value,
            path=None,
            loaded_at=time.monotonic(),
            mtime_at_load=max_mtime,
        )
        return value

    def invalidate(self, key: str) -> None:
        """Удаляет запись из кэша."""
        self._store.pop(key, None)

    def clear(self) -> None:
        """Сбрасывает весь кэш."""
        self._store.clear()


# ============================================================================
# Tag constants и helpers
# ============================================================================


def _load_canonical_tags() -> dict[str, dict[str, Any]]:
    """
    Загружает CANONICAL_TAGS из config/tag_map.json.
    Файл ищется относительно корня проекта (два уровня выше этого модуля).
    При отсутствии файла возвращает пустой словарь и логирует предупреждение.
    """
    tag_map_path = Path(__file__).parent.parent / "config" / "tag_map.json"
    if not tag_map_path.exists():
        logger.warning(
            "tag_map.json not found at %s, CANONICAL_TAGS will be empty. "
            "Tag-based retrieval will degrade to fallback.",
            tag_map_path,
        )
        return {}
    try:
        with open(tag_map_path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        # Файл есть, но невалидный JSON или недоступен для чтения.
        # Это не критично: CANONICAL_TAGS просто останется пустым.
        logger.error("Failed to load tag_map.json: %s", e)
        return {}


CANONICAL_TAGS: dict[str, dict[str, Any]] = _load_canonical_tags()

KB_TAGS_STRICT_VALIDATION: bool = False


def _normalize_tag_local(tag: str) -> str:
    """Локальная нормализация тега без импорта tag_registry (для bootstrap)."""
    return tag.lower().replace("-", "_").replace(" ", "_")


def _normalize_tags_local(tags: list[str]) -> list[str]:
    return [_normalize_tag_local(tag) for tag in tags if isinstance(tag, str)]


def _resolve_normalizers() -> (
    tuple[
        Callable[[str], str],
        Callable[[list[str]], list[str]],
    ]
):
    """Resolve (normalize_tag, normalize_tags) from tag_registry or use local fallbacks."""
    try:
        from src.tag_registry import normalize_tag, normalize_tags
    except ImportError:
        return _normalize_tag_local, _normalize_tags_local
    return normalize_tag, normalize_tags


def _build_known_tags_from_canonical() -> set[str]:
    """Строит множество всех canonical тегов."""
    tags: set[str] = set()

    for category_data in CANONICAL_TAGS.values():
        for tag_data in category_data.values():
            if isinstance(tag_data, dict):
                for tag_list in tag_data.values():
                    if isinstance(tag_list, list):
                        tags.update(
                            _normalize_tag_local(tag) for tag in tag_list if isinstance(tag, str)
                        )

    return tags


KNOWN_TAGS: set[str] = _build_known_tags_from_canonical()


def get_canonical_tags_for_category(category: str, value: str) -> list[str]:
    """Возвращает primary + expanded теги для категории/значения."""
    normalize_tag, normalize_tags = _resolve_normalizers()

    norm_value = normalize_tag(value)
    data = CANONICAL_TAGS.get(category, {}).get(norm_value)

    if isinstance(data, dict):
        return normalize_tags(data.get("primary", []) + data.get("expanded", []))
    if isinstance(data, list):
        return normalize_tags(data)
    return normalize_tags([norm_value])


def get_primary_tags_for_category(category: str, value: str) -> list[str]:
    """Возвращает primary теги."""
    normalize_tag, normalize_tags = _resolve_normalizers()

    norm_value = normalize_tag(value)
    data = CANONICAL_TAGS.get(category, {}).get(norm_value)

    if isinstance(data, dict):
        return normalize_tags(data.get("primary", []))
    if isinstance(data, list):
        return normalize_tags(data)
    return normalize_tags([norm_value])


def get_expanded_tags_for_category(category: str, value: str) -> list[str]:
    """Возвращает expanded теги."""
    normalize_tag, normalize_tags = _resolve_normalizers()

    norm_value = normalize_tag(value)
    data = CANONICAL_TAGS.get(category, {}).get(norm_value)

    if isinstance(data, dict):
        return normalize_tags(data.get("expanded", []))
    return []


# ============================================================================
# Explainability structures (moved to src.config_models.explainability)
# ============================================================================

from src.config_models.explainability import (  # noqa: F401
    AssemblyBlockDiagnostics,
    AssemblyTrace,
    FeatureResolutionResult,
)
