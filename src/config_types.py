"""
config_types.py

Dataclasses, enum'ы и инфраструктурные типы для конфигурирования PromptBuilder.

Содержит:
- Domain types — RuleEntry, KnowledgeBase, CoreConfig и т.д.
- LimitsConfig — лимиты выдачи и кандидатов (ТП-3)
- KnowledgeLevel — режим включения блоков знаний (ТП-1)
- KnowledgeBlockPlan — описание блока для budget-aware сборки (ТП-1)
- BlockBudget — бюджет одного блока KB (ТП-1)
- KnowledgeBudget — совокупный бюджет всех блоков (ТП-1)
- KnowledgeBudgetManager — вычисляет бюджет (ТП-1)
- CachePolicy — политика инвалидации кэша (ФП-1)
- FileCache — кэш-менеджер с поддержкой TTL/mtime (ФП-1)
- Tag constants — CANONICAL_TAGS, KNOWN_TAGS, get_*_tags_for_category
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    TypeVar,
    Union,
)

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict


logger = logging.getLogger(__name__)
V = TypeVar("V")


# ============================================================================
# Domain types — TypedDict и dataclass'ы для конфигов и базы знаний
# ============================================================================

class RuleEntry(TypedDict, total=False):
    """Запись с правилом исправления (грамматика, стиль, логика)."""

    wrong: str
    correct: str
    rule: str
    description: str
    tags: List[str]
    category: str


class StructuralEntry(TypedDict, total=False):
    """Структурная запись (фреймворк, шаблон, приём)."""

    name: str
    description: str
    when_to_use: Union[str, List[str]]
    rule: str
    steps: List[Dict[str, Any]]
    sections: List[Dict[str, Any]]
    tags: List[str]


class EditorialTechniqueEntry(TypedDict, total=False):
    """Редакторский приём."""

    id: str
    name: str
    category: str
    description: str
    when_to_use: List[str]
    how_to_apply: List[str]
    example_wrong: str
    example_correct: str
    example_explanation: str
    tags: List[str]
    source: Dict[str, Any]


FlatEntry = Dict[str, Any]


# ============================================================================
# Исправленные датаклассы для задач 4, 5 и 6
# ============================================================================

@dataclass(frozen=True)
class CoreConfig:
    """Базовая конфигурация редактора."""
    role: str
    priorities: str
    basic_audit_instructions: tuple   # заменено с List[str] на tuple
    forbidden: tuple                  # заменено с List[str] на tuple
    ip_ceiling: float = 2.5           # добавлено


@dataclass(frozen=True)
class DomainConfig:
    """Конфигурация домена."""
    name: str
    system_rules: str
    tone: str
    allow_storytelling: bool = True
    allow_marketing: bool = True
    tasks: tuple = field(default_factory=tuple)          # добавлено
    constraints: tuple = field(default_factory=tuple)    # добавлено
    ip_ceiling: Optional[float] = None                   # добавлено


@dataclass(frozen=True)
class IntentConfig:
    """Конфигурация цели обработки."""
    name: str
    instructions: List[str]    # пока оставляем List[str], при необходимости можно заменить на tuple


@dataclass(frozen=True)
class OverlayConfig:
    """Конфигурация оверлея."""
    name: str
    instructions: tuple                      # заменено с List[str] на tuple
    conflicts_with: tuple = field(default_factory=tuple)   # добавлено


@dataclass(frozen=True)
class AudienceProfile:
    """Профиль аудитории."""
    kind: str
    expertise: str
    formality: str
    description: str = ""


# ============================================================================
# ИЗМЕНЕНИЕ: KnowledgeBase теперь динамический контейнер
# ============================================================================

class KnowledgeBase:
    """
    Динамическая база знаний.

    Записи хранятся в _blocks по ключу — имени блока (например, "grammar_errors").
    Обратная совместимость: доступ к старым атрибутам (grammar_errors, stylistic_issues и т.д.)
    реализован через __getattr__, поэтому существующий код не ломается.

    Методы:
        get(key, default=None) — получить блок по ключу.
        register(key, data) — установить блок.
        keys() — список всех ключей.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Создаёт KnowledgeBase из именованных аргументов.
        Каждый аргумент становится блоком с соответствующим именем.
        """
        self._blocks: Dict[str, Any] = {}
        for key, value in kwargs.items():
            self._blocks[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Возвращает блок по ключу или default (по умолчанию пустой список)."""
        return self._blocks.get(key, default or [])

    def register(self, key: str, data: Any) -> None:
        """Регистрирует (перезаписывает) блок с именем key."""
        self._blocks[key] = data

    def keys(self) -> Set[str]:
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
# ТП-3: LimitsConfig — лимиты выдачи и кандидатов
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
    stop_words_category: int = 8
    stop_words_items: int = 5

    grammar_candidates: Optional[int] = None
    style_candidates: Optional[int] = None
    logic_candidates: Optional[int] = None
    storytelling_candidates: Optional[int] = None
    marketing_candidates: Optional[int] = None
    rhetoric_candidates: Optional[int] = None


# ============================================================================
# ТП-1: KnowledgeLevel — режим включения блоков знаний
# ============================================================================

class KnowledgeLevel(str, Enum):
    """
    Режим включения блоков базы знаний в промпт.

    NONE — база знаний не включается совсем.
    CORE — только обязательные блоки: grammar, style, stop_words.
    STANDARD — CORE + logic, composition, cohesion, composition_errors,
    nkrj, glossary.
    FULL — все доступные блоки, включая storytelling, marketing,
    rhetoric, editorial.
    """

    NONE = "none"
    CORE = "core"
    STANDARD = "standard"
    FULL = "full"


KNOWLEDGE_BUDGET_CHARS: Dict[KnowledgeLevel, int] = {
    KnowledgeLevel.NONE: 0,
    KnowledgeLevel.CORE: 4_000,
    KnowledgeLevel.STANDARD: 10_000,
    KnowledgeLevel.FULL: 16_000,
}

_LEVEL_BLOCKS: Dict[KnowledgeLevel, Set[str]] = {
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
    },
}


def blocks_allowed_at_level(level: KnowledgeLevel) -> Set[str]:
    """Возвращает множество имён блоков, разрешённых на данном уровне."""
    return _LEVEL_BLOCKS.get(level, set())


# ============================================================================
# ТП-1: KnowledgeBlockPlan
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
    builder: Optional[Callable[[], str]] = field(default=None, repr=False)
    enable_condition: bool = True


# ============================================================================
# ТП-1: BlockBudget, KnowledgeBudget, KnowledgeBudgetManager
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
    char_budget: Optional[int]
    enabled: bool = True


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
    )

    def __init__(self, budgets: Dict[str, BlockBudget]) -> None:
        self._budgets = budgets

    def get(self, block_name: str) -> Optional[BlockBudget]:
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

    def __init__(self, token_budget: Optional[int] = None) -> None:
        """
        Args:
            token_budget: Приблизительный лимит токенов под блок «База знаний».
            1 токен ≈ 4 символа (heuristic). None = без ограничений.
        """
        self._token_budget = token_budget
        self._char_budget: Optional[int] = (
            token_budget * 4 if token_budget is not None else None
        )

    def allocate(
        self,
        limits: LimitsConfig,
        active_blocks: Optional[Set[str]] = None,
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
            effective_active & allowed
            if level != KnowledgeLevel.FULL
            else effective_active
        )

        n_enabled = len(enabled_set) or 1
        per_block_chars: Optional[int] = (
            self._char_budget // n_enabled if self._char_budget is not None else None
        )

        def _blk(name: str, entry_limit: int) -> BlockBudget:
            is_enabled = level == KnowledgeLevel.FULL or name in allowed
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
                "composition_errors": _blk(
                    "composition_errors", limits.composition_errors
                ),
                "storytelling": _blk("storytelling", limits.storytelling),
                "marketing": _blk("marketing", limits.marketing),
                "rhetoric": _blk("rhetoric", limits.rhetoric),
                "editorial": _blk("editorial", limits.editorial),
                "glossary": _blk("glossary", limits.glossary),
                "stop_words": _blk("stop_words", limits.stop_words_category),
                "nkrj": _blk("nkrj", 0),
            }
        )


# ============================================================================
# ФП-1: CachePolicy и FileCache
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
    ttl_seconds: Optional[float] = None


@dataclass
class _CacheEntry(Generic[V]):
    """Внутренняя запись кэша."""

    value: V
    path: Optional[Path]
    loaded_at: float
    mtime_at_load: Optional[float]


class FileCache:
    """
    Кэш файловых данных с поддержкой TTL и mtime-инвалидации (ФП-1).

    Использование:
        cache = FileCache(policy=CachePolicy(check_mtime=True))
        data = cache.get_or_load("key", path, loader_fn, *loader_args)
    """

    def __init__(self, policy: Optional[CachePolicy] = None) -> None:
        self._policy = policy or CachePolicy(check_mtime=True)
        self._store: Dict[str, _CacheEntry[Any]] = {}

    def _is_valid(self, entry: _CacheEntry[Any]) -> bool:
        """Проверяет актуальность записи кэша."""
        now = time.monotonic()

        if self._policy.ttl_seconds is not None:
            if now - entry.loaded_at > self._policy.ttl_seconds:
                return False

        if self._policy.check_mtime and entry.path is not None:
            try:
                current_mtime = entry.path.stat().st_mtime
                if (
                    entry.mtime_at_load is None
                    or current_mtime != entry.mtime_at_load
                ):
                    return False
            except OSError:
                return False

        return True

    def get_or_load(
        self,
        key: str,
        path: Optional[Path],
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

        mtime = None
        if path is not None and self._policy.check_mtime:
            try:
                mtime = path.stat().st_mtime
            except OSError:
                pass

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
        paths: List[Path],
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

            if (
                entry is not None
                and self._policy.check_mtime
                and entry.mtime_at_load is not None
            ):
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
            max_mtime: Optional[float] = max(
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

def _load_canonical_tags() -> Dict[str, Dict[str, Any]]:
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
    except Exception as e:
        logger.error("Failed to load tag_map.json: %s", e)
        return {}

CANONICAL_TAGS: Dict[str, Dict[str, Any]] = _load_canonical_tags()

KB_TAGS_STRICT_VALIDATION: bool = False


def _normalize_tag_local(tag: str) -> str:
    """Локальная нормализация тега без импорта tag_registry (для bootstrap)."""
    return tag.lower().replace("-", "_").replace(" ", "_")


def _normalize_tags_local(tags: List[str]) -> List[str]:
    return [_normalize_tag_local(tag) for tag in tags if isinstance(tag, str)]


def _build_known_tags_from_canonical() -> Set[str]:
    """Строит множество всех canonical тегов."""
    tags: Set[str] = set()

    for category_data in CANONICAL_TAGS.values():
        for tag_data in category_data.values():
            if isinstance(tag_data, dict):
                for tag_list in tag_data.values():
                    if isinstance(tag_list, list):
                        tags.update(
                            _normalize_tag_local(tag)
                            for tag in tag_list
                            if isinstance(tag, str)
                        )

    return tags


KNOWN_TAGS: Set[str] = _build_known_tags_from_canonical()


def get_canonical_tags_for_category(category: str, value: str) -> List[str]:
    """Возвращает primary + expanded теги для категории/значения."""
    try:
        from src.tag_registry import normalize_tag, normalize_tags
    except ImportError:
        normalize_tag = _normalize_tag_local
        normalize_tags = _normalize_tags_local

    norm_value = normalize_tag(value)
    data = CANONICAL_TAGS.get(category, {}).get(norm_value)

    if isinstance(data, dict):
        return normalize_tags(data.get("primary", []) + data.get("expanded", []))
    if isinstance(data, list):
        return normalize_tags(data)
    return normalize_tags([norm_value])


def get_primary_tags_for_category(category: str, value: str) -> List[str]:
    """Возвращает primary теги."""
    try:
        from src.tag_registry import normalize_tag, normalize_tags
    except ImportError:
        normalize_tag = _normalize_tag_local
        normalize_tags = _normalize_tags_local

    norm_value = normalize_tag(value)
    data = CANONICAL_TAGS.get(category, {}).get(norm_value)

    if isinstance(data, dict):
        return normalize_tags(data.get("primary", []))
    if isinstance(data, list):
        return normalize_tags(data)
    return normalize_tags([norm_value])


def get_expanded_tags_for_category(category: str, value: str) -> List[str]:
    """Возвращает expanded теги."""
    try:
        from src.tag_registry import normalize_tag, normalize_tags
    except ImportError:
        normalize_tag = _normalize_tag_local
        normalize_tags = _normalize_tags_local

    norm_value = normalize_tag(value)
    data = CANONICAL_TAGS.get(category, {}).get(norm_value)

    if isinstance(data, dict):
        return normalize_tags(data.get("expanded", []))
    return []