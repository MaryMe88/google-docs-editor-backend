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
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# Re-exports для обратной совместимости (итерация 7 дорожной карты).
# Код проекта исторически ожидает эти имена из src.config_types.
from src.config_models.cache import (  # noqa: F401
    CachePolicy,
    FileCache,
)
from src.config_models.domain import (  # noqa: F401
    AudienceProfile,
    CoreConfig,
    DomainConfig,
    EditorialTechniqueEntry,
    FlatEntry,
    IntentConfig,
    KnowledgeBase,
    OverlayConfig,
    RuleEntry,
    StructuralEntry,
)
from src.config_models.explainability import (  # noqa: F401
    AssemblyBlockDiagnostics,
    AssemblyTrace,
    FeatureResolutionResult,
)
from src.config_models.limits import (
    LimitsConfig,
)
from src.reason_codes import ReasonCode  # noqa: F401

logger = logging.getLogger(__name__)

# ============================================================================
# Domain types (moved to src.config_models.domain)
# ============================================================================


# ============================================================================
# LimitsConfig — лимиты выдачи и кандидатов (moved to src.config_models.limits)
# ============================================================================


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
# CachePolicy и FileCache (moved to src.config_models.cache)
# ============================================================================


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
