"""
config_types.py

Типы данных для конфигурации PromptBuilder.
Вынесены в отдельный модуль, чтобы не засорять prompt_builder.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# KnowledgeBudgetManager — ТП-1: динамическое управление символьным бюджетом
# ---------------------------------------------------------------------------

# Символьный эквивалент одного токена GPT-4 для русского текста.
# Средняя длина русского слова ~5–6 символов, токен ≈ 4 символа.
_CHARS_PER_TOKEN: float = 4.0

# Приоритеты блоков (чем выше — тем позже урезается при нехватке бюджета).
# 1 = первым кандидат на урезание, 10 = последним.
_BLOCK_PRIORITY: Dict[str, int] = {
    "grammar":            9,
    "style":              9,
    "logic":              8,
    "stop_words":         7,
    "composition":        6,
    "cohesion":           6,
    "composition_errors": 6,
    "storytelling":       5,
    "marketing":          5,
    "rhetoric":           4,
    "editorial":          4,
    "nkrj":               3,
    "glossary":           2,
}

# Минимальное кол-во записей, ниже которого блок просто отключается.
_BLOCK_MIN_ENTRIES: Dict[str, int] = {
    "grammar":            1,
    "style":              1,
    "logic":              1,
    "stop_words":         1,
    "composition":        1,
    "cohesion":           1,
    "composition_errors": 1,
    "storytelling":       1,
    "marketing":          1,
    "rhetoric":           1,
    "editorial":          1,
    "nkrj":               0,  # нет «записей», блок целиком включён/выключен
    "glossary":           1,
}


@dataclass
class BlockBudget:
    """
    Результат распределения бюджета для одного блока KB.

    Attributes:
        char_budget: Мягкий символьный лимит для _select_ranked_entries().
                     None означает «без лимита» (бюджет не задан).
        entry_limit: Максимальное число записей для этого блока.
        enabled:     False — блок пропускается целиком (бюджет исчерпан).
    """
    char_budget: Optional[int]
    entry_limit: int
    enabled: bool = True


@dataclass
class KnowledgeBudget:
    """
    Полное распределение бюджета по всем блокам KB для одного запроса.
    Создаётся методом KnowledgeBudgetManager.allocate().
    """
    grammar: BlockBudget
    style: BlockBudget
    logic: BlockBudget
    stop_words: BlockBudget
    composition: BlockBudget
    cohesion: BlockBudget
    composition_errors: BlockBudget
    storytelling: BlockBudget
    marketing: BlockBudget
    rhetoric: BlockBudget
    editorial: BlockBudget
    nkrj: BlockBudget
    glossary: BlockBudget

    def get(self, block: str) -> BlockBudget:
        """Возвращает BlockBudget по имени блока. KeyError если блок неизвестен."""
        try:
            return getattr(self, block)
        except AttributeError:
            raise KeyError(f"Unknown block: {block!r}")


class KnowledgeBudgetManager:
    """
    Динамически распределяет символьный бюджет между блоками KB (ТП-1).

    Принцип работы:
    1. Получает `token_budget` — максимальное число токенов, отведённых под
       весь блок «База знаний» в промпте.
    2. Переводит токены → символы (×_CHARS_PER_TOKEN).
    3. Делит символьный бюджет между активными блоками пропорционально
       `share_weights`, но с гарантиями минимума для высокоприоритетных блоков.
    4. Возвращает `KnowledgeBudget` — датаклас с `BlockBudget` на каждый блок.

    Если `token_budget` не задан (None), распределение не применяется:
    все блоки получают `char_budget=None` (без ограничений) — поведение
    полностью совпадает с предыдущей версией кода.

    Пример:
        manager = KnowledgeBudgetManager(token_budget=2000)
        budget = manager.allocate(limits, active_blocks={"grammar", "style", "logic"})
        # budget.grammar.char_budget → ~3200 символов
        # budget.nkrj.enabled        → False (блок не запрошен)
    """

    # Относительные веса блоков при делении бюджета.
    # Сумма весов активных блоков = 100%, каждый блок получает свою долю.
    DEFAULT_SHARE_WEIGHTS: Dict[str, float] = {
        "grammar":            1.5,
        "style":              1.5,
        "logic":              1.2,
        "stop_words":         0.8,
        "composition":        1.0,
        "cohesion":           1.0,
        "composition_errors": 1.0,
        "storytelling":       1.0,
        "marketing":          1.0,
        "rhetoric":           0.8,
        "editorial":          1.0,
        "nkrj":               0.6,
        "glossary":           0.6,
    }

    # Минимальная доля токенов, резервируемая для грамматики+стиля+логики вместе.
    _CORE_MIN_SHARE: float = 0.40

    def __init__(
        self,
        token_budget: Optional[int] = None,
        chars_per_token: float = _CHARS_PER_TOKEN,
        share_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Args:
            token_budget:   Токены под блок KB. None = без ограничений.
            chars_per_token: Коэффициент токен→символ (по умолчанию 4.0).
            share_weights:  Переопределение весов для отдельных блоков.
                            Передавай только блоки, которые хочешь изменить —
                            остальные берут DEFAULT_SHARE_WEIGHTS.
        """
        self.token_budget = token_budget
        self.chars_per_token = chars_per_token
        self._weights: Dict[str, float] = dict(self.DEFAULT_SHARE_WEIGHTS)
        if share_weights:
            self._weights.update(share_weights)

    @property
    def char_budget_total(self) -> Optional[int]:
        """Общий символьный бюджет (None если token_budget не задан)."""
        if self.token_budget is None:
            return None
        return int(self.token_budget * self.chars_per_token)

    def allocate(
        self,
        limits: LimitsConfig,
        active_blocks: Optional[set] = None,
    ) -> KnowledgeBudget:
        """
        Распределяет бюджет по блокам и возвращает KnowledgeBudget.

        Args:
            limits:        LimitsConfig с entry_limit по каждому блоку.
            active_blocks: Набор имён блоков, которые реально будут
                           использованы в этом запросе (None = все блоки).
                           Блоки не из этого набора получают enabled=False.

        Returns:
            KnowledgeBudget с заполненными BlockBudget для каждого блока.
        """
        all_blocks = list(_BLOCK_PRIORITY.keys())

        if active_blocks is None:
            active_blocks = set(all_blocks)

        char_total = self.char_budget_total

        # Если бюджет не задан — возвращаем «без ограничений» для всех активных блоков
        if char_total is None:
            return self._unlimited_budget(limits, active_blocks, all_blocks)

        # --- Шаг 1: вычисляем веса только активных блоков ---
        active_weights = {
            b: self._weights.get(b, 1.0)
            for b in all_blocks
            if b in active_blocks
        }
        total_weight = sum(active_weights.values()) or 1.0

        # --- Шаг 2: резервируем минимум для core-блоков ---
        core_blocks = {"grammar", "style", "logic"}
        core_active = core_blocks & active_blocks
        core_weight = sum(active_weights.get(b, 0.0) for b in core_active)
        core_natural_share = core_weight / total_weight

        if core_natural_share < self._CORE_MIN_SHARE and core_active:
            # Доплачиваем core-блокам до минимума за счёт остальных
            boost_factor = self._CORE_MIN_SHARE / max(core_natural_share, 1e-9)
            for b in core_active:
                active_weights[b] = active_weights[b] * boost_factor
            total_weight = sum(active_weights.values()) or 1.0

        # --- Шаг 3: раздаём символьные бюджеты ---
        block_chars: Dict[str, int] = {}
        for block, weight in active_weights.items():
            share = weight / total_weight
            block_chars[block] = max(256, int(char_total * share))

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "KnowledgeBudgetManager: total_chars=%d, active=%s, allocated=%s",
                char_total,
                sorted(active_blocks),
                {k: v for k, v in block_chars.items()},
            )

        # --- Шаг 4: собираем KnowledgeBudget ---
        def _make(block: str, entry_limit: int) -> BlockBudget:
            if block not in active_blocks:
                return BlockBudget(char_budget=None, entry_limit=entry_limit, enabled=False)
            return BlockBudget(
                char_budget=block_chars.get(block),
                entry_limit=entry_limit,
                enabled=True,
            )

        return KnowledgeBudget(
            grammar=_make("grammar", limits.grammar),
            style=_make("style", limits.style),
            logic=_make("logic", limits.logic),
            stop_words=_make("stop_words", limits.stop_words_category),
            composition=_make("composition", limits.composition),
            cohesion=_make("cohesion", limits.cohesion),
            composition_errors=_make("composition_errors", limits.composition_errors),
            storytelling=_make("storytelling", limits.storytelling),
            marketing=_make("marketing", limits.marketing),
            rhetoric=_make("rhetoric", limits.rhetoric),
            editorial=_make("editorial", limits.editorial),
            nkrj=_make("nkrj", 0),
            glossary=_make("glossary", limits.glossary),
        )

    def _unlimited_budget(
        self,
        limits: LimitsConfig,
        active_blocks: set,
        all_blocks: List[str],
    ) -> KnowledgeBudget:
        """Создаёт KnowledgeBudget без символьных ограничений."""
        def _make(block: str, entry_limit: int) -> BlockBudget:
            return BlockBudget(
                char_budget=None,
                entry_limit=entry_limit,
                enabled=(block in active_blocks),
            )

        return KnowledgeBudget(
            grammar=_make("grammar", limits.grammar),
            style=_make("style", limits.style),
            logic=_make("logic", limits.logic),
            stop_words=_make("stop_words", limits.stop_words_category),
            composition=_make("composition", limits.composition),
            cohesion=_make("cohesion", limits.cohesion),
            composition_errors=_make("composition_errors", limits.composition_errors),
            storytelling=_make("storytelling", limits.storytelling),
            marketing=_make("marketing", limits.marketing),
            rhetoric=_make("rhetoric", limits.rhetoric),
            editorial=_make("editorial", limits.editorial),
            nkrj=_make("nkrj", 0),
            glossary=_make("glossary", limits.glossary),
        )


# ---------------------------------------------------------------------------
# CachePolicy — ФП-1: mtime/TTL-инвалидация кэша
# ---------------------------------------------------------------------------

import time


@dataclass
class CachePolicy:
    """
    Политика инвалидации in-memory кэша PromptBuilder.

    Поддерживает два независимых механизма, которые можно комбинировать:

    1. **TTL** (time-to-live, секунды):
       Запись считается устаревшей, если прошло больше `ttl_seconds` с
       момента последней загрузки. Полезно для hot-reload в dev-среде
       без полной перезагрузки сервиса.
       - `ttl_seconds=None` (по умолчанию) → TTL отключён.
       - `ttl_seconds=300` → кэш живёт 5 минут.

    2. **mtime** (filesystem modification time):
       Запись инвалидируется, если файл на диске изменился с момента
       последней загрузки (сравниваем os.stat().st_mtime).
       - `check_mtime=True` (по умолчанию) → включено для всех кэшируемых файлов.
       - `check_mtime=False` → отключить (ускоряет hot-path, но без auto-reload).

    Примеры:
        # Только TTL, каждые 60 секунд
        policy = CachePolicy(ttl_seconds=60, check_mtime=False)

        # Только mtime (дефолт — без накладных расходов по времени)
        policy = CachePolicy(check_mtime=True)

        # TTL + mtime — инвалидируется при любом из условий
        policy = CachePolicy(ttl_seconds=300, check_mtime=True)

        # Полностью отключить инвалидацию (поведение до ФП-1)
        policy = CachePolicy(ttl_seconds=None, check_mtime=False)
    """

    ttl_seconds: Optional[float] = None
    check_mtime: bool = True


@dataclass
class _CacheEntry:
    """Внутренняя обёртка для одного закэшированного значения."""

    value: Any
    loaded_at: float             # time.monotonic() в момент загрузки
    file_mtime: Optional[float]  # os.stat().st_mtime файла при загрузке


class FileCache:
    """
    Generic кэш с mtime/TTL инвалидацией для одного или нескольких файлов.

    Используется внутри PromptBuilder для каждого типа конфигов и KB.

    Пример:
        cache = FileCache(policy=CachePolicy(ttl_seconds=60))
        value = cache.get_or_load("marketing", path, loader_func)
    """

    def __init__(self, policy: CachePolicy) -> None:
        self._policy = policy
        self._entries: Dict[str, _CacheEntry] = {}

    def get_or_load(
        self,
        key: str,
        path: Path,
        loader: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Возвращает закэшированное значение или перезагружает его.

        Args:
            key:    Уникальный ключ записи (например, имя домена или "core").
            path:   Основной файл, чей mtime проверяется. Для KB это может
                    быть директория — тогда mtime директории не проверяется
                    (используй multi-file вариант `get_or_load_multi`).
            loader: Callable без аргументов (или с *args/**kwargs), который
                    возвращает актуальное значение.
        """
        entry = self._entries.get(key)
        if entry is not None and not self._is_stale(entry, path):
            return entry.value

        value = loader(*args, **kwargs)
        mtime = self._read_mtime(path)
        self._entries[key] = _CacheEntry(
            value=value,
            loaded_at=time.monotonic(),
            file_mtime=mtime,
        )
        logger.debug("FileCache: loaded '%s' (mtime=%s)", key, mtime)
        return value

    def get_or_load_multi(
        self,
        key: str,
        paths: List[Path],
        loader: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Вариант для нескольких файлов — инвалидирует, если изменился
        хотя бы один. Сравниваем по максимальному mtime среди всех путей.
        """
        entry = self._entries.get(key)
        max_mtime = self._max_mtime(paths)
        if entry is not None and not self._is_stale_multi(entry, max_mtime):
            return entry.value

        value = loader(*args, **kwargs)
        self._entries[key] = _CacheEntry(
            value=value,
            loaded_at=time.monotonic(),
            file_mtime=max_mtime,
        )
        logger.debug("FileCache: loaded '%s' (max_mtime=%s)", key, max_mtime)
        return value

    def invalidate(self, key: Optional[str] = None) -> None:
        """Сбрасывает одну запись (key) или весь кэш (key=None)."""
        if key is None:
            self._entries.clear()
            logger.debug("FileCache: full invalidation")
        else:
            self._entries.pop(key, None)
            logger.debug("FileCache: invalidated '%s'", key)

    # --- Internal helpers ---

    def _is_stale(self, entry: _CacheEntry, path: Path) -> bool:
        """Возвращает True если запись устарела."""
        if self._ttl_expired(entry):
            return True
        if self._policy.check_mtime:
            current_mtime = self._read_mtime(path)
            if current_mtime != entry.file_mtime:
                return True
        return False

    def _is_stale_multi(self, entry: _CacheEntry, max_mtime: Optional[float]) -> bool:
        if self._ttl_expired(entry):
            return True
        if self._policy.check_mtime and max_mtime != entry.file_mtime:
            return True
        return False

    def _ttl_expired(self, entry: _CacheEntry) -> bool:
        if self._policy.ttl_seconds is None:
            return False
        age = time.monotonic() - entry.loaded_at
        return age > self._policy.ttl_seconds

    @staticmethod
    def _read_mtime(path: Path) -> Optional[float]:
        try:
            return path.stat().st_mtime
        except OSError:
            return None

    @staticmethod
    def _max_mtime(paths: List[Path]) -> Optional[float]:
        mtimes = []
        for p in paths:
            try:
                mtimes.append(p.stat().st_mtime)
            except OSError:
                pass
        return max(mtimes) if mtimes else None
