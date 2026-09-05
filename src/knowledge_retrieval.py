"""
knowledge_retrieval.py

Логика нормализации текста, скоринга и ранжирования записей базы знаний.
Отдельный модуль, чтобы разгрузить prompt_builder.py.
Поддерживает hybrid search: keyword-результаты дополняются
семантическим re-rankingом через SemanticIndex.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union, overload

from src.tag_registry import normalize_tag
from src.scoring_weights import get_scoring_weight

logger = logging.getLogger(__name__)


class FallbackStage(str, Enum):
    STRONG = "strong"
    TEXT_ONLY = "text_only"
    TAG_ONLY = "tag_only"
    NEUTRAL = "neutral"
    EMPTY = "empty"


@dataclass(frozen=True)
class FallbackPolicy:
    min_strong_score: int = 1
    allow_text_only: bool = True
    allow_tag_only: bool = True
    allow_neutral_fallback: bool = False
    neutral_tags: Tuple[str, ...] = ("neutral", "editing", "clarity")
    primary_only_for_tag_fallback: bool = True
    min_info_score_for_neutral: int = 1


RULE_FALLBACK_POLICY = FallbackPolicy(
    min_strong_score=1,
    allow_text_only=True,
    allow_tag_only=True,
    allow_neutral_fallback=False,
    primary_only_for_tag_fallback=True,
    min_info_score_for_neutral=2,
)

STRUCTURAL_FALLBACK_POLICY = FallbackPolicy(
    min_strong_score=1,
    allow_text_only=True,
    allow_tag_only=True,
    allow_neutral_fallback=True,
    neutral_tags=("neutral", "editing", "clarity"),
    primary_only_for_tag_fallback=True,
    min_info_score_for_neutral=2,
)


def normalize_text_for_match(text: str) -> str:
    """
    Приводит текст к нижнему регистру, заменяет 'ё' на 'е',
    оставляет только буквы, цифры и пробелы, схлопывает пробелы.
    """
    text = text.replace("ё", "е").replace("Ё", "Е")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def _contains_pattern(normalized_text: str, pattern: str) -> bool:
    """
    Проверяет, содержится ли паттерн в нормализованном тексте.
    Для однословных паттернов использует границы слова.
    """
    if not pattern:
        return False

    norm_pattern = normalize_text_for_match(pattern)
    if not norm_pattern or len(norm_pattern) < 2:
        return False

    if " " not in norm_pattern:
        return re.search(rf"\b{re.escape(norm_pattern)}\b", normalized_text) is not None

    return norm_pattern in normalized_text


def _get_entry_match_patterns(entry: Dict[str, Any]) -> List[str]:
    """
    Собирает кандидаты для текстового матчинга из полей:
    wrong, name, rule, description.
    """
    patterns: List[str] = []
    seen: Set[str] = set()

    for field in ("wrong", "name", "rule", "description"):
        value = entry.get(field)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped and stripped not in seen:
                seen.add(stripped)
                patterns.append(stripped)

    return patterns


def _entry_info_score(entry: Dict[str, Any]) -> int:
    """Количество информативных полей в записи."""
    score = 0

    for field in ("name", "description", "rule", "wrong", "when_to_use"):
        value = entry.get(field)
        if isinstance(value, str) and value.strip():
            score += 1
        elif isinstance(value, list) and value:
            score += 1

    for container_key in ("steps", "sections"):
        container = entry.get(container_key)
        if isinstance(container, list) and container:
            score += 1

    return score


def _estimate_entry_chars(entry: Dict[str, Any]) -> int:
    """Быстрая оценка размера записи в символах для char_budget."""
    total = 0

    for field in (
        "wrong",
        "correct",
        "rule",
        "description",
        "name",
        "example_wrong",
        "example_correct",
        "example_explanation",
    ):
        value = entry.get(field)
        if isinstance(value, str):
            total += len(value)

    for container_key in ("steps", "sections"):
        container = entry.get(container_key)
        if isinstance(container, list):
            for step in container:
                if not isinstance(step, dict):
                    continue
                for field in ("name", "description"):
                    value = step.get(field)
                    if isinstance(value, str):
                        total += len(value)

    return total


def score_rule_entry(
    entry: Dict[str, Any],
    normalized_text: str,
    wanted_tags: Set[str],
    idx: int,
    expanded_tags: Optional[Set[str]] = None,
) -> Tuple[int, int]:
    """Скоринг для grammar/style/logic."""
    score = 0

    wrong_val = entry.get("wrong", "")
    if isinstance(wrong_val, str):
        wrong_stripped = wrong_val.strip()
        if wrong_stripped and _contains_pattern(normalized_text, wrong_stripped):
            score += get_scoring_weight("wrong_exact_match")

    if score == 0:
        for field in ("name", "rule", "description"):
            value = entry.get(field)
            if not isinstance(value, str):
                continue

            stripped = value.strip()
            if stripped and _contains_pattern(normalized_text, stripped):
                score += get_scoring_weight("partial_text_match")
                break

    entry_tags = entry.get("tags", [])
    if not isinstance(entry_tags, (list, tuple)):
        entry_tags = []

    tag_set = {normalize_tag(tag) for tag in entry_tags if isinstance(tag, str)}
    overlap = len(tag_set & wanted_tags)
    score += overlap * get_scoring_weight("tag_primary")

    if overlap > 0:
        score += get_scoring_weight("tag_primary_bonus")

    if expanded_tags:
        score += len(tag_set & expanded_tags) * get_scoring_weight("tag_expanded")

    return score, -idx


def score_structural_entry(
    entry: Dict[str, Any],
    normalized_text: str,
    wanted_tags: Set[str],
    idx: int,
    expanded_tags: Optional[Set[str]] = None,
) -> Tuple[int, int]:
    """Скоринг для структурных записей."""
    score = 0
    patterns: List[str] = []

    def add_field(field: str) -> None:
        value = entry.get(field)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                patterns.append(stripped)

    add_field("name")
    add_field("description")
    add_field("rule")

    when_to_use = entry.get("when_to_use")
    if isinstance(when_to_use, str):
        stripped = when_to_use.strip()
        if stripped:
            patterns.append(stripped)
    elif isinstance(when_to_use, list):
        for item in when_to_use:
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    patterns.append(stripped)

    for container_key in ("steps", "sections"):
        container = entry.get(container_key)
        if isinstance(container, list):
            for step in container:
                if not isinstance(step, dict):
                    continue
                for field in ("name", "description"):
                    value = step.get(field)
                    if isinstance(value, str):
                        stripped = value.strip()
                        if stripped:
                            patterns.append(stripped)

    unique_patterns: List[str] = []
    seen: Set[str] = set()
    for pattern in patterns:
        if pattern not in seen:
            seen.add(pattern)
            unique_patterns.append(pattern)

    name_val = entry.get("name", "")
    name_stripped = name_val.strip() if isinstance(name_val, str) else ""

    for pattern in unique_patterns:
        if not _contains_pattern(normalized_text, pattern):
            continue

        if name_stripped and pattern == name_stripped:
            score += get_scoring_weight("name_exact_match")
        else:
            score += get_scoring_weight("partial_text_match")
        break

    entry_tags = entry.get("tags", [])
    if not isinstance(entry_tags, (list, tuple)):
        entry_tags = []

    tag_set = {normalize_tag(tag) for tag in entry_tags if isinstance(tag, str)}
    overlap = len(tag_set & wanted_tags)
    score += overlap * get_scoring_weight("tag_primary")

    if overlap > 0:
        score += get_scoring_weight("tag_primary_bonus")

    if expanded_tags:
        score += len(tag_set & expanded_tags) * get_scoring_weight("tag_expanded")

    return score, -idx


_score_entry = score_rule_entry


# ============================================================================
# BUG-6: расширенная дедупликация для структурных записей
# ============================================================================
def _make_dedupe_key(entry: Dict[str, Any]) -> Tuple[Any, ...]:
    if "id" in entry:
        return ("id", entry["id"])

    # Для записей без id учитываем и контейнерные поля,
    # чтобы не схлопывать разные структурные записи.
    def _container_signature(key: str) -> Tuple[Any, ...]:
        container = entry.get(key)
        if not isinstance(container, list):
            return ()
        parts: List[str] = []
        for item in container:
            if isinstance(item, dict):
                parts.append(str(item.get("name", "")) + "|" + str(item.get("description", "")))
        return tuple(parts)

    return (
        entry.get("wrong", ""),
        entry.get("rule", ""),
        entry.get("description", ""),
        entry.get("name", ""),
        _container_signature("steps"),
        _container_signature("sections"),
    )


def _normalize_tag_set(tags: Iterable[str]) -> Set[str]:
    return {normalize_tag(tag) for tag in tags if isinstance(tag, str)}


def _get_entry_tag_set(entry: Dict[str, Any]) -> Set[str]:
    raw_tags = entry.get("tags", [])
    if not isinstance(raw_tags, (list, tuple)):
        return set()
    return {normalize_tag(tag) for tag in raw_tags if isinstance(tag, str)}


def _get_text_match_strength(entry: Dict[str, Any], normalized_text: str) -> int:
    if not normalized_text:
        return 0

    patterns = _get_entry_match_patterns(entry)
    if not patterns:
        return 0

    first = patterns[0]
    if _contains_pattern(normalized_text, first):
        return get_scoring_weight("wrong_exact_match")

    for pattern in patterns[1:]:
        if _contains_pattern(normalized_text, pattern):
            return get_scoring_weight("partial_text_match")

    return 0


def _has_text_match(entry: Dict[str, Any], normalized_text: str) -> bool:
    return _get_text_match_strength(entry, normalized_text) > 0


def _get_primary_overlap(entry: Dict[str, Any], wanted_tags: Set[str]) -> int:
    if not wanted_tags:
        return 0
    return len(_get_entry_tag_set(entry) & wanted_tags)


def _get_any_overlap(
    entry: Dict[str, Any],
    wanted_tags: Set[str],
    expanded_tags: Optional[Set[str]],
) -> int:
    tag_set = _get_entry_tag_set(entry)
    overlap = len(tag_set & wanted_tags)
    if expanded_tags:
        overlap += len(tag_set & expanded_tags)
    return overlap


def _is_neutral_candidate(entry: Dict[str, Any], policy: FallbackPolicy) -> bool:
    tag_set = _get_entry_tag_set(entry)
    if not tag_set:
        return False

    if not (tag_set & set(policy.neutral_tags)):
        return False

    return _entry_info_score(entry) >= policy.min_info_score_for_neutral


def _collect_with_budget(
    ranked_entries: List[Dict[str, Any]],
    limit: int,
    char_budget: Optional[int],
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Выбирает записи с учётом лимита и символьного бюджета.
    Возвращает (выбранные_записи, количество_отброшенных_из-за_бюджета).
    """
    result: List[Dict[str, Any]] = []
    seen_keys: Set[Tuple[Any, ...]] = set()
    chars_used = 0
    dropped = 0

    for idx, entry in enumerate(ranked_entries):
        key = _make_dedupe_key(entry)
        if key in seen_keys:
            continue

        entry_chars = _estimate_entry_chars(entry)
        if char_budget is not None and chars_used + entry_chars > char_budget:
            dropped += 1
            continue

        seen_keys.add(key)
        result.append(entry)
        chars_used += entry_chars

        if len(result) >= limit:
            break

    return result, dropped


def _log_stage_debug(
    debug_context: str,
    stage: FallbackStage,
    candidates: List[Dict[str, Any]],
    selected: List[Dict[str, Any]],
) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return

    preview = [
        entry.get("name", entry.get("wrong", "?"))[:40]
        for entry in selected[:5]
    ]
    logger.debug(
        "[%s] stage=%s candidates=%s selected=%s preview=%s",
        debug_context,
        stage.value,
        len(candidates),
        len(selected),
        preview,
    )


def _sort_ranked(
    scored: List[Tuple[int, int, Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [entry for _, _, entry in scored]


def _ensure_return_type(
    result: Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FallbackStage, int]],
    return_meta: bool,
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FallbackStage, int]]:
    """
    Проверяет, что тип результата соответствует ожидаемому в зависимости от return_meta.
    """
    if return_meta:
        if not isinstance(result, tuple) or len(result) != 3:
            raise TypeError(
                f"_select_ranked_entries с return_meta=True должен возвращать tuple из 3 элементов "
                f"(entries, stage, dropped), но получил {type(result).__name__}: {result!r}"
            )
    else:
        if not isinstance(result, list):
            raise TypeError(
                f"_select_ranked_entries с return_meta=False должен возвращать list, "
                f"но получил {type(result).__name__}: {result!r}"
            )
    return result


# ---------------------------------------------------------------------------
# Semantic re-ranking (hybrid search)
# ---------------------------------------------------------------------------

def _semantic_rerank(
    entries: List[Dict[str, Any]],
    query: str,
    semantic_weight: float = 0.35,
    top_k_factor: int = 3,
) -> List[Dict[str, Any]]:
    """
    Дополняет keyword-ранжированные записи семантическим re-rankingом.

    Алгоритм:
    1. Берём keyword-результат (entries) как есть.
    2. Через SemanticIndex получаем семантический скор для каждой записи.
    3. Смешиваем: позиция в keyword-списке (1 - безразмерная, 0 - первая)
       + semantic_score * semantic_weight.
    4. Если индекс недоступен — возвращаем entries без изменений.

    :param entries: записи, уже отфильтрованные keyword-поиском
    :param query: оригинальный текст пользователя
    :param semantic_weight: доля семантики (0.0 — отключено, 1.0 — только семантика)
    :param top_k_factor: сколько кандидатов запрашивать у индекса (множитель len(entries))
    """
    if not entries or not query or not query.strip() or semantic_weight <= 0:
        return entries

    # Ленивая инициализация индекса при первом запросе с включённым реранкингом
    try:
        from src.semantic_index import get_semantic_index, init_semantic_index, _entries_for_index
        index = get_semantic_index()
        if index is None:
            if _entries_for_index:
                logger.info("SemanticIndex: ленивая инициализация индекса по первому запросу")
                init_semantic_index(_entries_for_index)
                index = get_semantic_index()
            else:
                logger.warning("SemanticIndex не инициализирован: нет записей для индексации")
                return entries
        if index is None or not index.is_ready():
            return entries
    except ImportError:
        return entries

    n = len(entries)
    top_k = min(n * top_k_factor, 200)
    semantic_results = index.search(query.strip(), top_k=top_k)

    # Строим карту entry_id → semantic_score
    sem_score_map: Dict[int, float] = {
        id(entry): score for entry, score in semantic_results
    }

    # Нормализуем позицию keyword-списка в [0, 1] (для первой = 1.0, для последней = 0.0)
    def keyword_rank_score(pos: int) -> float:
        return 1.0 - (pos / n) if n > 1 else 1.0

    combined: List[Tuple[float, int, Dict[str, Any]]] = []
    for pos, entry in enumerate(entries):
        kw_score = keyword_rank_score(pos)
        sem_score = sem_score_map.get(id(entry), 0.0)
        total = kw_score * (1.0 - semantic_weight) + sem_score * semantic_weight
        combined.append((total, pos, entry))

    combined.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    reranked = [entry for _, _, entry in combined]

    if logger.isEnabledFor(logging.DEBUG):
        before = [e.get("name", e.get("wrong", "?"))[:30] for e in entries[:3]]
        after = [e.get("name", e.get("wrong", "?"))[:30] for e in reranked[:3]]
        if before != after:
            logger.debug("Семантический re-ranking изменил порядок. До: %s. После: %s", before, after)

    return reranked


# ---------------------------------------------------------------------------
# Основной pipeline отбора
# ---------------------------------------------------------------------------

@overload
def _select_ranked_entries(
    entries: List[Dict[str, Any]],
    normalized_text: str,
    wanted_tags: Iterable[str],
    limit: int,
    require_text_match: bool = False,
    scorer=score_rule_entry,
    candidate_limit: Optional[int] = None,
    debug_context: str = "",
    expanded_tags: Optional[Set[str]] = None,
    min_score: Optional[int] = None,
    char_budget: Optional[int] = None,
    fallback_policy: Optional[FallbackPolicy] = None,
    return_meta: bool = False,
) -> List[Dict[str, Any]]:
    ...


@overload
def _select_ranked_entries(
    entries: List[Dict[str, Any]],
    normalized_text: str,
    wanted_tags: Iterable[str],
    limit: int,
    require_text_match: bool = False,
    scorer=score_rule_entry,
    candidate_limit: Optional[int] = None,
    debug_context: str = "",
    expanded_tags: Optional[Set[str]] = None,
    min_score: Optional[int] = None,
    char_budget: Optional[int] = None,
    fallback_policy: Optional[FallbackPolicy] = None,
    return_meta: bool = True,
) -> Tuple[List[Dict[str, Any]], FallbackStage, int]:
    ...


def _select_ranked_entries(
    entries: List[Dict[str, Any]],
    normalized_text: str,
    wanted_tags: Iterable[str],
    limit: int,
    require_text_match: bool = False,
    scorer=score_rule_entry,
    candidate_limit: Optional[int] = None,
    debug_context: str = "",
    expanded_tags: Optional[Set[str]] = None,
    min_score: Optional[int] = None,
    char_budget: Optional[int] = None,
    fallback_policy: Optional[FallbackPolicy] = None,
    return_meta: bool = False,
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FallbackStage, int]]:
    """
    Общая функция ранжирования записей с quality-gated fallback.
    Стадии:
    1. strong
    2. text_only
    3. tag_only
    4. neutral
    5. empty

    Если return_meta=True, возвращает (entries, stage, dropped).
    """
    if not entries or limit <= 0:
        stage = FallbackStage.EMPTY
        result = ([], stage, 0) if return_meta else []
        return _ensure_return_type(result, return_meta)

    policy = fallback_policy or RULE_FALLBACK_POLICY
    candidates = entries if candidate_limit is None else entries[:candidate_limit]
    wanted_set = _normalize_tag_set(wanted_tags)
    effective_min_score = (
        policy.min_strong_score if min_score is None else max(min_score, policy.min_strong_score)
    )

    scored: List[Tuple[int, int, Dict[str, Any]]] = []
    for idx, entry in enumerate(candidates):
        score, tie = scorer(
            entry,
            normalized_text,
            wanted_set,
            idx,
            expanded_tags=expanded_tags,
        )

        if require_text_match and not _has_text_match(entry, normalized_text):
            continue

        if score >= effective_min_score:
            scored.append((score, tie, entry))

    if scored:
        ranked = _sort_ranked(scored)
        result, dropped = _collect_with_budget(ranked, limit, char_budget)
        _log_stage_debug(debug_context, FallbackStage.STRONG, candidates, result)
        if dropped:
            logger.info(
                "[%s] char_budget truncated %d records (stage=%s)",
                debug_context, dropped, FallbackStage.STRONG.value,
            )
        if return_meta:
            res = (result, FallbackStage.STRONG, dropped)
        else:
            res = result
        return _ensure_return_type(res, return_meta)

    if policy.allow_text_only:
        text_only_scored: List[Tuple[int, int, Dict[str, Any]]] = []
        for idx, entry in enumerate(candidates):
            text_strength = _get_text_match_strength(entry, normalized_text)
            if text_strength <= 0:
                continue

            info = _entry_info_score(entry)
            text_only_scored.append((text_strength, info - idx, entry))

        if text_only_scored:
            text_only_scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
            ranked = [entry for _, _, entry in text_only_scored]
            result, dropped = _collect_with_budget(ranked, limit, char_budget)
            _log_stage_debug(debug_context, FallbackStage.TEXT_ONLY, candidates, result)
            if dropped:
                logger.info(
                    "[%s] char_budget truncated %d records (stage=%s)",
                    debug_context, dropped, FallbackStage.TEXT_ONLY.value,
                )
            if return_meta:
                res = (result, FallbackStage.TEXT_ONLY, dropped)
            else:
                res = result
            return _ensure_return_type(res, return_meta)

    if policy.allow_tag_only:
        tag_only_scored: List[Tuple[int, int, int, Dict[str, Any]]] = []
        for idx, entry in enumerate(candidates):
            if policy.primary_only_for_tag_fallback:
                overlap = _get_primary_overlap(entry, wanted_set)
            else:
                overlap = _get_any_overlap(entry, wanted_set, expanded_tags)

            if overlap <= 0:
                continue

            info = _entry_info_score(entry)
            tag_only_scored.append((overlap, info, -idx, entry))

        if tag_only_scored:
            tag_only_scored.sort(
                key=lambda item: (item[0], item[1], item[2]),
                reverse=True,
            )
            ranked = [entry for _, _, _, entry in tag_only_scored]
            result, dropped = _collect_with_budget(ranked, limit, char_budget)
            _log_stage_debug(debug_context, FallbackStage.TAG_ONLY, candidates, result)
            if dropped:
                logger.info(
                    "[%s] char_budget truncated %d records (stage=%s)",
                    debug_context, dropped, FallbackStage.TAG_ONLY.value,
                )
            if return_meta:
                res = (result, FallbackStage.TAG_ONLY, dropped)
            else:
                res = result
            return _ensure_return_type(res, return_meta)

    if policy.allow_neutral_fallback:
        neutral_scored: List[Tuple[int, int, Dict[str, Any]]] = []
        for idx, entry in enumerate(candidates):
            if not _is_neutral_candidate(entry, policy):
                continue

            info = _entry_info_score(entry)
            neutral_scored.append((info, -idx, entry))

        if neutral_scored:
            neutral_scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
            ranked = [entry for _, _, entry in neutral_scored]
            result, dropped = _collect_with_budget(ranked, limit, char_budget)
            _log_stage_debug(debug_context, FallbackStage.NEUTRAL, candidates, result)
            if dropped:
                logger.info(
                    "[%s] char_budget truncated %d records (stage=%s)",
                    debug_context, dropped, FallbackStage.NEUTRAL.value,
                )
            if return_meta:
                res = (result, FallbackStage.NEUTRAL, dropped)
            else:
                res = result
            return _ensure_return_type(res, return_meta)

    _log_stage_debug(debug_context, FallbackStage.EMPTY, candidates, [])
    if return_meta:
        res = ([], FallbackStage.EMPTY, 0)
    else:
        res = []
    return _ensure_return_type(res, return_meta)


@overload
def _select_by_tags_or_all(
    entries: List[Dict[str, Any]],
    tags: Iterable[str],
    limit: int,
    expanded_tags: Optional[Set[str]] = None,
    min_score: Optional[int] = None,
    char_budget: Optional[int] = None,
    return_meta: bool = False,
) -> List[Dict[str, Any]]:
    ...


@overload
def _select_by_tags_or_all(
    entries: List[Dict[str, Any]],
    tags: Iterable[str],
    limit: int,
    expanded_tags: Optional[Set[str]] = None,
    min_score: Optional[int] = None,
    char_budget: Optional[int] = None,
    return_meta: bool = True,
) -> Tuple[List[Dict[str, Any]], FallbackStage, int]:
    ...


def _select_by_tags_or_all(
    entries: List[Dict[str, Any]],
    tags: Iterable[str],
    limit: int,
    expanded_tags: Optional[Set[str]] = None,
    min_score: Optional[int] = None,
    char_budget: Optional[int] = None,
    return_meta: bool = False,
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FallbackStage, int]]:
    """Обёртка для структурных записей."""
    return _select_ranked_entries(
        entries=entries,
        normalized_text="",
        wanted_tags=tags,
        limit=limit,
        scorer=score_structural_entry,
        debug_context="tags_or_all",
        expanded_tags=expanded_tags,
        min_score=min_score,
        char_budget=char_budget,
        fallback_policy=STRUCTURAL_FALLBACK_POLICY,
        return_meta=return_meta,
    )


# ---------------------------------------------------------------------------
# Публичный API — добавлен параметр semantic_rerank
# ---------------------------------------------------------------------------

@overload
def select_grammar_rules(
    kb: Any,
    text: str,
    tags: Iterable[str],
    limit: int = 10,
    candidate_limit: Optional[int] = None,
    min_score: int = 1,
    char_budget: Optional[int] = None,
    return_meta: bool = False,
    semantic_rerank: bool = False,
) -> List[Dict[str, Any]]:
    ...


@overload
def select_grammar_rules(
    kb: Any,
    text: str,
    tags: Iterable[str],
    limit: int = 10,
    candidate_limit: Optional[int] = None,
    min_score: int = 1,
    char_budget: Optional[int] = None,
    return_meta: bool = True,
    semantic_rerank: bool = False,
) -> Tuple[List[Dict[str, Any]], FallbackStage, int]:
    ...


def select_grammar_rules(
    kb: Any,
    text: str,
    tags: Iterable[str],
    limit: int = 10,
    candidate_limit: Optional[int] = None,
    min_score: int = 1,
    char_budget: Optional[int] = None,
    return_meta: bool = False,
    semantic_rerank: bool = False,
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FallbackStage, int]]:
    normalized_text = normalize_text_for_match(text)
    effective_tags = list(tags) or ["grammar"]
    grammar_entries = getattr(kb, 'grammar_errors', [])
    raw = _select_ranked_entries(
        entries=grammar_entries,
        normalized_text=normalized_text,
        wanted_tags=effective_tags,
        limit=limit,
        scorer=score_rule_entry,
        candidate_limit=candidate_limit,
        debug_context="grammar",
        min_score=min_score,
        char_budget=char_budget,
        fallback_policy=RULE_FALLBACK_POLICY,
        return_meta=return_meta,
    )
    if return_meta:
        entries, stage, dropped = raw  # type: ignore[misc]
        return _semantic_rerank(entries, text, semantic_weight=0.35 if semantic_rerank else 0.0), stage, dropped
    return _semantic_rerank(raw, text, semantic_weight=0.35 if semantic_rerank else 0.0)  # type: ignore[arg-type]


@overload
def select_style_issues(
    kb: Any,
    text: str,
    tags: Iterable[str],
    limit: int = 10,
    candidate_limit: Optional[int] = None,
    min_score: int = 1,
    char_budget: Optional[int] = None,
    return_meta: bool = False,
    semantic_rerank: bool = False,
) -> List[Dict[str, Any]]:
    ...


@overload
def select_style_issues(
    kb: Any,
    text: str,
    tags: Iterable[str],
    limit: int = 10,
    candidate_limit: Optional[int] = None,
    min_score: int = 1,
    char_budget: Optional[int] = None,
    return_meta: bool = True,
    semantic_rerank: bool = False,
) -> Tuple[List[Dict[str, Any]], FallbackStage, int]:
    ...


def select_style_issues(
    kb: Any,
    text: str,
    tags: Iterable[str],
    limit: int = 10,
    candidate_limit: Optional[int] = None,
    min_score: int = 1,
    char_budget: Optional[int] = None,
    return_meta: bool = False,
    semantic_rerank: bool = False,
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FallbackStage, int]]:
    normalized_text = normalize_text_for_match(text)
    effective_tags = list(tags) or ["style"]
    stylistic_entries = getattr(kb, 'stylistic_issues', [])
    raw = _select_ranked_entries(
        entries=stylistic_entries,
        normalized_text=normalized_text,
        wanted_tags=effective_tags,
        limit=limit,
        scorer=score_rule_entry,
        candidate_limit=candidate_limit,
        debug_context="style",
        min_score=min_score,
        char_budget=char_budget,
        fallback_policy=RULE_FALLBACK_POLICY,
        return_meta=return_meta,
    )
    if return_meta:
        entries, stage, dropped = raw  # type: ignore[misc]
        return _semantic_rerank(entries, text, semantic_weight=0.35 if semantic_rerank else 0.0), stage, dropped
    return _semantic_rerank(raw, text, semantic_weight=0.35 if semantic_rerank else 0.0)  # type: ignore[arg-type]


@overload
def select_logic_issues(
    kb: Any,
    text: str,
    tags: Iterable[str],
    limit: int = 8,
    candidate_limit: Optional[int] = None,
    min_score: int = 1,
    char_budget: Optional[int] = None,
    return_meta: bool = False,
    semantic_rerank: bool = False,
) -> List[Dict[str, Any]]:
    ...


@overload
def select_logic_issues(
    kb: Any,
    text: str,
    tags: Iterable[str],
    limit: int = 8,
    candidate_limit: Optional[int] = None,
    min_score: int = 1,
    char_budget: Optional[int] = None,
    return_meta: bool = True,
    semantic_rerank: bool = False,
) -> Tuple[List[Dict[str, Any]], FallbackStage, int]:
    ...


def select_logic_issues(
    kb: Any,
    text: str,
    tags: Iterable[str],
    limit: int = 8,
    candidate_limit: Optional[int] = None,
    min_score: int = 1,
    char_budget: Optional[int] = None,
    return_meta: bool = False,
    semantic_rerank: bool = False,
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FallbackStage, int]]:
    logic_entries = getattr(kb, 'logic_issues', [])
    if not logic_entries:
        logger.warning(
            "select_logic_issues: kb.logic_issues пустой. "
            "Блок логики не будет добавлен в промпт. "
            "Проверь файл knowledge_base/logic_issues.json."
        )
        if return_meta:
            return [], FallbackStage.EMPTY, 0
        return []

    normalized_text = normalize_text_for_match(text)
    wanted_tags = list(tags) + ["logic"]
    raw = _select_ranked_entries(
        entries=logic_entries,
        normalized_text=normalized_text,
        wanted_tags=wanted_tags,
        limit=limit,
        scorer=score_rule_entry,
        candidate_limit=candidate_limit,
        debug_context="logic",
        min_score=min_score,
        char_budget=char_budget,
        fallback_policy=RULE_FALLBACK_POLICY,
        return_meta=return_meta,
    )
    if return_meta:
        entries, stage, dropped = raw  # type: ignore[misc]
        return _semantic_rerank(entries, text, semantic_weight=0.35 if semantic_rerank else 0.0), stage, dropped
    return _semantic_rerank(raw, text, semantic_weight=0.35 if semantic_rerank else 0.0)  # type: ignore[arg-type]


@overload
def select_structural_by_tags_or_all(
    entries: List[Dict[str, Any]],
    tags: Iterable[str],
    limit: int,
    expanded_tags: Optional[Set[str]] = None,
    min_score: Optional[int] = None,
    char_budget: Optional[int] = None,
    return_meta: bool = False,
) -> List[Dict[str, Any]]:
    ...


@overload
def select_structural_by_tags_or_all(
    entries: List[Dict[str, Any]],
    tags: Iterable[str],
    limit: int,
    expanded_tags: Optional[Set[str]] = None,
    min_score: Optional[int] = None,
    char_budget: Optional[int] = None,
    return_meta: bool = True,
) -> Tuple[List[Dict[str, Any]], FallbackStage, int]:
    ...


def select_structural_by_tags_or_all(
    entries: List[Dict[str, Any]],
    tags: Iterable[str],
    limit: int,
    expanded_tags: Optional[Set[str]] = None,
    min_score: Optional[int] = None,
    char_budget: Optional[int] = None,
    return_meta: bool = False,
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FallbackStage, int]]:
    return _select_by_tags_or_all(
        entries=entries,
        tags=tags,
        limit=limit,
        expanded_tags=expanded_tags,
        min_score=min_score,
        char_budget=char_budget,
        return_meta=return_meta,
    )