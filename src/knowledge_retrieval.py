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
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
    overload,
)

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


@dataclass(frozen=True)
class SelectionParams:
    """
    Параметры отбора записей из базы знаний.
    """
    require_text_match: bool = False
    scorer: Any = None
    candidate_limit: Optional[int] = None
    debug_context: str = ""
    expanded_tags: Optional[Set[str]] = None
    min_score: Optional[int] = None
    char_budget: Optional[int] = None
    fallback_policy: Optional[FallbackPolicy] = None
    return_meta: bool = False
    semantic_rerank: bool = False


# ---------------------------------------------------------------------------
# Вспомогательные функции (без изменений)
# ---------------------------------------------------------------------------

def normalize_text_for_match(text: str) -> str:
    text = text.replace("ё", "е").replace("Ё", "Е")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def _contains_pattern(normalized_text: str, pattern: str) -> bool:
    if not pattern:
        return False
    norm_pattern = normalize_text_for_match(pattern)
    if not norm_pattern or len(norm_pattern) < 2:
        return False
    if " " not in norm_pattern:
        return re.search(rf"\b{re.escape(norm_pattern)}\b", normalized_text) is not None
    return norm_pattern in normalized_text


def _get_entry_match_patterns(entry: Dict[str, Any]) -> List[str]:
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
    total = 0
    for field in (
        "wrong", "correct", "rule", "description", "name",
        "example_wrong", "example_correct", "example_explanation",
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


def _make_dedupe_key(entry: Dict[str, Any]) -> Tuple[Any, ...]:
    if "id" in entry:
        return ("id", entry["id"])
    def _container_signature(key: str) -> Tuple[Any, ...]:
        container = entry.get(key)
        if not isinstance(container, list):
            return ()
        parts: List[str] = []
        for item in container:
            if isinstance(item, dict):
                parts.append(
                    str(item.get("name", "")) + "|" + str(item.get("description", ""))
                )
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
        debug_context, stage.value, len(candidates), len(selected), preview,
    )


def _sort_ranked(scored: List[Tuple[int, int, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [entry for _, _, entry in scored]


def _ensure_return_type(
    result: Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FallbackStage, int]],
    return_meta: bool,
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FallbackStage, int]]:
    if return_meta:
        if not isinstance(result, tuple) or len(result) != 3:
            raise TypeError(
                "_select_ranked_entries с return_meta=True должен возвращать "
                "tuple из 3 элементов (entries, stage, dropped), "
                f"но получил {type(result).__name__}: {result!r}"
            )
    else:
        if not isinstance(result, list):
            raise TypeError(
                "_select_ranked_entries с return_meta=False должен возвращать "
                f"list, но получил {type(result).__name__}: {result!r}"
            )
    return result


def _semantic_rerank(
    entries: List[Dict[str, Any]],
    query: str,
    semantic_weight: float = 0.35,
    top_k_factor: int = 3,
) -> List[Dict[str, Any]]:
    if not entries or not query or not query.strip() or semantic_weight <= 0:
        return entries
    try:
        from src.semantic_index import (
            get_semantic_index,
            init_semantic_index,
            _entries_for_index,
        )
        index = get_semantic_index()
        if index is None:
            if _entries_for_index:
                logger.info(
                    "SemanticIndex: ленивая инициализация индекса по первому запросу"
                )
                init_semantic_index(_entries_for_index)
                index = get_semantic_index()
            else:
                logger.warning(
                    "SemanticIndex не инициализирован: нет записей для индексации"
                )
                return entries
        if index is None or not index.is_ready():
            return entries
    except ImportError:
        return entries

    n = len(entries)
    top_k = min(n * top_k_factor, 200)
    semantic_results = index.search(query.strip(), top_k=top_k)
    sem_score_map: Dict[int, float] = {
        id(entry): score for entry, score in semantic_results
    }

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
            logger.debug(
                "Семантический re-ranking изменил порядок. До: %s. После: %s",
                before, after,
            )
    return reranked


# ============================================================================
# ВЫДЕЛЕННЫЕ СТАДИИ ДЛЯ _select_ranked_entries (исправлены сигнатуры)
# ============================================================================

def _try_strong_stage(
    candidates: List[Dict[str, Any]],
    normalized_text: str,
    wanted_set: Set[str],
    params: SelectionParams,
    limit: int,
    policy: FallbackPolicy,
) -> Optional[Tuple[List[Dict[str, Any]], int]]:
    """Пытается выбрать записи strong-стадии."""
    effective_min_score = (
        policy.min_strong_score
        if params.min_score is None
        else max(params.min_score, policy.min_strong_score)
    )
    scorer = params.scorer or score_rule_entry

    scored: List[Tuple[int, int, Dict[str, Any]]] = []
    for idx, entry in enumerate(candidates):
        score, tie = scorer(
            entry,
            normalized_text,
            wanted_set,
            idx,
            expanded_tags=params.expanded_tags,
        )
        if params.require_text_match and not _has_text_match(entry, normalized_text):
            continue
        if score >= effective_min_score:
            scored.append((score, tie, entry))

    if not scored:
        return None

    ranked = _sort_ranked(scored)
    result, dropped = _collect_with_budget(ranked, limit, params.char_budget)
    _log_stage_debug(params.debug_context, FallbackStage.STRONG, candidates, result)
    if dropped:
        logger.info(
            "[%s] char_budget truncated %d records (stage=%s)",
            params.debug_context, dropped, FallbackStage.STRONG.value,
        )
    return result, dropped


def _try_text_only_stage(
    candidates: List[Dict[str, Any]],
    normalized_text: str,
    wanted_set: Set[str],  # добавлен, но не используется
    params: SelectionParams,
    limit: int,
    policy: FallbackPolicy,
) -> Optional[Tuple[List[Dict[str, Any]], int]]:
    """Пытается выбрать записи text_only-стадии."""
    if not policy.allow_text_only:
        return None

    text_only_scored: List[Tuple[int, int, Dict[str, Any]]] = []
    for idx, entry in enumerate(candidates):
        text_strength = _get_text_match_strength(entry, normalized_text)
        if text_strength <= 0:
            continue
        info = _entry_info_score(entry)
        text_only_scored.append((text_strength, info - idx, entry))

    if not text_only_scored:
        return None

    text_only_scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    ranked = [entry for _, _, entry in text_only_scored]
    result, dropped = _collect_with_budget(ranked, limit, params.char_budget)
    _log_stage_debug(params.debug_context, FallbackStage.TEXT_ONLY, candidates, result)
    if dropped:
        logger.info(
            "[%s] char_budget truncated %d records (stage=%s)",
            params.debug_context, dropped, FallbackStage.TEXT_ONLY.value,
        )
    return result, dropped


def _try_tag_only_stage(
    candidates: List[Dict[str, Any]],
    normalized_text: str,  # добавлен, но не используется
    wanted_set: Set[str],
    params: SelectionParams,
    limit: int,
    policy: FallbackPolicy,
) -> Optional[Tuple[List[Dict[str, Any]], int]]:
    """Пытается выбрать записи tag_only-стадии."""
    if not policy.allow_tag_only:
        return None

    tag_only_scored: List[Tuple[int, int, int, Dict[str, Any]]] = []
    for idx, entry in enumerate(candidates):
        if policy.primary_only_for_tag_fallback:
            overlap = _get_primary_overlap(entry, wanted_set)
        else:
            overlap = _get_any_overlap(entry, wanted_set, params.expanded_tags)
        if overlap <= 0:
            continue
        info = _entry_info_score(entry)
        tag_only_scored.append((overlap, info, -idx, entry))

    if not tag_only_scored:
        return None

    tag_only_scored.sort(
        key=lambda item: (item[0], item[1], item[2]),
        reverse=True,
    )
    ranked = [entry for _, _, _, entry in tag_only_scored]
    result, dropped = _collect_with_budget(ranked, limit, params.char_budget)
    _log_stage_debug(params.debug_context, FallbackStage.TAG_ONLY, candidates, result)
    if dropped:
        logger.info(
            "[%s] char_budget truncated %d records (stage=%s)",
            params.debug_context, dropped, FallbackStage.TAG_ONLY.value,
        )
    return result, dropped


def _try_neutral_stage(
    candidates: List[Dict[str, Any]],
    normalized_text: str,  # добавлен, но не используется
    wanted_set: Set[str],  # добавлен, но не используется
    params: SelectionParams,
    limit: int,
    policy: FallbackPolicy,
) -> Optional[Tuple[List[Dict[str, Any]], int]]:
    """Пытается выбрать записи neutral-стадии."""
    if not policy.allow_neutral_fallback:
        return None

    neutral_scored: List[Tuple[int, int, Dict[str, Any]]] = []
    for idx, entry in enumerate(candidates):
        if not _is_neutral_candidate(entry, policy):
            continue
        info = _entry_info_score(entry)
        neutral_scored.append((info, -idx, entry))

    if not neutral_scored:
        return None

    neutral_scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    ranked = [entry for _, _, entry in neutral_scored]
    result, dropped = _collect_with_budget(ranked, limit, params.char_budget)
    _log_stage_debug(params.debug_context, FallbackStage.NEUTRAL, candidates, result)
    if dropped:
        logger.info(
            "[%s] char_budget truncated %d records (stage=%s)",
            params.debug_context, dropped, FallbackStage.NEUTRAL.value,
        )
    return result, dropped


# ============================================================================
# ОСНОВНАЯ ФУНКЦИЯ (теперь координатор)
# ============================================================================

def _select_ranked_entries(
    entries: List[Dict[str, Any]],
    normalized_text: str,
    wanted_tags: Iterable[str],
    limit: int,
    params: SelectionParams,
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FallbackStage, int]]:
    """
    Общая функция ранжирования записей с quality-gated fallback.
    Стадии: strong -> text_only -> tag_only -> neutral -> empty.
    Если params.return_meta=True, возвращает (entries, stage, dropped).
    """
    if not entries or limit <= 0:
        stage = FallbackStage.EMPTY
        result = ([], stage, 0) if params.return_meta else []
        return _ensure_return_type(result, params.return_meta)

    policy = params.fallback_policy or RULE_FALLBACK_POLICY
    candidates = (
        entries if params.candidate_limit is None else entries[:params.candidate_limit]
    )
    wanted_set = _normalize_tag_set(wanted_tags)

    # Пробуем стадии по порядку
    for stage_func, stage_name in [
        (_try_strong_stage, FallbackStage.STRONG),
        (_try_text_only_stage, FallbackStage.TEXT_ONLY),
        (_try_tag_only_stage, FallbackStage.TAG_ONLY),
        (_try_neutral_stage, FallbackStage.NEUTRAL),
    ]:
        result_entries_dropped = stage_func(
            candidates=candidates,
            normalized_text=normalized_text,
            wanted_set=wanted_set,
            params=params,
            limit=limit,
            policy=policy,
        )
        if result_entries_dropped is not None:
            result_entries, dropped = result_entries_dropped
            if params.return_meta:
                res = (result_entries, stage_name, dropped)
            else:
                res = result_entries
            return _ensure_return_type(res, params.return_meta)

    # Если ни одна стадия не сработала
    _log_stage_debug(params.debug_context, FallbackStage.EMPTY, candidates, [])
    if params.return_meta:
        res = ([], FallbackStage.EMPTY, 0)
    else:
        res = []
    return _ensure_return_type(res, params.return_meta)


# ---------------------------------------------------------------------------
# Публичные функции (без изменений)
# ---------------------------------------------------------------------------

_CATEGORY_CONFIG = {
    "grammar": {
        "attr": "grammar_errors",
        "default_tags": ["grammar"],
        "fallback_policy": RULE_FALLBACK_POLICY,
        "default_limit": 10,
    },
    "style": {
        "attr": "stylistic_issues",
        "default_tags": ["style"],
        "fallback_policy": RULE_FALLBACK_POLICY,
        "default_limit": 10,
    },
    "logic": {
        "attr": "logic_issues",
        "default_tags": ["logic"],
        "fallback_policy": RULE_FALLBACK_POLICY,
        "default_limit": 8,
    },
    "structural": {
        "attr": None,
        "default_tags": [],
        "fallback_policy": STRUCTURAL_FALLBACK_POLICY,
        "default_limit": 10,
    },
}


@overload
def select_entries(
    kb_or_entries: Any,
    text: str,
    tags: Iterable[str],
    category: Literal["grammar", "style", "logic", "structural"],
    params: Optional[SelectionParams] = None,
    limit: int = 10,
    candidate_limit: Optional[int] = None,
    min_score: int = 1,
    char_budget: Optional[int] = None,
    return_meta: bool = False,
    semantic_rerank: bool = False,
) -> List[Dict[str, Any]]: ...


@overload
def select_entries(
    kb_or_entries: Any,
    text: str,
    tags: Iterable[str],
    category: Literal["grammar", "style", "logic", "structural"],
    params: Optional[SelectionParams] = None,
    limit: int = 10,
    candidate_limit: Optional[int] = None,
    min_score: int = 1,
    char_budget: Optional[int] = None,
    return_meta: Literal[True] = True,
    semantic_rerank: bool = False,
) -> Tuple[List[Dict[str, Any]], FallbackStage, int]: ...


def select_entries(
    kb_or_entries: Any,
    text: str,
    tags: Iterable[str],
    category: Literal["grammar", "style", "logic", "structural"],
    params: Optional[SelectionParams] = None,
    limit: int = 10,
    candidate_limit: Optional[int] = None,
    min_score: int = 1,
    char_budget: Optional[int] = None,
    return_meta: bool = False,
    semantic_rerank: bool = False,
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FallbackStage, int]]:
    if params is None:
        scorer = score_rule_entry if category != "structural" else score_structural_entry
        fb_policy = (
            STRUCTURAL_FALLBACK_POLICY
            if category == "structural"
            else RULE_FALLBACK_POLICY
        )
        params = SelectionParams(
            require_text_match=False,
            scorer=scorer,
            candidate_limit=candidate_limit,
            debug_context=category,
            expanded_tags=None,
            min_score=min_score,
            char_budget=char_budget,
            fallback_policy=fb_policy,
            return_meta=return_meta,
            semantic_rerank=semantic_rerank,
        )

    if category == "structural":
        if not isinstance(kb_or_entries, list):
            raise TypeError("For category 'structural', kb_or_entries must be a list of entries.")
        entries_source = kb_or_entries
    else:
        config = _CATEGORY_CONFIG[category]
        attr = config["attr"]
        entries_source = getattr(kb_or_entries, attr, [])
        if not entries_source:
            logger.warning(
                "select_entries: %s пустой. Блок %s не будет добавлен.",
                attr, category
            )
            if params.return_meta:
                return [], FallbackStage.EMPTY, 0
            return []

    if category == "structural":
        normalized_text = ""
        effective_tags = list(tags) or []
    else:
        normalized_text = normalize_text_for_match(text)
        effective_tags = list(tags) or config.get("default_tags", [category])

    raw = _select_ranked_entries(
        entries=entries_source,
        normalized_text=normalized_text,
        wanted_tags=effective_tags,
        limit=limit,
        params=params,
    )

    if category != "structural":
        weight = 0.35 if params.semantic_rerank else 0.0
        if params.return_meta:
            entries, stage, dropped = raw  # type: ignore
            entries = _semantic_rerank(entries, text, semantic_weight=weight)
            return entries, stage, dropped
        else:
            raw = _semantic_rerank(raw, text, semantic_weight=weight)  # type: ignore
            return raw
    else:
        return raw  # type: ignore


# ---------------------------------------------------------------------------
# Обёртки для обратной совместимости
# ---------------------------------------------------------------------------

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
    params = SelectionParams(
        scorer=score_rule_entry,
        candidate_limit=candidate_limit,
        debug_context="grammar",
        min_score=min_score,
        char_budget=char_budget,
        fallback_policy=RULE_FALLBACK_POLICY,
        return_meta=return_meta,
        semantic_rerank=semantic_rerank,
    )
    return select_entries(
        kb, text, tags, "grammar",
        params=params,
        limit=limit,
        return_meta=return_meta,
        semantic_rerank=semantic_rerank,
    )


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
    params = SelectionParams(
        scorer=score_rule_entry,
        candidate_limit=candidate_limit,
        debug_context="style",
        min_score=min_score,
        char_budget=char_budget,
        fallback_policy=RULE_FALLBACK_POLICY,
        return_meta=return_meta,
        semantic_rerank=semantic_rerank,
    )
    return select_entries(
        kb, text, tags, "style",
        params=params,
        limit=limit,
        return_meta=return_meta,
        semantic_rerank=semantic_rerank,
    )


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
    params = SelectionParams(
        scorer=score_rule_entry,
        candidate_limit=candidate_limit,
        debug_context="logic",
        min_score=min_score,
        char_budget=char_budget,
        fallback_policy=RULE_FALLBACK_POLICY,
        return_meta=return_meta,
        semantic_rerank=semantic_rerank,
    )
    return select_entries(
        kb, text, tags, "logic",
        params=params,
        limit=limit,
        return_meta=return_meta,
        semantic_rerank=semantic_rerank,
    )


def select_structural_by_tags_or_all(
    entries: List[Dict[str, Any]],
    tags: Iterable[str],
    limit: int,
    expanded_tags: Optional[Set[str]] = None,
    min_score: Optional[int] = None,
    char_budget: Optional[int] = None,
    return_meta: bool = False,
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FallbackStage, int]]:
    params = SelectionParams(
        scorer=score_structural_entry,
        candidate_limit=None,
        debug_context="tags_or_all",
        expanded_tags=expanded_tags,
        min_score=min_score,
        char_budget=char_budget,
        fallback_policy=STRUCTURAL_FALLBACK_POLICY,
        return_meta=return_meta,
        semantic_rerank=False,
    )
    return select_entries(
        entries, "", tags, "structural",
        params=params,
        limit=limit,
        return_meta=return_meta,
    )