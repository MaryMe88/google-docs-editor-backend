"""
knowledge_retrieval.py

Логика нормализации текста, скоринга и ранжирования записей базы знаний.
Единственный источник истины для всего retrieval.

Содержит:
  - normalize_text_for_match, _contains_pattern  — нормализация и матчинг
  - SCORE_WEIGHTS                                 — скоринговые константы
  - score_rule_entry, score_structural_entry      — scorer'ы
  - _select_ranked_entries_staged                 — quality-gated fallback (ФП-2)
  - _select_ranked_entries                        — публичный алиас / обёртка
  - select_grammar_rules, select_style_issues,
    select_logic_issues, _select_by_tags_or_all   — публичные селекторы
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Final, Iterable, List, Optional, Set, Tuple

from src.tag_registry import normalize_tag

logger = logging.getLogger(__name__)


# ============================================================================
# Нормализация текста
# ============================================================================

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
    wrong, name, rule, description (в указанном порядке).
    """
    patterns: List[str] = []
    seen: Set[str] = set()
    for field in ("wrong", "name", "rule", "description"):
        val = entry.get(field)
        if isinstance(val, str):
            stripped = val.strip()
            if stripped and stripped not in seen:
                seen.add(stripped)
                patterns.append(stripped)
    return patterns


def _entry_info_score(entry: Dict[str, Any]) -> int:
    """Количество информативных полей в записи (для fallback-ранжирования)."""
    score = 0
    for field in ("name", "description", "rule", "wrong", "when_to_use"):
        val = entry.get(field)
        if isinstance(val, str) and val.strip():
            score += 1
        elif isinstance(val, list) and val:
            score += 1
    for container_key in ("steps", "sections"):
        container = entry.get(container_key)
        if isinstance(container, list) and container:
            score += 1
    return score


def _estimate_entry_chars(entry: Dict[str, Any]) -> int:
    """
    Быстрая оценка размера записи в символах для char_budget.
    Суммирует длины строковых полей и имён шагов/секций.
    """
    total = 0
    for field in ("wrong", "correct", "rule", "description", "name",
                  "example_wrong", "example_correct", "example_explanation"):
        val = entry.get(field)
        if isinstance(val, str):
            total += len(val)
    for container_key in ("steps", "sections"):
        container = entry.get(container_key)
        if isinstance(container, list):
            for step in container:
                if isinstance(step, dict):
                    for f in ("name", "description"):
                        v = step.get(f)
                        if isinstance(v, str):
                            total += len(v)
    return total


# ============================================================================
# Скоринговые константы
# ============================================================================

SCORE_WEIGHTS: Final[Dict[str, int]] = {
    "wrong_exact_match":  1000,
    "name_exact_match":    500,
    "partial_text_match":  200,
    "tag_primary":          10,
    "tag_primary_bonus":     1,
    "tag_expanded":          2,
}

# Именованные пороги для использования в _log_selection_debug и внешних потребителях
SCORE_THRESHOLD_TEXT_MATCH    = SCORE_WEIGHTS["wrong_exact_match"]   # 1000
SCORE_THRESHOLD_PARTIAL_MATCH = SCORE_WEIGHTS["partial_text_match"]  # 200
SCORE_THRESHOLD_TAGS          = SCORE_WEIGHTS["tag_primary"]          # 10


# ============================================================================
# Scorer'ы
# ============================================================================

def score_rule_entry(
    entry: Dict[str, Any],
    normalized_text: str,
    wanted_tags: Set[str],
    idx: int,
    expanded_tags: Optional[Set[str]] = None,
) -> Tuple[int, int]:
    """Скоринг для «правильных» записей (грамматика, стиль, логика)."""
    score = 0
    wrong_val = entry.get("wrong", "")
    if isinstance(wrong_val, str):
        wrong_stripped = wrong_val.strip()
        if wrong_stripped and _contains_pattern(normalized_text, wrong_stripped):
            score += SCORE_WEIGHTS["wrong_exact_match"]
    if score == 0:
        for field in ("name", "rule", "description"):
            val = entry.get(field)
            if not isinstance(val, str):
                continue
            stripped = val.strip()
            if stripped and _contains_pattern(normalized_text, stripped):
                score += SCORE_WEIGHTS["partial_text_match"]
                break
    entry_tags = entry.get("tags", [])
    if not isinstance(entry_tags, (list, tuple)):
        entry_tags = []
    tag_set = {normalize_tag(t) for t in entry_tags if isinstance(t, str)}
    overlap = len(tag_set & wanted_tags)
    score += overlap * SCORE_WEIGHTS["tag_primary"]
    if overlap > 0:
        score += SCORE_WEIGHTS["tag_primary_bonus"]
    if expanded_tags:
        score += len(tag_set & expanded_tags) * SCORE_WEIGHTS["tag_expanded"]
    return (score, -idx)


def score_structural_entry(
    entry: Dict[str, Any],
    normalized_text: str,
    wanted_tags: Set[str],
    idx: int,
    expanded_tags: Optional[Set[str]] = None,
) -> Tuple[int, int]:
    """Скоринг для структурных записей (storytelling, marketing, rhetoric, composition, editorial)."""
    score = 0
    patterns: List[str] = []

    def _add_field(field: str) -> None:
        val = entry.get(field)
        if isinstance(val, str):
            stripped = val.strip()
            if stripped:
                patterns.append(stripped)

    _add_field("name")
    _add_field("description")
    _add_field("when_to_use")
    _add_field("rule")

    when = entry.get("when_to_use")
    if isinstance(when, list):
        for item in when:
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
                for f in ("name", "description"):
                    val = step.get(f)
                    if isinstance(val, str):
                        s = val.strip()
                        if s:
                            patterns.append(s)

    seen: Set[str] = set()
    unique_patterns: List[str] = []
    for p in patterns:
        if p not in seen:
            seen.add(p)
            unique_patterns.append(p)

    name_val = entry.get("name", "")
    name_stripped = name_val.strip() if isinstance(name_val, str) else ""
    match_bonus = 0
    for pat in unique_patterns:
        if not _contains_pattern(normalized_text, pat):
            continue
        match_bonus = (
            SCORE_WEIGHTS["name_exact_match"]
            if (name_stripped and pat == name_stripped)
            else SCORE_WEIGHTS["partial_text_match"]
        )
        break
    score += match_bonus

    entry_tags = entry.get("tags", [])
    if not isinstance(entry_tags, (list, tuple)):
        entry_tags = []
    tag_set = {normalize_tag(t) for t in entry_tags if isinstance(t, str)}
    overlap = len(tag_set & wanted_tags)
    score += overlap * SCORE_WEIGHTS["tag_primary"]
    if overlap > 0:
        score += SCORE_WEIGHTS["tag_primary_bonus"]
    if expanded_tags:
        score += SCORE_WEIGHTS["tag_expanded"] * len(tag_set & expanded_tags)

    return (score, -idx)


# Алиас для обратной совместимости
_score_entry = score_rule_entry


# ============================================================================
# ФП-2: Quality-gated fallback — ступенчатая лестница
# ============================================================================

class FallbackStage(str, Enum):
    """Ступень fallback-лестницы, на которой получен результат."""
    A = "A"   # text match + tag overlap            — наибольшая уверенность
    B = "B"   # text match, нет tag overlap
    C = "C"   # нет text match, есть tag overlap
    D = "D"   # neutral/global entries с high info score
    E = "E"   # пусто — silence over noise


@dataclass
class SelectionResult:
    """
    Результат ранжирования с диагностической информацией о ступени fallback.

    Атрибуты:
        entries:  Список отобранных записей KB.
        stage:    Ступень FallbackStage, на которой получен результат.
        scores:   Параллельный список итоговых score для каждой записи (0 для D/E).
    """
    entries: List[Dict[str, Any]]
    stage: FallbackStage
    scores: List[int]


# Минимальные пороги баллов по стадиям
_STAGE_A_MIN_SCORE: Final[int] = SCORE_THRESHOLD_TEXT_MATCH + SCORE_THRESHOLD_TAGS  # 1010
_STAGE_B_MIN_SCORE: Final[int] = SCORE_THRESHOLD_TEXT_MATCH                          # 1000
_STAGE_C_MIN_SCORE: Final[int] = SCORE_THRESHOLD_TAGS                                # 10
_STAGE_D_INFO_MIN:  Final[int] = 3  # минимальный info_score для Stage D


def _collect_deduped(
    scored: List[Tuple[int, Any, Dict[str, Any]]],
    limit: int,
    char_budget: Optional[int],
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    Применяет дедупликацию и char_budget к отсортированному списку scored.
    Возвращает (entries, score_list).
    """
    result: List[Dict[str, Any]] = []
    score_list: List[int] = []
    seen_keys: Set[Tuple[Any, ...]] = set()
    chars_used = 0
    for score_val, _tie, entry in scored:
        key = _make_dedupe_key(entry)
        if key in seen_keys:
            continue
        entry_chars = _estimate_entry_chars(entry)
        if char_budget is not None and result and chars_used + entry_chars > char_budget:
            break
        seen_keys.add(key)
        result.append(entry)
        score_list.append(score_val)
        chars_used += entry_chars
        if len(result) >= limit:
            break
    return result, score_list


def _select_ranked_entries_staged(
    entries: List[Dict[str, Any]],
    normalized_text: str,
    wanted_set: Set[str],
    limit: int,
    scorer: Callable[..., Tuple[int, int]],
    candidate_limit: Optional[int],
    debug_context: str,
    expanded_tags: Optional[Set[str]],
    char_budget: Optional[int],
) -> SelectionResult:
    """
    Ранжирование с quality-gated fallback (ФП-2).

    Лестница ступеней:
      Stage A — text match + tag overlap  (score >= _STAGE_A_MIN_SCORE)
      Stage B — text match, нет tag overlap  (score >= _STAGE_B_MIN_SCORE, < A)
      Stage C — нет text match, есть tag overlap  (score >= _STAGE_C_MIN_SCORE)
      Stage D — neutral entries с высоким info_score
      Stage E — пусто (silence over noise)

    На каждой ступени применяются: дедупликация, char_budget, limit.
    Следующая ступень активируется только если предыдущая дала пустой результат.
    """
    candidates = entries if candidate_limit is None else entries[:candidate_limit]
    if not candidates:
        return SelectionResult([], FallbackStage.E, [])

    # Скорим всех кандидатов один раз
    scored_all: List[Tuple[int, int, Dict[str, Any]]] = []
    for idx, entry in enumerate(candidates):
        score, tie = scorer(entry, normalized_text, wanted_set, idx, expanded_tags=expanded_tags)
        scored_all.append((score, tie, entry))

    scored_all.sort(key=lambda x: (x[0], x[1]), reverse=True)

    # ── Stage A: text match + tag overlap ────────────────────────────────
    stage_a = [(s, t, e) for s, t, e in scored_all if s >= _STAGE_A_MIN_SCORE]
    if stage_a:
        _log_stage(debug_context, FallbackStage.A, len(stage_a), stage_a[:3])
        entries_out, scores_out = _collect_deduped(stage_a, limit, char_budget)
        return SelectionResult(entries_out, FallbackStage.A, scores_out)

    # ── Stage B: text match без tag overlap ──────────────────────────────
    stage_b = [
        (s, t, e) for s, t, e in scored_all
        if _STAGE_B_MIN_SCORE <= s < _STAGE_A_MIN_SCORE
    ]
    if stage_b:
        _log_stage(debug_context, FallbackStage.B, len(stage_b), stage_b[:3])
        entries_out, scores_out = _collect_deduped(stage_b, limit, char_budget)
        return SelectionResult(entries_out, FallbackStage.B, scores_out)

    # ── Stage C: tag overlap без text match ──────────────────────────────
    stage_c = [(s, t, e) for s, t, e in scored_all if s >= _STAGE_C_MIN_SCORE]
    if stage_c:
        _log_stage(debug_context, FallbackStage.C, len(stage_c), stage_c[:3])
        entries_out, scores_out = _collect_deduped(stage_c, limit, char_budget)
        return SelectionResult(entries_out, FallbackStage.C, scores_out)

    # ── Stage D: neutral entries с высоким info_score ────────────────────
    # Записи без тегов или с тегом "neutral"/"global" и достаточным info_score
    stage_d_candidates: List[Tuple[int, int, int, Dict[str, Any]]] = []
    for idx, entry in enumerate(candidates):
        entry_tags = entry.get("tags", [])
        if not isinstance(entry_tags, (list, tuple)):
            entry_tags = []
        tag_set = {normalize_tag(t) for t in entry_tags if isinstance(t, str)}
        # Нейтральность: нет тегов, или есть только generic теги
        is_neutral = (
            not tag_set
            or bool(tag_set & {"neutral", "global", "general", "editing"})
        )
        if not is_neutral:
            continue
        info = _entry_info_score(entry)
        if info < _STAGE_D_INFO_MIN:
            continue
        stage_d_candidates.append((info, -idx, idx, entry))

    if stage_d_candidates:
        stage_d_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        stage_d_scored = [(0, -c[2], c[3]) for c in stage_d_candidates]
        _log_stage(debug_context, FallbackStage.D, len(stage_d_candidates), stage_d_scored[:3])
        entries_out, scores_out = _collect_deduped(stage_d_scored, limit, char_budget)
        return SelectionResult(entries_out, FallbackStage.D, scores_out)

    # ── Stage E: silence over noise ───────────────────────────────────────
    if debug_context:
        logger.debug(
            "[%s] Stage E: no suitable entries found, returning [] (silence over noise).",
            debug_context,
        )
    return SelectionResult([], FallbackStage.E, [])


def _log_stage(
    debug_context: str,
    stage: FallbackStage,
    count: int,
    top: List[Tuple[int, int, Dict[str, Any]]],
) -> None:
    """Логирует на каком stage получен результат и top-3 записи."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    top_info = []
    for s, _t, e in top:
        name = e.get("name", e.get("wrong", "?"))[:30]
        top_info.append((s, name))
    logger.debug(
        "[%s] Stage %s: %d candidates, top=%s",
        debug_context, stage.value, count, top_info,
    )


# ============================================================================
# Вспомогательные функции
# ============================================================================

def _log_selection_debug(
    debug_context: str,
    candidates: List[Dict[str, Any]],
    scored: List[Tuple[int, int, Dict[str, Any]]],
    limit: int,
) -> None:
    """Логирует диагностику ранжирования (для режима без staged-pipeline)."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    if not scored:
        logger.debug("[%s] No scored items (all below threshold).", debug_context)
        return
    top_info = []
    for s, _t, e in scored[:5]:
        name = e.get("name", e.get("wrong", "?"))[:30]
        if s >= SCORE_THRESHOLD_TEXT_MATCH:
            reason = "text_match"
        elif s >= SCORE_THRESHOLD_PARTIAL_MATCH:
            reason = "partial_text"
        elif s >= SCORE_THRESHOLD_TAGS:
            reason = "tags"
        else:
            reason = "fallback"
        top_info.append((s, name, reason))
    logger.debug(
        "[%s] Candidates: %d, selected: %d, top scores: %s",
        debug_context, len(candidates), min(limit, len(scored)), top_info,
    )
    if len(scored) > limit:
        missed = scored[limit: limit + 2]
        missed_info = [(s, e.get("name", "?")[:30]) for s, _, e in missed]
        logger.debug("[%s] Missed due to limit: %s", debug_context, missed_info)


def _make_dedupe_key(entry: Dict[str, Any]) -> Tuple[Any, ...]:
    """Строит ключ дедупликации записи."""
    if "id" in entry:
        return ("id", entry["id"])
    return (
        entry.get("wrong", ""),
        entry.get("rule", ""),
        entry.get("description", ""),
        entry.get("name", ""),
    )


# ============================================================================
# Публичный API ранжирования
# ============================================================================

def _select_ranked_entries(
    entries: List[Dict[str, Any]],
    normalized_text: str,
    wanted_tags: Iterable[str],
    limit: int,
    require_text_match: bool = False,
    scorer: Callable[..., Tuple[int, int]] = score_rule_entry,
    candidate_limit: Optional[int] = None,
    debug_context: str = "",
    expanded_tags: Optional[Set[str]] = None,
    min_score: Optional[int] = None,
    char_budget: Optional[int] = None,
    return_result: bool = False,
) -> List[Dict[str, Any]]:
    """
    Ранжирование записей KB с quality-gated fallback (ФП-2).

    Интерфейс полностью совместим со старым _select_ranked_entries.
    Внутри делегирует в _select_ranked_entries_staged.

    Args:
        entries:            Список записей KB.
        normalized_text:    Нормализованный текст запроса.
        wanted_tags:        Теги для скоринга.
        limit:              Максимальное количество записей в выдаче.
        require_text_match: Если True — возвращать только Stage A/B.
        scorer:             Функция скоринга.
        candidate_limit:    Сколько записей рассматривать (None = все).
        debug_context:      Метка для диагностических логов.
        expanded_tags:      Расширенные теги (меньший вес).
        min_score:          Минимальный балл (legacy-параметр; заменён staged-логикой,
                            но учитывается для обратной совместимости при staged=False).
        char_budget:        Мягкий лимит символов суммарного размера выдачи.
        return_result:      Если True — возвращает SelectionResult вместо List.
    """
    if not entries:
        return [] if not return_result else SelectionResult([], FallbackStage.E, [])

    wanted_set = {normalize_tag(t) for t in wanted_tags if isinstance(t, str)}

    result = _select_ranked_entries_staged(
        entries=entries,
        normalized_text=normalized_text,
        wanted_set=wanted_set,
        limit=limit,
        scorer=scorer,
        candidate_limit=candidate_limit,
        debug_context=debug_context,
        expanded_tags=expanded_tags,
        char_budget=char_budget,
    )

    # require_text_match: разрешаем только Stage A и B
    if require_text_match and result.stage not in (FallbackStage.A, FallbackStage.B):
        if debug_context:
            logger.debug(
                "[%s] require_text_match=True, stage=%s — returning []",
                debug_context, result.stage.value,
            )
        return [] if not return_result else SelectionResult([], FallbackStage.E, [])

    if return_result:
        return result  # type: ignore[return-value]

    return result.entries


def _select_by_tags_or_all(
    entries: List[Dict[str, Any]],
    tags: Iterable[str],
    limit: int,
    expanded_tags: Optional[Set[str]] = None,
    min_score: Optional[int] = None,
    char_budget: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Обёртка для структурных записей без текстового матчинга."""
    return _select_ranked_entries(
        entries,
        normalized_text="",
        wanted_tags=tags,
        limit=limit,
        scorer=score_structural_entry,
        debug_context="tags_or_all",
        expanded_tags=expanded_tags,
        char_budget=char_budget,
    )


# ============================================================================
# Публичные селекторы
# ============================================================================

def select_grammar_rules(
    kb: Any,
    text: str,
    tags: Iterable[str],
    limit: int = 10,
    candidate_limit: Optional[int] = None,
    min_score: int = 1,
    char_budget: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Выбирает релевантные правила грамматики из KB."""
    normalized_text = normalize_text_for_match(text)
    effective_tags = list(tags) or ["grammar"]
    return _select_ranked_entries(
        kb.grammar_errors, normalized_text, effective_tags, limit,
        scorer=score_rule_entry, candidate_limit=candidate_limit,
        debug_context="grammar", char_budget=char_budget,
    )


def select_style_issues(
    kb: Any,
    text: str,
    tags: Iterable[str],
    limit: int = 10,
    candidate_limit: Optional[int] = None,
    min_score: int = 1,
    char_budget: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Выбирает релевантные стилистические проблемы из KB."""
    normalized_text = normalize_text_for_match(text)
    effective_tags = list(tags) or ["style"]
    return _select_ranked_entries(
        kb.stylistic_issues, normalized_text, effective_tags, limit,
        scorer=score_rule_entry, candidate_limit=candidate_limit,
        debug_context="style", char_budget=char_budget,
    )


def select_logic_issues(
    kb: Any,
    text: str,
    tags: Iterable[str],
    limit: int = 8,
    candidate_limit: Optional[int] = None,
    min_score: int = 1,
    char_budget: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Выбирает релевантные логические проблемы из KB."""
    normalized_text = normalize_text_for_match(text)
    wanted_tags = list(tags) + ["logic"]
    candidates: List[Dict[str, Any]] = (
        kb.logic_issues if kb.logic_issues else kb.stylistic_issues + kb.grammar_errors
    )
    return _select_ranked_entries(
        candidates, normalized_text, wanted_tags, limit,
        scorer=score_rule_entry, candidate_limit=candidate_limit,
        debug_context="logic", char_budget=char_budget,
    )


# Алиасы для обратной совместимости
select_structural_by_tags_or_all = _select_by_tags_or_all
_score_entry = score_rule_entry
