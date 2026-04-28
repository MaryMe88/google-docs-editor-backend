"""
knowledge_retrieval.py

Логика нормализации текста, скоринга и ранжирования записей базы знаний.
Отдельный модуль, чтобы разгрузить prompt_builder.py.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Final, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ============================================================================
# Нормализация текста и матчинга
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
    if not norm_pattern:
        return False
    if len(norm_pattern) < 2:
        return False
    if " " not in norm_pattern:
        return (
            re.search(rf"\b{re.escape(norm_pattern)}\b", normalized_text) is not None
        )
    return norm_pattern in normalized_text


def _get_entry_match_patterns(entry: Dict[str, Any]) -> List[str]:
    """
    Собирает кандидаты для текстового матчинга из полей:
    wrong, name, rule, description (в указанном порядке).
    Возвращает уникальные непустые строки после strip().
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
    """Количество информативных полей в записи (для fallback)."""
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

# ============================================================================
# Скоринговые константы
# ============================================================================

SCORE_WEIGHTS: Final[Dict[str, int]] = {
    "wrong_exact_match": 1000,      # точное совпадение с полем wrong
    "name_exact_match": 500,        # точное совпадение с полем name (структурный)
    "partial_text_match": 200,      # совпадение с другими полями
    "tag_primary": 10,              # overlap с primary тегами (за каждый тег)
    "tag_primary_bonus": 1,         # бонус за наличие любого primary overlap
    "tag_expanded": 2,              # overlap с expanded тегами (за каждый тег)
}

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
    """
    Скоринг для «правильных» записей (грамматика, стиль, логика).
    Поля: wrong, correct, rule, description, tags.

    Баллы:
    - точное совпадение с 'wrong' -> +1000
    - совпадение с name/rule/description -> +200
    - overlap с primary тегами * 10 -> + overlap * 10
    - overlap с expanded тегами * 2 -> + overlap_exp * 2
    - дополнительный +1, если есть primary overlap
    """
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
            if not stripped:
                continue
            if _contains_pattern(normalized_text, stripped):
                score += SCORE_WEIGHTS["partial_text_match"]
                break

    entry_tags = entry.get("tags", [])
    if not isinstance(entry_tags, (list, tuple)):
        entry_tags = []
    tag_set = {t.strip().lower() for t in entry_tags if isinstance(t, str)}

    overlap = len(tag_set & wanted_tags)
    score += overlap * SCORE_WEIGHTS["tag_primary"]
    if overlap > 0:
        score += SCORE_WEIGHTS["tag_primary_bonus"]

    if expanded_tags:
        overlap_exp = len(tag_set & expanded_tags)
        score += overlap_exp * SCORE_WEIGHTS["tag_expanded"]

    return (score, -idx)


def score_structural_entry(
    entry: Dict[str, Any],
    normalized_text: str,
    wanted_tags: Set[str],
    idx: int,
    expanded_tags: Optional[Set[str]] = None,
) -> Tuple[int, int]:
    """
    Скоринг для структурных записей (storytelling, marketing, rhetoric,
    composition, editorial). Поля: name, description, when_to_use, steps/sections.

    Баллы:
    - текстовое совпадение с 'name' -> +500
    - иначе совпадение с другими полями -> +200
    - overlap primary тегов * 10
    - overlap expanded тегов * 2
    - дополнительный +1 при primary overlap
    """
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
                step_name = step.get("name")
                if isinstance(step_name, str):
                    s = step_name.strip()
                    if s:
                        patterns.append(s)
                step_desc = step.get("description")
                if isinstance(step_desc, str):
                    s = step_desc.strip()
                    if s:
                        patterns.append(s)

    # Удаляем дубликаты, сохраняя порядок
    seen: Set[str] = set()
    unique_patterns: List[str] = []
    for p in patterns:
        if p not in seen:
            seen.add(p)
            unique_patterns.append(p)

    match_bonus = 0
    name_val = entry.get("name", "")
    name_stripped = name_val.strip() if isinstance(name_val, str) else ""
    for pat in unique_patterns:
        if not _contains_pattern(normalized_text, pat):
            continue
        if name_stripped and pat == name_stripped:
            match_bonus = SCORE_WEIGHTS["name_exact_match"]
        else:
            match_bonus = SCORE_WEIGHTS["partial_text_match"]
        break

    score += match_bonus

    entry_tags = entry.get("tags", [])
    if not isinstance(entry_tags, (list, tuple)):
        entry_tags = []
    tag_set = {t.strip().lower() for t in entry_tags if isinstance(t, str)}

    overlap = len(tag_set & wanted_tags)
    score += overlap * SCORE_WEIGHTS["tag_primary"]
    if overlap > 0:
        score += SCORE_WEIGHTS["tag_primary_bonus"]

    if expanded_tags:
        overlap_exp = len(tag_set & expanded_tags)
        score += SCORE_WEIGHTS["tag_expanded"] * overlap_exp

    return (score, -idx)

# Для обратной совместимости с прежним именем
_score_entry = score_rule_entry

# ============================================================================
# Вспомогательные функции для ranked‑выбора
# ============================================================================


def _log_selection_debug(
    debug_context: str,
    candidates: List[Dict[str, Any]],
    scored: List[Tuple[int, int, Dict[str, Any]]],
    limit: int,
) -> None:
    """Логирует диагностику ранжирования, если уровень DEBUG активен."""
    if not logging.getLogger().isEnabledFor(logging.DEBUG):
        return
    if not scored:
        logging.debug(f"[{debug_context}] No scored items (all below threshold).")
        return
    top_info = []
    for s in scored[:5]:
        entry = s[2]
        score_val = s[0]
        name = entry.get("name", entry.get("wrong", "?"))[:30]
        if score_val >= SCORE_WEIGHTS["wrong_exact_match"]:
            reason = "text_match"
        elif score_val >= SCORE_WEIGHTS["partial_text_match"]:
            reason = "partial_text"
        elif score_val >= SCORE_WEIGHTS["tag_primary"]:
            reason = "tags"
        else:
            reason = "fallback"
        top_info.append((score_val, name, reason))
    logging.debug(
        f"[{debug_context}] Candidates: {len(candidates)}, "
        f"selected: {min(limit, len(scored))}, "
        f"top scores: {top_info}"
    )
    if len(scored) > limit:
        missed = scored[limit : limit + 2]
        missed_info = [(s[0], s[2].get("name", "?")[:30]) for s in missed]
        logging.debug(f"[{debug_context}] Missed due to limit: {missed_info}")


def _make_dedupe_key(entry: Dict[str, Any]) -> Tuple[Any, ...]:
    if "id" in entry:
        return ("id", entry["id"])
    return (
        entry.get("wrong", ""),
        entry.get("rule", ""),
        entry.get("description", ""),
        entry.get("name", ""),
    )


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
) -> List[Dict[str, Any]]:
    """
    Общая функция ранжирования записей.

    - candidate_limit: сколько записей рассматривать (None = все).
    - min_score: минимальный балл, чтобы попасть в ranked‑путь.
      По умолчанию None – все записи проходят.
    - debug_context: метка для диагностики.
    """
    if not entries:
        return []

    candidates = entries if candidate_limit is None else entries[:candidate_limit]

    wanted_set = {t.strip().lower() for t in wanted_tags if isinstance(t, str)}
    scored: List[Tuple[int, int, Dict[str, Any]]] = []
    for idx, entry in enumerate(candidates):
        score, tie = scorer(
            entry, normalized_text, wanted_set, idx, expanded_tags=expanded_tags
        )
        if require_text_match and score < SCORE_WEIGHTS["wrong_exact_match"]:
            continue
        if min_score is not None and score < min_score:
            continue
        scored.append((score, tie, entry))

    if not scored:
        if require_text_match:
            if debug_context:
                logging.debug(
                    f"[{debug_context}] require_text_match=True, "
                    f"no text matches found, returning []"
                )
            return []

# СТАЛО — fallback только по записям с тегами, иначе молчим
fallback_candidates: List[Tuple[int, int, int, Dict[str, Any]]] = []
for idx, entry in enumerate(candidates):
    entry_tags = entry.get("tags", [])
    if not isinstance(entry_tags, (list, tuple)):
        entry_tags = []
    tag_set = {t.strip().lower() for t in entry_tags if isinstance(t, str)}
    overlap = len(tag_set & wanted_set)
    if overlap == 0:  # ← ключевое изменение: без overlap — пропускаем
        continue
    info = _entry_info_score(entry)
    fallback_candidates.append((overlap, info, -idx, entry))

if not fallback_candidates:
    if debug_context:
        logging.debug(
            f"[{debug_context}] Fallback: no tag overlap found, "
            f"returning [] (silence over noise)."
        )
    return []

fallback_candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)

if debug_context:
    logging.debug(
        f"[{debug_context}] Fallback (tag-only): "
        f"{len(fallback_candidates)} candidates with tag overlap, "
        f"top overlap={fallback_candidates[0][0]}"
    )

result: List[Dict[str, Any]] = []
seen_keys: Set[Tuple[Any, ...]] = set()
for _, _, _, entry in fallback_candidates:
    key = _make_dedupe_key(entry)
    if key in seen_keys:
        continue
    seen_keys.add(key)
    result.append(entry)
    if len(result) >= limit:
        break
return result

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    _log_selection_debug(debug_context, candidates, scored, limit)

    result: List[Dict[str, Any]] = []
    seen_keys: Set[Tuple[Any, ...]] = set()
    for _, _, entry in scored:
        key = _make_dedupe_key(entry)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        result.append(entry)
        if len(result) >= limit:
            break

    return result


def _select_by_tags_or_all(
    entries: List[Dict[str, Any]],
    tags: Iterable[str],
    limit: int,
    expanded_tags: Optional[Set[str]] = None,
    min_score: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Обёртка для структурных записей без текстового матчинга:
    ранжируем только по тегам и информативности.
    """
    normalized_text = ""
    return _select_ranked_entries(
        entries,
        normalized_text,
        tags,
        limit,
        scorer=score_structural_entry,
        debug_context="tags_or_all",
        expanded_tags=expanded_tags,
        min_score=min_score,
    )

# ============================================================================
# Публичные селекторы (API, совместимый с прежним кодом)
# ============================================================================


# СТАЛО
def select_grammar_rules(
    kb: Any,
    text: str,
    tags: Iterable[str],
    limit: int = 10,
    candidate_limit: Optional[int] = None,
    min_score: int = 1,
) -> List[Dict[str, Any]]:
    normalized_text = normalize_text_for_match(text)
    effective_tags = list(tags) or ["grammar"]
    return _select_ranked_entries(
        kb.grammar_errors,
        normalized_text,
        effective_tags,
        limit,
        scorer=score_rule_entry,
        candidate_limit=candidate_limit,
        debug_context="grammar",
        min_score=min_score,
    )


def select_style_issues(
    kb: Any,
    text: str,
    tags: Iterable[str],
    limit: int = 10,
    candidate_limit: Optional[int] = None,
    min_score: int = 1,
) -> List[Dict[str, Any]]:
    normalized_text = normalize_text_for_match(text)
    effective_tags = list(tags) or ["style"]
    return _select_ranked_entries(
        kb.stylistic_issues,
        normalized_text,
        effective_tags,
        limit,
        scorer=score_rule_entry,
        candidate_limit=candidate_limit,
        debug_context="style",
        min_score=min_score,
    )


def select_logic_issues(
    kb: Any,
    text: str,
    tags: Iterable[str],
    limit: int = 8,
    candidate_limit: Optional[int] = None,
    min_score: int = 1,
) -> List[Dict[str, Any]]:
    normalized_text = normalize_text_for_match(text)
    wanted_tags = list(tags) + ["logic"]
    candidates: List[Dict[str, Any]] = (
        kb.logic_issues if kb.logic_issues else kb.stylistic_issues + kb.grammar_errors
    )
    return _select_ranked_entries(
        candidates,
        normalized_text,
        wanted_tags,
        limit,
        scorer=score_rule_entry,
        candidate_limit=candidate_limit,
        debug_context="logic",
        min_score=min_score,
    )


# Экспортируем частично приватный селектор для использования в PromptBuilder
select_structural_by_tags_or_all = _select_by_tags_or_all