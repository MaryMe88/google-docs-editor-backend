"""
prompt_builder.py

Модуль для сборки финальных промптов из конфигов и базы знаний.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TypedDict, Union

from src.tag_registry import build_known_tags, normalize_tag, normalize_tags
from src.config_types import (
    LimitsConfig,
    KnowledgeBudget,
    KnowledgeBudgetManager,
    CachePolicy,
    FileCache,
)

logger = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class CoreConfig:
    """Базовая конфигурация редактора."""

    role: str
    priorities: str
    basic_audit_instructions: List[str]
    forbidden: List[str]


@dataclass(frozen=True)
class DomainConfig:
    """Конфигурация домена."""

    name: str
    system_rules: str
    tone: str
    allow_storytelling: bool = True
    allow_marketing: bool = True


@dataclass(frozen=True)
class IntentConfig:
    """Конфигурация цели обработки."""

    name: str
    instructions: List[str]


@dataclass(frozen=True)
class OverlayConfig:
    """Конфигурация оверлея."""

    name: str
    instructions: List[str]


@dataclass(frozen=True)
class AudienceProfile:
    """Профиль аудитории."""

    kind: str
    expertise: str
    formality: str
    description: str = ""


@dataclass(frozen=True)
class KnowledgeBase:
    """База знаний редактора."""

    stop_words: Dict[str, List[str]]
    grammar_errors: List[RuleEntry]
    stylistic_issues: List[RuleEntry]
    logic_issues: List[RuleEntry]
    storytelling_frameworks: List[StructuralEntry]
    marketing_templates: List[StructuralEntry]
    domain_glossary: Dict[str, Any]
    composition_principles: List[StructuralEntry]
    local_cohesion: List[StructuralEntry]
    composition_errors: List[StructuralEntry]
    rhetoric_frameworks: List[StructuralEntry]
    editorial_techniques: List[EditorialTechniqueEntry]
    nkrj_structure_patterns: Dict[str, Any]


def load_json_file(path: Path) -> dict:
    """Загружает JSON-файл."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional_json(path: Path, default: Any = None) -> Any:
    """Загружает JSON из файла, если он существует."""
    if path.exists():
        return load_json_file(path)
    return default


def load_core_config(base_path: Path = Path("config")) -> CoreConfig:
    """Загружает core config."""
    data = load_json_file(base_path / "core.json")
    return CoreConfig(
        role=data["role"],
        priorities=data["priorities"],
        basic_audit_instructions=data["basic_audit_instructions"],
        forbidden=data["forbidden"],
    )


def load_domain_config(domain: str, base_path: Path = Path("config")) -> DomainConfig:
    """Загружает config домена."""
    data = load_json_file(base_path / "domains" / f"{domain}.json")
    return DomainConfig(
        name=data["name"],
        system_rules=data["system_rules"],
        tone=data["tone"],
        allow_storytelling=data.get("allow_storytelling", True),
        allow_marketing=data.get("allow_marketing", True),
    )


def load_intent_config(
    intent: Optional[str],
    base_path: Path = Path("config"),
) -> Optional[IntentConfig]:
    """Загружает config intent."""
    if intent is None or intent == "neutral":
        return None
    data = load_json_file(base_path / "intents" / f"{intent}.json")
    return IntentConfig(
        name=data["name"],
        instructions=data["instructions"],
    )


def load_overlay_config(
    overlay: str,
    base_path: Path = Path("config"),
) -> OverlayConfig:
    """Загружает config overlay."""
    data = load_json_file(base_path / "overlays" / f"{overlay}.json")
    return OverlayConfig(
        name=data["name"],
        instructions=data["instructions"],
    )


def load_overlay_configs(
    overlays: Sequence[str],
    base_path: Path = Path("config"),
) -> List[OverlayConfig]:
    """Legacy helper для загрузки нескольких overlays."""
    return [load_overlay_config(ov, base_path) for ov in overlays]


def load_output_format(
    mode: str,
    base_path: Path = Path("config"),
) -> str:
    """Загружает шаблон формата ответа."""
    data = load_json_file(base_path / "output_format.json")
    return data.get(mode, data["text_only"])


def _flatten_examples_block(
    items: List[Dict[str, Any]],
    category: str = "",
) -> List[FlatEntry]:
    """Разворачивает блок examples в плоский список записей."""
    flat: List[FlatEntry] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if "examples" in item:
            examples = item.get("examples")
            if not isinstance(examples, list):
                flat.append(item)
                continue
            cat = item.get("category", category)
            for example in examples:
                if not isinstance(example, dict):
                    continue
                entry = dict(example)
                if "tags" not in entry:
                    entry["tags"] = ["style"]
                entry["category"] = cat
                flat.append(entry)
        else:
            flat.append(item)
    return flat


def _flatten_stylistic_issues(raw: Dict[str, Any]) -> List[FlatEntry]:
    """Разворачивает stylistic_issues.json."""
    flat: List[FlatEntry] = []
    flat.extend(_flatten_examples_block(raw.get("stylistic_errors", [])))
    flat.extend(_flatten_examples_block(raw.get("common_issues", [])))
    return flat


def _flatten_editorial_techniques(raw: Dict[str, Any]) -> List[EditorialTechniqueEntry]:
    """Разворачивает editorial_techniques.json."""
    flat: List[EditorialTechniqueEntry] = []
    for block in raw.get("editorial_techniques", []):
        category = block.get("category", "")
        block_tags = block.get("tags", [])
        techniques = block.get("techniques", [])
        if not isinstance(techniques, list):
            continue
        for tech in techniques:
            tech_id = tech.get("id", "")
            name = tech.get("name", "")
            desc = tech.get("description", "")
            when_to_use = tech.get("when_to_use", [])
            how_to_apply = tech.get("how_to_apply", [])
            tags = list(block_tags) + list(tech.get("tags", []))
            source = tech.get("source", {})
            examples = tech.get("examples", [])
            if examples and isinstance(examples, list):
                example = examples[0]
                wrong = example.get("wrong", "")
                correct = example.get("correct", "")
                explanation = example.get("explanation", "")
            else:
                wrong = ""
                correct = ""
                explanation = ""

            flat.append(
                {
                    "id": tech_id,
                    "name": name,
                    "category": category,
                    "description": desc,
                    "when_to_use": when_to_use,
                    "how_to_apply": how_to_apply,
                    "example_wrong": wrong,
                    "example_correct": correct,
                    "example_explanation": explanation,
                    "tags": normalize_tags(tags) if tags else normalize_tags(["editing", "noragal"]),
                    "source": source,
                }
            )
    return flat


def normalize_entry_tags_inplace(entry: Dict[str, Any]) -> None:
    """Нормализует поле tags у KB-записи inplace."""
    raw_tags = entry.get("tags")
    if isinstance(raw_tags, list):
        entry["tags"] = normalize_tags(raw_tags)


def normalize_entries_tags(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Нормализует tags у всех записей."""
    for entry in entries:
        if isinstance(entry, dict):
            normalize_entry_tags_inplace(entry)
    return entries


def load_knowledge_base(base_path: Path = Path("knowledge_base")) -> KnowledgeBase:
    """Загружает базу знаний."""
    stop_words = load_json_file(base_path / "stop_words.json")
    grammar = load_json_file(base_path / "grammar_errors.json")
    style_raw = load_json_file(base_path / "stylistic_issues.json")
    storytelling = load_json_file(base_path / "storytelling_frameworks.json")
    marketing = load_json_file(base_path / "marketing_templates.json")
    logic_data = _load_optional_json(base_path / "logic_issues.json", {"issues": []})
    domain_glossary = _load_optional_json(base_path / "domain_glossary.json", {})
    composition_principles_raw = _load_optional_json(base_path / "composition_principles.json", {})
    local_cohesion_raw = _load_optional_json(base_path / "local_cohesion.json", {})
    composition_errors_raw = _load_optional_json(base_path / "composition_errors.json", {})
    rhetoric_raw = _load_optional_json(base_path / "rhetoric.json", {})
    editorial_raw = _load_optional_json(base_path / "editorial_techniques.json", {})
    structure_data = _load_optional_json(base_path / "nkrj_structure_patterns.json", {})

    return KnowledgeBase(
        stop_words=stop_words,
        grammar_errors=normalize_entries_tags(grammar.get("common_mistakes", [])),
        stylistic_issues=normalize_entries_tags(_flatten_stylistic_issues(style_raw)),
        logic_issues=normalize_entries_tags(logic_data.get("issues", [])),
        storytelling_frameworks=normalize_entries_tags(storytelling.get("frameworks", [])),
        marketing_templates=normalize_entries_tags(marketing.get("templates", [])),
        domain_glossary=domain_glossary,
        composition_principles=normalize_entries_tags(
            composition_principles_raw.get("composition_principles", [])
        ),
        local_cohesion=normalize_entries_tags(local_cohesion_raw.get("local_cohesion", [])),
        composition_errors=normalize_entries_tags(
            composition_errors_raw.get("composition_errors", [])
        ),
        rhetoric_frameworks=normalize_entries_tags(rhetoric_raw.get("frameworks", [])),
        editorial_techniques=normalize_entries_tags(
            _flatten_editorial_techniques(editorial_raw) if editorial_raw else []
        ),
        nkrj_structure_patterns=structure_data,
    )


# --- KB file list (для mtime multi-check) ---
_KB_FILES = [
    "stop_words.json",
    "grammar_errors.json",
    "stylistic_issues.json",
    "storytelling_frameworks.json",
    "marketing_templates.json",
    "logic_issues.json",
    "domain_glossary.json",
    "composition_principles.json",
    "local_cohesion.json",
    "composition_errors.json",
    "rhetoric.json",
    "editorial_techniques.json",
    "nkrj_structure_patterns.json",
]


def _normalize_text_for_match(text: str) -> str:
    """Нормализует текст для поиска."""
    text = text.replace("ё", "е").replace("Ё", "Е")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def _contains_pattern(normalized_text: str, pattern: str) -> bool:
    """Проверяет вхождение паттерна в текст."""
    if not pattern:
        return False
    norm_pattern = _normalize_text_for_match(pattern)
    if not norm_pattern or len(norm_pattern) < 2:
        return False
    if " " not in norm_pattern:
        return re.search(rf"\b{re.escape(norm_pattern)}\b", normalized_text) is not None
    return norm_pattern in normalized_text


def _get_entry_match_patterns(entry: RuleEntry) -> List[str]:
    """Собирает паттерны для матчинга."""
    patterns: List[str] = []
    seen = set()
    for field in ("wrong", "name", "rule", "description"):
        val = entry.get(field)
        if isinstance(val, str):
            stripped = val.strip()
            if stripped and stripped not in seen:
                seen.add(stripped)
                patterns.append(stripped)
    return patterns


def _entry_info_score(entry: Dict[str, Any]) -> int:
    """Оценка информативности записи."""
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


def _score_rule_entry(
    entry: RuleEntry,
    normalized_text: str,
    wanted_tags: Set[str],
    idx: int,
    expanded_tags: Optional[Set[str]] = None,
) -> Tuple[int, int]:
    """Скоринг для grammar/style/logic."""
    score = 0
    match_patterns = _get_entry_match_patterns(entry)
    if match_patterns:
        if _contains_pattern(normalized_text, match_patterns[0]):
            score += 1000
        else:
            for pat in match_patterns[1:]:
                if _contains_pattern(normalized_text, pat):
                    score += 200
                    break

    entry_tags = entry.get("tags", [])
    if not isinstance(entry_tags, (list, tuple)):
        entry_tags = []
    tag_set = {normalize_tag(t) for t in entry_tags if isinstance(t, str)}

    overlap = len(tag_set & wanted_tags)
    score += overlap * 10
    if overlap > 0:
        score += 1

    if expanded_tags:
        overlap_exp = len(tag_set & expanded_tags)
        score += overlap_exp * 2

    return score, -idx


def _score_structural_entry(
    entry: StructuralEntry,
    normalized_text: str,
    wanted_tags: Set[str],
    idx: int,
    expanded_tags: Optional[Set[str]] = None,
) -> Tuple[int, int]:
    """Скоринг для storytelling/marketing/composition/rhetoric/editorial."""
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
                patterns.append(item.strip())

    for container_key in ("steps", "sections"):
        container = entry.get(container_key)
        if isinstance(container, list):
            for step in container:
                if isinstance(step, dict):
                    step_name = step.get("name")
                    if isinstance(step_name, str) and step_name.strip():
                        patterns.append(step_name.strip())
                    step_desc = step.get("description")
                    if isinstance(step_desc, str) and step_desc.strip():
                        patterns.append(step_desc.strip())

    seen: Set[str] = set()
    unique_patterns: List[str] = []
    for pattern in patterns:
        if pattern not in seen:
            seen.add(pattern)
            unique_patterns.append(pattern)

    match_bonus = 0
    for pattern in unique_patterns:
        if _contains_pattern(normalized_text, pattern):
            if pattern == entry.get("name", "").strip():
                match_bonus = 500
            else:
                match_bonus = 200
            break

    score += match_bonus

    entry_tags = entry.get("tags", [])
    if not isinstance(entry_tags, (list, tuple)):
        entry_tags = []
    tag_set = {normalize_tag(t) for t in entry_tags if isinstance(t, str)}

    overlap = len(tag_set & wanted_tags)
    score += overlap * 10
    if overlap > 0:
        score += 1

    if expanded_tags:
        overlap_exp = len(tag_set & expanded_tags)
        score += overlap_exp * 2

    return score, -idx


_score_entry = _score_rule_entry


def _log_selection_debug(
    debug_context: str,
    candidates: List[Dict[str, Any]],
    scored: List[Tuple[int, int, Dict[str, Any]]],
    limit: int,
) -> None:
    """Логирует диагностику ранжирования."""
    if not logging.getLogger().isEnabledFor(logging.DEBUG):
        return
    if not scored:
        logging.debug("[%s] No scored items (all below threshold).", debug_context)
        return

    top_info = []
    for score_value, _, entry in scored[:5]:
        name = entry.get("name", entry.get("wrong", "?"))[:30]
        if score_value >= 1000:
            reason = "text_match"
        elif score_value >= 200:
            reason = "partial_text"
        elif score_value >= 10:
            reason = "tags"
        else:
            reason = "fallback"
        top_info.append((score_value, name, reason))

    logging.debug(
        "[%s] Candidates: %s, selected: %s, top scores: %s",
        debug_context,
        len(candidates),
        min(limit, len(scored)),
        top_info,
    )

    if len(scored) > limit:
        missed = scored[limit: limit + 2]
        missed_info = [(item[0], item[2].get("name", "?")[:30]) for item in missed]
        logging.debug("[%s] Missed due to limit: %s", debug_context, missed_info)


def _make_dedupe_key(entry: Dict[str, Any]) -> Tuple[Any, ...]:
    """Строит ключ дедупликации записи."""
    if "id" in entry:
        return "id", entry["id"]
    return (
        entry.get("wrong", ""),
        entry.get("rule", ""),
        entry.get("description", ""),
        entry.get("name", ""),
    )


def _truncate_entries_by_chars(
    entries: List[Dict[str, Any]],
    char_budget: Optional[int],
) -> List[Dict[str, Any]]:
    """
    Обрезает список записей до символьного бюджета.
    Гарантирует, что хотя бы одна запись всегда включается.
    """
    if char_budget is None or not entries:
        return entries
    result: List[Dict[str, Any]] = []
    used = 0
    for entry in entries:
        entry_len = len(str(entry))
        if result and used + entry_len > char_budget:
            break
        result.append(entry)
        used += entry_len
    return result


# ---------------------------------------------------------------------------
# ФП-2: Quality-gated fallback — пороги стадий
# ---------------------------------------------------------------------------
#
# Стадии срабатывают последовательно; каждая следующая — мягче предыдущей.
# Если стадия вернула >= 1 результат — останавливаемся.
#
# Stage A  text_match (≥1000) AND tag_overlap (≥10)  → min_score = 1010
# Stage B  text_match (≥1000) OR  tag_overlap (≥10)  → min_score =  200
# Stage C  tag_overlap only                           → min_score =   10
# Stage D  нейтральные записи с высоким info_score   → min_score =    0
# Stage E  абсолютный fallback — топ-N по info_score → без порога
#
_FALLBACK_STAGES: List[Tuple[str, int]] = [
    ("A: text+tags",  1010),
    ("B: text|tags",   200),
    ("C: tags_only",    10),
    ("D: neutral",       0),
]

_STAGE_E_LABEL = "E: absolute"

# ---------------------------------------------------------------------------
# ФП-3: Dual validation — mandatory vs optional блоки KB
# ---------------------------------------------------------------------------
#
# Mandatory: падение блока = падение всего build_prompt() → fail fast.
# Optional:  падение блока = warning + пустая строка → graceful degradation.
#
_MANDATORY_KB_BLOCKS: frozenset = frozenset({"grammar", "style", "logic", "stop_words"})
_OPTIONAL_KB_BLOCKS: frozenset = frozenset({
    "storytelling", "marketing", "composition", "cohesion",
    "composition_errors", "rhetoric", "editorial", "nkrj", "glossary",
})


def _collect_deduped(
    scored: List[Tuple[int, int, Dict[str, Any]]],
    limit: int,
) -> List[Dict[str, Any]]:
    """Извлекает дедуплицированный список из отсортированного scored."""
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


def _select_ranked_entries(
    entries: List[Dict[str, Any]],
    normalized_text: str,
    wanted_tags: Iterable[str],
    limit: int,
    require_text_match: bool = False,
    scorer: Any = _score_rule_entry,
    candidate_limit: Optional[int] = None,
    debug_context: str = "",
    expanded_tags: Optional[Set[str]] = None,
    min_score: Optional[int] = None,
    char_budget: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Ранжирует записи с качественно-уровневым fallback (ФП-2).

    Алгоритм:
    1. Скоринг всех кандидатов через scorer().
    2. Попытка Stage A→D с убывающим min_score порогом.
       Первая стадия, давшая ≥ 1 результат, используется.
    3. Если все стадии пусты — Stage E: топ-N по info_score без фильтра.
    4. Символьный бюджет применяется к финальному результату.

    Диагностика: при DEBUG-логировании выводит использованную стадию
    и причину выбора каждой записи из топ-5.
    """
    if not entries:
        return []

    candidates = entries if candidate_limit is None else entries[:candidate_limit]
    wanted_set = {normalize_tag(tag) for tag in wanted_tags if isinstance(tag, str)}

    # --- Полный скоринг всех кандидатов ---
    all_scored: List[Tuple[int, int, Dict[str, Any]]] = []
    for idx, entry in enumerate(candidates):
        score, tie = scorer(
            entry,
            normalized_text,
            wanted_set,
            idx,
            expanded_tags=expanded_tags,
        )
        if require_text_match and score < 1000:
            continue
        all_scored.append((score, tie, entry))

    all_scored.sort(key=lambda item: (item[0], item[1]), reverse=True)

    # Внешний min_score — «жёсткий» порог вызывающего кода;
    # перекрывает stage-пороги снизу, но не выше Stage A.
    effective_floor = min_score if min_score is not None else 0

    # --- Stage A→D ---
    used_stage: Optional[str] = None
    result: List[Dict[str, Any]] = []

    for stage_label, stage_threshold in _FALLBACK_STAGES:
        threshold = max(stage_threshold, effective_floor)
        filtered = [item for item in all_scored if item[0] >= threshold]
        if filtered:
            result = _collect_deduped(filtered, limit)
            used_stage = stage_label
            break

    # --- Stage E: абсолютный fallback по info_score ---
    if not result:
        stage_e_candidates = [
            (_entry_info_score(entry), -idx, entry)
            for idx, entry in enumerate(candidates)
        ]
        stage_e_candidates.sort(reverse=True)
        seen_keys: Set[Tuple[Any, ...]] = set()
        for _, _, entry in stage_e_candidates:
            key = _make_dedupe_key(entry)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            result.append(entry)
            if len(result) >= limit:
                break
        used_stage = _STAGE_E_LABEL

    # --- Диагностика ---
    if debug_context and logging.getLogger().isEnabledFor(logging.DEBUG):
        top_items = [(s, e) for s, _, e in all_scored[:5]]
        top_info = []
        for score_value, entry in top_items:
            name = entry.get("name", entry.get("wrong", "?"))[:30]
            if score_value >= 1010:
                reason = "text+tags"
            elif score_value >= 1000:
                reason = "text_match"
            elif score_value >= 200:
                reason = "partial_text"
            elif score_value >= 10:
                reason = "tags_only"
            elif score_value > 0:
                reason = "weak_signal"
            else:
                reason = "neutral"
            top_info.append((score_value, name, reason))

        logging.debug(
            "[%s] stage=%s | candidates=%s | selected=%s | top=%s",
            debug_context,
            used_stage,
            len(candidates),
            len(result),
            top_info,
        )

        if all_scored and len(result) < limit:
            missed_count = len(all_scored) - len(result)
            logging.debug(
                "[%s] Missed due to stage threshold or limit: %s entries",
                debug_context,
                missed_count,
            )

    return _truncate_entries_by_chars(result, char_budget)


def _match_tags(entry_tags: Iterable[str], wanted_tags: Iterable[str]) -> bool:
    """Проверяет пересечение тегов."""
    entry = {normalize_tag(tag) for tag in (entry_tags or [])}
    wanted = {normalize_tag(tag) for tag in (wanted_tags or [])}
    return bool(entry & wanted) if wanted else True


def select_grammar_rules(
    kb: KnowledgeBase,
    text: str,
    tags: Iterable[str],
    limit: int = 10,
    candidate_limit: Optional[int] = None,
    min_score: int = 1,
) -> List[Dict[str, Any]]:
    """Публичный селектор grammar."""
    normalized_text = _normalize_text_for_match(text)
    return _select_ranked_entries(
        kb.grammar_errors,
        normalized_text,
        tags,
        limit,
        scorer=_score_rule_entry,
        candidate_limit=candidate_limit,
        debug_context="grammar",
        min_score=min_score,
    )


def select_style_issues(
    kb: KnowledgeBase,
    text: str,
    tags: Iterable[str],
    limit: int = 10,
    candidate_limit: Optional[int] = None,
    min_score: int = 1,
) -> List[Dict[str, Any]]:
    """Публичный селектор style."""
    normalized_text = _normalize_text_for_match(text)
    return _select_ranked_entries(
        kb.stylistic_issues,
        normalized_text,
        tags,
        limit,
        scorer=_score_rule_entry,
        candidate_limit=candidate_limit,
        debug_context="style",
        min_score=min_score,
    )


def select_logic_issues(
    kb: KnowledgeBase,
    text: str,
    tags: Iterable[str],
    limit: int = 8,
    candidate_limit: Optional[int] = None,
    min_score: int = 1,
) -> List[Dict[str, Any]]:
    """Публичный селектор logic."""
    normalized_text = _normalize_text_for_match(text)
    wanted_tags = list(tags) + ["logic"]
    candidates = kb.logic_issues if kb.logic_issues else kb.stylistic_issues + kb.grammar_errors
    return _select_ranked_entries(
        candidates,
        normalized_text,
        wanted_tags,
        limit,
        scorer=_score_rule_entry,
        candidate_limit=candidate_limit,
        debug_context="logic",
        min_score=min_score,
    )


def _select_by_tags_or_all(
    entries: List[Dict[str, Any]],
    tags: Iterable[str],
    limit: int,
    expanded_tags: Optional[Set[str]] = None,
    min_score: Optional[int] = None,
    char_budget: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Селектор по тегам для структурных блоков."""
    normalized_text = ""
    return _select_ranked_entries(
        entries,
        normalized_text,
        tags,
        limit,
        scorer=_score_structural_entry,
        debug_context="tags_or_all",
        expanded_tags=expanded_tags,
        min_score=min_score,
        char_budget=char_budget,
    )


def _safe_float(value: Any) -> Optional[float]:
    """Безопасно приводит значение к float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_nkrj_norms_lines(
    kb: KnowledgeBase,
    limit_sources: int = 4,
) -> List[str]:
    """Строит компактный блок норм по NKRJ/Taiga."""
    raw = kb.nkrj_structure_patterns
    if not raw:
        return []

    lines: List[str] = []

    corpus = raw.get("corpus")
    if corpus:
        lines.append(f" • Корпус-ориентир: {corpus}.")

    aggregate = raw.get("aggregate_norms", {})
    norm_sentence = aggregate.get("norm_sentence_length", {})
    thresholds = aggregate.get("thresholds", {})

    avg = _safe_float(norm_sentence.get("avg"))
    variation = _safe_float(norm_sentence.get("variation_coeff"))
    short_share = _safe_float(norm_sentence.get("short_share"))
    medium_share = _safe_float(norm_sentence.get("medium_share"))
    long_share = _safe_float(norm_sentence.get("long_share"))

    if avg is not None:
        lines.append(
            f" • Ориентир по длине предложения: в среднем около {avg:.2f} слов; "
            "держи фразы преимущественно короткими и средними."
        )

    if short_share is not None and medium_share is not None and long_share is not None:
        lines.append(
            f" • Распределение длины предложений: короткие ≈ {short_share:.1%}, "
            f"средние ≈ {medium_share:.1%}, длинные ≈ {long_share:.1%}; "
            "не перегружай текст длинными периодами."
        )

    if variation is not None:
        lines.append(
            f" • Коэффициент вариативности длины предложений — около {variation:.2f}; "
            "избегай монотонного ритма и чередуй длину фраз."
        )

    flat_paragraph = _safe_float(aggregate.get("norm_flat_paragraph_share"))
    if flat_paragraph is not None:
        lines.append(
            f" • Плоские абзацы почти не встречаются: норма flat paragraph share ≈ "
            f"{flat_paragraph:.2%}; абзацы должны двигать мысль."
        )

    passive_rate = _safe_float(aggregate.get("norm_passive_rate"))
    if passive_rate is not None:
        lines.append(
            f" • Ориентир по пассиву: около {passive_rate:.2f} на 100 строк; "
            "предпочитай активные конструкции."
        )

    deepr_rate = _safe_float(aggregate.get("norm_deepr_rate"))
    if deepr_rate is not None:
        lines.append(
            f" • Глубокие шаблонные клише почти отсутствуют: норма ≈ {deepr_rate:.2f} "
            "на 100 строк; избегай пластиковых связок."
        )

    plasticity_live = _safe_float(thresholds.get("plasticity_index_live"))
    plasticity_grey = _safe_float(thresholds.get("plasticity_index_grey_zone"))
    if plasticity_live is not None and plasticity_grey is not None:
        lines.append(
            f" • Индекс пластичности: до {plasticity_live:.1f} — живой текст, "
            f"около {plasticity_grey:.1f} и выше — зона риска искусственности."
        )

    sentence_variation_min = _safe_float(thresholds.get("sentence_variation_coeff_min"))
    if sentence_variation_min is not None:
        lines.append(
            f" • Минимально допустимая вариативность длины фраз: "
            f"{sentence_variation_min:.2f}."
        )

    short_sentence_share_min = _safe_float(thresholds.get("short_sentence_share_min"))
    if short_sentence_share_min is not None:
        lines.append(
            f" • Доля коротких предложений должна быть не ниже "
            f"{short_sentence_share_min:.1%}."
        )

    flat_alert = _safe_float(thresholds.get("flat_paragraph_share_alert"))
    if flat_alert is not None:
        lines.append(
            f" • Тревожный порог для плоских абзацев — {flat_alert:.0%}."
        )

    passive_alert = _safe_float(thresholds.get("passive_rate_alert"))
    if passive_alert is not None:
        lines.append(
            f" • Тревожный порог по пассиву — {passive_alert:.1f} на 100 строк."
        )

    sources = raw.get("sources", [])
    if isinstance(sources, list):
        for source_data in sources[:limit_sources]:
            if not isinstance(source_data, dict):
                continue
            source_name = str(source_data.get("source", "")).strip()
            sentence_data = source_data.get("sentence_length", {})
            source_avg = _safe_float(sentence_data.get("avg"))
            source_long = _safe_float(sentence_data.get("long_share"))
            source_passive = _safe_float(source_data.get("passive_rate_per_100_lines"))
            if source_name and source_avg is not None:
                line = f" • Источник {source_name}: средняя длина предложения ≈ {source_avg:.2f} слов"
                if source_long is not None:
                    line += f", длинных предложений ≈ {source_long:.1%}"
                if source_passive is not None:
                    line += f", пассив ≈ {source_passive:.2f} на 100 строк"
                line += "."
                lines.append(line)

    marker_examples: List[str] = []
    if isinstance(sources, list):
        for source_data in sources:
            markers = source_data.get("plastic_markers_per_1000_lines", {})
            if not isinstance(markers, dict):
                continue
            for marker, value in markers.items():
                score = _safe_float(value)
                if score is not None and score > 0:
                    marker_examples.append(f"{marker} ({score:.3f})")

    if marker_examples:
        unique_markers = list(dict.fromkeys(marker_examples))
        lines.append(
            " • Маркеры пластика, которые стоит особенно контролировать: "
            + ", ".join(unique_markers[:8])
            + "."
        )

    return lines


def _has_mode(
    intent: Optional[str],
    overlays: Sequence[str],
    aliases: Iterable[str],
) -> bool:
    """Проверяет, активирован ли режим по intent/overlay."""
    normalized_aliases = {normalize_tag(alias) for alias in aliases if isinstance(alias, str)}
    values = {normalize_tag(item) for item in overlays if isinstance(item, str)}
    if isinstance(intent, str):
        values.add(normalize_tag(intent))
    return bool(values & normalized_aliases)


CANONICAL_TAGS: Dict[str, Dict[str, Any]] = {
    "domains": {
        "marketing": {
            "primary": ["marketing"],
            "expanded": ["sales", "promo", "conversion"],
        },
        "blog": {
            "primary": ["blog"],
            "expanded": ["nonmarketing", "article", "educational"],
        },
        "deai": {
            "primary": ["deai"],
            "expanded": ["antiai", "humanize", "natural"],
        },
    },
    "intents": {
        "storytelling": {
            "primary": ["storytelling", "structure"],
            "expanded": ["narrative", "engagement"],
        },
        "noragal": {
            "primary": ["editing", "noragal"],
            "expanded": ["brevity", "clarity"],
        },
        "deai": {
            "primary": ["antiai", "humanize"],
            "expanded": ["authentic"],
        },
    },
    "overlays": {
        "logic": {
            "primary": ["logic"],
            "expanded": ["coherence", "argumentation"],
        },
        "factcheck": {
            "primary": ["factcheck"],
            "expanded": ["accuracy", "verification"],
        },
        "infostyle": {
            "primary": ["infostyle"],
            "expanded": ["clarity", "precision"],
        },
        "composition": {
            "primary": ["composition"],
            "expanded": [],
        },
        "cohesion": {
            "primary": ["cohesion"],
            "expanded": [],
        },
        "rhetoric": {
            "primary": ["rhetoric"],
            "expanded": [],
        },
        "marketingpush": {
            "primary": ["marketing"],
            "expanded": ["persuasion", "cta"],
        },
    },
}

KNOWN_TAGS: Set[str] = build_known_tags(CANONICAL_TAGS)

KNOWN_INTENTS: Set[str] = {
    "storytelling",
    "noragal",
    "deai",
    "neutral",
}

KNOWN_OVERLAYS: Set[str] = {
    "logic",
    "factcheck",
    "infostyle",
    "marketingpush",
    "composition",
    "cohesion",
    "rhetoric",
}

KB_TAGS_STRICT_VALIDATION: bool = False


def _get_canonical_tags_for_category(category: str, value: str) -> List[str]:
    """Возвращает primary + expanded теги."""
    norm_value = normalize_tag(value)
    data = CANONICAL_TAGS.get(category, {}).get(norm_value)
    if isinstance(data, dict):
        return normalize_tags(data.get("primary", []) + data.get("expanded", []))
    if isinstance(data, list):
        return normalize_tags(data)
    return normalize_tags([norm_value])


def _get_primary_tags_for_category(category: str, value: str) -> List[str]:
    """Возвращает primary теги."""
    norm_value = normalize_tag(value)
    data = CANONICAL_TAGS.get(category, {}).get(norm_value)
    if isinstance(data, dict):
        return normalize_tags(data.get("primary", []))
    return normalize_tags(data) if isinstance(data, list) else normalize_tags([norm_value])


def _get_expanded_tags_for_category(category: str, value: str) -> List[str]:
    """Возвращает expanded теги."""
    norm_value = normalize_tag(value)
    data = CANONICAL_TAGS.get(category, {}).get(norm_value)
    if isinstance(data, dict):
        return normalize_tags(data.get("expanded", []))
    return []


class PromptBuilder:
    """Собирает финальный промпт из конфигов, базы знаний и параметров запроса."""

    def __init__(
        self,
        config_path: Path = Path("config"),
        kb_path: Path = Path("knowledge_base"),
        limits: LimitsConfig = LimitsConfig(),
        token_budget: Optional[int] = None,
        cache_policy: Optional[CachePolicy] = None,
        enable_selection_diagnostics: bool = False,
    ) -> None:
        """
        Инициализирует PromptBuilder.

        Args:
            config_path: Путь к директории конфигов.
            kb_path: Путь к директории базы знаний.
            limits: Лимиты выдачи и кандидатов для всех блоков KB.
                    Пример: PromptBuilder(limits=LimitsConfig(grammar=5, style=5))
            token_budget: Токены под блок «База знаний» (None = без ограничений).
                          При задании активирует KnowledgeBudgetManager.
            cache_policy: Политика инвалидации кэша (ФП-1).
                          None → CachePolicy(check_mtime=True) — только mtime,
                          без TTL (рекомендуемый дефолт для prod).
                          Пример: CachePolicy(ttl_seconds=60, check_mtime=True)
            enable_selection_diagnostics: Включить DEBUG-логирование тегов и режимов.
        """
        self.config_path = config_path
        self.kb_path = kb_path
        self._limits = limits
        self._budget_manager = KnowledgeBudgetManager(token_budget=token_budget)
        self._cache_policy = cache_policy if cache_policy is not None else CachePolicy(check_mtime=True)
        self.enable_selection_diagnostics = enable_selection_diagnostics

        # FileCache — единый кэш-менеджер для всех конфигов и KB
        self._cache = FileCache(policy=self._cache_policy)

        # Кэши для доступных intents/overlays (сканирование директорий)
        self._available_intents_cache: Optional[Set[str]] = None
        self._available_overlays_cache: Optional[Set[str]] = None

    def reload_configs(self) -> None:
        """Принудительно сбрасывает весь кэш (все конфиги и KB)."""
        self._cache.invalidate()
        self._available_intents_cache = None
        self._available_overlays_cache = None
        logger.debug("PromptBuilder: full cache reset via reload_configs()")

    def _get_core_config(self) -> CoreConfig:
        path = self.config_path / "core.json"
        return self._cache.get_or_load("core", path, load_core_config, self.config_path)

    def _get_domain_config(self, domain: str) -> DomainConfig:
        path = self.config_path / "domains" / f"{domain}.json"
        return self._cache.get_or_load(
            f"domain:{domain}", path, load_domain_config, domain, self.config_path
        )

    def _get_output_format(self, mode: str) -> str:
        path = self.config_path / "output_format.json"
        return self._cache.get_or_load(
            f"output_format:{mode}", path, load_output_format, mode, self.config_path
        )

    def _get_overlay_config(self, overlay: str) -> OverlayConfig:
        path = self.config_path / "overlays" / f"{overlay}.json"
        return self._cache.get_or_load(
            f"overlay:{overlay}", path, load_overlay_config, overlay, self.config_path
        )

    def _get_intent_config(self, intent: Optional[str]) -> Optional[IntentConfig]:
        if intent is None or intent == "neutral":
            return None
        path = self.config_path / "intents" / f"{intent}.json"
        return self._cache.get_or_load(
            f"intent:{intent}", path, load_intent_config, intent, self.config_path
        )

    def _get_knowledge_base(self) -> KnowledgeBase:
        kb_files = [self.kb_path / f for f in _KB_FILES]
        return self._cache.get_or_load_multi(
            "kb", kb_files, load_knowledge_base, self.kb_path
        )

    def _get_available_intents(self) -> Set[str]:
        # Сканирование директории кэшируем отдельно — без mtime (дёшево)
        if self._available_intents_cache is None:
            intents_dir = self.config_path / "intents"
            if not intents_dir.exists():
                self._available_intents_cache = set()
            else:
                self._available_intents_cache = {path.stem for path in intents_dir.glob("*.json")}
        return self._available_intents_cache

    def _get_available_overlays(self) -> Set[str]:
        if self._available_overlays_cache is None:
            overlays_dir = self.config_path / "overlays"
            if not overlays_dir.exists():
                self._available_overlays_cache = set()
            else:
                self._available_overlays_cache = {path.stem for path in overlays_dir.glob("*.json")}
        return self._available_overlays_cache

    # --- Публичные методы, которые ждут main.py и контрактные тесты ---

    def get_core_config(self) -> CoreConfig:
        """Публичный доступ к core config."""
        return self._get_core_config()

    def get_knowledge_base(self) -> KnowledgeBase:
        """Публичный доступ к knowledge base."""
        return self._get_knowledge_base()

    def get_available_intents(self) -> Set[str]:
        """Публичный список доступных intents."""
        return set(self._get_available_intents())

    def get_available_overlays(self) -> Set[str]:
        """Публичный список доступных overlays."""
        return set(self._get_available_overlays())

    # --- Legacy алиасы для совместимости с кодом/тестами старого стиля ---

    def getcoreconfig(self) -> CoreConfig:
        """Legacy alias."""
        return self.get_core_config()

    def getknowledgebase(self) -> KnowledgeBase:
        """Legacy alias."""
        return self.get_knowledge_base()

    def getavailableintents(self) -> Set[str]:
        """Legacy alias."""
        return self.get_available_intents()

    def getavailableoverlays(self) -> Set[str]:
        """Legacy alias."""
        return self.get_available_overlays()

    def build(
        self,
        text: str,
        domain: str,
        intent: Optional[str] = None,
        audience: Optional[AudienceProfile] = None,
        overlays: Sequence[str] = (),
        output_mode: str = "text_only",
        include_knowledge: bool = True,
        **kwargs: Any,
    ) -> str:
        """
        Собирает финальный промпт.

        Поддерживает legacy kwargs:
        - outputmode -> output_mode
        - includeknowledge -> include_knowledge
        """
        if "outputmode" in kwargs and "output_mode" not in kwargs:
            output_mode = kwargs.pop("outputmode")
        if "includeknowledge" in kwargs and "include_knowledge" not in kwargs:
            include_knowledge = kwargs.pop("includeknowledge")
        if kwargs:
            unexpected = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")

        parts: List[str] = []
        parts.append(self._build_core_block())
        parts.append(self._build_domain_block(domain))

        if intent:
            intent_block = self._build_intent_block(intent)
            if intent_block:
                parts.append(intent_block)

        parts.append(self._build_audience_block(audience))

        if overlays:
            parts.append(self._build_overlays_block(overlays))

        if include_knowledge:
            parts.append(
                self._build_knowledge_block(
                    text=text,
                    domain=domain,
                    intent=intent,
                    overlays=overlays,
                )
            )

        parts.append(self._build_output_format_block(output_mode))
        parts.append(self._build_text_block(text))
        return "\n\n".join(parts)

    def _build_core_block(self) -> str:
        core = self._get_core_config()
        instructions = "\n".join(f"- {instruction}" for instruction in core.basic_audit_instructions)
        forbidden = "\n".join(f"❌ {rule}" for rule in core.forbidden)
        return (
            f"{core.role}\n\n"
            f"{core.priorities}\n\n"
            f"Задачи:\n{instructions}\n\n"
            f"Запреты:\n{forbidden}"
        )

    def _build_domain_block(self, domain: str) -> str:
        domain_cfg = self._get_domain_config(domain)
        return f"Домен: {domain_cfg.system_rules}\nТон: {domain_cfg.tone}"

    def _build_intent_block(self, intent: str) -> str:
        intent_cfg = self._get_intent_config(intent)
        if intent_cfg is None:
            return ""
        instructions = "\n".join(f"- {instruction}" for instruction in intent_cfg.instructions)
        return f"Цель обработки: {intent_cfg.name}\n\nТребования:\n{instructions}"

    def _build_audience_block(self, audience: Optional[AudienceProfile]) -> str:
        if audience is None:
            return "Аудитория: не указана. Используй нейтральный профессиональный тон."

        if not audience.description:
            kind_display = {
                "b2b": "B2B",
                "b2c": "B2C",
                "mixed": "смешанная",
                "custom": "особая",
            }.get(audience.kind, audience.kind)
            expertise_display = {
                "novice": "новички",
                "pro": "практики",
                "expert": "глубокие эксперты",
            }.get(audience.expertise, audience.expertise)
            formality_display = {
                "casual": "расслабленный",
                "neutral": "нейтральный",
                "formal": "официальный",
            }.get(audience.formality, audience.formality)
            return f"Аудитория: {kind_display}, {expertise_display}, {formality_display} тон."

        description_line = f"\n- Описание: {audience.description}" if audience.description else ""
        return (
            f"Аудитория:\n"
            f"- Тип: {audience.kind}\n"
            f"- Уровень экспертизы: {audience.expertise}\n"
            f"- Формальность: {audience.formality}"
            f"{description_line}"
        )

    def _build_overlays_block(self, overlays: Sequence[str]) -> str:
        parts: List[str] = ["Дополнительные режимы:"]
        for overlay in overlays:
            cfg = self._get_overlay_config(overlay)
            instructions = "\n".join(f" - {instruction}" for instruction in cfg.instructions)
            parts.append(f"\n• {cfg.name}:\n{instructions}")
        return "\n".join(parts)

    def _resolve_prompt_features(
        self,
        domain_cfg: DomainConfig,
        domain: str,
        intent: Optional[str],
        overlays: Sequence[str],
    ) -> Dict[str, Any]:
        """Централизованно определяет теги и режимы."""
        primary_tags: List[str] = []
        expanded_tags: List[str] = []

        primary_tags.extend(_get_primary_tags_for_category("domains", domain))
        expanded_tags.extend(_get_expanded_tags_for_category("domains", domain))

        available_intents = self._get_available_intents()
        available_overlays = self._get_available_overlays()

        if intent is not None:
            primary_tags.extend(_get_primary_tags_for_category("intents", intent))
            expanded_tags.extend(_get_expanded_tags_for_category("intents", intent))
            norm_intent = normalize_tag(intent)
            normalized_available_intents = {normalize_tag(item) for item in available_intents}
            if norm_intent not in KNOWN_INTENTS and norm_intent not in normalized_available_intents:
                logger.warning("Unknown intent '%s' passed to PromptBuilder", intent)

        for overlay in overlays:
            primary_tags.extend(_get_primary_tags_for_category("overlays", overlay))
            expanded_tags.extend(_get_expanded_tags_for_category("overlays", overlay))
            norm_overlay = normalize_tag(overlay)
            normalized_available_overlays = {normalize_tag(item) for item in available_overlays}
            if norm_overlay not in KNOWN_OVERLAYS and norm_overlay not in normalized_available_overlays:
                logger.warning("Unknown overlay '%s' passed to PromptBuilder", overlay)

        primary_set = {normalize_tag(tag) for tag in primary_tags if isinstance(tag, str)}
        expanded_set = {normalize_tag(tag) for tag in expanded_tags if isinstance(tag, str)}
        expanded_set -= primary_set

        storytelling_requested = _has_mode(
            intent,
            overlays,
            {"storytelling", "story", "narrative"},
        )
        marketing_requested = _has_mode(
            intent,
            overlays,
            {"marketingpush", "marketing", "sales"},
        )

        return {
            "tags": list(primary_set),
            "expanded_tags": list(expanded_set),
            "storytelling_enabled": domain_cfg.allow_storytelling and storytelling_requested,
            "marketing_enabled": domain_cfg.allow_marketing
            and (domain == "marketing" or marketing_requested),
        }

    def _build_grammar_style_logic_block(
        self,
        kb: KnowledgeBase,
        text: str,
        tags: List[str],
        expanded_tags: Set[str],
        budget: Optional[KnowledgeBudget] = None,
    ) -> str:
        grammar_limit = budget.grammar.entry_limit if budget else self._limits.grammar
        style_limit = budget.style.entry_limit if budget else self._limits.style
        logic_limit = budget.logic.entry_limit if budget else self._limits.logic
        grammar_char = budget.grammar.char_budget if budget else None
        style_char = budget.style.char_budget if budget else None
        logic_char = budget.logic.char_budget if budget else None

        grammar_sample = _select_ranked_entries(
            kb.grammar_errors,
            _normalize_text_for_match(text),
            tags,
            grammar_limit,
            scorer=_score_rule_entry,
            candidate_limit=self._limits.grammar_candidates,
            debug_context="grammar",
            expanded_tags=expanded_tags if expanded_tags else None,
            min_score=1,
            char_budget=grammar_char,
        )

        style_sample = _select_ranked_entries(
            kb.stylistic_issues,
            _normalize_text_for_match(text),
            tags,
            style_limit,
            scorer=_score_rule_entry,
            candidate_limit=self._limits.style_candidates,
            debug_context="style",
            expanded_tags=expanded_tags if expanded_tags else None,
            min_score=1,
            char_budget=style_char,
        )

        logic_sample = _select_ranked_entries(
            kb.logic_issues if kb.logic_issues else kb.stylistic_issues + kb.grammar_errors,
            _normalize_text_for_match(text),
            list(tags) + ["logic"],
            logic_limit,
            scorer=_score_rule_entry,
            candidate_limit=self._limits.logic_candidates,
            debug_context="logic",
            expanded_tags=expanded_tags if expanded_tags else None,
            min_score=1,
            char_budget=logic_char,
        )

        grammar_lines: List[str] = [
            f" • {err.get('wrong', '')} → {err.get('correct', '').strip()} "
            f"({err.get('rule', '').strip()})"
            for err in grammar_sample
            if err.get("wrong") and err.get("correct")
        ] or [" • (нет примеров в базе)"]

        style_lines: List[str] = [
            f" • {issue.get('wrong', '')} → {issue.get('correct', '').strip()} "
            f"({issue.get('rule', '').strip()})"
            for issue in style_sample
            if issue.get("wrong")
        ] or [" • (нет примеров в базе)"]

        logic_lines: List[str] = [
            f" • {item.get('name', item.get('wrong', 'Проблема'))}: "
            f"{item.get('rule', item.get('description', '')).strip()}"
            for item in logic_sample
        ] or [" • (нет логических правил в базе)"]

        return (
            "Типичные грамматические и лексические ошибки (исправляй по аналогии):\n"
            + "\n".join(grammar_lines)
            + "\n\nТипичные стилистические проблемы (канцелярит, штампы, вода — устраняй):\n"
            + "\n".join(style_lines)
            + "\n\nТипичные логические проблемы и риски связности:\n"
            + "\n".join(logic_lines)
        )

    def _build_composition_cohesion_errors_block(
        self,
        kb: KnowledgeBase,
        tags: List[str],
        expanded_tags: Set[str],
        budget: Optional[KnowledgeBudget] = None,
    ) -> str:
        comp_limit = budget.composition.entry_limit if budget else self._limits.composition
        coh_limit = budget.cohesion.entry_limit if budget else self._limits.cohesion
        cerr_limit = budget.composition_errors.entry_limit if budget else self._limits.composition_errors
        comp_char = budget.composition.char_budget if budget else None
        coh_char = budget.cohesion.char_budget if budget else None
        cerr_char = budget.composition_errors.char_budget if budget else None

        composition_principles_sample = _select_by_tags_or_all(
            kb.composition_principles,
            tags=tags + ["composition"],
            limit=comp_limit,
            expanded_tags=expanded_tags,
            min_score=1,
            char_budget=comp_char,
        )

        composition_principles_lines: List[str] = [
            f" • {entry.get('name', '')}: "
            f"{entry.get('rule', entry.get('description', '')).strip()}"
            for entry in composition_principles_sample
        ] or [" • (нет принципов композиции в базе)"]

        local_cohesion_sample = _select_by_tags_or_all(
            kb.local_cohesion,
            tags=tags + ["cohesion"],
            limit=coh_limit,
            expanded_tags=expanded_tags,
            min_score=1,
            char_budget=coh_char,
        )

        local_cohesion_lines: List[str] = [
            f" • {entry.get('name', '')}: "
            f"{entry.get('rule', entry.get('description', '')).strip()}"
            for entry in local_cohesion_sample
        ] or [" • (нет приёмов локальной связности в базе)"]

        composition_errors_sample = _select_by_tags_or_all(
            kb.composition_errors,
            tags=tags + ["composition"],
            limit=cerr_limit,
            expanded_tags=expanded_tags,
            min_score=1,
            char_budget=cerr_char,
        )

        composition_errors_lines: List[str] = [
            f" • {entry.get('name', '')}: "
            f"{entry.get('rule', entry.get('description', '')).strip()}"
            for entry in composition_errors_sample
        ] or [" • (нет примеров композиционных ошибок в базе)"]

        return (
            "Принципы композиции (типы построения и глобальная связность):\n"
            + "\n".join(composition_principles_lines)
            + "\n\nПриёмы локальной связности (абзац, тема-рема, местоимения, союзы):\n"
            + "\n".join(local_cohesion_lines)
            + "\n\nТипичные композиционные ошибки (что искать и как исправлять):\n"
            + "\n".join(composition_errors_lines)
        )

    def _build_nkrj_block(self, kb: KnowledgeBase, budget: Optional[KnowledgeBudget] = None) -> str:
        if budget is not None and not budget.nkrj.enabled:
            return ""
        nkrj_norms_lines = build_nkrj_norms_lines(kb)
        if not nkrj_norms_lines:
            return ""
        return (
            "\n\nНормы живого текста по корпусу Taiga Social Media "
            "(используй как статистический ориентир, а не как жёсткий шаблон):\n"
            + "\n".join(nkrj_norms_lines)
        )

    def _build_storytelling_block(
        self,
        kb: KnowledgeBase,
        text: str,
        tags: List[str],
        expanded_tags: Set[str],
        storytelling_enabled: bool,
        budget: Optional[KnowledgeBudget] = None,
    ) -> str:
        if not storytelling_enabled or not kb.storytelling_frameworks:
            return ""
        if budget is not None and not budget.storytelling.enabled:
            return ""

        st_limit = budget.storytelling.entry_limit if budget else self._limits.storytelling
        st_char = budget.storytelling.char_budget if budget else None

        normalized_text = _normalize_text_for_match(text)
        frameworks_sample = _select_ranked_entries(
            kb.storytelling_frameworks,
            normalized_text,
            tags + ["storytelling"],
            st_limit,
            require_text_match=False,
            scorer=_score_structural_entry,
            expanded_tags=expanded_tags if expanded_tags else None,
            candidate_limit=self._limits.storytelling_candidates,
            debug_context="storytelling",
            min_score=1,
            char_budget=st_char,
        )

        framework_lines: List[str] = []
        for framework in frameworks_sample:
            name = framework.get("name", "")
            steps = framework.get("steps", [])
            step_names = [
                step.get("name", "")
                for step in steps
                if isinstance(step, dict) and step.get("name")
            ]
            if name and step_names:
                framework_lines.append(f" • {name}: " + " → ".join(step_names))

        if not framework_lines:
            return ""

        return "\n\nФреймворки сторителлинга (для структуры рассказа):\n" + "\n".join(
            framework_lines
        )

    def _build_marketing_block(
        self,
        kb: KnowledgeBase,
        text: str,
        tags: List[str],
        expanded_tags: Set[str],
        marketing_enabled: bool,
        budget: Optional[KnowledgeBudget] = None,
    ) -> str:
        if not marketing_enabled or not kb.marketing_templates:
            return ""
        if budget is not None and not budget.marketing.enabled:
            return ""

        mk_limit = budget.marketing.entry_limit if budget else self._limits.marketing
        mk_char = budget.marketing.char_budget if budget else None

        normalized_text = _normalize_text_for_match(text)
        templates_sample = _select_ranked_entries(
            kb.marketing_templates,
            normalized_text,
            tags + ["marketing"],
            mk_limit,
            require_text_match=False,
            scorer=_score_structural_entry,
            expanded_tags=expanded_tags if expanded_tags else None,
            candidate_limit=self._limits.marketing_candidates,
            debug_context="marketing",
            min_score=1,
            char_budget=mk_char,
        )

        template_lines: List[str] = []
        for template in templates_sample:
            name = template.get("name", "")
            sections = template.get("sections", [])
            section_names = [
                section.get("name", "")
                for section in sections
                if isinstance(section, dict) and section.get("name")
            ]
            if name and section_names:
                template_lines.append(f" • {name}: " + ", ".join(section_names))

        if not template_lines:
            return ""

        return "\n\nМаркетинговые шаблоны (структура текста по типу):\n" + "\n".join(
            template_lines
        )

    def _build_rhetoric_editorial_glossary_block(
        self,
        kb: KnowledgeBase,
        domain: str,
        text: str,
        tags: List[str],
        expanded_tags: Set[str],
        budget: Optional[KnowledgeBudget] = None,
    ) -> str:
        parts: List[str] = []

        if kb.rhetoric_frameworks:
            rhet_enabled = budget is None or budget.rhetoric.enabled
            if rhet_enabled:
                rhet_limit = budget.rhetoric.entry_limit if budget else self._limits.rhetoric
                rhet_char = budget.rhetoric.char_budget if budget else None

                normalized_text = _normalize_text_for_match(text)
                rhetoric_sample = _select_ranked_entries(
                    kb.rhetoric_frameworks,
                    normalized_text,
                    tags + ["rhetoric"],
                    rhet_limit,
                    require_text_match=False,
                    scorer=_score_structural_entry,
                    expanded_tags=expanded_tags if expanded_tags else None,
                    candidate_limit=self._limits.rhetoric_candidates,
                    debug_context="rhetoric",
                    min_score=1,
                    char_budget=rhet_char,
                )

                rhetoric_lines: List[str] = []
                for framework in rhetoric_sample:
                    name = framework.get("name", "")
                    steps = framework.get("steps", [])
                    step_names = [
                        step.get("name", "")
                        for step in steps
                        if isinstance(step, dict) and step.get("name")
                    ]
                    if name and step_names:
                        rhetoric_lines.append(f" • {name}: " + " → ".join(step_names))

                if rhetoric_lines:
                    parts.append(
                        "Риторические топосы и приёмы аргументации:\n" + "\n".join(rhetoric_lines)
                    )

        if kb.editorial_techniques:
            ed_enabled = budget is None or budget.editorial.enabled
            if ed_enabled:
                ed_limit = budget.editorial.entry_limit if budget else self._limits.editorial
                ed_char = budget.editorial.char_budget if budget else None

                editorial_sample = _select_by_tags_or_all(
                    kb.editorial_techniques,
                    tags=tags + ["editing"],
                    limit=ed_limit,
                    expanded_tags=expanded_tags,
                    min_score=1,
                    char_budget=ed_char,
                )

                editorial_lines: List[str] = []
                for tech in editorial_sample:
                    name = tech.get("name", "")
                    category = tech.get("category", "")
                    description = tech.get("description", "")
                    wrong = tech.get("example_wrong", "")
                    correct = tech.get("example_correct", "")
                    explanation = tech.get("example_explanation", "")

                    line = f" • {name}"
                    if category:
                        line += f" ({category})"
                    if description:
                        line += f": {description.strip()}"
                    if wrong or correct:
                        pair = f"Пример: {wrong} → {correct}"
                        if explanation:
                            pair += f" ({explanation.strip()})"
                        line += f". {pair}"
                    editorial_lines.append(line)

                if editorial_lines:
                    parts.append(
                        "Редакторские приёмы (по Норе Галь и другим редакторам):\n"
                        + "\n".join(editorial_lines)
                    )

        if kb.domain_glossary:
            gl_enabled = budget is None or budget.glossary.enabled
            if gl_enabled:
                gl_limit = budget.glossary.entry_limit if budget else self._limits.glossary

                relevant_terms: Dict[str, str] = {}
                wanted_tags_set = {normalize_tag(tag) for tag in tags if isinstance(tag, str)}
                normalized_text = _normalize_text_for_match(text)

                for _, dom_terms in kb.domain_glossary.items():
                    if isinstance(dom_terms, dict):
                        for term, definition in dom_terms.items():
                            if _contains_pattern(normalized_text, term):
                                relevant_terms[term] = definition
                            if len(relevant_terms) >= gl_limit:
                                break
                    if len(relevant_terms) >= gl_limit:
                        break

                if len(relevant_terms) < gl_limit:
                    domains_to_check = [domain] + [
                        item for item in kb.domain_glossary.keys() if item != domain
                    ]
                    for dom in domains_to_check:
                        if dom in kb.domain_glossary:
                            dom_terms = kb.domain_glossary[dom]
                            if isinstance(dom_terms, dict):
                                if dom == domain or any(
                                    tag in wanted_tags_set for tag in [dom.lower()]
                                ):
                                    for term, definition in dom_terms.items():
                                        if term not in relevant_terms:
                                            relevant_terms[term] = definition
                                        if len(relevant_terms) >= gl_limit:
                                            break

                if relevant_terms:
                    sample_items = list(relevant_terms.items())[:gl_limit]
                    term_lines = [f" • {key}: {value}" for key, value in sample_items]
                    parts.append("Глоссарий (релевантные термины):\n" + "\n".join(term_lines))

        if not parts:
            return ""

        return "\n\n" + "\n\n".join(parts)

    def _build_stop_words_block(self, kb: KnowledgeBase, primary_tags: List[str]) -> str:
        stop_words_dict = kb.stop_words
        if not isinstance(stop_words_dict, dict) or not stop_words_dict:
            return "Стоп-слова и нежелательные конструкции: (нет данных)"

        tag_set = {normalize_tag(tag) for tag in primary_tags if isinstance(tag, str)}

        priority_categories: List[Tuple[str, List[str]]] = []
        other_categories: List[Tuple[str, List[str]]] = []

        for category, words in stop_words_dict.items():
            if not isinstance(category, str) or not isinstance(words, (list, tuple)):
                continue
            category_norm = normalize_tag(category)
            if category_norm in tag_set:
                priority_categories.append((category, list(words)))
            else:
                other_categories.append((category, list(words)))

        ordered_categories = priority_categories + sorted(
            other_categories,
            key=lambda item: item[0].lower(),
        )

        lines: List[str] = []
        for category, words in ordered_categories[: self._limits.stop_words_category]:
            clean_words: List[str] = []
            seen: Set[str] = set()
            for word in words:
                if not isinstance(word, str):
                    continue
                stripped = word.strip()
                if stripped and stripped not in seen:
                    seen.add(stripped)
                    clean_words.append(stripped)

            if not clean_words:
                continue

            limited_words = clean_words[: self._limits.stop_words_items]
            quoted_words = [f'"{word}"' for word in limited_words]
            if len(clean_words) > self._limits.stop_words_items:
                quoted_words.append("…")
            lines.append(f" • {category}: {', '.join(quoted_words)}")

        if not lines:
            return "Стоп-слова и нежелательные конструкции: (нет данных)"

        header = "Стоп-слова и нежелательные конструкции (удаляй или переписывай):"
        return header + "\n" + "\n".join(lines)

    def _build_knowledge_block(
        self,
        text: str,
        domain: str,
        intent: Optional[str],
        overlays: Sequence[str],
    ) -> str:
        kb = self._get_knowledge_base()
        domain_cfg = self._get_domain_config(domain)
        features = self._resolve_prompt_features(domain_cfg, domain, intent, overlays)

        tags = features["tags"]
        expanded_tags = set(features["expanded_tags"])
        storytelling_enabled = features["storytelling_enabled"]
        marketing_enabled = features["marketing_enabled"]

        if self.enable_selection_diagnostics:
            logger.debug("Resolved tags: %s", tags)
            logger.debug("Expanded tags: %s", expanded_tags)
            logger.debug(
                "Storytelling enabled: %s, Marketing enabled: %s",
                storytelling_enabled,
                marketing_enabled,
            )

        active_blocks: Set[str] = {"grammar", "style", "logic", "stop_words"}
        if storytelling_enabled:
            active_blocks.add("storytelling")
        if marketing_enabled:
            active_blocks.add("marketing")
        if kb.composition_principles:
            active_blocks.add("composition")
        if kb.local_cohesion:
            active_blocks.add("cohesion")
        if kb.composition_errors:
            active_blocks.add("composition_errors")
        if kb.rhetoric_frameworks:
            active_blocks.add("rhetoric")
        if kb.editorial_techniques:
            active_blocks.add("editorial")
        if kb.nkrj_structure_patterns:
            active_blocks.add("nkrj")
        if kb.domain_glossary:
            active_blocks.add("glossary")

        budget = self._budget_manager.allocate(self._limits, active_blocks=active_blocks)

        if self.enable_selection_diagnostics:
            logger.debug(
                "KnowledgeBudget: %s",
                {
                    b: (budget.get(b).char_budget, budget.get(b).entry_limit, budget.get(b).enabled)
                    for b in active_blocks
                },
            )

        def _safe_optional(block_name: str, builder_fn: Any, *args: Any, **kwargs: Any) -> str:
            """
            Вызывает builder_fn для optional-блока.
            При любой ошибке: WARNING в лог, возврат пустой строки.
            Mandatory-блоки вызываются напрямую без обёртки — fail fast.
            """
            try:
                return builder_fn(*args, **kwargs)
            except Exception as exc:
                logger.warning(
                    "Optional KB block '%s' failed and will be skipped: %s: %s",
                    block_name,
                    type(exc).__name__,
                    exc,
                )
                return ""

        # --- Mandatory блоки: падение = исключение наверх ---
        stop_words_block = self._build_stop_words_block(kb, tags)
        grammar_style_logic = self._build_grammar_style_logic_block(
            kb, text, tags, expanded_tags, budget=budget
        )

        # --- Optional блоки: падение = warning + пустая строка ---
        composition_cohesion = _safe_optional(
            "composition/cohesion",
            self._build_composition_cohesion_errors_block,
            kb, tags, expanded_tags, budget=budget,
        )
        nkrj_block = _safe_optional(
            "nkrj",
            self._build_nkrj_block,
            kb, budget=budget,
        )
        storytelling_block = _safe_optional(
            "storytelling",
            self._build_storytelling_block,
            kb, text, tags, expanded_tags, storytelling_enabled, budget=budget,
        )
        marketing_block = _safe_optional(
            "marketing",
            self._build_marketing_block,
            kb, text, tags, expanded_tags, marketing_enabled, budget=budget,
        )
        rhetoric_editorial_glossary = _safe_optional(
            "rhetoric/editorial/glossary",
            self._build_rhetoric_editorial_glossary_block,
            kb, domain, text, tags, expanded_tags, budget=budget,
        )

        return (
            "База знаний:\n\n"
            f"{stop_words_block}\n\n"
            f"{grammar_style_logic}\n\n"
            f"{composition_cohesion}"
            f"{nkrj_block}"
            f"{storytelling_block}"
            f"{marketing_block}"
            f"{rhetoric_editorial_glossary}"
        )

    def _build_output_format_block(self, mode: str) -> str:
        format_text = self._get_output_format(mode)
        return f"Формат ответа:\n{format_text}"

    def _build_text_block(self, text: str) -> str:
        return f'Текст для обработки:\n"""\n{text}\n"""'


_DEFAULT_BUILDER: Optional[PromptBuilder] = None


def _get_default_builder() -> PromptBuilder:
    global _DEFAULT_BUILDER
    if _DEFAULT_BUILDER is None:
        _DEFAULT_BUILDER = PromptBuilder()
    return _DEFAULT_BUILDER


def build_prompt(
    text: str,
    domain: str = "marketing",
    intent: Optional[str] = None,
    audience_type: str = "b2b",
    overlays: Sequence[str] = (),
    output_mode: str = "text_only",
) -> str:
    """Legacy wrapper для внешнего кода."""
    audience_map = {
        "b2b": AudienceProfile(kind="b2b", expertise="pro", formality="neutral"),
        "b2c": AudienceProfile(kind="b2c", expertise="novice", formality="casual"),
        "mixed": AudienceProfile(kind="mixed", expertise="pro", formality="neutral"),
    }

    if audience_type not in audience_map:
        logger.warning("Unknown audience_type '%s', falling back to 'b2b'", audience_type)

    audience = audience_map.get(audience_type, audience_map["b2b"])
    builder = _get_default_builder()
    return builder.build(
        text=text,
        domain=domain,
        intent=intent,
        audience=audience,
        overlays=overlays,
        output_mode=output_mode,
    )


def validate_entry_tags(
    entry: Dict[str, Any],
    entry_name: str,
    index: int,
    strict: bool = True,
) -> None:
    """Проверяет поле tags у KB-записи."""
    if "tags" not in entry or entry["tags"] is None:
        return

    tags = entry["tags"]
    if not isinstance(tags, list):
        raise ValueError(f"{entry_name}[{index}].tags must be a list, got {type(tags)}")

    unknown: List[str] = []
    for j, tag in enumerate(tags):
        if not isinstance(tag, str):
            raise ValueError(f"{entry_name}[{index}].tags[{j}] must be str, got {type(tag)}")
        norm = normalize_tag(tag)
        if norm not in KNOWN_TAGS:
            unknown.append(norm)

    if unknown:
        message = f"{entry_name}[{index}] contains unknown tags: {sorted(set(unknown))}"
        if strict:
            raise ValueError(message)
        logger.warning(message)


def _validate_stop_words_structure(stop_words: Any) -> None:
    """Проверяет stop_words."""
    if not isinstance(stop_words, dict):
        raise ValueError("stop_words must be a dict")
    for category, words in stop_words.items():
        if not isinstance(category, str):
            raise ValueError(f"stop_words category key must be str, got {type(category)}")
        if not isinstance(words, (list, tuple)):
            raise ValueError(
                f"stop_words['{category}'] must be a list or tuple, got {type(words)}"
            )
        for i, word in enumerate(words):
            if not isinstance(word, str):
                raise ValueError(
                    f"stop_words['{category}'][{i}] must be str, got {type(word)}"
                )


def _validate_rule_entries(entries: List[Dict[str, Any]], name: str, sample_size: int = 5) -> None:
    """Проверяет набор rule entries."""
    if not isinstance(entries, list):
        raise ValueError(f"{name} must be a list")
    for i, entry in enumerate(entries[:sample_size]):
        if not isinstance(entry, dict):
            raise ValueError(f"{name}[{i}] must be a dict, got {type(entry)}")
        for str_key in ("wrong", "correct", "rule", "description", "name", "category"):
            if str_key in entry:
                val = entry[str_key]
                if not isinstance(val, str):
                    raise ValueError(f"{name}[{i}].{str_key} must be str, got {type(val)}")
        validate_entry_tags(entry, name, i, strict=KB_TAGS_STRICT_VALIDATION)

        has_info = False
        for info_key in ("wrong", "rule", "description", "name"):
            val = entry.get(info_key)
            if isinstance(val, str) and val.strip():
                has_info = True
                break
        if not has_info:
            raise ValueError(
                f"{name}[{i}] must contain non-empty 'wrong', 'rule', "
                "'description', or 'name'"
            )


def _validate_named_entries(entries: List[Dict[str, Any]], name: str, sample_size: int = 5) -> None:
    """Проверяет named entries."""
    if not isinstance(entries, list):
        raise ValueError(f"{name} must be a list")
    for i, entry in enumerate(entries[:sample_size]):
        if not isinstance(entry, dict):
            raise ValueError(f"{name}[{i}] must be a dict, got {type(entry)}")
        for str_key in ("name", "rule", "description", "category"):
            if str_key in entry:
                val = entry[str_key]
                if not isinstance(val, str):
                    raise ValueError(f"{name}[{i}].{str_key} must be str, got {type(val)}")
        validate_entry_tags(entry, name, i, strict=KB_TAGS_STRICT_VALIDATION)

        has_info = False
        for info_key in ("name", "rule", "description"):
            val = entry.get(info_key)
            if isinstance(val, str) and val.strip():
                has_info = True
                break
        if not has_info:
            raise ValueError(
                f"{name}[{i}] must contain non-empty 'name', 'rule', or 'description'"
            )


def _validate_structural_entries(
    entries: List[Dict[str, Any]],
    name: str,
    sample_size: int = 5,
) -> None:
    """Проверяет structural entries."""
    if not isinstance(entries, list):
        raise ValueError(f"{name} must be a list")
    for i, entry in enumerate(entries[:sample_size]):
        if not isinstance(entry, dict):
            raise ValueError(f"{name}[{i}] must be a dict, got {type(entry)}")
        if "name" not in entry or not isinstance(entry["name"], str) or not entry["name"].strip():
            raise ValueError(f"{name}[{i}] must have a non-empty 'name'")
        for container_key in ("steps", "sections"):
            container = entry.get(container_key)
            if container is not None and not isinstance(container, list):
                raise ValueError(f"{name}[{i}].{container_key} must be a list if present")
        validate_entry_tags(entry, name, i, strict=KB_TAGS_STRICT_VALIDATION)


def _validate_logic_entries(entries: List[Dict[str, Any]], name: str, sample_size: int = 5) -> None:
    """Проверяет logic entries."""
    if not isinstance(entries, list):
        raise ValueError(f"{name} must be a list")
    for i, entry in enumerate(entries[:sample_size]):
        if not isinstance(entry, dict):
            raise ValueError(f"{name}[{i}] must be a dict, got {type(entry)}")
        for str_key in ("name", "wrong", "rule", "description", "category"):
            if str_key in entry:
                val = entry[str_key]
                if not isinstance(val, str):
                    raise ValueError(f"{name}[{i}].{str_key} must be str, got {type(val)}")
        validate_entry_tags(entry, name, i, strict=KB_TAGS_STRICT_VALIDATION)

        has_info = False
        for info_key in ("name", "wrong", "rule", "description"):
            val = entry.get(info_key)
            if isinstance(val, str) and val.strip():
                has_info = True
                break
        if not has_info:
            raise ValueError(
                f"{name}[{i}] must contain non-empty 'name', 'wrong', "
                "'rule', or 'description'"
            )


def _validate_list_of_dicts(entries: List[Any], name: str) -> None:
    """Проверяет, что это список словарей."""
    if not isinstance(entries, list):
        raise ValueError(f"{name} must be a list")
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{name}[{i}] must be a dict, got {type(entry)}")


def validate_configs_and_kb(
    config_path: Path = Path("config"),
    kb_path: Path = Path("knowledge_base"),
    mode: str = "strict",
) -> None:
    """
    Проверяет загрузку конфигов и базы знаний.

    Args:
        config_path: Путь к директории конфигов.
        kb_path:     Путь к директории базы знаний.
        mode:        Режим валидации:
                     - "strict"  (default) — любая ошибка бросает RuntimeError.
                       Используется при старте сервиса (lifespan).
                     - "soft"    — ошибки в optional-блоках логируются как WARNING,
                       mandatory-блоки по-прежнему бросают RuntimeError.
                       Удобно для миграции и hot-reload.
    """
    if mode not in ("strict", "soft"):
        raise ValueError(f"mode must be 'strict' or 'soft', got {mode!r}")

    def _handle_optional_error(block: str, error: Exception) -> None:
        msg = f"Optional KB block '{block}' validation failed: {error}"
        if mode == "strict":
            raise RuntimeError(msg) from error
        logger.warning("%s — skipping (soft mode)", msg)
    try:
        core = load_core_config(config_path)
        if not core.role:
            raise ValueError("Core config missing role")
    except Exception as error:
        raise RuntimeError(f"Core config validation failed: {error}") from error

    domains_dir = config_path / "domains"
    try:
        domain_files = sorted(domains_dir.glob("*.json")) if domains_dir.exists() else []
        if domain_files:
            first_domain = domain_files[0].stem
            domain_cfg = load_domain_config(first_domain, config_path)
            if not domain_cfg.system_rules:
                raise ValueError("Domain config missing system_rules")
    except Exception as error:
        raise RuntimeError(f"Domain config validation failed: {error}") from error

    intents_dir = config_path / "intents"
    try:
        intent_files = sorted(intents_dir.glob("*.json")) if intents_dir.exists() else []
        if intent_files:
            first_intent = intent_files[0].stem
            intent_cfg = load_intent_config(first_intent, config_path)
            if intent_cfg is not None and not intent_cfg.instructions:
                raise ValueError("Intent config missing instructions")
    except Exception as error:
        raise RuntimeError(f"Intent config validation failed: {error}") from error

    overlays_dir = config_path / "overlays"
    try:
        overlay_files = sorted(overlays_dir.glob("*.json")) if overlays_dir.exists() else []
        if overlay_files:
            first_overlay = overlay_files[0].stem
            overlay_cfg = load_overlay_config(first_overlay, config_path)
            if not overlay_cfg.instructions:
                raise ValueError("Overlay config missing instructions")
    except Exception as error:
        raise RuntimeError(f"Overlay config validation failed: {error}") from error

    try:
        fmt = load_output_format("text_only", config_path)
        if not isinstance(fmt, str):
            raise ValueError("Output format not a string")
    except Exception as error:
        raise RuntimeError(f"Output format validation failed: {error}") from error

    try:
        kb = load_knowledge_base(kb_path)
        _validate_stop_words_structure(kb.stop_words)
        _validate_rule_entries(kb.grammar_errors, "grammar_errors")
        _validate_rule_entries(kb.stylistic_issues, "stylistic_issues")
        if kb.logic_issues:
            _validate_logic_entries(kb.logic_issues, "logic_issues")
    except Exception as error:
        raise RuntimeError(f"Knowledge base validation failed (mandatory): {error}") from error

    # --- Optional KB blocks: ошибки обрабатываются по mode ---
    _optional_kb_checks = [
        ("storytelling_frameworks", kb.storytelling_frameworks, _validate_structural_entries),
        ("marketing_templates",     kb.marketing_templates,     _validate_structural_entries),
        ("rhetoric_frameworks",     kb.rhetoric_frameworks,     _validate_structural_entries),
        ("composition_principles",  kb.composition_principles,  _validate_structural_entries),
        ("local_cohesion",          kb.local_cohesion,          _validate_structural_entries),
        ("composition_errors",      kb.composition_errors,      _validate_structural_entries),
        ("editorial_techniques",    kb.editorial_techniques,    _validate_structural_entries),
    ]
    for block_name, entries, validator in _optional_kb_checks:
        if entries:
            try:
                validator(entries, block_name)
            except Exception as error:
                _handle_optional_error(block_name, error)

    dummy_text = "тестовый текст"
    dummy_tags = ["marketing", "test"]

    try:
        _ = select_grammar_rules(kb, dummy_text, dummy_tags, limit=1)
        _ = select_style_issues(kb, dummy_text, dummy_tags, limit=1)
        _ = select_logic_issues(kb, dummy_text, dummy_tags, limit=1)
        _ = _select_by_tags_or_all(kb.composition_principles, dummy_tags, limit=1)
    except Exception as error:
        raise RuntimeError(f"Knowledge selectors smoke test failed: {error}") from error

    try:
        _ = build_nkrj_norms_lines(kb)
    except Exception as error:
        raise RuntimeError(f"NKRJ validation failed: {error}") from error

    logger.info("Config and knowledge base validation passed successfully.")


def _self_test_tag_normalization() -> None:
    """Self-test нормализации тегов."""
    assert normalize_tag("anti_ai") == "antiai"
    assert normalize_tag("anti-ai") == "antiai"
    assert normalize_tag("de-ai") == "deai"
    assert normalize_tag("nora_gal") == "noragal"
    assert normalize_tag("marketing_push") == "marketingpush"
    assert normalize_tag("non_marketing") == "nonmarketing"
    assert normalize_tag(" ANTIAI ") == "antiai"
    assert normalize_tag("info-style") == "infostyle"

    result = normalize_tags(["anti_ai", "humanize", "anti-ai", "HUMANIZE", " antiai "])
    assert result == ["antiai", "humanize"]

    entry: Dict[str, Any] = {"tags": ["anti_ai", "humanize", "anti-ai"]}
    normalize_entry_tags_inplace(entry)
    assert entry["tags"] == ["antiai", "humanize"]

    primary = _get_primary_tags_for_category("domains", "deai")
    assert "deai" in primary

    expanded = _get_expanded_tags_for_category("domains", "deai")
    assert "antiai" in expanded

    primary_noragal = _get_primary_tags_for_category("intents", "noragal")
    assert "noragal" in primary_noragal

    assert "antiai" in KNOWN_TAGS
    assert "noragal" in KNOWN_TAGS
    assert "marketingpush" in KNOWN_TAGS
    assert "factcheck" in KNOWN_TAGS
    assert "infostyle" in KNOWN_TAGS

    validate_entry_tags(
        {"tags": ["antiai", "humanize"]},
        entry_name="test",
        index=0,
        strict=True,
    )

    print("✓ ТП-4 self-test passed")


if __name__ == "__main__":
    import sys

    if "--self-test" in sys.argv:
        _self_test_tag_normalization()
