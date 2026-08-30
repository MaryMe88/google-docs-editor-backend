# src/prompt_builder/kb_rendering.py
"""
Рендеринг блоков базы знаний в промпт.
"""

from __future__ import annotations

import hashlib
import logging
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

from src.config_types import (
    BlockBudget,
    KnowledgeBase,
    KnowledgeBudget,
    LimitsConfig,
    get_canonical_tags_for_category,
    get_primary_tags_for_category,
)
from src.knowledge_retrieval import (
    FallbackStage,
    select_grammar_rules,
    select_logic_issues,
    select_style_issues,
    select_structural_by_tags_or_all,
)
from src.reason_codes import ReasonCode

logger = logging.getLogger(__name__)


def _unpack_retrieval_result(
    result: Any,
) -> Tuple[List[Dict[str, Any]], "FallbackStage", int]:
    if isinstance(result, tuple) and len(result) == 3:
        return result
    if isinstance(result, tuple) and len(result) == 2:
        return result[0], result[1], 0
    return result, FallbackStage.STRONG, 0


def _get_confidence_note(stage: FallbackStage) -> str:
    if stage == FallbackStage.STRONG:
        return ""
    if stage == FallbackStage.TEXT_ONLY:
        return (
            "⚠ Правила подобраны по смысловому совпадению с текстом, "
            "не по точному образцу. Применяй только если явно уместно."
        )
    if stage in (FallbackStage.TAG_ONLY, FallbackStage.NEUTRAL):
        return (
            "⚠ Правила подобраны по теме раздела, конкретных совпадений "
            "с текстом не найдено. Применяй с осторожностью — "
            "только если ошибка очевидна."
        )
    if stage == FallbackStage.EMPTY:
        return ""
    return ""


def _collect_retrieval_tags(
    domain: str,
    intent: Optional[str],
    overlays: Sequence[str],
) -> Dict[str, Set[str]]:
    primary: Set[str] = set()
    expanded: Set[str] = set()
    domain_primary = get_primary_tags_for_category("domains", domain)
    if not domain_primary:
        logger.warning("_collect_retrieval_tags: no tags found for domain=%r", domain)
    primary.update(domain_primary)
    expanded.update(get_canonical_tags_for_category("domains", domain))
    if intent and intent != "neutral":
        primary.update(get_primary_tags_for_category("intents", intent))
        expanded.update(get_canonical_tags_for_category("intents", intent))
    for overlay in overlays:
        primary.update(get_primary_tags_for_category("overlays", overlay))
        expanded.update(get_canonical_tags_for_category("overlays", overlay))
    primary.update({"grammar", "style", "editing", "clarity"})
    return {"primary": primary, "expanded": expanded - primary}


def _append_rule_entries(lines: List[str], title: str, entries: List[Dict[str, Any]]) -> None:
    if not entries:
        return
    lines.append(title)
    for entry in entries:
        wrong = entry.get("wrong")
        correct = entry.get("correct")
        rule = entry.get("rule") or entry.get("description") or entry.get("name")
        fragments = []
        if wrong:
            fragments.append(f"плохо: {wrong}")
        if correct:
            fragments.append(f"лучше: {correct}")
        if rule:
            fragments.append(f"пояснение: {rule}")
        if fragments:
            lines.append("- " + "; ".join(fragments))


def _append_structural_entries(lines: List[str], title: str, entries: List[Dict[str, Any]]) -> None:
    if not entries:
        return
    lines.append(title)
    for entry in entries:
        name = entry.get("name", "")
        description = entry.get("description", "")
        when_to_use = entry.get("when_to_use", "")
        if isinstance(when_to_use, list):
            when_to_use = "; ".join(str(item) for item in when_to_use[:3])
        fragments = []
        if name:
            fragments.append(str(name))
        if description:
            fragments.append(str(description))
        if when_to_use:
            fragments.append(f"когда применять: {when_to_use}")
        if fragments:
            lines.append("- " + " | ".join(fragments))


def _append_editorial_entries(lines: List[str], title: str, entries: List[Dict[str, Any]]) -> None:
    if not entries:
        return
    lines.append(title)
    for entry in entries:
        name = entry.get("name", "")
        description = entry.get("description", "")
        how_to_apply = entry.get("how_to_apply", [])
        if isinstance(how_to_apply, list):
            how_to_apply_str = "; ".join(str(item) for item in how_to_apply[:3])
        else:
            how_to_apply_str = ""
        fragments = []
        if name:
            fragments.append(str(name))
        if description:
            fragments.append(str(description))
        if how_to_apply_str:
            fragments.append(f"как применять: {how_to_apply_str}")
        if fragments:
            lines.append("- " + " | ".join(fragments))


def _append_case_study_entries(lines: List[str], title: str, entries: List[Dict[str, Any]]) -> None:
    if not entries:
        return
    lines.append(title)
    for entry in entries:
        name = str(entry.get("name", "")).strip()
        description = str(entry.get("description", "")).strip()
        header = f"- {name}" if name else "-"
        if description:
            header = f"{header}: {description}" if name else f"- {description}"
        lines.append(header)

        constraints = entry.get("constraints", [])
        if isinstance(constraints, list):
            for constraint in constraints[:4]:
                text = str(constraint).strip()
                if text:
                    lines.append(f"  Ограничение: {text}")

        sections = entry.get("sections", [])
        if isinstance(sections, list):
            for index, section in enumerate(sections[:8], start=1):
                if not isinstance(section, dict):
                    continue
                section_name = str(section.get("name", "")).strip()
                goal = str(section.get("goal", "")).strip()
                hint = str(section.get("hint", "")).strip()
                if not (section_name or goal):
                    continue
                parts = [p for p in (section_name, goal) if p]
                lines.append(f"  {index}. " + " — ".join(parts))
                if hint:
                    lines.append(f"     Ориентир: {hint}")


def _append_evaluation_techniques(lines: List[str], title: str, data: Dict[str, Any]) -> None:
    if not data:
        return
    lines.append(title)

    if "category" in data:
        lines.append(f"Категория: {data['category']}")
    if "description" in data:
        lines.append(f"Описание: {data['description']}")

    editor_rules = data.get("editor_rules", [])
    if isinstance(editor_rules, list) and editor_rules:
        lines.append("Правила редактора:")
        for rule in editor_rules[:5]:
            if isinstance(rule, str) and rule.strip():
                lines.append(f"- {rule}")

    strategies = data.get("replacement_strategies", [])
    if isinstance(strategies, list) and strategies:
        lines.append("Стратегии замены оценок:")
        for strategy in strategies[:3]:
            if not isinstance(strategy, dict):
                continue
            name = strategy.get("name", "")
            desc = strategy.get("description", "")
            if name or desc:
                lines.append(f"- {name}: {desc}" if name and desc else f"- {name or desc}")

    diagnostics = data.get("diagnostics", [])
    if isinstance(diagnostics, list) and diagnostics:
        lines.append("Чек-листы для обнаружения оценок:")
        for diag in diagnostics[:3]:
            if isinstance(diag, dict):
                dname = diag.get("name", "")
                if dname:
                    lines.append(f"- {dname}")

    tests = data.get("tests", [])
    if isinstance(tests, list) and tests:
        lines.append("Тесты на замену оценок:")
        for test in tests[:3]:
            if isinstance(test, dict):
                tname = test.get("name", "")
                if tname:
                    lines.append(f"- {tname}")

    source = data.get("source")
    if isinstance(source, dict):
        title = source.get("title", "")
        if title:
            lines.append(f"Источник: {title}")


def _append_glossary(lines: List[str], glossary: Dict[str, Any], limit: int) -> None:
    if not glossary:
        return
    lines.append("Глоссарий домена:")
    count = 0
    for term, value in glossary.items():
        if count >= limit:
            break
        if isinstance(value, str) and value.strip():
            lines.append(f"- {term}: {value.strip()}")
            count += 1
        elif isinstance(value, dict):
            description = value.get("description") or value.get("meaning") or ""
            if isinstance(description, str) and description.strip():
                lines.append(f"- {term}: {description.strip()}")
                count += 1


def _append_nkrj(lines: List[str], nkrj: Dict[str, Any]) -> None:
    if not nkrj:
        return
    lines.append("Структурные паттерны НКРЯ:")
    for key, value in list(nkrj.items())[:5]:
        if isinstance(value, str) and value.strip():
            lines.append(f"- {key}: {value.strip()}")
        elif isinstance(value, dict):
            description = value.get("description", "")
            if isinstance(description, str) and description.strip():
                lines.append(f"- {key}: {description.strip()}")


def _warn_if_empty_retrieval(
    block: str,
    stage: "FallbackStage",
    domain: str,
    intent: Optional[str],
    overlays: List[str],
    text_len: int,
    primary_tags: Set[str],
) -> None:
    if intent not in ("analytical", "storytelling", "engagement", "marketingpush"):
        return
    logger.warning(
        "KB retrieval empty: block=%s, stage=%s, domain=%s, intent=%s, overlays=%s, text_length=%d, primary_tags=%s",
        block, stage.value, domain, intent, overlays, text_len, sorted(primary_tags),
    )


@dataclass(frozen=True)
class KBBlockConfig:
    name: str
    budget_key: str
    retrieval_fn: Callable
    append_fn: Callable
    title: str
    kb_attr: Optional[str] = None
    uses_structural_call: bool = False
    candidate_attr: Optional[str] = None


KB_BLOCK_REGISTRY: List[KBBlockConfig] = [
    KBBlockConfig(name="grammar", budget_key="grammar", retrieval_fn=select_grammar_rules,
                  append_fn=_append_rule_entries, title="Грамматические ориентиры:",
                  kb_attr=None, uses_structural_call=False, candidate_attr="grammar_candidates"),
    KBBlockConfig(name="style", budget_key="style", retrieval_fn=select_style_issues,
                  append_fn=_append_rule_entries, title="Стилистические ориентиры:",
                  kb_attr=None, uses_structural_call=False, candidate_attr="style_candidates"),
    KBBlockConfig(name="logic", budget_key="logic", retrieval_fn=select_logic_issues,
                  append_fn=_append_rule_entries, title="Логические ориентиры:",
                  kb_attr=None, uses_structural_call=False, candidate_attr="logic_candidates"),
    KBBlockConfig(name="composition", budget_key="composition", retrieval_fn=select_structural_by_tags_or_all,
                  append_fn=_append_structural_entries, title="Принципы композиции:",
                  kb_attr="composition_principles", uses_structural_call=True, candidate_attr=None),
    KBBlockConfig(name="composition_errors", budget_key="composition_errors", retrieval_fn=select_structural_by_tags_or_all,
                  append_fn=_append_structural_entries, title="Ошибки композиции:",
                  kb_attr="composition_errors", uses_structural_call=True, candidate_attr=None),
    KBBlockConfig(name="cohesion", budget_key="cohesion", retrieval_fn=select_structural_by_tags_or_all,
                  append_fn=_append_structural_entries, title="Локальная связность:",
                  kb_attr="local_cohesion", uses_structural_call=True, candidate_attr=None),
    KBBlockConfig(name="storytelling", budget_key="storytelling", retrieval_fn=select_structural_by_tags_or_all,
                  append_fn=_append_structural_entries, title="Сторителлинг-фреймворки:",
                  kb_attr="storytelling_frameworks", uses_structural_call=True, candidate_attr=None),
    KBBlockConfig(name="marketing", budget_key="marketing", retrieval_fn=select_structural_by_tags_or_all,
                  append_fn=_append_structural_entries, title="Маркетинговые шаблоны:",
                  kb_attr="marketing_templates", uses_structural_call=True, candidate_attr=None),
    KBBlockConfig(name="casestudy", budget_key="casestudy", retrieval_fn=select_structural_by_tags_or_all,
                  append_fn=_append_case_study_entries, title="Жанровые ориентиры бизнес-кейса:",
                  kb_attr="case_study_templates", uses_structural_call=True, candidate_attr=None),
    KBBlockConfig(name="rhetoric", budget_key="rhetoric", retrieval_fn=select_structural_by_tags_or_all,
                  append_fn=_append_structural_entries, title="Риторические приёмы:",
                  kb_attr="rhetoric_frameworks", uses_structural_call=True, candidate_attr=None),
    KBBlockConfig(name="editorial", budget_key="editorial", retrieval_fn=select_structural_by_tags_or_all,
                  append_fn=_append_editorial_entries, title="Редакторские приёмы:",
                  kb_attr="editorial_techniques", uses_structural_call=True, candidate_attr=None),
    KBBlockConfig(name="evaluation_techniques", budget_key="evaluation_techniques",
                  retrieval_fn=None, append_fn=None, title="",
                  kb_attr="evaluation_techniques", uses_structural_call=False, candidate_attr=None),
]


DEFAULT_CANDIDATE_LIMIT = 10


def _has_few_shot_pair(entry: Dict[str, Any]) -> bool:
    wrong = entry.get("wrong") or entry.get("example_wrong")
    correct = entry.get("correct") or entry.get("example_correct")
    return bool(wrong and correct)


def _format_few_shot_example(entry: Dict[str, Any]) -> str:
    wrong = entry.get("wrong") or entry.get("example_wrong")
    correct = entry.get("correct") or entry.get("example_correct")
    return f"Было: {wrong}\nСтало: {correct}"


def _select_few_shot_examples(
    entries_with_pairs: List[Dict[str, Any]],
    max_examples: int,
    pool_size: int = 10,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if not entries_with_pairs or max_examples <= 0:
        return []
    pool = entries_with_pairs[:pool_size]
    if len(pool) <= max_examples:
        return pool
    rng = random.Random(seed)
    return rng.sample(pool, max_examples)


def _derive_seed(text: str) -> int:
    # Используем sha256 для детерминированного seed (не зависит от PYTHONHASHSEED)
    digest = hashlib.sha256(text[:256].encode()).hexdigest()
    return int(digest[:8], 16)


def _process_kb_block(
    config: KBBlockConfig,
    lines: List[str],
    meta: Dict[str, Any],
    kb: KnowledgeBase,
    text: str,
    primary_tags: Set[str],
    expanded_tags: Set[str],
    budget: BlockBudget,
    domain: str,
    intent: Optional[str],
    overlays: List[str],
    include_few_shot: bool,
    total_few_shot_used: int,
    limits: LimitsConfig,
    few_shot_seed: Optional[int] = None,
) -> int:
    if not config.uses_structural_call:
        candidate_limit = getattr(limits, config.candidate_attr) if config.candidate_attr else None
        if candidate_limit is None:
            candidate_limit = DEFAULT_CANDIDATE_LIMIT
        result = config.retrieval_fn(
            kb=kb,
            text=text,
            tags=primary_tags,
            limit=budget.entry_limit,
            candidate_limit=candidate_limit,
            char_budget=getattr(budget, 'char_budget', None),
            return_meta=True,
        )
    else:
        if not config.kb_attr:
            return total_few_shot_used
        entries_source = kb.get(config.kb_attr)
        if not entries_source:
            return total_few_shot_used
        result = config.retrieval_fn(
            entries=entries_source,
            tags=primary_tags,
            limit=budget.entry_limit,
            expanded_tags=expanded_tags,
            char_budget=getattr(budget, 'char_budget', None),
            return_meta=True,
        )
    entries, stage, dropped = _unpack_retrieval_result(result)
    pair_entries = [e for e in entries if _has_few_shot_pair(e)]
    rule_entries = [e for e in entries if not _has_few_shot_pair(e)]
    few_shot_examples = []
    if include_few_shot:
        allowed = min(3, 5 - total_few_shot_used)
        if allowed > 0:
            few_shot_examples = _select_few_shot_examples(pair_entries, allowed, seed=few_shot_seed)
    confidence_note = _get_confidence_note(stage)
    if (rule_entries or few_shot_examples) and confidence_note:
        lines.append(confidence_note)
    if rule_entries:
        config.append_fn(lines, config.title, rule_entries)
    if few_shot_examples:
        lines.append("Примеры редактирования:")
        for ex in few_shot_examples:
            lines.append(_format_few_shot_example(ex))
        lines.append("")
    total_few_shot_used += len(few_shot_examples)
    meta[config.name] = {
        "stage": stage.value,
        "entries_count": len(entries),
        "rules_count": len(rule_entries),
        "few_shot_count": len(few_shot_examples),
        "few_shot_ids": [e.get("id") for e in few_shot_examples if e.get("id")],
        "entry_ids": [e.get("id") for e in entries[:5] if e.get("id")],
        "entry_names": [e.get("name") for e in entries[:5] if e.get("name")],
        "truncated_count": dropped,
    }
    if dropped > 0:
        logger.info("Char budget truncated %d records for block='%s'", dropped, config.name)
    if stage in (FallbackStage.EMPTY, FallbackStage.NEUTRAL):
        _warn_if_empty_retrieval(
            block=config.name, stage=stage, domain=domain, intent=intent,
            overlays=overlays, text_len=len(text), primary_tags=primary_tags,
        )
    return total_few_shot_used