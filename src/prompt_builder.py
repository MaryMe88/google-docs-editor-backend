"""
prompt_builder.py

Модуль для сборки финальных промптов из конфигов и базы знаний.

"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

from src.config_types import (
    AudienceProfile,
    BlockBudget,
    CoreConfig,
    DomainConfig,
    IntentConfig,
    KnowledgeBase,
    KnowledgeBudget,
    KnowledgeBudgetManager,
    KnowledgeLevel,
    LimitsConfig,
    OverlayConfig,
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
from src.shared_contracts import (
    ALLOWED_DOMAINS,
    ALLOWED_INTENTS,
    ALLOWED_OUTPUT_MODES,
    ALLOWED_OVERLAYS,
)
from src.tag_registry import normalize_tag, normalize_tags

logger = logging.getLogger(__name__)

# Интенты, для которых отсутствие знаний из KB — реальная проблема.
# Для intent=None или intent="neutral" WARNING не нужен.
_KNOWLEDGE_DEPENDENT_INTENTS: frozenset = frozenset({
    "analytical", "storytelling", "engagement", "marketingpush",
})

# ---------------------------------------------------------------------------
# Few-shot safety limits (PR‑2)
# ---------------------------------------------------------------------------
FEW_SHOT_MAX_EXAMPLES_PER_BLOCK = 3
FEW_SHOT_MAX_TOTAL_EXAMPLES = 5
FEW_SHOT_POOL_SIZE = 10
FEW_SHOT_RULES_FIRST = True


# ---------------------------------------------------------------------------
# Конфигурация блока базы знаний
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class KBBlockConfig:
    """Конфигурация одного блока базы знаний."""
    name: str
    budget_key: str
    retrieval_fn: Callable
    append_fn: Callable
    title: str
    kb_attr: Optional[str] = None
    uses_structural_call: bool = False
    candidate_attr: Optional[str] = None


# ---------------------------------------------------------------------------
# ТП-2: квалификатор уверенности для блоков KB
# ---------------------------------------------------------------------------
def _get_confidence_note(stage: "FallbackStage") -> str:
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


# ---------------------------------------------------------------------------
# Хелпер: распаковка результата retrieval
# ---------------------------------------------------------------------------
def _unpack_retrieval_result(
    result: Any,
) -> Tuple[List[Dict[str, Any]], "FallbackStage", int]:
    if isinstance(result, tuple) and len(result) == 3:
        return result
    if isinstance(result, tuple) and len(result) == 2:
        return result[0], result[1], 0
    return result, FallbackStage.STRONG, 0


# ---------------------------------------------------------------------------
# JSON-загрузчики
# ---------------------------------------------------------------------------
def load_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional_json(path: Path, default: Any) -> Any:
    if path.exists():
        return load_json_file(path)
    return default


# ---------------------------------------------------------------------------
# Few-shot helper functions
# ---------------------------------------------------------------------------
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
    pool_size: int = FEW_SHOT_POOL_SIZE,
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
    digest = hashlib.md5(text[:256].encode()).hexdigest()
    return int(digest[:8], 16)


# ---------------------------------------------------------------------------
# Загрузка конфигов
# ---------------------------------------------------------------------------
def load_core_config(base_path: Path = Path("config")) -> CoreConfig:
    data = load_json_file(base_path / "core.json")
    ip_ceiling_raw = data.get("ip_ceiling", {})
    ip_ceiling_value = (
        ip_ceiling_raw.get("value", 2.5)
        if isinstance(ip_ceiling_raw, dict)
        else float(ip_ceiling_raw) if ip_ceiling_raw is not None else 2.5
    )
    return CoreConfig(
        role=data.get("role", "You are a careful Russian editor."),
        priorities=data.get("priorities", "clarity, accuracy, readability"),
        basic_audit_instructions=tuple(data.get("basic_audit_instructions", [])),
        forbidden=tuple(data.get("forbidden", [])),
        ip_ceiling=ip_ceiling_value,
    )


def load_domain_config(
    domain: str,
    base_path: Path = Path("config"),
) -> DomainConfig:
    normalized_domain = domain.strip().lower()
    data = load_json_file(base_path / "domains" / f"{normalized_domain}.json")
    raw_tasks = data.get("tasks", [])
    raw_constraints = data.get("constraints", [])
    raw_ip = data.get("ip_ceiling")
    domain_ip_ceiling: Optional[float] = None
    if isinstance(raw_ip, (int, float)):
        domain_ip_ceiling = float(raw_ip)
    elif isinstance(raw_ip, dict):
        domain_ip_ceiling = float(raw_ip.get("value", 2.5))
    return DomainConfig(
        name=data.get("name", normalized_domain),
        system_rules=data.get("system_rules", ""),
        tone=data.get("tone", "neutral"),
        allow_storytelling=data.get("allow_storytelling", False),
        allow_marketing=data.get("allow_marketing", False),
        tasks=tuple(t for t in raw_tasks if isinstance(t, str)),
        constraints=tuple(c for c in raw_constraints if isinstance(c, str)),
        ip_ceiling=domain_ip_ceiling,
    )


def load_intent_config(
    intent: Optional[str],
    base_path: Path = Path("config"),
) -> Optional[IntentConfig]:
    if intent is None or intent == "neutral":
        return None
    normalized_intent = normalize_tag(intent)
    data = load_json_file(base_path / "intents" / f"{normalized_intent}.json")
    return IntentConfig(
        name=data.get("name", normalized_intent),
        instructions=data.get("instructions", []),
    )


# ---------------------------------------------------------------------------
# ИСПРАВЛЕННАЯ ФУНКЦИЯ ДЛЯ ЗАДАЧИ 6 (убрана нормализация)
# ---------------------------------------------------------------------------
def load_overlay_config(
    overlay: str,
    base_path: Path = Path("config"),
) -> OverlayConfig:
    # overlay уже приведён к нижнему регистру в _validate_overlays, не нормализуем
    data = load_json_file(base_path / "overlays" / f"{overlay}.json")
    return OverlayConfig(
        name=data.get("name", overlay),
        instructions=tuple(data.get("instructions", [])),
        conflicts_with=tuple(data.get("conflicts_with", [])),
    )


def load_overlay_configs(
    overlays: Sequence[str],
    base_path: Path = Path("config"),
) -> List[OverlayConfig]:
    return [load_overlay_config(overlay, base_path) for overlay in overlays]


# ---------------------------------------------------------------------------
# Кеширующие обёртки для конфигов
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=32)
def _cached_load_domain_config(domain: str, config_path: str) -> DomainConfig:
    return load_domain_config(domain, Path(config_path))


@functools.lru_cache(maxsize=64)
def _cached_load_intent_config(
    intent: Optional[str], config_path: str
) -> Optional[IntentConfig]:
    return load_intent_config(intent, Path(config_path))


# ---------------------------------------------------------------------------
# ИСПРАВЛЕННАЯ ФУНКЦИЯ ДЛЯ ЗАДАЧИ 8
# ---------------------------------------------------------------------------
def load_output_format(
    mode: str,
    base_path: Path = Path("config"),
) -> str:
    data = load_json_file(base_path / "output_format.json")
    mode_instruction = data.get(mode, data.get("text_only", "Верни только отредактированный текст."))
    global_rules = data.get("global_formatting_rules", {})
    if not global_rules:
        return mode_instruction
    global_parts: List[str] = []
    allowed_formatting = global_rules.get("allowed_formatting", "")
    if allowed_formatting:
        global_parts.append(allowed_formatting)
    if not global_parts:
        return mode_instruction
    return "\n".join(global_parts) + "\n\n" + mode_instruction


# ---------------------------------------------------------------------------
# Загрузка базы знаний
# ---------------------------------------------------------------------------
def _normalize_kb_list(items: Any) -> List[Dict[str, Any]]:
    if not isinstance(items, list):
        return []
    result: List[Dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            normalized = dict(item)
            tags = normalized.get("tags")
            if isinstance(tags, list):
                normalized["tags"] = normalize_tags(tags)
            result.append(normalized)
    return result


def _extract_records(container: Any, inherited_tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    if inherited_tags is None:
        inherited_tags = []
    records = []
    if isinstance(container, list):
        for item in container:
            records.extend(_extract_records(item, inherited_tags))
    elif isinstance(container, dict):
        if "examples" in container and isinstance(container["examples"], list):
            cat_tags = container.get("tags", [])
            if isinstance(cat_tags, list):
                new_tags = inherited_tags + cat_tags
            else:
                new_tags = inherited_tags
            for ex in container["examples"]:
                if isinstance(ex, dict):
                    ex = dict(ex)
                    if "tags" in ex and isinstance(ex["tags"], list):
                        ex["tags"] = ex["tags"] + new_tags
                    else:
                        ex["tags"] = new_tags.copy()
                    records.append(ex)
        elif "techniques" in container and isinstance(container["techniques"], list):
            cat_tags = container.get("tags", [])
            if isinstance(cat_tags, list):
                new_tags = inherited_tags + cat_tags
            else:
                new_tags = inherited_tags
            for tech in container["techniques"]:
                if isinstance(tech, dict):
                    tech = dict(tech)
                    if "tags" in tech and isinstance(tech["tags"], list):
                        tech["tags"] = tech["tags"] + new_tags
                    else:
                        tech["tags"] = new_tags.copy()
                    records.append(tech)
        else:
            records.append(container)
    return records


def _load_kb_from_dir(dirpath: Path, key: Optional[str] = None) -> List[Dict[str, Any]]:
    effective_key = key if key is not None else "techniques"
    records: List[Dict[str, Any]] = []
    for filepath in sorted(dirpath.glob("*.json")):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load {filepath}: {e}")
            continue
        logger.debug(f"Loaded KB file: {filepath.name}")
        if isinstance(data, dict):
            if effective_key in data:
                records.append(data)
            else:
                for value in data.values():
                    if isinstance(value, list):
                        records.extend(_extract_records(value))
        elif isinstance(data, list):
            records.extend(_extract_records(data))
        else:
            logger.warning(f"Unexpected format in {filepath}, skipping")
    return records


def _load_kb_list(file_name: str, base_path: Path, key: Optional[str] = None) -> List[Dict[str, Any]]:
    path = base_path / file_name
    if path.is_dir():
        return _load_kb_from_dir(path, key=key)
    if not path.exists():
        logger.warning(f"KB file not found: {path}")
        return []
    data = _load_optional_json(path, {})
    if isinstance(data, list):
        items = data
    elif key is not None:
        items = data.get(key, [])
    else:
        items = []
    records = _extract_records(items)
    if file_name == "grammar_errors.json":
        for rec in records:
            if not rec.get("tags"):
                rec["tags"] = ["grammar"]
    normalized = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        rec = dict(rec)
        if "tags" in rec and isinstance(rec["tags"], list):
            rec["tags"] = normalize_tags(rec["tags"])
        normalized.append(rec)
    return normalized


def _load_kb_file_or_dir(name: str, base_path: Path, key: Optional[str] = None) -> List[Dict[str, Any]]:
    dir_path = base_path / name
    if dir_path.is_dir():
        logger.debug(f"Loading KB from directory: {dir_path}")
        return _load_kb_from_dir(dir_path, key=key)
    file_path = base_path / f"{name}.json"
    if file_path.exists():
        logger.debug(f"Loading KB from file: {file_path}")
        return _load_kb_list(f"{name}.json", base_path, key)
    file_path_exact = base_path / name
    if file_path_exact.exists():
        logger.debug(f"Loading KB from file: {file_path_exact}")
        return _load_kb_list(name, base_path, key)
    logger.warning(f"KB source not found (tried dir and .json): {base_path / name}")
    return []


def _load_kb_multi(
    prefixes: List[str],
    base_path: Path,
    key: str,
    fallback_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for prefix in prefixes:
        file_path = base_path / f"{prefix}.json"
        if file_path.exists():
            records.extend(_load_kb_list(f"{prefix}.json", base_path, key))
    if records:
        return records
    group = prefixes[0].split("_")[0]
    dir_path = base_path / group
    if dir_path.is_dir():
        logger.debug(f"Loading KB from directory: {dir_path}")
        return _load_kb_from_dir(dir_path, key=key)
    if fallback_name:
        fallback_path = base_path / fallback_name
        if fallback_path.exists():
            logger.debug(f"Loading KB from legacy file: {fallback_path}")
            return _load_kb_list(fallback_name, base_path, key)
    logger.warning(
        "KB source not found for prefixes=%s, fallback=%s",
        prefixes,
        fallback_name,
    )
    return []


def load_knowledge_base(base_path: Path = Path("knowledge_base")) -> KnowledgeBase:
    sw_path = base_path / "stop_words.json"
    if not sw_path.exists():
        logger.warning("stop_words.json not found at %s — stop-word filtering disabled", sw_path)
    stop_words = _load_optional_json(sw_path, {})
    domain_glossary = _load_optional_json(base_path / "domain_glossary.json", {})
    nkrj = _load_optional_json(base_path / "nkrj_structure_patterns.json", {})

    grammar_errors = _load_kb_list("grammar_errors.json", base_path, "common_mistakes")
    stylistic_issues = _load_kb_list("stylistic_issues", base_path)
    logic_issues = _load_kb_list("logic_issues.json", base_path, "issues")
    composition_principles = _load_kb_list("composition_principles.json", base_path, "composition_principles")
    local_cohesion = _load_kb_list("local_cohesion.json", base_path, "local_cohesion")
    composition_errors = _load_kb_list("composition_errors.json", base_path, "composition_errors")
    editorial_techniques = _load_kb_list("editorial_techniques", base_path)

    rhetoric_frameworks = _load_kb_multi(
        prefixes=["rhetoric_figures", "rhetoric_topoi", "rhetoric_tropes_and_strategies"],
        base_path=base_path,
        key="frameworks",
        fallback_name="rhetoric_frameworks.json",
    )
    storytelling_frameworks = _load_kb_multi(
        prefixes=["storytelling_macrostructures", "storytelling_microtechniques"],
        base_path=base_path,
        key="frameworks",
        fallback_name="storytelling_frameworks.json",
    )
    marketing_templates = _load_kb_multi(
        prefixes=["marketing_email", "marketing_social", "marketing_web", "marketing_other"],
        base_path=base_path,
        key="templates",
        fallback_name="marketing_templates.json",
    )

    total_records = (
        len(grammar_errors) + len(stylistic_issues) + len(logic_issues) +
        len(storytelling_frameworks) + len(marketing_templates) +
        len(composition_principles) + len(local_cohesion) +
        len(composition_errors) + len(rhetoric_frameworks) +
        len(editorial_techniques)
    )
    logger.info(f"Loaded {total_records} knowledge base records from multiple files")

    return KnowledgeBase(
        stop_words=stop_words,
        grammar_errors=grammar_errors,
        stylistic_issues=stylistic_issues,
        logic_issues=logic_issues,
        storytelling_frameworks=storytelling_frameworks,
        marketing_templates=marketing_templates,
        domain_glossary=domain_glossary,
        composition_principles=composition_principles,
        local_cohesion=local_cohesion,
        composition_errors=composition_errors,
        rhetoric_frameworks=rhetoric_frameworks,
        editorial_techniques=editorial_techniques,
        nkrj_structure_patterns=nkrj,
    )


# ---------------------------------------------------------------------------
# Вспомогательные функции для сборки тегов и блоков
# ---------------------------------------------------------------------------
def _collect_retrieval_tags(
    domain: str,
    intent: Optional[str],
    overlays: Sequence[str],
) -> Dict[str, Set[str]]:
    primary: Set[str] = set()
    expanded: Set[str] = set()
    primary.update(get_primary_tags_for_category("domains", domain))
    expanded.update(get_canonical_tags_for_category("domains", domain))
    if intent and intent != "neutral":
        primary.update(get_primary_tags_for_category("intents", intent))
        expanded.update(get_canonical_tags_for_category("intents", intent))
    for overlay in overlays:
        primary.update(get_primary_tags_for_category("overlays", overlay))
        expanded.update(get_canonical_tags_for_category("overlays", overlay))
    primary.update({"grammar", "style", "editing", "clarity"})
    return {"primary": primary, "expanded": expanded - primary}


def _append_rule_entries(
    lines: List[str],
    title: str,
    entries: List[Dict[str, Any]],
) -> None:
    if not entries:
        return
    lines.append(title)
    for entry in entries:
        wrong = entry.get("wrong")
        correct = entry.get("correct")
        rule = entry.get("rule") or entry.get("description") or entry.get("name")
        fragments: List[str] = []
        if wrong:
            fragments.append(f"плохо: {wrong}")
        if correct:
            fragments.append(f"лучше: {correct}")
        if rule:
            fragments.append(f"пояснение: {rule}")
        if fragments:
            lines.append("- " + "; ".join(fragments))


def _append_structural_entries(
    lines: List[str],
    title: str,
    entries: List[Dict[str, Any]],
) -> None:
    if not entries:
        return
    lines.append(title)
    for entry in entries:
        name = entry.get("name", "")
        description = entry.get("description", "")
        when_to_use = entry.get("when_to_use", "")
        if isinstance(when_to_use, list):
            when_to_use = "; ".join(str(item) for item in when_to_use[:3])
        fragments: List[str] = []
        if name:
            fragments.append(str(name))
        if description:
            fragments.append(str(description))
        if when_to_use:
            fragments.append(f"когда применять: {when_to_use}")
        if fragments:
            lines.append("- " + " | ".join(fragments))


def _append_editorial_entries(
    lines: List[str],
    title: str,
    entries: List[Dict[str, Any]],
) -> None:
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
        fragments: List[str] = []
        if name:
            fragments.append(str(name))
        if description:
            fragments.append(str(description))
        if how_to_apply_str:
            fragments.append(f"как применять: {how_to_apply_str}")
        if fragments:
            lines.append("- " + " | ".join(fragments))


def _append_glossary(
    lines: List[str],
    glossary: Dict[str, Any],
    limit: int,
) -> None:
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
    if intent not in _KNOWLEDGE_DEPENDENT_INTENTS:
        return
    logger.warning(
        "KB retrieval empty: block=%s, stage=%s, domain=%s, intent=%s, "
        "overlays=%s, text_length=%d, primary_tags=%s",
        block,
        stage.value,
        domain,
        intent,
        overlays,
        text_len,
        sorted(primary_tags),
    )


# ---------------------------------------------------------------------------
# Реестр блоков базы знаний
# ---------------------------------------------------------------------------
KB_BLOCK_REGISTRY: List[KBBlockConfig] = [
    KBBlockConfig(
        name="grammar",
        budget_key="grammar",
        retrieval_fn=select_grammar_rules,
        append_fn=_append_rule_entries,
        title="Грамматические ориентиры:",
        kb_attr=None,
        uses_structural_call=False,
        candidate_attr="grammar_candidates",
    ),
    KBBlockConfig(
        name="style",
        budget_key="style",
        retrieval_fn=select_style_issues,
        append_fn=_append_rule_entries,
        title="Стилистические ориентиры:",
        kb_attr=None,
        uses_structural_call=False,
        candidate_attr="style_candidates",
    ),
    KBBlockConfig(
        name="logic",
        budget_key="logic",
        retrieval_fn=select_logic_issues,
        append_fn=_append_rule_entries,
        title="Логические ориентиры:",
        kb_attr=None,
        uses_structural_call=False,
        candidate_attr="logic_candidates",
    ),
    KBBlockConfig(
        name="composition",
        budget_key="composition",
        retrieval_fn=select_structural_by_tags_or_all,
        append_fn=_append_structural_entries,
        title="Принципы композиции:",
        kb_attr="composition_principles",
        uses_structural_call=True,
        candidate_attr=None,
    ),
    KBBlockConfig(
        name="composition_errors",
        budget_key="composition_errors",
        retrieval_fn=select_structural_by_tags_or_all,
        append_fn=_append_structural_entries,
        title="Ошибки композиции:",
        kb_attr="composition_errors",
        uses_structural_call=True,
        candidate_attr=None,
    ),
    KBBlockConfig(
        name="cohesion",
        budget_key="cohesion",
        retrieval_fn=select_structural_by_tags_or_all,
        append_fn=_append_structural_entries,
        title="Локальная связность:",
        kb_attr="local_cohesion",
        uses_structural_call=True,
        candidate_attr=None,
    ),
    KBBlockConfig(
        name="storytelling",
        budget_key="storytelling",
        retrieval_fn=select_structural_by_tags_or_all,
        append_fn=_append_structural_entries,
        title="Сторителлинг-фреймворки:",
        kb_attr="storytelling_frameworks",
        uses_structural_call=True,
        candidate_attr=None,
    ),
    KBBlockConfig(
        name="marketing",
        budget_key="marketing",
        retrieval_fn=select_structural_by_tags_or_all,
        append_fn=_append_structural_entries,
        title="Маркетинговые шаблоны:",
        kb_attr="marketing_templates",
        uses_structural_call=True,
        candidate_attr=None,
    ),
    KBBlockConfig(
        name="rhetoric",
        budget_key="rhetoric",
        retrieval_fn=select_structural_by_tags_or_all,
        append_fn=_append_structural_entries,
        title="Риторические приёмы:",
        kb_attr="rhetoric_frameworks",
        uses_structural_call=True,
        candidate_attr=None,
    ),
    KBBlockConfig(
        name="editorial",
        budget_key="editorial",
        retrieval_fn=select_structural_by_tags_or_all,
        append_fn=_append_editorial_entries,
        title="Редакторские приёмы:",
        kb_attr="editorial_techniques",
        uses_structural_call=True,
        candidate_attr=None,
    ),
]


# ---------------------------------------------------------------------------
# Обработчик одного блока KB
# ---------------------------------------------------------------------------
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
            candidate_limit = 10
        result = config.retrieval_fn(
            kb=kb,
            text=text,
            tags=primary_tags,
            limit=budget.entry_limit,
            candidate_limit=candidate_limit,
            char_budget=budget.char_budget,
            return_meta=True,
        )
    else:
        if not config.kb_attr:
            return total_few_shot_used
        entries_source = getattr(kb, config.kb_attr, None)
        if entries_source is None:
            return total_few_shot_used
        result = config.retrieval_fn(
            entries=entries_source,
            tags=primary_tags,
            limit=budget.entry_limit,
            expanded_tags=expanded_tags,
            char_budget=budget.char_budget,
            return_meta=True,
        )

    entries, stage, dropped = _unpack_retrieval_result(result)

    pair_entries = [e for e in entries if _has_few_shot_pair(e)]
    rule_entries = [e for e in entries if not _has_few_shot_pair(e)]

    few_shot_examples = []
    if include_few_shot:
        allowed = min(
            FEW_SHOT_MAX_EXAMPLES_PER_BLOCK,
            FEW_SHOT_MAX_TOTAL_EXAMPLES - total_few_shot_used
        )
        if allowed > 0:
            few_shot_examples = _select_few_shot_examples(
                pair_entries,
                allowed,
                seed=few_shot_seed,
            )

    confidence_note = _get_confidence_note(stage)
    if (rule_entries or few_shot_examples) and confidence_note:
        lines.append(confidence_note)

    if FEW_SHOT_RULES_FIRST:
        if rule_entries:
            config.append_fn(lines, config.title, rule_entries)
        if few_shot_examples:
            lines.append("Примеры редактирования:")
            for ex in few_shot_examples:
                lines.append(_format_few_shot_example(ex))
            lines.append("")
    else:
        if few_shot_examples:
            lines.append("Примеры редактирования:")
            for ex in few_shot_examples:
                lines.append(_format_few_shot_example(ex))
            lines.append("")
        if rule_entries:
            config.append_fn(lines, config.title, rule_entries)

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
        logger.info(
            "Char budget truncated %d records for block='%s', domain=%s, intent=%s",
            dropped, config.name, domain, intent
        )
    if stage in (FallbackStage.EMPTY, FallbackStage.NEUTRAL):
        _warn_if_empty_retrieval(
            block=config.name,
            stage=stage,
            domain=domain,
            intent=intent,
            overlays=overlays,
            text_len=len(text),
            primary_tags=primary_tags,
        )

    return total_few_shot_used


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------
class PromptBuilder:
    """
    Фасад для сборки промпта из конфигов и базы знаний.
    """

    def __init__(
        self,
        config_path: Path = Path("config"),
        kb_path: Path = Path("knowledge_base"),
        limits: Optional[LimitsConfig] = None,
    ) -> None:
        self.config_path = config_path
        self.kb_path = kb_path
        self._limits = limits or LimitsConfig()
        self.core_config: Optional[CoreConfig] = None
        self.knowledge_base: Optional[KnowledgeBase] = None

    # ------------------------------------------------------------------
    # Startup / reload
    # ------------------------------------------------------------------
    def startup_check(self) -> None:
        self.core_config = load_core_config(self.config_path)
        self.knowledge_base = load_knowledge_base(self.kb_path)

    def reload_configs(self) -> None:
        _cached_load_domain_config.cache_clear()
        _cached_load_intent_config.cache_clear()
        self.core_config = load_core_config(self.config_path)
        self.knowledge_base = load_knowledge_base(self.kb_path)

    # ------------------------------------------------------------------
    # Доступные intents / overlays
    # ------------------------------------------------------------------
    def get_available_intents(self) -> Set[str]:
        intents_dir = self.config_path / "intents"
        if not intents_dir.exists():
            return set(ALLOWED_INTENTS)
        values = {path.stem for path in intents_dir.glob("*.json")}
        return values or set(ALLOWED_INTENTS)

    def getavailableintents(self) -> Set[str]:
        warnings.warn(
            "getavailableintents() is deprecated, use get_available_intents()",
            DeprecationWarning,
            stacklevel=2
        )
        return self.get_available_intents()

    def get_available_overlays(self) -> Set[str]:
        overlays_dir = self.config_path / "overlays"
        if not overlays_dir.exists():
            return set(ALLOWED_OVERLAYS)
        values = {path.stem for path in overlays_dir.glob("*.json")}
        return values or set(ALLOWED_OVERLAYS)

    def getavailableoverlays(self) -> Set[str]:
        warnings.warn(
            "getavailableoverlays() is deprecated, use get_available_overlays()",
            DeprecationWarning,
            stacklevel=2
        )
        return self.get_available_overlays()

    # ------------------------------------------------------------------
    # Валидация
    # ------------------------------------------------------------------
    def _validate_domain(self, domain: str) -> str:
        normalized = domain.strip().lower()
        if normalized not in ALLOWED_DOMAINS:
            raise ValueError(
                f"Unsupported domain: {domain!r}. "
                f"Must be one of {sorted(ALLOWED_DOMAINS)}"
            )
        return normalized

    def _validate_intent(self, intent: Optional[str]) -> Optional[str]:
        if intent is None or not intent.strip():
            return None
        normalized = normalize_tag(intent)
        available = set(self.get_available_intents()) | ALLOWED_INTENTS
        if normalized not in available:
            raise ValueError(
                f"Unsupported intent: {intent!r}. "
                f"Must be one of {sorted(available)}"
            )
        return normalized

    # ---------------------------------------------------------------------------
    # ИСПРАВЛЕННАЯ ВАЛИДАЦИЯ ОВЕРЛЕЕВ (задача 6)
    # ---------------------------------------------------------------------------
    def _validate_overlays(self, overlays: Sequence[str]) -> List[str]:
        # Приводим к нижнему регистру, но не нормализуем (чтобы сохранить подчёркивания)
        normalized = [o.lower() for o in overlays]
        available = set(self.get_available_overlays()) | ALLOWED_OVERLAYS
        invalid = [item for item in normalized if item not in available]
        if invalid:
            raise ValueError(
                f"Unsupported overlays: {invalid}. "
                f"Must be from {sorted(available)}"
            )

        # Проверка конфликтов
        overlay_configs = load_overlay_configs(normalized, self.config_path)
        for ov_cfg in overlay_configs:
            for conflict in ov_cfg.conflicts_with:
                if conflict.lower() in normalized:
                    raise ValueError(
                        f"Overlays conflict: '{ov_cfg.name}' and '{conflict}' "
                        f"cannot be used together. Choose one."
                    )
        return normalized

    def _validate_output_mode(self, output_mode: str) -> str:
        normalized = output_mode.strip().lower()
        if normalized not in ALLOWED_OUTPUT_MODES:
            raise ValueError(
                f"Unsupported output_mode: {output_mode!r}. "
                f"Must be one of {sorted(ALLOWED_OUTPUT_MODES)}"
            )
        return normalized

    # ------------------------------------------------------------------
    # Блоки промпта
    # ------------------------------------------------------------------
    def _build_audience_block(self, audience: Optional[AudienceProfile]) -> str:
        if audience is None:
            return ""
        parts = [
            f"Тип аудитории: {audience.kind}",
            f"Уровень экспертизы: {audience.expertise}",
            f"Формальность: {audience.formality}",
        ]
        if getattr(audience, "description", ""):
            parts.append(f"Описание аудитории: {audience.description}")
        return "\n".join(parts)

    # ---- НОВЫЙ МЕТОД (Задача 2) ----
    def _build_mode_constraints_block(self, domain_config: DomainConfig) -> str:
        """
        Формирует блок явных ограничений режима на основе флагов домена.
        Всегда добавляется в промпт, если хотя бы один флаг False.
        """
        lines: List[str] = []
        if not domain_config.allow_storytelling:
            lines.append(
                "Сторителлинг запрещён: не добавляй нарративные отступления, "
                "личные истории и метафорические сравнения, уводящие от сути."
            )
        if not domain_config.allow_marketing:
            lines.append(
                "Маркетинг запрещён: удаляй призывы к действию, триггерные слова "
                "(«уникальный», «лучший», «срочно») и конструкции давления на читателя."
            )
        if not lines:
            return ""
        return "Режимные ограничения:\n- " + "\n- ".join(lines)
    # ---- КОНЕЦ НОВОГО МЕТОДА ----

    # ---- НОВЫЙ МЕТОД (Задача 5) ----
    def _build_ip_ceiling_block(self, domain_config: DomainConfig) -> str:
        """Формирует блок с целевым значением ИП."""
        effective_ceiling = (
            domain_config.ip_ceiling
            if domain_config.ip_ceiling is not None
            else (self.core_config.ip_ceiling if self.core_config else 2.5)
        )
        return (
            f"Целевой Индекс пластиковости (ИП): ≤ {effective_ceiling}. "
            "После редактирования укажи итоговый ИП. "
            "Если ИП превышает целевое значение — предупреди и предложи второй проход."
        )
    # ---- КОНЕЦ НОВОГО МЕТОДА ----

    def _ensure_knowledge_base(self) -> Optional[KnowledgeBase]:
        if self.knowledge_base is not None:
            return self.knowledge_base
        try:
            self.knowledge_base = load_knowledge_base(self.kb_path)
        except FileNotFoundError:
            logger.warning("Knowledge base directory not found: %s", self.kb_path)
            return None
        return self.knowledge_base

    # ------------------------------------------------------------------
    # _build_knowledge_block — версия с реестром
    # ------------------------------------------------------------------
    def _build_knowledge_block(
        self,
        text: str,
        primary_tags: Set[str],
        expanded_tags: Set[str],
        budget: KnowledgeBudget,
        domain: str,
        intent: Optional[str],
        overlays: List[str],
        include_few_shot: bool,
        total_few_shot_used: int,
        few_shot_seed: Optional[int] = None,
    ) -> Tuple[str, Dict[str, Any], int]:
        kb = self._ensure_knowledge_base()
        if kb is None:
            return "", {}, total_few_shot_used

        lines: List[str] = []
        meta: Dict[str, Any] = {}
        current_total = total_few_shot_used

        # Стоп-слова
        stop_words_budget = budget.get("stop_words")
        if stop_words_budget and stop_words_budget.enabled and kb.stop_words:
            lines.append("Стоп-слова и нежелательные формулировки:")
            category_limit = stop_words_budget.entry_limit or self._limits.stop_words_category
            for category, words in list(kb.stop_words.items())[:category_limit]:
                if isinstance(words, list) and words:
                    joined = ", ".join(str(w) for w in words[:self._limits.stop_words_items])
                    lines.append(f"- {category}: {joined}")

        # Основные блоки через реестр
        for block_cfg in KB_BLOCK_REGISTRY:
            block_budget = budget.get(block_cfg.budget_key)
            if not (block_budget and block_budget.enabled):
                continue
            if block_cfg.uses_structural_call and block_cfg.kb_attr:
                if not getattr(kb, block_cfg.kb_attr, None):
                    continue
            current_total = _process_kb_block(
                config=block_cfg,
                lines=lines,
                meta=meta,
                kb=kb,
                text=text,
                primary_tags=primary_tags,
                expanded_tags=expanded_tags,
                budget=block_budget,
                domain=domain,
                intent=intent,
                overlays=overlays,
                include_few_shot=include_few_shot,
                total_few_shot_used=current_total,
                limits=self._limits,
                few_shot_seed=few_shot_seed,
            )

        # Глоссарий
        glossary_budget = budget.get("glossary")
        if glossary_budget and glossary_budget.enabled and kb.domain_glossary:
            _append_glossary(lines, kb.domain_glossary, glossary_budget.entry_limit)

        # НКРЯ
        nkrj_budget = budget.get("nkrj")
        if nkrj_budget and nkrj_budget.enabled and kb.nkrj_structure_patterns:
            _append_nkrj(lines, kb.nkrj_structure_patterns)

        return "\n".join(lines), meta, current_total

    # ------------------------------------------------------------------
    # Вспомогательный метод для сборки финального промпта
    # ------------------------------------------------------------------
    def _assemble_prompt(self, blocks: List[str]) -> str:
        """Собирает финальный промпт из списка блоков."""
        return "\n\n".join(block for block in blocks if block.strip())

    # ------------------------------------------------------------------
    # Главный метод — build()
    # ------------------------------------------------------------------
    def build(
        self,
        text: str,
        domain: str,
        intent: Optional[str] = None,
        audience: Optional[AudienceProfile] = None,
        overlays: Optional[Sequence[str]] = None,
        output_mode: str = "text_only",
        include_knowledge: bool = True,
        include_few_shot: bool = True,
        knowledge_level: KnowledgeLevel = KnowledgeLevel.STANDARD,
        token_budget: Optional[int] = None,
        include_retrieval_meta: bool = False,
        few_shot_seed: Optional[int] = None,
        **legacy_kwargs: Any,
    ) -> Union[str, Tuple[str, Dict[str, Any]]]:
        """
        Собирает промпт из конфигов и базы знаний.
        """
        # Поддержка legacy camelCase kwargs
        legacy_output_mode = legacy_kwargs.pop("outputmode", None)
        legacy_include_knowledge = legacy_kwargs.pop("includeknowledge", None)

        if legacy_kwargs:
            unknown = ", ".join(sorted(legacy_kwargs))
            raise TypeError(f"Unexpected keyword arguments: {unknown}")

        if legacy_output_mode is not None:
            output_mode = legacy_output_mode
        if legacy_include_knowledge is not None:
            include_knowledge = legacy_include_knowledge

        if not text or not text.strip():
            raise ValueError("Text must not be empty")

        validated_domain = self._validate_domain(domain)
        validated_intent = self._validate_intent(intent)
        validated_overlays = self._validate_overlays(overlays or [])
        validated_output_mode = self._validate_output_mode(output_mode)

        if self.core_config is None:
            self.core_config = load_core_config(self.config_path)

        domain_config = _cached_load_domain_config(validated_domain, str(self.config_path))
        intent_config = _cached_load_intent_config(validated_intent, str(self.config_path))
        overlay_configs = load_overlay_configs(validated_overlays, self.config_path)
        output_format = load_output_format(validated_output_mode, self.config_path)

        blocks: List[str] = []

        blocks.append(f"Роль: {self.core_config.role}")
        blocks.append(f"Приоритеты: {self.core_config.priorities}")
        blocks.append(f"Домен: {domain_config.name}")
        blocks.append(f"Тон: {domain_config.tone}")

        if domain_config.system_rules:
            blocks.append("Правила домена:\n" + domain_config.system_rules)

        # ---- ВСТАВКА НОВОГО БЛОКА (Задача 2) ----
        mode_constraints = self._build_mode_constraints_block(domain_config)
        if mode_constraints:
            blocks.append(mode_constraints)
        # -----------------------------------------

        # ---- НОВЫЕ БЛОКИ (Задача 4) ----
        if domain_config.tasks:
            blocks.append(
                "Задачи редактора в этом домене:\n- "
                + "\n- ".join(domain_config.tasks)
            )
        if domain_config.constraints:
            blocks.append(
                "Ограничения домена:\n- "
                + "\n- ".join(domain_config.constraints)
            )
        # ---------------------------------

        if self.core_config.basic_audit_instructions:
            blocks.append(
                "Базовые инструкции:\n- "
                + "\n- ".join(self.core_config.basic_audit_instructions)
            )

        if self.core_config.forbidden:
            blocks.append(
                "Запрещено:\n- " + "\n- ".join(self.core_config.forbidden)
            )

        if intent_config and intent_config.instructions:
            blocks.append(
                f"Intent: {intent_config.name}\n- "
                + "\n- ".join(intent_config.instructions)
            )

        if overlay_configs:
            overlay_lines: List[str] = []
            for overlay in overlay_configs:
                if overlay.instructions:
                    overlay_lines.append(
                        f"[{overlay.name}] " + " | ".join(overlay.instructions)
                    )
            if overlay_lines:
                blocks.append("Overlay-инструкции:\n- " + "\n- ".join(overlay_lines))

        audience_block = self._build_audience_block(audience)
        if audience_block:
            blocks.append("Аудитория:\n" + audience_block)

        retrieval_meta_total: Dict[str, Any] = {}
        if include_knowledge:
            tag_sets = _collect_retrieval_tags(
                validated_domain,
                validated_intent,
                validated_overlays,
            )
            budget = KnowledgeBudgetManager(token_budget).allocate(
                limits=self._limits,
                level=knowledge_level,
            )
            if not domain_config.allow_storytelling:
                budget.disable("storytelling")
            if not domain_config.allow_marketing:
                budget.disable("marketing")

            effective_seed = few_shot_seed if few_shot_seed is not None else _derive_seed(text)

            knowledge_block, block_meta, _ = self._build_knowledge_block(
                text=text,
                primary_tags=tag_sets["primary"],
                expanded_tags=tag_sets["expanded"],
                budget=budget,
                domain=validated_domain,
                intent=validated_intent,
                overlays=validated_overlays,
                include_few_shot=include_few_shot,
                total_few_shot_used=0,
                few_shot_seed=effective_seed,
            )
            retrieval_meta_total = block_meta
            if knowledge_block:
                blocks.append("База знаний:\n" + knowledge_block)

        # ---- НОВЫЙ БЛОК (Задача 5) ----
        blocks.append(self._build_ip_ceiling_block(domain_config))
        # --------------------------------

        blocks.append("Формат ответа:\n" + output_format)
        blocks.append("Исходный текст:\n" + text.strip())

        # Используем вспомогательный метод для сборки промпта
        prompt = self._assemble_prompt(blocks)

        if include_retrieval_meta:
            return prompt, retrieval_meta_total
        return prompt

    # legacy alias
    def build_prompt(self, **kwargs: Any) -> str:
        warnings.warn(
            "build_prompt() is deprecated, use build()",
            DeprecationWarning,
            stacklevel=2
        )
        return self.build(**kwargs)