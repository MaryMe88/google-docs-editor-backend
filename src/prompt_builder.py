# src/prompt_builder.py

"""
prompt_builder.py

Модуль для сборки финальных промптов из конфигов и базы знаний.
Четвёртая итерация: кэширование, централизованный доступ к конфигам,
канонический registry.
"""
from __future__ import annotations

import functools
import hashlib
import json
import logging
import random
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union, overload, Literal

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
    FileCache,
    CachePolicy,
    FeatureResolutionResult,
    AssemblyBlockDiagnostics,
    AssemblyTrace,
)
from src.reason_codes import ReasonCode
from src.registry import (
    CANONICAL_FEATURE_ALIASES,
    KNOWN_FEATURE_ALIASES,
    get_features_from_tags,
    check_alias_consistency,
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
from src.kb_manifest_loader import load_manifest, select_files_for_request, ManifestEntry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Константы для валидации kb_limits
# ---------------------------------------------------------------------------
ALLOWED_KB_LIMIT_KEYS: frozenset = frozenset({
    "grammar", "style", "logic", "composition", "cohesion", "local_cohesion",
    "composition_errors", "storytelling", "marketing", "rhetoric", "editorial",
    "glossary", "stop_words", "stop_words_items", "nkrj",
    "grammar_candidates", "style_candidates", "logic_candidates",
    "storytelling_candidates", "marketing_candidates", "rhetoric_candidates",
})
# ИЗМЕНЕНИЕ (Итерация 5): разрешаем 0 как допустимое значение для отключения категории
KB_LIMIT_MIN: int = 0
KB_LIMIT_MAX: int = 100

# ---------------------------------------------------------------------------
# КАНОНИЧЕСКИЙ СЛОЙ: алиасы для фич (уже в registry, оставляем для совместимости)
# ---------------------------------------------------------------------------
# Теперь импортируем из registry, но сохраняем ссылки для обратной совместимости
_TAG_TO_FEATURE = {alias: feature for feature, aliases in CANONICAL_FEATURE_ALIASES.items() for alias in aliases}

# ---------------------------------------------------------------------------
# Вспомогательные функции нормализации
# ---------------------------------------------------------------------------
def normalize_intent(intent: Optional[str]) -> Optional[str]:
    if intent is None or intent == "neutral":
        return None
    normalized = intent.lower().strip()
    if normalized not in ALLOWED_INTENTS:
        logger.warning(f"Unknown intent '{normalized}', treating as neutral.")
        return None
    return normalized

def normalize_overlays(overlays: Sequence[str]) -> List[str]:
    result = []
    for ov in overlays:
        norm = ov.lower().strip()
        if norm in ALLOWED_OVERLAYS:
            result.append(norm)
        else:
            logger.warning(f"Unknown overlay '{norm}', ignoring.")
    return result

# (get_features_from_tags импортируется из registry)

# ---------------------------------------------------------------------------
# Дефолтные конфиги
# ---------------------------------------------------------------------------
_DEFAULT_DOMAIN_CONFIG: DomainConfig = DomainConfig(
    name="general",
    system_rules="",
    tone="neutral",
    allow_storytelling=False,
    allow_marketing=False,
    tasks=(),
    constraints=(),
    ip_ceiling=None,
    kb_limits={},
    priority=100,
    suppresses=(),
    conflicts_with=(),
    incompatible_intents=(),
    incompatible_overlays=(),
)

def _make_default_overlay_config(name: str) -> OverlayConfig:
    return OverlayConfig(
        name=name,
        instructions=(),
        conflicts_with=(),
        priority=70,
        suppresses=(),
    )

# ---------------------------------------------------------------------------
# JSON-загрузчики и хелперы (остаются публичными)
# ---------------------------------------------------------------------------
def normalize_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if isinstance(item, str):
            stripped = item.strip()
            if stripped:
                result.append(stripped.lower())
    return result

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

def load_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

def _load_optional_json(path: Path, default: Any) -> Any:
    if path.exists():
        return load_json_file(path)
    return default

# ---------------------------------------------------------------------------
# Загрузчики конфигов (публичные, используются также извне)
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

def load_domain_config(domain: str, base_path: Path = Path("config")) -> DomainConfig:
    normalized_domain = domain.strip().lower()
    domain_path = base_path / "domains" / f"{normalized_domain}.json"
    if not domain_path.exists():
        logger.warning(
            "load_domain_config: file not found for domain=%r at %s — "
            "using default domain config.",
            normalized_domain, domain_path,
        )
        return _DEFAULT_DOMAIN_CONFIG
    data = load_json_file(domain_path)
    raw_tasks = data.get("tasks", [])
    raw_constraints = data.get("constraints", [])
    raw_ip = data.get("ip_ceiling")
    domain_ip_ceiling: Optional[float] = None
    if isinstance(raw_ip, (int, float)):
        domain_ip_ceiling = float(raw_ip)
    elif isinstance(raw_ip, dict):
        domain_ip_ceiling = float(raw_ip.get("value", 2.5))

    raw_kb_limits = data.get("kb_limits", {})
    kb_limits: Dict[str, int] = {}
    if isinstance(raw_kb_limits, dict):
        domain_name = data.get("name", normalized_domain)
        for k, v in raw_kb_limits.items():
            if not isinstance(k, str):
                logger.warning("kb_limits[%r] в домене '%s': ключ не строка — пропущен", k, domain_name)
                continue
            if k not in ALLOWED_KB_LIMIT_KEYS:
                logger.warning(
                    "kb_limits: неизвестный ключ '%s' в домене '%s' — проигнорирован. "
                    "Допустимые: %s",
                    k, domain_name, ", ".join(sorted(ALLOWED_KB_LIMIT_KEYS))
                )
                continue
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                logger.warning("kb_limits['%s'] в домене '%s': значение %r не число — пропущено", k, domain_name, v)
                continue
            value = int(v)
            clamped = max(KB_LIMIT_MIN, min(KB_LIMIT_MAX, value))
            if clamped != value:
                logger.warning("kb_limits['%s']=%d в домене '%s' вне диапазона [%d, %d] — приведено к %d",
                               k, value, domain_name, KB_LIMIT_MIN, KB_LIMIT_MAX, clamped)
            kb_limits[k] = clamped

    priority = data.get("priority", 100)
    if not isinstance(priority, int):
        priority = 100
    suppresses = tuple(normalize_string_list(data.get("suppresses", [])))
    conflicts_with = tuple(normalize_string_list(data.get("conflicts_with", [])))
    incompatible_intents = tuple(normalize_string_list(data.get("incompatible_intents", [])))
    incompatible_overlays = tuple(normalize_string_list(data.get("incompatible_overlays", [])))

    return DomainConfig(
        name=data.get("name", normalized_domain),
        system_rules=data.get("system_rules", ""),
        tone=data.get("tone", "neutral"),
        allow_storytelling=data.get("allow_storytelling", False),
        allow_marketing=data.get("allow_marketing", False),
        tasks=tuple(t for t in raw_tasks if isinstance(t, str)),
        constraints=tuple(c for c in raw_constraints if isinstance(c, str)),
        ip_ceiling=domain_ip_ceiling,
        kb_limits=kb_limits,
        priority=priority,
        suppresses=suppresses,
        conflicts_with=conflicts_with,
        incompatible_intents=incompatible_intents,
        incompatible_overlays=incompatible_overlays,
    )

def load_intent_config(intent: Optional[str], base_path: Path = Path("config")) -> Optional[IntentConfig]:
    if intent is None or intent == "neutral":
        return None
    normalized = normalize_tag(intent)
    intent_path = base_path / "intents" / f"{normalized}.json"
    if not intent_path.exists():
        logger.warning("load_intent_config: file not found for intent=%r — skipping.", normalized)
        return None
    data = load_json_file(intent_path)
    priority = data.get("priority", 50)
    if not isinstance(priority, int):
        priority = 50
    suppresses = tuple(normalize_string_list(data.get("suppresses", [])))
    conflicts_with = tuple(normalize_string_list(data.get("conflicts_with", [])))
    return IntentConfig(
        name=data.get("name", normalized),
        instructions=tuple(data.get("instructions", [])),
        priority=priority,
        suppresses=suppresses,
        conflicts_with=conflicts_with,
    )

def load_overlay_config(overlay: str, base_path: Path = Path("config")) -> OverlayConfig:
    overlay_path = base_path / "overlays" / f"{overlay}.json"
    if not overlay_path.exists():
        logger.warning("load_overlay_config: file not found for overlay=%r — using default.", overlay)
        return _make_default_overlay_config(overlay)
    data = load_json_file(overlay_path)
    priority = data.get("priority", 70)
    if not isinstance(priority, int):
        priority = 70
    suppresses = tuple(normalize_string_list(data.get("suppresses", [])))
    conflicts_with = tuple(normalize_string_list(data.get("conflicts_with", [])))
    return OverlayConfig(
        name=data.get("name", overlay),
        instructions=tuple(data.get("instructions", [])),
        conflicts_with=conflicts_with,
        priority=priority,
        suppresses=suppresses,
    )

def load_overlay_configs(overlays: Sequence[str], base_path: Path = Path("config")) -> List[OverlayConfig]:
    return [load_overlay_config(ov, base_path) for ov in overlays]

def load_output_format(mode: str, base_path: Path = Path("config")) -> str:
    data = load_json_file(base_path / "output_format.json")
    mode_instruction = data.get(mode, data.get("text_only", "Верни только отредактированный текст."))
    global_rules = data.get("global_formatting_rules", {})
    known_keys = {"allowed_formatting"}
    unknown_keys = set(global_rules.keys()) - known_keys
    if unknown_keys:
        logger.warning("load_output_format: ключи %s в 'global_formatting_rules' не используются.", unknown_keys)
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
# Загрузка KB (публичная)
# ---------------------------------------------------------------------------
def _load_kb_file(
    path: Path,
    expected_key: Optional[str] = None,
    use_known_keys: bool = True,
) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    if not path.exists():
        logger.warning("KB file not found: %s", path)
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Failed to load KB file %s: %s", path, e)
        return []
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        logger.warning("KB file %s has unexpected top-level type %s", path, type(data).__name__)
        return []
    if expected_key and isinstance(data.get(expected_key), list):
        return data[expected_key]
    if not use_known_keys:
        logger.debug("KB file %s treated as dict block (use_known_keys=False)", path)
        return data
    known_list_keys = ("items", "examples", "techniques", "frameworks", "templates", "common_mistakes", "issues", "rules", "entries")
    for key in known_list_keys:
        value = data.get(key)
        if isinstance(value, list):
            return value
    list_valued_keys = [k for k, v in data.items() if isinstance(v, list)]
    if len(list_valued_keys) == 1:
        return data[list_valued_keys[0]]
    logger.debug("KB file %s treated as dict block (no list key found)", path)
    return data

def load_knowledge_base(
    kb_path: Path,
    active_tags: Optional[Set[str]] = None,
    intent: Optional[str] = None,
    load_all: bool = False,
) -> KnowledgeBase:
    manifest = load_manifest(kb_path / "kb_manifest.json")
    if load_all:
        if manifest:
            selected = list(manifest)
        else:
            selected = []
            for json_path in kb_path.rglob("*.json"):
                rel_path = json_path.relative_to(kb_path)
                stem = json_path.stem
                block_type = "dict" if stem in ("stop_words", "nkrj_structure_patterns", "domain_glossary") else "list"
                entry = ManifestEntry(
                    file=str(rel_path),
                    stage="default",
                    load_mode="always",
                    tags=[],
                    intents=[],
                    budget_weight="medium",
                    status="active",
                    priority=99,
                    block_name=None,
                    block_type=block_type,
                )
                selected.append(entry)
    else:
        selected = select_files_for_request(manifest, active_tags or set(), intent)

    block_data: Dict[str, Any] = {}
    for entry in selected:
        full_path = kb_path / entry.file
        if not full_path.exists():
            logger.warning("KB file not found: %s", full_path)
            continue
        if entry.block_name:
            key = entry.block_name
        elif "/" in entry.file:
            key = entry.file.split("/")[0]
        else:
            key = Path(entry.file).stem
        if getattr(entry, "block_type", "list") == "dict":
            records = _load_kb_file(full_path, expected_key=None, use_known_keys=False)
        else:
            records = _load_kb_file(full_path, expected_key=entry.block_name or Path(entry.file).stem)
        if not records:
            continue
        if isinstance(records, dict):
            if key in block_data and isinstance(block_data[key], dict):
                block_data[key].update(records)
            elif key in block_data:
                logger.warning("KB block '%s' type conflict: existing=%s, new=dict — skipping %s",
                               key, type(block_data[key]).__name__, entry.file)
            else:
                block_data[key] = records
        else:
            if key not in block_data:
                block_data[key] = []
            if isinstance(block_data[key], list):
                block_data[key].extend(records)
            else:
                logger.warning("KB block '%s' type conflict: existing=%s, new=list — skipping %s",
                               key, type(block_data[key]).__name__, entry.file)

    if "domain_glossary" not in block_data:
        domain_glossary_path = kb_path / "domain_glossary.json"
        if domain_glossary_path.exists():
            try:
                data = json.loads(domain_glossary_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    block_data["domain_glossary"] = data
            except Exception as e:
                logger.error("Failed to load domain_glossary.json: %s", e)

    kb = KnowledgeBase()
    for key, records in block_data.items():
        kb.register(key, records)
    logger.info("Loaded KB with %d blocks from manifest (selected %d files)", len(block_data), len(selected))
    return kb

# ---------------------------------------------------------------------------
# NEW: Определение KBBlockConfig
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Вспомогательные функции сборки
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Реестр блоков базы знаний
# ---------------------------------------------------------------------------
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
    KBBlockConfig(name="rhetoric", budget_key="rhetoric", retrieval_fn=select_structural_by_tags_or_all,
                  append_fn=_append_structural_entries, title="Риторические приёмы:",
                  kb_attr="rhetoric_frameworks", uses_structural_call=True, candidate_attr=None),
    KBBlockConfig(name="editorial", budget_key="editorial", retrieval_fn=select_structural_by_tags_or_all,
                  append_fn=_append_editorial_entries, title="Редакторские приёмы:",
                  kb_attr="editorial_techniques", uses_structural_call=True, candidate_attr=None),
]

# ---------------------------------------------------------------------------
# Обработчик одного блока KB (с константой для candidate_limit)
# ---------------------------------------------------------------------------
DEFAULT_CANDIDATE_LIMIT = 10

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
            char_budget=budget.char_budget,
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
            char_budget=budget.char_budget,
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
    digest = hashlib.md5(text[:256].encode()).hexdigest()
    return int(digest[:8], 16)

# ---------------------------------------------------------------------------
# Вспомогательные функции для explainability
# ---------------------------------------------------------------------------
def _add_activation_reason(
    result: FeatureResolutionResult,
    feature: str,
    reason: str,
) -> None:
    if feature not in result.activation_reasons:
        result.activation_reasons[feature] = []
    result.activation_reasons[feature].append(reason)
    if feature not in result.activated_features:
        result.activated_features.append(feature)

def _add_suppression_reason(
    result: FeatureResolutionResult,
    feature: str,
    reason: str,
) -> None:
    if feature not in result.suppression_reasons:
        result.suppression_reasons[feature] = []
    result.suppression_reasons[feature].append(reason)
    if feature not in result.suppressed_features:
        result.suppressed_features.append(feature)

def _add_recognized_alias(
    result: FeatureResolutionResult,
    feature: str,
    alias: str,
) -> None:
    if feature not in result.recognized_aliases:
        result.recognized_aliases[feature] = []
    if alias not in result.recognized_aliases[feature]:
        result.recognized_aliases[feature].append(alias)

def _add_ignored_unknown(
    result: FeatureResolutionResult,
    value: str,
) -> None:
    if value not in result.ignored_unknown_values:
        result.ignored_unknown_values.append(value)

# ============================================================================
# Новая функция для построения карты оверлеев по слагу
# ============================================================================
def _build_overlay_slug_map(
    overlays: Sequence[str],
    overlay_configs: Sequence[OverlayConfig],
) -> Dict[str, OverlayConfig]:
    """Сопоставляет слаг оверлея (как в config/overlays/<slug>.json) с его конфигом.
    Нельзя использовать cfg.name как ключ — это человекочитаемое название из JSON,
    а не слаг файла."""
    mapping: Dict[str, OverlayConfig] = {}
    for slug, cfg in zip(overlays, overlay_configs):
        mapping[slug] = cfg
    # Резерв для тестов, где cfg.name может совпадать со слагом
    for cfg in overlay_configs:
        mapping.setdefault(cfg.name, cfg)
    return mapping

# ---------------------------------------------------------------------------
# НОВАЯ ЦЕНТРАЛЬНАЯ ФУНКЦИЯ РАЗРЕШЕНИЯ ФИЧ (с explainability)
# ---------------------------------------------------------------------------
def resolve_prompt_features(
    domain: str,
    intent: Optional[str],
    overlays: Sequence[str],
    domain_config: DomainConfig,
    intent_config: Optional[IntentConfig],
    overlay_configs: List[OverlayConfig],
    knowledge_level: Optional[KnowledgeLevel] = None,  # НОВЫЙ ПАРАМЕТР
) -> Dict[str, Any]:
    """
    Единый канонический источник feature flags с explainability.
    Возвращает dict с полями, включая диагностические.
    """
    # 1. Нормализация
    norm_intent = normalize_intent(intent)
    norm_overlays = normalize_overlays(overlays)

    effective_intent = norm_intent
    effective_overlays = list(norm_overlays)
    suppressed_layers = []
    warnings = []

    # 2. Создаём результат с explainability
    result = FeatureResolutionResult(
        tags=[],
        effective_intent=effective_intent,
        effective_overlays=effective_overlays,
        suppressed_layers=suppressed_layers,
        warnings=warnings,
        storytelling_enabled=False,
        marketing_enabled=False,
        antiai_enabled=False,
        rhetoric_enabled=False,
        nkrj_enabled=False,
        editorial_enabled=False,
        activated_features=[],
        suppressed_features=[],
        activation_reasons={},
        suppression_reasons={},
        recognized_aliases={},
        ignored_unknown_values=[],
    )

    # Фиксируем unknown intent
    if intent and not norm_intent:
        _add_ignored_unknown(result, intent)

    # Фиксируем unknown overlays
    for ov in overlays:
        if ov.lower().strip() not in ALLOWED_OVERLAYS:
            _add_ignored_unknown(result, ov)

    # 3. Базовые теги
    tags = [domain]
    if effective_intent:
        tags.append(effective_intent)
    tags.extend(effective_overlays)

    # 4. Проверка несовместимости интента с доменом
    if effective_intent and effective_intent in domain_config.incompatible_intents:
        suppressed_layers.append(f"intent '{effective_intent}' suppressed by domain '{domain}'")
        warnings.append(f"Intent '{effective_intent}' incompatible with domain '{domain}', ignoring.")
        _add_suppression_reason(result, "intent", ReasonCode.SUPPRESSED_BY_DOMAIN_INCOMPATIBLE_INTENT)
        suppressed_intent = effective_intent
        effective_intent = None
        tags = [t for t in tags if t != suppressed_intent]

    # 5. Проверка несовместимости оверлеев с доменом
    for overlay in list(effective_overlays):
        if overlay in domain_config.incompatible_overlays:
            effective_overlays.remove(overlay)
            suppressed_layers.append(f"overlay '{overlay}' suppressed by domain '{domain}'")
            warnings.append(f"Overlay '{overlay}' incompatible with domain '{domain}', removed.")
            _add_suppression_reason(result, f"overlay:{overlay}", ReasonCode.SUPPRESSED_BY_DOMAIN_INCOMPATIBLE_OVERLAY)
            tags = [t for t in tags if t != overlay]

    # ======================================================================
    # 5.5. Явные suppresses между оверлеями (независимо от conflicts_with)
    # ======================================================================
    overlay_map = _build_overlay_slug_map(effective_overlays, overlay_configs)
    suppressed_by_overlay: Set[str] = set()

    for ov in list(effective_overlays):
        cfg = overlay_map.get(ov)
        if not cfg or not cfg.suppresses:
            continue
        for target in cfg.suppresses:
            if target in effective_overlays and target != ov:
                suppressed_by_overlay.add(target)
                suppressed_layers.append(
                    f"overlay '{target}' suppressed by overlay '{ov}' (explicit suppress)"
                )
                warnings.append(
                    f"Overlay '{target}' explicitly suppressed by '{ov}'."
                )
                _add_suppression_reason(
                    result,
                    f"overlay:{target}",
                    ReasonCode.SUPPRESSED_BY_OVERLAY_RULE,
                )
                tags = [t for t in tags if t != target]

    if suppressed_by_overlay:
        effective_overlays = [
            ov for ov in effective_overlays if ov not in suppressed_by_overlay
        ]
    # ======================================================================

    # ======================================================================
    # 6. Конфликты между оверлеями (с учётом priority и явных suppresses)
    # ======================================================================
    overlay_map = _build_overlay_slug_map(effective_overlays, overlay_configs)
    # Сначала собираем все конфликты, чтобы не удалять элементы во время итерации
    conflicts_to_resolve = []
    for ov in effective_overlays:
        cfg = overlay_map.get(ov)
        if cfg and cfg.conflicts_with:
            for conflict in cfg.conflicts_with:
                if conflict in effective_overlays and conflict != ov:
                    conflicts_to_resolve.append((ov, conflict))

    # Разрешаем каждый конфликт, выбирая победителя по priority и явным suppresses
    for ov, conflict in conflicts_to_resolve:
        # Проверяем, не был ли уже удалён один из них
        if ov not in effective_overlays or conflict not in effective_overlays:
            continue

        cfg_ov = overlay_map.get(ov)
        cfg_conflict = overlay_map.get(conflict)

        if cfg_ov is None or cfg_conflict is None:
            continue

        # Проверяем явное suppresses: если один подавляет другой, то он побеждает
        if conflict in cfg_ov.suppresses:
            # ov подавляет conflict -> удаляем conflict
            effective_overlays.remove(conflict)
            suppressed_layers.append(f"overlay '{conflict}' suppressed by overlay '{ov}' (explicit suppress)")
            warnings.append(f"Overlay '{conflict}' explicitly suppressed by '{ov}'.")
            _add_suppression_reason(result, f"overlay:{conflict}", ReasonCode.SUPPRESSED_BY_OVERLAY_RULE)
            tags = [t for t in tags if t != conflict]
            continue
        if ov in cfg_conflict.suppresses:
            # conflict подавляет ov -> удаляем ov
            effective_overlays.remove(ov)
            suppressed_layers.append(f"overlay '{ov}' suppressed by overlay '{conflict}' (explicit suppress)")
            warnings.append(f"Overlay '{ov}' explicitly suppressed by '{conflict}'.")
            _add_suppression_reason(result, f"overlay:{ov}", ReasonCode.SUPPRESSED_BY_OVERLAY_RULE)
            tags = [t for t in tags if t != ov]
            continue

        # Сравниваем priority
        if cfg_ov.priority > cfg_conflict.priority:
            # ov побеждает -> удаляем conflict
            effective_overlays.remove(conflict)
            suppressed_layers.append(f"overlay '{conflict}' suppressed due to conflict with '{ov}' (higher priority)")
            warnings.append(f"Overlay conflict: '{conflict}' removed (priority {cfg_conflict.priority}) < '{ov}' (priority {cfg_ov.priority}).")
            _add_suppression_reason(result, f"overlay:{conflict}", ReasonCode.SUPPRESSED_BY_OVERLAY_CONFLICT)
            tags = [t for t in tags if t != conflict]
        elif cfg_conflict.priority > cfg_ov.priority:
            # conflict побеждает -> удаляем ov
            effective_overlays.remove(ov)
            suppressed_layers.append(f"overlay '{ov}' suppressed due to conflict with '{conflict}' (higher priority)")
            warnings.append(f"Overlay conflict: '{ov}' removed (priority {cfg_ov.priority}) < '{conflict}' (priority {cfg_conflict.priority}).")
            _add_suppression_reason(result, f"overlay:{ov}", ReasonCode.SUPPRESSED_BY_OVERLAY_CONFLICT)
            tags = [t for t in tags if t != ov]
        else:
            # Приоритеты равны и нет явного подавления — ошибка конфигурации
            raise ValueError(
                f"Overlay conflict between '{ov}' and '{conflict}' with equal priority ({cfg_ov.priority}) "
                "and no explicit suppress rule. Please define a winner or adjust priorities."
            )
    # ======================================================================

    # 7. Получаем фичи из тегов (используя канонические алиасы)
    all_tags = [domain]
    if effective_intent:
        all_tags.append(effective_intent)
    all_tags.extend(effective_overlays)
    features = get_features_from_tags(all_tags)

    # 8. Базовые флаги с explainability
    # Storytelling
    if domain_config.allow_storytelling and "storytelling" in features:
        result.storytelling_enabled = True
        _add_activation_reason(result, "storytelling", ReasonCode.DOMAIN_ALLOWS_STORYTELLING)
        _add_activation_reason(result, "storytelling", ReasonCode.RECOGNIZED_STORYTELLING_ALIAS)
        for tag in all_tags:
            if tag in _TAG_TO_FEATURE and _TAG_TO_FEATURE[tag] == "storytelling":
                _add_recognized_alias(result, "storytelling", tag)
    elif "storytelling" in features:
        _add_suppression_reason(result, "storytelling", ReasonCode.DOMAIN_DENIES_STORYTELLING)
    else:
        _add_suppression_reason(result, "storytelling", ReasonCode.NO_RECOGNIZED_ALIAS)

    # Marketing
    if domain_config.allow_marketing and "marketing" in features:
        result.marketing_enabled = True
        _add_activation_reason(result, "marketing", ReasonCode.DOMAIN_ALLOWS_MARKETING)
        _add_activation_reason(result, "marketing", ReasonCode.RECOGNIZED_MARKETING_ALIAS)
        for tag in all_tags:
            if tag in _TAG_TO_FEATURE and _TAG_TO_FEATURE[tag] == "marketing":
                _add_recognized_alias(result, "marketing", tag)
    elif "marketing" in features:
        _add_suppression_reason(result, "marketing", ReasonCode.DOMAIN_DENIES_MARKETING)
    else:
        _add_suppression_reason(result, "marketing", ReasonCode.NO_RECOGNIZED_ALIAS)

    # anti-ai
    if "antiai" in features:
        result.antiai_enabled = True
        _add_activation_reason(result, "antiai", ReasonCode.RECOGNIZED_ANTIAI_ALIAS)
        for tag in all_tags:
            if tag in _TAG_TO_FEATURE and _TAG_TO_FEATURE[tag] == "antiai":
                _add_recognized_alias(result, "antiai", tag)
    else:
        _add_suppression_reason(result, "antiai", ReasonCode.NO_RECOGNIZED_ALIAS)

    # rhetoric
    if "rhetoric" in features:
        result.rhetoric_enabled = True
        _add_activation_reason(result, "rhetoric", ReasonCode.RECOGNIZED_RHETORIC_ALIAS)
        for tag in all_tags:
            if tag in _TAG_TO_FEATURE and _TAG_TO_FEATURE[tag] == "rhetoric":
                _add_recognized_alias(result, "rhetoric", tag)
    else:
        _add_suppression_reason(result, "rhetoric", ReasonCode.NO_RECOGNIZED_ALIAS)

    # nkrj
    if "nkrj" in features:
        result.nkrj_enabled = True
        _add_activation_reason(result, "nkrj", ReasonCode.RECOGNIZED_NKRJ_ALIAS)
        for tag in all_tags:
            if tag in _TAG_TO_FEATURE and _TAG_TO_FEATURE[tag] == "nkrj":
                _add_recognized_alias(result, "nkrj", tag)
    else:
        _add_suppression_reason(result, "nkrj", ReasonCode.NO_RECOGNIZED_ALIAS)

    # editorial
    if "editorial" in features:
        result.editorial_enabled = True
        _add_activation_reason(result, "editorial", ReasonCode.RECOGNIZED_EDITORIAL_ALIAS)
        for tag in all_tags:
            if tag in _TAG_TO_FEATURE and _TAG_TO_FEATURE[tag] == "editorial":
                _add_recognized_alias(result, "editorial", tag)
    else:
        _add_suppression_reason(result, "editorial", ReasonCode.NO_RECOGNIZED_ALIAS)

    # 8.5 Принудительное включение при FULL уровне
    if knowledge_level == KnowledgeLevel.FULL:
        # Storytelling
        if domain_config.allow_storytelling and not result.storytelling_enabled:
            result.storytelling_enabled = True
            _add_activation_reason(result, "storytelling", ReasonCode.FULL_LEVEL_OVERRIDE)
        # Marketing
        if domain_config.allow_marketing and not result.marketing_enabled:
            result.marketing_enabled = True
            _add_activation_reason(result, "marketing", ReasonCode.FULL_LEVEL_OVERRIDE)
        # Editorial, Rhetoric, NKRJ — включаем всегда при FULL (они не имеют флагов разрешения в домене)
        if not result.editorial_enabled:
            result.editorial_enabled = True
            _add_activation_reason(result, "editorial", ReasonCode.FULL_LEVEL_OVERRIDE)
        if not result.rhetoric_enabled:
            result.rhetoric_enabled = True
            _add_activation_reason(result, "rhetoric", ReasonCode.FULL_LEVEL_OVERRIDE)
        if not result.nkrj_enabled:
            result.nkrj_enabled = True
            _add_activation_reason(result, "nkrj", ReasonCode.FULL_LEVEL_OVERRIDE)

    # 9. Применяем suppress правила (первая итерация)
    if "storytelling" in domain_config.suppresses:
        result.storytelling_enabled = False
        suppressed_layers.append("storytelling suppressed by domain")
        warnings.append("Storytelling disabled by domain 'suppresses' rule.")
        _add_suppression_reason(result, "storytelling", ReasonCode.SUPPRESSED_BY_DOMAIN_RULE)
    if "marketing" in domain_config.suppresses or "marketingpush" in domain_config.suppresses:
        result.marketing_enabled = False
        suppressed_layers.append("marketing suppressed by domain")
        warnings.append("Marketing disabled by domain 'suppresses' rule.")
        _add_suppression_reason(result, "marketing", ReasonCode.SUPPRESSED_BY_DOMAIN_RULE)

    for cfg in overlay_configs:
        if cfg.name in effective_overlays:
            if "storytelling" in cfg.suppresses:
                result.storytelling_enabled = False
                suppressed_layers.append(f"storytelling suppressed by overlay '{cfg.name}'")
                warnings.append(f"Storytelling disabled by overlay '{cfg.name}' suppress rule.")
                _add_suppression_reason(result, "storytelling", ReasonCode.SUPPRESSED_BY_OVERLAY_RULE)
            if "marketing" in cfg.suppresses or "marketingpush" in cfg.suppresses:
                result.marketing_enabled = False
                suppressed_layers.append(f"marketing suppressed by overlay '{cfg.name}'")
                warnings.append(f"Marketing disabled by overlay '{cfg.name}' suppress rule.")
                _add_suppression_reason(result, "marketing", ReasonCode.SUPPRESSED_BY_OVERLAY_RULE)

    # 9.5 Применяем suppress правила интента (НОВОЕ для Итерации 4)
    if intent_config:
        for suppressed in intent_config.suppresses:
            if suppressed == "storytelling":
                result.storytelling_enabled = False
                if "storytelling" not in result.suppressed_features:
                    result.suppressed_features.append("storytelling")
                _add_suppression_reason(result, "storytelling", ReasonCode.SUPPRESSED_BY_INTENT_RULE)
            elif suppressed in ("marketing", "marketingpush"):
                result.marketing_enabled = False
                if "marketing" not in result.suppressed_features:
                    result.suppressed_features.append("marketing")
                _add_suppression_reason(result, "marketing", ReasonCode.SUPPRESSED_BY_INTENT_RULE)
            elif suppressed == "antiai":
                result.antiai_enabled = False
                if "antiai" not in result.suppressed_features:
                    result.suppressed_features.append("antiai")
                _add_suppression_reason(result, "antiai", ReasonCode.SUPPRESSED_BY_INTENT_RULE)
            elif suppressed == "rhetoric":
                result.rhetoric_enabled = False
                if "rhetoric" not in result.suppressed_features:
                    result.suppressed_features.append("rhetoric")
                _add_suppression_reason(result, "rhetoric", ReasonCode.SUPPRESSED_BY_INTENT_RULE)
            elif suppressed == "nkrj":
                result.nkrj_enabled = False
                if "nkrj" not in result.suppressed_features:
                    result.suppressed_features.append("nkrj")
                _add_suppression_reason(result, "nkrj", ReasonCode.SUPPRESSED_BY_INTENT_RULE)
            elif suppressed == "editorial":
                result.editorial_enabled = False
                if "editorial" not in result.suppressed_features:
                    result.suppressed_features.append("editorial")
                _add_suppression_reason(result, "editorial", ReasonCode.SUPPRESSED_BY_INTENT_RULE)
            # Если suppressed начинается с "overlay:" или "intent:", мы игнорируем (пока)
            # Другие значения не обрабатываем

    # 10. Итоговые теги: уникальные
    final_tags = list(dict.fromkeys(tags))
    result.tags = final_tags
    result.effective_intent = effective_intent
    result.effective_overlays = effective_overlays
    result.suppressed_layers = suppressed_layers
    result.warnings = warnings

    # 11. Возвращаем dict для обратной совместимости
    return result.to_dict()

# ---------------------------------------------------------------------------
# PromptBuilder (основной класс)
# ---------------------------------------------------------------------------
class PromptBuilder:
    def __init__(self, config_path: Path = Path("config"), kb_path: Path = Path("knowledge_base"),
                 limits: Optional[LimitsConfig] = None) -> None:
        self.config_path = config_path
        self.kb_path = kb_path
        self._limits = limits or LimitsConfig()
        self.core_config: Optional[CoreConfig] = None

        # NEW: кэши для конфигов и output format
        self._core_cache: Optional[CoreConfig] = None
        self._domain_cache: Dict[str, DomainConfig] = {}
        self._intent_cache: Dict[str, Optional[IntentConfig]] = {}
        self._overlay_cache: Dict[str, OverlayConfig] = {}
        self._output_format_cache: Dict[str, str] = {}
        self._kb_cache = FileCache(policy=CachePolicy(check_mtime=True))

        self._load_core_config()

    # ------------------------------------------------------------------
    # Internal get-helper'ы (единая точка доступа)
    # ------------------------------------------------------------------
    def _load_core_config(self) -> CoreConfig:
        if self._core_cache is None:
            self._core_cache = load_core_config(self.config_path)
        return self._core_cache

    def get_core_config(self) -> CoreConfig:
        return self._load_core_config()

    def get_domain_config(self, domain: str) -> DomainConfig:
        if domain not in self._domain_cache:
            self._domain_cache[domain] = load_domain_config(domain, self.config_path)
        return self._domain_cache[domain]

    def get_intent_config(self, intent: Optional[str]) -> Optional[IntentConfig]:
        if intent is None or intent == "neutral":
            return None
        if intent not in self._intent_cache:
            self._intent_cache[intent] = load_intent_config(intent, self.config_path)
        return self._intent_cache[intent]

    def get_overlay_config(self, overlay: str) -> OverlayConfig:
        if overlay not in self._overlay_cache:
            self._overlay_cache[overlay] = load_overlay_config(overlay, self.config_path)
        return self._overlay_cache[overlay]

    def get_overlay_configs(self, overlays: Sequence[str]) -> List[OverlayConfig]:
        return [self.get_overlay_config(ov) for ov in overlays]

    def get_output_format(self, mode: str) -> str:
        if mode not in self._output_format_cache:
            self._output_format_cache[mode] = load_output_format(mode, self.config_path)
        return self._output_format_cache[mode]

    def get_knowledge_base(self, primary_tags: Set[str], intent: Optional[str]) -> KnowledgeBase:
        cache_key = f"kb:{','.join(sorted(primary_tags))}:{intent or 'none'}"
        manifest_path = self.kb_path / "kb_manifest.json"
        kb_files = [manifest_path]
        if self.kb_path.exists():
            kb_files.extend(sorted(self.kb_path.rglob("*.json")))
        return self._kb_cache.get_or_load_multi(
            cache_key, kb_files, load_knowledge_base,
            self.kb_path, primary_tags, intent,
        )

    # ------------------------------------------------------------------
    # Cache invalidation
    # ------------------------------------------------------------------
    def _invalidate_caches(self) -> None:
        self._core_cache = None
        self._domain_cache.clear()
        self._intent_cache.clear()
        self._overlay_cache.clear()
        self._output_format_cache.clear()
        self._kb_cache.clear()

    def reload_configs(self) -> None:
        self._invalidate_caches()
        self.core_config = self._load_core_config()
        logger.info("PromptBuilder caches invalidated and reloaded.")

    # ------------------------------------------------------------------
    # Existing public methods (startup_check, get_available_*, _validate_*, etc.)
    # ------------------------------------------------------------------
    def startup_check(self) -> None:
        self.core_config = self.get_core_config()
        self._validate_kb_manifest()
        # Проверка согласованности алиасов (не блокирующая)
        try:
            warnings_list = check_alias_consistency()
            for w in warnings_list:
                logger.warning(w)
        except Exception as e:
            logger.warning("Alias consistency check failed: %s", e)

    def get_available_intents(self) -> Set[str]:
        intents_dir = self.config_path / "intents"
        if not intents_dir.exists():
            return set(ALLOWED_INTENTS)
        values = {path.stem for path in intents_dir.glob("*.json")}
        return values or set(ALLOWED_INTENTS)

    def getavailableintents(self) -> Set[str]:
        warnings.warn("getavailableintents() deprecated, use get_available_intents()", DeprecationWarning, stacklevel=2)
        return self.get_available_intents()

    def get_available_overlays(self) -> Set[str]:
        overlays_dir = self.config_path / "overlays"
        if not overlays_dir.exists():
            return set(ALLOWED_OVERLAYS)
        values = {path.stem for path in overlays_dir.glob("*.json")}
        return values or set(ALLOWED_OVERLAYS)

    def getavailableoverlays(self) -> Set[str]:
        warnings.warn("getavailableoverlays() deprecated, use get_available_overlays()", DeprecationWarning, stacklevel=2)
        return self.get_available_overlays()

    def _validate_domain(self, domain: str) -> str:
        if domain not in ALLOWED_DOMAINS:
            raise ValueError(f"Unknown domain: {domain!r}. Available: {sorted(ALLOWED_DOMAINS)}")
        return domain

    def _validate_intent(self, intent: Optional[str]) -> Optional[str]:
        if intent is None:
            return None
        if intent not in ALLOWED_INTENTS:
            raise ValueError(f"Unknown intent: {intent!r}. Available: {sorted(ALLOWED_INTENTS)}")
        return intent

    def _validate_overlays(self, overlays: Sequence[str]) -> List[str]:
        normalized = [o.lower() for o in overlays]
        for o in normalized:
            if o not in ALLOWED_OVERLAYS:
                raise ValueError(f"Unknown overlay: {o!r}. Available: {sorted(ALLOWED_OVERLAYS)}")
        overlay_configs = self.get_overlay_configs(normalized)
        for ov_cfg in overlay_configs:
            for conflict in ov_cfg.conflicts_with:
                if conflict.lower() in normalized:
                    raise ValueError(f"Overlays conflict: '{ov_cfg.name}' and '{conflict}' cannot be used together.")
        return normalized

    def _validate_output_mode(self, output_mode: str) -> str:
        normalized = output_mode.strip().lower()
        if normalized not in ALLOWED_OUTPUT_MODES:
            raise ValueError(f"Unsupported output_mode: {output_mode!r}. Must be one of {sorted(ALLOWED_OUTPUT_MODES)}")
        return normalized

    def _build_audience_block(self, audience: Optional[AudienceProfile]) -> str:
        if audience is None:
            return ""
        parts = [f"Тип аудитории: {audience.kind}", f"Уровень экспертизы: {audience.expertise}",
                 f"Формальность: {audience.formality}"]
        if getattr(audience, "description", ""):
            parts.append(f"Описание аудитории: {audience.description}")
        return "\n".join(parts)

    def _build_mode_constraints_block(self, domain_config: DomainConfig) -> str:
        lines = []
        if not domain_config.allow_storytelling:
            lines.append("Сторителлинг запрещён: не добавляй нарративные отступления, личные истории и метафоры.")
        if not domain_config.allow_marketing:
            lines.append("Маркетинг запрещён: удаляй призывы к действию, триггерные слова и конструкции давления.")
        return "\n".join(lines) if lines else ""

    def _build_ip_ceiling_block(self, domain_config: DomainConfig) -> str:
        effective_ceiling = domain_config.ip_ceiling if domain_config.ip_ceiling is not None else (
            self.core_config.ip_ceiling if self.core_config else 2.5)
        return (f"Целевой Индекс пластиковости (ИП): ≤ {effective_ceiling}. "
                "После редактирования укажи итоговый ИП. "
                "Если ИП превышает целевое значение — предупреди и предложи второй проход.")

    def _merge_domain_limits(self, domain_config: DomainConfig) -> LimitsConfig:
        overrides = domain_config.kb_limits or {}
        base = self._limits
        return LimitsConfig(
            grammar=overrides.get("grammar", base.grammar),
            style=overrides.get("style", base.style),
            logic=overrides.get("logic", base.logic),
            composition=overrides.get("composition", base.composition),
            cohesion=overrides.get("cohesion", overrides.get("local_cohesion", base.cohesion)),
            composition_errors=overrides.get("composition_errors", base.composition_errors),
            storytelling=overrides.get("storytelling", base.storytelling),
            marketing=overrides.get("marketing", base.marketing),
            rhetoric=overrides.get("rhetoric", base.rhetoric),
            editorial=overrides.get("editorial", base.editorial),
            glossary=overrides.get("glossary", base.glossary),
            stop_words_category=overrides.get("stop_words", base.stop_words_category),
            stop_words_items=overrides.get("stop_words_items", base.stop_words_items),
            nkrj=overrides.get("nkrj", base.nkrj),
            grammar_candidates=overrides.get("grammar_candidates", base.grammar_candidates),
            style_candidates=overrides.get("style_candidates", base.style_candidates),
            logic_candidates=overrides.get("logic_candidates", base.logic_candidates),
            storytelling_candidates=overrides.get("storytelling_candidates", base.storytelling_candidates),
            marketing_candidates=overrides.get("marketing_candidates", base.marketing_candidates),
            rhetoric_candidates=overrides.get("rhetoric_candidates", base.rhetoric_candidates),
        )

    # ------------------------------------------------------------------
    # Основной метод сборки knowledge block (с диагностикой)
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
        limits: Optional[LimitsConfig] = None,
        storytelling_enabled: bool = True,
        marketing_enabled: bool = True,
        antiai_enabled: bool = False,
        rhetoric_enabled: bool = False,
        nkrj_enabled: bool = False,
        editorial_enabled: bool = False,
        return_trace: bool = False,
    ) -> Union[Tuple[str, Dict[str, Any], int], Tuple[str, Dict[str, Any], int, AssemblyTrace]]:
        """
        Строит блок базы знаний. Если return_trace=True, возвращает также AssemblyTrace.
        """
        effective_limits = limits if limits is not None else self._limits
        # Используем кэш KB
        kb = self.get_knowledge_base(primary_tags, intent)
        lines: List[str] = []
        meta: Dict[str, Any] = {}
        current_total = total_few_shot_used
        trace = AssemblyTrace() if return_trace else None

        # Стоп-слова
        stop_words_budget = budget.get("stop_words")
        if stop_words_budget and stop_words_budget.enabled:
            stop_words = kb.get("stop_words", {})
            if stop_words:
                lines.append("Стоп-слова и нежелательные формулировки:")
                category_limit = stop_words_budget.entry_limit or effective_limits.stop_words_category
                for category, words in list(stop_words.items())[:category_limit]:
                    if isinstance(words, list) and words:
                        joined = ", ".join(str(w) for w in words[:effective_limits.stop_words_items])
                        lines.append(f"- {category}: {joined}")
                if trace:
                    trace.add_block(AssemblyBlockDiagnostics(
                        name="stop_words",
                        eligible=True,
                        included=True,
                        reason_codes=[ReasonCode.BLOCK_INCLUDED],
                        empty=False,
                        char_count=sum(len(l) for l in lines[-len(stop_words)-1:]),
                        entries_count=len(stop_words),
                    ))
            else:
                if trace:
                    trace.add_block(AssemblyBlockDiagnostics(
                        name="stop_words",
                        eligible=True,
                        included=False,
                        reason_codes=[ReasonCode.BLOCK_EMPTY_AFTER_BUILD],
                        empty=True,
                    ))

        # Основные блоки через реестр
        for block_cfg in KB_BLOCK_REGISTRY:
            block_budget = budget.get(block_cfg.budget_key)
            if not (block_budget and block_budget.enabled):
                if trace:
                    trace.add_block(AssemblyBlockDiagnostics(
                        name=block_cfg.name,
                        eligible=False,
                        included=False,
                        reason_codes=[ReasonCode.BLOCK_INELIGIBLE_BUDGET_DISABLED],
                    ))
                continue

            # Проверка feature-флагов
            feature_gated = False
            if block_cfg.name == "storytelling" and not storytelling_enabled:
                feature_gated = True
                reason = ReasonCode.BLOCK_INELIGIBLE_FEATURE_DISABLED
            elif block_cfg.name == "marketing" and not marketing_enabled:
                feature_gated = True
                reason = ReasonCode.BLOCK_INELIGIBLE_FEATURE_DISABLED
            elif block_cfg.name == "rhetoric" and not rhetoric_enabled:
                feature_gated = True
                reason = ReasonCode.BLOCK_INELIGIBLE_FEATURE_DISABLED
            elif block_cfg.name == "editorial" and not editorial_enabled:
                feature_gated = True
                reason = ReasonCode.BLOCK_INELIGIBLE_FEATURE_DISABLED

            if feature_gated:
                if trace:
                    trace.add_block(AssemblyBlockDiagnostics(
                        name=block_cfg.name,
                        eligible=False,
                        included=False,
                        reason_codes=[reason],
                    ))
                continue

            # Проверка наличия KB-данных
            if block_cfg.uses_structural_call and block_cfg.kb_attr:
                if not kb.get(block_cfg.kb_attr):
                    if trace:
                        trace.add_block(AssemblyBlockDiagnostics(
                            name=block_cfg.name,
                            eligible=False,
                            included=False,
                            reason_codes=[ReasonCode.BLOCK_INELIGIBLE_KB_UNAVAILABLE],
                        ))
                    continue

            # Блок eligible
            eligible = True
            if trace:
                trace.add_block(AssemblyBlockDiagnostics(
                    name=block_cfg.name,
                    eligible=True,
                    included=False,
                    reason_codes=[ReasonCode.BLOCK_ELIGIBLE],
                ))

            # Собираем блок
            before_len = len("".join(lines))
            before_entries = current_total
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
                limits=effective_limits,
                few_shot_seed=few_shot_seed,
            )
            after_len = len("".join(lines))
            included = (after_len > before_len) and (block_cfg.title in "\n".join(lines))
            empty = included and (after_len == before_len)
            char_count = after_len - before_len
            entries_added = current_total - before_entries

            if trace:
                if trace.blocks and trace.blocks[-1].name == block_cfg.name:
                    trace.blocks[-1].included = included
                    trace.blocks[-1].empty = empty
                    trace.blocks[-1].char_count = char_count
                    trace.blocks[-1].entries_count = entries_added
                    if not included and not empty:
                        trace.blocks[-1].reason_codes.append(ReasonCode.BLOCK_SKIPPED)
                    elif included:
                        trace.blocks[-1].reason_codes.append(ReasonCode.BLOCK_INCLUDED)
                    if empty:
                        trace.blocks[-1].reason_codes.append(ReasonCode.BLOCK_EMPTY_AFTER_BUILD)
                else:
                    trace.add_block(AssemblyBlockDiagnostics(
                        name=block_cfg.name,
                        eligible=eligible,
                        included=included,
                        reason_codes=[ReasonCode.BLOCK_INCLUDED if included else ReasonCode.BLOCK_SKIPPED],
                        empty=empty,
                        char_count=char_count,
                        entries_count=entries_added,
                    ))

        # Глоссарий
        glossary_budget = budget.get("glossary")
        if glossary_budget and glossary_budget.enabled:
            glossary = kb.get("domain_glossary", {})
            if glossary:
                before_len = len("".join(lines))
                _append_glossary(lines, glossary, glossary_budget.entry_limit)
                after_len = len("".join(lines))
                if trace:
                    trace.add_block(AssemblyBlockDiagnostics(
                        name="glossary",
                        eligible=True,
                        included=True,
                        reason_codes=[ReasonCode.BLOCK_INCLUDED],
                        empty=False,
                        char_count=after_len - before_len,
                        entries_count=len(glossary),
                    ))
            else:
                if trace:
                    trace.add_block(AssemblyBlockDiagnostics(
                        name="glossary",
                        eligible=True,
                        included=False,
                        reason_codes=[ReasonCode.BLOCK_EMPTY_AFTER_BUILD],
                        empty=True,
                    ))

        # NKRJ
        nkrj_budget = budget.get("nkrj")
        if nkrj_budget and nkrj_budget.enabled and nkrj_enabled:
            nkrj = kb.get("nkrj_structure_patterns", {})
            if nkrj:
                before_len = len("".join(lines))
                _append_nkrj(lines, nkrj)
                after_len = len("".join(lines))
                if trace:
                    trace.add_block(AssemblyBlockDiagnostics(
                        name="nkrj",
                        eligible=True,
                        included=True,
                        reason_codes=[ReasonCode.BLOCK_INCLUDED],
                        empty=False,
                        char_count=after_len - before_len,
                        entries_count=len(nkrj),
                    ))
            else:
                if trace:
                    trace.add_block(AssemblyBlockDiagnostics(
                        name="nkrj",
                        eligible=True,
                        included=False,
                        reason_codes=[ReasonCode.BLOCK_EMPTY_AFTER_BUILD],
                        empty=True,
                    ))
        else:
            if trace:
                trace.add_block(AssemblyBlockDiagnostics(
                    name="nkrj",
                    eligible=False,
                    included=False,
                    reason_codes=[ReasonCode.BLOCK_INELIGIBLE_FEATURE_DISABLED],
                ))

        if return_trace:
            return "\n".join(lines), meta, current_total, trace
        return "\n".join(lines), meta, current_total

    def _assemble_prompt(self, blocks: List[str]) -> str:
        return "\n\n".join(block for block in blocks if block.strip())

    def _validate_kb_manifest(self) -> None:
        from src.kb_manifest_loader import load_manifest
        manifest = load_manifest(self.kb_path / "kb_manifest.json")
        block_types: Dict[str, str] = {}
        for entry in manifest:
            key = entry.block_name or entry.file.split("/")[0] if "/" in entry.file else Path(entry.file).stem
            btype = getattr(entry, "block_type", "list")
            existing = block_types.get(key)
            if existing is None:
                block_types[key] = btype
            elif existing != btype:
                raise ValueError(f"KB manifest inconsistent: block '{key}' has mixed block_type ('{existing}' vs '{btype}').")

    # ------------------------------------------------------------------
    # Основной метод build (обновлён с использованием get-helper'ов)
    # ------------------------------------------------------------------
    @overload
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
        include_retrieval_meta: Literal[False] = False,
        few_shot_seed: Optional[int] = None,
        **legacy_kwargs: Any,
    ) -> str: ...

    @overload
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
        include_retrieval_meta: Literal[True] = True,
        few_shot_seed: Optional[int] = None,
        **legacy_kwargs: Any,
    ) -> Tuple[str, Dict[str, Any]]: ...

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
        # Legacy support
        legacy_output_mode = legacy_kwargs.pop("outputmode", None)
        legacy_include_knowledge = legacy_kwargs.pop("includeknowledge", None)
        if legacy_kwargs:
            raise TypeError(f"Unexpected keyword arguments: {', '.join(sorted(legacy_kwargs))}")
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
            self.core_config = self.get_core_config()

        domain_config = self.get_domain_config(validated_domain)
        intent_config = self.get_intent_config(validated_intent)
        overlay_configs = self.get_overlay_configs(validated_overlays)
        output_format = self.get_output_format(validated_output_mode)

        # Разрешение фич через каноническую функцию (теперь с knowledge_level)
        features = resolve_prompt_features(
            domain=validated_domain,
            intent=validated_intent,
            overlays=validated_overlays,
            domain_config=domain_config,
            intent_config=intent_config,
            overlay_configs=overlay_configs,
            knowledge_level=knowledge_level,  # передаём уровень знаний
        )
        effective_overlays = features["effective_overlays"]
        storytelling_enabled = features["storytelling_enabled"]
        marketing_enabled = features["marketing_enabled"]
        antiai_enabled = features["antiai_enabled"]
        rhetoric_enabled = features["rhetoric_enabled"]
        nkrj_enabled = features["nkrj_enabled"]
        editorial_enabled = features["editorial_enabled"]
        warnings_list = features["warnings"]
        for warn in warnings_list:
            logger.warning("PromptBuilder feature resolution: %s", warn)

        # Теги для KB
        tag_sets = _collect_retrieval_tags(validated_domain, validated_intent, effective_overlays)

        # Явно добавляем теги для включённых фич, чтобы загрузить соответствующие KB-блоки
        # Используем готовые флаги из resolve_prompt_features
        if editorial_enabled:
            tag_sets["primary"].add("editorial")
        if storytelling_enabled:
            tag_sets["primary"].add("storytelling")
        if marketing_enabled:
            tag_sets["primary"].add("marketing")
        if rhetoric_enabled:
            tag_sets["primary"].add("rhetoric")
        if nkrj_enabled:
            tag_sets["primary"].add("nkrj")
        if antiai_enabled:
            tag_sets["primary"].add("antiai")

        blocks: List[str] = []

        # Базовые блоки
        blocks.append(f"Роль: {self.core_config.role}")
        blocks.append(f"Приоритеты: {self.core_config.priorities}")
        blocks.append(f"Домен: {domain_config.name}")
        blocks.append(f"Тон: {domain_config.tone}")

        if domain_config.system_rules:
            blocks.append("Правила домена:\n" + domain_config.system_rules)

        mode_constraints = self._build_mode_constraints_block(domain_config)
        if mode_constraints:
            blocks.append(mode_constraints)

        if domain_config.tasks:
            blocks.append("Задачи редактора в этом домене:\n- " + "\n- ".join(domain_config.tasks))
        if domain_config.constraints:
            blocks.append("Ограничения домена:\n- " + "\n- ".join(domain_config.constraints))

        if self.core_config.basic_audit_instructions:
            blocks.append("Базовые инструкции:\n- " + "\n- ".join(self.core_config.basic_audit_instructions))
        if self.core_config.forbidden:
            blocks.append("Запрещено:\n- " + "\n- ".join(self.core_config.forbidden))

        if intent_config and intent_config.instructions:
            blocks.append(f"Intent: {intent_config.name}\n- " + "\n- ".join(intent_config.instructions))

        effective_overlay_configs = [cfg for cfg in overlay_configs if cfg.name in effective_overlays]
        if effective_overlay_configs:
            overlay_lines = []
            for overlay in effective_overlay_configs:
                if overlay.instructions:
                    overlay_lines.append(f"[{overlay.name}] " + " | ".join(overlay.instructions))
            if overlay_lines:
                blocks.append("Overlay-инструкции:\n- " + "\n- ".join(overlay_lines))

        audience_block = self._build_audience_block(audience)
        if audience_block:
            blocks.append("Аудитория:\n" + audience_block)

        retrieval_meta_total: Dict[str, Any] = {}
        if include_knowledge:
            effective_limits = self._merge_domain_limits(domain_config)
            budget = KnowledgeBudgetManager(token_budget).allocate(
                limits=effective_limits,
                level=knowledge_level,
            )
            # Отключаем блоки, для которых фичи выключены (уже с учётом FULL)
            if not storytelling_enabled:
                budget.disable("storytelling")
            if not marketing_enabled:
                budget.disable("marketing")
            # Editorial, rhetoric, nkrj не отключаем, даже если их фичи выключены,
            # потому что они управляются через knowledge_level, а не через отдельные флаги.
            # Однако если они выключены, то они не будут включены в промпт,
            # потому что в _build_knowledge_block мы передаём соответствующие флаги.
            # Здесь мы просто не отключаем их в бюджете.

            effective_seed = few_shot_seed if few_shot_seed is not None else _derive_seed(text)

            knowledge_block, block_meta, _, trace = self._build_knowledge_block(
                text=text,
                primary_tags=tag_sets["primary"],
                expanded_tags=tag_sets["expanded"],
                budget=budget,
                domain=validated_domain,
                intent=validated_intent,
                overlays=effective_overlays,
                include_few_shot=include_few_shot,
                total_few_shot_used=0,
                few_shot_seed=effective_seed,
                limits=effective_limits,
                storytelling_enabled=storytelling_enabled,
                marketing_enabled=marketing_enabled,
                antiai_enabled=antiai_enabled,
                rhetoric_enabled=rhetoric_enabled,
                nkrj_enabled=nkrj_enabled,
                editorial_enabled=editorial_enabled,
                return_trace=True,
            )
            retrieval_meta_total = block_meta
            if knowledge_block:
                blocks.append("База знаний:\n" + knowledge_block)

            self._last_trace = trace  # для тестов и диагностики

        blocks.append(self._build_ip_ceiling_block(domain_config))
        blocks.append("Формат ответа:\n" + output_format)
        blocks.append("Исходный текст:\n" + text.strip())

        prompt = self._assemble_prompt(blocks)
        if include_retrieval_meta:
            return prompt, retrieval_meta_total
        return prompt

    def build_prompt(self, **kwargs: Any) -> str:
        warnings.warn("build_prompt() is deprecated, use build()", DeprecationWarning, stacklevel=2)
        return self.build(**kwargs)