"""
prompt_builder.py

Модуль для сборки финальных промптов из конфигов и базы знаний.
Фаза 1, Шаг 1: все доменные типы импортируются из config_types.py,
shared_contracts — единственный источник ALLOWED_*.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from src.config_types import (
    AudienceProfile,
    CoreConfig,
    DomainConfig,
    IntentConfig,
    KnowledgeBase,
    KnowledgeBudget,
    KnowledgeBudgetManager,
    KnowledgeBlockPlan,
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
# Загрузка конфигов (возвращают типы из config_types.py)
# ---------------------------------------------------------------------------


def load_core_config(base_path: Path = Path("config")) -> CoreConfig:
    data = load_json_file(base_path / "core.json")
    return CoreConfig(
        role=data.get("role", "You are a careful Russian editor."),
        priorities=data.get("priorities", "clarity, accuracy, readability"),
        basic_audit_instructions=data.get("basic_audit_instructions", []),
        forbidden=data.get("forbidden", []),
    )


def load_domain_config(
    domain: str,
    base_path: Path = Path("config"),
) -> DomainConfig:
    normalized_domain = domain.strip().lower()
    data = load_json_file(base_path / "domains" / f"{normalized_domain}.json")
    return DomainConfig(
        name=data.get("name", normalized_domain),
        system_rules=data.get("system_rules", ""),
        tone=data.get("tone", "neutral"),
        allow_storytelling=data.get("allow_storytelling", True),
        allow_marketing=data.get("allow_marketing", True),
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


def load_overlay_config(
    overlay: str,
    base_path: Path = Path("config"),
) -> OverlayConfig:
    normalized_overlay = normalize_tag(overlay)
    data = load_json_file(base_path / "overlays" / f"{normalized_overlay}.json")
    return OverlayConfig(
        name=data.get("name", normalized_overlay),
        instructions=data.get("instructions", []),
    )


def load_overlay_configs(
    overlays: Sequence[str],
    base_path: Path = Path("config"),
) -> List[OverlayConfig]:
    return [load_overlay_config(overlay, base_path) for overlay in overlays]


def load_output_format(
    mode: str,
    base_path: Path = Path("config"),
) -> str:
    data = load_json_file(base_path / "output_format.json")
    return data.get(mode, data.get("text_only", "Верни только отредактированный текст."))


# ---------------------------------------------------------------------------
# Загрузка базы знаний
# ---------------------------------------------------------------------------


def _normalize_kb_list(items: Any) -> List[Dict[str, Any]]:
    """Принимает список записей, нормализует теги в каждой записи."""
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


def _extract_records(container: Any, inherited_tags: List[str] = None) -> List[Dict[str, Any]]:
    """
    Рекурсивно извлекает все записи (словари с полями wrong, correct, name, rule и т.д.)
    из любых вложенных структур: списков, словарей с ключами 'examples', 'techniques', 'steps'.
    Добавляет унаследованные теги (например, из категории) к каждой записи.
    """
    if inherited_tags is None:
        inherited_tags = []
    records = []

    if isinstance(container, list):
        for item in container:
            records.extend(_extract_records(item, inherited_tags))
    elif isinstance(container, dict):
        # Если это категория с examples или techniques
        if "examples" in container and isinstance(container["examples"], list):
            # Наследуем теги категории
            cat_tags = container.get("tags", [])
            if isinstance(cat_tags, list):
                new_tags = inherited_tags + cat_tags
            else:
                new_tags = inherited_tags
            for ex in container["examples"]:
                if isinstance(ex, dict):
                    # Добавляем теги категории к записи
                    if "tags" in ex and isinstance(ex["tags"], list):
                        ex["tags"].extend(new_tags)
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
                    if "tags" in tech and isinstance(tech["tags"], list):
                        tech["tags"].extend(new_tags)
                    else:
                        tech["tags"] = new_tags.copy()
                    records.append(tech)
        else:
            # Обычная запись – добавляем как есть
            records.append(container)
    return records


def _load_kb_list(file_name: str, base_path: Path, key: str = None) -> List[Dict[str, Any]]:
    """
    Загружает JSON-файл базы знаний, извлекает все записи (даже из вложенных структур),
    нормализует теги и возвращает список записей.

    Поддерживает два формата JSON:
      - Словарь с ключом: {"key": [...]}  — стандартный формат
      - Плоский список:   [...]            — новый формат (key игнорируется)
    """
    file_path = base_path / file_name
    if not file_path.exists():
        logger.warning(f"KB file not found: {file_path}")
        return []
    data = _load_optional_json(file_path, {})

    # Поддержка обоих форматов: плоский список [...] и словарь {"key": [...]}
    if isinstance(data, list):
        items = data
    elif key is not None:
        items = data.get(key, [])
    else:
        items = []

    # Извлекаем записи рекурсивно
    records = _extract_records(items)

    # Для grammar_errors, если в записи нет тегов, добавляем "grammar"
    if file_name == "grammar_errors.json":
        for rec in records:
            if not rec.get("tags"):
                rec["tags"] = ["grammar"]

    # Нормализация тегов
    normalized = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        rec = dict(rec)
        if "tags" in rec and isinstance(rec["tags"], list):
            rec["tags"] = normalize_tags(rec["tags"])
        normalized.append(rec)
    return normalized


def load_knowledge_base(base_path: Path = Path("knowledge_base")) -> KnowledgeBase:
    # Файлы, которые являются словарями, а не списками (загружаем как есть)
    stop_words = _load_optional_json(base_path / "stop_words.json", {})
    domain_glossary = _load_optional_json(base_path / "domain_glossary.json", {})
    nkrj = _load_optional_json(base_path / "nkrj_structure_patterns.json", {})

    # Файлы с записями, используем универсальную функцию
    grammar_errors = _load_kb_list("grammar_errors.json", base_path, "common_mistakes")
    stylistic_issues = _load_kb_list("stylistic_issues.json", base_path, "stylistic_errors")
    logic_issues = _load_kb_list("logic_issues.json", base_path, "issues")
    storytelling_frameworks = _load_kb_list("storytelling_frameworks.json", base_path, "frameworks")
    marketing_templates = _load_kb_list("marketing_templates.json", base_path, "templates")
    composition_principles = _load_kb_list("composition_principles.json", base_path, "composition_principles")
    local_cohesion = _load_kb_list("local_cohesion.json", base_path, "local_cohesion")
    composition_errors = _load_kb_list("composition_errors.json", base_path, "composition_errors")
    rhetoric_frameworks = _load_kb_list("rhetoric_frameworks.json", base_path, "frameworks")
    editorial_techniques = _load_kb_list("editorial_techniques.json", base_path, "editorial_techniques")

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
    """Логирует WARNING только для знание-зависимых режимов.

    Для intent=None и intent='neutral' WARNING не выдаётся,
    даже если retrieval вернул EMPTY/NEUTRAL.
    """
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
# PromptBuilder
# ---------------------------------------------------------------------------


class PromptBuilder:
    """
    Фасад для сборки промпта из конфигов и базы знаний.

    Публичный API (инвариант):
      - PromptBuilder()                              — без обязательных аргументов
      - build(text, domain, intent, audience,
              overlays, output_mode, include_knowledge,
              knowledge_level, token_budget, include_retrieval_meta)
            -> Union[str, Tuple[str, Dict[str, Any]]]
      - get_available_intents()  -> Set[str]
      - get_available_overlays() -> Set[str]
      - reload_configs()         -> None
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
        self.core_config = None
        self.knowledge_base = None

    # ------------------------------------------------------------------
    # Доступные intents / overlays
    # ------------------------------------------------------------------

    def get_available_intents(self) -> Set[str]:
        intents_dir = self.config_path / "intents"
        if not intents_dir.exists():
            return set(ALLOWED_INTENTS)
        values = {path.stem for path in intents_dir.glob("*.json")}
        return values or set(ALLOWED_INTENTS)

    # legacy alias
    def getavailableintents(self) -> Set[str]:
        return self.get_available_intents()

    def get_available_overlays(self) -> Set[str]:
        overlays_dir = self.config_path / "overlays"
        if not overlays_dir.exists():
            return set(ALLOWED_OVERLAYS)
        values = {path.stem for path in overlays_dir.glob("*.json")}
        return values or set(ALLOWED_OVERLAYS)

    # legacy alias
    def getavailableoverlays(self) -> Set[str]:
        return self.get_available_overlays()

    # ------------------------------------------------------------------
    # Валидация входных параметров
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

    def _validate_overlays(self, overlays: Sequence[str]) -> List[str]:
        normalized = normalize_tags(overlays)
        available = set(self.get_available_overlays()) | ALLOWED_OVERLAYS
        invalid = [item for item in normalized if item not in available]
        if invalid:
            raise ValueError(
                f"Unsupported overlays: {invalid}. "
                f"Must be from {sorted(available)}"
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

    def _ensure_knowledge_base(self) -> Optional[KnowledgeBase]:
        if self.knowledge_base is not None:
            return self.knowledge_base
        try:
            self.knowledge_base = load_knowledge_base(self.kb_path)
        except FileNotFoundError:
            logger.warning("Knowledge base directory not found: %s", self.kb_path)
            return None
        return self.knowledge_base

    def _build_knowledge_block(
        self,
        text: str,
        primary_tags: Set[str],
        expanded_tags: Set[str],
        budget: KnowledgeBudget,
        domain: str,
        intent: Optional[str],
        overlays: List[str],
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Возвращает (текст блока знаний, метаданные по блокам).
        Метаданные: dict {block_name: {stage, entries_count, entry_ids, entry_names, truncated_count}}
        """
        kb = self._ensure_knowledge_base()
        if kb is None:
            return "", {}

        lines: List[str] = []
        meta: Dict[str, Any] = {}

        # Стоп-слова (не собираем метаданные)
        stop_words_budget = budget.get("stop_words")
        if stop_words_budget and stop_words_budget.enabled and kb.stop_words:
            lines.append("Стоп-слова и нежелательные формулировки:")
            category_limit = (
                stop_words_budget.entry_limit or self._limits.stop_words_category
            )
            for category, words in list(kb.stop_words.items())[:category_limit]:
                if isinstance(words, list) and words:
                    joined_words = ", ".join(
                        str(word) for word in words[: self._limits.stop_words_items]
                    )
                    lines.append(f"- {category}: {joined_words}")

        # Грамматика
        grammar_budget = budget.get("grammar")
        if grammar_budget and grammar_budget.enabled:
            grammar_result = select_grammar_rules(
                kb=kb,
                text=text,
                tags=primary_tags,
                limit=grammar_budget.entry_limit,
                candidate_limit=self._limits.grammar_candidates,
                char_budget=grammar_budget.char_budget,
                return_meta=True,
            )
            # Распаковка: теперь (entries, stage, dropped)
            if isinstance(grammar_result, tuple) and len(grammar_result) == 3:
                grammar_entries, stage, dropped = grammar_result
            elif isinstance(grammar_result, tuple) and len(grammar_result) == 2:
                grammar_entries, stage = grammar_result
                dropped = 0
            else:
                grammar_entries, stage = grammar_result, FallbackStage.STRONG
                dropped = 0

            _append_rule_entries(lines, "Грамматические ориентиры:", grammar_entries)

            meta["grammar"] = {
                "stage": stage.value,
                "entries_count": len(grammar_entries),
                "entry_ids": [e.get("id") for e in grammar_entries[:5] if e.get("id")],
                "entry_names": [e.get("name") for e in grammar_entries[:5] if e.get("name")],
                "truncated_count": dropped,
            }
            if dropped > 0:
                logger.info(
                    "Char budget truncated %d records for block='grammar', domain=%s, intent=%s",
                    dropped, domain, intent
                )
            if stage in (FallbackStage.EMPTY, FallbackStage.NEUTRAL):
                _warn_if_empty_retrieval(
                    block="grammar",
                    stage=stage,
                    domain=domain,
                    intent=intent,
                    overlays=overlays,
                    text_len=len(text),
                    primary_tags=primary_tags,
                )

        # Стиль
        style_budget = budget.get("style")
        if style_budget and style_budget.enabled:
            style_result = select_style_issues(
                kb=kb,
                text=text,
                tags=primary_tags,
                limit=style_budget.entry_limit,
                candidate_limit=self._limits.style_candidates,
                char_budget=style_budget.char_budget,
                return_meta=True,
            )
            if isinstance(style_result, tuple) and len(style_result) == 3:
                style_entries, stage, dropped = style_result
            elif isinstance(style_result, tuple) and len(style_result) == 2:
                style_entries, stage = style_result
                dropped = 0
            else:
                style_entries, stage = style_result, FallbackStage.STRONG
                dropped = 0

            _append_rule_entries(lines, "Стилистические ориентиры:", style_entries)

            meta["style"] = {
                "stage": stage.value,
                "entries_count": len(style_entries),
                "entry_ids": [e.get("id") for e in style_entries[:5] if e.get("id")],
                "entry_names": [e.get("name") for e in style_entries[:5] if e.get("name")],
                "truncated_count": dropped,
            }
            if dropped > 0:
                logger.info(
                    "Char budget truncated %d records for block='style', domain=%s, intent=%s",
                    dropped, domain, intent
                )
            if stage in (FallbackStage.EMPTY, FallbackStage.NEUTRAL):
                _warn_if_empty_retrieval(
                    block="style",
                    stage=stage,
                    domain=domain,
                    intent=intent,
                    overlays=overlays,
                    text_len=len(text),
                    primary_tags=primary_tags,
                )

        # Логика
        logic_budget = budget.get("logic")
        if logic_budget and logic_budget.enabled:
            logic_result = select_logic_issues(
                kb=kb,
                text=text,
                tags=primary_tags,
                limit=logic_budget.entry_limit,
                candidate_limit=self._limits.logic_candidates,
                char_budget=logic_budget.char_budget,
                return_meta=True,
            )
            if isinstance(logic_result, tuple) and len(logic_result) == 3:
                logic_entries, stage, dropped = logic_result
            elif isinstance(logic_result, tuple) and len(logic_result) == 2:
                logic_entries, stage = logic_result
                dropped = 0
            else:
                logic_entries, stage = logic_result, FallbackStage.STRONG
                dropped = 0

            _append_rule_entries(lines, "Логические ориентиры:", logic_entries)

            meta["logic"] = {
                "stage": stage.value,
                "entries_count": len(logic_entries),
                "entry_ids": [e.get("id") for e in logic_entries[:5] if e.get("id")],
                "entry_names": [e.get("name") for e in logic_entries[:5] if e.get("name")],
                "truncated_count": dropped,
            }
            if dropped > 0:
                logger.info(
                    "Char budget truncated %d records for block='logic', domain=%s, intent=%s",
                    dropped, domain, intent
                )
            if stage in (FallbackStage.EMPTY, FallbackStage.NEUTRAL):
                _warn_if_empty_retrieval(
                    block="logic",
                    stage=stage,
                    domain=domain,
                    intent=intent,
                    overlays=overlays,
                    text_len=len(text),
                    primary_tags=primary_tags,
                )

        # Композиция
        composition_budget = budget.get("composition")
        if composition_budget and composition_budget.enabled and kb.composition_principles:
            composition_result = select_structural_by_tags_or_all(
                entries=kb.composition_principles,
                tags=primary_tags,
                limit=composition_budget.entry_limit,
                expanded_tags=expanded_tags,
                char_budget=composition_budget.char_budget,
                return_meta=True,
            )
            if isinstance(composition_result, tuple) and len(composition_result) == 3:
                composition_entries, stage, dropped = composition_result
            elif isinstance(composition_result, tuple) and len(composition_result) == 2:
                composition_entries, stage = composition_result
                dropped = 0
            else:
                composition_entries, stage = composition_result, FallbackStage.STRONG
                dropped = 0

            _append_structural_entries(
                lines, "Принципы композиции:", composition_entries
            )

            meta["composition"] = {
                "stage": stage.value,
                "entries_count": len(composition_entries),
                "entry_ids": [e.get("id") for e in composition_entries[:5] if e.get("id")],
                "entry_names": [e.get("name") for e in composition_entries[:5] if e.get("name")],
                "truncated_count": dropped,
            }
            if dropped > 0:
                logger.info(
                    "Char budget truncated %d records for block='composition', domain=%s, intent=%s",
                    dropped, domain, intent
                )
            if stage in (FallbackStage.EMPTY, FallbackStage.NEUTRAL):
                _warn_if_empty_retrieval(
                    block="composition",
                    stage=stage,
                    domain=domain,
                    intent=intent,
                    overlays=overlays,
                    text_len=len(text),
                    primary_tags=primary_tags,
                )

        # Ошибки композиции
        composition_errors_budget = budget.get("composition_errors")
        if (
            composition_errors_budget
            and composition_errors_budget.enabled
            and kb.composition_errors
        ):
            comp_err_result = select_structural_by_tags_or_all(
                entries=kb.composition_errors,
                tags=primary_tags,
                limit=composition_errors_budget.entry_limit,
                expanded_tags=expanded_tags,
                char_budget=composition_errors_budget.char_budget,
                return_meta=True,
            )
            if isinstance(comp_err_result, tuple) and len(comp_err_result) == 3:
                comp_err_entries, stage, dropped = comp_err_result
            elif isinstance(comp_err_result, tuple) and len(comp_err_result) == 2:
                comp_err_entries, stage = comp_err_result
                dropped = 0
            else:
                comp_err_entries, stage = comp_err_result, FallbackStage.STRONG
                dropped = 0

            _append_structural_entries(
                lines, "Ошибки композиции:", comp_err_entries
            )

            meta["composition_errors"] = {
                "stage": stage.value,
                "entries_count": len(comp_err_entries),
                "entry_ids": [e.get("id") for e in comp_err_entries[:5] if e.get("id")],
                "entry_names": [e.get("name") for e in comp_err_entries[:5] if e.get("name")],
                "truncated_count": dropped,
            }
            if dropped > 0:
                logger.info(
                    "Char budget truncated %d records for block='composition_errors', domain=%s, intent=%s",
                    dropped, domain, intent
                )
            if stage in (FallbackStage.EMPTY, FallbackStage.NEUTRAL):
                _warn_if_empty_retrieval(
                    block="composition_errors",
                    stage=stage,
                    domain=domain,
                    intent=intent,
                    overlays=overlays,
                    text_len=len(text),
                    primary_tags=primary_tags,
                )

        # Локальная связность
        cohesion_budget = budget.get("cohesion")
        if cohesion_budget and cohesion_budget.enabled and kb.local_cohesion:
            cohesion_result = select_structural_by_tags_or_all(
                entries=kb.local_cohesion,
                tags=primary_tags,
                limit=cohesion_budget.entry_limit,
                expanded_tags=expanded_tags,
                char_budget=cohesion_budget.char_budget,
                return_meta=True,
            )
            if isinstance(cohesion_result, tuple) and len(cohesion_result) == 3:
                cohesion_entries, stage, dropped = cohesion_result
            elif isinstance(cohesion_result, tuple) and len(cohesion_result) == 2:
                cohesion_entries, stage = cohesion_result
                dropped = 0
            else:
                cohesion_entries, stage = cohesion_result, FallbackStage.STRONG
                dropped = 0

            _append_structural_entries(lines, "Локальная связность:", cohesion_entries)

            meta["cohesion"] = {
                "stage": stage.value,
                "entries_count": len(cohesion_entries),
                "entry_ids": [e.get("id") for e in cohesion_entries[:5] if e.get("id")],
                "entry_names": [e.get("name") for e in cohesion_entries[:5] if e.get("name")],
                "truncated_count": dropped,
            }
            if dropped > 0:
                logger.info(
                    "Char budget truncated %d records for block='cohesion', domain=%s, intent=%s",
                    dropped, domain, intent
                )
            if stage in (FallbackStage.EMPTY, FallbackStage.NEUTRAL):
                _warn_if_empty_retrieval(
                    block="cohesion",
                    stage=stage,
                    domain=domain,
                    intent=intent,
                    overlays=overlays,
                    text_len=len(text),
                    primary_tags=primary_tags,
                )

        # Сторителлинг
        storytelling_budget = budget.get("storytelling")
        if (
            storytelling_budget
            and storytelling_budget.enabled
            and kb.storytelling_frameworks
        ):
            story_result = select_structural_by_tags_or_all(
                entries=kb.storytelling_frameworks,
                tags=primary_tags,
                limit=storytelling_budget.entry_limit,
                expanded_tags=expanded_tags,
                char_budget=storytelling_budget.char_budget,
                return_meta=True,
            )
            if isinstance(story_result, tuple) and len(story_result) == 3:
                story_entries, stage, dropped = story_result
            elif isinstance(story_result, tuple) and len(story_result) == 2:
                story_entries, stage = story_result
                dropped = 0
            else:
                story_entries, stage = story_result, FallbackStage.STRONG
                dropped = 0

            _append_structural_entries(
                lines, "Сторителлинг-фреймворки:", story_entries
            )

            meta["storytelling"] = {
                "stage": stage.value,
                "entries_count": len(story_entries),
                "entry_ids": [e.get("id") for e in story_entries[:5] if e.get("id")],
                "entry_names": [e.get("name") for e in story_entries[:5] if e.get("name")],
                "truncated_count": dropped,
            }
            if dropped > 0:
                logger.info(
                    "Char budget truncated %d records for block='storytelling', domain=%s, intent=%s",
                    dropped, domain, intent
                )
            if stage in (FallbackStage.EMPTY, FallbackStage.NEUTRAL):
                _warn_if_empty_retrieval(
                    block="storytelling",
                    stage=stage,
                    domain=domain,
                    intent=intent,
                    overlays=overlays,
                    text_len=len(text),
                    primary_tags=primary_tags,
                )

        # Маркетинговые шаблоны
        marketing_budget = budget.get("marketing")
        if marketing_budget and marketing_budget.enabled and kb.marketing_templates:
            marketing_result = select_structural_by_tags_or_all(
                entries=kb.marketing_templates,
                tags=primary_tags,
                limit=marketing_budget.entry_limit,
                expanded_tags=expanded_tags,
                char_budget=marketing_budget.char_budget,
                return_meta=True,
            )
            if isinstance(marketing_result, tuple) and len(marketing_result) == 3:
                marketing_entries, stage, dropped = marketing_result
            elif isinstance(marketing_result, tuple) and len(marketing_result) == 2:
                marketing_entries, stage = marketing_result
                dropped = 0
            else:
                marketing_entries, stage = marketing_result, FallbackStage.STRONG
                dropped = 0

            _append_structural_entries(
                lines, "Маркетинговые шаблоны:", marketing_entries
            )

            meta["marketing"] = {
                "stage": stage.value,
                "entries_count": len(marketing_entries),
                "entry_ids": [e.get("id") for e in marketing_entries[:5] if e.get("id")],
                "entry_names": [e.get("name") for e in marketing_entries[:5] if e.get("name")],
                "truncated_count": dropped,
            }
            if dropped > 0:
                logger.info(
                    "Char budget truncated %d records for block='marketing', domain=%s, intent=%s",
                    dropped, domain, intent
                )
            if stage in (FallbackStage.EMPTY, FallbackStage.NEUTRAL):
                _warn_if_empty_retrieval(
                    block="marketing",
                    stage=stage,
                    domain=domain,
                    intent=intent,
                    overlays=overlays,
                    text_len=len(text),
                    primary_tags=primary_tags,
                )

        # Риторические приёмы
        rhetoric_budget = budget.get("rhetoric")
        if rhetoric_budget and rhetoric_budget.enabled and kb.rhetoric_frameworks:
            rhetoric_result = select_structural_by_tags_or_all(
                entries=kb.rhetoric_frameworks,
                tags=primary_tags,
                limit=rhetoric_budget.entry_limit,
                expanded_tags=expanded_tags,
                char_budget=rhetoric_budget.char_budget,
                return_meta=True,
            )
            if isinstance(rhetoric_result, tuple) and len(rhetoric_result) == 3:
                rhetoric_entries, stage, dropped = rhetoric_result
            elif isinstance(rhetoric_result, tuple) and len(rhetoric_result) == 2:
                rhetoric_entries, stage = rhetoric_result
                dropped = 0
            else:
                rhetoric_entries, stage = rhetoric_result, FallbackStage.STRONG
                dropped = 0

            _append_structural_entries(lines, "Риторические приёмы:", rhetoric_entries)

            meta["rhetoric"] = {
                "stage": stage.value,
                "entries_count": len(rhetoric_entries),
                "entry_ids": [e.get("id") for e in rhetoric_entries[:5] if e.get("id")],
                "entry_names": [e.get("name") for e in rhetoric_entries[:5] if e.get("name")],
                "truncated_count": dropped,
            }
            if dropped > 0:
                logger.info(
                    "Char budget truncated %d records for block='rhetoric', domain=%s, intent=%s",
                    dropped, domain, intent
                )
            if stage in (FallbackStage.EMPTY, FallbackStage.NEUTRAL):
                _warn_if_empty_retrieval(
                    block="rhetoric",
                    stage=stage,
                    domain=domain,
                    intent=intent,
                    overlays=overlays,
                    text_len=len(text),
                    primary_tags=primary_tags,
                )

        # Редакторские приёмы
        editorial_budget = budget.get("editorial")
        if editorial_budget and editorial_budget.enabled and kb.editorial_techniques:
            editorial_result = select_structural_by_tags_or_all(
                entries=kb.editorial_techniques,
                tags=primary_tags,
                limit=editorial_budget.entry_limit,
                expanded_tags=expanded_tags,
                char_budget=editorial_budget.char_budget,
                return_meta=True,
            )
            if isinstance(editorial_result, tuple) and len(editorial_result) == 3:
                editorial_entries, stage, dropped = editorial_result
            elif isinstance(editorial_result, tuple) and len(editorial_result) == 2:
                editorial_entries, stage = editorial_result
                dropped = 0
            else:
                editorial_entries, stage = editorial_result, FallbackStage.STRONG
                dropped = 0

            _append_editorial_entries(lines, "Редакторские приёмы:", editorial_entries)

            meta["editorial"] = {
                "stage": stage.value,
                "entries_count": len(editorial_entries),
                "entry_ids": [e.get("id") for e in editorial_entries[:5] if e.get("id")],
                "entry_names": [e.get("name") for e in editorial_entries[:5] if e.get("name")],
                "truncated_count": dropped,
            }
            if dropped > 0:
                logger.info(
                    "Char budget truncated %d records for block='editorial', domain=%s, intent=%s",
                    dropped, domain, intent
                )
            if stage in (FallbackStage.EMPTY, FallbackStage.NEUTRAL):
                _warn_if_empty_retrieval(
                    block="editorial",
                    stage=stage,
                    domain=domain,
                    intent=intent,
                    overlays=overlays,
                    text_len=len(text),
                    primary_tags=primary_tags,
                )

        # Глоссарий (без метаданных)
        glossary_budget = budget.get("glossary")
        if glossary_budget and glossary_budget.enabled and kb.domain_glossary:
            _append_glossary(lines, kb.domain_glossary, glossary_budget.entry_limit)

        # НКРЯ (без метаданных)
        nkrj_budget = budget.get("nkrj")
        if nkrj_budget and nkrj_budget.enabled and kb.nkrj_structure_patterns:
            _append_nkrj(lines, kb.nkrj_structure_patterns)

        return "\n".join(lines), meta

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
        knowledge_level: KnowledgeLevel = KnowledgeLevel.STANDARD,
        token_budget: Optional[int] = None,
        include_retrieval_meta: bool = False,
        **legacy_kwargs: Any,
    ) -> Union[str, Tuple[str, Dict[str, Any]]]:
        """
        Собирает промпт из конфигов и базы знаний.

        Параметры:
            text             — исходный текст для редактирования
            domain           — домен ('marketing' | 'blog' | 'deai')
            intent           — опциональный intent ('storytelling' | 'noragal' | 'deai' | 'neutral')
            audience         — профиль аудитории (AudienceProfile или None)
            overlays         — список оверлеев
            output_mode      — формат ответа ('text_only' | 'text_and_report')
            include_knowledge — включать ли блок базы знаний
            knowledge_level  — уровень детализации знаний (KnowledgeLevel)
            token_budget     — лимит токенов для knowledge-блока (None = без лимита)
            include_retrieval_meta — если True, возвращает (prompt, retrieval_meta)

        Возвращает:
            если include_retrieval_meta=False: строка с промптом
            если include_retrieval_meta=True: кортеж (prompt, meta)
        """
        # Поддержка legacy camelCase kwargs от старых клиентов
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

        domain_config = load_domain_config(validated_domain, self.config_path)
        intent_config = load_intent_config(validated_intent, self.config_path)
        overlay_configs = load_overlay_configs(validated_overlays, self.config_path)
        output_format = load_output_format(validated_output_mode, self.config_path)

        blocks: List[str] = []

        blocks.append(f"Роль: {self.core_config.role}")
        blocks.append(f"Приоритеты: {self.core_config.priorities}")
        blocks.append(f"Домен: {domain_config.name}")
        blocks.append(f"Тон: {domain_config.tone}")

        if domain_config.system_rules:
            blocks.append("Правила домена:\n" + domain_config.system_rules)

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
            knowledge_block, block_meta = self._build_knowledge_block(
                text=text,
                primary_tags=tag_sets["primary"],
                expanded_tags=tag_sets["expanded"],
                budget=budget,
                domain=validated_domain,
                intent=validated_intent,
                overlays=validated_overlays,
            )
            retrieval_meta_total = block_meta
            if knowledge_block:
                blocks.append("База знаний:\n" + knowledge_block)

        blocks.append("Формат ответа:\n" + output_format)
        blocks.append("Исходный текст:\n" + text.strip())

        prompt = "\n\n".join(block for block in blocks if block.strip())

        if include_retrieval_meta:
            return prompt, retrieval_meta_total
        return prompt

    # legacy alias
    def build_prompt(self, **kwargs: Any) -> str:
        return self.build(**kwargs)
