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
from typing import Any, Dict, List, Optional, Sequence, Set

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
    return data.get(mode, data.get("textonly", "Верни только отредактированный текст."))


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


def load_knowledge_base(base_path: Path = Path("knowledge_base")) -> KnowledgeBase:
    return KnowledgeBase(
        stop_words=_load_optional_json(base_path / "stop_words.json", {}),
        grammar_errors=_normalize_kb_list(
            _load_optional_json(base_path / "grammar_errors.json", [])
        ),
        stylistic_issues=_normalize_kb_list(
            _load_optional_json(base_path / "stylistic_issues.json", [])
        ),
        logic_issues=_normalize_kb_list(
            _load_optional_json(base_path / "logic_issues.json", [])
        ),
        storytelling_frameworks=_normalize_kb_list(
            _load_optional_json(base_path / "storytelling_frameworks.json", [])
        ),
        marketing_templates=_normalize_kb_list(
            _load_optional_json(base_path / "marketing_templates.json", [])
        ),
        domain_glossary=_load_optional_json(base_path / "domain_glossary.json", {}),
        composition_principles=_normalize_kb_list(
            _load_optional_json(base_path / "composition_principles.json", [])
        ),
        local_cohesion=_normalize_kb_list(
            _load_optional_json(base_path / "local_cohesion.json", [])
        ),
        composition_errors=_normalize_kb_list(
            _load_optional_json(base_path / "composition_errors.json", [])
        ),
        rhetoric_frameworks=_normalize_kb_list(
            _load_optional_json(base_path / "rhetoric_frameworks.json", [])
        ),
        editorial_techniques=_normalize_kb_list(
            _load_optional_json(base_path / "editorial_techniques.json", [])
        ),
        nkrj_structure_patterns=_load_optional_json(
            base_path / "nkrj_structure_patterns.json", {}
        ),
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
              knowledge_level, token_budget)         — только дефолтные доп. параметры
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
    ) -> str:
        kb = self._ensure_knowledge_base()
        if kb is None:
            return ""

        lines: List[str] = []

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

        grammar_budget = budget.get("grammar")
        if grammar_budget and grammar_budget.enabled:
            grammar_entries = select_grammar_rules(
                kb=kb,
                text=text,
                tags=primary_tags,
                limit=grammar_budget.entry_limit,
                candidate_limit=self._limits.grammar_candidates,
                char_budget=grammar_budget.char_budget,
            )
            _append_rule_entries(lines, "Грамматические ориентиры:", grammar_entries)

        style_budget = budget.get("style")
        if style_budget and style_budget.enabled:
            style_entries = select_style_issues(
                kb=kb,
                text=text,
                tags=primary_tags,
                limit=style_budget.entry_limit,
                candidate_limit=self._limits.style_candidates,
                char_budget=style_budget.char_budget,
            )
            _append_rule_entries(lines, "Стилистические ориентиры:", style_entries)

        logic_budget = budget.get("logic")
        if logic_budget and logic_budget.enabled:
            logic_entries = select_logic_issues(
                kb=kb,
                text=text,
                tags=primary_tags,
                limit=logic_budget.entry_limit,
                candidate_limit=self._limits.logic_candidates,
                char_budget=logic_budget.char_budget,
            )
            _append_rule_entries(lines, "Логические ориентиры:", logic_entries)

        composition_budget = budget.get("composition")
        if composition_budget and composition_budget.enabled and kb.composition_principles:
            composition_entries = select_structural_by_tags_or_all(
                entries=kb.composition_principles,
                tags=primary_tags,
                limit=composition_budget.entry_limit,
                expanded_tags=expanded_tags,
                char_budget=composition_budget.char_budget,
            )
            _append_structural_entries(
                lines, "Принципы композиции:", composition_entries
            )

        composition_errors_budget = budget.get("composition_errors")
        if (
            composition_errors_budget
            and composition_errors_budget.enabled
            and kb.composition_errors
        ):
            composition_error_entries = select_structural_by_tags_or_all(
                entries=kb.composition_errors,
                tags=primary_tags,
                limit=composition_errors_budget.entry_limit,
                expanded_tags=expanded_tags,
                char_budget=composition_errors_budget.char_budget,
            )
            _append_structural_entries(
                lines, "Ошибки композиции:", composition_error_entries
            )

        cohesion_budget = budget.get("cohesion")
        if cohesion_budget and cohesion_budget.enabled and kb.local_cohesion:
            cohesion_entries = select_structural_by_tags_or_all(
                entries=kb.local_cohesion,
                tags=primary_tags,
                limit=cohesion_budget.entry_limit,
                expanded_tags=expanded_tags,
                char_budget=cohesion_budget.char_budget,
            )
            _append_structural_entries(lines, "Локальная связность:", cohesion_entries)

        storytelling_budget = budget.get("storytelling")
        if (
            storytelling_budget
            and storytelling_budget.enabled
            and kb.storytelling_frameworks
        ):
            storytelling_entries = select_structural_by_tags_or_all(
                entries=kb.storytelling_frameworks,
                tags=primary_tags,
                limit=storytelling_budget.entry_limit,
                expanded_tags=expanded_tags,
                char_budget=storytelling_budget.char_budget,
            )
            _append_structural_entries(
                lines, "Сторителлинг-фреймворки:", storytelling_entries
            )

        marketing_budget = budget.get("marketing")
        if marketing_budget and marketing_budget.enabled and kb.marketing_templates:
            marketing_entries = select_structural_by_tags_or_all(
                entries=kb.marketing_templates,
                tags=primary_tags,
                limit=marketing_budget.entry_limit,
                expanded_tags=expanded_tags,
                char_budget=marketing_budget.char_budget,
            )
            _append_structural_entries(
                lines, "Маркетинговые шаблоны:", marketing_entries
            )

        rhetoric_budget = budget.get("rhetoric")
        if rhetoric_budget and rhetoric_budget.enabled and kb.rhetoric_frameworks:
            rhetoric_entries = select_structural_by_tags_or_all(
                entries=kb.rhetoric_frameworks,
                tags=primary_tags,
                limit=rhetoric_budget.entry_limit,
                expanded_tags=expanded_tags,
                char_budget=rhetoric_budget.char_budget,
            )
            _append_structural_entries(lines, "Риторические приёмы:", rhetoric_entries)

        editorial_budget = budget.get("editorial")
        if editorial_budget and editorial_budget.enabled and kb.editorial_techniques:
            editorial_entries = select_structural_by_tags_or_all(
                entries=kb.editorial_techniques,
                tags=primary_tags,
                limit=editorial_budget.entry_limit,
                expanded_tags=expanded_tags,
                char_budget=editorial_budget.char_budget,
            )
            _append_editorial_entries(lines, "Редакторские приёмы:", editorial_entries)

        glossary_budget = budget.get("glossary")
        if glossary_budget and glossary_budget.enabled and kb.domain_glossary:
            _append_glossary(lines, kb.domain_glossary, glossary_budget.entry_limit)

        nkrj_budget = budget.get("nkrj")
        if nkrj_budget and nkrj_budget.enabled and kb.nkrj_structure_patterns:
            _append_nkrj(lines, kb.nkrj_structure_patterns)

        return "\n".join(lines)

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
        output_mode: str = "textonly",
        include_knowledge: bool = True,
        knowledge_level: KnowledgeLevel = KnowledgeLevel.STANDARD,
        token_budget: Optional[int] = None,
        **legacy_kwargs: Any,
    ) -> str:
        """
        Собирает промпт из конфигов и базы знаний.

        Параметры:
            text             — исходный текст для редактирования
            domain           — домен ('marketing' | 'blog' | 'deai')
            intent           — опциональный intent ('storytelling' | 'noragal' | 'deai' | 'neutral')
            audience         — профиль аудитории (AudienceProfile или None)
            overlays         — список оверлеев
            output_mode      — формат ответа ('textonly' | 'textandreport')
            include_knowledge — включать ли блок базы знаний
            knowledge_level  — уровень детализации знаний (KnowledgeLevel)
            token_budget     — лимит токенов для knowledge-блока (None = без лимита)
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
            knowledge_block = self._build_knowledge_block(
                text=text,
                primary_tags=tag_sets["primary"],
                expanded_tags=tag_sets["expanded"],
                budget=budget,
            )
            if knowledge_block:
                blocks.append("База знаний:\n" + knowledge_block)

        blocks.append("Формат ответа:\n" + output_format)
        blocks.append("Исходный текст:\n" + text.strip())

        return "\n\n".join(block for block in blocks if block.strip())

    # legacy alias
    def build_prompt(self, **kwargs: Any) -> str:
        return self.build(**kwargs)
