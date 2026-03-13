"""
prompt_builder.py

Модуль для сборки финальных промптов из конфигов и базы знаний.
Следует принципам Clean Code: типизация, одна ответственность на функцию,
явные зависимости, читаемость.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


# ============================================================================
# Data Models (типы данных для конфигов)
# ============================================================================


@dataclass(frozen=True)
class CoreConfig:
    """Базовая конфигурация редактора (общая для всех режимов)."""
    role: str
    priorities: str
    basic_audit_instructions: List[str]
    forbidden: List[str]


@dataclass(frozen=True)
class DomainConfig:
    """Конфигурация домена (тип текста: маркетинг, блог и т.п.)."""
    name: str
    system_rules: str
    tone: str
    allow_storytelling: bool = True
    allow_marketing: bool = True


@dataclass(frozen=True)
class IntentConfig:
    """Конфигурация цели обработки (точнее, увлекательнее и т.п.)."""
    name: str
    instructions: List[str]


@dataclass(frozen=True)
class OverlayConfig:
    """Конфигурация надстройки (инфостиль, логика, фактчек и т.п.)."""
    name: str
    instructions: List[str]


@dataclass(frozen=True)
class AudienceProfile:
    """Профиль аудитории."""
    kind: str  # "b2b" | "b2c" | "mixed" | "custom"
    expertise: str  # "novice" | "pro" | "expert"
    formality: str  # "casual" | "neutral" | "formal"
    description: str = ""


@dataclass(frozen=True)
class KnowledgeBase:
    """
    База знаний:
    - stop_words: словари стоп-слов и нежелательных конструкций
    - grammar_errors: типичные грамматические / орфографические ошибки
    - stylistic_issues: типичные стилистические проблемы (канцелярит, штампы и т.п.)
    - storytelling_frameworks: фреймворки для сторителлинга/структуры истории
    - marketing_templates: шаблоны маркетинговых текстов (лендинг, письма, посты)
    - domain_glossary: термины и определения по доменам (опционально)
    """
    stop_words: Dict[str, List[str]]
    grammar_errors: List[Dict[str, Any]]
    stylistic_issues: List[Dict[str, Any]]
    storytelling_frameworks: List[Dict[str, Any]]
    marketing_templates: List[Dict[str, Any]]
    domain_glossary: Dict[str, Any]


# ============================================================================
# Config Loaders (функции загрузки конфигов)
# ============================================================================


def load_json_file(path: Path) -> dict:
    """
    Загружает JSON-файл.

    Args:
        path: Путь к JSON-файлу

    Returns:
        Распарсенный JSON как словарь
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_core_config(base_path: Path = Path("config")) -> CoreConfig:
    """Загружает базовую конфигурацию редактора."""
    data = load_json_file(base_path / "core.json")
    return CoreConfig(
        role=data["role"],
        priorities=data["priorities"],
        basic_audit_instructions=data["basic_audit_instructions"],
        forbidden=data["forbidden"],
    )


def load_domain_config(domain: str, base_path: Path = Path("config")) -> DomainConfig:
    """
    Загружает конфигурацию домена.
    """
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
    """
    Загружает конфигурацию цели обработки.
    """
    if intent is None or intent == "neutral":
        return None

    data = load_json_file(base_path / "intents" / f"{intent}.json")
    return IntentConfig(
        name=data["name"],
        instructions=data["instructions"],
    )


def load_overlay_configs(
    overlays: Sequence[str],
    base_path: Path = Path("config"),
) -> List[OverlayConfig]:
    """
    Загружает конфигурации надстроек.
    """
    configs: List[OverlayConfig] = []
    for overlay in overlays:
        data = load_json_file(base_path / "overlays" / f"{overlay}.json")
        configs.append(
            OverlayConfig(
                name=data["name"],
                instructions=data["instructions"],
            )
        )
    return configs


def load_output_format(
    mode: str,
    base_path: Path = Path("config"),
) -> str:
    """
    Загружает шаблон формата вывода.
    """
    data = load_json_file(base_path / "output_format.json")
    return data.get(mode, data["text_only"])


def load_knowledge_base(base_path: Path = Path("knowledge_base")) -> KnowledgeBase:
    """
    Загружает базу знаний из папки knowledge_base.
    """
    stop_words = load_json_file(base_path / "stop_words.json")
    grammar = load_json_file(base_path / "grammar_errors.json")
    style = load_json_file(base_path / "stylistic_issues.json")
    storytelling = load_json_file(base_path / "storytelling_frameworks.json")
    marketing = load_json_file(base_path / "marketing_templates.json")

    glossary_path = base_path / "domain_glossary.json"
    domain_glossary: Dict[str, Any] = {}
    if glossary_path.exists():
        domain_glossary = load_json_file(glossary_path)

    return KnowledgeBase(
        stop_words=stop_words,
        grammar_errors=grammar.get("common_mistakes", []),
        stylistic_issues=style.get("common_issues", []),
        storytelling_frameworks=storytelling.get("frameworks", []),
        marketing_templates=marketing.get("templates", []),
        domain_glossary=domain_glossary,
    )


# ============================================================================
# Knowledge selection helpers
# ============================================================================


def _match_tags(entry_tags: Iterable[str], wanted_tags: Iterable[str]) -> bool:
    entry = set(t.strip().lower() for t in (entry_tags or []))
    wanted = set(t.strip().lower() for t in (wanted_tags or []))
    return bool(entry & wanted) if wanted else True


def select_grammar_rules(
    kb: KnowledgeBase,
    text: str,
    tags: Iterable[str],
    limit: int = 10,
) -> List[Dict[str, Any]]:
    text_lower = text.lower()
    wanted_tags = list(tags)

    matches: List[Dict[str, Any]] = []
    for err in kb.grammar_errors:
        wrong = str(err.get("wrong", "")).lower()
        entry_tags = err.get("tags", [])
        if wrong and wrong in text_lower and _match_tags(entry_tags, wanted_tags):
            matches.append(err)
            if len(matches) >= limit:
                break

    if not matches:
        for err in kb.grammar_errors:
            if _match_tags(err.get("tags", []), wanted_tags):
                matches.append(err)
                if len(matches) >= limit:
                    break

    return matches


def select_style_issues(
    kb: KnowledgeBase,
    text: str,
    tags: Iterable[str],
    limit: int = 10,
) -> List[Dict[str, Any]]:
    text_lower = text.lower()
    wanted_tags = list(tags)

    matches: List[Dict[str, Any]] = []
    for issue in kb.stylistic_issues:
        wrong = str(issue.get("wrong", "")).lower()
        entry_tags = issue.get("tags", [])
        if wrong and wrong in text_lower and _match_tags(entry_tags, wanted_tags):
            matches.append(issue)
            if len(matches) >= limit:
                break

    if not matches:
        for issue in kb.stylistic_issues:
            if _match_tags(issue.get("tags", []), wanted_tags):
                matches.append(issue)
                if len(matches) >= limit:
                    break

    return matches


def select_logic_issues(
    kb: KnowledgeBase,
    text: str,
    tags: Iterable[str],
    limit: int = 8,
) -> List[Dict[str, Any]]:
    """
    Логические проблемы пока берём из stylistic_issues / grammar по тегам "logic".
    Если заведёшь отдельный JSON под логику, можно будет переключить на него.
    """
    text_lower = text.lower()
    wanted_tags = list(tags) + ["logic"]

    candidates: List[Dict[str, Any]] = kb.stylistic_issues + kb.grammar_errors
    matches: List[Dict[str, Any]] = []

    for item in candidates:
        pattern = str(item.get("wrong", "")).lower()
        entry_tags = item.get("tags", [])
        if pattern and pattern in text_lower and _match_tags(entry_tags, wanted_tags):
            matches.append(item)
            if len(matches) >= limit:
                break

    if not matches:
        for item in candidates:
            if _match_tags(item.get("tags", []), wanted_tags):
                matches.append(item)
                if len(matches) >= limit:
                    break

    return matches


# ============================================================================
# Prompt Builder (сборщик промпта)
# ============================================================================


class PromptBuilder:
    """
    Собирает финальный промпт из конфигов, базы знаний и параметров запроса.
    """

    def __init__(
        self,
        config_path: Path = Path("config"),
        kb_path: Path = Path("knowledge_base"),
    ):
        self.config_path = config_path
        self.kb_path = kb_path

    def build(
        self,
        text: str,
        domain: str,
        intent: Optional[str] = None,
        audience: Optional[AudienceProfile] = None,
        overlays: Sequence[str] = (),
        output_mode: str = "text_only",
        include_knowledge: bool = True,
    ) -> str:
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
        core = load_core_config(self.config_path)

        instructions = "\n".join(f"- {instr}" for instr in core.basic_audit_instructions)
        forbidden = "\n".join(f"❌ {rule}" for rule in core.forbidden)

        return f"""{core.role}

{core.priorities}

Задачи:
{instructions}

Запреты:
{forbidden}"""

    def _build_domain_block(self, domain: str) -> str:
        domain_cfg = load_domain_config(domain, self.config_path)
        return f"""Домен: {domain_cfg.system_rules}
Тон: {domain_cfg.tone}"""

    def _build_intent_block(self, intent: str) -> str:
        intent_cfg = load_intent_config(intent, self.config_path)
        if intent_cfg is None:
            return ""

        instructions = "\n".join(f"- {instr}" for instr in intent_cfg.instructions)
        return f"""Цель обработки: {intent_cfg.name}

Требования:
{instructions}"""

    def _build_audience_block(self, audience: Optional[AudienceProfile]) -> str:
        if audience is None:
            return "Аудитория: не указана. Используй нейтральный профессиональный тон."

        description_line = f"\n- Описание: {audience.description}" if audience.description else ""
        return f"""Аудитория:
- Тип: {audience.kind}
- Уровень экспертизы: {audience.expertise}
- Формальность: {audience.formality}{description_line}"""

    def _build_overlays_block(self, overlays: Sequence[str]) -> str:
        overlay_configs = load_overlay_configs(overlays, self.config_path)

        parts: List[str] = ["Дополнительные режимы:"]
        for cfg in overlay_configs:
            instructions = "\n".join(f"  - {instr}" for instr in cfg.instructions)
            parts.append(f"\n• {cfg.name}:\n{instructions}")

        return "\n".join(parts)

    def _build_knowledge_block(
        self,
        text: str,
        domain: str,
        intent: Optional[str],
        overlays: Sequence[str],
    ) -> str:
        kb = load_knowledge_base(self.kb_path)

        stop_words_text = json.dumps(kb.stop_words, ensure_ascii=False, indent=2)

        tags: List[str] = [domain]
        if intent:
            tags.append(intent)
        tags.extend(overlays)

        grammar_sample = select_grammar_rules(kb, text=text, tags=tags, limit=10)
        style_sample = select_style_issues(kb, text=text, tags=tags, limit=10)
        logic_sample = select_logic_issues(kb, text=text, tags=tags, limit=8)

        grammar_lines: List[str] = [
            f"  • {err.get('wrong', '')} → {err.get('correct', '').strip()} ({err.get('rule', '').strip()})"
            for err in grammar_sample
            if err.get("wrong") and err.get("correct")
        ] or ["  • (нет подходящих примеров)"]

        style_lines: List[str] = [
            f"  • {issue.get('wrong', '')} → {issue.get('correct', '').strip()} ({issue.get('rule', '').strip()})"
            for issue in style_sample
            if issue.get("wrong")
        ] or ["  • (нет подходящих примеров)"]

        logic_lines: List[str] = [
            f"  • {item.get('wrong', '')}: {item.get('rule', item.get('description', '')).strip()}"
            for item in logic_sample
            if item.get("wrong")
        ] or ["  • (нет подходящих примеров)"]

        frameworks_text = ""
        if intent == "storytelling" and kb.storytelling_frameworks:
            frameworks_sample = kb.storytelling_frameworks[:4]
            framework_lines: List[str] = []
            for fw in frameworks_sample:
                name = fw.get("name", "")
                steps = fw.get("steps", [])
                step_names = [
                    step.get("name", "")
                    for step in steps
                    if isinstance(step, Dict) and step.get("name")
                ]
                if not name or not step_names:
                    continue
                framework_lines.append(f"  • {name}: " + " → ".join(step_names))
            if framework_lines:
                frameworks_text = (
                    "\n\nФреймворки сторителлинга (для структуры рассказа):\n"
                    + "\n".join(framework_lines)
                )

        marketing_text = ""
        if domain == "marketing" and kb.marketing_templates:
            templates_sample = kb.marketing_templates[:4]
            template_lines: List[str] = []
            for tpl in templates_sample:
                name = tpl.get("name", "")
                sections = tpl.get("sections", [])
                section_names = [
                    sec.get("name", "")
                    for sec in sections
                    if isinstance(sec, Dict) and sec.get("name")
                ]
                if not name or not section_names:
                    continue
                template_lines.append(f"  • {name}: " + ", ".join(section_names))
            if template_lines:
                marketing_text = (
                    "\n\nМаркетинговые шаблоны (структура текста по типу):\n"
                    + "\n".join(template_lines)
                )

        glossary_text = ""
        if kb.domain_glossary and domain in kb.domain_glossary:
            terms = kb.domain_glossary.get(domain, {})
            if isinstance(terms, dict) and terms:
                sample_items = list(terms.items())[:10]
                term_lines = [f"  • {k}: {v}" for k, v in sample_items]
                glossary_text = (
                    "\n\nГлоссарий по домену (ключевые термины):\n"
                    + "\n".join(term_lines)
                )

        return (
            "База знаний:\n\n"
            "Стоп-слова и нежелательные конструкции (удаляй или переписывай):\n"
            f"{stop_words_text}\n\n"
            "Типичные грамматические и лексические ошибки (исправляй по аналогии):\n"
            + "\n".join(grammar_lines)
            + "\n\nТипичные стилистические проблемы (канцелярит, штампы, вода — устраняй):\n"
            + "\n".join(style_lines)
            + "\n\nТипичные логические проблемы и риски связности:\n"
            + "\n".join(logic_lines)
            + frameworks_text
            + marketing_text
            + glossary_text
        )

    def _build_output_format_block(self, mode: str) -> str:
        format_text = load_output_format(mode, self.config_path)
        return f"Формат ответа:\n{format_text}"

    def _build_text_block(self, text: str) -> str:
        return f'Текст для обработки:\n"""\n{text}\n"""'


def build_prompt(
    text: str,
    domain: str = "marketing",
    intent: Optional[str] = None,
    audience_type: str = "b2b",
    overlays: Sequence[str] = (),
    output_mode: str = "text_only",
) -> str:
    audience_map = {
        "b2b": AudienceProfile(kind="b2b", expertise="pro", formality="neutral"),
        "b2c": AudienceProfile(kind="b2c", expertise="novice", formality="casual"),
        "mixed": AudienceProfile(kind="mixed", expertise="pro", formality="neutral"),
    }
    audience = audience_map.get(audience_type, audience_map["b2b"])

    builder = PromptBuilder()
    return builder.build(
        text=text,
        domain=domain,
        intent=intent,
        audience=audience,
        overlays=overlays,
        output_mode=output_mode,
    )
