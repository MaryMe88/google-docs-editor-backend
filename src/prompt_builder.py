"""
prompt_builder.py

Модуль для сборки финальных промптов из конфигов и базы знаний.
Следует принципам Clean Code: типизация, одна ответственность на функцию,
явные зависимости, читаемость.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypedDict,
    Union,
)

from src.knowledge_retrieval import (
    normalize_text_for_match,
    score_rule_entry,
    score_structural_entry,
    select_grammar_rules,
    select_style_issues,
    select_logic_issues,
    select_structural_by_tags_or_all,
    _select_ranked_entries,
)
from src.config_types import AudienceProfile, LimitsConfig

logger = logging.getLogger(__name__)

# ============================================================================
# Типы для записей Knowledge Base
# ============================================================================

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

FlatEntry = Dict[str, Any]  # TODO: заменить на TypedDict после стабилизации форматов KB

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
class KnowledgeBase:
    """
    База знаний:
    - stop_words: словари стоп-слов и нежелательных конструкций
    - grammar_errors: типичные грамматические / орфографические ошибки
    - stylistic_issues: типичные стилистические проблемы (канцелярит, штампы и т.п.)
    - logic_issues: логические ошибки и проблемы связности
    - storytelling_frameworks: фреймворки для сторителлинга/структуры истории
    - marketing_templates: шаблоны маркетинговых текстов (лендинг, письма, посты)
    - domain_glossary: термины и определения по доменам (опционально)
    - composition_principles: принципы композиции (типы построения, глобальная связность)
    - local_cohesion: приёмы локальной связности (абзац, тема-рема, местоимения)
    - composition_errors: типичные композиционные ошибки
    - rhetoric_frameworks: риторические топосы и приёмы аргументации
    - editorial_techniques: редакторские приёмы (Нора Галь, Мильчин и др.)
    - nkrj_structure_patterns: шаблоны структуры по данным НКРЯ
    """
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

# ============================================================================
# Config Loaders
# ============================================================================

def load_json_file(path: Path) -> dict:
    """Загружает JSON-файл."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

def _load_optional_json(path: Path, default: Any = None) -> Any:
    """Загружает JSON из файла, если он существует, иначе возвращает default."""
    if path.exists():
        return load_json_file(path)
    return default

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
    """Загружает конфигурацию домена."""
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
    """Загружает конфигурацию цели обработки."""
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
    """Загружает конфигурацию одного оверлея."""
    data = load_json_file(base_path / "overlays" / f"{overlay}.json")
    return OverlayConfig(
        name=data["name"],
        instructions=data["instructions"],
    )

def load_overlay_configs(
    overlays: Sequence[str],
    base_path: Path = Path("config"),
) -> List[OverlayConfig]:
    """Загружает конфигурации нескольких надстроек (legacy)."""
    return [load_overlay_config(ov, base_path) for ov in overlays]

def load_output_format(
    mode: str,
    base_path: Path = Path("config"),
) -> str:
    """Загружает шаблон формата вывода."""
    data = load_json_file(base_path / "output_format.json")
    return data.get(mode, data["text_only"])

# ============================================================================
# Flatten helpers
# ============================================================================

def _flatten_examples_block(
    items: List[Dict[str, Any]],
    category: str = "",
) -> List[FlatEntry]:
    """Разворачивает список записей, каждая из которых может содержать examples."""
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
    """Разворачивает stylistic_issues.json в плоский список записей."""
    flat: List[FlatEntry] = []
    flat.extend(_flatten_examples_block(raw.get("stylistic_errors", [])))
    flat.extend(_flatten_examples_block(raw.get("common_issues", [])))
    return flat

def _flatten_editorial_techniques(raw: Dict[str, Any]) -> List[EditorialTechniqueEntry]:
    """Разворачивает editorial_techniques.json в плоский список приёмов."""
    flat: List[EditorialTechniqueEntry] = []
    for block in raw.get("editorial_techniques", []):
        if not isinstance(block, dict):
            continue
        category = block.get("category", "")
        block_tags = block.get("tags", [])
        techniques = block.get("techniques", [])
        if not isinstance(techniques, list):
            continue
        for tech in techniques:
            if not isinstance(tech, dict):
                continue
            examples = tech.get("examples", [])
            if examples and isinstance(examples, list):
                example = examples[0]
                wrong = example.get("wrong", "")
                correct = example.get("correct", "")
                explanation = example.get("explanation", "")
            else:
                wrong = correct = explanation = ""
            tags = list(block_tags) + list(tech.get("tags", []))
            flat.append({
                "id": tech.get("id", ""),
                "name": tech.get("name", ""),
                "category": category,
                "description": tech.get("description", ""),
                "when_to_use": tech.get("when_to_use", []),
                "how_to_apply": tech.get("how_to_apply", []),
                "example_wrong": wrong,
                "example_correct": correct,
                "example_explanation": explanation,
                "tags": tags or ["editing", "nora_gal"],
                "source": tech.get("source", {}),
            })
    return flat

# ============================================================================
# Knowledge Base Loader
# ============================================================================

def load_knowledge_base(base_path: Path = Path("knowledge_base")) -> KnowledgeBase:
    """Загружает базу знаний из папки knowledge_base."""
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
        grammar_errors=grammar.get("common_mistakes", []),
        stylistic_issues=_flatten_stylistic_issues(style_raw),
        logic_issues=logic_data.get("issues", []),
        storytelling_frameworks=storytelling.get("frameworks", []),
        marketing_templates=marketing.get("templates", []),
        domain_glossary=domain_glossary,
        composition_principles=composition_principles_raw.get("composition_principles", []),
        local_cohesion=local_cohesion_raw.get("local_cohesion", []),
        composition_errors=composition_errors_raw.get("composition_errors", []),
        rhetoric_frameworks=rhetoric_raw.get("frameworks", []),
        editorial_techniques=_flatten_editorial_techniques(editorial_raw) if editorial_raw else [],
        nkrj_structure_patterns=structure_data,
    )

# ============================================================================
# Утилиты
# ============================================================================

def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def build_nkrj_norms_lines(
    kb: KnowledgeBase,
    limit_sources: int = 4,
) -> List[str]:
    """Превращает nkrj_structure_patterns.json в компактный набор норм для промпта."""
    raw = kb.nkrj_structure_patterns
    if not raw:
        return []

    lines: List[str] = []
    corpus = raw.get("corpus")
    if corpus:
        lines.append(f" • Корпус-ориентир: {corpus}.")

    aggregate = raw.get("aggregate_norms", {})
    thresholds = aggregate.get("thresholds", {})
    norm_sentence = aggregate.get("norm_sentence_length", {})

    avg = _safe_float(norm_sentence.get("avg"))
    variation = _safe_float(norm_sentence.get("variation_coeff"))
    short_share = _safe_float(norm_sentence.get("short_share"))
    medium_share = _safe_float(norm_sentence.get("medium_share"))
    long_share = _safe_float(norm_sentence.get("long_share"))

    if avg is not None:
        lines.append(f" • Ориентир по длине предложения: в среднем около {avg:.2f} слов; держи фразы преимущественно короткими и средними.")
    if short_share is not None and medium_share is not None and long_share is not None:
        lines.append(f" • Распределение длины предложений: короткие ≈ {short_share:.1%}, средние ≈ {medium_share:.1%}, длинные ≈ {long_share:.1%}; не перегружай текст длинными периодами.")
    if variation is not None:
        lines.append(f" • Коэффициент вариативности длины предложений — около {variation:.2f}; избегай монотонного ритма и чередуй длину фраз.")

    flat_paragraph = _safe_float(aggregate.get("norm_flat_paragraph_share"))
    if flat_paragraph is not None:
        lines.append(f" • Плоские абзацы почти не встречаются: норма flat paragraph share ≈ {flat_paragraph:.2%}; абзацы должны двигать мысль, а не быть механически однотипными.")

    passive_rate = _safe_float(aggregate.get("norm_passive_rate"))
    if passive_rate is not None:
        lines.append(f" • Ориентир по пассиву: около {passive_rate:.2f} на 100 строк; предпочитай активные конструкции.")

    deepr_rate = _safe_float(aggregate.get("norm_deepr_rate"))
    if deepr_rate is not None:
        lines.append(f" • Глубокие шаблонные клише почти отсутствуют: норма ≈ {deepr_rate:.2f} на 100 строк; избегай формульных вводок и пластиковых связок.")

    plasticity_live = _safe_float(thresholds.get("plasticity_index_live"))
    plasticity_grey = _safe_float(thresholds.get("plasticity_index_grey_zone"))
    if plasticity_live is not None and plasticity_grey is not None:
        lines.append(f" • Индекс пластичности: до {plasticity_live:.1f} — живой текст, около {plasticity_grey:.1f} и выше — серая зона / риск искусственности.")

    sentence_variation_min = _safe_float(thresholds.get("sentence_variation_coeff_min"))
    if sentence_variation_min is not None:
        lines.append(f" • Минимально допустимая вариативность длины фраз: {sentence_variation_min:.2f}; не делай весь текст одинаково рубленым или одинаково растянутым.")

    short_sentence_share_min = _safe_float(thresholds.get("short_sentence_share_min"))
    if short_sentence_share_min is not None:
        lines.append(f" • Доля коротких предложений должна быть не ниже {short_sentence_share_min:.1%}; оставляй в тексте быстрые, простые фразы.")

    flat_alert = _safe_float(thresholds.get("flat_paragraph_share_alert"))
    if flat_alert is not None:
        lines.append(f" • Тревожный порог для плоских абзацев — {flat_alert:.0%}; если абзацы становятся однотипными, перестрой композицию.")

    passive_alert = _safe_float(thresholds.get("passive_rate_alert"))
    if passive_alert is not None:
        lines.append(f" • Тревожный порог по пассиву — {passive_alert:.1f} на 100 строк; выше этого текст становится тяжёлым и безличным.")

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
            if not isinstance(source_data, dict):
                continue
            markers = source_data.get("plastic_markers_per_1000_lines", {})
            if not isinstance(markers, dict):
                continue
            for marker, value in markers.items():
                score = _safe_float(value)
                if score is not None and score > 0:
                    marker_examples.append(f"{marker} ({score:.3f})")

    if marker_examples:
        unique_markers = list(dict.fromkeys(marker_examples))
        lines.append(" • Маркеры пластика, которые стоит особенно контролировать: " + ", ".join(unique_markers[:8]) + ".")

    return lines

def _has_mode(
    intent: Optional[str],
    overlays: Sequence[str],
    aliases: Iterable[str],
) -> bool:
    """Проверяет, активирован ли режим по intent или overlay."""
    normalized_aliases = {alias.strip().lower() for alias in aliases}
    values = {item.strip().lower() for item in overlays}
    if intent:
        values.add(intent.strip().lower())
    return bool(values & normalized_aliases)

# ============================================================================
# Централизованный маппинг на canonical теги
# ============================================================================

CANONICAL_TAGS: Dict[str, Dict[str, Any]] = {
    "domains": {
        "marketing": {"primary": ["marketing"], "expanded": ["sales", "promo", "conversion"]},
        "blog": {"primary": ["blog"], "expanded": ["non_marketing", "article", "educational"]},
        "deai": {"primary": ["deai"], "expanded": ["anti_ai", "humanize", "natural"]},
    },
    "intents": {
        "storytelling": {"primary": ["storytelling", "structure"], "expanded": ["narrative", "engagement"]},
        "noragal": {"primary": ["editing", "nora_gal"], "expanded": ["brevity", "clarity"]},
        "deai": {"primary": ["anti_ai", "humanize"], "expanded": ["authentic"]},
    },
    "overlays": {
        "logic": {"primary": ["logic"], "expanded": ["coherence", "argumentation"]},
        "factcheck": {"primary": ["factcheck"], "expanded": ["accuracy", "verification"]},
        "infostyle": {"primary": ["infostyle"], "expanded": ["clarity", "precision"]},
        "marketing_push": {"primary": ["marketing"], "expanded": ["persuasion", "cta"]},
    },
}

KNOWN_INTENTS: Set[str] = {"storytelling", "noragal", "deai", "neutral"}
KNOWN_OVERLAYS: Set[str] = {"logic", "factcheck", "infostyle", "marketing_push"}


def _get_canonical_tags_for_category(category: str, value: str) -> List[str]:
    data = CANONICAL_TAGS.get(category, {}).get(value)
    if isinstance(data, dict):
        return data.get("primary", []) + data.get("expanded", [])
    if isinstance(data, list):
        return data
    return [value]

def _get_primary_tags_for_category(category: str, value: str) -> List[str]:
    data = CANONICAL_TAGS.get(category, {}).get(value)
    if isinstance(data, dict):
        return data.get("primary", [])
    return data if isinstance(data, list) else [value]

def _get_expanded_tags_for_category(category: str, value: str) -> List[str]:
    data = CANONICAL_TAGS.get(category, {}).get(value)
    if isinstance(data, dict):
        return data.get("expanded", [])
    return []

# ============================================================================
# Prompt Builder
# ============================================================================

class PromptBuilder:
    """Собирает финальный промпт из конфигов, базы знаний и параметров запроса."""

    def __init__(
        self,
        config_path: Path = Path("config"),
        kb_path: Path = Path("knowledge_base"),
        limits: LimitsConfig = None,           # ТП-3: все 16 параметров → один объект
        enable_selection_diagnostics: bool = False,
    ) -> None:
        self.config_path = config_path
        self.kb_path = kb_path
        self.limits = limits if limits is not None else LimitsConfig()
        self.enable_selection_diagnostics = enable_selection_diagnostics

        # Кэши конфигов
        self._core_cache: Optional[CoreConfig] = None
        self._domain_cache: Dict[str, DomainConfig] = {}
        self._output_format_cache: Dict[str, str] = {}
        self._overlay_cache: Dict[str, OverlayConfig] = {}
        self._intent_cache: Dict[str, IntentConfig] = {}
        self._kb_cache: Optional[KnowledgeBase] = None
        self._available_intents_cache: Optional[Set[str]] = None
        self._available_overlays_cache: Optional[Set[str]] = None

    def reload_configs(self) -> None:
        """Сбрасывает все кэши, заставляя следующую загрузку читать с диска."""
        self._core_cache = None
        self._domain_cache.clear()
        self._output_format_cache.clear()
        self._overlay_cache.clear()
        self._intent_cache.clear()
        self._kb_cache = None
        self._available_intents_cache = None
        self._available_overlays_cache = None

    # -------------------------------------------------------------------------
    # Приватные методы с кэшированием загрузки
    # -------------------------------------------------------------------------

    def _get_core_config(self) -> CoreConfig:
        if self._core_cache is None:
            self._core_cache = load_core_config(self.config_path)
        return self._core_cache

    def _get_domain_config(self, domain: str) -> DomainConfig:
        if domain not in self._domain_cache:
            self._domain_cache[domain] = load_domain_config(domain, self.config_path)
        return self._domain_cache[domain]

    def _get_output_format(self, mode: str) -> str:
        if mode not in self._output_format_cache:
            self._output_format_cache[mode] = load_output_format(mode, self.config_path)
        return self._output_format_cache[mode]

    def _get_overlay_config(self, overlay: str) -> OverlayConfig:
        if overlay not in self._overlay_cache:
            self._overlay_cache[overlay] = load_overlay_config(overlay, self.config_path)
        return self._overlay_cache[overlay]

    def _get_intent_config(self, intent: Optional[str]) -> Optional[IntentConfig]:
        if intent is None or intent == "neutral":
            return None
        if intent not in self._intent_cache:
            cfg = load_intent_config(intent, self.config_path)
            if cfg is None:
                return None
            self._intent_cache[intent] = cfg
        return self._intent_cache[intent]

    def _get_knowledge_base(self) -> KnowledgeBase:
        if self._kb_cache is None:
            self._kb_cache = load_knowledge_base(self.kb_path)
        return self._kb_cache

    def _get_available_intents(self) -> Set[str]:
        if self._available_intents_cache is None:
            intents_dir = self.config_path / "intents"
            self._available_intents_cache = (
                {p.stem for p in intents_dir.glob("*.json")}
                if intents_dir.exists()
                else set()
            )
        return self._available_intents_cache

    def _get_available_overlays(self) -> Set[str]:
        if self._available_overlays_cache is None:
            overlays_dir = self.config_path / "overlays"
            self._available_overlays_cache = (
                {p.stem for p in overlays_dir.glob("*.json")}
                if overlays_dir.exists()
                else set()
            )
        return self._available_overlays_cache

    # -------------------------------------------------------------------------
    # Основной публичный метод
    # -------------------------------------------------------------------------

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
            parts.append(self._build_knowledge_block(text=text, domain=domain, intent=intent, overlays=overlays))
        parts.append(self._build_output_format_block(output_mode))
        parts.append(self._build_text_block(text))
        return "\n\n".join(parts)

    # -------------------------------------------------------------------------
    # Сборка отдельных блоков
    # -------------------------------------------------------------------------

    def _build_core_block(self) -> str:
        core = self._get_core_config()
        instructions = "\n".join(f"- {instr}" for instr in core.basic_audit_instructions)
        forbidden = "\n".join(f"❌ {rule}" for rule in core.forbidden)
        return f"""{core.role}

{core.priorities}

Задачи:
{instructions}

Запреты:
{forbidden}"""

    def _build_domain_block(self, domain: str) -> str:
        domain_cfg = self._get_domain_config(domain)
        return f"""Домен: {domain_cfg.system_rules}
Тон: {domain_cfg.tone}"""

    def _build_intent_block(self, intent: str) -> str:
        intent_cfg = self._get_intent_config(intent)
        if intent_cfg is None:
            return ""
        instructions = "\n".join(f"- {instr}" for instr in intent_cfg.instructions)
        return f"""Цель обработки: {intent_cfg.name}

Требования:
{instructions}"""

    def _build_audience_block(self, audience: Optional[AudienceProfile]) -> str:
        if audience is None:
            return "Аудитория: не указана. Используй нейтральный профессиональный тон."
        if not audience.description:
            kind_display = {"b2b": "B2B", "b2c": "B2C", "mixed": "смешанная", "custom": "особая"}.get(audience.kind, audience.kind)
            expertise_display = {"novice": "новички", "pro": "эксперты", "expert": "глубокие эксперты"}.get(audience.expertise, audience.expertise)
            formality_display = {"casual": "расслабленный", "neutral": "нейтральный", "formal": "официальный"}.get(audience.formality, audience.formality)
            return f"Аудитория: {kind_display}, {expertise_display}, {formality_display} тон."
        description_line = f"\n- Описание: {audience.description}" if audience.description else ""
        return (
            "Аудитория:\n"
            f"- Тип: {audience.kind}\n"
            f"- Уровень экспертизы: {audience.expertise}\n"
            f"- Формальность: {audience.formality}{description_line}"
        )

    def _build_overlays_block(self, overlays: Sequence[str]) -> str:
        parts: List[str] = ["Дополнительные режимы:"]
        for overlay in overlays:
            cfg = self._get_overlay_config(overlay)
            instructions = "\n".join(f" - {instr}" for instr in cfg.instructions)
            parts.append(f"\n• {cfg.name}:\n{instructions}")
        return "\n".join(parts)

    # -------------------------------------------------------------------------
    # Feature resolution
    # -------------------------------------------------------------------------

    def _resolve_prompt_features(
        self,
        domain_cfg: DomainConfig,
        domain: str,
        intent: Optional[str],
        overlays: Sequence[str],
    ) -> Dict[str, Any]:
        primary_tags: List[str] = []
        expanded_tags: List[str] = []

        primary_tags.extend(_get_primary_tags_for_category("domains", domain))
        expanded_tags.extend(_get_expanded_tags_for_category("domains", domain))

        available_intents = self._get_available_intents()
        available_overlays = self._get_available_overlays()

        if intent is not None:
            primary_tags.extend(_get_primary_tags_for_category("intents", intent))
            expanded_tags.extend(_get_expanded_tags_for_category("intents", intent))
            if intent not in KNOWN_INTENTS and intent not in available_intents:
                logger.warning("Unknown intent '%s' passed to PromptBuilder", intent)

        for ov in overlays:
            primary_tags.extend(_get_primary_tags_for_category("overlays", ov))
            expanded_tags.extend(_get_expanded_tags_for_category("overlays", ov))
            if ov not in KNOWN_OVERLAYS and ov not in available_overlays:
                logger.warning("Unknown overlay '%s' passed to PromptBuilder", ov)

        primary_set = {t.strip().lower() for t in primary_tags if isinstance(t, str)}
        expanded_set = {t.strip().lower() for t in expanded_tags if isinstance(t, str)}
        expanded_set -= primary_set

        storytelling_requested = _has_mode(intent, overlays, {"storytelling", "story", "narrative"})
        marketing_requested = _has_mode(intent, overlays, {"marketing_push", "marketing", "sales"})

        return {
            "tags": list(primary_set),
            "expanded_tags": list(expanded_set),
            "storytelling_enabled": domain_cfg.allow_storytelling and storytelling_requested,
            "marketing_enabled": domain_cfg.allow_marketing and (domain == "marketing" or marketing_requested),
        }

    # -------------------------------------------------------------------------
    # Helper builders для knowledge блоков (используют self.limits)
    # -------------------------------------------------------------------------

    def _build_grammar_style_logic_block(
        self, kb: KnowledgeBase, text: str, tags: List[str], expanded_tags: Set[str],
    ) -> str:
        lim = self.limits
        normalized_text = normalize_text_for_match(text)
        exp = expanded_tags if expanded_tags else None

        grammar_sample = _select_ranked_entries(kb.grammar_errors, normalized_text, tags, lim.grammar, scorer=score_rule_entry, candidate_limit=lim.grammar_candidates, debug_context="grammar", expanded_tags=exp, min_score=1)
        style_sample = _select_ranked_entries(kb.stylistic_issues, normalized_text, tags, lim.style, scorer=score_rule_entry, candidate_limit=lim.style_candidates, debug_context="style", expanded_tags=exp, min_score=1)
        logic_source: List[Dict[str, Any]] = kb.logic_issues if kb.logic_issues else kb.stylistic_issues + kb.grammar_errors
        logic_sample = _select_ranked_entries(logic_source, normalized_text, list(tags) + ["logic"], lim.logic, scorer=score_rule_entry, candidate_limit=lim.logic_candidates, debug_context="logic", expanded_tags=exp, min_score=1)

        grammar_lines = [f" • {err.get('wrong', '')} → {err.get('correct', '').strip()} ({err.get('rule', '').strip()})" for err in grammar_sample if err.get("wrong") and err.get("correct")] or [" • (нет примеров в базе)"]
        style_lines = [f" • {issue.get('wrong', '')} → {issue.get('correct', '').strip()} ({issue.get('rule', '').strip()})" for issue in style_sample if issue.get("wrong")] or [" • (нет примеров в базе)"]
        logic_lines = [f" • {item.get('name', item.get('wrong', 'Проблема'))}: {item.get('rule', item.get('description', '')).strip()}" for item in logic_sample] or [" • (нет логических правил в базе)"]

        return (
            "Типичные грамматические и лексические ошибки (исправляй по аналогии):\n" + "\n".join(grammar_lines)
            + "\n\nТипичные стилистические проблемы (канцелярит, штампы, вода — устраняй):\n" + "\n".join(style_lines)
            + "\n\nТипичные логические проблемы и риски связности:\n" + "\n".join(logic_lines)
        )

    def _build_composition_cohesion_errors_block(
        self, kb: KnowledgeBase, tags: List[str], expanded_tags: Set[str],
    ) -> str:
        lim = self.limits
        composition_sample = select_structural_by_tags_or_all(kb.composition_principles, tags=tags + ["composition"], limit=lim.composition, expanded_tags=expanded_tags, min_score=1)
        cohesion_sample = select_structural_by_tags_or_all(kb.local_cohesion, tags=tags + ["cohesion"], limit=lim.cohesion, expanded_tags=expanded_tags, min_score=1)
        errors_sample = select_structural_by_tags_or_all(kb.composition_errors, tags=tags + ["composition"], limit=lim.composition_errors, expanded_tags=expanded_tags, min_score=1)

        comp_lines = [f" • {e.get('name', '')}: {e.get('rule', e.get('description', '')).strip()}" for e in composition_sample] or [" • (нет принципов композиции в базе)"]
        coh_lines = [f" • {e.get('name', '')}: {e.get('rule', e.get('description', '')).strip()}" for e in cohesion_sample] or [" • (нет приёмов локальной связности в базе)"]
        err_lines = [f" • {e.get('name', '')}: {e.get('rule', e.get('description', '')).strip()}" for e in errors_sample] or [" • (нет примеров композиционных ошибок в базе)"]

        return (
            "Принципы композиции (типы построения и глобальная связность):\n" + "\n".join(comp_lines)
            + "\n\nПриёмы локальной связности (абзац, тема-рема, местоимения, союзы):\n" + "\n".join(coh_lines)
            + "\n\nТипичные композиционные ошибки (что искать и как исправлять):\n" + "\n".join(err_lines)
        )

    def _build_nkrj_block(self, kb: KnowledgeBase) -> str:
        nkrj_norms_lines = build_nkrj_norms_lines(kb)
        if not nkrj_norms_lines:
            return ""
        return "\n\nНормы живого текста по корпусу Taiga Social Media (используй как статистический ориентир, а не как жёсткий шаблон):\n" + "\n".join(nkrj_norms_lines)

    def _build_storytelling_block(self, kb: KnowledgeBase, text: str, tags: List[str], expanded_tags: Set[str], storytelling_enabled: bool) -> str:
        if not storytelling_enabled or not kb.storytelling_frameworks:
            return ""
        lim = self.limits
        normalized_text = normalize_text_for_match(text)
        frameworks_sample = _select_ranked_entries(kb.storytelling_frameworks, normalized_text, tags + ["storytelling"], lim.storytelling, require_text_match=False, scorer=score_structural_entry, expanded_tags=expanded_tags if expanded_tags else None, candidate_limit=lim.storytelling_candidates, debug_context="storytelling", min_score=1)
        framework_lines = [f" • {fw.get('name', '')}: " + " → ".join(step.get("name", "") for step in fw.get("steps", []) if isinstance(step, dict) and step.get("name")) for fw in frameworks_sample if fw.get("name") and fw.get("steps")]
        framework_lines = [l for l in framework_lines if l.strip(" • ")]
        return ("\n\nФреймворки сторителлинга (для структуры рассказа):\n" + "\n".join(framework_lines)) if framework_lines else ""

    def _build_marketing_block(self, kb: KnowledgeBase, text: str, tags: List[str], expanded_tags: Set[str], marketing_enabled: bool) -> str:
        if not marketing_enabled or not kb.marketing_templates:
            return ""
        lim = self.limits
        normalized_text = normalize_text_for_match(text)
        templates_sample = _select_ranked_entries(kb.marketing_templates, normalized_text, tags + ["marketing"], lim.marketing, require_text_match=False, scorer=score_structural_entry, expanded_tags=expanded_tags if expanded_tags else None, candidate_limit=lim.marketing_candidates, debug_context="marketing", min_score=1)
        template_lines = [f" • {tpl.get('name', '')}: " + ", ".join(sec.get("name", "") for sec in tpl.get("sections", []) if isinstance(sec, dict) and sec.get("name")) for tpl in templates_sample if tpl.get("name") and tpl.get("sections")]
        template_lines = [l for l in template_lines if l.strip(" • ")]
        return ("\n\nМаркетинговые шаблоны (структура текста по типу):\n" + "\n".join(template_lines)) if template_lines else ""

    def _build_rhetoric_editorial_glossary_block(
        self, kb: KnowledgeBase, domain: str, text: str, tags: List[str], expanded_tags: Set[str],
    ) -> str:
        lim = self.limits
        parts: List[str] = []

        if kb.rhetoric_frameworks:
            normalized_text = normalize_text_for_match(text)
            rhetoric_sample = _select_ranked_entries(kb.rhetoric_frameworks, normalized_text, tags + ["rhetoric"], lim.rhetoric, require_text_match=False, scorer=score_structural_entry, expanded_tags=expanded_tags if expanded_tags else None, candidate_limit=lim.rhetoric_candidates, debug_context="rhetoric", min_score=1)
            rhetoric_lines = [f" • {fw.get('name', '')}: " + " → ".join(step.get("name", "") for step in fw.get("steps", []) if isinstance(step, dict) and step.get("name")) for fw in rhetoric_sample if fw.get("name") and fw.get("steps")]
            if rhetoric_lines:
                parts.append("Риторические топосы и приёмы аргументации:\n" + "\n".join(rhetoric_lines))

        if kb.editorial_techniques:
            editorial_sample = select_structural_by_tags_or_all(kb.editorial_techniques, tags=tags + ["editing"], limit=lim.editorial, expanded_tags=expanded_tags, min_score=1)
            editorial_lines = []
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
                parts.append("Редакторские приёмы (по Норе Галь и другим редакторам):\n" + "\n".join(editorial_lines))

        if kb.domain_glossary:
            relevant_terms: Dict[str, str] = {}
            wanted_tags_set = {t.lower() for t in tags}
            normalized_text = normalize_text_for_match(text)
            for dom, dom_terms in kb.domain_glossary.items():
                if not isinstance(dom_terms, dict):
                    continue
                for term, definition in dom_terms.items():
                    if normalize_text_for_match(term) in normalized_text:
                        relevant_terms[term] = definition
                    if len(relevant_terms) >= lim.glossary:
                        break
                if len(relevant_terms) >= lim.glossary:
                    break
            if len(relevant_terms) < lim.glossary:
                for dom in [domain] + [d for d in kb.domain_glossary if d != domain]:
                    dom_terms = kb.domain_glossary.get(dom)
                    if not isinstance(dom_terms, dict):
                        continue
                    if dom == domain or dom.lower() in wanted_tags_set:
                        for term, definition in dom_terms.items():
                            if term not in relevant_terms:
                                relevant_terms[term] = definition
                            if len(relevant_terms) >= lim.glossary:
                                break
                    if len(relevant_terms) >= lim.glossary:
                        break
            if relevant_terms:
                sample_items = list(relevant_terms.items())[:lim.glossary]
                parts.append("Глоссарий (релевантные термины):\n" + "\n".join(f" • {k}: {v}" for k, v in sample_items))

        return ("\n\n" + "\n\n".join(parts)) if parts else ""

    def _build_stop_words_block(self, kb: KnowledgeBase, primary_tags: List[str]) -> str:
        lim = self.limits
        stop_words_dict = kb.stop_words
        if not isinstance(stop_words_dict, dict) or not stop_words_dict:
            return "Стоп-слова и нежелательные конструкции: (нет данных)"
        tag_set = {t.lower() for t in primary_tags if isinstance(t, str)}
        priority_categories: List[Tuple[str, List[str]]] = []
        other_categories: List[Tuple[str, List[str]]] = []
        for category, words in stop_words_dict.items():
            if not isinstance(category, str) or not isinstance(words, (list, tuple)):
                continue
            (priority_categories if category.strip().lower() in tag_set else other_categories).append((category, list(words)))
        ordered = priority_categories + sorted(other_categories, key=lambda x: x[0].lower())
        lines: List[str] = []
        for category, words in ordered[:lim.stop_words_category]:
            clean_words: List[str] = []
            seen: Set[str] = set()
            for w in words:
                if isinstance(w, str) and w.strip() and w.strip() not in seen:
                    seen.add(w.strip())
                    clean_words.append(w.strip())
            if not clean_words:
                continue
            limited = clean_words[:lim.stop_words_items]
            quoted = [f'"{w}"' for w in limited]
            if len(clean_words) > lim.stop_words_items:
                quoted.append("…")
            lines.append(f" • {category}: {', '.join(quoted)}")
        if not lines:
            return "Стоп-слова и нежелательные конструкции: (нет данных)"
        return "Стоп-слова и нежелательные конструкции (удаляй или переписывай):\n" + "\n".join(lines)

    def _build_knowledge_block(
        self, text: str, domain: str, intent: Optional[str], overlays: Sequence[str],
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
            logger.debug("Storytelling enabled: %s, Marketing enabled: %s", storytelling_enabled, marketing_enabled)

        return (
            "База знаний:\n\n"
            + self._build_stop_words_block(kb, tags) + "\n\n"
            + self._build_grammar_style_logic_block(kb, text, tags, expanded_tags) + "\n\n"
            + self._build_composition_cohesion_errors_block(kb, tags, expanded_tags)
            + self._build_nkrj_block(kb)
            + self._build_storytelling_block(kb, text, tags, expanded_tags, storytelling_enabled)
            + self._build_marketing_block(kb, text, tags, expanded_tags, marketing_enabled)
            + self._build_rhetoric_editorial_glossary_block(kb, domain, text, tags, expanded_tags)
        )

    def _build_output_format_block(self, mode: str) -> str:
        return f"Формат ответа:\n{self._get_output_format(mode)}"

    def _build_text_block(self, text: str) -> str:
        return f'Tекст для обработки:\n"""\\n{text}\n"""'

# ============================================================================
# Legacy wrapper
# ============================================================================

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
    audience_map: Dict[str, AudienceProfile] = {
        "b2b": AudienceProfile(kind="b2b", expertise="pro", formality="neutral"),
        "b2c": AudienceProfile(kind="b2c", expertise="novice", formality="casual"),
        "mixed": AudienceProfile(kind="mixed", expertise="pro", formality="neutral"),
    }
    if audience_type not in audience_map:
        logger.warning("Unknown audience_type '%s', falling back to 'b2b'", audience_type)
    audience = audience_map.get(audience_type, audience_map["b2b"])
    return _get_default_builder().build(text=text, domain=domain, intent=intent, audience=audience, overlays=overlays, output_mode=output_mode)

# ============================================================================
# Валидация конфигов и knowledge base
# ============================================================================

def _validate_stop_words_structure(stop_words: Any) -> None:
    if not isinstance(stop_words, dict):
        raise ValueError("stop_words must be a dict")
    for category, words in stop_words.items():
        if not isinstance(category, str):
            raise ValueError(f"stop_words category key must be str, got {type(category)}")
        if not isinstance(words, (list, tuple)):
            raise ValueError(f"stop_words['{category}'] must be a list or tuple, got {type(words)}")
        for i, w in enumerate(words):
            if not isinstance(w, str):
                raise ValueError(f"stop_words['{category}'][{i}] must be str, got {type(w)}")

def _validate_rule_entries(entries: List[Dict[str, Any]], name: str, sample_size: int = 5) -> None:
    if not isinstance(entries, list):
        raise ValueError(f"{name} must be a list")
    for i, entry in enumerate(entries[:sample_size]):
        if not isinstance(entry, dict):
            raise ValueError(f"{name}[{i}] must be a dict, got {type(entry)}")
        for str_key in ("wrong", "correct", "rule", "description", "name", "category"):
            if str_key in entry and not isinstance(entry[str_key], str):
                raise ValueError(f"{name}[{i}].{str_key} must be str, got {type(entry[str_key])}")
        if "tags" in entry:
            if not isinstance(entry["tags"], list):
                raise ValueError(f"{name}[{i}].tags must be a list")
            for j, tag in enumerate(entry["tags"]):
                if not isinstance(tag, str):
                    raise ValueError(f"{name}[{i}].tags[{j}] must be str")
        if not any(isinstance(entry.get(k), str) and entry.get(k, "").strip() for k in ("wrong", "rule", "description", "name")):
            raise ValueError(f"{name}[{i}] must contain non-empty 'wrong', 'rule', 'description', or 'name'")

def _validate_structural_entries(entries: List[Dict[str, Any]], name: str, sample_size: int = 5) -> None:
    if not isinstance(entries, list):
        raise ValueError(f"{name} must be a list")
    for i, entry in enumerate(entries[:sample_size]):
        if not isinstance(entry, dict):
            raise ValueError(f"{name}[{i}] must be a dict")
        if not (isinstance(entry.get("name"), str) and entry.get("name", "").strip()):
            raise ValueError(f"{name}[{i}] must have a non-empty 'name'")
        for container_key in ("steps", "sections"):
            if entry.get(container_key) is not None and not isinstance(entry[container_key], list):
                raise ValueError(f"{name}[{i}].{container_key} must be a list if present")
        if "tags" in entry:
            if not isinstance(entry["tags"], list):
                raise ValueError(f"{name}[{i}].tags must be a list")
            for j, tag in enumerate(entry["tags"]):
                if not isinstance(tag, str):
                    raise ValueError(f"{name}[{i}].tags[{j}] must be str")

def _validate_logic_entries(entries: List[Dict[str, Any]], name: str, sample_size: int = 5) -> None:
    _validate_rule_entries(entries, name, sample_size)

def validate_configs_and_kb(
    config_path: Path = Path("config"),
    kb_path: Path = Path("knowledge_base"),
) -> None:
    """Проверяет загрузку конфигов и KB. Выбрасывает исключение при критических проблемах."""
    try:
        core = load_core_config(config_path)
        if not core.role:
            raise ValueError("Core config missing role")
    except Exception as e:
        raise RuntimeError(f"Core config validation failed: {e}") from e

    domains_dir = config_path / "domains"
    try:
        domain_files = sorted(domains_dir.glob("*.json")) if domains_dir.exists() else []
        if domain_files:
            domain_cfg = load_domain_config(domain_files[0].stem, config_path)
            if not domain_cfg.system_rules:
                raise ValueError("Domain config missing system_rules")
    except Exception as e:
        raise RuntimeError(f"Domain config validation failed: {e}") from e

    intents_dir = config_path / "intents"
    try:
        intent_files = sorted(intents_dir.glob("*.json")) if intents_dir.exists() else []
        if intent_files:
            intent_cfg = load_intent_config(intent_files[0].stem, config_path)
            if intent_cfg is not None and not intent_cfg.instructions:
                raise ValueError("Intent config missing instructions")
    except Exception as e:
        raise RuntimeError(f"Intent config validation failed: {e}") from e

    overlays_dir = config_path / "overlays"
    try:
        overlay_files = sorted(overlays_dir.glob("*.json")) if overlays_dir.exists() else []
        if overlay_files:
            overlay_cfg = load_overlay_config(overlay_files[0].stem, config_path)
            if not overlay_cfg.instructions:
                raise ValueError("Overlay config missing instructions")
    except Exception as e:
        raise RuntimeError(f"Overlay config validation failed: {e}") from e

    try:
        fmt = load_output_format("text_only", config_path)
        if not isinstance(fmt, str):
            raise ValueError("Output format not a string")
    except Exception as e:
        raise RuntimeError(f"Output format validation failed: {e}") from e

    try:
        kb = load_knowledge_base(kb_path)
        _validate_stop_words_structure(kb.stop_words)
        _validate_rule_entries(kb.grammar_errors, "grammar_errors")
        _validate_rule_entries(kb.stylistic_issues, "stylistic_issues")
        if kb.logic_issues:
            _validate_logic_entries(kb.logic_issues, "logic_issues")
        for attr, attr_name in [
            (kb.storytelling_frameworks, "storytelling_frameworks"),
            (kb.marketing_templates, "marketing_templates"),
            (kb.rhetoric_frameworks, "rhetoric_frameworks"),
            (kb.composition_principles, "composition_principles"),
            (kb.local_cohesion, "local_cohesion"),
            (kb.composition_errors, "composition_errors"),
            (kb.editorial_techniques, "editorial_techniques"),
        ]:
            if attr:
                _validate_structural_entries(attr, attr_name)
    except Exception as e:
        raise RuntimeError(f"Knowledge base validation failed: {e}") from e

    dummy_text = "тестовый текст"
    dummy_tags = ["marketing", "test"]
    try:
        _ = select_grammar_rules(kb, dummy_text, dummy_tags, limit=1)
        _ = select_style_issues(kb, dummy_text, dummy_tags, limit=1)
        _ = select_logic_issues(kb, dummy_text, dummy_tags, limit=1)
        _ = select_structural_by_tags_or_all(kb.composition_principles, dummy_tags, limit=1)
    except Exception as e:
        raise RuntimeError(f"Knowledge selectors smoke test failed: {e}") from e

    try:
        _ = build_nkrj_norms_lines(kb)
    except Exception as e:
        raise RuntimeError(f"NKRJ validation failed: {e}") from e

    logger.info("Config and knowledge base validation passed successfully.")
