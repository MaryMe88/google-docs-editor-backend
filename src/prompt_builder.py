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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

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
    grammar_errors: List[Dict[str, Any]]
    stylistic_issues: List[Dict[str, Any]]
    logic_issues: List[Dict[str, Any]]
    storytelling_frameworks: List[Dict[str, Any]]
    marketing_templates: List[Dict[str, Any]]
    domain_glossary: Dict[str, Any]
    composition_principles: List[Dict[str, Any]]
    local_cohesion: List[Dict[str, Any]]
    composition_errors: List[Dict[str, Any]]
    rhetoric_frameworks: List[Dict[str, Any]]
    editorial_techniques: List[Dict[str, Any]]
    nkrj_structure_patterns: Dict[str, Any]


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


def load_overlay_configs(
    overlays: Sequence[str],
    base_path: Path = Path("config"),
) -> List[OverlayConfig]:
    """Загружает конфигурации надстроек."""
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
    """Загружает шаблон формата вывода."""
    data = load_json_file(base_path / "output_format.json")
    return data.get(mode, data["text_only"])


def _flatten_stylistic_issues(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Разворачивает stylistic_issues.json в плоский список записей с полями
    wrong / correct / rule / tags / category.

    Поддерживает два формата:
    - Новый категорийный:
      {"stylistic_errors": [{category, description, examples: [{wrong, correct, rule, tags}]}]}
    - Старый плоский (обратная совместимость):
      {"common_issues": [{wrong, correct, rule, tags}]}
    """
    flat: List[Dict[str, Any]] = []

    for item in raw.get("stylistic_errors", []):
        if "examples" in item:
            category = item.get("category", "")
            for example in item["examples"]:
                entry = dict(example)
                if "tags" not in entry:
                    entry["tags"] = ["style"]
                entry["category"] = category
                flat.append(entry)
        else:
            flat.append(item)

    for item in raw.get("common_issues", []):
        if "examples" in item:
            category = item.get("category", "")
            for example in item["examples"]:
                entry = dict(example)
                if "tags" not in entry:
                    entry["tags"] = ["style"]
                entry["category"] = category
                flat.append(entry)
        else:
            flat.append(item)

    return flat


def _flatten_editorial_techniques(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Разворачивает editorial_techniques.json в плоский список приёмов.

    Ожидаемый формат:
    {
      "editorial_techniques": [
        {
          "category": "...",
          "description": "...",
          "tags": [...],
          "techniques": [
            {
              "id": "...",
              "name": "...",
              "description": "...",
              "when_to_use": [...],
              "how_to_apply": [...],
              "examples": [
                {"wrong": "...", "correct": "...", "explanation": "..."}
              ],
              "tags": [...],
              ...
            }
          ]
        }
      ]
    }

    Выходная запись:
    {
      "id": str,
      "name": str,
      "category": str,
      "description": str,
      "when_to_use": List[str],
      "how_to_apply": List[str],
      "example_wrong": str,
      "example_correct": str,
      "example_explanation": str,
      "tags": List[str],
      "source": Dict[str, Any]
    }
    """
    flat: List[Dict[str, Any]] = []

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
                    "tags": tags or ["editing", "nora_gal"],
                    "source": source,
                }
            )

    return flat


def load_knowledge_base(base_path: Path = Path("knowledge_base")) -> KnowledgeBase:
    """
    Загружает базу знаний из папки knowledge_base.

    Ожидаемые файлы:
    - stop_words.json
    - grammar_errors.json
    - stylistic_issues.json
    - storytelling_frameworks.json
    - marketing_templates.json
    - logic_issues.json (опционально)
    - domain_glossary.json (опционально)
    - composition_principles.json (опционально)
    - local_cohesion.json (опционально)
    - composition_errors.json (опционально)
    - rhetoric.json (опционально, риторические топосы)
    - editorial_techniques.json (опционально, редакторские приёмы)
    - nkrj_structure_patterns.json (опционально, структуры по НКРЯ)
    """
    stop_words = load_json_file(base_path / "stop_words.json")
    grammar = load_json_file(base_path / "grammar_errors.json")
    style_raw = load_json_file(base_path / "stylistic_issues.json")
    storytelling = load_json_file(base_path / "storytelling_frameworks.json")
    marketing = load_json_file(base_path / "marketing_templates.json")

    logic_path = base_path / "logic_issues.json"
    logic_data: Dict[str, Any] = {"issues": []}
    if logic_path.exists():
        logic_data = load_json_file(logic_path)

    glossary_path = base_path / "domain_glossary.json"
    domain_glossary: Dict[str, Any] = {}
    if glossary_path.exists():
        domain_glossary = load_json_file(glossary_path)

    composition_principles_path = base_path / "composition_principles.json"
    composition_principles: List[Dict[str, Any]] = []
    if composition_principles_path.exists():
        composition_principles_data = load_json_file(composition_principles_path)
        composition_principles = composition_principles_data.get(
            "composition_principles",
            [],
        )

    local_cohesion_path = base_path / "local_cohesion.json"
    local_cohesion: List[Dict[str, Any]] = []
    if local_cohesion_path.exists():
        local_cohesion_data = load_json_file(local_cohesion_path)
        local_cohesion = local_cohesion_data.get("local_cohesion", [])

    composition_errors_path = base_path / "composition_errors.json"
    composition_errors: List[Dict[str, Any]] = []
    if composition_errors_path.exists():
        composition_errors_data = load_json_file(composition_errors_path)
        composition_errors = composition_errors_data.get("composition_errors", [])

    rhetoric_path = base_path / "rhetoric.json"
    rhetoric_frameworks: List[Dict[str, Any]] = []
    if rhetoric_path.exists():
        rhetoric_data = load_json_file(rhetoric_path)
        rhetoric_frameworks = rhetoric_data.get("frameworks", [])

    editorial_path = base_path / "editorial_techniques.json"
    editorial_techniques: List[Dict[str, Any]] = []
    if editorial_path.exists():
        editorial_raw = load_json_file(editorial_path)
        editorial_techniques = _flatten_editorial_techniques(editorial_raw)

    structure_path = base_path / "nkrj_structure_patterns.json"
    nkrj_structure_patterns: Dict[str, Any] = {}
    if structure_path.exists():
        nkrj_structure_patterns = load_json_file(structure_path)

    return KnowledgeBase(
        stop_words=stop_words,
        grammar_errors=grammar.get("common_mistakes", []),
        stylistic_issues=_flatten_stylistic_issues(style_raw),
        logic_issues=logic_data.get("issues", []),
        storytelling_frameworks=storytelling.get("frameworks", []),
        marketing_templates=marketing.get("templates", []),
        domain_glossary=domain_glossary,
        composition_principles=composition_principles,
        local_cohesion=local_cohesion,
        composition_errors=composition_errors,
        rhetoric_frameworks=rhetoric_frameworks,
        editorial_techniques=editorial_techniques,
        nkrj_structure_patterns=nkrj_structure_patterns,
    )


# ============================================================================
# Knowledge selection helpers
# ============================================================================


def _match_tags(entry_tags: Iterable[str], wanted_tags: Iterable[str]) -> bool:
    entry = {t.strip().lower() for t in (entry_tags or [])}
    wanted = {t.strip().lower() for t in (wanted_tags or [])}
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
    Если есть отдельный logic_issues.json — используем его.
    Если нет, падаем обратно на stylistic_issues + grammar_errors с тегом "logic".
    """
    text_lower = text.lower()
    wanted_tags = list(tags) + ["logic"]

    candidates: List[Dict[str, Any]]
    if kb.logic_issues:
        candidates = kb.logic_issues
    else:
        candidates = kb.stylistic_issues + kb.grammar_errors

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


def _select_by_tags_or_all(
    entries: List[Dict[str, Any]],
    tags: Iterable[str],
    limit: int,
) -> List[Dict[str, Any]]:
    """Универсальный селектор по тегам для новых блоков знаний."""
    wanted_tags = list(tags)

    if not entries:
        return []

    matches: List[Dict[str, Any]] = []
    for entry in entries:
        entry_tags = entry.get("tags", [])
        if _match_tags(entry_tags, wanted_tags):
            matches.append(entry)
        if len(matches) >= limit:
            break

    if matches:
        return matches

    return entries[:limit]


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
    """
    Превращает nkrj_structure_patterns.json формата Taiga Social Media
    в компактный набор норм для промпта.
    """
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
            " • Ориентир по длине предложения: в среднем около "
            f"{avg:.2f} слов; держи фразы преимущественно короткими и средними."
        )

    if (
        short_share is not None
        and medium_share is not None
        and long_share is not None
    ):
        lines.append(
            " • Распределение длины предложений: "
            f"короткие ≈ {short_share:.1%}, "
            f"средние ≈ {medium_share:.1%}, "
            f"длинные ≈ {long_share:.1%}; "
            "не перегружай текст длинными периодами."
        )

    if variation is not None:
        lines.append(
            " • Коэффициент вариативности длины предложений — около "
            f"{variation:.2f}; избегай монотонного ритма и чередуй длину фраз."
        )

    flat_paragraph = _safe_float(aggregate.get("norm_flat_paragraph_share"))
    if flat_paragraph is not None:
        lines.append(
            " • Плоские абзацы почти не встречаются: "
            f"норма flat paragraph share ≈ {flat_paragraph:.2%}; "
            "абзацы должны двигать мысль, а не быть механически однотипными."
        )

    passive_rate = _safe_float(aggregate.get("norm_passive_rate"))
    if passive_rate is not None:
        lines.append(
            " • Ориентир по пассиву: около "
            f"{passive_rate:.2f} на 100 строк; предпочитай активные конструкции."
        )

    deepr_rate = _safe_float(aggregate.get("norm_deepr_rate"))
    if deepr_rate is not None:
        lines.append(
            " • Глубокие шаблонные клише почти отсутствуют: "
            f"норма ≈ {deepr_rate:.2f} на 100 строк; "
            "избегай формульных вводок и пластиковых связок."
        )

    plasticity_live = _safe_float(thresholds.get("plasticity_index_live"))
    plasticity_grey = _safe_float(thresholds.get("plasticity_index_grey_zone"))
    if plasticity_live is not None and plasticity_grey is not None:
        lines.append(
            " • Индекс пластичности: "
            f"до {plasticity_live:.1f} — живой текст, "
            f"около {plasticity_grey:.1f} и выше — серая зона / риск искусственности."
        )

    sentence_variation_min = _safe_float(thresholds.get("sentence_variation_coeff_min"))
    if sentence_variation_min is not None:
        lines.append(
            " • Минимально допустимая вариативность длины фраз: "
            f"{sentence_variation_min:.2f}; "
            "не делай весь текст одинаково рубленым или одинаково растянутым."
        )

    short_sentence_share_min = _safe_float(thresholds.get("short_sentence_share_min"))
    if short_sentence_share_min is not None:
        lines.append(
            " • Доля коротких предложений должна быть не ниже "
            f"{short_sentence_share_min:.1%}; "
            "оставляй в тексте быстрые, простые фразы."
        )

    flat_alert = _safe_float(thresholds.get("flat_paragraph_share_alert"))
    if flat_alert is not None:
        lines.append(
            " • Тревожный порог для плоских абзацев — "
            f"{flat_alert:.0%}; "
            "если абзацы становятся однотипными, перестрой композицию."
        )

    passive_alert = _safe_float(thresholds.get("passive_rate_alert"))
    if passive_alert is not None:
        lines.append(
            " • Тревожный порог по пассиву — "
            f"{passive_alert:.1f} на 100 строк; "
            "выше этого текст становится тяжёлым и безличным."
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
                line = (
                    f" • Источник {source_name}: средняя длина предложения "
                    f"≈ {source_avg:.2f} слов"
                )
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
    """
    Проверяет, активирован ли режим по intent или overlay.
    Поддерживает алиасы, чтобы не зависеть от точного имени режима.
    """
    normalized_aliases = {alias.strip().lower() for alias in aliases}
    values = {item.strip().lower() for item in overlays}

    if intent:
        values.add(intent.strip().lower())

    return bool(values & normalized_aliases)


def _extend_tags_with_feature_aliases(
    tags: List[str],
    intent: Optional[str],
    overlays: Sequence[str],
) -> List[str]:
    """
    Расширяет набор тегов служебными алиасами для knowledge base.
    Например, режим deai -> тег anti_ai.
    """
    result = list(tags)

    if _has_mode(intent, overlays, {"deai", "anti_ai", "anti-llm", "humanize"}):
        result.append("anti_ai")

    if _has_mode(intent, overlays, {"storytelling", "story", "narrative"}):
        result.append("storytelling")

    if _has_mode(intent, overlays, {"marketing_push", "marketing", "sales"}):
        result.append("marketing")

    return list(dict.fromkeys(result))


# ============================================================================
# Централизованный маппинг на canonical теги
# ============================================================================

# Источник истины: какие теги соответствуют доменам / интентам / оверлеям.
# Если значение отсутствует в словаре, оно добавляется как тег «как есть».
CANONICAL_TAGS = {
    "domains": {
        "marketing": ["marketing"],
        "blog": ["blog", "non_marketing"],
        # дополнительные домены могут быть добавлены здесь
    },
    "intents": {
        "storytelling": ["storytelling", "structure"],
        "noragal": ["editing", "nora_gal"],
        "deai": ["anti_ai", "humanize"],
        # дополнительные интенты
    },
    "overlays": {
        "logic": ["logic"],
        "factcheck": ["factcheck"],
        "infostyle": ["infostyle"],
        # дополнительные оверлеи
    },
}

# Известные интенты и оверлеи (для диагностики)
KNOWN_INTENTS: Set[str] = {
    "storytelling", "noragal", "deai", "neutral",
}
KNOWN_OVERLAYS: Set[str] = {
    "logic", "factcheck", "infostyle", "marketing_push",
}


def _get_canonical_tags_for_category(
    category: str,
    value: str,
) -> List[str]:
    """Возвращает список канонических тегов для значения из указанной категории."""
    category_map = CANONICAL_TAGS.get(category, {})
    return category_map.get(value, [value])


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
        # Лимиты для knowledge блоков (управление полнотой)
        grammar_limit: int = 10,
        style_limit: int = 10,
        logic_limit: int = 8,
        composition_limit: int = 6,
        cohesion_limit: int = 6,
        composition_errors_limit: int = 6,
        storytelling_limit: int = 4,
        marketing_limit: int = 4,
        rhetoric_limit: int = 4,
        editorial_limit: int = 6,
        glossary_limit: int = 10,
    ) -> None:
        self.config_path = config_path
        self.kb_path = kb_path

        # Лимиты
        self.grammar_limit = grammar_limit
        self.style_limit = style_limit
        self.logic_limit = logic_limit
        self.composition_limit = composition_limit
        self.cohesion_limit = cohesion_limit
        self.composition_errors_limit = composition_errors_limit
        self.storytelling_limit = storytelling_limit
        self.marketing_limit = marketing_limit
        self.rhetoric_limit = rhetoric_limit
        self.editorial_limit = editorial_limit
        self.glossary_limit = glossary_limit

        # Кэши
        self._core_cache: Optional[CoreConfig] = None
        self._domain_cache: Dict[str, DomainConfig] = {}
        self._output_format_cache: Dict[str, str] = {}
        self._overlay_cache: Dict[str, OverlayConfig] = {}
        self._kb_cache: Optional[KnowledgeBase] = None

    def reload_configs(self) -> None:
        """Сбрасывает все кэши, заставляя следующую загрузку читать с диска."""
        self._core_cache = None
        self._domain_cache.clear()
        self._output_format_cache.clear()
        self._overlay_cache.clear()
        self._kb_cache = None

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
            data = load_json_file(self.config_path / "overlays" / f"{overlay}.json")
            self._overlay_cache[overlay] = OverlayConfig(
                name=data["name"],
                instructions=data["instructions"],
            )
        return self._overlay_cache[overlay]

    def _get_knowledge_base(self) -> KnowledgeBase:
        if self._kb_cache is None:
            self._kb_cache = load_knowledge_base(self.kb_path)
        return self._kb_cache

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

    # -------------------------------------------------------------------------
    # Сборка отдельных блоков
    # -------------------------------------------------------------------------
    def _build_core_block(self) -> str:
        core = self._get_core_config()

        instructions = "\n".join(
            f"- {instr}" for instr in core.basic_audit_instructions
        )
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
        intent_cfg = load_intent_config(intent, self.config_path)
        if intent_cfg is None:
            return ""

        instructions = "\n".join(f"- {instr}" for instr in intent_cfg.instructions)
        return f"""Цель обработки: {intent_cfg.name}

Требования:
{instructions}"""

    def _build_audience_block(self, audience: Optional[AudienceProfile]) -> str:
        if audience is None:
            return (
                "Аудитория: не указана. "
                "Используй нейтральный профессиональный тон."
            )

        # Для стандартных профилей без кастомного описания – компактный формат
        if not audience.description:
            kind_display = {
                "b2b": "B2B",
                "b2c": "B2C",
                "mixed": "смешанная",
                "custom": "особая",
            }.get(audience.kind, audience.kind)

            expertise_display = {
                "novice": "новички",
                "pro": "эксперты",
                "expert": "глубокие эксперты",
            }.get(audience.expertise, audience.expertise)

            formality_display = {
                "casual": "расслабленный",
                "neutral": "нейтральный",
                "formal": "официальный",
            }.get(audience.formality, audience.formality)

            return (
                f"Аудитория: {kind_display}, "
                f"{expertise_display}, "
                f"{formality_display} тон."
            )

        # Полный формат для кастомных профилей
        description_line = (
            f"\n- Описание: {audience.description}"
            if audience.description
            else ""
        )
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
    # Feature resolution (централизованное принятие решений)
    # -------------------------------------------------------------------------
    @staticmethod
    def _resolve_prompt_features(
        domain_cfg: DomainConfig,
        domain: str,
        intent: Optional[str],
        overlays: Sequence[str],
    ) -> Dict[str, Any]:
        """
        Централизованно определяет:
        - итоговый список тегов для knowledge base
        - включён ли storytelling (с учётом allow_storytelling домена)
        - включён ли marketing (с учётом allow_marketing домена)
        """
        # Базовые теги из канонического маппинга
        base_tags: List[str] = []
        base_tags.extend(_get_canonical_tags_for_category("domains", domain))
        if intent:
            base_tags.extend(_get_canonical_tags_for_category("intents", intent))
            if intent not in KNOWN_INTENTS:
                logging.warning(f"Unknown intent '{intent}' passed to PromptBuilder")
        for ov in overlays:
            base_tags.extend(_get_canonical_tags_for_category("overlays", ov))
            if ov not in KNOWN_OVERLAYS:
                logging.warning(f"Unknown overlay '{ov}' passed to PromptBuilder")

        # Расширение алиасами
        tags = _extend_tags_with_feature_aliases(base_tags, intent, overlays)

        storytelling_requested = _has_mode(
            intent, overlays, {"storytelling", "story", "narrative"}
        )
        marketing_requested = _has_mode(
            intent, overlays, {"marketing_push", "marketing", "sales"}
        )

        return {
            "tags": tags,
            "storytelling_enabled": (
                domain_cfg.allow_storytelling and storytelling_requested
            ),
            "marketing_enabled": (
                domain_cfg.allow_marketing
                and (domain == "marketing" or marketing_requested)
            ),
        }

    # -------------------------------------------------------------------------
    # Helper builders для отдельных блоков knowledge
    # -------------------------------------------------------------------------
    def _build_grammar_style_logic_block(
        self,
        kb: KnowledgeBase,
        text: str,
        tags: List[str],
    ) -> str:
        """Грамматика, стилистика и логика."""
        grammar_sample = select_grammar_rules(
            kb, text=text, tags=tags, limit=self.grammar_limit
        )
        style_sample = select_style_issues(
            kb, text=text, tags=tags, limit=self.style_limit
        )
        logic_sample = select_logic_issues(
            kb, text=text, tags=tags, limit=self.logic_limit
        )

        grammar_lines: List[str] = [
            (
                f" • {err.get('wrong', '')} → "
                f"{err.get('correct', '').strip()} "
                f"({err.get('rule', '').strip()})"
            )
            for err in grammar_sample
            if err.get("wrong") and err.get("correct")
        ] or [" • (нет примеров в базе)"]

        style_lines: List[str] = [
            (
                f" • {issue.get('wrong', '')} → "
                f"{issue.get('correct', '').strip()} "
                f"({issue.get('rule', '').strip()})"
            )
            for issue in style_sample
            if issue.get("wrong")
        ] or [" • (нет примеров в базе)"]

        logic_lines: List[str] = [
            (
                f" • {item.get('name', item.get('wrong', 'Проблема'))}: "
                f"{item.get('rule', item.get('description', '')).strip()}"
            )
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
    ) -> str:
        """Композиция, локальная связность, композиционные ошибки."""
        composition_principles_sample = _select_by_tags_or_all(
            kb.composition_principles,
            tags=tags + ["composition"],
            limit=self.composition_limit,
        )
        composition_principles_lines: List[str] = [
            (
                f" • {entry.get('name', '')}: "
                f"{entry.get('rule', entry.get('description', '')).strip()}"
            )
            for entry in composition_principles_sample
        ] or [" • (нет принципов композиции в базе)"]

        local_cohesion_sample = _select_by_tags_or_all(
            kb.local_cohesion,
            tags=tags + ["cohesion"],
            limit=self.cohesion_limit,
        )
        local_cohesion_lines: List[str] = [
            (
                f" • {entry.get('name', '')}: "
                f"{entry.get('rule', entry.get('description', '')).strip()}"
            )
            for entry in local_cohesion_sample
        ] or [" • (нет приёмов локальной связности в базе)"]

        composition_errors_sample = _select_by_tags_or_all(
            kb.composition_errors,
            tags=tags + ["composition"],
            limit=self.composition_errors_limit,
        )
        composition_errors_lines: List[str] = [
            (
                f" • {entry.get('name', '')}: "
                f"{entry.get('rule', entry.get('description', '')).strip()}"
            )
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

    def _build_nkrj_block(self, kb: KnowledgeBase) -> str:
        """Блок норм НКРЯ, если есть."""
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
        storytelling_enabled: bool,
    ) -> str:
        """Фреймворки сторителлинга, если разрешено."""
        if not storytelling_enabled or not kb.storytelling_frameworks:
            return ""

        frameworks_sample = kb.storytelling_frameworks[: self.storytelling_limit]
        framework_lines: List[str] = []

        for fw in frameworks_sample:
            name = fw.get("name", "")
            steps = fw.get("steps", [])
            step_names = [
                step.get("name", "")
                for step in steps
                if isinstance(step, dict) and step.get("name")
            ]
            if not name or not step_names:
                continue

            framework_lines.append(f" • {name}: " + " → ".join(step_names))

        if not framework_lines:
            return ""

        return (
            "\n\nФреймворки сторителлинга (для структуры рассказа):\n"
            + "\n".join(framework_lines)
        )

    def _build_marketing_block(
        self,
        kb: KnowledgeBase,
        marketing_enabled: bool,
    ) -> str:
        """Маркетинговые шаблоны, если разрешено."""
        if not marketing_enabled or not kb.marketing_templates:
            return ""

        templates_sample = kb.marketing_templates[: self.marketing_limit]
        template_lines: List[str] = []

        for tpl in templates_sample:
            name = tpl.get("name", "")
            sections = tpl.get("sections", [])
            section_names = [
                sec.get("name", "")
                for sec in sections
                if isinstance(sec, dict) and sec.get("name")
            ]
            if not name or not section_names:
                continue

            template_lines.append(f" • {name}: " + ", ".join(section_names))

        if not template_lines:
            return ""

        return (
            "\n\nМаркетинговые шаблоны (структура текста по типу):\n"
            + "\n".join(template_lines)
        )

    def _build_rhetoric_editorial_glossary_block(
        self,
        kb: KnowledgeBase,
        domain: str,
        tags: List[str],
    ) -> str:
        """Риторика, редакторские приёмы и глоссарий."""
        parts: List[str] = []

        # Риторика
        if kb.rhetoric_frameworks:
            rhetoric_sample = kb.rhetoric_frameworks[: self.rhetoric_limit]
            rhetoric_lines: List[str] = []
            for fw in rhetoric_sample:
                name = fw.get("name", "")
                steps = fw.get("steps", [])
                step_names = [
                    step.get("name", "")
                    for step in steps
                    if isinstance(step, dict) and step.get("name")
                ]
                if name and step_names:
                    rhetoric_lines.append(f" • {name}: " + " → ".join(step_names))

            if rhetoric_lines:
                parts.append(
                    "Риторические топосы и приёмы аргументации:\n"
                    + "\n".join(rhetoric_lines)
                )

        # Редакторские приёмы
        if kb.editorial_techniques:
            editorial_sample = _select_by_tags_or_all(
                kb.editorial_techniques,
                tags=tags + ["editing"],
                limit=self.editorial_limit,
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

        # Глоссарий
        if kb.domain_glossary and domain in kb.domain_glossary:
            terms = kb.domain_glossary.get(domain, {})
            if isinstance(terms, dict) and terms:
                sample_items = list(terms.items())[: self.glossary_limit]
                term_lines = [f" • {key}: {value}" for key, value in sample_items]
                parts.append(
                    "Глоссарий по домену (ключевые термины):\n"
                    + "\n".join(term_lines)
                )

        if not parts:
            return ""

        return "\n\n" + "\n\n".join(parts)

    # -------------------------------------------------------------------------
    # Основной метод сборки knowledge блока (упрощённый)
    # -------------------------------------------------------------------------
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
        storytelling_enabled = features["storytelling_enabled"]
        marketing_enabled = features["marketing_enabled"]

        stop_words_text = json.dumps(kb.stop_words, ensure_ascii=False, indent=2)

        grammar_style_logic = self._build_grammar_style_logic_block(kb, text, tags)
        composition_cohesion = self._build_composition_cohesion_errors_block(kb, tags)
        nkrj_block = self._build_nkrj_block(kb)
        storytelling_block = self._build_storytelling_block(kb, storytelling_enabled)
        marketing_block = self._build_marketing_block(kb, marketing_enabled)
        rhetoric_editorial_glossary = self._build_rhetoric_editorial_glossary_block(
            kb, domain, tags
        )

        return (
            "База знаний:\n\n"
            "Стоп-слова и нежелательные конструкции (удаляй или переписывай):\n"
            f"{stop_words_text}\n\n"
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


# ============================================================================
# Legacy wrapper (обратная совместимость)
# ============================================================================
def build_prompt(
    text: str,
    domain: str = "marketing",
    intent: Optional[str] = None,
    audience_type: str = "b2b",
    overlays: Sequence[str] = (),
    output_mode: str = "text_only",
) -> str:
    audience_map = {
        "b2b": AudienceProfile(
            kind="b2b",
            expertise="pro",
            formality="neutral",
        ),
        "b2c": AudienceProfile(
            kind="b2c",
            expertise="novice",
            formality="casual",
        ),
        "mixed": AudienceProfile(
            kind="mixed",
            expertise="pro",
            formality="neutral",
        ),
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


# ============================================================================
# Валидация конфигов и knowledge base (для вызова при старте приложения)
# ============================================================================
def validate_configs_and_kb(
    config_path: Path = Path("config"),
    kb_path: Path = Path("knowledge_base"),
) -> None:
    """
    Проверяет загрузку конфигов и KB, а также базовую работоспособность селекторов.
    Выбрасывает исключение при обнаружении критических проблем.

    Использует первые доступные файлы из папок domains/, intents/, overlays/.
    Если папки пусты — валидация соответствующей части пропускается.
    """
    # 1. Core config
    try:
        core = load_core_config(config_path)
        assert core.role, "Core config missing role"
    except Exception as e:
        raise RuntimeError(f"Core config validation failed: {e}") from e

    # 2. Domain config – берём первый попавшийся JSON
    domains_dir = config_path / "domains"
    try:
        domain_files = list(domains_dir.glob("*.json")) if domains_dir.exists() else []
        if domain_files:
            first_domain = domain_files[0].stem  # имя файла без расширения
            domain_cfg = load_domain_config(first_domain, config_path)
            assert domain_cfg.system_rules, "Domain config missing system_rules"
    except Exception as e:
        raise RuntimeError(f"Domain config validation failed: {e}") from e

    # 3. Intent config – первый попавшийся
    intents_dir = config_path / "intents"
    try:
        intent_files = list(intents_dir.glob("*.json")) if intents_dir.exists() else []
        if intent_files:
            first_intent = intent_files[0].stem
            intent_cfg = load_intent_config(first_intent, config_path)
            if intent_cfg is not None:
                assert intent_cfg.instructions, "Intent config missing instructions"
    except Exception as e:
        raise RuntimeError(f"Intent config validation failed: {e}") from e

    # 4. Overlay config – первый попавшийся
    overlays_dir = config_path / "overlays"
    try:
        overlay_files = list(overlays_dir.glob("*.json")) if overlays_dir.exists() else []
        if overlay_files:
            first_overlay = overlay_files[0].stem
            overlay_cfgs = load_overlay_configs([first_overlay], config_path)
            assert overlay_cfgs, "Overlay config missing"
    except Exception as e:
        raise RuntimeError(f"Overlay config validation failed: {e}") from e

    # 5. Output format
    try:
        fmt = load_output_format("text_only", config_path)
        assert isinstance(fmt, str), "Output format not a string"
    except Exception as e:
        raise RuntimeError(f"Output format validation failed: {e}") from e

    # 6. Knowledge base – загружаем и проверяем типы (не обязательную непустоту)
    try:
        kb = load_knowledge_base(kb_path)
        assert isinstance(kb.stop_words, dict), "KB stop_words is not a dict"
        assert isinstance(kb.grammar_errors, list), "KB grammar_errors is not a list"
        assert isinstance(kb.stylistic_issues, list), "KB stylistic_issues is not a list"
    except Exception as e:
        raise RuntimeError(f"Knowledge base validation failed: {e}") from e

    # 7. Smoke-тесты селекторов на dummy-данных
    dummy_text = "тестовый текст"
    dummy_tags = ["marketing", "test"]

    try:
        _ = select_grammar_rules(kb, dummy_text, dummy_tags, limit=1)
        _ = select_style_issues(kb, dummy_text, dummy_tags, limit=1)
        _ = select_logic_issues(kb, dummy_text, dummy_tags, limit=1)
        _ = _select_by_tags_or_all(kb.composition_principles, dummy_tags, limit=1)
    except Exception as e:
        raise RuntimeError(f"Knowledge selectors smoke test failed: {e}") from e

    # 8. NKRJ блок (опционально, но с отловом ошибок)
    try:
        _ = build_nkrj_norms_lines(kb)
    except Exception as e:
        raise RuntimeError(f"NKRJ validation failed: {e}") from e

    logging.info("Config and knowledge base validation passed successfully.")