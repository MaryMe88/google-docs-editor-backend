from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from src.prompt_builder import (
    KnowledgeBase,
    PromptBuilder,
    _flatten_editorial_techniques,
    _flatten_examples_block,
    _flatten_stylistic_issues,
    _normalize_text_for_match,
    _score_rule_entry,
    _score_structural_entry,
    _select_ranked_entries,
    load_knowledge_base,
    select_grammar_rules,
    select_logic_issues,
    select_style_issues,
    validate_configs_and_kb,
)

# ----------------------------------------------------------------------
# Фикстуры
# ----------------------------------------------------------------------

@pytest.fixture
def minimal_config(tmp_path: Path) -> Path:
    """Создаёт минимальную валидную структуру config/."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "core.json").write_text(json.dumps({
        "role": "Test Editor", "priorities": "Test priorities",
        "basic_audit_instructions": ["Check grammar", "Check style"],
        "forbidden": ["No hate speech"]
    }, ensure_ascii=False))
    (config_dir / "domains").mkdir()
    (config_dir / "domains" / "marketing.json").write_text(json.dumps({
        "name": "Маркетинг", "system_rules": "Текст для привлечения клиентов",
        "tone": "Убедительный", "allow_storytelling": True, "allow_marketing": True
    }, ensure_ascii=False))
    (config_dir / "intents").mkdir()
    (config_dir / "intents" / "storytelling.json").write_text(json.dumps({
        "name": "Сторителлинг", "instructions": ["Используй структуру истории"]
    }, ensure_ascii=False))
    (config_dir / "overlays").mkdir()
    (config_dir / "overlays" / "logic.json").write_text(json.dumps({
        "name": "Логика", "instructions": ["Проверь логические связи"]
    }, ensure_ascii=False))
    # marketing_push.json намеренно отсутствует – будет использован для fail‑fast теста
    (config_dir / "output_format.json").write_text(json.dumps({"text_only": "Только текст"}, ensure_ascii=False))
    return config_dir


@pytest.fixture
def minimal_kb(tmp_path: Path) -> Path:
    """Создаёт минимальную валидную структуру knowledge_base/."""
    kb_dir = tmp_path / "knowledge_base"
    kb_dir.mkdir()
    (kb_dir / "stop_words.json").write_text(json.dumps({
        "канцелярит": ["осуществлять", "в рамках"],
        "усилители": ["максимально", "достаточно"]
    }, ensure_ascii=False))
    (kb_dir / "grammar_errors.json").write_text(json.dumps({
        "common_mistakes": [
            {"wrong": "ихний", "correct": "их", "rule": "Притяжательное местоимение", "tags": ["grammar"]}
        ]
    }, ensure_ascii=False))
    (kb_dir / "stylistic_issues.json").write_text(json.dumps({
        "stylistic_errors": [
            {
                "category": "канцелярит",
                "description": "Канцелярские обороты",
                "examples": [
                    {"wrong": "в целях", "correct": "чтобы", "rule": "Избегай канцелярита", "tags": ["style"]}
                ]
            }
        ]
    }, ensure_ascii=False))
    (kb_dir / "storytelling_frameworks.json").write_text(json.dumps({
        "frameworks": [
            {"name": "Путь героя", "steps": [{"name": "Обычный мир"}, {"name": "Зов к приключению"}]}
        ]
    }, ensure_ascii=False))
    (kb_dir / "marketing_templates.json").write_text(json.dumps({
        "templates": [
            {"name": "AIDA", "sections": [{"name": "Attention"}, {"name": "Interest"}]}
        ]
    }, ensure_ascii=False))
    (kb_dir / "logic_issues.json").write_text(json.dumps({
        "issues": [{"name": "Ложная дилемма", "description": "Только два варианта", "tags": ["logic"]}]
    }, ensure_ascii=False))
    (kb_dir / "composition_principles.json").write_text(json.dumps({
        "composition_principles": [
            {"name": "Единство", "description": "Одна главная мысль", "tags": ["composition"]}
        ]
    }, ensure_ascii=False))
    (kb_dir / "editorial_techniques.json").write_text(json.dumps({
        "editorial_techniques": [
            {
                "category": "Нора Галь", "tags": ["editing"],
                "techniques": [
                    {
                        "id": "ng1", "name": "Убрать вводные",
                        "description": "Удаляй лишние вводные слова",
                        "when_to_use": ["Слишком много вводных"],
                        "how_to_apply": ["Удалить"],
                        "examples": [{"wrong": "Следует отметить, что", "correct": "", "explanation": ""}],
                        "tags": ["brevity"]
                    }
                ]
            }
        ]
    }, ensure_ascii=False))
    (kb_dir / "nkrj_structure_patterns.json").write_text(json.dumps({
        "corpus": "Taiga Social Media",
        "aggregate_norms": {
            "norm_sentence_length": {"avg": 12.5, "variation_coeff": 0.6, "short_share": 0.3, "medium_share": 0.5, "long_share": 0.2},
            "norm_flat_paragraph_share": 0.05,
            "norm_passive_rate": 2.1
        }
    }, ensure_ascii=False))
    return kb_dir


# ----------------------------------------------------------------------
# Тесты
# ----------------------------------------------------------------------

def test_load_knowledge_base_happy_path(minimal_kb: Path) -> None:
    kb = load_knowledge_base(minimal_kb)
    assert isinstance(kb, KnowledgeBase)
    assert "канцелярит" in kb.stop_words
    assert len(kb.grammar_errors) == 1
    assert len(kb.stylistic_issues) == 1
    assert len(kb.storytelling_frameworks) == 1
    assert len(kb.marketing_templates) == 1
    assert len(kb.logic_issues) == 1
    assert len(kb.composition_principles) == 1
    assert len(kb.editorial_techniques) == 1
    assert kb.nkrj_structure_patterns.get("corpus") == "Taiga Social Media"


def test_flatten_stylistic_issues_new_format() -> None:
    raw = {"stylistic_errors": [{"category": "канцелярит", "examples": [{"wrong": "в целях", "correct": "чтобы", "rule": "R1", "tags": ["s1"]}]}]}
    result = _flatten_stylistic_issues(raw)
    assert len(result) == 1
    entry = result[0]
    assert entry["wrong"] == "в целях"
    assert entry["correct"] == "чтобы"
    assert entry["category"] == "канцелярит"
    assert "s1" in entry["tags"]


def test_flatten_stylistic_issues_legacy_format() -> None:
    raw = {"common_issues": [{"wrong": "слово1", "correct": "слово2", "rule": "Правило", "tags": ["tag1"]}]}
    result = _flatten_stylistic_issues(raw)
    assert len(result) == 1
    entry = result[0]
    assert entry["wrong"] == "слово1"
    assert entry["correct"] == "слово2"
    # тег style не добавляется автоматически
    assert entry["tags"] == ["tag1"]


def test_flatten_examples_block_defensive() -> None:
    items = [
        "not a dict",
        {"no_examples": True},
        {"examples": "not a list"},
        {"examples": [{"wrong": "ok", "correct": "ok2"}]},
    ]
    result = _flatten_examples_block(items)
    assert isinstance(result, list)
    # Должны получить хотя бы одну запись (из последнего элемента)
    assert len(result) >= 1


def test_select_grammar_rules_match(minimal_kb: Path) -> None:
    kb = load_knowledge_base(minimal_kb)
    rules = select_grammar_rules(kb, "Это ихний дом.", ["grammar"], limit=5)
    assert any("ихний" in str(r.get("wrong", "")) for r in rules)


def test_select_grammar_rules_fallback_with_min_score() -> None:
    """При отсутствии meaningful сигналов селектор возвращает результат через fallback."""
    kb = KnowledgeBase(
        stop_words={},
        grammar_errors=[
            {"wrong": "a", "correct": "a1", "rule": "r1"},
            {"wrong": "b", "correct": "b1", "rule": "r2"},
        ],
        stylistic_issues=[],
        logic_issues=[],
        storytelling_frameworks=[],
        marketing_templates=[],
        domain_glossary={},
        composition_principles=[],
        local_cohesion=[],
        composition_errors=[],
        rhetoric_frameworks=[],
        editorial_techniques=[],
        nkrj_structure_patterns={},
    )
    rules = select_grammar_rules(kb, "Нет совпадений.", ["grammar"], limit=2)
    # fallback отработает и вернёт записи, отсортированные по info_score/порядку
    assert len(rules) == 2


def test_select_style_issues_fallback_with_min_score() -> None:
    """Аналогично для стилистики."""
    kb = KnowledgeBase(
        stop_words={},
        grammar_errors=[],
        stylistic_issues=[
            {"wrong": "x", "correct": "x1", "rule": "r1"},
            {"wrong": "y", "correct": "y1", "rule": "r2"},
        ],
        logic_issues=[],
        storytelling_frameworks=[],
        marketing_templates=[],
        domain_glossary={},
        composition_principles=[],
        local_cohesion=[],
        composition_errors=[],
        rhetoric_frameworks=[],
        editorial_techniques=[],
        nkrj_structure_patterns={},
    )
    issues = select_style_issues(kb, "Нет совпадений.", ["style"], limit=2)
    assert len(issues) == 2


def test_select_by_tags_or_all_uses_structural_scorer() -> None:
    """Структурные записи ранжируются по текстовым совпадениям."""
    entries = [
        {"name": "Обычный принцип", "tags": ["composition"]},
        {"name": "Принцип единства", "description": "Всё должно быть едино", "tags": ["composition"]},
    ]
    text = "важна единство композиции"
    normalized = _normalize_text_for_match(text)
    result = _select_ranked_entries(
        entries, normalized, ["composition"], limit=2,
        scorer=_score_structural_entry, min_score=1,
    )
    # запись с совпадением по name/description окажется первой
    assert result[0]["name"] == "Принцип единства"


def test_regression_storytelling_block(minimal_config: Path, minimal_kb: Path) -> None:
    builder = PromptBuilder(config_path=minimal_config, kb_path=minimal_kb)
    prompt = builder.build("test", domain="marketing", intent="storytelling")
    assert "Фреймворки сторителлинга" in prompt


def test_regression_marketing_block(minimal_config: Path, minimal_kb: Path) -> None:
    builder = PromptBuilder(config_path=minimal_config, kb_path=minimal_kb)
    prompt = builder.build("test", domain="marketing")
    assert "Маркетинговые шаблоны" in prompt


def test_validate_configs_and_kb_rejects_structural_entry_without_name(tmp_path: Path) -> None:
    """Структурная запись без name вызывает ошибку валидации."""
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    (kb_dir / "stop_words.json").write_text("{}")
    (kb_dir / "grammar_errors.json").write_text(json.dumps({"common_mistakes": []}))
    (kb_dir / "stylistic_issues.json").write_text(json.dumps({"stylistic_errors": []}))
    # фреймворк без поля name
    (kb_dir / "storytelling_frameworks.json").write_text(json.dumps({
        "frameworks": [{"steps": []}]
    }))
    (kb_dir / "marketing_templates.json").write_text(json.dumps({"templates": []}))
    with pytest.raises(RuntimeError, match="Knowledge base validation failed"):
        validate_configs_and_kb(Path(tmp_path / "config"), kb_dir)


def test_load_overlay_config_fails_for_unknown_overlay(minimal_config: Path, minimal_kb: Path) -> None:
    """Неизвестный overlay вызывает FileNotFoundError (fail‑fast)."""
    builder = PromptBuilder(config_path=minimal_config, kb_path=minimal_kb)
    with pytest.raises(FileNotFoundError):
        builder.build("text", domain="marketing", overlays=["nonexistent"])


def test_public_selectors_min_score_behavior() -> None:
    """min_score=1 пропускает только значимые записи; можно обойти, передав min_score=0."""
    kb = KnowledgeBase(
        stop_words={},
        grammar_errors=[{"wrong": "a", "correct": "a1", "rule": "r1", "tags": []}],
        stylistic_issues=[],
        logic_issues=[],
        storytelling_frameworks=[],
        marketing_templates=[],
        domain_glossary={},
        composition_principles=[],
        local_cohesion=[],
        composition_errors=[],
        rhetoric_frameworks=[],
        editorial_techniques=[],
        nkrj_structure_patterns={},
    )
    # с min_score=1 (по умолчанию) результат будет из fallback
    res_default = select_grammar_rules(kb, "текст", ["grammar"], limit=1)
    assert len(res_default) == 1  # fallback отработает
    # с min_score=0 все записи пройдут ranked‑путь
    res_zero = select_grammar_rules(kb, "текст", ["grammar"], limit=1, min_score=0)
    assert len(res_zero) == 1


def test_text_match_wins_over_tags(minimal_kb: Path) -> None:
    kb = load_knowledge_base(minimal_kb)
    kb.grammar_errors = [
        {"wrong": "тестовый", "correct": "тест", "rule": "r1", "tags": ["grammar"]},
        {"wrong": "другой", "correct": "иной", "rule": "r2", "tags": ["grammar"]},
    ]
    rules = select_grammar_rules(kb, "Это тестовый пример.", ["grammar"], limit=5)
    assert rules[0]["wrong"] == "тестовый"


def test_tag_overlap_no_text_match(minimal_kb: Path) -> None:
    kb = load_knowledge_base(minimal_kb)
    kb.grammar_errors = [
        {"wrong": "a", "correct": "a1", "rule": "r1", "tags": ["tagA"]},
        {"wrong": "b", "correct": "b1", "rule": "r2", "tags": ["tagA", "tagB"]},
        {"wrong": "c", "correct": "c1", "rule": "r3", "tags": ["tagC"]},
    ]
    rules = select_grammar_rules(kb, "Нет совпадений.", ["tagA", "tagB"], limit=5)
    assert rules[0]["wrong"] == "b"


def test_fallback_activates_with_min_score() -> None:
    entries = [
        {"name": "low_info", "tags": []},
        {"name": "high_info", "description": "detailed", "rule": "rule", "tags": []},
    ]
    result = _select_ranked_entries(entries, "", ["any"], limit=2, min_score=1)
    assert result[0]["name"] == "high_info"


def test_fallback_original_order_stability() -> None:
    entries = [
        {"name": "first", "description": "desc"},
        {"name": "second", "description": "desc"},
    ]
    result = _select_ranked_entries(entries, "", ["any"], limit=2, min_score=1)
    assert result[0]["name"] == "first"


def test_primary_tag_wins_over_expanded() -> None:
    entry_primary = {"wrong": "e1", "tags": ["marketing"]}
    entry_expanded = {"wrong": "e2", "tags": ["promo"]}
    expanded_tags = {"promo"}
    score_p, _ = _score_rule_entry(entry_primary, "", {"marketing"}, 0, expanded_tags=expanded_tags)
    score_e, _ = _score_rule_entry(entry_expanded, "", {"marketing"}, 1, expanded_tags=expanded_tags)
    assert score_p > score_e


def test_structural_entry_scoring_storytelling(tmp_path: Path) -> None:
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    (kb_dir / "stop_words.json").write_text("{}")
    (kb_dir / "grammar_errors.json").write_text(json.dumps({"common_mistakes": []}))
    (kb_dir / "stylistic_issues.json").write_text(json.dumps({"stylistic_errors": []}))
    (kb_dir / "storytelling_frameworks.json").write_text(json.dumps({
        "frameworks": [
            {"name": "Путь героя", "steps": [], "tags": ["story"]},
            {"name": "Спираль", "description": "Для сложных сюжетов", "when_to_use": ["запутанный сюжет"]},
        ]
    }))
    (kb_dir / "marketing_templates.json").write_text(json.dumps({"templates": []}))
    kb = load_knowledge_base(kb_dir)
    text = "У нас запутанный сюжет, нужно что-то придумать."
    norm = _normalize_text_for_match(text)
    scores = []
    for idx, entry in enumerate(kb.storytelling_frameworks):
        s, _ = _score_structural_entry(entry, norm, {"storytelling"}, idx)
        scores.append((s, entry.get("name")))
    assert scores[1][0] > scores[0][0]
    assert scores[1][1] == "Спираль"


def test_deduplication_in_selection(minimal_kb: Path) -> None:
    kb = load_knowledge_base(minimal_kb)
    duplicate = {"wrong": "дубль", "correct": "оригинал", "rule": "правило", "tags": ["grammar"]}
    kb.grammar_errors = [duplicate, duplicate.copy()]
    rules = select_grammar_rules(kb, "дубль", ["grammar"], limit=5)
    assert len(rules) == 1


def test_diagnostics_does_not_change_global_log_level() -> None:
    root = logging.getLogger()
    original_level = root.level
    PromptBuilder(enable_selection_diagnostics=True)
    assert root.level == original_level