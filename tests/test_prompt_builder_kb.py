"""
Unit-тесты для prompt_builder.py.
Используют временные директории с минимальными валидными JSON.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Импортируем тестируемый модуль
from prompt_builder import (
    KnowledgeBase,
    PromptBuilder,
    _flatten_editorial_techniques,
    _flatten_stylistic_issues,
    load_knowledge_base,
    select_grammar_rules,
    select_logic_issues,
    select_style_issues,
    validate_configs_and_kb,
)


# ----------------------------------------------------------------------
# Фикстуры для создания временных файловых структур
# ----------------------------------------------------------------------

@pytest.fixture
def minimal_config(tmp_path: Path) -> Path:
    """Создаёт минимальную валидную структуру config/."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    # core.json
    (config_dir / "core.json").write_text(
        json.dumps({
            "role": "Test Editor",
            "priorities": "Test priorities",
            "basic_audit_instructions": ["Check grammar", "Check style"],
            "forbidden": ["No hate speech"]
        }, ensure_ascii=False)
    )

    # domains/marketing.json
    domains_dir = config_dir / "domains"
    domains_dir.mkdir()
    (domains_dir / "marketing.json").write_text(
        json.dumps({
            "name": "Маркетинг",
            "system_rules": "Текст для привлечения клиентов",
            "tone": "Убедительный",
            "allow_storytelling": True,
            "allow_marketing": True
        }, ensure_ascii=False)
    )

    # intents/storytelling.json
    intents_dir = config_dir / "intents"
    intents_dir.mkdir()
    (intents_dir / "storytelling.json").write_text(
        json.dumps({
            "name": "Сторителлинг",
            "instructions": ["Используй структуру истории"]
        }, ensure_ascii=False)
    )

    # overlays/logic.json
    overlays_dir = config_dir / "overlays"
    overlays_dir.mkdir()
    (overlays_dir / "logic.json").write_text(
        json.dumps({
            "name": "Логика",
            "instructions": ["Проверь логические связи"]
        }, ensure_ascii=False)
    )

    # output_format.json
    (config_dir / "output_format.json").write_text(
        json.dumps({
            "text_only": "Только текст"
        }, ensure_ascii=False)
    )

    return config_dir


@pytest.fixture
def minimal_kb(tmp_path: Path) -> Path:
    """Создаёт минимальную валидную структуру knowledge_base/."""
    kb_dir = tmp_path / "knowledge_base"
    kb_dir.mkdir()

    # stop_words.json
    (kb_dir / "stop_words.json").write_text(
        json.dumps({
            "канцелярит": ["осуществлять", "в рамках"],
            "усилители": ["максимально", "достаточно"]
        }, ensure_ascii=False)
    )

    # grammar_errors.json
    (kb_dir / "grammar_errors.json").write_text(
        json.dumps({
            "common_mistakes": [
                {"wrong": "ихний", "correct": "их", "rule": "Притяжательное местоимение", "tags": ["grammar"]}
            ]
        }, ensure_ascii=False)
    )

    # stylistic_issues.json (новый формат)
    (kb_dir / "stylistic_issues.json").write_text(
        json.dumps({
            "stylistic_errors": [
                {
                    "category": "канцелярит",
                    "description": "Канцелярские обороты",
                    "examples": [
                        {"wrong": "в целях", "correct": "чтобы", "rule": "Избегай канцелярита", "tags": ["style"]}
                    ]
                }
            ]
        }, ensure_ascii=False)
    )

    # storytelling_frameworks.json
    (kb_dir / "storytelling_frameworks.json").write_text(
        json.dumps({
            "frameworks": [
                {"name": "Путь героя", "steps": [{"name": "Обычный мир"}, {"name": "Зов к приключению"}]}
            ]
        }, ensure_ascii=False)
    )

    # marketing_templates.json
    (kb_dir / "marketing_templates.json").write_text(
        json.dumps({
            "templates": [
                {"name": "AIDA", "sections": [{"name": "Attention"}, {"name": "Interest"}]}
            ]
        }, ensure_ascii=False)
    )

    # Опциональные файлы (для проверки, что они загружаются)
    (kb_dir / "logic_issues.json").write_text(
        json.dumps({
            "issues": [
                {"name": "Ложная дилемма", "description": "Только два варианта", "tags": ["logic"]}
            ]
        }, ensure_ascii=False)
    )

    (kb_dir / "composition_principles.json").write_text(
        json.dumps({
            "composition_principles": [
                {"name": "Единство", "description": "Одна главная мысль", "tags": ["composition"]}
            ]
        }, ensure_ascii=False)
    )

    (kb_dir / "editorial_techniques.json").write_text(
        json.dumps({
            "editorial_techniques": [
                {
                    "category": "Нора Галь",
                    "tags": ["editing"],
                    "techniques": [
                        {
                            "id": "ng1",
                            "name": "Убрать вводные",
                            "description": "Удаляй лишние вводные слова",
                            "when_to_use": ["Слишком много вводных"],
                            "how_to_apply": ["Удалить"],
                            "examples": [{"wrong": "Следует отметить, что", "correct": "", "explanation": ""}],
                            "tags": ["brevity"]
                        }
                    ]
                }
            ]
        }, ensure_ascii=False)
    )

    (kb_dir / "nkrj_structure_patterns.json").write_text(
        json.dumps({
            "corpus": "Taiga Social Media",
            "aggregate_norms": {
                "norm_sentence_length": {"avg": 12.5, "variation_coeff": 0.6, "short_share": 0.3, "medium_share": 0.5, "long_share": 0.2},
                "norm_flat_paragraph_share": 0.05,
                "norm_passive_rate": 2.1
            }
        }, ensure_ascii=False)
    )

    return kb_dir


# ----------------------------------------------------------------------
# Тесты загрузки KB и flattening
# ----------------------------------------------------------------------

def test_load_knowledge_base_happy_path(minimal_kb: Path):
    """Проверяет загрузку KB с полным набором файлов."""
    kb = load_knowledge_base(minimal_kb)

    assert isinstance(kb, KnowledgeBase)
    assert isinstance(kb.stop_words, dict)
    assert "канцелярит" in kb.stop_words
    assert len(kb.grammar_errors) == 1
    assert len(kb.stylistic_issues) == 1
    assert len(kb.storytelling_frameworks) == 1
    assert len(kb.marketing_templates) == 1
    assert len(kb.logic_issues) == 1
    assert len(kb.composition_principles) == 1
    assert len(kb.editorial_techniques) == 1
    assert kb.nkrj_structure_patterns.get("corpus") == "Taiga Social Media"


def test_flatten_stylistic_issues_new_format():
    """Новый категорийный формат stylistic_issues."""
    raw = {
        "stylistic_errors": [
            {
                "category": "канцелярит",
                "description": "...",
                "examples": [
                    {"wrong": "в целях", "correct": "чтобы", "rule": "R1", "tags": ["s1"]}
                ]
            }
        ]
    }
    result = _flatten_stylistic_issues(raw)
    assert len(result) == 1
    entry = result[0]
    assert entry["wrong"] == "в целях"
    assert entry["correct"] == "чтобы"
    assert entry["rule"] == "R1"
    assert entry["category"] == "канцелярит"
    assert "tags" in entry
    assert "s1" in entry["tags"]


def test_flatten_stylistic_issues_legacy_format():
    """Legacy формат common_issues."""
    raw = {
        "common_issues": [
            {"wrong": "слово1", "correct": "слово2", "rule": "Правило", "tags": ["tag1"]}
        ]
    }
    result = _flatten_stylistic_issues(raw)
    assert len(result) == 1
    entry = result[0]
    assert entry["wrong"] == "слово1"
    assert entry["correct"] == "слово2"
    assert "style" in entry.get("tags", [])   # тег style добавляется автоматически


def test_flatten_editorial_techniques():
    """Проверяет структуру развёрнутого editorial_techniques."""
    raw = {
        "editorial_techniques": [
            {
                "category": "Нора Галь",
                "tags": ["editing"],
                "techniques": [
                    {
                        "id": "ng1",
                        "name": "Убрать вводные",
                        "description": "Удаляй лишние вводные слова",
                        "when_to_use": ["Слишком много вводных"],
                        "how_to_apply": ["Удалить"],
                        "examples": [{"wrong": "Следует отметить, что", "correct": "", "explanation": ""}],
                        "tags": ["brevity"],
                        "source": {"book": "Слово живое и мёртвое"}
                    }
                ]
            }
        ]
    }
    result = _flatten_editorial_techniques(raw)
    assert len(result) == 1
    tech = result[0]
    assert tech["id"] == "ng1"
    assert tech["name"] == "Убрать вводные"
    assert tech["category"] == "Нора Галь"
    assert tech["description"] == "Удаляй лишние вводные слова"
    assert "editing" in tech["tags"]
    assert "brevity" in tech["tags"]
    assert tech["example_wrong"] == "Следует отметить, что"
    assert tech["example_correct"] == ""
    assert isinstance(tech["source"], dict)


# ----------------------------------------------------------------------
# Тесты селекторов
# ----------------------------------------------------------------------

def test_select_grammar_rules_match(minimal_kb: Path):
    """Прямое совпадение по тексту."""
    kb = load_knowledge_base(minimal_kb)
    text = "Это ихний дом."
    rules = select_grammar_rules(kb, text, tags=["grammar"], limit=5)
    assert len(rules) > 0
    assert any("ихний" in str(rule.get("wrong", "")) for rule in rules)


def test_select_grammar_rules_fallback(minimal_kb: Path):
    """Без прямого совпадения возвращает записи по тегам."""
    kb = load_knowledge_base(minimal_kb)
    text = "Всё правильно написано."
    rules = select_grammar_rules(kb, text, tags=["grammar"], limit=5)
    # fallback всё равно вернёт записи (если они есть в базе)
    assert len(rules) > 0


def test_select_style_issues_match(minimal_kb: Path):
    """Прямое совпадение по стилистическому паттерну."""
    kb = load_knowledge_base(minimal_kb)
    text = "В целях улучшения качества."
    issues = select_style_issues(kb, text, tags=["style"], limit=5)
    assert len(issues) > 0
    assert any("в целях" in str(issue.get("wrong", "")) for issue in issues)


def test_select_logic_issues_with_dedicated_file(minimal_kb: Path):
    """При наличии logic_issues.json использует его."""
    kb = load_knowledge_base(minimal_kb)
    text = "Либо мы делаем так, либо всё пропало."
    issues = select_logic_issues(kb, text, tags=["logic"], limit=5)
    assert len(issues) > 0
    # В нашем файле одна запись
    assert issues[0].get("name") == "Ложная дилемма"


def test_select_logic_issues_fallback_to_combined(tmp_path: Path):
    """При отсутствии logic_issues.json использует grammar+style с тегом logic."""
    kb_dir = tmp_path / "kb_no_logic"
    kb_dir.mkdir()
    (kb_dir / "stop_words.json").write_text("{}")
    (kb_dir / "grammar_errors.json").write_text(json.dumps({"common_mistakes": []}))
    (kb_dir / "stylistic_issues.json").write_text(json.dumps({"stylistic_errors": []}))
    (kb_dir / "storytelling_frameworks.json").write_text(json.dumps({"frameworks": []}))
    (kb_dir / "marketing_templates.json").write_text(json.dumps({"templates": []}))
    kb = load_knowledge_base(kb_dir)
    # Добавим в grammar запись с тегом logic
    kb.grammar_errors.append({"wrong": "пример", "rule": "логика", "tags": ["logic"]})
    issues = select_logic_issues(kb, "любой текст", tags=["test"], limit=5)
    assert len(issues) > 0


# ----------------------------------------------------------------------
# Тесты validate_configs_and_kb
# ----------------------------------------------------------------------

def test_validate_configs_and_kb_success(minimal_config: Path, minimal_kb: Path):
    """Полностью валидные конфиги и KB должны проходить без ошибок."""
    # Перенаправляем логи, чтобы не засорять вывод тестов
    validate_configs_and_kb(minimal_config, minimal_kb)


def test_validate_configs_and_kb_failure_missing_core_role(minimal_config: Path, minimal_kb: Path):
    """Удаляем обязательное поле в core.json."""
    core_path = minimal_config / "core.json"
    data = json.loads(core_path.read_text())
    del data["role"]
    core_path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="Core config validation failed"):
        validate_configs_and_kb(minimal_config, minimal_kb)


def test_validate_configs_and_kb_failure_bad_stop_words(minimal_config: Path, minimal_kb: Path):
    """stop_words содержит неверный тип значения."""
    kb_path = minimal_kb
    sw_path = kb_path / "stop_words.json"
    data = json.loads(sw_path.read_text())
    data["канцелярит"] = "не список"  # type: ignore
    sw_path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="Knowledge base validation failed"):
        validate_configs_and_kb(minimal_config, minimal_kb)


# ----------------------------------------------------------------------
# Тест PromptBuilder.build()
# ----------------------------------------------------------------------

def test_prompt_builder_build_happy_path(minimal_config: Path, minimal_kb: Path):
    """Базовый вызов build с минимальными параметрами."""
    builder = PromptBuilder(config_path=minimal_config, kb_path=minimal_kb)
    text = "Это тестовый текст с ошибкой: ихний."
    prompt = builder.build(
        text=text,
        domain="marketing",
        intent="storytelling",
        overlays=["logic"],
        output_mode="text_only",
        include_knowledge=True,
    )
    # Проверяем наличие основных блоков
    assert "Test Editor" in prompt
    assert "Домен: Текст для привлечения клиентов" in prompt
    assert "Цель обработки: Сторителлинг" in prompt
    assert "Формат ответа:" in prompt
    assert text in prompt
    # Knowledge блок присутствует
    assert "База знаний" in prompt
    # Стоп-слова в компактном виде
    assert "канцелярит" in prompt