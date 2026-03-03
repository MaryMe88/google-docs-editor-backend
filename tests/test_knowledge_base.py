"""
test_knowledge_base.py

Проверяет, что все JSON-файлы базы знаний корректны и содержат ожидаемую структуру.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.conftest import KB_PATH, load_json


# ============================================================================
# Валидация JSON-синтаксиса для всех файлов
# ============================================================================


KB_FILES = [
    "stop_words.json",
    "grammar_errors.json",
    "stylistic_issues.json",
    "storytelling_frameworks.json",
    "marketing_templates.json",
    "domain_glossary.json",
]


@pytest.mark.parametrize("filename", KB_FILES)
def test_json_is_valid(filename: str) -> None:
    """Каждый файл базы знаний — валидный JSON."""
    path = KB_PATH / filename
    if not path.exists():
        pytest.skip(f"{filename} не найден (опционален)")
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)  # упадёт при невалидном JSON
    assert isinstance(data, dict), f"{filename} должен быть JSON-объектом"


# ============================================================================
# stop_words.json
# ============================================================================


def test_stop_words_structure() -> None:
    """stop_words.json — словарь категорий, каждая категория — список строк."""
    data = load_json(KB_PATH / "stop_words.json")
    assert len(data) > 0, "stop_words.json не должен быть пустым"

    for category, words in data.items():
        assert isinstance(words, list), f"Категория '{category}' должна быть списком"
        for word in words:
            assert isinstance(word, str), f"Элемент в '{category}' должен быть строкой"
            assert len(word.strip()) > 0, f"Пустая строка в категории '{category}'"


# ============================================================================
# grammar_errors.json
# ============================================================================


def test_grammar_errors_structure() -> None:
    """grammar_errors.json содержит common_mistakes с полями wrong/correct/rule."""
    data = load_json(KB_PATH / "grammar_errors.json")
    mistakes = data.get("common_mistakes", [])
    assert len(mistakes) > 0, "common_mistakes не должен быть пустым"

    for i, entry in enumerate(mistakes):
        assert "wrong" in entry, f"Элемент #{i}: отсутствует 'wrong'"
        assert "correct" in entry, f"Элемент #{i}: отсутствует 'correct'"
        assert "rule" in entry, f"Элемент #{i}: отсутствует 'rule'"


# ============================================================================
# stylistic_issues.json
# ============================================================================


def test_stylistic_issues_structure() -> None:
    """stylistic_issues.json содержит common_issues с полями wrong/correct/rule."""
    data = load_json(KB_PATH / "stylistic_issues.json")
    issues = data.get("common_issues", [])
    assert len(issues) > 0, "common_issues не должен быть пустым"

    for i, entry in enumerate(issues):
        assert "wrong" in entry, f"Элемент #{i}: отсутствует 'wrong'"
        assert "correct" in entry, f"Элемент #{i}: отсутствует 'correct'"


# ============================================================================
# storytelling_frameworks.json
# ============================================================================


def test_storytelling_frameworks_structure() -> None:
    """storytelling_frameworks.json — массив фреймворков со steps."""
    data = load_json(KB_PATH / "storytelling_frameworks.json")
    frameworks = data.get("frameworks", [])
    assert len(frameworks) > 0, "frameworks не должен быть пустым"

    for fw in frameworks:
        assert "id" in fw, f"Фреймворк без id: {fw.get('name', '???')}"
        assert "name" in fw, "Фреймворк без name"
        steps = fw.get("steps", [])
        assert len(steps) > 0, f"Фреймворк '{fw['name']}' без шагов"
        for step in steps:
            assert "name" in step, f"Шаг без name в '{fw['name']}'"
            assert "goal" in step, f"Шаг без goal в '{fw['name']}'"


# ============================================================================
# marketing_templates.json
# ============================================================================


def test_marketing_templates_structure() -> None:
    """marketing_templates.json — массив шаблонов с sections."""
    data = load_json(KB_PATH / "marketing_templates.json")
    templates = data.get("templates", [])
    assert len(templates) > 0, "templates не должен быть пустым"

    for tpl in templates:
        assert "id" in tpl, f"Шаблон без id: {tpl.get('name', '???')}"
        assert "name" in tpl, "Шаблон без name"
        sections = tpl.get("sections", [])
        assert len(sections) > 0, f"Шаблон '{tpl['name']}' без секций"
        for sec in sections:
            assert "name" in sec, f"Секция без name в '{tpl['name']}'"
            assert "goal" in sec, f"Секция без goal в '{tpl['name']}'"


# ============================================================================
# domain_glossary.json (опционально, может быть пустым)
# ============================================================================


def test_domain_glossary_is_valid_dict() -> None:
    """domain_glossary.json — словарь (может быть пустым)."""
    path = KB_PATH / "domain_glossary.json"
    if not path.exists():
        pytest.skip("domain_glossary.json не найден")
    data = load_json(path)
    assert isinstance(data, dict), "domain_glossary.json должен быть объектом"
