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

# Файлы с корневым типом list (а не dict)
_LIST_ROOT_FILES = {"stylistic_issues.json"}


@pytest.mark.parametrize("filename", KB_FILES)
def test_json_is_valid(filename: str) -> None:
    """Каждый файл базы знаний — валидный JSON с ожидаемым корневым типом."""
    path = KB_PATH / filename
    if not path.exists():
        pytest.skip(f"{filename} не найден (опционален)")
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)  # упадёт при невалидном JSON
    if filename in _LIST_ROOT_FILES:
        assert isinstance(data, list), f"{filename} должен быть JSON-массивом"
    else:
        assert isinstance(data, dict), f"{filename} должен быть JSON-объектом"


# ============================================================================
# stop_words.json
# ============================================================================

def test_stop_words_structure() -> None:
    """stop_words.json поддерживает смешанную структуру: списки и словари списков."""
    data = load_json(KB_PATH / "stop_words.json")
    assert isinstance(data, dict)
    assert len(data) > 0, "stop_words.json не должен быть пустым"

    def assert_list_payload(items: list, ctx: str) -> None:
        for item in items:
            assert isinstance(item, (str, dict)), (
                f"Элемент в '{ctx}' должен быть строкой или словарём, "
                f"получено: {type(item)}"
            )
            if isinstance(item, str):
                assert len(item.strip()) > 0, f"Пустая строка в '{ctx}'"
            else:
                assert "id" in item or "name" in item or "pattern" in item, (
                    f"Объект в '{ctx}' должен иметь поле 'id', 'name' или 'pattern'"
                )

    for category, value in data.items():
        if isinstance(value, list):
            assert_list_payload(value, category)
        elif isinstance(value, dict):
            assert len(value) > 0, f"Словарь '{category}' не должен быть пустым"
            for nested_category, nested_value in value.items():
                assert isinstance(
                    nested_value, list
                ), f"Подкатегория '{category}.{nested_category}' должна быть списком"
                assert_list_payload(nested_value, f"{category}.{nested_category}")
        else:
            raise AssertionError(
                f"Категория '{category}' должна быть списком или словарём списков"
            )


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
    """stylistic_issues.json — список записей с полями wrong/correct/rule."""
    path = KB_PATH / "stylistic_issues.json"
    if not path.exists():
        pytest.skip("stylistic_issues.json не найден")
    data = load_json(path)
    assert isinstance(data, list), "stylistic_issues.json должен быть JSON-массивом"
    assert len(data) > 0, "stylistic_issues.json не должен быть пустым"

    for i, entry in enumerate(data):
        assert isinstance(entry, dict), f"Элемент #{i} должен быть словарём"
        assert "wrong" in entry, f"Элемент #{i}: нет поля 'wrong'"
        assert "correct" in entry, f"Элемент #{i}: нет поля 'correct'"
        assert "rule" in entry, f"Элемент #{i}: нет поля 'rule'"


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
