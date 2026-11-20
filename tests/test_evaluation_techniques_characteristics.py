"""Характеристические тесты для _append_evaluation_techniques (итерация 18, PR-9).

Фиксируют текущее поведение до рефакторинга: какие строки пишутся в lines,
как обрабатываются category/description/editor_rules/replacement_strategies/
diagnostics/tests/source, как работают truncation и фильтрация.
"""

from __future__ import annotations

from typing import Any

from src.prompt_builder.kb_rendering import _append_evaluation_techniques


def _call(data: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    _append_evaluation_techniques(lines, "Заголовок:", data)
    return lines


def test_empty_data_writes_nothing() -> None:
    """Пустой data → пустой lines (guard, даже title не пишется)."""
    assert _call({}) == []


def test_category_and_description() -> None:
    """category и description → две строки."""
    assert _call({"category": "evals", "description": "описание"}) == [
        "Заголовок:",
        "Категория: evals",
        "Описание: описание",
    ]


def test_editor_rules_truncated_to_5_then_filtered() -> None:
    """editor_rules: срез [:5], затем фильтр строк (не-строки и пустые пропускаются)."""
    assert _call({"editor_rules": ["a", "", 42, "b", "c", "d", "e"]}) == [
        "Заголовок:",
        "Правила редактора:",
        "- a",
        "- b",
        "- c",
    ]


def test_replacement_strategies_truncated_to_3_with_variants() -> None:
    """strategies: срез [:3]; name+desc → «- n: d»; только name → «- n»; только desc → «- d»."""
    assert _call(
        {
            "replacement_strategies": [
                {"name": "n1", "description": "d1"},
                {"name": "n2"},
                {"description": "d3"},
                {"name": "n4", "description": "d4"},
                "not_a_dict",
            ]
        }
    ) == [
        "Заголовок:",
        "Стратегии замены оценок:",
        "- n1: d1",
        "- n2",
        "- d3",
    ]


def test_diagnostics_and_tests_truncated_to_3() -> None:
    """diagnostics и tests: срез [:3], не-dict пропускаются, только name."""
    assert _call(
        {
            "diagnostics": [
                {"name": "d1"},
                {"name": "d2"},
                {"name": "d3"},
                {"name": "d4"},
            ],
            "tests": [
                {"name": "t1"},
                "not_a_dict",
                {"name": "t2"},
                {"name": "t3"},
            ],
        }
    ) == [
        "Заголовок:",
        "Чек-листы для обнаружения оценок:",
        "- d1",
        "- d2",
        "- d3",
        "Тесты на замену оценок:",
        "- t1",
        "- t2",
    ]


def test_source_title() -> None:
    """source с title → строка «Источник: title»."""
    assert _call({"source": {"title": "Источник X"}}) == [
        "Заголовок:",
        "Источник: Источник X",
    ]
