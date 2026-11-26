"""Характеристические тесты для _append_case_study_entries (итерация 18, PR-3).

Фиксируют текущее поведение до рефакторинга: какие строки пишутся в lines,
как обрабатываются name/description/constraints/sections.
"""

from __future__ import annotations

from typing import Any

from src.prompt_builder.kb_rendering import _append_case_study_entries


def _call(entries: list[dict[str, Any]]) -> list[str]:
    """Прогоняет _append_case_study_entries и возвращает накопленные строки."""
    lines: list[str] = []
    _append_case_study_entries(lines, "Заголовок:", entries)
    return lines


def test_case_study_empty_entries_writes_nothing() -> None:
    """Пустой entries — не пишется ничего, даже title."""
    lines: list[str] = []
    _append_case_study_entries(lines, "Заголовок:", [])
    assert lines == []


def test_case_study_name_and_description() -> None:
    """name + description → строка «- name: description»."""
    assert _call([{"name": "cs1", "description": "desc"}]) == [
        "Заголовок:",
        "- cs1: desc",
    ]


def test_case_study_description_without_name() -> None:
    """description без name → строка «- description»."""
    assert _call([{"description": "описание"}]) == [
        "Заголовок:",
        "- описание",
    ]


def test_case_study_name_without_description() -> None:
    """name без description → строка «- name»."""
    assert _call([{"name": "only-name"}]) == [
        "Заголовок:",
        "- only-name",
    ]


def test_case_study_constraints_truncated_to_4() -> None:
    """constraints обрезается до 4; каждая строка «  Ограничение: ...»."""
    constraints = [f"c{i}" for i in range(6)]
    assert _call([{"name": "cs", "constraints": constraints}]) == [
        "Заголовок:",
        "- cs",
        "  Ограничение: c0",
        "  Ограничение: c1",
        "  Ограничение: c2",
        "  Ограничение: c3",
    ]


def test_case_study_sections_with_name_goal_hint() -> None:
    """sections: name+goal → «  N. name — goal»; hint → «     Ориентир: ...»; без name/goal — пропуск."""
    sections = [
        {"name": "Введение", "goal": "зацепить", "hint": "цифра"},
        {"name": "Только-имя"},
        {"goal": "только-цель"},
        {"hint": "без name и goal — пропуск"},
    ]
    assert _call([{"name": "cs", "sections": sections}]) == [
        "Заголовок:",
        "- cs",
        "  1. Введение — зацепить",
        "     Ориентир: цифра",
        "  2. Только-имя",
        "  3. только-цель",
        # 4-я секция пропущена (ни name, ни goal)
    ]


def test_case_study_sections_truncated_to_8() -> None:
    """sections обрезается до 8; каждая → одна строка «  N. name — goal»."""
    sections = [{"name": f"s{i}", "goal": "g"} for i in range(10)]
    lines = _call([{"name": "cs", "sections": sections}])
    assert len(lines) == 2 + 8  # title + header + 8 секций
    assert lines[-1] == "  8. s7 — g"
