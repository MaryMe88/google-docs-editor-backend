"""
tests/test_retrieval.py

Юнит-тесты для few‑shot функций (PR‑2).
Тестируют детерминизм, лимиты, ротацию и форматирование.
"""

from __future__ import annotations

import pytest

from src.prompt_builder import (
    _has_few_shot_pair,
    _select_few_shot_examples,
    _format_few_shot_example,
)


def test_has_few_shot_pair_missing_fields() -> None:
    """Проверка обнаружения пары 'было → стало'."""
    assert not _has_few_shot_pair({"id": "1", "rule": "some rule"})
    assert not _has_few_shot_pair({"wrong": "x"})
    assert not _has_few_shot_pair({"correct": "y"})
    assert _has_few_shot_pair({"wrong": "x", "correct": "y"})
    assert _has_few_shot_pair({"example_wrong": "x", "example_correct": "y"})


def test_format_few_shot_example() -> None:
    """Форматирование одной пары."""
    # Обычные поля wrong/correct
    entry = {"wrong": "ошибочный текст", "correct": "правильный текст"}
    result = _format_few_shot_example(entry)
    assert result == "Было: ошибочный текст\nСтало: правильный текст"

    # Если есть оба варианта, приоритет у wrong/correct (так работает 'or')
    entry2 = {
        "wrong": "wrong",
        "correct": "correct",
        "example_wrong": "example_wrong",
        "example_correct": "example_correct",
    }
    result2 = _format_few_shot_example(entry2)
    assert result2 == "Было: wrong\nСтало: correct"


def test_select_few_shot_deterministic() -> None:
    """Выборка с фиксированным seed должна быть детерминированной."""
    entries = [
        {"id": str(i), "wrong": f"w{i}", "correct": f"c{i}"}
        for i in range(15)
    ]
    result_a = _select_few_shot_examples(entries, max_examples=3, seed=42)
    result_b = _select_few_shot_examples(entries, max_examples=3, seed=42)
    assert result_a == result_b


def test_select_few_shot_random_rotation() -> None:
    """Без seed выборка должна давать разные комбинации."""
    entries = [
        {"id": str(i), "wrong": f"w{i}", "correct": f"c{i}"}
        for i in range(15)
    ]
    results = set()
    for _ in range(20):
        selected = _select_few_shot_examples(entries, max_examples=3)
        ids = tuple(e["id"] for e in selected)
        results.add(ids)
    # Хотя бы две разных комбинации (обычно гораздо больше)
    assert len(results) > 1


def test_select_few_shot_respects_max_examples() -> None:
    """Не возвращает больше запрошенного количества."""
    entries = [
        {"wrong": f"w{i}", "correct": f"c{i}"}
        for i in range(20)
    ]
    result = _select_few_shot_examples(entries, max_examples=3)
    assert len(result) <= 3


def test_select_few_shot_respects_pool_size() -> None:
    """Выбирает только из первых pool_size записей."""
    entries = [
        {"id": str(i), "wrong": f"w{i}", "correct": f"c{i}"}
        for i in range(50)
    ]
    # pool_size по умолчанию = 10
    selected = _select_few_shot_examples(entries, max_examples=5, seed=123)
    # Все выбранные id должны быть от 0 до 9
    ids = [int(e["id"]) for e in selected]
    assert all(0 <= i < 10 for i in ids)


def test_select_few_shot_returns_less_when_pool_small() -> None:
    """Если в пуле меньше записей, чем max_examples, возвращает всё."""
    entries = [
        {"id": str(i), "wrong": f"w{i}", "correct": f"c{i}"}
        for i in range(4)
    ]
    result = _select_few_shot_examples(entries, max_examples=10)
    assert len(result) == 4


def test_select_few_shot_empty_entries() -> None:
    """Пустой список записей → пустой результат."""
    assert _select_few_shot_examples([], max_examples=3) == []


def test_select_few_shot_zero_max() -> None:
    """max_examples = 0 → пустой результат."""
    entries = [{"wrong": "x", "correct": "y"}]
    assert _select_few_shot_examples(entries, max_examples=0) == []