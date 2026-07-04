"""Юнит-тесты детектора служебных плейсхолдеров (src.output_guard).

Покрывают:
- текст без PII не даёт срабатываний;
- явные PII-плейсхолдеры детектируются;
- ложноположительные кейсы (русские скобки, блоки отчёта, markdown-ссылки)
  НЕ детектируются;
- harden_prompt_against_placeholders добавляет инструкцию идемпотентно.
"""

from __future__ import annotations

import pytest

from src.output_guard import (
    PLACEHOLDER_GUARD_INSTRUCTION,
    find_placeholder_leaks,
    harden_prompt_against_placeholders,
    has_placeholder_leak,
)


# ---------------------------------------------------------------------------
# Текст без PII: срабатываний быть не должно
# ---------------------------------------------------------------------------
CLEAN_TEXTS = [
    "Обычный отредактированный текст без всяких токенов.",
    "Мария написала письмо в понедельник и отправила его коллегам.",
    "",
    "Список: 1. первое 2. второе 3. третье.",
    "Ссылка в тексте: [подробнее](https://example.com) — переход на сайт.",
    "В скобках русское примечание [см. выше] и [важно] — это не токены.",
    "Блок отчёта: Исходный ИП: 2.4, Итоговый ИП: 0.9.",
    "Аббревиатура [ГОСТ] написана кириллицей и не является плейсхолдером.",
]


@pytest.mark.parametrize("text", CLEAN_TEXTS)
def test_clean_text_has_no_leak(text: str) -> None:
    assert has_placeholder_leak(text) is False
    assert find_placeholder_leaks(text) == []


# ---------------------------------------------------------------------------
# Явные PII-плейсхолдеры должны детектироваться
# ---------------------------------------------------------------------------
LEAKY_CASES = [
    ("[PERSON_NAME] важно учесть заранее", ["[PERSON_NAME]"]),
    ("[ADDRESS] мне нужно волонтёрство?", ["[ADDRESS]"]),
    ("Позвоните [PHONE] или напишите [EMAIL].", ["[EMAIL]", "[PHONE]"]),
    ("Двойные скобки [[PERSON_1]] тоже ловим.", ["[[PERSON_1]]"]),
    ("Токен с суффиксом [EMAIL_001] здесь.", ["[EMAIL_001]"]),
    ("[NAME] и снова [NAME] — уникальны в списке.", ["[NAME]"]),
]


@pytest.mark.parametrize("text,expected", LEAKY_CASES)
def test_leaky_text_is_detected(text: str, expected: list[str]) -> None:
    assert has_placeholder_leak(text) is True
    assert find_placeholder_leaks(text) == expected


def test_generic_unknown_token_detected() -> None:
    """Незнакомый, но структурно похожий токен тоже ловится."""
    text = "Незнакомый токен [[CUSTOM_TAG_2]] в тексте."
    assert has_placeholder_leak(text) is True
    assert "[[CUSTOM_TAG_2]]" in find_placeholder_leaks(text)


def test_multiple_distinct_leaks_sorted_unique() -> None:
    text = "[ADDRESS] и [PERSON_NAME] и снова [ADDRESS]"
    leaks = find_placeholder_leaks(text)
    assert leaks == ["[ADDRESS]", "[PERSON_NAME]"]


# ---------------------------------------------------------------------------
# harden_prompt_against_placeholders
# ---------------------------------------------------------------------------
def test_harden_prompt_adds_instruction_once() -> None:
    prompt = "Отредактируй текст."
    hardened = harden_prompt_against_placeholders(prompt)
    assert PLACEHOLDER_GUARD_INSTRUCTION.strip() in hardened
    # Идемпотентность: повторное применение не дублирует инструкцию.
    hardened_twice = harden_prompt_against_placeholders(hardened)
    assert hardened_twice.count("КРИТИЧЕСКОЕ ТРЕБОВАНИЕ К ВЫВОДУ") == 1
