from __future__ import annotations

import pytest
from types import SimpleNamespace
from unittest.mock import patch

from src.config_types import KnowledgeLevel
from src.prompt_builder import PromptBuilder, _get_confidence_note
from src.knowledge_retrieval import FallbackStage, _collect_with_budget


# ============================================================================
# Старые тесты (без изменений)
# ============================================================================

def test_build_returns_string(builder: PromptBuilder) -> None:
    result = builder.build(
        text="Это тестовый текст для проверки PromptBuilder.",
        domain="blog",
    )
    assert isinstance(result, str)
    assert "Роль:" in result
    assert "Домен:" in result
    assert "Исходный текст:" in result
    assert "Это тестовый текст" in result


def test_build_prompt_alias_matches_build(builder: PromptBuilder) -> None:
    kwargs = {
        "text": "Проверяем alias build_prompt.",
        "domain": "blog",
        "intent": "neutral",
        "overlays": [],
        "include_knowledge": False,
    }
    direct = builder.build(**kwargs)
    alias = builder.build_prompt(**kwargs)
    assert alias == direct


def test_include_knowledge_false_omits_knowledge_block(builder: PromptBuilder) -> None:
    result = builder.build(
        text="Текст без KB блока.",
        domain="blog",
        include_knowledge=False,
    )
    assert "База знаний:" not in result


def test_knowledge_level_none_disables_knowledge_content(builder: PromptBuilder) -> None:
    result = builder.build(
        text="Текст с knowledge_level none.",
        domain="blog",
        include_knowledge=True,
        knowledge_level=KnowledgeLevel.NONE,
    )
    assert "Исходный текст:" in result
    assert "База знаний:" not in result


def test_knowledge_level_core_or_standard_does_not_crash(builder: PromptBuilder) -> None:
    core_result = builder.build(
        text="Проверка уровня core.",
        domain="blog",
        include_knowledge=True,
        knowledge_level=KnowledgeLevel.CORE,
    )
    standard_result = builder.build(
        text="Проверка уровня standard.",
        domain="blog",
        include_knowledge=True,
        knowledge_level=KnowledgeLevel.STANDARD,
    )
    assert isinstance(core_result, str)
    assert isinstance(standard_result, str)
    assert "Исходный текст:" in core_result
    assert "Исходный текст:" in standard_result


def test_full_level_without_optional_configs_does_not_crash(builder: PromptBuilder) -> None:
    result = builder.build(
        text="Нужно сделать текст логичнее, чище и убедительнее.",
        domain="marketing",
        include_knowledge=True,
        knowledge_level=KnowledgeLevel.FULL,
        token_budget=1200,
    )
    assert isinstance(result, str)
    assert "Роль:" in result
    assert "Домен:" in result
    assert "Исходный текст:" in result


def test_invalid_domain_raises_value_error(builder: PromptBuilder) -> None:
    with pytest.raises(ValueError, match="Unsupported domain"):
        builder.build(text="Текст.", domain="science")


def test_invalid_intent_raises_value_error(builder: PromptBuilder) -> None:
    with pytest.raises(ValueError, match="Unsupported intent"):
        builder.build(text="Текст.", domain="blog", intent="unknown_intent")


def test_invalid_overlay_raises_value_error(builder: PromptBuilder) -> None:
    with pytest.raises(ValueError, match="Unsupported overlays"):
        builder.build(text="Текст.", domain="blog", overlays=["unknown_overlay"])


def test_empty_text_raises_value_error(builder: PromptBuilder) -> None:
    with pytest.raises(ValueError, match="Text must not be empty"):
        builder.build(text="   ", domain="blog")


@pytest.mark.parametrize(("domain", "intent"), [("blog", "neutral")])
def test_supported_domain_intent_combinations(builder: PromptBuilder, domain: str, intent: str) -> None:
    result = builder.build(
        text="Проверка допустимой комбинации домена и intent.",
        domain=domain,
        intent=intent,
        include_knowledge=False,
    )
    assert isinstance(result, str)
    assert "Домен:" in result


# ---------------------------------------------------------------------------
# Few-shot тесты (PR‑2)
# ---------------------------------------------------------------------------

def test_include_few_shot_false_omits_examples(builder: PromptBuilder) -> None:
    result = builder.build(
        text="Он согласился согласно приказа начальника.",
        domain="blog",
        include_knowledge=True,
        include_few_shot=False,
    )
    assert "Примеры редактирования" not in result


def test_include_few_shot_true_does_not_crash(builder: PromptBuilder) -> None:
    result = builder.build(
        text="Он согласился согласно приказа начальника.",
        domain="blog",
        include_knowledge=True,
        include_few_shot=True,
    )
    assert isinstance(result, str)
    assert "Исходный текст:" in result


def test_include_few_shot_without_knowledge_does_nothing(builder: PromptBuilder) -> None:
    result = builder.build(
        text="Текст без знаний.",
        domain="blog",
        include_knowledge=False,
        include_few_shot=True,
    )
    assert "База знаний:" not in result
    assert "Примеры редактирования" not in result

    result2 = builder.build(
        text="Текст без знаний.",
        domain="blog",
        include_knowledge=False,
        include_few_shot=False,
    )
    assert result == result2


# ---------------------------------------------------------------------------
# Новые тесты для ТП-2 (квалификатор уверенности)
# ---------------------------------------------------------------------------

def test_get_confidence_note_returns_correct_strings() -> None:
    """Проверяем, что функция возвращает правильные строки для разных stage."""
    assert _get_confidence_note(FallbackStage.STRONG) == ""
    assert _get_confidence_note(FallbackStage.EMPTY) == ""

    text_only = _get_confidence_note(FallbackStage.TEXT_ONLY)
    assert "смысловому совпадению" in text_only

    tag_only = _get_confidence_note(FallbackStage.TAG_ONLY)
    assert "теме раздела" in tag_only

    neutral = _get_confidence_note(FallbackStage.NEUTRAL)
    assert "теме раздела" in neutral


# ЗАМЕНЁННЫЙ ТЕСТ №1
def test_confidence_note_inserted_for_tag_only_stage(builder: PromptBuilder) -> None:
    """
    При stage=TAG_ONLY в промпте должен появиться квалификатор "теме раздела".
    Мокаем select_grammar_rules напрямую — тест не должен зависеть
    от конфигурации тегов домена.
    """
    fake_entry = {
        "wrong": "несовпадающий текст",
        "rule": "правило грамматики",
        "tags": ["grammar"],
    }

    with patch("src.prompt_builder.select_grammar_rules",
               return_value=([fake_entry], FallbackStage.TAG_ONLY, 0)):
        result = builder.build(
            text="Тестовый текст без совпадений.",
            domain="blog",
            include_knowledge=True,
            include_few_shot=False,
            knowledge_level=KnowledgeLevel.FULL,
            token_budget=None,
        )

    assert "Грамматические ориентиры:" in result
    assert "теме раздела" in result


def test_confidence_note_not_inserted_for_strong_stage(builder: PromptBuilder) -> None:
    """
    При stage=STRONG квалификатор не должен появляться.
    Используем запись с точным совпадением по wrong (без correct, чтобы это было правило).
    """
    kb = SimpleNamespace(
        grammar_errors=[
            {"wrong": "Тестовый текст", "rule": "правило грамматики", "tags": ["grammar"]}
        ],
        stylistic_issues=[],
        logic_issues=[],
        composition_principles=[],
        composition_errors=[],
        local_cohesion=[],
        storytelling_frameworks=[],
        marketing_templates=[],
        rhetoric_frameworks=[],
        editorial_techniques=[],
        stop_words={},
        domain_glossary={},
        nkrj_structure_patterns={},
    )

    with patch.object(builder, '_ensure_knowledge_base', return_value=kb):
        result = builder.build(
            text="Тестовый текст",
            domain="blog",
            include_knowledge=True,
            include_few_shot=False,
            knowledge_level=KnowledgeLevel.FULL,
            token_budget=None,
        )
        # Блок грамматики должен быть
        assert "Грамматические ориентиры:" in result
        # Квалификатор не должен появиться
        assert "теме раздела" not in result
        assert "смысловому совпадению" not in result


def test_confidence_note_not_inserted_when_no_knowledge(builder: PromptBuilder) -> None:
    """
    Если блоки знаний пусты, квалификатор не добавляется.
    """
    kb = SimpleNamespace(
        grammar_errors=[],
        stylistic_issues=[],
        logic_issues=[],
        composition_principles=[],
        composition_errors=[],
        local_cohesion=[],
        storytelling_frameworks=[],
        marketing_templates=[],
        rhetoric_frameworks=[],
        editorial_techniques=[],
        stop_words={},
        domain_glossary={},
        nkrj_structure_patterns={},
    )

    with patch.object(builder, '_ensure_knowledge_base', return_value=kb):
        result = builder.build(
            text="Тестовый текст.",
            domain="blog",
            include_knowledge=True,
            include_few_shot=False,
            knowledge_level=KnowledgeLevel.FULL,
            token_budget=None,
        )
        # В промпте не должно быть квалификаторов
        assert "теме раздела" not in result
        assert "смысловому совпадению" not in result
        # Блоков знаний быть не должно
        assert "Грамматические ориентиры:" not in result


# ЗАМЕНЁННЫЙ ТЕСТ №2
def test_confidence_note_appears_only_once_per_block(builder: PromptBuilder) -> None:
    """
    Квалификатор добавляется ровно один раз для блока,
    даже если в блоке несколько записей.
    """
    fake_entries = [
        {"wrong": "ошибка1", "rule": "правило1", "tags": ["grammar"]},
        {"wrong": "ошибка2", "rule": "правило2", "tags": ["grammar"]},
    ]

    with patch("src.prompt_builder.select_grammar_rules",
               return_value=(fake_entries, FallbackStage.TAG_ONLY, 0)):
        result = builder.build(
            text="Тестовый текст без совпадений.",
            domain="blog",
            include_knowledge=True,
            include_few_shot=False,
            knowledge_level=KnowledgeLevel.FULL,
            token_budget=None,
        )

    count = result.count("теме раздела")
    assert count == 1, f"Квалификатор должен встречаться 1 раз, найдено {count}"
    assert "ошибка1" in result
    assert "ошибка2" in result


# ЗАМЕНЁННЫЙ ТЕСТ №3
def test_confidence_note_position_before_rules(builder: PromptBuilder) -> None:
    """
    Проверяем, что квалификатор идёт перед заголовком
    "Грамматические ориентиры:".
    """
    fake_entry = {
        "wrong": "несовпадающий текст",
        "rule": "правило грамматики",
        "tags": ["grammar"],
    }

    with patch("src.prompt_builder.select_grammar_rules",
               return_value=([fake_entry], FallbackStage.TAG_ONLY, 0)):
        result = builder.build(
            text="Тестовый текст без совпадений.",
            domain="blog",
            include_knowledge=True,
            include_few_shot=False,
            knowledge_level=KnowledgeLevel.FULL,
            token_budget=None,
        )

    note_pos = result.find("теме раздела")
    header_pos = result.find("Грамматические ориентиры:")
    assert note_pos != -1, "Квалификатор не найден в промпте"
    assert header_pos != -1, "Заголовок блока не найден в промпте"
    assert note_pos < header_pos, "Квалификатор должен стоять перед заголовком блока"


# ============================================================================
# Новые тесты ТП-1 (исправление _collect_with_budget)
# ============================================================================

def test_collect_with_budget_applies_to_first_entry() -> None:
    """
    Проверяет, что char_budget применяется даже к первой записи.
    Раньше из-за условия `and result` первая запись всегда проходила,
    теперь исправлено.
    """
    # Создаём запись с большим размером (>100 символов)
    huge_entry = {
        "wrong": "x" * 5000,
        "correct": "y" * 5000,
        "rule": "z" * 1000,
    }
    entries = [huge_entry]
    result, dropped = _collect_with_budget(entries, limit=1, char_budget=100)
    assert result == [], "Первая запись не должна быть включена из-за превышения бюджета"
    assert dropped == 1, "Одна запись должна быть отброшена"