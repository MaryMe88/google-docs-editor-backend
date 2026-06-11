from __future__ import annotations

import pytest

from src.config_types import KnowledgeLevel
from src.prompt_builder import PromptBuilder


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


def test_include_knowledge_false_omits_knowledge_block(
    builder: PromptBuilder,
) -> None:
    result = builder.build(
        text="Текст без KB блока.",
        domain="blog",
        include_knowledge=False,
    )

    assert "База знаний:" not in result


def test_knowledge_level_none_disables_knowledge_content(
    builder: PromptBuilder,
) -> None:
    result = builder.build(
        text="Текст с knowledge_level none.",
        domain="blog",
        include_knowledge=True,
        knowledge_level=KnowledgeLevel.NONE,
    )

    assert "Исходный текст:" in result
    assert "База знаний:" not in result


def test_knowledge_level_core_or_standard_does_not_crash(
    builder: PromptBuilder,
) -> None:
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


def test_full_level_without_optional_configs_does_not_crash(
    builder: PromptBuilder,
) -> None:
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
        builder.build(
            text="Текст.",
            domain="science",
        )


def test_invalid_intent_raises_value_error(builder: PromptBuilder) -> None:
    with pytest.raises(ValueError, match="Unsupported intent"):
        builder.build(
            text="Текст.",
            domain="blog",
            intent="unknown_intent",
        )


def test_invalid_overlay_raises_value_error(builder: PromptBuilder) -> None:
    with pytest.raises(ValueError, match="Unsupported overlays"):
        builder.build(
            text="Текст.",
            domain="blog",
            overlays=["unknown_overlay"],
        )


def test_empty_text_raises_value_error(builder: PromptBuilder) -> None:
    with pytest.raises(ValueError, match="Text must not be empty"):
        builder.build(
            text="   ",
            domain="blog",
        )


@pytest.mark.parametrize(
    ("domain", "intent"),
    [
        ("blog", "neutral"),
    ],
)
def test_supported_domain_intent_combinations(
    builder: PromptBuilder,
    domain: str,
    intent: str,
) -> None:
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
    """
    При include_few_shot=False в промпте не должно быть заголовка 'Примеры редактирования'.
    """
    # Используем текст, который может триггерить примеры из KB (грамматическая ошибка)
    # Если в KB нет записей с парами, тест всё равно проходит (просто нет заголовка)
    result = builder.build(
        text="Он согласился согласно приказа начальника.",
        domain="blog",
        include_knowledge=True,
        include_few_shot=False,
    )
    assert "Примеры редактирования" not in result


def test_include_few_shot_true_does_not_crash(builder: PromptBuilder) -> None:
    """
    При include_few_shot=True вызов не должен падать, даже если в KB нет подходящих пар.
    """
    result = builder.build(
        text="Он согласился согласно приказа начальника.",
        domain="blog",
        include_knowledge=True,
        include_few_shot=True,
    )
    assert isinstance(result, str)
    # Заголовок может присутствовать или отсутствовать — не проверяем
    assert "Исходный текст:" in result


def test_include_few_shot_without_knowledge_does_nothing(builder: PromptBuilder) -> None:
    """
    Если include_knowledge=False, параметр include_few_shot игнорируется.
    """
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
    assert result == result2  # одинаковый промпт
