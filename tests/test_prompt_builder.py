"""
test_prompt_builder.py

Тесты для PromptBuilder: проверяет сборку блоков и итоговый промпт.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.prompt_builder import (
    AudienceProfile,
    PromptBuilder,
    load_knowledge_base,
)
from tests.conftest import KB_PATH, PROJECT_ROOT


# ============================================================================
# load_knowledge_base
# ============================================================================


def test_load_knowledge_base_returns_all_fields() -> None:
    """load_knowledge_base загружает все 6 полей."""
    kb = load_knowledge_base(KB_PATH)
    assert isinstance(kb.stop_words, dict)
    assert isinstance(kb.grammar_errors, list)
    assert isinstance(kb.stylistic_issues, list)
    assert isinstance(kb.storytelling_frameworks, list)
    assert isinstance(kb.marketing_templates, list)
    assert isinstance(kb.domain_glossary, dict)


def test_knowledge_base_has_content() -> None:
    """В базе знаний есть хотя бы что-то."""
    kb = load_knowledge_base(KB_PATH)
    assert len(kb.stop_words) > 0
    assert len(kb.grammar_errors) > 0
    assert len(kb.stylistic_issues) > 0


# ============================================================================
# PromptBuilder.build — общие проверки
# ============================================================================


def test_build_returns_string(builder: PromptBuilder, sample_text: str) -> None:
    """build() возвращает непустую строку."""
    prompt = builder.build(text=sample_text, domain="marketing")
    assert isinstance(prompt, str)
    assert len(prompt) > 100


def test_build_contains_user_text(builder: PromptBuilder, sample_text: str) -> None:
    """Промпт содержит исходный текст пользователя."""
    prompt = builder.build(text=sample_text, domain="marketing")
    assert sample_text in prompt


def test_build_contains_stop_words(builder: PromptBuilder, sample_text: str) -> None:
    """Промпт содержит блок стоп-слов."""
    prompt = builder.build(text=sample_text, domain="marketing")
    assert "Стоп-слова" in prompt


def test_build_contains_grammar(builder: PromptBuilder, sample_text: str) -> None:
    """Промпт содержит блок грамматических ошибок."""
    prompt = builder.build(text=sample_text, domain="marketing")
    assert "грамматические" in prompt.lower()


def test_build_contains_stylistic(builder: PromptBuilder, sample_text: str) -> None:
    """Промпт содержит блок стилистики."""
    prompt = builder.build(text=sample_text, domain="marketing")
    assert "стилистические" in prompt.lower()


# ============================================================================
# Knowledge Base: условное подключение
# ============================================================================


def test_storytelling_included_only_for_storytelling_intent(
    builder: PromptBuilder,
    sample_text: str,
) -> None:
    """Фреймворки сторителлинга появляются только при intent=storytelling."""
    prompt_story = builder.build(
        text=sample_text,
        domain="marketing",
        intent="storytelling",
    )
    prompt_analytical = builder.build(
        text=sample_text,
        domain="marketing",
        intent="analytical",
    )
    assert "сторителлинга" in prompt_story.lower()
    assert "сторителлинга" not in prompt_analytical.lower()


def test_marketing_templates_included_for_marketing_domain(
    builder: PromptBuilder,
    sample_text: str,
) -> None:
    """Маркетинговые шаблоны появляются при domain=marketing."""
    prompt = builder.build(text=sample_text, domain="marketing")
    assert "Маркетинговые шаблоны" in prompt


def test_marketing_templates_not_included_for_blog_domain(
    builder: PromptBuilder,
    sample_text: str,
) -> None:
    """Маркетинговые шаблоны НЕ появляются при domain=blog."""
    prompt = builder.build(text=sample_text, domain="blog")
    assert "Маркетинговые шаблоны" not in prompt


# ============================================================================
# include_knowledge=False
# ============================================================================


def test_no_knowledge_when_flag_is_false(
    builder: PromptBuilder,
    sample_text: str,
) -> None:
    """include_knowledge=False убирает весь блок базы знаний."""
    prompt = builder.build(
        text=sample_text,
        domain="marketing",
        include_knowledge=False,
    )
    assert "Стоп-слова" not in prompt
    assert "грамматические" not in prompt.lower()


# ============================================================================
# Audience
# ============================================================================


def test_audience_block_present(
    builder: PromptBuilder,
    sample_text: str,
    sample_audience: AudienceProfile,
) -> None:
    """Профиль аудитории попадает в промпт."""
    prompt = builder.build(
        text=sample_text,
        domain="marketing",
        audience=sample_audience,
    )
    assert "b2b" in prompt
    assert "pro" in prompt


def test_no_audience_shows_neutral(
    builder: PromptBuilder,
    sample_text: str,
) -> None:
    """Без аудитории — подсказка про нейтральный тон."""
    prompt = builder.build(text=sample_text, domain="marketing", audience=None)
    assert "нейтральный" in prompt.lower()


# ============================================================================
# Реальный текст из test_texts
# ============================================================================


def test_build_with_real_marketing_text() -> None:
    """Интеграционный тест с реальным текстом из test_texts/marketing_short.txt."""
    builder = PromptBuilder()
    text_path: Path = PROJECT_ROOT / "test_texts" / "marketing_short.txt"

    if not text_path.exists():
        pytest.skip("marketing_short.txt не найден в test_texts")

    text = text_path.read_text(encoding="utf-8")

    prompt = builder.build(text=text, domain="marketing", intent="marketing_push")
    assert "Стоп-слова" in prompt
    assert "Маркетинговые шаблоны" in prompt
