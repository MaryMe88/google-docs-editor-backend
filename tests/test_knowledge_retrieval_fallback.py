from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Set

import pytest

from src.knowledge_retrieval import (
    FallbackPolicy,
    RULE_FALLBACK_POLICY,
    STRUCTURAL_FALLBACK_POLICY,
    _select_ranked_entries,
    normalize_text_for_match,
    score_rule_entry,
    score_structural_entry,
    select_grammar_rules,
    select_structural_by_tags_or_all,
)


def make_rule_entry(
    *,
    wrong: str = "",
    rule: str = "",
    description: str = "",
    tags: List[str] | None = None,
    name: str = "",
) -> Dict[str, Any]:
    return {
        "wrong": wrong,
        "rule": rule,
        "description": description,
        "name": name,
        "tags": tags or [],
    }


def make_structural_entry(
    *,
    name: str,
    description: str = "",
    when_to_use: List[str] | None = None,
    rule: str = "",
    tags: List[str] | None = None,
    steps: List[Dict[str, Any]] | None = None,
    sections: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "when_to_use": when_to_use or [],
        "rule": rule,
        "tags": tags or [],
        "steps": steps or [],
        "sections": sections or [],
    }


@pytest.fixture
def rule_kb() -> SimpleNamespace:
    return SimpleNamespace(
        grammar_errors=[],
        stylistic_issues=[],
        logic_issues=[],
    )


def test_strong_match_beats_other_candidates() -> None:
    entries = [
        make_rule_entry(
            wrong="ихний",
            rule="Используйте литературную форму.",
            description="Просторечная форма.",
            tags=["style"],
        ),
        make_rule_entry(
            wrong="",
            rule="Избегайте канцелярита.",
            description="Общая рекомендация по стилю.",
            tags=["style"],
        ),
    ]

    result = _select_ranked_entries(
        entries=entries,
        normalized_text=normalize_text_for_match("В тексте встретилось слово ихний."),
        wanted_tags=["style"],
        limit=1,
        scorer=score_rule_entry,
        min_score=1,
        fallback_policy=RULE_FALLBACK_POLICY,
    )

    assert len(result) == 1
    assert result[0]["wrong"] == "ихний"


def test_text_only_fallback_beats_tag_only() -> None:
    entries = [
        make_rule_entry(
            wrong="",
            name="Краткие фразы",
            rule="Делайте фразы короче.",
            description="Подходит, когда текст перегружен.",
            tags=["editing"],
        ),
        make_rule_entry(
            wrong="",
            name="Маркетинговый совет",
            rule="Добавьте CTA.",
            description="Совет по маркетинговому тексту.",
            tags=["style"],
        ),
    ]

    strict_policy = FallbackPolicy(
        min_strong_score=500,
        allow_text_only=True,
        allow_tag_only=True,
        allow_neutral_fallback=False,
        primary_only_for_tag_fallback=True,
    )

    result = _select_ranked_entries(
        entries=entries,
        normalized_text=normalize_text_for_match("Нам нужны краткие фразы и проще подача."),
        wanted_tags=["style"],
        limit=1,
        scorer=score_rule_entry,
        min_score=500,
        fallback_policy=strict_policy,
    )

    assert len(result) == 1
    assert result[0]["name"] == "Краткие фразы"


def test_tag_only_uses_primary_tags_not_expanded_noise() -> None:
    entries = [
        make_structural_entry(
            name="Primary match",
            description="Нужный приём по основному тегу.",
            tags=["storytelling"],
        ),
        make_structural_entry(
            name="Expanded-only noise",
            description="Шумный expanded-кандидат.",
            tags=["narrative"],
        ),
    ]

    policy = FallbackPolicy(
        min_strong_score=9999,
        allow_text_only=False,
        allow_tag_only=True,
        allow_neutral_fallback=False,
        primary_only_for_tag_fallback=True,
    )

    result = _select_ranked_entries(
        entries=entries,
        normalized_text="",
        wanted_tags=["storytelling"],
        limit=1,
        scorer=score_structural_entry,
        expanded_tags={"narrative"},
        min_score=9999,
        fallback_policy=policy,
    )

    assert len(result) == 1
    assert result[0]["name"] == "Primary match"


def test_rule_selector_returns_empty_when_only_neutral_exists(
    rule_kb: SimpleNamespace,
) -> None:
    rule_kb.grammar_errors = [
        make_rule_entry(
            name="Нейтральное правило",
            rule="Общая редакторская рекомендация.",
            description="Подходит почти ко всему.",
            tags=["neutral", "editing", "clarity"],
        )
    ]

    result = select_grammar_rules(
        kb=rule_kb,
        text="Совсем другой текст без совпадений.",
        tags=["science"],
        limit=3,
        min_score=1,
    )

    assert result == []


def test_structural_selector_can_use_neutral_fallback() -> None:
    entries = [
        make_structural_entry(
            name="Нейтральный приём",
            description="Универсальный редакторский приём.",
            tags=["neutral", "editing", "clarity"],
            steps=[{"name": "Сократить", "description": "Убрать лишнее"}],
        )
    ]

    result = select_structural_by_tags_or_all(
        entries=entries,
        tags=["science"],
        limit=1,
        expanded_tags={"research"},
        min_score=9999,
    )

    assert len(result) == 1
    assert result[0]["name"] == "Нейтральный приём"


def test_empty_result_when_no_stage_passes() -> None:
    entries = [
        make_structural_entry(
            name="Слабый кандидат",
            description="Без нужных тегов и без нейтрального профиля.",
            tags=["misc"],
        )
    ]

    policy = FallbackPolicy(
        min_strong_score=9999,
        allow_text_only=False,
        allow_tag_only=True,
        allow_neutral_fallback=False,
        primary_only_for_tag_fallback=True,
    )

    result = _select_ranked_entries(
        entries=entries,
        normalized_text="",
        wanted_tags=["storytelling"],
        limit=1,
        scorer=score_structural_entry,
        expanded_tags={"narrative"},
        min_score=9999,
        fallback_policy=policy,
    )

    assert result == []