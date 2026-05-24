"""
tests/test_knowledge_retrieval.py
================================
Тесты внутренней логики knowledge_retrieval.py.

Важно:
- это НЕ контрактные тесты API;
- они проверяют поведение отбора знаний;
- файл добавляется отдельно и не меняет рабочий код.

Запуск:
    pytest tests/test_knowledge_retrieval.py -v
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

from src.knowledge_retrieval import (
    FallbackPolicy,
    RULE_FALLBACK_POLICY,
    STRUCTURAL_FALLBACK_POLICY,
    _collect_with_budget,
    _select_ranked_entries,
    score_rule_entry,
    score_structural_entry,
    select_grammar_rules,
    select_structural_by_tags_or_all,
)


def make_rule_entry(
    wrong: str = "",
    correct: str = "",
    rule: str = "",
    description: str = "",
    tags: List[str] | None = None,
    entry_id: str | None = None,
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "wrong": wrong,
        "correct": correct,
        "rule": rule,
        "description": description,
        "tags": tags or [],
    }
    if entry_id is not None:
        entry["id"] = entry_id
    return entry



def make_structural_entry(
    name: str = "",
    description: str = "",
    when_to_use: List[str] | None = None,
    tags: List[str] | None = None,
    entry_id: str | None = None,
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "name": name,
        "description": description,
        "when_to_use": when_to_use or [],
        "tags": tags or [],
    }
    if entry_id is not None:
        entry["id"] = entry_id
    return entry


class TestRuleFallbackOrder:
    """Проверяем порядок fallback для grammar/style/logic."""

    def test_strong_match_wins_over_weaker_candidates(self) -> None:
        kb = SimpleNamespace(
            grammar_errors=[
                make_rule_entry(
                    wrong="ихний",
                    correct="их",
                    rule="просторечие",
                    tags=["grammar"],
                    entry_id="strong",
                ),
                make_rule_entry(
                    wrong="",
                    correct="",
                    rule="общая грамматическая рекомендация",
                    description="полезно проверять согласование",
                    tags=["grammar"],
                    entry_id="weak_tag_only",
                ),
            ]
        )

        result = select_grammar_rules(
            kb=kb,
            text="В тексте встречается ихний вариант.",
            tags=["grammar"],
            limit=1,
        )

        assert len(result) == 1
        assert result[0]["id"] == "strong"

    def test_text_only_fallback_works_when_no_tag_match(self) -> None:
        entries = [
            make_rule_entry(
                wrong="канцелярит",
                correct="",
                rule="избегайте канцелярита",
                tags=["bureaucratic"],
                entry_id="text_only",
            ),
            make_rule_entry(
                wrong="",
                correct="",
                rule="нейтральное правило",
                tags=["other_tag"],
                entry_id="other",
            ),
        ]

        result = _select_ranked_entries(
            entries=entries,
            normalized_text="в тексте есть канцелярит",
            wanted_tags=["missing_tag"],
            limit=1,
            scorer=score_rule_entry,
            fallback_policy=RULE_FALLBACK_POLICY,
        )

        assert len(result) == 1
        assert result[0]["id"] == "text_only"

    def test_tag_only_fallback_works_when_no_text_match(self) -> None:
        entries = [
            make_rule_entry(
                wrong="",
                correct="",
                rule="проверьте согласование",
                tags=["grammar"],
                entry_id="tag_only",
            ),
            make_rule_entry(
                wrong="",
                correct="",
                rule="другое правило",
                tags=["other"],
                entry_id="other",
            ),
        ]

        result = _select_ranked_entries(
            entries=entries,
            normalized_text="совсем другой текст без совпадений",
            wanted_tags=["grammar"],
            limit=1,
            scorer=score_rule_entry,
            min_score=100,
            fallback_policy=RULE_FALLBACK_POLICY,
        )

        assert len(result) == 1
        assert result[0]["id"] == "tag_only"

    def test_rule_policy_does_not_use_neutral_fallback(self) -> None:
        entries = [
            make_rule_entry(
                wrong="",
                correct="",
                rule="нейтральная рекомендация",
                description="полезная общая подсказка",
                tags=["neutral", "editing"],
                entry_id="neutral_candidate",
            )
        ]

        result = _select_ranked_entries(
            entries=entries,
            normalized_text="текст без совпадений",
            wanted_tags=["missing_tag"],
            limit=1,
            scorer=score_rule_entry,
            min_score=100,
            fallback_policy=RULE_FALLBACK_POLICY,
        )

        assert result == []
        assert RULE_FALLBACK_POLICY.allow_neutral_fallback is False


class TestStructuralFallbackOrder:
    """Проверяем fallback для структурных записей."""

    def test_neutral_does_not_beat_tag_match_when_tag_match_exists(self) -> None:
        entries = [
            make_structural_entry(
                name="Композиция для блога",
                description="Подходит для блоговых текстов",
                when_to_use=["когда нужен понятный блоговый текст"],
                tags=["blog"],
                entry_id="tag_match",
            ),
            make_structural_entry(
                name="Нейтральный шаблон",
                description="Общий шаблон без привязки к домену",
                when_to_use=["когда нужен универсальный вариант"],
                tags=["neutral", "editing"],
                entry_id="neutral",
            ),
        ]

        result = select_structural_by_tags_or_all(
            entries=entries,
            tags=["blog"],
            limit=1,
            expanded_tags={"article"},
            min_score=100,
        )

        assert len(result) == 1
        assert result[0]["id"] == "tag_match"

    def test_tag_only_fallback_uses_primary_tags_only(self) -> None:
        entries = [
            make_structural_entry(
                name="Основной вариант",
                description="Совпадает по primary-тегу",
                when_to_use=["для blog"],
                tags=["blog"],
                entry_id="primary",
            ),
            make_structural_entry(
                name="Расширенный вариант",
                description="Совпадает только по expanded-тегу",
                when_to_use=["для article"],
                tags=["article"],
                entry_id="expanded",
            ),
        ]

        result = select_structural_by_tags_or_all(
            entries=entries,
            tags=["blog"],
            limit=2,
            expanded_tags={"article"},
            min_score=100,
        )

        assert len(result) == 1
        assert result[0]["id"] == "primary"

    def test_neutral_fallback_works_for_structural_entries(self) -> None:
        entries = [
            make_structural_entry(
                name="Нейтральный каркас",
                description="Универсальная структура",
                when_to_use=["когда нет точного совпадения"],
                tags=["neutral", "editing"],
                entry_id="neutral",
            )
        ]

        result = _select_ranked_entries(
            entries=entries,
            normalized_text="совсем другой текст",
            wanted_tags=["missing_tag"],
            limit=1,
            scorer=score_structural_entry,
            min_score=100,
            fallback_policy=STRUCTURAL_FALLBACK_POLICY,
        )

        assert len(result) == 1
        assert result[0]["id"] == "neutral"


class TestBudgetAndDeduplication:
    """Проверяем ограничение по размеру и удаление дублей."""

    def test_collect_with_budget_deduplicates_same_entries(self) -> None:
        first = make_rule_entry(
            wrong="ихний",
            correct="их",
            rule="просторечие",
            tags=["grammar"],
        )
        duplicate = make_rule_entry(
            wrong="ихний",
            correct="их",
            rule="просторечие",
            tags=["grammar"],
        )
        unique = make_rule_entry(
            wrong="ложить",
            correct="класть",
            rule="нелитературная форма",
            tags=["grammar"],
        )

        result = _collect_with_budget(
            ranked_entries=[first, duplicate, unique],
            limit=10,
            char_budget=None,
        )

        assert len(result) == 2
        assert result[0]["wrong"] == "ихний"
        assert result[1]["wrong"] == "ложить"

    def test_collect_with_budget_respects_char_budget_after_first_entry(self) -> None:
        first = make_rule_entry(
            wrong="ихний",
            correct="их",
            rule="просторечие",
            description="короткая запись",
            tags=["grammar"],
            entry_id="first",
        )
        second = make_rule_entry(
            wrong="это очень длинная и явно лишняя запись для теста бюджета",
            correct="",
            rule="длинное правило",
            description="эта запись должна не поместиться, если бюджет уже почти исчерпан",
            tags=["grammar"],
            entry_id="second",
        )

        result = _collect_with_budget(
            ranked_entries=[first, second],
            limit=10,
            char_budget=80,
        )

        assert len(result) == 1
        assert result[0]["id"] == "first"

    def test_collect_with_budget_currently_keeps_first_entry_even_if_it_exceeds_budget(self) -> None:
        oversized = make_rule_entry(
            wrong="слишком длинная запись для маленького бюджета",
            correct="",
            rule="очень длинное правило",
            description="эта запись больше самого бюджета, но текущая логика всё равно оставляет первую запись",
            tags=["grammar"],
            entry_id="oversized",
        )

        result = _collect_with_budget(
            ranked_entries=[oversized],
            limit=10,
            char_budget=10,
        )

        # Это намеренно фиксирует ТЕКУЩЕЕ поведение.
        # Если позже решим, что это баг, тест нужно будет поменять вместе с кодом.
        assert len(result) == 1
        assert result[0]["id"] == "oversized"


class TestFallbackPolicySmoke:
    """Короткие проверки настроек policy, чтобы их не сломали случайно."""

    def test_rule_policy_flags(self) -> None:
        assert isinstance(RULE_FALLBACK_POLICY, FallbackPolicy)
        assert RULE_FALLBACK_POLICY.allow_text_only is True
        assert RULE_FALLBACK_POLICY.allow_tag_only is True
        assert RULE_FALLBACK_POLICY.allow_neutral_fallback is False
        assert RULE_FALLBACK_POLICY.primary_only_for_tag_fallback is True

    def test_structural_policy_flags(self) -> None:
        assert isinstance(STRUCTURAL_FALLBACK_POLICY, FallbackPolicy)
        assert STRUCTURAL_FALLBACK_POLICY.allow_text_only is True
        assert STRUCTURAL_FALLBACK_POLICY.allow_tag_only is True
        assert STRUCTURAL_FALLBACK_POLICY.allow_neutral_fallback is True
        assert STRUCTURAL_FALLBACK_POLICY.primary_only_for_tag_fallback is True
