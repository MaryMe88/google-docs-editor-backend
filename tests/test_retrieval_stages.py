# tests/test_retrieval_stages.py
"""
Unit-тесты для стадий _select_ranked_entries.
"""

from __future__ import annotations

import pytest

from src.knowledge_retrieval import (
    SelectionParams,
    RULE_FALLBACK_POLICY,
    STRUCTURAL_FALLBACK_POLICY,
    FallbackPolicy,
    _try_strong_stage,
    _try_text_only_stage,
    _try_tag_only_stage,
    _try_neutral_stage,
    score_rule_entry,
    normalize_text_for_match,
)


def make_rule_entry(wrong="", rule="", tags=None, **kwargs):
    return {"wrong": wrong, "rule": rule, "tags": tags or [], **kwargs}


def test_try_strong_stage_returns_strong():
    entries = [make_rule_entry(wrong="ошибка", rule="правило", tags=["grammar"])]
    params = SelectionParams(scorer=score_rule_entry)
    wanted_set = {"grammar"}
    result = _try_strong_stage(
        candidates=entries,
        normalized_text=normalize_text_for_match("ошибка"),
        wanted_set=wanted_set,
        params=params,
        limit=10,
        policy=RULE_FALLBACK_POLICY,
    )
    assert result is not None
    selected, dropped = result
    assert len(selected) == 1
    assert dropped == 0


def test_try_strong_stage_returns_none_if_no_score():
    entries = [make_rule_entry(wrong="", rule="правило", tags=["other"])]
    params = SelectionParams(scorer=score_rule_entry)
    wanted_set = {"grammar"}
    result = _try_strong_stage(
        candidates=entries,
        normalized_text=normalize_text_for_match("ошибка"),
        wanted_set=wanted_set,
        params=params,
        limit=10,
        policy=RULE_FALLBACK_POLICY,
    )
    assert result is None


def test_try_text_only_stage_returns_text_only():
    entries = [make_rule_entry(wrong="канцелярит", rule="избегайте", tags=["other"])]
    params = SelectionParams(scorer=score_rule_entry)
    wanted_set = {"missing"}  # не используется, но нужно для единой сигнатуры
    norm_text = normalize_text_for_match("канцелярит")
    result = _try_text_only_stage(
        candidates=entries,
        normalized_text=norm_text,
        wanted_set=wanted_set,  # добавлен
        params=params,
        limit=10,
        policy=RULE_FALLBACK_POLICY,
    )
    assert result is not None
    selected, dropped = result
    assert len(selected) == 1
    assert selected[0]["wrong"] == "канцелярит"


def test_try_text_only_stage_returns_none_if_disabled():
    policy = FallbackPolicy(
        min_strong_score=1,
        allow_text_only=False,
        allow_tag_only=True,
        allow_neutral_fallback=False,
    )
    entries = [make_rule_entry(wrong="канцелярит")]
    params = SelectionParams(scorer=score_rule_entry)
    wanted_set = set()
    norm_text = normalize_text_for_match("канцелярит")
    result = _try_text_only_stage(
        candidates=entries,
        normalized_text=norm_text,
        wanted_set=wanted_set,  # добавлен
        params=params,
        limit=10,
        policy=policy,
    )
    assert result is None


def test_try_tag_only_stage_returns_tag_only():
    entries = [make_rule_entry(wrong="", rule="правило", tags=["grammar"])]
    params = SelectionParams(scorer=score_rule_entry)
    wanted_set = {"grammar"}
    result = _try_tag_only_stage(
        candidates=entries,
        normalized_text="",  # добавлен, не используется
        wanted_set=wanted_set,
        params=params,
        limit=10,
        policy=RULE_FALLBACK_POLICY,
    )
    assert result is not None
    selected, dropped = result
    assert len(selected) == 1


def test_try_tag_only_stage_returns_none_if_no_overlap():
    entries = [make_rule_entry(wrong="", rule="правило", tags=["other"])]
    params = SelectionParams(scorer=score_rule_entry)
    wanted_set = {"grammar"}
    result = _try_tag_only_stage(
        candidates=entries,
        normalized_text="",  # добавлен
        wanted_set=wanted_set,
        params=params,
        limit=10,
        policy=RULE_FALLBACK_POLICY,
    )
    assert result is None


def test_try_neutral_stage_returns_neutral():
    entries = [make_rule_entry(
        wrong="",
        rule="нейтральное правило",
        tags=["neutral", "editing"],
        description="общее описание"
    )]
    params = SelectionParams(scorer=score_rule_entry)
    wanted_set = {"missing"}  # не используется
    result = _try_neutral_stage(
        candidates=entries,
        normalized_text="",  # добавлен
        wanted_set=wanted_set,  # добавлен
        params=params,
        limit=10,
        policy=STRUCTURAL_FALLBACK_POLICY,
    )
    assert result is not None
    selected, dropped = result
    assert len(selected) == 1
    assert selected[0]["rule"] == "нейтральное правило"


def test_try_neutral_stage_returns_none_if_disabled():
    policy = RULE_FALLBACK_POLICY  # allow_neutral_fallback=False
    entries = [make_rule_entry(tags=["neutral", "editing"])]
    params = SelectionParams(scorer=score_rule_entry)
    wanted_set = set()
    result = _try_neutral_stage(
        candidates=entries,
        normalized_text="",  # добавлен
        wanted_set=wanted_set,  # добавлен
        params=params,
        limit=10,
        policy=policy,
    )
    assert result is None