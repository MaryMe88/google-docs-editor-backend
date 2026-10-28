"""Тесты для kb_rendering: guard недоделанных блоков (bugfix evaluation_techniques)."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.prompt_builder.kb_rendering import (
    KBBlockConfig,
    ProcessContext,
    _process_kb_block,
)


def test_process_kb_block_skips_unconfigured_block() -> None:
    """Блок с retrieval_fn=None/append_fn=None безопасно пропускается.

    Bugfix для evaluation_techniques: до правки вызов None(...) давал
    TypeError, если блок включался в budget.
    """
    cfg = KBBlockConfig(
        name="test_stub",
        budget_key="test_stub",
        retrieval_fn=None,
        append_fn=None,
        title="",
        kb_attr=None,
        uses_structural_call=False,
    )
    ctx = ProcessContext(
        lines=[],
        meta={},
        kb=MagicMock(),
        text="test",
        primary_tags={"grammar"},
        expanded_tags=set(),
        budget=MagicMock(),
        domain="blog",
        intent=None,
        overlays=[],
        include_few_shot=False,
        total_few_shot_used=3,
        limits=MagicMock(),
    )
    result = _process_kb_block(cfg, ctx)
    assert result == 3
    assert ctx.lines == []


def test_process_kb_block_skips_when_only_append_fn_is_none() -> None:
    """Guard срабатывает, даже если только append_fn=None (асимметричный случай)."""
    cfg = KBBlockConfig(
        name="test_stub2",
        budget_key="test_stub2",
        retrieval_fn=MagicMock(),
        append_fn=None,
        title="",
        kb_attr=None,
        uses_structural_call=False,
    )
    ctx = ProcessContext(
        lines=[],
        meta={},
        kb=MagicMock(),
        text="test",
        primary_tags={"grammar"},
        expanded_tags=set(),
        budget=MagicMock(),
        domain="blog",
        intent=None,
        overlays=[],
        include_few_shot=False,
        total_few_shot_used=5,
        limits=MagicMock(),
    )
    result = _process_kb_block(cfg, ctx)
    assert result == 5
    assert ctx.lines == []
